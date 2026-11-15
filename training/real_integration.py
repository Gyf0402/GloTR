"""Legal-value math shared by the purpose-limited q/o integration runner."""

from __future__ import annotations

import hashlib
import json
import math
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor
from torch.nn import functional as F

from losses.joint_objective import JointOptimizationLoss
from models.film_rgnn import SemanticDirectedGNN
from physics.corridor_physics import create_physics_calculator
from representations.basis import GlobalProfileBasis, MeanFunction
from training.real_scope import RealDataScope, read_pinned_json, sha256_file, write_json_new
from training.stages import FrozenTeacher, StagedRepresentation
from training.telemetry import EvidenceWriter, gradient_census


def mask_inputs(values: Tensor, visible: Tensor, outage: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    if (values.shape != visible.shape or visible.dtype != torch.bool
            or outage.shape != values.shape[:2] or outage.dtype != torch.bool):
        raise ValueError("matching boolean profile masks and day/station outage required")
    artificial = outage[..., None, None].expand_as(visible)
    m_in = visible & ~artificial
    score = visible & artificial
    if not torch.isfinite(values[m_in]).all():
        raise ValueError("legal inputs must be finite")
    return torch.where(m_in, values, torch.zeros_like(values)), m_in, score


def legal_huber(prediction: Tensor, truth: Tensor, support: Tensor) -> tuple[Tensor, int]:
    if prediction.shape != truth.shape or truth.shape != support.shape or support.dtype != torch.bool:
        raise ValueError("legal loss requires matching shapes and a boolean support mask")
    count = int(support.sum().item())
    if not torch.isfinite(prediction[support]).all() or not torch.isfinite(truth[support]).all():
        raise ValueError("legal supervision must be finite")
    if count == 0:
        return prediction.flatten()[:0].sum(), 0
    return F.smooth_l1_loss(prediction[support], truth[support]), count


def channel_scale(
    values: NDArray[Any], visible: NDArray[np.bool_], *,
    floors: tuple[float, float], source_id: str,
) -> dict[str, Any]:
    """Integration-only channel MAD, before any top-level component balance.

    No Gaussian multiplier; even medians use their midpoint. Scope selection
    excludes U and inner assessment before calling this function. This is not a
    formal PeMS MAD artifact and cannot be loaded through synthetic MADArtifact.
    """
    if values.shape != visible.shape or values.shape[-1] != 2 or visible.dtype != np.bool_:
        raise ValueError("scale requires matching q/o values and boolean support")
    if not source_id or any(not np.isfinite(x) or x <= 0 for x in floors):
        raise ValueError("positive registered floors and a source identity are required")
    medians, mads, scales, counts, digests = [], [], [], [], []
    for channel, floor in enumerate(floors):
        selected = np.asarray(values[..., channel][visible[..., channel]], dtype=np.float64)
        if not selected.size:
            raise ValueError("every channel needs legal training support")
        if not np.isfinite(selected).all():
            raise ValueError("scale observations must be finite")
        median = float(np.median(selected))
        mad = float(np.median(np.abs(selected - median)))
        medians.append(median)
        mads.append(mad)
        scales.append(max(mad, floor))
        counts.append(int(selected.size))
        digests.append(hashlib.sha256(selected.tobytes()).hexdigest())
    return {"version": 1, "purpose": "integration_smoke_throwaway", "pilot_only": True,
            "source_id": source_id, "layer": "channel", "units": ["veh/h", "fraction"],
            "channels": ["q", "o"], "definition": "median(abs(x-median(x))); midpoint median",
            "medians": medians, "mads": mads, "scales": scales, "floors": list(floors),
            "counts": counts, "sample_sha256": digests, "formal_MAD": False,
            "component_MAD_fitted": False}


def prior_history(values: Tensor, visible: Tensor) -> Tensor:
    """Cumulative legal history strictly before each day: q/o means plus presence.

    Call only after the outage and permanent-U input boundary. Day zero has no
    history. The daily current target never appears in its own conditioning.
    """
    safe = torch.where(visible, values, torch.zeros_like(values))
    totals = safe.sum(dim=2)
    counts = visible.sum(dim=2)
    zero = torch.zeros_like(totals[:1])
    past_sum = torch.cat((zero, totals[:-1].cumsum(dim=0)), dim=0)
    past_count = torch.cat((zero, counts[:-1].cumsum(dim=0)), dim=0)
    means = past_sum / past_count.clamp_min(1)
    return torch.cat((means, (past_count > 0).all(dim=-1, keepdim=True).to(values)), dim=-1)


def project_available(
    basis: GlobalProfileBasis, values: Tensor, visible: Tensor, mean: Tensor,
) -> tuple[Tensor, Tensor]:
    """No fabricated coefficient observation for empty/insufficient real profiles.

    Zero is an internal placeholder accompanied by observed=False, never a
    teacher label. A sufficiently populated but deficient design still raises
    through the core pre-ridge rank check.
    """
    if values.shape != visible.shape or visible.dtype != torch.bool:
        raise ValueError("matching boolean visible mask required")
    supported = visible.sum(dim=(-1, -2)) >= basis.rank
    coefficients = values.new_zeros((*values.shape[:-2], basis.rank))
    if supported.any():
        means = torch.broadcast_to(mean, values.shape)
        coefficients[supported] = basis.project(values[supported], visible[supported], means[supported])
    return coefficients, supported


def _calendar_features(dates: tuple[str, ...], device: torch.device) -> Tensor:
    return torch.tensor([[math.sin(2 * math.pi * date.fromisoformat(d).weekday() / 7),
                          math.cos(2 * math.pi * date.fromisoformat(d).weekday() / 7)]
                         for d in dates], dtype=torch.float32, device=device)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _memory(device: torch.device) -> dict[str, int]:
    return {"cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device)
            if device.type == "cuda" else 0,
            "cuda_reserved_bytes": torch.cuda.memory_reserved(device) if device.type == "cuda" else 0}


def _boundary_checks(
    rep: StagedRepresentation, values: Tensor, visible: Tensor, static: Tensor, calendar: Tensor,
    graph: dict[str, Any] | None = None,
) -> dict[str, bool]:
    """Frozen numerical checks; no student optimizer, graph construction or evaluation."""
    checks: dict[str, bool] = {}
    rep.assert_frozen()
    outage = torch.zeros(values.shape[:2], dtype=torch.bool, device=values.device)
    outage[:, 0] = True  # Whole target interval, including its source-reference gaps.
    clean, mask, score = mask_inputs(values, visible, outage)
    changed = values.clone()
    changed[~mask] = torch.inf
    other, other_mask, _ = mask_inputs(changed, visible, outage)
    checks["whole_target_and_natural_missing_input_invariance"] = torch.equal(clean, other)
    checks["target_mask_independent_of_reference_values"] = torch.equal(mask, other_mask)
    checks["score_excludes_natural_missing"] = bool(not score[~visible].any())
    history = prior_history(clean, mask)
    mean = rep.mean(static, calendar, history, protocol="K")
    # A constant coefficient prediction checks WLS fusion and hard copy without
    # claiming that an invented/empty topology is a real graph model.
    coefficients = values.new_zeros((*values.shape[:2], rep.basis.rank))
    prediction = rep.basis.decode_with_observations(coefficients, clean, mask, mean)
    second = rep.basis.decode_with_observations(coefficients, changed, mask, mean)
    checks["decode_hidden_value_invariance"] = torch.equal(prediction, second)
    checks["observed_hard_copy"] = torch.equal(prediction[mask], clean[mask])
    checks["empty_target_has_finite_prediction"] = bool(torch.isfinite(prediction[:, 0]).all())
    checks["all_qo_predictions_finite"] = bool(torch.isfinite(prediction).all())
    partial = mask.clone()
    partial[:, -1, :30] = False
    partial_changed = clean.clone()
    partial_changed[~partial] = torch.nan
    left = rep.basis.decode_with_observations(coefficients, clean, partial, mean)
    right = rep.basis.decode_with_observations(coefficients, partial_changed, partial, mean)
    checks["partial_donor_hidden_value_invariance"] = torch.equal(left, right)

    # Use identical representation state in clones for chunk/dtype audits; never
    # alter the frozen teacher coordinate binding on the fitted object.
    projected, supported = project_available(rep.basis, values, visible, mean)
    for chunk in (1, 7, values.shape[0] * values.shape[1]):
        clone = GlobalProfileBasis(288, 2, rep.basis.rank, wls_chunk_size=chunk).to(values.device)
        clone.load_state_dict(rep.basis.state_dict())
        result, _ = project_available(clone, values, visible, mean)
        checks[f"wls_chunk_{chunk}_agrees"] = bool(torch.allclose(result, projected, rtol=2e-4, atol=2e-4))
    audit = GlobalProfileBasis(288, 2, rep.basis.rank, wls_compute_dtype=torch.float64).to(values.device)
    audit.load_state_dict(rep.basis.state_dict())
    result, _ = project_available(audit, values, visible, mean)
    checks["float64_audit_keeps_float32_output"] = result.dtype == values.dtype == torch.float32
    checks["float64_audit_agrees"] = bool(torch.allclose(result, projected, rtol=3e-4, atol=3e-4))

    # Actual real support still cannot identify an intentionally collapsed design.
    deficient = GlobalProfileBasis(288, 2, 16).to(values.device)
    with torch.no_grad():
        deficient.raw_basis.copy_(torch.eye(576, 16, device=values.device))
    rank_mask = visible[:1, :1].clone()
    rank_mask[..., :8, :] = False
    try:
        deficient.project(values[:1, :1], rank_mask, torch.zeros_like(mean[:1, :1]))
    except ValueError as exc:
        checks["rank_deficient_teacher_rejected"] = "rank deficient" in str(exc)
    else:
        checks["rank_deficient_teacher_rejected"] = False

    # An untrained token unit check with all-empty relations is explicitly not the
    # real four-relation baseline, and there is no student fit or performance score.
    features = values.new_zeros((*values.shape[:2], 4))
    features[..., :2] = history[..., :2]
    student = SemanticDirectedGNN(4, 1, 2, 16, 16, num_layers=1, dropout=0).to(values.device).eval()
    edges = {name: torch.empty((2, 0), dtype=torch.long, device=values.device)
             for name in student.RELATIONS}
    token_target = outage[0]
    first = student(features, static, calendar, edges, protocol="U", target_mask=token_target)
    features[:, 0] = torch.nan
    second = student(features, static, calendar, edges, protocol="U", target_mask=token_target)
    checks["student_U_token_replacement"] = torch.equal(first, second)
    mu_first = rep.mean(static, calendar, history, protocol="U")
    mu_second = rep.mean(static, calendar, torch.full_like(history, torch.inf), protocol="U")
    checks["mu_U_history_replacement"] = torch.equal(mu_first, mu_second)
    checks["natural_empty_profiles_have_no_coefficient_observation"] = bool(
        torch.equal(projected[~supported], torch.zeros_like(projected[~supported])))
    if graph is not None:
        checks.update(_graph_checks(rep, clean, mask, mean, static, calendar, graph))
    # Eligibility short circuit remains differentiable even without speed data.
    flow = prediction[..., 0].detach().clone().requires_grad_(True)
    occupancy = prediction[..., 1].detach().clone().requires_grad_(True)
    physics = create_physics_calculator(audited=False, semantic_dim=1, state_dim=1,
                                        reason="Q1 audited corridors=0; q/o integration only")
    losses, diagnostics = physics(flow, occupancy, speed=torch.full_like(flow, torch.nan))
    zero = sum(losses.values())
    torch.autograd.backward(zero)
    checks["unaudited_physics_disabled_differentiable_zero"] = (
        diagnostics["enabled"] is False and bool(diagnostics["reason"])
        and diagnostics["speed_bounds_evaluated"] is False and float(zero.item()) == 0
        and flow.grad is not None and bool(torch.equal(flow.grad, torch.zeros_like(flow))))
    rep.assert_frozen()
    return checks


def _graph_checks(
    rep: StagedRepresentation, clean: Tensor, mask: Tensor, mean: Tensor,
    static: Tensor, calendar: Tensor, graph: dict[str, Any],
) -> dict[str, bool]:
    """Actual graph node/edge axis and two-layer message direction smoke; no fit."""
    if len(graph["station_order"]) != clean.shape[1]:
        raise ValueError("graph node axis differs from registered K tensor axis")
    relations = graph["edge_index"]
    if set(relations) != set(SemanticDirectedGNN.RELATIONS):
        raise ValueError("exactly four geographic relation keys required")
    edges = {key: torch.tensor(value, dtype=torch.long, device=clean.device)
             for key, value in relations.items()}
    if any(x.ndim != 2 or x.shape[0] != 2 for x in edges.values()):
        raise ValueError("edge_index must be [2,E], including [2,0] empty relations")
    n = clean.shape[1]
    if any(x.numel() and (int(x.min()) < 0 or int(x.max()) >= n) for x in edges.values()):
        raise ValueError("edge endpoint outside frozen station axis")
    if not sum(x.shape[1] for x in edges.values()):
        raise ValueError("an all-empty graph is not a geographic graph check")
    expected = {(x["relation"], x["source_index"], x["target_index"]) for x in graph["edges"]}
    actual = {(name, int(a), int(b)) for name, edge in edges.items() for a, b in edge.T.tolist()}
    if expected != actual:
        raise ValueError("edge provenance/indices differ")
    for entry in graph["edges"]:
        if (graph["station_order"][entry["source_index"]] != entry["source_station_id"]
                or graph["station_order"][entry["target_index"]] != entry["target_station_id"]
                or not entry["source_ids"]):
            raise ValueError("edge station identity/source evidence mismatch")
    # Check the actual RGCN source_to_target path, with no target-reference inputs.
    coefficients, available = project_available(rep.basis, clean, mask, mean)
    features = torch.cat((coefficients, available[..., None].to(clean),
                          mask.float().mean(dim=(-1, -2))[..., None]), dim=-1)
    student = SemanticDirectedGNN(18, 1, 2, 32, 16, num_layers=2, dropout=0,
                                  temporal_mode="offline_bidirectional").to(clean.device).eval()
    episode_ids = torch.zeros(clean.shape[0], dtype=torch.long, device=clean.device)
    ordinals = torch.arange(clean.shape[0], device=clean.device)

    def infer(x: Tensor) -> Tensor:
        result: Tensor = student(x, static, calendar, edges,
                                 episode_ids=episode_ids, day_ordinals=ordinals)
        return result

    with torch.no_grad():
        baseline = infer(features)
        changed = features.clone()
        changed[:, 2, 0] += 10  # Third travel-order node -> second -> endpoint zero.
        through_two = infer(changed)
        changed = features.clone()
        changed[:, 3, 0] += 10  # Three hops cannot reach endpoint in two spatial layers.
        through_three = infer(changed)
    return {"real_graph_provenance_and_ID_axis": True,
            "real_graph_two_message_layers": len(student.convolutions) == 2,
            "real_graph_two_hop_influence": not torch.equal(baseline[:, 0], through_two[:, 0]),
            "real_graph_no_three_hop_shortcut": torch.equal(baseline[:, 0], through_three[:, 0]),
            "real_graph_empty_ramp_relations_supported": all(
                edges[x].shape == (2, 0) and bool(graph["empty_relation_explanation"][x])
                for x in ("on_ramp", "off_ramp")),
            "real_graph_untrained_outputs_finite": bool(torch.isfinite(baseline).all())}


def run_integration(scope: RealDataScope, output: Path) -> dict[str, Any]:
    """One registered throwaway stage-0 fit, then read-only frozen checks.

    This entry point cannot assess pilot episodes, fit a student or unlock W3.
    An existing output directory is an error; failed evidence is never reused.
    """
    scope = RealDataScope.load(scope.manifest_path, scope.manifest_sha256)
    output.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    write_json_new(output / "registered_input.json", scope.config)
    fit = scope.config["fit"]
    device = torch.device(fit["device"])
    if device.type not in {"cpu", "cuda"} or (device.type == "cuda" and not torch.cuda.is_available()):
        raise ValueError("registered CPU/CUDA device unavailable")
    torch.set_num_threads(4)
    torch.manual_seed(int(fit["seed"]))
    np.random.seed(int(fit["seed"]))
    torch.use_deterministic_algorithms(True)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(fit["seed"]))
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats(device)
    writer = EvidenceWriter(output)
    report: dict[str, Any] = {"status": "running", "purpose": "engineering-integration-only",
        "pilot_only": True, "throwaway": True, "started_utc": started,
        "run_manifest_sha256": scope.manifest_sha256,
        "formal_W3_allowed": False, "formal_experiment_allowed": False,
        "assessment_read": False, "C_checkpoint_reuse_allowed": False,
        "student_fitted": False, "graph_claim": False, "physics_enabled": False,
        "fit_dates": list(scope.dates), "registered_station_ids": list(scope.stations),
        "source_clock_view": "fixed_UTC_minus_8_candidate", "source_clock_verified": False}
    try:
        views = []
        load_started = time.perf_counter()
        for day in scope.dates:
            print(json.dumps({"event": "loading_registered_shard", "date": day}), flush=True)
            views.append(scope.read_day(day))
        report["load_seconds"] = time.perf_counter() - load_started
        report["input_days"] = {day: view.diagnostics for day, view in zip(scope.dates, views)}
        keep = [i for i, station in enumerate(scope.stations) if station not in scope.excluded]
        report["fit_station_ids"] = [scope.stations[i] for i in keep]
        raw_all = np.stack([x.values for x in views])
        masks_all = np.stack([x.visible for x in views])
        raw, legal = raw_all[:, keep], masks_all[:, keep]
        scale = channel_scale(raw, legal, floors=tuple(fit["channel_floors"]),
                              source_id=scope.manifest_sha256)
        # Exact numeric input/statistic invariance; provenance SHA is intentionally
        # not falsified when testing an in-memory adversary.
        poisoned = raw_all.copy()
        poisoned[~masks_all] = np.inf
        invariant_scale = channel_scale(poisoned[:, keep], legal,
            floors=tuple(fit["channel_floors"]), source_id=scope.manifest_sha256) == scale
        scale.update(frozen_at=datetime.now(timezone.utc).isoformat(), fit_dates=list(scope.dates),
                     station_ids=report["fit_station_ids"], excluded_u=list(scope.excluded))
        write_json_new(output / "channel_scale.json", scale)
        scales = torch.tensor(scale["scales"], dtype=torch.float32, device=device)
        values = torch.tensor(raw.copy(), dtype=torch.float32, device=device) / scales
        visible = torch.tensor(legal.copy(), dtype=torch.bool, device=device)
        day_index = torch.arange(values.shape[0], device=device)[:, None, None]
        station_index = torch.arange(values.shape[1], device=device)[None, :, None]
        slot_index = torch.arange(288, device=device)[None, None, :]
        artificial = ((slot_index + 3 * day_index + 2 * station_index)
                      % fit["mask_period"] == 0)[..., None]
        m_fit, supervision = visible & ~artificial, visible & artificial
        if not m_fit.any() or not supervision.any():
            raise ValueError("no legal observed/held-out pretraining support")
        clean = torch.where(m_fit, values, torch.zeros_like(values))
        static = values.new_ones((*values.shape[:2], 1))
        calendar = _calendar_features(scope.dates, device)
        history = prior_history(clean, m_fit)
        rep = StagedRepresentation(GlobalProfileBasis(288, 2, 16),
                                   MeanFunction(1, 2, 3, 288, 2, fit["hidden_dim"])).to(device)
        # B is K-only fitting. Freeze the U-only token explicitly, rather than
        # calling its absent gradient an unexpected training path failure.
        rep.mean_function.no_history_token.requires_grad_(False)
        rep.fit_reference(values, m_fit, split="train")
        rep.reference_provenance.update(purpose="integration_smoke_throwaway",
            run_manifest_sha256=scope.manifest_sha256, fit_dates=list(scope.dates),
            station_ids=report["fit_station_ids"], scale_sha256=sha256_file(output / "channel_scale.json"))
        optimizer = torch.optim.Adam(rep.optimizer_groups(None, fit["lr"]))
        report["expected_inactive"] = {
            "basis.global_mean": "frozen legal train reference",
            "mean_function.no_history_token": "U-only path; B performs K-only fitting",
            "physics": "entire module disabled before construction; no slack parameters",
            "student": "not fitted in B"}
        report["conditioning"] = "neutral constant static placeholder; weekday sin/cos; prior legal history"
        report["condition_limit"] = "B interface health only; not the semantic graph model or W3"
        report["loss_definition"] = "Huber on channel-scaled artificial legal holes + 0.1 observed Huber"
        report["normalization"] = "channel MAD only; no component MAD or weight-balance conclusion"
        start_fit = time.perf_counter()
        for epoch in range(1, fit["epochs"] + 1):
            epoch_start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            mean = rep.mean(static, calendar, history, protocol="K")
            coeff, coefficient_observed = project_available(rep.basis, values, m_fit, mean)
            reconstruction = rep.basis.decode(coeff, mean)
            imp, imp_count = legal_huber(reconstruction, values, supervision)
            obs, obs_count = legal_huber(reconstruction, values, m_fit)
            loss = imp + .1 * obs
            if not torch.isfinite(reconstruction).all() or not torch.isfinite(loss):
                raise RuntimeError("nonfinite legal reconstruction or loss")
            torch.autograd.backward(loss)
            gradients = gradient_census({"representation": rep})
            if any(not x["finite"] for x in gradients["parameters"].values()):
                raise RuntimeError("nonfinite gradient")
            optimizer.step()
            _synchronize(device)
            raw_loss = {name: 0.0 for name in JointOptimizationLoss.LOSS_NAMES}
            raw_loss.update(imp=float(imp.item()), obs=float(obs.item()))
            counts = {name: 0 for name in raw_loss}
            counts.update(imp=imp_count, obs=obs_count)
            record = {"purpose": "engineering-integration-only", "pilot_only": True,
                      "stage": "B_throwaway_stage0", "epoch": epoch, "total": float(loss.item()),
                      "raw": raw_loss, "support_counts": counts, "physics_weight": 0.0,
                      "available_projection_profiles": int(coefficient_observed.sum().item()),
                      "unavailable_projection_profiles": int((~coefficient_observed).sum().item()),
                      "inactive_reason": "stage0 fits imp/obs only; no teacher/student/physics/R/pinball",
                      "gradients": gradients, "lr": [fit["lr"]],
                      "epoch_seconds": time.perf_counter() - epoch_start, **_memory(device)}
            alerts = writer.write(record)
            print(json.dumps({"epoch": epoch, "total": record["total"], "alerts": len(alerts)}), flush=True)
            if any(x["action"] == "stop" for x in alerts):
                raise RuntimeError("deterministic rule stopped the integration fit")
        report["fit_seconds"] = time.perf_counter() - start_fit
        report["completed_epochs"] = fit["epochs"]
        fingerprint = rep.freeze()
        rep.save_checkpoint(output / "throwaway_representation.pt")
        with torch.no_grad():
            mean = rep.mean(static, calendar, history, protocol="K")
            full_coefficients, teacher_observed = project_available(rep.basis, values, visible, mean)
            teacher = FrozenTeacher(rep, values[teacher_observed], visible[teacher_observed],
                                    mean[teacher_observed])
            report["teacher_drift"] = teacher.assert_unchanged(rep, mean[teacher_observed])
            report["teacher_observed_profiles"] = int(teacher_observed.sum().item())
            report["teacher_unavailable_profiles"] = int((~teacher_observed).sum().item())
            report["coordinate_fingerprint"] = fingerprint
            write_json_new(output / "coordinate_binding.json", {
                **rep.bind_coordinates("teacher", scale_fingerprint=sha256_file(output / "channel_scale.json")),
                "throwaway": True, "pilot_only": True, "sign_order": "deterministic QR; saved column order",
                "mean_source": "frozen legal reference + K offset; inner-fit only"})
        graph = read_pinned_json(scope.config["actual_graph"]) if scope.config.get("actual_graph") else None
        checks = _boundary_checks(rep, values, visible, static, calendar, graph)
        checks["scale_U_and_hidden_values_invariant"] = invariant_scale
        checks["source_U_values_invariant"] = all(x.diagnostics["source_u_poison_invariant"] for x in views)
        checks["frozen_checkpoint_roundtrip"] = (
            StagedRepresentation.load_checkpoint(output / "throwaway_representation.pt", device)
            .coordinate_fingerprint() == fingerprint)
        rep.assert_frozen()
        report["assertions"] = checks
        if not all(checks.values()):
            raise RuntimeError(f"integration assertions failed: {[k for k, v in checks.items() if not v]}")
        # An interface-only training fit diagnostic, never a W3 threshold test.
        with torch.no_grad():
            decoded = rep.basis.decode(full_coefficients, mean)
            report["interface_health_only"] = {
                "full_reference_projection": "representation capacity diagnostic; not imputation evidence",
                "native_unit_train_MAE": [float(((decoded[..., c] - values[..., c])
                    [visible[..., c]].abs().mean() * scales[c]).item()) for c in range(2)],
                "orthogonal_basis_singular_values": torch.linalg.svdvals(rep.basis.orthonormal_basis()).tolist()}
        # Recheck source bytes without loading values a second time.
        report["source_hashes_unchanged"] = all(
            sha256_file(Path(scope.release["output_root"]) / day / "arrays.npz")
            == scope.shards[day]["arrays_sha256"] for day in scope.dates)
        if not report["source_hashes_unchanged"]:
            raise RuntimeError("source changed during integration")
        report.update(status="passed", **_memory(device))
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        writer.write_alert({"level": "blocking", "action": "stop", "rule": "integration_failure",
                            "evidence": {"exception": str(exc), "run_manifest_sha256": scope.manifest_sha256}})
        raise
    finally:
        report["ended_utc"] = datetime.now(timezone.utc).isoformat()
        writer.close()
        write_json_new(output / "result.json", report)
    return report
