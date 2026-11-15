"""Legal-visible profile encoding and an anchored coefficient matrix-completion null.

This is a project-specific masked ALS null, not a reimplementation of SoftImpute.
Each coefficient is completed over a station-by-day matrix. There is no teacher
coefficient input or sensor-day-as-independent-row expansion.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import date
from typing import Sequence

import numpy as np
import torch
from torch import Tensor

from representations.basis import GlobalProfileBasis


@dataclass(frozen=True)
class VisibleCoefficients:
    values: np.ndarray  # [S, D, R]; unavailable entries are NaN, never zero observations.
    observed: np.ndarray
    profile_status: np.ndarray  # [S, D]
    station_ids: tuple[str, ...]
    dates: tuple[str, ...]
    provenance: dict


@dataclass(frozen=True)
class LowRankCompletion:
    values: np.ndarray
    metadata: dict


def encode_visible_profiles(
    profiles: Tensor,
    visible: Tensor,
    *,
    basis: GlobalProfileBasis,
    legal_profiles: Tensor,
    station_ids: Sequence[str],
    dates: Sequence[str],
    basis_source_id: str,
    mean_source_id: str,
    coordinate_id: str,
    mean: Tensor | None = None,
) -> VisibleCoefficients:
    """Encode only legal, currently visible profiles using an already frozen basis.

    ``legal_profiles`` is an externally audited information-set boundary. A visible
    value outside it is an error. Hidden profile values are never encoded, even if
    the caller supplies finite hidden ground truth. Inadequately observed profiles
    get no coefficient observation and an explicit reason.
    """
    if profiles.ndim != 4 or profiles.shape[-2:] != (basis.time_steps, basis.channels):
        raise ValueError("profiles must have [station, day, basis.time_steps, basis.channels] shape")
    if visible.shape != profiles.shape or visible.dtype != torch.bool:
        raise ValueError("visible must be a matching boolean mask")
    if legal_profiles.shape != profiles.shape[:2] or legal_profiles.dtype != torch.bool:
        raise ValueError("legal_profiles must be a boolean [station, day] mask")
    if (visible & ~legal_profiles[..., None, None]).any():
        raise ValueError("visible entries outside legal_profiles are forbidden")
    if any(parameter.requires_grad for parameter in basis.parameters()):
        raise ValueError("coefficient encoding requires a frozen basis and reference mean")
    reference = basis.global_mean if mean is None else mean
    if reference.requires_grad or not torch.isfinite(reference).all():
        raise ValueError("the reference mean must be frozen and finite")
    stations, model_dates = tuple(station_ids), tuple(dates)
    if len(stations) != profiles.shape[0] or len(model_dates) != profiles.shape[1]:
        raise ValueError("station and day identities must match profile axes")
    if (not all((*stations, basis_source_id, mean_source_id, coordinate_id))
            or len(set(stations)) != len(stations)):
        raise ValueError("unique station IDs and frozen source/coordinate identities are required")
    parsed_dates = tuple(date.fromisoformat(day) for day in model_dates)
    if tuple(sorted(set(parsed_dates))) != parsed_dates:
        raise ValueError("model dates must be sorted and unique")
    if not torch.isfinite(profiles[visible]).all():
        raise ValueError("visible profile values must be finite")
    observed_counts = visible.flatten(2).sum(dim=-1)
    statuses = np.full(profiles.shape[:2], "no_visible_profile", dtype="U40")
    statuses[(observed_counts > 0).cpu().numpy()] = "insufficient_visible_entries"
    candidates = (observed_counts >= basis.rank) & legal_profiles
    with torch.no_grad():
        matrix = basis.orthonormal_basis()
        # Match the WLS numerical-rank gate, before adding its solving ridge.
        # A row-matrix rank check has a different tolerance from the Gram check.
        compute_matrix = matrix.to(basis.wls_compute_dtype or profiles.dtype)
        products = (compute_matrix[:, :, None] * compute_matrix[:, None, :]).flatten(1)
        rank_rtol = basis.rank * torch.finfo(profiles.dtype).eps
        valid = torch.zeros_like(candidates)
        for station, day in torch.nonzero(candidates, as_tuple=False).tolist():
            weights = visible[station, day].reshape(-1).to(compute_matrix.dtype)
            gram = (weights @ products).reshape(basis.rank, basis.rank)
            if int(torch.linalg.matrix_rank(gram, rtol=rank_rtol).item()) == basis.rank:
                valid[station, day] = True
                statuses[station, day] = "encoded_visible_profile"
            else:
                statuses[station, day] = "rank_deficient_visible_basis"
        coefficients = np.full((*profiles.shape[:2], basis.rank), np.nan, dtype=np.float64)
        if valid.any():
            sanitized = torch.where(visible, profiles, torch.full_like(profiles, torch.nan))
            means = torch.broadcast_to(reference, profiles.shape)
            # Existing WLS retains its observed-rank gate. No hidden target teacher is formed.
            projected = basis.project(sanitized[valid], visible[valid], mean=means[valid])
            coefficients[valid.cpu().numpy()] = projected.detach().cpu().numpy()
        observed = np.broadcast_to(valid.cpu().numpy()[..., None], coefficients.shape).copy()
        basis_digest = hashlib.sha256(matrix.detach().cpu().numpy().tobytes()).hexdigest()
        mean_digest = hashlib.sha256(reference.detach().cpu().numpy().tobytes()).hexdigest()
    provenance = {
        "input_source": "legal_visible_profiles_only", "basis_frozen": True,
        "basis_source_id": basis_source_id, "mean_source_id": mean_source_id,
        "coordinate_id": coordinate_id, "basis_sha256": basis_digest,
        "mean_sha256": mean_digest,
        "visible_mask_sha256": hashlib.sha256(visible.cpu().numpy().tobytes()).hexdigest(),
        "visible_entry_count": int(visible.sum().item()),
        "encoded_profile_count": int(valid.sum().item()),
        "visible_rank_gate": "WLS Gram rank before ridge; rtol=rank*eps(profile dtype)",
        "teacher_input_used": False,
    }
    for array in (coefficients, observed, statuses):
        array.setflags(write=False)
    return VisibleCoefficients(coefficients, observed, statuses, stations, model_dates, provenance)


def _anchored_core(mask: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray]:
    """Necessary degree checks, iterated after removing insufficiently anchored axes."""
    rows, columns = np.ones(mask.shape[0], dtype=bool), np.ones(mask.shape[1], dtype=bool)
    while True:
        next_rows = rows & (mask[:, columns].sum(axis=1) >= rank)
        next_columns = columns & (mask[next_rows].sum(axis=0) >= rank)
        if np.array_equal(rows, next_rows) and np.array_equal(columns, next_columns):
            return rows, columns
        rows, columns = next_rows, next_columns


def _component_support(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Keep only row/column pairs connected by legal observations in the core.

    Factors in disconnected bipartite components have independent coordinate
    choices, so their cross-component products cannot be inferred from the data.
    Connectivity remains a necessary check, not a uniqueness certificate.
    """
    row_labels = np.full(mask.shape[0], -1, dtype=int)
    column_labels = np.full(mask.shape[1], -1, dtype=int)
    component_count = 0
    for start in range(mask.shape[0]):
        if row_labels[start] >= 0:
            continue
        pending = [start]
        row_labels[start] = component_count
        while pending:
            row = pending.pop()
            columns = np.flatnonzero(mask[row] & (column_labels < 0))
            column_labels[columns] = component_count
            for column in columns:
                rows = np.flatnonzero(mask[:, column] & (row_labels < 0))
                row_labels[rows] = component_count
                pending.extend(rows.tolist())
        component_count += 1
    return row_labels[:, None] == column_labels[None, :], component_count


def _objective(u: np.ndarray, v: np.ndarray, values: np.ndarray,
               mask: np.ndarray, ridge: float) -> float:
    residual = (u @ v.T)[mask] - values[mask]
    return float(residual @ residual + ridge * (np.square(u).sum() + np.square(v).sum()))


def masked_low_rank_completion(
    encoded: VisibleCoefficients,
    *,
    rank: int = 1,
    iterations: int = 100,
    ridge: float = 1e-8,
    seed: int = 0,
    on_unidentifiable: str = "raise",
) -> LowRankCompletion:
    """Fit masked alternating ridge least squares separately for each coefficient r.

    The objective is sum_r(sum_observed((A_r/scale_r - U_r V_r^T)^2)
    + ridge*(||U_r||_F^2 + ||V_r||_F^2)). Scales use visible entries only.
    Degree and connected-support checks are necessary, not sufficient, for unique
    matrix completion. Mark mode keeps known unidentifiable missing entries NaN.
    """
    if not isinstance(encoded, VisibleCoefficients):
        raise ValueError("input must be a legal-visible coefficient encoding")
    values, observed = encoded.values, encoded.observed
    if values.ndim != 3 or observed.shape != values.shape or observed.dtype != np.bool_:
        raise ValueError("coefficient input and boolean observation mask must have [S, D, R] shape")
    if (encoded.provenance.get("input_source") != "legal_visible_profiles_only"
            or encoded.provenance.get("teacher_input_used") is not False):
        raise ValueError("hidden teacher coefficients are not legal input")
    if not np.isfinite(values[observed]).all() or not np.isnan(values[~observed]).all():
        raise ValueError("visible coefficients must be finite; hidden coefficients must remain NaN")
    if not isinstance(rank, int) or isinstance(rank, bool) or not 0 < rank <= min(values.shape[:2]):
        raise ValueError("rank must be a positive integer within station/day dimensions")
    if not isinstance(iterations, int) or iterations <= 0 or not np.isfinite(ridge) or ridge <= 0:
        raise ValueError("iterations and finite ridge must be positive")
    if on_unidentifiable not in ("raise", "mark"):
        raise ValueError("on_unidentifiable must be 'raise' or 'mark'")
    cores = [_anchored_core(observed[..., coefficient], rank) for coefficient in range(values.shape[2])]
    bad_rows = sorted({index for rows, _ in cores for index in np.flatnonzero(~rows).tolist()})
    bad_columns = sorted({index for _, columns in cores for index in np.flatnonzero(~columns).tolist()})
    component_supports, component_counts = [], []
    cross_component_count = 0
    for coefficient, (rows, columns) in enumerate(cores):
        local_mask = observed[..., coefficient][np.ix_(rows, columns)]
        connected, count = _component_support(local_mask)
        component_supports.append(connected)
        component_counts.append(count)
        cross_component_count += int((~connected).sum())
    if (bad_rows or bad_columns or cross_component_count) and on_unidentifiable == "raise":
        raise ValueError(f"unidentifiable axes lack rank={rank} legal anchors: "
                         f"station_indices={bad_rows}, day_indices={bad_columns}; "
                         f"disconnected-support missing coefficients={cross_component_count}")
    completed = values.copy()
    generator = np.random.default_rng(seed)
    history = np.zeros(iterations + 1, dtype=np.float64)
    scales, active_parameters = [], 0
    for coefficient, (rows, columns) in enumerate(cores):
        if not rows.any() or not columns.any():
            scales.append(None)
            continue
        selection = np.ix_(np.flatnonzero(rows), np.flatnonzero(columns))
        local_mask = observed[..., coefficient][selection]
        local_values = values[..., coefficient][selection]
        scale = float(np.sqrt(np.mean(np.square(local_values[local_mask]))))
        scale = scale if scale > 0 else 1.0
        if not np.isfinite(scale):
            raise ValueError("visible coefficient scale must be finite")
        scales.append(scale)
        normalized = local_values / scale  # Hidden cells remain NaN and are never in an objective.
        u = generator.normal(0.0, 0.1, size=(int(rows.sum()), rank))
        v = generator.normal(1.0, 0.1, size=(int(columns.sum()), rank))
        active_parameters += u.size + v.size
        identity = np.eye(rank) * ridge
        history[0] += _objective(u, v, normalized, local_mask, ridge)
        for iteration in range(iterations):
            for row in range(u.shape[0]):
                chosen = local_mask[row]
                design = v[chosen]
                u[row] = np.linalg.solve(design.T @ design + identity,
                                         design.T @ normalized[row, chosen])
            for column in range(v.shape[0]):
                chosen = local_mask[:, column]
                design = u[chosen]
                v[column] = np.linalg.solve(design.T @ design + identity,
                                            design.T @ normalized[chosen, column])
            history[iteration + 1] += _objective(u, v, normalized, local_mask, ridge)
        reconstruction = (u @ v.T) * scale
        if not np.isfinite(reconstruction).all():
            raise ValueError("matrix completion produced nonfinite values")
        reconstruction[~component_supports[coefficient]] = np.nan
        completed[..., coefficient][selection] = np.where(local_mask, local_values, reconstruction)
    completed.setflags(write=False)
    metadata = {
        "method": "coefficient_masked_als_null", "matrix_layout": "per-coefficient station-by-day",
        "input_provenance": encoded.provenance,
        "config": {"rank": rank, "iterations": iterations, "ridge": ridge, "seed": seed,
                   "optimizer": "alternating ridge normal-equation solves", "dtype": "float64",
                   "on_unidentifiable": on_unidentifiable, "early_stopping": False},
        "factor_parameters": int(active_parameters),
        "nominal_factor_parameters": int(values.shape[2] * rank * sum(values.shape[:2])),
        "frozen_representation_parameters_excluded": True,
        "observed_coefficient_count": int(observed.sum()), "visible_rms_scales": scales,
        "objective": "normalized visible SSE plus ridge factor penalties",
        "objective_history": history.tolist() if active_parameters else [],
        "fit_performed": bool(active_parameters), "observed_coefficients_hard_copied": True,
        "unidentifiable_station_ids": [encoded.station_ids[index] for index in bad_rows],
        "unidentifiable_dates": [encoded.dates[index] for index in bad_columns],
        "support_component_counts": component_counts,
        "cross_component_missing_coefficient_count": cross_component_count,
        "unidentifiable_missing_coefficient_count": int(np.isnan(completed).sum()),
        "status": ("unidentifiable" if not active_parameters else
                   "partially_unidentifiable" if bad_rows or bad_columns or cross_component_count else
                   "necessary_anchor_checks_passed"),
        "unique_identifiability_proven": False,
    }
    return LowRankCompletion(completed, metadata)
