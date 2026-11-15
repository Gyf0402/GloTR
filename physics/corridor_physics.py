"""Sparse station-to-cell mapping and differentiable corridor physics losses."""

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, Optional, cast

import torch
import torch.nn.functional as functional
from torch import Tensor, nn
from torch.nn.modules.module import _IncompatibleKeys

from models.temporal import consecutive_day_mask
from physics.artifacts import (
    EffectiveLengthArtifact,
    ReliabilityArtifact,
    validate_residual_scales,
)
from training.scaling import MADArtifact


def _as_sparse(matrix: Tensor) -> Tensor:
    """Convert a dense two-dimensional tensor to coalesced COO format."""
    indices = matrix.nonzero(as_tuple=False).T  # [2, nnz]
    values = matrix[indices[0], indices[1]]  # [nnz]
    invariant_context = cast(Callable[[], AbstractContextManager[object]],
                             torch.sparse.check_sparse_tensor_invariants)
    with invariant_context():
        return torch.sparse_coo_tensor(
            indices,
            values,
            matrix.shape,
            is_coalesced=True,
        )  # [rows, cols]


class TopologyMapper(nn.Module):
    """Frozen, unit-consistent maps for a topology-audited directed corridor.

    ``pi_k`` and ``pi_q`` are nonnegative allocation matrices. Conservation
    signs live only in ``incidence``: -1 means flow enters a cell and +1 means
    flow leaves it, so B_G q has units of net outflow (vehicles/hour).
    """

    sensor_order: Tensor
    cell_lengths_km: Tensor
    pi_k: Tensor
    pi_q: Tensor
    incidence: Tensor

    def __init__(
        self,
        sensor_postmiles: Sequence[float],
        cell_lengths_km: Sequence[float],
        audited: bool,
        density_overlap_lengths_km: Optional[Tensor] = None,
        flow_boundary_sensor_indices: Optional[Sequence[int]] = None,
        edge_connections: Optional[Sequence[tuple[Optional[int], Optional[int]]]] = None,
    ) -> None:
        super().__init__()
        if not audited:
            raise ValueError("local physics is disabled until the corridor passes topology audit")

        positions = torch.as_tensor(sensor_postmiles, dtype=torch.float32)  # [N]
        lengths = torch.as_tensor(cell_lengths_km, dtype=torch.float32)  # [M]
        if positions.ndim != 1 or positions.numel() < 2:
            raise ValueError("sensor_postmiles must contain at least two sensors")
        if not torch.isfinite(positions).all():
            raise ValueError("sensor postmiles must be finite")
        if (lengths.ndim != 1 or lengths.numel() < 1 or not torch.isfinite(lengths).all()
                or torch.any(lengths <= 0)):
            raise ValueError("cell_lengths_km must be a finite positive one-dimensional sequence")
        if torch.unique(positions).numel() != positions.numel():
            raise ValueError("sensor postmiles must be unique")

        order = torch.argsort(positions)  # [N], caller supplies the directed coordinate
        ordered_positions = positions[order]  # [N]
        if torch.any(ordered_positions[1:] <= ordered_positions[:-1]):
            raise ValueError("directed postmile coordinates must define a strict corridor order")

        node_count = positions.numel()
        cell_count = lengths.numel()
        if density_overlap_lengths_km is None:
            if cell_count != node_count - 1:
                raise ValueError(
                    "default midpoint cells require M=N-1; otherwise provide density overlap lengths"
                )
            dense_pi_k = torch.zeros(cell_count, node_count)  # [M, N]
            for cell_index in range(cell_count):
                left_sensor = int(order[cell_index].item())
                right_sensor = int(order[cell_index + 1].item())
                half_length = lengths[cell_index] / 2.0
                dense_pi_k[cell_index, left_sensor] = half_length
                dense_pi_k[cell_index, right_sensor] = half_length
        else:
            dense_pi_k = torch.as_tensor(
                density_overlap_lengths_km, dtype=torch.float32
            ).clone()  # [M, N]
            if dense_pi_k.shape != (cell_count, node_count):
                raise ValueError("density_overlap_lengths_km must have shape [M, N]")
            if not torch.isfinite(dense_pi_k).all() or torch.any(dense_pi_k < 0):
                raise ValueError("density overlap lengths must be finite and nonnegative")
        row_lengths = dense_pi_k.sum(dim=1, keepdim=True)  # [M, 1]
        if torch.any(row_lengths <= 0):
            raise ValueError("each density-map row must have positive support")
        dense_pi_k = dense_pi_k / row_lengths  # [M, N]

        if edge_connections is None:
            connections: list[tuple[Optional[int], Optional[int]]] = [(None, 0)]
            connections.extend((cell - 1, cell) for cell in range(1, cell_count))
            connections.append((cell_count - 1, None))
        else:
            connections = list(edge_connections)
        edge_count = len(connections)
        if edge_count < 1:
            raise ValueError("at least one directed boundary edge is required")

        if flow_boundary_sensor_indices is None:
            if edge_connections is not None or edge_count != node_count:
                raise ValueError(
                    "explicit edges require one flow_boundary_sensor_indices entry per edge"
                )
            flow_indices = [int(index.item()) for index in order]
        else:
            flow_indices = list(flow_boundary_sensor_indices)
        if len(flow_indices) != edge_count:
            raise ValueError("flow boundary assignment count must equal edge count")

        dense_pi_q = torch.zeros(edge_count, node_count)  # [E, N]
        for edge_index, sensor_index in enumerate(flow_indices):
            if not 0 <= sensor_index < node_count:
                raise ValueError("flow boundary sensor index is out of range")
            dense_pi_q[edge_index, sensor_index] = 1.0

        incidence = torch.zeros(cell_count, edge_count)  # [M, E]
        for edge_index, (source_cell, destination_cell) in enumerate(connections):
            if source_cell is None and destination_cell is None:
                raise ValueError("an edge must enter or leave at least one cell")
            if source_cell is not None:
                if not 0 <= source_cell < cell_count:
                    raise ValueError("source cell index is out of range")
                incidence[source_cell, edge_index] += 1.0
            if destination_cell is not None:
                if not 0 <= destination_cell < cell_count:
                    raise ValueError("destination cell index is out of range")
                incidence[destination_cell, edge_index] -= 1.0

        self.audited = True
        self.node_count = node_count
        self.cell_count = cell_count
        self.edge_count = edge_count
        self.register_buffer("sensor_order", order)  # [N]
        self.register_buffer("cell_lengths_km", lengths)  # [M]
        self.register_buffer("pi_k", _as_sparse(dense_pi_k))  # sparse [M, N]
        self.register_buffer("pi_q", _as_sparse(dense_pi_q))  # sparse [E, N]
        self.register_buffer("incidence", incidence)  # [M, E]

    def map_density(self, node_density: Tensor) -> Tensor:
        """Map node density [..., N] in vehicles/km to cell density [..., M]."""
        return self._sparse_map(node_density, self.pi_k, self.node_count)  # [..., M]

    def map_flow(self, node_flow: Tensor) -> Tensor:
        """Map node flow [..., N] in vehicles/hour to boundary flow [..., E]."""
        return self._sparse_map(node_flow, self.pi_q, self.node_count)  # [..., E]

    @staticmethod
    def _sparse_map(values: Tensor, matrix: Tensor, input_size: int) -> Tensor:
        if values.shape[-1] != input_size:
            raise ValueError(f"mapped tensor must end with {input_size} nodes")
        leading_shape = values.shape[:-1]
        flat_values = values.reshape(-1, input_size)  # [P, N]
        mapped: Tensor = torch.sparse.mm(matrix, flat_values.T).T  # [P, M_or_E]
        return mapped.reshape(*leading_shape, matrix.shape[0])  # [..., M_or_E]


class PhysicsResidualCalculator(nn.Module):
    """Audited local physics with frozen scaling and bounded source/sink slack.

    Legacy tensors are [B,S,N]. Cross-day tensors are [B,D,T,N], with explicit
    [B,D] episode IDs and day ordinals. Occupancy is a fraction, lengths km,
    time h, speed km/h, density veh/km and flow veh/h. Flow/density are lane
    totals; occupancy_per_lane means lane-average occupancy is multiplied by
    the lane count. No observed values are clipped to make constraints pass.
    """

    station_ids: Tensor
    slack_rate_caps: Tensor
    slack_cell_mask: Tensor
    variable_scales: Tensor
    parameter_lower: Tensor
    parameter_upper: Tensor
    residual_scale_values: Tensor
    calibrated_effective_length_km: Tensor
    fixed_semantic_prior: Tensor
    physical_contract: Tensor
    artifact_fingerprints: Tensor

    def __init__(
        self, mapper: TopologyMapper, semantic_dim: int, state_dim: int,
        dt_hours: float = 5.0 / 60.0,
        effective_length_bounds_km: tuple[float, float] = (0.004, 0.012),
        free_speed_bounds_kmh: tuple[float, float] = (40.0, 130.0),
        wave_speed_bounds_kmh: tuple[float, float] = (5.0, 35.0),
        jam_density_bounds_per_lane: tuple[float, float] = (80.0, 220.0),
        capacity_bounds_per_lane: tuple[float, float] = (1200.0, 2800.0),
        smooth_min_temperature: float = 20.0, max_slack_rate: float = 300.0,
        slack_l1_weight: float = 1e-3, slack_tv_weight: float = 1e-3,
        slack_relative_weight: float = 2.0,
        slack_rate_caps: Optional[Tensor | Sequence[float]] = None,
        slack_cell_mask: Optional[Tensor | Sequence[bool]] = None,
        occupancy_per_lane: bool = True, *,
        residual_scales: MADArtifact | str | Path | None = None,
        reliability_artifact: ReliabilityArtifact | str | Path | None = None,
        effective_length_artifact: EffectiveLengthArtifact | str | Path | None = None,
        station_ids: Sequence[int] | None = None, num_regimes: int = 1,
        theta_max_offset_fraction: float = .1,
        semantic_prior_parameters: Tensor | None = None,
        flow_scale_veh_h: float = 1000.0, density_scale_veh_km: float = 100.0,
        speed_scale_kmh: float = 100.0, bound_temperature: float = .05,
        capacity_equality_weight: float = 1.0,
    ) -> None:
        super().__init__()
        if not isinstance(mapper, TopologyMapper) or not mapper.audited:
            raise ValueError("enabled physics requires a topology-audited mapper")
        positive = (dt_hours, smooth_min_temperature, max_slack_rate, flow_scale_veh_h,
                    density_scale_veh_km, speed_scale_kmh, bound_temperature)
        penalties = (slack_l1_weight, slack_tv_weight, slack_relative_weight,
                     capacity_equality_weight)
        if (semantic_dim <= 0 or state_dim <= 0 or num_regimes <= 0
                or any(not math.isfinite(v) or v <= 0 for v in positive)
                or any(not math.isfinite(v) or v < 0 for v in penalties)):
            raise ValueError("dimensions, units and temperatures must be finite positive; weights nonnegative")
        bounds = (free_speed_bounds_kmh, wave_speed_bounds_kmh,
                  jam_density_bounds_per_lane, capacity_bounds_per_lane)
        for lower, upper in (*bounds, effective_length_bounds_km):
            if not (math.isfinite(lower) and math.isfinite(upper) and 0 < lower < upper):
                raise ValueError("physical parameter bounds must be finite and satisfy 0 < lower < upper")
        if not math.isfinite(theta_max_offset_fraction) or not 0 <= theta_max_offset_fraction <= 1:
            raise ValueError("theta offset fraction must lie in [0,1]")
        self.mapper, self.semantic_dim, self.state_dim = mapper, semantic_dim, state_dim
        self.dt_hours, self.num_regimes = dt_hours, num_regimes
        self.effective_length_bounds_km = effective_length_bounds_km
        self.free_speed_bounds_kmh, self.wave_speed_bounds_kmh = bounds[:2]
        self.jam_density_bounds_per_lane, self.capacity_bounds_per_lane = bounds[2:]
        self.smooth_min_temperature, self.max_slack_rate = smooth_min_temperature, max_slack_rate
        self.slack_l1_weight, self.slack_tv_weight = slack_l1_weight, slack_tv_weight
        self.slack_relative_weight, self.occupancy_per_lane = slack_relative_weight, occupancy_per_lane
        self.theta_max_offset_fraction = theta_max_offset_fraction
        self.bound_temperature, self.capacity_equality_weight = bound_temperature, capacity_equality_weight

        rate_caps = (torch.full((mapper.cell_count,), float(max_slack_rate))
                     if slack_rate_caps is None else
                     torch.as_tensor(slack_rate_caps, dtype=torch.float32).flatten().detach().clone())
        if (rate_caps.shape != (mapper.cell_count,) or not torch.isfinite(rate_caps).all()
                or torch.any(rate_caps <= 0)):
            raise ValueError("slack_rate_caps must be finite positive with one value per cell")
        cell_mask = (torch.zeros(mapper.cell_count, dtype=torch.bool) if slack_cell_mask is None
                     else torch.as_tensor(slack_cell_mask, dtype=torch.bool).flatten().clone())
        if cell_mask.shape != (mapper.cell_count,):
            raise ValueError("slack_cell_mask must have one boolean per cell")
        identifiers = tuple(range(mapper.node_count)) if station_ids is None else tuple(station_ids)
        if (len(identifiers) != mapper.node_count or len(set(identifiers)) != len(identifiers)
                or any(isinstance(i, bool) or not isinstance(i, int) for i in identifiers)):
            raise ValueError("station_ids must contain one unique integer per mapper node")
        self.register_buffer("station_ids", torch.tensor(identifiers, dtype=torch.long))
        self.register_buffer("slack_rate_caps", rate_caps)
        self.register_buffer("slack_cell_mask", cell_mask)
        self.register_buffer("variable_scales", torch.tensor(
            [flow_scale_veh_h, density_scale_veh_km, speed_scale_kmh], dtype=torch.float64))
        self.register_buffer("parameter_lower", torch.tensor([v[0] for v in bounds]))
        self.register_buffer("parameter_upper", torch.tensor([v[1] for v in bounds]))

        self.residual_scales = (MADArtifact.load(residual_scales)
                                if isinstance(residual_scales, (str, Path)) else residual_scales)
        if self.residual_scales is not None:
            validate_residual_scales(self.residual_scales)
        self.residual_scale_source = ("frozen_training_residual_mad" if self.residual_scales is not None
                                      else "synthetic_unit_fallback")
        self.register_buffer("residual_scale_values", torch.tensor(
            [self.residual_scales.entry(n).scale if self.residual_scales else 1.
             for n in ("fd", "cons")], dtype=torch.float64))
        self.reliability_artifact = (ReliabilityArtifact.load(reliability_artifact)
                                     if isinstance(reliability_artifact, (str, Path)) else reliability_artifact)
        if self.reliability_artifact is not None:
            if (self.residual_scales is None or self.reliability_artifact.residual_scale_fingerprint
                    != self.residual_scales.fingerprint):
                raise ValueError("reliability requires its exact frozen residual scale fingerprint")
        self.effective_length_artifact = (
            EffectiveLengthArtifact.load(effective_length_artifact)
            if isinstance(effective_length_artifact, (str, Path)) else effective_length_artifact)
        calibrated = torch.empty(0)
        if self.effective_length_artifact is not None:
            artifact = self.effective_length_artifact
            lookup = dict(zip(artifact.station_ids, artifact.values_km))
            if any(i not in lookup for i in identifiers):
                raise ValueError("independent ell artifact must cover every requested mapper station")
            calibrated = torch.tensor([lookup[i] for i in identifiers])
            self._validate_length(calibrated)
        self.register_buffer("calibrated_effective_length_km", calibrated.detach().clone())
        self.effective_length_source = ("independent_training_calibration"
                                        if self.effective_length_artifact is not None
                                        else "frozen_semantic_synthetic_fallback")

        # Keep the three legacy network names and initialization order.
        hidden_dim = max(16, 2 * semantic_dim)
        self.effective_length_network = nn.Sequential(
            nn.Linear(semantic_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))
        self.fd_parameter_network = nn.Sequential(
            nn.Linear(semantic_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 4))
        self.slack_state_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))
        self.effective_length_network.requires_grad_(False)
        self.semantic_prior_network = copy.deepcopy(self.fd_parameter_network).requires_grad_(False)
        self.fd_regime_baseline = nn.Embedding(num_regimes, 4)
        nn.init.zeros_(self.fd_regime_baseline.weight)
        self.fd_day_offset_network = nn.Linear(state_dim + num_regimes, 4)
        nn.init.zeros_(self.fd_day_offset_network.weight)
        nn.init.zeros_(self.fd_day_offset_network.bias)
        prior = torch.empty(0)
        if semantic_prior_parameters is not None:
            prior = semantic_prior_parameters.detach().clone()
            if (prior.shape != (mapper.node_count, 4) or not torch.isfinite(prior).all()
                    or torch.any(prior < self.parameter_lower.to(prior))
                    or torch.any(prior > self.parameter_upper.to(prior))):
                raise ValueError("semantic prior parameters require finite bounded [N,4] per-lane units")
        self.register_buffer("fixed_semantic_prior", prior)
        self.register_buffer("physical_contract", torch.tensor([
            1., dt_hours, smooth_min_temperature, max_slack_rate, slack_l1_weight,
            slack_tv_weight, slack_relative_weight, float(occupancy_per_lane),
            float(num_regimes), theta_max_offset_fraction, bound_temperature,
            capacity_equality_weight, *effective_length_bounds_km], dtype=torch.float64))
        self._bound_fingerprints = (
            self.residual_scales.fingerprint if self.residual_scales else None,
            self.reliability_artifact.fingerprint if self.reliability_artifact else None,
            self.effective_length_artifact.fingerprint if self.effective_length_artifact else None)
        digest_bytes = b"".join(bytes.fromhex(fp) if fp else bytes(32)
                                for fp in self._bound_fingerprints)
        self.register_buffer("artifact_fingerprints", torch.tensor(list(digest_bytes), dtype=torch.uint8))
        self._frozen_buffers = {name: value.detach().clone() for name, value in self.named_buffers()
                                if not name.startswith("mapper.")}
        self._frozen_networks = {
            name: {key: value.detach().clone() for key, value in network.state_dict().items()}
            for name, network in (("effective_length_network", self.effective_length_network),
                                  ("semantic_prior_network", self.semantic_prior_network))}
        self.artifact_metadata = {
            "version": "b5-physics/v1", "residual_scale_source": self.residual_scale_source,
            "residual_fingerprint": self.residual_scales.fingerprint if self.residual_scales else None,
            "reliability_fingerprint": (self.reliability_artifact.fingerprint
                                        if self.reliability_artifact else None),
            "effective_length_source": self.effective_length_source,
            "effective_length_fingerprint": (self.effective_length_artifact.fingerprint
                                             if self.effective_length_artifact else None),
            "residual_unit": "veh/h", "raw_conservation_unit": "veh/km",
            "reliability_layer": "local_reliability", "component_mad_applied_here": False,
            "semantic_prior_source": ("explicit_frozen_per_lane_parameters" if prior.numel()
                                      else "initial_frozen_semantic_snapshot"),
            "n13_empirically_closed": False,
        }

    @staticmethod
    def _bounded(raw: Tensor, bounds: tuple[float, float]) -> Tensor:
        lower, upper = bounds
        if not (math.isfinite(lower) and math.isfinite(upper) and 0 < lower < upper):
            raise ValueError("physical parameter bounds must be finite and satisfy 0 < lower < upper")
        return lower + (upper - lower) * torch.sigmoid(raw)

    def _validate_length(self, length: Tensor) -> None:
        lower, upper = self.effective_length_bounds_km
        if not torch.isfinite(length).all():
            raise ValueError("effective_length_km must be finite")
        if torch.any((length < lower) | (length > upper)):
            raise ValueError("effective_length_km violates its calibrated absolute bounds")

    def assert_frozen_artifacts(self) -> None:
        """Detect mutation or accidental re-enabling of length/prior gradients."""
        current_fingerprints = (
            self.residual_scales.fingerprint if self.residual_scales else None,
            self.reliability_artifact.fingerprint if self.reliability_artifact else None,
            self.effective_length_artifact.fingerprint if self.effective_length_artifact else None)
        if current_fingerprints != self._bound_fingerprints:
            raise RuntimeError("frozen physics artifact provenance changed")
        buffers = dict(self.named_buffers())
        for name, snapshot in self._frozen_buffers.items():
            current = buffers[name]
            if current.requires_grad or not torch.equal(current, snapshot.to(current)):
                raise RuntimeError(f"frozen physics artifact changed: {name}")
        for name, network_snapshot in self._frozen_networks.items():
            network = getattr(self, name)
            if any(p.requires_grad or p.grad is not None for p in network.parameters()):
                raise RuntimeError(f"frozen physics gradient contract changed: {name}")
            for key, value in network.state_dict().items():
                if not torch.equal(value, network_snapshot[key].to(value)):
                    raise RuntimeError(f"frozen physics network changed: {name}.{key}")

    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False,
    ) -> _IncompatibleKeys:
        """Restore network weights only under the identical external physics contract.

        Check every fixed buffer before any mutation, including sparse topology,
        unit/config values and source fingerprints. All checkpoint items remain
        tensors for the existing checkpoint/freeze tooling.
        """
        for name, expected in self.named_buffers():
            incoming = state_dict.get(name)
            same = (isinstance(incoming, Tensor) and incoming.shape == expected.shape
                    and incoming.layout == expected.layout)
            if same and isinstance(incoming, Tensor):
                if expected.is_sparse:
                    a, b = incoming.coalesce(), expected.coalesce()
                    same = (torch.equal(a.indices().to(b.indices()), b.indices())
                            and torch.equal(a.values().to(b.values()), b.values()))
                else:
                    same = torch.equal(incoming.to(expected), expected)
            if not same:
                raise RuntimeError(f"frozen physics checkpoint contract mismatch: {name}")
        for network_name, snapshot in self._frozen_networks.items():
            for name in snapshot:
                value = state_dict.get(f"{network_name}.{name}")
                if not isinstance(value, Tensor) or not torch.isfinite(value).all():
                    raise RuntimeError("frozen physics checkpoint network must be present and finite")
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        self._frozen_networks = {
            name: {key: value.detach().clone() for key, value in network.state_dict().items()}
            for name, network in (("effective_length_network", self.effective_length_network),
                                  ("semantic_prior_network", self.semantic_prior_network))}
        self.assert_frozen_artifacts()
        return cast(_IncompatibleKeys, result)

    def optimizer_groups(self, *, lr: float, student: nn.Module | None = None) -> list[dict[str, Any]]:
        """Expose actual FD/student optimizer groups; ell and prior stay excluded."""
        self.assert_frozen_artifacts()
        if not math.isfinite(lr) or lr <= 0:
            raise ValueError("optimizer learning rate must be finite and positive")
        groups: list[dict[str, Any]] = [{"name": "physics", "lr": lr,
                                       "params": [p for p in self.parameters() if p.requires_grad]}]
        if student is not None:
            used = {id(p) for group in groups for p in group["params"]}
            groups.append({"name": "student", "lr": lr,
                           "params": [p for p in student.parameters()
                                      if p.requires_grad and id(p) not in used]})
        return groups

    def _parameter_components(
        self, sensor_semantics: Tensor, lane_counts: Tensor, regime_ids: Tensor,
        day_context: Tensor,
    ) -> tuple[dict[str, Tensor], Tensor, Tensor, Tensor]:
        raw: Tensor = self.fd_parameter_network(sensor_semantics)
        baseline = torch.sigmoid(raw + self.fd_regime_baseline(regime_ids)[:, None, :])
        features = torch.cat((day_context, functional.one_hot(regime_ids, self.num_regimes)
                              .to(day_context)), dim=-1)
        proposed: Tensor = torch.tanh(self.fd_day_offset_network(features))[:, None, :]
        # The distance toward the chosen bound keeps both absolute and delta bounds.
        margin = torch.where(proposed >= 0, 1. - baseline, baseline)
        offset = self.theta_max_offset_fraction * proposed * margin
        fraction = baseline + offset
        value = self.parameter_lower + (self.parameter_upper - self.parameter_lower) * fraction
        if self.fixed_semantic_prior.numel():
            prior = ((self.fixed_semantic_prior - self.parameter_lower)
                     / (self.parameter_upper - self.parameter_lower)).expand_as(fraction)
        else:
            prior = torch.sigmoid(self.semantic_prior_network(sensor_semantics))
        parameters = {"free_speed": value[..., 0], "wave_speed": value[..., 1],
                      "jam_density": value[..., 2] * lane_counts,
                      "capacity": value[..., 3] * lane_counts}
        return parameters, fraction, prior, offset

    def adaptive_fd_parameters(
        self, sensor_semantics: Tensor, lane_counts: Tensor, *,
        regime_ids: Tensor | None = None, day_context: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Bounded semantic/regime baseline plus day/regime offsets, [B,N]."""
        batch = sensor_semantics.shape[0]
        regime_ids = (torch.zeros(batch, dtype=torch.long, device=sensor_semantics.device)
                      if regime_ids is None else regime_ids)
        day_context = (sensor_semantics.new_zeros(batch, self.state_dim)
                       if day_context is None else day_context)
        return self._parameter_components(sensor_semantics, lane_counts, regime_ids,
                                           day_context)[0]

    def triangular_flow(self, density: Tensor, parameters: dict[str, Tensor]) -> Tensor:
        """Evaluate native stable smin(v_f*k,q_max,w*(k_j-k)) in veh/h."""
        expanded = {name: (value[:, None, :] if value.ndim == 2 else value)
                    for name, value in parameters.items()}
        candidates = torch.stack((expanded["free_speed"] * density,
                                  expanded["capacity"].expand_as(density),
                                  expanded["wave_speed"] * (expanded["jam_density"] - density)),
                                 dim=-1)
        return -self.smooth_min_temperature * torch.logsumexp(
            -candidates / self.smooth_min_temperature, dim=-1)

    @staticmethod
    def _mean(value: Tensor, mask: Tensor) -> Tensor:
        supported = mask.expand_as(value)
        return torch.where(supported, value, 0.).sum() / supported.sum().clamp_min(1)

    def forward(
        self, flow: Tensor, occupancy: Tensor, sensor_semantics: Tensor,
        cell_semantics: Tensor, cell_state: Tensor, lane_counts: Tensor,
        effective_length_km: Optional[Tensor] = None, *, speed: Tensor | None = None,
        episode_ids: Tensor | None = None, day_ordinals: Tensor | None = None,
        regime_ids: Tensor | None = None, day_context: Tensor | None = None,
        valid_cells: Tensor | None = None,
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """Return five scalar families and raw/normalized, support-aware diagnostics.

        cell_state remains [B,D*T-1,M,H] on the 4D path, including one slot per
        possible midnight. Invalid midnight slots are masked, never bridged.
        day_context is [B,D,H], regime_ids [B,D]; both default to zero.
        Optional valid_cells [B,S-1,M] is combined with temporal continuity.
        """
        self.assert_frozen_artifacts()
        if flow.ndim not in (3, 4) or occupancy.shape != flow.shape:
            raise ValueError("flow and occupancy require matching [B,S,N] or [B,D,T,N]")
        batch, nodes = flow.shape[0], flow.shape[-1]
        days, per_day = (flow.shape[1], flow.shape[2]) if flow.ndim == 4 else (1, flow.shape[1])
        steps = days * per_day
        if not batch or nodes != self.mapper.node_count or steps < 2 or not days or not per_day:
            raise ValueError("physics requires nonempty mapper nodes and at least two steps")
        if sensor_semantics.shape != (batch, nodes, self.semantic_dim):
            raise ValueError("sensor_semantics must have shape [B,N,semantic_dim]")
        if cell_semantics.shape != (batch, self.mapper.cell_count, self.semantic_dim):
            raise ValueError("cell_semantics must have shape [B,M,semantic_dim]")
        if cell_state.shape != (batch, steps - 1, self.mapper.cell_count, self.state_dim):
            raise ValueError("cell_state must have shape [B,S-1,M,state_dim]")
        if lane_counts.shape != (batch, nodes):
            raise ValueError("lane_counts must have shape [B,N]")
        supplied = {"flow": flow, "occupancy": occupancy, "sensor_semantics": sensor_semantics,
                    "cell_semantics": cell_semantics, "cell_state": cell_state, "lane_counts": lane_counts}
        if speed is not None:
            if speed.shape != flow.shape:
                raise ValueError("speed must have the same [B,S,N] or [B,D,T,N] shape as flow")
            supplied["speed"] = speed
        if effective_length_km is not None:
            supplied["effective_length_km"] = effective_length_km
        if day_context is not None:
            supplied["day_context"] = day_context
        for name, tensor in supplied.items():
            if not torch.isfinite(tensor).all():
                raise ValueError(f"enabled physics input must be finite: {name}")
            if tensor.device != flow.device:
                raise ValueError(f"physics inputs must share a device: {name}")
        if torch.any(lane_counts <= 0):
            raise ValueError("lane_counts must be positive")
        day_connected = torch.ones(batch, max(0, days - 1), dtype=torch.bool, device=flow.device)
        if flow.ndim == 4 or episode_ids is not None or day_ordinals is not None:
            if episode_ids is None or day_ordinals is None:
                raise ValueError("cross-day physics requires explicit episode IDs and day ordinals")
            if episode_ids.shape != (batch, days) or day_ordinals.shape != (batch, days):
                raise ValueError("episode IDs and day ordinals must have shape [B,D]")
            day_connected = consecutive_day_mask(episode_ids, day_ordinals).to(flow.device)
        transition = torch.ones(batch, steps - 1, dtype=torch.bool, device=flow.device)
        if days > 1:
            transition[:, per_day - 1::per_day] = day_connected
        support = transition[..., None].expand(batch, steps - 1, self.mapper.cell_count)
        if valid_cells is not None:
            if valid_cells.shape != support.shape or valid_cells.dtype != torch.bool:
                raise ValueError("valid_cells must be boolean [B,S-1,M]")
            support = support & valid_cells.to(flow.device)
        if regime_ids is None:
            regime_ids = torch.zeros(batch, days, dtype=torch.long, device=flow.device)
        if (regime_ids.shape != (batch, days) or regime_ids.dtype not in
                (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)
                or torch.any(regime_ids < 0) or torch.any(regime_ids >= self.num_regimes)):
            raise ValueError("regime_ids must be in-range integer [B,D]")
        regime_ids = regime_ids.to(device=flow.device, dtype=torch.long)
        if day_context is None:
            day_context = flow.new_zeros(batch, days, self.state_dim)
        if day_context.shape != (batch, days, self.state_dim):
            raise ValueError("day_context must have shape [B,D,state_dim]")
        flow, occupancy = flow.reshape(batch, steps, nodes), occupancy.reshape(batch, steps, nodes)
        if speed is not None:
            speed = speed.reshape_as(flow)
        if effective_length_km is not None:
            if effective_length_km.shape != (batch, nodes):
                raise ValueError("effective_length_km must have shape [B,N]")
            self._validate_length(effective_length_km)
            if self.calibrated_effective_length_km.numel() and not torch.equal(
                    effective_length_km, self.calibrated_effective_length_km.expand_as(effective_length_km)):
                raise ValueError("cannot override the frozen independent ell artifact")
            effective_length = effective_length_km.detach()
        elif self.calibrated_effective_length_km.numel():
            effective_length = self.calibrated_effective_length_km.expand(batch, -1)
        else:
            effective_length = self._bounded(
                self.effective_length_network(sensor_semantics).squeeze(-1),
                self.effective_length_bounds_km).detach()
        self._validate_length(effective_length)
        occupancy_scale = lane_counts[:, None, :] if self.occupancy_per_lane else 1.
        node_density = occupancy * occupancy_scale / effective_length[:, None, :]
        static_days = sensor_semantics[:, None].expand(-1, days, -1, -1).reshape(
            batch * days, nodes, self.semantic_dim)
        lane_days = lane_counts[:, None].expand(-1, days, -1).reshape(batch * days, nodes)
        daily_parameters, fraction, prior, offset = self._parameter_components(
            static_days, lane_days, regime_ids.reshape(-1), day_context.reshape(-1, self.state_dim))
        parameters = {name: value.reshape(batch, days, nodes).repeat_interleave(per_day, dim=1)
                      for name, value in daily_parameters.items()}
        if not all(torch.isfinite(p).all() for p in parameters.values()):
            raise ValueError("enabled physics parameters must be finite")
        fd_flow = self.triangular_flow(node_density, parameters)
        fd_residual = flow - fd_flow
        fd_normalized = fd_residual / self.residual_scale_values[0].to(flow)
        regimes_per_step = regime_ids.repeat_interleave(per_day, dim=1)
        fd_weights, cons_node_weights = torch.ones_like(flow), torch.ones_like(flow)
        missing_reliability = torch.ones_like(flow, dtype=torch.bool)
        if self.reliability_artifact is not None:
            fd_weights, missing_reliability = self.reliability_artifact.weights(
                "fd", self.station_ids, regimes_per_step)
            cons_node_weights, _ = self.reliability_artifact.weights(
                "cons", self.station_ids, regimes_per_step)
            fd_weights, cons_node_weights = fd_weights.to(flow), cons_node_weights.to(flow)
        fd_penalty = functional.huber_loss(fd_normalized, torch.zeros_like(fd_normalized),
                                           reduction="none", delta=1.)
        fd_loss = (fd_weights * fd_penalty).mean()

        cell_density, edge_flow = self.mapper.map_density(node_density), self.mapper.map_flow(flow)
        net_outflow = torch.einsum("bse,me->bsm", edge_flow[:, :-1], self.mapper.incidence)
        slack_rate_max = self.slack_rate_caps.view(1, -1).expand(batch, -1)
        slack_max = self.dt_hours * slack_rate_max / self.mapper.cell_lengths_km.view(1, -1)
        raw_slack: Tensor = self.slack_state_network(cell_state).squeeze(-1)
        slack = (slack_max[:, None] * torch.tanh(raw_slack)
                 * self.slack_cell_mask.view(1, 1, -1).to(raw_slack.dtype))
        density_change = cell_density[:, 1:] - cell_density[:, :-1]
        scaled_divergence = self.dt_hours * net_outflow / self.mapper.cell_lengths_km.view(1, 1, -1)
        conservation_residual = density_change + scaled_divergence - slack
        conservation_rate = conservation_residual * (
            self.mapper.cell_lengths_km.view(1, 1, -1) / self.dt_hours)
        cons_normalized = conservation_rate / self.residual_scale_values[1].to(flow)
        cons_penalty = functional.huber_loss(cons_normalized, torch.zeros_like(cons_normalized),
                                             reduction="none", delta=1.)
        cons_weights = self.mapper.map_density(cons_node_weights[:, :-1])
        cons_loss = self._mean(cons_weights * cons_penalty, support)

        variable_scales = self.variable_scales.to(flow)
        violations = {"negative_flow": -flow / variable_scales[0],
                      "excess_flow": (flow - parameters["capacity"]) / variable_scales[0],
                      "negative_density": -node_density / variable_scales[1],
                      "excess_density": (node_density - parameters["jam_density"]) / variable_scales[1]}
        if speed is not None:
            violations.update(negative_speed=-speed / variable_scales[2],
                              excess_speed=(speed - self.free_speed_bounds_kmh[1]) / variable_scales[2])
        cap_terms = {name: (self.bound_temperature * functional.softplus(
            violation / self.bound_temperature)).mean() for name, violation in violations.items()}
        zero = flow.sum() * 0.
        for name in ("negative_speed", "excess_speed"):
            cap_terms.setdefault(name, zero)
        triangular_capacity = (parameters["free_speed"] * parameters["wave_speed"]
                               * parameters["jam_density"]
                               / (parameters["free_speed"] + parameters["wave_speed"]))
        # Preserve the original named relative equality subterm under cap.
        capacity_equality = ((parameters["capacity"] - triangular_capacity)
                             / parameters["capacity"].clamp_min(1.)).square().mean()
        capacity_loss = sum(cap_terms.values(), zero) + self.capacity_equality_weight * capacity_equality

        normalized_upper = torch.stack((
            daily_parameters["free_speed"] / self.free_speed_bounds_kmh[1],
            daily_parameters["wave_speed"] / self.wave_speed_bounds_kmh[1],
            daily_parameters["jam_density"] / (self.jam_density_bounds_per_lane[1] * lane_days),
            daily_parameters["capacity"] / (self.capacity_bounds_per_lane[1] * lane_days)), dim=-1)
        ordered = normalized_upper[:, self.mapper.sensor_order]
        theta_spatial = (ordered[:, 1:] - ordered[:, :-1]).square().mean()
        fractions = fraction.reshape(batch, days, nodes, 4)
        theta_day_tv = self._mean((fractions[:, 1:] - fractions[:, :-1]).abs(),
                                  day_connected[..., None, None])
        theta_anchor = (fraction - prior).square().mean()
        theta_bounds = (functional.relu(-fraction) + functional.relu(fraction - 1.)).square().mean()
        theta_loss = theta_spatial + theta_day_tv + theta_anchor + theta_bounds

        # Frozen slack algebra and coefficients are deliberately unchanged.
        slack_l1 = self._mean(slack.abs(), support)
        if steps > 2:
            adjacent = support[:, 1:] & support[:, :-1]
            slack_delta = torch.where(adjacent, slack[:, 1:] - slack[:, :-1], 0.)
            slack_tv = self._mean(torch.sqrt(slack_delta.square().sum(dim=-1) + 1e-12),
                                  adjacent.any(dim=-1))
        else:
            slack_tv = zero
        raw_imbalance = (density_change + scaled_divergence).abs()
        explanation = (slack.abs() / (raw_imbalance + 1e-6)).clamp_max(1.)
        slack_relative = self._mean(explanation * raw_imbalance.square(), support)
        slack_loss = (self.slack_l1_weight * slack_l1 + self.slack_tv_weight * slack_tv
                      + self.slack_relative_weight * slack_relative)
        enabled_support = support & self.slack_cell_mask.view(1, 1, -1)
        enabled_ratio = self._mean(explanation, enabled_support)
        enabled_count = enabled_support.sum()
        losses = {"fd": fd_loss, "cons": cons_loss, "cap": capacity_loss,
                  "theta": theta_loss, "s": slack_loss}
        if not all(torch.isfinite(value).all() for value in losses.values()):
            raise ValueError("enabled physics produced nonfinite residuals or losses")
        diagnostics = {
            "enabled": torch.tensor(True, device=flow.device),
            "speed_bounds_evaluated": torch.tensor(speed is not None, device=flow.device),
            "residual_scales_fitted": torch.tensor(self.residual_scales is not None, device=flow.device),
            "effective_length_independently_calibrated": torch.tensor(
                self.effective_length_artifact is not None, device=flow.device),
            "effective_length_km": effective_length, "node_density": node_density,
            "fd_flow": fd_flow, "fd_residual": fd_residual, "fd_normalized_residual": fd_normalized,
            "cell_density": cell_density, "edge_flow": edge_flow,
            "conservation_residual": conservation_residual, "conservation_rate_residual": conservation_rate,
            "conservation_normalized_residual": cons_normalized, "conservation_valid_mask": support,
            "valid_transition_count": transition.sum(), "valid_conservation_cell_count": support.sum(),
            "reliability_fd_weights": fd_weights, "reliability_cons_weights": cons_weights,
            "reliability_missing_fraction": missing_reliability.to(flow.dtype).mean(),
            "fd_unweighted_huber": fd_penalty.mean(), "cons_unweighted_huber": self._mean(cons_penalty, support),
            "slack": slack, "slack_max": slack_max, "slack_rate_max": slack_rate_max,
            "slack_activation": (slack.abs() >= .95 * slack_max[:, None]).to(slack.dtype).mean(),
            "slack_explanation_ratio": explanation.mean(),
            "slack_relative_loss": slack_relative, "slack_l1": slack_l1, "slack_tv": slack_tv,
            "slack_enabled_explanation_ratio": enabled_ratio, "slack_enabled_valid_count": enabled_count,
            "slack_explanation_alarm": (enabled_count > 0) & (enabled_ratio > .5),
            "softplus_boundary_baseline": flow.new_tensor(self.bound_temperature * math.log(2.)),
            "cap_capacity_equality": capacity_equality,
            "cap_capacity_equality_weight": flow.new_tensor(self.capacity_equality_weight),
            "theta_spatial_smoothing": theta_spatial, "theta_day_tv": theta_day_tv,
            "theta_semantic_anchor": theta_anchor, "theta_parameter_bounds": theta_bounds,
            "theta_offset_fraction": offset.reshape(batch, days, nodes, 4),
            "theta_parameter_fraction": fractions,
        }
        diagnostics.update({f"cap_{name}": value for name, value in cap_terms.items()})
        diagnostics.update({f"hard_{name}_fraction": (value > 0).to(flow.dtype).mean()
                            for name, value in violations.items()})
        return losses, diagnostics


class DisabledPhysicsResidualCalculator(nn.Module):
    """Eligibility short-circuit before inspecting unavailable physical values."""

    def __init__(self, reason: str) -> None:
        super().__init__()
        if not reason:
            raise ValueError("disabled physics requires an explicit reason")
        self.reason = reason

    def forward(
        self, flow: Tensor, occupancy: Tensor | None = None, *args: object, **kwargs: object,
    ) -> tuple[dict[str, Tensor], dict[str, Any]]:
        # Empty sums retain the autograd connection without touching NaN/Inf data.
        zero = flow.reshape(-1)[:0].sum()
        if occupancy is not None:
            zero = zero + occupancy.reshape(-1)[:0].sum()
        speed = kwargs.get("speed")
        if isinstance(speed, Tensor):
            zero = zero + speed.reshape(-1)[:0].sum()
        return {name: zero for name in ("fd", "cons", "cap", "theta", "s")}, {
            "enabled": False, "reason": self.reason, "speed_bounds_evaluated": False}


def create_physics_calculator(
    *, audited: bool, semantic_dim: int, state_dim: int, mapper: TopologyMapper | None = None,
    eligible: bool = True, reason: str = "", **kwargs: Any,
) -> PhysicsResidualCalculator | DisabledPhysicsResidualCalculator:
    """Gate eligibility before constructing physics; never bypass TopologyMapper audit."""
    if not isinstance(audited, bool) or not isinstance(eligible, bool):
        raise ValueError("audited and eligible must be explicit booleans")
    if not audited or not eligible:
        return DisabledPhysicsResidualCalculator(
            reason or ("corridor topology audit pending" if not audited else "corridor is ineligible"))
    if mapper is None:
        raise ValueError("enabled physics requires an existing topology-audited mapper")
    return PhysicsResidualCalculator(mapper, semantic_dim, state_dim, **kwargs)
