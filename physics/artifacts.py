"""Frozen synthetic train artifacts, separate from residual and component scales."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from training.scaling import MADArtifact


def _stamp(stamp: str) -> None:
    if datetime.fromisoformat(stamp).tzinfo is None:
        raise ValueError("freeze timestamp requires an explicit timezone")


def _fingerprint(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()


def validate_residual_scales(scales: MADArtifact) -> None:
    """Require the frozen residual layer in veh/h, never top-level component MAD."""
    if scales.layer != "residual" or scales.schema_version != 1:
        raise ValueError("physics requires the frozen residual scale layer")
    if scales.dataset_kind != "synthetic":
        raise ValueError("this physics artifact path is synthetic-only")
    _stamp(scales.frozen_at)
    if not scales.source_ids:
        raise ValueError("residual scales require legal training source IDs")
    for name in ("fd", "cons"):
        item = scales.entry(name)
        if item.unit != "veh/h" or not math.isfinite(item.scale) or item.scale <= 0:
            raise ValueError("fd and cons residual scales must be finite positive veh/h values")


@dataclass(frozen=True)
class ReliabilityEntry:
    name: str
    station_id: int
    regime_id: int
    unit: str
    median: float
    mad: float
    floor: float
    floor_applied: bool
    weight: float
    count: int
    sample_sha256: str


@dataclass(frozen=True)
class ReliabilityArtifact:
    schema_version: int
    layer: str
    definition: str
    dataset_kind: str
    frozen_at: str
    source_ids: tuple[str, ...]
    excluded_station_ids: tuple[int, ...]
    residual_scale_fingerprint: str
    entries: tuple[ReliabilityEntry, ...]

    def __post_init__(self) -> None:
        if (self.schema_version != 1 or self.layer != "local_reliability"
                or self.dataset_kind != "synthetic"):
            raise ValueError("unsupported reliability schema/layer or non-synthetic scope")
        _stamp(self.frozen_at)
        if (len(self.residual_scale_fingerprint) != 64 or not self.source_ids
                or not self.entries or len(set(self.source_ids)) != len(self.source_ids)):
            raise ValueError("reliability needs frozen residual fingerprint and unique source IDs")
        keys = set()
        for entry in self.entries:
            key = (entry.name, entry.station_id, entry.regime_id)
            if key in keys or entry.station_id in self.excluded_station_ids:
                raise ValueError("duplicate reliability group or permanent-U training support")
            keys.add(key)
            if (entry.name not in {"fd", "cons"} or entry.unit != "dimensionless"
                    or entry.count < 1 or entry.regime_id < 0
                    or not all(math.isfinite(v) for v in
                               (entry.median, entry.mad, entry.floor, entry.weight))
                    or entry.mad < 0 or entry.floor <= 0 or not .25 <= entry.weight <= 4.):
                raise ValueError("reliability weight/MAD metadata violates frozen bounds")
            expected = min(4., max(.25, 1. / max(entry.mad, entry.floor)))
            if entry.weight != expected or entry.floor_applied != (entry.mad < entry.floor):
                raise ValueError("reliability weight does not match its recorded MAD and floor")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(asdict(self))

    def entry(self, name: str, station_id: int, regime_id: int) -> ReliabilityEntry:
        for item in self.entries:
            if (item.name, item.station_id, item.regime_id) == (name, station_id, regime_id):
                return item
        raise KeyError((name, station_id, regime_id))

    def weights(self, name: str, station_ids: Tensor, regime_ids: Tensor) -> tuple[Tensor, Tensor]:
        """Return fixed [...,N] weights and missing-group flags.

        Unseen stations/regimes (including U) receive the explicit neutral 1.0
        fallback. No target labels are fitted, and missing support stays visible.
        """
        if name not in {"fd", "cons"} or station_ids.ndim != 1:
            raise ValueError("reliability lookup needs fd/cons and one-dimensional station IDs")
        result = torch.ones(*regime_ids.shape, station_ids.numel(), device=regime_ids.device)
        missing = torch.ones_like(result, dtype=torch.bool)
        stations = station_ids.to(regime_ids.device)
        for entry in self.entries:
            if entry.name != name:
                continue
            selected = ((regime_ids[..., None] == entry.regime_id)
                        & (stations == entry.station_id))
            result = torch.where(selected, entry.weight, result)
            missing &= ~selected
        return result, missing

    def save(self, path: str | Path) -> None:
        payload = {**asdict(self), "fingerprint": self.fingerprint}
        with Path(path).open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")

    @classmethod
    def load(cls, path: str | Path) -> ReliabilityArtifact:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        expected = payload.pop("fingerprint")
        payload["entries"] = tuple(ReliabilityEntry(**e) for e in payload["entries"])
        for name in ("source_ids", "excluded_station_ids"):
            payload[name] = tuple(payload[name])
        artifact = cls(**payload)
        if artifact.fingerprint != expected:
            raise ValueError("reliability artifact fingerprint mismatch")
        return artifact


def fit_reliability_artifact(
    values: Mapping[str, Tensor], masks: Mapping[str, Tensor], *, residual_scales: MADArtifact,
    floor: float, source_ids: Sequence[str], station_ids: Sequence[int],
    regime_ids: Sequence[int], splits: Sequence[str], excluded_station_ids: Sequence[int],
    dataset_kind: str, frozen_at: str,
) -> ReliabilityArtifact:
    """Fit clip(1/max(MAD(raw residual / frozen scale),floor), .25,4).

    Inputs are raw veh/h residuals. Masking precedes finite checks and statistics;
    only train rows outside the permanent-U exclusion contribute to any bytes.
    """
    validate_residual_scales(residual_scales)
    if dataset_kind != "synthetic":
        raise ValueError("reliability fitting is synthetic-only")
    _stamp(frozen_at)
    if not math.isfinite(floor) or floor <= 0:
        raise ValueError("standardized reliability floor must be finite and positive")
    if not set(excluded_station_ids).issubset(residual_scales.excluded_station_ids):
        raise ValueError("residual scales must have excluded every permanent-U station")
    n = len(source_ids)
    if (not n or any(len(items) != n for items in (station_ids, regime_ids, splits))
            or len(set(source_ids)) != n or any(not item for item in source_ids)):
        raise ValueError("reliability provenance must match rows with unique nonempty source IDs")
    if any(s not in {"train", "validation", "tuning", "calibration", "test"} for s in splits):
        raise ValueError("unknown split provenance")
    if any(isinstance(g, bool) or not isinstance(g, int) or g < 0 for g in regime_ids):
        raise ValueError("regime IDs must be nonnegative integers")
    if set(values) != {"fd", "cons"} or set(masks) != set(values):
        raise ValueError("reliability requires both fd and cons raw residuals and masks")
    allowed = [s == "train" and station not in excluded_station_ids
               for s, station in zip(splits, station_ids)]
    if not any(allowed):
        raise ValueError("no legal train support after permanent-U exclusion")
    groups = sorted({(station, regime) for station, regime, keep in
                     zip(station_ids, regime_ids, allowed) if keep})
    entries = []
    for name in sorted(values):
        value, mask = values[name], masks[name]
        if value.ndim < 1 or value.shape[0] != n or mask.shape != value.shape:
            raise ValueError("residuals and masks must match the provenance row axis")
        if mask.dtype != torch.bool:
            raise ValueError("reliability masks must be boolean")
        for station, regime in groups:
            rows = torch.tensor([keep and st == station and rg == regime
                                 for keep, st, rg in zip(allowed, station_ids, regime_ids)],
                                device=value.device)
            legal_mask = mask.to(value.device)[rows]
            selected = value.detach()[rows][legal_mask].to(device="cpu", dtype=torch.float64)
            if not selected.numel():
                continue  # Missing station/regime support is an explicit runtime fallback.
            if not torch.isfinite(selected).all():
                raise ValueError("visible train reliability residuals must be finite")
            normalized = selected / residual_scales.entry(name).scale
            median = float(torch.quantile(normalized, .5).item())
            mad = float(torch.quantile((normalized - median).abs(), .5).item())
            digest = hashlib.sha256(normalized.contiguous().numpy().tobytes())
            digest.update(legal_mask.detach().cpu().contiguous().numpy().tobytes())
            entries.append(ReliabilityEntry(
                name, station, regime, "dimensionless", median, mad, floor, mad < floor,
                min(4., max(.25, 1. / max(mad, floor))), selected.numel(), digest.hexdigest()))
    return ReliabilityArtifact(
        1, "local_reliability", "clip(1/max(MAD(raw_veh_h/S),floor),0.25,4);midpoint-even;factor=1",
        dataset_kind, frozen_at, tuple(s for s, keep in zip(source_ids, allowed) if keep),
        tuple(sorted(set(excluded_station_ids))), residual_scales.fingerprint, tuple(entries))


@dataclass(frozen=True)
class EffectiveLengthArtifact:
    schema_version: int
    source_kind: str
    dataset_kind: str
    split: str
    frozen_at: str
    source_ids: tuple[str, ...]
    station_ids: tuple[int, ...]
    values_km: tuple[float, ...]
    bounds_km: tuple[float, float]
    unit: str = "km"

    def __post_init__(self) -> None:
        if (self.schema_version != 1 or self.source_kind != "independent_training_calibration"
                or self.dataset_kind != "synthetic" or self.split != "train" or self.unit != "km"):
            raise ValueError("ell requires an independent synthetic train calibration source in km")
        _stamp(self.frozen_at)
        if (not self.station_ids or len(self.station_ids) != len(self.values_km)
                or len(set(self.station_ids)) != len(self.station_ids)
                or not self.source_ids or any(not s for s in self.source_ids)):
            raise ValueError("ell requires unique stations, one value per station and source IDs")
        lower, upper = self.bounds_km
        if not (math.isfinite(lower) and math.isfinite(upper) and 0 < lower < upper):
            raise ValueError("ell bounds must be finite, positive, and ordered")
        if any(not math.isfinite(v) or not lower <= v <= upper for v in self.values_km):
            raise ValueError("ell values must be finite and within calibrated absolute bounds")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(asdict(self))

    def save(self, path: str | Path) -> None:
        with Path(path).open("x", encoding="utf-8") as stream:
            json.dump({**asdict(self), "fingerprint": self.fingerprint}, stream,
                      ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")

    @classmethod
    def load(cls, path: str | Path) -> EffectiveLengthArtifact:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        expected = payload.pop("fingerprint")
        for name in ("source_ids", "station_ids", "values_km", "bounds_km"):
            payload[name] = tuple(payload[name])
        artifact = cls(**payload)
        if artifact.fingerprint != expected:
            raise ValueError("ell artifact fingerprint mismatch")
        return artifact


def make_effective_length_artifact(
    values_km: Tensor, *, station_ids: Sequence[int], source_ids: Sequence[str], split: str,
    independent_source: bool, dataset_kind: str, frozen_at: str,
    bounds_km: tuple[float, float] = (.004, .012),
) -> EffectiveLengthArtifact:
    """Freeze already independently calibrated values; do not fit from FD q/o.

    Independence is explicit provenance supplied by the caller, not an empirical
    claim verified by this synthetic-only boundary. N13 stays open.
    """
    if independent_source is not True:
        raise ValueError("ell needs an independent calibration source, not joint q/o fitting")
    if values_km.ndim != 1:
        raise ValueError("ell calibration values must have shape [N]")
    values = values_km.detach().cpu()
    lower, upper = bounds_km
    # A float32 encoding of a declared endpoint may round outside its decimal
    # Python float. Canonicalize only exact dtype-encoded endpoints; no tolerance
    # or general clipping admits out-of-bound calibration values.
    normalized_values = tuple(lower if v == values.new_tensor(lower) else
                              upper if v == values.new_tensor(upper) else float(v.item())
                              for v in values)
    return EffectiveLengthArtifact(
        1, "independent_training_calibration", dataset_kind, split, frozen_at,
        tuple(source_ids), tuple(station_ids), normalized_values, bounds_km)
