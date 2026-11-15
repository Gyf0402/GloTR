"""Frozen, training-only MAD artifacts with explicit units and normalization layers."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import torch
from torch import Tensor


@dataclass(frozen=True)
class MADEntry:
    name: str
    unit: str
    median: float
    mad: float
    floor: float
    scale: float
    count: int
    floor_applied: bool
    sample_sha256: str


@dataclass(frozen=True)
class MADArtifact:
    schema_version: int
    definition: str
    layer: str
    dataset_kind: str
    frozen_at: str
    source_ids: tuple[str, ...]
    excluded_station_ids: tuple[int, ...]
    upstream_fingerprint: str | None
    entries: tuple[MADEntry, ...]

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(asdict(self), sort_keys=True, allow_nan=False).encode()
        return hashlib.sha256(encoded).hexdigest()

    def entry(self, name: str) -> MADEntry:
        for item in self.entries:
            if item.name == name:
                return item
        raise KeyError(name)

    def normalize_error(self, name: str, error: Tensor) -> Tensor:
        """Scale an error, without subtracting the median or changing its target."""
        return error / error.new_tensor(self.entry(name).scale)

    def assert_upstream(self, fingerprint: str) -> None:
        if self.layer != "component" or self.upstream_fingerprint != fingerprint:
            raise ValueError("component MAD has a different upstream normalization artifact")

    def save(self, path: str | Path) -> None:
        payload = {**asdict(self), "fingerprint": self.fingerprint}
        with Path(path).open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")

    @classmethod
    def load(cls, path: str | Path) -> MADArtifact:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        expected = payload.pop("fingerprint")
        payload["entries"] = tuple(MADEntry(**entry) for entry in payload["entries"])
        payload["source_ids"] = tuple(payload["source_ids"])
        payload["excluded_station_ids"] = tuple(payload["excluded_station_ids"])
        artifact = cls(**payload)
        if artifact.fingerprint != expected:
            raise ValueError("MAD artifact fingerprint mismatch")
        if artifact.schema_version != 1 or artifact.dataset_kind != "synthetic":
            raise ValueError("unsupported MAD artifact schema or data scope")
        return artifact


def fit_mad_artifact(
    values: Mapping[str, Tensor], masks: Mapping[str, Tensor], *,
    units: Mapping[str, str], floors: Mapping[str, float], layer: str,
    source_ids: Sequence[str], station_ids: Sequence[int], splits: Sequence[str],
    excluded_station_ids: Sequence[int], dataset_kind: str, frozen_at: str,
    upstream_fingerprint: str | None = None,
) -> MADArtifact:
    """Fit once from legal rows, each followed by any sample dimensions.

    Median uses the midpoint of the two central observations for even samples;
    MAD is median(abs(x - median(x))), with no Gaussian-consistency multiplier.
    Units belong to the samples, e.g. veh/h for residuals; component artifacts
    require dimensionless samples and the upstream artifact identity. This
    entry point is synthetic-only until the real-data gate is explicitly opened.
    """
    if dataset_kind != "synthetic":
        raise ValueError("real-data MAD fitting is gated; this entry point is synthetic-only")
    if layer not in {"channel", "residual", "component"}:
        raise ValueError("unknown normalization layer")
    if layer == "component" and (not upstream_fingerprint or len(upstream_fingerprint) != 64):
        raise ValueError("component normalization requires an upstream fingerprint")
    if layer != "component" and upstream_fingerprint is not None:
        raise ValueError("only a component artifact has an upstream normalization layer")
    if not values or not (set(values) == set(masks) == set(units) == set(floors)):
        raise ValueError("values, masks, units and floors need identical nonempty keys")
    if layer == "component" and any(unit != "dimensionless" for unit in units.values()):
        raise ValueError("top-level component MAD samples must already be dimensionless")
    if any(not unit for unit in units.values()):
        raise ValueError("every scale requires a unit")
    stamp = datetime.fromisoformat(frozen_at)
    if stamp.tzinfo is None:
        raise ValueError("freeze timestamp requires an explicit timezone")
    n = len(source_ids)
    if len(station_ids) != n or len(splits) != n or n == 0:
        raise ValueError("provenance must match the profile axis")
    if len(set(source_ids)) != n or any(not identifier for identifier in source_ids):
        raise ValueError("source IDs must be unique and nonempty")
    if any(split not in {"train", "validation", "calibration", "test"} for split in splits):
        raise ValueError("unknown split provenance")
    excluded = set(excluded_station_ids)
    allowed = [split == "train" and station not in excluded
               for split, station in zip(splits, station_ids)]
    if not any(allowed):
        raise ValueError("no legal training support after permanent-U exclusion")
    entries = []
    for name in sorted(values):
        value, mask, floor = values[name], masks[name], float(floors[name])
        if value.ndim < 1 or value.shape[0] != n or mask.shape != value.shape:
            raise ValueError("sample tensors and masks must match the provenance profile axis")
        if mask.dtype != torch.bool:
            raise ValueError("MAD visibility masks must be boolean")
        if not math.isfinite(floor) or floor <= 0:
            raise ValueError("MAD floors must be finite and positive in their declared units")
        selected_rows = torch.tensor(allowed, dtype=torch.bool, device=value.device)
        legal_values = value.detach()[selected_rows]
        legal_mask = mask.to(value.device)[selected_rows]
        selected = legal_values[legal_mask].to(device="cpu", dtype=torch.float64).contiguous()
        if not selected.numel():
            raise ValueError(f"no visible training support for {name}")
        if not torch.isfinite(selected).all():
            raise ValueError(f"visible training samples must be finite: {name}")
        median = torch.quantile(selected, 0.5)
        mad = torch.quantile((selected - median).abs(), 0.5)
        raw_mad = float(mad.item())
        digest = hashlib.sha256(selected.numpy().tobytes())
        digest.update(legal_mask.detach().cpu().contiguous().numpy().tobytes())
        entries.append(MADEntry(name, units[name], float(median.item()), raw_mad,
                                floor, max(raw_mad, floor), selected.numel(),
                                raw_mad < floor, digest.hexdigest()))
    return MADArtifact(
        1, "median(abs(x-median(x)));midpoint-even;gaussian_factor=1", layer,
        dataset_kind, frozen_at,
        tuple(identifier for identifier, keep in zip(source_ids, allowed) if keep),
        tuple(sorted(excluded)), upstream_fingerprint, tuple(entries),
    )
