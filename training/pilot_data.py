"""Pure data transforms for the approved, purpose-limited q/o pilot."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from losses.joint_objective import JointOptimizationLoss
from training.real_integration import channel_scale, mask_inputs
from training.real_scope import RULE_SHA, canonical_stations

PROCESSING_SHA = "9851165e96716ec270ab2c5cbbcd4339b0014d3a1d121d64e21ea033b416f12d"
ELIGIBILITY_SHA = "119782a6993889cfd6eb0628710f184cc9593285976c1b27762cc7629634da7c"


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def array_digest(value: NDArray[Any], channels: tuple[str, ...]) -> str:
    metadata = {"dtype": value.dtype.str, "shape": list(value.shape), "channels": channels}
    return hashlib.sha256(canonical_digest(metadata).encode()
                          + np.ascontiguousarray(value).tobytes()).hexdigest()


def _summary(values: Tensor, visible: Tensor) -> Tensor:
    if values.shape != visible.shape or visible.dtype != torch.bool or values.ndim != 4:
        raise ValueError("history needs matching [D,N,T,C] boolean support")
    if not torch.isfinite(values[visible]).all():
        raise ValueError("visible history must be finite")
    safe = torch.where(visible, values, torch.zeros_like(values))
    count = visible.sum(dim=(0, 2))
    mean = safe.sum(dim=(0, 2)) / count.clamp_min(1)
    centered = torch.where(visible, safe - mean[None, :, None], torch.zeros_like(values))
    deviation = (centered.square().sum(dim=(0, 2)) / count.clamp_min(1)).sqrt()
    return torch.cat((mean, deviation, (count > 0).to(values)), dim=-1)


def rolling_history(values: Tensor, visible: Tensor) -> Tensor:
    """The current day and hidden entries cannot enter the previous-three-day summary."""
    return torch.stack([_summary(values[max(0, day - 3):day], visible[max(0, day - 3):day])
                        for day in range(values.shape[0])])


@dataclass(frozen=True)
class EpisodeInputs:
    values: Tensor
    visible: Tensor
    history: Tensor
    outage: Tensor


def episode_inputs(values: Tensor, visible: Tensor, *, target_index: int) -> EpisodeInputs:
    """No reference labels or target reference metadata in this model-side object."""
    if values.ndim != 4 or values.shape[0] != 13 or values.shape[-1] != 2:
        raise ValueError("pilot episodes require 13 days and two q/o channels")
    if not 0 <= target_index < values.shape[1]:
        raise ValueError("target node outside graph axis")
    outage = torch.zeros(values.shape[:2], dtype=torch.bool, device=values.device)
    outage[3:10, target_index] = True
    clean, m_in, _ = mask_inputs(values, visible, outage)
    history = _summary(clean[:3], m_in[:3])[None].expand(13, -1, -1).clone()
    return EpisodeInputs(clean, m_in, history, outage)


def training_anchors(
    visible: NDArray[np.bool_], dates: tuple[str, ...], stations: tuple[str, ...],
    *, maximum: int = 20,
) -> list[dict[str, Any]]:
    canonical_stations(stations)
    if (visible.shape != (len(dates), len(stations), 288, 2)
            or visible.dtype != np.bool_ or maximum < 1):
        raise ValueError("training anchors require matching legal q/o masks")
    ordinals = [date.fromisoformat(x).toordinal() for x in dates]
    if any(b != a + 1 for a, b in zip(ordinals, ordinals[1:])):
        raise ValueError("training date axis must be contiguous")
    complete = visible.all(axis=-1).sum(axis=-1) >= 274
    grouped: list[list[dict[str, Any]]] = []
    for station in sorted(stations, key=int):
        index = stations.index(station)
        available, next_start = [], 3
        for start in range(3, len(dates) - 9):
            if start >= next_start and complete[start - 3:start + 10, index].all():
                available.append({"target_station_id": station, "start_day": dates[start],
                                  "start_index": start, "duration_days": 7,
                                  "support_start_day": dates[start - 3],
                                  "support_end_day_exclusive": date.fromordinal(
                                      ordinals[start] + 10).isoformat()})
                next_start = start + 13
        if available:
            grouped.append(available)
    return ([group[0] for group in grouped]
            + [item for group in grouped for item in group[1:]])[:maximum]


def static_features(graph: dict[str, Any]) -> Tensor:
    order = canonical_stations(tuple(str(x) for x in graph["station_order"]))
    lookup = {str(node["station_id"]): node for node in graph["nodes"]}
    if len(lookup) != len(graph["nodes"]) or set(lookup) != set(order):
        raise ValueError("static node records must match the unique graph IDs")
    rows = [[float(lookup[x]["abs_pm"]) / 100, float(lookup[x]["latitude"]) / 90,
             float(lookup[x]["longitude"]) / 180, float(lookup[x]["candidate_lanes"]) / 8,
             float(lookup[x]["type"] == "ML")] for x in order]
    result = torch.tensor(rows, dtype=torch.float32)
    if not torch.isfinite(result).all():
        raise ValueError("static metadata must be finite")
    return result


def calendar_features(dates: tuple[str, ...]) -> Tensor:
    rows = []
    for value in dates:
        day = date.fromisoformat(value)
        week = 2 * math.pi * day.weekday() / 7
        annual = 2 * math.pi * (day - date(day.year, 1, 1)).days / 365
        rows.append([math.sin(week), math.cos(week), math.sin(annual), math.cos(annual)])
    return torch.tensor(rows, dtype=torch.float32)


def _stamp(value: str) -> None:
    if datetime.fromisoformat(value).tzinfo is None:
        raise ValueError("freeze timestamp needs a timezone")


def fit_channel_artifact(
    values: NDArray[Any], visible: NDArray[np.bool_], *, source_id: str, frozen_at: str,
) -> dict[str, Any]:
    """Same audited MAD arithmetic as B, newly fitted for C and never a B artifact."""
    _stamp(frozen_at)
    artifact = channel_scale(values, visible, floors=(1., .001), source_id=source_id)
    artifact.update(purpose="restricted_pilot", frozen_at=frozen_at,
                    provenance_scope="inner-fit only; permanent U excluded by PilotScope")
    return artifact


def fit_component_artifact(
    rows: list[dict[str, float]], *, source_ids: list[str], channel_sha: str, frozen_at: str,
) -> dict[str, Any]:
    _stamp(frozen_at)
    if (not rows or len(rows) != len(source_ids) or len(set(source_ids)) != len(source_ids)
            or len(channel_sha) != 64):
        raise ValueError("component MAD needs distinct training sources and upstream SHA")
    entries, scales = {}, {}
    for name in JointOptimizationLoss.LOSS_NAMES:
        if name in ("imp", "obs", "coef") and any(name not in row for row in rows):
            raise ValueError("active data components cannot be omitted")
        values = np.array([row.get(name, 0.) for row in rows], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("component MAD samples must be finite")
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        scales[name] = max(mad, 1e-6)
        entries[name] = {"median": median, "mad": mad, "floor": 1e-6,
                         "floor_applied": mad < 1e-6, "count": len(values),
                         "sample_sha256": hashlib.sha256(values.tobytes()).hexdigest()}
    return {"version": 1, "purpose": "restricted_pilot", "pilot_only": True,
            "formal_MAD": False, "layer": "component", "units": "dimensionless",
            "definition": "median(abs(x-median(x))); midpoint median; no multiplier",
            "fit_state": "initial student; frozen representation; training episodes only",
            "channel_sha256": channel_sha, "source_ids": source_ids, "frozen_at": frozen_at,
            "entries": entries, "scales": scales}


def episode_identity(
    station: str, start: str, outage: NDArray[np.bool_], support: NDArray[np.bool_],
) -> dict[str, str]:
    canonical_stations((station,))
    if (outage.dtype != np.bool_ or support.dtype != np.bool_
            or support.shape != (*outage.shape, 288, 2)):
        raise ValueError("identity requires actual outage and scoring mask shapes")
    if (support & ~outage[..., None, None]).any():
        raise ValueError("scoring support must be inside the outage")
    fields = {"namespace": "pilot-only", "processing_sha": PROCESSING_SHA,
              "eligibility_sha": ELIGIBILITY_SHA, "rule_sha": RULE_SHA,
              "clock_view": "fixed_UTC_minus_8_candidate", "target_station_ids": [station],
              "start_day": date.fromisoformat(start).isoformat(), "duration_days": 7,
              "protocol": "K"}
    artificial = np.broadcast_to(outage[..., None, None], (*outage.shape, 288, 3))
    a_digest = array_digest(artificial, ("q", "v", "o"))
    e_digest = array_digest(support, ("q", "o"))
    return {"episode_id": "pilot-only:" + canonical_digest(fields),
            "mask_id": "pilot-only:" + canonical_digest(
                {**fields, "A_digest": a_digest, "E_digest": e_digest}),
            "A_digest": a_digest, "E_digest": e_digest}
