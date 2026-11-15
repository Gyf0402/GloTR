"""The approved q/o integration boundary. This does not enable real experiments.

All learned inputs pass through a pinned, purpose-specific run manifest. The
older synthetic TrainingScope and MAD entry points deliberately remain closed.
"""

from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

RULE_SHA = "f507941abb276006eca13df7e8ed635c44a6e9cf15a2c743c830061fff57a800"
ERRATUM_SHA = "6438f11c7a3dbec95b4f24d4affb9ad8a8496d9a1d70c2c4348ec90bd3ef3350"
CHANNELS = ("q", "v", "o")
UNITS = ("veh/h", "km/h", "fraction")
ACTIVE = (0, 2)
U_PREFIX = "permanent-u-v1|20260906|"
CODE_FILES = ("training/real_scope.py", "training/real_integration.py",
              "scripts/check_real_integration.py", "representations/basis.py",
              "training/stages.py", "training/telemetry.py", "models/film_rgnn.py",
              "models/temporal.py", "physics/corridor_physics.py", "losses/joint_objective.py")


def sha256_file(path: str | Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def pin(path: str | Path) -> dict[str, str]:
    resolved = Path(path).resolve()
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def pinned_bytes(binding: Mapping[str, Any]) -> bytes:
    digest = binding.get("sha256")
    if (not isinstance(digest, str) or len(digest) != 64
            or not isinstance(binding.get("path"), str)):
        raise ValueError("a non-null path/SHA binding is required")
    content = Path(binding["path"]).read_bytes()
    if hashlib.sha256(content).hexdigest() != digest:
        raise ValueError(f"SHA mismatch: {binding['path']}")
    return content


def read_pinned_json(binding: Mapping[str, Any]) -> dict[str, Any]:
    value = json.loads(pinned_bytes(binding))
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object")
    return value


def write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def canonical_stations(stations: Sequence[str]) -> tuple[str, ...]:
    result = tuple(stations)
    if (not result or len(set(result)) != len(result)
            or any(not isinstance(x, str) or not x.isascii() or not x.isdecimal()
                   or int(x) <= 0 or str(int(x)) != x for x in result)):
        raise ValueError("station IDs must be unique canonical positive decimal strings")
    return result


def permanent_u(stations: Sequence[str]) -> tuple[str, ...]:
    return tuple(sorted((x for x in canonical_stations(stations)
                         if 5 * int(hashlib.sha256((U_PREFIX + x).encode()).hexdigest(), 16)
                         < 2**256), key=int))


def _calendar(axis: Sequence[str]) -> tuple[date, ...]:
    parsed = tuple(date.fromisoformat(x) for x in axis)
    if (not parsed or any(b - a != timedelta(days=1) for a, b in zip(parsed, parsed[1:]))
            or tuple(x.isoformat() for x in parsed) != tuple(axis)):
        raise ValueError("calendar must be canonical, ordered and contiguous")
    return parsed


def calendar_roles(axis: Sequence[str]) -> dict[str, str]:
    n = len(_calendar(axis))
    cuts = [0, n * 60 // 100, n * 75 // 100, n * 85 // 100, n]
    return {day: role for a, b, role in zip(cuts, cuts[1:],
            ("train", "selection", "calibration", "test")) for day in axis[a:b]}


def inner_fit_dates(pool: Sequence[str]) -> tuple[str, ...]:
    _calendar(pool)
    return tuple(pool[:len(pool) * 60 // 100])


def support_inside(start: str, duration: int, allowed_dates: Sequence[str]) -> bool:
    if duration < 1:
        raise ValueError("positive duration required")
    first = date.fromisoformat(start) - timedelta(days=3)
    allowed = set(allowed_dates)
    return all((first + timedelta(days=i)).isoformat() in allowed for i in range(duration + 6))


@dataclass(frozen=True)
class DayView:
    values: NDArray[np.float64]  # Selected station, time, q/o; excluded values are NaN.
    visible: NDArray[np.bool_]
    native_speed_available: NDArray[np.bool_]
    diagnostics: dict[str, Any]


def validate_day(
    arrays: Mapping[str, NDArray[Any]], stations: tuple[str, ...], excluded: tuple[str, ...],
) -> DayView:
    """Validate the stored q/v/o contract, then form a legal, U-free q/o view.

    n_present is a provenance diagnostic only. Neither it nor partial_values or
    hidden target reference metadata is included in a model feature tensor.
    """
    canonical_stations(stations)
    ids = arrays["station_ids"]
    if ids.ndim != 1 or ids.dtype != np.int32 or len(set(ids.tolist())) != len(ids):
        raise ValueError("source station IDs must be unique int32 vector")
    canonical_stations(tuple(str(x) for x in ids.tolist()))
    lookup = {str(x): i for i, x in enumerate(ids.tolist())}
    if not set(stations).issubset(lookup):
        raise ValueError("a registered station is absent from this day")
    indices = [lookup[x] for x in stations]
    n, shape = len(ids), (len(ids), 288, 3)
    raw = arrays["values"]
    if raw.shape != shape or raw.dtype != np.float64:
        raise ValueError("values must be float64 [N,288,3] in frozen q/v/o units")
    masks: dict[str, NDArray[np.bool_]] = {}
    for name in ("source_present", "value_available", "quality_accepted", "scorable"):
        item = arrays[name]
        if item.dtype != np.bool_ or item.shape != shape:
            raise ValueError(f"{name} must be matching bool [N,288,3]")
        masks[name] = item
    if not np.array_equal(masks["scorable"], masks["value_available"] & masks["quality_accepted"]):
        raise ValueError("derived scorable identity failed")
    for name in ("coverage_available", "calendar_slot_exists"):
        item = arrays[name]
        if item.dtype != np.bool_ or item.shape != (288,):
            raise ValueError(f"{name} must be bool [288]")
    if not arrays["calendar_slot_exists"].all():
        raise ValueError("fixed 288-slot calendar cannot absorb source clock gaps")
    config = arrays["configuration_consensus"]
    if config.dtype != np.bool_ or config.shape != (n,):
        raise ValueError("configuration_consensus must be bool [N]")
    hard = np.broadcast_to(config[:, None], (n, 288)).copy()
    for name in ("unexpected_reported_lane", "conflicting_measurement", "ambiguous_source_time"):
        item = arrays[name]
        if item.dtype != np.bool_ or item.shape != (n, 288):
            raise ValueError(f"{name} must be bool [N,288]")
        hard &= ~item
    hard &= arrays["coverage_available"][None, :]
    hard &= arrays["calendar_slot_exists"][None, :]
    excluded_nodes = np.asarray([x in excluded for x in stations], dtype=bool)
    selected = raw[indices]
    legal3 = masks["scorable"][indices] & hard[indices, :, None]
    legal3[excluded_nodes] = False  # Before finite checks or any statistics.
    if not np.isfinite(selected[legal3]).all():
        raise ValueError("legal values must be finite; natural missing values may be NaN")
    if np.any(masks["value_available"] & ~masks["source_present"]):
        raise ValueError("value availability cannot exceed source presence")
    visible = legal3[..., list(ACTIVE)].copy()
    values = np.where(visible, selected[..., list(ACTIVE)], np.nan)
    native = arrays["n_present"]
    if native.shape != (n, 288, 8, 3) or native.dtype != np.uint16:
        raise ValueError("n_present must be uint16 [N,288,8,3]")
    speed = np.asarray((native[indices, :, :, 1] > 0).any(axis=(1, 2)), dtype=np.bool_)
    diagnostics = {
        "canonical_channels": list(CHANNELS), "canonical_units": list(UNITS),
        "active_channels": ["q", "o"], "unit_conversion": "already converted; no second scaling",
        "scorable_qo_counts": visible.sum(axis=1).tolist(),
        "joint_qo_counts": visible.all(axis=-1).sum(axis=1).tolist(),
        "partial_support_diagnostic_count": int((masks["source_present"][indices]
                                                   & ~masks["value_available"][indices]).sum()),
        "partial_values_used": False, "excluded_u_stations": int(excluded_nodes.sum()),
        "source_clock_view": "fixed_UTC_minus_8_candidate", "source_clock_verified": False,
    }
    for item in (values, visible, speed):
        item.setflags(write=False)
    return DayView(values, visible, speed, diagnostics)


@dataclass(frozen=True)
class RealDataScope:
    """Only batch B is implemented; C needs a separate, verified release later."""

    config: dict[str, Any]
    manifest_sha256: str
    stations: tuple[str, ...]
    excluded: tuple[str, ...]
    dates: tuple[str, ...]
    release: dict[str, Any]
    shards: dict[str, dict[str, Any]]
    manifest_path: Path

    @classmethod
    def load(cls, path: Path, expected_sha256: str) -> RealDataScope:
        config = read_pinned_json({"path": str(path), "sha256": expected_sha256})
        if config.get("purpose") != "integration_smoke":
            raise ValueError("only integration_smoke is implemented; C/formal scopes remain closed")
        if (config.get("namespace") != "pilot-only" or config.get("throwaway") is not True
                or config.get("formal_evidence") is not False):
            raise ValueError("purpose requires pilot-only, throwaway, no formal evidence")
        bindings = config["bindings"]
        if bindings["rules"].get("sha256") != RULE_SHA:
            raise ValueError("rules must bind the exact user-confirmed SHA")
        rules = read_pinned_json(bindings["rules"])
        if rules["execution_state"].get("B_integration_scope_authorized") is not True:
            raise ValueError("B authorization is missing")
        for name, field in (("processing", "processing_dependency"),
                            ("eligibility", "eligibility_dependency")):
            if bindings[name]["sha256"] != rules[field]["sha256"]:
                raise ValueError(f"{name} dependency differs from approved rules")
            pinned_bytes(bindings[name])
        if bindings["erratum"]["sha256"] != ERRATUM_SHA:
            raise ValueError("five-mask erratum SHA is required")
        pinned_bytes(bindings["erratum"])
        release = read_pinned_json(bindings["release"])
        if (release.get("status") != "PASS_94_source_days_merged_and_audited"
                or release.get("target_days") != 90 or release.get("source_days") != 94):
            raise ValueError("B requires the audited Q1 release")
        if release["frozen_binding"]["manifest_sha256"] != bindings["processing"]["sha256"]:
            raise ValueError("release numeric contract mismatch")
        coverage = read_pinned_json(release["coverage_manifest"])
        read_pinned_json(release["final_audit"])
        u = read_pinned_json(bindings["permanent_u"])
        all_ids = canonical_stations(u["station_ids"])
        excluded = permanent_u(all_ids)
        if (tuple(u["excluded_station_ids"]) != excluded
                or u.get("rule_sha256") != RULE_SHA):
            raise ValueError("permanent U manifest differs from the frozen hash rule")
        if u.get("source_binding") is not None:
            upstream_u = read_pinned_json(u["source_binding"])
            if tuple(str(x["station_id"]) for x in upstream_u["stations"]) != all_ids:
                raise ValueError("permanent U source universe mismatch")
        stations = canonical_stations(config["station_ids"])
        if len(stations) > 30 or not set(stations).issubset(all_ids):
            raise ValueError("registered cohort must be known and at most 30 stations including U")
        if not set(stations).difference(excluded):
            raise ValueError("no legal K training station")
        year_axis = tuple((date(2019, 1, 1) + timedelta(days=i)).isoformat() for i in range(365))
        pool = tuple(x["date"] for x in coverage["days"] if x["role"] == "target_day")
        if pool != year_axis[:90]:
            raise ValueError("Q1 pool must be the declared first 90 calendar days")
        roles = calendar_roles(year_axis)
        if any(roles[x] != "train" for x in pool):
            raise ValueError("Q1 pool crosses the outer training boundary")
        dates = tuple(config["fit_dates"])
        _calendar(dates)
        if not set(dates).issubset(inner_fit_dates(pool)) or len(dates) > 14:
            raise ValueError("B fit must be <=14 registered days inside inner fit only")
        if config.get("channels") != list(CHANNELS) or config.get("units") != list(UNITS):
            raise ValueError("canonical channel order and unit contract mismatch")
        fit = config["fit"]
        if (type(fit["epochs"]) is not int or not 1 <= fit["epochs"] <= 25
                or fit["rank"] != 16 or fit["lr"] != .001 or fit["optimizer"] != "Adam"
                or fit["mask_period"] != 5 or fit["channel_floors"] != [1.0, .001]):
            raise ValueError("fit budget differs from the integration preregistration")
        if not config.get("code_bindings"):
            raise ValueError("runner code bindings are required before fitting")
        root = Path(__file__).resolve().parents[1]
        if {Path(x["path"]).resolve() for x in config["code_bindings"]} != {
            (root / name).resolve() for name in CODE_FILES
        }:
            raise ValueError("all runner and core code bindings are required")
        for item in config["code_bindings"]:
            pinned_bytes(item)
        if config.get("actual_graph") is not None:
            graph = read_pinned_json(config["actual_graph"])
            legal_order = tuple(x for x in stations if x not in excluded)
            if (tuple(str(x) for x in graph["station_order"]) != legal_order
                    or graph["rule_sha256"] != RULE_SHA
                    or graph["physics_audited"] is not False):
                raise ValueError("graph station order/rules/physics boundary mismatch")
        shards = {x["date"]: x for x in coverage["days"]}
        if any(not shards[d]["processed_and_audited"] for d in dates):
            raise ValueError("unprocessed shard")
        return cls(config, expected_sha256, stations, excluded, dates, release, shards, path.resolve())

    def read_day(self, day: str) -> DayView:
        if day not in self.dates:
            raise ValueError("reading outside the registered inner-fit dates is prohibited")
        shard = self.shards[day]
        folder = Path(self.release["output_root"]).resolve() / day
        marker = read_pinned_json({"path": str(folder / "complete.json"),
                                   "sha256": shard["complete_marker_sha256"]})
        if marker.get("status") != "complete" or marker["binding"]["date"] != day:
            raise ValueError("invalid completed shard marker")
        for name, value in self.release["frozen_binding"].items():
            if marker["binding"].get(name) != value:
                raise ValueError(f"shard binding changed: {name}")
        if marker["binding"]["source_sha256"] != shard["source_sha256"]:
            raise ValueError("source archive binding changed")
        output = next(x for x in marker["outputs"] if x["file"] == "arrays.npz")
        if output["sha256"] != shard["arrays_sha256"]:
            raise ValueError("array binding changed")
        payload = pinned_bytes({"path": str(folder / "arrays.npz"),
                                "sha256": shard["arrays_sha256"]})
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            keys = ("station_ids", "values", "source_present", "value_available", "scorable",
                    "quality_accepted", "coverage_available", "calendar_slot_exists",
                    "configuration_consensus", "unexpected_reported_lane", "conflicting_measurement",
                    "ambiguous_source_time", "n_present")
            arrays = {name: archive[name] for name in keys}
            view = validate_day(arrays, self.stations, self.excluded)
            # Source-value adversary runs before training, on the actual day mask.
            poisoned = dict(arrays)
            poisoned["values"] = arrays["values"].copy()
            ids = arrays["station_ids"]
            is_u = np.isin(ids, [int(x) for x in self.excluded])
            poisoned["values"][is_u] = np.inf
            other = validate_day(poisoned, self.stations, self.excluded)
            if (not np.array_equal(view.values, other.values, equal_nan=True)
                    or not np.array_equal(view.visible, other.visible)):
                raise RuntimeError("permanent U source poisoning changed legal training inputs")
            view.diagnostics["source_u_poison_invariant"] = True
            return view
