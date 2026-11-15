"""Candidate q/o views; whole-day hard exclusion matches the annual census."""

import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from training.real_qo_release import load_qo_release

Profile = tuple[str, str, NDArray[np.float64], NDArray[np.bool_]]


def select_day(
    arrays: Mapping[str, NDArray[Any]], universe: Mapping[str, bool], *, min_slots: int = 274,
) -> tuple[NDArray[Any], NDArray[np.float64], NDArray[np.bool_]]:
    """Select joint q/o profiles with whole-day hard checks before statistics."""
    if type(min_slots) is not int or not 1 <= min_slots <= 288:
        raise ValueError('min_slots must be an integer in [1, 288]')
    ids = arrays['station_ids']
    if ids.ndim != 1 or ids.dtype != np.int32 or len(np.unique(ids)) != len(ids):
        raise ValueError('station IDs must be a unique int32 vector')
    if not set(ids.astype(str)).issubset(universe):
        raise ValueError('station outside registered U/K universe')
    n = len(ids)
    values = arrays['values']
    if values.shape != (n, 288, 3) or values.dtype != np.float64:
        raise ValueError('values must be float64 [N,288,3]')
    for key in ('source_present', 'value_available', 'quality_accepted', 'scorable'):
        if arrays[key].shape != values.shape or arrays[key].dtype != np.bool_:
            raise ValueError(f'{key} must be bool [N,288,3]')
    if not np.array_equal(arrays['scorable'],
                          arrays['value_available'] & arrays['quality_accepted']):
        raise ValueError('scorable identity failed')
    if np.any(arrays['value_available'] & ~arrays['source_present']):
        raise ValueError('value availability exceeds source presence')
    hard = arrays['configuration_consensus']
    if hard.shape != (n,) or hard.dtype != np.bool_:
        raise ValueError('configuration_consensus must be bool [N]')
    hard = hard.copy()
    for key in ('unexpected_reported_lane', 'conflicting_measurement', 'ambiguous_source_time'):
        if arrays[key].shape != (n, 288) or arrays[key].dtype != np.bool_:
            raise ValueError(f'{key} must be bool [N,288]')
        hard &= ~arrays[key].any(axis=1)
    for key in ('coverage_available', 'calendar_slot_exists'):
        if arrays[key].shape != (288,) or arrays[key].dtype != np.bool_:
            raise ValueError(f'{key} must be bool [288]')
        hard &= arrays[key].all()
    legal = arrays['scorable'][..., 0] & arrays['scorable'][..., 2]
    keep = hard & (legal.sum(axis=1) >= min_slots)
    keep &= np.asarray([not universe[str(i)] for i in ids], dtype=bool)
    mask = np.repeat(legal[keep, :, None], 2, axis=2)
    selected = values[keep][..., [0, 2]]
    if not np.isfinite(selected[mask]).all():
        raise ValueError('legal q/o values must be finite')
    return ids[keep], np.where(mask, selected, np.nan), mask


def iter_role(binding_path: Path, role: str, *, min_slots: int = 274) -> Iterator[Profile]:
    """Stream one day at a time; this does not authorize model execution."""
    rel = load_qo_release(binding_path)
    a, b = rel.roles[role]
    root = Path(rel.output_root)
    entries = json.loads(Path(rel.u_manifest).read_text())['stations']
    universe = {str(x['station_id']): x['permanent_U'] for x in entries}
    if len(universe) != len(entries) or any(type(x) is not bool for x in universe.values()):
        raise ValueError('U universe must be unique with boolean memberships')
    for path in sorted(x for x in root.iterdir() if x.is_dir() and a <= x.name <= b):
        with np.load(path / 'arrays.npz', allow_pickle=False) as archive:
            arrays = {k: archive[k] for k in (
                'station_ids', 'values', 'scorable', 'source_present', 'value_available',
                'quality_accepted', 'configuration_consensus', 'unexpected_reported_lane',
                'conflicting_measurement', 'ambiguous_source_time', 'coverage_available',
                'calendar_slot_exists',
            )}
        ids, values, masks = select_day(arrays, universe, min_slots=min_slots)
        for i, station in enumerate(ids):
            yield path.name, str(station), values[i], masks[i]


def load_role(binding_path: Path, role: str, *, min_slots: int = 274) -> list[Profile]:
    """Compatibility materialization; prefer iter_role for annual inputs."""
    return list(iter_role(binding_path, role, min_slots=min_slots))
