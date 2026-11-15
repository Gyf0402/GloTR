"""SHA-bound 2020 reader and Protocol-K episode assembler."""

from __future__ import annotations

import hashlib
import io
import json
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'reviews/2026-09-28/2020_blind19/plan.json'
SLOTS = 288
BUFFER = 3


class Frozen2020Reader:
    def __init__(self, *, plan_path: Path = PLAN):
        self.plan = json.loads(plan_path.read_text(encoding='utf-8'))
        if self.plan['status'] != 'frozen_before_any_2020_traffic_array_read':
            raise ValueError('2020 plan is not frozen')
        roster_info = self.plan['roster']
        roster_path = ROOT / roster_info['path']
        if hashlib.sha256(roster_path.read_bytes()).hexdigest() != roster_info['sha256']:
            raise ValueError('2020 roster SHA drift')
        self.roster = json.loads(roster_path.read_text(encoding='utf-8'))
        self.days = self.plan['2020_processed_days']
        self.bounds = (min(self.days), max(self.days))
        self.stations = tuple(str(s) for s in json.loads(
            (ROOT / self.plan['pins']['graph']['path']).read_text(encoding='utf-8'))
            ['station_order'])
        permanent = ROOT / 'data/pems/releases/partition_v1/permanent_u.json'
        self.universe = {str(r['station_id']): bool(r['permanent_U']) for r in
                         json.loads(permanent.read_text(encoding='utf-8'))['stations']}
        if len(self.stations) != 20 or any(self.universe.get(s, True) for s in self.stations):
            raise ValueError('graph station universe drift')
        self.access_log: list[dict] = []

    def __call__(self, day: str):
        date.fromisoformat(day)
        if day not in self.days:
            raise ValueError(f'day outside frozen 2020 whitelist: {day}')
        entry = self.days[day]
        path = Path(entry['path'])
        if not path.is_absolute():
            # corridor-cropped plan records the day directory relative to the repo
            path = ROOT / path
        blob = path.read_bytes()
        digest = hashlib.sha256(blob).hexdigest()
        if digest != entry['sha256'] or len(blob) != entry['bytes']:
            raise ValueError(f'2020 daily array SHA/length mismatch: {day}')
        from training.real_qo_loader import select_day
        with np.load(io.BytesIO(blob), allow_pickle=False) as archive:
            all_ids = archive['station_ids']
            if (all_ids.dtype != np.int32 or all_ids.ndim != 1 or
                    len(np.unique(all_ids)) != len(all_ids)):
                raise ValueError('2020 daily station ID vector invalid')
            # The 2020 product contains detector IDs absent from the frozen
            # 2019 U/K universe. Restrict to the registered 20 graph stations
            # before applying the unchanged select_day legality logic.
            chosen = np.isin(all_ids, np.asarray(self.stations, dtype=np.int32))
            subset = {}
            for key in archive.files:
                value = archive[key]
                subset[key] = (value[chosen] if value.ndim >= 1 and
                               value.shape[0] == len(all_ids) and
                               key not in ('coverage_available',
                                           'calendar_slot_exists') else value)
            ids, values, masks = select_day(subset, self.universe)
        self.access_log.append({'day': day, 'sha256': digest})
        return [(day, str(s), values[i], masks[i]) for i, s in enumerate(ids)
                if str(s) in self.stations]


def assemble_2020_episode(row, station_order, bounds, read_day):
    from training.current_release_student_data import CurrentEpisode

    if row['split'] != 'test2020' or row['protocol'] != 'K':
        raise ValueError('expected frozen 2020 Protocol-K row')
    target = str(row['station_id'])
    if target not in station_order or row['duration_days'] not in (7, 14):
        raise ValueError('target or duration outside graph protocol')
    duration = int(row['duration_days'])
    first = date.fromisoformat(row['start_day']) - timedelta(days=BUFFER)
    days = tuple((first + timedelta(days=k)).isoformat()
                 for k in range(duration + 2 * BUFFER))
    if days[0] < bounds[0] or days[-1] > bounds[1]:
        raise ValueError('episode leaves 2020 role band')
    if 'input_support' in row and (row['input_support']['start'] != days[0] or
                                   row['input_support']['end'] != days[-1]):
        raise ValueError('episode support differs from frozen roster')
    n = len(station_order)
    lookup = {s: k for k, s in enumerate(station_order)}
    reference = np.zeros((len(days), n, SLOTS, 2), dtype=np.float64)
    visible = np.zeros(reference.shape, dtype=bool)
    for d, day in enumerate(days):
        seen = set()
        for label, station, values, mask in read_day(day):
            if label != day or station in seen:
                raise ValueError('duplicate station or incorrect date label')
            seen.add(station)
            if station not in lookup:
                continue
            if values.shape != (SLOTS, 2) or mask.shape != values.shape or mask.dtype != bool:
                raise ValueError('daily station shape/mask invalid')
            if not np.isfinite(values[mask]).all():
                raise ValueError('nonfinite legal value')
            i = lookup[station]
            reference[d, i] = np.where(mask, values, 0.)
            visible[d, i] = mask
        if int(visible[d, lookup[target]].all(axis=-1).sum()) < 274:
            raise ValueError(f'target day fails frozen ≥274 joint-slot rule: {day}')
    scoring = np.zeros_like(visible)
    gap = slice(BUFFER, BUFFER + duration)
    scoring[gap, lookup[target]] = visible[gap, lookup[target]]
    visible[gap, lookup[target]] = False
    values = np.where(visible, reference, 0.)
    if np.any(visible & scoring):
        raise AssertionError('scored target truth visible in episode inputs')
    return CurrentEpisode(row['episode_id'], row['mask_id'], 'test2020',
                          days, tuple(station_order), lookup[target], values,
                          visible, reference, scoring)


@lru_cache(maxsize=1)
def frozen_2020_static_features() -> np.ndarray:
    """Transfer unchanged last-2019 station attributes to supported Jan-Feb 2020."""
    from training.qo_attributes import AnnualAttributes
    plan = json.loads(PLAN.read_text(encoding='utf-8'))
    registration = json.loads((ROOT / plan['pins']['attribute_registration']['path'])
                              .read_text(encoding='utf-8'))
    attribute = AnnualAttributes.from_files(
        ROOT / 'data/pems/releases/partition_v1',
        manifest_sha256=registration['config']['attribute_manifest_sha256'])
    graph = json.loads((ROOT / plan['pins']['graph']['path']).read_text(encoding='utf-8'))
    order = tuple(str(s) for s in graph['station_order'])
    values = np.asarray([attribute.lookup('2019-12-31', s)['features'] for s in order],
                        dtype=np.float32)
    if values.shape != (20, 5) or not np.isfinite(values).all():
        raise ValueError('invalid supported 2020 static feature table')
    return values
