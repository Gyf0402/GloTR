"""Fixed G80/v3/v6/o-tree inference on the frozen 2020 episodes; no scoring."""

from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np

from models.o_pseudo_calibration import calibrate_occupancy
from models.o_tree_features import append_tree_features
from scripts import train_g80_v2_arm as arm
from scripts import run_transductive as rt
from scripts.loading_split_candidate import station_order
from scripts.reader_2020_blind import (Frozen2020Reader, assemble_2020_episode,
                                       frozen_2020_static_features)
from scripts.run_dual_channel_probe import episode_features
from scripts.run_o_pseudo_calibration import attach_pseudo
from scripts.run_o_spatial_loading import source_rows

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reviews/2026-09-28/2020_blind19'
SEEDS = (20260905, 20260906, 20260907)


class TransferredStatic:
    def __init__(self, reader: Frozen2020Reader):
        self.days = set(reader.days)
        self.features = {station: frozen_2020_static_features()[i]
                         for i, station in enumerate(reader.stations)}

    def lookup(self, day: str, station: str):
        if day not in self.days or station not in self.features:
            raise ValueError('static lookup leaves frozen 2020 graph/band')
        return {'features': tuple(float(x) for x in self.features[station])}


def engine_and_queue(seed: int):
    reader = Frozen2020Reader()
    control = ROOT / reader.plan['pins'][f'g80_seed{str(seed)[-2:]}']['path']
    engine = rt.Engine(control)
    arm.ATTRIBUTES = TransferredStatic(reader)
    queue = {'rows': reader.roster['episodes'], 'reader': reader,
             'assemble': lambda row, read: assemble_2020_episode(
                 row, reader.stations, reader.bounds, read)}
    engine.queues = {'test2020': queue}
    engine.band_reader = lambda q: engine.reader(q)
    return engine, queue, reader


def cache_cases(seed: int) -> None:
    engine, queue, reader = engine_and_queue(seed)
    availability = json.loads((OUT / 'pseudo_availability.json').read_text(encoding='utf-8'))
    placement = {('test2020', r['episode_id']): r for r in availability['rows']}
    folder = OUT / f'field_seed{seed}'
    folder.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(queue['rows'], 1):
        name = row['episode_id'].replace('|', '_')
        path = folder / f'case_{name}.npz'
        if path.exists():
            print(f'reuse {index}/19 {name}', flush=True)
            continue
        case = rt.build_case(engine, 'test2020', row, availability=placement)
        if case['leak_hidden_cells'] != 0 or case['pseudo_masked_cells'] != 0:
            raise AssertionError('target gap or pseudo truth leaked')
        rt.save_case(case, path)
        print(f'field {index}/19 seed={seed} {name} '
              f'pseudo={bool(case["pseudo_usable"])} '
              f'donor_failures={case["donor_failures"]}', flush=True)
    (folder / 'access.json').write_text(json.dumps(reader.access_log, indent=2),
                                         encoding='utf-8')


def save_prediction(arm_name: str, seed: int, identity: str, prediction) -> None:
    folder = OUT / 'predictions' / arm_name / str(seed)
    folder.mkdir(parents=True, exist_ok=True)
    if not np.isfinite(prediction).all():
        raise ValueError('nonfinite G80-family output')
    np.savez_compressed(folder / ('case_' + identity.replace('|', '_') + '.npz'),
                        episode_id=identity, prediction_norm=prediction)


def predict_family(seed: int) -> None:
    engine, queue, reader = engine_and_queue(seed)
    folder = OUT / f'field_seed{seed}'
    cases = list(folder.glob('case_*.npz'))
    if len(cases) != 19:
        raise ValueError('field cache incomplete')
    rows = attach_pseudo(source_rows(folder, station_order()), folder)
    if len(rows) != 19:
        raise ValueError('source row count drift')
    tree = joblib.load(OUT.parent / 'o_tree_probe/model_train58.joblib')
    by_id = {r['episode_id']: r for r in queue['rows']}

    @lru_cache(maxsize=60)
    def read(day):
        return reader(day)

    for index, r in enumerate(rows, 1):
        identity = r['id']
        control = r['control']
        correction = r['o'][str(16.)]
        v3 = np.stack((control[..., 0] + r['q'],
                       control[..., 1] + r['o'][str(np.inf)]), axis=-1)
        v5 = np.stack((control[..., 0] + r['q'],
                       control[..., 1] + correction), axis=-1)
        v6 = calibrate_occupancy(v5, r['pseudo_mean'] or 0.,
                                 float(np.mean(correction)),
                                 .25 if r['pseudo_mean'] is not None else 0.)
        if not np.array_equal(v3[..., 0], v6[..., 0]):
            raise AssertionError('q drift between v3 and v6')
        if seed == 20260905:
            save_prediction('G80_frozen_seed05', seed, identity, control)
        save_prediction('G80v3', seed, identity, v3)
        save_prediction('G80v6', seed, identity, v6)

        row = by_id[identity]
        episode = queue['assemble'](row, read)
        item = engine.prepare(episode, row)
        features33, _, _ = episode_features(item, int(item.target), row['start_day'])
        gap_visible = episode.visible[3:3 + row['duration_days']]
        features45 = append_tree_features(features33, v6, control, gap_visible,
                                          target=int(item.target))
        delta = np.clip(tree.predict(features45.reshape(-1, 45)), -.5, .5)
        candidate = v6.copy()
        candidate[..., 1] += delta.reshape(candidate.shape[:2])
        if not np.array_equal(candidate[..., 0], v6[..., 0]):
            raise AssertionError('tree moved q')
        save_prediction('G80v6_plus_o_tree', seed, identity, candidate)
        print(f'family {index}/19 seed={seed} {identity}', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('cache', 'predict'))
    parser.add_argument('--seed', type=int, required=True, choices=SEEDS)
    args = parser.parse_args()
    if args.phase == 'cache':
        cache_cases(args.seed)
    else:
        predict_family(args.seed)


if __name__ == '__main__':
    main()
