"""Selection-first G80v6 pre-gap occupancy mean calibration trial."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.o_pseudo_calibration import calibrate_occupancy
from scripts.run_o_spatial_loading import (OUT as V5_OUT, SEL, SEEDS, TEST,
                                            sensor, source_rows)
from scripts.run_transductive import load_case
from scripts.loading_split_candidate import station_order
from scripts.score_selection_three_arm import OUR_SCALES, profile_macro

OUT = ROOT / 'reviews/2026-09-28/g80v6_pseudo_mean'
ALPHAS = (0., .1, .25, .5, 1.)
SPATIAL_RIDGE = 16.


def attach_pseudo(rows, directory):
    case_by_id = {str(c['episode_id']): c for c in (
        load_case(path) for path in sorted(directory.glob('case_*.npz')))}
    for row in rows:
        case = case_by_id[row['id']]
        use = bool(case['pseudo_usable'])
        mask = case['pseudo_observed'][..., 1]
        row['pseudo_mean'] = (float(np.mean(case['pseudo_error'][..., 1][mask]))
                              if use and mask.any() else None)
    return rows


def evaluate(rows, alpha, *, save_dir=None):
    records = []
    for row in rows:
        correction = row['o'][str(SPATIAL_RIDGE)]
        v5 = np.stack((row['control'][..., 0] + row['q'],
                       row['control'][..., 1] + correction), axis=-1)
        effective = alpha if row['pseudo_mean'] is not None else 0.
        v6 = calibrate_occupancy(v5, row['pseudo_mean'] or 0.,
                                 float(np.mean(correction)), effective)
        if not np.array_equal(v5[..., 0], v6[..., 0]):
            raise AssertionError('q moved')
        entry = {'id': row['id'], 'station': row['station'],
                 'duration': row['duration'], 'pseudo_usable': effective > 0}
        for name, prediction in (('v5', v5), ('v6', v6)):
            entry[name] = {
                'nrmse': profile_macro(prediction, row['truth'], row['mask']),
                'mae': {ch: float(np.mean(np.abs(
                    prediction[..., j][row['mask'][..., j]]
                    - row['truth'][..., j][row['mask'][..., j]]
                )) * OUR_SCALES[j]) for ch, j in (('q', 0), ('o', 1))}}
        records.append(entry)
        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(save_dir / ('case_' + row['id'].replace('|', '_') + '.npz'),
                                episode_id=row['id'], station=row['station'],
                                prediction_normalised=v6,
                                truth_normalised=row['truth'], scoring=row['mask'])
    aggregate = {name: {metric: {ch: sensor(records, name, metric, ch)
                                 for ch in ('q', 'o')}
                         for metric in ('nrmse', 'mae')}
                 for name in ('v5', 'v6')}
    by_duration = {}
    for duration in (7, 14):
        group = [r for r in records if r['duration'] == duration]
        if group:
            by_duration[str(duration)] = {
                'n': len(group), 'usable': sum(r['pseudo_usable'] for r in group),
                'v5_o': sensor(group, 'v5', 'nrmse', 'o'),
                'v6_o': sensor(group, 'v6', 'nrmse', 'o')}
    return {'n': len(records), 'sensor_macro': aggregate,
            'by_duration': by_duration,
            'wins_o_vs_v5': sum(r['v6']['nrmse']['o'] < r['v5']['nrmse']['o']
                                for r in records), 'per_case': records}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plan = ROOT / 'reviews/2026-09-28/G80V6_PSEUDO_MEAN_PLAN.md'
    hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in (plan, ROOT / 'models/o_pseudo_calibration.py', Path(__file__),
                           ROOT / 'reviews/2026-09-28/g80v5_spatial_loading/selection_result.json')}
    order = station_order()
    rows = attach_pseudo(source_rows(SEL, order), SEL)
    if len(rows) != 42:
        raise AssertionError('Selection size')
    baseline = evaluate(rows, 0.)['sensor_macro']['v5']
    trials = []
    for alpha in ALPHAS:
        metric = evaluate(rows, alpha)['sensor_macro']['v6']
        trials.append({'alpha': alpha, 'o_nrmse': metric['nrmse']['o'],
                       'o_mae': metric['mae']['o']})
        print('Selection', trials[-1], flush=True)
    eligible = [r for r in trials if r['alpha'] > 0
                and r['o_nrmse'] < baseline['nrmse']['o']
                and r['o_mae'] <= baseline['mae']['o']]
    chosen = min(eligible, key=lambda r: (r['o_nrmse'], r['alpha'])) if eligible else None
    alpha = chosen['alpha'] if chosen else 0.
    selection = {'hashes': hashes, 'baseline': baseline, 'trials': trials,
                 'chosen': chosen, 'decision': 'candidate' if chosen else 'retain_v5'}
    (OUT / 'selection_result.json').write_text(json.dumps(selection, indent=2), encoding='utf-8')
    print('selection frozen', selection['decision'], alpha, flush=True)
    test = []
    selected_ids = {r['id'] for r in rows}
    for seed in SEEDS:
        directory = TEST / f'cases_seed{seed}'
        cases = attach_pseudo(source_rows(directory, order), directory)
        if len(cases) != 21 or selected_ids & {r['id'] for r in cases}:
            raise AssertionError('Test roster')
        report = evaluate(cases, alpha, save_dir=OUT / f'cases_seed{seed}')
        test.append({'seed': seed, **report})
        print('Test', seed, report['sensor_macro'], flush=True)
    (OUT / 'results.json').write_text(json.dumps({'selection': selection,
                                                   'test': test}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
