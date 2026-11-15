"""Selection-frozen local-linear occupancy loading trial for G80v5."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.o_spatial_loading import local_linear_loading
from scripts.loading_split_candidate import o_loading, q_residual, station_order
from scripts.run_transductive import load_case
from scripts.score_selection_three_arm import OUR_SCALES, profile_macro

OUT = ROOT / 'reviews/2026-09-28/g80v5_spatial_loading'
SEL = ROOT / 'reviews/2026-09-26/transductive/selection_42'
TEST = ROOT / 'reviews/2026-09-27/test21_reeval'
RIDGES = (0., 1., 4., 16., np.inf)
SEEDS = (20260905, 20260906, 20260907)


def source_rows(directory, order):
    rows = []
    for path in sorted(directory.glob('case_*.npz')):
        case = load_case(path)
        diag, base_o = o_loading(case, order, 4, .01, .1)
        q = q_residual(case, 4, .1, .1)
        control = case['pred_control']
        positions = np.array([order.index(d) for d in diag.donor_ids])
        target = order.index(str(case['target_station']))
        alternatives = {}
        for ridge in RIDGES:
            loading = (diag.loading if np.isinf(ridge) else local_linear_loading(
                positions, diag.per_donor, diag.weights,
                target_position=target, ridge=ridge))
            alternatives[str(ridge)] = diag.basis @ loading
        if not np.array_equal(alternatives[str(np.inf)], base_o):
            raise AssertionError('v3 baseline drift')
        rows.append({'id': str(case['episode_id']), 'station': str(case['target_station']),
                     'duration': int(case['duration_days']),
                     'control': control, 'truth': case['truth_control'] / OUR_SCALES,
                     'mask': case['scoring_control'].astype(bool),
                     'q': q, 'o': alternatives})
    print('sources', directory.name, len(rows), flush=True)
    return rows


def sensor(records, arm, metric, channel):
    grouped = {}
    for r in records:
        grouped.setdefault(r['station'], []).append(r[arm][metric][channel])
    return float(np.mean([np.mean(v) for v in grouped.values()]))


def score_rows(rows, ridge, *, save_dir=None):
    records = []
    for row in rows:
        baseline = np.stack((row['control'][..., 0] + row['q'],
                             row['control'][..., 1] + row['o'][str(np.inf)]), axis=-1)
        candidate = np.stack((row['control'][..., 0] + row['q'],
                              row['control'][..., 1] + row['o'][str(ridge)]), axis=-1)
        if not np.array_equal(baseline[..., 0], candidate[..., 0]):
            raise AssertionError('q changed')
        record = {'id': row['id'], 'station': row['station'],
                  'duration': row['duration']}
        for name, prediction in (('v3', baseline), ('v5', candidate)):
            record[name] = {'nrmse': profile_macro(prediction, row['truth'], row['mask']),
                            'mae': {ch: float(np.mean(np.abs(
                                prediction[..., j][row['mask'][..., j]]
                                - row['truth'][..., j][row['mask'][..., j]]
                            )) * OUR_SCALES[j])
                            for ch, j in (('q', 0), ('o', 1))}}
        records.append(record)
        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(save_dir / ('case_' + row['id'].replace('|', '_') + '.npz'),
                                episode_id=row['id'], station=row['station'],
                                prediction_normalised=candidate,
                                truth_normalised=row['truth'], scoring=row['mask'])
    aggregate = {name: {metric: {ch: sensor(records, name, metric, ch)
                                 for ch in ('q', 'o')}
                         for metric in ('nrmse', 'mae')}
                 for name in ('v3', 'v5')}
    by_duration = {}
    for duration in (7, 14):
        group = [r for r in records if r['duration'] == duration]
        if group:
            by_duration[str(duration)] = {'n': len(group),
                'v3_o': sensor(group, 'v3', 'nrmse', 'o'),
                'v5_o': sensor(group, 'v5', 'nrmse', 'o')}
    return {'n': len(records), 'sensor_macro': aggregate,
            'by_duration': by_duration,
            'wins_o_vs_v3': sum(r['v5']['nrmse']['o'] < r['v3']['nrmse']['o']
                                for r in records), 'per_case': records}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plan = ROOT / 'reviews/2026-09-28/G80V5_SPATIAL_LOADING_PLAN.md'
    file_hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in (plan, ROOT / 'models/o_spatial_loading.py', Path(__file__))}
    order = station_order()
    rows = source_rows(SEL, order)
    if len(rows) != 42:
        raise AssertionError('Selection roster size')
    base = score_rows(rows, np.inf)
    baseline_o = base['sensor_macro']['v3']
    trials = []
    for ridge in RIDGES:
        scored = score_rows(rows, ridge)
        metric = scored['sensor_macro']['v5']
        trials.append({'ridge': 'inf' if np.isinf(ridge) else ridge,
                       'o_nrmse': metric['nrmse']['o'],
                       'o_mae': metric['mae']['o'],
                       'wins_o_vs_v3': scored['wins_o_vs_v3']})
        print('Selection', trials[-1], flush=True)
    eligible = [t for t in trials if t['o_mae'] <= baseline_o['mae']['o']
                and t['o_nrmse'] < baseline_o['nrmse']['o']]
    selected = min(eligible, key=lambda t: (t['o_nrmse'],
                                            -float(t['ridge']))) if eligible else None
    choice = float(selected['ridge']) if selected else np.inf
    selection = {'hashes': file_hashes, 'baseline': baseline_o,
                 'trials': trials, 'chosen': selected,
                 'decision': 'candidate' if selected else 'retain_v3'}
    (OUT / 'selection_result.json').write_text(json.dumps(selection, indent=2),
                                                encoding='utf-8')
    print('selection frozen', selection['decision'], choice, flush=True)
    sel_ids = {r['id'] for r in rows}
    test = []
    for seed in SEEDS:
        cases = source_rows(TEST / f'cases_seed{seed}', order)
        if len(cases) != 21 or sel_ids & {r['id'] for r in cases}:
            raise AssertionError('Test roster')
        scored = score_rows(cases, choice, save_dir=OUT / f'cases_seed{seed}')
        test.append({'seed': seed, **scored})
        print('Test', seed, scored['sensor_macro'], flush=True)
    (OUT / 'results.json').write_text(json.dumps({'selection': selection,
                                                   'test': test}, indent=2),
                                       encoding='utf-8')


if __name__ == '__main__':
    main()
