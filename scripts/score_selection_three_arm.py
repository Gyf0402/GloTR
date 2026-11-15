"""One scorer for three arms on selection_42, in the G80 line's own normalisation.

Why this exists: the archived modified-RTTC-v7 Selection run stores raw
predictions but was normalised with scales [6264.0, 0.212854996], while the G80
line uses [6240.0, 0.2127663], and its reported aggregation could not be
reproduced from the archive (my episode-macro recomputation is ~1.2% below its
headline value under its own scales). Rather than compare across two uncertain
calibres, this script recomputes every arm from RAW predictions with one
normalisation and one aggregation.

Self-validation: the `full` arm's numbers produced here must match the official
evaluation's `retrained_arms_result.json` for the same queue; that is the anchor
that makes the archived RTTC number trustworthy in this pipeline.

Read-only w.r.t. every SHA-pinned file: inference reuses the trainer's frozen
queue/assembly/forward path by import.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.train_g80_v2_arm as arm_mod  # noqa: E402

CHANNELS = ('q', 'o')
OUR_SCALES = np.asarray([6240.0, 0.21276629999999888])
ARCHIVED_SCALES = np.asarray([6264.0, 0.212854996])


def profile_macro(pred_norm: np.ndarray, truth_norm: np.ndarray,
                  scoring: np.ndarray) -> dict:
    """Official reduction: per-day profile RMSE over slots, then mean over days."""
    out = {}
    for c, channel in enumerate(CHANNELS):
        active = scoring[..., c]
        count = active.sum(axis=-1)
        if not np.all(count > 0):
            raise ValueError('empty scoring profile')
        error = np.where(active, pred_norm[..., c] - truth_norm[..., c], 0.0)
        out[channel] = float(np.mean(np.sqrt(np.square(error).sum(axis=-1) / count)))
    return out


def sensor_macro(per_case: list[dict], key: str) -> dict:
    """Group cases by target station, average, then average over stations."""
    grouped: dict[str, list[dict]] = {}
    for case in per_case:
        grouped.setdefault(case['target_station'], []).append(case)
    return {channel: float(np.mean([np.mean([c[key][channel] for c in group])
                                    for group in grouped.values()]))
            for channel in CHANNELS}


def our_arm_predictions(output: Path, arm: str, queue_name: str) -> list[dict]:
    """Re-run inference for one arm; returns raw (physical) prediction blocks."""
    arm_mod.setup_torch()
    device = torch.device('cuda:0')
    scales = arm_mod.S3
    graph = arm_mod.load_json(ROOT / arm_mod.GRAPH_REL)
    order = tuple(str(x) for x in graph['station_order'])
    edges = {name: torch.tensor(value, dtype=torch.long)
             for name, value in graph['edge_index'].items()}
    basis, mu, core = arm_mod.load_models(scales)
    from models.sparse_global_tokens import global_token_context
    from models.arm_profiles import build_profile

    payload = torch.load(output / f'{arm}.pt', map_location='cpu',
                         weights_only=False)
    model = build_profile(arm, dropout=.1).requires_grad_(False).eval()
    model.load_state_dict(payload['model'], strict=True)
    model = model.to(device)

    queues = arm_mod.build_queues(order)
    queue = queues[queue_name]
    cases = []
    for row in queue['rows']:
        episode = queue['assemble'](row, lru_cache(maxsize=64)(
            lambda day: queue['reader'](day)))
        if np.any(episode.visible & episode.scoring):
            raise ValueError('score visible leak')
        duration = row['duration_days']
        target = episode.target_index
        base_rows = arm_mod.BUFFER_DAYS
        pred = arm_mod.prepare_item(episode, row, basis, mu, core, scales,
                                    edges, device)
        static_c, hops_c = arm_mod.context_hook(arm, pred)
        key_groups = arm_mod.arm_key_groups(arm)
        blocks = []
        with torch.no_grad():
            for day in range(base_rows, base_rows + duration):
                q, k, valid = global_token_context(
                    values=pred.values, visible=pred.visible,
                    base=pred.base[day - arm_mod.BUFFER_DAYS], static=static_c,
                    calendar=pred.calendar, target=pred.target, day=day,
                    signed_hops=hops_c, scales=scales, raw_values=True,
                    key_groups=key_groups)
                out, _ = model(q, k, valid,
                               pred.base[day - arm_mod.BUFFER_DAYS])
                blocks.append(out.reshape(arm_mod.SLOTS, 2))
        cases.append({
            'episode_id': row['episode_id'],
            'target_station': str(row['station_id']),
            'duration_days': duration,
            'prediction': torch.cat(blocks).reshape(duration, arm_mod.SLOTS,
                                                    2).cpu().numpy().astype(np.float64),
            'truth': (episode.reference[base_rows:base_rows + duration, target]
                      ).astype(np.float64),
            'scoring': episode.scoring[base_rows:base_rows + duration,
                                       target].astype(bool),
        })
        print(json.dumps({'phase': 'infer', 'arm': arm,
                          'episode': row['episode_id']}), flush=True)
    return cases


def archived_rttc_predictions() -> list[dict]:
    """Load the archived modified-RTTC-v7 Selection run (raw units)."""
    archive = ROOT / 'reviews/2026-09-21/rttc_repair/rttc_real_v7/selection'
    cases = []
    for path in sorted(archive.glob('*.npz')):
        with np.load(path, allow_pickle=True) as z:
            pred = z['prediction_raw'].astype(np.float64)
            truth_all = z['truth_raw'].astype(np.float64)
            scoring = z['scoring'].astype(bool)
            coords = z['coordinates'].astype(np.int64)
            order = [str(x) for x in z['station_order']]
            episode_id = str(z['episode_id'])
        gap = np.flatnonzero(scoring.any(axis=(1, 2, 3)))
        target = int(np.flatnonzero(scoring.any(axis=(0, 2, 3)))[0])
        first, last = int(gap[0]), int(gap[-1]) + 1
        block = scoring[first:last, target]
        keep = ((coords[:, 0] >= first) & (coords[:, 0] < last)
                & (coords[:, 1] == target))
        sel = coords[keep]
        truth = np.zeros_like(pred[first:last, target])
        truth[sel[:, 0] - first, sel[:, 2], sel[:, 3]] = truth_all[keep]
        cases.append({
            'episode_id': episode_id, 'target_station': order[target],
            'duration_days': last - first,
            'prediction': pred[first:last, target],
            'truth': truth, 'scoring': block})
    return cases


def evaluate(cases: list[dict], scales: np.ndarray, *,
             prediction_is_raw: bool) -> dict:
    """Score cases under one normalisation.

    ``prediction_is_raw`` marks the source: our own arms return predictions that
    ``prepare_item`` has ALREADY divided by the channel scales together with the
    reference, so they must not be divided again -- dividing twice is a units
    error that silently inflates every score. The archived RTTC run stores raw
    physical units, so it does need the division.
    """
    per_case = []
    for case in cases:
        scoring = case['scoring']
        pred_norm = (case['prediction'] / scales if prediction_is_raw
                     else case['prediction'])
        truth_norm = case['truth'] / scales
        macro = profile_macro(pred_norm, truth_norm, scoring)
        per_case.append({'episode_id': case['episode_id'],
                         'target_station': case['target_station'],
                         'duration_days': case['duration_days'],
                         'profile_macro_nrmse': macro})
    episode = {ch: float(np.mean([c['profile_macro_nrmse'][ch]
                                  for c in per_case])) for ch in CHANNELS}
    episode_by_duration = {}
    for duration in sorted({c['duration_days'] for c in per_case}):
        group = [c for c in per_case if c['duration_days'] == duration]
        episode_by_duration[str(duration)] = {
            'n': len(group),
            **{ch: float(np.mean([c['profile_macro_nrmse'][ch] for c in group]))
               for ch in CHANNELS}}
    return {'episode_macro': episode,
            'episode_macro_by_duration': episode_by_duration,
            'sensor_macro': sensor_macro(per_case, 'profile_macro_nrmse'),
            'per_case': per_case}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'reviews/2026-09-26/loss/dev/seed20260905')
    parser.add_argument('--arms', default='full')
    parser.add_argument('--queue', default='selection_42')
    parser.add_argument('--rttc', action='store_true', default=True)
    args = parser.parse_args()

    report = {'queue': args.queue, 'scales': list(OUR_SCALES), 'arms': {}}
    for arm in [a for a in args.arms.split(',') if a]:
        cases = our_arm_predictions(args.output, arm, args.queue)
        report['arms'][arm] = {
            'n_cases': len(cases),
            'source': str(args.output.relative_to(ROOT)),
            **evaluate(cases, OUR_SCALES, prediction_is_raw=False)}
        np.savez_compressed(
            ROOT / f'reviews/2026-09-26/transductive/raw_{arm}.npz',
            **{f'prediction_{i}': c['prediction'] for i, c in enumerate(cases)})

    if args.rttc:
        cases = archived_rttc_predictions()
        report['arms']['rttc_v7_archived'] = {
            'n_cases': len(cases),
            'source': 'reviews/2026-09-21/rttc_repair/rttc_real_v7/selection',
            **evaluate(cases, OUR_SCALES, prediction_is_raw=True)}
        report['arms']['rttc_v7_archived_in_its_own_scales'] = evaluate(
            cases, ARCHIVED_SCALES, prediction_is_raw=True)['episode_macro']

    out = ROOT / 'reviews/2026-09-26/transductive/unified_scores.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1),
                   encoding='utf-8')
    print('\n=== 统一口径（我们的归一化 [6240.0, 0.2127663]）===')
    for arm, entry in report['arms'].items():
        if 'episode_macro' not in entry:
            continue
        e = entry['episode_macro']
        s = entry['sensor_macro']
        print(f'{arm:<26} episode-macro  q={e["q"]:.6f} o={e["o"]:.6f} | '
              f'sensor-macro q={s["q"]:.6f} o={s["o"]:.6f}')
    print(f'\nwritten {out.relative_to(ROOT)}')
    print('sha256', hashlib.sha256(out.read_bytes()).hexdigest())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
