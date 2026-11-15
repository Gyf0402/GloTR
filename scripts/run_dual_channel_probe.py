"""Dual-channel identifiability probe: how much of the required correction is
predictable from inference-legal information only?

Zero training. Does not modify any SHA-pinned file; imports the pinned training
script read-only to reuse its frozen data-preparation path.

See reviews/2026-09-26/DUAL_CHANNEL_IDENTIFIABILITY_PREREGISTRATION.md for the
frozen feature list, statistics and gates. Legality is enforced by construction:
every feature is built from values/visible of the *neighbouring* nodes, the
target station's buffer days, the frozen base, and calendar/geometry -- never
from the target station's values inside the gap.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.train_g80_v2_arm as arm_mod  # noqa: E402  (read-only import)

OUT = ROOT / 'reviews/2026-09-26/dual_channel_probe'
ARM = 'full'
SEED = 20260905
CHECKPOINT = ROOT / 'reviews/2026-09-26/g80_v2_round2/stability/seed20260905/full.pt'
BUFFER = arm_mod.BUFFER_DAYS
SLOTS = arm_mod.SLOTS
CHANNELS = arm_mod.CHANNELS
OFFSETS = (-2, -1, 1, 2)
RIDGE_ALPHAS = (1e-2, 1e-1, 1.0, 1e1, 1e2)
GBT_KWARGS = dict(max_iter=300, learning_rate=0.1, max_leaf_nodes=31,
                  min_samples_leaf=200, early_stopping=False,
                  random_state=20260926)

GROUPS = ('neighbors', 'neighbors+selfclim', 'all')


def feature_names() -> list[str]:
    names = ['selfclim_q', 'selfclim_o']
    for k in OFFSETS:
        for ch in CHANNELS:
            names.append(f'lvl_{k:+d}_{ch}')
            names.append(f'dev_{k:+d}_{ch}')
    for ch in CHANNELS:
        names += [f'agg_lvl_mean_{ch}', f'agg_lvl_std_{ch}',
                  f'agg_dev_mean_{ch}', f'agg_dev_std_{ch}']
    names += ['base_q', 'base_o', 'rel_day', 'rel_delta', 'duration',
              'dow_sin', 'dow_cos']
    return names


def group_mask(group: str) -> np.ndarray:
    names = feature_names()
    if group == 'neighbors':
        keep = [n.startswith(('lvl_', 'dev_', 'agg_')) for n in names]
    elif group == 'neighbors+selfclim':
        keep = [n.startswith(('lvl_', 'dev_', 'agg_', 'selfclim_')) for n in names]
    else:
        keep = [True] * len(names)
    return np.asarray(keep)


def episode_features(item, target: int, start_day: str) -> tuple[np.ndarray, ...]:
    """Return (features (D,288,F), required (D,288,2), scored (D,288,2))."""
    values = item.values.detach().cpu().numpy()
    visible = item.visible.detach().cpu().numpy()
    base = item.base.detach().cpu().numpy()
    reference = item.reference.detach().cpu().numpy()
    scoring = item.scoring.detach().cpu().numpy()
    duration = item.duration
    days_all, n_nodes, _, n_ch = values.shape
    buf = [0, 1, 2, duration + 3, duration + 4, duration + 5]

    # buffer-day climatology per (node, slot, channel), masked and normalised
    num = (values[buf] * visible[buf]).sum(axis=0)
    den = visible[buf].sum(axis=0)
    clim = np.divide(num, np.maximum(den, 1), out=np.zeros_like(num),
                     where=den > 0)

    gap = slice(BUFFER, BUFFER + duration)
    vals = values[gap]                      # (D, N, 288, C)
    vis = visible[gap]
    vals = np.where(vis, vals, 0.0)

    cols: list[np.ndarray] = []
    for ch_i in range(n_ch):
        cols.append(np.broadcast_to(clim[target, :, ch_i], (duration, SLOTS)))
    for k in OFFSETS:
        node = target + k
        if 0 <= node < n_nodes:
            for ch_i in range(n_ch):
                cols.append(np.where(vis[:, node, :, ch_i], vals[:, node, :, ch_i], 0.0))
                cols.append(np.where(vis[:, node, :, ch_i],
                                     vals[:, node, :, ch_i] - clim[node, :, ch_i], 0.0))
        else:
            for ch_i in range(n_ch):
                cols.append(np.zeros((duration, SLOTS), dtype=np.float32))
                cols.append(np.zeros((duration, SLOTS), dtype=np.float32))

    others = [v for v in range(n_nodes) if v != target]
    for ch_i in range(n_ch):
        # np.take keeps the result on the requested axis; mixing a slice with an
        # advanced index would move the indexed axis to the front instead.
        sel = np.take(vis[..., ch_i], others, axis=1)     # (D, N-1, 288)
        lev = np.take(vals[..., ch_i], others, axis=1)
        dev = lev - np.take(clim[..., ch_i], others, axis=0)[None]
        safe = np.maximum(sel.sum(axis=1), 1)
        m_lev = (lev * sel).sum(axis=1) / safe
        m_dev = (dev * sel).sum(axis=1) / safe
        sd_lev = np.sqrt(np.maximum(((lev - m_lev[:, None, :]) ** 2 * sel
                                     ).sum(axis=1) / safe, 0.0))
        sd_dev = np.sqrt(np.maximum(((dev - m_dev[:, None, :]) ** 2 * sel
                                     ).sum(axis=1) / safe, 0.0))
        cols += [m_lev, sd_lev, m_dev, sd_dev]

    cols.append(base[..., 0])
    cols.append(base[..., 1])
    d_idx = np.arange(duration, dtype=np.float32)
    cols.append(np.broadcast_to((d_idx / duration)[:, None], (duration, SLOTS)))
    delta = np.minimum(d_idx, duration - 1 - d_idx)
    cols.append(np.broadcast_to((delta / max(duration - 1, 1))[:, None],
                                 (duration, SLOTS)))
    cols.append(np.full((duration, SLOTS), float(duration), dtype=np.float32))
    start = date.fromisoformat(start_day)
    dow = np.array([(start.toordinal() + d) % 7 for d in range(duration)],
                   dtype=np.float32)
    cols.append(np.broadcast_to(np.sin(2 * np.pi * dow / 7)[:, None], (duration, SLOTS)))
    cols.append(np.broadcast_to(np.cos(2 * np.pi * dow / 7)[:, None], (duration, SLOTS)))

    feats = np.stack(cols, axis=-1).astype(np.float32)
    required = (reference - base).astype(np.float32)
    scored = scoring.astype(bool)
    assert feats.shape[-1] == len(feature_names()), (feats.shape, len(feature_names()))
    return feats, required, scored


def main() -> int:
    from models.sparse_global_tokens import global_token_context
    from training.current_release_student_data import (VerifiedRoleReader,
                                                       assemble_episode)
    from scripts.fit_sparse_neighbor_context import signed_chain_hops

    arm_mod.setup_torch()
    device = torch.device('cuda:0')
    split = arm_mod.load_json(ROOT / arm_mod.SPLIT_REL)
    stage3 = arm_mod.load_json(ROOT / arm_mod.STAGE3_RESULT)
    stage3_plan = arm_mod.load_json(ROOT / arm_mod.STAGE3_PLAN)
    scales = tuple(float(x) for x in stage3['scales'])
    graph = arm_mod.load_json(ROOT / arm_mod.GRAPH_REL)
    order = tuple(str(x) for x in graph['station_order'])
    edges = {name: torch.tensor(value, dtype=torch.long)
             for name, value in graph['edge_index'].items()}
    signed_chain_hops(edges, target=0, nodes=len(order))
    basis, mu, core = arm_mod.load_models(scales)
    roster = {row['episode_id']: row for row in
              arm_mod.load_json(ROOT / arm_mod.ROSTER_REL)['episodes']}
    reader = VerifiedRoleReader(ROOT, ROOT / arm_mod.RELEASE_REL,
                               arm_mod.RELEASE_SHA, 'train', order)
    from functools import lru_cache

    @lru_cache(maxsize=240)
    def read(day):
        return reader(day)

    fit_days = set(stage3_plan['fit_days'])
    fit_ids = sorted(split['fit_ids'])
    print(json.dumps({'phase': 'prepare_fit', 'episodes': len(fit_ids)}), flush=True)
    items, targets, starts = [], [], []
    for identity in fit_ids:
        row = roster[identity]
        episode = assemble_episode(row, order, reader.bounds, read)
        if any(day not in fit_days for day in episode.dates):
            raise ValueError(f'episode crosses fit boundary: {identity}')
        item = arm_mod.prepare_item(episode, row, basis, mu, core, scales,
                                    edges, device)
        items.append(item)
        targets.append(int(item.target))
        starts.append(row['start_day'])

    torch.manual_seed(SEED)
    from models.arm_profiles import build_profile, initial_state

    source = torch.load(ROOT / arm_mod.G40_CK, map_location='cpu', weights_only=False)
    model = build_profile(ARM, dropout=.1).to(device)
    model.load_state_dict(initial_state(ARM, source['model']), strict=True)
    model.load_state_dict(torch.load(CHECKPOINT, map_location='cpu',
                                     weights_only=False)['model'], strict=True)
    model.requires_grad_(False).eval()
    key_groups = arm_mod.arm_key_groups(ARM)

    feats, required, scored, corr, ep_id, dur_col = [], [], [], [], [], []
    # rows are CELLS, not (cell, channel): the scored mask is per channel, so it
    # is stored alongside and applied inside each channel's probe separately.
    for idx, item in enumerate(items):
        f, r, sc = episode_features(item, targets[idx], starts[idx])
        with torch.no_grad():
            contexts = []
            for d in range(item.duration):
                static_c, hops_c = arm_mod.context_hook(ARM, item)
                common = dict(values=item.values, visible=item.visible,
                              base=item.base[d], static=static_c,
                              calendar=item.calendar, target=item.target,
                              day=d + BUFFER, signed_hops=hops_c, scales=scales)
                q, k, valid = global_token_context(**common, raw_values=True,
                                                   key_groups=key_groups)
                contexts.append((q, k, valid, item.base[d]))
            query = torch.cat([c[0] for c in contexts])
            keys = torch.cat([c[1] for c in contexts])
            valid = torch.cat([c[2] for c in contexts])
            base_cat = torch.cat([c[3] for c in contexts])
            _, correction = model(query, keys, valid, base_cat)
            a = correction.reshape(item.duration, SLOTS, 2).cpu().numpy()
        n_cells = item.duration * SLOTS
        feats.append(f.reshape(n_cells, -1))
        required.append(r.reshape(n_cells, 2))
        corr.append(a.reshape(n_cells, 2))
        scored.append(sc.reshape(n_cells, 2))
        ep_id.append(np.full(n_cells, idx, dtype=np.int32))
        dur_col.append(np.full(n_cells, float(item.duration), dtype=np.float32))
        if (idx + 1) % 50 == 0:
            print(json.dumps({'phase': 'infer', 'done': idx + 1}), flush=True)

    X = np.concatenate(feats)
    R = np.concatenate(required)
    A = np.concatenate(corr)
    S = np.concatenate(scored)
    E = np.concatenate(ep_id)
    np.savez_compressed(OUT / 'probe_matrix.npz', X=X, R=R, A=A, S=S, E=E)
    print(json.dumps({'phase': 'matrix', 'rows': int(X.shape[0]),
                      'features': int(X.shape[1])}), flush=True)

    names = feature_names()
    fold_a = np.isin(E, [i for i in range(len(items)) if i % 2 == 0])
    report = {'rows': int(X.shape[0]),
              'rows_per_channel': {ch: int(S[:, i].sum())
                                   for i, ch in enumerate(CHANNELS)},
              'episodes': len(items),
              'features': names, 'groups': {}, 'per_channel': {}}

    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GridSearchCV
    from sklearn.preprocessing import StandardScaler

    for c_i, channel in enumerate(CHANNELS):
        roi = S[:, c_i]
        Xc = X[roi]
        y = R[roi, c_i]
        a_c = A[roi, c_i]
        fold_c = fold_a[roi]
        rms_none = float(np.sqrt(np.mean(y ** 2)))
        rms_g80 = float(np.sqrt(np.mean((y - a_c) ** 2)))
        entry = {'rms_none': rms_none, 'rms_g80': rms_g80, 'groups': {}}
        for group in GROUPS:
            keep = group_mask(group)
            Xg = Xc[:, keep]
            out = {}
            for direction, (tr, te) in (('A_to_B', (fold_c, ~fold_c)),
                                        ('B_to_A', (~fold_c, fold_c))):
                scaler = StandardScaler().fit(Xg[tr])
                grid = GridSearchCV(Ridge(), {'alpha': list(RIDGE_ALPHAS)},
                                    cv=5, scoring='neg_root_mean_squared_error')
                grid.fit(scaler.transform(Xg[tr]), y[tr])
                pred = grid.best_estimator_.predict(scaler.transform(Xg[te]))
                rms = float(np.sqrt(np.mean((y[te] - pred) ** 2)))
                base_te = float(np.sqrt(np.mean((y[te] - a_c[te]) ** 2)))
                out[direction] = {
                    'alpha': float(grid.best_params_['alpha']),
                    'rms_probe': rms, 'rms_g80_fold': base_te,
                    'gain_over_g80': float(1 - rms / base_te)}
            out['gain_over_g80_mean'] = float(np.mean(
                [out['A_to_B']['gain_over_g80'], out['B_to_A']['gain_over_g80']]))
            out['rms_probe_mean'] = float(np.mean(
                [out['A_to_B']['rms_probe'], out['B_to_A']['rms_probe']]))
            # constant baseline (train-fold mean)
            tr, te = fold_c, ~fold_c
            const = float(np.sqrt(np.mean((y[te] - y[tr].mean()) ** 2)))
            out['rms_constant_baseline'] = const
            entry['groups'][group] = out

        # generous probe: gradient boosting on the full feature set
        from sklearn.ensemble import HistGradientBoostingRegressor
        keep = group_mask('all')
        Xg = Xc[:, keep]
        gb = {}
        for direction, (tr, te) in (('A_to_B', (fold_c, ~fold_c)),
                                    ('B_to_A', (~fold_c, fold_c))):
            est = HistGradientBoostingRegressor(**GBT_KWARGS).fit(Xg[tr], y[tr])
            pred = est.predict(Xg[te])
            rms = float(np.sqrt(np.mean((y[te] - pred) ** 2)))
            base_te = float(np.sqrt(np.mean((y[te] - a_c[te]) ** 2)))
            gb[direction] = {'rms_probe': rms, 'rms_g80_fold': base_te,
                             'gain_over_g80': float(1 - rms / base_te)}
        gb['gain_over_g80_mean'] = float(np.mean(
            [gb['A_to_B']['gain_over_g80'], gb['B_to_A']['gain_over_g80']]))
        gb['rms_probe_mean'] = float(np.mean(
            [gb['A_to_B']['rms_probe'], gb['B_to_A']['rms_probe']]))
        entry['gradient_boosting_all_features'] = gb
        gain = gb['gain_over_g80_mean']
        entry['verdict'] = ('HAS SIGNAL' if gain >= .15
                            else 'WEAK' if gain >= .05 else 'NO SIGNAL')
        report['per_channel'][channel] = entry
        print(json.dumps({'channel': channel, 'rms_none': rms_none,
                          'rms_g80': rms_g80, 'gb_gain_mean': gain,
                          'verdict': entry['verdict']}), flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'probe_results.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding='utf-8')

    print('\n' + '=' * 78)
    for channel in CHANNELS:
        e = report['per_channel'][channel]
        print(f'[{channel}] rms_none={e["rms_none"]:.5f}  '
              f'rms_g80={e["rms_g80"]:.5f}  '
              f'GB gain={e["gradient_boosting_all_features"]["gain_over_g80_mean"]*100:.1f}%'
              f'  -> {e["verdict"]}')
        for group in GROUPS:
            g = e['groups'][group]
            print(f'    ridge {group:<20} rms={g["rms_probe_mean"]:.5f} '
                  f'gain={g["gain_over_g80_mean"]*100:+.1f}%')
        print(f'    constant baseline     rms={e["groups"]["all"]["rms_constant_baseline"]:.5f}')
    print('=' * 78)
    print(f'written {(OUT / "probe_results.json").relative_to(ROOT)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
