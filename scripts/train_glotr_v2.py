"""GloTR-v2 training, Selection evaluation, donor-field caching and correction.

Stages (run from ROOT with the GloTR venv):

  train     python scripts/train_glotr_v2.py train --tag S1_w24 --seed 20260905 \
                [--layers 3 --d 128 --heads 4 --dilations 1,2,4 --kernel 9 \
                 --lam-r 0 --lam-d 0 --drop-max 0 --no-gate --epochs 40]
  evaluate  python scripts/train_glotr_v2.py evaluate --tag S1_w24 [--queues selection_42]
            neural-only (no error transfer) metrics of v2 vs frozen G80 on Selection
  field     python scripts/train_glotr_v2.py field --tag S1_w24 --queue selection_42|test_21|test2020
            donor error fields with v2 as the fixed predictor (same builder as G80)
  correct   python scripts/train_glotr_v2.py correct --tag S1_w24 --queue selection_42
            frozen v6 correction (q low-rank transfer, o spatial loading eta=16,
            pseudo-outage level alpha=.25) on the cached fields + scores

Outputs: reviews/2026-11-09/glotr_v2/<tag>/ . Nothing outside that folder is written.
Discipline: develop on selection_42 only; test_21 / test2020 are run once at S5.
"""
from __future__ import annotations

import os

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
for _var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
             'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_var, '4')

import argparse
import json
import sys
import time
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = ROOT / 'reviews/2026-09-20/baseline_takeover'
for _p in (str(ROOT), str(BASE_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch

import scripts.train_g80_v2_arm as arm
from models.arm_profiles import build_profile
from models.glotr_v2 import GloTRv2Head, stack_inputs, v2_day_inputs
from models.sparse_global_tokens import global_token_context

OUT_ROOT = ROOT / 'reviews/2026-11-09/glotr_v2'
BUFFER, SLOTS = arm.BUFFER_DAYS, arm.SLOTS
SCALES = (6240.0, 0.21276629999999888)
G80_OF = {seed: ROOT / f'reviews/2026-09-26/g80_v2_round2/stability/seed{seed}/full.pt'
          for seed in arm.SEEDS}
KEY_GROUPS = arm.arm_key_groups('full')


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------

def setup():
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required')
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    return torch.device('cuda:0')


def load_g80(seed, device):
    payload = torch.load(G80_OF[seed], map_location='cpu', weights_only=False)
    model = build_profile('full', dropout=.1).requires_grad_(False).eval()
    model.load_state_dict(payload['model'], strict=True)
    return model.to(device)


def head_from_config(cfg):
    return GloTRv2Head(d=cfg['d'], heads=cfg['heads'], layers=cfg['layers'],
                       kernel=cfg['kernel'], dilations=tuple(cfg['dilations']),
                       dropout=.1, gate=cfg['gate'],
                       split_heads=cfg.get('split_heads', False))


def load_v2(tag, device):
    ck = torch.load(OUT_ROOT / tag / 'glotr_v2.pt', map_location='cpu', weights_only=False)
    head = head_from_config(ck['config'])
    head.load_state_dict(ck['model'], strict=True)
    return head.requires_grad_(False).eval().to(device), ck['config']


def g80_profile(g80, item, day, values, visible):
    q, k, valid = global_token_context(
        values=values, visible=visible, base=item.base[day - BUFFER],
        static=item.static, calendar=item.calendar, target=item.target, day=day,
        signed_hops=item.hops, scales=SCALES, raw_values=True, key_groups=KEY_GROUPS)
    out, _ = g80(q, k, valid, item.base[day - BUFFER])
    return out


def day_inputs(item, day, values, visible):
    return v2_day_inputs(values=values, visible=visible, base=item.base[day - BUFFER],
                         static=item.static, calendar=item.calendar,
                         target=item.target, day=day, signed_hops=item.hops,
                         scales=SCALES)


def predict_item(head, g80, item, duration):
    """Normalised v2 prediction (duration, 288, 2) as float64 numpy."""
    outs = []
    with torch.no_grad():
        for day in range(BUFFER, BUFFER + duration):
            prof = g80_profile(g80, item, day, item.values, item.visible)
            inp = stack_inputs([day_inputs(item, day, item.values, item.visible)])
            final, _, _ = head(inp['seq_vals'], inp['seq_vis'], inp['seq_meta'],
                               inp['query'], prof[None], inp['avail'])
            outs.append(final[0])
    return torch.stack(outs).cpu().numpy().astype(np.float64)


def predict_g80(g80, item, duration):
    with torch.no_grad():
        return torch.stack([g80_profile(g80, item, d, item.values, item.visible)
                            for d in range(BUFFER, BUFFER + duration)]
                           ).cpu().numpy().astype(np.float64)


def steepness(pred, truth, scoring, channel=0):
    pair = scoring[:, 1:, channel] & scoring[:, :-1, channel]
    dp = np.abs(np.diff(pred[..., channel], axis=1))[pair]
    dt = np.abs(np.diff(truth[..., channel], axis=1))[pair]
    return float(dp.mean() / dt.mean()) if dt.size and dt.mean() > 0 else float('nan')


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------

def prepare_fit_items(device, limit=0):
    from training.current_release_student_data import VerifiedRoleReader, assemble_episode
    split = arm.load_json(ROOT / arm.SPLIT_REL)
    if len(split['fit_ids']) != 210:
        raise ValueError('frozen Train split differs')
    stage3 = arm.load_json(ROOT / arm.STAGE3_RESULT)
    plan3 = arm.load_json(ROOT / arm.STAGE3_PLAN)
    scales = tuple(float(x) for x in stage3['scales'])
    graph = arm.load_json(ROOT / arm.GRAPH_REL)
    order = tuple(str(x) for x in graph['station_order'])
    edges = {n: torch.tensor(v, dtype=torch.long) for n, v in graph['edge_index'].items()}
    basis, mu, core = arm.load_models(scales)
    roster = {r['episode_id']: r for r in arm.load_json(ROOT / arm.ROSTER_REL)['episodes']}
    reader = VerifiedRoleReader(ROOT, ROOT / arm.RELEASE_REL, arm.RELEASE_SHA, 'train', order)
    read = lru_cache(maxsize=240)(lambda day: reader(day))
    fit_days = set(plan3['fit_days'])
    items = []
    for identity in (split['fit_ids'][:limit] if limit else split['fit_ids']):
        row = roster[identity]
        episode = assemble_episode(row, order, reader.bounds, read)
        if any(day not in fit_days for day in episode.dates):
            raise ValueError(f'episode crosses fit boundary: {identity}')
        items.append(arm.prepare_item(episode, row, basis, mu, core, scales, edges, device))
    return items


def augmented(item, day, drop_max, rng, chunks=False):
    """Change C. Resampled for every (episode, day) sample at every epoch:
    p ~ U(0, drop_max); each OTHER station is hidden for the whole target day
    with probability p, and (``chunks``) each remaining one additionally loses a
    random contiguous 1-6 h block with probability p, so sparsity varies by
    station AND by time of day as in the 2020 cohort."""
    if drop_max <= 0:
        return item.values, item.visible
    p = float(rng.uniform(0, drop_max))
    others = [n for n in range(item.values.shape[1]) if n != item.target]
    drop = [n for n in others if rng.random() < p]
    blocks = []
    if chunks:
        for n in others:
            if n not in drop and rng.random() < p:
                length = int(rng.integers(12, 73))
                start = int(rng.integers(0, SLOTS - length + 1))
                blocks.append((n, start, start + length))
    if not drop and not blocks:
        return item.values, item.visible
    visible = item.visible.clone()
    if drop:
        visible[day, drop] = False
    for n, a, b in blocks:
        visible[day, n, a:b] = False
    values = torch.where(visible, item.values, torch.zeros_like(item.values))
    return values, visible


def derivative_loss(pred, truth, scoring, *, duration, episode_count):
    parts = []
    for c in range(2):
        pair = scoring[1:, c] & scoring[:-1, c]
        if bool(pair.any()):
            dp = pred[1:, c] - pred[:-1, c]
            dt = truth[1:, c] - truth[:-1, c]
            parts.append((dp - dt).abs()[pair].mean())
    if not parts:
        return pred.new_zeros(())
    return torch.stack(parts).mean() / (duration * episode_count)


def stage_train(args):
    from scripts.fit_sparse_neighbor_context import weighted_day_loss
    device = setup()
    out = OUT_ROOT / args.tag
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'glotr_v2.pt').exists():
        raise SystemExit(f'{out}/glotr_v2.pt exists; use a new --tag')
    cfg = {'d': args.d, 'heads': args.heads, 'layers': args.layers, 'kernel': args.kernel,
           'dilations': [int(x) for x in args.dilations.split(',')],
           'gate': not args.no_gate, 'split_heads': args.split_heads, 'o_weight': args.o_weight, 'lam_r': args.lam_r, 'lam_d': args.lam_d,
           'drop_max': args.drop_max, 'drop_chunks': args.drop_chunks, 'epochs': args.epochs, 'lr': args.lr,
           'seed': args.seed, 'batch': arm.BATCH}
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    started = time.perf_counter()
    items = prepare_fit_items(device, 8 if args.smoke else 0)
    if args.smoke:
        args.epochs = cfg['epochs'] = 1
    days = [(i, d) for i, it in enumerate(items)
            for d in range(BUFFER, BUFFER + it.duration)]
    g80 = load_g80(args.seed, device)
    head = head_from_config(cfg).to(device)
    cfg['params'] = int(sum(p.numel() for p in head.parameters()))
    cfg['receptive_radius_slots'] = head.receptive_radius
    print(json.dumps({'phase': 'start', **cfg, 'fit_days': len(days),
                      'prepare_s': round(time.perf_counter() - started, 1)}), flush=True)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    # Resume appends to the log and continues from the stored epoch, so an
    # interrupted run reaches the same 40-epoch state as an uninterrupted one.
    log = (out / 'train_log.jsonl').open('a' if args.resume else 'w',
                                         encoding='utf-8')
    n_ep = len(items)
    start_epoch = 0
    if args.resume:
        payload = torch.load(args.resume, map_location=device, weights_only=False)
        head.load_state_dict(payload['model'])
        if 'optimizer' in payload:
            opt.load_state_dict(payload['optimizer'])
        if 'rng' in payload:
            rng.bit_generator.state = payload['rng']
        start_epoch = int(payload.get('epoch', 0))
        print(json.dumps({'phase': 'resume', 'from_epoch': start_epoch,
                          'path': str(args.resume)}), flush=True)
    for epoch in range(start_epoch, args.epochs):
        head.train()
        t0 = time.perf_counter()
        order = torch.randperm(len(days),
                               generator=torch.Generator().manual_seed(args.seed + epoch)).tolist()
        total = comp_l1 = comp_d = 0.
        for off in range(0, len(days), arm.BATCH):
            batch = [days[j] for j in order[off:off + arm.BATCH]]
            inputs, profs = [], []
            for i, day in batch:
                it = items[i]
                values, visible = augmented(it, day, args.drop_max, rng, args.drop_chunks)
                try:
                    with torch.no_grad():
                        prof = g80_profile(g80, it, day, values, visible)
                except ValueError as err:          # dropout removed every legal key
                    if 'legal key' not in str(err):
                        raise
                    values, visible = it.values, it.visible
                    with torch.no_grad():
                        prof = g80_profile(g80, it, day, values, visible)
                profs.append(prof)
                inputs.append(day_inputs(it, day, values, visible))
            inp = stack_inputs(inputs)
            final, delta, _ = head(inp['seq_vals'], inp['seq_vis'], inp['seq_meta'],
                                   inp['query'], torch.stack(profs), inp['avail'])
            loss = final.new_zeros(())
            for j, (i, day) in enumerate(batch):
                it = items[i]
                ref, sc = it.reference[day - BUFFER], it.scoring[day - BUFFER]
                if args.o_weight == 1.:
                    l1 = weighted_day_loss(final[j], ref, sc, duration=it.duration,
                                           episode_count=n_ep)
                else:                      # channel-weighted L1, same reduction
                    ch = torch.stack([(final[j][:, c] - ref[:, c]).abs()[sc[:, c]].mean()
                                      for c in range(2)])
                    w = final.new_tensor([1., args.o_weight])
                    l1 = (ch * w).sum() / w.sum() / (it.duration * n_ep)
                dl = derivative_loss(final[j], ref, sc, duration=it.duration,
                                     episode_count=n_ep)
                comp_l1 += float(l1.detach()); comp_d += float(dl.detach())
                loss = loss + l1 + (args.lam_d * dl if args.lam_d else 0.)
                if args.lam_r:
                    loss = loss + args.lam_r * delta[j].square().mean() / (it.duration * n_ep)
            if not torch.isfinite(loss):
                raise ValueError('nonfinite loss')
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            total += float(loss.detach())
        rec = {'epoch': epoch + 1, 'loss': total, 'l1_term': comp_l1, 'deriv_term': comp_d,
               'l1_over_deriv': comp_l1 / comp_d if comp_d else None,
               'seconds': round(time.perf_counter() - t0, 1)}
        log.write(json.dumps(rec) + '\n'); log.flush()
        print(json.dumps(rec), flush=True)
        # Interim checkpoint every 10 epochs. Write-only safety net: nothing reads
        # these unless a run is interrupted, so the final result is unchanged.
        # Optimizer and RNG state are stored so ``--resume`` continues the exact
        # same trajectory (shuffle uses seed+epoch, dropout uses ``rng``).
        if (epoch + 1) % 10 == 0 and (epoch + 1) < args.epochs:
            torch.save({'model': head.state_dict(), 'optimizer': opt.state_dict(),
                        'config': cfg, 'epoch': epoch + 1,
                        'rng': rng.bit_generator.state},
                       out / f'interim_epoch{epoch + 1:02d}.pt')
    log.close()
    torch.save({'model': head.state_dict(), 'config': cfg}, out / 'glotr_v2.pt')
    (out / 'config.json').write_text(json.dumps(cfg, indent=1), encoding='utf-8')
    print(json.dumps({'phase': 'done', 'tag': args.tag}), flush=True)


# ---------------------------------------------------------------------------
# evaluate (neural only, no error transfer)
# ---------------------------------------------------------------------------

def stage_evaluate(args):
    device = setup()
    head, cfg = load_v2(args.tag, device)
    g80 = load_g80(cfg['seed'], device)
    graph = arm.load_json(ROOT / arm.GRAPH_REL)
    order = tuple(str(x) for x in graph['station_order'])
    edges = {n: torch.tensor(v, dtype=torch.long) for n, v in graph['edge_index'].items()}
    basis, mu, core = arm.load_models(SCALES)
    queues = arm.build_queues(order)
    wanted = [q for q in args.queues.split(',') if q]
    if any(q.startswith('test') for q in wanted) and not args.final:
        raise SystemExit('test queues are run once at S5; pass --final to confirm')
    report = {'tag': args.tag, 'config': cfg}
    for qname in wanted:
        queue = queues[qname]
        rows_out = []
        read = lru_cache(maxsize=64)(lambda day, q=queue: q['reader'](day))
        for row in queue['rows']:
            episode = queue['assemble'](row, read)
            if np.any(episode.visible & episode.scoring):
                raise ValueError('score visible leak')
            L, t = row['duration_days'], episode.target_index
            sc = episode.scoring[BUFFER:BUFFER + L, t].astype(bool)
            truth = (episode.reference[BUFFER:BUFFER + L, t] / arm.S3).astype(np.float32)
            item = arm.prepare_item(episode, row, basis, mu, core, SCALES, edges, device)
            p2 = predict_item(head, g80, item, L).astype(np.float32)
            p1 = predict_g80(g80, item, L).astype(np.float32)
            others = [n for n in range(episode.visible.shape[1]) if n != t]
            density = float(episode.visible[BUFFER:BUFFER + L][:, others].mean())
            rows_out.append({'episode_id': row['episode_id'], 'station_id': str(row['station_id']),
                             'duration_days': L, 'neighbor_density': density,
                             'scores': {'v2': arm.score_physical(p2, truth, sc),
                                        'g80': arm.score_physical(p1, truth, sc)},
                             'steep_q': {'v2': steepness(p2, truth, sc), 'g80': steepness(p1, truth, sc)}})
            pred_dir = OUT_ROOT / args.tag / 'neural_predictions' / qname
            pred_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(pred_dir / ('case_' + row['episode_id'].replace('|', '_') + '.npz'),
                                episode_id=row['episode_id'], v2=p2, g80=p1, truth=truth, scoring=sc)

        def summary(rows):
            if not rows:
                return None
            return {m: {c: {k: arm.sensor_macro(rows, m, c, k)
                            for k in ('profile_macro_nrmse', 'raw_mae')} for c in ('q', 'o')}
                    | {'steep_q': float(np.nanmean([r['steep_q'][m] for r in rows]))}
                    for m in ('v2', 'g80')} | {'n': len(rows)}
        sparse = [r for r in rows_out if r['neighbor_density'] < .5]
        report[qname] = {'all': summary(rows_out), 'sparse_neighbors_lt50': summary(sparse),
                         'per_case': rows_out}
        print(json.dumps({qname: report[qname]['all']}, indent=1), flush=True)
    (OUT_ROOT / args.tag / f'evaluate_{"_".join(wanted)}.json').write_text(
        json.dumps(report, indent=1), encoding='utf-8')


def stage_evaluate_sparse(args):
    """Selection-42 under extra neighbour masking (pre-registered S3 criterion).

    Masks come from scripts/sparsity_masks.py (same rule as the Test sparsity
    experiment, queue name 'selection_42', draw 0): hidden neighbours are dark
    on the outage days. Inputs, reference profile and G80 context are all
    recomputed from the masked episode. Neural outputs only (no error transfer).
    """
    from scripts.sparsity_masks import apply_mask, build_masks
    device = setup()
    head, cfg = load_v2(args.tag, device)
    g80 = load_g80(cfg['seed'], device)
    graph = arm.load_json(ROOT / arm.GRAPH_REL)
    order = tuple(str(x) for x in graph['station_order'])
    edges = {n: torch.tensor(v, dtype=torch.long) for n, v in graph['edge_index'].items()}
    basis, mu, core = arm.load_models(SCALES)
    queue = arm.build_queues(order)['selection_42']
    read = lru_cache(maxsize=64)(lambda day: queue['reader'](day))
    episodes = [(row, queue['assemble'](row, read)) for row in queue['rows']]
    masks = build_masks('selection_42', [(r['episode_id'], e.visible, e.target_index,
                                         int(r['duration_days'])) for r, e in episodes])
    levels = [float(x) for x in args.levels.split(',')]
    report = {'tag': args.tag, 'levels': {}}
    for level in levels:
        rows_out, skipped = [], []
        for row, episode in episodes:
            L, t = int(row['duration_days']), episode.target_index
            masked = apply_mask(episode, masks[row['episode_id']]['hidden'][str(level)], L)
            sc = masked.scoring[BUFFER:BUFFER + L, t].astype(bool)
            truth = (masked.reference[BUFFER:BUFFER + L, t] / arm.S3).astype(np.float32)
            try:
                item = arm.prepare_item(masked, row, basis, mu, core, SCALES, edges, device)
                p2 = predict_item(head, g80, item, L).astype(np.float32)
                p1 = predict_g80(g80, item, L).astype(np.float32)
            except ValueError as err:
                if 'legal key' not in str(err):
                    raise
                skipped.append(row['episode_id'])
                continue
            rows_out.append({'episode_id': row['episode_id'], 'station_id': str(row['station_id']),
                             'scores': {'v2': arm.score_physical(p2, truth, sc),
                                        'g80': arm.score_physical(p1, truth, sc)}})
        summ = {m: {c: arm.sensor_macro(rows_out, m, c, 'profile_macro_nrmse') for c in ('q', 'o')}
                for m in ('v2', 'g80')}
        for m in summ:
            summ[m]['mean'] = (summ[m]['q'] + summ[m]['o']) / 2
        report['levels'][str(level)] = {'n': len(rows_out), 'skipped': skipped, **summ}
        print(json.dumps({'level': level, **report['levels'][str(level)]}), flush=True)
    report['criterion_mean_over_levels'] = {
        m: float(np.mean([report['levels'][str(l)][m]['mean'] for l in levels])) for m in ('v2', 'g80')}
    (OUT_ROOT / args.tag / 'evaluate_sparse_selection_42.json').write_text(
        json.dumps(report, indent=1), encoding='utf-8')
    print(json.dumps(report['criterion_mean_over_levels']), flush=True)


# ---------------------------------------------------------------------------
# donor fields with v2 as predictor, then frozen v6 correction
# ---------------------------------------------------------------------------

def v2_engine(tag, queue_name):
    import scripts.run_transductive as rt
    device = setup()
    head, cfg = load_v2(tag, device)
    if queue_name == 'test2020':
        from scripts.run_2020_g80_family import engine_and_queue
        engine, _, _ = engine_and_queue(cfg['seed'])
        availability = json.loads((ROOT / 'reviews/2026-09-28/2020_blind19/'
                                   'pseudo_availability.json').read_text(encoding='utf-8'))
        placement = {('test2020', r['episode_id']): r for r in availability['rows']}
    else:
        engine = rt.Engine(G80_OF[cfg['seed']])
        if queue_name == 'test_21':
            from scripts.run_test21_reeval import OUT as T21, install_relaxed_test_queue
            install_relaxed_test_queue(engine)
            placement = {('test_21', r['episode_id']): r for r in json.loads(
                (T21 / 'pseudo_availability_test21.json').read_text(encoding='utf-8'))['rows']}
        else:
            _, placement = rt.load_availability()
    g80 = engine.model
    engine.forward = lambda item, duration: predict_item(head, g80, item, duration)
    return rt, engine, placement


def stage_field(args):
    if args.queue.startswith('test') and not args.final:
        raise SystemExit('test queues are run once at S5; pass --final to confirm')
    rt, engine, placement = v2_engine(args.tag, args.queue)
    folder = OUT_ROOT / args.tag / 'fields' / args.queue
    folder.mkdir(parents=True, exist_ok=True)
    rows = engine.queues[args.queue]['rows']
    for k, row in enumerate(rows, 1):
        path = folder / ('case_' + row['episode_id'].replace('|', '_') + '.npz')
        if path.exists():
            continue
        case = rt.build_case(engine, args.queue, row, availability=placement)
        if case['leak_hidden_cells'] != 0:
            raise AssertionError('target gap leaked into a donor pass')
        rt.save_case(case, path)
        print(f'field {k}/{len(rows)} {row["episode_id"]} {case["case_seconds"]:.1f}s', flush=True)


def stage_correct(args):
    from scripts.loading_split_candidate import station_order
    from scripts.run_o_pseudo_calibration import attach_pseudo, evaluate
    from scripts.run_o_spatial_loading import source_rows
    folder = OUT_ROOT / args.tag / 'fields' / args.queue
    rows = attach_pseudo(source_rows(folder, station_order()), folder)
    report = evaluate(rows, .25, save_dir=OUT_ROOT / args.tag / 'final_predictions' / args.queue)
    out = {'queue': args.queue, 'n': report['n'],
           'v2_full_pipeline_sensor_macro': report['sensor_macro']['v6'],
           'v2_without_o_level_sensor_macro': report['sensor_macro']['v5'],
           'by_duration': report['by_duration']}
    (OUT_ROOT / args.tag / f'correct_{args.queue}.json').write_text(
        json.dumps(out, indent=1), encoding='utf-8')
    print(json.dumps(out, indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('stage', choices=('train', 'evaluate', 'evaluate_sparse', 'field', 'correct'))
    p.add_argument('--tag', required=True)
    p.add_argument('--seed', type=int, default=arm.SEED, choices=arm.SEEDS)
    p.add_argument('--d', type=int, default=128)
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--layers', type=int, default=3)
    p.add_argument('--kernel', type=int, default=9)
    p.add_argument('--dilations', default='1,2,4', help='radius = (k-1)/2*sum; 1,2,4 -> 28 slots')
    p.add_argument('--no-gate', action='store_true')
    p.add_argument('--lam-r', type=float, default=0.)
    p.add_argument('--lam-d', type=float, default=0.)
    p.add_argument('--split-heads', action='store_true', help='separate q/o read-out heads')
    p.add_argument('--o-weight', type=float, default=1., help='weight of the o channel in the L1 term')
    p.add_argument('--drop-max', type=float, default=0.)
    p.add_argument('--drop-chunks', action='store_true', help='also hide random 1-6 h blocks')
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--queues', default='selection_42')
    p.add_argument('--levels', default='0.5,0.75', help='evaluate_sparse: extra masking levels')
    p.add_argument('--queue', default='selection_42')
    p.add_argument('--final', action='store_true', help='allow test queues (S5 only)')
    p.add_argument('--smoke', action='store_true', help='train: 8 episodes, 1 epoch')
    p.add_argument('--resume', default='', help='train: continue from an interim_epoch*.pt '
                                                '(optimizer and RNG state are restored)')
    args = p.parse_args()
    {'train': stage_train, 'evaluate': stage_evaluate, 'evaluate_sparse': stage_evaluate_sparse,
     'field': stage_field, 'correct': stage_correct}[args.stage](args)


if __name__ == '__main__':
    main()
