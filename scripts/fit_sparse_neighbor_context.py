"""Train-only matched neighbor-context extension of the global q/o head."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch import Tensor

from models.sparse_global_tokens import GlobalTokenProfile, global_token_context
from models.sparse_neighbor_context import NeighborContextProfile, neighbor_context
from scripts.fit_profile_residual_candidate import file_sha256, group_for_case, score_case, summarize


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'reviews/2026-09-24/SPARSE_NEIGHBOR_CONTEXT_WORK_ORDER.md'
BASE_RUN = ROOT / 'reviews/2026-09-24/sparse_global_token_full_v1'
BASE_CHECKPOINT_SHA = 'c4adb3dbc6ed0968860546372daaf5e042249f271ddbafbf0fb8d4e737f80332'
PINS = {
    'split': ('reviews/2026-09-23/optimization_v1/split.json',
              '7dd2d8a170c3e0d9e8710888ced50ca2638a9c1cc95a33d5e9a214b2992e789e'),
    'capacity_result': ('reviews/2026-09-23/optimization_v1/stage3_run_v1/result.json',
                        '21864f1313d6ade9a8483375a049e7a94a54bd7e66c1165560d5fddfc4941eda'),
    'basis': ('reviews/2026-09-23/optimization_v1/stage3_run_v1/basis.pt',
              '79c0f3e4f3cd04138b0b86126609fd56578108c9fb0ae20a209e05d237891e9f'),
    'mu': ('reviews/2026-09-23/optimization_v1/stage3_run_v1/mu.pt',
           'ec4e8fa8ee2a51c37ed4a6e39f59af88d6ac57adae54558148ca5d15020fe7fe'),
    'core': ('reviews/2026-09-23/optimization_v1/stage4_run_v1/core/student.pt',
             '261ef98f7371f1a5c8d4f8883f94d8fb8a39d3acd121f1f80d99cdaae09c7ce5'),
    'stage4_result': ('reviews/2026-09-23/optimization_v1/stage4_run_v1/result.json',
                      '75fb83873b0fed427173ff8bf2b0ff72bfb4f9474375a631603302790087a87e'),
    'release': ('reviews/2026-09-12/data_release_final/QO_FORMAL_RELEASE.json',
                '5c6dd879beb0f70bcd2388601736cbf0bf341f111a267f79801c478e780b9287'),
    'graph': ('data/pems/graphs/qo_pilot_v1/release_v1/graph.json',
              '6373d9a58173b4e828fced08b57f944a1ba78ac8733125e1ee100c2f5a034eb4'),
    'roster': ('reviews/2026-09-12/formal_mask_bank/EPISODE_ROSTER_NON_TEST.json',
               'cb38dca7027a8bc087290b01cf61e6f6f2550ddbc2f2a4f4a92b807e8813bbee'),
}


def signed_chain_hops(edges: dict[str, Tensor], *, target: int, nodes: int) -> Tensor:
    if not 0 <= target < nodes or nodes < 2:
        raise ValueError('target and graph size required')
    line = torch.arange(nodes - 1, dtype=torch.long)
    upstream = torch.stack((line, line + 1))
    downstream = torch.stack((line + 1, line))
    if (not torch.equal(edges['upstream'].cpu(), upstream) or
            not torch.equal(edges['downstream'].cpu(), downstream) or
            edges['on_ramp'].numel() or edges['off_ramp'].numel()):
        raise ValueError('frozen 20-station directed chain differs')
    return (target - torch.arange(nodes)).to(torch.float32)


def weighted_day_loss(predicted: Tensor, truth: Tensor, scoring: Tensor,
                      *, duration: int, episode_count: int) -> Tensor:
    if (predicted.ndim != 2 or predicted.shape != truth.shape or
            predicted.shape != scoring.shape or predicted.shape[-1] != 2 or
            scoring.dtype != torch.bool or duration <= 0 or episode_count <= 0 or
            not bool(scoring.any(dim=0).all())):
        raise ValueError('nonempty scored q/o day required')
    channel_l1 = torch.stack([(predicted[:, i] - truth[:, i]).abs()[scoring[:, i]].mean()
                              for i in range(2)])
    return channel_l1.mean() / (duration * episode_count)


def neighbor_candidate_gate(candidate: dict[str, float], control: dict[str, float],
                            before: dict[str, float],
                            candidate_mae: dict[str, float],
                            control_mae: dict[str, float],
                            groups: dict[int, dict[str, float]],
                            control_groups: dict[int, dict[str, float]]) -> bool:
    return bool(
        candidate['q'] <= .95 * control['q'] and
        candidate['q'] <= before['q'] and
        candidate['o'] <= min(control['o'], before['o']) and
        all(candidate_mae[channel] <= control_mae[channel]
            for channel in ('q', 'o')) and
        all(groups[group][channel] <= 1.05 * control_groups[group][channel]
            for group in (4, 6) for channel in ('q', 'o')) and
        all(candidate[channel] >= 0 for channel in ('q', 'o')))


@dataclass
class PreparedEpisode:
    identity: str
    mask_id: str
    station_id: str
    duration: int
    target: int
    group: int | None
    values: Tensor
    visible: Tensor
    static: Tensor
    calendar: Tensor
    base: Tensor
    reference: Tensor
    scoring: Tensor
    hops: Tensor


def main() -> None:
    from representations.basis import GlobalProfileBasis, MeanFunction
    from training.current_release_student_data import (
        VerifiedRoleReader, assemble_episode, prepare_case, predict_case, verified_bytes,
    )
    from training.pilot_engine import make_student
    from training.qo_attributes import AnnualAttributes

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('preflight', 'full'), required=True)
    parser.add_argument('--overfit-lr', type=float, default=3e-4)
    parser.add_argument('--train-lr', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=.1)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--resume-run', type=Path, default=None)
    parser.add_argument('--resume-epoch', type=int, default=0)
    parser.add_argument('--resume-raw-epoch', type=int, default=0)
    parser.add_argument('--resume-masked-epoch', type=int, default=0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available():
        raise RuntimeError('single GPU required by frozen plan')
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    for name, (relative, digest) in PINS.items():
        if file_sha256(ROOT / relative) != digest:
            raise ValueError(f'frozen {name} SHA differs')
    if file_sha256(BASE_RUN / 'raw_values.pt') != BASE_CHECKPOINT_SHA:
        raise ValueError('retained global-token checkpoint SHA differs')
    split = json.loads((ROOT / PINS['split'][0]).read_text(encoding='utf-8'))
    stage3 = json.loads((ROOT / PINS['capacity_result'][0]).read_text(encoding='utf-8'))
    stage3_plan = json.loads((ROOT / 'reviews/2026-09-23/optimization_v1/stage3_run_v1/plan.json').read_text(encoding='utf-8'))
    stage4 = json.loads((ROOT / PINS['stage4_result'][0]).read_text(encoding='utf-8'))
    if (split['validation_group_indices'] != [4, 6] or
            len(split['fit_ids']) != 210 or len(split['validation_ids']) != 58 or
            len(stage3_plan['fit_days']) != 171 or len(stage3_plan['excluded_days']) != 48 or
            set(stage3_plan['fit_days']) & set(stage3_plan['excluded_days']) or
            stage4['candidate_retained_by_frozen_rule'] is not True):
        raise ValueError('Train split or frozen stage4 gate differs')
    core_state = torch.load(ROOT / PINS['core'][0], map_location='cpu', weights_only=False)
    if (core_state['epochs'] != 200 or core_state['lr'] != .001 or
            core_state['provenance']['basis_sha256'] != PINS['basis'][1] or
            core_state['provenance']['mu_sha256'] != PINS['mu'][1]):
        raise ValueError('frozen Train-only core provenance differs')
    graph = json.loads(verified_bytes(ROOT / PINS['graph'][0], PINS['graph'][1]))
    order = tuple(str(x) for x in graph['station_order'])
    edges = {name: torch.tensor(value, dtype=torch.long)
             for name, value in graph['edge_index'].items()}
    if len(order) != 20:
        raise ValueError('frozen station count differs')
    signed_chain_hops(edges, target=0, nodes=20)
    basis_ck = torch.load(ROOT / PINS['basis'][0], map_location='cpu', weights_only=False)
    mu_ck = torch.load(ROOT / PINS['mu'][0], map_location='cpu', weights_only=False)
    scales = tuple(float(x) for x in stage3['scales'])
    if not np.array_equal(np.asarray(scales), np.asarray(mu_ck['scales'])):
        raise ValueError('Train-separated channel scales differ')
    basis = GlobalProfileBasis(288, 2, 16).requires_grad_(False).eval()
    basis.load_state_dict(basis_ck['basis_state'])
    mu = MeanFunction(5, 4, 6, channels=2).requires_grad_(False).eval()
    mu.load_state_dict(mu_ck['mean_state'])
    core = make_student(rank=16, hidden=32).requires_grad_(False).eval()
    core.load_state_dict(core_state['student'])
    cfg = json.loads((ROOT / 'runs/current_student_train_preparation_v2/registration.json').read_text(encoding='utf-8'))['config']
    if (cfg['release_sha256'] != PINS['release'][1] or
            cfg['graph_sha256'] != PINS['graph'][1] or
            cfg['roster_sha256'] != PINS['roster'][1] or tuple(cfg['station_order']) != order):
        raise ValueError('frozen release registration differs')
    attributes = AnnualAttributes.from_files(ROOT / 'data/pems/releases/partition_v1',
                                             manifest_sha256=cfg['attribute_manifest_sha256'])
    roster = json.loads(verified_bytes(ROOT / PINS['roster'][0], PINS['roster'][1]))['episodes']
    by_id = {row['episode_id']: row for row in roster if row['split'] == 'train'}
    if not set(split['fit_ids'] + split['validation_ids']) <= set(by_id):
        raise ValueError('Train episode identities differ')
    args.output.mkdir(parents=True)
    code_paths = [Path(__file__), ROOT / 'models/sparse_neighbor_context.py',
                  ROOT / 'models/sparse_global_tokens.py',
                  ROOT / 'models/sparse_observation_profile.py',
                  ROOT / 'training/current_release_student_data.py',
                  ROOT / 'training/pilot_data.py', ROOT / 'representations/basis.py']
    fit_ids = split['fit_ids']
    probe_ids = split['validation_ids']
    if args.train_lr <= 0 or args.batch_size < 1 or not 0 <= args.dropout < 1 or args.weight_decay < 0 or args.epochs < 1:
        raise ValueError('train learning rate must be positive')
    if (args.epochs != 40 or args.train_lr != .001 or args.batch_size != 8 or
            args.dropout != .1 or args.weight_decay != .0001 or
            args.resume_run is not None or args.resume_epoch or
            args.resume_raw_epoch or args.resume_masked_epoch):
        raise ValueError('fixed global-token training contract differs')
    plan = {'scope': 'Train-only matched per-station neighbor context extension',
            'mode': args.mode, 'work_order_sha256': file_sha256(PLAN),
            'source_sha256': {name: digest for name, (_, digest) in PINS.items()},
            'baseline_checkpoint_sha256': BASE_CHECKPOINT_SHA,
            'code_sha256': {str(path.relative_to(ROOT)): file_sha256(path) for path in code_paths},
            'fit_episodes': len(fit_ids),
            'validation_episodes': 58 if args.mode == 'full' else 0,
            'smoke_probe_episodes': 10 if args.mode == 'smoke' else 0,
            'smoke_fit_ids': fit_ids if args.mode == 'smoke' else None,
            'smoke_probe_ids': probe_ids if args.mode == 'smoke' else None,
            'candidate': {'seed': 20260905, 'epochs': args.epochs, 'lr': args.train_lr,
                          'weight_decay': args.weight_decay, 'batch_target_days': args.batch_size,
                          'dropout': args.dropout, 'heads': 2, 'hidden_dim': 64,
                          'correction_l2_weight': .001,
                          'global_keys': 'retained 285 plus 38 adjacent-day station bins and 19 station day profiles'},
            'executed_epochs_per_arm': 1 if args.mode == 'preflight' else
                                       (10 if args.mode == 'smoke' else
                                        0 if args.mode in ('lr_range', 'overfit', 'fixed_batch') else args.epochs),
            'fixed_batch_probe': {'target_day_offset': 0, 'steps': 800,
                                  'learning_rates': [3e-4, 1e-3, 3e-3],
                                  'dropout': 0., 'weight_decay': 0., 'penalty': 0.}
                                 if args.mode == 'fixed_batch' else None,
            'overfit_lr': args.overfit_lr if args.mode == 'overfit' else None,
            'arms': ['neighbor_context', 'global_control'],
            'closed_roles': ['selection', 'calibration', 'test']}
    (args.output / 'plan.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    device = torch.device('cuda:0')
    reader = VerifiedRoleReader(ROOT, ROOT / PINS['release'][0], PINS['release'][1],
                                'train', order)
    read = lru_cache(maxsize=240)(reader)
    fit_days = set(stage3_plan['fit_days'])
    excluded_days = set(stage3_plan['excluded_days'])

    def prepare_one(identity: str, *, validation: bool) -> PreparedEpisode:
        row = by_id[identity]
        episode = assemble_episode(row, order, reader.bounds, read)
        allowed = excluded_days if validation else fit_days
        if any(day not in allowed for day in episode.dates):
            raise ValueError('episode crosses frozen fit/validation date boundary')
        features = np.asarray([[attributes.lookup(day, station)['features'] for station in order]
                               for day in episode.dates], dtype=np.float32)
        if not np.array_equal(features, np.broadcast_to(features[0], features.shape)):
            raise ValueError('date-varying static features unsupported')
        case = prepare_case(episode, basis, mu, torch.tensor(scales, dtype=torch.float32),
                            torch.from_numpy(features[0]), edges)
        selected = case['teacher_target']
        geometry = torch.zeros_like(selected)
        duration = row['duration_days']
        geometry[3:3 + duration, episode.target_index] = True
        if not torch.equal(selected, geometry):
            raise ValueError('target support differs from registered outage geometry')
        with torch.no_grad():
            coefficients = predict_case(core, case)
            base = basis.decode(coefficients[selected], case['inputs'].mean[selected])
        return PreparedEpisode(identity, row['mask_id'], str(row['station_id']),
                               duration, episode.target_index,
                               group_for_case(case, split['groups']),
                               torch.as_tensor(episode.values, dtype=torch.float32, device=device),
                               torch.as_tensor(episode.visible, dtype=torch.bool, device=device),
                               case['static'].to(device), case['inputs'].calendar.to(device),
                               base.to(device), case['reference'][selected].to(device),
                               case['scoring'][selected].to(device),
                               signed_chain_hops(edges, target=episode.target_index,
                                                 nodes=len(order)).to(device))

    print(json.dumps({'phase': 'prepare_fit', 'episodes': len(fit_ids)}), flush=True)
    started_prepare = time.perf_counter()
    fit_items = [prepare_one(identity, validation=False) for identity in fit_ids]
    fit_prepare_seconds = time.perf_counter() - started_prepare
    fit_access_cut = len(reader.access_log)
    (args.output / 'fit_access_log.json').write_text(
        json.dumps(reader.access_log, indent=2), encoding='utf-8')
    days = [(i, day) for i, item in enumerate(fit_items)
            for day in range(3, 3 + item.duration)]
    torch.manual_seed(20260905)
    initial = GlobalTokenProfile(dropout=0. if args.mode in ('lr_range','overfit') else args.dropout).to(device)
    frozen_initial = {name: value.detach().clone() for name, value in initial.state_dict().items()}
    del initial

    if args.mode == 'fixed_batch':
        from training.sparse_training_probe import fit_fixed_batch

        item = fit_items[0]
        query, keys, valid = global_token_context(
            values=item.values, visible=item.visible, base=item.base[0],
            static=item.static, calendar=item.calendar, target=item.target, day=3,
            signed_hops=item.hops, scales=scales, raw_values=True)
        fixture = {'query':query.cpu(), 'keys':keys.cpu(), 'valid':valid.cpu(),
                   'base':item.base[0].cpu(), 'truth':item.reference[0].cpu(),
                   'scoring':item.scoring[0].cpu(), 'episode_id':item.identity,
                   'day_index':3, 'scales':scales}
        torch.save(fixture, args.output/'fixed_batch.pt')
        experiments = []
        for rate in (3e-4, 1e-3, 3e-3):
            fitted = fit_fixed_batch(frozen_initial, query, keys, valid,
                                     item.base[0], item.reference[0], item.scoring[0],
                                     lr=rate, steps=800, record_every=25)
            tag = str(rate).replace('.', '_')
            torch.save(fitted['model_state'], args.output/f'weights_{tag}.pt')
            np.savez_compressed(args.output/f'prediction_{tag}.npz',
                                prediction=fitted['prediction'].cpu().numpy())
            summary = {k:v for k,v in fitted.items() if k not in ('prediction', 'model_state')}
            experiments.append(summary)
            print(json.dumps({'phase':'fixed_batch', 'lr':rate,
                              'initial':summary['history'][0],
                              'final':summary['history'][-1],
                              'seconds':summary['seconds']}),flush=True)
        report = {'status':'fixed_batch_diagnostic_completed',
                  'scope':'one Train fit-group day, optimization diagnostic only',
                  'episode_id':item.identity, 'day_index':3, 'slots':288,
                  'scales':scales, 'experiments':experiments,
                  'fixture_sha256':file_sha256(args.output/'fixed_batch.pt'),
                  'probe_code_sha256':file_sha256(ROOT/'training/sparse_training_probe.py'),
                  'fit_source_accesses':fit_access_cut,
                  'gpu_peak_reserved_bytes':torch.cuda.max_memory_reserved(),
                  'validation_selection_calibration_test_read':False}
        (args.output/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
        print(json.dumps({'status':report['status']}),flush=True)
        return

    if args.mode in ('lr_range', 'overfit'):
        if args.mode == 'overfit' and not 1e-6 <= args.overfit_lr <= 1e-2:
            raise ValueError('overfit learning rate must be in [1e-6,1e-2]')
        torch.manual_seed(20260905)
        diagnostic = GlobalTokenProfile(dropout=0.).to(device)
        diagnostic.load_state_dict(frozen_initial)
        diagnostic.train()
        optimizer = torch.optim.AdamW(diagnostic.parameters(), lr=1e-6 if args.mode == 'lr_range'
                                       else args.overfit_lr, weight_decay=0.)
        def entire_fit_l1() -> dict[str, float]:
            diagnostic.eval()
            totals = torch.zeros(2, device=device)
            with torch.no_grad():
                for item in fit_items:
                    for day in range(3, 3 + item.duration):
                        q, k, mask = global_token_context(
                            values=item.values, visible=item.visible,
                            base=item.base[day-3], static=item.static,
                            calendar=item.calendar, target=item.target, day=day,
                            signed_hops=item.hops, scales=scales, raw_values=True)
                        predicted, _ = diagnostic(q, k, mask, item.base[day-3])
                        for channel in range(2):
                            active = item.scoring[day-3, :, channel]
                            totals[channel] += (predicted[:, channel] -
                                                item.reference[day-3, :, channel]).abs()[active].mean() / (
                                                    item.duration * len(fit_items))
            diagnostic.train()
            return {'q':float(totals[0]), 'o':float(totals[1])}
        initial_fit_l1 = entire_fit_l1() if args.mode == 'overfit' else None
        schedule = torch.logspace(-6, -1, 120).tolist() if args.mode == 'lr_range' else [args.overfit_lr]*200
        order = torch.randperm(len(days), generator=torch.Generator().manual_seed(20260905)).tolist()
        history = []
        best = float('inf')
        stop_reason = None
        for step, rate in enumerate(schedule):
            for group in optimizer.param_groups:
                group['lr'] = float(rate)
            indices = [order[(step*8 + j) % len(order)] for j in range(8)]
            selected_days = [days[index] for index in indices]
            contexts = []
            for item_index, day in selected_days:
                item = fit_items[item_index]
                contexts.append(global_token_context(
                    values=item.values, visible=item.visible,
                    base=item.base[day-3], static=item.static,
                    calendar=item.calendar, target=item.target, day=day,
                    signed_hops=item.hops, scales=scales, raw_values=True))
            query = torch.cat([c[0] for c in contexts])
            keys = torch.cat([c[1] for c in contexts])
            valid = torch.cat([c[2] for c in contexts])
            base = torch.cat([fit_items[item_index].base[day-3]
                              for item_index, day in selected_days])
            optimizer.zero_grad(set_to_none=True)
            prediction, _ = diagnostic(query, keys, valid, base)
            loss = prediction.new_zeros(())
            for j,(item_index,day) in enumerate(selected_days):
                item = fit_items[item_index]
                part = slice(j*288,(j+1)*288)
                loss = loss + weighted_day_loss(
                    prediction[part], item.reference[day-3], item.scoring[day-3],
                    duration=item.duration, episode_count=len(fit_items))
            value = float(loss.detach())
            if not np.isfinite(value):
                stop_reason = 'nonfinite_loss'
                break
            if args.mode == 'lr_range' and step >= 20 and value > 4*best:
                stop_reason = 'loss_explosion'
                break
            loss.backward()
            if not all(p.grad is None or torch.isfinite(p.grad).all()
                       for p in diagnostic.parameters()):
                stop_reason = 'nonfinite_gradient'
                break
            optimizer.step()
            best = min(best,value)
            history.append({'step':step+1,'lr':float(rate),'loss':value})
            if (step+1) % 20 == 0:
                print(json.dumps({'diagnostic':args.mode,'step':step+1,
                                  'lr':float(rate),'loss':value}),flush=True)
        torch.cuda.synchronize()
        final_fit_l1 = entire_fit_l1() if args.mode == 'overfit' else None
        with (args.output/'diagnostic_steps.jsonl').open('x',encoding='utf-8') as log:
            for row in history:
                log.write(json.dumps(row)+'\n')
        report = {'status':args.mode+'_completed','fit_episodes':len(fit_items),
                  'steps_completed':len(history),'stop_reason':stop_reason,
                  'first_loss':history[0]['loss'] if history else None,
                  'last_loss':history[-1]['loss'] if history else None,
                  'minimum_loss':best if history else None,
                  'last_lr':history[-1]['lr'] if history else None,
                  'initial_entire_fit_l1':initial_fit_l1,
                  'final_entire_fit_l1':final_fit_l1,
                  'gpu_peak_reserved_bytes':torch.cuda.max_memory_reserved(),
                  'fit_source_accesses':fit_access_cut,
                  'selection_calibration_test_read':False,
                  'diagnostic_steps_sha256':file_sha256(args.output/'diagnostic_steps.jsonl')}
        if args.mode == 'overfit':
            torch.save({'model':diagnostic.state_dict(), 'plan':plan},
                       args.output/'diagnostic.pt')
            report['diagnostic_checkpoint_sha256'] = file_sha256(args.output/'diagnostic.pt')
        (args.output/'result.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps(report),flush=True)
        return

    def train_arm(name: str, *, epochs: int) -> tuple[torch.nn.Module, float]:
        torch.manual_seed(20260905)
        source = torch.load(BASE_RUN / 'raw_values.pt', map_location='cpu',
                            weights_only=False)
        if name == 'neighbor_context':
            model = NeighborContextProfile.from_global_state(
                source['model'], dropout=args.dropout).to(device)
        else:
            model = GlobalTokenProfile(dropout=args.dropout).to(device)
            model.load_state_dict(source['model'], strict=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.train_lr, weight_decay=args.weight_decay)
        start_epoch = 0
        resume_epoch = (args.resume_raw_epoch if name == 'neighbor_context' else
                        args.resume_masked_epoch) or args.resume_epoch
        if args.resume_run is not None and resume_epoch > 0:
            ckpt = args.resume_run / f'{name}_checkpoint_epoch_{resume_epoch:03}.pt'
            saved = torch.load(ckpt, map_location=device, weights_only=False)
            model.load_state_dict(saved['model'])
            optimizer.load_state_dict(saved['optimizer'])
            start_epoch = int(saved['epoch'])
        epoch_seconds = 0.
        with (args.output / f'{name}_metrics.jsonl').open('x', encoding='utf-8') as log:
            for epoch in range(start_epoch, epochs):
                model.train()
                started = time.perf_counter()
                shuffle = torch.randperm(len(days), generator=torch.Generator().manual_seed(20260905 + epoch)).tolist()
                epoch_loss = 0.
                for offset in range(0, len(days), args.batch_size):
                    selected_days = [days[j] for j in shuffle[offset:offset + args.batch_size]]
                    contexts = []
                    for item_index, day in selected_days:
                        item = fit_items[item_index]
                        common = dict(values=item.values, visible=item.visible,
                                      base=item.base[day - 3], static=item.static,
                                      calendar=item.calendar, target=item.target,
                                      day=day, signed_hops=item.hops,
                                      scales=scales)
                        q, k, valid = (neighbor_context(**common)
                                       if name == 'neighbor_context' else
                                       global_token_context(**common,
                                                            raw_values=True))
                        contexts.append((q, k, valid, item.base[day - 3]))
                    query = torch.cat([c[0] for c in contexts])
                    keys = torch.cat([c[1] for c in contexts])
                    valid = torch.cat([c[2] for c in contexts])
                    base = torch.cat([c[3] for c in contexts])
                    optimizer.zero_grad(set_to_none=True)
                    predicted, correction = model(query, keys, valid, base)
                    loss = predicted.new_zeros(())
                    for j, (item_index, day) in enumerate(selected_days):
                        item = fit_items[item_index]
                        part = slice(j * 288, (j + 1) * 288)
                        loss = loss + weighted_day_loss(
                            predicted[part], item.reference[day - 3],
                            item.scoring[day - 3], duration=item.duration,
                            episode_count=len(fit_items))
                        loss = loss + .001 * correction[part].square().mean() / (
                            item.duration * len(fit_items))
                    if not torch.isfinite(loss):
                        raise ValueError('nonfinite attention loss')
                    loss.backward()
                    optimizer.step()
                    epoch_loss += float(loss.detach())
                torch.cuda.synchronize()
                duration_s = time.perf_counter() - started
                epoch_seconds += duration_s
                log.write(json.dumps({'epoch': epoch + 1, 'loss': epoch_loss,
                                      'seconds': duration_s}) + '\n')
                log.flush()
                if epoch == 0 or (epoch + 1) % 20 == 0:
                    print(json.dumps({'arm': name, 'epoch': epoch + 1,
                                      'loss': epoch_loss, 'seconds': duration_s}), flush=True)
                    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                                'epoch': epoch + 1},
                               args.output / f'{name}_checkpoint_epoch_{epoch + 1:03}.pt')
        model.requires_grad_(False).eval()
        torch.save({'model': model.state_dict(), 'plan': plan}, args.output / f'{name}.pt')
        return model, epoch_seconds

    print(json.dumps({'phase': 'fit_neighbor', 'target_days': len(days)}), flush=True)
    raw_model, raw_seconds = train_arm('neighbor_context',
                                       epochs=1 if args.mode == 'preflight' else args.epochs)
    if args.mode == 'preflight':
        report = {'status': 'preflight_completed', 'fit_episodes': len(fit_ids),
                  'fit_target_days': len(days), 'fit_prepare_seconds': fit_prepare_seconds,
                  'first_epoch_seconds': raw_seconds,
                  'projected_two_arm_40_epoch_seconds': fit_prepare_seconds + 80 * raw_seconds,
                  'gpu_peak_reserved_bytes': torch.cuda.max_memory_reserved(),
                  'fit_source_accesses': fit_access_cut,
                  'selection_calibration_test_read': False}
        (args.output / 'result.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(report), flush=True)
        return
    print(json.dumps({'phase': 'fit_control', 'target_days': len(days)}), flush=True)
    masked_model, masked_seconds = train_arm('global_control', epochs=args.epochs)
    del fit_items
    torch.cuda.empty_cache()
    print(json.dumps({'phase': 'prepare_probe' if args.mode == 'smoke' else 'prepare_validation',
                      'episodes': len(probe_ids)}), flush=True)
    val_items = [prepare_one(identity, validation=args.mode == 'full') for identity in probe_ids]
    (args.output / 'access_log.json').write_text(json.dumps(reader.access_log, indent=2), encoding='utf-8')
    (args.output / 'validation').mkdir()

    def infer(model: torch.nn.Module, item: PreparedEpisode,
              *, neighbor: bool) -> Tensor:
        profiles = []
        with torch.no_grad():
            for day in range(3, 3 + item.duration):
                common = dict(values=item.values, visible=item.visible,
                              base=item.base[day - 3], static=item.static,
                              calendar=item.calendar, target=item.target,
                              day=day, signed_hops=item.hops, scales=scales)
                q, k, valid = (neighbor_context(**common) if neighbor else
                               global_token_context(**common, raw_values=True))
                predicted, _ = model(q, k, valid, item.base[day - 3])
                profiles.append(predicted.cpu())
        return torch.stack(profiles)

    rows_neighbor, rows_control = [], []
    for i, item in enumerate(val_items):
        core_pred = item.base.cpu()
        neighbor_pred = infer(raw_model, item, neighbor=True)
        control_pred = infer(masked_model, item, neighbor=False)
        reference = item.reference.cpu()
        scoring = item.scoring.cpu()
        archive = args.output / 'validation' / f'{i:03}.npz'
        np.savez_compressed(archive, core=core_pred.numpy(),
                            neighbor=neighbor_pred.numpy(),
                            control=control_pred.numpy(), scoring=scoring.numpy(),
                            coordinates=np.argwhere(scoring.numpy()),
                            truth_scored=reference.numpy()[scoring.numpy()],
                            episode_id=np.asarray(item.identity),
                            mask_id=np.asarray(item.mask_id))
        common = {'episode_id': item.identity, 'mask_id': item.mask_id,
                  'station_id': item.station_id, 'duration_days': item.duration,
                  'support_group': item.group, 'archive_sha256': file_sha256(archive)}
        base_score = score_case(core_pred.reshape(-1, 288, 2),
                                reference.reshape(-1, 288, 2),
                                scoring.reshape(-1, 288, 2), scales)
        rows_neighbor.append({**common, 'scores': {
            'core': base_score,
            'residual': score_case(neighbor_pred, reference, scoring, scales)}})
        rows_control.append({**common, 'scores': {
            'core': base_score,
            'residual': score_case(control_pred, reference, scoring, scales)}})
    for name, rows in (('neighbor', rows_neighbor), ('control', rows_control)):
        with (args.output / f'{name}_validation_metrics.jsonl').open('x', encoding='utf-8') as log:
            for row in rows:
                log.write(json.dumps(row, allow_nan=False) + '\n')
    overall_neighbor = summarize(rows_neighbor)
    overall_control = summarize(rows_control)
    by_group = {str(g): summarize([r for r in rows_neighbor if r['support_group'] == g])
                for g in (4, 6)}
    by_group_control = {str(g): summarize([r for r in rows_control if r['support_group'] == g])
                        for g in (4, 6)}
    by_duration = {str(d): summarize([r for r in rows_neighbor if r['duration_days'] == d])
                   for d in (7, 14)}
    q_new = overall_neighbor['residual']['q']['profile_macro_nrmse']['sensor_macro']
    o_new = overall_neighbor['residual']['o']['profile_macro_nrmse']['sensor_macro']
    previous = json.loads((BASE_RUN / 'result.json').read_text(encoding='utf-8'))
    baseline = {channel: previous['overall_raw']['residual'][channel]
                ['profile_macro_nrmse']['sensor_macro'] for channel in ('q', 'o')}
    candidate = {'q': q_new, 'o': o_new}
    control_nrmse = {channel: overall_control['residual'][channel]
                     ['profile_macro_nrmse']['sensor_macro'] for channel in ('q', 'o')}
    candidate_mae = {channel: overall_neighbor['residual'][channel]
                     ['raw_mae']['sensor_macro'] for channel in ('q', 'o')}
    control_mae = {channel: overall_control['residual'][channel]
                   ['raw_mae']['sensor_macro'] for channel in ('q', 'o')}
    group_candidate = {group: {channel: by_group[str(group)]['residual'][channel]
                       ['profile_macro_nrmse']['sensor_macro']
                       for channel in ('q', 'o')} for group in (4, 6)}
    group_control = {group: {channel: by_group_control[str(group)]['residual'][channel]
                     ['profile_macro_nrmse']['sensor_macro']
                     for channel in ('q', 'o')} for group in (4, 6)}
    keep = neighbor_candidate_gate(candidate, control_nrmse, baseline,
                                   candidate_mae, control_mae,
                                   group_candidate, group_control)
    result = {'status': 'completed', 'scope': plan['scope'],
              'fit_episodes': 210, 'validation_episodes': 58,
              'overall_neighbor': overall_neighbor, 'overall_control': overall_control,
              'by_group_neighbor': by_group, 'by_group_control': by_group_control,
              'by_duration_neighbor': by_duration,
              'candidate_retained_by_frozen_rule': keep,
              'retained_global_baseline': baseline,
              'candidate_nrmse': candidate,
              'control_nrmse': control_nrmse,
              'candidate_mae': candidate_mae,
              'control_mae': control_mae,
              'candidate_group_nrmse': group_candidate,
              'control_group_nrmse': group_control,
              'candidate_q_relative_reduction_vs_control':
                  1 - q_new / control_nrmse['q'],
              'fit_prepare_seconds': fit_prepare_seconds,
              'neighbor_fit_seconds': raw_seconds,
              'control_fit_seconds': masked_seconds,
              'gpu_peak_reserved_bytes': torch.cuda.max_memory_reserved(),
              'model_parameters': sum(p.numel() for p in raw_model.parameters()),
              'fit_source_accesses_before_validation': fit_access_cut,
              'validation_source_accesses': len(reader.access_log) - fit_access_cut,
              'selection_read': False, 'calibration_read': False, 'test_read': False,
              'neighbor_checkpoint_sha256': file_sha256(args.output / 'neighbor_context.pt'),
              'control_checkpoint_sha256': file_sha256(args.output / 'global_control.pt'),
              'neighbor_metrics_sha256': file_sha256(args.output / 'neighbor_validation_metrics.jsonl'),
              'control_metrics_sha256': file_sha256(args.output / 'control_validation_metrics.jsonl')}
    (args.output / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    if file_sha256(PLAN) != plan['work_order_sha256'] or any(
            file_sha256(ROOT / relative) != digest for relative, digest in plan['code_sha256'].items()):
        raise RuntimeError('work order or code changed during run')
    print(json.dumps({'status': 'completed', 'q': q_new, 'o': o_new,
                      'control_q': control_nrmse['q'],
                      'control_o': control_nrmse['o'],
                      'candidate_retained': keep}), flush=True)


if __name__ == '__main__':
    main()
