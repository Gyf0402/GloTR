"""Retrained ablation arms for the G80 v2 development program (protocol 2.2).

Each arm reproduces the G80 continuation contract exactly — same G40 base
checkpoint (raw_values.pt, SHA-pinned), same 210 fit episodes, 40 epochs,
seed 20260905, AdamW(1e-3, wd=1e-4), dropout .1, batch 8 days, identical
loss — with ONE pathway removed at the residual-head token context:

  full       no change (control; must reproduce global_control.pt bitwise)
  no_attr    static features zeroed inside global_token_context (train+eval)
  no_hops    signed chain-hop feature zeroed (train+eval)
  no_pool    pool + same-slot station key segments pruned (A1: depooling)
  no_fourier Fourier key segment pruned (A1: time-encoding removal)

The frozen core/mu/basis keep their inputs: these arms isolate the marginal
value of each pathway inside the residual head, paired against G80 v1 on the
three already-exposed windows (58 train-holdout / 31 calibration / 21 test).

Stability runs use the pre-registered seed triple SEEDS (default SEED keeps the
v1 bitwise anchor). Stages: train --arm X --seed S -> evaluate --arms a,b,c

NOTE the A1 arms (no_pool / no_fourier) prune key segments inside
global_token_context; the default key_groups reproduces the frozen contract
exactly, so the full arm and every pre-existing arm are bitwise unchanged.
"""
from __future__ import annotations

import os

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
for _var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
             'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_var, '4')

import argparse
import hashlib
import json
import sys
import time
from datetime import date
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[0]
BASE = ROOT / 'reviews/2026-09-20/baseline_takeover'
for _path in (str(ROOT), str(BASE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np
import torch

CHANNELS = ('q', 'o')
BUFFER_DAYS = 3
SLOTS = 288
S3 = np.asarray([6240.0, 0.21276629999999888], dtype=np.float64)
SEED = 20260905
# Pre-registered seed triple for the stability requirement (PROTOCOL v2.2 2.2 /
# proposal 4). SEED is the historical v1 anchor and must keep its value so the
# `full` arm stays bitwise-equal to the frozen v1 checkpoint.
SEEDS = (20260905, 20260906, 20260907)
EPOCHS = 40
BATCH = 8

# Arms differ only by which key segments reach the residual-head attention and
# by which token-context pathway is zeroed. `full`, `no_pool`, `no_fourier`
# change no inputs (static/hops) — they prune the global key set only.
ARM_KEY_GROUPS = {
    'full': ('local', 'pool', 'station', 'fourier'),
    'no_attr': ('local', 'pool', 'station', 'fourier'),
    'no_hops': ('local', 'pool', 'station', 'fourier'),
    'no_pool': ('local', 'fourier'),
    'no_fourier': ('local', 'pool', 'station'),
    # A6 head-variant arms (see reviews/2026-09-26/A6_PREREGISTRATION.md). They
    # keep the four key segments and differ only in the q/o read-out head.
    'a6_split': ('local', 'pool', 'station', 'fourier'),
    'a6_wide_shared': ('local', 'pool', 'station', 'fourier'),
    'a6_split_matched': ('local', 'pool', 'station', 'fourier'),
    # Occupancy output-transform arms (see
    # reviews/2026-09-26/OCCUPANCY_TRANSFORM_PREREGISTRATION.md). Same four key
    # segments; only the o channel's correction magnitude is reshaped.
    'ot_identity': ('local', 'pool', 'station', 'fourier'),
    'ot_gain': ('local', 'pool', 'station', 'fourier'),
    'ot_bound': ('local', 'pool', 'station', 'fourier'),
    # Loss-objective arm (see reviews/2026-09-26/LOSS_OBJECTIVE_PREREGISTRATION.md).
    # Identical input, architecture, initialisation and schedule to ``full``; the
    # only difference is the training loss (see HUBER_DELTA below).
    'huber90': ('local', 'pool', 'station', 'fourier'),
}

#: Huber threshold per channel for the ``huber90`` arm, in normalised units.
#: Pre-registered rule: the 90th percentile of |residual| of the *control* arm
#: ``full@20260905`` over the Fit-210 scored cells, read from
#: ``reviews/2026-09-26/dual_channel_probe/probe_matrix.npz`` (q: P90 0.09362,
#: o: P90 0.15519). Frozen before the arm was trained; the same percentile rule
#: is applied to each channel, and the channel aggregate stays an unweighted mean.
HUBER_DELTA = {'q': 0.09362, 'o': 0.15519}
ARMS = tuple(ARM_KEY_GROUPS)

G40_DIR = 'reviews/2026-09-24/sparse_global_token_full_v1'
G80_DIR = 'reviews/2026-09-24/sparse_neighbor_context_full_v1'
G40_CK = f'{G40_DIR}/raw_values.pt'
G40_CK_SHA = 'c4adb3dbc6ed0968860546372daaf5e042249f271ddbafbf0fb8d4e737f80332'
G80_CK = f'{G80_DIR}/global_control.pt'
G80_CK_SHA = 'd60e585ca94793aa903c5dda5c5306393e2bb20acc138c97393f8ddf20785659'
STAGE3_BASIS = 'reviews/2026-09-23/optimization_v1/stage3_run_v1/basis.pt'
STAGE3_BASIS_SHA = '79c0f3e4f3cd04138b0b86126609fd56578108c9fb0ae20a209e05d237891e9f'
STAGE3_MU = 'reviews/2026-09-23/optimization_v1/stage3_run_v1/mu.pt'
STAGE3_MU_SHA = 'ec4e8fa8ee2a51c37ed4a6e39f59af88d6ac57adae54558148ca5d15020fe7fe'
STAGE3_RESULT = 'reviews/2026-09-23/optimization_v1/stage3_run_v1/result.json'
STAGE3_RESULT_SHA = '21864f1313d6ade9a8483375a049e7a94a54bd7e66c1165560d5fddfc4941eda'
STAGE3_PLAN = 'reviews/2026-09-23/optimization_v1/stage3_run_v1/plan.json'
STAGE4_CORE = 'reviews/2026-09-23/optimization_v1/stage4_run_v1/core/student.pt'
STAGE4_CORE_SHA = '261ef98f7371f1a5c8d4f8883f94d8fb8a39d3acd121f1f80d99cdaae09c7ce5'
STAGE4_RESULT = 'reviews/2026-09-23/optimization_v1/stage4_run_v1/result.json'
STAGE4_RESULT_SHA = '75fb83873b0fed427173ff8bf2b0ff72bfb4f9474375a631603302790087a87e'
RELEASE_REL = 'reviews/2026-09-12/data_release_final/QO_FORMAL_RELEASE.json'
RELEASE_SHA = '5c6dd879beb0f70bcd2388601736cbf0bf341f111a267f79801c478e780b9287'
ROSTER_REL = 'reviews/2026-09-12/formal_mask_bank/EPISODE_ROSTER_NON_TEST.json'
ROSTER_SHA = 'cb38dca7027a8bc087290b01cf61e6f6f2550ddbc2f2a4f4a92b807e8813bbee'
GRAPH_REL = 'data/pems/graphs/qo_pilot_v1/release_v1/graph.json'
GRAPH_SHA = '6373d9a58173b4e828fced08b57f944a1ba78ac8733125e1ee100c2f5a034eb4'
SPLIT_REL = 'reviews/2026-09-23/optimization_v1/split.json'
SPLIT_SHA = '7dd2d8a170c3e0d9e8710888ced50ca2638a9c1cc95a33d5e9a214b2992e789e'
# Frozen Selection-42 list for the A6 phase (see
# scripts/build_a6_selection_manifest.py and reviews/2026-09-26/A6_PREREGISTRATION.md).
SELECTION_MANIFEST_REL = 'reviews/2026-09-26/a6/selection_manifest.json'
SELECTION_MANIFEST_SHA = '90fc66c64d4ac58e2f9dcd3ce189cfae9db82098549b47231740ab95b9c06863'
REG2_REL = 'runs/current_student_train_preparation_v2/registration.json'
CODE_FILES = (
    'scripts/train_g80_v2_arm.py',
    'scripts/fit_sparse_neighbor_context.py',
    'scripts/fit_profile_residual_candidate.py',
    'models/sparse_global_tokens.py',
    'models/sparse_neighbor_context.py',
    'models/sparse_observation_profile.py',
    'training/current_release_student_data.py',
    'training/pilot_data.py',
    'training/pilot_engine.py',
    'training/real_student_adapter.py',
    'training/qo_attributes.py',
    'representations/basis.py',
    'models/film_rgnn.py',
)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def setup_torch():
    if not torch.cuda.is_available():
        raise RuntimeError('single GPU required by the frozen contract')
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def sha_file(path: Path) -> str:
    return sha(path)


# ---------------------------------------------------------------------------
# shared loaders (mirrors of fit_sparse_neighbor_context.prepare_one)
# ---------------------------------------------------------------------------

ATTRIBUTES = None


def load_models(scales):
    global ATTRIBUTES
    from representations.basis import GlobalProfileBasis, MeanFunction
    from training.pilot_engine import make_student
    from training.qo_attributes import AnnualAttributes
    cfg = load_json(ROOT / REG2_REL)['config']
    ATTRIBUTES = AnnualAttributes.from_files(
        ROOT / 'data/pems/releases/partition_v1',
        manifest_sha256=cfg['attribute_manifest_sha256'])
    basis_ck = torch.load(ROOT / STAGE3_BASIS, map_location='cpu',
                          weights_only=False)
    basis = GlobalProfileBasis(SLOTS, 2, 16).requires_grad_(False).eval()
    basis.load_state_dict(basis_ck['basis_state'])
    mu_ck = torch.load(ROOT / STAGE3_MU, map_location='cpu',
                       weights_only=False)
    if not np.array_equal(np.asarray(scales), np.asarray(mu_ck['scales'])):
        raise ValueError('stage-3 scales differ from mu checkpoint')
    mu = MeanFunction(5, 4, 6, channels=2).requires_grad_(False).eval()
    mu.load_state_dict(mu_ck['mean_state'])
    core_state = torch.load(ROOT / STAGE4_CORE, map_location='cpu',
                            weights_only=False)
    core = make_student(rank=16, hidden=32).requires_grad_(False).eval()
    core.load_state_dict(core_state['student'])
    return basis, mu, core


def prepare_item(episode, row, basis, mu, core, scales, edges, device):
    """Mirror of prepare_one (train-role geometry checks, base decode)."""
    from scripts.fit_sparse_neighbor_context import PreparedEpisode, signed_chain_hops
    from training.current_release_student_data import predict_case
    scales_t = torch.tensor(scales, dtype=torch.float32)
    features = np.asarray(
        [[ATTRIBUTES.lookup(day, station)['features']
          for station in episode.station_order] for day in episode.dates],
        dtype=np.float32)
    if not np.array_equal(features,
                          np.broadcast_to(features[0], features.shape)):
        raise ValueError('date-varying static features unsupported')
    from training.pilot_data import calendar_features, rolling_history
    from training.real_student_adapter import build_student_inputs
    device_cpu = next(basis.parameters()).device
    with torch.no_grad():
        values = torch.as_tensor(episode.values, dtype=torch.float32,
                                 device=device_cpu) / scales_t
        visible = torch.as_tensor(episode.visible, device=device_cpu)
        values = torch.where(visible, values, torch.zeros_like(values))
        history = rolling_history(values, visible)
        calendar = calendar_features(episode.dates).to(device_cpu)
        per_day_static = torch.from_numpy(features[0])[None].expand(
            len(episode.dates), -1, -1)
        mean = basis.global_mean + mu(per_day_static, calendar, history,
                                      protocol='K')
        inputs = build_student_inputs(basis, values, visible, mean,
                                      torch.from_numpy(features[0]), edges,
                                      episode.dates)
        scoring = torch.as_tensor(episode.scoring, device=device_cpu)
        reference = torch.as_tensor(episode.reference, dtype=torch.float32,
                                    device=device_cpu) / scales_t
        selected = scoring.any(dim=(-1, -2))
        geometry = torch.zeros_like(selected)
        duration = row['duration_days']
        geometry[BUFFER_DAYS:BUFFER_DAYS + duration, episode.target_index] = True
        if not torch.equal(selected, geometry):
            raise ValueError('target support differs from outage geometry')
        coefficients = predict_case(core, {'inputs': inputs,
                                           'static': torch.from_numpy(
                                               features[0]),
                                           'ordinals': torch.tensor(
                                               [date.fromisoformat(d).toordinal()
                                                for d in episode.dates],
                                               device=device_cpu)})
        base = basis.decode(coefficients[selected], inputs.mean[selected])
    return PreparedEpisode(
        episode.episode_id, episode.mask_id, str(row['station_id']), duration,
        episode.target_index, None,
        torch.as_tensor(episode.values, dtype=torch.float32, device=device),
        torch.as_tensor(episode.visible, dtype=torch.bool, device=device),
        torch.from_numpy(features[0]).to(device),
        inputs.calendar.to(device), base.to(device),
        reference[selected].to(device), scoring[selected].to(device),
        signed_chain_hops(edges, target=episode.target_index,
                          nodes=len(episode.station_order)).to(device))


def context_hook(arm, item):
    """Return (static, hops) as seen by global_token_context for this arm."""
    if arm == 'no_attr':
        return torch.zeros_like(item.static), item.hops
    if arm == 'no_hops':
        return item.static, torch.zeros_like(item.hops)
    if arm in ('full', 'no_pool', 'no_fourier', 'a6_split',
               'a6_wide_shared', 'a6_split_matched', 'ot_identity',
               'ot_gain', 'ot_bound', 'huber90'):
        return item.static, item.hops
    raise ValueError(f'unknown arm {arm}')


def arm_key_groups(arm):
    """Key segments admitted to the residual-head attention for this arm."""
    if arm not in ARM_KEY_GROUPS:
        raise ValueError(f'unknown arm {arm}')
    return ARM_KEY_GROUPS[arm]


def weighted_day_loss(predicted, truth, scoring, *, duration, episode_count,
                      delta=None):
    """Per-channel / per-day data term.

    ``delta=None`` (the default, and every pre-existing arm) delegates straight
    to the frozen L1 implementation, so those arms stay bitwise unchanged. When
    ``delta`` is given, the same reduction is applied to a Huber loss whose
    per-channel threshold is ``delta``: quadratic up to the threshold, linear
    beyond it. Identical shape rule for both channels; the aggregate remains an
    unweighted mean over channels.
    """
    if delta is None:
        from scripts.fit_sparse_neighbor_context import weighted_day_loss as wdl
        return wdl(predicted, truth, scoring, duration=duration,
                   episode_count=episode_count)
    parts = []
    for index, threshold in enumerate(delta):
        error = (predicted[:, index] - truth[:, index])[scoring[:, index]]
        magnitude = error.abs()
        parts.append(torch.where(magnitude <= threshold,
                                 .5 * error * error,
                                 threshold * (magnitude - .5 * threshold)).mean())
    return torch.stack(parts).mean() / (duration * episode_count)


# ---------------------------------------------------------------------------
# train stage
# ---------------------------------------------------------------------------

def stage_train(out: Path, arm: str, seed: int = SEED) -> None:
    from models.sparse_global_tokens import GlobalTokenProfile, global_token_context
    from training.current_release_student_data import (VerifiedRoleReader,
                                                       assemble_episode)
    key_groups = arm_key_groups(arm)
    split = load_json(ROOT / SPLIT_REL)
    if (split['validation_group_indices'] != [4, 6] or
            len(split['fit_ids']) != 210 or len(split['validation_ids']) != 58):
        raise ValueError('frozen Train split differs')
    stage3 = load_json(ROOT / STAGE3_RESULT)
    stage3_plan = load_json(ROOT / STAGE3_PLAN)
    stage4 = load_json(ROOT / STAGE4_RESULT)
    if (len(stage3_plan['fit_days']) != 171 or
            len(stage3_plan['excluded_days']) != 48 or
            set(stage3_plan['fit_days']) & set(stage3_plan['excluded_days']) or
            stage4['candidate_retained_by_frozen_rule'] is not True):
        raise ValueError('frozen stage gates differ')
    for rel, expected in ((G40_CK, G40_CK_SHA), (STAGE3_BASIS, STAGE3_BASIS_SHA),
                          (STAGE3_MU, STAGE3_MU_SHA), (STAGE4_CORE,
                                                       STAGE4_CORE_SHA),
                          (RELEASE_REL, RELEASE_SHA), (GRAPH_REL, GRAPH_SHA),
                          (ROSTER_REL, ROSTER_SHA), (SPLIT_REL, SPLIT_SHA),
                          (STAGE3_RESULT, STAGE3_RESULT_SHA),
                          (STAGE4_RESULT, STAGE4_RESULT_SHA)):
        if sha(ROOT / rel) != expected:
            raise ValueError(f'pin drift {rel}')
    if sha(ROOT / G80_CK) != G80_CK_SHA:
        raise ValueError('v1 checkpoint drift')
    setup_torch()
    device = torch.device('cuda:0')
    scales = tuple(float(x) for x in stage3['scales'])
    graph = load_json(ROOT / GRAPH_REL)
    order = tuple(str(x) for x in graph['station_order'])
    edges = {name: torch.tensor(value, dtype=torch.long)
             for name, value in graph['edge_index'].items()}
    from scripts.fit_sparse_neighbor_context import signed_chain_hops
    signed_chain_hops(edges, target=0, nodes=20)
    basis, mu, core = load_models(scales)
    roster = {row['episode_id']: row for row in
              load_json(ROOT / ROSTER_REL)['episodes']}
    reader = VerifiedRoleReader(ROOT, ROOT / RELEASE_REL, RELEASE_SHA,
                                'train', order)

    @lru_cache(maxsize=240)
    def read(day):
        return reader(day)

    fit_days = set(stage3_plan['fit_days'])
    print(json.dumps({'phase': 'prepare_fit', 'episodes': 210}), flush=True)
    started = time.perf_counter()
    fit_items = []
    for identity in split['fit_ids']:
        row = roster[identity]
        episode = assemble_episode(row, order, reader.bounds, read)
        if any(day not in fit_days for day in episode.dates):
            raise ValueError(f'episode crosses fit boundary: {identity}')
        fit_items.append(prepare_item(episode, row, basis, mu, core, scales,
                                      edges, device))
    prepare_seconds = time.perf_counter() - started
    days = [(i, day) for i, item in enumerate(fit_items)
            for day in range(BUFFER_DAYS, BUFFER_DAYS + item.duration)]

    torch.manual_seed(seed)
    source = torch.load(ROOT / G40_CK, map_location='cpu', weights_only=False)
    # A6 integration: build_profile returns the frozen class for every
    # pre-existing arm, and initial_state is the identity for them, so those
    # arms stay bitwise unchanged. Variant arms get their head transplanted
    # from the same shared state (see models/head_variant_profile.py).
    from models.arm_profiles import build_profile, initial_state
    model = build_profile(arm, dropout=.1).to(device)
    model.load_state_dict(initial_state(arm, source['model']), strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    # Only the loss-objective arm departs from the frozen L1 data term.
    loss_delta = (tuple(HUBER_DELTA[channel] for channel in CHANNELS)
                  if arm == 'huber90' else None)
    metrics_path = out / f'{arm}_train_metrics.jsonl'
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open('x', encoding='utf-8') as log:
        for epoch in range(EPOCHS):
            model.train()
            epoch_started = time.perf_counter()
            shuffle = torch.randperm(
                len(days),
                generator=torch.Generator().manual_seed(seed + epoch)).tolist()
            epoch_loss = 0.
            for offset in range(0, len(days), BATCH):
                selected_days = [days[j] for j in shuffle[offset:offset + BATCH]]
                contexts = []
                for item_index, day in selected_days:
                    item = fit_items[item_index]
                    static_c, hops_c = context_hook(arm, item)
                    common = dict(values=item.values, visible=item.visible,
                                  base=item.base[day - 3], static=static_c,
                                  calendar=item.calendar, target=item.target,
                                  day=day, signed_hops=hops_c, scales=scales)
                    q, k, valid = global_token_context(**common,
                                                       raw_values=True,
                                                       key_groups=key_groups)
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
                    part = slice(j * SLOTS, (j + 1) * SLOTS)
                    loss = loss + weighted_day_loss(
                        predicted[part], item.reference[day - 3],
                        item.scoring[day - 3], duration=item.duration,
                        episode_count=len(fit_items), delta=loss_delta)
                    loss = loss + .001 * correction[part].square().mean() / (
                        item.duration * len(fit_items))
                if not torch.isfinite(loss):
                    raise ValueError('nonfinite attention loss')
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.detach())
            duration_s = time.perf_counter() - epoch_started
            log.write(json.dumps({'epoch': epoch + 1, 'loss': epoch_loss,
                                  'seconds': duration_s}) + '\n')
            log.flush()
            if epoch == 0 or (epoch + 1) % 10 == 0:
                print(json.dumps({'arm': arm, 'epoch': epoch + 1,
                                  'loss': epoch_loss}), flush=True)
    model.requires_grad_(False).eval()
    ckpt_path = out / f'{arm}.pt'
    with ckpt_path.open('xb') as stream:
        torch.save({'model': model.state_dict(), 'arm': arm, 'seed': seed,
                    'key_groups': key_groups, 'epochs': EPOCHS}, stream)
    control_match = None
    if arm == 'full' and seed == SEED:
        v1 = torch.load(ROOT / G80_CK, map_location='cpu', weights_only=False)
        control_match = bool(all(
            torch.equal(model.state_dict()[k], v1['model'][k].cpu())
            for k in v1['model']))
    record = {'arm': arm, 'seed': seed, 'key_groups': list(key_groups),
              'checkpoint_sha256': sha(ckpt_path),
              'prepare_seconds': prepare_seconds,
              'bitwise_equal_to_v1': control_match,
              'note': 'contract replication of fit_sparse_neighbor_context '
                      'control arm; hooks only in token context'}
    write_json_atomic = out / f'{arm}_train_record.json'
    with write_json_atomic.open('w', encoding='utf-8') as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2)
    print(json.dumps(record), flush=True)


# ---------------------------------------------------------------------------
# evaluate stage
# ---------------------------------------------------------------------------

def build_queues(order):
    from training.current_release_student_data import (VerifiedRoleReader,
                                                       assemble_episode)
    from scripts.run_g80_independent_confirmation import (
        SealedRoleReader, assemble_sealed_episode)
    from scripts.run_g80_test import TestRoleReader, assemble_test_episode
    roster = {row['episode_id']: row for row in
              load_json(ROOT / ROSTER_REL)['episodes']}
    queues = {}
    split = load_json(ROOT / SPLIT_REL)
    train_reader = VerifiedRoleReader(ROOT, ROOT / RELEASE_REL, RELEASE_SHA,
                                      'train', order)
    queues['train_holdout_58'] = {
        'rows': [roster[i] for i in split['validation_ids']],
        'reader': train_reader,
        'assemble': lambda row, read: assemble_episode(
            row, order, train_reader.bounds, read)}
    confirm_plan = load_json(ROOT / 'reviews/2026-09-25/'
                             'g80_independent_confirmation_v1/plan.json')
    calib_reader = SealedRoleReader(ROOT, ROOT / RELEASE_REL, RELEASE_SHA,
                                    order)
    queues['calibration_31'] = {
        'rows': [roster[i] for i in
                 confirm_plan['sealed_set']['episode_ids']],
        'reader': calib_reader,
        'assemble': lambda row, read: assemble_sealed_episode(
            row, order, calib_reader.bounds, read)}
    test_manifest = load_json(ROOT / 'reviews/2026-09-25/g80_test_v1/'
                              'test_manifest.json')
    test_reader = TestRoleReader(ROOT, ROOT / RELEASE_REL, order,
                                 set(test_manifest['allowed_days']))
    queues['test_21'] = {
        'rows': [roster[e['episode_id']] for e in
                 test_manifest['candidates']],
        'reader': test_reader,
        'assemble': lambda row, read: assemble_test_episode(
            row, order, test_reader.bounds, read)}
    # Selection queue for the A6 phase. The list is a frozen manifest (built by
    # scripts/build_a6_selection_manifest.py from the rule below), because A6
    # selection must happen on Selection and Test21 may only be displayed
    # post hoc. Re-deriving the rule here keeps the manifest honest: a drifted
    # or hand-edited list fails loudly instead of quietly changing the set.
    manifest = load_json(ROOT / SELECTION_MANIFEST_REL)
    if sha(ROOT / SELECTION_MANIFEST_REL) != SELECTION_MANIFEST_SHA:
        raise ValueError('selection manifest SHA drift; refusing to evaluate')
    derived = sorted(
        row['episode_id'] for row in roster.values()
        if row['split'] == 'selection' and str(row['station_id']) in set(order))
    if sorted(manifest['episode_ids_in_order']) != derived:
        raise ValueError('selection manifest disagrees with the derivation '
                         'rule; refusing to evaluate')
    sel_reader = VerifiedRoleReader(ROOT, ROOT / RELEASE_REL, RELEASE_SHA,
                                    'selection', order)
    queues['selection_42'] = {
        'rows': [roster[i] for i in manifest['episode_ids_in_order']],
        'reader': sel_reader,
        'assemble': lambda row, read: assemble_episode(
            row, order, sel_reader.bounds, read)}
    return queues


def v1_reference(queue_name, identity, duration, target):
    """Archived v1 prediction (normalized) for bitwise consistency checks."""
    if queue_name == 'train_holdout_58':
        for path in sorted((ROOT / G80_DIR / 'validation').glob('*.npz')):
            with np.load(path, allow_pickle=False) as archive:
                if str(archive['episode_id']) == identity:
                    return archive['control'], 'norm'
        raise ValueError(f'no archived prediction for {identity}')
    if queue_name == 'calibration_31':
        plan = load_json(ROOT / 'reviews/2026-09-25/'
                         'g80_independent_confirmation_v1/plan.json')
        index = plan['sealed_set']['episode_ids'].index(identity)
        with np.load(ROOT / 'reviews/2026-09-25/'
                     'g80_independent_confirmation_v1/'
                     f'case_{index:03d}.npz', allow_pickle=False) as archive:
            return archive['pred_g80'], 'norm'
    if queue_name == 'test_21':
        manifest = load_json(ROOT / 'reviews/2026-09-25/g80_test_v1/'
                             'test_manifest.json')
        index = [e['episode_id'] for e in
                 manifest['candidates']].index(identity)
        with np.load(ROOT / 'reviews/2026-09-25/g80_test_v1/predictions/'
                     f'g80/case_{index:03d}.npz',
                     allow_pickle=False) as archive:
            return archive['pred_g80_phys'], 'phys'
    # No archived v1 prediction exists for this queue (selection_42 has none),
    # so the bitwise anchor check cannot be applied here. Returning None is
    # explicit; the caller records it instead of silently skipping.
    return None, 'none'


def v1_matches(queue_name, pred_norm32, reference, space):
    if space == 'phys':
        # physical archive; compare in normalized float32 space (f64 physical
        # roundtrip differs by <=1 ulp, see run notes)
        return bool(np.array_equal(pred_norm32.numpy(),
                                   (np.asarray(reference, dtype=np.float64) /
                                    S3).astype(np.float32)))
    return bool(np.array_equal(pred_norm32.numpy(), reference))


def stage_evaluate(out: Path, arms: str, queues_only: str = '') -> None:
    arm_list = [a for a in arms.split(',') if a]
    if 'full' not in arm_list:
        arm_list = ['full'] + arm_list
    setup_torch()
    device = torch.device('cuda:0')
    scales = (6240.0, 0.21276629999999888)
    graph = load_json(ROOT / GRAPH_REL)
    order = tuple(str(x) for x in graph['station_order'])
    edges = {name: torch.tensor(value, dtype=torch.long)
             for name, value in graph['edge_index'].items()}
    basis, mu, core = load_models(scales)
    from models.sparse_global_tokens import global_token_context
    queues = build_queues(order)
    if queues_only:
        wanted = [name for name in queues_only.split(',') if name]
        unknown = [name for name in wanted if name not in queues]
        if unknown:
            raise ValueError(f'unknown queue(s) {unknown}; known {list(queues)}')
        queues = {name: queues[name] for name in wanted}

    checkpoints = {}
    seeds = {}
    for arm in arm_list:
        path = out / f'{arm}.pt'
        if not path.exists():
            raise RuntimeError(f'missing checkpoint {path}')
        payload = torch.load(path, map_location='cpu', weights_only=False)
        from models.arm_profiles import build_profile
        model = build_profile(arm, dropout=.1).requires_grad_(False).eval()
        model.load_state_dict(payload['model'], strict=True)
        model.to(device)
        checkpoints[arm] = model
        seeds[arm] = int(payload['seed'])
    if len(set(seeds.values())) != 1:
        raise ValueError(f'arms trained under different seeds: {seeds}')
    run_seed = seeds[arm_list[0]]
    anchor_expected = run_seed == SEED
    anchor_by_queue = {}

    results = {}
    for queue_name, queue in queues.items():
        anchor_here = anchor_expected
        queue_rows = []
        for row in queue['rows']:
            identity = row['episode_id']
            episode = queue['assemble'](row, lru_cache(maxsize=64)(
                lambda day: queue['reader'](day)))
            if np.any(episode.visible & episode.scoring):
                raise ValueError('score visible leak')
            duration = row['duration_days']
            target = episode.target_index
            scoring = episode.scoring[BUFFER_DAYS:BUFFER_DAYS + duration,
                                      target]
            reference_norm = (episode.reference[
                BUFFER_DAYS:BUFFER_DAYS + duration,
                target] / S3).astype(np.float32)
            item = prepare_item(episode, row, basis, mu, core, scales, edges,
                                device)
            scores = {}
            occupancy = {}
            for arm in arm_list:
                static_c, hops_c = context_hook(arm, item)
                key_groups = arm_key_groups(arm)
                profiles = []
                with torch.no_grad():
                    for day in range(BUFFER_DAYS, BUFFER_DAYS + duration):
                        q, k, valid = global_token_context(
                            values=item.values, visible=item.visible,
                            base=item.base[day - 3], static=static_c,
                            calendar=item.calendar, target=item.target,
                            day=day, signed_hops=hops_c, scales=scales,
                            raw_values=True, key_groups=key_groups)
                        predicted, _ = checkpoints[arm](
                            q, k, valid, item.base[day - 3])
                        profiles.append(predicted.cpu())
                pred = torch.stack(profiles)
                if arm == 'full' and anchor_here:
                    ref_array, space = v1_reference(queue_name, identity,
                                                    duration, target)
                    if ref_array is None:
                        anchor_here = False
                    elif not v1_matches(queue_name, pred, ref_array, space):
                        raise RuntimeError(
                            f'full arm does not reproduce archived v1 '
                            f'prediction for {identity}')
                scores[arm] = score_physical(pred.numpy(), reference_norm,
                                             scoring)
                occupancy[arm] = occupancy_error(pred.numpy(), reference_norm,
                                                 scoring)
            queue_rows.append({'episode_id': identity,
                               'station_id': str(row['station_id']),
                               'duration_days': duration,
                               'start_day': row['start_day'], 'scores': scores,
                               'occupancy': occupancy})
        anchor_by_queue[queue_name] = bool(anchor_here)
        results[queue_name] = summarize(queue_rows, arm_list)
        print(json.dumps({'queue': queue_name, 'arms': arm_list,
                          'n': len(queue_rows)}), flush=True)

    payload = {
        'status': 'completed',
        'label': 'retrained ablation arms vs full control (v1-anchored)',
        'seed': run_seed,
        'arms': arm_list,
        'key_groups': {arm: list(arm_key_groups(arm)) for arm in arm_list},
        'v1_anchor_check_applied': bool(anchor_expected and anchor_by_queue
                                        and all(anchor_by_queue.values())),
        'v1_anchor_check_by_queue': anchor_by_queue,
        'queues': results,
        'note': 'full-arm predictions bitwise-equal the archived v1 '
                'predictions on every evaluated case where an archive exists; '
                'other arms differ only by their token-context hook during '
                'training and inference. selection_42 has no archived v1 '
                'prediction, so its anchor check is recorded as not applied.',
    }
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'retrained_arms_result.json').open('w', encoding='utf-8') as s:
        json.dump(payload, s, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({'status': 'completed', 'queues':
                      list(results)}), flush=True)


#: True-occupancy intervals for the interpretation-only error decomposition
#: (same edges as the S1-2 diagnostics, so the two readouts are comparable).
OCC_BINS = ((0.0, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 0.30), (0.30, 1.01))
OCC_LABELS = ('[0,.05)', '[.05,.10)', '[.10,.20)', '[.20,.30)', '[.30,1]')


def occupancy_error(pred_norm32, truth_norm32, scoring):
    """Pooled normalized error inside true-occupancy intervals.

    Interpretation only: the bins are keyed on the TRUE occupancy (channel 1),
    which is unavailable at inference, so this cannot be a selection criterion.
    It exists so a candidate's high-occupancy behaviour can be read next to its
    NRMSE instead of being inferred from the aggregate. Accumulators are sums so
    the caller can pool whole queues without losing precision.
    """
    # physical occupancy: the S1-2 intervals are stated on the physical scale
    occ = truth_norm32[..., 1] * float(S3[1])
    out = {}
    for c, channel in enumerate(CHANNELS):
        active = scoring[..., c]
        error = np.where(active, pred_norm32[..., c].astype(np.float64) -
                         truth_norm32[..., c].astype(np.float64), 0.0)
        out[channel] = {}
        for label, (lo, hi) in zip(OCC_LABELS, OCC_BINS):
            keep = active & (occ >= lo) & (occ < hi)
            n = int(keep.sum())
            out[channel][label] = {
                'n': n,
                'sq': float(np.square(error[keep]).sum()) if n else 0.0,
                'sum': float(error[keep].sum()) if n else 0.0}
    return out


def score_physical(pred_norm32, truth_norm32, scoring):
    scores = {}
    for c, channel in enumerate(CHANNELS):
        active = scoring[..., c]
        count = active.sum(axis=-1)
        if not np.all(count > 0):
            raise ValueError('empty scoring profile')
        error = np.where(active, pred_norm32[..., c].astype(np.float64) -
                         truth_norm32[..., c].astype(np.float64), 0.0)
        profile_rmse = np.sqrt(np.square(error).sum(axis=-1) / count)
        scores[channel] = {
            'profile_macro_nrmse': float(profile_rmse.mean()),
            'raw_mae': float(np.abs(error[active]).mean() * float(S3[c])),
        }
    return scores


def sensor_macro(rows, arm, channel, metric):
    grouped = {}
    for row in rows:
        grouped.setdefault(row['station_id'], []).append(
            row['scores'][arm][channel][metric])
    return float(sum(sum(v) / len(v) for v in grouped.values()) /
                 len(grouped))


def episode_macro(rows, arm, channel, metric):
    return float(sum(r['scores'][arm][channel][metric] for r in rows) /
                 len(rows))


def summarize(rows, arms):
    table = {'n_cases': len(rows)}
    for arm in arms:
        table[arm] = {}
        for channel in CHANNELS:
            table[arm][channel] = {
                metric: {'episode_macro': episode_macro(rows, arm, channel,
                                                        metric),
                         'sensor_macro': sensor_macro(rows, arm, channel,
                                                      metric)}
                for metric in ('profile_macro_nrmse', 'raw_mae')}
    # Pooled-by-cell occupancy decomposition (interpretation only).
    occupancy = {}
    for arm in arms:
        occupancy[arm] = {}
        for channel in CHANNELS:
            occupancy[arm][channel] = {}
            for label in OCC_LABELS:
                cells = sum(r['occupancy'][arm][channel][label]['n']
                            for r in rows)
                square = sum(r['occupancy'][arm][channel][label]['sq']
                             for r in rows)
                total = sum(r['occupancy'][arm][channel][label]['sum']
                            for r in rows)
                occupancy[arm][channel][label] = {
                    'n_cells': cells,
                    'rmse': float(np.sqrt(square / cells)) if cells else None,
                    'bias': float(total / cells) if cells else None}
    table['occupancy_bins'] = occupancy
    table['occupancy_note'] = ('binned on TRUE occupancy (unavailable at '
                               'inference); pooled by cell, interpretation only')
    deltas = {}
    for arm in arms:
        if arm == 'full':
            continue
        deltas[arm] = {}
        for channel in CHANNELS:
            diffs = np.array([r['scores'][arm][channel]['profile_macro_nrmse']
                              - r['scores']['full'][channel][
                                  'profile_macro_nrmse'] for r in rows])
            deltas[arm][channel] = {
                'mean_diff': float(diffs.mean()),
                'improved_cases': int((diffs < 0).sum()),
                'worsened_cases': int((diffs > 0).sum())}
    table['paired_vs_full'] = deltas
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['train', 'evaluate'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--arm', choices=ARMS)
    parser.add_argument('--arms', default='full,no_attr,no_hops')
    parser.add_argument('--seed', type=int, default=SEED,
                        help=f'pre-registered seed, one of {SEEDS}')
    parser.add_argument('--queues', default='',
                        help='comma-separated evaluation queues to restrict to '
                             '(default: all). Use to evaluate Selection only '
                             'during the A6 phase instead of recomputing the '
                             'three round-2 windows.')
    args = parser.parse_args()
    if args.seed not in SEEDS:
        parser.error(f'--seed must be one of the pre-registered seeds {SEEDS}')
    bad = [a for a in args.arms.split(',') if a and a not in ARMS]
    if bad:
        parser.error(f'unknown arms {bad}; known arms {ARMS}')
    if args.stage == 'train':
        if not args.arm:
            parser.error('--arm is required for train')
        stage_train(args.output, args.arm, args.seed)
    else:
        stage_evaluate(args.output, args.arms, args.queues)


if __name__ == '__main__':
    main()
