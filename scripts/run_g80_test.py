"""G80 Test main comparison and core-only ablation (work order v2).

Implements reviews/2026-09-25/G80_TEST_HANDOFF_WORK_ORDER.md.
Main arms: g80 (frozen) and rttc_v7 (candidate 0, fixed config). Ablation:
core_only (the mu+Bhat base profile saved inside the same G80 forward pass).
mu_only appendix is disabled; G40 is not run. The shared loaders hard-close
the Test role, so this runner carries its own restricted Test reader and
episode assembler mirroring the shared implementations.

Stages: draft (takeover checks, no Test arrays) -> preflight (exposed-case
bitwise replication, tolerance freeze, immutable protocol.json) -> bind
(source digests, test_manifest.json) -> freeze (final plan.json) -> predict
(serial, technical status only) -> score (unified scoring after every
registered prediction exists).
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
import subprocess
import sys
import threading
import time
import traceback
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[0]
BASE = ROOT / 'reviews/2026-09-20/baseline_takeover'
RTTC_REPAIR = ROOT / 'reviews/2026-09-21/rttc_repair'
TOOLS = ROOT / 'reviews/2026-09-19/ai_handoff_diagnosis_20260919T051738Z/tools'
for _path in (str(ROOT), str(BASE), str(RTTC_REPAIR), str(TOOLS)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np
import psutil
import torch

WORK_ORDER_REL = 'reviews/2026-09-25/G80_TEST_HANDOFF_WORK_ORDER.md'
CANDIDATES_REL = 'reviews/2026-09-25/G80_TEST_CANDIDATES_METADATA.json'
G80_CK = 'reviews/2026-09-24/sparse_neighbor_context_full_v1/global_control.pt'
G80_CK_SHA = 'd60e585ca94793aa903c5dda5c5306393e2bb20acc138c97393f8ddf20785659'
G80_DIR = 'reviews/2026-09-24/sparse_neighbor_context_full_v1'
G40_DIR = 'reviews/2026-09-24/sparse_global_token_full_v1'
STAGE3_BASIS = 'reviews/2026-09-23/optimization_v1/stage3_run_v1/basis.pt'
STAGE3_BASIS_SHA = '79c0f3e4f3cd04138b0b86126609fd56578108c9fb0ae20a209e05d237891e9f'
STAGE3_MU = 'reviews/2026-09-23/optimization_v1/stage3_run_v1/mu.pt'
STAGE3_MU_SHA = 'ec4e8fa8ee2a51c37ed4a6e39f59af88d6ac57adae54558148ca5d15020fe7fe'
STAGE3_RESULT = 'reviews/2026-09-23/optimization_v1/stage3_run_v1/result.json'
STAGE3_RESULT_SHA = '21864f1313d6ade9a8483375a049e7a94a54bd7e66c1165560d5fddfc4941eda'
STAGE4_CORE = 'reviews/2026-09-23/optimization_v1/stage4_run_v1/core/student.pt'
STAGE4_CORE_SHA = '261ef98f7371f1a5c8d4f8883f94d8fb8a39d3acd121f1f80d99cdaae09c7ce5'
RELEASE_REL = 'reviews/2026-09-12/data_release_final/QO_FORMAL_RELEASE.json'
RELEASE_SHA = '5c6dd879beb0f70bcd2388601736cbf0bf341f111a267f79801c478e780b9287'
ROSTER_REL = 'reviews/2026-09-12/formal_mask_bank/EPISODE_ROSTER_NON_TEST.json'
ROSTER_SHA = 'cb38dca7027a8bc087290b01cf61e6f6f2550ddbc2f2a4f4a92b807e8813bbee'
GRAPH_REL = 'data/pems/graphs/qo_pilot_v1/release_v1/graph.json'
GRAPH_SHA = '6373d9a58173b4e828fced08b57f944a1ba78ac8733125e1ee100c2f5a034eb4'
SPLIT_REL = 'reviews/2026-09-23/optimization_v1/split.json'
REG2_REL = 'runs/current_student_train_preparation_v2/registration.json'
V7_PLAN = 'reviews/2026-09-21/rttc_repair/rttc_real_v7/plan.json'
V7_PLAN_SHA = 'a1d067c3525ea4b3beeb2de7c2f06564fb2cc384b80bac5a8e8f9f7dd0a57117'
V7_CHOSEN = 'reviews/2026-09-21/rttc_repair/rttc_real_v7/chosen.json'
V7_CHOSEN_SHA = '55ab4e76bbe5fdc784ed524712f6554b13a49833dd6b5d7980ca05c47755742f'
V7_SELECTION_NPZ = 'reviews/2026-09-21/rttc_repair/rttc_real_v7/selection/000.npz'
CONFIRM_PLAN = 'reviews/2026-09-25/g80_independent_confirmation_v1/plan.json'
CONFIRM_PLAN_SHA = 'db61bcf3d035cf15ff24c29234893a4c3dbc17bdd5dad74b2f418920e3c9bf1c'
S3_SCALE_Q, S3_SCALE_O = 6240.0, 0.21276629999999888
V7_SCALE_Q, V7_SCALE_O = 6264.0, 0.2128549963235855

CHANNELS = ('q', 'o')
BUFFER_DAYS = 3
SLOTS = 288
LIMITS = {'threads': 4, 'process_rss_gib': 12, 'launch_available_gib': 10,
          'runtime_available_floor_gib': 4, 'rttc_case_wall_s': 3600,
          'neural_case_wall_s': 600}
BOOTSTRAP_SEED = 20260925
BOOTSTRAP_RESAMPLES = 10000

MAIN_ARMS = ('g80', 'rttc_v7')
ABLATION_ARMS = ('core_only',)
APPENDIX_ARMS: tuple = ()

CODE_FILES = (
    'scripts/run_g80_test.py',
    'scripts/verify_g80_test.py',
    'scripts/fit_sparse_neighbor_context.py',
    'scripts/fit_sparse_global_tokens.py',
    'scripts/fit_profile_residual_candidate.py',
    'models/sparse_global_tokens.py',
    'models/sparse_neighbor_context.py',
    'models/sparse_observation_profile.py',
    'training/current_release_student_data.py',
    'training/pilot_data.py',
    'training/pilot_engine.py',
    'training/real_student_adapter.py',
    'training/qo_attributes.py',
    'training/real_qo_loader.py',
    'representations/basis.py',
    'models/film_rgnn.py',
    'reviews/2026-09-21/rttc_repair/rttc_real.py',
    'reviews/2026-09-19/ai_handoff_diagnosis_20260919T051738Z/tools/rttc_stable.py',
    'reviews/2026-09-19/ai_handoff_diagnosis_20260919T051738Z/tools/rttc_pelt.py',
    'reviews/2026-09-19/ai_handoff_diagnosis_20260919T051738Z/tools/rttc_glue.py',
)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def append_jsonl(path: Path, row: dict) -> None:
    with Path(path).open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def load_json(path: Path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json_atomic(path: Path, data: dict) -> None:
    tmp = Path(str(path) + '.tmp')
    with tmp.open('w', encoding='utf-8') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def available_gib() -> float:
    return psutil.virtual_memory().available / 2 ** 30


def check_memory_floor(floor_gib: float) -> float:
    value = available_gib()
    if value < floor_gib:
        raise RuntimeError(f'available memory {value:.2f} GiB below {floor_gib} GiB')
    return value


class TreePeakRSS:
    def __init__(self, interval: float = 1.0) -> None:
        self._interval = interval
        self._peak = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample(self) -> int:
        proc = psutil.Process()
        total = proc.memory_info().rss
        try:
            children = proc.children(recursive=True)
        except psutil.Error:
            children = []
        for child in children:
            try:
                total += child.memory_info().rss
            except psutil.Error:
                pass
        return total

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._peak = max(self._peak, self._sample())
            self._stop.wait(self._interval)

    def __enter__(self) -> 'TreePeakRSS':
        self._peak = self._sample()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self._peak = max(self._peak, self._sample())

    @property
    def peak_bytes(self) -> int:
        return int(self._peak)


def setup_torch() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError('single GPU required to replicate the frozen pipeline')
    torch.set_num_threads(LIMITS['threads'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def git_status_snapshot() -> dict:
    head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, check=True,
                          capture_output=True, text=True).stdout.strip()
    status = subprocess.run(['git', 'status', '--short'], cwd=ROOT, check=True,
                            capture_output=True, text=True).stdout
    return {'head': head,
            'modified_or_untracked_lines': status.splitlines(),
            'note': 'recorded, not rolled back, not staged'}


# ---------------------------------------------------------------------------
# restricted Test reader and assembler (mirrors the shared implementations)
# ---------------------------------------------------------------------------

class TestRoleReader:
    """Whitelisted Test-day reader; mirrors VerifiedRoleReader for role=test."""

    def __init__(self, root: Path, release_path: Path, stations: tuple[str, ...],
                 allowed_days: set[str]):
        raw = release_path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != RELEASE_SHA:
            raise ValueError('release SHA mismatch')
        release = json.loads(raw)
        if release['status'] != 'frozen':
            raise ValueError('release must be frozen')
        self.bounds = tuple(release['roles']['test'])
        self.allowed_days = frozenset(allowed_days)

        def pinned(item):
            blob = (root / item['path']).read_bytes()
            if hashlib.sha256(blob).hexdigest() != item['sha256']:
                raise ValueError(f"pin mismatch {item['path']}")
            return json.loads(blob)

        entries = pinned({'path': release['u_manifest'],
                          'sha256': release['permanent_u_sha256']})['stations']
        self.universe = {str(x['station_id']): x['permanent_U'] for x in entries}
        if len(set(stations)) != len(stations) or \
                any(self.universe.get(s, True) for s in stations):
            raise ValueError('graph includes unknown, duplicate or U station')
        days = pinned(release['array_manifest'])['days']
        self.day_index = {x['day']: x for x in days}
        self.stations = set(stations)
        self.output_root = Path(release['output_root']).resolve()
        self.access_log: list[dict[str, str]] = []

    def __call__(self, day: str, purpose: str = 'episode_assembly'):
        date.fromisoformat(day)
        if day not in self.allowed_days:
            raise ValueError(f'day outside whitelisted Test input dates: {day}')
        if not self.bounds[0] <= day <= self.bounds[1]:
            raise ValueError('day outside Test role bounds')
        entry = self.day_index[day]
        path = Path(entry['path']).resolve() / 'arrays.npz'
        if path.parent != self.output_root / day:
            raise ValueError('array path escapes registered day')
        blob = path.read_bytes()
        if hashlib.sha256(blob).hexdigest() != entry['sha256']:
            raise ValueError(f'SHA-256 mismatch: {path}')
        import io
        with np.load(io.BytesIO(blob), allow_pickle=False) as archive:
            from training.real_qo_loader import select_day
            ids, values, masks = select_day(archive, self.universe)
        self.access_log.append({'day': day, 'sha256': entry['sha256'],
                                'purpose': purpose})
        return [(day, str(s), values[i], masks[i]) for i, s in enumerate(ids)
                if str(s) in self.stations]


def assemble_test_episode(row, station_order, role_bounds, read_day):
    """Mirror of assemble_episode for Test rows (identical geometry)."""
    if row['split'] != 'test' or row['protocol'] != 'K':
        raise ValueError('only Test Protocol-K episodes supported')
    target = str(row['station_id'])
    if target not in station_order:
        raise ValueError('target outside graph station order')
    duration = row['duration_days']
    if type(duration) is not int or duration not in {7, 14}:
        raise ValueError('registered 7/14-day duration required')
    start = date.fromisoformat(row['start_day']) - timedelta(days=BUFFER_DAYS)
    dates = tuple((start + timedelta(days=i)).isoformat()
                  for i in range(duration + 2 * BUFFER_DAYS))
    if dates[0] < role_bounds[0] or dates[-1] > role_bounds[1]:
        raise ValueError('episode or buffer crosses role boundary')
    shape = (len(dates), len(station_order), SLOTS, 2)
    reference = np.zeros(shape, dtype=np.float64)
    visible = np.zeros(shape, dtype=bool)
    lookup = {station: i for i, station in enumerate(station_order)}
    for d, day in enumerate(dates):
        seen = set()
        for label, station, values, mask in read_day(day):
            if label != day or station in seen:
                raise ValueError('duplicate station or day label mismatch')
            seen.add(station)
            if station not in lookup:
                continue
            if values.shape != (SLOTS, 2) or mask.shape != values.shape or \
                    mask.dtype != bool:
                raise ValueError('expected q/o profile and boolean legal mask')
            if not np.isfinite(values[mask]).all():
                raise ValueError('nonfinite legal values')
            n = lookup[station]
            reference[d, n] = np.where(mask, values, 0.)
            visible[d, n] = mask
        if not visible[d, lookup[target]].all(axis=-1).sum() >= 274:
            raise ValueError('target or buffer fails registered complete-day '
                             'eligibility')
    scoring = np.zeros(shape, dtype=bool)
    scoring[BUFFER_DAYS:BUFFER_DAYS + duration, lookup[target]] = \
        visible[BUFFER_DAYS:BUFFER_DAYS + duration, lookup[target]]
    visible[BUFFER_DAYS:BUFFER_DAYS + duration, lookup[target]] = False
    values = np.where(visible, reference, 0.)
    from training.current_release_student_data import CurrentEpisode
    return CurrentEpisode(row['episode_id'], row['mask_id'], row['split'],
                          dates, tuple(station_order), lookup[target], values,
                          visible, reference, scoring)


ATTRIBUTES = None


def load_models():
    global ATTRIBUTES
    from models.sparse_global_tokens import GlobalTokenProfile
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
    mu_ck = torch.load(ROOT / STAGE3_MU, map_location='cpu', weights_only=False)
    if not np.array_equal(np.asarray([S3_SCALE_Q, S3_SCALE_O]),
                          np.asarray(mu_ck['scales'])):
        raise ValueError('stage-3 scales differ from mu checkpoint')
    mu = MeanFunction(5, 4, 6, channels=2).requires_grad_(False).eval()
    mu.load_state_dict(mu_ck['mean_state'])
    core_state = torch.load(ROOT / STAGE4_CORE, map_location='cpu',
                            weights_only=False)
    core = make_student(rank=16, hidden=32).requires_grad_(False).eval()
    core.load_state_dict(core_state['student'])
    g80_state = torch.load(ROOT / G80_CK, map_location='cpu', weights_only=False)
    g80 = GlobalTokenProfile(dropout=.1).requires_grad_(False).eval()
    g80.load_state_dict(g80_state['model'], strict=True)
    g80.to('cuda:0')
    return basis, mu, core, g80


def prepare_case_cpu(episode, basis, mu, scales, static, edges):
    """Mirror of prepare_case for Test (CPU basis/mu/core, no teacher)."""
    from training.pilot_data import calendar_features, rolling_history
    from training.real_student_adapter import build_student_inputs
    device = next(basis.parameters()).device
    scales_t = torch.tensor(scales, dtype=torch.float32)
    if any(p.requires_grad or p.grad is not None for m in (basis, mu)
           for p in m.parameters()):
        raise ValueError('representation must be frozen')
    with torch.no_grad():
        values = torch.as_tensor(episode.values, dtype=torch.float32,
                                 device=device) / scales_t
        visible = torch.as_tensor(episode.visible, device=device)
        values = torch.where(visible, values, torch.zeros_like(values))
        history = rolling_history(values, visible)
        calendar = calendar_features(episode.dates).to(device)
        per_day_static = static[None].expand(len(episode.dates), -1, -1)
        mean = basis.global_mean + mu(per_day_static, calendar, history,
                                      protocol='K')
        inputs = build_student_inputs(basis, values, visible, mean, static,
                                      edges, episode.dates)
        scoring = torch.as_tensor(episode.scoring, device=device)
        reference = torch.as_tensor(episode.reference, dtype=torch.float32,
                                    device=device) / scales_t
        ordinals = torch.tensor([date.fromisoformat(d).toordinal()
                                 for d in episode.dates], device=device)
        return {'inputs': inputs, 'reference': reference, 'scoring': scoring,
                'static': static, 'ordinals': ordinals}


def build_item(episode, row, basis, mu, core, scales, edges, device):
    from scripts.fit_sparse_neighbor_context import PreparedEpisode, signed_chain_hops
    from training.current_release_student_data import predict_case
    features = np.asarray(
        [[ATTRIBUTES.lookup(day, station)['features']
          for station in episode.station_order] for day in episode.dates],
        dtype=np.float32)
    if not np.array_equal(features, np.broadcast_to(features[0], features.shape)):
        raise ValueError('date-varying static features unsupported')
    case = prepare_case_cpu(episode, basis, mu, scales,
                            torch.from_numpy(features[0]), edges)
    selected = case['scoring'].any(dim=(-1, -2))
    geometry = torch.zeros_like(selected)
    duration = row['duration_days']
    geometry[BUFFER_DAYS:BUFFER_DAYS + duration, episode.target_index] = True
    if not torch.equal(selected, geometry):
        raise ValueError('target support differs from registered outage geometry')
    with torch.no_grad():
        coefficients = predict_case(core, case)
        base = basis.decode(coefficients[selected], case['inputs'].mean[selected])
    return PreparedEpisode(
        episode.episode_id, episode.mask_id, str(row['station_id']), duration,
        episode.target_index, None,
        torch.as_tensor(episode.values, dtype=torch.float32, device=device),
        torch.as_tensor(episode.visible, dtype=torch.bool, device=device),
        case['static'].to(device), case['inputs'].calendar.to(device),
        base.to(device), case['reference'][selected].to(device),
        case['scoring'][selected].to(device),
        signed_chain_hops(edges, target=episode.target_index,
                          nodes=len(episode.station_order)).to(device))


def infer_g80(model, item, scales, device):
    from models.sparse_global_tokens import global_token_context
    profiles = []
    with torch.no_grad():
        for day in range(BUFFER_DAYS, BUFFER_DAYS + item.duration):
            q, k, valid = global_token_context(
                values=item.values, visible=item.visible, base=item.base[day - 3],
                static=item.static, calendar=item.calendar, target=item.target,
                day=day, signed_hops=item.hops, scales=scales, raw_values=True)
            predicted, _ = model(q, k, valid, item.base[day - 3])
            profiles.append(predicted.cpu())
    return torch.stack(profiles)


# ---------------------------------------------------------------------------
# scoring (physical units; stage-3 denominators for NRMSE)
# ---------------------------------------------------------------------------

def score_physical(pred_phys64, truth_phys64, scoring, s3_scales):
    if pred_phys64.shape != truth_phys64.shape or \
            pred_phys64.shape != scoring.shape:
        raise ValueError('score shape mismatch')
    scores = {}
    for c, channel in enumerate(CHANNELS):
        active = scoring[..., c]
        count = active.sum(axis=-1)
        if not np.all(count > 0):
            raise ValueError('empty scoring profile')
        error = np.where(active, pred_phys64[..., c] - truth_phys64[..., c], 0.0)
        error_norm = error / s3_scales[c]
        profile_rmse = np.sqrt(np.square(error).sum(axis=-1) / count)
        profile_nrmse = np.sqrt(np.square(error_norm).sum(axis=-1) / count)
        scores[channel] = {
            'profile_macro_nrmse': float(profile_nrmse.mean()),
            'profile_rmse': float(profile_rmse.mean()),
            'raw_mae': float(np.abs(error[active]).mean()),
            'legal_elements': int(active.sum()),
        }
    return scores


def sensor_macro(rows, arm, channel, metric):
    grouped = {}
    for row in rows:
        grouped.setdefault(row['station_id'], []).append(
            row['scores'][arm][channel][metric])
    return float(sum(sum(v) / len(v) for v in grouped.values()) / len(grouped))


def episode_macro(rows, arm, channel, metric):
    return float(sum(r['scores'][arm][channel][metric] for r in rows) / len(rows))


def macro_table(rows, arms):
    out = {'n_cases': len(rows), 'arms': list(arms)}
    for arm in arms:
        out[arm] = {}
        for channel in CHANNELS:
            out[arm][channel] = {}
            for metric in ('profile_macro_nrmse', 'profile_rmse', 'raw_mae'):
                out[arm][channel][metric] = {
                    'episode_macro': episode_macro(rows, arm, channel, metric),
                    'sensor_macro': sensor_macro(rows, arm, channel, metric),
                }
    return out


def paired_rows(rows, arm_a, arm_b):
    common = [r for r in rows if arm_a in r['scores'] and arm_b in r['scores']]
    out = {'n_cases': len(common),
           'excluded_ids': [r['episode_id'] for r in rows
                            if not (arm_a in r['scores'] and
                                    arm_b in r['scores'])],
           'channels': {}}
    for channel in CHANNELS:
        diffs = np.array([r['scores'][arm_a][channel]['profile_macro_nrmse'] -
                          r['scores'][arm_b][channel]['profile_macro_nrmse']
                          for r in common], dtype=np.float64)
        base_err = np.array([r['scores'][arm_b][channel]['profile_macro_nrmse']
                             for r in common], dtype=np.float64)
        with np.errstate(divide='ignore', invalid='ignore'):
            rel = np.where(base_err > 0, 1 - (base_err + diffs) / base_err, np.nan)
        mae_diff = np.array([r['scores'][arm_a][channel]['raw_mae'] -
                             r['scores'][arm_b][channel]['raw_mae']
                             for r in common], dtype=np.float64)
        out['channels'][channel] = {
            'mean_diff': float(diffs.mean()),
            'median_diff': float(np.median(diffs)),
            'improved_cases': int((diffs < 0).sum()),
            'worsened_cases': int((diffs > 0).sum()),
            'equal_cases': int((diffs == 0).sum()),
            'mean_relative_improvement_nrmse': float(np.nanmean(rel))
            if np.isfinite(rel).any() else None,
            'mean_raw_mae_diff': float(mae_diff.mean()),
        }
    return out


def station_bootstrap(rows, arm_a, arm_b, channel, metric,
                      resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED):
    grouped_a, grouped_b = {}, {}
    for row in rows:
        grouped_a.setdefault(row['station_id'], []).append(
            row['scores'][arm_a][channel][metric])
        grouped_b.setdefault(row['station_id'], []).append(
            row['scores'][arm_b][channel][metric])
    stations = sorted(grouped_a)
    mean_a = np.array([float(np.mean(grouped_a[s])) for s in stations])
    mean_b = np.array([float(np.mean(grouped_b[s])) for s in stations])
    rng = np.random.default_rng(seed)
    index = rng.integers(0, len(stations), size=(resamples, len(stations)))
    diffs = mean_a[index].mean(axis=1) - mean_b[index].mean(axis=1)
    return {'point_diff': float(mean_a.mean() - mean_b.mean()),
            'ci95_low': float(np.percentile(diffs, 2.5)),
            'ci95_high': float(np.percentile(diffs, 97.5)),
            'resamples': resamples, 'seed': seed, 'stations': len(stations),
            'units': 'station re-weighting sensitivity only; cross-station '
                     'correlation within the corridor remains'}


def slope_pairs(rows, arm):
    """(MAE14-MAE7)/(log14-log7) on same-station same-start-day pairs."""
    grouped = {}
    for row in rows:
        if arm not in row['scores']:
            continue
        key = (row['station_id'], row['start_day'])
        grouped.setdefault(key, {})[row['duration_days']] = \
            (row['scores'][arm]['q']['raw_mae'],
             row['scores'][arm]['o']['raw_mae'])
    matched, slopes_q, slopes_o = [], [], []
    for (station, start), per in sorted(grouped.items()):
        if 7 in per and 14 in per:
            mae7, o7 = per[7]
            mae14, o14 = per[14]
            denom = float(np.log(14) - np.log(7))
            slopes_q.append((mae14 - mae7) / denom)
            slopes_o.append((o14 - o7) / denom)
            matched.append({'station_id': station, 'start_day': start})
    unmatched = [f'{k[0]}/{k[1]}' for k, v in grouped.items()
                 if not (7 in v and 14 in v)]
    return {'matched_events': matched, 'n_matched': len(matched),
            'unmatched_keys': unmatched,
            'slope_q_median': float(np.median(slopes_q)) if slopes_q else None,
            'slope_o_median': float(np.median(slopes_o)) if slopes_o else None,
            'definition': '(MAE14-MAE7)/(log14-log7) on same-station '
                          'same-start-day pairs; changes outage length and '
                          'both-side context simultaneously'}


# ---------------------------------------------------------------------------
# stage A: draft
# ---------------------------------------------------------------------------

def verify_static_pins() -> dict:
    pins = {
        'work_order': (WORK_ORDER_REL, None),
        'candidates': (CANDIDATES_REL, None),
        'g80_checkpoint': (G80_CK, G80_CK_SHA),
        'stage3_basis': (STAGE3_BASIS, STAGE3_BASIS_SHA),
        'stage3_mu': (STAGE3_MU, STAGE3_MU_SHA),
        'stage3_result': (STAGE3_RESULT, STAGE3_RESULT_SHA),
        'stage4_core': (STAGE4_CORE, STAGE4_CORE_SHA),
        'release': (RELEASE_REL, RELEASE_SHA),
        'roster': (ROSTER_REL, ROSTER_SHA),
        'graph': (GRAPH_REL, GRAPH_SHA),
        'rttc_v7_plan': (V7_PLAN, V7_PLAN_SHA),
        'rttc_v7_chosen': (V7_CHOSEN, V7_CHOSEN_SHA),
        'confirm_plan': (CONFIRM_PLAN, CONFIRM_PLAN_SHA),
    }
    resolved = {}
    for name, (rel, expected) in pins.items():
        digest = sha(ROOT / rel)
        if expected is not None and digest != expected:
            raise ValueError(f'pin drift {name}: {rel}')
        resolved[name] = {'path': rel, 'sha256': digest}
    return resolved


def derive_candidates(order):
    roster = load_json(ROOT / ROSTER_REL)['episodes']
    graph = set(order)
    rows = [row for row in roster
            if row['split'] == 'test' and row['protocol'] == 'K' and
            str(row['station_id']) in graph and row['duration_days'] in (7, 14)]
    rows.sort(key=lambda r: r['episode_id'])
    return rows


def episode_dates(row):
    start = date.fromisoformat(row['start_day']) - timedelta(days=BUFFER_DAYS)
    return [(start + timedelta(days=i)).isoformat()
            for i in range(row['duration_days'] + 2 * BUFFER_DAYS)]


def stage_draft(out: Path) -> None:
    if out.exists():
        existing = out / 'plan.json'
        if existing.exists():
            raise FileExistsError('plan.json exists; use the same plan to '
                                  'resume or create a v2 directory')
    pins = verify_static_pins()
    graph = load_json(ROOT / GRAPH_REL)
    order = tuple(str(x) for x in graph['station_order'])
    rows = derive_candidates(order)
    attachment = load_json(ROOT / CANDIDATES_REL)
    attach_ids = sorted(e['episode_id'] for e in attachment['episodes'])
    if sorted(r['episode_id'] for r in rows) != attach_ids:
        raise ValueError('re-derived candidates differ from the attachment')
    if (len(rows) != attachment['n_candidates'] or
            attachment['by_duration'] != {'7': 13, '14': 8} or
            attachment['n_stations'] != 10):
        raise ValueError('attachment counts differ from roster derivation')
    allowed_days = list(attachment['allowed_model_input_dates'])
    for row in rows:
        window = episode_dates(row)
        if not set(window) <= set(allowed_days):
            raise ValueError(f"candidate outside whitelist: {row['episode_id']}")
    exposed = {
        'train_prepared_268': set(load_json(ROOT / REG2_REL)['episode_ids']),
        'train_validation_58': set(load_json(ROOT / SPLIT_REL)['validation_ids']),
        'rttc_tuning_30': {r['episode_id'] for r in load_json(
            ROOT / V7_PLAN)['episodes']['train']},
        'selection_prepared_42': set(load_json(
            ROOT / 'runs/current_student_selection_preparation_v1/'
            'registration.json')['episode_ids']),
        'prev_comparison_58': set(load_json(ROOT / 'reviews/2026-09-25/'
            'global80_rttc_init_v1/plan.json')['validation_ids']),
        'prev_confirmation_31': set(load_json(ROOT / CONFIRM_PLAN)
                                    ['sealed_set']['episode_ids']),
    }
    test_ids = {r['episode_id'] for r in rows}
    intersections = {name: sorted(used & test_ids)
                     for name, used in exposed.items()}
    exposed_support_days = set()
    for name, used_ids in exposed.items():
        roster_map = {r['episode_id']: r for r in
                      load_json(ROOT / ROSTER_REL)['episodes']}
        for episode_id in used_ids:
            row = roster_map.get(episode_id)
            if row is not None:
                exposed_support_days.update(episode_dates(row))
    test_support_days = {d for row in rows for d in episode_dates(row)}
    date_overlap = sorted(exposed_support_days & test_support_days)
    access_evidence = []
    for log_rel in ('reviews/2026-09-25/global80_rttc_init_v1/access.jsonl',
                    'reviews/2026-09-25/g80_independent_confirmation_v1/'
                    'access.jsonl',
                    'reviews/2026-09-21/rttc_repair/rttc_real_v7/access.jsonl'):
        path = ROOT / log_rel
        if not path.exists():
            continue
        entries = [json.loads(line) for line in
                   path.read_text(encoding='utf-8').splitlines() if line]
        roles = {e.get('role') for e in entries}
        days = {e.get('day') for e in entries}
        access_evidence.append({
            'log': log_rel, 'roles': sorted(roles), 'n_days': len(days),
            'test_role_present': 'test' in roles,
            'test_days_present': sorted(d for d in days
                                        if d >= '2019-11-07')[:5],
        })
    audit = {
        'id_intersections': {name: {'count': len(v), 'ids': v}
                             for name, v in intersections.items()},
        'date_support_overlap_with_exposed_episodes': {
            'count': len(date_overlap), 'days': date_overlap[:10],
            'note': 'Test support days (2019-11-27..2019-12-28) lie outside '
                    'every exposed episode role window (train <= 2019-08-07, '
                    'selection <= 2019-09-30, calibration <= 2019-11-06)'},
        'station_support_overlap': {
            'same_20_station_graph_shared': True,
            'note': 'donor stations and graph are shared infrastructure across '
                    'roles; ID- and date-level supports are disjoint, but '
                    'station-level sharing is declared, not denied'},
        'historical_access_evidence': access_evidence,
        'evidence_boundary': 'absence of historical Test reads is supported by '
                             'role-bounded readers and the audited access logs '
                             'listed above; logs cover the documented runs, not '
                             'every historical shell command',
        'git_status': git_status_snapshot(),
    }
    protocol_draft = {
        'status': 'draft_before_exposed_case_preflight',
        'work_order': dict(rel=WORK_ORDER_REL, **pins['work_order']),
        'main_arms': list(MAIN_ARMS),
        'ablation_arms': list(ABLATION_ARMS),
        'appendix_arms': list(APPENDIX_ARMS),
        'g40_run': False,
        'arm_identities': {
            'g80': 'frozen stage-4 core + stage-3 basis/mu base profile + '
                   'GlobalTokenProfile residual head (global_control.pt)',
            'rttc_v7': 'modified RTTC stable adapter, v7 candidate 0 '
                       '(mu=.001, lam=.01, rho=.01) with v7 fixed config; '
                       'internal scales 6264/0.2128549963235855',
            'core_only': 'mu+Bhat base profile saved inside the same G80 '
                         'forward pass (frozen module removal, no retraining)',
        },
        'scoring': {
            'truth_units': 'physical (q veh/h, o fraction), float64',
            'nrmse_denominator': 'stage-3 scales [6240.0, '
                                 '0.21276629999999888] for every arm',
            'metrics': ['profile_macro_nrmse', 'profile_rmse', 'raw_mae'],
            'aggregation': 'per-day profile metric -> episode mean -> '
                           'station mean (sensor macro); episode macro also '
                           'reported',
        },
        'eligibility': 'roster Test rows, protocol K, 20-station graph, 7/14 '
                       'days, >=274 fully legal target slots per day '
                       '(assemble-time, pre-declared)',
        'failure_policy': {
            'implementation_errors': 'mask/reference/identity misalignment or '
                                     'input violations stop the run immediately',
            'method_numeric_failure': 'recorded; other cases continue; full '
                                      '21-case status always reported; paired '
                                      'comparisons restricted to common '
                                      'successful support',
            'rttc_max_iter': 'reaching 250 iterations with finite output is '
                             'kept and flagged not_converged',
        },
        'resources': LIMITS,
        'statistics': {
            'main': 'descriptive sensor-macro errors and paired differences; '
                    'no independent-episode significance claim',
            'bootstrap': {'unit': 'station', 'resamples': BOOTSTRAP_RESAMPLES,
                          'seed': BOOTSTRAP_SEED, 'level': 0.95,
                          'scope': 'appendix table; station re-weighting '
                                   'sensitivity only'},
            'slope': 'same-station same-start-day 7/14 pairs; '
                     '(MAE14-MAE7)/(log14-log7)',
            'w5_status': 'not declared passed: same-week-median and '
                         'historical-anchor low-rank null comparators are '
                         'absent this round',
        },
        'tolerances': None,
        'rttc_timeout_s': None,
        'pending_preflight': ['tolerances', 'rttc_timeout_s'],
    }
    out.mkdir(parents=True, exist_ok=True)
    write_json_atomic(out / 'overlap_audit.json', audit)
    write_json_atomic(out / 'protocol_draft.json', protocol_draft)
    print(json.dumps({'candidates': len(rows), 'id_intersections':
                      {k: v['count'] for k, v in audit['id_intersections'].items()},
                      'date_overlap': len(date_overlap)}), flush=True)


# ---------------------------------------------------------------------------
# stage B: preflight on exposed cases
# ---------------------------------------------------------------------------

def stage_preflight(out: Path) -> None:
    draft = load_json(out / 'protocol_draft.json')
    if draft['status'] != 'draft_before_exposed_case_preflight':
        raise ValueError('protocol draft missing or already finalized')
    setup_torch()
    check_memory_floor(LIMITS['launch_available_gib'])
    device = torch.device('cuda:0')
    scales = (S3_SCALE_Q, S3_SCALE_O)
    graph = load_json(ROOT / GRAPH_REL)
    order = tuple(str(x) for x in graph['station_order'])
    edges = {name: torch.tensor(value, dtype=torch.long)
             for name, value in graph['edge_index'].items()}
    basis, mu, core, g80 = load_models()

    # (1) G80 + core replication on exposed validation episode 000
    roster = {row['episode_id']: row for row in
              load_json(ROOT / ROSTER_REL)['episodes']}
    from training.current_release_student_data import VerifiedRoleReader, assemble_episode
    train_reader = VerifiedRoleReader(ROOT, ROOT / RELEASE_REL, RELEASE_SHA,
                                      'train', order)

    @lru_cache(maxsize=256)
    def read_train(day):
        return train_reader(day)

    with np.load(ROOT / G80_DIR / 'validation' / '000.npz',
                 allow_pickle=False) as archive:
        g80_identity = str(archive['episode_id'])
        archived_control = archive['control']
        archived_core = archive['core']
        archived_scoring = archive['scoring']
    episode = assemble_episode(roster[g80_identity], order,
                               train_reader.bounds, read_train)
    if np.any(episode.visible & episode.scoring):
        raise ValueError('score visible leak')
    item = build_item(episode, roster[g80_identity], basis, mu, core, scales,
                      edges, device)
    if not np.array_equal(item.scoring.cpu().numpy(), archived_scoring):
        raise ValueError('replication scoring mask differs from archive')
    pred_g80, _, _ = timed('g80_replication',
                           lambda: infer_g80(g80, item, scales, device),
                           LIMITS['neural_case_wall_s'])
    core_only = item.base.cpu()
    bitwise_g80 = bool(np.array_equal(pred_g80.numpy(), archived_control))
    bitwise_core = bool(np.array_equal(core_only.numpy(), archived_core))
    if not bitwise_g80 or not bitwise_core:
        raise RuntimeError('G80/core replication is not bitwise equal to the '
                           'archived predictions')

    # (2) RTTC v7 replication on exposed Selection case rttc_real_v7/selection/000
    from rttc_real import solve_episode
    sel_reader = VerifiedRoleReader(ROOT, ROOT / RELEASE_REL, RELEASE_SHA,
                                    'selection', order)

    @lru_cache(maxsize=256)
    def read_sel(day):
        return sel_reader(day)

    with np.load(ROOT / V7_SELECTION_NPZ, allow_pickle=False) as archive:
        rttc_identity = str(archive['episode_id'])
        archived_rttc = archive['prediction_raw']
        archived_scales = archive['scales']
    if not np.array_equal(archived_scales, np.asarray([V7_SCALE_Q, V7_SCALE_O])):
        raise ValueError('v7 archive scales differ from plan')
    v7_plan = load_json(ROOT / V7_PLAN)
    candidate0 = v7_plan['candidates'][0]
    fixed = v7_plan['fixed']
    rttc_episode = assemble_episode(roster[rttc_identity], order,
                                    sel_reader.bounds, read_sel)
    if np.any(rttc_episode.visible & rttc_episode.scoring):
        raise ValueError('score visible leak')
    rttc_payload, _, wall_rttc, rss_rttc = timed_rttc(
        lambda: solve_episode(rttc_episode, candidate0, fixed))
    pred_rttc, telemetry = rttc_payload
    bitwise_rttc = bool(np.array_equal(pred_rttc, archived_rttc))
    if not bitwise_rttc:
        raise RuntimeError('RTTC v7 replication is not bitwise equal to the '
                           'archived prediction')

    # (3) hidden-reference perturbation on the same two exposed cases
    perturbed_reference = episode.reference.copy()
    perturbed_reference[~episode.visible] += 123.456
    import dataclasses
    perturbed = dataclasses.replace(episode, reference=perturbed_reference)
    item_p = build_item(perturbed, roster[g80_identity], basis, mu, core,
                        scales, edges, device)
    pred_g80_p, _, _ = timed('g80_perturbation',
                             lambda: infer_g80(g80, item_p, scales, device),
                             LIMITS['neural_case_wall_s'])
    perturb_rttc_reference = rttc_episode.reference.copy()
    perturb_rttc_reference[~rttc_episode.visible] += 123.456
    rttc_perturbed = dataclasses.replace(rttc_episode,
                                         reference=perturb_rttc_reference)
    perturb_payload, _, _, _ = timed_rttc(
        lambda: solve_episode(rttc_perturbed, candidate0, fixed))
    pred_rttc_p = perturb_payload[0]
    checks = {
        'g80_bitwise': bitwise_g80,
        'core_only_bitwise': bitwise_core,
        'rttc_v7_bitwise': bitwise_rttc,
        'g80_perturbation_bitwise': bool(np.array_equal(pred_g80_p.numpy(),
                                                        archived_control)),
        'rttc_perturbation_bitwise': bool(np.array_equal(pred_rttc_p,
                                                         archived_rttc)),
    }
    if not all(checks.values()):
        raise RuntimeError(f'preflight checks failed: {checks}')

    # tolerance and timeout freeze (from observed replication, before Test)
    tolerances = {
        'replication': 'bitwise (np.array_equal) on the same GPU/environment; '
                       'no cross-device tolerance needed',
        'truth_scored_dtype': 'float64 physical units, aligned to source '
                              'arrays at bind time',
        'score_recompute': 'float64 arithmetic from stored float64 artifacts; '
                           'verifier agreement required within 1e-12 relative',
        'neural_prediction_storage': 'physical units, float64 = float32 '
                                     'normalized output x float64 stage-3 scale',
    }
    rttc_timeout = LIMITS['rttc_case_wall_s']
    protocol = dict(draft)
    protocol['status'] = 'frozen_after_exposed_case_preflight'
    protocol['tolerances'] = tolerances
    protocol['rttc_timeout_s'] = rttc_timeout
    protocol['rttc_timeout_basis'] = (
        f"v7 plan contract wall_s={v7_plan['limits']['wall_s']}; observed "
        f"replication {wall_rttc:.1f}s for a 7-day case; limit is far above "
        'any observed solve and is not extended during Test')
    protocol['preflight'] = {
        'g80_episode': g80_identity, 'rttc_episode': rttc_identity,
        'checks': checks, 'rttc_replication_wall_s': wall_rttc,
        'rttc_replication_peak_rss_bytes': rss_rttc,
        'rttc_replication_iterations': [t['iterations'] for t in telemetry],
    }
    write_json_atomic(out / 'preflight.json', {
        'status': 'pass', 'checks': checks,
        'rttc_replication_wall_s': wall_rttc,
        'rttc_replication_iterations': [t['iterations'] for t in telemetry],
        'tolerances': tolerances, 'rttc_timeout_s': rttc_timeout,
        'note': 'no Test array was read in this stage'})
    write_json_atomic(out / 'protocol.json', protocol)
    print(json.dumps({'status': 'pass', 'checks': checks,
                      'rttc_wall_s': wall_rttc}), flush=True)


def timed(label, fn, wall_limit):
    check_memory_floor(LIMITS['runtime_available_floor_gib'])
    started = time.perf_counter()
    with TreePeakRSS() as rss:
        result = fn()
    wall = time.perf_counter() - started
    if rss.peak_bytes > LIMITS['process_rss_gib'] * 2 ** 30:
        raise RuntimeError(f'{label} peak RSS exceeds cap')
    if wall > wall_limit:
        raise RuntimeError(f'{label} wall {wall:.1f}s exceeds limit')
    return result, wall, rss.peak_bytes


def timed_rttc(fn):
    check_memory_floor(LIMITS['runtime_available_floor_gib'])
    started = time.perf_counter()
    with TreePeakRSS() as rss:
        result = fn()
    wall = time.perf_counter() - started
    if rss.peak_bytes > LIMITS['process_rss_gib'] * 2 ** 30:
        raise RuntimeError('RTTC peak RSS exceeds cap')
    if wall > LIMITS['rttc_case_wall_s']:
        raise RuntimeError(f'RTTC wall {wall:.1f}s exceeds frozen timeout')
    return result, None, wall, rss.peak_bytes


# ---------------------------------------------------------------------------
# stage C: bind + freeze
# ---------------------------------------------------------------------------

def make_test_reader(order, allowed_days):
    return TestRoleReader(ROOT, ROOT / RELEASE_REL, order, set(allowed_days))


def stage_bind(out: Path) -> None:
    protocol_path = out / 'protocol.json'
    if not protocol_path.exists():
        raise RuntimeError('protocol.json missing; run preflight first')
    protocol = load_json(protocol_path)
    if protocol['status'] != 'frozen_after_exposed_case_preflight':
        raise RuntimeError('protocol is not the finalized frozen version')
    draft = load_json(out / 'protocol_draft.json')
    if draft['status'] != 'draft_before_exposed_case_preflight':
        raise RuntimeError('protocol draft was mutated')
    setup_torch()
    check_memory_floor(LIMITS['launch_available_gib'])
    graph = load_json(ROOT / GRAPH_REL)
    order = tuple(str(x) for x in graph['station_order'])
    rows = derive_candidates(order)
    attachment = load_json(ROOT / CANDIDATES_REL)
    allowed_days = list(attachment['allowed_model_input_dates'])
    reader = make_test_reader(order, allowed_days)
    access_path = out / 'access.jsonl'
    from training.qo_attributes import AnnualAttributes
    cfg = load_json(ROOT / REG2_REL)['config']
    attributes = AnnualAttributes.from_files(
        ROOT / 'data/pems/releases/partition_v1',
        manifest_sha256=cfg['attribute_manifest_sha256'])

    def read(day):
        return reader(day, purpose='bind_source_digest')

    manifest_entries = []
    for row in rows:
        window = episode_dates(row)
        digests = {}
        eligibility_ok = True
        blocker = None
        try:
            episode = assemble_test_episode(row, order, reader.bounds, read)
            scoring_slice = episode.scoring[BUFFER_DAYS:BUFFER_DAYS +
                                            row['duration_days'],
                                            episode.target_index]
            reference_slice = episode.reference[BUFFER_DAYS:BUFFER_DAYS +
                                                row['duration_days'],
                                                episode.target_index]
            features = np.asarray(
                [[attributes.lookup(day, station)['features']
                  for station in episode.station_order]
                 for day in episode.dates], dtype=np.float32)
            if not np.array_equal(features,
                                  np.broadcast_to(features[0], features.shape)):
                raise ValueError('date-varying static features unsupported')
            digests = {
                'input_sha256': hashlib.sha256(
                    np.ascontiguousarray(episode.values).tobytes()).hexdigest(),
                'scoring_sha256': hashlib.sha256(
                    np.ascontiguousarray(scoring_slice).tobytes()).hexdigest(),
                'reference_sha256': hashlib.sha256(
                    np.ascontiguousarray(reference_slice).tobytes()).hexdigest(),
                'visible_sha256': hashlib.sha256(
                    np.ascontiguousarray(episode.visible).tobytes()).hexdigest(),
            }
            counts = {'scoring_slots': int(scoring_slice.sum()),
                      'legal_elements': int(episode.visible.sum())}
        except Exception as error:  # noqa: BLE001 - binding blocker
            eligibility_ok = False
            blocker = repr(error)
            counts = {}
        manifest_entries.append({
            'episode_id': row['episode_id'], 'mask_id': row['mask_id'],
            'station_id': str(row['station_id']),
            'duration_days': row['duration_days'],
            'start_day': row['start_day'],
            'input_support': {'start': window[0], 'end': window[-1]},
            'eligibility_ok': eligibility_ok, 'blocker': blocker,
            'digests': digests, 'counts': counts,
        })
    for entry in reader.access_log:
        append_jsonl(access_path, {'stage': 'bind', 'role': 'test',
                                   'day': entry['day'],
                                   'sha256': entry['sha256'],
                                   'purpose': entry['purpose']})
    failed = [e for e in manifest_entries if not e['eligibility_ok']]
    if failed:
        raise RuntimeError(f'source binding blockers: '
                           f"{[e['episode_id'] for e in failed][:3]}")
    manifest = {
        'status': 'bound_no_scores',
        'protocol_sha256': sha(protocol_path),
        'candidates': manifest_entries,
        'allowed_days': allowed_days,
        'note': 'digests cover inputs, scoring mask, reference and visible '
                'support; no model score exists at this point',
    }
    write_json_atomic(out / 'test_manifest.json', manifest)
    print(json.dumps({'bound': len(manifest_entries),
                      'days_read': len(reader.access_log)}), flush=True)


def stage_freeze(out: Path) -> None:
    protocol = load_json(out / 'protocol.json')
    manifest = load_json(out / 'test_manifest.json')
    if protocol['status'] != 'frozen_after_exposed_case_preflight':
        raise RuntimeError('protocol not frozen')
    if manifest['status'] != 'bound_no_scores':
        raise RuntimeError('test manifest not bound')
    from training.qo_attributes import AnnualAttributes
    cfg = load_json(ROOT / REG2_REL)['config']
    attrs = AnnualAttributes.from_files(
        ROOT / 'data/pems/releases/partition_v1',
        manifest_sha256=cfg['attribute_manifest_sha256'])
    probe_day = manifest['allowed_days'][-1]
    probe_station = manifest['candidates'][0]['station_id']
    attrs.lookup(probe_day, probe_station)

    model_manifest = {
        'g80_checkpoint': {'path': G80_CK, 'sha256': G80_CK_SHA},
        'stage3_basis': {'path': STAGE3_BASIS, 'sha256': STAGE3_BASIS_SHA},
        'stage3_mu': {'path': STAGE3_MU, 'sha256': STAGE3_MU_SHA},
        'stage4_core': {'path': STAGE4_CORE, 'sha256': STAGE4_CORE_SHA},
        'attribute_manifest_sha256': cfg['attribute_manifest_sha256'],
        'environment': {'python': sys.version.split()[0],
                        'torch': torch.__version__, 'numpy': np.__version__,
                        'cuda': torch.cuda.get_device_name(0),
                        'deterministic_algorithms': True},
    }
    write_json_atomic(out / 'model_manifest.json', model_manifest)
    plan = {
        'status': 'frozen_before_test_prediction',
        'work_order': protocol['work_order'],
        'authorization': 'work order v2 section 0: user selected G80 and '
                         'explicitly required the Test run; whitelisted Test '
                         'inputs and scoring references only; no additional '
                         'training or Test-side tuning',
        'protocol_sha256': sha(out / 'protocol.json'),
        'protocol_draft_sha256': sha(out / 'protocol_draft.json'),
        'test_manifest_sha256': sha(out / 'test_manifest.json'),
        'model_manifest': model_manifest,
        'code_sha256': {rel: sha(ROOT / rel) for rel in CODE_FILES},
        'main_arms': protocol['main_arms'],
        'ablation_arms': protocol['ablation_arms'],
        'appendix_arms': protocol['appendix_arms'],
        'g40_run': False,
        'sealed_candidates': [e['episode_id'] for e in
                              manifest['candidates']],
        'scoring': protocol['scoring'],
        'statistics': protocol['statistics'],
        'failure_policy': protocol['failure_policy'],
        'resources': protocol['resources'],
        'tolerances': protocol['tolerances'],
        'rttc_timeout_s': protocol['rttc_timeout_s'],
        'paper_outline_update': 'reviews/paper_outline.md Test sections '
                                'updated after verified results only',
    }
    write_json_atomic(out / 'plan.json', plan)
    print(json.dumps({'plan_sha256': sha(out / 'plan.json'),
                      'candidates': len(manifest['candidates'])}), flush=True)


# ---------------------------------------------------------------------------
# stage D: predict (serial RTTC; technical status only; no scoring)
# ---------------------------------------------------------------------------

def solve_episode_v7(episode, candidate, fixed):
    from rttc_real import solve_episode
    return solve_episode(episode, candidate, fixed)


def _timed(fn, wall_limit):
    return timed('stage_call', fn, wall_limit)


def _is_implementation_error(error: Exception) -> bool:
    text = repr(error)
    markers = ('score visible leak', 'visible input integrity',
               'target support differs', 'outside whitelisted',
               'outside Test role bounds', 'escapes registered day',
               'SHA-256 mismatch', 'duplicate station', 'fails registered',
               'expected q/o profile', 'crosses role boundary',
               'peak RSS exceeds', 'exceeds frozen limit',
               'date-varying static features')
    return any(marker in text for marker in markers)


def _atomic_savez(path: Path, **arrays) -> None:
    tmp = Path(str(path) + '.tmp')
    with tmp.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def _plan_sha(out: Path) -> str:
    return sha(out / 'plan.json')


def stage_predict(out: Path) -> None:
    plan = load_json(out / 'plan.json')
    if plan['status'] != 'frozen_before_test_prediction':
        raise RuntimeError('plan not frozen for prediction')
    if sha(out / 'protocol.json') != plan['protocol_sha256']:
        raise RuntimeError('protocol SHA drift')
    if sha(out / 'test_manifest.json') != plan['test_manifest_sha256']:
        raise RuntimeError('test manifest SHA drift')
    setup_torch()
    check_memory_floor(LIMITS['launch_available_gib'])
    device = torch.device('cuda:0')
    scales = (S3_SCALE_Q, S3_SCALE_O)
    graph = load_json(ROOT / GRAPH_REL)
    order = tuple(str(x) for x in graph['station_order'])
    edges = {name: torch.tensor(value, dtype=torch.long)
             for name, value in graph['edge_index'].items()}
    basis, mu, core, g80 = load_models()
    roster = {row['episode_id']: row for row in
              load_json(ROOT / ROSTER_REL)['episodes']}
    manifest = load_json(out / 'test_manifest.json')
    allowed_days = manifest['allowed_days']
    reader = make_test_reader(order, allowed_days)
    access_path = out / 'access.jsonl'
    v7_plan = load_json(ROOT / V7_PLAN)
    candidate0 = v7_plan['candidates'][0]
    fixed = v7_plan['fixed']

    @lru_cache(maxsize=64)
    def read(day):
        return reader(day, purpose='prediction_inputs')

    predictions_dir = out / 'predictions'
    (predictions_dir / 'g80').mkdir(parents=True, exist_ok=True)
    (predictions_dir / 'rttc_v7').mkdir(parents=True, exist_ok=True)
    s3 = np.asarray(scales, dtype=np.float64)

    for index, entry in enumerate(manifest['candidates']):
        identity = entry['episode_id']
        row = roster[identity]
        g80_path = predictions_dir / 'g80' / f'case_{index:03d}.npz'
        rttc_path = predictions_dir / 'rttc_v7' / f'case_{index:03d}.npz'

        if not g80_path.exists():
            try:
                def build():
                    episode = assemble_test_episode(row, order, reader.bounds,
                                                    read)
                    if np.any(episode.visible & episode.scoring):
                        raise ValueError('score visible leak')
                    if not np.array_equal(
                            episode.values,
                            np.where(episode.visible, episode.reference, 0.)):
                        raise ValueError('visible input integrity violated')
                    return episode, build_item(episode, row, basis, mu, core,
                                               scales, edges, device)
                (episode, item), wall_b, rss_b = _timed(
                    build, LIMITS['neural_case_wall_s'])
                pred, wall_i, rss_i = _timed(
                    lambda: infer_g80(g80, item, scales, device),
                    LIMITS['neural_case_wall_s'])
                if not np.isfinite(pred.numpy()).all():
                    raise ValueError('nonfinite g80 prediction')
                duration = row['duration_days']
                scoring = item.scoring.cpu().numpy()
                truth_phys = episode.reference[BUFFER_DAYS:BUFFER_DAYS +
                                               duration,
                                               episode.target_index]
                _atomic_savez(
                    g80_path,
                    episode_id=identity, mask_id=episode.mask_id,
                    station_id=str(row['station_id']),
                    duration_days=duration, arm='g80', role='main',
                    pred_g80_phys=pred.numpy().astype(np.float64) * s3,
                    pred_core_only_phys=item.base.cpu().numpy().astype(
                        np.float64) * s3,
                    scoring=scoring, truth_phys=truth_phys,
                    coordinates=np.argwhere(scoring),
                    visible=episode.visible,
                    scales=s3, wall_s=wall_b + wall_i,
                    peak_rss_bytes=max(rss_b, rss_i),
                    plan_sha256=_plan_sha(out))
            except Exception as error:  # noqa: BLE001
                append_jsonl(out / 'case_metrics.jsonl', {
                    'index': index, 'episode_id': identity, 'arm': 'g80',
                    'role': 'main', 'status': 'failed', 'error': repr(error),
                    'traceback': traceback.format_exc()})
                print(json.dumps({'case': index + 1, 'arm': 'g80',
                                  'status': 'failed',
                                  'error': repr(error)}), flush=True)
                if _is_implementation_error(error):
                    raise SystemExit(2)

        if not rttc_path.exists():
            try:
                def rttc():
                    episode = assemble_test_episode(row, order, reader.bounds,
                                                    read)
                    if np.any(episode.visible & episode.scoring):
                        raise ValueError('score visible leak')
                    return episode, solve_episode_v7(episode, candidate0,
                                                     fixed)
                (episode, (pred, telemetry)), wall, rss = _timed(
                    rttc, plan['rttc_timeout_s'])
                duration = row['duration_days']
                target = episode.target_index
                pred_slice = pred[BUFFER_DAYS:BUFFER_DAYS + duration, target]
                if not np.isfinite(pred_slice).all():
                    raise ValueError('nonfinite rttc prediction')
                scoring = episode.scoring[BUFFER_DAYS:BUFFER_DAYS + duration,
                                          target]
                truth_phys = episode.reference[BUFFER_DAYS:BUFFER_DAYS +
                                               duration, target]
                _atomic_savez(
                    rttc_path,
                    episode_id=identity, mask_id=episode.mask_id,
                    station_id=str(row['station_id']),
                    duration_days=duration, arm='rttc_v7', role='main',
                    pred_rttc_phys=pred_slice,
                    scoring=scoring, truth_phys=truth_phys,
                    coordinates=np.argwhere(scoring),
                    scales=s3,
                    iterations=[t['iterations'] for t in telemetry],
                    residual=[t['residual'] for t in telemetry],
                    internal_scales=np.asarray([V7_SCALE_Q, V7_SCALE_O]),
                    wall_s=wall, peak_rss_bytes=rss,
                    plan_sha256=_plan_sha(out))
            except Exception as error:  # noqa: BLE001
                append_jsonl(out / 'case_metrics.jsonl', {
                    'index': index, 'episode_id': identity, 'arm': 'rttc_v7',
                    'role': 'main', 'status': 'failed', 'error': repr(error),
                    'traceback': traceback.format_exc()})
                print(json.dumps({'case': index + 1, 'arm': 'rttc_v7',
                                  'status': 'failed',
                                  'error': repr(error)}), flush=True)
                if _is_implementation_error(error):
                    raise SystemExit(2)
        for log_entry in reader.access_log:
            append_jsonl(access_path, {'stage': 'predict', 'role': 'test',
                                       'day': log_entry['day'],
                                       'sha256': log_entry['sha256'],
                                       'purpose': log_entry['purpose']})
        reader.access_log.clear()
        print(json.dumps({'case': index + 1, 'total': 21,
                          'episode_id': identity,
                          'g80': g80_path.exists(),
                          'rttc_v7': rttc_path.exists()}), flush=True)


# ---------------------------------------------------------------------------
# stage E: score (only after every registered prediction exists)
# ---------------------------------------------------------------------------

def stage_score(out: Path) -> None:
    plan = load_json(out / 'plan.json')
    if plan['status'] != 'frozen_before_test_prediction':
        raise RuntimeError('plan not frozen')
    if sha(out / 'protocol.json') != plan['protocol_sha256']:
        raise RuntimeError('protocol SHA drift')
    manifest = load_json(out / 'test_manifest.json')
    predictions_dir = out / 'predictions'
    s3 = np.asarray([S3_SCALE_Q, S3_SCALE_O], dtype=np.float64)
    rows = []
    missing = []
    for index, entry in enumerate(manifest['candidates']):
        identity = entry['episode_id']
        g80_path = predictions_dir / 'g80' / f'case_{index:03d}.npz'
        rttc_path = predictions_dir / 'rttc_v7' / f'case_{index:03d}.npz'
        if not g80_path.exists() or not rttc_path.exists():
            missing.append(identity)
            continue
        with np.load(g80_path, allow_pickle=False) as archive:
            if str(archive['plan_sha256']) != _plan_sha(out):
                raise RuntimeError(f'prediction pinned to a different plan: '
                                   f'{identity}')
            if str(archive['episode_id']) != identity:
                raise ValueError(f'prediction identity mismatch: {identity}')
            truth = archive['truth_phys']
            scoring = archive['scoring']
            pred_g80 = archive['pred_g80_phys']
            pred_core = archive['pred_core_only_phys']
            g80_meta = {'wall_s': float(archive['wall_s']),
                        'peak_rss_bytes': int(archive['peak_rss_bytes'])}
        with np.load(rttc_path, allow_pickle=False) as archive:
            if str(archive['episode_id']) != identity:
                raise ValueError(f'prediction identity mismatch: {identity}')
            if not np.array_equal(archive['scoring'], scoring):
                raise ValueError(f'scoring mask differs between arms: '
                                 f'{identity}')
            if not np.array_equal(archive['truth_phys'], truth):
                raise ValueError(f'truth differs between arms: {identity}')
            pred_rttc = archive['pred_rttc_phys']
            rttc_meta = {'wall_s': float(archive['wall_s']),
                         'peak_rss_bytes': int(archive['peak_rss_bytes']),
                         'iterations': [int(x) for x in
                                        archive['iterations']],
                         'not_converged': bool(np.any(
                             np.asarray(archive['iterations']) >= 250))}
        row = {'index': index, 'episode_id': identity,
               'station_id': entry['station_id'],
               'duration_days': entry['duration_days'],
               'start_day': entry['start_day'], 'scores': {}}
        row['scores']['g80'] = score_physical(pred_g80, truth, scoring, s3)
        row['scores']['rttc_v7'] = score_physical(pred_rttc, truth, scoring, s3)
        row['scores']['core_only'] = score_physical(pred_core, truth, scoring,
                                                    s3)
        row['resources'] = {'g80': g80_meta, 'rttc_v7': rttc_meta}
        rows.append(row)
        with (out / 'case_metrics.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')

    main_table = macro_table(rows, ('g80', 'rttc_v7'))
    ablation_table = macro_table(rows, ('g80', 'core_only'))
    by_duration = {}
    for duration in (7, 14):
        subset = [r for r in rows if r['duration_days'] == duration]
        if subset:
            by_duration[f'{duration}d'] = macro_table(subset,
                                                      ('g80', 'rttc_v7'))
    main_table['by_duration'] = by_duration
    main_paired = paired_rows(rows, 'g80', 'rttc_v7')
    ablation_paired = paired_rows(rows, 'g80', 'core_only')
    bootstrap = {channel: station_bootstrap(rows, 'g80', 'rttc_v7', channel,
                                            'profile_macro_nrmse')
                 for channel in CHANNELS}
    slopes = {'g80': slope_pairs(rows, 'g80'),
              'rttc_v7': slope_pairs(rows, 'rttc_v7')}

    def sensor(arm, channel):
        return main_table[arm][channel]['profile_macro_nrmse']['sensor_macro']

    relative = {}
    for channel in CHANNELS:
        for metric in ('profile_macro_nrmse', 'profile_rmse', 'raw_mae'):
            g80_value = main_table['g80'][channel][metric]['sensor_macro']
            rttc_value = main_table['rttc_v7'][channel][metric]['sensor_macro']
            relative[f'{channel}_{metric}'] = {
                'g80': g80_value, 'rttc_v7': rttc_value,
                'relative_improvement': (1 - g80_value / rttc_value)
                if rttc_value != 0 else None,
                'absolute_diff': g80_value - rttc_value}

    access_rows = [json.loads(line) for line in
                   (out / 'access.jsonl').read_text(encoding='utf-8')
                   .splitlines() if line]
    roles = {e['role'] for e in access_rows}
    if roles - {'test'}:
        raise RuntimeError(f'role boundary violation in access log: {roles}')
    days = {e['day'] for e in access_rows}
    allowed = set(manifest['allowed_days'])
    if not days <= allowed:
        raise RuntimeError('access log contains non-whitelisted days')

    result = {
        'status': 'completed',
        'label': 'G80 Test main comparison and core-only ablation',
        'plan_sha256': _plan_sha(out),
        'n_candidates': len(rows),
        'main_comparison': {
            'arms': {'g80': 'frozen stage-4 core + residual head',
                     'rttc_v7': 'modified RTTC v7 candidate 0 (fixed '
                                'configuration, internal scales '
                                '6264/0.212855)'},
            'table': main_table,
            'paired_differences': main_paired,
            'relative_improvement': relative,
            'station_bootstrap': bootstrap,
            'slope': slopes,
            'role': 'external-method main result (single fixed RTTC '
                    'configuration; not all RTTC configurations, not official '
                    'RTTC reproduction)',
        },
        'ablation': {
            'arms': {'g80': 'full model',
                     'core_only': 'mu+Bhat base profile (residual head '
                                  'removed, no retraining)'},
            'table': ablation_table,
            'paired_differences': ablation_paired,
            'role': 'internal module-removal ablation; not an external '
                    'baseline',
        },
        'appendix': {},
        'failures': [],
        'case_metrics_sha256': sha(out / 'case_metrics.jsonl'),
        'access_summary': {'rows': len(access_rows), 'days': len(days),
                           'roles': sorted(roles), 'test_read': True,
                           'selection_read': False, 'calibration_read': False,
                           'train_read_for_inference': False},
        'interpretation_bounds': 'descriptive Test results on the 21-case '
                                 'whitelist; no independent-episode '
                                 'significance claim; RTTC is one fixed '
                                 'configuration with a different development '
                                 'history than G80',
    }
    write_json_atomic(out / 'result.json', result)
    print(json.dumps({'status': 'completed',
                      'g80_q': sensor('g80', 'q'),
                      'rttc_q': sensor('rttc_v7', 'q'),
                      'g80_o': sensor('g80', 'o'),
                      'rttc_o': sensor('rttc_v7', 'o')}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',
                        choices=['draft', 'preflight', 'bind', 'freeze',
                                 'predict', 'score'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    stages = {'draft': stage_draft, 'preflight': stage_preflight,
              'bind': stage_bind, 'freeze': stage_freeze,
              'predict': stage_predict, 'score': stage_score}
    try:
        stages[args.stage](args.output)
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
