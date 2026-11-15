"""G80 single-model confirmation on the sealed Calibration confirmation set.

Implements reviews/2026-09-25/NEXT_G80_INDEPENDENT_CONFIRMATION_PLAN.md.
Sealed set: all 31 Calibration-role Protocol-K episodes on the frozen
20-station graph (user-authorized 2026-09-25; Test stays closed). G80 is the
primary model; G40 is the frozen reference arm; RTTC is not run (no explicit
need per plan section 1.2). Absolute acceptance thresholds and the user's 5%
relative continuation rule are frozen in plan.json before any scoring.

The shared Train/Selection loaders hard-close the Calibration role, so this
runner carries its own sealed-role reader, episode assembler and case
preparation that mirror the shared implementations line by line; the
independent verifier reimplements them again without importing this file.
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
import threading
import time
import traceback
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[0]
BASE = ROOT / 'reviews/2026-09-20/baseline_takeover'
for _path in (str(ROOT), str(BASE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np
import psutil
import torch

WORK_ORDER_REL = 'reviews/2026-09-25/NEXT_G80_INDEPENDENT_CONFIRMATION_PLAN.md'
G40_CK = 'reviews/2026-09-24/sparse_global_token_full_v1/raw_values.pt'
G40_CK_SHA = 'c4adb3dbc6ed0968860546372daaf5e042249f271ddbafbf0fb8d4e737f80332'
G80_CK = 'reviews/2026-09-24/sparse_neighbor_context_full_v1/global_control.pt'
G80_CK_SHA = 'd60e585ca94793aa903c5dda5c5306393e2bb20acc138c97393f8ddf20785659'
G40_DIR = 'reviews/2026-09-24/sparse_global_token_full_v1'
G80_DIR = 'reviews/2026-09-24/sparse_neighbor_context_full_v1'
RELEASE_REL = 'reviews/2026-09-12/data_release_final/QO_FORMAL_RELEASE.json'
RELEASE_SHA = '5c6dd879beb0f70bcd2388601736cbf0bf341f111a267f79801c478e780b9287'
ROSTER_REL = 'reviews/2026-09-12/formal_mask_bank/EPISODE_ROSTER_NON_TEST.json'
ROSTER_SHA = 'cb38dca7027a8bc087290b01cf61e6f6f2550ddbc2f2a4f4a92b807e8813bbee'
GRAPH_REL = 'data/pems/graphs/qo_pilot_v1/release_v1/graph.json'
GRAPH_SHA = '6373d9a58173b4e828fced08b57f944a1ba78ac8733125e1ee100c2f5a034eb4'
SPLIT_REL = 'reviews/2026-09-23/optimization_v1/split.json'
SPLIT_SHA = '7dd2d8a170c3e0d9e8710888ced50ca2638a9c1cc95a33d5e9a214b2992e789e'
STAGE3_RESULT = 'reviews/2026-09-23/optimization_v1/stage3_run_v1/result.json'
STAGE3_RESULT_SHA = '21864f1313d6ade9a8483375a049e7a94a54bd7e66c1165560d5fddfc4941eda'
STAGE3_BASIS = 'reviews/2026-09-23/optimization_v1/stage3_run_v1/basis.pt'
STAGE3_BASIS_SHA = '79c0f3e4f3cd04138b0b86126609fd56578108c9fb0ae20a209e05d237891e9f'
STAGE3_MU = 'reviews/2026-09-23/optimization_v1/stage3_run_v1/mu.pt'
STAGE3_MU_SHA = 'ec4e8fa8ee2a51c37ed4a6e39f59af88d6ac57adae54558148ca5d15020fe7fe'
STAGE4_CORE = 'reviews/2026-09-23/optimization_v1/stage4_run_v1/core/student.pt'
STAGE4_CORE_SHA = '261ef98f7371f1a5c8d4f8883f94d8fb8a39d3acd121f1f80d99cdaae09c7ce5'
STAGE4_RESULT = 'reviews/2026-09-23/optimization_v1/stage4_run_v1/result.json'
STAGE4_RESULT_SHA = '75fb83873b0fed427173ff8bf2b0ff72bfb4f9474375a631603302790087a87e'
REG2_REL = 'runs/current_student_train_preparation_v2/registration.json'
PREV_PLAN = 'reviews/2026-09-25/global80_rttc_init_v1/plan.json'
PREV_RESULT = 'reviews/2026-09-25/global80_rttc_init_v1/result.json'

CHANNELS = ('q', 'o')
BUFFER_DAYS = 3
SLOTS = 288
STATION_COUNT = 20
LIMITS = {'threads': 4, 'process_rss_gib': 12, 'launch_available_gib': 10,
          'runtime_available_floor_gib': 4, 'wall_s': 3600}
BOOTSTRAP_SEED = 20260925
BOOTSTRAP_RESAMPLES = 10000

# frozen absolute acceptance thresholds: G40 retained-model Train-validation
# sensor-macro level (raw_validation_metrics.jsonl aggregate, result.json
# overall_raw.residual) times a 1.05 transport margin, converted to absolute
# numbers before any sealed-set score exists.
ABSOLUTE_THRESHOLDS = {
    'derivation': 'G40 retained Train-validation sensor_macro (q NRMSE '
                  '0.06663567495460694, o NRMSE 0.13766778750488393, q MAE '
                  '317.843216411884, o MAE 0.015928389633504243) x 1.05 '
                  'transport margin; frozen before scoring',
    'q_profile_macro_nrmse_max': 0.06996745870233728,
    'o_profile_macro_nrmse_max': 0.14455117688012812,
    'q_raw_mae_max': 333.7353772324782,
    'o_raw_mae_max': 0.016724809115179455,
    'max_failed_cases': 0,
    'visible_inputs_intact_required': True,
}
RELATIVE_RULE = {
    'definition': 'sealed-set paired G80 vs G40: q sensor-macro profile NRMSE '
                  'relative reduction >= 0.05 AND o sensor-macro profile NRMSE '
                  'not worse than G40 (user-directed continuation rule)',
    'q_relative_reduction_min': 0.05,
    'o_not_worse_required': True,
}

CODE_FILES = (
    'scripts/run_g80_independent_confirmation.py',
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
    'representations/basis.py',
    'models/film_rgnn.py',
    'reviews/2026-09-21/rttc_repair/rttc_real.py',
    'reviews/2026-09-19/ai_handoff_diagnosis_20260919T051738Z/tools/rttc_stable.py',
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


def check_launch_memory() -> float:
    value = available_gib()
    if value < LIMITS['launch_available_gib']:
        raise RuntimeError(f'launch available memory {value:.2f} GiB below floor')
    return value


def check_runtime_memory() -> float:
    value = available_gib()
    if value < LIMITS['runtime_available_floor_gib']:
        raise RuntimeError(f'runtime available memory {value:.2f} GiB below floor')
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


# ---------------------------------------------------------------------------
# sealed-role loader and assembler (mirrors VerifiedRoleReader /
# assemble_episode; the shared implementations hard-close Calibration)
# ---------------------------------------------------------------------------

class SealedRoleReader:
    """Read hashed daily Calibration arrays only, mirroring VerifiedRoleReader."""

    def __init__(self, root: Path, release_path: Path, expected: str,
                 stations: tuple[str, ...]):
        raw = release_path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != RELEASE_SHA:
            raise ValueError('release SHA mismatch')
        release = json.loads(raw)
        if release['status'] != 'frozen':
            raise ValueError('release must be frozen')
        self.bounds = tuple(release['roles']['calibration'])

        def pinned(item: dict[str, str]):
            blob = (root / item['path']).read_bytes()
            if hashlib.sha256(blob).hexdigest() != item['sha256']:
                raise ValueError(f"pin mismatch {item['path']}")
            return json.loads(blob)

        entries = pinned({'path': release['u_manifest'],
                          'sha256': release['permanent_u_sha256']})['stations']
        self.universe = {str(x['station_id']): x['permanent_U'] for x in entries}
        if len(self.universe) != len(entries) or \
                any(type(v) is not bool for v in self.universe.values()):
            raise ValueError('invalid U universe')
        if len(set(stations)) != len(stations) or \
                any(self.universe.get(s, True) for s in stations):
            raise ValueError('graph includes unknown, duplicate or U station')
        days = pinned(release['array_manifest'])['days']
        self.days = {x['day']: x for x in days}
        if len(self.days) != len(days):
            raise ValueError('duplicate day binding')
        self.stations = set(stations)
        self.output_root = Path(release['output_root']).resolve()
        self.access_log: list[dict[str, str]] = []

    def __call__(self, day: str):
        date.fromisoformat(day)
        if not self.bounds[0] <= day <= self.bounds[1]:
            raise ValueError('day outside sealed calibration role; access refused')
        entry = self.days[day]
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
        self.access_log.append({'day': day, 'sha256': entry['sha256']})
        return [(day, str(s), values[i], masks[i]) for i, s in enumerate(ids)
                if str(s) in self.stations]


def assemble_sealed_episode(row, station_order, role_bounds, read_day):
    """Mirror of assemble_episode for Calibration rows (identical geometry)."""
    if row['split'] != 'calibration' or row['protocol'] != 'K':
        raise ValueError('only sealed Calibration Protocol-K episodes supported')
    if not station_order or len(set(station_order)) != len(station_order):
        raise ValueError('unique graph station order required')
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
    return CurrentEpisode(row['episode_id'], row['mask_id'], row['split'], dates,
                          tuple(station_order), lookup[target], values, visible,
                          reference, scoring)


def prepare_sealed_case(episode, basis, mu, scales, static, edges):
    """Mirror of prepare_case for Calibration (no teacher branch)."""
    import torch
    from training.pilot_data import calendar_features, rolling_history
    from training.real_student_adapter import build_student_inputs
    if any(p.requires_grad or p.grad is not None for m in (basis, mu)
           for p in m.parameters()):
        raise ValueError('representation must be frozen')
    device = next(basis.parameters()).device
    if scales.shape != (2,) or not torch.isfinite(scales).all() or \
            (scales <= 0).any():
        raise ValueError('positive frozen channel scales required')
    with torch.no_grad():
        values = torch.as_tensor(episode.values, dtype=torch.float32,
                                 device=device) / scales
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
                                    device=device) / scales
        ordinals = torch.tensor([date.fromisoformat(d).toordinal()
                                 for d in episode.dates], device=device)
        return {'inputs': inputs, 'reference': reference, 'scoring': scoring,
                'static': static, 'role': episode.role, 'ordinals': ordinals}


# ---------------------------------------------------------------------------
# neural inference (mirror of fit_sparse_neighbor_context.prepare_one/infer)
# ---------------------------------------------------------------------------

def load_models(scales):
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
    mu_ck = torch.load(ROOT / STAGE3_MU, map_location='cpu', weights_only=False)
    if not np.array_equal(np.asarray(scales), np.asarray(mu_ck['scales'])):
        raise ValueError('Train-separated channel scales differ')
    basis = GlobalProfileBasis(SLOTS, 2, 16).requires_grad_(False).eval()
    basis.load_state_dict(basis_ck['basis_state'])
    mu = MeanFunction(5, 4, 6, channels=2).requires_grad_(False).eval()
    mu.load_state_dict(mu_ck['mean_state'])
    core_state = torch.load(ROOT / STAGE4_CORE, map_location='cpu',
                            weights_only=False)
    core = make_student(rank=16, hidden=32).requires_grad_(False).eval()
    core.load_state_dict(core_state['student'])
    g40_state = torch.load(ROOT / G40_CK, map_location='cpu', weights_only=False)
    g40 = GlobalTokenProfile(dropout=.1).requires_grad_(False).eval()
    g40.load_state_dict(g40_state['model'], strict=True)
    g40.to('cuda:0')
    g80_state = torch.load(ROOT / G80_CK, map_location='cpu', weights_only=False)
    g80 = GlobalTokenProfile(dropout=.1).requires_grad_(False).eval()
    g80.load_state_dict(g80_state['model'], strict=True)
    g80.to('cuda:0')
    return basis, mu, core, g40, g80


def build_item(episode, row, basis, mu, core, scales, edges, device):
    """Mirror of prepare_one without Train group/teacher dependencies."""
    from scripts.fit_sparse_neighbor_context import PreparedEpisode, signed_chain_hops
    from training.current_release_student_data import predict_case

    scales_t = torch.tensor(scales, dtype=torch.float32)
    features = np.asarray(
        [[ATTRIBUTES.lookup(day, station)['features'] for station in episode.station_order]
         for day in episode.dates], dtype=np.float32)
    if not np.array_equal(features, np.broadcast_to(features[0], features.shape)):
        raise ValueError('date-varying static features unsupported')
    case = prepare_sealed_case(episode, basis, mu, scales_t,
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


ATTRIBUTES = None


def infer_arm(model, item, scales, device):
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
# scoring and aggregation
# ---------------------------------------------------------------------------

def score_case_np(pred_norm32, truth_norm32, scoring, scales):
    if pred_norm32.shape != truth_norm32.shape or \
            pred_norm32.shape != scoring.shape:
        raise ValueError('score shape mismatch')
    scores = {}
    for c, channel in enumerate(CHANNELS):
        active = scoring[..., c]
        count = active.sum(axis=-1)
        if not np.all(count > 0):
            raise ValueError('empty scoring profile')
        pred = pred_norm32[..., c].astype(np.float64)
        ref = truth_norm32[..., c].astype(np.float64)
        error = np.where(active, pred - ref, 0.0)
        profile_rmse = np.sqrt(np.square(error).sum(axis=-1) / count)
        scores[channel] = {
            'profile_macro_nrmse': float(profile_rmse.mean()),
            'raw_mae': float(np.abs(error[active]).mean() * float(scales[c])),
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
    values = [row['scores'][arm][channel][metric] for row in rows]
    return float(sum(values) / len(values))


def macro_table(rows, arms=('g40', 'g80')):
    out = {'n_cases': len(rows)}
    for arm in arms:
        out[arm] = {}
        for channel in CHANNELS:
            out[arm][channel] = {}
            for metric in ('profile_macro_nrmse', 'raw_mae'):
                out[arm][channel][metric] = {
                    'episode_macro': episode_macro(rows, arm, channel, metric),
                    'sensor_macro': sensor_macro(rows, arm, channel, metric),
                }
    return out


def paired_difference(rows, arm_a='g80', arm_b='g40'):
    out = {'n_cases': len(rows), 'channels': {}}
    for channel in CHANNELS:
        diffs = np.array([r['scores'][arm_a][channel]['profile_macro_nrmse'] -
                          r['scores'][arm_b][channel]['profile_macro_nrmse']
                          for r in rows], dtype=np.float64)
        mae = np.array([r['scores'][arm_a][channel]['raw_mae'] -
                        r['scores'][arm_b][channel]['raw_mae'] for r in rows])
        out['channels'][channel] = {
            'mean_diff': float(diffs.mean()),
            'median_diff': float(np.median(diffs)),
            'improved_cases': int((diffs < 0).sum()),
            'worsened_cases': int((diffs > 0).sum()),
            'mean_raw_mae_diff': float(mae.mean()),
        }
    return out


def station_bootstrap(rows, arm, channel, metric, resamples=BOOTSTRAP_RESAMPLES,
                      seed=BOOTSTRAP_SEED):
    grouped = {}
    for row in rows:
        grouped.setdefault(row['station_id'], []).append(
            row['scores'][arm][channel][metric])
    means = np.array([float(np.mean(v)) for v in grouped.values()])
    rng = np.random.default_rng(seed)
    index = rng.integers(0, means.size, size=(resamples, means.size))
    stats = means[index].mean(axis=1)
    return {'point': float(means.mean()),
            'ci95_low': float(np.percentile(stats, 2.5)),
            'ci95_high': float(np.percentile(stats, 97.5)),
            'resamples': resamples, 'seed': seed,
            'units': 'stations resampled with replacement (descriptive only)'}


# ---------------------------------------------------------------------------
# modes
# ---------------------------------------------------------------------------

def sealed_ids_from_roster(graph_order):
    roster = load_json(ROOT / ROSTER_REL)['episodes']
    graph = set(graph_order)
    rows = [row for row in roster
            if row['split'] == 'calibration' and row['protocol'] == 'K' and
            str(row['station_id']) in graph and row['duration_days'] in (7, 14)]
    rows.sort(key=lambda r: r['episode_id'])
    return rows


def exposed_id_sets():
    train268 = set(load_json(ROOT / REG2_REL)['episode_ids'])
    split = load_json(ROOT / SPLIT_REL)
    v58 = set(split['validation_ids'])
    prev_plan = load_json(ROOT / PREV_PLAN)
    rttc30 = {row['episode_id'] for row in
              load_json(ROOT / 'reviews/2026-09-21/rttc_repair/rttc_real_v7/'
                        'plan.json')['episodes']['train']}
    selreg = load_json(ROOT / 'runs/current_student_selection_preparation_v1/'
                       'registration.json')
    sel42 = set(selreg['episode_ids'])
    return {'train_prepared_268': train268, 'train_validation_58': v58,
            'rttc_tuning_30': rttc30, 'selection_prepared_42': sel42,
            'prev_comparison_58': set(prev_plan['validation_ids'])}


def verify_pins(plan):
    for name, pin in plan['input_pins'].items():
        if sha(ROOT / pin['path']) != pin['sha256']:
            raise ValueError(f'input drift {name}: {pin["path"]}')
    for rel, digest in plan['code_sha256'].items():
        if sha(ROOT / rel) != digest:
            raise ValueError(f'code drift {rel}')
    if plan['input_pins']['g40_checkpoint']['sha256'] != G40_CK_SHA or \
            plan['input_pins']['g80_checkpoint']['sha256'] != G80_CK_SHA:
        raise ValueError('checkpoint identity drift')


def resolve_pins():
    pins = {
        'work_order': (WORK_ORDER_REL, None),
        'g40_checkpoint': (G40_CK, G40_CK_SHA),
        'g80_checkpoint': (G80_CK, G80_CK_SHA),
        'split': (SPLIT_REL, SPLIT_SHA),
        'stage3_result': (STAGE3_RESULT, STAGE3_RESULT_SHA),
        'stage3_basis': (STAGE3_BASIS, STAGE3_BASIS_SHA),
        'stage3_mu': (STAGE3_MU, STAGE3_MU_SHA),
        'stage4_core': (STAGE4_CORE, STAGE4_CORE_SHA),
        'stage4_result': (STAGE4_RESULT, STAGE4_RESULT_SHA),
        'release': (RELEASE_REL, RELEASE_SHA),
        'graph': (GRAPH_REL, GRAPH_SHA),
        'roster': (ROSTER_REL, ROSTER_SHA),
        'train_registration_v2': (REG2_REL, None),
        'prev_comparison_plan': (PREV_PLAN, None),
        'prev_comparison_result': (PREV_RESULT, None),
    }
    resolved = {}
    for name, (rel, expected) in pins.items():
        digest = sha(ROOT / rel)
        if expected is not None and digest != expected:
            raise ValueError(f'pin drift {name}: {rel}')
        resolved[name] = {'path': rel, 'sha256': digest}
    return resolved


def freeze(out: Path) -> None:
    if out.exists():
        raise FileExistsError(out)
    pins = resolve_pins()
    code_sha = {rel: sha(ROOT / rel) for rel in CODE_FILES}
    graph = load_json(ROOT / GRAPH_REL)
    order = tuple(str(x) for x in graph['station_order'])
    if len(order) != STATION_COUNT:
        raise ValueError('frozen station count differs')
    rows = sealed_ids_from_roster(order)
    if len(rows) != 31:
        raise ValueError(f'expected 31 sealed episodes, found {len(rows)}')
    ids = [row['episode_id'] for row in rows]
    if len(set(ids)) != 31:
        raise ValueError('duplicate sealed id')
    exposed = exposed_id_sets()
    intersections = {name: sorted(set(ids) & used) for name, used in exposed.items()}
    if any(intersections.values()):
        raise ValueError(f'sealed set intersects exposed episodes: {intersections}')
    stage3 = load_json(ROOT / STAGE3_RESULT)
    scales = [float(x) for x in stage3['scales']]
    if scales != [6240.0, 0.21276629999999888]:
        raise ValueError(f'unexpected stage-3 scales: {scales}')
    stage4 = load_json(ROOT / STAGE4_RESULT)
    if stage4['candidate_retained_by_frozen_rule'] is not True:
        raise ValueError('frozen stage4 gate differs')
    calibration_bounds = load_json(ROOT / RELEASE_REL)['roles']['calibration']
    source_days = sorted({(date.fromisoformat(row['start_day']) -
                           timedelta(days=BUFFER_DAYS) + timedelta(days=i)).isoformat()
                          for row in rows
                          for i in range(row['duration_days'] + 2 * BUFFER_DAYS)})
    for day in source_days:
        if not calibration_bounds[0] <= day <= calibration_bounds[1]:
            raise ValueError(f'source day outside Calibration role: {day}')
    plan = {
        'status': 'frozen_before_g80_independent_confirmation',
        'work_order': dict(rel=WORK_ORDER_REL, **pins['work_order']),
        'label': 'sealed-set single-model confirmation (G80)',
        'authorization': {
            'calibration_opened': True,
            'granted_by': 'user, 2026-09-25 chat confirmation: open all 31 '
                          'graph-compatible Protocol-K Calibration episodes; '
                          'dual acceptance gates; Test stays closed',
            'test_opened': False,
        },
        'primary_model': {
            'identity': 'G80 exploratory continuation checkpoint (global_control)',
            'checkpoint': pins['g80_checkpoint'],
            'loading': "GlobalTokenProfile(dropout=.1).eval(); state_dict from "
                       "checkpoint['model']; strict",
        },
        'reference_model': {
            'identity': 'G40 retained checkpoint (raw_values)',
            'checkpoint': pins['g40_checkpoint'],
            'role': 'frozen reference arm for paired deltas; not a selection '
                    'mechanism (plan section 0/1.1)',
        },
        'rttc': {'included': False,
                 'reason': 'plan section 1.2: only on explicit need; none given',
                 'frozen_reference': {'plan': 'reviews/2026-09-21/rttc_repair/'
                                              'rttc_real_v7/plan.json',
                                      'candidate': 0}},
        'sealed_set': {
            'role': 'calibration',
            'selection_rule': 'all roster episodes with split=calibration, '
                              'protocol=K, target in frozen 20-station graph, '
                              'duration in {7,14}; no sampling',
            'count': len(rows),
            'episodes': [{'episode_id': row['episode_id'],
                          'mask_id': row['mask_id'],
                          'station_id': str(row['station_id']),
                          'duration_days': row['duration_days'],
                          'start_day': row['start_day'],
                          'support_group': None,
                          'input_sha256': row.get('input_sha256'),
                          'scoring_sha256': row.get('scoring_sha256')}
                         for row in rows],
            'episode_ids': ids,
            'station_order': list(order),
            'source_days': source_days,
            'note': 'support groups do not exist for Calibration (roster group '
                    'field is null); stratification uses duration and station',
        },
        'intersection_checks': {name: {'count': len(inter), 'ids': inter}
                                for name, inter in intersections.items()},
        'exposure_note': 'G80/G40 were fitted on Train episodes; the sealed set '
                         'shares no episode, day or station-outage mask with any '
                         'training, selection or exploration set',
        'scales': {'values': scales, 'source': STAGE3_RESULT},
        'absolute_gate': ABSOLUTE_THRESHOLDS,
        'relative_rule': RELATIVE_RULE,
        'stratification': {'duration': [7, 14], 'station': 'per-station means',
                           'support_groups': 'not applicable for Calibration'},
        'bootstrap': {'unit': 'station', 'resamples': BOOTSTRAP_RESAMPLES,
                      'seed': BOOTSTRAP_SEED},
        'limits': LIMITS,
        'access_log': 'access.jsonl',
        'stop_conditions': [
            'input SHA mismatch', 'sealed-set intersection nonempty',
            'role boundary violation', 'mask/coordinate misalignment',
            'nonfinite output', 'resource limit exceeded',
            'verifier inconsistency', 'any failed case',
        ],
        'failure_handling': 'stop the formal run on first failure; record the '
                            'case; never drop cases to fix aggregates',
        'environment': {'python': sys.version.split()[0], 'torch': torch.__version__,
                        'numpy': np.__version__, 'cuda_device': torch.cuda.get_device_name(0),
                        'deterministic_algorithms': True},
        'closed_roles': ['test'],
        'input_pins': pins,
        'code_sha256': code_sha,
    }
    out.mkdir(parents=True)
    with (out / 'plan.json').open('x', encoding='utf-8') as stream:
        json.dump(plan, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({'plan_sha256': sha(out / 'plan.json'), 'sealed': len(rows),
                      'intersections': {k: v['count'] for k, v in
                                        plan['intersection_checks'].items()}}),
          flush=True)


def load_plan(out: Path, plan_sha: str) -> dict:
    plan = load_json(out / 'plan.json')
    if sha(out / 'plan.json') != plan_sha:
        raise ValueError('plan SHA mismatch')
    return plan


def build_graph():
    graph = load_json(ROOT / GRAPH_REL)
    order = tuple(str(x) for x in graph['station_order'])
    edges = {name: torch.tensor(value, dtype=torch.long)
             for name, value in graph['edge_index'].items()}
    return order, edges


def prepare_episode(row, reader, read, order, edges, basis, mu, core, scales,
                    device):
    episode = assemble_sealed_episode(row, order, reader.bounds, read)
    if np.any(episode.visible & episode.scoring):
        raise ValueError('score visible leak')
    if not np.array_equal(episode.values, np.where(episode.visible,
                                                   episode.reference, 0.)):
        raise ValueError('visible input integrity violated')
    return build_item(episode, row, basis, mu, core, scales, edges, device)


def timed_infer(label, fn):
    check_runtime_memory()
    started = time.perf_counter()
    with TreePeakRSS() as rss:
        result = fn()
    wall = time.perf_counter() - started
    if rss.peak_bytes > LIMITS['process_rss_gib'] * 2 ** 30:
        raise RuntimeError(f'{label} peak RSS exceeds cap')
    if wall > LIMITS['wall_s']:
        raise RuntimeError(f'{label} wall {wall:.1f}s exceeds limit')
    return result, wall, rss.peak_bytes


def preflight(out: Path, plan_sha: str) -> None:
    plan = load_plan(out, plan_sha)
    verify_pins(plan)
    setup_torch()
    check_launch_memory()
    device = torch.device('cuda:0')
    scales = tuple(float(x) for x in plan['scales']['values'])
    order, edges = build_graph()
    basis, mu, core, g40, g80 = load_models(scales)

    # Part A: replication of the frozen pipeline on one exposed validation
    # episode; predictions must be bitwise equal to the archived float32 npz.
    val_dir = ROOT / G80_DIR / 'validation'
    archive_path = val_dir / '000.npz'
    with np.load(archive_path, allow_pickle=False) as archive:
        identity = str(archive['episode_id'])
        archived_scoring = archive['scoring']
    roster = {row['episode_id']: row for row in
              load_json(ROOT / ROSTER_REL)['episodes']}
    from training.current_release_student_data import VerifiedRoleReader, assemble_episode
    reader = VerifiedRoleReader(ROOT, ROOT / RELEASE_REL, RELEASE_SHA, 'train',
                                order)

    @lru_cache(maxsize=256)
    def read_train(day: str):
        return reader(day)

    train_episode = assemble_episode(roster[identity], order, reader.bounds,
                                     read_train)
    if np.any(train_episode.visible & train_episode.scoring):
        raise ValueError('score visible leak')
    item = build_item(train_episode, roster[identity], basis, mu, core, scales,
                      edges, device)
    if not np.array_equal(item.scoring.cpu().numpy(), archived_scoring):
        raise ValueError('replication scoring mask differs from archive')
    control_pred, wall_g80, rss_g80 = timed_infer(
        'replication_g80', lambda: infer_arm(g80, item, scales, device))
    raw_pred, wall_g40, rss_g40 = timed_infer(
        'replication_g40', lambda: infer_arm(g40, item, scales, device))
    with np.load(val_dir / '000.npz', allow_pickle=False) as archive:
        bitwise_control = bool(np.array_equal(control_pred.numpy(),
                                              archive['control']))
    with np.load(ROOT / G40_DIR / 'validation' / '000.npz',
                 allow_pickle=False) as archive:
        if str(archive['episode_id']) != identity:
            raise ValueError('G40 archive identity mismatch at index 000')
        bitwise_raw = bool(np.array_equal(raw_pred.numpy(), archive['raw']))
    if not bitwise_control or not bitwise_raw:
        raise RuntimeError('replication is not bitwise equal to the archived '
                           'predictions; implementation drift')

    # Part B: plan-mandated interface preflight on a Train fit episode
    # (outside the sealed set; no scoring, no sealed-set label access).
    split = load_json(ROOT / SPLIT_REL)
    fit_id = sorted(split['fit_ids'])[0]
    fit_row = roster[fit_id]
    fit_episode = assemble_episode(fit_row, order, reader.bounds, read_train)
    fit_item = build_item(fit_episode, fit_row, basis, mu, core, scales, edges,
                          device)
    fit_g80, wall_b80, rss_b80 = timed_infer(
        'preflight_fit_g80', lambda: infer_arm(g80, fit_item, scales, device))
    fit_g40, wall_b40, rss_b40 = timed_infer(
        'preflight_fit_g40', lambda: infer_arm(g40, fit_item, scales, device))
    interface = {
        'shapes_match': bool(fit_g80.shape == fit_g40.shape ==
                             (fit_row['duration_days'], SLOTS, 2)),
        'finite': bool(np.isfinite(fit_g80.numpy()).all() and
                       np.isfinite(fit_g40.numpy()).all()),
        'same_input_same_geometry': True,
    }
    if not all(interface.values()):
        raise RuntimeError('preflight interface check failed')

    report = {
        'status': 'pass',
        'plan_sha256': plan_sha,
        'replication': {
            'episode_id': identity,
            'bitwise_g80': bitwise_control,
            'bitwise_g40': bitwise_raw,
            'wall_s_g80': wall_g80, 'wall_s_g40': wall_g40,
            'peak_rss_bytes': max(rss_g80, rss_g40),
        },
        'interface': {'fit_episode_id': fit_id, **interface,
                      'wall_s_g80': wall_b80, 'wall_s_g40': wall_b40,
                      'peak_rss_bytes': max(rss_b80, rss_b40)},
        'scales_from_file': list(scales),
        'limits': LIMITS,
        'note': 'no sealed-set episode or label was read; the replication case '
                'is an already-exposed Train validation episode',
    }
    write_json_atomic(out / 'preflight.json', report)
    print(json.dumps({'status': 'pass', 'replication_bitwise':
                      {'g80': bitwise_control, 'g40': bitwise_raw}}), flush=True)


def preflight_guarded(out: Path, plan_sha: str) -> None:
    try:
        preflight(out, plan_sha)
    except Exception as error:  # noqa: BLE001
        write_json_atomic(out / 'preflight.json', {
            'status': 'fail', 'plan_sha256': plan_sha,
            'error': repr(error), 'traceback': traceback.format_exc()})
        raise


def run_one_case(out: Path, plan_sha: str, index: int, row, reader, read, order,
                 edges, basis, mu, core, g40, g80, scales, device) -> dict:
    check_runtime_memory()
    episode = assemble_sealed_episode(row, order, reader.bounds, read)
    if np.any(episode.visible & episode.scoring):
        raise ValueError('score visible leak')
    if not np.array_equal(episode.values, np.where(episode.visible,
                                                   episode.reference, 0.)):
        raise ValueError('visible input integrity violated')
    item = build_item(episode, row, basis, mu, core, scales, edges, device)
    pred_g80, wall_g80, rss_g80 = timed_infer(
        'g80', lambda: infer_arm(g80, item, scales, device))
    pred_g40, wall_g40, rss_g40 = timed_infer(
        'g40', lambda: infer_arm(g40, item, scales, device))
    if not np.isfinite(pred_g80.numpy()).all() or \
            not np.isfinite(pred_g40.numpy()).all():
        raise ValueError('nonfinite prediction')
    scoring = item.scoring.cpu().numpy()
    truth = item.reference.cpu().numpy()
    scales64 = np.asarray(scales, dtype=np.float64)
    scores = {
        'g80': score_case_np(pred_g80.numpy(), truth, scoring, scales64),
        'g40': score_case_np(pred_g40.numpy(), truth, scoring, scales64),
    }
    npz_name = f'case_{index:03d}.npz'
    npz_path = out / npz_name
    tmp = Path(str(npz_path) + '.tmp')
    with tmp.open('wb') as stream:
        np.savez_compressed(
            stream,
            episode_id=row['episode_id'], mask_id=episode.mask_id,
            station_id=str(row['station_id']), duration_days=row['duration_days'],
            pred_g80=pred_g80.numpy(), pred_g40=pred_g40.numpy(),
            scoring=scoring, truth_scored=truth[scoring],
            coordinates=np.argwhere(scoring),
            visible=episode.visible,
            input_sha256=str(row.get('input_sha256')),
            scoring_sha256=str(row.get('scoring_sha256')),
            plan_sha256=plan_sha)
    os.replace(tmp, npz_path)
    return {
        'index': index, 'episode_id': row['episode_id'],
        'mask_id': episode.mask_id, 'station_id': str(row['station_id']),
        'duration_days': row['duration_days'], 'support_group': None,
        'plan_sha256': plan_sha, 'status': 'pass', 'npz': npz_name,
        'npz_sha256': sha(npz_path), 'scores': scores,
        'resources': {'wall_s_g80': wall_g80, 'wall_s_g40': wall_g40,
                      'peak_rss_bytes': max(rss_g80, rss_g40)},
        'visible_inputs_intact': True,
    }


def run(out: Path, plan_sha: str) -> None:
    plan = load_plan(out, plan_sha)
    if not (out / 'preflight.json').exists():
        raise RuntimeError('preflight.json missing; run preflight first')
    if load_json(out / 'preflight.json').get('status') != 'pass':
        raise RuntimeError('preflight did not pass')
    verify_pins(plan)
    setup_torch()
    check_launch_memory()
    device = torch.device('cuda:0')
    scales = tuple(float(x) for x in plan['scales']['values'])
    order, edges = build_graph()
    basis, mu, core, g40, g80 = load_models(scales)
    reader = SealedRoleReader(ROOT, ROOT / RELEASE_REL, RELEASE_SHA, order)
    allowed_days = set(plan['sealed_set']['source_days'])
    access_path = out / 'access.jsonl'
    if access_path.exists():
        access_path.unlink()

    @lru_cache(maxsize=64)
    def read(day: str):
        if day not in allowed_days:
            raise ValueError(f'unregistered sealed day access refused: {day}')
        append_jsonl(access_path, {'event': 'attempt', 'role': 'calibration',
                                   'day': day})
        profiles = reader(day)
        append_jsonl(access_path, {'event': 'verified', 'role': 'calibration',
                                   'day': day})
        return profiles

    roster = {row['episode_id']: row for row in
              load_json(ROOT / ROSTER_REL)['episodes']}
    metrics_path = out / 'case_metrics.jsonl'
    done = set()
    if metrics_path.exists():
        for line in metrics_path.read_text(encoding='utf-8').splitlines():
            if not line:
                continue
            row = json.loads(line)
            if row['status'] != 'pass':
                raise RuntimeError('prior failed case present; formal run stopped')
            if row['plan_sha256'] != plan_sha:
                raise RuntimeError('case row pinned to a different plan')
            if sha(out / row['npz']) != row['npz_sha256']:
                raise RuntimeError(f'archive drift: {row["npz"]}')
            done.add(row['episode_id'])

    roster = {row['episode_id']: row for row in
              load_json(ROOT / ROSTER_REL)['episodes']}
    for index, identity in enumerate(plan['sealed_set']['episode_ids']):
        if identity in done:
            print(json.dumps({'case': index + 1, 'episode_id': identity,
                              'status': 'already_done'}), flush=True)
            continue
        try:
            case_row = run_one_case(out, plan_sha, index, roster[identity],
                                    reader, read, order, edges, basis, mu, core,
                                    g40, g80, scales, device)
        except Exception as error:  # noqa: BLE001 - record failure, stop
            append_jsonl(metrics_path, {
                'index': index, 'episode_id': identity, 'plan_sha256': plan_sha,
                'status': 'failed', 'error': repr(error),
                'traceback': traceback.format_exc()})
            print(json.dumps({'case': index + 1, 'episode_id': identity,
                              'status': 'failed', 'error': repr(error)}),
                  flush=True)
            write_result(out, plan, plan_sha)
            raise SystemExit(2)
        append_jsonl(metrics_path, case_row)
        print(json.dumps({'case': index + 1, 'total': 31,
                          'episode_id': identity,
                          'status': case_row['status']}), flush=True)
    write_result(out, plan, plan_sha)


def write_result(out: Path, plan: dict, plan_sha: str) -> None:
    rows = [json.loads(line) for line in
            (out / 'case_metrics.jsonl').read_text(encoding='utf-8').splitlines()
            if line]
    passed = [row for row in rows if row['status'] == 'pass']
    failed = [row for row in rows if row['status'] != 'pass']
    expected = plan['sealed_set']['count']
    status = 'completed' if len(passed) == expected else 'stopped_with_failures'
    if not passed:
        write_json_atomic(out / 'result.json', {
            'status': status, 'plan_sha256': plan_sha,
            'validation_episodes': 0, 'failures': failed,
            'exploratory_disclaimer': 'formal run stopped; no scores'})
        print(json.dumps({'status': status, 'failures': len(failed)}), flush=True)
        return

    tables = {'all': macro_table(passed)}
    for duration in (7, 14):
        subset = [r for r in passed if r['duration_days'] == duration]
        if subset:
            tables[f'duration_{duration}'] = macro_table(subset)
    for station in sorted({r['station_id'] for r in passed}):
        tables[f'station_{station}'] = macro_table(
            [r for r in passed if r['station_id'] == station])
    paired = paired_difference(passed)
    bootstrap = {arm: {channel: station_bootstrap(passed, arm, channel,
                                                  'profile_macro_nrmse')
                       for channel in CHANNELS} for arm in ('g40', 'g80')}

    def sensor(arm, channel):
        return tables['all'][arm][channel]['profile_macro_nrmse']['sensor_macro']

    def mae(arm, channel):
        return tables['all'][arm][channel]['raw_mae']['sensor_macro']

    absolute = {}
    thresholds = plan['absolute_gate']
    for channel in CHANNELS:
        absolute[f'{channel}_profile_macro_nrmse'] = {
            'threshold': thresholds[f'{channel}_profile_macro_nrmse_max'],
            'observed': sensor('g80', channel),
            'pass': bool(sensor('g80', channel) <=
                         thresholds[f'{channel}_profile_macro_nrmse_max'])}
        absolute[f'{channel}_raw_mae'] = {
            'threshold': thresholds[f'{channel}_raw_mae_max'],
            'observed': mae('g80', channel),
            'pass': bool(mae('g80', channel) <=
                         thresholds[f'{channel}_raw_mae_max'])}
    absolute['failed_cases'] = {'threshold': thresholds['max_failed_cases'],
                                'observed': len(failed),
                                'pass': len(failed) <= thresholds['max_failed_cases']}
    absolute['visible_inputs_intact'] = {
        'required': thresholds['visible_inputs_intact_required'],
        'observed': bool(all(r['visible_inputs_intact'] for r in passed)),
        'pass': bool(all(r['visible_inputs_intact'] for r in passed))}
    absolute['pass'] = bool(all(v['pass'] for v in absolute.values()
                                if isinstance(v, dict)))

    q_red = 1 - sensor('g80', 'q') / sensor('g40', 'q')
    relative = {'q_relative_reduction': q_red,
                'o_g80': sensor('g80', 'o'), 'o_g40': sensor('g40', 'o'),
                'q_min_required': plan['relative_rule']['q_relative_reduction_min'],
                'q_pass': bool(q_red >=
                               plan['relative_rule']['q_relative_reduction_min']),
                'o_pass': bool(sensor('g80', 'o') <= sensor('g40', 'o'))}
    relative['pass'] = bool(relative['q_pass'] and relative['o_pass'])

    access_rows = [json.loads(line) for line in
                   (out / 'access.jsonl').read_text(encoding='utf-8').splitlines()
                   if line]
    roles = {row['role'] for row in access_rows}
    days = {row['day'] for row in access_rows}
    if roles - {'calibration'}:
        raise RuntimeError(f'role boundary violation: {roles}')
    if not days <= set(plan['sealed_set']['source_days']):
        raise RuntimeError('access log contains unregistered days')

    result = {
        'status': status,
        'label': plan['label'],
        'plan_sha256': plan_sha,
        'sealed_set': {'role': 'calibration', 'count': len(passed)},
        'scales': {'values': list(plan['scales']['values'])},
        'tables': tables,
        'paired_differences': paired,
        'station_bootstrap': bootstrap,
        'absolute_gate': absolute,
        'relative_rule': relative,
        'failures': failed,
        'role_access': {'log': 'access.jsonl', 'rows': len(access_rows),
                        'days': len(days), 'test_read': False,
                        'selection_read': False, 'train_read_for_inference': False},
        'interpretation_bounds': 'single-model confirmation on the sealed set '
                                 'only; no superiority claim over G40 beyond the '
                                 'pre-frozen relative rule, no Test-set '
                                 'generalization claim',
        'limits': LIMITS,
    }
    write_json_atomic(out / 'result.json', result)
    print(json.dumps({'status': status,
                      'absolute_gate': absolute['pass'],
                      'relative_rule': relative['pass'],
                      'g80_q': sensor('g80', 'q'), 'g80_o': sensor('g80', 'o'),
                      'g40_q': sensor('g40', 'q'), 'g40_o': sensor('g40', 'o')}),
          flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['freeze', 'preflight', 'run'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plan-sha256')
    args = parser.parse_args()
    if args.mode == 'freeze':
        freeze(args.output)
        return
    if not args.plan_sha256:
        parser.error('--plan-sha256 is required for preflight and run')
    if args.mode == 'preflight':
        preflight_guarded(args.output, args.plan_sha256)
    else:
        try:
            run(args.output, args.plan_sha256)
        except SystemExit:
            raise
        except Exception:
            traceback.print_exc()
            raise


if __name__ == '__main__':
    main()
