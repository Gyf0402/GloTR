"""Frozen G80 + dual-channel low-rank transductive residual correction.

Implements ``reviews/2026-09-26/TRANSDUCTIVE_CORRECTION_PREREGISTRATION.md`` with
the mechanism filled in by ``..._AMENDMENT.md`` (the prereg's section 2 is not
literally implementable under the frozen target-centric forward).

Stages
------
``gates``  frozen-file SHAs and the completeness gates G0b/G1/G2/G2b/G2c/G3/G4/G5
``field``  per-episode, CONFIG-INDEPENDENT caches: control prediction, donor error
           field, target pseudo-outage errors.  Resumable per case.
``sweep``  the pre-registered 18 configurations scored on the Train queue only
``score``  the chosen configuration applied on Selection-42, three arms, one scorer
``all``    gates -> field(train) -> sweep -> field(selection) -> score -> report

Nothing here writes to a SHA-pinned artefact; inference reuses the trainer's
frozen queue/assembly/forward path by import.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.train_g80_v2_arm as arm_mod  # noqa: E402
from models.arm_profiles import build_profile  # noqa: E402
from models.sparse_global_tokens import global_token_context  # noqa: E402
from models.transductive_residual import (  # noqa: E402
    estimate_loading, fit_temporal_factors)
from scripts.score_selection_three_arm import (  # noqa: E402
    OUR_SCALES, archived_rttc_predictions, evaluate, profile_macro)

BUFFER = arm_mod.BUFFER_DAYS
SLOTS = arm_mod.SLOTS
CHANNELS = ('q', 'o')

PREREG = ROOT / 'reviews/2026-09-26/TRANSDUCTIVE_CORRECTION_PREREGISTRATION.md'
AMENDMENT = ROOT / 'reviews/2026-09-26/TRANSDUCTIVE_CORRECTION_AMENDMENT.md'
CONTROL = (ROOT / 'reviews/2026-09-26/g80_v2_round2/stability/'
           'seed20260905/full.pt')
CONTROL_SHA = 'a5fac4ac159ccc08be146ae41e41c8de771e03e8fc8d2eeec85b78b9a3f646d5'
SELECTION_MANIFEST = ROOT / 'reviews/2026-09-26/a6/selection_manifest.json'
OUTBASE = ROOT / 'reviews/2026-09-26/transductive'

#: Pre-registered search space. Do not extend while looking at Selection.
RANKS = (1, 2, 4)
LAM_MAGS = (0.1, 1.0, 10.0)
LAM_SMOOTHS = (0.1, 1.0)
N_ITER = 20
#: every 5th donor is held out of the low-rank fit and used only for stopping
VALIDATION_STRIDE = 5
#: Pseudo-outage length for the target's own loading regression. The frozen
#: ``sparse_context`` accepts only 7/14/28-day outages (``days - 6 in {7,14,28}``),
#: so the amendment's "3-day" outage is not implementable; 7 is the smallest
#: legal value. The window is placed by the rule frozen in
#: ``pseudo_availability.json`` (see the amendment), because some stations are
#: dark for whole weeks and some days are dark for every station.
PSEUDO_DURATION = 7
AVAILABILITY = OUTBASE / 'pseudo_availability.json'

TRAIN_QUEUE = 'train_holdout_58'
SELECT_QUEUE = 'selection_42'


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# variant assembler (mirror of the frozen one, arbitrary duration / target)
# ---------------------------------------------------------------------------

def assemble_variant(order, read, *, target_station: str, start_day: str,
                     duration: int, extra_masked: tuple[str, ...] = (),
                     support: str = 'legal'):
    """Mirror ``assemble_episode`` with an arbitrary target window and duration.

    Returns ``(episode, legal)`` where ``legal`` is the target station's true
    legality mask on the gap days (captured BEFORE it is hidden), which is what
    the frozen assembler puts into ``scoring``.

    Differences from the frozen assembler, all deliberate:

    * ``duration`` is not restricted to {7,14,28} (the pseudo-outage uses 3).
    * the registered target-completeness gate (>=274 legal slots per gap day) is
      not applied to EXTRA masked stations -- it is a target-eligibility rule, and
      a donor with a partly missing day is still a usable error sample.
    * ``support='geometry'`` sets the gap support to the FULL geometry instead of
      the legality mask. ``prepare_item`` decodes one base row per selected
      (day, station) pair, so a donor with a wholly missing gap day would
      otherwise shift the base rows and break the forward loop. For any station
      whose every gap day has at least one legal slot -- which the eligibility
      gate guarantees for a registered target -- the two supports select the same
      rows and the prediction is identical, so ``support='legal'`` is used for the
      G0b equivalence check against the frozen assembler.

    Everything else -- the reference/visible fill, capturing legality before the
    target is hidden, the values zeroing -- follows the frozen code line for line.
    """
    from training.current_release_student_data import CurrentEpisode

    if support not in ('legal', 'geometry'):
        raise ValueError('support must be legal or geometry')
    lookup = {station: index for index, station in enumerate(order)}
    if target_station not in lookup:
        raise ValueError('target outside graph order')
    for station in extra_masked:
        if station not in lookup:
            raise ValueError('masked station outside graph order')
    start = date.fromisoformat(start_day) - timedelta(days=BUFFER)
    dates = tuple((start + timedelta(days=i)).isoformat()
                  for i in range(duration + 2 * BUFFER))
    shape = (len(dates), len(order), SLOTS, 2)
    reference = np.zeros(shape, dtype=np.float64)
    visible = np.zeros(shape, dtype=bool)
    for day_index, day in enumerate(dates):
        seen = set()
        for label, station, values, mask in read(day):
            if label != day or station in seen:
                raise ValueError('duplicate station or day label mismatch')
            seen.add(station)
            if station not in lookup:
                continue
            if values.shape != (SLOTS, 2) or mask.shape != values.shape \
                    or mask.dtype != bool:
                raise ValueError('expected q/o profile and boolean legal mask')
            if not np.isfinite(values[mask]).all():
                raise ValueError('nonfinite legal values')
            station_index = lookup[station]
            reference[day_index, station_index] = np.where(mask, values, 0.)
            visible[day_index, station_index] = mask
    target_index = lookup[target_station]
    gap = slice(BUFFER, BUFFER + duration)
    legal = visible[gap, target_index].copy()
    scoring = np.zeros(shape, dtype=bool)
    if support == 'geometry':
        scoring[gap, target_index] = True
    else:
        scoring[gap, target_index] = legal
    visible[gap, target_index] = False
    for station in extra_masked:
        visible[gap, lookup[station]] = False
    values = np.where(visible, reference, 0.)
    episode = CurrentEpisode(f'variant:{target_station}:{start_day}:{duration}',
                             'variant', 'train', dates, tuple(order),
                             target_index, values, visible, reference, scoring)
    return episode, legal


# ---------------------------------------------------------------------------
# frozen forward (identical to scripts/score_selection_three_arm.py)
# ---------------------------------------------------------------------------

def run_forward(item, model, key_groups, duration):
    blocks = []
    with torch.no_grad():
        for day in range(BUFFER, BUFFER + duration):
            query, keys, valid = global_token_context(
                values=item.values, visible=item.visible,
                base=item.base[day - BUFFER], static=item.static,
                calendar=item.calendar, target=item.target, day=day,
                signed_hops=item.hops, scales=arm_mod.S3, raw_values=True,
                key_groups=key_groups)
            out, _ = model(query, keys, valid, item.base[day - BUFFER])
            blocks.append(out.reshape(SLOTS, 2))
    return (torch.cat(blocks).reshape(duration, SLOTS, 2)
            .cpu().numpy().astype(np.float64))


class Engine:
    """Loaded frozen components, shared by every stage."""

    def __init__(self, control: Path = CONTROL):
        self.control = Path(control)
        self.control_sha = sha(self.control)
        arm_mod.setup_torch()
        self.device = torch.device('cuda:0')
        self.scales = arm_mod.S3
        graph = arm_mod.load_json(ROOT / arm_mod.GRAPH_REL)
        self.order = tuple(str(x) for x in graph['station_order'])
        self.edges = {name: torch.tensor(value, dtype=torch.long)
                      for name, value in graph['edge_index'].items()}
        self.basis, self.mu, self.core = arm_mod.load_models(self.scales)
        payload = torch.load(self.control, map_location='cpu',
                             weights_only=False)
        self.model = build_profile('full', dropout=.1).requires_grad_(False).eval()
        self.model.load_state_dict(payload['model'], strict=True)
        self.model = self.model.to(self.device)
        self.key_groups = arm_mod.arm_key_groups('full')
        self.queues = arm_mod.build_queues(self.order)

    def prepare(self, episode, row):
        return arm_mod.prepare_item(episode, row, self.basis, self.mu, self.core,
                                    self.scales, self.edges, self.device)

    def forward(self, item, duration):
        return run_forward(item, self.model, self.key_groups, duration)

    def reader(self, queue):
        cache = lru_cache(maxsize=512)(lambda day: queue['reader'](day))
        return cache

    def band_reader(self, queue):
        """Reader that also permits days of the role band BEFORE this queue's.

        Needed only for the pseudo-outage window of the earliest Selection
        episodes (see build_case). The Train band precedes Selection in this
        release, so the fallback only ever reaches already fully exposed Train
        days; the count is recorded per case and reported.
        """
        from training.current_release_student_data import VerifiedRoleReader

        fallback = VerifiedRoleReader(ROOT, ROOT / arm_mod.RELEASE_REL,
                                      arm_mod.RELEASE_SHA, 'train', self.order)
        readers = [queue['reader'], fallback]

        def read(day):
            for candidate in readers:
                if candidate.bounds[0] <= day <= candidate.bounds[1]:
                    return candidate(day)
            raise ValueError(f'day {day} outside every permitted band')

        return lru_cache(maxsize=512)(read)


def station_index(order, station):
    return order.index(station)


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------

def gate_sha() -> dict:
    files = {
        'preregistration': PREREG,
        'amendment': AMENDMENT,
        'selection_manifest': SELECTION_MANIFEST,
        'control_checkpoint': CONTROL,
        'module': ROOT / 'models/transductive_residual.py',
        'module_test': ROOT / 'scripts/test_transductive_residual.py',
        'scorer': ROOT / 'scripts/score_selection_three_arm.py',
    }
    report = {name: {'path': str(path.relative_to(ROOT)), 'sha256': sha(path)}
              for name, path in files.items()}
    report['control_checkpoint_matches_expected'] = (
        report['control_checkpoint']['sha256'] == CONTROL_SHA)
    return report


def gate_g0b(engine) -> dict:
    """The variant assembler must reproduce the frozen one bitwise."""
    queue = engine.queues[TRAIN_QUEUE]
    read = engine.reader(queue)
    row = queue['rows'][0]
    frozen = queue['assemble'](row, read)
    variant, _ = assemble_variant(
        engine.order, read, target_station=str(row['station_id']),
        start_day=row['start_day'], duration=row['duration_days'],
        support='legal')
    checks = {
        'episode_id': variant.episode_id,
        'target_index_equal': variant.target_index == frozen.target_index,
        'values_bitwise_equal': bool(np.array_equal(variant.values, frozen.values)),
        'visible_bitwise_equal': bool(np.array_equal(variant.visible, frozen.visible)),
        'reference_bitwise_equal': bool(np.array_equal(variant.reference,
                                                       frozen.reference)),
        'scoring_bitwise_equal': bool(np.array_equal(variant.scoring, frozen.scoring)),
        'dates_equal': variant.dates == frozen.dates,
    }
    checks['passed'] = all(v for k, v in checks.items()
                           if k.endswith('equal') or k == 'target_index_equal')
    return checks


def gate_g5(tolerance: float = 1e-8) -> dict:
    """Our control numbers must reproduce the official `full` row.

    Exact equality is NOT achievable: the official pipeline accumulates in
    float32 while this scorer keeps float64 from the first reduction, so the two
    agree only to ~1e-9 relative. The gate therefore uses an absolute tolerance
    and records the actual deviation instead of claiming bitwise identity.
    """
    unified = OUTBASE / 'unified_scores.json'
    official = (ROOT / 'reviews/2026-09-26/loss/dev/seed20260905/'
                'retrained_arms_result.json')
    if not unified.exists() or not official.exists():
        return {'passed': False, 'why': 'missing scorer output or official result',
                'unified': str(unified), 'official': str(official)}
    ours = json.loads(unified.read_text(encoding='utf-8'))['arms']['full']
    theirs = json.loads(official.read_text(encoding='utf-8'))
    theirs = theirs['queues'][SELECT_QUEUE]['full']
    checks = {}
    for channel in CHANNELS:
        official_value = theirs[channel]['profile_macro_nrmse']['episode_macro']
        mine = ours['episode_macro'][channel]
        checks[channel] = {
            'ours_episode_macro': mine,
            'official_episode_macro': official_value,
            'abs_diff': abs(mine - official_value),
            'rel_diff': abs(mine - official_value) / abs(official_value),
            'within_tolerance': abs(mine - official_value) <= tolerance,
        }
    return {'passed': all(v['within_tolerance'] for v in checks.values()),
            'tolerance': tolerance,
            'why_not_bitwise': ('official accumulates float32, this scorer '
                                'float64; agreement is ~1e-9 relative'),
            'episode_macro': checks}


def gate_g1_g3(engine) -> dict:
    """G1 zero correction is bitwise neutral; G3 q/o state is never shared."""
    from models.transductive_residual import compose

    rng = np.random.default_rng(0)
    g80 = rng.normal(size=(7, SLOTS, 2))
    zero = np.zeros_like(g80)
    g1 = bool(np.array_equal(compose(g80, zero), g80)) and compose(g80, None) is g80

    field_q = rng.normal(size=(19, 7 * SLOTS))
    field_o = rng.normal(size=(19, 7 * SLOTS))
    observed = np.ones_like(field_q, dtype=bool)
    validation = np.array([0, 5, 10, 15])
    fit_rows = np.array([i for i in range(19) if i not in set(validation)])
    fit_q = fit_temporal_factors(field_q, observed, rank=2, lam_mag=0.1,
                                 lam_smooth=0.1, n_iter=5, day_length=SLOTS,
                                 validation_rows=validation, fit_rows=fit_rows)
    fit_o = fit_temporal_factors(field_o, observed, rank=3, lam_mag=1.0,
                                 lam_smooth=0.1, n_iter=5, day_length=SLOTS,
                                 validation_rows=validation, fit_rows=fit_rows)
    snapshot_q = fit_q.V.copy()
    _ = estimate_loading(rng.normal(size=fit_o.V.shape[0]), fit_o.V,
                         np.ones(fit_o.V.shape[0], dtype=bool), lam=1.0)
    g3 = bool(np.array_equal(snapshot_q, fit_q.V)) and not np.array_equal(
        fit_q.V.shape, fit_o.V.shape)
    return {'passed': bool(g1 and g3),
            'G1_zero_correction_bitwise_neutral': g1,
            'G3_channels_independent': bool(g3)}


# ---------------------------------------------------------------------------
# field: per-case, config-independent caches
# ---------------------------------------------------------------------------

def load_availability() -> tuple[dict, dict]:
    """The frozen pseudo-outage placement rule, one row per episode."""
    if not AVAILABILITY.exists():
        raise SystemExit('pseudo_availability.json missing; run '
                         'scripts/scan_pseudo_availability.py first')
    payload = json.loads(AVAILABILITY.read_text(encoding='utf-8'))
    table = {(queue, row['episode_id']): row
             for queue, entry in payload['queues'].items()
             for row in entry['rows']}
    return payload, table


def case_cache(queue_name: str, episode_id: str, tag: str = '') -> Path:
    # episode ids contain '|', which is not a legal Windows filename character
    safe = re.sub(r'[^0-9A-Za-z._-]', '_', episode_id)
    return OUTBASE / (queue_name + (f'__{tag}' if tag else '')) / f'case_{safe}.npz'


def build_case(engine, queue_name: str, row, donor_limit: int = 0,
               availability: dict | None = None) -> dict:
    queue = engine.queues[queue_name]
    read = engine.reader(queue)
    band_read = engine.band_reader(queue)
    order = engine.order
    scales = np.asarray(engine.scales, dtype=np.float64)
    duration = row['duration_days']
    target = str(row['station_id'])
    target_index = station_index(order, target)
    gap = slice(BUFFER, BUFFER + duration)
    plan = None
    if availability is not None:
        plan = availability.get((queue_name, row['episode_id']))
        if plan is None:
            raise ValueError(f'no availability row for {row["episode_id"]}')

    settings = []
    started = time.perf_counter()

    # --- control arm through the FROZEN path (must match the scorer bitwise)
    episode = queue['assemble'](row, read)
    if np.any(episode.visible & episode.scoring):
        raise ValueError('G2 failed: a scored cell is visible')
    item = engine.prepare(episode, row)
    pred_control = engine.forward(item, duration)
    truth_control = episode.reference[gap, target_index].astype(np.float64)
    scoring_control = episode.scoring[gap, target_index].astype(bool)
    settings.append({'step': 'control', 'seconds': time.perf_counter() - started})

    # --- donor error field: re-target to each donor, mask donor AND the target
    donors = [station for station in order if station != target]
    if donor_limit:
        donors = donors[:donor_limit]
    field = np.zeros((len(donors), duration, SLOTS, 2))
    observed = np.zeros((len(donors), duration, SLOTS, 2), dtype=bool)
    donor_failures: list[dict] = []
    leak_max = 0.0
    for index, donor in enumerate(donors):
        started = time.perf_counter()
        donor_index = station_index(order, donor)
        donor_row = {**row, 'station_id': donor}
        variant, donor_legal = assemble_variant(
            order, read, target_station=donor, start_day=row['start_day'],
            duration=duration, extra_masked=(target,), support='geometry')
        # G2 : the donor's own gap truth is never an input to its own prediction
        # G2b: neither is the registered target's gap truth (else the correction
        #      would be derived from the scoring labels through the donor field)
        leak_max = max(leak_max,
                       float(variant.visible[gap, donor_index].sum()),
                       float(variant.visible[gap, target_index].sum()))
        donor_item = engine.prepare(variant, donor_row)
        try:
            donor_pred = engine.forward(donor_item, duration)
        except ValueError as error:
            # A wholly dark day inside the gap leaves fewer than one legal key per
            # target slot. That donor then contributes no error sample, which is
            # recorded rather than silently dropped or allowed to crash the batch.
            if 'legal key' not in str(error):
                raise
            donor_failures.append({'donor': donor, 'reason': str(error)})
            field[index] = 0.0
            observed[index] = False
            settings.append({'step': f'donor:{donor}:failed',
                             'seconds': time.perf_counter() - started})
            continue
        # zero the illegal cells: `reference` is 0 where a slot is illegal, so an
        # unmasked error there would be a fake -prediction of full magnitude. The
        # fitter also masks on `observed`; both belts are kept on purpose.
        error = (variant.reference[gap, donor_index] / scales - donor_pred)
        field[index] = np.where(episode.visible[gap, donor_index], error, 0.0)
        # truth validity comes from the ORIGINAL episode (the donor is visible
        # there during the gap); the variant has it masked by construction. The
        # variant's own legality copy is asserted to agree, so the two sources
        # cannot silently diverge.
        if not np.array_equal(donor_legal, episode.visible[gap, donor_index]):
            raise ValueError(f'donor legality mismatch for {donor}')
        observed[index] = episode.visible[gap, donor_index]
        settings.append({'step': f'donor:{donor}',
                         'seconds': time.perf_counter() - started})

    # --- target pseudo-outage, placed by the frozen availability rule
    started = time.perf_counter()
    pseudo_offset = None if plan is None else plan['chosen_offset']
    pseudo_usable = pseudo_offset is not None
    pseudo_error = np.zeros((PSEUDO_DURATION, SLOTS, 2))
    pseudo_observed = np.zeros((PSEUDO_DURATION, SLOTS, 2), dtype=bool)
    pseudo_days_legal = 0
    crossed = False
    if pseudo_usable:
        pseudo_start = (date.fromisoformat(row['start_day'])
                        - timedelta(days=PSEUDO_DURATION + BUFFER
                                    + 7 * pseudo_offset)).isoformat()
        pseudo, pseudo_legal = assemble_variant(
            order, band_read, target_station=target, start_day=pseudo_start,
            duration=PSEUDO_DURATION, support='geometry')
        pseudo_gap = slice(BUFFER, BUFFER + PSEUDO_DURATION)
        pseudo_dates = set(pseudo.dates)
        gap_dates = set(episode.dates[BUFFER:BUFFER + duration])
        if pseudo_dates & gap_dates:
            raise ValueError('G2c failed: pseudo-outage overlaps the real gap')
        band_start = queue['reader'].bounds[0]
        crossed = bool(pseudo.dates[0] < band_start)
        pseudo_masked = float(pseudo.visible[pseudo_gap, target_index].sum())
        pseudo_days_legal = int(pseudo_legal.all(axis=(1, 2)).sum())
        pseudo_item = engine.prepare(pseudo, {**row, 'start_day': pseudo_start,
                                              'duration_days': PSEUDO_DURATION})
        pseudo_pred = engine.forward(pseudo_item, PSEUDO_DURATION)
        pseudo_error = (pseudo.reference[pseudo_gap, target_index] / scales
                        - pseudo_pred)
        pseudo_observed = pseudo_legal.astype(bool)
        pseudo_error = np.where(pseudo_observed, pseudo_error, 0.0)
        settings.append({'step': 'pseudo',
                         'seconds': time.perf_counter() - started})

    return {
        'episode_id': row['episode_id'],
        'target_station': target,
        'duration_days': duration,
        'start_day': row['start_day'],
        'donors': np.asarray(donors),
        'donor_density': observed.reshape(len(donors), -1).mean(axis=1),
        'pred_control': pred_control,
        'truth_control': truth_control,
        'scoring_control': scoring_control,
        'field': field,
        'observed': observed,
        'pseudo_error': pseudo_error,
        'pseudo_observed': pseudo_observed,
        'leak_hidden_cells': leak_max,
        'pseudo_masked_cells': pseudo_masked if pseudo_usable else 0.0,
        'pseudo_days_fully_legal': pseudo_days_legal,
        'pseudo_crosses_band_start': bool(crossed),
        'pseudo_usable': bool(pseudo_usable),
        'pseudo_offset_weeks': -1 if pseudo_offset is None else int(pseudo_offset),
        'donor_failures': len(donor_failures),
        'case_seconds': float(sum(s['seconds'] for s in settings)),
        'step_seconds': np.asarray([s['seconds'] for s in settings]),
    }


def save_case(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp.npz')
    np.savez_compressed(tmp, **payload)
    tmp.replace(path)


def load_case(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


# ---------------------------------------------------------------------------
# fitting
# ---------------------------------------------------------------------------

def _slot_average(signal, observed):
    """Average a (..., slots) field over its leading axes, per slot."""
    counts = observed.sum(axis=tuple(range(observed.ndim - 1)))
    flat = observed.reshape(-1, observed.shape[-1])
    values = signal.reshape(-1, signal.shape[-1])
    total = (values * flat).sum(axis=0)
    return np.divide(total, np.maximum(counts, 1), out=np.zeros_like(total),
                     where=counts > 0), counts


def fit_channel(field, observed, pseudo_error, pseudo_observed, duration,
                rank, lam_mag, lam_smooth, pseudo_usable, drop_dark=False):
    """Return (residual, fit, loading, loading_source, corridor_loading).

    ``field`` is (S, D, 288). Two loading estimators are available:

    * the pre-registered one, from the target's own MASKED pseudo-outage errors;
    * a fallback that is the mean of the donors' own loadings on the same temporal
      basis ("this station looks like the corridor average"). It is used only when
      the frozen availability rule found no admissible pseudo window -- some
      stations are dark for whole weeks -- and it uses nothing but the donor error
      field, so it stays free of the target's gap labels. The split is reported.

    The stop-iteration holdout takes every ``VALIDATION_STRIDE``-th donor **among
    those with usable observations**: a donor that is dark for the whole window
    cannot score a validation RMSE, and an all-NaN validation column would pin
    the stopping rule at the first iteration.
    """
    n_station, n_day, n_slot = field.shape
    density = observed.reshape(n_station, -1).mean(axis=1)
    # ``drop_dark`` is used ONLY by the post-hoc sensitivity row: a station that is
    # dark for the whole window cannot inform a low-rank error field, yet soft
    # imputation would insert a constant row for it that the SVD then has to
    # explain. The primary path keeps all 19 donors as pre-registered.
    if drop_dark:
        keep = np.flatnonzero(density > 0.0)
        if keep.size:
            field, observed, density = field[keep], observed[keep], density[keep]
            n_station = field.shape[0]
    eligible = np.flatnonzero(density > 0.5)
    if eligible.size < VALIDATION_STRIDE:
        eligible = np.arange(n_station)
    validation = eligible[::VALIDATION_STRIDE]
    fit_rows = np.asarray([i for i in range(n_station)
                           if i not in set(validation.tolist())])
    fit = fit_temporal_factors(
        field.reshape(n_station, n_day * n_slot),
        observed.reshape(n_station, n_day * n_slot),
        rank=rank, lam_mag=lam_mag, lam_smooth=lam_smooth, n_iter=N_ITER,
        day_length=n_slot, validation_rows=validation, fit_rows=fit_rows)
    basis = fit.V.reshape(n_day, n_slot, fit.V.shape[1])
    average_basis = basis.mean(axis=0)

    per_donor = []
    for index in range(n_station):
        slot_valid = observed[index].any(axis=0)
        if int(slot_valid.sum()) < fit.V.shape[1]:
            continue
        average, _ = _slot_average(field[index], observed[index])
        per_donor.append(estimate_loading(average, average_basis, slot_valid,
                                          lam=lam_mag))
    corridor = (np.mean(np.stack(per_donor), axis=0) if per_donor
                else np.zeros(fit.V.shape[1]))

    source = 'corridor_mean'
    loading = corridor
    if pseudo_usable:
        average, counts = _slot_average(pseudo_error, pseudo_observed)
        if int((counts > 0).sum()) >= fit.V.shape[1]:
            loading = estimate_loading(average, average_basis, counts > 0,
                                       lam=lam_mag)
            source = 'pseudo_outage'
    residual = basis @ loading
    return residual, fit, loading, source, corridor, len(per_donor)


def corrected_case(case: dict, rank, lam_mag, lam_smooth, loading_mode='auto',
                   drop_dark=False):
    """Per-channel corrections: (2, D, 288)."""
    duration = int(case['duration_days'])
    usable = bool(case.get('pseudo_usable', True))
    out = []
    diagnostics = {}
    corridor_only = []
    for channel_index, channel in enumerate(CHANNELS):
        residual, fit, loading, source, corridor, n_donor = fit_channel(
            case['field'][..., channel_index],
            case['observed'][..., channel_index],
            case['pseudo_error'][..., channel_index],
            case['pseudo_observed'][..., channel_index],
            duration, rank, lam_mag, lam_smooth, usable, drop_dark=drop_dark)
        basis = fit.V.reshape(duration, SLOTS, fit.V.shape[1])
        if loading_mode == 'corridor':
            source = 'corridor_mean(forced)'
            loading = corridor
            residual = basis @ loading
        out.append(residual)
        corridor_only.append(basis @ corridor)
        diagnostics[channel] = {
            'n_iter': fit.n_iter,
            'train_rmse': fit.train_rmse,
            'validation_rmse': fit.validation_rmse,
            'loading': [float(x) for x in loading],
            'loading_norm': float(np.linalg.norm(loading)),
            'loading_source': source,
            'donors_with_loadings': n_donor,
            'corridor_loading_norm': float(np.linalg.norm(corridor)),
            'residual_rms': float(np.sqrt(np.mean(residual ** 2))),
        }
    return np.stack(out, axis=-1), diagnostics, np.stack(corridor_only, axis=-1)


def case_metric(pred_norm, truth_norm, scoring):
    return profile_macro(pred_norm, truth_norm, scoring)


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------

def stage_field(engine, queue_name: str, limit: int = 0,
                donor_limit: int = 0, tag: str = '') -> dict:
    queue = engine.queues[queue_name]
    rows = queue['rows'][:limit] if limit else queue['rows']
    _, availability = load_availability()
    done, skipped = [], 0
    for row in rows:
        path = case_cache(queue_name, row['episode_id'], tag)
        if path.exists():
            skipped += 1
            continue
        started = time.perf_counter()
        payload = build_case(engine, queue_name, row, donor_limit=donor_limit,
                             availability=availability)
        save_case(payload, path)
        done.append({'episode_id': row['episode_id'],
                     'seconds': round(time.perf_counter() - started, 1),
                     'leak_hidden_cells': payload['leak_hidden_cells']})
        print(json.dumps({'phase': 'field', 'queue': queue_name,
                          'case': row['episode_id'],
                          'seconds': round(payload['case_seconds'], 1),
                          'pseudo_offset': payload['pseudo_offset_weeks'],
                          'donor_failures': payload['donor_failures'],
                          'cached': len(done), 'skipped': skipped}), flush=True)
    summary = {'queue': queue_name, 'tag': tag, 'n_cases': len(rows),
               'built': len(done), 'resumed': skipped,
               'control_checkpoint': str(engine.control),
               'control_sha256': engine.control_sha}
    (OUTBASE / (queue_name + (f'__{tag}' if tag else ''))
     / 'field_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding='utf-8')
    return summary


def load_queue_cases(queue_name: str, limit: int = 0, tag: str = '') -> list[dict]:
    directory = OUTBASE / (queue_name + (f'__{tag}' if tag else ''))
    paths = sorted(directory.glob('case_*.npz'))
    if limit:
        paths = paths[:limit]
    return [load_case(path) for path in paths]


def configs():
    for rank in RANKS:
        for lam_mag in LAM_MAGS:
            for lam_smooth in LAM_SMOOTHS:
                yield {'rank': rank, 'lam_mag': lam_mag,
                       'lam_smooth': lam_smooth}


def evaluate_config(cases, config, drop_dark=False) -> dict:
    """Aggregate the pre-registered objective on a cached queue."""
    q_ratio, o_ratio = [], []
    improved_q = improved_o = 0
    diagnostics = []
    for case in cases:
        scales = OUR_SCALES
        truth = case['truth_control'] / scales
        scoring = case['scoring_control'].astype(bool)
        control = case['pred_control']
        residual, diag, _ = corrected_case(case, config['rank'],
                                           config['lam_mag'],
                                           config['lam_smooth'],
                                           drop_dark=drop_dark)
        corrected = control + residual
        base = case_metric(control, truth, scoring)
        now = case_metric(corrected, truth, scoring)
        for channel in CHANNELS:
            if now[channel] < base[channel]:
                if channel == 'q':
                    improved_q += 1
                else:
                    improved_o += 1
        q_ratio.append(now['q'] / base['q'])
        o_ratio.append(now['o'] / base['o'])
        diagnostics.append({'episode_id': case['episode_id'],
                            'base': base, 'corrected': now})
    mean_q, mean_o = float(np.mean(q_ratio)), float(np.mean(o_ratio))
    return {'config': config,
            'ratio_q': mean_q, 'ratio_o': mean_o,
            'objective': 0.5 * (mean_q + mean_o),
            'both_channels_improved': bool(mean_q < 1 and mean_o < 1),
            'per_case_improved': {'q': improved_q, 'o': improved_o,
                                  'n': len(cases)},
            'per_case': diagnostics}


def stage_sweep(queue_name: str = TRAIN_QUEUE, tag: str = '') -> dict:
    cases = load_queue_cases(queue_name, tag=tag)
    if not cases:
        raise SystemExit(f'no cached cases for {queue_name}')
    table = []
    for config in configs():
        entry = evaluate_config(cases, config)
        table.append({k: v for k, v in entry.items() if k != 'per_case'})
        print(json.dumps({'phase': 'sweep', **entry['config'],
                          'ratio_q': round(entry['ratio_q'], 5),
                          'ratio_o': round(entry['ratio_o'], 5),
                          'objective': round(entry['objective'], 5),
                          'both': entry['both_channels_improved']}), flush=True)
    usable = [row for row in table if row['both_channels_improved']]
    chosen = min(usable, key=lambda row: row['objective']) if usable else None
    # Post-hoc sensitivity ONLY (a data-quality, not a hyper-parameter, question:
    # does the field suffer from donors that are dark for the whole window?).
    # It is never a selection candidate and the budget stays at 18.
    sensitivity = None
    if chosen:
        cases = load_queue_cases(queue_name, tag=tag)
        entry = evaluate_config(cases, chosen['config'], drop_dark=True)
        sensitivity = {k: v for k, v in entry.items() if k != 'per_case'}
        print(json.dumps({'phase': 'sweep_sensitivity_drop_dark',
                          **chosen['config'],
                          'ratio_q': round(entry['ratio_q'], 5),
                          'ratio_o': round(entry['ratio_o'], 5),
                          'both': entry['both_channels_improved']}), flush=True)
    report = {'queue': queue_name, 'n_cases': len(cases), 'table': table,
              'n_configs': len(table), 'n_passing_both': len(usable),
              'chosen': chosen,
              'post_hoc_sensitivity_drop_dark_donors': sensitivity,
              'note': ('objective = 0.5*(ratio_q + ratio_o); a configuration is '
                       'eligible only if BOTH ratios < 1 pre/post. The drop-dark '
                       'row is a post-hoc sensitivity, not a candidate.')}
    (OUTBASE / (f'config_sweep__{tag}.json' if tag
                else 'config_sweep.json')).write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding='utf-8')
    return report


def stage_score(chosen: dict | None, tag: str = '') -> dict:
    """Selection-42 three-arm comparison in one scorer.

    ``chosen`` may be either a bare configuration or the sweep's ranked row that
    wraps it under ``config``; both are normalised here so a caller cannot pass
    the wrong level by accident.
    """
    if chosen is not None and 'config' in chosen:
        chosen = chosen['config']
    cases = load_queue_cases(SELECT_QUEUE, tag=tag)
    if not cases:
        raise SystemExit(f'no cached cases for {SELECT_QUEUE}')

    control_cases, corrected_cases, corridor_cases = [], [], []
    drop_dark_cases = []
    extras = []
    for case in cases:
        scales = OUR_SCALES
        truth = case['truth_control'] / scales
        scoring = case['scoring_control'].astype(bool)
        control = case['pred_control']
        common = {'episode_id': str(case['episode_id']),
                  'target_station': str(case['target_station']),
                  'duration_days': int(case['duration_days'])}
        # `evaluate` always divides `truth` by the scales, so it stays raw; our
        # model output is already normalised, hence prediction_is_raw=False.
        control_cases.append({**common, 'prediction': control,
                              'truth': case['truth_control'],
                              'scoring': scoring})
        row = {'control_rmse': case_metric(control, truth, scoring)}
        if chosen:
            residual, diag, corridor_residual = corrected_case(
                case, chosen['rank'], chosen['lam_mag'], chosen['lam_smooth'])
            corrected = control + residual
            corrected_cases.append({**common, 'prediction': corrected,
                                    'truth': case['truth_control'],
                                    'scoring': scoring})
            # Zero-extra-cost diagnostic (no GPU: the caches already hold the
            # donor field). It is reported as a sensitivity row, never as a
            # selection candidate.
            corridor_cases.append({**common,
                                   'prediction': control + corridor_residual,
                                   'truth': case['truth_control'],
                                   'scoring': scoring})
            dark_residual, _, _ = corrected_case(
                case, chosen['rank'], chosen['lam_mag'], chosen['lam_smooth'],
                drop_dark=True)
            drop_dark_cases.append({**common,
                                    'prediction': control + dark_residual,
                                    'truth': case['truth_control'],
                                    'scoring': scoring})
            row['corrected_rmse'] = case_metric(corrected, truth, scoring)
            row['corridor_rmse'] = case_metric(control + corridor_residual,
                                               truth, scoring)
            active = scoring
            row['control_mae'] = {
                channel: float(np.abs(
                    np.where(active[..., i], control[..., i] - truth[..., i], 0)
                ).sum() / active[..., i].sum()) for i, channel in
                enumerate(CHANNELS)}
            row['corrected_mae'] = {
                channel: float(np.abs(
                    np.where(active[..., i], corrected[..., i] - truth[..., i], 0)
                ).sum() / active[..., i].sum()) for i, channel in
                enumerate(CHANNELS)}
            row['diagnostics'] = diag
            row['residual_rms'] = {
                channel: float(np.sqrt(np.mean(residual[..., i] ** 2)))
                for i, channel in enumerate(CHANNELS)}
        row['case_seconds'] = float(case['case_seconds'])
        row['pseudo_usable'] = bool(case.get('pseudo_usable', True))
        extras.append(row)

    report = {'queue': SELECT_QUEUE, 'config': chosen, 'tag': tag,
              'scales': list(OUR_SCALES),
              'loading_source_census': {
                  'pseudo_outage': int(sum(1 for e in extras
                                           if 'diagnostics' in e and
                                           all(e['diagnostics'][c]['loading_source']
                                               == 'pseudo_outage'
                                               for c in CHANNELS))),
                  'n_cases': len(extras)},
              'arms': {}, 'per_case': extras}
    report['arms']['g80'] = evaluate(control_cases, OUR_SCALES,
                                     prediction_is_raw=False)
    if chosen:
        report['arms']['g80_plus_lowrank'] = evaluate(
            corrected_cases, OUR_SCALES, prediction_is_raw=False)
        report['arms']['diagnostic_corridor_loading_only'] = evaluate(
            corridor_cases, OUR_SCALES, prediction_is_raw=False)
        report['arms']['diagnostic_drop_dark_donors'] = evaluate(
            drop_dark_cases, OUR_SCALES, prediction_is_raw=False)
    rttc = archived_rttc_predictions()
    report['arms']['rttc_v7_archived'] = evaluate(rttc, OUR_SCALES,
                                                  prediction_is_raw=True)
    report['arms']['rttc_v7_archived_in_its_own_scales'] = evaluate(
        rttc, np.asarray([6264.0, 0.212854996]), prediction_is_raw=True)
    for arm, entry in report['arms'].items():
        entry.pop('per_case', None)

    report['timing'] = {
        'cases': len(cases),
        'field_seconds_total': float(sum(c['case_seconds'] for c in cases)),
        'field_seconds_per_case_mean': float(np.mean(
            [c['case_seconds'] for c in cases])),
        'note': ('field time is the one-off cache build (control + donors + '
                 'pseudo); the 18-config search itself runs on the cache, so '
                 'the config search adds no GPU time'),
    }
    (OUTBASE / (f'final_report__{tag}.json' if tag
                else 'final_report.json')).write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding='utf-8')
    return report


def print_report(report: dict) -> None:
    print('\n=== Selection-42, one scorer, our normalisation '
          f'[{OUR_SCALES[0]}, {OUR_SCALES[1]}] ===')
    print(f'{"arm":<26}{"q":>10}{"o":>10}')
    for arm, entry in report['arms'].items():
        if 'episode_macro' not in entry:
            e = entry
            print(f'{arm:<26}{e["q"]:>10.6f}{e["o"]:>10.6f}')
            continue
        e = entry['episode_macro']
        print(f'{arm:<26}{e["q"]:>10.6f}{e["o"]:>10.6f}')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', default='all',
                        choices=('gates', 'field', 'sweep', 'score', 'all'))
    parser.add_argument('--queue', default='')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--donor-limit', type=int, default=0)
    parser.add_argument('--config', default='')
    parser.add_argument('--control', default='',
                        help='override the frozen G80 checkpoint (cross-seed '
                             'replication); the SHA is recorded in the log')
    parser.add_argument('--expect-sha', default='',
                        help='assert the control checkpoint has this SHA')
    parser.add_argument('--tag', default='',
                        help='suffix for the output/cache directory')
    args = parser.parse_args()
    if args.expect_sha and args.control:
        actual = sha(Path(args.control))
        if actual != args.expect_sha:
            raise SystemExit(f'control SHA mismatch: {actual} != {args.expect_sha}')

    OUTBASE.mkdir(parents=True, exist_ok=True)
    record = {'phase': 'start', 'stage': args.stage, 'tag': args.tag,
              'control_override': args.control or None,
              'control_sha256': (sha(Path(args.control)) if args.control
                                 else CONTROL_SHA),
              'selection_manifest_sha256': sha(SELECTION_MANIFEST),
              'availability_sha256': (sha(AVAILABILITY)
                                      if AVAILABILITY.exists() else None),
              'python': sys.executable}
    log = OUTBASE / 'runner_log.jsonl'
    with log.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + '\n')

    if args.stage == 'gates':
        engine = Engine(args.control or CONTROL)
        result = {**gate_g1_g3(engine), 'G0b_variant_equals_frozen': gate_g0b(engine),
                  'G5_scorer_consistency': gate_g5()}
        print(json.dumps(result, ensure_ascii=False, indent=1))
        (OUTBASE / 'gates.json').write_text(
            json.dumps(result, ensure_ascii=False, indent=1), encoding='utf-8')
        return 0

    if args.stage in ('field', 'all'):
        engine = Engine(args.control or CONTROL)
        gates = {**gate_g1_g3(engine),
                 'G0b_variant_equals_frozen': gate_g0b(engine),
                 'G5_scorer_consistency': gate_g5()}
        (OUTBASE / 'gates.json').write_text(
            json.dumps(gates, ensure_ascii=False, indent=1), encoding='utf-8')
        failed = [name for name, value in gates.items()
                  if isinstance(value, dict) and not value.get('passed')]
        if failed:
            raise SystemExit(
                f'completeness gates failed: {failed}; the batch is void and no '
                f'conclusion may be drawn (see TRANSDUCTIVE_CORRECTION_'
                f'PREREGISTRATION.md section 6)')
        stage_field(engine, args.queue or TRAIN_QUEUE, args.limit,
                    args.donor_limit, args.tag)
        if args.stage == 'all':
            sweep = stage_sweep(TRAIN_QUEUE, args.tag)
            stage_field(engine, SELECT_QUEUE, args.limit, args.donor_limit,
                        args.tag)
            report = stage_score(sweep['chosen'], args.tag)
            print_report(report)
        return 0

    if args.stage == 'field':
        engine = Engine(args.control or CONTROL)
        print(json.dumps(stage_field(engine, args.queue or TRAIN_QUEUE,
                                     args.limit, args.donor_limit, args.tag),
                         ensure_ascii=False))
        return 0

    if args.stage == 'sweep':
        print(json.dumps({k: v for k, v in stage_sweep(
            args.queue or TRAIN_QUEUE, args.tag).items() if k != 'table'},
            ensure_ascii=False, indent=1))
        return 0

    sweep_path = OUTBASE / (f'config_sweep__{args.tag}.json' if args.tag
                            else 'config_sweep.json')
    chosen = None
    if args.config:
        rank, lam_mag, lam_smooth = (float(x) for x in args.config.split(','))
        chosen = {'rank': int(rank), 'lam_mag': lam_mag, 'lam_smooth': lam_smooth}
    elif sweep_path.exists():
        chosen = json.loads(sweep_path.read_text(encoding='utf-8'))['chosen']
    report = stage_score(chosen, args.tag)
    print_report(report)
    with log.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps({'phase': 'done', 'stage': 'score',
                                 'config': chosen}, ensure_ascii=False) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
