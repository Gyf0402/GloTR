"""Test-21 re-evaluation of the frozen transductive correction (3 G80 seeds).

Authorised by the user on 2026-09-27 as a RE-EVALUATION on an already-exposed
test set: the configuration and the code are the frozen ones
(``rank=4, lam_mag=0.1, lam_smooth=0.1`` from ``config_sweep.json``), and nothing
here may be used to choose a configuration. It answers the question the
Selection run could not: whether the cases on which G80 lost to RTTC in December
are improved or overturned.

Nothing in this file modifies a SHA-pinned artefact. It imports the frozen
machinery (``run_transductive`` for the field/gate/fit path, ``score_selection_
three_arm`` for the one scorer) and adds only Test-specific plumbing:

* a Test-band reader with the registered day whitelist lifted -- needed because the
  registered whitelist starts 2019-11-27 while a pseudo window may start earlier.
  Every day actually accessed is logged and diffed against the whitelist;
* the frozen Test assembler for the control arm, so the anchor check can be exact;
* RTTC v7 predictions read from the archived Test run.

Outputs land in ``reviews/2026-09-27/test21_reeval``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_transductive as rt  # noqa: E402
import scripts.train_g80_v2_arm as arm_mod  # noqa: E402
from scripts.run_g80_test import TestRoleReader, assemble_test_episode  # noqa: E402
from scripts.score_selection_three_arm import (  # noqa: E402
    evaluate, profile_macro)
from scripts.scan_test21_availability import (  # noqa: E402
    relaxed_test_reader)

OUT = ROOT / 'reviews/2026-09-27/test21_reeval'
QUEUE = 'test_21'
TEST_ROOT = ROOT / 'reviews/2026-09-25/g80_test_v1'
SEEDS = {20260905: 'a5fac4ac159ccc08be146ae41e41c8de771e03e8fc8d2eeec85b78b9a3f646d5',
         20260906: 'fb3278e586cf5a27186549d88712af84d53d63248c78e7dff517b05d7fa73cab',
         20260907: '05b4d7b9f9b4ea2c442143214f6ed1452dc79860be94615b296ad263b2917982'}
CONTROL_OF = {seed: ROOT / f'reviews/2026-09-26/g80_v2_round2/stability/'
                          f'seed{seed}/full.pt' for seed in SEEDS}


def frozen_config() -> dict:
    sweep = json.loads((ROOT / 'reviews/2026-09-26/transductive/'
                        'config_sweep.json').read_text(encoding='utf-8'))
    chosen = sweep['chosen']
    return chosen['config'] if 'config' in chosen else chosen


def test_rows() -> list[dict]:
    """Registered Test rows: the manifest lists ids, the roster holds the fields.

    ``build_queues`` maps exactly this way, so doing the same keeps this
    re-evaluation on the same rows the original Test run used (including the
    ``split``/``protocol`` fields the Test assembler validates).
    """
    roster = {row['episode_id']: row for row in
              arm_mod.load_json(ROOT / arm_mod.ROSTER_REL)['episodes']}
    manifest = arm_mod.load_json(TEST_ROOT / 'test_manifest.json')
    return [roster[entry['episode_id']] for entry in manifest['candidates']]


def archived(name: str) -> dict[str, dict]:
    out = {}
    for path in sorted((TEST_ROOT / 'predictions' / name).glob('*.npz')):
        with np.load(path, allow_pickle=False) as archive:
            record = {key: archive[key] for key in archive.files}
        out[str(record['episode_id'])] = record
    return out


def install_relaxed_test_queue(engine) -> dict:
    """Swap in a Test queue that reads the Test band without the whitelist."""
    reader = relaxed_test_reader(engine.order)
    engine.queues[QUEUE] = {
        'rows': test_rows(),
        'reader': reader,
        'assemble': lambda row, read: assemble_test_episode(
            row, engine.order, reader.bounds, read),
    }
    return {'role_band': list(reader.bounds)}


def gate_variant_against_test_assembler(engine) -> dict:
    """The runner's variant assembler must reproduce the Test assembler bitwise."""
    frozen_reader = TestRoleReader(
        ROOT, ROOT / arm_mod.RELEASE_REL, engine.order,
        set(arm_mod.load_json(TEST_ROOT / 'test_manifest.json')['allowed_days']))
    read = rt.lru_cache(maxsize=64)(lambda day: frozen_reader(day))
    row = test_rows()[0]
    frozen = assemble_test_episode(row, engine.order, frozen_reader.bounds, read)
    variant, _ = rt.assemble_variant(
        engine.order, read, target_station=str(row['station_id']),
        start_day=row['start_day'], duration=row['duration_days'],
        support='legal')
    checks = {
        'target_index_equal': variant.target_index == frozen.target_index,
        'values_bitwise_equal': bool(np.array_equal(variant.values, frozen.values)),
        'visible_bitwise_equal': bool(np.array_equal(variant.visible,
                                                     frozen.visible)),
        'reference_bitwise_equal': bool(np.array_equal(variant.reference,
                                                       frozen.reference)),
        'scoring_bitwise_equal': bool(np.array_equal(variant.scoring,
                                                     frozen.scoring)),
        'dates_equal': variant.dates == frozen.dates,
    }
    checks['passed'] = all(value for key, value in checks.items()
                           if key.endswith('equal'))
    return checks


def build_seed(seed: int, limit: int = 0, rebuild: bool = False) -> dict:
    engine = rt.Engine(CONTROL_OF[seed])
    if engine.control_sha != SEEDS[seed]:
        raise SystemExit(f'seed {seed} checkpoint SHA {engine.control_sha} '
                         f'!= registered {SEEDS[seed]}')
    band = install_relaxed_test_queue(engine)
    gate = gate_variant_against_test_assembler(engine)
    if not gate['passed']:
        raise SystemExit(f'variant assembler differs from the Test assembler: {gate}')

    availability = {}
    for row in json.loads((OUT / 'pseudo_availability_test21.json')
                          .read_text(encoding='utf-8'))['rows']:
        availability[(QUEUE, row['episode_id'])] = row

    archive = archived('g80')
    directory = OUT / f'cases_seed{seed}'
    directory.mkdir(parents=True, exist_ok=True)
    anchors, built = [], []
    for row in (test_rows()[:limit] if limit else test_rows()):
        path = directory / f"case_{row['episode_id'].replace('|', '_')}.npz"
        if rebuild or not path.exists():
            payload = rt.build_case(engine, QUEUE, row, availability=availability)
            rt.save_case(payload, path)
            built.append(row['episode_id'])
        else:
            payload = rt.load_case(path)
        reference = archive[row['episode_id']]
        # The archive stores RAW physical units and was produced as
        # ``normalised * scales``; multiplying our own normalised output by the
        # archive's OWN scales vector reproduces it bit for bit (the literal
        # 0.2127663 in this repo is not the exact same float). This is a genuine
        # bitwise anchor: the control arm of this re-evaluation IS the published
        # G80 Test prediction.
        anchor = {
            'episode_id': row['episode_id'],
            'control_matches_archive_bitwise': bool(np.array_equal(
                payload['pred_control'] * reference['scales'],
                reference['pred_g80_phys'])),
            'truth_matches_archive_bitwise': bool(np.array_equal(
                payload['truth_control'], reference['truth_phys'])),
            'scoring_matches_archive_bitwise': bool(np.array_equal(
                payload['scoring_control'], reference['scoring'].astype(bool))),
            'max_abs_pred_diff': float(np.abs(
                payload['pred_control'] * reference['scales']
                - reference['pred_g80_phys']).max()),
        }
        anchors.append(anchor)
    anchor_required = seed == 20260905
    truth_ok = all(a['truth_matches_archive_bitwise']
                   and a['scoring_matches_archive_bitwise'] for a in anchors)
    if not truth_ok:
        raise SystemExit(f'seed {seed}: truth/scoring do not reproduce the archive')
    if anchor_required and not all(a['control_matches_archive_bitwise']
                                   for a in anchors):
        raise SystemExit(f'anchor seed {seed}: control does not reproduce the '
                         'archived Test prediction bitwise')
    return {'seed': seed, 'control_sha256': engine.control_sha,
            'role_band': band, 'variant_gate': gate,
            'anchor_checks': anchors,
            'anchors_all_bitwise': all(a['control_matches_archive_bitwise']
                                       for a in anchors),
            'anchor_applicable': anchor_required,
            'anchor_why': (None if anchor_required else
                           'no archived prediction exists for this seed; only '
                           'truth/scoring are anchorable'),
            'truth_and_scoring_anchorable': truth_ok,
            'built': len(built),
            'donor_failures_total': int(sum(c['donor_failures'] for c in
                                            [rt.load_case(p) for p in
                                             sorted(directory.glob('case_*.npz'))])),
            'max_leak_cells': float(max([rt.load_case(p)['leak_hidden_cells']
                                         for p in sorted(
                                             directory.glob('case_*.npz'))])),
            'cases_pseudo_usable': int(sum(
                1 for p in sorted(directory.glob('case_*.npz'))
                if bool(rt.load_case(p)['pseudo_usable'])))}


def load_cases(seed: int) -> list[dict]:
    directory = OUT / f'cases_seed{seed}'
    return [rt.load_case(path) for path in sorted(directory.glob('case_*.npz'))]


def score_seed(seed: int, config: dict) -> dict:
    cases = load_cases(seed)
    if not cases:
        raise SystemExit(f'no cached cases for seed {seed}')
    rttc = archived('rttc_v7')
    scales = rt.OUR_SCALES

    arms = {name: [] for name in ('g80', 'g80_plus_lowrank', 'diagnostic_corridor',
                                  'diagnostic_drop_dark', 'rttc_v7')}
    per_case = []
    for case in cases:
        identity = str(case['episode_id'])
        reference = rttc[identity]
        truth_raw = reference['truth_phys']
        scoring = reference['scoring'].astype(bool)
        truth_norm = truth_raw / scales
        control = case['pred_control']
        residual, diag, corridor = rt.corrected_case(
            case, config['rank'], config['lam_mag'], config['lam_smooth'])
        dark, _, _ = rt.corrected_case(case, config['rank'], config['lam_mag'],
                                       config['lam_smooth'], drop_dark=True)
        common = {'episode_id': identity,
                  'target_station': str(case['target_station']),
                  'duration_days': int(case['duration_days'])}
        arms['g80'].append({**common, 'prediction': control,
                            'truth': truth_raw, 'scoring': scoring})
        arms['g80_plus_lowrank'].append({**common, 'prediction': control + residual,
                                         'truth': truth_raw, 'scoring': scoring})
        arms['diagnostic_corridor'].append({**common,
                                            'prediction': control + corridor,
                                            'truth': truth_raw, 'scoring': scoring})
        arms['diagnostic_drop_dark'].append({**common,
                                             'prediction': control + dark,
                                             'truth': truth_raw, 'scoring': scoring})
        arms['rttc_v7'].append({**common, 'prediction': reference['pred_rttc_phys'],
                                'truth': truth_raw, 'scoring': scoring})
        per_case.append({
            **common,
            'g80': profile_macro(control, truth_norm, scoring),
            'corrected': profile_macro(control + residual, truth_norm, scoring),
            'corridor': profile_macro(control + corridor, truth_norm, scoring),
            'drop_dark': profile_macro(control + dark, truth_norm, scoring),
            'rttc': profile_macro(reference['pred_rttc_phys'] / scales,
                                  truth_norm, scoring),
            'loading_source': {channel: diag[channel]['loading_source']
                               for channel in rt.CHANNELS},
            'residual_rms': {channel: float(np.sqrt(np.mean(
                residual[..., index] ** 2)))
                for index, channel in enumerate(rt.CHANNELS)},
            'pseudo_usable': bool(case.get('pseudo_usable', True)),
            'donor_failures': int(case['donor_failures']),
            'field_seconds': float(case['case_seconds']),
        })

    report = {'seed': seed, 'config': config, 'n_cases': len(cases),
              'arms': {}, 'per_case': per_case}
    # Unit bookkeeping matters here: the archived predictions are RAW physical
    # units while our own forward output is already normalised by `prepare_item`.
    # Mixing the two flags would silently rescale one arm.
    raw_flag = {'g80': False, 'g80_plus_lowrank': False,
                'diagnostic_corridor': False, 'diagnostic_drop_dark': False,
                'rttc_v7': True}
    for name, records in arms.items():
        entry = evaluate(records, scales, prediction_is_raw=raw_flag[name])
        entry.pop('per_case', None)
        report['arms'][name] = entry
    return report


def by_station_and_duration(per_case: list[dict]) -> dict:
    stations = {}
    for case in per_case:
        stations.setdefault(case['target_station'], []).append(case)
    table = []
    for station, group in sorted(stations.items()):
        row = {'station': station, 'n': len(group)}
        for channel in rt.CHANNELS:
            row[channel] = {
                'g80': float(np.mean([c['g80'][channel] for c in group])),
                'corrected': float(np.mean([c['corrected'][channel]
                                            for c in group])),
                'rttc': float(np.mean([c['rttc'][channel] for c in group])),
            }
        table.append(row)
    durations = {}
    for case in per_case:
        durations.setdefault(str(case['duration_days']), []).append(case)
    by_duration = {key: {'n': len(group),
                         **{channel: {
                             'g80': float(np.mean([c['g80'][channel]
                                                   for c in group])),
                             'corrected': float(np.mean([c['corrected'][channel]
                                                         for c in group])),
                             'rttc': float(np.mean([c['rttc'][channel]
                                                    for c in group]))}
                             for channel in rt.CHANNELS}}
                   for key, group in sorted(durations.items())}
    return {'by_station': table, 'by_duration': by_duration}


def lost_to_rttc(per_case: list[dict]) -> dict:
    """The headline question: cases where G80 lost, and what happened to them."""
    out = {}
    for channel in rt.CHANNELS:
        lost = [c for c in per_case if c['g80'][channel] > c['rttc'][channel]]
        gained = [c for c in per_case if c['g80'][channel] <= c['rttc'][channel]]
        recovered = [c for c in lost if c['corrected'][channel]
                     <= c['rttc'][channel]]
        still = [c for c in lost if c['corrected'][channel] > c['rttc'][channel]]
        newly_lost = [c for c in gained if c['corrected'][channel]
                      > c['rttc'][channel]]
        out[channel] = {
            'n_lost_under_g80': len(lost),
            'of_those_overtaken': len(recovered),
            'of_those_still_behind': len(still),
            'n_not_lost_under_g80': len(gained),
            'of_those_newly_behind': len(newly_lost),
            'better_than_g80_among_lost': sum(
                1 for c in lost if c['corrected'][channel]
                < c['g80'][channel]),
            'lost_ids': [c['episode_id'] for c in lost],
            'recovered_ids': [c['episode_id'] for c in recovered],
            'still_behind_ids': [c['episode_id'] for c in still],
            'newly_behind_ids': [c['episode_id'] for c in newly_lost],
            'mean_gap_g80_minus_rttc_among_lost': float(np.mean(
                [c['g80'][channel] - c['rttc'][channel] for c in lost]))
            if lost else None,
            'mean_gap_corrected_minus_rttc_among_lost': float(np.mean(
                [c['corrected'][channel] - c['rttc'][channel] for c in lost]))
            if lost else None,
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', default='all',
                        choices=('build', 'score', 'all'))
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--rebuild', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    config = frozen_config()
    seeds = [args.seed] if args.seed else list(SEEDS)
    if args.stage in ('build', 'all'):
        integrity = []
        for seed in seeds:
            result = build_seed(seed, args.limit, args.rebuild)
            integrity.append(result)
            print(json.dumps({k: v for k, v in result.items()
                              if k != 'anchor_checks'}, ensure_ascii=False),
                  flush=True)
        (OUT / 'integrity.json').write_text(
            json.dumps({'config_frozen': config, 'seeds': integrity},
                       ensure_ascii=False, indent=1), encoding='utf-8')
        if args.stage == 'build':
            return 0

    reports = []
    for seed in seeds:
        report = score_seed(seed, config)
        report.update(by_station_and_duration(report['per_case']))
        report['lost_to_rttc'] = lost_to_rttc(report['per_case'])
        reports.append(report)
        print(json.dumps({'seed': seed, **{
            name: entry['episode_macro'] for name, entry in
            report['arms'].items()}}, ensure_ascii=False), flush=True)
    (OUT / 'test21_reeval.json').write_text(
        json.dumps({'config_frozen': config, 'reports': reports},
                   ensure_ascii=False, indent=1), encoding='utf-8')

    print('\n=== Test-21, one scorer, physical truth from the archive ===')
    print(f'{"seed":<10}{"arm":<32}{"q":>10}{"o":>10}{"q_sen":>10}{"o_sen":>10}')
    for report in reports:
        for name, entry in report['arms'].items():
            print(f"{report['seed']:<10}{name:<32}"
                  f"{entry['episode_macro']['q']:>10.6f}"
                  f"{entry['episode_macro']['o']:>10.6f}"
                  f"{entry['sensor_macro']['q']:>10.6f}"
                  f"{entry['sensor_macro']['o']:>10.6f}")
        print('-' * 72)
    print('written', OUT / 'test21_reeval.json')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
