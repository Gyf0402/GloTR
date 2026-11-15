"""Availability scan for the Test-21 transductive re-evaluation (read-only).

Same placement rule as the frozen one for Train/Selection (README in
``reviews/2026-09-26/TRANSDUCTIVE_CORRECTION_AMENDMENT.md`` section 2.2'):

    candidate k = 0..6 weeks; window = [start-13-7k, start-1-7k]; admissible iff
    (a) every day lies inside the episode's role band, (b) no day is globally dark,
    (c) the target's legal density over the pseudo outage days is >= 0.5.
    The smallest admissible k wins.

Two Test-specific facts are recorded here instead of being assumed:

* The registered ``allowed_days`` whitelist only covers 2019-11-27..12-28, while
  the pseudo window for the earliest episode starts 2019-11-17. The extra days are
  inside the Test role band and far from the sealed Calibration band (which ends
  2019-11-06), so they are readable as model INPUTS; the exact day list and counts
  are written out for disclosure.
* Every candidate is required to stay inside the Test band, so the scan never
  reaches Calibration even though the reader would allow earlier bands.
"""

from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.train_g80_v2_arm as arm_mod  # noqa: E402
from scripts.run_g80_test import TestRoleReader  # noqa: E402

MAX_OFFSET_WEEKS = 6
DURATION = 7
BUFFER = 3
TARGET_DENSITY_MIN = 0.5
OUT = ROOT / 'reviews/2026-09-27/test21_reeval'


def relaxed_test_reader(order):
    """Test-band reader with the whitelist lifted, bounds still enforced.

    The whitelist is a protocol artefact of the original evaluation; the
    transductive method needs days that predate it. Reading remains fixed to the
    Test role band, so the sealed Calibration band is unreachable, and every day
    accessed is logged for disclosure.
    """
    release = json.loads((ROOT / arm_mod.RELEASE_REL).read_text(encoding='utf-8'))
    manifest = json.loads(
        (ROOT / release['array_manifest']['path']).read_text(encoding='utf-8'))
    every_day = {entry['day'] for entry in manifest['days']}
    return TestRoleReader(ROOT, ROOT / arm_mod.RELEASE_REL, order, every_day)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    order = tuple(str(x) for x in
                  arm_mod.load_json(ROOT / arm_mod.GRAPH_REL)['station_order'])
    manifest = arm_mod.load_json(
        ROOT / 'reviews/2026-09-25/g80_test_v1/test_manifest.json')
    rows = manifest['candidates']
    whitelist = set(manifest['allowed_days'])
    reader = relaxed_test_reader(order)
    bands = reader.bounds

    plan, wanted = {}, set()
    for row in rows:
        start = date.fromisoformat(row['start_day'])
        candidates = []
        for offset in range(MAX_OFFSET_WEEKS + 1):
            window_start = start - timedelta(days=2 * BUFFER + DURATION
                                            + 7 * offset)
            days = [(window_start + timedelta(days=i)).isoformat()
                    for i in range(DURATION + 2 * BUFFER)]
            gap = days[BUFFER:BUFFER + DURATION]
            candidates.append({'offset': offset, 'window': days, 'gap': gap})
            wanted.update(days)
        plan[row['episode_id']] = candidates

    print(f'distinct days to read: {len(wanted)}', flush=True)
    presence, legality, accessed = {}, {}, set()
    for index, day in enumerate(sorted(wanted), 1):
        if not (bands[0] <= day <= bands[1]):
            presence[day] = -1                      # out of band, never read
            continue
        record = {}
        for label, station, values, mask in reader(day):
            record[station] = int(mask.sum())
        accessed.add(day)
        presence[day] = len(record)
        legality[day] = record
        if index % 20 == 0:
            print(f'  read {index}/{len(wanted)}', flush=True)

    out_of_band = sorted(day for day, count in presence.items() if count == -1)
    dark_days = sorted(day for day, count in presence.items() if count == 0)
    extra_days = sorted(accessed - whitelist)

    evaluated = []
    for row in rows:
        target = str(row['station_id'])
        candidates = []
        for candidate in plan[row['episode_id']]:
            days, gap = candidate['window'], candidate['gap']
            in_band = all(bands[0] <= day <= bands[1] for day in days)
            viable = in_band and all(presence.get(day, 0) > 0 for day in days)
            density = (sum(legality.get(day, {}).get(target, 0)
                           for day in gap) / (DURATION * 288 * 2))
            candidates.append({'offset': candidate['offset'],
                               'in_band': in_band, 'days_all_viable': viable,
                               'target_density': round(float(density), 4),
                               'window_start': days[0], 'window_end': days[-1],
                               'days_outside_whitelist': sum(1 for day in days
                                                             if day not in whitelist)})
        admissible = [c for c in candidates
                      if c['days_all_viable']
                      and c['target_density'] >= TARGET_DENSITY_MIN]
        chosen = min(admissible, key=lambda c: c['offset']) if admissible else None
        evaluated.append({'episode_id': row['episode_id'], 'target': target,
                          'duration_days': row['duration_days'],
                          'start_day': row['start_day'],
                          'candidates': candidates,
                          'chosen_offset': None if chosen is None else chosen['offset'],
                          'chosen_window': None if chosen is None else
                          [chosen['window_start'], chosen['window_end']],
                          'chosen_days_outside_whitelist':
                              None if chosen is None
                              else chosen['days_outside_whitelist']})

    histogram: dict[str, int] = {}
    for row in evaluated:
        key = str(row['chosen_offset'])
        histogram[key] = histogram.get(key, 0) + 1

    report = {
        'queue': 'test_21', 'duration': DURATION, 'buffer': BUFFER,
        'target_density_min': TARGET_DENSITY_MIN,
        'max_offset_weeks': MAX_OFFSET_WEEKS,
        'role_band': list(bands),
        'registered_whitelist_span': [min(whitelist), max(whitelist)],
        'n_registered_whitelist_days': len(whitelist),
        'n_days_read_total': len(accessed),
        'days_outside_whitelist': extra_days,
        'n_days_outside_whitelist': len(extra_days),
        'days_out_of_band_never_read': out_of_band,
        'globally_dark_days_in_band': dark_days,
        'n_globally_dark_days_in_band': len(dark_days),
        'chosen_offset_histogram': histogram,
        'unavailable_cases': sum(1 for r in evaluated
                                 if r['chosen_offset'] is None),
        'cases_using_whitelist_external_days': sum(
            1 for r in evaluated if r['chosen_days_outside_whitelist']),
        'total_whitelist_external_day_uses': sum(
            r['chosen_days_outside_whitelist'] or 0 for r in evaluated),
        'rows': evaluated,
    }
    (OUT / 'pseudo_availability_test21.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'rows'},
                     ensure_ascii=False, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
