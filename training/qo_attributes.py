"""Date/station keyed candidate attributes using the existing five pilot encodings.

No learned encoder, latest-version substitution, missing-value filling, or claim
of historical effectiveness. Reuses project metadata contracts; no dependency added.
"""

import csv
import gzip
import hashlib
import io
import json
from collections import defaultdict
from collections.abc import Iterator, Sequence
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from training.real_qo_loader import Profile, iter_role

FEATURES = ('abs_pm_div100', 'latitude_div90', 'longitude_div180', 'lanes_div8', 'is_ml')


class AnnualAttributes:
    """Reject ambiguous or unavailable metadata; caller cannot silently drop profiles."""

    def __init__(self, records: list[dict[str, Any]], rows: list[dict[str, str]],
                 universe: dict[str, bool]) -> None:
        self.universe = dict(universe)
        if any(type(v) is not bool for v in universe.values()):
            raise ValueError('U/K membership must be boolean')
        self.records: dict[tuple[str, str, str], dict[str, Any]] = {}
        for record in records:
            key = (str(record['station_id']), record['catalog'], record['configuration_id'])
            if key in self.records:
                raise ValueError('duplicate metadata key')
            self.records[key] = record
        self.rows: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            if date.fromisoformat(row['start']) >= date.fromisoformat(row['end_exclusive']):
                raise ValueError('invalid version interval')
            self.rows[str(row['station_id'])].append(dict(row))
        for intervals in self.rows.values():
            intervals.sort(key=lambda r: r['start'])
            if any(a['end_exclusive'] > b['start'] for a, b in zip(intervals, intervals[1:])):
                raise ValueError('overlapping version intervals')
        self.bindings: dict[str, str] = {}
        self._cache: dict[tuple[str, str], dict[str, Any]] = {}

    @classmethod
    def from_files(cls, root: Path, *, manifest_sha256: str) -> 'AnnualAttributes':
        marker = (root / 'complete.json').read_bytes()
        if hashlib.sha256(marker).hexdigest() != manifest_sha256:
            raise ValueError('partition manifest SHA mismatch')
        outputs = {x['file']: x['sha256'] for x in json.loads(marker)['outputs']}
        payloads = {}
        for name in ('station_versions.json', 'station_version_mapping.csv.gz', 'permanent_u.json'):
            data = (root / name).read_bytes()
            if hashlib.sha256(data).hexdigest() != outputs[name]:
                raise ValueError(f'attribute source SHA mismatch: {name}')
            payloads[name] = data
        records = json.loads(payloads['station_versions.json'])['records']
        rows = list(csv.DictReader(io.StringIO(gzip.decompress(
            payloads['station_version_mapping.csv.gz']).decode('utf-8'))))
        entries = json.loads(payloads['permanent_u.json'])['stations']
        universe = {str(x['station_id']): x['permanent_U'] for x in entries}
        if len(universe) != len(entries):
            raise ValueError('duplicate U/K membership')
        instance = cls(records, rows, universe)
        instance.bindings = {name: outputs[name] for name in payloads}
        instance.bindings['complete.json'] = manifest_sha256
        return instance

    def lookup(self, day: str, station: str) -> dict[str, Any]:
        date.fromisoformat(day)
        if station not in self.universe or self.universe[station]:
            raise ValueError(f'unknown or permanent U station: {station}')
        matches = [r for r in self.rows.get(station, []) if r['start'] <= day < r['end_exclusive']]
        if len(matches) != 1:
            raise ValueError(f'missing unique version for {day}/{station}')
        row = matches[0]
        if row['disposition'] != 'candidate_consensus_only':
            raise ValueError(f'excluded attributes for {day}/{station}: {row["reason"]}')
        key = (station, row['start'])
        if key in self._cache:
            return dict(self._cache[key])
        catalogs = row['candidate_catalogs'].split(';')
        configs = row['configuration_ids'].split(';')
        if len(catalogs) != len(configs) or len(set(configs)) != 1 or 'missing' in configs:
            raise ValueError('configuration consensus is not unique')
        records = []
        for catalog, config in zip(catalogs, configs):
            record = self.records.get((station, catalog, config))
            if record is None:
                raise ValueError('consensus record missing from source catalog')
            records.append(record)
        fields = records[0]['fields']
        if any(r['fields'] != fields for r in records):
            raise ValueError('catalog fields contradict configuration consensus')
        if any(not str(fields.get(k, '')).strip() for k in
               ('Fwy', 'Dir', 'Type', 'Abs_PM', 'Latitude', 'Longitude', 'Lanes')):
            raise ValueError('missing attribute; no filling allowed')
        pm, lat, lon, lanes = (float(fields[k]) for k in ('Abs_PM', 'Latitude', 'Longitude', 'Lanes'))
        if (not np.isfinite([pm, lat, lon, lanes]).all() or not -90 <= lat <= 90
                or not -180 <= lon <= 180 or not 1 <= lanes <= 8 or lanes != int(lanes)):
            raise ValueError('invalid numeric attributes')
        value = {'features': (pm / 100, lat / 90, lon / 180, lanes / 8, float(fields['Type'] == 'ML')),
                 'configuration_id': configs[0], 'interval': (row['start'], row['end_exclusive']),
                 'catalogs': tuple(catalogs), 'source_sha256': tuple(r['source_sha256'] for r in records),
                 'historical_effectiveness_verified': False}
        self._cache[key] = value
        return dict(value)

    def align(self, profile_ids: Sequence[str]) -> NDArray[np.float32]:
        if len(set(profile_ids)) != len(profile_ids):
            raise ValueError('duplicate profile IDs')
        rows = [self.lookup(*identity.split('/'))['features'] for identity in profile_ids]
        return np.asarray(rows, dtype=np.float32).reshape(-1, len(FEATURES))


def iter_conditioned_role(binding: Path, role: str, attributes: AnnualAttributes
                          ) -> Iterator[tuple[Profile, NDArray[np.float32]]]:
    """Attach static rows to actual loader profiles without changing their order/masks."""
    for profile in iter_role(binding, role):
        features = attributes.align([f'{profile[0]}/{profile[1]}'])[0]
        yield profile, features
