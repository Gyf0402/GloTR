"""Candidate calendar validation only; not a SHA-verified training release."""

import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path


class ReleaseError(ValueError):
    """Invalid candidate calendar or source references."""


@dataclass(frozen=True)
class QoRelease:
    primary_clock: str
    sensitivity_clock: str
    output_root: str
    roles: dict[str, tuple[str, str]]
    u_manifest: str


def load_qo_release(path: Path) -> QoRelease:
    x = json.loads(path.read_text(encoding='utf-8'))
    required = ('train', 'selection', 'calibration', 'test')
    if (x.get('primary_clock') != 'fixed_UTC_minus_8'
            or x.get('sensitivity_clock') != 'America/Los_Angeles'):
        raise ReleaseError('clock binding')
    for field in ('output_root', 'u_manifest'):
        if not isinstance(x.get(field), str) or not x[field]:
            raise ReleaseError('missing bindings')
    raw_roles = x.get('roles')
    if not isinstance(raw_roles, dict) or set(raw_roles) != set(required):
        raise ReleaseError('exactly four roles required')
    roles: dict[str, tuple[str, str]] = {}
    previous = None
    for key in required:
        interval = raw_roles[key]
        if (not isinstance(interval, list) or len(interval) != 2
                or not all(isinstance(v, str) for v in interval)):
            raise ReleaseError('each role needs two ISO dates')
        a, b = interval
        try:
            start, end = date.fromisoformat(a), date.fromisoformat(b)
        except ValueError as exc:
            raise ReleaseError('invalid date') from exc
        if start > end:
            raise ReleaseError('reversed role')
        if previous is not None and start != previous + timedelta(days=1):
            raise ReleaseError('overlap or gap between roles')
        previous = end
        roles[key] = (a, b)
    return QoRelease(x['primary_clock'], x['sensitivity_clock'], x['output_root'],
                     roles, x['u_manifest'])
