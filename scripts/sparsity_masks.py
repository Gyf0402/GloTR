"""Shared neighbour-sparsity masks for the controlled sparsity experiment.

Every method must hide exactly the same neighbour stations, so the masks are
generated once, deterministically, and stored as JSON:

  reviews/2026-11-09/sparsity/masks/<queue>.json
  {episode_id: {"available": [...], "order": [...], "hidden": {"0.25": [...], ...}}}

Rule (fixed before any method is run):
* candidates = OTHER stations with at least one visible entry during the outage days;
* a permutation of the candidates is drawn with a seed derived from the episode id
  and the draw index (sha256), and level f hides the first round(f * n) of it, so
  the 25 % set is contained in the 50 % set, which is contained in the 75 % set;
* hidden stations lose ALL entries on the outage days only (buffer days intact),
  i.e. only contemporaneous neighbour information is reduced.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MASK_DIR = ROOT / 'reviews/2026-11-09/sparsity/masks'
LEVELS = (0.25, 0.5, 0.75)
BUFFER = 3


def _seed(episode_id: str, draw: int) -> int:
    return int(hashlib.sha256(f'{episode_id}|{draw}'.encode()).hexdigest()[:8], 16)


def build_masks(queue: str, episodes: list, draw: int = 0) -> dict:
    """``episodes``: iterable of (episode_id, visible[D,N,288,2], target, duration)."""
    out = {}
    for eid, visible, target, duration in episodes:
        gap = slice(BUFFER, BUFFER + duration)
        nodes = visible.shape[1]
        available = [n for n in range(nodes)
                     if n != target and bool(visible[gap, n].any())]
        order = list(np.random.default_rng(_seed(eid, draw)).permutation(available))
        out[eid] = {'available': available, 'order': [int(x) for x in order],
                    'hidden': {str(f): [int(x) for x in order[:int(round(f * len(order)))]]
                               for f in LEVELS}}
    path = MASK_DIR / f'{queue}__draw{draw}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = json.loads(path.read_text(encoding='utf-8'))
        if existing != out:
            raise SystemExit(f'{path} exists and differs; masks must not change')
    else:
        path.write_text(json.dumps(out, indent=1), encoding='utf-8')
    return out


def load_masks(queue: str, draw: int = 0) -> dict:
    return json.loads((MASK_DIR / f'{queue}__draw{draw}.json').read_text(encoding='utf-8'))


def apply_mask(episode, hidden: list[int], duration: int):
    """Return a copy of a CurrentEpisode with ``hidden`` stations dark on outage days."""
    if not hidden:
        return episode
    visible = episode.visible.copy()
    gap = slice(BUFFER, BUFFER + duration)
    visible[gap, hidden] = False
    values = np.where(visible, episode.values, 0.)
    if np.any(visible & episode.scoring):
        raise AssertionError('scored truth visible after masking')
    return dataclasses.replace(episode, values=values, visible=visible)
