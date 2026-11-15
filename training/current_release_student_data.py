"""Current-release role loader for the Train→Selection student experiment."""
from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from io import BytesIO
from pathlib import Path
from typing import Any, Iterator, cast

import numpy as np
from numpy.typing import NDArray
from torch import Tensor

from models.film_rgnn import SemanticDirectedGNN
from representations.basis import GlobalProfileBasis, MeanFunction
from training.real_qo_loader import iter_role, select_day


def verified_bytes(path: Path, expected: str) -> bytes:
    if not isinstance(expected, str) or re.fullmatch('[0-9a-f]{64}', expected) is None:
        raise ValueError('invalid SHA-256')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f'SHA-256 mismatch: {path}')
    return raw


class VerifiedRoleReader:
    """Read hashed daily bytes only within one explicitly selected role."""

    def __init__(self, root: Path, release_path: Path, expected: str,
                 role: str, stations: tuple[str, ...]):
        if role not in {'train', 'selection'}:
            raise ValueError('only train or selection is permitted')
        release = json.loads(verified_bytes(release_path, expected))
        if release['status'] != 'frozen':
            raise ValueError('release must be frozen')
        self.root = root
        self.bounds = tuple(release['roles'][role])
        def pinned(item: dict[str, str]) -> dict[str, Any]:
            return cast(dict[str, Any], json.loads(verified_bytes(root / item['path'], item['sha256'])))
        entries = pinned({'path': release['u_manifest'],
                          'sha256': release['permanent_u_sha256']})['stations']
        self.universe = {str(x['station_id']): x['permanent_U'] for x in entries}
        if len(self.universe) != len(entries) or any(type(v) is not bool for v in self.universe.values()):
            raise ValueError('invalid U universe')
        if len(set(stations)) != len(stations) or any(self.universe.get(s, True) for s in stations):
            raise ValueError('graph includes unknown, duplicate or U station')
        days = pinned(release['array_manifest'])['days']
        self.days = {x['day']: x for x in days}
        if len(self.days) != len(days):
            raise ValueError('duplicate day binding')
        self.stations = set(stations)
        root_dir = Path(release['output_root'])
        if not root_dir.is_absolute():
            # A corridor-cropped release records the array root relative to the
            # repository, so the tree stays portable.
            root_dir = root / root_dir
        self.output_root = root_dir.resolve()
        self.access_log: list[dict[str, str]] = []

    def __call__(self, day: str) -> list[Profile]:
        date.fromisoformat(day)
        if not self.bounds[0] <= day <= self.bounds[1]:
            raise ValueError('day outside selected role; access refused')
        entry = self.days[day]
        day_dir = Path(entry['path'])
        if not day_dir.is_absolute():
            day_dir = self.root / day_dir
        path = day_dir.resolve() / 'arrays.npz'
        if path.parent != self.output_root / day:
            raise ValueError('array path escapes registered day')
        raw = verified_bytes(path, entry['sha256'])
        with np.load(BytesIO(raw), allow_pickle=False) as archive:
            ids, values, masks = select_day(archive, self.universe)
        self.access_log.append({'day': day, 'sha256': entry['sha256']})
        return [(day, str(s), values[i], masks[i]) for i, s in enumerate(ids)
                if str(s) in self.stations]

Profile = tuple[str, str, NDArray[np.float64], NDArray[np.bool_]]

def iter_current_role(binding: Path, role: str) -> Iterator[Profile]:
    """Yield only the registered Train or Selection role; never Calibration/Test."""
    if role not in {"train", "selection"}:
        raise ValueError("current student experiment permits only train or selection")
    yield from iter_role(binding, role)

def group_profiles(binding: Path, role: str) -> dict[str, list[Profile]]:
    grouped: dict[str, list[Profile]] = defaultdict(list)
    for item in iter_current_role(binding, role):
        grouped[item[0]].append(item)
    return dict(grouped)


@dataclass(frozen=True)
class CurrentEpisode:
    episode_id: str
    mask_id: str
    role: str
    dates: tuple[str, ...]
    station_order: tuple[str, ...]
    target_index: int
    values: NDArray[np.float64]
    visible: NDArray[np.bool_]
    reference: NDArray[np.float64]
    scoring: NDArray[np.bool_]


def train_teacher(role: str, basis: GlobalProfileBasis, reference: Tensor,
                  scoring: Tensor, mean: Tensor) -> tuple[Tensor, Tensor]:
    """Return labels only for Train; never call this on Selection references."""
    if role != 'train':
        raise ValueError('teacher generation is Train-only')
    import torch
    if any(p.requires_grad or p.grad is not None for p in basis.parameters()):
        raise ValueError('teacher requires frozen basis')
    with torch.no_grad():
        selected = scoring.any(dim=(-1, -2))
        if not selected.any():
            raise ValueError('no teacher support')
        labels = basis.project(reference[selected], scoring[selected], mean[selected])
        if not torch.isfinite(labels).all():
            raise ValueError('nonfinite teacher')
        return selected, labels.detach().clone()


def prepare_case(episode: CurrentEpisode, basis: GlobalProfileBasis, mu: MeanFunction,
                 scales: Tensor, static: Tensor, edges: dict[str, Tensor]) -> dict[str, Any]:
    """Prepare frozen features and separate labels, without optimizing anything."""
    import torch

    from training.pilot_data import calendar_features, rolling_history
    from training.real_student_adapter import build_student_inputs
    if episode.role not in {'train', 'selection'}:
        raise ValueError('unsupported episode role')
    if any(p.requires_grad or p.grad is not None for m in (basis, mu) for p in m.parameters()):
        raise ValueError('representation must be frozen')
    device = next(basis.parameters()).device
    if scales.shape != (basis.channels,) or not torch.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError('positive frozen channel scales required')
    with torch.no_grad():
        values = torch.as_tensor(episode.values, dtype=torch.float32, device=device) / scales
        visible = torch.as_tensor(episode.visible, device=device)
        values = torch.where(visible, values, torch.zeros_like(values))
        history = rolling_history(values, visible)
        calendar = calendar_features(episode.dates).to(device)
        per_day_static = static[None].expand(len(episode.dates), -1, -1)
        mean = basis.global_mean + mu(per_day_static, calendar, history, protocol='K')
        inputs = build_student_inputs(basis, values, visible, mean, static, edges, episode.dates)
        scoring = torch.as_tensor(episode.scoring, device=device)
        reference = torch.as_tensor(episode.reference, dtype=torch.float32, device=device) / scales
        target: Tensor | None = None
        teacher: Tensor | None = None
        if episode.role == 'train':
            target, teacher = train_teacher('train', basis, reference, scoring, mean)
        return {'inputs': inputs, 'reference': reference, 'scoring': scoring,
                'teacher_target': target, 'teacher': teacher, 'static': static,
                'role': episode.role,
                'ordinals': torch.tensor([date.fromisoformat(d).toordinal() for d in episode.dates],
                                         device=device)}


def predict_case(model: SemanticDirectedGNN, case: dict[str, Any]) -> Tensor:
    import torch
    inputs = case['inputs']
    return cast(Tensor, model(inputs.features, case['static'], inputs.calendar, inputs.edges,
                 protocol='K', episode_ids=torch.zeros_like(case['ordinals']),
                 day_ordinals=case['ordinals']))


def fit_student(model: SemanticDirectedGNN, cases: list[dict[str, Any]], *,
                epochs: int, lr: float, output: Path | None = None,
                checkpoint_provenance: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Fixed coefficient-only fit; one averaged Adam step per complete epoch."""
    import time

    import torch
    if not cases or any(c['role'] != 'train' or c['teacher'] is None for c in cases):
        raise ValueError('only nonempty Train teacher cases may be fitted')
    if type(epochs) is not int or epochs <= 0 or lr <= 0:
        raise ValueError('positive epochs and learning rate required')
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    labels = [c['teacher'].clone() for c in cases]
    records = []
    model.train()
    for epoch in range(epochs):
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        total = 0.
        for case, original in zip(cases, labels):
            if not torch.equal(case['teacher'], original):
                raise ValueError('teacher drift detected')
            prediction = predict_case(model, case)
            loss = torch.nn.functional.l1_loss(prediction[case['teacher_target']], original)
            if not torch.isfinite(loss):
                raise ValueError('nonfinite coefficient loss')
            torch.autograd.backward(loss / len(cases))
            total += float(loss.detach()) / len(cases)
        gradients = {}
        for name, parameter in model.named_parameters():
            grad = parameter.grad
            if grad is not None and not torch.isfinite(grad).all():
                raise ValueError('nonfinite gradient')
            gradients[name] = None if grad is None else float(grad.norm())
        optimizer.step()
        row = {'epoch': epoch + 1, 'coef': total, 'teacher_drift': 0., 'lr': lr,
               'seconds': time.perf_counter() - started, 'gradient_norms': gradients}
        records.append(row)
        if output is not None:
            with (output / 'metrics.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(row, allow_nan=False) + '\n')
    if output is not None:
        with (output / 'student.pt').open('xb') as stream:
            torch.save({'student': model.state_dict(), 'optimizer': optimizer.state_dict(),
                        'epochs': epochs, 'lr': lr, 'provenance': checkpoint_provenance}, stream)
    model.eval()
    return records


def assemble_episode(
    row: Mapping[str, Any], station_order: tuple[str, ...],
    role_bounds: tuple[str, str], read_day: Callable[[str], Sequence[Profile]],
) -> CurrentEpisode:
    """Assemble one roster episode with three-day buffers and hidden target inputs.

    This pure assembler receives already verified q/o profiles. The caller owns
    release/array hashes and U membership checks. Reference/scoring are labels,
    never model features. Missing donor profiles remain explicitly unavailable.
    """
    if row['split'] not in {'train', 'selection'} or row['protocol'] != 'K':
        raise ValueError('only Train/Selection Protocol-K episodes are supported')
    if not station_order or len(set(station_order)) != len(station_order):
        raise ValueError('unique graph station order required')
    target = str(row['station_id'])
    if target not in station_order:
        raise ValueError('target outside graph station order')
    duration = row['duration_days']
    if type(duration) is not int or duration not in {7, 14, 28}:
        raise ValueError('registered 7/14/28-day duration required')
    start = date.fromisoformat(row['start_day']) - timedelta(days=3)
    dates = tuple((start + timedelta(days=i)).isoformat() for i in range(duration + 6))
    if dates[0] < role_bounds[0] or dates[-1] > role_bounds[1]:
        raise ValueError('episode or buffer crosses role boundary')
    shape = (len(dates), len(station_order), 288, 2)
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
            if values.shape != (288, 2) or mask.shape != values.shape or mask.dtype != bool:
                raise ValueError('expected q/o profile and boolean legal mask')
            if not np.isfinite(values[mask]).all():
                raise ValueError('nonfinite legal values')
            n = lookup[station]
            reference[d, n] = np.where(mask, values, 0.)
            visible[d, n] = mask
        if not visible[d, lookup[target]].all(axis=-1).sum() >= 274:
            raise ValueError('target or buffer fails registered complete-day eligibility')
    scoring = np.zeros(shape, dtype=bool)
    scoring[3:3 + duration, lookup[target]] = visible[3:3 + duration, lookup[target]]
    visible[3:3 + duration, lookup[target]] = False
    values = np.where(visible, reference, 0.)
    return CurrentEpisode(row['episode_id'], row['mask_id'], row['split'], dates,
                          station_order, lookup[target], values, visible, reference, scoring)
