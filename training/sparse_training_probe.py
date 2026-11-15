"""Fixed-data optimization probes; no validation data or checkpoint selection."""

from __future__ import annotations

import time

import torch
from torch import Tensor

from models.sparse_observation_profile import SparseObservationProfile


def channel_l1(predicted: Tensor, truth: Tensor, scoring: Tensor) -> Tensor:
    if (predicted.shape != truth.shape or predicted.shape != scoring.shape or
            predicted.ndim != 2 or predicted.shape[-1] != 2 or
            scoring.dtype != torch.bool or not bool(scoring.any(dim=0).all())):
        raise ValueError('nonempty matched two-channel scoring required')
    return torch.stack([(predicted[:, c][scoring[:, c]] -
                         truth[:, c][scoring[:, c]]).abs().mean() for c in range(2)])


def fit_fixed_batch(initial: dict, query: Tensor, keys: Tensor, valid: Tensor,
                    base: Tensor, truth: Tensor, scoring: Tensor, *,
                    lr: float, steps: int, record_every: int = 25) -> dict:
    """Fresh AdamW with all regularizers disabled; retain final iterate only."""
    if lr <= 0 or steps < 1 or record_every < 1:
        raise ValueError('positive learning rate and iteration counts required')
    model = SparseObservationProfile(dropout=0.).to(query.device)
    model.load_state_dict(initial)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.)
    history = []
    started = time.perf_counter()

    def record(step: int) -> Tensor:
        model.eval()
        with torch.no_grad():
            prediction, _ = model(query, keys, valid, base)
            losses = channel_l1(prediction, truth, scoring)
        history.append({'step': step, 'q_l1': float(losses[0]),
                        'o_l1': float(losses[1]), 'mean_l1': float(losses.mean())})
        return prediction

    prediction = record(0)
    stop_reason = None
    completed = 0
    for step in range(1, steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction, _ = model(query, keys, valid, base)
        loss = channel_l1(prediction, truth, scoring).mean()
        if not torch.isfinite(loss):
            stop_reason = 'nonfinite_loss'
            break
        loss.backward()
        if not all(p.grad is None or bool(torch.isfinite(p.grad).all())
                   for p in model.parameters()):
            stop_reason = 'nonfinite_gradient'
            break
        optimizer.step()
        completed = step
        if step % record_every == 0 or step == steps:
            prediction = record(step)
    if history[-1]['step'] != completed:
        prediction = record(completed)
    return {'history': history, 'prediction': prediction.detach(),
            'model_state': model.state_dict(), 'steps_completed': completed,
            'stop_reason': stop_reason, 'seconds': time.perf_counter() - started,
            'lr': lr}
