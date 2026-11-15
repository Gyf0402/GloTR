"""Legal per-cell features for the target-conditioned occupancy probe."""

from __future__ import annotations

import numpy as np

OFFSETS = (-2, -1, 1, 2)


def append_tree_features(base_features, prediction, control, gap_visible, *, target):
    """Append current prediction, time-of-day and neighbour visibility.

    `gap_visible` is the original episode's visible array restricted to gap days;
    the target's gap is masked by the episode assembler. No reference is read.
    """
    base = np.asarray(base_features, dtype=np.float32)
    pred = np.asarray(prediction, dtype=np.float64)
    raw = np.asarray(control, dtype=np.float64)
    visible = np.asarray(gap_visible, dtype=bool)
    if (base.ndim != 3 or pred.shape != base.shape[:2] + (2,) or
            raw.shape != pred.shape or visible.ndim != 4 or
            visible.shape[0] != base.shape[0] or
            visible.shape[2:] != (base.shape[1], 2)):
        raise ValueError('feature inputs have inconsistent dimensions')
    days, slots = base.shape[:2]
    angle = 2 * np.pi * np.arange(slots) / slots
    columns = [base, pred[..., 1:2].astype(np.float32),
               (pred[..., 1:2] - raw[..., 1:2]).astype(np.float32),
               np.broadcast_to(np.sin(angle)[None, :, None], (days, slots, 1)).astype(np.float32),
               np.broadcast_to(np.cos(angle)[None, :, None], (days, slots, 1)).astype(np.float32)]
    for offset in OFFSETS:
        node = target + offset
        if 0 <= node < visible.shape[1]:
            columns.append(visible[:, node].astype(np.float32))
        else:
            columns.append(np.zeros((days, slots, 2), dtype=np.float32))
    return np.concatenate(columns, axis=-1)
