"""Case-local spatial interpolation of occupancy residual loadings."""

from __future__ import annotations

import numpy as np


def local_linear_loading(positions, loadings, weights, *, target_position, ridge):
    """Predict target loading by a weighted local line along graph position.

    The slope is regularised; infinite ridge becomes the neighbour-weighted
    constant estimate. No target outage truth is an input.
    """
    x = np.asarray(positions, dtype=np.float64) - float(target_position)
    y = np.asarray(loadings, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if (x.ndim != 1 or y.ndim != 2 or y.shape[0] != len(x) or
            w.shape != x.shape or np.any(w < 0) or ridge < 0 or
            not np.isfinite(x).all() or not np.isfinite(y).all() or
            not np.isfinite(w).all()):
        raise ValueError('invalid spatial loading inputs')
    if len(x) == 0:
        raise ValueError('no donors')
    if w.sum() <= 0:
        w = np.ones_like(w)
    w = w / w.sum()
    mean_x = np.sum(w * x)
    mean_y = np.sum(w[:, None] * y, axis=0)
    if np.isinf(ridge):
        return mean_y
    centered = x - mean_x
    variance = np.sum(w * centered ** 2)
    slope = np.sum(w[:, None] * centered[:, None] * (y - mean_y), axis=0)
    slope = slope / (variance + ridge) if variance + ridge > 0 else np.zeros_like(mean_y)
    return mean_y - slope * mean_x
