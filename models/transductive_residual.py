"""Per-episode low-rank transductive residual correction on top of frozen G80.

Model (pre-registered):  X_hat_c = X_G80,c + R_c,  R_c = U_c V_cᵀ,  c in {q,o}

Design notes that matter for correctness:

  * ``V`` is returned as an ORTHONORMAL temporal basis. A basis carrying the
    singular values cannot be used in ``filled @ V @ V.T`` (that inflates the
    fill by s²) and, worse, any regression of the target's errors onto a scaled
    basis would silently undo the shrinkage -- so the magnitude control must sit
    in the target's loading regression, not in the basis.
  * ``lam_mag`` therefore enters twice, in two distinct roles: as the
    singular-value threshold that decides which temporal components survive, and
    as the ridge penalty in :func:`estimate_loading`, which is what actually
    limits the correction magnitude.
  * The temporal factors are fitted on ``fit_rows`` only; ``validation_rows``
    are held out, and their error decides the stopping iteration, so the choice
    is not made on the rows that later supply the correction.
  * q and o are fitted independently and never share arrays.
  * A zero correction reproduces G80 exactly (``R`` is additive, defaults to 0).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LowRankFit:
    """Orthonormal temporal basis plus the iteration chosen by holdout rows."""

    V: np.ndarray            # (n_time, rank), orthonormal columns
    n_iter: int
    train_rmse: float
    validation_rmse: float


def _soft_threshold(values: np.ndarray, lam: float) -> np.ndarray:
    return np.sign(values) * np.maximum(np.abs(values) - lam, 0.0)


def _smooth_slots(V: np.ndarray, day_length: int, strength: float) -> np.ndarray:
    """One gradient step on ½Σ|V[i+1]-V[i]|² along the within-day slot axis.

    Smoothness is enforced on the slot index only, so a factor may still change
    freely from day to day; it just may not jump between adjacent 5-minute bins.
    The step is scaled by 1/4, the stable limit for the 1-D Laplacian, so
    strength in [0,1] never overshoots into rougher solutions. Columns are
    NOT re-orthonormalised: doing so would rescale them and silently undo the
    measured effect (the ridge regression downstream tolerates the collinearity).
    """
    if strength <= 0:
        return V
    days = V.reshape(-1, day_length, V.shape[1])
    diff = np.diff(days, axis=1)                       # (days, L-1, r)
    # gradient of 0.5*sum(d^2) is d_{j-1} - d_j; the interior sign matters, and
    # flipping it turns this into an ANTI-smoothing step that grows roughness.
    step = np.zeros_like(days)
    step[:, :-1, :] -= diff
    step[:, 1:, :] += diff
    return (days - strength * step / 4.0).reshape(V.shape)


def fit_temporal_factors(E: np.ndarray, observed: np.ndarray, *,
                         rank: int, lam_mag: float, lam_smooth: float,
                         n_iter: int, day_length: int,
                         validation_rows: np.ndarray,
                         fit_rows: np.ndarray) -> LowRankFit:
    """Soft-impute fit of the donor error field on the fit rows only.

    ``E`` is (n_station, n_time); ``observed`` marks usable entries.
    """
    if rank < 1:
        raise ValueError('rank must be positive')
    usable = observed.copy()
    if validation_rows.size:
        usable[validation_rows] = False
    masked = np.where(usable, E, 0.0)
    column_mean = np.divide(masked.sum(axis=0),
                            np.maximum(usable.sum(axis=0), 1))
    filled = np.where(usable, E, column_mean[None, :])

    best: LowRankFit | None = None
    for step in range(1, n_iter + 1):
        u, s, vt = np.linalg.svd(filled, full_matrices=False)
        shrunk = _soft_threshold(s, lam_mag)
        chosen = [i for i in np.argsort(-shrunk)
                  if shrunk[i] > 1e-12][:rank]
        if chosen:
            estimate = (u[:, chosen] * shrunk[chosen]) @ vt[chosen]
            basis = vt[chosen].T
        else:
            estimate = np.zeros_like(filled)
            basis = np.zeros((filled.shape[1], rank))
        basis = _smooth_slots(basis, day_length, lam_smooth)
        filled = np.where(usable, E, estimate)

        train_rmse = float(np.sqrt(np.mean(
            (estimate[fit_rows] - E[fit_rows])[observed[fit_rows]] ** 2)))
        if validation_rows.size and observed[validation_rows].any():
            validation_rmse = float(np.sqrt(np.mean(
                (estimate[validation_rows] - E[validation_rows])
                [observed[validation_rows]] ** 2)))
        else:
            validation_rmse = float('nan')
        if best is None or (validation_rmse == validation_rmse
                           and validation_rmse < best.validation_rmse):
            best = LowRankFit(V=basis.copy(), n_iter=step,
                              train_rmse=train_rmse,
                              validation_rmse=validation_rmse)
    assert best is not None
    return best


def estimate_loading(target_errors: np.ndarray, V: np.ndarray,
                     observed: np.ndarray, *, lam: float) -> np.ndarray:
    """Ridge estimate of the target loading from its MASKED error samples.

    ``lam`` is the magnitude control: it shrinks the loading towards zero, which
    is what limits how far the correction may move the prediction.
    """
    keep = np.asarray(observed, dtype=bool) & np.isfinite(target_errors)
    if keep.sum() < V.shape[1] or not keep.any():
        return np.zeros(V.shape[1])
    design = V[keep]
    target = target_errors[keep]
    gram = design.T @ design + lam * np.eye(V.shape[1])
    return np.linalg.solve(gram, design.T @ target)


def correction(u: np.ndarray, V: np.ndarray) -> np.ndarray:
    """R(t) = V(t) · u."""
    return V @ u


def compose(g80: np.ndarray, residual: np.ndarray | None) -> np.ndarray:
    """Add the correction; ``None`` returns G80 exactly."""
    if residual is None:
        return g80
    return g80 + residual
