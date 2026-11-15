"""Frozen candidate ``loading_split``: channel-split loading on top of frozen G80.

Registered in ``reviews/2026-09-27/LOADING_SPLIT_PREREGISTRATION.md`` and
``..._AMENDMENT.md`` (v2):

    q channel  <- target-station self-estimated loading (``pseudo_outage``)
    o channel  <- NEIGHBOR loading: weighted mean of the donors' own loadings,
                  w_j proportional to  density_j / (1 + d_j)

where ``d_j`` is the hop distance from the target station to donor ``j`` on the
registered graph (a simple chain) and ``density_j`` is the donor's effective
observation ratio inside the gap window (inference-time available).

Why this file exists as a separate module: ``scripts/run_transductive.py`` and
``models/transductive_residual.py`` are SHA-pinned by the Test-21 report and by
the gate record, so the candidate may not be written into them. This module
therefore holds the candidate's own math and reuses the frozen primitives by
import. It is the SINGLE source of truth from here on; the earlier exploration
script ``scripts/explore_o_loading_neighbors.py`` was a copy used to compare
loading sources, and the integration gate asserts the two agree bit for bit.

The q channel is taken VERBATIM from ``rt.corrected_case`` channel 0, so it is
bit-identical to the current correction by construction (the gate re-checks
this).

Correctness notes that matter:
  * ``V`` is the orthonormal temporal basis returned by ``fit_temporal_factors``;
    the magnitude control stays in the loading regression (``lam_mag``), never in
    the basis -- rescaling the basis would silently undo the shrinkage.
  * Distance is computed from the chain order rather than hard-coded, so it stays
    correct if the graph's station order ever changes.
  * The neighbour kernel is parameter-free on purpose: a length scale would be
    one more knob decided on an already-exposed window.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import scripts.run_transductive as rt
from models.transductive_residual import estimate_loading, fit_temporal_factors

CHANNEL_Q, CHANNEL_O = 0, 1
DEFAULT_O_LOADING = 'neighbor'
LOADING_SOURCES = ('neighbor', 'corridor', 'half')


@dataclass(frozen=True)
class ODiagnostics:
    """Everything the o channel needs to be auditable."""

    basis: np.ndarray          # (duration, SLOTS, rank)
    per_donor: np.ndarray      # (n_donor, rank)
    donor_ids: tuple[str, ...]
    density: np.ndarray        # (n_donor,)
    weights: np.ndarray        # (n_donor,)
    loading: np.ndarray        # (rank,)
    corridor: np.ndarray       # (rank,)
    n_iter: int
    train_rmse: float
    validation_rmse: float
    loading_source: str


def station_order() -> tuple[str, ...]:
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    graph = json.loads((root / 'data/pems/graphs/qo_pilot_v1/release_v1/'
                        'graph.json').read_text(encoding='utf-8'))
    return tuple(str(x) for x in graph['station_order'])


def hop_distance(order: tuple[str, ...], a: str, b: str) -> int:
    """Graph distance in hops on the registered chain."""
    return abs(order.index(a) - order.index(b))


def neighbor_weights(order, target, donor_ids, density) -> np.ndarray:
    """``w_j = density_j / (1 + hop(target, donor_j))`` -- parameter-free."""
    if len(donor_ids) == 0:
        return np.zeros(0)
    return np.asarray([float(density[i])
                       / (1.0 + hop_distance(order, target, donor_ids[i]))
                       for i in range(len(donor_ids))])


def weighted_mean(per_donor: np.ndarray, w: np.ndarray) -> np.ndarray:
    if per_donor.shape[0] == 0:
        return np.zeros(per_donor.shape[1] if per_donor.ndim == 2 else 0)
    if (not np.isfinite(w).all()) or w.sum() <= 0:
        return np.mean(per_donor, axis=0)
    return (w[:, None] * per_donor).sum(axis=0) / w.sum()


def o_fit(case, o_rank, lam_mag, lam_smooth, drop_dark=False) -> ODiagnostics:
    """Fit the o channel's temporal basis and per-donor loadings.

    Mirrors ``rt.fit_channel``'s loading block operation for operation (same
    frozen primitives, same order), but keeps the per-donor vectors so a
    neighbour-weighted loading can be formed.
    """
    field = case['field'][..., CHANNEL_O]
    observed = case['observed'][..., CHANNEL_O]
    n_station, n_day, n_slot = field.shape
    density = observed.reshape(n_station, -1).mean(axis=1)
    donor_ids = [str(x) for x in case['donors']]
    if drop_dark:
        keep = np.flatnonzero(density > 0.0)
        if keep.size:
            field, observed = field[keep], observed[keep]
            density = density[keep]
            donor_ids = [donor_ids[i] for i in keep.tolist()]
            n_station = field.shape[0]

    eligible = np.flatnonzero(density > 0.5)
    if eligible.size < rt.VALIDATION_STRIDE:
        eligible = np.arange(n_station)
    validation = eligible[::rt.VALIDATION_STRIDE]
    fit_rows = np.asarray([i for i in range(n_station)
                           if i not in set(validation.tolist())])
    fit = fit_temporal_factors(
        field.reshape(n_station, n_day * n_slot),
        observed.reshape(n_station, n_day * n_slot),
        rank=o_rank, lam_mag=lam_mag, lam_smooth=lam_smooth, n_iter=rt.N_ITER,
        day_length=n_slot, validation_rows=validation, fit_rows=fit_rows)
    basis = fit.V.reshape(n_day, n_slot, fit.V.shape[1])
    average_basis = basis.mean(axis=0)

    per_donor, kept_ids, kept_density = [], [], []
    for index in range(n_station):
        slot_valid = observed[index].any(axis=0)
        if int(slot_valid.sum()) < fit.V.shape[1]:
            continue
        average, _ = rt._slot_average(field[index], observed[index])
        per_donor.append(estimate_loading(average, average_basis, slot_valid,
                                         lam=lam_mag))
        kept_ids.append(donor_ids[index])
        kept_density.append(float(density[index]))
    stacked = (np.stack(per_donor) if per_donor
               else np.zeros((0, fit.V.shape[1])))
    return ODiagnostics(basis=basis, per_donor=stacked,
                        donor_ids=tuple(kept_ids),
                        density=np.asarray(kept_density),
                        weights=np.zeros(len(kept_ids)), loading=None,
                        corridor=(np.mean(stacked, axis=0) if stacked.shape[0]
                                  else np.zeros(fit.V.shape[1])),
                        n_iter=fit.n_iter, train_rmse=fit.train_rmse,
                        validation_rmse=fit.validation_rmse,
                        loading_source='')


def o_loading(case, order, o_rank, lam_mag, lam_smooth, *,
              loading=DEFAULT_O_LOADING, drop_dark=False):
    """Return (diagnostics with the chosen loading, o residual)."""
    if loading not in LOADING_SOURCES:
        raise ValueError(f'unknown o loading source {loading!r}')
    diag = o_fit(case, o_rank, lam_mag, lam_smooth, drop_dark=drop_dark)
    target = str(case['target_station'])
    w = neighbor_weights(order, target, diag.donor_ids, diag.density)
    neighbour = weighted_mean(diag.per_donor, w)
    if loading == 'neighbor':
        chosen = neighbour
    elif loading == 'corridor':
        chosen = diag.corridor
    else:
        chosen = 0.5 * diag.corridor + 0.5 * neighbour
    diag = ODiagnostics(**{**diag.__dict__, 'weights': w, 'loading': chosen,
                           'loading_source': loading})
    return diag, diag.basis @ chosen


def q_residual(case, rank, lam_mag, lam_smooth):
    """The frozen path's q channel (channel 0), verbatim."""
    return rt.corrected_case(case, rank, lam_mag, lam_smooth)[0][..., CHANNEL_Q]


def corrected_case(case, order, *, rank=4, lam_mag=0.1, lam_smooth=0.1,
                   o_rank=None, loading=DEFAULT_O_LOADING, drop_dark=False,
                   q_lam_mag=None, q_lam_smooth=None):
    """Candidate correction: (residual (duration, SLOTS, 2), diagnostics).

    q is the frozen path's channel 0; o is the neighbour (or chosen) loading on
    the o channel's own basis at ``o_rank`` (default: same as ``rank``).

    ``q_lam_mag`` / ``q_lam_smooth`` exist so an O-ONLY change can leave q alone:
    they default to the values passed for o, but whenever only the o channel's
    regularisation is being varied the caller MUST pass the frozen q values
    explicitly, otherwise the q channel silently moves too.
    """
    o_rank = rank if o_rank is None else o_rank
    q = q_residual(case, rank,
                   lam_mag if q_lam_mag is None else q_lam_mag,
                   lam_smooth if q_lam_smooth is None else q_lam_smooth)
    diag, o = o_loading(case, order, o_rank, lam_mag, lam_smooth,
                        loading=loading, drop_dark=drop_dark)
    diagnostics = {
        'q': {'loading_source': 'pseudo_outage(main)', 'rank': rank},
        'o': {'loading_source': diag.loading_source, 'rank': o_rank,
              'n_iter': diag.n_iter, 'train_rmse': diag.train_rmse,
              'validation_rmse': diag.validation_rmse,
              'n_donor_with_loadings': int(diag.per_donor.shape[0]),
              'loading_norm': float(np.linalg.norm(diag.loading)),
              'corridor_loading_norm': float(np.linalg.norm(diag.corridor)),
              'weights_sum': float(np.sum(diag.weights)),
              'residual_rms': float(np.sqrt(np.mean(o ** 2)))},
    }
    return np.stack([q, o], axis=-1), diagnostics


def compose(control: np.ndarray, residual: np.ndarray) -> np.ndarray:
    return control + residual
