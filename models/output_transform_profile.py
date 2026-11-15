"""Output-transform arms for the occupancy experiment (hypothesis under test).

Purpose
-------
The residual head under-delivers magnitude at high occupancy: on the exposed
Test-21 diagnostics it supplied only 26.6 percent of the correction the frozen
base needed in the ``[.30,1]`` occupancy interval, and on Selection-42 the first
head-shape experiment (A6) moved that interval's o error by -3.74 percent while
the low-occupancy intervals got worse, leaving the aggregate unchanged.

This module lets the **o channel's correction magnitude be reshaped** without
touching the base profile, the q channel, or the attention body::

    correction = head(hidden)                       # unchanged, both channels
    correction[:, 1] = transform(correction[:, 1])  # o only

What this is and is not
-----------------------
This is a **hypothesis under test**, not a presumed fix. An earlier shape
diagnostic does *not* support the "logit" story: regressing the required
correction on ``logit(true occupancy)`` fit worse than on the occupancy itself
(R^2 0.191 vs 0.361), and that regression used the true occupancy, which is
unavailable at inference. Nothing here assumes a sigmoid is the right shape.

The transforms target the mechanism that *is* measured -- the head's magnitude
shortfall -- and each starts at exactly the identity:

``identity``  the frozen head verbatim. Same-seed control; must reproduce the
              existing ``full`` arm bitwise (hard gate in the pre-registration).
``gain``      ``z * (1 + phi * |z|)``, a magnitude-dependent gain. ``phi > 0``
              gives large corrections more room while leaving small ones almost
              untouched, which is what the measured shortfall calls for.
              ``phi = 0`` gives **exactly** ``z`` (multiplying by 1 is exact in
              binary floating point), so the transplanted model's step-0
              function equals the shared head's bitwise.
``bound``     ``(1 - lam) * z + lam * M * tanh(z / M)``. ``lam = 0`` gives
              exactly ``z`` by construction. This is the literal
              "logit/squash" family, included so that its failure or success is
              *observable* rather than assumed.

Why not an exponent map
-----------------------
The obvious "power" map ``sign(z) * ((|z| + eps)**p - eps**p)`` has to bypass
the transform when ``p == 1`` to be bitwise identity at init, and that bypass
makes ``d(loss)/d(parameter)`` exactly zero at ``p = 1``: the arm would stay at
the identity forever while appearing to be a live hypothesis. (This is the same
class of trap as zero-initialised rows in a widened head, where the added
capacity can never switch on.) Both maps above avoid it: they are exactly the
identity at their initial scalar *and* have a non-zero derivative there.

Both transforms act on the correction that is *added to* the frozen base, so
they cannot undo a compression inside the base itself; they can only give the
head more room to compensate. ``M`` is the legal maximum of the normalised
occupancy (``1 / 0.2127663``).

Only the o channel is transformed. The q channel and the attention body keep the
frozen path, so a q change can only arrive indirectly through shared training,
never through this transform.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from models.sparse_global_tokens import GlobalTokenProfile

TRANSFORM_KINDS = ('identity', 'gain', 'bound')
TRANSFORM_ARMS = ('ot_identity', 'ot_gain', 'ot_bound')
ARM_TRANSFORMS = dict(zip(TRANSFORM_ARMS, TRANSFORM_KINDS))
#: Parameter name of the single learned scalar, per kind.
SCALAR_NAME = {'gain': 'phi', 'bound': 'lam'}
#: The o channel is index 1 everywhere in this codebase.
TRANSFORMED_CHANNEL = 1
#: Legal maximum of the normalised occupancy: physical o <= 1, scale 0.2127663.
OCCUPANCY_MAX_NORM = 1.0 / 0.21276629999999888


class OutputTransformProfile(GlobalTokenProfile):
    """The frozen model with a monotone reshape of the o channel's correction.

    ``kind='identity'`` is delegation to the parent class, so that arm is the
    frozen model verbatim. The other kinds keep the parent's body and head and
    add exactly one scalar parameter; nothing else is added or removed.
    """

    def __init__(self, *, dropout: float = .1, kind: str = 'identity'):
        super().__init__(dropout=dropout)
        if kind not in TRANSFORM_KINDS:
            raise ValueError(f'unknown transform kind {kind!r}; '
                             f'{TRANSFORM_KINDS}')
        self.kind = kind
        if kind == 'identity':
            return
        setattr(self, SCALAR_NAME[kind], nn.Parameter(torch.zeros(())))

    def _transform(self, value: Tensor) -> Tensor:
        if self.kind == 'identity':
            return value
        if self.kind == 'gain':
            phi = self.phi
            return value * (1.0 + phi * value.abs())
        lam = self.lam
        return (1.0 - lam) * value + lam * OCCUPANCY_MAX_NORM * torch.tanh(
            value / OCCUPANCY_MAX_NORM)

    def forward(self, query: Tensor, keys: Tensor, valid: Tensor,
                base: Tensor) -> tuple[Tensor, Tensor]:
        if self.kind == 'identity':
            return super().forward(query, keys, valid, base)
        # Reuse the frozen forward for the body and the shared head, then reshape
        # only the o correction: the q column is copied through unchanged.
        _, correction = super().forward(query, keys, valid, base)
        reshaped = correction.clone()
        reshaped[:, TRANSFORMED_CHANNEL] = self._transform(
            correction[:, TRANSFORMED_CHANNEL])
        return base + reshaped, reshaped


def build_transform_profile(arm: str, *, dropout: float = .1) -> GlobalTokenProfile:
    if arm not in ARM_TRANSFORMS:
        raise ValueError(f'unknown output-transform arm {arm!r}; '
                         f'{TRANSFORM_ARMS}')
    return OutputTransformProfile(dropout=dropout, kind=ARM_TRANSFORMS[arm])


def transform_state(arm: str, shared_state: dict[str, Tensor]) -> dict[str, Tensor]:
    """Initial state: the shared state plus a zero-initialised transform scalar.

    Both transform kinds are exactly the identity at their initial scalar, so
    the transplanted model starts at the shared head's function bitwise. That
    equality is what makes the identity arm a genuine same-seed control and what
    removes "different starting point" as an explanation for a ``gain`` or
    ``bound`` result.
    """
    if arm not in ARM_TRANSFORMS:
        raise ValueError(f'unknown output-transform arm {arm!r}')
    kind = ARM_TRANSFORMS[arm]
    state = dict(shared_state)
    if kind == 'identity':
        return state
    state[SCALAR_NAME[kind]] = torch.zeros(())
    return state


def initial_scalar(arm: str) -> float:
    kind = ARM_TRANSFORMS[arm]
    return 0.0 if kind in SCALAR_NAME else float('nan')


def transform_report() -> dict[str, dict[str, object]]:
    """Parameter counts and the initial transform value per kind."""
    report = {}
    frozen_total = sum(p.numel() for p in GlobalTokenProfile(dropout=.0)
                       .parameters())
    for kind in TRANSFORM_KINDS:
        model = OutputTransformProfile(dropout=.0, kind=kind)
        total = sum(p.numel() for p in model.parameters())
        report[kind] = {
            'total': total,
            'extra_parameters': total - frozen_total,
            'initial_scalar': (float(getattr(model, SCALAR_NAME[kind]).detach())
                               if kind in SCALAR_NAME else 0.0),
            'identity_at_init': True,
        }
    return report
