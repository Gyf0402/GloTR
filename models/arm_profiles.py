"""Arm -> model dispatcher for the G80 phases.

Single place that maps an arm name to the model class it needs, so the trainer
does not have to know which experiment module owns which arm:

  * every pre-existing arm, and the A6 head variants -> ``head_variant_profile``
  * the occupancy output-transform arms            -> ``output_transform_profile``

Keeping the dispatch here means neither experiment module has to import the
other, and a phase can pin the SHA of its own module without freezing unrelated
ones.
"""
from __future__ import annotations

from torch import Tensor

from models.head_variant_profile import ARM_HEADS, build_profile as build_head
from models.head_variant_profile import initial_state as head_state
from models.output_transform_profile import (ARM_TRANSFORMS,
                                             build_transform_profile,
                                             transform_state)
from models.sparse_global_tokens import GlobalTokenProfile

#: Arms that change only the TRAINING OBJECTIVE, not the architecture. They
#: reuse the same frozen shared head, so they need no new module and no new
#: model SHA -- the treatment lives entirely in the trainer's loss.
#: Maps each loss-only arm to the pre-existing arm whose architecture it
#: reuses (the value is an ARM name, not a head kind).
LOSS_ONLY_ARMS = {'huber90': 'full'}


def known_arms() -> tuple[str, ...]:
    return tuple(ARM_HEADS) + tuple(ARM_TRANSFORMS) + tuple(LOSS_ONLY_ARMS)


def build_profile(arm: str, *, dropout: float = .1) -> GlobalTokenProfile:
    """Construct the model class an arm needs."""
    if arm in ARM_TRANSFORMS:
        return build_transform_profile(arm, dropout=dropout)
    if arm in LOSS_ONLY_ARMS:
        return build_head(LOSS_ONLY_ARMS[arm], dropout=dropout)
    if arm in ARM_HEADS:
        return build_head(arm, dropout=dropout)
    raise ValueError(f'unknown arm {arm!r}; {sorted(known_arms())}')


def initial_state(arm: str, shared_state: dict[str, Tensor]) -> dict[str, Tensor]:
    """Initial state for ``arm``, derived from a shared-head checkpoint.

    For every pre-existing arm this is the identity, which is what keeps those
    arms bitwise unchanged by the experiment modules.
    """
    if arm in ARM_TRANSFORMS:
        return transform_state(arm, shared_state)
    if arm in LOSS_ONLY_ARMS:
        return head_state(LOSS_ONLY_ARMS[arm], shared_state)
    if arm in ARM_HEADS:
        return head_state(arm, shared_state)
    raise ValueError(f'unknown arm {arm!r}')
