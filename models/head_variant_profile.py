"""A6 head variants: q/o shared read-out versus per-channel read-out.

Preparation-only module. It never edits the frozen contract in
``models/sparse_global_tokens``; it subclasses ``GlobalTokenProfile`` so the
shared-head variant is the frozen model itself, and adds separate-head variants
that change nothing except the read-out head.

The frozen head is::

    hidden = LayerNorm(proj + attention(proj, keys))          # width 64
    correction = Linear(128, 2)(SiLU(Linear(64, 128)(hidden)))  # one trunk, 2 outputs

so q and o are two linear read-outs of the SAME 128-dim hidden feature. The
variants replace that with two independent branches::

    correction_c = Linear(W, 1)(SiLU(Linear(64, W)(hidden)))   # c in {q, o}

Variant kinds
-------------
``shared``         the frozen head, unchanged (the A6 control is the existing
                   ``full`` arm at the same seed, so no extra training).
``split``          W = 128 per channel. Initialised by exact transplant: both
                   branches receive the shared trunk and branch ``c`` receives
                   read-out row ``c``. At step 0 the split arm therefore
                   computes *exactly* the shared arm's function, channel by
                   channel, so any later difference is caused by decoupling
                   plus capacity, never by a different starting point.
``split_matched``  W = 64 per channel. Head parameters 8450 vs the shared
                   8578, i.e. total model parameters within 0.5 percent. Its
                   initialisation is a deterministic *slice* of the shared head
                   (branch ``c`` takes hidden units ``c*64..c*64+63`` and the
                   matching read-out slice), which is NOT functionally equal to
                   the shared head. This arm exists only to test whether any
                   ``split`` gain survives at equal parameter count; it is not
                   a tied-start comparison.
``wide_shared``    a WIDER shared head: ``Linear(64, 252) -> SiLU ->
                   Linear(252, 2)``, i.e. one trunk, two read-outs, exactly like
                   the frozen head but wide enough to match the split arm's head
                   size (16886 vs 16898 parameters, -0.07 percent; total model
                   36022 vs 36034, -0.03 percent). ``split`` and ``wide_shared``
                   are therefore the capacity-matched pair, and the difference
                   between them is sharing versus splitting, not parameter count.
                   Initialised by exact transplant as well: trunk rows 0..127
                   take the shared trunk, the read-out columns for units
                   128..251 start at ZERO, so at step 0 the arm computes exactly
                   the shared arm's function. Extra units contribute nothing at
                   step 0 but start receiving read-out gradient immediately.

Held fixed across arms: the G40 base checkpoint, the 210 fit episodes, the
40-epoch AdamW(1e-3, wd=1e-4) budget, batch 8, dropout .1, the four key
segments, the static and hop pathways, and ``weighted_day_loss`` (which is
already per-channel: ``channel_l1.mean()``), so nothing about the objective
changes when the head is split.

Deliberately not claimed
------------------------
Splitting the head cannot, by construction, change the frozen base profile
``mu + B_hat``. If the base compresses high-occupancy values, a per-channel
read-out is a candidate remedy but only a candidate; whether it works is
decided by running A6, not by this module.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from models.sparse_global_tokens import GlobalTokenProfile

HEAD_KINDS = ('shared', 'wide_shared', 'split', 'split_matched')
BRANCH_WIDTH = {'split': 128, 'split_matched': 64}
#: Trunk width of the wide shared head. 252 is the width whose head parameter
#: count (64*252 + 252 + 252*2 + 2 = 16886) best matches the split arm's 16898.
WIDE_SHARED_WIDTH = 252
#: Deterministic seed for the padding trunk rows of the wide shared head.
WIDE_INIT_SEED = 20260926

#: A6 arms that need a fresh training run, and the control they pair against.
A6_TRAIN_ARMS = ('a6_split', 'a6_wide_shared', 'a6_split_matched')
A6_CONTROL_ARM = 'full'

#: arm -> head kind. The pre-existing arms keep the frozen shared head.
ARM_HEADS = {
    'full': 'shared', 'no_attr': 'shared', 'no_hops': 'shared',
    'no_pool': 'shared', 'no_fourier': 'shared',
    'a6_shared': 'shared', 'a6_wide_shared': 'wide_shared',
    'a6_split': 'split', 'a6_split_matched': 'split_matched',
}


class HeadVariantProfile(GlobalTokenProfile):
    """``GlobalTokenProfile`` with a selectable q/o read-out head.

    ``kind='shared'`` is delegation to the parent class, so that arm is the
    frozen model bitwise. The other kinds delete the parent head and build
    independent per-channel branches; every other submodule is inherited
    untouched.
    """

    def __init__(self, *, dropout: float = .1, kind: str = 'shared'):
        super().__init__(dropout=dropout)
        if kind not in HEAD_KINDS:
            raise ValueError(f'unknown head kind {kind!r}; {HEAD_KINDS}')
        self.kind = kind
        if kind == 'shared':
            return
        if kind == 'wide_shared':
            feature = self.query_projection.out_features
            self.output = nn.Sequential(
                nn.Linear(feature, WIDE_SHARED_WIDTH), nn.SiLU(),
                nn.Linear(WIDE_SHARED_WIDTH, 2))
            # Same contract as the frozen head: the final read-out starts at
            # zero, so a freshly built wide arm also predicts base + 0.
            nn.init.zeros_(self.output[-1].weight)
            nn.init.zeros_(self.output[-1].bias)
            return
        width = BRANCH_WIDTH[kind]
        feature = self.query_projection.out_features
        del self.output  # the frozen shared head is unused by variant arms
        self.trunks = nn.ModuleList([
            nn.Sequential(nn.Linear(feature, width), nn.SiLU())
            for _ in range(2)])
        self.readouts = nn.ModuleList([nn.Linear(width, 1) for _ in range(2)])
        # Mirror the frozen contract: the final read-out starts at zero, so a
        # freshly built variant arm predicts base + 0 exactly like a freshly
        # built shared arm.
        for readout in self.readouts:
            nn.init.zeros_(readout.weight)
            nn.init.zeros_(readout.bias)

    # -- shared-path plumbing ------------------------------------------------
    def _check(self, query: Tensor, keys: Tensor, valid: Tensor,
               base: Tensor) -> None:
        """Mirror of the frozen argument checks (tests assert the parity)."""
        if (query.ndim != 2 or query.shape[-1] != 14 or keys.ndim != 3 or
                keys.shape[0] != len(query) or keys.shape[-1] != 21 or
                valid.shape != keys.shape[:2] or valid.dtype != torch.bool or
                base.shape != (len(query), 2) or
                not bool(valid.any(dim=1).all())):
            raise ValueError('matching query, key, mask, and base profiles '
                             'required')

    def _encode(self, query: Tensor, keys: Tensor, valid: Tensor) -> Tensor:
        """Mirror of the frozen body; identical to the parent's hidden."""
        projected = self.query_projection(query).unsqueeze(1)
        encoded = self.key_projection(keys)
        attended, _ = self.attention(projected, encoded, encoded,
                                     key_padding_mask=~valid, need_weights=False)
        return self.normalization(projected + attended).squeeze(1)

    def forward(self, query: Tensor, keys: Tensor, valid: Tensor,
                base: Tensor) -> tuple[Tensor, Tensor]:
        if self.kind in ('shared', 'wide_shared'):
            # the wide arm keeps the frozen single-trunk path; only self.output
            # differs, and the parent forward reads the width from it
            return super().forward(query, keys, valid, base)
        self._check(query, keys, valid, base)
        hidden = self._encode(query, keys, valid)
        correction = torch.cat(
            [readout(trunk(hidden))
             for trunk, readout in zip(self.trunks, self.readouts)], dim=-1)
        return base + correction, correction


def build_profile(arm: str, *, dropout: float = .1) -> GlobalTokenProfile:
    """Construct the model class an arm needs (integration point for A6)."""
    if arm not in ARM_HEADS:
        raise ValueError(f'unknown arm {arm!r}; {sorted(ARM_HEADS)}')
    return HeadVariantProfile(dropout=dropout, kind=ARM_HEADS[arm])


def tied_state(shared_state: dict[str, Tensor], kind: str,
               *, matched_init: str = 'slice') -> dict[str, Tensor]:
    """Initial state for a variant arm, derived deterministically.

    ``split`` gets an exact transplant: the output equals the shared head's
    output, channel by channel. ``split_matched`` cannot be transplanted (the
    branch hidden width differs), so it receives a deterministic slice of the
    shared head, or zeros when ``matched_init='zero'``.
    """
    if kind == 'shared':
        return dict(shared_state)
    if kind not in HEAD_KINDS:
        raise ValueError(f'unknown variant kind {kind!r}')
    if kind == 'wide_shared':
        return _wide_state(shared_state)
    if matched_init not in ('slice', 'zero'):
        raise ValueError(f'unknown matched_init {matched_init!r}')
    width = BRANCH_WIDTH[kind]
    trunk_weight = shared_state['output.0.weight']
    trunk_bias = shared_state['output.0.bias']
    readout_weight = shared_state['output.2.weight']
    readout_bias = shared_state['output.2.bias']
    if kind == 'split' and trunk_weight.shape[0] != width:
        raise ValueError('shared trunk width differs from the branch width')
    if kind == 'split_matched' and trunk_weight.shape[0] != 2 * width:
        raise ValueError('shared trunk does not split into two branch slices')
    state = {k: v for k, v in shared_state.items()
             if not k.startswith('output.')}
    for channel in range(2):
        if kind == 'split':
            trunk_slice = slice(None)
            read_slice = readout_weight[channel:channel + 1]
        elif matched_init == 'slice':
            lo = channel * width
            trunk_slice = slice(lo, lo + width)
            read_slice = readout_weight[channel, lo:lo + width].reshape(1, width)
        else:
            trunk_slice = slice(channel * width, (channel + 1) * width)
            read_slice = torch.zeros((1, width), dtype=readout_weight.dtype)
        state[f'trunks.{channel}.0.weight'] = trunk_weight[trunk_slice].clone()
        state[f'trunks.{channel}.0.bias'] = trunk_bias[trunk_slice].clone()
        state[f'readouts.{channel}.weight'] = read_slice.clone()
        bias = readout_bias[channel:channel + 1]
        state[f'readouts.{channel}.bias'] = (
            bias.clone() if kind == 'split' or matched_init == 'slice'
            else torch.zeros_like(bias))
    return state


def _wide_state(shared_state: dict[str, Tensor]) -> dict[str, Tensor]:
    """Widen the shared trunk, pin the extra read-out columns to zero.

    The padding units must NOT be dead. Their read-out columns start at exactly
    zero (so the step-0 function equals the shared head, since those columns
    cannot influence any output), and their *trunk* rows must be non-zero: if
    the trunk rows were zero the unit activation would be a constant zero, the
    read-out gradient would be ``grad * activation = 0``, and the added capacity
    could never switch on. So the padding rows get a deterministic draw from the
    same uniform range ``nn.Linear`` uses (``1/sqrt(fan_in)``), seeded by
    ``WIDE_INIT_SEED`` so the derivation stays reproducible.
    """
    trunk_weight = shared_state['output.0.weight']
    trunk_bias = shared_state['output.0.bias']
    readout_weight = shared_state['output.2.weight']
    readout_bias = shared_state['output.2.bias']
    base_width = trunk_weight.shape[0]
    if base_width >= WIDE_SHARED_WIDTH:
        raise ValueError('shared trunk is not narrower than the wide trunk')
    in_features = trunk_weight.shape[1]
    extra = WIDE_SHARED_WIDTH - base_width
    generator = torch.Generator().manual_seed(WIDE_INIT_SEED)
    bound = 1.0 / (in_features ** 0.5)
    extra_weight = ((torch.rand((extra, in_features), generator=generator) * 2 - 1)
                    * bound).to(trunk_weight.dtype)
    extra_bias = ((torch.rand(extra, generator=generator) * 2 - 1)
                  * bound).to(trunk_bias.dtype)
    state = {k: v for k, v in shared_state.items() if not k.startswith('output.')}
    wide_trunk = torch.empty((WIDE_SHARED_WIDTH, in_features),
                             dtype=trunk_weight.dtype)
    wide_trunk[:base_width] = trunk_weight
    wide_trunk[base_width:] = extra_weight
    wide_bias = torch.empty(WIDE_SHARED_WIDTH, dtype=trunk_bias.dtype)
    wide_bias[:base_width] = trunk_bias
    wide_bias[base_width:] = extra_bias
    wide_readout = torch.zeros((readout_weight.shape[0], WIDE_SHARED_WIDTH),
                               dtype=readout_weight.dtype)
    wide_readout[:, :base_width] = readout_weight
    state['output.0.weight'] = wide_trunk
    state['output.0.bias'] = wide_bias
    state['output.2.weight'] = wide_readout
    state['output.2.bias'] = readout_bias.clone()
    return state


def initial_state(arm: str, shared_state: dict[str, Tensor]) -> dict[str, Tensor]:
    """State to initialise ``arm`` from a G40-style shared-head checkpoint."""
    if arm not in ARM_HEADS:
        raise ValueError(f'unknown arm {arm!r}')
    return tied_state(shared_state, ARM_HEADS[arm])


def head_parameter_report() -> dict[str, dict[str, float]]:
    """Parameter counts per kind, to make the capacity delta explicit."""
    report = {}
    for kind in HEAD_KINDS:
        model = HeadVariantProfile(dropout=.0, kind=kind)
        total = sum(p.numel() for p in model.parameters())
        if kind in ('shared', 'wide_shared'):
            head = sum(p.numel() for p in model.output.parameters())
        else:
            head = (sum(p.numel() for p in model.trunks.parameters()) +
                    sum(p.numel() for p in model.readouts.parameters()))
        report[kind] = {'total': total, 'head': head, 'body': total - head}
    shared_total = report['shared']['total']
    for kind in HEAD_KINDS:
        report[kind]['total_ratio_vs_shared'] = (
            round(report[kind]['total'] / shared_total, 6))
    return report
