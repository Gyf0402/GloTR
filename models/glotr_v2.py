"""GloTR-v2: gated same-day spatial correction on top of the frozen G80 output.

Design (see reviews/2026-11-09/GLOTR_V2_PLAN_zh.md):

  x_final = x_profile + g * delta

* ``x_profile``  frozen G80 neural output (reference profile + global-token
  residual), unchanged.
* ``delta``      same-day spatial branch: every other station on the target day
  and the six target buffer days are encoded as full 288-slot sequences with
  dilated depthwise-separable temporal convolutions; the target stream then
  cross-attends, slot by slot, over those sequences for ``layers`` rounds,
  interleaved with temporal convolutions on the target stream itself.
* ``g``          per-slot, per-channel sigmoid gate fed with neighbour
  availability features, so the correction can be trusted when same-day
  neighbours are dense and suppressed when they are sparse.

``delta``'s last layer is zero-initialised, so an untrained head reproduces
G80 exactly. All inputs are built from visible entries only; the target outage
is hidden by construction (asserted in ``v2_day_inputs``).
"""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

SLOTS = 288
SEQ_META = 12        # static(5) + hop(1) + is_buffer(1) + day offset(1) + calendar(4)
AVAIL = 6            # neighbour vis frac (2) + near (|hop|<=2) vis frac (2) + buffer vis frac (2)
QUERY = 14


def _phase(device, dtype):
    phase = torch.arange(SLOTS, device=device, dtype=dtype) * (2 * math.pi / SLOTS)
    return torch.stack((phase.sin(), phase.cos()), dim=-1)          # (288, 2)


def v2_day_inputs(*, values: Tensor, visible: Tensor, base: Tensor, static: Tensor,
                  calendar: Tensor, target: int, day: int, signed_hops: Tensor,
                  scales) -> dict[str, Tensor]:
    """Inputs of one target day. ``values`` physical, zero-filled where hidden."""
    days, nodes = values.shape[:2]
    if bool(visible[3:days - 3, target].any()):
        raise ValueError('target outage must be hidden')
    scale = values.new_tensor(tuple(scales))
    others = [n for n in range(nodes) if n != target]
    buffer_days = [0, 1, 2, days - 3, days - 2, days - 1]

    nb_vis = visible[day, others]                                    # (N-1, 288, 2)
    nb_val = torch.where(nb_vis, values[day, others] / scale, 0.)
    bf_vis = visible[buffer_days, target]                            # (6, 288, 2)
    bf_val = torch.where(bf_vis, values[buffer_days, target] / scale, 0.)
    seq_vals = torch.cat((nb_val, bf_val))                           # (S, 288, 2)
    seq_vis = torch.cat((nb_vis, bf_vis)).to(values.dtype)

    n_nb, n_bf = len(others), len(buffer_days)
    meta = values.new_zeros((n_nb + n_bf, SEQ_META))
    meta[:n_nb, :5] = static[others]
    meta[n_nb:, :5] = static[target]
    meta[:n_nb, 5] = signed_hops[others] / 19.
    meta[n_nb:, 6] = 1.
    meta[n_nb:, 7] = (torch.tensor(buffer_days, device=values.device,
                                   dtype=values.dtype) - day) / 20.
    meta[:n_nb, 8:12] = calendar[day]
    meta[n_nb:, 8:12] = calendar[buffer_days]

    phase = _phase(values.device, values.dtype)
    query = torch.cat((
        base,
        static[target][None].expand(SLOTS, -1),
        calendar[day][None].expand(SLOTS, -1),
        values.new_full((SLOTS, 1), (day - 3) / (days - 6)),
        phase), dim=-1)                                              # (288, 14)

    near = (signed_hops[others].abs() <= 2)
    nb_frac = nb_vis.to(values.dtype).mean(dim=0)                    # (288, 2)
    near_frac = (nb_vis[near].to(values.dtype).mean(dim=0) if bool(near.any())
                 else torch.zeros_like(nb_frac))
    bf_frac = bf_vis.to(values.dtype).mean(dim=0)
    avail = torch.cat((nb_frac, near_frac, bf_frac), dim=-1)          # (288, 6)
    return {'seq_vals': seq_vals, 'seq_vis': seq_vis, 'seq_meta': meta,
            'query': query, 'avail': avail}


def stack_inputs(items: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    return {k: torch.stack([it[k] for it in items]) for k in items[0]}


class DSConv(nn.Module):
    """Depthwise-separable dilated temporal conv block on (B, L, d)."""

    def __init__(self, d: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        self.depth = nn.Conv1d(d, d, kernel, padding=dilation * (kernel - 1) // 2,
                               dilation=dilation, groups=d)
        self.point = nn.Linear(d, d)
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        y = self.depth(x.transpose(1, 2)).transpose(1, 2)
        y = self.point(F.gelu(y))
        return self.norm(x + self.drop(y))


class GloTRv2Head(nn.Module):
    def __init__(self, *, d: int = 128, heads: int = 4, layers: int = 3,
                 kernel: int = 9, dilations: tuple[int, ...] = (1, 2, 4),
                 dropout: float = .1, gate: bool = True, split_heads: bool = False):
        super().__init__()
        self.kernel, self.dilations, self.use_gate = kernel, tuple(dilations), gate
        self.seq_in = nn.Linear(4 + SEQ_META + 2, d)
        self.seq_conv = nn.ModuleList(DSConv(d, kernel, r, dropout) for r in dilations)
        self.q_in = nn.Linear(QUERY + 2 + AVAIL, d)
        self.null_key = nn.Parameter(torch.zeros(1, 1, d))
        self.attn = nn.ModuleList(nn.MultiheadAttention(d, heads, dropout=dropout,
                                                        batch_first=True)
                                  for _ in range(layers))
        self.attn_norm = nn.ModuleList(nn.LayerNorm(d) for _ in range(layers))
        self.t_conv = nn.ModuleList(DSConv(d, kernel, dilations[i % len(dilations)],
                                           dropout) for i in range(layers))
        self.ffn = nn.ModuleList(nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(),
                                               nn.Dropout(dropout), nn.Linear(2 * d, d))
                                 for _ in range(layers))
        self.ffn_norm = nn.ModuleList(nn.LayerNorm(d) for _ in range(layers))
        self.split_heads = split_heads
        if split_heads:          # separate q / o read-out so o is not dominated by q
            self.delta_q = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, 1))
            self.delta_o = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, 1))
            for head in (self.delta_q, self.delta_o):
                nn.init.zeros_(head[-1].weight)
                nn.init.zeros_(head[-1].bias)
        else:
            self.delta = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, 2))
            nn.init.zeros_(self.delta[-1].weight)
            nn.init.zeros_(self.delta[-1].bias)
        self.gate = nn.Linear(d + AVAIL, 2)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    @property
    def receptive_radius(self) -> int:
        return (self.kernel - 1) // 2 * sum(self.dilations)

    def forward(self, seq_vals: Tensor, seq_vis: Tensor, seq_meta: Tensor,
                query: Tensor, x_profile: Tensor, avail: Tensor
                ) -> tuple[Tensor, Tensor, Tensor]:
        """Shapes: seq_* (B,S,288,·), meta (B,S,12), query (B,288,14),
        x_profile (B,288,2), avail (B,288,6). Returns final, delta, gate."""
        b, s = seq_vals.shape[:2]
        phase = _phase(seq_vals.device, seq_vals.dtype)
        feats = torch.cat((seq_vals, seq_vis,
                           seq_meta[:, :, None].expand(-1, -1, SLOTS, -1),
                           phase[None, None].expand(b, s, -1, -1)), dim=-1)
        h_seq = self.seq_in(feats).reshape(b * s, SLOTS, -1)
        for block in self.seq_conv:
            h_seq = block(h_seq)
        d = h_seq.shape[-1]
        keys = h_seq.reshape(b, s, SLOTS, d).permute(0, 2, 1, 3).reshape(b * SLOTS, s, d)
        keys = torch.cat((self.null_key.expand(b * SLOTS, -1, -1), keys), dim=1)
        any_vis = seq_vis.amax(dim=-1).reshape(b * s, 1, SLOTS)
        radius = self.receptive_radius
        near_vis = F.max_pool1d(any_vis, 2 * radius + 1, stride=1, padding=radius)
        valid = near_vis.reshape(b, s, SLOTS).permute(0, 2, 1).reshape(b * SLOTS, s) > 0
        valid = torch.cat((valid.new_ones((b * SLOTS, 1)), valid), dim=1)

        h = self.q_in(torch.cat((query, x_profile, avail), dim=-1))   # (B,288,d)
        for attn, an, tc, ffn, fn in zip(self.attn, self.attn_norm, self.t_conv,
                                         self.ffn, self.ffn_norm):
            q = h.reshape(b * SLOTS, 1, d)
            out, _ = attn(q, keys, keys, key_padding_mask=~valid, need_weights=False)
            h = an(h + out.reshape(b, SLOTS, d))
            h = tc(h)
            h = fn(h + ffn(h))
        delta = (torch.cat((self.delta_q(h), self.delta_o(h)), dim=-1)
                 if self.split_heads else self.delta(h))
        if self.use_gate:
            g = torch.sigmoid(self.gate(torch.cat((h, avail), dim=-1)))
        else:
            g = torch.ones_like(delta)
        return x_profile + g * delta, delta, g
