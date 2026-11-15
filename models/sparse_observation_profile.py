"""Sparse valid-observation cross-attention for q/o outage profiles."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def sparse_context(*, values: Tensor, visible: Tensor, base: Tensor,
                   static: Tensor, calendar: Tensor, target: int, day: int,
                   signed_hops: Tensor, scales: tuple[float, float],
                   raw_values: bool) -> tuple[Tensor, Tensor, Tensor]:
    """Build one target day's fixed sparse key set from zero-filled observations.

    No reference, scoring mask, or teacher is accepted. Key order is all other
    stations at the target day, then six target-station buffer days; each uses
    the aligned slot and three slots to either side.
    """
    if (values.ndim != 4 or values.shape != visible.shape or
            values.shape[-2:] != (288, 2) or visible.dtype != torch.bool or
            base.shape != (288, 2) or static.shape != (values.shape[1], 5) or
            calendar.shape != (values.shape[0], 4) or
            signed_hops.shape != (values.shape[1],) or
            len(scales) != 2 or min(scales) <= 0 or
            values.device != visible.device or base.device != values.device or
            static.device != values.device or calendar.device != values.device or
            signed_hops.device != values.device):
        raise ValueError('registered q/o episode tensors required')
    days, nodes = values.shape[:2]
    if (days - 6 not in (7, 14, 28) or not 0 <= target < nodes or
            not 3 <= day < days - 3 or not torch.isfinite(values).all() or
            not torch.isfinite(base).all() or not torch.isfinite(signed_hops).all() or
            torch.any(values[~visible] != 0)):
        raise ValueError('legal zero-filled outage inputs required')
    if bool(visible[3:days - 3, target].any()):
        raise ValueError('target outage must be hidden')
    device = values.device
    other = [node for node in range(nodes) if node != target]
    buffer_days = [0, 1, 2, days - 3, days - 2, days - 1]
    key_nodes = torch.tensor(other + [target] * 6, device=device).repeat_interleave(7)
    key_days = torch.tensor([day] * len(other) + buffer_days, device=device).repeat_interleave(7)
    offset = torch.arange(-3, 4, device=device).repeat(len(other) + 6)
    slots = torch.arange(288, device=device)[:, None] + offset[None, :]
    in_day = (slots >= 0) & (slots < 288)
    slots = slots.clamp(0, 287)
    key_visible = visible[key_days[None, :], key_nodes[None, :], slots] & in_day[..., None]
    valid = key_visible.any(dim=-1)
    if not bool(valid.any(dim=-1).all()):
        raise ValueError('at least one legal key is required per target slot')
    scale = values.new_tensor(scales)
    observed = torch.where(key_visible,
                           values[key_days[None, :], key_nodes[None, :], slots],
                           0.) / scale
    if not raw_values:
        observed = torch.zeros_like(observed)
    count = key_nodes.numel()
    key_features = torch.cat((
        observed,
        key_visible.to(values.dtype),
        static[key_nodes][None].expand(288, -1, -1),
        calendar[key_days][None].expand(288, -1, -1),
        ((key_days - day) / 20.)[None, :, None].expand(288, -1, -1),
        (offset / 3.)[None, :, None].expand(288, -1, -1),
        (signed_hops[key_nodes] / 19.)[None, :, None].expand(288, -1, -1),
        (key_nodes == target).to(values.dtype)[None, :, None].expand(288, -1, -1),
    ), dim=-1)
    if key_features.shape != (288, count, 17):
        raise RuntimeError('sparse key feature width changed')
    phase = torch.arange(288, device=device, dtype=values.dtype) * (2 * math.pi / 288)
    query = torch.cat((
        base,
        static[target][None].expand(288, -1),
        calendar[day][None].expand(288, -1),
        values.new_full((288, 1), (day - 3) / (days - 6)),
        phase.sin()[:, None], phase.cos()[:, None],
    ), dim=-1)
    if query.shape != (288, 14):
        raise RuntimeError('sparse query feature width changed')
    return query, key_features, valid


class SparseObservationProfile(nn.Module):
    """One 2-head cross-attention block and a zero-initialized q/o correction."""

    def __init__(self, *, dropout: float = .1):
        super().__init__()
        self.query_projection = nn.Linear(14, 64)
        self.key_projection = nn.Linear(17, 64)
        self.attention = nn.MultiheadAttention(64, 2, dropout=dropout, batch_first=True)
        self.normalization = nn.LayerNorm(64)
        self.output = nn.Sequential(nn.Linear(64, 128), nn.SiLU(), nn.Linear(128, 2))
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(self, query: Tensor, keys: Tensor, valid: Tensor,
                base: Tensor) -> tuple[Tensor, Tensor]:
        if (query.ndim != 2 or query.shape[-1] != 14 or keys.ndim != 3 or
                keys.shape[0] != len(query) or keys.shape[-1] != 17 or
                valid.shape != keys.shape[:2] or valid.dtype != torch.bool or
                base.shape != (len(query), 2) or not bool(valid.any(dim=1).all())):
            raise ValueError('matching query, key, mask, and base profiles required')
        projected = self.query_projection(query).unsqueeze(1)
        encoded_keys = self.key_projection(keys)
        attended, _ = self.attention(projected, encoded_keys, encoded_keys,
                                     key_padding_mask=~valid, need_weights=False)
        hidden = self.normalization(projected + attended).squeeze(1)
        correction = self.output(hidden)
        return base + correction, correction
