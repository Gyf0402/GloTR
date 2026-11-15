"""Per-station cross-day tokens on top of the retained global q/o branch."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from models.sparse_global_tokens import global_token_context


def neighbor_context(*, values: Tensor, visible: Tensor, base: Tensor,
                     static: Tensor, calendar: Tensor, target: int, day: int,
                     signed_hops: Tensor,
                     scales: tuple[float, float]) -> tuple[Tensor, Tensor, Tensor]:
    """Append legal adjacent-day same-bin and current-day whole-profile keys."""
    query, old_keys, old_valid = global_token_context(
        values=values, visible=visible, base=base, static=static,
        calendar=calendar, target=target, day=day,
        signed_hops=signed_hops, scales=scales, raw_values=True)
    device, dtype = values.device, values.dtype
    other = torch.tensor([node for node in range(values.shape[1]) if node != target],
                         device=device)
    n = len(other)
    scale = values.new_tensor(scales)
    neighbor_days = torch.tensor([day - 1, day + 1], device=device)
    pooled_visible = visible[neighbor_days[:, None], other[None]].reshape(
        2, n, 18, 16, 2)
    pooled_values = (values[neighbor_days[:, None], other[None]] /
                     scale).reshape(2, n, 18, 16, 2)
    count = pooled_visible.sum(dim=3)
    mean = (pooled_values * pooled_visible).sum(dim=3) / count.clamp_min(1)
    variance = ((((pooled_values - mean[:, :, :, None]) ** 2) *
                 pooled_visible).sum(dim=3) / count.clamp_min(1))
    std = variance.sqrt()
    bin_index = torch.arange(288, device=device) // 16
    mean = mean[:, :, bin_index].permute(2, 0, 1, 3).reshape(288, 2 * n, 2)
    std = std[:, :, bin_index].permute(2, 0, 1, 3).reshape(288, 2 * n, 2)
    count = count[:, :, bin_index].permute(2, 0, 1, 3).reshape(288, 2 * n, 2)
    adjacent = values.new_zeros((288, 2 * n, 23))
    adjacent[..., :2] = mean
    adjacent[..., 2:4] = count / 16
    adjacent[..., 4:9] = static[other].repeat(2, 1)
    adjacent[..., 9:13] = calendar[neighbor_days].repeat_interleave(n, dim=0)
    adjacent[..., 13] = ((neighbor_days - day) / 20).repeat_interleave(n)
    centers = (bin_index.to(dtype) * 16 + 7.5)
    adjacent[..., 14] = ((centers - torch.arange(288, device=device,
                                                dtype=dtype)) / 288)[:, None]
    adjacent[..., 15] = (signed_hops[other] / 19).repeat(2)
    adjacent[..., 21] = 1
    adjacent[..., 17:19] = std
    adjacent_valid = count.any(dim=-1)

    current_visible = visible[day, other]
    current_values = values[day, other] / scale
    full_count = current_visible.sum(dim=1)
    full_mean = ((current_values * current_visible).sum(dim=1) /
                 full_count.clamp_min(1))
    full_std = (((((current_values - full_mean[:, None]) ** 2) *
                  current_visible).sum(dim=1) /
                 full_count.clamp_min(1)).sqrt())
    full = values.new_zeros((288, n, 23))
    full[..., :2] = full_mean
    full[..., 2:4] = full_count / 288
    full[..., 4:9] = static[other]
    full[..., 9:13] = calendar[day]
    full[..., 15] = signed_hops[other] / 19
    full[..., 17:19] = full_std
    full[..., 22] = 1
    full_valid = full_count.any(dim=-1)[None].expand(288, -1)

    old_extra = values.new_zeros((*old_keys.shape[:2], 2))
    keys = torch.cat((torch.cat((old_keys, old_extra), dim=-1), adjacent,
                      full), dim=1)
    valid = torch.cat((old_valid, adjacent_valid, full_valid), dim=1)
    return query, keys, valid


class NeighborContextProfile(nn.Module):
    """Inherited 2-head attention with two added key-type coordinates."""

    def __init__(self, *, dropout: float = .1):
        super().__init__()
        self.query_projection = nn.Linear(14, 64)
        self.key_projection = nn.Linear(23, 64)
        self.attention = nn.MultiheadAttention(64, 2, dropout=dropout,
                                               batch_first=True)
        self.normalization = nn.LayerNorm(64)
        self.output = nn.Sequential(nn.Linear(64, 128), nn.SiLU(), nn.Linear(128, 2))
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    @classmethod
    def from_global_state(cls, state: dict[str, Tensor], *, dropout: float = .1):
        model = cls(dropout=dropout)
        copied = model.state_dict()
        if set(state) != set(copied):
            raise ValueError('global checkpoint parameter names differ')
        for name, tensor in state.items():
            if name == 'key_projection.weight':
                if tensor.shape != (64, 21):
                    raise ValueError('global checkpoint key width differs')
                copied[name][:, :21] = tensor
                copied[name][:, 21:] = 0
            else:
                if copied[name].shape != tensor.shape:
                    raise ValueError('global checkpoint architecture differs')
                copied[name] = tensor
        model.load_state_dict(copied, strict=True)
        return model

    def forward(self, query: Tensor, keys: Tensor, valid: Tensor,
                base: Tensor) -> tuple[Tensor, Tensor]:
        if (query.ndim != 2 or query.shape[-1] != 14 or keys.ndim != 3 or
                keys.shape[0] != len(query) or keys.shape[-1] != 23 or
                valid.shape != keys.shape[:2] or valid.dtype != torch.bool or
                base.shape != (len(query), 2) or not bool(valid.any(dim=1).all())):
            raise ValueError('matching query, key, mask, and base profiles required')
        projected = self.query_projection(query).unsqueeze(1)
        encoded = self.key_projection(keys)
        attended, _ = self.attention(projected, encoded, encoded,
                                     key_padding_mask=~valid, need_weights=False)
        hidden = self.normalization(projected + attended).squeeze(1)
        correction = self.output(hidden)
        return base + correction, correction
