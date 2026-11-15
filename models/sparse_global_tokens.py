"""Fixed global context tokens for the Train-only sparse q/o candidate."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from models.sparse_observation_profile import sparse_context


KEY_GROUPS = ('local', 'pool', 'station', 'fourier')


def global_token_context(*, values: Tensor, visible: Tensor, base: Tensor,
                         static: Tensor, calendar: Tensor, target: int, day: int,
                         signed_hops: Tensor, scales: tuple[float, float],
                         raw_values: bool,
                         key_groups: tuple[str, ...] = KEY_GROUPS
                         ) -> tuple[Tensor, Tensor, Tensor]:
    """Append six full buffer days, same-slot station statistics, and Fourier.

    Each buffer day produces 18 masked means over fixed, nonoverlapping
    16-slot blocks. No target outage value, reference, or scoring mask enters.

    ``key_groups`` selects which key segments enter the attention context. The
    default reproduces the frozen 285-key contract exactly; ablations drop whole
    segments without perturbing any other code path, so paired arms differ only
    by which keys the residual head can attend to.
    """
    unknown = set(key_groups) - set(KEY_GROUPS)
    if unknown:
        raise ValueError(f'unknown key groups {sorted(unknown)}')
    if 'local' not in key_groups:
        raise ValueError('the local key segment is required by the contract')
    query, local_keys, local_valid = sparse_context(
        values=values, visible=visible, base=base, static=static,
        calendar=calendar, target=target, day=day,
        signed_hops=signed_hops, scales=scales, raw_values=raw_values)
    device, dtype = values.device, values.dtype
    scale = values.new_tensor(scales)
    days, nodes = values.shape[:2]
    slot_index = torch.arange(288, device=device, dtype=dtype)

    key_parts = []
    valid_parts = []
    local_extra = values.new_zeros((*local_keys.shape[:2], 4))
    key_parts.append(torch.cat((local_keys, local_extra), dim=-1))
    valid_parts.append(local_valid)

    if 'pool' in key_groups:
        buffer_days = torch.tensor([0, 1, 2, days - 3, days - 2, days - 1],
                                   device=device)
        buffer_visible = visible[buffer_days, target].reshape(6, 18, 16, 2)
        buffer_values = values[buffer_days, target].reshape(6, 18, 16, 2) / scale
        pool_count = buffer_visible.sum(dim=2)
        pool_mean = (buffer_values * buffer_visible).sum(dim=2) / pool_count.clamp_min(1)
        pool_variance = (((buffer_values - pool_mean[:, :, None]) ** 2) *
                         buffer_visible).sum(dim=2) / pool_count.clamp_min(1)
        pool_std = pool_variance.sqrt()
        if not raw_values:
            pool_mean = torch.zeros_like(pool_mean)
            pool_std = torch.zeros_like(pool_std)
        pool_features = values.new_zeros((6, 18, 21))
        pool_features[..., :2] = pool_mean
        pool_features[..., 2:4] = pool_count / 16
        pool_features[..., 4:9] = static[target]
        pool_features[..., 9:13] = calendar[buffer_days, None]
        pool_features[..., 13] = (buffer_days[:, None] - day) / 20
        pool_features[..., 16] = 1
        pool_features[..., 17:19] = pool_std
        pool_features[..., 19] = 1
        pool_features = pool_features.reshape(108, 21)
        centers = torch.arange(18, device=device, dtype=dtype) * 16 + 7.5
        pool_keys = pool_features[None].expand(288, -1, -1).clone()
        pool_keys[..., 14] = ((centers[None].expand(6, -1).reshape(108)[None] -
                               slot_index[:, None]) / 288)
        pool_valid = pool_count.any(dim=-1).reshape(108)[None].expand(288, -1)
        key_parts.append(pool_keys)
        valid_parts.append(pool_valid)

    if 'station' in key_groups:
        others = [node for node in range(nodes) if node != target]
        station_visible = visible[day, others].permute(1, 0, 2)
        station_values = values[day, others].permute(1, 0, 2) / scale
        station_count = station_visible.sum(dim=1)
        station_mean = ((station_values * station_visible).sum(dim=1) /
                        station_count.clamp_min(1))
        station_variance = ((((station_values - station_mean[:, None]) ** 2) *
                             station_visible).sum(dim=1) /
                            station_count.clamp_min(1))
        station_std = station_variance.sqrt()
        if not raw_values:
            station_mean = torch.zeros_like(station_mean)
            station_std = torch.zeros_like(station_std)
        station_key = values.new_zeros((288, 1, 21))
        station_key[:, 0, :2] = station_mean
        station_key[:, 0, 2:4] = station_count / len(others)
        station_key[:, 0, 4:9] = static[others].mean(dim=0)
        station_key[:, 0, 9:13] = calendar[day]
        station_key[:, 0, 17:19] = station_std
        station_key[:, 0, 20] = 1
        station_valid = station_count.any(dim=-1, keepdim=True)
        key_parts.append(station_key)
        valid_parts.append(station_valid)

    if 'fourier' in key_groups:
        phase = slot_index * (2 * math.pi / 288)
        fourier_key = values.new_zeros((288, 1, 21))
        fourier_key[:, 0, 9:13] = calendar[day]
        fourier_key[:, 0, 13] = phase.sin()
        fourier_key[:, 0, 14] = phase.cos()
        fourier_key[:, 0, 15] = (2 * phase).sin()
        fourier_key[:, 0, 16] = (2 * phase).cos()
        fourier_key[:, 0, 19:21] = 1
        fourier_valid = torch.ones((288, 1), device=device, dtype=torch.bool)
        key_parts.append(fourier_key)
        valid_parts.append(fourier_valid)

    keys = torch.cat(key_parts, dim=1)
    valid = torch.cat(valid_parts, dim=1)
    return query, keys, valid


class GlobalTokenProfile(nn.Module):
    """The original attention head with four added key feature coordinates."""

    def __init__(self, *, dropout: float = .1):
        super().__init__()
        self.query_projection = nn.Linear(14, 64)
        self.key_projection = nn.Linear(21, 64)
        self.attention = nn.MultiheadAttention(64, 2, dropout=dropout,
                                               batch_first=True)
        self.normalization = nn.LayerNorm(64)
        self.output = nn.Sequential(nn.Linear(64, 128), nn.SiLU(), nn.Linear(128, 2))
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    @classmethod
    def from_sparse_state(cls, state: dict[str, Tensor], *, dropout: float = .1):
        model = cls(dropout=dropout)
        new_state = model.state_dict()
        for name, tensor in state.items():
            if name == 'key_projection.weight':
                if tensor.shape != (64, 17):
                    raise ValueError('sparse checkpoint key width differs')
                new_state[name][:, :17] = tensor
                new_state[name][:, 17:] = 0
            else:
                if name not in new_state or new_state[name].shape != tensor.shape:
                    raise ValueError('sparse checkpoint architecture differs')
                new_state[name] = tensor
        model.load_state_dict(new_state, strict=True)
        return model

    def forward(self, query: Tensor, keys: Tensor, valid: Tensor,
                base: Tensor) -> tuple[Tensor, Tensor]:
        if (query.ndim != 2 or query.shape[-1] != 14 or keys.ndim != 3 or
                keys.shape[0] != len(query) or keys.shape[-1] != 21 or
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
