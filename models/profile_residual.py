"""Small profile residual head for an isolated q/o student candidate."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ProfileResidualHead(nn.Module):
    def __init__(self, *, rank: int, feature_dim: int, static_dim: int,
                 calendar_dim: int, hidden_dim: int, slots: int, channels: int) -> None:
        super().__init__()
        if min(rank, feature_dim, static_dim, calendar_dim, hidden_dim, slots, channels) <= 0:
            raise ValueError("positive dimensions required")
        self.rank = rank
        self.feature_dim = feature_dim
        self.static_dim = static_dim
        self.calendar_dim = calendar_dim
        self.slots = slots
        self.channels = channels
        self.network = nn.Sequential(
            nn.Linear(rank + feature_dim + static_dim + calendar_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, slots * channels),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, coefficients: Tensor, features: Tensor,
                static: Tensor, calendar: Tensor) -> Tensor:
        n = coefficients.shape[0]
        if (coefficients.shape != (n, self.rank) or
                features.shape != (n, self.feature_dim) or
                static.shape != (n, self.static_dim) or
                calendar.shape != (n, self.calendar_dim)):
            raise ValueError("profile head inputs must share profile count and registered dimensions")
        joined = torch.cat((coefficients, features, static, calendar), dim=-1)
        return self.network(joined).reshape(n, self.slots, self.channels)


def scored_profile_l1(prediction: Tensor, reference: Tensor, mask: Tensor) -> Tensor:
    if (prediction.ndim != 3 or prediction.shape != reference.shape or
            mask.shape != prediction.shape or mask.dtype != torch.bool):
        raise ValueError("matching [profile,slot,channel] tensors and boolean mask required")
    count = mask.sum(dim=1)
    if not torch.all(count > 0):
        raise ValueError("every profile/channel needs scored support")
    absolute = torch.where(mask, torch.abs(prediction - reference), 0.)
    return (absolute.sum(dim=1) / count).mean()


def balanced_episode_loss(prediction: Tensor, reference: Tensor, mask: Tensor,
                          episode_slices: tuple[tuple[int, int], ...]) -> Tensor:
    if not episode_slices or episode_slices[0][0] != 0 or episode_slices[-1][1] != len(prediction):
        raise ValueError("episode slices must cover all profiles")
    if any(start >= end or (i and start != episode_slices[i - 1][1])
           for i, (start, end) in enumerate(episode_slices)):
        raise ValueError("episode slices must be nonempty and contiguous")
    return torch.stack([
        scored_profile_l1(prediction[start:end], reference[start:end], mask[start:end])
        for start, end in episode_slices
    ]).mean()
