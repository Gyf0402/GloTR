"""MAD-normalized joint objective with a staged local-physics schedule."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn


class JointOptimizationLoss(nn.Module):
    """Combine eleven objectives; pinball is a data term outside the physics ramp."""

    mad_scales: Tensor
    loss_weights: Tensor

    LOSS_NAMES = ("imp", "obs", "coef", "basis", "fd", "cons", "cap", "theta", "r", "s", "pinball")
    PHYSICS_NAMES = frozenset(("fd", "cons", "cap", "theta", "s"))

    def __init__(
        self,
        mad_scales: Mapping[str, float],
        weights: Mapping[str, float] | None = None,
        epsilon: float = 1e-8,
    ) -> None:
        super().__init__()
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        self._validate_keys(mad_scales, "mad_scales")
        resolved_weights = {name: 1.0 for name in self.LOSS_NAMES}
        if weights is not None:
            self._validate_keys(weights, "weights")
            resolved_weights.update(weights)
        scales = torch.tensor([float(mad_scales[name]) for name in self.LOSS_NAMES])  # [11]
        loss_weights = torch.tensor(
            [float(resolved_weights[name]) for name in self.LOSS_NAMES]
        )  # [11]
        if torch.any(scales <= 0) or not torch.isfinite(scales).all():
            raise ValueError("every training MAD scale must be finite and positive")
        if torch.any(loss_weights < 0) or not torch.isfinite(loss_weights).all():
            raise ValueError("loss weights must be finite and nonnegative")
        self.epsilon = epsilon
        self.register_buffer("mad_scales", scales)  # [11]
        self.register_buffer("loss_weights", loss_weights)  # [11]

    @classmethod
    def _validate_keys(cls, values: Mapping[str, object], label: str) -> None:
        supplied = set(values)
        expected = set(cls.LOSS_NAMES)
        if supplied != expected:
            raise ValueError(f"{label} keys must be exactly {cls.LOSS_NAMES}")

    @staticmethod
    def physics_weight(epoch: int) -> float:
        """10-epoch zero-weight warm-up followed by a 20-epoch linear ramp."""
        if epoch < 0:
            raise ValueError("epoch must be nonnegative")
        if epoch <= 10:
            return 0.0
        if epoch <= 30:
            return (epoch - 10) / 20.0
        return 1.0

    def forward(
        self,
        losses: Mapping[str, Tensor],
        epoch: int,
        return_details: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        """Return a scalar total and, optionally, its normalized contributions."""
        self._validate_keys(losses, "losses")
        reference = losses[self.LOSS_NAMES[0]]  # scalar
        if reference.ndim != 0:
            raise ValueError("all component losses must be scalar tensors")
        schedule = self.physics_weight(epoch)
        contributions: list[Tensor] = []
        details: dict[str, Tensor] = {}
        for index, name in enumerate(self.LOSS_NAMES):
            raw_loss = losses[name]
            if raw_loss.ndim != 0:
                raise ValueError(f"loss {name!r} must be a scalar tensor")
            if not torch.isfinite(raw_loss):
                raise ValueError(f"loss {name!r} must be finite")
            scale = self.mad_scales[index].to(device=raw_loss.device, dtype=raw_loss.dtype)  # scalar
            weight = self.loss_weights[index].to(device=raw_loss.device, dtype=raw_loss.dtype)  # scalar
            if name in self.PHYSICS_NAMES:
                weight = weight * schedule  # scalar
            normalized = raw_loss / scale.clamp_min(self.epsilon)  # scalar
            contribution = weight * normalized  # scalar
            contributions.append(contribution)
            details[f"normalized_{name}"] = normalized
            details[f"weighted_{name}"] = contribution
        total = torch.stack(contributions).sum()  # scalar
        details["physics_weight"] = total.new_tensor(schedule)  # scalar
        details["total"] = total
        if return_details:
            return total, details
        return total
