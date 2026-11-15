"""Offline daily context with explicit episode and calendar discontinuities."""

from __future__ import annotations

import torch
from torch import Tensor, nn


def consecutive_day_mask(episode_ids: Tensor, day_ordinals: Tensor) -> Tensor:
    """Mark adjacent consecutive days in integer ``[D]`` or ``[B,D]`` metadata.

    Input order is authoritative. A gap, repeated/reversed day, or changed
    episode ID cuts the sequence. Equal IDs appearing later are never regrouped.
    IDs must distinguish independent samples and data splits at the caller.
    """
    integer_types = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
    if not isinstance(episode_ids, Tensor) or not isinstance(day_ordinals, Tensor):
        raise ValueError("episode_ids and day_ordinals must be integer tensors")
    if episode_ids.ndim not in (1, 2) or episode_ids.shape != day_ordinals.shape:
        raise ValueError("episode_ids and day_ordinals must share shape [D] or [B,D]")
    if episode_ids.numel() == 0:
        raise ValueError("episode_ids and day_ordinals must be nonempty")
    if episode_ids.dtype not in integer_types or day_ordinals.dtype not in integer_types:
        raise ValueError("episode_ids and day_ordinals must use integer dtypes, not bool/float")
    if episode_ids.device != day_ordinals.device:
        raise ValueError("episode_ids and day_ordinals must be on the same device")
    days = day_ordinals.to(dtype=torch.int64)
    previous, following = days[..., :-1], days[..., 1:]
    # The inequality also excludes int64 wraparound at the largest ordinal.
    return ((episode_ids[..., 1:] == episode_ids[..., :-1])
            & (following > previous) & (following == previous + 1))


def contiguous_day_segments(
    episode_ids: Tensor, day_ordinals: Tensor,
) -> tuple[tuple[int, int], ...]:
    """Return half-open ``(start, stop)`` segments for one ordered daily batch."""
    connected = consecutive_day_mask(episode_ids, day_ordinals)
    if episode_ids.ndim != 1:
        raise ValueError("contiguous_day_segments requires one-dimensional [D] metadata")
    cuts = (torch.nonzero(~connected, as_tuple=False).flatten() + 1).tolist()
    boundaries = [0, *cuts, episode_ids.shape[0]]
    return tuple(zip(boundaries[:-1], boundaries[1:]))


class OfflineTemporalHead(nn.Module):
    """Condition legal ``[D,N,H]`` graph states using a shared bidirectional GRU.

    ``independent`` uses a separate length-one sequence for each day;
    ``offline_bidirectional`` uses each contiguous episode segment. Both modes
    use endpoint-conditioned initial states, so even the length-one control's
    recurrent matrices participate. No hidden state is cached between calls.
    """

    MODES = ("independent", "offline_bidirectional")

    def __init__(
        self, hidden_dim: int, recurrent_dim: int | None = None,
        *, mode: str = "offline_bidirectional",
    ) -> None:
        super().__init__()
        if hidden_dim <= 0 or (recurrent_dim is not None and recurrent_dim <= 0):
            raise ValueError("temporal hidden dimensions must be positive")
        if mode not in self.MODES:
            raise ValueError(f"temporal mode must be one of {self.MODES}")
        recurrent_dim = max(1, hidden_dim // 2) if recurrent_dim is None else recurrent_dim
        self.hidden_dim = hidden_dim
        self.mode = mode
        self.forward_initial = nn.Linear(hidden_dim, recurrent_dim)
        self.backward_initial = nn.Linear(hidden_dim, recurrent_dim)
        self.gru = nn.GRU(hidden_dim, recurrent_dim, bidirectional=True)
        self.projection = nn.Linear(2 * recurrent_dim, hidden_dim)

    def forward(
        self, hidden: Tensor, *, episode_ids: Tensor | None, day_ordinals: Tensor | None,
    ) -> Tensor:
        """Apply only within explicitly continuous segments; never read raw profiles."""
        if hidden.ndim != 3 or hidden.shape[-1] != self.hidden_dim:
            raise ValueError("hidden must have shape [D,N,hidden_dim]")
        if min(hidden.shape[:2]) <= 0:
            raise ValueError("hidden must contain at least one day and one node")
        if episode_ids is None or day_ordinals is None:
            raise ValueError("temporal head requires explicit episode_ids and day_ordinals")
        segments = contiguous_day_segments(episode_ids, day_ordinals)
        if episode_ids.shape[0] != hidden.shape[0]:
            raise ValueError("episode_ids and day_ordinals must match hidden day count")
        if self.mode == "independent":
            segments = tuple((day, day + 1) for day in range(hidden.shape[0]))
        outputs = []
        for start, stop in segments:
            segment = hidden[start:stop]  # [D_segment,N,H]; each node is a GRU batch member.
            initial = torch.stack((
                torch.tanh(self.forward_initial(segment[0])),
                torch.tanh(self.backward_initial(segment[-1])),
            ))  # [2,N,H_recurrent], derived exclusively from this segment.
            context, _ = self.gru(segment, initial)
            outputs.append(segment + self.projection(context))
        return torch.cat(outputs, dim=0)
