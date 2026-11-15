"""FiLM-conditioned multi-relational directed graph coefficient predictor."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn
from torch_geometric.nn import RGCNConv

from models.temporal import OfflineTemporalHead


class FiLMLayer(nn.Module):
    """Feature-wise affine modulation from road semantics and calendar data."""

    def __init__(self, static_dim: int, calendar_dim: int, hidden_dim: int) -> None:
        super().__init__()
        if min(static_dim, calendar_dim, hidden_dim) <= 0:
            raise ValueError("FiLM dimensions must be positive")
        self.static_dim = static_dim
        self.calendar_dim = calendar_dim
        self.hidden_dim = hidden_dim
        self.conditioner = nn.Sequential(
            nn.Linear(static_dim + calendar_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim),
        )

    def forward(
        self,
        hidden: Tensor,
        static_features: Tensor,
        calendar_features: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Apply FiLM(h) = gamma * h + beta.

        Args:
            hidden: Node states [B, N, H].
            static_features: Static semantics [B, N, F].
            calendar_features: Calendar data [B, K] or [B, N, K].

        Returns:
            Tuple of modulated states, gamma, beta; each [B, N, H].
        """
        if hidden.ndim != 3 or hidden.shape[-1] != self.hidden_dim:
            raise ValueError("hidden must have shape [B, N, hidden_dim]")
        batch_size, node_count, _ = hidden.shape
        if static_features.shape != (batch_size, node_count, self.static_dim):
            raise ValueError("static_features must have shape [B, N, static_dim]")
        calendar = self._expand_calendar(calendar_features, batch_size, node_count)  # [B, N, K]
        conditioning = torch.cat((static_features, calendar), dim=-1)  # [B, N, F+K]
        raw_gamma, beta = self.conditioner(conditioning).chunk(2, dim=-1)  # each [B, N, H]
        gamma = 1.0 + torch.tanh(raw_gamma)  # [B, N, H]
        return gamma * hidden + beta, gamma, beta  # each [B, N, H]

    def _expand_calendar(self, calendar: Tensor, batch_size: int, node_count: int) -> Tensor:
        if calendar.ndim == 2 and calendar.shape == (batch_size, self.calendar_dim):
            return calendar[:, None, :].expand(-1, node_count, -1)  # [B, N, K]
        if calendar.ndim == 3 and calendar.shape == (
            batch_size,
            node_count,
            self.calendar_dim,
        ):
            return calendar  # [B, N, K]
        raise ValueError("calendar_features must have shape [B, K] or [B, N, K]")


class SemanticDirectedGNN(nn.Module):
    """Four-relation directed R-GCN with semantic FiLM after each aggregation."""

    RELATIONS = ("upstream", "downstream", "on_ramp", "off_ramp")

    def __init__(
        self,
        input_dim: int,
        static_dim: int,
        calendar_dim: int,
        hidden_dim: int,
        output_rank: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        temporal_mode: str = "none",
        temporal_hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        if min(input_dim, static_dim, calendar_dim, hidden_dim, output_rank, num_layers) <= 0:
            raise ValueError("all model dimensions must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if temporal_mode not in ("none", *OfflineTemporalHead.MODES):
            raise ValueError("temporal_mode must be none, independent, or offline_bidirectional")
        if temporal_mode == "none" and temporal_hidden_dim is not None:
            raise ValueError("temporal_hidden_dim requires an enabled temporal head")

        self.input_dim = input_dim
        self.static_dim = static_dim
        self.calendar_dim = calendar_dim
        self.hidden_dim = hidden_dim
        self.output_rank = output_rank
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.no_history_token = nn.Parameter(torch.zeros(input_dim))  # [D_in]
        self.convolutions = nn.ModuleList(
            RGCNConv(hidden_dim, hidden_dim, num_relations=len(self.RELATIONS))
            for _ in range(num_layers)
        )
        self.film_layers = nn.ModuleList(
            FiLMLayer(static_dim, calendar_dim, hidden_dim) for _ in range(num_layers)
        )
        self.normalizations = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(num_layers))
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.output_head = nn.Linear(hidden_dim, output_rank)
        self.temporal_mode = temporal_mode
        self.temporal_head = (
            None if temporal_mode == "none" else OfflineTemporalHead(
                hidden_dim, temporal_hidden_dim, mode=temporal_mode,
            )
        )

    def forward(
        self,
        node_features: Tensor,
        static_features: Tensor,
        calendar_features: Tensor,
        edge_sets: Mapping[str, Tensor],
        protocol: str = "K",
        target_mask: Tensor | None = None,
        *,
        episode_ids: Tensor | None = None,
        day_ordinals: Tensor | None = None,
    ) -> Tensor:
        """Predict daily profile coefficients.

        Args:
            node_features: Allowed dynamic/context features [B, N, D_in].
            static_features: Road semantics [B, N, F] or [N, F].
            calendar_features: Day variables [B, K] or [B, N, K].
            edge_sets: Four directed PyG edge_index tensors, each [2, E_r].
            protocol: "K" or "U".
            target_mask: Protocol-U held-out targets [N] or [B, N].
            episode_ids: Explicit integer [B] IDs for independent daily episodes.
            day_ordinals: Explicit integer [B] calendar day ordinals. Both metadata
                arrays are required for enabled temporal modes, where B means D.

        Returns:
            Predicted profile coefficients [B, N, R].
        """
        if protocol not in {"K", "U"}:
            raise ValueError("protocol must be 'K' or 'U'")
        if node_features.ndim != 3 or node_features.shape[-1] != self.input_dim:
            raise ValueError("node_features must have shape [B, N, input_dim]")
        batch_size, node_count, _ = node_features.shape
        static = self._expand_static(static_features, batch_size, node_count)  # [B, N, F]

        model_input = node_features  # [B, N, D_in]
        if protocol == "U":
            held_out = self._expand_target_mask(target_mask, batch_size, node_count)  # [B, N]
            shared = self.no_history_token.view(1, 1, -1).expand_as(model_input)  # [B, N, D_in]
            model_input = torch.where(held_out.unsqueeze(-1), shared, model_input)  # [B, N, D_in]

        edge_index, edge_type = self._batch_relations(
            edge_sets, batch_size, node_count, node_features.device
        )  # [2, E_total*B], [E_total*B]
        hidden = self.input_projection(model_input)  # [B, N, H]
        for convolution, film, normalization in zip(
            self.convolutions, self.film_layers, self.normalizations
        ):
            flat_hidden = hidden.reshape(batch_size * node_count, self.hidden_dim)  # [BN, H]
            messages = convolution(flat_hidden, edge_index, edge_type)  # [BN, H]
            messages = messages.reshape(batch_size, node_count, self.hidden_dim)  # [B, N, H]
            messages, _, _ = film(messages, static, calendar_features)  # [B, N, H]
            hidden = normalization(hidden + self.dropout(self.activation(messages)))  # [B, N, H]
        if self.temporal_head is not None:
            # Only the graph states after Protocol-U replacement enter this head.
            hidden = self.temporal_head(
                hidden, episode_ids=episode_ids, day_ordinals=day_ordinals,
            )
        prediction: Tensor = self.output_head(hidden)  # [B, N, R]
        return prediction

    def _expand_static(self, static: Tensor, batch_size: int, node_count: int) -> Tensor:
        if static.ndim == 2 and static.shape == (node_count, self.static_dim):
            return static.unsqueeze(0).expand(batch_size, -1, -1)  # [B, N, F]
        if static.ndim == 3 and static.shape == (batch_size, node_count, self.static_dim):
            return static  # [B, N, F]
        raise ValueError("static_features must have shape [N, F] or [B, N, F]")

    @staticmethod
    def _expand_target_mask(
        target_mask: Tensor | None, batch_size: int, node_count: int
    ) -> Tensor:
        if target_mask is None:
            raise ValueError("Protocol-U requires target_mask")
        if target_mask.ndim == 1 and target_mask.shape == (node_count,):
            return target_mask.bool().unsqueeze(0).expand(batch_size, -1)  # [B, N]
        if target_mask.ndim == 2 and target_mask.shape == (batch_size, node_count):
            return target_mask.bool()  # [B, N]
        raise ValueError("target_mask must have shape [N] or [B, N]")

    def _batch_relations(
        self,
        edge_sets: Mapping[str, Tensor],
        batch_size: int,
        node_count: int,
        device: torch.device,
    ) -> tuple[Tensor, Tensor]:
        missing = set(self.RELATIONS) - set(edge_sets)
        extra = set(edge_sets) - set(self.RELATIONS)
        if missing or extra:
            raise ValueError(f"edge_sets must contain exactly {self.RELATIONS}")

        batched_edges: list[Tensor] = []
        batched_types: list[Tensor] = []
        offsets = torch.arange(batch_size, device=device).view(-1, 1, 1) * node_count  # [B, 1, 1]
        for relation_id, relation in enumerate(self.RELATIONS):
            edge_index = edge_sets[relation].to(device=device, dtype=torch.long)  # [2, E_r]
            if edge_index.ndim != 2 or edge_index.shape[0] != 2:
                raise ValueError(f"edge set {relation!r} must have shape [2, E_r]")
            if edge_index.numel() and (
                edge_index.min().item() < 0 or edge_index.max().item() >= node_count
            ):
                raise ValueError(f"edge set {relation!r} contains an invalid node index")
            repeated = edge_index.unsqueeze(0) + offsets  # [B, 2, E_r]
            repeated = repeated.permute(1, 0, 2).reshape(2, -1)  # [2, B*E_r]
            batched_edges.append(repeated)
            batched_types.append(
                torch.full(
                    (repeated.shape[1],), relation_id, dtype=torch.long, device=device
                )
            )  # [B*E_r]
        return torch.cat(batched_edges, dim=1), torch.cat(batched_types, dim=0)  # [2, E], [E]
