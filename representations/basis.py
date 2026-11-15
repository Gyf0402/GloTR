"""Orthogonal daily-profile basis and protocol-aware conditional mean."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn


class GlobalProfileBasis(nn.Module):
    """Shared orthogonal basis over flattened multivariate daily profiles.

    The trainable matrix is retracted to the Stiefel manifold with a reduced QR
    decomposition on every use. Profile fitting must use training data only.
    """

    def __init__(
        self,
        time_steps: int = 288,
        channels: int = 3,
        rank: int = 16,
        ridge: float = 1e-6,
        *,
        wls_chunk_size: int | None = 256,
        wls_compute_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        feature_count = time_steps * channels
        if not 0 < rank <= feature_count:
            raise ValueError("rank must be in [1, time_steps * channels]")
        if ridge <= 0:
            raise ValueError("ridge must be positive")
        if wls_chunk_size is not None and (type(wls_chunk_size) is not int or wls_chunk_size < 1):
            raise ValueError("wls chunk size must be a positive integer or None")
        if wls_compute_dtype not in (None, torch.float32, torch.float64):
            raise ValueError("WLS compute dtype must be float32 or float64")

        self.time_steps = time_steps
        self.channels = channels
        self.rank = rank
        self.ridge = ridge
        self.wls_chunk_size = wls_chunk_size
        self.wls_compute_dtype = wls_compute_dtype

        initial = torch.randn(feature_count, rank)  # [TC, R]
        initial, _ = torch.linalg.qr(initial, mode="reduced")  # [TC, R]
        self.raw_basis = nn.Parameter(initial)  # [TC, R]
        self.global_mean = nn.Parameter(torch.zeros(time_steps, channels))  # [T, C]

    def orthonormal_basis(self) -> Tensor:
        """Return B with a deterministic QR sign convention."""
        basis: Tensor
        upper: Tensor
        basis, upper = torch.linalg.qr(self.raw_basis, mode="reduced")  # [TC, R], [R, R]
        diagonal = torch.diagonal(upper)  # [R]
        signs = torch.where(diagonal >= 0, 1.0, -1.0).to(basis)  # [R]
        return basis * signs.unsqueeze(0)  # [TC, R]

    def project(
        self,
        x_obs: Tensor,
        mask: Tensor,
        mean: Optional[Tensor] = None,
    ) -> Tensor:
        """Fit teacher coefficients by batched masked ridge least squares.

        Args:
            x_obs: Observed profiles [..., T, C].
            mask: Observation indicators [..., T, C].
            mean: Optional conditional means broadcastable to [..., T, C].

        Returns:
            Teacher coefficients [..., R], always in the input profile dtype.
            QR stays in the parameter dtype. The optional float64 audit promotes
            the Gram/RHS/solve only. ``wls_chunk_size=None`` retains the historical
            expanded kernel for archived-run replay and numerical comparison.
        """
        if x_obs.shape[-2:] != (self.time_steps, self.channels):
            raise ValueError("x_obs must end with [time_steps, channels]")
        if mask.shape != x_obs.shape:
            raise ValueError("mask and x_obs must have identical shapes")

        profile_shape = x_obs.shape[:-2]
        flat_profiles = x_obs.reshape(-1, self.time_steps * self.channels)  # [P, TC]
        flat_mask = mask.to(dtype=x_obs.dtype).reshape_as(flat_profiles)  # [P, TC]
        if not torch.isfinite(flat_profiles[flat_mask.bool()]).all():
            raise ValueError("observed profile entries must be finite")
        observed_counts = flat_mask.sum(dim=-1)  # [P]
        if torch.any(observed_counts < self.rank):
            raise ValueError(
                "each teacher profile needs at least rank observed entries; "
                "a whole-day outage has no teacher coefficient"
            )

        mean_tensor = self.global_mean if mean is None else mean  # [T, C] or [..., T, C]
        if not torch.isfinite(mean_tensor).all():
            raise ValueError("mean profile entries must be finite")
        centered = x_obs - mean_tensor  # [..., T, C]
        flat_centered = torch.where(
            flat_mask.bool(), centered.reshape_as(flat_profiles), torch.zeros_like(flat_profiles)
        )  # [P, TC]
        basis = self.orthonormal_basis()  # [TC, R]
        compute_dtype = self.wls_compute_dtype or x_obs.dtype
        basis = basis.to(compute_dtype)
        identity = torch.eye(self.rank, dtype=compute_dtype, device=x_obs.device)
        profile_count = flat_profiles.shape[0]
        if profile_count == 0:
            return x_obs.new_empty((*profile_shape, self.rank)) + basis.sum().to(x_obs) * 0
        chunk_size = self.wls_chunk_size or profile_count
        # Factoring B_l,r B_l,s once avoids the P*TC*R intermediate, including
        # tensors retained by autograd. Only Gram/solve batches scale with chunk.
        products = ((basis[:, :, None] * basis[:, None, :]).flatten(1)
                    if self.wls_chunk_size is not None else None)
        pieces = []
        for start in range(0, profile_count, chunk_size):
            weights = flat_mask[start:start + chunk_size].to(compute_dtype)
            centered_chunk = flat_centered[start:start + chunk_size].to(compute_dtype)
            if self.wls_chunk_size is None:
                weighted = weights.unsqueeze(-1) * basis.unsqueeze(0)
                gram = torch.einsum("lr,pls->prs", basis, weighted)
                rhs = torch.einsum("plr,pl->pr", weighted, centered_chunk)
            else:
                assert products is not None
                gram = (weights @ products).reshape(-1, self.rank, self.rank)
                rhs = (weights * centered_chunk) @ basis
            # Keep the input-precision identifiability threshold in double audits.
            rank_rtol = self.rank * torch.finfo(x_obs.dtype).eps
            if torch.any(torch.linalg.matrix_rank(gram, rtol=rank_rtol) < self.rank):
                raise ValueError("observed basis is rank deficient")
            solved = torch.linalg.solve(gram + self.ridge * identity, rhs.unsqueeze(-1))
            pieces.append(solved.squeeze(-1).to(x_obs.dtype))
        coefficients = torch.cat(pieces, dim=0)
        return coefficients.reshape(*profile_shape, self.rank)  # [..., R]

    def decode(self, a_pred: Tensor, mean: Optional[Tensor] = None) -> Tensor:
        """Decode profile coefficients into multivariate daily trajectories.

        Args:
            a_pred: Predicted coefficients [..., R].
            mean: Optional conditional means broadcastable to [..., T, C].

        Returns:
            Reconstructed profiles [..., T, C].
        """
        if a_pred.shape[-1] != self.rank:
            raise ValueError("a_pred must end with rank coefficients")
        basis = self.orthonormal_basis()  # [TC, R]
        flat_profile = torch.einsum("...r,lr->...l", a_pred, basis)  # [..., TC]
        profile = flat_profile.reshape(*a_pred.shape[:-1], self.time_steps, self.channels)  # [..., T, C]
        mean_tensor = self.global_mean if mean is None else mean  # [T, C] or [..., T, C]
        return profile + mean_tensor  # [..., T, C]

    def decode_with_observations(
        self,
        a_pred: Tensor,
        x_obs: Tensor,
        mask: Tensor,
        mean: Optional[Tensor] = None,
    ) -> Tensor:
        """Fuse reliable observed coefficients and preserve input-visible values.

        Profiles with fewer than ``rank`` observed entries fall back entirely to
        the predicted coefficients. The final hard copy uses only ``mask`` so
        pseudo-outage targets cannot leak into the prediction.
        """
        decoded = self.decode_fused(a_pred, x_obs, mask, mean)
        return torch.where(mask.bool(), x_obs, decoded)

    def decode_fused(
        self, a_pred: Tensor, x_obs: Tensor, mask: Tensor, mean: Optional[Tensor] = None,
    ) -> Tensor:
        """Expose the existing rho-fused value before final observation hard copy.

        This is the same WLS/valid-count/rho arithmetic as decode_with_observations.
        It allows R/quantile heads to audit B*a separately from the copied data.
        """
        if x_obs.shape[-2:] != (self.time_steps, self.channels):
            raise ValueError("x_obs must end with [time_steps, channels]")
        if mask.shape != x_obs.shape:
            raise ValueError("mask and x_obs must have identical shapes")
        if a_pred.shape[:-1] != x_obs.shape[:-2] or a_pred.shape[-1] != self.rank:
            raise ValueError("a_pred and x_obs leading shapes are incompatible")

        profile_shape = x_obs.shape[:-2]
        flat_mask = mask.reshape(-1, self.time_steps * self.channels).bool()
        valid = flat_mask.sum(dim=-1) >= self.rank
        rho = flat_mask.to(dtype=a_pred.dtype).mean(dim=-1)
        flat_pred = a_pred.reshape(-1, self.rank)
        flat_observed_coefficients = torch.zeros_like(flat_pred)
        if valid.any():
            mean_for_projection: Optional[Tensor]
            if mean is None or mean.ndim == 2:
                mean_for_projection = mean
            else:
                mean_for_projection = mean.reshape(-1, self.time_steps, self.channels)[valid]
            flat_observed_coefficients[valid] = self.project(
                x_obs.reshape(-1, self.time_steps, self.channels)[valid],
                flat_mask[valid].reshape(-1, self.time_steps, self.channels),
                mean=mean_for_projection,
            )
        fusion_weight = torch.where(valid, rho, torch.zeros_like(rho)).unsqueeze(-1)
        fused = fusion_weight * flat_observed_coefficients + (1.0 - fusion_weight) * flat_pred
        return self.decode(fused.reshape(*profile_shape, self.rank), mean=mean)

    def orthogonality_loss(self) -> Tensor:
        """Numerical audit loss for B^T B = I_R."""
        basis = self.orthonormal_basis()  # [TC, R]
        gram = basis.T @ basis  # [R, R]
        identity = torch.eye(self.rank, dtype=basis.dtype, device=basis.device)  # [R, R]
        return (gram - identity).square().mean()  # scalar


class MeanFunction(nn.Module):
    """Conditional mean mu_psi(Z_i, U_d, h_i_pre) with Protocol-U fallback."""

    def __init__(
        self,
        static_dim: int,
        calendar_dim: int,
        history_dim: int,
        time_steps: int = 288,
        channels: int = 3,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        if min(static_dim, calendar_dim, history_dim, time_steps, channels, hidden_dim) <= 0:
            raise ValueError("all dimensions must be positive")
        self.static_dim = static_dim
        self.calendar_dim = calendar_dim
        self.history_dim = history_dim
        self.time_steps = time_steps
        self.channels = channels
        self.no_history_token = nn.Parameter(torch.zeros(history_dim))  # [H_pre]
        self.network = nn.Sequential(
            nn.Linear(static_dim + calendar_dim + history_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, time_steps * channels),
        )

    def forward(
        self,
        static_features: Tensor,
        calendar_features: Tensor,
        history_embedding: Optional[Tensor],
        protocol: str,
    ) -> Tensor:
        """Estimate means without any sensor-ID parameter.

        Args:
            static_features: Road semantics [B, N, F].
            calendar_features: Day features [B, K] or [B, N, K].
            history_embedding: Pre-outage histories [B, N, H_pre], ignored in U.
            protocol: "K" or "U".

        Returns:
            Conditional means [B, N, T, C].
        """
        if protocol not in {"K", "U"}:
            raise ValueError("protocol must be 'K' or 'U'")
        if static_features.ndim != 3 or static_features.shape[-1] != self.static_dim:
            raise ValueError("static_features must have shape [B, N, static_dim]")
        batch_size, node_count, _ = static_features.shape
        calendar = self._expand_calendar(calendar_features, batch_size, node_count)  # [B, N, K]

        if protocol == "U":
            history = self.no_history_token.view(1, 1, -1).expand(
                batch_size, node_count, -1
            )  # [B, N, H_pre]
        else:
            if history_embedding is None:
                raise ValueError("Protocol-K requires pre-outage history_embedding")
            expected = (batch_size, node_count, self.history_dim)
            if history_embedding.shape != expected:
                raise ValueError(f"history_embedding must have shape {expected}")
            history = history_embedding  # [B, N, H_pre]

        conditioning = torch.cat((static_features, calendar, history), dim=-1)  # [B, N, F+K+H_pre]
        flat_mean: Tensor = self.network(conditioning)  # [B, N, TC]
        return flat_mean.reshape(
            batch_size, node_count, self.time_steps, self.channels
        )  # [B, N, T, C]

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
