"""Representation lifecycle, legal mean reference, and immutable teacher coordinates."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from representations.basis import GlobalProfileBasis, MeanFunction

COORDINATE_VERSION = "qr-reduced-diag-nonnegative/v1;columns=insertion-order/v1"


class StagedRepresentation(nn.Module):
    """Fit a mean offset and basis, freeze them, optionally fine-tune at 0.1 LR.

    ``global_mean`` is a frozen training-only reference, not an unused trainable
    variable. The original core modules and their public APIs remain unchanged.
    """

    def __init__(self, basis: GlobalProfileBasis, mean_function: MeanFunction) -> None:
        super().__init__()
        if (basis.time_steps, basis.channels) != (
            mean_function.time_steps, mean_function.channels
        ):
            raise ValueError("basis and mean profile dimensions must agree")
        self.basis = basis
        self.mean_function = mean_function
        self.stage = "fit"
        self.generation = 0
        self.reference_provenance: dict[str, Any] = {}
        self._frozen_snapshot: dict[str, Tensor] = {}
        self.basis.global_mean.requires_grad_(False)
        head = self.mean_function.network[-1]
        if not isinstance(head, nn.Linear):
            raise TypeError("conditional offset needs a final Linear output head")
        nn.init.zeros_(head.weight)
        nn.init.zeros_(head.bias)

    def fit_reference(self, values: Tensor, mask: Tensor, *, split: str) -> None:
        """Compute each reference coordinate from visible training entries only.

        The caller must exclude permanent-U nodes before this boundary. This API
        rejects validation/test provenance and requires support at every (T,C).
        """
        if self.stage != "fit" or self.reference_provenance:
            raise RuntimeError("reference can only be fitted once in fit state")
        if split != "train":
            raise ValueError("reference requires train provenance")
        if values.shape != mask.shape or values.ndim < 3:
            raise ValueError("reference requires matching [..., T, C] values/mask")
        if values.shape[-2:] != (self.basis.time_steps, self.basis.channels):
            raise ValueError("reference profile dimensions are incompatible")
        visible = mask.bool()
        if not torch.isfinite(values[visible]).all():
            raise ValueError("visible reference values must be finite")
        axes = tuple(range(values.ndim - 2))
        support = visible.sum(dim=axes)
        if torch.any(support == 0):
            raise ValueError("every reference coordinate needs training support")
        safe = torch.where(visible, values, torch.zeros_like(values))
        with torch.no_grad():
            self.basis.global_mean.copy_(safe.sum(dim=axes) / support)
        self.reference_provenance = {
            "split": split, "visible_entries": int(visible.sum().item()),
            "minimum_coordinate_support": int(support.min().item()),
            "role": "frozen legal reference; mu = reference + conditional offset",
        }

    def mean(
        self, static: Tensor, calendar: Tensor, history: Tensor | None, *, protocol: str
    ) -> Tensor:
        if not self.reference_provenance:
            raise RuntimeError("fit training reference before using mu")
        return self.basis.global_mean + self.mean_function(static, calendar, history, protocol)

    def reconstruct(self, coefficients: Tensor, mean: Tensor, mode: str) -> Tensor:
        """Three named mu ablations; residual R is absent in this skeleton."""
        if mode == "mu-only":
            return mean
        if mode == "basis-only":
            return self.basis.decode(coefficients, mean=self.basis.global_mean)
        if mode == "mu+projection":
            return self.basis.decode(coefficients, mean=mean)
        raise ValueError("mode must be mu-only, basis-only, or mu+projection")

    def project_visible(self, values: Tensor, mask: Tensor, mean: Tensor, *, mode: str) -> Tensor:
        """Encode only visible profiles in the ablation's declared centering."""
        self.assert_frozen()
        if mode == "mu-only":
            raise ValueError("mu-only has no coefficient projection or teacher")
        if mode not in {"basis-only", "mu+projection"}:
            raise ValueError("unknown representation mode")
        return self.basis.project(values, mask,
                                  self.basis.global_mean if mode == "basis-only" else mean)

    def bind_coordinates(
        self, role: str, *, mode: str = "mu+projection", scale_fingerprint: str | None = None,
    ) -> dict[str, Any]:
        """Metadata contract only; this does not construct a behavior graph."""
        self.assert_frozen()
        if role not in {"teacher", "student", "behavior_graph", "low_rank_null"}:
            raise ValueError("unknown coordinate consumer role")
        if mode not in {"mu-only", "basis-only", "mu+projection"}:
            raise ValueError("unknown coordinate mode")
        if scale_fingerprint is not None and len(scale_fingerprint) != 64:
            raise ValueError("scale fingerprint must be SHA-256")
        return {
            "coordinate_version": COORDINATE_VERSION,
            "coordinate_fingerprint": self.coordinate_fingerprint(),
            "role": role, "mode": mode, "scale_fingerprint": scale_fingerprint,
            "projection_chunk_size": self.basis.wls_chunk_size,
            "projection_compute_dtype": str(self.basis.wls_compute_dtype),
        }

    def assert_coordinates(self, binding: dict[str, Any], *, role: str, mode: str) -> None:
        if binding.get("mode") != mode:
            raise ValueError("coordinate centering mode mismatch")
        expected = self.bind_coordinates(role, mode=mode,
                                         scale_fingerprint=binding.get("scale_fingerprint"))
        if binding != expected:
            raise ValueError("coordinate consumer/version/projection mismatch")

    def freeze(self) -> str:
        if not self.reference_provenance:
            raise RuntimeError("fit training reference before freezing")
        self.requires_grad_(False)
        for parameter in self.parameters():
            parameter.grad = None
        self.eval()
        self.stage = "frozen"
        self._frozen_snapshot = {k: v.detach().clone() for k, v in self.state_dict().items()}
        return self.coordinate_fingerprint()

    def assert_frozen(self) -> None:
        if self.stage != "frozen" or not self._frozen_snapshot:
            raise RuntimeError("representation must be frozen")
        for name, parameter in self.named_parameters():
            if parameter.requires_grad or parameter.grad is not None:
                raise RuntimeError(f"frozen gradient contract changed: {name}")
        for name, value in self.state_dict().items():
            if not torch.equal(value, self._frozen_snapshot[name].to(value.device)):
                raise RuntimeError(f"frozen representation changed: {name}")

    def coordinate_fingerprint(self) -> str:
        """Versioned raw QR coordinates plus mean state; no sign/order relabeling.

        Store raw tensors rather than recomputed QR bytes so checkpoints retain
        the same identity across devices. Exact numerical assertions use the
        saved state and per-run teacher; cross-device QR rounding is not hidden.
        """
        digest = hashlib.sha256(f"{COORDINATE_VERSION};generation={self.generation}".encode())
        for name, value in sorted(self.state_dict().items()):
            cpu = value.detach().cpu().contiguous()
            digest.update(f"{name}:{tuple(cpu.shape)}:{cpu.dtype}".encode())
            digest.update(cpu.numpy().tobytes())
        return digest.hexdigest()

    def enable_finetune(self) -> None:
        self.assert_frozen()
        self.requires_grad_(True)
        self.basis.global_mean.requires_grad_(False)
        self.stage = "optional_finetune"
        self.generation += 1  # All existing teacher caches are now invalid.
        self._frozen_snapshot = {}
        self.train()

    def optimizer_groups(
        self, student: nn.Module | None, main_lr: float
    ) -> list[dict[str, Any]]:
        if main_lr <= 0:
            raise ValueError("main_lr must be positive")
        if self.stage == "fit":
            if student is not None:
                raise ValueError("fit stage has no student")
            return [{"name": "representation_fit", "params": [
                p for p in self.parameters() if p.requires_grad
            ], "lr": main_lr}]
        if student is None:
            raise ValueError("student is required after fit")
        groups = [{"name": "student", "params": [
            p for p in student.parameters() if p.requires_grad
        ], "lr": main_lr}]
        if self.stage == "optional_finetune":
            groups.append({"name": "representation_finetune", "params": [
                p for p in self.parameters() if p.requires_grad
            ], "lr": main_lr * 0.1})
        else:
            self.assert_frozen()
        return groups

    def checkpoint(self) -> dict[str, Any]:
        hidden = self.mean_function.network[0]
        assert isinstance(hidden, nn.Linear)
        return {
            "schema_version": 1, "coordinate_version": COORDINATE_VERSION,
            "config": {
                "time_steps": self.basis.time_steps, "channels": self.basis.channels,
                "rank": self.basis.rank, "ridge": self.basis.ridge,
                "wls_chunk_size": self.basis.wls_chunk_size,
                "wls_compute_dtype": str(self.basis.wls_compute_dtype),
                "static_dim": self.mean_function.static_dim,
                "calendar_dim": self.mean_function.calendar_dim,
                "history_dim": self.mean_function.history_dim, "hidden_dim": hidden.out_features,
            },
            "state_dict": self.state_dict(), "stage": self.stage,
            "orthonormal_basis_snapshot": self.basis.orthonormal_basis().detach(),
            "column_order": list(range(self.basis.rank)),
            "generation": self.generation, "reference_provenance": self.reference_provenance,
            "coordinate_fingerprint": self.coordinate_fingerprint(),
        }

    def save_checkpoint(self, path: str | Path) -> None:
        if self.stage == "frozen":
            self.assert_frozen()
        torch.save(self.checkpoint(), path)

    @classmethod
    def load_checkpoint(
        cls, path: str | Path, device: str | torch.device = "cpu"
    ) -> StagedRepresentation:
        payload = torch.load(path, map_location=device, weights_only=True)
        if payload["schema_version"] != 1 or payload["coordinate_version"] != COORDINATE_VERSION:
            raise ValueError("unsupported checkpoint coordinate/schema version")
        config = payload["config"]
        dtype_names = {"None": None, "torch.float32": torch.float32,
                       "torch.float64": torch.float64}
        rep = cls(
            GlobalProfileBasis(config["time_steps"], config["channels"], config["rank"],
                               config["ridge"], wls_chunk_size=config.get("wls_chunk_size"),
                               wls_compute_dtype=dtype_names[config.get("wls_compute_dtype", "None")]),
            MeanFunction(config["static_dim"], config["calendar_dim"], config["history_dim"],
                         config["time_steps"], config["channels"], config["hidden_dim"]),
        ).to(device)
        rep.load_state_dict(payload["state_dict"])
        rep.reference_provenance = payload["reference_provenance"]
        rep.generation = payload["generation"]
        if payload["stage"] == "frozen":
            rep.freeze()
        elif payload["stage"] in {"fit", "optional_finetune"}:
            rep.stage = payload["stage"]
        else:
            raise ValueError("unsupported checkpoint stage")
        if rep.coordinate_fingerprint() != payload["coordinate_fingerprint"]:
            raise ValueError("checkpoint coordinate fingerprint mismatch")
        return rep


class FrozenTeacher:
    """A teacher bound to an immutable representation and fixed legal profiles."""

    def __init__(
        self, representation: StagedRepresentation, values: Tensor, mask: Tensor, mean: Tensor,
        *, mode: str = "mu+projection",
    ) -> None:
        representation.assert_frozen()
        self.fingerprint = representation.coordinate_fingerprint()
        self.mode = mode
        self.binding = representation.bind_coordinates("teacher", mode=mode)
        self.values, self.mask = values.detach().clone(), mask.detach().clone()
        with torch.no_grad():
            self.coefficients = representation.project_visible(
                self.values, self.mask, mean, mode=mode).clone()

    def assert_unchanged(self, representation: StagedRepresentation, mean: Tensor) -> float:
        representation.assert_frozen()
        if representation.coordinate_fingerprint() != self.fingerprint:
            raise RuntimeError("teacher coordinate fingerprint changed")
        representation.assert_coordinates(self.binding, role="teacher", mode=self.mode)
        with torch.no_grad():
            current = representation.project_visible(self.values, self.mask, mean, mode=self.mode)
        if not torch.equal(current, self.coefficients):
            drift = float((current - self.coefficients).abs().max().item())
            raise RuntimeError(f"teacher changed: maximum coefficient drift={drift}")
        return 0.0
