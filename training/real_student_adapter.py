"""Boundary adapter from frozen real q/o episodes to the coefficient student.

This module performs no fitting and has no Test access. It deliberately requires
an audited four-relation graph; an empty or incomplete graph is not substituted.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor

from models.film_rgnn import SemanticDirectedGNN
from training.pilot_data import calendar_features
from training.pilot_data import rolling_history
from training.real_integration import project_available
from representations.basis import GlobalProfileBasis


@dataclass(frozen=True)
class RealStudentInputs:
    features: Tensor
    calendar: Tensor
    mean: Tensor
    coefficients: Tensor
    coefficient_observed: Tensor
    edges: Mapping[str, Tensor]


def _check_edges(edges: Mapping[str, Tensor], nodes: int, device: torch.device) -> None:
    if set(edges) != set(SemanticDirectedGNN.RELATIONS):
        raise ValueError("audited four-relation edge sets are required")
    for name in SemanticDirectedGNN.RELATIONS:
        edge = edges[name]
        if edge.device != device or edge.dtype != torch.long or edge.ndim != 2 or edge.shape[0] != 2:
            raise ValueError(f"{name} edge_index must be [2,E] long on the input device")
        if edge.numel() and (edge.min() < 0 or edge.max() >= nodes):
            raise ValueError(f"{name} edge_index is outside station order")


def build_student_inputs(basis: GlobalProfileBasis, values: Tensor, visible: Tensor,
                         mean: Tensor, static: Tensor,
                         edges: Mapping[str, Tensor], dates: tuple[str, ...] | None = None) -> RealStudentInputs:
    """Build finite student features from legal visible values and frozen mean."""
    if any(p.requires_grad or p.grad is not None for p in basis.parameters()):
        raise ValueError("basis must be frozen")
    if values.ndim != 4 or values.shape != visible.shape or values.shape != mean.shape:
        raise ValueError("values, visible and mean must share [D,N,T,2] shape")
    days, nodes, steps, channels = values.shape
    if (steps, channels) != (basis.time_steps, basis.channels) or visible.dtype != torch.bool:
        raise ValueError("q/o dimensions and boolean visible mask are required")
    if static.shape != (nodes, 5) or not torch.isfinite(static).all():
        raise ValueError("static features must be finite [N,5]")
    if not torch.isfinite(values[visible]).all() or not torch.isfinite(mean).all():
        raise ValueError("visible values and frozen mean must be finite")
    _check_edges(edges, nodes, values.device)
    history = rolling_history(torch.where(visible, values, torch.zeros_like(values)), visible)
    coefficients, observed = project_available(basis, values, visible, mean)
    ratio = visible.to(values).mean(dim=(-1, -2))[..., None]
    features = torch.cat((coefficients, ratio, observed.to(coefficients)[..., None], history), dim=-1)
    if dates is None:
        raise ValueError("real date labels are required")
    if len(dates) != days:
        raise ValueError("one ISO date per day is required")
    calendar = calendar_features(dates).to(values.device)
    if not torch.isfinite(features).all() or not torch.isfinite(calendar).all():
        raise ValueError("student inputs must be finite")
    return RealStudentInputs(features, calendar, mean, coefficients, observed, edges)


def build_student_inputs_with_dates(basis: GlobalProfileBasis, values: Tensor, visible: Tensor,
                                    mean: Tensor, static: Tensor, edges: Mapping[str, Tensor],
                                    dates: tuple[str, ...]) -> RealStudentInputs:
    """Date-aware version used by the real runner."""
    if len(dates) != values.shape[0]:
        raise ValueError("one ISO date per day is required")
    # Reuse validation while avoiding a fabricated calendar.
    if any(p.requires_grad or p.grad is not None for p in basis.parameters()):
        raise ValueError("basis must be frozen")
    if values.ndim != 4 or values.shape != visible.shape or values.shape != mean.shape:
        raise ValueError("values, visible and mean must share [D,N,T,C] shape")
    if visible.dtype != torch.bool or values.shape[-2:] != (basis.time_steps, basis.channels):
        raise ValueError("dimensions and boolean visible mask are required")
    _check_edges(edges, values.shape[1], values.device)
    history = rolling_history(torch.where(visible, values, torch.zeros_like(values)), visible)
    coefficients, observed = project_available(basis, values, visible, mean)
    features = torch.cat((coefficients, visible.to(values).mean(dim=(-1, -2))[..., None],
                          observed.to(coefficients)[..., None], history), dim=-1)
    calendar = calendar_features(dates).to(values.device)
    if not torch.isfinite(features).all() or not torch.isfinite(calendar).all():
        raise ValueError("student inputs must be finite")
    return RealStudentInputs(features, calendar, mean, coefficients, observed, edges)
