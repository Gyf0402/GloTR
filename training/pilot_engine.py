"""Shared frozen coordinates for model, train-only teacher and visible-only null."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import torch
from torch import Tensor
from torch.nn import functional as F

from evaluation.low_rank import VisibleCoefficients, encode_visible_profiles
from losses.joint_objective import JointOptimizationLoss
from models.film_rgnn import SemanticDirectedGNN
from training.pilot_data import EpisodeInputs, episode_inputs, rolling_history
from training.real_integration import legal_huber, project_available
from training.stages import FrozenTeacher, StagedRepresentation


def make_student(*, rank: int = 16, hidden: int = 32) -> SemanticDirectedGNN:
    model = SemanticDirectedGNN(rank + 8, 5, 4, hidden, rank, num_layers=2, dropout=0.,
                                temporal_mode="offline_bidirectional")
    model.no_history_token.requires_grad_(False)  # Only K is authorized in this pilot.
    return model


def _empty_components(reference: Tensor) -> dict[str, Tensor]:
    return {name: reference.flatten()[:0].sum() for name in JointOptimizationLoss.LOSS_NAMES}


def reconstruction_objective(
    rep: StagedRepresentation, values: Tensor, visible: Tensor, static: Tensor, calendar: Tensor,
) -> tuple[Tensor, dict[str, Tensor], dict[str, int]]:
    if rep.stage != "fit":
        raise ValueError("masked representation reconstruction requires fit state")
    days, nodes, steps, _ = values.shape
    d = torch.arange(days, device=values.device)[:, None, None, None]
    n = torch.arange(nodes, device=values.device)[None, :, None, None]
    t = torch.arange(steps, device=values.device)[None, None, :, None]
    input_mask = visible & ((d + n + t) % 5 != 0)
    history = rolling_history(values, input_mask)
    mean = rep.mean(static[None].expand(days, -1, -1), calendar, history, protocol="K")
    coefficients, _ = project_available(rep.basis, values, input_mask, mean)
    prediction = rep.basis.decode(coefficients, mean)
    imp, imp_count = legal_huber(prediction, values, visible & ~input_mask)
    obs, obs_count = legal_huber(prediction, values, input_mask)
    components = _empty_components(prediction)
    components.update(imp=imp, obs=obs)
    return imp + .1 * obs, components, {"imp": imp_count, "obs": obs_count, "coef": 0}


@dataclass(frozen=True)
class ModelContext:
    inputs: EpisodeInputs
    mean: Tensor
    features: Tensor
    calendar: Tensor
    dates: tuple[str, ...]
    ordinals: Tensor
    observed_profiles: Tensor
    coordinate_fingerprint: str


@torch.no_grad()
def prepare_context(
    rep: StagedRepresentation, inputs: EpisodeInputs, static: Tensor,
    calendar: Tensor, dates: tuple[str, ...],
) -> ModelContext:
    rep.assert_frozen()
    if len(dates) != 13:
        raise ValueError("a model context needs exactly one 13-day episode")
    ordinals = torch.tensor([date.fromisoformat(x).toordinal() for x in dates], dtype=torch.long,
                            device=inputs.values.device)
    if not torch.all(ordinals[1:] == ordinals[:-1] + 1):
        raise ValueError("within-episode days must be consecutive")
    mean = rep.mean(static[None].expand(13, -1, -1), calendar, inputs.history, protocol="K")
    coefficients, observed = project_available(rep.basis, inputs.values, inputs.visible, mean)
    ratio = inputs.visible.to(inputs.values).mean(dim=(-1, -2))[..., None]
    features = torch.cat((coefficients, ratio, observed.to(coefficients)[..., None],
                          inputs.history), dim=-1)
    if not torch.isfinite(features).all():
        raise ValueError("all student features must be finite after the input boundary")
    return ModelContext(inputs, mean, features, calendar, dates, ordinals, observed,
                        rep.coordinate_fingerprint())


def predict(
    rep: StagedRepresentation, model: SemanticDirectedGNN, context: ModelContext,
    static: Tensor, edges: dict[str, Tensor],
) -> tuple[Tensor, Tensor, Tensor]:
    rep.assert_frozen()
    if rep.coordinate_fingerprint() != context.coordinate_fingerprint:
        raise ValueError("student input coordinate version changed")
    coefficients = model(context.features, static, context.calendar, edges, protocol="K",
                         episode_ids=torch.zeros_like(context.ordinals),
                         day_ordinals=context.ordinals)
    # Observation supervision is before both coefficient fusion and hard copy.
    prediction = rep.basis.decode(coefficients, context.mean)
    published = rep.basis.decode_with_observations(coefficients, context.inputs.values,
                                                  context.inputs.visible, context.mean)
    return coefficients, prediction, published


@dataclass(frozen=True)
class TrainingCase:
    context: ModelContext
    truth: Tensor
    reference_mask: Tensor
    target_index: int
    teacher: FrozenTeacher

    def assert_teacher(self, rep: StagedRepresentation) -> float:
        return self.teacher.assert_unchanged(rep, self.context.mean[3:10, self.target_index])


@torch.no_grad()
def prepare_training_case(
    rep: StagedRepresentation, values: Tensor, visible: Tensor, static: Tensor,
    calendar: Tensor, dates: tuple[str, ...], *, target_index: int,
) -> TrainingCase:
    """Caller must take these labels exclusively from PilotScope.read_fit()."""
    inputs = episode_inputs(values, visible, target_index=target_index)
    context = prepare_context(rep, inputs, static, calendar, dates)
    teacher = FrozenTeacher(rep, values[3:10, target_index], visible[3:10, target_index],
                            context.mean[3:10, target_index])
    return TrainingCase(context, values, visible, target_index, teacher)


def student_components(
    rep: StagedRepresentation, case: TrainingCase, coefficients: Tensor,
    prediction: Tensor, published: Tensor,
) -> tuple[dict[str, Tensor], dict[str, int]]:
    support = case.reference_mask & case.context.inputs.outage[..., None, None]
    imp, imp_count = legal_huber(published, case.truth, support)
    obs, obs_count = legal_huber(prediction, case.truth, case.context.inputs.visible)
    coef = F.l1_loss(coefficients[3:10, case.target_index], case.teacher.coefficients)
    components = _empty_components(coefficients)
    components.update(imp=imp, obs=obs, coef=coef, basis=rep.basis.orthogonality_loss())
    return components, {"imp": imp_count, "obs": obs_count,
                        "coef": case.teacher.coefficients.numel()}


@torch.no_grad()
def encode_null_inputs(
    rep: StagedRepresentation, historical_values: Tensor, historical_visible: Tensor,
    historical_mean: Tensor, historical_dates: tuple[str, ...], context: ModelContext,
    station_ids: tuple[str, ...], *, source_id: str,
) -> VisibleCoefficients:
    """Only observed inner-fit and current M_in profiles; no reference/teacher argument."""
    rep.assert_frozen()
    dates = historical_dates + context.dates
    profiles = torch.cat((historical_values, context.inputs.values)).permute(1, 0, 2, 3)
    visible = torch.cat((historical_visible, context.inputs.visible)).permute(1, 0, 2, 3)
    mean = torch.cat((historical_mean, context.mean)).permute(1, 0, 2, 3)
    legal = torch.cat((torch.ones(historical_values.shape[:2], dtype=torch.bool,
                                 device=profiles.device), ~context.inputs.outage)).T
    return encode_visible_profiles(profiles, visible, basis=rep.basis, legal_profiles=legal,
                                   station_ids=station_ids, dates=dates, basis_source_id=source_id,
                                   mean_source_id=source_id,
                                   coordinate_id=context.coordinate_fingerprint, mean=mean)
