"""Additional strict-observable baselines for Milestone 2C.

These methods intentionally remain small, CPU-compatible and explicit about
their motion assumptions.  None accepts a full sample or hidden trajectory.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn

from actmask.models.milestone2b_baselines import (
    ConstantVelocityKalmanProximity,
    CurrentPositionProximity,
    EstimatedTimeAlignedTrajectoryProximity,
    EstimatedVelocityFutureProximity,
    _ObservableBaseline,
    _observable_tensors,
    _trajectory,
    _utility_from_logits,
)
from actmask.models.temporal_actmask import (
    estimated_acceleration_from_history,
    last_visible_state,
)


def _trajectory_logits(
    value: Mapping[str, Tensor], future: Tensor, *, temperature: float
) -> tuple[Tensor, Tensor]:
    _, gripper, _ = _trajectory(
        value["action_command"], value["nominal_action_delay"], future.shape[2]
    )
    distance = torch.linalg.vector_norm(future - gripper[:, None], dim=-1)
    logits = (value["action_command"][:, 7:8] - distance.amin(dim=-1)) / temperature
    return logits, distance


class ConstantAccelerationTrajectoryProximity(_ObservableBaseline):
    """Bounded finite-difference acceleration from noisy visible history."""

    assumptions = "constant acceleration estimated from visible observation history"

    def forward(self, observable: Mapping[str, Any]) -> dict[str, Tensor]:
        value = _observable_tensors(observable)
        history = value["points_history"]
        visibility = value["visibility_history"].bool()
        timestamps = value["timestamps"]
        last, last_time, any_visible = last_visible_state(history, visibility, timestamps)
        acceleration = estimated_acceleration_from_history(history, visibility, timestamps)
        acceleration = 1.5 * torch.tanh(acceleration / 1.5) * value[
            "velocity_confidence"
        ][:, :, None]
        times, _, _ = _trajectory(
            value["action_command"], value["nominal_action_delay"], self.horizon_steps
        )
        elapsed = times[:, None, :] - last_time[:, :, None]
        future = (
            last[:, :, None]
            + value["estimated_velocity"][:, :, None] * elapsed[:, :, :, None]
            + 0.5 * acceleration[:, :, None] * elapsed[:, :, :, None].square()
        )
        logits, _ = _trajectory_logits(value, future, temperature=self.temperature)
        logits = torch.where(any_visible, logits, torch.full_like(logits, -8.0))
        return self._output(logits, future)


class AlphaBetaFilterTrajectoryProximity(_ObservableBaseline):
    """Observable alpha-beta position/velocity filter."""

    assumptions = "constant-velocity alpha-beta filter with fixed validation-frozen gains"

    def __init__(
        self, *, alpha: float = 0.72, beta: float = 0.18, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        if not 0.0 < alpha <= 1.0 or not 0.0 < beta <= 1.0:
            raise ValueError("alpha and beta must lie in (0,1]")
        self.alpha = float(alpha)
        self.beta = float(beta)

    def forward(self, observable: Mapping[str, Any]) -> dict[str, Tensor]:
        value = _observable_tensors(observable)
        history = value["points_history"]
        visibility = value["visibility_history"].bool()
        timestamps = value["timestamps"]
        batch, frames, points, _ = history.shape
        position = history[:, 0].clone()
        velocity = torch.zeros_like(value["estimated_velocity"])
        initialized = visibility[:, 0].clone()
        for frame in range(1, frames):
            dt = (timestamps[:, frame] - timestamps[:, frame - 1]).clamp_min(1.0e-4)
            prediction = position + velocity * dt[:, None, None]
            valid = visibility[:, frame]
            residual = history[:, frame] - prediction
            corrected_position = prediction + self.alpha * residual
            corrected_velocity = velocity + (self.beta / dt[:, None, None]) * residual
            position = torch.where(valid[:, :, None], corrected_position, prediction)
            velocity = torch.where(valid[:, :, None], corrected_velocity, velocity)
            initialized = torch.logical_or(initialized, valid)
        last, last_time, any_visible = last_visible_state(history, visibility, timestamps)
        position = torch.where(initialized[:, :, None], position, last)
        velocity = torch.where(
            initialized[:, :, None], velocity, value["estimated_velocity"]
        )
        times, _, _ = _trajectory(
            value["action_command"], value["nominal_action_delay"], self.horizon_steps
        )
        elapsed = times[:, None] - last_time[:, :, None]
        future = position[:, :, None] + velocity[:, :, None] * elapsed[:, :, :, None]
        logits, _ = _trajectory_logits(value, future, temperature=self.temperature)
        logits = torch.where(any_visible, logits, torch.full_like(logits, -8.0))
        return self._output(logits, future)


class KalmanAccelerationStateProximity(_ObservableBaseline):
    """Small diagonal constant-acceleration state filter without dependencies."""

    assumptions = "diagonal alpha-beta-gamma constant-acceleration state filter"

    def __init__(
        self,
        *,
        alpha: float = 0.66,
        beta: float = 0.16,
        gamma: float = 0.035,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if min(alpha, beta, gamma) <= 0.0:
            raise ValueError("filter gains must be positive")
        self.alpha, self.beta, self.gamma = float(alpha), float(beta), float(gamma)

    def forward(self, observable: Mapping[str, Any]) -> dict[str, Tensor]:
        value = _observable_tensors(observable)
        history = value["points_history"]
        visibility = value["visibility_history"].bool()
        timestamps = value["timestamps"]
        position = history[:, 0].clone()
        velocity = torch.zeros_like(value["estimated_velocity"])
        acceleration = torch.zeros_like(value["estimated_velocity"])
        initialized = visibility[:, 0].clone()
        for frame in range(1, history.shape[1]):
            dt = (timestamps[:, frame] - timestamps[:, frame - 1]).clamp_min(1.0e-4)
            prediction = position + velocity * dt[:, None, None] + 0.5 * acceleration * dt[:, None, None].square()
            predicted_velocity = velocity + acceleration * dt[:, None, None]
            residual = history[:, frame] - prediction
            valid = visibility[:, frame]
            position = torch.where(valid[:, :, None], prediction + self.alpha * residual, prediction)
            velocity = torch.where(
                valid[:, :, None],
                predicted_velocity + (self.beta / dt[:, None, None]) * residual,
                predicted_velocity,
            )
            acceleration = torch.where(
                valid[:, :, None],
                acceleration + (self.gamma / dt[:, None, None].square()) * residual,
                acceleration,
            )
            initialized = torch.logical_or(initialized, valid)
        last, last_time, any_visible = last_visible_state(history, visibility, timestamps)
        position = torch.where(initialized[:, :, None], position, last)
        velocity = torch.where(initialized[:, :, None], velocity, value["estimated_velocity"])
        acceleration = 1.5 * torch.tanh(acceleration / 1.5)
        times, _, _ = _trajectory(
            value["action_command"], value["nominal_action_delay"], self.horizon_steps
        )
        elapsed = times[:, None] - last_time[:, :, None]
        future = (
            position[:, :, None]
            + velocity[:, :, None] * elapsed[:, :, :, None]
            + 0.5 * acceleration[:, :, None] * elapsed[:, :, :, None].square()
        )
        logits, _ = _trajectory_logits(value, future, temperature=self.temperature)
        logits = torch.where(any_visible, logits, torch.full_like(logits, -8.0))
        return self._output(logits, future)


class MultiHypothesisTrajectoryProximity(_ObservableBaseline):
    """Contact probability marginalized over observable velocity uncertainty."""

    assumptions = "constant velocity with five confidence-scaled velocity hypotheses"

    def __init__(self, *, uncertainty_scale: float = 0.22, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if uncertainty_scale <= 0.0:
            raise ValueError("uncertainty_scale must be positive")
        self.uncertainty_scale = float(uncertainty_scale)

    def forward(self, observable: Mapping[str, Any]) -> dict[str, Tensor]:
        value = _observable_tensors(observable)
        history = value["points_history"]
        visibility = value["visibility_history"].bool()
        timestamps = value["timestamps"]
        last, last_time, any_visible = last_visible_state(history, visibility, timestamps)
        times, gripper, _ = _trajectory(
            value["action_command"], value["nominal_action_delay"], self.horizon_steps
        )
        elapsed = times[:, None] - last_time[:, :, None]
        velocity = value["estimated_velocity"]
        norm = torch.linalg.vector_norm(velocity, dim=-1, keepdim=True).clamp_min(1.0e-5)
        direction = velocity / norm
        uncertainty = self.uncertainty_scale * (1.0 - value["velocity_confidence"])[
            :, :, None
        ]
        offsets = torch.tensor(
            [-1.0, -0.45, 0.0, 0.45, 1.0], dtype=velocity.dtype, device=velocity.device
        )
        hypotheses = []
        for offset in offsets:
            hypothesis_velocity = velocity + offset * uncertainty * direction
            hypotheses.append(
                last[:, :, None]
                + hypothesis_velocity[:, :, None] * elapsed[:, :, :, None]
            )
        future_hypotheses = torch.stack(hypotheses, dim=1)
        distance = torch.linalg.vector_norm(
            future_hypotheses - gripper[:, None, None], dim=-1
        )
        contact_logit = (
            value["action_command"][:, 7:8, None]
            - distance.amin(dim=-1)
        ) / self.temperature
        logits = torch.logsumexp(contact_logit, dim=1) - math.log(len(offsets))
        logits = torch.where(any_visible, logits, torch.full_like(logits, -8.0))
        future = future_hypotheses[:, len(offsets) // 2]
        return self._output(logits, future)


class GeometryWithLearnedResidual(nn.Module):
    """Fair learned residual layered on observable time-aligned geometry."""

    learned = True
    assumptions = "small residual over observable constant-velocity time-aligned geometry"

    def __init__(self, hidden_dim: int = 12, *, temperature: float = 0.035) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        self.temperature = float(temperature)
        self.residual = nn.Sequential(
            nn.Linear(19, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )
        self.logit_scale = nn.Parameter(torch.tensor(0.15))

    def forward(self, observable: Mapping[str, Any]) -> dict[str, Tensor]:
        value = _observable_tensors(observable)
        history = value["points_history"]
        visibility = value["visibility_history"].bool()
        timestamps = value["timestamps"]
        last, last_time, any_visible = last_visible_state(history, visibility, timestamps)
        times, _, _ = _trajectory(
            value["action_command"], value["nominal_action_delay"], 32
        )
        elapsed = times[:, None] - last_time[:, :, None]
        future = last[:, :, None] + value["estimated_velocity"][:, :, None] * elapsed[:, :, :, None]
        base_logits, _ = _trajectory_logits(value, future, temperature=self.temperature)
        batch, points, _ = last.shape
        action = value["action_command"][:, None].expand(batch, points, -1)
        scalar = torch.stack(
            (
                value["velocity_confidence"],
                value["nominal_action_delay"][:, None].expand(batch, points),
                value["observation_delay"][:, None].expand(batch, points),
                visibility.to(last.dtype).mean(dim=1),
                base_logits,
            ),
            dim=-1,
        )
        feature = torch.cat((last, value["estimated_velocity"], action, scalar), dim=-1)
        logits = base_logits + self.logit_scale * self.residual(feature).squeeze(-1)
        logits = torch.where(any_visible, logits, torch.full_like(logits, -8.0))
        return {
            "mask_logits": logits,
            "utility_score": _utility_from_logits(logits),
            "future_hypotheses": future,
        }


FAIR_BASELINES_2C = {
    "CurrentPositionProximity": CurrentPositionProximity,
    "EstimatedVelocityFutureProximity": EstimatedVelocityFutureProximity,
    "EstimatedTimeAlignedTrajectoryProximity": EstimatedTimeAlignedTrajectoryProximity,
    "ConstantVelocityKalmanProximity": ConstantVelocityKalmanProximity,
    "ConstantAccelerationTrajectoryProximity": ConstantAccelerationTrajectoryProximity,
    "AlphaBetaFilterTrajectoryProximity": AlphaBetaFilterTrajectoryProximity,
    "KalmanAccelerationStateProximity": KalmanAccelerationStateProximity,
    "MultiHypothesisTrajectoryProximity": MultiHypothesisTrajectoryProximity,
    "GeometryWithLearnedResidual": GeometryWithLearnedResidual,
}


__all__ = [
    "AlphaBetaFilterTrajectoryProximity",
    "ConstantAccelerationTrajectoryProximity",
    "FAIR_BASELINES_2C",
    "GeometryWithLearnedResidual",
    "KalmanAccelerationStateProximity",
    "MultiHypothesisTrajectoryProximity",
]
