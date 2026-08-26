"""Observable-only geometric baselines and the excluded 2B hidden oracle."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn

from actmask.data.milestone2b_dataset import (
    FUTURE_STEPS,
    HIDDEN_STATE_FIELDS,
    validate_observable_mapping,
)
from actmask.models.temporal_actmask import last_visible_state


def _observable_tensors(observable: Mapping[str, Any]) -> dict[str, Tensor]:
    validate_observable_mapping(observable)
    result: dict[str, Tensor] = {}
    for key, value in observable.items():
        if not isinstance(value, Tensor):
            raise TypeError(f"observable[{key!r}] must be a tensor")
        result[key] = value
    history = result["points_history"]
    if history.ndim != 4 or history.shape[-1] != 3:
        raise ValueError("points_history must have shape [B,K,N,3]")
    if result["visibility_history"].shape != history.shape[:3]:
        raise ValueError("visibility_history shape does not match points_history")
    if result["timestamps"].shape != history.shape[:2]:
        raise ValueError("timestamps must have shape [B,K]")
    return result


def _trajectory(
    action: Tensor, delay: Tensor, steps: int
) -> tuple[Tensor, Tensor, Tensor]:
    fraction = torch.linspace(
        0.0, 1.0, steps, dtype=action.dtype, device=action.device
    )
    duration = action[:, 6].clamp_min(1.0e-4)
    times = delay[:, None] + fraction[None, :] * duration[:, None]
    gripper = action[:, None, :3] + fraction[None, :, None] * (
        action[:, None, 3:6] - action[:, None, :3]
    )
    return times, gripper, fraction


def _distance_to_segment(points: Tensor, action: Tensor) -> Tensor:
    start = action[:, None, :3]
    segment = action[:, None, 3:6] - start
    squared = (segment * segment).sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
    fraction = ((points - start) * segment).sum(dim=-1, keepdim=True) / squared
    closest = start + fraction.clamp(0.0, 1.0) * segment
    return torch.linalg.vector_norm(points - closest, dim=-1)


def _utility_from_logits(logits: Tensor, topk: int = 8) -> Tensor:
    probability = torch.sigmoid(logits)
    count = min(int(topk), probability.shape[1])
    return probability.topk(count, dim=1).values.mean(dim=1)


class _ObservableBaseline(nn.Module):
    def __init__(self, *, temperature: float = 0.035, horizon_steps: int = FUTURE_STEPS) -> None:
        super().__init__()
        if temperature <= 0.0 or horizon_steps < 8:
            raise ValueError("temperature must be positive and horizon_steps at least 8")
        self.temperature = float(temperature)
        self.horizon_steps = int(horizon_steps)

    def _output(self, logits: Tensor, future: Tensor | None = None) -> dict[str, Tensor]:
        output = {
            "mask_logits": logits,
            "utility_score": _utility_from_logits(logits),
        }
        if future is not None:
            output["future_hypotheses"] = future
        return output


class CurrentPositionProximity(_ObservableBaseline):
    """Distance from the last visible point to the commanded action segment."""

    def forward(self, observable: Mapping[str, Any]) -> dict[str, Tensor]:
        value = _observable_tensors(observable)
        last, _, any_visible = last_visible_state(
            value["points_history"],
            value["visibility_history"],
            value["timestamps"],
        )
        distance = _distance_to_segment(last, value["action_command"])
        radius = value["action_command"][:, 7:8]
        logits = (radius - distance) / self.temperature
        logits = torch.where(any_visible, logits, torch.full_like(logits, -8.0))
        return self._output(logits)


class EstimatedVelocityFutureProximity(_ObservableBaseline):
    """Estimated-velocity endpoint distance using observable history only."""

    def forward(self, observable: Mapping[str, Any]) -> dict[str, Tensor]:
        value = _observable_tensors(observable)
        last, last_time, any_visible = last_visible_state(
            value["points_history"],
            value["visibility_history"],
            value["timestamps"],
        )
        action = value["action_command"]
        endpoint_time = value["nominal_action_delay"][:, None] + action[:, 6:7]
        delta = endpoint_time - last_time
        future = last + value["estimated_velocity"] * delta[:, :, None]
        distance = torch.linalg.vector_norm(future - action[:, None, 3:6], dim=-1)
        logits = (action[:, 7:8] - distance) / self.temperature
        logits = torch.where(any_visible, logits, torch.full_like(logits, -8.0))
        return self._output(logits, future[:, :, None, :])


class EstimatedTimeAlignedTrajectoryProximity(_ObservableBaseline):
    """Synchronous constant-velocity extrapolation from noisy history."""

    def forward(self, observable: Mapping[str, Any]) -> dict[str, Tensor]:
        value = _observable_tensors(observable)
        last, last_time, any_visible = last_visible_state(
            value["points_history"],
            value["visibility_history"],
            value["timestamps"],
        )
        action = value["action_command"]
        times, gripper, _ = _trajectory(
            action, value["nominal_action_delay"], self.horizon_steps
        )
        elapsed = times[:, None, :] - last_time[:, :, None]
        future = last[:, :, None, :] + value["estimated_velocity"][:, :, None, :] * elapsed[
            :, :, :, None
        ]
        distance = torch.linalg.vector_norm(future - gripper[:, None, :, :], dim=-1)
        minimum = distance.amin(dim=-1)
        logits = (action[:, 7:8] - minimum) / self.temperature
        logits = torch.where(any_visible, logits, torch.full_like(logits, -8.0))
        return self._output(logits, future)


class ConstantVelocityKalmanProximity(_ObservableBaseline):
    """Dependency-free alpha-beta style velocity smoothing and extrapolation."""

    def __init__(
        self,
        *,
        temperature: float = 0.035,
        horizon_steps: int = FUTURE_STEPS,
        velocity_gain: float = 0.62,
    ) -> None:
        super().__init__(temperature=temperature, horizon_steps=horizon_steps)
        if not 0.0 < velocity_gain <= 1.0:
            raise ValueError("velocity_gain must lie in (0,1]")
        self.velocity_gain = float(velocity_gain)

    def forward(self, observable: Mapping[str, Any]) -> dict[str, Tensor]:
        value = _observable_tensors(observable)
        history = value["points_history"]
        visibility = value["visibility_history"].bool()
        timestamps = value["timestamps"]
        last, last_time, any_visible = last_visible_state(history, visibility, timestamps)
        velocity = torch.zeros_like(value["estimated_velocity"])
        initialized = torch.zeros_like(any_visible)
        for frame in range(1, history.shape[1]):
            valid = torch.logical_and(visibility[:, frame], visibility[:, frame - 1])
            dt = (timestamps[:, frame] - timestamps[:, frame - 1]).clamp_min(1.0e-4)
            measurement = (history[:, frame] - history[:, frame - 1]) / dt[:, None, None]
            updated = torch.where(
                initialized[:, :, None],
                (1.0 - self.velocity_gain) * velocity + self.velocity_gain * measurement,
                measurement,
            )
            velocity = torch.where(valid[:, :, None], updated, velocity)
            initialized = torch.logical_or(initialized, valid)
        velocity = torch.where(
            initialized[:, :, None], velocity, value["estimated_velocity"]
        )
        action = value["action_command"]
        times, gripper, _ = _trajectory(
            action, value["nominal_action_delay"], self.horizon_steps
        )
        elapsed = times[:, None, :] - last_time[:, :, None]
        future = last[:, :, None, :] + velocity[:, :, None, :] * elapsed[:, :, :, None]
        distance = torch.linalg.vector_norm(future - gripper[:, None, :, :], dim=-1)
        logits = (action[:, 7:8] - distance.amin(dim=-1)) / self.temperature
        logits = torch.where(any_visible, logits, torch.full_like(logits, -8.0))
        return self._output(logits, future)


class ExactHiddenTrajectoryOracle(nn.Module):
    """Excluded oracle that consumes exact hidden contact state only."""

    excluded_from_fair_comparison = True

    def forward(self, hidden_state: Mapping[str, Any]) -> dict[str, Tensor]:
        unknown = set(hidden_state) - set(HIDDEN_STATE_FIELDS)
        missing = set(HIDDEN_STATE_FIELDS) - set(hidden_state)
        if unknown or missing:
            raise ValueError(
                f"hidden oracle fields mismatch; missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        contact = hidden_state["exact_contact_state"]
        if not isinstance(contact, Tensor) or contact.ndim != 2:
            raise ValueError("exact_contact_state must have shape [B,N]")
        logits = torch.where(
            contact.bool(), torch.full_like(contact, 12.0, dtype=torch.float32),
            torch.full_like(contact, -12.0, dtype=torch.float32)
        )
        return {
            "mask_logits": logits,
            "utility_score": contact.to(torch.float32).mean(dim=1),
        }


FAIR_BASELINES = {
    "CurrentPositionProximity": CurrentPositionProximity,
    "EstimatedVelocityFutureProximity": EstimatedVelocityFutureProximity,
    "EstimatedTimeAlignedTrajectoryProximity": EstimatedTimeAlignedTrajectoryProximity,
    "ConstantVelocityKalmanProximity": ConstantVelocityKalmanProximity,
}


__all__ = [
    "ConstantVelocityKalmanProximity",
    "CurrentPositionProximity",
    "EstimatedTimeAlignedTrajectoryProximity",
    "EstimatedVelocityFutureProximity",
    "ExactHiddenTrajectoryOracle",
    "FAIR_BASELINES",
]
