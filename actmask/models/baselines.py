"""Parameter-free baselines with the same forward interface as ActMask."""

from __future__ import annotations

import torch
from torch import Tensor, nn


def _validate_inputs(points: Tensor, velocities: Tensor, action: Tensor) -> None:
    for name, tensor in (
        ("points", points),
        ("velocities", velocities),
        ("action", action),
    ):
        if not isinstance(tensor, Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if not tensor.is_floating_point():
            raise TypeError(f"{name} must have a floating-point dtype")
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError(f"points must have shape [B, N, 3], got {tuple(points.shape)}")
    if velocities.shape != points.shape:
        raise ValueError("velocities must have the same [B, N, 3] shape as points")
    if action.ndim != 2 or action.shape[-1] != 8:
        raise ValueError(f"action must have shape [B, 8], got {tuple(action.shape)}")
    if action.shape[0] != points.shape[0]:
        raise ValueError("points, velocities, and action must share the batch dimension")
    if points.shape[0] == 0 or points.shape[1] == 0:
        raise ValueError("batch size and number of points must both be non-zero")
    if points.device != velocities.device or points.device != action.device:
        raise ValueError("points, velocities, and action must be on the same device")


class NoMask(nn.Module):
    """Assign the same logit to every point (0.0 means probability 0.5)."""

    def __init__(self, logit: float = 0.0) -> None:
        super().__init__()
        self.logit = float(logit)

    def forward(self, points: Tensor, velocities: Tensor, action: Tensor) -> Tensor:
        _validate_inputs(points, velocities, action)
        return torch.full(
            points.shape[:2], self.logit, dtype=points.dtype, device=points.device
        )

    def extra_repr(self) -> str:
        return f"logit={self.logit}"


class MotionMagnitudeMask(nn.Module):
    """Score points only by speed, independent of the candidate action."""

    def __init__(self, threshold: float = 0.05, temperature: float = 0.05) -> None:
        super().__init__()
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if threshold < 0.0:
            raise ValueError("threshold must be non-negative")
        self.threshold = float(threshold)
        self.temperature = float(temperature)

    def forward(self, points: Tensor, velocities: Tensor, action: Tensor) -> Tensor:
        _validate_inputs(points, velocities, action)
        speed = torch.linalg.vector_norm(velocities, dim=-1)
        return (speed - self.threshold) / self.temperature

    def extra_repr(self) -> str:
        return f"threshold={self.threshold}, temperature={self.temperature}"


class ActionProximityMask(nn.Module):
    """Score current points by distance to the finite action line segment.

    This baseline deliberately ignores point motion and action duration.  Its
    zero-logit decision boundary is the gripper radius encoded by the action.
    """

    def __init__(self, temperature: float = 0.05, eps: float = 1e-8) -> None:
        super().__init__()
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if eps <= 0.0:
            raise ValueError("eps must be positive")
        self.temperature = float(temperature)
        self.eps = float(eps)

    def forward(self, points: Tensor, velocities: Tensor, action: Tensor) -> Tensor:
        _validate_inputs(points, velocities, action)
        start = action[:, None, :3]
        end = action[:, None, 3:6]
        segment = end - start
        squared_length = torch.sum(segment * segment, dim=-1, keepdim=True).clamp_min(
            self.eps
        )
        fraction = torch.sum((points - start) * segment, dim=-1, keepdim=True)
        fraction = (fraction / squared_length).clamp(0.0, 1.0)
        closest = start + fraction * segment
        distance = torch.linalg.vector_norm(points - closest, dim=-1)
        radius = action[:, 7:8].clamp_min(0.0)
        return (radius - distance) / self.temperature

    def extra_repr(self) -> str:
        return f"temperature={self.temperature}, eps={self.eps}"


class CurrentPositionProximity(ActionProximityMask):
    """Current-position proximity to the finite candidate-action segment.

    This is the explicitly named Milestone-2 fair baseline.  It has access to
    the current observed point positions and the candidate action, but not to
    point velocity, future trajectories, labels, or simulator metadata.  The
    implementation is intentionally the same as :class:`ActionProximityMask`
    so the legacy milestone-1 name remains fully compatible.
    """




def _validate_future_baseline_parameters(
    temperature: float, eps: float, trajectory_steps: int | None = None
) -> tuple[float, float, int | None]:
    temperature = float(temperature)
    eps = float(eps)
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if eps <= 0.0:
        raise ValueError("eps must be positive")
    if trajectory_steps is not None:
        if (
            not isinstance(trajectory_steps, int)
            or isinstance(trajectory_steps, bool)
            or trajectory_steps < 2
        ):
            raise ValueError("trajectory_steps must be an integer of at least 2")
    return temperature, eps, trajectory_steps


class FutureEndpointProximityMask(nn.Module):
    """Compare each constant-velocity future endpoint with the action endpoint.

    This deployable diagnostic uses only the public observation and action
    tensors.  It does not receive nonlinear ground-truth trajectories.
    """

    def __init__(self, temperature: float = 0.05, eps: float = 1e-8) -> None:
        super().__init__()
        self.temperature, self.eps, _ = _validate_future_baseline_parameters(
            temperature, eps
        )

    def forward(self, points: Tensor, velocities: Tensor, action: Tensor) -> Tensor:
        _validate_inputs(points, velocities, action)
        duration = action[:, None, 6:7].clamp_min(0.0)
        predicted_endpoint = points + velocities * duration
        action_endpoint = action[:, None, 3:6]
        distance = torch.linalg.vector_norm(predicted_endpoint - action_endpoint, dim=-1)
        radius = action[:, 7:8].clamp_min(0.0)
        return (radius - distance) / self.temperature

    def extra_repr(self) -> str:
        return f"temperature={self.temperature}, eps={self.eps}"


class FuturePositionProximity(FutureEndpointProximityMask):
    """Constant-velocity future endpoint proximity baseline.

    It uses only observed ``points``, observed ``velocities``, and candidate
    action duration/end position.  It is a deployable geometric baseline, not
    an oracle: nonlinear acceleration, curvature, and delayed onset are not
    available to it.
    """




class MinFutureSpatialDistanceMask(nn.Module):
    """Minimum *untimed* distance from predicted point futures to the action path.

    Constant-velocity point positions are sampled over the action duration.
    Each sampled position is compared with the whole finite action segment, so
    this method deliberately ignores whether point and gripper reach their
    closest spatial locations at the same time.
    """

    def __init__(
        self,
        trajectory_steps: int = 24,
        temperature: float = 0.05,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.temperature, self.eps, validated_steps = (
            _validate_future_baseline_parameters(temperature, eps, trajectory_steps)
        )
        assert validated_steps is not None
        self.trajectory_steps = validated_steps

    def forward(self, points: Tensor, velocities: Tensor, action: Tensor) -> Tensor:
        _validate_inputs(points, velocities, action)
        fractions = torch.linspace(
            0.0,
            1.0,
            self.trajectory_steps,
            dtype=points.dtype,
            device=points.device,
        )
        duration = action[:, None, None, 6:7].clamp_min(0.0)
        future_points = (
            points[:, :, None, :]
            + velocities[:, :, None, :] * fractions[None, None, :, None] * duration
        )
        start = action[:, None, None, :3]
        segment = action[:, None, None, 3:6] - start
        squared_length = torch.sum(segment * segment, dim=-1, keepdim=True).clamp_min(
            self.eps
        )
        projection = torch.sum((future_points - start) * segment, dim=-1, keepdim=True)
        projection = (projection / squared_length).clamp(0.0, 1.0)
        closest = start + projection * segment
        minimum_distance = torch.linalg.vector_norm(
            future_points - closest, dim=-1
        ).amin(dim=-1)
        radius = action[:, 7:8].clamp_min(0.0)
        return (radius - minimum_distance) / self.temperature

    def extra_repr(self) -> str:
        return (
            f"trajectory_steps={self.trajectory_steps}, "
            f"temperature={self.temperature}, eps={self.eps}"
        )


class TimeAlignedFutureProximityMask(nn.Module):
    """Sample synchronous constant-velocity point and gripper trajectories.

    This is the strongest sampled geometric baseline with the same information
    as ActMask.  On the milestone-1 toy generator, using the same number of
    trajectory samples reproduces its label rule exactly.
    """

    def __init__(
        self,
        trajectory_steps: int = 24,
        temperature: float = 0.05,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.temperature, self.eps, validated_steps = (
            _validate_future_baseline_parameters(temperature, eps, trajectory_steps)
        )
        assert validated_steps is not None
        self.trajectory_steps = validated_steps

    def forward(self, points: Tensor, velocities: Tensor, action: Tensor) -> Tensor:
        _validate_inputs(points, velocities, action)
        fractions = torch.linspace(
            0.0,
            1.0,
            self.trajectory_steps,
            dtype=points.dtype,
            device=points.device,
        )
        duration = action[:, None, None, 6:7].clamp_min(0.0)
        point_trajectory = (
            points[:, :, None, :]
            + velocities[:, :, None, :] * fractions[None, None, :, None] * duration
        )
        start = action[:, None, None, :3]
        gripper_trajectory = start + fractions[None, None, :, None] * (
            action[:, None, None, 3:6] - start
        )
        minimum_distance = torch.linalg.vector_norm(
            point_trajectory - gripper_trajectory, dim=-1
        ).amin(dim=-1)
        radius = action[:, 7:8].clamp_min(0.0)
        return (radius - minimum_distance) / self.temperature

    def extra_repr(self) -> str:
        return (
            f"trajectory_steps={self.trajectory_steps}, "
            f"temperature={self.temperature}, eps={self.eps}"
        )


class TimeAlignedTrajectoryProximity(TimeAlignedFutureProximityMask):
    """Synchronous constant-velocity point/gripper trajectory proximity.

    This fair baseline has the same deployable inputs as ActMask, but assumes
    constant point velocity and a straight constant-speed candidate action.
    It deliberately cannot receive exact simulated trajectories or labels.
    """


class AnalyticClosestApproachMask(nn.Module):
    """Continuous-time closest approach under constant relative velocity.

    For a straight constant-speed gripper, relative point motion is linear.
    The minimizing time has a closed form and is clamped to the action interval,
    avoiding temporal discretization while retaining the same observed-input
    information budget as the learned model.
    """

    def __init__(self, temperature: float = 0.05, eps: float = 1e-8) -> None:
        super().__init__()
        self.temperature, self.eps, _ = _validate_future_baseline_parameters(
            temperature, eps
        )

    def forward(self, points: Tensor, velocities: Tensor, action: Tensor) -> Tensor:
        _validate_inputs(points, velocities, action)
        start = action[:, None, :3]
        end = action[:, None, 3:6]
        duration = action[:, None, 6].clamp_min(self.eps)
        gripper_velocity = (end - start) / duration[:, :, None]
        relative_position = points - start
        relative_velocity = velocities - gripper_velocity
        squared_speed = torch.sum(relative_velocity * relative_velocity, dim=-1)
        closest_time = -torch.sum(
            relative_position * relative_velocity, dim=-1
        ) / squared_speed.clamp_min(self.eps)
        closest_time = closest_time.clamp_min(0.0)
        closest_time = torch.minimum(closest_time, duration)
        closest_offset = relative_position + relative_velocity * closest_time[:, :, None]
        minimum_distance = torch.linalg.vector_norm(closest_offset, dim=-1)
        radius = action[:, 7:8].clamp_min(0.0)
        return (radius - minimum_distance) / self.temperature

    def extra_repr(self) -> str:
        return f"temperature={self.temperature}, eps={self.eps}"


__all__ = [
    "NoMask",
    "MotionMagnitudeMask",
    "ActionProximityMask",
    "CurrentPositionProximity",
    "FutureEndpointProximityMask",
    "FuturePositionProximity",
    "MinFutureSpatialDistanceMask",
    "TimeAlignedFutureProximityMask",
    "TimeAlignedTrajectoryProximity",
    "AnalyticClosestApproachMask",
]
