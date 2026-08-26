"""Lightweight history-conditioned ActMask models for Milestone 2B."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Final

import torch
from torch import Tensor, nn

from actmask.data.milestone2b_dataset import (
    ACTION_DIM,
    FUTURE_STEPS,
    OBSERVABLE_FIELDS,
    validate_observable_mapping,
)


TEMPORAL_FEATURE_MODES: Final[tuple[str, ...]] = (
    "full",
    "no_history",
    "no_action",
    "no_visibility",
)


def _validate_batched_observable(observable: Mapping[str, Any]) -> dict[str, Tensor]:
    validate_observable_mapping(observable)
    result: dict[str, Tensor] = {}
    for key in OBSERVABLE_FIELDS:
        value = observable[key]
        if not isinstance(value, Tensor):
            raise TypeError(f"observable[{key!r}] must be a torch.Tensor")
        result[key] = value
    history = result["points_history"]
    visibility = result["visibility_history"]
    timestamps = result["timestamps"]
    estimated_velocity = result["estimated_velocity"]
    confidence = result["velocity_confidence"]
    action = result["action_command"]
    if history.ndim != 4 or history.shape[-1] != 3:
        raise ValueError("points_history must have shape [B,K,N,3]")
    if visibility.shape != history.shape[:3]:
        raise ValueError("visibility_history must have shape [B,K,N]")
    if timestamps.shape != history.shape[:2]:
        raise ValueError("timestamps must have shape [B,K]")
    if estimated_velocity.shape != (history.shape[0], history.shape[2], 3):
        raise ValueError("estimated_velocity must have shape [B,N,3]")
    if confidence.shape != history.shape[::2]:
        # history.shape[::2] is (B,N) for [B,K,N,3].
        raise ValueError("velocity_confidence must have shape [B,N]")
    if action.shape != (history.shape[0], ACTION_DIM):
        raise ValueError(f"action_command must have shape [B,{ACTION_DIM}]")
    for scalar_name in ("nominal_action_delay", "observation_delay"):
        scalar = result[scalar_name]
        if scalar.ndim == 2 and scalar.shape[1] == 1:
            result[scalar_name] = scalar[:, 0]
            scalar = result[scalar_name]
        if scalar.shape != (history.shape[0],):
            raise ValueError(f"{scalar_name} must have shape [B]")
    if history.shape[1] < 2 or history.shape[2] == 0 or history.shape[0] == 0:
        raise ValueError("history needs a non-empty batch, at least two frames, and points")
    devices = {tensor.device for tensor in result.values()}
    if len(devices) != 1:
        raise ValueError("all observable tensors must share one device")
    return result


def last_visible_state(
    points_history: Tensor,
    visibility_history: Tensor,
    timestamps: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return last visible positions, their times, and any-visible flags."""

    batch, frames, points, _ = points_history.shape
    last = points_history.new_zeros((batch, points, 3))
    last_time = timestamps.new_zeros((batch, points))
    any_visible = torch.zeros(
        (batch, points), dtype=torch.bool, device=points_history.device
    )
    for frame in range(frames):
        visible = visibility_history[:, frame].bool()
        last = torch.where(visible[:, :, None], points_history[:, frame], last)
        last_time = torch.where(
            visible, timestamps[:, frame, None].expand(batch, points), last_time
        )
        any_visible = torch.logical_or(any_visible, visible)
    return last, last_time, any_visible


def estimated_acceleration_from_history(
    points_history: Tensor,
    visibility_history: Tensor,
    timestamps: Tensor,
) -> Tensor:
    """Finite-difference acceleration using only consecutive visible frames."""

    if points_history.shape[1] < 3:
        return torch.zeros_like(points_history[:, 0])
    dt = (timestamps[:, 1:] - timestamps[:, :-1]).clamp_min(1.0e-4)
    segment_velocity = (
        points_history[:, 1:] - points_history[:, :-1]
    ) / dt[:, :, None, None]
    pair_visible = torch.logical_and(
        visibility_history[:, 1:].bool(), visibility_history[:, :-1].bool()
    )
    midpoint = 0.5 * (timestamps[:, 1:] + timestamps[:, :-1])
    midpoint_dt = (midpoint[:, 1:] - midpoint[:, :-1]).clamp_min(1.0e-4)
    acceleration = (
        segment_velocity[:, 1:] - segment_velocity[:, :-1]
    ) / midpoint_dt[:, :, None, None]
    triple_visible = torch.logical_and(pair_visible[:, 1:], pair_visible[:, :-1])
    weights = triple_visible.to(acceleration.dtype)
    numerator = (acceleration * weights[:, :, :, None]).sum(dim=1)
    denominator = weights.sum(dim=1).clamp_min(1.0)[:, :, None]
    return numerator / denominator


def _future_grid(action: Tensor, delay: Tensor, steps: int) -> tuple[Tensor, Tensor, Tensor]:
    fractions = torch.linspace(
        0.0, 1.0, steps, dtype=action.dtype, device=action.device
    )
    duration = action[:, 6].clamp_min(1.0e-4)
    execution_time = delay[:, None] + fractions[None, :] * duration[:, None]
    gripper = action[:, None, :3] + fractions[None, :, None] * (
        action[:, None, 3:6] - action[:, None, :3]
    )
    return execution_time, gripper, fractions


def _safe_vector_norm(value: Tensor, *, dim: int = -1, keepdim: bool = False) -> Tensor:
    """Finite norm with a well-defined derivative at the zero vector."""

    return torch.sqrt(value.square().sum(dim=dim, keepdim=keepdim).clamp_min(1.0e-12))


class ActMaskMLP(nn.Module):
    """Original concatenation-style MLP adapted to the observable 2B contract."""

    def __init__(self, hidden_dim: int = 48, utility_hidden_dim: int = 32) -> None:
        super().__init__()
        if hidden_dim <= 0 or utility_hidden_dim <= 0:
            raise ValueError("hidden dimensions must be positive")
        # last xyz, estimated velocity, point-to-start/end, action[8],
        # velocity confidence, nominal action delay, observation delay.
        self.point_encoder = nn.Sequential(
            nn.Linear(23, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mask_head = nn.Linear(hidden_dim, 1)
        self.utility_head = nn.Sequential(
            nn.Linear(2 * hidden_dim + ACTION_DIM + 3, utility_hidden_dim),
            nn.ReLU(),
            nn.Linear(utility_hidden_dim, 1),
        )

    def forward(self, observable: Mapping[str, Any]) -> dict[str, Tensor]:
        values = _validate_batched_observable(observable)
        history = values["points_history"]
        visibility = values["visibility_history"]
        timestamps = values["timestamps"]
        velocity = values["estimated_velocity"]
        confidence = values["velocity_confidence"]
        action = values["action_command"]
        nominal_delay = values["nominal_action_delay"]
        observation_delay = values["observation_delay"]
        last, last_time, any_visible = last_visible_state(history, visibility, timestamps)
        current = last + velocity * (-last_time).clamp_min(0.0)[:, :, None]
        batch, points, _ = current.shape
        expanded_action = action[:, None, :].expand(batch, points, ACTION_DIM)
        scalar = torch.stack(
            (
                confidence,
                nominal_delay[:, None].expand(batch, points),
                observation_delay[:, None].expand(batch, points),
            ),
            dim=-1,
        )
        features = torch.cat(
            (
                current,
                velocity,
                current - action[:, None, :3],
                current - action[:, None, 3:6],
                expanded_action,
                scalar,
            ),
            dim=-1,
        )
        encoded = self.point_encoder(features)
        encoded = encoded * any_visible[:, :, None].to(encoded.dtype)
        mask_logits = self.mask_head(encoded).squeeze(-1)
        global_features = torch.cat(
            (
                encoded.mean(dim=1),
                encoded.amax(dim=1),
                action,
                nominal_delay[:, None],
                observation_delay[:, None],
                confidence.mean(dim=1, keepdim=True),
            ),
            dim=-1,
        )
        utility_logits = self.utility_head(global_features).squeeze(-1)

        execution_time, _, _ = _future_grid(action, nominal_delay, FUTURE_STEPS)
        future = current[:, :, None, :] + velocity[:, :, None, :] * execution_time[
            :, None, :, None
        ]
        contact_logits = mask_logits[:, :, None].expand(
            batch, points, FUTURE_STEPS
        )
        return {
            "mask_logits": mask_logits,
            "contact_logits": contact_logits,
            "utility_logits": utility_logits,
            "future_hypotheses": future,
            "estimated_current_points": current,
        }


class TemporalActMask(nn.Module):
    """Explicitly align learned point futures with the candidate trajectory."""

    def __init__(
        self,
        *,
        observation_frames: int = 4,
        horizon_steps: int = FUTURE_STEPS,
        history_hidden_dim: int = 32,
        action_hidden_dim: int = 20,
        fusion_hidden_dim: int = 56,
        contact_hidden_dim: int = 32,
        utility_hidden_dim: int = 32,
        utility_head: bool = True,
    ) -> None:
        super().__init__()
        for name, value in (
            ("observation_frames", observation_frames),
            ("horizon_steps", horizon_steps),
            ("history_hidden_dim", history_hidden_dim),
            ("action_hidden_dim", action_hidden_dim),
            ("fusion_hidden_dim", fusion_hidden_dim),
            ("contact_hidden_dim", contact_hidden_dim),
            ("utility_hidden_dim", utility_hidden_dim),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if observation_frames < 2 or horizon_steps < 4:
            raise ValueError("TemporalActMask needs at least 2 history and 4 future steps")
        self.observation_frames = observation_frames
        self.horizon_steps = horizon_steps
        self.utility_head_enabled = bool(utility_head)

        self.history_encoder = nn.Sequential(
            nn.Linear(observation_frames * 5, history_hidden_dim),
            nn.ReLU(),
            nn.Linear(history_hidden_dim, history_hidden_dim),
            nn.ReLU(),
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(ACTION_DIM + 2, action_hidden_dim),
            nn.ReLU(),
            nn.Linear(action_hidden_dim, action_hidden_dim),
            nn.ReLU(),
        )
        # relative position[3], distance, relative velocity[3], normalized
        # execution time, active flag, confidence, visibility fraction.
        self.contact_encoder = nn.Sequential(
            nn.Linear(11, contact_hidden_dim),
            nn.ReLU(),
            nn.Linear(contact_hidden_dim, 1),
        )
        local_dim = (
            history_hidden_dim
            + action_hidden_dim
            + 3  # estimated current
            + 3  # velocity
            + 3  # acceleration
            + 6  # point relative to start/end
            + 6  # contact aggregate statistics
            + 2  # confidence and visibility
        )
        self.mask_head = nn.Sequential(
            nn.Linear(local_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.Linear(fusion_hidden_dim, 1),
        )
        # Observable-only analytic prior: the temporal network starts from the
        # distance between its own history-derived future hypotheses and the
        # nominal candidate trajectory, then learns a residual correction.
        # This is not an oracle field and remains sensitive to history/action
        # ablations by construction.
        self.geometric_prior_scale = nn.Parameter(torch.tensor(1.0))
        self.geometric_prior_temperature = 0.045
        self.residual_logit_scale = nn.Parameter(torch.tensor(0.10))
        self.contact_logit_scale = nn.Parameter(torch.tensor(0.10))
        self.acceleration_scale_logit = nn.Parameter(torch.tensor(-3.0))
        if self.utility_head_enabled:
            self.utility_head: nn.Module | None = nn.Sequential(
                nn.Linear(action_hidden_dim + 9, utility_hidden_dim),
                nn.ReLU(),
                nn.Linear(utility_hidden_dim, 1),
            )
        else:
            self.utility_head = None

    def forward(
        self,
        observable: Mapping[str, Any],
        *,
        feature_mode: str = "full",
    ) -> dict[str, Tensor]:
        if feature_mode not in TEMPORAL_FEATURE_MODES:
            raise ValueError(
                f"feature_mode must be one of {TEMPORAL_FEATURE_MODES}, got {feature_mode!r}"
            )
        values = _validate_batched_observable(observable)
        history = values["points_history"]
        visibility = values["visibility_history"].bool()
        timestamps = values["timestamps"]
        velocity = values["estimated_velocity"]
        confidence = values["velocity_confidence"]
        action = values["action_command"]
        nominal_delay = values["nominal_action_delay"]
        observation_delay = values["observation_delay"]
        batch, frames, points, _ = history.shape
        if frames != self.observation_frames:
            raise ValueError(
                f"expected {self.observation_frames} history frames, got {frames}"
            )

        # The no-visibility ablation removes the mask/confidence everywhere,
        # including state selection and finite differences.  Missing entries
        # therefore remain zero-valued observations rather than benefiting
        # from a hidden use of the true visibility mask.
        if feature_mode == "no_visibility":
            visibility = torch.ones_like(visibility)
            confidence = torch.ones_like(confidence)
        if feature_mode == "no_history":
            last = history[:, -1]
            last_time = timestamps[:, -1, None].expand(batch, points)
            any_visible = visibility[:, -1]
            velocity = torch.zeros_like(velocity)
            acceleration = torch.zeros_like(velocity)
            confidence = torch.zeros_like(confidence)
            history_embedding = history.new_zeros(
                (batch, points, self.history_encoder[-2].out_features)
            )
        else:
            last, last_time, any_visible = last_visible_state(
                history, visibility, timestamps
            )
            acceleration = estimated_acceleration_from_history(
                history, visibility, timestamps
            )
            acceleration = (
                1.5
                * torch.tanh(acceleration / 1.5)
                * confidence[:, :, None]
            )
            relative_history = history - last[:, None, :, :]
            visible_feature = visibility.to(history.dtype)
            time_feature = timestamps[:, :, None].expand(batch, frames, points)
            history_features = torch.cat(
                (
                    relative_history,
                    visible_feature[:, :, :, None],
                    time_feature[:, :, :, None],
                ),
                dim=-1,
            ).permute(0, 2, 1, 3)
            history_embedding = self.history_encoder(
                history_features.reshape(batch, points, frames * 5)
            )

        effective_acceleration = acceleration * torch.sigmoid(
            self.acceleration_scale_logit
        )
        extrapolation_time = (-last_time).clamp_min(0.0)
        current = (
            last
            + velocity * extrapolation_time[:, :, None]
            + 0.5 * effective_acceleration * extrapolation_time[:, :, None] ** 2
        )
        action_features = torch.cat(
            (action, nominal_delay[:, None], observation_delay[:, None]), dim=-1
        )
        if feature_mode == "no_action":
            action_embedding = history.new_zeros(
                (batch, self.action_encoder[-2].out_features)
            )
            execution_time = torch.linspace(
                0.0,
                1.0,
                self.horizon_steps,
                dtype=history.dtype,
                device=history.device,
            )[None, :].expand(batch, self.horizon_steps)
            gripper = torch.zeros(
                (batch, self.horizon_steps, 3),
                dtype=history.dtype,
                device=history.device,
            )
            fractions = torch.linspace(
                0.0,
                1.0,
                self.horizon_steps,
                dtype=history.dtype,
                device=history.device,
            )
        else:
            action_embedding = self.action_encoder(action_features)
            execution_time, gripper, fractions = _future_grid(
                action, nominal_delay, self.horizon_steps
            )

        future = (
            current[:, :, None, :]
            + velocity[:, :, None, :] * execution_time[:, None, :, None]
            + 0.5
            * effective_acceleration[:, :, None, :]
            * execution_time[:, None, :, None] ** 2
        )
        relative = future - gripper[:, None, :, :]
        distance = torch.linalg.vector_norm(relative, dim=-1)
        duration = action[:, 6].clamp_min(1.0e-4)
        gripper_velocity = (action[:, 3:6] - action[:, :3]) / duration[:, None]
        future_velocity = velocity[:, :, None, :] + effective_acceleration[:, :, None, :] * execution_time[
            :, None, :, None
        ]
        relative_velocity = future_velocity - gripper_velocity[:, None, None, :]
        normalized_time = fractions[None, None, :, None].expand(
            batch, points, self.horizon_steps, 1
        )
        active = torch.ones_like(normalized_time)
        visibility_fraction = visibility.to(history.dtype).mean(dim=1)
        contact_features = torch.cat(
            (
                relative,
                distance[:, :, :, None],
                relative_velocity,
                normalized_time,
                active,
                confidence[:, :, None, None].expand(
                    batch, points, self.horizon_steps, 1
                ),
                visibility_fraction[:, :, None, None].expand(
                    batch, points, self.horizon_steps, 1
                ),
            ),
            dim=-1,
        )
        if feature_mode == "no_action":
            contact_logits = torch.zeros(
                (batch, points, self.horizon_steps),
                dtype=history.dtype,
                device=history.device,
            )
            distance_features = history.new_zeros((batch, points, 6))
            relative_start = history.new_zeros((batch, points, 3))
            relative_end = history.new_zeros((batch, points, 3))
        else:
            contact_logits = self.contact_encoder(contact_features).squeeze(-1)
            closest_index = distance.argmin(dim=-1)
            closest_time = fractions[closest_index]
            distance_features = torch.stack(
                (
                    distance.amin(dim=-1),
                    distance.mean(dim=-1),
                    torch.sigmoid(contact_logits).amax(dim=-1),
                    torch.sigmoid(contact_logits).mean(dim=-1),
                    closest_time,
                    torch.linalg.vector_norm(relative_velocity, dim=-1).mean(dim=-1),
                ),
                dim=-1,
            )
            relative_start = current - action[:, None, :3]
            relative_end = current - action[:, None, 3:6]

        expanded_action = action_embedding[:, None, :].expand(
            batch, points, action_embedding.shape[-1]
        )
        local = torch.cat(
            (
                history_embedding,
                expanded_action,
                current,
                velocity,
                effective_acceleration,
                relative_start,
                relative_end,
                distance_features,
                confidence[:, :, None],
                visibility_fraction[:, :, None],
            ),
            dim=-1,
        )
        residual_logits = self.mask_head(local).squeeze(-1)
        contact_aggregate = torch.logsumexp(contact_logits, dim=-1) - math.log(
            self.horizon_steps
        )
        if feature_mode == "no_action":
            geometric_prior = torch.zeros_like(residual_logits)
        else:
            geometric_prior = (
                action[:, 7:8] - distance.amin(dim=-1)
            ) / self.geometric_prior_temperature
        mask_logits = (
            self.residual_logit_scale * residual_logits
            + self.contact_logit_scale * contact_aggregate
            + self.geometric_prior_scale * geometric_prior
        )
        mask_logits = torch.where(
            any_visible, mask_logits, self.residual_logit_scale * residual_logits
        )

        probability = torch.sigmoid(mask_logits)
        topk = min(8, points)
        topk_mean = probability.topk(topk, dim=1).values.mean(dim=1)
        utility_features = torch.cat(
            (
                action_embedding,
                probability.mean(dim=1, keepdim=True),
                probability.amax(dim=1, keepdim=True),
                probability.sum(dim=1, keepdim=True) / float(points),
                topk_mean[:, None],
                torch.sigmoid(contact_logits).amax(dim=(1, 2), keepdim=False)[:, None],
                torch.sigmoid(contact_logits).mean(dim=(1, 2), keepdim=False)[:, None],
                confidence.mean(dim=1, keepdim=True),
                visibility_fraction.mean(dim=1, keepdim=True),
                observation_delay[:, None],
            ),
            dim=-1,
        )
        if self.utility_head is not None:
            utility_logits = self.utility_head(utility_features).squeeze(-1)
        else:
            utility_logits = torch.logit(topk_mean.clamp(1.0e-5, 1.0 - 1.0e-5))
        return {
            "mask_logits": mask_logits,
            "contact_logits": contact_logits,
            "utility_logits": utility_logits,
            "future_hypotheses": future,
            "estimated_current_points": current,
            "estimated_acceleration": effective_acceleration,
            "time_aligned_distance": distance,
        }



class RotationInvariantTemporalActMask(nn.Module):
    """Temporal ActMask backbone using only rotation-invariant learned inputs.

    Vectors are used only to form norms, distances, and dot products before an
    MLP sees them.  Predicted future points remain vector-valued outputs and
    therefore transform equivariantly with the observations.
    """

    def __init__(
        self,
        *,
        observation_frames: int = 4,
        horizon_steps: int = FUTURE_STEPS,
        history_hidden_dim: int = 32,
        action_hidden_dim: int = 20,
        fusion_hidden_dim: int = 56,
        contact_hidden_dim: int = 32,
        utility_hidden_dim: int = 32,
        utility_head: bool = True,
    ) -> None:
        super().__init__()
        if observation_frames < 2 or horizon_steps < 4:
            raise ValueError("RotationInvariantTemporalActMask needs K>=2 and H>=4")
        self.observation_frames = int(observation_frames)
        self.horizon_steps = int(horizon_steps)
        self.utility_head_enabled = bool(utility_head)
        self.history_encoder = nn.Sequential(
            nn.Linear(self.observation_frames * 4, history_hidden_dim), nn.ReLU(),
            nn.Linear(history_hidden_dim, history_hidden_dim), nn.ReLU(),
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(5, action_hidden_dim), nn.ReLU(),
            nn.Linear(action_hidden_dim, action_hidden_dim), nn.ReLU(),
        )
        self.contact_encoder = nn.Sequential(
            nn.Linear(7, contact_hidden_dim), nn.ReLU(),
            nn.Linear(contact_hidden_dim, 1),
        )
        self.mask_head = nn.Sequential(
            nn.Linear(history_hidden_dim + action_hidden_dim + 15, fusion_hidden_dim), nn.ReLU(),
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim), nn.ReLU(),
            nn.Linear(fusion_hidden_dim, 1),
        )
        self.geometric_prior_scale = nn.Parameter(torch.tensor(1.0))
        self.geometric_prior_temperature = 0.045
        self.residual_logit_scale = nn.Parameter(torch.tensor(0.10))
        self.contact_logit_scale = nn.Parameter(torch.tensor(0.10))
        self.acceleration_scale_logit = nn.Parameter(torch.tensor(-3.0))
        self.utility_head: nn.Module | None
        self.utility_head = nn.Sequential(
            nn.Linear(action_hidden_dim + 9, utility_hidden_dim), nn.ReLU(),
            nn.Linear(utility_hidden_dim, 1),
        ) if self.utility_head_enabled else None

    def forward(self, observable: Mapping[str, Any], *, feature_mode: str = "full") -> dict[str, Tensor]:
        if feature_mode not in TEMPORAL_FEATURE_MODES:
            raise ValueError(f"unknown feature mode {feature_mode!r}")
        values = _validate_batched_observable(observable)
        history = values["points_history"]
        visibility = values["visibility_history"].bool()
        timestamps = values["timestamps"]
        velocity = values["estimated_velocity"]
        confidence = values["velocity_confidence"]
        action = values["action_command"]
        nominal_delay = values["nominal_action_delay"]
        observation_delay = values["observation_delay"]
        batch, frames, points, _ = history.shape
        if frames != self.observation_frames:
            raise ValueError(f"expected {self.observation_frames} history frames, got {frames}")
        if feature_mode == "no_visibility":
            visibility = torch.ones_like(visibility)
            confidence = torch.ones_like(confidence)
        if feature_mode == "no_history":
            last = history[:, -1]
            last_time = timestamps[:, -1, None].expand(batch, points)
            any_visible = visibility[:, -1]
            velocity = torch.zeros_like(velocity)
            acceleration = torch.zeros_like(velocity)
            confidence = torch.zeros_like(confidence)
            history_embedding = history.new_zeros((batch, points, self.history_encoder[-2].out_features))
        else:
            last, last_time, any_visible = last_visible_state(history, visibility, timestamps)
            acceleration = estimated_acceleration_from_history(history, visibility, timestamps)
            # ``torch.linalg.vector_norm`` has an undefined derivative at an
            # exactly zero vector.  A zero-motion point is common in the
            # synthetic clouds, so use a clamped squared norm to keep the
            # invariant feature path finite during CPU training.
            acceleration_norm = _safe_vector_norm(acceleration, keepdim=True)
            acceleration = acceleration * (
                1.5 * torch.tanh(acceleration_norm / 1.5) / acceleration_norm.clamp_min(1.0e-6)
            ) * confidence[:, :, None]
            delta = action[:, 3:6] - action[:, :3]
            length = torch.linalg.vector_norm(delta, dim=-1).clamp_min(1.0e-6)
            axis = delta / length[:, None]
            relative_history = history - last[:, None]
            history_axis = axis if feature_mode != "no_action" else torch.zeros_like(axis)
            history_features = torch.stack((
                _safe_vector_norm(relative_history),
                (relative_history * history_axis[:, None, None]).sum(dim=-1),
                visibility.to(history.dtype),
                timestamps[:, :, None].expand(batch, frames, points),
            ), dim=-1).permute(0, 2, 1, 3)
            history_embedding = self.history_encoder(history_features.reshape(batch, points, frames * 4))
        delta = action[:, 3:6] - action[:, :3]
        action_length = torch.linalg.vector_norm(delta, dim=-1).clamp_min(1.0e-6)
        axis = delta / action_length[:, None]
        effective_acceleration = acceleration * torch.sigmoid(self.acceleration_scale_logit)
        extrapolation_time = (-last_time).clamp_min(0.0)
        current = last + velocity * extrapolation_time[:, :, None] + 0.5 * effective_acceleration * extrapolation_time[:, :, None].square()
        visibility_fraction = visibility.to(history.dtype).mean(dim=1)
        if feature_mode == "no_action":
            axis = torch.zeros_like(axis)
            action_embedding = history.new_zeros((batch, self.action_encoder[-2].out_features))
            execution_time = torch.linspace(0.0, 1.0, self.horizon_steps, dtype=history.dtype, device=history.device)[None].expand(batch, self.horizon_steps)
            fractions = torch.linspace(0.0, 1.0, self.horizon_steps, dtype=history.dtype, device=history.device)
            gripper = torch.zeros((batch, self.horizon_steps, 3), dtype=history.dtype, device=history.device)
            action_radius = torch.zeros((batch,), dtype=history.dtype, device=history.device)
        else:
            action_features = torch.stack((action_length, action[:, 6], action[:, 7], nominal_delay, observation_delay), dim=-1)
            action_embedding = self.action_encoder(action_features)
            execution_time, gripper, fractions = _future_grid(action, nominal_delay, self.horizon_steps)
            action_radius = action[:, 7]
        future = current[:, :, None] + velocity[:, :, None] * execution_time[:, None, :, None] + 0.5 * effective_acceleration[:, :, None] * execution_time[:, None, :, None].square()
        relative = future - gripper[:, None]
        distance = _safe_vector_norm(relative)
        gripper_velocity = delta / action[:, 6].clamp_min(1.0e-6)[:, None]
        future_velocity = velocity[:, :, None] + effective_acceleration[:, :, None] * execution_time[:, None, :, None]
        relative_velocity = future_velocity - gripper_velocity[:, None, None]
        normalized_time = fractions[None, None, :, None].expand(batch, points, self.horizon_steps, 1)
        contact_features = torch.cat((
            distance[:, :, :, None],
            _safe_vector_norm(relative_velocity)[:, :, :, None],
            (relative_velocity * axis[:, None, None]).sum(dim=-1)[:, :, :, None],
            normalized_time,
            torch.ones_like(normalized_time),
            confidence[:, :, None, None].expand(batch, points, self.horizon_steps, 1),
            visibility_fraction[:, :, None, None].expand(batch, points, self.horizon_steps, 1),
        ), dim=-1)
        if feature_mode == "no_action":
            contact_logits = torch.zeros((batch, points, self.horizon_steps), dtype=history.dtype, device=history.device)
            distance_features = history.new_zeros((batch, points, 6))
            state_features = history.new_zeros((batch, points, 7))
        else:
            contact_logits = self.contact_encoder(contact_features).squeeze(-1)
            closest_index = distance.argmin(dim=-1)
            closest_time = fractions[closest_index]
            distance_features = torch.stack((
                distance.amin(dim=-1), distance.mean(dim=-1),
                torch.sigmoid(contact_logits).amax(dim=-1), torch.sigmoid(contact_logits).mean(dim=-1),
                closest_time, _safe_vector_norm(relative_velocity).mean(dim=-1),
            ), dim=-1)
            from_start = current - action[:, None, :3]
            projection = (from_start * axis[:, None]).sum(dim=-1)
            perpendicular = torch.sqrt(
                (from_start.square().sum(dim=-1) - projection.square()).clamp_min(1.0e-12)
            )
            state_features = torch.stack((
                _safe_vector_norm(velocity), (velocity * axis[:, None]).sum(dim=-1),
                _safe_vector_norm(effective_acceleration), (effective_acceleration * axis[:, None]).sum(dim=-1),
                _safe_vector_norm(from_start),
                _safe_vector_norm(current - action[:, None, 3:6]), projection, perpendicular,
            ), dim=-1)[..., :7]
        local = torch.cat((
            history_embedding, action_embedding[:, None].expand(batch, points, action_embedding.shape[-1]),
            state_features, distance_features, confidence[:, :, None], visibility_fraction[:, :, None],
        ), dim=-1)
        residual_logits = self.mask_head(local).squeeze(-1)
        contact_aggregate = torch.logsumexp(contact_logits, dim=-1) - math.log(self.horizon_steps)
        geometric_prior = (action_radius[:, None] - distance.amin(dim=-1)) / self.geometric_prior_temperature if feature_mode != "no_action" else torch.zeros_like(residual_logits)
        mask_logits = self.residual_logit_scale * residual_logits + self.contact_logit_scale * contact_aggregate + self.geometric_prior_scale * geometric_prior
        mask_logits = torch.where(any_visible, mask_logits, self.residual_logit_scale * residual_logits)
        probability = torch.sigmoid(mask_logits)
        topk = min(8, points)
        topk_mean = probability.topk(topk, dim=1).values.mean(dim=1)
        utility_features = torch.cat((
            action_embedding, probability.mean(dim=1, keepdim=True), probability.amax(dim=1, keepdim=True),
            probability.sum(dim=1, keepdim=True) / float(points), topk_mean[:, None],
            torch.sigmoid(contact_logits).amax(dim=(1, 2))[:, None], torch.sigmoid(contact_logits).mean(dim=(1, 2))[:, None],
            confidence.mean(dim=1, keepdim=True), visibility_fraction.mean(dim=1, keepdim=True), observation_delay[:, None],
        ), dim=-1)
        utility_logits = self.utility_head(utility_features).squeeze(-1) if self.utility_head is not None else torch.logit(topk_mean.clamp(1.0e-5, 1.0 - 1.0e-5))
        return {
            "mask_logits": mask_logits, "contact_logits": contact_logits, "utility_logits": utility_logits,
            "future_hypotheses": future, "estimated_current_points": current,
            "estimated_acceleration": effective_acceleration, "time_aligned_distance": distance,
        }


class ActionFrameRelativeTemporalActMask(nn.Module):
    """Action-relative temporal backbone without a privileged world axis.

    The longitudinal axis is the observable candidate trajectory.  A secondary
    radial reference is the visible scene centroid projected orthogonal to that
    trajectory.  It transforms equivariantly under any orthogonal scene
    transform, so the signed progress and centroid-relative lateral component
    consumed by the MLP remain invariant.  If the radial reference is
    degenerate, the lateral component is set to zero rather than falling back
    to a global coordinate axis.
    """

    def __init__(
        self,
        *,
        observation_frames: int = 4,
        horizon_steps: int = FUTURE_STEPS,
        history_hidden_dim: int = 32,
        action_hidden_dim: int = 20,
        fusion_hidden_dim: int = 56,
        contact_hidden_dim: int = 32,
        utility_hidden_dim: int = 32,
        utility_head: bool = True,
    ) -> None:
        super().__init__()
        if observation_frames < 2 or horizon_steps < 4:
            raise ValueError("ActionFrameRelativeTemporalActMask needs K>=2 and H>=4")
        self.observation_frames = int(observation_frames)
        self.horizon_steps = int(horizon_steps)
        self.utility_head_enabled = bool(utility_head)
        self.history_encoder = nn.Sequential(
            nn.Linear(4 * self.observation_frames, history_hidden_dim), nn.ReLU(),
            nn.Linear(history_hidden_dim, history_hidden_dim), nn.ReLU(),
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(5, action_hidden_dim), nn.ReLU(),
            nn.Linear(action_hidden_dim, action_hidden_dim), nn.ReLU(),
        )
        self.contact_encoder = nn.Sequential(
            nn.Linear(7, contact_hidden_dim), nn.ReLU(),
            nn.Linear(contact_hidden_dim, 1),
        )
        # history, candidate action, ten action-frame state features, four
        # time-aligned distance/contact features, confidence and visibility.
        self.mask_head = nn.Sequential(
            nn.Linear(history_hidden_dim + action_hidden_dim + 16, fusion_hidden_dim), nn.ReLU(),
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim), nn.ReLU(),
            nn.Linear(fusion_hidden_dim, 1),
        )
        self.utility_head: nn.Module | None = nn.Sequential(
            nn.Linear(action_hidden_dim + 9, utility_hidden_dim), nn.ReLU(),
            nn.Linear(utility_hidden_dim, 1),
        ) if self.utility_head_enabled else None
        self.geometric_prior_scale = nn.Parameter(torch.tensor(1.0))
        self.geometric_prior_temperature = 0.045
        self.residual_logit_scale = nn.Parameter(torch.tensor(0.10))
        self.contact_logit_scale = nn.Parameter(torch.tensor(0.10))
        self.acceleration_scale_logit = nn.Parameter(torch.tensor(-3.0))

    @staticmethod
    def _safe_unit(value: Tensor) -> tuple[Tensor, Tensor]:
        norm = _safe_vector_norm(value, keepdim=True)
        valid = norm > 1.0e-5
        return torch.where(valid, value / norm, torch.zeros_like(value)), valid.squeeze(-1)

    def forward(self, observable: Mapping[str, Any], *, feature_mode: str = "full") -> dict[str, Tensor]:
        if feature_mode not in TEMPORAL_FEATURE_MODES:
            raise ValueError(f"unknown feature mode {feature_mode!r}")
        values = _validate_batched_observable(observable)
        history = values["points_history"]
        visibility = values["visibility_history"].bool()
        timestamps = values["timestamps"]
        velocity = values["estimated_velocity"]
        confidence = values["velocity_confidence"]
        action = values["action_command"]
        nominal_delay = values["nominal_action_delay"]
        observation_delay = values["observation_delay"]
        batch, frames, points, _ = history.shape
        if frames != self.observation_frames:
            raise ValueError(f"expected {self.observation_frames} history frames, got {frames}")
        if feature_mode == "no_visibility":
            visibility = torch.ones_like(visibility)
            confidence = torch.ones_like(confidence)
        if feature_mode == "no_history":
            last = history[:, -1]
            last_time = timestamps[:, -1, None].expand(batch, points)
            any_visible = visibility[:, -1]
            velocity = torch.zeros_like(velocity)
            acceleration = torch.zeros_like(velocity)
            confidence = torch.zeros_like(confidence)
        else:
            last, last_time, any_visible = last_visible_state(history, visibility, timestamps)
            acceleration = estimated_acceleration_from_history(history, visibility, timestamps)
        delta = action[:, 3:6] - action[:, :3]
        action_length = _safe_vector_norm(delta).clamp_min(1.0e-6)
        axis, _ = self._safe_unit(delta)
        visible_weight = any_visible.to(history.dtype)
        centroid = (last * visible_weight[:, :, None]).sum(dim=1) / visible_weight.sum(dim=1, keepdim=True).clamp_min(1.0)
        centroid_offset = centroid - action[:, :3]
        centroid_perp = centroid_offset - (centroid_offset * axis).sum(dim=-1, keepdim=True) * axis
        radial_axis, radial_valid = self._safe_unit(centroid_perp)
        acceleration = acceleration * confidence[:, :, None]
        effective_acceleration = acceleration * torch.sigmoid(self.acceleration_scale_logit)
        extrapolation_time = (-last_time).clamp_min(0.0)
        current = last + velocity * extrapolation_time[:, :, None] + 0.5 * effective_acceleration * extrapolation_time[:, :, None].square()
        visibility_fraction = visibility.to(history.dtype).mean(dim=1)
        if feature_mode == "no_history":
            history_embedding = history.new_zeros((batch, points, self.history_encoder[-2].out_features))
        else:
            relative_history = history - last[:, None]
            # The no-action ablation must not retain the action direction
            # indirectly through the history representation.
            history_axis = axis if feature_mode != "no_action" else torch.zeros_like(axis)
            progress_history = (relative_history * history_axis[:, None, None]).sum(dim=-1)
            radial_history = _safe_vector_norm(
                relative_history - progress_history[:, :, :, None] * history_axis[:, None, None]
            )
            history_features = torch.stack((
                progress_history, radial_history, visibility.to(history.dtype),
                timestamps[:, :, None].expand(batch, frames, points),
            ), dim=-1).permute(0, 2, 1, 3)
            history_embedding = self.history_encoder(history_features.reshape(batch, points, frames * 4))
        if feature_mode == "no_action":
            action_embedding = history.new_zeros((batch, self.action_encoder[-2].out_features))
            execution_time = torch.linspace(0.0, 1.0, self.horizon_steps, dtype=history.dtype, device=history.device)[None].expand(batch, self.horizon_steps)
            fractions = torch.linspace(0.0, 1.0, self.horizon_steps, dtype=history.dtype, device=history.device)
            gripper = torch.zeros((batch, self.horizon_steps, 3), dtype=history.dtype, device=history.device)
            action_radius = torch.zeros(batch, dtype=history.dtype, device=history.device)
        else:
            action_embedding = self.action_encoder(torch.stack((action_length, action[:, 6], action[:, 7], nominal_delay, observation_delay), dim=-1))
            execution_time, gripper, fractions = _future_grid(action, nominal_delay, self.horizon_steps)
            action_radius = action[:, 7]
        future = current[:, :, None] + velocity[:, :, None] * execution_time[:, None, :, None] + 0.5 * effective_acceleration[:, :, None] * execution_time[:, None, :, None].square()
        relative = future - gripper[:, None]
        distance = _safe_vector_norm(relative)
        gripper_velocity = delta / action[:, 6].clamp_min(1.0e-6)[:, None]
        relative_velocity = velocity[:, :, None] + effective_acceleration[:, :, None] * execution_time[:, None, :, None] - gripper_velocity[:, None, None]
        from_start = current - action[:, None, :3]
        progress = (from_start * axis[:, None]).sum(dim=-1)
        perpendicular = _safe_vector_norm(from_start - progress[:, :, None] * axis[:, None])
        lateral = (from_start * radial_axis[:, None]).sum(dim=-1) * radial_valid[:, None].to(history.dtype)
        velocity_long = (velocity * axis[:, None]).sum(dim=-1)
        velocity_lateral = (velocity * radial_axis[:, None]).sum(dim=-1) * radial_valid[:, None].to(history.dtype)
        velocity_perp = _safe_vector_norm(velocity - velocity_long[:, :, None] * axis[:, None])
        acceleration_long = (effective_acceleration * axis[:, None]).sum(dim=-1)
        state_features = torch.stack((
            progress, perpendicular, lateral, velocity_long, velocity_lateral,
            velocity_perp, acceleration_long, _safe_vector_norm(effective_acceleration),
            radial_valid[:, None].expand(batch, points).to(history.dtype),
            action_length[:, None].expand(batch, points),
        ), dim=-1)
        if feature_mode == "no_action":
            contact_logits = torch.zeros((batch, points, self.horizon_steps), dtype=history.dtype, device=history.device)
            distance_features = history.new_zeros((batch, points, 4))
            state_features = history.new_zeros((batch, points, 10))
        else:
            normalized_time = fractions[None, None, :, None].expand(batch, points, self.horizon_steps, 1)
            contact_features = torch.cat((
                distance[:, :, :, None], _safe_vector_norm(relative_velocity)[:, :, :, None],
                (relative_velocity * axis[:, None, None]).sum(dim=-1)[:, :, :, None],
                normalized_time, confidence[:, :, None, None].expand(batch, points, self.horizon_steps, 1),
                visibility_fraction[:, :, None, None].expand(batch, points, self.horizon_steps, 1),
                torch.ones_like(normalized_time),
            ), dim=-1)
            contact_logits = self.contact_encoder(contact_features).squeeze(-1)
            closest = distance.argmin(dim=-1)
            distance_features = torch.stack((
                distance.amin(dim=-1), distance.mean(dim=-1), fractions[closest],
                _safe_vector_norm(relative_velocity).mean(dim=-1),
            ), dim=-1)
        local = torch.cat((
            history_embedding, action_embedding[:, None].expand(batch, points, action_embedding.shape[-1]),
            state_features, distance_features, confidence[:, :, None], visibility_fraction[:, :, None],
        ), dim=-1)
        residual = self.mask_head(local).squeeze(-1)
        contact_aggregate = torch.logsumexp(contact_logits, dim=-1) - math.log(self.horizon_steps)
        prior = (action_radius[:, None] - distance.amin(dim=-1)) / self.geometric_prior_temperature if feature_mode != "no_action" else torch.zeros_like(residual)
        mask_logits = self.residual_logit_scale * residual + self.contact_logit_scale * contact_aggregate + self.geometric_prior_scale * prior
        mask_logits = torch.where(any_visible, mask_logits, self.residual_logit_scale * residual)
        probability = torch.sigmoid(mask_logits)
        topk = min(8, points)
        topk_mean = probability.topk(topk, dim=1).values.mean(dim=1)
        utility_features = torch.cat((
            action_embedding, probability.mean(dim=1, keepdim=True), probability.amax(dim=1, keepdim=True),
            probability.sum(dim=1, keepdim=True) / float(points), topk_mean[:, None], torch.sigmoid(contact_logits).amax(dim=(1, 2))[:, None],
            torch.sigmoid(contact_logits).mean(dim=(1, 2))[:, None], confidence.mean(dim=1, keepdim=True),
            visibility_fraction.mean(dim=1, keepdim=True), observation_delay[:, None],
        ), dim=-1)
        utility_logits = self.utility_head(utility_features).squeeze(-1) if self.utility_head is not None else torch.logit(topk_mean.clamp(1.0e-5, 1.0 - 1.0e-5))
        return {
            "mask_logits": mask_logits, "contact_logits": contact_logits, "utility_logits": utility_logits,
            "future_hypotheses": future, "estimated_current_points": current,
            "estimated_acceleration": effective_acceleration, "time_aligned_distance": distance,
        }


__all__ = [
    "ActMaskMLP",
    "TEMPORAL_FEATURE_MODES",
    "TemporalActMask",
    "RotationInvariantTemporalActMask",
    "ActionFrameRelativeTemporalActMask",
    "estimated_acceleration_from_history",
    "last_visible_state",
]
