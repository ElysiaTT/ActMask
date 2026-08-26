"""Hardened synthetic benchmark for ActMask milestone 2A.

This module preserves the milestone-1 model-facing contract (current points,
observed point velocities, and an eight-value straight-line action) while
generating labels from explicit nonlinear future trajectories.  Related
counterfactual variants are expanded only after an indivisible group manifest
has assigned their base scene to a split.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .grouped_splits import (
    DEFAULT_SCENARIOS,
    OOD_DOMAINS,
    GroupRecord,
    build_group_manifest,
)
from .toy_dynamic_dataset import ACTION_DIM, ACTION_SCHEMA


MILESTONE2_SCENARIOS = DEFAULT_SCENARIOS
PARAMETER_SCHEMA = (
    "velocity_magnitude",
    "acceleration_magnitude",
    "duration",
    "gripper_radius",
    "curvature",
    "point_noise_std",
    "point_dropout_rate",
    "distractor_count",
    "velocity_noise_std",
    "temporal_velocity_delay",
    "motion_delay",
)
POINT_ROLE_SCHEMA = (
    "background",
    "target",
    "moving_distractor",
    "static_near_action",
)

ROLE_BACKGROUND = 0
ROLE_TARGET = 1
ROLE_MOVING_DISTRACTOR = 2
ROLE_STATIC_NEAR_ACTION = 3

# High endpoints are exclusive for continuous sampling.  Every OOD support is
# separated from its corresponding ID support by a non-zero gap.
ID_PARAMETER_RANGES: dict[str, tuple[float, float]] = {
    "velocity": (0.18, 0.32),
    "acceleration": (0.04, 0.14),
    "duration": (0.70, 1.10),
    "radius": (0.055, 0.095),
    "curvature": (0.30, 0.75),
    "point_noise": (0.0, 0.004),
    "distractor_count": (1.0, 3.0),  # integers 1 or 2
}
OOD_PARAMETER_RANGES: dict[str, tuple[float, float]] = {
    "velocity": (0.40, 0.54),
    "acceleration": (0.22, 0.34),
    "duration": (1.35, 1.70),
    "radius": (0.120, 0.150),
    "curvature": (1.05, 1.45),
    "point_noise": (0.010, 0.016),
    "distractor_count": (4.0, 7.0),  # integers 4, 5, or 6
}

_OOD_AXIS = {name: name.removeprefix("ood_") for name in OOD_DOMAINS}
_COUNTERFACTUAL_TYPES = {
    "velocity_counterfactual": "velocity",
    "action_counterfactual": "action",
    "timing_counterfactual": "action_timing",
    "constant_acceleration": "future_dynamics",
    "circular_motion": "velocity_and_future_dynamics",
    "delayed_onset": "future_dynamics",
    "multiple_moving_objects": "action",
    "fast_moving_distractor": "action",
    "static_near_noncontact": "action",
    "far_future_entry": "velocity",
    "near_miss_action": "action",
    "radius_counterfactual": "action_radius",
}
_DYNAMICS_TYPES = {
    "constant_acceleration": "constant_acceleration",
    "circular_motion": "circular",
    "delayed_onset": "delayed_constant_velocity",
}


def _validate_int(name: str, value: int, minimum: int) -> int:
    if not isinstance(value, (int, np.integer)) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _rng(identifier: int, tag: int) -> np.random.Generator:
    identifier = int(identifier)
    words = [identifier & 0xFFFFFFFF, (identifier >> 32) & 0xFFFFFFFF, int(tag)]
    return np.random.default_rng(np.random.SeedSequence(words))


def _uniform(rng: np.random.Generator, bounds: tuple[float, float]) -> float:
    return float(rng.uniform(bounds[0], bounds[1]))


def _axis_bounds(domain: str, axis: str) -> tuple[float, float]:
    if domain == f"ood_{axis}":
        return OOD_PARAMETER_RANGES[axis]
    return ID_PARAMETER_RANGES[axis]


def _rotation_z(points: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """Rotate vectors ``[N,3]`` by each angle, returning ``[T,N,3]``."""

    cosine = np.cos(angles).astype(np.float32)[:, None]
    sine = np.sin(angles).astype(np.float32)[:, None]
    x = points[None, :, 0]
    y = points[None, :, 1]
    z = np.broadcast_to(points[None, :, 2], (len(angles), len(points)))
    return np.stack(
        (cosine * x - sine * y, sine * x + cosine * y, z), axis=-1
    ).astype(np.float32)


def compute_time_aligned_contact(
    point_trajectory: np.ndarray, gripper_trajectory: np.ndarray, radius: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return the sampled synchronous contact mask and minimum distances."""

    points = np.asarray(point_trajectory, dtype=np.float32)
    gripper = np.asarray(gripper_trajectory, dtype=np.float32)
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError("point_trajectory must have shape [T, N, 3]")
    if gripper.shape != (points.shape[0], 3):
        raise ValueError("gripper_trajectory must have shape [T, 3]")
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius must be finite and positive")
    distance = np.linalg.norm(points - gripper[:, None, :], axis=-1)
    minimum = distance.min(axis=0).astype(np.float32)
    return (minimum <= np.float32(radius)).astype(np.float32), minimum


class Milestone2DynamicDataset(Dataset[dict[str, Tensor | str]]):
    """Deterministic paired nonlinear point-trajectory benchmark.

    Exactly two variants are generated for every manifest group.  ID groups
    use ``split`` in ``train/val/test``; OOD datasets use ``split='test'`` and
    one of the seven :data:`OOD_DOMAINS` values.
    """

    def __init__(
        self,
        split: str = "train",
        domain: str = "id",
        num_points: int = 96,
        trajectory_steps: int = 32,
        master_seed: int = 2026,
        total_id_groups: int = 96,
        split_fractions: Sequence[float] = (0.6, 0.2, 0.2),
        ood_groups_per_domain: int = 12,
        point_noise_std: float | None = None,
        velocity_noise_std: float = 0.006,
        temporal_velocity_delay: float = 0.0,
        point_dropout_rate: float = 0.0,
        split_counts: Mapping[str, int] | None = None,
    ) -> None:
        super().__init__()
        split = str(split)
        domain = str(domain)
        if split in OOD_DOMAINS and domain == "id":
            domain, split = split, "test"
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test")
        if domain != "id" and domain not in OOD_DOMAINS:
            raise ValueError(f"unknown domain {domain!r}")
        if domain != "id" and split != "test":
            raise ValueError("OOD domains are test-only")
        self.num_points = _validate_int("num_points", num_points, 24)
        self.trajectory_steps = _validate_int("trajectory_steps", trajectory_steps, 8)
        if not isinstance(master_seed, (int, np.integer)) or isinstance(master_seed, bool):
            raise TypeError("master_seed must be an integer")
        self.master_seed = int(master_seed)
        if point_noise_std is not None:
            point_noise_std = float(point_noise_std)
            if not np.isfinite(point_noise_std) or point_noise_std < 0.0:
                raise ValueError("point_noise_std must be finite and non-negative")
        velocity_noise_std = float(velocity_noise_std)
        if not np.isfinite(velocity_noise_std) or velocity_noise_std < 0.0:
            raise ValueError("velocity_noise_std must be finite and non-negative")
        temporal_velocity_delay = float(temporal_velocity_delay)
        if (
            not np.isfinite(temporal_velocity_delay)
            or temporal_velocity_delay < 0.0
        ):
            raise ValueError(
                "temporal_velocity_delay must be finite and non-negative"
            )
        point_dropout_rate = float(point_dropout_rate)
        if (
            not np.isfinite(point_dropout_rate)
            or point_dropout_rate < 0.0
            or point_dropout_rate > 1.0
        ):
            raise ValueError("point_dropout_rate must be finite and lie in [0, 1]")
        self.point_noise_std = point_noise_std
        self.velocity_noise_std = velocity_noise_std
        self.temporal_velocity_delay = temporal_velocity_delay
        self.point_dropout_rate = point_dropout_rate
        self.split = split
        self.domain = domain
        self.manifest = build_group_manifest(
            master_seed=self.master_seed,
            total_id_groups=total_id_groups,
            split_fractions=split_fractions,
            split_counts=split_counts,
            ood_groups_per_domain=ood_groups_per_domain,
            scenarios=MILESTONE2_SCENARIOS,
        )
        self.group_records: tuple[GroupRecord, ...] = self.manifest[
            split if domain == "id" else domain
        ]

    def __len__(self) -> int:
        return len(self.group_records) * 2

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        if isinstance(index, torch.Tensor):
            if index.numel() != 1:
                raise IndexError("dataset index tensor must contain one value")
            index = int(index.item())
        if not isinstance(index, (int, np.integer)) or isinstance(index, bool):
            raise TypeError("index must be an integer")
        index = int(index)
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(f"index {index} is outside a dataset of length {len(self)}")
        record = self.group_records[index // 2]
        variant_id = index % 2
        pair = self._generate_pair(record)
        return pair[variant_id]

    def _sample_parameters(self, record: GroupRecord) -> dict[str, float]:
        rng = _rng(record.scenario_seed, 101)
        result = {
            axis: _uniform(rng, _axis_bounds(record.domain, axis))
            for axis in ("velocity", "acceleration", "duration", "radius", "curvature")
        }
        distractor_low, distractor_high = _axis_bounds(record.domain, "distractor_count")
        result["distractor_count"] = float(
            rng.integers(int(distractor_low), int(distractor_high))
        )
        if self.point_noise_std is not None and record.domain != "ood_point_noise":
            result["point_noise"] = self.point_noise_std
        else:
            result["point_noise"] = _uniform(
                rng, _axis_bounds(record.domain, "point_noise")
            )

        # Dynamics scenarios use a high in-range value so their paired masks
        # differ by a safe margin rather than by floating-point boundary noise.
        for scenario, axis in (
            ("constant_acceleration", "acceleration"),
            ("far_future_entry", "velocity"),
            ("delayed_onset", "velocity"),
        ):
            if record.scenario == scenario:
                low, high = _axis_bounds(record.domain, axis)
                result[axis] = low + 0.88 * (high - low)
        if record.scenario in {
            "constant_acceleration",
            "far_future_entry",
            "delayed_onset",
        }:
            low, high = _axis_bounds(record.domain, "duration")
            result["duration"] = low + 0.90 * (high - low)
        if record.scenario in {"multiple_moving_objects", "fast_moving_distractor"}:
            _, high = _axis_bounds(record.domain, "distractor_count")
            result["distractor_count"] = float(max(2, int(high) - 1))
        return result

    def _base_geometry(
        self, record: GroupRecord, parameters: Mapping[str, float]
    ) -> dict[str, Any]:
        rng = _rng(record.geometry_seed, 211)
        center = rng.uniform((-0.12, -0.12, -0.035), (0.12, 0.12, 0.035)).astype(
            np.float32
        )
        angle = float(rng.uniform(0.0, 2.0 * np.pi))
        direction = np.asarray([np.cos(angle), np.sin(angle), 0.0], dtype=np.float32)
        perpendicular = np.asarray(
            [-direction[1], direction[0], 0.0], dtype=np.float32
        )

        target_count = max(5, self.num_points // 5)
        static_count = max(3, self.num_points // 10)
        distractor_count = int(parameters["distractor_count"])
        remaining = self.num_points - target_count - static_count - 4
        points_per_distractor = max(1, min(6, remaining // distractor_count))
        distractor_total = distractor_count * points_per_distractor
        background_count = self.num_points - target_count - static_count - distractor_total
        if background_count < 4:
            raise RuntimeError("point allocation left too few background points")

        target_offsets = np.clip(
            rng.normal(0.0, 0.003, size=(target_count, 3)), -0.007, 0.007
        ).astype(np.float32)
        target_offsets[0] = 0.0
        target_points = center[None, :] + target_offsets

        distractor_parts: list[np.ndarray] = []
        for object_index in range(distractor_count):
            distractor_center = (
                center
                + perpendicular * np.float32(0.52 + 0.22 * object_index)
                + np.asarray([0.0, 0.0, 0.045 * ((object_index % 2) * 2 - 1)], np.float32)
            )
            offsets = np.clip(
                rng.normal(0.0, 0.009, size=(points_per_distractor, 3)),
                -0.022,
                0.022,
            ).astype(np.float32)
            offsets[0] = 0.0
            distractor_parts.append(distractor_center[None, :] + offsets)
        distractor_points = np.concatenate(distractor_parts, axis=0)

        radius = float(parameters["radius"])
        static_center = center + perpendicular * np.float32(1.55 * radius)
        static_offsets = np.clip(
            rng.normal(0.0, 0.0015, size=(static_count, 3)), -0.004, 0.004
        ).astype(np.float32)
        static_points = static_center[None, :] + static_offsets

        # Background is deliberately well spread; the irrelevant-change helper
        # selects only points whose measured contact margin proves invariance.
        background_points = rng.uniform(
            center - np.asarray([0.85, 0.85, 0.42], dtype=np.float32),
            center + np.asarray([0.85, 0.85, 0.42], dtype=np.float32),
            size=(background_count, 3),
        ).astype(np.float32)

        true_points = np.concatenate(
            (target_points, distractor_points, static_points, background_points), axis=0
        ).astype(np.float32)
        object_ids = np.full(self.num_points, -1, dtype=np.int64)
        point_roles = np.full(self.num_points, ROLE_BACKGROUND, dtype=np.int64)
        target_slice = slice(0, target_count)
        object_ids[target_slice] = 0
        point_roles[target_slice] = ROLE_TARGET
        distractor_start = target_count
        for object_index in range(distractor_count):
            begin = distractor_start + object_index * points_per_distractor
            end = begin + points_per_distractor
            object_ids[begin:end] = object_index + 1
            point_roles[begin:end] = ROLE_MOVING_DISTRACTOR
        static_start = target_count + distractor_total
        static_slice = slice(static_start, static_start + static_count)
        object_ids[static_slice] = distractor_count + 1
        point_roles[static_slice] = ROLE_STATIC_NEAR_ACTION

        permutation = _rng(record.geometry_seed, 223).permutation(self.num_points)
        position_noise = _rng(record.geometry_seed, 227).normal(
            0.0, parameters["point_noise"], size=(self.num_points, 3)
        ).astype(np.float32)
        velocity_noise = _rng(record.geometry_seed, 229).normal(
            0.0, self.velocity_noise_std, size=(self.num_points, 3)
        ).astype(np.float32)
        return {
            "center": center,
            "direction": direction,
            "perpendicular": perpendicular,
            "true_points": true_points,
            "object_ids": object_ids,
            "point_roles": point_roles,
            "target_indices": np.arange(target_count, dtype=np.int64),
            "distractor_count": distractor_count,
            "permutation": permutation,
            "position_noise": position_noise,
            "velocity_noise": velocity_noise,
        }

    def _paired_duration_radius(
        self, scenario: str, variant_id: int, parameters: Mapping[str, float], domain: str
    ) -> tuple[float, float]:
        duration = float(parameters["duration"])
        radius = float(parameters["radius"])
        if scenario == "timing_counterfactual":
            low, high = _axis_bounds(domain, "duration")
            duration = low + (0.82 if variant_id == 0 else 0.18) * (high - low)
        if scenario == "radius_counterfactual":
            low, high = _axis_bounds(domain, "radius")
            radius = low + (0.84 if variant_id == 0 else 0.16) * (high - low)
        return duration, radius

    def _target_motion(
        self,
        scenario: str,
        variant_id: int,
        times: np.ndarray,
        geometry: Mapping[str, Any],
        parameters: Mapping[str, float],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        points = np.asarray(geometry["true_points"], dtype=np.float32)
        target_indices = np.asarray(geometry["target_indices"], dtype=np.int64)
        target = points[target_indices]
        direction = np.asarray(geometry["direction"], dtype=np.float32)
        perpendicular = np.asarray(geometry["perpendicular"], dtype=np.float32)
        speed = float(parameters["velocity"])
        acceleration_magnitude = float(parameters["acceleration"])
        curvature = float(parameters["curvature"])
        delay = 0.0

        if scenario == "velocity_counterfactual":
            velocity = direction * np.float32(speed * (1.0 if variant_id == 0 else -1.0))
            trajectory = target[None, :, :] + times[:, None, None] * velocity[None, None, :]
            true_velocity = np.broadcast_to(velocity, target.shape).copy()
            acceleration = np.zeros_like(target)
        elif scenario == "constant_acceleration":
            velocity = direction * np.float32(speed)
            acceleration_vector = direction * np.float32(
                acceleration_magnitude * (1.0 if variant_id == 0 else -1.0)
            )
            trajectory = (
                target[None, :, :]
                + times[:, None, None] * velocity[None, None, :]
                + 0.5
                * (times * times)[:, None, None]
                * acceleration_vector[None, None, :]
            )
            true_velocity = np.broadcast_to(velocity, target.shape).copy()
            acceleration = np.broadcast_to(acceleration_vector, target.shape).copy()
        elif scenario == "circular_motion":
            omega = curvature * (1.0 if variant_id == 0 else -1.0)
            center = np.asarray(geometry["center"], dtype=np.float32)
            orbit_radius = speed / curvature
            pivot = center + perpendicular * np.float32(orbit_radius)
            relative = target - pivot[None, :]
            trajectory = _rotation_z(relative, times * np.float32(omega)) + pivot[None, None, :]
            true_velocity = np.stack(
                (-omega * relative[:, 1], omega * relative[:, 0], np.zeros(len(relative))),
                axis=-1,
            ).astype(np.float32)
            acceleration = (-omega * omega * relative).astype(np.float32)
            acceleration[:, 2] = 0.0
        elif scenario == "delayed_onset":
            delay = float(times[-1]) * (0.05 if variant_id == 0 else 0.80)
            active_time = np.maximum(times - np.float32(delay), 0.0)
            velocity = direction * np.float32(speed)
            trajectory = target[None, :, :] + active_time[:, None, None] * velocity[None, None, :]
            true_velocity = np.zeros_like(target)
            acceleration = np.zeros_like(target)
        elif scenario == "far_future_entry":
            velocity = perpendicular * np.float32(speed * (1.0 if variant_id == 0 else -1.0))
            trajectory = target[None, :, :] + times[:, None, None] * velocity[None, None, :]
            true_velocity = np.broadcast_to(velocity, target.shape).copy()
            acceleration = np.zeros_like(target)
        else:
            velocity = direction * np.float32(speed)
            trajectory = target[None, :, :] + times[:, None, None] * velocity[None, None, :]
            true_velocity = np.broadcast_to(velocity, target.shape).copy()
            acceleration = np.zeros_like(target)
        return (
            trajectory.astype(np.float32),
            true_velocity.astype(np.float32),
            acceleration.astype(np.float32),
            delay,
        )

    def _action_for_variant(
        self,
        scenario: str,
        variant_id: int,
        duration: float,
        radius: float,
        geometry: Mapping[str, Any],
        parameters: Mapping[str, float],
        target_trajectory: np.ndarray,
    ) -> np.ndarray:
        center = np.asarray(geometry["center"], dtype=np.float32)
        direction = np.asarray(geometry["direction"], dtype=np.float32)
        perpendicular = np.asarray(geometry["perpendicular"], dtype=np.float32)
        endpoint = target_trajectory[-1, 0].copy()
        start = center - direction * np.float32(0.42) + perpendicular * np.float32(0.24)
        end = endpoint.copy()

        if scenario == "velocity_counterfactual":
            # Shared action is defined from the variant-0 future.
            speed = float(parameters["velocity"])
            end = center + direction * np.float32(speed * duration)
            start = center - direction * np.float32(0.42) + perpendicular * np.float32(0.65)
        elif scenario == "action_counterfactual":
            if variant_id == 1:
                end = end + perpendicular * np.float32(0.46)
        elif scenario == "timing_counterfactual":
            duration_high = _axis_bounds(self.domain, "duration")[0] + 0.82 * (
                _axis_bounds(self.domain, "duration")[1]
                - _axis_bounds(self.domain, "duration")[0]
            )
            speed = float(parameters["velocity"])
            # A shared endpoint just inside the long-duration contact boundary
            # makes the shorter execution a genuine time-aligned miss even
            # when the allowed ID duration range is relatively narrow.
            end = center + direction * np.float32(
                speed * duration_high + 0.90 * radius
            )
            start = center - direction * np.float32(0.42) + perpendicular * np.float32(1.30)
        elif scenario == "constant_acceleration":
            speed = float(parameters["velocity"])
            accel = float(parameters["acceleration"])
            # Offset toward the positive-acceleration future.  Variant 0 stays
            # within the radius while the negative-acceleration future is more
            # than one radius behind at the endpoint.
            end = center + direction * np.float32(
                speed * duration + 0.5 * accel * duration**2 + 0.78 * radius
            )
            # A large transverse approach offset prevents the gripper from
            # crossing the negative-acceleration trajectory before the two
            # futures have separated.  It still reaches variant 0 at t=D.
            start = center - direction * np.float32(0.42) + perpendicular * np.float32(0.70)
        elif scenario == "circular_motion":
            if variant_id == 1:
                # Recompute the shared action endpoint from the positive-turn trajectory.
                omega = float(parameters["curvature"])
                orbit_radius = float(parameters["velocity"]) / omega
                pivot = center + perpendicular * np.float32(orbit_radius)
                positive_endpoint = _rotation_z(
                    (np.asarray(geometry["true_points"])[geometry["target_indices"]] - pivot),
                    np.asarray([duration * omega], dtype=np.float32),
                )[0, 0]
                end = pivot + positive_endpoint
            start = center - direction * np.float32(0.42) + perpendicular * np.float32(0.75)
        elif scenario == "delayed_onset":
            speed = float(parameters["velocity"])
            end = center + direction * np.float32(speed * duration * 0.95)
            start = center - direction * np.float32(0.42) + perpendicular * np.float32(0.70)
        elif scenario in {"multiple_moving_objects", "fast_moving_distractor"}:
            start = center - direction * np.float32(0.42)
            if variant_id == 1:
                end = end + perpendicular * np.float32(0.46)
        elif scenario == "static_near_noncontact":
            start = center - direction * np.float32(0.42)
            if variant_id == 1:
                end = end + perpendicular * np.float32(0.46)
        elif scenario == "far_future_entry":
            speed = float(parameters["velocity"])
            end = center + perpendicular * np.float32(speed * duration)
            start = end - direction * np.float32(0.42)
        elif scenario == "near_miss_action":
            start = center - direction * np.float32(0.42)
            if variant_id == 1:
                miss_offset = radius + 0.045
                start = start + perpendicular * np.float32(miss_offset)
                end = end + perpendicular * np.float32(miss_offset)
        elif scenario == "radius_counterfactual":
            low, high = _axis_bounds(self.domain, "radius")
            small = low + 0.16 * (high - low)
            large = low + 0.84 * (high - low)
            middle = 0.5 * (small + large)
            start = center + perpendicular * np.float32(middle)
            end = endpoint + perpendicular * np.float32(middle)

        return np.concatenate(
            (start, end, np.asarray([duration, radius], dtype=np.float32))
        ).astype(np.float32)

    def _generate_pair(self, record: GroupRecord) -> tuple[dict[str, Tensor | str], ...]:
        parameters = self._sample_parameters(record)
        geometry = self._base_geometry(record, parameters)
        logical_points = np.asarray(geometry["true_points"], dtype=np.float32)
        target_indices = np.asarray(geometry["target_indices"], dtype=np.int64)
        distractor_mask = np.asarray(geometry["point_roles"]) == ROLE_MOVING_DISTRACTOR
        permutation = np.asarray(geometry["permutation"], dtype=np.int64)
        pair: list[dict[str, Tensor | str]] = []

        for variant_id in (0, 1):
            duration, radius = self._paired_duration_radius(
                record.scenario, variant_id, parameters, record.domain
            )
            times = np.linspace(
                0.0, duration, self.trajectory_steps, endpoint=True, dtype=np.float32
            )
            point_trajectory = np.broadcast_to(
                logical_points[None, :, :],
                (self.trajectory_steps, self.num_points, 3),
            ).copy()
            true_velocities = np.zeros_like(logical_points)
            accelerations = np.zeros_like(logical_points)

            target_trajectory, target_velocity, target_acceleration, motion_delay = (
                self._target_motion(
                    record.scenario,
                    variant_id,
                    times,
                    geometry,
                    parameters,
                )
            )
            point_trajectory[:, target_indices, :] = target_trajectory
            true_velocities[target_indices] = target_velocity
            accelerations[target_indices] = target_acceleration

            distractor_speed = float(parameters["velocity"]) * (
                1.85 if record.scenario == "fast_moving_distractor" else 0.72
            )
            distractor_velocity = (
                np.asarray(geometry["perpendicular"], dtype=np.float32)
                * np.float32(distractor_speed)
            )
            point_trajectory[:, distractor_mask, :] = (
                logical_points[distractor_mask][None, :, :]
                + times[:, None, None] * distractor_velocity[None, None, :]
            )
            true_velocities[distractor_mask] = distractor_velocity

            action = self._action_for_variant(
                record.scenario,
                variant_id,
                duration,
                radius,
                geometry,
                parameters,
                target_trajectory,
            )
            fractions = times / np.float32(duration)
            gripper_trajectory = (
                action[None, :3]
                + fractions[:, None] * (action[3:6] - action[:3])[None, :]
            ).astype(np.float32)
            mask, minimum_distance = compute_time_aligned_contact(
                point_trajectory, gripper_trajectory, radius
            )
            success = np.float32(bool(np.any(mask[target_indices] > 0.5)))

            observed_points = logical_points + np.asarray(
                geometry["position_noise"], dtype=np.float32
            )
            # Model inputs receive a stale velocity observation, not future
            # oracle dynamics.  A first-order backward propagation by the
            # configured observation latency is exact for constant
            # acceleration and intentionally exposes a realistic temporal
            # misalignment for nonlinear motion.  Contact labels remain based
            # solely on the unchanged physical future trajectory above.
            stale_velocities = true_velocities - accelerations * np.float32(
                self.temporal_velocity_delay
            )
            observed_velocities = stale_velocities + np.asarray(
                geometry["velocity_noise"], dtype=np.float32
            )
            variant_parameters = np.asarray(
                [
                    parameters["velocity"],
                    parameters["acceleration"],
                    duration,
                    radius,
                    parameters["curvature"],
                    parameters["point_noise"],
                    self.point_dropout_rate,
                    parameters["distractor_count"],
                    self.velocity_noise_std,
                    self.temporal_velocity_delay,
                    motion_delay,
                ],
                dtype=np.float32,
            )

            def f32(value: np.ndarray) -> Tensor:
                return torch.from_numpy(np.asarray(value, dtype=np.float32).copy())

            def i64(value: np.ndarray) -> Tensor:
                return torch.from_numpy(np.asarray(value, dtype=np.int64).copy())

            sample_id = record.pair_id * 2 + variant_id
            pair.append(
                {
                    "points": f32(observed_points[permutation]),
                    "velocities": f32(observed_velocities[permutation]),
                    "action": f32(action),
                    "mask": f32(mask[permutation]),
                    "success": torch.tensor(success, dtype=torch.float32),
                    "pair_id": torch.tensor(record.pair_id, dtype=torch.int64),
                    "trajectory_family_id": torch.tensor(
                        record.trajectory_family_id, dtype=torch.int64
                    ),
                    "variant_id": torch.tensor(variant_id, dtype=torch.int64),
                    "scenario": record.scenario,
                    "group_id": torch.tensor(record.group_id, dtype=torch.int64),
                    "base_scene_id": torch.tensor(record.base_scene_id, dtype=torch.int64),
                    "geometry_seed": torch.tensor(record.geometry_seed, dtype=torch.int64),
                    "scenario_seed": torch.tensor(record.scenario_seed, dtype=torch.int64),
                    "sample_id": torch.tensor(sample_id, dtype=torch.int64),
                    "split": record.split,
                    "domain": record.domain,
                    "trajectory_family": _DYNAMICS_TYPES.get(
                        record.scenario, "constant_velocity"
                    ),
                    "counterfactual_type": _COUNTERFACTUAL_TYPES[record.scenario],
                    "dynamics_type": _DYNAMICS_TYPES.get(
                        record.scenario, "constant_velocity"
                    ),
                    "times": f32(times),
                    "point_trajectory": f32(point_trajectory[:, permutation, :]),
                    "gripper_trajectory": f32(gripper_trajectory),
                    "true_points": f32(logical_points[permutation]),
                    "true_velocities": f32(true_velocities[permutation]),
                    "observed_points": f32(observed_points[permutation]),
                    "observed_velocities": f32(observed_velocities[permutation]),
                    "accelerations": f32(accelerations[permutation]),
                    "object_ids": i64(np.asarray(geometry["object_ids"])[permutation]),
                    "point_roles": i64(np.asarray(geometry["point_roles"])[permutation]),
                    "target_object_id": torch.tensor(0, dtype=torch.int64),
                    "min_contact_distance": f32(minimum_distance[permutation]),
                    "parameter_values": f32(variant_parameters),
                    "point_dropout_rate": torch.tensor(
                        self.point_dropout_rate, dtype=torch.float32
                    ),
                    "temporal_velocity_delay": torch.tensor(
                        self.temporal_velocity_delay, dtype=torch.float32
                    ),
                    "point_dropout_mask": torch.zeros(
                        self.num_points, dtype=torch.bool
                    ),
                    "valid_point_mask": torch.ones(
                        self.num_points, dtype=torch.bool
                    ),
                    "changed_input_mask": torch.zeros(self.num_points, dtype=torch.bool),
                    "invariant_mask": torch.ones(self.num_points, dtype=torch.bool),
                }
            )

        self._apply_pair_point_dropout(pair, record)

        # Base point geometry and its observation corruption are a pair-level
        # invariant even when velocity or future dynamics changes.
        if not torch.equal(pair[0]["points"], pair[1]["points"]):  # type: ignore[arg-type]
            raise RuntimeError("counterfactual pair did not preserve current points")
        return tuple(pair)

    def _apply_pair_point_dropout(
        self, pair: list[dict[str, Tensor | str]], record: GroupRecord
    ) -> None:
        """Hide a deterministic subset of jointly safe background inputs.

        Fixed-size batching is retained by writing an explicitly invalid,
        provably far-away placeholder into the *observed* point/velocity
        tensors.  Ground-truth trajectories and masks are never changed.  The
        same dropout indices are used for both variants of a counterfactual
        group, and an index is eligible only when it is a background point,
        negative in both variants, and has a contact-margin buffer in both.
        """

        if len(pair) != 2:
            raise RuntimeError("Milestone-2 point dropout requires exactly two variants")
        if self.point_dropout_rate <= 0.0:
            return

        roles = torch.as_tensor(pair[0]["point_roles"], dtype=torch.int64)
        eligible = roles == ROLE_BACKGROUND
        for sample in pair:
            sample_roles = torch.as_tensor(sample["point_roles"], dtype=torch.int64)
            if not torch.equal(sample_roles, roles):
                raise RuntimeError("counterfactual variants must share point roles")
            target = torch.as_tensor(sample["mask"], dtype=torch.float32)
            minimum = torch.as_tensor(sample["min_contact_distance"], dtype=torch.float32)
            radius = float(torch.as_tensor(sample["action"], dtype=torch.float32)[7].item())
            # The buffer makes the "safe negative" contract explicit even if
            # a downstream consumer inspects the hidden trajectory metadata.
            clearance = max(0.03, 0.5 * radius)
            eligible &= target < 0.5
            eligible &= minimum > radius + clearance

        eligible_indices = torch.nonzero(eligible, as_tuple=False).flatten()
        if eligible_indices.numel() == 0:
            return
        count = min(
            int(eligible_indices.numel()),
            max(1, int(np.ceil(self.point_dropout_rate * int(eligible_indices.numel())))),
        )
        selection_rng = _rng(record.geometry_seed, 239)
        selected_np = selection_rng.choice(
            eligible_indices.detach().cpu().numpy(), size=count, replace=False
        )
        selected = torch.as_tensor(np.sort(selected_np), dtype=torch.long)

        # A pair-global placeholder keeps unchanged observed positions exactly
        # equal across counterfactual variants while remaining far outside the
        # complete simulated scene/action extent.
        bound = 0.0
        maximum_radius = 0.0
        for sample in pair:
            bound = max(
                bound,
                float(torch.as_tensor(sample["point_trajectory"]).abs().max().item()),
                float(torch.as_tensor(sample["action"]).abs().max().item()),
            )
            maximum_radius = max(
                maximum_radius,
                float(torch.as_tensor(sample["action"])[7].item()),
            )
        placeholder_value = bound + 4.0 * maximum_radius + 1.0

        for sample in pair:
            target = torch.as_tensor(sample["mask"], dtype=torch.float32)
            roles = torch.as_tensor(sample["point_roles"], dtype=torch.int64)
            if bool(torch.any(target[selected] >= 0.5)) or bool(
                torch.any(roles[selected] != ROLE_BACKGROUND)
            ):
                raise RuntimeError("point dropout selected a non-background positive point")
            for key in ("points", "observed_points"):
                observed = torch.as_tensor(sample[key], dtype=torch.float32).clone()
                observed[selected] = placeholder_value
                sample[key] = observed
            for key in ("velocities", "observed_velocities"):
                observed_velocity = torch.as_tensor(sample[key], dtype=torch.float32).clone()
                observed_velocity[selected] = 0.0
                sample[key] = observed_velocity
            dropped = torch.zeros(self.num_points, dtype=torch.bool)
            dropped[selected] = True
            sample["point_dropout_mask"] = dropped
            sample["valid_point_mask"] = ~dropped


def make_irrelevant_background_counterfactual(
    sample: Mapping[str, Tensor | str],
    *,
    seed: int = 0,
    displacement: float = 0.04,
    max_points: int | None = None,
) -> dict[str, Tensor | str]:
    """Move provably irrelevant background points without changing the GT mask.

    Candidate points must have more than ``2 * displacement`` clearance beyond
    the gripper radius.  The triangle inequality then guarantees that any shift
    of the requested magnitude remains a non-contact.  Returned tensors are
    clones and the input mapping is never mutated.
    """

    if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool):
        raise TypeError("seed must be an integer")
    displacement = float(displacement)
    if not np.isfinite(displacement) or displacement <= 0.0:
        raise ValueError("displacement must be finite and positive")
    result: dict[str, Tensor | str] = {
        key: value.clone() if isinstance(value, Tensor) else value for key, value in sample.items()
    }
    action = torch.as_tensor(result["action"], dtype=torch.float32)
    minimum = torch.as_tensor(result["min_contact_distance"], dtype=torch.float32)
    roles = torch.as_tensor(result["point_roles"], dtype=torch.int64)
    eligible = torch.logical_and(
        roles == ROLE_BACKGROUND,
        minimum > float(action[7].item()) + 2.0 * displacement,
    )
    if "valid_point_mask" in result:
        eligible &= torch.as_tensor(result["valid_point_mask"], dtype=torch.bool)
    indices = torch.nonzero(eligible, as_tuple=False).flatten()
    if max_points is None:
        max_points = max(1, int(indices.numel()) // 2)
    max_points = _validate_int("max_points", max_points, 1)
    if indices.numel() == 0:
        raise RuntimeError("sample has no background point with sufficient contact margin")
    generator = torch.Generator(device="cpu").manual_seed(int(seed) & 0x7FFFFFFFFFFFFFFF)
    ordering = torch.randperm(indices.numel(), generator=generator)
    indices = indices[ordering[: min(max_points, indices.numel())]]
    shifts = torch.randn((len(indices), 3), generator=generator, dtype=torch.float32)
    shifts = shifts / torch.linalg.vector_norm(shifts, dim=-1, keepdim=True).clamp_min(1e-8)
    shifts = shifts * displacement

    for key in ("points", "observed_points", "true_points"):
        tensor = torch.as_tensor(result[key], dtype=torch.float32).clone()
        tensor[indices] += shifts
        result[key] = tensor
    trajectory = torch.as_tensor(result["point_trajectory"], dtype=torch.float32).clone()
    trajectory[:, indices, :] += shifts[None, :, :]
    result["point_trajectory"] = trajectory
    gripper = torch.as_tensor(result["gripper_trajectory"], dtype=torch.float32)
    new_minimum = torch.linalg.vector_norm(
        trajectory - gripper[:, None, :], dim=-1
    ).amin(dim=0)
    new_mask = (new_minimum <= action[7]).to(torch.float32)
    old_mask = torch.as_tensor(sample["mask"], dtype=torch.float32)
    if not torch.equal(new_mask, old_mask):
        raise RuntimeError("irrelevant perturbation unexpectedly changed the contact mask")
    result["min_contact_distance"] = new_minimum
    result["mask"] = new_mask
    changed = torch.zeros_like(old_mask, dtype=torch.bool)
    changed[indices] = True
    result["changed_input_mask"] = changed
    result["invariant_mask"] = ~changed
    result["counterfactual_type"] = "irrelevant_background"
    return result


__all__ = [
    "ACTION_DIM",
    "ACTION_SCHEMA",
    "ID_PARAMETER_RANGES",
    "MILESTONE2_SCENARIOS",
    "OOD_DOMAINS",
    "OOD_PARAMETER_RANGES",
    "PARAMETER_SCHEMA",
    "POINT_ROLE_SCHEMA",
    "ROLE_BACKGROUND",
    "ROLE_MOVING_DISTRACTOR",
    "ROLE_STATIC_NEAR_ACTION",
    "ROLE_TARGET",
    "Milestone2DynamicDataset",
    "compute_time_aligned_contact",
    "make_irrelevant_background_counterfactual",
]
