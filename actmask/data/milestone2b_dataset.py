"""Milestone 2B history-conditioned dynamics benchmark.

The public sample contract deliberately separates observable inputs, supervised
targets, and hidden physical state.  Learned models and fair baselines receive
only ``sample["observable"]``.  Exact trajectories are retained under
``sample["hidden_state"]`` for label construction and the excluded oracle.

Every base scene is assigned to a grouped split before two motion variants,
five candidate actions, and an irrelevant-observation perturbation are
expanded.  The reference action contains deliberately balanced hard positives
and hard negatives, making current Euclidean proximity insufficient.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


OBSERVATION_FRAMES: Final[int] = 4
FUTURE_STEPS: Final[int] = 32
ACTION_DIM: Final[int] = 8
SAMPLES_PER_GROUP: Final[int] = 11

OBSERVABLE_FIELDS: Final[tuple[str, ...]] = (
    "points_history",
    "visibility_history",
    "timestamps",
    "estimated_velocity",
    "velocity_confidence",
    "action_command",
    "nominal_action_delay",
    "observation_delay",
)
TARGET_FIELDS: Final[tuple[str, ...]] = (
    "ground_truth_future_mask",
    "contact_matrix",
    "success",
    "candidate_utility",
    "hard_positive_mask",
    "hard_negative_mask",
    "target_object_mask",
)
HIDDEN_STATE_FIELDS: Final[tuple[str, ...]] = (
    "future_times",
    "exact_future_point_trajectories",
    "exact_point_velocity",
    "exact_point_acceleration",
    "exact_gripper_execution_trajectory",
    "exact_execution_delay",
    "exact_execution_duration",
    "exact_contact_matrix",
    "exact_contact_state",
)

MILESTONE2B_SCENARIOS: Final[tuple[str, ...]] = (
    "opposite_velocities",
    "different_accelerations",
    "delayed_motion_onset",
    "curved_circular_motion",
    "piecewise_motion",
    "multiple_objects_crossing",
    "faster_irrelevant_distractor",
    "action_execution_delay",
    "action_duration_mismatch",
    "near_miss_space",
    "near_miss_time",
    "intermittent_occlusion",
)

MILESTONE2B_OOD_AXES: Final[tuple[str, ...]] = (
    "unseen_velocity_magnitude",
    "unseen_acceleration",
    "unseen_curvature",
    "unseen_action_delay",
    "unseen_observation_delay",
    "unseen_occlusion_rate",
    "unseen_noise_level",
    "unseen_object_count",
    "unseen_hard_case_combination",
)

CANDIDATE_TYPES: Final[tuple[str, ...]] = (
    "successful",
    "failed",
    "wrong_timing",
    "wrong_direction",
    "near_miss",
)

NONLINEAR_SCENARIOS: Final[frozenset[str]] = frozenset(
    {
        "different_accelerations",
        "delayed_motion_onset",
        "curved_circular_motion",
        "piecewise_motion",
        "multiple_objects_crossing",
    }
)


@dataclass(frozen=True)
class Milestone2BGroupRecord:
    """One indivisible scene identity assigned before sample expansion."""

    group_id: int
    base_scene_id: int
    pair_id: int
    trajectory_family_id: int
    geometry_seed: int
    scenario_seed: int
    scenario: str
    split: str
    domain: str
    ood_axis: str


def _positive_int(name: str, value: int, minimum: int = 1) -> int:
    if not isinstance(value, (int, np.integer)) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _group_identifier(master_seed: int, domain_index: int, local_index: int) -> int:
    namespace = int(master_seed) & 0x7FFFFFFF
    if local_index >= 2**20:
        raise ValueError("too many groups in one Milestone 2B partition")
    return (namespace << 32) | (int(domain_index) << 20) | int(local_index)


def build_milestone2b_group_manifest(
    *,
    master_seed: int = 20260714,
    split_counts: Mapping[str, int] | None = None,
    ood_groups_per_axis: int = 12,
) -> dict[str, tuple[Milestone2BGroupRecord, ...]]:
    """Build deterministic ID/OOD group assignments with disjoint identities."""

    counts = dict(split_counts or {"train": 72, "val": 24, "test": 24})
    if set(counts) != {"train", "val", "test"}:
        raise ValueError("split_counts must contain exactly train, val, and test")
    for key, value in counts.items():
        counts[key] = _positive_int(f"split_counts[{key!r}]", value)
    ood_groups_per_axis = _positive_int("ood_groups_per_axis", ood_groups_per_axis)

    total = sum(counts.values())
    rng = np.random.default_rng(np.random.SeedSequence([int(master_seed), 0x3242]))
    order = rng.permutation(total).tolist()
    manifest: dict[str, tuple[Milestone2BGroupRecord, ...]] = {}
    offset = 0
    for split in ("train", "val", "test"):
        records: list[Milestone2BGroupRecord] = []
        for local_index in order[offset : offset + counts[split]]:
            group_id = _group_identifier(master_seed, 0, local_index)
            records.append(
                Milestone2BGroupRecord(
                    group_id=group_id,
                    base_scene_id=group_id,
                    pair_id=group_id,
                    trajectory_family_id=group_id,
                    geometry_seed=group_id * 2,
                    scenario_seed=group_id * 2 + 1,
                    scenario=MILESTONE2B_SCENARIOS[
                        local_index % len(MILESTONE2B_SCENARIOS)
                    ],
                    split=split,
                    domain="id",
                    ood_axis="none",
                )
            )
        manifest[split] = tuple(records)
        offset += counts[split]

    scenario_by_axis = {
        "unseen_velocity_magnitude": "opposite_velocities",
        "unseen_acceleration": "different_accelerations",
        "unseen_curvature": "curved_circular_motion",
        "unseen_action_delay": "action_execution_delay",
        "unseen_observation_delay": "delayed_motion_onset",
        "unseen_occlusion_rate": "intermittent_occlusion",
        "unseen_noise_level": "faster_irrelevant_distractor",
        "unseen_object_count": "multiple_objects_crossing",
        "unseen_hard_case_combination": "near_miss_time",
    }
    for domain_index, axis in enumerate(MILESTONE2B_OOD_AXES, start=1):
        records = []
        for local_index in range(ood_groups_per_axis):
            group_id = _group_identifier(master_seed, domain_index, local_index)
            records.append(
                Milestone2BGroupRecord(
                    group_id=group_id,
                    base_scene_id=group_id,
                    pair_id=group_id,
                    trajectory_family_id=group_id,
                    geometry_seed=group_id * 2,
                    scenario_seed=group_id * 2 + 1,
                    scenario=scenario_by_axis[axis],
                    split="test",
                    domain=f"ood_{axis.removeprefix('unseen_')}",
                    ood_axis=axis,
                )
            )
        manifest[axis] = tuple(records)

    assert_milestone2b_no_group_leakage(manifest)
    return manifest


def assert_milestone2b_no_group_leakage(
    manifest: Mapping[str, Sequence[Milestone2BGroupRecord]],
) -> None:
    """Reject any identity shared across ID splits or OOD partitions."""

    fields = (
        "group_id",
        "base_scene_id",
        "pair_id",
        "trajectory_family_id",
        "geometry_seed",
        "scenario_seed",
    )
    owners: dict[str, dict[int, str]] = {field: {} for field in fields}
    for partition, records in manifest.items():
        for record in records:
            expected = record.split if record.domain == "id" else record.ood_axis
            if expected != partition:
                raise ValueError(
                    f"group {record.group_id} belongs to {expected!r}, not {partition!r}"
                )
            for field in fields:
                value = int(getattr(record, field))
                previous = owners[field].get(value)
                if previous is not None and previous != partition:
                    raise ValueError(
                        f"{field}={value} leaks between {previous!r} and {partition!r}"
                    )
                owners[field][value] = partition


def validate_observable_mapping(observable: Mapping[str, Any]) -> None:
    """Validate the strict fair-method input whitelist.

    Passing a full sample, a hidden-state mapping, or an observable mapping
    polluted with exact physical fields is an explicit error rather than a
    silently ignored leak.
    """

    keys = set(observable)
    hidden = keys.intersection(HIDDEN_STATE_FIELDS)
    if hidden or "hidden_state" in keys:
        raise ValueError(f"fair observable input contains hidden state: {sorted(hidden)}")
    unknown = keys - set(OBSERVABLE_FIELDS)
    missing = set(OBSERVABLE_FIELDS) - keys
    if unknown or missing:
        raise ValueError(
            "observable fields do not match the Milestone 2B whitelist; "
            f"missing={sorted(missing)}, unknown={sorted(unknown)}"
        )


def _rng(identifier: int, tag: int) -> np.random.Generator:
    words = [int(identifier) & 0xFFFFFFFF, (int(identifier) >> 32) & 0xFFFFFFFF, tag]
    return np.random.default_rng(np.random.SeedSequence(words))


def _distance_to_segment(points: np.ndarray, action: np.ndarray) -> np.ndarray:
    start = action[:3]
    segment = action[3:6] - start
    denominator = max(float(np.dot(segment, segment)), 1.0e-12)
    fraction = np.sum((points - start) * segment[None, :], axis=-1) / denominator
    fraction = np.clip(fraction, 0.0, 1.0)
    closest = start[None, :] + fraction[:, None] * segment[None, :]
    return np.linalg.norm(points - closest, axis=-1)


def _action_trajectory(
    future_times: np.ndarray,
    action: np.ndarray,
    *,
    exact_delay: float,
    exact_duration: float,
) -> tuple[np.ndarray, np.ndarray]:
    fraction = np.clip((future_times - exact_delay) / max(exact_duration, 1.0e-6), 0.0, 1.0)
    trajectory = action[:3][None, :] + fraction[:, None] * (
        action[3:6] - action[:3]
    )[None, :]
    active = np.logical_and(
        future_times >= exact_delay, future_times <= exact_delay + exact_duration
    )
    return trajectory.astype(np.float32), active


def _estimate_velocity(
    points_history: np.ndarray,
    visibility_history: np.ndarray,
    timestamps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    frames, points, _ = points_history.shape
    velocity = np.zeros((points, 3), dtype=np.float32)
    confidence = np.zeros(points, dtype=np.float32)
    full_span = max(float(timestamps[-1] - timestamps[0]), 1.0e-6)
    for point_index in range(points):
        visible = np.flatnonzero(visibility_history[:, point_index])
        if len(visible) < 2:
            continue
        time = timestamps[visible].astype(np.float64)
        centered = time - float(time.mean())
        denominator = float(np.dot(centered, centered))
        if denominator <= 1.0e-12:
            continue
        position = points_history[visible, point_index].astype(np.float64)
        velocity[point_index] = np.sum(
            centered[:, None] * (position - position.mean(axis=0)), axis=0
        ) / denominator
        span = float(time[-1] - time[0])
        confidence[point_index] = min(
            1.0, (len(visible) / frames) * max(span / full_span, 0.0)
        )
    return velocity, confidence


def estimate_velocity_from_observation_history(
    points_history: np.ndarray,
    visibility_history: np.ndarray,
    timestamps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Public deterministic least-squares velocity estimator for tests/tools."""

    return _estimate_velocity(points_history, visibility_history, timestamps)


def _clone_tensor_tree(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.clone()
    if isinstance(value, Mapping):
        return {key: _clone_tensor_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_tensor_tree(item) for item in value]
    return deepcopy(value)


class Milestone2BDataset(Dataset[dict[str, Any]]):
    """Grouped history/physics dataset with five action candidates per world."""

    def __init__(
        self,
        *,
        split: str = "train",
        domain: str = "id",
        master_seed: int = 20260714,
        split_counts: Mapping[str, int] | None = None,
        ood_groups_per_axis: int = 12,
        num_points: int = 48,
        observation_frames: int = OBSERVATION_FRAMES,
        future_steps: int = FUTURE_STEPS,
        observation_noise_std: float = 0.006,
        point_dropout_rate: float = 0.04,
        occlusion_rate: float = 0.08,
        include_hidden_state: bool = True,
    ) -> None:
        super().__init__()
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test")
        if domain != "id" and domain not in MILESTONE2B_OOD_AXES:
            raise ValueError(f"unknown Milestone 2B domain {domain!r}")
        if domain != "id" and split != "test":
            raise ValueError("Milestone 2B OOD domains are test-only")
        self.num_points = _positive_int("num_points", num_points, 40)
        if self.num_points != 48:
            raise ValueError("Milestone 2B currently uses exactly 48 role-balanced points")
        self.observation_frames = _positive_int("observation_frames", observation_frames, 3)
        self.future_steps = _positive_int("future_steps", future_steps, 16)
        for name, value, upper in (
            ("observation_noise_std", observation_noise_std, None),
            ("point_dropout_rate", point_dropout_rate, 0.5),
            ("occlusion_rate", occlusion_rate, 0.8),
        ):
            value = float(value)
            if not math.isfinite(value) or value < 0.0 or (upper is not None and value > upper):
                raise ValueError(f"{name} is outside its supported finite range")
            setattr(self, name, value)
        self.master_seed = int(master_seed)
        self.split = split
        self.domain = domain
        self.include_hidden_state = bool(include_hidden_state)
        self.manifest = build_milestone2b_group_manifest(
            master_seed=self.master_seed,
            split_counts=split_counts,
            ood_groups_per_axis=ood_groups_per_axis,
        )
        self.group_records = self.manifest[split if domain == "id" else domain]
        self._cache: dict[int, tuple[dict[str, Any], ...]] = {}

    def __len__(self) -> int:
        return len(self.group_records) * SAMPLES_PER_GROUP

    def __getitem__(self, index: int) -> dict[str, Any]:
        if isinstance(index, Tensor):
            if index.numel() != 1:
                raise IndexError("dataset index tensor must be scalar")
            index = int(index.item())
        if not isinstance(index, (int, np.integer)) or isinstance(index, bool):
            raise TypeError("dataset index must be an integer")
        index = int(index)
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        group_index, expansion_index = divmod(index, SAMPLES_PER_GROUP)
        if group_index not in self._cache:
            self._cache[group_index] = self._generate_group(self.group_records[group_index])
        sample = _clone_tensor_tree(self._cache[group_index][expansion_index])
        if not self.include_hidden_state:
            sample.pop("hidden_state", None)
        return sample

    def _parameters(self, record: Milestone2BGroupRecord) -> dict[str, float]:
        rng = _rng(record.group_id, 0x504152)
        parameters = {
            "velocity_scale": float(rng.uniform(0.90, 1.12)),
            "acceleration_scale": float(rng.uniform(0.85, 1.15)),
            "curvature": float(rng.uniform(0.035, 0.075)),
            "nominal_delay": float(rng.uniform(0.07, 0.13)),
            "observation_delay": float(rng.uniform(0.0, 0.035)),
            "noise_std": self.observation_noise_std,
            "dropout_rate": self.point_dropout_rate,
            "occlusion_rate": self.occlusion_rate,
            "duration": float(rng.uniform(0.90, 1.12)),
            "radius": float(rng.uniform(0.060, 0.076)),
            "object_count": 3.0,
        }
        axis = record.ood_axis
        if axis == "unseen_velocity_magnitude":
            parameters["velocity_scale"] = float(rng.uniform(1.55, 1.85))
        elif axis == "unseen_acceleration":
            parameters["acceleration_scale"] = float(rng.uniform(1.65, 2.05))
        elif axis == "unseen_curvature":
            parameters["curvature"] = float(rng.uniform(0.13, 0.20))
        elif axis == "unseen_action_delay":
            parameters["nominal_delay"] = float(rng.uniform(0.25, 0.38))
        elif axis == "unseen_observation_delay":
            parameters["observation_delay"] = float(rng.uniform(0.16, 0.24))
        elif axis == "unseen_occlusion_rate":
            parameters["occlusion_rate"] = float(rng.uniform(0.48, 0.62))
        elif axis == "unseen_noise_level":
            parameters["noise_std"] = float(rng.uniform(0.025, 0.040))
        elif axis == "unseen_object_count":
            parameters["object_count"] = 7.0
        elif axis == "unseen_hard_case_combination":
            parameters["observation_delay"] = float(rng.uniform(0.10, 0.15))
            parameters["occlusion_rate"] = float(rng.uniform(0.28, 0.38))
            parameters["noise_std"] = float(rng.uniform(0.016, 0.024))
            parameters["curvature"] = float(rng.uniform(0.09, 0.13))
        return parameters

    def _base_geometry(
        self, record: Milestone2BGroupRecord, parameters: Mapping[str, float]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        rng = _rng(record.geometry_seed, 0x47454F)
        radius = float(parameters["radius"])
        action = np.asarray(
            [-0.52, 0.0, 0.0, 0.52, 0.0, 0.0, parameters["duration"], radius],
            dtype=np.float32,
        )
        roles = np.empty(self.num_points, dtype=np.int64)
        # 10 hard positives, 4 easy positives, 18 hard negatives, 16 easy negatives.
        roles[:10] = 0
        roles[10:14] = 1
        roles[14:32] = 2
        roles[32:] = 3
        # Candidate success concerns the moving target object.  Static easy
        # contacts remain valid mask positives but cannot make a temporally
        # wrong action count as a successful manipulation.
        target_object = roles == 0

        points = np.zeros((self.num_points, 3), dtype=np.float32)
        fractions = np.linspace(0.18, 0.82, self.num_points, dtype=np.float32)
        rng.shuffle(fractions)
        points[:, 0] = action[0] + fractions * (action[3] - action[0])
        signs = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), self.num_points)
        points[:10, 1] = signs[:10] * rng.uniform(0.15, 0.21, 10)
        points[:10, 2] = rng.normal(0.0, 0.008, 10)
        points[10:14, 1] = signs[10:14] * rng.uniform(0.006, radius * 0.35, 4)
        points[10:14, 2] = rng.normal(0.0, 0.006, 4)
        points[14:32, 1] = signs[14:32] * rng.uniform(0.012, radius * 0.45, 18)
        points[14:32, 2] = rng.normal(0.0, 0.006, 18)
        points[32:, 1] = signs[32:] * rng.uniform(0.24, 0.36, 16)
        points[32:, 2] = rng.normal(0.0, 0.05, 16)
        if record.ood_axis == "unseen_object_count":
            # The point budget remains fixed, while the far background is
            # partitioned into six instead of two independently moving object
            # bands (plus the target object): a genuine held-out object count.
            points[32:, 2] += np.resize(
                np.asarray([-0.15, -0.09, -0.03, 0.03, 0.09, 0.15], dtype=np.float32),
                16,
            )
        return points, roles, target_object.astype(np.bool_), action

    def _execution_parameters(
        self,
        record: Milestone2BGroupRecord,
        parameters: Mapping[str, float],
        action: np.ndarray,
        candidate_type: str,
    ) -> tuple[float, float]:
        exact_delay = float(parameters["nominal_delay"])
        exact_duration = float(action[6])
        if record.scenario == "action_execution_delay":
            exact_delay += 0.16
        if record.scenario == "action_duration_mismatch":
            exact_duration *= 1.28
        if record.ood_axis == "unseen_action_delay":
            exact_delay += 0.12
        if candidate_type == "failed":
            exact_delay += 1.05
        elif candidate_type == "wrong_timing":
            exact_delay += 0.30
        return exact_delay, exact_duration

    def _world_state(
        self,
        record: Milestone2BGroupRecord,
        parameters: Mapping[str, float],
        current_points: np.ndarray,
        roles: np.ndarray,
        base_action: np.ndarray,
        dynamics_variant: int,
        times: np.ndarray,
    ) -> np.ndarray:
        """Evaluate the exact point state at arbitrary past/future times."""

        times = np.asarray(times, dtype=np.float32)
        output = np.empty((len(times), self.num_points, 3), dtype=np.float32)
        exact_delay, exact_duration = self._execution_parameters(
            record, parameters, base_action, "successful"
        )
        speed_scale = float(parameters["velocity_scale"])
        accel_scale = float(parameters["acceleration_scale"])
        curvature = float(parameters["curvature"])
        rng = _rng(record.scenario_seed, 0x44594E + dynamics_variant)

        for point_index in range(self.num_points):
            p0 = current_points[point_index].astype(np.float64)
            role = int(roles[point_index])
            fraction = float(
                np.clip(
                    (p0[0] - float(base_action[0]))
                    / max(float(base_action[3] - base_action[0]), 1.0e-6),
                    0.12,
                    0.88,
                )
            )
            contact_time = exact_delay + fraction * exact_duration
            target = np.asarray([p0[0], 0.0, 0.0], dtype=np.float64)
            delta = target - p0
            t = times.astype(np.float64)

            if role == 0:
                base_velocity = delta / max(contact_time, 1.0e-4) * speed_scale
                if dynamics_variant == 1:
                    if record.scenario == "different_accelerations":
                        velocity = 0.18 * base_velocity
                        acceleration = (
                            -1.25
                            * 2.0
                            * (delta - velocity * contact_time)
                            / max(contact_time**2, 1.0e-6)
                            * accel_scale
                        )
                        position = (
                            p0[None, :]
                            + t[:, None] * velocity[None, :]
                            + 0.5 * t[:, None] ** 2 * acceleration[None, :]
                        )
                    else:
                        position = p0[None, :] - 0.92 * t[:, None] * base_velocity[None, :]
                elif record.scenario == "different_accelerations":
                    velocity = 0.18 * base_velocity
                    acceleration = (
                        2.0
                        * (delta - velocity * contact_time)
                        / max(contact_time**2, 1.0e-6)
                        * accel_scale
                    )
                    # Rescale acceleration so the OOD magnitude changes shape
                    # without moving every point completely outside the horizon.
                    if record.ood_axis == "unseen_acceleration":
                        acceleration *= 1.18
                    position = (
                        p0[None, :]
                        + t[:, None] * velocity[None, :]
                        + 0.5 * t[:, None] ** 2 * acceleration[None, :]
                    )
                elif record.scenario == "delayed_motion_onset":
                    onset = 0.075
                    moving_time = max(contact_time - onset, 1.0e-4)
                    fraction_time = np.clip((t - onset) / moving_time, 0.0, None)
                    position = p0[None, :] + fraction_time[:, None] * delta[None, :]
                    before = t < onset
                    position[before] = p0
                elif record.scenario == "curved_circular_motion":
                    # A genuine semicircular arc in the y-z plane connects
                    # the observed point to its time-aligned contact.  Past
                    # and post-contact states continue along endpoint tangents
                    # so the trajectory is continuous and has observable
                    # curvature rather than a cosmetic sinusoidal offset.
                    normalized = t / max(contact_time, 1.0e-4)
                    clipped = np.clip(normalized, 0.0, 1.0)
                    center_y = 0.5 * p0[1]
                    circle_radius = max(abs(center_y), 1.0e-4)
                    direction = 1.0 if point_index % 2 else -1.0
                    position = np.empty((len(t), 3), dtype=np.float64)
                    position[:, 0] = p0[0]
                    position[:, 1] = center_y + center_y * np.cos(np.pi * clipped)
                    position[:, 2] = (
                        (1.0 - clipped) * p0[2]
                        + direction * circle_radius * np.sin(np.pi * clipped)
                    )
                    before = normalized < 0.0
                    after = normalized > 1.0
                    start_tangent = np.asarray(
                        [0.0, 0.0, direction * circle_radius * np.pi / contact_time],
                        dtype=np.float64,
                    )
                    end_tangent = -start_tangent
                    position[before] = p0[None, :] + t[before, None] * start_tangent[None, :]
                    position[after] = target[None, :] + (
                        t[after] - contact_time
                    )[:, None] * end_tangent[None, :]
                elif record.scenario == "piecewise_motion":
                    breakpoint = 0.42 * contact_time
                    first_velocity = 0.28 * base_velocity
                    breakpoint_position = p0 + first_velocity * breakpoint
                    second_velocity = (target - breakpoint_position) / max(
                        contact_time - breakpoint, 1.0e-6
                    )
                    position = p0[None, :] + t[:, None] * first_velocity[None, :]
                    second = t > breakpoint
                    position[second] = breakpoint_position[None, :] + (
                        t[second] - breakpoint
                    )[:, None] * second_velocity[None, :]
                else:
                    position = p0[None, :] + t[:, None] * base_velocity[None, :]

                if record.scenario == "multiple_objects_crossing" or record.ood_axis in {
                    "unseen_curvature",
                    "unseen_hard_case_combination",
                }:
                    normalized = t / max(contact_time, 1.0e-4)
                    bulge = curvature * np.sin(np.pi * normalized)
                    position[:, 2] += bulge * (1.0 if point_index % 2 else -1.0)
            elif role == 1:
                position = np.broadcast_to(p0, (len(t), 3)).copy()
            elif role == 2:
                away = np.asarray(
                    [0.0, math.copysign(0.34 * speed_scale, p0[1]), 0.0],
                    dtype=np.float64,
                )
                position = p0[None, :] + t[:, None] * away[None, :]
                if record.scenario == "near_miss_time":
                    position[:, 0] += 0.10 * t
            else:
                position = np.broadcast_to(p0, (len(t), 3)).copy()
                if record.scenario == "faster_irrelevant_distractor" or record.ood_axis in {
                    "unseen_noise_level",
                    "unseen_object_count",
                }:
                    direction = rng.normal(size=3)
                    direction /= max(float(np.linalg.norm(direction)), 1.0e-8)
                    position += t[:, None] * direction[None, :] * 0.55 * speed_scale
            output[:, point_index] = position.astype(np.float32)
        return output

    def _observable_history(
        self,
        record: Milestone2BGroupRecord,
        parameters: Mapping[str, float],
        current_points: np.ndarray,
        roles: np.ndarray,
        base_action: np.ndarray,
        dynamics_variant: int,
    ) -> tuple[dict[str, Tensor], np.ndarray, np.ndarray, np.ndarray]:
        rng = _rng(record.group_id, 0x4F4253 + dynamics_variant)
        delay = float(parameters["observation_delay"])
        timestamps = np.linspace(-0.30, 0.0, self.observation_frames, dtype=np.float32)
        # Timestamp jitter is held constant across domains; each OOD partition
        # changes only its declared physical/observable parameter family.
        jitter = rng.normal(0.0, 0.006, len(timestamps))
        jitter[0] = 0.0
        jitter[-1] = 0.0
        timestamps = np.sort(timestamps + jitter.astype(np.float32)) - np.float32(delay)
        exact_history = self._world_state(
            record,
            parameters,
            current_points,
            roles,
            base_action,
            dynamics_variant,
            timestamps,
        )
        noise_std = float(parameters["noise_std"])
        observed = exact_history + rng.normal(0.0, noise_std, exact_history.shape).astype(
            np.float32
        )
        occlusion_rate = float(parameters["occlusion_rate"])
        if record.scenario == "intermittent_occlusion":
            occlusion_rate = max(occlusion_rate, 0.36)
        visibility = rng.random((self.observation_frames, self.num_points)) >= occlusion_rate
        permanent_dropout = rng.random(self.num_points) < float(parameters["dropout_rate"])
        visibility[:, permanent_dropout] = False
        # Intermittent visibility is deterministic but never repaired with exact
        # state.  Missing coordinates are zeros plus a visibility indicator.
        observed[~visibility] = 0.0
        estimated_velocity, confidence = _estimate_velocity(
            observed, visibility, timestamps
        )
        observable = {
            "points_history": torch.from_numpy(observed.copy()),
            "visibility_history": torch.from_numpy(visibility.copy()),
            "timestamps": torch.from_numpy(timestamps.copy()),
            "estimated_velocity": torch.from_numpy(estimated_velocity.copy()),
            "velocity_confidence": torch.from_numpy(confidence.copy()),
            # Filled per candidate below.
            "action_command": torch.zeros(ACTION_DIM, dtype=torch.float32),
            "nominal_action_delay": torch.tensor(
                float(parameters["nominal_delay"]), dtype=torch.float32
            ),
            "observation_delay": torch.tensor(delay, dtype=torch.float32),
        }
        return observable, exact_history, visibility, timestamps

    @staticmethod
    def _candidate_action(base_action: np.ndarray, candidate_type: str) -> np.ndarray:
        action = base_action.copy()
        if candidate_type == "failed":
            # A very late, brief execution misses the moving target while the
            # current geometry stays close; it is a temporal hard failure, not
            # a trivially far geometric negative.
            action[6] *= np.float32(0.45)
        elif candidate_type == "wrong_timing":
            action[6] *= np.float32(0.58)
        elif candidate_type == "wrong_direction":
            action[:3], action[3:6] = action[3:6].copy(), action[:3].copy()
            # A lateral reverse sweep is poor for the reference dynamics but
            # can intercept the opposite-velocity counterfactual.  Thus every
            # candidate set retains at least one physically successful option
            # without changing the paired reference action itself.
            action[1] += np.float32(0.24)
            action[4] += np.float32(0.24)
            action[7] *= np.float32(2.50)
        elif candidate_type == "near_miss":
            offset = np.float32(action[7] * 1.35)
            action[1] += offset
            action[4] += offset
        elif candidate_type != "successful":
            raise ValueError(f"unknown candidate type {candidate_type!r}")
        return action

    def _condition(self, record: Milestone2BGroupRecord) -> str:
        if record.ood_axis == "unseen_noise_level":
            return "noisy"
        if record.ood_axis == "unseen_occlusion_rate" or record.scenario == "intermittent_occlusion":
            return "occluded"
        if record.ood_axis in {"unseen_action_delay", "unseen_observation_delay"} or record.scenario in {
            "action_execution_delay",
            "action_duration_mismatch",
        }:
            return "delayed"
        if record.scenario in NONLINEAR_SCENARIOS or record.ood_axis in {
            "unseen_acceleration",
            "unseen_curvature",
            "unseen_hard_case_combination",
        }:
            return "nonlinear"
        return "clean"

    def _make_sample(
        self,
        *,
        record: Milestone2BGroupRecord,
        parameters: Mapping[str, float],
        current_points: np.ndarray,
        roles: np.ndarray,
        target_object: np.ndarray,
        base_action: np.ndarray,
        observable_template: Mapping[str, Tensor],
        dynamics_variant: int,
        candidate_type: str,
        candidate_id: int,
        expansion_index: int,
    ) -> dict[str, Any]:
        action = self._candidate_action(base_action, candidate_type)
        exact_delay, exact_duration = self._execution_parameters(
            record, parameters, action, candidate_type
        )
        # The supervised contact index is execution-phase aligned: index h in
        # the target corresponds to the same nominal phase used by the model's
        # H contact logits.  Exact and nominal timing may differ, but the time
        # axis itself is no longer mismatched during BCE supervision.
        future_times = np.linspace(
            exact_delay,
            exact_delay + exact_duration,
            self.future_steps,
            dtype=np.float32,
        )
        future_points = self._world_state(
            record,
            parameters,
            current_points,
            roles,
            base_action,
            dynamics_variant,
            future_times,
        )
        gripper, active = _action_trajectory(
            future_times,
            action,
            exact_delay=exact_delay,
            exact_duration=exact_duration,
        )
        distance = np.linalg.norm(future_points - gripper[:, None, :], axis=-1)
        contact = np.logical_and(distance <= float(action[7]), active[:, None])
        mask = contact.any(axis=0)
        target_contacts = int(np.logical_and(mask, target_object).sum())
        target_count = max(int(target_object.sum()), 1)
        utility = float(target_contacts / target_count)
        success = bool(target_contacts >= max(2, int(math.ceil(0.20 * target_count))))
        current_distance = _distance_to_segment(current_points, action)
        hard_positive = np.logical_and(mask, current_distance > float(action[7]))
        hard_negative = np.logical_and(~mask, current_distance <= float(action[7]) + 0.075)

        epsilon = 1.0e-3
        local_times = np.asarray([-epsilon, 0.0, epsilon], dtype=np.float32)
        local_state = self._world_state(
            record,
            parameters,
            current_points,
            roles,
            base_action,
            dynamics_variant,
            local_times,
        )
        velocity = (local_state[2] - local_state[0]) / (2.0 * epsilon)
        acceleration = (local_state[2] - 2.0 * local_state[1] + local_state[0]) / (
            epsilon**2
        )
        observable = _clone_tensor_tree(observable_template)
        observable["action_command"] = torch.from_numpy(action.copy())
        # A deliberately late candidate carries its scheduled delay in the
        # observable command.  Hidden execution delay can still differ due to
        # actuator/scenario mismatch, but candidate identity is never a
        # secret label cue that fair methods are unable to observe.
        nominal_candidate_delay = float(parameters["nominal_delay"])
        if candidate_type == "failed":
            nominal_candidate_delay += 1.05
        elif candidate_type == "wrong_timing":
            nominal_candidate_delay += 0.30
        observable["nominal_action_delay"] = torch.tensor(
            nominal_candidate_delay, dtype=torch.float32
        )
        validate_observable_mapping(observable)
        sample_id = record.group_id * 100 + expansion_index
        return {
            "observable": observable,
            "targets": {
                "ground_truth_future_mask": torch.from_numpy(mask.astype(np.float32)),
                "contact_matrix": torch.from_numpy(contact.astype(np.float32)),
                "success": torch.tensor(float(success), dtype=torch.float32),
                "candidate_utility": torch.tensor(utility, dtype=torch.float32),
                "hard_positive_mask": torch.from_numpy(hard_positive.copy()),
                "hard_negative_mask": torch.from_numpy(hard_negative.copy()),
                "target_object_mask": torch.from_numpy(target_object.copy()),
            },
            "hidden_state": {
                "future_times": torch.from_numpy(future_times.copy()),
                "exact_future_point_trajectories": torch.from_numpy(future_points.copy()),
                "exact_point_velocity": torch.from_numpy(velocity.astype(np.float32)),
                "exact_point_acceleration": torch.from_numpy(
                    acceleration.astype(np.float32)
                ),
                "exact_gripper_execution_trajectory": torch.from_numpy(gripper.copy()),
                "exact_execution_delay": torch.tensor(exact_delay, dtype=torch.float32),
                "exact_execution_duration": torch.tensor(
                    exact_duration, dtype=torch.float32
                ),
                "exact_contact_matrix": torch.from_numpy(contact.copy()),
                "exact_contact_state": torch.from_numpy(mask.copy()),
            },
            "metadata": {
                "sample_id": sample_id,
                "group_id": record.group_id,
                "base_scene_id": record.base_scene_id,
                "pair_id": record.pair_id,
                "trajectory_family_id": record.trajectory_family_id,
                "geometry_seed": record.geometry_seed,
                "scenario_seed": record.scenario_seed,
                "scenario": record.scenario,
                "split": record.split,
                "domain": record.domain,
                "distribution": "ID" if record.domain == "id" else "OOD",
                "ood_axis": record.ood_axis,
                "condition": self._condition(record),
                "dynamics_variant": dynamics_variant,
                "candidate_type": candidate_type,
                "candidate_id": candidate_id,
                "candidate_set_id": record.group_id * 2 + dynamics_variant,
                "counterfactual_type": (
                    "velocity"
                    if dynamics_variant == 1 and candidate_type == "successful"
                    else "action"
                    if dynamics_variant == 0 and candidate_type == "wrong_timing"
                    else "none"
                ),
                "is_irrelevant_perturbation": False,
                "observation_noise_std": float(parameters["noise_std"]),
                "occlusion_rate": float(parameters["occlusion_rate"]),
                "object_count": int(parameters["object_count"]),
            },
        }

    def _generate_group(
        self, record: Milestone2BGroupRecord
    ) -> tuple[dict[str, Any], ...]:
        parameters = self._parameters(record)
        current_points, roles, target_object, base_action = self._base_geometry(
            record, parameters
        )
        candidate_rng = _rng(record.group_id, 0x43414E)
        candidate_ids = candidate_rng.permutation(len(CANDIDATE_TYPES)).tolist()
        samples: list[dict[str, Any]] = []
        for dynamics_variant in (0, 1):
            observable, _, _, _ = self._observable_history(
                record,
                parameters,
                current_points,
                roles,
                base_action,
                dynamics_variant,
            )
            for candidate_index, candidate_type in enumerate(CANDIDATE_TYPES):
                samples.append(
                    self._make_sample(
                        record=record,
                        parameters=parameters,
                        current_points=current_points,
                        roles=roles,
                        target_object=target_object,
                        base_action=base_action,
                        observable_template=observable,
                        dynamics_variant=dynamics_variant,
                        candidate_type=candidate_type,
                        candidate_id=int(candidate_ids[candidate_index]),
                        expansion_index=len(samples),
                    )
                )

        # Scene-wide oracle provenance is derived only after all candidates are
        # labeled.  Candidate IDs are permuted and never act as a success cue.
        for dynamics_variant in (0, 1):
            start = dynamics_variant * len(CANDIDATE_TYPES)
            members = samples[start : start + len(CANDIDATE_TYPES)]
            oracle = max(
                members,
                key=lambda item: (
                    float(item["targets"]["candidate_utility"]),
                    -int(item["metadata"]["candidate_id"]),
                ),
            )
            oracle_id = int(oracle["metadata"]["candidate_id"])
            oracle_utility = float(oracle["targets"]["candidate_utility"])
            for item in members:
                item["metadata"]["oracle_candidate_id"] = oracle_id
                item["metadata"]["oracle_utility"] = oracle_utility
                item["metadata"]["is_oracle_candidate"] = (
                    int(item["metadata"]["candidate_id"]) == oracle_id
                )

        # Irrelevant partner: perturb only non-target observed histories while
        # keeping the reference action, exact physics, and every target intact.
        irrelevant = _clone_tensor_tree(samples[0])
        rng = _rng(record.group_id, 0x495252)
        background = np.flatnonzero(~target_object)
        selected = rng.choice(background, size=min(8, len(background)), replace=False)
        history = irrelevant["observable"]["points_history"]
        visibility = irrelevant["observable"]["visibility_history"]
        noise = torch.from_numpy(
            rng.normal(0.0, 0.025, (history.shape[0], len(selected), 3)).astype(np.float32)
        )
        selected_tensor = torch.as_tensor(selected, dtype=torch.long)
        visible_selected = visibility[:, selected_tensor]
        history[:, selected_tensor] += noise * visible_selected[:, :, None]
        velocity, confidence = _estimate_velocity(
            history.numpy(), visibility.numpy(), irrelevant["observable"]["timestamps"].numpy()
        )
        irrelevant["observable"]["estimated_velocity"] = torch.from_numpy(velocity)
        irrelevant["observable"]["velocity_confidence"] = torch.from_numpy(confidence)
        irrelevant["metadata"].update(
            {
                "sample_id": record.group_id * 100 + 10,
                "candidate_set_id": record.group_id * 2 + 2,
                "counterfactual_type": "irrelevant_background",
                "is_irrelevant_perturbation": True,
                "source_sample_id": samples[0]["metadata"]["sample_id"],
            }
        )
        validate_observable_mapping(irrelevant["observable"])
        samples.append(irrelevant)
        if len(samples) != SAMPLES_PER_GROUP:
            raise RuntimeError("Milestone 2B group expansion is incomplete")
        return tuple(samples)


def milestone2b_dataset_statistics(dataset: Dataset[dict[str, Any]]) -> dict[str, Any]:
    """Compute hard-case proportions over every non-duplicate candidate."""

    positive = hard_positive = negative = hard_negative = 0
    groups: set[int] = set()
    conditions: dict[str, int] = defaultdict(int)
    scenarios: dict[str, int] = defaultdict(int)
    candidate_sets: set[str] = set()
    samples = 0
    reference_samples = 0
    for index in range(len(dataset)):
        sample = dataset[index]
        metadata = sample["metadata"]
        if metadata["is_irrelevant_perturbation"]:
            continue
        groups.add(int(metadata["group_id"]))
        candidate_sets.add(str(metadata["candidate_set_id"]))
        if metadata["candidate_type"] == "successful":
            reference_samples += 1
        target = sample["targets"]["ground_truth_future_mask"] >= 0.5
        hp = sample["targets"]["hard_positive_mask"].bool()
        hn = sample["targets"]["hard_negative_mask"].bool()
        positive += int(target.sum())
        negative += int((~target).sum())
        hard_positive += int(hp.sum())
        hard_negative += int(hn.sum())
        conditions[str(metadata["condition"])] += 1
        scenarios[str(metadata["scenario"])] += 1
        samples += 1
    return {
        "groups": len(groups),
        "ordinary_samples": samples,
        "reference_samples": reference_samples,
        "candidate_sets": len(candidate_sets),
        "positive_points": positive,
        "negative_points": negative,
        "hard_positive_points": hard_positive,
        "hard_negative_points": hard_negative,
        "hard_positive_fraction": float(hard_positive / positive) if positive else 0.0,
        "hard_negative_fraction": float(hard_negative / negative) if negative else 0.0,
        "conditions": dict(sorted(conditions.items())),
        "scenarios": dict(sorted(scenarios.items())),
    }


__all__ = [
    "ACTION_DIM",
    "CANDIDATE_TYPES",
    "FUTURE_STEPS",
    "HIDDEN_STATE_FIELDS",
    "MILESTONE2B_OOD_AXES",
    "MILESTONE2B_SCENARIOS",
    "Milestone2BDataset",
    "Milestone2BGroupRecord",
    "NONLINEAR_SCENARIOS",
    "OBSERVABLE_FIELDS",
    "OBSERVATION_FRAMES",
    "SAMPLES_PER_GROUP",
    "TARGET_FIELDS",
    "assert_milestone2b_no_group_leakage",
    "build_milestone2b_group_manifest",
    "estimate_velocity_from_observation_history",
    "milestone2b_dataset_statistics",
    "validate_observable_mapping",
]
