"""Deterministic multi-candidate action-ranking views for Milestone 2.

Each source counterfactual group contributes the variant-0 physical scene and
five candidate actions: a successful reference, an off-target failure, a
wrong-timing execution, a wrong-direction execution, and a near miss.  Labels
are recomputed from the source's exact simulated point trajectory rather than
copied from the source action.  The class is intentionally a CPU-only dataset
view; it neither trains a policy nor uses oracle trajectory fields as model
inputs.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .milestone2_dataset import compute_time_aligned_contact


CANDIDATE_ACTION_TYPES = (
    "successful",
    "failed",
    "wrong_timing",
    "wrong_direction",
    "near_miss",
)


def _clone_value(value: Any) -> Any:
    return value.clone() if isinstance(value, Tensor) else value


def _scalar_int(value: Any, default: int) -> int:
    if value is None:
        return int(default)
    tensor = torch.as_tensor(value)
    if tensor.numel() != 1:
        raise ValueError("ranking identifiers must be scalar")
    return int(tensor.detach().cpu().item())


def _float_array(value: Any, *, name: str, shape_last: int | None = None) -> np.ndarray:
    array = torch.as_tensor(value, dtype=torch.float32).detach().cpu().numpy()
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    if shape_last is not None and (array.ndim == 0 or array.shape[-1] != shape_last):
        raise ValueError(f"{name} must have final dimension {shape_last}")
    return np.asarray(array, dtype=np.float32).copy()


def _resample_trajectory(
    source_times: np.ndarray,
    source_trajectory: np.ndarray,
    candidate_times: np.ndarray,
) -> np.ndarray:
    """Linearly resample a simulated trajectory at candidate physical times."""

    if source_times.ndim != 1 or source_times.size < 2:
        raise ValueError("source times must be a one-dimensional sequence of at least two steps")
    if source_trajectory.ndim != 3 or source_trajectory.shape[0] != source_times.size:
        raise ValueError("point trajectory must have shape [T, N, 3] matching source times")
    if np.any(np.diff(source_times) <= 0.0):
        raise ValueError("source times must be strictly increasing")
    if candidate_times.ndim != 1:
        raise ValueError("candidate times must be one-dimensional")
    # Ranking candidates only shorten the reference duration, but clipping also
    # keeps this dataset well-defined for a manually supplied custom action.
    clipped = np.clip(candidate_times, source_times[0], source_times[-1])
    right = np.searchsorted(source_times, clipped, side="right")
    right = np.clip(right, 1, source_times.size - 1)
    left = right - 1
    denominator = source_times[right] - source_times[left]
    fraction = (clipped - source_times[left]) / denominator
    return (
        source_trajectory[left]
        + fraction[:, None, None]
        * (source_trajectory[right] - source_trajectory[left])
    ).astype(np.float32)


def _action_unit_and_perpendicular(action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    direction = np.asarray(action[3:6] - action[:3], dtype=np.float32)
    norm = float(np.linalg.norm(direction))
    if norm <= 1.0e-8:
        unit = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        unit = direction / norm
    axis = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    if abs(float(np.dot(unit, axis))) > 0.9:
        axis = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    perpendicular = np.cross(unit, axis).astype(np.float32)
    perpendicular /= max(float(np.linalg.norm(perpendicular)), 1.0e-8)
    return unit, perpendicular


class ActionRankingDataset(Dataset[dict[str, Any]]):
    """Expand every variant-0 Milestone-2 group into five ranked actions.

    ``source_dataset`` must expose the standard Milestone-2 fields, including
    ``times``, ``point_trajectory``, ``point_roles``/``object_ids``, and the
    eight-value action.  The output preserves model-facing tensors while
    adding explicit scene/candidate/oracle metadata for ranking evaluation.
    """

    candidate_types = CANDIDATE_ACTION_TYPES

    def __init__(self, source_dataset: Dataset[Mapping[str, Any]]) -> None:
        super().__init__()
        if len(source_dataset) == 0:
            raise ValueError("source_dataset cannot be empty")
        self.source_dataset = source_dataset
        source_indices: list[int] = []
        seen_scenes: set[int] = set()
        for index in range(len(source_dataset)):
            sample = source_dataset[index]
            scene_id = _scalar_int(sample.get("group_id", sample.get("pair_id")), index)
            variant_id = _scalar_int(sample.get("variant_id"), 0)
            if scene_id not in seen_scenes and variant_id == 0:
                source_indices.append(index)
                seen_scenes.add(scene_id)
        if not source_indices:
            raise ValueError("source_dataset contains no variant-0 Milestone-2 samples")
        self.source_indices = tuple(source_indices)

    def __len__(self) -> int:
        return len(self.source_indices) * len(self.candidate_types)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if isinstance(index, Tensor):
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
        scene_index, candidate_index = divmod(index, len(self.candidate_types))
        source = self.source_dataset[self.source_indices[scene_index]]
        candidates = self._build_candidates(source)
        return candidates[candidate_index]

    @staticmethod
    def _evaluate_action(source: Mapping[str, Any], action: np.ndarray) -> dict[str, Any]:
        action = _float_array(action, name="action", shape_last=8).reshape(8)
        duration = float(action[6])
        radius = float(action[7])
        if not np.isfinite(duration) or duration <= 0.0:
            raise ValueError("candidate action duration must be positive")
        if not np.isfinite(radius) or radius <= 0.0:
            raise ValueError("candidate action radius must be positive")

        source_times = _float_array(source["times"], name="times").reshape(-1)
        source_trajectory = _float_array(
            source["point_trajectory"], name="point_trajectory", shape_last=3
        )
        candidate_times = np.linspace(
            0.0, duration, source_times.size, endpoint=True, dtype=np.float32
        )
        trajectory = _resample_trajectory(
            source_times, source_trajectory, candidate_times
        )
        fractions = candidate_times / np.float32(duration)
        gripper = (
            action[None, :3]
            + fractions[:, None] * (action[3:6] - action[:3])[None, :]
        ).astype(np.float32)
        mask, minimum_distance = compute_time_aligned_contact(trajectory, gripper, radius)

        object_ids = torch.as_tensor(source["object_ids"], dtype=torch.int64).cpu().numpy()
        target_object_id = _scalar_int(source.get("target_object_id"), 0)
        target = object_ids == target_object_id
        if not bool(np.any(target)):
            raise RuntimeError("ranking scene contains no target-object point")
        target_mask = mask[target] > 0.5
        target_utility = float(target_mask.mean())
        return {
            "action": action,
            "times": candidate_times,
            "point_trajectory": trajectory,
            "gripper_trajectory": gripper,
            "mask": mask.astype(np.float32),
            "min_contact_distance": minimum_distance.astype(np.float32),
            "success": bool(np.any(target_mask)),
            "target_utility": target_utility,
        }

    @classmethod
    def _first_failure(
        cls, source: Mapping[str, Any], actions: list[np.ndarray], *, label: str
    ) -> dict[str, Any]:
        for action in actions:
            evaluation = cls._evaluate_action(source, action)
            if not bool(evaluation["success"]):
                return evaluation
        raise RuntimeError(f"could not construct a failed {label} candidate")

    @classmethod
    def _build_candidates(cls, source: Mapping[str, Any]) -> list[dict[str, Any]]:
        source_action = _float_array(source["action"], name="action", shape_last=8).reshape(8)
        success = cls._evaluate_action(source, source_action)
        if not bool(success["success"]):
            raise RuntimeError("variant-0 source action must be a successful ranking reference")

        _, perpendicular = _action_unit_and_perpendicular(source_action)
        radius = float(source_action[7])
        source_duration = float(source_action[6])

        def shifted(distance: float) -> np.ndarray:
            result = source_action.copy()
            result[:3] += perpendicular * np.float32(distance)
            result[3:6] += perpendicular * np.float32(distance)
            return result

        failed = cls._first_failure(
            source,
            [shifted(distance) for distance in (0.45, 0.9, 1.8, 3.6)],
            label="failed",
        )

        timing_actions: list[np.ndarray] = []
        timing_fractions = (0.001, 0.005, 0.01, 0.02, 0.04, 0.08, 0.15, 0.25, 0.40, 0.60)
        for fraction in timing_fractions:
            candidate = source_action.copy()
            candidate[6] = max(1.0e-4, source_duration * np.float32(fraction))
            timing_actions.append(candidate)
        timing_evaluations = [cls._evaluate_action(source, action) for action in timing_actions]
        # Some deliberately static scenes are insensitive to duration: every
        # duration still succeeds because the path eventually passes through
        # the target.  Retain the actual recomputed label in that case rather
        # than fabricating a failure; choose the most damaging duration and a
        # deterministic earliest-fraction tie-break.
        wrong_timing = min(
            enumerate(timing_evaluations),
            key=lambda item: (float(item[1]["target_utility"]), item[0]),
        )[1]

        direction_actions: list[np.ndarray] = []
        reversed_path = source_action.copy()
        reversed_path[3:6] = source_action[:3] - (
            source_action[3:6] - source_action[:3]
        )
        direction_actions.append(reversed_path)
        for distance in (0.20, 0.50, 1.0, 2.0):
            candidate = reversed_path.copy()
            candidate[:3] += perpendicular * np.float32(distance)
            candidate[3:6] += perpendicular * np.float32(distance)
            direction_actions.append(candidate)
        wrong_direction = cls._first_failure(
            source, direction_actions, label="wrong-direction"
        )

        near_miss_actions = [
            shifted(distance)
            for distance in (
                radius + 0.01,
                radius + 0.03,
                radius + 0.06,
                radius + 0.12,
                radius + 0.24,
                radius + 0.48,
            )
        ]
        near_miss = cls._first_failure(source, near_miss_actions, label="near-miss")

        evaluations = (success, failed, wrong_timing, wrong_direction, near_miss)
        scene_id = _scalar_int(source.get("group_id", source.get("pair_id")), 0)
        utilities = [float(item["target_utility"]) for item in evaluations]
        oracle_candidate_id = max(
            range(len(utilities)), key=lambda item: (utilities[item], -item)
        )
        oracle_utility = float(utilities[oracle_candidate_id])

        outputs: list[dict[str, Any]] = []
        for candidate_id, (candidate_type, evaluation) in enumerate(
            zip(cls.candidate_types, evaluations, strict=True)
        ):
            result = {key: _clone_value(value) for key, value in source.items()}
            source_sample_id = _scalar_int(source.get("sample_id"), scene_id * 2)
            result["action"] = torch.from_numpy(np.asarray(evaluation["action"], dtype=np.float32))
            result["mask"] = torch.from_numpy(np.asarray(evaluation["mask"], dtype=np.float32))
            result["success"] = torch.tensor(
                float(bool(evaluation["success"])), dtype=torch.float32
            )
            result["times"] = torch.from_numpy(np.asarray(evaluation["times"], dtype=np.float32))
            result["point_trajectory"] = torch.from_numpy(
                np.asarray(evaluation["point_trajectory"], dtype=np.float32)
            )
            result["gripper_trajectory"] = torch.from_numpy(
                np.asarray(evaluation["gripper_trajectory"], dtype=np.float32)
            )
            result["min_contact_distance"] = torch.from_numpy(
                np.asarray(evaluation["min_contact_distance"], dtype=np.float32)
            )
            result["ranking_scene_id"] = torch.tensor(scene_id, dtype=torch.int64)
            result["candidate_set_id"] = torch.tensor(scene_id, dtype=torch.int64)
            result["candidate_id"] = torch.tensor(candidate_id, dtype=torch.int64)
            result["ranking_candidate_id"] = torch.tensor(candidate_id, dtype=torch.int64)
            result["candidate_type"] = candidate_type
            result["candidate_success"] = torch.tensor(
                float(bool(evaluation["success"])), dtype=torch.float32
            )
            result["target_utility"] = torch.tensor(
                float(evaluation["target_utility"]), dtype=torch.float32
            )
            # Canonical evaluator-facing utility fields.  The target contact
            # fraction is computed from exact trajectories for supervision and
            # oracle regret only; learned/fair predictors still receive just
            # points, observed velocity, and the candidate action.
            result["candidate_utility"] = torch.tensor(
                float(evaluation["target_utility"]), dtype=torch.float32
            )
            result["oracle_candidate_id"] = torch.tensor(
                oracle_candidate_id, dtype=torch.int64
            )
            result["oracle_target_utility"] = torch.tensor(
                oracle_utility, dtype=torch.float32
            )
            result["oracle_utility"] = torch.tensor(
                oracle_utility, dtype=torch.float32
            )
            result["is_oracle_candidate"] = torch.tensor(
                candidate_id == oracle_candidate_id, dtype=torch.bool
            )
            # Each candidate is a distinct evaluation item.  Preserve the
            # source identity separately so counterfactual and failure-case
            # joins never confuse five candidates with duplicate samples.
            result["source_sample_id"] = torch.tensor(source_sample_id, dtype=torch.int64)
            result["sample_id"] = torch.tensor(
                source_sample_id * 10 + candidate_id, dtype=torch.int64
            )
            result["counterfactual_type"] = "candidate_action_ranking"
            outputs.append(result)
        return outputs


__all__ = ["ActionRankingDataset", "CANDIDATE_ACTION_TYPES"]
