"""Deterministic paired dynamic point-cloud data for ActMask.

The compact action representation has eight float32 values::

    [start_x, start_y, start_z,
     end_x, end_y, end_z,
     duration_seconds, gripper_radius]

Ground-truth masks are computed by sampling time along the action.  At every
sampled time, points are linearly extrapolated using their velocity and are
compared with the gripper position on its start-to-end trajectory.  A point is
positive when the two trajectories come within the gripper radius at any
sampled time.
"""

from __future__ import annotations

from typing import Dict, Union

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

ACTION_DIM = 8
ACTION_SCHEMA = (
    "start_x",
    "start_y",
    "start_z",
    "end_x",
    "end_y",
    "end_z",
    "duration_seconds",
    "gripper_radius",
)
SCENARIOS = (
    "opposite_velocity",
    "different_action_direction",
    "different_timing",
    "interception_success",
)

SampleValue = Union[Tensor, str]


class ToyDynamicDataset(Dataset[Dict[str, SampleValue]]):
    """Small procedural dataset of paired action-conditioned point masks.

    Consecutive samples form a pair: ``(0, 1)``, ``(2, 3)``, and so on.  Both
    variants in a pair contain bit-identical current point positions.  The
    scenario determines whether velocities, actions, or action timing differ.

    Args:
        num_samples: Total number of variants. Must be positive and even.
        num_points: Number of 3D points per variant. Must be at least four.
        seed: Dataset seed. Generation is deterministic for each pair.
        trajectory_steps: Number of uniformly spaced trajectory samples used
            for mask generation, including both action endpoints.
    """

    def __init__(
        self,
        num_samples: int = 128,
        num_points: int = 256,
        seed: int = 0,
        trajectory_steps: int = 24,
    ) -> None:
        super().__init__()
        if not isinstance(num_samples, int) or isinstance(num_samples, bool):
            raise TypeError("num_samples must be an integer")
        if num_samples <= 0 or num_samples % 2 != 0:
            raise ValueError("num_samples must be a positive even integer")
        if not isinstance(num_points, int) or isinstance(num_points, bool):
            raise TypeError("num_points must be an integer")
        if num_points < 4:
            raise ValueError("num_points must be at least 4")
        if not isinstance(trajectory_steps, int) or isinstance(trajectory_steps, bool):
            raise TypeError("trajectory_steps must be an integer")
        if trajectory_steps < 2:
            raise ValueError("trajectory_steps must be at least 2")
        if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool):
            raise TypeError("seed must be an integer")

        self.num_samples = num_samples
        self.num_points = num_points
        self.seed = int(seed)
        self.trajectory_steps = trajectory_steps

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> Dict[str, SampleValue]:
        if isinstance(index, torch.Tensor):
            if index.numel() != 1:
                raise IndexError("dataset index tensor must contain one value")
            index = int(index.item())
        if not isinstance(index, (int, np.integer)) or isinstance(index, bool):
            raise TypeError("index must be an integer")
        index = int(index)
        if index < 0:
            index += self.num_samples
        if index < 0 or index >= self.num_samples:
            raise IndexError(f"index {index} is outside a dataset of length {self.num_samples}")

        pair_id = index // 2
        variant_id = index % 2
        sample = self._generate_variant(pair_id, variant_id)

        return {
            "points": torch.from_numpy(sample["points"]),
            "velocities": torch.from_numpy(sample["velocities"]),
            "action": torch.from_numpy(sample["action"]),
            "mask": torch.from_numpy(sample["mask"]),
            "success": torch.tensor(sample["success"], dtype=torch.float32),
            "pair_id": torch.tensor(pair_id, dtype=torch.int64),
            "variant_id": torch.tensor(variant_id, dtype=torch.int64),
            "scenario": sample["scenario"],
        }

    def _rng_for_pair(self, pair_id: int) -> np.random.Generator:
        # SeedSequence avoids dependence on access order and safely maps signed
        # Python seeds into the uint32 input domain.
        seed_word = self.seed % (2**32)
        return np.random.default_rng(np.random.SeedSequence([seed_word, pair_id]))

    def _generate_variant(self, pair_id: int, variant_id: int) -> Dict[str, np.ndarray]:
        rng = self._rng_for_pair(pair_id)
        scenario = SCENARIOS[pair_id % len(SCENARIOS)]

        target_count = max(1, self.num_points // 3)
        distractor_count = max(1, self.num_points // 4)
        background_count = self.num_points - target_count - distractor_count

        target_center = rng.uniform(-0.16, 0.16, size=3).astype(np.float32)
        target_center[2] *= np.float32(0.4)

        angle = float(rng.uniform(0.0, 2.0 * np.pi))
        speed = float(rng.uniform(0.26, 0.30))
        direction = np.asarray([np.cos(angle), np.sin(angle), 0.0], dtype=np.float32)
        perpendicular = np.asarray([-direction[1], direction[0], 0.0], dtype=np.float32)
        target_velocity = direction * np.float32(speed)
        distractor_center = target_center + perpendicular * np.float32(0.42)
        distractor_center[2] += np.float32(0.04)

        target_offsets = np.clip(
            rng.normal(0.0, 0.018, size=(target_count, 3)), -0.045, 0.045
        ).astype(np.float32)
        distractor_offsets = np.clip(
            rng.normal(0.0, 0.022, size=(distractor_count, 3)), -0.055, 0.055
        ).astype(np.float32)
        # Exact centers guarantee a genuine target contact for successful
        # actions and a non-empty distractor contact for unsuccessful actions.
        target_offsets[0] = 0.0
        distractor_offsets[0] = 0.0

        target_points = target_center[None, :] + target_offsets
        distractor_points = distractor_center[None, :] + distractor_offsets
        background_points = rng.uniform(
            low=target_center - np.asarray([0.75, 0.75, 0.35], dtype=np.float32),
            high=target_center + np.asarray([0.75, 0.75, 0.35], dtype=np.float32),
            size=(background_count, 3),
        ).astype(np.float32)
        points = np.concatenate(
            (target_points, distractor_points, background_points), axis=0
        ).astype(np.float32, copy=False)

        target_velocities = np.repeat(
            target_velocity[None, :], target_count, axis=0
        ).astype(np.float32)
        distractor_velocities = np.zeros((distractor_count, 3), dtype=np.float32)
        background_velocities = np.clip(
            rng.normal(0.0, 0.012, size=(background_count, 3)), -0.03, 0.03
        ).astype(np.float32)
        base_velocities = np.concatenate(
            (target_velocities, distractor_velocities, background_velocities), axis=0
        ).astype(np.float32, copy=False)

        radius = np.float32(0.09)
        start = distractor_center.copy()

        if scenario == "opposite_velocity":
            duration = np.float32(1.0)
            velocities = base_velocities if variant_id == 0 else -base_velocities
            end = target_center + target_velocity * duration
        elif scenario == "different_action_direction":
            duration = np.float32(1.0)
            velocities = base_velocities
            successful_end = target_center + target_velocity * duration
            action_delta = successful_end - start
            end = successful_end if variant_id == 0 else start - action_delta
        elif scenario == "different_timing":
            successful_duration = np.float32(1.15)
            duration = successful_duration if variant_id == 0 else np.float32(0.35)
            velocities = base_velocities
            # Spatial trajectory is identical; only execution duration changes.
            end = target_center + target_velocity * successful_duration
        else:  # interception_success
            duration = np.float32(0.9)
            velocities = base_velocities
            successful_end = target_center + target_velocity * duration
            end = (
                successful_end
                if variant_id == 0
                else successful_end + perpendicular * np.float32(0.36)
            )

        action = np.concatenate(
            (
                start,
                np.asarray(end, dtype=np.float32),
                np.asarray([duration, radius], dtype=np.float32),
            )
        ).astype(np.float32, copy=False)
        velocities = np.asarray(velocities, dtype=np.float32)
        mask = self._swept_volume_mask(points, velocities, action)
        success = np.float32(bool(np.any(mask[:target_count] > 0.5)))

        return {
            "points": points,
            "velocities": velocities,
            "action": action,
            "mask": mask,
            "success": success,
            "scenario": scenario,
        }

    def _swept_volume_mask(
        self, points: np.ndarray, velocities: np.ndarray, action: np.ndarray
    ) -> np.ndarray:
        start = action[:3]
        end = action[3:6]
        duration = float(action[6])
        radius = float(action[7])

        times = np.linspace(
            0.0, duration, num=self.trajectory_steps, endpoint=True, dtype=np.float32
        )
        fractions = times / np.float32(duration)
        gripper_positions = start[None, :] + fractions[:, None] * (end - start)[None, :]
        point_positions = points[None, :, :] + times[:, None, None] * velocities[None, :, :]
        squared_distances = np.sum(
            (point_positions - gripper_positions[:, None, :]) ** 2, axis=-1
        )
        return np.any(squared_distances <= np.float32(radius * radius), axis=0).astype(
            np.float32
        )


__all__ = ["ACTION_DIM", "ACTION_SCHEMA", "SCENARIOS", "ToyDynamicDataset"]
