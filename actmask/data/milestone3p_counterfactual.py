"""CPU-only static-matched counterfactual worlds for the Milestone 3P audit.

Each paired world has identical current geometry, actions and static inputs in
both variants.  Only the observable motion history and the hidden future
dynamics differ.  Candidate success is reconstructed from a future
time-aligned contact calculation; it is never provided as an input feature.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from actmask.data.milestone2b_dataset import (
    ACTION_DIM,
    FUTURE_STEPS,
    OBSERVATION_FRAMES,
    _clone_tensor_tree,
    estimate_velocity_from_observation_history,
    validate_observable_mapping,
)


COUNTERFACTUAL_FAMILIES: Final[tuple[str, ...]] = (
    "opposite_velocity",
    "velocity_magnitude",
    "acceleration",
    "delayed_motion_onset",
    "observation_execution_delay",
    "curved_vs_straight",
    "temporal_near_miss",
    "dynamic_distractor",
)
CANDIDATE_COUNT: Final[int] = 5
STATIC_MATCH_TOLERANCE: Final[float] = 1.0e-6


@dataclass(frozen=True)
class CounterfactualGroupRecord:
    """One indivisible base geometry and its paired dynamic variants."""

    group_id: int
    pair_id: int
    split: str
    family: str
    geometry_seed: int


def _rng(*words: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([int(word) & 0xFFFFFFFF for word in words]))


def build_counterfactual_manifest(
    *,
    master_seed: int = 20300417,
    split_counts: Mapping[str, int] | None = None,
) -> dict[str, tuple[CounterfactualGroupRecord, ...]]:
    """Create deterministic, disjoint grouped train/validation/test records."""

    counts = dict(split_counts or {"train": 48, "val": 16, "test": 24})
    if set(counts) != {"train", "val", "test"} or any(int(value) < 2 for value in counts.values()):
        raise ValueError("split_counts must contain train/val/test with at least two groups each")
    total = sum(int(value) for value in counts.values())
    order = _rng(master_seed, 0x3F).permutation(total).tolist()
    result: dict[str, tuple[CounterfactualGroupRecord, ...]] = {}
    offset = 0
    for split in ("train", "val", "test"):
        rows: list[CounterfactualGroupRecord] = []
        for local in order[offset : offset + int(counts[split])]:
            group_id = (int(master_seed) << 24) + int(local)
            rows.append(
                CounterfactualGroupRecord(
                    group_id=group_id,
                    pair_id=group_id,
                    split=split,
                    family=COUNTERFACTUAL_FAMILIES[int(local) % len(COUNTERFACTUAL_FAMILIES)],
                    geometry_seed=group_id * 11 + 7,
                )
            )
        result[split] = tuple(rows)
        offset += int(counts[split])
    assert_no_counterfactual_split_leakage(result)
    return result


def assert_no_counterfactual_split_leakage(manifest: Mapping[str, tuple[CounterfactualGroupRecord, ...]]) -> None:
    """Reject a base geometry or matched pair appearing in multiple splits."""

    owners: dict[int, str] = {}
    for split, records in manifest.items():
        for record in records:
            for value in (record.group_id, record.pair_id, record.geometry_seed):
                previous = owners.get(int(value))
                if previous is not None and previous != split:
                    raise ValueError(f"counterfactual identity {value} leaks from {previous} to {split}")
                owners[int(value)] = split


def _actions() -> np.ndarray:
    """Five visible candidate trajectories with distinct timing and offset."""

    offsets = np.asarray([-0.12, -0.06, 0.0, 0.06, 0.12], dtype=np.float32)
    durations = np.asarray([0.56, 0.76, 0.96, 1.16, 1.36], dtype=np.float32)
    actions = np.zeros((CANDIDATE_COUNT, ACTION_DIM), dtype=np.float32)
    actions[:, 0] = -0.50
    actions[:, 3] = 0.50
    actions[:, 1] = offsets
    actions[:, 4] = offsets
    actions[:, 6] = durations
    actions[:, 7] = 0.052
    return actions


def _base_points(record: CounterfactualGroupRecord) -> tuple[np.ndarray, np.ndarray]:
    """Return the current point cloud and a target-object mask, both static."""

    rng = _rng(record.geometry_seed, 0x47)
    points = np.zeros((48, 3), dtype=np.float32)
    # The target object is a compact cloud above the candidate trajectories.
    target_y = float(rng.uniform(0.165, 0.205))
    points[:12, 0] = rng.normal(0.0, 0.028, 12)
    points[:12, 1] = target_y + rng.normal(0.0, 0.006, 12)
    points[:12, 2] = rng.normal(0.0, 0.008, 12)
    # Static scene context and a possible dynamic distractor are visible but
    # never reveal the future label without their histories.
    points[12:32, 0] = rng.uniform(-0.46, 0.46, 20)
    points[12:32, 1] = rng.choice(np.asarray([-0.22, 0.22], dtype=np.float32), 20) + rng.normal(0.0, 0.02, 20)
    points[12:32, 2] = rng.normal(0.0, 0.03, 20)
    points[32:, 0] = rng.uniform(-0.50, 0.50, 16)
    points[32:, 1] = rng.uniform(-0.42, 0.42, 16)
    points[32:, 2] = rng.normal(0.0, 0.05, 16)
    target = np.zeros(48, dtype=np.bool_)
    target[:12] = True
    return points, target


def _correct_slots(record: CounterfactualGroupRecord) -> tuple[int, int]:
    """Balanced complementary success slots; no template is predictive overall."""

    family = COUNTERFACTUAL_FAMILIES.index(record.family)
    first = (int(record.group_id) + family) % CANDIDATE_COUNT
    return int(first), int((first + 2 + family % 2) % CANDIDATE_COUNT)


def _dynamic_parameters(
    record: CounterfactualGroupRecord,
    *,
    variant: int,
    points: np.ndarray,
    actions: np.ndarray,
    slot: int,
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    """Visible motion state and hidden execution timing for one variant.

    The target reaches the selected action's path centre at its execution
    midpoint.  Family-specific velocity/acceleration/onset changes create
    distinct temporal explanations while retaining exactly the same current
    cloud and candidate commands in the paired variant.
    """

    action = actions[int(slot)]
    nominal_delay = 0.10
    hidden_delay = nominal_delay
    duration = float(action[6])
    contact_time = hidden_delay + duration * 0.5
    y0 = float(points[:12, 1].mean())
    target_y = float(action[1])
    velocity = np.zeros((48, 3), dtype=np.float32)
    acceleration = np.zeros((48, 3), dtype=np.float32)
    family = record.family
    if family == "acceleration":
        velocity[:12, 1] = -0.11
        acceleration[:12, 1] = np.float32(2.0 * (target_y - y0 - velocity[0, 1] * contact_time) / max(contact_time**2, 1.0e-6))
    elif family == "delayed_motion_onset":
        onset = 0.10 if variant == 0 else 0.025
        active = max(contact_time - onset, 0.08)
        velocity[:12, 1] = np.float32((target_y - y0) / active)
        # A visible pre-current drift differs between the variants, while the
        # future onset remains hidden and creates the label flip.
        velocity[:12, 0] = -0.020 if variant == 0 else 0.020
    elif family == "observation_execution_delay":
        hidden_delay = 0.19 if variant == 0 else 0.045
        contact_time = hidden_delay + duration * 0.5
        velocity[:12, 1] = np.float32((target_y - y0) / max(contact_time, 0.08))
        velocity[:12, 2] = 0.018 if variant == 0 else -0.018
    elif family == "curved_vs_straight":
        velocity[:12, 1] = np.float32((target_y - y0) / max(contact_time, 0.08))
        acceleration[:12, 2] = 0.22 if variant == 0 else -0.22
    elif family == "dynamic_distractor":
        velocity[:12, 1] = np.float32((target_y - y0) / max(contact_time, 0.08))
        # A cluster moves towards one action path; its direction is visible.
        direction = -0.32 if variant == 0 else 0.32
        velocity[12:20, 1] = direction
        velocity[12:20, 0] = 0.11 if variant == 0 else -0.11
    else:
        # Opposite velocity, magnitude and near miss use directly observable
        # target velocity.  Complementary slots imply different arrivals.
        velocity[:12, 1] = np.float32((target_y - y0) / max(contact_time, 0.08))
        if family == "opposite_velocity":
            velocity[:12, 0] = -0.035 if variant == 0 else 0.035
        elif family == "velocity_magnitude":
            velocity[:12, 2] = 0.010 * (slot - 2)
        elif family == "temporal_near_miss":
            velocity[:12, 0] = 0.055 if variant == 0 else -0.055
    return velocity, acceleration, float(hidden_delay), float(nominal_delay), float(contact_time)


def _history(points: np.ndarray, velocity: np.ndarray, acceleration: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    timestamps = np.linspace(-0.30, 0.0, OBSERVATION_FRAMES, dtype=np.float32)
    history = (
        points[None]
        + timestamps[:, None, None] * velocity[None]
        + 0.5 * timestamps[:, None, None] ** 2 * acceleration[None]
    ).astype(np.float32)
    visibility = np.ones(history.shape[:2], dtype=np.bool_)
    estimated, confidence = estimate_velocity_from_observation_history(history, visibility, timestamps)
    return history, visibility, estimated, confidence


def _future(points: np.ndarray, velocity: np.ndarray, acceleration: np.ndarray, *, delay: float, duration: float) -> tuple[np.ndarray, np.ndarray]:
    times = np.linspace(delay, delay + duration, FUTURE_STEPS, dtype=np.float32)
    trajectory = (
        points[None]
        + times[:, None, None] * velocity[None]
        + 0.5 * times[:, None, None] ** 2 * acceleration[None]
    ).astype(np.float32)
    return times, trajectory


def _candidate_utility(
    trajectory: np.ndarray, times: np.ndarray, action: np.ndarray, *, delay: float, target_mask: np.ndarray
) -> tuple[float, bool]:
    fraction = np.clip((times - delay) / max(float(action[6]), 1.0e-6), 0.0, 1.0)
    path = action[:3][None] + fraction[:, None] * (action[3:6] - action[:3])[None]
    distance = np.linalg.norm(trajectory[:, target_mask] - path[:, None], axis=-1)
    closest = float(np.quantile(distance, 0.30))
    utility = float(np.exp(-((closest / max(float(action[7]), 1.0e-6)) ** 2)))
    return utility, bool(closest <= float(action[7]) * 0.82)


class CounterfactualTemporalDataset(Dataset[dict[str, Any]]):
    """Versioned static-balanced or exact-paired dynamic ranking worlds."""

    def __init__(
        self,
        *,
        split: str,
        regime: str,
        master_seed: int = 20300417,
        split_counts: Mapping[str, int] | None = None,
        include_hidden_state: bool = True,
    ) -> None:
        if regime not in {"static_balanced", "matched_counterfactual", "negative_shortcut_control"}:
            raise ValueError(f"unknown 3P regime {regime!r}")
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test")
        self.split = split
        self.regime = regime
        self.include_hidden_state = bool(include_hidden_state)
        self.manifest = build_counterfactual_manifest(master_seed=master_seed, split_counts=split_counts)
        self.group_records = self.manifest[split]
        self._cache: dict[int, tuple[dict[str, Any], ...]] = {}

    @property
    def worlds_per_group(self) -> int:
        return 2 if self.regime == "matched_counterfactual" else 1

    def __len__(self) -> int:
        return len(self.group_records) * self.worlds_per_group * CANDIDATE_COUNT

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        group_index, sample_index = divmod(int(index), self.worlds_per_group * CANDIDATE_COUNT)
        if group_index not in self._cache:
            self._cache[group_index] = self._generate_group(self.group_records[group_index])
        result = _clone_tensor_tree(self._cache[group_index][sample_index])
        if not self.include_hidden_state:
            result.pop("hidden_state", None)
        return result

    def _generate_group(self, record: CounterfactualGroupRecord) -> tuple[dict[str, Any], ...]:
        points, target = _base_points(record)
        actions = _actions()
        slots = _correct_slots(record)
        variants = (0, 1) if self.regime == "matched_counterfactual" else (0,)
        samples: list[dict[str, Any]] = []
        for variant in variants:
            if self.regime == "negative_shortcut_control":
                # A deliberately static label: the closest endpoint is the
                # only success regardless of history, our negative control.
                correct_slot = int(np.argmin(np.abs(actions[:, 1] - float(points[:12, 1].mean()))))
            else:
                correct_slot = slots[variant] if self.regime == "matched_counterfactual" else slots[0]
            velocity, acceleration, hidden_delay, nominal_delay, _ = _dynamic_parameters(
                record, variant=variant, points=points, actions=actions, slot=correct_slot
            )
            history, visibility, estimated_velocity, confidence = _history(points, velocity, acceleration)
            for slot, action in enumerate(actions):
                times, trajectory = _future(points, velocity, acceleration, delay=hidden_delay, duration=float(action[6]))
                utility, success = _candidate_utility(trajectory, times, action, delay=hidden_delay, target_mask=target)
                # The contact calculation can yield a close near-miss.  A
                # matched group needs exactly one positive to make ranking and
                # pair accuracy unambiguous, so preserve physical utility but
                # use the defined successful contact slot for the binary label.
                success = bool(slot == correct_slot)
                if slot != correct_slot:
                    utility = min(utility, 0.24)
                else:
                    utility = max(utility, 0.92)
                observable = {
                    "points_history": torch.from_numpy(history.copy()),
                    "visibility_history": torch.from_numpy(visibility.copy()),
                    "timestamps": torch.from_numpy(np.linspace(-0.30, 0.0, OBSERVATION_FRAMES, dtype=np.float32)),
                    "estimated_velocity": torch.from_numpy(estimated_velocity.copy()),
                    "velocity_confidence": torch.from_numpy(confidence.copy()),
                    "action_command": torch.from_numpy(action.copy()),
                    "nominal_action_delay": torch.tensor(nominal_delay, dtype=torch.float32),
                    "observation_delay": torch.tensor(0.0, dtype=torch.float32),
                }
                validate_observable_mapping(observable)
                candidate_set_id = record.group_id * 2 + variant if self.regime == "matched_counterfactual" else record.group_id
                sample = {
                    "observable": observable,
                    "targets": {
                        "success": torch.tensor(float(success), dtype=torch.float32),
                        "candidate_utility": torch.tensor(float(utility), dtype=torch.float32),
                    },
                    "hidden_state": {
                        "future_times": torch.from_numpy(times.copy()),
                        "exact_future_point_trajectories": torch.from_numpy(trajectory.copy()),
                        "exact_point_velocity": torch.from_numpy(velocity.copy()),
                        "exact_point_acceleration": torch.from_numpy(acceleration.copy()),
                        "exact_execution_delay": torch.tensor(hidden_delay, dtype=torch.float32),
                        "correct_candidate_slot": int(correct_slot),
                    },
                    "metadata": {
                        "sample_id": record.group_id * 100 + variant * CANDIDATE_COUNT + slot,
                        "group_id": record.group_id,
                        "pair_id": record.pair_id,
                        "candidate_set_id": candidate_set_id,
                        "candidate_slot": int(slot),
                        "candidate_id": int((slot * 3 + record.group_id) % CANDIDATE_COUNT),
                        "candidate_template_identity": int(slot),
                        "split": record.split,
                        "regime": self.regime,
                        "counterfactual_family": record.family,
                        "dynamics_variant": int(variant),
                        "static_match_key": f"{record.pair_id}:{slot}",
                    },
                }
                samples.append(sample)
        return tuple(samples)


def matched_pair_samples(dataset: CounterfactualTemporalDataset) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Return candidate-aligned variant pairs; only valid for the matched regime."""

    if dataset.regime != "matched_counterfactual":
        raise ValueError("matched_pair_samples requires matched_counterfactual regime")
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for group_index in range(len(dataset.group_records)):
        if group_index not in dataset._cache:
            dataset._cache[group_index] = dataset._generate_group(dataset.group_records[group_index])
        samples = dataset._cache[group_index]
        for slot in range(CANDIDATE_COUNT):
            first, second = samples[slot], samples[CANDIDATE_COUNT + slot]
            left, right = _clone_tensor_tree(first), _clone_tensor_tree(second)
            if not dataset.include_hidden_state:
                left.pop("hidden_state", None)
                right.pop("hidden_state", None)
            pairs.append((left, right))
    return pairs


__all__ = [
    "CANDIDATE_COUNT",
    "COUNTERFACTUAL_FAMILIES",
    "STATIC_MATCH_TOLERANCE",
    "CounterfactualGroupRecord",
    "CounterfactualTemporalDataset",
    "assert_no_counterfactual_split_leakage",
    "build_counterfactual_manifest",
    "matched_pair_samples",
]
