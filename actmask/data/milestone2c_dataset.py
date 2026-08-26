"""Milestone 2C correspondence corruptions and hard candidate-action sets.

All transformations operate on the public observable sample contract.  When a
last-frame point order changes, every point-level target is reordered with it
so that corruption never leaks a label through a stale index.
"""

from __future__ import annotations

from collections.abc import Mapping
from itertools import permutations, product
from copy import deepcopy
from typing import Any, Final

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from actmask.data.milestone2b_dataset import (
    CANDIDATE_TYPES,
    Milestone2BDataset,
    _clone_tensor_tree,
    _estimate_velocity,
    _rng,
    validate_observable_mapping,
)


CORRESPONDENCE_MODES: Final[tuple[str, ...]] = (
    "clean",
    "per_frame_permutation",
    "birth_death",
    "per_frame_resampling",
    "nearest_neighbor_noise",
    "incorrect_local_matches",
    "full_part_occlusion",
    "instance_ambiguity",
    "background_entry_exit",
)


def _numpy_rng(seed: int, sample_id: int, tag: int) -> np.random.Generator:
    return np.random.default_rng(
        np.random.SeedSequence([int(seed), int(sample_id) & 0xFFFFFFFF, int(tag)])
    )


def _point_indices(rng: np.random.Generator, original: int, target: int) -> np.ndarray:
    if target <= original:
        return rng.permutation(original)[:target]
    return rng.integers(0, original, size=target, endpoint=False)


def _reorder_point_fields(sample: dict[str, Any], index: np.ndarray) -> None:
    """Apply a final-frame order to all point-indexed public/hidden fields."""

    point_index = torch.as_tensor(index, dtype=torch.long)
    observable = sample["observable"]
    observable["points_history"] = observable["points_history"][:, point_index]
    observable["visibility_history"] = observable["visibility_history"][:, point_index]
    observable["estimated_velocity"] = observable["estimated_velocity"][point_index]
    observable["velocity_confidence"] = observable["velocity_confidence"][point_index]
    targets = sample["targets"]
    for name in (
        "ground_truth_future_mask",
        "hard_positive_mask",
        "hard_negative_mask",
        "target_object_mask",
    ):
        targets[name] = targets[name][point_index]
    targets["contact_matrix"] = targets["contact_matrix"][:, point_index]
    hidden = sample.get("hidden_state")
    if hidden is not None:
        hidden["exact_future_point_trajectories"] = hidden[
            "exact_future_point_trajectories"
        ][:, point_index]
        hidden["exact_point_velocity"] = hidden["exact_point_velocity"][point_index]
        hidden["exact_point_acceleration"] = hidden["exact_point_acceleration"][point_index]
        hidden["exact_contact_matrix"] = hidden["exact_contact_matrix"][:, point_index]
        hidden["exact_contact_state"] = hidden["exact_contact_state"][point_index]


def _recompute_observable_motion(sample: dict[str, Any]) -> None:
    observable = sample["observable"]
    velocity, confidence = _estimate_velocity(
        observable["points_history"].numpy(),
        observable["visibility_history"].numpy(),
        observable["timestamps"].numpy(),
    )
    observable["estimated_velocity"] = torch.from_numpy(velocity)
    observable["velocity_confidence"] = torch.from_numpy(confidence)
    validate_observable_mapping(observable)


class CorrespondenceCorruptionDataset(Dataset[dict[str, Any]]):
    """Deterministic observable-only correspondence perturbation view."""

    def __init__(
        self,
        dataset: Dataset[dict[str, Any]],
        *,
        mode: str,
        seed: int = 20260716,
        point_count: int | None = None,
    ) -> None:
        if mode not in CORRESPONDENCE_MODES:
            raise ValueError(f"unknown correspondence mode {mode!r}")
        if point_count is not None and point_count < 8:
            raise ValueError("point_count must be at least 8")
        self.dataset = dataset
        self.mode = mode
        self.seed = int(seed)
        self.point_count = point_count
        self.group_records = getattr(dataset, "group_records", ())

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = _clone_tensor_tree(self.dataset[index])
        metadata = sample["metadata"]
        sample_id = int(metadata["sample_id"])
        rng = _numpy_rng(self.seed, sample_id, 0x3243)
        observable = sample["observable"]
        history = observable["points_history"]
        visibility = observable["visibility_history"]
        frames, points, _ = history.shape
        # Evaluation-only synthetic identity: never exposed to the model.
        identity_history = torch.arange(points, dtype=torch.long).repeat(frames, 1)

        if self.point_count is not None and self.point_count != points:
            selection = _point_indices(rng, points, int(self.point_count))
            _reorder_point_fields(sample, selection)
            identity_history = identity_history[:, torch.as_tensor(selection, dtype=torch.long)]
            observable = sample["observable"]
            history = observable["points_history"]
            visibility = observable["visibility_history"]
            frames, points, _ = history.shape

        if self.mode in {
            "per_frame_permutation",
            "per_frame_resampling",
            "nearest_neighbor_noise",
            "incorrect_local_matches",
            "instance_ambiguity",
        }:
            for frame in range(frames):
                permutation = torch.from_numpy(rng.permutation(points)).long()
                if frame == frames - 1:
                    # Last-frame order defines output mask order; targets and
                    # hidden labels must follow it exactly.
                    _reorder_point_fields(sample, permutation.numpy())
                    identity_history = identity_history[:, permutation]
                    observable = sample["observable"]
                    history = observable["points_history"]
                    visibility = observable["visibility_history"]
                else:
                    history[frame] = history[frame, permutation]
                    visibility[frame] = visibility[frame, permutation]
                    identity_history[frame] = identity_history[frame, permutation]
            if self.mode == "per_frame_resampling":
                noise = torch.from_numpy(
                    rng.normal(0.0, 0.012, history.shape).astype(np.float32)
                )
                history += noise * visibility[:, :, None]
            elif self.mode == "nearest_neighbor_noise":
                noise = torch.from_numpy(
                    rng.normal(0.0, 0.006, history.shape).astype(np.float32)
                )
                history += noise * visibility[:, :, None]
            elif self.mode == "incorrect_local_matches":
                # Preserve a final-frame anchor but deliberately make a subset
                # of prior observations spatially plausible wrong matches.
                for frame in range(frames - 1):
                    chosen = rng.random(points) < 0.30
                    source = torch.from_numpy(rng.permutation(points)).long()
                    chosen_tensor = torch.from_numpy(chosen)
                    history[frame, chosen_tensor] = history[frame, source[chosen_tensor]]
                    visibility[frame, chosen_tensor] = visibility[
                        frame, source[chosen_tensor]
                    ]
                    identity_history[frame, chosen_tensor] = identity_history[
                        frame, source[chosen_tensor]
                    ]
            elif self.mode == "instance_ambiguity":
                ambiguous = torch.from_numpy(rng.random(points) < 0.25)
                history[:, ambiguous, 0] = history[:, ambiguous, 0].mean(dim=1, keepdim=True)

        if self.mode in {"birth_death", "full_part_occlusion", "background_entry_exit"}:
            if self.mode == "birth_death":
                missing = rng.random((frames, points)) < 0.20
            elif self.mode == "full_part_occlusion":
                missing = np.zeros((frames, points), dtype=np.bool_)
                part = rng.choice(points, size=max(1, points // 3), replace=False)
                start = int(rng.integers(0, max(1, frames - 1)))
                missing[start:, part] = True
            else:
                missing = np.zeros((frames, points), dtype=np.bool_)
                background = np.arange(points // 2, points)
                for frame in range(frames):
                    missing[frame, background] = rng.random(len(background)) < 0.35
            missing_tensor = torch.from_numpy(missing)
            visibility &= ~missing_tensor
            history[~visibility] = 0.0

        # This is evaluation-only provenance.  Models receive ``observable``
        # alone and the strict whitelist rejects hidden-state contamination.
        sample.setdefault("hidden_state", {})["evaluation_identity_history"] = identity_history
        _recompute_observable_motion(sample)
        metadata.update(
            {
                "correspondence_mode": self.mode,
                "point_count": int(points),
                "corruption_seed": self.seed,
            }
        )
        return sample


_CANDIDATE_TEMPLATES: Final[tuple[tuple[str, float, float, float, float], ...]] = (
    # family, x/y spatial offset, duration multiplier, radius multiplier
    # The first five have two plausible successes and three close temporal or
    # spatial failures.  The 10/20 candidate sets deliberately grow the
    # near-miss pool rather than the number of positives, so a score that only
    # recognizes broad trajectory proximity cannot win by candidate count.
    ("successful", 0.000, 0.000, 1.00, 1.00),
    ("successful", 0.000, 0.010, 0.94, 0.88),
    ("wrong_timing", 0.000, 0.000, 0.20, 0.82),
    ("near_miss", 0.000, 0.028, 0.82, 0.58),
    ("failed", 0.018, -0.010, 0.62, 0.72),
    ("wrong_direction", 0.018, 0.000, 0.82, 0.52),
    ("wrong_timing", -0.016, 0.020, 0.34, 0.70),
    ("near_miss", -0.030, 0.090, 1.14, 0.55),
    ("failed", 0.040, -0.035, 0.72, 0.62),
    ("wrong_direction", 0.035, 0.160, 1.16, 0.48),
    ("successful", -0.009, -0.006, 1.02, 0.80),
    ("wrong_timing", 0.010, -0.025, 0.34, 0.68),
    ("near_miss", 0.020, 0.090, 0.98, 0.54),
    ("failed", -0.040, 0.028, 0.55, 0.58),
    ("wrong_direction", -0.032, 0.160, 1.24, 0.46),
    ("successful", 0.012, -0.008, 0.96, 0.76),
    ("near_miss", -0.026, 0.090, 1.18, 0.56),
    ("wrong_timing", 0.028, -0.020, 0.34, 0.64),
    ("wrong_direction", -0.018, 0.160, 0.74, 0.50),
    ("failed", 0.045, 0.020, 0.48, 0.54),
)


def _extended_candidate_templates() -> tuple[tuple[str, float, float, float, float], ...]:
    """Additional physically relabelled near-misses for the C50 protocol.

    The template is only an observable action perturbation.  Success and
    utility continue to be regenerated from the 2B physics label constructor;
    neither slot nor template name is exposed to a learned model.
    """

    records: list[tuple[str, float, float, float, float]] = []
    families = ("near_miss", "wrong_timing", "failed", "wrong_direction", "near_miss")
    for index in range(30):
        family = families[index % len(families)]
        ring = 1 + index // len(families)
        sign = -1.0 if index % 2 else 1.0
        dx = sign * (0.012 + 0.006 * ring)
        dy = sign * (0.018 + 0.011 * (index % 5))
        duration = 0.42 + 0.10 * ((3 * index) % 7)
        radius = 0.42 + 0.06 * ((5 * index) % 7)
        records.append((family, dx, dy, duration, radius))
    return tuple(records)


_CANDIDATE_TEMPLATES_50: Final[tuple[tuple[str, float, float, float, float], ...]] = (
    *_CANDIDATE_TEMPLATES,
    *_extended_candidate_templates(),
)


class HardCandidateActionDataset(Dataset[dict[str, Any]]):
    """Physically relabeled 5/10/20/50-candidate ranking worlds.

    Samples are regenerated through the 2B label constructor rather than
    duplicated.  Candidate IDs are independently permuted, so neither ID nor
    generation slot is a success/utility cue.
    """

    def __init__(
        self,
        *,
        candidate_count: int,
        split: str = "test",
        domain: str = "id",
        master_seed: int = 20260717,
        split_counts: Mapping[str, int] | None = None,
        ood_groups_per_axis: int = 1,
        template_indices: tuple[int, ...] | None = None,
        candidate_distribution_id: str = "legacy_prefix",
        **dataset_kwargs: Any,
    ) -> None:
        if candidate_count not in {5, 10, 20, 50}:
            raise ValueError("candidate_count must be one of 5, 10, 20, 50")
        if not isinstance(candidate_distribution_id, str) or not candidate_distribution_id:
            raise ValueError("candidate_distribution_id must be a non-empty string")
        # The default is deliberately the historical prefix construction.  A
        # caller may opt into a fixed alternative mixture for a separately
        # versioned evaluation, but must name and fully specify that mixture
        # rather than relying on an implicit slice of the template pool.
        if template_indices is None:
            resolved_template_indices = tuple(range(int(candidate_count)))
            custom_template_indices = False
        else:
            resolved_template_indices = tuple(int(value) for value in template_indices)
            custom_template_indices = True
            if len(resolved_template_indices) != int(candidate_count):
                raise ValueError("template_indices length must equal candidate_count")
            if len(set(resolved_template_indices)) != len(resolved_template_indices):
                raise ValueError("template_indices must not contain duplicates")
            if any(value < 0 or value >= len(_CANDIDATE_TEMPLATES_50) for value in resolved_template_indices):
                raise ValueError(
                    f"template_indices must be within [0, {len(_CANDIDATE_TEMPLATES_50) - 1}]"
                )
        self.base = Milestone2BDataset(
            split=split,
            domain=domain,
            master_seed=master_seed,
            split_counts=split_counts,
            ood_groups_per_axis=ood_groups_per_axis,
            **dataset_kwargs,
        )
        self.candidate_count = int(candidate_count)
        self.template_indices = resolved_template_indices
        self.candidate_distribution_id = str(candidate_distribution_id)
        self._custom_template_indices = custom_template_indices
        self.group_records = self.base.group_records
        self._cache: dict[int, tuple[dict[str, Any], ...]] = {}

    def __len__(self) -> int:
        return len(self.group_records) * self.candidate_count

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        group_index, candidate_index = divmod(int(index), self.candidate_count)
        if group_index not in self._cache:
            self._cache[group_index] = self._generate_group(self.group_records[group_index])
        return _clone_tensor_tree(self._cache[group_index][candidate_index])

    def _generate_group(self, record: Any) -> tuple[dict[str, Any], ...]:
        parameters = self.base._parameters(record)
        current_points, roles, target_object, base_action = self.base._base_geometry(
            record, parameters
        )
        observable, _, _, _ = self.base._observable_history(
            record,
            parameters,
            current_points,
            roles,
            base_action,
            0,
        )
        identifier_rng = _rng(record.group_id, 0x324352)
        identifiers = identifier_rng.permutation(self.candidate_count).tolist()
        samples: list[dict[str, Any]] = []
        for slot, template_index in enumerate(self.template_indices):
            template = _CANDIDATE_TEMPLATES_50[template_index]
            candidate_type, dx, dy, duration_scale, radius_scale = template
            modified = base_action.copy()
            modified[[0, 3]] += np.float32(dx)
            modified[[1, 4]] += np.float32(dy)
            modified[6] *= np.float32(duration_scale)
            modified[7] *= np.float32(radius_scale)
            sample = self.base._make_sample(
                record=record,
                parameters=parameters,
                current_points=current_points,
                roles=roles,
                target_object=target_object,
                base_action=modified,
                observable_template=observable,
                dynamics_variant=0,
                candidate_type=candidate_type,
                candidate_id=int(identifiers[slot]),
                expansion_index=slot,
            )
            sample["metadata"].update(
                {
                    "candidate_set_id": int(record.group_id),
                    "candidate_slot": slot,
                    "candidate_count": self.candidate_count,
                    "candidate_family": candidate_type,
                    "is_irrelevant_perturbation": False,
                    "counterfactual_type": "none",
                }
            )
            # Keep legacy samples byte-for-byte compatible at the metadata
            # contract level.  The additional template provenance is exposed
            # only for an explicitly opted-in, versioned distribution shift;
            # it is evaluator metadata and never part of ``observable``.
            if self._custom_template_indices or self.candidate_distribution_id != "legacy_prefix":
                sample["metadata"].update(
                    {
                        "candidate_template_index": int(template_index),
                        "candidate_distribution_id": self.candidate_distribution_id,
                    }
                )
            samples.append(sample)
        best = max(
            samples,
            key=lambda item: (
                float(item["targets"]["candidate_utility"]),
                -int(item["metadata"]["candidate_id"]),
            ),
        )
        for sample in samples:
            sample["metadata"]["oracle_candidate_id"] = int(best["metadata"]["candidate_id"])
            sample["metadata"]["oracle_utility"] = float(best["targets"]["candidate_utility"])
        return tuple(samples)


def candidate_set_statistics(dataset: Dataset[dict[str, Any]]) -> dict[str, Any]:
    """Summarize physical success/utility properties of ranking worlds."""

    groups: dict[int, list[dict[str, Any]]] = {}
    for index in range(len(dataset)):
        sample = dataset[index]
        groups.setdefault(int(sample["metadata"]["candidate_set_id"]), []).append(sample)
    successful = [
        sum(float(item["targets"]["success"]) >= 0.5 for item in members)
        for members in groups.values()
    ]
    return {
        "groups": len(groups),
        "expanded_samples": len(dataset),
        "candidate_count": len(next(iter(groups.values()))) if groups else 0,
        "successful_candidates_mean": float(np.mean(successful)) if successful else 0.0,
        "successful_candidates_min": int(min(successful)) if successful else 0,
        "worlds_with_success_fraction": float(np.mean(np.asarray(successful) > 0))
        if successful
        else 0.0,
    }


TRANSFORM_MODES: Final[tuple[str, ...]] = (
    "random_so3",
    "axis_permutation",
    "sign_flip",
)


def _orthogonal_transform(seed: int, group_id: int, mode: str) -> Tensor:
    """Deterministic group-shared transform; rows use ``value @ R.T``."""

    if mode not in TRANSFORM_MODES:
        raise ValueError(f"unknown transform mode {mode!r}")
    rng = _numpy_rng(seed, group_id, 0x524F54)
    if mode == "random_so3":
        matrix, upper = np.linalg.qr(rng.normal(size=(3, 3)))
        matrix *= np.sign(np.diag(upper))[None, :]
        if np.linalg.det(matrix) < 0.0:
            matrix[:, 0] *= -1.0
    elif mode == "axis_permutation":
        order = tuple(permutations((0, 1, 2)))[int(rng.integers(6))]
        matrix = np.zeros((3, 3), dtype=np.float64)
        matrix[np.arange(3), np.asarray(order)] = 1.0
    else:
        signs = tuple(product((-1.0, 1.0), repeat=3))
        matrix = np.diag(np.asarray(signs[int(rng.integers(len(signs)))], dtype=np.float64))
    return torch.from_numpy(matrix.astype(np.float32))


class RigidTransformDataset(Dataset[dict[str, Any]]):
    """Observation-only rigid transform view with candidate-group consistency.

    Targets and candidate construction remain unchanged.  All candidate actions
    in the same ranking world use the same transform, so transformed ranking is
    a genuine invariance test rather than a different candidate-set task.
    """

    def __init__(self, dataset: Dataset[dict[str, Any]], *, mode: str, seed: int = 20260911) -> None:
        if mode not in TRANSFORM_MODES:
            raise ValueError(f"unknown transform mode {mode!r}")
        self.dataset = dataset
        self.mode = mode
        self.seed = int(seed)
        self.group_records = getattr(dataset, "group_records", ())

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = _clone_tensor_tree(self.dataset[index])
        metadata = sample["metadata"]
        group_id = int(metadata.get("candidate_set_id", metadata["group_id"]))
        rotation = _orthogonal_transform(self.seed, group_id, self.mode)
        observable = sample["observable"]
        observable["points_history"] = observable["points_history"] @ rotation.T
        observable["estimated_velocity"] = observable["estimated_velocity"] @ rotation.T
        action = observable["action_command"].clone()
        action[:3] = action[:3] @ rotation.T
        action[3:6] = action[3:6] @ rotation.T
        observable["action_command"] = action
        hidden = sample.get("hidden_state")
        if hidden is not None:
            for key in (
                "exact_future_point_trajectories",
                "exact_point_velocity",
                "exact_point_acceleration",
                "exact_gripper_execution_trajectory",
            ):
                if key in hidden:
                    hidden[key] = hidden[key] @ rotation.T
        metadata.update({
            "rigid_transform_mode": self.mode,
            "rigid_transform_seed": self.seed,
        })
        validate_observable_mapping(observable)
        return sample


__all__ = [
    "CORRESPONDENCE_MODES",
    "CorrespondenceCorruptionDataset",
    "HardCandidateActionDataset",
    "RigidTransformDataset",
    "TRANSFORM_MODES",
    "candidate_set_statistics",
]
