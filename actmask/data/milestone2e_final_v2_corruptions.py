"""Versioned group-shared ranking corruptions for Milestone 2E-Final.

The legacy :mod:`milestone2c_dataset` corruption view intentionally keys its
randomness by ``sample_id``.  That is appropriate for its original
per-sample correspondence diagnostic, but it is not a valid ranking-world
view: candidate actions belonging to one scene can then receive different
observations.  This module is deliberately separate from that legacy code.

``GroupSharedRankingCorruptionDataset`` constructs one observable scene view
per *candidate set*, keyed only by ``candidate_set_id`` (or ``group_id`` when
the former is not present).  It overlays that same view on every candidate in
the ranking world while preserving the candidate's action, timing, scalar
labels, identifiers, and utility.  Point-indexed targets and evaluator-only
provenance follow the final-frame point order, so the view does not create a
stale-index label leak.

This module is v2-only and must not be used to regenerate or reinterpret
historical Milestone 2E artifacts.
"""

from __future__ import annotations

import hashlib
from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from typing import Any, Final, Literal

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from actmask.data.milestone2b_dataset import (
    _clone_tensor_tree,
    _estimate_velocity,
    validate_observable_mapping,
)


GROUP_SHARED_RANKING_CORRUPTION_MODES: Final[tuple[str, ...]] = (
    "per_frame_resampling",
    "full_part_occlusion",
)
"""Observable ranking-world corruptions supported by the final-v2 protocol."""

_GROUP_SHARED_SCHEMA_VERSION: Final[str] = "milestone2e-final-v2-group-shared-corruption-v1"
_RESAMPLING_NOISE_STD: Final[float] = 0.012
_OCCLUDED_PART_FRACTION: Final[float] = 1.0 / 3.0


def _scalar_metadata_value(value: Any, *, field: str) -> Hashable:
    """Normalize a scalar metadata value without ever consulting sample ID."""

    if isinstance(value, Tensor):
        if value.numel() != 1:
            raise ValueError(f"metadata[{field!r}] must be scalar")
        value = value.detach().cpu().item()
    elif isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (str, int, float, bool)):
        if isinstance(value, str) and not value:
            raise ValueError(f"metadata[{field!r}] must not be empty")
        return value
    raise TypeError(f"metadata[{field!r}] must be a scalar string or number")


def _ranking_world_key(metadata: Mapping[str, Any]) -> tuple[str, Hashable]:
    """Return the protocol's group key, preferring ``candidate_set_id``.

    ``sample_id`` is intentionally absent from this function and from all RNG
    calls in this module.  A fallback ``group_id`` keeps the wrapper useful for
    a grouped ranking fixture that predates explicit candidate-set IDs.
    """

    if "candidate_set_id" in metadata:
        return "candidate_set_id", _scalar_metadata_value(
            metadata["candidate_set_id"], field="candidate_set_id"
        )
    if "group_id" in metadata:
        return "group_id", _scalar_metadata_value(metadata["group_id"], field="group_id")
    raise KeyError("group-shared ranking corruption requires candidate_set_id or group_id")


def _group_rng(*, seed: int, group_key: tuple[str, Hashable], tag: str) -> np.random.Generator:
    """Create a stable RNG keyed only by seed, world identity, and mode tag."""

    # Python's process-randomized ``hash`` would make a cache/view unstable.
    # The type tag avoids silently conflating e.g. integer ``1`` and string
    # ``"1"`` if an external grouped dataset uses both identifiers.
    encoded = repr((group_key[0], type(group_key[1]).__qualname__, group_key[1], tag)).encode("utf-8")
    digest = hashlib.sha256(encoded).digest()
    words = [
        int(seed) & 0xFFFFFFFF,
        (int(seed) >> 32) & 0xFFFFFFFF,
        *[int.from_bytes(digest[offset : offset + 4], "big") for offset in range(0, 16, 4)],
    ]
    return np.random.default_rng(np.random.SeedSequence(words))


def _identity_history(sample: Mapping[str, Any], *, frames: int, points: int, device: torch.device) -> Tensor:
    """Return evaluator-only point identity provenance aligned to the input."""

    hidden = sample.get("hidden_state")
    if isinstance(hidden, Mapping) and isinstance(hidden.get("evaluation_identity_history"), Tensor):
        identity = hidden["evaluation_identity_history"]
        if tuple(identity.shape) != (frames, points):
            raise ValueError(
                "evaluation_identity_history must match points_history [frames, points]"
            )
        return identity.to(device=device, dtype=torch.long).clone()
    return torch.arange(points, dtype=torch.long, device=device).repeat(frames, 1)


def _as_numpy(tensor: Tensor, *, name: str) -> np.ndarray:
    if tensor.device.type != "cpu":
        raise ValueError(f"{name} must be on CPU for the frozen final-v2 corruption protocol")
    return tensor.detach().numpy()


def _recompute_motion(
    *, history: Tensor, visibility: Tensor, timestamps: Tensor
) -> tuple[Tensor, Tensor]:
    """Recompute the public motion fields from the corrupted observable view."""

    if visibility.dtype != torch.bool:
        raise TypeError("visibility_history must use bool dtype")
    velocity, confidence = _estimate_velocity(
        _as_numpy(history, name="points_history"),
        _as_numpy(visibility, name="visibility_history"),
        _as_numpy(timestamps, name="timestamps"),
    )
    return (
        torch.as_tensor(velocity, dtype=history.dtype, device=history.device),
        torch.as_tensor(confidence, dtype=history.dtype, device=history.device),
    )


def _take_point_axis(value: Tensor, *, order: Tensor, axis: int, field: str) -> Tensor:
    """Reorder a validated point axis, retaining each candidate's own labels."""

    if value.ndim <= axis or value.shape[axis] != order.numel():
        raise ValueError(f"{field} does not have the expected point axis")
    return value.index_select(axis, order.to(device=value.device)).clone()


def _point_order_for_identity(
    *, source_identity: Tensor, destination_identity: Tensor
) -> Tensor:
    """Map a candidate's current final point order to the canonical scene.

    Raw ranking worlds use ``arange(points)`` identity, in which case this is
    simply the reference world's final-frame permutation.  Computing it from
    identity provenance also makes the v2 wrapper safe to compose with an
    already point-reordered *group-consistent* source: candidate-local labels
    still follow the physical point represented at every output position.
    """

    if source_identity.ndim != 1 or destination_identity.ndim != 1:
        raise ValueError("final-frame identity provenance must be one-dimensional")
    if source_identity.numel() != destination_identity.numel():
        raise ValueError("candidate and canonical identity histories disagree on point count")
    source_values = [int(value) for value in source_identity.detach().cpu().tolist()]
    destination_values = [int(value) for value in destination_identity.detach().cpu().tolist()]
    if len(set(source_values)) != len(source_values) or len(set(destination_values)) != len(destination_values):
        raise ValueError("final-frame evaluation identity must be a unique point permutation")
    source_index = {identity: index for index, identity in enumerate(source_values)}
    if set(source_index) != set(destination_values):
        raise ValueError("candidate and canonical identity histories refer to different points")
    return torch.as_tensor(
        [source_index[identity] for identity in destination_values],
        dtype=torch.long,
        device=source_identity.device,
    )


def _reorder_candidate_point_fields(sample: dict[str, Any], *, final_order: Tensor) -> None:
    """Align candidate-local targets and hidden evaluator fields to final order."""

    if final_order.ndim != 1 or final_order.dtype != torch.long:
        raise TypeError("final point order must be a one-dimensional long tensor")
    expected = set(range(final_order.numel()))
    if set(final_order.detach().cpu().tolist()) != expected:
        raise ValueError("final point order must be a complete permutation")

    targets = sample.get("targets")
    if not isinstance(targets, Mapping):
        raise TypeError("ranking corruption requires a targets mapping")
    for field in (
        "ground_truth_future_mask",
        "hard_positive_mask",
        "hard_negative_mask",
        "target_object_mask",
    ):
        value = targets.get(field)
        if not isinstance(value, Tensor):
            raise TypeError(f"targets[{field!r}] must be a tensor")
        targets[field] = _take_point_axis(value, order=final_order, axis=0, field=f"targets.{field}")
    contact = targets.get("contact_matrix")
    if not isinstance(contact, Tensor):
        raise TypeError("targets['contact_matrix'] must be a tensor")
    targets["contact_matrix"] = _take_point_axis(
        contact, order=final_order, axis=1, field="targets.contact_matrix"
    )

    hidden = sample.get("hidden_state")
    if hidden is None:
        return
    if not isinstance(hidden, Mapping):
        raise TypeError("hidden_state must be a mapping when present")
    point_axes = {
        "exact_future_point_trajectories": 1,
        "exact_point_velocity": 0,
        "exact_point_acceleration": 0,
        "exact_contact_matrix": 1,
        "exact_contact_state": 0,
    }
    for field, axis in point_axes.items():
        value = hidden.get(field)
        if value is not None:
            if not isinstance(value, Tensor):
                raise TypeError(f"hidden_state[{field!r}] must be a tensor")
            hidden[field] = _take_point_axis(
                value, order=final_order, axis=axis, field=f"hidden_state.{field}"
            )


@dataclass(frozen=True)
class _CorruptedScene:
    """One canonical observable scene and identity provenance for a world."""

    points_history: Tensor
    visibility_history: Tensor
    timestamps: Tensor
    estimated_velocity: Tensor
    velocity_confidence: Tensor
    evaluation_identity_history: Tensor
    final_order: Tensor


class GroupSharedRankingCorruptionDataset(Dataset[dict[str, Any]]):
    """A deterministic, candidate-set-shared ranking corruption view.

    The wrapped dataset must store each ranking world in a contiguous block of
    ``candidate_count`` samples.  ``HardCandidateActionDataset`` has this
    contract natively.  A wrapper that obscures ``candidate_count`` can pass it
    explicitly.  The first candidate's scene becomes the world reference;
    only its scene fields are used, never its action, timing, scalar labels,
    IDs, or utility.

    ``per_frame_resampling`` retains the frozen legacy condition's semantics:
    each frame gets a separate deterministic point permutation plus
    visibility-masked measurement jitter.  ``full_part_occlusion`` removes
    one deterministic point subset from a suffix of the history.  Both are
    keyed by candidate-set/group identity, not by sample identity.
    """

    schema_version: Final[str] = _GROUP_SHARED_SCHEMA_VERSION

    def __init__(
        self,
        dataset: Dataset[dict[str, Any]],
        *,
        mode: Literal["per_frame_resampling", "full_part_occlusion"],
        seed: int = 20262031,
        candidate_count: int | None = None,
    ) -> None:
        if mode not in GROUP_SHARED_RANKING_CORRUPTION_MODES:
            raise ValueError(f"unknown group-shared ranking corruption mode {mode!r}")
        inferred = getattr(dataset, "candidate_count", None)
        if candidate_count is None:
            candidate_count = inferred
        if candidate_count is None:
            records = tuple(getattr(dataset, "group_records", ()))
            if not records or len(dataset) % len(records):
                raise TypeError(
                    "candidate_count is required unless dataset exposes complete grouped worlds"
                )
            candidate_count = len(dataset) // len(records)
        if isinstance(candidate_count, Tensor):
            if candidate_count.numel() != 1:
                raise ValueError("candidate_count must be scalar")
            candidate_count = int(candidate_count.item())
        if int(candidate_count) < 2:
            raise ValueError("group-shared ranking corruption needs at least two candidates per world")
        if len(dataset) % int(candidate_count):
            raise ValueError("dataset length must contain complete contiguous candidate worlds")

        self.dataset = dataset
        self.mode = mode
        self.seed = int(seed)
        self.candidate_count = int(candidate_count)
        self.group_records = getattr(dataset, "group_records", ())
        self.view_name = f"v2_group_shared_{mode}"
        self._scene_cache: dict[tuple[str, Hashable], _CorruptedScene] = {}

    def __len__(self) -> int:
        return len(self.dataset)

    def _reference_sample(self, *, index: int, group_key: tuple[str, Hashable]) -> dict[str, Any]:
        """Load the contiguous world's canonical source scene and verify grouping."""

        reference_index = (int(index) // self.candidate_count) * self.candidate_count
        reference = _clone_tensor_tree(self.dataset[reference_index])
        metadata = reference.get("metadata")
        if not isinstance(metadata, Mapping):
            raise TypeError("ranking corruption requires a metadata mapping")
        if _ranking_world_key(metadata) != group_key:
            raise ValueError(
                "candidate worlds must be contiguous and share candidate_set_id/group_id"
            )
        return reference

    def _make_scene(
        self, *, reference: Mapping[str, Any], group_key: tuple[str, Hashable]
    ) -> _CorruptedScene:
        observable = reference.get("observable")
        if not isinstance(observable, Mapping):
            raise TypeError("ranking corruption requires an observable mapping")
        validate_observable_mapping(observable)
        history = observable["points_history"]
        visibility = observable["visibility_history"]
        timestamps = observable["timestamps"]
        if not all(isinstance(value, Tensor) for value in (history, visibility, timestamps)):
            raise TypeError("points_history, visibility_history, and timestamps must be tensors")
        if history.ndim != 3 or history.shape[-1] != 3:
            raise ValueError("points_history must have [frames, points, 3] shape")
        frames, points, _ = history.shape
        if tuple(visibility.shape) != (frames, points):
            raise ValueError("visibility_history must have [frames, points] shape")
        if tuple(timestamps.shape) != (frames,):
            raise ValueError("timestamps must have [frames] shape")
        if visibility.dtype != torch.bool:
            raise TypeError("visibility_history must use bool dtype")
        if history.device.type != "cpu" or visibility.device.type != "cpu" or timestamps.device.type != "cpu":
            raise ValueError("final-v2 group-shared corruption is CPU-only")

        corrupted_history = history.clone()
        corrupted_visibility = visibility.clone()
        identity = _identity_history(reference, frames=frames, points=points, device=history.device)
        final_order = torch.arange(points, dtype=torch.long, device=history.device)
        rng = _group_rng(seed=self.seed, group_key=group_key, tag=self.mode)

        if self.mode == "per_frame_resampling":
            for frame in range(frames):
                order = torch.as_tensor(rng.permutation(points), dtype=torch.long, device=history.device)
                corrupted_history[frame] = history[frame].index_select(0, order)
                corrupted_visibility[frame] = visibility[frame].index_select(0, order)
                identity[frame] = identity[frame].index_select(0, order)
                if frame == frames - 1:
                    final_order = order
            noise = torch.as_tensor(
                rng.normal(0.0, _RESAMPLING_NOISE_STD, size=tuple(corrupted_history.shape)),
                dtype=history.dtype,
                device=history.device,
            )
            corrupted_history = corrupted_history + noise * corrupted_visibility.unsqueeze(-1).to(history.dtype)
        else:
            missing = np.zeros((frames, points), dtype=np.bool_)
            part_size = max(1, int(np.floor(points * _OCCLUDED_PART_FRACTION)))
            part = rng.choice(points, size=part_size, replace=False)
            start = int(rng.integers(0, max(1, frames - 1)))
            missing[start:, part] = True
            missing_tensor = torch.as_tensor(missing, dtype=torch.bool, device=history.device)
            corrupted_visibility = corrupted_visibility & ~missing_tensor
            corrupted_history = corrupted_history.clone()
            corrupted_history[~corrupted_visibility] = 0.0

        velocity, confidence = _recompute_motion(
            history=corrupted_history,
            visibility=corrupted_visibility,
            timestamps=timestamps,
        )
        return _CorruptedScene(
            points_history=corrupted_history,
            visibility_history=corrupted_visibility,
            timestamps=timestamps.clone(),
            estimated_velocity=velocity,
            velocity_confidence=confidence,
            evaluation_identity_history=identity,
            final_order=final_order,
        )

    def _scene_for(self, *, index: int, group_key: tuple[str, Hashable]) -> _CorruptedScene:
        cached = self._scene_cache.get(group_key)
        if cached is None:
            cached = self._make_scene(
                reference=self._reference_sample(index=index, group_key=group_key), group_key=group_key
            )
            self._scene_cache[group_key] = cached
        return cached

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        sample = _clone_tensor_tree(self.dataset[index])
        metadata = sample.get("metadata")
        if not isinstance(metadata, Mapping):
            raise TypeError("ranking corruption requires a metadata mapping")
        group_key = _ranking_world_key(metadata)
        scene = self._scene_for(index=index, group_key=group_key)

        observable = sample.get("observable")
        if not isinstance(observable, Mapping):
            raise TypeError("ranking corruption requires an observable mapping")
        history = observable["points_history"]
        if not isinstance(history, Tensor) or history.ndim != 3:
            raise ValueError("points_history must have [frames, points, 3] shape")
        frames, points, _ = history.shape
        if scene.final_order.numel() != points:
            raise ValueError("all candidates in a ranking world must share point count")
        candidate_identity = _identity_history(
            sample, frames=frames, points=points, device=history.device
        )
        candidate_final_order = _point_order_for_identity(
            source_identity=candidate_identity[-1],
            destination_identity=scene.evaluation_identity_history[-1],
        )

        # Deliberately replace scene fields only.  Candidate-specific
        # ``action_command``, ``nominal_action_delay``, and
        # ``observation_delay`` remain in this candidate's original mapping.
        observable["points_history"] = scene.points_history.clone()
        observable["visibility_history"] = scene.visibility_history.clone()
        observable["timestamps"] = scene.timestamps.clone()
        observable["estimated_velocity"] = scene.estimated_velocity.clone()
        observable["velocity_confidence"] = scene.velocity_confidence.clone()
        _reorder_candidate_point_fields(sample, final_order=candidate_final_order)
        hidden = sample.setdefault("hidden_state", {})
        if not isinstance(hidden, Mapping):
            raise TypeError("hidden_state must be a mapping when present")
        # Identity provenance tracks the per-frame source point at every
        # output index.  It is evaluator-only and is never added to
        # ``observable``.
        hidden["evaluation_identity_history"] = scene.evaluation_identity_history.clone()
        validate_observable_mapping(observable)
        metadata.update(
            {
                "ranking_view": self.view_name,
                "ranking_corruption_mode": self.mode,
                "ranking_corruption_seed": self.seed,
                "ranking_corruption_group_key_field": group_key[0],
                "ranking_corruption_group_key": str(group_key[1]),
                "ranking_corruption_schema_version": self.schema_version,
            }
        )
        return sample


__all__ = [
    "GROUP_SHARED_RANKING_CORRUPTION_MODES",
    "GroupSharedRankingCorruptionDataset",
]
