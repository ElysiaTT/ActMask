"""Correspondence-robust temporal ActMask variants for Milestone 2C.

The 2B temporal model consumes an ordered ``[K, N, 3]`` history.  This module
keeps the last observation as the prediction anchor but re-associates every
earlier frame using only visible spatial geometry.  Thus neither model uses a
point index as a physical identity across frames.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from actmask.data.milestone2b_dataset import validate_observable_mapping
from actmask.models.temporal_actmask import RotationInvariantTemporalActMask, TemporalActMask


def _euclidean_cost(predicted: Tensor, source: Tensor) -> Tensor:
    """Return full-3D Euclidean association cost ``[B,N,N]``.

    This cost is invariant to coordinate-axis permutations and to a shared
    rigid rotation of both point sets.  It is the only geometry cost used by
    the primary correspondence method.
    """

    return torch.linalg.vector_norm(
        predicted[:, :, None, :] - source[:, None, :, :], dim=-1
    )


def _legacy_axis_biased_cost(anchors: Tensor, source: Tensor) -> Tensor:
    """Old diagnostic-only rule.  It is intentionally axis specific."""

    delta = anchors[:, :, None, :] - source[:, None, :, :]
    return delta[..., 0].abs() + 0.28 * torch.linalg.vector_norm(delta[..., 1:], dim=-1)


def _least_squares_velocity(
    history: Tensor, visibility: Tensor, timestamps: Tensor
) -> tuple[Tensor, Tensor]:
    """Estimate motion from re-associated visible observations only."""

    weight = visibility.to(history.dtype)
    time = timestamps[:, :, None]
    count = weight.sum(dim=1).clamp_min(1.0)
    mean_time = (weight * time).sum(dim=1) / count
    mean_position = (history * weight[:, :, :, None]).sum(dim=1) / count[:, :, None]
    centered_time = time - mean_time[:, None, :]
    numerator = (
        centered_time[:, :, :, None]
        * (history - mean_position[:, None, :, :])
        * weight[:, :, :, None]
    ).sum(dim=1)
    denominator = (centered_time.square() * weight).sum(dim=1).clamp_min(1.0e-6)
    velocity = numerator / denominator[:, :, None]
    full_span = (timestamps[:, -1] - timestamps[:, 0]).clamp_min(1.0e-6)
    observed_span = torch.where(
        visibility,
        time.expand_as(visibility.to(history.dtype)),
        torch.full_like(time.expand_as(visibility.to(history.dtype)), float("nan")),
    )
    minimum = torch.nan_to_num(observed_span, nan=float("inf")).amin(dim=1)
    maximum = torch.nan_to_num(observed_span, nan=-float("inf")).amax(dim=1)
    span = torch.nan_to_num(maximum - minimum, nan=0.0).clamp_min(0.0)
    confidence = (count / float(history.shape[1])) * (span / full_span[:, None])
    confidence = confidence.clamp(0.0, 1.0)
    velocity = torch.where((count >= 2.0)[:, :, None], velocity, torch.zeros_like(velocity))
    return velocity, confidence


class _SpatialAssociationTemporalActMask(nn.Module):
    """Temporal model front-end that canonicalizes frame sets around frame K."""

    association: str

    def __init__(
        self,
        *,
        association: str,
        soft_temperature: float = 0.035,
        **temporal_kwargs: Any,
    ) -> None:
        super().__init__()
        if association not in {"nearest", "soft"}:
            raise ValueError("association must be nearest or soft")
        if soft_temperature <= 0.0:
            raise ValueError("soft_temperature must be positive")
        self.association = association
        self.soft_temperature = float(soft_temperature)
        self.core = TemporalActMask(**temporal_kwargs)

    @property
    def utility_head_enabled(self) -> bool:
        return self.core.utility_head_enabled

    def canonicalize(self, observable: Mapping[str, Any]) -> dict[str, Tensor]:
        """Reassociate each source frame to last-frame anchors without IDs."""

        validate_observable_mapping(observable)
        history = observable["points_history"]
        visibility = observable["visibility_history"].bool()
        timestamps = observable["timestamps"]
        if not isinstance(history, Tensor) or not isinstance(visibility, Tensor):
            raise TypeError("history and visibility must be tensors")
        if history.ndim != 4 or history.shape[-1] != 3:
            raise ValueError("points_history must be [B,K,N,3]")
        batch, frames, points, _ = history.shape
        anchors = history[:, -1]
        associated_history: list[Tensor] = []
        associated_visibility: list[Tensor] = []
        for frame in range(frames):
            source = history[:, frame]
            source_visible = visibility[:, frame]
            if frame == frames - 1:
                associated_history.append(source)
                associated_visibility.append(source_visible)
                continue
            cost = _euclidean_cost(anchors, source)
            invalid = ~source_visible[:, None, :]
            cost = cost.masked_fill(invalid, float("inf"))
            if self.association == "nearest":
                index = cost.argmin(dim=-1)
                gathered = source.gather(
                    1, index[:, :, None].expand(batch, points, 3)
                )
                valid = torch.isfinite(cost.amin(dim=-1))
            else:
                finite_cost = torch.where(
                    torch.isfinite(cost), cost, torch.full_like(cost, 50.0)
                )
                weights = torch.softmax(-finite_cost / self.soft_temperature, dim=-1)
                weights = weights * source_visible[:, None, :].to(weights.dtype)
                normalizer = weights.sum(dim=-1, keepdim=True)
                valid = normalizer[:, :, 0] > 1.0e-6
                weights = weights / normalizer.clamp_min(1.0e-6)
                gathered = torch.matmul(weights, source)
            associated_history.append(
                torch.where(valid[:, :, None], gathered, torch.zeros_like(gathered))
            )
            associated_visibility.append(valid)
        aligned_history = torch.stack(associated_history, dim=1)
        aligned_visibility = torch.stack(associated_visibility, dim=1)
        velocity, confidence = _least_squares_velocity(
            aligned_history, aligned_visibility, timestamps
        )
        result = {key: value for key, value in observable.items()}
        result.update(
            {
                "points_history": aligned_history,
                "visibility_history": aligned_visibility,
                "estimated_velocity": velocity,
                "velocity_confidence": confidence,
            }
        )
        return result

    def forward(
        self, observable: Mapping[str, Any], *, feature_mode: str = "full"
    ) -> dict[str, Tensor]:
        return self.core(self.canonicalize(observable), feature_mode=feature_mode)


class IndexCorrespondenceTemporalActMask(TemporalActMask):
    """Named 2B-compatible baseline that assumes frame indices correspond."""


class SetHistoryTemporalActMask(_SpatialAssociationTemporalActMask):
    """Soft set-to-set frame association, invariant to per-frame permutation."""

    def __init__(self, **temporal_kwargs: Any) -> None:
        super().__init__(association="soft", **temporal_kwargs)


class LocalNeighborhoodTemporalActMask(_SpatialAssociationTemporalActMask):
    """Nearest local spatial association without fixed point identity."""

    def __init__(self, **temporal_kwargs: Any) -> None:
        super().__init__(association="nearest", **temporal_kwargs)



@dataclass(frozen=True)
class AssociationResult:
    """Observable-only matches; ``-1`` is an explicit unmatched state."""

    aligned_history: Tensor
    aligned_visibility: Tensor
    source_index: Tensor
    mutual: Tensor
    normalized_distance: Tensor
    velocity: Tensor
    confidence: Tensor


def _cloud_scale(points: Tensor, visible: Tensor) -> Tensor:
    """Robust 3D local spacing for dimensionless rejection."""

    cost = _euclidean_cost(points, points)
    valid = visible[:, :, None] & visible[:, None, :]
    diagonal = torch.eye(points.shape[1], dtype=torch.bool, device=points.device)[None]
    cost = cost.masked_fill(~valid | diagonal, float("inf"))
    nearest = cost.amin(dim=-1)
    result: list[Tensor] = []
    for row, present in zip(nearest, visible, strict=True):
        values = row[present & torch.isfinite(row) & (row > 1.0e-6)]
        result.append(values.median() if values.numel() else row.new_tensor(0.05))
    return torch.stack(result).clamp_min(0.01)


def _mutual_matches(
    cost: Tensor,
    anchors_visible: Tensor,
    source_visible: Tensor,
    scale: Tensor,
    maximum_normalized_distance: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Mutual-NN one-to-one matching, with rejected rows encoded as ``-1``."""

    forward = cost.masked_fill(~source_visible[:, None, :], float("inf"))
    distance, index = forward.min(dim=-1)
    reverse = cost.masked_fill(~anchors_visible[:, :, None], float("inf"))
    reverse_index = reverse.min(dim=1).indices
    expected = torch.arange(cost.shape[1], device=cost.device)[None]
    back = reverse_index.gather(1, index.clamp_min(0))
    normalized = distance / scale[:, None].clamp_min(1.0e-6)
    mutual = torch.isfinite(distance) & anchors_visible & (back == expected)
    accepted = mutual & (normalized <= float(maximum_normalized_distance))
    index = torch.where(accepted, index, torch.full_like(index, -1))
    return index, accepted, mutual, normalized


class MotionConsistentMutualTemporalActMask(_SpatialAssociationTemporalActMask):
    """Primary full-3D motion-predicted, mutual-NN correspondence front end.

    The nearest preceding frame provides a Euclidean mutual-NN motion bootstrap.
    Older frames are matched against the predicted 3D point location.  Low
    confidence, non-mutual, and distant correspondences remain unmatched; the
    model never substitutes a forced source point.
    """

    def __init__(
        self,
        *,
        maximum_normalized_distance: float = 3.0,
        minimum_motion_confidence: float = 0.10,
        **temporal_kwargs: Any,
    ) -> None:
        if maximum_normalized_distance <= 0.0:
            raise ValueError("maximum_normalized_distance must be positive")
        if not 0.0 <= minimum_motion_confidence <= 1.0:
            raise ValueError("minimum_motion_confidence must be in [0, 1]")
        super().__init__(association="nearest", **temporal_kwargs)
        self.maximum_normalized_distance = float(maximum_normalized_distance)
        self.minimum_motion_confidence = float(minimum_motion_confidence)

    def associate(self, observable: Mapping[str, Any]) -> AssociationResult:
        validate_observable_mapping(observable)
        history = observable["points_history"]
        visibility = observable["visibility_history"].bool()
        timestamps = observable["timestamps"]
        if not isinstance(history, Tensor) or history.ndim != 4 or history.shape[-1] != 3:
            raise ValueError("points_history must be [B,K,N,3]")
        batch, frames, points, _ = history.shape
        anchors = history[:, -1]
        anchors_visible = visibility[:, -1]
        scale = _cloud_scale(anchors, anchors_visible)
        aligned = torch.zeros_like(history)
        aligned_visible = torch.zeros_like(visibility)
        source_index = torch.full((batch, frames, points), -1, dtype=torch.long, device=history.device)
        mutual = torch.zeros((batch, frames, points), dtype=torch.bool, device=history.device)
        normalized = torch.full((batch, frames, points), float("inf"), dtype=history.dtype, device=history.device)
        aligned[:, -1] = anchors
        aligned_visible[:, -1] = anchors_visible
        identity = torch.arange(points, device=history.device)[None].expand(batch, points)
        source_index[:, -1] = torch.where(anchors_visible, identity, torch.full_like(identity, -1))
        mutual[:, -1] = anchors_visible
        normalized[:, -1] = torch.where(anchors_visible, torch.zeros_like(anchors[..., 0]), torch.full_like(anchors[..., 0], float("inf")))

        velocity = torch.zeros_like(anchors)
        motion_confidence = torch.zeros((batch, points), dtype=history.dtype, device=history.device)
        if frames >= 2:
            source = history[:, -2]
            source_visible = visibility[:, -2]
            index, accepted, is_mutual, distance = _mutual_matches(
                _euclidean_cost(anchors, source), anchors_visible, source_visible,
                scale, self.maximum_normalized_distance,
            )
            gathered = source.gather(1, index.clamp_min(0)[:, :, None].expand(batch, points, 3))
            dt = (timestamps[:, -1] - timestamps[:, -2]).clamp_min(1.0e-6)
            velocity = torch.where(accepted[:, :, None], (anchors - gathered) / dt[:, None, None], velocity)
            motion_confidence = accepted.to(history.dtype) * torch.exp(-distance.clamp_max(50.0))
            aligned[:, -2] = torch.where(accepted[:, :, None], gathered, torch.zeros_like(gathered))
            aligned_visible[:, -2] = accepted
            source_index[:, -2] = index
            mutual[:, -2] = is_mutual
            normalized[:, -2] = distance

        usable = anchors_visible & (motion_confidence >= self.minimum_motion_confidence)
        for frame in range(frames - 3, -1, -1):
            source = history[:, frame]
            source_visible = visibility[:, frame]
            dt = (timestamps[:, -1] - timestamps[:, frame]).clamp_min(0.0)
            predicted = anchors - velocity * dt[:, None, None]
            index, accepted, is_mutual, distance = _mutual_matches(
                _euclidean_cost(predicted, source), usable, source_visible,
                scale, self.maximum_normalized_distance,
            )
            gathered = source.gather(1, index.clamp_min(0)[:, :, None].expand(batch, points, 3))
            aligned[:, frame] = torch.where(accepted[:, :, None], gathered, torch.zeros_like(gathered))
            aligned_visible[:, frame] = accepted
            source_index[:, frame] = index
            mutual[:, frame] = is_mutual
            normalized[:, frame] = distance
        refined_velocity, confidence = _least_squares_velocity(aligned, aligned_visible, timestamps)
        return AssociationResult(aligned, aligned_visible, source_index, mutual, normalized, refined_velocity, confidence)

    def canonicalize(self, observable: Mapping[str, Any]) -> dict[str, Tensor]:
        result = self.associate(observable)
        canonical = {key: value for key, value in observable.items()}
        canonical.update({
            "points_history": result.aligned_history,
            "visibility_history": result.aligned_visibility,
            "estimated_velocity": result.velocity,
            "velocity_confidence": result.confidence,
        })
        return canonical


class LegacyAxisBiasedNeighborhoodTemporalActMask(_SpatialAssociationTemporalActMask):
    """Diagnostic-only negative control retaining the old axis-biased rule."""

    def __init__(self, **temporal_kwargs: Any) -> None:
        super().__init__(association="nearest", **temporal_kwargs)

    def canonicalize(self, observable: Mapping[str, Any]) -> dict[str, Tensor]:
        validate_observable_mapping(observable)
        history = observable["points_history"]
        visibility = observable["visibility_history"].bool()
        timestamps = observable["timestamps"]
        batch, frames, points, _ = history.shape
        anchors = history[:, -1]
        aligned: list[Tensor] = []
        aligned_visible: list[Tensor] = []
        for frame in range(frames):
            source = history[:, frame]
            source_visible = visibility[:, frame]
            cost = _legacy_axis_biased_cost(anchors, source).masked_fill(~source_visible[:, None, :], float("inf"))
            index = cost.argmin(dim=-1)
            accepted = visibility[:, -1] & torch.isfinite(cost.amin(dim=-1))
            gathered = source.gather(1, index[:, :, None].expand(batch, points, 3))
            aligned.append(torch.where(accepted[:, :, None], gathered, torch.zeros_like(gathered)))
            aligned_visible.append(accepted)
        aligned_history = torch.stack(aligned, dim=1)
        aligned_visibility = torch.stack(aligned_visible, dim=1)
        velocity, confidence = _least_squares_velocity(aligned_history, aligned_visibility, timestamps)
        result = {key: value for key, value in observable.items()}
        result.update({
            "points_history": aligned_history,
            "visibility_history": aligned_visibility,
            "estimated_velocity": velocity,
            "velocity_confidence": confidence,
        })
        return result


class LocalNeighborhoodTemporalActMask(MotionConsistentMutualTemporalActMask):
    """Compatibility name; now uses the coordinate-invariant primary method."""


class InvariantFeatureMotionTemporalActMask(MotionConsistentMutualTemporalActMask):
    """Motion-invariant correspondence with a rotation-invariant scalar backbone."""

    def __init__(self, **temporal_kwargs: Any) -> None:
        super().__init__(**temporal_kwargs)
        self.core = RotationInvariantTemporalActMask(**temporal_kwargs)


__all__ = [
    "AssociationResult",
    "IndexCorrespondenceTemporalActMask",
    "InvariantFeatureMotionTemporalActMask",
    "LegacyAxisBiasedNeighborhoodTemporalActMask",
    "LocalNeighborhoodTemporalActMask",
    "MotionConsistentMutualTemporalActMask",
    "SetHistoryTemporalActMask",
]
