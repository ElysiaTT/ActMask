"""Composable correspondence/backbone variants for the Milestone 2E study."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

import torch
from torch import Tensor, nn

from actmask.data.milestone2b_dataset import validate_observable_mapping
from actmask.models.correspondence import (
    AssociationResult,
    _cloud_scale,
    _euclidean_cost,
    _least_squares_velocity,
    _mutual_matches,
)
from actmask.models.temporal_actmask import (
    ActionFrameRelativeTemporalActMask,
    RotationInvariantTemporalActMask,
    TemporalActMask,
)


CORRESPONDENCE_VARIANTS: Final[tuple[str, ...]] = (
    "GroundTruthIdentityCorrespondence",
    "CoordinateInvariantMutualNN",
    "SoftCoordinateInvariantCorrespondence",
    "NoExplicitCorrespondence",
)
BACKBONE_VARIANTS: Final[tuple[str, ...]] = (
    "OriginalVectorBackbone",
    "ScalarInvariantFeatureBackbone",
    "ActionFrameRelativeBackbone",
)


def _last_frame_result(history: Tensor, visibility: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    batch, frames, points, _ = history.shape
    anchors = history[:, -1]
    anchors_visible = visibility[:, -1].bool()
    aligned = torch.zeros_like(history)
    aligned_visible = torch.zeros_like(visibility, dtype=torch.bool)
    source_index = torch.full((batch, frames, points), -1, dtype=torch.long, device=history.device)
    mutual = torch.zeros((batch, frames, points), dtype=torch.bool, device=history.device)
    normalized = torch.full((batch, frames, points), float("inf"), dtype=history.dtype, device=history.device)
    identity = torch.arange(points, device=history.device)[None].expand(batch, points)
    aligned[:, -1] = anchors
    aligned_visible[:, -1] = anchors_visible
    source_index[:, -1] = torch.where(anchors_visible, identity, torch.full_like(identity, -1))
    mutual[:, -1] = anchors_visible
    normalized[:, -1] = torch.where(
        anchors_visible, torch.zeros_like(anchors[..., 0]), torch.full_like(anchors[..., 0], float("inf"))
    )
    return aligned, aligned_visible, source_index, mutual, normalized


class FactorialTemporalActMask(nn.Module):
    """Pair one 2E association method with one temporal representation.

    Identity correspondence is an explicitly diagnostic path: it requires an
    `identity_history` argument supplied only by the evaluation/training
    harness from hidden synthetic provenance.  The other three paths consume
    the public observable mapping alone.
    """

    def __init__(
        self,
        *,
        correspondence: str,
        backbone: str,
        maximum_normalized_distance: float = 3.0,
        soft_temperature: float = 0.035,
        soft_minimum_peak_weight: float = 0.15,
        **temporal_kwargs: Any,
    ) -> None:
        super().__init__()
        if correspondence not in CORRESPONDENCE_VARIANTS:
            raise ValueError(f"unknown correspondence {correspondence!r}")
        if backbone not in BACKBONE_VARIANTS:
            raise ValueError(f"unknown backbone {backbone!r}")
        if maximum_normalized_distance <= 0.0 or soft_temperature <= 0.0:
            raise ValueError("correspondence thresholds must be positive")
        if not 0.0 < soft_minimum_peak_weight <= 1.0:
            raise ValueError("soft_minimum_peak_weight must be in (0,1]")
        self.correspondence = correspondence
        self.backbone = backbone
        self.maximum_normalized_distance = float(maximum_normalized_distance)
        self.soft_temperature = float(soft_temperature)
        self.soft_minimum_peak_weight = float(soft_minimum_peak_weight)
        core_type: type[nn.Module]
        if backbone == "OriginalVectorBackbone":
            core_type = TemporalActMask
        elif backbone == "ScalarInvariantFeatureBackbone":
            core_type = RotationInvariantTemporalActMask
        else:
            core_type = ActionFrameRelativeTemporalActMask
        self.core = core_type(**temporal_kwargs)

    @property
    def utility_head_enabled(self) -> bool:
        return bool(self.core.utility_head_enabled)

    @staticmethod
    def _validate(observable: Mapping[str, Any]) -> tuple[Tensor, Tensor, Tensor]:
        validate_observable_mapping(observable)
        history = observable["points_history"]
        visibility = observable["visibility_history"].bool()
        timestamps = observable["timestamps"]
        if not isinstance(history, Tensor) or history.ndim != 4 or history.shape[-1] != 3:
            raise ValueError("points_history must be [B,K,N,3]")
        if not isinstance(visibility, Tensor) or visibility.shape != history.shape[:3]:
            raise ValueError("visibility_history must be [B,K,N]")
        return history, visibility, timestamps

    def _identity_associate(self, observable: Mapping[str, Any], identity_history: Tensor | None) -> AssociationResult:
        history, visibility, timestamps = self._validate(observable)
        if identity_history is None:
            raise ValueError("GroundTruthIdentityCorrespondence requires diagnostic identity_history")
        if identity_history.shape != history.shape[:3]:
            raise ValueError("identity_history must be [B,K,N]")
        batch, frames, points, _ = history.shape
        aligned, aligned_visible, source_index, mutual, normalized = _last_frame_result(history, visibility)
        anchor_id = identity_history[:, -1]
        anchors_visible = visibility[:, -1]
        for frame in range(frames - 1):
            source_id = identity_history[:, frame]
            equal = anchor_id[:, :, None] == source_id[:, None, :]
            present = equal.any(dim=-1)
            index = equal.long().argmax(dim=-1)
            accepted = anchors_visible & present & visibility[:, frame].gather(1, index)
            gathered = history[:, frame].gather(1, index[:, :, None].expand(batch, points, 3))
            aligned[:, frame] = torch.where(accepted[:, :, None], gathered, torch.zeros_like(gathered))
            aligned_visible[:, frame] = accepted
            source_index[:, frame] = torch.where(accepted, index, torch.full_like(index, -1))
            mutual[:, frame] = accepted
            normalized[:, frame] = torch.where(accepted, torch.zeros_like(normalized[:, frame]), normalized[:, frame])
        velocity, confidence = _least_squares_velocity(aligned, aligned_visible, timestamps)
        return AssociationResult(aligned, aligned_visible, source_index, mutual, normalized, velocity, confidence)

    def _mutual_associate(self, observable: Mapping[str, Any]) -> AssociationResult:
        history, visibility, timestamps = self._validate(observable)
        batch, frames, points, _ = history.shape
        anchors = history[:, -1]
        anchors_visible = visibility[:, -1]
        scale = _cloud_scale(anchors, anchors_visible)
        aligned, aligned_visible, source_index, mutual, normalized = _last_frame_result(history, visibility)
        velocity = torch.zeros_like(anchors)
        confidence_bootstrap = torch.zeros((batch, points), dtype=history.dtype, device=history.device)
        if frames >= 2:
            source = history[:, -2]
            index, accepted, is_mutual, distance = _mutual_matches(
                _euclidean_cost(anchors, source), anchors_visible, visibility[:, -2], scale,
                self.maximum_normalized_distance,
            )
            gathered = source.gather(1, index.clamp_min(0)[:, :, None].expand(batch, points, 3))
            dt = (timestamps[:, -1] - timestamps[:, -2]).clamp_min(1.0e-6)
            velocity = torch.where(accepted[:, :, None], (anchors - gathered) / dt[:, None, None], velocity)
            confidence_bootstrap = accepted.to(history.dtype) * torch.exp(-distance.clamp_max(50.0))
            aligned[:, -2] = torch.where(accepted[:, :, None], gathered, torch.zeros_like(gathered))
            aligned_visible[:, -2] = accepted
            source_index[:, -2] = index
            mutual[:, -2] = is_mutual
            normalized[:, -2] = distance
        usable = anchors_visible & (confidence_bootstrap >= 0.10)
        for frame in range(frames - 3, -1, -1):
            source = history[:, frame]
            dt = (timestamps[:, -1] - timestamps[:, frame]).clamp_min(0.0)
            predicted = anchors - velocity * dt[:, None, None]
            index, accepted, is_mutual, distance = _mutual_matches(
                _euclidean_cost(predicted, source), usable, visibility[:, frame], scale,
                self.maximum_normalized_distance,
            )
            gathered = source.gather(1, index.clamp_min(0)[:, :, None].expand(batch, points, 3))
            aligned[:, frame] = torch.where(accepted[:, :, None], gathered, torch.zeros_like(gathered))
            aligned_visible[:, frame] = accepted
            source_index[:, frame] = index
            mutual[:, frame] = is_mutual
            normalized[:, frame] = distance
        velocity, confidence = _least_squares_velocity(aligned, aligned_visible, timestamps)
        return AssociationResult(aligned, aligned_visible, source_index, mutual, normalized, velocity, confidence)

    def _soft_matches(
        self, cost: Tensor, anchors_visible: Tensor, source_visible: Tensor, scale: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        masked = cost.masked_fill(~source_visible[:, None, :], float("inf"))
        safe = torch.where(torch.isfinite(masked), masked, torch.full_like(masked, 50.0))
        weights = torch.softmax(-safe / self.soft_temperature, dim=-1)
        weights = weights * source_visible[:, None, :].to(weights.dtype)
        normalizer = weights.sum(dim=-1, keepdim=True)
        weights = weights / normalizer.clamp_min(1.0e-8)
        peak, index = weights.max(dim=-1)
        expected = (weights * safe).sum(dim=-1) / scale[:, None].clamp_min(1.0e-6)
        reverse = masked.masked_fill(~anchors_visible[:, :, None], float("inf")).min(dim=1).indices
        back = reverse.gather(1, index)
        expected_anchor = torch.arange(cost.shape[1], device=cost.device)[None]
        is_mutual = anchors_visible & (back == expected_anchor) & torch.isfinite(masked.amin(dim=-1))
        accepted = is_mutual & (peak >= self.soft_minimum_peak_weight) & (expected <= self.maximum_normalized_distance)
        index = torch.where(accepted, index, torch.full_like(index, -1))
        return weights, index, accepted, is_mutual, expected

    def _soft_associate(self, observable: Mapping[str, Any]) -> AssociationResult:
        history, visibility, timestamps = self._validate(observable)
        batch, frames, points, _ = history.shape
        anchors = history[:, -1]
        anchors_visible = visibility[:, -1]
        scale = _cloud_scale(anchors, anchors_visible)
        aligned, aligned_visible, source_index, mutual, normalized = _last_frame_result(history, visibility)
        velocity = torch.zeros_like(anchors)
        confidence_bootstrap = torch.zeros((batch, points), dtype=history.dtype, device=history.device)
        if frames >= 2:
            source = history[:, -2]
            weights, index, accepted, is_mutual, distance = self._soft_matches(
                _euclidean_cost(anchors, source), anchors_visible, visibility[:, -2], scale
            )
            gathered = torch.matmul(weights, source)
            dt = (timestamps[:, -1] - timestamps[:, -2]).clamp_min(1.0e-6)
            velocity = torch.where(accepted[:, :, None], (anchors - gathered) / dt[:, None, None], velocity)
            confidence_bootstrap = accepted.to(history.dtype) * weights.amax(dim=-1) * torch.exp(-distance.clamp_max(50.0))
            aligned[:, -2] = torch.where(accepted[:, :, None], gathered, torch.zeros_like(gathered))
            aligned_visible[:, -2] = accepted
            source_index[:, -2] = index
            mutual[:, -2] = is_mutual
            normalized[:, -2] = distance
        usable = anchors_visible & (confidence_bootstrap >= 0.10)
        for frame in range(frames - 3, -1, -1):
            source = history[:, frame]
            dt = (timestamps[:, -1] - timestamps[:, frame]).clamp_min(0.0)
            weights, index, accepted, is_mutual, distance = self._soft_matches(
                _euclidean_cost(anchors - velocity * dt[:, None, None], source), usable, visibility[:, frame], scale
            )
            gathered = torch.matmul(weights, source)
            aligned[:, frame] = torch.where(accepted[:, :, None], gathered, torch.zeros_like(gathered))
            aligned_visible[:, frame] = accepted
            source_index[:, frame] = index
            mutual[:, frame] = is_mutual
            normalized[:, frame] = distance
        velocity, confidence = _least_squares_velocity(aligned, aligned_visible, timestamps)
        return AssociationResult(aligned, aligned_visible, source_index, mutual, normalized, velocity, confidence)

    def _set_associate(self, observable: Mapping[str, Any]) -> AssociationResult:
        history, visibility, timestamps = self._validate(observable)
        batch, frames, points, _ = history.shape
        anchors_visible = visibility[:, -1]
        aligned, aligned_visible, source_index, mutual, normalized = _last_frame_result(history, visibility)
        for frame in range(frames - 1):
            source_visible = visibility[:, frame]
            count = source_visible.sum(dim=1, keepdim=True).to(history.dtype)
            centroid = (history[:, frame] * source_visible[:, :, None].to(history.dtype)).sum(dim=1) / count.clamp_min(1.0)
            accepted = anchors_visible & (count[:, 0, None] > 0.0)
            aligned[:, frame] = torch.where(
                accepted[:, :, None], centroid[:, None, :].expand(batch, points, 3), torch.zeros_like(aligned[:, frame])
            )
            aligned_visible[:, frame] = accepted
            # No source point is claimed as an identity correspondence.
            normalized[:, frame] = torch.where(accepted, torch.zeros_like(normalized[:, frame]), normalized[:, frame])
        velocity, confidence = _least_squares_velocity(aligned, aligned_visible, timestamps)
        return AssociationResult(aligned, aligned_visible, source_index, mutual, normalized, velocity, confidence)

    def associate(self, observable: Mapping[str, Any], *, identity_history: Tensor | None = None) -> AssociationResult:
        if self.correspondence == "GroundTruthIdentityCorrespondence":
            return self._identity_associate(observable, identity_history)
        if identity_history is not None:
            raise ValueError("identity_history is only allowed for the diagnostic correspondence")
        if self.correspondence == "CoordinateInvariantMutualNN":
            return self._mutual_associate(observable)
        if self.correspondence == "SoftCoordinateInvariantCorrespondence":
            return self._soft_associate(observable)
        return self._set_associate(observable)

    def canonicalize(self, observable: Mapping[str, Any], *, identity_history: Tensor | None = None) -> dict[str, Tensor]:
        result = self.associate(observable, identity_history=identity_history)
        canonical = {key: value for key, value in observable.items()}
        canonical.update(
            {
                "points_history": result.aligned_history,
                "visibility_history": result.aligned_visibility,
                "estimated_velocity": result.velocity,
                "velocity_confidence": result.confidence,
            }
        )
        return canonical

    def forward(
        self, observable: Mapping[str, Any], *, feature_mode: str = "full", identity_history: Tensor | None = None,
    ) -> dict[str, Tensor]:
        return self.core(self.canonicalize(observable, identity_history=identity_history), feature_mode=feature_mode)


__all__ = [
    "BACKBONE_VARIANTS",
    "CORRESPONDENCE_VARIANTS",
    "FactorialTemporalActMask",
]
