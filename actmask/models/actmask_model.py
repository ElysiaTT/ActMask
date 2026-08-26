"""A lightweight action-conditioned point-wise mask predictor."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ActMaskModel(nn.Module):
    """Fuse point motion features with a compact candidate action.

    The module is intentionally device-agnostic and contains only small MLPs.
    Inputs follow the batched public contract: points and velocities are
    ``[B, N, 3]``, actions are ``[B, 8]``, and returned raw logits are
    ``[B, N]``.
    """

    def __init__(
        self,
        action_dim: int = 8,
        point_hidden_dim: int = 64,
        action_hidden_dim: int = 32,
        fusion_hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        if action_dim != 8:
            raise ValueError("ActMaskModel uses the fixed 8-value action schema")
        for name, value in (
            ("point_hidden_dim", point_hidden_dim),
            ("action_hidden_dim", action_hidden_dim),
            ("fusion_hidden_dim", fusion_hidden_dim),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

        self.action_dim = action_dim
        self.point_hidden_dim = point_hidden_dim
        self.action_hidden_dim = action_hidden_dim
        self.fusion_hidden_dim = fusion_hidden_dim

        # xyz, velocity, relative-to-start xyz, relative-to-end xyz
        point_feature_dim = 12
        self.point_encoder = nn.Sequential(
            nn.Linear(point_feature_dim, point_hidden_dim),
            nn.ReLU(),
            nn.Linear(point_hidden_dim, point_hidden_dim),
            nn.ReLU(),
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, action_hidden_dim),
            nn.ReLU(),
            nn.Linear(action_hidden_dim, action_hidden_dim),
            nn.ReLU(),
        )
        self.mask_head = nn.Sequential(
            nn.Linear(point_hidden_dim + action_hidden_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.Linear(fusion_hidden_dim, 1),
        )

    def forward(self, points: Tensor, velocities: Tensor, action: Tensor) -> Tensor:
        """Return one unnormalized mask logit per input point."""

        self._validate_inputs(points, velocities, action)
        batch_size, num_points, _ = points.shape

        start = action[:, None, :3]
        end = action[:, None, 3:6]
        point_features = torch.cat(
            (points, velocities, points - start, points - end), dim=-1
        )
        encoded_points = self.point_encoder(point_features)
        encoded_action = self.action_encoder(action)
        expanded_action = encoded_action[:, None, :].expand(
            batch_size, num_points, self.action_hidden_dim
        )
        fused = torch.cat((encoded_points, expanded_action), dim=-1)
        return self.mask_head(fused).squeeze(-1)

    def _validate_inputs(self, points: Tensor, velocities: Tensor, action: Tensor) -> None:
        for name, tensor in (
            ("points", points),
            ("velocities", velocities),
            ("action", action),
        ):
            if not isinstance(tensor, Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
            if not tensor.is_floating_point():
                raise TypeError(f"{name} must have a floating-point dtype")

        if points.ndim != 3 or points.shape[-1] != 3:
            raise ValueError(f"points must have shape [B, N, 3], got {tuple(points.shape)}")
        if velocities.shape != points.shape:
            raise ValueError(
                "velocities must have the same [B, N, 3] shape as points, "
                f"got {tuple(velocities.shape)} and {tuple(points.shape)}"
            )
        if action.ndim != 2 or action.shape[-1] != self.action_dim:
            raise ValueError(
                f"action must have shape [B, {self.action_dim}], got {tuple(action.shape)}"
            )
        if action.shape[0] != points.shape[0]:
            raise ValueError("points, velocities, and action must share the batch dimension")
        if points.shape[0] == 0 or points.shape[1] == 0:
            raise ValueError("batch size and number of points must both be non-zero")
        if points.device != velocities.device or points.device != action.device:
            raise ValueError("points, velocities, and action must be on the same device")
        if points.dtype != velocities.dtype or points.dtype != action.dtype:
            raise ValueError("points, velocities, and action must have the same dtype")


__all__ = ["ActMaskModel"]
