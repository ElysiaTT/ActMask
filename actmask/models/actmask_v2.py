"""CPU-friendly ActMask model with explicit feature-ablation boundaries."""

from __future__ import annotations

from typing import Final

import torch
from torch import Tensor, nn


FEATURE_MODES: Final[tuple[str, ...]] = (
    "full",
    "no_velocity",
    "no_action",
    "position_only",
)


def _canonical_feature_mode(feature_mode: object) -> str:
    value = getattr(feature_mode, "value", feature_mode)
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized not in FEATURE_MODES:
        raise ValueError(
            f"feature_mode must be one of {FEATURE_MODES}, got {feature_mode!r}"
        )
    return normalized


def _positive_dimension(name: str, value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


class ActMaskV2(nn.Module):
    """Predict a future interaction mask with clean feature branches.

    Position, velocity, and action-relative information are encoded by three
    disjoint MLPs.  Ablations replace a complete branch *embedding* with exact
    zeros; they never feed zero-valued inputs through a biased encoder.  In
    particular, every feature involving the action (including point offsets to
    the action endpoints) lives exclusively in ``action_relative_encoder``.

    The constructor accepts the four Milestone-1 model configuration keys, so
    existing width dictionaries can instantiate this model directly.
    """

    def __init__(
        self,
        action_dim: int = 8,
        point_hidden_dim: int = 64,
        action_hidden_dim: int = 32,
        fusion_hidden_dim: int = 64,
        *,
        use_global_context: bool = True,
        global_hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        if action_dim != 8:
            raise ValueError("ActMaskV2 uses the fixed 8-value action schema")
        self.action_dim = action_dim
        self.point_hidden_dim = _positive_dimension(
            "point_hidden_dim", point_hidden_dim
        )
        self.action_hidden_dim = _positive_dimension(
            "action_hidden_dim", action_hidden_dim
        )
        self.fusion_hidden_dim = _positive_dimension(
            "fusion_hidden_dim", fusion_hidden_dim
        )
        if not isinstance(use_global_context, bool):
            raise TypeError("use_global_context must be a bool")
        self.use_global_context = use_global_context
        self.global_hidden_dim = _positive_dimension(
            "global_hidden_dim",
            fusion_hidden_dim if global_hidden_dim is None else global_hidden_dim,
        )

        self.position_encoder = self._branch_mlp(3, self.point_hidden_dim)
        self.velocity_encoder = self._branch_mlp(3, self.point_hidden_dim)
        # Broadcast action[8], point-to-start[3], and point-to-end[3].  All
        # action-dependent geometry is deliberately confined to this branch.
        self.action_relative_feature_dim = action_dim + 6
        self.action_relative_encoder = self._branch_mlp(
            self.action_relative_feature_dim, self.action_hidden_dim
        )

        local_dim = 2 * self.point_hidden_dim + self.action_hidden_dim
        if self.use_global_context:
            self.global_encoder: nn.Module | None = nn.Sequential(
                nn.Linear(2 * local_dim, self.global_hidden_dim),
                nn.ReLU(),
            )
            head_input_dim = local_dim + self.global_hidden_dim
        else:
            self.global_encoder = None
            head_input_dim = local_dim
        self.mask_head = nn.Sequential(
            nn.Linear(head_input_dim, self.fusion_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.fusion_hidden_dim, 1),
        )

    @staticmethod
    def _branch_mlp(input_dim: int, output_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
        )

    def encode_branches(
        self,
        points: Tensor,
        velocities: Tensor,
        action: Tensor,
        feature_mode: str = "full",
    ) -> dict[str, Tensor]:
        """Return branch embeddings, exposing the exact ablation boundary."""

        self._validate_inputs(points, velocities, action)
        mode = _canonical_feature_mode(feature_mode)
        batch_size, num_points, _ = points.shape

        position_embedding = self.position_encoder(points)

        if mode in {"full", "no_action"}:
            velocity_embedding = self.velocity_encoder(velocities)
        else:
            velocity_embedding = points.new_zeros(
                (batch_size, num_points, self.point_hidden_dim)
            )

        if mode in {"full", "no_velocity"}:
            expanded_action = action[:, None, :].expand(
                batch_size, num_points, self.action_dim
            )
            start = action[:, None, :3]
            end = action[:, None, 3:6]
            action_relative_features = torch.cat(
                (expanded_action, points - start, points - end), dim=-1
            )
            action_embedding = self.action_relative_encoder(action_relative_features)
        else:
            action_embedding = points.new_zeros(
                (batch_size, num_points, self.action_hidden_dim)
            )

        return {
            "position": position_embedding,
            "velocity": velocity_embedding,
            "action_relative": action_embedding,
        }

    def forward(
        self,
        points: Tensor,
        velocities: Tensor,
        action: Tensor,
        feature_mode: str = "full",
    ) -> Tensor:
        """Return raw mask logits of shape ``[B, N]``."""

        branches = self.encode_branches(points, velocities, action, feature_mode)
        local = torch.cat(
            (
                branches["position"],
                branches["velocity"],
                branches["action_relative"],
            ),
            dim=-1,
        )
        if self.global_encoder is not None:
            global_statistics = torch.cat(
                (local.mean(dim=1), local.amax(dim=1)), dim=-1
            )
            global_embedding = self.global_encoder(global_statistics)
            expanded_global = global_embedding[:, None, :].expand(
                local.shape[0], local.shape[1], self.global_hidden_dim
            )
            local = torch.cat((local, expanded_global), dim=-1)
        return self.mask_head(local).squeeze(-1)

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


__all__ = ["ActMaskV2", "FEATURE_MODES"]
