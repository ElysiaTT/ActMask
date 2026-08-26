"""Frozen shortcut-control and fair-verifier architectures."""
from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class TemporalEncoder(nn.Module):
    def __init__(self, input_dimension: int, hidden: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(input_dimension, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, 3, padding=1),
            nn.GELU(),
        )
        self.norm = nn.LayerNorm(hidden)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        encoded = self.network(value.transpose(1, 2)).mean(-1)
        return self.norm(encoded)


class StateActionTCN(nn.Module):
    required_inputs = ("history_state", "candidate_action")

    def __init__(self):
        super().__init__()
        self.state = TemporalEncoder(32, 64)
        self.action = TemporalEncoder(32, 64)
        self.head = nn.Sequential(
            nn.Linear(256, 96),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(96, 1),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        state = self.state(batch["history_state"])
        action = self.action(batch["candidate_action"])
        representation = torch.cat(
            (state, action, state * action, torch.abs(state - action)), dim=-1
        )
        return self.head(representation).squeeze(-1), representation


class JointTransformer(nn.Module):
    def __init__(self, include_visual: bool):
        super().__init__()
        self.include_visual = include_visual
        self.required_inputs = (
            ("history_state", "candidate_action", "language_embedding", "visual_history")
            if include_visual
            else ("history_state", "candidate_action", "language_embedding")
        )
        dimension = 64
        self.state_projection = nn.Linear(32, dimension)
        self.action_projection = nn.Linear(32, dimension)
        self.language_projection = nn.Linear(128, dimension)
        self.visual_projection = nn.Linear(22, dimension) if include_visual else None
        self.cls = nn.Parameter(torch.zeros(1, 1, dimension))
        self.type_embedding = nn.Embedding(5, dimension)
        self.position = nn.Parameter(torch.zeros(1, 40, dimension))
        layer = nn.TransformerEncoderLayer(
            d_model=dimension,
            nhead=4,
            dim_feedforward=128,
            dropout=0.10,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.norm = nn.LayerNorm(dimension)
        self.head = nn.Linear(dimension, 1)
        nn.init.normal_(self.cls, std=0.02)
        nn.init.normal_(self.position, std=0.02)

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        count = len(batch["history_state"])
        tokens = [
            self.cls.expand(count, -1, -1) + self.type_embedding.weight[0],
            self.state_projection(batch["history_state"])
            + self.type_embedding.weight[1],
        ]
        if self.include_visual:
            assert self.visual_projection is not None
            tokens.append(
                self.visual_projection(batch["visual_history"])
                + self.type_embedding.weight[2]
            )
        tokens.extend(
            (
                self.language_projection(batch["language_embedding"])[:, None]
                + self.type_embedding.weight[3],
                self.action_projection(batch["candidate_action"])
                + self.type_embedding.weight[4],
            )
        )
        value = torch.cat(tokens, dim=1)
        value = value + self.position[:, : value.shape[1]]
        representation = self.norm(self.encoder(value)[:, 0])
        return self.head(representation).squeeze(-1), representation


class TaskConditionedActionTransformer(JointTransformer):
    def __init__(self):
        super().__init__(include_visual=False)


class VisualStateActionTransformer(JointTransformer):
    def __init__(self):
        super().__init__(include_visual=True)


class ContrastiveContextActionVerifier(nn.Module):
    required_inputs = (
        "history_state",
        "candidate_action",
        "language_embedding",
        "visual_history",
    )

    def __init__(self):
        super().__init__()
        self.state = TemporalEncoder(32, 64)
        self.visual = TemporalEncoder(22, 64)
        self.language = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.LayerNorm(64))
        self.context = nn.Sequential(
            nn.Linear(192, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
        )
        self.action = TemporalEncoder(32, 64)
        self.log_scale = nn.Parameter(torch.tensor(2.0))
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        context = self.context(
            torch.cat(
                (
                    self.state(batch["history_state"]),
                    self.visual(batch["visual_history"]),
                    self.language(batch["language_embedding"]),
                ),
                dim=-1,
            )
        )
        action = self.action(batch["candidate_action"])
        context_normalized = F.normalize(context, dim=-1)
        action_normalized = F.normalize(action, dim=-1)
        compatibility = context_normalized * action_normalized
        logit = torch.exp(self.log_scale.clamp(-2.0, 4.0)) * compatibility.sum(-1)
        logit = logit + self.bias
        representation = torch.cat(
            (context, action, compatibility, torch.abs(context - action)), dim=-1
        )
        return logit, representation


class CrossAttentionEnergyVerifier(nn.Module):
    required_inputs = (
        "history_state",
        "candidate_action",
        "language_embedding",
        "visual_history",
    )

    def __init__(self):
        super().__init__()
        dimension = 64
        self.state_projection = nn.Linear(32, dimension)
        self.visual_projection = nn.Linear(22, dimension)
        self.language_projection = nn.Linear(128, dimension)
        self.action_projection = nn.Linear(32, dimension)
        self.context_type = nn.Embedding(3, dimension)
        self.action_position = nn.Parameter(torch.zeros(1, 16, dimension))
        self.cross_attention = nn.MultiheadAttention(
            dimension, 4, dropout=0.10, batch_first=True
        )
        self.action_self_attention = nn.MultiheadAttention(
            dimension, 4, dropout=0.10, batch_first=True
        )
        self.norm_query = nn.LayerNorm(dimension)
        self.norm_cross = nn.LayerNorm(dimension)
        self.norm_self = nn.LayerNorm(dimension)
        self.head = nn.Sequential(
            nn.Linear(256, 96),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(96, 1),
        )
        nn.init.normal_(self.action_position, std=0.02)

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        context = torch.cat(
            (
                self.state_projection(batch["history_state"])
                + self.context_type.weight[0],
                self.visual_projection(batch["visual_history"])
                + self.context_type.weight[1],
                self.language_projection(batch["language_embedding"])[:, None]
                + self.context_type.weight[2],
            ),
            dim=1,
        )
        action = self.action_projection(batch["candidate_action"])
        action = action + self.action_position
        self_value, _ = self.action_self_attention(action, action, action, need_weights=False)
        action = self.norm_self(action + self_value)
        cross, _ = self.cross_attention(action, context, context, need_weights=False)
        cross = self.norm_cross(action + cross)
        action_pool = self.norm_query(action).mean(1)
        cross_pool = cross.mean(1)
        representation = torch.cat(
            (
                action_pool,
                cross_pool,
                action_pool * cross_pool,
                torch.abs(action_pool - cross_pool),
            ),
            dim=-1,
        )
        return self.head(representation).squeeze(-1), representation


VERIFIER_CLASSES: dict[str, type[nn.Module]] = {
    "StateActionTCN": StateActionTCN,
    "TaskConditionedActionTransformer": TaskConditionedActionTransformer,
    "VisualStateActionTransformer": VisualStateActionTransformer,
    "ContrastiveContextActionVerifier": ContrastiveContextActionVerifier,
    "CrossAttentionEnergyVerifier": CrossAttentionEnergyVerifier,
}


class VectorLinearControl(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        self.linear = nn.Linear(dimension, 1)

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        value = batch["vector"]
        return self.linear(value).squeeze(-1), value


class VectorMLPControl(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(dimension, 128),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(128, 64),
            nn.GELU(),
        )
        self.head = nn.Linear(64, 1)

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        representation = self.encoder(batch["vector"])
        return self.head(representation).squeeze(-1), representation


class TemporalActionOnlyControl(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = TemporalEncoder(32, 64)
        self.head = nn.Linear(64, 1)

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        representation = self.encoder(batch["candidate_action"])
        return self.head(representation).squeeze(-1), representation


class GroupAdditiveControl(nn.Module):
    def __init__(self, dimensions: dict[str, int]):
        super().__init__()
        self.groups = tuple(dimensions)
        self.encoders = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(dimension, 32),
                    nn.GELU(),
                    nn.Linear(32, 1),
                )
                for name, dimension in dimensions.items()
            }
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        group_logits = torch.cat(
            [self.encoders[name](batch[name]) for name in self.groups], dim=-1
        )
        return group_logits.sum(-1), group_logits


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def model_description(model: nn.Module) -> dict[str, Any]:
    return {
        "class": model.__class__.__name__,
        "learned_parameters": parameter_count(model),
        "trainable_parameters": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
    }
