"""Counterfactual and ranking losses for Milestone 2B."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class Milestone2BLossCoefficients:
    """Auditable coefficients for every term in the 2B objective."""

    mask_bce: float = 1.0
    contact_time_bce: float = 0.20
    success_bce: float = 0.45
    pairwise_ranking: float = 0.30
    velocity_counterfactual: float = 0.25
    action_counterfactual: float = 0.25
    irrelevant_stability: float = 0.10

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if not isinstance(value, (int, float)) or not torch.isfinite(
                torch.tensor(float(value))
            ) or float(value) < 0.0:
                raise ValueError(f"loss coefficient {name} must be finite and non-negative")

    def to_dict(self) -> dict[str, float]:
        return {name: float(value) for name, value in self.__dict__.items()}


def counterfactual_assignment_loss(
    first_logits: Tensor,
    second_logits: Tensor,
    first_targets: Tensor,
    second_targets: Tensor,
    *,
    margin: float = 0.05,
) -> Tensor:
    """Prefer correct over swapped prediction/target assignments.

    Only pairs whose ground-truth masks differ participate.  Identical targets
    return an exact differentiable zero, so the objective never forces a
    prediction difference for an unchanged causal outcome.
    """

    if first_logits.shape != second_logits.shape or first_targets.shape != second_targets.shape:
        raise ValueError("counterfactual pair tensors must have matching shapes")
    if first_logits.shape != first_targets.shape:
        raise ValueError("counterfactual logits and targets must have matching shapes")
    if first_logits.ndim == 1:
        first_logits = first_logits[None, :]
        second_logits = second_logits[None, :]
        first_targets = first_targets[None, :]
        second_targets = second_targets[None, :]
    if first_logits.ndim != 2:
        raise ValueError("counterfactual tensors must have shape [P,N] or [N]")
    changed = torch.ne(first_targets, second_targets).any(dim=1)
    if not bool(changed.any()):
        return (first_logits.sum() + second_logits.sum()) * 0.0
    correct = F.binary_cross_entropy_with_logits(
        first_logits, first_targets, reduction="none"
    ).mean(dim=1) + F.binary_cross_entropy_with_logits(
        second_logits, second_targets, reduction="none"
    ).mean(dim=1)
    swapped = F.binary_cross_entropy_with_logits(
        first_logits, second_targets, reduction="none"
    ).mean(dim=1) + F.binary_cross_entropy_with_logits(
        second_logits, first_targets, reduction="none"
    ).mean(dim=1)
    margin_tensor = first_logits.new_tensor(float(margin))
    return F.relu(margin_tensor + correct[changed] - swapped[changed]).mean()


def irrelevant_background_stability_loss(
    reference_logits: Tensor,
    perturbed_logits: Tensor,
    reference_targets: Tensor,
    perturbed_targets: Tensor,
) -> Tensor:
    """Penalize prediction drift only when the ground truth is unchanged."""

    if reference_logits.shape != perturbed_logits.shape:
        raise ValueError("irrelevant pair logits must match")
    if reference_targets.shape != perturbed_targets.shape:
        raise ValueError("irrelevant pair targets must match")
    if not torch.equal(reference_targets, perturbed_targets):
        raise ValueError("irrelevant perturbation must preserve ground truth")
    return F.mse_loss(torch.sigmoid(reference_logits), torch.sigmoid(perturbed_logits))


def pairwise_candidate_ranking_loss(
    utility_logits: Tensor,
    success: Tensor,
    candidate_set_ids: Iterable[object],
    *,
    margin: float = 0.15,
) -> Tensor:
    """Margin loss over every successful/unsuccessful candidate pair per scene."""

    if utility_logits.ndim != 1 or success.shape != utility_logits.shape:
        raise ValueError("utility_logits and success must both have shape [B]")
    identifiers = [str(value) for value in candidate_set_ids]
    if len(identifiers) != len(utility_logits):
        raise ValueError("candidate_set_ids length must match the batch")
    losses: list[Tensor] = []
    for scene in sorted(set(identifiers)):
        indices = [index for index, value in enumerate(identifiers) if value == scene]
        positives = [index for index in indices if float(success[index]) >= 0.5]
        negatives = [index for index in indices if float(success[index]) < 0.5]
        for positive in positives:
            for negative in negatives:
                losses.append(
                    F.relu(
                        utility_logits.new_tensor(float(margin))
                        - utility_logits[positive]
                        + utility_logits[negative]
                    )
                )
    if not losses:
        return utility_logits.sum() * 0.0
    return torch.stack(losses).mean()


def base_supervised_losses(
    *,
    mask_logits: Tensor,
    contact_logits: Tensor,
    utility_logits: Tensor,
    mask_targets: Tensor,
    contact_targets: Tensor,
    success_targets: Tensor,
    positive_weight: Tensor | None = None,
) -> dict[str, Tensor]:
    """Return the mask/contact/success terms before scalar coefficients."""

    if mask_logits.shape != mask_targets.shape:
        raise ValueError("mask logits and targets must have shape [B,N]")
    if contact_logits.shape != contact_targets.transpose(1, 2).shape:
        raise ValueError("contact logits [B,N,H] must match targets [B,H,N]")
    if utility_logits.shape != success_targets.shape:
        raise ValueError("utility logits and success targets must have shape [B]")
    mask_loss = F.binary_cross_entropy_with_logits(
        mask_logits,
        mask_targets,
        pos_weight=positive_weight,
    )
    contact_loss = F.binary_cross_entropy_with_logits(
        contact_logits, contact_targets.transpose(1, 2)
    )
    success_loss = F.binary_cross_entropy_with_logits(utility_logits, success_targets)
    return {
        "mask_bce": mask_loss,
        "contact_time_bce": contact_loss,
        "success_bce": success_loss,
    }


__all__ = [
    "Milestone2BLossCoefficients",
    "base_supervised_losses",
    "counterfactual_assignment_loss",
    "irrelevant_background_stability_loss",
    "pairwise_candidate_ranking_loss",
]
