"""CPU training utilities for the ActMask prototype.

Exports are resolved lazily so ``python -m actmask.training.train`` does not
pre-import its own entrypoint and trigger a ``runpy`` warning.
"""

_TRAIN_EXPORTS = [
    "build_dataset",
    "compute_positive_weight",
    "run_training",
    "seed_everything",
    "train_one_epoch",
    "validate_one_epoch",
]
_MILESTONE2B_LOSS_EXPORTS = [
    "Milestone2BLossCoefficients",
    "base_supervised_losses",
    "counterfactual_assignment_loss",
    "irrelevant_background_stability_loss",
    "pairwise_candidate_ranking_loss",
]
__all__ = [*_TRAIN_EXPORTS, *_MILESTONE2B_LOSS_EXPORTS]


def __getattr__(name: str):
    if name in _TRAIN_EXPORTS:
        from . import train

        return getattr(train, name)
    if name in _MILESTONE2B_LOSS_EXPORTS:
        from . import milestone2b_losses

        return getattr(milestone2b_losses, name)
    raise AttributeError(name)
