"""Evaluation helpers for learned and analytic ActMask models."""

__all__ = [
    "evaluate_model",
    "paired_scene_consistency",
    "run_evaluation",
    "success_ranking_accuracy",
]


def __getattr__(name: str):
    if name in __all__:
        from . import evaluate

        return getattr(evaluate, name)
    raise AttributeError(name)
