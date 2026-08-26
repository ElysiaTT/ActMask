"""Threshold-independent precision/recall metrics for Milestone 2.

The functions in this module intentionally avoid a scikit-learn dependency.
Scores are treated as probabilities in ``[0, 1]`` and equal scores are
processed as a single threshold, so results do not depend on input ordering.

``precision_recall_curve`` returns points in increasing-recall order.  The
first point is the conventional ``(recall=0, precision=1)`` anchor and each
remaining point corresponds to predicting ``score >= threshold`` for one of
the unique thresholds, visited from highest to lowest score.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np
import torch


ArrayLike = Any


def _to_numpy_1d(value: ArrayLike, *, name: str) -> np.ndarray:
    """Convert a tensor/array-like value to a finite one-dimensional array."""

    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    if array.ndim == 0:
        array = array.reshape(1)
    else:
        array = array.reshape(-1)
    if array.size == 0:
        raise ValueError(f"{name} cannot be empty")
    if not np.issubdtype(array.dtype, np.number) and array.dtype != np.bool_:
        raise TypeError(f"{name} must contain numeric values")
    array = array.astype(np.float64, copy=False)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _validated_inputs(scores: ArrayLike, targets: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    score_array = _to_numpy_1d(scores, name="scores")
    target_array = _to_numpy_1d(targets, name="targets")
    if score_array.shape != target_array.shape:
        raise ValueError(
            "scores and targets must contain the same number of values, got "
            f"{score_array.size} and {target_array.size}"
        )
    if np.any((score_array < 0.0) | (score_array > 1.0)):
        raise ValueError("scores must lie in [0, 1]")
    if np.any((target_array != 0.0) & (target_array != 1.0)):
        raise ValueError("targets must be binary values 0 or 1")
    return score_array, target_array.astype(np.bool_, copy=False)


@dataclass(frozen=True)
class PRCurve:
    """Precision/recall curve and its two common scalar summaries.

    ``precision`` and ``recall`` have length ``len(thresholds) + 1``.  The
    threshold at index ``i`` corresponds to curve values at index ``i + 1``.
    Average precision uses the right-continuous step integral, while
    ``pr_auc`` uses trapezoidal interpolation.  When no positive labels exist,
    both summaries are defined as ``0.0`` and ``has_positive_labels`` is false.
    """

    precision: np.ndarray
    recall: np.ndarray
    thresholds: np.ndarray
    average_precision: float
    pr_auc: float
    has_positive_labels: bool

    def __iter__(self) -> Iterator[np.ndarray]:
        """Allow tuple-style unpacking as precision, recall, thresholds."""

        yield self.precision
        yield self.recall
        yield self.thresholds

    def to_dict(self) -> dict[str, Any]:
        return {
            "precision": self.precision.tolist(),
            "recall": self.recall.tolist(),
            "thresholds": self.thresholds.tolist(),
            "average_precision": self.average_precision,
            "pr_auc": self.pr_auc,
            "has_positive_labels": self.has_positive_labels,
        }


def precision_recall_curve(scores: ArrayLike, targets: ArrayLike) -> PRCurve:
    """Compute a deterministic precision/recall curve with tied scores grouped."""

    score_array, target_array = _validated_inputs(scores, targets)
    # Stable sorting plus explicit equal-score grouping makes the result
    # invariant to the ordering of examples within a score tie.
    order = np.argsort(-score_array, kind="mergesort")
    sorted_scores = score_array[order]
    sorted_targets = target_array[order].astype(np.int64, copy=False)

    cumulative_tp = np.cumsum(sorted_targets, dtype=np.int64)
    cumulative_fp = np.cumsum(1 - sorted_targets, dtype=np.int64)
    group_ends = np.flatnonzero(
        np.r_[sorted_scores[1:] != sorted_scores[:-1], True]
    )
    true_positives = cumulative_tp[group_ends].astype(np.float64)
    false_positives = cumulative_fp[group_ends].astype(np.float64)
    thresholds = sorted_scores[group_ends].astype(np.float64, copy=True)

    point_precision = true_positives / (true_positives + false_positives)
    positive_count = int(target_array.sum())
    if positive_count:
        point_recall = true_positives / float(positive_count)
    else:
        point_recall = np.zeros_like(true_positives)

    precision = np.concatenate((np.asarray([1.0]), point_precision))
    recall = np.concatenate((np.asarray([0.0]), point_recall))
    if positive_count:
        recall_steps = np.diff(recall)
        average_precision_value = float(np.sum(recall_steps * precision[1:]))
        pr_auc_value = float(np.trapezoid(precision, recall))
    else:
        average_precision_value = 0.0
        pr_auc_value = 0.0

    return PRCurve(
        precision=precision,
        recall=recall,
        thresholds=thresholds,
        average_precision=average_precision_value,
        pr_auc=pr_auc_value,
        has_positive_labels=positive_count > 0,
    )


def average_precision(scores: ArrayLike, targets: ArrayLike) -> float:
    """Return the right-continuous step integral of the PR curve."""

    return precision_recall_curve(scores, targets).average_precision


def pr_auc(scores: ArrayLike, targets: ArrayLike) -> float:
    """Return trapezoidal area under the precision/recall curve."""

    return precision_recall_curve(scores, targets).pr_auc


def apply_probability_threshold(scores: ArrayLike, threshold: float) -> np.ndarray:
    """Threshold finite probabilities using the project-wide ``>=`` rule."""

    score_array = _to_numpy_1d(scores, name="scores")
    if np.any((score_array < 0.0) | (score_array > 1.0)):
        raise ValueError("scores must lie in [0, 1]")
    if not np.isfinite(threshold) or not 0.0 <= float(threshold) <= 1.0:
        raise ValueError(f"threshold must lie in [0, 1], got {threshold}")
    return score_array >= float(threshold)


def _confusion_at_threshold(
    scores: np.ndarray, targets: np.ndarray, threshold: float
) -> dict[str, float | int]:
    predicted = scores >= threshold
    tp = int(np.logical_and(predicted, targets).sum())
    fp = int(np.logical_and(predicted, ~targets).sum())
    fn = int(np.logical_and(~predicted, targets).sum())
    tn = int(np.logical_and(~predicted, ~targets).sum())
    precision = float(tp / (tp + fp)) if tp + fp else 0.0
    recall = float(tp / (tp + fn)) if tp + fn else 0.0
    f1 = float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.0
    iou = float(tp / (tp + fp + fn)) if tp + fp + fn else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
    }


@dataclass(frozen=True)
class ThresholdSelection:
    """A threshold fitted exclusively to validation predictions."""

    threshold: float
    validation_f1: float
    validation_precision: float
    validation_recall: float
    validation_iou: float
    tp: int
    fp: int
    fn: int
    tn: int
    candidate_count: int
    selection_split: str = "validation"
    objective: str = "f1"
    tie_break: str = "closest_to_0.5_then_higher"

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "threshold": self.threshold,
            "validation_f1": self.validation_f1,
            "validation_precision": self.validation_precision,
            "validation_recall": self.validation_recall,
            "validation_iou": self.validation_iou,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "tn": self.tn,
            "candidate_count": self.candidate_count,
            "selection_split": self.selection_split,
            "objective": self.objective,
            "tie_break": self.tie_break,
        }


def select_validation_threshold(scores: ArrayLike, targets: ArrayLike) -> ThresholdSelection:
    """Select the validation-only threshold that maximizes micro F1.

    Candidate thresholds are every unique validation probability plus the
    endpoints 0 and 1.  Ties are resolved first by proximity to 0.5 and then by
    the higher threshold.  Test labels are deliberately absent from this API.
    """

    score_array, target_array = _validated_inputs(scores, targets)
    candidates = np.unique(np.concatenate((score_array, np.asarray([0.0, 1.0]))))

    best_threshold = float(candidates[0])
    best_metrics = _confusion_at_threshold(score_array, target_array, best_threshold)
    best_key = (
        float(best_metrics["f1"]),
        -abs(best_threshold - 0.5),
        best_threshold,
    )
    for candidate in candidates[1:]:
        threshold = float(candidate)
        metrics = _confusion_at_threshold(score_array, target_array, threshold)
        key = (float(metrics["f1"]), -abs(threshold - 0.5), threshold)
        if key > best_key:
            best_threshold = threshold
            best_metrics = metrics
            best_key = key

    return ThresholdSelection(
        threshold=best_threshold,
        validation_f1=float(best_metrics["f1"]),
        validation_precision=float(best_metrics["precision"]),
        validation_recall=float(best_metrics["recall"]),
        validation_iou=float(best_metrics["iou"]),
        tp=int(best_metrics["tp"]),
        fp=int(best_metrics["fp"]),
        fn=int(best_metrics["fn"]),
        tn=int(best_metrics["tn"]),
        candidate_count=int(candidates.size),
    )


# A concise alias for callers that already make the validation provenance clear.
select_best_f1_threshold = select_validation_threshold


__all__ = [
    "PRCurve",
    "ThresholdSelection",
    "apply_probability_threshold",
    "average_precision",
    "pr_auc",
    "precision_recall_curve",
    "select_best_f1_threshold",
    "select_validation_threshold",
]
