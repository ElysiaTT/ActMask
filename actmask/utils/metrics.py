"""Binary point-mask metrics used by training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


def _as_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach()
    return torch.as_tensor(value)


def _safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _metrics_from_counts(tp: int, fp: int, fn: int, tn: int) -> dict[str, float | int]:
    precision = _safe_ratio(tp, tp + fp)
    recall = _safe_ratio(tp, tp + fn)
    f1 = _safe_ratio(2 * tp, 2 * tp + fp + fn)
    iou = _safe_ratio(tp, tp + fp + fn)
    accuracy = _safe_ratio(tp + tn, tp + fp + fn + tn)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "accuracy": accuracy,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def binary_mask_metrics(
    predictions: Any,
    targets: Any,
    threshold: float = 0.5,
    *,
    from_logits: bool = True,
) -> dict[str, float | int]:
    """Compute binary metrics over all supplied points.

    Args:
        predictions: Logits by default, or probabilities when
            ``from_logits=False``.
        targets: Binary labels with a shape matching ``predictions``.
        threshold: Probability decision threshold.
        from_logits: Apply a sigmoid before thresholding when true.
    """

    prediction_tensor = _as_tensor(predictions)
    target_tensor = _as_tensor(targets)
    if prediction_tensor.shape != target_tensor.shape:
        raise ValueError(
            "Prediction and target shapes must match, got "
            f"{tuple(prediction_tensor.shape)} and {tuple(target_tensor.shape)}"
        )
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must lie in [0, 1], got {threshold}")

    if prediction_tensor.dtype == torch.bool:
        predicted_positive = prediction_tensor
    else:
        probabilities = torch.sigmoid(prediction_tensor) if from_logits else prediction_tensor
        predicted_positive = probabilities >= threshold
    target_positive = target_tensor >= 0.5

    tp = int(torch.logical_and(predicted_positive, target_positive).sum().item())
    fp = int(torch.logical_and(predicted_positive, ~target_positive).sum().item())
    fn = int(torch.logical_and(~predicted_positive, target_positive).sum().item())
    tn = int(torch.logical_and(~predicted_positive, ~target_positive).sum().item())
    return _metrics_from_counts(tp, fp, fn, tn)


@dataclass
class BinaryMaskMetrics:
    """Accumulate confusion counts without averaging per-batch ratios."""

    threshold: float = 0.5
    from_logits: bool = True
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    def update(self, predictions: Any, targets: Any) -> None:
        counts = binary_mask_metrics(
            predictions,
            targets,
            threshold=self.threshold,
            from_logits=self.from_logits,
        )
        self.tp += int(counts["tp"])
        self.fp += int(counts["fp"])
        self.fn += int(counts["fn"])
        self.tn += int(counts["tn"])

    def compute(self) -> dict[str, float | int]:
        return _metrics_from_counts(self.tp, self.fp, self.fn, self.tn)

    def reset(self) -> None:
        self.tp = self.fp = self.fn = self.tn = 0
