"""Validation-only calibration and reliability diagnostics."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from actmask.eval.pr_metrics import precision_recall_curve


def _arrays(probability: Sequence[float], target: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    score = np.asarray(probability, dtype=np.float64).reshape(-1)
    label = np.asarray(target, dtype=np.float64).reshape(-1)
    if score.shape != label.shape or score.size == 0:
        raise ValueError("probabilities and targets must be equally sized and non-empty")
    if not np.isfinite(score).all() or not np.isfinite(label).all():
        raise ValueError("calibration inputs must be finite")
    if np.any((score < 0.0) | (score > 1.0)) or np.any((label != 0.0) & (label != 1.0)):
        raise ValueError("probabilities must be [0,1] and labels binary")
    return score, label


def reliability_metrics(
    probability: Sequence[float], target: Sequence[float], *, bins: int = 15
) -> dict[str, Any]:
    """ECE, Brier, PR-AUC and deterministic reliability-bin records."""

    score, label = _arrays(probability, target)
    if bins < 2:
        raise ValueError("bins must be at least two")
    edges = np.linspace(0.0, 1.0, bins + 1)
    ids = np.minimum(np.searchsorted(edges, score, side="right") - 1, bins - 1)
    records: list[dict[str, float | int]] = []
    ece = 0.0
    for index in range(bins):
        selected = ids == index
        count = int(selected.sum())
        if not count:
            records.append(
                {
                    "bin": index,
                    "lower": float(edges[index]),
                    "upper": float(edges[index + 1]),
                    "count": 0,
                    "mean_probability": 0.0,
                    "empirical_frequency": 0.0,
                }
            )
            continue
        confidence = float(score[selected].mean())
        frequency = float(label[selected].mean())
        ece += (count / score.size) * abs(confidence - frequency)
        records.append(
            {
                "bin": index,
                "lower": float(edges[index]),
                "upper": float(edges[index + 1]),
                "count": count,
                "mean_probability": confidence,
                "empirical_frequency": frequency,
            }
        )
    curve = precision_recall_curve(score, label)
    return {
        "ece": float(ece),
        "brier": float(np.mean((score - label) ** 2)),
        "average_precision": float(curve.average_precision),
        "pr_auc": float(curve.pr_auc),
        "bins": records,
        "count": int(score.size),
    }


def fit_temperature_scaling(
    logits: Tensor, targets: Tensor, *, max_iter: int = 80
) -> dict[str, float]:
    """Fit one positive temperature strictly on validation logits."""

    if logits.shape != targets.shape:
        raise ValueError("logits and targets must match")
    log_temperature = torch.zeros((), dtype=torch.float64, requires_grad=True)
    value = logits.detach().to(dtype=torch.float64).reshape(-1)
    label = targets.detach().to(dtype=torch.float64).reshape(-1)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.25, max_iter=max_iter)

    def closure() -> Tensor:
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(value / log_temperature.exp(), label)
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = float(log_temperature.detach().exp().clamp(0.05, 20.0))
    return {"temperature": temperature}


def apply_temperature(logits: Tensor | np.ndarray, parameters: dict[str, float]) -> np.ndarray:
    value = logits.detach().cpu().numpy() if isinstance(logits, Tensor) else np.asarray(logits)
    return 1.0 / (1.0 + np.exp(-value / float(parameters["temperature"])))


def fit_platt_scaling(
    scores: Tensor | np.ndarray, targets: Tensor | np.ndarray, *, max_iter: int = 100
) -> dict[str, float]:
    """Fit sigmoid(a * logit(score) + b) only from validation candidates."""

    source = torch.as_tensor(scores, dtype=torch.float64).reshape(-1).clamp(1.0e-5, 1.0 - 1.0e-5)
    label = torch.as_tensor(targets, dtype=torch.float64).reshape(-1)
    if source.shape != label.shape:
        raise ValueError("scores and targets must match")
    feature = torch.logit(source)
    coefficient = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    intercept = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([coefficient, intercept], lr=0.2, max_iter=max_iter)

    def closure() -> Tensor:
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(coefficient * feature + intercept, label)
        loss.backward()
        return loss

    optimizer.step(closure)
    return {"coefficient": float(coefficient.detach()), "intercept": float(intercept.detach())}


def apply_platt(scores: Tensor | np.ndarray, parameters: dict[str, float]) -> np.ndarray:
    source = np.asarray(scores.detach().cpu() if isinstance(scores, Tensor) else scores, dtype=np.float64)
    logit = np.log(np.clip(source, 1.0e-5, 1.0 - 1.0e-5) / np.clip(1.0 - source, 1.0e-5, 1.0))
    return 1.0 / (1.0 + np.exp(-(float(parameters["coefficient"]) * logit + float(parameters["intercept"]))))


def risk_coverage_curve(probability: Sequence[float], target: Sequence[float], *, points: int = 20) -> list[dict[str, float]]:
    """Selective-prediction error as low-confidence samples are deferred."""

    score, label = _arrays(probability, target)
    confidence = np.maximum(score, 1.0 - score)
    order = np.argsort(-confidence, kind="mergesort")
    records: list[dict[str, float]] = []
    for coverage in np.linspace(0.05, 1.0, points):
        count = max(1, int(round(coverage * score.size)))
        selected = order[:count]
        prediction = score[selected] >= 0.5
        records.append(
            {
                "coverage": float(count / score.size),
                "risk": float(np.mean(prediction != (label[selected] >= 0.5))),
            }
        )
    return records


__all__ = [
    "apply_platt",
    "apply_temperature",
    "fit_platt_scaling",
    "fit_temperature_scaling",
    "reliability_metrics",
    "risk_coverage_curve",
]
