"""Dependency-light metrics and validation-only calibration for the audit."""
from __future__ import annotations

from typing import Any

import numpy as np


def sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    result = np.empty_like(value)
    positive = value >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exp_value = np.exp(value[~positive])
    result[~positive] = exp_value / (1.0 + exp_value)
    return result


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Select one scalar temperature by deterministic validation NLL search."""
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    log_temperatures = np.linspace(-3.0, 3.0, 1201)
    temperatures = np.exp(log_temperatures)
    scaled = logits[:, None] / temperatures[None]
    losses = (
        np.maximum(scaled, 0.0)
        - scaled * labels[:, None]
        + np.log1p(np.exp(-np.abs(scaled)))
    ).mean(0)
    best = int(np.argmin(losses))
    return float(temperatures[best])


def select_balanced_accuracy_threshold(
    probability: np.ndarray, labels: np.ndarray
) -> tuple[float, float]:
    """Validation-only exact threshold search with deterministic tie breaks."""
    probability = np.asarray(probability, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int8)
    best_key: tuple[float, float, float] | None = None
    best_threshold = 0.5
    best_ba = 0.0
    positives = max(1, int(np.sum(labels == 1)))
    negatives = max(1, int(np.sum(labels == 0)))

    def consider(threshold: float, true_positive: int, false_positive: int) -> None:
        nonlocal best_key, best_threshold, best_ba
        true_negative = negatives - false_positive
        ba = 0.5 * (
            true_positive / positives + true_negative / negatives
        )
        key = (ba, -abs(float(threshold) - 0.5), -float(threshold))
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = float(threshold)
            best_ba = float(ba)

    order = np.argsort(-probability, kind="mergesort")
    sorted_probability = probability[order]
    sorted_labels = labels[order]
    true_positive = 0
    false_positive = 0
    consider(1.0, true_positive, false_positive)
    start = 0
    while start < len(order):
        score = float(sorted_probability[start])
        consider(
            float(np.nextafter(score, np.inf)), true_positive, false_positive
        )
        end = start + 1
        while (
            end < len(order)
            and sorted_probability[end] == sorted_probability[start]
        ):
            end += 1
        group = sorted_labels[start:end]
        true_positive += int(np.sum(group == 1))
        false_positive += int(np.sum(group == 0))
        consider(score, true_positive, false_positive)
        start = end
    consider(0.0, true_positive, false_positive)
    return best_threshold, best_ba


def _weighted_mean(value: np.ndarray, weight: np.ndarray) -> float:
    return float(np.sum(value * weight) / max(float(np.sum(weight)), 1e-12))


def _binary_rank_auc(
    probability: np.ndarray, labels: np.ndarray, weights: np.ndarray
) -> float | None:
    """Weighted pairwise AUROC with half credit for ties."""
    positive = labels == 1
    negative = labels == 0
    positive_weight = float(weights[positive].sum())
    negative_weight = float(weights[negative].sum())
    if positive_weight <= 0 or negative_weight <= 0:
        return None
    order = np.argsort(probability, kind="mergesort")
    score = probability[order]
    y = labels[order]
    w = weights[order]
    concordant = 0.0
    negative_before = 0.0
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and score[end] == score[start]:
            end += 1
        group_negative = float(w[start:end][y[start:end] == 0].sum())
        group_positive = float(w[start:end][y[start:end] == 1].sum())
        concordant += group_positive * (
            negative_before + 0.5 * group_negative
        )
        negative_before += group_negative
        start = end
    return float(concordant / (positive_weight * negative_weight))


def _average_precision(
    probability: np.ndarray, labels: np.ndarray, weights: np.ndarray
) -> float | None:
    positive_weight = float(weights[labels == 1].sum())
    if positive_weight <= 0:
        return None
    order = np.argsort(-probability, kind="mergesort")
    y = labels[order]
    w = weights[order]
    true_positive = np.cumsum(w * (y == 1))
    predicted_positive = np.cumsum(w)
    precision = true_positive / np.maximum(predicted_positive, 1e-12)
    return float(np.sum(precision * w * (y == 1)) / positive_weight)


def binary_metrics(
    probability: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    weights: np.ndarray | None = None,
    bins: int = 15,
) -> dict[str, Any]:
    probability = np.asarray(probability, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int8)
    if weights is None:
        weights = np.ones(len(labels), dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64)
    valid = (
        np.isfinite(probability)
        & np.isfinite(weights)
        & (weights > 0)
        & np.isin(labels, (0, 1))
    )
    probability = probability[valid]
    labels = labels[valid]
    weights = weights[valid]
    prediction = probability >= threshold
    positive = labels == 1
    negative = labels == 0
    positive_weight = float(weights[positive].sum())
    negative_weight = float(weights[negative].sum())
    true_positive_rate = (
        float(weights[prediction & positive].sum()) / max(positive_weight, 1e-12)
    )
    true_negative_rate = (
        float(weights[(~prediction) & negative].sum())
        / max(negative_weight, 1e-12)
    )
    false_accept = (
        float(weights[prediction & negative].sum())
        / max(negative_weight, 1e-12)
    )
    false_reject = (
        float(weights[(~prediction) & positive].sum())
        / max(positive_weight, 1e-12)
    )
    ece = 0.0
    bin_rows = []
    edges = np.linspace(0.0, 1.0, bins + 1)
    total_weight = max(float(weights.sum()), 1e-12)
    for index in range(bins):
        if index == bins - 1:
            selected = (probability >= edges[index]) & (
                probability <= edges[index + 1]
            )
        else:
            selected = (probability >= edges[index]) & (
                probability < edges[index + 1]
            )
        selected_weight = float(weights[selected].sum())
        if selected_weight <= 0:
            continue
        confidence = _weighted_mean(probability[selected], weights[selected])
        accuracy = _weighted_mean(labels[selected], weights[selected])
        contribution = selected_weight / total_weight * abs(confidence - accuracy)
        ece += contribution
        bin_rows.append(
            {
                "lower": float(edges[index]),
                "upper": float(edges[index + 1]),
                "weight": selected_weight,
                "confidence": confidence,
                "positive_rate": accuracy,
                "ece_contribution": float(contribution),
            }
        )
    return {
        "samples": int(len(labels)),
        "effective_weight": float(weights.sum()),
        "threshold": float(threshold),
        "balanced_accuracy": float(
            0.5 * (true_positive_rate + true_negative_rate)
        ),
        "AUROC": _binary_rank_auc(probability, labels, weights),
        "AUPRC": _average_precision(probability, labels, weights),
        "ECE": float(ece),
        "Brier": _weighted_mean(
            np.square(probability - labels), weights
        ),
        "false_accept_rate": false_accept,
        "false_reject_rate": false_reject,
        "calibration_bins": bin_rows,
    }
