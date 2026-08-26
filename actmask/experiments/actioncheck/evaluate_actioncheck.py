"""Evaluation metrics for proposal-level ActionCheck predictions."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from .common import read_jsonl, write_json


def _as_arrays(
    labels: Sequence[int | bool], scores: Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
    target = np.asarray(labels, dtype=np.int64)
    value = np.asarray(scores, dtype=np.float64)
    if target.ndim != 1 or value.ndim != 1 or len(target) != len(value):
        raise ValueError("labels and scores must be equal-length vectors")
    if len(target) == 0 or not np.isfinite(value).all():
        raise ValueError("labels/scores must be non-empty and finite")
    if not set(np.unique(target)).issubset({0, 1}):
        raise ValueError("labels must be binary")
    return target, value


def auroc(labels: Sequence[int | bool], scores: Sequence[float]) -> float:
    target, value = _as_arrays(labels, scores)
    positive = value[target == 1]
    negative = value[target == 0]
    if not len(positive) or not len(negative):
        return float("nan")
    comparison = positive[:, None] - negative[None, :]
    return float(np.mean((comparison > 0) + 0.5 * (comparison == 0)))


def auprc(labels: Sequence[int | bool], scores: Sequence[float]) -> float:
    target, value = _as_arrays(labels, scores)
    positives = int(target.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-value, kind="mergesort")
    ranked = target[order]
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float(np.sum(precision * ranked) / positives)


def best_balanced_threshold(
    labels: Sequence[int | bool], scores: Sequence[float]
) -> float:
    target, value = _as_arrays(labels, scores)
    unique = np.unique(value)
    candidates = np.concatenate(
        (
            [np.nextafter(unique[0], -np.inf)],
            (unique[:-1] + unique[1:]) / 2.0,
            [np.nextafter(unique[-1], np.inf)],
        )
    )
    best = (float("-inf"), 0.5)
    for threshold in candidates:
        metric = classification_metrics(target, value, float(threshold))[
            "balanced_accuracy"
        ]
        candidate = (metric, -abs(float(threshold) - 0.5))
        if candidate > (best[0], -abs(best[1] - 0.5)):
            best = (metric, float(threshold))
    return best[1]


def classification_metrics(
    labels: Sequence[int | bool],
    scores: Sequence[float],
    threshold: float = 0.5,
) -> dict[str, float | int]:
    target, value = _as_arrays(labels, scores)
    prediction = value >= threshold
    positive = target == 1
    negative = ~positive
    true_positive = int(np.sum(prediction & positive))
    true_negative = int(np.sum(~prediction & negative))
    false_positive = int(np.sum(prediction & negative))
    false_negative = int(np.sum(~prediction & positive))
    tpr = true_positive / max(1, int(positive.sum()))
    tnr = true_negative / max(1, int(negative.sum()))
    return {
        "samples": len(target),
        "positives": int(positive.sum()),
        "negatives": int(negative.sum()),
        "threshold": float(threshold),
        "accuracy": float(np.mean(prediction == target)),
        "balanced_accuracy": float((tpr + tnr) / 2.0),
        "auroc": auroc(target, value),
        "auprc": auprc(target, value),
        "false_accept_rate": float(false_positive / max(1, int(negative.sum()))),
        "false_reject_rate": float(false_negative / max(1, int(positive.sum()))),
    }


def grouped_ranking_metrics(
    labels: Sequence[int | bool],
    scores: Sequence[float],
    context_ids: Sequence[str],
) -> dict[str, float | int]:
    target, value = _as_arrays(labels, scores)
    if len(context_ids) != len(target):
        raise ValueError("context_ids length mismatch")
    groups: dict[str, list[int]] = defaultdict(list)
    for index, context_id in enumerate(context_ids):
        groups[str(context_id)].append(index)
    recall = []
    reciprocal_rank = []
    ndcg = []
    margins = []
    best_rank = []
    chance = []
    for indices in groups.values():
        labels_group = target[indices]
        scores_group = value[indices]
        if labels_group.sum() == 0:
            continue
        order = np.argsort(-scores_group, kind="mergesort")
        relevance = labels_group[order]
        positive_ranks = np.flatnonzero(relevance == 1) + 1
        recall.append(float(relevance[0] == 1))
        reciprocal_rank.append(float(1.0 / positive_ranks[0]))
        best_rank.append(float(positive_ranks[0]))
        discounts = 1.0 / np.log2(np.arange(2, len(relevance) + 2))
        dcg = float(np.sum(relevance * discounts))
        ideal = np.sort(labels_group)[::-1]
        idcg = float(np.sum(ideal * discounts))
        ndcg.append(dcg / idcg if idcg else 0.0)
        positive_scores = scores_group[labels_group == 1]
        negative_scores = scores_group[labels_group == 0]
        margins.append(
            float(positive_scores.max() - negative_scores.max())
            if len(negative_scores)
            else float("nan")
        )
        chance.append(float(labels_group.mean()))
    return {
        "eligible_context_groups": len(recall),
        "recall_at_1": float(np.mean(recall)) if recall else float("nan"),
        "mrr": float(np.mean(reciprocal_rank))
        if reciprocal_rank
        else float("nan"),
        "ndcg": float(np.mean(ndcg)) if ndcg else float("nan"),
        "positive_negative_score_margin": float(np.nanmean(margins))
        if margins
        else float("nan"),
        "best_accept_rank": float(np.mean(best_rank))
        if best_rank
        else float("nan"),
        "chance_recall_at_1": float(np.mean(chance)) if chance else float("nan"),
    }


def calibration_metrics(
    labels: Sequence[int | bool], scores: Sequence[float], bins: int = 10
) -> dict[str, float | int]:
    target, value = _as_arrays(labels, scores)
    clipped = np.clip(value, 0.0, 1.0)
    ece = 0.0
    for low in np.linspace(0.0, 1.0, bins + 1)[:-1]:
        high = low + 1.0 / bins
        selected = (clipped >= low) & (
            clipped <= high if high >= 1.0 else clipped < high
        )
        if selected.any():
            ece += float(
                selected.mean()
                * abs(clipped[selected].mean() - target[selected].mean())
            )
    return {
        "samples": len(target),
        "bins": bins,
        "ece": ece,
        "brier_score": float(np.mean(np.square(clipped - target))),
    }


def risk_coverage_metrics(
    labels: Sequence[int | bool],
    scores: Sequence[float],
    *,
    threshold: float,
    coverages: Sequence[float] = (1.0, 0.9, 0.8, 0.7),
) -> list[dict[str, float | int]]:
    target, value = _as_arrays(labels, scores)
    confidence = np.abs(value - threshold)
    order = np.argsort(-confidence, kind="mergesort")
    rows = []
    for coverage in coverages:
        count = max(1, int(np.ceil(len(target) * coverage)))
        selected = order[:count]
        metric = classification_metrics(target[selected], value[selected], threshold)
        rows.append(
            {
                "coverage": float(count / len(target)),
                "requested_coverage": float(coverage),
                "samples": count,
                "selective_accuracy": metric["accuracy"],
                "false_accept_risk": metric["false_accept_rate"],
            }
        )
    return rows


def grouped_bootstrap_ci(
    rows: Sequence[dict[str, Any]],
    metric: Callable[[list[dict[str, Any]]], float],
    *,
    group_key: str = "context_id",
    draws: int = 200,
    seed: int = 1701,
) -> list[float]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[group_key])].append(row)
    keys = sorted(groups)
    if not keys:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(draws):
        sampled = rng.choice(keys, size=len(keys), replace=True)
        bootstrap = [
            row
            for occurrence, key in enumerate(sampled)
            for row in groups[str(key)]
        ]
        values.append(metric(bootstrap))
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def _audit_strata(
    rows: Sequence[dict[str, Any]],
    *,
    threshold: float,
    field: str,
    list_valued: bool = False,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        value = row.get(field)
        values = value if list_valued and isinstance(value, list) else [value]
        if list_valued and not values:
            values = ["none"]
        for item in values:
            grouped[str(item if item not in (None, "") else "missing")].append(row)
    result = {}
    for name, values in sorted(grouped.items()):
        labels = np.asarray([int(row["label"]) for row in values], dtype=np.int64)
        scores = np.asarray([float(row["score"]) for row in values], dtype=np.float64)
        classes = sorted(np.unique(labels).astype(int).tolist())
        record: dict[str, Any] = {
            "samples": len(values),
            "label_distribution": {
                str(label): int(np.sum(labels == label)) for label in classes
            },
            "both_classes_present": len(classes) == 2,
        }
        if len(classes) == 2:
            record["classification"] = classification_metrics(
                labels, scores, threshold
            )
        elif classes == [0]:
            record["classification"] = None
            record["false_accept_rate"] = float(np.mean(scores >= threshold))
            record["undefined_reason"] = "reject-only audit stratum"
        elif classes == [1]:
            record["classification"] = None
            record["false_reject_rate"] = float(np.mean(scores < threshold))
            record["undefined_reason"] = "accept-only audit stratum"
        result[name] = record
    return result


def evaluate_prediction_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    labels = [int(row["label"]) for row in rows]
    scores = [float(row["score"]) for row in rows]
    contexts = [str(row["context_id"]) for row in rows]
    validation = [row for row in rows if row["split"] == "validation"]
    test = [row for row in rows if row["split"] == "test"]
    if not validation or not test:
        raise ValueError("prediction rows require validation and test samples")
    threshold = best_balanced_threshold(
        [row["label"] for row in validation],
        [row["score"] for row in validation],
    )
    test_labels = [row["label"] for row in test]
    test_scores = [row["score"] for row in test]
    test_contexts = [row["context_id"] for row in test]
    classification = classification_metrics(test_labels, test_scores, threshold)
    ranking = grouped_ranking_metrics(test_labels, test_scores, test_contexts)
    ci = grouped_bootstrap_ci(
        test,
        lambda sample: float(
            classification_metrics(
                [row["label"] for row in sample],
                [row["score"] for row in sample],
                threshold,
            )["balanced_accuracy"]
        ),
    )
    return {
        "threshold_selected_on_validation": threshold,
        "classification": classification,
        "grouped_ranking": ranking,
        "calibration": calibration_metrics(test_labels, test_scores),
        "risk_coverage": risk_coverage_metrics(
            test_labels, test_scores, threshold=threshold
        ),
        "context_grouped_balanced_accuracy_ci95": ci,
        "generalization_strata": {
            "per_task": _audit_strata(
                test, threshold=threshold, field="task_id"
            ),
            "per_policy_source": _audit_strata(
                test, threshold=threshold, field="source_policy"
            ),
            "per_reason_code_audit_only": _audit_strata(
                test,
                threshold=threshold,
                field="reason_codes_audit_only",
                list_valued=True,
            ),
            "per_evidence_type_audit_only": _audit_strata(
                test,
                threshold=threshold,
                field="evidence_type_audit_only",
            ),
        },
        "reason_code_and_evidence_type_used_as_model_inputs": False,
        "candidates_from_same_context_treated_independent": False,
    }


def evaluate_file(path: str | Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["scheme"], row["model"])].append(row)
    return {
        "schema": "actioncheck-evaluation-report-v1",
        "models": {
            f"{scheme}/{model}": evaluate_prediction_rows(values)
            for (scheme, model), values in sorted(groups.items())
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = evaluate_file(args.predictions)
    write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
