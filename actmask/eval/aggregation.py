"""Scenario-wise aggregation for Milestone-2 prediction records."""

from __future__ import annotations

import itertools
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import torch
from torch import Tensor

from actmask.eval.pr_metrics import precision_recall_curve
from actmask.eval.prediction import PredictionRecord


def _record_value(record: PredictionRecord | Mapping[str, Any], key: str, default: Any) -> Any:
    if isinstance(record, PredictionRecord):
        return getattr(record, key, default)
    return record.get(key, default)


def _probabilities(record: PredictionRecord | Mapping[str, Any]) -> Tensor:
    value = _record_value(record, "probabilities", None)
    if value is None:
        value = _record_value(record, "probability", None)
    if value is None:
        raise KeyError("prediction record has no probabilities")
    result = torch.as_tensor(value, dtype=torch.float32).detach().cpu().reshape(-1)
    if not torch.isfinite(result).all():
        raise ValueError("record probabilities must be finite")
    return result


def _targets(record: PredictionRecord | Mapping[str, Any]) -> Tensor:
    value = _record_value(record, "targets", None)
    if value is None:
        value = _record_value(record, "target", None)
    if value is None:
        raise KeyError("prediction record has no targets")
    return torch.as_tensor(value).detach().cpu().reshape(-1) >= 0.5


def _safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _confusion_metrics(probabilities: Tensor, targets: Tensor, threshold: float) -> dict[str, Any]:
    predictions = probabilities >= threshold
    tp = int(torch.logical_and(predictions, targets).sum().item())
    fp = int(torch.logical_and(predictions, ~targets).sum().item())
    fn = int(torch.logical_and(~predictions, targets).sum().item())
    tn = int(torch.logical_and(~predictions, ~targets).sum().item())
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": _safe_ratio(tp, tp + fp),
        "recall": _safe_ratio(tp, tp + fn),
        "f1": _safe_ratio(2 * tp, 2 * tp + fp + fn),
        "iou": _safe_ratio(tp, tp + fp + fn),
        "accuracy": _safe_ratio(tp + tn, tp + fp + fn + tn),
    }


def _pair_groups(
    records: Iterable[PredictionRecord | Mapping[str, Any]],
) -> dict[tuple[str, Any], list[PredictionRecord | Mapping[str, Any]]]:
    groups: dict[tuple[str, Any], list[PredictionRecord | Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        pair_id = _record_value(record, "pair_id", None)
        if pair_id is None:
            continue
        scenario = str(_record_value(record, "scenario", "unknown"))
        groups[(scenario, pair_id)].append(record)
    return groups


def _ordered_pairs(
    records: Iterable[PredictionRecord | Mapping[str, Any]],
    *,
    differing_success: bool = False,
) -> list[tuple[PredictionRecord | Mapping[str, Any], PredictionRecord | Mapping[str, Any]]]:
    pairs: list[
        tuple[PredictionRecord | Mapping[str, Any], PredictionRecord | Mapping[str, Any]]
    ] = []
    for group in _pair_groups(records).values():
        ordered = sorted(
            group,
            key=lambda record: (
                str(_record_value(record, "variant_id", "")),
                str(_record_value(record, "sample_id", "")),
            ),
        )
        for first, second in itertools.combinations(ordered, 2):
            if differing_success and bool(_record_value(first, "success", 0.0)) == bool(
                _record_value(second, "success", 0.0)
            ):
                continue
            pairs.append((first, second))
    return pairs


def paired_scene_consistency(
    records: Iterable[PredictionRecord | Mapping[str, Any]], threshold: float
) -> float:
    """Mean IoU between predicted and target XOR masks for complete pairs."""

    scores: list[float] = []
    for first, second in _ordered_pairs(records):
        first_probabilities, second_probabilities = _probabilities(first), _probabilities(second)
        first_targets, second_targets = _targets(first), _targets(second)
        if first_probabilities.shape != second_probabilities.shape or first_targets.shape != second_targets.shape:
            raise ValueError("paired prediction records must have matching point shapes")
        predicted_change = torch.logical_xor(
            first_probabilities >= threshold, second_probabilities >= threshold
        )
        target_change = torch.logical_xor(first_targets, second_targets)
        intersection = int(torch.logical_and(predicted_change, target_change).sum().item())
        union = int(torch.logical_or(predicted_change, target_change).sum().item())
        scores.append(float(intersection / union) if union else 1.0)
    return float(np.mean(scores)) if scores else 0.0


def success_ranking_accuracy(
    records: Iterable[PredictionRecord | Mapping[str, Any]],
) -> float:
    """Rank successful variants by mean mask probability; exact ties score 0.5."""

    scores: list[float] = []
    for first, second in _ordered_pairs(records, differing_success=True):
        successful, unsuccessful = (
            (first, second)
            if bool(_record_value(first, "success", 0.0))
            else (second, first)
        )
        delta = float(_probabilities(successful).mean()) - float(
            _probabilities(unsuccessful).mean()
        )
        if abs(delta) <= 1.0e-12:
            scores.append(0.5)
        else:
            scores.append(1.0 if delta > 0.0 else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def _uniform_record_value(
    records: list[PredictionRecord | Mapping[str, Any]], key: str, default: Any
) -> Any:
    values = {_record_value(record, key, default) for record in records}
    if len(values) == 1:
        return next(iter(values))
    return "mixed"


def _aggregate_one(
    records: list[PredictionRecord | Mapping[str, Any]],
    *,
    scenario: str,
    threshold: float,
    method_id: str,
    method_kind: str,
    seed: int | None,
    split: str,
    distribution: str | None,
    ood_axis: str | None,
    threshold_source: str,
    training_variant: str | None,
    eval_perturbation: str,
) -> dict[str, Any]:
    probabilities = torch.cat([_probabilities(record) for record in records])
    targets = torch.cat([_targets(record) for record in records])
    if probabilities.shape != targets.shape:
        raise ValueError("concatenated probabilities and targets must match")
    confusion = _confusion_metrics(probabilities, targets, threshold)
    curve = precision_recall_curve(probabilities, targets)
    point_count = int(targets.numel())
    positive_count = int(targets.sum().item())
    pairs = _ordered_pairs(records)
    ranking_pairs = _ordered_pairs(records, differing_success=True)
    pair_consistency = paired_scene_consistency(records, threshold)

    row: dict[str, Any] = {
        "row_type": "run",
        "statistic": "single",
        "seed": seed,
        "method_id": str(method_id),
        "method_kind": str(method_kind),
        "training_variant": training_variant,
        "eval_perturbation": str(eval_perturbation),
        "split": str(split),
        "distribution": str(
            distribution
            if distribution is not None
            else _uniform_record_value(records, "distribution", "ID")
        ),
        "ood_axis": str(
            ood_axis
            if ood_axis is not None
            else _uniform_record_value(records, "ood_axis", "none")
        ),
        "scenario": scenario,
        "threshold": float(threshold),
        "threshold_source": str(threshold_source),
        "samples": len(records),
        "points": point_count,
        "positives": positive_count,
        "negatives": point_count - positive_count,
        "positive_prevalence": float(positive_count / point_count),
        "positive_mask_prevalence": float(positive_count / point_count),
        "has_positive_labels": bool(positive_count),
        "average_precision": curve.average_precision,
        "pr_auc": curve.pr_auc,
        "paired_consistency": pair_consistency,
        "paired_scene_consistency": pair_consistency,
        "pair_count": len(pairs),
        "paired_comparisons": len(pairs),
        "success_ranking_accuracy": success_ranking_accuracy(records),
        "ranking_pair_count": len(ranking_pairs),
        "ranking_comparisons": len(ranking_pairs),
    }
    row.update(confusion)
    return row


def aggregate_scenario_metrics(
    records: Iterable[PredictionRecord | Mapping[str, Any]],
    threshold: float,
    method_id: str = "ActMask",
    method_kind: str = "learned",
    seed: int | None = None,
    split: str = "id_test",
    distribution: str | None = None,
    ood_axis: str | None = None,
    threshold_source: str = "id_validation",
    training_variant: str | None = None,
    eval_perturbation: str = "none",
) -> list[dict[str, Any]]:
    """Return a global row followed by one row for every observed scenario."""

    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError(f"threshold must lie in [0, 1], got {threshold}")
    record_list = list(records)
    if not record_list:
        raise ValueError("cannot aggregate an empty prediction record collection")

    scenario_groups: dict[str, list[PredictionRecord | Mapping[str, Any]]] = defaultdict(list)
    for record in record_list:
        scenario_groups[str(_record_value(record, "scenario", "unknown"))].append(record)

    common = {
        "threshold": float(threshold),
        "method_id": method_id,
        "method_kind": method_kind,
        "seed": seed,
        "split": split,
        "distribution": distribution,
        "ood_axis": ood_axis,
        "threshold_source": threshold_source,
        "training_variant": training_variant,
        "eval_perturbation": eval_perturbation,
    }
    rows = [_aggregate_one(record_list, scenario="__all__", **common)]
    rows.extend(
        _aggregate_one(scenario_groups[scenario], scenario=scenario, **common)
        for scenario in sorted(scenario_groups)
    )
    return rows


# Short alias for experiment orchestration code.
aggregate_records = aggregate_scenario_metrics


__all__ = [
    "aggregate_records",
    "aggregate_scenario_metrics",
    "paired_scene_consistency",
    "success_ranking_accuracy",
]
