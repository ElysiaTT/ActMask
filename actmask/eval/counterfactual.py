"""Quantitative counterfactual consistency metrics for Milestone 2A."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor

from actmask.eval.pr_metrics import precision_recall_curve
from actmask.eval.prediction import PredictionRecord


Record = PredictionRecord | Mapping[str, Any]
_IRRELEVANT_TYPE = "irrelevant_background"
_MISSING = object()
_MATCHING_EPS = 1.0e-12
_MATCHING_TIE_TOLERANCE = 1.0e-12


def _value(record: Record, key: str, default: Any = None) -> Any:
    """Read a field from a record, including nested provenance metadata.

    ``PredictionRecord`` intentionally has a small fixed public schema while
    benchmark-specific provenance (such as ``source_sample_id``) can live in
    its ``metadata`` mapping.  Accepting both locations keeps counterfactual
    pairing independent of the data-loader representation.
    """

    if isinstance(record, PredictionRecord):
        value = getattr(record, key, _MISSING)
        metadata = record.metadata
    else:
        value = record.get(key, _MISSING)
        metadata = record.get("metadata", {})
    if value is not _MISSING and value is not None:
        return value
    if isinstance(metadata, Mapping) and key in metadata:
        metadata_value = metadata[key]
        if metadata_value is not None:
            return metadata_value
    return default


def _identity_key(value: Any) -> tuple[str, Any]:
    """Return a deterministic hashable identity key for scalar metadata."""

    if isinstance(value, Tensor):
        if value.numel() == 1:
            value = value.detach().cpu().item()
        else:
            value = tuple(value.detach().cpu().reshape(-1).tolist())
    elif isinstance(value, np.ndarray):
        if value.size == 1:
            value = value.reshape(-1)[0].item()
        else:
            value = tuple(value.reshape(-1).tolist())
    elif isinstance(value, np.generic):
        value = value.item()
    try:
        hash(value)
    except TypeError:
        value = repr(value)
    return (type(value).__qualname__, value)


def _source_sample_key(record: Record) -> tuple[str, Any] | None:
    """Find the stable source identity for an irrelevant-background pair.

    New datasets should preserve ``source_sample_id`` on both the untouched
    and perturbed record.  ``original_sample_id`` is accepted as a descriptive
    alias, and the historical same-``sample_id`` convention remains supported
    as a compatibility fallback.
    """

    for key in ("source_sample_id", "original_sample_id", "sample_id"):
        value = _value(record, key, _MISSING)
        if value is not _MISSING and value is not None:
            return _identity_key(value)
    return None


def _probabilities(record: Record) -> Tensor:
    value = _value(record, "probabilities")
    if value is None:
        raise KeyError("counterfactual record has no probabilities")
    result = torch.as_tensor(value, dtype=torch.float32).detach().cpu().reshape(-1)
    if not torch.isfinite(result).all() or bool(torch.any((result < 0.0) | (result > 1.0))):
        raise ValueError("counterfactual probabilities must be finite and lie in [0, 1]")
    return result


def _targets(record: Record) -> Tensor:
    value = _value(record, "targets", _value(record, "target"))
    if value is None:
        raise KeyError("counterfactual record has no targets")
    return torch.as_tensor(value).detach().cpu().reshape(-1) >= 0.5


def _optional_mask(record: Record, key: str, length: int, default: bool) -> Tensor:
    value = _value(record, key)
    if value is None:
        return torch.full((length,), default, dtype=torch.bool)
    result = torch.as_tensor(value).detach().cpu().reshape(-1) >= 0.5
    if result.numel() != length:
        raise ValueError(f"{key} must contain {length} values, got {result.numel()}")
    return result


def _pair_type(first: Record, second: Record) -> str:
    first_type = str(_value(first, "counterfactual_type", "none"))
    second_type = str(_value(second, "counterfactual_type", "none"))
    if _IRRELEVANT_TYPE in {first_type, second_type}:
        return _IRRELEVANT_TYPE
    if first_type == second_type:
        return first_type
    nonempty = [value for value in (first_type, second_type) if value not in {"", "none"}]
    return nonempty[0] if len(set(nonempty)) == 1 else "+".join(sorted(set(nonempty)))


@dataclass(frozen=True)
class CounterfactualPair:
    counterfactual_type: str
    scenario: str
    first: Record
    second: Record


def build_counterfactual_pairs(records: Iterable[Record]) -> list[CounterfactualPair]:
    """Build regular variant pairs and explicit irrelevant-background pairs.

    Every returned pair is *strictly* a two-sample counterfactual.  Ordinary
    velocity/action/timing groups must therefore contain exactly two variants;
    silently expanding a larger group into all combinations would invalidate
    assignment-based matching metrics.

    Irrelevant-background records are paired by the preserved
    ``source_sample_id`` (also accepted inside ``metadata``).  This supports a
    materialized original/perturbation dataset even when the two derived
    samples have distinct ``sample_id`` values.  The historical same-sample or
    same-pair/variant convention remains as a fallback.
    """

    record_list = list(records)
    result: list[CounterfactualPair] = []
    irrelevant_indices = {
        index
        for index, record in enumerate(record_list)
        if str(_value(record, "counterfactual_type", "none")) == _IRRELEVANT_TYPE
    }

    # First resolve explicit preserved-source pairs.  The grouping includes
    # non-irrelevant originals as well as records already labelled irrelevant,
    # so either storage convention is valid.  A source group with an
    # irrelevant member has to be exactly a two-record group: otherwise there
    # is no unambiguous correct-vs-swapped assignment.
    source_groups: dict[tuple[str, Any], list[int]] = defaultdict(list)
    for index, record in enumerate(record_list):
        source_key = _source_sample_key(record)
        if source_key is not None:
            source_groups[source_key].append(index)

    consumed_indices: set[int] = set()
    for source_key, indices in sorted(source_groups.items(), key=lambda item: repr(item[0])):
        irrelevant_members = [index for index in indices if index in irrelevant_indices]
        if not irrelevant_members:
            continue
        if len(indices) == 1:
            # Permit the legacy pair-id fallback below for partially annotated
            # records, but do not manufacture a one-sample pair.
            continue
        if len(indices) != 2:
            raise ValueError(
                "irrelevant-background source_sample_id group must contain exactly "
                f"two samples, found {len(indices)} for {source_key!r}"
            )
        first_index, second_index = sorted(
            indices,
            key=lambda index: (
                str(_value(record_list[index], "counterfactual_type", "none"))
                == _IRRELEVANT_TYPE,
                index,
            ),
        )
        first, second = record_list[first_index], record_list[second_index]
        changed = next(
            record_list[index]
            for index in (first_index, second_index)
            if index in irrelevant_indices
        )
        result.append(
            CounterfactualPair(
                counterfactual_type=_IRRELEVANT_TYPE,
                scenario=str(
                    _value(changed, "scenario", _value(first, "scenario", "unknown"))
                ),
                first=first,
                second=second,
            )
        )
        consumed_indices.update(indices)

    # Preserve the legacy identifier fallback for an irrelevant perturbation
    # which has not yet been paired by an explicit source identity.
    for changed_index in sorted(irrelevant_indices):
        if changed_index in consumed_indices:
            continue
        changed = record_list[changed_index]
        changed_sample = _value(changed, "sample_id")
        changed_pair = _value(changed, "pair_id")
        changed_variant = _value(changed, "variant_id")
        candidates: list[tuple[int, Record]] = []
        for index, candidate in enumerate(record_list):
            if (
                index == changed_index
                or index in irrelevant_indices
                or index in consumed_indices
            ):
                continue
            same_sample = changed_sample is not None and _value(candidate, "sample_id") == changed_sample
            same_pair_variant = (
                changed_pair is not None
                and _value(candidate, "pair_id") == changed_pair
                and _value(candidate, "variant_id") == changed_variant
            )
            if same_sample or same_pair_variant:
                candidates.append((index, candidate))
        if not candidates:
            continue
        if len(candidates) != 1:
            raise ValueError(
                "irrelevant-background counterfactual has ambiguous original "
                f"for sample {changed_sample!r}: found {len(candidates)} candidates"
            )
        original_index, original = candidates[0]
        result.append(
            CounterfactualPair(
                counterfactual_type=_IRRELEVANT_TYPE,
                scenario=str(_value(changed, "scenario", _value(original, "scenario", "unknown"))),
                first=original,
                second=changed,
            )
        )
        consumed_indices.update((original_index, changed_index))

    grouped: dict[tuple[str, str, Any], list[Record]] = defaultdict(list)
    for index, record in enumerate(record_list):
        if index in irrelevant_indices or index in consumed_indices:
            continue
        counterfactual_type = str(_value(record, "counterfactual_type", "none"))
        pair_id = _value(record, "pair_id")
        if counterfactual_type in {"", "none", _IRRELEVANT_TYPE} or pair_id is None:
            continue
        scenario = str(_value(record, "scenario", "unknown"))
        grouped[(counterfactual_type, scenario, pair_id)].append(record)

    for (counterfactual_type, scenario, _), group in sorted(
        grouped.items(), key=lambda item: tuple(str(value) for value in item[0])
    ):
        ordered = sorted(
            group,
            key=lambda record: (
                str(_value(record, "variant_id", "")),
                str(_value(record, "sample_id", "")),
            ),
        )
        if len(ordered) != 2:
            raise ValueError(
                "counterfactual group must contain exactly two samples for "
                f"assignment matching, found {len(ordered)} for "
                f"type={counterfactual_type!r}, scenario={scenario!r}"
            )
        result.append(CounterfactualPair(counterfactual_type, scenario, *ordered))
    return result


def _validate_pair_shapes(pair: CounterfactualPair) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    first_probabilities = _probabilities(pair.first)
    second_probabilities = _probabilities(pair.second)
    first_targets = _targets(pair.first)
    second_targets = _targets(pair.second)
    shapes = {
        tuple(first_probabilities.shape),
        tuple(second_probabilities.shape),
        tuple(first_targets.shape),
        tuple(second_targets.shape),
    }
    if len(shapes) != 1:
        raise ValueError("counterfactual pair tensors must have matching point shapes")
    return first_probabilities, second_probabilities, first_targets, second_targets


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _log_probability_label_similarity(probabilities: Tensor, targets: Tensor) -> Tensor:
    """Mean Bernoulli log likelihood: a calibrated probability--label similarity."""

    probabilities = probabilities.to(dtype=torch.float64).clamp(
        _MATCHING_EPS, 1.0 - _MATCHING_EPS
    )
    target_values = targets.to(dtype=torch.float64)
    return (
        target_values * torch.log(probabilities)
        + (1.0 - target_values) * torch.log1p(-probabilities)
    ).mean()


def _matching_metrics(pairs: list[CounterfactualPair]) -> dict[str, float | int]:
    """Score correct versus swapped prediction--GT assignments for each pair.

    ``matching_margin`` is the mean per-pair difference between correct and
    swapped Bernoulli log likelihood.  A positive margin means the prediction
    pair matches its own two ground-truth masks better than the swapped
    assignment.  Identical assignments (notably label-preserving irrelevant
    background perturbations) receive the required tie score of ``0.5``.
    """

    outcomes: list[float] = []
    margins: list[float] = []
    for pair in pairs:
        first_prob, second_prob, first_target, second_target = _validate_pair_shapes(pair)
        correct = 0.5 * (
            _log_probability_label_similarity(first_prob, first_target)
            + _log_probability_label_similarity(second_prob, second_target)
        )
        swapped = 0.5 * (
            _log_probability_label_similarity(first_prob, second_target)
            + _log_probability_label_similarity(second_prob, first_target)
        )
        margin = float((correct - swapped).item())
        margins.append(margin)
        if abs(margin) <= _MATCHING_TIE_TOLERANCE:
            outcomes.append(0.5)
        else:
            outcomes.append(1.0 if margin > 0.0 else 0.0)
    return {
        "matching_accuracy": _mean(outcomes),
        "matching_pair_count": len(pairs),
        "matching_margin": _mean(margins),
    }


def _response_metrics(pairs: list[CounterfactualPair], threshold: float) -> dict[str, Any]:
    change_ious: list[float] = []
    change_scores: list[Tensor] = []
    change_targets: list[Tensor] = []
    signed_correct = 0
    signed_total = 0
    changed_deltas: list[Tensor] = []
    invariant_deltas: list[Tensor] = []
    invariant_agreements: list[Tensor] = []

    for pair in pairs:
        first_prob, second_prob, first_target, second_target = _validate_pair_shapes(pair)
        first_prediction = first_prob >= threshold
        second_prediction = second_prob >= threshold
        target_change = torch.logical_xor(first_target, second_target)
        predicted_change = torch.logical_xor(first_prediction, second_prediction)
        intersection = int(torch.logical_and(predicted_change, target_change).sum().item())
        union = int(torch.logical_or(predicted_change, target_change).sum().item())
        change_ious.append(float(intersection / union) if union else 1.0)

        probability_delta = second_prob - first_prob
        absolute_delta = probability_delta.abs()
        change_scores.append(absolute_delta)
        change_targets.append(target_change)
        if bool(target_change.any()):
            target_direction = second_target.to(torch.int8) - first_target.to(torch.int8)
            signed_correct += int(
                (probability_delta[target_change] * target_direction[target_change] > 0.0)
                .sum()
                .item()
            )
            signed_total += int(target_change.sum().item())
            changed_deltas.append(absolute_delta[target_change])

        invariant = ~target_change
        invariant &= _optional_mask(pair.first, "invariant_mask", len(invariant), True)
        invariant &= _optional_mask(pair.second, "invariant_mask", len(invariant), True)
        if bool(invariant.any()):
            invariant_deltas.append(absolute_delta[invariant])
            invariant_agreements.append(
                (first_prediction[invariant] == second_prediction[invariant]).to(torch.float32)
            )

    concatenated_scores = torch.cat(change_scores)
    concatenated_targets = torch.cat(change_targets)
    curve = precision_recall_curve(concatenated_scores, concatenated_targets)
    invariant_mae = (
        float(torch.cat(invariant_deltas).mean().item()) if invariant_deltas else 0.0
    )
    return {
        "pair_count": len(pairs),
        "change_iou": _mean(change_ious),
        "change_ap": curve.average_precision,
        "change_pr_auc": curve.pr_auc,
        "signed_direction_accuracy": float(signed_correct / signed_total) if signed_total else 0.0,
        "signed_response_accuracy": float(signed_correct / signed_total) if signed_total else 0.0,
        "ground_truth_changed_points": int(concatenated_targets.sum().item()),
        "changed_region_probability_delta": (
            float(torch.cat(changed_deltas).mean().item()) if changed_deltas else 0.0
        ),
        "invariant_points": int(sum(value.numel() for value in invariant_deltas)),
        "invariant_probability_mae": invariant_mae,
        "invariant_probability_stability": 1.0 - invariant_mae,
        "invariant_binary_agreement": (
            float(torch.cat(invariant_agreements).mean().item())
            if invariant_agreements
            else 0.0
        ),
        "invariant_prediction_iou": None,
        "changed_background_false_positive_rate": None,
    }


def _irrelevant_metrics(pairs: list[CounterfactualPair], threshold: float) -> dict[str, Any]:
    invariant_deltas: list[Tensor] = []
    invariant_agreements: list[Tensor] = []
    invariant_ious: list[float] = []
    changed_predictions: list[Tensor] = []
    changed_point_count = 0

    for pair in pairs:
        first_prob, second_prob, first_target, second_target = _validate_pair_shapes(pair)
        if not torch.equal(first_target, second_target):
            raise ValueError("irrelevant-background counterfactuals must preserve ground truth")
        point_count = len(first_target)
        changed = torch.logical_or(
            _optional_mask(pair.first, "changed_input_mask", point_count, False),
            _optional_mask(pair.second, "changed_input_mask", point_count, False),
        )
        invariant = torch.logical_and(
            _optional_mask(pair.first, "invariant_mask", point_count, True),
            _optional_mask(pair.second, "invariant_mask", point_count, True),
        )
        invariant &= ~changed
        if bool(torch.any(first_target & changed)):
            raise ValueError("changed irrelevant-background points must be ground-truth negative")

        first_prediction = first_prob >= threshold
        second_prediction = second_prob >= threshold
        if bool(invariant.any()):
            invariant_deltas.append((second_prob[invariant] - first_prob[invariant]).abs())
            invariant_agreements.append(
                (first_prediction[invariant] == second_prediction[invariant]).to(torch.float32)
            )
            intersection = int(
                torch.logical_and(first_prediction[invariant], second_prediction[invariant])
                .sum()
                .item()
            )
            union = int(
                torch.logical_or(first_prediction[invariant], second_prediction[invariant])
                .sum()
                .item()
            )
            invariant_ious.append(float(intersection / union) if union else 1.0)
        if bool(changed.any()):
            changed_predictions.extend(
                (first_prediction[changed].to(torch.float32), second_prediction[changed].to(torch.float32))
            )
            changed_point_count += int(changed.sum().item())

    invariant_mae = (
        float(torch.cat(invariant_deltas).mean().item()) if invariant_deltas else 0.0
    )
    return {
        "pair_count": len(pairs),
        "change_iou": None,
        "change_ap": None,
        "change_pr_auc": None,
        "signed_direction_accuracy": None,
        "signed_response_accuracy": None,
        "ground_truth_changed_points": 0,
        "changed_region_probability_delta": None,
        "invariant_points": int(sum(value.numel() for value in invariant_deltas)),
        "changed_background_points": changed_point_count,
        "invariant_probability_mae": invariant_mae,
        "invariant_probability_stability": 1.0 - invariant_mae,
        "invariant_binary_agreement": (
            float(torch.cat(invariant_agreements).mean().item())
            if invariant_agreements
            else 0.0
        ),
        "invariant_prediction_iou": _mean(invariant_ious),
        "changed_background_false_positive_rate": (
            float(torch.cat(changed_predictions).mean().item())
            if changed_predictions
            else 0.0
        ),
    }


def counterfactual_pair_metrics(
    pairs: Iterable[CounterfactualPair], threshold: float
) -> dict[str, Any]:
    """Aggregate one homogeneous collection of counterfactual pairs."""

    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError(f"threshold must lie in [0, 1], got {threshold}")
    pair_list = list(pairs)
    if not pair_list:
        raise ValueError("counterfactual pair collection cannot be empty")
    types = {pair.counterfactual_type for pair in pair_list}
    if len(types) != 1:
        raise ValueError("counterfactual_pair_metrics requires one counterfactual type")
    if next(iter(types)) == _IRRELEVANT_TYPE:
        metrics = _irrelevant_metrics(pair_list, float(threshold))
    else:
        metrics = _response_metrics(pair_list, float(threshold))
    metrics.update(_matching_metrics(pair_list))
    return metrics


def _uniform_pair_value(pairs: list[CounterfactualPair], key: str, default: Any) -> Any:
    values = {
        _value(record, key, default)
        for pair in pairs
        for record in (pair.first, pair.second)
    }
    return next(iter(values)) if len(values) == 1 else "mixed"


def aggregate_counterfactual_metrics(
    records: Iterable[Record],
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
    """Emit per-type global rows and per-scenario counterfactual rows."""

    pairs = build_counterfactual_pairs(records)
    if not pairs:
        raise ValueError("no complete counterfactual pairs were found")
    grouped: dict[tuple[str, str], list[CounterfactualPair]] = defaultdict(list)
    by_type: dict[str, list[CounterfactualPair]] = defaultdict(list)
    for pair in pairs:
        grouped[(pair.counterfactual_type, pair.scenario)].append(pair)
        by_type[pair.counterfactual_type].append(pair)

    rows: list[dict[str, Any]] = []
    for counterfactual_type in sorted(by_type):
        scenario_groups = [("__all__", by_type[counterfactual_type])]
        scenario_groups.extend(
            (scenario, grouped[(counterfactual_type, scenario)])
            for scenario in sorted(
                scenario
                for observed_type, scenario in grouped
                if observed_type == counterfactual_type
            )
        )
        for scenario, scenario_pairs in scenario_groups:
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
                    else _uniform_pair_value(scenario_pairs, "distribution", "ID")
                ),
                "ood_axis": str(
                    ood_axis
                    if ood_axis is not None
                    else _uniform_pair_value(scenario_pairs, "ood_axis", "none")
                ),
                "counterfactual_type": counterfactual_type,
                "scenario": scenario,
                "threshold": float(threshold),
                "threshold_source": str(threshold_source),
            }
            row.update(counterfactual_pair_metrics(scenario_pairs, float(threshold)))
            rows.append(row)
    return rows


__all__ = [
    "CounterfactualPair",
    "aggregate_counterfactual_metrics",
    "build_counterfactual_pairs",
    "counterfactual_pair_metrics",
]
