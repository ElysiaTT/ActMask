"""Deterministic, evidence-preserving visualizations for Milestone 2B.

The renderer consumes the same observable-only predictions used by the
experiment.  Exact hidden trajectories are deliberately *not* drawn: the
history panels show noisy/occluded observations, the supplied Temporal model
future hypotheses, and the nominal candidate-action trajectory.  Ground truth
is used only for the mask panel and for honest example selection.

Every requested category produces a PNG.  If the supplied predictions contain
no real qualifying case, the PNG is an explicit placeholder and the manifest
records ``qualifying: false``; the code never relabels a fallback as evidence.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


MILESTONE2B_VISUAL_CATEGORIES: Final[tuple[str, ...]] = (
    "hard_positive",
    "hard_negative",
    "velocity_counterfactual_pair",
    "action_counterfactual_pair",
    "clean_vs_noisy",
    "occluded",
    "temporal_success",
    "temporal_failure",
    "observable_geometry_wins",
    "temporal_wins",
    "action_ranking_success",
    "action_ranking_failure",
)


def _as_numpy(value: Any, *, dtype: np.dtype[Any] | None = None) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        result = value.detach().cpu().numpy()
    else:
        result = np.asarray(value)
    return np.asarray(result, dtype=dtype)


def _scalar(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("expected a scalar tensor")
        return value.detach().cpu().item()
    if isinstance(value, np.ndarray):
        if value.size != 1:
            raise ValueError("expected a scalar array")
        return value.reshape(-1)[0].item()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _bundle_value(bundle: Any, name: str) -> Any:
    if isinstance(bundle, Mapping):
        if name not in bundle:
            raise ValueError(f"prediction bundle is missing {name!r}")
        return bundle[name]
    if not hasattr(bundle, name):
        raise ValueError(f"prediction bundle is missing {name!r}")
    return getattr(bundle, name)


def _validate_threshold(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be finite and lie in [0, 1]")
    return value


def _iou(probability: np.ndarray, target: np.ndarray, threshold: float) -> float:
    predicted = np.asarray(probability).reshape(-1) >= threshold
    truth = np.asarray(target).reshape(-1) >= 0.5
    union = int(np.logical_or(predicted, truth).sum())
    return float(np.logical_and(predicted, truth).sum() / union) if union else 1.0


def _accuracy(probability: np.ndarray, target: np.ndarray, threshold: float) -> float:
    predicted = np.asarray(probability).reshape(-1) >= threshold
    truth = np.asarray(target).reshape(-1) >= 0.5
    return float(np.mean(predicted == truth))


@dataclass(frozen=True)
class _Example:
    """One aligned dataset/prediction row without retaining hidden state."""

    global_index: int
    partition: str
    local_index: int
    dataset: Any
    metadata: Mapping[str, Any]
    temporal_probability: np.ndarray
    baseline_probability: np.ndarray
    target: np.ndarray
    temporal_utility: float
    temporal_future: np.ndarray | None
    model_iou: float
    baseline_iou: float
    model_accuracy: float
    baseline_accuracy: float
    success: bool
    candidate_utility: float
    hard_positive_count: int
    hard_negative_count: int
    visible_fraction: float

    @property
    def sample_id(self) -> Any:
        return self.metadata.get("sample_id", self.global_index)

    @property
    def identity(self) -> tuple[str, int, str]:
        return self.partition, self.local_index, str(self.sample_id)

    def sample(self) -> Mapping[str, Any]:
        sample = self.dataset[self.local_index]
        if not isinstance(sample, Mapping):
            raise TypeError("Milestone 2B visualization requires mapping samples")
        return sample


@dataclass(frozen=True)
class VisualizationSelection:
    """One requested category and the real rows selected for its figure."""

    category: str
    qualifying: bool
    reason: str
    examples: tuple[_Example, ...]
    score: float | None = None
    details: Mapping[str, Any] | None = None

    def manifest_record(self, artifact_path: Path) -> dict[str, Any]:
        return {
            "category": self.category,
            "qualifying": bool(self.qualifying),
            "reason": self.reason,
            "score": self.score,
            "artifact_path": str(artifact_path),
            "examples": [
                {
                    "global_index": example.global_index,
                    "partition": example.partition,
                    "local_index": example.local_index,
                    "sample_id": example.sample_id,
                    "group_id": example.metadata.get("group_id"),
                    "candidate_set_id": example.metadata.get("candidate_set_id"),
                    "candidate_id": example.metadata.get("candidate_id"),
                    "candidate_type": example.metadata.get("candidate_type"),
                    "dynamics_variant": example.metadata.get("dynamics_variant"),
                    "scenario": example.metadata.get("scenario"),
                    "condition": example.metadata.get("condition"),
                    "domain": example.metadata.get("domain"),
                    "model_iou": example.model_iou,
                    "baseline_iou": example.baseline_iou,
                    "success": example.success,
                    "temporal_utility_score": example.temporal_utility,
                    "oracle_candidate_utility": example.candidate_utility,
                }
                for example in self.examples
            ],
            "details": _json_safe(dict(self.details or {})),
        }


@dataclass(frozen=True)
class Milestone2BVisualizationArtifacts:
    """Immutable paths and selections returned by the public API."""

    paths: Mapping[str, Path]
    manifest_path: Path
    selections: tuple[VisualizationSelection, ...]


def _normalise_partitions(
    dataset: Any,
    temporal_predictions: Any,
    fair_baseline_predictions: Any,
) -> list[tuple[str, Any, Any, Any]]:
    dataset_is_mapping = isinstance(dataset, Mapping)
    temporal_is_mapping = isinstance(temporal_predictions, Mapping)
    baseline_is_mapping = isinstance(fair_baseline_predictions, Mapping)
    if dataset_is_mapping:
        if not temporal_is_mapping or not baseline_is_mapping:
            raise TypeError(
                "mapping datasets require matching mappings of Temporal and baseline bundles"
            )
        keys = {str(key) for key in dataset}
        if keys != {str(key) for key in temporal_predictions} or keys != {
            str(key) for key in fair_baseline_predictions
        }:
            raise ValueError("dataset and prediction partition keys must match")
        by_dataset = {str(key): value for key, value in dataset.items()}
        by_temporal = {str(key): value for key, value in temporal_predictions.items()}
        by_baseline = {
            str(key): value for key, value in fair_baseline_predictions.items()
        }
        return [
            (key, by_dataset[key], by_temporal[key], by_baseline[key])
            for key in sorted(keys)
        ]
    if temporal_is_mapping != baseline_is_mapping:
        raise TypeError("Temporal and baseline bundle container types must agree")
    # A plain dictionary can itself be a bundle.  It is not interpreted as a
    # partition mapping when the dataset is singular.
    return [("dataset", dataset, temporal_predictions, fair_baseline_predictions)]


def _metadata_row(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("prediction metadata rows must be mappings")
    return {str(key): _scalar(item) for key, item in value.items()}


def _aligned_examples(
    dataset: Any,
    temporal_predictions: Any,
    fair_baseline_predictions: Any,
    *,
    temporal_threshold: float,
    baseline_threshold: float,
) -> tuple[list[_Example], list[dict[str, Any]]]:
    examples: list[_Example] = []
    partitions_summary: list[dict[str, Any]] = []
    global_index = 0
    for partition, part_dataset, temporal, baseline in _normalise_partitions(
        dataset, temporal_predictions, fair_baseline_predictions
    ):
        size = len(part_dataset)
        temporal_probability = _as_numpy(
            _bundle_value(temporal, "probabilities"), dtype=np.float64
        )
        baseline_probability = _as_numpy(
            _bundle_value(baseline, "probabilities"), dtype=np.float64
        )
        temporal_target = _as_numpy(_bundle_value(temporal, "targets"), dtype=np.float64)
        baseline_target = _as_numpy(_bundle_value(baseline, "targets"), dtype=np.float64)
        temporal_utility = _as_numpy(_bundle_value(temporal, "utility"), dtype=np.float64)
        temporal_metadata = list(_bundle_value(temporal, "metadata"))
        baseline_metadata = list(_bundle_value(baseline, "metadata"))
        temporal_future_raw = _bundle_value(temporal, "future_hypotheses")
        temporal_future = (
            None if temporal_future_raw is None else _as_numpy(temporal_future_raw, dtype=np.float64)
        )
        arrays = (
            temporal_probability,
            baseline_probability,
            temporal_target,
            baseline_target,
            temporal_utility,
        )
        if any(len(array) != size for array in arrays):
            raise ValueError(f"prediction length does not match dataset in {partition!r}")
        if len(temporal_metadata) != size or len(baseline_metadata) != size:
            raise ValueError(f"metadata length does not match dataset in {partition!r}")
        if temporal_future is not None and len(temporal_future) != size:
            raise ValueError(f"future hypothesis length mismatch in {partition!r}")
        if temporal_probability.shape != temporal_target.shape:
            raise ValueError("Temporal probability and target arrays must have equal shape")
        if baseline_probability.shape != temporal_target.shape:
            raise ValueError("baseline probability shape must match the Temporal target shape")
        if not np.array_equal(temporal_target, baseline_target):
            raise ValueError("Temporal and baseline bundles disagree on ground truth")
        if not all(np.isfinite(array).all() for array in arrays):
            raise ValueError("prediction bundles must contain only finite values")
        if (
            (temporal_probability < 0.0).any()
            or (temporal_probability > 1.0).any()
            or (baseline_probability < 0.0).any()
            or (baseline_probability > 1.0).any()
        ):
            raise ValueError("mask probabilities must lie in [0, 1]")

        for local_index in range(size):
            metadata = _metadata_row(temporal_metadata[local_index])
            other_metadata = _metadata_row(baseline_metadata[local_index])
            for key in ("sample_id", "group_id", "candidate_id"):
                if key in metadata and key in other_metadata and metadata[key] != other_metadata[key]:
                    raise ValueError(
                        f"Temporal/baseline metadata mismatch for {key!r} at {partition}[{local_index}]"
                    )
            sample = part_dataset[local_index]
            if not isinstance(sample, Mapping):
                raise TypeError("Milestone 2B visualization requires mapping samples")
            sample_metadata = sample.get("metadata", {})
            if (
                "sample_id" in metadata
                and "sample_id" in sample_metadata
                and metadata["sample_id"] != _scalar(sample_metadata["sample_id"])
            ):
                raise ValueError(
                    f"dataset/prediction sample order mismatch at {partition}[{local_index}]"
                )
            targets = sample["targets"]
            observable = sample["observable"]
            hp = _as_numpy(targets["hard_positive_mask"], dtype=np.bool_)
            hn = _as_numpy(targets["hard_negative_mask"], dtype=np.bool_)
            visibility = _as_numpy(observable["visibility_history"], dtype=np.bool_)
            probability = temporal_probability[local_index].reshape(-1)
            geometry = baseline_probability[local_index].reshape(-1)
            target = temporal_target[local_index].reshape(-1)
            examples.append(
                _Example(
                    global_index=global_index,
                    partition=partition,
                    local_index=local_index,
                    dataset=part_dataset,
                    metadata=metadata,
                    temporal_probability=probability,
                    baseline_probability=geometry,
                    target=target,
                    temporal_utility=float(np.asarray(temporal_utility[local_index]).reshape(-1)[0]),
                    temporal_future=(
                        None
                        if temporal_future is None
                        else np.asarray(temporal_future[local_index], dtype=np.float64)
                    ),
                    model_iou=_iou(probability, target, temporal_threshold),
                    baseline_iou=_iou(geometry, target, baseline_threshold),
                    model_accuracy=_accuracy(probability, target, temporal_threshold),
                    baseline_accuracy=_accuracy(geometry, target, baseline_threshold),
                    success=bool(float(_scalar(targets["success"])) >= 0.5),
                    candidate_utility=float(_scalar(targets["candidate_utility"])),
                    hard_positive_count=int(hp.sum()),
                    hard_negative_count=int(hn.sum()),
                    visible_fraction=float(visibility.mean()),
                )
            )
            global_index += 1
        partitions_summary.append({"partition": partition, "samples": size})
    if not examples:
        raise ValueError("at least one aligned prediction is required")
    return examples, partitions_summary


def _is_irrelevant(example: _Example) -> bool:
    return bool(example.metadata.get("is_irrelevant_perturbation", False))


def _ordinary(examples: Sequence[_Example]) -> list[_Example]:
    return [example for example in examples if not _is_irrelevant(example)]


def _best(
    candidates: Sequence[_Example], key: Any
) -> _Example | None:
    if not candidates:
        return None
    # ``max`` keeps the first exact tie, so a prior identity sort supplies a
    # simple deterministic final tie breaker independent of mapping order.
    return max(sorted(candidates, key=lambda item: item.identity), key=key)


def _placeholder(category: str, reason: str) -> VisualizationSelection:
    return VisualizationSelection(
        category=category,
        qualifying=False,
        reason=reason,
        examples=(),
        score=None,
        details={"selection_policy": "no fallback; explicit placeholder"},
    )


def _single_selection(
    category: str,
    candidates: Sequence[_Example],
    *,
    key: Any,
    reason: str,
    absent_reason: str,
    details: Mapping[str, Any] | None = None,
) -> VisualizationSelection:
    selected = _best(candidates, key)
    if selected is None:
        return _placeholder(category, absent_reason)
    score_values = key(selected)
    return VisualizationSelection(
        category=category,
        qualifying=True,
        reason=reason,
        examples=(selected,),
        score=float(score_values[0]),
        details=details,
    )


def _group_key(example: _Example) -> tuple[str, int]:
    return example.partition, int(example.metadata["group_id"])


def _counterfactual_selection(
    category: str,
    examples: Sequence[_Example],
    *,
    second_key: tuple[int, str],
    label: str,
) -> VisualizationSelection:
    grouped: dict[tuple[str, int], dict[tuple[int, str], _Example]] = defaultdict(dict)
    for example in _ordinary(examples):
        key = (
            int(example.metadata.get("dynamics_variant", -1)),
            str(example.metadata.get("candidate_type", "")),
        )
        grouped[_group_key(example)][key] = example
    pairs: list[tuple[float, _Example, _Example]] = []
    reference_key = (0, "successful")
    for group in sorted(grouped):
        members = grouped[group]
        if reference_key not in members or second_key not in members:
            continue
        first, second = members[reference_key], members[second_key]
        changed = float(np.mean(first.target != second.target))
        if changed > 0.0:
            pairs.append((changed, first, second))
    if not pairs:
        return _placeholder(
            category,
            f"no real {label} pair with a changed ground-truth mask was supplied",
        )
    changed, first, second = max(
        pairs,
        key=lambda item: (item[0], -item[1].global_index, -item[2].global_index),
    )
    return VisualizationSelection(
        category=category,
        qualifying=True,
        reason=f"real paired samples with the largest {label} target-mask change",
        examples=(first, second),
        score=changed,
        details={"target_mask_change_fraction": changed, "pairing": label},
    )


def _clean_noisy_selection(examples: Sequence[_Example]) -> VisualizationSelection:
    ordinary = _ordinary(examples)
    clean = [example for example in ordinary if str(example.metadata.get("condition")) == "clean"]
    noisy = [example for example in ordinary if str(example.metadata.get("condition")) == "noisy"]
    if not clean or not noisy:
        return _placeholder(
            "clean_vs_noisy",
            "both a real clean sample and a real noisy sample were not supplied",
        )
    noisy_example = max(
        noisy,
        key=lambda item: (1.0 - item.model_accuracy, -item.global_index),
    )
    matched = [
        item
        for item in clean
        if item.metadata.get("candidate_type") == noisy_example.metadata.get("candidate_type")
        and item.metadata.get("dynamics_variant")
        == noisy_example.metadata.get("dynamics_variant")
    ]
    if not matched:
        matched = clean
    target_rate = float(noisy_example.target.mean())
    clean_example = min(
        matched,
        key=lambda item: (
            abs(float(item.target.mean()) - target_rate),
            item.global_index,
        ),
    )
    return VisualizationSelection(
        category="clean_vs_noisy",
        qualifying=True,
        reason="real clean and noisy samples matched by candidate type and dynamics variant",
        examples=(clean_example, noisy_example),
        score=float((1.0 - noisy_example.model_accuracy) - (1.0 - clean_example.model_accuracy)),
        details={
            "scene_pairing": "comparison, not a shared hidden scene identity",
            "clean_condition": clean_example.metadata.get("condition"),
            "noisy_condition": noisy_example.metadata.get("condition"),
        },
    )


def _ranking_selections(examples: Sequence[_Example]) -> tuple[VisualizationSelection, VisualizationSelection]:
    grouped: dict[tuple[str, str], list[_Example]] = defaultdict(list)
    for example in _ordinary(examples):
        set_id = example.metadata.get("candidate_set_id")
        if set_id is not None:
            grouped[(example.partition, str(set_id))].append(example)

    successful: list[tuple[float, list[_Example], dict[str, Any]]] = []
    failed: list[tuple[float, list[_Example], dict[str, Any]]] = []
    tolerance = 1.0e-12
    for group_key in sorted(grouped):
        members = grouped[group_key]
        if len(members) < 2:
            continue
        ordered = sorted(
            members,
            key=lambda item: (
                -item.temporal_utility,
                int(item.metadata.get("candidate_id", item.global_index)),
                item.global_index,
            ),
        )
        maximum = ordered[0].temporal_utility
        top_tie = [item for item in ordered if abs(item.temporal_utility - maximum) <= tolerance]
        top_has_success = any(item.success for item in top_tie)
        any_success = any(item.success for item in ordered)
        best_oracle = max(item.candidate_utility for item in ordered)
        displayed = top_tie[0]
        oracle_regret = best_oracle - displayed.candidate_utility
        details = {
            "candidate_set_id": group_key[1],
            "candidate_count": len(ordered),
            "top_tie_count": len(top_tie),
            "top_tie_has_success": top_has_success,
            "top_candidate_id": displayed.metadata.get("candidate_id"),
            "oracle_regret": oracle_regret,
            "tie_tolerance": tolerance,
            "candidate_rows": [
                {
                    "candidate_id": item.metadata.get("candidate_id"),
                    "candidate_type": item.metadata.get("candidate_type"),
                    "temporal_utility_score": item.temporal_utility,
                    "success": item.success,
                    "oracle_candidate_utility": item.candidate_utility,
                }
                for item in ordered
            ],
        }
        if top_has_success:
            margin = maximum - ordered[len(top_tie)].temporal_utility if len(top_tie) < len(ordered) else 0.0
            successful.append((float(margin), ordered, details))
        elif any_success:
            failed.append((float(oracle_regret), ordered, details))

    if successful:
        score, members, details = max(
            successful,
            key=lambda item: (item[0], -item[1][0].global_index),
        )
        success_selection = VisualizationSelection(
            category="action_ranking_success",
            qualifying=True,
            reason="the maximum Temporal utility tie contains a successful real action",
            examples=tuple(members),
            score=score,
            details=details,
        )
    else:
        success_selection = _placeholder(
            "action_ranking_success",
            "no real candidate set had a successful maximum-score Temporal action",
        )
    if failed:
        score, members, details = max(
            failed,
            key=lambda item: (item[0], -item[1][0].global_index),
        )
        failure_selection = VisualizationSelection(
            category="action_ranking_failure",
            qualifying=True,
            reason="no maximum-score tie member succeeds although a lower-ranked real action does",
            examples=tuple(members),
            score=score,
            details=details,
        )
    else:
        failure_selection = _placeholder(
            "action_ranking_failure",
            "no strict real action-ranking failure was present in the supplied predictions",
        )
    return success_selection, failure_selection


def select_milestone2b_visualizations(
    examples: Sequence[_Example],
) -> tuple[VisualizationSelection, ...]:
    """Apply all fixed, deterministic category predicates."""

    ordinary = _ordinary(examples)
    hard_positive = _single_selection(
        "hard_positive",
        [item for item in ordinary if item.hard_positive_count > 0],
        key=lambda item: (
            float(
                np.mean(
                    item.temporal_probability[
                        _as_numpy(item.sample()["targets"]["hard_positive_mask"], dtype=np.bool_)
                    ]
                )
            ),
            float(item.hard_positive_count),
            -float(item.global_index),
        ),
        reason="real sample containing ground-truth hard-positive points",
        absent_reason="no real sample with a hard-positive point was supplied",
        details={"definition": "positive outside current-action proximity"},
    )
    hard_negative = _single_selection(
        "hard_negative",
        [item for item in ordinary if item.hard_negative_count > 0],
        key=lambda item: (
            float(
                np.mean(
                    1.0
                    - item.temporal_probability[
                        _as_numpy(item.sample()["targets"]["hard_negative_mask"], dtype=np.bool_)
                    ]
                )
            ),
            float(item.hard_negative_count),
            -float(item.global_index),
        ),
        reason="real sample containing ground-truth hard-negative points",
        absent_reason="no real sample with a hard-negative point was supplied",
        details={"definition": "current-near point with no future contact"},
    )
    velocity_pair = _counterfactual_selection(
        "velocity_counterfactual_pair",
        examples,
        second_key=(1, "successful"),
        label="velocity/dynamics counterfactual",
    )
    action_pair = _counterfactual_selection(
        "action_counterfactual_pair",
        examples,
        second_key=(0, "wrong_timing"),
        label="action-timing counterfactual",
    )
    clean_noisy = _clean_noisy_selection(examples)
    occluded = _single_selection(
        "occluded",
        [
            item
            for item in ordinary
            if str(item.metadata.get("condition")) == "occluded"
            or str(item.metadata.get("ood_axis")) == "unseen_occlusion_rate"
        ],
        key=lambda item: (
            1.0 - item.visible_fraction,
            1.0 - item.model_accuracy,
            -float(item.global_index),
        ),
        reason="real occluded sample with the largest missing-observation fraction",
        absent_reason="no real occluded sample was supplied",
    )
    temporal_success = _single_selection(
        "temporal_success",
        [
            item
            for item in ordinary
            if item.model_iou >= 0.5 and item.model_iou > item.baseline_iou + 1.0e-12
        ],
        key=lambda item: (
            item.model_iou,
            item.model_iou - item.baseline_iou,
            -float(item.global_index),
        ),
        reason="Temporal reaches sample IoU >= 0.5 and strictly beats the fair baseline",
        absent_reason="no supplied sample met the fixed Temporal-success predicate",
    )
    temporal_failure = _single_selection(
        "temporal_failure",
        [item for item in ordinary if item.model_iou < 0.5],
        key=lambda item: (
            1.0 - item.model_iou,
            item.baseline_iou - item.model_iou,
            -float(item.global_index),
        ),
        reason="real sample with Temporal IoU < 0.5",
        absent_reason="no supplied sample met the fixed Temporal-failure predicate",
    )
    geometry_wins = _single_selection(
        "observable_geometry_wins",
        [item for item in ordinary if item.baseline_iou > item.model_iou + 1.0e-12],
        key=lambda item: (
            item.baseline_iou - item.model_iou,
            item.baseline_iou,
            -float(item.global_index),
        ),
        reason="strongest fair observable geometry baseline has strictly higher sample IoU",
        absent_reason="no supplied sample had a strict fair-baseline IoU win",
    )
    temporal_wins = _single_selection(
        "temporal_wins",
        [item for item in ordinary if item.model_iou > item.baseline_iou + 1.0e-12],
        key=lambda item: (
            item.model_iou - item.baseline_iou,
            item.model_iou,
            -float(item.global_index),
        ),
        reason="Temporal has strictly higher sample IoU than the strongest fair baseline",
        absent_reason="no supplied sample had a strict Temporal IoU win",
    )
    ranking_success, ranking_failure = _ranking_selections(examples)
    by_category = {
        selection.category: selection
        for selection in (
            hard_positive,
            hard_negative,
            velocity_pair,
            action_pair,
            clean_noisy,
            occluded,
            temporal_success,
            temporal_failure,
            geometry_wins,
            temporal_wins,
            ranking_success,
            ranking_failure,
        )
    }
    if set(by_category) != set(MILESTONE2B_VISUAL_CATEGORIES):
        raise RuntimeError("visual selector did not produce every required category")
    return tuple(by_category[category] for category in MILESTONE2B_VISUAL_CATEGORIES)


def _last_observable_state(observable: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    history = _as_numpy(observable["points_history"], dtype=np.float64)
    visibility = _as_numpy(observable["visibility_history"], dtype=np.bool_)
    timestamps = _as_numpy(observable["timestamps"], dtype=np.float64).reshape(-1)
    velocity = _as_numpy(observable["estimated_velocity"], dtype=np.float64)
    if history.ndim != 3 or history.shape[-1] != 3:
        raise ValueError("points_history must have shape [K,N,3]")
    if visibility.shape != history.shape[:2]:
        raise ValueError("visibility_history must have shape [K,N]")
    points = np.zeros((history.shape[1], 3), dtype=np.float64)
    times = np.zeros(history.shape[1], dtype=np.float64)
    any_visible = np.zeros(history.shape[1], dtype=np.bool_)
    for frame in range(history.shape[0]):
        mask = visibility[frame]
        points[mask] = history[frame, mask]
        times[mask] = timestamps[frame]
        any_visible |= mask
    points[any_visible] += velocity[any_visible] * np.maximum(-times[any_visible], 0.0)[:, None]
    return points, any_visible


def _candidate_trajectory(observable: Mapping[str, Any], steps: int = 32) -> np.ndarray:
    action = _as_numpy(observable["action_command"], dtype=np.float64).reshape(-1)
    if action.size < 8 or not np.isfinite(action).all():
        raise ValueError("action_command must be a finite vector with eight entries")
    fractions = np.linspace(0.0, 1.0, steps, dtype=np.float64)
    return action[None, :3] + fractions[:, None] * (action[None, 3:6] - action[None, :3])


def _draw_candidate(ax: Any, trajectory: np.ndarray) -> None:
    ax.plot(
        trajectory[:, 0],
        trajectory[:, 1],
        trajectory[:, 2],
        color="#16853b",
        linewidth=2.8,
        label="nominal candidate trajectory",
        zorder=8,
    )
    ax.scatter(*trajectory[0], marker="o", s=45, c="#16853b", edgecolors="black", linewidths=0.4)
    ax.scatter(*trajectory[-1], marker="X", s=55, c="#16853b", edgecolors="black", linewidths=0.4)


def _future_point_indices(example: _Example, maximum: int = 18) -> np.ndarray:
    relevance = np.maximum.reduce(
        (example.temporal_probability, example.baseline_probability, example.target)
    )
    order = np.lexsort((np.arange(len(relevance)), -relevance))
    return order[: min(maximum, len(order))]


def _draw_context(ax: Any, example: _Example, sample: Mapping[str, Any]) -> None:
    observable = sample["observable"]
    history = _as_numpy(observable["points_history"], dtype=np.float64)
    visibility = _as_numpy(observable["visibility_history"], dtype=np.bool_)
    frames = history.shape[0]
    palette = plt.get_cmap("viridis")
    for frame in range(frames):
        visible = visibility[frame]
        if visible.any():
            ax.scatter(
                history[frame, visible, 0],
                history[frame, visible, 1],
                history[frame, visible, 2],
                s=9 + 4 * frame,
                color=palette(frame / max(frames - 1, 1)),
                alpha=0.32 + 0.14 * frame,
                label="observed history" if frame == frames - 1 else None,
            )
    for point_index in range(history.shape[1]):
        observed = history[visibility[:, point_index], point_index]
        if len(observed) >= 2:
            ax.plot(observed[:, 0], observed[:, 1], observed[:, 2], color="#6b7280", alpha=0.18, linewidth=0.6)
    future = example.temporal_future
    if future is not None:
        if future.ndim != 3 or future.shape[0] != history.shape[1] or future.shape[-1] != 3:
            raise ValueError("Temporal future hypotheses must have shape [N,T,3]")
        indices = _future_point_indices(example)
        cmap = plt.get_cmap("coolwarm")
        for rank, point_index in enumerate(indices):
            values = future[point_index]
            finite = np.isfinite(values).all(axis=1)
            if finite.any():
                ax.plot(
                    values[finite, 0],
                    values[finite, 1],
                    values[finite, 2],
                    color=cmap(float(example.temporal_probability[point_index])),
                    alpha=0.48,
                    linewidth=1.0,
                    label="Temporal future hypotheses" if rank == 0 else None,
                )
    else:
        ax.text2D(0.03, 0.96, "future hypotheses not supplied", transform=ax.transAxes, va="top", color="#9b1c1c")
    _draw_candidate(ax, _candidate_trajectory(observable))
    missing = 1.0 - float(visibility.mean())
    ax.set_title(f"Observable history + predicted futures\nmissing observations={missing:.1%}", fontsize=9.5)


def _draw_mask_panel(
    ax: Any,
    sample: Mapping[str, Any],
    values: np.ndarray,
    *,
    title: str,
    highlight: np.ndarray | None = None,
) -> None:
    observable = sample["observable"]
    points, visible = _last_observable_state(observable)
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.shape != visible.shape:
        raise ValueError("mask values must contain one value per point")
    if visible.any():
        ax.scatter(
            points[visible, 0],
            points[visible, 1],
            points[visible, 2],
            c=values[visible],
            cmap="coolwarm",
            vmin=0.0,
            vmax=1.0,
            s=34,
            edgecolors="#202020",
            linewidths=0.25,
        )
    if highlight is not None:
        highlight = np.asarray(highlight, dtype=np.bool_).reshape(-1) & visible
        if highlight.any():
            ax.scatter(
                points[highlight, 0],
                points[highlight, 1],
                points[highlight, 2],
                facecolors="none",
                edgecolors="#ffd400",
                linewidths=1.8,
                s=85,
                label="hard-case points",
            )
    _draw_candidate(ax, _candidate_trajectory(observable))
    ax.set_title(title, fontsize=9.5)


def _set_equal_limits(axes: Sequence[Any], example: _Example, sample: Mapping[str, Any]) -> None:
    observable = sample["observable"]
    history = _as_numpy(observable["points_history"], dtype=np.float64)
    visibility = _as_numpy(observable["visibility_history"], dtype=np.bool_)
    candidate = _candidate_trajectory(observable)
    geometry = [candidate, history[visibility]]
    current, any_visible = _last_observable_state(observable)
    if any_visible.any():
        geometry.append(current[any_visible])
    base = np.concatenate([item.reshape(-1, 3) for item in geometry if item.size], axis=0)
    low = np.nanmin(base, axis=0)
    high = np.nanmax(base, axis=0)
    center = 0.5 * (low + high)
    base_half = max(float(np.max(high - low)) * 0.58, 0.32)
    half = base_half
    # Include finite, non-explosive portions of predicted futures without letting
    # a single unstable extrapolation collapse the observable geometry to a dot.
    if example.temporal_future is not None:
        future = np.asarray(example.temporal_future, dtype=np.float64).reshape(-1, 3)
        finite = np.isfinite(future).all(axis=1)
        nearby = finite & (
            np.abs(future - center[None, :]) <= 1.6 * base_half
        ).all(axis=1)
        if nearby.any():
            future_low = future[nearby].min(axis=0)
            future_high = future[nearby].max(axis=0)
            future_half = float(
                np.max(np.maximum(center - future_low, future_high - center))
            ) * 1.05
            half = max(base_half, min(future_half, 1.65 * base_half))
    for ax in axes:
        ax.set_xlim(center[0] - half, center[0] + half)
        ax.set_ylim(center[1] - half, center[1] + half)
        ax.set_zlim(center[2] - half, center[2] + half)
        ax.set_box_aspect((1.0, 1.0, 1.0))
        ax.set_xlabel("x", fontsize=7)
        ax.set_ylabel("y", fontsize=7)
        ax.set_zlabel("z", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.view_init(elev=25, azim=-62)


def _row_caption(example: _Example, sample: Mapping[str, Any]) -> str:
    observable = sample["observable"]
    action = _as_numpy(observable["action_command"], dtype=np.float64).reshape(-1)
    nominal_delay = float(_scalar(observable["nominal_action_delay"]))
    return "\n".join(
        (
            f"sample={example.sample_id}",
            f"{example.metadata.get('scenario', 'unknown')} / "
            f"{example.metadata.get('condition', 'unknown')}",
            f"candidate={example.metadata.get('candidate_type', 'unknown')}  "
            f"variant={example.metadata.get('dynamics_variant', '—')}",
            f"IoU: Temporal={example.model_iou:.3f}  fair={example.baseline_iou:.3f}",
            f"rank score={example.temporal_utility:.3f}  success={example.success}",
            f"observable command: delay={nominal_delay:.3f}  "
            f"duration={action[6]:.3f}  radius={action[7]:.3f}",
        )
    )


def _render_placeholder(selection: VisualizationSelection, destination: Path) -> None:
    figure, axis = plt.subplots(figsize=(14, 5.5))
    axis.axis("off")
    axis.text(
        0.5,
        0.64,
        "NO QUALIFYING REAL EXAMPLE",
        transform=axis.transAxes,
        ha="center",
        va="center",
        fontsize=22,
        weight="bold",
        color="#9b1c1c",
    )
    axis.text(
        0.5,
        0.44,
        selection.reason,
        transform=axis.transAxes,
        ha="center",
        va="center",
        fontsize=12,
        wrap=True,
    )
    axis.text(
        0.5,
        0.26,
        "This placeholder is not evidence for the requested category.\nSee selection_manifest.json.",
        transform=axis.transAxes,
        ha="center",
        va="center",
        fontsize=10,
        family="monospace",
    )
    figure.suptitle(f"Milestone 2B: {selection.category} (nonqualifying)", fontsize=15)
    figure.savefig(destination, dpi=150, bbox_inches="tight")
    plt.close(figure)


def render_milestone2b_selection(
    selection: VisualizationSelection,
    output_path: str | Path,
    *,
    temporal_name: str = "TemporalActMask",
    baseline_name: str = "Strongest fair observable baseline",
) -> Path:
    """Render one selected category to a PNG."""

    destination = Path(output_path)
    if destination.suffix.lower() != ".png":
        raise ValueError("Milestone 2B visualization output must end in .png")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not selection.qualifying:
        if selection.examples:
            raise ValueError("a nonqualifying placeholder cannot contain examples")
        _render_placeholder(selection, destination)
        return destination
    if not selection.examples:
        raise ValueError("a qualifying visualization must contain real examples")

    rows = len(selection.examples)
    figure = plt.figure(figsize=(18.5, 4.65 * rows + 1.55), constrained_layout=False)
    axes = np.empty((rows, 4), dtype=object)
    for row in range(rows):
        for column in range(4):
            axes[row, column] = figure.add_subplot(
                rows,
                4,
                row * 4 + column + 1,
                projection="3d",
            )
    for row, example in enumerate(selection.examples):
        sample = example.sample()
        highlight = None
        if selection.category == "hard_positive":
            highlight = _as_numpy(sample["targets"]["hard_positive_mask"], dtype=np.bool_)
        elif selection.category == "hard_negative":
            highlight = _as_numpy(sample["targets"]["hard_negative_mask"], dtype=np.bool_)
        _draw_context(axes[row, 0], example, sample)
        _draw_mask_panel(
            axes[row, 1],
            sample,
            example.target,
            title="Ground-truth future-contact mask",
            highlight=highlight,
        )
        _draw_mask_panel(
            axes[row, 2],
            sample,
            example.baseline_probability,
            title=f"{baseline_name}\nIoU={example.baseline_iou:.3f}",
            highlight=highlight,
        )
        _draw_mask_panel(
            axes[row, 3],
            sample,
            example.temporal_probability,
            title=f"{temporal_name}\nIoU={example.model_iou:.3f}",
            highlight=highlight,
        )
        _set_equal_limits(list(axes[row]), example, sample)
        axes[row, 0].text2D(
            0.015,
            0.015,
            _row_caption(example, sample),
            transform=axes[row, 0].transAxes,
            ha="left",
            va="bottom",
            fontsize=6.8,
            family="monospace",
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "white",
                "edgecolor": "#777777",
                "alpha": 0.82,
            },
        )
    scalar = ScalarMappable(norm=Normalize(0.0, 1.0), cmap="coolwarm")
    scalar.set_array([])
    color_axis = figure.add_axes((0.955, 0.15, 0.012, 0.68))
    figure.colorbar(
        scalar,
        cax=color_axis,
        label="ground truth / mask probability",
    )
    figure.suptitle(
        f"Milestone 2B: {selection.category}\n{selection.reason}",
        fontsize=14,
        y=0.994,
    )
    figure.text(
        0.012,
        0.008,
        "Only observable history and model-predicted future hypotheses are drawn; "
        "the green path is the nominal candidate command, not hidden execution state.",
        ha="left",
        va="bottom",
        fontsize=8,
        family="monospace",
    )
    figure.subplots_adjust(
        top=0.90,
        bottom=0.045,
        left=0.015,
        right=0.94,
        hspace=0.22,
        wspace=0.03,
    )
    figure.savefig(destination, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return destination


def write_milestone2b_selection_manifest(
    path: str | Path,
    selections: Sequence[VisualizationSelection],
    artifact_paths: Mapping[str, Path],
    *,
    temporal_threshold: float,
    baseline_threshold: float,
    partitions: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Write deterministic provenance for every qualifying or placeholder PNG."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    by_category = {selection.category: selection for selection in selections}
    if set(by_category) != set(MILESTONE2B_VISUAL_CATEGORIES):
        raise ValueError("selection manifest requires all Milestone 2B categories")
    payload = {
        "schema_version": "1.0",
        "selection_is_deterministic": True,
        "hidden_state_drawn": False,
        "thresholds": {
            "TemporalActMask": temporal_threshold,
            "fair_observable_baseline": baseline_threshold,
        },
        "partitions": _json_safe(list(partitions)),
        "metadata": _json_safe(dict(metadata or {})),
        "selections": [
            by_category[category].manifest_record(artifact_paths[category])
            for category in MILESTONE2B_VISUAL_CATEGORIES
        ],
    }
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    temporary.replace(destination)
    return destination


def generate_milestone2b_visualizations(
    dataset: Any,
    temporal_predictions: Any,
    fair_baseline_predictions: Any,
    threshold: float,
    output_dir: str | Path,
    *,
    fair_baseline_threshold: float | None = None,
    temporal_name: str = "TemporalActMask",
    baseline_name: str = "Strongest fair observable baseline",
    manifest_metadata: Mapping[str, Any] | None = None,
) -> Milestone2BVisualizationArtifacts:
    """Generate all twelve required Milestone 2B evidence figures.

    ``dataset`` can be a single indexable dataset or a mapping of partition name
    to dataset.  For a mapping, both prediction arguments must be mappings with
    the same keys.  Supplying ID and OOD partitions together enables real
    clean-vs-noisy and occlusion selections without merging or mutating data.
    """

    temporal_threshold = _validate_threshold("threshold", threshold)
    baseline_threshold = _validate_threshold(
        "fair_baseline_threshold",
        temporal_threshold if fair_baseline_threshold is None else fair_baseline_threshold,
    )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    examples, partitions = _aligned_examples(
        dataset,
        temporal_predictions,
        fair_baseline_predictions,
        temporal_threshold=temporal_threshold,
        baseline_threshold=baseline_threshold,
    )
    selections = select_milestone2b_visualizations(examples)
    paths: dict[str, Path] = {}
    for selection in selections:
        path = destination / f"{selection.category}.png"
        paths[selection.category] = render_milestone2b_selection(
            selection,
            path,
            temporal_name=temporal_name,
            baseline_name=baseline_name,
        )
    manifest_path = write_milestone2b_selection_manifest(
        destination / "selection_manifest.json",
        selections,
        paths,
        temporal_threshold=temporal_threshold,
        baseline_threshold=baseline_threshold,
        partitions=partitions,
        metadata=manifest_metadata,
    )
    return Milestone2BVisualizationArtifacts(
        paths=paths,
        manifest_path=manifest_path,
        selections=selections,
    )


__all__ = [
    "MILESTONE2B_VISUAL_CATEGORIES",
    "Milestone2BVisualizationArtifacts",
    "VisualizationSelection",
    "generate_milestone2b_visualizations",
    "render_milestone2b_selection",
    "select_milestone2b_visualizations",
    "write_milestone2b_selection_manifest",
]
