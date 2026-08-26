"""Deterministic Milestone-2 failure-case selection and 3D rendering.

The module deliberately selects from supplied predictions only.  It never
manufactures a counterfactual, an OOD example, or a ranking error merely to
fill a figure slot.  When a requested failure class is absent, it selects the
closest real example and records ``qualifying: false`` plus the exact fallback
reason in ``selection_manifest.json``.

Every rendered figure has three columns: ground-truth contact mask, the chosen
geometric baseline, and ActMask.  Counterfactual-pair figures use two rows,
one per real paired variant, while preserving those same three columns.
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from actmask.eval.prediction import PredictionRecord
from actmask.eval.ranking import (
    CandidateAction,
    RANKING_SCORE_TIE_TOLERANCE,
    RANKING_TIE_POLICY,
    build_candidate_sets,
    record_value,
    score_tie_blocks,
)


Record = PredictionRecord | Mapping[str, Any]

FAILURE_CATEGORIES = (
    "actmask_beats_geometric",
    "geometric_beats_actmask",
    "velocity_counterfactual_pair",
    "action_counterfactual_pair",
    "ood_failure",
    "action_ranking_failure",
)
_VELOCITY_COUNTERFACTUAL_TYPES = {"velocity", "velocity_and_future_dynamics"}
_ACTION_COUNTERFACTUAL_TYPES = {"action", "action_timing", "action_radius"}


def _scalar(value: Any, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return value
        return value.detach().cpu().item()
    if isinstance(value, np.ndarray):
        if value.size != 1:
            return value
        return value.reshape(-1)[0].item()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _token(value: Any) -> str:
    value = _scalar(value)
    if value is None:
        return "<none>"
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    except (TypeError, ValueError):
        return str(value)


def _array(record: Record, key: str, *, required: bool = True) -> np.ndarray | None:
    value = record_value(record, key, None)
    if value is None and key == "targets":
        value = record_value(record, "target", None)
    if value is None and key == "targets":
        value = record_value(record, "mask", None)
    if value is None and key == "probabilities":
        value = record_value(record, "probability", None)
    if value is None:
        if required:
            raise ValueError(f"failure visualization record is missing {key}")
        return None
    if isinstance(value, torch.Tensor):
        result = value.detach().cpu().numpy()
    else:
        result = np.asarray(value)
    return np.asarray(result)


def _mask(record: Record, *, probability: bool) -> np.ndarray:
    key = "probabilities" if probability else "targets"
    values = _array(record, key, required=True)
    assert values is not None
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError(f"{key} must be finite")
    if probability and ((values < 0.0).any() or (values > 1.0).any()):
        raise ValueError("probabilities must lie in [0, 1]")
    return values


def _iou(prediction: np.ndarray, target: np.ndarray) -> float:
    predicted = np.asarray(prediction, dtype=bool).reshape(-1)
    target = np.asarray(target, dtype=bool).reshape(-1)
    if predicted.shape != target.shape:
        raise ValueError("prediction and target masks must have equal shapes")
    union = int(np.logical_or(predicted, target).sum())
    return float(np.logical_and(predicted, target).sum() / union) if union else 1.0


def _record_identity(record: Record, index: int) -> str:
    sample_id = record_value(record, "sample_id", None)
    counterfactual_type = _token(record_value(record, "counterfactual_type", "none"))
    if sample_id is not None:
        return f"sample={_token(sample_id)}|counterfactual={counterfactual_type}"
    pair_id = record_value(record, "pair_id", None)
    variant_id = record_value(record, "variant_id", None)
    if pair_id is not None or variant_id is not None:
        return (
            f"pair={_token(pair_id)}|variant={_token(variant_id)}"
            f"|counterfactual={counterfactual_type}"
        )
    return f"input_index={index}|counterfactual={counterfactual_type}"


@dataclass(frozen=True)
class FailureExample:
    """A matched real ActMask/geometric prediction for one sample."""

    identity: str
    actmask: Record
    geometric: Record
    actmask_iou: float
    geometric_iou: float
    actmask_threshold: float
    geometric_threshold: float

    @property
    def iou_delta(self) -> float:
        return float(self.actmask_iou - self.geometric_iou)

    @property
    def actmask_error(self) -> float:
        return float(1.0 - self.actmask_iou)


@dataclass(frozen=True)
class FailureSelection:
    """One requested visual category, potentially a documented fallback."""

    category: str
    qualifying: bool
    reason: str
    selection_score: float
    examples: tuple[FailureExample, ...]
    details: Mapping[str, Any]

    def manifest_record(self, artifact_path: str | None = None) -> dict[str, Any]:
        record = {
            "category": self.category,
            "qualifying": bool(self.qualifying),
            "reason": self.reason,
            "selection_score": float(self.selection_score),
            "details": _json_safe(dict(self.details)),
            "examples": [_example_manifest(example) for example in self.examples],
        }
        if artifact_path is not None:
            record["artifact_path"] = artifact_path
        return record


@dataclass(frozen=True)
class FailureVisualizationArtifacts:
    """Paths and immutable selections produced by one rendering invocation."""

    selections: tuple[FailureSelection, ...]
    paths: Mapping[str, Path]
    manifest_path: Path


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return _json_safe(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _summary_fields(record: Record) -> dict[str, Any]:
    keys = (
        "sample_id",
        "group_id",
        "base_scene_id",
        "pair_id",
        "variant_id",
        "candidate_set_id",
        "candidate_id",
        "scenario",
        "counterfactual_type",
        "success",
        "distribution",
        "domain",
        "ood_axis",
    )
    return {
        key: _json_safe(record_value(record, key, None))
        for key in keys
        if record_value(record, key, None) is not None
    }


def _example_manifest(example: FailureExample) -> dict[str, Any]:
    return {
        "identity": example.identity,
        "actmask_iou": float(example.actmask_iou),
        "geometric_iou": float(example.geometric_iou),
        "iou_delta_actmask_minus_geometric": float(example.iou_delta),
        "actmask_threshold": float(example.actmask_threshold),
        "geometric_threshold": float(example.geometric_threshold),
        "metadata": _summary_fields(example.actmask),
    }


def match_prediction_records(
    actmask_records: Iterable[Record],
    geometric_records: Iterable[Record],
    *,
    actmask_threshold: float = 0.5,
    geometric_threshold: float = 0.5,
) -> list[FailureExample]:
    """Match real predictions by stable sample/counterfactual identity."""

    if not 0.0 <= float(actmask_threshold) <= 1.0:
        raise ValueError("actmask_threshold must lie in [0, 1]")
    if not 0.0 <= float(geometric_threshold) <= 1.0:
        raise ValueError("geometric_threshold must lie in [0, 1]")
    actmask_map: dict[str, Record] = {}
    geometric_map: dict[str, Record] = {}
    for index, record in enumerate(actmask_records):
        identity = _record_identity(record, index)
        if identity in actmask_map:
            raise ValueError(f"duplicate ActMask prediction identity: {identity}")
        actmask_map[identity] = record
    for index, record in enumerate(geometric_records):
        identity = _record_identity(record, index)
        if identity in geometric_map:
            raise ValueError(f"duplicate geometric prediction identity: {identity}")
        geometric_map[identity] = record
    common = sorted(set(actmask_map).intersection(geometric_map))
    if not common:
        raise ValueError("ActMask and geometric records have no matching sample identities")
    examples: list[FailureExample] = []
    for identity in common:
        actmask = actmask_map[identity]
        geometric = geometric_map[identity]
        target = _mask(actmask, probability=False) >= 0.5
        baseline_target = _mask(geometric, probability=False) >= 0.5
        if target.shape != baseline_target.shape or not np.array_equal(target, baseline_target):
            raise ValueError(f"matched records disagree on ground truth for {identity}")
        actmask_iou = _iou(_mask(actmask, probability=True) >= actmask_threshold, target)
        geometric_iou = _iou(
            _mask(geometric, probability=True) >= geometric_threshold, target
        )
        examples.append(
            FailureExample(
                identity=identity,
                actmask=actmask,
                geometric=geometric,
                actmask_iou=actmask_iou,
                geometric_iou=geometric_iou,
                actmask_threshold=float(actmask_threshold),
                geometric_threshold=float(geometric_threshold),
            )
        )
    return examples


def _best(examples: Sequence[FailureExample], *, reverse: bool = True) -> FailureExample:
    if not examples:
        raise ValueError("failure selection needs at least one matched prediction")
    return sorted(
        examples,
        key=lambda item: (item.iou_delta, item.identity),
        reverse=reverse,
    )[0]


def _counterfactual_type(example: FailureExample) -> str:
    return str(record_value(example.actmask, "counterfactual_type", "none")).lower()


def _is_velocity_counterfactual(example: FailureExample) -> bool:
    # Candidate-ranking records inherit a source scenario for provenance, but
    # their counterfactual type is `candidate_action_ranking` and they are not
    # a physical velocity-only pair.  Classify solely by the explicit source
    # intervention type rather than a free-form scenario name.
    return _counterfactual_type(example) in _VELOCITY_COUNTERFACTUAL_TYPES


def _is_action_counterfactual(example: FailureExample) -> bool:
    return _counterfactual_type(example) in _ACTION_COUNTERFACTUAL_TYPES


def _pair_token(example: FailureExample) -> str | None:
    pair_id = record_value(example.actmask, "pair_id", None)
    if pair_id is not None:
        return _token(pair_id)
    group_id = record_value(example.actmask, "group_id", None)
    return _token(group_id) if group_id is not None else None


def _variant_token(example: FailureExample) -> str:
    value = record_value(example.actmask, "variant_id", None)
    return _token(value if value is not None else example.identity)


def _counterfactual_change_error(
    first: FailureExample, second: FailureExample
) -> tuple[float, bool]:
    first_target = _mask(first.actmask, probability=False) >= 0.5
    second_target = _mask(second.actmask, probability=False) >= 0.5
    target_change = np.logical_xor(first_target, second_target)
    if not target_change.any():
        return 0.0, False
    first_prediction = _mask(first.actmask, probability=True) >= first.actmask_threshold
    second_prediction = _mask(second.actmask, probability=True) >= second.actmask_threshold
    predicted_change = np.logical_xor(first_prediction, second_prediction)
    return float(1.0 - _iou(predicted_change, target_change)), True


def _select_counterfactual_pair(
    examples: Sequence[FailureExample],
    *,
    category: str,
    predicate: Any,
    fallback: FailureExample,
    epsilon: float,
) -> FailureSelection:
    grouped: dict[str, list[FailureExample]] = defaultdict(list)
    for example in examples:
        if predicate(example):
            pair = _pair_token(example)
            if pair is not None:
                grouped[pair].append(example)
    candidates: list[tuple[float, bool, tuple[FailureExample, FailureExample], str]] = []
    for pair_id, members in sorted(grouped.items()):
        # Only the source dataset's complete binary variant pair is evidence
        # for a velocity/action counterfactual panel.  In particular, do not
        # pair ranking candidates that happen to retain a source pair_id.
        ordered = sorted(members, key=lambda item: (_variant_token(item), item.identity))
        variants = {_variant_token(item) for item in ordered}
        if len(ordered) != 2 or variants != {"0", "1"}:
            continue
        first, second = ordered
        error, targets_changed = _counterfactual_change_error(first, second)
        candidates.append((error, targets_changed, (first, second), pair_id))
    strict = [item for item in candidates if item[1] and item[0] > epsilon]
    if strict:
        score, _, pair, pair_id = sorted(
            strict, key=lambda item: (-item[0], item[3], item[2][0].identity, item[2][1].identity)
        )[0]
        return FailureSelection(
            category=category,
            qualifying=True,
            reason="paired real counterfactual changes ground truth and ActMask has non-zero change-mask error",
            selection_score=score,
            examples=pair,
            details={"pair_id": pair_id, "actmask_change_error": score},
        )
    if candidates:
        score, targets_changed, pair, pair_id = sorted(
            candidates,
            key=lambda item: (-item[0], item[3], item[2][0].identity, item[2][1].identity),
        )[0]
        reason = (
            "no real paired counterfactual has an ActMask change-mask failure"
            if targets_changed
            else "real paired counterfactuals did not change ground truth"
        )
        return FailureSelection(
            category=category,
            qualifying=False,
            reason=reason,
            selection_score=score,
            examples=pair,
            details={"pair_id": pair_id, "actmask_change_error": score},
        )
    return FailureSelection(
        category=category,
        qualifying=False,
        reason="no complete real counterfactual pair of this requested type was available",
        selection_score=fallback.actmask_error,
        examples=(fallback,),
        details={"fallback_identity": fallback.identity},
    )


def _is_ood(example: FailureExample) -> bool:
    distribution = str(record_value(example.actmask, "distribution", "")).lower()
    domain = str(record_value(example.actmask, "domain", "")).lower()
    axis = str(record_value(example.actmask, "ood_axis", "none")).lower()
    return distribution == "ood" or domain == "ood" or (axis not in {"", "none", "null"})


def _candidate_sets_for_examples(
    examples: Sequence[FailureExample],
    *,
    candidate_set_key: str | None,
    candidate_id_key: str | None,
) -> tuple[list[Any], dict[int, FailureExample]]:
    by_record = {id(example.actmask): example for example in examples}
    # Existing counterfactual data may contain records lacking a candidate-set
    # field.  Such records are not evidence of an action-ranking failure, so
    # safely return no candidate sets instead of conflating them with one.
    viable: list[Record] = []
    for example in examples:
        if candidate_set_key is not None:
            if record_value(example.actmask, candidate_set_key, None) is None:
                continue
        elif all(
            record_value(example.actmask, key, None) is None
            for key in (
                "candidate_set_id",
                "candidate_scene_id",
                "ranking_scene_id",
                "action_ranking_scene_id",
            )
        ):
            continue
        viable.append(example.actmask)
    if not viable:
        return [], by_record
    try:
        return (
            build_candidate_sets(
                viable,
                candidate_set_key=candidate_set_key,
                candidate_id_key=candidate_id_key,
                require_success=False,
                min_candidates=2,
            ),
            by_record,
        )
    except ValueError:
        # A malformed candidate set must not be silently treated as a ranking
        # failure.  The manifest will explicitly say no usable set existed.
        return [], by_record


def _select_ranking_failure(
    examples: Sequence[FailureExample],
    *,
    fallback: FailureExample,
    candidate_set_key: str | None,
    candidate_id_key: str | None,
    epsilon: float,
) -> FailureSelection:
    candidate_sets, by_record = _candidate_sets_for_examples(
        examples,
        candidate_set_key=candidate_set_key,
        candidate_id_key=candidate_id_key,
    )
    candidates: list[
        tuple[float, bool, FailureExample, Any, CandidateAction, tuple[CandidateAction, ...]]
    ] = []
    for candidate_set in candidate_sets:
        blocks = score_tie_blocks(candidate_set.candidates)
        top_block = blocks[0]
        display_selected = top_block[0]
        example = by_record.get(id(display_selected.record))
        if example is None:
            continue
        # A tied successful candidate is not a strict ranking failure: an
        # arbitrary candidate-ID display representative must not turn it into
        # a fabricated top-one error.  Require every maximum-score candidate
        # to be unsuccessful and use the best utility in that tied block.
        best_top_utility = max(candidate.oracle_utility for candidate in top_block)
        raw_regret = candidate_set.oracle_utility - best_top_utility
        regret = float(max(0.0, raw_regret))
        has_success = any(candidate.success for candidate in candidate_set.candidates)
        top_has_success = any(candidate.success for candidate in top_block)
        qualifying = bool(has_success and not top_has_success and regret > epsilon)
        candidates.append(
            (regret, qualifying, example, candidate_set, display_selected, top_block)
        )
    strict = [item for item in candidates if item[1]]
    pool = strict or candidates
    if pool:
        regret, qualifying, example, candidate_set, selected, top_block = sorted(
            pool,
            key=lambda item: (-item[0], item[3].scene_id, item[4].candidate_id),
        )[0]
        return FailureSelection(
            category="action_ranking_failure",
            qualifying=qualifying,
            reason=(
                "every real maximum-score candidate is unsuccessful while a higher-utility oracle candidate exists"
                if qualifying
                else "no real candidate set exhibited a strict top-score action-ranking failure"
            ),
            selection_score=regret,
            examples=(example,),
            details={
                "ranking_scene_id": candidate_set.scene_id,
                "candidate_count": len(candidate_set.candidates),
                "top1_display_candidate_id": selected.candidate_id,
                "top1_tied_candidate_ids": [candidate.candidate_id for candidate in top_block],
                "top1_tie_count": len(top_block),
                "top1_tie_has_success": any(candidate.success for candidate in top_block),
                "oracle_candidate_id": candidate_set.oracle_candidate_id,
                "oracle_utility": candidate_set.oracle_utility,
                "best_top_tie_utility": max(
                    candidate.oracle_utility for candidate in top_block
                ),
                "oracle_regret": regret,
                "tie_policy": (
                    "strict failure only when no maximum-score tie member succeeds; "
                    f"{RANKING_TIE_POLICY} uses the score tolerance below"
                ),
                "score_tie_tolerance": RANKING_SCORE_TIE_TOLERANCE,
            },
        )
    return FailureSelection(
        category="action_ranking_failure",
        qualifying=False,
        reason="no usable real multi-candidate ranking set was available",
        selection_score=fallback.actmask_error,
        examples=(fallback,),
        details={"fallback_identity": fallback.identity},
    )


def select_failure_cases(
    actmask_records: Iterable[Record],
    geometric_records: Iterable[Record],
    *,
    actmask_threshold: float = 0.5,
    geometric_threshold: float = 0.5,
    candidate_set_key: str | None = None,
    candidate_id_key: str | None = None,
    epsilon: float = 1.0e-12,
) -> list[FailureSelection]:
    """Select all six required categories without inventing missing cases."""

    epsilon = float(epsilon)
    if not math.isfinite(epsilon) or epsilon < 0.0:
        raise ValueError("epsilon must be finite and non-negative")
    examples = match_prediction_records(
        actmask_records,
        geometric_records,
        actmask_threshold=actmask_threshold,
        geometric_threshold=geometric_threshold,
    )
    fallback = sorted(examples, key=lambda item: (-item.actmask_error, item.identity))[0]

    actmask_better = _best(examples, reverse=True)
    geometric_better = _best(examples, reverse=False)
    selections: list[FailureSelection] = [
        FailureSelection(
            category="actmask_beats_geometric",
            qualifying=actmask_better.iou_delta > epsilon,
            reason=(
                "ActMask has higher sample IoU than the matched geometric baseline"
                if actmask_better.iou_delta > epsilon
                else "no matched sample has strictly higher ActMask IoU than the geometric baseline"
            ),
            selection_score=actmask_better.iou_delta,
            examples=(actmask_better,),
            details={"iou_delta_actmask_minus_geometric": actmask_better.iou_delta},
        ),
        FailureSelection(
            category="geometric_beats_actmask",
            qualifying=geometric_better.iou_delta < -epsilon,
            reason=(
                "matched geometric baseline has higher sample IoU than ActMask"
                if geometric_better.iou_delta < -epsilon
                else "no matched sample has strictly higher geometric-baseline IoU than ActMask"
            ),
            selection_score=-geometric_better.iou_delta,
            examples=(geometric_better,),
            details={"iou_delta_actmask_minus_geometric": geometric_better.iou_delta},
        ),
        _select_counterfactual_pair(
            examples,
            category="velocity_counterfactual_pair",
            predicate=_is_velocity_counterfactual,
            fallback=fallback,
            epsilon=epsilon,
        ),
        _select_counterfactual_pair(
            examples,
            category="action_counterfactual_pair",
            predicate=_is_action_counterfactual,
            fallback=fallback,
            epsilon=epsilon,
        ),
    ]

    ood_examples = [example for example in examples if _is_ood(example)]
    if ood_examples:
        chosen_ood = sorted(ood_examples, key=lambda item: (-item.actmask_error, item.identity))[0]
        selections.append(
            FailureSelection(
                category="ood_failure",
                qualifying=chosen_ood.actmask_error > epsilon,
                reason=(
                    "real OOD sample has non-zero ActMask mask error"
                    if chosen_ood.actmask_error > epsilon
                    else "real OOD samples exist but none has a non-zero ActMask mask error"
                ),
                selection_score=chosen_ood.actmask_error,
                examples=(chosen_ood,),
                details={
                    "ood_axis": record_value(chosen_ood.actmask, "ood_axis", "none"),
                    "actmask_error": chosen_ood.actmask_error,
                },
            )
        )
    else:
        selections.append(
            FailureSelection(
                category="ood_failure",
                qualifying=False,
                reason="no real OOD prediction was available; selected closest non-OOD fallback",
                selection_score=fallback.actmask_error,
                examples=(fallback,),
                details={"fallback_identity": fallback.identity},
            )
        )

    selections.append(
        _select_ranking_failure(
            examples,
            fallback=fallback,
            candidate_set_key=candidate_set_key,
            candidate_id_key=candidate_id_key,
            epsilon=epsilon,
        )
    )
    by_category = {selection.category: selection for selection in selections}
    if set(by_category) != set(FAILURE_CATEGORIES):
        raise RuntimeError("failure selector did not produce every required category")
    return [by_category[category] for category in FAILURE_CATEGORIES]


def _points_velocities_action(record: Record) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = _array(record, "points", required=True)
    velocities = _array(record, "velocities", required=True)
    action = _array(record, "action", required=True)
    assert points is not None and velocities is not None and action is not None
    points = np.asarray(points, dtype=np.float64)
    velocities = np.asarray(velocities, dtype=np.float64)
    action = np.asarray(action, dtype=np.float64).reshape(-1)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape [N, 3] for failure visualization")
    if velocities.shape != points.shape:
        raise ValueError("velocities must have the same [N, 3] shape as points")
    if action.size < 6:
        raise ValueError("action needs at least start and end XYZ values")
    if not np.isfinite(points).all() or not np.isfinite(velocities).all() or not np.isfinite(action).all():
        raise ValueError("points, velocities, and action must be finite")
    return points, velocities, action


def _set_equal_limits(axis: Any, points: np.ndarray, action: np.ndarray) -> None:
    geometry = np.concatenate((points, action[:3][None, :], action[3:6][None, :]), axis=0)
    low = geometry.min(axis=0)
    high = geometry.max(axis=0)
    center = (low + high) / 2.0
    half_extent = max(float((high - low).max()) / 2.0, 0.1) * 1.12
    axis.set_xlim(center[0] - half_extent, center[0] + half_extent)
    axis.set_ylim(center[1] - half_extent, center[1] + half_extent)
    axis.set_zlim(center[2] - half_extent, center[2] + half_extent)
    axis.set_box_aspect((1.0, 1.0, 1.0))


def _draw_panel(
    axis: Any,
    record: Record,
    values: np.ndarray,
    *,
    title: str,
    max_velocity_vectors: int,
    velocity_scale: float,
) -> None:
    points, velocities, action = _points_velocities_action(record)
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.shape != (points.shape[0],):
        raise ValueError("mask values must contain one value per point")
    axis.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c=values,
        cmap="coolwarm",
        vmin=0.0,
        vmax=1.0,
        s=15,
        alpha=0.92,
        depthshade=False,
    )
    vector_count = min(max(int(max_velocity_vectors), 1), len(points))
    vector_indices = np.linspace(0, len(points) - 1, vector_count, dtype=np.int64)
    origin = points[vector_indices]
    vectors = velocities[vector_indices] * float(velocity_scale)
    axis.quiver(
        origin[:, 0],
        origin[:, 1],
        origin[:, 2],
        vectors[:, 0],
        vectors[:, 1],
        vectors[:, 2],
        color="black",
        linewidth=0.55,
        alpha=0.60,
        arrow_length_ratio=0.20,
    )
    start, end = action[:3], action[3:6]
    axis.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], color="darkorange", linewidth=2.5)
    axis.scatter(*start, c="limegreen", marker="^", s=45, edgecolors="black", linewidths=0.4)
    axis.scatter(*end, c="gold", marker="X", s=50, edgecolors="black", linewidths=0.4)
    _set_equal_limits(axis, points, action)
    axis.set_title(title, fontsize=9)
    axis.set_xlabel("x", labelpad=-4, fontsize=7)
    axis.set_ylabel("y", labelpad=-4, fontsize=7)
    axis.set_zlabel("z", labelpad=-4, fontsize=7)
    axis.tick_params(labelsize=6, pad=0)
    axis.view_init(elev=24, azim=-58)


def _info_text(selection: FailureSelection) -> str:
    lines = [
        f"category: {selection.category}",
        f"qualifying: {selection.qualifying}",
        f"reason: {selection.reason}",
    ]
    for example in selection.examples:
        metadata = _summary_fields(example.actmask)
        action = _array(example.actmask, "action", required=False)
        duration = float(np.asarray(action).reshape(-1)[6]) if action is not None and np.asarray(action).size > 6 else None
        radius = float(np.asarray(action).reshape(-1)[7]) if action is not None and np.asarray(action).size > 7 else None
        lines.append(
            " | ".join(
                (
                    f"sample={metadata.get('sample_id', example.identity)}",
                    f"scenario={metadata.get('scenario', 'unknown')}",
                    f"pair={metadata.get('pair_id', '—')}",
                    f"variant={metadata.get('variant_id', '—')}",
                    f"ActMask IoU={example.actmask_iou:.3f}",
                    f"geo IoU={example.geometric_iou:.3f}",
                    f"duration={duration:.3f}" if duration is not None else "duration=—",
                    f"radius={radius:.3f}" if radius is not None else "radius=—",
                )
            )
        )
    return "\n".join(lines)


def render_failure_case(
    selection: FailureSelection,
    output_path: str | Path,
    *,
    actmask_name: str = "ActMask",
    geometric_name: str = "Geometric baseline",
    max_velocity_vectors: int = 24,
    velocity_scale: float = 0.65,
) -> Path:
    """Render one selected case using only real records carried by selection."""

    if not selection.examples:
        raise ValueError("cannot render a failure selection without examples")
    destination = Path(output_path)
    if destination.suffix.lower() != ".png":
        raise ValueError("failure figure output path must end in .png")
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = len(selection.examples)
    figure = plt.figure(figsize=(14.5, 5.0 * rows + 1.35), constrained_layout=False)
    axes = np.empty((rows, 3), dtype=object)
    for row in range(rows):
        for column in range(3):
            axes[row, column] = figure.add_subplot(rows, 3, row * 3 + column + 1, projection="3d")
    for row, example in enumerate(selection.examples):
        target = _mask(example.actmask, probability=False)
        baseline = _mask(example.geometric, probability=True)
        actmask = _mask(example.actmask, probability=True)
        _draw_panel(
            axes[row, 0],
            example.actmask,
            target,
            title="Ground truth",
            max_velocity_vectors=max_velocity_vectors,
            velocity_scale=velocity_scale,
        )
        _draw_panel(
            axes[row, 1],
            example.geometric,
            baseline,
            title=f"{geometric_name}\nIoU={example.geometric_iou:.3f}",
            max_velocity_vectors=max_velocity_vectors,
            velocity_scale=velocity_scale,
        )
        _draw_panel(
            axes[row, 2],
            example.actmask,
            actmask,
            title=f"{actmask_name}\nIoU={example.actmask_iou:.3f}",
            max_velocity_vectors=max_velocity_vectors,
            velocity_scale=velocity_scale,
        )
    scalar_mappable = ScalarMappable(norm=Normalize(0.0, 1.0), cmap="coolwarm")
    scalar_mappable.set_array([])
    figure.colorbar(
        scalar_mappable,
        ax=list(axes.ravel()),
        shrink=0.62,
        pad=0.02,
        label="ground-truth mask / prediction probability",
    )
    figure.suptitle(
        f"Milestone 2 failure view: {selection.category}"
        + ("" if selection.qualifying else " (nonqualifying fallback)"),
        fontsize=14,
        y=0.985,
    )
    figure.text(0.012, 0.01, _info_text(selection), va="bottom", ha="left", fontsize=8.2, family="monospace")
    figure.subplots_adjust(top=0.89, bottom=min(0.12 + 0.035 * (4 + len(selection.examples)), 0.30), wspace=0.04)
    figure.savefig(destination, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return destination


def write_selection_manifest(
    path: str | Path,
    selections: Sequence[FailureSelection],
    *,
    metadata: Mapping[str, Any] | None = None,
    artifact_paths: Mapping[str, str | Path] | None = None,
) -> Path:
    """Write byte-stable selection evidence without timestamps or hidden state."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    paths = {str(key): str(value) for key, value in dict(artifact_paths or {}).items()}
    by_category = {selection.category: selection for selection in selections}
    if set(by_category) != set(FAILURE_CATEGORIES):
        raise ValueError("selection manifest requires exactly the six failure categories")
    payload = {
        "schema_version": "1.0",
        "metadata": _json_safe(dict(metadata or {})),
        "selections": [
            by_category[category].manifest_record(paths.get(category))
            for category in FAILURE_CATEGORIES
        ],
    }
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    temporary.replace(destination)
    return destination


def render_failure_cases(
    selections: Sequence[FailureSelection],
    output_dir: str | Path,
    *,
    actmask_name: str = "ActMask",
    geometric_name: str = "Geometric baseline",
    max_velocity_vectors: int = 24,
    velocity_scale: float = 0.65,
    manifest_metadata: Mapping[str, Any] | None = None,
) -> FailureVisualizationArtifacts:
    """Render all required categories and persist their selection manifest."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    by_category = {selection.category: selection for selection in selections}
    if set(by_category) != set(FAILURE_CATEGORIES):
        raise ValueError("render_failure_cases requires exactly the six failure categories")
    paths: dict[str, Path] = {}
    for category in FAILURE_CATEGORIES:
        path = destination / f"{category}.png"
        paths[category] = render_failure_case(
            by_category[category],
            path,
            actmask_name=actmask_name,
            geometric_name=geometric_name,
            max_velocity_vectors=max_velocity_vectors,
            velocity_scale=velocity_scale,
        )
    manifest = write_selection_manifest(
        destination / "selection_manifest.json",
        [by_category[category] for category in FAILURE_CATEGORIES],
        metadata=manifest_metadata,
        artifact_paths=paths,
    )
    return FailureVisualizationArtifacts(
        selections=tuple(by_category[category] for category in FAILURE_CATEGORIES),
        paths=paths,
        manifest_path=manifest,
    )


def generate_failure_visualizations(
    actmask_records: Iterable[Record],
    geometric_records: Iterable[Record],
    output_dir: str | Path,
    *,
    actmask_threshold: float = 0.5,
    geometric_threshold: float = 0.5,
    candidate_set_key: str | None = None,
    candidate_id_key: str | None = None,
    actmask_name: str = "ActMask",
    geometric_name: str = "Geometric baseline",
    max_velocity_vectors: int = 24,
    velocity_scale: float = 0.65,
    manifest_metadata: Mapping[str, Any] | None = None,
) -> FailureVisualizationArtifacts:
    """Select and render the six categories in one reproducible operation."""

    selections = select_failure_cases(
        list(actmask_records),
        list(geometric_records),
        actmask_threshold=actmask_threshold,
        geometric_threshold=geometric_threshold,
        candidate_set_key=candidate_set_key,
        candidate_id_key=candidate_id_key,
    )
    return render_failure_cases(
        selections,
        output_dir,
        actmask_name=actmask_name,
        geometric_name=geometric_name,
        max_velocity_vectors=max_velocity_vectors,
        velocity_scale=velocity_scale,
        manifest_metadata=manifest_metadata,
    )


__all__ = [
    "FAILURE_CATEGORIES",
    "FailureExample",
    "FailureSelection",
    "FailureVisualizationArtifacts",
    "generate_failure_visualizations",
    "match_prediction_records",
    "render_failure_case",
    "render_failure_cases",
    "select_failure_cases",
    "write_selection_manifest",
]
