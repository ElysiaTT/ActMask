"""Deterministic candidate-action ranking metrics for Milestone 2.

The point-mask model produces one probability per current point for every
candidate action.  This module turns that mask into a deliberately simple,
auditable action score (the mean point probability by default) and evaluates
the ordering of *multiple candidates from the same scene*.  It is separate
from the two-variant counterfactual diagnostic in :mod:`actmask.eval.aggregation`:
candidate sets here may contain any number of successful and unsuccessful
actions.

The evaluator consumes either :class:`~actmask.eval.prediction.PredictionRecord`
instances or mappings.  Candidate-ranking metadata can live directly on a
mapping or in ``PredictionRecord.metadata``.  Supported aliases make the API
usable while a dataset evolves, but new data should prefer the canonical keys:

``candidate_set_id``, ``candidate_id``, ``success``, and optionally
``oracle_utility`` / ``oracle_candidate_id``.

No label is used to choose a threshold or a model score.  Labels only define
the post-hoc action-ranking metrics.

Scores within the documented absolute tie tolerance are evaluated as a
uniform random ordering within the tied score block.  Candidate IDs remain
stable provenance fields, but they never decide a reported top-k success
metric or regret value.  This prevents a fixed dataset candidate order from
becoming an implicit action-selection cue.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from actmask.eval.prediction import PredictionRecord


Record = PredictionRecord | Mapping[str, Any]

_SCENE_ID_ALIASES = (
    "candidate_set_id",
    "candidate_scene_id",
    "ranking_scene_id",
    "action_ranking_scene_id",
    "scene_id",
)
_CANDIDATE_ID_ALIASES = (
    "candidate_id",
    "ranking_candidate_id",
    "action_candidate_id",
    "candidate_index",
)
_SCORE_ALIASES = (
    "ranking_score",
    "action_score",
    "candidate_score",
    "score",
)
_UTILITY_ALIASES = (
    "candidate_utility",
    # `oracle_utility` is allowed for legacy records that carry only an
    # action-level utility.  New ranking data stores the scene's best utility
    # on every record as `oracle_utility` and the *candidate's* utility under
    # `candidate_utility`, which must therefore take precedence.
    "oracle_utility",
    "utility",
    "oracle_score",
)
# Mean point-mask scores can differ by a few float32 ULPs solely because a
# batch changes the reduction order.  Treating those 1e-8-scale differences as
# a strict action preference would reintroduce an arbitrary candidate-order
# effect, so the benchmark defines an explicitly auditable score tie band.
RANKING_SCORE_TIE_TOLERANCE = 1.0e-6
RANKING_TIE_POLICY = "uniform_expected_within_score_tolerance"
_UTILITY_TIE_TOLERANCE = 1.0e-12


def _scalar(value: Any, default: Any = None) -> Any:
    """Convert scalar tensor/NumPy values without guessing vector semantics."""

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


def record_value(record: Record, key: str, default: Any = None) -> Any:
    """Return a field from a record, including nested prediction metadata.

    Direct fields win over nested metadata so a caller can explicitly override
    a dataset value without mutating the original record.
    """

    sentinel = object()
    value: Any = sentinel
    metadata: Mapping[str, Any] | None = None
    if isinstance(record, Mapping):
        value = record.get(key, sentinel)
        candidate_metadata = record.get("metadata")
        if isinstance(candidate_metadata, Mapping):
            metadata = candidate_metadata
    else:
        value = getattr(record, key, sentinel)
        candidate_metadata = getattr(record, "metadata", None)
        if isinstance(candidate_metadata, Mapping):
            metadata = candidate_metadata
    # PredictionRecord has optional convenience attributes whose default is
    # ``None``.  A populated metadata value must still be visible in that
    # case; otherwise simply adding an optional field would shadow the
    # dataset provenance used by ranking.
    if value is not sentinel and value is not None:
        return _scalar(value, default)
    if metadata is not None and key in metadata:
        return _scalar(metadata[key], default)
    return default


def _first_value(record: Record, names: Sequence[str]) -> tuple[Any, str | None]:
    sentinel = object()
    for name in names:
        value = record_value(record, name, sentinel)
        if value is not sentinel and value is not None:
            return value, name
    return None, None


def _identifier(value: Any, *, field: str) -> str:
    value = _scalar(value)
    if value is None:
        raise ValueError(f"candidate ranking requires non-null {field}")
    if isinstance(value, (str, int, float, bool)):
        result = str(value)
    else:
        try:
            result = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
        except (TypeError, ValueError):
            result = str(value)
    if not result:
        raise ValueError(f"candidate ranking requires non-empty {field}")
    return result


def _as_float(value: Any, *, field: str) -> float:
    value = _scalar(value)
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be a finite scalar, got {value!r}") from error
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite, got {result!r}")
    return result


def _as_bool(value: Any, *, field: str) -> bool:
    value = _scalar(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "success", "successful"}:
            return True
        if normalized in {"0", "false", "no", "failure", "failed"}:
            return False
        raise ValueError(f"{field} cannot be parsed as bool: {value!r}")
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        if not math.isfinite(float(value)):
            raise ValueError(f"{field} must be finite")
        return bool(value)
    raise ValueError(f"{field} cannot be parsed as bool: {value!r}")


def _probability_mean(record: Record) -> float:
    probabilities = record_value(record, "probabilities", None)
    if probabilities is None:
        probabilities = record_value(record, "probability", None)
    if probabilities is None:
        raise ValueError(
            "candidate action score requires probabilities or an explicit ranking score"
        )
    if isinstance(probabilities, torch.Tensor):
        values = probabilities.detach().cpu().numpy()
    else:
        values = np.asarray(probabilities)
    if values.size == 0:
        raise ValueError("candidate probabilities cannot be empty")
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError("candidate probabilities must be finite")
    return float(values.mean())


def candidate_action_score(record: Record, *, score_key: str | None = None) -> float:
    """Return the deterministic action score used for candidate ordering.

    An explicitly supplied ``score_key`` is mandatory when requested.  Without
    one, an existing ranking/action score is used before falling back to the
    mean predicted point-mask probability.  The fallback is intentionally
    transparent and is recorded in every metric row.
    """

    if score_key is not None:
        value = record_value(record, score_key, None)
        if value is None:
            raise ValueError(f"ranking score key {score_key!r} is missing")
        return _as_float(value, field=score_key)
    value, name = _first_value(record, _SCORE_ALIASES)
    if name is not None:
        return _as_float(value, field=name)
    return _probability_mean(record)


def _score_source(record: Record, score_key: str | None) -> str:
    if score_key is not None:
        return score_key
    _, name = _first_value(record, _SCORE_ALIASES)
    # ``PredictionRecord.score`` is explicitly a compatibility property for
    # the mean point-mask score, not a separately trained action-value head.
    if isinstance(record, PredictionRecord) and name == "score":
        return "mean_mask_probability"
    return name or "mean_mask_probability"


@dataclass(frozen=True)
class CandidateAction:
    """One candidate action and its model score / oracle supervision."""

    scene_id: str
    candidate_id: str
    score: float
    success: bool
    oracle_utility: float
    record: Record
    score_source: str
    has_explicit_oracle_marker: bool = False


@dataclass(frozen=True)
class CandidateSet:
    """Canonical sorted candidate actions belonging to a single scene."""

    scene_id: str
    candidates: tuple[CandidateAction, ...]
    oracle_candidate_id: str
    oracle_utility: float
    oracle_source: str


def _explicit_oracle_marker(record: Record, candidate_id: str) -> bool:
    marker = record_value(record, "is_oracle_candidate", None)
    if marker is not None:
        return _as_bool(marker, field="is_oracle_candidate")
    explicit_id = record_value(record, "oracle_candidate_id", None)
    if explicit_id is None:
        return False
    return _identifier(explicit_id, field="oracle_candidate_id") == candidate_id


def build_candidate_sets(
    records: Iterable[Record],
    *,
    candidate_set_key: str | None = None,
    candidate_id_key: str | None = None,
    score_key: str | None = None,
    require_success: bool = True,
    min_candidates: int = 2,
) -> list[CandidateSet]:
    """Group candidate records into deterministic, validated scene sets.

    ``oracle_utility`` is optional.  In its absence, binary ``success`` is the
    oracle utility, which yields the ordinary successful-action ranking task.
    A designated ``oracle_candidate_id`` / ``is_oracle_candidate`` is honored
    for regret reporting; otherwise the max-utility candidate is the oracle.
    """

    if min_candidates < 1:
        raise ValueError("min_candidates must be positive")
    scene_aliases = (candidate_set_key,) if candidate_set_key else _SCENE_ID_ALIASES
    candidate_aliases = (candidate_id_key,) if candidate_id_key else _CANDIDATE_ID_ALIASES
    grouped: dict[str, list[CandidateAction]] = defaultdict(list)
    for record in records:
        scene_value, scene_name = _first_value(record, scene_aliases)
        if scene_name is None:
            expected = candidate_set_key or "/".join(_SCENE_ID_ALIASES)
            raise ValueError(f"candidate ranking record is missing scene key ({expected})")
        candidate_value, candidate_name = _first_value(record, candidate_aliases)
        if candidate_name is None:
            expected = candidate_id_key or "/".join(_CANDIDATE_ID_ALIASES)
            raise ValueError(f"candidate ranking record is missing candidate key ({expected})")
        success_value = record_value(record, "success", None)
        if success_value is None:
            raise ValueError("candidate ranking record is missing success")
        utility_value, utility_name = _first_value(record, _UTILITY_ALIASES)
        scene_id = _identifier(scene_value, field=str(scene_name))
        candidate_id = _identifier(candidate_value, field=str(candidate_name))
        success = _as_bool(success_value, field="success")
        utility = _as_float(utility_value, field=str(utility_name)) if utility_name else float(success)
        grouped[scene_id].append(
            CandidateAction(
                scene_id=scene_id,
                candidate_id=candidate_id,
                score=candidate_action_score(record, score_key=score_key),
                success=success,
                oracle_utility=utility,
                record=record,
                score_source=_score_source(record, score_key),
                has_explicit_oracle_marker=_explicit_oracle_marker(record, candidate_id),
            )
        )

    if not grouped:
        raise ValueError("candidate ranking requires at least one record")

    result: list[CandidateSet] = []
    for scene_id in sorted(grouped):
        candidates = tuple(sorted(grouped[scene_id], key=lambda item: item.candidate_id))
        if len(candidates) < min_candidates:
            raise ValueError(
                f"candidate scene {scene_id!r} has {len(candidates)} candidate(s), "
                f"expected at least {min_candidates}"
            )
        candidate_ids = [item.candidate_id for item in candidates]
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError(f"candidate scene {scene_id!r} has duplicate candidate IDs")
        if require_success and not any(item.success for item in candidates):
            raise ValueError(f"candidate scene {scene_id!r} has no successful action")

        declared_oracle_ids = {
            _identifier(value, field="oracle_candidate_id")
            for item in candidates
            if (value := record_value(item.record, "oracle_candidate_id", None)) is not None
        }
        if len(declared_oracle_ids) > 1:
            raise ValueError(
                f"candidate scene {scene_id!r} declares conflicting oracle_candidate_id values"
            )
        declared_oracle_id = next(iter(declared_oracle_ids), None)
        if declared_oracle_id is not None and declared_oracle_id not in set(candidate_ids):
            raise ValueError(
                f"candidate scene {scene_id!r} declares oracle_candidate_id "
                f"{declared_oracle_id!r} that is not a candidate"
            )

        marked = [item for item in candidates if item.has_explicit_oracle_marker]
        if len(marked) > 1:
            raise ValueError(
                f"candidate scene {scene_id!r} has more than one is_oracle_candidate marker"
            )
        if (
            marked
            and declared_oracle_id is not None
            and marked[0].candidate_id != declared_oracle_id
        ):
            raise ValueError(
                f"candidate scene {scene_id!r} has inconsistent oracle marker and oracle_candidate_id"
            )

        if declared_oracle_id is not None:
            oracle = next(item for item in candidates if item.candidate_id == declared_oracle_id)
            source = "explicit_candidate_id"
        elif marked:
            oracle = marked[0]
            source = "explicit_marker"
        else:
            oracle = sorted(
                candidates, key=lambda item: (-item.oracle_utility, item.candidate_id)
            )[0]
            source = "max_oracle_utility"
        result.append(
            CandidateSet(
                scene_id=scene_id,
                candidates=candidates,
                oracle_candidate_id=oracle.candidate_id,
                oracle_utility=float(oracle.oracle_utility),
                oracle_source=source,
            )
        )
    return result


def _pairwise_accuracy(candidates: Sequence[CandidateAction]) -> tuple[float | None, int]:
    values: list[float] = []
    for left_index, left in enumerate(candidates):
        for right in candidates[left_index + 1 :]:
            utility_delta = left.oracle_utility - right.oracle_utility
            if abs(utility_delta) <= _UTILITY_TIE_TOLERANCE:
                continue
            score_delta = left.score - right.score
            if abs(score_delta) <= RANKING_SCORE_TIE_TOLERANCE:
                values.append(0.5)
            else:
                values.append(1.0 if score_delta * utility_delta > 0.0 else 0.0)
    if not values:
        return None, 0
    return float(np.mean(values)), len(values)


def binary_ranking_auc(candidates: Sequence[CandidateAction]) -> float | None:
    """Tie-aware ROC AUC over successful vs unsuccessful candidate actions."""

    positives = [item for item in candidates if item.success]
    negatives = [item for item in candidates if not item.success]
    if not positives or not negatives:
        return None
    wins: list[float] = []
    for positive in positives:
        for negative in negatives:
            delta = positive.score - negative.score
            wins.append(
                1.0
                if delta > RANKING_SCORE_TIE_TOLERANCE
                else 0.5
                if abs(delta) <= RANKING_SCORE_TIE_TOLERANCE
                else 0.0
            )
    return float(np.mean(wins))


def _mean_or_none(values: Iterable[float | None]) -> float | None:
    valid = [float(value) for value in values if value is not None]
    return float(np.mean(valid)) if valid else None


def score_tie_blocks(
    candidates: Sequence[CandidateAction], *, tolerance: float = RANKING_SCORE_TIE_TOLERANCE
) -> tuple[tuple[CandidateAction, ...], ...]:
    """Return descending score blocks without using IDs to break metric ties.

    IDs provide a deterministic display/provenance order *inside* a block, but
    the block itself is the atomic unit for tie-aware top-k expectations.  A
    caller may use the first display item for a panel title only; it must not
    treat that arbitrary representative as the model's unique choice.
    """

    tolerance = float(tolerance)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("score tie tolerance must be finite and non-negative")
    ordered = sorted(candidates, key=lambda item: (-item.score, item.candidate_id))
    if not ordered:
        raise ValueError("score tie blocks require at least one candidate")
    blocks: list[list[CandidateAction]] = []
    for candidate in ordered:
        if not blocks or abs(candidate.score - blocks[-1][0].score) > tolerance:
            blocks.append([candidate])
        else:
            blocks[-1].append(candidate)
    return tuple(tuple(block) for block in blocks)


def _tie_aware_top_k(
    blocks: Sequence[Sequence[CandidateAction]], *, k: int
) -> tuple[float | None, float, float, int]:
    """Expected recall/hit/utility for uniform ordering within score ties.

    The return is ``(success_recall, success_hit_probability,
    expected_selected_utility, cutoff_tie_size)``.  The selected utility is
    only meaningful for ``k == 1`` but is returned generically to keep the
    tie semantics in one auditable implementation.
    """

    if k <= 0:
        raise ValueError("top-k must be positive")
    candidates = [candidate for block in blocks for candidate in block]
    if not candidates:
        raise ValueError("tie-aware top-k needs candidates")
    success_count = sum(candidate.success for candidate in candidates)
    remaining = min(int(k), len(candidates))
    expected_successes = 0.0
    probability_no_success = 1.0
    cutoff_tie_size = 0
    expected_utility = 0.0
    for block in blocks:
        if remaining <= 0:
            break
        block_count = len(block)
        take = min(remaining, block_count)
        block_successes = sum(candidate.success for candidate in block)
        expected_successes += float(take) * float(block_successes) / float(block_count)
        if take == block_count:
            if block_successes:
                probability_no_success = 0.0
        else:
            # Under a uniformly random tie order, the top-k cutoff receives a
            # uniform subset without replacement from this tied score block.
            failures = block_count - block_successes
            probability_no_success *= float(math.comb(failures, take)) / float(
                math.comb(block_count, take)
            )
        if k == 1 and take:
            expected_utility = float(
                np.mean([candidate.oracle_utility for candidate in block])
            )
        if take < block_count or remaining == take:
            cutoff_tie_size = block_count
        remaining -= take
    recall = float(expected_successes / success_count) if success_count else None
    return recall, float(1.0 - probability_no_success), expected_utility, cutoff_tie_size


def _context(
    *,
    method_id: str,
    method_kind: str,
    seed: int | None,
    split: str,
    distribution: str,
    ood_axis: str,
    training_variant: str | None,
    eval_perturbation: str,
) -> dict[str, Any]:
    return {
        "method_id": str(method_id),
        "method_kind": str(method_kind),
        "seed": None if seed is None else int(seed),
        "split": str(split),
        "distribution": str(distribution),
        "ood_axis": str(ood_axis),
        "training_variant": training_variant,
        "eval_perturbation": str(eval_perturbation),
    }


def action_ranking_records(
    records: Iterable[Record],
    *,
    method_id: str = "ActMask",
    method_kind: str = "learned",
    seed: int | None = None,
    split: str = "id_test",
    distribution: str = "ID",
    ood_axis: str = "none",
    training_variant: str | None = None,
    eval_perturbation: str = "none",
    candidate_set_key: str | None = None,
    candidate_id_key: str | None = None,
    score_key: str | None = None,
    require_success: bool = True,
) -> list[dict[str, Any]]:
    """Return one aggregate and deterministic per-scene ranking metric rows.

    ``top3_success_recall`` is the fraction of all successful candidates that
    appear in the top three ranked actions, averaged over scenes.  The adjacent
    ``top3_success_hit`` reports the complementary operational question: does
    the top three contain at least one success?  Scores within the documented
    tie tolerance use the uniform-within-tie expected value for all top-k
    metrics and regret; an ID merely names a deterministic display
    representative.  ``oracle_regret`` is the expected top-one candidate's
    utility shortfall relative to the designated or max-utility oracle
    candidate.
    """

    candidate_sets = build_candidate_sets(
        records,
        candidate_set_key=candidate_set_key,
        candidate_id_key=candidate_id_key,
        score_key=score_key,
        require_success=require_success,
    )
    context = _context(
        method_id=method_id,
        method_kind=method_kind,
        seed=seed,
        split=split,
        distribution=distribution,
        ood_axis=ood_axis,
        training_variant=training_variant,
        eval_perturbation=eval_perturbation,
    )
    scene_rows: list[dict[str, Any]] = []
    for candidate_set in candidate_sets:
        blocks = score_tie_blocks(candidate_set.candidates)
        ordered = tuple(candidate for block in blocks for candidate in block)
        top_block = blocks[0]
        display_selected = top_block[0]
        _, top1_hit, expected_top1_utility, _ = _tie_aware_top_k(
            blocks, k=1
        )
        top3_recall, top3_hit, _, top3_cutoff_tie_count = _tie_aware_top_k(blocks, k=3)
        success_count = sum(item.success for item in ordered)
        pairwise, comparisons = _pairwise_accuracy(ordered)
        auc = binary_ranking_auc(ordered)
        raw_regret = candidate_set.oracle_utility - expected_top1_utility
        row = {
            "row_type": "action_ranking_scene",
            "statistic": "tie_aware_expected",
            **context,
            "ranking_scene_id": candidate_set.scene_id,
            "candidate_count": len(ordered),
            "successful_candidate_count": success_count,
            # This ID is a deterministic panel/display representative only;
            # metrics below use uniform expectations across its whole score tie.
            "top1_candidate_id": display_selected.candidate_id,
            "top1_display_candidate_id": display_selected.candidate_id,
            "top1_display_success": float(display_selected.success),
            "top1_score": float(top_block[0].score),
            "top1_tie_count": len(top_block),
            "top1_tied_candidate_ids": [item.candidate_id for item in top_block],
            "top1_successful_action_accuracy": top1_hit,
            "top3_success_recall": top3_recall,
            "top3_success_hit": top3_hit,
            "top3_cutoff_tie_count": top3_cutoff_tie_count,
            "pairwise_ranking_accuracy": pairwise,
            "pairwise_ranking_comparisons": comparisons,
            "binary_ranking_auc": auc,
            "ranking_auc": auc,
            "oracle_candidate_id": candidate_set.oracle_candidate_id,
            "oracle_utility": candidate_set.oracle_utility,
            "selected_candidate_utility": expected_top1_utility,
            "expected_top1_candidate_utility": expected_top1_utility,
            "display_top1_candidate_utility": display_selected.oracle_utility,
            "oracle_regret": float(max(0.0, raw_regret)),
            "raw_oracle_utility_delta": float(raw_regret),
            "oracle_source": candidate_set.oracle_source,
            "score_reduction": score_key or display_selected.score_source,
            "tie_policy": RANKING_TIE_POLICY,
            "score_tie_tolerance": RANKING_SCORE_TIE_TOLERANCE,
        }
        scene_rows.append(row)

    # Macro averages preserve equal scene weight.  Global AUC is also emitted
    # so consumers that prefer a candidate-weighted statistic have it without
    # silently substituting one definition for the other.
    all_candidates = [candidate for group in candidate_sets for candidate in group.candidates]
    aggregate = {
        "row_type": "action_ranking_aggregate",
        "statistic": "tie_aware_expected",
        **context,
        "ranking_scene_id": "__all__",
        "ranking_scene_count": len(candidate_sets),
        "candidate_count": len(all_candidates),
        "successful_candidate_count": sum(item.success for item in all_candidates),
        "top1_successful_action_accuracy": _mean_or_none(
            row["top1_successful_action_accuracy"] for row in scene_rows
        ),
        "top3_success_recall": _mean_or_none(row["top3_success_recall"] for row in scene_rows),
        "top3_success_hit": _mean_or_none(row["top3_success_hit"] for row in scene_rows),
        "pairwise_ranking_accuracy": _mean_or_none(
            row["pairwise_ranking_accuracy"] for row in scene_rows
        ),
        "pairwise_ranking_comparisons": int(
            sum(int(row["pairwise_ranking_comparisons"]) for row in scene_rows)
        ),
        "binary_ranking_auc": _mean_or_none(row["binary_ranking_auc"] for row in scene_rows),
        "ranking_auc": _mean_or_none(row["binary_ranking_auc"] for row in scene_rows),
        "global_binary_ranking_auc": binary_ranking_auc(all_candidates),
        "oracle_regret": _mean_or_none(row["oracle_regret"] for row in scene_rows),
        "raw_oracle_utility_delta": _mean_or_none(
            row["raw_oracle_utility_delta"] for row in scene_rows
        ),
        "score_reduction": score_key or "record_score_or_mean_mask_probability",
        "tie_policy": RANKING_TIE_POLICY,
        "score_tie_tolerance": RANKING_SCORE_TIE_TOLERANCE,
    }
    return [aggregate, *scene_rows]


# More discoverable alias for experiment orchestration.
evaluate_action_ranking = action_ranking_records


__all__ = [
    "CandidateAction",
    "CandidateSet",
    "RANKING_SCORE_TIE_TOLERANCE",
    "RANKING_TIE_POLICY",
    "action_ranking_records",
    "binary_ranking_auc",
    "build_candidate_sets",
    "candidate_action_score",
    "evaluate_action_ranking",
    "record_value",
    "score_tie_blocks",
]
