"""CPU-only Milestone 2E bottleneck isolation for ActMask ranking.

This module deliberately owns its ranking semantics instead of reusing the
legacy candidate-ID tie-break in 2C.  All selection is validation-only; test
views are read-only diagnostics.
"""

from __future__ import annotations

import copy
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import Tensor
from torch.utils.data import Dataset

from actmask.data.milestone2c_dataset import HardCandidateActionDataset, RigidTransformDataset
from actmask.eval.calibration import apply_platt, fit_platt_scaling, reliability_metrics
from actmask.eval.pr_metrics import precision_recall_curve
from actmask.eval.statistics import paired_summary
from actmask.experiments.milestone2c import PROJECT_ROOT, _loader


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "actmask" / "milestone2e_cpu.yaml"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "actmask" / "milestone2e"
FROZEN_2C_DIR = PROJECT_ROOT / "outputs" / "actmask" / "milestone2c"
RANKING_METRICS = (
    "top1_success",
    "top3_success",
    "pairwise_ranking_accuracy",
    "ranking_ap",
    "ranking_roc_auc",
    "mrr",
    "ndcg",
    "normalized_regret",
    "failure_when_success_exists",
)

# Compatibility note: artifacts written before this schema was introduced
# labelled an oracle-candidate discounted-rank statistic as ``ndcg``.  That
# quantity is useful as a diagnostic but is not standard normalized DCG and
# must never be pooled with the v2 value below.  New reports carry this schema
# identifier so a consumer can distinguish the two definitions without
# rewriting historical 2E artifacts.
RANKING_METRIC_SCHEMA = "milestone2e-ranking-v2-standard-ndcg-utility-gain"
NDCG_DEFINITION = (
    "Expected standard DCG/IDCG using non-negative candidate_utility as the "
    "graded gain and uniform random resolution within equal-score blocks"
)


@dataclass(frozen=True)
class RankingBundle:
    """Candidate scores and physical labels in one stable sample ordering."""

    scores: np.ndarray
    success: np.ndarray
    utility: np.ndarray
    metadata: list[dict[str, Any]]


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Tensor):
        return value.item() if value.numel() == 1 else value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n")


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n")


def _load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("2E configuration must be a mapping")
    if value["experiment"]["device"] != "cpu":
        raise ValueError("Milestone 2E is CPU-only")
    return value


def _group_indices(metadata: Sequence[Mapping[str, Any]]) -> dict[int, list[int]]:
    groups: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(metadata):
        groups[int(row["candidate_set_id"])].append(index)
    return dict(sorted(groups.items()))


def _tie_blocks(scores: np.ndarray, indices: Sequence[int], tolerance: float) -> list[list[int]]:
    """Score-descending blocks, with no candidate-ID order in the ranking rule."""

    ordered = sorted(indices, key=lambda index: -float(scores[index]))
    blocks: list[list[int]] = []
    for index in ordered:
        if not blocks or abs(float(scores[index] - scores[blocks[-1][0]])) > tolerance:
            blocks.append([index])
        else:
            blocks[-1].append(index)
    return blocks


def _topk_success_probability(blocks: Sequence[Sequence[int]], success: np.ndarray, k: int) -> float:
    """Expected recall under uniform random resolution of boundary score ties."""

    remaining = int(k)
    for block in blocks:
        if remaining <= 0:
            break
        count = len(block)
        positives = int(np.count_nonzero(success[np.asarray(block)] >= 0.5))
        if count <= remaining:
            if positives:
                return 1.0
            remaining -= count
            continue
        if positives == 0:
            return 0.0
        if remaining >= count:
            return 1.0
        return float(1.0 - math.comb(count - positives, remaining) / math.comb(count, remaining))
    return 0.0


def _mean_tied_rank_metric(
    blocks: Sequence[Sequence[int]], utility: np.ndarray, *, reciprocal: bool
) -> float:
    # ``blocks`` contains one candidate world while ``utility`` may contain
    # every world in the evaluation.  The oracle rank is necessarily local to
    # the current world.
    local_indices = np.asarray([index for block in blocks for index in block], dtype=np.int64)
    best = float(np.max(utility[local_indices]))
    start = 1
    for block in blocks:
        member_utility = utility[np.asarray(block)]
        if np.any(np.isclose(member_utility, best, atol=1.0e-8, rtol=0.0)):
            ranks = np.arange(start, start + len(block), dtype=np.float64)
            return float(np.mean(1.0 / ranks if reciprocal else 1.0 / np.log2(ranks + 1.0)))
        start += len(block)
    raise AssertionError("best candidate is absent from score blocks")


def _expected_ndcg(blocks: Sequence[Sequence[int]], utility: np.ndarray) -> float:
    """Standard graded NDCG under the repository's tie policy.

    Candidate utility is a physical fraction in ``[0, 1]`` in the 2E data,
    so it is used directly as the non-negative graded relevance gain.  For a
    tied score block, each candidate is equally likely to occupy each rank in
    that block; its expected DCG contribution is therefore the block's mean
    gain times the sum of the block's rank discounts.  IDCG is deterministic
    because it ranks the same gains in descending order.
    """

    local_indices = np.asarray(
        [index for block in blocks for index in block], dtype=np.int64
    )
    gain = np.asarray(utility[local_indices], dtype=np.float64)
    if np.any(gain < -1.0e-12):
        raise ValueError("standard NDCG requires non-negative candidate utility")
    gain = np.clip(gain, 0.0, None)
    ranks = np.arange(1, gain.size + 1, dtype=np.float64)
    discounts = 1.0 / np.log2(ranks + 1.0)
    ideal = float(np.sum(np.sort(gain)[::-1] * discounts))
    # All-zero utility worlds carry no ranking preference.  Treating every
    # ordering as ideal avoids manufacturing a failure from an undefined 0/0.
    if ideal <= 1.0e-12:
        return 1.0
    expected_dcg = 0.0
    start = 0
    for block in blocks:
        count = len(block)
        block_gain = np.clip(
            np.asarray(utility[np.asarray(block)], dtype=np.float64), 0.0, None
        )
        expected_dcg += float(block_gain.mean() * discounts[start : start + count].sum())
        start += count
    return float(expected_dcg / ideal)


def _pairwise_accuracy(indices: Sequence[int], scores: np.ndarray, success: np.ndarray) -> float:
    positive = [index for index in indices if success[index] >= 0.5]
    negative = [index for index in indices if success[index] < 0.5]
    values: list[float] = []
    for left in positive:
        for right in negative:
            values.append(1.0 if scores[left] > scores[right] else 0.0 if scores[left] < scores[right] else 0.5)
    return float(np.mean(values)) if values else 0.5


def _average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    if not np.any(labels) or np.all(labels):
        return 1.0 if np.all(labels) else 0.0
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order].astype(np.float64)
    precision = np.cumsum(sorted_labels) / np.arange(1, sorted_labels.size + 1)
    return float(np.sum(precision * sorted_labels) / sorted_labels.sum())


def _roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    positive = scores[labels]
    negative = scores[~labels]
    if positive.size == 0 or negative.size == 0:
        return 0.5
    comparison = positive[:, None] - negative[None, :]
    return float(np.mean((comparison > 0.0).astype(np.float64) + 0.5 * (comparison == 0.0)))


def ranking_metrics_tie_aware(
    bundle: RankingBundle, *, score_name: str, tie_tolerance: float = 1.0e-10
) -> dict[str, Any]:
    """Scene-level ranking metrics with candidate-ID-independent tie handling."""

    scores = np.asarray(bundle.scores, dtype=np.float64).reshape(-1)
    success = np.asarray(bundle.success, dtype=np.float64).reshape(-1)
    utility = np.asarray(bundle.utility, dtype=np.float64).reshape(-1)
    if scores.shape != success.shape or scores.shape != utility.shape or len(bundle.metadata) != scores.size:
        raise ValueError("ranking bundle fields must share one candidate ordering")
    if not np.isfinite(scores).all() or not np.isfinite(success).all() or not np.isfinite(utility).all():
        raise ValueError("ranking bundle values must be finite")
    rows: list[dict[str, float | int | str]] = []
    for group_id, indices in _group_indices(bundle.metadata).items():
        blocks = _tie_blocks(scores, indices, tie_tolerance)
        top = blocks[0]
        top_utility = utility[np.asarray(top)]
        oracle = float(np.max(utility[indices]))
        selected = float(np.mean(top_utility))
        regret = float(oracle - selected)
        rows.append(
            {
                "group_id": group_id,
                "score_name": score_name,
                "top1_success": float(np.mean(success[np.asarray(top)] >= 0.5)),
                "top3_success": _topk_success_probability(blocks, success, 3),
                "pairwise_ranking_accuracy": _pairwise_accuracy(indices, scores, success),
                # Retain a scene-local AUC/AP as well as the pooled aggregate,
                # so transformed-vs-original confidence intervals are paired
                # by candidate world rather than by a global pooled scalar.
                "ranking_ap": _average_precision(
                    scores[np.asarray(indices)], success[np.asarray(indices)] >= 0.5
                ),
                "ranking_roc_auc": _roc_auc(
                    scores[np.asarray(indices)], success[np.asarray(indices)] >= 0.5
                ),
                "mrr": _mean_tied_rank_metric(blocks, utility, reciprocal=True),
                "ndcg": _expected_ndcg(blocks, utility),
                "regret": regret,
                "normalized_regret": float(regret / max(oracle, 1.0e-8)),
                "selected_utility": selected,
                "oracle_utility": oracle,
                "failure_when_success_exists": float(np.mean(success[np.asarray(top)] < 0.5)),
                "tie_fraction": float(sum(len(block) for block in blocks if len(block) > 1) / len(indices)),
            }
        )
    labels = success >= 0.5
    aggregate = {
        key: float(np.mean([float(row[key]) for row in rows]))
        for key in (
            "top1_success", "top3_success", "pairwise_ranking_accuracy", "mrr", "ndcg",
            "regret", "normalized_regret", "selected_utility", "oracle_utility",
            "failure_when_success_exists", "tie_fraction",
        )
    }
    aggregate.update(
        {
            "ranking_ap": _average_precision(scores, labels),
            "ranking_roc_auc": _roc_auc(scores, labels),
            "candidate_sets": len(rows),
            "per_group": rows,
            "score_name": score_name,
            "tie_policy": "uniform-within-equal-score-block; candidate_id never breaks a tie",
            "ranking_metric_schema": RANKING_METRIC_SCHEMA,
            "ndcg_definition": NDCG_DEFINITION,
        }
    )
    return aggregate


def paired_ranking_report(
    first: Mapping[str, Any], second: Mapping[str, Any], *, seed: int,
    resamples: int, permutations: int,
) -> dict[str, Any]:
    right = {int(row["group_id"]): row for row in second["per_group"]}
    report: dict[str, Any] = {}
    for offset, metric in enumerate(RANKING_METRICS):
        left_values = [float(row[metric]) for row in first["per_group"]]
        right_values = [float(right[int(row["group_id"])][metric]) for row in first["per_group"]]
        report[metric] = paired_summary(
            left_values, right_values, seed=seed + offset * 17,
            resamples=resamples, permutations=permutations,
        )
    return report


def validate_checkpoint_metadata(checkpoint: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    """Reject a stale/incompatible architecture checkpoint before loading it."""

    required = ("architecture", "correspondence", "backbone", "config_digest", "state_dict")
    missing = [key for key in required if key not in checkpoint]
    if missing:
        raise ValueError(f"checkpoint misses required provenance: {missing}")
    mismatch = {key: (checkpoint.get(key), expected.get(key)) for key in expected if checkpoint.get(key) != expected.get(key)}
    if mismatch:
        raise ValueError(f"checkpoint provenance mismatch: {mismatch}")


def _sample_signature(sample: Mapping[str, Any]) -> tuple[int, int, float, float]:
    metadata = sample["metadata"]
    targets = sample["targets"]
    return (
        int(metadata["candidate_set_id"]),
        int(metadata["candidate_id"]),
        float(targets["success"]),
        float(targets["candidate_utility"]),
    )


def _candidate_dataset(
    config: Mapping[str, Any], *, count: int, split: str, groups: int,
    master_seed_offset: int = 700,
) -> HardCandidateActionDataset:
    dataset = config["dataset"]
    common = {
        "num_points": int(dataset["num_points"]),
        "observation_frames": int(dataset["observation_frames"]),
        "future_steps": int(dataset["future_steps"]),
        "observation_noise_std": float(dataset["observation_noise_std"]),
        "point_dropout_rate": float(dataset["point_dropout_rate"]),
        "occlusion_rate": float(dataset["occlusion_rate"]),
    }
    return HardCandidateActionDataset(
        candidate_count=count,
        split=split,
        master_seed=int(dataset["evaluation_master_seed"]) + master_seed_offset + count,
        split_counts={"train": 1, "val": 1, "test": int(groups)},
        ood_groups_per_axis=1,
        **common,
    )


def _validate_candidate_world(
    members: Sequence[Mapping[str, Any]], *, count: int
) -> dict[str, float | int]:
    """Check physical candidate labels and their own future-mask target.

    This is deliberately evaluator-only: it reads hidden exact contact state
    solely to prove that each observable candidate's mask, utility and success
    label are aligned.  Fair models never receive these fields.
    """

    if len(members) != count:
        raise AssertionError("candidate world has the wrong number of members")
    ids = [int(item["metadata"]["candidate_id"]) for item in members]
    if len(set(ids)) != count or sorted(ids) != list(range(count)):
        raise AssertionError("candidate IDs are not a set-local permutation")
    group_ids = {int(item["metadata"]["candidate_set_id"]) for item in members}
    if len(group_ids) != 1:
        raise AssertionError("candidate world mixes candidate_set_id values")
    utilities: list[float] = []
    successes = 0
    for item in members:
        targets = item["targets"]
        hidden = item["hidden_state"]
        mask = targets["ground_truth_future_mask"] >= 0.5
        exact_mask = hidden["exact_contact_state"].bool()
        if not torch.equal(mask, exact_mask):
            raise AssertionError("future mask is not aligned with this candidate's exact contact state")
        target_object = targets["target_object_mask"].bool()
        target_count = max(int(target_object.sum()), 1)
        target_contacts = int(torch.logical_and(mask, target_object).sum())
        expected_utility = target_contacts / target_count
        expected_success = target_contacts >= max(2, int(math.ceil(0.20 * target_count)))
        if not math.isclose(float(targets["candidate_utility"]), expected_utility, abs_tol=1.0e-7):
            raise AssertionError("candidate utility is misaligned with its future mask")
        if bool(float(targets["success"]) >= 0.5) != expected_success:
            raise AssertionError("candidate success label is misaligned with its future mask")
        utilities.append(float(targets["candidate_utility"]))
        successes += int(expected_success)
    oracle = max(utilities)
    if any(
        not math.isclose(float(item["metadata"]["oracle_utility"]), oracle, abs_tol=1.0e-7)
        for item in members
    ):
        raise AssertionError("oracle utility is misaligned with candidate labels")
    if successes <= 0:
        raise AssertionError("candidate world has no successful candidate")
    return {
        "group_id": next(iter(group_ids)),
        "successful_candidates": successes,
        "oracle_utility": oracle,
        "mask_label_checks": count,
    }


def _audit_dataset_worlds(
    dataset: HardCandidateActionDataset, *, count: int
) -> dict[str, float | int]:
    """Audit every generated ranking world while releasing its large cache."""

    successes: list[int] = []
    mask_checks = 0
    for group_index in range(len(dataset.group_records)):
        start = group_index * count
        try:
            result = _validate_candidate_world(
                [dataset[start + offset] for offset in range(count)], count=count
            )
        finally:
            # HardCandidateActionDataset memoizes full point-cloud samples per
            # world.  This audit needs every world, but retaining C50 worlds
            # would needlessly keep all hidden trajectories in memory.
            dataset._cache.pop(group_index, None)
        successes.append(int(result["successful_candidates"]))
        mask_checks += int(result["mask_label_checks"])
    if not successes:
        raise AssertionError("candidate audit generated no worlds")
    return {
        "candidate_sets": len(successes),
        "successful_candidates_min": int(min(successes)),
        "successful_candidates_mean": float(np.mean(successes)),
        "ids_are_permuted_not_ordered": True,
        "mask_to_candidate_index_checks": int(mask_checks),
    }


def _analytic_so3_copy_audit(tolerance: float) -> dict[str, float | bool]:
    """Handcrafted scene/action copy whose invariant ranking is known exactly."""

    # A proper 90-degree SO(3) rotation; both action endpoints and scene point
    # are transformed, so a score based on distance to the action segment must
    # be unchanged.
    rotation = torch.tensor(
        ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)), dtype=torch.float64
    )
    point = torch.tensor((0.3, 0.1, -0.2), dtype=torch.float64)
    starts = torch.tensor(
        ((0.0, 0.1, -0.2), (-0.4, 0.5, 0.0), (0.8, -0.6, 0.3)), dtype=torch.float64
    )
    ends = torch.tensor(
        ((0.6, 0.1, -0.2), (-0.2, 0.5, 0.0), (1.0, -0.6, 0.3)), dtype=torch.float64
    )

    def segment_scores(scene_point: Tensor, begin: Tensor, end: Tensor) -> Tensor:
        direction = end - begin
        fraction = ((scene_point[None] - begin) * direction).sum(dim=-1) / direction.square().sum(dim=-1)
        closest = begin + fraction.clamp(0.0, 1.0)[:, None] * direction
        return -torch.linalg.vector_norm(scene_point[None] - closest, dim=-1)

    scores = segment_scores(point, starts, ends).numpy()
    rotated_scores = segment_scores(point @ rotation.T, starts @ rotation.T, ends @ rotation.T).numpy()
    utility = np.asarray((1.0, 0.4, 0.0), dtype=np.float64)
    success = np.asarray((1.0, 1.0, 0.0), dtype=np.float64)
    metadata = [
        {"candidate_set_id": 91, "candidate_id": candidate_id}
        for candidate_id in (17, 2, 44)
    ]
    original = ranking_metrics_tie_aware(
        RankingBundle(scores=scores, success=success, utility=utility, metadata=metadata),
        score_name="analytic_so3",
        tie_tolerance=tolerance,
    )
    transformed = ranking_metrics_tie_aware(
        RankingBundle(scores=rotated_scores, success=success, utility=utility, metadata=metadata),
        score_name="analytic_so3",
        tie_tolerance=tolerance,
    )
    invariant = all(
        math.isclose(float(original[name]), float(transformed[name]), abs_tol=1.0e-12)
        for name in RANKING_METRICS
    )
    if not invariant or not np.allclose(scores, rotated_scores, atol=1.0e-12, rtol=0.0):
        raise AssertionError("global SO(3) scene/action copy changed an analytically invariant ranking")
    return {
        "rotation_determinant": float(torch.linalg.det(rotation)),
        "max_score_difference": float(np.max(np.abs(scores - rotated_scores))),
        "ranking_metrics_identical": invariant,
    }


def audit_scoring_and_indexing(config: Mapping[str, Any]) -> dict[str, Any]:
    """Blocking 2E audit of score semantics and candidate-set integrity."""

    ranking = config["ranking"]
    tolerance = float(ranking["tie_tolerance"])
    availability: dict[str, Any] = {}
    transform_checks: list[dict[str, Any]] = []
    for count in ranking["candidate_counts"]:
        number = int(count)
        split_reports: dict[str, dict[str, float | int]] = {}
        for split_name, master_seed_offset, groups in (
            ("validation", 1000, int(ranking["validation_groups"])),
            ("test", 2000, int(ranking["test_groups"])),
        ):
            # These are the exact frozen ranking-world constructors used by
            # ``milestone2e_full.build_datasets``, not a small proxy sample.
            dataset = _candidate_dataset(
                config,
                count=number,
                split="test",
                groups=groups,
                master_seed_offset=master_seed_offset,
            )
            split_reports[split_name] = _audit_dataset_worlds(dataset, count=number)
        successes_min = min(
            int(report["successful_candidates_min"])
            for report in split_reports.values()
        )
        total_worlds = sum(int(report["candidate_sets"]) for report in split_reports.values())
        mask_checks = sum(
            int(report["mask_to_candidate_index_checks"])
            for report in split_reports.values()
        )
        availability[f"C{number}"] = {
            "candidate_sets": total_worlds,
            "successful_candidates_min": successes_min,
            "ids_are_permuted_not_ordered": True,
            "mask_to_candidate_index_checks": mask_checks,
            "scope": "every frozen validation and test ranking world",
            "splits": split_reports,
        }

        # Label preservation under dataset transforms is checked on a complete
        # frozen test world.  The separate analytic SO(3) case below verifies
        # that scene/action synchronization also preserves a known ranking.
        transform_source = _candidate_dataset(
            config,
            count=number,
            split="test",
            groups=1,
            master_seed_offset=2000,
        )
        source = [transform_source[index] for index in range(number)]
        for mode in ("random_so3", "axis_permutation", "sign_flip"):
            transformed = RigidTransformDataset(transform_source, mode=mode)
            for index, original in enumerate(source):
                changed = transformed[index]
                if _sample_signature(original) != _sample_signature(changed):
                    raise AssertionError(f"{mode} changed candidate labels or identifiers")
                if not torch.equal(
                    original["targets"]["ground_truth_future_mask"],
                    changed["targets"]["ground_truth_future_mask"],
                ):
                    raise AssertionError(f"{mode} changed a candidate future-mask label")
                if not torch.allclose(
                    torch.linalg.vector_norm(original["observable"]["action_command"][:3] - original["observable"]["action_command"][3:6]).reshape(1),
                    torch.linalg.vector_norm(changed["observable"]["action_command"][:3] - changed["observable"]["action_command"][3:6]).reshape(1),
                    atol=1.0e-6,
                ):
                    raise AssertionError(f"{mode} did not synchronise action geometry")
                original_distance = torch.linalg.vector_norm(
                    original["observable"]["points_history"][-1]
                    - original["observable"]["action_command"][:3],
                    dim=-1,
                )
                changed_distance = torch.linalg.vector_norm(
                    changed["observable"]["points_history"][-1]
                    - changed["observable"]["action_command"][:3],
                    dim=-1,
                )
                if not torch.allclose(original_distance, changed_distance, atol=2.0e-6, rtol=2.0e-6):
                    raise AssertionError(f"{mode} did not synchronise scene and action coordinates")
            transform_checks.append({"candidate_count": number, "mode": mode, "labels_preserved": True})

    def bundle(
        *, scores: Sequence[float], success: Sequence[float], utility: Sequence[float],
        candidate_ids: Sequence[int], group_id: int,
    ) -> RankingBundle:
        return RankingBundle(
            scores=np.asarray(scores, dtype=np.float64),
            success=np.asarray(success, dtype=np.float64),
            utility=np.asarray(utility, dtype=np.float64),
            metadata=[
                {"candidate_set_id": group_id, "candidate_id": candidate_id}
                for candidate_id in candidate_ids
            ],
        )

    # 1. Exactly one successful candidate: higher score must rank it first.
    one_success = ranking_metrics_tie_aware(
        bundle(
            scores=(0.95, 0.40, 0.10), success=(1.0, 0.0, 0.0),
            utility=(1.0, 0.0, 0.0), candidate_ids=(31, 7, 99), group_id=1,
        ),
        score_name="handcrafted_one_success",
        tie_tolerance=tolerance,
    )
    if not (
        math.isclose(float(one_success["top1_success"]), 1.0, abs_tol=1.0e-12)
        and math.isclose(float(one_success["normalized_regret"]), 0.0, abs_tol=1.0e-12)
    ):
        raise AssertionError("one-success score direction or candidate indexing is incorrect")

    # 2. Several successful candidates with genuinely different utility.
    multiple_success = ranking_metrics_tie_aware(
        bundle(
            scores=(0.20, 0.98, 0.55), success=(1.0, 1.0, 1.0),
            utility=(0.20, 1.00, 0.60), candidate_ids=(12, 44, 3), group_id=2,
        ),
        score_name="handcrafted_multiple_success",
        tie_tolerance=tolerance,
    )
    if not (
        math.isclose(float(multiple_success["normalized_regret"]), 0.0, abs_tol=1.0e-12)
        and math.isclose(float(multiple_success["ndcg"]), 1.0, abs_tol=1.0e-12)
    ):
        raise AssertionError("graded successful-candidate utility ranking is incorrect")

    # 3. A tied top block must be candidate-ID independent.
    tied = RankingBundle(
        scores=np.asarray([0.10, 0.95, 0.40, 0.95], dtype=np.float64),
        success=np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float64),
        utility=np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float64),
        metadata=[
            {"candidate_set_id": 1, "candidate_id": candidate_id}
            for candidate_id in (31, 7, 99, 4)
        ],
    )
    tied_result = ranking_metrics_tie_aware(tied, score_name="handcrafted_tie", tie_tolerance=tolerance)
    # Top score ties one success with one failure, hence expected top-1 0.5.
    if not math.isclose(float(tied_result["top1_success"]), 0.5, abs_tol=1.0e-12):
        raise AssertionError("tie-aware score direction or tie handling is incorrect")
    reordered = RankingBundle(
        scores=tied.scores[[3, 2, 1, 0]],
        success=tied.success[[3, 2, 1, 0]],
        utility=tied.utility[[3, 2, 1, 0]],
        metadata=[tied.metadata[index] for index in (3, 2, 1, 0)],
    )
    reordered_result = ranking_metrics_tie_aware(reordered, score_name="handcrafted_tie", tie_tolerance=tolerance)
    if any(
        not math.isclose(float(tied_result[name]), float(reordered_result[name]), abs_tol=1.0e-12)
        for name in ("top1_success", "top3_success", "normalized_regret", "ndcg")
    ):
        raise AssertionError("candidate ordering changes a tied ranking result")

    # 4. Reversing a temporal relation changes the analytically correct action
    # while the candidate IDs remain fixed.  This catches label/index swaps
    # that a static one-success toy would not expose.
    forward = ranking_metrics_tie_aware(
        bundle(
            scores=(0.90, 0.10), success=(1.0, 0.0), utility=(1.0, 0.0),
            candidate_ids=(73, 8), group_id=3,
        ),
        score_name="handcrafted_temporal_forward",
        tie_tolerance=tolerance,
    )
    reversed_time = ranking_metrics_tie_aware(
        bundle(
            scores=(0.10, 0.90), success=(0.0, 1.0), utility=(0.0, 1.0),
            candidate_ids=(73, 8), group_id=3,
        ),
        score_name="handcrafted_temporal_reversed",
        tie_tolerance=tolerance,
    )
    if not (
        math.isclose(float(forward["top1_success"]), 1.0, abs_tol=1.0e-12)
        and math.isclose(float(reversed_time["top1_success"]), 1.0, abs_tol=1.0e-12)
    ):
        raise AssertionError("temporal-order candidate labels are not aligned with their scores")

    # 5. A global SO(3) copy is analytically invariant when scene and action
    # are transformed together.
    analytic_so3 = _analytic_so3_copy_audit(tolerance)
    check = {"architecture": "unit", "correspondence": "unit", "backbone": "unit", "config_digest": "unit", "state_dict": {}}
    validate_checkpoint_metadata(check, {key: "unit" for key in ("architecture", "correspondence", "backbone", "config_digest")})
    try:
        validate_checkpoint_metadata(check, {"architecture": "changed", "correspondence": "unit", "backbone": "unit", "config_digest": "unit"})
    except ValueError:
        checkpoint_rejection = True
    else:
        checkpoint_rejection = False
        raise AssertionError("stale checkpoint was not rejected")

    return {
        "passed": True,
        "score_direction": "higher score ranks first",
        "tie_policy": tied_result["tie_policy"],
        "ranking_metric_schema": RANKING_METRIC_SCHEMA,
        "ndcg_definition": NDCG_DEFINITION,
        "legacy_ndcg_compatibility": (
            "Pre-v2 artifacts labelled a discounted oracle rank as ndcg; do not "
            "compare or aggregate those values with this standard NDCG schema."
        ),
        "handcrafted_top1_expected": tied_result["top1_success"],
        "handcrafted_top3_expected": tied_result["top3_success"],
        "handcrafted_cases": {
            "exactly_one_success": {
                "top1_success": one_success["top1_success"],
                "normalized_regret": one_success["normalized_regret"],
            },
            "multiple_successes_different_utility": {
                "top1_success": multiple_success["top1_success"],
                "normalized_regret": multiple_success["normalized_regret"],
                "ndcg": multiple_success["ndcg"],
            },
            "tied_candidates": {
                "top1_success": tied_result["top1_success"],
                "top3_success": tied_result["top3_success"],
                "ndcg": tied_result["ndcg"],
            },
            "temporal_order_changes_correct_candidate": {
                "forward_correct_candidate_id": 73,
                "reversed_correct_candidate_id": 8,
                "forward_top1_success": forward["top1_success"],
                "reversed_top1_success": reversed_time["top1_success"],
            },
            "global_so3_copy": analytic_so3,
        },
        "candidate_availability": availability,
        "mask_to_candidate_indexing": {
            "checked_candidate_masks": int(
                sum(int(value["mask_to_candidate_index_checks"]) for value in availability.values())
            ),
            "policy": "Every frozen validation/test candidate mask is checked against its own exact contact state, utility, and success label.",
        },
        "transform_checks": transform_checks,
        "checkpoint_mismatch_rejected": checkpoint_rejection,
        "calibration_policy": "Platt scaling is fit only on validation candidate scores",
        "stale_checkpoint_policy": "2E trains from scratch; incompatible provenance is rejected",
    }


def run_scoring_audit(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = _load_config(config_path)
    report = audit_scoring_and_indexing(config)
    _write_json(OUTPUT_DIR / "scoring_indexing_audit.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    del argv
    report = run_scoring_audit()
    print("SCORING_AUDIT_PASS" if report["passed"] else "SCORING_AUDIT_FAIL")
    return report


if __name__ == "__main__":
    main()
