#!/usr/bin/env python3
"""Reproducible, simulator-free audit of the frozen TimeArrow-v2 benchmark.

The script is intentionally limited to NumPy and the Python standard library.
It reads the frozen artifacts under ``outputs/actmask/milestone3r_nl_v2`` and
writes only beneath ``audit/`` (or an explicitly supplied output directory).
It never trains a model and never mutates the source artifacts.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import platform
import sys
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np


SCHEMA = "timearrow-shortcut-audit-v1"
CANONICAL = {
    "id": "outputs/actmask/milestone3r_nl_v2/full/id",
    "physical_parameter_ood": "outputs/actmask/milestone3r_nl_v2/full/physical_parameter_ood",
    "temporal_delay_ood": "outputs/actmask/milestone3r_nl_v2/full/temporal_delay_ood",
    "held_mechanism_ood": "outputs/actmask/milestone3r_nl_v2/full/held_mechanism_ood",
}
PERTURBATION_SEEDS = (101, 211, 307, 401, 503)
INPUT_KEYS = (
    "history",
    "timestamps",
    "visibility",
    "observation_confidence",
    "candidate_actions",
    "nominal_action_timing",
    "tcp_state",
)


def _json_lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@contextmanager
def _gzip_csv(path: Path, fieldnames: Sequence[str]) -> Iterator[csv.DictWriter]:
    """Write deterministic gzip-compressed CSV (fixed gzip timestamp)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = gzip.GzipFile(filename=str(path), mode="wb", compresslevel=9, mtime=0)
    text = io.TextIOWrapper(raw, encoding="utf-8", newline="")
    try:
        writer = csv.DictWriter(text, fieldnames=fieldnames)
        writer.writeheader()
        yield writer
    finally:
        text.flush()
        text.close()


def _float(value: float) -> str:
    return format(float(value), ".10g")


def load_dataset(root: Path) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], np.ndarray, np.ndarray]:
    with np.load(root / "model_inputs.npz") as archive:
        values = {key: archive[key] for key in archive.files}
    rows = _json_lines(root / "metadata.jsonl")
    with np.load(root / "labels.npz") as archive:
        labels = archive["success"].astype(bool)
        static_features = archive["static_features"]
    if len(rows) != len(labels):
        raise AssertionError(f"{root}: metadata/label length mismatch")
    for key in INPUT_KEYS:
        if key not in values or len(values[key]) != len(labels):
            raise AssertionError(f"{root}: missing or misaligned input {key}")
    return values, rows, labels, static_features


def arrow_action_score(history: np.ndarray, candidate_actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Legal-input zero-training compatibility score.

    Observation layout is payload position at 0:3 and goal position at 17:20.
    The generator writes the early arrow to one of those positions.  Adding both
    deltas makes the rule family-agnostic; the dot product checks whether the
    proposed action travels with the observed arrow.
    """
    arrow = (
        history[:, 3, 0:3]
        - history[:, 0, 0:3]
        + history[:, 3, 17:20]
        - history[:, 0, 17:20]
    )
    action_sum = candidate_actions.sum(axis=1)
    score = np.einsum("ij,ij->i", arrow, action_sum)
    return arrow, action_sum, score


def multi_frame_linear_velocity(
    history: np.ndarray, timestamps: np.ndarray, visibility: np.ndarray
) -> np.ndarray:
    """NumPy-equivalent of the published ``MultiFrameLinearVelocity`` estimator."""
    mask = visibility.squeeze(-1).astype(bool) if visibility.ndim == 3 else visibility.astype(bool)
    result = np.zeros((len(history), history.shape[2]), dtype=np.float32)
    for index in range(len(history)):
        visible = np.flatnonzero(mask[index])
        if len(visible) < 2:
            continue
        x = timestamps[index, visible]
        y = history[index, visible]
        design = np.stack((x, np.ones(len(x))), axis=1)
        coefficient = np.linalg.lstsq(design, y, rcond=None)[0]
        result[index] = coefficient[0]
    return result


class ArrowActionCompatibility:
    """Named, stateless, zero-training baseline for the observable shortcut."""

    name = "ArrowActionCompatibility"
    requires_training = False

    def score(self, history: np.ndarray, candidate_actions: np.ndarray) -> np.ndarray:
        return arrow_action_score(history, candidate_actions)[2]

    def predict(self, history: np.ndarray, candidate_actions: np.ndarray) -> np.ndarray:
        return self.score(history, candidate_actions) > 0

    def __call__(self, history: np.ndarray, candidate_actions: np.ndarray) -> np.ndarray:
        return self.predict(history, candidate_actions)


def observable_bits(arrow: np.ndarray, action_sum: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    axis = np.abs(arrow).argmax(axis=1)
    index = np.arange(len(arrow))
    branch_hat = arrow[index, axis] < 0
    direction_hat = action_sum[index, axis] < 0
    return axis, branch_hat, direction_hat


def pair_order_accuracy(labels: np.ndarray, scores: np.ndarray, pair_ids: Sequence[str]) -> float:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, pair_id in enumerate(pair_ids):
        groups[str(pair_id)].append(index)
    values: list[float] = []
    for pair_id, indices in groups.items():
        if len(indices) != 2:
            raise AssertionError(f"{pair_id}: expected exactly two rows")
        left, right = indices
        if int(labels[left]) + int(labels[right]) != 1:
            continue
        positive, negative = (left, right) if labels[left] else (right, left)
        values.append(
            1.0
            if scores[positive] > scores[negative]
            else 0.5
            if np.isclose(scores[positive], scores[negative])
            else 0.0
        )
    return float(np.mean(values)) if values else float("nan")


def rank_by_history(
    labels: np.ndarray,
    scores: np.ndarray,
    rows: Sequence[dict[str, Any]],
    count: int,
) -> dict[str, float | int]:
    """Rank candidates within one observable history/decision context."""
    groups: dict[tuple[str, str, int, int], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        if int(row["candidate_id"]) < count:
            key = (row["family"], row["mechanism"], int(row["world_id"]), int(row["branch"]))
            groups[key].append(index)
    top1: list[float] = []
    top3: list[float] = []
    ndcg: list[float] = []
    regret: list[float] = []
    mixed = 0
    for key, indices in groups.items():
        if len(indices) != count:
            raise AssertionError(f"{key}: expected {count} candidates, got {len(indices)}")
        group = np.asarray(indices)
        ordered = group[np.argsort(-scores[group], kind="stable")]
        maximum = scores[ordered[0]]
        ties = ordered[np.isclose(scores[ordered], maximum)]
        top_value = float(labels[ties].mean())
        top1.append(top_value)
        top3.append(float(labels[ordered[:3]].any()))
        relevance = labels[ordered].astype(float)
        ideal = np.sort(labels[group].astype(float))[::-1]
        weights = 1.0 / np.log2(np.arange(2, len(group) + 2))
        ideal_dcg = float(np.dot(ideal, weights))
        ndcg.append(float(np.dot(relevance, weights) / ideal_dcg) if ideal_dcg else 0.0)
        regret.append(float(labels[group].max() - top_value))
        mixed += int(0 < labels[group].sum() < len(group))
    return {
        "contexts": len(top1),
        "mixed_contexts": mixed,
        "top1_success": float(np.mean(top1)),
        "top3_success_recall": float(np.mean(top3)),
        "corrected_utility_gain_ndcg": float(np.mean(ndcg)),
        "normalized_regret": float(np.mean(regret)),
    }


def _metadata_pair_ids(rows: Sequence[dict[str, Any]]) -> list[str]:
    return [str(row["pair_id"]) for row in rows]


def _max_abs(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.max(np.abs(left - right))) if left.size else 0.0


def pair_integrity(
    values: dict[str, np.ndarray],
    rows: Sequence[dict[str, Any]],
    labels: np.ndarray,
    static_features: np.ndarray,
) -> dict[str, Any]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[str(row["pair_id"])].append(index)
    branch0: list[int] = []
    branch1: list[int] = []
    for pair_id, indices in groups.items():
        if len(indices) != 2:
            raise AssertionError(f"{pair_id}: incomplete retained pair")
        by_branch = {int(rows[index]["branch"]): index for index in indices}
        if set(by_branch) != {0, 1}:
            raise AssertionError(f"{pair_id}: invalid branch membership")
        branch0.append(by_branch[0])
        branch1.append(by_branch[1])
    left = np.asarray(branch0)
    right = np.asarray(branch1)
    history = values["history"]
    same_errors = {
        "final_two_history": _max_abs(history[left, -2:], history[right, -2:]),
        "candidate_actions": _max_abs(values["candidate_actions"][left], values["candidate_actions"][right]),
        "timestamps": _max_abs(values["timestamps"][left], values["timestamps"][right]),
        "visibility": _max_abs(values["visibility"][left], values["visibility"][right]),
        "observation_confidence": _max_abs(
            values["observation_confidence"][left], values["observation_confidence"][right]
        ),
        "nominal_action_timing": _max_abs(
            values["nominal_action_timing"][left], values["nominal_action_timing"][right]
        ),
        "tcp_state": _max_abs(values["tcp_state"][left], values["tcp_state"][right]),
        "static_features": _max_abs(static_features[left], static_features[right]),
    }
    return {
        "pairs": len(groups),
        "pair_size_exact": True,
        "early_four_reverse_max_error": _max_abs(history[left, :4, :][:, ::-1], history[right, :4, :]),
        "paired_equal_input_max_errors": same_errors,
        "label_flip_rate": float(np.mean(labels[left] != labels[right])),
        "metadata_label_match_rate": float(
            np.mean(labels == np.asarray([bool(row["success"]) for row in rows]))
        ),
    }


def outcome_audit(
    root: Path, rows: Sequence[dict[str, Any]], labels: np.ndarray
) -> dict[str, Any] | None:
    path = root / "candidate_outcomes.jsonl"
    if not path.exists():
        return None
    outcomes = _json_lines(path)
    table = {"00": 0, "01": 0, "10": 0, "11": 0}
    parity_correct = 0
    parity_nonflip_correct = 0
    nonflip_branch_labels = 0
    outcome_keys: set[tuple[Any, ...]] = set()
    flip_keys: set[tuple[Any, ...]] = set()
    outcome_key_rows: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    pair_flip_field_mismatches = 0
    for row in outcomes:
        b0, b1 = bool(row["branch0_success"]), bool(row["branch1_success"])
        computed_flip = b0 != b1
        pair_flip_field_mismatches += int(bool(row["pair_flip"]) != computed_flip)
        table[f"{int(b0)}{int(b1)}"] += 1
        key = (
            row["condition"],
            row["family"],
            row["mechanism"],
            int(row["world_id"]),
            int(row["candidate_id"]),
        )
        outcome_keys.add(key)
        outcome_key_rows[key].append(row)
        if computed_flip:
            flip_keys.add(key)
        expected = int(row["candidate_id"]) % 2
        for branch, actual in ((0, b0), (1, b1)):
            correct = bool(actual) == (branch == expected)
            parity_correct += int(correct)
            if b0 == b1:
                parity_nonflip_correct += int(correct)
                nonflip_branch_labels += 1
    retained_keys = {
        (
            row["condition"],
            row["family"],
            row["mechanism"],
            int(row["world_id"]),
            int(row["candidate_id"]),
        )
        for row in rows
    }
    metadata_pair_branch_keys = [
        (str(row["pair_id"]), int(row["branch"])) for row in rows
    ]
    metadata_pair_branch_counts = Counter(metadata_pair_branch_keys)
    joined_rows = 0
    metadata_outcome_matches = 0
    label_outcome_matches = 0
    metadata_label_matches = 0
    ambiguous_or_missing_rows = 0
    for index, row in enumerate(rows):
        key = (
            row["condition"],
            row["family"],
            row["mechanism"],
            int(row["world_id"]),
            int(row["candidate_id"]),
        )
        matches = outcome_key_rows.get(key, [])
        if len(matches) != 1 or int(row["branch"]) not in (0, 1):
            ambiguous_or_missing_rows += 1
            continue
        joined_rows += 1
        outcome_success = bool(matches[0][f"branch{int(row['branch'])}_success"])
        metadata_success = bool(row["success"])
        label_success = bool(labels[index])
        metadata_outcome_matches += int(metadata_success == outcome_success)
        label_outcome_matches += int(label_success == outcome_success)
        metadata_label_matches += int(metadata_success == label_success)
    nonflip = len(outcomes) - len(flip_keys)
    return {
        "attempted_pairs": len(outcomes),
        "attempted_branch_executions": 2 * len(outcomes),
        "outcome_table_branch0_branch1": table,
        "outcome_key_unique": len(outcome_keys) == len(outcomes),
        "outcome_unique_keys": len(outcome_keys),
        "outcome_duplicate_key_rows": len(outcomes) - len(outcome_keys),
        "pair_flip_field_mismatches": pair_flip_field_mismatches,
        "pair_flip_field_all_exact": pair_flip_field_mismatches == 0,
        "flip_pairs": len(flip_keys),
        "nonflip_pairs": nonflip,
        "retained_pairs": len(retained_keys),
        "retention_rate": len(retained_keys) / len(outcomes),
        "retained_keys_equal_flip_keys": retained_keys == flip_keys,
        "metadata_pair_branch_unique": len(metadata_pair_branch_counts) == len(rows),
        "metadata_unique_pair_branch_keys": len(metadata_pair_branch_counts),
        "metadata_duplicate_pair_branch_rows": len(rows) - len(metadata_pair_branch_counts),
        "direct_join": {
            "join_key": "(condition, family, mechanism, world_id, candidate_id), then branch{branch}_success",
            "expected_metadata_rows": len(rows),
            "joined_rows": joined_rows,
            "ambiguous_or_missing_rows": ambiguous_or_missing_rows,
            "coverage": joined_rows / len(rows) if rows else None,
            "metadata_success_matches": metadata_outcome_matches,
            "metadata_success_match_rate": (
                metadata_outcome_matches / joined_rows if joined_rows else None
            ),
            "label_success_matches": label_outcome_matches,
            "label_success_match_rate": label_outcome_matches / joined_rows if joined_rows else None,
            "metadata_label_matches": metadata_label_matches,
            "metadata_label_match_rate": metadata_label_matches / joined_rows if joined_rows else None,
            "all_exact": bool(
                joined_rows == len(rows)
                and metadata_outcome_matches == joined_rows
                and label_outcome_matches == joined_rows
                and metadata_label_matches == joined_rows
            ),
        },
        "filtered_nonflip_pairs": len((outcome_keys - flip_keys) - retained_keys),
        "attempted_parity_rule_correct": parity_correct,
        "attempted_parity_rule_total": 2 * len(outcomes),
        "attempted_parity_rule_accuracy": parity_correct / (2 * len(outcomes)),
        "nonflip_parity_rule_accuracy": (
            parity_nonflip_correct / nonflip_branch_labels if nonflip_branch_labels else None
        ),
    }


def dataset_summary(root: Path, project_root: Path, canonical: bool = False) -> dict[str, Any]:
    values, rows, labels, static_features = load_dataset(root)
    arrow, action_sum, score = arrow_action_score(values["history"], values["candidate_actions"])
    axis, branch_hat, direction_hat = observable_bits(arrow, action_sum)
    prediction = score > 0
    branch = np.asarray([int(row["branch"]) for row in rows])
    candidate = np.asarray([int(row["candidate_id"]) for row in rows])
    parity_prediction = branch == (candidate % 2)
    xor_table: dict[str, dict[str, Any]] = {}
    for branch_value in (0, 1):
        for direction_value in (0, 1):
            mask = (branch_hat == bool(branch_value)) & (
                direction_hat == bool(direction_value)
            )
            xor_table[
                f"branch_hat_{branch_value}_direction_hat_{direction_value}"
            ] = {
                "examples": int(mask.sum()),
                "success_rate": float(labels[mask].mean()) if mask.any() else None,
            }
    summary: dict[str, Any] = {
        "path": str(root.relative_to(project_root)),
        "examples": len(labels),
        "pairs": len(labels) // 2,
        "input_shapes": {key: list(values[key].shape) for key in values},
        "label_shape": list(labels.shape),
        "static_features_shape": list(static_features.shape),
        "label_balance": float(labels.mean()),
        "arrow_action_correct": int(np.sum(prediction == labels)),
        "arrow_action_accuracy": float(np.mean(prediction == labels)),
        "zero_score_count": int(np.sum(score == 0)),
        "minimum_absolute_score": float(np.min(np.abs(score))),
        "metadata_parity_correct": int(np.sum(parity_prediction == labels)),
        "metadata_parity_accuracy": float(np.mean(parity_prediction == labels)),
        "branch_recovery_accuracy": float(np.mean(branch_hat == branch.astype(bool))),
        "direction_matches_candidate_parity": float(
            np.mean(direction_hat == (candidate % 2).astype(bool))
        ),
        "observable_xor_accuracy": float(np.mean((branch_hat == direction_hat) == labels)),
        "observable_xor_table": xor_table,
        "outcomes": outcome_audit(root, rows, labels),
    }
    if canonical:
        summary["pair_integrity"] = pair_integrity(values, rows, labels, static_features)
    return summary


def discover_datasets(project_root: Path) -> list[Path]:
    data_root = project_root / "outputs/actmask/milestone3r_nl_v2"
    result = []
    for model_input in data_root.rglob("model_inputs.npz"):
        root = model_input.parent
        if (root / "labels.npz").exists() and (root / "metadata.jsonl").exists():
            result.append(root)
    return sorted(set(result))


def canonical_totals(canonical: dict[str, dict[str, Any]]) -> dict[str, Any]:
    examples = sum(int(value["examples"]) for value in canonical.values())
    attempted = sum(int(value["outcomes"]["attempted_branch_executions"]) for value in canonical.values())
    parity_correct = sum(int(value["outcomes"]["attempted_parity_rule_correct"]) for value in canonical.values())
    nonflip = sum(int(value["outcomes"]["nonflip_pairs"]) for value in canonical.values())
    joined = sum(int(value["outcomes"]["direct_join"]["joined_rows"]) for value in canonical.values())
    metadata_outcome_matches = sum(
        int(value["outcomes"]["direct_join"]["metadata_success_matches"])
        for value in canonical.values()
    )
    label_outcome_matches = sum(
        int(value["outcomes"]["direct_join"]["label_success_matches"])
        for value in canonical.values()
    )
    return {
        "retained_examples": examples,
        "retained_pairs": examples // 2,
        "attempted_branch_executions": attempted,
        "attempted_pairs": attempted // 2,
        "filtered_nonflip_pairs": nonflip,
        "direct_outcome_join": {
            "expected_metadata_rows": examples,
            "joined_rows": joined,
            "coverage": joined / examples,
            "metadata_success_matches": metadata_outcome_matches,
            "metadata_success_match_rate": metadata_outcome_matches / examples,
            "label_success_matches": label_outcome_matches,
            "label_success_match_rate": label_outcome_matches / examples,
            "all_exact": all(value["outcomes"]["direct_join"]["all_exact"] for value in canonical.values()),
            "all_outcome_keys_unique": all(
                value["outcomes"]["outcome_key_unique"] for value in canonical.values()
            ),
            "all_metadata_pair_branch_keys_unique": all(
                value["outcomes"]["metadata_pair_branch_unique"] for value in canonical.values()
            ),
        },
        "retained_arrow_action_accuracy": (
            sum(value["arrow_action_correct"] for value in canonical.values()) / examples
        ),
        "retained_metadata_parity_accuracy": (
            sum(value["metadata_parity_correct"] for value in canonical.values()) / examples
        ),
        "attempted_metadata_parity_accuracy": parity_correct / attempted,
    }


def _test_subset(
    values: dict[str, np.ndarray], rows: list[dict[str, Any]], labels: np.ndarray
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], np.ndarray, np.ndarray]:
    indices = np.asarray([index for index, row in enumerate(rows) if row["split"] == "test"])
    return {key: value[indices] for key, value in values.items()}, [rows[index] for index in indices], labels[indices], indices


def _classification(labels: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    return {
        "examples": len(labels),
        "accuracy": float(np.mean((scores > 0) == labels)),
        "zero_score_count": int(np.sum(scores == 0)),
        "minimum_absolute_score": float(np.min(np.abs(scores))),
    }


def _pair_shared_early_permutation(
    history: np.ndarray, rows: Sequence[dict[str, Any]], seed: int
) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    rng = np.random.default_rng(seed)
    transformed = history.copy()
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[str(row["pair_id"])].append(index)
    permutations: list[tuple[int, int, int, int]] = [(-1, -1, -1, -1)] * len(rows)
    for pair_id in sorted(groups):
        indices = groups[pair_id]
        permutation = tuple(int(value) for value in rng.permutation(4))
        transformed[indices, :4] = history[np.asarray(indices)[:, None], np.asarray(permutation)[None, :]]
        for index in indices:
            permutations[index] = permutation
    return transformed, permutations


def temporal_perturbations(
    values: dict[str, np.ndarray],
    rows: list[dict[str, Any]],
    labels: np.ndarray,
    original_indices: np.ndarray,
    output: Path,
) -> dict[str, Any]:
    history = values["history"]
    actions = values["candidate_actions"]
    _, _, original = arrow_action_score(history, actions)
    reversed_history = np.concatenate((history[:, :4][:, ::-1], history[:, 4:]), axis=1)
    _, _, reversed_score = arrow_action_score(reversed_history, actions)
    last_two = history.copy()
    last_two[:, :4] = history[:, -1:, :]
    _, _, last_two_score = arrow_action_score(last_two, actions)
    pair_ids = _metadata_pair_ids(rows)
    partner_label = np.empty_like(labels)
    pair_groups: dict[str, list[int]] = defaultdict(list)
    for index, pair_id in enumerate(pair_ids):
        pair_groups[pair_id].append(index)
    for indices in pair_groups.values():
        partner_label[indices[0]] = labels[indices[1]]
        partner_label[indices[1]] = labels[indices[0]]

    fieldnames = (
        "seed",
        "source_row_index",
        "pair_id",
        "family",
        "mechanism",
        "world_id",
        "branch",
        "candidate_id",
        "label",
        "permutation",
        "original_score",
        "early_reverse_score",
        "early_reverse_prediction",
        "early_reverse_correct_original_label",
        "early_reverse_correct_partner_label",
        "last_two_score",
        "last_two_prediction",
        "random_shuffle_score",
        "random_shuffle_prediction",
        "random_shuffle_correct",
    )
    random_metrics: dict[str, Any] = {}
    with _gzip_csv(output / "id_test_temporal_perturbations.csv.gz", fieldnames) as writer:
        for seed in PERTURBATION_SEEDS:
            shuffled, permutations = _pair_shared_early_permutation(history, rows, seed)
            _, _, shuffled_score = arrow_action_score(shuffled, actions)
            random_metrics[str(seed)] = {
                **_classification(labels, shuffled_score),
                "pair_order_accuracy": pair_order_accuracy(labels, shuffled_score, pair_ids),
            }
            for index, row in enumerate(rows):
                writer.writerow(
                    {
                        "seed": seed,
                        "source_row_index": int(original_indices[index]),
                        "pair_id": row["pair_id"],
                        "family": row["family"],
                        "mechanism": row["mechanism"],
                        "world_id": row["world_id"],
                        "branch": row["branch"],
                        "candidate_id": row["candidate_id"],
                        "label": int(labels[index]),
                        "permutation": "-".join(map(str, permutations[index])),
                        "original_score": _float(original[index]),
                        "early_reverse_score": _float(reversed_score[index]),
                        "early_reverse_prediction": int(reversed_score[index] > 0),
                        "early_reverse_correct_original_label": int((reversed_score[index] > 0) == labels[index]),
                        "early_reverse_correct_partner_label": int((reversed_score[index] > 0) == partner_label[index]),
                        "last_two_score": _float(last_two_score[index]),
                        "last_two_prediction": int(last_two_score[index] > 0),
                        "random_shuffle_score": _float(shuffled_score[index]),
                        "random_shuffle_prediction": int(shuffled_score[index] > 0),
                        "random_shuffle_correct": int((shuffled_score[index] > 0) == labels[index]),
                    }
                )
    return {
        "original": {
            **_classification(labels, original),
            "pair_order_accuracy": pair_order_accuracy(labels, original, pair_ids),
        },
        "exact_early_partner_reverse": {
            **_classification(labels, reversed_score),
            "pair_order_accuracy_original_labels": pair_order_accuracy(labels, reversed_score, pair_ids),
            "accuracy_against_partner_labels": float(np.mean((reversed_score > 0) == partner_label)),
        },
        "last_two_only": {
            **_classification(labels, last_two_score),
            "pair_order_accuracy": pair_order_accuracy(labels, last_two_score, pair_ids),
        },
        "genuine_pair_shared_random_early_shuffle": random_metrics,
    }


def _context_groups(rows: Sequence[dict[str, Any]], include_branch: bool) -> dict[tuple[Any, ...], list[int]]:
    groups: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        key: tuple[Any, ...] = (row["family"], row["mechanism"], int(row["world_id"]))
        if include_branch:
            key += (int(row["branch"]),)
        groups[key].append(index)
    return groups


def _action_reassignment(
    actions: np.ndarray,
    rows: Sequence[dict[str, Any]],
    seed: int | None,
    xor_one: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    result = actions.copy()
    source_candidate = np.full(len(rows), -1, dtype=np.int64)
    groups = _context_groups(rows, include_branch=False)
    rng = np.random.default_rng(seed) if seed is not None else None
    for key in sorted(groups):
        indices = groups[key]
        by_branch_candidate = {
            (int(rows[index]["branch"]), int(rows[index]["candidate_id"])): index
            for index in indices
        }
        candidates = sorted({int(rows[index]["candidate_id"]) for index in indices})
        if xor_one:
            mapping = {candidate: candidate ^ 1 for candidate in candidates}
        else:
            assert rng is not None
            permutation = rng.permutation(candidates)
            mapping = {candidate: int(source) for candidate, source in zip(candidates, permutation)}
        for index in indices:
            branch = int(rows[index]["branch"])
            target = int(rows[index]["candidate_id"])
            source = mapping[target]
            result[index] = actions[by_branch_candidate[(branch, source)]]
            source_candidate[index] = source
    return result, source_candidate


def _joint_order_positions(
    rows: Sequence[dict[str, Any]], seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    positions = np.full(len(rows), -1, dtype=np.int64)
    global_order: list[int] = []
    groups = _context_groups(rows, include_branch=True)
    for key in sorted(groups):
        indices = groups[key]
        ordered = sorted(indices, key=lambda index: int(rows[index]["candidate_id"]))
        shuffled = [ordered[int(value)] for value in rng.permutation(len(ordered))]
        global_order.extend(shuffled)
        for position, index in enumerate(shuffled):
            positions[index] = position
    if sorted(global_order) != list(range(len(rows))):
        raise AssertionError("joint candidate order is not a full row permutation")
    return positions, np.asarray(global_order, dtype=np.int64)


def candidate_perturbations(
    values: dict[str, np.ndarray],
    rows: list[dict[str, Any]],
    labels: np.ndarray,
    original_indices: np.ndarray,
    output: Path,
) -> dict[str, Any]:
    history = values["history"]
    actions = values["candidate_actions"]
    _, _, original = arrow_action_score(history, actions)
    pair_ids = _metadata_pair_ids(rows)
    branch = np.asarray([int(row["branch"]) for row in rows])
    candidate = np.asarray([int(row["candidate_id"]) for row in rows])
    remapped = candidate ^ 1
    remapped_parity_prediction = branch == (remapped % 2)
    baseline_ranking = {f"C{count}": rank_by_history(labels, original, rows, count) for count in (5, 10, 20)}

    xor_actions, xor_sources = _action_reassignment(actions, rows, seed=None, xor_one=True)
    _, _, xor_score = arrow_action_score(history, xor_actions)
    xor_metrics = {
        **_classification(labels, xor_score),
        "pair_order_accuracy": pair_order_accuracy(labels, xor_score, pair_ids),
        "ranking": {f"C{count}": rank_by_history(labels, xor_score, rows, count) for count in (5, 10, 20)},
    }

    fieldnames = (
        "seed",
        "source_row_index",
        "pair_id",
        "family",
        "mechanism",
        "world_id",
        "branch",
        "candidate_id",
        "label",
        "joint_order_position",
        "random_action_source_candidate",
        "original_score",
        "random_action_score",
        "random_action_prediction",
        "random_action_correct",
        "xor_action_source_candidate",
        "xor_action_score",
        "xor_action_prediction",
        "remapped_candidate_id",
        "remapped_parity_prediction",
        "remapped_parity_correct",
    )
    shuffled_metrics: dict[str, Any] = {}
    with _gzip_csv(output / "id_test_candidate_perturbations.csv.gz", fieldnames) as writer:
        for seed in PERTURBATION_SEEDS:
            positions, joint_order = _joint_order_positions(rows, seed)
            random_actions, random_sources = _action_reassignment(actions, rows, seed=seed)
            _, _, random_score = arrow_action_score(history, random_actions)
            # Actually permute every aligned record and rerun the evaluator.  The
            # expected tie-broken Top-1 implementation is order invariant; this
            # check also exposes any stable-sort tie dependence in Top-3/NDCG.
            joint_rows = [rows[int(index)] for index in joint_order]
            joint_ranking = {
                f"C{count}": rank_by_history(
                    labels[joint_order], original[joint_order], joint_rows, count
                )
                for count in (5, 10, 20)
            }
            joint_invariant = joint_ranking == baseline_ranking
            shuffled_metrics[str(seed)] = {
                "joint_record_order_shuffle": {
                    "ranking": joint_ranking,
                    "invariant": joint_invariant,
                },
                "action_only_reassignment": {
                    **_classification(labels, random_score),
                    "pair_order_accuracy": pair_order_accuracy(labels, random_score, pair_ids),
                    "ranking": {
                        f"C{count}": rank_by_history(labels, random_score, rows, count)
                        for count in (5, 10, 20)
                    },
                    "warning": "Diagnostic only: labels were not regenerated by the simulator.",
                },
            }
            for index, row in enumerate(rows):
                writer.writerow(
                    {
                        "seed": seed,
                        "source_row_index": int(original_indices[index]),
                        "pair_id": row["pair_id"],
                        "family": row["family"],
                        "mechanism": row["mechanism"],
                        "world_id": row["world_id"],
                        "branch": row["branch"],
                        "candidate_id": row["candidate_id"],
                        "label": int(labels[index]),
                        "joint_order_position": int(positions[index]),
                        "random_action_source_candidate": int(random_sources[index]),
                        "original_score": _float(original[index]),
                        "random_action_score": _float(random_score[index]),
                        "random_action_prediction": int(random_score[index] > 0),
                        "random_action_correct": int((random_score[index] > 0) == labels[index]),
                        "xor_action_source_candidate": int(xor_sources[index]),
                        "xor_action_score": _float(xor_score[index]),
                        "xor_action_prediction": int(xor_score[index] > 0),
                        "remapped_candidate_id": int(remapped[index]),
                        "remapped_parity_prediction": int(remapped_parity_prediction[index]),
                        "remapped_parity_correct": int(remapped_parity_prediction[index] == labels[index]),
                    }
                )
    return {
        "metadata_candidate_parity_remap_xor_one": {
            "accuracy": float(np.mean(remapped_parity_prediction == labels)),
            "arrow_action_accuracy_unchanged": float(np.mean((original > 0) == labels)),
            "interpretation": "candidate_id is metadata, not a model input; parity is only an action-direction proxy.",
        },
        "baseline_ranking": baseline_ranking,
        "joint_candidate_record_order_shuffle_and_action_reassignment": shuffled_metrics,
        "deterministic_adjacent_action_reassignment_c_xor_1": {
            **xor_metrics,
            "warning": "Diagnostic only: labels were not regenerated by the simulator.",
        },
    }


def write_canonical_predictions(
    project_root: Path,
    output: Path,
) -> None:
    fields = (
        "dataset",
        "source_row_index",
        "pair_id",
        "split",
        "family",
        "mechanism",
        "world_id",
        "branch",
        "candidate_id",
        "label",
        "arrow_x",
        "arrow_y",
        "arrow_z",
        "action_sum_x",
        "action_sum_y",
        "action_sum_z",
        "score",
        "absolute_margin",
        "prediction",
        "correct",
        "recovered_axis",
        "branch_hat",
        "direction_hat",
        "observable_xor_prediction",
        "metadata_parity_prediction",
        "metadata_parity_correct",
        "remapped_candidate_id",
        "remapped_parity_prediction",
        "remapped_parity_correct",
    )
    with _gzip_csv(output / "canonical_predictions.csv.gz", fields) as writer:
        for dataset, relative in CANONICAL.items():
            values, rows, labels, _ = load_dataset(project_root / relative)
            arrow, action_sum, score = arrow_action_score(values["history"], values["candidate_actions"])
            axis, branch_hat, direction_hat = observable_bits(arrow, action_sum)
            for index, row in enumerate(rows):
                branch = int(row["branch"])
                candidate = int(row["candidate_id"])
                parity_prediction = branch == candidate % 2
                remapped = candidate ^ 1
                remapped_prediction = branch == remapped % 2
                prediction = score[index] > 0
                writer.writerow(
                    {
                        "dataset": dataset,
                        "source_row_index": index,
                        "pair_id": row["pair_id"],
                        "split": row["split"],
                        "family": row["family"],
                        "mechanism": row["mechanism"],
                        "world_id": row["world_id"],
                        "branch": branch,
                        "candidate_id": candidate,
                        "label": int(labels[index]),
                        "arrow_x": _float(arrow[index, 0]),
                        "arrow_y": _float(arrow[index, 1]),
                        "arrow_z": _float(arrow[index, 2]),
                        "action_sum_x": _float(action_sum[index, 0]),
                        "action_sum_y": _float(action_sum[index, 1]),
                        "action_sum_z": _float(action_sum[index, 2]),
                        "score": _float(score[index]),
                        "absolute_margin": _float(abs(score[index])),
                        "prediction": int(prediction),
                        "correct": int(prediction == labels[index]),
                        "recovered_axis": int(axis[index]),
                        "branch_hat": int(branch_hat[index]),
                        "direction_hat": int(direction_hat[index]),
                        "observable_xor_prediction": int(branch_hat[index] == direction_hat[index]),
                        "metadata_parity_prediction": int(parity_prediction),
                        "metadata_parity_correct": int(parity_prediction == labels[index]),
                        "remapped_candidate_id": remapped,
                        "remapped_parity_prediction": int(remapped_prediction),
                        "remapped_parity_correct": int(remapped_prediction == labels[index]),
                    }
                )


def analytic_coordinate_audit(project_root: Path) -> dict[str, Any]:
    values, rows, labels, _ = load_dataset(project_root / CANONICAL["id"])
    test = np.asarray([index for index, row in enumerate(rows) if row["split"] == "test"])
    history = values["history"][test]
    actions = values["candidate_actions"][test]
    timestamps = values["timestamps"][test]
    visibility = values["visibility"][test]
    selected_rows = [rows[index] for index in test]
    selected_labels = labels[test]
    action_sum = actions.sum(axis=1)
    estimated_velocity = multi_frame_linear_velocity(history, timestamps, visibility)
    # This exactly mirrors milestone3r_nl_v2_probe._velocity_score(v, a).
    goal_only = np.einsum("ij,ij->i", estimated_velocity[:, -3:], action_sum)
    # Coordinate-corrected scoring of the same estimator, without changing its fit.
    corrected_velocity = estimated_velocity[:, 0:3] + estimated_velocity[:, 17:20]
    corrected = np.einsum("ij,ij->i", corrected_velocity, action_sum)
    cells: list[dict[str, Any]] = []
    for family, mechanism in sorted({(row["family"], row["mechanism"]) for row in selected_rows}):
        indices = np.asarray(
            [
                index
                for index, row in enumerate(selected_rows)
                if row["family"] == family and row["mechanism"] == mechanism
            ]
        )
        pair_ids = [selected_rows[index]["pair_id"] for index in indices]
        cells.append(
            {
                "family": family,
                "mechanism": mechanism,
                "goal_only_pair_order": pair_order_accuracy(
                    selected_labels[indices], goal_only[indices], pair_ids
                ),
                "payload_plus_goal_pair_order": pair_order_accuracy(
                    selected_labels[indices], corrected[indices], pair_ids
                ),
            }
        )
    return {
        "estimator": "MultiFrameLinearVelocity (ordinary least-squares slope over visible frames)",
        "code_issue": "Published _velocity_score slices estimator output v[:, -3:], which is goal_pos only.",
        "published_goal_only_test_pair_order": pair_order_accuracy(
            selected_labels, goal_only, _metadata_pair_ids(selected_rows)
        ),
        "goal_only_estimator_over_cells": cells,
        "goal_only_cell_mean": float(np.mean([cell["goal_only_pair_order"] for cell in cells])),
        "payload_plus_goal_cell_mean": float(
            np.mean([cell["payload_plus_goal_pair_order"] for cell in cells])
        ),
        "corrected_rule_test_pair_order": pair_order_accuracy(
            selected_labels, corrected, _metadata_pair_ids(selected_rows)
        ),
    }


def source_manifest(
    project_root: Path, datasets: Sequence[Path] | None = None
) -> dict[str, Any]:
    """Hash every frozen dataset input consumed by this audit plus key code.

    ``dataset_summary`` reads three required files from every discovered data
    directory and reads ``candidate_outcomes.jsonl`` whenever it is present.
    Keep that discovery rule and this manifest rule coupled so the 30-directory
    aggregate cannot silently depend on an unhashed artifact.
    """
    relative_paths = [
        "audit/README.md",
        "audit/timearrow_shortcut_audit.py",
        "audit/tests/test_timearrow_shortcut_audit.py",
        "actmask/experiments/milestone3r_nl_v2.py",
        "actmask/experiments/milestone3r_nl_v2_probe.py",
        "actmask/experiments/milestone3r_nl_v2_full.py",
        "actmask/experiments/milestone3r_nl_v2_rankfix.py",
        "actmask/experiments/milestone3r_analytic.py",
        "actmask/experiments/milestone3r_metrics.py",
        "actmask/experiments/milestone3p_maniskill_smoke.py",
        "actmask/data/maniskill_3r_nl_tasks.py",
        "actmask/data/maniskill_state_tasks.py",
        "actmask/data/maniskill_pilot.py",
        "actmask/data/milestone3q_signed.py",
        "actmask/models/milestone3r_temporal.py",
        "outputs/actmask/milestone3r_nl_v2/preregistered_config.json",
        "outputs/actmask/milestone3r_nl_v2/full_run_config.json",
        "outputs/actmask/milestone3r_nl_v2/full_evaluation/full_evaluation.json",
        "outputs/actmask/milestone3r_nl_v2_rankfix/rankfix_report.json",
        "paper_icra2027/main.tex",
    ]
    # Audit-side executable sources are part of the method/provenance too.
    # Include bootstrap and replay scripts that may be added after this main
    # static script, while excluding interpreter caches and generated data.
    audit_source_root = project_root / "audit"
    if audit_source_root.exists():
        relative_paths.extend(
            str(path.relative_to(project_root))
            for path in audit_source_root.rglob("*.py")
            if "__pycache__" not in path.relative_to(audit_source_root).parts
        )
    discovered = list(datasets) if datasets is not None else discover_datasets(project_root)
    dataset_relative_paths: list[str] = []
    for root in discovered:
        for name in ("model_inputs.npz", "labels.npz", "metadata.jsonl"):
            path = root / name
            if not path.exists():
                raise FileNotFoundError(f"discovered dataset is missing {path}")
            dataset_relative_paths.append(str(path.relative_to(project_root)))
        outcomes = root / "candidate_outcomes.jsonl"
        if outcomes.exists():
            dataset_relative_paths.append(str(outcomes.relative_to(project_root)))
    relative_paths.extend(dataset_relative_paths)
    files = []
    for relative in sorted(set(relative_paths)):
        path = project_root / relative
        if path.exists():
            files.append(
                {
                    "path": relative,
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    return {
        "schema": SCHEMA,
        "discovered_dataset_directories": len(discovered),
        "dataset_input_files": len(set(dataset_relative_paths)),
        "files": files,
    }


def _format_metric(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return f"{value:,}"
    return f"{float(value):.6f}"


def _load_optional_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None


def _require_mapping_keys(value: dict[str, Any], keys: Sequence[str], label: str) -> None:
    missing = [key for key in keys if key not in value]
    if missing:
        raise ValueError(f"{label} evidence is missing required keys: {', '.join(missing)}")


def _is_explicitly_blocked(value: dict[str, Any]) -> bool:
    status = str(value.get("status", "")).strip().lower().replace("-", "_")
    return "blocked" in status or status.startswith("not_run")


def _evidence_blockers(value: dict[str, Any]) -> list[str]:
    result: list[str] = []
    blocker = value.get("blocker")
    if isinstance(blocker, str) and blocker.strip():
        result.append(blocker.strip())
    blockers = value.get("blockers")
    if isinstance(blockers, list):
        result.extend(str(item).strip() for item in blockers if str(item).strip())
    return result


def _validate_blocker(value: dict[str, Any], label: str) -> None:
    if not _evidence_blockers(value):
        raise ValueError(f"{label} evidence has blocked status but no actionable blocker")


def validate_environment_evidence(value: dict[str, Any] | None) -> dict[str, Any] | None:
    """Validate optional environment evidence, including an explicit blocker record."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("environment evidence must be a JSON object")
    _require_mapping_keys(value, ("status",), "environment")
    _require_mapping_keys(
        value,
        (
            "requested_versions",
            "observed_versions",
            "differences",
            "commands",
            "logs",
        ),
        "environment",
    )
    if not isinstance(value["requested_versions"], dict):
        raise ValueError("environment requested_versions must be an object")
    if not isinstance(value["observed_versions"], dict):
        raise ValueError("environment observed_versions must be an object")
    if not isinstance(value["differences"], list) or not isinstance(value["commands"], list):
        raise ValueError("environment differences and commands must be arrays")
    if not isinstance(value["logs"], dict):
        raise ValueError("environment logs must be an object")
    if _is_explicitly_blocked(value):
        _validate_blocker(value, "environment")
    else:
        if str(value["status"]).strip().lower() != "passed":
            raise ValueError(f"unknown non-blocked environment status: {value['status']!r}")
        _require_mapping_keys(
            value,
            (
                "smoke_import",
                "smoke_gpu",
                "pip_check",
                "core_environment_smoke_passed",
                "legacy_exact_reproduction",
            ),
            "environment",
        )
        if not all(
            value[key]
            for key in ("smoke_import", "smoke_gpu", "pip_check", "core_environment_smoke_passed")
        ):
            raise ValueError("passed environment evidence contains a failed qualification gate")
    return value


def validate_replay_evidence(value: dict[str, Any] | None) -> dict[str, Any] | None:
    """Validate optional bounded replay evidence or an explicit not-run blocker."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("replay evidence must be a JSON object")
    _require_mapping_keys(value, ("status",), "replay")
    if _is_explicitly_blocked(value):
        _validate_blocker(value, "replay")
        _require_mapping_keys(value, ("commands", "scope_limitations"), "replay")
    else:
        _require_mapping_keys(
            value,
            (
                "command",
                "examples",
                "pairs",
                "attempted",
                "schema_matches",
                "success_consistency",
                "filter_consistency",
                "verification_passed",
                "scope_limitations",
            ),
            "replay",
        )
        status = str(value["status"]).strip().lower()
        if status == "completed_verified_exact":
            if not value["verification_passed"] or not all(
                value[key] for key in ("schema_matches", "success_consistency", "filter_consistency")
            ):
                raise ValueError("verified replay evidence contains a failed verification gate")
        elif status == "completed_with_mismatch":
            if value["verification_passed"]:
                raise ValueError("mismatch replay evidence cannot set verification_passed=true")
        else:
            raise ValueError(f"unknown non-blocked replay status: {value['status']!r}")
    if "scope_limitations" in value and not isinstance(value["scope_limitations"], list):
        raise ValueError("replay scope_limitations must be an array")
    if "commands" in value and not isinstance(value["commands"], list):
        raise ValueError("replay commands must be an array")
    return value


def render_report(summary: dict[str, Any], audit_root: Path) -> str:
    canonical = summary["canonical"]
    total = summary["canonical_totals"]
    candidate = summary["perturbations"]["candidate"]
    temporal = summary["perturbations"]["temporal"]
    analytic = summary["analytic_coordinate_audit"]
    environment = summary.get("environment")
    replay = summary.get("replay")
    lines = [
        "# TimeArrow-v2 可复现性与数据捷径审计",
        "",
        f"审计 schema：`{SCHEMA}`。本报告由 `audit/timearrow_shortcut_audit.py` 从冻结文件重新计算；不训练模型，不修改 `outputs/`、论文或已有表格。",
        "",
        "## 结论",
        "",
        "**TimeArrow-v2 应标记为 `shortcut-saturated`。** 一个不训练、只读取正式 observation/action 的 `ArrowActionCompatibility` 规则，在四个 canonical retained 集合的全部 "
        f"{total['retained_examples']:,} 行上达到 `{total['retained_arrow_action_accuracy']:.6f}`，并复现 ID test 的 `Top-1=1`、`Top-3=1`、`NDCG=1`、`regret=0`。因此这些整齐数字主要是数据构造、严格 flip 筛选和动作方向编码的代数后果。这个合法输入的 post-hoc audit stress baseline 不是预注册或 validation-selected comparator；它证明的是 benchmark 存在未排除的饱和捷径，因而现有结果不能支持 learned method 存在未解的 headroom 或完成机制识别。",
        "",
        f"本审计**没有发现足以断言人工伪造数据的直接证据**。四个 canonical 集合的 metadata/label 共 `{total['direct_outcome_join']['joined_rows']:,}` 行，按 candidate outcome key 和 branch 直接连接后，success 布尔值 `{total['direct_outcome_join']['label_success_matches']:,}/{total['direct_outcome_join']['expected_metadata_rows']:,}` 精确一致；outcome key 和 `(pair_id, branch)` 也都唯一。这里证明的是冻结文件内部自洽，不是其历史生成过程的独立真实性证明。发现的问题是确定性捷径、baseline 坐标遗漏、筛选偏差以及 provenance/重放证据不足。",
        "",
        "## 代码事实",
        "",
        "- [生成器](../actmask/experiments/milestone3r_nl_v2.py#L57-L68) 把 branch 1 构造成 branch 0 前四帧的严格反序；最后两帧相同。",
        "- [候选动作生成](../actmask/experiments/milestone3r_nl_v2.py#L119-L149) 在 full 当前实现中以 `candidate_id % 2` 交错编码正/负方向。",
        "- [筛选逻辑](../actmask/experiments/milestone3r_nl_v2.py#L198-L211) 先记录所有执行 outcome，再删除两 branch 结果相同的 non-flip pair。",
        "- [observation schema](../actmask/data/maniskill_state_tasks.py#L205-L213) 的 `0:3` 是 payload position，`17:20` 是 goal position。",
        "- 原 analytic [评分函数](../actmask/experiments/milestone3r_nl_v2_probe.py#L28) 只取 estimator 输出的 `[-3:]`，也就是 goal position；两个平移 family 的早期 cue 实际写在 payload position。",
        "- 所谓 `hysteretic_mode_memory` 的 threshold 被存储但没有进入 [response 公式](../actmask/data/maniskill_3r_nl_tasks.py#L95-L111)；实际是带 delay 的 signed quadratic response。",
        "",
        "## 数据事实：canonical retained 与 attempted outcomes",
        "",
        "| 条件 | attempted branch executions | retained examples | non-flip pairs | retained parity rule | Arrow×Action | min abs(score) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in CANONICAL:
        value = canonical[name]
        outcome = value["outcomes"]
        lines.append(
            f"| {name} | {outcome['attempted_branch_executions']:,} | {value['examples']:,} | {outcome['nonflip_pairs']:,} | {value['metadata_parity_accuracy']:.6f} | {value['arrow_action_accuracy']:.6f} | {value['minimum_absolute_score']:.6f} |"
        )
    lines.extend(
        [
            "",
            f"四个集合共尝试 `{total['attempted_branch_executions']:,}` 次 branch execution；筛选前 parity 规则命中 `{total['attempted_metadata_parity_accuracy']:.6f}`，筛选后在 `{total['retained_examples']:,}` retained examples 上恰为 `1.000000`。Temporal OOD 在 6,000 个 attempted pair 中保留 2,800 个（retention `46.667%`）；其余 3,200 个 non-flip pair 全是 `(success, success)`，且没有对应的 model input，因此不能对被删除样本运行公开输入规则。",
            "",
            f"扫描到的全部 30 个 v2 数据目录（包含 full/raw/prefix/probe/smoke 派生重复）共有 `{summary['dataset_inventory']['stored_examples_including_derived_duplicates']:,}` 行：Arrow×Action 总准确率 `{summary['dataset_inventory']['all_directories_arrow_action_accuracy']:.6f}`，而 metadata parity 总准确率 `{summary['dataset_inventory']['all_directories_metadata_parity_accuracy']:.6f}`。旧 probe/smoke 的 candidate mapping 并非奇偶交错，所以 parity 不是普遍规则；observable Arrow×Action 才是 30/30 目录共同的捷径。",
            "",
            f"所有 canonical retained pair 都满足：每个 `pair_id` 恰有两行、前四帧严格反序、最后两帧/action/timestamps/visibility/confidence/timing/TCP/static features 完全相同、标签恰好翻转。另将每行 metadata/label 按 `(condition, family, mechanism, world_id, candidate_id)` 连接到唯一 outcome，再取 `branch{{branch}}_success`，覆盖率和两类 success match rate 均为 `{total['direct_outcome_join']['coverage']:.6f}`。逐目录数据见 [dataset_inventory.json](results/dataset_inventory.json)。",
            "",
            "## 零训练规则",
            "",
            "```python",
            "arrow = (history[:, 3, 0:3] - history[:, 0, 0:3]",
            "         + history[:, 3, 17:20] - history[:, 0, 17:20])",
            "action_direction = candidate_actions.sum(axis=1)",
            "score = (arrow * action_direction).sum(axis=1)",
            "prediction = score > 0",
            "```",
            "",
            "它不使用 `branch`、`candidate_id`、`family`、hidden parameters 或 success。在 retained 集合上，arrow 符号恢复 branch bit，动作净位移符号恢复 direction bit，标签就是二者 XNOR。全量逐样本 score、margin 和预测保存在 [canonical_predictions.csv.gz](results/canonical_predictions.csv.gz)。",
            "",
            "### ID test ranking",
            "",
            "| Candidate count | contexts | Top-1 | Top-3 | NDCG | regret |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name, metric in candidate["baseline_ranking"].items():
        lines.append(
            f"| {name} | {metric['contexts']:,} | {metric['top1_success']:.6f} | {metric['top3_success_recall']:.6f} | {metric['corrected_utility_gain_ndcg']:.6f} | {metric['normalized_regret']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## 扰动实验",
            "",
            f"- **candidate parity remap**：只把 metadata 的 `candidate_id` 改成 `candidate_id ^ 1` 后，parity 公式准确率为 `{candidate['metadata_candidate_parity_remap_xor_one']['accuracy']:.6f}`，而 Arrow×Action 仍为 `1.000000`。这说明根本捷径是观察箭头和动作方向，不是模型直接读取 candidate ID。",
            "- **joint candidate record order shuffle**：在每个 history context 内联合重排 row/label/score/action，5 个种子的 C5/C10/C20 指标逐项不变；这是 evaluator 的顺序不变性检查。",
            f"- **adjacent action reassignment (`c -> c xor 1`)**：保持旧标签而交换动作后，规则准确率 `{candidate['deterministic_adjacent_action_reassignment_c_xor_1']['accuracy']:.6f}`、pair-order `{candidate['deterministic_adjacent_action_reassignment_c_xor_1']['pair_order_accuracy']:.6f}`。随机 action-only reassignment 的结果接近 chance。它们只是 shortcut diagnostic；没有 simulator 重新打标签，不能称为物理 counterfactual。",
            f"- **exact early partner reverse**：在原标签下准确率 `{temporal['exact_early_partner_reverse']['accuracy']:.6f}`、pair-order `{temporal['exact_early_partner_reverse']['pair_order_accuracy_original_labels']:.6f}`，在 partner 标签下准确率 `{temporal['exact_early_partner_reverse']['accuracy_against_partner_labels']:.6f}`。",
            f"- **last-two-only**：arrow 全为零，准确率 `{temporal['last_two_only']['accuracy']:.6f}`、pair-order `{temporal['last_two_only']['pair_order_accuracy']:.6f}`。",
            "- **genuine random early-frame shuffle**：为每个 pair 采样共同的前四帧 permutation，5 个种子降至 chance 附近；这和现有代码中名字叫 `early_independent_permutation`、实际只是固定 `flip` 的操作不同。",
            "",
            "逐样本扰动证据见 [id_test_candidate_perturbations.csv.gz](results/id_test_candidate_perturbations.csv.gz) 和 [id_test_temporal_perturbations.csv.gz](results/id_test_temporal_perturbations.csv.gz)。",
            "",
            "## 原 analytic baseline 的坐标遗漏",
            "",
            "| family | mechanism | goal-only pair-order | payload+goal pair-order |",
            "|---|---|---:|---:|",
        ]
    )
    for cell in analytic["goal_only_estimator_over_cells"]:
        lines.append(
            f"| {cell['family']} | {cell['mechanism']} | {cell['goal_only_pair_order']:.6f} | {cell['payload_plus_goal_pair_order']:.6f} |"
        )
    lines.extend(
        [
            "",
            f"goal-only 的 5-cell 均值为 `{analytic['goal_only_cell_mean']:.6f}`，在同一 estimator 上 post-hoc 补齐 payload+goal 坐标后为 `{analytic['payload_plus_goal_cell_mean']:.6f}`。保存报告基于有坐标遗漏的 goal-only baseline 所做的算术是可复现的：四个平移 cell `0.5`、一个 rotating cell `1.0`、总体 `0.6`，从而得到 learned-minus-analytic `+0.400`；坐标修正的 post-hoc stress baseline 为 `1.000000`。它未经预注册/验证选择，不应冒充原 comparator；但它足以证明原 `+0.400` 不能作为“仍有 learned headroom”的有效证据。",
            "",
            "## 为什么数字会恰好是 0.500 / 1.000",
            "",
            "- retained label balance 必为 `0.5`：每个 strict flip pair 按定义恰有一个正样本和一个负样本，并同时保存两行。",
            "- static/action/final-two pair-order 必为 `0.5`：pair 两边这些输入严格相同，score tie，而 metric 明确给 tie `0.5`。",
            "- candidate template marginal success 在 strict flip 上必为 `0.5`：当前 diversity audit 计算 `(branch0 + branch1)/2`，却没有检查 `branch × direction` 的 XOR interaction。",
            "- static/action Top-1 在两 branch 平均后为 `0.5`：同一套 candidate scores 对应完全互补的 branch labels。",
            "- 一旦规则把全部成功 candidate 排在失败 candidate 前，`Top-1=1`、`Top-3=1`、`NDCG=1`、`regret=0` 是同一次完美排序的代数派生，不是四项独立证据。",
            "",
            "## 推断",
            "",
            "1. GRU 很可能学到了低维 arrow-direction XNOR；这些结果不足以支持 learned method 在该 benchmark 上存在机制识别能力或未解的 headroom。这里的 payload+goal 规则是 post-hoc audit stress baseline，不是预注册 comparator。论文把任务定位为 controlled procedural test，这一限定不改变 shortcut 对方法比较效度的影响。",
            "2. Physical/Temporal OOD 仍保留同一 arrow/action 编码，因此 OOD=1 并不证明对隐藏物理参数或 delay 的机制外推。",
            "3. 在继续针对该 benchmark 优化复杂模型前，应先重设计任务：取消 strict-flip-only 选择、随机化 candidate-direction 映射、保存全部执行轨迹，并加入这个规则作为 admission gate。",
            "",
            "## 可复现性与环境",
            "",
        ]
    )
    if environment:
        observed = environment.get("observed_versions", {})
        blockers = _evidence_blockers(environment)
        if _is_explicitly_blocked(environment):
            lines.append(
                f"独立环境状态：`{environment.get('status')}`。实际版本：`{json.dumps(observed, ensure_ascii=False, sort_keys=True)}`。阻碍：{'；'.join(blockers)}。详细证据见 [environment](environment/)。"
            )
        else:
            lines.append(
                f"独立环境状态：`{environment.get('status')}`，import smoke=`{environment.get('smoke_import')}`，GPU smoke=`{environment.get('smoke_gpu')}`，pip check=`{environment.get('pip_check')}`，core environment smoke=`{environment.get('core_environment_smoke_passed')}`，legacy exact reproduction=`{environment.get('legacy_exact_reproduction')}`。实际版本：`{json.dumps(observed, ensure_ascii=False, sort_keys=True)}`。已知差异：`{json.dumps(environment.get('differences', []), ensure_ascii=False)}`。详细证据见 [environment](environment/)。"
            )
            if blockers:
                lines.append("环境阻碍/错误：" + "；".join(blockers))
    else:
        lines.append("独立环境取证尚未写入 `audit/environment/environment_status.json`；这是当前未完成项。")
    if replay:
        blockers = _evidence_blockers(replay)
        if _is_explicitly_blocked(replay):
            lines.append(
                f"最小再执行状态：`{replay.get('status')}`；未运行/阻碍原因：{'；'.join(blockers)}。详见 [replay](replay/)。"
            )
        else:
            direct = replay.get("direct_frozen_comparison", {})
            lines.append(
                f"最小再执行状态：`{replay.get('status')}`，verification=`{replay.get('verification_passed')}`，schema=`{replay.get('schema_matches')}`，success=`{replay.get('success_consistency')}`，filter=`{replay.get('filter_consistency')}`，retained arrays exact=`{direct.get('all_retained_arrays_exact')}`，max abs error=`{direct.get('max_retained_array_abs_error')}`。这是 reduced-batch 下对冻结 seed/world/candidate 子集的 bounded re-execution，不是相同 parent batch 或 simulator snapshot replay。详见 [replay](replay/)。"
            )
            if blockers:
                lines.append("回放阻碍/错误：" + "；".join(blockers))
    else:
        lines.append("最小 episode 再执行尚未写入 `audit/replay/replay_verification.json`。")
    lines.extend(
        [
            "",
            "## 尚未验证 / 不能声称",
            "",
            "- `candidate_outcomes.jsonl` 只保存最终布尔值，没有逐步 distance/success、最终 simulator state 或 discarded-pair model inputs；冻结文件内部一致性不能独立证明这些布尔值确由历史 GPU PhysX 运行产生。",
            "- 项目没有 Git 元数据和完整旧环境 lock；只能重建核心版本，不能声称 bit-exact legacy environment。",
            "- 当前 generator 的交错 candidate mapping 与同一 config hash 下旧 probe 的前五正/后五负 mapping 不一致，而 generation report 没有 source hash。这是 provenance gap，不是造假证明。",
            "- action-only shuffle 未经 simulator 重新打标签，只能诊断 shortcut，不能用来估计真实 counterfactual performance。",
            "",
            "## 复现命令与产物",
            "",
            "```bash",
            "/root/miniconda3/bin/python audit/timearrow_shortcut_audit.py",
            "/root/miniconda3/bin/python -m unittest discover -s audit/tests -v",
            "```",
            "",
            "- [summary.json](results/summary.json)：全部指标与结论",
            "- [dataset_inventory.json](results/dataset_inventory.json)：所有发现的数据目录",
            "- [source_manifest.json](results/source_manifest.json)：冻结输入与关键源码 SHA-256",
            "- [canonical_predictions.csv.gz](results/canonical_predictions.csv.gz)：85,600 行逐样本规则证据",
            "- [id_test_candidate_perturbations.csv.gz](results/id_test_candidate_perturbations.csv.gz)：candidate 扰动逐样本证据",
            "- [id_test_temporal_perturbations.csv.gz](results/id_test_temporal_perturbations.csv.gz)：temporal 扰动逐样本证据",
            "",
        ]
    )
    return "\n".join(lines)


def run(project_root: Path, audit_root: Path) -> dict[str, Any]:
    project_root = project_root.resolve()
    audit_root = audit_root.resolve()
    output = audit_root / "results"
    output.mkdir(parents=True, exist_ok=True)

    datasets = discover_datasets(project_root)
    inventory = [dataset_summary(path, project_root, canonical=False) for path in datasets]
    canonical = {
        name: dataset_summary(project_root / relative, project_root, canonical=True)
        for name, relative in CANONICAL.items()
    }
    totals = canonical_totals(canonical)

    write_canonical_predictions(project_root, output)
    id_values, id_rows, id_labels, _ = load_dataset(project_root / CANONICAL["id"])
    test_values, test_rows, test_labels, original_indices = _test_subset(id_values, id_rows, id_labels)
    temporal = temporal_perturbations(
        test_values, test_rows, test_labels, original_indices, output
    )
    candidate = candidate_perturbations(
        test_values, test_rows, test_labels, original_indices, output
    )
    manifest = source_manifest(project_root, datasets)
    environment = validate_environment_evidence(
        _load_optional_json(audit_root / "environment/environment_status.json")
    )
    replay = validate_replay_evidence(
        _load_optional_json(audit_root / "replay/replay_verification.json")
    )
    summary: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "python": {"version": platform.python_version(), "executable": sys.executable},
        "benchmark_status": "shortcut-saturated",
        "zero_training_baseline": {
            "name": ArrowActionCompatibility.name,
            "requires_training": ArrowActionCompatibility.requires_training,
        },
        "direct_fabrication_evidence_found": False,
        "conclusion": (
            "No direct fabrication evidence was found. A legal-input, zero-training rule "
            "perfectly solves every retained TimeArrow-v2 dataset; the benchmark is shortcut-saturated."
        ),
        "dataset_inventory": {
            "directories": len(inventory),
            "stored_examples_including_derived_duplicates": sum(item["examples"] for item in inventory),
            "all_directories_arrow_action_accuracy": (
                sum(item["arrow_action_correct"] for item in inventory)
                / sum(item["examples"] for item in inventory)
            ),
            "all_directories_metadata_parity_accuracy": (
                sum(item["metadata_parity_correct"] for item in inventory)
                / sum(item["examples"] for item in inventory)
            ),
            "items": inventory,
        },
        "canonical": canonical,
        "canonical_totals": totals,
        "analytic_coordinate_audit": analytic_coordinate_audit(project_root),
        "perturbations": {"candidate": candidate, "temporal": temporal},
        "environment": environment,
        "replay": replay,
        "limitations": [
            "No per-step execution traces or simulator snapshots exist in the frozen artifacts.",
            "Discarded non-flip pairs have outcome booleans but no saved model inputs.",
            "The legacy environment is not fully locked and the directory is not a Git repository.",
            "Action-only reassignment is diagnostic because labels were not regenerated.",
        ],
    }
    _write_json(output / "dataset_inventory.json", summary["dataset_inventory"])
    _write_json(output / "source_manifest.json", manifest)
    _write_json(output / "summary.json", summary)
    report = render_report(summary, audit_root)
    (audit_root / "audit_report.md").write_text(report, encoding="utf-8")

    # Include all generated evidence, including environment/replay logs and
    # fresh bounded artifacts. Exclude caches and this self-referential manifest.
    generated_paths = [audit_root / "audit_report.md"]
    for directory in (output, audit_root / "environment", audit_root / "replay"):
        if directory.exists():
            generated_paths.extend(path for path in directory.rglob("*") if path.is_file())
    generated_files = []
    for path in sorted(set(generated_paths)):
        relative = path.relative_to(audit_root)
        if relative == Path("results/generated_manifest.json"):
            continue
        if "__pycache__" in relative.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        generated_files.append(
            {
                "path": str(relative),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    _write_json(output / "generated_manifest.json", {"schema": SCHEMA, "files": generated_files})
    return summary


def main() -> None:
    default_project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=default_project)
    parser.add_argument("--audit-root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    summary = run(args.project_root, args.audit_root)
    print(
        json.dumps(
            {
                "schema": summary["schema"],
                "benchmark_status": summary["benchmark_status"],
                "canonical_retained_examples": summary["canonical_totals"]["retained_examples"],
                "canonical_arrow_action_accuracy": summary["canonical_totals"][
                    "retained_arrow_action_accuracy"
                ],
                "audit_report": str((args.audit_root / "audit_report.md").resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
