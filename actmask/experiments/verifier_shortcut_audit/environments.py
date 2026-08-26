"""V5 frozen audit environments and deterministic matching manifests."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .common import (
    HISTORICAL,
    OUT,
    append_log,
    read_json,
    read_jsonl,
    sha256_file,
    sha256_json,
    write_json,
    write_jsonl,
)


PILOT = HISTORICAL / "rm_acv_pilot166"
PRIMARY = (
    "N1_same_task_near_state",
    "N2_local_temporal_permutation",
    "N3_arm_gripper_desynchronization",
    "N4_phase_shifted_same_task",
    "N5_joint_coordination_corruption",
    "N6_endpoint_matched_path_corruption",
)
LEAVE_ONE = (
    "N1_same_task_near_state",
    "N2_local_temporal_permutation",
    "N4_phase_shifted_same_task",
    "N5_joint_coordination_corruption",
    "N6_endpoint_matched_path_corruption",
)
GROUP_E = (
    "N2_local_temporal_permutation",
    "N3_arm_gripper_desynchronization",
    "N5_joint_coordination_corruption",
    "N6_endpoint_matched_path_corruption",
)
GROUP_R = (
    "N1_same_task_near_state",
    "N4_phase_shifted_same_task",
)
GROUP_B = (
    "N7_reciprocal_near_state_context_swap",
    "N8_visual_substate_reciprocal_context_swap",
)


def _indices(mask: np.ndarray) -> np.ndarray:
    return np.flatnonzero(mask).astype(np.int32)


def _sha_indices(indices: np.ndarray) -> str:
    return hashlib.sha256(
        np.asarray(indices, dtype=np.int32).tobytes()
    ).hexdigest()


def _safe_key(value: str) -> str:
    return (
        value.replace("N1_", "n1_")
        .replace("N2_", "n2_")
        .replace("N3_", "n3_")
        .replace("N4_", "n4_")
        .replace("N5_", "n5_")
        .replace("N6_", "n6_")
        .replace("N7_", "n7_")
        .replace("N8_", "n8_")
        .replace("-", "_")
        .replace("/", "_")
    )


def _count_summary(
    indices: np.ndarray,
    rows: list[dict[str, Any]],
    labels: np.ndarray,
) -> dict[str, Any]:
    families = Counter(rows[int(index)]["family"] for index in indices)
    tasks = Counter(rows[int(index)]["task"] for index in indices)
    episodes = {rows[int(index)]["episode"] for index in indices}
    sources = {rows[int(index)]["source_episode"] for index in indices}
    return {
        "samples": int(len(indices)),
        "labels": {
            str(label): int(np.sum(labels[indices] == label)) for label in (0, 1)
        },
        "families": dict(sorted(families.items())),
        "tasks": dict(sorted(tasks.items())),
        "context_episodes": len(episodes),
        "source_episodes": len(sources),
        "index_sha256": _sha_indices(indices),
    }


def _squared_distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    result = (
        np.sum(np.square(left), axis=1, keepdims=True)
        + np.sum(np.square(right), axis=1)[None]
        - 2.0 * left @ right.T
    )
    return np.maximum(result, 0.0)


def _standardized_mean_difference(
    feature: np.ndarray,
    positive: np.ndarray,
    negative: np.ndarray,
    names: list[str],
) -> dict[str, Any]:
    if not len(positive) or not len(negative):
        return {
            "mean_absolute": None,
            "maximum_absolute": None,
            "features": [],
        }
    p = feature[positive]
    n = feature[negative]
    pooled = np.sqrt((p.var(0) + n.var(0)) / 2.0)
    values = np.abs(p.mean(0) - n.mean(0)) / np.maximum(pooled, 1e-8)
    ordered = np.argsort(-values)
    return {
        "mean_absolute": float(values.mean()),
        "maximum_absolute": float(values.max()),
        "features": [
            {
                "feature": names[int(index)],
                "absolute_standardized_mean_difference": float(values[index]),
            }
            for index in ordered
        ],
    }


def _match(
    *,
    family: str,
    kind: str,
    feature: np.ndarray,
    names: list[str],
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    labels: np.ndarray,
    sample_ids: list[str],
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    mean = feature[train_indices].mean(0)
    std = feature[train_indices].std(0)
    active = std > 1e-8
    if not np.any(active):
        raise RuntimeError(f"{family}/{kind}: every matching feature is constant")
    standardized = (
        (feature[:, active] - mean[active]) / std[active]
    ).astype(np.float32)
    train_positive = train_indices[labels[train_indices] == 1]
    train_negative = train_indices[labels[train_indices] == 0]
    development_distance = np.sqrt(
        _squared_distances(
            standardized[train_positive], standardized[train_negative]
        )
    )
    nearest = np.concatenate(
        (development_distance.min(1), development_distance.min(0))
    )
    caliper = float(np.quantile(nearest, 0.90))
    test_positive = test_indices[labels[test_indices] == 1]
    test_negative = test_indices[labels[test_indices] == 0]
    test_distance = np.sqrt(
        _squared_distances(
            standardized[test_positive], standardized[test_negative]
        )
    )
    candidates = []
    for p_row, p_index in enumerate(test_positive):
        for n_row, n_index in enumerate(test_negative):
            distance = float(test_distance[p_row, n_row])
            if distance <= caliper + 1e-12:
                candidates.append(
                    (
                        distance,
                        sample_ids[int(p_index)],
                        sample_ids[int(n_index)],
                        int(p_index),
                        int(n_index),
                    )
                )
    candidates.sort()
    used_positive: set[int] = set()
    used_negative: set[int] = set()
    pairs: list[dict[str, Any]] = []
    for distance, positive_id, negative_id, positive_index, negative_index in candidates:
        if positive_index in used_positive or negative_index in used_negative:
            continue
        used_positive.add(positive_index)
        used_negative.add(negative_index)
        pairs.append(
            {
                "schema": "vsa-deterministic-matching-pair-v1",
                "environment": kind,
                "family": family,
                "pair_index": len(pairs),
                "positive_sample_id": positive_id,
                "positive_unified_index": positive_index,
                "negative_sample_id": negative_id,
                "negative_unified_index": negative_index,
                "standardized_euclidean_distance": distance,
                "within_frozen_caliper": True,
            }
        )
    matched_positive = np.asarray(
        [pair["positive_unified_index"] for pair in pairs], dtype=np.int32
    )
    matched_negative = np.asarray(
        [pair["negative_unified_index"] for pair in pairs], dtype=np.int32
    )
    matched = np.sort(
        np.concatenate((matched_positive, matched_negative))
    ).astype(np.int32)
    before = _standardized_mean_difference(
        feature, test_positive, test_negative, names
    )
    after = _standardized_mean_difference(
        feature, matched_positive, matched_negative, names
    )
    quality = {
        "schema": "vsa-matching-quality-v1",
        "environment": kind,
        "family": family,
        "algorithm": (
            "deterministic one-to-one greedy minimum standardized Euclidean "
            "distance; lexical sample-ID tie-break"
        ),
        "normalization_partition": "same-family train partition only",
        "caliper_definition": (
            "90th percentile of train/development opposite-label "
            "nearest-neighbor distances"
        ),
        "active_feature_count": int(active.sum()),
        "constant_feature_count": int((~active).sum()),
        "active_features": [
            name for name, keep in zip(names, active, strict=True) if keep
        ],
        "normalization_mean_sha256": hashlib.sha256(
            mean.astype(np.float32).tobytes()
        ).hexdigest(),
        "normalization_std_sha256": hashlib.sha256(
            std.astype(np.float32).tobytes()
        ).hexdigest(),
        "caliper": caliper,
        "train_samples": int(len(train_indices)),
        "test_samples_before": int(len(test_indices)),
        "pairs_retained": len(pairs),
        "samples_retained": int(len(matched)),
        "coverage": float(len(matched) / max(1, len(test_indices))),
        "candidate_pairs_within_caliper": len(candidates),
        "distance": {
            "mean": float(np.mean([pair[0] for pair in candidates]))
            if candidates
            else None,
            "matched_mean": float(
                np.mean(
                    [
                        pair["standardized_euclidean_distance"]
                        for pair in pairs
                    ]
                )
            )
            if pairs
            else None,
            "matched_max": float(
                max(
                    pair["standardized_euclidean_distance"]
                    for pair in pairs
                )
            )
            if pairs
            else None,
        },
        "standardized_mean_difference_before": before,
        "standardized_mean_difference_after": after,
        "heldout_model_performance_used": False,
        "pass": bool(
            pairs
            and len(matched_positive) == len(matched_negative)
            and len(set(matched.tolist())) == len(matched)
        ),
    }
    return matched, pairs, quality


def _feature_indices(
    names: list[str], prefixes: tuple[str, ...]
) -> list[int]:
    result = [
        index
        for index, name in enumerate(names)
        if any(name == prefix or name.startswith(prefix + "[") for prefix in prefixes)
    ]
    if not result:
        raise RuntimeError(f"no features found for {prefixes}")
    return result


def _variant_arrays(
    arrays: dict[str, np.ndarray],
    nuisance: dict[str, np.ndarray],
    schema: dict[str, Any],
) -> dict[str, np.ndarray]:
    language_names = schema["groups"]["language"]

    def language_block(prefix: str) -> np.ndarray:
        positions = _feature_indices(language_names, (prefix,))
        result = nuisance["language"][:, positions].astype(np.float32)
        if result.shape[1] != 128:
            raise RuntimeError(f"{prefix}: expected 128 dimensions, got {result.shape}")
        return result

    visual = arrays["visual_history"].astype(np.float32)
    context_anchor = arrays["context_anchor_index"].astype(np.int64)
    raw = nuisance["raw_visual_unique"][context_anchor].astype(np.float32)
    current = np.zeros_like(visual)
    current[:, -1] = visual[:, -1]
    difference = np.abs(
        np.diff(visual, axis=1, prepend=visual[:, :1])
    ).astype(np.float32)
    border = np.zeros_like(visual)
    border[:, -1, :6] = raw[:, 30:36]
    center = np.zeros_like(visual)
    center[:, -1, :6] = raw[:, 36:42]
    global_statistics = np.zeros_like(visual)
    global_statistics[:, -2, :8] = raw[:, :8]
    global_statistics[:, -1, :] = raw[:, 8:30]
    return {
        "language_correct": arrays["language_embedding"].astype(np.float32),
        "language_shuffled": language_block("shuffled_language_embedding"),
        "language_empty": language_block("empty_language_embedding"),
        "language_task_id_only": language_block("task_id_only_embedding"),
        "visual_full": visual,
        "visual_current_only": current,
        "visual_history_difference_only": difference,
        "visual_static_border_only": border,
        "visual_center_workspace_only": center,
        "visual_global_statistics_only": global_statistics,
    }


def _prior_weights(
    test_indices: np.ndarray,
    labels: np.ndarray,
    values: list[str],
    target_positive: float | None = None,
) -> np.ndarray:
    weights = np.zeros(len(labels), dtype=np.float32)
    if target_positive is not None:
        observed = float(labels[test_indices].mean())
        weights[test_indices[labels[test_indices] == 1]] = (
            target_positive / max(observed, 1e-12)
        )
        weights[test_indices[labels[test_indices] == 0]] = (
            (1.0 - target_positive) / max(1.0 - observed, 1e-12)
        )
    else:
        counts = Counter(values[int(index)] for index in test_indices)
        for index in test_indices:
            weights[int(index)] = 1.0 / counts[values[int(index)]]
    nonzero = weights[test_indices]
    weights[test_indices] /= max(float(nonzero.mean()), 1e-12)
    return weights


def build() -> dict[str, Any]:
    rows = read_jsonl(OUT / "unified_audit_manifest.jsonl")
    with np.load(OUT / "unified_audit_arrays.npz") as stored:
        arrays = {name: stored[name].copy() for name in stored.files}
    with np.load(OUT / "nuisance_features.npz") as stored:
        nuisance = {name: stored[name].copy() for name in stored.files}
    schema = read_json(OUT / "nuisance_feature_schema.json")
    task_cv = read_json(PILOT / "task_cv_preregistered.json")
    labels = arrays["label"].astype(np.int8)
    families = np.asarray([row["family"] for row in rows], dtype=object)
    tasks = np.asarray([row["task"] for row in rows], dtype=object)
    splits = np.asarray(
        [row["split_group"]["episode_split"] for row in rows], dtype=object
    )
    sample_ids = [row["sample_id"] for row in rows]
    source_safe = np.asarray(
        [row["split_group"]["source_safe_for_split"] for row in rows],
        dtype=bool,
    )
    primary_mask = np.isin(families, PRIMARY)
    split_masks = {
        split: splits == split for split in ("train", "validation", "test")
    }
    index_arrays: dict[str, np.ndarray] = {}
    train_specs: dict[str, Any] = {}

    def add_spec(
        name: str,
        train_mask: np.ndarray,
        validation_mask: np.ndarray,
        test_mask: np.ndarray,
        environment: str,
        purpose: str,
    ) -> None:
        partitions = {
            "train": _indices(train_mask),
            "validation": _indices(validation_mask),
            "test": _indices(test_mask),
        }
        for partition, indices in partitions.items():
            index_arrays[f"spec__{name}__{partition}"] = indices
        train_specs[name] = {
            "environment": environment,
            "purpose": purpose,
            "partitions": {
                partition: {
                    "array_key": f"spec__{name}__{partition}",
                    **_count_summary(indices, rows, labels),
                }
                for partition, indices in partitions.items()
            },
        }

    add_spec(
        "pooled_n1_n6",
        primary_mask & split_masks["train"],
        primary_mask & split_masks["validation"],
        primary_mask & split_masks["test"],
        "pooled_reference",
        "shared reference model for E5/E6/E7/E9/E10/E11/E12",
    )
    for family in PRIMARY:
        short = family.split("_", 1)[0].lower()
        family_mask = families == family
        add_spec(
            f"within_{short}",
            family_mask & split_masks["train"],
            family_mask & split_masks["validation"],
            family_mask & split_masks["test"],
            "E1",
            f"within-family IID for {family}",
        )
    for heldout in LEAVE_ONE:
        short = heldout.split("_", 1)[0].lower()
        training_families = tuple(family for family in PRIMARY if family != heldout)
        add_spec(
            f"leaveout_{short}",
            np.isin(families, training_families) & split_masks["train"],
            np.isin(families, training_families) & split_masks["validation"],
            (families == heldout) & split_masks["test"],
            "E2",
            f"negative-generator-held-out test for {heldout}",
        )
    add_spec(
        "group_e_to_r",
        np.isin(families, GROUP_E) & split_masks["train"],
        np.isin(families, GROUP_E) & split_masks["validation"],
        np.isin(families, GROUP_R) & split_masks["test"],
        "E3",
        "easy synthetic corruption families to real-action retrieval families",
    )
    add_spec(
        "group_r_to_e",
        np.isin(families, GROUP_R) & split_masks["train"],
        np.isin(families, GROUP_R) & split_masks["validation"],
        np.isin(families, GROUP_E) & split_masks["test"],
        "E4",
        "real-action retrieval families to easy synthetic corruption families",
    )
    for fold_record in task_cv["folds"]:
        fold = int(fold_record["fold"][0].split("_")[-1])
        add_spec(
            f"task_fold_{fold}",
            primary_mask & np.isin(tasks, fold_record["train_tasks"]),
            primary_mask & np.isin(tasks, fold_record["validation_tasks"]),
            primary_mask & np.isin(tasks, fold_record["test_tasks"]),
            "E8",
            f"frozen complete-task group CV fold {fold}",
        )

    action_names = schema["groups"]["action"]
    e5_positions = _feature_indices(
        action_names,
        (
            "absolute_magnitude_mean",
            "absolute_magnitude_rms",
            "action_variance",
            "smoothness_energy",
            "jerk_abs_mean",
            "total_displacement",
            "start_action",
            "end_action",
            "arm_gripper_event_count",
            "gripper_event_count_per_dimension",
        ),
    )
    e5_feature = nuisance["action"][:, e5_positions]
    e5_names = [action_names[index] for index in e5_positions]
    context_names = schema["groups"]["context"]
    progress_position = context_names.index("normalized_episode_progress")
    duration_position = context_names.index("episode_length")
    horizon_position = action_names.index("action_horizon")
    e6_feature = np.column_stack(
        (
            nuisance["context"][:, progress_position],
            nuisance["context"][:, duration_position],
            nuisance["action"][:, horizon_position],
        )
    ).astype(np.float32)
    e6_names = [
        "normalized_episode_progress",
        "episode_duration_seconds",
        "action_horizon",
    ]
    matching_directory = OUT / "matching_manifests"
    matching_directory.mkdir(parents=True, exist_ok=True)
    quality_records: list[dict[str, Any]] = []
    matching_files: list[dict[str, Any]] = []
    for family in PRIMARY:
        short = family.split("_", 1)[0].lower()
        train_indices = _indices((families == family) & split_masks["train"])
        test_indices = _indices((families == family) & split_masks["test"])
        for kind, feature, names in (
            ("E5_action_summary", e5_feature, e5_names),
            ("E6_progress", e6_feature, e6_names),
        ):
            matched, pairs, quality = _match(
                family=family,
                kind=kind,
                feature=feature,
                names=names,
                train_indices=train_indices,
                test_indices=test_indices,
                labels=labels,
                sample_ids=sample_ids,
            )
            key = f"{kind.lower()}__{short}"
            index_arrays[key] = matched
            path = matching_directory / f"{kind.lower()}__{short}.jsonl"
            write_jsonl(path, pairs)
            quality["manifest_path"] = str(path.resolve())
            quality["manifest_sha256"] = sha256_file(path)
            quality["index_array_key"] = key
            quality["matched_index_sha256"] = _sha_indices(matched)
            quality_records.append(quality)
            matching_files.append(
                {
                    "environment": kind,
                    "family": family,
                    "path": str(path.resolve()),
                    "sha256": quality["manifest_sha256"],
                    "pairs": len(pairs),
                }
            )

    index_arrays["e5_action_summary__all"] = np.sort(
        np.concatenate(
            [
                index_arrays[
                    f"e5_action_summary__{family.split('_', 1)[0].lower()}"
                ]
                for family in PRIMARY
            ]
        )
    ).astype(np.int32)
    index_arrays["e6_progress__all"] = np.sort(
        np.concatenate(
            [
                index_arrays[f"e6_progress__{family.split('_', 1)[0].lower()}"]
                for family in PRIMARY
            ]
        )
    ).astype(np.int32)
    index_arrays["e7_source_heldout__train"] = _indices(
        primary_mask & source_safe & split_masks["train"]
    )
    index_arrays["e7_source_heldout__validation"] = _indices(
        primary_mask & source_safe & split_masks["validation"]
    )
    index_arrays["e7_source_heldout__test"] = _indices(
        primary_mask & source_safe & split_masks["test"]
    )
    for family in GROUP_B:
        short = family.split("_", 1)[0].lower()
        index_arrays[f"e9_reciprocal__{short}"] = _indices(families == family)
    index_arrays["e9_reciprocal__all"] = _indices(np.isin(families, GROUP_B))
    pooled_test = index_arrays["spec__pooled_n1_n6__test"]
    index_arrays["e10_language__test"] = pooled_test.copy()
    index_arrays["e11_visual__test"] = pooled_test.copy()
    index_arrays["e12_prior__test"] = pooled_test.copy()

    variant_arrays = _variant_arrays(arrays, nuisance, schema)
    np.savez_compressed(OUT / "audit_variant_arrays.npz", **variant_arrays)
    e12_weights = {
        "balanced_labels": _prior_weights(
            pooled_test, labels, tasks.tolist(), target_positive=0.5
        ),
        "positive_prior_0_20": _prior_weights(
            pooled_test, labels, tasks.tolist(), target_positive=0.2
        ),
        "positive_prior_0_50": _prior_weights(
            pooled_test, labels, tasks.tolist(), target_positive=0.5
        ),
        "positive_prior_0_80": _prior_weights(
            pooled_test, labels, tasks.tolist(), target_positive=0.8
        ),
        "task_balanced": _prior_weights(
            pooled_test, labels, tasks.tolist(), target_positive=None
        ),
        "family_balanced": _prior_weights(
            pooled_test, labels, families.tolist(), target_positive=None
        ),
    }
    np.savez_compressed(OUT / "audit_environment_weights.npz", **e12_weights)
    np.savez_compressed(OUT / "audit_environment_indices.npz", **index_arrays)

    split_audit: dict[str, Any] = {}
    for name, spec in train_specs.items():
        partition_indices = {
            partition: index_arrays[f"spec__{name}__{partition}"]
            for partition in ("train", "validation", "test")
        }
        context_sets = {
            partition: {
                rows[int(index)]["episode"] for index in indices
            }
            for partition, indices in partition_indices.items()
        }
        source_sets = {
            partition: {
                rows[int(index)]["source_episode"] for index in indices
            }
            for partition, indices in partition_indices.items()
        }
        split_audit[name] = {
            "sample_index_overlap": {
                "train_validation": len(
                    set(partition_indices["train"])
                    & set(partition_indices["validation"])
                ),
                "train_test": len(
                    set(partition_indices["train"])
                    & set(partition_indices["test"])
                ),
                "validation_test": len(
                    set(partition_indices["validation"])
                    & set(partition_indices["test"])
                ),
            },
            "context_episode_overlap": {
                "train_validation": len(
                    context_sets["train"] & context_sets["validation"]
                ),
                "train_test": len(context_sets["train"] & context_sets["test"]),
                "validation_test": len(
                    context_sets["validation"] & context_sets["test"]
                ),
            },
            "source_episode_overlap": {
                "train_validation": len(
                    source_sets["train"] & source_sets["validation"]
                ),
                "train_test": len(source_sets["train"] & source_sets["test"]),
                "validation_test": len(
                    source_sets["validation"] & source_sets["test"]
                ),
            },
        }
        if name.startswith("task_fold_"):
            split_audit[name]["task_overlap"] = {
                left + "_" + right: len(
                    {
                        rows[int(index)]["task"]
                        for index in partition_indices[left]
                    }
                    & {
                        rows[int(index)]["task"]
                        for index in partition_indices[right]
                    }
                )
                for left, right in (
                    ("train", "validation"),
                    ("train", "test"),
                    ("validation", "test"),
                )
            }

    environments = {
        "E1": {
            "name": "within-family IID",
            "training_specs": [
                f"within_{family.split('_', 1)[0].lower()}" for family in PRIMARY
            ],
            "episode_disjoint": True,
        },
        "E2": {
            "name": "negative-generator-held-out",
            "training_specs": [
                f"leaveout_{family.split('_', 1)[0].lower()}"
                for family in LEAVE_ONE
            ],
            "N3_policy": "included in source training pools; not held out because frozen coverage is smaller",
        },
        "E3": {
            "name": "easy-to-real transfer",
            "training_specs": ["group_e_to_r"],
        },
        "E4": {
            "name": "real-to-easy transfer",
            "training_specs": ["group_r_to_e"],
            "diagnostic_only": True,
        },
        "E5": {
            "name": "action-summary-matched test",
            "base_training_spec": "pooled_n1_n6",
            "all_index_key": "e5_action_summary__all",
            "family_index_keys": {
                family: f"e5_action_summary__{family.split('_', 1)[0].lower()}"
                for family in PRIMARY
            },
        },
        "E6": {
            "name": "progress/duration/horizon-matched test",
            "base_training_spec": "pooled_n1_n6",
            "all_index_key": "e6_progress__all",
            "family_index_keys": {
                family: f"e6_progress__{family.split('_', 1)[0].lower()}"
                for family in PRIMARY
            },
        },
        "E7": {
            "name": "source-held-out test",
            "base_training_spec": "pooled_n1_n6",
            "partition_keys": {
                partition: f"e7_source_heldout__{partition}"
                for partition in ("train", "validation", "test")
            },
            "source_policy": "logged_demonstration (only available policy)",
            "source_camera": "camera_front (only available camera)",
            "source_scene": "unavailable in imported manifests",
            "source_directory": "held out jointly through imported source-episode split",
        },
        "E8": {
            "name": "task-held-out",
            "training_specs": [f"task_fold_{fold}" for fold in range(5)],
            "fold_source": str((PILOT / "task_cv_preregistered.json").resolve()),
            "fold_source_sha256": sha256_file(PILOT / "task_cv_preregistered.json"),
        },
        "E9": {
            "name": "reciprocal balance diagnostic",
            "base_training_spec": "pooled_n1_n6",
            "index_keys": {
                "N7": "e9_reciprocal__n7",
                "N8": "e9_reciprocal__n8",
                "all": "e9_reciprocal__all",
            },
            "physical_validity_interpretation_forbidden": True,
        },
        "E10": {
            "name": "language reliance",
            "base_training_spec": "pooled_n1_n6",
            "index_key": "e10_language__test",
            "variants": {
                name: f"audit_variant_arrays.npz#{name}"
                for name in (
                    "language_correct",
                    "language_shuffled",
                    "language_empty",
                    "language_task_id_only",
                )
            },
        },
        "E11": {
            "name": "visual nuisance",
            "base_training_spec": "pooled_n1_n6",
            "index_key": "e11_visual__test",
            "variants": {
                name: f"audit_variant_arrays.npz#{name}"
                for name in (
                    "visual_full",
                    "visual_current_only",
                    "visual_history_difference_only",
                    "visual_static_border_only",
                    "visual_center_workspace_only",
                    "visual_global_statistics_only",
                )
            },
            "semantic_edits": False,
            "generative_image_modification": False,
        },
        "E12": {
            "name": "label-prior and balance",
            "base_training_spec": "pooled_n1_n6",
            "index_key": "e12_prior__test",
            "weight_arrays": {
                key: f"audit_environment_weights.npz#{key}"
                for key in e12_weights
            },
        },
    }
    manifest = {
        "schema": "vsa-audit-environment-manifest-v1",
        "study_id": "VERIFIER-SHORTCUT-AUDIT-V1",
        "frozen_before_model_training": True,
        "historical_labels_only": True,
        "physical_ground_truth": False,
        "index_array_path": str(
            (OUT / "audit_environment_indices.npz").resolve()
        ),
        "index_array_sha256": sha256_file(
            OUT / "audit_environment_indices.npz"
        ),
        "variant_array_path": str((OUT / "audit_variant_arrays.npz").resolve()),
        "variant_array_sha256": sha256_file(OUT / "audit_variant_arrays.npz"),
        "weight_array_path": str(
            (OUT / "audit_environment_weights.npz").resolve()
        ),
        "weight_array_sha256": sha256_file(
            OUT / "audit_environment_weights.npz"
        ),
        "training_specs": train_specs,
        "environments": environments,
        "matching_manifests": matching_files,
        "split_audit": split_audit,
    }
    write_json(OUT / "audit_environment_manifest.json", manifest)

    environment_statistics = {
        "schema": "vsa-environment-statistics-v1",
        "train_specifications": len(train_specs),
        "index_arrays": {
            key: _count_summary(value, rows, labels)
            for key, value in sorted(index_arrays.items())
        },
        "variant_arrays": {
            key: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "finite": bool(np.isfinite(value).all()),
                "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
            }
            for key, value in variant_arrays.items()
        },
        "weight_arrays": {
            key: {
                "nonzero_samples": int(np.count_nonzero(value)),
                "mean_on_test": float(value[pooled_test].mean()),
                "minimum_on_test": float(value[pooled_test].min()),
                "maximum_on_test": float(value[pooled_test].max()),
                "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
            }
            for key, value in e12_weights.items()
        },
        "all_source_safe_for_split": bool(source_safe.all()),
        "future_information_used": False,
    }
    write_json(OUT / "environment_statistics.json", environment_statistics)
    matching_report = {
        "schema": "vsa-matching-quality-report-v1",
        "frozen_before_heldout_model_performance": True,
        "matching_records": quality_records,
        "all_pass": all(record["pass"] for record in quality_records),
        "coverage": {
            "minimum": min(record["coverage"] for record in quality_records),
            "mean": float(
                np.mean([record["coverage"] for record in quality_records])
            ),
            "maximum": max(record["coverage"] for record in quality_records),
        },
        "report_sha256_prewrite": sha256_json(quality_records),
    }
    write_json(OUT / "matching_quality_report.json", matching_report)
    append_log(
        "V5_AUDIT_ENVIRONMENTS_FROZEN",
        environments=12,
        train_specifications=len(train_specs),
        index_arrays=len(index_arrays),
        matching_manifests=len(matching_files),
        matching_all_pass=matching_report["all_pass"],
        minimum_matching_coverage=matching_report["coverage"]["minimum"],
        environment_manifest_sha256=sha256_file(
            OUT / "audit_environment_manifest.json"
        ),
        model_training_started=False,
    )
    return {
        "pass": bool(matching_report["all_pass"]),
        "environments": len(environments),
        "training_specs": len(train_specs),
        "matching_manifests": len(matching_files),
        "minimum_matching_coverage": matching_report["coverage"]["minimum"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    print(json.dumps(build(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
