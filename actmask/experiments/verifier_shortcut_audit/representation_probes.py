"""V9 frozen-backbone linear representation probes.

The probes in this module never instantiate, load, or update a verifier.  They
consume only the validation/test representations already emitted by V7.  A
fixed class-balanced ridge classifier is fitted on the V7 validation
partition and evaluated once on the episode-held-out V7 test partition.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .common import (
    OUT,
    SEEDS,
    append_log,
    read_json,
    read_jsonl,
    sha256_file,
    stable_int,
    write_json,
)
from .metrics import binary_metrics
from .modeling import VERIFIER_CLASSES


TRAINING_SPEC = "pooled_n1_n6"
TARGETS = (
    "negative_family",
    "action_magnitude_bin",
    "progress_bin",
    "task_identity",
    "source_episode_group",
    "visual_background_statistics_bin",
    "label",
)
RIDGE_ALPHA = 1.0
PERMUTATIONS = 999
ENCODING_EFFECT_MINIMUM = 0.05
ENCODING_P_VALUE_MAXIMUM = 0.01


def _representation_path(architecture: str, seed: int) -> Path:
    return (
        OUT
        / "verifier_representations"
        / architecture
        / TRAINING_SPEC
        / f"seed_{seed}.npz"
    )


def _run_record_path(architecture: str, seed: int) -> Path:
    return (
        OUT
        / "verifier_run_records"
        / architecture
        / TRAINING_SPEC
        / f"seed_{seed}.json"
    )


def _raw_path(architecture: str, seed: int, target: str) -> Path:
    return (
        OUT
        / "representation_probe_raw_predictions"
        / architecture
        / TRAINING_SPEC
        / f"seed_{seed}"
        / f"{target}.npz"
    )


def _softmax(score: np.ndarray) -> np.ndarray:
    shifted = score.astype(np.float64) - np.max(
        score.astype(np.float64), axis=1, keepdims=True
    )
    exponential = np.exp(shifted)
    return (exponential / exponential.sum(axis=1, keepdims=True)).astype(
        np.float32
    )


def _multiclass_metrics(
    score: np.ndarray, target: np.ndarray, class_names: list[str]
) -> dict[str, Any]:
    """Return macro-recall balanced accuracy and supporting multiclass metrics."""
    score = np.asarray(score, dtype=np.float64)
    target = np.asarray(target, dtype=np.int64)
    prediction = np.argmax(score, axis=1).astype(np.int64)
    probability = _softmax(score)
    per_class = []
    recalls = []
    f1_values = []
    auc_values = []
    for class_index, class_name in enumerate(class_names):
        positive = target == class_index
        predicted_positive = prediction == class_index
        true_positive = int(np.sum(positive & predicted_positive))
        false_positive = int(np.sum((~positive) & predicted_positive))
        false_negative = int(np.sum(positive & (~predicted_positive)))
        support = int(np.sum(positive))
        recall = true_positive / support if support else None
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if recall is not None and precision + recall > 0
            else 0.0
        )
        auc = binary_metrics(
            probability[:, class_index],
            positive.astype(np.int8),
            threshold=0.5,
        )["AUROC"]
        if recall is not None:
            recalls.append(recall)
        f1_values.append(f1)
        if auc is not None:
            auc_values.append(float(auc))
        per_class.append(
            {
                "class_index": class_index,
                "class_name": class_name,
                "support": support,
                "predicted": int(np.sum(predicted_positive)),
                "recall": None if recall is None else float(recall),
                "precision": float(precision),
                "F1": float(f1),
                "one_vs_rest_AUROC": auc,
            }
        )
    if len(recalls) != len(class_names):
        raise RuntimeError("a probe target class is absent from the held-out test")
    return {
        "samples": int(len(target)),
        "classes": len(class_names),
        "accuracy": float(np.mean(prediction == target)),
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_F1": float(np.mean(f1_values)),
        "macro_one_vs_rest_AUROC": (
            float(np.mean(auc_values)) if auc_values else None
        ),
        "chance_balanced_accuracy": 1.0 / len(class_names),
        "per_class": per_class,
    }


def _fit_class_balanced_ridge(
    validation_representation: np.ndarray,
    validation_target: np.ndarray,
    classes: int,
) -> dict[str, np.ndarray]:
    """Fit a deterministic one-vs-rest ridge probe with balanced class weight."""
    value = validation_representation.astype(np.float64)
    target = validation_target.astype(np.int64)
    mean = value.mean(axis=0)
    standard_deviation = value.std(axis=0)
    standard_deviation[standard_deviation < 1e-8] = 1.0
    normalized = (value - mean) / standard_deviation
    design = np.concatenate(
        (normalized, np.ones((len(normalized), 1), dtype=np.float64)), axis=1
    )
    counts = np.bincount(target, minlength=classes).astype(np.float64)
    if np.any(counts == 0):
        raise RuntimeError(f"probe validation target has empty classes: {counts}")
    class_weight = len(target) / (classes * counts)
    sample_weight = class_weight[target]
    weighted_design = design * np.sqrt(sample_weight)[:, None]
    one_hot = np.eye(classes, dtype=np.float64)[target]
    weighted_target = one_hot * np.sqrt(sample_weight)[:, None]
    regularizer = np.eye(design.shape[1], dtype=np.float64) * RIDGE_ALPHA
    regularizer[-1, -1] = 0.0
    coefficient_with_intercept = np.linalg.solve(
        weighted_design.T @ weighted_design + regularizer,
        weighted_design.T @ weighted_target,
    )
    return {
        "mean": mean.astype(np.float32),
        "standard_deviation": standard_deviation.astype(np.float32),
        "coefficient": coefficient_with_intercept[:-1].astype(np.float32),
        "intercept": coefficient_with_intercept[-1].astype(np.float32),
        "class_weight": class_weight.astype(np.float32),
    }


def _score(
    representation: np.ndarray, fitted: dict[str, np.ndarray]
) -> np.ndarray:
    normalized = (
        representation.astype(np.float32) - fitted["mean"]
    ) / fitted["standard_deviation"]
    return (
        normalized @ fitted["coefficient"] + fitted["intercept"]
    ).astype(np.float32)


def _permutation_test(
    target: np.ndarray,
    prediction: np.ndarray,
    classes: int,
    seed: int,
) -> tuple[np.ndarray, float]:
    """Fixed-label-count permutation null for macro-recall balanced accuracy."""
    target = np.asarray(target, dtype=np.int64)
    prediction = np.asarray(prediction, dtype=np.int64)
    observed = np.mean(
        [
            np.mean(prediction[target == class_index] == class_index)
            for class_index in range(classes)
        ]
    )
    generator = np.random.default_rng(seed)
    null = np.empty(PERMUTATIONS, dtype=np.float32)
    for permutation_index in range(PERMUTATIONS):
        shuffled = generator.permutation(target)
        null[permutation_index] = np.mean(
            [
                np.mean(prediction[shuffled == class_index] == class_index)
                for class_index in range(classes)
            ]
        )
    # Cast explicitly: NumPy may otherwise cast the scalar comparison
    # threshold down to float32 (NEP 50 weak-scalar promotion), causing a
    # rounded null value just below ``observed`` to count as an exceedance.
    p_value = (
        1.0
        + float(
            np.sum(
                null.astype(np.float64)
                >= float(observed) - 1e-12
            )
        )
    ) / (
        PERMUTATIONS + 1.0
    )
    return null, p_value


def _validation_quantile_edges(value: np.ndarray) -> list[float]:
    edges = np.quantile(
        value.astype(np.float64), [0.25, 0.50, 0.75], method="linear"
    )
    if np.any(np.diff(edges) <= 0):
        raise RuntimeError(f"non-distinct validation quantile edges: {edges}")
    return [float(edge) for edge in edges]


def _bin(value: np.ndarray, edges: list[float]) -> np.ndarray:
    return np.digitize(
        value.astype(np.float64), np.asarray(edges, dtype=np.float64), right=False
    ).astype(np.int16)


def _target_definitions(
    arrays: dict[str, np.ndarray],
    nuisance: dict[str, np.ndarray],
    schema: dict[str, Any],
    manifest: list[dict[str, Any]],
    validation_index: np.ndarray,
) -> tuple[dict[str, dict[str, Any]], dict[str, np.ndarray]]:
    family_names = read_json(OUT / "unified_audit_statistics.json")["families"]
    primary_family_names = family_names[:6]
    task_names = sorted({row["task"] for row in manifest})
    task_values = arrays["task_index"].astype(np.int16)
    if set(np.unique(task_values)) != set(range(len(task_names))):
        raise RuntimeError("task index/name mapping is not contiguous")

    action_names = schema["groups"]["action"]
    magnitude_position = action_names.index("absolute_magnitude_rms")
    action_magnitude = nuisance["action"][:, magnitude_position]
    action_edges = _validation_quantile_edges(action_magnitude[validation_index])

    context_names = schema["groups"]["context"]
    border_positions = [
        context_names.index(f"static_border_mean[{channel}]")
        for channel in range(3)
    ]
    background_statistic = nuisance["context"][:, border_positions].mean(axis=1)
    background_edges = _validation_quantile_edges(
        background_statistic[validation_index]
    )

    progress_edges = [0.25, 0.50, 0.75]
    source_episode_group = np.empty(len(manifest), dtype=np.int16)
    for index, row in enumerate(manifest):
        try:
            episode_ordinal = int(row["source_episode"].rsplit(":", 1)[1])
        except (IndexError, ValueError) as error:
            raise RuntimeError(
                f"cannot parse source episode ordinal at row {index}"
            ) from error
        source_episode_group[index] = min(episode_ordinal // 4, 3)

    target_values = {
        "negative_family": arrays["family_index"].astype(np.int16),
        "action_magnitude_bin": _bin(action_magnitude, action_edges),
        "progress_bin": _bin(arrays["normalized_progress"], progress_edges),
        "task_identity": task_values,
        "source_episode_group": source_episode_group,
        "visual_background_statistics_bin": _bin(
            background_statistic, background_edges
        ),
        "label": arrays["label"].astype(np.int16),
    }
    definitions = {
        "negative_family": {
            "selection": "construction_label == 0 only",
            "source": "unified_audit_arrays.npz#family_index",
            "classes": primary_family_names,
            "rationale": (
                "Only construction-negative samples are used because the same "
                "positive sample is intentionally repeated across negative families."
            ),
        },
        "action_magnitude_bin": {
            "selection": "all pooled N1-N6 samples",
            "source": "nuisance_features.npz#action:absolute_magnitude_rms",
            "binning": "validation-only quartile edges",
            "edges": action_edges,
            "classes": [
                "validation_quartile_1",
                "validation_quartile_2",
                "validation_quartile_3",
                "validation_quartile_4",
            ],
        },
        "progress_bin": {
            "selection": "all pooled N1-N6 samples",
            "source": "unified_audit_arrays.npz#normalized_progress",
            "binning": "fixed normalized-progress boundaries",
            "edges": progress_edges,
            "classes": ["[0,.25)", "[.25,.50)", "[.50,.75)", "[.75,1]"],
        },
        "task_identity": {
            "selection": "all pooled N1-N6 samples",
            "source": "unified_audit_arrays.npz#task_index",
            "classes": task_names,
        },
        "source_episode_group": {
            "selection": "all pooled N1-N6 samples",
            "source": "unified_audit_manifest.jsonl#source_episode",
            "binning": (
                "fixed within-task source-episode ordinal groups: "
                "0-3, 4-7, 8-11, and >=12"
            ),
            "classes": [
                "episode_ordinal_0_3",
                "episode_ordinal_4_7",
                "episode_ordinal_8_11",
                "episode_ordinal_ge_12",
            ],
            "limitation": (
                "This is a coarse collection-order provenance target; exact "
                "source-episode classes cannot be shared by episode-disjoint "
                "validation and test partitions."
            ),
        },
        "visual_background_statistics_bin": {
            "selection": "all pooled N1-N6 samples",
            "source": (
                "nuisance_features.npz#context mean of "
                "static_border_mean[0:3]"
            ),
            "binning": "validation-only quartile edges",
            "edges": background_edges,
            "classes": [
                "validation_quartile_1",
                "validation_quartile_2",
                "validation_quartile_3",
                "validation_quartile_4",
            ],
        },
        "label": {
            "selection": "all pooled N1-N6 samples",
            "source": "unified_audit_arrays.npz#label",
            "classes": ["construction_negative", "logged_continuation"],
            "physical_ground_truth": False,
        },
    }
    return definitions, target_values


def _probe_config(
    definitions: dict[str, dict[str, Any]],
    source_representations: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema": "vsa-representation-probe-config-v1",
        "study_id": "VERIFIER-SHORTCUT-AUDIT-V1",
        "phase": "V9",
        "training_specification": TRAINING_SPEC,
        "architectures": list(VERIFIER_CLASSES),
        "seeds": list(SEEDS),
        "targets": definitions,
        "estimator": {
            "type": "deterministic class-balanced one-vs-rest linear ridge",
            "ridge_alpha": RIDGE_ALPHA,
            "representation_standardization": "validation partition only",
            "fit_partition": "V7 pooled_n1_n6 validation",
            "evaluation_partition": "V7 pooled_n1_n6 held-out test",
            "hyperparameter_selection": "none; fixed alpha",
            "test_used_for_training_or_selection": False,
        },
        "linear_encoding_call": {
            "null": "999 deterministic fixed-label-count test permutations",
            "per_run_effect_minimum_over_chance_BA": ENCODING_EFFECT_MINIMUM,
            "per_run_p_value_maximum": ENCODING_P_VALUE_MAXIMUM,
            "architecture_target_call": "criterion holds for at least 2 of 3 seeds",
            "family_gate_threshold_from_V0": 0.70,
        },
        "backbone_control": {
            "verifier_backbone_frozen": True,
            "model_or_checkpoint_loaded": False,
            "gradients_computed_through_backbone": False,
            "input": "saved V7 validation/test representation arrays only",
            "source_representations": source_representations,
        },
        "claim_boundary": {
            "interpretation": "descriptive evidence of linear decodability",
            "not_causal_proof": True,
            "must_be_combined_with": [
                "generator/task/source-held-out evidence",
                "nuisance-matched evidence",
            ],
            "labels": "benchmark-construction labels, not physical ground truth",
        },
        "input_hashes": {
            "unified_arrays": sha256_file(OUT / "unified_audit_arrays.npz"),
            "unified_manifest": sha256_file(OUT / "unified_audit_manifest.jsonl"),
            "nuisance_features": sha256_file(OUT / "nuisance_features.npz"),
            "nuisance_schema": sha256_file(OUT / "nuisance_feature_schema.json"),
            "environment_indices": sha256_file(
                OUT / "audit_environment_indices.npz"
            ),
            "verifier_config": sha256_file(OUT / "verifier_config.json"),
        },
    }


def _source_representations_and_indices() -> tuple[
    list[dict[str, Any]], np.ndarray, np.ndarray
]:
    source_rows = []
    expected_validation: np.ndarray | None = None
    expected_test: np.ndarray | None = None
    for architecture in VERIFIER_CLASSES:
        for seed in SEEDS:
            path = _representation_path(architecture, seed)
            record_path = _run_record_path(architecture, seed)
            if not path.is_file() or not record_path.is_file():
                raise RuntimeError(f"missing V7 pooled representation: {path}")
            record = read_json(record_path)
            actual_hash = sha256_file(path)
            recorded_hash = record["artifact_hashes"]["representation"]
            if actual_hash != recorded_hash:
                raise RuntimeError(f"V7 representation hash mismatch: {path}")
            with np.load(path, allow_pickle=False) as stored:
                validation_index = stored["validation_index"].astype(np.int64)
                test_index = stored["test_index"].astype(np.int64)
                validation_shape = list(stored["validation_representation"].shape)
                test_shape = list(stored["test_representation"].shape)
            if expected_validation is None:
                expected_validation = validation_index
                expected_test = test_index
            elif not np.array_equal(validation_index, expected_validation) or not np.array_equal(
                test_index, expected_test
            ):
                raise RuntimeError(
                    "V7 pooled representation partitions differ across model/seed"
                )
            source_rows.append(
                {
                    "architecture": architecture,
                    "seed": seed,
                    "path": str(path.resolve()),
                    "sha256": actual_hash,
                    "run_record_path": str(record_path.resolve()),
                    "run_record_sha256": sha256_file(record_path),
                    "validation_shape": validation_shape,
                    "test_shape": test_shape,
                }
            )
    if expected_validation is None or expected_test is None:
        raise RuntimeError("no V7 pooled representations found")
    with np.load(OUT / "audit_environment_indices.npz") as stored:
        frozen_validation = stored[
            "spec__pooled_n1_n6__validation"
        ].astype(np.int64)
        frozen_test = stored["spec__pooled_n1_n6__test"].astype(np.int64)
    if not np.array_equal(expected_validation, frozen_validation) or not np.array_equal(
        expected_test, frozen_test
    ):
        raise RuntimeError("V7 representations do not use frozen V5 pooled indices")
    return source_rows, expected_validation, expected_test


def _save_raw(
    path: Path,
    *,
    validation_index: np.ndarray,
    test_index: np.ndarray,
    validation_target: np.ndarray,
    test_target: np.ndarray,
    class_names: list[str],
    validation_score: np.ndarray,
    test_score: np.ndarray,
    fitted: dict[str, np.ndarray],
    null_balanced_accuracy: np.ndarray,
    permutation_seed: int,
    target_edges: list[float],
    representation_sha256: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        validation_index=validation_index.astype(np.int32),
        test_index=test_index.astype(np.int32),
        validation_target=validation_target.astype(np.int16),
        test_target=test_target.astype(np.int16),
        class_names=np.asarray(class_names, dtype=np.str_),
        validation_score=validation_score.astype(np.float32),
        test_score=test_score.astype(np.float32),
        validation_probability=_softmax(validation_score),
        test_probability=_softmax(test_score),
        validation_prediction=np.argmax(validation_score, axis=1).astype(np.int16),
        test_prediction=np.argmax(test_score, axis=1).astype(np.int16),
        representation_mean=fitted["mean"],
        representation_standard_deviation=fitted["standard_deviation"],
        coefficient=fitted["coefficient"],
        intercept=fitted["intercept"],
        class_weight=fitted["class_weight"],
        ridge_alpha=np.asarray([RIDGE_ALPHA], dtype=np.float32),
        target_edges=np.asarray(target_edges, dtype=np.float64),
        permutation_seed=np.asarray([permutation_seed], dtype=np.int64),
        permutation_balanced_accuracy=null_balanced_accuracy.astype(np.float32),
        source_representation_sha256=np.asarray(
            [representation_sha256], dtype=np.str_
        ),
    )


def _verify_raw_row(row: dict[str, Any]) -> None:
    path = Path(row["raw_prediction_path"])
    if sha256_file(path) != row["raw_prediction_sha256"]:
        raise RuntimeError(f"raw probe hash mismatch: {path}")
    representation_path = Path(row["source_representation_path"])
    if sha256_file(representation_path) != row["source_representation_sha256"]:
        raise RuntimeError(f"source representation hash changed: {representation_path}")
    with np.load(path, allow_pickle=False) as raw, np.load(
        representation_path, allow_pickle=False
    ) as representation:
        validation_index = representation["validation_index"].astype(np.int64)
        test_index = representation["test_index"].astype(np.int64)
        if not np.array_equal(
            validation_index[np.isin(validation_index, raw["validation_index"])],
            raw["validation_index"],
        ):
            raise RuntimeError(f"validation raw indices are not ordered subset: {path}")
        if not np.array_equal(
            test_index[np.isin(test_index, raw["test_index"])], raw["test_index"]
        ):
            raise RuntimeError(f"test raw indices are not ordered subset: {path}")
        validation_mask = np.isin(validation_index, raw["validation_index"])
        test_mask = np.isin(test_index, raw["test_index"])
        fitted = {
            "mean": raw["representation_mean"],
            "standard_deviation": raw["representation_standard_deviation"],
            "coefficient": raw["coefficient"],
            "intercept": raw["intercept"],
        }
        validation_score = _score(
            representation["validation_representation"][validation_mask],
            fitted,
        )
        test_score = _score(
            representation["test_representation"][test_mask], fitted
        )
        if not np.allclose(
            validation_score, raw["validation_score"], rtol=1e-5, atol=1e-5
        ) or not np.allclose(
            test_score, raw["test_score"], rtol=1e-5, atol=1e-5
        ):
            raise RuntimeError(f"linear scores do not independently recompute: {path}")
        metrics = _multiclass_metrics(
            test_score,
            raw["test_target"],
            [str(value) for value in raw["class_names"]],
        )
        expected = row["test_metrics"]
        for metric in ("accuracy", "balanced_accuracy", "macro_F1"):
            if abs(metrics[metric] - expected[metric]) > 1e-10:
                raise RuntimeError(f"{metric} does not recompute for {path}")
        observed = metrics["balanced_accuracy"]
        null = raw["permutation_balanced_accuracy"]
        p_value = (
            1.0
            + float(
                np.sum(
                    null.astype(np.float64)
                    >= float(observed) - 1e-12
                )
            )
        ) / (
            len(null) + 1.0
        )
        if abs(p_value - row["permutation_p_value"]) > 1e-12:
            raise RuntimeError(f"permutation p-value does not recompute for {path}")


def run() -> dict[str, Any]:
    source_rows, validation_index, test_index = (
        _source_representations_and_indices()
    )
    with np.load(OUT / "unified_audit_arrays.npz", allow_pickle=False) as stored:
        arrays = {name: stored[name].copy() for name in stored.files}
    with np.load(OUT / "nuisance_features.npz", allow_pickle=False) as stored:
        nuisance = {name: stored[name].copy() for name in stored.files}
    schema = read_json(OUT / "nuisance_feature_schema.json")
    manifest = read_jsonl(OUT / "unified_audit_manifest.jsonl")
    if len(manifest) != len(arrays["label"]):
        raise RuntimeError("unified manifest/array length mismatch")
    definitions, target_values = _target_definitions(
        arrays, nuisance, schema, manifest, validation_index
    )

    config = _probe_config(definitions, source_rows)
    config_path = OUT / "representation_probe_config.json"
    # This file is materialized before any probe sees held-out test performance.
    write_json(config_path, config)

    rows = []
    for architecture in VERIFIER_CLASSES:
        for seed in SEEDS:
            representation_path = _representation_path(architecture, seed)
            representation_sha256 = sha256_file(representation_path)
            with np.load(representation_path, allow_pickle=False) as stored:
                source_validation_index = stored["validation_index"].astype(np.int64)
                source_test_index = stored["test_index"].astype(np.int64)
                validation_representation = stored[
                    "validation_representation"
                ].astype(np.float32)
                test_representation = stored["test_representation"].astype(np.float32)
            for target_name in TARGETS:
                definition = definitions[target_name]
                if target_name == "negative_family":
                    validation_mask = arrays["label"][source_validation_index] == 0
                    test_mask = arrays["label"][source_test_index] == 0
                else:
                    validation_mask = np.ones(
                        len(source_validation_index), dtype=bool
                    )
                    test_mask = np.ones(len(source_test_index), dtype=bool)
                selected_validation_index = source_validation_index[validation_mask]
                selected_test_index = source_test_index[test_mask]
                validation_target = target_values[target_name][
                    selected_validation_index
                ].astype(np.int16)
                test_target = target_values[target_name][selected_test_index].astype(
                    np.int16
                )
                class_names = list(definition["classes"])
                classes = len(class_names)
                if np.any(validation_target < 0) or np.any(
                    validation_target >= classes
                ):
                    raise RuntimeError(f"{target_name}: invalid validation class")
                if np.any(test_target < 0) or np.any(test_target >= classes):
                    raise RuntimeError(f"{target_name}: invalid test class")
                fitted = _fit_class_balanced_ridge(
                    validation_representation[validation_mask],
                    validation_target,
                    classes,
                )
                validation_score = _score(
                    validation_representation[validation_mask], fitted
                )
                test_score = _score(test_representation[test_mask], fitted)
                validation_metrics = _multiclass_metrics(
                    validation_score, validation_target, class_names
                )
                test_metrics = _multiclass_metrics(
                    test_score, test_target, class_names
                )
                permutation_seed = stable_int(
                    f"V9|{architecture}|{seed}|{target_name}"
                )
                null, p_value = _permutation_test(
                    test_target,
                    np.argmax(test_score, axis=1),
                    classes,
                    permutation_seed,
                )
                effect = (
                    test_metrics["balanced_accuracy"]
                    - test_metrics["chance_balanced_accuracy"]
                )
                linearly_encoded_run = (
                    effect >= ENCODING_EFFECT_MINIMUM
                    and p_value <= ENCODING_P_VALUE_MAXIMUM
                )
                raw_path = _raw_path(architecture, seed, target_name)
                _save_raw(
                    raw_path,
                    validation_index=selected_validation_index,
                    test_index=selected_test_index,
                    validation_target=validation_target,
                    test_target=test_target,
                    class_names=class_names,
                    validation_score=validation_score,
                    test_score=test_score,
                    fitted=fitted,
                    null_balanced_accuracy=null,
                    permutation_seed=permutation_seed,
                    target_edges=list(definition.get("edges", [])),
                    representation_sha256=representation_sha256,
                )
                rows.append(
                    {
                        "architecture": architecture,
                        "seed": seed,
                        "training_specification": TRAINING_SPEC,
                        "target": target_name,
                        "classes": class_names,
                        "validation_samples": int(len(validation_target)),
                        "test_samples": int(len(test_target)),
                        "validation_class_counts": {
                            class_names[class_index]: int(count)
                            for class_index, count in enumerate(
                                np.bincount(
                                    validation_target, minlength=classes
                                )
                            )
                        },
                        "test_class_counts": {
                            class_names[class_index]: int(count)
                            for class_index, count in enumerate(
                                np.bincount(test_target, minlength=classes)
                            )
                        },
                        "validation_fit_metrics": validation_metrics,
                        "test_metrics": test_metrics,
                        "balanced_accuracy_over_chance": float(effect),
                        "permutation_p_value": float(p_value),
                        "linearly_encoded_run": bool(linearly_encoded_run),
                        "family_gate_BA_ge_0_70": (
                            bool(test_metrics["balanced_accuracy"] >= 0.70)
                            if target_name == "negative_family"
                            else None
                        ),
                        "source_representation_path": str(
                            representation_path.resolve()
                        ),
                        "source_representation_sha256": representation_sha256,
                        "raw_prediction_path": str(raw_path.resolve()),
                        "raw_prediction_sha256": sha256_file(raw_path),
                        "backbone_frozen": True,
                        "test_used_for_training_or_selection": False,
                    }
                )
                print(
                    "V9 "
                    f"architecture={architecture} seed={seed} "
                    f"target={target_name} "
                    f"test_BA={test_metrics['balanced_accuracy']:.4f} "
                    f"chance={test_metrics['chance_balanced_accuracy']:.4f} "
                    f"p={p_value:.4f}",
                    flush=True,
                )

    for row in rows:
        _verify_raw_row(row)

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["architecture"], row["target"])].append(row)
    aggregate = []
    for (architecture, target_name), group in sorted(grouped.items()):
        balanced_accuracy = [
            row["test_metrics"]["balanced_accuracy"] for row in group
        ]
        encoded_seeds = [
            row["seed"] for row in group if row["linearly_encoded_run"]
        ]
        family_gate_seeds = [
            row["seed"]
            for row in group
            if row["family_gate_BA_ge_0_70"] is True
        ]
        aggregate.append(
            {
                "architecture": architecture,
                "target": target_name,
                "seeds": [row["seed"] for row in group],
                "test_balanced_accuracy_mean": float(
                    np.mean(balanced_accuracy)
                ),
                "test_balanced_accuracy_standard_deviation": float(
                    np.std(balanced_accuracy)
                ),
                "test_balanced_accuracy_minimum": float(
                    np.min(balanced_accuracy)
                ),
                "test_balanced_accuracy_maximum": float(
                    np.max(balanced_accuracy)
                ),
                "encoded_seeds": encoded_seeds,
                "linearly_encoded": len(encoded_seeds) >= 2,
                "family_gate_BA_ge_0_70_seeds": family_gate_seeds,
                "family_gate_holds_across_at_least_two_seeds": (
                    len(family_gate_seeds) >= 2
                    if target_name == "negative_family"
                    else None
                ),
            }
        )
    target_summary = []
    for target_name in TARGETS:
        group = [row for row in aggregate if row["target"] == target_name]
        target_summary.append(
            {
                "target": target_name,
                "architecture_count": len(group),
                "architectures_linearly_encoding": [
                    row["architecture"] for row in group if row["linearly_encoded"]
                ],
                "architecture_mean_BA_range": [
                    float(
                        min(row["test_balanced_accuracy_mean"] for row in group)
                    ),
                    float(
                        max(row["test_balanced_accuracy_mean"] for row in group)
                    ),
                ],
            }
        )
    family_rows = [
        row for row in aggregate if row["target"] == "negative_family"
    ]
    summary = {
        "schema": "vsa-representation-probe-summary-v1",
        "study_id": "VERIFIER-SHORTCUT-AUDIT-V1",
        "phase": "V9",
        "complete": len(rows) == len(VERIFIER_CLASSES) * len(SEEDS) * len(TARGETS),
        "architectures": len(VERIFIER_CLASSES),
        "seeds": list(SEEDS),
        "targets": list(TARGETS),
        "probe_runs": len(rows),
        "source_training_specification": TRAINING_SPEC,
        "fit_partition": "validation",
        "evaluation_partition": "held-out test",
        "backbone_frozen": True,
        "test_used_for_training_or_selection": False,
        "independent_raw_recompute_passed": True,
        "rows": rows,
        "architecture_target_summary": aggregate,
        "target_summary": target_summary,
        "family_representation_probe_gate": {
            "threshold": 0.70,
            "architectures_meeting_threshold_across_at_least_two_seeds": [
                row["architecture"]
                for row in family_rows
                if row["family_gate_holds_across_at_least_two_seeds"]
            ],
            "per_architecture": [
                {
                    "architecture": row["architecture"],
                    "mean_BA": row["test_balanced_accuracy_mean"],
                    "seeds_meeting_threshold": row[
                        "family_gate_BA_ge_0_70_seeds"
                    ],
                }
                for row in family_rows
            ],
        },
        "claim_boundary": {
            "probe_accuracy_is_causal_proof": False,
            "interpretation": "linear decodability in frozen V7 representations",
            "must_be_combined_with_heldout_and_matched_evidence": True,
            "construction_labels_are_physical_ground_truth": False,
        },
        "config_path": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path),
        "raw_prediction_files": len(rows),
        "raw_prediction_bytes": int(
            sum(Path(row["raw_prediction_path"]).stat().st_size for row in rows)
        ),
    }
    summary_path = OUT / "representation_probe_summary.json"
    write_json(summary_path, summary)
    append_log(
        "V9_REPRESENTATION_PROBES_COMPLETE",
        architectures=len(VERIFIER_CLASSES),
        seeds=list(SEEDS),
        targets=len(TARGETS),
        probe_runs=len(rows),
        backbone_frozen=True,
        validation_fit_test_heldout=True,
        independent_raw_recompute_passed=True,
        config_sha256=sha256_file(config_path),
        summary_sha256=sha256_file(summary_path),
    )
    return {
        "complete": summary["complete"],
        "probe_runs": len(rows),
        "raw_prediction_files": len(rows),
        "independent_raw_recompute_passed": True,
        "family_gate_architectures": summary[
            "family_representation_probe_gate"
        ]["architectures_meeting_threshold_across_at_least_two_seeds"],
        "summary_path": str(summary_path.resolve()),
    }


def verify() -> dict[str, Any]:
    summary_path = OUT / "representation_probe_summary.json"
    summary = read_json(summary_path)
    if not summary.get("complete"):
        raise RuntimeError("V9 summary is not complete")
    if len(summary["rows"]) != len(VERIFIER_CLASSES) * len(SEEDS) * len(TARGETS):
        raise RuntimeError("V9 summary row count mismatch")
    for row in summary["rows"]:
        _verify_raw_row(row)
    return {
        "complete": True,
        "probe_runs_verified": len(summary["rows"]),
        "raw_prediction_hashes_and_metrics_recomputed": True,
        "summary_sha256": sha256_file(summary_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true")
    arguments = parser.parse_args()
    result = verify() if arguments.verify_only else run()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
