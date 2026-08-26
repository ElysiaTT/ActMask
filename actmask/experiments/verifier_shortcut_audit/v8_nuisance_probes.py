"""V8 nuisance-score predictability probes for fair pooled verifiers.

The probe fitting partition is the frozen pooled-verifier validation split.
The pooled-verifier test split is touched only after every probe parameter,
normalizer, and decision threshold has been fitted.  The implementation uses
fixed-L2 linear ridge probes so that no test-set hyperparameter selection is
possible and the fitted predictions can be independently recomputed from the
saved coefficients.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .common import (
    OUT,
    SEEDS,
    STUDY_ID,
    read_json,
    sha256_file,
    sha256_json,
    write_json,
)
ARCHITECTURES = (
    "StateActionTCN",
    "TaskConditionedActionTransformer",
    "VisualStateActionTransformer",
    "ContrastiveContextActionVerifier",
    "CrossAttentionEnergyVerifier",
)
RIDGE_ALPHA = 1.0
VARIANCE_EPSILON = 1e-12
R2_GATE_THRESHOLD = 0.30


def _feature_definitions(schema: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Resolve the five requested nuisance sets from frozen schema names."""
    action_names = schema["groups"]["action"]
    context_names = schema["groups"]["context"]
    provenance_names = schema["groups"]["provenance"]

    progress = [
        index
        for index, name in enumerate(context_names)
        if name == "normalized_episode_progress"
    ]
    family = [
        index
        for index, name in enumerate(provenance_names)
        if name.startswith("family_identity[")
    ]
    source_prefixes = (
        "source_episode[",
        "source_trajectory_length",
        "source_path_hash_bucket[",
        "manifest_order",
        "pair_mining_rank",
        "pair_rank_present",
        "source_policy_",
        "camera_",
        "scene_identifier_",
    )
    source = [
        index
        for index, name in enumerate(provenance_names)
        if name.startswith(source_prefixes)
    ]
    visual_prefixes = (
        "current_rgb_mean[",
        "current_rgb_std[",
        "current_rgb_histogram[",
    )
    visual = [
        index
        for index, name in enumerate(context_names)
        if name.startswith(visual_prefixes)
    ]

    definitions = {
        "action_summaries": {
            "source_group": "action",
            "indices": list(range(len(action_names))),
            "description": (
                "All 334 frozen action-summary coordinates from V4, including "
                "moments, derivatives, endpoints, event counts, padding, and horizon."
            ),
        },
        "progress": {
            "source_group": "context",
            "indices": progress,
            "description": "Frozen normalized_episode_progress scalar only.",
        },
        "source_metadata": {
            "source_group": "provenance",
            "indices": source,
            "description": (
                "Source episode identity, source trajectory length, source-path "
                "hash buckets, manifest and pair ranks, policy/camera/scene "
                "availability fields; family and context-episode identities are "
                "explicitly excluded."
            ),
        },
        "family_identity": {
            "source_group": "provenance",
            "indices": family,
            "description": "Frozen one-hot negative-family identity coordinates.",
        },
        "visual_global_statistics": {
            "source_group": "context",
            "indices": visual,
            "description": (
                "Current-frame RGB channel means, standard deviations, and "
                "24-bin global color histogram; tokens/crops/borders are excluded."
            ),
        },
    }
    expected = {
        "action_summaries": 334,
        "progress": 1,
        "source_metadata": 205,
        "family_identity": 8,
        "visual_global_statistics": 30,
    }
    for name, definition in definitions.items():
        indices = definition["indices"]
        if len(indices) != expected[name]:
            raise RuntimeError(
                f"{name}: resolved {len(indices)} features, expected {expected[name]}"
            )
        source_names = schema["groups"][definition["source_group"]]
        definition["feature_names"] = [source_names[index] for index in indices]
        definition["feature_count"] = len(indices)
        definition["definition_sha256"] = sha256_json(
            {
                "source_group": definition["source_group"],
                "indices": indices,
                "feature_names": definition["feature_names"],
                "description": definition["description"],
            }
        )
    return definitions


def _standardize_fit(
    matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    matrix = np.asarray(matrix, dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise RuntimeError("non-finite nuisance feature in probe-fitting partition")
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    active = np.flatnonzero(scale > VARIANCE_EPSILON)
    if len(active) == 0:
        raise RuntimeError("nuisance feature set is constant on probe-fitting split")
    standardized = (matrix[:, active] - mean[active]) / scale[active]
    return standardized, mean[active], scale[active], active


def _standardize_apply(
    matrix: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    active: np.ndarray,
) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise RuntimeError("non-finite nuisance feature in held-out partition")
    return (matrix[:, active] - mean) / scale


def _fit_multioutput_ridge(
    features: np.ndarray, targets: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Fit fixed-alpha ridge with an unregularized intercept."""
    features = np.asarray(features, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    intercept = targets.mean(axis=0)
    centered_targets = targets - intercept
    gram = features.T @ features
    gram.flat[:: gram.shape[0] + 1] += RIDGE_ALPHA
    coefficient = np.linalg.solve(gram, features.T @ centered_targets)
    return coefficient, intercept


def _r2(prediction: np.ndarray, target: np.ndarray) -> float | None:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    denominator = float(np.square(target - target.mean()).sum())
    if denominator <= 1e-15:
        return None
    numerator = float(np.square(target - prediction).sum())
    return float(1.0 - numerator / denominator)


def _rank_auc(score: np.ndarray, target: np.ndarray) -> float | None:
    """Unweighted binary AUROC with half credit for score ties."""
    score = np.asarray(score, dtype=np.float64)
    target = np.asarray(target, dtype=np.int8)
    positive_count = int(np.sum(target == 1))
    negative_count = int(np.sum(target == 0))
    if positive_count == 0 or negative_count == 0:
        return None
    order = np.argsort(score, kind="mergesort")
    sorted_score = score[order]
    sorted_target = target[order]
    concordant = 0.0
    negative_before = 0
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and sorted_score[end] == sorted_score[start]:
            end += 1
        group = sorted_target[start:end]
        group_negative = int(np.sum(group == 0))
        group_positive = int(np.sum(group == 1))
        concordant += group_positive * (
            negative_before + 0.5 * group_negative
        )
        negative_before += group_negative
        start = end
    return float(concordant / (positive_count * negative_count))


def _balanced_accuracy(
    score: np.ndarray, target: np.ndarray, threshold: float
) -> float | None:
    score = np.asarray(score, dtype=np.float64)
    target = np.asarray(target, dtype=np.int8)
    positive = target == 1
    negative = target == 0
    if not positive.any() or not negative.any():
        return None
    decision = score >= threshold
    sensitivity = float(np.mean(decision[positive]))
    specificity = float(np.mean(~decision[negative]))
    return float(0.5 * (sensitivity + specificity))


def _select_stable_ba_threshold(
    score: np.ndarray, target: np.ndarray
) -> tuple[float, float]:
    """Select validation BA threshold at a stable midpoint between unique scores."""
    score = np.asarray(score, dtype=np.float32).astype(np.float64)
    target = np.asarray(target, dtype=np.int8)
    unique = np.unique(score)
    if len(unique) == 1:
        return float(unique[0]), 0.5
    margin_low = max(1.0, abs(float(unique[0]))) * 1e-5
    margin_high = max(1.0, abs(float(unique[-1]))) * 1e-5
    candidates = np.concatenate(
        (
            np.asarray([unique[0] - margin_low]),
            unique[:-1] + 0.5 * (unique[1:] - unique[:-1]),
            np.asarray([unique[-1] + margin_high]),
        )
    )
    best_key: tuple[float, float, float] | None = None
    best_threshold = 0.5
    best_ba = 0.0
    for threshold in candidates:
        ba = _balanced_accuracy(score, target, float(threshold))
        if ba is None:
            return float(threshold), 0.5
        key = (ba, -abs(float(threshold) - 0.5), -float(threshold))
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = float(threshold)
            best_ba = float(ba)
    return best_threshold, best_ba


def _safe_stats(values: list[float | None]) -> dict[str, Any]:
    finite = np.asarray(
        [value for value in values if value is not None and np.isfinite(value)],
        dtype=np.float64,
    )
    if not len(finite):
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "count": int(len(finite)),
        "mean": float(finite.mean()),
        "std": float(finite.std(ddof=0)),
        "min": float(finite.min()),
        "max": float(finite.max()),
    }


def _load_targets(
    prediction_paths: list[Path],
    expected_validation: np.ndarray,
    expected_test: np.ndarray,
    nuisance_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    validation_scores = []
    test_scores = []
    validation_decisions = []
    test_decisions = []
    manifests = []
    for path in prediction_paths:
        with np.load(path, allow_pickle=False) as stored:
            validation_index = stored["validation_index"].astype(np.int64)
            test_index = stored["test_index"].astype(np.int64)
            validation_label = stored["validation_label"].astype(np.int8)
            test_label = stored["test_label"].astype(np.int8)
            if not np.array_equal(validation_index, expected_validation):
                raise RuntimeError(f"validation indices do not match frozen split: {path}")
            if not np.array_equal(test_index, expected_test):
                raise RuntimeError(f"test indices do not match frozen split: {path}")
            if not np.array_equal(validation_label, nuisance_labels[validation_index]):
                raise RuntimeError(f"validation construction labels mismatch: {path}")
            if not np.array_equal(test_label, nuisance_labels[test_index]):
                raise RuntimeError(f"test construction labels mismatch: {path}")
            validation_score = stored["validation_calibrated_probability"].astype(
                np.float64
            )
            test_score = stored["test_calibrated_probability"].astype(np.float64)
            threshold = float(stored["threshold"][0])
            if not (
                np.isfinite(validation_score).all()
                and np.isfinite(test_score).all()
                and np.isfinite(threshold)
            ):
                raise RuntimeError(f"non-finite verifier score/threshold: {path}")
            validation_scores.append(validation_score)
            test_scores.append(test_score)
            validation_decisions.append((validation_score >= threshold).astype(np.int8))
            test_decisions.append((test_score >= threshold).astype(np.int8))
            manifests.append(
                {
                    "path": str(path.resolve()),
                    "sha256": sha256_file(path),
                    "verifier_threshold": threshold,
                    "validation_samples": int(len(validation_index)),
                    "test_samples": int(len(test_index)),
                    "validation_decision_positive_rate": float(
                        validation_decisions[-1].mean()
                    ),
                    "test_decision_positive_rate": float(test_decisions[-1].mean()),
                }
            )
    return (
        np.column_stack(validation_scores),
        np.column_stack(test_scores),
        np.column_stack(validation_decisions),
        np.column_stack(test_decisions),
        manifests,
    )


def _independent_recompute(
    rows: list[dict[str, Any]],
    definitions: dict[str, dict[str, Any]],
    nuisance: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Recompute all saved predictions/metrics from coefficients and source arrays."""
    max_score_error = 0.0
    max_decision_score_error = 0.0
    decision_mismatches = 0
    metric_mismatches = 0
    for row in rows:
        definition = definitions[row["feature_set"]]
        source = nuisance[definition["source_group"]][
            :, definition["indices"]
        ].astype(np.float64)
        path = Path(row["raw_prediction_path"])
        if sha256_file(path) != row["raw_prediction_sha256"]:
            raise RuntimeError(f"raw prediction hash changed before verification: {path}")
        with np.load(path, allow_pickle=False) as stored:
            active = stored["active_feature_local_indices"].astype(np.int64)
            mean = stored["feature_mean"].astype(np.float64)
            scale = stored["feature_scale"].astype(np.float64)
            score_coefficient = stored["score_coefficient"].astype(np.float64)
            score_intercept = float(stored["score_intercept"][0])
            decision_coefficient = stored["decision_coefficient"].astype(np.float64)
            decision_intercept = float(stored["decision_intercept"][0])
            threshold = float(stored["decision_probe_threshold"][0])

            split_records = (
                ("validation", "validation_index", row["validation"]),
                ("heldout", "heldout_index", row["heldout"]),
            )
            for prefix, index_key, expected_metrics in split_records:
                index = stored[index_key].astype(np.int64)
                x = (source[index][:, active] - mean) / scale
                score_prediction = x @ score_coefficient + score_intercept
                decision_score = x @ decision_coefficient + decision_intercept
                saved_score = stored[f"{prefix}_predicted_score"].astype(np.float64)
                saved_decision_score = stored[
                    f"{prefix}_predicted_decision_score"
                ].astype(np.float64)
                max_score_error = max(
                    max_score_error,
                    float(np.max(np.abs(score_prediction - saved_score))),
                )
                max_decision_score_error = max(
                    max_decision_score_error,
                    float(np.max(np.abs(decision_score - saved_decision_score))),
                )
                predicted_decision = (
                    saved_decision_score >= threshold
                ).astype(np.int8)
                decision_mismatches += int(
                    np.sum(
                        predicted_decision
                        != stored[f"{prefix}_predicted_decision"].astype(np.int8)
                    )
                )
                target_score = stored[f"{prefix}_target_score"].astype(np.float64)
                target_decision = stored[f"{prefix}_target_decision"].astype(np.int8)
                recomputed_metrics = {
                    "continuous_score_R2": _r2(saved_score, target_score),
                    "thresholded_decision_BA": _balanced_accuracy(
                        saved_decision_score, target_decision, threshold
                    ),
                    "thresholded_decision_AUROC": _rank_auc(
                        saved_decision_score, target_decision
                    ),
                }
                for name, recomputed in recomputed_metrics.items():
                    expected = expected_metrics[name]
                    if recomputed is None or expected is None:
                        metric_mismatches += int(recomputed is not expected)
                    elif abs(float(recomputed) - float(expected)) > 2e-6:
                        metric_mismatches += 1
    passed = bool(
        max_score_error <= 2e-6
        and max_decision_score_error <= 2e-6
        and decision_mismatches == 0
        and metric_mismatches == 0
    )
    return {
        "method": (
            "Reload every raw NPZ, reconstruct validation and held-out predictions "
            "from frozen source features plus saved normalizer/intercept/coefficients, "
            "then recompute R2/BA/AUROC."
        ),
        "files_checked": len(rows),
        "max_absolute_score_prediction_error": max_score_error,
        "max_absolute_decision_score_error": max_decision_score_error,
        "thresholded_decision_mismatches": decision_mismatches,
        "metric_mismatches_at_tolerance_2e_6": metric_mismatches,
        "tolerance": 2e-6,
        "pass": passed,
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_feature: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_architecture_feature: dict[tuple[str, str], list[dict[str, Any]]] = (
        defaultdict(list)
    )
    for row in rows:
        by_feature[row["feature_set"]].append(row)
        by_architecture_feature[(row["architecture"], row["feature_set"])].append(
            row
        )

    feature_summary = {}
    for name, selected in sorted(by_feature.items()):
        feature_summary[name] = {
            "runs": len(selected),
            "heldout_score_R2": _safe_stats(
                [row["heldout"]["continuous_score_R2"] for row in selected]
            ),
            "heldout_decision_BA": _safe_stats(
                [row["heldout"]["thresholded_decision_BA"] for row in selected]
            ),
            "heldout_decision_AUROC": _safe_stats(
                [row["heldout"]["thresholded_decision_AUROC"] for row in selected]
            ),
            "runs_R2_at_least_0_30": int(
                sum(
                    row["heldout"]["continuous_score_R2"] is not None
                    and row["heldout"]["continuous_score_R2"] >= R2_GATE_THRESHOLD
                    for row in selected
                )
            ),
        }

    architecture_feature_rows = []
    architecture_gate: dict[str, Any] = {}
    for (architecture, name), selected in sorted(by_architecture_feature.items()):
        r2_values = [
            row["heldout"]["continuous_score_R2"] for row in selected
        ]
        architecture_feature_rows.append(
            {
                "architecture": architecture,
                "feature_set": name,
                "seeds": sorted(row["seed"] for row in selected),
                "heldout_score_R2": _safe_stats(r2_values),
                "heldout_decision_BA": _safe_stats(
                    [row["heldout"]["thresholded_decision_BA"] for row in selected]
                ),
                "heldout_decision_AUROC": _safe_stats(
                    [
                        row["heldout"]["thresholded_decision_AUROC"]
                        for row in selected
                    ]
                ),
                "seeds_R2_at_least_0_30": int(
                    sum(
                        value is not None and value >= R2_GATE_THRESHOLD
                        for value in r2_values
                    )
                ),
            }
        )

    for architecture in ARCHITECTURES:
        candidates = [
            row
            for row in architecture_feature_rows
            if row["architecture"] == architecture
        ]
        best = max(
            candidates,
            key=lambda row: (
                row["seeds_R2_at_least_0_30"],
                -np.inf
                if row["heldout_score_R2"]["mean"] is None
                else row["heldout_score_R2"]["mean"],
            ),
        )
        qualifying = [
            row["feature_set"]
            for row in candidates
            if row["seeds_R2_at_least_0_30"] >= 2
        ]
        architecture_gate[architecture] = {
            "feature_sets_with_R2_at_least_0_30_in_at_least_two_seeds": qualifying,
            "qualifies_nuisance_score_probe_indicator": bool(qualifying),
            "best_feature_set_by_seed_support_then_mean_R2": best["feature_set"],
            "best_feature_set_seed_support": best["seeds_R2_at_least_0_30"],
            "best_feature_set_mean_R2": best["heldout_score_R2"]["mean"],
        }

    return {
        "by_feature_set": feature_summary,
        "by_architecture_and_feature_set": architecture_feature_rows,
        "gate_threshold_R2": R2_GATE_THRESHOLD,
        "gate_interpretation": (
            "An architecture-level nuisance-score indicator is true only when "
            "the same nuisance set reaches held-out R2 >= 0.30 for at least two "
            "of the three frozen seeds."
        ),
        "by_architecture_gate_view": architecture_gate,
        "architectures_qualifying_nuisance_score_probe_indicator": [
            architecture
            for architecture in ARCHITECTURES
            if architecture_gate[architecture][
                "qualifies_nuisance_score_probe_indicator"
            ]
        ],
    }


def run() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    schema_path = OUT / "nuisance_feature_schema.json"
    nuisance_path = OUT / "nuisance_features.npz"
    indices_path = OUT / "audit_environment_indices.npz"
    preregistration_path = OUT / "preregistered_config.json"
    schema = read_json(schema_path)
    definitions = _feature_definitions(schema)

    with np.load(nuisance_path, allow_pickle=False) as stored:
        nuisance = {name: stored[name].copy() for name in stored.files}
    with np.load(indices_path, allow_pickle=False) as stored:
        expected_validation = stored[
            "spec__pooled_n1_n6__validation"
        ].astype(np.int64)
        expected_test = stored["spec__pooled_n1_n6__test"].astype(np.int64)
    if len(np.intersect1d(expected_validation, expected_test)):
        raise RuntimeError("frozen pooled validation and test partitions overlap")
    if len(np.unique(expected_validation)) != len(expected_validation):
        raise RuntimeError("duplicate sample in pooled validation partition")
    if len(np.unique(expected_test)) != len(expected_test):
        raise RuntimeError("duplicate sample in pooled test partition")

    run_keys = [
        (architecture, seed)
        for architecture in ARCHITECTURES
        for seed in SEEDS
    ]
    prediction_paths = [
        OUT
        / "raw_verifier_predictions"
        / architecture
        / "pooled_n1_n6"
        / f"seed_{seed}.npz"
        for architecture, seed in run_keys
    ]
    missing = [str(path) for path in prediction_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing pooled verifier predictions: {missing}")

    (
        validation_scores,
        test_scores,
        validation_decisions,
        test_decisions,
        input_manifests,
    ) = _load_targets(
        prediction_paths,
        expected_validation,
        expected_test,
        nuisance["label"].astype(np.int8),
    )

    rows: list[dict[str, Any]] = []
    for feature_set, definition in definitions.items():
        source_matrix = nuisance[definition["source_group"]][
            :, definition["indices"]
        ].astype(np.float64)
        validation_x, mean, scale, active = _standardize_fit(
            source_matrix[expected_validation]
        )
        test_x = _standardize_apply(source_matrix[expected_test], mean, scale, active)

        score_coefficient, score_intercept = _fit_multioutput_ridge(
            validation_x, validation_scores
        )
        decision_coefficient, decision_intercept = _fit_multioutput_ridge(
            validation_x, validation_decisions
        )
        validation_score_prediction = (
            validation_x @ score_coefficient + score_intercept
        ).astype(np.float32)
        test_score_prediction = (
            test_x @ score_coefficient + score_intercept
        ).astype(np.float32)
        validation_decision_score = (
            validation_x @ decision_coefficient + decision_intercept
        ).astype(np.float32)
        test_decision_score = (
            test_x @ decision_coefficient + decision_intercept
        ).astype(np.float32)

        for run_index, ((architecture, seed), input_manifest) in enumerate(
            zip(run_keys, input_manifests, strict=True)
        ):
            decision_threshold, _ = _select_stable_ba_threshold(
                validation_decision_score[:, run_index],
                validation_decisions[:, run_index],
            )
            validation_decision_prediction = (
                validation_decision_score[:, run_index] >= decision_threshold
            ).astype(np.int8)
            test_decision_prediction = (
                test_decision_score[:, run_index] >= decision_threshold
            ).astype(np.int8)

            raw_path = (
                OUT
                / "nuisance_probe_raw_predictions"
                / architecture
                / f"seed_{seed}"
                / f"{feature_set}.npz"
            )
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                raw_path,
                validation_index=expected_validation.astype(np.int32),
                heldout_index=expected_test.astype(np.int32),
                validation_construction_label=nuisance["label"][
                    expected_validation
                ].astype(np.int8),
                heldout_construction_label=nuisance["label"][expected_test].astype(
                    np.int8
                ),
                validation_target_score=validation_scores[:, run_index].astype(
                    np.float32
                ),
                heldout_target_score=test_scores[:, run_index].astype(np.float32),
                validation_target_decision=validation_decisions[:, run_index].astype(
                    np.int8
                ),
                heldout_target_decision=test_decisions[:, run_index].astype(np.int8),
                validation_predicted_score=validation_score_prediction[
                    :, run_index
                ].astype(np.float32),
                heldout_predicted_score=test_score_prediction[:, run_index].astype(
                    np.float32
                ),
                validation_predicted_decision_score=validation_decision_score[
                    :, run_index
                ].astype(np.float32),
                heldout_predicted_decision_score=test_decision_score[
                    :, run_index
                ].astype(np.float32),
                validation_predicted_decision=validation_decision_prediction,
                heldout_predicted_decision=test_decision_prediction,
                verifier_threshold=np.asarray(
                    [input_manifest["verifier_threshold"]], dtype=np.float64
                ),
                decision_probe_threshold=np.asarray(
                    [decision_threshold], dtype=np.float64
                ),
                feature_indices=np.asarray(definition["indices"], dtype=np.int32),
                active_feature_local_indices=active.astype(np.int32),
                feature_mean=mean.astype(np.float64),
                feature_scale=scale.astype(np.float64),
                score_coefficient=score_coefficient[:, run_index].astype(np.float64),
                score_intercept=np.asarray(
                    [score_intercept[run_index]], dtype=np.float64
                ),
                decision_coefficient=decision_coefficient[:, run_index].astype(
                    np.float64
                ),
                decision_intercept=np.asarray(
                    [decision_intercept[run_index]], dtype=np.float64
                ),
                ridge_alpha=np.asarray([RIDGE_ALPHA], dtype=np.float64),
                feature_definition_sha256=np.asarray(
                    [definition["definition_sha256"]]
                ),
                verifier_prediction_sha256=np.asarray([input_manifest["sha256"]]),
            )

            validation_metrics = {
                "continuous_score_R2": _r2(
                    validation_score_prediction[:, run_index],
                    validation_scores[:, run_index],
                ),
                "thresholded_decision_BA": _balanced_accuracy(
                    validation_decision_score[:, run_index],
                    validation_decisions[:, run_index],
                    decision_threshold,
                ),
                "thresholded_decision_AUROC": _rank_auc(
                    validation_decision_score[:, run_index],
                    validation_decisions[:, run_index],
                ),
                "target_decision_positive_rate": float(
                    validation_decisions[:, run_index].mean()
                ),
                "samples": int(len(expected_validation)),
            }
            heldout_metrics = {
                "continuous_score_R2": _r2(
                    test_score_prediction[:, run_index], test_scores[:, run_index]
                ),
                "thresholded_decision_BA": _balanced_accuracy(
                    test_decision_score[:, run_index],
                    test_decisions[:, run_index],
                    decision_threshold,
                ),
                "thresholded_decision_AUROC": _rank_auc(
                    test_decision_score[:, run_index],
                    test_decisions[:, run_index],
                ),
                "target_decision_positive_rate": float(
                    test_decisions[:, run_index].mean()
                ),
                "samples": int(len(expected_test)),
            }
            rows.append(
                {
                    "architecture": architecture,
                    "seed": seed,
                    "feature_set": feature_set,
                    "source_group": definition["source_group"],
                    "feature_count": definition["feature_count"],
                    "active_validation_feature_count": int(len(active)),
                    "constant_validation_feature_count": int(
                        definition["feature_count"] - len(active)
                    ),
                    "ridge_alpha": RIDGE_ALPHA,
                    "decision_probe_threshold_selected_on_validation": float(
                        decision_threshold
                    ),
                    "validation": validation_metrics,
                    "heldout": heldout_metrics,
                    "raw_prediction_path": str(raw_path.resolve()),
                    "raw_prediction_sha256": sha256_file(raw_path),
                    "verifier_prediction_path": input_manifest["path"],
                    "verifier_prediction_sha256": input_manifest["sha256"],
                }
            )

    expected_probe_count = len(ARCHITECTURES) * len(SEEDS) * len(definitions)
    raw_hashes = [row["raw_prediction_sha256"] for row in rows]
    independent_recomputation = _independent_recompute(
        rows, definitions, nuisance
    )
    if not independent_recomputation["pass"]:
        raise RuntimeError(
            f"independent nuisance-probe recomputation failed: "
            f"{independent_recomputation}"
        )
    report: dict[str, Any] = {
        "schema": "vsa-nuisance-probe-report-v1",
        "study_id": STUDY_ID,
        "phase": "V8",
        "metric": "Nuisance Score Predictability",
        "status": "PASS",
        "interpretation_boundary": (
            "Predictability is associational audit evidence, not causal proof and "
            "not physical-validity evidence. Targets are verifier outputs for "
            "construction-labelled samples."
        ),
        "target_definitions": {
            "continuous_score": (
                "The fair pooled verifier's validation-calibrated probability."
            ),
            "thresholded_decision": (
                "The fair pooled verifier calibrated probability thresholded by "
                "that verifier run's own validation-selected threshold."
            ),
            "construction_label_use": (
                "Retained in raw files for alignment audit only; it is not a probe target."
            ),
        },
        "probe_definition": {
            "model": "linear L2 ridge probe with unregularized intercept",
            "continuous_fit": "least-squares ridge regression",
            "decision_fit": (
                "least-squares ridge classifier; its operating threshold is selected "
                "by exact balanced-accuracy maximization on validation only"
            ),
            "ridge_alpha": RIDGE_ALPHA,
            "hyperparameter_policy": (
                "alpha fixed globally before held-out evaluation; no sweep or "
                "held-out selection"
            ),
            "normalization": (
                "mean/std fitted on pooled verifier validation only; coordinates "
                "constant there are removed before fitting"
            ),
            "variance_epsilon": VARIANCE_EPSILON,
            "probe_train_partition": "spec__pooled_n1_n6__validation",
            "probe_heldout_partition": "spec__pooled_n1_n6__test",
            "test_tuning": False,
            "feature_sets": definitions,
        },
        "coverage": {
            "architectures_expected": list(ARCHITECTURES),
            "architectures_observed": sorted({row["architecture"] for row in rows}),
            "seeds_expected": list(SEEDS),
            "seeds_observed": sorted({row["seed"] for row in rows}),
            "feature_sets_expected": list(definitions),
            "feature_sets_observed": sorted({row["feature_set"] for row in rows}),
            "validation_samples_per_probe": int(len(expected_validation)),
            "heldout_samples_per_probe": int(len(expected_test)),
            "expected_probes": expected_probe_count,
            "observed_probes": len(rows),
            "raw_prediction_files": len(raw_hashes),
            "unique_raw_prediction_hashes": len(set(raw_hashes)),
            "complete": bool(
                len(rows) == expected_probe_count
                and len(raw_hashes) == expected_probe_count
                and len(set(raw_hashes)) == expected_probe_count
            ),
        },
        "split_integrity": {
            "validation_test_overlap": int(
                len(np.intersect1d(expected_validation, expected_test))
            ),
            "validation_unique": bool(
                len(np.unique(expected_validation)) == len(expected_validation)
            ),
            "test_unique": bool(len(np.unique(expected_test)) == len(expected_test)),
            "all_15_verifier_inputs_exactly_match_frozen_indices": True,
            "all_15_construction_label_arrays_match_nuisance_labels": True,
            "normalizer_fit_partition": "validation",
            "coefficient_fit_partition": "validation",
            "probe_threshold_selection_partition": "validation",
            "heldout_access": "evaluation only after fit",
        },
        "input_hashes": {
            "preregistered_config.json": sha256_file(preregistration_path),
            "nuisance_feature_schema.json": sha256_file(schema_path),
            "nuisance_features.npz": sha256_file(nuisance_path),
            "audit_environment_indices.npz": sha256_file(indices_path),
            "pooled_verifier_predictions": input_manifests,
        },
        "independent_recomputation": independent_recomputation,
        "per_probe": rows,
        "aggregates": _aggregate(rows),
    }
    if not report["coverage"]["complete"]:
        raise RuntimeError("nuisance probe coverage or raw-file uniqueness failed")
    report["canonical_payload_sha256"] = sha256_json(report)
    report_path = OUT / "nuisance_probe_report.json"
    write_json(report_path, report)
    (OUT / "nuisance_probe_report.sha256").write_text(
        f"{sha256_file(report_path)}  nuisance_probe_report.json\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    report = run()
    print(
        json.dumps(
            {
                "status": report["status"],
                "probes": report["coverage"]["observed_probes"],
                "raw_prediction_files": report["coverage"][
                    "raw_prediction_files"
                ],
                "qualifying_architectures": report["aggregates"][
                    "architectures_qualifying_nuisance_score_probe_indicator"
                ],
                "report": str((OUT / "nuisance_probe_report.json").resolve()),
                "report_sha256": sha256_file(OUT / "nuisance_probe_report.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
