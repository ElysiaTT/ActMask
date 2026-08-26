"""Final frozen IID confirmation and independent raw-metric recomputation."""
from __future__ import annotations

import dataclasses
import json
import math
from collections import defaultdict
from typing import Any

import numpy as np
import torch

from actmask.experiments.verifier_shortcut_audit.modeling import VERIFIER_CLASSES

from .common import (
    ARCHITECTURES,
    EASY_FAMILIES,
    OUT,
    SEEDS,
    SPECS,
    STUDY_ID,
    append_log,
    read_json,
    read_jsonl,
    sha256_file,
    sha256_json,
    verify_bound_source_snapshot,
    verify_payload_sha256,
    write_json,
)
from .diagnostics import _train_with_fallback, _verify_frozen_implementation
from .training import (
    SpecData,
    StudyStore,
    output_paths,
    protocol_for,
    resource_status,
)


GROUP = "final_confirmation"
RESULT = OUT / "iid_learnability_results.json"
METRIC_TOLERANCE = 1e-12
FINAL_BA_GATE = 0.65


def _sigmoid(value: np.ndarray) -> np.ndarray:
    """Independent stable sigmoid; intentionally does not call shared metrics."""
    array = np.asarray(value, dtype=np.float64)
    result = np.empty_like(array)
    positive = array >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-array[positive]))
    exponential = np.exp(array[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def _bce(logits: np.ndarray, labels: np.ndarray) -> float:
    value = np.asarray(logits, dtype=np.float64)
    target = np.asarray(labels, dtype=np.float64)
    return float(
        np.mean(
            np.maximum(value, 0.0)
            - value * target
            + np.log1p(np.exp(-np.abs(value)))
        )
    )


def _fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Independently reproduce the frozen validation-only temperature search."""
    value = np.asarray(logits, dtype=np.float64)
    target = np.asarray(labels, dtype=np.float64)
    temperatures = np.exp(np.linspace(-3.0, 3.0, 1201))
    scaled = value[:, None] / temperatures[None, :]
    losses = (
        np.maximum(scaled, 0.0)
        - scaled * target[:, None]
        + np.log1p(np.exp(-np.abs(scaled)))
    ).mean(0)
    return float(temperatures[int(np.argmin(losses))])


def _select_threshold(
    probability: np.ndarray, labels: np.ndarray
) -> tuple[float, float]:
    """Independent exact BA sweep with the preregistered deterministic tie break."""
    score = np.asarray(probability, dtype=np.float64)
    target = np.asarray(labels, dtype=np.int8)
    positives = max(1, int(np.sum(target == 1)))
    negatives = max(1, int(np.sum(target == 0)))
    best_key: tuple[float, float, float] | None = None
    best_threshold = 0.5
    best_ba = 0.0

    def consider(
        threshold: float, true_positive: int, false_positive: int
    ) -> None:
        nonlocal best_key, best_threshold, best_ba
        true_negative = negatives - false_positive
        balanced_accuracy = 0.5 * (
            true_positive / positives + true_negative / negatives
        )
        key = (
            float(balanced_accuracy),
            -abs(float(threshold) - 0.5),
            -float(threshold),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = float(threshold)
            best_ba = float(balanced_accuracy)

    order = np.argsort(-score, kind="mergesort")
    sorted_score = score[order]
    sorted_target = target[order]
    true_positive = 0
    false_positive = 0
    consider(1.0, true_positive, false_positive)
    start = 0
    while start < len(order):
        current = float(sorted_score[start])
        consider(
            float(np.nextafter(current, np.inf)),
            true_positive,
            false_positive,
        )
        end = start + 1
        while end < len(order) and sorted_score[end] == sorted_score[start]:
            end += 1
        group = sorted_target[start:end]
        true_positive += int(np.sum(group == 1))
        false_positive += int(np.sum(group == 0))
        consider(current, true_positive, false_positive)
        start = end
    consider(0.0, true_positive, false_positive)
    return best_threshold, best_ba


def _rank_auc(probability: np.ndarray, labels: np.ndarray) -> float | None:
    score = np.asarray(probability, dtype=np.float64)
    target = np.asarray(labels, dtype=np.int8)
    positives = int(np.sum(target == 1))
    negatives = int(np.sum(target == 0))
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(score, kind="mergesort")
    score = score[order]
    target = target[order]
    concordant = 0.0
    negatives_before = 0
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and score[end] == score[start]:
            end += 1
        group = target[start:end]
        group_negative = int(np.sum(group == 0))
        group_positive = int(np.sum(group == 1))
        concordant += group_positive * (
            negatives_before + 0.5 * group_negative
        )
        negatives_before += group_negative
        start = end
    return float(concordant / (positives * negatives))


def _average_precision(
    probability: np.ndarray, labels: np.ndarray
) -> float | None:
    score = np.asarray(probability, dtype=np.float64)
    target = np.asarray(labels, dtype=np.int8)
    positives = int(np.sum(target == 1))
    if positives == 0:
        return None
    order = np.argsort(-score, kind="mergesort")
    ordered = target[order]
    true_positive = np.cumsum(ordered == 1)
    precision = true_positive / np.arange(1, len(ordered) + 1)
    return float(np.sum(precision[ordered == 1]) / positives)


def _binary_metrics(
    probability: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    bins: int = 15,
) -> dict[str, Any]:
    score = np.asarray(probability, dtype=np.float64)
    target = np.asarray(labels, dtype=np.int8)
    valid = np.isfinite(score) & np.isin(target, (0, 1))
    score = score[valid]
    target = target[valid]
    prediction = score >= threshold
    positive = target == 1
    negative = target == 0
    positives = max(int(np.sum(positive)), 1)
    negatives = max(int(np.sum(negative)), 1)
    true_positive_rate = float(np.sum(prediction & positive) / positives)
    true_negative_rate = float(np.sum((~prediction) & negative) / negatives)
    false_accept_rate = float(np.sum(prediction & negative) / negatives)
    false_reject_rate = float(np.sum((~prediction) & positive) / positives)

    calibration_bins = []
    ece = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for index in range(bins):
        if index == bins - 1:
            selected = (score >= edges[index]) & (score <= edges[index + 1])
        else:
            selected = (score >= edges[index]) & (score < edges[index + 1])
        count = int(np.sum(selected))
        if count == 0:
            continue
        confidence = float(np.mean(score[selected]))
        positive_rate = float(np.mean(target[selected]))
        contribution = float(count / len(target) * abs(confidence - positive_rate))
        ece += contribution
        calibration_bins.append(
            {
                "lower": float(edges[index]),
                "upper": float(edges[index + 1]),
                "weight": float(count),
                "confidence": confidence,
                "positive_rate": positive_rate,
                "ece_contribution": contribution,
            }
        )
    return {
        "samples": int(len(target)),
        "effective_weight": float(len(target)),
        "threshold": float(threshold),
        "balanced_accuracy": float(
            0.5 * (true_positive_rate + true_negative_rate)
        ),
        "AUROC": _rank_auc(score, target),
        "AUPRC": _average_precision(score, target),
        "ECE": float(ece),
        "Brier": float(np.mean(np.square(score - target))),
        "false_accept_rate": false_accept_rate,
        "false_reject_rate": false_reject_rate,
        "calibration_bins": calibration_bins,
    }


def _evaluate_logits(
    logits: np.ndarray, labels: np.ndarray, threshold: float = 0.5
) -> dict[str, Any]:
    return {
        **_binary_metrics(_sigmoid(logits), labels, threshold),
        "BCE": _bce(logits, labels),
        "logit_mean": float(np.mean(logits)),
        "logit_standard_deviation": float(np.std(logits)),
        "logit_minimum": float(np.min(logits)),
        "logit_maximum": float(np.max(logits)),
    }


def _compare_nested(
    expected: Any,
    actual: Any,
    path: str,
) -> tuple[int, list[dict[str, Any]]]:
    """Compare complete nested metric payloads and count scalar checks."""
    mismatches: list[dict[str, Any]] = []
    if isinstance(expected, dict) and isinstance(actual, dict):
        expected_keys = set(expected)
        actual_keys = set(actual)
        if expected_keys != actual_keys:
            mismatches.append(
                {
                    "field": path,
                    "reason": "dictionary_keys",
                    "expected": sorted(expected_keys),
                    "actual": sorted(actual_keys),
                }
            )
        count = 0
        for key in sorted(expected_keys & actual_keys):
            checked, differences = _compare_nested(
                expected[key], actual[key], f"{path}.{key}"
            )
            count += checked
            mismatches.extend(differences)
        return count, mismatches
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            mismatches.append(
                {
                    "field": path,
                    "reason": "list_length",
                    "expected": len(expected),
                    "actual": len(actual),
                }
            )
        count = 0
        for index, (left, right) in enumerate(zip(expected, actual, strict=False)):
            checked, differences = _compare_nested(
                left, right, f"{path}[{index}]"
            )
            count += checked
            mismatches.extend(differences)
        return count, mismatches
    if expected is None or actual is None:
        equal = expected is None and actual is None
    elif isinstance(expected, bool) or isinstance(actual, bool):
        equal = isinstance(expected, bool) and isinstance(actual, bool) and expected == actual
    elif isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        left = float(expected)
        right = float(actual)
        equal = (
            math.isfinite(left)
            and math.isfinite(right)
            and abs(left - right) <= METRIC_TOLERANCE
        )
    else:
        equal = expected == actual
    if not equal:
        mismatches.append(
            {
                "field": path,
                "reason": "value",
                "expected": expected,
                "actual": actual,
            }
        )
    return 1, mismatches


def _diagnostics_complete() -> dict[str, Any]:
    paths = {
        "memorization": OUT / "memorization_report.json",
        "sanity": OUT / "sanity_control_report.json",
        "gradient": OUT / "gradient_flow_report.json",
        "optimization": OUT / "optimization_diagnostics.json",
        "representation": OUT / "representation_mismatch_report.json",
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise RuntimeError(f"learned diagnostics are incomplete: {missing}")
    reports = {name: read_json(path) for name, path in paths.items()}
    completeness = {
        "memorization_runs": (
            reports["memorization"].get("tiny_runs") == 20
            and reports["memorization"].get("full_runs") == 20
        ),
        "sanity_runs": all(
            len(reports["sanity"].get(name, {}).get("rows", [])) == 5
            for name in (
                "positive_observable_signal",
                "shuffled_label_negative",
                "constant_input_negative",
            )
        ),
        "gradient_runs": reports["gradient"].get("runs") == 20,
        "alternate_runs": (
            reports["optimization"]
            .get("alternate_schedule", {})
            .get("run_count")
            == 60
        ),
        "representation_control_runs": (
            reports["representation"]
            .get("diagnostic_controls", {})
            .get("run_count")
            == 24
        ),
    }
    events = {row["event"] for row in read_jsonl(OUT / "run_log.jsonl")}
    completeness["completion_event"] = (
        "B_E_LEARNED_DIAGNOSTICS_COMPLETE" in events
    )
    if not all(completeness.values()):
        raise RuntimeError(
            f"learned diagnostic completion checks failed: {completeness}"
        )
    return {
        "pass": True,
        **completeness,
        "report_hashes": {
            name: sha256_file(path) for name, path in paths.items()
        },
    }


def _verify_record_artifacts(
    record: dict[str, Any],
    architecture: str,
    spec: str,
    seed: int,
) -> dict[str, Any]:
    expected_paths = output_paths(GROUP, architecture, spec, seed)
    issues = []
    if not verify_payload_sha256(record):
        issues.append(
            {
                "field": "record_payload_sha256",
                "expected": "valid self-hash",
                "actual": record.get("record_payload_sha256"),
            }
        )
    expected_identity = {
        "group": GROUP,
        "model": architecture,
        "spec": spec,
        "seed": seed,
        "complete": True,
        "heldout_test_used_for_selection": False,
        "physical_ground_truth": False,
    }
    for field, expected in expected_identity.items():
        if record.get(field) != expected:
            issues.append(
                {
                    "field": field,
                    "expected": expected,
                    "actual": record.get(field),
                }
            )
    expected_protocol = dataclasses.asdict(protocol_for("final_original"))
    if record.get("requested_protocol") != expected_protocol:
        issues.append(
            {
                "field": "requested_protocol",
                "expected": expected_protocol,
                "actual": record.get("requested_protocol"),
            }
        )
    logical_identity = record.get("logical_identity")
    if not isinstance(logical_identity, dict):
        issues.append(
            {
                "field": "logical_identity",
                "expected": "dictionary",
                "actual": type(logical_identity).__name__,
            }
        )
    else:
        logical_payload = dict(logical_identity)
        recorded_logical_hash = logical_payload.pop("identity_sha256", None)
        recomputed_logical_hash = sha256_json(logical_payload)
        if (
            recorded_logical_hash != recomputed_logical_hash
            or record.get("logical_identity_sha256")
            != recomputed_logical_hash
        ):
            issues.append(
                {
                    "field": "logical_identity_sha256",
                    "expected": recomputed_logical_hash,
                    "actual": record.get("logical_identity_sha256"),
                }
            )
        logical_expected = {
            "config_sha256": sha256_file(
                OUT / "preregistered_config.json"
            ),
            "model": architecture,
            "group": GROUP,
            "spec": spec,
            "seed": seed,
            "protocol": expected_protocol,
        }
        for field, expected in logical_expected.items():
            if logical_identity.get(field) != expected:
                issues.append(
                    {
                        "field": f"logical_identity.{field}",
                        "expected": expected,
                        "actual": logical_identity.get(field),
                    }
                )
    allowed_batches = read_json(OUT / "preregistered_config.json")[
        "OOM_fallback"
    ]["final_original_batch_sequence"]
    for field, expected in expected_protocol.items():
        actual = record.get("protocol", {}).get(field)
        if field == "batch_size":
            valid = actual in allowed_batches
        else:
            valid = actual == expected
        if not valid:
            issues.append(
                {
                    "field": f"protocol.{field}",
                    "expected": (
                        allowed_batches if field == "batch_size" else expected
                    ),
                    "actual": actual,
                }
            )
    hash_failures = []
    path_failures = []
    for name in ("checkpoint", "prediction", "curve", "runtime"):
        path = expected_paths[name]
        recorded_path = record.get("artifact_paths", {}).get(name)
        if recorded_path != str(path.resolve()):
            path_failures.append(
                {
                    "artifact": name,
                    "expected": str(path.resolve()),
                    "actual": recorded_path,
                }
            )
        expected_hash = record.get("artifact_hashes", {}).get(name)
        actual_hash = sha256_file(path) if path.is_file() else None
        if expected_hash is None or actual_hash != expected_hash:
            hash_failures.append(
                {
                    "artifact": name,
                    "path": str(path.resolve()),
                    "expected": expected_hash,
                    "actual": actual_hash,
                }
            )
    curve_path = expected_paths["curve"]
    if curve_path.is_file():
        curve_payload = read_json(curve_path)
        best_value: float | None = None
        independently_selected_epoch = -1
        for row in curve_payload.get("curve", []):
            value = float(
                row["validation_balanced_accuracy_optimal_threshold"]
            )
            if best_value is None or value > best_value + 1e-12:
                best_value = value
                independently_selected_epoch = int(row["epoch"])
        if independently_selected_epoch != record.get("best_epoch"):
            issues.append(
                {
                    "field": "best_epoch",
                    "expected_from_curve": independently_selected_epoch,
                    "actual": record.get("best_epoch"),
                }
            )
    return {
        "pass": not issues and not hash_failures and not path_failures,
        "identity_or_protocol_issues": issues,
        "path_failures": path_failures,
        "hash_failures": hash_failures,
    }


def _recompute_run(
    record: dict[str, Any],
    store: StudyStore,
    architecture: str,
    family: str,
    spec: str,
    seed: int,
) -> dict[str, Any]:
    artifact = _verify_record_artifacts(record, architecture, spec, seed)
    path = output_paths(GROUP, architecture, spec, seed)["prediction"]
    with np.load(path, allow_pickle=False) as stored:
        required = {
            "train_index",
            "train_label",
            "train_logit",
            "validation_index",
            "validation_label",
            "validation_logit",
            "test_index",
            "test_label",
            "test_logit",
            "temperature",
            "threshold",
        }
        missing = sorted(required - set(stored.files))
        if missing:
            raise RuntimeError(f"missing raw fields in {path}: {missing}")
        raw = {name: stored[name].copy() for name in required}

    train, validation, test = store.spec_indices(spec)
    expected_indices = {
        "train": train,
        "validation": validation,
        "test": test,
    }
    integrity_issues = []
    for partition, expected_index in expected_indices.items():
        index = raw[f"{partition}_index"].astype(np.int64)
        label = raw[f"{partition}_label"].astype(np.int8)
        logit = raw[f"{partition}_logit"]
        if not np.array_equal(index, expected_index):
            integrity_issues.append(
                {
                    "partition": partition,
                    "field": "index",
                    "expected_samples": int(len(expected_index)),
                    "actual_samples": int(len(index)),
                }
            )
        if not np.array_equal(
            label, store.labels[expected_index].astype(np.int8)
        ):
            integrity_issues.append(
                {"partition": partition, "field": "label"}
            )
        if logit.shape != (len(expected_index),) or not np.isfinite(logit).all():
            integrity_issues.append(
                {
                    "partition": partition,
                    "field": "logit_shape_or_finiteness",
                    "shape": list(logit.shape),
                    "finite": bool(np.isfinite(logit).all()),
                }
            )
    if integrity_issues:
        raise RuntimeError(
            f"raw prediction integrity failed for {architecture}/{spec}/{seed}: "
            f"{integrity_issues}"
        )

    train_label = raw["train_label"].astype(np.int8)
    validation_label = raw["validation_label"].astype(np.int8)
    test_label = raw["test_label"].astype(np.int8)
    train_logit = raw["train_logit"].astype(np.float32)
    validation_logit = raw["validation_logit"].astype(np.float32)
    test_logit = raw["test_logit"].astype(np.float32)
    temperature = _fit_temperature(validation_logit, validation_label)
    validation_calibrated = _sigmoid(
        validation_logit / temperature
    ).astype(np.float32)
    threshold, _ = _select_threshold(
        validation_calibrated, validation_label
    )
    test_calibrated = _sigmoid(test_logit / temperature).astype(np.float32)
    oracle_threshold, oracle_ba = _select_threshold(
        test_calibrated, test_label
    )

    recomputed = {
        "train_metrics_fixed_threshold": _evaluate_logits(
            train_logit, train_label, 0.5
        ),
        "validation_metrics_fixed_threshold": _evaluate_logits(
            validation_logit, validation_label, 0.5
        ),
        "validation_metrics_selected_threshold": _binary_metrics(
            validation_calibrated, validation_label, threshold
        ),
        "test_metrics_fixed_threshold": _evaluate_logits(
            test_logit, test_label, 0.5
        ),
        "test_metrics_selected_threshold": _binary_metrics(
            test_calibrated, test_label, threshold
        ),
        "oracle_test_threshold_diagnostic": {
            "diagnostic_only": True,
            "threshold": oracle_threshold,
            "balanced_accuracy": oracle_ba,
            "gain_over_validation_selected_threshold": (
                oracle_ba
                - _binary_metrics(
                    test_calibrated, test_label, threshold
                )["balanced_accuracy"]
            ),
        },
    }
    metric_comparisons = 0
    metric_mismatches = []
    for field, actual in recomputed.items():
        checked, differences = _compare_nested(
            record.get(field), actual, field
        )
        metric_comparisons += checked
        metric_mismatches.extend(differences)

    serialized_temperature = float(raw["temperature"].reshape(-1)[0])
    serialized_threshold = float(raw["threshold"].reshape(-1)[0])
    scalar_checks = {
        "temperature_record_vs_recomputed": (
            float(record["temperature"]),
            temperature,
        ),
        "temperature_serialized_vs_recomputed": (
            serialized_temperature,
            temperature,
        ),
        "threshold_record_vs_recomputed": (
            float(record["threshold"]),
            threshold,
        ),
        "threshold_serialized_vs_recomputed": (
            serialized_threshold,
            threshold,
        ),
    }
    scalar_mismatches = [
        {
            "field": name,
            "expected_or_serialized": left,
            "independently_recomputed": right,
        }
        for name, (left, right) in scalar_checks.items()
        if not (
            math.isfinite(left)
            and math.isfinite(right)
            and abs(left - right) <= METRIC_TOLERANCE
        )
    ]
    selected = recomputed["test_metrics_selected_threshold"]
    oracle = recomputed["oracle_test_threshold_diagnostic"]
    calibration_seed_evidence = bool(
        selected["AUROC"] is not None
        and selected["AUROC"] >= 0.70
        and selected["balanced_accuracy"] < FINAL_BA_GATE
        and oracle["gain_over_validation_selected_threshold"] >= 0.10
    )
    return {
        "architecture": architecture,
        "family": family,
        "spec": spec,
        "seed": seed,
        "prediction_path": str(path.resolve()),
        "prediction_sha256": sha256_file(path),
        "artifact_verification": artifact,
        "raw_integrity_pass": True,
        "temperature": temperature,
        "threshold": threshold,
        "metrics": recomputed,
        "metric_scalar_comparisons": metric_comparisons,
        "metric_mismatches": metric_mismatches,
        "calibration_scalar_checks": {
            name: {
                "expected_or_serialized": left,
                "independently_recomputed": right,
            }
            for name, (left, right) in scalar_checks.items()
        },
        "calibration_scalar_mismatches": scalar_mismatches,
        "calibration_failure_seed_evidence": calibration_seed_evidence,
        "pass": bool(
            artifact["pass"]
            and not metric_mismatches
            and not scalar_mismatches
        ),
    }


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "standard_deviation_population": float(np.std(array, ddof=0)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def _aggregate(
    recomputed_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    by_cell: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in recomputed_rows:
        by_cell[(row["architecture"], row["family"])].append(row)

    cells = []
    for architecture in ARCHITECTURES:
        for family, spec in zip(EASY_FAMILIES, SPECS, strict=True):
            selected = sorted(
                by_cell[(architecture, family)],
                key=lambda row: row["seed"],
            )
            if [row["seed"] for row in selected] != list(SEEDS):
                raise RuntimeError(
                    f"incomplete final cell {architecture}/{family}: "
                    f"{[row['seed'] for row in selected]}"
                )
            seed_rows = []
            passing_seeds = []
            calibration_seeds = []
            for row in selected:
                metrics = row["metrics"]["test_metrics_selected_threshold"]
                oracle = row["metrics"]["oracle_test_threshold_diagnostic"]
                balanced_accuracy = float(metrics["balanced_accuracy"])
                if balanced_accuracy >= FINAL_BA_GATE:
                    passing_seeds.append(row["seed"])
                if row["calibration_failure_seed_evidence"]:
                    calibration_seeds.append(row["seed"])
                seed_rows.append(
                    {
                        "seed": row["seed"],
                        "balanced_accuracy": balanced_accuracy,
                        "AUROC": metrics["AUROC"],
                        "AUPRC": metrics["AUPRC"],
                        "BCE_uncalibrated": row["metrics"][
                            "test_metrics_fixed_threshold"
                        ]["BCE"],
                        "validation_selected_temperature": row["temperature"],
                        "validation_selected_threshold": row["threshold"],
                        "oracle_threshold_diagnostic_balanced_accuracy": oracle[
                            "balanced_accuracy"
                        ],
                        "oracle_threshold_diagnostic_gain": oracle[
                            "gain_over_validation_selected_threshold"
                        ],
                        "calibration_failure_seed_evidence": row[
                            "calibration_failure_seed_evidence"
                        ],
                    }
                )
            cells.append(
                {
                    "architecture": architecture,
                    "family": family,
                    "spec": spec,
                    "seed_rows": seed_rows,
                    "balanced_accuracy": _summary(
                        [row["balanced_accuracy"] for row in seed_rows]
                    ),
                    "AUROC": _summary(
                        [float(row["AUROC"]) for row in seed_rows]
                    ),
                    "passing_seeds": passing_seeds,
                    "qualifies": len(passing_seeds) >= 2,
                    "calibration_failure_evidence_seeds": calibration_seeds,
                    "calibration_failure_cell_qualifies": (
                        len(calibration_seeds) >= 2
                    ),
                }
            )

    architecture_rows = []
    for architecture in ARCHITECTURES:
        selected_cells = [
            row for row in cells if row["architecture"] == architecture
        ]
        qualifying_families = [
            row["family"] for row in selected_cells if row["qualifies"]
        ]
        calibration_families = [
            row["family"]
            for row in selected_cells
            if row["calibration_failure_cell_qualifies"]
        ]
        all_seed_ba = [
            seed_row["balanced_accuracy"]
            for cell in selected_cells
            for seed_row in cell["seed_rows"]
        ]
        architecture_rows.append(
            {
                "architecture": architecture,
                "family_cells": selected_cells,
                "qualifying_families": qualifying_families,
                "qualifying_family_count": len(qualifying_families),
                "qualifies": bool(qualifying_families),
                "calibration_failure_families": calibration_families,
                "calibration_failure_qualifies": bool(calibration_families),
                "balanced_accuracy_across_12_runs": _summary(all_seed_ba),
            }
        )

    qualifying_architectures = [
        row["architecture"] for row in architecture_rows if row["qualifies"]
    ]
    calibration_architectures = [
        row["architecture"]
        for row in architecture_rows
        if row["calibration_failure_qualifies"]
    ]
    study_pass = len(qualifying_architectures) >= 2
    calibration = {
        "applicable": not study_pass,
        "criteria": {
            "same_architecture_family_cell": True,
            "ranking_AUROC_min": 0.70,
            "validation_threshold_test_BA_strictly_less_than": FINAL_BA_GATE,
            "oracle_test_threshold_BA_gain_min_diagnostic_only": 0.10,
            "required_seeds_per_cell": 2,
            "required_architectures": 2,
        },
        "qualifying_architectures": calibration_architectures,
        "qualifying_architecture_count": len(calibration_architectures),
        "gate_pass": bool(
            not study_pass and len(calibration_architectures) >= 2
        ),
        "oracle_test_threshold_used_for_selection": False,
    }
    global_gate = {
        "balanced_accuracy_threshold": FINAL_BA_GATE,
        "required_seeds_per_architecture_family_cell": 2,
        "required_architectures": 2,
        "qualifying_architectures": qualifying_architectures,
        "qualifying_architecture_count": len(qualifying_architectures),
        "study_pass": study_pass,
        "heldout_test_used_for_selection": False,
    }
    return cells, architecture_rows, {
        "IID_gate": global_gate,
        "calibration_failure_diagnostic": calibration,
    }


def run() -> dict[str, Any]:
    freeze_before = _verify_frozen_implementation()
    diagnostics = _diagnostics_complete()
    source_before = verify_bound_source_snapshot()
    if not source_before["pass"]:
        raise RuntimeError(f"source study changed before confirmation: {source_before}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for final IID confirmation")

    store = StudyStore()
    records = []
    protocol = protocol_for("final_original")
    for family, spec in zip(EASY_FAMILIES, SPECS, strict=True):
        data = SpecData(store, spec)
        train, validation, test = store.spec_indices(spec)
        for architecture in ARCHITECTURES:
            required = tuple(VERIFIER_CLASSES[architecture]().required_inputs)
            for seed in SEEDS:
                record = _train_with_fallback(
                    model_factory=VERIFIER_CLASSES[architecture],
                    model_name=architecture,
                    group=GROUP,
                    spec=spec,
                    seed=seed,
                    train_indices=train,
                    validation_indices=validation,
                    test_indices=test,
                    labels_numpy=store.labels,
                    batch_function=lambda index, r=required: data.batch(index, r),
                    protocol=protocol,
                    input_identity=data.input_identity(required),
                    normalizer_path=data.normalizer_path,
                )
                records.append(record)
                print(
                    f"FINAL family={family} arch={architecture} seed={seed} "
                    f"epochs={record['epochs_run']} "
                    f"test_BA={record['test_metrics_selected_threshold']['balanced_accuracy']:.4f}",
                    flush=True,
                )
        data.close()
    if len(records) != 60:
        raise RuntimeError(f"expected 60 final records, found {len(records)}")

    recomputed_rows = []
    for record in records:
        spec = record["spec"]
        family = spec.replace("within_", "").upper()
        recomputed_rows.append(
            _recompute_run(
                record,
                store,
                record["model"],
                family,
                spec,
                int(record["seed"]),
            )
        )
    independent_pass = all(row["pass"] for row in recomputed_rows)
    metric_comparisons = sum(
        row["metric_scalar_comparisons"] for row in recomputed_rows
    )
    metric_mismatches = [
        {
            "architecture": row["architecture"],
            "family": row["family"],
            "seed": row["seed"],
            **mismatch,
        }
        for row in recomputed_rows
        for mismatch in row["metric_mismatches"]
    ]
    scalar_mismatches = [
        {
            "architecture": row["architecture"],
            "family": row["family"],
            "seed": row["seed"],
            **mismatch,
        }
        for row in recomputed_rows
        for mismatch in row["calibration_scalar_mismatches"]
    ]
    artifact_failures = [
        {
            "architecture": row["architecture"],
            "family": row["family"],
            "seed": row["seed"],
            "verification": row["artifact_verification"],
        }
        for row in recomputed_rows
        if not row["artifact_verification"]["pass"]
    ]
    cells, architecture_rows, aggregation = _aggregate(recomputed_rows)
    source_after = verify_bound_source_snapshot()
    freeze_after = _verify_frozen_implementation()
    resources = resource_status()
    result = {
        "schema": "vild-iid-learnability-results-v1",
        "study_id": STUDY_ID,
        "protocol": dataclasses.asdict(protocol),
        "scope": {
            "architectures": list(ARCHITECTURES),
            "families": list(EASY_FAMILIES),
            "seeds": list(SEEDS),
            "expected_runs": 60,
            "completed_runs": len(records),
        },
        "prerequisite_diagnostics": diagnostics,
        "implementation_freeze_before": freeze_before,
        "implementation_freeze_after": freeze_after,
        "source_preservation_before": source_before,
        "source_preservation_after": source_after,
        "training_runs": records,
        "run_count": len(records),
        "independent_metric_recomputation": {
            "method": (
                "reread float32 raw logits and float64 serialized calibration "
                "scalars; independently refit validation temperature, independently "
                "reselect the validation BA threshold, and recompute every stored "
                "train/validation/test metric plus test-oracle diagnostic"
            ),
            "shared_metric_implementation_called": False,
            "tolerance": METRIC_TOLERANCE,
            "raw_prediction_files": len(recomputed_rows),
            "rows": recomputed_rows,
            "metric_scalar_comparisons": metric_comparisons,
            "metric_mismatches": metric_mismatches,
            "calibration_scalar_mismatches": scalar_mismatches,
            "artifact_failures": artifact_failures,
            "pass": independent_pass,
        },
        "cells": cells,
        "per_architecture": architecture_rows,
        "qualifying_architectures": aggregation["IID_gate"][
            "qualifying_architectures"
        ],
        "qualifying_architecture_count": aggregation["IID_gate"][
            "qualifying_architecture_count"
        ],
        "study_pass": aggregation["IID_gate"]["study_pass"],
        "IID_gate": aggregation["IID_gate"],
        "calibration_failure_diagnostic": aggregation[
            "calibration_failure_diagnostic"
        ],
        "test_used_for_selection": False,
        "oracle_test_threshold_diagnostic_only": True,
        "additional_verifier_architectures": 0,
        "physical_ground_truth": False,
        "OOM_fallback_runs": sum(
            bool(record.get("OOM_fallback_used", False)) for record in records
        ),
        "resource_status": resources,
        "valid_for_terminal_decision": bool(
            independent_pass
            and source_before["pass"]
            and source_after["pass"]
            and not freeze_before["failures"]
            and not freeze_after["failures"]
            and resources["resource_ledger_pass"]
            and resources["resource_record_coverage_pass"]
            and not resources["open_GPU_attempts"]
            and resources["within_GPU_limit"]
            and resources["within_artifact_limit"]
        ),
    }
    write_json(RESULT, result)
    if not result["valid_for_terminal_decision"]:
        raise RuntimeError(
            "final confirmation integrity failure; inspect "
            f"{RESULT}: independent={independent_pass}, "
            f"source_after={source_after['pass']}"
        )
    events = {row["event"] for row in read_jsonl(OUT / "run_log.jsonl")}
    if "FINAL_IID_CONFIRMATION_COMPLETE" not in events:
        append_log(
            "FINAL_IID_CONFIRMATION_COMPLETE",
            final_runs=len(records),
            independent_metric_recomputation_pass=independent_pass,
            metric_scalar_comparisons=metric_comparisons,
            qualifying_architectures=aggregation["IID_gate"][
                "qualifying_architectures"
            ],
            qualifying_architecture_count=aggregation["IID_gate"][
                "qualifying_architecture_count"
            ],
            study_pass=aggregation["IID_gate"]["study_pass"],
            calibration_failure_gate_pass=aggregation[
                "calibration_failure_diagnostic"
            ]["gate_pass"],
            heldout_test_used_for_selection=False,
            source_study_unchanged=source_after["pass"],
            OOM_fallback_runs=result["OOM_fallback_runs"],
            GPU_hours=resources["GPU_hours"],
            result_sha256=sha256_file(RESULT),
        )
    return {
        "run_count": len(records),
        "independent_recomputation_pass": independent_pass,
        "metric_scalar_comparisons": metric_comparisons,
        "qualifying_architectures": aggregation["IID_gate"][
            "qualifying_architectures"
        ],
        "study_pass": aggregation["IID_gate"]["study_pass"],
        "calibration_failure_gate_pass": aggregation[
            "calibration_failure_diagnostic"
        ]["gate_pass"],
        "source_study_unchanged": source_after["pass"],
        "resource_status": resources,
        "result_path": str(RESULT.resolve()),
    }


def main() -> None:
    print(json.dumps(run(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
