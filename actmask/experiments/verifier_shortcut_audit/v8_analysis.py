"""V8 shortcut gaps, matching, calibration, language, and relational analyses."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .common import OUT, SEEDS, append_log, read_json, read_jsonl, sha256_file, write_json
from .metrics import (
    binary_metrics,
    fit_temperature,
    select_balanced_accuracy_threshold,
    sigmoid,
)
from .train_core import (
    DEVICE,
    AuditStore,
    ControlSpecData,
    _predict,
    output_paths,
    seed_everything,
)


ARCHITECTURES = (
    "StateActionTCN",
    "TaskConditionedActionTransformer",
    "VisualStateActionTransformer",
    "ContrastiveContextActionVerifier",
    "CrossAttentionEnergyVerifier",
)
FAMILIES = tuple(f"N{number}" for number in range(1, 7))
EASY = ("N2", "N3", "N5", "N6")
REAL = ("N1", "N4")


def _short_family(value: str) -> str:
    return value.split("_", 1)[0]


def _load_prediction(path: str | Path) -> dict[str, np.ndarray | float]:
    with np.load(path) as stored:
        files = set(stored.files)
        if "test_index" in files:
            return {
                "index": stored["test_index"].astype(np.int64),
                "label": stored["test_label"].astype(np.int8),
                "logit": stored["test_logit"].astype(np.float32),
                "probability": stored["test_calibrated_probability"].astype(np.float32),
                "threshold": float(stored["threshold"][0]),
                "temperature": float(stored["temperature"][0]),
            }
        return {
            "index": stored["index"].astype(np.int64),
            "label": stored["label"].astype(np.int8),
            "logit": stored["logit"].astype(np.float32),
            "probability": stored["calibrated_probability"].astype(np.float32),
            "threshold": float(stored["threshold"][0]),
            "temperature": float(stored["temperature"][0]),
        }


def _subset_metrics(
    prediction: dict[str, np.ndarray | float], selected: np.ndarray
) -> dict[str, Any] | None:
    index = prediction["index"]
    assert isinstance(index, np.ndarray)
    mask = np.isin(index, np.asarray(selected, dtype=np.int64))
    label = prediction["label"]
    probability = prediction["probability"]
    assert isinstance(label, np.ndarray) and isinstance(probability, np.ndarray)
    if not np.any(mask) or len(np.unique(label[mask])) < 2:
        return None
    return binary_metrics(
        probability[mask], label[mask], float(prediction["threshold"])
    )


def _rows_by_key(rows: list[dict[str, Any]]) -> dict[tuple[Any, ...], dict[str, Any]]:
    result = {}
    for row in rows:
        result[
            (
                row["architecture"],
                int(row["seed"]),
                row["environment"],
                row["condition"],
            )
        ] = row
    return result


def _training_by_key(records: list[dict[str, Any]]) -> dict[tuple[str, int, str], dict[str, Any]]:
    return {
        (record["model"], int(record["seed"]), record["spec"]): record
        for record in records
    }


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _std(values: list[float]) -> float | None:
    return float(np.std(values)) if values else None


def _rank(value: np.ndarray) -> np.ndarray:
    order = np.argsort(value, kind="mergesort")
    result = np.empty(len(value), dtype=np.float64)
    start = 0
    while start < len(value):
        end = start + 1
        while end < len(value) and value[order[end]] == value[order[start]]:
            end += 1
        result[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return result


def _score_change(
    correct: np.ndarray,
    variant: np.ndarray,
    correct_metrics: dict[str, Any],
    variant_metrics: dict[str, Any],
) -> dict[str, Any]:
    if len(correct) < 2 or np.std(correct) <= 1e-12 or np.std(variant) <= 1e-12:
        rank_correlation = None
    else:
        rank_correlation = float(np.corrcoef(_rank(correct), _rank(variant))[0, 1])
    return {
        "mean_signed_score_delta_correct_minus_variant": float(
            np.mean(correct - variant)
        ),
        "mean_absolute_score_delta": float(np.mean(np.abs(correct - variant))),
        "score_delta_standard_deviation": float(np.std(correct - variant)),
        "rank_correlation": rank_correlation,
        "balanced_accuracy_correct": correct_metrics["balanced_accuracy"],
        "balanced_accuracy_variant": variant_metrics["balanced_accuracy"],
        "balanced_accuracy_delta": (
            correct_metrics["balanced_accuracy"]
            - variant_metrics["balanced_accuracy"]
        ),
    }


def _generator_and_easy_gaps(
    training: dict[tuple[str, int, str], dict[str, Any]],
    manifest: list[dict[str, Any]],
) -> dict[str, Any]:
    family_names = {
        _short_family(row["family"]): row["family"] for row in manifest
    }
    family_values = np.asarray(
        [_short_family(row["family"]) for row in manifest], dtype=object
    )
    task_values = np.asarray([row["task"] for row in manifest], dtype=object)
    generator_rows = []
    generator_task_rows = []
    easy_rows = []
    easy_task_rows = []
    for architecture in ARCHITECTURES:
        for seed in SEEDS:
            within_predictions = {}
            for family in FAMILIES:
                record = training[(architecture, seed, f"within_{family.lower()}")]
                within_predictions[family] = _load_prediction(
                    record["artifact_paths"]["prediction"]
                )
            for family in ("N1", "N2", "N4", "N5", "N6"):
                heldout = training[
                    (architecture, seed, f"leaveout_{family.lower()}")
                ]
                iid_ba = training[
                    (architecture, seed, f"within_{family.lower()}")
                ]["test_metrics"]["balanced_accuracy"]
                heldout_ba = heldout["test_metrics"]["balanced_accuracy"]
                generator_rows.append(
                    {
                        "architecture": architecture,
                        "seed": seed,
                        "heldout_family": family,
                        "family_full_name": family_names[family],
                        "IID_balanced_accuracy": iid_ba,
                        "generator_heldout_balanced_accuracy": heldout_ba,
                        "generator_transfer_gap": iid_ba - heldout_ba,
                    }
                )
                iid_prediction = within_predictions[family]
                held_prediction = _load_prediction(
                    heldout["artifact_paths"]["prediction"]
                )
                indices = iid_prediction["index"]
                assert isinstance(indices, np.ndarray)
                for task in sorted(set(task_values[indices])):
                    selected = indices[task_values[indices] == task]
                    iid_metrics = _subset_metrics(iid_prediction, selected)
                    held_metrics = _subset_metrics(held_prediction, selected)
                    if iid_metrics is None or held_metrics is None:
                        continue
                    generator_task_rows.append(
                        {
                            "architecture": architecture,
                            "seed": seed,
                            "heldout_family": family,
                            "task": task,
                            "samples": int(len(selected)),
                            "IID_balanced_accuracy": iid_metrics[
                                "balanced_accuracy"
                            ],
                            "generator_heldout_balanced_accuracy": held_metrics[
                                "balanced_accuracy"
                            ],
                            "generator_transfer_gap": iid_metrics[
                                "balanced_accuracy"
                            ]
                            - held_metrics["balanced_accuracy"],
                        }
                    )
            easy_iid = float(
                np.mean(
                    [
                        training[
                            (architecture, seed, f"within_{family.lower()}")
                        ]["test_metrics"]["balanced_accuracy"]
                        for family in EASY
                    ]
                )
            )
            transfer = training[(architecture, seed, "group_e_to_r")][
                "test_metrics"
            ]["balanced_accuracy"]
            easy_rows.append(
                {
                    "architecture": architecture,
                    "seed": seed,
                    "group_E_IID_balanced_accuracy": easy_iid,
                    "N1_N4_transfer_balanced_accuracy": transfer,
                    "easy_to_real_gap": easy_iid - transfer,
                }
            )
            transfer_prediction = _load_prediction(
                training[(architecture, seed, "group_e_to_r")][
                    "artifact_paths"
                ]["prediction"]
            )
            transfer_indices = transfer_prediction["index"]
            assert isinstance(transfer_indices, np.ndarray)
            for task in sorted(set(task_values[transfer_indices])):
                task_transfer_indices = transfer_indices[
                    task_values[transfer_indices] == task
                ]
                transfer_metrics = _subset_metrics(
                    transfer_prediction, task_transfer_indices
                )
                iid_task_values = []
                for family in EASY:
                    prediction = within_predictions[family]
                    prediction_indices = prediction["index"]
                    assert isinstance(prediction_indices, np.ndarray)
                    selected = prediction_indices[
                        task_values[prediction_indices] == task
                    ]
                    metric = _subset_metrics(prediction, selected)
                    if metric is not None:
                        iid_task_values.append(metric["balanced_accuracy"])
                if transfer_metrics is None or not iid_task_values:
                    continue
                easy_task_rows.append(
                    {
                        "architecture": architecture,
                        "seed": seed,
                        "task": task,
                        "group_E_IID_balanced_accuracy": float(
                            np.mean(iid_task_values)
                        ),
                        "N1_N4_transfer_balanced_accuracy": transfer_metrics[
                            "balanced_accuracy"
                        ],
                        "easy_to_real_gap": float(np.mean(iid_task_values))
                        - transfer_metrics["balanced_accuracy"],
                    }
                )
    aggregate = []
    for architecture in ARCHITECTURES:
        relevant_generator = [
            row["generator_transfer_gap"]
            for row in generator_rows
            if row["architecture"] == architecture
        ]
        relevant_easy = [
            row["easy_to_real_gap"]
            for row in easy_rows
            if row["architecture"] == architecture
        ]
        aggregate.append(
            {
                "architecture": architecture,
                "generator_transfer_gap_mean": _mean(relevant_generator),
                "generator_transfer_gap_standard_deviation": _std(
                    relevant_generator
                ),
                "generator_transfer_gap_maximum": max(relevant_generator),
                "easy_to_real_gap_mean": _mean(relevant_easy),
                "easy_to_real_gap_standard_deviation": _std(relevant_easy),
            }
        )
    return {
        "generator_rows": generator_rows,
        "generator_task_rows": generator_task_rows,
        "easy_to_real_rows": easy_rows,
        "easy_to_real_task_rows": easy_task_rows,
        "architecture_aggregate": aggregate,
    }


def _matching_gaps(
    training: dict[tuple[str, int, str], dict[str, Any]],
    evaluation: dict[tuple[Any, ...], dict[str, Any]],
    manifest: list[dict[str, Any]],
) -> dict[str, Any]:
    family_values = np.asarray(
        [_short_family(row["family"]) for row in manifest], dtype=object
    )
    task_values = np.asarray([row["task"] for row in manifest], dtype=object)
    rows = []
    for architecture in ARCHITECTURES:
        for seed in SEEDS:
            base = _load_prediction(
                training[(architecture, seed, "pooled_n1_n6")][
                    "artifact_paths"
                ]["prediction"]
            )
            base_indices = base["index"]
            assert isinstance(base_indices, np.ndarray)
            for environment, gap_name in (
                ("E5", "action_matching_gap"),
                ("E6", "progress_matching_gap"),
            ):
                matched_row = evaluation[(architecture, seed, environment, "all")]
                matched = _load_prediction(matched_row["raw_prediction_path"])
                matched_indices = matched["index"]
                assert isinstance(matched_indices, np.ndarray)
                for grouping, values in (
                    ("all", ["all"]),
                    ("family", list(FAMILIES)),
                    ("task", sorted(set(task_values[matched_indices]))),
                ):
                    for value in values:
                        if grouping == "all":
                            base_selected = base_indices
                            matched_selected = matched_indices
                        elif grouping == "family":
                            base_selected = base_indices[
                                family_values[base_indices] == value
                            ]
                            matched_selected = matched_indices[
                                family_values[matched_indices] == value
                            ]
                        else:
                            base_selected = base_indices[
                                task_values[base_indices] == value
                            ]
                            matched_selected = matched_indices[
                                task_values[matched_indices] == value
                            ]
                        base_metrics = _subset_metrics(base, base_selected)
                        matched_metrics = _subset_metrics(
                            matched, matched_selected
                        )
                        if base_metrics is None or matched_metrics is None:
                            continue
                        rows.append(
                            {
                                "architecture": architecture,
                                "seed": seed,
                                "environment": environment,
                                "grouping": grouping,
                                "group": value,
                                "original_samples": int(len(base_selected)),
                                "matched_samples": int(len(matched_selected)),
                                "original_balanced_accuracy": base_metrics[
                                    "balanced_accuracy"
                                ],
                                "matched_balanced_accuracy": matched_metrics[
                                    "balanced_accuracy"
                                ],
                                gap_name: base_metrics["balanced_accuracy"]
                                - matched_metrics["balanced_accuracy"],
                            }
                        )
    aggregate = []
    for architecture in ARCHITECTURES:
        for environment, gap_name in (
            ("E5", "action_matching_gap"),
            ("E6", "progress_matching_gap"),
        ):
            values = [
                row[gap_name]
                for row in rows
                if row["architecture"] == architecture
                and row["environment"] == environment
                and row["grouping"] == "all"
            ]
            aggregate.append(
                {
                    "architecture": architecture,
                    "environment": environment,
                    f"{gap_name}_mean": _mean(values),
                    f"{gap_name}_standard_deviation": _std(values),
                }
            )
    return {
        "schema": "vsa-matching-gap-report-v1",
        "rows": rows,
        "architecture_aggregate": aggregate,
        "matching_quality_report_path": str(
            (OUT / "matching_quality_report.json").resolve()
        ),
        "matching_quality_report_sha256": sha256_file(
            OUT / "matching_quality_report.json"
        ),
        "heldout_model_performance_used_for_matching": False,
    }


def _language_reliance(
    evaluation: dict[tuple[Any, ...], dict[str, Any]],
    manifest: list[dict[str, Any]],
) -> dict[str, Any]:
    family_values = np.asarray(
        [_short_family(row["family"]) for row in manifest], dtype=object
    )
    task_values = np.asarray([row["task"] for row in manifest], dtype=object)
    rows = []
    for architecture in ARCHITECTURES:
        for seed in SEEDS:
            predictions = {
                condition: _load_prediction(
                    evaluation[(architecture, seed, "E10", condition)][
                        "raw_prediction_path"
                    ]
                )
                for condition in ("correct", "shuffled", "empty", "task_id_only")
            }
            correct = predictions["correct"]
            indices = correct["index"]
            assert isinstance(indices, np.ndarray)
            for grouping, groups in (
                ("all", ["all"]),
                ("family", list(FAMILIES)),
                ("task", sorted(set(task_values[indices]))),
            ):
                for group in groups:
                    if grouping == "all":
                        mask = np.ones(len(indices), dtype=bool)
                    elif grouping == "family":
                        mask = family_values[indices] == group
                    else:
                        mask = task_values[indices] == group
                    if not np.any(mask):
                        continue
                    correct_label = correct["label"]
                    correct_probability = correct["probability"]
                    assert isinstance(correct_label, np.ndarray)
                    assert isinstance(correct_probability, np.ndarray)
                    if len(np.unique(correct_label[mask])) < 2:
                        continue
                    correct_metrics = binary_metrics(
                        correct_probability[mask],
                        correct_label[mask],
                        float(correct["threshold"]),
                    )
                    for condition in ("shuffled", "empty", "task_id_only"):
                        variant = predictions[condition]
                        variant_probability = variant["probability"]
                        variant_label = variant["label"]
                        assert isinstance(variant_probability, np.ndarray)
                        assert isinstance(variant_label, np.ndarray)
                        variant_metrics = binary_metrics(
                            variant_probability[mask],
                            variant_label[mask],
                            float(variant["threshold"]),
                        )
                        rows.append(
                            {
                                "architecture": architecture,
                                "seed": seed,
                                "grouping": grouping,
                                "group": group,
                                "condition": condition,
                                "samples": int(mask.sum()),
                                **_score_change(
                                    correct_probability[mask],
                                    variant_probability[mask],
                                    correct_metrics,
                                    variant_metrics,
                                ),
                            }
                        )
    aggregate = []
    for architecture in ARCHITECTURES:
        for condition in ("shuffled", "empty", "task_id_only"):
            relevant = [
                row
                for row in rows
                if row["architecture"] == architecture
                and row["condition"] == condition
                and row["grouping"] == "all"
            ]
            aggregate.append(
                {
                    "architecture": architecture,
                    "condition": condition,
                    "mean_absolute_score_delta": _mean(
                        [row["mean_absolute_score_delta"] for row in relevant]
                    ),
                    "balanced_accuracy_delta_mean": _mean(
                        [row["balanced_accuracy_delta"] for row in relevant]
                    ),
                    "rank_correlation_mean": _mean(
                        [
                            row["rank_correlation"]
                            for row in relevant
                            if row["rank_correlation"] is not None
                        ]
                    ),
                }
            )
    return {
        "rows": rows,
        "architecture_aggregate": aggregate,
        "interpretation": (
            "language reliance may legitimately differ by task and family; "
            "score changes are not physical-validity evidence"
        ),
    }


def _calibration_shift(
    training: dict[tuple[str, int, str], dict[str, Any]],
    evaluation: dict[tuple[Any, ...], dict[str, Any]],
) -> dict[str, Any]:
    rows = []
    for architecture in ARCHITECTURES:
        for seed in SEEDS:
            iid = training[(architecture, seed, "pooled_n1_n6")][
                "test_metrics"
            ]
            settings = {
                "generator_heldout": [
                    training[
                        (architecture, seed, f"leaveout_{family.lower()}")
                    ]["test_metrics"]
                    for family in ("N1", "N2", "N4", "N5", "N6")
                ],
                "task_heldout": [
                    training[(architecture, seed, f"task_fold_{fold}")][
                        "test_metrics"
                    ]
                    for fold in range(5)
                ],
                "source_heldout": [
                    evaluation[
                        (architecture, seed, "E7", "source_heldout")
                    ]["metrics"]
                ],
            }
            for setting, metrics in settings.items():
                shifted_ece = float(np.mean([value["ECE"] for value in metrics]))
                shifted_brier = float(
                    np.mean([value["Brier"] for value in metrics])
                )
                rows.append(
                    {
                        "architecture": architecture,
                        "seed": seed,
                        "setting": setting,
                        "IID_ECE": iid["ECE"],
                        "shifted_ECE": shifted_ece,
                        "ECE_shift": shifted_ece - iid["ECE"],
                        "IID_Brier": iid["Brier"],
                        "shifted_Brier": shifted_brier,
                        "Brier_shift": shifted_brier - iid["Brier"],
                    }
                )
    aggregate = []
    for architecture in ARCHITECTURES:
        for setting in (
            "generator_heldout",
            "task_heldout",
            "source_heldout",
        ):
            relevant = [
                row
                for row in rows
                if row["architecture"] == architecture
                and row["setting"] == setting
            ]
            aggregate.append(
                {
                    "architecture": architecture,
                    "setting": setting,
                    "ECE_shift_mean": _mean(
                        [row["ECE_shift"] for row in relevant]
                    ),
                    "ECE_shift_standard_deviation": _std(
                        [row["ECE_shift"] for row in relevant]
                    ),
                    "Brier_shift_mean": _mean(
                        [row["Brier_shift"] for row in relevant]
                    ),
                }
            )
    return {
        "schema": "vsa-calibration-shift-report-v1",
        "rows": rows,
        "architecture_aggregate": aggregate,
        "calibration": "validation-only scalar temperature retained unchanged under shift",
    }


def _context_only_scores(
    store: AuditStore,
) -> tuple[
    dict[str, np.ndarray | float],
    dict[str, dict[str, np.ndarray | float]],
]:
    data = ControlSpecData(store, "pooled_n1_n6")
    control = "combined_non_relational_nuisance_classifier"
    seed_everything(17)
    model = data.create_model(control).to(DEVICE)
    checkpoint = torch.load(
        output_paths("shortcut_control", control, "pooled_n1_n6", 17)[
            "checkpoint"
        ],
        map_location=DEVICE,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["state_dict"])
    _, validation_indices, test_indices = store.spec_indices("pooled_n1_n6")
    _, validation_representation = _predict(
        model,
        validation_indices,
        lambda index: data.batch(index, control),
    )
    validation_context = validation_representation[:, 1]
    validation_label = store.labels[validation_indices].astype(np.int8)
    temperature = fit_temperature(validation_context, validation_label)
    validation_probability = sigmoid(validation_context / temperature)
    threshold, _ = select_balanced_accuracy_threshold(
        validation_probability, validation_label
    )
    _, test_representation = _predict(
        model, test_indices, lambda index: data.batch(index, control)
    )
    base = {
        "index": test_indices,
        "label": store.labels[test_indices].astype(np.int8),
        "logit": test_representation[:, 1],
        "probability": sigmoid(test_representation[:, 1] / temperature).astype(
            np.float32
        ),
        "temperature": temperature,
        "threshold": threshold,
    }
    reciprocal = {}
    for condition, key in store.environment_manifest["environments"]["E9"][
        "index_keys"
    ].items():
        indices = store.indices[key].astype(np.int64)
        _, representation = _predict(
            model, indices, lambda index: data.batch(index, control)
        )
        reciprocal[condition] = {
            "index": indices,
            "label": store.labels[indices].astype(np.int8),
            "logit": representation[:, 1],
            "probability": sigmoid(representation[:, 1] / temperature).astype(
                np.float32
            ),
            "temperature": temperature,
            "threshold": threshold,
        }
    del model, checkpoint
    data.close()
    return base, reciprocal


def _reciprocal_consistency(
    prediction: dict[str, np.ndarray | float],
    manifest: list[dict[str, Any]],
) -> dict[str, Any]:
    indices = prediction["index"]
    probability = prediction["probability"]
    assert isinstance(indices, np.ndarray) and isinstance(probability, np.ndarray)
    by_pair_context: dict[tuple[str, str], dict[int, float]] = defaultdict(dict)
    for row_index, sample_index in enumerate(indices):
        row = manifest[int(sample_index)]
        pair_id = row["provenance"]["pair_id"]
        if pair_id is None:
            continue
        by_pair_context[(pair_id, row["context_id"])][
            int(row["construction_label"])
        ] = float(probability[row_index])
    comparisons = [
        values[1] > values[0]
        for values in by_pair_context.values()
        if 0 in values and 1 in values
    ]
    pair_results: dict[str, list[bool]] = defaultdict(list)
    for (pair_id, _), values in by_pair_context.items():
        if 0 in values and 1 in values:
            pair_results[pair_id].append(values[1] > values[0])
    strict = [
        len(results) == 2 and all(results) for results in pair_results.values()
    ]
    return {
        "contextwise_positive_over_swapped_rate": float(np.mean(comparisons))
        if comparisons
        else None,
        "strict_two_direction_pair_consistency": float(np.mean(strict))
        if strict
        else None,
        "context_comparisons": len(comparisons),
        "reciprocal_pairs": len(strict),
    }


def _relational_gain(
    training: dict[tuple[str, int, str], dict[str, Any]],
    evaluation: dict[tuple[Any, ...], dict[str, Any]],
    control_rows: list[dict[str, Any]],
    manifest: list[dict[str, Any]],
) -> dict[str, Any]:
    store = AuditStore()
    context_base, context_reciprocal = _context_only_scores(store)
    action_default_record = next(
        record
        for record in read_json(OUT / "shortcut_control_per_environment.json")[
            "rows"
        ]
        if record["control"] == "temporal_action_only_TCN"
        and record["training_spec"] == "pooled_n1_n6"
        and record["condition"] == "default_test"
    )
    action_base = _load_prediction(action_default_record["raw_prediction_path"])
    action_e9 = {
        condition: _load_prediction(
            next(
                row
                for row in control_rows
                if row["control"] == "temporal_action_only_TCN"
                and row["environment"] == "E9"
                and row["condition"] == condition
            )["raw_prediction_path"]
        )
        for condition in ("N7", "N8", "all")
    }
    family_values = np.asarray(
        [_short_family(row["family"]) for row in manifest], dtype=object
    )
    task_values = np.asarray([row["task"] for row in manifest], dtype=object)
    rows = []
    reciprocal_rows = []
    for architecture in ARCHITECTURES:
        for seed in SEEDS:
            full = _load_prediction(
                training[(architecture, seed, "pooled_n1_n6")][
                    "artifact_paths"
                ]["prediction"]
            )
            full_indices = full["index"]
            assert isinstance(full_indices, np.ndarray)
            for grouping, groups in (
                ("all", ["all"]),
                ("family", list(FAMILIES)),
                ("task", sorted(set(task_values[full_indices]))),
            ):
                for group in groups:
                    if grouping == "all":
                        selected = full_indices
                    elif grouping == "family":
                        selected = full_indices[
                            family_values[full_indices] == group
                        ]
                    else:
                        selected = full_indices[
                            task_values[full_indices] == group
                        ]
                    full_metric = _subset_metrics(full, selected)
                    action_metric = _subset_metrics(action_base, selected)
                    context_metric = _subset_metrics(context_base, selected)
                    if (
                        full_metric is None
                        or action_metric is None
                        or context_metric is None
                    ):
                        continue
                    baseline = max(
                        action_metric["balanced_accuracy"],
                        context_metric["balanced_accuracy"],
                    )
                    rows.append(
                        {
                            "architecture": architecture,
                            "seed": seed,
                            "environment": "pooled_reference",
                            "grouping": grouping,
                            "group": group,
                            "full_verifier_BA": full_metric[
                                "balanced_accuracy"
                            ],
                            "action_only_BA": action_metric[
                                "balanced_accuracy"
                            ],
                            "context_only_BA": context_metric[
                                "balanced_accuracy"
                            ],
                            "relational_gain": full_metric[
                                "balanced_accuracy"
                            ]
                            - baseline,
                        }
                    )
            for condition in ("N7", "N8", "all"):
                full_reciprocal = _load_prediction(
                    evaluation[(architecture, seed, "E9", condition)][
                        "raw_prediction_path"
                    ]
                )
                reciprocal_indices = full_reciprocal["index"]
                assert isinstance(reciprocal_indices, np.ndarray)
                full_metric = _subset_metrics(
                    full_reciprocal, reciprocal_indices
                )
                action_metric = _subset_metrics(
                    action_e9[condition], reciprocal_indices
                )
                context_metric = _subset_metrics(
                    context_reciprocal[condition], reciprocal_indices
                )
                assert (
                    full_metric is not None
                    and action_metric is not None
                    and context_metric is not None
                )
                consistency = _reciprocal_consistency(
                    full_reciprocal, manifest
                )
                reciprocal_rows.append(
                    {
                        "architecture": architecture,
                        "seed": seed,
                        "family": condition,
                        "samples": int(len(reciprocal_indices)),
                        "full_verifier_BA": full_metric[
                            "balanced_accuracy"
                        ],
                        "action_only_BA": action_metric[
                            "balanced_accuracy"
                        ],
                        "context_only_BA": context_metric[
                            "balanced_accuracy"
                        ],
                        "relational_gain": full_metric["balanced_accuracy"]
                        - max(
                            action_metric["balanced_accuracy"],
                            context_metric["balanced_accuracy"],
                        ),
                        **consistency,
                    }
                )
    aggregate = []
    for architecture in ARCHITECTURES:
        relevant = [
            row["relational_gain"]
            for row in rows
            if row["architecture"] == architecture
            and row["grouping"] == "all"
        ]
        reciprocal_relevant = [
            row["relational_gain"]
            for row in reciprocal_rows
            if row["architecture"] == architecture and row["family"] == "all"
        ]
        aggregate.append(
            {
                "architecture": architecture,
                "pooled_relational_gain_mean": _mean(relevant),
                "pooled_relational_gain_standard_deviation": _std(relevant),
                "reciprocal_relational_gain_mean": _mean(
                    reciprocal_relevant
                ),
            }
        )
    return {
        "schema": "vsa-relational-gain-report-v1",
        "definition": (
            "full verifier BA minus max(action-only temporal TCN BA, "
            "standalone context-group contribution BA)"
        ),
        "context_only_definition": (
            "context group logit of the frozen additive non-relational nuisance "
            "control, independently calibrated and thresholded on validation"
        ),
        "rows": rows,
        "reciprocal_rows": reciprocal_rows,
        "architecture_aggregate": aggregate,
        "physical_validity_interpretation_forbidden": True,
    }


def build() -> dict[str, Any]:
    manifest = read_jsonl(OUT / "unified_audit_manifest.jsonl")
    verifier_per_seed = read_json(OUT / "verifier_per_seed.json")
    verifier_per_environment = read_json(OUT / "verifier_per_environment.json")
    control_per_environment = read_json(
        OUT / "shortcut_control_per_environment.json"
    )
    training = _training_by_key(verifier_per_seed["training_runs"])
    evaluation = _rows_by_key(verifier_per_environment["rows"])
    gap_report = _generator_and_easy_gaps(training, manifest)
    matching_report = _matching_gaps(training, evaluation, manifest)
    language_report = _language_reliance(evaluation, manifest)
    calibration_report = _calibration_shift(training, evaluation)
    relational_report = _relational_gain(
        training, evaluation, control_per_environment["rows"], manifest
    )
    write_json(OUT / "matching_gap_report.json", matching_report)
    write_json(OUT / "calibration_shift_report.json", calibration_report)
    write_json(OUT / "relational_gain_report.json", relational_report)
    source_gap_rows = []
    for architecture in ARCHITECTURES:
        for seed in SEEDS:
            iid = training[(architecture, seed, "pooled_n1_n6")][
                "test_metrics"
            ]["balanced_accuracy"]
            source = evaluation[
                (architecture, seed, "E7", "source_heldout")
            ]["metrics"]["balanced_accuracy"]
            source_gap_rows.append(
                {
                    "architecture": architecture,
                    "seed": seed,
                    "episode_IID_balanced_accuracy": iid,
                    "source_heldout_balanced_accuracy": source,
                    "source_heldout_gap": iid - source,
                }
            )
    required_external = (
        "generator_transfer_matrix.json",
        "generator_transfer_matrix.csv",
        "nuisance_probe_report.json",
    )
    missing_external = [
        name for name in required_external if not (OUT / name).is_file()
    ]
    if missing_external:
        raise RuntimeError(
            "run delegated V8 components before final summary: "
            + ", ".join(missing_external)
        )
    summary = {
        "schema": "vsa-shortcut-reliance-summary-v1",
        "study_id": "VERIFIER-SHORTCUT-AUDIT-V1",
        "standard_metrics": [
            "balanced_accuracy",
            "AUROC",
            "AUPRC",
            "ECE",
            "Brier",
            "false_accept_rate",
            "false_reject_rate",
        ],
        "standard_metric_source": str(
            (OUT / "verifier_per_environment.json").resolve()
        ),
        "standard_metric_source_sha256": sha256_file(
            OUT / "verifier_per_environment.json"
        ),
        "generator_and_easy_to_real": gap_report,
        "source_heldout_rows": source_gap_rows,
        "language_reliance": language_report,
        "supporting_reports": {
            name: {
                "path": str((OUT / name).resolve()),
                "sha256": sha256_file(OUT / name),
            }
            for name in (
                "generator_transfer_matrix.csv",
                "generator_transfer_matrix.json",
                "matching_gap_report.json",
                "nuisance_probe_report.json",
                "calibration_shift_report.json",
                "relational_gain_report.json",
            )
        },
        "all_metrics_recomputed_from_raw_predictions": True,
        "physical_ground_truth": False,
        "claim_boundary": (
            "benchmark-construction shortcut and transfer audit only; "
            "not physical safety or failure validity"
        ),
    }
    write_json(OUT / "shortcut_reliance_summary.json", summary)
    append_log(
        "V8_SHORTCUT_RELIANCE_METRICS_COMPLETE",
        generator_gap_rows=len(gap_report["generator_rows"]),
        easy_to_real_rows=len(gap_report["easy_to_real_rows"]),
        matching_gap_rows=len(matching_report["rows"]),
        language_reliance_rows=len(language_report["rows"]),
        calibration_shift_rows=len(calibration_report["rows"]),
        relational_gain_rows=len(relational_report["rows"]),
        raw_predictions_recomputed=True,
        summary_sha256=sha256_file(OUT / "shortcut_reliance_summary.json"),
    )
    return {
        "pass": True,
        "generator_gap_rows": len(gap_report["generator_rows"]),
        "matching_gap_rows": len(matching_report["rows"]),
        "language_reliance_rows": len(language_report["rows"]),
        "calibration_shift_rows": len(calibration_report["rows"]),
        "relational_gain_rows": len(relational_report["rows"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    print(json.dumps(build(), indent=2))


if __name__ == "__main__":
    main()
