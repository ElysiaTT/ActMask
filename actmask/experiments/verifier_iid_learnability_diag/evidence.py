"""Read-only historical and representation evidence used by later diagnostics."""
from __future__ import annotations

import json
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
    SOURCE,
    SPECS,
    STUDY_ID,
    read_json,
    read_jsonl,
    sha256_file,
    verify_bound_source_snapshot,
    write_json,
)


def rank_auc(score: np.ndarray, label: np.ndarray) -> float:
    score = np.asarray(score, dtype=np.float64)
    label = np.asarray(label, dtype=np.int8)
    positive = label == 1
    negative = label == 0
    if not np.any(positive) or not np.any(negative):
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    sorted_score = score[order]
    sorted_label = label[order]
    negative_before = 0
    concordant = 0.0
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and sorted_score[end] == sorted_score[start]:
            end += 1
        group = sorted_label[start:end]
        group_negative = int(np.sum(group == 0))
        group_positive = int(np.sum(group == 1))
        concordant += group_positive * (negative_before + 0.5 * group_negative)
        negative_before += group_negative
        start = end
    return float(concordant / (int(positive.sum()) * int(negative.sum())))


def _summary(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(array)),
        "mean": float(array.mean()),
        "standard_deviation": float(array.std()),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "median": float(np.median(array)),
    }


def _paired_indices(
    rows: list[dict[str, Any]],
    indices: np.ndarray,
    labels: np.ndarray,
) -> list[tuple[int, int]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index in indices:
        grouped[rows[int(index)]["anchor_id"]].append(int(index))
    pairs = []
    for anchor, group in sorted(grouped.items()):
        if len(group) != 2:
            raise RuntimeError(f"not a pair: {anchor} -> {group}")
        positive = [index for index in group if labels[index] == 1]
        negative = [index for index in group if labels[index] == 0]
        if len(positive) != 1 or len(negative) != 1:
            raise RuntimeError(f"bad labels: {anchor} -> {group}")
        pairs.append((positive[0], negative[0]))
    return pairs


def _pair_logit_statistics(
    rows: list[dict[str, Any]],
    global_indices: np.ndarray,
    logits: np.ndarray,
    labels: np.ndarray,
) -> dict[str, Any]:
    position = {int(index): offset for offset, index in enumerate(global_indices)}
    pairs = _paired_indices(rows, global_indices, labels)
    positive = np.asarray([logits[position[left]] for left, _ in pairs])
    negative = np.asarray([logits[position[right]] for _, right in pairs])
    delta = np.abs(positive - negative)
    global_std = float(np.std(logits))
    return {
        "pairs": len(pairs),
        "absolute_pair_logit_delta": _summary(delta.tolist()),
        "global_logit_standard_deviation": global_std,
        "median_pair_delta_over_global_std": float(
            np.median(delta) / max(global_std, 1e-12)
        ),
        "positive_negative_pair_correlation": float(
            np.corrcoef(positive, negative)[0, 1]
        ),
    }


def _parameter_change(architecture: str, seed: int, checkpoint_path: str) -> float:
    torch.manual_seed(seed)
    model = VERIFIER_CLASSES[architecture]()
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    initial = dict(model.named_parameters())
    changed_squared = 0.0
    initial_squared = 0.0
    for name, value in checkpoint["state_dict"].items():
        if name not in initial or not torch.is_floating_point(value):
            continue
        difference = value.float() - initial[name].detach().float()
        changed_squared += float(torch.sum(difference * difference))
        initial_squared += float(
            torch.sum(initial[name].detach().float() ** 2)
        )
    return float(np.sqrt(changed_squared / max(initial_squared, 1e-30)))


def build_baseline() -> dict[str, Any]:
    source_check = verify_bound_source_snapshot()
    if not source_check["pass"]:
        raise RuntimeError(f"source study changed: {source_check}")
    rows = read_jsonl(SOURCE / "unified_audit_manifest.jsonl")
    with np.load(SOURCE / "unified_audit_arrays.npz") as stored:
        arrays = {name: stored[name].copy() for name in stored.files}
    with np.load(SOURCE / "nuisance_features.npz") as stored:
        nuisance = {name: stored[name].copy() for name in stored.files}
    with np.load(SOURCE / "audit_environment_indices.npz") as stored:
        indices = {name: stored[name].copy() for name in stored.files}
    schema = read_json(SOURCE / "nuisance_feature_schema.json")
    labels = arrays["label"].astype(np.int8)

    family_evidence = []
    historical_runs = []
    all_temperature = []
    for family, spec in zip(EASY_FAMILIES, SPECS, strict=True):
        train = indices[f"spec__{spec}__train"].astype(np.int64)
        validation = indices[f"spec__{spec}__validation"].astype(np.int64)
        test = indices[f"spec__{spec}__test"].astype(np.int64)
        pairs = _paired_indices(rows, test, labels)
        pair_positive = np.asarray([left for left, _ in pairs], dtype=np.int64)
        pair_negative = np.asarray([right for _, right in pairs], dtype=np.int64)

        with np.load(SOURCE / "verifier_normalizers" / f"{spec}.npz") as stored:
            action_mean = stored["candidate_action_mean"].copy()
            action_std = stored["candidate_action_std"].copy()
        normalized_action = (
            arrays["candidate_action"].astype(np.float32) - action_mean
        ) / action_std
        with np.load(
            SOURCE / "shortcut_control_normalizers" / f"{spec}.npz"
        ) as stored:
            summary_mean = stored["action_summary_mean"].copy()
            summary_std = stored["action_summary_std"].copy()
        normalized_summary = (
            nuisance["action"].astype(np.float32) - summary_mean
        ) / summary_std

        raw_delta = (
            normalized_action[pair_positive]
            - normalized_action[pair_negative]
        )
        summary_delta = (
            normalized_summary[pair_positive]
            - normalized_summary[pair_negative]
        )
        raw_rmse = np.sqrt(np.mean(raw_delta * raw_delta, axis=(1, 2)))
        summary_rmse = np.sqrt(np.mean(summary_delta * summary_delta, axis=1))
        shifted_negative = np.roll(pair_negative, 1)
        raw_random = np.sqrt(
            np.mean(
                (
                    normalized_action[pair_positive]
                    - normalized_action[shifted_negative]
                )
                ** 2,
                axis=(1, 2),
            )
        )
        summary_random = np.sqrt(
            np.mean(
                (
                    normalized_summary[pair_positive]
                    - normalized_summary[shifted_negative]
                )
                ** 2,
                axis=1,
            )
        )
        raw_bfloat = (
            torch.from_numpy(normalized_action[pair_positive])
            .to(torch.bfloat16)
            .float()
            .numpy()
            - torch.from_numpy(normalized_action[pair_negative])
            .to(torch.bfloat16)
            .float()
            .numpy()
        )
        originally_changed = raw_delta != 0
        collapsed = originally_changed & (raw_bfloat == 0)

        feature_rows = []
        for feature, name in enumerate(schema["groups"]["action"]):
            train_auc = rank_auc(
                nuisance["action"][train, feature], labels[train]
            )
            orientation = 1.0 if train_auc >= 0.5 else -1.0
            feature_rows.append(
                {
                    "feature": name,
                    "train_AUROC_oriented": max(train_auc, 1.0 - train_auc),
                    "validation_AUROC_oriented": rank_auc(
                        orientation * nuisance["action"][validation, feature],
                        labels[validation],
                    ),
                    "test_AUROC_oriented": rank_auc(
                        orientation * nuisance["action"][test, feature],
                        labels[test],
                    ),
                    "orientation_selected_on_train": int(orientation),
                }
            )
        top_features = sorted(
            feature_rows,
            key=lambda row: row["train_AUROC_oriented"],
            reverse=True,
        )[:10]

        control_rows = {}
        for control in (
            "current_state_action_summary_MLP",
            "temporal_action_only_TCN",
        ):
            record = read_json(
                SOURCE
                / "shortcut_control_run_records"
                / control
                / spec
                / "seed_17.json"
            )
            with np.load(record["artifact_paths"]["prediction"]) as stored:
                prediction_index = stored["test_index"].astype(np.int64)
                logit = stored["test_logit"].astype(np.float64)
            control_rows[control] = {
                "test_metrics": record["test_metrics"],
                "final_train_BCE": read_json(
                    record["artifact_paths"]["curve"]
                )["curve"][-1]["train_BCE"],
                "pair_logit_sensitivity": _pair_logit_statistics(
                    rows, prediction_index, logit, labels
                ),
            }

        fair_rows = []
        for architecture in ARCHITECTURES:
            for seed in SEEDS:
                record = read_json(
                    SOURCE
                    / "verifier_run_records"
                    / architecture
                    / spec
                    / f"seed_{seed}.json"
                )
                curve = read_json(record["artifact_paths"]["curve"])
                with np.load(record["artifact_paths"]["prediction"]) as stored:
                    prediction_index = stored["test_index"].astype(np.int64)
                    logit = stored["test_logit"].astype(np.float64)
                parameter_relative_change = _parameter_change(
                    architecture, seed, record["artifact_paths"]["checkpoint"]
                )
                run = {
                    "family": family,
                    "spec": spec,
                    "architecture": architecture,
                    "seed": seed,
                    "epochs_run": record["epochs_run"],
                    "best_epoch": record["best_epoch"],
                    "initial_train_BCE": curve["curve"][0]["train_BCE"],
                    "final_train_BCE": curve["curve"][-1]["train_BCE"],
                    "validation_balanced_accuracy": record[
                        "validation_metrics"
                    ]["balanced_accuracy"],
                    "test_balanced_accuracy": record["test_metrics"][
                        "balanced_accuracy"
                    ],
                    "test_AUROC": record["test_metrics"]["AUROC"],
                    "temperature": record["temperature"],
                    "threshold": record["threshold"],
                    "test_logit_mean": float(np.mean(logit)),
                    "test_logit_standard_deviation": float(np.std(logit)),
                    "test_logit_range": float(np.ptp(logit)),
                    "parameter_relative_change": parameter_relative_change,
                    "pair_logit_sensitivity": _pair_logit_statistics(
                        rows, prediction_index, logit, labels
                    ),
                }
                fair_rows.append(run)
                historical_runs.append(run)
                all_temperature.append(record["temperature"])

        family_evidence.append(
            {
                "family": family,
                "spec": spec,
                "train_samples": int(len(train)),
                "validation_samples": int(len(validation)),
                "test_samples": int(len(test)),
                "test_pairs": len(pairs),
                "normalized_pair_RMSE": {
                    "raw_action": _summary(raw_rmse.tolist()),
                    "engineered_action_summary": _summary(
                        summary_rmse.tolist()
                    ),
                    "median_summary_to_raw_amplification": float(
                        np.median(summary_rmse)
                        / max(float(np.median(raw_rmse)), 1e-12)
                    ),
                    "raw_pair_over_random_median": float(
                        np.median(raw_rmse)
                        / max(float(np.median(raw_random)), 1e-12)
                    ),
                    "summary_pair_over_random_median": float(
                        np.median(summary_rmse)
                        / max(float(np.median(summary_random)), 1e-12)
                    ),
                },
                "bfloat16_quantization": {
                    "originally_changed_scalars": int(
                        np.sum(originally_changed)
                    ),
                    "changed_scalars_collapsed_to_equal": int(
                        np.sum(collapsed)
                    ),
                    "collapsed_fraction": float(
                        np.sum(collapsed)
                        / max(int(np.sum(originally_changed)), 1)
                    ),
                    "entire_pairs_equal_after_quantization": int(
                        np.sum(np.all(raw_bfloat == 0, axis=(1, 2)))
                    ),
                    "L1_difference_retained_fraction": float(
                        np.sum(np.abs(raw_bfloat))
                        / max(float(np.sum(np.abs(raw_delta))), 1e-12)
                    ),
                },
                "top_train_selected_univariate_action_features": top_features,
                "historical_controls": control_rows,
                "historical_fair_runs": fair_rows,
            }
        )

    by_architecture = []
    for architecture in ARCHITECTURES:
        selected = [
            row for row in historical_runs if row["architecture"] == architecture
        ]
        by_architecture.append(
            {
                "architecture": architecture,
                "runs": len(selected),
                "epochs": _summary(
                    [float(row["epochs_run"]) for row in selected]
                ),
                "best_epoch": _summary(
                    [float(row["best_epoch"]) for row in selected]
                ),
                "final_train_BCE": _summary(
                    [row["final_train_BCE"] for row in selected]
                ),
                "test_balanced_accuracy": _summary(
                    [row["test_balanced_accuracy"] for row in selected]
                ),
                "test_AUROC": _summary(
                    [row["test_AUROC"] for row in selected]
                ),
                "parameter_relative_change": _summary(
                    [row["parameter_relative_change"] for row in selected]
                ),
            }
        )

    payload = {
        "schema": "vild-historical-baseline-evidence-v1",
        "study_id": STUDY_ID,
        "complete": True,
        "source_preservation": source_check,
        "source_files": {
            name: {
                "path": str((SOURCE / name).resolve()),
                "sha256": sha256_file(SOURCE / name),
            }
            for name in (
                "unified_audit_manifest.jsonl",
                "unified_audit_arrays.npz",
                "nuisance_features.npz",
                "audit_environment_indices.npz",
                "verifier_per_seed.json",
                "shortcut_control_summary.json",
            )
        },
        "input_information_equivalence": {
            "current_state_exactly_equals_last_history_state": bool(
                np.array_equal(
                    arrays["current_state"], arrays["history_state"][:, -1]
                )
            ),
            "all_334_action_summaries_are_deterministic_candidate_action_functions": (
                read_json(OUT / "alignment_audit.json")[
                    "action_feature_recomputation"
                ]["pass"]
            ),
            "strong_control_has_additional_raw_information": False,
            "difference": (
                "engineered high-order action representation and per-summary "
                "normalization, not extra source observations"
            ),
        },
        "family_evidence": family_evidence,
        "historical_runs": historical_runs,
        "architecture_summary": by_architecture,
        "temperature_search_upper_bound": float(np.exp(3.0)),
        "runs_at_temperature_upper_bound": int(
            np.sum(
                np.isclose(
                    np.asarray(all_temperature),
                    np.exp(3.0),
                    rtol=0,
                    atol=1e-12,
                )
            )
        ),
        "interpretation": (
            "Alignment and raw information availability are established. "
            "Engineered summaries amplify small construction perturbations; "
            "historical fair models show weak raw ranking before calibration."
        ),
        "causal_root_cause_claim": False,
    }
    write_json(OUT / "historical_baseline_evidence.json", payload)
    return payload


def main() -> None:
    payload = build_baseline()
    print(
        json.dumps(
            {
                "complete": payload["complete"],
                "families": len(payload["family_evidence"]),
                "historical_runs": len(payload["historical_runs"]),
                "runs_at_temperature_upper_bound": payload[
                    "runs_at_temperature_upper_bound"
                ],
                "output": str(
                    (OUT / "historical_baseline_evidence.json").resolve()
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
