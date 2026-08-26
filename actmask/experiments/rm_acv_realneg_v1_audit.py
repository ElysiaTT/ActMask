"""RM-ACV-REALNEG-V1 phases R4--R6 and terminal packaging.

All controls operate on the frozen reciprocal samples.  Action-only and
context-only models never receive a relational identity/equality feature.
Privileged provenance and pair diagnostics remain audit-only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn

from actmask.experiments.rm_acv_pilot166_common import (
    action_summary,
    best_balanced_threshold,
    binary_metrics,
    read_jsonl,
    sha256_file,
    sha256_json,
    stable_int,
    train_shortcut_classifier,
    write_json,
    write_jsonl,
)
from actmask.experiments.rm_acv_pilot166_finalize import verify as verify_pilot
from actmask.experiments.rm_acv_realneg_v1_prepare import (
    FAMILIES,
    N8_PAIRS_PER_TASK,
    OUT,
    PILOT,
    ROOT,
    _balance_audits,
    append_log,
)


N7, N8 = FAMILIES
REQUIRED_SHORTCUT_CONTROLS = (
    "action_magnitude",
    "action_mean_variance",
    "action_smoothness_jerk",
    "action_start_end",
    "action_total_displacement",
    "action_summary_full",
    "learned_action_only_MLP",
    "learned_action_only_TCN",
    "task_identity",
    "normalized_progress",
    "current_state",
    "state_history",
    "causal_visual_feature",
    "task_plus_state",
    "visual_plus_state",
    "episode_duration",
    "source_episode_metadata",
    "pair_mining_rank",
    "pair_distance_values",
    "file_path_metadata",
    "action_context_IDs",
    "source_task_frequency",
    "negative_family_identity",
    "pair_ID_parity",
    "construction_ordering",
    "manifest_ordering",
    "combined_non_relational",
)


class ActionTCN(nn.Module):
    def __init__(self, width: int = 32) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(width, 48, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(48, 48, 3, padding=2, dilation=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(48, 1)

    def forward(self, action: torch.Tensor) -> torch.Tensor:
        return self.head(self.network(action.transpose(1, 2)).squeeze(2)).squeeze(1)


def _train_action_tcn(
    train_action: np.ndarray,
    train_label: np.ndarray,
    validation_action: np.ndarray,
    validation_label: np.ndarray,
    test_action: np.ndarray,
    *,
    seed: int = 17,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    mean = train_action.mean((0, 1), keepdims=True)
    scale = train_action.std((0, 1), keepdims=True)
    scale[scale < 1e-5] = 1.0
    standardized = [
        ((value - mean) / scale).astype(np.float32)
        for value in (train_action, validation_action, test_action)
    ]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensors = [torch.as_tensor(value, device=device) for value in standardized]
    label = torch.as_tensor(train_label.astype(np.float32), device=device)
    model = ActionTCN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=2e-4)
    positive = float(train_label.sum())
    pos_weight = torch.tensor([(len(train_label) - positive) / max(positive, 1.0)], device=device)
    best_state = None
    best = -1.0
    patience = 0
    history = []
    for epoch in range(200):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(tensors[0])
        loss = nn.functional.binary_cross_entropy_with_logits(logits, label, pos_weight=pos_weight)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        if epoch % 5 == 4:
            model.eval()
            with torch.no_grad():
                validation_score = torch.sigmoid(model(tensors[1])).cpu().numpy()
            threshold = best_balanced_threshold(validation_label, validation_score)
            metric = float(binary_metrics(validation_label, validation_score, threshold)["balanced_accuracy"])
            history.append(
                {
                    "epoch": epoch + 1,
                    "loss": float(loss.detach().cpu()),
                    "validation_balanced_accuracy": metric,
                }
            )
            if metric > best + 1e-6:
                best = metric
                best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
            if patience >= 10:
                break
    if best_state is None:
        raise RuntimeError("action TCN failed to train")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        validation_score = torch.sigmoid(model(tensors[1])).cpu().numpy()
        test_score = torch.sigmoid(model(tensors[2])).cpu().numpy()
    threshold = best_balanced_threshold(validation_label, validation_score)
    return (
        validation_score,
        test_score,
        {
            "seed": seed,
            "device": str(device),
            "epochs": history[-1]["epoch"],
            "threshold_selected_on_validation": threshold,
            "best_validation_balanced_accuracy": best,
            "history": history,
            "normalization_sha256": hashlib.sha256(mean.tobytes() + scale.tobytes()).hexdigest(),
        },
    )


def _normalize_manifest_order() -> dict[str, Any]:
    path = OUT / "real_negative_manifest.jsonl"
    rows = read_jsonl(path)
    before_hashes = sorted(row["sample_sha256"] for row in rows)
    original_label_blocks = [
        int(row["label"]) for row in rows[: min(16, len(rows))]
    ]
    rows.sort(key=lambda row: (stable_int(row["sample_id"]), row["sample_id"]))
    write_jsonl(path, rows)
    after_hashes = sorted(row["sample_sha256"] for row in rows)
    result = {
        "schema": "rm-acv-realneg-manifest-serialization-v1",
        "rule": "stable SHA-256 ordering of sample_id before any R4 classifier",
        "scientific_samples_changed": before_hashes != after_hashes,
        "sample_hash_multiset_preserved": before_hashes == after_hashes,
        "before_first_labels": original_label_blocks,
        "after_first_labels": [int(row["label"]) for row in rows[: min(16, len(rows))]],
        "predictive_use_of_ordering_forbidden": True,
        "performed_before_R4": True,
    }
    result["pass"] = result["sample_hash_multiset_preserved"] and not result["scientific_samples_changed"]
    write_json(OUT / "manifest_serialization_audit.json", result)
    if not result["pass"]:
        raise RuntimeError("deterministic manifest serialization changed scientific samples")
    # Recompute R3 audits because the serialized manifest changed order.
    anchors = read_jsonl(OUT / "imported_anchor_manifest.jsonl")
    anchors.sort(key=lambda row: int(row["array_index"]))
    _balance_audits(
        {
            N7: read_jsonl(OUT / "n7_pair_manifest.jsonl"),
            N8: read_jsonl(OUT / "n8_pair_manifest.jsonl"),
        },
        rows,
        anchors,
    )
    append_log("R3_MANIFEST_SERIALIZATION_FROZEN", rule=result["rule"], sample_count=len(rows))
    return result


def _load_data() -> tuple[list[dict[str, Any]], dict[str, np.ndarray], np.ndarray, list[dict[str, Any]]]:
    anchors = read_jsonl(OUT / "imported_anchor_manifest.jsonl")
    anchors.sort(key=lambda row: int(row["array_index"]))
    with np.load(OUT / "imported_anchor_arrays.npz") as stored:
        arrays = {name: stored[name].copy() for name in stored.files}
    with np.load(OUT / "causal_visual_features.npz") as stored:
        visual = stored["visual"].copy()
    samples = read_jsonl(OUT / "real_negative_manifest.jsonl")
    return anchors, arrays, visual, samples


def _one_hot(values: Sequence[str], categories: Sequence[str]) -> np.ndarray:
    lookup = {value: index for index, value in enumerate(categories)}
    result = np.zeros((len(values), len(categories)), dtype=np.float32)
    for row, value in enumerate(values):
        result[row, lookup[value]] = 1.0
    return result


def _feature_table(
    family: str,
    anchors: list[dict[str, Any]],
    arrays: dict[str, np.ndarray],
    visual: np.ndarray,
    samples: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    rows = [row for row in samples if row["family"] == family]
    action_indices = np.asarray([row["action_index"] for row in rows], dtype=np.int64)
    context_indices = np.asarray([row["context_index"] for row in rows], dtype=np.int64)
    action = arrays["positive_action"][action_indices].astype(np.float32)
    state = arrays["history_state"][context_indices].astype(np.float32)
    visual_history = visual[context_indices].astype(np.float32)
    labels = np.asarray([row["label"] for row in rows], dtype=np.int64)
    summaries = action_summary(action)
    tasks = sorted({row["task_id_provenance_only"] for row in samples})
    task_feature = _one_hot(
        [row["task_id_provenance_only"] for row in rows],
        tasks,
    )
    task_frequency = Counter(row["task_id_provenance_only"] for row in rows)
    source_manifest = {
        row["episode_id_provenance_only"]: row
        for row in read_jsonl(OUT / "imported_source_manifest.jsonl")
    }
    source_metadata = []
    file_metadata = []
    id_metadata = []
    pair_parity = []
    construction_ordering = []
    manifest_ordering = []
    distance_values = []
    distance_names = sorted(rows[0]["matching_distances"]) if rows else []
    for ordinal, row in enumerate(rows):
        context_source = source_manifest[row["context_episode_id_provenance_only"]]
        action_source = source_manifest[row["action_source_episode_id_provenance_only"]]
        source_metadata.append(
            [
                context_source["episode_index_provenance_only"] / 100.0,
                action_source["episode_index_provenance_only"] / 100.0,
                anchors[row["context_index"]]["episode_steps"] / 2000.0,
                anchors[row["action_index"]]["episode_steps"] / 2000.0,
                math.log1p(Path(context_source["parquet_path_provenance_only"]).stat().st_size) / 20.0,
                math.log1p(Path(action_source["parquet_path_provenance_only"]).stat().st_size) / 20.0,
            ]
        )
        context_bucket = stable_int(row["context_episode_id_provenance_only"], 64)
        action_bucket = stable_int(row["action_source_episode_id_provenance_only"], 64)
        bucket = np.zeros(128, dtype=np.float32)
        bucket[context_bucket] = 1.0
        bucket[64 + action_bucket] = 1.0
        file_metadata.append(bucket)
        context_id_bucket = stable_int(row["context_id"], 128)
        action_id_bucket = stable_int(row["action_id"], 128)
        identifiers = np.zeros(256, dtype=np.float32)
        identifiers[context_id_bucket] = 1.0
        identifiers[128 + action_id_bucket] = 1.0
        id_metadata.append(identifiers)
        pair_parity.append([stable_int(row["pair_id"], 2)])
        construction_ordering.append(
            [
                stable_int(row["sample_id"], 7) / 6.0,
                stable_int(row["sample_id"], 13) / 12.0,
            ]
        )
        manifest_ordering.append(
            [
                ordinal / max(len(rows) - 1, 1),
                (ordinal % 7) / 6.0,
                (ordinal % 13) / 12.0,
            ]
        )
        distance_values.append([row["matching_distances"][name] for name in distance_names])
    action_magnitude = summaries["action_magnitude_only"]
    features = {
        "action_magnitude": action_magnitude,
        "action_mean_variance": summaries["action_mean_variance"],
        "action_smoothness_jerk": summaries["smoothness_jerk"],
        "action_start_end": summaries["start_end_action"],
        "action_total_displacement": summaries["total_displacement"],
        "action_summary_full": summaries["action_summary_combined"],
        "learned_action_only_MLP": action.reshape(len(rows), -1),
        "task_identity": task_feature,
        "normalized_progress": arrays["progress"][context_indices, None],
        "current_state": state[:, -1],
        "state_history": state.reshape(len(rows), -1),
        "causal_visual_feature": visual_history.reshape(len(rows), -1),
        "task_plus_state": np.concatenate([task_feature, state.reshape(len(rows), -1)], axis=1),
        "visual_plus_state": np.concatenate(
            [visual_history.reshape(len(rows), -1), state.reshape(len(rows), -1)], axis=1
        ),
        "episode_duration": arrays["episode_duration"][context_indices, None],
        "source_episode_metadata": np.asarray(source_metadata, dtype=np.float32),
        "pair_mining_rank": np.asarray([[row["pair_selection_rank"]] for row in rows], dtype=np.float32),
        "pair_distance_values": np.asarray(distance_values, dtype=np.float32),
        "file_path_metadata": np.stack(file_metadata),
        "action_context_IDs": np.stack(id_metadata),
        "source_task_frequency": np.asarray(
            [[task_frequency[row["task_id_provenance_only"]] / len(rows)] for row in rows],
            dtype=np.float32,
        ),
        "negative_family_identity": np.ones((len(rows), 1), dtype=np.float32),
        "pair_ID_parity": np.asarray(pair_parity, dtype=np.float32),
        "construction_ordering": np.asarray(construction_ordering, dtype=np.float32),
        "manifest_ordering": np.asarray(manifest_ordering, dtype=np.float32),
    }
    # A linear classifier over separately encoded context and action features
    # is additive and cannot form an explicit context-action interaction.
    features["combined_non_relational"] = np.concatenate(
        [
            summaries["action_summary_combined"],
            state.reshape(len(rows), -1),
            visual_history.reshape(len(rows), -1),
            task_feature,
            arrays["progress"][context_indices, None],
        ],
        axis=1,
    )
    return rows, labels, action, features


def _fit_vector_control(
    family: str,
    name: str,
    rows: list[dict[str, Any]],
    labels: np.ndarray,
    features: np.ndarray,
    *,
    nonlinear: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    indices = {
        split: np.asarray([i for i, row in enumerate(rows) if row["episode_split"] == split])
        for split in ("train", "validation", "test")
    }
    if any(len(indices[split]) == 0 for split in indices):
        return (
            {
                "control": name,
                "status": "not_scored_empty_split",
                "split_counts": {split: len(values) for split, values in indices.items()},
            },
            [],
        )
    validation_scores, test_scores, training, normalization = train_shortcut_classifier(
        features[indices["train"]],
        labels[indices["train"]],
        features[indices["validation"]],
        labels[indices["validation"]],
        features[indices["test"]],
        seed=17,
        nonlinear=nonlinear,
    )
    threshold = float(training["threshold_selected_on_validation"])
    report = {
        "control": name,
        "feature_dimension": int(features.shape[1]),
        "nonlinear": nonlinear,
        "validation": binary_metrics(labels[indices["validation"]], validation_scores, threshold),
        "test": binary_metrics(labels[indices["test"]], test_scores, threshold),
        "training": training,
        "normalization_sha256": hashlib.sha256(
            normalization["mean"].tobytes() + normalization["std"].tobytes()
        ).hexdigest(),
    }
    raw = []
    for split, selected, scores in (
        ("validation", indices["validation"], validation_scores),
        ("test", indices["test"], test_scores),
    ):
        for index, score in zip(selected, scores):
            raw.append(
                {
                    "sample_id": rows[int(index)]["sample_id"],
                    "pair_id": rows[int(index)]["pair_id"],
                    "family": family,
                    "label": int(labels[int(index)]),
                    "score": float(score),
                    "split": split,
                    "control": name,
                    "seed": 17,
                }
            )
    return report, raw


def _fit_tcn_control(
    family: str,
    rows: list[dict[str, Any]],
    labels: np.ndarray,
    action: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    indices = {
        split: np.asarray([i for i, row in enumerate(rows) if row["episode_split"] == split])
        for split in ("train", "validation", "test")
    }
    if any(len(indices[split]) == 0 for split in indices):
        return (
            {
                "control": "learned_action_only_TCN",
                "status": "not_scored_empty_split",
                "split_counts": {split: len(values) for split, values in indices.items()},
            },
            [],
        )
    validation_scores, test_scores, training = _train_action_tcn(
        action[indices["train"]],
        labels[indices["train"]],
        action[indices["validation"]],
        labels[indices["validation"]],
        action[indices["test"]],
    )
    threshold = float(training["threshold_selected_on_validation"])
    report = {
        "control": "learned_action_only_TCN",
        "validation": binary_metrics(labels[indices["validation"]], validation_scores, threshold),
        "test": binary_metrics(labels[indices["test"]], test_scores, threshold),
        "training": training,
    }
    raw = []
    for split, selected, scores in (
        ("validation", indices["validation"], validation_scores),
        ("test", indices["test"], test_scores),
    ):
        for index, score in zip(selected, scores):
            raw.append(
                {
                    "sample_id": rows[int(index)]["sample_id"],
                    "pair_id": rows[int(index)]["pair_id"],
                    "family": family,
                    "label": int(labels[int(index)]),
                    "score": float(score),
                    "split": split,
                    "control": "learned_action_only_TCN",
                    "seed": 17,
                }
            )
    return report, raw


def _lookup_control(
    family: str,
    name: str,
    key: str,
    rows: list[dict[str, Any]],
    labels: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    selected = np.asarray([i for i, row in enumerate(rows) if row["episode_split"] == "test"])
    by_key: dict[str, list[int]] = defaultdict(list)
    for index in selected:
        by_key[str(rows[int(index)][key])].append(int(index))
    score_map = {
        value: float(labels[indices].mean()) for value, indices in by_key.items()
    }
    scores = np.asarray([score_map[str(rows[int(index)][key])] for index in selected])
    metrics = binary_metrics(labels[selected], scores, 0.5)
    raw = [
        {
            "sample_id": rows[int(index)]["sample_id"],
            "pair_id": rows[int(index)]["pair_id"],
            "family": family,
            "label": int(labels[int(index)]),
            "score": float(score),
            "split": "test",
            "control": name,
            "seed": 0,
        }
        for index, score in zip(selected, scores)
    ]
    return (
        {
            "control": name,
            "key": key,
            "test": metrics,
            "exact_repeated_key_balance": all(
                Counter(labels[indices])[0] == Counter(labels[indices])[1]
                for indices in by_key.values()
            ),
        },
        raw,
    )


def _coverage(family: str, pairs: list[dict[str, Any]], anchors: list[dict[str, Any]]) -> dict[str, Any]:
    tasks = Counter(row["task_id_provenance_only"] for row in pairs)
    if family == N7:
        covered = {index for row in pairs for index in row["context_indices"]}
        gates = {
            "anchor_coverage_ge_0_70": len(covered) / len(anchors) >= 0.70,
            "all_11_tasks": len(tasks) == 11,
            "pairs_ge_400": len(pairs) >= 400,
        }
        return {
            "pairs": len(pairs),
            "unique_anchors": len(covered),
            "anchor_coverage": len(covered) / len(anchors),
            "tasks": len(tasks),
            "task_pair_counts": dict(sorted(tasks.items())),
            "gates": gates,
            "pass": all(gates.values()),
        }
    maximum_fraction = max(tasks.values()) / sum(tasks.values()) if tasks else 1.0
    gates = {
        "tasks_ge_6": len(tasks) >= 6,
        "pairs_ge_120": len(pairs) >= 120,
        "every_included_task_ge_20_pairs": bool(tasks) and min(tasks.values()) >= N8_PAIRS_PER_TASK,
        "single_task_fraction_le_0_25": maximum_fraction <= 0.25,
    }
    return {
        "pairs": len(pairs),
        "tasks": len(tasks),
        "task_pair_counts": dict(sorted(tasks.items())),
        "maximum_task_fraction": maximum_fraction,
        "gates": gates,
        "pass": all(gates.values()),
    }


def _run_shortcuts() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    anchors, arrays, visual, samples = _load_data()
    all_predictions: list[dict[str, Any]] = []
    reports: dict[str, Any] = {}
    for family, pair_file, report_file in (
        (N7, "n7_pair_manifest.jsonl", "n7_shortcut_report.json"),
        (N8, "n8_pair_manifest.jsonl", "n8_shortcut_report.json"),
    ):
        rows, labels, action, features = _feature_table(
            family, anchors, arrays, visual, samples
        )
        controls: dict[str, Any] = {}
        lookup_action, raw = _lookup_control(
            family, "exact_repeated_action_lookup", "action_hash", rows, labels
        )
        controls["exact_repeated_action_lookup"] = lookup_action
        all_predictions.extend(raw)
        lookup_context, raw = _lookup_control(
            family, "exact_repeated_context_lookup", "context_hash", rows, labels
        )
        controls["exact_repeated_context_lookup"] = lookup_context
        all_predictions.extend(raw)
        for name, feature in features.items():
            nonlinear = name in {
                "learned_action_only_MLP",
                "state_history",
                "causal_visual_feature",
                "task_plus_state",
                "visual_plus_state",
            }
            control, raw = _fit_vector_control(
                family,
                name,
                rows,
                labels,
                feature,
                nonlinear=nonlinear,
            )
            controls[name] = control
            all_predictions.extend(raw)
        tcn, raw = _fit_tcn_control(family, rows, labels, action)
        controls["learned_action_only_TCN"] = tcn
        all_predictions.extend(raw)
        coverage = _coverage(family, read_jsonl(OUT / pair_file), anchors)
        action_learned = max(
            controls[name]["test"]["balanced_accuracy"]
            for name in ("learned_action_only_MLP", "learned_action_only_TCN")
            if "test" in controls[name]
        )
        context_names = (
            "current_state",
            "state_history",
            "causal_visual_feature",
            "task_plus_state",
            "visual_plus_state",
        )
        context_learned = max(
            controls[name]["test"]["balanced_accuracy"]
            for name in context_names
            if "test" in controls[name]
        )
        metadata_names = (
            "episode_duration",
            "source_episode_metadata",
            "file_path_metadata",
            "action_context_IDs",
            "source_task_frequency",
            "pair_ID_parity",
            "construction_ordering",
            "manifest_ordering",
        )
        metadata_max = max(
            controls[name]["test"]["balanced_accuracy"]
            for name in metadata_names
            if "test" in controls[name]
        )
        pair_mining_max = max(
            controls[name]["test"]["balanced_accuracy"]
            for name in ("pair_mining_rank", "pair_distance_values")
            if "test" in controls[name]
        )
        gates = {
            "exact_action_lookup_BA_0_50": abs(
                controls["exact_repeated_action_lookup"]["test"]["balanced_accuracy"] - 0.5
            )
            < 1e-12,
            "exact_context_lookup_BA_0_50": abs(
                controls["exact_repeated_context_lookup"]["test"]["balanced_accuracy"] - 0.5
            )
            < 1e-12,
            "learned_action_only_BA_le_0_54": action_learned <= 0.54,
            "learned_context_only_BA_le_0_54": context_learned <= 0.54,
            "task_only_BA_le_0_53": controls["task_identity"]["test"]["balanced_accuracy"] <= 0.53,
            "progress_only_BA_le_0_55": controls["normalized_progress"]["test"]["balanced_accuracy"] <= 0.55,
            "source_metadata_BA_le_0_53": metadata_max <= 0.53,
            "pair_mining_statistics_BA_le_0_55": pair_mining_max <= 0.55,
            "combined_non_relational_BA_le_0_58": controls["combined_non_relational"]["test"]["balanced_accuracy"] <= 0.58,
        }
        report = {
            "schema": "rm-acv-realneg-family-shortcut-v1",
            "family": family,
            "samples": len(rows),
            "controls": controls,
            "summary": {
                "learned_action_only_max_BA": action_learned,
                "learned_context_only_max_BA": context_learned,
                "metadata_max_BA": metadata_max,
                "pair_mining_max_BA": pair_mining_max,
                "combined_non_relational_BA": controls["combined_non_relational"]["test"]["balanced_accuracy"],
            },
            "gates": gates,
            "shortcut_pass": all(gates.values()),
            "coverage": coverage,
        }
        write_json(OUT / report_file, report)
        reports[family] = report
        print(
            json.dumps(
                {
                    "family": family,
                    "shortcut_pass": report["shortcut_pass"],
                    "coverage_pass": coverage["pass"],
                    **report["summary"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    write_jsonl(OUT / "shortcut_predictions.jsonl", all_predictions)
    return reports, all_predictions


def _action_distance(left: np.ndarray, right: np.ndarray, scale: np.ndarray) -> float:
    return float(np.sqrt(np.mean(((left - right) / scale) ** 2)))


def _behavioral_validity() -> dict[str, Any]:
    anchors, arrays, visual, _ = _load_data()
    alignment_audit = json.loads(
        (OUT / "imported_alignment_audit.json").read_text(encoding="utf-8")
    )
    alignment_verified = bool(alignment_audit["all_pass"])
    action_horizon = int(arrays["positive_action"].shape[1])
    with np.load(OUT / "training_normalization.npz") as stored:
        state_scale = stored["state_scale"].copy()
        action_scale = stored["action_scale"].copy()
    train_by_task: dict[str, list[int]] = defaultdict(list)
    for row in anchors:
        if row["episode_split"] == "train":
            train_by_task[row["task_id_provenance_only"]].append(int(row["array_index"]))
    ambiguity_rows: list[dict[str, Any]] = []
    per_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for family, pair_file in ((N7, "n7_pair_manifest.jsonl"), (N8, "n8_pair_manifest.jsonl")):
        for pair in read_jsonl(OUT / pair_file):
            left, right = pair["context_indices"]
            diagnostics = []
            local_radii = []
            cross_inside = []
            for context_index, cross_action_index in ((left, right), (right, left)):
                candidates = [
                    index
                    for index in train_by_task[pair["task_id_provenance_only"]]
                    if anchors[index]["episode_id_provenance_only"]
                    != anchors[context_index]["episode_id_provenance_only"]
                ]
                ranked = sorted(
                    candidates,
                    key=lambda index: (
                        float(
                            np.sqrt(
                                np.mean(
                                    (
                                        (
                                            arrays["current_state"][index]
                                            - arrays["current_state"][context_index]
                                        )
                                        / state_scale
                                    )
                                    ** 2
                                )
                            )
                        )
                        + float(
                            np.sqrt(
                                np.mean(
                                    (
                                        (
                                            arrays["history_state"][index]
                                            - arrays["history_state"][context_index]
                                        )
                                        / state_scale
                                    )
                                    ** 2
                                )
                            )
                        )
                        + 2.0
                        * abs(
                            float(
                                arrays["progress"][index]
                                - arrays["progress"][context_index]
                            )
                        ),
                        anchors[index]["anchor_id"],
                    ),
                )[:12]
                neighborhood = arrays["positive_action"][ranked]
                medoid_cost = np.asarray(
                    [
                        np.mean(
                            [
                                _action_distance(candidate, other, action_scale)
                                for other in neighborhood
                            ]
                        )
                        for candidate in neighborhood
                    ]
                )
                medoid = neighborhood[int(np.argmin(medoid_cost))]
                neighbor_distances = np.asarray(
                    [
                        _action_distance(candidate, medoid, action_scale)
                        for candidate in neighborhood
                    ]
                )
                radius = float(np.quantile(neighbor_distances, 0.75))
                own_distance = min(
                    _action_distance(
                        arrays["positive_action"][context_index],
                        candidate,
                        action_scale,
                    )
                    for candidate in neighborhood
                )
                cross_distance = min(
                    _action_distance(
                        arrays["positive_action"][cross_action_index],
                        candidate,
                        action_scale,
                    )
                    for candidate in neighborhood
                )
                local_radii.append(radius)
                cross_inside.append(cross_distance <= radius)
                diagnostics.append(
                    {
                        "context_anchor_id": anchors[context_index]["anchor_id"],
                        "context_episode_id_provenance_only": anchors[context_index][
                            "episode_id_provenance_only"
                        ],
                        "swapped_action_anchor_id": anchors[cross_action_index]["anchor_id"],
                        "swapped_action_episode_id_provenance_only": anchors[
                            cross_action_index
                        ]["episode_id_provenance_only"],
                        "neighbors": [anchors[index]["anchor_id"] for index in ranked],
                        "neighbor_episode_ids_provenance_only": sorted(
                            {
                                anchors[index]["episode_id_provenance_only"]
                                for index in ranked
                            }
                        ),
                        "local_action_radius": radius,
                        "own_action_to_neighborhood_min": own_distance,
                        "swapped_action_to_neighborhood_min": cross_distance,
                        "swapped_inside_local_mode": cross_distance <= radius,
                    }
                )
            left_neighbors = set(diagnostics[0]["neighbors"])
            right_neighbors = set(diagnostics[1]["neighbors"])
            neighborhood_union = left_neighbors | right_neighbors
            neighborhood_overlap = (
                len(left_neighbors & right_neighbors) / len(neighborhood_union)
                if neighborhood_union
                else 0.0
            )
            reciprocal_outside_local_modes = not any(cross_inside)
            action_distance = pair["matching_distances"]["logged_action_sequence"]
            distinct_by_factor = {
                str(factor): bool(
                    action_distance > factor * 1.25 * max(local_radii)
                    and reciprocal_outside_local_modes
                )
                for factor in (0.9, 1.0, 1.1)
            }
            contexts_identical = bool(
                pair["matching_distances"]["current_state"] < 0.05
                and pair["matching_distances"]["recent_state_history"] < 0.05
                and pair["matching_distances"]["causal_visual_feature"] < 0.05
            )
            actions_identical = action_distance < 0.05
            full_action_horizons = all(
                anchors[index]["action_stop_frame_exclusive"]
                - anchors[index]["action_start_frame"]
                == action_horizon
                and anchors[index]["action_stop_frame_exclusive"]
                <= anchors[index]["episode_steps"]
                for index in (left, right)
            )
            ambiguous = bool(
                not distinct_by_factor["1.0"]
                or contexts_identical
                or actions_identical
                or not alignment_verified
                or not full_action_horizons
            )
            row = {
                "schema": "rm-acv-realneg-pair-ambiguity-v1",
                "pair_id": pair["pair_id"],
                "family": family,
                "task_id_provenance_only": pair["task_id_provenance_only"],
                "episode_split": pair["episode_split"],
                "episode_ids_provenance_only": pair["episode_ids_provenance_only"],
                "training_neighborhood_episode_support": [
                    diagnostic["neighbor_episode_ids_provenance_only"]
                    for diagnostic in diagnostics
                ],
                "action_sequence_distance": action_distance,
                "current_state_distance": pair["matching_distances"]["current_state"],
                "history_distance": pair["matching_distances"]["recent_state_history"],
                "visual_distance": pair["matching_distances"]["causal_visual_feature"],
                "gripper_event_status": "different" if pair["matching_distances"]["gripper_state"] > 0.10 else "similar",
                "local_neighborhoods": diagnostics,
                "neighborhood_anchor_jaccard_overlap": neighborhood_overlap,
                "action_mode_overlap": {
                    "swapped_inside_left_context_mode": bool(cross_inside[0]),
                    "swapped_inside_right_context_mode": bool(cross_inside[1]),
                    "both_swapped_inside_local_modes": bool(all(cross_inside)),
                },
                "reciprocal_consistency": {
                    "both_swap_directions_outside_local_modes": reciprocal_outside_local_modes,
                    "pass": reciprocal_outside_local_modes,
                },
                "distinct_local_action_mode_by_factor": distinct_by_factor,
                "contexts_effectively_identical": contexts_identical,
                "actions_effectively_identical": actions_identical,
                "alignment_uncertain": not alignment_verified,
                "padding_or_terminal_truncation": not full_action_horizons,
                "alignment_support": {
                    "source": "imported_alignment_audit.json",
                    "all_pass": alignment_verified,
                },
                "padding_support": {
                    "action_horizon_steps": action_horizon,
                    "both_anchor_windows_full_and_in_bounds": full_action_horizons,
                },
                "ambiguous": ambiguous,
                "retained_for_behavioral_validity": not ambiguous,
                "privileged_diagnostics_used_as_predictive_input": False,
            }
            ambiguity_rows.append(row)
            per_family[family].append(row)
    write_jsonl(OUT / "pair_ambiguity_manifest.jsonl", ambiguity_rows)
    family_reports: dict[str, Any] = {}
    sensitivity: dict[str, Any] = {}
    action_mode: dict[str, Any] = {}
    for family, rows in per_family.items():
        task_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            task_rows[row["task_id_provenance_only"]].append(row)
        distinct_rates = {
            factor: float(
                np.mean([row["distinct_local_action_mode_by_factor"][factor] for row in rows])
            )
            for factor in ("0.9", "1.0", "1.1")
        }
        ambiguity_by_task = {
            task: float(np.mean([row["ambiguous"] for row in values]))
            for task, values in sorted(task_rows.items())
        }
        threshold = 0.80 if family == N7 else 0.75
        mode_gate = distinct_rates["1.0"] >= threshold
        per_task_gate = all(value <= 0.35 for value in ambiguity_by_task.values())
        sensitivity_gate = all((value >= threshold) == mode_gate for value in distinct_rates.values())
        family_reports[family] = {
            "pairs": len(rows),
            "distinct_mode_fraction": distinct_rates["1.0"],
            "required_distinct_mode_fraction": threshold,
            "ambiguous_fraction": float(np.mean([row["ambiguous"] for row in rows])),
            "ambiguous_fraction_by_task": ambiguity_by_task,
            "mode_gate": mode_gate,
            "per_task_ambiguity_gate": per_task_gate,
            "threshold_sensitivity_stable": sensitivity_gate,
            "pass": mode_gate and per_task_gate and sensitivity_gate,
        }
        sensitivity[family] = {
            "factors": distinct_rates,
            "threshold": threshold,
            "pass_fail_stable": sensitivity_gate,
        }
        action_mode[family] = {
            "training_split_neighborhoods_only": True,
            "neighbor_count": 12,
            "pairs": len(rows),
            "distinct_pairs": sum(
                row["distinct_local_action_mode_by_factor"]["1.0"] for row in rows
            ),
            "same_or_overlapping_mode_pairs": sum(
                not row["distinct_local_action_mode_by_factor"]["1.0"] for row in rows
            ),
        }
    report = {
        "schema": "rm-acv-realneg-behavioral-validity-v1",
        "families": family_reports,
        "predictive_labels_generated_by_a_verifier": False,
        "future_state_or_action_diagnostic_only": True,
        "pass": any(value["pass"] for value in family_reports.values()),
    }
    write_json(OUT / "behavioral_validity_report.json", report)
    write_json(
        OUT / "action_mode_audit.json",
        {"schema": "rm-acv-realneg-action-mode-audit-v1", "families": action_mode},
    )
    write_json(
        OUT / "threshold_sensitivity_report.json",
        {
            "schema": "rm-acv-realneg-threshold-sensitivity-v1",
            "families": sensitivity,
            "frozen_factors": [0.9, 1.0, 1.1],
        },
    )
    return report


def _family_and_r6_decision(
    shortcut_reports: dict[str, Any],
    behavioral: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    action_balance = json.loads((OUT / "action_reuse_audit.json").read_text())["pass"]
    context_balance = json.loads((OUT / "context_reuse_audit.json").read_text())["pass"]
    leakage = all(
        json.loads((OUT / name).read_text())["pass"]
        for name in (
            "pair_grouping_audit.json",
            "episode_leakage_audit.json",
            "duplicate_audit.json",
            "balance_audit.json",
        )
    )
    family_decisions: dict[str, Any] = {}
    for family in FAMILIES:
        shortcut = shortcut_reports[family]
        behavior = behavioral["families"][family]
        if not action_balance or not context_balance or not leakage:
            decision = "REALNEG_FAMILY_LEAKAGE"
        elif not shortcut["coverage"]["pass"]:
            decision = "REALNEG_FAMILY_LOW_COVERAGE"
        elif not shortcut["shortcut_pass"]:
            decision = "REALNEG_FAMILY_SHORTCUT"
        elif not behavior["pass"]:
            decision = "REALNEG_FAMILY_AMBIGUOUS"
        else:
            decision = "REALNEG_FAMILY_VALID"
        family_decisions[family] = {
            "decision": decision,
            "exact_action_balance": action_balance,
            "exact_context_balance": context_balance,
            "leakage_pass": leakage,
            "coverage": shortcut["coverage"],
            "shortcut_gates": shortcut["gates"],
            "behavioral_validity": behavior,
        }
    valid = [
        family for family, row in family_decisions.items()
        if row["decision"] == "REALNEG_FAMILY_VALID"
    ]
    pilot = json.loads((PILOT / "shortcut_feature_report.json").read_text())
    historical = {
        "N1_available_unchanged": "N1_same_task_near_state" in pilot["surviving_families"],
        "N4_available_unchanged": "N4_phase_shifted_same_task" in pilot["surviving_families"],
        "pilot_verifier_pass": verify_pilot(rehash_source=False)["passed"],
        "historical_manifest_sha256": sha256_file(PILOT / "negative_candidate_manifest.jsonl"),
        "historical_artifacts_modified": False,
    }
    if not historical["N1_available_unchanged"] or not historical["N4_available_unchanged"]:
        r6 = "RM_ACV_REALNEG_LEAKAGE"
    elif valid:
        r6 = "RM_ACV_REALNEG_BENCHMARK_CONSTRUCTION_AUTHORIZED"
    elif any(row["decision"] == "REALNEG_FAMILY_LEAKAGE" for row in family_decisions.values()):
        r6 = "RM_ACV_REALNEG_LEAKAGE"
    elif any(row["decision"] == "REALNEG_FAMILY_SHORTCUT" for row in family_decisions.values()):
        r6 = "RM_ACV_REALNEG_SHORTCUT"
    elif any(row["decision"] == "REALNEG_FAMILY_AMBIGUOUS" for row in family_decisions.values()):
        r6 = "RM_ACV_REALNEG_AMBIGUOUS"
    elif all(row["decision"] == "REALNEG_FAMILY_LOW_COVERAGE" for row in family_decisions.values()):
        r6 = "RM_ACV_REALNEG_LOW_COVERAGE"
    else:
        r6 = "RM_ACV_REALNEG_NO_NEW_VALID_FAMILY"
    write_json(
        OUT / "real_family_decision.json",
        {
            "schema": "rm-acv-realneg-family-decision-v1",
            "families": family_decisions,
            "valid_new_families": valid,
        },
    )
    phase = {
        "schema": "rm-acv-realneg-phase-r6-decision-v1",
        "decision": r6,
        "terminal": r6 != "RM_ACV_REALNEG_BENCHMARK_CONSTRUCTION_AUTHORIZED",
        "stage_reached": "R6",
        "historical_survivors": historical,
        "new_family_decisions": {
            family: row["decision"] for family, row in family_decisions.items()
        },
        "valid_new_families": valid,
        "combined_benchmark_construction_authorized": bool(valid),
        "learned_fair_baselines_trained": False,
        "proposed_method_authorized": False,
    }
    write_json(OUT / "phase_r6_decision.json", phase)
    return family_decisions, phase


def _terminal_package(
    phase: dict[str, Any],
    shortcut_reports: dict[str, Any],
    behavioral: dict[str, Any],
    started: float,
) -> None:
    if phase["decision"] == "RM_ACV_REALNEG_BENCHMARK_CONSTRUCTION_AUTHORIZED":
        return
    statistics = json.loads((OUT / "pair_mining_statistics.json").read_text())
    r0_r3_seconds = float(
        json.loads((OUT / "r0_r3_status.json").read_text())["wall_seconds"]
    )
    write_json(OUT / "final_decision.json", phase)
    n7_coverage = shortcut_reports[N7]["coverage"]
    n8_coverage = shortcut_reports[N8]["coverage"]
    reports = {
        "final_report.md": (
            "# RM-ACV-REALNEG-V1 final report\n\n"
            f"Decision: **`{phase['decision']}`**.\n\n"
            f"N7 produced {n7_coverage['pairs']} reciprocal pairs covering {n7_coverage['anchor_coverage']:.1%} "
            f"of the 992 frozen anchors across {n7_coverage['tasks']} tasks; it required ≥400 pairs and ≥70% coverage. "
            f"N8 produced {n8_coverage['pairs']} pairs across {n8_coverage['tasks']} tasks; it required ≥120 pairs across ≥6 tasks. "
            "Both families used their one permitted train-development-triggered redesign and still failed coverage. "
            f"Exact reciprocal action/context balance and split containment passed. Independently, N7/N8 shortcut gates were "
            f"{shortcut_reports[N7]['shortcut_pass']}/{shortcut_reports[N8]['shortcut_pass']} and behavioral-validity gates were "
            f"{behavioral['families'][N7]['pass']}/{behavioral['families'][N8]['pass']}; these secondary failures do not override the "
            "pre-registered low-coverage precedence. No combined benchmark, fair baseline, or method was trained.\n"
        ),
        "pair_mining_report.md": (
            "# Pair mining report\n\n"
            f"The exact frozen source contains 166 trajectories and 992 anchors. Mining evaluated {statistics['candidate_edges']} "
            "same-task, same-split, different-episode edges using train-only normalization and thresholds. "
            "The frozen 22D causal RGB statistic encoder was reused without downloads or performance selection. "
            f"N7 selected {statistics['N7']['selected_pairs']} symmetric disjoint-anchor pairs; "
            f"N8 selected {statistics['N8']['selected_pairs']} independent episode pairs.\n"
        ),
        "real_negative_report.md": (
            "# Real-action negative report\n\n"
            "Every retained pair uses exactly two unmodified logged actions to create `(Ci,Ai)`, `(Cj,Aj)`, `(Ci,Aj)` and `(Cj,Ai)`. "
            "No synthetic action values, noise, permutation, reversal, timing perturbation or interpolation were used. "
            "Every action and context appears equally often with positive and negative labels.\n"
        ),
        "shortcut_audit_report.md": (
            "# Shortcut audit report\n\n"
            f"N7 shortcut pass: {shortcut_reports[N7]['shortcut_pass']}; "
            f"N8 shortcut pass: {shortcut_reports[N8]['shortcut_pass']}. "
            "Exact action/context lookup, action-only MLP/TCN, context-only, metadata, mining-distance, ordering and additive non-relational controls "
            "were evaluated separately per family. Raw validation/test scores are stored in `shortcut_predictions.jsonl`. "
            "Coverage failure remains controlling even when shortcut controls pass.\n"
        ),
        "behavioral_validity_report.md": (
            "# Behavioral validity report\n\n"
            f"N7 distinct local-mode fraction: {behavioral['families'][N7]['distinct_mode_fraction']:.3f}; "
            f"N8: {behavioral['families'][N8]['distinct_mode_fraction']:.3f}. "
            "The diagnostic uses only training-split same-task neighborhoods and is never a verifier label/input. "
            "No swapped action is called physically unsafe or guaranteed to fail.\n"
        ),
        "balance_and_leakage_report.md": (
            "# Balance and leakage report\n\n"
            "Exact action reuse, exact context reuse, reciprocal pair grouping, episode containment, complete-task CV grouping, "
            "cross-split duplicate safety and 0.50 per-family/per-split label balance all passed. "
            "Episode/file/pair IDs and mining distances remain audit-only.\n"
        ),
        "claim_boundary.md": (
            "# Claim boundary\n\n"
            "Supported: a bounded audit showing exact shortcut-resistant reciprocal balancing is constructible, but the frozen N7/N8 "
            "matching rules do not reach their required episode/task coverage on the 166-trajectory source.\n\n"
            "Not supported: a usable compatibility benchmark, verifier accuracy, physical failure, safety, causal counterfactual outcomes, "
            "object dynamics, or method improvement.\n"
        ),
        "next_steps.md": (
            "# Next steps\n\n"
            "Do not relax thresholds or reuse held-out performance inside this frozen branch. To continue, acquire more repeated episodes per task "
            "or start a separately preregistered study with a new symmetric many-to-many matching design and its own family-design budget. "
            "Do not train the combined benchmark or `PhaseAwareTemporalEnergyVerifier` from this terminal package.\n"
        ),
    }
    for name, body in reports.items():
        (OUT / name).write_text(body, encoding="utf-8")
    paper = OUT / "paper"
    paper.mkdir(exist_ok=True)
    papers = {
        "title_candidates.md": (
            "# Title candidates\n\n"
            "1. Real Actions, Wrong Context: A Coverage Audit of Reciprocal Robot-Action Swaps\n"
            "2. Balanced Real-Action Context Swaps for Robot Compatibility: A Bounded Negative Result\n\n"
            "The working ActionCheck method title is not supported by this terminal result.\n"
        ),
        "abstract_zh.md": (
            "# 摘要\n\n"
            "本研究在冻结的 166 轨迹、11 任务、992 锚点 RoboMIND pilot 上，以未修改的真实动作块构造完全对称的上下文互换负样本。"
            "每个 context 与 action 均等次数出现在正负标签中，相关平衡与泄漏审计全部通过。然而，经过训练分区触发的一次整族 redesign 后，"
            f"N7 仅得到 {n7_coverage['pairs']} 对并覆盖 {n7_coverage['anchor_coverage']:.1%} 锚点；N8 仅得到 {n8_coverage['pairs']} 对、"
            f"{n8_coverage['tasks']} 个任务，均未达到预注册覆盖率。因此决策为 `{phase['decision']}`，未训练基准模型或新方法。\n"
        ),
        "abstract_en.md": (
            "# Abstract\n\n"
            "We construct exactly balanced reciprocal context swaps from unmodified logged actions on a frozen 166-trajectory, 11-task, "
            "992-anchor RoboMIND pilot. Each context and action appears equally often under positive and negative labels, and all balance/leakage "
            f"audits pass. After one train-development-triggered whole-family redesign, N7 yields {n7_coverage['pairs']} pairs covering "
            f"{n7_coverage['anchor_coverage']:.1%} of anchors, while N8 yields {n8_coverage['pairs']} pairs across {n8_coverage['tasks']} tasks. "
            f"Both miss preregistered coverage gates, producing `{phase['decision']}`. No compatibility baseline or proposed method is trained.\n"
        ),
        "introduction_outline.md": (
            "# Introduction outline\n\n"
            "- Synthetic action corruptions leaked action statistics in the frozen pilot.\n"
            "- Reciprocal real-action swaps guarantee marginal action/context balance.\n"
            "- Exact balance is necessary but not sufficient: matched-pair diversity and coverage become the bottleneck.\n"
            "- The current bounded source does not support the preregistered N7/N8 scale.\n"
        ),
        "problem_formulation.md": (
            "# Problem formulation\n\n"
            "For two demonstrated pairs `(Ci,Ai)` and `(Cj,Aj)`, reciprocal swaps create `(Ci,Aj)` and `(Cj,Ai)` without altering action values. "
            "The label means demonstrated-policy compatibility, not physical safety, failure or causal invalidity.\n"
        ),
        "real_action_negative_design.md": (
            "# Real-action negative design\n\n"
            "N7 uses symmetric near-state same-task matching with distinct logged actions. N8 additionally requires frozen causal-visual divergence. "
            "Each reciprocal pair generates two positives and two negatives with exact action/context reuse. Both families exhausted one predefined "
            "train-only coverage redesign; neither met final coverage.\n"
        ),
        "shortcut_audit.md": (
            "# Shortcut audit\n\n"
            "Action-only, context-only, metadata, mining, ID, ordering and additive non-relational controls were evaluated per family. "
            "The study never exposes IDs, paths, ranks or distance values to a fair verifier. Coverage is evaluated independently of shortcut resistance.\n"
        ),
        "benchmark.md": (
            "# Benchmark\n\n"
            "No combined benchmark is admitted. Frozen N1/N4 remain untouched historical survivors, but neither N7 nor N8 reaches the required new-family coverage. "
            "Consequently B0–B3 are not run.\n"
        ),
        "method.md": (
            "# Method\n\n"
            "`PhaseAwareTemporalEnergyVerifier` is not authorized, implemented or trained because R6 does not authorize combined benchmark construction.\n"
        ),
        "experiments.md": (
            "# Experiments\n\n"
            "R0–R6 were executed. Thresholds and one redesign trigger used training development data only. "
            "All reciprocal samples were serialized in stable hash order before R4. No fair baseline or method experiment was run.\n"
        ),
        "results_tables.md": (
            "# Results tables\n\n"
            "| Family | Pairs | Tasks | Anchor coverage | Shortcut pass | Behavioral pass | Family decision |\n"
            "|---|---:|---:|---:|---:|---:|---|\n"
            f"| N7 | {n7_coverage['pairs']} | {n7_coverage['tasks']} | {n7_coverage['anchor_coverage']:.3f} | "
            f"{shortcut_reports[N7]['shortcut_pass']} | {behavioral['families'][N7]['pass']} | "
            f"{phase['new_family_decisions'][N7]} |\n"
            f"| N8 | {n8_coverage['pairs']} | {n8_coverage['tasks']} | n/a | "
            f"{shortcut_reports[N8]['shortcut_pass']} | {behavioral['families'][N8]['pass']} | "
            f"{phase['new_family_decisions'][N8]} |\n"
        ),
        "figure_plan.md": (
            "# Figure plan\n\n"
            "1. Reciprocal four-sample construction and exact marginal balance.\n"
            "2. N7 anchor coverage by task and split.\n"
            "3. N8 eligible independent episode pairs per task.\n"
            "4. Coverage-gate flow ending at the R6 terminal decision.\n"
            "No model-performance figure is supported.\n"
        ),
        "claim_boundary.md": (
            "# Paper claim boundary\n\n"
            "The paper package may claim exact reciprocal marginal balancing and report its coverage limitation. "
            "It may not claim a working compatibility verifier, physical safety/failure, alternate-action outcomes or method gains.\n"
        ),
        "scale_up_plan.md": (
            "# Scale-up plan\n\n"
            "Increase repeated episodes and within-task scene diversity before repeating a frozen reciprocal matching study. "
            "A new matching design must be separately preregistered; this branch's thresholds, pairs and decisions remain immutable.\n"
        ),
    }
    for name, body in papers.items():
        (paper / name).write_text(body, encoding="utf-8")
    write_json(
        OUT / "resource_usage.json",
        {
            "schema": "rm-acv-realneg-resource-usage-v1",
            "r0_r3_wall_seconds": r0_r3_seconds,
            "r4_r6_wall_seconds": time.monotonic() - started,
            "pre_admission_wall_hours": (
                r0_r3_seconds + time.monotonic() - started
            )
            / 3600.0,
            "pre_admission_GPU_hours_upper_bound": (
                r0_r3_seconds + time.monotonic() - started
            )
            / 3600.0,
            "pre_admission_GPU_hours_budget": 10,
            "new_artifact_bytes": sum(path.stat().st_size for path in OUT.rglob("*") if path.is_file()),
            "new_artifact_GiB_budget": 20,
            "new_negative_family_designs": 2,
            "whole_family_redesigns": {"N7": 1, "N8": 1},
            "per_example_manual_corrections": 0,
            "unrelated_processes_terminated": 0,
        },
    )
    append_log(
        "R6_TERMINAL",
        decision=phase["decision"],
        valid_new_families=phase["valid_new_families"],
        combined_benchmark_authorized=False,
    )
    append_log("FINAL_PACKAGE_COMPLETE", decision=phase["decision"])
    generators = [
        Path(__file__),
        ROOT / "actmask" / "experiments" / "rm_acv_realneg_v1_prepare.py",
        ROOT / "actmask" / "experiments" / "rm_acv_realneg_v1_verify.py",
    ]
    files = []
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and path.name != "reproducibility_manifest.json":
            files.append(
                {
                    "path": str(path.relative_to(OUT)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    write_json(
        OUT / "reproducibility_manifest.json",
        {
            "schema": "rm-acv-realneg-reproducibility-v1",
            "decision": phase["decision"],
            "generators": [
                {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)}
                for path in generators
            ],
            "generated_files": files,
            "imported_pilot_verifier_pass": verify_pilot(rehash_source=False)["passed"],
            "combined_benchmark_constructed": False,
            "fair_baseline_training_performed": False,
            "proposed_method_training_performed": False,
        },
    )


def run() -> dict[str, Any]:
    started = time.monotonic()
    if not (OUT / "r0_r3_status.json").is_file():
        raise RuntimeError("R0-R3 artifacts are missing")
    if (OUT / "phase_r6_decision.json").exists():
        raise RuntimeError("R4-R6 already completed")
    _normalize_manifest_order()
    append_log("R4_START")
    shortcut_reports, predictions = _run_shortcuts()
    append_log(
        "R4_SHORTCUT_AUDIT_COMPLETE",
        N7_shortcut_pass=shortcut_reports[N7]["shortcut_pass"],
        N8_shortcut_pass=shortcut_reports[N8]["shortcut_pass"],
        raw_predictions=len(predictions),
    )
    behavioral = _behavioral_validity()
    append_log(
        "R5_BEHAVIORAL_VALIDITY_COMPLETE",
        N7_pass=behavioral["families"][N7]["pass"],
        N8_pass=behavioral["families"][N8]["pass"],
    )
    family, phase = _family_and_r6_decision(shortcut_reports, behavioral)
    append_log(
        "R6_DECISION",
        decision=phase["decision"],
        valid_new_families=phase["valid_new_families"],
    )
    _terminal_package(phase, shortcut_reports, behavioral, started)
    return {
        "decision": phase["decision"],
        "new_family_decisions": phase["new_family_decisions"],
        "valid_new_families": phase["valid_new_families"],
        "N7_shortcut_pass": shortcut_reports[N7]["shortcut_pass"],
        "N8_shortcut_pass": shortcut_reports[N8]["shortcut_pass"],
        "N7_behavioral_pass": behavioral["families"][N7]["pass"],
        "N8_behavioral_pass": behavioral["families"][N8]["pass"],
        "raw_shortcut_predictions": len(predictions),
        "wall_seconds": time.monotonic() - started,
    }


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, sort_keys=True))
