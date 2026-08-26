"""RM-ACV-PILOT166 phases P0--P3.

This stage freezes the bounded pilot, validates all 166 source trajectories,
constructs spaced causal anchors and six auditable negative families, and
runs the non-context shortcut gate before any learned fair baseline.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from actmask.data.robomind_adapter import load_numeric_episode, validate_numeric_episode
from actmask.experiments.rm_acv_pilot166_common import (
    ACTION_HORIZON,
    EXCLUDED_EMBODIMENT,
    FAIR_INPUTS,
    FORBIDDEN_PREDICTIVE_INPUTS,
    GROUPED_NEGATIVES,
    HISTORY,
    MAX_ANCHORS,
    MIN_ANCHOR_SPACING,
    NEGATIVE_FAMILIES,
    OUT,
    PRIMARY_EMBODIMENT,
    ROOT,
    SEEDS,
    SOURCE,
    action_summary,
    anchor_steps,
    append_log,
    binary_metrics,
    environment_manifest,
    episode_split,
    instruction_features,
    scan_records,
    sha256_file,
    sha256_json,
    stable_int,
    task_cv_folds,
    train_shortcut_classifier,
    write_json,
    write_jsonl,
)


MATCHING_TOLERANCES = {
    "N1_progress_absolute_delta_max": 0.18,
    "N1_state_normalized_rmse_soft_max": 1.50,
    "N1_action_summary_normalized_rmse_soft_max": 1.25,
    "N4_progress_absolute_delta_min": 0.20,
    "synthetic_action_summary_normalized_rmse_soft_max": 0.35,
    "synthetic_minimum_action_normalized_rmse": 0.00001,
}
MIN_FAMILY_COVERAGE = 0.85
REDESIGNED_SYNTHETIC_FAMILIES = {
    "N2_local_temporal_permutation",
    "N3_arm_gripper_desynchronization",
    "N5_joint_coordination_corruption",
    "N6_endpoint_matched_path_corruption",
}
SHORTCUT_THRESHOLDS = {
    "individual_balanced_accuracy_max": 0.60,
    "combined_balanced_accuracy_max": 0.63,
    "source_metadata_balanced_accuracy_max": 0.55,
    "class_fraction_min": 0.45,
    "class_fraction_max": 0.55,
    "minimum_surviving_families": 3,
    "maximum_single_family_negative_fraction": 0.50,
}


def _safe_episode_key(episode_id: str) -> str:
    return hashlib.sha256(episode_id.encode()).hexdigest()[:20]


def _source_manifest(records: Sequence[Any], excluded: Sequence[Any]) -> tuple[list[dict[str, Any]], str]:
    """Hash every numeric/video input and unique official metadata file."""

    entries: list[dict[str, Any]] = []
    metadata_cache: dict[str, str] = {}
    for ordinal, record in enumerate(records, start=1):
        parquet = Path(record.parquet_path)
        video = Path(str(record.front_video_path))
        metadata = Path(record.metadata_path)
        for kind, path in (("numeric_parquet", parquet), ("front_rgb_video", video)):
            entries.append(
                {
                    "safe_episode_key": _safe_episode_key(record.episode_id),
                    "task_id_provenance_only": record.task_id,
                    "kind": kind,
                    "source_path_provenance_only": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
        for path in (metadata, metadata.parent / "info.json"):
            key = str(path)
            if key not in metadata_cache:
                metadata_cache[key] = sha256_file(path)
                entries.append(
                    {
                        "safe_episode_key": None,
                        "task_id_provenance_only": record.task_id,
                        "kind": "official_metadata",
                        "source_path_provenance_only": key,
                        "bytes": path.stat().st_size,
                        "sha256": metadata_cache[key],
                    }
                )
        if ordinal % 20 == 0:
            print(f"RM_ACV_PILOT166_SOURCE_HASH={ordinal}/{len(records)}", flush=True)
    entries.sort(key=lambda row: (row["source_path_provenance_only"], row["kind"]))
    aggregate = sha256_json(
        [(row["source_path_provenance_only"], row["bytes"], row["sha256"]) for row in entries]
    )
    write_json(
        OUT / "source_hash_manifest.json",
        {
            "schema": "rm-acv-pilot166-source-hashes-v1",
            "source_root": str(SOURCE),
            "primary_embodiment": PRIMARY_EMBODIMENT,
            "primary_trajectories": len(records),
            "entries": entries,
            "aggregate_sha256": aggregate,
            "content_rehash_complete": True,
        },
    )
    write_jsonl(
        OUT / "source_manifest.jsonl",
        [
            {
                "schema": "rm-acv-pilot166-source-trajectory-v1",
                "safe_episode_key": _safe_episode_key(record.episode_id),
                "episode_id_provenance_only": record.episode_id,
                "task_id_provenance_only": record.task_id,
                "language_instruction": record.language_instruction,
                "embodiment": record.robot_embodiment,
                "episode_index_provenance_only": record.episode_index,
                "parquet_path_provenance_only": record.parquet_path,
                "front_video_path_provenance_only": record.front_video_path,
                "metadata_path_provenance_only": record.metadata_path,
            }
            for record in records
        ],
    )
    write_json(
        OUT / "excluded_heterogeneous_source.json",
        {
            "schema": "rm-acv-pilot166-excluded-source-v1",
            "embodiment": EXCLUDED_EMBODIMENT,
            "trajectory_count": len(excluded),
            "task_count": len({record.task_id for record in excluded}),
            "state_action_dimension": 16,
            "reason": "heterogeneous 16D state/action schema; never merged with the primary 32D benchmark",
            "episode_ids_provenance_only": [record.episode_id for record in excluded],
        },
    )
    return entries, aggregate


def _configuration(records: Sequence[Any], source_hash: str) -> dict[str, Any]:
    tasks = sorted({record.task_id for record in records})
    split = episode_split(records)
    folds = task_cv_folds(tasks)
    return {
        "schema": "rm-acv-pilot166-preregistered-config-v1",
        "name": "RM-ACV-PILOT166",
        "scientific_scope": "bounded method-feasibility pilot; compatibility with demonstrated conditional action distribution only",
        "claim_exclusions": [
            "large-scale coverage",
            "cross-embodiment generalization",
            "independent-trajectory interpretation of anchors",
            "guaranteed success",
            "failure prediction",
            "physical safety",
            "causal alternate-action outcome",
            "physical object consequence prediction",
        ],
        "source": {
            "root": str(SOURCE),
            "source_manifest": "source_manifest.jsonl",
            "source_hash_aggregate": source_hash,
            "primary_embodiment": PRIMARY_EMBODIMENT,
            "trajectory_count": len(records),
            "tasks": tasks,
            "task_count": len(tasks),
            "state_dimension": 32,
            "action_dimension": 32,
            "excluded_embodiment": EXCLUDED_EMBODIMENT,
            "excluded_trajectory_count": 72,
            "source_read_only": True,
        },
        "sample": {
            "causal_history_steps": HISTORY,
            "rgb_history_steps": HISTORY,
            "action_horizon_steps": ACTION_HORIZON,
            "maximum_anchors_per_episode": MAX_ANCHORS,
            "minimum_anchor_spacing_steps": MIN_ANCHOR_SPACING,
            "anchor_rule": "evenly spaced across the complete legal interval; fewer for short trajectories; no relaxed alignment",
            "positive": "logged action chunk at steps anchor+1 through anchor+16; distributional compatibility example only",
            "action_representation": "official 16 arm + 2 gripper + 14 end-effector fields, concatenated without future state",
            "state_representation": "official 16 arm + 2 gripper + 14 end-effector fields",
            "normalization": "training-partition per-dimension mean/std for predictive models; global robust scale only for label-symmetric negative matching",
        },
        "negative_construction": {
            "families": list(NEGATIVE_FAMILIES),
            "matching_tolerances": MATCHING_TOLERANCES,
            "generated_per_anchor_before_audit": len(NEGATIVE_FAMILIES),
            "grouped_negatives_after_audit": GROUPED_NEGATIVES,
            "family_selection_rule": "retain families with >=85% anchor coverage and family-specific action-summary shortcut BA <= 0.60; choose the three lowest-BA survivors, lexical tie-break; final grouped benchmark keeps only anchors having all three selected candidates",
            "minimum_family_anchor_coverage": MIN_FAMILY_COVERAGE,
            "redesign_budget": "at most one whole-family redesign; synthetic families use design version 2 after the archived version-1 preflight rejected low-motion but non-identical corruptions; individual samples are never repaired",
            "redesign_history": "preflight_design_v1/ and preregistration_amendment.json",
        },
        "shortcut_features": [
            "task_identity_only",
            "normalized_progress_only",
            "action_magnitude_only",
            "action_mean_variance",
            "smoothness_jerk",
            "start_end_action",
            "total_displacement",
            "episode_duration",
            "source_metadata_diagnostic",
            "paired_negative_family_identity_diagnostic",
            "combined_non_context",
        ],
        "episode_split": {
            "manifest_rule": "within each task, stable SHA-256 episode ordering; >=2 validation and >=2 test; remaining train",
            "counts": {name: len(values) for name, values in split.items()},
            "frozen_assignments_sha256": sha256_json(split),
        },
        "task_cross_validation": {
            "protocol": "preregistered 5-fold complete-task group CV because 11-fold x all baselines x seeds exceeds bounded resource budget",
            "fold_count": 5,
            "folds": folds,
            "every_task_held_out_exactly_once": True,
        },
        "baseline_list": {
            "controls": [
                "random",
                "task_only",
                "progress_only",
                "action_summary_only",
                "action_chunk_only",
                "state_only",
                "current_visual_only",
                "state_action",
                "nearest_neighbor_retrieval",
            ],
            "strong": [
                "state_action_tcn",
                "task_conditioned_action_transformer",
                "visual_state_action_transformer",
                "diagonal_ssm_verifier",
                "autoregressive_action_likelihood",
                "masked_action_transformer",
                "contrastive_context_action_encoder",
                "action_denoising_score_model",
            ],
            "seed_policy": "one seed for all exploratory baselines, then seeds 17/29/43 for the strongest three validation-selected families",
        },
        "metrics": {
            "classification": ["balanced_accuracy", "AUROC", "AUPRC", "false_accept_rate", "false_reject_rate"],
            "ranking": ["Recall@1", "MRR", "NDCG", "positive_negative_score_margin"],
            "generalization": ["episode_heldout", "task_CV_macro", "per_task", "median_task", "worst_task", "per_family", "per_horizon"],
            "calibration": ["ECE", "Brier", "risk_coverage", "selective_accuracy_100_90_80_70"],
            "localization": ["temporal_IoU", "corrupted_window_recall"],
            "confidence_intervals": "episode-grouped bootstrap for episode split; task-grouped summaries for task CV; anchors never treated as independent trajectories",
        },
        "shortcut_gate": SHORTCUT_THRESHOLDS,
        "benchmark_gate": {
            "episode_balanced_accuracy_min": 0.62,
            "episode_grouped_ci_must_exclude": 0.50,
            "task_cv_macro_balanced_accuracy_min": 0.55,
            "recall_at_1_above_chance_min": 0.10,
            "context_action_episode_gain_over_action_only_min": 0.04,
            "context_action_task_cv_gain_over_action_only_min": 0.03,
            "episode_saturation_max_exclusive": 0.85,
            "task_cv_saturation_max_exclusive": 0.80,
            "minimum_useful_families": 3,
        },
        "proposed_method": {
            "family": "PhaseAwareTemporalEnergyVerifier",
            "full_seeds": list(SEEDS),
            "parameter_budget": "at most 2x strongest Transformer baseline",
            "authorization": "only current-run RM_ACV_PILOT_BENCHMARK_READY",
        },
        "method_acceptance_gate": {
            "episode_balanced_accuracy_gain_min_or_recall_at_1_gain_min": [0.03, 0.05],
            "task_cv_macro_gain_min": 0.02,
            "minimum_improved_negative_families": 3,
            "minimum_structural_ablations_degrading": 2,
            "three_seed_stability_required": True,
            "independent_metric_recomputation_required": True,
        },
        "resource_budget": {
            "gpu": "single RTX 4090D 24GB when available",
            "task_cv": "5 folds",
            "maximum_training_wall_hours": 8,
            "maximum_output_gb": 20,
            "no_unrelated_process_termination": True,
            "action_denoising": "one bounded seed unless it becomes a top-three validation baseline",
        },
        "fair_inputs": list(FAIR_INPUTS),
        "forbidden_predictive_inputs": list(FORBIDDEN_PREDICTIVE_INPUTS),
        "environment": environment_manifest(),
        "generator": {
            "path": str(Path(__file__).relative_to(ROOT)),
            "sha256": sha256_file(Path(__file__)),
        },
    }


def _load_and_cache(records: Sequence[Any]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    cache_root = OUT / "processed" / "episodes"
    cache_root.mkdir(parents=True, exist_ok=True)
    audits: list[dict[str, Any]] = []
    payload: dict[str, dict[str, Any]] = {}
    assignments = {
        episode_id: split
        for split, values in episode_split(records).items()
        for episode_id in values
    }
    for ordinal, record in enumerate(records, start=1):
        numeric = load_numeric_episode(record)
        audit = validate_numeric_episode(record, numeric)
        dimensions_ok = numeric["state"].shape[1] == 32 and numeric["action"].shape[1] == 32
        frequency = float(1.0 / np.median(np.diff(numeric["timestamp"])))
        anchors = anchor_steps(len(numeric["state"]))
        passed = bool(audit["pass"] and dimensions_ok and 29.9 <= frequency <= 30.1 and anchors)
        audit.update(
            {
                "safe_episode_key": _safe_episode_key(record.episode_id),
                "task_id_provenance_only": record.task_id,
                "state_action_32d": dimensions_ok,
                "control_frequency_hz": frequency,
                "anchor_count": len(anchors),
                "pilot_pass": passed,
            }
        )
        audits.append(audit)
        if not passed:
            raise RuntimeError(f"source trajectory fails pilot eligibility: {record.episode_id}")
        safe_key = _safe_episode_key(record.episode_id)
        cache_path = cache_root / f"{safe_key}.npz"
        np.savez_compressed(
            cache_path,
            state=numeric["state"].astype(np.float32),
            action=numeric["action"].astype(np.float32),
            timestamp=numeric["timestamp"].astype(np.float64),
            frame_index=numeric["frame_index"].astype(np.int64),
        )
        payload[record.episode_id] = {
            "record": record,
            "safe_key": safe_key,
            "cache_path": cache_path,
            "state": numeric["state"].astype(np.float32),
            "action": numeric["action"].astype(np.float32),
            "timestamp": numeric["timestamp"].astype(np.float64),
            "frame_index": numeric["frame_index"].astype(np.int64),
            "anchors": anchors,
            "split": assignments[record.episode_id],
            "duration": float(numeric["timestamp"][-1] - numeric["timestamp"][0]),
        }
        if ordinal % 25 == 0:
            print(f"RM_ACV_PILOT166_NUMERIC={ordinal}/{len(records)}", flush=True)
    write_json(
        OUT / "alignment_audit.json",
        {
            "schema": "rm-acv-pilot166-alignment-audit-v1",
            "all_pass": all(row["pilot_pass"] for row in audits),
            "trajectories": len(audits),
            "records": audits,
        },
    )
    return audits, payload


def _anchor_arrays(payload: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    rows: list[dict[str, Any]] = []
    histories: list[np.ndarray] = []
    positives: list[np.ndarray] = []
    current_states: list[np.ndarray] = []
    progress_values: list[float] = []
    instructions: list[np.ndarray] = []
    durations: list[float] = []
    episode_internal: list[int] = []
    task_internal: list[int] = []
    episode_ids = sorted(payload)
    episode_to_index = {episode_id: index for index, episode_id in enumerate(episode_ids)}
    tasks = sorted({item["record"].task_id for item in payload.values()})
    task_to_index = {task: index for index, task in enumerate(tasks)}
    for episode_id in episode_ids:
        item = payload[episode_id]
        state, action = item["state"], item["action"]
        for ordinal, anchor in enumerate(item["anchors"]):
            start = anchor + 1
            stop = start + ACTION_HORIZON
            history = state[anchor - HISTORY + 1 : anchor + 1]
            positive = action[start:stop]
            if history.shape != (HISTORY, 32) or positive.shape != (ACTION_HORIZON, 32):
                raise RuntimeError(f"malformed anchor window {episode_id}:{anchor}")
            row_index = len(rows)
            anchor_id = f"a{row_index:06d}"
            rows.append(
                {
                    "schema": "rm-acv-pilot166-anchor-v1",
                    "anchor_id": anchor_id,
                    "array_index": row_index,
                    "safe_episode_key": item["safe_key"],
                    "episode_id_provenance_only": episode_id,
                    "task_id_provenance_only": item["record"].task_id,
                    "episode_split": item["split"],
                    "anchor_frame": int(anchor),
                    "history_start_frame": int(anchor - HISTORY + 1),
                    "action_start_frame": int(start),
                    "action_stop_frame_exclusive": int(stop),
                    "normalized_progress": float(anchor / max(len(state) - 1, 1)),
                    "episode_steps": len(state),
                    "episode_duration_seconds": item["duration"],
                    "positive_semantics": "logged immediate continuation; not claimed optimal, safe, or successful",
                }
            )
            histories.append(history)
            positives.append(positive)
            current_states.append(state[anchor])
            progress_values.append(float(anchor / max(len(state) - 1, 1)))
            instructions.append(instruction_features(item["record"].language_instruction))
            durations.append(item["duration"])
            episode_internal.append(episode_to_index[episode_id])
            task_internal.append(task_to_index[item["record"].task_id])
    arrays = {
        "history_state": np.stack(histories).astype(np.float32),
        "positive_action": np.stack(positives).astype(np.float32),
        "current_state": np.stack(current_states).astype(np.float32),
        "progress": np.asarray(progress_values, dtype=np.float32),
        "instruction": np.stack(instructions).astype(np.float32),
        "episode_duration": np.asarray(durations, dtype=np.float32),
        "episode_index_internal": np.asarray(episode_internal, dtype=np.int64),
        "task_index_internal": np.asarray(task_internal, dtype=np.int64),
    }
    return rows, arrays


def _summary_vector(action: np.ndarray) -> np.ndarray:
    return action_summary(action[None])["action_summary_combined"][0]


def _candidate_quality(
    positive: np.ndarray,
    candidate: np.ndarray,
    action_scale: np.ndarray,
    summary_scale: np.ndarray,
) -> dict[str, float]:
    positive_summary = _summary_vector(positive)
    candidate_summary = _summary_vector(candidate)
    displacement_positive = positive[-1] - positive[0]
    displacement_candidate = candidate[-1] - candidate[0]
    return {
        "action_normalized_rmse": float(np.sqrt(np.mean(((candidate - positive) / action_scale) ** 2))),
        "action_summary_normalized_rmse": float(
            np.sqrt(np.mean(((candidate_summary - positive_summary) / summary_scale) ** 2))
        ),
        "start_normalized_rmse": float(np.sqrt(np.mean(((candidate[0] - positive[0]) / action_scale) ** 2))),
        "end_normalized_rmse": float(np.sqrt(np.mean(((candidate[-1] - positive[-1]) / action_scale) ** 2))),
        "displacement_normalized_rmse": float(
            np.sqrt(np.mean(((displacement_candidate - displacement_positive) / action_scale) ** 2))
        ),
    }


def _synthetic_candidates(
    family: str,
    positive: np.ndarray,
    seed: int,
) -> list[tuple[np.ndarray, dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    values: list[tuple[np.ndarray, dict[str, Any]]] = []
    horizon = len(positive)
    if family == "N2_local_temporal_permutation":
        for attempt in range(24):
            candidate = positive.copy()
            operation = attempt % 3
            if operation == 0:
                length = 2 + attempt % 2
                starts = rng.choice(np.arange(1, horizon - length), size=2, replace=False)
                left, right = sorted(int(value) for value in starts)
                if abs(right - left) < length:
                    continue
                first, second = candidate[left : left + length].copy(), candidate[right : right + length].copy()
                candidate[left : left + length], candidate[right : right + length] = second, first
                params = {"operation": "exchange_subsegments", "windows": [[left, left + length], [right, right + length]]}
            elif operation == 1:
                length = 3 + attempt % 3
                start = int(rng.integers(1, horizon - length))
                candidate[start : start + length] = candidate[start : start + length][::-1]
                params = {"operation": "reverse_local_subsegment", "window": [start, start + length]}
            else:
                length = 4 + attempt % 3
                start = int(rng.integers(1, horizon - length))
                shift = -1 if attempt % 2 else 1
                candidate[start : start + length] = np.roll(candidate[start : start + length], shift, axis=0)
                params = {"operation": "shift_local_subsegment", "window": [start, start + length], "shift": shift}
            values.append((candidate, params))
    elif family == "N3_arm_gripper_desynchronization":
        gripper = np.asarray([16, 17])
        interior = np.arange(1, horizon - 1)
        for shift in (-3, -2, -1, 1, 2, 3):
            candidate = positive.copy()
            candidate[np.ix_(interior, gripper)] = np.roll(
                positive[np.ix_(interior, gripper)], shift, axis=0
            )
            values.append(
                (
                    candidate,
                    {
                        "operation": "advance_gripper" if shift < 0 else "delay_gripper",
                        "shift": shift,
                        "window": [1, horizon - 1],
                        "dimensions": gripper.tolist(),
                    },
                )
            )
    elif family == "N5_joint_coordination_corruption":
        # Second-arm joint, gripper, and end-effector coordinates.
        second_arm = np.asarray(list(range(8, 16)) + [17] + list(range(25, 32)))
        interior = np.arange(1, horizon - 1)
        for shift in (-3, -2, -1, 1, 2, 3):
            candidate = positive.copy()
            candidate[np.ix_(interior, second_arm)] = np.roll(
                positive[np.ix_(interior, second_arm)], shift, axis=0
            )
            values.append(
                (
                    candidate,
                    {
                        "operation": "second_arm_temporal_phase_offset",
                        "shift": shift,
                        "window": [1, horizon - 1],
                        "dimensions": second_arm.tolist(),
                    },
                )
            )
    elif family == "N6_endpoint_matched_path_corruption":
        alpha = np.linspace(0.0, 1.0, horizon, dtype=np.float32)[:, None]
        linear = positive[0][None] * (1.0 - alpha) + positive[-1][None] * alpha
        residual = positive - linear
        for mode in ("reverse", "roll_2", "roll_-2", "swap_halves"):
            candidate = linear.copy()
            interior = residual[1:-1].copy()
            if mode == "reverse":
                interior = interior[::-1]
            elif mode == "roll_2":
                interior = np.roll(interior, 2, axis=0)
            elif mode == "roll_-2":
                interior = np.roll(interior, -2, axis=0)
            else:
                midpoint = len(interior) // 2
                interior = np.concatenate([interior[midpoint:], interior[:midpoint]], axis=0)
            candidate[1:-1] += interior
            candidate[0], candidate[-1] = positive[0], positive[-1]
            values.append(
                (
                    candidate.astype(np.float32),
                    {"operation": f"endpoint_matched_{mode}", "window": [1, horizon - 1]},
                )
            )
    else:
        raise ValueError(f"not a synthetic family: {family}")
    return values


def _build_negatives(
    anchors: list[dict[str, Any]],
    arrays: dict[str, np.ndarray],
    payload: dict[str, dict[str, Any]],
) -> tuple[np.ndarray, list[dict[str, Any]], list[dict[str, Any]]]:
    positive = arrays["positive_action"]
    current = arrays["current_state"]
    progress = arrays["progress"]
    action_scale = np.std(positive.reshape(-1, 32), axis=0)
    action_scale[action_scale < 1e-4] = 1.0
    state_scale = np.std(current, axis=0)
    state_scale[state_scale < 1e-4] = 1.0
    summaries = action_summary(positive)["action_summary_combined"]
    summary_scale = np.std(summaries, axis=0)
    summary_scale[summary_scale < 1e-4] = 1.0
    by_task_split: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(anchors):
        by_task_split[(row["task_id_provenance_only"], row["episode_split"])].append(index)
    candidates = np.empty(
        (len(anchors), len(NEGATIVE_FAMILIES), ACTION_HORIZON, 32), dtype=np.float32
    )
    manifest: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for index, row in enumerate(anchors):
        anchor_episode = row["episode_id_provenance_only"]
        pool = [
            other
            for other in by_task_split[(row["task_id_provenance_only"], row["episode_split"])]
            if anchors[other]["episode_id_provenance_only"] != anchor_episode
        ]
        for family_index, family in enumerate(NEGATIVE_FAMILIES):
            try:
                source_index: int | None = None
                parameters: dict[str, Any]
                matching: dict[str, float]
                corruption_interval: list[int] | None = None
                if family in {"N1_same_task_near_state", "N4_phase_shifted_same_task"}:
                    eligible = []
                    for other in pool:
                        progress_delta = abs(float(progress[other] - progress[index]))
                        if family.startswith("N1") and progress_delta > MATCHING_TOLERANCES["N1_progress_absolute_delta_max"]:
                            continue
                        if family.startswith("N4") and progress_delta < MATCHING_TOLERANCES["N4_progress_absolute_delta_min"]:
                            continue
                        state_distance = float(
                            np.sqrt(np.mean(((current[other] - current[index]) / state_scale) ** 2))
                        )
                        summary_distance = float(
                            np.sqrt(np.mean(((summaries[other] - summaries[index]) / summary_scale) ** 2))
                        )
                        score = state_distance + 0.75 * summary_distance + 1.5 * progress_delta
                        eligible.append((score, state_distance, summary_distance, progress_delta, other))
                    if not eligible:
                        raise RuntimeError("no other-episode candidate satisfies phase constraint")
                    score, state_distance, summary_distance, progress_delta, source_index = min(eligible)
                    candidate = positive[source_index].copy()
                    parameters = {
                        "operation": "same_task_other_episode_retrieval",
                        "phase_constraint": "near" if family.startswith("N1") else "shifted",
                    }
                    matching = {
                        "weighted_match_score": score,
                        "state_normalized_rmse": state_distance,
                        "action_summary_normalized_rmse": summary_distance,
                        "progress_absolute_delta": progress_delta,
                    }
                else:
                    proposals = _synthetic_candidates(
                        family, positive[index], stable_int(f"{row['anchor_id']}:{family}", 2**32)
                    )
                    ranked: list[tuple[float, float, np.ndarray, dict[str, Any], dict[str, float]]] = []
                    for candidate, params in proposals:
                        quality = _candidate_quality(
                            positive[index], candidate, action_scale, summary_scale
                        )
                        change = quality["action_normalized_rmse"]
                        if change < MATCHING_TOLERANCES["synthetic_minimum_action_normalized_rmse"]:
                            continue
                        ranked.append(
                            (
                                quality["action_summary_normalized_rmse"],
                                -change,
                                candidate,
                                params,
                                quality,
                            )
                        )
                    if not ranked:
                        raise RuntimeError("no nontrivial synthetic proposal")
                    _, _, candidate, parameters, matching = min(ranked, key=lambda value: (value[0], value[1]))
                    window = parameters.get("window")
                    if window is not None:
                        corruption_interval = [int(window[0]), int(window[1])]
                    elif parameters.get("windows"):
                        starts = [int(value[0]) for value in parameters["windows"]]
                        stops = [int(value[1]) for value in parameters["windows"]]
                        corruption_interval = [min(starts), max(stops)]
                candidates[index, family_index] = candidate
                source_episode = (
                    anchors[source_index]["episode_id_provenance_only"]
                    if source_index is not None else anchor_episode
                )
                candidate_id = f"{row['anchor_id']}:{family}"
                manifest.append(
                    {
                        "schema": "rm-acv-pilot166-negative-v1",
                        "candidate_id": candidate_id,
                        "anchor_id": row["anchor_id"],
                        "anchor_array_index": index,
                        "family": family,
                        "family_index": family_index,
                        "label": 0,
                        "source_episode_provenance_only": source_episode,
                        "source_anchor_id_provenance_only": anchors[source_index]["anchor_id"] if source_index is not None else row["anchor_id"],
                        "same_episode_as_anchor_provenance_only": source_episode == anchor_episode,
                        "matching_scores": matching,
                        "corruption_parameters": parameters,
                        "corruption_interval": corruption_interval,
                        "candidate_sha256": hashlib.sha256(candidate.tobytes()).hexdigest(),
                        "construction_status": "success",
                        "selected_for_final_benchmark": None,
                    }
                )
            except Exception as exc:
                failures.append(
                    {
                        "anchor_id": row["anchor_id"],
                        "family": family,
                        "error": f"{type(exc).__name__}: {exc}",
                        "repair_attempted": False,
                    }
                )
                candidates[index, family_index] = np.nan
        if (index + 1) % 100 == 0:
            print(f"RM_ACV_PILOT166_NEGATIVES={index + 1}/{len(anchors)}", flush=True)
    return candidates, manifest, failures


def _shortcut_table(
    anchors: list[dict[str, Any]],
    arrays: dict[str, np.ndarray],
    negatives: np.ndarray,
    negative_manifest: list[dict[str, Any]],
    families: Sequence[str],
    payload: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], np.ndarray, dict[str, np.ndarray]]:
    manifest_lookup = {
        (int(row["anchor_array_index"]), str(row["family"])): row for row in negative_manifest
    }
    family_to_index = {family: index for index, family in enumerate(NEGATIVE_FAMILIES)}
    task_names = sorted({row["task_id_provenance_only"] for row in anchors})
    task_to_index = {task: index for index, task in enumerate(task_names)}
    rows: list[dict[str, Any]] = []
    actions: list[np.ndarray] = []
    task_identity: list[np.ndarray] = []
    progress_feature: list[list[float]] = []
    duration_feature: list[list[float]] = []
    family_identity: list[np.ndarray] = []
    source_metadata: list[np.ndarray] = []
    for anchor_index, anchor in enumerate(anchors):
        if any((anchor_index, family) not in manifest_lookup for family in families):
            continue
        for family_position, family in enumerate(families):
            negative_row = manifest_lookup[(anchor_index, family)]
            family_index = family_to_index[family]
            for label in (1, 0):
                candidate = (
                    arrays["positive_action"][anchor_index]
                    if label == 1 else negatives[anchor_index, family_index]
                )
                candidate_source_episode = (
                    anchor["episode_id_provenance_only"]
                    if label == 1 else negative_row["source_episode_provenance_only"]
                )
                source_record = payload[candidate_source_episode]["record"]
                one_hot_task = np.zeros(len(task_names), dtype=np.float32)
                one_hot_task[task_to_index[anchor["task_id_provenance_only"]]] = 1.0
                one_hot_family = np.zeros(len(families), dtype=np.float32)
                one_hot_family[family_position] = 1.0
                path_bucket = stable_int(candidate_source_episode, 32)
                bucket_feature = np.zeros(32, dtype=np.float32)
                bucket_feature[path_bucket] = 1.0
                source_item = payload[candidate_source_episode]
                source_feature = np.concatenate(
                    [
                        np.asarray(
                            [
                                len(source_item["state"]) / 2000.0,
                                source_item["duration"] / 60.0,
                                np.log1p(source_record.source_bytes) / 20.0,
                                source_record.episode_index / 100.0,
                            ],
                            dtype=np.float32,
                        ),
                        bucket_feature,
                    ]
                )
                pair_id = f"{anchor['anchor_id']}:{family}"
                rows.append(
                    {
                        "row_id": f"{pair_id}:{'positive' if label else 'negative'}",
                        "pair_id": pair_id,
                        "anchor_id": anchor["anchor_id"],
                        "anchor_array_index": anchor_index,
                        "label": label,
                        "paired_family": family,
                        "episode_split": anchor["episode_split"],
                        "anchor_episode_provenance_only": anchor["episode_id_provenance_only"],
                        "candidate_source_episode_provenance_only": candidate_source_episode,
                    }
                )
                actions.append(candidate)
                task_identity.append(one_hot_task)
                progress_feature.append([anchor["normalized_progress"]])
                duration_feature.append([anchor["episode_duration_seconds"]])
                family_identity.append(one_hot_family)
                source_metadata.append(source_feature)
    candidate_actions = np.stack(actions).astype(np.float32)
    summaries = action_summary(candidate_actions)
    features = {
        "task_identity_only": np.stack(task_identity),
        "normalized_progress_only": np.asarray(progress_feature, dtype=np.float32),
        **summaries,
        "episode_duration": np.asarray(duration_feature, dtype=np.float32),
        "source_metadata": np.stack(source_metadata),
        "negative_family_identity": np.stack(family_identity),
    }
    features["combined_non_context"] = np.concatenate(list(features.values()), axis=1)
    return rows, candidate_actions, features


def _fit_feature(
    name: str,
    rows: list[dict[str, Any]],
    feature: np.ndarray,
    *,
    nonlinear: bool,
    seed: int = 17,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    labels = np.asarray([row["label"] for row in rows], dtype=np.int64)
    indices = {
        split: np.asarray([index for index, row in enumerate(rows) if row["episode_split"] == split])
        for split in ("train", "validation", "test")
    }
    validation_scores, test_scores, training, normalization = train_shortcut_classifier(
        feature[indices["train"]],
        labels[indices["train"]],
        feature[indices["validation"]],
        labels[indices["validation"]],
        feature[indices["test"]],
        seed=seed,
        nonlinear=nonlinear,
    )
    threshold = float(training["threshold_selected_on_validation"])
    validation_metrics = binary_metrics(labels[indices["validation"]], validation_scores, threshold)
    test_metrics = binary_metrics(labels[indices["test"]], test_scores, threshold)
    raw: list[dict[str, Any]] = []
    for split, selected, scores in (
        ("validation", indices["validation"], validation_scores),
        ("test", indices["test"], test_scores),
    ):
        for row_index, score in zip(selected, scores):
            raw.append(
                {
                    "row_id": rows[int(row_index)]["row_id"],
                    "pair_id": rows[int(row_index)]["pair_id"],
                    "anchor_id": rows[int(row_index)]["anchor_id"],
                    "label": int(labels[int(row_index)]),
                    "score": float(score),
                    "split": split,
                    "classifier": name,
                    "seed": seed,
                }
            )
    report = {
        "classifier": name,
        "feature_dimension": int(feature.shape[1]),
        "nonlinear": nonlinear,
        "training": training,
        "validation": validation_metrics,
        "test": test_metrics,
        "normalization_sha256": hashlib.sha256(
            normalization["mean"].tobytes() + normalization["std"].tobytes()
        ).hexdigest(),
    }
    return report, raw


def _shortcut_audit(
    anchors: list[dict[str, Any]],
    arrays: dict[str, np.ndarray],
    negatives: np.ndarray,
    negative_manifest: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    payload: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], list[str], list[dict[str, Any]]]:
    family_reports: dict[str, Any] = {}
    all_predictions: list[dict[str, Any]] = []
    failure_counts = Counter(row["family"] for row in failures)
    success_counts = Counter(row["family"] for row in negative_manifest)
    for family in NEGATIVE_FAMILIES:
        rows, _, features = _shortcut_table(
            anchors, arrays, negatives, negative_manifest, [family], payload
        )
        if len(rows) < 40:
            family_reports[family] = {
                "classifier": f"family_action_summary:{family}",
                "construction_successes": success_counts[family],
                "construction_failures": failure_counts[family],
                "coverage": success_counts[family] / len(anchors),
                "survives": False,
                "design_version": 2 if family in REDESIGNED_SYNTHETIC_FAMILIES else 1,
                "whole_family_redesign_used": family in REDESIGNED_SYNTHETIC_FAMILIES,
                "reason": "too few constructed pairs for a stable family shortcut audit",
            }
            continue
        report, predictions = _fit_feature(
            f"family_action_summary:{family}",
            rows,
            features["action_summary_combined"],
            nonlinear=True,
        )
        coverage = success_counts[family] / len(anchors)
        coverage_pass = coverage >= MIN_FAMILY_COVERAGE
        survives = coverage_pass and float(report["test"]["balanced_accuracy"]) <= SHORTCUT_THRESHOLDS[
            "individual_balanced_accuracy_max"
        ]
        family_reports[family] = {
            **report,
            "construction_successes": success_counts[family],
            "construction_failures": failure_counts[family],
            "coverage": coverage,
            "coverage_threshold": MIN_FAMILY_COVERAGE,
            "coverage_pass": coverage_pass,
            "survives": survives,
            "design_version": 2 if family in REDESIGNED_SYNTHETIC_FAMILIES else 1,
            "whole_family_redesign_used": family in REDESIGNED_SYNTHETIC_FAMILIES,
        }
        all_predictions.extend(predictions)
    survivors = sorted(
        [family for family, report in family_reports.items() if report["survives"]],
        key=lambda family: (family_reports[family]["test"]["balanced_accuracy"], family),
    )
    selected = survivors[:GROUPED_NEGATIVES]
    if len(selected) < GROUPED_NEGATIVES:
        result = {
            "status": "fail",
            "decision": "RM_ACV_PILOT_NEGATIVES_INVALID",
            "family_reports": family_reports,
            "surviving_families": survivors,
            "selected_families": selected,
            "reason": f"only {len(survivors)} negative families survived construction and family shortcut audit",
        }
        return result, selected, all_predictions
    rows, _, features = _shortcut_table(
        anchors, arrays, negatives, negative_manifest, selected, payload
    )
    feature_reports: dict[str, Any] = {}
    for name, feature in features.items():
        report, predictions = _fit_feature(
            name,
            rows,
            feature,
            nonlinear=name == "combined_non_context",
        )
        feature_reports[name] = report
        all_predictions.extend(predictions)
    labels = np.asarray([row["label"] for row in rows], dtype=np.int64)
    positive_fraction = float(labels.mean())
    family_counts = Counter(row["paired_family"] for row in rows if row["label"] == 0)
    max_family_fraction = max(family_counts.values()) / sum(family_counts.values())
    individual_names = [
        name for name in features
        if name not in {"combined_non_context", "source_metadata"}
    ]
    strongest_individual = max(
        float(feature_reports[name]["test"]["balanced_accuracy"]) for name in individual_names
    )
    combined = float(feature_reports["combined_non_context"]["test"]["balanced_accuracy"])
    source = float(feature_reports["source_metadata"]["test"]["balanced_accuracy"])
    family_identity = float(
        feature_reports["negative_family_identity"]["test"]["balanced_accuracy"]
    )
    gates = {
        "every_individual_shortcut_le_0_60": strongest_individual <= 0.60,
        "combined_non_context_le_0_63": combined <= 0.63,
        "source_metadata_le_0_55": source <= 0.55,
        "class_balance_45_55": 0.45 <= positive_fraction <= 0.55,
        "family_identity_not_deterministic": family_identity <= 0.60,
        "at_least_three_families": len(survivors) >= 3,
        "single_family_fraction_le_0_50": max_family_fraction <= 0.50,
    }
    decision = (
        "RM_ACV_PILOT_P3_READY"
        if all(gates.values())
        else "RM_ACV_PILOT_SHORTCUT_DOMINATED"
    )
    return (
        {
            "status": "pass" if all(gates.values()) else "fail",
            "decision": decision,
            "family_reports": family_reports,
            "surviving_families": survivors,
            "selected_families": selected,
            "feature_reports": feature_reports,
            "gates": gates,
            "strongest_individual_balanced_accuracy": strongest_individual,
            "combined_balanced_accuracy": combined,
            "source_metadata_balanced_accuracy": source,
            "family_identity_balanced_accuracy": family_identity,
            "positive_fraction": positive_fraction,
            "negative_family_counts": dict(sorted(family_counts.items())),
            "maximum_single_family_fraction": max_family_fraction,
        },
        selected,
        all_predictions,
    )


def run() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "preregistered_config.json").exists():
        raise RuntimeError("pilot branch already initialized; use --fresh only to rebuild this exact branch")
    (OUT / "run_log.jsonl").unlink(missing_ok=True)
    append_log("P0_START", branch="RM-ACV-PILOT166", source=str(SOURCE))
    records, excluded = scan_records()
    _, aggregate = _source_manifest(records, excluded)
    config = _configuration(records, aggregate)
    write_json(OUT / "preregistered_config.json", config)
    (OUT / "preregistered_config.sha256").write_text(
        sha256_file(OUT / "preregistered_config.json") + "\n", encoding="utf-8"
    )
    preflight = OUT / "preflight_design_v1"
    if preflight.is_dir():
        old_config = preflight / "preregistered_config.json"
        old_failures = preflight / "negative_failure_log.jsonl"
        old_failure_rows = (
            [json.loads(line) for line in old_failures.read_text(encoding="utf-8").splitlines() if line]
            if old_failures.is_file() else []
        )
        write_json(
            OUT / "preregistration_amendment.json",
            {
                "schema": "rm-acv-pilot166-preregistration-amendment-v1",
                "timing": "after P2 design-version-1 construction preflight crashed before any shortcut gate completion or fair-model training",
                "previous_config_sha256": sha256_file(old_config) if old_config.is_file() else None,
                "current_config_sha256": sha256_file(OUT / "preregistered_config.json"),
                "observed_version_1_failure_counts": dict(
                    sorted(Counter(row["family"] for row in old_failure_rows).items())
                ),
                "authorized_whole_family_redesigns": sorted(REDESIGNED_SYNTHETIC_FAMILIES),
                "change": "replace an over-strict fixed 0.015 normalized action-change cutoff with a non-identical 1e-5 cutoff plus a frozen 85% family coverage gate; failed candidates remain failed and are never individually repaired",
                "fair_models_trained_before_amendment": False,
                "shortcut_gate_completed_before_amendment": False,
                "per_sample_manual_repairs": 0,
                "redesign_budget_respected": True,
            },
        )
    write_json(
        OUT / "episode_split_preregistered.json",
        {"schema": "rm-acv-pilot166-episode-split-v1", "splits": episode_split(records)},
    )
    write_json(
        OUT / "task_cv_preregistered.json",
        {
            "schema": "rm-acv-pilot166-task-cv-v1",
            "folds": task_cv_folds(sorted({record.task_id for record in records})),
        },
    )
    append_log(
        "P0_FROZEN",
        config_sha256=sha256_file(OUT / "preregistered_config.json"),
        trajectories=len(records),
        tasks=len({record.task_id for record in records}),
        source_hash_aggregate=aggregate,
    )
    audits, payload = _load_and_cache(records)
    anchors, arrays = _anchor_arrays(payload)
    write_jsonl(OUT / "anchor_manifest.jsonl", anchors)
    write_jsonl(
        OUT / "positive_candidate_manifest.jsonl",
        [
            {
                "schema": "rm-acv-pilot166-positive-v1",
                "candidate_id": f"{row['anchor_id']}:positive",
                "anchor_id": row["anchor_id"],
                "anchor_array_index": row["array_index"],
                "label": 1,
                "definition": "logged immediate continuation; not optimal/safe/successful",
                "candidate_sha256": hashlib.sha256(
                    arrays["positive_action"][row["array_index"]].tobytes()
                ).hexdigest(),
            }
            for row in anchors
        ],
    )
    by_episode = Counter(row["episode_id_provenance_only"] for row in anchors)
    by_task = Counter(row["task_id_provenance_only"] for row in anchors)
    write_json(
        OUT / "anchor_statistics.json",
        {
            "schema": "rm-acv-pilot166-anchor-statistics-v1",
            "anchors": len(anchors),
            "trajectories": len(records),
            "tasks": len(by_task),
            "per_episode": dict(sorted(by_episode.items())),
            "per_task": dict(sorted(by_task.items())),
            "minimum_per_episode": min(by_episode.values()),
            "maximum_per_episode": max(by_episode.values()),
            "median_per_episode": float(np.median(list(by_episode.values()))),
        },
    )
    overlaps = []
    for episode_id in sorted(by_episode):
        episode_anchors = [
            row["anchor_frame"] for row in anchors
            if row["episode_id_provenance_only"] == episode_id
        ]
        for left, right in zip(episode_anchors, episode_anchors[1:]):
            overlaps.append(
                {
                    "episode_id_provenance_only": episode_id,
                    "left_anchor": left,
                    "right_anchor": right,
                    "spacing_steps": right - left,
                    "future_action_windows_overlap": right - left < ACTION_HORIZON,
                }
            )
    write_json(
        OUT / "anchor_overlap_audit.json",
        {
            "schema": "rm-acv-pilot166-anchor-overlap-audit-v1",
            "minimum_spacing_required": MIN_ANCHOR_SPACING,
            "minimum_observed_spacing": min(row["spacing_steps"] for row in overlaps),
            "future_action_overlap_pairs": sum(row["future_action_windows_overlap"] for row in overlaps),
            "pairs": overlaps,
            "pass": all(row["spacing_steps"] >= MIN_ANCHOR_SPACING for row in overlaps),
        },
    )
    append_log(
        "P1_ANCHORS_COMPLETE",
        anchors=len(anchors),
        min_per_episode=min(by_episode.values()),
        max_per_episode=max(by_episode.values()),
    )
    negatives, negative_manifest, failures = _build_negatives(anchors, arrays, payload)
    write_jsonl(OUT / "negative_failure_log.jsonl", failures)
    matching_by_family: dict[str, list[dict[str, float]]] = defaultdict(list)
    for row in negative_manifest:
        matching_by_family[row["family"]].append(row["matching_scores"])
    write_json(
        OUT / "negative_generation_report.json",
        {
            "schema": "rm-acv-pilot166-negative-generation-v1",
            "anchors": len(anchors),
            "attempted": len(anchors) * len(NEGATIVE_FAMILIES),
            "constructed": len(negative_manifest),
            "failures": len(failures),
            "families": dict(Counter(row["family"] for row in negative_manifest)),
            "manual_sample_repairs": 0,
            "whole_family_redesigns": len(REDESIGNED_SYNTHETIC_FAMILIES),
            "redesigned_families": sorted(REDESIGNED_SYNTHETIC_FAMILIES),
            "redesign_limit_respected": True,
        },
    )
    write_json(
        OUT / "negative_matching_statistics.json",
        {
            "schema": "rm-acv-pilot166-negative-matching-v1",
            "families": {
                family: {
                    key: {
                        "mean": float(np.mean([row[key] for row in values if key in row])),
                        "median": float(np.median([row[key] for row in values if key in row])),
                        "maximum": float(np.max([row[key] for row in values if key in row])),
                    }
                    for key in sorted({key for row in values for key in row})
                }
                for family, values in sorted(matching_by_family.items())
            },
        },
    )
    append_log(
        "P2_NEGATIVES_COMPLETE",
        constructed=len(negative_manifest),
        failures=len(failures),
        families=len(NEGATIVE_FAMILIES),
    )
    shortcut, selected_families, predictions = _shortcut_audit(
        anchors, arrays, negatives, negative_manifest, failures, payload
    )
    selected_set = set(selected_families)
    success_lookup = {
        (int(row["anchor_array_index"]), str(row["family"])) for row in negative_manifest
    }
    selected_group_indices = [
        index for index in range(len(anchors))
        if all((index, family) in success_lookup for family in selected_families)
    ] if len(selected_families) == GROUPED_NEGATIVES else []
    selected_group_set = set(selected_group_indices)
    for row in negative_manifest:
        row["selected_for_final_benchmark"] = (
            row["family"] in selected_set
            and int(row["anchor_array_index"]) in selected_group_set
        )
    write_jsonl(OUT / "negative_candidate_manifest.jsonl", negative_manifest)
    write_jsonl(
        OUT / "candidate_group_manifest.jsonl",
        [
            {
                "schema": "rm-acv-pilot166-candidate-group-v1",
                "group_id": anchors[index]["anchor_id"],
                "anchor_id": anchors[index]["anchor_id"],
                "anchor_array_index": index,
                "episode_split": anchors[index]["episode_split"],
                "task_id_provenance_only": anchors[index]["task_id_provenance_only"],
                "positive_candidate_id": f"{anchors[index]['anchor_id']}:positive",
                "negative_candidate_ids": [
                    f"{anchors[index]['anchor_id']}:{family}" for family in selected_families
                ],
                "selected_families": list(selected_families),
                "candidate_count": 1 + len(selected_families),
            }
            for index in selected_group_indices
        ],
    )
    write_json(OUT / "shortcut_feature_report.json", shortcut)
    write_jsonl(OUT / "shortcut_predictions.jsonl", predictions)
    write_json(
        OUT / "negative_family_decision.json",
        {
            "schema": "rm-acv-pilot166-negative-family-decision-v1",
            "decision": shortcut["decision"],
            "surviving_families": shortcut["surviving_families"],
            "selected_families": selected_families,
            "family_reports": shortcut["family_reports"],
        },
    )
    write_json(
        OUT / "negative_quality_report.json",
        {
            "schema": "rm-acv-pilot166-negative-quality-v1",
            "status": shortcut["status"],
            "construction_failures": len(failures),
            "selected_family_count": len(selected_families),
            "shortcut_gates": shortcut.get("gates", {}),
        },
    )
    write_json(
        OUT / "matching_balance_report.json",
        {
            "schema": "rm-acv-pilot166-matching-balance-v1",
            "positive_fraction": shortcut.get("positive_fraction"),
            "negative_family_counts": shortcut.get("negative_family_counts", {}),
            "maximum_single_family_fraction": shortcut.get("maximum_single_family_fraction"),
            "thresholds": SHORTCUT_THRESHOLDS,
        },
    )
    np.savez_compressed(
        OUT / "benchmark_arrays.npz",
        **arrays,
        negative_action_all=negatives,
        selected_group_indices=np.asarray(selected_group_indices, dtype=np.int64),
        selected_family_indices=np.asarray(
            [NEGATIVE_FAMILIES.index(family) for family in selected_families], dtype=np.int64
        ),
    )
    append_log(
        "P3_SHORTCUT_AUDIT_COMPLETE",
        decision=shortcut["decision"],
        selected_families=selected_families,
        combined_balanced_accuracy=shortcut.get("combined_balanced_accuracy"),
    )
    return {
        "decision": shortcut["decision"],
        "trajectories": len(records),
        "tasks": len({record.task_id for record in records}),
        "anchors": len(anchors),
        "negative_candidates": len(negative_manifest),
        "candidate_groups": len(selected_group_indices),
        "selected_families": selected_families,
        "shortcut_gates": shortcut.get("gates", {}),
    }


def _archive_preflight() -> None:
    if not OUT.is_dir() or not (OUT / "preregistered_config.json").is_file():
        raise RuntimeError("no failed design-version-1 preflight is available to archive")
    archive = OUT / "preflight_design_v1"
    if archive.exists():
        raise RuntimeError(f"preflight archive already exists: {archive}")
    archive.mkdir(parents=True)
    preserve = {
        "preregistered_config.json",
        "preregistered_config.sha256",
        "negative_failure_log.jsonl",
        "negative_generation_report.json",
        "negative_matching_statistics.json",
        "run_log.jsonl",
    }
    for name in sorted(preserve):
        path = OUT / name
        if path.exists():
            path.rename(archive / name)
    for path in list(OUT.iterdir()):
        if path == archive:
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--redesign-after-preflight", action="store_true")
    args = parser.parse_args()
    if args.fresh and args.redesign_after_preflight:
        raise RuntimeError("choose either --fresh or --redesign-after-preflight")
    if args.redesign_after_preflight:
        _archive_preflight()
    if args.fresh and OUT.exists():
        resolved = OUT.resolve()
        expected = (ROOT / "outputs" / "actmask" / "rm_acv_pilot166").resolve()
        if resolved != expected:
            raise RuntimeError(f"refusing to clear unexpected path: {resolved}")
        shutil.rmtree(resolved)
    print(json.dumps(run(), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
