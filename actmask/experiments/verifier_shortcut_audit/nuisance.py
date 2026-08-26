"""V4 audit-only nuisance feature extraction."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .common import (
    HISTORICAL,
    OUT,
    append_log,
    read_jsonl,
    sha256_file,
    stable_int,
    write_json,
)


PILOT = HISTORICAL / "rm_acv_pilot166"


def _append(
    values: list[np.ndarray],
    names: list[str],
    value: Any,
    prefix: str,
) -> None:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    values.append(array)
    names.extend(
        [prefix] if len(array) == 1 else [f"{prefix}[{index}]" for index in range(len(array))]
    )


def action_features(action: np.ndarray) -> tuple[np.ndarray, list[str]]:
    action = np.asarray(action, dtype=np.float32)
    velocity = np.diff(action, axis=0)
    acceleration = np.diff(velocity, axis=0)
    jerk = np.diff(acceleration, axis=0)
    values: list[np.ndarray] = []
    names: list[str] = []
    _append(values, names, action.mean(0), "action_mean")
    _append(values, names, action.var(0), "action_variance")
    _append(values, names, np.mean(np.abs(action)), "absolute_magnitude_mean")
    _append(
        values,
        names,
        np.sqrt(np.mean(np.square(action))),
        "absolute_magnitude_rms",
    )
    _append(values, names, np.mean(np.abs(velocity), axis=0), "velocity_abs_mean")
    _append(
        values,
        names,
        np.sqrt(np.mean(np.square(velocity), axis=0)),
        "velocity_rms",
    )
    _append(
        values, names, np.mean(np.abs(acceleration), axis=0), "acceleration_abs_mean"
    )
    _append(values, names, np.mean(np.abs(jerk), axis=0), "jerk_abs_mean")
    _append(values, names, np.mean(np.square(acceleration)), "smoothness_energy")
    displacement = action[-1] - action[0]
    _append(values, names, displacement, "total_displacement")
    _append(values, names, action[0], "start_action")
    _append(values, names, action[-1], "end_action")
    _append(values, names, np.abs(displacement), "endpoint_difference_abs")
    ranges = np.ptp(action, axis=0)
    _append(values, names, np.sum(ranges > 1e-6), "active_joint_count")
    gripper = action[:, 16:18]
    gripper_delta = np.abs(np.diff(gripper, axis=0))
    events = gripper_delta > 1e-5
    _append(values, names, events.sum(), "arm_gripper_event_count")
    _append(values, names, events.sum(0), "gripper_event_count_per_dimension")
    _append(values, names, np.mean(np.square(action[:, :8])), "left_arm_energy")
    _append(values, names, np.mean(np.square(action[:, 8:16])), "right_arm_energy")
    first_events = []
    for dimension in range(2):
        indices = np.flatnonzero(events[:, dimension])
        first_events.append(float(indices[0] + 1) / len(action) if len(indices) else 1.0)
    _append(values, names, first_events, "gripper_first_event_time")
    _append(
        values,
        names,
        np.mean(np.all(np.abs(action) <= 1e-12, axis=1)),
        "zero_padding_step_fraction",
    )
    _append(values, names, np.mean(np.abs(action) <= 1e-12), "zero_value_fraction")
    _append(values, names, len(action), "action_horizon")
    return np.concatenate(values).astype(np.float32), names


def _frame_statistics(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    small = cv2.resize(rgb, (64, 64), interpolation=cv2.INTER_AREA).astype(
        np.float32
    ) / 255.0
    height, width = small.shape[:2]
    border_width = 8
    border_mask = np.zeros((height, width), dtype=bool)
    border_mask[:border_width] = True
    border_mask[-border_width:] = True
    border_mask[:, :border_width] = True
    border_mask[:, -border_width:] = True
    border = small[border_mask]
    center = small[16:48, 16:48].reshape(-1, 3)
    histograms = []
    for channel in range(3):
        hist, _ = np.histogram(
            small[:, :, channel], bins=8, range=(0.0, 1.0), density=False
        )
        histograms.extend((hist / max(1, hist.sum())).astype(np.float32).tolist())
    feature = np.concatenate(
        (
            small.mean((0, 1)),
            small.std((0, 1)),
            np.asarray(histograms, dtype=np.float32),
            border.mean(0),
            border.std(0),
            center.mean(0),
            center.std(0),
        )
    ).astype(np.float32)
    return feature, small


def _raw_visual_features(
    anchors: list[dict[str, Any]],
    source_rows: list[dict[str, Any]],
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    by_safe = {row["safe_episode_key"]: row for row in source_rows}
    grouped: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for anchor_index, anchor in enumerate(anchors):
        grouped[anchor["safe_episode_key"]].append((anchor_index, anchor))
    width = 42 + 4
    result = np.zeros((len(anchors), width), dtype=np.float32)
    decode_failures = []
    decoded_frames = 0
    for episode_count, (safe_key, entries) in enumerate(sorted(grouped.items()), start=1):
        source = by_safe[safe_key]
        video_path = Path(source["front_video_path_provenance_only"])
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            decode_failures.append({"safe_episode_key": safe_key, "path": str(video_path)})
            continue
        required = sorted(
            {
                frame
                for _, anchor in entries
                for frame in range(
                    int(anchor["history_start_frame"]), int(anchor["anchor_frame"]) + 1
                )
            }
        )
        frames: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        required_set = set(required)
        maximum = max(required)
        frame_index = 0
        while frame_index <= maximum:
            ok, bgr = capture.read()
            if not ok:
                break
            if frame_index in required_set:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                frames[frame_index] = _frame_statistics(rgb)
                decoded_frames += 1
            frame_index += 1
        capture.release()
        for anchor_index, anchor in entries:
            indices = list(
                range(
                    int(anchor["history_start_frame"]), int(anchor["anchor_frame"]) + 1
                )
            )
            if any(index not in frames for index in indices):
                decode_failures.append(
                    {
                        "anchor_id": anchor["anchor_id"],
                        "safe_episode_key": safe_key,
                        "missing_frames": [index for index in indices if index not in frames],
                    }
                )
                continue
            current_feature = frames[indices[-1]][0]
            images = [frames[index][1] for index in indices]
            differences = [
                np.abs(images[index] - images[index - 1])
                for index in range(1, len(images))
            ]
            if differences:
                difference = np.stack(differences)
                difference_feature = np.asarray(
                    [
                        difference.mean(),
                        *difference.mean((0, 1, 2)).tolist(),
                    ],
                    dtype=np.float32,
                )
            else:
                difference_feature = np.zeros(4, dtype=np.float32)
            result[anchor_index] = np.concatenate((current_feature, difference_feature))
        if episode_count % 25 == 0 or episode_count == len(grouped):
            print(
                f"VSA_VISUAL_NUISANCE={episode_count}/{len(grouped)} "
                f"failures={len(decode_failures)}",
                flush=True,
            )
    names = (
        [f"current_rgb_mean[{index}]" for index in range(3)]
        + [f"current_rgb_std[{index}]" for index in range(3)]
        + [f"current_rgb_histogram[{index}]" for index in range(24)]
        + [f"static_border_mean[{index}]" for index in range(3)]
        + [f"static_border_std[{index}]" for index in range(3)]
        + [f"center_workspace_mean[{index}]" for index in range(3)]
        + [f"center_workspace_std[{index}]" for index in range(3)]
        + ["history_frame_difference_mean"]
        + [f"history_frame_difference_channel[{index}]" for index in range(3)]
    )
    return result, names, {
        "episodes": len(grouped),
        "decoded_unique_frames": decoded_frames,
        "failures": decode_failures,
        "pass": not decode_failures,
        "future_frames_read": False,
    }


def _one_hot(values: list[str], categories: list[str]) -> np.ndarray:
    lookup = {value: index for index, value in enumerate(categories)}
    result = np.zeros((len(values), len(categories)), dtype=np.float32)
    for row, value in enumerate(values):
        result[row, lookup[value]] = 1.0
    return result


def _statistics(
    value: np.ndarray, names: list[str], labels: np.ndarray
) -> dict[str, Any]:
    mean = value.mean(0)
    std = value.std(0)
    positive = value[labels == 1].mean(0)
    negative = value[labels == 0].mean(0)
    separation = np.abs(positive - negative) / np.maximum(std, 1e-8)
    strongest = np.argsort(-separation)[:20]
    return {
        "rows": len(value),
        "features": value.shape[1],
        "finite": bool(np.isfinite(value).all()),
        "mean_sha256": __import__("hashlib").sha256(
            mean.astype(np.float32).tobytes()
        ).hexdigest(),
        "std_sha256": __import__("hashlib").sha256(
            std.astype(np.float32).tobytes()
        ).hexdigest(),
        "strongest_label_standardized_mean_differences": [
            {
                "feature": names[index],
                "absolute_standardized_mean_difference": float(separation[index]),
            }
            for index in strongest
        ],
    }


def extract() -> dict[str, Any]:
    manifest = read_jsonl(OUT / "unified_audit_manifest.jsonl")
    with np.load(OUT / "unified_audit_arrays.npz") as stored:
        arrays = {name: stored[name].copy() for name in stored.files}
    anchors = read_jsonl(PILOT / "anchor_manifest.jsonl")
    anchors.sort(key=lambda row: int(row["array_index"]))
    source_rows = read_jsonl(PILOT / "source_manifest.jsonl")
    context_index = arrays["context_anchor_index"].astype(np.int64)
    source_index = arrays["source_anchor_index"].astype(np.int64)
    labels = arrays["label"].astype(np.int64)
    action_rows = []
    action_names: list[str] | None = None
    for action in arrays["candidate_action"]:
        feature, names = action_features(action)
        action_rows.append(feature)
        action_names = names
    action = np.stack(action_rows)
    assert action_names is not None

    raw_visual_unique, raw_visual_names, visual_audit = _raw_visual_features(
        anchors, source_rows
    )
    tasks = sorted({row["task"] for row in manifest})
    task_values = [row["task"] for row in manifest]
    task_one_hot = _one_hot(task_values, tasks)
    history = arrays["history_state"].astype(np.float32)
    current = arrays["current_state"].astype(np.float32)
    state_velocity = history[:, -1] - history[:, -2]
    gripper_state = current[:, 16:18]
    visual = arrays["visual_history"].astype(np.float32)
    cached_current = visual[:, -1]
    cached_difference = np.mean(np.abs(np.diff(visual, axis=1)), axis=1)
    context_parts = [
        arrays["normalized_progress"][:, None].astype(np.float32),
        arrays["episode_duration"][:, None].astype(np.float32),
        task_one_hot,
        current,
        state_velocity,
        gripper_state,
        cached_current,
        cached_difference,
        raw_visual_unique[context_index],
    ]
    context_names = (
        ["normalized_episode_progress", "episode_length"]
        + [f"task_identity[{task}]" for task in tasks]
        + [f"current_state[{index}]" for index in range(32)]
        + [f"recent_state_velocity[{index}]" for index in range(32)]
        + [f"current_gripper_state[{index}]" for index in range(2)]
        + [f"current_frame_frozen_visual[{index}]" for index in range(22)]
        + [f"history_visual_difference[{index}]" for index in range(22)]
        + raw_visual_names
    )
    context = np.concatenate(context_parts, axis=1).astype(np.float32)

    families = sorted({row["family"] for row in manifest})
    family_values = [row["family"] for row in manifest]
    episodes = sorted(
        {
            row["episode"] for row in manifest
        }
        | {row["source_episode"] for row in manifest}
    )
    source_episode_one_hot = _one_hot(
        [row["source_episode"] for row in manifest], episodes
    )
    context_episode_one_hot = _one_hot(
        [row["episode"] for row in manifest], episodes
    )
    path_buckets = np.zeros((len(manifest), 32), dtype=np.float32)
    for index, row in enumerate(manifest):
        path_buckets[index, stable_int(row["source_episode"], 32)] = 1.0
    manifest_order = np.asarray(
        [row["provenance"]["manifest_order"] for row in manifest], dtype=np.float32
    )[:, None]
    manifest_order /= max(1.0, float(len(manifest) - 1))
    pair_rank = np.asarray(
        [
            row["provenance"]["pair_mining_rank"]
            if row["provenance"]["pair_mining_rank"] is not None
            else 0
            for row in manifest
        ],
        dtype=np.float32,
    )[:, None]
    pair_present = (pair_rank > 0).astype(np.float32)
    source_length = np.asarray(
        [row["source_trajectory_length"] for row in manifest], dtype=np.float32
    )[:, None]
    provenance = np.concatenate(
        (
            _one_hot(family_values, families),
            context_episode_one_hot,
            source_episode_one_hot,
            source_length,
            path_buckets,
            manifest_order,
            pair_rank,
            pair_present,
            np.ones((len(manifest), 1), dtype=np.float32),  # logged policy
            np.ones((len(manifest), 1), dtype=np.float32),  # camera_front
            np.zeros((len(manifest), 1), dtype=np.float32),  # scene unavailable
        ),
        axis=1,
    )
    provenance_names = (
        [f"family_identity[{family}]" for family in families]
        + [f"context_episode[{episode}]" for episode in episodes]
        + [f"source_episode[{episode}]" for episode in episodes]
        + ["source_trajectory_length"]
        + [f"source_path_hash_bucket[{index}]" for index in range(32)]
        + [
            "manifest_order",
            "pair_mining_rank",
            "pair_rank_present",
            "source_policy_logged_demonstration",
            "camera_front",
            "scene_identifier_unavailable",
        ]
    )

    source_by_safe = {row["safe_episode_key"]: row for row in source_rows}
    instruction_lengths_unique = []
    for anchor in anchors:
        text = source_by_safe[anchor["safe_episode_key"]]["language_instruction"]
        instruction_lengths_unique.append(
            [len(text), len(text.split())]
        )
    instruction_lengths = np.asarray(
        instruction_lengths_unique, dtype=np.float32
    )[context_index]
    language_embedding = arrays["language_embedding"].astype(np.float32)
    task_id_embedding = np.zeros_like(language_embedding)
    for row, task in enumerate(task_values):
        task_id_embedding[row, stable_int(f"task-token:{task}", 128)] = 1.0
    task_to_embedding = {}
    for task in tasks:
        index = next(i for i, row in enumerate(manifest) if row["task"] == task)
        task_to_embedding[task] = language_embedding[index]
    shuffled = np.stack(
        [
            task_to_embedding[tasks[(tasks.index(task) + 1) % len(tasks)]]
            for task in task_values
        ]
    ).astype(np.float32)
    empty = np.zeros_like(language_embedding)
    language = np.concatenate(
        (
            task_one_hot,
            language_embedding,
            instruction_lengths,
            shuffled,
            empty,
            task_id_embedding,
        ),
        axis=1,
    )
    language_names = (
        [f"task_string_identity[{task}]" for task in tasks]
        + [f"language_embedding[{index}]" for index in range(128)]
        + ["instruction_character_length", "instruction_token_length"]
        + [f"shuffled_language_embedding[{index}]" for index in range(128)]
        + [f"empty_language_embedding[{index}]" for index in range(128)]
        + [f"task_id_only_embedding[{index}]" for index in range(128)]
    )
    combined = np.concatenate((action, context, provenance, language), axis=1)
    combined_names = (
        [f"action::{name}" for name in action_names]
        + [f"context::{name}" for name in context_names]
        + [f"provenance::{name}" for name in provenance_names]
        + [f"language::{name}" for name in language_names]
    )
    np.savez_compressed(
        OUT / "nuisance_features.npz",
        action=action,
        context=context,
        provenance=provenance,
        language=language,
        combined=combined,
        raw_visual_unique=raw_visual_unique,
        label=labels.astype(np.int8),
    )
    schema = {
        "schema": "vsa-nuisance-feature-schema-v1",
        "audit_only": True,
        "fair_verifier_input_forbidden": True,
        "groups": {
            "action": action_names,
            "context": context_names,
            "provenance": provenance_names,
            "language": language_names,
            "combined": combined_names,
        },
        "visual_definitions": {
            "cached_22D": "RGB mean/std plus 4x4 luminance map per causal frame",
            "global_color_histogram": "8 bins per RGB channel on current causal frame",
            "static_border": "8-pixel border of a 64x64 current causal frame",
            "center_workspace": "central 32x32 crop of current causal frame",
            "history_difference": "absolute difference of consecutive causal frames",
        },
        "action_dimension_convention": {
            "left_arm": "0:8",
            "right_arm": "8:16",
            "gripper": "16:18",
            "end_effector": "18:32",
        },
    }
    write_json(OUT / "nuisance_feature_schema.json", schema)
    statistics = {
        "schema": "vsa-nuisance-feature-statistics-v1",
        "action": _statistics(action, action_names, labels),
        "context": _statistics(context, context_names, labels),
        "provenance": _statistics(provenance, provenance_names, labels),
        "language": _statistics(language, language_names, labels),
        "combined": _statistics(combined, combined_names, labels),
    }
    write_json(OUT / "nuisance_feature_statistics.json", statistics)
    audit = {
        "schema": "vsa-nuisance-extraction-audit-v1",
        "samples": len(manifest),
        "arrays_sha256": sha256_file(OUT / "nuisance_features.npz"),
        "schema_sha256": sha256_file(OUT / "nuisance_feature_schema.json"),
        "all_finite": bool(np.isfinite(combined).all()),
        "labels_in_feature_matrix": False,
        "features_audit_only": True,
        "future_state_used": False,
        "future_rgb_used": False,
        "SAM3_used": False,
        "new_robot_masks_generated": False,
        "raw_visual_decode": visual_audit,
        "pass": bool(np.isfinite(combined).all() and visual_audit["pass"]),
    }
    write_json(OUT / "nuisance_extraction_audit.json", audit)
    append_log(
        "V4_NUISANCE_FEATURES_COMPLETE",
        samples=len(manifest),
        action_features=action.shape[1],
        context_features=context.shape[1],
        provenance_features=provenance.shape[1],
        language_features=language.shape[1],
        combined_features=combined.shape[1],
        visual_decode_failures=len(visual_audit["failures"]),
        pass_=audit["pass"],
    )
    return {
        "pass": audit["pass"],
        "samples": len(manifest),
        "feature_dimensions": {
            "action": action.shape[1],
            "context": context.shape[1],
            "provenance": provenance.shape[1],
            "language": language.shape[1],
            "combined": combined.shape[1],
        },
        "visual_decode": visual_audit,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.parse_args()
    print(json.dumps(extract(), indent=2, sort_keys=True))
