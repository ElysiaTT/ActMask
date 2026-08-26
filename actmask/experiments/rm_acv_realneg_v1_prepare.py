"""RM-ACV-REALNEG-V1 phases R0--R3.

The study imports the frozen 166-trajectory / 992-anchor pilot source without
regeneration, extracts the already established causal 22-D RGB statistics,
mines symmetric real-action anchor pairs, creates reciprocal context swaps,
and proves exact context/action reuse plus split containment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from actmask.experiments.rm_acv_pilot166_common import (
    ACTION_HORIZON,
    FAIR_INPUTS,
    FORBIDDEN_PREDICTIVE_INPUTS,
    HISTORY,
    PRIMARY_EMBODIMENT,
    ROOT,
    SEEDS,
    SOURCE,
    environment_manifest,
    read_jsonl,
    scan_records,
    sha256_file,
    sha256_json,
    stable_int,
    write_json,
    write_jsonl,
)
from actmask.experiments.rm_acv_pilot166_finalize import verify as verify_pilot


OUT = ROOT / "outputs" / "actmask" / "rm_acv_realneg_v1"
PILOT = ROOT / "outputs" / "actmask" / "rm_acv_pilot166"
VISUAL_WORKER = ROOT / "actmask" / "experiments" / "rm_series_video_feature_worker.py"
FAMILIES = ("N7_reciprocal_near_state_context_swap", "N8_visual_substate_reciprocal_context_swap")
PAIR_PROGRESS_BROAD_MAX = 0.20
N8_PAIRS_PER_TASK = 20


def append_log(event: str, **payload: Any) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "run_log.jsonl"
    previous = ""
    seq = 0
    if path.is_file() and path.stat().st_size:
        last = json.loads(path.read_text(encoding="utf-8").splitlines()[-1])
        previous = str(last["event_sha256"])
        seq = int(last["seq"])
    row = {
        "seq": seq + 1,
        "utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "event": event,
        "payload": payload,
        "previous_event_sha256": previous,
    }
    row["event_sha256"] = sha256_json(row)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _import_frozen_pilot(*, rehash_source: bool) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    verification = verify_pilot(rehash_source=rehash_source)
    if not verification["passed"]:
        raise RuntimeError(f"frozen pilot verification failed: {verification}")
    prior_repro = json.loads((PILOT / "reproducibility_manifest.json").read_text(encoding="utf-8"))
    expected = {row["path"]: row["sha256"] for row in prior_repro["generated_files"]}
    imports = (
        "source_manifest.jsonl",
        "source_hash_manifest.json",
        "anchor_manifest.jsonl",
        "positive_candidate_manifest.jsonl",
        "anchor_statistics.json",
        "anchor_overlap_audit.json",
        "alignment_audit.json",
        "episode_split_preregistered.json",
        "task_cv_preregistered.json",
        "benchmark_arrays.npz",
    )
    checks = []
    for name in imports:
        source = PILOT / name
        actual = sha256_file(source)
        if expected.get(name) != actual:
            raise RuntimeError(f"imported pilot artifact hash mismatch: {name}")
        destination = OUT / f"imported_{name}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        if sha256_file(destination) != actual:
            raise RuntimeError(f"copied import hash mismatch: {name}")
        checks.append(
            {
                "pilot_path": str(source.relative_to(ROOT)),
                "imported_path": str(destination.relative_to(OUT)),
                "sha256": actual,
                "match": True,
            }
        )
    anchors = read_jsonl(OUT / "imported_anchor_manifest.jsonl")
    if len(anchors) != 992 or sorted(row["array_index"] for row in anchors) != list(range(992)):
        raise RuntimeError("frozen anchor manifest is not the exact 992-anchor source")
    anchors.sort(key=lambda row: int(row["array_index"]))
    with np.load(OUT / "imported_benchmark_arrays.npz") as source_arrays:
        arrays = {
            key: source_arrays[key].copy()
            for key in (
                "history_state",
                "positive_action",
                "current_state",
                "progress",
                "instruction",
                "episode_duration",
                "episode_index_internal",
                "task_index_internal",
            )
        }
    if arrays["history_state"].shape != (992, HISTORY, 32):
        raise RuntimeError(f"unexpected imported history shape: {arrays['history_state'].shape}")
    if arrays["positive_action"].shape != (992, ACTION_HORIZON, 32):
        raise RuntimeError(f"unexpected imported action shape: {arrays['positive_action'].shape}")
    np.savez_compressed(OUT / "imported_anchor_arrays.npz", **arrays)
    source_manifest = read_jsonl(OUT / "imported_source_manifest.jsonl")
    write_json(
        OUT / "import_verification_audit.json",
        {
            "schema": "rm-acv-realneg-import-audit-v1",
            "pilot_decision": verification["decision"],
            "pilot_verifier_pass": verification["passed"],
            "pilot_source_content_rehashed": rehash_source,
            "pilot_source_mismatches": verification["source_verification"]["mismatches"],
            "source_trajectories": len(source_manifest),
            "anchors": len(anchors),
            "tasks": len({row["task_id_provenance_only"] for row in source_manifest}),
            "embodiment": PRIMARY_EMBODIMENT,
            "imports": checks,
            "anchors_regenerated": False,
            "pass": True,
        },
    )
    return anchors, arrays


def _extract_visual_features(
    anchors: list[dict[str, Any]],
) -> tuple[np.ndarray, dict[str, Any]]:
    source = {
        row["episode_id_provenance_only"]: row
        for row in read_jsonl(OUT / "imported_source_manifest.jsonl")
    }
    by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in anchors:
        by_episode[row["episode_id_provenance_only"]].append(row)
    scratch = OUT / "visual_features" / "per_episode"
    scratch.mkdir(parents=True, exist_ok=True)
    requests = []
    for episode_id, rows in sorted(by_episode.items()):
        ordered = sorted(rows, key=lambda row: row["anchor_frame"])
        history_steps = [
            step
            for row in ordered
            for step in range(row["history_start_frame"], row["anchor_frame"] + 1)
        ]
        requests.append(
            {
                "episode_id": episode_id,
                "video_path": source[episode_id]["front_video_path_provenance_only"],
                "history_frame_steps": history_steps,
                "anchors": len(ordered),
                "history": HISTORY,
            }
        )
    worker_manifests = OUT / "visual_features" / "worker_manifests"
    worker_manifests.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(requests), 8):
        batch = requests[start : start + 8]
        manifest = worker_manifests / f"batch_{start // 8:03d}.json"
        write_json(manifest, batch)
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "actmask.experiments.rm_series_video_feature_worker",
                "--manifest",
                str(manifest),
                "--output",
                str(scratch),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        if completed.returncode:
            raise RuntimeError(
                f"visual feature worker failed at batch {start // 8}: "
                f"{completed.stderr.strip() or completed.stdout.strip()}"
            )
        print(
            f"RM_ACV_REALNEG_VISUAL={min(start + len(batch), len(requests))}/{len(requests)}",
            flush=True,
        )
    combined = np.empty((len(anchors), HISTORY, 22), dtype=np.float32)
    for episode_id, rows in sorted(by_episode.items()):
        ordered = sorted(rows, key=lambda row: row["anchor_frame"])
        path = scratch / f"{episode_id.replace(':', '_')}.npz"
        values = np.load(path)["value"].astype(np.float32)
        if values.shape != (len(ordered), HISTORY, 22):
            raise RuntimeError(f"visual feature shape mismatch for {episode_id}: {values.shape}")
        for ordinal, row in enumerate(ordered):
            combined[int(row["array_index"])] = values[ordinal]
    np.savez_compressed(OUT / "causal_visual_features.npz", visual=combined)
    config = {
        "schema": "rm-acv-realneg-visual-encoder-v1",
        "encoder": "frozen deterministic RGB statistics used by prior RM experiments",
        "source_module": str(VISUAL_WORKER.relative_to(ROOT)),
        "source_module_sha256": sha256_file(VISUAL_WORKER),
        "camera": "observation.rgb_images.camera_front",
        "causal_history_steps": HISTORY,
        "feature_per_frame": "RGB mean/std plus 4x4 luminance map",
        "feature_dimension": 22,
        "external_download": False,
        "selected_using_benchmark_performance": False,
        "future_rgb_used": False,
        "feature_sha256": hashlib.sha256(combined.tobytes()).hexdigest(),
    }
    write_json(OUT / "visual_encoder_config.json", config)
    return combined, config


def _training_normalization(
    anchors: list[dict[str, Any]],
    arrays: dict[str, np.ndarray],
    visual: np.ndarray,
) -> dict[str, np.ndarray]:
    train = np.asarray(
        [int(row["array_index"]) for row in anchors if row["episode_split"] == "train"],
        dtype=np.int64,
    )
    history = arrays["history_state"]
    action = arrays["positive_action"]
    velocity = history[:, -1] - history[:, -2]
    visual_mean = visual.mean(1)
    values = {
        "state": history[train, -1],
        "history": history[train].reshape(-1, 32),
        "velocity": velocity[train],
        "action": action[train].reshape(-1, 32),
        "visual": visual_mean[train],
    }
    normalization: dict[str, np.ndarray] = {}
    manifest: dict[str, Any] = {
        "schema": "rm-acv-realneg-training-normalization-v1",
        "training_anchor_count": len(train),
        "training_episode_split_only": True,
        "test_statistics_used": False,
        "entries": {},
    }
    for name, value in values.items():
        mean = value.mean(0).astype(np.float32)
        scale = value.std(0).astype(np.float32)
        scale[scale < 1e-4] = 1.0
        normalization[f"{name}_mean"] = mean
        normalization[f"{name}_scale"] = scale
        manifest["entries"][name] = {
            "width": len(mean),
            "mean_sha256": hashlib.sha256(mean.tobytes()).hexdigest(),
            "scale_sha256": hashlib.sha256(scale.tobytes()).hexdigest(),
        }
    np.savez_compressed(OUT / "training_normalization.npz", **normalization)
    write_json(OUT / "training_normalization.json", manifest)
    return normalization


def _scalar_action_features(action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    magnitude = np.linalg.norm(action, axis=2).mean(1)
    smoothness = np.linalg.norm(np.diff(action, axis=1), axis=2).mean(1)
    return magnitude.astype(np.float32), smoothness.astype(np.float32)


def _pair_candidates(
    anchors: list[dict[str, Any]],
    arrays: dict[str, np.ndarray],
    visual: np.ndarray,
    normalization: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    history = arrays["history_state"]
    current = history[:, -1]
    velocity = history[:, -1] - history[:, -2]
    action = arrays["positive_action"]
    progress = arrays["progress"]
    visual_mean = visual.mean(1)
    magnitude, smoothness = _scalar_action_features(action)
    train_indices = np.asarray(
        [row["array_index"] for row in anchors if row["episode_split"] == "train"],
        dtype=np.int64,
    )
    magnitude_scale = max(float(np.std(magnitude[train_indices])), 1e-4)
    smoothness_scale = max(float(np.std(smoothness[train_indices])), 1e-4)
    by_task_split: dict[tuple[str, str], list[int]] = defaultdict(list)
    for row in anchors:
        by_task_split[(row["task_id_provenance_only"], row["episode_split"])].append(
            int(row["array_index"])
        )
    rows: list[dict[str, Any]] = []
    for (task, split), indices in sorted(by_task_split.items()):
        for left_position, left in enumerate(indices):
            for right in indices[left_position + 1 :]:
                if (
                    anchors[left]["episode_id_provenance_only"]
                    == anchors[right]["episode_id_provenance_only"]
                ):
                    continue
                progress_distance = abs(float(progress[left] - progress[right]))
                if progress_distance > PAIR_PROGRESS_BROAD_MAX:
                    continue
                state_distance = float(
                    np.sqrt(
                        np.mean(
                            (
                                (current[left] - current[right])
                                / normalization["state_scale"]
                            )
                            ** 2
                        )
                    )
                )
                history_distance = float(
                    np.sqrt(
                        np.mean(
                            (
                                (history[left] - history[right])
                                / normalization["history_scale"]
                            )
                            ** 2
                        )
                    )
                )
                velocity_distance = float(
                    np.sqrt(
                        np.mean(
                            (
                                (velocity[left] - velocity[right])
                                / normalization["velocity_scale"]
                            )
                            ** 2
                        )
                    )
                )
                gripper_distance = float(
                    np.sqrt(
                        np.mean(
                            (
                                (current[left, 16:18] - current[right, 16:18])
                                / normalization["state_scale"][16:18]
                            )
                            ** 2
                        )
                    )
                )
                action_distance = float(
                    np.sqrt(
                        np.mean(
                            (
                                (action[left] - action[right])
                                / normalization["action_scale"]
                            )
                            ** 2
                        )
                    )
                )
                endpoint_distance = float(
                    np.sqrt(
                        np.mean(
                            (
                                (action[left, -1] - action[right, -1])
                                / normalization["action_scale"]
                            )
                            ** 2
                        )
                    )
                )
                magnitude_distance = abs(float(magnitude[left] - magnitude[right])) / magnitude_scale
                smoothness_distance = (
                    abs(float(smoothness[left] - smoothness[right])) / smoothness_scale
                )
                visual_distance = float(
                    np.sqrt(
                        np.mean(
                            (
                                (visual_mean[left] - visual_mean[right])
                                / normalization["visual_scale"]
                            )
                            ** 2
                        )
                    )
                )
                left_episode = anchors[left]["episode_id_provenance_only"]
                right_episode = anchors[right]["episode_id_provenance_only"]
                episode_pair = "::".join(sorted((left_episode, right_episode)))
                rows.append(
                    {
                        "schema": "rm-acv-realneg-pair-candidate-v1",
                        "candidate_edge_id": f"e{len(rows):07d}",
                        "task_id_provenance_only": task,
                        "episode_split": split,
                        "left_anchor_id": anchors[left]["anchor_id"],
                        "right_anchor_id": anchors[right]["anchor_id"],
                        "left_anchor_index": left,
                        "right_anchor_index": right,
                        "left_episode_id_provenance_only": left_episode,
                        "right_episode_id_provenance_only": right_episode,
                        "episode_pair_id_provenance_only": episode_pair,
                        "distances": {
                            "current_state": state_distance,
                            "recent_state_history": history_distance,
                            "recent_state_velocity": velocity_distance,
                            "normalized_progress": progress_distance,
                            "gripper_state": gripper_distance,
                            "logged_action_sequence": action_distance,
                            "action_endpoint": endpoint_distance,
                            "action_magnitude": magnitude_distance,
                            "action_smoothness": smoothness_distance,
                            "causal_visual_feature": visual_distance,
                        },
                    }
                )
    return rows


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        raise RuntimeError("empty training distance distribution")
    return float(np.quantile(np.asarray(values, dtype=np.float64), q))


def _derive_thresholds(
    candidates: list[dict[str, Any]],
    anchors: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    train = [row for row in candidates if row["episode_split"] == "train"]
    distance = lambda name: [float(row["distances"][name]) for row in train]
    designs = {
        "N7_v1": {
            "progress_max": 0.12,
            "current_state_max": _quantile(distance("current_state"), 0.65),
            "history_max": _quantile(distance("recent_state_history"), 0.65),
            "velocity_max": _quantile(distance("recent_state_velocity"), 0.70),
            "gripper_max": _quantile(distance("gripper_state"), 0.70),
            "action_sequence_min": _quantile(distance("logged_action_sequence"), 0.35),
        },
        "N7_v2": {
            "progress_max": 0.15,
            "current_state_max": _quantile(distance("current_state"), 0.80),
            "history_max": _quantile(distance("recent_state_history"), 0.80),
            "velocity_max": _quantile(distance("recent_state_velocity"), 0.85),
            "gripper_max": _quantile(distance("gripper_state"), 0.85),
            "action_sequence_min": _quantile(distance("logged_action_sequence"), 0.20),
        },
        "N8_v1": {
            "progress_max": 0.12,
            "current_state_max": _quantile(distance("current_state"), 0.55),
            "history_max": _quantile(distance("recent_state_history"), 0.55),
            "velocity_max": _quantile(distance("recent_state_velocity"), 0.75),
            "visual_min": _quantile(distance("causal_visual_feature"), 0.70),
            "action_sequence_min": _quantile(distance("logged_action_sequence"), 0.40),
        },
        "N8_v2": {
            "progress_max": 0.15,
            "current_state_max": _quantile(distance("current_state"), 0.70),
            "history_max": _quantile(distance("recent_state_history"), 0.70),
            "velocity_max": _quantile(distance("recent_state_velocity"), 0.85),
            "visual_min": _quantile(distance("causal_visual_feature"), 0.60),
            "action_sequence_min": _quantile(distance("logged_action_sequence"), 0.25),
        },
    }
    probe_n7_v1 = _select_n7(train, designs["N7_v1"])
    train_anchor_count = sum(row["episode_split"] == "train" for row in anchors)
    train_coverage = len(
        {
            index
            for pair in probe_n7_v1
            for index in (pair["left_anchor_index"], pair["right_anchor_index"])
        }
    ) / train_anchor_count
    n7_version = "N7_v2" if train_coverage < 0.75 else "N7_v1"
    probe_n8_v1 = _eligible_n8(train, designs["N8_v1"])
    n8_episode_pairs_by_task: dict[str, set[str]] = defaultdict(set)
    for row in probe_n8_v1:
        n8_episode_pairs_by_task[row["task_id_provenance_only"]].add(
            row["episode_pair_id_provenance_only"]
        )
    n8_qualifying_tasks = sum(
        len(values) >= N8_PAIRS_PER_TASK for values in n8_episode_pairs_by_task.values()
    )
    n8_version = "N8_v2" if n8_qualifying_tasks < 6 else "N8_v1"
    selected = {
        "N7": {"version": n7_version, "thresholds": designs[n7_version]},
        "N8": {"version": n8_version, "thresholds": designs[n8_version]},
    }
    probe = {
        "schema": "rm-acv-realneg-development-threshold-probe-v1",
        "training_candidates": len(train),
        "test_candidates_used": 0,
        "N7_v1_train_coverage": train_coverage,
        "N7_redesign_triggered": n7_version == "N7_v2",
        "N8_v1_tasks_with_20_episode_pairs": n8_qualifying_tasks,
        "N8_redesign_triggered": n8_version == "N8_v2",
        "selected": selected,
        "one_redesign_per_family_respected": True,
    }
    return selected, probe


def _n7_eligible(row: dict[str, Any], threshold: dict[str, float]) -> bool:
    d = row["distances"]
    return bool(
        d["normalized_progress"] <= threshold["progress_max"]
        and d["current_state"] <= threshold["current_state_max"]
        and d["recent_state_history"] <= threshold["history_max"]
        and d["recent_state_velocity"] <= threshold["velocity_max"]
        and d["gripper_state"] <= threshold["gripper_max"]
        and d["logged_action_sequence"] >= threshold["action_sequence_min"]
    )


def _select_n7(
    candidates: list[dict[str, Any]], threshold: dict[str, float]
) -> list[dict[str, Any]]:
    eligible = [row for row in candidates if _n7_eligible(row, threshold)]
    by_group: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        by_group[(row["task_id_provenance_only"], row["episode_split"])].append(row)
    selected: list[dict[str, Any]] = []
    for group, rows in sorted(by_group.items()):
        ranked = sorted(
            rows,
            key=lambda row: (
                row["distances"]["current_state"]
                + row["distances"]["recent_state_history"]
                + 0.5 * row["distances"]["recent_state_velocity"]
                + 2.0 * row["distances"]["normalized_progress"]
                - 0.15 * row["distances"]["logged_action_sequence"],
                row["candidate_edge_id"],
            ),
        )
        used: set[int] = set()
        for rank, row in enumerate(ranked, start=1):
            left, right = row["left_anchor_index"], row["right_anchor_index"]
            if left in used or right in used:
                continue
            selected.append({**row, "pair_selection_rank": rank})
            used.update((left, right))
    return selected


def _eligible_n8(
    candidates: list[dict[str, Any]], threshold: dict[str, float]
) -> list[dict[str, Any]]:
    result = []
    for row in candidates:
        d = row["distances"]
        if (
            d["normalized_progress"] <= threshold["progress_max"]
            and d["current_state"] <= threshold["current_state_max"]
            and d["recent_state_history"] <= threshold["history_max"]
            and d["recent_state_velocity"] <= threshold["velocity_max"]
            and d["causal_visual_feature"] >= threshold["visual_min"]
            and d["logged_action_sequence"] >= threshold["action_sequence_min"]
        ):
            result.append(row)
    return result


def _select_n8(
    candidates: list[dict[str, Any]], threshold: dict[str, float]
) -> list[dict[str, Any]]:
    eligible = _eligible_n8(candidates, threshold)
    best_by_episode_pair: dict[tuple[str, str], dict[str, Any]] = {}
    for row in eligible:
        key = (row["task_id_provenance_only"], row["episode_pair_id_provenance_only"])
        score = (
            row["distances"]["current_state"]
            + row["distances"]["recent_state_history"]
            + row["distances"]["recent_state_velocity"] * 0.5
            + row["distances"]["normalized_progress"] * 2.0
            - row["distances"]["causal_visual_feature"] * 0.35
            - row["distances"]["logged_action_sequence"] * 0.15
        )
        previous = best_by_episode_pair.get(key)
        if previous is None or (score, row["candidate_edge_id"]) < (
            previous["_selection_score"],
            previous["candidate_edge_id"],
        ):
            best_by_episode_pair[key] = {**row, "_selection_score": score}
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (task, _), row in best_by_episode_pair.items():
        by_task[task].append(row)
    qualifying = {
        task: sorted(rows, key=lambda row: (row["_selection_score"], row["candidate_edge_id"]))
        for task, rows in by_task.items()
        if len(rows) >= N8_PAIRS_PER_TASK
    }
    selected: list[dict[str, Any]] = []
    for task, rows in sorted(qualifying.items()):
        for rank, row in enumerate(rows[:N8_PAIRS_PER_TASK], start=1):
            cleaned = {key: value for key, value in row.items() if key != "_selection_score"}
            selected.append({**cleaned, "pair_selection_rank": rank})
    return selected


def _configuration(
    anchors: list[dict[str, Any]],
    thresholds: dict[str, Any],
    threshold_probe: dict[str, Any],
    visual_config: dict[str, Any],
) -> dict[str, Any]:
    source_hash = sha256_file(OUT / "imported_source_manifest.jsonl")
    anchor_hash = sha256_file(OUT / "imported_anchor_manifest.jsonl")
    return {
        "schema": "rm-acv-realneg-v1-preregistered-config-v1",
        "study": "RM-ACV-REALNEG-V1",
        "hypothesis": "unmodified real logged actions assigned reciprocally to matched wrong contexts avoid synthetic action-value artifacts",
        "source": {
            "trajectories": 166,
            "anchors": len(anchors),
            "tasks": sorted({row["task_id_provenance_only"] for row in anchors}),
            "task_count": 11,
            "embodiment": PRIMARY_EMBODIMENT,
            "state_dimension": 32,
            "action_dimension": 32,
            "history_steps": HISTORY,
            "action_horizon_steps": ACTION_HORIZON,
            "imported_source_manifest_sha256": source_hash,
            "imported_anchor_manifest_sha256": anchor_hash,
            "anchors_regenerated": False,
        },
        "causal_rgb": visual_config,
        "causal_state": "8 official 32D proprioceptive steps ending at the frozen anchor",
        "pair_distance_definitions": {
            "current_state": "RMSE after train-only per-dimension standardization",
            "recent_state_history": "8x32 RMSE after train-only state standardization",
            "recent_state_velocity": "last-step delta RMSE after train-only velocity standardization",
            "normalized_progress": "absolute anchor progress difference",
            "gripper_state": "RMSE of state dimensions 16:18",
            "logged_action_sequence": "16x32 RMSE after train-only action standardization",
            "action_endpoint": "terminal action RMSE",
            "action_magnitude": "absolute mean-norm difference normalized by train-only scale",
            "action_smoothness": "absolute derivative-norm difference normalized by train-only scale",
            "causal_visual_feature": "RMSE of frozen causal RGB-stat history mean after train-only standardization",
        },
        "matching_thresholds": thresholds,
        "threshold_freezing": {
            "development_only": True,
            "episode_heldout_or_task_cv_test_used": False,
            "quantile_rules": {
                "N7_v1": "context upper 65/65/70/70 percentiles; action lower 35th; progress <=.12",
                "N7_v2": "single coverage-triggered redesign using context 80/80/85/85; action lower 20th; progress <=.15",
                "N8_v1": "state/history upper 55th, velocity 75th, visual/action lower 70/40th; progress <=.12",
                "N8_v2": "single coverage-triggered redesign using state/history 70th, velocity 85th, visual/action lower 60/25th; progress <=.15",
            },
            "development_probe_sha256": sha256_json(threshold_probe),
        },
        "families": {
            "N7": {
                "name": FAMILIES[0],
                "matching": "same-task/split, different-episode, symmetric greedy minimum-cost matching with each anchor used at most once",
                "reciprocal_four_sample_construction": True,
                "coverage_gate": {"anchors_min_fraction": 0.70, "tasks": 11, "pairs_min": 400},
            },
            "N8": {
                "name": FAMILIES[1],
                "matching": "same-task/split, different episode, state-similar/visual-divergent; best anchor edge per unordered episode pair",
                "reciprocal_four_sample_construction": True,
                "coverage_gate": {
                    "tasks_min": 6,
                    "independent_episode_pairs_per_task_min": 20,
                    "pairs_min": 120,
                    "single_task_fraction_max": 0.25,
                },
            },
        },
        "pair_balancing": {
            "samples_per_pair": 4,
            "positive": ["(C_i,A_i)", "(C_j,A_j)"],
            "negative": ["(C_i,A_j)", "(C_j,A_i)"],
            "exact_action_positive_negative_counts": True,
            "exact_context_positive_negative_counts": True,
        },
        "splits": {
            "episode_split": "imported unchanged from pilot166; all four reciprocal samples stay together",
            "task_cv": "imported unchanged 5-fold complete-task CV",
        },
        "shortcut_gates": {
            "exact_action_lookup_BA": 0.50,
            "exact_context_lookup_BA": 0.50,
            "learned_action_only_BA_max": 0.54,
            "learned_context_only_BA_max": 0.54,
            "task_only_BA_max": 0.53,
            "progress_only_BA_max": 0.55,
            "source_metadata_BA_max": 0.53,
            "pair_mining_statistics_BA_max": 0.55,
            "combined_non_relational_BA_max": 0.58,
        },
        "behavioral_validity": {
            "training_split_neighborhoods_only": True,
            "neighbor_count": 12,
            "mode_distinct_rule": "cross-action distance must exceed 1.25x the maximum local demonstrated action radius",
            "N7_distinct_mode_fraction_min": 0.80,
            "N8_distinct_mode_fraction_min": 0.75,
            "per_task_ambiguous_fraction_max": 0.35,
            "sensitivity_factors": [0.9, 1.0, 1.1],
        },
        "baseline_list_if_R6_authorizes": [
            "state_action_tcn",
            "task_conditioned_action_transformer",
            "visual_state_action_transformer",
            "contrastive_context_action_encoder",
            "masked_action_compatibility_transformer",
            "diagonal_ssm_or_autoregressive_action_likelihood",
        ],
        "benchmark_gate": {
            "minimum_valid_families_including_frozen_N1_N4": 3,
            "non_relational_shortcut_BA_max": 0.60,
            "episode_BA_min": 0.62,
            "task_CV_macro_BA_min": 0.55,
            "context_and_action_gain_min": 0.04,
            "episode_saturation_max_exclusive": 0.85,
            "task_CV_saturation_max_exclusive": 0.80,
        },
        "proposed_method": {
            "family": "PhaseAwareTemporalEnergyVerifier",
            "authorization": "only current-run B3 benchmark-ready decision",
            "seeds": list(SEEDS),
            "explicit_context_action_cross_attention_required": True,
        },
        "method_gate": {
            "episode_BA_gain_min_or_recall_gain_min": [0.03, 0.05],
            "task_CV_macro_gain_min": 0.02,
            "gain_on_new_real_action_family_required": True,
            "three_seed_stability_required": True,
        },
        "resource_limits": {
            "pre_admission_GPU_hours_max": 10,
            "new_artifacts_GiB_max": 20,
            "new_negative_family_designs_max": 2,
            "whole_family_redesign_per_family_max": 1,
            "per_example_manual_correction": False,
            "post_admission_additional_GPU_hours_max": 16,
            "learned_baseline_families_max": 6,
            "proposed_method_families_max": 1,
            "no_unrelated_process_termination": True,
        },
        "fair_model_inputs": list(FAIR_INPUTS),
        "forbidden_predictive_inputs": list(FORBIDDEN_PREDICTIVE_INPUTS)
        + ["pair_id", "matching_distances", "mining_rank"],
        "forbidden_construction": [
            "synthetic action permutation",
            "action reversal",
            "arm-gripper delay injection",
            "joint-phase perturbation",
            "endpoint path interpolation",
            "Gaussian noise",
        ],
        "environment": environment_manifest(),
        "generator": {
            "path": str(Path(__file__).relative_to(ROOT)),
            "sha256": sha256_file(Path(__file__)),
        },
    }


def _pair_manifest(
    selected: Sequence[dict[str, Any]],
    family: str,
    anchors: list[dict[str, Any]],
    arrays: dict[str, np.ndarray],
    visual: np.ndarray,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pairs: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    for ordinal, edge in enumerate(selected):
        left = int(edge["left_anchor_index"])
        right = int(edge["right_anchor_index"])
        pair_id = f"{family.lower()}_p{ordinal:06d}"
        context_hashes = {
            left: hashlib.sha256(
                arrays["history_state"][left].tobytes()
                + visual[left].tobytes()
                + arrays["instruction"][left].tobytes()
            ).hexdigest(),
            right: hashlib.sha256(
                arrays["history_state"][right].tobytes()
                + visual[right].tobytes()
                + arrays["instruction"][right].tobytes()
            ).hexdigest(),
        }
        action_hashes = {
            left: hashlib.sha256(arrays["positive_action"][left].tobytes()).hexdigest(),
            right: hashlib.sha256(arrays["positive_action"][right].tobytes()).hexdigest(),
        }
        pair = {
            "schema": "rm-acv-realneg-reciprocal-pair-v1",
            "pair_id": pair_id,
            "family": family,
            "task_id_provenance_only": edge["task_id_provenance_only"],
            "episode_split": edge["episode_split"],
            "context_ids": [anchors[left]["anchor_id"], anchors[right]["anchor_id"]],
            "context_indices": [left, right],
            "action_ids": [
                f"{anchors[left]['anchor_id']}:logged_action",
                f"{anchors[right]['anchor_id']}:logged_action",
            ],
            "action_hashes": [action_hashes[left], action_hashes[right]],
            "episode_ids_provenance_only": [
                anchors[left]["episode_id_provenance_only"],
                anchors[right]["episode_id_provenance_only"],
            ],
            "episode_pair_id_provenance_only": edge["episode_pair_id_provenance_only"],
            "matching_distances": edge["distances"],
            "pair_selection_rank": edge["pair_selection_rank"],
            "construction": "reciprocal_unmodified_real_action_swap",
            "construction_failure_reason": None,
            "action_values_modified": False,
        }
        arrangements = (
            (left, left, 1, "Ci_Ai"),
            (right, right, 1, "Cj_Aj"),
            (left, right, 0, "Ci_Aj"),
            (right, left, 0, "Cj_Ai"),
        )
        pair_samples: list[dict[str, Any]] = []
        for context_index, action_index, label, role in arrangements:
            sample = {
                "schema": "rm-acv-realneg-sample-v1",
                "sample_id": f"{pair_id}:{role}",
                "pair_id": pair_id,
                "family": family,
                "role": role,
                "label": label,
                "episode_split": edge["episode_split"],
                "task_id_provenance_only": edge["task_id_provenance_only"],
                "context_id": anchors[context_index]["anchor_id"],
                "context_index": context_index,
                "context_hash": context_hashes[context_index],
                "context_episode_id_provenance_only": anchors[context_index][
                    "episode_id_provenance_only"
                ],
                "action_id": f"{anchors[action_index]['anchor_id']}:logged_action",
                "action_index": action_index,
                "action_hash": action_hashes[action_index],
                "action_source_episode_id_provenance_only": anchors[action_index][
                    "episode_id_provenance_only"
                ],
                "matching_distances": edge["distances"],
                "pair_selection_rank": edge["pair_selection_rank"],
            }
            sample["sample_sha256"] = sha256_json(sample)
            samples.append(sample)
            pair_samples.append(sample)
        pair["positive_sample_hashes"] = [
            sample["sample_sha256"] for sample in pair_samples if sample["label"] == 1
        ]
        pair["negative_sample_hashes"] = [
            sample["sample_sha256"] for sample in pair_samples if sample["label"] == 0
        ]
        pair["action_reuse_counts"] = [
            {
                "action_id": action_id,
                "action_hash": action_hash,
                "positive": 1,
                "negative": 1,
            }
            for action_id, action_hash in zip(pair["action_ids"], pair["action_hashes"])
        ]
        pair["context_reuse_counts"] = [
            {
                "context_id": context_id,
                "positive": 1,
                "negative": 1,
            }
            for context_id in pair["context_ids"]
        ]
        pair["pair_sha256"] = sha256_json(pair)
        pairs.append(pair)
    return pairs, samples


def _balance_audits(
    pairs_by_family: dict[str, list[dict[str, Any]]],
    samples: list[dict[str, Any]],
    anchors: list[dict[str, Any]],
) -> dict[str, Any]:
    action_audit: dict[str, Any] = {}
    context_audit: dict[str, Any] = {}
    balance: dict[str, Any] = {}
    for family in FAMILIES:
        rows = [row for row in samples if row["family"] == family]
        action_counts: dict[str, Counter[int]] = defaultdict(Counter)
        context_counts: dict[str, Counter[int]] = defaultdict(Counter)
        for row in rows:
            action_counts[row["action_hash"]][int(row["label"])] += 1
            context_counts[row["context_id"]][int(row["label"])] += 1
        action_failures = [
            key for key, count in action_counts.items() if count[0] != count[1]
        ]
        context_failures = [
            key for key, count in context_counts.items() if count[0] != count[1]
        ]
        action_audit[family] = {
            "unique_action_hashes": len(action_counts),
            "imbalanced_hashes": action_failures,
            "exact": not action_failures,
        }
        context_audit[family] = {
            "unique_context_ids": len(context_counts),
            "imbalanced_contexts": context_failures,
            "exact": not context_failures,
        }
        split_rows = defaultdict(list)
        for row in rows:
            split_rows[row["episode_split"]].append(row)
        balance[family] = {
            split: {
                "samples": len(values),
                "positive_fraction": (
                    sum(row["label"] for row in values) / len(values) if values else None
                ),
                "exact_0_50": (
                    sum(row["label"] for row in values) * 2 == len(values) if values else None
                ),
            }
            for split, values in sorted(split_rows.items())
        }
    write_json(
        OUT / "action_reuse_audit.json",
        {
            "schema": "rm-acv-realneg-action-reuse-v1",
            "families": action_audit,
            "pass": all(value["exact"] for value in action_audit.values()),
        },
    )
    write_json(
        OUT / "context_reuse_audit.json",
        {
            "schema": "rm-acv-realneg-context-reuse-v1",
            "families": context_audit,
            "pass": all(value["exact"] for value in context_audit.values()),
        },
    )
    anchor_by_id = {row["anchor_id"]: row for row in anchors}
    pair_group_pass = all(
        all(sample["episode_split"] == pair["episode_split"] for sample in samples if sample["pair_id"] == pair["pair_id"])
        for pairs in pairs_by_family.values()
        for pair in pairs
    )
    task_folds = json.loads(
        (OUT / "imported_task_cv_preregistered.json").read_text(encoding="utf-8")
    )["folds"]
    held_out_tasks = [
        task for fold in task_folds for task in fold["test_tasks"]
    ]
    all_tasks = sorted({row["task_id_provenance_only"] for row in anchors})
    task_cv_complete = (
        sorted(held_out_tasks) == all_tasks
        and len(held_out_tasks) == len(set(held_out_tasks))
        and all(
            not (
                set(fold["train_tasks"]) & set(fold["validation_tasks"])
                or set(fold["train_tasks"]) & set(fold["test_tasks"])
                or set(fold["validation_tasks"]) & set(fold["test_tasks"])
            )
            for fold in task_folds
        )
    )
    write_json(
        OUT / "pair_grouping_audit.json",
        {
            "schema": "rm-acv-realneg-pair-grouping-v1",
            "pairs": sum(len(values) for values in pairs_by_family.values()),
            "all_four_samples_same_split": pair_group_pass,
            "task_cv_holds_out_complete_tasks": task_cv_complete,
            "every_task_held_out_exactly_once": task_cv_complete,
            "pass": pair_group_pass and task_cv_complete,
        },
    )
    episode_split = {
        row["episode_id_provenance_only"]: row["episode_split"] for row in anchors
    }
    episode_containment = all(
        episode_split[row["context_episode_id_provenance_only"]]
        == row["episode_split"]
        == episode_split[row["action_source_episode_id_provenance_only"]]
        for row in samples
    )
    write_json(
        OUT / "episode_leakage_audit.json",
        {
            "schema": "rm-acv-realneg-episode-leakage-v1",
            "episode_grouping": True,
            "retrieval_source_containment": episode_containment,
            "anchors_regenerated": False,
            "pass": episode_containment,
        },
    )
    sample_splits: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in samples:
        sample_splits[(row["context_hash"], row["action_hash"])].add(row["episode_split"])
    cross_split_duplicates = [
        {"context_hash": key[0], "action_hash": key[1], "splits": sorted(values)}
        for key, values in sample_splits.items()
        if len(values) > 1
    ]
    write_json(
        OUT / "duplicate_audit.json",
        {
            "schema": "rm-acv-realneg-duplicate-audit-v1",
            "cross_split_identical_context_action_pairs": cross_split_duplicates,
            "pass": not cross_split_duplicates,
        },
    )
    balance_pass = all(
        item["exact_0_50"] is not False
        for family in balance.values()
        for item in family.values()
    )
    write_json(
        OUT / "balance_audit.json",
        {
            "schema": "rm-acv-realneg-balance-audit-v1",
            "families": balance,
            "pass": balance_pass,
        },
    )
    return {
        "action_exact": all(value["exact"] for value in action_audit.values()),
        "context_exact": all(value["exact"] for value in context_audit.values()),
        "pair_grouping": pair_group_pass and task_cv_complete,
        "episode_containment": episode_containment,
        "duplicate_safety": not cross_split_duplicates,
        "balance": balance_pass,
    }


def run(*, rehash_source: bool = True) -> dict[str, Any]:
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError("RM-ACV-REALNEG-V1 already initialized; use --fresh for this exact branch")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "run_log.jsonl").unlink(missing_ok=True)
    started = time.monotonic()
    append_log("R0_START", study="RM-ACV-REALNEG-V1")
    anchors, arrays = _import_frozen_pilot(rehash_source=rehash_source)
    append_log(
        "R0_IMPORTS_VERIFIED",
        trajectories=166,
        anchors=len(anchors),
        source_content_rehashed=rehash_source,
    )
    visual, visual_config = _extract_visual_features(anchors)
    append_log(
        "R0_VISUAL_ENCODER_FROZEN",
        encoder_sha256=visual_config["source_module_sha256"],
        feature_sha256=visual_config["feature_sha256"],
    )
    normalization = _training_normalization(anchors, arrays, visual)
    candidates = _pair_candidates(anchors, arrays, visual, normalization)
    thresholds, threshold_probe = _derive_thresholds(candidates, anchors)
    write_json(OUT / "development_threshold_probe.json", threshold_probe)
    config = _configuration(anchors, thresholds, threshold_probe, visual_config)
    write_json(OUT / "preregistered_config.json", config)
    (OUT / "preregistered_config.sha256").write_text(
        sha256_file(OUT / "preregistered_config.json") + "\n", encoding="utf-8"
    )
    append_log(
        "R0_PREREGISTRATION_FROZEN",
        config_sha256=sha256_file(OUT / "preregistered_config.json"),
        N7_version=thresholds["N7"]["version"],
        N8_version=thresholds["N8"]["version"],
        test_statistics_used=False,
    )
    n7_selected = _select_n7(candidates, thresholds["N7"]["thresholds"])
    n8_selected = _select_n8(candidates, thresholds["N8"]["thresholds"])
    selected_edge_ids = {
        "N7": {row["candidate_edge_id"] for row in n7_selected},
        "N8": {row["candidate_edge_id"] for row in n8_selected},
    }
    for row in candidates:
        row["selected_N7"] = row["candidate_edge_id"] in selected_edge_ids["N7"]
        row["selected_N8"] = row["candidate_edge_id"] in selected_edge_ids["N8"]
    write_jsonl(OUT / "pair_candidate_manifest.jsonl", candidates)
    write_json(
        OUT / "pair_distance_schema.json",
        {
            "schema": "rm-acv-realneg-pair-distance-schema-v1",
            "definitions": config["pair_distance_definitions"],
            "normalization": "training_normalization.npz",
            "thresholds": thresholds,
            "future_observations_used": False,
        },
    )
    n7_tasks = Counter(row["task_id_provenance_only"] for row in n7_selected)
    n8_tasks = Counter(row["task_id_provenance_only"] for row in n8_selected)
    n7_covered = {
        index for row in n7_selected
        for index in (row["left_anchor_index"], row["right_anchor_index"])
    }
    write_json(
        OUT / "pair_mining_statistics.json",
        {
            "schema": "rm-acv-realneg-pair-mining-statistics-v1",
            "candidate_edges": len(candidates),
            "N7": {
                "selected_pairs": len(n7_selected),
                "unique_anchors": len(n7_covered),
                "anchor_coverage": len(n7_covered) / len(anchors),
                "task_pair_counts": dict(sorted(n7_tasks.items())),
                "tasks": len(n7_tasks),
            },
            "N8": {
                "selected_pairs": len(n8_selected),
                "task_pair_counts": dict(sorted(n8_tasks.items())),
                "tasks": len(n8_tasks),
                "maximum_task_fraction": (
                    max(n8_tasks.values()) / sum(n8_tasks.values()) if n8_tasks else None
                ),
                "independent_episode_pair_rule": True,
            },
            "training_only_threshold_probe": threshold_probe,
        },
    )
    candidate_anchor_ids = {
        row["left_anchor_id"] for row in candidates
    } | {row["right_anchor_id"] for row in candidates}
    pair_failures = [
        {
            "anchor_id": row["anchor_id"],
            "family": "PAIR_FOUNDATION",
            "reason": "no broad same-task, same-split, different-episode candidate within progress 0.20",
            "manual_repair": False,
        }
        for row in anchors
        if row["anchor_id"] not in candidate_anchor_ids
    ]
    if len(n7_selected) < 400:
        pair_failures.append(
            {
                "family": "N7",
                "reason": f"coverage gate shortfall: {len(n7_selected)} reciprocal pairs < 400",
                "manual_repair": False,
            }
        )
    if len(n8_selected) < 120:
        pair_failures.append(
            {
                "family": "N8",
                "reason": f"coverage gate shortfall: {len(n8_selected)} reciprocal pairs < 120",
                "manual_repair": False,
            }
        )
    write_jsonl(OUT / "pair_failure_log.jsonl", pair_failures)
    append_log(
        "R1_PAIR_MINING_COMPLETE",
        candidate_edges=len(candidates),
        N7_pairs=len(n7_selected),
        N8_pairs=len(n8_selected),
    )
    n7_pairs, n7_samples = _pair_manifest(n7_selected, FAMILIES[0], anchors, arrays, visual)
    n8_pairs, n8_samples = _pair_manifest(n8_selected, FAMILIES[1], anchors, arrays, visual)
    write_jsonl(OUT / "n7_pair_manifest.jsonl", n7_pairs)
    write_jsonl(OUT / "n8_pair_manifest.jsonl", n8_pairs)
    all_samples = n7_samples + n8_samples
    write_jsonl(OUT / "real_negative_manifest.jsonl", all_samples)
    write_jsonl(OUT / "real_negative_failure_log.jsonl", pair_failures)
    write_json(
        OUT / "real_negative_statistics.json",
        {
            "schema": "rm-acv-realneg-statistics-v1",
            "families": {
                FAMILIES[0]: {
                    "pairs": len(n7_pairs),
                    "samples": len(n7_samples),
                    "positive": sum(row["label"] for row in n7_samples),
                    "negative": len(n7_samples) - sum(row["label"] for row in n7_samples),
                    "actions_modified": 0,
                },
                FAMILIES[1]: {
                    "pairs": len(n8_pairs),
                    "samples": len(n8_samples),
                    "positive": sum(row["label"] for row in n8_samples),
                    "negative": len(n8_samples) - sum(row["label"] for row in n8_samples),
                    "actions_modified": 0,
                },
            },
            "synthetic_action_values_created": False,
            "per_example_manual_corrections": 0,
        },
    )
    append_log(
        "R2_REAL_ACTION_NEGATIVES_COMPLETE",
        N7_samples=len(n7_samples),
        N8_samples=len(n8_samples),
        synthetic_action_values=0,
    )
    audits = _balance_audits(
        {FAMILIES[0]: n7_pairs, FAMILIES[1]: n8_pairs},
        all_samples,
        anchors,
    )
    append_log("R3_BALANCE_AND_LEAKAGE_COMPLETE", **audits)
    result = {
        "status": "R0_R3_COMPLETE",
        "N7_pairs": len(n7_pairs),
        "N7_anchor_coverage": len(n7_covered) / len(anchors),
        "N7_tasks": len(n7_tasks),
        "N8_pairs": len(n8_pairs),
        "N8_tasks": len(n8_tasks),
        "exact_balance_and_leakage_pass": all(audits.values()),
        "wall_seconds": time.monotonic() - started,
    }
    write_json(OUT / "r0_r3_status.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--skip-source-rehash", action="store_true")
    args = parser.parse_args()
    if args.fresh and OUT.exists():
        expected = (ROOT / "outputs" / "actmask" / "rm_acv_realneg_v1").resolve()
        if OUT.resolve() != expected:
            raise RuntimeError(f"refusing to clear unexpected output path: {OUT}")
        shutil.rmtree(OUT)
    print(json.dumps(run(rehash_source=not args.skip_source_rehash), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
