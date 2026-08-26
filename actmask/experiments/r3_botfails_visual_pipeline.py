"""Causal multi-horizon R3 BotFails task construction and frozen RGB features."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

import cv2
import imageio.v3 as iio
import numpy as np
import torch
from torchvision.models import resnet18

from actmask.data.r2_real_outcome_adapter import r2_append_log
from actmask.data.real_robot_dataset_adapter import forbidden_input_audit, read_jsonl, sha256_file, write_json, write_jsonl
from actmask.data.r3_botfails_video_adapter import R2_PROCESSED, R3_ROOT, SOURCE_ROOT, VIEW, _local_video_path


OUT = R3_ROOT
TASK_DIR = OUT / "task_formulation"
FEATURE_DIR = OUT / "processed_features"
BACKBONE = OUT / "backbones" / "resnet18-f37072fd.pth"
HISTORY = 32
OFFSETS = (31, 23, 15, 7)
HORIZONS = (8, 16, 32)
SEED = 20260725
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class R3TaskError(RuntimeError):
    pass


def _load_array(row: Mapping[str, Any], field: str) -> np.ndarray:
    return np.load(R2_PROCESSED / str(row[field]))["value"].astype(np.float32)


def _hard_split(row: Mapping[str, Any]) -> str:
    return str(dict(row["split_metadata"])["hard_task_heldout"])


def _target(label: np.ndarray, anchor: int, horizon: int) -> int:
    return int(bool(label[anchor:anchor + horizon].any()))


def _window_metrics(state: np.ndarray, action: np.ndarray, anchor: int, horizon: int) -> dict[str, float]:
    chunk = action[anchor:anchor + min(16, horizon)]
    return {
        "state_norm": float(np.linalg.norm(state[anchor - 1])),
        "action_magnitude": float(np.linalg.norm(chunk, axis=1).mean()),
        "action_smoothness": float(np.linalg.norm(np.diff(chunk, axis=0), axis=1).mean()) if len(chunk) > 1 else 0.0,
    }


def _safe_anchors(label: np.ndarray, onset: int, horizon: int) -> list[int]:
    return [anchor for anchor in range(HISTORY, onset - horizon + 1, 4) if not _target(label, anchor, horizon)]


def _best_matched_safe(state: np.ndarray, action: np.ndarray, label: np.ndarray, onset: int, positive_anchor: int, horizon: int, *, exclude: set[int]) -> tuple[int, dict[str, float]]:
    candidates = [anchor for anchor in _safe_anchors(label, onset, horizon) if anchor not in exclude]
    if not candidates:
        raise R3TaskError("no valid pre-failure safe anchor")
    reference = _window_metrics(state, action, positive_anchor, horizon)
    length = len(label)
    scored: list[tuple[float, int, dict[str, float]]] = []
    for anchor in candidates:
        values = _window_metrics(state, action, anchor, horizon)
        score = (
            abs(anchor / max(1, length - 1) - positive_anchor / max(1, length - 1))
            + 0.05 * abs(values["state_norm"] - reference["state_norm"])
            + 0.15 * abs(values["action_magnitude"] - reference["action_magnitude"])
            + 0.30 * abs(values["action_smoothness"] - reference["action_smoothness"])
        )
        scored.append((score, anchor, values))
    _, anchor, values = min(scored, key=lambda item: (item[0], -item[1]))
    return anchor, values


def _early_safe_anchor(label: np.ndarray, onset: int, horizon: int) -> int:
    candidates = _safe_anchors(label, onset, horizon)
    if not candidates:
        raise R3TaskError("no early safe anchor")
    # First quartile of the available pre-failure segment, yet still fully causal.
    return candidates[max(0, (len(candidates) - 1) // 4)]


def _near_safe_anchor(label: np.ndarray, onset: int, horizon: int) -> int:
    candidates = _safe_anchors(label, onset, horizon)
    if not candidates:
        raise R3TaskError("no near-failure safe anchor")
    return candidates[-1]


def build_task() -> dict[str, Any]:
    """Build all causal windows from the 50 R2 failure trajectories.

    The normal R2 source partition is intentionally excluded: it would make
    source identity a label proxy. Every R3 sample comes from BotFails `test`.
    """

    episodes = read_jsonl(R2_PROCESSED / "episodes.jsonl")
    failures = [row for row in episodes if row["success_or_failure"] == "failure"]
    if len(failures) != 50 or any(dict(row["split_metadata"])["source_split_not_model_input"] != "test" for row in failures):
        raise R3TaskError("R3 requires exactly 50 test-source official failure trajectories")
    windows: list[dict[str, Any]] = []
    state_rows: list[np.ndarray] = []
    action_rows: list[np.ndarray] = []
    label_rows: list[int] = []
    for episode in sorted(failures, key=lambda row: str(row["episode_id"])):
        state = _load_array(episode, "robot_state_path")
        action = _load_array(episode, "action_path")
        failure = _load_array(episode, "failure_label_path").astype(np.int8)
        onset_indices = np.flatnonzero(failure)
        if not len(onset_indices):
            raise R3TaskError(f"failure label missing in {episode['episode_id']}")
        onset = int(onset_indices[0])
        for horizon in HORIZONS:
            imminent = onset - max(1, horizon // 2)
            if imminent < HISTORY or imminent + horizon > len(action):
                raise R3TaskError(f"invalid imminent window for {episode['episode_id']} / {horizon}")
            if failure[imminent - HISTORY:imminent].any() or not _target(failure, imminent, horizon):
                raise R3TaskError(f"imminent causal label failure for {episode['episode_id']} / {horizon}")
            early = _early_safe_anchor(failure, onset, horizon)
            near = _near_safe_anchor(failure, onset, horizon)
            matched, matched_metrics = _best_matched_safe(state, action, failure, onset, imminent, horizon, exclude={early, near})
            kinds = [
                ("early_safe", early),
                ("matched_safe", matched),
                ("near_failure", near),
                ("failure_imminent", imminent),
            ]
            for kind, anchor in kinds:
                target = _target(failure, anchor, horizon)
                if kind == "failure_imminent" and target != 1:
                    raise R3TaskError("positive target is not imminent")
                if kind != "failure_imminent" and target != 0:
                    raise R3TaskError("safe target unexpectedly positive")
                indices = [anchor - offset for offset in OFFSETS]
                if any(index < 0 or index >= anchor for index in indices):
                    raise R3TaskError("noncausal RGB frame index")
                metrics = _window_metrics(state, action, anchor, horizon)
                windows.append({
                    "window_id": f"{episode['episode_id']}:h{horizon}:{kind}", "episode_id": episode["episode_id"],
                    "task_name": episode["task_name"], "horizon_steps": horizon, "window_kind": kind,
                    "anchor": anchor, "rgb_indices": indices, "history_start": anchor - HISTORY,
                    "candidate_action_end_exclusive": anchor + 16, "target_end_exclusive": anchor + horizon,
                    "target_failure_within_horizon": target, "official_failure_onset": onset,
                    "episode_split": episode["split"], "task_split": _hard_split(episode), "source_partition": "test",
                    "matching": {"matched_anchor": matched, "matched_metrics": matched_metrics, "window_metrics": metrics} if kind == "matched_safe" else {"window_metrics": metrics},
                })
                state_rows.append(state[anchor - HISTORY:anchor])
                action_rows.append(action[anchor:anchor + 16])
                label_rows.append(target)
    labels = np.asarray(label_rows, dtype=np.int64)
    history = np.stack(state_rows).astype(np.float32)
    action = np.stack(action_rows).astype(np.float32)
    groups = defaultdict(list)
    for row in windows:
        groups[str(row["episode_id"])].append(row)
    if len(windows) != 50 * len(HORIZONS) * 4 or labels.sum() != 50 * len(HORIZONS):
        raise R3TaskError("unexpected multi-horizon cardinality")
    leakage = {
        "pass": True,
        "future_rgb_input": False, "post_anchor_robot_state_input": False, "failure_label_input": False,
        "source_identity_input": False, "episode_identity_input": False, "absolute_frame_index_input": False,
        "input_fields": ["pre_anchor_rgb_history", "pre_anchor_robot_state_history", "candidate_action_chunk"],
        "input_audit": forbidden_input_audit(["history_rgb", "history_robot_state", "candidate_action_chunk"]),
        "causal_frame_check": all(max(row["rgb_indices"]) < row["anchor"] for row in windows),
    }
    leakage["pass"] = leakage["input_audit"]["pass"] and leakage["causal_frame_check"]
    split_audit = {
        "pass": all(len({item["episode_split"] for item in rows}) == 1 and len({item["task_split"] for item in rows}) == 1 for rows in groups.values()),
        "episode_groups": len(groups),
        "episode_split_counts": dict(Counter(str(row["episode_split"]) for row in windows)),
        "task_split_counts": dict(Counter(str(row["task_split"]) for row in windows)),
        "failure_pattern_split": {"status": "unavailable", "reason": "R2 frozen manifest has onset labels but no independent per-episode failure-category field"},
    }
    matching = {
        "pass": True,
        "policy": "same episode and task; candidate minimizes normalized position, last-state norm, action magnitude and action smoothness difference among valid pre-onset windows",
        "per_horizon_windows": {str(horizon): int(sum(row["horizon_steps"] == horizon for row in windows)) for horizon in HORIZONS},
        "negative_counts": dict(Counter(row["window_kind"] for row in windows if row["target_failure_within_horizon"] == 0)),
        "no_fallback_pool": True,
    }
    if not leakage["pass"] or not split_audit["pass"]:
        raise R3TaskError("task audit failed")
    TASK_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(TASK_DIR / "windows.jsonl", windows)
    np.savez_compressed(TASK_DIR / "numeric_windows.npz", history=history, action=action, label=labels)
    write_json(TASK_DIR / "task_manifest.json", {"schema": "r3-multi-horizon-task-v1", "windows": len(windows), "horizons": list(HORIZONS), "r2_episodes_sha256": sha256_file(R2_PROCESSED / "episodes.jsonl"), "windows_sha256": sha256_file(TASK_DIR / "windows.jsonl")})
    write_json(TASK_DIR / "window_matching_audit.json", matching)
    write_json(TASK_DIR / "leakage_audit.json", leakage)
    write_json(TASK_DIR / "split_audit.json", split_audit)
    r2_append_log(OUT / "run_log.jsonl", event="r3_multi_horizon_task_constructed", payload={"windows": len(windows), "positive": int(labels.sum()), "horizons": list(HORIZONS)})
    return {"windows": windows, "history": history, "action": action, "label": labels}


def _backbone() -> torch.nn.Module:
    if not BACKBONE.is_file():
        raise FileNotFoundError(f"missing frozen R3 backbone weights: {BACKBONE}")
    model = resnet18(weights=None)
    state = torch.load(BACKBONE, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.fc = torch.nn.Identity()
    model.eval().to(DEVICE)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _mode_a(frames: list[np.ndarray]) -> np.ndarray:
    small = [cv2.resize(frame, (112, 84), interpolation=cv2.INTER_AREA) for frame in frames]
    gray = [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in small]
    brightness = float(np.mean([image.mean() / 255.0 for image in gray]))
    contrast = float(np.mean([image.std() / 255.0 for image in gray]))
    diffs = [float(np.abs(current.astype(np.float32) - previous.astype(np.float32)).mean() / 255.0) for previous, current in zip(gray, gray[1:])]
    flow = []
    for previous, current in zip(gray, gray[1:]):
        values = cv2.calcOpticalFlowFarneback(previous, current, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flow.append(float(np.linalg.norm(values, axis=2).mean()))
    duplicate = float(sum(np.array_equal(previous, current) for previous, current in zip(gray, gray[1:]))) / max(1, len(gray) - 1)
    return np.asarray([brightness, contrast, float(np.mean(diffs) if diffs else 0.0), float(np.mean(flow) if flow else 0.0), duplicate], dtype=np.float32)


def _preprocess(frames: list[np.ndarray]) -> torch.Tensor:
    values = []
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    for frame in frames:
        image = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(image.transpose(2, 0, 1).copy()).float() / 255.0
        values.append((tensor - mean) / std)
    return torch.stack(values)


def extract_features() -> dict[str, Any]:
    alignment_path = OUT / "video_alignment_audit.json"
    if not alignment_path.is_file() or not json.loads(alignment_path.read_text()).get("pass"):
        raise R3TaskError("full video alignment audit must pass before feature extraction")
    task = build_task()
    windows = task["windows"]
    by_episode: dict[str, list[tuple[int, Mapping[str, Any]]]] = defaultdict(list)
    for index, window in enumerate(windows):
        by_episode[str(window["episode_id"])].append((index, window))
    mode_a = np.zeros((len(windows), 5), np.float32)
    resnet = np.zeros((len(windows), len(OFFSETS), 512), np.float32)
    future_resnet = np.zeros((len(windows), 512), np.float32)
    frame_hashes: list[str] = [""] * len(windows)
    model = _backbone()
    for episode_id, entries in sorted(by_episode.items()):
        video = SOURCE_ROOT / VIEW / f"{episode_id}.mp4"
        desired = sorted({index for _, row in entries for index in (list(row["rgb_indices"]) + [int(row["target_end_exclusive"]) - 1])})
        required = set(desired)
        frames: dict[int, np.ndarray] = {}
        for index, frame in enumerate(iio.imiter(video, plugin="FFMPEG")):
            if index in required:
                frames[index] = np.asarray(frame, dtype=np.uint8)
            if len(frames) == len(required):
                break
        if set(frames) != required:
            raise R3TaskError(f"missing causal sampled frames for {episode_id}: {sorted(required.difference(frames))[:3]}")
        ordered_indices = sorted(required)
        with torch.no_grad():
            batch = _preprocess([frames[index] for index in ordered_indices]).to(DEVICE)
            feature = model(batch).detach().cpu().numpy().astype(np.float32)
        lookup = {index: feature[position] for position, index in enumerate(ordered_indices)}
        for row_index, row in entries:
            selected = [frames[index] for index in row["rgb_indices"]]
            mode_a[row_index] = _mode_a(selected)
            resnet[row_index] = np.stack([lookup[index] for index in row["rgb_indices"]])
            # Strictly diagnostic leakage oracle: it is after anchor and never
            # part of the fair visual history tensor or recoverability gate.
            future_resnet[row_index] = lookup[int(row["target_end_exclusive"]) - 1]
            frame_hashes[row_index] = hashlib.sha256(b"".join(hashlib.sha256(frame.tobytes()).digest() for frame in selected)).hexdigest()
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(FEATURE_DIR / "visual_features.npz", mode_a=mode_a, resnet_history=resnet, current_resnet=resnet[:, -1], future_resnet_diagnostic=future_resnet, frame_hash=np.asarray(frame_hashes))
    manifest = {
        "schema": "r3-visual-feature-manifest-v1", "samples": len(windows), "view": VIEW,
        "mode_a": {"dimension": 5, "features": ["brightness", "contrast", "frame_difference", "farneback_motion", "camera_freeze_proxy"]},
        "mode_b": {"backbone": "torchvision ResNet-18 ImageNet-1K V1", "torchvision_version": "0.21.0+cu124", "weights_path": str(BACKBONE.relative_to(OUT)), "weights_sha256": sha256_file(BACKBONE), "feature_dimension": 512, "history_frames": len(OFFSETS), "preprocess": "RGB resize 224x224; ImageNet mean/std; frozen weights"},
        "causality": "frame hashes are for indices anchor-[31,23,15,7], all strictly before anchor", "future_rgb_diagnostic": "future_resnet_diagnostic uses target_end_exclusive-1 only as an explicitly leakage-marked privileged diagnostic; it is excluded from all fair models and gates", "feature_file_sha256": sha256_file(FEATURE_DIR / "visual_features.npz"),
        "extraction_command": "python -m actmask.experiments.r3_botfails_visual_pipeline --extract-features",
    }
    write_json(FEATURE_DIR / "feature_manifest.json", manifest)
    r2_append_log(OUT / "run_log.jsonl", event="r3_frozen_visual_features_extracted", payload={"samples": len(windows), "feature_sha256": manifest["feature_file_sha256"], "device": str(DEVICE)})
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-task", action="store_true")
    parser.add_argument("--extract-features", action="store_true")
    args = parser.parse_args()
    if args.build_task == args.extract_features:
        parser.error("choose exactly one action")
    result = build_task() if args.build_task else extract_features()
    print(json.dumps({"windows": len(result["windows"])} if "windows" in result else {"samples": result["samples"]}, sort_keys=True))


if __name__ == "__main__":
    main()
