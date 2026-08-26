"""S0: freeze the only permitted RM-SPD data and protocol before execution."""
from __future__ import annotations

import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2

from actmask.data.robomind_adapter import RoboMINDRecord, load_numeric_episode, validate_numeric_episode
from actmask.experiments.rm_spd_common import OUT, RM_SERIES, append_log, sha256_file, write_json, write_jsonl


SAM3_ROOT = Path("/data/projects/tzh/papers/poke_and_splat/sam3")
SAM3_CHECKPOINT = SAM3_ROOT / "checkpoint" / "sam3.pt"
RESNET18_WEIGHTS = Path(
    "/data/projects/tzh/papers/ActMask/outputs/actmask/r3_botfails_visual_failure/backbones/resnet18-f37072fd.pth"
)


def _command(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT, timeout=30).strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def _record(row: dict[str, Any]) -> RoboMINDRecord:
    return RoboMINDRecord(**{field: row[field] for field in RoboMINDRecord.__dataclass_fields__})


def _video_metadata(path: str) -> dict[str, Any]:
    video = cv2.VideoCapture(path)
    if not video.isOpened():
        raise RuntimeError(f"cannot open frozen RoboMIND front video: {path}")
    try:
        return {
            "frames": int(video.get(cv2.CAP_PROP_FRAME_COUNT)),
            "fps": float(video.get(cv2.CAP_PROP_FPS)),
            "width": int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
    finally:
        video.release()


def _split(selected: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        by_task[str(row["task_id"])].append(row)
    episode_split: dict[str, str] = {}
    for task, rows in by_task.items():
        rows.sort(key=lambda x: int(x["episode_index"]))
        if len(rows) != 10:
            raise RuntimeError(f"expected exactly ten frozen episodes for {task}, found {len(rows)}")
        for index, row in enumerate(rows):
            episode_split[str(row["episode_id"])] = "train" if index < 6 else "val" if index < 8 else "test"
    tasks = sorted(by_task)
    task_split = {task: "train" if index < 6 else "val" if index < 8 else "test" for index, task in enumerate(tasks)}
    return {"episode_held_out": episode_split, "task_held_out": task_split}


def run() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    inherited = json.loads((RM_SERIES / "source_subset" / "subset_manifest.json").read_text())
    inherited_hashes = json.loads((RM_SERIES / "source_hash_manifest.json").read_text())["files"]
    selected = list(inherited["episodes_detail"])
    tasks = sorted({str(row["task_id"]) for row in selected})
    if len(selected) != 100 or len(tasks) != 10:
        raise RuntimeError(f"RM-SPD requires exact inherited 100/10 selection, got {len(selected)}/{len(tasks)}")
    if not SAM3_CHECKPOINT.is_file() or not RESNET18_WEIGHTS.is_file():
        raise RuntimeError("fixed local SAM3 checkpoint or frozen ResNet-18 weights are missing")

    config = {
        "schema": "rm-spd-preregistered-config-v1",
        "branch": "RM-SPD",
        "source": {
            "dataset": "RoboMIND2.0",
            "selection": "exact inherited balanced Franka RM-Series subset",
            "trajectories": 100,
            "tasks": tasks,
            "source_hash_manifest": "source_hash_manifest.json",
            "source_policy": "read-only; no additional trajectories, cameras, datasets, or human annotations",
        },
        "camera": {"view": "observation.rgb_images.camera_front", "resolution": [640, 480], "camera_mixing": False},
        "temporal": {
            "history_length": 2,
            "future_action_horizon": 8,
            "prediction_horizons_frames": {"short": 4, "medium": 12, "long": 24},
            "anchors_per_trajectory": 1,
            "anchor_rule": "integer floor of half trajectory length, shifted only earlier if needed to retain all fixed history/action/future indices",
            "prior_neighbor_frames": "immediately preceding source frame for each anchor/history/future frame when available",
        },
        "visual_encoder": {
            "architecture": "torchvision ResNet18 ImageNet-1K frozen layer4 patch grid",
            "weights": str(RESNET18_WEIGHTS),
            "weights_sha256": sha256_file(RESNET18_WEIGHTS),
            "input_resolution": [224, 224],
            "patch_grid": [7, 7],
            "feature_dim": 512,
            "normalization": "ImageNet mean/std",
            "frozen": True,
        },
        "sam3": {
            "source_root": str(SAM3_ROOT),
            "source_revision": _command(["git", "-c", f"safe.directory={SAM3_ROOT}", "-C", str(SAM3_ROOT), "rev-parse", "HEAD"]),
            "checkpoint": str(SAM3_CHECKPOINT),
            "checkpoint_sha256": sha256_file(SAM3_CHECKPOINT),
            "prompts": ["robot arm", "robotic arm", "left robot arm", "right robot arm", "robot gripper", "left gripper", "right gripper", "Franka robot"],
            "prompt_policy": "fixed text prompts for every frame; no task/frame-dependent tuning; no spatial boxes or point prompts because fixed reliable geometry is unavailable",
            "processor_confidence_threshold": 0.20,
            "minimum_instance_area_pixels": 64,
            "maximum_instances_per_prompt": 8,
            "proposal_role": "uncertain robotness prior for loss weighting, routing and shortcut controls only; never a label, target, model input, or reference",
        },
        "prior": {
            "probability": "mean over prompt score-weighted retained binary proposals; scores are clipped to [0,1]",
            "temporal_agreement": "IoU between proposal-union maps for a frame and its immediately preceding frame; flow-free conservative correspondence",
            "robot_core": {"robotness_min": 0.90, "temporal_iou_min": 0.80, "erosion_kernel": 3},
            "uncertain": {"robotness_low": 0.10, "robotness_high": 0.90, "boundary_ring_radius": 5, "temporal_iou_min": 0.80},
            "scene_trusted": "low robotness outside uncertain ring and without temporal instability",
            "prior_future_change_prohibited": True,
            "unusable_gate": {"max_median_core_plus_uncertain_fraction": 0.50, "max_p95_core_plus_uncertain_fraction": 0.70},
        },
        "loss_regimes": {
            "full_frame": "uniform patch loss",
            "hard_sam3": "negative control; exclude patches whose fixed binary prompt-union overlap is >=0.5",
            "soft_sam3": {"weight": "(1-p_robot)^gamma", "gamma": 2.0},
            "tri_state_sam3": {"scene_trusted": 1.0, "uncertain": 0.20, "robot_core": 0.0},
        },
        "model_protocol": {
            "learned_seeds": [17, 29, 43],
            "gpu_hour_cap_before_benchmark_decision": 8.0,
            "standard_baselines": ["copy_current", "mean_future_residual", "task_language_only", "episode_progress_diagnostic", "action_magnitude_only", "action_chunk_only", "state_only", "state_action", "current_visual_only", "camera_background_statistics", "global_visual_history_gru", "visual_action_gru", "visual_state_action_gru", "patch_action_mlp", "patch_state_action_mlp", "temporal_patch_transformer", "two_head_patch_gru"],
            "proposed_family": "SAM3PriorRelationalDynamicsNetwork",
            "method_gate": {"relative_feature_error_improvement": 0.10, "absolute_patch_retrieval_improvement": 0.05, "required_horizons": 2, "required_structural_ablations": 2, "seeds": [17, 29, 43]},
        },
        "acceptance_gates": {
            "prior_not_broad": True,
            "full_frame_differs_from_scene_trusted": True,
            "hard_mask_brittle_on_contact_heavy": True,
            "fair_temporal_beats_copy_action_state_action": True,
            "task_held_out_beats_trivial": True,
            "standard_not_privileged_saturated": True,
            "headroom_relative_feature_error": 0.10,
            "headroom_patch_retrieval": 0.05,
            "raw_predictions_saved": True,
            "metrics_independently_recompute": True,
            "artifacts_and_hashes_pass": True,
        },
        "fair_inputs": ["pre_anchor_rgb_or_frozen_features", "pre_anchor_robot_state", "future_action_chunk", "task_instruction_only_when_consistently_available"],
        "forbidden_inputs": ["future_rgb", "future_features", "sam3_future_masks", "source_path", "episode_id", "human_pixel_masks"],
        "artifact_policy": "write only below outputs/actmask/rm_spd_sam3_prior_dynamics; save raw predictions, hashes, manifests and reproducibility audit",
        "claim_boundary": "action-conditioned future scene-feature prediction from real RoboMIND trajectories using uncertain foundation-model robot priors; no segmentation, object-mask, success/failure, physical counterfactual, or alternate-action outcome claim",
    }
    config_path = OUT / "preregistered_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise RuntimeError("refusing to alter frozen RM-SPD config")
    write_json(config_path, config)
    (OUT / "preregistered_config.sha256").write_text(sha256_file(config_path) + "\n")

    hash_by_path = {str(item["local_path"]): item for item in inherited_hashes}
    source_hashes = []
    samples: list[dict[str, Any]] = []
    source_frames: dict[str, dict[str, Any]] = {}
    alignment: list[dict[str, Any]] = []
    for row in selected:
        record = _record(row)
        numeric = load_numeric_episode(record)
        audit = validate_numeric_episode(record, numeric)
        video_meta = _video_metadata(str(record.front_video_path))
        if not audit["pass"] or audit["steps"] != video_meta["frames"]:
            raise RuntimeError(f"unusable source alignment: {record.episode_id}: {audit} {video_meta}")
        length = int(audit["steps"])
        anchor = min(length - 25, max(1, length // 2))
        if anchor < 1 or anchor + 24 >= length or anchor + 7 >= length:
            raise RuntimeError(f"episode too short for frozen RM-SPD window: {record.episode_id} {length}")
        prior_indices = sorted({anchor - 1, anchor, anchor + 4, anchor + 12, anchor + 24, anchor + 3, anchor + 11, anchor + 23})
        history = [anchor - 1, anchor]
        futures = {"short": anchor + 4, "medium": anchor + 12, "long": anchor + 24}
        sample = {
            "schema": "rm-spd-sample-v1",
            "sample_id": record.episode_id.replace(":", "__") + f"__a{anchor:06d}",
            "episode_id": record.episode_id,
            "task_id": record.task_id,
            "episode_index": int(record.episode_index),
            "anchor_frame": anchor,
            "history_frames": history,
            "action_frames": list(range(anchor, anchor + 8)),
            "future_frames": futures,
            "video_path": str(record.front_video_path),
            "parquet_path": str(record.parquet_path),
            "state_dim": int(numeric["state"].shape[1]),
            "action_dim": int(numeric["action"].shape[1]),
            "language_available": bool(record.language_instruction),
        }
        samples.append(sample)
        for frame in prior_indices:
            frame_id = sample["sample_id"] + f"__f{frame:06d}"
            source_frames[frame_id] = {"schema": "rm-spd-source-frame-v1", "frame_id": frame_id, "sample_id": sample["sample_id"], "episode_id": record.episode_id, "task_id": record.task_id, "frame_index": frame, "video_path": str(record.front_video_path), "role": "history" if frame in history else "future" if frame in futures.values() else "temporal_neighbor"}
        alignment.append({"episode_id": record.episode_id, "numeric": audit, "video": video_meta, "anchor": anchor, "pass": True})
        for modality, path in (("parquet_numeric", record.parquet_path), ("front_rgb_video", record.front_video_path)):
            inherited_entry = hash_by_path.get(str(path))
            if inherited_entry is None:
                raise RuntimeError(f"missing inherited hash for frozen RM-SPD source: {path}")
            source_hashes.append({"episode_id": record.episode_id, "task_id": record.task_id, "modality": modality, "local_path": str(path), "bytes": int(inherited_entry["bytes"]), "sha256": str(inherited_entry["sha256"]), "hash_origin": "rm_series_robomind/source_hash_manifest.json"})

    splits = _split(selected)
    for sample in samples:
        sample["episode_split"] = splits["episode_held_out"][sample["episode_id"]]
        sample["task_split"] = splits["task_held_out"][sample["task_id"]]
    write_jsonl(OUT / "sample_manifest.jsonl", samples)
    write_jsonl(OUT / "source_frame_manifest.jsonl", [source_frames[key] for key in sorted(source_frames)])
    write_json(OUT / "source_hash_manifest.json", {"schema": "rm-spd-source-hash-manifest-v1", "files": source_hashes, "all_hashes_inherited": True})
    write_json(OUT / "source_alignment_audit.json", {"schema": "rm-spd-source-alignment-audit-v1", "episodes": alignment, "all_pass": all(item["pass"] for item in alignment), "samples": len(samples), "source_frames": len(source_frames)})
    write_json(OUT / "split_manifest.json", {"schema": "rm-spd-split-v1", **splits, "episode_split_counts": {name: list(splits["episode_held_out"].values()).count(name) for name in ("train", "val", "test")}, "task_split_counts": {name: list(splits["task_held_out"].values()).count(name) for name in ("train", "val", "test")}})
    docs = OUT / "docs"
    docs.mkdir(exist_ok=True)
    (docs / "rm_spd_plan.md").write_text("# RM-SPD plan\n\nS0 freezes exactly 100 local RoboMIND 2.0 Franka trajectories from 10 repeated tasks. SAM3 is an uncertain prior only: it is never a target, reference, predictive input, or human-derived label. The next permitted phase is the fixed eight-prompt proposal ensemble.\n")
    (docs / "rm_spd_results.md").write_text("# RM-SPD results\n\nS0 complete. No SAM3 proposal, future target, baseline, or learned method result exists yet.\n")
    (docs / "rm_spd_handoff.md").write_text("# RM-SPD handoff\n\nRead `preregistered_config.json` and verify its SHA-256 before running a phase. Only artifacts under this branch may be written. A benchmark decision is required before the proposed model.\n")
    append_log("S0_CONFIG_AND_SOURCES_FROZEN", samples=len(samples), source_frames=len(source_frames), tasks=len(tasks), config_sha256=sha256_file(config_path))
    return {"samples": len(samples), "source_frames": len(source_frames), "tasks": len(tasks), "config_sha256": sha256_file(config_path)}


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, sort_keys=True))
