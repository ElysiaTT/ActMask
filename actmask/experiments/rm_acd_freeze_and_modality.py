"""Freeze RM-ACD and audit actual RoboMIND visual modalities before target work."""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyarrow.parquet as pq

from actmask.data.robomind_adapter import RoboMINDRecord, load_numeric_episode, validate_numeric_episode


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "actmask" / "rm_acd_action_conditioned_change"
RM = ROOT / "outputs" / "actmask" / "rm_series_robomind"
SOURCE = Path("/data/projects/tzh/RoboMIND2")
CAMERAS = ("front", "left", "right", "top", "wrist_left", "wrist_right")
DEPTH_COLUMNS = tuple(f"observation.depth_images.camera_{name}" for name in CAMERAS)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_json(value))


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def append_log(event: str, **payload: Any) -> None:
    path = OUT / "run_log.jsonl"
    previous, sequence = "", 0
    if path.is_file() and path.stat().st_size:
        lines = [json.loads(line) for line in path.read_text().splitlines() if line]
        previous, sequence = str(lines[-1]["event_sha256"]), int(lines[-1]["seq"])
    row = {"seq": sequence + 1, "utc": datetime.now(timezone.utc).isoformat(), "event": event, "payload": payload, "prev_event_sha256": previous}
    row["event_sha256"] = hashlib.sha256(json.dumps(row, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    with path.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def _record(row: dict[str, Any]) -> RoboMINDRecord:
    return RoboMINDRecord(**{key: row[key] for key in RoboMINDRecord.__dataclass_fields__})


def _video_path(record: RoboMINDRecord, camera: str) -> Path:
    return Path(record.parquet_path).parents[2] / "videos" / "chunk-000" / f"observation.rgb_images.camera_{camera}" / f"episode_{record.episode_index:06d}.mp4"


def _video_meta(path: Path) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    opened = bool(capture.isOpened())
    result = {
        "available": path.is_file(),
        "opened": opened,
        "path": str(path),
        "frame_count": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) if opened else 0,
        "fps": float(capture.get(cv2.CAP_PROP_FPS)) if opened else 0.0,
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) if opened else 0,
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) if opened else 0,
    }
    capture.release()
    return result


def _read_frame(path: Path, frame: int) -> np.ndarray | None:
    capture = cv2.VideoCapture(str(path))
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ok, image = capture.read()
    capture.release()
    return image if ok else None


def _border_motion(path: Path, frames: int) -> dict[str, Any]:
    """Camera-status diagnostic from a conservative border feature set only."""
    if frames < 4:
        return {"status": "invalid", "samples": 0, "median_border_displacement_px": None}
    shifts: list[float] = []
    for step in (max(1, frames // 4), max(1, frames // 2), max(1, 3 * frames // 4)):
        first, second = _read_frame(path, step), _read_frame(path, min(step + 1, frames - 1))
        if first is None or second is None:
            continue
        gray1, gray2 = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY), cv2.cvtColor(second, cv2.COLOR_BGR2GRAY)
        mask = np.zeros_like(gray1)
        band = max(24, min(gray1.shape) // 10)
        mask[:band] = 255; mask[-band:] = 255; mask[:, :band] = 255; mask[:, -band:] = 255
        points = cv2.goodFeaturesToTrack(gray1, maxCorners=120, qualityLevel=0.01, minDistance=6, mask=mask)
        if points is None:
            continue
        tracked, status, _ = cv2.calcOpticalFlowPyrLK(gray1, gray2, points, None)
        good = (status.reshape(-1) > 0) if status is not None else np.zeros(len(points), dtype=bool)
        if tracked is not None and int(good.sum()) >= 8:
            delta = tracked.reshape(-1, 2)[good] - points.reshape(-1, 2)[good]
            shifts.append(float(np.median(np.linalg.norm(delta, axis=1))))
    median = float(np.median(shifts)) if shifts else None
    return {
        "status": "fixed_likely" if median is not None and median <= 1.5 else "moving_or_unverified",
        "samples": len(shifts),
        "median_border_displacement_px": median,
        "method": "adjacent-frame border Lucas-Kanade diagnostic; it is an audit, not geometric calibration",
    }


def _depth_schema_and_sample(record: RoboMINDRecord, columns: set[str]) -> dict[str, Any]:
    result: dict[str, Any] = {"all_depth_columns_declared": all(name in columns for name in DEPTH_COLUMNS), "front_depth_column": "observation.depth_images.camera_front", "front_sample": None}
    column = result["front_depth_column"]
    if column not in columns:
        return result
    table = pq.read_table(record.parquet_path, columns=[column])
    probes = [0, len(table) // 2, len(table) - 1]
    samples = []
    for index in probes:
        item = table.column(column)[index].as_py()
        raw = item.get("bytes") if item else None
        image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED) if raw else None
        samples.append({"frame": int(index), "bytes": len(raw) if raw else 0, "decoded": image is not None, "shape": list(image.shape) if image is not None else None, "dtype": str(image.dtype) if image is not None else None, "value_range": [int(image.min()), int(image.max())] if image is not None else None, "zero_fraction": float((image == 0).mean()) if image is not None else None})
    result["front_sample"] = samples
    result["depth_metric_units_verified"] = False
    result["depth_use_decision"] = "Depth bytes decode as uint8 PNG but no documented metric scale/registration contract is present in this local export; do not claim metric depth, 3D displacement, or scene flow."
    return result


def _configuration(selected: list[dict[str, Any]], source_entries: list[dict[str, Any]]) -> dict[str, Any]:
    tasks = sorted({row["task_id"] for row in selected})
    return {
        "schema": "rm-acd-preregistered-config-v1",
        "source": {"dataset": "RoboMIND2.0", "source_root": str(SOURCE), "selected_episodes": 100, "selected_tasks": tasks, "selection": "exact frozen RM-Series 100-trajectory manifest; no unrelated task or embodiment", "source_hash_manifest": "source_hash_manifest.json", "hashed_source_files": len(source_entries)},
        "views": {"target_view": "observation.rgb_images.camera_front", "all_views_audited": [f"observation.rgb_images.camera_{name}" for name in CAMERAS], "multi_view_target": False, "reason": "Metric depth scale and a stable common camera/world extrinsic contract are not verified in this local snapshot; target is conservative single-view RGB with future depth excluded from model inputs."},
        "modalities": {"pre_anchor_fair_input": ["front RGB history", "robot state history", "action chunk"], "depth": "available as uint8 PNG in source Parquet but not used as metric geometry or fair input in pilot", "future_rgb_depth_state_forbidden_as_fair_input": True},
        "temporal": {"source_fps_hz": 30.0, "history_steps": 4, "action_steps": 4, "anchors_per_episode": 8, "anchor_fractions": [0.12, 0.22, 0.32, 0.42, 0.52, 0.62, 0.72, 0.82], "anchor_policy": "for each episode, select exactly eight unique integer anchors uniformly over the legal inclusive interval [history_steps, length-action_steps-long_horizon_steps-1]; nominal fractions are descriptive only and may not create a collision", "horizons": {"short": {"steps": 8, "seconds": 0.267}, "medium": {"steps": 15, "seconds": 0.5}, "long": {"steps": 30, "seconds": 1.0}}},
        "target": {"construction": "single-view RGB dense forward/backward optical-flow and patch-residual target after conservative robot exclusion; no 3D/scene-flow claim", "affected_region": "confident non-robot pixels with temporally consistent displacement or luminance/chromatic residual above frozen thresholds and surviving connected-component filtering", "target_resolution": [160, 120], "flow_algorithm": "OpenCV Farneback forward/backward consistency", "flow_magnitude_threshold_px_at_target_resolution": 0.75, "residual_threshold_0_1": 0.10, "minimum_component_pixels": 12, "meaningful_affected_area_fraction_min": 0.003, "robot_mask": "conservative motion/appearance heuristic audited before target acceptance", "confidence": "forward/backward flow consistency, image-gradient support, nonrobot mask, and component support"},
        "splits": {"episode_held_out": "6/2/2 episodes per task", "task_held_out": "6/2/2 task groups", "all_fragments_from_one_episode_stay_in_one_partition": True, "no_duplicate_source_frame_across_splits": True},
        "baselines": ["constant_no_change", "mean_training_mask", "action_magnitude_only", "action_trajectory_only", "robot_state_only", "state_action", "episode_progress_diagnostic", "task_only", "current_rgb_only", "previous_optical_flow_extrapolation", "global_frame_difference", "camera_background_diagnostic", "current_frame_spatial_decoder", "rgb_history_convgru", "rgb_history_action_convgru", "global_visual_action_mlp", "patch_visual_action_mlp", "visual_state_action_temporal", "video_feature_prediction", "flow_or_feature_residual_prediction"],
        "acceptance_gates": {"meaningful_change_fraction_min": 0.20, "meaningful_change_fraction_max": 0.80, "confidence_window_fraction_min": 0.60, "static_background_false_positive_area_max": 0.05, "robot_residual_fraction_max": 0.50, "action_or_progress_abs_correlation_max": 0.90, "target_diversity_min_tasks": 10, "standard_headroom_mask_iou": 0.05, "standard_headroom_f1": 0.08, "future_diagnostic_margin": 0.05, "task_held_out_above_trivial": True, "raw_prediction_recompute_required": True},
        "method_gates": {"family": "ActionConditionedObjectDynamicsNetwork", "seeds": [17, 29, 43], "mask_iou_improvement": 0.05, "f1_improvement": 0.05, "motion_relative_error_improvement": 0.10, "ablation_drop": 0.03, "minimum_structural_ablations": 2, "one_implementation_repair_only": True},
        "resource_budget": {"derived_data_max_bytes": 10 * 1024**3, "gpu_hours_pre_benchmark_max": 8, "expansion_requires_target_valid": True, "pilot_episodes": 100},
        "artifact_policy": {"raw_predictions": True, "per_sample_metrics": True, "metric_recomputation": True, "source_hashes": True, "configs_checkpoints_logs": True, "no_frozen_rm_or_rm_od_modification": True},
        "claim_boundary": "action-conditioned dense real-world scene/object change prediction from logged RoboMIND trajectories only; no success/failure, retrieval, physical counterfactual, alternate-action outcome, metric 3D scene flow, or segmentation-ground-truth claim",
    }


def run() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    prior = json.loads((RM / "source_subset" / "subset_manifest.json").read_text())
    selected = list(prior["episodes_detail"])
    inherited = {str(row["local_path"]): row for row in json.loads((RM / "source_hash_manifest.json").read_text())["files"]}
    source_entries = []
    for row in selected:
        for modality, key in (("parquet_rgb_depth_numeric", "parquet_path"), ("front_rgb_video", "front_video_path")):
            source = str(row[key]); old = inherited.get(source)
            if old is None:
                raise RuntimeError(f"missing inherited source hash: {source}")
            source_entries.append({"episode_id": row["episode_id"], "task": row["task_id"], "modality": modality, "local_path": source, "bytes": int(old["bytes"]), "sha256": str(old["sha256"]), "hash_origin": "rm_series_robomind/source_hash_manifest.json"})
    config = _configuration(selected, source_entries)
    config_path = OUT / "preregistered_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise RuntimeError("refusing to alter frozen RM-ACD configuration")
    write_json(config_path, config)
    (OUT / "preregistered_config.sha256").write_text(sha(config_path) + "\n")
    write_json(OUT / "selected_trajectory_manifest.json", {"schema": "rm-acd-selected-trajectories-v1", "source": "exact frozen RM-Series balanced source selection", "episodes": selected, "episode_count": len(selected), "task_count": len({row["task_id"] for row in selected})})
    write_json(OUT / "source_hash_manifest.json", {"schema": "rm-acd-source-hash-manifest-v1", "files": source_entries, "all_hashes_inherited_from_verified_rm_series": True})

    per_episode, per_task_view, alignment_rows = [], defaultdict(list), []
    representative: dict[str, RoboMINDRecord] = {}
    for row in selected:
        record = _record(row)
        numeric = load_numeric_episode(record)
        numeric_audit = validate_numeric_episode(record, numeric)
        schema = pq.ParquetFile(record.parquet_path).schema_arrow
        videos = {camera: _video_meta(_video_path(record, camera)) for camera in CAMERAS}
        for camera, value in videos.items():
            value["aligned_to_numeric_frames"] = bool(value["opened"] and value["frame_count"] == numeric_audit["steps"])
            per_task_view[(record.task_id, camera)].append(value)
        if record.task_id not in representative:
            representative[record.task_id] = record
        per_episode.append({"episode_id": record.episode_id, "task": record.task_id, "robot_embodiment": record.robot_embodiment, "numeric_alignment": numeric_audit, "rgb_views": videos, "depth_columns_present": sorted(name for name in DEPTH_COLUMNS if name in schema.names)})
        alignment_rows.append({"episode_id": record.episode_id, "numeric_pass": numeric_audit["pass"], "rgb_views_all_aligned": all(value["aligned_to_numeric_frames"] for value in videos.values()), "front_frames": videos["front"]["frame_count"], "numeric_steps": numeric_audit["steps"]})
    depth_samples = {task: _depth_schema_and_sample(record, set(pq.ParquetFile(record.parquet_path).schema_arrow.names)) for task, record in representative.items()}
    camera_motion = {}
    for task, record in representative.items():
        camera_motion[task] = {camera: _border_motion(_video_path(record, camera), _video_meta(_video_path(record, camera))["frame_count"]) for camera in CAMERAS}
    task_view_summary = []
    for (task, camera), values in sorted(per_task_view.items()):
        motion = camera_motion[task][camera]
        task_view_summary.append({"task": task, "camera": f"camera_{camera}", "episodes": len(values), "rgb_available_all": all(value["available"] and value["opened"] for value in values), "resolution_set": sorted({(value["width"], value["height"]) for value in values}), "fps_set": sorted({value["fps"] for value in values}), "numeric_frame_aligned_all": all(value["aligned_to_numeric_frames"] for value in values), "camera_status": motion["status"], "median_border_displacement_px": motion["median_border_displacement_px"], "camera_status_evidence": motion["method"]})
    modality = {"schema": "rm-acd-modality-inventory-v1", "dataset": "RoboMIND2.0", "episodes": per_episode, "task_view_summary": task_view_summary, "depth_samples_per_task": depth_samples, "summary": {"episodes": len(per_episode), "tasks": len(representative), "all_numeric_alignment_pass": all(row["numeric_pass"] for row in alignment_rows), "all_six_rgb_views_frame_aligned": all(row["rgb_views_all_aligned"] for row in alignment_rows), "depth_available_in_all_selected_parquets": all(len(row["depth_columns_present"]) == len(DEPTH_COLUMNS) for row in per_episode), "depth_metric_units_verified": False, "target_modality_choice": "single-view front RGB; depth is audit-only because metric scale/registration is unverified"}}
    write_json(OUT / "modality_inventory.json", modality)
    camera_audit = {"schema": "rm-acd-camera-alignment-audit-v1", "rows": alignment_rows, "all_numeric_and_rgb_alignment_pass": all(row["numeric_pass"] and row["rgb_views_all_aligned"] for row in alignment_rows), "target_view": "camera_front", "target_view_fixed_likely_tasks": sum(row["camera_status"] == "fixed_likely" for row in task_view_summary if row["camera"] == "camera_front"), "target_view_moving_or_unverified_tasks": sum(row["camera_status"] != "fixed_likely" for row in task_view_summary if row["camera"] == "camera_front"), "task_view_camera_status": task_view_summary}
    write_json(OUT / "camera_alignment_audit.json", camera_audit)
    docs = OUT / "docs"; docs.mkdir(exist_ok=True)
    (docs / "rm_acd_plan.md").write_text("# RM-ACD plan\n\nThe frozen pilot uses the exact 100-trajectory RoboMIND2 Franka manifest. It will predict future non-robot 2D scene/object change maps from past front RGB, state, and action only. RGB-D bytes and six synchronized RGB views are audited, but the local export does not establish metric depth/world registration, so no 3D or scene-flow claim is permitted. Robot masking and target feasibility must pass before any learned baseline.\n")
    (docs / "rm_acd_results.md").write_text("# RM-ACD results\n\nPending robot-mask and future-change target construction. No model has been trained.\n")
    (docs / "rm_acd_handoff.md").write_text("# RM-ACD handoff\n\nRead `preregistered_config.json` first. The frozen target is direct supervised future non-robot scene change, not candidate retrieval or a counterfactual task.\n")
    report = ["# Selected view and modality report", "", "Target choice: `camera_front` single-view RGB. All six RGB streams are recorded in the inventory; target construction will use only front RGB, past state and the action chunk.", "", "Depth is present as embedded uint8 PNG but lacks a verified metric scale/registration contract; it is not used to claim 3D displacement and is not a fair input.", "", "| task | front status | resolution | fps | border-motion status |", "| --- | --- | --- | --- | --- |"]
    for row in task_view_summary:
        if row["camera"] == "camera_front":
            report.append(f"| {row['task']} | {row['camera_status']} | {row['resolution_set']} | {row['fps_set']} | {row['median_border_displacement_px']} |")
    (OUT / "selected_view_report.md").write_text("\n".join(report) + "\n")
    append_log("phase_a0_a1_frozen_and_audited", config_sha256=sha(config_path), episodes=len(selected), tasks=len(representative), alignment_pass=camera_audit["all_numeric_and_rgb_alignment_pass"], target_view_fixed_likely_tasks=camera_audit["target_view_fixed_likely_tasks"])
    return {"config_sha256": sha(config_path), "episodes": len(selected), "tasks": len(representative), "alignment_pass": camera_audit["all_numeric_and_rgb_alignment_pass"], "front_fixed_likely_tasks": camera_audit["target_view_fixed_likely_tasks"]}


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, sort_keys=True))
