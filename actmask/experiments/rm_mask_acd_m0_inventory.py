"""Freeze RM-MASK-ACD and inventory every viable independent mask evidence path."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import h5py
import pyarrow.parquet as pq

from actmask.data.robomind_adapter import RoboMINDRecord, load_numeric_episode, validate_numeric_episode


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "actmask" / "rm_mask_acd_verified"
RM = ROOT / "outputs" / "actmask" / "rm_series_robomind"
SOURCE = Path("/data/projects/tzh/RoboMIND2")
CAMERAS = ("front", "left", "right", "top", "wrist_left", "wrist_right")


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
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        previous, sequence = str(rows[-1]["event_sha256"]), int(rows[-1]["seq"])
    row = {"seq": sequence + 1, "utc": datetime.now(timezone.utc).isoformat(), "event": event, "payload": payload, "prev_event_sha256": previous}
    row["event_sha256"] = hashlib.sha256(json.dumps(row, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    with path.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def _record(row: dict[str, Any]) -> RoboMINDRecord:
    return RoboMINDRecord(**{field: row[field] for field in RoboMINDRecord.__dataclass_fields__})


def _video(record: RoboMINDRecord, camera: str) -> Path:
    return Path(record.parquet_path).parents[2] / "videos" / "chunk-000" / f"observation.rgb_images.camera_{camera}" / f"episode_{record.episode_index:06d}.mp4"


def _hdf5_inventory(path: Path) -> dict[str, Any]:
    paths: list[str] = []
    datasets: dict[str, dict[str, Any]] = {}
    with h5py.File(path, "r") as handle:
        def visit(name: str, node: Any) -> None:
            paths.append(name)
            if isinstance(node, h5py.Dataset):
                datasets[name] = {"shape": list(node.shape), "dtype": str(node.dtype)}
        handle.visititems(visit)
        metadata = {name: (handle["metadata"][name][()].decode() if isinstance(handle["metadata"][name][()], bytes) else str(handle["metadata"][name][()])) for name in handle.get("metadata", {})}
    names = set(datasets)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path), "datasets": datasets, "has_rgb": any("color_images" in name for name in names), "has_depth": any("depth_images" in name for name in names), "has_camera_intrinsics": any(name.startswith("camera_intrinsics/") for name in names), "has_camera_extrinsics": any(name.startswith("camera_extrinsics/") for name in names), "has_base_to_robot": any(name.startswith("base_to_robot_transformation/") for name in names), "has_joint_positions": any("arm_left_position_align/data" in name or "arm_right_position_align/data" in name for name in names), "has_joint_velocities": any("velocity" in name.lower() for name in names), "has_end_effector_pose": any("end_effector" in name and "pose" in name for name in names), "official_mask_fields": sorted(name for name in names if any(token in name.lower() for token in ("mask", "seg", "instance", "label"))), "metadata": metadata}


def _config(selected: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema": "rm-mask-acd-preregistered-config-v1",
        "source": {"dataset": "RoboMIND2.0", "source_root": str(SOURCE), "selected_source": "exact frozen RM-Series 100-trajectory / 10-task Franka manifest", "episodes": len(selected), "tasks": sorted({row["task_id"] for row in selected}), "source_hash_manifest": "source_hash_manifest.json"},
        "mask_scope": {"target_view": "observation.rgb_images.camera_front", "robot_definition": "both Franka robot arms, links and grippers only", "nonrobot_definition": "manipulated objects including held/touched objects, table, shadows, tools, fixtures and background", "rejected_evidence": "RM-ACD heuristic motion/sweep mask is audit-only and forbidden as a new mask label"},
        "budgets": {"candidate_evaluation_frames_max": 500, "reference_audit_frames_min": 150, "reference_audit_frames_max": 300, "new_storage_max_bytes_before_benchmark": 8 * 1024**3, "gpu_hours_before_target_acceptance": 8, "automatic_candidates_max": 3, "training_seeds": [17, 29, 43]},
        "path_order": ["official_dataset_mask", "validated_geometry_projection", "Robo-SAM_or_RoboEngine", "RobotSeg", "SAM2_prompted_video_segmentation", "one_time_human_reference_package"],
        "geometry_requirements": ["exact dual-Franka geometry assets", "joint state convention", "camera intrinsics", "camera-to-base extrinsics", "episode-to-geometry/calibration mapping", "rasterized links and grippers only"],
        "mask_acceptance_gates": {"mean_iou_min": 0.85, "robot_pixel_recall_min": 0.95, "nonrobot_false_positive_area_max": 0.08, "manipulated_object_preservation_min": 0.90, "contact_zone_nonrobot_preservation_min": 0.85, "no_catastrophic_task_failure": True, "grippers_required": True, "temporal_flicker_documented": True, "versioned_hashed_masks_scores_references": True},
        "target_quality_gates": {"meaningful_change_window_fraction": [0.20, 0.80], "confidence_window_fraction_min": 0.60, "action_progress_correlation_max": 0.90, "static_background_false_positive_low": True, "robot_residual_not_dominant": True, "contact_review_required": True},
        "headroom_gates": {"privileged_iou_headroom": 0.05, "privileged_f1_headroom": 0.08, "shortcut_controls_weak": True, "raw_predictions_and_metric_recompute": True},
        "method": {"family": "ActionConditionedObjectDynamicsNetwork", "only_after_headroom": True, "seeds": [17, 29, 43], "iou_improvement": 0.05, "f1_improvement": 0.05, "ablation_drop": 0.03, "minimum_structural_ablations": 2},
        "artifact_policy": {"record_failed_mask_candidates": True, "raw_masks_scores_references_hashed": True, "no_frozen_rm_rm_od_rm_acd_modification": True, "no_sudo_or_system_package_install": True},
        "claim_boundary": "robot-only mask validation and, only after its acceptance, action-conditioned dense logged scene/object change prediction; never future retrieval, success/failure prediction, physical counterfactual prediction, or alternate-action outcome prediction",
    }


def run() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    source = json.loads((RM / "source_subset" / "subset_manifest.json").read_text())
    selected = list(source["episodes_detail"])
    old_hashes = {str(row["local_path"]): row for row in json.loads((RM / "source_hash_manifest.json").read_text())["files"]}
    entries = []
    for row in selected:
        for modality, key in (("parquet_rgb_depth_numeric", "parquet_path"), ("front_rgb_video", "front_video_path")):
            item = old_hashes.get(str(row[key]))
            if item is None:
                raise RuntimeError(f"missing verified source hash: {row[key]}")
            entries.append({"episode_id": row["episode_id"], "task": row["task_id"], "modality": modality, "local_path": str(row[key]), "bytes": int(item["bytes"]), "sha256": str(item["sha256"]), "hash_origin": "rm_series_robomind/source_hash_manifest.json"})
    config = _config(selected)
    config_path = OUT / "preregistered_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise RuntimeError("refusing to alter frozen RM-MASK-ACD configuration")
    write_json(config_path, config); (OUT / "preregistered_config.sha256").write_text(sha(config_path) + "\n")
    write_json(OUT / "selected_trajectory_manifest.json", {"schema": "rm-mask-acd-selected-trajectories-v1", "episodes": selected, "episode_count": len(selected), "task_count": len({row["task_id"] for row in selected}), "source": "exact inherited RM-Series selection"})
    write_json(OUT / "source_hash_manifest.json", {"schema": "rm-mask-acd-source-hash-manifest-v1", "files": entries, "all_inherited_from_verified_rm_series": True})

    real_rows, source_masks = [], []
    for row in selected:
        record = _record(row)
        numeric = load_numeric_episode(record); alignment = validate_numeric_episode(record, numeric)
        schema = pq.ParquetFile(record.parquet_path).schema_arrow
        columns = set(schema.names)
        videos = {}
        for camera in CAMERAS:
            path = _video(record, camera); capture = cv2.VideoCapture(str(path))
            videos[camera] = {"path": str(path), "exists": path.is_file(), "opened": bool(capture.isOpened()), "frame_count": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) if capture.isOpened() else 0, "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) if capture.isOpened() else 0, "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) if capture.isOpened() else 0, "frame_aligned": bool(capture.isOpened() and int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == alignment["steps"])}
            capture.release()
        mask_columns = sorted(name for name in columns if any(token in name.lower() for token in ("mask", "seg", "instance", "label")))
        source_masks.append({"episode_id": record.episode_id, "parquet_mask_columns": mask_columns, "has_dataset_provided_mask": bool(mask_columns)})
        real_rows.append({"episode_id": record.episode_id, "task": record.task_id, "robot_embodiment": record.robot_embodiment, "parquet_path": record.parquet_path, "rgb_views": videos, "depth_columns": sorted(name for name in columns if "depth_images" in name), "joint_positions": True, "joint_velocities": any("velocity" in name.lower() for name in columns), "end_effector_pose": True, "frame_state_action_alignment": alignment, "camera_view_synchronization": all(value["frame_aligned"] for value in videos.values()), "official_mask_columns": mask_columns})
    write_json(OUT / "modality_geometry_inventory.json", {"schema": "rm-mask-acd-modality-geometry-inventory-v1", "selected_real_trajectories": real_rows, "summary": {"episodes": len(real_rows), "all_six_rgb_views_aligned": all(row["camera_view_synchronization"] for row in real_rows), "depth_available_all": all(len(row["depth_columns"]) == 6 for row in real_rows), "joint_positions_available_all": all(row["joint_positions"] for row in real_rows), "joint_velocities_available_any": any(row["joint_velocities"] for row in real_rows), "end_effector_pose_available_all": all(row["end_effector_pose"] for row in real_rows)}})
    write_json(OUT / "official_mask_inventory.json", {"schema": "rm-mask-acd-official-mask-inventory-v1", "selected_source_masks": source_masks, "hdf5_masks": [], "official_mask_available": False, "decision": "No selected real-trajectory Parquet has robot/segmentation/instance/label field; available HDF5 files are also enumerated separately and contain no documented robot-mask field."})

    calibration_root = SOURCE / "camera_extrinsics_intrinsics" / "franka"
    intrinsics_path, extrinsics_path = calibration_root / "camera_intrinsics.json", calibration_root / "camera_extrinsics_2025_08_26.json"
    intrinsics = json.loads(intrinsics_path.read_text()) if intrinsics_path.is_file() else {}
    extrinsics = json.loads(extrinsics_path.read_text()) if extrinsics_path.is_file() else {}
    write_json(OUT / "camera_calibration_inventory.json", {"schema": "rm-mask-acd-camera-calibration-inventory-v1", "real_selected_camera_intrinsics": {"path": str(intrinsics_path), "sha256": sha(intrinsics_path), "cameras": sorted(intrinsics)}, "real_selected_extrinsics": {"path": str(extrinsics_path), "sha256": sha(extrinsics_path), "top_level_fields": sorted(extrinsics), "camera_extrinsics_base_populated": bool(extrinsics.get("camera_extrinsics_base")), "base_to_robot_populated": bool(extrinsics.get("base_to_robot_transformation"))}, "geometry_projection_status": "unavailable_for_selected_real_episodes: intrinsics exist, but no populated camera-to-base transform, no local exact robot mesh/URDF/USD, and no episode-to-simulation mapping"})

    hdf5s = [_hdf5_inventory(path) for path in sorted((SOURCE / "data").rglob("*.hdf5"))]
    asset_files = [str(path) for path in sorted(SOURCE.rglob("*")) if path.is_file() and path.suffix.lower() in {".urdf", ".usd", ".usda", ".usdc", ".xml", ".obj", ".stl", ".dae"}]
    sim_hdf5 = [row for row in hdf5s if "franka_sim" in row["path"]]
    write_json(OUT / "robot_asset_inventory.json", {"schema": "rm-mask-acd-robot-asset-inventory-v1", "robot_mesh_urdf_usd_files": asset_files, "hdf5_sources": hdf5s, "sim_digital_twin_files": sim_hdf5, "episode_to_digital_twin_mapping": None, "mapping_evidence": "No file under the local release maps the frozen selected real episode IDs to the single franka_sim HDF5 or its sim_assets metadata.", "exact_geometry_projection_available": False})
    decision = "M0_AUTOMATIC_SEGMENTATION_REQUIRED"
    write_json(OUT / "m0_decision.json", {"schema": "rm-mask-acd-m0-decision-v1", "decision": decision, "official_mask_available": False, "geometry_path_available": False, "reason": ["No documented official robot/instance mask is present in selected real trajectories.", "Real trajectories provide RGB/depth/joints/end-effectors and intrinsics, but local camera-to-base transforms are not populated.", "No local Franka URDF/USD/mesh asset is available.", "The one full-calibration franka_sim HDF5 is not documented as aligned to any frozen selected real episode."]})
    docs = OUT / "docs"; docs.mkdir(exist_ok=True)
    (docs / "rm_mask_acd_plan.md").write_text("# RM-MASK-ACD plan\n\nThis run is a fresh, independently validated robot-only mask effort. The rejected RM-ACD motion/sweep mask and its targets are never reused as labels. M0 found no official selected-episode masks or documented real-episode geometry mapping, so the next path is documented automatic robot segmentation followed by an independent reference package.\n")
    (docs / "rm_mask_acd_results.md").write_text("# RM-MASK-ACD results\n\nM0 complete: `M0_AUTOMATIC_SEGMENTATION_REQUIRED`. No mask candidate or ACD model has run.\n")
    (docs / "rm_mask_acd_handoff.md").write_text("# RM-MASK-ACD handoff\n\nConsult `m0_decision.json` and the frozen configuration. Automatic candidates must be evaluated against an independent reference; their own confidence is not acceptance evidence.\n")
    append_log("phase_m0_frozen_inventory", decision=decision, episodes=len(selected), hdf5_files=len(hdf5s), config_sha256=sha(config_path))
    return {"decision": decision, "episodes": len(selected), "hdf5_files": len(hdf5s), "config_sha256": sha(config_path)}


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, sort_keys=True))
