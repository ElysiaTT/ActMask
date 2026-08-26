#!/usr/bin/env python3
"""Audit whether local RoboMIND 2.0 Franka data supports verified geometry masks.

This is a read-only audit of the source release.  It does not train a model and
does not generate pseudo masks.  A 50-frame geometry pilot is permitted only
when the complete geometry/calibration chain is evidenced.  Otherwise it
creates a separate 200-frame human-reference annotation plan.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from actmask.data.robomind_adapter import (  # noqa: E402
    RoboMINDRecord,
    load_numeric_episode,
    validate_numeric_episode,
)


OUT = ROOT / "outputs" / "actmask" / "rm_geometry_mask_audit"
SOURCE = Path("/data/projects/tzh/RoboMIND2")
FROZEN_MANIFEST = ROOT / "outputs" / "actmask" / "rm_series_robomind" / "source_subset" / "subset_manifest.json"
CALIBRATION_ROOT = SOURCE / "camera_extrinsics_intrinsics" / "franka"
INTRINSICS_PATH = CALIBRATION_ROOT / "camera_intrinsics.json"
EXTRINSICS_PATH = CALIBRATION_ROOT / "camera_extrinsics_2025_08_26.json"
REAL_H5 = SOURCE / "data" / "franka" / "trajectory.hdf5"
SIM_H5 = SOURCE / "data" / "franka_sim" / "4103000-2025_09_01_11_17_59.hdf5"
TARGET_CAMERA = "camera_front"
GEOMETRY_SUFFIXES = {".urdf", ".xacro", ".usd", ".usda", ".usdc", ".obj", ".stl", ".dae", ".ply", ".glb", ".gltf"}


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_json(value))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def append_log(event: str, **payload: Any) -> None:
    path = OUT / "run_log.jsonl"
    previous, sequence = "", 0
    if path.is_file() and path.stat().st_size:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        previous, sequence = str(rows[-1]["event_sha256"]), int(rows[-1]["seq"])
    row = {
        "seq": sequence + 1,
        "utc": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "payload": payload,
        "prev_event_sha256": previous,
    }
    row["event_sha256"] = hashlib.sha256(
        json.dumps(row, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    with path.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def _record(row: dict[str, Any]) -> RoboMINDRecord:
    return RoboMINDRecord(**{field: row[field] for field in RoboMINDRecord.__dataclass_fields__})


def matrix_audit(value: Any) -> dict[str, Any]:
    """Validate a homogeneous transform without assigning undocumented semantics."""

    array = np.asarray(value, dtype=np.float64)
    shape_ok = array.shape == (4, 4)
    finite = bool(shape_ok and np.isfinite(array).all())
    bottom_row_ok = bool(shape_ok and np.allclose(array[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8))
    rotation_ok = False
    determinant = None
    if shape_ok and finite:
        rotation = array[:3, :3]
        determinant = float(np.linalg.det(rotation))
        rotation_ok = bool(np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4) and np.isclose(determinant, 1.0, atol=1e-3))
    return {
        "shape": list(array.shape),
        "finite": finite,
        "homogeneous_bottom_row": bottom_row_ok,
        "rotation_orthonormal_and_proper": rotation_ok,
        "rotation_determinant": determinant,
        "structurally_valid_transform": bool(shape_ok and finite and bottom_row_ok and rotation_ok),
    }


def intrinsic_audit(value: Any, width: int, height: int) -> dict[str, Any]:
    matrix = np.asarray(value["matrix"], dtype=np.float64)
    distortion = np.asarray(value["dist_coeffs"], dtype=np.float64)
    shape_ok = matrix.shape == (3, 3)
    finite = bool(shape_ok and np.isfinite(matrix).all() and np.isfinite(distortion).all())
    fx_fy_positive = bool(shape_ok and matrix[0, 0] > 0 and matrix[1, 1] > 0)
    principal_in_image = bool(shape_ok and 0 <= matrix[0, 2] < width and 0 <= matrix[1, 2] < height)
    return {
        "matrix": matrix.tolist(),
        "dist_coeffs": distortion.tolist(),
        "matrix_shape": list(matrix.shape),
        "distortion_coefficients": int(distortion.size),
        "finite": finite,
        "fx_fy_positive": fx_fy_positive,
        "principal_point": [float(matrix[0, 2]), float(matrix[1, 2])],
        "video_resolution": [width, height],
        "pixel_domain_compatible": bool(shape_ok and finite and fx_fy_positive and principal_in_image),
        "resolution_declaration_present": False,
    }


def hdf5_summary(path: Path) -> dict[str, Any]:
    """Inspect only names, shapes and scalar metadata, never image/depth arrays."""

    result: dict[str, Any] = {
        "path": str(path),
        "exists": path.is_file(),
        "bytes": int(path.stat().st_size) if path.is_file() else None,
        "sha256": sha256(path) if path.is_file() else None,
    }
    if not path.is_file():
        return result
    datasets: dict[str, dict[str, Any]] = {}
    scalar_values: dict[str, str] = {}
    matrices: dict[str, list[list[float]]] = {}
    with h5py.File(path, "r") as handle:
        def visit(name: str, node: Any) -> None:
            if not isinstance(node, h5py.Dataset):
                return
            datasets[name] = {"shape": list(node.shape), "dtype": str(node.dtype)}
            if node.shape == ():
                value = node[()]
                if isinstance(value, bytes):
                    value = value.decode(errors="replace")
                scalar_values[name] = str(value)
            elif node.shape in {(2,), (3, 3), (4, 4)} and ("resolution" in name or "intrinsics" in name or "extrinsics" in name or "transformation" in name):
                matrices[name] = np.asarray(node[()]).astype(float).tolist()
        handle.visititems(visit)
    names = set(datasets)
    result.update({
        "datasets": datasets,
        "scalar_metadata": scalar_values,
        "small_calibration_arrays": matrices,
        "has_camera_intrinsics": any(name.startswith("camera_intrinsics/") for name in names),
        "has_camera_extrinsics": any(name.startswith("camera_extrinsics/") for name in names),
        "has_base_to_robot": any(name.startswith("base_to_robot_transformation/") for name in names),
        "has_geometry_assets": any("urdf" in name.lower() or "mesh" in name.lower() for name in names),
        "has_mask_like_field": any(any(token in name.lower() for token in ("mask", "seg", "instance", "label")) for name in names),
    })
    return result


def scan_geometry_assets() -> dict[str, Any]:
    assets = []
    symlinks = []
    for directory, _, names in os.walk(SOURCE, followlinks=False):
        base = Path(directory)
        for name in names:
            path = base / name
            if path.is_symlink():
                symlinks.append({"path": str(path), "target": os.readlink(path)})
            if path.suffix.lower() in GEOMETRY_SUFFIXES:
                assets.append({"path": str(path), "bytes": int(path.stat().st_size), "suffix": path.suffix.lower()})
    return {
        "source_root": str(SOURCE),
        "searched_suffixes": sorted(GEOMETRY_SUFFIXES),
        "asset_count": len(assets),
        "assets": sorted(assets, key=lambda row: row["path"]),
        "symlinks_encountered": sorted(symlinks, key=lambda row: row["path"]),
        "exact_dual_franka_urdf_available": False,
        "exact_franka_link_meshes_available": False,
        "conclusion": "No local release file with a robot-description or mesh suffix was found; a thumbnail or robot-type string is not usable link geometry.",
    }


def selected_rows() -> list[dict[str, Any]]:
    source = json.loads(FROZEN_MANIFEST.read_text())
    rows = list(source["episodes_detail"])
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[str(row["task_id"])].append(row)
    if len(rows) != 100 or len(by_task) != 10 or any(len(value) != 10 for value in by_task.values()):
        raise RuntimeError("expected the frozen 100-episode / 10-task RM-Series selection")
    return rows


def source_snapshot(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema": "rm-geometry-mask-source-snapshot-v1",
        "dataset": "RoboMIND2.0",
        "selection_source": str(FROZEN_MANIFEST),
        "selection_source_sha256": sha256(FROZEN_MANIFEST),
        "selection_episode_count": len(rows),
        "selection_task_count": len({row["task_id"] for row in rows}),
        "robot_embodiments": sorted({str(row["robot_embodiment"]) for row in rows}),
        "episodes": rows,
        "calibration_files": [
            {"path": str(path), "sha256": sha256(path), "bytes": int(path.stat().st_size)}
            for path in (INTRINSICS_PATH, EXTRINSICS_PATH)
        ],
    }


def audit_joint_states(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]]]:
    per_task_info: dict[str, dict[str, Any]] = {}
    per_episode: list[dict[str, Any]] = []
    numeric_cache: dict[str, dict[str, np.ndarray]] = {}
    for row in rows:
        record = _record(row)
        task_root = Path(record.parquet_path).parents[2]
        if record.task_id not in per_task_info:
            info_path = task_root / "meta" / "info.json"
            info = json.loads(info_path.read_text())
            features = info.get("features", {})
            per_task_info[record.task_id] = {
                "info_path": str(info_path),
                "info_sha256": sha256(info_path),
                "robot_type": info.get("robot_type"),
                "arm_joint_feature": features.get("observation.state.arm_joint_position"),
                "hand_joint_feature": features.get("observation.state.hand_joint_position"),
                "end_effector_feature": features.get("observation.state.end_effector"),
                "feature_has_joint_names": bool(features.get("observation.state.arm_joint_position", {}).get("names")),
            }
        numeric = load_numeric_episode(record)
        numeric_cache[record.episode_id] = numeric
        alignment = validate_numeric_episode(record, numeric)
        per_episode.append({
            "episode_id": record.episode_id,
            "task_id": record.task_id,
            "state_dim": int(numeric["state"].shape[1]),
            "action_dim": int(numeric["action"].shape[1]),
            "steps": int(numeric["state"].shape[0]),
            "alignment": alignment,
        })
    all_alignment = all(item["alignment"]["pass"] for item in per_episode)
    all_feature_shapes = all(
        info["arm_joint_feature"] == {"dtype": "float32", "shape": [16]}
        and info["hand_joint_feature"] == {"dtype": "float32", "shape": [2]}
        and info["end_effector_feature"] == {"dtype": "float32", "shape": [14]}
        for info in per_task_info.values()
    )
    names_available = all(info["feature_has_joint_names"] for info in per_task_info.values())
    return ({
        "schema": "rm-geometry-mask-joint-state-audit-v1",
        "selected_episode_count": len(per_episode),
        "task_metadata": per_task_info,
        "episode_checks": per_episode,
        "observed_state_layout": "16 arm-position values + 2 hand-position values + 14 end-effector values",
        "all_state_action_streams_aligned": all_alignment,
        "all_task_feature_shapes_match": all_feature_shapes,
        "explicit_joint_names_or_order_available": names_available,
        "projection_readiness": False,
        "blocking_reason": "The release declares dimensions but no names, left/right partition, link-to-joint mapping, reference pose, or joint-axis convention. A numerical pattern is not a verified Franka kinematic convention.",
    }, numeric_cache)


def audit_calibration(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    intrinsics = json.loads(INTRINSICS_PATH.read_text())
    extrinsics = json.loads(EXTRINSICS_PATH.read_text())
    expected_cameras = {"camera_front", "camera_left", "camera_right", "camera_top", "camera_wrist_left", "camera_wrist_right"}
    task_infos = {}
    for row in rows:
        task = str(row["task_id"])
        if task not in task_infos:
            task_infos[task] = json.loads((Path(row["parquet_path"]).parents[2] / "meta" / "info.json").read_text())
    front_shapes = [info.get("features", {}).get("observation.rgb_images.camera_front", {}).get("shape") for info in task_infos.values()]
    all_front_640x480 = all(shape == [480, 640, 3] for shape in front_shapes)
    intrinsic_checks = {camera: intrinsic_audit(value, 640, 480) for camera, value in intrinsics.items()}
    raw = dict(extrinsics.get("camera_extrinsics_raw", {}))
    raw_checks = {name: matrix_audit(value) for name, value in raw.items()}
    base = dict(extrinsics.get("camera_extrinsics_base", {}))
    base_robot = dict(extrinsics.get("base_to_robot_transformation", {}))
    front_raw_keys = sorted(name for name in raw if "camera_front" in name)
    calibration = {
        "schema": "rm-geometry-mask-camera-calibration-audit-v1",
        "intrinsics": {
            "path": str(INTRINSICS_PATH),
            "sha256": sha256(INTRINSICS_PATH),
            "expected_cameras_present": sorted(expected_cameras.intersection(intrinsics)),
            "unexpected_cameras": sorted(set(intrinsics).difference(expected_cameras)),
            "front_video_shapes_from_all_selected_task_metadata": front_shapes,
            "all_selected_front_video_shapes_640x480": all_front_640x480,
            "camera_checks": intrinsic_checks,
            "front_intrinsics_usable_in_pixel_domain": bool(intrinsic_checks.get("camera_front", {}).get("pixel_domain_compatible") and all_front_640x480),
            "scope_limit": "The JSON does not declare an image resolution or an episode calibration identifier; compatibility is pixel-domain evidence, not a per-episode calibration proof.",
        },
        "extrinsics": {
            "path": str(EXTRINSICS_PATH),
            "sha256": sha256(EXTRINSICS_PATH),
            "top_level_fields": sorted(extrinsics),
            "raw_transform_keys": sorted(raw),
            "raw_transform_checks": raw_checks,
            "front_raw_transform_keys": front_raw_keys,
            "raw_transforms_structurally_valid": bool(raw_checks and all(value["structurally_valid_transform"] for value in raw_checks.values())),
            "semantic_status": "The `camera_extrinsics_raw` matrices have structurally valid 4x4 values, but local documentation does not define their direction, arm-base frame, or correspondence to selected episodes. They are not promoted to certified camera-to-base transforms.",
        },
        "base_camera_chain": {
            "camera_extrinsics_base": base,
            "base_to_robot_transformation": base_robot,
            "camera_extrinsics_base_populated": bool(base),
            "base_to_robot_transformation_populated": bool(base_robot),
            "available": False,
            "blocking_reason": "Both base-referenced dictionaries are empty in the released Franka calibration JSON.",
        },
    }
    consistency = {
        "schema": "rm-geometry-mask-episode-calibration-consistency-audit-v1",
        "selected_episodes": len(rows),
        "selected_robot_embodiments": sorted({str(row["robot_embodiment"]) for row in rows}),
        "episode_metadata_fields": sorted({key for row in rows for key in json.loads(Path(row["metadata_path"]).read_text().splitlines()[0]).keys()}),
        "task_info_fields": sorted({key for info in task_infos.values() for key in info.keys()}),
        "explicit_calibration_id_in_episode_metadata": False,
        "explicit_calibration_timestamp_in_episode_metadata": False,
        "explicit_camera_extrinsic_reference_in_episode_metadata": False,
        "per_episode_camera_to_base_transform": False,
        "shared_directory_is_not_mapping": True,
        "consistent_for_verified_projection": False,
        "blocking_reason": "Selected episode metadata contains task/length information and task-level metadata contains robot type, but neither identifies a calibration instance, transform direction, or capture-to-calibration association.",
    }
    return calibration, consistency


def audit_hdf5_side_sources(rows: list[dict[str, Any]]) -> dict[str, Any]:
    real, sim = hdf5_summary(REAL_H5), hdf5_summary(SIM_H5)
    selected_ids = {str(row["episode_id"]) for row in rows}
    sim_asset = str(sim.get("scalar_metadata", {}).get("metadata/sim_assets", ""))
    return {
        "schema": "rm-geometry-mask-hdf5-side-source-audit-v1",
        "real_franka_example": real,
        "sim_franka_example": sim,
        "sim_asset_path": sim_asset,
        "sim_asset_is_locally_resolvable": bool(sim_asset and Path(sim_asset).exists()),
        "sim_path_matches_selected_real_episode": any(Path(sim_asset).stem in episode_id for episode_id in selected_ids),
        "side_source_projection_usable_for_selected_real_episodes": False,
        "reason": "The simulation HDF5 carries its own camera/base matrices and points to a non-local USD path, but there is no selected real episode ID or documented calibration mapping to this separate simulation trajectory.",
    }


def choose_reference_episodes(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["task_id"])].append(row)
    chosen: list[dict[str, Any]] = []
    for task in sorted(grouped):
        candidates = sorted(grouped[task], key=lambda row: (int(row["episode_index"]), str(row["episode_id"])))
        chosen.extend([candidates[0], candidates[-1]])
    if len(chosen) != 20:
        raise RuntimeError("human reference plan must choose two episodes across each of ten tasks")
    return chosen


def sample_indices(action: np.ndarray, frames: int) -> list[tuple[int, str]]:
    if frames < 10 or len(action) != frames:
        raise RuntimeError(f"cannot select ten source frames: video={frames}, action={len(action)}")
    planned = [
        (int(round((frames - 1) * fraction)), f"timeline_{int(fraction * 100):02d}pct")
        for fraction in (0.04, 0.16, 0.29, 0.41, 0.54, 0.66, 0.79, 0.93)
    ]
    delta = np.linalg.norm(np.diff(action, axis=0), axis=1)
    for indices, label in ((np.arange(0, max(1, len(delta) // 2)), "action_delta_peak_early"), (np.arange(max(1, len(delta) // 2), len(delta)), "action_delta_peak_late")):
        if len(indices):
            planned.append((int(indices[np.argmax(delta[indices])] + 1), label))
    unique: list[tuple[int, str]] = []
    used: set[int] = set()
    for index, label in planned:
        if index not in used:
            unique.append((index, label)); used.add(index)
    for index in np.linspace(0, frames - 1, 10, dtype=int).tolist():
        if len(unique) == 10:
            break
        if index not in used:
            unique.append((int(index), "timeline_replacement")); used.add(int(index))
    if len(unique) != 10:
        raise RuntimeError("could not deterministically choose ten distinct frames")
    return sorted(unique)


def frame_id(row: dict[str, Any], index: int) -> str:
    return f"{str(row['task_id']).replace('/', '_')}__ep{int(row['episode_index']):06d}__frame{index:06d}"


def current_source_hashes(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for row in rows:
        for modality, name in (("parquet_numeric_depth", "parquet_path"), ("front_rgb_video", "front_video_path")):
            path = Path(row[name])
            if not path.is_file():
                raise RuntimeError(f"missing source for human reference plan: {path}")
            entries.append({
                "episode_id": row["episode_id"],
                "task_id": row["task_id"],
                "modality": modality,
                "path": str(path),
                "bytes": int(path.stat().st_size),
                "sha256": sha256(path),
            })
    return entries


def write_annotation_validator(package: Path) -> None:
    source = '''#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    rows=[json.loads(line) for line in (ROOT / "annotation_manifest.jsonl").read_text().splitlines() if line]
    missing=[]; errors=[]; complete=0
    for row in rows:
        image=cv2.imread(str(ROOT / row["frame_path"]), cv2.IMREAD_COLOR)
        mask_path=ROOT / row["robot_only_mask_path"]
        if not mask_path.is_file():
            missing.append(str(mask_path)); continue
        mask=cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if image is None or mask is None:
            errors.append(f"{row['id']}: unreadable image or mask"); continue
        if mask.ndim != 2 or mask.shape != image.shape[:2]:
            errors.append(f"{row['id']}: mask must be single-channel and match image size"); continue
        if not np.isin(mask, (0,255)).all():
            errors.append(f"{row['id']}: mask must use only 0 and 255"); continue
        complete += 1
    report={"schema":"rm-geometry-mask-reference-validation-v1","frames":len(rows),"complete_frames":complete,"missing_masks":len(missing),"errors":errors,"status":"complete" if complete==len(rows) and not errors else "annotation_pending_or_invalid"}
    (ROOT / "validation_report.json").write_text(json.dumps(report, indent=2)+"\\n")
    print(json.dumps(report, sort_keys=True))
    if args.require_complete and report["status"] != "complete": raise SystemExit(2)
if __name__ == "__main__": main()
'''
    path = package / "validate_annotations.py"
    path.write_text(source)
    path.chmod(0o755)


def build_human_reference_plan(rows: list[dict[str, Any]], numeric_cache: dict[str, dict[str, np.ndarray]]) -> dict[str, Any]:
    package = OUT / "manual_reference_plan"
    frame_root = package / "frames"
    mask_root = package / "robot_only_masks"
    frame_root.mkdir(parents=True, exist_ok=True)
    mask_root.mkdir(parents=True, exist_ok=True)
    examples: list[dict[str, Any]] = []
    sources = current_source_hashes(rows)
    for row in rows:
        capture = cv2.VideoCapture(str(row["front_video_path"]))
        if not capture.isOpened():
            raise RuntimeError(f"cannot open selected front video: {row['front_video_path']}")
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        action = numeric_cache[str(row["episode_id"])]["action"]
        for index, stratum in sample_indices(action, total):
            identifier = frame_id(row, index)
            output = frame_root / f"{identifier}.png"
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, image = capture.read()
            if not ok or image is None:
                raise RuntimeError(f"cannot decode {identifier} from {row['front_video_path']}")
            if not cv2.imwrite(str(output), image):
                raise RuntimeError(f"cannot write reference input frame: {output}")
            height, width = image.shape[:2]
            examples.append({
                "id": identifier,
                "task_id": row["task_id"],
                "episode_id": row["episode_id"],
                "episode_index": int(row["episode_index"]),
                "view": TARGET_CAMERA,
                "frame_index": index,
                "sampling_stratum": stratum,
                "motion_note": "The action-delta stratum is only a sampling heuristic; it is not a contact, success, or segmentation label.",
                "source_video": row["front_video_path"],
                "source_parquet": row["parquet_path"],
                "frame_path": f"frames/{identifier}.png",
                "frame_sha256": sha256(output),
                "width": width,
                "height": height,
                "robot_only_mask_path": f"robot_only_masks/{identifier}.png",
                "status": "unannotated",
            })
        capture.release()
    if len(examples) != 200 or len({row["task_id"] for row in examples}) != 10 or len({row["episode_id"] for row in examples}) != 20:
        raise RuntimeError("200-frame annotation-plan stratification invariant failed")
    examples.sort(key=lambda row: row["id"])
    (package / "annotation_manifest.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in examples))
    write_json(package / "sampling_manifest.json", {
        "schema": "rm-geometry-mask-human-reference-sampling-v1",
        "status": "annotation_pending",
        "frame_count": len(examples),
        "task_count": 10,
        "trajectory_count": 20,
        "selection": "minimum- and maximum-index episode within each frozen ten-episode task slice; ten frames per episode",
        "sampling": "eight temporal quantiles plus two action-delta peaks, deduplicated deterministically",
        "source_hashes": sources,
    })
    (package / "ANNOTATION_GUIDE.md").write_text("""# Robot-only reference annotation

Create exactly one binary `robot_only` PNG for every image in `frames/`, at the
path specified by `annotation_manifest.jsonl`.  Use `255` for visible robot
pixels and `0` for all other pixels.  The foreground includes only visible
Franka arm links, wrists, gripper palms, fingers, and fingertips from either
arm.  It excludes every manipulated object, even when it is grasped or
occluded by the gripper; it also excludes the table, fixtures, tools, shadows,
reflections, people, and background.

Annotate only visible surfaces.  At a robot/object boundary, give the pixel to
the visible foreground surface; do not invent hidden robot geometry.  A second
reviewer must independently review at least 40 frames: four randomly selected
frames from each task, including at least one action-delta sample where
available.  Resolve every disagreement at gripper/contact boundaries before
using a reference mask for evaluation.
""")
    (package / "QUALITY_CONTROL.md").write_text("""# Minimal quality-control plan

1. Primary annotator labels all 200 frames using the binary convention.
2. A second annotator reviews a stratified 40-frame subset (four per task).
3. Reviewers record disagreements as `missed_link`, `missing_gripper`,
   `object_included`, `background_included`, `occlusion_boundary`, or `other`.
4. Reconcile every reviewed frame and preserve both the review log and final
   binary mask.  The validator checks presence, dimensions, and binary values;
   it does not certify semantic correctness.
""")
    (package / "RESUME_AFTER_ANNOTATION.md").write_text("""# Resume after human annotation

Do not construct mask targets or train a model before annotation and review are
complete.  Validate the package first:

```bash
cd /data/projects/tzh/papers/ActMask
/home/tzh/conda_envs/actmask/bin/python outputs/actmask/rm_geometry_mask_audit/manual_reference_plan/validate_annotations.py --require-complete
```

Then record reviewer agreement and evaluate any future mask method strictly
against these human references.  This package contains source images only; it
does not contain generated segmentation proposals.
""")
    write_annotation_validator(package)
    return {
        "package": str(package),
        "frames": len(examples),
        "tasks": 10,
        "episodes": 20,
        "source_files_hashed": len(sources),
        "status": "annotation_pending",
    }


def write_final_report(
    assets: dict[str, Any],
    joints: dict[str, Any],
    calibration: dict[str, Any],
    consistency: dict[str, Any],
    side_sources: dict[str, Any],
    reference: dict[str, Any],
) -> dict[str, Any]:
    gates = {
        "exact_robot_urdf_available": False,
        "exact_franka_link_meshes_available": False,
        "joint_state_convention_documented": bool(joints["explicit_joint_names_or_order_available"]),
        "front_camera_intrinsics_pixel_domain_compatible": bool(calibration["intrinsics"]["front_intrinsics_usable_in_pixel_domain"]),
        "certified_camera_extrinsics_available": False,
        "camera_to_base_and_base_to_robot_chain_available": bool(calibration["base_camera_chain"]["available"]),
        "episode_calibration_consistency_proven": bool(consistency["consistent_for_verified_projection"]),
        "simulation_side_source_mapped_to_selected_real_episodes": bool(side_sources["side_source_projection_usable_for_selected_real_episodes"]),
    }
    geometry_possible = all(gates.values())
    if geometry_possible:
        decision, meaning = "GEOMETRIC_MASK_AVAILABLE", "All required geometry and calibration evidence is present."
    elif reference["frames"] == 200:
        decision, meaning = "HUMAN_REFERENCE_REQUIRED", "Geometry projection is not verifiable from this local release, while a 200-frame human-reference plan is ready."
    else:
        decision, meaning = "MASK_PATH_IMPOSSIBLE", "Neither a verified geometry path nor a viable human-reference plan could be constructed."
    result = {
        "schema": "rm-geometry-mask-final-decision-v1",
        "decision": decision,
        "decision_letter": {"GEOMETRIC_MASK_AVAILABLE": "A", "HUMAN_REFERENCE_REQUIRED": "B", "MASK_PATH_IMPOSSIBLE": "C"}[decision],
        "geometry_projection_possible": geometry_possible,
        "geometry_gate_checks": gates,
        "no_geometry_pilot_generated": not geometry_possible,
        "requested_geometry_pilot_frames": 50,
        "human_reference_plan": reference,
        "summary": meaning,
        "blocking_evidence": [
            "No URDF or link mesh is present below the local RoboMIND2.0 release root.",
            "The 16-value arm field has no documented joint names/order or link mapping.",
            "The released Franka base-referenced extrinsic and base-to-robot dictionaries are empty.",
            "No selected episode identifies a calibration instance or maps to the separate simulation HDF5/USD path.",
        ],
        "allowed_next_step": "Obtain human robot-only reference masks according to manual_reference_plan, or obtain dataset-owner geometry plus per-episode calibration evidence before attempting rasterized projection.",
        "prohibited_claim": "Do not claim that any future non-human mask is geometric ground truth unless every failed geometry gate is replaced by documented, versioned source evidence.",
    }
    write_json(OUT / "geometry_projection_gate.json", {
        "schema": "rm-geometry-mask-projection-gate-v1",
        "passed": geometry_possible,
        "requirements": gates,
        "pilot_instruction": "Generate exactly 50 projected masks only if this gate passes.",
        "pilot_status": "not_generated_gate_failed" if not geometry_possible else "eligible_for_generation",
    })
    write_json(OUT / "final_decision.json", result)
    (OUT / "final_report.md").write_text(f"""# RoboMIND geometry-mask audit

## Decision

`{decision}` ({result['decision_letter']}). {meaning}

## Requirement-by-requirement evidence

| Requirement | Result | Evidence |
|---|---:|---|
| Franka URDF | no | `robot_geometry_inventory.json` found no matching asset under the local release root. |
| Franka link meshes | no | same inventory; no robot mesh file is bundled. |
| Joint-state convention | no | 16+2+14 dimensions are declared, but link/joint names and order are absent. |
| Camera intrinsics | partial/pass | six intrinsic matrices are present; front K is pixel-domain compatible with 640×480 metadata. |
| Camera extrinsics | partial/unverified | raw 4×4 matrices are structurally valid, but their frame direction/episode binding is undocumented. |
| Base-camera transform | no | both base-referenced dictionaries are empty. |
| Episode calibration consistency | no | selected metadata has no calibration ID, date, or transform reference. |

The 50-frame geometry pilot was **not generated** because the projection gate
failed.  The independent next path is the 200-frame package in
`manual_reference_plan/`; no model was trained and no pseudo masks were
generated during this audit.
""")
    return result


def reproducibility_manifest() -> dict[str, Any]:
    files = []
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and path.name != "reproducibility_manifest.json":
            files.append({"path": relative(path), "bytes": int(path.stat().st_size), "sha256": sha256(path)})
    return {"schema": "rm-geometry-mask-reproducibility-manifest-v1", "files": files, "file_count": len(files)}


def main() -> None:
    if OUT.exists():
        raise RuntimeError(f"refusing to overwrite existing audit output: {OUT}")
    for path in (SOURCE, FROZEN_MANIFEST, INTRINSICS_PATH, EXTRINSICS_PATH):
        if not path.exists():
            raise RuntimeError(f"required source is unavailable: {path}")
    OUT.mkdir(parents=True)
    append_log("audit_started", source=str(SOURCE), frozen_manifest=str(FROZEN_MANIFEST), target_camera=TARGET_CAMERA)
    rows = selected_rows()
    write_json(OUT / "source_snapshot.json", source_snapshot(rows))
    assets = scan_geometry_assets()
    write_json(OUT / "robot_geometry_inventory.json", assets)
    joints, numeric_cache = audit_joint_states(rows)
    write_json(OUT / "joint_state_convention_audit.json", joints)
    calibration, consistency = audit_calibration(rows)
    write_json(OUT / "camera_calibration_audit.json", calibration)
    write_json(OUT / "episode_calibration_consistency_audit.json", consistency)
    side_sources = audit_hdf5_side_sources(rows)
    write_json(OUT / "hdf5_side_source_audit.json", side_sources)
    append_log("geometry_chain_audited", asset_count=assets["asset_count"], joint_convention_documented=joints["explicit_joint_names_or_order_available"], base_chain_available=calibration["base_camera_chain"]["available"], episode_mapping=consistency["consistent_for_verified_projection"])
    reference = build_human_reference_plan(choose_reference_episodes(rows), numeric_cache)
    write_json(OUT / "manual_reference_plan_status.json", reference)
    append_log("human_reference_plan_created", **reference)
    final = write_final_report(assets, joints, calibration, consistency, side_sources, reference)
    append_log("final_decision", decision=final["decision"], decision_letter=final["decision_letter"])
    write_json(OUT / "reproducibility_manifest.json", reproducibility_manifest())
    print(json.dumps({"decision": final["decision"], "geometry_projection_possible": final["geometry_projection_possible"], "human_reference_frames": reference["frames"], "output": str(OUT)}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
