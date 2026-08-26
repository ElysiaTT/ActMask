"""Build conservative RM-ACD robot masks and direct multi-horizon RGB targets.

This is target construction, not model training. Future observations are read
only to make supervised targets and never enter the saved fair inputs.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from actmask.data.robomind_adapter import fair_input_audit, load_numeric_episode, validate_numeric_episode
from actmask.experiments.rm_acd_freeze_and_modality import OUT, append_log, write_json
from actmask.experiments.rm_acd_freeze_and_modality import _record, _video_path


CONFIG = json.loads((OUT / "preregistered_config.json").read_text())
WIDTH, HEIGHT = CONFIG["target"]["target_resolution"]
HISTORY = int(CONFIG["temporal"]["history_steps"])
ACTION = int(CONFIG["temporal"]["action_steps"])
FRACTIONS = tuple(float(value) for value in CONFIG["temporal"]["anchor_fractions"])
HORIZONS = {name: int(value["steps"]) for name, value in CONFIG["temporal"]["horizons"].items()}
FLOW_THRESHOLD = float(CONFIG["target"]["flow_magnitude_threshold_px_at_target_resolution"])
RESIDUAL_THRESHOLD = float(CONFIG["target"]["residual_threshold_0_1"])
MIN_COMPONENT = int(CONFIG["target"]["minimum_component_pixels"])
MEANINGFUL_AREA = float(CONFIG["target"]["meaningful_affected_area_fraction_min"])


def _anchors(length: int) -> list[int]:
    latest = length - ACTION - max(HORIZONS.values()) - 1
    if latest - HISTORY + 1 < len(FRACTIONS):
        raise ValueError(f"fewer than {len(FRACTIONS)} legal anchors for length={length}; interval=[{HISTORY}, {latest}]")
    result = np.rint(np.linspace(HISTORY, latest, len(FRACTIONS))).astype(int).tolist()
    if len(set(result)) != len(result):
        raise ValueError(f"anchor collision despite legal interval length={length}: {result}")
    return result


def _frames(path: Path) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    result: list[np.ndarray] = []
    while True:
        ok, image = capture.read()
        if not ok:
            break
        result.append(cv2.resize(image, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA))
    capture.release()
    if not result:
        raise ValueError(f"cannot decode source RGB: {path}")
    return result


def _gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _flow(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return cv2.calcOpticalFlowFarneback(_gray(left), _gray(right), None, 0.5, 3, 15, 3, 5, 1.2, 0)


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    return cv2.dilate(mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))).astype(bool)


def _filter_components(mask: np.ndarray, minimum: int = MIN_COMPONENT) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = np.zeros_like(mask, dtype=bool)
    for index in range(1, count):
        if int(stats[index, cv2.CC_STAT_AREA]) >= minimum:
            keep[labels == index] = True
    return keep


def _edge_connected(mask: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    result = np.zeros_like(mask, dtype=bool)
    h, w = mask.shape
    for index in range(1, count):
        x, y, width, height, area = (int(value) for value in stats[index])
        touches = x <= 2 or y <= 2 or x + width >= w - 2 or y + height >= h - 2
        if touches and area >= MIN_COMPONENT:
            result[labels == index] = True
    return result


def _motion_seed(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    diff = np.abs(before.astype(np.float32) - after.astype(np.float32)).mean(axis=2) / 255.0
    flow = _flow(before, after)
    magnitude = np.linalg.norm(flow, axis=2)
    return _filter_components((diff >= RESIDUAL_THRESHOLD) | (magnitude >= FLOW_THRESHOLD))


def _task_robot_priors(rows: list[dict[str, Any]]) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    """Estimate a conservative arm sweep prior from boundary-connected motion."""
    counts: dict[str, np.ndarray] = defaultdict(lambda: np.zeros((HEIGHT, WIDTH), np.uint16))
    totals: Counter[str] = Counter()
    for row in rows:
        frames = _frames(Path(row["front_video_path"]))
        anchors = _anchors(len(frames))
        for anchor in anchors:
            seed = _edge_connected(_motion_seed(frames[anchor - 1], frames[anchor + ACTION - 1]))
            counts[row["task_id"]] += seed.astype(np.uint16)
            totals[row["task_id"]] += 1
    priors, audit = {}, {}
    for task, count in counts.items():
        frequency = count.astype(np.float32) / max(totals[task], 1)
        # A robot arm must repeatedly touch the image boundary across episodes.
        raw = frequency >= 0.08
        prior = _dilate(_filter_components(raw, 20), 3)
        priors[task] = prior
        audit[task] = {"samples": int(totals[task]), "edge_motion_frequency_threshold": 0.08, "raw_area_fraction": float(raw.mean()), "dilated_prior_area_fraction": float(prior.mean())}
    return priors, audit


def _robot_mask(before: np.ndarray, action_end: np.ndarray, prior: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    dynamic = _motion_seed(before, action_end)
    edge = _edge_connected(dynamic)
    # The task-level arm prior is intentionally dilated; the episode-specific
    # edge component captures arm portions outside the recurrent sweep.
    core = prior | _dilate(edge, 2)
    mask = _dilate(core, 2)
    return mask, {"dynamic_area": float(dynamic.mean()), "edge_dynamic_area": float(edge.mean()), "robot_core_area": float(core.mean()), "robot_mask_area": float(mask.mean())}


def _brightness_compensated_residual(before: np.ndarray, future: np.ndarray) -> np.ndarray:
    left, right = before.astype(np.float32) / 255.0, future.astype(np.float32) / 255.0
    scale = float(np.mean(left) / max(np.mean(right), 1e-4))
    right = np.clip(right * np.clip(scale, 0.85, 1.18), 0.0, 1.0)
    return np.abs(left - right).mean(axis=2)


def _target(before: np.ndarray, future: np.ndarray, robot: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    forward, backward = _flow(before, future), _flow(future, before)
    h, w = robot.shape
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x, map_y = grid_x + forward[..., 0], grid_y + forward[..., 1]
    warped_backward = cv2.remap(backward, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    fb_error = np.linalg.norm(forward + warped_backward, axis=2)
    magnitude = np.linalg.norm(forward, axis=2)
    residual = _brightness_compensated_residual(before, future)
    gradient = cv2.Sobel(_gray(before), cv2.CV_32F, 1, 0, ksize=3) ** 2 + cv2.Sobel(_gray(before), cv2.CV_32F, 0, 1, ksize=3) ** 2
    gradient_support = gradient >= np.percentile(gradient, 30)
    flow_confidence = np.exp(-fb_error / 1.25) * gradient_support.astype(np.float32)
    raw_motion = (magnitude >= FLOW_THRESHOLD) & (fb_error <= 1.25)
    raw_residual = (residual >= RESIDUAL_THRESHOLD) & (fb_error <= 2.0)
    raw = _filter_components(raw_motion | raw_residual)
    affected = _filter_components(raw & ~robot)
    confidence = flow_confidence * (~robot)
    nonrobot = ~robot
    border = np.zeros((h, w), dtype=bool); border[:12] = True; border[-12:] = True; border[:, :12] = True; border[:, -12:] = True
    values = {
        "mask": affected.astype(np.uint8),
        "confidence": np.clip(np.round(confidence * 255), 0, 255).astype(np.uint8),
        "flow": forward.astype(np.float16),
        "feature_residual": np.clip(np.round(residual * 255), 0, 255).astype(np.uint8),
    }
    stats = {
        "affected_area": float(affected.mean()),
        "raw_change_area": float(raw.mean()),
        "robot_raw_overlap_fraction": float((raw & robot).sum() / max(int(raw.sum()), 1)),
        "static_border_affected_area": float((affected & border & nonrobot).sum() / max(int((border & nonrobot).sum()), 1)),
        "confidence_coverage": float((confidence[nonrobot] >= 0.5).mean()) if nonrobot.any() else 0.0,
        "mean_confidence": float(confidence[nonrobot].mean()) if nonrobot.any() else 0.0,
        "mean_flow_magnitude": float(magnitude[affected].mean()) if affected.any() else 0.0,
        "mean_residual": float(residual[affected].mean()) if affected.any() else 0.0,
        "brightness_shift": float(np.mean(future.astype(np.float32) / 255.0) - np.mean(before.astype(np.float32) / 255.0)),
    }
    return values, stats


def _overlay(image: np.ndarray, robot: np.ndarray, affected: np.ndarray) -> np.ndarray:
    output = image.copy()
    output[robot] = (0.35 * output[robot] + 0.65 * np.asarray([0, 0, 255])).astype(np.uint8)
    output[affected.astype(bool)] = (0.30 * output[affected.astype(bool)] + 0.70 * np.asarray([0, 255, 0])).astype(np.uint8)
    return output


def _contact(path: Path, before: np.ndarray, robot: np.ndarray, horizon_rows: list[tuple[str, np.ndarray, np.ndarray]]) -> None:
    rows = []
    for name, future, affected in horizon_rows:
        labelled = _overlay(before, robot, affected)
        cv2.putText(labelled, f"{name}: red=robot-mask green=target", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1, cv2.LINE_AA)
        rows.append(np.concatenate([before, future, labelled], axis=1))
    cv2.imwrite(str(path), np.concatenate(rows, axis=0))


def _correlation(left: list[float], right: list[float]) -> float | None:
    if len(left) < 3 or np.std(left) < 1e-9 or np.std(right) < 1e-9:
        return None
    return float(np.corrcoef(np.asarray(left), np.asarray(right))[0, 1])


def run() -> dict[str, Any]:
    selected = json.loads((OUT / "selected_trajectory_manifest.json").read_text())["episodes"]
    priors, prior_audit = _task_robot_priors(selected)
    target_root, input_root, contact_root = OUT / "processed_subset" / "targets", OUT / "processed_subset" / "fair_inputs", OUT / "target_audit_contact_sheets"
    target_root.mkdir(parents=True, exist_ok=True); input_root.mkdir(parents=True, exist_ok=True); contact_root.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    robot_stats, target_stats = [], []
    contacts = []
    examples_per_task: Counter[str] = Counter()
    contact_episodes: set[str] = set()
    for source in selected:
        record = _record(source)
        numeric = load_numeric_episode(record)
        alignment = validate_numeric_episode(record, numeric)
        if not alignment["pass"]:
            raise ValueError(f"numeric alignment failure: {record.episode_id}")
        frames = _frames(_video_path(record, "front"))
        if len(frames) != len(numeric["state"]):
            raise ValueError(f"RGB/numeric frame mismatch: {record.episode_id}")
        task = record.task_id
        for ordinal, anchor in enumerate(_anchors(len(frames))):
            before, action_end = frames[anchor - 1], frames[anchor + ACTION - 1]
            robot, mask_detail = _robot_mask(before, action_end, priors[task])
            input_id = f"{record.episode_id.replace(':', '_')}_a{ordinal:02d}"
            input_path = input_root / f"{input_id}.npz"
            pre = np.asarray(frames[anchor - HISTORY : anchor], dtype=np.uint8)
            np.savez_compressed(input_path, pre_rgb=pre, pre_state=numeric["state"][anchor - HISTORY : anchor].astype(np.float32), action=numeric["action"][anchor : anchor + ACTION].astype(np.float32))
            robot_stats.append({"episode_id": record.episode_id, "task": task, "anchor": anchor, **mask_detail})
            horizon_rows = []
            for name, steps in HORIZONS.items():
                future = frames[anchor + ACTION + steps - 1]
                values, detail = _target(before, future, robot)
                path = target_root / f"{input_id}_{name}.npz"
                np.savez_compressed(path, **values)
                action_mag = float(np.linalg.norm(numeric["action"][anchor : anchor + ACTION]))
                row = {"sample_id": input_id, "horizon": name, "horizon_steps": steps, "episode_id": record.episode_id, "task": task, "anchor_step": anchor, "pre_frame_indices": list(range(anchor - HISTORY, anchor)), "action_frame_indices": list(range(anchor, anchor + ACTION)), "future_frame_index": int(anchor + ACTION + steps - 1), "input_path": str(input_path.relative_to(OUT)), "target_path": str(path.relative_to(OUT)), "source_provenance_audit_only": {"parquet": record.parquet_path, "front_video": str(_video_path(record, 'front'))}, "action_magnitude": action_mag, "episode_progress": float(anchor / len(frames)), "robot_mask_area": mask_detail["robot_mask_area"], **detail}
                all_rows.append(row); target_stats.append(row)
                horizon_rows.append((name, future, values["mask"]))
            # Two distinct source trajectories from every task, not two
            # adjacent anchors from one video, form the 20-trajectory review
            # package.  A middle anchor provides visible pre/action context.
            if ordinal == 3 and examples_per_task[task] < 2 and record.episode_id not in contact_episodes:
                output = contact_root / f"{task}_{record.episode_index:06d}_a{ordinal:02d}.jpg"
                _contact(output, before, robot, horizon_rows)
                contacts.append({"task": task, "episode_id": record.episode_id, "anchor": anchor, "path": str(output.relative_to(OUT)), "robot_mask_area": mask_detail["robot_mask_area"]})
                examples_per_task[task] += 1
                contact_episodes.add(record.episode_id)
    manifest_path = OUT / "target_manifest.jsonl"
    manifest_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in all_rows))
    by_horizon: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in target_stats:
        by_horizon[row["horizon"]].append(row); by_task[row["task"]].append(row)
    summary = {}
    for name, rows in sorted(by_horizon.items()):
        area = [row["affected_area"] for row in rows]
        confidence = [row["confidence_coverage"] for row in rows]
        action = [row["action_magnitude"] for row in rows]
        progress = [row["episode_progress"] for row in rows]
        brightness = [row["brightness_shift"] for row in rows]
        summary[name] = {"samples": len(rows), "affected_area": {"mean": float(np.mean(area)), "median": float(np.median(area)), "p10": float(np.percentile(area, 10)), "p90": float(np.percentile(area, 90))}, "meaningful_area_threshold": MEANINGFUL_AREA, "meaningful_window_fraction": float(np.mean(np.asarray(area) >= MEANINGFUL_AREA)), "confidence_coverage": {"mean": float(np.mean(confidence)), "window_fraction_ge_0_60": float(np.mean(np.asarray(confidence) >= 0.60))}, "static_background_false_positive_area": float(np.mean([row["static_border_affected_area"] for row in rows])), "robot_raw_overlap_fraction": float(np.mean([row["robot_raw_overlap_fraction"] for row in rows])), "action_magnitude_correlation": _correlation(action, area), "episode_progress_correlation": _correlation(progress, area), "brightness_correlation": _correlation(brightness, area), "mean_flow_magnitude_affected": float(np.mean([row["mean_flow_magnitude"] for row in rows])), "mean_feature_residual_affected": float(np.mean([row["mean_residual"] for row in rows]))}
    target_schema = {"schema": "rm-acd-target-schema-v1", "fair_input_fields": ["pre_rgb", "pre_state", "action"], "fair_input_audit": fair_input_audit(["pre_rgb", "pre_state", "action"]), "future_derived_target_fields": ["mask", "confidence", "flow", "feature_residual"], "forbidden_from_fair_inputs": ["future RGB", "future depth", "future state", "target mask", "target confidence", "future flow", "episode ID", "source path"], "resolution": [WIDTH, HEIGHT], "horizons": HORIZONS, "temporal_separation": "pre frames end at anchor-1; action begins at anchor; each future target begins after the frozen action chunk", "flow_semantics": "2D image displacement only on confident non-robot pixels; no 3D or scene-flow interpretation"}
    write_json(OUT / "target_schema.json", target_schema)
    write_json(OUT / "robot_mask_method.json", {"schema": "rm-acd-robot-mask-method-v1", "selected_method_rank": 5, "method": "conservative image-domain recurrent boundary-motion arm-sweep prior plus per-window boundary-connected motion, with dilation", "geometric_projection_rejected": "local intrinsics exist but the provided raw camera-to-arm transforms do not yield a verified camera/world projection contract for the recorded end-effector coordinates; no robot geometry projection is asserted", "limitations": ["held objects touching the gripper can be masked", "not ground-truth segmentation", "manual visual audit is required before accepting targets"], "prior_audit": prior_audit})
    robot_summary = {"schema": "rm-acd-robot-mask-audit-v1", "samples": len(robot_stats), "task_summary": {task: {"samples": len(rows), "mean_mask_area": float(np.mean([row["robot_mask_area"] for row in rows])), "p90_mask_area": float(np.percentile([row["robot_mask_area"] for row in rows], 90)), "mean_edge_dynamic_area": float(np.mean([row["edge_dynamic_area"] for row in rows]))} for task, rows in ((task, [row for row in robot_stats if row["task"] == task]) for task in sorted({row["task"] for row in robot_stats}))}, "all_mask_area_below_half": bool(all(row["robot_mask_area"] < 0.50 for row in robot_stats)), "manual_review_required": True, "automatic_stability_check": bool(all(0.01 <= np.mean([row["robot_mask_area"] for row in robot_stats if row["task"] == task]) <= 0.50 for task in {row["task"] for row in robot_stats})), "interpretation": "The audit measures heuristic mask extent/stability, not segmentation accuracy. Contact sheets must be reviewed before target acceptance."}
    write_json(OUT / "robot_mask_audit.json", robot_summary)
    write_json(OUT / "robot_mask_contact_sheet_manifest.json", {"schema": "rm-acd-robot-mask-contact-sheets-v1", "samples": contacts, "coverage": {task: sum(row["task"] == task for row in contacts) for task in sorted(examples_per_task)}, "legend": "red=conservative robot mask; green=affected nonrobot target"})
    construction = {"schema": "rm-acd-target-construction-audit-v1", "samples": len(all_rows), "episodes": len(selected), "horizons": HORIZONS, "all_temporally_separated": all(max(row["pre_frame_indices"]) < min(row["action_frame_indices"]) < row["future_frame_index"] for row in all_rows), "fair_input_audit": target_schema["fair_input_audit"], "target_paths_separate_from_fair_inputs": True, "front_rgb_frame_alignment_verified": True, "camera_background_compensation": "per-pair global mean brightness ratio clipped to [0.85,1.18] before residual", "forward_backward_consistency": "flow target accepts forward/backward error <=1.25 pixels; residual evidence accepts <=2.0 pixels", "compression_noise_control": f"connected components smaller than {MIN_COMPONENT} target pixels are removed", "no_3d_claim": True}
    write_json(OUT / "target_construction_audit.json", construction)
    write_json(OUT / "target_statistics.json", {"schema": "rm-acd-target-statistics-v1", "per_horizon": summary, "per_task_samples": {task: len(rows) for task, rows in by_task.items()}, "no_change_vs_change": {name: {"change": int(sum(row["affected_area"] >= MEANINGFUL_AREA for row in rows)), "no_change": int(sum(row["affected_area"] < MEANINGFUL_AREA for row in rows))} for name, rows in by_horizon.items()}})
    write_json(OUT / "confidence_statistics.json", {"schema": "rm-acd-confidence-statistics-v1", "per_horizon": {name: {"mean_confidence_coverage": float(np.mean([row["confidence_coverage"] for row in rows])), "median_confidence_coverage": float(np.median([row["confidence_coverage"] for row in rows])), "window_fraction_ge_0_60": float(np.mean([row["confidence_coverage"] >= 0.60 for row in rows]))} for name, rows in by_horizon.items()}})
    append_log("phase_a2_a3_targets_constructed", inputs=800, targets=len(all_rows), contacts=len(contacts), target_manifest=str(manifest_path.relative_to(OUT)))
    return {"inputs": 800, "targets": len(all_rows), "contacts": len(contacts), "robot_mask_automatic_stability": robot_summary["automatic_stability_check"]}


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, sort_keys=True))
