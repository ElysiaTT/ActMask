#!/usr/bin/env python3
"""Construct SAM3_PSEUDOMASK-excluded future-change pseudo-targets after P2."""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from actmask.data.robomind_adapter import fair_input_audit, load_numeric_episode, validate_numeric_episode


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "actmask" / "rm_sam3_acd_pilot"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def append_log(event: str, **payload: Any) -> None:
    path = OUT / "run_log.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    previous, sequence = rows[-1]["event_sha256"], int(rows[-1]["seq"])
    row = {"seq": sequence + 1, "utc": datetime.now(timezone.utc).isoformat(), "event": event, "payload": payload, "prev_event_sha256": previous}
    row["event_sha256"] = hashlib.sha256(json.dumps(row, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    with path.open("a") as handle: handle.write(json.dumps(row, sort_keys=True) + "\n")


def pseudomask_id(sample: dict[str, Any], frame: int) -> str:
    return f"{sample['task_id']}__ep{int(sample['episode_index']):06d}__f{int(frame):06d}"


def read_frame(capture: cv2.VideoCapture, index: int, width: int, height: int) -> np.ndarray:
    capture.set(cv2.CAP_PROP_POS_FRAMES, index); ok, frame = capture.read()
    if not ok: raise RuntimeError(f"cannot decode source RGB frame {index}")
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def read_mask(relative: str, width: int, height: int) -> np.ndarray:
    source = cv2.imread(str(OUT / relative), cv2.IMREAD_GRAYSCALE)
    if source is None: raise RuntimeError(f"missing SAM3_PSEUDOMASK {relative}")
    return cv2.resize(source, (width, height), interpolation=cv2.INTER_NEAREST) > 0


def flow(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    a, b = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY), cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    return cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0)


def components(mask: np.ndarray, minimum: int) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = np.zeros_like(mask, dtype=bool)
    for index in range(1, count):
        if int(stats[index, cv2.CC_STAT_AREA]) >= minimum: keep[labels == index] = True
    return keep


def residual(before: np.ndarray, future: np.ndarray, clip: list[float]) -> np.ndarray:
    a, b = before.astype(np.float32) / 255.0, future.astype(np.float32) / 255.0
    ratio = np.clip(float(np.mean(a) / max(np.mean(b), 1e-4)), clip[0], clip[1])
    return np.abs(a - np.clip(b * ratio, 0.0, 1.0)).mean(axis=2)


def target(before: np.ndarray, future: np.ndarray, robot: np.ndarray, protocol: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    spec, gates = protocol["construction"], protocol["target_gates"]
    forward, backward = flow(before, future), flow(future, before)
    h, w = robot.shape; x, y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    warped = cv2.remap(backward, x + forward[..., 0], y + forward[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    fb = np.linalg.norm(forward + warped, axis=2); magnitude = np.linalg.norm(forward, axis=2); resid = residual(before, future, spec["brightness_ratio_clip"])
    gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY); gradient = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3) ** 2 + cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3) ** 2
    support = gradient >= np.percentile(gradient, spec["gradient_support_percentile"])
    raw_motion = (magnitude >= spec["flow_magnitude_threshold_px"]) & (fb <= spec["forward_backward_error_strict_px"])
    raw_residual = (resid >= spec["residual_threshold_0_1"]) & (fb <= spec["forward_backward_error_residual_px"])
    raw = components(raw_motion | raw_residual, int(spec["minimum_component_pixels"])); affected = components(raw & ~robot, int(spec["minimum_component_pixels"]))
    confidence = np.exp(-fb / spec["forward_backward_error_strict_px"]) * support.astype(np.float32) * (~robot)
    border = np.zeros((h, w), dtype=bool); band = 12; border[:band] = True; border[-band:] = True; border[:, :band] = True; border[:, -band:] = True
    nonrobot = ~robot
    values = {"mask": affected.astype(np.uint8), "confidence": np.clip(np.round(confidence * 255), 0, 255).astype(np.uint8), "flow": forward.astype(np.float16), "feature_residual": np.clip(np.round(resid * 255), 0, 255).astype(np.uint8)}
    metrics = {"affected_area": float(affected.mean()), "raw_change_area": float(raw.mean()), "robot_raw_overlap_fraction": float((raw & robot).sum() / max(int(raw.sum()), 1)), "static_border_affected_area": float((affected & border & nonrobot).sum() / max(int((border & nonrobot).sum()), 1)), "confidence_coverage": float((confidence[nonrobot] >= 0.5).mean()) if nonrobot.any() else 0.0, "mean_confidence": float(confidence[nonrobot].mean()) if nonrobot.any() else 0.0, "mean_flow_magnitude_affected": float(magnitude[affected].mean()) if affected.any() else 0.0, "mean_residual_affected": float(resid[affected].mean()) if affected.any() else 0.0, "brightness_shift": float(np.mean(future.astype(np.float32) / 255.0) - np.mean(before.astype(np.float32) / 255.0))}
    return values, metrics


def correlation(a: list[float], b: list[float]) -> float | None:
    if len(a) < 3 or np.std(a) < 1e-9 or np.std(b) < 1e-9: return None
    return float(np.corrcoef(a, b)[0, 1])


def overlay(before: np.ndarray, robot: np.ndarray, affected: np.ndarray) -> np.ndarray:
    result = before.copy(); result[robot] = (0.35 * result[robot] + 0.65 * np.asarray([0, 0, 255])).astype(np.uint8); result[affected.astype(bool)] = (0.30 * result[affected.astype(bool)] + 0.70 * np.asarray([0, 255, 0])).astype(np.uint8)
    return result


def run() -> dict[str, Any]:
    screening = json.loads((OUT / "pseudomask_screening_audit.json").read_text())
    if screening["decision"] != "SAM3_PSEUDOMASK_PILOT_ACCEPTED": raise RuntimeError("P3 is gated on SAM3_PSEUDOMASK_PILOT_ACCEPTED")
    protocol_path = OUT / "pseudotarget_preregistered_protocol.json"
    if sha256(protocol_path) != (OUT / "pseudotarget_preregistered_protocol.sha256").read_text().strip(): raise RuntimeError("SAM3_PSEUDOMASK target protocol hash mismatch")
    protocol = json.loads(protocol_path.read_text()); width, height = protocol["target_resolution"]
    samples = [json.loads(line) for line in (OUT / "target_window_manifest.jsonl").read_text().splitlines() if line]
    masks = {row["frame_id"]: row for row in [json.loads(line) for line in (OUT / "pseudomask_manifest.jsonl").read_text().splitlines() if line]}
    if len(samples) != 2400: raise RuntimeError("expected exact frozen 2,400 pseudo-target samples")
    target_root, input_root, contact_root = OUT / "pseudotargets" / "targets", OUT / "pseudotargets" / "fair_inputs", OUT / "pseudotarget_contact_sheets"
    for path in (target_root, input_root, contact_root): path.mkdir(parents=True, exist_ok=True)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples: groups[sample["episode_id"]].append(sample)
    numeric_cache: dict[str, dict[str, np.ndarray]] = {}
    records = {row["episode_id"]: row for row in json.loads((OUT / "selected_trajectory_manifest.json").read_text())["episodes"]}
    output_rows, contacts, examples = [], [], Counter()
    for episode_id, episode_samples in sorted(groups.items()):
        source = records[episode_id]
        class Record: pass
        record = Record(); record.parquet_path = source["parquet_path"]; record.episode_id = source["episode_id"]; record.task_id = source["task_id"]; record.robot_embodiment = source["robot_embodiment"]; record.language_instruction = source["language_instruction"]; record.front_video_path = source["front_video_path"]; record.source_bytes = source["source_bytes"]; record.episode_index = source["episode_index"]; record.metadata_path = source["metadata_path"]
        numeric = load_numeric_episode(record); audit = validate_numeric_episode(record, numeric)
        if not audit["pass"]: raise RuntimeError(f"invalid source alignment {episode_id}")
        needed = set()
        for sample in episode_samples:
            needed.update(sample["history_frame_indices"]); needed.add(sample["pre_frame_index"]); needed.add(sample["future_frame_index"])
        capture = cv2.VideoCapture(source["front_video_path"]); frames = {index: read_frame(capture, index, width, height) for index in sorted(needed)}; capture.release()
        if max(frames) >= len(numeric["state"]): raise RuntimeError(f"future frame beyond numeric source for {episode_id}")
        by_anchor: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for sample in episode_samples: by_anchor[int(sample["anchor_index"])].append(sample)
        for anchor, anchor_samples in sorted(by_anchor.items()):
            exemplar = anchor_samples[0]; input_id = f"{exemplar['task_id']}__ep{int(exemplar['episode_index']):06d}__a{anchor:04d}"
            input_path = input_root / f"{input_id}.npz"
            np.savez_compressed(input_path, pre_rgb=np.asarray([frames[index] for index in exemplar["history_frame_indices"]], dtype=np.uint8), pre_state=numeric["state"][anchor - len(exemplar["history_frame_indices"]):anchor].astype(np.float32), action=numeric["action"][anchor:anchor + 4].astype(np.float32))
            before = frames[exemplar["pre_frame_index"]]; horizon_visuals = []
            for sample in sorted(anchor_samples, key=lambda row: row["horizon_steps"]):
                future = frames[sample["future_frame_index"]]
                pre_key, future_key = pseudomask_id(sample, sample["pre_frame_index"]), pseudomask_id(sample, sample["future_frame_index"])
                if pre_key not in masks or future_key not in masks: raise RuntimeError(f"missing required SAM3_PSEUDOMASK for {sample['sample_id']}")
                robot = read_mask(masks[pre_key]["union_mask_path"], width, height) | read_mask(masks[future_key]["union_mask_path"], width, height)
                values, metrics = target(before, future, robot, protocol)
                target_path = target_root / f"{input_id}__{sample['horizon']}.npz"; np.savez_compressed(target_path, **values)
                row = {"schema": "SAM3_PSEUDOMASK_target_window_v1", "sample_id": sample["sample_id"], "input_id": input_id, "episode_id": episode_id, "task_id": sample["task_id"], "episode_index": sample["episode_index"], "horizon": sample["horizon"], "horizon_steps": sample["horizon_steps"], "anchor_index": anchor, "pre_frame_index": sample["pre_frame_index"], "history_frame_indices": sample["history_frame_indices"], "action_frame_indices": sample["action_frame_indices"], "future_frame_index": sample["future_frame_index"], "input_path": str(input_path.relative_to(OUT)), "target_path": str(target_path.relative_to(OUT)), "pseudomask_provenance_audit_only": {"pre": masks[pre_key]["union_mask_path"], "future": masks[future_key]["union_mask_path"]}, "action_magnitude": float(np.linalg.norm(numeric["action"][anchor:anchor + 4])), "episode_progress": float(anchor / len(numeric["state"])), "robot_pseudomask_area": float(robot.mean()), **metrics}
                output_rows.append(row); horizon_visuals.append((sample["horizon"], future, values["mask"], robot))
            if exemplar["anchor_ordinal"] == 3 and examples[exemplar["task_id"]] < 2:
                rows = []
                for horizon, future, affected, robot in horizon_visuals:
                    panel = overlay(before, robot, affected); cv2.putText(panel, f"SAM3_PSEUDOMASK {horizon}: red=excluded green=target", (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (255,255,255), 1, cv2.LINE_AA); rows.append(np.concatenate([before, future, panel], axis=1))
                contact = contact_root / f"{exemplar['task_id']}__ep{int(exemplar['episode_index']):06d}__a{anchor:04d}.jpg"; cv2.imwrite(str(contact), np.concatenate(rows, axis=0)); contacts.append({"task_id": exemplar["task_id"], "episode_id": episode_id, "anchor": anchor, "path": str(contact.relative_to(OUT))}); examples[exemplar["task_id"]] += 1
    manifest = OUT / "target_manifest_sam3.jsonl"; manifest.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in output_rows))
    by_horizon: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in output_rows: by_horizon[row["horizon"]].append(row)
    gates = protocol["target_gates"]; summary = {}; all_pass = True
    for horizon, rows in sorted(by_horizon.items()):
        areas, confidence = [row["affected_area"] for row in rows], [row["confidence_coverage"] for row in rows]
        meaningful = float(np.mean(np.asarray(areas) >= gates["meaningful_area_fraction_threshold"])); conf_good = float(np.mean(np.asarray(confidence) >= 0.60)); static = float(np.mean([row["static_border_affected_area"] for row in rows])); overlap = float(np.mean([row["robot_raw_overlap_fraction"] for row in rows])); action_corr, progress_corr = correlation([row["action_magnitude"] for row in rows], areas), correlation([row["episode_progress"] for row in rows], areas)
        passed = gates["meaningful_window_fraction"][0] <= meaningful <= gates["meaningful_window_fraction"][1] and conf_good >= gates["confidence_window_fraction_min"] and static <= gates["static_border_affected_area_max"] and overlap <= gates["robot_raw_overlap_fraction_max"] and (action_corr is None or abs(action_corr) <= gates["action_or_progress_absolute_correlation_max"]) and (progress_corr is None or abs(progress_corr) <= gates["action_or_progress_absolute_correlation_max"])
        all_pass &= passed
        summary[horizon] = {"samples": len(rows), "affected_area": {"mean": float(np.mean(areas)), "median": float(np.median(areas)), "p10": float(np.percentile(areas,10)), "p90": float(np.percentile(areas,90))}, "meaningful_window_fraction": meaningful, "confidence_window_fraction_ge_0_60": conf_good, "static_background_border_area": static, "robot_pseudomask_raw_overlap_fraction": overlap, "action_magnitude_correlation": action_corr, "episode_progress_correlation": progress_corr, "passes_frozen_pseudotarget_gates": passed}
    task_count = len({row["task_id"] for row in output_rows}); all_pass &= task_count >= gates["minimum_task_diversity"] and len(contacts) >= 20
    decision = "SAM3_PSEUDOMASK_TARGET_VALID" if all_pass else "SAM3_PSEUDOMASK_TARGET_INVALID"
    schema = {"schema": "SAM3_PSEUDOMASK_target_schema_v1", "fair_input_fields": ["pre_rgb", "pre_state", "action"], "fair_input_audit": fair_input_audit(["pre_rgb", "pre_state", "action"]), "forbidden_from_fair_inputs": ["SAM3 pseudo-mask", "future RGB/depth/state", "target mask", "target confidence", "future flow", "episode ID", "source path"], "future_derived_fields": ["mask", "confidence", "flow", "feature_residual"], "resolution": [width,height], "semantics": "SAM3_PSEUDOMASK-derived exploratory targets only; no segmentation-ground-truth or validated benchmark claim"}
    dump(OUT / "target_schema_sam3.json", schema)
    dump(OUT / "target_construction_audit_sam3.json", {"schema": "SAM3_PSEUDOMASK_target_construction_audit_v1", "samples": len(output_rows), "inputs": len({row["input_id"] for row in output_rows}), "episodes": len(groups), "horizons": sorted(by_horizon), "all_temporally_separated": all(max(row["history_frame_indices"]) < min(row["action_frame_indices"]) < row["future_frame_index"] for row in output_rows), "fair_input_audit": schema["fair_input_audit"], "pseudomasks_target_only": True, "no_3d_claim": True, "contact_sheets": contacts})
    dump(OUT / "target_statistics_sam3.json", {"schema": "SAM3_PSEUDOMASK_target_statistics_v1", "decision": decision, "per_horizon": summary, "tasks": task_count, "contacts": len(contacts)})
    report = ["# SAM3_PSEUDOMASK target quality report", "", f"Decision: `{decision}`", "", "All values are properties of exploratory SAM3_PSEUDOMASK-derived targets, not real-world segmentation or prediction accuracy.", ""]
    for horizon, values in summary.items(): report.append(f"- {horizon}: meaningful={values['meaningful_window_fraction']:.3f}, confidence>=0.60={values['confidence_window_fraction_ge_0_60']:.3f}, robot raw overlap={values['robot_pseudomask_raw_overlap_fraction']:.3f}, pass={values['passes_frozen_pseudotarget_gates']}")
    (OUT / "target_quality_report.md").write_text("\n".join(report) + "\n")
    failure = sorted(output_rows, key=lambda row: (-row["robot_raw_overlap_fraction"], -row["static_border_affected_area"]))[:30]
    (OUT / "target_failure_cases_sam3.md").write_text("# SAM3_PSEUDOMASK pseudo-target review cases\n\n" + "\n".join(f"- `{row['sample_id']}`: raw-robot-overlap={row['robot_raw_overlap_fraction']:.3f}, static-border={row['static_border_affected_area']:.3f}, affected={row['affected_area']:.3f}" for row in failure) + "\n")
    append_log("P3_SAM3_PSEUDOMASK_targets_constructed", decision=decision, inputs=len({row["input_id"] for row in output_rows}), targets=len(output_rows), contacts=len(contacts))
    return {"decision": decision, "inputs": len({row["input_id"] for row in output_rows}), "targets": len(output_rows), "contacts": len(contacts)}


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, sort_keys=True))
