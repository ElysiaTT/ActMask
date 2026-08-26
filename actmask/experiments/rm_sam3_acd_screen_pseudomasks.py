#!/usr/bin/env python3
"""Descriptive, non-reference screening for fixed SAM3_PSEUDOMASK artifacts."""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


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


def load_mask(relative: str) -> np.ndarray:
    return np.asarray(Image.open(OUT / relative).convert("L")) > 0


def run() -> dict[str, Any]:
    config = json.loads((OUT / "preregistered_config.json").read_text())
    if sha256(OUT / "preregistered_config.json") != (OUT / "preregistered_config.sha256").read_text().strip():
        raise RuntimeError("SAM3_PSEUDOMASK config hash mismatch")
    all_records = [json.loads(line) for line in (OUT / "pseudomask_manifest.jsonl").read_text().splitlines() if line]
    selection = json.loads((OUT / "pseudomask_audit_selection.json").read_text())
    by_id = {row["frame_id"]: row for row in all_records}
    audit = [by_id[f"{row['task_id']}__ep{int(row['episode_index']):06d}__f{int(row['frame_index']):06d}"] for row in selection["frames"]]
    if len(audit) != 100 or len({row["task_id"] for row in audit}) != 10 or len({row["episode_id"] for row in audit}) != 20:
        raise RuntimeError("SAM3_PSEUDOMASK audit coverage invariant failed")
    details, masks = [], {}
    for row in audit:
        mask = load_mask(row["union_mask_path"]); masks[row["frame_id"]] = mask
        components, _ = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
        arm = load_mask(row["robot_arm_prompt_union_path"]); gripper = load_mask(row["robotic_gripper_prompt_union_path"])
        details.append({"frame_id": row["frame_id"], "task_id": row["task_id"], "episode_id": row["episode_id"], "episode_index": row["episode_index"], "frame_index": row["frame_index"], "union_area_fraction": float(mask.mean()), "union_components": int(components - 1), "robot_arm_prompt_area_fraction": float(arm.mean()), "robotic_gripper_prompt_area_fraction": float(gripper.mean()), "gripper_prompt_nonempty": bool(gripper.any()), "broad_workspace_proxy": bool(mask.mean() > config["screening_acceptance"]["broad_workspace_proxy_area_fraction"]), "empty_union": not bool(mask.any())})
    temporal = []
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in details: groups[row["episode_id"]].append(row)
    for episode, group in sorted(groups.items()):
        group.sort(key=lambda row: row["frame_index"])
        for before, after in zip(group, group[1:]):
            a, b = masks[before["frame_id"]], masks[after["frame_id"]]
            union = (a | b).sum(); iou = float((a & b).sum() / union) if union else 1.0
            temporal.append({"episode_id": episode, "task_id": before["task_id"], "before_frame": before["frame_index"], "after_frame": after["frame_index"], "frame_gap": after["frame_index"] - before["frame_index"], "union_iou": iou, "absolute_area_delta": abs(before["union_area_fraction"] - after["union_area_fraction"])})
    area = np.asarray([row["union_area_fraction"] for row in details], dtype=np.float64)
    audit_acceptance = config["screening_acceptance"]
    nonempty = float(np.mean([not row["empty_union"] for row in details])); p95, maximum = float(np.percentile(area, 95)), float(area.max())
    passed = nonempty >= audit_acceptance["nonempty_union_fraction_min"] and p95 <= audit_acceptance["p95_union_area_fraction_max"] and maximum <= audit_acceptance["max_union_area_fraction_max"]
    failures = []
    for row in sorted(details, key=lambda item: (-item["union_area_fraction"], item["frame_id"]))[:10]:
        flags = []
        if row["broad_workspace_proxy"]: flags.append("broad_workspace_area_proxy")
        if row["empty_union"]: flags.append("empty_union")
        if not row["gripper_prompt_nonempty"]: flags.append("gripper_prompt_empty")
        if row["union_components"] >= 6: flags.append("fragmented_union")
        failures.append({"frame_id": row["frame_id"], "task_id": row["task_id"], "union_area_fraction": row["union_area_fraction"], "flags": flags or ["largest_area_review_example"]})
    result = {"schema": "SAM3_PSEUDOMASK_screening_audit_v1", "decision": "SAM3_PSEUDOMASK_PILOT_ACCEPTED" if passed else "SAM3_PSEUDOMASK_PILOT_REJECTED", "scope": "exploratory pseudo-target construction only; not segmentation validation", "audit_coverage": {"frames": len(details), "tasks": len({row["task_id"] for row in details}), "trajectories": len({row["episode_id"] for row in details})}, "descriptive_metrics": {"union_mask_area_fraction": {"min": float(area.min()), "mean": float(area.mean()), "median": float(np.median(area)), "p95": p95, "max": maximum}, "nonempty_union_fraction": nonempty, "connected_components": {"mean": float(np.mean([row["union_components"] for row in details])), "p95": float(np.percentile([row["union_components"] for row in details], 95))}, "gripper_prompt_nonempty_rate": float(np.mean([row["gripper_prompt_nonempty"] for row in details])), "broad_workspace_area_proxy_count": int(sum(row["broad_workspace_proxy"] for row in details)), "temporal_selected_window_iou": {"median": float(np.median([row["union_iou"] for row in temporal])), "mean": float(np.mean([row["union_iou"] for row in temporal])), "pairs": len(temporal)}, "temporal_absolute_area_delta": {"mean": float(np.mean([row["absolute_area_delta"] for row in temporal])), "p95": float(np.percentile([row["absolute_area_delta"] for row in temporal], 95))}}, "frozen_screening": {"nonempty_union_fraction_min": audit_acceptance["nonempty_union_fraction_min"], "p95_union_area_fraction_max": audit_acceptance["p95_union_area_fraction_max"], "max_union_area_fraction_max": audit_acceptance["max_union_area_fraction_max"]}, "qualitative_limitations": ["Visible gripper coverage is represented only by nonempty gripper-prompt output; no coverage accuracy is measured.", "Manipulated-object absorption has no quantitative determination without a separate reference; existing ten-frame prelabel review is descriptive support only.", "No IoU, recall, precision, or segmentation-quality claim is made."], "review_examples": failures, "per_frame": details, "temporal_pairs": temporal}
    dump(OUT / "pseudomask_screening_audit.json", result)
    report = ["# SAM3_PSEUDOMASK screening report", "", f"Decision: `{result['decision']}`", "", "This is a descriptive suitability screen for exploratory pseudo-target construction, not a segmentation validation.", "", f"- Audit coverage: {len(details)} frames / 10 tasks / 20 trajectories.", f"- Union area: mean {area.mean():.4f}, p95 {p95:.4f}, max {maximum:.4f}.", f"- Nonempty union rate: {nonempty:.3f}; gripper-prompt nonempty rate: {result['descriptive_metrics']['gripper_prompt_nonempty_rate']:.3f}.", f"- Broad-workspace area proxy: {result['descriptive_metrics']['broad_workspace_area_proxy_count']} frames.", "", "No human label, verified mask, or real segmentation metric is used or claimed. Contact sheets and the ten-frame pilot remain qualitative review aids."]
    (OUT / "pseudomask_quality_report.md").write_text("\n".join(report) + "\n")
    failure_lines = ["# SAM3_PSEUDOMASK screening examples", "", "These are review prompts, not labeled failure adjudications.", ""]
    for item in failures: failure_lines.append(f"- `{item['frame_id']}`: area={item['union_area_fraction']:.4f}; flags={', '.join(item['flags'])}")
    (OUT / "pseudomask_failure_examples.md").write_text("\n".join(failure_lines) + "\n")
    append_log("P2_SAM3_PSEUDOMASK_screening_complete", decision=result["decision"], audit_frames=len(details), p95_area=p95, nonempty_rate=nonempty)
    return {"decision": result["decision"], "frames": len(details), "p95": p95, "nonempty": nonempty}


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, sort_keys=True))
