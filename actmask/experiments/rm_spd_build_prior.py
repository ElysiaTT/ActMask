"""S2: derive conservative three-state robotness priors from S1 proposals.

No frame-difference, target feature, or future visual change is accessed here.
SAM3 proposals and their immediately preceding proposal maps are the only
pixel-level inputs to the prior construction.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

from actmask.experiments.rm_spd_common import OUT, append_log, iter_jsonl, require_preregistered_config, sha256_file, write_json, write_jsonl


STATE_SCENE_TRUSTED = 0
STATE_UNCERTAIN = 1
STATE_ROBOT_CORE = 2


def _proposal_probability(record: dict[str, Any], config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Return score-weighted prompt consensus and its binary prompt union."""

    payload = np.load(OUT / record["raw_proposals_path"], allow_pickle=False)
    maps: list[np.ndarray] = []
    unions: list[np.ndarray] = []
    for prompt in config["sam3"]["prompts"]:
        slug = str(prompt).lower().replace(" ", "_")
        masks = payload[f"{slug}__masks"].astype(bool)
        scores = np.clip(payload[f"{slug}__scores"].astype(np.float32), 0.0, 1.0)
        retained = payload[f"{slug}__retained_indices"].astype(np.int64)
        if len(retained):
            weighted = masks[retained].astype(np.float32) * scores[retained, None, None]
            maps.append(weighted.max(axis=0))
            unions.append(masks[retained].any(axis=0))
        else:
            height, width = record["image_shape"]
            maps.append(np.zeros((height, width), dtype=np.float32))
            unions.append(np.zeros((height, width), dtype=bool))
    return np.mean(np.stack(maps), axis=0).astype(np.float32), np.any(np.stack(unions), axis=0)


def _temporal_iou(current: np.ndarray, previous: np.ndarray | None) -> tuple[float | None, bool]:
    if previous is None:
        return None, False
    union = np.logical_or(current, previous).sum()
    return (float(np.logical_and(current, previous).sum() / union) if union else 1.0), True


def _regions(probability: np.ndarray, union: np.ndarray, temporal_iou: float | None, config: dict[str, Any]) -> np.ndarray:
    settings = config["prior"]
    core_cfg, uncertain_cfg = settings["robot_core"], settings["uncertain"]
    stable = temporal_iou is not None and temporal_iou >= float(core_cfg["temporal_iou_min"])
    erode_kernel = np.ones((int(core_cfg["erosion_kernel"]), int(core_cfg["erosion_kernel"])), dtype=np.uint8)
    core = probability >= float(core_cfg["robotness_min"])
    if stable:
        core = cv2.erode(core.astype(np.uint8), erode_kernel, iterations=1).astype(bool)
    else:
        core = np.zeros_like(core, dtype=bool)
    ring_radius = int(uncertain_cfg["boundary_ring_radius"])
    ring_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ring_radius + 1, 2 * ring_radius + 1))
    expanded = cv2.dilate(union.astype(np.uint8), ring_kernel, iterations=1).astype(bool)
    eroded_union = cv2.erode(union.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), iterations=1).astype(bool)
    boundary_ring = expanded & ~eroded_union
    disagreement = (probability >= float(uncertain_cfg["robotness_low"])) & (probability < float(uncertain_cfg["robotness_high"]))
    unstable = np.zeros_like(union, dtype=bool)
    if temporal_iou is not None and temporal_iou < float(uncertain_cfg["temporal_iou_min"]):
        unstable = expanded
    uncertain = (boundary_ring | disagreement | unstable) & ~core
    state = np.full(probability.shape, STATE_SCENE_TRUSTED, dtype=np.uint8)
    state[uncertain] = STATE_UNCERTAIN
    state[core] = STATE_ROBOT_CORE
    return state


def _load_image(row: dict[str, Any]) -> Image.Image:
    video = cv2.VideoCapture(str(row["video_path"]))
    if not video.isOpened():
        raise RuntimeError(f"cannot open source video {row['video_path']}")
    try:
        video.set(cv2.CAP_PROP_POS_FRAMES, int(row["frame_index"]))
        ok, bgr = video.read()
        if not ok or bgr is None:
            raise RuntimeError(f"cannot decode source frame {row['frame_id']}")
        return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    finally:
        video.release()


def _contact_sheets(rows: list[dict[str, Any]], prior: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    output_dir = OUT / "prior_contact_sheets"
    output_dir.mkdir(exist_ok=True)
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["role"] == "future":
            by_task[str(row["task_id"])].append(row)
    results = []
    for task, task_rows in sorted(by_task.items()):
        chosen = sorted(task_rows, key=lambda r: (str(r["episode_id"]), int(r["frame_index"])))[:3]
        sheet = Image.new("RGB", (3 * 320, 270), "white")
        draw = ImageDraw.Draw(sheet)
        for index, row in enumerate(chosen):
            base = np.asarray(_load_image(row)).copy()
            record = prior[str(row["frame_id"])]
            state = np.asarray(Image.open(OUT / record["state_path"]).convert("L"))
            # red=core, yellow=uncertain; uncoloured pixels are scene-trusted.
            base[state == STATE_UNCERTAIN] = (0.55 * np.asarray([255, 210, 0]) + 0.45 * base[state == STATE_UNCERTAIN]).astype(np.uint8)
            base[state == STATE_ROBOT_CORE] = (0.60 * np.asarray([245, 50, 50]) + 0.40 * base[state == STATE_ROBOT_CORE]).astype(np.uint8)
            panel = Image.fromarray(base).resize((320, 240))
            x = index * 320
            sheet.paste(panel, (x, 0))
            draw.text((x + 3, 243), f"{row['episode_id'].split(':')[-1]} f{row['frame_index']} core={record['fractions']['robot_core']:.3f}", fill="black")
        path = output_dir / f"{task}.jpg"
        sheet.save(path, quality=92)
        results.append({"task_id": task, "path": str(path.relative_to(OUT)), "sha256": sha256_file(path), "panels": len(chosen)})
    return results


def run() -> dict[str, Any]:
    config = require_preregistered_config()
    proposals = list(iter_jsonl(OUT / "sam3_proposal_manifest.jsonl"))
    frames = list(iter_jsonl(OUT / "source_frame_manifest.jsonl"))
    if len(proposals) != 800 or {row["frame_id"] for row in proposals} != {row["frame_id"] for row in frames}:
        raise RuntimeError("S2 requires the complete frozen 800-frame S1 manifest")
    by_id = {str(row["frame_id"]): row for row in proposals}
    probability_dir, state_dir = OUT / "robotness_prior" / "probability", OUT / "robotness_prior" / "state"
    probability_dir.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)
    frame_lookup = {str(row["frame_id"]): row for row in frames}
    prior_records: list[dict[str, Any]] = []
    temporal_rows: list[dict[str, Any]] = []
    for frame_id in sorted(by_id):
        row = by_id[frame_id]
        source = frame_lookup[frame_id]
        probability, union = _proposal_probability(row, config)
        predecessor_id = str(source["sample_id"]) + f"__f{int(source['frame_index']) - 1:06d}"
        prior_union = None
        if predecessor_id in by_id:
            _, prior_union = _proposal_probability(by_id[predecessor_id], config)
        temporal_iou, temporal_available = _temporal_iou(union, prior_union)
        state = _regions(probability, union, temporal_iou, config)
        probability_path = probability_dir / f"{frame_id}.npz"
        state_path = state_dir / f"{frame_id}.png"
        np.savez_compressed(probability_path, probability=probability.astype(np.float16), proposal_union=union.astype(np.uint8), temporal_iou=np.asarray([-1.0 if temporal_iou is None else temporal_iou], dtype=np.float32))
        Image.fromarray(state, mode="L").save(state_path)
        fractions = {"robot_core": float((state == STATE_ROBOT_CORE).mean()), "uncertain": float((state == STATE_UNCERTAIN).mean()), "scene_trusted": float((state == STATE_SCENE_TRUSTED).mean())}
        prior_records.append({"schema": "rm-spd-robotness-prior-v1", "frame_id": frame_id, "sample_id": source["sample_id"], "episode_id": source["episode_id"], "task_id": source["task_id"], "frame_index": int(source["frame_index"]), "role": source["role"], "probability_path": str(probability_path.relative_to(OUT)), "state_path": str(state_path.relative_to(OUT)), "proposal_manifest_path": "sam3_proposal_manifest.jsonl", "proposal_probability_definition": config["prior"]["probability"], "temporal_predecessor_frame_id": predecessor_id if predecessor_id in by_id else None, "temporal_agreement_iou": temporal_iou, "temporal_available": temporal_available, "fractions": fractions, "semantics": "three-state uncertain robotness prior for loss routing and metric partitioning only; no target construction and no predictive input"})
        temporal_rows.append({"frame_id": frame_id, "task_id": source["task_id"], "role": source["role"], "temporal_available": temporal_available, "proposal_union_iou_with_previous_frame": temporal_iou, "prior_uses_future_target_change": False})
    write_jsonl(OUT / "robotness_prior_manifest.jsonl", prior_records)
    write_json(OUT / "robotness_prior_schema.json", {"schema": "rm-spd-robotness-prior-schema-v1", "states": {str(STATE_SCENE_TRUSTED): "scene_trusted", str(STATE_UNCERTAIN): "uncertain", str(STATE_ROBOT_CORE): "robot_core"}, "probability_dtype": "float16 on disk / float32 construction", "probability_semantics": config["prior"]["probability"], "prohibited": ["segmentation_label", "visual_change", "future_feature_target", "predictive_model_input"]})
    core = np.asarray([row["fractions"]["robot_core"] for row in prior_records])
    uncertain = np.asarray([row["fractions"]["uncertain"] for row in prior_records])
    trusted = np.asarray([row["fractions"]["scene_trusted"] for row in prior_records])
    occupied = core + uncertain
    gate = config["prior"]["unusable_gate"]
    usable = float(np.median(occupied)) <= float(gate["max_median_core_plus_uncertain_fraction"]) and float(np.percentile(occupied, 95)) <= float(gate["max_p95_core_plus_uncertain_fraction"])
    statistics = {"schema": "rm-spd-robotness-statistics-v1", "frames": len(prior_records), "fractions": {"robot_core": {"mean": float(core.mean()), "median": float(np.median(core)), "p95": float(np.percentile(core, 95)), "max": float(core.max())}, "uncertain": {"mean": float(uncertain.mean()), "median": float(np.median(uncertain)), "p95": float(np.percentile(uncertain, 95)), "max": float(uncertain.max())}, "scene_trusted": {"mean": float(trusted.mean()), "median": float(np.median(trusted)), "p05": float(np.percentile(trusted, 5)), "min": float(trusted.min())}, "core_plus_uncertain": {"mean": float(occupied.mean()), "median": float(np.median(occupied)), "p95": float(np.percentile(occupied, 95)), "max": float(occupied.max())}}, "decision": "RM_SPD_PRIOR_USABLE" if usable else "RM_SPD_PRIOR_UNUSABLE", "gate": gate, "qualitative_boundary": "High precision is prioritized over full robot recall. This remains an unverified SAM3-derived prior, not segmentation quality evidence."}
    write_json(OUT / "robotness_statistics.json", statistics)
    available = np.asarray([row["proposal_union_iou_with_previous_frame"] for row in temporal_rows if row["temporal_available"]], dtype=np.float64)
    temporal = {"schema": "rm-spd-temporal-consistency-audit-v1", "frames": len(temporal_rows), "temporally_comparable_frames": int(len(available)), "agreement_iou": {"mean": float(available.mean()) if len(available) else None, "median": float(np.median(available)) if len(available) else None, "p05": float(np.percentile(available, 5)) if len(available) else None}, "uses_optical_flow": False, "uses_future_target_change": False, "per_frame": temporal_rows}
    write_json(OUT / "temporal_consistency_audit.json", temporal)
    contact_sheets = _contact_sheets(frames, {row["frame_id"]: row for row in prior_records})
    write_json(OUT / "prior_contact_sheet_manifest.json", {"schema": "rm-spd-prior-contact-sheets-v1", "sheets": contact_sheets, "legend": "red=conservative robot_core; yellow=uncertain; uncoloured=scene_trusted"})
    report = ["# RM-SPD SAM3 prior report", "", f"Decision: `{'RM_SPD_PRIOR_USABLE' if usable else 'RM_SPD_PRIOR_UNUSABLE'}`", "", "The prior was constructed only from fixed SAM3 proposals and immediately preceding proposal maps. It did not inspect future target features, frame differences, or human annotations.", "", f"- Frames: {len(prior_records)}", f"- Core + uncertain fraction: median {np.median(occupied):.4f}, p95 {np.percentile(occupied, 95):.4f}", f"- Scene-trusted fraction: mean {trusted.mean():.4f}", f"- Comparable temporal proposal pairs: {len(available)}", "", "SAM3 output is an uncertain loss-routing/metric-partitioning prior only. No segmentation label, object mask, or reference claim is made."]
    (OUT / "sam3_prior_report.md").write_text("\n".join(report) + "\n")
    append_log("S2_THREE_STATE_PRIOR_COMPLETE", frames=len(prior_records), decision=statistics["decision"], occupied_median=float(np.median(occupied)), occupied_p95=float(np.percentile(occupied, 95)))
    return {"decision": statistics["decision"], "frames": len(prior_records), "occupied_median": float(np.median(occupied)), "occupied_p95": float(np.percentile(occupied, 95))}


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, sort_keys=True))
