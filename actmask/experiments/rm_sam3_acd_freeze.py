#!/usr/bin/env python3
"""Freeze the independent RM-SAM3-ACD-PILOT protocol and its exact windows."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import cv2

from actmask.data.robomind_adapter import RoboMINDRecord, load_numeric_episode, validate_numeric_episode


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "actmask" / "rm_sam3_acd_pilot"
RM = ROOT / "outputs" / "actmask" / "rm_series_robomind"
SAM3_ROOT = Path("/data/projects/tzh/papers/poke_and_splat/sam3")
SAM3_CHECKPOINT = SAM3_ROOT / "checkpoint" / "sam3.pt"
HISTORY_STEPS, ACTION_STEPS = 4, 4
HORIZONS = {"short": 8, "medium": 15, "long": 30}
ANCHOR_FRACTIONS = (0.12, 0.22, 0.32, 0.42, 0.52, 0.62, 0.72, 0.82)
LOCAL_SAM3_COMMIT = "8e451d5eb43c817b64ae7577fb7b9ae223db88a9"


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
    previous, sequence = "", 0
    if path.is_file() and path.stat().st_size:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        previous, sequence = rows[-1]["event_sha256"], int(rows[-1]["seq"])
    row = {"seq": sequence + 1, "utc": datetime.now(timezone.utc).isoformat(), "event": event, "payload": payload, "prev_event_sha256": previous}
    row["event_sha256"] = hashlib.sha256(json.dumps(row, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    with path.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def command(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def record(row: dict[str, Any]) -> RoboMINDRecord:
    return RoboMINDRecord(**{field: row[field] for field in RoboMINDRecord.__dataclass_fields__})


def anchors(steps: int) -> list[int]:
    lower, upper = HISTORY_STEPS, steps - ACTION_STEPS - max(HORIZONS.values())
    if upper - lower + 1 < len(ANCHOR_FRACTIONS):
        raise RuntimeError(f"too few legal anchors for {steps} source frames")
    chosen: list[int] = []
    for fraction in ANCHOR_FRACTIONS:
        candidate = int(round(lower + fraction * (upper - lower)))
        candidate = max(lower, min(upper, candidate))
        while candidate in chosen and candidate < upper:
            candidate += 1
        if candidate in chosen:
            candidate = next(index for index in range(lower, upper + 1) if index not in chosen)
        chosen.append(candidate)
    return sorted(chosen)


def run(*, allow_preinference_revision_correction: bool = False) -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    subset = json.loads((RM / "source_subset" / "subset_manifest.json").read_text())
    selected = list(subset["episodes_detail"])
    if len(selected) != 100 or len({row["task_id"] for row in selected}) != 10:
        raise RuntimeError("expected exact frozen RM-Series 100-episode / 10-task source")
    inherited = {row["local_path"]: row for row in json.loads((RM / "source_hash_manifest.json").read_text())["files"]}
    sources = []
    for item in selected:
        for modality, key in (("parquet_rgb_numeric", "parquet_path"), ("front_rgb_video", "front_video_path")):
            prior = inherited.get(item[key])
            if prior is None:
                raise RuntimeError(f"missing inherited source hash for {item[key]}")
            sources.append({"episode_id": item["episode_id"], "task_id": item["task_id"], "modality": modality, "path": item[key], "bytes": int(prior["bytes"]), "sha256": prior["sha256"], "hash_origin": "rm_series_robomind/source_hash_manifest.json"})
    if not SAM3_CHECKPOINT.is_file():
        raise RuntimeError(f"missing fixed local SAM3 checkpoint: {SAM3_CHECKPOINT}")
    sample = cv2.VideoCapture(selected[0]["front_video_path"])
    resolution = [int(sample.get(cv2.CAP_PROP_FRAME_WIDTH)), int(sample.get(cv2.CAP_PROP_FRAME_HEIGHT))]
    fps = float(sample.get(cv2.CAP_PROP_FPS)); sample.release()
    source_revision = command(["git", "-c", f"safe.directory={SAM3_ROOT}", "-C", str(SAM3_ROOT), "rev-parse", "HEAD"]) or LOCAL_SAM3_COMMIT
    config = {
        "schema": "SAM3_PSEUDOMASK_rm_sam3_acd_pilot_config_v1",
        "branch_tag": "SAM3_PSEUDOMASK",
        "source": {"dataset": "RoboMIND2.0", "selection": "exact inherited RM-Series 100-trajectory balanced Franka subset", "episodes": 100, "tasks": sorted({row["task_id"] for row in selected}), "source_hash_manifest": "source_hash_manifest.json"},
        "sam3": {"source_root": str(SAM3_ROOT), "source_revision": source_revision, "checkpoint": str(SAM3_CHECKPOINT), "checkpoint_bytes": SAM3_CHECKPOINT.stat().st_size, "checkpoint_sha256": sha256(SAM3_CHECKPOINT), "prompts": ["robot arm", "robotic gripper"], "processor_confidence_threshold": 0.20, "minimum_instance_area_pixels": 64, "raw_instance_cap_per_prompt": 8, "postprocessing": "fixed per-prompt union of all retained SAM3 instances, then fixed union across prompts; no dilation, optical flow, frame difference, future observation, or manipulated-object-motion expansion", "input_resolution": resolution, "source_fps": fps},
        "temporal": {"history_steps": HISTORY_STEPS, "action_steps": ACTION_STEPS, "anchors_per_episode": len(ANCHOR_FRACTIONS), "anchor_fractions": list(ANCHOR_FRACTIONS), "horizons_steps": HORIZONS, "pre_rgb": "frame anchor-1", "future_target_frame": "anchor + action_steps + horizon_steps - 1"},
        "target": {"name": "SAM3_PSEUDOMASK robot-excluded dense 2D affected-region target", "resolution": [160, 120], "robot_pseudomask_use": "target construction/evaluation only; forbidden as fair predictive model input", "fair_inputs": ["pre-anchor RGB history", "pre-anchor robot-state history", "future action chunk"], "forbidden_inputs": ["SAM3 pseudo-mask", "future RGB/depth/state", "future target", "episode ID", "source path"]},
        "screening_acceptance": {"all_outputs_present": True, "nonempty_union_fraction_min": 0.99, "p95_union_area_fraction_max": 0.30, "max_union_area_fraction_max": 0.40, "broad_workspace_proxy_area_fraction": 0.30, "manual_visual_support": "existing ten-frame SAM3_PSEUDOMASK pilot is descriptive support only; no label/reference claim"},
        "resource_budget": {"generated_masks_only_for_frozen_target_windows": True, "max_gpu_hours_before_pseudotarget_acceptance": 8, "max_new_bytes_before_target_acceptance": 8 * 1024**3},
        "claim_boundary": "Exploratory SAM3_PSEUDOMASK target pilot only. It is neither verified robot segmentation nor a human-labeled/validated real-robot benchmark. It makes no physical counterfactual claim.",
        "no_downstream_tuning": "All SAM3 and screening parameters above are fixed before pseudo-target construction or baseline evaluation.",
    }
    config_path = OUT / "preregistered_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        prior = json.loads(config_path.read_text())
        only_revision = prior.get("sam3", {}).get("source_revision") is None and {**prior, "sam3": {**prior.get("sam3", {}), "source_revision": source_revision}} == config
        no_masks = not (OUT / "pseudomask_manifest.jsonl").exists()
        if not (allow_preinference_revision_correction and only_revision and no_masks):
            raise RuntimeError("refusing to alter frozen SAM3_PSEUDOMASK pilot protocol")
        dump(OUT / "protocol_amendment_before_p1.json", {"schema": "SAM3_PSEUDOMASK_protocol_amendment_v1", "allowed_before_p1": True, "reason": "P0 source-revision probe did not bypass local Git safe-directory ownership; only missing commit field is corrected before any pseudomask generation.", "old_source_revision": None, "new_source_revision": source_revision})
    dump(config_path, config); (OUT / "preregistered_config.sha256").write_text(sha256(config_path) + "\n")
    dump(OUT / "selected_trajectory_manifest.json", {"schema": "SAM3_PSEUDOMASK_selected_trajectories_v1", "episodes": selected, "episode_count": 100, "task_count": 10, "source": "exact inherited RM-Series selection"})
    dump(OUT / "source_hash_manifest.json", {"schema": "SAM3_PSEUDOMASK_source_hash_manifest_v1", "files": sources, "all_hashes_inherited": True})
    samples, windows, episode_audit = [], {}, []
    for raw in selected:
        item = record(raw); numeric = load_numeric_episode(item); alignment = validate_numeric_episode(item, numeric)
        if not alignment["pass"]:
            raise RuntimeError(f"invalid frozen numeric source: {item.episode_id}")
        caps = cv2.VideoCapture(item.front_video_path or "")
        frames = int(caps.get(cv2.CAP_PROP_FRAME_COUNT)); caps.release()
        if frames != alignment["steps"]:
            raise RuntimeError(f"front video/numeric mismatch: {item.episode_id}")
        episode_audit.append({"episode_id": item.episode_id, "task_id": item.task_id, "steps": frames, "numeric_alignment": alignment})
        for anchor_index, anchor in enumerate(anchors(frames)):
            pre = anchor - 1
            windows.setdefault((item.episode_id, pre), {"episode_id": item.episode_id, "task_id": item.task_id, "episode_index": item.episode_index, "frame_index": pre, "video_path": item.front_video_path, "role": "pre_anchor", "anchor_indices": []})["anchor_indices"].append(anchor)
            for horizon_name, horizon in HORIZONS.items():
                future = anchor + ACTION_STEPS + horizon - 1
                windows.setdefault((item.episode_id, future), {"episode_id": item.episode_id, "task_id": item.task_id, "episode_index": item.episode_index, "frame_index": future, "video_path": item.front_video_path, "role": "future", "anchor_indices": []})["anchor_indices"].append(anchor)
                samples.append({"sample_id": f"{item.task_id}__ep{item.episode_index:06d}__a{anchor:04d}__{horizon_name}", "episode_id": item.episode_id, "task_id": item.task_id, "episode_index": item.episode_index, "anchor_index": anchor, "anchor_ordinal": anchor_index, "pre_frame_index": pre, "history_frame_indices": list(range(anchor - HISTORY_STEPS, anchor)), "action_frame_indices": list(range(anchor, anchor + ACTION_STEPS)), "future_frame_index": future, "horizon": horizon_name, "horizon_steps": horizon, "video_path": item.front_video_path, "parquet_path": item.parquet_path})
    window_rows = sorted(windows.values(), key=lambda row: (row["task_id"], row["episode_index"], row["frame_index"]))
    (OUT / "target_window_manifest.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in samples))
    (OUT / "pseudomask_window_manifest.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in window_rows))
    dump(OUT / "source_alignment_audit.json", {"schema": "SAM3_PSEUDOMASK_source_alignment_audit_v1", "episodes": episode_audit, "all_pass": all(row["numeric_alignment"]["pass"] for row in episode_audit), "all_front_video_aligned": all(row["steps"] == row["numeric_alignment"]["steps"] for row in episode_audit)})
    env = {name: package_version(name) for name in ("torch", "torchvision", "timm", "ftfy", "iopath", "numpy", "opencv-python", "Pillow")}
    dump(OUT / "runtime_environment.json", {"schema": "SAM3_PSEUDOMASK_runtime_environment_v1", "python": sys.version, "dependencies": env, "nvidia_smi": command(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]), "created_utc": datetime.now(timezone.utc).isoformat()})
    docs = OUT / "docs"; docs.mkdir(exist_ok=True)
    (docs / "plan.md").write_text("# RM-SAM3-ACD-PILOT plan\n\nThis isolated branch uses fixed SAM3_PSEUDOMASK proposals only to exclude likely robot pixels in exploratory future-change targets. It does not validate SAM3 segmentation and will not call pseudo-masks labels, references, or ground truth.\n")
    (docs / "results.md").write_text("# RM-SAM3-ACD-PILOT results\n\nP0 complete. P1 pseudo-mask generation is pending. No pseudo-target, baseline, or method has run.\n")
    (docs / "handoff.md").write_text("# RM-SAM3-ACD-PILOT handoff\n\nRead the preregistered configuration and window manifests. Only frozen selected target-window frames may be sent to SAM3.\n")
    append_log("P0_SAM3_PSEUDOMASK_protocol_frozen", config_sha256=sha256(config_path), samples=len(samples), pseudomask_windows=len(window_rows), episodes=100, tasks=10)
    return {"config_sha256": sha256(config_path), "target_samples": len(samples), "pseudomask_windows": len(window_rows), "episodes": 100}


if __name__ == "__main__":
    print(json.dumps(run(allow_preinference_revision_correction="--allow-preinference-revision-correction" in sys.argv), ensure_ascii=False, sort_keys=True))
