#!/usr/bin/env python3
"""Generate fixed SAM3_PSEUDOMASK artifacts for frozen RM-SAM3 pilot windows."""
from __future__ import annotations

import hashlib
import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "actmask" / "rm_sam3_acd_pilot"
SAM3_ROOT = Path("/data/projects/tzh/papers/poke_and_splat/sam3")
PAPERS_SITE = Path("/home/tzh/conda_envs/papers/lib/python3.11/site-packages")
PROMPTS = (("robot arm", "robot_arm"), ("robotic gripper", "robotic_gripper"))


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


def require_config() -> dict[str, Any]:
    path = OUT / "preregistered_config.json"
    actual = sha256(path); expected = (OUT / "preregistered_config.sha256").read_text().strip()
    if actual != expected:
        raise RuntimeError("SAM3_PSEUDOMASK preregistered configuration hash mismatch")
    config = json.loads(path.read_text())
    if config.get("branch_tag") != "SAM3_PSEUDOMASK":
        raise RuntimeError("wrong branch configuration")
    return config


def frame_id(row: dict[str, Any]) -> str:
    return f"{row['task_id']}__ep{int(row['episode_index']):06d}__f{int(row['frame_index']):06d}"


def patch_sam3_fused_mlp(torch: Any) -> None:
    import torch.nn.functional as functional
    import sam3.model.vitdet as vitdet
    def safe_addmm_act(activation: Any, linear: Any, mat1: Any) -> Any:
        output = linear(mat1)
        if activation in {functional.relu, torch.nn.ReLU}: return functional.relu(output)
        if activation in {functional.gelu, torch.nn.GELU}: return functional.gelu(output)
        if isinstance(activation, type) and issubclass(activation, torch.nn.Module): return activation()(output)
        raise ValueError(f"unexpected SAM3 activation {activation}")
    vitdet.addmm_act = safe_addmm_act


def audit_selection(rows: list[dict[str, Any]]) -> list[str]:
    """Exactly 10 tasks * 2 trajectories * 5 deterministic window frames."""
    grouped: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[row["task_id"]][int(row["episode_index"])] .append(row)
    result: list[str] = []
    for task, episodes in sorted(grouped.items()):
        episode_ids = sorted(episodes)
        picks = (episode_ids[0], episode_ids[-1])
        for episode in picks:
            candidates = sorted(episodes[episode], key=lambda row: int(row["frame_index"]))
            indexes = np.linspace(0, len(candidates) - 1, 5, dtype=int).tolist()
            result.extend(frame_id(candidates[index]) for index in indexes)
    if len(result) != 100 or len(set(result)) != 100:
        raise RuntimeError("SAM3_PSEUDOMASK audit selection must contain 100 unique frames")
    return result


def load_progress(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file(): return {}
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    return {row["frame_id"]: row for row in rows}


def make_overlay(image: Image.Image, mask: np.ndarray, title: str, output: Path) -> None:
    rgb = np.asarray(image).copy(); active = mask.astype(bool)
    rgb[active] = (0.55 * np.array([255, 70, 70]) + 0.45 * rgb[active]).astype(np.uint8)
    rendered = Image.fromarray(rgb)
    draw = ImageDraw.Draw(rendered); draw.rectangle((0, 0, min(500, rendered.width), 20), fill="white")
    draw.text((3, 3), title, fill="black")
    output.parent.mkdir(parents=True, exist_ok=True); rendered.save(output, quality=93)


def contact_sheets(audit_rows: list[dict[str, Any]], records: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    target = OUT / "pseudomask_contact_sheets"; target.mkdir(exist_ok=True)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in audit_rows: grouped[row["task_id"]].append(row)
    result = []
    for task, task_rows in sorted(grouped.items()):
        sheet = Image.new("RGB", (5 * 256, 2 * 212), "white"); draw = ImageDraw.Draw(sheet)
        for index, row in enumerate(sorted(task_rows, key=lambda r: (r["episode_index"], r["frame_index"]))):
            record = records[frame_id(row)]
            image = Image.open(OUT / record["overlay_path"]).convert("RGB").resize((256, 192))
            x, y = (index % 5) * 256, (index // 5) * 212
            sheet.paste(image, (x, y)); draw.text((x + 3, y + 194), f"ep{row['episode_index']:06d} f{row['frame_index']:06d}", fill="black")
        path = target / f"{task}.jpg"; sheet.save(path, quality=93)
        result.append({"task_id": task, "path": str(path.relative_to(OUT)), "sha256": sha256(path), "audit_frames": len(task_rows)})
    return result


def run() -> dict[str, Any]:
    config = require_config(); sam3 = config["sam3"]
    rows = [json.loads(line) for line in (OUT / "pseudomask_window_manifest.jsonl").read_text().splitlines() if line]
    expected = {frame_id(row): row for row in rows}
    if len(expected) != len(rows): raise RuntimeError("duplicate frozen pseudomask frame IDs")
    audit_ids = set(audit_selection(rows)); audit_rows = [expected[item] for item in sorted(audit_ids)]
    dump(OUT / "pseudomask_audit_selection.json", {"schema": "SAM3_PSEUDOMASK_audit_selection_v1", "count": len(audit_rows), "tasks": 10, "trajectories": 20, "selection": "per task: smallest and largest frozen episode index, five evenly spaced required target-window frames each", "frames": audit_rows})
    root = OUT / "pseudomasks"; union_dir, arm_dir, gripper_dir, raw_dir, overlay_dir = root / "union", root / "robot_arm_prompt_union", root / "robotic_gripper_prompt_union", root / "raw_prompt_outputs", OUT / "pseudomask_audit_overlays"
    for path in (union_dir, arm_dir, gripper_dir, raw_dir, overlay_dir): path.mkdir(parents=True, exist_ok=True)
    progress_path = OUT / "pseudomask_progress.jsonl"; records = load_progress(progress_path)
    for item, row in expected.items():
        required = (union_dir / f"{item}.png", arm_dir / f"{item}.png", gripper_dir / f"{item}.png", raw_dir / f"{item}.npz")
        if item in records and all(path.is_file() for path in required): continue
        records.pop(item, None)
    remaining = [row for item, row in sorted(expected.items()) if item not in records]
    if remaining:
        for path in (PAPERS_SITE, SAM3_ROOT):
            if str(path) not in sys.path: sys.path.insert(0, str(path))
        import torch
        if not torch.cuda.is_available(): raise RuntimeError("SAM3_PSEUDOMASK generation requires CUDA")
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        patch_sam3_fused_mlp(torch); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        model = build_sam3_image_model(device="cuda", checkpoint_path=sam3["checkpoint"], load_from_HF=False, compile=False)
        processor = Sam3Processor(model, device="cuda", confidence_threshold=float(sam3["processor_confidence_threshold"]))
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in remaining: grouped[row["video_path"]].append(row)
        total, completed = len(expected), len(records); start = time.monotonic()
        with progress_path.open("a") as handle:
            for video_path, video_rows in sorted(grouped.items()):
                capture = cv2.VideoCapture(video_path)
                if not capture.isOpened(): raise RuntimeError(f"cannot open frozen source video {video_path}")
                for row in sorted(video_rows, key=lambda value: int(value["frame_index"])):
                    item = frame_id(row); capture.set(cv2.CAP_PROP_POS_FRAMES, int(row["frame_index"])); ok, bgr = capture.read()
                    if not ok: raise RuntimeError(f"cannot decode {video_path} frame {row['frame_index']}")
                    image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)); state = processor.set_image(image)
                    prompt_masks: dict[str, np.ndarray] = {}; raw_payload: dict[str, np.ndarray] = {}; prompt_info = []
                    for prompt, slug in PROMPTS:
                        output = processor.set_text_prompt(prompt=prompt, state=state)
                        masks, scores = output.get("masks"), output.get("scores")
                        masks = masks.detach().cpu().numpy() if isinstance(masks, torch.Tensor) else np.asarray(masks)
                        scores = scores.detach().cpu().numpy() if isinstance(scores, torch.Tensor) else np.asarray(scores)
                        if masks.ndim == 4: masks = masks[:, 0]
                        if masks.ndim != 3: masks = np.zeros((0, image.height, image.width), dtype=bool)
                        order = np.argsort(-scores) if len(scores) else np.asarray([], dtype=np.int64)
                        kept = [int(index) for index in order.tolist() if int(masks[index].astype(bool).sum()) >= int(sam3["minimum_instance_area_pixels"])][:int(sam3["raw_instance_cap_per_prompt"])]
                        selected = masks[kept].astype(np.uint8) if kept else np.zeros((0, image.height, image.width), dtype=np.uint8)
                        union = np.any(selected.astype(bool), axis=0) if len(selected) else np.zeros((image.height, image.width), dtype=bool)
                        prompt_masks[slug] = union; raw_payload[f"{slug}_masks"] = selected; raw_payload[f"{slug}_scores"] = scores.astype(np.float32); raw_payload[f"{slug}_retained_indices"] = np.asarray(kept, dtype=np.int32)
                        prompt_info.append({"prompt": prompt, "slug": slug, "raw_instances": int(len(scores)), "raw_scores": [float(value) for value in scores.tolist()], "retained_indices": kept, "retained_instances": len(kept), "retained_area_pixels": [int(masks[index].astype(bool).sum()) for index in kept]})
                    union = prompt_masks["robot_arm"] | prompt_masks["robotic_gripper"]
                    union_path, arm_path, gripper_path, raw_path = union_dir / f"{item}.png", arm_dir / f"{item}.png", gripper_dir / f"{item}.png", raw_dir / f"{item}.npz"
                    Image.fromarray((union.astype(np.uint8) * 255), mode="L").save(union_path); Image.fromarray((prompt_masks["robot_arm"].astype(np.uint8) * 255), mode="L").save(arm_path); Image.fromarray((prompt_masks["robotic_gripper"].astype(np.uint8) * 255), mode="L").save(gripper_path); np.savez_compressed(raw_path, **raw_payload)
                    record = {"schema": "SAM3_PSEUDOMASK_window_v1", "frame_id": item, "episode_id": row["episode_id"], "task_id": row["task_id"], "episode_index": row["episode_index"], "frame_index": row["frame_index"], "source_video": row["video_path"], "prompt_outputs": prompt_info, "union_mask_path": str(union_path.relative_to(OUT)), "robot_arm_prompt_union_path": str(arm_path.relative_to(OUT)), "robotic_gripper_prompt_union_path": str(gripper_path.relative_to(OUT)), "raw_prompt_output_path": str(raw_path.relative_to(OUT)), "union_area_pixels": int(union.sum()), "union_area_fraction": float(union.mean()), "postprocessing": sam3["postprocessing"], "semantics": "SAM3_PSEUDOMASK only; not label/reference/ground-truth/verified segmentation"}
                    if item in audit_ids:
                        overlay_path = overlay_dir / f"{item}.jpg"; make_overlay(image, union, f"SAM3_PSEUDOMASK {item}", overlay_path); record["overlay_path"] = str(overlay_path.relative_to(OUT))
                    handle.write(json.dumps(record, sort_keys=True) + "\n"); handle.flush(); records[item] = record; completed += 1
                    print(f"SAM3_PSEUDOMASK={completed}/{total} frame={item}", flush=True)
                capture.release()
        runtime = {"device": "cuda", "inference_seconds_excluding_model_init": round(time.monotonic() - start, 3), "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()), "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()), "cuda_device_name": torch.cuda.get_device_name(0), "torch": torch.__version__}
        del processor, model; torch.cuda.empty_cache()
    else:
        runtime = {"device": "cuda", "status": "resumed_with_no_remaining_windows"}
    ordered = [records[item] for item in sorted(expected)]
    if len(ordered) != len(expected): raise RuntimeError("incomplete SAM3_PSEUDOMASK manifest")
    (OUT / "pseudomask_manifest.jsonl").write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in ordered))
    sheets = contact_sheets(audit_rows, records)
    areas = np.asarray([record["union_area_fraction"] for record in ordered], dtype=np.float64)
    dump(OUT / "pseudomask_runtime.json", {"schema": "SAM3_PSEUDOMASK_runtime_v1", "config_sha256": sha256(OUT / "preregistered_config.json"), "windows": len(ordered), "runtime": runtime, "checkpoint_sha256": sam3["checkpoint_sha256"]})
    dump(OUT / "pseudomask_generation_summary.json", {"schema": "SAM3_PSEUDOMASK_generation_summary_v1", "windows": len(ordered), "tasks": len({record["task_id"] for record in ordered}), "episodes": len({record["episode_id"] for record in ordered}), "union_area_fraction": {"min": float(areas.min()), "mean": float(areas.mean()), "median": float(np.median(areas)), "p95": float(np.percentile(areas, 95)), "max": float(areas.max())}, "contact_sheets": sheets, "semantics": "Fixed SAM3_PSEUDOMASK artifacts only; not segmentation labels or verified masks."})
    append_log("P1_SAM3_PSEUDOMASK_generation_complete", windows=len(ordered), audit_frames=len(audit_rows), contact_sheets=len(sheets), mean_area=float(areas.mean()))
    return {"windows": len(ordered), "audit_frames": len(audit_rows), "mean_area": float(areas.mean())}


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, sort_keys=True))
