"""S1: fixed multi-prompt SAM3 proposal generation for RM-SPD.

The output is deliberately called a *proposal*.  It is never a segmentation
label and is not consumed by the predictive model as an input.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from actmask.experiments.rm_spd_common import OUT, append_log, iter_jsonl, require_preregistered_config, sha256_file, write_json, write_jsonl


PAPERS_SITE = Path("/data/projects/tzh/papers/poke_and_splat")
# The local SAM3 checkout was installed in this sibling Python-3.11 environment.
# Insert it *after* the active ActMask environment so torch/CUDA remain the
# explicitly frozen ActMask runtime; only absent SAM3 support modules resolve
# there.  This is recorded in the S1 runtime report.
SAM3_SUPPORT_SITE = Path("/home/tzh/conda_envs/papers/lib/python3.11/site-packages")


def _slug(prompt: str) -> str:
    return prompt.lower().replace(" ", "_")


def _patch_sam3_fused_mlp(torch: Any) -> None:
    """Compatibility patch required by this local SAM3 checkout on torch 2.6."""

    import torch.nn.functional as functional
    import sam3.model.vitdet as vitdet

    def safe_addmm_act(activation: Any, linear: Any, mat1: Any) -> Any:
        output = linear(mat1)
        if activation in {functional.relu, torch.nn.ReLU}:
            return functional.relu(output)
        if activation in {functional.gelu, torch.nn.GELU}:
            return functional.gelu(output)
        if isinstance(activation, type) and issubclass(activation, torch.nn.Module):
            return activation()(output)
        raise ValueError(f"unexpected local SAM3 activation: {activation}")

    vitdet.addmm_act = safe_addmm_act


def _load_progress(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    return {row["frame_id"]: row for row in iter_jsonl(path)}


def _required(record: dict[str, Any]) -> bool:
    return all((OUT / relative).is_file() for relative in (record["raw_proposals_path"], record["prompt_union_path"], record["proposal_union_path"]))


def _write_preview(image: Image.Image, union: np.ndarray, path: Path, title: str) -> None:
    visible = np.asarray(image).copy()
    visible[union] = (0.55 * np.asarray([255, 72, 72]) + 0.45 * visible[union]).astype(np.uint8)
    rendered = Image.fromarray(visible)
    # The filename carries source identity; this short text makes contact sheets self-describing.
    from PIL import ImageDraw

    draw = ImageDraw.Draw(rendered)
    draw.rectangle((0, 0, min(620, rendered.width), 21), fill="white")
    draw.text((3, 3), title, fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered.save(path, quality=92)


def run(*, max_frames: int | None = None) -> dict[str, Any]:
    config = require_preregistered_config()
    sam3 = config["sam3"]
    rows = list(iter_jsonl(OUT / "source_frame_manifest.jsonl"))
    expected = {str(row["frame_id"]): row for row in rows}
    if len(rows) != 800 or len(expected) != len(rows):
        raise RuntimeError("RM-SPD S1 requires exactly the frozen 800 unique source frames")
    base = OUT / "sam3_proposals"
    raw_dir, prompt_union_dir, union_dir, preview_dir = base / "raw", base / "prompt_unions", base / "unions", base / "previews"
    for directory in (raw_dir, prompt_union_dir, union_dir, preview_dir):
        directory.mkdir(parents=True, exist_ok=True)
    progress_path = OUT / "sam3_proposal_progress.jsonl"
    records = _load_progress(progress_path)
    records = {key: value for key, value in records.items() if key in expected and _required(value)}
    remaining = [expected[key] for key in sorted(expected) if key not in records]
    selected = remaining if max_frames is None else remaining[:max_frames]
    if selected:
        for path in (PAPERS_SITE, Path(str(sam3["source_root"]))):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
        if str(SAM3_SUPPORT_SITE) not in sys.path:
            sys.path.append(str(SAM3_SUPPORT_SITE))
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("RM-SPD S1 requires CUDA for the fixed SAM3 runtime")
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        _patch_sam3_fused_mlp(torch)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model_started = time.monotonic()
        model = build_sam3_image_model(device="cuda", checkpoint_path=str(sam3["checkpoint"]), load_from_HF=False, compile=False)
        processor = Sam3Processor(model, device="cuda", confidence_threshold=float(sam3["processor_confidence_threshold"]))
        model_seconds = time.monotonic() - model_started
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in selected:
            grouped[str(row["video_path"])].append(row)
        generation_started = time.monotonic()
        with progress_path.open("a") as handle:
            for video_path, video_rows in sorted(grouped.items()):
                video = cv2.VideoCapture(video_path)
                if not video.isOpened():
                    raise RuntimeError(f"cannot decode frozen source video: {video_path}")
                try:
                    for row in sorted(video_rows, key=lambda item: int(item["frame_index"])):
                        started = time.monotonic()
                        video.set(cv2.CAP_PROP_POS_FRAMES, int(row["frame_index"]))
                        ok, bgr = video.read()
                        if not ok or bgr is None:
                            raise RuntimeError(f"cannot decode frame {row['frame_index']} from {video_path}")
                        image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                        state = processor.set_image(image)
                        raw_payload: dict[str, np.ndarray] = {}
                        prompt_info: list[dict[str, Any]] = []
                        prompt_unions: list[np.ndarray] = []
                        for prompt in sam3["prompts"]:
                            slug = _slug(str(prompt))
                            output = processor.set_text_prompt(prompt=str(prompt), state=state)
                            masks, scores = output.get("masks"), output.get("scores")
                            masks = masks.detach().cpu().numpy() if isinstance(masks, torch.Tensor) else np.asarray(masks)
                            scores = scores.detach().cpu().numpy() if isinstance(scores, torch.Tensor) else np.asarray(scores)
                            if masks.ndim == 4:
                                masks = masks[:, 0]
                            if masks.ndim != 3:
                                masks = np.zeros((0, image.height, image.width), dtype=np.uint8)
                            masks = masks.astype(np.uint8, copy=False)
                            scores = scores.astype(np.float32, copy=False).reshape(-1)
                            if len(masks) != len(scores):
                                raise RuntimeError(f"SAM3 returned mismatched masks/scores for {row['frame_id']} {prompt}")
                            areas = masks.reshape(len(masks), -1).sum(axis=1) if len(masks) else np.zeros((0,), dtype=np.int64)
                            retained = [int(index) for index in np.argsort(-scores).tolist() if int(areas[index]) >= int(sam3["minimum_instance_area_pixels"])][: int(sam3["maximum_instances_per_prompt"])]
                            union = np.any(masks[retained].astype(bool), axis=0) if retained else np.zeros((image.height, image.width), dtype=bool)
                            # Keep raw model candidate masks and scores, as well as the fixed retained subset.
                            raw_payload[f"{slug}__masks"] = masks
                            raw_payload[f"{slug}__scores"] = scores
                            raw_payload[f"{slug}__areas"] = areas.astype(np.int32)
                            raw_payload[f"{slug}__retained_indices"] = np.asarray(retained, dtype=np.int32)
                            raw_payload[f"{slug}__retained_union"] = union.astype(np.uint8)
                            prompt_unions.append(union)
                            prompt_info.append({"prompt": prompt, "slug": slug, "raw_instances": int(len(scores)), "raw_scores": [float(value) for value in scores.tolist()], "retained_indices": retained, "retained_instances": len(retained), "retained_area_pixels": [int(areas[index]) for index in retained]})
                        stack = np.stack(prompt_unions, axis=0)
                        proposal_union = np.any(stack, axis=0)
                        frame_id = str(row["frame_id"])
                        raw_path = raw_dir / f"{frame_id}.npz"
                        prompt_path = prompt_union_dir / f"{frame_id}.npz"
                        union_path = union_dir / f"{frame_id}.png"
                        np.savez_compressed(raw_path, **raw_payload)
                        np.savez_compressed(prompt_path, prompts=np.asarray([_slug(str(p)) for p in sam3["prompts"]]), unions=stack.astype(np.uint8))
                        Image.fromarray((proposal_union.astype(np.uint8) * 255), mode="L").save(union_path)
                        record = {
                            "schema": "rm-spd-sam3-proposal-v1",
                            "frame_id": frame_id,
                            "sample_id": row["sample_id"],
                            "episode_id": row["episode_id"],
                            "task_id": row["task_id"],
                            "frame_index": int(row["frame_index"]),
                            "role": row["role"],
                            "source_video": row["video_path"],
                            "image_shape": [image.height, image.width],
                            "prompt_outputs": prompt_info,
                            "raw_proposals_path": str(raw_path.relative_to(OUT)),
                            "prompt_union_path": str(prompt_path.relative_to(OUT)),
                            "proposal_union_path": str(union_path.relative_to(OUT)),
                            "proposal_union_area_fraction": float(proposal_union.mean()),
                            "runtime_seconds": round(time.monotonic() - started, 5),
                            "model_revision": sam3["source_revision"],
                            "checkpoint_sha256": sam3["checkpoint_sha256"],
                            "semantics": "uncertain SAM3 robotness proposal only; not a label, target, reference, or predictive model input",
                        }
                        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                        handle.flush()
                        records[frame_id] = record
                        print(f"RM_SPD_SAM3={len(records)}/{len(expected)} frame={frame_id} seconds={record['runtime_seconds']}", flush=True)
                finally:
                    video.release()
        runtime = {
            "device": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "model_initialization_seconds": round(model_seconds, 4),
            "proposal_seconds": round(time.monotonic() - generation_started, 4),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "sam3_support_site": str(SAM3_SUPPORT_SITE),
        }
        del processor, model
        torch.cuda.empty_cache()
    else:
        runtime = {"status": "nothing_remaining"}
    ordered = [records[key] for key in sorted(expected) if key in records]
    write_jsonl(OUT / "sam3_proposal_manifest.jsonl", ordered)
    status = "complete" if len(ordered) == len(expected) else "partial"
    report = {
        "schema": "rm-spd-sam3-runtime-v1",
        "status": status,
        "config_sha256": sha256_file(OUT / "preregistered_config.json"),
        "expected_frames": len(expected),
        "completed_frames": len(ordered),
        "remaining_frames": len(expected) - len(ordered),
        "runtime": runtime,
        "checkpoint_sha256": sam3["checkpoint_sha256"],
        "proposal_policy": sam3["proposal_role"],
    }
    write_json(OUT / "sam3_runtime_report.json", report)
    if status == "complete":
        areas = np.asarray([row["proposal_union_area_fraction"] for row in ordered], dtype=np.float64)
        write_json(OUT / "sam3_proposal_summary.json", {"schema": "rm-spd-sam3-proposal-summary-v1", "frames": len(ordered), "tasks": len({row["task_id"] for row in ordered}), "area_fraction": {"min": float(areas.min()), "mean": float(areas.mean()), "median": float(np.median(areas)), "p95": float(np.percentile(areas, 95)), "max": float(areas.max())}, "raw_proposal_manifest_sha256": sha256_file(OUT / "sam3_proposal_manifest.jsonl"), "semantics": sam3["proposal_role"]})
        append_log("S1_SAM3_MULTI_PROMPT_PROPOSALS_COMPLETE", frames=len(ordered), prompts=len(sam3["prompts"]), manifest_sha256=sha256_file(OUT / "sam3_proposal_manifest.jsonl"))
    else:
        append_log("S1_SAM3_MULTI_PROMPT_PROPOSALS_PARTIAL", completed=len(ordered), expected=len(expected), requested_limit=max_frames)
    return {"status": status, "completed_frames": len(ordered), "expected_frames": len(expected), "runtime": runtime}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-frames", type=int, default=None, help="bounded smoke run; omit to finish the frozen S1 set")
    args = parser.parse_args()
    print(json.dumps(run(max_frames=args.max_frames), ensure_ascii=False, sort_keys=True))
