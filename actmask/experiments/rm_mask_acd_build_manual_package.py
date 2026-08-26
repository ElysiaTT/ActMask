#!/usr/bin/env python3
"""Build the RM-MASK-ACD human-reference package and optional SAM3 proposals.

The package is deliberately an annotation input, not an automatically accepted
robot mask dataset.  In particular, neither SAM3 scores nor the rejected
RM-ACD motion masks may be used as reference labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyarrow.parquet as pq
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "actmask" / "rm_mask_acd_verified"
PACKAGE = OUT / "manual_annotation_package"
CONFIG = OUT / "preregistered_config.json"
SAM3_ROOT = Path("/data/projects/tzh/papers/poke_and_splat/sam3")
PAPERS_SITE = Path("/home/tzh/conda_envs/papers/lib/python3.11/site-packages")
SAM3_CHECKPOINT = Path("/data/projects/tzh/papers/poke_and_splat/sam3/checkpoint/sam3.pt")
CLASSES = ("robot_arm", "gripper", "manipulated_object", "background")


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
    prior, sequence = "", 0
    if path.is_file() and path.stat().st_size:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        prior, sequence = rows[-1]["event_sha256"], int(rows[-1]["seq"])
    row = {"seq": sequence + 1, "utc": datetime.now(timezone.utc).isoformat(), "event": event, "payload": payload, "prev_event_sha256": prior}
    row["event_sha256"] = hashlib.sha256(json.dumps(row, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    with path.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def require_frozen_config() -> str:
    expected = (OUT / "preregistered_config.sha256").read_text().strip()
    actual = sha256(CONFIG)
    if expected != actual:
        raise RuntimeError("refusing to build package: preregistered configuration hash changed")
    return actual


def action_peak_indices(parquet_path: str, frames: int) -> list[tuple[int, str]]:
    table = pq.read_table(parquet_path, columns=["action.arm_joint_position", "action.hand_joint_position"])
    arm = np.asarray(table["action.arm_joint_position"].to_pylist(), dtype=np.float32)
    hand = np.asarray(table["action.hand_joint_position"].to_pylist(), dtype=np.float32)
    magnitude = np.linalg.norm(arm, axis=1) + np.linalg.norm(hand, axis=1)
    # A high action magnitude is only a contact-heavy *proxy*; it is not a
    # ground-truth contact label and is explicitly named as such in the manifest.
    halves = (np.arange(0, max(1, frames // 2)), np.arange(max(1, frames // 2), frames))
    results: list[tuple[int, str]] = []
    for part, name in zip(halves, ("motion_peak_proxy_early", "motion_peak_proxy_late")):
        if len(part):
            results.append((int(part[np.argmax(magnitude[part])]), name))
    return results


def sampling_indices(parquet_path: str, frames: int) -> list[tuple[int, str]]:
    temporal = [(int(round((frames - 1) * fraction)), f"timeline_{int(fraction * 100):02d}pct") for fraction in (0.04, 0.16, 0.29, 0.41, 0.54, 0.66, 0.79, 0.93)]
    planned = temporal + action_peak_indices(parquet_path, frames)
    unique: list[tuple[int, str]] = []
    used: set[int] = set()
    for index, stratum in planned:
        if index not in used:
            unique.append((index, stratum)); used.add(index)
    if len(unique) < 10:
        for index in np.linspace(0, frames - 1, 10, dtype=int).tolist():
            if index not in used:
                unique.append((index, "timeline_replacement")); used.add(index)
            if len(unique) == 10:
                break
    if len(unique) != 10:
        raise RuntimeError(f"could not choose ten unique frames from a {frames}-frame video")
    return sorted(unique)


def selected_records() -> list[dict[str, Any]]:
    payload = json.loads((OUT / "selected_trajectory_manifest.json").read_text())
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in payload["episodes"]:
        grouped[record["task_id"]].append(record)
    if len(grouped) != 10:
        raise RuntimeError(f"expected ten frozen tasks, found {len(grouped)}")
    records: list[dict[str, Any]] = []
    for task in sorted(grouped):
        choices = sorted(grouped[task], key=lambda item: (int(item["episode_index"]), item["episode_id"]))
        # Two source trajectories per task, deliberately separated within the
        # frozen 10-trajectory task slice; no unseen source episode is added.
        records.extend((choices[0], choices[-1]))
    return records


def frame_name(record: dict[str, Any], index: int) -> str:
    safe_task = record["task_id"].replace("/", "_")
    return f"{safe_task}__ep{int(record['episode_index']):06d}__frame{index:06d}"


def write_static_package(records: list[dict[str, Any]], config_hash: str) -> list[dict[str, Any]]:
    (PACKAGE / "frames").mkdir(parents=True, exist_ok=True)
    for category in CLASSES:
        (PACKAGE / "annotations" / category).mkdir(parents=True, exist_ok=True)
    examples: list[dict[str, Any]] = []
    for record in records:
        capture = cv2.VideoCapture(record["front_video_path"])
        if not capture.isOpened():
            raise RuntimeError(f"cannot open source RGB video: {record['front_video_path']}")
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 10:
            raise RuntimeError(f"source RGB video has too few frames: {record['front_video_path']}")
        for frame_index, stratum in sampling_indices(record["parquet_path"], total):
            identifier = frame_name(record, frame_index)
            output = PACKAGE / "frames" / f"{identifier}.png"
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"cannot decode frame {frame_index} from {record['front_video_path']}")
            height, width = frame.shape[:2]
            if not output.exists():
                if not cv2.imwrite(str(output), frame):
                    raise RuntimeError(f"cannot write annotation frame {output}")
            examples.append({
                "id": identifier,
                "task_id": record["task_id"],
                "episode_id": record["episode_id"],
                "episode_index": int(record["episode_index"]),
                "view": "camera_front",
                "frame_index": frame_index,
                "sampling_stratum": stratum,
                "contact_note": "motion_peak_proxy is a sampling heuristic only; annotate visible contact/occlusion directly.",
                "source_video": record["front_video_path"],
                "source_parquet": record["parquet_path"],
                "frame_path": f"frames/{identifier}.png",
                "width": width,
                "height": height,
                "required_labels": {category: f"annotations/{category}/{identifier}.png" for category in CLASSES},
                "status": "unannotated",
            })
        capture.release()
    if len(examples) != 200 or len({item["task_id"] for item in examples}) != 10 or len({item["episode_id"] for item in examples}) != 20:
        raise RuntimeError("manual package stratification invariant failed")
    examples.sort(key=lambda item: item["id"])
    manifest = PACKAGE / "annotation_manifest.jsonl"
    manifest.write_text("".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in examples))
    source_hashes = json.loads((OUT / "source_hash_manifest.json").read_text())
    dump(PACKAGE / "sampling_manifest.json", {
        "schema": "rm-mask-acd-human-reference-sampling-v1",
        "status": "annotation_pending",
        "config_sha256": config_hash,
        "frame_count": len(examples), "task_count": 10, "trajectory_count": 20,
        "source_selection": "two separated episodes per frozen RM-Series task; ten frames per episode",
        "sampling": "eight temporal quantiles plus two action-magnitude motion-peak proxies, deduplicated deterministically",
        "source_hash_manifest": "../source_hash_manifest.json",
        "source_hash_records": source_hashes["files"],
    })
    dump(PACKAGE / "annotation_schema.json", {
        "schema": "rm-mask-acd-human-reference-label-v1",
        "image_format": "PNG", "mask_format": "single-channel PNG; 0=absent, 255=present", "classes": list(CLASSES),
        "semantic_rules": {
            "robot_arm": "Visible Franka links and wrist bodies only. Exclude gripper fingers/palm.",
            "gripper": "Visible gripper palm, fingers, and fingertips of either robot. Include pixels occluding an object.",
            "manipulated_object": "Object(s) being grasped, touched, inserted, or directly manipulated; never assign these pixels to robot.",
            "background": "All remaining image pixels: table, fixtures, tools, static objects, shadows, and unlabelled scene.",
        },
        "constraints": ["Masks must match frame dimensions.", "Classes are mutually exclusive and their union must cover every pixel.", "For occlusion boundaries, label the visible foreground pixel.", "Do not infer hidden robot/object pixels.", "SAM3 proposal masks are optional visual aids and never labels."],
    })
    (PACKAGE / "ANNOTATION_GUIDE.md").write_text("""# RM-MASK-ACD annotation guide

Annotate each frame in `frames/` into four masks under `annotations/`: `robot_arm`,
`gripper`, `manipulated_object`, and `background`.  Save a single-channel PNG at
the exact input resolution, using 255 for class pixels and 0 otherwise.

Robot means **only visible Franka arm links and grippers**.  A grasped or touched
object is always `manipulated_object`, even when it is within the gripper.  Table,
fixtures, tools, shadows, and all non-manipulated scene regions are `background`.
At boundaries, annotate the visible foreground surface; do not hallucinate behind
an occluder.  The four classes must be exhaustive and pairwise disjoint.

`sam3_proposals/` and `sam3_previews/`, when present, are weak review aids.  They
are not reference labels, may omit a gripper, and must be corrected rather than
traced.  Do not use any RM-ACD motion/sweep mask: those masks are rejected audit
artifacts and are absent from this package.

Before handoff, run:

```bash
/home/tzh/conda_envs/actmask/bin/python validate_annotations.py --require-complete
```

Then resume the held-out acceptance audit with the exact command in `RESUME.md`.
""")
    (PACKAGE / "README.md").write_text("# RM-MASK-ACD human reference package\n\nStatus: `annotation_pending`. This package contains 200 front-view frames from 20 frozen RoboMIND trajectories across 10 tasks. It is the independent reference required before any automatic robot mask can be accepted.\n")
    (PACKAGE / "RESUME.md").write_text("""# Resume after annotation

Do not start target construction or model training before all labels pass validation.

```bash
cd /data/projects/tzh/papers/ActMask
/home/tzh/conda_envs/actmask/bin/python outputs/actmask/rm_mask_acd_verified/manual_annotation_package/validate_annotations.py --require-complete
/home/tzh/conda_envs/actmask/bin/python -m actmask.experiments.rm_mask_acd_resume_from_annotations --package outputs/actmask/rm_mask_acd_verified/manual_annotation_package
```

The second command is intentionally unavailable until the independent labels have been reviewed and accepted; it prevents accidental construction of targets from unverified proposals.
""")
    write_validator()
    return examples


def write_validator() -> None:
    source = '''#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
CLASSES = ("robot_arm", "gripper", "manipulated_object", "background")
def binary(path):
    data = np.asarray(Image.open(path).convert("L"))
    if not np.isin(data, (0, 255)).all(): raise ValueError(f"non-binary PNG: {path}")
    return data == 255
def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--require-complete", action="store_true"); args = parser.parse_args()
    rows = [json.loads(line) for line in (ROOT / "annotation_manifest.jsonl").read_text().splitlines() if line]
    missing=[]; errors=[]; complete=0
    for row in rows:
        frame = Image.open(ROOT / row["frame_path"]); shape=(frame.height, frame.width); masks=[]
        for name, rel in row["required_labels"].items():
            path=ROOT / rel
            if not path.exists(): missing.append(str(path)); continue
            try:
                mask=binary(path)
                if mask.shape != shape: raise ValueError(f"size {mask.shape}, expected {shape}")
                masks.append(mask)
            except Exception as exc: errors.append(f"{path}: {exc}")
        if len(masks)==len(CLASSES):
            total=sum(mask.astype(np.uint8) for mask in masks)
            if not np.all(total == 1): errors.append(f"{row['id']}: classes must be exhaustive and disjoint")
            else: complete += 1
    report={"schema":"rm-mask-acd-annotation-validation-v1","frames":len(rows),"complete_frames":complete,"missing_masks":len(missing),"errors":errors,"status":"complete" if complete==len(rows) and not errors else "annotation_pending_or_invalid"}
    (ROOT / "validation_report.json").write_text(json.dumps(report, indent=2)+"\\n")
    print(json.dumps(report, sort_keys=True))
    if args.require_complete and report["status"] != "complete": raise SystemExit(2)
if __name__ == "__main__": main()
'''
    path = PACKAGE / "validate_annotations.py"
    path.write_text(source)
    path.chmod(0o755)


def patch_sam3_fused_mlp(torch: Any) -> None:
    import torch.nn.functional as functional
    import sam3.model.vitdet as vitdet
    def safe_addmm_act(activation: Any, linear: Any, mat1: Any) -> Any:
        value = linear(mat1)
        if activation in {functional.relu, torch.nn.ReLU}: return functional.relu(value)
        if activation in {functional.gelu, torch.nn.GELU}: return functional.gelu(value)
        if isinstance(activation, type) and issubclass(activation, torch.nn.Module): return activation()(value)
        raise ValueError(f"unexpected SAM3 activation: {activation}")
    vitdet.addmm_act = safe_addmm_act


def run_sam3(examples: list[dict[str, Any]], limit: int, config_hash: str) -> dict[str, Any]:
    if limit <= 0:
        return {"status": "not_requested", "proposal_count": 0}
    for path in (PAPERS_SITE, SAM3_ROOT):
        if str(path) not in sys.path: sys.path.insert(0, str(path))
    if not SAM3_CHECKPOINT.is_file(): raise RuntimeError(f"missing local SAM3 checkpoint: {SAM3_CHECKPOINT}")
    import torch
    if not torch.cuda.is_available(): raise RuntimeError("SAM3 proposals require CUDA, but CUDA is unavailable")
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    patch_sam3_fused_mlp(torch)
    torch.cuda.empty_cache()
    model = build_sam3_image_model(device="cuda", checkpoint_path=str(SAM3_CHECKPOINT), load_from_HF=False, compile=False)
    processor = Sam3Processor(model, device="cuda", confidence_threshold=0.20)
    proposal_dir, preview_dir = PACKAGE / "sam3_proposals", PACKAGE / "sam3_previews"
    proposal_dir.mkdir(exist_ok=True); preview_dir.mkdir(exist_ok=True)
    chosen = examples[:min(limit, len(examples))]
    records: list[dict[str, Any]] = []
    started = time.monotonic()
    for position, item in enumerate(chosen, start=1):
        image = Image.open(PACKAGE / item["frame_path"]).convert("RGB")
        state = processor.set_image(image)
        union = np.zeros((image.height, image.width), dtype=bool)
        prompts=[]
        for prompt in ("robot arm", "robotic gripper"):
            result = processor.set_text_prompt(prompt=prompt, state=state)
            masks = result.get("masks")
            scores = result.get("scores")
            if isinstance(masks, torch.Tensor): masks=masks.detach().cpu().numpy()
            if isinstance(scores, torch.Tensor): scores=scores.detach().cpu().numpy()
            masks=np.asarray(masks) if masks is not None else np.zeros((0, 1, image.height, image.width), dtype=bool)
            scores=np.asarray(scores) if scores is not None else np.zeros((0,), dtype=np.float32)
            if masks.ndim == 4: masks=masks[:, 0]
            kept=0
            for index in np.argsort(-scores).tolist():
                mask=masks[index].astype(bool)
                if int(mask.sum()) >= 64:
                    union |= mask; kept += 1
            prompts.append({"prompt": prompt, "raw_instances": int(len(scores)), "kept_instances": kept, "scores": [float(value) for value in scores.tolist()]})
        mask_path=proposal_dir / f"{item['id']}.png"
        Image.fromarray((union.astype(np.uint8)*255), mode="L").save(mask_path)
        rgb=np.asarray(image).copy(); rgb[union]=np.array([255, 64, 64], dtype=np.uint8)*0.55 + rgb[union]*0.45
        Image.fromarray(rgb).save(preview_dir / f"{item['id']}.jpg", quality=92)
        records.append({"id":item["id"],"proposal_path":str(mask_path.relative_to(PACKAGE)),"preview_path":str((preview_dir / f"{item['id']}.jpg").relative_to(PACKAGE)),"robot_proposal_area_fraction":float(union.mean()),"prompts":prompts,"status":"proposal_only_not_reference_label"})
        print(f"sam3_proposal={position}/{len(chosen)} id={item['id']}", flush=True)
    status={"schema":"rm-mask-acd-sam3-proposals-v1","status":"completed_proposal_only","config_sha256":config_hash,"checkpoint":str(SAM3_CHECKPOINT),"checkpoint_sha256":sha256(SAM3_CHECKPOINT),"runtime":{"device":"cuda","torch":torch.__version__,"seconds":round(time.monotonic()-started,3)},"count":len(records),"records":records,"prohibition":"SAM3 proposals and scores are annotation aids only; they are not independent reference labels and cannot be used for M4 acceptance."}
    dump(PACKAGE / "sam3_proposal_status.json", status)
    del processor, model
    torch.cuda.empty_cache()
    return status


def write_candidate_registry(config_hash: str, sam3_status: dict[str, Any]) -> None:
    status = sam3_status["status"]
    candidates = [
        {"rank":1,"name":"RoboEngine / Robo-SAM","public_source":"https://robotengine.github.io/","status":"not_runnable","reason":"No locally verified code checkout or checkpoint is present. It is not silently substituted with a heuristic.","metrics":None},
        {"rank":2,"name":"RobotSeg","public_source":"https://github.com/showlab/RobotSeg","status":"not_runnable","reason":"No locally verified checkout or released checkpoint is present. Dependency/weight acquisition must be recorded before an evaluation.","metrics":None},
        {"rank":3,"name":"Local SAM3 text proposals","public_source":"https://github.com/facebookresearch/sam3","status":status,"reason":"Available only as a manual-annotation proposal producer: no verified first-frame robot mask exists, so it cannot be accepted as an M2 automatic candidate.","metrics":None,"proposal_status":"manual_annotation_package/sam3_proposal_status.json" if status.startswith("completed") else None},
    ]
    dump(OUT / "m2_candidate_registry.json", {"schema":"rm-mask-acd-m2-candidate-registry-v1","config_sha256":config_hash,"automatic_candidates_max":3,"candidates":candidates,"acceptance_status":"no_automatic_candidate_can_be_accepted_without_the_independent_human_reference"})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sam3-proposals", type=int, default=0, help="Number of package frames for local SAM3 proposal generation (0 = do not load SAM3).")
    args = parser.parse_args()
    config_hash = require_frozen_config()
    records = selected_records()
    examples = write_static_package(records, config_hash)
    try:
        sam3_status = run_sam3(examples, args.sam3_proposals, config_hash)
    except Exception as exc:
        sam3_status = {"schema":"rm-mask-acd-sam3-proposals-v1","status":"blocked","exception_type":type(exc).__name__,"exception":str(exc),"prohibition":"No SAM3 output was used as a label or acceptance reference."}
        dump(PACKAGE / "sam3_proposal_status.json", sam3_status)
    write_candidate_registry(config_hash, sam3_status)
    append_log("phase_m2_m3_manual_reference_package", frames=len(examples), tasks=10, trajectories=20, sam3_status=sam3_status["status"], sam3_requested=args.sam3_proposals)
    print(json.dumps({"frames":len(examples),"tasks":10,"trajectories":20,"sam3_status":sam3_status["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
