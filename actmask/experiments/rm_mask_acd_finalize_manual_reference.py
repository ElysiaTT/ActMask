#!/usr/bin/env python3
"""Finish the auditable terminal package for RM-MASK-ACD manual reference.

The final decision is only valid when automatic masks remain unaccepted and the
bounded human reference package is complete, but still unannotated.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "actmask" / "rm_mask_acd_verified"
PACKAGE = OUT / "manual_annotation_package"
SAM3_ROOT = Path("/data/projects/tzh/papers/poke_and_splat/sam3")
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


def text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value)


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


def load_rows() -> list[dict[str, Any]]:
    return [json.loads(line) for line in (PACKAGE / "annotation_manifest.jsonl").read_text().splitlines() if line]


def contact_sheets(rows: list[dict[str, Any]], status: dict[str, Any]) -> list[dict[str, Any]]:
    by_id = {record["id"]: record for record in status["records"]}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["task_id"]].append(row)
    target = OUT / "automatic_mask_contact_sheets"
    target.mkdir(exist_ok=True)
    outputs: list[dict[str, Any]] = []
    for task, task_rows in sorted(grouped.items()):
        task_rows.sort(key=lambda value: value["id"])
        thumb_w, thumb_h, columns = 256, 192, 4
        page = Image.new("RGB", (columns * thumb_w, 5 * (thumb_h + 20)), "white")
        draw = ImageDraw.Draw(page)
        for index, row in enumerate(task_rows):
            proposal = by_id[row["id"]]
            preview = Image.open(PACKAGE / proposal["preview_path"]).convert("RGB").resize((thumb_w, thumb_h))
            x, y = (index % columns) * thumb_w, (index // columns) * (thumb_h + 20)
            page.paste(preview, (x, y)); draw.text((x + 3, y + thumb_h + 2), f"ep{row['episode_index']:06d} f{row['frame_index']:06d}", fill="black")
        name = task.replace("/", "_") + ".jpg"
        path = target / name
        page.save(path, quality=92)
        outputs.append({"task_id": task, "path": str(path.relative_to(OUT)), "sha256": sha256(path), "frames": len(task_rows)})
    return outputs


def run() -> dict[str, Any]:
    config_path = OUT / "preregistered_config.json"
    config_hash = sha256(config_path)
    if config_hash != (OUT / "preregistered_config.sha256").read_text().strip():
        raise RuntimeError("preregistered configuration hash mismatch")
    if json.loads((OUT / "m0_decision.json").read_text())["decision"] != "M0_AUTOMATIC_SEGMENTATION_REQUIRED":
        raise RuntimeError("this finalizer is only valid after the M0 automatic-segmentation decision")
    rows, status = load_rows(), json.loads((PACKAGE / "sam3_proposal_status.json").read_text())
    if len(rows) != 200 or len(status.get("records", [])) != 200 or status.get("status") != "completed_proposal_only":
        raise RuntimeError("human package/SAM3 proposal count must be exactly 200")
    if len({row["task_id"] for row in rows}) != 10 or len({row["episode_id"] for row in rows}) != 20:
        raise RuntimeError("human package stratification invariant failed")
    for record in status["records"]:
        for key in ("proposal_path", "preview_path"):
            if not (PACKAGE / record[key]).is_file():
                raise RuntimeError(f"missing SAM3 annotation aid: {record[key]}")
    report = json.loads((PACKAGE / "validation_report.json").read_text())
    if report.get("status") != "annotation_pending_or_invalid" or report.get("missing_masks") != 800:
        raise RuntimeError("manual reference must be prepared but not spuriously claimed complete")
    areas = sorted(float(record["robot_proposal_area_fraction"]) for record in status["records"])
    sheets = contact_sheets(rows, status)
    commit = command(["git", "-C", str(SAM3_ROOT), "rev-parse", "HEAD"])
    env = {name: package_version(name) for name in ("torch", "torchvision", "timm", "ftfy", "iopath", "numpy", "Pillow", "huggingface-hub")}
    candidate_report = {
        "schema": "rm-mask-acd-automatic-mask-candidate-report-v1",
        "config_sha256": config_hash,
        "pilot_requirement": "all 10 tasks, at least 20 trajectories, five or more frames each, including temporal and motion-peak-proxy strata",
        "pilot_actual": {"frames": 200, "tasks": 10, "trajectories": 20, "automatic_candidate_evaluations": 0},
        "candidates": [
            {"rank": 1, "name": "RoboEngine / Robo-SAM", "repository_or_model_id": "https://robotengine.github.io/", "status": "not_runnable", "reason": "No locally verified source checkout or checkpoint; no heuristic substitution was made.", "metrics": None},
            {"rank": 2, "name": "RobotSeg", "repository_or_model_id": "https://github.com/showlab/RobotSeg", "status": "not_runnable", "reason": "No locally verified source checkout or released checkpoint; no unrecorded dependency/weight acquisition was attempted.", "metrics": None},
            {"rank": 3, "name": "Local SAM3 text-prompt proposal producer", "repository_or_model_id": "https://github.com/facebookresearch/sam3", "revision": commit, "status": "annotation_aid_only", "reason": "No verified first-frame robot mask existed, so this prompt-only result cannot qualify as an automatic M2 candidate or pass M4. It was used solely to prepare the allowed M3 human-review aids.", "checkpoint_sha256": status["checkpoint_sha256"], "input_resolution": sorted({(row["width"], row["height"]) for row in rows}), "prompts": ["robot arm", "robotic gripper"], "postprocessing": "union of prompt instances with area >=64 pixels; no visual-change expansion; not a reference label", "metrics": None},
        ],
        "acceptance": "No automatic candidate was accepted. All M4 quality gates remain unevaluated until independent human labels are complete.",
    }
    dump(OUT / "automatic_mask_candidate_report.json", candidate_report)
    dump(OUT / "automatic_mask_runtime.json", {
        "schema": "rm-mask-acd-automatic-mask-runtime-v1", "config_sha256": config_hash,
        "sam3_annotation_aid": {"runtime": status["runtime"], "checkpoint_sha256": status["checkpoint_sha256"], "source_revision": commit, "python": sys.version, "dependencies": env, "gpu_memory": "not sampled during this proposal-only run; no peak-memory claim is made", "raw_detector_outputs": "not retained because SAM3 was not an eligible automatic candidate"},
        "resource_limit_note": "Proposal inference was completed before target acceptance; no training GPU budget was consumed.",
    })
    artifacts = []
    for record in status["records"]:
        artifacts.append({"id": record["id"], "proposal": record["proposal_path"], "proposal_sha256": sha256(PACKAGE / record["proposal_path"]), "preview": record["preview_path"], "preview_sha256": sha256(PACKAGE / record["preview_path"]), "status": record["status"]})
    dump(OUT / "automatic_mask_manifest.json", {
        "schema": "rm-mask-acd-sam3-annotation-aid-manifest-v1", "config_sha256": config_hash, "count": len(artifacts), "artifacts": artifacts,
        "semantics": "These are SAM3 proposal unions for annotation assistance only. They are not automatic candidate raw masks, not human references, and must never be used as ACD labels.",
    })
    dump(OUT / "m3_decision.json", {"schema": "rm-mask-acd-m3-decision-v1", "decision": "RM_MASK_HUMAN_REFERENCE_REQUIRED", "reason": ["Selected real trajectories have no official robot masks.", "Exact geometry projection is unavailable for selected real episodes.", "No eligible automatic candidate can be independently accepted without human reference."], "manual_package": {"path": str(PACKAGE), "frames": 200, "tasks": 10, "trajectories": 20, "labels": list(CLASSES), "sam3_aids": 200, "validation_status": report["status"]}, "next_gate": "Complete labels, run validate_annotations.py --require-complete, then run the resume command; do not construct targets before that."})
    text(OUT / "mask_quality_report.md", """# Mask quality report

No robot mask is verified. M4 metrics (IoU, recall, false-positive area, boundary
F1, gripper recall, manipulated-object preservation, contact preservation, and
temporal flicker) are deliberately **not reported**: no independent reference
exists yet. The 200 SAM3 masks are annotation aids, not a quality measurement.

The only valid conclusion is `RM_MASK_HUMAN_REFERENCE_REQUIRED`.
""")
    text(OUT / "target_quality_report.md", """# Target quality report

M5 was not run. Target reconstruction is forbidden until a robot-only mask has
passed M4 against an independent reference. Rejected RM-ACD heuristic masks and
their derived targets remain audit-only and were not read as labels here.
""")
    dump(OUT / "baseline_summary.json", {"schema": "rm-mask-acd-baseline-summary-v1", "status": "not_run", "reason": "M6 is gated on RM_MASK_VERIFIED and valid v2 targets; neither exists.", "runs": []})
    dump(OUT / "raw_prediction_manifest.json", {"schema": "rm-mask-acd-raw-prediction-manifest-v1", "status": "empty_by_design", "reason": "No targets, baselines, or method may run before independent mask validation.", "predictions": []})
    text(OUT / "claim_boundary.md", """# Claim boundary

This run establishes only that a bounded, stratified robot-only human reference
package is ready. It makes no claim that SAM3 is accurate, that a robot mask is
verified, that scene-change targets are valid, or that any ACD model improves a
baseline. It does not predict alternate-action outcomes or physical
counterfactuals.
""")
    text(OUT / "paper_positioning.md", """# Paper positioning

Current evidence supports a reproducible benchmark/audit contribution about the
need for robot-only exclusions in real dual-arm manipulation data. It does not
support an ActMask or action-conditioned dynamics method claim. A method paper
requires completed human references, M4 acceptance, valid v2 targets, baseline
headroom, and the prescribed three-seed method evaluation.
""")
    text(OUT / "next_steps.md", """# Next steps

1. Annotate all 200 frames using the package guide; label robot arm, gripper,
   manipulated object, and background as exhaustive disjoint PNGs.
2. Run `validate_annotations.py --require-complete`.
3. Run `python -m actmask.experiments.rm_mask_acd_resume_from_annotations --package outputs/actmask/rm_mask_acd_verified/manual_annotation_package`.
4. Only then create a held-out M4 evaluator. It must not reuse rejected RM-ACD
   masks or loosen the preregistered gates.
""")
    text(OUT / "final_report.md", f"""# RM-MASK-ACD final report

## Decision

`RM_MASK_HUMAN_REFERENCE_REQUIRED`

M0 found neither official selected-episode masks nor an exact, documented
real-episode geometry path. Three candidate paths were recorded without
silently replacing unavailable sources. A local SAM3 checkpoint generated 200
proposal-only review aids across 10 tasks and 20 frozen trajectories. It is not
a reference and was not accepted as an automatic robot mask.

The package is ready for the single permitted human intervention: 200 frames,
four disjoint label classes, a validator, review contact sheets, and a resume
gate. No target construction, baselines, methods, predictions, or method claims
were run.

## Audit facts

- Frozen configuration SHA-256: `{config_hash}`
- SAM3 proposal masks: 200; area fraction is descriptive only (min {areas[0]:.4f}, mean {sum(areas)/len(areas):.4f}, p95 {areas[int(.95*(len(areas)-1))]:.4f}, max {areas[-1]:.4f}).
- Human annotation state: 0/200 complete frames, 800 required masks pending.
- Contact sheets: {len(sheets)} task sheets.
""")
    final = {"schema": "rm-mask-acd-final-decision-v1", "decision": "RM_MASK_HUMAN_REFERENCE_REQUIRED", "terminal": True, "config_sha256": config_hash, "evidence": {"m0": "M0_AUTOMATIC_SEGMENTATION_REQUIRED", "m2": "no eligible automatic candidate accepted", "m3": "200-frame human reference package prepared with 200 SAM3 review aids", "m4_m7": "not run by gate"}, "prohibitions_honored": ["No rejected heuristic mask was reused as a label.", "No frozen RM/RM-OD/RM-ACD artifact was modified.", "No ACD target, baseline, or method run occurred."], "resume": "manual_annotation_package/RESUME.md"}
    dump(OUT / "final_decision.json", final)
    reproducibility = {"schema": "rm-mask-acd-reproducibility-manifest-v1", "config_sha256": config_hash, "source_hash_manifest_sha256": sha256(OUT / "source_hash_manifest.json"), "scripts": {name: sha256(ROOT / "actmask" / "experiments" / name) for name in ("rm_mask_acd_m0_inventory.py", "rm_mask_acd_build_manual_package.py", "rm_mask_acd_resume_from_annotations.py", "rm_mask_acd_finalize_manual_reference.py")}, "environment": {"python": sys.version, "platform": platform.platform(), "dependencies": env}, "manual_package": {"sampling_manifest_sha256": sha256(PACKAGE / "sampling_manifest.json"), "annotation_manifest_sha256": sha256(PACKAGE / "annotation_manifest.jsonl"), "sam3_status_sha256": sha256(PACKAGE / "sam3_proposal_status.json"), "proposal_count": 200}, "commands": ["/home/tzh/conda_envs/actmask/bin/python -m actmask.experiments.rm_mask_acd_build_manual_package --sam3-proposals 200", "/home/tzh/conda_envs/actmask/bin/python outputs/actmask/rm_mask_acd_verified/manual_annotation_package/validate_annotations.py --require-complete", "/home/tzh/conda_envs/actmask/bin/python -m actmask.experiments.rm_mask_acd_resume_from_annotations --package outputs/actmask/rm_mask_acd_verified/manual_annotation_package"]}
    dump(OUT / "reproducibility_manifest.json", reproducibility)
    text(OUT / "docs" / "rm_mask_acd_results.md", "# RM-MASK-ACD results\n\nTerminal decision: `RM_MASK_HUMAN_REFERENCE_REQUIRED`. M0 found no official/geometric path. The 200-frame manual reference package and 200 SAM3 proposal-only review aids are complete. M4-M7 are correctly not run pending independent human labels.\n")
    text(OUT / "docs" / "rm_mask_acd_handoff.md", "# RM-MASK-ACD handoff\n\nAnnotate `manual_annotation_package` as instructed, then validate and invoke its resume command. Do not treat SAM3 masks as labels, and do not restart target construction from the rejected RM-ACD data.\n")
    append_log("phase_m8_terminal_manual_reference_required", decision="RM_MASK_HUMAN_REFERENCE_REQUIRED", frames=200, sam3_annotation_aids=200, contact_sheets=len(sheets), config_sha256=config_hash)
    return final


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, sort_keys=True))
