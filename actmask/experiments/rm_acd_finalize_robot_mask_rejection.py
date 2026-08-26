"""Finalize RM-ACD when the pre-model robot exclusion gate fails clearly."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from actmask.experiments.rm_acd_freeze_and_modality import OUT, append_log, sha, write_json


DECISION = "RM_ACD_ROBOT_MASK_INVALID"


def _load(name: str) -> dict[str, Any]:
    return json.loads((OUT / name).read_text())


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run() -> dict[str, Any]:
    config_sha = (OUT / "preregistered_config.sha256").read_text().strip()
    robot = _load("robot_mask_audit.json")
    stats = _load("target_statistics.json")["per_horizon"]
    contacts = _load("robot_mask_contact_sheet_manifest.json")
    construction = _load("target_construction_audit.json")
    if robot["all_mask_area_below_half"] or robot["automatic_stability_check"]:
        raise RuntimeError("this finalizer is only valid for the observed clearly invalid robot-mask result")
    if len(contacts["samples"]) < 20 or set(contacts["coverage"].values()) != {2}:
        raise RuntimeError("required 20-trajectory, all-task visual audit package is incomplete")
    task_mask = {task: row["mean_mask_area"] for task, row in robot["task_summary"].items()}
    visual_review = {
        "schema": "rm-acd-robot-mask-manual-review-v1",
        "reviewed_contact_sheets": [
            "target_audit_contact_sheets/robogene_twoArm_franka_adjust_lamp_000001_a03.jpg",
            "target_audit_contact_sheets/robogene_twoArm_franka_assemble_pvc_clamps_000000_a03.jpg",
            "target_audit_contact_sheets/robogene_twoArm_franka_clean_hang_lab_cloth_000001_a03.jpg",
            "target_audit_contact_sheets/robogene_twoArm_franka_assemble_adjust_lab_set_000000_a03.jpg",
        ],
        "review_scope": "four task-diverse sheets were directly inspected, while the generated package covers two distinct trajectories from every one of ten task groups",
        "findings": [
            "The red heuristic region spans broad table/background areas rather than only robot links and grippers.",
            "Visible manipulated objects are often covered by the red region, including objects near but not visually inseparable from the gripper.",
            "The mask leaves too little stable non-robot field for a dense object-change target.",
        ],
        "decision": "reject the mask before training; these sheets are audit evidence, not human segmentation labels",
    }
    write_json(OUT / "robot_mask_manual_review.json", visual_review)
    quality = {
        "schema": "rm-acd-target-quality-audit-v1",
        "status": "not_accepted_not_fully_evaluated",
        "prerequisite_failure": DECISION,
        "robot_mask_mean_area_by_task": task_mask,
        "robot_mask_area_range": [float(min(task_mask.values())), float(max(task_mask.values()))],
        "robot_mask_all_below_half": robot["all_mask_area_below_half"],
        "robot_mask_automatic_stability": robot["automatic_stability_check"],
        "raw_change_robot_overlap_by_horizon": {name: row["robot_raw_overlap_fraction"] for name, row in stats.items()},
        "other_target_diagnostics_recorded_but_not_sufficient_to_override_mask_failure": {name: {"meaningful_window_fraction": row["meaningful_window_fraction"], "confidence_window_fraction_ge_0_60": row["confidence_coverage"]["window_fraction_ge_0_60"], "static_border_false_positive": row["static_background_false_positive_area"], "action_area_correlation": row["action_magnitude_correlation"], "progress_area_correlation": row["episode_progress_correlation"], "brightness_area_correlation": row["brightness_correlation"]} for name, row in stats.items()},
        "pass": False,
        "reason": "A target cannot be accepted as non-robot scene/object change when its exclusion mask covers 61.6%–79.5% of images and removes approximately 97% of raw changes.",
    }
    write_json(OUT / "target_quality_audit.json", quality)
    failure_cases = """# RM-ACD target failure cases

Decision: `RM_ACD_ROBOT_MASK_INVALID`.

The conservative boundary-motion sweep prior over-masks the visual workspace:
mean mask area is 0.616–0.795 across the ten tasks. Direct inspection of the
contact sheets shows broad table/background coverage and frequent removal of
visible manipulated objects near the grippers. Raw temporal-change evidence
overlaps the mask by about 0.97 at all three horizons.

This is a target-construction failure, not evidence that the robot cannot
move objects. The retained pilot targets are audit artifacts only and must not
be used to train or report a dense non-robot change predictor.
"""
    (OUT / "target_failure_cases.md").write_text(failure_cases)
    quality_report = f"""# RM-ACD target quality report

Target construction was stopped before A4/A5/A6/A7 because A2 failed.

Robot-mask mean area ranged from {min(task_mask.values()):.3f} to
{max(task_mask.values()):.3f}; the pre-model requirement that the mask leave a
usable non-robot scene therefore failed. The raw-change/robot-mask overlap was
{', '.join(f'{name}={row["robot_raw_overlap_fraction"]:.3f}' for name, row in stats.items())}.

Some downstream diagnostics are numerically nontrivial, but they do not
rescue a target that is visually dominated by an invalid exclusion mask.
"""
    (OUT / "target_quality_report.md").write_text(quality_report)
    baseline = {"schema": "rm-acd-baseline-summary-v1", "status": "not_run", "reason": "A7 is prohibited because the A2 robot-mask prerequisite failed.", "method_training": "not_run"}
    write_json(OUT / "baseline_summary.json", baseline)
    raw = {"schema": "rm-acd-raw-prediction-manifest-v1", "status": "empty_by_design", "reason": "No baseline or method predictions may be created after RM_ACD_ROBOT_MASK_INVALID.", "baseline_predictions": [], "method_predictions": [], "all_entries_present": True}
    write_json(OUT / "raw_prediction_manifest.json", raw)
    final = {
        "schema": "rm-acd-final-decision-v1",
        "decision": DECISION,
        "method_ran": False,
        "baseline_ran": False,
        "preregistered_config_sha256": config_sha,
        "source": "RoboMIND 2.0 only; exact frozen 100-trajectory Franka / 10-task source manifest",
        "trigger": {"robot_mask_area_range": [float(min(task_mask.values())), float(max(task_mask.values()))], "all_masks_below_half": robot["all_mask_area_below_half"], "mask_stability": robot["automatic_stability_check"], "raw_change_robot_overlap_by_horizon": quality["raw_change_robot_overlap_by_horizon"]},
        "prohibited_follow_on": ["dataset expansion", "split construction", "baseline training", "ActionConditionedObjectDynamicsNetwork training", "method-performance claim"],
        "claim_status": "No dense non-robot scene/object-change prediction claim is supported. The result is a reproducible target-validity rejection.",
    }
    write_json(OUT / "final_decision.json", final)
    completion = {
        "schema": "rm-acd-completion-audit-v1",
        "terminal_decision": DECISION,
        "requirements": [
            {"phase": "A0", "requirement": "frozen task specification and source hashes", "status": "pass", "evidence": ["preregistered_config.json", "preregistered_config.sha256", "source_hash_manifest.json", "selected_trajectory_manifest.json"]},
            {"phase": "A1", "requirement": "RGB/depth/camera/alignment audit for selected data", "status": "pass", "evidence": ["modality_inventory.json", "camera_alignment_audit.json", "selected_view_report.md"]},
            {"phase": "A2", "requirement": "robot exclusion audit before model training", "status": "terminal_gate_failed_correctly", "evidence": ["robot_mask_method.json", "robot_mask_audit.json", "robot_mask_manual_review.json", "robot_mask_contact_sheet_manifest.json"]},
            {"phase": "A3", "requirement": "temporally separated multi-horizon targets and no future fair input", "status": "pass_for_audit_only_targets", "evidence": ["target_schema.json", "target_construction_audit.json", "target_manifest.jsonl", "target_statistics.json", "confidence_statistics.json"]},
            {"phase": "A4", "requirement": "target-quality decision before learned baselines", "status": "terminal_gate_failed_at_A2", "evidence": ["target_quality_audit.json", "target_quality_report.md", "target_failure_cases.md"]},
            {"phase": "A5-A9", "requirement": "expansion, splits, baselines, method, robustness", "status": "not_applicable_by_A2_stop_rule", "evidence": ["final_decision.json", "baseline_summary.json", "raw_prediction_manifest.json"]},
            {"phase": "A10", "requirement": "terminal decision and required final package", "status": "pass", "evidence": ["final_decision.json", "final_report.md", "claim_boundary.md", "paper_positioning.md", "next_steps.md", "reproducibility_manifest.json"]},
        ],
        "all_required_terminal_artifacts_present": True,
        "conclusion": "The decision tree is complete: RM_ACD_ROBOT_MASK_INVALID requires stopping before splits, baselines, or method training.",
    }
    write_json(OUT / "completion_audit.json", completion)
    claim = """# Claim boundary

RM-ACD does not support a claim of real-world dense object/scene-change
prediction, failure/success prediction, retrieval, physical counterfactuals,
alternate-action outcomes, metric 3D scene flow, or ActMask method advantage.
It supports only a negative data/target audit: the available automatic robot
exclusion is too broad to create a valid non-robot target on this frozen pilot.
"""
    positioning = """# Paper positioning

Do not use RM-ACD as ActMask method evidence. It is a reproducible benchmark
construction finding: without a validated robot mask or camera/robot geometry
contract, dense future visual change is dominated by robot exclusion error.
"""
    next_steps = """# Next steps

The next scientific decision requires one of the following new capabilities:

1. dataset-provided robot/instance masks aligned to the RGB frames;
2. a verified dual-Franka geometry/kinematics plus camera-to-base calibration
   that projects robot links into the target view; or
3. bounded human robot-vs-object annotations sufficient to validate a frozen
   mask model.

Do not repair the frozen RM-ACD target heuristically, construct another future
retrieval task, or train a method on the invalid pilot labels.
"""
    (OUT / "claim_boundary.md").write_text(claim)
    (OUT / "paper_positioning.md").write_text(positioning)
    (OUT / "next_steps.md").write_text(next_steps)
    docs = OUT / "docs"
    (docs / "rm_acd_results.md").write_text("# RM-ACD results\n\nFinal decision: `RM_ACD_ROBOT_MASK_INVALID`. The required non-robot exclusion gate failed before baseline or method training: the conservative heuristic masks 0.616–0.795 of each image and raw visual change overlaps it by about 0.97.\n")
    (docs / "rm_acd_handoff.md").write_text("# RM-ACD handoff\n\nRM-ACD is rejected before model training. See `robot_mask_manual_review.json`, `target_quality_audit.json`, and `next_steps.md`. The derived pilot targets are audit-only and must not be reused for training.\n")
    report = f"""# RM-ACD final report

## Decision

`{DECISION}` — reject before any baseline or method training.

The A2 robot-region exclusion is unusable. Across the ten selected tasks, the
conservative image-domain mask covers {min(task_mask.values()):.3f}–
{max(task_mask.values()):.3f} of pixels, exceeds the configured 0.50 maximum
for every task, and is not stable under the pre-registered automatic check.
Raw change overlaps the mask by about 0.97 at short, medium and long horizons.
The 20-trajectory, all-task contact-sheet package shows broad workspace and
visible-object removal. A dense non-robot target cannot be accepted.

The pilot has preserved auditable source alignment, temporal separation,
target-schema and confidence artifacts. They are not training labels. No
dataset expansion, split, baseline, checkpoint, or ACD model artifact exists.
"""
    (OUT / "final_report.md").write_text(report)
    target_rows = [json.loads(line) for line in (OUT / "target_manifest.jsonl").read_text().splitlines() if line]
    chosen = [target_rows[0], target_rows[len(target_rows) // 2], target_rows[-1]]
    artifacts = [
        "preregistered_config.json", "source_hash_manifest.json", "modality_inventory.json", "camera_alignment_audit.json",
        "robot_mask_method.json", "robot_mask_audit.json", "robot_mask_manual_review.json", "robot_mask_contact_sheet_manifest.json",
        "target_schema.json", "target_construction_audit.json", "target_statistics.json", "confidence_statistics.json", "target_quality_audit.json",
        "baseline_summary.json", "raw_prediction_manifest.json", "final_decision.json", "final_report.md", "target_quality_report.md",
        "target_failure_cases.md", "claim_boundary.md", "paper_positioning.md", "next_steps.md", "completion_audit.json", "docs/rm_acd_plan.md", "docs/rm_acd_results.md", "docs/rm_acd_handoff.md",
    ]
    reproducibility = {"schema": "rm-acd-reproducibility-manifest-v1", "python": "/home/tzh/conda_envs/actmask/bin/python", "config_sha256": config_sha, "source_hash_manifest_sha256": sha(OUT / "source_hash_manifest.json"), "target_manifest_rows": len(target_rows), "target_artifacts_audit_only": True, "method_trained": False, "sample_input_target_hashes": [{"sample_id": row["sample_id"], "horizon": row["horizon"], "input_path": row["input_path"], "input_sha256": _hash(OUT / row["input_path"]), "target_path": row["target_path"], "target_sha256": _hash(OUT / row["target_path"])} for row in chosen], "tracked_artifact_sha256": {name: _hash(OUT / name) for name in artifacts}, "commands": ["python -m actmask.experiments.rm_acd_freeze_and_modality", "python -m actmask.experiments.rm_acd_preconstruction_clarification", "python -m actmask.experiments.rm_acd_construct_targets", "python -m actmask.experiments.rm_acd_finalize_robot_mask_rejection"]}
    write_json(OUT / "reproducibility_manifest.json", reproducibility)
    append_log("phase_a10_final_decision", decision=DECISION, method_ran=False, baseline_ran=False, target_rows=len(target_rows))
    return final


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, sort_keys=True))
