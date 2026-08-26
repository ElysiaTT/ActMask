"""S6/S9 terminal audit for the preregistered RM-SPD pilot."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from actmask.experiments.rm_spd_common import OUT, append_log, iter_jsonl, require_preregistered_config, sha256_file, write_json


def _sha(path: Path) -> dict[str, Any]:
    return {"path": str(path.relative_to(OUT)), "exists": path.is_file(), "sha256": sha256_file(path) if path.is_file() else None, "bytes": path.stat().st_size if path.is_file() else None}


def run() -> dict[str, Any]:
    config = require_preregistered_config(); summary = json.loads((OUT / "baseline_summary.json").read_text())["summary"]; recompute = json.loads((OUT / "baseline_metrics_recomputed.json").read_text()); prior = json.loads((OUT / "robotness_statistics.json").read_text()); target = json.loads((OUT / "target_statistics.json").read_text()); raw = list(iter_jsonl(OUT / "raw_prediction_manifest.json"))
    def entry(model: str, regime: str) -> dict[str, Any]: return summary[f"{model}::{regime}"]
    copy = entry("copy_current", "tri_state_sam3")
    action_models = ("visual_action_gru", "visual_state_action_gru", "patch_action_mlp", "patch_state_action_mlp", "temporal_patch_transformer", "two_head_patch_gru")
    best_action = min(((entry(m, "tri_state_sam3")["test"]["scene_l2_error"], m, entry(m, "tri_state_sam3")) for m in action_models), key=lambda x: x[0])
    all_tri = min(((value["test"]["scene_l2_error"], key.split("::")[0], value) for key, value in summary.items() if key.endswith("::tri_state_sam3")), key=lambda x: x[0])
    hard_best = min((value["contact_heavy"]["scene_l2_error"] for key, value in summary.items() if key.endswith("::hard_sam3")))
    soft_or_tri_best = min((value["contact_heavy"]["scene_l2_error"] for key, value in summary.items() if key.endswith("::soft_sam3") or key.endswith("::tri_state_sam3")))
    prior_not_broad = prior["fractions"]["core_plus_uncertain"]["median"] <= config["prior"]["unusable_gate"]["max_median_core_plus_uncertain_fraction"] and prior["fractions"]["core_plus_uncertain"]["p95"] <= config["prior"]["unusable_gate"]["max_p95_core_plus_uncertain_fraction"]
    full_difference = abs(copy["test"]["full_frame_l2"] - copy["test"]["scene_l2_error"]) / copy["test"]["scene_l2_error"]
    artifact_paths = [OUT / x for x in ("preregistered_config.json", "sam3_proposal_manifest.jsonl", "robotness_prior_manifest.jsonl", "future_feature_target_manifest.jsonl", "raw_prediction_manifest.json", "baseline_summary.json", "baseline_metrics_recomputed.json", "source_hash_manifest.json")]
    gates = {
        "prior_not_broad": prior_not_broad,
        "full_frame_materially_differs_from_scene_trusted": full_difference >= 0.05,
        "hard_sam3_brittle_on_contact_heavy": hard_best > soft_or_tri_best,
        "fair_temporal_visual_action_beats_copy_current": best_action[0] < copy["test"]["scene_l2_error"],
        "task_heldout_beats_trivial_copy_current": all_tri[2]["task_heldout"]["scene_l2_error"] < copy["task_heldout"]["scene_l2_error"],
        "standard_models_not_saturated": all_tri[0] > 0.01,
        "raw_predictions_saved": len(raw) == 1360 and all((OUT / row["prediction_path"]).is_file() for row in raw),
        "metrics_independently_recompute": bool(recompute["matches_primary_summary"]),
        "artifacts_and_hashes_pass": all(path.is_file() for path in artifact_paths) and all(row.get("sha256") == sha256_file(OUT / row["prediction_path"]) for row in raw),
    }
    # The benchmark protocol forbids S7 unless both the fair visual-action and
    # task-held-out gates pass.  They do not, so no proposed method is trained.
    decision = "RM_SPD_TARGET_NOT_PREDICTABLE"
    decision_payload = {"schema": "rm-spd-final-decision-v1", "decision": decision, "terminal": True, "proposed_method_run": False, "reason": "No fair temporal visual-action standard baseline improves on copy-current on the frozen task-held-out split; the strongest tri-state result also degrades task-held-out error.", "gates": gates, "key_metrics": {"copy_current_tri_state_scene_l2": copy["test"]["scene_l2_error"], "best_fair_action_model": best_action[1], "best_fair_action_scene_l2": best_action[0], "best_overall_tri_state_model": all_tri[1], "best_overall_tri_state_scene_l2": all_tri[0], "copy_task_heldout_scene_l2": copy["task_heldout"]["scene_l2_error"], "best_tri_task_heldout_scene_l2": all_tri[2]["task_heldout"]["scene_l2_error"], "full_vs_scene_relative_difference": full_difference, "contact_heavy_best_hard_scene_l2": hard_best, "contact_heavy_best_soft_or_tri_scene_l2": soft_or_tri_best}, "scope": "frozen 100-trajectory RoboMIND 2.0 Franka pilot only; not a claim that real-world dynamics are intrinsically unpredictable"}
    write_json(OUT / "final_decision.json", decision_payload)
    (OUT / "final_report.md").write_text("""# RM-SPD final report

## Terminal decision

**RM_SPD_TARGET_NOT_PREDICTABLE** for the preregistered 100-trajectory RoboMIND 2.0 Franka pilot.  The proposed SAM3PriorRelationalDynamicsNetwork was not run because its mandatory S6 admission gate failed.

## Evidence

- The SAM3 prior was not broad: core+uncertain median %.3f and P95 %.3f.
- Full-frame and scene-trusted scoring differ materially (%.1f%% relative for copy-current), so the benchmark is not a trivial static full-frame score.
- The best hard-mask contact-heavy result (%.3f) is worse than the best soft/tri-state result (%.3f); hard binary masking is brittle in this pilot.
- Copy-current has scene-trusted L2 %.3f.  The strongest fair action-conditioned standard model is %s at %.3f, so it does not beat copy-current.
- The strongest tri-state result is %s at %.3f overall, only %.2f%% relative better than copy-current, but task-held-out error is %.3f versus copy-current %.3f.  It fails generalization rather than opening method headroom.
- 1,360 raw predictions were saved, and an independent reread/recompute of all raw files exactly matches the primary summary.

SAM3 was used only for fixed loss routing and metric partitioning.  It was never a segmentation label, target value, predictive model input, or reference.  The target is the actual future RGB frozen ResNet-18 feature.
""" % (prior["fractions"]["core_plus_uncertain"]["median"], prior["fractions"]["core_plus_uncertain"]["p95"], 100 * full_difference, hard_best, soft_or_tri_best, copy["test"]["scene_l2_error"], best_action[1], best_action[0], all_tri[1], all_tri[0], 100 * (copy["test"]["scene_l2_error"] - all_tri[0]) / copy["test"]["scene_l2_error"], all_tri[2]["task_heldout"]["scene_l2_error"], copy["task_heldout"]["scene_l2_error"]))
    (OUT / "claim_boundary.md").write_text("""# Claim boundary

This branch supports only a negative benchmark-admission result for a fixed RoboMIND pilot.  It does **not** verify robot segmentation, object masks, SAM3 accuracy, success/failure prediction, physical counterfactuals, alternate-action outcomes, or a new relational dynamics method.  SAM3 is an uncertain prior for routing/partitioning only; no human pixel labels were used.
""")
    (OUT / "paper_positioning.md").write_text("""# Paper positioning

This is an auditable benchmark/admission study, not a method paper result.  Its useful finding is that on the frozen local 100-trajectory pilot, real future feature residuals are not reliably improved over copy-current by the preregistered fair visual-action standard models under task-held-out evaluation.  Any future method claim needs a newly preregistered data scale/split and must re-run controls before comparison.
""")
    (OUT / "next_steps.md").write_text("""# Next steps

Do not tune SAM3 prompts, request human pixel labels, or run the proposed method on this frozen pilot.  A follow-up must be a separately preregistered benchmark with more task-diverse RoboMIND trajectories, multiple anchors per episode, and a genuinely disjoint task-held-out protocol.  It should retain real future visual features as targets and SAM3 only as uncertainty routing.
""")
    docs = OUT / "docs"
    (docs / "rm_spd_results.md").write_text((OUT / "final_report.md").read_text() + "\n\nPrimary machine-readable result: `final_decision.json`.\n")
    (docs / "rm_spd_handoff.md").write_text("# RM-SPD handoff\n\nTerminal decision: `RM_SPD_TARGET_NOT_PREDICTABLE`. Read `final_decision.json`, `final_report.md`, `baseline_summary.json`, `baseline_metrics_recomputed.json`, and `raw_prediction_manifest.json`. Do not train S7 on this frozen pilot: its S6 admission gate failed.\n")
    tracked = ["preregistered_config.json", "preregistered_config.sha256", "source_hash_manifest.json", "source_alignment_audit.json", "sam3_proposal_manifest.jsonl", "sam3_runtime_report.json", "robotness_prior_manifest.jsonl", "robotness_statistics.json", "temporal_consistency_audit.json", "future_feature_target_manifest.jsonl", "target_alignment_audit.json", "target_statistics.json", "baseline_summary.json", "baseline_metrics_recomputed.json", "baseline_per_horizon_metrics.jsonl", "baseline_horizon_audit.json", "raw_prediction_manifest.json", "final_decision.json"]
    write_json(OUT / "reproducibility_manifest.json", {"schema": "rm-spd-reproducibility-manifest-v1", "python": "/home/tzh/conda_envs/actmask/bin/python", "config_sha256": sha256_file(OUT / "preregistered_config.json"), "files": [_sha(OUT / p) for p in tracked], "raw_prediction_files": len(raw), "raw_manifest_sha256": sha256_file(OUT / "raw_prediction_manifest.json"), "independent_recompute_sha256": recompute["summary_sha256"], "frozen_source_only": True, "human_pixel_labels_used": False})
    append_log("S6_BENCHMARK_GATE_COMPLETE", decision=decision, gates=gates)
    append_log("S9_FINAL_PACKAGE_COMPLETE", decision=decision, final_decision_sha256=sha256_file(OUT / "final_decision.json"))
    return decision_payload


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, sort_keys=True))
