"""Freeze VERIFIER-SHORTCUT-AUDIT-V1 before any new model training."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .common import (
    HISTORICAL,
    OUT,
    PYTHON,
    ROOT,
    SEEDS,
    STUDY_ID,
    append_log,
    sha256_file,
    write_json,
)


FAMILIES = {
    "N1_same_task_near_state": "R",
    "N2_local_temporal_permutation": "E",
    "N3_arm_gripper_desynchronization": "E",
    "N4_phase_shifted_same_task": "R",
    "N5_joint_coordination_corruption": "E",
    "N6_endpoint_matched_path_corruption": "E",
    "N7_reciprocal_near_state_context_swap": "B",
    "N8_visual_substate_reciprocal_context_swap": "B",
}


IMPORTED_ARTIFACTS = {
    "rm_series_robomind": [
        "final_package/final_decision.json",
        "processed_subset/data_manifest.json",
        "task_formulations/split_audit_action_future_consistency.json",
        "baseline_report.json",
        "shortcut_audit.json",
        "source_hash_manifest.json",
        "final_package/reproducibility_manifest.json",
    ],
    "rm_acv_pilot166": [
        "final_decision.json",
        "preregistered_config.json",
        "preregistered_config.sha256",
        "anchor_manifest.jsonl",
        "positive_candidate_manifest.jsonl",
        "negative_candidate_manifest.jsonl",
        "benchmark_arrays.npz",
        "negative_family_decision.json",
        "shortcut_feature_report.json",
        "shortcut_predictions.jsonl",
        "episode_split_preregistered.json",
        "task_cv_preregistered.json",
        "split_manifest.json",
        "source_manifest.jsonl",
        "source_hash_manifest.json",
        "alignment_audit.json",
        "anchor_overlap_audit.json",
        "reproducibility_manifest.json",
    ],
    "rm_acv_realneg_v1": [
        "final_decision.json",
        "preregistered_config.json",
        "preregistered_config.sha256",
        "n7_pair_manifest.jsonl",
        "n8_pair_manifest.jsonl",
        "balance_audit.json",
        "action_reuse_audit.json",
        "context_reuse_audit.json",
        "behavioral_validity_report.json",
        "real_family_decision.json",
        "n7_shortcut_report.json",
        "n8_shortcut_report.json",
        "shortcut_predictions.jsonl",
        "import_verification_audit.json",
        "imported_anchor_manifest.jsonl",
        "imported_benchmark_arrays.npz",
        "causal_visual_features.npz",
        "imported_episode_split_preregistered.json",
        "imported_task_cv_preregistered.json",
        "imported_source_hash_manifest.json",
        "reproducibility_manifest.json",
    ],
    "rm_od_future_object_dynamics": [
        "final_decision.json",
        "preregistered_config.json",
        "initial_shortcut_precheck.json",
        "repair1_source_path_precheck.json",
        "baseline_summary.json",
        "metric_recomputation_audit.json",
        "source_hash_manifest.json",
        "reproducibility_manifest.json",
    ],
    "rm_spd_sam3_prior_dynamics": [
        "final_decision.json",
        "preregistered_config.json",
        "baseline_summary.json",
        "baseline_metrics_recomputed.json",
        "split_manifest.json",
        "source_hash_manifest.json",
        "reproducibility_manifest.json",
    ],
}


def artifact_records() -> list[dict[str, Any]]:
    records = []
    for branch, relatives in IMPORTED_ARTIFACTS.items():
        for relative in relatives:
            path = HISTORICAL / branch / relative
            if not path.is_file():
                raise FileNotFoundError(path)
            records.append(
                {
                    "branch": branch,
                    "relative_path": relative,
                    "absolute_path": str(path.resolve()),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return records


def config() -> dict[str, Any]:
    imported = artifact_records()
    return {
        "schema": "verifier-shortcut-audit-preregistered-config-v1",
        "study_id": STUDY_ID,
        "scientific_question": (
            "Do pre-execution robot action verifiers model context-action "
            "compatibility or exploit benchmark-construction shortcuts?"
        ),
        "claim_boundary": {
            "labels": "historical benchmark-construction labels, not physical ground truth",
            "allowed": [
                "shortcut solvability",
                "nuisance dependence",
                "generator/task/source transfer failure",
                "action-only or context-only explanatory power",
                "matching-induced performance change",
                "construction-shift robustness",
                "calibration shift",
            ],
            "forbidden": [
                "physical failure guarantee",
                "physical safety",
                "task success guarantee",
                "true counterfactual failure",
                "real-robot safety implication",
                "N1-N8 as a final compatibility benchmark",
                "PhaseAwareTemporalEnergyVerifier validation",
            ],
        },
        "historical_imports": {
            "branch_paths": {
                branch: str((HISTORICAL / branch).resolve())
                for branch in IMPORTED_ARTIFACTS
            },
            "artifacts": imported,
            "artifact_set_sha256": hashlib.sha256(
                json.dumps(imported, sort_keys=True).encode("utf-8")
            ).hexdigest(),
            "regeneration_forbidden": True,
            "historical_results_mutation_forbidden": True,
        },
        "families": {
            "roles": FAMILIES,
            "group_E": [
                "N2_local_temporal_permutation",
                "N3_arm_gripper_desynchronization",
                "N5_joint_coordination_corruption",
                "N6_endpoint_matched_path_corruption",
            ],
            "group_R": [
                "N1_same_task_near_state",
                "N4_phase_shifted_same_task",
            ],
            "group_B": [
                "N7_reciprocal_near_state_context_swap",
                "N8_visual_substate_reciprocal_context_swap",
            ],
            "N7_N8_role": (
                "diagnostic only: action/context balance, source dependence, "
                "reciprocal consistency, and construction shift"
            ),
        },
        "representation": {
            "history_steps": 8,
            "state_dimension": 32,
            "action_horizon": 16,
            "action_dimension": 32,
            "language_dimension": 128,
            "visual_history_steps": 8,
            "visual_token_dimension": 22,
            "visual_view": "camera_front frozen RGB mean/std plus 4x4 luminance map",
            "future_observations_forbidden": True,
            "action_tensors_and_historical_labels_immutable": True,
        },
        "nuisance_features": {
            "action": [
                "per_dimension_mean",
                "per_dimension_variance",
                "absolute_magnitude",
                "velocity",
                "acceleration",
                "jerk",
                "smoothness",
                "total_displacement",
                "start_action",
                "end_action",
                "endpoint_difference",
                "active_joint_count",
                "arm_gripper_event_count",
                "left_right_arm_energy",
                "gripper_timing",
                "zero_padding_fraction",
                "action_horizon",
            ],
            "context": [
                "normalized_episode_progress",
                "episode_length",
                "task_identity",
                "current_state",
                "recent_state_velocity",
                "current_gripper_state",
                "visual_brightness",
                "visual_contrast",
                "global_color_statistics",
                "static_border_features",
                "current_frame_visual_token",
                "history_frame_difference",
            ],
            "provenance": [
                "family_identity",
                "negative_generator",
                "source_episode",
                "source_trajectory_length",
                "path_metadata",
                "manifest_order",
                "pair_mining_rank",
                "source_policy",
                "camera_identity",
                "scene_identifier",
            ],
            "language": [
                "task_string_identity",
                "language_embedding",
                "instruction_length",
                "shuffled_language",
                "empty_language",
            ],
            "audit_only": True,
        },
        "audit_environments": {
            "E1": "within-family IID, episode disjoint, N1-N6",
            "E2": "leave-one-family-out for N1,N2,N4,N5,N6",
            "E3": "train Group E, test Group R",
            "E4": "train Group R, test Group E",
            "E5": "action-summary-matched test",
            "E6": "progress/duration/horizon-matched test",
            "E7": "context and source episode held out together",
            "E8": "frozen five-fold task group CV",
            "E9": "N7/N8 reciprocal diagnostic",
            "E10": "correct/shuffled/empty/task-ID-only language",
            "E11": "full/current/difference/border/center/global visual tokens",
            "E12": "balanced and shifted label/family/task priors",
        },
        "matching": {
            "algorithm": (
                "deterministic one-to-one greedy minimum standardized Euclidean "
                "distance across opposite labels; lexical sample-ID tie-break"
            ),
            "normalization": "training partition only",
            "caliper": (
                "90th percentile of development/train opposite-label nearest-neighbor "
                "distance, frozen before test matching"
            ),
            "E5_features": [
                "action_magnitude",
                "variance",
                "smoothness",
                "jerk",
                "total_displacement",
                "start_action",
                "end_action",
                "gripper_event_count",
            ],
            "E6_features": [
                "normalized_progress",
                "episode_duration",
                "action_horizon",
            ],
            "heldout_model_performance_used": False,
        },
        "splits": {
            "episode": "imported unchanged from RM-ACV-PILOT166",
            "task_cv": "imported unchanged five-fold complete-task groups",
            "source": (
                "sample eligible only when context episode and negative source "
                "episode share the same imported split"
            ),
            "candidate_mixing_across_splits": False,
            "threshold_selection": "validation only",
        },
        "shortcut_controls": {
            "models": [
                "logistic_regression_combined_nuisance",
                "shallow_mlp_combined_nuisance",
                "temporal_action_only_TCN",
                "current_state_action_summary_MLP",
                "provenance_only_classifier",
                "combined_non_relational_nuisance_classifier",
            ],
            "seed": 17,
            "audit_only_inputs_allowed": True,
        },
        "verifier_architectures": {
            "StateActionTCN": {
                "inputs": ["causal_state_history", "candidate_action"],
                "explicit_interaction": True,
            },
            "TaskConditionedActionTransformer": {
                "inputs": [
                    "causal_state_history",
                    "task_instruction",
                    "candidate_action",
                ],
                "explicit_interaction": True,
            },
            "VisualStateActionTransformer": {
                "inputs": [
                    "frozen_causal_visual_tokens",
                    "causal_state_history",
                    "task_instruction",
                    "candidate_action",
                ],
                "explicit_interaction": True,
            },
            "ContrastiveContextActionVerifier": {
                "inputs": [
                    "causal_state_history",
                    "frozen_causal_visual_tokens",
                    "task_instruction",
                    "candidate_action",
                ],
                "explicit_interaction": "dot-product compatibility",
            },
            "CrossAttentionEnergyVerifier": {
                "inputs": [
                    "causal_state_history",
                    "frozen_causal_visual_tokens",
                    "task_instruction",
                    "candidate_action",
                ],
                "explicit_interaction": "context-action cross-attention",
            },
        },
        "forbidden_fair_inputs": [
            "family_identity",
            "source_episode",
            "source_path",
            "progress_scalar",
            "matching_distance",
            "mining_rank",
            "reason_code",
            "evidence_type",
            "future_state",
            "future_rgb",
        ],
        "training": {
            "seeds": list(SEEDS),
            "optimizer": "AdamW",
            "learning_rate": 0.002,
            "weight_decay": 0.0003,
            "batch_size": 512,
            "epochs_max": 40,
            "early_stop_patience": 6,
            "selection_metric": "validation balanced accuracy",
            "gradient_clip": 2.0,
            "normalization": "train partition only, separately per train specification",
            "train_specs": [
                "pooled_N1_to_N6",
                "within_family_N1_to_N6",
                "leave_one_family_out_N1_N2_N4_N5_N6",
                "group_E_to_R",
                "group_R_to_E",
                "task_cv_fold_0_to_4",
            ],
            "identical_protocol_across_architectures": True,
        },
        "metrics": {
            "standard": [
                "balanced_accuracy",
                "AUROC",
                "AUPRC",
                "ECE",
                "Brier",
                "false_accept_rate",
                "false_reject_rate",
            ],
            "shortcut": [
                "generator_transfer_gap",
                "easy_to_real_gap",
                "action_matching_gap",
                "progress_matching_gap",
                "source_held_out_gap",
                "relational_gain",
                "language_reliance_delta",
                "nuisance_score_predictability",
                "calibration_shift",
                "family_transfer_matrix",
            ],
            "confidence_unit": "episode or task group, never sample row",
        },
        "shortcut_evidence_gate": {
            "architectures_with_easy_IID_BA_ge_0_65_min": 2,
            "shortcut_indicators_per_architecture_min": 2,
            "architectures_meeting_indicator_count_min": 2,
            "indicator_thresholds": {
                "generator_heldout_drop": 0.10,
                "easy_to_real_drop": 0.10,
                "action_matched_drop": 0.08,
                "progress_matched_drop": 0.08,
                "source_heldout_drop": 0.08,
                "nuisance_control_within_full": 0.03,
                "nuisance_score_probe_r2": 0.30,
                "family_representation_probe_BA": 0.70,
            },
            "seeds_min": 2,
            "tasks_min": 2,
            "independent_recompute_required": True,
        },
        "mitigation": {
            "conditional_on": "VSA_SYSTEMATIC_SHORTCUT_SUPPORTED",
            "name": "ShortcutResistantVerifierTraining",
            "base_selection": (
                "highest validation-balanced-accuracy fair cross-attention or "
                "Transformer model; lexical tie-break"
            ),
            "components": [
                "family_and_task_balanced_sampling",
                "Group_DRO_over_family_task_source",
                "gradient_reversal_heads_for_family_magnitude_progress_source",
                "matched_pair_consistency",
            ],
            "lambda_group": 0.20,
            "lambda_adversarial": 0.10,
            "lambda_match": 0.10,
            "ablations": [
                "ERM",
                "family_balanced_only",
                "task_balanced_only",
                "Group_DRO_only",
                "nuisance_adversarial_only",
                "matched_training_only",
                "full",
            ],
            "seeds": list(SEEDS),
            "acceptance": {
                "generator_heldout_BA_gain_min": 0.04,
                "easy_to_real_BA_gain_min": 0.04,
                "one_matched_environment_gain_min": 0.04,
                "task_heldout_regression_max": 0.02,
                "IID_regression_max": 0.03,
                "families_improved_min": 3,
                "tasks_improved_min": 3,
                "nuisance_predictability_must_decrease": True,
                "calibration_or_false_accept_must_improve": True,
                "three_seed_and_recompute_required": True,
            },
        },
        "resource_limits": {
            "python": str(PYTHON),
            "GPU_hours_max": 20,
            "new_artifacts_GiB_max": 30,
            "learned_verifier_architectures_max": 5,
            "implementation_repairs_per_model_max": 1,
            "large_foundation_model_downloads": 0,
            "world_models": 0,
            "video_generation": 0,
            "SAM3_inference": 0,
            "new_robot_masks": 0,
            "unrelated_process_termination_forbidden": True,
        },
        "terminal_decisions": [
            "VSA_HISTORICAL_ARTIFACTS_INCOMPLETE",
            "VSA_NO_LEARNABLE_IID_TASK",
            "VSA_NO_SYSTEMATIC_SHORTCUT_EVIDENCE",
            "VSA_SHORTCUT_EVIDENCE_TASK_SPECIFIC",
            "VSA_SYSTEMATIC_SHORTCUT_SUPPORTED",
        ],
    }


def freeze(output: str | Path = OUT) -> dict[str, Any]:
    target = Path(output)
    target.mkdir(parents=True, exist_ok=True)
    path = target / "preregistered_config.json"
    hash_path = target / "preregistered_config.sha256"
    value = config()
    serialized = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    expected = hashlib.sha256(serialized).hexdigest()
    if path.exists():
        if sha256_file(path) != expected:
            raise RuntimeError("existing preregistration differs; refusing overwrite")
    else:
        write_json(path, value)
    if hash_path.exists():
        if hash_path.read_text(encoding="utf-8").strip() != expected:
            raise RuntimeError("existing preregistration hash mismatch")
    else:
        hash_path.write_text(expected + "\n", encoding="utf-8")
    if not (target / "run_log.jsonl").exists():
        append_log(
            "V0_PREREGISTRATION_FROZEN",
            preregistered_config_sha256=expected,
            imported_artifacts=len(value["historical_imports"]["artifacts"]),
            gpu_training_started=False,
        )
    return {
        "config": str(path),
        "sha256": expected,
        "imported_artifacts": len(value["historical_imports"]["artifacts"]),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(OUT))
    args = parser.parse_args()
    print(json.dumps(freeze(args.output_dir), indent=2, sort_keys=True))
