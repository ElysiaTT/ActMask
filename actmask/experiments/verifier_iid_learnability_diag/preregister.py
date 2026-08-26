"""Freeze the IID-learnability diagnosis before any new model training."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .common import (
    ARCHITECTURES,
    EASY_FAMILIES,
    OUT,
    PYTHON,
    SEEDS,
    SOURCE,
    SPECS,
    STUDY_ID,
    append_log,
    directory_snapshot,
    read_json,
    sha256_file,
    sha256_json,
    verify_snapshot,
    write_json,
)


SNAPSHOT = OUT / "source_preservation_snapshot.json"
CONFIG = OUT / "preregistered_config.json"
CONFIG_HASH = OUT / "preregistered_config.sha256"


def _source_inputs() -> list[dict[str, Any]]:
    names = (
        "preregistered_config.json",
        "final_decision.json",
        "reproducibility_manifest.json",
        "run_log.jsonl",
        "unified_audit_manifest.jsonl",
        "unified_audit_arrays.npz",
        "unified_audit_hash_manifest.json",
        "nuisance_feature_schema.json",
        "nuisance_features.npz",
        "audit_environment_manifest.json",
        "audit_environment_indices.npz",
        "verifier_config.json",
        "verifier_summary.json",
        "verifier_per_seed.json",
        "shortcut_control_summary.json",
        "phase_v10_decision.json",
    )
    return [
        {
            "path": str((SOURCE / name).resolve()),
            "relative_path": name,
            "bytes": (SOURCE / name).stat().st_size,
            "sha256": sha256_file(SOURCE / name),
        }
        for name in names
    ]


def _config(snapshot_rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema": "vild-preregistered-config-v1",
        "study_id": STUDY_ID,
        "frozen_before_new_training": True,
        "scientific_objective": (
            "Diagnose why the five frozen fair verifier architectures did not "
            "learn N2/N3/N5/N6 within-family IID construction labels while "
            "audit-only current-state/action-summary controls were predictive."
        ),
        "claim_boundary": {
            "labels": "historical benchmark-construction labels, not physical truth",
            "physical_validity_labels_created": False,
            "robot_safety_claims_allowed": False,
            "N1_to_N8_relabeling_allowed": False,
        },
        "source_study": {
            "path": str(SOURCE.resolve()),
            "must_remain_byte_identical": True,
            "snapshot_file": str(SNAPSHOT.resolve()),
            "snapshot_records": len(snapshot_rows),
            "snapshot_bytes": sum(row["bytes"] for row in snapshot_rows),
            "snapshot_sha256": sha256_json(snapshot_rows),
            "key_inputs": _source_inputs(),
        },
        "families": list(EASY_FAMILIES),
        "training_specs": list(SPECS),
        "architectures": list(ARCHITECTURES),
        "additional_verifier_architectures": 0,
        "final_seeds": list(SEEDS),
        "diagnosis_order": [
            "A_alignment_normalization_split_batch_balance",
            "B_tiny_and_full_training_memorization",
            "C_positive_and_negative_sanity_controls",
            "D_gradient_optimization_logit_threshold_diagnostics",
            "E_fair_input_vs_summary_feature_comparison",
            "F_root_cause_decision",
            "final_frozen_IID_confirmation",
        ],
        "alignment_audit": {
            "required_zero_mismatch_checks": [
                "manifest row equals unified_array_index",
                "manifest construction_label equals label array",
                "manifest family equals family_index mapping",
                "manifest action SHA256 equals float32 candidate_action bytes",
                "context_anchor_index and source_anchor_index agree with provenance",
                "positive/negative rows for N2/N3/N5/N6 share context tensor per anchor",
                "current_state is tested against the final history_state step",
                "nuisance label equals unified label",
            ],
            "pass_rule": "zero mismatches across all 12,676 rows",
        },
        "normalization_split_batch_audit": {
            "normalization": "recompute mean/std from train partition only",
            "std_floor": 1e-6,
            "comparison_to_frozen_normalizers_absolute_tolerance": 1e-6,
            "forbidden_statistics": ["validation", "test"],
            "split_overlap_required": 0,
            "class_balance_required_positive_rate": 0.5,
            "class_balance_absolute_tolerance": 0.0,
            "batch_audit": (
                "for each spec and seeds 17/29/43, epoch-0 permutation must "
                "contain every train index exactly once and no other index"
            ),
        },
        "tiny_memorization": {
            "scope": "each architecture x each easy family, seed 17",
            "selection": (
                "sort train sample IDs independently by label and take the "
                "first 32 positives and first 32 negatives"
            ),
            "samples": 64,
            "optimizer": "AdamW",
            "learning_rate": 0.003,
            "weight_decay": 0.0,
            "batch_size": 64,
            "maximum_updates": 1000,
            "evaluation_interval_updates": 25,
            "precision": "float32",
            "gradient_clip": 2.0,
            "pass": {
                "training_balanced_accuracy_min": 0.98,
                "training_AUROC_min": 0.995,
                "training_BCE_max": 0.10,
            },
        },
        "full_training_memorization": {
            "scope": "each architecture x each easy family, seed 17",
            "optimizer": "AdamW",
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "batch_size": 128,
            "maximum_epochs": 300,
            "precision": "float32",
            "gradient_clip": 2.0,
            "early_stop_after_consecutive_pass_epochs": 3,
            "pass": {
                "training_balanced_accuracy_min": 0.95,
                "training_AUROC_min": 0.98,
                "training_BCE_max": 0.15,
            },
        },
        "sanity_controls": {
            "positive_observable_signal": {
                "scope": "each architecture, within_n2 train/validation, seed 17",
                "sample_selection": (
                    "SHA256 rank by sample ID; first 256 train and 128 validation"
                ),
                "synthetic_labels": (
                    "alternating zero/one after SHA256 ranking, balanced separately "
                    "within train and validation"
                ),
                "injection": (
                    "set normalized candidate_action channel 0 at every time step "
                    "to -4 for label 0 and +4 for label 1"
                ),
                "test_partition_used": False,
                "maximum_epochs": 100,
                "batch_size": 128,
                "learning_rate": 0.002,
                "pass": {
                    "validation_balanced_accuracy_min": 0.98,
                    "validation_AUROC_min": 0.995,
                },
            },
            "shuffled_label_negative": {
                "scope": "each architecture, within_n2 train/validation, seed 918273",
                "permutation": "independent deterministic permutation by partition",
                "test_partition_used": False,
                "maximum_epochs": 40,
                "pass": {
                    "validation_balanced_accuracy_max": 0.60,
                    "validation_AUROC_range": [0.35, 0.65],
                },
            },
            "constant_input_negative": {
                "scope": "each architecture, within_n2 train/validation, seed 17",
                "override": "all required normalized input tensors are zero",
                "test_partition_used": False,
                "maximum_epochs": 10,
                "pass": {
                    "validation_balanced_accuracy_exact": 0.5,
                    "validation_logit_std_max": 1e-6,
                },
            },
        },
        "gradient_flow": {
            "scope": "each architecture x each easy family, seed 17",
            "batch_selection": "first 32 train sample IDs per label",
            "precision": "float32",
            "optimizer_step_learning_rate": 0.001,
            "pass": {
                "finite_logits_loss_gradients": True,
                "total_gradient_norm_min": 1e-8,
                "parameter_tensor_gradient_coverage_min": 0.80,
                "parameter_update_norm_min": 1e-8,
            },
        },
        "historical_optimization_diagnostics": {
            "runs": 60,
            "inspect": [
                "train BCE curves",
                "validation BA curves",
                "best epoch and early stop",
                "raw logit mean/std/range",
                "validation-only threshold",
                "validation-only temperature",
                "test AUROC and oracle-threshold BA for diagnosis only",
                "parameter change from seeded initialization",
            ],
        },
        "alternate_schedule_diagnostic": {
            "scope": "each architecture x each easy family, seed 17 only",
            "architecture_unchanged": True,
            "optimizer": "AdamW",
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "batch_size": 128,
            "maximum_epochs": 200,
            "early_stop_patience": 30,
            "selection": "validation BCE only",
            "precision": "float32",
            "gradient_clip": 2.0,
            "test_not_used_for_selection": True,
        },
        "representation_mismatch": {
            "recompute_all_334_action_summary_features": True,
            "required_max_absolute_recomputation_error": 1e-6,
            "verify_current_state_available_in_history": True,
            "diagnostic_controls": [
                "engineered current-state plus 334 action summaries",
                "flattened raw candidate action plus current state",
            ],
            "control_architecture": "two-layer MLP diagnostic control, not a verifier",
            "control_seeds": [17, 29, 43],
            "selection": "validation BCE",
            "threshold": "validation-only BA optimum",
            "test_not_used_for_selection": True,
        },
        "final_confirmation": {
            "scope": "5 architectures x 4 easy families x 3 seeds = 60 runs",
            "architecture_definitions": "unchanged frozen VSA definitions",
            "optimizer": "AdamW",
            "learning_rate": 0.002,
            "weight_decay": 0.0003,
            "batch_size": 512,
            "maximum_epochs": 40,
            "early_stop_patience": 6,
            "gradient_clip": 2.0,
            "selection": "validation balanced accuracy",
            "calibration": "validation-only scalar temperature",
            "threshold": "validation-only balanced-accuracy optimum",
            "normalization": "train partition only",
            "precision": "CUDA bfloat16 autocast",
            "heldout_test_used_for_selection": False,
            "qualifying_cell": (
                "architecture-family BA >= 0.65 in at least two of seeds 17/29/43"
            ),
            "study_pass": "at least two architectures have a qualifying easy family",
        },
        "calibration_failure_gate": {
            "ranking_AUROC_min": 0.70,
            "required_seeds": 2,
            "frozen_BA_max": 0.65,
            "oracle_test_threshold_BA_gain_min_diagnostic_only": 0.10,
        },
        "root_cause_decision_order": [
            "VILD_ALIGNMENT_OR_IMPLEMENTATION_FAULT_CONFIRMED",
            "VILD_IMPLEMENTATION_FAULT_REPAIRED_AND_IID_RESTORED",
            "VILD_CALIBRATION_THRESHOLD_FAILURE",
            "VILD_OPTIMIZATION_TRAINING_FAILURE",
            "VILD_REPRESENTATION_INDUCTIVE_BIAS_MISMATCH",
            "VILD_CAPACITY_OR_ARCHITECTURE_FAILURE",
            "VILD_IID_LEARNABILITY_RESTORED_WITHOUT_REPAIR",
            "VILD_INSUFFICIENT_EVIDENCE",
        ],
        "decision_rules": {
            "alignment_or_implementation_fault": (
                "any required zero-mismatch invariant fails, or a deterministic "
                "pipeline defect is reproduced and causally isolated"
            ),
            "optimization_training_failure": (
                "alignment and sanity pass; original confirmation fails; alternate "
                "unchanged-architecture schedule reaches the frozen IID gate"
            ),
            "representation_inductive_bias_mismatch": (
                "alignment, gradients, positive sanity and memorization pass; "
                "engineered and/or flattened input-equivalent controls generalize "
                "at BA>=0.65; original and alternate fair verifiers do not qualify"
            ),
            "capacity_or_architecture_failure": (
                "alignment and gradients pass but at least two architectures fail "
                "tiny or full-training memorization acceptance"
            ),
            "calibration_threshold_failure": (
                "at least two architectures meet the frozen calibration-failure "
                "gate across at least two seeds"
            ),
            "iid_restored": "the final-confirmation study pass is true",
            "insufficient": "none of the preceding mutually ordered rules is proven",
        },
        "repair_policy": {
            "maximum_isolated_repairs": 1,
            "requires_confirmed_bug": True,
            "must_document_before_after_evidence": True,
            "must_not_modify_source_study": True,
            "post_repair_runs_use_new_versioned_paths": True,
        },
        "forbidden": [
            "held-out generator/task/source/matching tests for model selection",
            "ShortcutResistantVerifierTraining",
            "SAM3 inference",
            "new robot masks",
            "new physical-validity labels",
            "large foundation model download",
            "world model",
            "robot-safety claim",
            "termination of unrelated GPU processes",
        ],
        "resource_limits": {
            "GPU_hours_max": 5.0,
            "new_artifacts_GiB_max": 10.0,
            "additional_verifier_architectures_max": 0,
            "python": str(PYTHON),
            "seeds": list(SEEDS),
        },
        "required_outputs": [
            "preregistered_config.json",
            "alignment_audit.json",
            "normalization_and_split_audit.json",
            "sanity_control_report.json",
            "memorization_report.json",
            "gradient_flow_report.json",
            "optimization_diagnostics.json",
            "representation_mismatch_report.json",
            "per_architecture_diagnosis.json",
            "iid_learnability_results.json",
            "root_cause_decision.json",
            "final_report.md",
            "executive_summary_zh.md",
            "next_steps.md",
            "reproducibility_manifest.json",
            "run_log.jsonl",
        ],
    }


def run() -> dict[str, Any]:
    if not SOURCE.is_dir():
        raise FileNotFoundError(SOURCE)
    if CONFIG.exists() or CONFIG_HASH.exists():
        if not CONFIG.exists() or not CONFIG_HASH.exists():
            raise RuntimeError("partial preregistration exists")
        expected = CONFIG_HASH.read_text(encoding="utf-8").strip()
        actual = sha256_file(CONFIG)
        if expected != actual:
            raise RuntimeError("frozen preregistration hash mismatch")
        snapshot_rows = read_json(SNAPSHOT)["files"]
        source_check = verify_snapshot(SOURCE, snapshot_rows)
        if not source_check["pass"]:
            raise RuntimeError(f"source study changed: {source_check}")
        return {
            "already_frozen": True,
            "config_sha256": actual,
            "source_snapshot_sha256": source_check["snapshot_sha256"],
            "source_files": source_check["expected_files"],
        }

    OUT.mkdir(parents=True, exist_ok=True)
    snapshot_rows = directory_snapshot(SOURCE)
    snapshot_payload = {
        "schema": "vild-source-preservation-snapshot-v1",
        "study_id": STUDY_ID,
        "source_path": str(SOURCE.resolve()),
        "files": snapshot_rows,
        "file_count": len(snapshot_rows),
        "total_bytes": sum(row["bytes"] for row in snapshot_rows),
        "tree_sha256": sha256_json(snapshot_rows),
    }
    write_json(SNAPSHOT, snapshot_payload)
    config = _config(snapshot_rows)
    write_json(CONFIG, config)
    config_sha256 = sha256_file(CONFIG)
    CONFIG_HASH.write_text(config_sha256 + "\n", encoding="utf-8")
    append_log(
        "PREREGISTRATION_FROZEN",
        config_sha256=config_sha256,
        source_snapshot_sha256=snapshot_payload["tree_sha256"],
        source_files=snapshot_payload["file_count"],
        source_bytes=snapshot_payload["total_bytes"],
        new_training_started=False,
        GPU_hour_limit=5.0,
        artifact_GiB_limit=10.0,
    )
    return {
        "already_frozen": False,
        "config_sha256": config_sha256,
        "source_snapshot_sha256": snapshot_payload["tree_sha256"],
        "source_files": snapshot_payload["file_count"],
        "source_bytes": snapshot_payload["total_bytes"],
    }


def main() -> None:
    print(json.dumps(run(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
