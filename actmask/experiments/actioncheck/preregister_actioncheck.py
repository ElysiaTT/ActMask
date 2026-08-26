"""Freeze the ACTIONCHECK-PROPOSAL-DATA-V1 preregistration before training."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .common import (
    EVIDENCE_TYPES,
    OUT,
    PYTHON,
    REJECT_REASON_CODES,
    canonical_json,
    sha256_file,
    write_json,
)


STUDY_ID = "ACTIONCHECK-PROPOSAL-DATA-V1"
SEED = 1701


def preregistered_config() -> dict[str, Any]:
    """Return the immutable scientific and computational contract."""
    return {
        "study_id": STUDY_ID,
        "schema": "actioncheck-preregistered-config-v1",
        "schema_version": "actioncheck-context-group-v1",
        "scientific_question": (
            "Can an evidence-backed proposal-level benchmark measure whether a "
            "candidate robot action chunk is compatible with its causal context?"
        ),
        "context_representation": {
            "unit": "one causal context shared by multiple candidate action chunks",
            "required": [
                "task_id",
                "task_instruction",
                "episode_id",
                "timestamp",
                "history",
                "split_group",
                "provenance",
            ],
            "history": [
                "causal RGB references or causal visual summaries",
                "causal robot/simulator state",
                "gripper state",
                "optional preceding action history",
                "history length",
                "frequency",
            ],
            "post_execution_observation_as_context_forbidden": True,
        },
        "candidate_action_representation": {
            "dtype": "finite float32",
            "shape": "[action_horizon, action_dim]",
            "minimum_horizon": 1,
            "minimum_dimension": 1,
            "hash": "sha256(int64 shape bytes || C-order float32 bytes)",
            "all_candidates_for_context_atomic": True,
        },
        "labels": {
            "allowed": ["accept", "reject", "uncertain"],
            "primary_train_eval": ["accept", "reject"],
            "uncertain_retained_for_analysis": True,
        },
        "allowed_label_sources": list(EVIDENCE_TYPES),
        "allowed_reject_reason_codes": list(REJECT_REASON_CODES),
        "minimum_evidence_requirements": {
            "accept_or_reject_requires_allowed_evidence_type": True,
            "evidence_record_required": True,
            "allowed_primary_basis": [
                "demonstration",
                "real execution",
                "simulator execution",
                "operator intervention",
                "expert review",
                "explicit frozen rule checker",
                "policy timeout",
            ],
            "forbidden_sole_basis": [
                "logged trajectory mismatch",
                "source path",
                "action distance",
                "synthetic perturbation",
                "file name",
                "episode ID",
                "arbitrary logged action swap",
            ],
            "true_alternate_world_claim_requires_execution_or_simulation": True,
        },
        "pilot_admission": {
            "minimum_context_groups": 100,
            "minimum_average_candidates_per_context": 4.0,
            "minimum_mixed_label_context_fraction": 0.80,
            "minimum_tasks": 5,
            "minimum_candidate_sources": 2,
            "minimum_evidence_types": 2,
            "minimum_labeled_candidates": 500,
        },
        "shortcut_controls": [
            "random",
            "task_only",
            "progress_only",
            "context_only",
            "action_only",
            "action_summary_only",
            "state_only",
            "visual_only",
            "policy_source_only",
            "evidence_type_only",
            "reason_code_only",
            "metadata_only",
            "combined_non_relational",
        ],
        "shortcut_gates": {
            "policy_source_only_balanced_accuracy_max": 0.53,
            "metadata_only_balanced_accuracy_max": 0.53,
            "action_only_balanced_accuracy_max": 0.55,
            "context_only_balanced_accuracy_max": 0.55,
            "action_summary_only_balanced_accuracy_max": 0.56,
            "combined_non_relational_balanced_accuracy_max": 0.60,
            "candidate_source_label_contribution_max": 0.80,
            "single_reason_code_reject_fraction_max": 0.40,
        },
        "audit_only_fields_forbidden_as_method_inputs": [
            "reason_codes",
            "evidence_type",
            "label_source",
            "source path",
        ],
        "split_rules": {
            "atomic_unit": "context_id",
            "candidate_level_random_split": False,
            "schemes": [
                "episode_held_out",
                "task_held_out",
                "object_composition_held_out",
                "policy_source_held_out",
                "scene_environment_held_out",
            ],
            "validation_and_test_threshold_policy": (
                "select classification threshold on validation only; report test once"
            ),
            "grouped_confidence_interval_unit": "context_id or episode_id",
        },
        "metrics": {
            "pairwise": [
                "balanced_accuracy",
                "AUROC",
                "AUPRC",
                "false_accept_rate",
                "false_reject_rate",
            ],
            "grouped_ranking": [
                "Recall@1",
                "MRR",
                "NDCG",
                "positive_negative_score_margin",
                "best_accept_rank",
            ],
            "calibration_selective": [
                "ECE",
                "Brier_score",
                "risk_coverage_curve",
                "selective_accuracy@100/90/80/70%",
                "false_accept_risk_at_fixed_coverage",
            ],
            "generalization": [
                "episode",
                "task",
                "object/composition",
                "policy source",
                "scene/environment",
                "per task",
                "per policy source",
                "per reason code (audit)",
                "per evidence type (audit)",
            ],
        },
        "allowed_pilot_baselines": [
            "action_only",
            "context_only",
            "state_action_TCN",
            "task_conditioned_action_Transformer",
            "visual_state_action_Transformer",
            "contrastive_context_action_encoder",
        ],
        "method_authorization_conditions": {
            "method": "PhaseAwareTemporalEnergyVerifier",
            "authorized_in_this_study": False,
            "separate_future_goal_required": True,
            "minimum_context_groups_pilot": 300,
            "minimum_context_groups_paper": 1000,
            "minimum_average_candidates": 4.0,
            "all_primary_contexts_mixed_labels": True,
            "action_only_BA_max": 0.55,
            "context_only_BA_max": 0.55,
            "policy_source_only_BA_max": 0.53,
            "combined_non_relational_BA_max": 0.60,
            "minimum_fair_baseline_BA": 0.62,
            "minimum_recall_at_1_above_chance": 0.10,
            "held_out_evaluation_required": True,
            "raw_predictions_and_metric_recompute_required": True,
        },
        "decision_precedence": [
            "ACTIONCHECK_PROPOSAL_DATA_NOT_AVAILABLE",
            "ACTIONCHECK_PROPOSAL_LABELS_INVALID",
            "ACTIONCHECK_PROPOSAL_SHORTCUT_DOMINATED",
            "ACTIONCHECK_PROPOSAL_TARGET_NOT_RECOVERABLE",
            "ACTIONCHECK_PROPOSAL_BENCHMARK_READY_FOR_METHOD",
        ],
        "resource_limits": {
            "python": str(PYTHON),
            "seed": SEED,
            "gpu_required_for_training": True,
            "gpu_count_max": 1,
            "baseline_epochs_max": 50,
            "shortcut_epochs_max": 160,
            "baseline_training_jobs_max": 12,
            "phaseaware_training_jobs_max": 0,
            "unrelated_process_termination_forbidden": True,
            "output_root": str(OUT),
        },
        "claim_boundary": {
            "allowed": [
                "proposal-level action compatibility verification",
                "candidate reranking",
                "selective rejection",
                "false-accept risk evaluation",
                "benchmark and data protocol design",
            ],
            "forbidden": [
                "guaranteed physical safety",
                "guaranteed task success",
                "causal counterfactual correctness",
                "unexecuted alternate-world outcomes",
                "object dynamics prediction",
                "arbitrary logged swaps are negatives",
                "synthetic perturbations are valid primary negatives",
            ],
        },
    }


def _append_initial_log(output: Path, config_hash: str) -> None:
    path = output / "run_log.jsonl"
    if path.exists() and path.stat().st_size:
        return
    record = {
        "sequence": 0,
        "study_id": STUDY_ID,
        "event": "P0_PREREGISTRATION_FROZEN",
        "config_sha256": config_hash,
        "previous_record_sha256": None,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    record["record_sha256"] = hashlib.sha256(
        canonical_json(record).encode("utf-8")
    ).hexdigest()
    path.write_text(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def freeze(output_dir: str | Path = OUT) -> dict[str, str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "preregistered_config.json"
    hash_path = output / "preregistered_config.sha256"
    value = preregistered_config()
    expected = hashlib.sha256(
        (
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n"
        ).encode("utf-8")
    ).hexdigest()
    if config_path.exists():
        observed = sha256_file(config_path)
        if observed != expected:
            raise RuntimeError(
                "existing preregistration differs; refusing to overwrite it"
            )
    else:
        write_json(config_path, value)
    if hash_path.exists():
        recorded = hash_path.read_text(encoding="utf-8").strip()
        if recorded != expected:
            raise RuntimeError("existing preregistration hash is inconsistent")
    else:
        hash_path.write_text(expected + "\n", encoding="utf-8")
    _append_initial_log(output, expected)
    return {
        "config": str(config_path),
        "sha256": expected,
        "run_log": str(output / "run_log.jsonl"),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(OUT))
    args = parser.parse_args()
    print(json.dumps(freeze(args.output_dir), indent=2, sort_keys=True))
