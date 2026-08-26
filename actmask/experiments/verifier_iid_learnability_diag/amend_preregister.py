"""Transparent pre-training amendment of the initial VILD preregistration."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from .common import (
    OUT,
    ROOT,
    SEEDS,
    SOURCE,
    STUDY_ID,
    append_log,
    read_json,
    read_jsonl,
    sha256_file,
    verify_bound_source_snapshot,
    write_json,
)


CONFIG = OUT / "preregistered_config.json"
CONFIG_HASH = OUT / "preregistered_config.sha256"
INITIAL_CONFIG = OUT / "preregistered_config_initial.json"
INITIAL_HASH = OUT / "preregistered_config_initial.sha256"
AMENDMENT = OUT / "preregistration_amendment.json"
EXPECTED_INITIAL_HASH = (
    "7b8819a4dd652d3a5952d100483fe10096e08f49d13db41fd8e7999051865f63"
)
IMPLEMENTATION_PATHS = (
    "actmask/experiments/verifier_shortcut_audit/modeling.py",
    "actmask/experiments/verifier_shortcut_audit/metrics.py",
    "actmask/experiments/verifier_iid_learnability_diag/common.py",
    "actmask/experiments/verifier_iid_learnability_diag/data_audit.py",
    "actmask/experiments/verifier_iid_learnability_diag/evidence.py",
    "actmask/experiments/verifier_iid_learnability_diag/training.py",
    "actmask/experiments/verifier_iid_learnability_diag/diagnostics.py",
    "actmask/experiments/verifier_iid_learnability_diag/confirmation.py",
)
PRETRAINING_EVIDENCE_PATHS = (
    "alignment_audit.json",
    "normalization_and_split_audit.json",
    "historical_baseline_evidence.json",
)


def _implementation_hashes() -> list[dict[str, Any]]:
    missing = [name for name in IMPLEMENTATION_PATHS if not (ROOT / name).is_file()]
    if missing:
        raise RuntimeError(
            "training implementation is not ready to freeze: " + ", ".join(missing)
        )
    return [
        {
            "path": name,
            "bytes": (ROOT / name).stat().st_size,
            "sha256": sha256_file(ROOT / name),
        }
        for name in IMPLEMENTATION_PATHS
    ]


def _pretraining_evidence_hashes() -> list[dict[str, Any]]:
    missing = [
        name for name in PRETRAINING_EVIDENCE_PATHS if not (OUT / name).is_file()
    ]
    if missing:
        raise RuntimeError(
            "pre-training evidence is not ready to freeze: "
            + ", ".join(missing)
        )
    return [
        {
            "path": name,
            "bytes": (OUT / name).stat().st_size,
            "sha256": sha256_file(OUT / name),
        }
        for name in PRETRAINING_EVIDENCE_PATHS
    ]


def _amended(initial: dict[str, Any]) -> dict[str, Any]:
    config = copy.deepcopy(initial)
    config["schema"] = "vild-preregistered-config-v1.1"
    config["amendment"] = {
        "parent_schema": initial["schema"],
        "parent_sha256": EXPECTED_INITIAL_HASH,
        "timing": "before any new learned-model training",
        "training_artifacts_present_at_amendment": False,
        "scientific_hypothesis_changed": False,
        "families_architectures_final_seeds_changed": False,
        "reasons": [
            "make alternate-schedule two-seed gate executable",
            "make terminal branches mutually ordered and reachable",
            "remove calibration threshold overlap at BA=0.65",
            "fully freeze sanity and representation-control schedules",
            "require current-state/history equality and exact-input collision audit",
            "move terminal root-cause decision after final confirmation",
            "freeze per-cell/per-architecture aggregation and resource fallbacks",
        ],
    }
    config["diagnosis_order"] = [
        "A_alignment_normalization_split_batch_balance",
        "B_tiny_and_full_training_memorization",
        "C_positive_and_negative_sanity_controls",
        "D_gradient_optimization_logit_threshold_diagnostics",
        "E_fair_input_vs_summary_feature_comparison",
        "final_frozen_IID_confirmation",
        "F_terminal_root_cause_decision",
    ]
    config["alignment_audit"]["required_zero_mismatch_checks"].extend(
        [
            "current_state exactly equals final history_state step",
            "per-architecture required-input hashes and conflicting-label groups audited",
        ]
    )
    config["alignment_audit"]["pass_rule"] = (
        "zero row/hash/tensor mismatches across all 12,676 rows; "
        "current_state exact equality; finite arrays; exact action-summary recomputation"
    )
    config["alignment_audit"]["input_nonidentifiability_rule"] = (
        "for every architecture-family-partition, hash the complete required input; "
        "report opposite-label identical-input groups and empirical attainable BA "
        "ceiling. A ceiling below a gate is representation non-identifiability, "
        "not capacity failure."
    )
    config["normalization_split_batch_audit"]["std_floor"] = (
        "replace any train standard deviation <1e-6 with exactly 1.0"
    )
    config["metric_and_tie_breaks"] = {
        "balanced_accuracy": "0.5*(TPR+TNR)",
        "AUROC": "rank AUROC with half credit for tied scores",
        "validation_BA_threshold": (
            "exact score sweep; maximize BA; ties choose threshold closest to 0.5, "
            "then smaller threshold"
        ),
        "validation_BCE_selection": (
            "minimum validation BCE; improvements require >1e-12; exact ties keep "
            "the earliest epoch"
        ),
        "fixed_threshold": "probability>=0.5 is positive",
        "test_oracle_threshold": "diagnostic only; never model selection or final score",
    }
    config["machine_training_protocols"] = {
        "positive_sanity": {
            "name": "positive_sanity",
            "learning_rate": 0.002,
            "weight_decay": 0.0,
            "batch_size": 128,
            "maximum_epochs": 100,
            "early_stop_patience": 15,
            "selection": "validation_bce",
            "precision": "float32",
            "gradient_clip": 2.0,
        },
        "shuffled_sanity": {
            "name": "shuffled_sanity",
            "learning_rate": 0.002,
            "weight_decay": 0.0,
            "batch_size": 128,
            "maximum_epochs": 40,
            "early_stop_patience": 10,
            "selection": "validation_bce",
            "precision": "float32",
            "gradient_clip": 2.0,
        },
        "constant_sanity": {
            "name": "constant_sanity",
            "learning_rate": 0.002,
            "weight_decay": 0.0,
            "batch_size": 128,
            "maximum_epochs": 10,
            "early_stop_patience": 10,
            "selection": "validation_bce",
            "precision": "float32",
            "gradient_clip": 2.0,
        },
        "alternate": {
            "name": "alternate",
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "batch_size": 128,
            "maximum_epochs": 200,
            "early_stop_patience": 30,
            "selection": "validation_bce",
            "precision": "float32",
            "gradient_clip": 2.0,
        },
        "final_original": {
            "name": "final_original",
            "learning_rate": 0.002,
            "weight_decay": 0.0003,
            "batch_size": 512,
            "maximum_epochs": 40,
            "early_stop_patience": 6,
            "selection": "validation_ba",
            "precision": "bfloat16",
            "gradient_clip": 2.0,
        },
        "vector_control": {
            "name": "vector_control",
            "learning_rate": 0.001,
            "weight_decay": 0.0003,
            "batch_size": 128,
            "maximum_epochs": 100,
            "early_stop_patience": 15,
            "selection": "validation_bce",
            "precision": "float32",
            "gradient_clip": 2.0,
        },
    }
    config["tiny_memorization"].update(
        {
            "selection_order": (
                "lexicographic sample_id order independently within each label; "
                "take 32 per label, then sort the selected union by unified array "
                "index before training; this is not the SHA256 ranking used by the "
                "positive sanity control"
            ),
            "acceptance_threshold": "fixed probability 0.5",
            "reported_state": (
                "the third consecutive passing evaluation state when the success "
                "stop is reached; otherwise the minimum-BCE evaluated state, with "
                "earliest exact tie"
            ),
            "success_stop": (
                "stop after three consecutive evaluation intervals meeting every "
                "acceptance threshold; otherwise run maximum_updates"
            ),
        }
    )
    config["full_training_memorization"].update(
        {
            "acceptance_threshold": "fixed probability 0.5",
            "epoch_order": "deterministic seed*100003+epoch permutation",
            "reported_state": (
                "the third consecutive passing epoch state when the success stop "
                "is reached; otherwise the minimum-BCE epoch state, with earliest "
                "exact tie"
            ),
            "success_stop": (
                "stop after three consecutive epochs meeting every acceptance "
                "threshold; otherwise run maximum_epochs"
            ),
        }
    )
    config["gradient_flow"]["batch_selection_order"] = (
        "lexicographic sample_id order independently within each label; take 32 "
        "per label, then sort the selected union by unified array index"
    )
    positive = config["sanity_controls"]["positive_observable_signal"]
    positive.update(
        {
            "optimizer": "AdamW",
            "learning_rate": 0.002,
            "weight_decay": 0.0,
            "precision": "float32",
            "gradient_clip": 2.0,
            "selection": "minimum validation BCE, earliest exact tie",
            "early_stop_patience": 15,
            "acceptance_threshold": "fixed probability 0.5",
            "epoch_order": "deterministic seed*100003+epoch permutation",
        }
    )
    shuffled = config["sanity_controls"]["shuffled_label_negative"]
    shuffled.update(
        {
            "optimizer": "AdamW",
            "learning_rate": 0.002,
            "weight_decay": 0.0,
            "batch_size": 128,
            "precision": "float32",
            "gradient_clip": 2.0,
            "selection": "minimum validation BCE, earliest exact tie",
            "early_stop_patience": 10,
            "acceptance_threshold": "fixed probability 0.5",
            "epoch_order": "deterministic seed*100003+epoch permutation",
        }
    )
    constant = config["sanity_controls"]["constant_input_negative"]
    constant.update(
        {
            "optimizer": "AdamW",
            "learning_rate": 0.002,
            "weight_decay": 0.0,
            "batch_size": 128,
            "precision": "float32",
            "gradient_clip": 2.0,
            "selection": "minimum validation BCE, earliest exact tie",
            "early_stop_patience": 10,
            "acceptance_threshold": "fixed probability 0.5",
            "epoch_order": "deterministic seed*100003+epoch permutation",
        }
    )
    config["sanity_controls"]["aggregation"] = {
        "positive_required": "all five architectures pass both metrics",
        "shuffled_required": "all five architectures remain within frozen bounds",
        "constant_required": "all five architectures pass both frozen bounds",
        "interpretation": (
            "negative-control failure invalidates pipeline interpretation and yields "
            "unrepaired implementation fault or insufficient evidence; negative "
            "controls do not independently prove the scientific root cause"
        ),
    }
    config["alternate_schedule_diagnostic"].update(
        {
            "scope": (
                "5 architectures x 4 easy families x seeds 17/29/43 = 60 runs"
            ),
            "seeds": list(SEEDS),
            "qualifying_cell": (
                "same architecture-family reaches validation-selected test BA>=0.65 "
                "in at least two seeds"
            ),
            "study_pass": (
                "at least two architectures each have at least one qualifying "
                "architecture-family cell"
            ),
        }
    )
    config["representation_mismatch"].update(
        {
            "engineered_control_input_dimension": 366,
            "flattened_raw_control_input_dimension": 544,
            "control_hidden_widths": [256, 128],
            "control_activation": "GELU",
            "control_dropout": 0.10,
            "normalization": (
                "per input feature train-only mean/std; replace std<1e-6 by 1.0"
            ),
            "optimizer": "AdamW",
            "learning_rate": 0.001,
            "weight_decay": 0.0003,
            "batch_size": 128,
            "maximum_epochs": 100,
            "early_stop_patience": 15,
            "precision": "float32",
            "gradient_clip": 2.0,
            "selection": "minimum validation BCE, earliest exact tie",
            "threshold": "validation-only exact BA optimum with frozen tie-break",
            "qualifying_cell": (
                "same control-family test BA>=0.65 in at least two seeds"
            ),
            "engineered_control_support": (
                "engineered control qualifies in at least two easy families"
            ),
            "flattened_control_role": (
                "secondary diagnostic; reported but not required for representation "
                "mismatch because engineered summaries are deterministic raw-input "
                "functions"
            ),
        }
    )
    config["calibration_failure_gate"] = {
        "applies_only_if_final_confirmation_pass_is_false": True,
        "same_architecture_family_cell_required": True,
        "required_seeds_per_cell": 2,
        "required_architectures": 2,
        "ranking_AUROC_min": 0.70,
        "validation_threshold_test_BA_strictly_less_than": 0.65,
        "oracle_test_threshold_BA_gain_min_diagnostic_only": 0.10,
        "oracle_test_threshold_use": "diagnostic only; never selection or final score",
    }
    config["root_cause_decision_order"] = [
        "VILD_IMPLEMENTATION_FAULT_REPAIRED_AND_IID_RESTORED",
        "VILD_ALIGNMENT_OR_IMPLEMENTATION_FAULT_CONFIRMED",
        "VILD_IID_LEARNABILITY_RESTORED_WITHOUT_REPAIR",
        "VILD_CALIBRATION_THRESHOLD_FAILURE",
        "VILD_OPTIMIZATION_TRAINING_FAILURE",
        "VILD_REPRESENTATION_INDUCTIVE_BIAS_MISMATCH",
        "VILD_CAPACITY_OR_ARCHITECTURE_FAILURE",
        "VILD_INSUFFICIENT_OR_HETEROGENEOUS_EVIDENCE",
    ]
    config["decision_rules"] = {
        "implementation_fault_repaired_and_restored": (
            "one causally isolated implementation/alignment defect is documented, "
            "one allowed repair is applied only in versioned new outputs, and "
            "post-repair final confirmation passes"
        ),
        "unrepaired_alignment_or_implementation_fault": (
            "final confirmation does not pass and a required alignment invariant or "
            "positive/negative pipeline control fails with a reproduced causal defect"
        ),
        "iid_restored_without_repair": (
            "no repair was applied and final confirmation passes"
        ),
        "calibration_threshold_failure": (
            "final confirmation does not pass and at least two architectures each "
            "have one same-family cell meeting every frozen calibration criterion "
            "in at least two seeds"
        ),
        "optimization_training_failure": (
            "final confirmation does not pass; alignment and sanity controls pass; "
            "the unchanged-architecture alternate schedule passes its two-seed "
            "study gate"
        ),
        "representation_inductive_bias_mismatch": (
            "final and alternate schedules do not pass; alignment, gradients, all "
            "positive sanity controls and at least four architectures' per-family "
            "tiny/full memorization pass; engineered control support passes; exact "
            "input ceiling is >=0.65"
        ),
        "capacity_or_architecture_failure": (
            "final and alternate schedules do not pass; alignment and gradients "
            "pass; at least two architectures fail tiny or full memorization in at "
            "least two families despite exact-input ceiling >=0.95"
        ),
        "insufficient_or_heterogeneous": (
            "no preceding rule is proven, or per-architecture diagnoses conflict "
            "without a single rule supported by at least two architectures"
        ),
    }
    config["per_architecture_diagnosis"] = {
        "cell_order": [
            "exact_input_nonidentifiability",
            "gradient_or_pipeline_failure",
            "calibration_failure",
            "optimization_failure",
            "representation_mismatch",
            "capacity_failure",
            "IID_learned",
            "insufficient",
        ],
        "architecture_aggregation": (
            "count four family-cell diagnoses; use a label only when it occurs in "
            "at least two families, otherwise heterogeneous_or_insufficient"
        ),
        "global_aggregation": (
            "use the frozen root_cause_decision_order and exact global rules; never "
            "force a single root cause from only one architecture"
        ),
    }
    config["OOM_fallback"] = {
        "final_original_batch_sequence": [512, 256, 128],
        "diagnostic_batch_sequence": [128, 64, 32],
        "action": (
            "on CUDA out-of-memory only, clear this study's tensors, retry the same "
            "run at the next frozen batch size, and record the fallback; never "
            "terminate another process"
        ),
        "memorization_and_gradient_microbatch_semantics": (
            "tiny memorization and gradient-flow diagnostics accumulate weighted "
            "microbatch gradients so their frozen 64-sample effective update is "
            "preserved when the 64->32 OOM fallback is used; full memorization "
            "uses the effective fallback batch size directly"
        ),
        "evaluation_batch_semantics": (
            "all validation, memorization-evaluation and final prediction passes "
            "use the run's current effective batch size, so an OOM fallback also "
            "bounds evaluation memory; this choice is frozen before training"
        ),
        "exhausted": "stop the affected run and report resource failure evidence",
    }
    config["resource_accounting"] = {
        "GPU_hours": (
            "sum append-only resource-ledger CUDA-event elapsed seconds for every "
            "attempt, including successful, OOM and failed attempts, in sanity, "
            "memorization, gradients, alternate, controls and final runs"
        ),
        "timing_fallback": (
            "if CUDA-event finalization itself fails, conservatively account wall "
            "seconds and mark the accounting method"
        ),
        "ledger_integrity": (
            "hash-chained START/END records; an unclosed START or invalid chain "
            "prevents another GPU attempt"
        ),
        "record_ledger_coverage": (
            "every current complete learned run record must self-hash and link to "
            "a matching SUCCESS START/END attempt; older successful attempts that "
            "were superseded remain informational and continue to count toward "
            "GPU hours"
        ),
        "new_artifact_bytes": (
            "recursive byte sum of every file under the new OUT, including source "
            "snapshot and reports"
        ),
        "hard_stop": (
            "after every run, do not start another if cumulative GPU time >=5 hours "
            "or recursive output size >=10 GiB"
        ),
        "unrelated_process_policy": "never terminate or interfere",
    }
    config["implementation_freeze"] = _implementation_hashes()
    config["pretraining_evidence_freeze"] = _pretraining_evidence_hashes()
    return config


def run() -> dict[str, Any]:
    current_before_amendment = read_json(CONFIG)
    source_check = verify_bound_source_snapshot(current_before_amendment)
    if not source_check["pass"]:
        raise RuntimeError(f"source study changed: {source_check}")
    learned_artifacts = [
        str(path.relative_to(OUT))
        for path in OUT.rglob("*")
        if path.is_file()
        and (
            path.suffix in {".pt", ".pth"}
            or "checkpoint" in path.name.lower()
            or bool(
                {
                    "checkpoints",
                    "raw_predictions",
                    "run_records",
                    "runtime",
                    "training_curves",
                    "training_runs",
                }
                & set(path.parts)
            )
            or path.name == "resource_ledger.jsonl"
        )
    ]
    if learned_artifacts:
        raise RuntimeError(
            "cannot amend after learned artifacts exist: "
            + ", ".join(learned_artifacts[:20])
        )
    events = read_jsonl(OUT / "run_log.jsonl")
    forbidden_events = [
        row["event"]
        for row in events
        if any(
            token in row["event"]
            for token in ("TRAIN", "MEMORIZATION", "SANITY", "FINAL_CONFIRMATION")
        )
    ]
    if forbidden_events:
        raise RuntimeError(
            f"cannot amend after training events: {forbidden_events}"
        )
    if CONFIG_HASH.read_text(encoding="utf-8").strip() != sha256_file(CONFIG):
        raise RuntimeError("current preregistration hash mismatch")
    current = read_json(CONFIG)
    if current.get("schema") == "vild-preregistered-config-v1.1":
        if not INITIAL_CONFIG.is_file() or not INITIAL_HASH.is_file():
            raise RuntimeError("amended config lacks preserved initial config")
        return {
            "already_amended": True,
            "initial_sha256": INITIAL_HASH.read_text(encoding="utf-8").strip(),
            "amended_sha256": sha256_file(CONFIG),
            "implementation_files": len(current["implementation_freeze"]),
        }
    current_hash = sha256_file(CONFIG)
    if current_hash != EXPECTED_INITIAL_HASH:
        raise RuntimeError(
            f"unexpected initial preregistration hash {current_hash}"
        )
    write_json(INITIAL_CONFIG, current)
    INITIAL_HASH.write_text(current_hash + "\n", encoding="utf-8")
    amended = _amended(current)
    write_json(CONFIG, amended)
    amended_hash = sha256_file(CONFIG)
    CONFIG_HASH.write_text(amended_hash + "\n", encoding="utf-8")
    amendment = {
        "schema": "vild-preregistration-amendment-v1",
        "study_id": STUDY_ID,
        "timing": "before any new training",
        "training_artifacts_present": False,
        "parent_config_path": str(INITIAL_CONFIG.resolve()),
        "parent_config_sha256": current_hash,
        "amended_config_path": str(CONFIG.resolve()),
        "amended_config_sha256": amended_hash,
        "independent_review_required_amendment": True,
        "scientific_objective_changed": False,
        "source_study_unchanged": source_check["pass"],
        "implementation_freeze": amended["implementation_freeze"],
        "pretraining_evidence_freeze": amended[
            "pretraining_evidence_freeze"
        ],
        "reasons": amended["amendment"]["reasons"],
    }
    write_json(AMENDMENT, amendment)
    append_log(
        "PREREGISTRATION_AMENDED_BEFORE_TRAINING",
        parent_config_sha256=current_hash,
        amended_config_sha256=amended_hash,
        amendment_sha256=sha256_file(AMENDMENT),
        implementation_files=len(amended["implementation_freeze"]),
        pretraining_evidence=amended["pretraining_evidence_freeze"],
        learned_artifacts_present=False,
        new_training_started=False,
        source_snapshot_sha256=source_check["snapshot_sha256"],
    )
    return {
        "already_amended": False,
        "initial_sha256": current_hash,
        "amended_sha256": amended_hash,
        "implementation_files": len(amended["implementation_freeze"]),
    }


def main() -> None:
    print(json.dumps(run(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
