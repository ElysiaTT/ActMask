"""Requirement-by-requirement completion audit for VERIFIER-SHORTCUT-AUDIT-V1."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .common import OUT, canonical_json, read_json, read_jsonl, sha256_file, write_json


PAPER_FILES = (
    "title_candidates.md",
    "abstract_zh.md",
    "abstract_en.md",
    "introduction_outline.md",
    "problem_formulation.md",
    "audit_taxonomy.md",
    "audit_protocol.md",
    "negative_generator_analysis.md",
    "shortcut_metrics.md",
    "verifier_zoo.md",
    "mitigation_method.md",
    "experiments.md",
    "results_tables.md",
    "figure_plan.md",
    "claim_boundary.md",
    "limitations.md",
    "related_work_plan.md",
)
REQUIRED_ROOT = (
    "preregistered_config.json",
    "preregistered_config.sha256",
    "historical_branch_inventory.json",
    "imported_artifact_manifest.json",
    "imported_hash_verification.json",
    "branch_role_table.md",
    "family_role_manifest.json",
    "family_claim_boundary.json",
    "unified_audit_manifest.jsonl",
    "unified_audit_arrays.npz",
    "unified_audit_statistics.json",
    "unified_audit_hash_manifest.json",
    "nuisance_feature_schema.json",
    "nuisance_features.npz",
    "nuisance_feature_statistics.json",
    "nuisance_extraction_audit.json",
    "audit_environment_manifest.json",
    "environment_statistics.json",
    "matching_quality_report.json",
    "shortcut_control_config.json",
    "shortcut_control_summary.json",
    "shortcut_control_per_environment.json",
    "verifier_config.json",
    "verifier_summary.json",
    "verifier_per_seed.json",
    "verifier_per_environment.json",
    "shortcut_reliance_summary.json",
    "generator_transfer_matrix.csv",
    "generator_transfer_matrix.json",
    "matching_gap_report.json",
    "nuisance_probe_report.json",
    "calibration_shift_report.json",
    "relational_gain_report.json",
    "representation_probe_config.json",
    "representation_probe_summary.json",
    "phase_v10_decision.json",
    "independent_metric_recomputation.json",
    "failure_taxonomy.json",
    "failure_taxonomy.md",
    "per_family_analysis.md",
    "per_task_analysis.md",
    "per_shortcut_analysis.md",
    "false_accept_analysis.md",
    "final_decision.json",
    "final_report.md",
    "executive_summary_zh.md",
    "executive_summary_en.md",
    "claim_boundary.md",
    "next_steps.md",
    "reproducibility_manifest.json",
    "run_log.jsonl",
)


def _check(
    checks: list[dict[str, Any]],
    name: str,
    passed: bool,
    evidence: Any,
) -> None:
    checks.append({"requirement": name, "pass": bool(passed), "evidence": evidence})


def _decision(value: dict[str, Any]) -> str | None:
    return (
        value.get("decision")
        or value.get("final_decision")
        or value.get("terminal_decision")
    )


def _run_log_audit() -> dict[str, Any]:
    rows = read_jsonl(OUT / "run_log.jsonl")
    failures = []
    previous = None
    import hashlib

    for index, row in enumerate(rows):
        expected_record = dict(row)
        actual_hash = expected_record.pop("record_sha256", None)
        expected_hash = hashlib.sha256(
            canonical_json(expected_record).encode("utf-8")
        ).hexdigest()
        if row.get("sequence") != index:
            failures.append({"row": index, "reason": "sequence"})
        if row.get("previous_record_sha256") != previous:
            failures.append({"row": index, "reason": "previous_hash"})
        if actual_hash != expected_hash:
            failures.append({"row": index, "reason": "record_hash"})
        previous = actual_hash
    return {
        "records": len(rows),
        "events": [row["event"] for row in rows],
        "final_record_sha256": previous,
        "failures": failures,
        "pass": not failures,
    }


def audit(write: bool = False) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    missing = [name for name in REQUIRED_ROOT if not (OUT / name).is_file()]
    empty = [
        name
        for name in REQUIRED_ROOT
        if (OUT / name).is_file() and (OUT / name).stat().st_size == 0
    ]
    _check(
        checks,
        "all named V0-V13 root artifacts exist and are non-empty",
        not missing and not empty,
        {"missing": missing, "empty": empty, "required": len(REQUIRED_ROOT)},
    )
    missing_paper = [
        name for name in PAPER_FILES if not (OUT / "paper" / name).is_file()
    ]
    _check(
        checks,
        "all 17 V12 paper documents exist",
        not missing_paper,
        {"missing": missing_paper, "required": len(PAPER_FILES)},
    )
    preregistration_hash = (OUT / "preregistered_config.sha256").read_text().strip()
    _check(
        checks,
        "V0 preregistration hash remains frozen",
        preregistration_hash == sha256_file(OUT / "preregistered_config.json"),
        preregistration_hash,
    )
    preregistration = read_json(OUT / "preregistered_config.json")
    import_failures = [
        record["absolute_path"]
        for record in preregistration["historical_imports"]["artifacts"]
        if not Path(record["absolute_path"]).is_file()
        or sha256_file(record["absolute_path"]) != record["sha256"]
    ]
    imported = read_json(OUT / "imported_hash_verification.json")
    _check(
        checks,
        "V1 historical imports remain complete and byte-identical",
        not import_failures and imported["pass"],
        {
            "frozen_artifacts": len(
                preregistration["historical_imports"]["artifacts"]
            ),
            "rehash_failures": import_failures,
            "source_rehash": imported["source_content_rehash"],
        },
    )
    roles = read_json(OUT / "family_role_manifest.json")
    role_values = (
        roles.get("families")
        or roles.get("roles")
        or roles.get("family_roles")
    )
    _check(
        checks,
        "V2 contains all eight frozen family roles",
        role_values is not None and len(role_values) == 8,
        role_values,
    )
    unified = read_json(OUT / "unified_audit_statistics.json")
    with np.load(OUT / "unified_audit_arrays.npz") as arrays:
        array_evidence = {
            name: list(arrays[name].shape)
            for name in (
                "history_state",
                "candidate_action",
                "visual_history",
                "language_embedding",
                "label",
            )
        }
        arrays_finite = all(
            np.isfinite(arrays[name]).all()
            for name in (
                "history_state",
                "candidate_action",
                "visual_history",
                "language_embedding",
            )
        )
    _check(
        checks,
        "V3 unified immutable table has 12,676 finite N1-N8 samples",
        (
            unified["samples"] == 12676
            and len(unified["families"]) == 8
            and not unified["action_values_modified"]
            and not unified["historical_labels_modified"]
            and arrays_finite
        ),
        {"statistics": unified, "array_shapes": array_evidence},
    )
    nuisance = read_json(OUT / "nuisance_extraction_audit.json")
    _check(
        checks,
        "V4 nuisance extraction is finite, causal, and audit-only",
        (
            nuisance["pass"]
            and nuisance["all_finite"]
            and not nuisance["future_state_used"]
            and not nuisance["future_rgb_used"]
            and not nuisance["SAM3_used"]
            and not nuisance["new_robot_masks_generated"]
        ),
        nuisance,
    )
    environment = read_json(OUT / "audit_environment_manifest.json")
    matching = read_json(OUT / "matching_quality_report.json")
    matching_files = len(list((OUT / "matching_manifests").glob("*.jsonl")))
    _check(
        checks,
        "V5 freezes E1-E12, 19 train specs, and passing matches",
        (
            len(environment["environments"]) == 12
            and len(environment["training_specs"]) == 19
            and matching_files == 12
            and matching["all_pass"]
            and environment["frozen_before_model_training"]
        ),
        {
            "environments": len(environment["environments"]),
            "train_specs": len(environment["training_specs"]),
            "matching_manifests": matching_files,
            "matching": matching["coverage"],
        },
    )
    controls = read_json(OUT / "shortcut_control_summary.json")
    control_environment = read_json(OUT / "shortcut_control_per_environment.json")
    control_raw = len(
        list((OUT / "shortcut_control_raw_predictions").rglob("*.npz"))
    )
    control_checkpoints = len(
        list((OUT / "shortcut_control_checkpoints").rglob("*.pt"))
    )
    control_curves = len(
        list((OUT / "shortcut_control_training_curves").rglob("*.json"))
    )
    control_runtimes = len(
        list((OUT / "shortcut_control_runtime").rglob("*.json"))
    )
    control_representations = len(
        list((OUT / "shortcut_control_representations").rglob("*.npz"))
    )
    _check(
        checks,
        "V6 has exactly six controls, 114 runs, and E1-E12 raw scores",
        (
            controls["complete"]
            and controls["controls"] == 6
            and controls["training_runs"] == 114
            and control_raw == 318
            and control_checkpoints == 114
            and control_curves == 114
            and control_runtimes == 114
            and control_representations == 114
            and control_environment["raw_scores_saved_for_all_environments"]
        ),
        {
            "training_runs": controls["training_runs"],
            "raw_files": control_raw,
            "checkpoints": control_checkpoints,
            "curves": control_curves,
            "runtimes": control_runtimes,
            "representations": control_representations,
            "GPU_hours": controls["total_GPU_hours"],
        },
    )
    verifiers = read_json(OUT / "verifier_summary.json")
    verifier_environment = read_json(OUT / "verifier_per_environment.json")
    verifier_raw = len(list((OUT / "raw_verifier_predictions").rglob("*.npz")))
    verifier_checkpoints = len(
        list((OUT / "verifier_checkpoints").rglob("*.pt"))
    )
    verifier_curves = len(
        list((OUT / "verifier_training_curves").rglob("*.json"))
    )
    verifier_runtimes = len(
        list((OUT / "verifier_runtime").rglob("*.json"))
    )
    verifier_representations = len(
        list((OUT / "verifier_representations").rglob("*.npz"))
    )
    _check(
        checks,
        "V7 has exactly five fair architectures, 285 runs, and E1-E12 raw scores",
        (
            verifiers["complete"]
            and verifiers["learned_architectures"] == 5
            and verifiers["training_runs"] == 285
            and verifier_raw == 795
            and verifier_checkpoints == 285
            and verifier_curves == 285
            and verifier_runtimes == 285
            and verifier_representations == 795
            and verifier_environment["raw_scores_saved_for_all_environments"]
        ),
        {
            "training_runs": verifiers["training_runs"],
            "raw_files": verifier_raw,
            "checkpoints": verifier_checkpoints,
            "curves": verifier_curves,
            "runtimes": verifier_runtimes,
            "representations": verifier_representations,
            "GPU_hours": verifiers["total_GPU_hours"],
        },
    )
    transfer = read_json(OUT / "generator_transfer_matrix.json")
    nuisance_probe = read_json(OUT / "nuisance_probe_report.json")
    _check(
        checks,
        "V8 covers 720 family transfers and 75 held-out nuisance probes",
        (
            transfer["complete"]
            and transfer["per_seed_evaluations"] == 720
            and nuisance_probe["coverage"]["complete"]
            and nuisance_probe["coverage"]["observed_probes"] == 75
        ),
        {
            "family_transfer": transfer["per_seed_evaluations"],
            "nuisance_probes": nuisance_probe["coverage"]["observed_probes"],
        },
    )
    representation = read_json(OUT / "representation_probe_summary.json")
    _check(
        checks,
        "V9 freezes backbones and verifies 105 held-out representation probes",
        (
            representation["complete"]
            and representation["probe_runs"] == 105
            and representation["backbone_frozen"]
            and representation["independent_raw_recompute_passed"]
            and not representation["test_used_for_training_or_selection"]
        ),
        {
            "probe_runs": representation["probe_runs"],
            "family_gate": representation["family_representation_probe_gate"],
        },
    )
    phase = read_json(OUT / "phase_v10_decision.json")
    recomputation = read_json(OUT / "independent_metric_recomputation.json")
    _check(
        checks,
        "V10 independently recomputes all raw metrics and reaches frozen gate",
        (
            phase["decision"] == "VSA_NO_LEARNABLE_IID_TASK"
            and not phase["mitigation_authorized"]
            and recomputation["pass"]
            and recomputation["total_metric_rows_recomputed"] == 1833
        ),
        {
            "decision": phase["decision"],
            "easy_IID_qualifiers": phase["easy_IID_learnability_gate"][
                "qualifying_architectures"
            ],
            "indicator_qualifiers": phase["shortcut_indicator_gate"][
                "qualifying_architectures"
            ],
            "tasks": phase["task_span_gate"]["qualifying_task_count"],
            "recomputed_rows": recomputation["total_metric_rows_recomputed"],
        },
    )
    mitigation_names = [
        path.name
        for path in OUT.iterdir()
        if "mitigation" in path.name.lower() and path.name != "paper"
    ]
    _check(
        checks,
        "conditional mitigation correctly stopped and produced no mitigation artifacts",
        not mitigation_names,
        mitigation_names,
    )
    if (OUT / "final_decision.json").is_file():
        final = read_json(OUT / "final_decision.json")
        _check(
            checks,
            "V13 final decision exactly matches V10",
            _decision(final) == phase["decision"],
            {"V10": phase["decision"], "V13": _decision(final)},
        )
    output_bytes = sum(
        path.stat().st_size for path in OUT.rglob("*") if path.is_file()
    )
    _check(
        checks,
        "new-artifact resource limit remains below 30 GiB",
        output_bytes <= 30 * 1024**3,
        {"bytes": output_bytes, "GiB": output_bytes / 1024**3},
    )
    total_gpu_hours = controls["total_GPU_hours"] + verifiers["total_GPU_hours"]
    _check(
        checks,
        "GPU resource limit remains below 20 hours",
        total_gpu_hours <= 20.0,
        total_gpu_hours,
    )
    log_audit = _run_log_audit()
    _check(
        checks,
        "append-only run-log hash chain is valid",
        log_audit["pass"],
        log_audit,
    )
    result = {
        "schema": "vsa-completion-audit-v1",
        "study_id": "VERIFIER-SHORTCUT-AUDIT-V1",
        "checks": checks,
        "checks_passed": sum(check["pass"] for check in checks),
        "checks_total": len(checks),
        "pass": all(check["pass"] for check in checks),
        "decision": phase["decision"],
        "mitigation_authorized": phase["mitigation_authorized"],
        "claim_boundary": "construction-label benchmark audit; not physical safety",
    }
    if write:
        write_json(OUT / "completion_audit.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    arguments = parser.parse_args()
    result = audit(write=arguments.write)
    print(
        json.dumps(
            {
                "pass": result["pass"],
                "checks_passed": result["checks_passed"],
                "checks_total": result["checks_total"],
                "failed": [
                    check["requirement"]
                    for check in result["checks"]
                    if not check["pass"]
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
