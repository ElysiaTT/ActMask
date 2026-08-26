"""Independent, read-only verifier for ACTIONCHECK-PROPOSAL-DATA-V1 outputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .common import OUT, ROOT, action_hash, read_jsonl, sha256_file, write_json
from .evaluate_actioncheck import classification_metrics, evaluate_file
from .pilot_baselines import BASELINES
from .validate_actioncheck_schema import validate_group
from .validate_evidence import validate_evidence
from .validate_group_structure import validate_groups
from .validate_splits import validate_splits


ALWAYS_REQUIRED = (
    "preregistered_config.json",
    "preregistered_config.sha256",
    "run_log.jsonl",
    "final_report.md",
    "executive_summary_zh.md",
    "executive_summary_en.md",
    "local_data_inventory.json",
    "proposal_source_inventory.json",
    "execution_label_inventory.json",
    "sim_label_inventory.json",
    "expert_label_inventory.json",
    "data_availability_report.md",
    "schema/actioncheck_schema_v1.json",
    "schema/label_taxonomy.json",
    "schema/reason_code_taxonomy.json",
    "schema/example_context_group.json",
    "schema/README.md",
    "validators_report.md",
    "validator_unit_tests.md",
    "protocols/route1_real_robot_collection.md",
    "protocols/real_robot_logging_schema.json",
    "protocols/operator_review_form.md",
    "protocols/safety_gate_checklist.md",
    "protocols/route2_offline_policy_proposals.md",
    "protocols/offline_policy_proposal_schema.json",
    "protocols/expert_review_guidelines.md",
    "protocols/route3_simulator_branching.md",
    "protocols/sim_rollout_schema.json",
    "protocols/sim_label_rules.md",
    "benchmark_builder_report.md",
    "split_design_report.md",
    "split_audit_schema.json",
    "shortcut_audit_design.md",
    "shortcut_gate_config.json",
    "evaluation_metric_spec.md",
    "metric_unit_tests.md",
    "future_method_authorization.md",
    "phaseaware_training_conditions.json",
    "paper/title_candidates.md",
    "paper/abstract_zh.md",
    "paper/abstract_en.md",
    "paper/introduction_outline.md",
    "paper/problem_formulation.md",
    "paper/dataset_protocol.md",
    "paper/benchmark_protocol.md",
    "paper/evaluation_protocol.md",
    "paper/shortcut_audit_protocol.md",
    "paper/model_placeholder.md",
    "paper/claim_boundary.md",
    "paper/related_work_plan.md",
    "paper/experiment_plan.md",
    "paper/figure_plan.md",
    "paper/table_plan.md",
    "final_decision.json",
    "claim_boundary.md",
    "next_steps.md",
    "codex_next_goal_if_data_available.md",
    "codex_next_goal_for_real_robot_collection.md",
    "reproducibility_manifest.json",
)
PILOT_REQUIRED = (
    "pilot/source_actioncheck_context_groups.jsonl",
    "pilot/original_source_hash_manifest.json",
    "pilot/pilot_admission.json",
    "pilot/prebuilt_split_manifest.json",
    "pilot/benchmark/processed_contexts.jsonl",
    "pilot/benchmark/candidate_manifest.jsonl",
    "pilot/benchmark/grouped_candidate_manifest.jsonl",
    "pilot/benchmark/split_manifest.json",
    "pilot/benchmark/benchmark_arrays.npz",
    "pilot/benchmark/source_hash_manifest.json",
    "pilot/benchmark/benchmark_build_report.md",
    "pilot/benchmark/benchmark_build_report.json",
    "pilot/shortcut/shortcut_audit.json",
    "pilot/shortcut/shortcut_predictions.jsonl",
    "pilot/baselines/pilot_baseline_results.json",
    "pilot/baselines/raw_baseline_predictions.jsonl",
    "pilot/evaluation/evaluation.json",
    "pilot_benchmark_report.md",
    "pilot_shortcut_report.md",
    "pilot_baseline_report.md",
    "pilot_evaluation_report.md",
    "raw_prediction_manifest.json",
    "pilot_decision.json",
)


class Audit:
    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []

    def check(self, name: str, condition: bool, detail: Any = None) -> None:
        self.checks.append(
            {
                "name": name,
                "pass": bool(condition),
                "detail": detail,
            }
        )

    @property
    def passed(self) -> bool:
        return all(row["pass"] for row in self.checks)


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _equal_metrics(left: Any, right: Any, tolerance: float = 1e-12) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(
            _equal_metrics(left[key], right[key], tolerance) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _equal_metrics(a, b, tolerance) for a, b in zip(left, right)
        )
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        if np.isnan(left) and np.isnan(right):
            return True
        return bool(np.isclose(left, right, rtol=0.0, atol=tolerance))
    return left == right


def _required_files(root: Path, audit: Audit) -> None:
    missing = [
        relative
        for relative in ALWAYS_REQUIRED + PILOT_REQUIRED
        if not (root / relative).is_file()
    ]
    audit.check("required_files_complete", not missing, {"missing": missing})


def _preregistration(root: Path, audit: Audit) -> list[dict[str, Any]]:
    actual = sha256_file(root / "preregistered_config.json")
    recorded = (root / "preregistered_config.sha256").read_text(
        encoding="utf-8"
    ).strip()
    audit.check(
        "preregistered_config_hash",
        actual == recorded,
        {"actual": actual, "recorded": recorded},
    )
    rows = read_jsonl(root / "run_log.jsonl")
    chain_ok = bool(rows)
    for index, row in enumerate(rows):
        material = dict(row)
        observed = material.pop("record_sha256", None)
        expected = hashlib.sha256(
            json.dumps(
                material,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        chain_ok &= (
            row.get("sequence") == index
            and observed == expected
            and row.get("previous_record_sha256")
            == (None if index == 0 else rows[index - 1]["record_sha256"])
        )
    chain_ok &= (
        rows[0].get("event") == "P0_PREREGISTRATION_FROZEN"
        and rows[0].get("config_sha256") == recorded
    )
    first_training = next(
        (
            index
            for index, row in enumerate(rows)
            if row["event"].startswith("P9_")
        ),
        len(rows),
    )
    chain_ok &= first_training > 0
    audit.check(
        "run_log_hash_chain_and_prereg_before_training",
        chain_ok,
        {"records": len(rows), "first_P9_sequence": first_training},
    )
    return rows


def _dataset(root: Path, audit: Audit) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups = read_jsonl(root / "pilot/source_actioncheck_context_groups.jsonl")
    schema_failures = [
        {"context_id": row.get("context_id"), "errors": errors}
        for row in groups
        if (errors := validate_group(row))
    ]
    audit.check(
        "all_context_groups_schema_valid",
        not schema_failures,
        {"groups": len(groups), "failure_sample": schema_failures[:3]},
    )
    evidence = validate_evidence(groups)
    audit.check("all_primary_labels_evidence_valid", evidence["pass"], evidence)
    structure = validate_groups(groups, minimum_candidates=4)
    audit.check(
        "pilot_mixed_group_gate_recomputed",
        structure["pilot_80_percent_gate"],
        {
            "fully_supported": structure["fully_supported_contexts"],
            "fraction": structure["fully_supported_fraction"],
            "failures": len(structure["failures"]),
        },
    )
    labels = Counter(
        candidate["label"]
        for group in groups
        for candidate in group["candidates"]
    )
    sources = {
        candidate["source_policy"]
        for group in groups
        for candidate in group["candidates"]
    }
    evidences = {
        candidate["evidence_type"]
        for group in groups
        for candidate in group["candidates"]
    }
    tasks = {group["task_id"] for group in groups}
    candidates = sum(len(group["candidates"]) for group in groups)
    admission = _json(root / "pilot/pilot_admission.json")
    recomputed_values = {
        "context_groups": len(groups),
        "average_candidates_per_context": candidates / len(groups),
        "mixed_label_contexts": structure["fully_supported_contexts"],
        "mixed_label_context_fraction": structure["fully_supported_fraction"],
        "tasks": len(tasks),
        "candidate_sources": len(sources),
        "evidence_types": len(evidences),
        "labeled_candidates": candidates,
    }
    audit.check(
        "pilot_admission_values_recomputed",
        _equal_metrics(admission["values"], recomputed_values),
        {"recorded": admission["values"], "recomputed": recomputed_values},
    )
    audit.check(
        "expected_nonfabricated_dataset_counts",
        len(groups) == 1024
        and candidates == 10240
        and labels == Counter({"reject": 5153, "accept": 5087})
        and len(tasks) == 5
        and len(sources) == 2
        and evidences
        == {"sim_execution_success", "sim_execution_failure"},
        {
            "contexts": len(groups),
            "candidates": candidates,
            "labels": dict(labels),
            "tasks": len(tasks),
            "sources": len(sources),
            "evidence_types": sorted(evidences),
        },
    )
    action_failures = []
    for group in groups:
        for candidate in group["candidates"]:
            if action_hash(candidate["action_chunk"]) != candidate["hash"]:
                action_failures.append(candidate["candidate_id"])
    audit.check(
        "all_source_action_hashes_recomputed",
        not action_failures,
        {"checked": candidates, "failures": action_failures[:10]},
    )
    manifest = read_jsonl(root / "pilot/benchmark/candidate_manifest.jsonl")
    return groups, manifest


def _source_hashes(root: Path, audit: Audit) -> None:
    for relative, name in (
        ("pilot/original_source_hash_manifest.json", "original_source_hashes"),
        ("pilot/benchmark/source_hash_manifest.json", "builder_source_hashes"),
    ):
        data = _json(root / relative)
        key = "read_only_sources" if "read_only_sources" in data else "files"
        failures = []
        for row in data[key]:
            path = Path(row["path"])
            if (
                not path.is_file()
                or path.stat().st_size != row["bytes"]
                or sha256_file(path) != row["sha256"]
            ):
                failures.append(row["path"])
        audit.check(
            name,
            not failures and bool(data[key]),
            {"files": len(data[key]), "failures": failures},
        )


def _builder(
    root: Path,
    groups: list[dict[str, Any]],
    manifest: list[dict[str, Any]],
    audit: Audit,
) -> None:
    benchmark = root / "pilot/benchmark"
    processed = read_jsonl(benchmark / "processed_contexts.jsonl")
    audit.check(
        "builder_processed_contexts_identical_to_validated_input",
        processed == groups,
        {"groups": len(processed)},
    )
    split = _json(benchmark / "split_manifest.json")
    split_result = validate_splits(groups, split)
    audit.check("all_split_audits_recomputed", split_result["pass"], split_result)
    with np.load(benchmark / "benchmark_arrays.npz") as stored:
        arrays = {name: stored[name].copy() for name in stored.files}
    primary = [row for row in manifest if row["primary_eligible"]]
    primary.sort(key=lambda row: int(row["primary_array_index"]))
    shape_ok = (
        len(primary) == len(arrays["labels"]) == 10240
        and arrays["candidate_actions"].shape[0] == len(primary)
        and arrays["group_index"].shape[0] == len(primary)
    )
    failures = []
    for row, label, action, mask in zip(
        primary,
        arrays["labels"],
        arrays["candidate_actions"],
        arrays["candidate_action_mask"],
    ):
        horizon = int(row["action_horizon"])
        dimension = int(row["action_dim"])
        selected = action[:horizon, :dimension]
        selected_mask = mask[:horizon, :dimension]
        expected_label = 1 if row["label"] == "accept" else 0
        if (
            not selected_mask.all()
            or int(label) != expected_label
            or action_hash(selected.tolist()) != row["action_hash"]
        ):
            failures.append(row["candidate_id"])
    audit.check(
        "builder_arrays_manifest_and_action_hashes_recomputed",
        shape_ok and not failures,
        {
            "primary_candidates": len(primary),
            "array_shape": list(arrays["candidate_actions"].shape),
            "failures": failures[:10],
        },
    )
    grouped = read_jsonl(benchmark / "grouped_candidate_manifest.jsonl")
    atomic_ok = len(grouped) == len(groups)
    manifest_by_id = {row["candidate_id"]: row for row in manifest}
    for row in grouped:
        owners = {
            manifest_by_id[candidate_id]["split_assignments"]["episode_held_out"]
            for candidate_id in row["candidate_ids"]
        }
        atomic_ok &= len(owners) == 1 and row["primary_split"] in owners
    audit.check(
        "all_candidates_remain_context_atomic",
        atomic_ok,
        {"grouped_manifests": len(grouped)},
    )
    build = _json(benchmark / "benchmark_build_report.json")
    audit.check(
        "builder_forbidden_negative_construction_absent",
        build["reject_labels_synthesized"] is False
        and build["logged_action_swaps_used"] is False
        and build["candidate_level_random_split"] is False
        and build["uncertain_excluded_from_primary"] is True,
        {
            key: build[key]
            for key in (
                "reject_labels_synthesized",
                "logged_action_swaps_used",
                "candidate_level_random_split",
                "uncertain_excluded_from_primary",
            )
        },
    )


def _predictions_and_decision(
    root: Path,
    manifest: list[dict[str, Any]],
    audit: Audit,
) -> None:
    raw_manifest = _json(root / "raw_prediction_manifest.json")
    raw_failures = []
    for row in raw_manifest["files"]:
        path = root / row["path"]
        count = len(read_jsonl(path)) if path.is_file() else -1
        if (
            not path.is_file()
            or sha256_file(path) != row["sha256"]
            or path.stat().st_size != row["bytes"]
            or count != row["rows"]
        ):
            raw_failures.append(row["path"])
    audit.check(
        "raw_prediction_hashes_and_rows",
        not raw_failures,
        {"files": len(raw_manifest["files"]), "failures": raw_failures},
    )
    baseline_path = root / "pilot/baselines/raw_baseline_predictions.jsonl"
    recomputed = evaluate_file(baseline_path)
    recorded = _json(root / "pilot/evaluation/evaluation.json")
    audit.check(
        "baseline_metrics_recomputed_from_raw_predictions",
        _equal_metrics(recomputed, recorded),
        {"models": len(recomputed["models"])},
    )
    baseline_report = _json(root / "pilot/baselines/pilot_baseline_results.json")
    audit.check(
        "exact_six_allowed_baselines_and_two_heldouts",
        set(baseline_report["models"]) == set(BASELINES)
        and set(baseline_report["schemes"])
        == {"episode_held_out", "task_held_out"}
        and len(baseline_report["evaluation"]) == 12
        and all(
            training["device"] == "cuda"
            for training in baseline_report["training"].values()
        ),
        {
            "models": baseline_report["models"],
            "schemes": baseline_report["schemes"],
            "jobs": len(baseline_report["evaluation"]),
        },
    )
    shortcut = _json(root / "pilot/shortcut/shortcut_audit.json")
    shortcut_rows = read_jsonl(
        root / "pilot/shortcut/shortcut_predictions.jsonl"
    )
    by_control: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in shortcut_rows:
        by_control[row["control"]].append(row)
    shortcut_metric_ok = True
    for control, values in by_control.items():
        threshold = shortcut["controls"][control]["training"][
            "threshold_selected_on_validation"
        ]
        test = [row for row in values if row["split"] == "test"]
        metric = classification_metrics(
            [row["label"] for row in test],
            [row["score"] for row in test],
            threshold,
        )
        shortcut_metric_ok &= _equal_metrics(
            metric, shortcut["controls"][control]["test"]
        )
    audit.check(
        "all_13_shortcut_test_metrics_recomputed",
        shortcut_metric_ok and len(by_control) == 13,
        {"controls": sorted(by_control)},
    )
    rejects = [row for row in manifest if row["label"] == "reject"]
    reasons = Counter(
        reason
        for row in rejects
        for reason in row["reason_codes_audit_only"]
    )
    max_reason = max(reasons.values(), default=0) / max(1, len(rejects))
    expected_gates = dict(shortcut["gates"])
    expected_gates["single_reason_code_fraction_le_0_40"] = max_reason <= 0.40
    audit.check(
        "shortcut_gate_and_reason_failure_recomputed",
        shortcut["pass"] == all(expected_gates.values())
        and shortcut["pass"] is False
        and max_reason == 1.0
        and expected_gates["single_reason_code_fraction_le_0_40"] is False
        and all(
            value
            for name, value in expected_gates.items()
            if name != "single_reason_code_fraction_le_0_40"
        ),
        {
            "maximum_reason_fraction": max_reason,
            "reason_distribution": dict(reasons),
            "gates": expected_gates,
        },
    )
    pilot_decision = _json(root / "pilot_decision.json")
    final_decision = _json(root / "final_decision.json")
    expected_decision = "ACTIONCHECK_PROPOSAL_SHORTCUT_DOMINATED"
    audit.check(
        "decision_precedence_recomputed",
        pilot_decision["decision"] == expected_decision
        and final_decision["decision"] == expected_decision
        and pilot_decision["shortcut_pass"] is False,
        {
            "pilot": pilot_decision["decision"],
            "final": final_decision["decision"],
        },
    )
    strata_ok = all(
        set(result["generalization_strata"])
        == {
            "per_task",
            "per_policy_source",
            "per_reason_code_audit_only",
            "per_evidence_type_audit_only",
        }
        and result["reason_code_and_evidence_type_used_as_model_inputs"] is False
        for result in recorded["models"].values()
    )
    audit.check(
        "required_generalization_audit_strata_present",
        strata_ok,
        {"models": len(recorded["models"])},
    )


def _method_and_preservation(
    root: Path, run_rows: list[dict[str, Any]], audit: Audit
) -> None:
    conditions = _json(root / "phaseaware_training_conditions.json")
    pilot_files = [
        path
        for path in (root / "pilot").rglob("*")
        if path.is_file()
    ]
    checkpoint_suffixes = {".pt", ".pth", ".ckpt", ".safetensors"}
    phaseaware_checkpoints = [
        str(path.relative_to(root))
        for path in pilot_files
        if path.suffix.lower() in checkpoint_suffixes
        or "phaseaware" in path.name.lower()
    ]
    baseline = _json(root / "pilot/baselines/pilot_baseline_results.json")
    audit.check(
        "phaseaware_not_authorized_or_trained",
        conditions["authorized_now"] is False
        and baseline["phaseaware_temporal_energy_verifier_trained"] is False
        and not phaseaware_checkpoints,
        {
            "authorized_now": conditions["authorized_now"],
            "checkpoint_like_files": phaseaware_checkpoints,
        },
    )
    start = datetime.fromisoformat(run_rows[0]["timestamp_utc"]).timestamp()
    protected_root = ROOT / "outputs" / "actmask"
    touched = []
    if protected_root.is_dir():
        for path in protected_root.iterdir():
            if path.resolve() == root.resolve() or not path.is_dir():
                continue
            for file in path.rglob("*"):
                if file.is_file() and file.stat().st_mtime > start + 1.0:
                    touched.append(str(file.relative_to(ROOT)))
                    if len(touched) >= 100:
                        break
            if len(touched) >= 100:
                break
    audit.check(
        "no_other_actmask_output_branch_modified_after_P0",
        not touched,
        {
            "P0_timestamp_utc": run_rows[0]["timestamp_utc"],
            "files_with_newer_mtime": touched,
            "note": "mtime audit cannot attribute concurrent external writes",
        },
    )


def _reproducibility(root: Path, audit: Audit) -> None:
    manifest = _json(root / "reproducibility_manifest.json")
    failures = []
    for row in manifest["output_files"]:
        path = root / row["path"]
        if (
            not path.is_file()
            or path.stat().st_size != row["bytes"]
            or sha256_file(path) != row["sha256"]
        ):
            failures.append(row["path"])
    for row in manifest["implementation_files"]:
        path = ROOT / row["path"]
        if not path.is_file() or sha256_file(path) != row["sha256"]:
            failures.append(row["path"])
    audit.check(
        "reproducibility_manifest_hashes",
        not failures
        and manifest["preregistered_config_sha256"]
        == sha256_file(root / "preregistered_config.json"),
        {
            "output_files": len(manifest["output_files"]),
            "implementation_files": len(manifest["implementation_files"]),
            "failures": failures,
        },
    )


def verify(root: str | Path = OUT) -> dict[str, Any]:
    output = Path(root).resolve()
    audit = Audit()
    _required_files(output, audit)
    if not audit.passed:
        return {
            "schema": "actioncheck-independent-verification-v1",
            "pass": False,
            "checks": audit.checks,
        }
    run_rows = _preregistration(output, audit)
    groups, manifest = _dataset(output, audit)
    _source_hashes(output, audit)
    _builder(output, groups, manifest, audit)
    _predictions_and_decision(output, manifest, audit)
    _method_and_preservation(output, run_rows, audit)
    _reproducibility(output, audit)
    return {
        "schema": "actioncheck-independent-verification-v1",
        "study_id": "ACTIONCHECK-PROPOSAL-DATA-V1",
        "pass": audit.passed,
        "checks_passed": sum(row["pass"] for row in audit.checks),
        "checks_total": len(audit.checks),
        "checks": audit.checks,
    }


def _markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Independent verification",
        "",
        f"Overall: **{'PASS' if result['pass'] else 'FAIL'}** "
        f"({result.get('checks_passed', 0)}/{result.get('checks_total', len(result['checks']))} checks).",
        "",
    ]
    for row in result["checks"]:
        lines.append(
            f"- {'PASS' if row['pass'] else 'FAIL'} — `{row['name']}`"
        )
    lines.extend(
        [
            "",
            "This verifier recomputes scientific counts, action hashes, evidence,",
            "splits, builder arrays, raw-prediction metrics, shortcut termination,",
            "decision precedence, method authorization, source hashes, output",
            "hashes, the run-log chain, and protected-branch mtimes.",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(OUT))
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()
    result = verify(args.output_dir)
    if args.write_report:
        root = Path(args.output_dir)
        write_json(root / "independent_verification.json", result)
        (root / "independent_verification.md").write_text(
            _markdown(result), encoding="utf-8"
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["pass"] else 1)
