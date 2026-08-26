"""V10 independent recomputation and preregistered shortcut-evidence gate."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .common import OUT, SEEDS, append_log, read_json, sha256_file, write_json
from .metrics import binary_metrics
from .representation_probes import verify as verify_representation_probes


ARCHITECTURES = (
    "StateActionTCN",
    "TaskConditionedActionTransformer",
    "VisualStateActionTransformer",
    "ContrastiveContextActionVerifier",
    "CrossAttentionEnergyVerifier",
)
EASY_FAMILIES = ("N2", "N3", "N5", "N6")
STANDARD_METRICS = (
    "balanced_accuracy",
    "AUROC",
    "AUPRC",
    "ECE",
    "Brier",
    "false_accept_rate",
    "false_reject_rate",
)


def _prediction_payload(path: str | Path) -> dict[str, Any]:
    with np.load(path) as stored:
        if "test_index" in stored.files:
            return {
                "label": stored["test_label"].astype(np.int8),
                "probability": stored["test_calibrated_probability"].astype(
                    np.float32
                ),
                "threshold": float(stored["threshold"][0]),
                "weight": None,
            }
        return {
            "label": stored["label"].astype(np.int8),
            "probability": stored["calibrated_probability"].astype(np.float32),
            "threshold": float(stored["threshold"][0]),
            "weight": (
                stored["weight"].astype(np.float32)
                if "weight" in stored.files
                else None
            ),
        }


def _compare_metric_rows(
    rows: list[dict[str, Any]],
    *,
    path_key: str,
    hash_key: str,
    tolerance: float = 3e-6,
) -> dict[str, Any]:
    mismatches = []
    hash_failures = []
    checked = 0
    for row_number, row in enumerate(rows):
        path = Path(row[path_key])
        expected_hash = row[hash_key]
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            hash_failures.append(
                {
                    "row": row_number,
                    "path": str(path),
                    "expected": expected_hash,
                    "actual": actual_hash,
                }
            )
            continue
        payload = _prediction_payload(path)
        recomputed = binary_metrics(
            payload["probability"],
            payload["label"],
            payload["threshold"],
            payload["weight"],
        )
        expected = row["metrics"]
        for metric in STANDARD_METRICS:
            left = recomputed[metric]
            right = expected[metric]
            if left is None or right is None:
                equal = left is None and right is None
            else:
                equal = abs(float(left) - float(right)) <= tolerance
            if not equal:
                mismatches.append(
                    {
                        "row": row_number,
                        "path": str(path),
                        "metric": metric,
                        "expected": right,
                        "recomputed": left,
                    }
                )
        checked += 1
    return {
        "rows": len(rows),
        "rows_checked": checked,
        "hash_failures": hash_failures,
        "metric_mismatches": mismatches,
        "pass": not hash_failures and not mismatches and checked == len(rows),
    }


def _independent_recomputation() -> dict[str, Any]:
    verifier = read_json(OUT / "verifier_per_environment.json")
    controls = read_json(OUT / "shortcut_control_per_environment.json")
    transfer = read_json(OUT / "generator_transfer_matrix.json")
    verifier_result = _compare_metric_rows(
        verifier["rows"],
        path_key="raw_prediction_path",
        hash_key="raw_prediction_sha256",
    )
    control_result = _compare_metric_rows(
        controls["rows"],
        path_key="raw_prediction_path",
        hash_key="raw_prediction_sha256",
    )
    transfer_result = _compare_metric_rows(
        transfer["per_seed"],
        path_key="raw_prediction_path",
        hash_key="raw_prediction_sha256",
    )
    representation_result = verify_representation_probes()
    nuisance = read_json(OUT / "nuisance_probe_report.json")
    nuisance_result = nuisance["coverage"]["complete"] and nuisance["status"] == "PASS"
    result = {
        "schema": "vsa-independent-metric-recomputation-v1",
        "verifier_environment_predictions": verifier_result,
        "shortcut_control_predictions": control_result,
        "family_transfer_predictions": transfer_result,
        "representation_probe_predictions": representation_result,
        "nuisance_probe_predictions": {
            "pass": bool(nuisance_result),
            "probe_files": nuisance["coverage"]["raw_prediction_files"],
            "independent_recomputation": nuisance.get(
                "independent_recomputation"
            ),
        },
        "total_metric_rows_recomputed": (
            verifier_result["rows"]
            + control_result["rows"]
            + transfer_result["rows"]
        ),
        "pass": bool(
            verifier_result["pass"]
            and control_result["pass"]
            and transfer_result["pass"]
            and representation_result["complete"]
            and nuisance_result
        ),
    }
    write_json(OUT / "independent_metric_recomputation.json", result)
    return result


def _training_records() -> dict[tuple[str, int, str], dict[str, Any]]:
    return {
        (row["model"], int(row["seed"]), row["spec"]): row
        for row in read_json(OUT / "verifier_per_seed.json")["training_runs"]
    }


def _easy_iid_gate(
    training: dict[tuple[str, int, str], dict[str, Any]]
) -> dict[str, Any]:
    per_architecture = []
    qualifying = []
    for architecture in ARCHITECTURES:
        family_rows = []
        architecture_holds = False
        for family in EASY_FAMILIES:
            seed_rows = [
                {
                    "seed": seed,
                    "balanced_accuracy": training[
                        (architecture, seed, f"within_{family.lower()}")
                    ]["test_metrics"]["balanced_accuracy"],
                }
                for seed in SEEDS
            ]
            passing_seeds = [
                row["seed"]
                for row in seed_rows
                if row["balanced_accuracy"] >= 0.65
            ]
            holds = len(passing_seeds) >= 2
            architecture_holds |= holds
            family_rows.append(
                {
                    "family": family,
                    "seed_rows": seed_rows,
                    "passing_seeds": passing_seeds,
                    "holds_across_at_least_two_seeds": holds,
                }
            )
        if architecture_holds:
            qualifying.append(architecture)
        per_architecture.append(
            {
                "architecture": architecture,
                "qualifies": architecture_holds,
                "families": family_rows,
            }
        )
    return {
        "threshold": 0.65,
        "easy_families": list(EASY_FAMILIES),
        "required_architectures": 2,
        "qualifying_architectures": qualifying,
        "qualifying_architecture_count": len(qualifying),
        "pass": len(qualifying) >= 2,
        "per_architecture": per_architecture,
    }


def _indicator_gate(
    training: dict[tuple[str, int, str], dict[str, Any]],
    shortcut: dict[str, Any],
) -> dict[str, Any]:
    thresholds = read_json(OUT / "preregistered_config.json")[
        "shortcut_evidence_gate"
    ]["indicator_thresholds"]
    generator = shortcut["generator_and_easy_to_real"]["generator_rows"]
    easy = shortcut["generator_and_easy_to_real"]["easy_to_real_rows"]
    source = shortcut["source_heldout_rows"]
    matching = read_json(OUT / "matching_gap_report.json")["rows"]
    nuisance = read_json(OUT / "nuisance_probe_report.json")
    representation = read_json(OUT / "representation_probe_summary.json")
    controls = read_json(OUT / "shortcut_control_per_environment.json")["rows"]
    control_reference = {
        row["control"]: row["metrics"]["balanced_accuracy"]
        for row in controls
        if row["training_spec"] == "pooled_n1_n6"
        and row["condition"] == "default_test"
    }
    best_control_name = max(control_reference, key=control_reference.get)
    best_control_ba = control_reference[best_control_name]
    nuisance_qualifying = set(
        nuisance["aggregates"][
            "architectures_qualifying_nuisance_score_probe_indicator"
        ]
    )
    family_probe_qualifying = set(
        representation["family_representation_probe_gate"][
            "architectures_meeting_threshold_across_at_least_two_seeds"
        ]
    )
    per_architecture = []
    qualifying = []
    for architecture in ARCHITECTURES:
        evidence: dict[str, Any] = {}

        def seed_passes(
            rows: list[dict[str, Any]], field: str, threshold: float
        ) -> list[int]:
            return sorted(
                {
                    int(row["seed"])
                    for row in rows
                    if row["architecture"] == architecture
                    and float(row[field]) >= threshold
                }
            )

        generator_seeds = seed_passes(
            generator,
            "generator_transfer_gap",
            thresholds["generator_heldout_drop"],
        )
        easy_seeds = seed_passes(
            easy, "easy_to_real_gap", thresholds["easy_to_real_drop"]
        )
        action_seeds = seed_passes(
            [
                row
                for row in matching
                if row["environment"] == "E5" and row["grouping"] == "all"
            ],
            "action_matching_gap",
            thresholds["action_matched_drop"],
        )
        progress_seeds = seed_passes(
            [
                row
                for row in matching
                if row["environment"] == "E6" and row["grouping"] == "all"
            ],
            "progress_matching_gap",
            thresholds["progress_matched_drop"],
        )
        source_seeds = seed_passes(
            source,
            "source_heldout_gap",
            thresholds["source_heldout_drop"],
        )
        control_seeds = [
            seed
            for seed in SEEDS
            if best_control_ba
            >= training[(architecture, seed, "pooled_n1_n6")]["test_metrics"][
                "balanced_accuracy"
            ]
            - thresholds["nuisance_control_within_full"]
        ]
        seed_based = {
            "generator_heldout_drop": generator_seeds,
            "easy_to_real_drop": easy_seeds,
            "action_matched_drop": action_seeds,
            "progress_matched_drop": progress_seeds,
            "source_heldout_drop": source_seeds,
            "nuisance_control_within_full": control_seeds,
        }
        for name, seeds in seed_based.items():
            evidence[name] = {
                "passing_seeds": seeds,
                "holds_across_at_least_two_seeds": len(seeds) >= 2,
            }
        evidence["nuisance_score_probe_R2"] = {
            "qualifies": architecture in nuisance_qualifying
        }
        evidence["family_representation_probe_BA"] = {
            "qualifies": architecture in family_probe_qualifying
        }
        indicator_names = [
            name
            for name, value in evidence.items()
            if (
                value.get("holds_across_at_least_two_seeds", False)
                or value.get("qualifies", False)
            )
        ]
        meets = len(indicator_names) >= 2
        if meets:
            qualifying.append(architecture)
        per_architecture.append(
            {
                "architecture": architecture,
                "evidence": evidence,
                "indicator_names": indicator_names,
                "indicator_count": len(indicator_names),
                "required_indicator_count": 2,
                "qualifies": meets,
            }
        )
    return {
        "thresholds": thresholds,
        "best_shortcut_control": best_control_name,
        "best_shortcut_control_pooled_BA": best_control_ba,
        "required_architectures": 2,
        "qualifying_architectures": qualifying,
        "qualifying_architecture_count": len(qualifying),
        "pass": len(qualifying) >= 2,
        "per_architecture": per_architecture,
    }


def _task_gate(shortcut: dict[str, Any]) -> dict[str, Any]:
    generator_tasks = shortcut["generator_and_easy_to_real"][
        "generator_task_rows"
    ]
    matching_tasks = [
        row
        for row in read_json(OUT / "matching_gap_report.json")["rows"]
        if row["grouping"] == "task"
    ]
    task_evidence: dict[str, list[str]] = defaultdict(list)
    for row in generator_tasks:
        if row["generator_transfer_gap"] >= 0.10:
            task_evidence[row["task"]].append(
                f"{row['architecture']}:generator:{row['heldout_family']}:seed{row['seed']}"
            )
    for row in matching_tasks:
        field = (
            "action_matching_gap"
            if row["environment"] == "E5"
            else "progress_matching_gap"
        )
        threshold = 0.08
        if row[field] >= threshold:
            task_evidence[row["group"]].append(
                f"{row['architecture']}:{row['environment']}:seed{row['seed']}"
            )
    qualifying_tasks = sorted(
        task for task, evidence in task_evidence.items() if evidence
    )
    return {
        "required_tasks": 2,
        "qualifying_tasks": qualifying_tasks,
        "qualifying_task_count": len(qualifying_tasks),
        "pass": len(qualifying_tasks) >= 2,
        "evidence_by_task": {
            task: evidence for task, evidence in sorted(task_evidence.items())
        },
    }


def decide() -> dict[str, Any]:
    required = [
        "imported_hash_verification.json",
        "shortcut_reliance_summary.json",
        "generator_transfer_matrix.json",
        "matching_gap_report.json",
        "nuisance_probe_report.json",
        "calibration_shift_report.json",
        "relational_gain_report.json",
        "representation_probe_summary.json",
    ]
    missing = [name for name in required if not (OUT / name).is_file()]
    if missing:
        raise RuntimeError("V10 prerequisites missing: " + ", ".join(missing))
    imported = read_json(OUT / "imported_hash_verification.json")
    recomputation = _independent_recomputation()
    training = _training_records()
    shortcut = read_json(OUT / "shortcut_reliance_summary.json")
    easy_gate = _easy_iid_gate(training)
    indicator_gate = _indicator_gate(training, shortcut)
    task_gate = _task_gate(shortcut)
    historical_complete = bool(imported.get("pass", imported.get("all_pass", False)))
    if not historical_complete or not recomputation["pass"]:
        decision = "VSA_HISTORICAL_ARTIFACTS_INCOMPLETE"
    elif not easy_gate["pass"]:
        decision = "VSA_NO_LEARNABLE_IID_TASK"
    elif not indicator_gate["pass"]:
        decision = "VSA_NO_SYSTEMATIC_SHORTCUT_EVIDENCE"
    elif not task_gate["pass"]:
        decision = "VSA_SHORTCUT_EVIDENCE_TASK_SPECIFIC"
    else:
        decision = "VSA_SYSTEMATIC_SHORTCUT_SUPPORTED"
    mitigation_authorized = decision == "VSA_SYSTEMATIC_SHORTCUT_SUPPORTED"
    value = {
        "schema": "vsa-phase-v10-decision-v1",
        "study_id": "VERIFIER-SHORTCUT-AUDIT-V1",
        "decision": decision,
        "systematic_shortcut_supported": mitigation_authorized,
        "mitigation_authorized": mitigation_authorized,
        "must_stop_before_mitigation": not mitigation_authorized,
        "gate_order": [
            "historical completeness and independent recomputation",
            "at least two fair architectures learn an easy family at IID BA >= 0.65",
            "at least two architectures have at least two shortcut indicators",
            "evidence spans at least two tasks",
        ],
        "historical_artifact_gate": {
            "pass": historical_complete,
            "source": str((OUT / "imported_hash_verification.json").resolve()),
            "source_sha256": sha256_file(
                OUT / "imported_hash_verification.json"
            ),
        },
        "independent_recomputation_gate": recomputation,
        "easy_IID_learnability_gate": easy_gate,
        "shortcut_indicator_gate": indicator_gate,
        "task_span_gate": task_gate,
        "claim_boundary": (
            "decision concerns systematic shortcut evidence in construction-label "
            "benchmarks, not physical failure truth or robot safety"
        ),
        "physical_ground_truth": False,
    }
    write_json(OUT / "phase_v10_decision.json", value)
    append_log(
        "V10_SHORTCUT_EVIDENCE_GATE_COMPLETE",
        decision=decision,
        mitigation_authorized=mitigation_authorized,
        independent_recomputation_pass=recomputation["pass"],
        easy_IID_qualifying_architectures=easy_gate[
            "qualifying_architecture_count"
        ],
        indicator_qualifying_architectures=indicator_gate[
            "qualifying_architecture_count"
        ],
        qualifying_tasks=task_gate["qualifying_task_count"],
        decision_sha256=sha256_file(OUT / "phase_v10_decision.json"),
    )
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    result = decide()
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "mitigation_authorized": result["mitigation_authorized"],
                "easy_IID_qualifying_architectures": result[
                    "easy_IID_learnability_gate"
                ]["qualifying_architecture_count"],
                "indicator_qualifying_architectures": result[
                    "shortcut_indicator_gate"
                ]["qualifying_architecture_count"],
                "qualifying_tasks": result["task_span_gate"][
                    "qualifying_task_count"
                ],
                "independent_recomputation_pass": result[
                    "independent_recomputation_gate"
                ]["pass"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
