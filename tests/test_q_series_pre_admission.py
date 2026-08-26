"""Contracts for the non-training Q-Series RH20T pre-admission package."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT / "outputs" / "actmask" / "q_series_new_data_admission"


def load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text())


def test_q_series_candidate_inventory_is_sufficient_for_pre_admission() -> None:
    candidate = load("candidate_inventory.json")
    assert candidate["selected_candidate"].startswith("RH20T cfg2")
    assert candidate["source"]["public"] and not candidate["source"]["gated"]
    inventory = candidate["inventory"]
    assert inventory["episodes"] >= 100
    assert inventory["tasks"] > 1 and inventory["task_repeat_min"] >= 2
    assert inventory["state_dim"] > 0 and inventory["action_dim"] > 0 and inventory["force_torque"]
    assert "meta.rating" in candidate["available_fields"]


def test_q_series_gate_blocks_method_without_valid_outcome_semantics() -> None:
    gate = load("future_method_gate_status.json")
    final = load("final_decision.json")
    assert len(gate["checks"]) == 9
    assert not gate["all_checks_pass"] and not gate["method_authorized"]
    assert final["decision"] == "Q_ADMISSION_REJECTED_OUTCOME_SEMANTICS_UNAVAILABLE"
    assert not final["method_authorized"]
    assert sum(item["status"] == "not_evaluated" for item in gate["checks"]) == 5
    assert "rating-threshold" in " ".join(final["prohibited_actions"])


def test_q_series_complete_numeric_audit_stops_before_models() -> None:
    integrity = load("numeric_integrity_audit.json")
    labels = load("outcome_semantics_audit.json")
    subset = load("subset_manifest.json")
    assert integrity["rows"] == 956262 and integrity["episodes"] == 1789
    assert integrity["pass"] and integrity["bad_episode_count"] == 5
    assert integrity["metadata_rating_mismatches"] == 0
    assert integrity["metadata_task_namespace_differences"] > 0
    assert labels["status"] == "fail"
    assert labels["config_n_failed"] not in labels["candidate_threshold_counts"].values()
    assert subset["episodes"] == 100 and subset["not_used_for_training"]


def test_q_series_manifest_and_missing_evidence_are_self_consistent() -> None:
    missing = load("missing_evidence.json")
    transfer = load("transfer_attempt.json")
    assert len(missing["blocking_evidence"]) >= 3
    assert transfer["result"] == "complete_and_verified"
    assert len(transfer["files"]) == 2
    repro = load("reproducibility_manifest.json")
    assert repro["no_training"] and repro["source_files_hashed_in_integrity_audit"]
    for item in repro["generated_files"]:
        path = ROOT / item["path"]
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
    package = load("package_manifest.json")
    for item in package["files"]:
        path = ROOT / item["path"]
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
