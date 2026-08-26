"""RM-ACD must fail closed before training when non-robot masking is invalid."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1] / "outputs" / "actmask" / "rm_acd_action_conditioned_change"


def load(name: str) -> dict:
    return json.loads((ROOT / name).read_text())


def test_source_alignment_modalities_and_temporal_separation() -> None:
    modality = load("modality_inventory.json")["summary"]
    target = load("target_construction_audit.json")
    schema = load("target_schema.json")
    assert modality["all_numeric_alignment_pass"]
    assert modality["all_six_rgb_views_frame_aligned"]
    assert target["all_temporally_separated"]
    assert target["fair_input_audit"]["pass"]
    assert schema["fair_input_audit"]["forbidden_fields_present"] == []
    assert target["target_paths_separate_from_fair_inputs"]


def test_target_sample_schema_and_source_identity_are_reproducible() -> None:
    rows = [json.loads(line) for line in (ROOT / "target_manifest.jsonl").read_text().splitlines() if line]
    assert len(rows) == 2400
    assert all(max(row["pre_frame_indices"]) < min(row["action_frame_indices"]) < row["future_frame_index"] for row in rows)
    sample = rows[0]
    with np.load(ROOT / sample["input_path"]) as inputs, np.load(ROOT / sample["target_path"]) as target:
        assert set(inputs.files) == {"pre_rgb", "pre_state", "action"}
        assert set(target.files) == {"mask", "confidence", "flow", "feature_residual"}
        assert inputs["pre_rgb"].shape[0] == 4 and target["flow"].shape[-1] == 2
    source = load("source_hash_manifest.json")
    assert len(source["files"]) == 200 and all(len(item["sha256"]) == 64 for item in source["files"])


def test_robot_mask_gate_rejects_before_splits_baselines_or_method() -> None:
    robot = load("robot_mask_audit.json")
    quality = load("target_quality_audit.json")
    final = load("final_decision.json")
    raw = load("raw_prediction_manifest.json")
    assert not robot["all_mask_area_below_half"]
    assert not robot["automatic_stability_check"]
    assert all(row["mean_mask_area"] > 0.50 for row in robot["task_summary"].values())
    assert not quality["pass"] and quality["prerequisite_failure"] == "RM_ACD_ROBOT_MASK_INVALID"
    assert final["decision"] == "RM_ACD_ROBOT_MASK_INVALID"
    assert not final["baseline_ran"] and not final["method_ran"]
    assert raw["baseline_predictions"] == [] and raw["method_predictions"] == []


def test_contact_coverage_hashes_and_final_schema() -> None:
    contacts = load("robot_mask_contact_sheet_manifest.json")
    repro = load("reproducibility_manifest.json")
    assert len(contacts["samples"]) == 20 and set(contacts["coverage"].values()) == {2}
    assert all((ROOT / row["path"]).is_file() for row in contacts["samples"])
    assert not repro["method_trained"] and repro["target_artifacts_audit_only"]
    for row in repro["sample_input_target_hashes"]:
        assert hashlib.sha256((ROOT / row["input_path"]).read_bytes()).hexdigest() == row["input_sha256"]
        assert hashlib.sha256((ROOT / row["target_path"]).read_bytes()).hexdigest() == row["target_sha256"]
    assert all(len(value) == 64 for value in repro["tracked_artifact_sha256"].values())


def test_terminal_stop_prohibits_splits_shortcut_controls_and_checkpoints() -> None:
    completion = load("completion_audit.json")
    final = load("final_decision.json")
    assert completion["terminal_decision"] == final["decision"]
    stopped = next(row for row in completion["requirements"] if row["phase"] == "A5-A9")
    assert stopped["status"] == "not_applicable_by_A2_stop_rule"
    assert not (ROOT / "split_audit.json").exists()
    assert not (ROOT / "method_summary.json").exists()
    assert not any((ROOT / "checkpoints").glob("*")) if (ROOT / "checkpoints").exists() else True
