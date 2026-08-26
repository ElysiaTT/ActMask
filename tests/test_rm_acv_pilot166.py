"""Completion contracts for the bounded RM-ACV-PILOT166 terminal run."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "outputs" / "actmask" / "rm_acv_pilot166"


def load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def test_pilot_source_is_exact_and_heterogeneous_subset_is_excluded() -> None:
    config = load("preregistered_config.json")
    excluded = load("excluded_heterogeneous_source.json")
    source = [
        json.loads(line)
        for line in (ROOT / "source_manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert config["name"] == "RM-ACV-PILOT166"
    assert config["source"]["trajectory_count"] == 166
    assert config["source"]["task_count"] == 11
    assert config["source"]["state_dimension"] == config["source"]["action_dimension"] == 32
    assert len(source) == 166 and len({row["task_id_provenance_only"] for row in source}) == 11
    assert excluded["trajectory_count"] == 72 and excluded["state_action_dimension"] == 16
    assert "never merged" in excluded["reason"]
    assert (ROOT / "preregistered_config.sha256").read_text().strip() == hashlib.sha256(
        (ROOT / "preregistered_config.json").read_bytes()
    ).hexdigest()


def test_anchor_alignment_spacing_and_episode_grouping_are_audited() -> None:
    statistics = load("anchor_statistics.json")
    overlap = load("anchor_overlap_audit.json")
    alignment = load("alignment_audit.json")
    split = load("episode_split_audit.json")
    assert statistics["anchors"] == 992
    assert statistics["trajectories"] == 166 and statistics["tasks"] == 11
    assert 4 <= statistics["minimum_per_episode"] <= statistics["maximum_per_episode"] <= 6
    assert overlap["minimum_observed_spacing"] >= 16
    assert overlap["future_action_overlap_pairs"] == 0 and overlap["pass"]
    assert alignment["all_pass"] and alignment["trajectories"] == 166
    assert split["pass"] and split["no_episode_overlap"]
    assert split["all_retrieval_sources_within_episode_split"]


def test_negative_gate_terminal_is_supported_by_raw_recomputation() -> None:
    shortcut = load("shortcut_feature_report.json")
    recomputation = load("shortcut_recomputation_audit.json")
    generation = load("negative_generation_report.json")
    assert shortcut["decision"] == "RM_ACV_PILOT_NEGATIVES_INVALID"
    assert shortcut["surviving_families"] == [
        "N1_same_task_near_state",
        "N4_phase_shifted_same_task",
    ]
    assert len(shortcut["surviving_families"]) < 3
    assert generation["attempted"] == 992 * 6
    assert generation["constructed"] == 5600
    assert not generation["manual_sample_repairs"]
    assert generation["redesign_limit_respected"]
    assert recomputation["all_match"]
    assert all(row["match"] and row.get("absolute_delta", 0.0) < 1e-12 for row in recomputation["checks"])
    provisional = shortcut["provisional_survivor_only_controls"]
    assert provisional["completed"] and not provisional["can_authorize_fair_baselines"]
    assert provisional["rows"] == 3968 and provisional["positive_fraction"] == 0.5
    assert {
        "task_identity_only",
        "normalized_progress_only",
        "action_magnitude_only",
        "action_mean_variance",
        "smoothness_jerk",
        "start_end_action",
        "total_displacement",
        "episode_duration",
        "source_metadata",
        "negative_family_identity",
        "combined_non_context",
    }.issubset(provisional["feature_reports"])
    assert not (ROOT / "candidate_group_manifest.jsonl").read_text(encoding="utf-8").strip()


def test_splits_leakage_and_conditional_training_stop_are_valid() -> None:
    task_cv = load("task_cv_audit.json")
    leakage = load("leakage_audit.json")
    decision = load("final_decision.json")
    baseline = load("baseline_summary.json")
    completion = load("completion_audit.json")
    assert task_cv["pass"] and task_cv["every_task_held_out_exactly_once"]
    assert leakage["pass"] and not leakage["predictive_training_started"]
    assert decision["decision"] == "RM_ACV_PILOT_NEGATIVES_INVALID"
    assert decision["stage_reached"] == "P3" and decision["terminal"]
    assert not decision["learned_baselines_trained"]
    assert not decision["method_training_authorized"]
    assert not decision["proposed_method_trained"]
    assert baseline["status"] == "not_run"
    assert not list((ROOT / "checkpoints").glob("*.pt"))
    assert completion["terminal_condition_satisfied"]
    assert completion["phases"]["P3"]["status"] == "completed_terminal_gate"
    assert completion["phases"]["M0_M5"]["status"] == "not_authorized"


def test_prior_branch_and_paper_claim_boundary_are_preserved() -> None:
    prior = load("prior_branch_preservation_audit.json")
    assert prior["prior_decision"] == "RM_ACV_DATA_INSUFFICIENT"
    assert prior["prior_hash_verifier_pass"]
    abstract = (ROOT / "paper" / "abstract_en.md").read_text(encoding="utf-8")
    boundary = (ROOT / "paper" / "claim_boundary.md").read_text(encoding="utf-8")
    assert "166-trajectory, 11-task" in abstract
    assert "no compatibility baseline or proposed method was trained" in abstract.lower()
    assert "must not claim a working compatibility verifier" in boundary.lower()


def test_generated_hashes_and_run_log_chain_are_valid() -> None:
    manifest = load("reproducibility_manifest.json")
    assert manifest["decision"] == "RM_ACV_PILOT_NEGATIVES_INVALID"
    assert not manifest["fair_model_training_performed"]
    assert not manifest["proposed_method_training_performed"]
    for row in manifest["generated_files"]:
        path = ROOT / row["path"]
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]
    previous = ""
    sequence = 0
    for line in (ROOT / "run_log.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        actual = row.pop("event_sha256")
        assert row["previous_event_sha256"] == previous
        assert row["seq"] == sequence + 1
        expected = hashlib.sha256(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        assert actual == expected
        previous = actual
        sequence += 1
