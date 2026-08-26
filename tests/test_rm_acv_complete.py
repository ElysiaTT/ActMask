"""Contracts for the RM-ACV A1 data-admission terminal package."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "outputs" / "actmask" / "rm_acv_complete"


def load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def test_a1_terminal_decision_is_data_insufficiency_not_a_model_result() -> None:
    inventory = load("dataset_inventory.json")
    decision = load("phase_a9_decision.json")
    assert inventory["total_discovered_trajectories"] == 238
    assert inventory["numeric_and_metadata_usable_trajectories"] == 238
    assert inventory["primary_embodiment_usable_trajectories"] == 166
    assert inventory["tasks_meeting_minimum_5_trajectories"] == 11
    assert inventory["auxiliary_hdf5_inventory"]["file_count"] == 12
    assert len(inventory["incomplete_metadata_only_tasks"]) == 2
    assert not inventory["admitted"]
    assert inventory["requirements"]["minimum_5_entirely_held_out_tasks"]
    assert not inventory["requirements"]["minimum_300_usable_trajectories"]
    assert not inventory["requirements"]["minimum_20_tasks"]
    assert decision["decision"] == "RM_ACV_DATA_INSUFFICIENT"
    assert decision["terminal"] and decision["stage_reached"] == "A1"
    assert not decision["learned_baselines_run"]
    assert not decision["method_training_authorized"]


def test_required_terminal_artifacts_are_present_and_not_misrepresented() -> None:
    for name in [
        "anchor_manifest.jsonl", "positive_candidate_manifest.jsonl", "negative_candidate_manifest.jsonl",
        "shortcut_predictions.jsonl", "raw_prediction_manifest.json", "baseline_summary.json",
        "final_decision.json", "final_report.md", "claim_boundary.md", "next_steps.md",
        "paper_reframe/experiments.md", "paper_reframe/results_tables.md",
    ]:
        assert (ROOT / name).is_file()
    assert not (ROOT / "anchor_manifest.jsonl").read_text(encoding="utf-8").strip()
    assert not (ROOT / "negative_candidate_manifest.jsonl").read_text(encoding="utf-8").strip()
    baseline = load("baseline_summary.json")
    assert baseline["status"] == "not_run"
    assert not baseline["learned_baseline_trained"]


def test_reproducibility_manifest_and_hash_chained_log_are_valid() -> None:
    reproducibility = load("reproducibility_manifest.json")
    source_hashes = load("source_hash_manifest.json")
    assert reproducibility["decision"] == "RM_ACV_DATA_INSUFFICIENT"
    assert len(reproducibility["files"]) >= 50
    assert len(source_hashes["entries"]) == 284
    for entry in reproducibility["files"]:
        path = ROOT / entry["path"]
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
    previous = ""
    for line in (ROOT / "run_log.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        assert row["previous_event_sha256"] == previous
        asserted = dict(row)
        actual = asserted.pop("event_sha256")
        assert actual == hashlib.sha256(json.dumps(asserted, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        previous = actual
