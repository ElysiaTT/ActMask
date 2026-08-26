"""Frozen v3 diversity, integrity, and decision regression checks."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/"outputs"/"actmask"/"milestone4r_v3_all_family_candidate_diversity"
HASH="c1011cf8fead7eba68881a151ee9de3df922a9efaab7e4c331bc973d4ebf159f"

def test_v3_config_and_final_preprobe_are_frozen():
    assert hashlib.sha256((OUTPUT/"preregistered_config.json").read_bytes()).hexdigest()==HASH
    audit=json.loads((OUTPUT/"preprobes_attempt_3"/"all_family_preprobe_audit.json").read_text())
    assert audit["passed"] is True
    for task,value in audit["tasks"].items():
        assert value["passed"] is True
        assert value["candidate_diversity"]["mixed_success_fraction"]>=.8
        assert value["leakage"]["candidate_index_only_majority_accuracy"]<=.6

def test_v3_full_integrity_and_easy_benchmark_decision():
    root=OUTPUT/"full_probe"; audit=json.loads((root/"milestone4r_v3_full_audit.json").read_text()); decision=json.loads((root/"milestone4r_v3_final_decision.json").read_text()); headroom=json.loads((root/"headroom_metrics.json").read_text())
    assert audit["passed"] is True
    assert all(task["passed"] for task in audit["tasks"].values())
    assert decision["decision"]=="B. ROBUST VISUAL BENCHMARK STILL TOO EASY"
    assert decision["milestone4m_authorized"] is False
    assert all(value["centroid_velocity"]==1.0 and value["model_headroom_state_minus_best_fair"]==0.0 for value in headroom["tasks"].values())
    derived=json.loads((root/"derived_headroom_metrics.json").read_text())
    assert all(value["best_standard_fair"]=="centroid_velocity" and value["model_headroom_state_minus_best_standard_fair"]==0.0 for value in derived["tasks"].values())
