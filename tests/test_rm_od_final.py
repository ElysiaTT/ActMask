"""Final RM-OD rejection package must remain reproducible and scope-bounded."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from actmask.experiments.rm_od_metrics import group_metrics


ROOT = Path(__file__).resolve().parents[1] / "outputs" / "actmask" / "rm_od_future_object_dynamics"


def load(name: str) -> dict:
    return json.loads((ROOT / name).read_text())


def test_final_decision_and_benchmark_rejection_are_consistent() -> None:
    final = load("final_decision.json")
    baseline = load("baseline_report.json")
    feasibility = load("repair_feasibility_audit.json")
    assert final["decision"] == "RM_OD_ACTION_PATH_SHORTCUT"
    assert not final["method_ran"]
    assert baseline["summary"]["decision"] == final["decision"]
    assert feasibility["source_path_repair1_recall_at_1"] == 0.125
    assert feasibility["same_source_progress_recall_at_1"] == 1.0
    assert not feasibility["additional_hard_negative_repair_performed"]


def test_all_declared_raw_scores_hash_and_recompute() -> None:
    manifest = load("raw_score_manifest.json")
    audit = load("metric_recomputation_audit.json")
    assert manifest["all_entries_present"] and audit["pass"]
    assert audit["entries_checked"] == len(manifest["entries"])
    for entry in manifest["entries"]:
        path = ROOT / entry["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        actual = group_metrics(rows)
        assert all(abs(float(actual[key]) - float(entry["metrics"][key])) <= 1e-12 for key in actual)


def test_reproducibility_manifest_and_claim_boundary() -> None:
    reproducibility = load("reproducibility_manifest.json")
    assert reproducibility["metric_recomputation_pass"]
    assert not reproducibility["method_trained"]
    assert all(len(value) == 64 for value in reproducibility["tracked_artifact_sha256"].values())
    boundary = (ROOT / "claim_boundary.md").read_text().lower()
    assert "does **not** support" in boundary
    assert "failure" in boundary and "counterfactual" in boundary
