"""Cheap consistency contracts for the P-Series audit and access package."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT / "outputs" / "actmask" / "p_series_audit_and_access_package"


def _json(path: Path) -> dict:
    return json.loads(path.read_text())


def test_p_series_required_files_and_json_load() -> None:
    required = [
        "source_inventory.json", "artifact_manifest.json", "missing_artifacts_report.json",
        "failure_taxonomy.md", "failure_taxonomy.json", "claim_boundary_table.md", "claim_boundary_table.json",
        "paper_positioning.md", "future_method_gate.md", "future_method_gate.json",
        "final_report.md", "final_decision.json", "reproducibility_manifest.json", "next_steps.md",
        "docs/p_series_plan.md", "docs/p_series_results.md", "docs/p_series_handoff.md",
        "access_requests/robomind_access_request.md", "access_requests/rh20t_subset_request.md",
        "access_requests/dataset_requirement_checklist.md",
        "tables/synthetic_summary.json", "tables/real_dataset_summary.json",
        "tables/failure_taxonomy_table.json", "tables/claim_boundary_table.json", "tables/recoverability_table.json",
        "figures/fig1_pipeline_audit_flow.md", "figures/fig2_failure_taxonomy.md",
        "figures/fig3_synthetic_decision_tree.md", "figures/fig4_real_dataset_results.md",
    ]
    for relative in required:
        path = ROOT / relative
        assert path.is_file(), relative
        if path.suffix == ".json":
            _json(path)


def test_p_series_source_manifest_is_present_and_hashed() -> None:
    manifest = _json(ROOT / "artifact_manifest.json")
    missing = _json(ROOT / "missing_artifacts_report.json")
    assert manifest["all_key_artifacts_present"]
    assert len(manifest["artifacts"]) >= 62
    assert missing["required_key_artifacts_missing"] == []
    for item in manifest["artifacts"]:
        path = PROJECT / item["path"]
        assert item["exists"] and path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]


def test_taxonomy_claims_tables_and_gate_are_consistent() -> None:
    taxonomy = _json(ROOT / "failure_taxonomy.json")["categories"]
    taxonomy_table = _json(ROOT / "tables/failure_taxonomy_table.json")["rows"]
    expected = {
        "global-motion shortcut", "estimator / CI mismatch", "missing raw-score/checkpoint artifacts",
        "fair-observation unidentifiability", "token-slot leakage", "missing goal cue",
        "clean object-token temporal saturation", "logged real-trajectory consistency saturation",
        "numeric-only real failure signal insufficiency", "visual prehistory recoverability insufficiency",
    }
    assert {item["category"] for item in taxonomy} == expected
    assert taxonomy_table == taxonomy
    for item in taxonomy:
        assert all(item[key] for key in ("milestone", "evidence", "diagnosis", "invalidated", "valid"))
    claims = _json(ROOT / "claim_boundary_table.json")["rows"]
    assert _json(ROOT / "tables/claim_boundary_table.json")["rows"] == claims
    assert any(any("ActMask improves real-robot failure prediction" in value for value in row["not_supported"]) for row in claims)
    gate = _json(ROOT / "future_method_gate.json")
    assert len(gate["checks"]) == 9
    assert [row["id"] for row in gate["checks"]] == list(range(1, 10))
    assert "<= 0.65" in gate["checks"][4]["requirement"]
    assert ">= 0.65" in gate["checks"][5]["requirement"]
    assert "< 0.90" in gate["checks"][6]["requirement"]
    assert ">= 0.08" in gate["checks"][7]["requirement"]


def test_final_decision_access_requests_and_output_hashes() -> None:
    final = _json(ROOT / "final_decision.json")
    assert final["decision"] == "AUDIT_PAPER_READY_WITH_MINOR_WRITING"
    assert final["recommended_track"].startswith("B.")
    assert not final["method_claim_authorized"]
    robo = (ROOT / "access_requests/robomind_access_request.md").read_text().lower()
    rh20t = (ROOT / "access_requests/rh20t_subset_request.md").read_text().lower()
    for phrase in ("100 success/failure", "rgb", "robot state", "action", "multiple attempts", "file inventory"):
        assert phrase in robo
    for phrase in ("100 contact-rich", "completion rating", "rgb-d", "force/tactile", "manageable"):
        assert phrase in rh20t
    reproducibility = _json(ROOT / "reproducibility_manifest.json")
    assert reproducibility["no_training_or_source_mutation"]
    for item in reproducibility["generated_files"]:
        path = ROOT / item["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
    package = _json(ROOT / "package_manifest.json")
    for item in package["files"]:
        path = ROOT / item["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
