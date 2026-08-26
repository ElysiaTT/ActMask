"""Regression checks for the read-only Milestone 3S paper-package builder."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/reproduce_milestone3s_tables.py"
FULL_CONFIG = ROOT / "outputs/actmask/milestone3r_nl_v2/full_run_config.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_milestone3s_rebuild_is_read_only_and_audits_pass(tmp_path):
    before = digest(FULL_CONFIG)
    output = tmp_path / "paper_package"
    subprocess.run([sys.executable, str(SCRIPT), "--output", str(output), "--no-figures"], cwd=ROOT, check=True)
    assert digest(FULL_CONFIG) == before
    manifest = json.loads((output / "artifact_manifest.json").read_text())
    metrics = json.loads((output / "independent_metric_audit.json").read_text())
    red = json.loads((output / "red_team_audit.json").read_text())
    decision = json.loads((output / "milestone3s_decision.json").read_text())
    assert manifest["passed"] and metrics["passed"] and red["passed"]
    assert red["blocking_fail_count"] == 0
    assert decision["decision"] == "A. PAPER PACKAGE READY"
    assert not decision["visual_stage_authorized"]


def test_milestone3s_tables_are_traceable_and_no_v1_is_final_proof():
    output = ROOT / "outputs/actmask/milestone3s_paper_package"
    main = json.loads((output / "tables/main_v2_results.json").read_text())
    assert any(row["method"] == "GRU" and row["ID_pair_order"] == "1.000" for row in main["rows"])
    audit = json.loads((output / "red_team_audit.json").read_text())
    stale = next(item for item in audit["checks"] if item["issue"] == "stale_v1_as_v2_proof")
    assert stale["status"] == "PASS"
    claims = (ROOT / "docs/milestone3s_claim_boundaries.md").read_text()
    assert "real-robot success" in claims and "state observations" in claims
