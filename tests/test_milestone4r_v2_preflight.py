"""Regression checks for the isolated 4R-v2 candidate-diversity repair."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "actmask" / "milestone4r_v2_candidate_diversity"
EXPECTED_HASH = "f1678a155473f385a9cb3670c104af78a8d0f717726ae3e3989fc7162d31cc46"


def test_v2_config_and_preprobe_are_frozen_and_passed():
    assert hashlib.sha256((OUTPUT / "preregistered_config.json").read_bytes()).hexdigest() == EXPECTED_HASH
    audit = json.loads((OUTPUT / "preprobe_attempt_1" / "placement_preprobe_audit.json").read_text())
    assert audit["passed"] is True
    assert audit["execution"]["candidate_executions"] == 1280
    assert audit["candidate_diversity"]["mixed_success_fraction"] == 1.0
    assert audit["leakage_diagnostics"]["candidate_index_only_majority_accuracy"] == 0.5


def test_v2_full_audit_repairs_placement_but_stops_on_other_families():
    root = OUTPUT / "full_probe"
    audit = json.loads((root / "milestone4r_v2_full_audit.json").read_text())
    decision = json.loads((root / "milestone4r_v2_final_decision.json").read_text())
    placement = audit["tasks"]["SignedMovingWindowPlacement"]
    assert placement["passed"] is True
    assert placement["mixed_success_fraction"] == 1.0
    assert audit["tasks"]["MovingCubeIntercept"]["passed"] is False
    assert audit["tasks"]["FixedPhaseRotatingCaptureWindow"]["passed"] is False
    assert audit["passed"] is False
    assert decision["decision"] == "E. BENCHMARK INVALID"
    assert decision["milestone4m_authorized"] is False
