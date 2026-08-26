"""5A safeguards for frozen v3 forensic and reporting-stop evidence."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"outputs"/"actmask"/"milestone5a_relational_probe"
V3=ROOT/"outputs"/"actmask"/"milestone4r_v3_all_family_candidate_diversity"
CONFIG_HASH="dd21e5be3a517f754d9260ed38b25a5d61ac68a8021403e3d672677f3d6ff502"
V3_HASH="c1011cf8fead7eba68881a151ee9de3df922a9efaab7e4c331bc973d4ebf159f"

def test_5a_config_and_frozen_v3_nonregression():
    assert hashlib.sha256((OUT/"preregistered_config.json").read_bytes()).hexdigest()==CONFIG_HASH
    assert hashlib.sha256((V3/"preregistered_config.json").read_bytes()).hexdigest()==V3_HASH

def test_centroid_is_fair_but_ci_estimand_mismatch_stops_5a():
    forensic=json.loads((OUT/"centroid_forensics.json").read_text()); audit=json.loads((OUT/"metric_aggregation_audit.json").read_text()); decision=json.loads((OUT/"milestone5a_final_decision.json").read_text())
    assert forensic["decision"]=="A. FAIR_CENTROID_SHORTCUT_CONFIRMED"
    assert not any(forensic["forbidden_field_audit"].values())
    assert audit["ci_entries_checked"]==540 and audit["point_estimate_outside_reported_ci"]==73
    assert decision["decision"]=="D. METRIC_OR_AGGREGATION_BUG"
    assert decision["relational_task_generation_started"] is False
