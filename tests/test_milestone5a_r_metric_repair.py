"""5A-R frozen inventory and raw-score stop safeguards."""
from __future__ import annotations
import hashlib,json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"outputs"/"actmask"/"milestone5a_r_metric_repair"
HASH="a5d7bad1b7bf499940eb8071e6180cecd7bbb261063ac09d115ab8838d50868d"

def test_config_and_frozen_inventory_hashes_are_stable():
    assert hashlib.sha256((OUT/"preregistered_config.json").read_bytes()).hexdigest()==HASH
    inventory=json.loads((OUT/"source_inventory.json").read_text())
    for source in inventory["sources"]:
        path=ROOT/source["path"]
        assert source["frozen"] is True
        assert hashlib.sha256(path.read_bytes()).hexdigest()==source["sha256"]

def test_raw_scores_missing_prevents_mixed_estimand_repair():
    report=json.loads((OUT/"recomputed_standard_report.json").read_text()); ci=json.loads((OUT/"ci_consistency_audit.json").read_text()); decision=json.loads((OUT/"milestone5a_r_final_decision.json").read_text())
    assert report["raw_scores_sufficient"] is False
    assert ci["status"]=="RAW_SCORES_MISSING_STOP"
    assert decision["decision"]=="C. STOP_RAW_SCORES_MISSING"
