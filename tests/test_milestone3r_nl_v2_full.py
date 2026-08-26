import hashlib
import json
from pathlib import Path


ROOT = Path("outputs/actmask/milestone3r_nl_v2")
FULL_HASH = "12c692d032ba37bdb14761a1c63ff8ceccff72af5f037d936ebaed8748016a47"


def test_full_config_hash_and_execution_budget_are_frozen():
    assert hashlib.sha256((ROOT / "full_run_config.json").read_bytes()).hexdigest() == FULL_HASH
    reports = list((ROOT / "full_raw").glob("*/generation_report.json"))
    assert sum(json.loads(path.read_text())["candidate_executions"] for path in reports) == 92000


def test_c5_c10_prefixes_are_aligned_no_extra_simulation():
    for count in (5, 10):
        report = json.loads((ROOT / "full" / f"id_c{count}" / "derivation_report.json").read_text())
        assert report["candidate_prefix"] == count
        assert report["simulator_executions_added"] == 0


def test_full_gate_has_evidence_for_all_required_axes():
    report = json.loads((ROOT / "full_evaluation" / "full_evaluation.json").read_text())
    assert report["full_gate_passed"]
    assert all(report["gate"].values())
    assert report["bootstrap"]["ci95"][0] > 0
    assert sum(item["learned"] > item["analytic"] for item in report["ood"]) >= 2
    for prefix in ("C5", "C10", "C20"):
        ranking = report["rankings"][prefix]
        assert ranking["dynamic"]["top1_success"] - ranking["ActionMLP"]["top1_success"] >= .1
