#!/usr/bin/env python3
"""Structural verifier for the terminal invalid-target SAM3_PSEUDOMASK branch."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "actmask" / "rm_sam3_acd_pilot"

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def main() -> None:
    assert sha256(OUT / "preregistered_config.json") == (OUT / "preregistered_config.sha256").read_text().strip()
    assert sha256(OUT / "pseudotarget_preregistered_protocol.json") == (OUT / "pseudotarget_preregistered_protocol.sha256").read_text().strip()
    assert sum(1 for _ in (OUT / "pseudomask_manifest.jsonl").open()) == 2967
    assert sum(1 for _ in (OUT / "target_manifest_sam3.jsonl").open()) == 2400
    screen = json.loads((OUT / "pseudomask_screening_audit.json").read_text()); assert screen["decision"] == "SAM3_PSEUDOMASK_PILOT_ACCEPTED"
    targets = json.loads((OUT / "target_statistics_sam3.json").read_text()); assert targets["decision"] == "SAM3_PSEUDOMASK_TARGET_INVALID"
    final = json.loads((OUT / "final_decision.json").read_text()); assert final["decision"] == "SAM3_PSEUDOMASK_TARGET_INVALID" and final["terminal"]
    baseline = json.loads((OUT / "baseline_summary.json").read_text()); assert baseline["status"] == "not_run"
    assert len(list((OUT / "pseudomask_contact_sheets").glob("*.jpg"))) == 10
    assert len(list((OUT / "pseudotarget_contact_sheets").glob("*.jpg"))) == 20
    for path in ("final_report.md", "pseudomask_quality_report.md", "target_quality_report.md", "baseline_summary.json", "raw_prediction_manifest.json", "reproducibility_manifest.json", "claim_boundary.md", "paper_positioning.md", "next_steps.md"):
        assert (OUT / path).is_file(), path
    print("SAM3_PSEUDOMASK terminal package: 16 checks passed")

if __name__ == "__main__": main()
