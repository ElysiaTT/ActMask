#!/usr/bin/env python3
"""Independent structural verifier for the RM-MASK-ACD terminal package."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "actmask" / "rm_mask_acd_verified"
PACKAGE = OUT / "manual_annotation_package"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    assert sha256(OUT / "preregistered_config.json") == (OUT / "preregistered_config.sha256").read_text().strip()
    assert json.loads((OUT / "m0_decision.json").read_text())["decision"] == "M0_AUTOMATIC_SEGMENTATION_REQUIRED"
    assert json.loads((OUT / "m3_decision.json").read_text())["decision"] == "RM_MASK_HUMAN_REFERENCE_REQUIRED"
    assert json.loads((OUT / "final_decision.json").read_text())["decision"] == "RM_MASK_HUMAN_REFERENCE_REQUIRED"
    rows = [json.loads(line) for line in (PACKAGE / "annotation_manifest.jsonl").read_text().splitlines() if line]
    assert len(rows) == 200 and len({row["task_id"] for row in rows}) == 10 and len({row["episode_id"] for row in rows}) == 20
    proposal = json.loads((PACKAGE / "sam3_proposal_status.json").read_text())
    assert proposal["status"] == "completed_proposal_only" and proposal["count"] == 200 and len(proposal["records"]) == 200
    assert len(list((PACKAGE / "sam3_proposals").glob("*.png"))) == 200
    assert len(list((PACKAGE / "sam3_previews").glob("*.jpg"))) == 200
    validation = json.loads((PACKAGE / "validation_report.json").read_text())
    assert validation["status"] == "annotation_pending_or_invalid" and validation["missing_masks"] == 800
    candidates = json.loads((OUT / "automatic_mask_candidate_report.json").read_text())
    assert candidates["pilot_actual"]["automatic_candidate_evaluations"] == 0
    assert len(list((OUT / "automatic_mask_contact_sheets").glob("*.jpg"))) == 10
    for name in ("final_report.md", "mask_quality_report.md", "target_quality_report.md", "baseline_summary.json", "raw_prediction_manifest.json", "reproducibility_manifest.json", "claim_boundary.md", "paper_positioning.md", "next_steps.md"):
        assert (OUT / name).is_file(), name
    print("RM-MASK-ACD terminal package: 15 checks passed")


if __name__ == "__main__":
    main()
