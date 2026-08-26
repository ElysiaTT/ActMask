#!/usr/bin/env python3
"""Validate completed RM-MASK-ACD human labels and freeze their provenance.

This is intentionally a gate, rather than a target-builder: target construction
is forbidden until the independent reference itself is complete and reviewed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "actmask" / "rm_mask_acd_verified"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    args = parser.parse_args()
    package = args.package.resolve()
    validator = package / "validate_annotations.py"
    if not validator.is_file():
        raise SystemExit(f"missing annotation validator: {validator}")
    subprocess.run([sys.executable, str(validator), "--require-complete"], check=True)
    report = json.loads((package / "validation_report.json").read_text())
    if report.get("status") != "complete" or report.get("frames") != 200:
        raise SystemExit("annotation validation did not prove a complete 200-frame reference")
    rows = [json.loads(line) for line in (package / "annotation_manifest.jsonl").read_text().splitlines() if line]
    files = [package / row["frame_path"] for row in rows]
    files += [package / rel for row in rows for rel in row["required_labels"].values()]
    manifest = [{"path": str(path.relative_to(package)), "sha256": sha256(path), "bytes": path.stat().st_size} for path in files]
    output = {
        "schema": "rm-mask-acd-human-reference-ready-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "package": str(package),
        "annotation_validation": report,
        "files": manifest,
        "next_gate": "M4 mask acceptance against an episode/task-disjoint held-out partition of this reference",
        "prohibition": "This command does not construct targets or train models.",
    }
    dump(OUT / "m3_human_reference_ready.json", output)
    print(json.dumps({"status": "M3_HUMAN_REFERENCE_READY", "frames": len(rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
