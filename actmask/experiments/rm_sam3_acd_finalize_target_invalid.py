#!/usr/bin/env python3
"""Finalize RM-SAM3-ACD-PILOT when frozen pseudo-target quality gates fail."""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "actmask" / "rm_sam3_acd_pilot"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(value)


def package_version(name: str) -> str | None:
    try: return version(name)
    except PackageNotFoundError: return None


def append_log(event: str, **payload: Any) -> None:
    path = OUT / "run_log.jsonl"; rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    previous, sequence = rows[-1]["event_sha256"], int(rows[-1]["seq"])
    row = {"seq": sequence + 1, "utc": datetime.now(timezone.utc).isoformat(), "event": event, "payload": payload, "prev_event_sha256": previous}
    row["event_sha256"] = hashlib.sha256(json.dumps(row, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    with path.open("a") as handle: handle.write(json.dumps(row, sort_keys=True) + "\n")


def run() -> dict[str, Any]:
    config = OUT / "preregistered_config.json"
    config_hash = sha256(config)
    if config_hash != (OUT / "preregistered_config.sha256").read_text().strip(): raise RuntimeError("SAM3_PSEUDOMASK config hash mismatch")
    screening, stats = json.loads((OUT / "pseudomask_screening_audit.json").read_text()), json.loads((OUT / "target_statistics_sam3.json").read_text())
    if screening["decision"] != "SAM3_PSEUDOMASK_PILOT_ACCEPTED" or stats["decision"] != "SAM3_PSEUDOMASK_TARGET_INVALID": raise RuntimeError("this finalizer requires accepted screening and invalid frozen pseudo-targets")
    targets = [json.loads(line) for line in (OUT / "target_manifest_sam3.jsonl").read_text().splitlines() if line]
    if len(targets) != 2400 or len({row["input_id"] for row in targets}) != 800: raise RuntimeError("unexpected SAM3_PSEUDOMASK target manifest coverage")
    violations = []
    for horizon, values in sorted(stats["per_horizon"].items()):
        if values["meaningful_window_fraction"] > 0.80: violations.append(f"{horizon}: meaningful-window fraction {values['meaningful_window_fraction']:.4f} exceeds frozen maximum 0.8000")
        if values["confidence_window_fraction_ge_0_60"] < 0.60: violations.append(f"{horizon}: confidence>=0.60 fraction {values['confidence_window_fraction_ge_0_60']:.4f} is below frozen minimum 0.6000")
        if values["static_background_border_area"] > 0.05: violations.append(f"{horizon}: static-border affected area {values['static_background_border_area']:.4f} exceeds frozen maximum 0.0500")
    baseline = {"schema": "SAM3_PSEUDOMASK_baseline_summary_v1", "status": "not_run", "reason": "P4 is gated on valid SAM3_PSEUDOMASK pseudo-targets. P3 decided SAM3_PSEUDOMASK_TARGET_INVALID.", "runs": []}
    dump(OUT / "baseline_summary.json", baseline)
    dump(OUT / "raw_prediction_manifest.json", {"schema": "SAM3_PSEUDOMASK_raw_prediction_manifest_v1", "status": "empty_by_design", "reason": "No baseline or method prediction may be produced after invalid frozen pseudo-target gates.", "predictions": []})
    text(OUT / "claim_boundary.md", """# SAM3_PSEUDOMASK claim boundary

This branch generated fixed SAM3_PSEUDOMASK robot-exclusion proposals and
exploratory future-change pseudo-targets. It does not establish segmentation
accuracy, labels, references, ground truth, a validated real-robot benchmark,
or a real-world prediction result. No learned baseline or proposed method ran.
""")
    text(OUT / "paper_positioning.md", """# Paper positioning

The defensible result is a negative audit: a fixed SAM3_PSEUDOMASK exclusion
avoids a broad-table mask proxy in the screen, yet the derived RGB change target
is too diffuse, insufficiently confident at longer horizons, and has too much
static-border activity. This does not support an ActMask/ACD method claim.
""")
    text(OUT / "next_steps.md", """# Next steps

Do not tune the completed branch by downstream score. A future branch would need
a separately preregistered target redesign—for example, a target that explicitly
models camera/background stability or uses independently audited semantic
change—then rerun P0-P3 under a new output root. It must still call SAM3 output
`SAM3_PSEUDOMASK` and must not call it ground truth.
""")
    text(OUT / "final_report.md", """# RM-SAM3-ACD-PILOT final report

## Decision

`SAM3_PSEUDOMASK_TARGET_INVALID`

The fixed SAM3_PSEUDOMASK screen completed over 2,967 frozen target-window
frames. Its 100-frame/10-task/20-trajectory descriptive audit had no empty
unions or broad-workspace area proxies, so it was admitted only for exploratory
pseudo-target construction. This is not a segmentation-quality result.

P3 then constructed 800 fair inputs and 2,400 future-derived pseudo-targets.
All horizons failed the frozen target-quality gates: targets were meaningful in
nearly every window, confidence fell below the required level beyond short
horizons, and static-border activity exceeded its limit. Therefore P4 baselines
and P5 method training were correctly not run.
""")
    final = {"schema": "SAM3_PSEUDOMASK_final_decision_v1", "decision": "SAM3_PSEUDOMASK_TARGET_INVALID", "terminal": True, "config_sha256": config_hash, "evidence": {"p1": {"windows": 2967, "mask_manifest": "pseudomask_manifest.jsonl"}, "p2": screening["decision"], "p3": stats["decision"], "p4_p5": "not run by target-quality gate"}, "violations": violations, "prohibitions_honored": ["SAM3_PSEUDOMASK was never a fair predictive input.", "No SAM3 mask was called label/reference/ground truth/verified segmentation.", "No frozen prior RM/RM-OD/RM-ACD/RM-MASK-ACD artifact was modified.", "No baseline or method trained after invalid pseudo-target decision."]}
    dump(OUT / "final_decision.json", final)
    scripts = ("rm_sam3_acd_freeze.py", "rm_sam3_acd_generate_pseudomasks.py", "rm_sam3_acd_screen_pseudomasks.py", "rm_sam3_acd_freeze_pseudotarget_protocol.py", "rm_sam3_acd_construct_pseudotargets.py", "rm_sam3_acd_finalize_target_invalid.py")
    reproducibility = {"schema": "SAM3_PSEUDOMASK_reproducibility_manifest_v1", "config_sha256": config_hash, "source_hash_manifest_sha256": sha256(OUT / "source_hash_manifest.json"), "pseudomask_manifest_sha256": sha256(OUT / "pseudomask_manifest.jsonl"), "pseudotarget_manifest_sha256": sha256(OUT / "target_manifest_sam3.jsonl"), "target_protocol_sha256": sha256(OUT / "pseudotarget_preregistered_protocol.json"), "scripts": {script: sha256(ROOT / "actmask" / "experiments" / script) for script in scripts}, "environment": {"python": sys.version, "platform": platform.platform(), "dependencies": {name: package_version(name) for name in ("torch", "torchvision", "timm", "numpy", "opencv-python", "Pillow")}}, "counts": {"pseudomask_windows": 2967, "screening_frames": 100, "pseudo_inputs": 800, "pseudo_targets": 2400, "contact_sheets": 20}, "commands": ["/home/tzh/conda_envs/actmask/bin/python -m actmask.experiments.rm_sam3_acd_freeze", "/home/tzh/conda_envs/actmask/bin/python -m actmask.experiments.rm_sam3_acd_generate_pseudomasks", "/home/tzh/conda_envs/actmask/bin/python -m actmask.experiments.rm_sam3_acd_screen_pseudomasks", "/home/tzh/conda_envs/actmask/bin/python -m actmask.experiments.rm_sam3_acd_construct_pseudotargets"]}
    dump(OUT / "reproducibility_manifest.json", reproducibility)
    docs = OUT / "docs"
    text(docs / "results.md", "# RM-SAM3-ACD-PILOT results\n\nTerminal decision: `SAM3_PSEUDOMASK_TARGET_INVALID`. SAM3 pseudo-masks were screened descriptively, then 2,400 pseudo-targets failed frozen quality gates. No learned baseline or method ran.\n")
    text(docs / "handoff.md", "# RM-SAM3-ACD-PILOT handoff\n\nRead `final_decision.json` and `target_quality_report.md`. Do not revive this branch by threshold tuning or training. Any repair requires a new separately preregistered target-design branch.\n")
    append_log("FINAL_SAM3_PSEUDOMASK_TARGET_INVALID", targets=2400, inputs=800, violations=len(violations), config_sha256=config_hash)
    return final


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, sort_keys=True))
