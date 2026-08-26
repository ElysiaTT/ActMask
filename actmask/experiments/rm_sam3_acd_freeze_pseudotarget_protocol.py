#!/usr/bin/env python3
"""Freeze target-construction settings before any SAM3 pseudo-target is built."""
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
    config_hash = sha256(OUT / "preregistered_config.json")
    if config_hash != (OUT / "preregistered_config.sha256").read_text().strip():
        raise RuntimeError("SAM3_PSEUDOMASK parent config hash mismatch")
    protocol = {
        "schema": "SAM3_PSEUDOMASK_target_protocol_v1",
        "parent_config_sha256": config_hash,
        "target_resolution": [160, 120],
        "construction": {"flow": "OpenCV Farneback forward/backward consistency", "flow_magnitude_threshold_px": 0.75, "residual_threshold_0_1": 0.10, "forward_backward_error_strict_px": 1.25, "forward_backward_error_residual_px": 2.0, "minimum_component_pixels": 12, "brightness_ratio_clip": [0.85, 1.18], "gradient_support_percentile": 30},
        "robot_exclusion": "nearest-neighbor resize and union of SAM3_PSEUDOMASK at pre frame and future frame; used only while constructing/evaluating the future-derived target", 
        "target_gates": {"meaningful_area_fraction_threshold": 0.003, "meaningful_window_fraction": [0.20, 0.80], "confidence_window_fraction_min": 0.60, "static_border_affected_area_max": 0.05, "robot_raw_overlap_fraction_max": 0.50, "action_or_progress_absolute_correlation_max": 0.90, "minimum_task_diversity": 10},
        "prohibition": "No parameter here may be selected or changed using pseudo-target, baseline, or method performance. SAM3_PSEUDOMASK is neither label nor fair model input.",
    }
    path = OUT / "pseudotarget_preregistered_protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise RuntimeError("refusing to alter frozen SAM3_PSEUDOMASK target protocol")
    path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    (OUT / "pseudotarget_preregistered_protocol.sha256").write_text(sha256(path) + "\n")
    print(json.dumps({"schema": protocol["schema"], "sha256": sha256(path)}))


if __name__ == "__main__":
    main()
