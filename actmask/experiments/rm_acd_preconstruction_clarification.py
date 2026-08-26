"""One pre-target clarification for a legal long-horizon anchor schedule."""
from __future__ import annotations

import json

from actmask.experiments.rm_acd_freeze_and_modality import OUT, _configuration, append_log, sha, write_json


def run() -> dict:
    if (OUT / "target_manifest.jsonl").exists() or (OUT / "processed_subset" / "targets").exists():
        raise RuntimeError("target construction has started; frozen configuration cannot be clarified")
    selected = json.loads((OUT / "selected_trajectory_manifest.json").read_text())["episodes"]
    hashes = json.loads((OUT / "source_hash_manifest.json").read_text())["files"]
    expected = _configuration(selected, hashes)
    path = OUT / "preregistered_config.json"
    previous = json.loads(path.read_text())
    if previous != expected:
        old = previous.get("temporal", {})
        if "anchor_policy" in old or "meaningful_affected_area_fraction_min" in previous.get("target", {}):
            raise RuntimeError("unexpected frozen configuration mismatch")
        write_json(path, expected)
        (OUT / "preregistered_config.sha256").write_text(sha(path) + "\n")
        append_log("phase_a0_pre_target_anchor_clarification", reason="the shortest 108-frame source episode makes clamped nominal fractions collide at the one-second horizon; before any target or model result, freeze eight unique anchors uniformly over the legal interval", config_sha256=sha(path))
    return {"config_sha256": sha(path), "clarified_before_target_construction": True}


if __name__ == "__main__":
    print(json.dumps(run(), sort_keys=True))
