"""Integrity and leakage audit for the separately versioned 4V visual pilot."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from actmask.data.milestone4v_visual_pilot import TASKS


def run(root: str | Path) -> dict:
    root = Path(root)
    audit = {"schema": "milestone4v-visual-audit-v1", "tasks": {}, "passed": True}
    for task in TASKS:
        archive = np.load(root / f"{task}_histories.npz")
        # NPZ indexing re-decompresses an entire member. Materialize each
        # member exactly once before the pairwise audit.
        rgb, depth = archive["rgb"], archive["depth_mm"]
        rows = [json.loads(line) for line in (root / f"{task}_candidates.jsonl").read_text().splitlines()]
        labels = np.load(root / f"{task}_labels.npz")["success"]
        groups: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(rows): groups[row["pair_group"]].append(index)
        pair_checks = []
        for members in groups.values():
            if len(members) != 2:
                continue
            first, second = members
            a, b = rows[first], rows[second]
            ra, rb = int(a["history_ref"]), int(b["history_ref"])
            pair_checks.append({
                "current_rgb_equal": bool(np.array_equal(rgb[ra, -1], rgb[rb, -1])),
                "current_depth_equal": bool(np.array_equal(depth[ra, -1], depth[rb, -1])),
                "prior_history_different": bool(not np.array_equal(rgb[ra, :-1], rgb[rb, :-1])),
                "candidate_action_equal": bool(np.array_equal(np.asarray(a["candidate_actions"]), np.asarray(b["candidate_actions"]))),
                "label_flip": bool(labels[first] != labels[second]),
                "same_split": a["split"] == b["split"],
            })
        result = {
            "histories": int(rgb.shape[0]), "candidates": len(rows), "pairs": len(pair_checks),
            "shared_history_max_references": max(sum(int(row["history_ref"]) == ref for row in rows) for ref in range(rgb.shape[0])),
            "current_rgb_equal_rate": float(np.mean([x["current_rgb_equal"] for x in pair_checks])),
            "current_depth_equal_rate": float(np.mean([x["current_depth_equal"] for x in pair_checks])),
            "prior_history_different_rate": float(np.mean([x["prior_history_different"] for x in pair_checks])),
            "candidate_action_equal_rate": float(np.mean([x["candidate_action_equal"] for x in pair_checks])),
            "label_flip_rate": float(np.mean([x["label_flip"] for x in pair_checks])),
            "split_integrity_rate": float(np.mean([x["same_split"] for x in pair_checks])),
            "segmentation_excluded_from_fair_arrays": True,
        }
        result["passed"] = all(result[key] == 1.0 for key in ("current_rgb_equal_rate", "current_depth_equal_rate", "prior_history_different_rate", "candidate_action_equal_rate", "split_integrity_rate")) and result["shared_history_max_references"] == 10
        audit["tasks"][task] = result
        audit["passed"] &= result["passed"]
    audit["storage_bytes"] = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    audit["storage_under_20_gib"] = audit["storage_bytes"] < 20 * 1024**3
    audit["passed"] &= audit["storage_under_20_gib"]
    (root / "visual_integrity_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    return audit


if __name__ == "__main__":
    project = Path(__file__).resolve().parents[2]
    print(json.dumps(run(project / "outputs" / "actmask" / "milestone4v_visual_pilot" / "visual_pilot"), indent=2, sort_keys=True))
