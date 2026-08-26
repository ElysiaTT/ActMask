"""Strict v2 gate: wraps 4R storage integrity with candidate-diversity gates."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from actmask.experiments.milestone4r_audit import run as storage_audit


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def run(root: str | Path) -> dict:
    root = Path(root)
    storage = storage_audit(root)
    result = {"schema": "milestone4r-v2-full-audit-v1", "storage_audit": storage, "tasks": {}, "passed": True}
    for task, prior in storage["tasks"].items():
        rows = _rows(root / f"{task}_candidates.jsonl")
        labels = np.load(root / f"{task}_labels.npz")["success"].astype(bool)
        by_history: dict[int, list[bool]] = defaultdict(list)
        by_slot: dict[int, list[bool]] = defaultdict(list)
        for row, label in zip(rows, labels):
            by_history[row["history_ref"]].append(bool(label))
            by_slot[row["candidate_slot"]].append(bool(label))
        counts = [sum(values) for values in by_history.values()]
        rates = {str(slot): float(np.mean(values)) for slot, values in sorted(by_slot.items())}
        index_accuracy = float(np.mean([max(rate, 1.0-rate) for rate in rates.values()]))
        strict = {
            "mixed_success_fraction": float(np.mean([0 < count < 10 for count in counts])),
            "mixed_count_2_to_8_fraction": float(np.mean([2 <= count <= 8 for count in counts])),
            "success_count_histogram": {str(value): int(counts.count(value)) for value in sorted(set(counts))},
            "candidate_template_success_rates": rates,
            "candidate_template_success_span": max(rates.values()) - min(rates.values()),
            "candidate_index_only_majority_accuracy": index_accuracy,
            "pair_label_disagreement": prior["pair_label_disagreement"],
            "pair_actions_exact": prior["pair_matching"].get("action") == 1.0,
            "matching_camera_appearance_association": prior["passed"] or all([
                prior["schema"]["bundle_shape"], prior["schema"]["visibility_confidence"], prior["schema"]["no_segmentation_in_bundle"], prior["schema"]["forbidden_candidate_fields_absent"], prior["condition_coverage"], all(prior["asynchronous_masks"].values()), prior["calibration"]["passed"], prior["association"]["passed"], prior["appearance"]["world_randomization_observed"], prior["appearance"]["metadata_excluded"]
            ])
        }
        strict["passed"] = bool(
            strict["mixed_success_fraction"] >= .8 and strict["mixed_count_2_to_8_fraction"] >= .8
            and strict["candidate_template_success_span"] <= .30 and strict["candidate_index_only_majority_accuracy"] <= .70
            and strict["pair_label_disagreement"] >= .5 and strict["pair_actions_exact"] and strict["matching_camera_appearance_association"]
        )
        result["tasks"][task] = strict
        result["passed"] &= strict["passed"]
    (root / "milestone4r_v2_full_audit.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


if __name__ == "__main__":
    project = Path(__file__).resolve().parents[2]
    print(json.dumps(run(project / "outputs" / "actmask" / "milestone4r_v2_candidate_diversity" / "full_probe"), indent=2, sort_keys=True))
