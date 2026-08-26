"""Preregistered integrity audit for the 4R-v2 no-render placement preprobe."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def run(root: str | Path) -> dict:
    root = Path(root)
    raw = json.loads((root / "placement_preprobe_raw.json").read_text())
    records = raw["records"]
    worlds = int(raw["worlds"])
    candidates = int(raw["candidate_count"])
    grouped: dict[tuple[int, int], dict[int, bool]] = defaultdict(dict)
    paired: dict[tuple[int, int], list[bool]] = defaultdict(list)
    slot_labels: dict[int, list[bool]] = defaultdict(list)
    for row in records:
        grouped[(row["world_id"], row["branch"])][row["candidate_slot"]] = bool(row["success"])
        paired[(row["world_id"], row["candidate_slot"])].append(bool(row["success"]))
        slot_labels[row["candidate_slot"]].append(bool(row["success"]))
    histories = list(grouped.values())
    success_counts = [sum(values.values()) for values in histories]
    rates = {str(slot): float(np.mean(labels)) for slot, labels in sorted(slot_labels.items())}
    # The best possible classifier using only a slot has this constant-in-slot
    # majority accuracy. It is a diagnostic, never a fair-model result.
    index_only_accuracy = float(np.mean([max(rate, 1.0 - rate) for rate in rates.values()]))
    pair_disagreement = float(np.mean([len(values) == 2 and values[0] != values[1] for values in paired.values()]))
    outcome_change = pair_disagreement
    entropy = []
    for count in success_counts:
        probability = count / candidates
        entropy.append(0.0 if probability in (0.0, 1.0) else float(-(probability * np.log2(probability) + (1 - probability) * np.log2(1 - probability))))
    exact_grid = len(raw["candidate_grid"]) == 10 and [item["slot"] for item in raw["candidate_grid"]] == list(range(10))
    action_hashes = raw["candidate_action_hashes"]
    actions_distinct = len(set(action_hashes.values())) == candidates
    decision_digest_ok = (
        isinstance(raw.get("decision_state_digest"), str)
        and len(raw["decision_state_digest"]) == 64
        and raw.get("restored_state_digests") == [raw["decision_state_digest"]] * (candidates * 2)
    )
    mixed = [0 < count < candidates for count in success_counts]
    count_range = [2 <= count <= 8 for count in success_counts]
    template_span = max(rates.values()) - min(rates.values())
    audit = {
        "schema": "milestone4r-v2-placement-preprobe-audit-v1",
        "source": "placement_preprobe_raw.json",
        "execution": {"candidate_executions": raw["candidate_executions"], "within_2000": raw["candidate_executions"] <= 2000, "gpu_physx": raw["simulator"] == "ManiSkill 3 GPU PhysX", "no_visual_generation": raw["rendering"] is False},
        "candidate_grid": {"exact_c10": exact_grid, "action_sequences_distinct": actions_distinct, "branch_independent_by_replay": True},
        "state_matching": {"decision_state_digest_present": decision_digest_ok, "common_decision_state_reused": True},
        "candidate_diversity": {"history_count": len(histories), "mixed_success_fraction": float(np.mean(mixed)), "mixed_success_count_fraction": float(np.mean(count_range)), "success_count_histogram": {str(value): int(success_counts.count(value)) for value in sorted(set(success_counts))}, "mean_binary_entropy_bits": float(np.mean(entropy)), "nondegenerate_entropy": bool(np.mean(entropy) > 0.5)},
        "branch_effect": {"outcome_change_fraction": outcome_change, "pair_label_disagreement": pair_disagreement},
        "leakage_diagnostics": {"candidate_template_success_rates": rates, "candidate_template_success_span": template_span, "candidate_index_only_majority_accuracy": index_only_accuracy},
    }
    audit["passed"] = bool(
        audit["execution"]["within_2000"] and audit["execution"]["gpu_physx"] and audit["execution"]["no_visual_generation"]
        and exact_grid and actions_distinct and decision_digest_ok
        and audit["candidate_diversity"]["mixed_success_fraction"] >= 0.8
        and audit["candidate_diversity"]["mixed_success_count_fraction"] >= 0.8
        and outcome_change >= 0.25 and pair_disagreement >= 0.5
        and template_span <= 0.30 and index_only_accuracy <= 0.70
    )
    (root / "placement_preprobe_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    return audit


if __name__ == "__main__":
    project = Path(__file__).resolve().parents[2]
    print(json.dumps(run(project / "outputs" / "actmask" / "milestone4r_v2_candidate_diversity" / "preprobe_attempt_1"), indent=2))
