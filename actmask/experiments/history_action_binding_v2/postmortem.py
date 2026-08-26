"""Read-only, quantitative postmortem of the frozen v1 evidence."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from actmask.experiments.history_action_binding_v2.common import OUTPUT_ROOT, V1_ROOT, sha256_file, write_json


def build_postmortem(output: Path | None = None) -> dict:
    output = OUTPUT_ROOT / "v1_failure_postmortem.json" if output is None else Path(output)
    summary_path = V1_ROOT / "summary.json"
    summary = json.loads(summary_path.read_text())
    evidence = {}
    for task in ("task_a_discrete_mode", "task_b_continuous_response"):
        audit = summary["tasks"][task]
        with np.load(V1_ROOT / "pilot" / task / "dataset.npz", allow_pickle=False) as archive:
            utility = archive["utility"]
            success = archive["success"]
            probe_action = archive["probe_action"]
            probe_delta = archive["probe_delta"][..., 0]
            first_moment = np.mean(probe_action * probe_delta, axis=2)
            mean_utility = utility.mean(axis=2)
            result_scale = np.mean(np.abs(probe_delta), axis=2)
            branch_ease_difference = mean_utility[:, 1] - mean_utility[:, 0]
            top_crossing = np.mean(np.argmax(utility[:, 0], axis=1) != np.argmax(utility[:, 1], axis=1))
        metrics = audit["metrics"]["original"]
        evidence[task] = {
            "shortcut_saturated_baselines": audit["shortcut_saturated_baselines"],
            "arrow_first_moment_mean_by_branch": np.mean(first_moment, axis=0).tolist(),
            "arrow_first_moment_mean_abs": float(np.mean(np.abs(first_moment))),
            "branch_mean_utility_difference_mean": float(np.mean(branch_ease_difference)),
            "branch_mean_utility_difference_mean_abs": float(np.mean(np.abs(branch_ease_difference))),
            "probe_result_scale_branch_difference_mean_abs": float(np.mean(np.abs(result_scale[:, 1] - result_scale[:, 0]))),
            "candidate_top1_crossing_rate": float(top_crossing),
            "metrics": {name: {
                "raw_twin_preference_accuracy": metrics[name]["twin_preference_accuracy"],
                "ndcg": metrics[name]["ndcg"], "spearman": metrics[name]["spearman"],
            } for name in ("action_only", "result_only", "arrow_action_compatibility", "repeat_last_failed_action", "nearest_historical_action", "knn_action_result_lookup")},
        }
    report = {
        "v1_summary_sha256": sha256_file(summary_path), "v1_gate": summary["gate"],
        "learned_training_was_authorized": summary["learned_training_authorized"],
        "findings": {
            "task_a_arrow": "The signed first moment mean(action * displacement) changes with the hidden mode, so ArrowActionCompatibility is a sufficient branch decoder.",
            "task_a_nearest_knn": "The scalar response is smooth and candidate magnitudes lie close to signed probes; nearest/kNN preserve the same one-bit direction ordering and reach near-perfect or perfect ranking.",
            "task_b_action_only": "Both branches share almost the same magnitude-driven candidate ordering, so action-only reaches high NDCG without predicting which response law applies.",
            "task_b_result_only": "Probe response scale covaries with branch-average candidate utility; a branch-constant score therefore obtains raw Twin Preference Accuracy even though it cannot rank candidates.",
            "metric_confound": "v1 raw Twin Preference compares uncentered branch utility. It mixes branch-wide ease with candidate-specific compatibility, while NDCG can reward a common action-magnitude ordering shared by both twins.",
        },
        "numeric_evidence": evidence,
        "v2_repairs": [
            "exact probe action and result-multiset matching", "random public coordinate frame and balanced branch swaps",
            "novel candidate directions/magnitudes", "branch-centered candidate-conditioned twin preference",
            "second-harmonic Fourier SysID versus first-moment Arrow",
        ],
    }
    write_json(output, report)
    return report


if __name__ == "__main__":
    print(json.dumps(build_postmortem(), indent=2, sort_keys=True))

