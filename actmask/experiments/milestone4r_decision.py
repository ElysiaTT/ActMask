"""Frozen-gate aggregation for the 4R benchmark/headroom decision."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


PRIMARY = "combined_held_dropout"
ORDERED = ("ordered_rgbd_gru", "world_pointcloud_gru", "temporal_convolution", "masked_history_gru", "dropout_augmented_gru")
UNORDERED = ("unordered_rgbd", "unordered_pointcloud")
ASSOCIATION = ("association_none", "association_nearest_neighbour", "association_soft_geometric")


def _score(report: dict, family: str, name: str, regime: str) -> float:
    return float(report[family][name].get(regime, {}).get("pair_order_accuracy", float("nan")))


def _best(report: dict, family: str, names: tuple[str, ...], regime: str) -> tuple[str, float]:
    values = [(name, _score(report, family, name, regime)) for name in names]
    return max(values, key=lambda item: np.nan_to_num(item[1], nan=-np.inf))


def run(root: str | Path) -> dict:
    root = Path(root)
    audit = json.loads((root / "robust_audit.json").read_text())
    if not audit["passed"]:
        failed = {
            task: {
                "passed": value["passed"],
                "candidate_diversity": value["candidate_diversity"],
                "pair_matching": value["pair_matching"],
            }
            for task, value in audit["tasks"].items()
            if not value["passed"]
        }
        result = {
            "schema": "milestone4r-final-decision-v1",
            "decision": "E. BENCHMARK INVALID",
            "milestone4m_authorized": False,
            "reason": "A preregistered integrity audit failed; standard-model results would not be scientifically interpretable.",
            "failed_audits": failed,
            "integrity_audit": False,
            "baseline_evaluation_skipped": True,
        }
        (root / "headroom_metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        (root / "milestone4r_final_decision.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        return result
    baseline = json.loads((root / "standard_baseline_report.json").read_text())
    tasks = {}
    gates = []
    for task, report in baseline["tasks"].items():
        best_ordered_name, best_ordered = _best(report, "fair", ORDERED, PRIMARY)
        best_unordered_name, best_unordered = _best(report, "fair", UNORDERED, PRIMARY)
        state_oracle = _score(report, "oracle", "clean_state_history_gru", PRIMARY)
        action_only = _score(report, "fair", "action_only", PRIMARY)
        current_rgbd = _score(report, "fair", "current_rgbd", PRIMARY)
        pc = _score(report, "fair", "world_pointcloud_gru", PRIMARY)
        rgbd = _score(report, "fair", "ordered_rgbd_gru", PRIMARY)
        association_oracle = _score(report, "oracle", "oracle_segmentation_pointcloud_gru", PRIMARY)
        association_name, association_fair = _best(report, "association", ASSOCIATION, PRIMARY)
        id_score = _score(report, "fair", best_ordered_name, "multi_camera_id")
        held_score = _score(report, "fair", best_ordered_name, "held_camera_ood")
        complete_score = _score(report, "fair", best_ordered_name, "complete_history")
        partial_score = _score(report, "fair", best_ordered_name, PRIMARY)
        metrics = {
            "state_oracle": state_oracle,
            "action_only": action_only,
            "current_rgbd": current_rgbd,
            "best_unordered_name": best_unordered_name,
            "best_unordered": best_unordered,
            "best_ordered_name": best_ordered_name,
            "best_ordered": best_ordered,
            "ordered_temporal_advantage": best_ordered - best_unordered,
            "geometric_normalization_advantage": pc - rgbd,
            "association_oracle": association_oracle,
            "best_association_name": association_name,
            "best_fair_association": association_fair,
            "association_gap": association_oracle - association_fair,
            "cross_view_retention": held_score / id_score if id_score > 0 else float("nan"),
            "partial_observation_retention": partial_score / complete_score if complete_score > 0 else float("nan"),
            "held_camera_degradation": id_score - held_score,
            "partial_observation_degradation": id_score - partial_score,
            "model_headroom": state_oracle - best_ordered,
            "primary_regime": PRIMARY,
        }
        tasks[task] = metrics
        gates.append(metrics)
    average = {key: float(np.nanmean([task[key] for task in gates])) for key in gates[0] if isinstance(gates[0][key], (float, int))}
    accepted = {
        "state_oracle_min": min(task["state_oracle"] for task in gates) >= .90,
        "static_action_max": max(max(task["action_only"], task["current_rgbd"]) for task in gates) <= .60,
        "unordered_max": max(task["best_unordered"] for task in gates) <= .65,
        "ordered_primary_range": all(.60 <= task["best_ordered"] <= .88 for task in gates),
        "ordered_advantage": min(task["ordered_temporal_advantage"] for task in gates) >= .10,
        "headroom_two_families": sum(task["model_headroom"] >= .08 for task in gates) >= 2,
        "held_and_partial_drop": all(task["held_camera_degradation"] >= .10 and task["partial_observation_degradation"] >= .10 for task in gates),
        "geometry_nonclosing": all(task["geometric_normalization_advantage"] > 0 and task["model_headroom"] >= .08 for task in gates),
        "association_nontrivial": all(task["association_gap"] > 0 for task in gates),
        "candidate_ranking_nondegenerate": all(audit["tasks"][task]["candidate_diversity"]["mixed_success_fraction"] > 0 for task in tasks),
        "integrity": audit["passed"] is True,
    }
    if not accepted["integrity"]:
        decision = "E. BENCHMARK INVALID"
    elif not accepted["static_action_max"] or not accepted["unordered_max"]:
        decision = "D. UNORDERED OR CAMERA SHORTCUT REMAINS"
    elif any(task["best_ordered"] > .88 for task in gates):
        decision = "B. ROBUST VISUAL BENCHMARK STILL TOO EASY"
    elif all(task["best_ordered"] < .60 for task in gates) and accepted["state_oracle_min"]:
        decision = "C. PERCEPTION/ASSOCIATION BOTTLENECK TOO SEVERE"
    elif all(accepted.values()):
        decision = "A. GO TO MILESTONE 4M"
    else:
        decision = "F. HUMAN DECISION REQUIRED"
    result = {"schema": "milestone4r-final-decision-v1", "decision": decision, "milestone4m_authorized": decision.startswith("A."), "primary_regime": PRIMARY, "tasks": tasks, "mean": average, "gates": accepted, "integrity_audit": audit["passed"]}
    (root / "headroom_metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (root / "milestone4r_final_decision.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


if __name__ == "__main__":
    project = Path(__file__).resolve().parents[2]
    print(json.dumps(run(project / "outputs" / "actmask" / "milestone4r_robust_visual" / "robust_probe_v1"), indent=2, sort_keys=True))
