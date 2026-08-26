"""Pre-freeze development-seed validation for the v2.1 protocol."""

from __future__ import annotations

import json

from actmask.experiments.history_action_binding_v2_1.audit import evaluate_all, symbolic_gate_checks
from actmask.experiments.history_action_binding_v2_1.common import OUTPUT_ROOT, PROTOCOL, TASKS, write_json
from actmask.experiments.history_action_binding_v2_1.symbolic import candidate_sampler, symbolic_dataset


def run_development_validation() -> dict:
    if (OUTPUT_ROOT / "freeze_record.json").exists():
        raise RuntimeError("development evaluation is forbidden after freeze_record.json exists")
    tasks = {}
    for task in TASKS:
        seed_rows = []
        for seed in PROTOCOL["development_seeds"]:
            dataset = symbolic_dataset(
                task,
                twins=int(PROTOCOL["pilot_twins_per_task"]),
                seed=int(seed),
                phase="development",
            )
            _, metric_sets, _ = evaluate_all(dataset)
            checks, details = symbolic_gate_checks(dataset, metric_sets, candidate_sampler)
            seed_rows.append({
                "seed": int(seed),
                "passed": bool(all(checks.values())),
                "checks": checks,
                "fourier_score": details["fourier_score"],
                "oracle_score": details["oracle_score"],
                "nearest_score": details["interaction_scores"]["nearest_historical_action"],
                "knn_score": details["interaction_scores"]["knn_action_result_lookup"],
                "arrow_score": details["interaction_scores"]["arrow_action_compatibility"],
                "binding_drop": details["binding_drop"],
                "pair_preserving_change": details["pair_preserving_change"],
                "slot_min": details["slot_min"],
                "slot_max": details["slot_max"],
                "slot_permutation_score": details["slot_permutation_score"],
                "coordinate_rotation_score": details["coordinate_rotation_score"],
            })
        tasks[task] = {
            "all_development_seeds_passed": bool(all(row["passed"] for row in seed_rows)),
            "seeds": seed_rows,
            "score_ranges": {
                field: {
                    "min": min(row[field] for row in seed_rows),
                    "max": max(row[field] for row in seed_rows),
                }
                for field in ("fourier_score", "oracle_score", "nearest_score", "knn_score", "arrow_score", "binding_drop")
            },
        }
    result = {
        "status": "DEVELOPMENT_VALIDATION_COMPLETE",
        "development_seeds": list(PROTOCOL["development_seeds"]),
        "official_seed_used": False,
        "official_preflight_started": False,
        "task_a_repair_selected": {
            "probe_magnitudes": PROTOCOL["task_a_probe_magnitudes"],
            "candidate_directions_degrees": PROTOCOL["task_a_candidate_directions_degrees"],
            "rationale": (
                "A second probe radius removes the rank deficiency; restricting candidates to 10..80 degrees in the public "
                "quadrant removes the clipped 105/285-degree utility ties while keeping every candidate angle and magnitude novel."
            ),
        },
        "task_b_policy": "Equation, probes, candidate geometry, continuous parameter ranges, and thresholds unchanged from v2; only cyclic slots changed.",
        "tasks": tasks,
    }
    write_json(OUTPUT_ROOT / "development" / "development_validation.json", result)
    return result


def main() -> None:
    print(json.dumps(run_development_validation(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
