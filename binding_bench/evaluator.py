"""Independent evaluator for a complete intervention suite."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .adapters import load_method_predictions
from .interventions import CASES, load_suite
from .metrics import hierarchical_bootstrap_difference, per_seed_metrics, score_prediction


DEFAULT_GATES: dict[str, float | int] = {
    "minimum_seeds": 5,
    "minimum_twins_per_seed": 32,
    "relative_regret_improvement": 0.10,
    "seed_win_fraction": 0.80,
    "top1_improvement": 0.03,
    "binding_regret_increase": 0.05,
    "binding_tpa_drop": 0.05,
    "pair_regret_absolute_change": 0.005,
    "pair_tpa_absolute_change": 0.005,
    "twin_swap_regret_increase": 0.05,
    "slot_regret_absolute_change": 0.005,
    "anti_saturation_tpa_max": 0.95,
    "anti_saturation_regret_min": 0.01,
    "bootstrap_replicates": 1000,
}


def _load_methods(
    method_dirs: Mapping[str, str | Path],
    cases: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifests: dict[str, Any] = {}
    predictions: dict[str, Any] = {}
    for expected_id, directory in method_dirs.items():
        manifest, current = load_method_predictions(directory, cases)
        if manifest["method_id"] != expected_id:
            raise ValueError(f"method mapping key {expected_id!r} does not match its manifest")
        manifests[expected_id] = manifest
        predictions[expected_id] = current
    return manifests, predictions


def evaluate_suite(
    suite_path: str | Path,
    method_dirs: Mapping[str, str | Path],
    *,
    candidate_method: str,
    baseline_methods: tuple[str, ...] | list[str],
    gates: Mapping[str, float | int] | None = None,
    bootstrap_seed: int = 91073,
) -> dict[str, Any]:
    """Evaluate one split; model selection/tuning is deliberately out of scope."""

    configuration = dict(DEFAULT_GATES)
    if gates is not None:
        unknown = set(gates) - set(configuration)
        if unknown:
            raise ValueError(f"unknown gates: {sorted(unknown)}")
        configuration.update(gates)
    suite_manifest, cases = load_suite(suite_path)
    manifests, predictions = _load_methods(method_dirs, cases)
    baseline_methods = tuple(baseline_methods)
    if candidate_method not in predictions:
        raise ValueError("candidate method output is missing")
    if not baseline_methods or any(name not in predictions for name in baseline_methods):
        raise ValueError("every declared baseline must have a method output")
    if candidate_method in baseline_methods:
        raise ValueError("candidate method cannot also be an admission baseline")

    scores: dict[str, dict[str, Any]] = {}
    for method_id, method_predictions in predictions.items():
        scores[method_id] = {}
        for case in CASES:
            metric = score_prediction(cases[case], method_predictions[case])
            metric["per_seed"] = per_seed_metrics(cases[case], metric)
            scores[method_id][case] = metric

    original = cases["original"]
    candidate = scores[candidate_method]["original"]
    strongest_baseline = min(
        baseline_methods,
        key=lambda name: scores[name]["original"]["normalized_regret"],
    )
    baseline = scores[strongest_baseline]["original"]
    candidate_by_twin = np.asarray(candidate["normalized_regret_by_twin"])
    baseline_by_twin = np.asarray(baseline["normalized_regret_by_twin"])
    bootstrap = hierarchical_bootstrap_difference(
        original,
        baseline_by_twin,
        candidate_by_twin,
        replicates=int(configuration["bootstrap_replicates"]),
        seed=bootstrap_seed,
    )
    seeds = np.unique(original.seed_ids)
    twins_per_seed = [int(np.sum(original.seed_ids == seed)) for seed in seeds]
    seed_differences = {
        str(int(seed)): float(
            baseline["per_seed"][str(int(seed))]["normalized_regret"]
            - candidate["per_seed"][str(int(seed))]["normalized_regret"]
        )
        for seed in seeds
    }
    seed_wins = int(sum(value > 0.0 for value in seed_differences.values()))
    original_regret = float(candidate["normalized_regret"])
    original_tpa = float(candidate["tpa"])
    relative_improvement = (
        float(baseline["normalized_regret"]) - original_regret
    ) / max(float(baseline["normalized_regret"]), 1.0e-12)
    binding_regret_increase = (
        float(scores[candidate_method]["binding_breaking"]["normalized_regret"])
        - original_regret
    )
    binding_tpa_drop = original_tpa - float(scores[candidate_method]["binding_breaking"]["tpa"])
    pair_regret_change = abs(
        float(scores[candidate_method]["pair_preserving"]["normalized_regret"])
        - original_regret
    )
    pair_tpa_change = abs(float(scores[candidate_method]["pair_preserving"]["tpa"]) - original_tpa)
    twin_regret_increase = (
        float(scores[candidate_method]["twin_history_swap"]["normalized_regret"])
        - original_regret
    )
    slot_regret_change = abs(
        float(scores[candidate_method]["candidate_slot_permutation"]["normalized_regret"])
        - original_regret
    )
    admission_evaluable = (
        len(seeds) >= int(configuration["minimum_seeds"])
        and min(twins_per_seed) >= int(configuration["minimum_twins_per_seed"])
    )
    lower = float(bootstrap["lower_95"])
    checks = {
        "sample_size": admission_evaluable,
        "relative_regret_improvement": relative_improvement
        >= float(configuration["relative_regret_improvement"]),
        "seed_win_fraction": seed_wins
        >= math.ceil(float(configuration["seed_win_fraction"]) * len(seeds)),
        "bootstrap_lower_positive": bool(np.isfinite(lower) and lower > 0.0),
        "top1_improvement": (
            float(candidate["top1_accuracy"]) - float(baseline["top1_accuracy"])
            >= float(configuration["top1_improvement"])
        ),
        "binding_regret_increase": binding_regret_increase
        >= float(configuration["binding_regret_increase"]),
        "binding_tpa_drop": binding_tpa_drop >= float(configuration["binding_tpa_drop"]),
        "pair_regret_invariance": pair_regret_change
        <= float(configuration["pair_regret_absolute_change"]),
        "pair_tpa_invariance": pair_tpa_change
        <= float(configuration["pair_tpa_absolute_change"]),
        "twin_swap_sensitivity": twin_regret_increase
        >= float(configuration["twin_swap_regret_increase"]),
        "candidate_slot_invariance": slot_regret_change
        <= float(configuration["slot_regret_absolute_change"]),
        "anti_saturation_tpa": original_tpa < float(configuration["anti_saturation_tpa_max"]),
        "anti_saturation_regret": original_regret
        >= float(configuration["anti_saturation_regret_min"]),
    }
    result = {
        "schema_version": "actmask-binding-evaluation-v1",
        "dataset_id": suite_manifest["dataset_id"],
        "split": suite_manifest["split"],
        "candidate_method": candidate_method,
        "baseline_methods": list(baseline_methods),
        "strongest_baseline": strongest_baseline,
        "configuration": configuration,
        "method_manifests": manifests,
        "scores": scores,
        "comparison": {
            "relative_regret_improvement": float(relative_improvement),
            "top1_improvement": float(candidate["top1_accuracy"] - baseline["top1_accuracy"]),
            "seed_regret_differences": seed_differences,
            "seed_wins": seed_wins,
            "bootstrap_baseline_minus_candidate": bootstrap,
        },
        "intervention_effects": {
            "binding_regret_increase": float(binding_regret_increase),
            "binding_tpa_drop": float(binding_tpa_drop),
            "pair_regret_absolute_change": float(pair_regret_change),
            "pair_tpa_absolute_change": float(pair_tpa_change),
            "twin_swap_regret_increase": float(twin_regret_increase),
            "candidate_slot_regret_absolute_change": float(slot_regret_change),
        },
        "admission_evaluable": bool(admission_evaluable),
        "checks": {name: bool(value) for name, value in checks.items()},
        "decision": "METHOD_GO" if all(checks.values()) else "METHOD_NO_GO",
        "scope_note": "A METHOD_GO on one split is not GPU authorization; every preregistered split must pass.",
    }
    return result


def write_evaluation(path: str | Path, result: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


__all__ = ["DEFAULT_GATES", "evaluate_suite", "write_evaluation"]
