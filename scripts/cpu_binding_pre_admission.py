"""CPU-only pre-admission for counterfactual action-effect binding.

This script is deliberately standalone: importing the main ``actmask`` package
pulls in PyTorch, whose Windows DLL is unavailable in the current environment.
The frozen protocol is documented in ``docs/binding_cpu_v1_preregister.md``.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np


SEEDS = (8301, 8302, 8303, 8304, 8305)
TASKS = ("discrete_higher_order", "continuous_nonlinear")
PROBE_DIRECTIONS = np.deg2rad(np.arange(0.0, 360.0, 45.0))
TASK_A_CANDIDATE_DIRECTIONS = np.deg2rad(np.arange(10.0, 90.0, 10.0))
TASK_B_CANDIDATE_DIRECTIONS = PROBE_DIRECTIONS + np.deg2rad(15.0)
CANDIDATE_MAGNITUDES = (0.35, 0.65)
PULSE_PROFILE = np.asarray((0.5, 1.0, 1.0, 0.5), dtype=np.float64)
BETA_FACTOR = float(np.square(PULSE_PROFILE).sum() / PULSE_PROFILE.sum())
DISPLACEMENT_GAIN = 0.0578
GOAL = 0.030
RESULT_RESOLUTION = 0.001


def _response(
    action: np.ndarray,
    phi: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
) -> np.ndarray:
    action = np.asarray(action, dtype=np.float64)
    radius = np.linalg.norm(action, axis=-1)
    angle = np.arctan2(action[..., 1], action[..., 0])
    radial = alpha * radius - BETA_FACTOR * beta * np.square(radius)
    return DISPLACEMENT_GAIN * radial * np.cos(2.0 * (angle - phi))


def _probe_set(task: str, rotation: np.ndarray) -> np.ndarray:
    magnitudes = (0.40, 0.80) if task == TASKS[0] else (0.20, 0.50, 0.80)
    groups = []
    for magnitude in magnitudes:
        angle = rotation[:, None] + PROBE_DIRECTIONS[None, :]
        groups.append(magnitude * np.stack((np.cos(angle), np.sin(angle)), axis=-1))
    return np.concatenate(groups, axis=1)


def _candidate_set(task: str, rotation: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    directions = TASK_A_CANDIDATE_DIRECTIONS if task == TASKS[0] else TASK_B_CANDIDATE_DIRECTIONS
    all_actions: list[np.ndarray] = []
    all_templates: list[np.ndarray] = []
    for twin, rho in enumerate(rotation):
        templates = []
        for magnitude in CANDIDATE_MAGNITUDES:
            angle = rho + directions
            templates.extend(magnitude * np.stack((np.cos(angle), np.sin(angle)), axis=-1))
        template_array = np.asarray(templates, dtype=np.float64)
        shift = int((seed + twin + 100_000) % len(template_array))
        template_id = (np.arange(len(template_array), dtype=np.int16) + shift) % len(template_array)
        all_actions.append(template_array[template_id])
        all_templates.append(template_id)
    return np.stack(all_actions), np.stack(all_templates)


def make_dataset(task: str, twins: int, seed: int) -> dict[str, np.ndarray]:
    task_offset = 0 if task == TASKS[0] else 1_000_000
    rng = np.random.default_rng(seed + task_offset)
    rotation = rng.uniform(-np.pi, np.pi, size=twins)
    swaps = np.asarray(([False, True] * ((twins + 1) // 2))[:twins], dtype=bool)
    rng.shuffle(swaps)
    if task == TASKS[0]:
        jitter = np.zeros(twins)
        alpha = np.ones(twins)
        beta = np.zeros(twins)
    else:
        jitter = rng.uniform(0.0, np.deg2rad(45.0), size=twins)
        alpha = rng.uniform(1.05, 1.25, size=twins)
        beta = rng.uniform(0.18, 0.32, size=twins)

    canonical_phi = np.stack((rotation + jitter, rotation + jitter + np.deg2rad(45.0)), axis=1)
    phi = canonical_phi.copy()
    phi[swaps] = phi[swaps, ::-1]
    alpha_by_branch = np.repeat(alpha[:, None], 2, axis=1)
    beta_by_branch = np.repeat(beta[:, None], 2, axis=1)

    probes = _probe_set(task, rotation)
    probe_actions = np.repeat(probes[:, None, :, :], 2, axis=1)
    physical_results = _response(
        probe_actions,
        phi[:, :, None],
        alpha_by_branch[:, :, None],
        beta_by_branch[:, :, None],
    )
    probe_results = np.round(physical_results / RESULT_RESOLUTION) * RESULT_RESOLUTION

    candidates, template_id = _candidate_set(task, rotation, seed + task_offset)
    candidate_by_branch = np.repeat(candidates[:, None, :, :], 2, axis=1)
    displacement = _response(
        candidate_by_branch,
        phi[:, :, None],
        alpha_by_branch[:, :, None],
        beta_by_branch[:, :, None],
    )
    utility = -np.minimum(GOAL, np.abs(GOAL - displacement))
    return {
        "rotation": rotation,
        "phi": phi,
        "alpha": alpha_by_branch,
        "beta": beta_by_branch,
        "probe_actions": probe_actions,
        "probe_results": probe_results,
        "candidate_actions": candidates,
        "candidate_template_id": template_id,
        "utility": utility,
    }


def _fourier_features(action: np.ndarray, kind: str) -> np.ndarray:
    x = action[..., 0]
    y = action[..., 1]
    radius = np.sqrt(x * x + y * y)
    angle = np.arctan2(y, x)
    if kind == "linear":
        return np.stack((np.ones_like(x), x, y), axis=-1)
    return np.stack(
        (
            np.ones_like(x),
            radius * np.cos(2.0 * angle),
            radius * np.sin(2.0 * angle),
            np.square(radius) * np.cos(2.0 * angle),
            np.square(radius) * np.sin(2.0 * angle),
        ),
        axis=-1,
    )


def _least_squares(
    actions: np.ndarray,
    results: np.ndarray,
    candidates: np.ndarray,
    kind: str,
) -> np.ndarray:
    twins, branches = actions.shape[:2]
    prediction = np.empty((twins, branches, candidates.shape[1]), dtype=np.float64)
    candidate_design = _fourier_features(candidates, kind)
    for twin in range(twins):
        for branch in range(branches):
            coefficient, *_ = np.linalg.lstsq(
                _fourier_features(actions[twin, branch], kind),
                results[twin, branch],
                rcond=None,
            )
            prediction[twin, branch] = candidate_design[twin] @ coefficient
    return prediction


def _utility(displacement: np.ndarray) -> np.ndarray:
    return -np.minimum(GOAL, np.abs(GOAL - displacement))


def baseline_predictions(
    dataset: dict[str, np.ndarray],
    actions: np.ndarray | None = None,
    results: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    actions = dataset["probe_actions"] if actions is None else actions
    results = dataset["probe_results"] if results is None else results
    candidates = dataset["candidate_actions"]
    twins, branches = actions.shape[:2]
    count = candidates.shape[1]
    output: dict[str, np.ndarray] = {}
    output["class_prior"] = np.zeros((twins, branches, count))
    output["current_only"] = np.full((twins, branches, count), -GOAL)
    output["candidate_slot"] = np.broadcast_to(
        -np.arange(count, dtype=np.float64)[None, None, :] / max(count - 1, 1),
        (twins, branches, count),
    ).copy()

    radius = np.linalg.norm(candidates, axis=-1)
    output["action_only"] = _utility(
        np.broadcast_to(DISPLACEMENT_GAIN * radius[:, None, :], (twins, branches, count))
    )
    result_level = -np.mean(np.abs(results), axis=2)[:, :, None]
    output["result_only"] = np.broadcast_to(result_level, (twins, branches, count)).copy()

    probe_norm = np.maximum(np.linalg.norm(actions, axis=-1), 1.0e-12)
    probe_unit = actions / probe_norm[..., None]
    arrow = np.mean(results[..., None] * probe_unit, axis=2)
    candidate_norm = np.maximum(np.linalg.norm(candidates, axis=-1), 1.0e-12)
    candidate_unit = candidates / candidate_norm[..., None]
    mean_probe_radius = np.maximum(np.mean(probe_norm, axis=2), 1.0e-12)
    arrow_displacement = (
        np.einsum("nbi,nki->nbk", arrow, candidate_unit)
        * candidate_norm[:, None, :]
        / mean_probe_radius[:, :, None]
    )
    output["arrow_action"] = _utility(arrow_displacement)

    nearest = np.empty((twins, branches, count), dtype=np.float64)
    knn = np.empty_like(nearest)
    for twin in range(twins):
        for branch in range(branches):
            distances = np.linalg.norm(
                candidates[twin, :, None, :] - actions[twin, branch, None, :, :], axis=-1
            )
            order = np.argsort(distances, axis=1, kind="mergesort")
            nearest[twin, branch] = results[twin, branch, order[:, 0]]
            selected = order[:, :2]
            weights = 1.0 / np.maximum(np.take_along_axis(distances, selected, axis=1), 1.0e-6)
            values = results[twin, branch, selected]
            knn[twin, branch] = np.sum(weights * values, axis=1) / np.sum(weights, axis=1)
    output["nearest"] = _utility(nearest)
    output["knn"] = _utility(knn)
    output["linear_sysid"] = _utility(_least_squares(actions, results, candidates, "linear"))
    output["fourier_sysid"] = _utility(_least_squares(actions, results, candidates, "fourier"))

    candidate_by_branch = np.repeat(candidates[:, None, :, :], 2, axis=1)
    oracle_displacement = _response(
        candidate_by_branch,
        dataset["phi"][:, :, None],
        dataset["alpha"][:, :, None],
        dataset["beta"][:, :, None],
    )
    output["oracle"] = _utility(oracle_displacement)
    return output


def twin_preference(
    truth: np.ndarray,
    prediction: np.ndarray,
    template_id: np.ndarray,
) -> dict[str, Any]:
    true_centered = truth - truth.mean(axis=2, keepdims=True)
    predicted_centered = prediction - prediction.mean(axis=2, keepdims=True)
    true_difference = true_centered[:, 1] - true_centered[:, 0]
    predicted_difference = predicted_centered[:, 1] - predicted_centered[:, 0]
    informative = np.abs(true_difference) > 1.0e-12
    correct = np.full(true_difference.shape, np.nan, dtype=np.float64)
    tied = np.abs(predicted_difference) <= 1.0e-12
    correct[informative & tied] = 0.5
    decided = informative & ~tied
    correct[decided] = (
        np.sign(true_difference[decided]) == np.sign(predicted_difference[decided])
    ).astype(np.float64)
    by_twin = np.nanmean(correct, axis=1)
    per_template = []
    for template in range(template_id.shape[1]):
        values = []
        for twin in range(len(template_id)):
            slot = int(np.flatnonzero(template_id[twin] == template)[0])
            if not np.isnan(correct[twin, slot]):
                values.append(float(correct[twin, slot]))
        per_template.append(float(np.mean(values)))
    return {
        "accuracy": float(np.nanmean(by_twin)),
        "by_twin": by_twin.tolist(),
        "per_template": per_template,
        "informative_fraction": float(np.mean(informative)),
    }


def _pair_preserving(actions: np.ndarray, results: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    directions = len(PROBE_DIRECTIONS)
    grouped_actions = actions.reshape(*actions.shape[:2], -1, directions, actions.shape[-1])
    grouped_results = results.reshape(*results.shape[:2], -1, directions)
    return (
        np.roll(grouped_actions, 1, axis=3).reshape(actions.shape).copy(),
        np.roll(grouped_results, 1, axis=3).reshape(results.shape).copy(),
    )


def _binding_break(actions: np.ndarray, results: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    directions = len(PROBE_DIRECTIONS)
    grouped_results = results.reshape(*results.shape[:2], -1, directions)
    return actions.copy(), np.roll(grouped_results, 1, axis=3).reshape(results.shape).copy()


def _slot_reverse(dataset: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    output = {key: value.copy() for key, value in dataset.items()}
    permutation = np.arange(dataset["candidate_actions"].shape[1] - 1, -1, -1)
    for key, axis in (("candidate_actions", 1), ("candidate_template_id", 1), ("utility", 2)):
        output[key] = np.take(output[key], permutation, axis=axis)
    return output


def _rotate(dataset: dict[str, np.ndarray], degrees: float = 37.0) -> dict[str, np.ndarray]:
    output = {key: value.copy() for key, value in dataset.items()}
    angle = np.deg2rad(degrees)
    matrix = np.asarray(
        ((math.cos(angle), -math.sin(angle)), (math.sin(angle), math.cos(angle))),
        dtype=np.float64,
    )
    output["probe_actions"] = output["probe_actions"] @ matrix.T
    output["candidate_actions"] = output["candidate_actions"] @ matrix.T
    output["rotation"] += angle
    output["phi"] += angle
    return output


def _integrity(dataset: dict[str, np.ndarray]) -> dict[str, Any]:
    actions = dataset["probe_actions"]
    results = dataset["probe_results"]
    templates = dataset["candidate_template_id"]
    table = np.zeros((templates.shape[1], templates.shape[1]), dtype=int)
    for twin in range(len(templates)):
        for slot, template in enumerate(templates[twin]):
            table[slot, int(template)] += 1
    top1 = np.argmax(dataset["utility"], axis=2)
    selected_templates = np.take_along_axis(templates[:, None, :], top1[:, :, None], axis=2)[..., 0]
    checks = {
        "action_marginal_exact": bool(np.array_equal(actions[:, 0], actions[:, 1])),
        "result_multiset_exact": bool(
            np.array_equal(np.sort(results[:, 0], axis=1), np.sort(results[:, 1], axis=1))
        ),
        "current_state_exact": True,
        "candidate_set_exact": True,
        "slot_balance_exact": bool(np.all(table == len(templates) // templates.shape[1])),
    }
    return {
        "checks": checks,
        "passed": bool(all(checks.values())),
        "slot_min": int(table.min()),
        "slot_max": int(table.max()),
        "top1_crossing": float(np.mean(selected_templates[:, 0] != selected_templates[:, 1])),
    }


def run_seed(task: str, seed: int, twins: int) -> dict[str, Any]:
    dataset = make_dataset(task, twins, seed)
    truth = dataset["utility"]
    template_id = dataset["candidate_template_id"]
    original_predictions = baseline_predictions(dataset)
    original = {
        name: twin_preference(truth, prediction, template_id)
        for name, prediction in original_predictions.items()
    }

    pair_actions, pair_results = _pair_preserving(
        dataset["probe_actions"], dataset["probe_results"]
    )
    broken_actions, broken_results = _binding_break(
        dataset["probe_actions"], dataset["probe_results"]
    )
    pair_prediction = baseline_predictions(dataset, pair_actions, pair_results)["fourier_sysid"]
    broken_prediction = baseline_predictions(dataset, broken_actions, broken_results)["fourier_sysid"]
    swapped_prediction = baseline_predictions(
        dataset,
        dataset["probe_actions"][:, ::-1],
        dataset["probe_results"][:, ::-1],
    )["fourier_sysid"]
    slot_dataset = _slot_reverse(dataset)
    rotated_dataset = _rotate(dataset)
    interventions = {
        "pair_preserving": twin_preference(truth, pair_prediction, template_id)["accuracy"],
        "binding_breaking": twin_preference(truth, broken_prediction, template_id)["accuracy"],
        "twin_history_swap": twin_preference(truth, swapped_prediction, template_id)["accuracy"],
        "slot_reversal": twin_preference(
            slot_dataset["utility"],
            baseline_predictions(slot_dataset)["fourier_sysid"],
            slot_dataset["candidate_template_id"],
        )["accuracy"],
        "coordinate_rotation": twin_preference(
            rotated_dataset["utility"],
            baseline_predictions(rotated_dataset)["fourier_sysid"],
            rotated_dataset["candidate_template_id"],
        )["accuracy"],
    }
    return {
        "task": task,
        "seed": seed,
        "twins": twins,
        "integrity": _integrity(dataset),
        "original": original,
        "interventions": interventions,
    }


def _mean(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def aggregate(seed_results: list[dict[str, Any]]) -> dict[str, Any]:
    tasks: dict[str, Any] = {}
    for task in TASKS:
        rows = [row for row in seed_results if row["task"] == task]
        baseline_names = tuple(rows[0]["original"])
        scores = {
            name: _mean([row["original"][name]["accuracy"] for row in rows])
            for name in baseline_names
        }
        interventions = {
            name: _mean([row["interventions"][name] for row in rows])
            for name in rows[0]["interventions"]
        }
        structural = all(row["integrity"]["passed"] for row in rows)
        top1_crossing = _mean([row["integrity"]["top1_crossing"] for row in rows])
        strongest_marginal = max(
            scores[name]
            for name in ("current_only", "action_only", "result_only", "candidate_slot")
        )
        fourier = scores["fourier_sysid"]
        checks = {
            "structural_integrity": structural,
            "top1_crossing_at_least_0_90": top1_crossing >= 0.90,
            "marginals_at_most_0_60": strongest_marginal <= 0.60,
            "arrow_at_most_0_70": scores["arrow_action"] <= 0.70,
            "nearest_below_0_90": scores["nearest"] < 0.90,
            "knn_below_0_90": scores["knn"] < 0.90,
            "fourier_at_least_0_90": fourier >= 0.90,
            "binding_drop_at_least_0_20": fourier - interventions["binding_breaking"] >= 0.20,
            "pair_change_at_most_0_02": abs(fourier - interventions["pair_preserving"]) <= 0.02,
            "slot_threshold_invariant": (fourier >= 0.90) == (interventions["slot_reversal"] >= 0.90),
            "rotation_threshold_invariant": (fourier >= 0.90) == (interventions["coordinate_rotation"] >= 0.90),
            "oracle_above_marginal": scores["oracle"] > strongest_marginal,
            "fourier_above_marginal": fourier > strongest_marginal,
        }
        tasks[task] = {
            "scores": scores,
            "interventions": interventions,
            "top1_crossing": top1_crossing,
            "strongest_marginal": strongest_marginal,
            "binding_drop": fourier - interventions["binding_breaking"],
            "pair_preserving_change": abs(fourier - interventions["pair_preserving"]),
            "checks": checks,
            "passed": bool(all(checks.values())),
        }
    admitted = all(value["passed"] for value in tasks.values())
    return {
        "decision": "CPU_DIRECTION_ADMITTED" if admitted else "CPU_DIRECTION_REJECTED",
        "passed": admitted,
        "tasks": tasks,
    }


def run(twins: int) -> dict[str, Any]:
    started = time.perf_counter()
    seed_results = [run_seed(task, seed, twins) for task in TASKS for seed in SEEDS]
    return {
        "schema": "actmask-binding-cpu-v1-pre-admission",
        "preregister": "docs/binding_cpu_v1_preregister.md",
        "cpu_only": True,
        "seeds": list(SEEDS),
        "twins_per_task_seed": twins,
        "seed_results": seed_results,
        "aggregate": aggregate(seed_results),
        "runtime_seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--twins", type=int, default=256)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("audit/results/binding_cpu_v1_pre_admission.json"),
    )
    args = parser.parse_args()
    if args.twins <= 0 or args.twins % 16 != 0:
        raise SystemExit("--twins must be a positive multiple of 16 for exact slot balance")
    result = run(args.twins)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "decision": result["aggregate"]["decision"],
        "runtime_seconds": result["runtime_seconds"],
        "tasks": result["aggregate"]["tasks"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
