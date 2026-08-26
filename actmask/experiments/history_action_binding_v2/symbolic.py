"""Symbolic preflight using the exact preregistered response equations."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from actmask.experiments.history_action_binding_v2.audit import (
    MARGINAL_BASELINES, baseline_predictions, evaluate_all, save_prediction_rows,
)
from actmask.experiments.history_action_binding_v2.common import (
    OUTPUT_ROOT, PROTOCOL, TASKS, quantize, write_json,
)


def world_parameters(task: str, twins: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    rho = rng.uniform(-np.pi, np.pi, size=twins).astype(np.float32)
    swaps = np.asarray(([False, True] * ((twins + 1) // 2))[:twins], dtype=bool)
    rng.shuffle(swaps)
    if task == "task_a_discrete_higher_order":
        jitter = np.zeros(twins, dtype=np.float32)
        alpha = np.ones(twins, dtype=np.float32)
        beta = np.zeros(twins, dtype=np.float32)
    elif task == "task_b_continuous_nonlinear":
        jitter_range = np.deg2rad(PROTOCOL["task_b_hidden_phase_jitter_degrees"])
        jitter = rng.uniform(jitter_range[0], jitter_range[1], size=twins).astype(np.float32)
        alpha = rng.uniform(*PROTOCOL["task_b_alpha_range"], size=twins).astype(np.float32)
        beta = rng.uniform(*PROTOCOL["task_b_beta_range"], size=twins).astype(np.float32)
    else:
        raise ValueError(task)
    canonical_phi = np.stack((rho + jitter, rho + jitter + np.deg2rad(PROTOCOL["twin_phase_offset_degrees"])), axis=1).astype(np.float32)
    phi = canonical_phi.copy()
    for twin in np.flatnonzero(swaps):
        phi[twin] = phi[twin, ::-1]
    return {
        "public_rotation": rho,
        "latent_phi": phi,
        "latent_alpha": np.repeat(alpha[:, None], 2, axis=1),
        "latent_beta": np.repeat(beta[:, None], 2, axis=1),
        "branch_swapped": swaps,
        "hidden_phase_jitter": jitter,
    }


def probe_action_set(task: str, public_rotation: np.ndarray) -> np.ndarray:
    magnitudes = PROTOCOL["task_a_probe_magnitudes"] if task == "task_a_discrete_higher_order" else PROTOCOL["task_b_probe_magnitudes"]
    directions = np.deg2rad(PROTOCOL["probe_directions_degrees"])
    values = []
    for magnitude in magnitudes:
        angle = public_rotation[:, None] + directions[None, :]
        values.append(np.stack((magnitude * np.cos(angle), magnitude * np.sin(angle)), axis=2))
    return np.concatenate(values, axis=1).astype(np.float32)


def candidate_sampler(task: str, public_rotation: float, candidate_seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Sample public candidate trajectories from task, frame rotation, and RNG seed."""
    del task
    directions = np.deg2rad(PROTOCOL["probe_directions_degrees"]) + np.deg2rad(PROTOCOL["candidate_direction_offset_degrees"])
    templates = []
    for magnitude in PROTOCOL["candidate_magnitudes"]:
        angle = public_rotation + directions
        templates.extend(np.stack((magnitude * np.cos(angle), magnitude * np.sin(angle)), axis=1))
    templates = np.asarray(templates, dtype=np.float32)
    count = len(templates)
    shift = int(candidate_seed % count)
    multipliers = (1, 3, 5, 7, 9, 11, 13, 15)
    multiplier = multipliers[int((candidate_seed // count) % len(multipliers))]
    template_id = (multiplier * np.arange(count) + shift) % count
    peak = templates[template_id]
    profile = np.asarray(PROTOCOL["pulse_profile"], dtype=np.float32)
    actions = peak[:, None, :] * profile[None, :, None]
    return actions.astype(np.float32), template_id.astype(np.int16)


def candidate_arrays(task: str, public_rotation: np.ndarray, base_seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    actions, templates, seeds = [], [], []
    for twin, rho in enumerate(public_rotation):
        seed = base_seed + twin
        action, template = candidate_sampler(task, float(rho), seed)
        actions.append(action); templates.append(template); seeds.append(seed)
    return np.stack(actions), np.stack(templates), np.asarray(seeds, dtype=np.int64)


def ideal_displacement(action: np.ndarray, phi: np.ndarray, alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    action = np.asarray(action, dtype=np.float64)
    radius = np.linalg.norm(action, axis=-1)
    angle = np.arctan2(action[..., 1], action[..., 0])
    beta_factor = sum(np.square(PROTOCOL["pulse_profile"])) / sum(PROTOCOL["pulse_profile"])
    radial = alpha * radius - beta_factor * beta * np.square(radius)
    return float(PROTOCOL["symbolic_displacement_gain"]) * radial * np.cos(2.0 * (angle - phi))


def symbolic_dataset(task: str, twins: int = 64, seed: int | None = None) -> dict[str, np.ndarray]:
    seed = int(PROTOCOL["seed"] if seed is None else seed)
    task_offset = 0 if task == TASKS[0] else 1_000_000
    parameters = world_parameters(task, twins, seed + task_offset)
    probes = probe_action_set(task, parameters["public_rotation"])
    probes_by_branch = np.repeat(probes[:, None], 2, axis=1)
    phi = parameters["latent_phi"]
    alpha = parameters["latent_alpha"]
    beta = parameters["latent_beta"]
    probe_displacement = ideal_displacement(
        probes_by_branch, phi[:, :, None], alpha[:, :, None], beta[:, :, None]
    )
    probe_delta = quantize(probe_displacement, "public_probe_resolution")
    goal = float(PROTOCOL["goal_x"])
    probe_utility = quantize(-np.minimum(goal, np.abs(goal - probe_displacement)), "public_probe_resolution")
    probe_success = np.minimum(goal, np.abs(goal - probe_displacement)) <= float(PROTOCOL["success_radius"])
    actions, template_id, candidate_seeds = candidate_arrays(task, parameters["public_rotation"], seed + task_offset + 100_000)
    peak = actions[:, :, int(np.argmax(PROTOCOL["pulse_profile"]))]
    candidate_displacement = ideal_displacement(
        peak[:, None], phi[:, :, None], alpha[:, :, None], beta[:, :, None]
    )
    min_distance = np.minimum(goal, np.abs(goal - candidate_displacement))
    utility = -min_distance
    success = min_distance <= float(PROTOCOL["success_radius"])
    decision = np.zeros((twins, 2, 5), dtype=np.float32)
    decision[:, :, 2] = goal
    decision[:, :, 3] = np.cos(parameters["public_rotation"])[:, None]
    decision[:, :, 4] = np.sin(parameters["public_rotation"])[:, None]
    return {
        "task": np.asarray(task), "phase": np.asarray("symbolic"),
        "twin_id": np.arange(twins, dtype=np.int32),
        "public_rotation": parameters["public_rotation"],
        "latent_phi": phi, "latent_alpha": alpha, "latent_beta": beta,
        "latent_branch_swapped": parameters["branch_swapped"],
        "hidden_phase_jitter": parameters["hidden_phase_jitter"],
        "probe_actions": probes_by_branch, "probe_delta": probe_delta,
        "probe_delta_physical": probe_displacement.astype(np.float32),
        "probe_utility": probe_utility, "probe_success": probe_success,
        "decision_public_state": decision,
        "candidate_actions": actions,
        "candidate_actions_by_branch": np.repeat(actions[:, None], 2, axis=1),
        "candidate_template_id": template_id, "candidate_seed": candidate_seeds,
        "symbolic_candidate_displacement": candidate_displacement.astype(np.float32),
        "min_distance": min_distance.astype(np.float32),
        "final_error": np.abs(goal - candidate_displacement).astype(np.float32),
        "utility": utility.astype(np.float32), "success": success,
    }


def _top1_crossing(dataset: dict[str, np.ndarray]) -> float:
    selected = np.argmax(dataset["utility"], axis=2)
    templates = dataset["candidate_template_id"]
    left = templates[np.arange(len(templates)), selected[:, 0]]
    right = templates[np.arange(len(templates)), selected[:, 1]]
    return float(np.mean(left != right))


def _slot_balance(template_id: np.ndarray) -> tuple[bool, np.ndarray]:
    count = template_id.shape[1]
    table = np.zeros((count, count), dtype=np.int64)
    for twin in range(len(template_id)):
        for slot in range(count):
            table[slot, int(template_id[twin, slot])] += 1
    expected = len(template_id) / count
    return bool(np.all(table == expected)), table


def run_symbolic_preflight(output: Path | None = None) -> dict:
    output = OUTPUT_ROOT / "symbolic_preflight" if output is None else Path(output)
    output.mkdir(parents=True, exist_ok=True)
    tasks = {}
    for task_index, task in enumerate(TASKS):
        dataset = symbolic_dataset(task, twins=int(PROTOCOL["pilot_twins_per_task"]), seed=int(PROTOCOL["seed"]) + task_index * 1_000_000)
        task_output = output / task; task_output.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(task_output / "symbolic_samples.npz", **dataset)
        predictions, metric_sets, datasets = evaluate_all(dataset)
        rows = save_prediction_rows(task_output / "baseline_predictions.csv", datasets, predictions)
        original = metric_sets["original"]
        key = "candidate_conditioned_twin_preference_accuracy"
        slot_balanced, slot_table = _slot_balance(dataset["candidate_template_id"])
        result_multiset = bool(np.array_equal(np.sort(dataset["probe_delta"][:, 0], axis=1), np.sort(dataset["probe_delta"][:, 1], axis=1)))
        utility_multiset = bool(np.array_equal(np.sort(dataset["probe_utility"][:, 0], axis=1), np.sort(dataset["probe_utility"][:, 1], axis=1)))
        marginal_scores = {name: original[name][key] for name in MARGINAL_BASELINES}
        arrow = original["arrow_action_compatibility"][key]
        sysid = original["quadratic_fourier_sysid"][key]
        binding_drop = sysid - metric_sets["binding_breaking_shuffle"]["quadratic_fourier_sysid"][key]
        pair_change = abs(sysid - metric_sets["pair_preserving_shuffle"]["quadratic_fourier_sysid"][key])
        crossing = _top1_crossing(dataset)
        checks = {
            "action_marginal_exact": bool(np.array_equal(dataset["probe_actions"][:, 0], dataset["probe_actions"][:, 1])),
            "result_multiset_exact": result_multiset,
            "utility_multiset_exact": utility_multiset,
            "candidate_branch_exact": bool(np.array_equal(dataset["candidate_actions_by_branch"][:, 0], dataset["candidate_actions_by_branch"][:, 1])),
            "candidate_slots_balanced": slot_balanced,
            "marginals_below_0_60": max(marginal_scores.values()) <= float(PROTOCOL["symbolic_marginal_max"]),
            "arrow_below_0_70": arrow <= float(PROTOCOL["symbolic_arrow_max"]),
            "fourier_at_least_0_90": sysid >= float(PROTOCOL["symbolic_sysid_min"]),
            "binding_drop_at_least_0_20": binding_drop >= float(PROTOCOL["binding_drop_min"]),
            "pair_shuffle_at_most_0_02": pair_change <= float(PROTOCOL["pair_shuffle_max_change"]),
            "candidate_top1_crossing": crossing >= float(PROTOCOL["candidate_top1_crossing_min"]),
        }
        task_summary = {
            "task": task, "passed": all(checks.values()), "checks": checks,
            "marginal_scores": marginal_scores, "arrow_score": arrow, "fourier_score": sysid,
            "binding_break_score": metric_sets["binding_breaking_shuffle"]["quadratic_fourier_sysid"][key],
            "binding_drop": binding_drop, "pair_preserving_change": pair_change,
            "candidate_top1_crossing_rate": crossing, "slot_balance_table": slot_table.tolist(),
            "prediction_rows": rows, "metrics": metric_sets,
        }
        write_json(task_output / "preflight_summary.json", task_summary)
        tasks[task] = task_summary
    summary = {"passed": all(value["passed"] for value in tasks.values()), "formal_simulator_data_generated": False, "tasks": tasks}
    write_json(output / "summary.json", summary)
    return summary


if __name__ == "__main__":
    print(json.dumps(run_symbolic_preflight(), indent=2, sort_keys=True))

