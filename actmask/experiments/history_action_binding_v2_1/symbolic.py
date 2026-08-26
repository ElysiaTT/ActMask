"""Development and one-shot official symbolic evaluation for v2.1."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from actmask.experiments.history_action_binding_v2_1.audit import (
    binding_breaking_shuffle,
    candidate_slot_permutation,
    coordinate_rotation,
    evaluate_all,
    pair_preserving_shuffle,
    save_prediction_rows,
    symbolic_gate_checks,
    twin_history_swap,
)
from actmask.experiments.history_action_binding_v2_1.common import (
    OUTPUT_ROOT,
    PROJECT_ROOT,
    PROTOCOL,
    TASKS,
    canonical_json,
    quantize,
    scientific_core_manifest,
    sha256_bytes,
    sha256_file,
    write_json,
)


class OfficialPreflightRefused(RuntimeError):
    """Raised when the frozen one-shot conditions are not satisfied."""


def world_parameters(task: str, twins: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    rho = rng.uniform(-np.pi, np.pi, size=twins).astype(np.float32)
    swaps = np.asarray(([False, True] * ((twins + 1) // 2))[:twins], dtype=bool)
    rng.shuffle(swaps)
    if task == TASKS[0]:
        jitter = np.zeros(twins, dtype=np.float32)
        alpha = np.ones(twins, dtype=np.float32)
        beta = np.zeros(twins, dtype=np.float32)
    elif task == TASKS[1]:
        jitter_range = np.deg2rad(PROTOCOL["task_b_hidden_phase_jitter_degrees"])
        jitter = rng.uniform(jitter_range[0], jitter_range[1], size=twins).astype(np.float32)
        alpha = rng.uniform(*PROTOCOL["task_b_alpha_range"], size=twins).astype(np.float32)
        beta = rng.uniform(*PROTOCOL["task_b_beta_range"], size=twins).astype(np.float32)
    else:
        raise ValueError(task)
    canonical_phi = np.stack(
        (rho + jitter, rho + jitter + np.deg2rad(PROTOCOL["twin_phase_offset_degrees"])), axis=1
    ).astype(np.float32)
    phi = canonical_phi.copy()
    for twin in np.flatnonzero(swaps):
        phi[twin] = phi[twin, ::-1]
    return {
        "public_rotation": rho,
        "latent_phi": phi,
        "latent_alpha": np.repeat(alpha[:, None], 2, axis=1),
        "latent_beta": np.repeat(beta[:, None], 2, axis=1),
        "latent_branch_swapped": swaps,
        "hidden_phase_jitter": jitter,
    }


def probe_action_set(task: str, public_rotation: np.ndarray) -> np.ndarray:
    magnitudes = (
        PROTOCOL["task_a_probe_magnitudes"] if task == TASKS[0]
        else PROTOCOL["task_b_probe_magnitudes"]
    )
    directions = np.deg2rad(PROTOCOL["probe_directions_degrees"])
    values = []
    for magnitude in magnitudes:
        angle = public_rotation[:, None] + directions[None, :]
        values.append(np.stack((magnitude * np.cos(angle), magnitude * np.sin(angle)), axis=2))
    return np.concatenate(values, axis=1).astype(np.float32)


def candidate_sampler(task: str, public_rotation: float, candidate_seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return a deterministic public-input-only cyclic candidate schedule."""
    if task == TASKS[0]:
        directions = np.deg2rad(PROTOCOL["task_a_candidate_directions_degrees"])
    elif task == TASKS[1]:
        directions = (
            np.deg2rad(PROTOCOL["probe_directions_degrees"])
            + np.deg2rad(PROTOCOL["task_b_candidate_direction_offset_degrees"])
        )
    else:
        raise ValueError(task)
    templates = []
    for magnitude in PROTOCOL["candidate_magnitudes"]:
        angle = public_rotation + directions
        templates.extend(np.stack((magnitude * np.cos(angle), magnitude * np.sin(angle)), axis=1))
    templates = np.asarray(templates, dtype=np.float32)
    count = len(templates)
    shift = int(candidate_seed % count)
    template_id = (np.arange(count, dtype=np.int16) + shift) % count
    peak = templates[template_id]
    profile = np.asarray(PROTOCOL["pulse_profile"], dtype=np.float32)
    actions = peak[:, None, :] * profile[None, :, None]
    return actions.astype(np.float32), template_id


def candidate_arrays(
    task: str, public_rotation: np.ndarray, base_seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    actions, templates, seeds = [], [], []
    for twin, rho in enumerate(public_rotation):
        seed = int(base_seed + twin)
        action, template = candidate_sampler(task, float(rho), seed)
        actions.append(action)
        templates.append(template)
        seeds.append(seed)
    return np.stack(actions), np.stack(templates), np.asarray(seeds, dtype=np.int64)


def ideal_displacement(
    action: np.ndarray, phi: np.ndarray, alpha: np.ndarray, beta: np.ndarray
) -> np.ndarray:
    action = np.asarray(action, dtype=np.float64)
    radius = np.linalg.norm(action, axis=-1)
    angle = np.arctan2(action[..., 1], action[..., 0])
    beta_factor = sum(np.square(PROTOCOL["pulse_profile"])) / sum(PROTOCOL["pulse_profile"])
    radial = alpha * radius - beta_factor * beta * np.square(radius)
    return (
        float(PROTOCOL["symbolic_displacement_gain"])
        * radial
        * np.cos(2.0 * (angle - phi))
    )


def symbolic_dataset(task: str, twins: int, seed: int, phase: str) -> dict[str, np.ndarray]:
    if task not in TASKS:
        raise ValueError(task)
    task_offset = 0 if task == TASKS[0] else 1_000_000
    parameters = world_parameters(task, twins, int(seed) + task_offset)
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
    probe_utility = quantize(
        -np.minimum(goal, np.abs(goal - probe_displacement)), "public_probe_resolution"
    )
    probe_success = np.minimum(goal, np.abs(goal - probe_displacement)) <= float(PROTOCOL["success_radius"])
    actions, template_id, candidate_seeds = candidate_arrays(
        task, parameters["public_rotation"], int(seed) + task_offset + 100_000
    )
    peak_index = int(np.argmax(PROTOCOL["pulse_profile"]))
    peak = actions[:, :, peak_index]
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
        "task": np.asarray(task),
        "phase": np.asarray(phase),
        "twin_id": np.arange(twins, dtype=np.int32),
        "public_rotation": parameters["public_rotation"],
        "latent_phi": phi,
        "latent_alpha": alpha,
        "latent_beta": beta,
        "latent_branch_swapped": parameters["latent_branch_swapped"],
        "hidden_phase_jitter": parameters["hidden_phase_jitter"],
        "probe_actions": probes_by_branch,
        "probe_delta": probe_delta,
        "probe_delta_physical": probe_displacement.astype(np.float32),
        "probe_utility": probe_utility,
        "probe_success": probe_success,
        "decision_public_state": decision,
        "candidate_actions": actions,
        "candidate_actions_by_branch": np.repeat(actions[:, None], 2, axis=1),
        "candidate_template_id": template_id,
        "candidate_seed": candidate_seeds,
        "symbolic_candidate_displacement": candidate_displacement.astype(np.float32),
        "min_distance": min_distance.astype(np.float32),
        "final_error": np.abs(goal - candidate_displacement).astype(np.float32),
        "utility": utility.astype(np.float32),
        "success": success,
    }


def _save_interventions(path: Path, dataset: dict, datasets: dict) -> None:
    actions = dataset["probe_actions"]
    results = dataset["probe_delta"]
    pair_actions, pair_results = pair_preserving_shuffle(actions, results)
    broken_actions, broken_results = binding_breaking_shuffle(actions, results)
    swapped_actions, swapped_results = twin_history_swap(actions, results)
    permuted = datasets["candidate_slot_permutation"]
    rotated = datasets["coordinate_rotation"]
    np.savez_compressed(
        path,
        original_probe_actions=actions,
        original_probe_results=results,
        pair_preserving_probe_actions=pair_actions,
        pair_preserving_probe_results=pair_results,
        binding_break_probe_actions=broken_actions,
        binding_break_probe_results=broken_results,
        twin_swap_probe_actions=swapped_actions,
        twin_swap_probe_results=swapped_results,
        fixed_current_observation=dataset["decision_public_state"],
        fixed_candidate_actions=dataset["candidate_actions"],
        original_success=dataset["success"],
        original_utility=dataset["utility"],
        candidate_template_id=dataset["candidate_template_id"],
        slot_permuted_candidates=permuted["candidate_actions"],
        slot_permuted_template_id=permuted["candidate_template_id"],
        slot_permuted_success=permuted["success"],
        slot_permuted_utility=permuted["utility"],
        rotated_probe_actions=rotated["probe_actions"],
        rotated_candidates=rotated["candidate_actions"],
        rotated_current=rotated["decision_public_state"],
        novel_candidate_actions=dataset["candidate_actions"],
        simulator_expected_preference=np.sign(dataset["utility"][:, 1] - dataset["utility"][:, 0]).astype(np.int8),
        expected_preference_after_twin_swap=-np.sign(dataset["utility"][:, 1] - dataset["utility"][:, 0]).astype(np.int8),
    )


def _verify_freeze() -> dict:
    path = OUTPUT_ROOT / "freeze_record.json"
    if not path.exists():
        raise OfficialPreflightRefused("freeze_record.json is missing")
    freeze = json.loads(path.read_text())
    if freeze.get("status") != "FROZEN_BEFORE_OFFICIAL_PREFLIGHT":
        raise OfficialPreflightRefused("freeze record has the wrong status")
    current = scientific_core_manifest()
    if current != freeze.get("scientific_core_manifest"):
        raise OfficialPreflightRefused("scientific core differs from the frozen manifest")
    prereg = PROJECT_ROOT / "docs/history_action_binding_v2_1_preregister.md"
    if sha256_file(prereg) != freeze.get("preregister_sha256"):
        raise OfficialPreflightRefused("preregistration differs from the frozen digest")
    return freeze


def run_official_symbolic() -> dict:
    freeze = _verify_freeze()
    output = OUTPUT_ROOT / "official_symbolic_preflight"
    output.mkdir(parents=True, exist_ok=True)
    started_path = output / "official_run_started.json"
    try:
        descriptor = os.open(started_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as exc:
        raise OfficialPreflightRefused("official symbolic preflight has already started and cannot be rerun") from exc
    started = {
        "status": "STARTED_ONCE",
        "official_seed": int(freeze["official_seed"]),
        "freeze_record_sha256": sha256_file(OUTPUT_ROOT / "freeze_record.json"),
    }
    with os.fdopen(descriptor, "w") as handle:
        handle.write(json.dumps(started, indent=2, sort_keys=True) + "\n")

    summaries = {}
    for task in TASKS:
        task_output = output / task
        task_output.mkdir(parents=True, exist_ok=True)
        dataset = symbolic_dataset(
            task,
            twins=int(PROTOCOL["pilot_twins_per_task"]),
            seed=int(freeze["official_seed"]),
            phase="official_symbolic",
        )
        np.savez_compressed(task_output / "symbolic_samples.npz", **dataset)
        predictions, metric_sets, datasets = evaluate_all(dataset)
        prediction_rows = save_prediction_rows(
            task_output / "baseline_predictions.csv", datasets, predictions
        )
        _save_interventions(task_output / "intervention_samples.npz", dataset, datasets)
        checks, details = symbolic_gate_checks(dataset, metric_sets, candidate_sampler)
        summary = {
            "task": task,
            "passed": bool(all(checks.values())),
            "checks": checks,
            "details": details,
            "prediction_rows": int(prediction_rows),
            "sample_sha256": sha256_file(task_output / "symbolic_samples.npz"),
            "predictions_sha256": sha256_file(task_output / "baseline_predictions.csv"),
            "interventions_sha256": sha256_file(task_output / "intervention_samples.npz"),
        }
        write_json(task_output / "preflight_summary.json", summary)
        summaries[task] = summary
    overall = {
        "status": "SYMBOLIC_PREFLIGHT_PASS" if all(item["passed"] for item in summaries.values()) else "SYMBOLIC_PREFLIGHT_FAIL",
        "passed": bool(all(item["passed"] for item in summaries.values())),
        "official_seed": int(freeze["official_seed"]),
        "official_run_count": 1,
        "formal_simulator_data_generated": False,
        "tasks": summaries,
    }
    write_json(output / "symbolic_preflight_summary.json", overall)
    write_json(output / "official_run_completed.json", {
        "status": overall["status"],
        "summary_sha256": sha256_file(output / "symbolic_preflight_summary.json"),
        "scientific_core_combined_sha256": scientific_core_manifest()["combined_sha256"],
    })
    return overall


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official", action="store_true")
    args = parser.parse_args()
    if not args.official:
        raise SystemExit("development evaluation is provided by development.py; pass --official only after freeze")
    result = run_official_symbolic()
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
