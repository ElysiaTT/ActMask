"""Small CPU-only example exercising the complete external-method boundary."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .adapters import CallableAdapter, run_adapter
from .evaluator import evaluate_suite, write_evaluation
from .interventions import build_intervention_suite
from .schema import BindingDataset


def make_example_dataset(
    *,
    seeds: tuple[int, ...] = (2301, 2302, 2303, 2304, 2305),
    twins_per_seed: int = 48,
    history_length: int = 12,
    candidate_count: int = 12,
) -> BindingDataset:
    total = len(seeds) * twins_per_seed
    history_actions = np.empty((total, 2, history_length, 2), dtype=np.float64)
    history_effects = np.empty((total, 2, history_length, 1), dtype=np.float64)
    history_times = np.broadcast_to(
        np.linspace(0.0, 0.9, history_length)[None, None],
        (total, 2, history_length),
    ).copy()
    history_mask = np.ones((total, 2, history_length), dtype=bool)
    context = np.empty((total, 2, 1), dtype=np.float64)
    candidates = np.empty((total, 2, candidate_count, 2), dtype=np.float64)
    candidate_effects = np.empty((total, 2, candidate_count, 1), dtype=np.float64)
    true_utility = np.empty((total, 2, candidate_count), dtype=np.float64)
    candidate_ids = np.empty((total, 2, candidate_count), dtype="<U48")
    twin_ids = np.empty(total, dtype="<U48")
    episode_ids = np.empty((total, 2), dtype="<U56")
    split_group_ids = np.empty(total, dtype="<U56")
    seed_ids = np.empty(total, dtype=np.int64)

    row = 0
    for seed in seeds:
        rng = np.random.default_rng(seed)
        for local_twin in range(twins_per_seed):
            actions = rng.uniform(-1.0, 1.0, size=(history_length, 2))
            candidate = rng.uniform(-1.15, 1.15, size=(candidate_count, 2))
            base_angle = rng.uniform(-np.pi, np.pi)
            separation = rng.uniform(np.deg2rad(55.0), np.deg2rad(125.0))
            angles = np.asarray((base_angle, base_angle + separation))
            if rng.random() < 0.5:
                angles = angles[::-1]
            target = rng.uniform(-0.45, 0.45)
            history_actions[row] = actions[None]
            candidates[row] = candidate[None]
            context[row, :, 0] = target
            for branch, angle in enumerate(angles):
                direction = np.asarray((np.cos(angle), np.sin(angle)))
                perpendicular = np.asarray((-np.sin(angle), np.cos(angle)))

                def response(action: np.ndarray) -> np.ndarray:
                    projected = action @ direction
                    side = action @ perpendicular
                    return np.tanh(1.6 * projected) + 0.18 * projected * side

                history_effects[row, branch, :, 0] = response(actions) + rng.normal(
                    0.0, 0.025, size=history_length
                )
                candidate_effects[row, branch, :, 0] = response(candidate)
                true_utility[row, branch] = -np.abs(response(candidate) - target)
                candidate_ids[row, branch] = [
                    f"s{seed}-t{local_twin}-c{slot}" for slot in range(candidate_count)
                ]
                episode_ids[row, branch] = f"s{seed}-t{local_twin}-b{branch}"
            twin_ids[row] = f"s{seed}-t{local_twin}"
            split_group_ids[row] = f"s{seed}-world{local_twin}"
            seed_ids[row] = seed
            row += 1
    return BindingDataset(
        dataset_id="binding-example-nonlinear-v1",
        split="example_id",
        history_actions=history_actions,
        history_effects=history_effects,
        history_times=history_times,
        history_mask=history_mask,
        context=context,
        candidates=candidates,
        candidate_ids=candidate_ids,
        true_utility=true_utility,
        twin_ids=twin_ids,
        episode_ids=episode_ids,
        split_group_ids=split_group_ids,
        seed_ids=seed_ids,
        candidate_effects=candidate_effects,
    )


def _linear_binding(inputs: BindingDataset) -> dict[str, np.ndarray]:
    twins, _, candidates, _ = inputs.candidates.shape
    prediction = np.empty((twins, 2, candidates, 1), dtype=np.float64)
    for twin in range(twins):
        for branch in range(2):
            valid = inputs.history_mask[twin, branch]
            design = np.concatenate(
                (
                    np.ones((int(valid.sum()), 1)),
                    inputs.history_actions[twin, branch, valid],
                ),
                axis=1,
            )
            gram = design.T @ design + 0.03 * np.eye(design.shape[1])
            coefficients = np.linalg.solve(
                gram, design.T @ inputs.history_effects[twin, branch, valid]
            )
            query = np.concatenate(
                (np.ones((candidates, 1)), inputs.candidates[twin, branch]), axis=1
            )
            prediction[twin, branch] = query @ coefficients
    target = inputs.context[:, :, None, :]
    return {
        "scores": -np.abs(prediction[..., 0] - target[..., 0]),
        "predicted_effects": prediction,
    }


def _result_mean(inputs: BindingDataset) -> dict[str, np.ndarray]:
    valid = inputs.history_mask[..., None]
    mean = (inputs.history_effects * valid).sum(axis=2) / valid.sum(axis=2)
    prediction = np.broadcast_to(
        mean[:, :, None, :],
        (*inputs.candidates.shape[:3], inputs.history_effects.shape[-1]),
    ).copy()
    target = inputs.context[:, :, None, :]
    return {
        "scores": -np.abs(prediction[..., 0] - target[..., 0]),
        "predicted_effects": prediction,
    }


def run_example(output_dir: str | Path) -> dict[str, Any]:
    output_dir = Path(output_dir)
    suite_dir = output_dir / "suite"
    predictions_dir = output_dir / "predictions"
    dataset = make_example_dataset()
    build_intervention_suite(
        dataset,
        suite_dir,
        seed=44019,
        source={
            "kind": "synthetic_example",
            "generator": "binding_bench.example.make_example_dataset",
            "truth_visibility": "evaluator_only",
        },
    )
    run_adapter(
        CallableAdapter("linear_binding", _linear_binding),
        suite_dir,
        predictions_dir / "linear_binding",
    )
    run_adapter(
        CallableAdapter("result_mean", _result_mean),
        suite_dir,
        predictions_dir / "result_mean",
    )
    result = evaluate_suite(
        suite_dir,
        {
            "linear_binding": predictions_dir / "linear_binding",
            "result_mean": predictions_dir / "result_mean",
        },
        candidate_method="linear_binding",
        baseline_methods=("result_mean",),
    )
    write_evaluation(output_dir / "evaluation.json", result)
    return result


__all__ = ["make_example_dataset", "run_example"]
