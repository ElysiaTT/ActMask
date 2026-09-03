"""CPU-only anti-saturation test for raw action-effect binding.

Protocol: docs/binding_cpu_v2_antisaturation_preregister.md
This file deliberately avoids ActMask/PyTorch imports and any GPU backend.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor


SEEDS = (10501, 10502, 10503, 10504, 10505)
SPLITS = ("id", "parameter_ood", "noise_ood", "held_structure")
MAX_HISTORY = 28
CANDIDATES = 16
TRAIN_TWINS = 512
EVAL_TWINS = 256
TREES = 64
MAX_DEPTH = 18
MIN_LEAF = 2
MAX_FEATURES = 0.7
RBF_LENGTH_SCALE = 0.45
RBF_REGULARIZATION = 0.01
POLY_REGULARIZATION = 0.01
BOOTSTRAP_REPLICATES = 1000
PREREGISTER = Path("docs/binding_cpu_v2_antisaturation_preregister.md")
LEARNED_MODES = (
    "paired_raw", "unbound_raw", "action_only", "result_only", "context_only"
)
BASELINE_NAMES = (
    "unbound_raw", "action_only", "result_only", "context_only",
    "result_mean", "nearest", "knn5", "poly2_ridge", "rbf_ridge",
)


def _mechanism_codes(kind: str, twins: int, rng: np.random.Generator) -> np.ndarray:
    if kind == "balanced":
        value = np.asarray(([0, 1] * ((twins + 1) // 2))[:twins], dtype=np.int8)
        rng.shuffle(value)
        return value
    if kind == "held":
        return np.full(twins, 2, dtype=np.int8)
    raise ValueError(f"unknown mechanism kind: {kind}")


def _parameters(
    twins: int, parameter_range: str, rng: np.random.Generator
) -> dict[str, np.ndarray]:
    if parameter_range == "train":
        gain = rng.uniform(0.65, 1.00, twins)
        bias = rng.uniform(-0.15, 0.15, twins)
        slope = rng.uniform(0.80, 1.50, twins)
        coupling = rng.uniform(-0.25, 0.25, twins)
        drag = rng.uniform(0.20, 0.60, twins)
    elif parameter_range == "ood":
        gain = rng.uniform(1.05, 1.35, twins)
        bias = rng.uniform(-0.25, 0.25, twins)
        slope = rng.uniform(1.80, 2.50, twins)
        coupling = rng.choice((-1.0, 1.0), twins) * rng.uniform(0.35, 0.65, twins)
        drag = rng.uniform(0.80, 1.20, twins)
    elif parameter_range == "held":
        gain = rng.uniform(0.75, 1.15, twins)
        bias = rng.uniform(-0.20, 0.20, twins)
        slope = rng.uniform(1.00, 2.00, twins)
        coupling = rng.choice((-1.0, 1.0), twins) * rng.uniform(0.25, 0.55, twins)
        drag = rng.uniform(0.20, 0.50, twins)
    else:
        raise ValueError(f"unknown parameter range: {parameter_range}")
    return {
        "gain": gain,
        "bias": bias,
        "slope": slope,
        "coupling": coupling,
        "drag": drag,
        "deadzone": rng.uniform(0.15, 0.35, twins),
        "cap": rng.uniform(0.35, 0.65, twins),
    }


def _response(
    actions: np.ndarray,
    orientation: np.ndarray,
    family: np.ndarray,
    parameters: dict[str, np.ndarray],
) -> np.ndarray:
    """Evaluate train smooth/rational laws or the structure-held piecewise law."""
    action = np.asarray(actions, dtype=np.float64)
    orientation = np.asarray(orientation, dtype=np.float64)
    w = np.stack((np.cos(orientation), np.sin(orientation)), axis=-1)
    perpendicular = np.stack((-np.sin(orientation), np.cos(orientation)), axis=-1)
    while w.ndim < action.ndim:
        w = np.expand_dims(w, axis=-2)
        perpendicular = np.expand_dims(perpendicular, axis=-2)
    z = np.sum(action * w, axis=-1)
    p = np.sum(action * perpendicular, axis=-1)
    radius2 = np.sum(np.square(action), axis=-1)

    def expand(value: np.ndarray) -> np.ndarray:
        output = np.asarray(value, dtype=np.float64)
        while output.ndim < z.ndim:
            output = np.expand_dims(output, axis=-1)
        return output

    gain = expand(parameters["gain"])
    bias = expand(parameters["bias"])
    slope = expand(parameters["slope"])
    coupling = expand(parameters["coupling"])
    drag = expand(parameters["drag"])
    deadzone = expand(parameters["deadzone"])
    cap = expand(parameters["cap"])
    family_expanded = np.asarray(family)
    while family_expanded.ndim < z.ndim:
        family_expanded = np.expand_dims(family_expanded, axis=-1)

    smooth = bias + gain * np.tanh(slope * z + coupling * z * p)
    rational = bias + gain * (z + coupling * z * p) / (1.0 + drag * radius2)
    raw = z + coupling * z * p + 0.15 * np.sign(p) * np.square(z)
    activated = np.sign(raw) * np.maximum(np.abs(raw) - deadzone, 0.0)
    held = bias + gain * np.clip(activated, -cap, cap)
    return np.where(family_expanded == 0, smooth, np.where(family_expanded == 1, rational, held))


def _sample_history_actions(
    twins: int,
    min_history: int,
    max_history: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lengths = rng.integers(min_history, max_history + 1, size=twins)
    actions = np.zeros((twins, MAX_HISTORY, 2), dtype=np.float64)
    mask = np.zeros((twins, MAX_HISTORY), dtype=bool)
    centers = rng.uniform(-np.pi, np.pi, size=twins)
    for twin, length in enumerate(lengths):
        concentrated = int(round(0.70 * int(length)))
        angle = np.concatenate(
            (
                centers[twin] + rng.normal(0.0, 0.45, concentrated),
                rng.uniform(-np.pi, np.pi, int(length) - concentrated),
            )
        )
        radius = 0.15 + 0.85 * rng.beta(2.0, 2.0, int(length))
        order = rng.permutation(int(length))
        actions[twin, :length] = (
            radius[:, None] * np.stack((np.cos(angle), np.sin(angle)), axis=-1)
        )[order]
        mask[twin, :length] = True
    return actions, mask, lengths


def _sample_candidates(
    history_actions: np.ndarray,
    history_mask: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    twins = len(history_actions)
    output = np.empty((twins, CANDIDATES, 2), dtype=np.float64)
    for twin in range(twins):
        history = history_actions[twin, history_mask[twin]]
        for slot in range(CANDIDATES):
            for _ in range(1000):
                angle = rng.uniform(-np.pi, np.pi)
                radius = 0.15 + 0.85 * math.sqrt(float(rng.uniform()))
                candidate = radius * np.asarray((math.cos(angle), math.sin(angle)))
                if np.min(np.linalg.norm(history - candidate, axis=1)) > 0.03:
                    output[twin, slot] = candidate
                    break
            else:
                raise RuntimeError("candidate non-collision rejection sampler exhausted")
    return output


def make_dataset(
    twins: int,
    seed: int,
    *,
    mechanism_kind: str,
    parameter_range: str,
    history_range: tuple[int, int],
    noise_scale: float,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    family = _mechanism_codes(mechanism_kind, twins, rng)
    parameters = _parameters(twins, parameter_range, rng)
    orientation0 = rng.uniform(-np.pi, np.pi, twins)
    separation = rng.uniform(np.deg2rad(50.0), np.deg2rad(130.0), twins)
    orientation = np.stack((orientation0, orientation0 + separation), axis=1)
    actions, mask, lengths = _sample_history_actions(
        twins, history_range[0], history_range[1], rng
    )
    history_actions = np.repeat(actions[:, None, :, :], 2, axis=1)
    result = _response(
        history_actions,
        orientation,
        family,
        parameters,
    )
    valid = np.repeat(mask[:, None, :], 2, axis=1)
    if noise_scale > 0.0:
        radius = np.linalg.norm(history_actions, axis=-1)
        noise = rng.normal(size=result.shape) * noise_scale * (0.4 + 0.8 * radius)
        result = result + noise
    result = np.where(valid, result, 0.0)

    candidates = _sample_candidates(actions, mask, rng)
    candidate_by_branch = np.repeat(candidates[:, None, :, :], 2, axis=1)
    displacement = _response(
        candidate_by_branch,
        orientation,
        family,
        parameters,
    )
    target = rng.uniform(-0.50, 0.50, twins)

    swaps = np.asarray(([False, True] * ((twins + 1) // 2))[:twins], dtype=bool)
    rng.shuffle(swaps)
    result[swaps] = result[swaps, ::-1]
    displacement[swaps] = displacement[swaps, ::-1]
    orientation[swaps] = orientation[swaps, ::-1]
    utility = -np.abs(target[:, None, None] - displacement)
    return {
        "family": family,
        "orientation": orientation,
        "branch_swapped": swaps,
        "history_actions": history_actions,
        "history_results": result,
        "history_mask": mask,
        "history_length": lengths,
        "candidates": candidates,
        "target": target,
        "displacement": displacement,
        "utility": utility,
    }


def _canonical_order(
    parallel: np.ndarray, perpendicular: np.ndarray, radius: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    angle = np.mod(np.arctan2(perpendicular, parallel) + np.pi, 2.0 * np.pi)
    key = angle + 1.0e-4 * radius
    valid = mask[:, None, None, :]
    return np.argsort(np.where(valid, key, 100.0), axis=3, kind="mergesort")


def make_features(dataset: dict[str, np.ndarray], mode: str) -> np.ndarray:
    if mode not in LEARNED_MODES:
        raise ValueError(f"unknown feature mode: {mode}")
    actions = dataset["history_actions"]
    results = dataset["history_results"]
    mask = dataset["history_mask"]
    candidates = dataset["candidates"]
    candidate_radius = np.maximum(np.linalg.norm(candidates, axis=-1), 1.0e-12)
    candidate_unit = candidates / candidate_radius[..., None]
    candidate_perpendicular = np.stack(
        (-candidate_unit[..., 1], candidate_unit[..., 0]), axis=-1
    )
    parallel = np.einsum("nbhd,nkd->nbkh", actions, candidate_unit)
    perpendicular = np.einsum("nbhd,nkd->nbkh", actions, candidate_perpendicular)
    radius = np.linalg.norm(actions, axis=-1)[:, :, None, :]
    radius = np.broadcast_to(radius, parallel.shape)
    order = _canonical_order(parallel, perpendicular, radius, mask)

    def gather(value: np.ndarray) -> np.ndarray:
        return np.take_along_axis(value, order, axis=3)

    parallel = gather(parallel)
    perpendicular = gather(perpendicular)
    radius = gather(radius)
    gathered_mask = gather(
        np.broadcast_to(mask[:, None, None, :], parallel.shape)
    ).astype(np.float64)
    paired_results = gather(
        np.broadcast_to(results[:, :, None, :], parallel.shape)
    )

    masked_results = np.where(mask[:, None, :], results, np.inf)
    sorted_results = np.sort(masked_results, axis=2)
    sorted_results = np.where(np.isfinite(sorted_results), sorted_results, 0.0)
    sorted_results = np.broadcast_to(sorted_results[:, :, None, :], parallel.shape)

    if mode == "paired_raw":
        result_channel = paired_results
    elif mode in ("unbound_raw", "result_only"):
        result_channel = sorted_results
    else:
        result_channel = np.zeros_like(paired_results)

    if mode in ("result_only", "context_only"):
        parallel = np.zeros_like(parallel)
        perpendicular = np.zeros_like(perpendicular)
        radius = np.zeros_like(radius)
    if mode == "context_only":
        result_channel = np.zeros_like(result_channel)
        gathered_mask = np.zeros_like(gathered_mask)

    tokens = np.stack(
        (parallel, perpendicular, radius, result_channel, gathered_mask), axis=-1
    )
    context = np.stack(
        (
            candidates[..., 0],
            candidates[..., 1],
            np.broadcast_to(dataset["target"][:, None], candidates.shape[:2]),
            np.broadcast_to(
                dataset["history_length"][:, None] / MAX_HISTORY,
                candidates.shape[:2],
            ),
        ),
        axis=-1,
    )
    context = np.broadcast_to(context[:, None, :, :], tokens.shape[:3] + (4,))
    return np.concatenate((tokens.reshape(*tokens.shape[:3], -1), context), axis=-1).astype(
        np.float32
    )


def _new_model(seed: int) -> ExtraTreesRegressor:
    return ExtraTreesRegressor(
        n_estimators=TREES,
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_LEAF,
        max_features=MAX_FEATURES,
        bootstrap=False,
        random_state=seed,
        n_jobs=-1,
    )


def fit_models(dataset: dict[str, np.ndarray], seed: int) -> dict[str, ExtraTreesRegressor]:
    target = dataset["displacement"].reshape(-1)
    models = {}
    for offset, mode in enumerate(LEARNED_MODES):
        features = make_features(dataset, mode)
        model = _new_model(seed + 97 * offset)
        model.fit(features.reshape(-1, features.shape[-1]), target)
        models[mode] = model
    return models


def predict_model(
    model: ExtraTreesRegressor, dataset: dict[str, np.ndarray], mode: str
) -> np.ndarray:
    features = make_features(dataset, mode)
    return model.predict(features.reshape(-1, features.shape[-1])).reshape(features.shape[:3])


def _poly_design(actions: np.ndarray) -> np.ndarray:
    x = actions[..., 0]
    y = actions[..., 1]
    return np.stack((np.ones_like(x), x, y, x * x, x * y, y * y), axis=-1)


def nonlearned_predictions(dataset: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    actions = dataset["history_actions"]
    results = dataset["history_results"]
    mask = dataset["history_mask"]
    candidates = dataset["candidates"]
    twins, branches = results.shape[:2]
    output = {
        name: np.empty((twins, branches, CANDIDATES), dtype=np.float64)
        for name in ("result_mean", "nearest", "knn5", "poly2_ridge", "rbf_ridge")
    }
    for twin in range(twins):
        valid = mask[twin]
        query = candidates[twin]
        for branch in range(branches):
            history_action = actions[twin, branch, valid]
            history_result = results[twin, branch, valid]
            output["result_mean"][twin, branch] = float(np.mean(history_result))

            distance = np.linalg.norm(
                query[:, None, :] - history_action[None, :, :], axis=-1
            )
            order = np.argsort(distance, axis=1, kind="mergesort")
            output["nearest"][twin, branch] = history_result[order[:, 0]]
            selected = order[:, : min(5, len(history_action))]
            selected_distance = np.take_along_axis(distance, selected, axis=1)
            weight = 1.0 / np.maximum(selected_distance, 1.0e-5)
            values = history_result[selected]
            output["knn5"][twin, branch] = np.sum(weight * values, axis=1) / np.sum(
                weight, axis=1
            )

            design = _poly_design(history_action)
            regularized = design.T @ design + POLY_REGULARIZATION * np.eye(design.shape[1])
            coefficient = np.linalg.solve(regularized, design.T @ history_result)
            output["poly2_ridge"][twin, branch] = _poly_design(query) @ coefficient

            squared = np.sum(
                np.square(history_action[:, None, :] - history_action[None, :, :]),
                axis=-1,
            )
            kernel = np.exp(-squared / (2.0 * RBF_LENGTH_SCALE**2))
            alpha = np.linalg.solve(
                kernel + RBF_REGULARIZATION * np.eye(len(history_action)),
                history_result,
            )
            query_squared = np.sum(
                np.square(query[:, None, :] - history_action[None, :, :]), axis=-1
            )
            output["rbf_ridge"][twin, branch] = (
                np.exp(-query_squared / (2.0 * RBF_LENGTH_SCALE**2)) @ alpha
            )
    return output


def _utility(dataset: dict[str, np.ndarray], displacement: np.ndarray) -> np.ndarray:
    return -np.abs(dataset["target"][:, None, None] - displacement)


def twin_preference(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    true_centered = truth - truth.mean(axis=2, keepdims=True)
    pred_centered = prediction - prediction.mean(axis=2, keepdims=True)
    true_difference = true_centered[:, 1] - true_centered[:, 0]
    pred_difference = pred_centered[:, 1] - pred_centered[:, 0]
    informative = np.abs(true_difference) > 1.0e-12
    tied = np.abs(pred_difference) <= 1.0e-12
    correct = np.full(true_difference.shape, np.nan)
    correct[informative & tied] = 0.5
    decided = informative & ~tied
    correct[decided] = (
        np.sign(true_difference[decided]) == np.sign(pred_difference[decided])
    ).astype(np.float64)
    return {
        "tpa": float(np.nanmean(np.nanmean(correct, axis=1))),
        "informative_fraction": float(np.mean(informative)),
    }


def score_prediction(truth: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    true_best = np.argmax(truth, axis=2)
    predicted_best = np.argmax(prediction, axis=2)
    selected = np.take_along_axis(truth, predicted_best[..., None], axis=2)[..., 0]
    optimal = np.max(truth, axis=2)
    regret = optimal - selected
    span = np.max(truth, axis=2) - np.min(truth, axis=2)
    normalized = regret / np.maximum(span, 1.0e-12)
    top1 = true_best == predicted_best
    by_twin = np.mean(normalized, axis=1)

    ordered = np.sort(truth, axis=2)
    gap = ordered[:, :, -1] - ordered[:, :, -2]
    lower, upper = np.quantile(gap.reshape(-1), (1.0 / 3.0, 2.0 / 3.0))
    difficulty = {}
    for name, selected_mask in (
        ("hard", gap <= lower),
        ("medium", (gap > lower) & (gap <= upper)),
        ("easy", gap > upper),
    ):
        difficulty[name] = {
            "count": int(np.sum(selected_mask)),
            "normalized_regret": float(np.mean(normalized[selected_mask])),
            "top1_accuracy": float(np.mean(top1[selected_mask])),
        }
    preference = twin_preference(truth, prediction)
    return {
        "normalized_regret": float(np.mean(normalized)),
        "mean_regret": float(np.mean(regret)),
        "top1_accuracy": float(np.mean(top1)),
        "tpa": preference["tpa"],
        "informative_fraction": preference["informative_fraction"],
        "normalized_regret_by_twin": by_twin.tolist(),
        "difficulty": difficulty,
    }


def integrity(dataset: dict[str, np.ndarray]) -> dict[str, Any]:
    actions = dataset["history_actions"]
    mask = dataset["history_mask"]
    lengths = dataset["history_length"]
    candidates = dataset["candidates"]
    valid_distance = []
    for twin in range(len(actions)):
        history = actions[twin, 0, mask[twin]]
        valid_distance.append(
            float(
                np.min(
                    np.linalg.norm(
                        candidates[twin, :, None, :] - history[None, :, :], axis=-1
                    )
                )
            )
        )
    arrays = (
        actions, dataset["history_results"], candidates, dataset["target"],
        dataset["displacement"], dataset["utility"],
    )
    checks = {
        "shapes": bool(
            actions.shape == (len(actions), 2, MAX_HISTORY, 2)
            and dataset["history_results"].shape == (len(actions), 2, MAX_HISTORY)
            and candidates.shape == (len(actions), CANDIDATES, 2)
            and dataset["utility"].shape == (len(actions), 2, CANDIDATES)
        ),
        "finite": bool(all(np.all(np.isfinite(value)) for value in arrays)),
        "shared_history_actions": bool(np.array_equal(actions[:, 0], actions[:, 1])),
        "mask_matches_length": bool(np.array_equal(mask.sum(axis=1), lengths)),
        "length_range": bool(np.min(lengths) >= 8 and np.max(lengths) <= MAX_HISTORY),
        "branch_balance": bool(int(np.sum(dataset["branch_swapped"])) * 2 == len(actions)),
        "candidate_noncollision": bool(min(valid_distance) > 0.03),
        "nonzero_utility_span": bool(
            np.all(np.ptp(dataset["utility"], axis=2) > 1.0e-10)
        ),
    }
    result_gap = np.mean(
        np.abs(
            np.sort(dataset["history_results"][:, 0], axis=1)
            - np.sort(dataset["history_results"][:, 1], axis=1)
        )
    )
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "history_min": int(np.min(lengths)),
        "history_max": int(np.max(lengths)),
        "candidate_history_min_distance": float(min(valid_distance)),
        "mean_sorted_result_marginal_gap": float(result_gap),
    }


def _copy_dataset(dataset: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {key: value.copy() for key, value in dataset.items()}


def _pair_permutation(
    dataset: dict[str, np.ndarray], seed: int
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    output = _copy_dataset(dataset)
    for twin, length in enumerate(output["history_length"]):
        permutation = rng.permutation(int(length))
        output["history_actions"][twin, :, :length] = output["history_actions"][
            twin, :, :length
        ][:, permutation, :]
        output["history_results"][twin, :, :length] = output["history_results"][
            twin, :, :length
        ][:, permutation]
    return output


def _binding_break(
    dataset: dict[str, np.ndarray], seed: int
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    output = _copy_dataset(dataset)
    for twin, length in enumerate(output["history_length"]):
        for branch in range(2):
            permutation = rng.permutation(int(length))
            output["history_results"][twin, branch, :length] = output[
                "history_results"
            ][twin, branch, permutation]
    return output


def _twin_history_swap(dataset: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    output = _copy_dataset(dataset)
    output["history_actions"] = output["history_actions"][:, ::-1].copy()
    output["history_results"] = output["history_results"][:, ::-1].copy()
    return output


def _slot_reversal(dataset: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    output = _copy_dataset(dataset)
    output["candidates"] = output["candidates"][:, ::-1].copy()
    output["displacement"] = output["displacement"][:, :, ::-1].copy()
    output["utility"] = output["utility"][:, :, ::-1].copy()
    return output


def evaluate_split(
    dataset: dict[str, np.ndarray],
    models: dict[str, ExtraTreesRegressor],
    intervention_seed: int,
) -> dict[str, Any]:
    truth = dataset["utility"]
    scores: dict[str, Any] = {}
    for mode in LEARNED_MODES:
        displacement = predict_model(models[mode], dataset, mode)
        scores[mode] = score_prediction(truth, _utility(dataset, displacement))
    for name, displacement in nonlearned_predictions(dataset).items():
        scores[name] = score_prediction(truth, _utility(dataset, displacement))

    intervention_datasets = {
        "pair_preserving": _pair_permutation(dataset, intervention_seed + 1),
        "binding_breaking": _binding_break(dataset, intervention_seed + 2),
        "twin_history_swap": _twin_history_swap(dataset),
        "slot_reversal": _slot_reversal(dataset),
    }
    interventions = {}
    for name, changed in intervention_datasets.items():
        displacement = predict_model(models["paired_raw"], changed, "paired_raw")
        interventions[name] = score_prediction(
            changed["utility"], _utility(changed, displacement)
        )
    return {
        "integrity": integrity(dataset),
        "scores": scores,
        "interventions": interventions,
    }


def _datasets_for_seed(
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, np.ndarray]]]:
    train = make_dataset(
        TRAIN_TWINS,
        seed + 10_000,
        mechanism_kind="balanced",
        parameter_range="train",
        history_range=(14, 28),
        noise_scale=0.0,
    )
    evaluation = {
        "id": make_dataset(
            EVAL_TWINS,
            seed + 20_000,
            mechanism_kind="balanced",
            parameter_range="train",
            history_range=(14, 28),
            noise_scale=0.0,
        ),
        "parameter_ood": make_dataset(
            EVAL_TWINS,
            seed + 30_000,
            mechanism_kind="balanced",
            parameter_range="ood",
            history_range=(14, 28),
            noise_scale=0.0,
        ),
        "noise_ood": make_dataset(
            EVAL_TWINS,
            seed + 40_000,
            mechanism_kind="balanced",
            parameter_range="train",
            history_range=(10, 22),
            noise_scale=0.040,
        ),
        "held_structure": make_dataset(
            EVAL_TWINS,
            seed + 50_000,
            mechanism_kind="held",
            parameter_range="held",
            history_range=(10, 22),
            noise_scale=0.0,
        ),
    }
    return train, evaluation


def run_seed(seed: int) -> dict[str, Any]:
    started = time.perf_counter()
    train, evaluation = _datasets_for_seed(seed)
    models = fit_models(train, seed)
    splits = {
        split: evaluate_split(
            evaluation[split], models, seed + 100_000 + index * 1000
        )
        for index, split in enumerate(SPLITS)
    }
    return {
        "seed": seed,
        "train_integrity": integrity(train),
        "splits": splits,
        "runtime_seconds": time.perf_counter() - started,
    }


def _mean(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _metric_summary(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    scalar_metrics = (
        "normalized_regret", "mean_regret", "top1_accuracy", "tpa",
        "informative_fraction",
    )
    output = {}
    for metric in scalar_metrics:
        values = np.asarray([row[name][metric] for row in rows], dtype=np.float64)
        output[metric] = float(np.mean(values))
        output[f"{metric}_seed_std"] = float(np.std(values, ddof=1))
    output["difficulty"] = {
        level: {
            metric: _mean([row[name]["difficulty"][level][metric] for row in rows])
            for metric in ("normalized_regret", "top1_accuracy")
        }
        for level in ("hard", "medium", "easy")
    }
    return output


def _hierarchical_bootstrap(
    rows: list[dict[str, Any]], baseline: str, rng_seed: int
) -> dict[str, float]:
    rng = np.random.default_rng(rng_seed)
    differences = []
    for _ in range(BOOTSTRAP_REPLICATES):
        selected_seeds = rng.integers(0, len(rows), size=len(rows))
        seed_means = []
        for seed_index in selected_seeds:
            paired = np.asarray(
                rows[int(seed_index)]["scores"]["paired_raw"][
                    "normalized_regret_by_twin"
                ],
                dtype=np.float64,
            )
            comparison = np.asarray(
                rows[int(seed_index)]["scores"][baseline][
                    "normalized_regret_by_twin"
                ],
                dtype=np.float64,
            )
            indices = rng.integers(0, len(paired), size=len(paired))
            seed_means.append(float(np.mean(comparison[indices] - paired[indices])))
        differences.append(float(np.mean(seed_means)))
    lower, upper = np.quantile(differences, (0.025, 0.975))
    return {
        "replicates": BOOTSTRAP_REPLICATES,
        "lower_95": float(lower),
        "median": float(np.median(differences)),
        "upper_95": float(upper),
    }


def aggregate(seed_results: list[dict[str, Any]]) -> dict[str, Any]:
    split_summaries: dict[str, Any] = {}
    for split_index, split in enumerate(SPLITS):
        rows = [result["splits"][split] for result in seed_results]
        score_names = tuple(rows[0]["scores"])
        scores = {
            name: _metric_summary([row["scores"] for row in rows], name)
            for name in score_names
        }
        intervention_names = tuple(rows[0]["interventions"])
        interventions = {
            name: _metric_summary([row["interventions"] for row in rows], name)
            for name in intervention_names
        }
        strongest_baseline = min(
            BASELINE_NAMES, key=lambda name: scores[name]["normalized_regret"]
        )
        paired_regret = scores["paired_raw"]["normalized_regret"]
        baseline_regret = scores[strongest_baseline]["normalized_regret"]
        relative_improvement = (
            (baseline_regret - paired_regret) / max(baseline_regret, 1.0e-12)
        )
        seed_improvements = [
            row["scores"][strongest_baseline]["normalized_regret"]
            - row["scores"]["paired_raw"]["normalized_regret"]
            for row in rows
        ]
        seed_wins = int(np.sum(np.asarray(seed_improvements) > 0.0))
        bootstrap = _hierarchical_bootstrap(
            rows, strongest_baseline, 880_000 + split_index
        )
        paired_tpa = scores["paired_raw"]["tpa"]
        binding_regret_increase = (
            interventions["binding_breaking"]["normalized_regret"] - paired_regret
        )
        binding_tpa_drop = paired_tpa - interventions["binding_breaking"]["tpa"]
        checks = {
            "structural_integrity": all(row["integrity"]["passed"] for row in rows),
            "regret_relative_improvement_at_least_0_10": relative_improvement >= 0.10,
            "seed_wins_at_least_4": seed_wins >= 4,
            "bootstrap_lower_positive": bootstrap["lower_95"] > 0.0,
            "top1_gain_at_least_0_03": (
                scores["paired_raw"]["top1_accuracy"]
                - scores[strongest_baseline]["top1_accuracy"] >= 0.03
            ),
            "binding_regret_increase_at_least_0_05": binding_regret_increase >= 0.05,
            "binding_tpa_drop_at_least_0_05": binding_tpa_drop >= 0.05,
            "pair_regret_change_at_most_0_005": (
                abs(interventions["pair_preserving"]["normalized_regret"] - paired_regret)
                <= 0.005
            ),
            "pair_tpa_change_at_most_0_005": (
                abs(interventions["pair_preserving"]["tpa"] - paired_tpa) <= 0.005
            ),
            "twin_swap_regret_increase_at_least_0_05": (
                interventions["twin_history_swap"]["normalized_regret"] - paired_regret
                >= 0.05
            ),
            "slot_regret_change_at_most_0_005": (
                abs(interventions["slot_reversal"]["normalized_regret"] - paired_regret)
                <= 0.005
            ),
            "anti_saturation_tpa_below_0_95": paired_tpa < 0.95,
            "anti_saturation_regret_at_least_0_01": paired_regret >= 0.01,
        }
        split_summaries[split] = {
            "scores": scores,
            "interventions": interventions,
            "strongest_baseline": strongest_baseline,
            "relative_regret_improvement": float(relative_improvement),
            "seed_regret_improvements": [float(value) for value in seed_improvements],
            "seed_wins": seed_wins,
            "bootstrap_regret_improvement": bootstrap,
            "binding_regret_increase": float(binding_regret_increase),
            "binding_tpa_drop": float(binding_tpa_drop),
            "checks": checks,
            "passed": bool(all(checks.values())),
        }

    training_integrity = all(result["train_integrity"]["passed"] for result in seed_results)
    passed = training_integrity and all(row["passed"] for row in split_summaries.values())
    return {
        "decision": "CPU_V2_NONTRIVIAL_SIGNAL" if passed else "CPU_V2_NO_GO",
        "passed": passed,
        "training_integrity": training_integrity,
        "splits": split_summaries,
    }


def run() -> dict[str, Any]:
    started = time.perf_counter()
    preregister_bytes = PREREGISTER.read_bytes()
    seed_results = [run_seed(seed) for seed in SEEDS]
    return {
        "schema": "actmask-binding-cpu-v2-antisaturation",
        "preregister": str(PREREGISTER).replace("\\", "/"),
        "preregister_sha256": hashlib.sha256(preregister_bytes).hexdigest(),
        "cpu_only": True,
        "seeds": list(SEEDS),
        "configuration": {
            "train_twins": TRAIN_TWINS,
            "eval_twins": EVAL_TWINS,
            "candidates": CANDIDATES,
            "max_history": MAX_HISTORY,
            "trees": TREES,
            "max_depth": MAX_DEPTH,
            "min_leaf": MIN_LEAF,
            "max_features": MAX_FEATURES,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        },
        "seed_results": seed_results,
        "aggregate": aggregate(seed_results),
        "runtime_seconds": time.perf_counter() - started,
    }


def _console_summary(result: dict[str, Any]) -> dict[str, Any]:
    splits = {}
    for name, row in result["aggregate"]["splits"].items():
        baseline = row["strongest_baseline"]
        splits[name] = {
            "passed": row["passed"],
            "paired_regret": row["scores"]["paired_raw"]["normalized_regret"],
            "paired_top1": row["scores"]["paired_raw"]["top1_accuracy"],
            "paired_tpa": row["scores"]["paired_raw"]["tpa"],
            "strongest_baseline": baseline,
            "baseline_regret": row["scores"][baseline]["normalized_regret"],
            "relative_regret_improvement": row["relative_regret_improvement"],
            "bootstrap_lower": row["bootstrap_regret_improvement"]["lower_95"],
            "failed_checks": [key for key, value in row["checks"].items() if not value],
        }
    return {
        "decision": result["aggregate"]["decision"],
        "runtime_seconds": result["runtime_seconds"],
        "splits": splits,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("audit/results/binding_cpu_v2_antisaturation.json"),
    )
    args = parser.parse_args()
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(_console_summary(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
