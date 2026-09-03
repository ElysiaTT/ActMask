"""Frozen CPU learned transport test for action-effect binding.

The protocol is frozen in ``docs/binding_cpu_learned_v1_preregister.md``.
This module intentionally imports neither ActMask nor PyTorch.
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
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SEEDS = (9401, 9402, 9403, 9404, 9405)
SPLITS = ("id", "parameter_ood", "noise_ood", "held_mechanism")
PROBE_DIRECTIONS = np.deg2rad(np.arange(0.0, 360.0, 22.5))
PROBE_RADII = np.asarray((0.20, 0.50, 0.80), dtype=np.float64)
CANDIDATE_DIRECTIONS = PROBE_DIRECTIONS + np.deg2rad(11.25)
CANDIDATE_RADII = np.asarray((0.35, 0.65), dtype=np.float64)
PHASE_STEP = float(np.deg2rad(22.5))
GAIN = 0.060
GOAL = 0.030
RESULT_RESOLUTION = 0.001
RIDGE_ALPHA = 1.0e-3
TRAIN_TWINS_PER_MECHANISM = 512
EVAL_TWINS = 256
NOISE_STD = 0.002
PREREGISTER = Path("docs/binding_cpu_learned_v1_preregister.md")


def _rotate_vectors(value: np.ndarray, radians: float) -> np.ndarray:
    matrix = np.asarray(
        ((math.cos(radians), -math.sin(radians)),
         (math.sin(radians), math.cos(radians))),
        dtype=np.float64,
    )
    return np.asarray(value) @ matrix.T


def _action_basis(actions: np.ndarray) -> np.ndarray:
    """Radial Fourier basis frozen by the preregistration."""
    x = np.asarray(actions, dtype=np.float64)[..., 0]
    y = np.asarray(actions, dtype=np.float64)[..., 1]
    radius = np.sqrt(np.square(x) + np.square(y))
    angle = np.arctan2(y, x)
    columns = []
    for harmonic in (2, 4):
        columns.extend(
            (
                radius * np.cos(harmonic * angle),
                radius * np.sin(harmonic * angle),
                np.square(radius) * np.cos(harmonic * angle),
                np.square(radius) * np.sin(harmonic * angle),
            )
        )
    return np.stack(columns, axis=-1)


def _component(
    actions: np.ndarray,
    phi: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    harmonic: int,
) -> np.ndarray:
    radius = np.linalg.norm(actions, axis=-1)
    angle = np.arctan2(actions[..., 1], actions[..., 0])
    radial = alpha * radius - beta * np.square(radius)
    return GAIN * radial * np.cos(harmonic * (angle - phi))


def _response(
    actions: np.ndarray,
    phi: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    mechanism: np.ndarray,
) -> np.ndarray:
    """Evaluate h2, h4, or the held additive h2+h4 mechanism."""
    h2 = _component(actions, phi, alpha, beta, 2)
    h4 = _component(actions, phi, alpha, beta, 4)
    mechanism = np.asarray(mechanism)
    return np.where(
        mechanism == 0,
        h2,
        np.where(mechanism == 1, h4, 0.65 * h2 + 0.55 * h4),
    )


def _mechanisms(kind: str, twins: int) -> np.ndarray:
    if kind == "h2":
        return np.zeros(twins, dtype=np.int8)
    if kind == "h4":
        return np.ones(twins, dtype=np.int8)
    if kind == "held":
        return np.full(twins, 2, dtype=np.int8)
    if kind == "balanced":
        return np.asarray(([0, 1] * ((twins + 1) // 2))[:twins], dtype=np.int8)
    raise ValueError(f"unknown mechanism kind: {kind}")


def _probe_actions(rotation: np.ndarray) -> np.ndarray:
    blocks = []
    for radius in PROBE_RADII:
        angle = rotation[:, None] + PROBE_DIRECTIONS[None, :]
        blocks.append(radius * np.stack((np.cos(angle), np.sin(angle)), axis=-1))
    return np.concatenate(blocks, axis=1)


def _candidate_actions(rotation: np.ndarray, schedule_seed: int) -> tuple[np.ndarray, np.ndarray]:
    base_rows = []
    for radius in CANDIDATE_RADII:
        angle = rotation[:, None] + CANDIDATE_DIRECTIONS[None, :]
        base_rows.append(radius * np.stack((np.cos(angle), np.sin(angle)), axis=-1))
    base = np.concatenate(base_rows, axis=1)
    candidates = np.empty_like(base)
    template_id = np.empty(base.shape[:2], dtype=np.int16)
    count = base.shape[1]
    for twin in range(len(rotation)):
        shift = int((schedule_seed + twin) % count)
        order = (np.arange(count, dtype=np.int16) + shift) % count
        candidates[twin] = base[twin, order]
        template_id[twin] = order
    return candidates, template_id


def make_dataset(
    twins: int,
    seed: int,
    mechanism_kind: str,
    coefficient_range: str,
    noise_std: float = 0.0,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    rotation = rng.uniform(-np.pi, np.pi, size=twins)
    phase0 = rotation + rng.uniform(-np.pi, np.pi, size=twins)
    phase = np.stack((phase0, phase0 + PHASE_STEP), axis=1)
    mechanism = _mechanisms(mechanism_kind, twins)
    if mechanism_kind == "balanced":
        rng.shuffle(mechanism)

    if coefficient_range == "train":
        alpha = rng.uniform(0.90, 1.20, size=twins)
        beta = rng.uniform(0.10, 0.30, size=twins)
    elif coefficient_range == "ood":
        alpha = rng.uniform(1.30, 1.60, size=twins)
        beta = rng.uniform(0.35, 0.55, size=twins)
    else:
        raise ValueError(f"unknown coefficient range: {coefficient_range}")

    probe = _probe_actions(rotation)
    probe_actions = np.repeat(probe[:, None, :, :], 2, axis=1)
    radius_count = len(PROBE_RADII)
    direction_count = len(PROBE_DIRECTIONS)

    # Generate branch zero once and construct branch one by the exact phase-grid
    # permutation. Noise follows the physical outcome rather than the action slot,
    # preserving the result multiset exactly.
    result0 = _response(
        probe,
        phase[:, 0, None],
        alpha[:, None],
        beta[:, None],
        mechanism[:, None],
    ).reshape(twins, radius_count, direction_count)
    if noise_std > 0.0:
        result0 = result0 + rng.normal(0.0, noise_std, size=result0.shape)
    result1 = np.roll(result0, 1, axis=2)
    probe_results = np.stack((result0, result1), axis=1).reshape(twins, 2, -1)
    probe_results = (
        np.round(probe_results / RESULT_RESOLUTION) * RESULT_RESOLUTION
    ).astype(np.float64)

    candidates, template_id = _candidate_actions(rotation, seed + 700_001)
    candidate_by_branch = np.repeat(candidates[:, None, :, :], 2, axis=1)
    displacement = _response(
        candidate_by_branch,
        phase[:, :, None],
        alpha[:, None, None],
        beta[:, None, None],
        mechanism[:, None, None],
    )

    swaps = np.asarray(([False, True] * ((twins + 1) // 2))[:twins], dtype=bool)
    rng.shuffle(swaps)
    probe_results[swaps] = probe_results[swaps, ::-1]
    displacement[swaps] = displacement[swaps, ::-1]
    phase[swaps] = phase[swaps, ::-1]

    utility = -np.abs(GOAL - displacement)
    return {
        "rotation": rotation,
        "phase": phase,
        "alpha": alpha,
        "beta": beta,
        "mechanism": mechanism,
        "branch_swapped": swaps,
        "probe_actions": probe_actions,
        "probe_results": probe_results,
        "candidate_actions": candidates,
        "candidate_template_id": template_id,
        "displacement": displacement,
        "utility": utility,
    }


def _concatenate(rows: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    keys = rows[0].keys()
    return {key: np.concatenate([row[key] for row in rows], axis=0) for key in keys}


def _erase_binding(results: np.ndarray) -> np.ndarray:
    shape = results.shape
    grouped = np.asarray(results).reshape(
        shape[0], shape[1], len(PROBE_RADII), len(PROBE_DIRECTIONS)
    )
    return np.sort(grouped, axis=3).reshape(shape).copy()


def _pair_preserving(
    actions: np.ndarray, results: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    action_shape = actions.shape
    result_shape = results.shape
    grouped_actions = np.asarray(actions).reshape(
        action_shape[0], action_shape[1], len(PROBE_RADII),
        len(PROBE_DIRECTIONS), action_shape[-1]
    )
    grouped_results = np.asarray(results).reshape(
        result_shape[0], result_shape[1], len(PROBE_RADII), len(PROBE_DIRECTIONS)
    )
    return (
        np.roll(grouped_actions, 3, axis=3).reshape(action_shape).copy(),
        np.roll(grouped_results, 3, axis=3).reshape(result_shape).copy(),
    )


def _history_coefficients(actions: np.ndarray, results: np.ndarray) -> np.ndarray:
    twins, branches = actions.shape[:2]
    output = np.empty((twins, branches, 8), dtype=np.float64)
    for twin in range(twins):
        for branch in range(branches):
            output[twin, branch], *_ = np.linalg.lstsq(
                _action_basis(actions[twin, branch]),
                results[twin, branch],
                rcond=None,
            )
    return output


def _interaction_features(
    coefficients: np.ndarray, candidates: np.ndarray
) -> np.ndarray:
    candidate_basis = _action_basis(candidates)
    return coefficients[:, :, None, :] * candidate_basis[:, None, :, :]


def _fit_ridge(
    dataset: dict[str, np.ndarray], *, erase_binding: bool
) -> Any:
    results = dataset["probe_results"]
    if erase_binding:
        results = _erase_binding(results)
    coefficients = _history_coefficients(dataset["probe_actions"], results)
    features = _interaction_features(coefficients, dataset["candidate_actions"])
    model = make_pipeline(
        StandardScaler(),
        Ridge(alpha=RIDGE_ALPHA, fit_intercept=True),
    )
    model.fit(features.reshape(-1, features.shape[-1]), dataset["displacement"].reshape(-1))
    return model


def _predict_displacement(
    model: Any,
    dataset: dict[str, np.ndarray],
    *,
    actions: np.ndarray | None = None,
    results: np.ndarray | None = None,
) -> np.ndarray:
    history_actions = dataset["probe_actions"] if actions is None else actions
    history_results = dataset["probe_results"] if results is None else results
    coefficients = _history_coefficients(history_actions, history_results)
    features = _interaction_features(coefficients, dataset["candidate_actions"])
    return model.predict(features.reshape(-1, features.shape[-1])).reshape(features.shape[:-1])


def _analytic_displacement(
    dataset: dict[str, np.ndarray],
    *,
    actions: np.ndarray | None = None,
    results: np.ndarray | None = None,
) -> np.ndarray:
    history_actions = dataset["probe_actions"] if actions is None else actions
    history_results = dataset["probe_results"] if results is None else results
    coefficients = _history_coefficients(history_actions, history_results)
    candidate_basis = _action_basis(dataset["candidate_actions"])
    return np.einsum("nbd,nkd->nbk", coefficients, candidate_basis)


def _utility(displacement: np.ndarray) -> np.ndarray:
    return -np.abs(GOAL - np.asarray(displacement))


def _baseline_utilities(dataset: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    truth = dataset["utility"]
    twins, branches, candidates = truth.shape
    candidate_actions = dataset["candidate_actions"]
    history_actions = dataset["probe_actions"]
    history_results = dataset["probe_results"]
    output: dict[str, np.ndarray] = {
        "class_current": np.zeros_like(truth),
        "candidate_slot": np.broadcast_to(
            -np.arange(candidates, dtype=np.float64)[None, None, :], truth.shape
        ).copy(),
    }
    candidate_radius = np.linalg.norm(candidate_actions, axis=-1)
    output["action_only"] = _utility(
        np.broadcast_to(GAIN * candidate_radius[:, None, :], truth.shape)
    )
    result_summary = np.mean(np.abs(history_results), axis=2)[:, :, None]
    output["result_only"] = np.broadcast_to(-result_summary, truth.shape).copy()

    history_norm = np.maximum(np.linalg.norm(history_actions, axis=-1), 1.0e-12)
    history_unit = history_actions / history_norm[..., None]
    arrow = np.mean(history_results[..., None] * history_unit, axis=2)
    candidate_norm = np.maximum(candidate_radius, 1.0e-12)
    candidate_unit = candidate_actions / candidate_norm[..., None]
    arrow_displacement = np.einsum(
        "nbi,nki->nbk", arrow, candidate_unit
    ) * candidate_radius[:, None, :]
    output["arrow_action"] = _utility(arrow_displacement)
    return output


def twin_preference(
    truth: np.ndarray,
    prediction: np.ndarray,
    template_id: np.ndarray,
) -> dict[str, Any]:
    true_centered = truth - truth.mean(axis=2, keepdims=True)
    pred_centered = prediction - prediction.mean(axis=2, keepdims=True)
    true_difference = true_centered[:, 1] - true_centered[:, 0]
    pred_difference = pred_centered[:, 1] - pred_centered[:, 0]
    informative = np.abs(true_difference) > 1.0e-12
    tied = np.abs(pred_difference) <= 1.0e-12
    correct = np.full(true_difference.shape, np.nan, dtype=np.float64)
    correct[informative & tied] = 0.5
    decided = informative & ~tied
    correct[decided] = (
        np.sign(true_difference[decided]) == np.sign(pred_difference[decided])
    ).astype(np.float64)
    by_twin = np.nanmean(correct, axis=1)

    per_template = []
    for template in range(template_id.shape[1]):
        values = []
        for twin in range(len(template_id)):
            slot = int(np.flatnonzero(template_id[twin] == template)[0])
            if np.isfinite(correct[twin, slot]):
                values.append(float(correct[twin, slot]))
        per_template.append(float(np.mean(values)) if values else float("nan"))
    return {
        "tpa": float(np.nanmean(by_twin)),
        "informative_fraction": float(np.mean(informative)),
        "per_template": per_template,
    }


def ranking_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    true_best = np.argmax(truth, axis=2)
    pred_best = np.argmax(prediction, axis=2)
    selected = np.take_along_axis(truth, pred_best[..., None], axis=2)[..., 0]
    optimal = np.max(truth, axis=2)
    regret = optimal - selected
    span = np.max(truth, axis=2) - np.min(truth, axis=2)
    normalized = regret / np.maximum(span, 1.0e-12)
    return {
        "top1_accuracy": float(np.mean(true_best == pred_best)),
        "mean_regret": float(np.mean(regret)),
        "normalized_regret": float(np.mean(normalized)),
    }


def integrity(dataset: dict[str, np.ndarray]) -> dict[str, Any]:
    actions = dataset["probe_actions"]
    results = dataset["probe_results"]
    template_id = dataset["candidate_template_id"]
    truth = dataset["utility"]
    candidates = template_id.shape[1]
    table = np.zeros((candidates, candidates), dtype=np.int64)
    for twin in range(len(template_id)):
        table[template_id[twin], np.arange(candidates)] += 1

    selected_templates = np.stack(
        [
            template_id[
                np.arange(len(template_id)), np.argmax(truth[:, branch], axis=1)
            ]
            for branch in range(2)
        ],
        axis=1,
    )
    checks = {
        "action_multiset_exact": bool(np.array_equal(actions[:, 0], actions[:, 1])),
        "result_multiset_exact": bool(
            np.array_equal(np.sort(results[:, 0], axis=1), np.sort(results[:, 1], axis=1))
        ),
        "finite": bool(
            all(
                np.all(np.isfinite(dataset[key]))
                for key in (
                    "probe_actions", "probe_results", "candidate_actions",
                    "displacement", "utility",
                )
            )
        ),
        "slot_balance_exact": bool(table.min() == table.max()),
        "branch_assignment_balanced": bool(
            int(np.sum(dataset["branch_swapped"])) * 2 == len(template_id)
        ),
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "slot_min": int(table.min()),
        "slot_max": int(table.max()),
        "top1_crossing": float(
            np.mean(selected_templates[:, 0] != selected_templates[:, 1])
        ),
    }


def _copy_dataset(dataset: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {key: value.copy() for key, value in dataset.items()}


def _slot_reversal(dataset: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    output = _copy_dataset(dataset)
    for key in (
        "candidate_actions", "candidate_template_id", "displacement", "utility"
    ):
        axis = 1 if key in ("candidate_actions", "candidate_template_id") else 2
        output[key] = np.flip(output[key], axis=axis).copy()
    return output


def _coordinate_rotation(
    dataset: dict[str, np.ndarray], degrees: float = 37.0
) -> dict[str, np.ndarray]:
    output = _copy_dataset(dataset)
    radians = math.radians(degrees)
    output["probe_actions"] = _rotate_vectors(output["probe_actions"], radians)
    output["candidate_actions"] = _rotate_vectors(output["candidate_actions"], radians)
    output["rotation"] = output["rotation"] + radians
    output["phase"] = output["phase"] + radians
    return output


def _score(
    truth: np.ndarray,
    prediction: np.ndarray,
    template_id: np.ndarray,
) -> dict[str, Any]:
    output = twin_preference(truth, prediction, template_id)
    output.update(ranking_metrics(truth, prediction))
    return output


def evaluate_split(
    dataset: dict[str, np.ndarray], bound_model: Any, unbound_model: Any
) -> dict[str, Any]:
    truth = dataset["utility"]
    template_id = dataset["candidate_template_id"]

    bound = _utility(_predict_displacement(bound_model, dataset))
    erased_results = _erase_binding(dataset["probe_results"])
    unbound = _utility(
        _predict_displacement(unbound_model, dataset, results=erased_results)
    )
    analytic = _utility(_analytic_displacement(dataset))
    predictions: dict[str, np.ndarray] = {
        "bound_ridge": bound,
        "unbound_ridge": unbound,
        "analytic_spectral": analytic,
    }
    predictions.update(_baseline_utilities(dataset))
    scores = {
        name: _score(truth, prediction, template_id)
        for name, prediction in predictions.items()
    }

    pair_actions, pair_results = _pair_preserving(
        dataset["probe_actions"], dataset["probe_results"]
    )
    pair_prediction = _utility(
        _predict_displacement(
            bound_model, dataset, actions=pair_actions, results=pair_results
        )
    )
    erased_prediction = _utility(
        _predict_displacement(bound_model, dataset, results=erased_results)
    )
    swapped_prediction = _utility(
        _predict_displacement(
            bound_model,
            dataset,
            actions=dataset["probe_actions"][:, ::-1],
            results=dataset["probe_results"][:, ::-1],
        )
    )
    slot_dataset = _slot_reversal(dataset)
    slot_prediction = _utility(_predict_displacement(bound_model, slot_dataset))
    rotated_dataset = _coordinate_rotation(dataset)
    rotated_prediction = _utility(_predict_displacement(bound_model, rotated_dataset))
    interventions = {
        "pair_preserving": _score(truth, pair_prediction, template_id),
        "binding_erasure": _score(truth, erased_prediction, template_id),
        "twin_history_swap": _score(truth, swapped_prediction, template_id),
        "slot_reversal": _score(
            slot_dataset["utility"],
            slot_prediction,
            slot_dataset["candidate_template_id"],
        ),
        "coordinate_rotation": _score(
            rotated_dataset["utility"],
            rotated_prediction,
            rotated_dataset["candidate_template_id"],
        ),
    }
    return {
        "integrity": integrity(dataset),
        "scores": scores,
        "interventions": interventions,
    }


def _datasets_for_seed(seed: int) -> tuple[dict[str, np.ndarray], dict[str, dict[str, np.ndarray]]]:
    train = _concatenate(
        [
            make_dataset(
                TRAIN_TWINS_PER_MECHANISM, seed + 10_000, "h2", "train"
            ),
            make_dataset(
                TRAIN_TWINS_PER_MECHANISM, seed + 20_000, "h4", "train"
            ),
        ]
    )
    evaluation = {
        "id": make_dataset(EVAL_TWINS, seed + 30_000, "balanced", "train"),
        "parameter_ood": make_dataset(
            EVAL_TWINS, seed + 40_000, "balanced", "ood"
        ),
        "noise_ood": make_dataset(
            EVAL_TWINS, seed + 50_000, "balanced", "train", NOISE_STD
        ),
        "held_mechanism": make_dataset(
            EVAL_TWINS, seed + 60_000, "held", "train"
        ),
    }
    return train, evaluation


def run_seed(seed: int) -> dict[str, Any]:
    started = time.perf_counter()
    train, evaluations = _datasets_for_seed(seed)
    train_integrity = integrity(train)
    bound_model = _fit_ridge(train, erase_binding=False)
    unbound_model = _fit_ridge(train, erase_binding=True)
    split_results = {
        split: evaluate_split(evaluations[split], bound_model, unbound_model)
        for split in SPLITS
    }
    return {
        "seed": seed,
        "train_integrity": train_integrity,
        "splits": split_results,
        "runtime_seconds": time.perf_counter() - started,
    }


def _mean(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def aggregate(seed_results: list[dict[str, Any]]) -> dict[str, Any]:
    split_summaries: dict[str, Any] = {}
    for split in SPLITS:
        rows = [row["splits"][split] for row in seed_results]
        model_names = tuple(rows[0]["scores"])
        scores = {
            name: {
                metric: _mean([row["scores"][name][metric] for row in rows])
                for metric in ("tpa", "top1_accuracy", "mean_regret", "normalized_regret")
            }
            for name in model_names
        }
        intervention_names = tuple(rows[0]["interventions"])
        interventions = {
            name: {
                metric: _mean(
                    [row["interventions"][name][metric] for row in rows]
                )
                for metric in ("tpa", "top1_accuracy", "mean_regret", "normalized_regret")
            }
            for name in intervention_names
        }
        integrity_passed = all(row["integrity"]["passed"] for row in rows)
        top1_crossing = _mean([row["integrity"]["top1_crossing"] for row in rows])
        bound = scores["bound_ridge"]["tpa"]
        unbound = scores["unbound_ridge"]["tpa"]
        analytic = scores["analytic_spectral"]["tpa"]
        marginal_names = (
            "class_current", "candidate_slot", "action_only", "result_only",
            "arrow_action",
        )
        strongest_marginal = max(scores[name]["tpa"] for name in marginal_names)
        threshold = 0.85 if split == "noise_ood" else 0.90
        seed_floor = 0.80 if split == "noise_ood" else 0.85
        min_seed_bound = min(row["scores"]["bound_ridge"]["tpa"] for row in rows)
        checks = {
            "structural_integrity": integrity_passed,
            "top1_crossing_at_least_0_75": top1_crossing >= 0.75,
            "marginals_at_most_0_60": strongest_marginal <= 0.60,
            "bound_mean_threshold": bound >= threshold,
            "bound_seed_floor": min_seed_bound >= seed_floor,
            "bound_minus_unbound_at_least_0_25": bound - unbound >= 0.25,
            "binding_drop_at_least_0_25": (
                bound - interventions["binding_erasure"]["tpa"] >= 0.25
            ),
            "pair_change_at_most_0_01": (
                abs(bound - interventions["pair_preserving"]["tpa"]) <= 0.01
            ),
            "twin_swap_at_most_0_10": interventions["twin_history_swap"]["tpa"] <= 0.10,
            "slot_threshold_invariant": (
                (bound >= threshold)
                == (interventions["slot_reversal"]["tpa"] >= threshold)
            ),
            "rotation_threshold_invariant": (
                (bound >= threshold)
                == (interventions["coordinate_rotation"]["tpa"] >= threshold)
            ),
            "normalized_regret_at_most_0_20": (
                scores["bound_ridge"]["normalized_regret"] <= 0.20
            ),
            "analytic_at_least_0_90": analytic >= 0.90,
            "bound_within_0_10_of_analytic": bound >= analytic - 0.10,
        }
        split_summaries[split] = {
            "scores": scores,
            "interventions": interventions,
            "top1_crossing": top1_crossing,
            "strongest_marginal": strongest_marginal,
            "min_seed_bound_tpa": min_seed_bound,
            "bound_minus_unbound": bound - unbound,
            "binding_drop": bound - interventions["binding_erasure"]["tpa"],
            "pair_preserving_change": abs(
                bound - interventions["pair_preserving"]["tpa"]
            ),
            "checks": checks,
            "passed": bool(all(checks.values())),
        }

    training_integrity = all(row["train_integrity"]["passed"] for row in seed_results)
    passed = training_integrity and all(row["passed"] for row in split_summaries.values())
    return {
        "decision": (
            "LEARNED_CPU_AUDIT_TRANSPORTS"
            if passed else "LEARNED_CPU_AUDIT_NOT_TRANSPORTED"
        ),
        "passed": passed,
        "training_integrity": training_integrity,
        "splits": split_summaries,
    }


def run() -> dict[str, Any]:
    started = time.perf_counter()
    preregister_bytes = PREREGISTER.read_bytes()
    seed_results = [run_seed(seed) for seed in SEEDS]
    return {
        "schema": "actmask-binding-cpu-learned-v1",
        "preregister": str(PREREGISTER).replace("\\", "/"),
        "preregister_sha256": hashlib.sha256(preregister_bytes).hexdigest(),
        "cpu_only": True,
        "seeds": list(SEEDS),
        "configuration": {
            "train_twins_per_mechanism": TRAIN_TWINS_PER_MECHANISM,
            "eval_twins_per_split": EVAL_TWINS,
            "ridge_alpha": RIDGE_ALPHA,
            "noise_std": NOISE_STD,
            "probe_directions": len(PROBE_DIRECTIONS),
            "probe_radii": PROBE_RADII.tolist(),
            "candidate_count": len(CANDIDATE_DIRECTIONS) * len(CANDIDATE_RADII),
        },
        "seed_results": seed_results,
        "aggregate": aggregate(seed_results),
        "runtime_seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("audit/results/binding_cpu_learned_v1.json"),
    )
    args = parser.parse_args()
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    compact = {
        "decision": result["aggregate"]["decision"],
        "runtime_seconds": result["runtime_seconds"],
        "splits": result["aggregate"]["splits"],
    }
    print(json.dumps(compact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
