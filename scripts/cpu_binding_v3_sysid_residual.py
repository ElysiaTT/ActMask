"""BindingCheck CPU-v3: strong-baseline-first hybrid verifier.

Frozen protocol: docs/binding_cpu_v3_sysid_residual_preregister.md
Standalone NumPy/scikit-learn implementation; no ActMask/PyTorch/GPU import.
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
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesRegressor


SEEDS = (12001, 12002, 12003, 12004, 12005)
SPLITS = ("id", "parameter_ood", "sparse_history", "noise_ood", "held_structure")
COMPONENTS = ("nearest", "knn5", "local_linear", "poly2_ridge", "rbf_ridge")
RAW_MODES = ("raw_paired", "raw_unbound", "action_only", "result_only", "context_only")
MAX_HISTORY = 28
CANDIDATES = 16
TRAIN_TWINS = 640
META_TWINS = 480
CALIBRATION_TWINS = 160
EVAL_TWINS = 256
LOCAL_BANDWIDTH = 0.45
LINEAR_REGULARIZATION = 0.01
POLY_REGULARIZATION = 0.01
RBF_LENGTH_SCALE = 0.45
RBF_REGULARIZATION = 0.01
ERROR_TREES = 64
ERROR_DEPTH = 12
ERROR_MIN_LEAF = 8
WEIGHT_TEMPERATURE = 0.05
RESIDUAL_TREES = 96
RESIDUAL_DEPTH = 14
RESIDUAL_MIN_LEAF = 5
RAW_TREES = 64
RAW_DEPTH = 18
RAW_MIN_LEAF = 2
BOOTSTRAP_REPLICATES = 1000
PREREGISTER = Path("docs/binding_cpu_v3_sysid_residual_preregister.md")
ADMISSION_BASELINES = COMPONENTS + ("uniform_ensemble",) + RAW_MODES + ("result_mean",)


def _family_codes(kind: str, twins: int, rng: np.random.Generator) -> np.ndarray:
    if kind == "mixture":
        codes = np.resize(np.arange(5, dtype=np.int8), twins)
        rng.shuffle(codes)
        return codes
    if kind == "held":
        return np.full(twins, 5, dtype=np.int8)
    raise ValueError(f"unknown family kind: {kind}")


def _parameters(
    twins: int, parameter_range: str, rng: np.random.Generator
) -> dict[str, np.ndarray]:
    if parameter_range == "train":
        gain = rng.uniform(0.65, 1.00, twins)
        bias = rng.uniform(-0.15, 0.15, twins)
        slope = rng.uniform(0.80, 1.50, twins)
        coupling = rng.uniform(-0.25, 0.25, twins)
        drag = rng.uniform(0.20, 0.60, twins)
        drift = rng.uniform(-0.20, 0.20, twins)
    elif parameter_range == "ood":
        gain = rng.uniform(1.05, 1.35, twins)
        bias = rng.uniform(-0.25, 0.25, twins)
        slope = rng.uniform(1.80, 2.50, twins)
        coupling = rng.choice((-1.0, 1.0), twins) * rng.uniform(0.35, 0.65, twins)
        drag = rng.uniform(0.80, 1.20, twins)
        drift = rng.choice((-1.0, 1.0), twins) * rng.uniform(0.30, 0.50, twins)
    elif parameter_range == "held":
        gain = rng.uniform(0.75, 1.15, twins)
        bias = rng.uniform(-0.20, 0.20, twins)
        slope = rng.uniform(1.00, 2.00, twins)
        coupling = rng.choice((-1.0, 1.0), twins) * rng.uniform(0.25, 0.55, twins)
        drag = rng.uniform(0.30, 0.70, twins)
        drift = rng.uniform(-0.15, 0.15, twins)
    else:
        raise ValueError(f"unknown parameter range: {parameter_range}")
    return {
        "gain": gain,
        "bias": bias,
        "slope": slope,
        "coupling": coupling,
        "drag": drag,
        "drift": drift,
        "deadzone": rng.uniform(0.12, 0.32, twins),
        "cap": rng.uniform(0.35, 0.65, twins),
        "bump_width": rng.uniform(0.18, 0.32, twins),
        "bump_sign": rng.choice((-1.0, 1.0), twins),
        "ring": rng.uniform(0.45, 0.75, twins),
    }


def _response(
    actions: np.ndarray,
    public_time: np.ndarray,
    orientation: np.ndarray,
    family: np.ndarray,
    parameters: dict[str, np.ndarray],
) -> np.ndarray:
    action = np.asarray(actions, dtype=np.float64)
    time_value = np.asarray(public_time, dtype=np.float64)
    orientation = np.asarray(orientation, dtype=np.float64)
    w = np.stack((np.cos(orientation), np.sin(orientation)), axis=-1)
    perpendicular_vector = np.stack((-np.sin(orientation), np.cos(orientation)), axis=-1)
    while w.ndim < action.ndim:
        w = np.expand_dims(w, axis=-2)
        perpendicular_vector = np.expand_dims(perpendicular_vector, axis=-2)
    z = np.sum(action * w, axis=-1)
    p = np.sum(action * perpendicular_vector, axis=-1)
    radius = np.linalg.norm(action, axis=-1)
    angle = np.arctan2(p, z)

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
    drift = expand(parameters["drift"])
    deadzone = expand(parameters["deadzone"])
    cap = expand(parameters["cap"])
    bump_width = expand(parameters["bump_width"])
    bump_sign = expand(parameters["bump_sign"])
    ring_center = expand(parameters["ring"])
    family_value = np.asarray(family)
    while family_value.ndim < z.ndim:
        family_value = np.expand_dims(family_value, axis=-1)

    smooth = bias + gain * np.tanh(slope * z + coupling * z * p)
    rational = bias + gain * (z + coupling * z * p) / (1.0 + drag * radius * radius)
    bump = np.exp(
        -((z - 0.55) ** 2 + (p - 0.25) ** 2) / (2.0 * bump_width * bump_width)
    )
    local = bias + gain * (0.55 * np.tanh(slope * z) + 0.45 * bump_sign * bump)
    raw = z + coupling * z * p + 0.15 * np.sign(p) * z * z
    activated = np.sign(raw) * np.maximum(np.abs(raw) - deadzone, 0.0)
    threshold = bias + gain * np.clip(activated, -cap, cap)
    time_offset = time_value - 0.5
    nonstationary = (
        bias
        + drift * time_offset
        + gain * (1.0 + drift * time_offset) * (z + coupling * z * p)
        / (1.0 + drag * radius * radius)
    )
    cusp = np.sign(z) * np.sqrt(np.maximum(np.abs(z) - deadzone, 0.0))
    ring = np.exp(-((radius - ring_center) ** 2) / (2.0 * bump_width * bump_width))
    held = bias + gain * (0.65 * cusp + 0.35 * ring * np.sin(3.0 * angle))

    output = np.where(family_value == 0, smooth, rational)
    output = np.where(family_value == 2, local, output)
    output = np.where(family_value == 3, threshold, output)
    output = np.where(family_value == 4, nonstationary, output)
    output = np.where(family_value == 5, held, output)
    return output


def _sample_histories(
    twins: int,
    history_range: tuple[int, int],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lengths = rng.integers(history_range[0], history_range[1] + 1, twins)
    actions = np.zeros((twins, MAX_HISTORY, 2), dtype=np.float64)
    times = np.zeros((twins, MAX_HISTORY), dtype=np.float64)
    mask = np.zeros((twins, MAX_HISTORY), dtype=bool)
    centers = rng.uniform(-np.pi, np.pi, twins)
    for twin, length_value in enumerate(lengths):
        length = int(length_value)
        concentrated = int(round(0.70 * length))
        angles = np.concatenate(
            (
                centers[twin] + rng.normal(0.0, 0.45, concentrated),
                rng.uniform(-np.pi, np.pi, length - concentrated),
            )
        )
        radii = 0.15 + 0.85 * rng.beta(2.0, 2.0, length)
        order = rng.permutation(length)
        actions[twin, :length] = (
            radii[:, None] * np.stack((np.cos(angles), np.sin(angles)), axis=-1)
        )[order]
        times[twin, :length] = np.linspace(0.0, 1.0 - 1.0 / (length + 1), length)
        mask[twin, :length] = True
    return actions, times, mask, lengths


def _sample_candidates(
    history_actions: np.ndarray,
    history_mask: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    output = np.empty((len(history_actions), CANDIDATES, 2), dtype=np.float64)
    for twin in range(len(history_actions)):
        history = history_actions[twin, history_mask[twin]]
        for candidate_index in range(CANDIDATES):
            for _ in range(1000):
                angle = rng.uniform(-np.pi, np.pi)
                radius = 0.15 + 0.85 * math.sqrt(float(rng.uniform()))
                candidate = radius * np.asarray((math.cos(angle), math.sin(angle)))
                if np.min(np.linalg.norm(history - candidate, axis=1)) > 0.03:
                    output[twin, candidate_index] = candidate
                    break
            else:
                raise RuntimeError("candidate sampler exhausted")
    return output


def make_dataset(
    twins: int,
    seed: int,
    *,
    family_kind: str,
    parameter_range: str,
    history_range: tuple[int, int],
    noise_scale: float,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    family = _family_codes(family_kind, twins, rng)
    parameters = _parameters(twins, parameter_range, rng)
    orientation0 = rng.uniform(-np.pi, np.pi, twins)
    separation = rng.uniform(np.deg2rad(50.0), np.deg2rad(130.0), twins)
    orientation = np.stack((orientation0, orientation0 + separation), axis=1)
    actions, times, mask, lengths = _sample_histories(twins, history_range, rng)
    history_actions = np.repeat(actions[:, None, :, :], 2, axis=1)
    history_times = np.repeat(times[:, None, :], 2, axis=1)
    history_results = _response(
        history_actions, history_times, orientation, family, parameters
    )
    valid = np.repeat(mask[:, None, :], 2, axis=1)
    if noise_scale > 0.0:
        radius = np.linalg.norm(history_actions, axis=-1)
        noise = rng.normal(size=history_results.shape) * noise_scale * (0.4 + 0.8 * radius)
        history_results = history_results + noise
    history_results = np.where(valid, history_results, 0.0)

    candidates = _sample_candidates(actions, mask, rng)
    candidate_by_branch = np.repeat(candidates[:, None, :, :], 2, axis=1)
    candidate_time = np.ones(candidate_by_branch.shape[:-1], dtype=np.float64)
    displacement = _response(
        candidate_by_branch, candidate_time, orientation, family, parameters
    )
    target = rng.uniform(-0.50, 0.50, twins)

    swaps = np.asarray(([False, True] * ((twins + 1) // 2))[:twins], dtype=bool)
    rng.shuffle(swaps)
    history_results[swaps] = history_results[swaps, ::-1]
    displacement[swaps] = displacement[swaps, ::-1]
    orientation[swaps] = orientation[swaps, ::-1]
    utility = -np.abs(target[:, None, None] - displacement)
    return {
        "family": family,
        "orientation": orientation,
        "branch_swapped": swaps,
        "history_actions": history_actions,
        "history_results": history_results,
        "history_times": history_times,
        "history_mask": mask,
        "history_length": lengths,
        "candidates": candidates,
        "target": target,
        "displacement": displacement,
        "utility": utility,
    }


def _poly_design(actions: np.ndarray) -> np.ndarray:
    x = actions[..., 0]
    y = actions[..., 1]
    return np.stack((np.ones_like(x), x, y, x * x, x * y, y * y), axis=-1)


def _distances(query: np.ndarray, history: np.ndarray) -> np.ndarray:
    return np.linalg.norm(query[:, None, :] - history[None, :, :], axis=-1)


def _local_linear(
    history: np.ndarray, results: np.ndarray, query: np.ndarray
) -> np.ndarray:
    delta = history[None, :, :] - query[:, None, :]
    squared = np.sum(delta * delta, axis=2)
    weight = np.exp(-squared / (2.0 * LOCAL_BANDWIDTH**2))
    design = np.concatenate(
        (np.ones(delta.shape[:2] + (1,), dtype=np.float64), delta), axis=2
    )
    matrix = np.einsum("qhi,qh,qhj->qij", design, weight, design)
    matrix = matrix + LINEAR_REGULARIZATION * np.eye(3)[None, :, :]
    vector = np.einsum("qhi,qh,h->qi", design, weight, results)
    coefficient = np.linalg.solve(matrix, vector)
    return coefficient[:, 0]


def _component_predictions(
    history: np.ndarray, results: np.ndarray, query: np.ndarray
) -> np.ndarray:
    distance = _distances(query, history)
    order = np.argsort(distance, axis=1, kind="mergesort")
    nearest = results[order[:, 0]]
    selected = order[:, : min(5, len(history))]
    selected_distance = np.take_along_axis(distance, selected, axis=1)
    weights = 1.0 / np.maximum(selected_distance, 1.0e-5)
    knn = np.sum(weights * results[selected], axis=1) / np.sum(weights, axis=1)
    local = _local_linear(history, results, query)

    design = _poly_design(history)
    coefficient = np.linalg.solve(
        design.T @ design + POLY_REGULARIZATION * np.eye(design.shape[1]),
        design.T @ results,
    )
    poly = _poly_design(query) @ coefficient

    history_squared = np.sum(
        (history[:, None, :] - history[None, :, :]) ** 2, axis=-1
    )
    kernel = np.exp(-history_squared / (2.0 * RBF_LENGTH_SCALE**2))
    alpha = np.linalg.solve(
        kernel + RBF_REGULARIZATION * np.eye(len(history)), results
    )
    query_squared = np.sum((query[:, None, :] - history[None, :, :]) ** 2, axis=-1)
    rbf = np.exp(-query_squared / (2.0 * RBF_LENGTH_SCALE**2)) @ alpha
    return np.stack((nearest, knn, local, poly, rbf), axis=1)


def _loo_errors(history: np.ndarray, results: np.ndarray) -> np.ndarray:
    count = len(history)
    pair_distance = _distances(history, history)
    np.fill_diagonal(pair_distance, np.inf)
    order = np.argsort(pair_distance, axis=1, kind="mergesort")
    nearest = results[order[:, 0]]
    selected = order[:, : min(5, count - 1)]
    selected_distance = np.take_along_axis(pair_distance, selected, axis=1)
    weights = 1.0 / np.maximum(selected_distance, 1.0e-5)
    knn = np.sum(weights * results[selected], axis=1) / np.sum(weights, axis=1)

    delta = history[None, :, :] - history[:, None, :]
    local_weight = np.exp(
        -np.sum(delta * delta, axis=2) / (2.0 * LOCAL_BANDWIDTH**2)
    )
    np.fill_diagonal(local_weight, 0.0)
    local_design = np.concatenate(
        (np.ones(delta.shape[:2] + (1,), dtype=np.float64), delta), axis=2
    )
    local_matrix = np.einsum(
        "qhi,qh,qhj->qij", local_design, local_weight, local_design
    )
    local_matrix = local_matrix + LINEAR_REGULARIZATION * np.eye(3)[None, :, :]
    local_vector = np.einsum(
        "qhi,qh,h->qi", local_design, local_weight, results
    )
    local = np.linalg.solve(local_matrix, local_vector)[:, 0]

    design = _poly_design(history)
    inverse = np.linalg.inv(
        design.T @ design + POLY_REGULARIZATION * np.eye(design.shape[1])
    )
    coefficient = inverse @ design.T @ results
    fitted = design @ coefficient
    leverage = np.sum((design @ inverse) * design, axis=1)
    poly = results - (results - fitted) / np.maximum(1.0 - leverage, 1.0e-6)

    squared = np.sum((history[:, None, :] - history[None, :, :]) ** 2, axis=-1)
    kernel = np.exp(-squared / (2.0 * RBF_LENGTH_SCALE**2))
    inverse_kernel = np.linalg.inv(
        kernel + RBF_REGULARIZATION * np.eye(count)
    )
    alpha = inverse_kernel @ results
    rbf = results - alpha / np.maximum(np.diag(inverse_kernel), 1.0e-8)
    predictions = np.stack((nearest, knn, local, poly, rbf), axis=1)
    return np.mean(np.abs(predictions - results[:, None]), axis=0)


def _noise_estimate(history: np.ndarray, results: np.ndarray) -> float:
    distance = _distances(history, history)
    np.fill_diagonal(distance, np.inf)
    nearest = np.argmin(distance, axis=1)
    return float(np.median(np.abs(results - results[nearest])) / math.sqrt(2.0))


def _history_features(
    history: np.ndarray,
    results: np.ndarray,
    times: np.ndarray,
    query: np.ndarray,
    target: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    components = _component_predictions(history, results, query)
    loo = _loo_errors(history, results)
    distance = _distances(query, history)
    order = np.argsort(distance, axis=1, kind="mergesort")
    distance_columns = np.stack(
        (
            distance[np.arange(len(query)), order[:, 0]],
            distance[np.arange(len(query)), order[:, min(2, len(history) - 1)]],
            distance[np.arange(len(query)), order[:, min(4, len(history) - 1)]],
        ),
        axis=1,
    )
    local_indices = order[:, : min(5, len(history))]
    local_values = results[local_indices]
    local_summary = np.stack(
        (np.mean(local_values, axis=1), np.var(local_values, axis=1)), axis=1
    )
    disagreement = np.stack(
        (np.std(components, axis=1), np.ptp(components, axis=1)), axis=1
    )
    max_radius = max(float(np.max(np.linalg.norm(history, axis=1))), 1.0e-8)
    extrapolation = np.linalg.norm(query, axis=1, keepdims=True) / max_radius
    history_fraction = np.full((len(query), 1), len(history) / MAX_HISTORY)
    noise = np.full((len(query), 1), _noise_estimate(history, results))
    time_order = np.argsort(times, kind="mergesort")
    block = max(2, len(history) // 3)
    drift_value = float(
        np.mean(results[time_order[-block:]]) - np.mean(results[time_order[:block]])
    )
    drift = np.full((len(query), 1), drift_value)
    query_context = np.column_stack(
        (
            query,
            np.linalg.norm(query, axis=1),
            np.full(len(query), target),
            np.ones(len(query)),
        )
    )
    public = np.concatenate(
        (
            components,
            np.broadcast_to(loo[None, :], components.shape),
            distance_columns,
            local_summary,
            disagreement,
            extrapolation,
            history_fraction,
            noise,
            drift,
            query_context,
        ),
        axis=1,
    )
    return components, loo, public


def predictor_bundle(dataset: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    twins = len(dataset["history_actions"])
    components = np.empty((twins, 2, CANDIDATES, len(COMPONENTS)), dtype=np.float64)
    loo = np.empty((twins, 2, len(COMPONENTS)), dtype=np.float64)
    public = np.empty((twins, 2, CANDIDATES, 26), dtype=np.float64)
    for twin in range(twins):
        valid = dataset["history_mask"][twin]
        for branch in range(2):
            component_value, loo_value, public_value = _history_features(
                dataset["history_actions"][twin, branch, valid],
                dataset["history_results"][twin, branch, valid],
                dataset["history_times"][twin, branch, valid],
                dataset["candidates"][twin],
                float(dataset["target"][twin]),
            )
            components[twin, branch] = component_value
            loo[twin, branch] = loo_value
            public[twin, branch] = public_value
    return {"components": components, "loo": loo, "public": public}


def _softmax_negative_error(error: np.ndarray) -> np.ndarray:
    logits = -np.asarray(error) / WEIGHT_TEMPERATURE
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    value = np.exp(logits)
    return value / np.sum(value, axis=-1, keepdims=True)


def _error_model(seed: int) -> ExtraTreesRegressor:
    return ExtraTreesRegressor(
        n_estimators=ERROR_TREES,
        max_depth=ERROR_DEPTH,
        min_samples_leaf=ERROR_MIN_LEAF,
        max_features=1.0,
        bootstrap=False,
        random_state=seed,
        n_jobs=-1,
    )


def _residual_model(seed: int) -> ExtraTreesRegressor:
    return ExtraTreesRegressor(
        n_estimators=RESIDUAL_TREES,
        max_depth=RESIDUAL_DEPTH,
        min_samples_leaf=RESIDUAL_MIN_LEAF,
        max_features=0.8,
        bootstrap=False,
        random_state=seed,
        n_jobs=-1,
    )


def _residual_features(
    public: np.ndarray,
    weights: np.ndarray,
    blend: np.ndarray,
    weighted_error: np.ndarray,
    disagreement: np.ndarray,
) -> np.ndarray:
    return np.concatenate(
        (
            public,
            weights,
            blend[..., None],
            weighted_error[..., None],
            disagreement[..., None],
        ),
        axis=-1,
    )


def _predicted_component_errors(
    error_models: list[ExtraTreesRegressor], public: np.ndarray
) -> np.ndarray:
    flat = public.reshape(-1, public.shape[-1])
    values = np.stack(
        [np.maximum(model.predict(flat), 0.0) for model in error_models], axis=1
    )
    return values.reshape(public.shape[:-1] + (len(COMPONENTS),))


def hybrid_predict(
    state: dict[str, Any], bundle: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    public = bundle["public"]
    components = bundle["components"]
    predicted_error = _predicted_component_errors(state["error_models"], public)
    weights = _softmax_negative_error(predicted_error)
    blend = np.sum(weights * components, axis=-1)
    weighted_error = np.sum(weights * predicted_error, axis=-1)
    disagreement = np.std(components, axis=-1)
    residual_features = _residual_features(
        public, weights, blend, weighted_error, disagreement
    )
    flat = residual_features.reshape(-1, residual_features.shape[-1])
    residual = state["residual_model"].predict(flat).reshape(blend.shape)
    tree_values = np.stack(
        [tree.predict(flat) for tree in state["residual_model"].estimators_], axis=1
    )
    tree_std = np.std(tree_values, axis=1).reshape(blend.shape)
    raw_uncertainty = np.maximum(weighted_error + disagreement + tree_std, 1.0e-6)
    return {
        "response": blend + residual,
        "blend": blend,
        "residual": residual,
        "weights": weights,
        "predicted_component_error": predicted_error,
        "raw_uncertainty": raw_uncertainty,
        "uncertainty": raw_uncertainty * float(state["calibration_factor"]),
    }


def fit_hybrid(
    dataset: dict[str, np.ndarray], bundle: dict[str, np.ndarray], seed: int
) -> dict[str, Any]:
    fit_slice = slice(0, META_TWINS)
    calibration_slice = slice(META_TWINS, TRAIN_TWINS)
    public_fit = bundle["public"][fit_slice]
    components_fit = bundle["components"][fit_slice]
    target_fit = dataset["displacement"][fit_slice]
    flat_public = public_fit.reshape(-1, public_fit.shape[-1])
    flat_target = target_fit.reshape(-1)
    flat_components = components_fit.reshape(-1, len(COMPONENTS))
    error_models = []
    for index in range(len(COMPONENTS)):
        model = _error_model(seed + 101 * (index + 1))
        model.fit(flat_public, np.abs(flat_target - flat_components[:, index]))
        error_models.append(model)

    predicted_error = _predicted_component_errors(error_models, public_fit)
    weights = _softmax_negative_error(predicted_error)
    blend = np.sum(weights * components_fit, axis=-1)
    weighted_error = np.sum(weights * predicted_error, axis=-1)
    disagreement = np.std(components_fit, axis=-1)
    features = _residual_features(
        public_fit, weights, blend, weighted_error, disagreement
    )
    residual_model = _residual_model(seed + 991)
    residual_model.fit(
        features.reshape(-1, features.shape[-1]), (target_fit - blend).reshape(-1)
    )
    state: dict[str, Any] = {
        "error_models": error_models,
        "residual_model": residual_model,
        "calibration_factor": 1.0,
    }

    calibration_bundle = {
        key: value[calibration_slice] for key, value in bundle.items()
    }
    calibration_prediction = hybrid_predict(state, calibration_bundle)
    calibration_truth = dataset["displacement"][calibration_slice]
    ratio = np.abs(calibration_truth - calibration_prediction["response"]) / np.maximum(
        calibration_prediction["raw_uncertainty"], 1.0e-6
    )
    state["calibration_factor"] = float(np.quantile(ratio, 0.80))
    state["calibration_examples"] = int(ratio.size)
    return state


def _raw_features(dataset: dict[str, np.ndarray], mode: str) -> np.ndarray:
    if mode not in RAW_MODES:
        raise ValueError(f"unknown raw mode: {mode}")
    actions = dataset["history_actions"]
    results = dataset["history_results"]
    times = dataset["history_times"]
    mask = dataset["history_mask"]
    candidates = dataset["candidates"]
    candidate_radius = np.maximum(np.linalg.norm(candidates, axis=-1), 1.0e-12)
    unit = candidates / candidate_radius[..., None]
    perpendicular_unit = np.stack((-unit[..., 1], unit[..., 0]), axis=-1)
    parallel = np.einsum("nbhd,nkd->nbkh", actions, unit)
    perpendicular = np.einsum("nbhd,nkd->nbkh", actions, perpendicular_unit)
    radius = np.broadcast_to(
        np.linalg.norm(actions, axis=-1)[:, :, None, :], parallel.shape
    )
    valid = np.broadcast_to(mask[:, None, None, :], parallel.shape)
    angle = np.mod(np.arctan2(perpendicular, parallel) + np.pi, 2.0 * np.pi)
    key = angle + 1.0e-4 * radius
    order = np.argsort(np.where(valid, key, 100.0), axis=3, kind="mergesort")

    def gather(value: np.ndarray) -> np.ndarray:
        return np.take_along_axis(value, order, axis=3)

    parallel = gather(parallel)
    perpendicular = gather(perpendicular)
    radius = gather(radius)
    gathered_mask = gather(valid).astype(np.float64)
    gathered_result = gather(np.broadcast_to(results[:, :, None, :], parallel.shape))
    gathered_time = gather(np.broadcast_to(times[:, :, None, :], parallel.shape))
    independently_sorted = np.sort(
        np.where(mask[:, None, :], results, np.inf), axis=2
    )
    independently_sorted = np.where(np.isfinite(independently_sorted), independently_sorted, 0.0)
    independently_sorted = np.broadcast_to(
        independently_sorted[:, :, None, :], parallel.shape
    )

    if mode == "raw_paired":
        result_channel = gathered_result
    elif mode in ("raw_unbound", "result_only"):
        result_channel = independently_sorted
    else:
        result_channel = np.zeros_like(gathered_result)
    if mode in ("result_only", "context_only"):
        parallel = np.zeros_like(parallel)
        perpendicular = np.zeros_like(perpendicular)
        radius = np.zeros_like(radius)
    if mode == "context_only":
        result_channel = np.zeros_like(result_channel)
        gathered_time = np.zeros_like(gathered_time)
        gathered_mask = np.zeros_like(gathered_mask)

    tokens = np.stack(
        (
            parallel, perpendicular, radius, result_channel, gathered_time,
            gathered_mask,
        ),
        axis=-1,
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
    return np.concatenate(
        (tokens.reshape(*tokens.shape[:3], -1), context), axis=-1
    ).astype(np.float32)


def fit_raw_controls(
    dataset: dict[str, np.ndarray], seed: int
) -> dict[str, ExtraTreesRegressor]:
    fit_dataset = {key: value[:META_TWINS] for key, value in dataset.items()}
    target = fit_dataset["displacement"].reshape(-1)
    models = {}
    for index, mode in enumerate(RAW_MODES):
        features = _raw_features(fit_dataset, mode)
        model = ExtraTreesRegressor(
            n_estimators=RAW_TREES,
            max_depth=RAW_DEPTH,
            min_samples_leaf=RAW_MIN_LEAF,
            max_features=0.7,
            bootstrap=False,
            random_state=seed + 701 * index,
            n_jobs=-1,
        )
        model.fit(features.reshape(-1, features.shape[-1]), target)
        models[mode] = model
    return models


def predict_raw(
    model: ExtraTreesRegressor, dataset: dict[str, np.ndarray], mode: str
) -> np.ndarray:
    features = _raw_features(dataset, mode)
    return model.predict(features.reshape(-1, features.shape[-1])).reshape(
        features.shape[:3]
    )


def _utility(dataset: dict[str, np.ndarray], response: np.ndarray) -> np.ndarray:
    return -np.abs(dataset["target"][:, None, None] - response)


def _twin_preference(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
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


def score_prediction(
    truth_utility: np.ndarray,
    predicted_utility: np.ndarray,
    *,
    truth_response: np.ndarray | None = None,
    predicted_response: np.ndarray | None = None,
    uncertainty: np.ndarray | None = None,
) -> dict[str, Any]:
    true_best = np.argmax(truth_utility, axis=2)
    predicted_best = np.argmax(predicted_utility, axis=2)
    selected = np.take_along_axis(
        truth_utility, predicted_best[..., None], axis=2
    )[..., 0]
    optimal = np.max(truth_utility, axis=2)
    regret = optimal - selected
    span = np.max(truth_utility, axis=2) - np.min(truth_utility, axis=2)
    normalized = regret / np.maximum(span, 1.0e-12)
    top1 = true_best == predicted_best
    by_twin = np.mean(normalized, axis=1)
    ordered = np.sort(truth_utility, axis=2)
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
    preference = _twin_preference(truth_utility, predicted_utility)
    output: dict[str, Any] = {
        "normalized_regret": float(np.mean(normalized)),
        "mean_regret": float(np.mean(regret)),
        "top1_accuracy": float(np.mean(top1)),
        "tpa": preference["tpa"],
        "informative_fraction": preference["informative_fraction"],
        "normalized_regret_by_twin": by_twin.tolist(),
        "difficulty": difficulty,
    }
    if uncertainty is not None:
        if truth_response is None or predicted_response is None:
            raise ValueError("calibration metrics require response truth and prediction")
        absolute_error = np.abs(truth_response - predicted_response)
        coverage = absolute_error <= uncertainty
        correlation = float(
            spearmanr(uncertainty.reshape(-1), absolute_error.reshape(-1)).statistic
        )
        if not np.isfinite(correlation):
            correlation = 0.0
        output["calibration"] = {
            "coverage_80": float(np.mean(coverage)),
            "mean_interval_width": float(np.mean(2.0 * uncertainty)),
            "uncertainty_error_spearman": correlation,
            "mean_absolute_response_error": float(np.mean(absolute_error)),
        }
    return output


def integrity(dataset: dict[str, np.ndarray]) -> dict[str, Any]:
    actions = dataset["history_actions"]
    times = dataset["history_times"]
    mask = dataset["history_mask"]
    lengths = dataset["history_length"]
    candidates = dataset["candidates"]
    minimum_distances = []
    for twin in range(len(actions)):
        history = actions[twin, 0, mask[twin]]
        minimum_distances.append(
            float(
                np.min(
                    np.linalg.norm(
                        candidates[twin, :, None, :] - history[None, :, :], axis=-1
                    )
                )
            )
        )
    arrays = (
        actions, times, dataset["history_results"], candidates, dataset["target"],
        dataset["displacement"], dataset["utility"],
    )
    checks = {
        "shapes": bool(
            actions.shape == (len(actions), 2, MAX_HISTORY, 2)
            and times.shape == (len(actions), 2, MAX_HISTORY)
            and candidates.shape == (len(actions), CANDIDATES, 2)
            and dataset["utility"].shape == (len(actions), 2, CANDIDATES)
        ),
        "finite": bool(all(np.all(np.isfinite(value)) for value in arrays)),
        "shared_history_actions": bool(np.array_equal(actions[:, 0], actions[:, 1])),
        "shared_history_times": bool(np.array_equal(times[:, 0], times[:, 1])),
        "mask_matches_length": bool(np.array_equal(mask.sum(axis=1), lengths)),
        "length_range": bool(np.min(lengths) >= 6 and np.max(lengths) <= MAX_HISTORY),
        "branch_balance": bool(int(np.sum(dataset["branch_swapped"])) * 2 == len(actions)),
        "candidate_noncollision": bool(min(minimum_distances) > 0.03),
        "nonzero_utility_span": bool(
            np.all(np.ptp(dataset["utility"], axis=2) > 1.0e-10)
        ),
    }
    family_values, family_counts = np.unique(dataset["family"], return_counts=True)
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "history_min": int(np.min(lengths)),
        "history_max": int(np.max(lengths)),
        "candidate_history_min_distance": float(min(minimum_distances)),
        "family_counts": {
            str(int(key)): int(value) for key, value in zip(family_values, family_counts)
        },
    }


def _copy_dataset(dataset: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {key: value.copy() for key, value in dataset.items()}


def _pair_permutation(dataset: dict[str, np.ndarray], seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    output = _copy_dataset(dataset)
    for twin, length_value in enumerate(output["history_length"]):
        length = int(length_value)
        permutation = rng.permutation(length)
        output["history_actions"][twin, :, :length] = output["history_actions"][
            twin, :, :length
        ][:, permutation, :]
        output["history_results"][twin, :, :length] = output["history_results"][
            twin, :, :length
        ][:, permutation]
        output["history_times"][twin, :, :length] = output["history_times"][
            twin, :, :length
        ][:, permutation]
    return output


def _binding_break(dataset: dict[str, np.ndarray], seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    output = _copy_dataset(dataset)
    for twin, length_value in enumerate(output["history_length"]):
        length = int(length_value)
        for branch in range(2):
            permutation = rng.permutation(length)
            output["history_results"][twin, branch, :length] = output[
                "history_results"
            ][twin, branch, :length][permutation]
    return output


def _twin_history_swap(dataset: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    output = _copy_dataset(dataset)
    for key in ("history_actions", "history_results", "history_times"):
        output[key] = output[key][:, ::-1].copy()
    return output


def _slot_reversal(dataset: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    output = _copy_dataset(dataset)
    output["candidates"] = output["candidates"][:, ::-1].copy()
    output["displacement"] = output["displacement"][:, :, ::-1].copy()
    output["utility"] = output["utility"][:, :, ::-1].copy()
    return output


def evaluate_split(
    dataset: dict[str, np.ndarray],
    hybrid_state: dict[str, Any],
    raw_models: dict[str, ExtraTreesRegressor],
    intervention_seed: int,
) -> dict[str, Any]:
    truth_utility = dataset["utility"]
    bundle = predictor_bundle(dataset)
    hybrid = hybrid_predict(hybrid_state, bundle)
    scores: dict[str, Any] = {
        "hybrid": score_prediction(
            truth_utility,
            _utility(dataset, hybrid["response"]),
            truth_response=dataset["displacement"],
            predicted_response=hybrid["response"],
            uncertainty=hybrid["uncertainty"],
        )
    }
    scores["hybrid"]["mean_component_weights"] = np.mean(
        hybrid["weights"], axis=(0, 1, 2)
    ).tolist()
    for index, name in enumerate(COMPONENTS):
        response = bundle["components"][..., index]
        scores[name] = score_prediction(truth_utility, _utility(dataset, response))
    uniform = np.mean(bundle["components"], axis=-1)
    scores["uniform_ensemble"] = score_prediction(
        truth_utility, _utility(dataset, uniform)
    )
    result_mean = np.sum(
        dataset["history_results"] * dataset["history_mask"][:, None, :], axis=2
    ) / dataset["history_length"][:, None]
    result_mean = np.broadcast_to(result_mean[:, :, None], dataset["displacement"].shape)
    scores["result_mean"] = score_prediction(
        truth_utility, _utility(dataset, result_mean)
    )
    for mode in RAW_MODES:
        response = predict_raw(raw_models[mode], dataset, mode)
        scores[mode] = score_prediction(truth_utility, _utility(dataset, response))

    component_error = np.abs(
        bundle["components"] - dataset["displacement"][..., None]
    )
    oracle_index = np.argmin(component_error, axis=-1)
    oracle = np.take_along_axis(
        bundle["components"], oracle_index[..., None], axis=-1
    )[..., 0]
    scores["oracle_selector_ceiling"] = score_prediction(
        truth_utility, _utility(dataset, oracle)
    )

    # These three transforms have exact bundle-level equivalents. Reusing the
    # original fit is mathematically identical to refitting order-invariant
    # per-history predictors and avoids redundant CPU work.
    interventions = {
        "pair_preserving": score_prediction(
            truth_utility,
            _utility(dataset, hybrid["response"]),
            truth_response=dataset["displacement"],
            predicted_response=hybrid["response"],
            uncertainty=hybrid["uncertainty"],
        )
    }
    binding_dataset = _binding_break(dataset, intervention_seed + 2)
    binding_prediction = hybrid_predict(
        hybrid_state, predictor_bundle(binding_dataset)
    )
    interventions["binding_breaking"] = score_prediction(
        binding_dataset["utility"],
        _utility(binding_dataset, binding_prediction["response"]),
        truth_response=binding_dataset["displacement"],
        predicted_response=binding_prediction["response"],
        uncertainty=binding_prediction["uncertainty"],
    )

    twin_dataset = _twin_history_swap(dataset)
    twin_bundle = {key: value[:, ::-1].copy() for key, value in bundle.items()}
    twin_prediction = hybrid_predict(hybrid_state, twin_bundle)
    interventions["twin_history_swap"] = score_prediction(
        twin_dataset["utility"],
        _utility(twin_dataset, twin_prediction["response"]),
        truth_response=twin_dataset["displacement"],
        predicted_response=twin_prediction["response"],
        uncertainty=twin_prediction["uncertainty"],
    )

    slot_dataset = _slot_reversal(dataset)
    slot_bundle = {
        "components": bundle["components"][:, :, ::-1].copy(),
        "loo": bundle["loo"].copy(),
        "public": bundle["public"][:, :, ::-1].copy(),
    }
    slot_prediction = hybrid_predict(hybrid_state, slot_bundle)
    interventions["slot_reversal"] = score_prediction(
        slot_dataset["utility"],
        _utility(slot_dataset, slot_prediction["response"]),
        truth_response=slot_dataset["displacement"],
        predicted_response=slot_prediction["response"],
        uncertainty=slot_prediction["uncertainty"],
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
        family_kind="mixture",
        parameter_range="train",
        history_range=(16, 28),
        noise_scale=0.01,
    )
    evaluation = {
        "id": make_dataset(
            EVAL_TWINS,
            seed + 20_000,
            family_kind="mixture",
            parameter_range="train",
            history_range=(16, 28),
            noise_scale=0.01,
        ),
        "parameter_ood": make_dataset(
            EVAL_TWINS,
            seed + 30_000,
            family_kind="mixture",
            parameter_range="ood",
            history_range=(16, 28),
            noise_scale=0.015,
        ),
        "sparse_history": make_dataset(
            EVAL_TWINS,
            seed + 40_000,
            family_kind="mixture",
            parameter_range="train",
            history_range=(6, 12),
            noise_scale=0.01,
        ),
        "noise_ood": make_dataset(
            EVAL_TWINS,
            seed + 50_000,
            family_kind="mixture",
            parameter_range="train",
            history_range=(10, 20),
            noise_scale=0.05,
        ),
        "held_structure": make_dataset(
            EVAL_TWINS,
            seed + 60_000,
            family_kind="held",
            parameter_range="held",
            history_range=(10, 22),
            noise_scale=0.015,
        ),
    }
    return train, evaluation


def run_seed(seed: int) -> dict[str, Any]:
    started = time.perf_counter()
    train, evaluation = _datasets_for_seed(seed)
    train_bundle = predictor_bundle(train)
    hybrid_state = fit_hybrid(train, train_bundle, seed)
    raw_models = fit_raw_controls(train, seed + 5000)
    splits = {
        split: evaluate_split(
            evaluation[split], hybrid_state, raw_models, seed + 100_000 + index * 1000
        )
        for index, split in enumerate(SPLITS)
    }
    return {
        "seed": seed,
        "train_integrity": integrity(train),
        "calibration_factor": float(hybrid_state["calibration_factor"]),
        "calibration_examples": int(hybrid_state["calibration_examples"]),
        "splits": splits,
        "runtime_seconds": time.perf_counter() - started,
    }


def _mean(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _metric_summary(score_rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    scalar_metrics = (
        "normalized_regret", "mean_regret", "top1_accuracy", "tpa",
        "informative_fraction",
    )
    output: dict[str, Any] = {}
    for metric in scalar_metrics:
        values = np.asarray([row[name][metric] for row in score_rows], dtype=np.float64)
        output[metric] = float(np.mean(values))
        output[f"{metric}_seed_std"] = float(np.std(values, ddof=1))
    output["difficulty"] = {
        level: {
            metric: _mean([row[name]["difficulty"][level][metric] for row in score_rows])
            for metric in ("normalized_regret", "top1_accuracy")
        }
        for level in ("hard", "medium", "easy")
    }
    if all("calibration" in row[name] for row in score_rows):
        output["calibration"] = {
            metric: _mean([row[name]["calibration"][metric] for row in score_rows])
            for metric in (
                "coverage_80", "mean_interval_width", "uncertainty_error_spearman",
                "mean_absolute_response_error",
            )
        }
    if all("mean_component_weights" in row[name] for row in score_rows):
        output["mean_component_weights"] = np.mean(
            np.asarray([row[name]["mean_component_weights"] for row in score_rows]), axis=0
        ).tolist()
    return output


def _hierarchical_bootstrap(
    rows: list[dict[str, Any]], baseline: str, rng_seed: int
) -> dict[str, float]:
    rng = np.random.default_rng(rng_seed)
    values = []
    for _ in range(BOOTSTRAP_REPLICATES):
        seed_indices = rng.integers(0, len(rows), size=len(rows))
        seed_means = []
        for seed_index in seed_indices:
            hybrid = np.asarray(
                rows[int(seed_index)]["scores"]["hybrid"]["normalized_regret_by_twin"],
                dtype=np.float64,
            )
            comparison = np.asarray(
                rows[int(seed_index)]["scores"][baseline]["normalized_regret_by_twin"],
                dtype=np.float64,
            )
            twin_indices = rng.integers(0, len(hybrid), size=len(hybrid))
            seed_means.append(
                float(np.mean(comparison[twin_indices] - hybrid[twin_indices]))
            )
        values.append(float(np.mean(seed_means)))
    lower, upper = np.quantile(values, (0.025, 0.975))
    return {
        "replicates": BOOTSTRAP_REPLICATES,
        "lower_95": float(lower),
        "median": float(np.median(values)),
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
            ADMISSION_BASELINES, key=lambda name: scores[name]["normalized_regret"]
        )
        hybrid_regret = scores["hybrid"]["normalized_regret"]
        baseline_regret = scores[strongest_baseline]["normalized_regret"]
        relative_improvement = (
            baseline_regret - hybrid_regret
        ) / max(baseline_regret, 1.0e-12)
        seed_improvements = [
            row["scores"][strongest_baseline]["normalized_regret"]
            - row["scores"]["hybrid"]["normalized_regret"]
            for row in rows
        ]
        seed_wins = int(np.sum(np.asarray(seed_improvements) > 0.0))
        bootstrap = _hierarchical_bootstrap(
            rows, strongest_baseline, 990_000 + split_index
        )
        hybrid_tpa = scores["hybrid"]["tpa"]
        binding_regret_increase = (
            interventions["binding_breaking"]["normalized_regret"] - hybrid_regret
        )
        binding_tpa_drop = hybrid_tpa - interventions["binding_breaking"]["tpa"]
        calibration = scores["hybrid"]["calibration"]
        checks = {
            "structural_integrity": all(row["integrity"]["passed"] for row in rows),
            "regret_relative_improvement_at_least_0_10": relative_improvement >= 0.10,
            "seed_wins_at_least_4": seed_wins >= 4,
            "bootstrap_lower_positive": bootstrap["lower_95"] > 0.0,
            "top1_gain_at_least_0_03": (
                scores["hybrid"]["top1_accuracy"]
                - scores[strongest_baseline]["top1_accuracy"] >= 0.03
            ),
            "binding_regret_increase_at_least_0_05": binding_regret_increase >= 0.05,
            "binding_tpa_drop_at_least_0_05": binding_tpa_drop >= 0.05,
            "pair_regret_change_at_most_0_005": (
                abs(interventions["pair_preserving"]["normalized_regret"] - hybrid_regret)
                <= 0.005
            ),
            "pair_tpa_change_at_most_0_005": (
                abs(interventions["pair_preserving"]["tpa"] - hybrid_tpa) <= 0.005
            ),
            "twin_swap_regret_increase_at_least_0_05": (
                interventions["twin_history_swap"]["normalized_regret"] - hybrid_regret
                >= 0.05
            ),
            "slot_regret_change_at_most_0_005": (
                abs(interventions["slot_reversal"]["normalized_regret"] - hybrid_regret)
                <= 0.005
            ),
            "anti_saturation_tpa_below_0_95": hybrid_tpa < 0.95,
            "anti_saturation_regret_at_least_0_01": hybrid_regret >= 0.01,
            "coverage_between_0_70_and_0_90": (
                0.70 <= calibration["coverage_80"] <= 0.90
            ),
            "uncertainty_spearman_at_least_0_20": (
                calibration["uncertainty_error_spearman"] >= 0.20
            ),
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
        "decision": "CPU_V3_GO" if passed else "CPU_V3_NO_GO",
        "passed": passed,
        "training_integrity": training_integrity,
        "splits": split_summaries,
    }


def run() -> dict[str, Any]:
    started = time.perf_counter()
    preregister_bytes = PREREGISTER.read_bytes()
    seed_results = [run_seed(seed) for seed in SEEDS]
    return {
        "schema": "actmask-binding-cpu-v3-sysid-residual",
        "preregister": str(PREREGISTER).replace("\\", "/"),
        "preregister_sha256": hashlib.sha256(preregister_bytes).hexdigest(),
        "cpu_only": True,
        "seeds": list(SEEDS),
        "configuration": {
            "train_twins": TRAIN_TWINS,
            "meta_twins": META_TWINS,
            "calibration_twins": CALIBRATION_TWINS,
            "eval_twins": EVAL_TWINS,
            "candidates": CANDIDATES,
            "max_history": MAX_HISTORY,
            "components": list(COMPONENTS),
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
            "hybrid_regret": row["scores"]["hybrid"]["normalized_regret"],
            "hybrid_top1": row["scores"]["hybrid"]["top1_accuracy"],
            "hybrid_tpa": row["scores"]["hybrid"]["tpa"],
            "coverage_80": row["scores"]["hybrid"]["calibration"]["coverage_80"],
            "uncertainty_spearman": row["scores"]["hybrid"]["calibration"][
                "uncertainty_error_spearman"
            ],
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
        default=Path("audit/results/binding_cpu_v3_sysid_residual.json"),
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
