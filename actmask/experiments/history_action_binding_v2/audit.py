"""Shared no-training baselines, interventions, metrics, and v2 gates."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from actmask.experiments.history_action_binding_v2.common import PROTOCOL, write_json


BASELINES = (
    "class_prior", "current_only", "candidate_index", "action_only", "result_only",
    "arrow_action_compatibility", "repeat_last_failed_action", "nearest_historical_action",
    "knn_action_result_lookup", "linear_sysid", "quadratic_fourier_sysid",
    "oracle_latent_rollout",
)
MARGINAL_BASELINES = ("current_only", "action_only", "result_only", "candidate_index")
INTERACTION_BASELINES = ("arrow_action_compatibility", "nearest_historical_action", "knn_action_result_lookup")


def _scalar(value) -> str:
    return str(np.asarray(value).item())


def peak_candidates(dataset: dict[str, np.ndarray]) -> np.ndarray:
    profile = np.asarray(PROTOCOL["pulse_profile"], dtype=np.float32)
    index = int(np.argmax(profile))
    return dataset["candidate_actions"][:, :, index].astype(np.float64)


def pair_preserving_shuffle(actions: np.ndarray, results: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    directions = len(PROTOCOL["probe_directions_degrees"])
    shape_a = actions.shape
    shape_r = results.shape
    grouped_a = actions.reshape(*actions.shape[:2], -1, directions, actions.shape[-1])
    grouped_r = results.reshape(*results.shape[:2], -1, directions)
    return (
        np.roll(grouped_a, 1, axis=3).reshape(shape_a).copy(),
        np.roll(grouped_r, 1, axis=3).reshape(shape_r).copy(),
    )


def binding_breaking_shuffle(actions: np.ndarray, results: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    directions = len(PROTOCOL["probe_directions_degrees"])
    grouped_r = results.reshape(*results.shape[:2], -1, directions)
    return actions.copy(), np.roll(grouped_r, 1, axis=3).reshape(results.shape).copy()


def twin_history_swap(actions: np.ndarray, results: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return actions[:, ::-1].copy(), results[:, ::-1].copy()


def _history(dataset: dict[str, np.ndarray], variant: str) -> tuple[np.ndarray, np.ndarray]:
    actions = dataset["probe_actions"].astype(np.float64)
    results = dataset["probe_delta"].astype(np.float64)
    if variant == "original":
        return actions, results
    if variant == "pair_preserving_shuffle":
        return pair_preserving_shuffle(actions, results)
    if variant == "binding_breaking_shuffle":
        return binding_breaking_shuffle(actions, results)
    if variant == "twin_history_swap":
        return twin_history_swap(actions, results)
    raise ValueError(variant)


def candidate_slot_permutation(dataset: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    output = {key: value.copy() for key, value in dataset.items()}
    count = dataset["candidate_actions"].shape[1]
    permutation = np.arange(count - 1, -1, -1)
    axes = {
        "candidate_actions": 1, "candidate_actions_by_branch": 2, "candidate_template_id": 1,
        "candidate_trace_public": 2, "candidate_trace_response_pos": 2,
        "candidate_trace_response_vel": 2, "candidate_trace_actuator_pos": 2,
        "candidate_trace_target_pos": 2, "candidate_trace_distance": 2,
        "candidate_trace_contact_magnitude": 2, "candidate_trace_contact": 2,
        "first_success_step": 2, "min_distance": 2, "final_error": 2,
        "utility": 2, "success": 2, "final_state_hash": 2, "final_state_vector": 2,
        "symbolic_candidate_displacement": 2,
    }
    for key, axis in axes.items():
        if key in output:
            output[key] = np.take(output[key], permutation, axis=axis)
    return output


def coordinate_rotation(dataset: dict[str, np.ndarray], degrees: float = 37.0) -> dict[str, np.ndarray]:
    output = {key: value.copy() for key, value in dataset.items()}
    angle = np.deg2rad(degrees)
    matrix = np.asarray([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]], dtype=np.float64)
    for key in ("probe_actions", "candidate_actions", "candidate_actions_by_branch"):
        if key in output:
            output[key] = (np.asarray(output[key]) @ matrix.T).astype(output[key].dtype)
    if "public_rotation" in output:
        output["public_rotation"] = (output["public_rotation"] + angle).astype(output["public_rotation"].dtype)
    for key in ("latent_phi",):
        if key in output:
            output[key] = (output[key] + angle).astype(output[key].dtype)
    if "decision_public_state" in output:
        rho = np.asarray(output["public_rotation"])[:, None]
        output["decision_public_state"][..., 3] = np.cos(rho)
        output["decision_public_state"][..., 4] = np.sin(rho)
    return output


def _features(action: np.ndarray, kind: str) -> np.ndarray:
    x = action[..., 0]
    y = action[..., 1]
    radius = np.sqrt(x * x + y * y)
    angle = np.arctan2(y, x)
    if kind == "linear":
        return np.stack((np.ones_like(x), x, y), axis=-1)
    return np.stack(
        (
            np.ones_like(x),
            radius * np.cos(2.0 * angle), radius * np.sin(2.0 * angle),
            np.square(radius) * np.cos(2.0 * angle), np.square(radius) * np.sin(2.0 * angle),
        ), axis=-1,
    )


def _least_squares(actions: np.ndarray, results: np.ndarray, candidates: np.ndarray, kind: str) -> np.ndarray:
    twins, branches = actions.shape[:2]
    prediction = np.zeros((twins, branches, candidates.shape[1]), dtype=np.float64)
    for twin in range(twins):
        for branch in range(branches):
            design = _features(actions[twin, branch], kind)
            coefficient = np.linalg.lstsq(design, results[twin, branch], rcond=None)[0]
            prediction[twin, branch] = _features(candidates[twin], kind) @ coefficient
    return prediction


def _utility_from_displacement(dataset: dict[str, np.ndarray], displacement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    current = dataset["decision_public_state"][:, :, 0].astype(np.float64)
    goal = dataset["decision_public_state"][:, :, 2].astype(np.float64)
    initial_distance = np.abs(goal - current)[:, :, None]
    endpoint_distance = np.abs(goal[:, :, None] - current[:, :, None] - displacement)
    distance = np.minimum(initial_distance, endpoint_distance)
    utility = -distance
    scale = float(PROTOCOL["success_radius"]) / 2.0
    probability = 1.0 / (1.0 + np.exp((distance - float(PROTOCOL["success_radius"])) / scale))
    return probability, utility


def _constant_prediction(dataset: dict[str, np.ndarray], value: np.ndarray) -> dict[str, np.ndarray]:
    shape = dataset["success"].shape
    utility = np.broadcast_to(value, shape).astype(np.float64).copy()
    probability = np.full(shape, 0.5, dtype=np.float64)
    return {"probability": probability, "utility": utility, "displacement": np.zeros(shape, dtype=np.float64)}


def baseline_predictions(dataset: dict[str, np.ndarray], variant: str = "original") -> dict[str, dict[str, np.ndarray]]:
    actions, results = _history(dataset, variant)
    candidates = peak_candidates(dataset)
    twins, branches, count = dataset["success"].shape
    output: dict[str, dict[str, np.ndarray]] = {}
    output["class_prior"] = _constant_prediction(dataset, np.asarray(0.0))
    current = dataset["decision_public_state"][:, :, 0].astype(np.float64)
    goal = dataset["decision_public_state"][:, :, 2].astype(np.float64)
    output["current_only"] = _constant_prediction(dataset, -np.abs(goal - current)[:, :, None])
    slot = -np.arange(count, dtype=np.float64)[None, None, :] / max(count - 1, 1)
    output["candidate_index"] = _constant_prediction(dataset, slot)

    radius = np.linalg.norm(candidates, axis=2)
    generic_displacement = float(PROTOCOL["symbolic_displacement_gain"]) * radius
    displacement = np.broadcast_to(generic_displacement[:, None], (twins, branches, count)).copy()
    probability, utility = _utility_from_displacement(dataset, displacement)
    output["action_only"] = {"probability": probability, "utility": utility, "displacement": displacement}

    result_level = -np.mean(np.abs(results), axis=2)[:, :, None]
    output["result_only"] = _constant_prediction(dataset, result_level)

    norm = np.maximum(np.linalg.norm(actions, axis=3), 1.0e-12)
    unit = actions / norm[..., None]
    arrow = np.mean(results[..., None] * unit, axis=2)
    candidate_norm = np.maximum(np.linalg.norm(candidates, axis=2), 1.0e-12)
    candidate_unit = candidates / candidate_norm[..., None]
    mean_probe_radius = np.maximum(np.mean(norm, axis=2), 1.0e-12)
    displacement = np.einsum("nbi,nki->nbk", arrow, candidate_unit) * candidate_norm[:, None, :] / mean_probe_radius[:, :, None]
    probability, utility = _utility_from_displacement(dataset, displacement)
    output["arrow_action_compatibility"] = {"probability": probability, "utility": utility, "displacement": displacement}

    probe_success = dataset.get("probe_success")
    if probe_success is None:
        probe_success = np.zeros(results.shape, dtype=bool)
    failed = ~probe_success.astype(bool)
    last_action = np.zeros((twins, branches, 2), dtype=np.float64)
    last_result = np.zeros((twins, branches), dtype=np.float64)
    for probe in range(actions.shape[2]):
        last_action = np.where(failed[:, :, probe, None], actions[:, :, probe], last_action)
        last_result = np.where(failed[:, :, probe], results[:, :, probe], last_result)
    similarity = np.einsum(
        "nbi,nki->nbk", last_action / np.maximum(np.linalg.norm(last_action, axis=2, keepdims=True), 1.0e-12), candidate_unit
    )
    displacement = similarity * last_result[:, :, None] * candidate_norm[:, None, :] / np.maximum(np.linalg.norm(last_action, axis=2)[:, :, None], 1.0e-12)
    probability, utility = _utility_from_displacement(dataset, displacement)
    output["repeat_last_failed_action"] = {"probability": probability, "utility": utility, "displacement": displacement}

    nearest = np.zeros((twins, branches, count), dtype=np.float64)
    knn = np.zeros_like(nearest)
    for twin in range(twins):
        for branch in range(branches):
            difference = candidates[twin, :, None, :] - actions[twin, branch, None, :, :]
            distance = np.linalg.norm(difference, axis=2)
            order = np.argsort(distance, axis=1, kind="mergesort")
            nearest[twin, branch] = results[twin, branch, order[:, 0]]
            chosen = order[:, :2]
            weights = 1.0 / np.maximum(np.take_along_axis(distance, chosen, axis=1), 1.0e-6)
            values = results[twin, branch, chosen]
            knn[twin, branch] = np.sum(weights * values, axis=1) / np.sum(weights, axis=1)
    for name, displacement in (("nearest_historical_action", nearest), ("knn_action_result_lookup", knn)):
        probability, utility = _utility_from_displacement(dataset, displacement)
        output[name] = {"probability": probability, "utility": utility, "displacement": displacement}

    for name, kind in (("linear_sysid", "linear"), ("quadratic_fourier_sysid", "fourier")):
        displacement = _least_squares(actions, results, candidates, kind)
        probability, utility = _utility_from_displacement(dataset, displacement)
        output[name] = {"probability": probability, "utility": utility, "displacement": displacement}

    phi = dataset["latent_phi"].astype(np.float64)
    alpha = dataset["latent_alpha"].astype(np.float64)
    beta = dataset["latent_beta"].astype(np.float64)
    angle = np.arctan2(candidates[..., 1], candidates[..., 0])
    radius = np.linalg.norm(candidates, axis=2)
    beta_factor = sum(np.square(PROTOCOL["pulse_profile"])) / sum(PROTOCOL["pulse_profile"])
    radial = alpha[:, :, None] * radius[:, None, :] - beta_factor * beta[:, :, None] * np.square(radius[:, None, :])
    displacement = float(PROTOCOL["symbolic_displacement_gain"]) * radial * np.cos(2.0 * (angle[:, None, :] - phi[:, :, None]))
    probability, utility = _utility_from_displacement(dataset, displacement)
    output["oracle_latent_rollout"] = {"probability": probability, "utility": utility, "displacement": displacement}
    return output


def _average_ranks(value: np.ndarray) -> np.ndarray:
    order = np.argsort(value, kind="mergesort")
    ranks = np.empty(len(value), dtype=np.float64)
    start = 0
    while start < len(value):
        end = start + 1
        while end < len(value) and value[order[end]] == value[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def _auroc(label: np.ndarray, score: np.ndarray) -> float:
    positive, negative = score[label], score[~label]
    if len(positive) == 0 or len(negative) == 0:
        return float("nan")
    return float((positive[:, None] > negative).mean() + 0.5 * (positive[:, None] == negative).mean())


def _auprc(label: np.ndarray, score: np.ndarray) -> float:
    if not np.any(label):
        return float("nan")
    ranked = label[np.argsort(-score, kind="mergesort")].astype(np.float64)
    return float(np.sum(np.cumsum(ranked) / (np.arange(len(ranked)) + 1) * ranked) / np.sum(ranked))


def _balanced_accuracy(label: np.ndarray, score: np.ndarray) -> float:
    prediction = score >= 0.5
    positive = np.mean(prediction[label]) if np.any(label) else np.nan
    negative = np.mean(~prediction[~label]) if np.any(~label) else np.nan
    return float(np.nanmean((positive, negative)))


def _ece(label: np.ndarray, score: np.ndarray) -> float:
    value = 0.0
    for lower in np.linspace(0.0, 1.0, 10, endpoint=False):
        upper = lower + 0.1
        select = (score >= lower) & (score < upper if upper < 1.0 else score <= upper)
        if np.any(select):
            value += np.mean(select) * abs(float(np.mean(label[select])) - float(np.mean(score[select])))
    return float(value)


def _ranking(true: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    spearman, ndcg, regret, pairwise, top1 = [], [], [], [], []
    for truth, score in zip(true.reshape(-1, true.shape[-1]), predicted.reshape(-1, predicted.shape[-1])):
        tr, pr = _average_ranks(truth), _average_ranks(score)
        spearman.append(0.0 if np.std(tr) == 0 or np.std(pr) == 0 else float(np.corrcoef(tr, pr)[0, 1]))
        pairs = []
        for left in range(len(truth)):
            for right in range(left + 1, len(truth)):
                if abs(truth[left] - truth[right]) <= 1.0e-12:
                    continue
                difference = score[left] - score[right]
                pairs.append(0.5 if abs(difference) <= 1.0e-12 else float(np.sign(difference) == np.sign(truth[left] - truth[right])))
        pairwise.append(float(np.mean(pairs)) if pairs else 0.5)
        selected = int(np.argmax(score)); best = float(np.max(truth)); worst = float(np.min(truth))
        top1.append(float(truth[selected])); regret.append((best - float(truth[selected])) / max(best - worst, 1.0e-12))
        relevance = truth - np.min(truth)
        gains = np.power(2.0, relevance / max(float(np.max(relevance)), 1.0e-12)) - 1.0
        discount = 1.0 / np.log2(np.arange(len(truth)) + 2.0)
        dcg = float(np.sum(gains[np.argsort(-score, kind="mergesort")] * discount))
        ideal = float(np.sum(np.sort(gains)[::-1] * discount))
        ndcg.append(dcg / ideal if ideal > 0 else 1.0)
    return {
        "spearman": float(np.mean(spearman)), "ndcg": float(np.mean(ndcg)),
        "normalized_regret": float(np.mean(regret)), "top1_utility": float(np.mean(top1)),
        "within_twin_pairwise_ranking_accuracy": float(np.mean(pairwise)),
    }


def centered_twin_preference(
    true: np.ndarray, predicted: np.ndarray, template_id: np.ndarray
) -> tuple[float, np.ndarray, list[float], np.ndarray, np.ndarray]:
    true_centered = true - true.mean(axis=2, keepdims=True)
    predicted_centered = predicted - predicted.mean(axis=2, keepdims=True)
    true_difference = true_centered[:, 1] - true_centered[:, 0]
    predicted_difference = predicted_centered[:, 1] - predicted_centered[:, 0]
    informative = np.abs(true_difference) > 1.0e-12
    correct = np.where(
        informative,
        np.where(np.abs(predicted_difference) <= 1.0e-12, 0.5, (np.sign(true_difference) == np.sign(predicted_difference)).astype(np.float64)),
        np.nan,
    )
    by_twin = np.nanmean(correct, axis=1)
    per_template = []
    for template in range(int(np.max(template_id)) + 1):
        values = []
        for twin in range(len(template_id)):
            slot = np.flatnonzero(template_id[twin] == template)
            if len(slot) == 1 and not np.isnan(correct[twin, slot[0]]):
                values.append(correct[twin, slot[0]])
        per_template.append(float(np.mean(values)) if values else float("nan"))
    return float(np.nanmean(by_twin)), by_twin, per_template, true_centered, predicted_centered


def metrics(dataset: dict[str, np.ndarray], prediction: dict[str, np.ndarray]) -> dict:
    label = dataset["success"].reshape(-1).astype(bool)
    probability = np.clip(prediction["probability"].reshape(-1), 0.0, 1.0)
    true = dataset["utility"].astype(np.float64)
    predicted = prediction["utility"].astype(np.float64)
    centered, by_twin, per_template, true_centered, predicted_centered = centered_twin_preference(
        true, predicted, dataset["candidate_template_id"]
    )
    raw_difference = true[:, 1] - true[:, 0]
    raw_predicted = predicted[:, 1] - predicted[:, 0]
    informative = np.abs(raw_difference) > 1.0e-12
    raw_correct = np.where(informative, np.where(np.abs(raw_predicted) <= 1.0e-12, 0.5, np.sign(raw_difference) == np.sign(raw_predicted)), np.nan)
    return {
        "auroc": _auroc(label, probability), "auprc": _auprc(label, probability),
        "balanced_accuracy": _balanced_accuracy(label, probability),
        "brier": float(np.mean(np.square(probability - label.astype(np.float64)))),
        "ece": _ece(label, probability), "utility_mae": float(np.mean(np.abs(true - predicted))),
        "raw_twin_preference_accuracy": float(np.nanmean(raw_correct)),
        "candidate_conditioned_twin_preference_accuracy": centered,
        "centered_binding_score": float(2.0 * (centered - 0.5)),
        "candidate_conditioned_by_twin": by_twin.tolist(),
        "per_candidate_template_twin_preference": per_template,
        "true_centered_utility_mean_abs": float(np.mean(np.abs(true_centered))),
        "predicted_centered_utility_mean_abs": float(np.mean(np.abs(predicted_centered))),
        **_ranking(true, predicted),
    }


def bootstrap_difference(left: np.ndarray, right: np.ndarray) -> dict:
    values = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    rng = np.random.default_rng(int(PROTOCOL["bootstrap_seed"]))
    samples = [float(np.mean(values[rng.integers(0, len(values), len(values))])) for _ in range(int(PROTOCOL["bootstrap_replicates"]))]
    return {"mean": float(np.mean(values)), "ci95": [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))], "unit": "twin/world"}


def evaluate_all(dataset: dict[str, np.ndarray]) -> tuple[dict, dict, dict]:
    variants = ("original", "pair_preserving_shuffle", "binding_breaking_shuffle", "twin_history_swap")
    predictions = {variant: baseline_predictions(dataset, variant) for variant in variants}
    metric_sets = {variant: {name: metrics(dataset, value) for name, value in group.items()} for variant, group in predictions.items()}
    permuted = candidate_slot_permutation(dataset); rotated = coordinate_rotation(dataset)
    permuted_predictions = baseline_predictions(permuted); rotated_predictions = baseline_predictions(rotated)
    metric_sets["candidate_slot_permutation"] = {name: metrics(permuted, value) for name, value in permuted_predictions.items()}
    metric_sets["coordinate_rotation"] = {name: metrics(rotated, value) for name, value in rotated_predictions.items()}
    predictions["candidate_slot_permutation"] = permuted_predictions
    predictions["coordinate_rotation"] = rotated_predictions
    datasets = {variant: dataset for variant in variants}
    datasets["candidate_slot_permutation"] = permuted; datasets["coordinate_rotation"] = rotated
    return predictions, metric_sets, datasets


def save_prediction_rows(path: Path, datasets: dict, predictions: dict) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "variant", "baseline", "twin_id", "branch", "slot", "template_id", "probability",
        "predicted_displacement", "predicted_utility", "predicted_centered_utility",
        "success", "true_utility", "true_centered_utility",
    )
    rows = 0
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for variant, group in predictions.items():
            dataset = datasets[variant]
            true_centered = dataset["utility"] - dataset["utility"].mean(axis=2, keepdims=True)
            for baseline, value in group.items():
                centered = value["utility"] - value["utility"].mean(axis=2, keepdims=True)
                for twin in range(value["utility"].shape[0]):
                    for branch in range(2):
                        for slot in range(value["utility"].shape[2]):
                            writer.writerow({
                                "variant": variant, "baseline": baseline, "twin_id": twin, "branch": branch,
                                "slot": slot, "template_id": int(dataset["candidate_template_id"][twin, slot]),
                                "probability": float(value["probability"][twin, branch, slot]),
                                "predicted_displacement": float(value["displacement"][twin, branch, slot]),
                                "predicted_utility": float(value["utility"][twin, branch, slot]),
                                "predicted_centered_utility": float(centered[twin, branch, slot]),
                                "success": bool(dataset["success"][twin, branch, slot]),
                                "true_utility": float(dataset["utility"][twin, branch, slot]),
                                "true_centered_utility": float(true_centered[twin, branch, slot]),
                            }); rows += 1
    return rows


def gate_summary(dataset: dict[str, np.ndarray], metric_sets: dict, integrity: dict) -> dict:
    original = metric_sets["original"]
    key = "candidate_conditioned_twin_preference_accuracy"
    marginal_fail = [name for name in MARGINAL_BASELINES if original[name][key] >= float(PROTOCOL["marginal_shortcut_threshold"])]
    interaction_fail = [name for name in INTERACTION_BASELINES if original[name][key] >= float(PROTOCOL["interaction_shortcut_threshold"])]
    fit = "quadratic_fourier_sysid"
    fit_by_twin = np.asarray(original[fit]["candidate_conditioned_by_twin"])
    oracle_by_twin = np.asarray(original["oracle_latent_rollout"]["candidate_conditioned_by_twin"])
    best_marginal = max(MARGINAL_BASELINES, key=lambda name: original[name][key])
    marginal_by_twin = np.asarray(original[best_marginal]["candidate_conditioned_by_twin"])
    fit_bootstrap = bootstrap_difference(fit_by_twin, marginal_by_twin)
    oracle_bootstrap = bootstrap_difference(oracle_by_twin, marginal_by_twin)
    no_signal = fit_bootstrap["ci95"][0] <= 0.0 or oracle_bootstrap["ci95"][0] <= 0.0
    binding_drop = original[fit][key] - metric_sets["binding_breaking_shuffle"][fit][key]
    pair_change = abs(original[fit][key] - metric_sets["pair_preserving_shuffle"][fit][key])
    slot_error = max(abs(original[fit][metric] - metric_sets["candidate_slot_permutation"][fit][metric]) for metric in (key, "ndcg", "normalized_regret"))
    rotation_error = max(abs(original[fit][metric] - metric_sets["coordinate_rotation"][fit][metric]) for metric in (key, "ndcg", "normalized_regret"))
    pass_conditions = {
        "integrity": bool(integrity.get("passed", False)),
        "marginal_clear": not marginal_fail,
        "interaction_clear": not interaction_fail,
        "sysid_novel": original[fit][key] >= float(PROTOCOL["sysid_novel_min"]),
        "binding_drop": binding_drop >= float(PROTOCOL["binding_drop_min"]),
        "pair_shuffle": pair_change <= float(PROTOCOL["pair_shuffle_max_change"]),
        "slot_invariance": slot_error <= float(PROTOCOL["invariance_tolerance"]),
        "rotation_invariance": rotation_error <= float(PROTOCOL["invariance_tolerance"]),
        "identifiable": not no_signal,
    }
    if not pass_conditions["integrity"]:
        gate = "DATA_INTEGRITY_FAIL"
    elif marginal_fail:
        gate = "MARGINAL_SHORTCUT_FAIL"
    elif interaction_fail:
        gate = "INTERACTION_SHORTCUT_FAIL"
    elif no_signal:
        gate = "NO_IDENTIFIABLE_SIGNAL"
    elif all(pass_conditions.values()):
        gate = "BINDING_AUDIT_PASS"
    else:
        gate = "BINDING_AUDIT_FAIL"
    return {
        "task": _scalar(dataset["task"]), "gate": gate, "pass_conditions": pass_conditions,
        "marginal_failures": marginal_fail, "interaction_failures": interaction_fail,
        "best_marginal": best_marginal, "binding_drop": float(binding_drop),
        "pair_preserving_change": float(pair_change), "slot_invariance_error": float(slot_error),
        "coordinate_rotation_invariance_error": float(rotation_error),
        "fit_vs_marginal_bootstrap": fit_bootstrap, "oracle_vs_marginal_bootstrap": oracle_bootstrap,
        "metrics": metric_sets,
    }


def audit_dataset(dataset: dict[str, np.ndarray], output: Path, integrity: dict) -> dict:
    predictions, metric_sets, datasets = evaluate_all(dataset)
    rows = save_prediction_rows(output / "baseline_predictions.csv", datasets, predictions)
    original_actions, original_results = _history(dataset, "original")
    pair_actions, pair_results = _history(dataset, "pair_preserving_shuffle")
    broken_actions, broken_results = _history(dataset, "binding_breaking_shuffle")
    swapped_actions, swapped_results = _history(dataset, "twin_history_swap")
    permuted = datasets["candidate_slot_permutation"]; rotated = datasets["coordinate_rotation"]
    np.savez_compressed(
        output / "intervention_samples.npz",
        original_probe_actions=original_actions, original_probe_results=original_results,
        pair_probe_actions=pair_actions, pair_probe_results=pair_results,
        broken_probe_actions=broken_actions, broken_probe_results=broken_results,
        swapped_probe_actions=swapped_actions, swapped_probe_results=swapped_results,
        fixed_current=dataset["decision_public_state"], fixed_candidates=dataset["candidate_actions"],
        original_utility=dataset["utility"], original_success=dataset["success"],
        candidate_template_id=dataset["candidate_template_id"],
        slot_permuted_candidates=permuted["candidate_actions"], slot_permuted_utility=permuted["utility"],
        slot_permuted_success=permuted["success"],
        rotated_probe_actions=rotated["probe_actions"], rotated_candidates=rotated["candidate_actions"],
        rotated_current=rotated["decision_public_state"],
    )
    summary = gate_summary(dataset, metric_sets, integrity)
    summary["prediction_rows"] = rows
    summary["intervention_invariants"] = {
        "pair_binding_preserved": bool(all(
            sorted(zip(map(tuple, original_actions[t, b]), original_results[t, b].tolist()))
            == sorted(zip(map(tuple, pair_actions[t, b]), pair_results[t, b].tolist()))
            for t in range(len(original_actions)) for b in range(2)
        )),
        "binding_actions_unchanged": bool(np.array_equal(original_actions, broken_actions)),
        "binding_result_multiset_unchanged": bool(np.array_equal(np.sort(original_results, axis=2), np.sort(broken_results, axis=2))),
        "binding_correspondence_changed": bool(not np.array_equal(original_results, broken_results)),
        "twin_history_swap_exact": bool(np.array_equal(swapped_actions[:, 0], original_actions[:, 1]) and np.array_equal(swapped_results[:, 0], original_results[:, 1])),
        "current_unchanged": True, "candidates_unchanged": True,
    }
    write_json(output / "audit_summary.json", summary)
    return summary
