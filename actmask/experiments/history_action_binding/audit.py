"""No-training baselines, binding interventions, metrics, and admission gates."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from actmask.experiments.history_action_binding.common import OUTPUT_ROOT, PROTOCOL, write_json


BASELINES = (
    "class_prior",
    "current_only",
    "candidate_index",
    "action_only",
    "result_only",
    "arrow_action_compatibility",
    "repeat_last_failed_action",
    "nearest_historical_action",
    "knn_action_result_lookup",
    "linear_least_squares_response",
    "quadratic_least_squares_response",
    "oracle_latent_rollout",
)
MARGINAL_BASELINES = (
    "class_prior", "current_only", "candidate_index", "action_only", "result_only",
)
SIMPLE_SHORTCUTS = MARGINAL_BASELINES + (
    "arrow_action_compatibility", "repeat_last_failed_action", "nearest_historical_action",
    "knn_action_result_lookup",
)
EXPLICIT_FITS = ("linear_least_squares_response", "quadratic_least_squares_response")


def pair_preserving_shuffle(actions: np.ndarray, results: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    permutation = np.roll(np.arange(actions.shape[2]), 1)
    return actions[:, :, permutation].copy(), results[:, :, permutation].copy()


def binding_breaking_shuffle(actions: np.ndarray, results: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    permutation = np.roll(np.arange(results.shape[2]), 1)
    return actions.copy(), results[:, :, permutation].copy()


def twin_history_swap(actions: np.ndarray, results: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return actions[:, ::-1].copy(), results[:, ::-1].copy()


def candidate_slot_permutation(dataset: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    copied = {key: value.copy() for key, value in dataset.items()}
    count = dataset["candidate_actions"].shape[1]
    permutation = np.arange(count - 1, -1, -1)
    candidate_axis = {
        "candidate_actions": 1,
        "candidate_actions_by_branch": 2,
        "candidate_template_order": 1,
        "candidate_trace_public": 2,
        "candidate_trace_response_pos": 2,
        "candidate_trace_response_vel": 2,
        "candidate_trace_actuator_pos": 2,
        "candidate_trace_target_pos": 2,
        "candidate_trace_distance": 2,
        "candidate_trace_contact_magnitude": 2,
        "first_success_step": 2,
        "min_distance": 2,
        "final_error": 2,
        "utility": 2,
        "success": 2,
        "final_state_hash": 2,
    }
    for key, axis in candidate_axis.items():
        if key in copied:
            copied[key] = np.take(copied[key], permutation, axis=axis)
    return copied


def coordinate_reflection(dataset: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    copied = {key: value.copy() for key, value in dataset.items()}
    for key in ("probe_observation_before", "probe_observation_after", "probe_delta"):
        copied[key] *= -1.0
    copied["probe_action"] *= -1.0
    copied["candidate_actions"] *= -1.0
    copied["candidate_actions_by_branch"] *= -1.0
    copied["decision_public_state"] *= -1.0
    if "decision_physical_state" in copied:
        copied["decision_physical_state"] *= -1.0
    return copied


def _history_arrays(dataset: dict[str, np.ndarray], variant: str) -> tuple[np.ndarray, np.ndarray]:
    actions = dataset["probe_action"].astype(np.float64)
    results = dataset["probe_delta"][..., 0].astype(np.float64)
    if variant == "original":
        return actions, results
    if variant == "pair_preserving_shuffle":
        return pair_preserving_shuffle(actions, results)
    if variant == "binding_breaking_shuffle":
        return binding_breaking_shuffle(actions, results)
    if variant == "twin_history_swap":
        return twin_history_swap(actions, results)
    raise ValueError(variant)


def _candidate_amplitude(dataset: dict[str, np.ndarray]) -> np.ndarray:
    return dataset["candidate_actions"][:, :, 1, 0].astype(np.float64)


def _predict_from_distance(distance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    utility = -np.asarray(distance, dtype=np.float64)
    scale = float(PROTOCOL["success_radius"]) / 2.0
    probability = 1.0 / (1.0 + np.exp((distance - float(PROTOCOL["success_radius"])) / scale))
    return probability, utility


def _least_squares_prediction(
    actions: np.ndarray, results: np.ndarray, candidates: np.ndarray, *, quadratic: bool
) -> np.ndarray:
    twins, branches, _ = actions.shape
    prediction = np.zeros((twins, branches, candidates.shape[1]), dtype=np.float64)
    for twin in range(twins):
        for branch in range(branches):
            u = actions[twin, branch]
            columns = [np.ones_like(u), u]
            if quadratic:
                columns.append(u * np.abs(u))
            design = np.stack(columns, axis=1)
            coefficient = np.linalg.lstsq(design, results[twin, branch], rcond=None)[0]
            c = candidates[twin]
            candidate_design = [np.ones_like(c), c]
            if quadratic:
                candidate_design.append(c * np.abs(c))
            prediction[twin, branch] = np.stack(candidate_design, axis=1) @ coefficient
    return prediction


def baseline_predictions(
    dataset: dict[str, np.ndarray], *, variant: str = "original"
) -> dict[str, dict[str, np.ndarray]]:
    actions, results = _history_arrays(dataset, variant)
    candidates = _candidate_amplitude(dataset)
    current = dataset["decision_public_state"].astype(np.float64)
    goal = current[:, :, 3]
    current_x = current[:, :, 0]
    twins, branches, candidate_count = dataset["success"].shape
    output: dict[str, dict[str, np.ndarray]] = {}

    constant_distance = np.full((twins, branches, candidate_count), 0.055, dtype=np.float64)
    probability, utility = _predict_from_distance(constant_distance)
    output["class_prior"] = {"probability": np.full_like(probability, 0.5), "utility": utility}

    distance = np.broadcast_to(np.abs(goal - current_x)[:, :, None], constant_distance.shape)
    probability, utility = _predict_from_distance(distance)
    output["current_only"] = {"probability": probability, "utility": utility}

    slot = np.broadcast_to(np.arange(candidate_count, dtype=np.float64)[None, None, :], constant_distance.shape)
    output["candidate_index"] = {
        "probability": 1.0 - slot / max(candidate_count - 1, 1),
        "utility": -slot / max(candidate_count - 1, 1),
    }

    nominal_response = 0.0578 * candidates
    distance = np.abs(goal[:, :, None] - current_x[:, :, None] - nominal_response[:, None, :])
    probability, utility = _predict_from_distance(distance)
    output["action_only"] = {"probability": probability, "utility": utility}

    result_summary = np.mean(np.abs(results), axis=2)
    distance = np.broadcast_to(np.abs(goal - current_x - result_summary)[:, :, None], constant_distance.shape)
    probability, utility = _predict_from_distance(distance)
    output["result_only"] = {"probability": probability, "utility": utility}

    arrow = results[:, :, -1] - results[:, :, 0]
    signed = arrow[:, :, None] * candidates[:, None, :]
    distance = np.where(signed > 0, np.abs(goal[:, :, None] - current_x[:, :, None] - 0.0578 * np.abs(candidates[:, None, :])), np.abs(goal[:, :, None] - current_x[:, :, None]))
    probability, utility = _predict_from_distance(distance)
    output["arrow_action_compatibility"] = {"probability": probability, "utility": utility}

    failed = ~dataset["probe_success"]
    last_failed = np.zeros((twins, branches), dtype=np.float64)
    for probe in range(actions.shape[2]):
        last_failed = np.where(failed[:, :, probe], actions[:, :, probe], last_failed)
    distance_action = np.abs(candidates[:, None, :] - last_failed[:, :, None])
    output["repeat_last_failed_action"] = {
        "probability": np.exp(-distance_action / 0.25),
        "utility": -distance_action,
    }

    nearest_utility = np.zeros_like(constant_distance)
    knn_utility = np.zeros_like(constant_distance)
    for twin in range(twins):
        for branch in range(branches):
            distance_action = np.abs(candidates[twin, :, None] - actions[twin, branch, None, :])
            nearest = np.argmin(distance_action, axis=1)
            nearest_utility[twin, branch] = dataset["probe_utility"][twin, branch, nearest]
            weights = 1.0 / np.maximum(distance_action, 1.0e-3)
            knn_utility[twin, branch] = (weights * dataset["probe_utility"][twin, branch][None, :]).sum(1) / weights.sum(1)
    for name, predicted in (("nearest_historical_action", nearest_utility), ("knn_action_result_lookup", knn_utility)):
        probability = 1.0 / (1.0 + np.exp((-predicted - float(PROTOCOL["success_radius"])) / (float(PROTOCOL["success_radius"]) / 2.0)))
        output[name] = {"probability": probability, "utility": predicted}

    for name, quadratic in (("linear_least_squares_response", False), ("quadratic_least_squares_response", True)):
        displacement = _least_squares_prediction(actions, results, candidates, quadratic=quadratic)
        distance = np.abs(goal[:, :, None] - current_x[:, :, None] - displacement)
        probability, utility = _predict_from_distance(distance)
        output[name] = {"probability": probability, "utility": utility}

    task = str(dataset["task"])
    if task == "task_a_discrete_mode":
        response = dataset["latent_mode"].astype(np.float64)[:, :, None] * candidates[:, None, :]
    else:
        c = candidates[:, None, :]
        response = dataset["latent_alpha"].astype(np.float64)[:, :, None] * c - dataset["latent_beta"].astype(np.float64)[:, :, None] * c * np.abs(c)
    distance = np.abs(goal[:, :, None] - current_x[:, :, None] - 0.0578 * response)
    probability, utility = _predict_from_distance(distance)
    output["oracle_latent_rollout"] = {"probability": probability, "utility": utility}
    return output


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    index = 0
    while index < len(values):
        end = index + 1
        while end < len(values) and values[order[end]] == values[order[index]]:
            end += 1
        ranks[order[index:end]] = (index + end - 1) / 2.0
        index = end
    return ranks


def _auroc(label: np.ndarray, score: np.ndarray) -> float:
    positives = score[label]
    negatives = score[~label]
    if len(positives) == 0 or len(negatives) == 0:
        return float("nan")
    return float(((positives[:, None] > negatives[None, :]).mean() + 0.5 * (positives[:, None] == negatives[None, :]).mean()))


def _average_precision(label: np.ndarray, score: np.ndarray) -> float:
    if not np.any(label):
        return float("nan")
    order = np.argsort(-score, kind="mergesort")
    ranked = label[order].astype(np.float64)
    precision = np.cumsum(ranked) / (np.arange(len(ranked)) + 1)
    return float(np.sum(precision * ranked) / np.sum(ranked))


def _balanced_accuracy(label: np.ndarray, probability: np.ndarray) -> float:
    prediction = probability >= 0.5
    true_positive = np.mean(prediction[label]) if np.any(label) else np.nan
    true_negative = np.mean(~prediction[~label]) if np.any(~label) else np.nan
    return float(np.nanmean((true_positive, true_negative)))


def _ece(label: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    total = len(label)
    value = 0.0
    for lower in np.linspace(0.0, 1.0, bins, endpoint=False):
        upper = lower + 1.0 / bins
        select = (probability >= lower) & (probability < upper if upper < 1.0 else probability <= upper)
        if np.any(select):
            value += np.sum(select) / total * abs(float(label[select].mean()) - float(probability[select].mean()))
    return float(value)


def _ranking_metrics(true_utility: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    spearman = []
    top1 = []
    ndcg = []
    regret = []
    for true, score in zip(true_utility.reshape(-1, true_utility.shape[-1]), predicted.reshape(-1, predicted.shape[-1])):
        true_rank = _average_ranks(true)
        predicted_rank = _average_ranks(score)
        if np.std(true_rank) == 0 or np.std(predicted_rank) == 0:
            spearman.append(0.0)
        else:
            spearman.append(float(np.corrcoef(true_rank, predicted_rank)[0, 1]))
        selected = int(np.argmax(score))
        best = float(np.max(true))
        worst = float(np.min(true))
        top1.append(float(true[selected]))
        regret.append((best - float(true[selected])) / max(best - worst, 1.0e-12))
        relevance = true - np.min(true)
        gains = np.power(2.0, relevance / max(float(np.max(relevance)), 1.0e-12)) - 1.0
        discounts = 1.0 / np.log2(np.arange(len(true)) + 2.0)
        dcg = float(np.sum(gains[np.argsort(-score, kind="mergesort")] * discounts))
        ideal = float(np.sum(np.sort(gains)[::-1] * discounts))
        ndcg.append(dcg / ideal if ideal > 0 else 1.0)
    return dict(
        spearman=float(np.mean(spearman)),
        top1_utility=float(np.mean(top1)),
        ndcg=float(np.mean(ndcg)),
        normalized_regret=float(np.mean(regret)),
    )


def _twin_preference(true_utility: np.ndarray, predicted: np.ndarray) -> tuple[float, np.ndarray]:
    true_difference = true_utility[:, 1] - true_utility[:, 0]
    predicted_difference = predicted[:, 1] - predicted[:, 0]
    informative = np.abs(true_difference) > 1.0e-9
    correct = np.where(
        informative,
        np.where(np.abs(predicted_difference) <= 1.0e-12, 0.5, (np.sign(true_difference) == np.sign(predicted_difference)).astype(np.float64)),
        np.nan,
    )
    by_twin = np.nanmean(correct, axis=1)
    return float(np.nanmean(by_twin)), by_twin


def metrics(dataset: dict[str, np.ndarray], prediction: dict[str, np.ndarray]) -> dict:
    label = dataset["success"].reshape(-1).astype(bool)
    probability = np.clip(prediction["probability"].reshape(-1), 0.0, 1.0)
    true_utility = dataset["utility"].astype(np.float64)
    predicted_utility = prediction["utility"].astype(np.float64)
    twin_accuracy, by_twin = _twin_preference(true_utility, predicted_utility)
    return dict(
        auroc=_auroc(label, probability),
        auprc=_average_precision(label, probability),
        balanced_accuracy=_balanced_accuracy(label, probability),
        brier=float(np.mean(np.square(probability - label.astype(np.float64)))),
        ece=_ece(label, probability),
        utility_mae=float(np.mean(np.abs(predicted_utility - true_utility))),
        twin_preference_accuracy=twin_accuracy,
        twin_preference_by_twin=by_twin.tolist(),
        **_ranking_metrics(true_utility, predicted_utility),
    )


def _bootstrap_difference(left: np.ndarray, right: np.ndarray) -> dict:
    values = np.asarray(left) - np.asarray(right)
    rng = np.random.default_rng(int(PROTOCOL["bootstrap_seed"]))
    samples = []
    for _ in range(int(PROTOCOL["bootstrap_replicates"])):
        selection = rng.integers(0, len(values), size=len(values))
        samples.append(float(np.mean(values[selection])))
    return dict(
        mean=float(np.mean(values)),
        ci95=[float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))],
        independent_unit="twin/world",
    )


def _branch_predictability(dataset: dict[str, np.ndarray]) -> dict:
    current = dataset["decision_public_state"].astype(np.float64)
    label = (dataset["latent_mode"] > dataset["latent_mode"].mean()).astype(bool) if str(dataset["task"]) == "task_a_discrete_mode" else (dataset["latent_alpha"] > dataset["latent_alpha"].mean()).astype(bool)
    features = current.reshape(-1, current.shape[-1])
    labels = label.reshape(-1)
    prediction = np.zeros_like(labels)
    twin_index = np.repeat(np.arange(current.shape[0]), current.shape[1])
    for index in range(len(labels)):
        train = twin_index != twin_index[index]
        centroids = []
        for value in (False, True):
            selected = train & (labels == value)
            centroids.append(features[selected].mean(0) if np.any(selected) else np.zeros(features.shape[1]))
        distances = [np.linalg.norm(features[index] - centroid) for centroid in centroids]
        prediction[index] = bool(np.argmin(distances))
    balanced = _balanced_accuracy(labels, prediction.astype(np.float64))
    return dict(method="leave-one-whole-twin-out nearest centroid on quantized public current", balanced_accuracy=balanced)


def audit_dataset(path: Path, output: Path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        dataset = {key: archive[key] for key in archive.files}
    output.mkdir(parents=True, exist_ok=True)
    variants = ("original", "pair_preserving_shuffle", "binding_breaking_shuffle", "twin_history_swap")
    all_predictions = {variant: baseline_predictions(dataset, variant=variant) for variant in variants}
    all_metrics = {
        variant: {name: metrics(dataset, prediction) for name, prediction in predictions.items()}
        for variant, predictions in all_predictions.items()
    }
    permuted = candidate_slot_permutation(dataset)
    permuted_predictions = baseline_predictions(permuted)
    permuted_metrics = {name: metrics(permuted, prediction) for name, prediction in permuted_predictions.items()}
    reflected = coordinate_reflection(dataset)
    reflected_predictions = baseline_predictions(reflected)
    reflected_metrics = {name: metrics(reflected, prediction) for name, prediction in reflected_predictions.items()}
    rows = []
    prediction_sets = {
        **{variant: (dataset, predictions) for variant, predictions in all_predictions.items()},
        "candidate_slot_permutation": (permuted, permuted_predictions),
        "coordinate_reflection": (reflected, reflected_predictions),
    }
    for variant, (variant_dataset, predictions) in prediction_sets.items():
        for baseline, values in predictions.items():
            for twin in range(values["probability"].shape[0]):
                for branch in range(2):
                    for candidate in range(values["probability"].shape[2]):
                        rows.append(dict(
                            variant=variant, baseline=baseline, twin_id=twin, branch=branch,
                            candidate=candidate, probability=float(values["probability"][twin, branch, candidate]),
                            predicted_utility=float(values["utility"][twin, branch, candidate]),
                            success=bool(variant_dataset["success"][twin, branch, candidate]),
                            true_utility=float(variant_dataset["utility"][twin, branch, candidate]),
                        ))
    with (output / "baseline_predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    original_actions, original_results = _history_arrays(dataset, "original")
    pair_actions, pair_results = _history_arrays(dataset, "pair_preserving_shuffle")
    broken_actions, broken_results = _history_arrays(dataset, "binding_breaking_shuffle")
    swapped_actions, swapped_results = _history_arrays(dataset, "twin_history_swap")
    slot_permutation = np.arange(dataset["candidate_actions"].shape[1] - 1, -1, -1)
    simulator_preference = np.sign(dataset["utility"][:, 1] - dataset["utility"][:, 0]).astype(np.int8)
    np.savez_compressed(
        output / "intervention_samples.npz",
        original_probe_actions=original_actions,
        original_probe_results=original_results,
        pair_preserving_probe_actions=pair_actions,
        pair_preserving_probe_results=pair_results,
        binding_break_probe_actions=broken_actions,
        binding_break_probe_results=broken_results,
        twin_swap_probe_actions=swapped_actions,
        twin_swap_probe_results=swapped_results,
        fixed_current_observation=dataset["decision_public_state"],
        fixed_candidate_actions=dataset["candidate_actions"],
        simulator_utility_target_original=dataset["utility"],
        simulator_success_target_original=dataset["success"],
        simulator_expected_preference_original=simulator_preference,
        simulator_expected_prediction_preference_after_twin_swap=-simulator_preference,
        novel_probe_magnitudes=np.unique(np.abs(original_actions)),
        novel_candidate_magnitudes=np.unique(np.abs(_candidate_amplitude(dataset))),
        candidate_slot_permutation=slot_permutation,
        slot_permuted_candidate_actions=permuted["candidate_actions"],
        slot_permuted_success=permuted["success"],
        slot_permuted_utility=permuted["utility"],
        slot_permuted_trace_distance=permuted["candidate_trace_distance"],
        reflected_probe_actions=reflected["probe_action"],
        reflected_probe_results=reflected["probe_delta"][..., 0],
        reflected_current_observation=reflected["decision_public_state"],
        reflected_candidate_actions=reflected["candidate_actions"],
        reflected_success=reflected["success"],
        reflected_utility=reflected["utility"],
    )

    best_marginal_name = max(MARGINAL_BASELINES, key=lambda name: all_metrics["original"][name]["twin_preference_accuracy"])
    best_fit_name = max(EXPLICIT_FITS, key=lambda name: all_metrics["original"][name]["twin_preference_accuracy"])
    fit_original = np.asarray(all_metrics["original"][best_fit_name]["twin_preference_by_twin"])
    fit_broken = np.asarray(all_metrics["binding_breaking_shuffle"][best_fit_name]["twin_preference_by_twin"])
    fit_shuffled = np.asarray(all_metrics["pair_preserving_shuffle"][best_fit_name]["twin_preference_by_twin"])
    marginal = np.asarray(all_metrics["original"][best_marginal_name]["twin_preference_by_twin"])
    oracle = np.asarray(all_metrics["original"]["oracle_latent_rollout"]["twin_preference_by_twin"])
    shortcut_saturated = [
        name for name in SIMPLE_SHORTCUTS
        if max(
            all_metrics["original"][name]["balanced_accuracy"],
            all_metrics["original"][name]["twin_preference_accuracy"],
            all_metrics["original"][name]["ndcg"],
        )
        >= float(PROTOCOL["saturation_threshold"])
    ]
    slot_error = max(
        abs(permuted_metrics[best_fit_name][key] - all_metrics["original"][best_fit_name][key])
        for key in ("ndcg", "normalized_regret", "twin_preference_accuracy")
    )
    reflection_error = max(
        abs(reflected_metrics[best_fit_name][key] - all_metrics["original"][best_fit_name][key])
        for key in ("ndcg", "normalized_regret", "twin_preference_accuracy")
    )
    binding_drop = float(np.nanmean(fit_original - fit_broken))
    pair_shuffle_change = float(abs(np.nanmean(fit_original) - np.nanmean(fit_shuffled)))
    branch_predictability = _branch_predictability(dataset)
    pair_binding_preserved = bool(
        all(
            sorted(zip(original_actions[twin, branch].tolist(), original_results[twin, branch].tolist()))
            == sorted(zip(pair_actions[twin, branch].tolist(), pair_results[twin, branch].tolist()))
            for twin in range(original_actions.shape[0]) for branch in range(original_actions.shape[1])
        )
    )
    binding_actions_unchanged = bool(np.array_equal(original_actions, broken_actions))
    binding_result_multiset_unchanged = bool(np.array_equal(np.sort(original_results, axis=2), np.sort(broken_results, axis=2)))
    binding_correspondence_changed = bool(not np.array_equal(original_results, broken_results))
    twin_swap_exact = bool(
        np.array_equal(swapped_actions[:, 0], original_actions[:, 1])
        and np.array_equal(swapped_actions[:, 1], original_actions[:, 0])
        and np.array_equal(swapped_results[:, 0], original_results[:, 1])
        and np.array_equal(swapped_results[:, 1], original_results[:, 0])
    )
    novel_disjoint = bool(
        set(np.unique(np.abs(original_actions))).isdisjoint(set(np.unique(np.abs(_candidate_amplitude(dataset)))))
    )
    no_signal = bool(
        _bootstrap_difference(oracle, marginal)["ci95"][0] <= 0.0
        and _bootstrap_difference(fit_original, marginal)["ci95"][0] <= 0.0
    )
    integrity_report = json.loads((path.parent / "generation_report.json").read_text())["integrity"]
    integrity_pass = bool(integrity_report["passed"] and branch_predictability["balanced_accuracy"] <= float(PROTOCOL["branch_predictability_max_balanced_accuracy"]))
    binding_pass = bool(
        integrity_pass
        and not shortcut_saturated
        and binding_drop >= float(PROTOCOL["binding_drop_threshold"])
        and pair_shuffle_change <= float(PROTOCOL["pair_shuffle_max_change"])
        and all_metrics["original"][best_fit_name]["twin_preference_accuracy"] >= float(PROTOCOL["novel_candidate_min_twin_accuracy"])
        and slot_error <= 1.0e-12
        and reflection_error <= 1.0e-12
        and not no_signal
    )
    if not integrity_pass:
        gate = "DATA_INTEGRITY_FAIL"
    elif shortcut_saturated:
        gate = "SHORTCUT_SATURATED"
    elif no_signal:
        gate = "NO_IDENTIFIABLE_SIGNAL"
    elif binding_pass:
        gate = "BINDING_AUDIT_PASS"
    else:
        gate = "BINDING_AUDIT_FAIL"
    outcome = dataset["success"]
    outcome_counts = {
        key: int(np.sum((outcome[:, 0] == bool(int(key[0]))) & (outcome[:, 1] == bool(int(key[1])))))
        for key in ("00", "01", "10", "11")
    }
    summary = dict(
        task=str(dataset["task"]),
        gate=gate,
        learned_training_authorized=gate == "BINDING_AUDIT_PASS",
        independent_unit="twin/world",
        twins=int(outcome.shape[0]),
        candidates_per_twin=int(outcome.shape[2]),
        outcome_counts=outcome_counts,
        flip_ratio=float(np.mean(outcome[:, 0] != outcome[:, 1])),
        mean_abs_twin_utility_difference=float(np.mean(np.abs(dataset["utility"][:, 0] - dataset["utility"][:, 1]))),
        branch_predictability=branch_predictability,
        current_state_matching=dict(
            observable_pair_max_error=float(np.max(np.abs(dataset["decision_public_state"][:, 0] - dataset["decision_public_state"][:, 1]))),
            physical_pair_max_error=float(np.max(np.abs(dataset["decision_physical_state"][:, 0] - dataset["decision_physical_state"][:, 1]))),
            max_position_error=float(np.max(np.abs(dataset["decision_physical_state"][:, :, 0]))),
            max_velocity_error=float(np.max(np.abs(dataset["decision_physical_state"][:, :, 1]))),
        ),
        best_marginal_baseline=best_marginal_name,
        best_explicit_fit=best_fit_name,
        shortcut_saturated_baselines=shortcut_saturated,
        binding_drop=binding_drop,
        pair_preserving_shuffle_change=pair_shuffle_change,
        slot_permutation_metric_max_error=slot_error,
        coordinate_reflection_metric_max_error=reflection_error,
        explicit_fit_vs_marginal_bootstrap=_bootstrap_difference(fit_original, marginal),
        oracle_vs_marginal_bootstrap=_bootstrap_difference(oracle, marginal),
        thresholds={key: PROTOCOL[key] for key in (
            "saturation_threshold", "binding_drop_threshold", "pair_shuffle_max_change",
            "novel_candidate_min_twin_accuracy", "branch_predictability_max_balanced_accuracy",
        )},
        metrics=all_metrics,
        intervention_invariants=dict(
            pair_preserving_action_result_binding_preserved=pair_binding_preserved,
            binding_break_actions_unchanged=binding_actions_unchanged,
            binding_break_result_multiset_unchanged=binding_result_multiset_unchanged,
            binding_break_only_correspondence_changed=binding_correspondence_changed,
            twin_swap_histories_exact=twin_swap_exact,
            twin_swap_current_unchanged=True,
            twin_swap_candidates_unchanged=True,
            twin_swap_expected_simulator_direction="predicted branch preferences should exchange",
            novel_candidate_magnitudes_disjoint=novel_disjoint,
            slot_permutation=permuted_metrics,
            coordinate_reflection=reflected_metrics,
        ),
    )
    write_json(output / "audit_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.dataset.parent / "audit"
    print(json.dumps(audit_dataset(args.dataset, output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
