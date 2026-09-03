"""Tie-aware candidate metrics and deterministic hierarchical bootstrap."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .schema import BindingDataset, PredictionBundle


TOLERANCE = 1.0e-12


def _rank_average(values: np.ndarray) -> np.ndarray:
    """Average ranks with ties; sufficient for dependency-free Spearman."""

    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def spearman(values: np.ndarray, targets: np.ndarray) -> float:
    x = _rank_average(np.asarray(values, dtype=np.float64).reshape(-1))
    y = _rank_average(np.asarray(targets, dtype=np.float64).reshape(-1))
    x -= x.mean()
    y -= y.mean()
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    return 0.0 if denominator <= TOLERANCE else float(np.dot(x, y) / denominator)


def align_candidate_values(
    dataset: BindingDataset,
    prediction: PredictionBundle,
    values: np.ndarray,
) -> np.ndarray:
    """Align any candidate-indexed value by ID, independent of slot order."""

    values = np.asarray(values)
    if values.shape[:3] != dataset.true_utility.shape:
        raise ValueError("candidate-indexed values have an invalid leading shape")
    aligned = np.empty_like(values)
    for twin in range(dataset.twins):
        for branch in range(2):
            index = {
                str(candidate_id): slot
                for slot, candidate_id in enumerate(prediction.candidate_ids[twin, branch])
            }
            aligned[twin, branch] = values[
                twin,
                branch,
                [index[str(candidate_id)] for candidate_id in dataset.candidate_ids[twin, branch]],
            ]
    return aligned


def align_scores(dataset: BindingDataset, prediction: PredictionBundle) -> np.ndarray:
    """Align scores by candidate identity so slot order cannot change a metric."""

    return align_candidate_values(dataset, prediction, prediction.scores)


def _twin_preference(truth: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    true_centered = truth - truth.mean(axis=2, keepdims=True)
    score_centered = scores - scores.mean(axis=2, keepdims=True)
    true_difference = true_centered[:, 1] - true_centered[:, 0]
    score_difference = score_centered[:, 1] - score_centered[:, 0]
    informative = np.abs(true_difference) > TOLERANCE
    tied = np.abs(score_difference) <= TOLERANCE
    correctness = np.full(true_difference.shape, np.nan, dtype=np.float64)
    correctness[informative & tied] = 0.5
    decided = informative & ~tied
    correctness[decided] = (
        np.sign(true_difference[decided]) == np.sign(score_difference[decided])
    )
    by_twin = np.nanmean(correctness, axis=1)
    return {
        "tpa": float(np.nanmean(by_twin)),
        "informative_fraction": float(np.mean(informative)),
        "tpa_by_twin": by_twin.tolist(),
    }


def score_prediction(dataset: BindingDataset, prediction: PredictionBundle) -> dict[str, Any]:
    truth = np.asarray(dataset.true_utility, dtype=np.float64)
    scores = align_scores(dataset, prediction)
    predicted_max = scores.max(axis=2, keepdims=True)
    predicted_ties = np.abs(scores - predicted_max) <= TOLERANCE
    tie_counts = predicted_ties.sum(axis=2)
    selected_utility = (truth * predicted_ties).sum(axis=2) / tie_counts
    optimal = truth.max(axis=2)
    regret = optimal - selected_utility
    span = truth.max(axis=2) - truth.min(axis=2)
    normalized = regret / np.maximum(span, TOLERANCE)
    true_best = np.abs(truth - optimal[..., None]) <= TOLERANCE
    top1_credit = (true_best & predicted_ties).sum(axis=2) / tie_counts
    by_twin = normalized.mean(axis=1)
    top1_by_twin = top1_credit.mean(axis=1)

    sorted_truth = np.sort(truth, axis=2)
    gap = sorted_truth[:, :, -1] - sorted_truth[:, :, -2]
    lower, upper = np.quantile(gap.reshape(-1), (1.0 / 3.0, 2.0 / 3.0))
    difficulty: dict[str, Any] = {}
    for name, selected in (
        ("hard", gap <= lower),
        ("medium", (gap > lower) & (gap <= upper)),
        ("easy", gap > upper),
    ):
        difficulty[name] = {
            "count": int(selected.sum()),
            "normalized_regret": float(normalized[selected].mean()),
            "top1_accuracy": float(top1_credit[selected].mean()),
        }

    preference = _twin_preference(truth, scores)
    centered = scores - scores.mean(axis=2, keepdims=True)
    output: dict[str, Any] = {
        "normalized_regret": float(normalized.mean()),
        "mean_regret": float(regret.mean()),
        "top1_accuracy": float(top1_credit.mean()),
        "tpa": preference["tpa"],
        "informative_fraction": preference["informative_fraction"],
        "normalized_regret_by_twin": by_twin.tolist(),
        "top1_by_twin": top1_by_twin.tolist(),
        "tpa_by_twin": preference["tpa_by_twin"],
        "difficulty": difficulty,
        "shortcut_diagnostics": {
            "candidate_constant_branch_fraction": float(
                np.mean(np.ptp(scores, axis=2) <= TOLERANCE)
            ),
            "branch_centered_identical_twin_fraction": float(
                np.mean(np.max(np.abs(centered[:, 0] - centered[:, 1]), axis=1) <= TOLERANCE)
            ),
            "mean_predicted_tie_count": float(tie_counts.mean()),
        },
    }
    if prediction.predicted_effects is not None:
        truth_effect = np.asarray(dataset.candidate_effects)
        predicted_effect = align_candidate_values(
            dataset, prediction, np.asarray(prediction.predicted_effects)
        )
        absolute_error = np.abs(truth_effect - predicted_effect)
        calibration = {
            "mean_absolute_effect_error": float(absolute_error.mean()),
        }
        if prediction.uncertainty is not None:
            uncertainty = align_candidate_values(
                dataset, prediction, np.asarray(prediction.uncertainty)
            )
            calibration.update(
                {
                    "coverage": float(np.mean(absolute_error <= uncertainty)),
                    "mean_interval_width": float(np.mean(2.0 * uncertainty)),
                    "uncertainty_error_spearman": spearman(uncertainty, absolute_error),
                }
            )
        output["effect_prediction"] = calibration
    return output


def per_seed_metrics(
    dataset: BindingDataset,
    metric: Mapping[str, Any],
) -> dict[str, dict[str, float]]:
    normalized = np.asarray(metric["normalized_regret_by_twin"], dtype=np.float64)
    top1 = np.asarray(metric["top1_by_twin"], dtype=np.float64)
    tpa = np.asarray(metric["tpa_by_twin"], dtype=np.float64)
    output: dict[str, dict[str, float]] = {}
    for seed in np.unique(dataset.seed_ids):
        selected = dataset.seed_ids == seed
        output[str(int(seed))] = {
            "twins": int(selected.sum()),
            "normalized_regret": float(normalized[selected].mean()),
            "top1_accuracy": float(top1[selected].mean()),
            "tpa": float(np.nanmean(tpa[selected])),
        }
    return output


def hierarchical_bootstrap_difference(
    dataset: BindingDataset,
    baseline_by_twin: np.ndarray,
    candidate_by_twin: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, float | int]:
    """Resample seeds, then matched twins; report baseline minus candidate regret."""

    seeds = np.unique(dataset.seed_ids)
    if len(seeds) < 2:
        return {
            "replicates": 0,
            "lower_95": float("nan"),
            "median": float("nan"),
            "upper_95": float("nan"),
        }
    rng = np.random.default_rng(seed)
    values = np.empty(replicates, dtype=np.float64)
    baseline = np.asarray(baseline_by_twin, dtype=np.float64)
    candidate = np.asarray(candidate_by_twin, dtype=np.float64)
    for draw in range(replicates):
        sampled_seeds = rng.choice(seeds, size=len(seeds), replace=True)
        seed_means = []
        for current_seed in sampled_seeds:
            indices = np.flatnonzero(dataset.seed_ids == current_seed)
            selected = rng.choice(indices, size=len(indices), replace=True)
            seed_means.append(float(np.mean(baseline[selected] - candidate[selected])))
        values[draw] = float(np.mean(seed_means))
    lower, median, upper = np.quantile(values, (0.025, 0.5, 0.975))
    return {
        "replicates": int(replicates),
        "lower_95": float(lower),
        "median": float(median),
        "upper_95": float(upper),
    }


__all__ = [
    "align_scores",
    "align_candidate_values",
    "hierarchical_bootstrap_difference",
    "per_seed_metrics",
    "score_prediction",
    "spearman",
]
