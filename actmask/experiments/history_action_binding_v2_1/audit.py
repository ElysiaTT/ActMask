"""v2.1 audit surface using the frozen v2 metric implementation unchanged.

The corrected candidate-conditioned twin preference metric must not change in
v2.1.  This module therefore imports that implementation byte-for-byte from the
protected v2 source and only adds v2.1 preflight checks.
"""

from __future__ import annotations

import inspect

import numpy as np

from actmask.experiments.history_action_binding_v2.audit import (  # noqa: F401
    BASELINES,
    INTERACTION_BASELINES,
    MARGINAL_BASELINES,
    _features,
    _least_squares,
    _utility_from_displacement,
    audit_dataset,
    baseline_predictions,
    binding_breaking_shuffle,
    bootstrap_difference,
    candidate_slot_permutation,
    centered_twin_preference,
    coordinate_rotation,
    evaluate_all,
    gate_summary,
    metrics,
    pair_preserving_shuffle,
    peak_candidates,
    save_prediction_rows,
    twin_history_swap,
)

from actmask.experiments.history_action_binding_v2_1.common import PROTOCOL


METRIC_KEY = "candidate_conditioned_twin_preference_accuracy"
FIT_BASELINE = "quadratic_fourier_sysid"


def slot_balance_table(template_id: np.ndarray) -> np.ndarray:
    """Return slot-by-template counts for the complete candidate schedule."""
    template_id = np.asarray(template_id)
    count = int(template_id.shape[1])
    table = np.zeros((count, count), dtype=np.int64)
    for twin in range(len(template_id)):
        for slot in range(count):
            table[slot, int(template_id[twin, slot])] += 1
    return table


def top1_crossing_rate(dataset: dict[str, np.ndarray]) -> float:
    selected = np.argmax(dataset["utility"], axis=2)
    templates = dataset["candidate_template_id"]
    left = templates[np.arange(len(templates)), selected[:, 0]]
    right = templates[np.arange(len(templates)), selected[:, 1]]
    return float(np.mean(left != right))


def latent_blind_sampler_check(candidate_sampler) -> dict:
    signature = tuple(inspect.signature(candidate_sampler).parameters)
    source = inspect.getsource(candidate_sampler).lower()
    forbidden = ("latent_phi", "alpha", "beta", "branch", "success", "utility", "label", "future")
    hits = [word for word in forbidden if word in source]
    return {
        "passed": signature == ("task", "public_rotation", "candidate_seed") and not hits,
        "signature": list(signature),
        "forbidden_source_hits": hits,
    }


def symbolic_gate_checks(
    dataset: dict[str, np.ndarray], metric_sets: dict, candidate_sampler
) -> tuple[dict, dict]:
    original = metric_sets["original"]
    key = METRIC_KEY
    fit = FIT_BASELINE
    table = slot_balance_table(dataset["candidate_template_id"])
    expected = len(dataset["candidate_template_id"]) // int(dataset["candidate_template_id"].shape[1])
    marginal_scores = {name: float(original[name][key]) for name in MARGINAL_BASELINES}
    interaction_scores = {name: float(original[name][key]) for name in INTERACTION_BASELINES}
    fit_score = float(original[fit][key])
    binding_score = float(metric_sets["binding_breaking_shuffle"][fit][key])
    pair_score = float(metric_sets["pair_preserving_shuffle"][fit][key])
    slot_score = float(metric_sets["candidate_slot_permutation"][fit][key])
    rotation_score = float(metric_sets["coordinate_rotation"][fit][key])
    fit_by_twin = np.asarray(original[fit]["candidate_conditioned_by_twin"], dtype=np.float64)
    oracle_by_twin = np.asarray(original["oracle_latent_rollout"]["candidate_conditioned_by_twin"], dtype=np.float64)
    best_marginal = max(MARGINAL_BASELINES, key=lambda name: original[name][key])
    marginal_by_twin = np.asarray(original[best_marginal]["candidate_conditioned_by_twin"], dtype=np.float64)
    fit_bootstrap = bootstrap_difference(fit_by_twin, marginal_by_twin)
    oracle_bootstrap = bootstrap_difference(oracle_by_twin, marginal_by_twin)
    sampler_check = latent_blind_sampler_check(candidate_sampler)
    result_multiset = np.array_equal(
        np.sort(dataset["probe_delta"][:, 0], axis=1),
        np.sort(dataset["probe_delta"][:, 1], axis=1),
    )
    utility_multiset = np.array_equal(
        np.sort(dataset["probe_utility"][:, 0], axis=1),
        np.sort(dataset["probe_utility"][:, 1], axis=1),
    )

    # "Conclusion invariant" means the transformed score remains on the same
    # side of every applicable frozen threshold, not numerical bit identity.
    def same_conclusion(transformed: float) -> bool:
        return (
            (fit_score >= float(PROTOCOL["symbolic_sysid_min"]))
            == (transformed >= float(PROTOCOL["symbolic_sysid_min"]))
        )

    checks = {
        "action_marginal_exact": bool(np.array_equal(dataset["probe_actions"][:, 0], dataset["probe_actions"][:, 1])),
        "result_multiset_exact": bool(result_multiset),
        "probe_utility_multiset_exact": bool(utility_multiset),
        "current_state_exact": bool(np.array_equal(dataset["decision_public_state"][:, 0], dataset["decision_public_state"][:, 1])),
        "candidate_branch_exact": bool(np.array_equal(dataset["candidate_actions_by_branch"][:, 0], dataset["candidate_actions_by_branch"][:, 1])),
        "candidate_slots_exact": bool(np.all(table == expected)),
        "candidate_top1_crossing": top1_crossing_rate(dataset) >= float(PROTOCOL["candidate_top1_crossing_min"]),
        "candidate_sampler_latent_blind": bool(sampler_check["passed"]),
        "marginals_at_most_0_60": max(marginal_scores.values()) <= float(PROTOCOL["symbolic_marginal_max"]),
        "arrow_at_most_0_70": interaction_scores["arrow_action_compatibility"] <= float(PROTOCOL["symbolic_arrow_max"]),
        "nearest_below_0_90": interaction_scores["nearest_historical_action"] < float(PROTOCOL["interaction_shortcut_max"]),
        "knn_below_0_90": interaction_scores["knn_action_result_lookup"] < float(PROTOCOL["interaction_shortcut_max"]),
        "fourier_at_least_0_90": fit_score >= float(PROTOCOL["symbolic_sysid_min"]),
        "binding_drop_at_least_0_20": fit_score - binding_score >= float(PROTOCOL["binding_drop_min"]),
        "pair_shuffle_at_most_0_02": abs(fit_score - pair_score) <= float(PROTOCOL["pair_shuffle_max_change"]),
        "slot_permutation_conclusion_invariant": same_conclusion(slot_score),
        "coordinate_rotation_conclusion_invariant": same_conclusion(rotation_score),
        "oracle_significantly_above_marginal": oracle_bootstrap["ci95"][0] > 0.0,
        "fourier_significantly_above_marginal": fit_bootstrap["ci95"][0] > 0.0,
    }
    details = {
        "slot_balance_table": table.tolist(),
        "slot_expected_count": expected,
        "slot_min": int(table.min()),
        "slot_max": int(table.max()),
        "marginal_scores": marginal_scores,
        "interaction_scores": interaction_scores,
        "fourier_score": fit_score,
        "oracle_score": float(original["oracle_latent_rollout"][key]),
        "binding_break_score": binding_score,
        "binding_drop": fit_score - binding_score,
        "pair_preserving_score": pair_score,
        "pair_preserving_change": abs(fit_score - pair_score),
        "slot_permutation_score": slot_score,
        "coordinate_rotation_score": rotation_score,
        "candidate_top1_crossing_rate": top1_crossing_rate(dataset),
        "best_marginal": best_marginal,
        "fit_vs_marginal_bootstrap": fit_bootstrap,
        "oracle_vs_marginal_bootstrap": oracle_bootstrap,
        "sampler_check": sampler_check,
    }
    return checks, details
