"""Read-only, numeric postmortem of the frozen v2 symbolic failures."""

from __future__ import annotations

import json

import numpy as np

from actmask.experiments.history_action_binding_v2.audit import (
    _features,
    _least_squares,
    _utility_from_displacement,
    baseline_predictions,
    peak_candidates,
)
from actmask.experiments.history_action_binding_v2.common import OUTPUT_ROOT as V2_ROOT
from actmask.experiments.history_action_binding_v2_1.common import OUTPUT_ROOT, sha256_file, write_json


TASK = "task_a_discrete_higher_order"
METRIC_EPSILON = 1.0e-12
DIAGNOSTIC_TIE_EPSILON = 1.0e-8


def _correct(true_difference: np.ndarray, predicted_difference: np.ndarray, epsilon: float) -> np.ndarray:
    informative = np.abs(true_difference) > epsilon
    return np.where(
        informative,
        np.where(
            np.abs(predicted_difference) <= epsilon,
            0.5,
            (np.sign(true_difference) == np.sign(predicted_difference)).astype(np.float64),
        ),
        np.nan,
    )


def _group(values: np.ndarray, group: np.ndarray, count: int) -> list[dict]:
    rows = []
    for index in range(count):
        selected = group == index
        finite = values[selected & np.isfinite(values)]
        rows.append({
            "id": index,
            "informative": int(len(finite)),
            "score": float(np.mean(finite)) if len(finite) else None,
            "wrong_or_predicted_tie": int(np.sum(finite < 1.0)),
        })
    return rows


def build_diagnosis() -> dict:
    source = V2_ROOT / "symbolic_preflight" / TASK / "symbolic_samples.npz"
    with np.load(source, allow_pickle=False) as archive:
        dataset = {key: archive[key] for key in archive.files}
    predictions = baseline_predictions(dataset)
    true = dataset["utility"].astype(np.float64)
    fourier = predictions["quadratic_fourier_sysid"]["utility"]
    oracle = predictions["oracle_latent_rollout"]["utility"]
    true_centered = true - true.mean(axis=2, keepdims=True)
    fourier_centered = fourier - fourier.mean(axis=2, keepdims=True)
    oracle_centered = oracle - oracle.mean(axis=2, keepdims=True)
    true_difference = true_centered[:, 1] - true_centered[:, 0]
    fourier_difference = fourier_centered[:, 1] - fourier_centered[:, 0]
    oracle_difference = oracle_centered[:, 1] - oracle_centered[:, 0]
    metric_correct = _correct(true_difference, fourier_difference, METRIC_EPSILON)
    oracle_correct = _correct(true_difference, oracle_difference, METRIC_EPSILON)

    physical_displacement = _least_squares(
        dataset["probe_actions"].astype(np.float64),
        dataset["probe_delta_physical"].astype(np.float64),
        peak_candidates(dataset),
        "fourier",
    )
    _, physical_utility = _utility_from_displacement(dataset, physical_displacement)
    physical_centered = physical_utility - physical_utility.mean(axis=2, keepdims=True)
    physical_difference = physical_centered[:, 1] - physical_centered[:, 0]
    physical_correct = _correct(true_difference, physical_difference, METRIC_EPSILON)
    diagnostic_correct = _correct(true_difference, fourier_difference, DIAGNOSTIC_TIE_EPSILON)

    twins, _, candidates = true.shape
    template_by_position = dataset["candidate_template_id"]
    slot_group = np.broadcast_to(np.arange(candidates)[None, :], (twins, candidates))
    errors = []
    for twin, slot in np.argwhere(np.isfinite(metric_correct) & (metric_correct < 1.0)):
        template = int(template_by_position[twin, slot])
        peak = peak_candidates(dataset)[twin, slot]
        relative_angle = (
            np.rad2deg(np.arctan2(peak[1], peak[0]) - dataset["public_rotation"][twin]) % 360.0
        )
        errors.append({
            "twin": int(twin),
            "slot": int(slot),
            "template": template,
            "relative_candidate_angle_degrees": float(relative_angle),
            "candidate_radius": float(np.linalg.norm(peak)),
            "true_utility_by_branch": true[twin, :, slot].tolist(),
            "fourier_utility_by_branch": fourier[twin, :, slot].tolist(),
            "oracle_utility_by_branch": oracle[twin, :, slot].tolist(),
            "true_centered_twin_difference": float(true_difference[twin, slot]),
            "fourier_centered_twin_difference": float(fourier_difference[twin, slot]),
            "oracle_centered_twin_difference": float(oracle_difference[twin, slot]),
        })

    conditions = []
    ranks = []
    for twin in range(twins):
        for branch in range(2):
            design = _features(dataset["probe_actions"][twin, branch].astype(np.float64), "fourier")
            conditions.append(float(np.linalg.cond(design)))
            ranks.append(int(np.linalg.matrix_rank(design)))

    seeds = dataset["candidate_seed"].astype(np.int64)
    multipliers = (1, 3, 5, 7, 9, 11, 13, 15)
    schedules = [
        {
            "twin": int(twin),
            "seed": int(seed),
            "shift": int(seed % 16),
            "multiplier": int(multipliers[int((seed // 16) % len(multipliers))]),
        }
        for twin, seed in enumerate(seeds)
    ]
    table = np.zeros((16, 16), dtype=np.int64)
    for twin in range(twins):
        for slot in range(candidates):
            table[slot, int(template_by_position[twin, slot])] += 1

    quantization_error = np.abs(
        dataset["probe_delta"].astype(np.float64)
        - dataset["probe_delta_physical"].astype(np.float64)
    )
    true_abs = np.abs(true_difference)
    error_mask = np.isfinite(metric_correct) & (metric_correct < 1.0)
    affected_templates = sorted(set(int(row["template"]) for row in errors))
    diagnosis = {
        "status": "V2_FAILURE_DIAGNOSED_READ_ONLY",
        "source": str(source),
        "source_sha256": sha256_file(source),
        "frozen_v2_score": float(np.nanmean(np.nanmean(metric_correct, axis=1))),
        "frozen_v2_oracle_score": float(np.nanmean(np.nanmean(oracle_correct, axis=1))),
        "unquantized_probe_same_estimator_score": float(np.nanmean(np.nanmean(physical_correct, axis=1))),
        "diagnostic_score_if_theoretical_ties_at_1e_8_are_excluded": float(np.nanmean(np.nanmean(diagnostic_correct, axis=1))),
        "metric_epsilon_unchanged": METRIC_EPSILON,
        "diagnostic_tie_epsilon_not_used_for_evaluation": DIAGNOSTIC_TIE_EPSILON,
        "informative_candidate_positions_under_frozen_metric": int(np.sum(np.abs(true_difference) > METRIC_EPSILON)),
        "wrong_or_predicted_tie_positions": int(np.sum(error_mask)),
        "wrong_position_true_difference_abs": {
            "min": float(true_abs[error_mask].min()),
            "median": float(np.median(true_abs[error_mask])),
            "max": float(true_abs[error_mask].max()),
        },
        "quantization": {
            "probe_delta_error_max": float(quantization_error.max()),
            "probe_delta_error_mean": float(quantization_error.mean()),
            "primary_cause": False,
            "explanation": "Removing probe quantization does not remove the frozen-metric near-zero tie artifact.",
        },
        "candidate_geometry": {
            "primary_cause": True,
            "affected_templates": affected_templates,
            "affected_relative_angles_degrees": sorted(set(round(row["relative_candidate_angle_degrees"], 5) for row in errors)),
            "explanation": (
                "The 15-degree full-circle grid contains 105/285-degree directions. Both twins clip these away-from-goal responses "
                "to the same -goal utility, making their theoretical centered preference zero; float32 residues around 1e-10 exceed "
                "the frozen 1e-12 informativeness epsilon."
            ),
        },
        "probe_geometry_and_estimator": {
            "design_rank_values": sorted(set(ranks)),
            "condition_number_min": float(min(conditions)),
            "condition_number_max": float(max(conditions)),
            "secondary_cause": True,
            "explanation": (
                "Task A uses one probe radius with both r and r^2 Fourier columns, so the five-column design is rank deficient. "
                "The phase signs remain identifiable away from utility ties, but radius interpolation is poorly conditioned."
            ),
        },
        "per_template": _group(metric_correct, template_by_position, candidates),
        "per_slot": _group(metric_correct, slot_group, candidates),
        "per_twin": [
            {
                "twin": twin,
                "score": float(np.nanmean(metric_correct[twin])),
                "wrong_or_predicted_tie": int(np.sum(np.isfinite(metric_correct[twin]) & (metric_correct[twin] < 1.0))),
            }
            for twin in range(twins)
        ],
        "per_branch_prediction_error": {
            "fourier_displacement_mae": [
                float(np.mean(np.abs(predictions["quadratic_fourier_sysid"]["displacement"][:, branch] - dataset["symbolic_candidate_displacement"][:, branch])))
                for branch in range(2)
            ],
            "oracle_displacement_mae": [
                float(np.mean(np.abs(predictions["oracle_latent_rollout"]["displacement"][:, branch] - dataset["symbolic_candidate_displacement"][:, branch])))
                for branch in range(2)
            ],
        },
        "oracle_fourier_truth_disagreements": errors,
        "slot_failure": {
            "table": table.tolist(),
            "expected_per_cell": 4,
            "actual_min": int(table.min()),
            "actual_max": int(table.max()),
            "unique_rows": int(len({tuple(row) for row in template_by_position})),
            "seed_schedule": schedules,
            "root_cause": (
                "All 16 shifts are globally repeated four times, but the multiplier changes at seed//16 boundaries. "
                "The 64-seed interval starts at residue 15, producing partial multiplier-3 and multiplier-11 blocks around "
                "three complete blocks; multiplier and shift are therefore not crossed as a balanced product."
            ),
            "v2_1_proof_obligation": (
                "Use template=(slot+candidate_seed mod 16) mod 16. For any 64 consecutive seeds, each residue occurs "
                "exactly four times, hence each fixed slot sees every template exactly four times."
            ),
        },
        "conclusion": (
            "The 0.87890625 is not evidence that the phase law is absent. All 124 frozen-metric misses occur at four "
            "templates whose true twin differences are only about 1e-10. v2.1 must avoid these candidate utility ties and "
            "add a second probe radius, without changing the metric epsilon or the 0.90 threshold."
        ),
    }
    return diagnosis


def main() -> None:
    record = build_diagnosis()
    target = OUTPUT_ROOT / "development" / "v2_failure_diagnosis.json"
    write_json(target, record)
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
