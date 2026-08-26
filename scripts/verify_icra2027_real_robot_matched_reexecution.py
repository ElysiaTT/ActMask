#!/usr/bin/env python3
"""Verify the frozen matched real-robot re-execution admission gates."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREREG = ROOT / "docs/icra2027_real_robot_matched_reexecution_prereg.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def distance(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        return math.inf
    return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(left, right)))


def max_point_error(left: list[list[float]], right: list[list[float]]) -> float:
    if len(left) != len(right) or not left:
        return math.inf
    return max(distance(a, b) for a, b in zip(left, right))


def max_abs_error(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        return math.inf
    return max(abs(float(a) - float(b)) for a, b in zip(left, right))


def wilson_lower(successes: int, total: int, z: float = 1.959963984540054) -> float:
    if total == 0:
        return 0.0
    p = successes / total
    denominator = 1.0 + z * z / total
    center = p + z * z / (2.0 * total)
    radius = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total)
    return (center - radius) / denominator


def one_sided_sign_p(successes: int, total: int) -> float:
    if total == 0:
        return 1.0
    return sum(math.comb(total, k) for k in range(successes, total + 1)) / (2**total)


def vector_mean(points: list[list[float]]) -> list[float]:
    if not points or any(len(point) != 3 for point in points):
        raise ValueError("expected nonempty xyz points")
    return [sum(float(point[axis]) for point in points) / len(points) for axis in range(3)]


def linear_velocity(times: list[float], points: list[list[float]]) -> list[float]:
    if len(times) != len(points) or len(times) < 2 or any(len(point) != 3 for point in points):
        raise ValueError("invalid time/xyz sequence")
    numeric_times = [float(value) for value in times]
    mean_time = sum(numeric_times) / len(numeric_times)
    denominator = sum((value - mean_time) ** 2 for value in numeric_times)
    if denominator <= 0.0:
        raise ValueError("nonvarying history timestamps")
    means = vector_mean(points)
    return [
        sum(
            (time - mean_time) * (float(point[axis]) - means[axis])
            for time, point in zip(numeric_times, points)
        ) / denominator
        for axis in range(3)
    ]


def endpoint_velocity(
    times: list[float], points: list[list[float]], first: int, last: int
) -> list[float]:
    delta_time = float(times[last]) - float(times[first])
    if delta_time <= 0.0 or len(points[first]) != 3 or len(points[last]) != 3:
        raise ValueError("invalid endpoint velocity inputs")
    return [
        (float(points[last][axis]) - float(points[first][axis])) / delta_time
        for axis in range(3)
    ]


def candidate_distance_score(
    trial: dict[str, Any], anchor: list[float], velocity: list[float], window: list[float]
) -> float:
    times = trial["candidate_time_s"]
    trajectory = trial["candidate_trajectory"]
    if trial.get("candidate_space") != "tcp_xyz_m":
        raise ValueError("candidate_space must be tcp_xyz_m")
    if len(anchor) != 3 or len(velocity) != 3:
        raise ValueError("candidate scoring requires xyz anchor and velocity")
    if (
        len(times) != len(trajectory)
        or not times
        or any(len(point) != 3 for point in trajectory)
        or any(float(times[index]) >= float(times[index + 1]) for index in range(len(times) - 1))
    ):
        raise ValueError("candidate time/trajectory mismatch")
    scores = []
    for time, tcp in zip(times, trajectory):
        numeric_time = float(time)
        if float(window[0]) <= numeric_time <= float(window[1]):
            predicted_target = [
                float(anchor[axis]) + float(velocity[axis]) * numeric_time
                for axis in range(3)
            ]
            scores.append(-distance(tcp, predicted_target))
    if not scores:
        raise ValueError("no candidate waypoint in evaluation window")
    return max(scores)


def fixed_ladder_scores(trial: dict[str, Any], window: list[float]) -> dict[str, float]:
    times = trial["history_time_s"]
    history = trial["history_target_xyz_m"]
    if len(times) != len(history) or len(times) != 6:
        raise ValueError("ladder requires six timestamped history frames")
    current = trial["decision_target_xyz_m"]
    zero_velocity = [0.0, 0.0, 0.0]
    final_two = endpoint_velocity(times, history, 4, 5)
    multi_frame = linear_velocity(times, history)
    early_endpoint = endpoint_velocity(times, history, 0, 3)
    unordered_anchor = vector_mean(history)
    return {
        "action_only": 0.0,
        "current_only": candidate_distance_score(trial, current, zero_velocity, window),
        "unordered_history": candidate_distance_score(
            trial, unordered_anchor, zero_velocity, window
        ),
        "final_two_velocity": candidate_distance_score(trial, current, final_two, window),
        "multi_frame_linear_velocity": candidate_distance_score(
            trial, current, multi_frame, window
        ),
        "early_endpoint_velocity": candidate_distance_score(
            trial, current, early_endpoint, window
        ),
    }


def interpolate_xyz(times: list[float], points: list[list[float]], query: float) -> list[float]:
    numeric_times = [float(value) for value in times]
    if (
        len(numeric_times) != len(points)
        or len(points) < 2
        or any(len(point) != 3 for point in points)
        or any(numeric_times[index] >= numeric_times[index + 1] for index in range(len(times) - 1))
        or query < numeric_times[0]
        or query > numeric_times[-1]
    ):
        raise ValueError("cannot interpolate candidate trajectory")
    if math.isclose(query, numeric_times[-1], rel_tol=0.0, abs_tol=1e-12):
        return [float(value) for value in points[-1]]
    right = next(index for index, value in enumerate(numeric_times) if value > query)
    left = right - 1
    weight = (query - numeric_times[left]) / (numeric_times[right] - numeric_times[left])
    return [
        float(points[left][axis])
        + weight * (float(points[right][axis]) - float(points[left][axis]))
        for axis in range(3)
    ]


def candidate_tracking_error(trial: dict[str, Any], window: list[float]) -> float:
    candidate_times = trial["candidate_time_s"]
    candidate = trial["candidate_trajectory"]
    start_error = distance(candidate[0], trial["decision_tcp_xyz_m"])
    errors = [start_error]
    for time, measured_tcp, valid in zip(
        trial["post_time_s"], trial["post_tcp_xyz_m"], trial["tracking_valid"]
    ):
        numeric_time = float(time)
        if bool(valid) and float(window[0]) <= numeric_time <= float(window[1]):
            commanded_tcp = interpolate_xyz(candidate_times, candidate, numeric_time)
            errors.append(distance(commanded_tcp, measured_tcp))
    if len(errors) == 1:
        raise ValueError("no valid TCP tracking sample in evaluation window")
    return max(errors)


def summarize_pair_order(
    ladder_pairs: dict[str, dict[str, dict[str, float]]], method: str
) -> dict[str, Any]:
    correct = wrong = tied = 0
    for branches in ladder_pairs.values():
        score_a = float(branches["A"][method])
        score_b = float(branches["B"][method])
        if math.isclose(score_a, score_b, rel_tol=0.0, abs_tol=1e-12):
            tied += 1
        elif score_a > score_b:
            correct += 1
        else:
            wrong += 1
    total = correct + wrong + tied
    accuracy = (correct + 0.5 * tied) / total if total else 0.0
    sign_total = correct + wrong
    return {
        "pair_order": accuracy,
        "correct": correct,
        "wrong": wrong,
        "tied": tied,
        "one_sided_sign_p": one_sided_sign_p(correct, sign_total),
        "sign_test_non_ties": sign_total,
    }


def recompute_success(trial: dict[str, Any], threshold: float, window: list[float]) -> bool:
    times = trial["post_time_s"]
    target = trial["post_target_xyz_m"]
    tcp = trial["post_tcp_xyz_m"]
    valid = trial["tracking_valid"]
    if (
        not (len(times) == len(target) == len(tcp) == len(valid))
        or not times
        or any(len(point) != 3 for point in target)
        or any(len(point) != 3 for point in tcp)
        or any(float(times[index]) >= float(times[index + 1]) for index in range(len(times) - 1))
    ):
        raise ValueError(f"mismatched post-tracking lengths in {trial.get('pair_id')}")
    distances = [
        distance(target_point, tcp_point)
        for time, target_point, tcp_point, is_valid in zip(times, target, tcp, valid)
        if bool(is_valid) and float(window[0]) <= float(time) <= float(window[1])
    ]
    if not distances:
        raise ValueError(f"no valid tracking sample in evaluation window: {trial.get('pair_id')}")
    return min(distances) <= threshold


def verify(data: dict[str, Any], prereg: dict[str, Any], prereg_hash: str) -> dict[str, Any]:
    gates: list[dict[str, Any]] = []

    def gate(name: str, passed: bool, detail: Any) -> None:
        gates.append({"name": name, "status": "PASS" if passed else "FAIL", "detail": detail})

    gate("data_schema", data.get("schema") == "actmask-real-robot-matched-data-v1", data.get("schema"))
    gate("preregistration_hash", data.get("prereg_sha256") == prereg_hash, {
        "recorded": data.get("prereg_sha256"), "expected": prereg_hash
    })
    metadata = data.get("metadata", {})
    required_metadata = [
        "hardware_id", "fixture_id", "tracking_system", "calibration_sha256",
        "candidate_controller_sha256", "fixture_controller_sha256",
        "collection_started_utc",
    ]
    missing_metadata = [name for name in required_metadata if not metadata.get(name)]
    malformed_hashes = [
        name
        for name in (
            "calibration_sha256", "candidate_controller_sha256",
            "fixture_controller_sha256",
        )
        if metadata.get(name)
        and not re.fullmatch(r"[0-9a-f]{64}", str(metadata[name]))
    ]
    gate(
        "collection_metadata",
        not missing_metadata
        and not malformed_hashes
        and metadata.get("operator_safety_confirmation") is True,
        {
            "missing": missing_metadata,
            "malformed_hashes": malformed_hashes,
            "operator_safety_confirmation": metadata.get("operator_safety_confirmation"),
        },
    )

    trials = data.get("trials", [])
    expected_families = set(prereg["design"]["families"])
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trial in trials:
        grouped[str(trial.get("pair_id"))].append(trial)
    complete = {
        pair_id: {str(row.get("branch")): row for row in rows}
        for pair_id, rows in grouped.items()
        if len(rows) == 2
        and {str(row.get("branch")) for row in rows} == {"A", "B"}
        and len({str(row.get("family")) for row in rows}) == 1
        and str(rows[0].get("family")) in expected_families
    }
    family_by_pair = {
        pair_id: str(branches["A"].get("family"))
        for pair_id, branches in complete.items()
    }
    family_pair_counts = Counter(family_by_pair.values())
    min_pairs = int(prereg["design"]["min_complete_pairs"])
    gate("complete_pairs", len(complete) >= min_pairs and len(complete) == len(grouped), {
        "complete": len(complete), "observed_pair_ids": len(grouped), "minimum": min_pairs
    })
    min_family_pairs = int(prereg["design"]["min_complete_pairs_per_family"])
    gate(
        "family_coverage",
        set(family_pair_counts) == expected_families
        and all(family_pair_counts[family] >= min_family_pairs for family in expected_families),
        {
            "counts": dict(sorted(family_pair_counts.items())),
            "expected_families": sorted(expected_families),
            "minimum_per_family": min_family_pairs,
        },
    )

    tolerances = prereg["matching_tolerances"]
    pair_metrics = []
    candidate_hashes_by_family: dict[str, list[str]] = defaultdict(list)
    outcome_errors: list[str] = []
    source_errors: list[str] = []
    safety_aborts: list[str] = []
    tracking_errors: list[str] = []
    candidate_tracking_errors: dict[str, float] = {}
    candidate_tracking_failures: list[str] = []
    first_branches_by_family: dict[str, list[str]] = defaultdict(list)
    history_timing_errors: list[str] = []
    ladder_errors: list[str] = []
    ladder_pairs: dict[str, dict[str, dict[str, float]]] = {}
    correct = wrong = tied = 0
    outcome_counts_by_family: dict[str, Counter[str]] = {
        family: Counter() for family in expected_families
    }
    threshold = float(prereg["design"]["success_distance_threshold_m"])
    window = prereg["design"]["evaluation_window_s"]

    for pair_id, branches in sorted(complete.items()):
        a, b = branches["A"], branches["B"]
        family = family_by_pair[pair_id]
        history_a = a.get("history_target_xyz_m", [])
        history_b = b.get("history_target_xyz_m", [])
        early_swap = (
            max_point_error(history_a[:4], list(reversed(history_b[:4])))
            if len(history_a) == len(history_b) == 6 else math.inf
        )
        early_same_order_separation = max_point_error(history_a[:4], history_b[:4])
        suffix = max_point_error(history_a[-2:], history_b[-2:])
        target_error = distance(a.get("decision_target_xyz_m", []), b.get("decision_target_xyz_m", []))
        target_velocity_error = distance(
            a.get("decision_target_velocity_m_s", []),
            b.get("decision_target_velocity_m_s", []),
        )
        target_speed = max(
            distance(a.get("decision_target_velocity_m_s", []), [0.0, 0.0, 0.0]),
            distance(b.get("decision_target_velocity_m_s", []), [0.0, 0.0, 0.0]),
        )
        tcp_error = distance(a.get("decision_tcp_xyz_m", []), b.get("decision_tcp_xyz_m", []))
        joint_error = max_abs_error(a.get("decision_joint_rad", []), b.get("decision_joint_rad", []))
        joint_velocity_error = max_abs_error(
            a.get("decision_joint_velocity_rad_s", []), b.get("decision_joint_velocity_rad_s", [])
        )
        pair_metrics.append({
            "pair_id": pair_id,
            "family": family,
            "early_swap_position_m": early_swap,
            "early_same_order_separation_m": early_same_order_separation,
            "common_suffix_position_m": suffix,
            "decision_target_position_m": target_error,
            "decision_target_velocity_m_s": target_velocity_error,
            "maximum_decision_target_speed_m_s": target_speed,
            "decision_tcp_position_m": tcp_error,
            "decision_joint_position_rad": joint_error,
            "decision_joint_velocity_rad_s": joint_velocity_error,
        })
        time_a = a.get("history_time_s", [])
        time_b = b.get("history_time_s", [])
        valid_times = (
            len(time_a) == len(time_b) == 6
            and all(float(time_a[index]) < float(time_a[index + 1]) for index in range(5))
            and all(float(time_b[index]) < float(time_b[index + 1]) for index in range(5))
            and max_abs_error(time_a, time_b) <= tolerances["history_timestamp_s"]
            and abs(float(time_a[-1])) <= tolerances["history_timestamp_s"]
            and abs(float(time_b[-1])) <= tolerances["history_timestamp_s"]
        )
        if not valid_times:
            history_timing_errors.append(pair_id)

        pair_candidate_hashes = []
        observed_success = {}
        orders = {}
        pair_ladder: dict[str, dict[str, float]] = {}
        for branch, trial in branches.items():
            candidate = {
                "candidate_time_s": trial.get("candidate_time_s"),
                "candidate_trajectory": trial.get("candidate_trajectory"),
                "candidate_space": trial.get("candidate_space"),
            }
            candidate_hash = canonical_hash(candidate)
            candidate_hashes_by_family[family].append(candidate_hash)
            pair_candidate_hashes.append(candidate_hash)
            orders[branch] = int(trial.get("execution_order", -1))
            if trial.get("outcome_source") != "measured_tracking":
                source_errors.append(f"{pair_id}:{branch}")
            if trial.get("candidate_space") != prereg["design"]["candidate_space"]:
                ladder_errors.append(f"{pair_id}:{branch}:candidate-space")
            if bool(trial.get("safety_abort")):
                safety_aborts.append(f"{pair_id}:{branch}")
            if not trial.get("tracking_valid") or not all(bool(value) for value in trial["tracking_valid"]):
                tracking_errors.append(f"{pair_id}:{branch}")
            try:
                recomputed = recompute_success(trial, threshold, window)
            except (KeyError, TypeError, ValueError) as error:
                tracking_errors.append(f"{pair_id}:{branch}:{error}")
                recomputed = None
            observed_success[branch] = recomputed
            if recomputed is None or bool(trial.get("success")) != recomputed:
                outcome_errors.append(f"{pair_id}:{branch}")
            try:
                candidate_tracking_errors[f"{pair_id}:{branch}"] = candidate_tracking_error(
                    trial, window
                )
            except (KeyError, IndexError, TypeError, ValueError) as error:
                candidate_tracking_failures.append(f"{pair_id}:{branch}:{error}")
            try:
                pair_ladder[branch] = fixed_ladder_scores(trial, window)
            except (KeyError, TypeError, ValueError) as error:
                ladder_errors.append(f"{pair_id}:{branch}:{error}")
        if set(pair_ladder) == {"A", "B"}:
            ladder_pairs[pair_id] = pair_ladder
        if len(set(pair_candidate_hashes)) != 1:
            outcome_errors.append(f"{pair_id}:candidate-mismatch")
        if set(orders.values()) == {1, 2}:
            first_branches_by_family[family].append(min(orders, key=orders.get))
        else:
            outcome_errors.append(f"{pair_id}:execution-order")
        if observed_success.get("A") is True and observed_success.get("B") is False:
            correct += 1
            outcome_counts_by_family[family]["correct"] += 1
        elif observed_success.get("A") is False and observed_success.get("B") is True:
            wrong += 1
            outcome_counts_by_family[family]["wrong"] += 1
        else:
            tied += 1
            outcome_counts_by_family[family]["tied"] += 1

    maximum_metric_keys = [
        key
        for key in tolerances
        if key not in {"minimum_early_same_order_separation_m", "history_timestamp_s"}
    ]
    maxima = {
        key: max((row[key] for row in pair_metrics), default=math.inf)
        for key in maximum_metric_keys
    }
    maxima_by_family = {
        family: {
            key: max(
                (row[key] for row in pair_metrics if row["family"] == family),
                default=math.inf,
            )
            for key in maximum_metric_keys
        }
        for family in sorted(expected_families)
    }
    minimum_early_separation = min(
        (row["early_same_order_separation_m"] for row in pair_metrics),
        default=0.0,
    )
    gate("early_history_swap", maxima["early_swap_position_m"] <= tolerances["early_swap_position_m"], {
        "maximum": maxima["early_swap_position_m"], "tolerance": tolerances["early_swap_position_m"]
    })
    gate(
        "nontrivial_early_order",
        minimum_early_separation >= tolerances["minimum_early_same_order_separation_m"],
        {
            "minimum_observed_separation": minimum_early_separation,
            "required": tolerances["minimum_early_same_order_separation_m"],
        },
    )
    gate("history_timing", not history_timing_errors, history_timing_errors)
    gate("common_suffix_match", maxima["common_suffix_position_m"] <= tolerances["common_suffix_position_m"], {
        "maximum": maxima["common_suffix_position_m"], "tolerance": tolerances["common_suffix_position_m"]
    })
    for name in (
        "decision_target_position_m", "decision_target_velocity_m_s",
        "maximum_decision_target_speed_m_s", "decision_tcp_position_m",
        "decision_joint_position_rad", "decision_joint_velocity_rad_s"
    ):
        gate(name, maxima[name] <= tolerances[name], {
            "maximum": maxima[name], "tolerance": tolerances[name]
        })

    candidate_sha256_by_family = {
        family: hashes[0]
        for family, hashes in candidate_hashes_by_family.items()
        if hashes and len(set(hashes)) == 1
    }
    candidate_unique_counts = {
        family: len(set(candidate_hashes_by_family.get(family, [])))
        for family in sorted(expected_families)
    }
    gate(
        "identical_candidate_action",
        set(candidate_sha256_by_family) == expected_families,
        {
            "unique_hashes_by_family": candidate_unique_counts,
            "candidate_sha256_by_family": candidate_sha256_by_family,
        },
    )
    gate("measured_outcomes_recompute", not source_errors and not outcome_errors, {
        "inadmissible_sources": source_errors, "mismatches": outcome_errors
    })
    gate("complete_tracking", not tracking_errors, tracking_errors)
    maximum_candidate_tracking_error = max(
        candidate_tracking_errors.values(), default=math.inf
    )
    candidate_tracking_tolerance = float(
        prereg["design"]["maximum_candidate_tracking_error_m"]
    )
    gate(
        "candidate_execution_fidelity",
        not candidate_tracking_failures
        and len(candidate_tracking_errors) == 2 * len(complete)
        and maximum_candidate_tracking_error <= candidate_tracking_tolerance,
        {
            "maximum_error_m": maximum_candidate_tracking_error,
            "tolerance_m": candidate_tracking_tolerance,
            "failures": candidate_tracking_failures,
            "trials": len(candidate_tracking_errors),
            "expected_trials": 2 * len(complete),
        },
    )
    gate("no_safety_abort", not safety_aborts, safety_aborts)

    ladder = prereg["ladder"]
    all_ladder_methods = ladder["shortcut_controls"] + ladder["ordered_fair_estimators"]
    ladder_summaries = {
        method: summarize_pair_order(ladder_pairs, method)
        for method in all_ladder_methods
        if len(ladder_pairs) == len(complete)
    }
    ladder_pairs_by_family = {
        family: {
            pair_id: ladder_pairs[pair_id]
            for pair_id in ladder_pairs
            if family_by_pair[pair_id] == family
        }
        for family in sorted(expected_families)
    }
    ladder_summaries_by_family = {
        family: {
            method: summarize_pair_order(family_pairs, method)
            for method in all_ladder_methods
        }
        for family, family_pairs in ladder_pairs_by_family.items()
        if len(family_pairs) == family_pair_counts[family]
    }
    gate(
        "ladder_recomputed",
        not ladder_errors
        and len(ladder_pairs) == len(complete)
        and set(ladder_summaries) == set(all_ladder_methods),
        {
            "errors": ladder_errors,
            "pairs": len(ladder_pairs),
            "expected_pairs": len(complete),
            "methods": sorted(ladder_summaries),
            "family_pairs": {
                family: len(pairs)
                for family, pairs in ladder_pairs_by_family.items()
            },
        },
    )
    for method in ladder["shortcut_controls"]:
        summary = ladder_summaries.get(method, {})
        passed = (
            summary.get("pair_order", math.inf) <= ladder["maximum_shortcut_pair_order"]
            and summary.get("one_sided_sign_p", 0.0)
            >= ladder["minimum_shortcut_one_sided_sign_p"]
            and all(
                ladder_summaries_by_family.get(family, {})
                .get(method, {})
                .get("pair_order", math.inf)
                <= ladder["maximum_family_shortcut_pair_order"]
                for family in expected_families
            )
        )
        gate(
            f"shortcut_{method}",
            passed,
            {
                **summary,
                "maximum_pair_order": ladder["maximum_shortcut_pair_order"],
                "minimum_sign_p": ladder["minimum_shortcut_one_sided_sign_p"],
                "maximum_family_pair_order": ladder["maximum_family_shortcut_pair_order"],
                "family_summaries": {
                    family: ladder_summaries_by_family.get(family, {}).get(method, {})
                    for family in sorted(expected_families)
                },
            },
        )
    ordered_qualifying = []
    for method in ladder["ordered_fair_estimators"]:
        summary = ladder_summaries.get(method, {})
        if (
            summary.get("pair_order", -math.inf) >= ladder["minimum_ordered_pair_order"]
            and summary.get("one_sided_sign_p", math.inf)
            <= ladder["maximum_ordered_one_sided_sign_p"]
            and all(
                ladder_summaries_by_family.get(family, {})
                .get(method, {})
                .get("pair_order", -math.inf)
                >= ladder["minimum_family_ordered_pair_order"]
                for family in expected_families
            )
        ):
            ordered_qualifying.append(method)
    gate(
        "ordered_history_signal",
        bool(ordered_qualifying),
        {
            "qualifying": ordered_qualifying,
            "minimum_pair_order": ladder["minimum_ordered_pair_order"],
            "maximum_sign_p": ladder["maximum_ordered_one_sided_sign_p"],
            "minimum_family_pair_order": ladder["minimum_family_ordered_pair_order"],
            "summaries": {
                method: ladder_summaries.get(method, {})
                for method in ladder["ordered_fair_estimators"]
            },
            "family_summaries": {
                family: {
                    method: ladder_summaries_by_family.get(family, {}).get(method, {})
                    for method in ladder["ordered_fair_estimators"]
                }
                for family in sorted(expected_families)
            },
        },
    )

    max_imbalance = int(prereg["acceptance"]["maximum_first-branch_imbalance"])
    order_balance_by_family = {}
    for family in sorted(expected_families):
        first_counts = Counter(first_branches_by_family.get(family, []))
        imbalance = abs(first_counts.get("A", 0) - first_counts.get("B", 0))
        order_balance_by_family[family] = {
            "A_first": first_counts.get("A", 0),
            "B_first": first_counts.get("B", 0),
            "imbalance": imbalance,
            "observed": sum(first_counts.values()),
            "expected": family_pair_counts[family],
        }
    gate(
        "randomized_order_balance",
        all(
            row["observed"] == row["expected"] and row["imbalance"] <= max_imbalance
            for row in order_balance_by_family.values()
        ),
        {"families": order_balance_by_family, "maximum_imbalance": max_imbalance},
    )

    n = len(complete)
    fraction = correct / n if n else 0.0
    lower = wilson_lower(correct, n)
    sign_p = one_sided_sign_p(correct, n)
    acceptance = prereg["acceptance"]
    gate("correct_discordance_fraction", fraction >= acceptance["minimum_correct_discordant_fraction"], {
        "correct": correct, "total": n, "fraction": fraction,
        "minimum": acceptance["minimum_correct_discordant_fraction"]
    })
    gate("wrong_discordance", wrong <= acceptance["maximum_wrong_discordant_pairs"], {
        "wrong": wrong, "maximum": acceptance["maximum_wrong_discordant_pairs"]
    })
    family_outcome_summaries = {}
    for family in sorted(expected_families):
        counts = outcome_counts_by_family[family]
        family_total = family_pair_counts[family]
        family_outcome_summaries[family] = {
            "correct": counts["correct"],
            "wrong": counts["wrong"],
            "tied": counts["tied"],
            "total": family_total,
            "correct_fraction": counts["correct"] / family_total if family_total else 0.0,
        }
    gate(
        "family_outcome_consistency",
        all(
            row["correct_fraction"]
            >= acceptance["minimum_family_correct_discordant_fraction"]
            and row["wrong"] <= acceptance["maximum_wrong_discordant_pairs_per_family"]
            for row in family_outcome_summaries.values()
        ),
        {
            "families": family_outcome_summaries,
            "minimum_correct_fraction": acceptance[
                "minimum_family_correct_discordant_fraction"
            ],
            "maximum_wrong": acceptance["maximum_wrong_discordant_pairs_per_family"],
        },
    )
    gate("wilson_lower_95", lower > acceptance["minimum_wilson_lower_95"], {
        "lower": lower, "minimum_exclusive": acceptance["minimum_wilson_lower_95"]
    })
    gate("one_sided_sign_test", sign_p <= acceptance["maximum_one_sided_sign_p"], {
        "p": sign_p, "maximum": acceptance["maximum_one_sided_sign_p"]
    })

    passed = all(row["status"] == "PASS" for row in gates)
    learned_method_decision = (
        "reject_analytic_saturation"
        if passed and ordered_qualifying
        else "not_admitted_or_unresolved"
    )
    return {
        "schema": "actmask-real-robot-matched-verification-v1",
        "passed": passed,
        "claim_admitted": passed,
        "physical_counterfactual_decision": "admit" if passed else "reject",
        "learned_method_decision": learned_method_decision,
        "provenance": {
            "prereg_sha256": prereg_hash,
            "candidate_sha256_by_family": candidate_sha256_by_family,
            "hardware_id": metadata.get("hardware_id"),
            "fixture_id": metadata.get("fixture_id"),
            "tracking_system": metadata.get("tracking_system"),
            "calibration_sha256": metadata.get("calibration_sha256"),
            "candidate_controller_sha256": metadata.get("candidate_controller_sha256"),
            "fixture_controller_sha256": metadata.get("fixture_controller_sha256"),
            "collection_started_utc": metadata.get("collection_started_utc"),
        },
        "counts": {
            "complete_pairs": n,
            "correct": correct,
            "wrong": wrong,
            "tied": tied,
            "by_family": family_outcome_summaries,
        },
        "statistics": {"correct_fraction": fraction, "wilson_lower_95": lower, "one_sided_sign_p": sign_p},
        "maximum_matching_errors": maxima,
        "maximum_matching_errors_by_family": maxima_by_family,
        "maximum_candidate_tracking_error_m": maximum_candidate_tracking_error,
        "ladder": ladder_summaries,
        "ladder_by_family": ladder_summaries_by_family,
        "gates": gates,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--prereg", type=Path, default=DEFAULT_PREREG)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    prereg = json.loads(args.prereg.read_text(encoding="utf-8"))
    data = json.loads(args.data.read_text(encoding="utf-8"))
    report = verify(data, prereg, sha256(args.prereg))
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
