"""Preregistered CPU-only TimeArrow-v3 feasibility experiment.

See ``docs/timearrow_v3_cpu_preregister.md``.  The experiment uses only NumPy
and scikit-learn and keeps every same-state candidate execution.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor


SEEDS = (7301, 7302, 7303, 7304, 7305)
CANDIDATE_ACTIONS = np.asarray([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5], dtype=np.float64)
PROBE_ACTIONS = np.asarray([0.0, 1.0, -0.5, 1.5, -1.0, 0.5, -1.5], dtype=np.float64)
DT = 0.25
CANDIDATE_STEPS = 5


@dataclass(frozen=True)
class Dynamics:
    family: str
    damping: float
    gain: float
    friction: float
    nonlinearity: float
    threshold: float
    memory_decay: float


@dataclass
class State:
    x: float
    v: float
    memory: float


def _sample_dynamics(rng: np.random.Generator, family: str, ood: bool) -> Dynamics:
    if ood:
        damping = rng.uniform(0.45, 0.60) if rng.random() < 0.5 else rng.uniform(0.92, 0.97)
        gain = rng.uniform(0.55, 0.75) if rng.random() < 0.5 else rng.uniform(1.35, 1.60)
        friction = rng.uniform(0.09, 0.16)
        threshold = rng.uniform(0.50, 0.75)
        nonlinearity = rng.uniform(1.6, 2.2)
    else:
        damping = rng.uniform(0.65, 0.90)
        gain = rng.uniform(0.80, 1.30)
        friction = rng.uniform(0.02, 0.08)
        threshold = rng.uniform(0.15, 0.45)
        nonlinearity = rng.uniform(0.8, 1.5)
    return Dynamics(
        family=family,
        damping=float(damping),
        gain=float(gain),
        friction=float(friction),
        nonlinearity=float(nonlinearity),
        threshold=float(threshold),
        memory_decay=float(rng.uniform(0.55, 0.85)),
    )


def _force(dynamics: Dynamics, state: State, action: float) -> float:
    if dynamics.family == "linear":
        response = action
    elif dynamics.family == "saturated":
        response = math.tanh(dynamics.nonlinearity * action)
    elif dynamics.family == "deadzone":
        response = math.copysign(max(abs(action) - dynamics.threshold, 0.0), action)
    elif dynamics.family == "hysteretic":
        response = math.tanh(dynamics.nonlinearity * (action + 0.8 * state.memory))
    else:
        raise ValueError(f"Unknown family: {dynamics.family}")
    return dynamics.gain * response


def _step(dynamics: Dynamics, state: State, action: float) -> State:
    memory = dynamics.memory_decay * state.memory + action
    force = _force(dynamics, state, action)
    drag = dynamics.friction * math.tanh(5.0 * state.v)
    velocity = dynamics.damping * state.v + force - drag
    return State(x=state.x + DT * velocity, v=velocity, memory=memory)


def _rollout(dynamics: Dynamics, state: State, actions: np.ndarray) -> tuple[np.ndarray, State]:
    current = State(state.x, state.v, state.memory)
    positions = [current.x]
    for action in actions:
        current = _step(dynamics, current, float(action))
        positions.append(current.x)
    return np.asarray(positions, dtype=np.float64), current


def _candidate_outcome(dynamics: Dynamics, state: State, action: float) -> float:
    actions = np.full(CANDIDATE_STEPS, action, dtype=np.float64)
    _, final_state = _rollout(dynamics, state, actions)
    return final_state.x


def make_split(
    n_worlds: int,
    seed: int,
    split_name: str,
    families: tuple[str, ...],
    ood: bool,
    world_offset: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    histories: list[np.ndarray] = []
    targets: list[float] = []
    actions: list[float] = []
    candidate_ids: list[int] = []
    worlds: list[int] = []
    family_ids: list[int] = []
    utilities: list[float] = []
    final_positions: list[float] = []

    family_index = {name: index for index, name in enumerate(("linear", "saturated", "deadzone", "hysteretic"))}
    for local_world in range(n_worlds):
        family = families[int(rng.integers(0, len(families)))]
        dynamics = _sample_dynamics(rng, family, ood)
        initial = State(x=float(rng.uniform(-0.8, 0.8)), v=float(rng.normal(0.0, 0.08)), memory=0.0)
        clean_history, decision_state = _rollout(dynamics, initial, PROBE_ACTIONS)
        observed_history = clean_history + rng.normal(0.0, 0.012, size=clean_history.shape)
        outcomes = np.asarray(
            [_candidate_outcome(dynamics, decision_state, float(action)) for action in CANDIDATE_ACTIONS]
        )
        anchor = int(rng.integers(0, len(CANDIDATE_ACTIONS)))
        target = float(outcomes[anchor] + rng.normal(0.0, 0.025))
        world_id = world_offset + local_world
        permutation = rng.permutation(len(CANDIDATE_ACTIONS))

        for candidate_id, action_index in enumerate(permutation):
            action = float(CANDIDATE_ACTIONS[action_index])
            final_position = float(outcomes[action_index])
            histories.append(observed_history)
            targets.append(target)
            actions.append(action)
            candidate_ids.append(candidate_id)
            worlds.append(world_id)
            family_ids.append(family_index[family])
            utilities.append(-abs(final_position - target))
            final_positions.append(final_position)

    return {
        "split": np.asarray([split_name] * len(worlds)),
        "history": np.asarray(histories, dtype=np.float64),
        "target": np.asarray(targets, dtype=np.float64),
        "action": np.asarray(actions, dtype=np.float64),
        "candidate_id": np.asarray(candidate_ids, dtype=int),
        "world": np.asarray(worlds, dtype=int),
        "family": np.asarray(family_ids, dtype=int),
        "utility": np.asarray(utilities, dtype=np.float64),
        "final_position": np.asarray(final_positions, dtype=np.float64),
    }


def features(data: dict[str, np.ndarray], kind: str) -> np.ndarray:
    history = data["history"]
    current = history[:, -1]
    goal_delta = data["target"] - current
    action = data["action"]
    common = np.stack([goal_delta, action], axis=1)
    if kind == "action_target":
        return common
    if kind == "current":
        return np.column_stack([current, common])
    if kind == "unordered":
        summary = np.column_stack(
            [
                history.mean(axis=1),
                history.std(axis=1),
                history.min(axis=1),
                history.max(axis=1),
                np.quantile(history, 0.25, axis=1),
                np.quantile(history, 0.75, axis=1),
                current,
            ]
        )
        return np.column_stack([summary, common])
    if kind == "ordered":
        relative = history - current[:, None]
        return np.column_stack([relative, current, common])
    raise ValueError(f"Unknown feature kind: {kind}")


def _metrics(data: dict[str, np.ndarray], scores: np.ndarray) -> dict[str, float]:
    pair_values: list[float] = []
    top1_values: list[float] = []
    regrets: list[float] = []
    for world in np.unique(data["world"]):
        indices = np.flatnonzero(data["world"] == world)
        truth = data["utility"][indices]
        predicted = scores[indices]
        for left in range(len(indices)):
            for right in range(left + 1, len(indices)):
                truth_delta = truth[left] - truth[right]
                score_delta = predicted[left] - predicted[right]
                if abs(truth_delta) <= 1e-12:
                    continue
                pair_values.append(
                    1.0 if truth_delta * score_delta > 0 else 0.0 if score_delta != 0 else 0.5
                )
        selected = int(np.argmax(predicted))
        best = float(np.max(truth))
        worst = float(np.min(truth))
        chosen = float(truth[selected])
        top1_values.append(float(chosen >= best - 1e-12))
        regrets.append((best - chosen) / max(best - worst, 1e-12))
    return {
        "pair_order": float(np.mean(pair_values)),
        "top1": float(np.mean(top1_values)),
        "normalized_regret": float(np.mean(regrets)),
    }


def _analytic_scores(data: dict[str, np.ndarray], kind: str) -> np.ndarray:
    history = data["history"]
    target = data["target"]
    action = data["action"]
    current = history[:, -1]
    last_velocity = (history[:, -1] - history[:, -2]) / DT
    horizon = CANDIDATE_STEPS * DT

    if kind == "unit_gain":
        prediction = current + horizon * action
    elif kind == "last_velocity":
        prediction = current + horizon * last_velocity + 0.5 * horizon**2 * action
    elif kind == "constant_acceleration":
        times = np.arange(history.shape[1], dtype=np.float64) * DT
        prediction = np.empty(len(history), dtype=np.float64)
        future_time = times[-1] + horizon
        for index, row in enumerate(history):
            coefficients = np.polyfit(times, row, deg=2)
            prediction[index] = np.polyval(coefficients, future_time) + 0.5 * horizon**2 * action[index]
    elif kind == "linear_system_id":
        prediction = np.empty(len(history), dtype=np.float64)
        for index, row in enumerate(history):
            velocities = np.diff(row) / DT
            design = np.column_stack([velocities[:-1], PROBE_ACTIONS[1:], np.ones(len(PROBE_ACTIONS) - 1)])
            coefficients, *_ = np.linalg.lstsq(design, velocities[1:], rcond=None)
            damping, gain, bias = coefficients
            damping = float(np.clip(damping, -0.25, 1.25))
            gain = float(np.clip(gain, -2.0, 2.0))
            velocity = float(velocities[-1])
            position = float(row[-1])
            for _ in range(CANDIDATE_STEPS):
                velocity = damping * velocity + gain * action[index] + bias
                position += DT * velocity
            prediction[index] = position
    elif kind == "range_scaled":
        observed_range = np.maximum(history.max(axis=1) - history.min(axis=1), 0.05)
        prediction = current + horizon * observed_range * action
    elif kind == "probe_correlation":
        velocities = np.diff(history) / DT
        numerator = np.sum(velocities * PROBE_ACTIONS[None, :], axis=1)
        denominator = float(np.sum(PROBE_ACTIONS**2))
        gain = np.clip(numerator / denominator, -2.0, 2.0)
        prediction = current + horizon * gain * action
    elif kind == "goal_action_sign":
        return np.sign(target - current) * action
    elif kind == "endpoint_action":
        return (history[:, -1] - history[:, 0]) * action
    elif kind == "velocity_action":
        return last_velocity * action
    else:
        raise ValueError(f"Unknown analytic kind: {kind}")
    return -np.abs(prediction - target)


ANALYTIC_METHODS = (
    "unit_gain",
    "last_velocity",
    "constant_acceleration",
    "linear_system_id",
)
AUDIT_RULES = (
    "goal_action_sign",
    "endpoint_action",
    "velocity_action",
    "range_scaled",
    "probe_correlation",
)
LEARNED_METHODS = ("action_target", "current", "unordered", "ordered")


def _fit_model(train: dict[str, np.ndarray], kind: str, seed: int) -> Any:
    model = ExtraTreesRegressor(
        n_estimators=80,
        max_depth=12,
        min_samples_leaf=3,
        max_features=1.0,
        n_jobs=-1,
        random_state=seed,
    )
    model.fit(features(train, kind), train["utility"])
    return model


def _candidate_integrity(data: dict[str, np.ndarray]) -> dict[str, Any]:
    counts = [int(np.sum(data["world"] == world)) for world in np.unique(data["world"])]
    return {
        "worlds": int(len(counts)),
        "rows": int(len(data["world"])),
        "minimum_candidates": min(counts),
        "maximum_candidates": max(counts),
        "all_seven_candidates": all(count == len(CANDIDATE_ACTIONS) for count in counts),
    }


def run_seed(seed: int, sizes: dict[str, int]) -> dict[str, Any]:
    families = ("linear", "saturated", "deadzone")
    offsets = {"train": 0, "validation": 1_000_000, "id": 2_000_000, "parameter_ood": 3_000_000, "held_mechanism": 4_000_000}
    splits = {
        "train": make_split(sizes["train"], seed + 11, "train", families, False, offsets["train"]),
        "validation": make_split(sizes["validation"], seed + 23, "validation", families, False, offsets["validation"]),
        "id": make_split(sizes["id"], seed + 37, "id", families, False, offsets["id"]),
        "parameter_ood": make_split(sizes["parameter_ood"], seed + 41, "parameter_ood", families, True, offsets["parameter_ood"]),
        "held_mechanism": make_split(sizes["held_mechanism"], seed + 53, "held_mechanism", ("hysteretic",), True, offsets["held_mechanism"]),
    }

    analytic_validation = {
        method: _metrics(splits["validation"], _analytic_scores(splits["validation"], method))
        for method in ANALYTIC_METHODS
    }
    selected_analytic = max(ANALYTIC_METHODS, key=lambda method: analytic_validation[method]["pair_order"])

    models = {method: _fit_model(splits["train"], method, seed) for method in LEARNED_METHODS}
    evaluations: dict[str, Any] = {}
    for split_name in ("validation", "id", "parameter_ood", "held_mechanism"):
        data = splits[split_name]
        evaluations[split_name] = {
            "selected_analytic": _metrics(data, _analytic_scores(data, selected_analytic)),
            "analytic_all": {
                method: _metrics(data, _analytic_scores(data, method)) for method in ANALYTIC_METHODS
            },
            "audit_rules": {
                method: _metrics(data, _analytic_scores(data, method)) for method in AUDIT_RULES
            },
            "learned": {
                method: _metrics(data, models[method].predict(features(data, method)))
                for method in LEARNED_METHODS
            },
        }

    world_sets = {name: set(data["world"].tolist()) for name, data in splits.items()}
    disjoint = all(
        world_sets[left].isdisjoint(world_sets[right])
        for index, left in enumerate(world_sets)
        for right in list(world_sets)[index + 1 :]
    )
    return {
        "seed": seed,
        "selected_analytic": selected_analytic,
        "validation_analytic_selection": analytic_validation,
        "integrity": {name: _candidate_integrity(data) for name, data in splits.items()},
        "world_splits_disjoint": disjoint,
        "evaluations": evaluations,
    }


def _mean(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def aggregate(seed_results: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for split in ("id", "parameter_ood", "held_mechanism"):
        summary[split] = {}
        for label, getter in {
            "selected_analytic": lambda result, split=split: result["evaluations"][split]["selected_analytic"],
            "action_target": lambda result, split=split: result["evaluations"][split]["learned"]["action_target"],
            "current": lambda result, split=split: result["evaluations"][split]["learned"]["current"],
            "unordered": lambda result, split=split: result["evaluations"][split]["learned"]["unordered"],
            "ordered": lambda result, split=split: result["evaluations"][split]["learned"]["ordered"],
            "best_declared_audit_rule": lambda result, split=split: max(
                result["evaluations"][split]["audit_rules"].values(), key=lambda value: value["pair_order"]
            ),
        }.items():
            records = [getter(result) for result in seed_results]
            summary[split][label] = {
                metric: _mean([record[metric] for record in records])
                for metric in ("pair_order", "top1", "normalized_regret")
            }

    ordered_id = summary["id"]["ordered"]
    analytic_id = summary["id"]["selected_analytic"]
    ordered_ood = summary["parameter_ood"]["ordered"]
    analytic_ood = summary["parameter_ood"]["selected_analytic"]
    checks = {
        "all_candidates_retained": all(
            split["all_seven_candidates"]
            for result in seed_results
            for split in result["integrity"].values()
        ),
        "world_splits_disjoint": all(result["world_splits_disjoint"] for result in seed_results),
        "action_only_top1_near_chance": abs(summary["id"]["action_target"]["top1"] - 1 / 7) <= 0.10,
        "id_headroom_at_least_0_05": ordered_id["pair_order"] - analytic_id["pair_order"] >= 0.05,
        "parameter_ood_headroom_at_least_0_03": ordered_ood["pair_order"] - analytic_ood["pair_order"] >= 0.03,
        "ordered_beats_unordered_by_0_05": ordered_id["pair_order"] - summary["id"]["unordered"]["pair_order"] >= 0.05,
        "ordered_has_lower_id_regret": ordered_id["normalized_regret"] < analytic_id["normalized_regret"],
        "declared_rule_search_does_not_saturate": summary["id"]["best_declared_audit_rule"]["pair_order"] < ordered_id["pair_order"],
    }
    return {
        "metrics": summary,
        "checks": checks,
        "go_for_larger_simulator_experiment": all(checks.values()),
        "selected_analytic_by_seed": [result["selected_analytic"] for result in seed_results],
    }


def run(sizes: dict[str, int]) -> dict[str, Any]:
    started = time.perf_counter()
    seed_results = [run_seed(seed, sizes) for seed in SEEDS]
    return {
        "schema": "actmask-timearrow-v3-cpu-preregistered-v1",
        "preregister": "docs/timearrow_v3_cpu_preregister.md",
        "seeds": list(SEEDS),
        "sizes": sizes,
        "candidate_actions": CANDIDATE_ACTIONS.tolist(),
        "probe_actions": PROBE_ACTIONS.tolist(),
        "seed_results": seed_results,
        "aggregate": aggregate(seed_results),
        "runtime_seconds": time.perf_counter() - started,
        "scope": "CPU synthetic feasibility only; not a learned-dynamics, real-robot, or safety claim.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-worlds", type=int, default=1_200)
    parser.add_argument("--validation-worlds", type=int, default=300)
    parser.add_argument("--test-worlds", type=int, default=400)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    sizes = {
        "train": args.train_worlds,
        "validation": args.validation_worlds,
        "id": args.test_worlds,
        "parameter_ood": args.test_worlds,
        "held_mechanism": args.test_worlds,
    }
    result = run(sizes)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        output = args.output
        if not output.is_absolute():
            output = Path(__file__).resolve().parents[1] / output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
