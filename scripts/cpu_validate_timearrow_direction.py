"""CPU-only sanity check for the TimeArrow audit and redesign direction.

The script has two independent parts:

1. Recompute the zero-training shortcut accuracy from the tracked, compressed
   per-example audit evidence.  This does not require the omitted ``outputs/``
   tree or a simulator.
2. Compare a legacy early-reversal construction with an order-only redesign on
   a small synthetic grouped split.  The redesign matches endpoints, final
   frames, and unordered history statistics, while randomizing the mapping from
   candidate IDs to action directions.

This is a sensitivity check for the audit logic, not evidence of real-robot
performance or of learned-method headroom.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def recompute_tracked_audit(evidence_path: Path) -> dict[str, Any]:
    totals: dict[str, dict[str, int]] = defaultdict(
        lambda: {"rows": 0, "arrow_correct": 0, "parity_correct": 0, "remapped_parity_correct": 0}
    )
    with gzip.open(evidence_path, "rt", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            entry = totals[row["dataset"]]
            entry["rows"] += 1
            entry["arrow_correct"] += int(row["correct"])
            entry["parity_correct"] += int(row["metadata_parity_correct"])
            entry["remapped_parity_correct"] += int(row["remapped_parity_correct"])

    by_dataset: dict[str, dict[str, float | int]] = {}
    all_rows = all_arrow = all_parity = all_remapped = 0
    for name, values in sorted(totals.items()):
        rows = values["rows"]
        by_dataset[name] = {
            "rows": rows,
            "arrow_action_accuracy": values["arrow_correct"] / rows,
            "candidate_parity_accuracy": values["parity_correct"] / rows,
            "remapped_candidate_parity_accuracy": values["remapped_parity_correct"] / rows,
        }
        all_rows += rows
        all_arrow += values["arrow_correct"]
        all_parity += values["parity_correct"]
        all_remapped += values["remapped_parity_correct"]

    return {
        "rows": all_rows,
        "arrow_action_accuracy": all_arrow / all_rows,
        "candidate_parity_accuracy": all_parity / all_rows,
        "remapped_candidate_parity_accuracy": all_remapped / all_rows,
        "by_dataset": by_dataset,
    }


def make_dataset(n_worlds: int, seed: int, construction: str) -> dict[str, np.ndarray]:
    if construction not in {"legacy", "order_only"}:
        raise ValueError(f"Unknown construction: {construction}")

    rng = np.random.default_rng(seed)
    histories: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    labels: list[int] = []
    worlds: list[int] = []
    modes: list[int] = []
    candidate_ids: list[int] = []
    action_signs: list[int] = []

    for world in range(n_worlds):
        offset = rng.normal(0.0, 0.5, size=2)
        amplitude = rng.uniform(0.7, 1.3)
        candidate_sign_by_id = np.array([-1, 1], dtype=int)
        rng.shuffle(candidate_sign_by_id)

        if construction == "legacy":
            points = np.stack(
                [
                    offset + np.array([-amplitude, 0.0]),
                    offset + np.array([-amplitude / 3.0, 0.0]),
                    offset + np.array([amplitude / 3.0, 0.0]),
                    offset + np.array([amplitude, 0.0]),
                ]
            )
        else:
            point_noise = rng.normal(0.0, 0.02, size=(4, 2))
            points = np.stack(
                [
                    offset,
                    offset + np.array([amplitude, 0.0]),
                    offset + np.array([0.0, amplitude * rng.uniform(0.8, 1.2)]),
                    offset,
                ]
            ) + point_noise

        final_two = np.stack([offset, offset]) + rng.normal(0.0, 0.01, size=(2, 2))

        for mode in (-1, 1):
            if construction == "legacy":
                early = points if mode == 1 else points[::-1]
            else:
                early = points if mode == 1 else points[[0, 2, 1, 3]]
            history = np.concatenate([early, final_two], axis=0)

            for candidate_id, action_sign in enumerate(candidate_sign_by_id):
                action = np.array(
                    [float(action_sign), rng.normal(0.0, 0.05)], dtype=np.float64
                )
                histories.append(history)
                actions.append(action)
                labels.append(int(mode == action_sign))
                worlds.append(world)
                modes.append(mode)
                candidate_ids.append(candidate_id)
                action_signs.append(int(action_sign))

    return {
        "history": np.asarray(histories, dtype=np.float64),
        "action": np.asarray(actions, dtype=np.float64),
        "label": np.asarray(labels, dtype=int),
        "world": np.asarray(worlds, dtype=int),
        "mode": np.asarray(modes, dtype=int),
        "candidate_id": np.asarray(candidate_ids, dtype=int),
        "action_sign": np.asarray(action_signs, dtype=int),
    }


def feature_matrix(data: dict[str, np.ndarray], kind: str) -> np.ndarray:
    history = data["history"]
    action = data["action"]
    if kind == "action_only":
        return action
    if kind == "current_action":
        return np.concatenate([history[:, -1], action], axis=1)
    if kind == "final_two_action":
        return np.concatenate([history[:, -2:].reshape(len(history), -1), action], axis=1)
    if kind == "unordered_action":
        early = history[:, :4]
        summary = np.concatenate(
            [early.mean(axis=1), early.std(axis=1), early.min(axis=1), early.max(axis=1)], axis=1
        )
        return np.concatenate([summary, action], axis=1)
    if kind == "ordered_action":
        return np.concatenate([history.reshape(len(history), -1), action], axis=1)
    raise ValueError(f"Unknown feature kind: {kind}")


def grouped_split(data: dict[str, np.ndarray], train_fraction: float = 0.7) -> tuple[np.ndarray, np.ndarray]:
    worlds = np.unique(data["world"])
    cutoff = int(len(worlds) * train_fraction)
    train_worlds = set(worlds[:cutoff].tolist())
    train = np.asarray([world in train_worlds for world in data["world"]], dtype=bool)
    return train, ~train


def pair_order_accuracy(scores: np.ndarray, labels: np.ndarray, keys: list[tuple[int, int]]) -> float:
    grouped: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, key in enumerate(keys):
        grouped[key].append(index)
    values: list[float] = []
    for indices in grouped.values():
        positive = [index for index in indices if labels[index] == 1]
        negative = [index for index in indices if labels[index] == 0]
        if len(positive) != 1 or len(negative) != 1:
            raise AssertionError("Expected exactly one positive and one negative per pair")
        delta = scores[positive[0]] - scores[negative[0]]
        values.append(1.0 if delta > 0 else 0.0 if delta < 0 else 0.5)
    return float(np.mean(values))


def evaluate_construction(data: dict[str, np.ndarray], seed: int) -> dict[str, Any]:
    train, test = grouped_split(data)
    labels = data["label"]
    test_keys = list(zip(data["world"][test].tolist(), data["candidate_id"][test].tolist()))

    learned: dict[str, dict[str, float]] = {}
    for kind in ("action_only", "current_action", "final_two_action", "unordered_action"):
        features = feature_matrix(data, kind)
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=500, random_state=seed),
        )
        model.fit(features[train], labels[train])
        scores = model.predict_proba(features[test])[:, 1]
        learned[kind] = {
            "accuracy": float(accuracy_score(labels[test], scores >= 0.5)),
            "pair_order": pair_order_accuracy(scores, labels[test], test_keys),
        }

    ordered = feature_matrix(data, "ordered_action")
    temporal_model = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(32, 16),
            activation="tanh",
            solver="adam",
            alpha=1e-4,
            batch_size=128,
            learning_rate_init=0.003,
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            random_state=seed,
        ),
    )
    temporal_model.fit(ordered[train], labels[train])
    ordered_scores = temporal_model.predict_proba(ordered[test])[:, 1]
    learned["ordered_mlp"] = {
        "accuracy": float(accuracy_score(labels[test], ordered_scores >= 0.5)),
        "pair_order": pair_order_accuracy(ordered_scores, labels[test], test_keys),
    }

    history = data["history"][test]
    action_sign = data["action_sign"][test]
    endpoint_signal = history[:, 3, 0] - history[:, 0, 0]
    endpoint_scores = endpoint_signal * action_sign
    endpoint_prediction = endpoint_scores > 0

    order_signal = (
        (history[:, 1, 0] - history[:, 1, 1])
        - (history[:, 2, 0] - history[:, 2, 1])
    )
    order_scores = order_signal * action_sign
    order_prediction = order_scores > 0

    parity_mode = (data["candidate_id"][test] * 2 - 1) * data["mode"][test]
    parity_prediction = parity_mode > 0

    return {
        "test_rows": int(test.sum()),
        "test_worlds": int(len(np.unique(data["world"][test]))),
        "models": learned,
        "zero_training_controls": {
            "endpoint_arrow_x_action_accuracy": float(
                accuracy_score(labels[test], endpoint_prediction)
            ),
            "endpoint_arrow_x_action_pair_order": pair_order_accuracy(
                endpoint_scores, labels[test], test_keys
            ),
            "candidate_id_parity_accuracy": float(
                accuracy_score(labels[test], parity_prediction)
            ),
            "posthoc_order_x_action_accuracy": float(
                accuracy_score(labels[test], order_prediction)
            ),
            "posthoc_order_x_action_pair_order": pair_order_accuracy(
                order_scores, labels[test], test_keys
            ),
        },
    }


def run(project_root: Path, n_worlds: int, seed: int) -> dict[str, Any]:
    evidence = project_root / "audit/results/canonical_predictions.csv.gz"
    tracked = recompute_tracked_audit(evidence)
    legacy = evaluate_construction(make_dataset(n_worlds, seed, "legacy"), seed)
    order_only = evaluate_construction(make_dataset(n_worlds, seed + 1, "order_only"), seed)

    checks = {
        "tracked_rows_are_complete": tracked["rows"] == 85_600,
        "tracked_arrow_rule_is_saturated": tracked["arrow_action_accuracy"] >= 0.999,
        "candidate_id_remap_breaks_parity_not_arrow": (
            tracked["remapped_candidate_parity_accuracy"] <= 0.001
        ),
        "legacy_endpoint_shortcut_is_saturated": (
            legacy["zero_training_controls"]["endpoint_arrow_x_action_pair_order"] >= 0.99
        ),
        "order_only_removes_endpoint_shortcut": (
            order_only["zero_training_controls"]["endpoint_arrow_x_action_pair_order"] <= 0.55
        ),
        "order_only_requires_ordered_features": (
            order_only["models"]["ordered_mlp"]["pair_order"] >= 0.90
            and order_only["models"]["unordered_action"]["pair_order"] <= 0.60
        ),
        "order_only_still_has_posthoc_analytic_ceiling": (
            order_only["zero_training_controls"]["posthoc_order_x_action_pair_order"] >= 0.99
        ),
    }
    return {
        "schema": "actmask-timearrow-cpu-direction-pilot-v1",
        "seed": seed,
        "n_worlds_per_construction": n_worlds,
        "tracked_audit_recomputation": tracked,
        "legacy_construction": legacy,
        "order_only_redesign": order_only,
        "checks": checks,
        "all_checks_pass": all(checks.values()),
        "scope": (
            "CPU sensitivity check only; it validates audit behavior and temporal learnability, "
            "not real-robot performance or learned-method headroom."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--n-worlds", type=int, default=1_200)
    parser.add_argument("--seed", type=int, default=7301)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = run(args.project_root.resolve(), args.n_worlds, args.seed)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        output = args.output
        if not output.is_absolute():
            output = args.project_root / output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if not result["all_checks_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
