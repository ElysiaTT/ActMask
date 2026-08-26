#!/usr/bin/env python3
"""Correct the state CI by clustering candidate pairs at the base-world unit.

The original frozen evaluator bootstrapped 6,000 candidate-specific pairs.
Those pairs are nested in 300 test worlds, so they are not independent
sampling units.  This downstream-only correction reads saved scores and
observables, recomputes the selected analytic score, aggregates paired
accuracy differences within each base world, and resamples worlds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = (
    ROOT
    / "outputs/actmask/milestone3r_nl_v2_statsfix/world_cluster_bootstrap.json"
)
SEED = 7301
DRAWS = 4000

SOURCES = {
    "model_inputs": (
        "outputs/actmask/milestone3r_nl_v2/full/id/model_inputs.npz",
        "30ccf4025c911d4c0bb9ab43e9dc7a3fd5de599d30a280886c214ac73a7da6f2",
    ),
    "labels": (
        "outputs/actmask/milestone3r_nl_v2/full/id/labels.npz",
        "74997e73d1944013f8a71aa7953a687e27250b2d0235d4e29356f8a267a33d3d",
    ),
    "metadata": (
        "outputs/actmask/milestone3r_nl_v2/full/id/metadata.jsonl",
        "ff02dece7a19876fe3053450fffa6ce1937e92d855b34611ef82403b3d5bdbd3",
    ),
    "saved_scores": (
        "outputs/actmask/milestone3r_nl_v2_rankfix/raw_scores.npz",
        "8a46655dd195b704c221408a5e41f1708ec095df456cad475f22bd0178772947",
    ),
    "ranking_report": (
        "outputs/actmask/milestone3r_nl_v2_rankfix/rankfix_report.json",
        "e8b2e813ee5dfe07463bc544d6c39d6fed1658eb44051116f14db05e1734e982",
    ),
    "parent_evaluation": (
        "outputs/actmask/milestone3r_nl_v2/full_evaluation/full_evaluation.json",
        "abc371e50fda1e4969f37683201ef0aa9a01e6e5d00ac828521adbceaf7a96e4",
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_inventory() -> list[dict[str, Any]]:
    inventory = []
    for name, (relative, expected) in SOURCES.items():
        path = ROOT / relative
        observed = _sha256(path)
        if observed != expected:
            raise RuntimeError(
                f"source hash mismatch for {name}: expected {expected}, observed {observed}"
            )
        inventory.append(
            {
                "name": name,
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": observed,
            }
        )
    return inventory


def _multi_frame_linear(
    history: np.ndarray, timestamps: np.ndarray, visibility: np.ndarray
) -> np.ndarray:
    """Exact frozen MultiFrameLinearVelocity implementation."""
    mask = visibility.squeeze(-1).astype(bool)
    velocity = np.zeros((len(history), history.shape[2]), dtype=np.float32)
    for index in range(len(history)):
        valid = np.flatnonzero(mask[index])
        if len(valid) < 2:
            continue
        time = timestamps[index, valid]
        design = np.stack((time, np.ones(len(valid))), axis=1)
        coefficient = np.linalg.lstsq(
            design, history[index, valid], rcond=None
        )[0]
        velocity[index] = coefficient[0]
    return velocity


def _velocity_score(velocity: np.ndarray, actions: np.ndarray) -> np.ndarray:
    return (velocity[:, -3:] * actions.sum(axis=1)).sum(axis=1)


def _pair_correct(
    labels: np.ndarray, scores: np.ndarray, indices: list[int]
) -> float:
    if len(indices) != 2 or int(labels[indices].sum()) != 1:
        raise RuntimeError("each test pair must contain one positive and one negative")
    positive = indices[int(not bool(labels[indices[0]]))]
    negative = indices[int(bool(labels[indices[0]]))]
    if scores[positive] > scores[negative]:
        return 1.0
    if np.isclose(scores[positive], scores[negative]):
        return 0.5
    return 0.0


def build(output: Path) -> dict[str, Any]:
    inventory = _source_inventory()
    input_root = ROOT / "outputs/actmask/milestone3r_nl_v2/full/id"
    with np.load(input_root / "model_inputs.npz") as values:
        history = values["history"]
        timestamps = values["timestamps"]
        visibility = values["visibility"]
        actions = values["candidate_actions"]
    with np.load(input_root / "labels.npz") as values:
        labels = values["success"].astype(bool)
    rows = [
        json.loads(line)
        for line in (input_root / "metadata.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    with np.load(ROOT / SOURCES["saved_scores"][0]) as values:
        learned_scores = values["gru_mean"].copy()

    analytic_scores = _velocity_score(
        _multi_frame_linear(history, timestamps, visibility), actions
    )
    test = [index for index, row in enumerate(rows) if row["split"] == "test"]
    pair_groups: dict[str, list[int]] = defaultdict(list)
    for index in test:
        pair_groups[rows[index]["pair_id"]].append(index)

    world_differences: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    learned_correct = []
    analytic_correct = []
    for indices in pair_groups.values():
        learned = _pair_correct(labels, learned_scores, indices)
        analytic = _pair_correct(labels, analytic_scores, indices)
        row = rows[indices[0]]
        world = (row["family"], row["mechanism"], int(row["world_id"]))
        world_differences[world].append(learned - analytic)
        learned_correct.append(learned)
        analytic_correct.append(analytic)

    per_world = np.asarray(
        [np.mean(values) for _, values in sorted(world_differences.items())],
        dtype=np.float64,
    )
    if len(pair_groups) != 6000 or len(per_world) != 300:
        raise RuntimeError(
            f"unexpected test units: pairs={len(pair_groups)}, worlds={len(per_world)}"
        )
    if not all(len(values) == 20 for values in world_differences.values()):
        raise RuntimeError("every test world must contain exactly 20 candidate pairs")

    rng = np.random.default_rng(SEED)
    sampled = per_world[
        rng.integers(0, len(per_world), size=(DRAWS, len(per_world)))
    ].mean(axis=1)
    delta = float(per_world.mean())
    interval = [
        float(np.quantile(sampled, 0.025)),
        float(np.quantile(sampled, 0.975)),
    ]
    distribution = Counter(float(value) for value in per_world)

    parent = json.loads(
        (ROOT / SOURCES["parent_evaluation"][0]).read_text(encoding="utf-8")
    )
    if not np.isclose(delta, parent["bootstrap"]["delta"]):
        raise RuntimeError("world-cluster point estimate does not match frozen parent")
    if not np.isclose(np.mean(learned_correct), 1.0) or not np.isclose(
        np.mean(analytic_correct), 0.6
    ):
        raise RuntimeError("recomputed pair metrics do not match frozen parent")

    report = {
        "schema": "actmask-state-world-cluster-bootstrap-v1",
        "pass": True,
        "correction": {
            "superseded_sampling_unit": "candidate-specific matched pair",
            "correct_sampling_unit": "base world",
            "reason": "twenty candidate pairs are nested within each test world",
            "point_estimate_changed": False,
        },
        "estimator": {
            "learned": "mean logit across saved GRU seeds 17/29/43/59/71",
            "analytic": "validation-selected MultiFrameLinearVelocity",
            "pair_tie_value": 0.5,
            "world_aggregation": "mean paired accuracy difference over C20 candidate pairs",
            "bootstrap": "nonparametric resampling of base worlds with replacement",
            "seed": SEED,
            "draws": DRAWS,
        },
        "test_units": {
            "family_mechanism_cells": len(
                {(family, mechanism) for family, mechanism, _ in world_differences}
            ),
            "base_worlds": len(per_world),
            "candidate_pairs": len(pair_groups),
            "candidate_pairs_per_world": 20,
        },
        "pair_order": {
            "gru": float(np.mean(learned_correct)),
            "analytic": float(np.mean(analytic_correct)),
            "delta": delta,
            "world_cluster_ci95": interval,
        },
        "world_delta_distribution": [
            {"value": value, "worlds": count}
            for value, count in sorted(distribution.items())
        ],
        "superseded_pair_cluster_ci95": parent["bootstrap"]["ci95"],
        "sources": inventory,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    arguments = parser.parse_args()
    report = build(arguments.output.resolve())
    print(json.dumps(report["pair_order"], indent=2))


if __name__ == "__main__":
    main()
