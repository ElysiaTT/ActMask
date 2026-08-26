"""Full observable-only baselines for the accepted Milestone-3Q data."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from actmask.data.maniskill_pilot import load_model_inputs
from actmask.data.milestone3q_signed import TASK_IDS
from actmask.experiments.milestone3q_order_invariant_audit import (
    order_invariant_features,
    sorted_frame_features,
)


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _split(row: dict) -> str:
    return "train" if int(row["world_id"]) % 5 < 3 else ("val" if int(row["world_id"]) % 5 == 3 else "test")


def _features(values: dict[str, np.ndarray], history: np.ndarray | None = None) -> dict[str, np.ndarray]:
    h = values["history"].astype(np.float32) if history is None else history.astype(np.float32)
    action = values["candidate_actions"].astype(np.float32)
    action_summary = np.concatenate((action[:, 0], action[:, -1], action.mean(axis=1), action.sum(axis=1)), axis=1)
    return dict(
        static=np.concatenate((h[:, -1], values["tcp_state"], action_summary), axis=1),
        action=action_summary,
        # Canonicalize at the frozen pair-audit tolerance.  This prevents an
        # order-invariant baseline from exploiting sub-tolerance reduction
        # roundoff that has no observable/simulator interpretation.
        unordered=np.round(order_invariant_features(h), decimals=6),
        sorted=np.round(sorted_frame_features(h), decimals=6),
        temporal=np.concatenate(((h - h[:, -1:, :]).reshape(len(h), -1), action_summary, values["tcp_state"]), axis=1),
    )


def _fit_predict(train_x: np.ndarray, train_y: np.ndarray, eval_x: np.ndarray, *, hidden: int, seed: int) -> np.ndarray:
    torch.manual_seed(seed)
    mean = train_x.mean(axis=0, keepdims=True)
    scale = train_x.std(axis=0, keepdims=True)
    scale[scale < 1.0e-6] = 1.0
    device = torch.device("cuda:0")
    model = nn.Sequential(nn.Linear(train_x.shape[1], hidden), nn.ReLU(), nn.Linear(hidden, 1)).to(device)
    x = torch.from_numpy(((train_x - mean) / scale).astype(np.float32)).to(device)
    y = torch.from_numpy(train_y.astype(np.float32)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02, weight_decay=1.0e-4)
    for _ in range(260):
        loss = nn.functional.binary_cross_entropy_with_logits(model(x).squeeze(-1), y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        return model(torch.from_numpy(((eval_x - mean) / scale).astype(np.float32)).to(device)).squeeze(-1).cpu().numpy()


def _signed_analytic(history: np.ndarray) -> np.ndarray:
    """Observable orientation of the closed pre-decision loop; no oracle field."""
    u = history[:, 1, :2] - history[:, 0, :2]
    v = history[:, 2, :2] - history[:, 0, :2]
    return -(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0])


def _pair_metrics(labels: np.ndarray, score: np.ndarray, rows: list[dict], indices: np.ndarray) -> dict[str, float]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index in indices:
        grouped[rows[int(index)]["signed_group"]].append(int(index))
    values = []
    for members in grouped.values():
        if len(members) != 2:
            raise AssertionError("signed counterfactual group crossed a split")
        first, second = members
        positive, negative = (first, second) if labels[first] else (second, first)
        values.append(1.0 if score[positive] > score[negative] else (0.5 if np.isclose(score[positive], score[negative]) else 0.0))
    return dict(pair_order_accuracy=float(np.mean(values)), pairs=int(len(values)))


def _indices(rows: list[dict], split: str, *, mechanism: str | None = None, exclude: str | None = None) -> np.ndarray:
    return np.asarray([
        index for index, row in enumerate(rows)
        if row["regime"] == "signed_order" and _split(row) == split
        and (mechanism is None or row["mechanism"] == mechanism)
        and (exclude is None or row["mechanism"] != exclude)
    ], dtype=np.int64)


def _aggregate(records: list[dict]) -> dict:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for record in records:
        grouped[(record["setting"], record["method"])].append(record["pair_order_accuracy"])
    return {setting: {method: float(np.mean(values)) for (candidate_setting, method), values in grouped.items() if candidate_setting == setting} for setting in sorted({key[0] for key in grouped})}


def run(root: str | Path, seeds: tuple[int, ...], name: str) -> dict:
    root = Path(root)
    if not torch.cuda.is_available():
        raise RuntimeError("3Q full baselines require CUDA")
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    records: list[dict] = []
    for task in TASK_IDS:
        values = load_model_inputs(root / f"{task}_model_inputs.npz")
        labels = np.load(root / f"{task}_labels.npz")["success"].astype(np.float32)
        rows = _rows(root / f"{task}_metadata.jsonl")
        train = _indices(rows, "train")
        test = _indices(rows, "test")
        if not (0 < labels[train].sum() < len(train)):
            raise AssertionError(f"{task}: signed training labels are degenerate")
        feature = _features(values)
        reversed_feature = _features(values, values["history"][:, ::-1].copy())
        for seed in seeds:
            predictors = {
                "static_mlp": _fit_predict(feature["static"][train], labels[train], feature["static"], hidden=24, seed=seed),
                "action_mlp": _fit_predict(feature["action"][train], labels[train], feature["action"], hidden=12, seed=seed + 101),
                "unordered_history_mlp": _fit_predict(feature["unordered"][train], labels[train], feature["unordered"], hidden=28, seed=seed + 202),
                "sorted_history_mlp": _fit_predict(feature["sorted"][train], labels[train], feature["sorted"], hidden=28, seed=seed + 303),
                "signed_finite_difference": _signed_analytic(values["history"]),
                "ordered_temporal_mlp": _fit_predict(feature["temporal"][train], labels[train], feature["temporal"], hidden=36, seed=seed + 404),
            }
            for method, score in predictors.items():
                records.append(dict(task=task, seed=seed, setting="id_correct_history", method=method, **_pair_metrics(labels, score, rows, test)))
            records.append(dict(task=task, seed=seed, setting="id_reversed_history", method="signed_finite_difference", **_pair_metrics(labels, _signed_analytic(values["history"][:, ::-1]), rows, test)))
            records.append(dict(task=task, seed=seed, setting="id_reversed_history", method="ordered_temporal_mlp", **_pair_metrics(labels, _fit_predict(feature["temporal"][train], labels[train], reversed_feature["temporal"], hidden=36, seed=seed + 404), rows, test)))

            # Mechanism OOD: hold out the phase/delay mechanism from the signed
            # training worlds and evaluate it only on held-out test worlds.
            ood_train = _indices(rows, "train", exclude="phase_delay")
            ood_test = _indices(rows, "test", mechanism="phase_delay")
            for method, score in {
                "static_mlp": _fit_predict(feature["static"][ood_train], labels[ood_train], feature["static"], hidden=24, seed=seed + 505),
                "unordered_history_mlp": _fit_predict(feature["unordered"][ood_train], labels[ood_train], feature["unordered"], hidden=28, seed=seed + 606),
                "signed_finite_difference": _signed_analytic(values["history"]),
                "ordered_temporal_mlp": _fit_predict(feature["temporal"][ood_train], labels[ood_train], feature["temporal"], hidden=36, seed=seed + 707),
            }.items():
                records.append(dict(task=task, seed=seed, setting="ood_phase_delay", method=method, **_pair_metrics(labels, score, rows, ood_test)))
    summary = dict(
        schema_version=1,
        seeds=list(seeds),
        training="signed-order grouped train worlds only; no hidden metadata is loaded as a feature",
        records=records,
        mean_pair_order_accuracy=_aggregate(records),
        runtime=dict(device=torch.cuda.get_device_name(0), elapsed_seconds=time.perf_counter() - started, peak_gpu_memory_bytes=int(torch.cuda.max_memory_allocated())),
    )
    (root / f"signed_baseline_{name}.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--five-seed", action="store_true")
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[2] / "outputs" / "actmask" / "milestone3q_signed_dynamics" / "full_four_task"
    seeds = (17, 29, 43, 59, 71) if arguments.five_seed else (17, 29, 43)
    name = "five_seed" if arguments.five_seed else "three_seed"
    print(json.dumps(run(root, seeds, name), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
