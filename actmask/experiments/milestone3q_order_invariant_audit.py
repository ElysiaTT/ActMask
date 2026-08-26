"""Phase-0 audit: show which unordered cues solve the 3P matched pilot."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from actmask.data.maniskill_pilot import TASK_IDS
from actmask.experiments.milestone3p_maniskill_baselines import (
    _fit_tree,
    _group_split,
    _metrics,
    _read_task,
    _torch_predictor,
    _tree_predict,
)


SEEDS = (17, 29, 43)
METHODS = (
    "scalar_logistic",
    "shallow_tree",
    "small_mlp",
    "deepsets_history",
    "sorted_frame_history",
    "velocity_magnitude_only",
    "temporal_without_timestamps",
    "unordered_multiset_mlp",
)


def order_invariant_features(history: np.ndarray) -> np.ndarray:
    """Explicitly discard signed direction and frame order from a K-frame history."""
    delta = np.diff(history, axis=1)
    magnitude = np.linalg.norm(delta, axis=2)
    pairwise = np.stack(
        [np.linalg.norm(history[:, right] - history[:, left], axis=1)
        for left in range(history.shape[1]) for right in range(left + 1, history.shape[1])],
        axis=1,
    )
    centroid = history.mean(axis=1)
    centered = history - centroid[:, None]
    covariance_diag = centered.var(axis=1)
    return np.concatenate(
        (
            magnitude.sum(axis=1, keepdims=True),
            magnitude.mean(axis=1, keepdims=True),
            magnitude.max(axis=1, keepdims=True),
            np.square(magnitude).sum(axis=1, keepdims=True),
            np.sort(magnitude, axis=1),
            np.sort(pairwise, axis=1),
            history.var(axis=1).mean(axis=1, keepdims=True),
            centroid.mean(axis=1, keepdims=True),
            covariance_diag.mean(axis=1, keepdims=True),
            np.linalg.norm(history[:, -1] - history[:, 0], axis=1, keepdims=True),
        ),
        axis=1,
    ).astype(np.float32)


def velocity_magnitude_features(history: np.ndarray) -> np.ndarray:
    delta = np.diff(history, axis=1)
    magnitude = np.linalg.norm(delta, axis=2)
    return np.concatenate((np.sort(magnitude, axis=1), magnitude.mean(axis=1, keepdims=True), magnitude.max(axis=1, keepdims=True)), axis=1).astype(np.float32)


def sorted_frame_features(history: np.ndarray) -> np.ndarray:
    """Coordinate-wise sorted frames: a deliberately order-free representation."""
    return np.sort(history, axis=1).reshape(history.shape[0], -1).astype(np.float32)


def multiset_features(history: np.ndarray) -> np.ndarray:
    return np.concatenate((history.mean(axis=1), history.std(axis=1), history.min(axis=1), history.max(axis=1)), axis=1).astype(np.float32)


def _deepsets_predict(train_history: np.ndarray, train_y: np.ndarray, eval_history: np.ndarray, seed: int) -> np.ndarray:
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean = train_history.mean(axis=(0, 1), keepdims=True)
    scale = train_history.std(axis=(0, 1), keepdims=True)
    scale[scale < 1.0e-5] = 1.0
    x_train = torch.from_numpy(((train_history - mean) / scale).astype(np.float32)).to(device)
    x_eval = torch.from_numpy(((eval_history - mean) / scale).astype(np.float32)).to(device)
    y = torch.from_numpy(train_y.astype(np.float32)).to(device)
    encoder = nn.Sequential(nn.Linear(train_history.shape[2], 24), nn.ReLU(), nn.Linear(24, 16), nn.ReLU()).to(device)
    head = nn.Sequential(nn.Linear(32, 24), nn.ReLU(), nn.Linear(24, 1)).to(device)
    optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(head.parameters()), lr=0.02, weight_decay=1.0e-4)
    for _ in range(280):
        encoded = encoder(x_train)
        logits = head(torch.cat((encoded.mean(dim=1), encoded.amax(dim=1)), dim=1)).squeeze(1)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        encoded = encoder(x_eval)
        return head(torch.cat((encoded.mean(dim=1), encoded.amax(dim=1)), dim=1)).squeeze(1).float().cpu().numpy()


def run(output_dir: str | Path) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[2] / "outputs" / "actmask" / "milestone3p_gpu_benchmark" / "maniskill_pilot"
    records: list[dict] = []
    for task in TASK_IDS:
        values, labels, rows = _read_task(root, task)
        history = values["history"].astype(np.float32)
        train = np.asarray([index for index, row in enumerate(rows) if row["regime"] == "original" and _group_split(row) == "train"], dtype=np.int64)
        test = np.asarray([index for index, row in enumerate(rows) if row["regime"] == "matched_counterfactual" and _group_split(row) == "test"], dtype=np.int64)
        scalar = order_invariant_features(history)
        velocity = velocity_magnitude_features(history)
        unordered = multiset_features(history)
        sorted_frames = sorted_frame_features(history)
        for seed in SEEDS:
            scores = {
                "scalar_logistic": _torch_predictor(scalar[train], labels[train], scalar, seed=seed, hidden=None),
                "shallow_tree": _tree_predict(_fit_tree(scalar[train], labels[train]), scalar),
                "small_mlp": _torch_predictor(scalar[train], labels[train], scalar, seed=seed, hidden=24),
                "deepsets_history": _deepsets_predict(history[train], labels[train], history, seed),
                "sorted_frame_history": _torch_predictor(sorted_frames[train], labels[train], sorted_frames, seed=seed, hidden=32),
                "velocity_magnitude_only": _torch_predictor(velocity[train], labels[train], velocity, seed=seed, hidden=None),
                "temporal_without_timestamps": _torch_predictor(history[train].reshape(len(train), -1), labels[train], history.reshape(len(history), -1), seed=seed, hidden=48, steps=320),
                "unordered_multiset_mlp": _torch_predictor(unordered[train], labels[train], unordered, seed=seed, hidden=48, steps=320),
            }
            for method, score in scores.items():
                metrics, _ = _metrics(labels, score, rows, test)
                records.append(dict(task=task, seed=seed, method=method, metrics=metrics))
    summary: dict[str, dict] = {}
    for task in TASK_IDS:
        summary[task] = {}
        for method in METHODS:
            matching = [record["metrics"] for record in records if record["task"] == task and record["method"] == method]
            summary[task][method] = {key: float(np.nanmean([item[key] for item in matching])) for key in matching[0]}
    aggregate = {method: float(np.mean([summary[task][method]["top1_success"] for task in TASK_IDS])) for method in METHODS}
    report = dict(
        phase="3Q-0",
        source="3P original-train to matched-counterfactual-test",
        seeds=list(SEEDS),
        methods=list(METHODS),
        per_task=summary,
        aggregate_top1=aggregate,
        conclusion=(
            "3P matched pilot does not require signed temporal direction: at least one explicitly order-invariant method reaches Top-1 >= 0.90."
            if max(aggregate.values()) >= 0.90 else
            "No evaluated order-invariant method reaches Top-1 >= 0.90 on the 3P matched pilot."
        ),
    )
    (output / "order_invariant_audit_records.json").write_text(json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "order_invariant_audit.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    report = run(root / "outputs" / "actmask" / "milestone3q_signed_dynamics")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
