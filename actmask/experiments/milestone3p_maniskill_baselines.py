"""Observable-only B5 validation for the ManiSkill state-based GPU pilot.

The pilot deliberately keeps all simulator futures and execution-only fields
out of ``*_model_inputs.npz``.  This module loads that strict contract, trains
small GPU models only on original-distribution training worlds, and evaluates
the original, static-balanced, and matched-counterfactual held-out worlds.
"""

from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import nn

from actmask.data.maniskill_pilot import TASK_IDS, load_model_inputs
from actmask.models.temporal_actmask import TemporalActMask


STATIC_METHODS = (
    "current_position_proximity",
    "endpoint_distance",
    "static_analytic",
    "logistic_regression",
    "shallow_decision_tree",
    "static_mlp",
    "action_only",
    "candidate_template_only",
)
DYNAMIC_METHODS = (
    "time_aligned_history",
    "temporal_mlp",
    "temporal_actmask_adapter",
)
PILOT_SEEDS = (17, 29, 43)
CONFIRMATION_SEEDS = (17, 29, 43, 59, 71)
CORRUPTIONS = (
    "correct_history",
    "history_reversed",
    "history_independently_permuted",
    "history_mismatched_across_scenes",
    "zeroed_motion",
    "shuffled_motion",
)


def _group_split(row: dict) -> str:
    remainder = int(row["world_id"]) % 5
    return "train" if remainder < 3 else ("val" if remainder == 3 else "test")


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _read_task(root: Path, task: str) -> tuple[dict[str, np.ndarray], np.ndarray, list[dict]]:
    values = load_model_inputs(root / f"{task}_model_inputs.npz")
    labels = np.load(root / f"{task}_labels.npz")["success"].astype(np.float32)
    rows = _rows(root / f"{task}_metadata.jsonl")
    if len(rows) != len(labels) or len(rows) != values["history"].shape[0]:
        raise AssertionError(f"{task}: model inputs, labels, and metadata misalign")
    groups: dict[tuple[str, int], set[str]] = defaultdict(set)
    for row in rows:
        groups[(row["regime"], int(row["world_id"]))].add(_group_split(row))
    if any(len(group) != 1 for group in groups.values()):
        raise AssertionError(f"{task}: a candidate group crosses a split")
    return values, labels, rows


def _static_features(values: dict[str, np.ndarray]) -> np.ndarray:
    history = values["history"].astype(np.float32)
    action = values["candidate_actions"].astype(np.float32)
    action_summary = np.concatenate(
        (action[:, 0], action[:, -1], action.mean(axis=1), action.std(axis=1), action.sum(axis=1)),
        axis=1,
    )
    return np.concatenate((history[:, -1], values["tcp_state"], action_summary), axis=1)


def _action_features(values: dict[str, np.ndarray]) -> np.ndarray:
    action = values["candidate_actions"].astype(np.float32)
    return np.concatenate((action[:, 0], action[:, -1], action.mean(axis=1), action.std(axis=1), action.sum(axis=1)), axis=1)


def _temporal_features(values: dict[str, np.ndarray], history: np.ndarray | None = None) -> np.ndarray:
    h = values["history"] if history is None else history
    return np.concatenate((h.reshape(h.shape[0], -1).astype(np.float32), _action_features(values), values["tcp_state"]), axis=1)


def _standardize(train: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, keepdims=True)
    scale = train.std(axis=0, keepdims=True)
    scale[scale < 1.0e-5] = 1.0
    return (train - mean) / scale, (values - mean) / scale, (mean, scale)


def _torch_predictor(
    train_x: np.ndarray,
    train_y: np.ndarray,
    eval_x: np.ndarray,
    *,
    seed: int,
    hidden: int | None,
    steps: int = 260,
) -> np.ndarray:
    """Tiny full-batch CUDA learner; no hidden simulator field is accepted."""
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_train, x_eval, _ = _standardize(train_x, eval_x)
    if hidden is None:
        model: nn.Module = nn.Linear(x_train.shape[1], 1)
    else:
        model = nn.Sequential(nn.Linear(x_train.shape[1], hidden), nn.ReLU(), nn.Linear(hidden, 1))
    model.to(device)
    x = torch.from_numpy(x_train).to(device)
    y = torch.from_numpy(train_y.astype(np.float32)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.025, weight_decay=1.0e-4)
    for _ in range(steps):
        logits = model(x).squeeze(-1)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        return model(torch.from_numpy(x_eval).to(device)).squeeze(-1).float().cpu().numpy()


def _fit_tree(train_x: np.ndarray, train_y: np.ndarray, depth: int = 2) -> dict:
    """A deterministic shallow CART-style tree without a third-party package."""
    def build(indices: np.ndarray, remaining: int) -> dict:
        mean = float(train_y[indices].mean())
        if remaining == 0 or len(indices) < 8 or np.all(train_y[indices] == train_y[indices][0]):
            return {"value": mean}
        best: tuple[float, int, float, np.ndarray, np.ndarray] | None = None
        for feature in range(train_x.shape[1]):
            column = train_x[indices, feature]
            for threshold in np.unique(np.quantile(column, (0.2, 0.4, 0.6, 0.8))):
                left = indices[column <= threshold]
                right = indices[column > threshold]
                if len(left) < 3 or len(right) < 3:
                    continue
                impurity = len(left) * train_y[left].var() + len(right) * train_y[right].var()
                candidate = (float(impurity), feature, float(threshold), left, right)
                if best is None or candidate[0] < best[0]:
                    best = candidate
        if best is None:
            return {"value": mean}
        _, feature, threshold, left, right = best
        return {"feature": feature, "threshold": threshold, "left": build(left, remaining - 1), "right": build(right, remaining - 1)}
    return build(np.arange(len(train_y)), depth)


def _tree_predict(tree: dict, values: np.ndarray) -> np.ndarray:
    def value(node: dict, row: np.ndarray) -> float:
        if "value" in node:
            return float(node["value"])
        branch = "left" if row[node["feature"]] <= node["threshold"] else "right"
        return value(node[branch], row)
    return np.asarray([value(tree, row) for row in values], dtype=np.float32)


def _template_scores(train_x: np.ndarray, train_y: np.ndarray, eval_x: np.ndarray) -> np.ndarray:
    """Action-template-only nearest-template diagnostic (no scene state)."""
    x_train, x_eval, _ = _standardize(train_x, eval_x)
    distance = ((x_eval[:, None] - x_train[None]) ** 2).mean(axis=2)
    nearest = np.argpartition(distance, kth=min(4, len(train_x) - 1), axis=1)[:, : min(5, len(train_x))]
    return train_y[nearest].mean(axis=1).astype(np.float32)


def _rank_auc(labels: np.ndarray, score: np.ndarray) -> float:
    positives = labels.astype(bool)
    n_pos = int(positives.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and score[order[end]] == score[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _average_precision(labels: np.ndarray, score: np.ndarray) -> float:
    positives = int(labels.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-score, kind="mergesort")
    sorted_labels = labels[order]
    precision = np.cumsum(sorted_labels) / np.arange(1, len(labels) + 1)
    return float((precision * sorted_labels).sum() / positives)


def _expected_tie_top(labels: np.ndarray, score: np.ndarray) -> float:
    maximum = score.max()
    return float(labels[np.isclose(score, maximum)].mean())


def _group_values(labels: np.ndarray, score: np.ndarray, rows: list[dict], indices: np.ndarray) -> dict[str, float]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index in indices:
        groups[f"{rows[index]['regime']}:{rows[index]['world_id']}"].append(int(index))
    values = {}
    for key, members in groups.items():
        member = np.asarray(members, dtype=np.int64)
        values[key] = _expected_tie_top(labels[member], score[member])
    return values


def _corrected_ndcg(labels: np.ndarray, score: np.ndarray, rows: list[dict], indices: np.ndarray) -> float:
    groups: dict[str, list[int]] = defaultdict(list)
    for index in indices:
        groups[f"{rows[index]['regime']}:{rows[index]['world_id']}"].append(int(index))
    group_scores = []
    for members in groups.values():
        member = np.asarray(members)
        # Ties have the expected discount across their occupied ranks, avoiding
        # candidate-file ordering as an accidental ranking advantage.
        order = np.argsort(-score[member], kind="mergesort")
        ordered_scores = score[member][order]
        ordered_labels = labels[member][order]
        dcg = 0.0
        start = 0
        while start < len(member):
            end = start + 1
            while end < len(member) and np.isclose(ordered_scores[end], ordered_scores[start]):
                end += 1
            expected_gain = float(ordered_labels[start:end].mean())
            dcg += expected_gain * sum(1.0 / math.log2(rank + 2) for rank in range(start, end))
            start = end
        ideal = sorted(labels[member], reverse=True)
        idcg = sum(float(gain) / math.log2(rank + 2) for rank, gain in enumerate(ideal))
        group_scores.append(dcg / idcg if idcg else 0.0)
    return float(np.mean(group_scores))


def _counterfactual_pair_accuracy(labels: np.ndarray, score: np.ndarray, rows: list[dict], indices: np.ndarray) -> float:
    groups: dict[str, list[int]] = defaultdict(list)
    for index in indices:
        group = rows[index]["pair_group"]
        if group is not None:
            groups[group].append(int(index))
    outcomes = []
    for members in groups.values():
        if len(members) != 2:
            continue
        first, second = members
        if labels[first] == labels[second]:
            continue
        positive, negative = (first, second) if labels[first] else (second, first)
        outcomes.append(1.0 if score[positive] > score[negative] else (0.5 if np.isclose(score[positive], score[negative]) else 0.0))
    return float(np.mean(outcomes)) if outcomes else float("nan")


def _metrics(labels: np.ndarray, score: np.ndarray, rows: list[dict], indices: np.ndarray) -> tuple[dict, dict[str, float]]:
    labels = labels.astype(np.float32)
    group = _group_values(labels, score, rows, indices)
    top = float(np.mean(list(group.values())))
    index_labels = labels[indices]
    index_score = score[indices]
    pairwise = []
    grouped: dict[str, list[int]] = defaultdict(list)
    for index in indices:
        grouped[f"{rows[index]['regime']}:{rows[index]['world_id']}"].append(int(index))
    for members in grouped.values():
        for positive in members:
            for negative in members:
                if labels[positive] > labels[negative]:
                    pairwise.append(1.0 if score[positive] > score[negative] else (0.5 if np.isclose(score[positive], score[negative]) else 0.0))
    return dict(
        top1_success=top,
        pairwise_accuracy=float(np.mean(pairwise)),
        average_precision=_average_precision(index_labels, index_score),
        roc_auc=_rank_auc(index_labels, index_score),
        corrected_utility_gain_ndcg=_corrected_ndcg(labels, score, rows, indices),
        normalized_regret=1.0 - top,
        counterfactual_pair_accuracy=_counterfactual_pair_accuracy(labels, score, rows, indices),
    ), group


def _observable_for_adapter(values: dict[str, np.ndarray], history: np.ndarray | None = None) -> dict[str, torch.Tensor]:
    h = values["history"] if history is None else history
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    h_tensor = torch.from_numpy(h.astype(np.float32)).to(device)
    points = h_tensor[:, :, :9].reshape(h_tensor.shape[0], h_tensor.shape[1], 3, 3)
    timestamps = torch.from_numpy(values["timestamps"].astype(np.float32)).to(device)
    duration = (timestamps[:, -1] - timestamps[:, 0]).clamp_min(1.0e-3)
    velocity = (points[:, -1] - points[:, 0]) / duration[:, None, None]
    tcp = torch.from_numpy(values["tcp_state"].astype(np.float32)).to(device)
    action = torch.from_numpy(values["candidate_actions"].astype(np.float32)).to(device)
    endpoint = tcp + 0.04 * action.sum(dim=1)
    action_command = torch.cat((tcp, endpoint, duration[:, None], torch.full((len(tcp), 1), 0.10, device=device)), dim=1)
    visibility = torch.ones((len(tcp), points.shape[1], points.shape[2]), device=device)
    return dict(
        points_history=points,
        visibility_history=visibility,
        timestamps=timestamps,
        estimated_velocity=velocity,
        velocity_confidence=torch.ones((len(tcp), points.shape[2]), device=device),
        action_command=action_command,
        nominal_action_delay=torch.zeros(len(tcp), device=device),
        observation_delay=torch.zeros(len(tcp), device=device),
    )


def _adapter_predictor(values: dict[str, np.ndarray], labels: np.ndarray, train: np.ndarray, seed: int) -> Callable[[np.ndarray | None], np.ndarray]:
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TemporalActMask(observation_frames=values["history"].shape[1], horizon_steps=12, history_hidden_dim=18, action_hidden_dim=14, fusion_hidden_dim=28, contact_hidden_dim=18, utility_hidden_dim=18).to(device)
    model.train()
    observable = _observable_for_adapter(values)
    train_tensor = torch.as_tensor(train, dtype=torch.long, device=device)
    target = torch.from_numpy(labels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.012, weight_decay=1.0e-4)
    for _ in range(220):
        logits = model({key: value[train_tensor] for key, value in observable.items()})["utility_logits"]
        loss = nn.functional.binary_cross_entropy_with_logits(logits, target[train_tensor])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    model.eval()
    def predict(history: np.ndarray | None = None) -> np.ndarray:
        with torch.no_grad():
            return model(_observable_for_adapter(values, history))["utility_logits"].float().cpu().numpy()
    return predict


def _corrupt_history(history: np.ndarray, variant: str) -> np.ndarray:
    if variant == "correct_history":
        return history
    if variant == "history_reversed":
        return history[:, ::-1].copy()
    if variant == "history_independently_permuted":
        rng = np.random.default_rng(20260723)
        return np.stack([sample[rng.permutation(sample.shape[0])] for sample in history]).astype(np.float32)
    if variant == "history_mismatched_across_scenes":
        # Do not use a one-row roll: files are candidate-major, so a local
        # roll can leave the same label-correlated candidate family intact.
        # This fixed, label-blind permutation supplies another scene's
        # history while retaining every row's own action and current TCP.
        rng = np.random.default_rng(9917)
        return history[rng.permutation(len(history))].copy()
    if variant == "zeroed_motion":
        return np.repeat(history[:, -1:, :], history.shape[1], axis=1)
    if variant == "shuffled_motion":
        rng = np.random.default_rng(811)
        result = history.copy()
        for index in range(len(result)):
            result[index, :-1] = result[index, rng.permutation(result.shape[1] - 1)]
        return result
    raise ValueError(f"Unknown corruption: {variant}")


def _history_score(history: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    """Time-aligned observable motion score; smaller pre-decision motion is safer."""
    dt = np.maximum(timestamps[:, -1] - timestamps[:, 0], 1.0e-4)
    return -np.linalg.norm(history[:, -1] - history[:, 0], axis=1) / dt


def _summary(records: list[dict]) -> dict:
    buckets: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for record in records:
        buckets[(record["task"], record["regime"], record["method"])].append(record["metrics"])
    result: dict[str, dict] = {}
    for (task, regime, method), values in sorted(buckets.items()):
        result.setdefault(task, {}).setdefault(regime, {})[method] = {
            metric: float(np.nanmean([entry[metric] for entry in values]))
            for metric in values[0]
        }
    return result


def _mean_metric(records: list[dict], method: str, regime: str, metric: str) -> float:
    values = [record["metrics"][metric] for record in records if record["method"] == method and record["regime"] == regime]
    return float(np.nanmean(values))


def _bootstrap_delta(records: list[dict], static_method: str, dynamic_method: str) -> dict:
    static = {(r["seed"], r["task"]): r["group_top1"] for r in records if r["method"] == static_method and r["regime"] == "matched_counterfactual"}
    dynamic = {(r["seed"], r["task"]): r["group_top1"] for r in records if r["method"] == dynamic_method and r["regime"] == "matched_counterfactual"}
    per_group: dict[tuple[str, str], list[float]] = defaultdict(list)
    for key in sorted(set(static) & set(dynamic)):
        for world, dynamic_value in dynamic[key].items():
            per_group[(key[1], world)].append(dynamic_value - static[key][world])
    deltas = np.asarray([np.mean(value) for value in per_group.values()], dtype=np.float64)
    rng = np.random.default_rng(3107)
    sampled = np.mean(rng.choice(deltas, size=(12000, len(deltas)), replace=True), axis=1)
    return dict(delta=float(deltas.mean()), ci95=[float(np.quantile(sampled, 0.025)), float(np.quantile(sampled, 0.975))], groups=int(len(deltas)))


def run(output_dir: str | Path) -> dict:
    root = Path(output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("B5 pilot requires the approved CUDA runtime")
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    records: list[dict] = []
    corruption_records: list[dict] = []
    for task in TASK_IDS:
        values, labels, rows = _read_task(root, task)
        original_train = np.asarray([i for i, row in enumerate(rows) if row["regime"] == "original" and _group_split(row) == "train"], dtype=np.int64)
        if len(original_train) == 0 or not np.all(labels[original_train].sum() > 0):
            raise AssertionError(f"{task}: original training labels are degenerate")
        static = _static_features(values)
        action = _action_features(values)
        temporal = _temporal_features(values)
        for seed in PILOT_SEEDS:
            scores = {
                "current_position_proximity": -np.linalg.norm(values["history"][:, -1, :3] - values["tcp_state"], axis=1),
                "endpoint_distance": -np.linalg.norm(values["tcp_state"] + 0.04 * values["candidate_actions"].sum(axis=1) - values["history"][:, -1, :3], axis=1),
            }
            scores["static_analytic"] = 0.5 * scores["current_position_proximity"] + 0.5 * scores["endpoint_distance"]
            scores["logistic_regression"] = _torch_predictor(static[original_train], labels[original_train], static, seed=seed, hidden=None)
            scores["shallow_decision_tree"] = _tree_predict(_fit_tree(static[original_train], labels[original_train]), static)
            scores["static_mlp"] = _torch_predictor(static[original_train], labels[original_train], static, seed=seed, hidden=32)
            scores["action_only"] = _torch_predictor(action[original_train], labels[original_train], action, seed=seed, hidden=None)
            scores["candidate_template_only"] = _template_scores(action[original_train], labels[original_train], action)
            scores["time_aligned_history"] = _history_score(values["history"], values["timestamps"])
            scores["temporal_mlp"] = _torch_predictor(temporal[original_train], labels[original_train], temporal, seed=seed, hidden=48, steps=320)
            adapter = _adapter_predictor(values, labels, original_train, seed)
            scores["temporal_actmask_adapter"] = adapter()
            for regime in ("original", "static_balanced", "matched_counterfactual"):
                evaluation = np.asarray([i for i, row in enumerate(rows) if row["regime"] == regime and _group_split(row) == "test"], dtype=np.int64)
                for method, score in scores.items():
                    metrics, group_top1 = _metrics(labels, score, rows, evaluation)
                    records.append(dict(seed=seed, task=task, regime=regime, method=method, metrics=metrics, group_top1=group_top1))
            matched_test = np.asarray([i for i, row in enumerate(rows) if row["regime"] == "matched_counterfactual" and _group_split(row) == "test"], dtype=np.int64)
            for corruption in CORRUPTIONS:
                history = _corrupt_history(values["history"], corruption)
                dynamic_scores = {
                    "time_aligned_history": _history_score(history, values["timestamps"]),
                    "temporal_mlp": _torch_predictor(temporal[original_train], labels[original_train], _temporal_features(values, history), seed=seed, hidden=48, steps=320),
                    "temporal_actmask_adapter": adapter(history),
                }
                for method, score in dynamic_scores.items():
                    metrics, _ = _metrics(labels, score, rows, matched_test)
                    corruption_records.append(dict(seed=seed, task=task, corruption=corruption, method=method, metrics=metrics))
    summary = _summary(records)
    static_strength = {method: _mean_metric(records, method, "original", "top1_success") for method in STATIC_METHODS}
    dynamic_strength = {method: _mean_metric(records, method, "matched_counterfactual", "top1_success") for method in DYNAMIC_METHODS}
    best_static = max(static_strength, key=static_strength.get)
    best_dynamic = max(dynamic_strength, key=dynamic_strength.get)
    bootstrap = _bootstrap_delta(records, best_static, best_dynamic)
    corruption_summary: dict[str, dict] = {}
    for method in DYNAMIC_METHODS:
        corruption_summary[method] = {}
        for corruption in CORRUPTIONS:
            rows_for_key = [entry["metrics"] for entry in corruption_records if entry["method"] == method and entry["corruption"] == corruption]
            corruption_summary[method][corruption] = {key: float(np.nanmean([entry[key] for entry in rows_for_key])) for key in rows_for_key[0]}
    elapsed = time.perf_counter() - started
    report = dict(
        schema_version=1,
        training_distribution="original training worlds only; grouped 60/20/20 split by base world",
        static_methods=list(STATIC_METHODS),
        dynamic_methods=list(DYNAMIC_METHODS),
        pilot_seeds=list(PILOT_SEEDS),
        best_static_method=best_static,
        best_dynamic_method=best_dynamic,
        best_static_original_top1=static_strength[best_static],
        best_static_matched_top1=_mean_metric(records, best_static, "matched_counterfactual", "top1_success"),
        best_dynamic_matched_top1=dynamic_strength[best_dynamic],
        matched_dynamic_minus_static=bootstrap,
        metrics=summary,
        temporal_corruptions=corruption_summary,
        runtime=dict(device=torch.cuda.get_device_name(0), elapsed_seconds=elapsed, peak_gpu_memory_bytes=int(torch.cuda.max_memory_allocated(device))),
    )
    (root / "baseline_records.json").write_text(json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (root / "temporal_corruption_records.json").write_text(json.dumps(corruption_records, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (root / "baseline_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def run_key_confirmation(output_dir: str | Path, static_method: str, dynamic_method: str) -> dict:
    """Five-seed confirmation of the post-pilot key matched comparison."""
    if static_method not in {"logistic_regression", "static_mlp", "action_only"}:
        raise ValueError(f"No five-seed trainer is defined for {static_method}")
    if dynamic_method != "temporal_mlp":
        raise ValueError(f"No five-seed trainer is defined for {dynamic_method}")
    root = Path(output_dir)
    records: list[dict] = []
    for task in TASK_IDS:
        values, labels, rows = _read_task(root, task)
        train = np.asarray([i for i, row in enumerate(rows) if row["regime"] == "original" and _group_split(row) == "train"], dtype=np.int64)
        test = np.asarray([i for i, row in enumerate(rows) if row["regime"] == "matched_counterfactual" and _group_split(row) == "test"], dtype=np.int64)
        static = _static_features(values)
        action = _action_features(values)
        temporal = _temporal_features(values)
        for seed in CONFIRMATION_SEEDS:
            if static_method == "logistic_regression":
                static_score = _torch_predictor(static[train], labels[train], static, seed=seed, hidden=None)
            elif static_method == "static_mlp":
                static_score = _torch_predictor(static[train], labels[train], static, seed=seed, hidden=32)
            else:
                static_score = _torch_predictor(action[train], labels[train], action, seed=seed, hidden=None)
            dynamic_score = _torch_predictor(temporal[train], labels[train], temporal, seed=seed, hidden=48, steps=320)
            for method, score in ((static_method, static_score), (dynamic_method, dynamic_score)):
                metrics, group_top1 = _metrics(labels, score, rows, test)
                records.append(dict(seed=seed, task=task, regime="matched_counterfactual", method=method, metrics=metrics, group_top1=group_top1))
    comparison = _bootstrap_delta(records, static_method, dynamic_method)
    report = dict(
        confirmation_seeds=list(CONFIRMATION_SEEDS),
        static_method=static_method,
        dynamic_method=dynamic_method,
        static_matched_top1=_mean_metric(records, static_method, "matched_counterfactual", "top1_success"),
        dynamic_matched_top1=_mean_metric(records, dynamic_method, "matched_counterfactual", "top1_success"),
        dynamic_minus_static=comparison,
        per_task={
            task: {
                method: float(np.mean([r["metrics"]["top1_success"] for r in records if r["task"] == task and r["method"] == method]))
                for method in (static_method, dynamic_method)
            }
            for task in TASK_IDS
        },
    )
    (root / "key_confirmation_records.json").write_text(json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (root / "key_confirmation.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    output = root / "outputs" / "actmask" / "milestone3p_gpu_benchmark" / "maniskill_pilot"
    report = run(output)
    report["key_confirmation"] = run_key_confirmation(
        output, report["best_static_method"], report["best_dynamic_method"]
    )
    (output / "baseline_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
