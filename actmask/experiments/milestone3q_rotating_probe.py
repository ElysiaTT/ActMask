"""Observable-only controls and GPU qualification for the 3Q rotating probe."""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from actmask.data import maniskill_3q_tasks as _three_q_tasks  # registers task
from actmask.data import maniskill_state_tasks as _state_tasks  # registers tasks
from actmask.data.maniskill_pilot import MODEL_INPUT_KEYS, load_model_inputs
from actmask.data.milestone3q_signed import TASK_IDS
from actmask.experiments.milestone3q_order_invariant_audit import (
    order_invariant_features,
    sorted_frame_features,
    velocity_magnitude_features,
)


TASK = "fixed_phase_rotating_capture_window"


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _split(row: dict) -> str:
    return "train" if int(row["world_id"]) % 5 < 3 else ("val" if int(row["world_id"]) % 5 == 3 else "test")


def _standardize(train: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, keepdims=True)
    scale = train.std(axis=0, keepdims=True)
    scale[scale < 1.0e-6] = 1.0
    return (train - mean) / scale, (values - mean) / scale


def _mlp_score(train_x: np.ndarray, train_y: np.ndarray, values: np.ndarray, *, hidden: int | None, seed: int) -> np.ndarray:
    torch.manual_seed(seed)
    x_train, x_values = _standardize(train_x, values)
    device = torch.device("cuda:0")
    if hidden is None:
        model: nn.Module = nn.Linear(x_train.shape[1], 1)
    else:
        model = nn.Sequential(nn.Linear(x_train.shape[1], hidden), nn.ReLU(), nn.Linear(hidden, 1))
    model.to(device)
    x = torch.from_numpy(x_train.astype(np.float32)).to(device)
    y = torch.from_numpy(train_y.astype(np.float32)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.025, weight_decay=1.0e-4)
    for _ in range(400):
        loss = nn.functional.binary_cross_entropy_with_logits(model(x).squeeze(-1), y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        return model(torch.from_numpy(x_values.astype(np.float32)).to(device)).squeeze(-1).cpu().numpy()


def _pair_metrics(labels: np.ndarray, scores: np.ndarray, rows: list[dict], indices: np.ndarray) -> dict[str, float]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index in indices:
        groups[rows[int(index)]["signed_group"]].append(int(index))
    outcomes = []
    for members in groups.values():
        if len(members) != 2:
            raise AssertionError("a signed pair was split or malformed")
        first, second = members
        positive, negative = (first, second) if labels[first] else (second, first)
        outcomes.append(1.0 if scores[positive] > scores[negative] else (0.5 if np.isclose(scores[positive], scores[negative]) else 0.0))
    prediction = (scores[indices] > 0.0).astype(np.float32)
    return dict(pair_order_accuracy=float(np.mean(outcomes)), classification_accuracy=float(np.mean(prediction == labels[indices])), pairs=int(len(outcomes)))


def _smoke() -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("3Q GPU probe requires CUDA")
    device = torch.device("cuda:0")
    result = {}
    for name, task_id in TASK_IDS.items():
        env = gym.make(task_id, obs_mode="state", num_envs=8, sim_backend="gpu", render_backend="none")
        try:
            observation, _ = env.reset(seed=list(range(7101, 7109)))
            _, _, _, _, info = env.step(torch.zeros((8, 3), device=device))
            result[name] = dict(
                observation_shape=list(observation.shape),
                finite=bool(torch.isfinite(observation).all()),
                success_field_present="success" in info,
            )
        finally:
            env.close()
    return result


def run(root: str | Path) -> dict:
    root = Path(root)
    if not torch.cuda.is_available():
        raise RuntimeError("3Q anti-shortcut probe requires the approved CUDA runtime")
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    values = load_model_inputs(root / f"{TASK}_model_inputs.npz")
    if set(values) != set(MODEL_INPUT_KEYS):
        raise AssertionError("fair model-input schema changed")
    labels = np.load(root / f"{TASK}_labels.npz")["success"].astype(np.float32)
    rows = _rows(root / f"{TASK}_metadata.jsonl")
    signed = np.asarray([i for i, row in enumerate(rows) if row["regime"] == "signed_order"], dtype=np.int64)
    train = np.asarray([i for i in signed if _split(rows[int(i)]) == "train"], dtype=np.int64)
    test = np.asarray([i for i in signed if _split(rows[int(i)]) == "test"], dtype=np.int64)
    if not len(train) or not len(test) or not (0 < labels[train].sum() < len(train)):
        raise AssertionError("signed grouped probe split is degenerate")
    history = values["history"].astype(np.float32)
    action = values["candidate_actions"].astype(np.float32)
    action_summary = np.concatenate((action[:, 0], action[:, -1], action.mean(axis=1), action.sum(axis=1)), axis=1)
    static = np.concatenate((history[:, -1], values["tcp_state"], action_summary), axis=1)
    action_only = action_summary
    # Reuse the exact frozen magnitude representation used by pair auditing;
    # direct norm reductions can otherwise introduce an irrelevant rounding
    # asymmetry between the two ordered encodings.
    magnitude_only = velocity_magnitude_features(history)
    unordered = order_invariant_features(history)
    sorted_history = sorted_frame_features(history)
    ordered = np.concatenate(((history - history[:, -1:, :]).reshape(len(history), -1), action_summary, values["tcp_state"]), axis=1)
    signed_score = -(history[:, 1, 0] - history[:, 0, 0]) * (history[:, 2, 1] - history[:, 0, 1]) + (history[:, 1, 1] - history[:, 0, 1]) * (history[:, 2, 0] - history[:, 0, 0])

    scores = {
        "static_only": _mlp_score(static[train], labels[train], static, hidden=None, seed=17),
        "action_only": _mlp_score(action_only[train], labels[train], action_only, hidden=None, seed=19),
        "candidate_template_only": _mlp_score(action_only[train], labels[train], action_only, hidden=8, seed=23),
        "angular_speed_magnitude_only": _mlp_score(magnitude_only[train], labels[train], magnitude_only, hidden=None, seed=29),
        "unordered_history_deepsets_equivalent": _mlp_score(unordered[train], labels[train], unordered, hidden=24, seed=31),
        "sorted_history": _mlp_score(sorted_history[train], labels[train], sorted_history, hidden=24, seed=37),
        "signed_finite_difference": signed_score,
        "ordered_temporal_mlp": _mlp_score(ordered[train], labels[train], ordered, hidden=32, seed=41),
    }
    controls = {name: _pair_metrics(labels, score, rows, test) for name, score in scores.items()}
    reversed_history = history[:, ::-1].copy()
    reversed_ordered = np.concatenate(((reversed_history - reversed_history[:, -1:, :]).reshape(len(history), -1), action_summary, values["tcp_state"]), axis=1)
    reversed_signed = -(reversed_history[:, 1, 0] - reversed_history[:, 0, 0]) * (reversed_history[:, 2, 1] - reversed_history[:, 0, 1]) + (reversed_history[:, 1, 1] - reversed_history[:, 0, 1]) * (reversed_history[:, 2, 0] - reversed_history[:, 0, 0])
    reversed_controls = {
        "signed_finite_difference": _pair_metrics(labels, reversed_signed, rows, test),
        "ordered_temporal_mlp": _pair_metrics(labels, _mlp_score(ordered[train], labels[train], reversed_ordered, hidden=32, seed=41), rows, test),
    }
    report = dict(
        schema_version=1,
        task=TASK,
        fair_model_input_keys=sorted(values),
        grouped_split=dict(train_examples=int(len(train)), test_examples=int(len(test)), test_pairs=int(len(test) // 2)),
        controls=controls,
        exact_reversed_history=reversed_controls,
        correct_minus_reversed_pair_accuracy={name: controls[name]["pair_order_accuracy"] - reversed_controls[name]["pair_order_accuracy"] for name in reversed_controls},
        cuda_smoke=_smoke(),
        runtime=dict(device=torch.cuda.get_device_name(0), elapsed_seconds=time.perf_counter() - started, peak_gpu_memory_bytes=int(torch.cuda.max_memory_allocated())),
    )
    (root / "rotating_anti_shortcut_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    project = Path(__file__).resolve().parents[2]
    print(json.dumps(run(project / "outputs" / "actmask" / "milestone3q_signed_dynamics" / "rotating_probe_v2"), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
