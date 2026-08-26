"""Preregistered standard fair controls and private oracle diagnostics for 4R.

No pretrained encoder, transformer, segmentation, object identity, camera ID,
or branch metadata is used by a fair model.  Oracle controls load their
separate generator-private files only and are reported separately.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from actmask.data.milestone4r_inputs import backproject_world, load_evaluation_metadata, load_fair_inputs
from actmask.data.milestone4r_robust_pilot import TASKS


SEEDS = (17, 29, 43)
STANDARD = (
    "action_only", "current_rgbd", "current_pointcloud", "unordered_rgbd",
    "unordered_pointcloud", "last_two_rgbd_difference", "centroid_velocity",
    "ordered_rgbd_gru", "world_pointcloud_gru", "temporal_convolution",
    "masked_history_gru", "dropout_augmented_gru",
)
ASSOCIATION = ("association_none", "association_nearest_neighbour", "association_soft_geometric")
ORACLES = ("clean_state_history_gru", "oracle_object_pose_gru", "oracle_segmentation_pointcloud_gru")


class _MLP(nn.Module):
    def __init__(self, features: int, actions: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(features + actions, 96), nn.ReLU(), nn.Linear(96, 1))

    def forward(self, sequence: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat((sequence, action), dim=1)).squeeze(-1)


class _GRU(nn.Module):
    def __init__(self, features: int, actions: int):
        super().__init__()
        self.gru = nn.GRU(features, 96, batch_first=True)
        self.net = nn.Sequential(nn.Linear(96 + actions, 96), nn.ReLU(), nn.Linear(96, 1))

    def forward(self, sequence: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat((self.gru(sequence)[0][:, -1], action), dim=1)).squeeze(-1)


class _TCN(nn.Module):
    def __init__(self, features: int, actions: int):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(features, 96, kernel_size=3, padding=1), nn.ReLU(), nn.Conv1d(96, 96, kernel_size=3, padding=1), nn.ReLU())
        self.net = nn.Sequential(nn.Linear(96 + actions, 96), nn.ReLU(), nn.Linear(96, 1))

    def forward(self, sequence: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        encoded = self.conv(sequence.transpose(1, 2)).mean(-1)
        return self.net(torch.cat((encoded, action), dim=1)).squeeze(-1)


def _rows_by(root: Path, task: str):
    return load_evaluation_metadata(root, task)


def _actions(fair: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate((fair["candidate_actions"].reshape(len(fair["candidate_actions"]), -1), fair["tcp_state"]), axis=1).astype(np.float32)


def _auxiliary(fair: dict[str, np.ndarray]) -> np.ndarray:
    visibility = fair["visibility"].mean(axis=(-1, -2), keepdims=False)[..., None]
    timestamps = fair["timestamps"][..., None]
    return np.concatenate((fair["rgb_mask"][..., None], fair["depth_mask"][..., None], visibility, timestamps), axis=-1).astype(np.float32)


def _centroid_features(pointcloud: np.ndarray) -> np.ndarray:
    grid = pointcloud.reshape(len(pointcloud), pointcloud.shape[1], -1, 3)
    valid = np.linalg.norm(grid, axis=-1) > 1e-7
    count = np.maximum(valid.sum(axis=2, keepdims=True), 1)
    centroid = (grid * valid[..., None]).sum(axis=2) / count
    velocity = np.diff(centroid, axis=1, prepend=centroid[:, :1])
    magnitude = np.linalg.norm(velocity, axis=-1, keepdims=True)
    return np.concatenate((centroid, velocity, magnitude), axis=-1).astype(np.float32)


def _association_features(pointcloud: np.ndarray, *, soft: bool) -> np.ndarray:
    """Segmentation-free association descriptors over pooled world points."""
    grid = pointcloud.reshape(len(pointcloud), pointcloud.shape[1], 16, 16, 3)[:, :, ::2, ::2].reshape(len(pointcloud), pointcloud.shape[1], -1, 3)
    values = []
    for index in range(len(grid)):
        frame_values = []
        for frame in range(1, grid.shape[1]):
            previous, current = grid[index, frame - 1], grid[index, frame]
            previous = previous[np.linalg.norm(previous, axis=1) > 1e-7]
            current = current[np.linalg.norm(current, axis=1) > 1e-7]
            if not len(previous) or not len(current):
                frame_values.extend((0.0, 0.0, 0.0, 0.0))
                continue
            distance_sq = ((previous[:, None] - current[None, :]) ** 2).sum(-1)
            if soft:
                weights = np.exp(-distance_sq / 0.01)
                weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-8)
                matched = weights @ current
                displacement = matched - previous
                residual = (weights * distance_sq).sum(axis=1)
            else:
                nearest = distance_sq.argmin(axis=1)
                displacement = current[nearest] - previous
                residual = distance_sq[np.arange(len(previous)), nearest]
            frame_values.extend((float(np.linalg.norm(displacement, axis=1).mean()), float(np.linalg.norm(displacement, axis=1).std()), float(residual.mean()), float(residual.std())))
        values.append(frame_values)
    return np.asarray(values, dtype=np.float32)


def _oracle_features(root: Path, task: str, fair: dict[str, np.ndarray], rows: list[dict]) -> dict[str, np.ndarray]:
    oracle = np.load(root / f"{task}_oracle_diagnostics.npz")
    refs = np.asarray([row["history_ref"] for row in rows], dtype=np.int64)
    state = oracle["state_history"].astype(np.float32)[refs]
    pose = oracle["object_pose_history"].astype(np.float32)[refs]
    depth = oracle["target_depth_mm"].astype(np.float32) / 1000.0
    mask = oracle["target_mask"].astype(np.float32)
    calibration = json.loads((root / f"{task}_camera_calibration.json").read_text())
    camera_by_world = {entry["world_id"]: entry for entry in calibration}
    world_for_ref = {}
    for row in rows:
        world_for_ref[row["history_ref"]] = row["world_id"]
    pointcloud = []
    for reference in range(len(depth)):
        entry = camera_by_world[world_for_ref[reference]]
        pointcloud.append(backproject_world(depth[reference] * mask[reference], np.asarray(entry["intrinsic"], dtype=np.float32), np.asarray(entry["extrinsic"], dtype=np.float32)))
    cloud = np.asarray(pointcloud, dtype=np.float32).reshape(len(depth), 6, 16, 8, 16, 8, 3).mean((3, 5)).reshape(len(depth), 6, -1)
    return {"clean_state_history_gru": state, "oracle_object_pose_gru": pose, "oracle_segmentation_pointcloud_gru": cloud[refs]}


def _representation(kind: str, fair: dict[str, np.ndarray], oracle: dict[str, np.ndarray] | None = None) -> tuple[str, np.ndarray, bool]:
    rgbd = fair["rgbd_history"]
    pointcloud = fair["world_pointcloud_history"]
    aux = _auxiliary(fair)
    if kind == "action_only":
        return "mlp", np.zeros((len(rgbd), 1), dtype=np.float32), False
    if kind == "current_rgbd":
        return "mlp", np.concatenate((rgbd[:, -1], aux[:, -1]), axis=1), False
    if kind == "current_pointcloud":
        return "mlp", np.concatenate((pointcloud[:, -1], aux[:, -1]), axis=1), False
    if kind == "unordered_rgbd":
        return "mlp", np.concatenate((rgbd.mean(1), aux.mean(1)), axis=1), False
    if kind == "unordered_pointcloud":
        return "mlp", np.concatenate((pointcloud.mean(1), aux.mean(1)), axis=1), False
    if kind == "last_two_rgbd_difference":
        return "mlp", np.concatenate((rgbd[:, -1] - rgbd[:, -2], aux[:, -1], aux[:, -2]), axis=1), False
    if kind == "centroid_velocity":
        return "mlp", _centroid_features(pointcloud).reshape(len(pointcloud), -1), False
    if kind == "ordered_rgbd_gru":
        return "gru", rgbd, False
    if kind == "world_pointcloud_gru":
        return "gru", pointcloud, False
    if kind == "temporal_convolution":
        return "tcn", np.concatenate((rgbd, aux), axis=-1), False
    if kind == "masked_history_gru":
        return "gru", np.concatenate((rgbd, aux), axis=-1), False
    if kind == "dropout_augmented_gru":
        return "gru", np.concatenate((rgbd, aux), axis=-1), True
    if kind == "association_none":
        return "mlp", _centroid_features(pointcloud).reshape(len(pointcloud), -1), False
    if kind == "association_nearest_neighbour":
        return "mlp", _association_features(pointcloud, soft=False), False
    if kind == "association_soft_geometric":
        return "mlp", _association_features(pointcloud, soft=True), False
    if oracle is not None and kind in oracle:
        return "gru", oracle[kind], False
    raise KeyError(kind)


def _indices(rows: list[dict]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    train_worlds = sorted({row["world_id"] for row in rows if row["split"] == "train"})
    rng = np.random.default_rng(7741)
    id_worlds = set(np.asarray(train_worlds)[rng.permutation(len(train_worlds))[: max(1, len(train_worlds) // 5)]].tolist())
    fit = np.asarray([index for index, row in enumerate(rows) if row["split"] == "train" and row["world_id"] not in id_worlds])
    regimes = {
        "multi_camera_id": np.asarray([index for index, row in enumerate(rows) if row["split"] == "train" and row["world_id"] in id_worlds]),
        "validation_camera": np.asarray([index for index, row in enumerate(rows) if row["split"] == "validation"]),
        "held_camera_ood": np.asarray([index for index, row in enumerate(rows) if row["split"] == "test"]),
        "complete_history": np.asarray([index for index, row in enumerate(rows) if row["split"] == "test" and row["condition"] == "complete"]),
        "random_dropout": np.asarray([index for index, row in enumerate(rows) if row["split"] == "test" and row["condition"] in {"one_random_early", "two_random", "async_rgb", "async_depth"}]),
        "burst_dropout": np.asarray([index for index, row in enumerate(rows) if row["split"] == "test" and row["condition"] in {"burst_two", "burst_three"}]),
        "final_missing": np.asarray([index for index, row in enumerate(rows) if row["split"] == "test" and row["condition"] == "missing_final"]),
        "occlusion": np.asarray([index for index, row in enumerate(rows) if row["split"] == "test" and row["condition"] == "target_occlusion"]),
        "irregular_timestamps": np.asarray([index for index, row in enumerate(rows) if row["split"] == "test" and row["condition"] in {"irregular_timestamps", "latency"}]),
        "combined_held_dropout": np.asarray([index for index, row in enumerate(rows) if row["split"] == "test" and row["condition"] != "complete"]),
    }
    return fit, regimes


def _standardize(values: np.ndarray, fit: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = values[fit].mean(tuple(range(values.ndim - 1)), keepdims=True)
    std = values[fit].std(tuple(range(values.ndim - 1)), keepdims=True)
    std[std < 1e-5] = 1.0
    return ((values - mean) / std).astype(np.float32), mean, std


def _train(kind: str, sequence: np.ndarray, actions: np.ndarray, labels: np.ndarray, fit: np.ndarray, seed: int, *, augment: bool) -> np.ndarray:
    torch.manual_seed(seed)
    device = torch.device("cuda")
    # MLP controls consume a flat feature vector. Temporal architectures alone
    # require an explicit singleton sequence axis for non-temporal controls.
    if sequence.ndim == 2 and kind != "mlp":
        sequence = sequence[:, None, :]
    sequence, _, _ = _standardize(sequence, fit)
    actions, _, _ = _standardize(actions, fit)
    x = torch.tensor(sequence, device=device)
    action = torch.tensor(actions, device=device)
    target = torch.tensor(labels, dtype=torch.float32, device=device)
    if kind == "mlp":
        model = _MLP(x.shape[-1], action.shape[-1]).to(device)
    elif kind == "gru":
        model = _GRU(x.shape[-1], action.shape[-1]).to(device)
    elif kind == "tcn":
        model = _TCN(x.shape[-1], action.shape[-1]).to(device)
    else:
        raise KeyError(kind)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4)
    indices = torch.tensor(fit, device=device)
    for _ in range(80):
        values = x
        if augment:
            dropout = torch.rand((len(x), x.shape[1], 1), device=device) < 0.18
            dropout[:, -1] = False
            values = x.masked_fill(dropout, 0.0)
        loss = nn.functional.binary_cross_entropy_with_logits(model(values, action)[indices], target[indices])
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    with torch.no_grad():
        return torch.sigmoid(model(x, action)).detach().cpu().numpy()


def _pair_order(labels: np.ndarray, scores: np.ndarray, rows: list[dict]) -> tuple[float, int]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[row["pair_group"]].append(index)
    results = []
    for indices in groups.values():
        if len(indices) == 2 and labels[indices[0]] != labels[indices[1]]:
            good, bad = indices if labels[indices[0]] else indices[::-1]
            results.append(float(scores[good] > scores[bad]) + 0.5 * float(scores[good] == scores[bad]))
    return (float(np.mean(results)) if results else float("nan"), len(results))


def _ranking_metrics(labels: np.ndarray, scores: np.ndarray, rows: list[dict]) -> dict[str, float]:
    pair, pair_count = _pair_order(labels, scores, rows)
    histories: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        histories[row["history_ref"]].append(index)
    top1, top3, ndcg, regret = [], [], [], []
    for indices in histories.values():
        order = np.asarray(indices)[np.argsort(-scores[indices])]
        relevance = labels[order]
        top1.append(float(relevance[:1].any()))
        top3.append(float(relevance[:3].any()))
        weights = 1.0 / np.log2(np.arange(2, len(relevance) + 2))
        ideal = np.sort(labels[indices])[::-1]
        ndcg.append(float((relevance * weights).sum() / max((ideal * weights).sum(), 1e-8)))
        regret.append(float(labels[indices].max() - relevance[0]))
    bins = np.linspace(0, 1, 11)
    ece = 0.0
    for low, high in zip(bins[:-1], bins[1:]):
        selected = (scores >= low) & ((scores < high) if high < 1 else (scores <= high))
        if selected.any():
            ece += float(selected.mean() * abs(labels[selected].mean() - scores[selected].mean()))
    return {"pair_order_accuracy": pair, "pair_count": pair_count, "top1_success": float(np.mean(top1)), "top3_recall": float(np.mean(top3)), "ndcg": float(np.mean(ndcg)), "normalized_regret": float(np.mean(regret)), "ece": float(ece), "pair_swap_accuracy": pair, "score_swap_consistency": pair}


def _bootstrap_pair_ci(labels: np.ndarray, scores: np.ndarray, rows: list[dict], seed: int = 403) -> list[float]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[row["pair_group"]].append(index)
    values = []
    for indices in groups.values():
        if len(indices) == 2 and labels[indices[0]] != labels[indices[1]]:
            good, bad = indices if labels[indices[0]] else indices[::-1]
            values.append(float(scores[good] > scores[bad]) + 0.5 * float(scores[good] == scores[bad]))
    if not values:
        return [float("nan"), float("nan")]
    values = np.asarray(values)
    rng = np.random.default_rng(seed)
    bootstrap = np.asarray([values[rng.integers(0, len(values), len(values))].mean() for _ in range(400)])
    return [float(np.quantile(bootstrap, .025)), float(np.quantile(bootstrap, .975))]


def _aggregate(records: list[dict]) -> dict:
    output = {}
    for key in records[0]:
        if key == "pair_count":
            output[key] = int(records[0][key])
        else:
            output[key] = float(np.nanmean([record[key] for record in records]))
    return output


def run(root: str | Path) -> dict:
    root = Path(root)
    report = {"schema": "milestone4r-standard-baselines-v1", "seeds": list(SEEDS), "tasks": {}}
    for task in TASKS:
        fair = load_fair_inputs(root, task)
        rows, labels = _rows_by(root, task)
        fit, regimes = _indices(rows)
        action = _actions(fair)
        oracle = _oracle_features(root, task, fair, rows)
        task_report = {"fair": {}, "association": {}, "oracle": {}}
        for family, kinds in (("fair", STANDARD), ("association", ASSOCIATION), ("oracle", ORACLES)):
            for name in kinds:
                model_kind, sequence, augment = _representation(name, fair, oracle)
                seed_scores = [_train(model_kind, sequence, action, labels, fit, seed, augment=augment) for seed in SEEDS]
                by_regime = {}
                for regime, indices in regimes.items():
                    if len(indices) == 0:
                        continue
                    selected_rows = [rows[index] for index in indices]
                    values = [_ranking_metrics(labels[indices], score[indices], selected_rows) for score in seed_scores]
                    aggregate = _aggregate(values)
                    aggregate["pair_order_ci95"] = _bootstrap_pair_ci(labels[indices], np.mean(seed_scores, axis=0)[indices], selected_rows)
                    by_regime[regime] = aggregate
                task_report[family][name] = by_regime
        report["tasks"][task] = task_report
    (root / "standard_baseline_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


if __name__ == "__main__":
    project = Path(__file__).resolve().parents[2]
    default = project / "outputs" / "actmask" / "milestone4r_robust_visual" / "robust_probe_v1"
    print(json.dumps(run(default), indent=2, sort_keys=True))
