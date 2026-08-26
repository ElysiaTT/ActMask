"""Three-seed causal R3 visual controls, standards, and recoverability gate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch
from torch import nn

from actmask.data.r2_real_outcome_adapter import r2_append_log
from actmask.data.real_robot_dataset_adapter import write_json, write_jsonl
from actmask.experiments.r3_botfails_visual_pipeline import DEVICE, FEATURE_DIR, OUT, SEED, TASK_DIR, build_task


SEEDS = (20260725, 20260726, 20260727)
RESULT_DIR = OUT / "baseline_results"


def _auc(label: np.ndarray, score: np.ndarray) -> float:
    pos, neg = score[label == 1], score[label == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def _threshold(label: np.ndarray, score: np.ndarray) -> float:
    candidates = np.unique(np.concatenate(([0.0], score, [1.0])))
    values = []
    for value in candidates:
        pred = score >= value
        tpr = (pred[label == 1]).mean() if (label == 1).any() else 0.0
        tnr = (~pred[label == 0]).mean() if (label == 0).any() else 0.0
        values.append(((tpr + tnr) / 2, float(value)))
    return max(values)[1]


def _metrics(label: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, float]:
    pred = score >= threshold
    tpr = float(pred[label == 1].mean()) if (label == 1).any() else 0.0
    tnr = float((~pred[label == 0]).mean()) if (label == 0).any() else 0.0
    return {"accuracy": float((pred == label).mean()), "balanced_accuracy": (tpr + tnr) / 2, "auroc": _auc(label, score), "threshold": float(threshold), "n": int(len(label))}


def _standardize(train: np.ndarray, value: np.ndarray) -> np.ndarray:
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    return ((value - mean) / np.where(std < 1e-5, 1.0, std)).astype(np.float32)


class FlatHead(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width, 128), nn.GELU(), nn.Dropout(0.15), nn.Linear(128, 1))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value).squeeze(-1)


class StateActionGRU(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.state = nn.GRU(6, 48, batch_first=True)
        self.action = nn.GRU(6, 32, batch_first=True)
        self.head = nn.Sequential(nn.Linear(80, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, history: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        state = self.state(history)[1][-1]
        control = self.action(action)[1][-1]
        return self.head(torch.cat((state, control), dim=1)).squeeze(-1)


class VisualGRU(nn.Module):
    def __init__(self, static_width: int = 0) -> None:
        super().__init__()
        self.visual = nn.GRU(512, 64, batch_first=True)
        self.static = nn.Sequential(nn.Linear(static_width, 48), nn.GELU()) if static_width else None
        self.head = nn.Sequential(nn.Linear(64 + (48 if self.static else 0), 64), nn.GELU(), nn.Dropout(0.15), nn.Linear(64, 1))

    def forward(self, visual: torch.Tensor, static: torch.Tensor | None = None) -> torch.Tensor:
        value = self.visual(visual)[1][-1]
        if self.static is not None and static is not None:
            value = torch.cat((value, self.static(static)), dim=1)
        return self.head(value).squeeze(-1)


class VisualTCN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(512, 96, 3, padding=1), nn.GELU(), nn.Conv1d(96, 64, 3, padding=1), nn.GELU(), nn.AdaptiveAvgPool1d(1))
        self.head = nn.Linear(64, 1)

    def forward(self, visual: torch.Tensor) -> torch.Tensor:
        return self.head(self.net(visual.transpose(1, 2)).squeeze(-1)).squeeze(-1)


def _train(model: nn.Module, tensors: tuple[torch.Tensor, ...], label: np.ndarray, split: np.ndarray, seed: int, *, epochs: int = 120) -> tuple[np.ndarray, np.ndarray, float]:
    train, validation, test = (split == name for name in ("train", "validation", "test"))
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = model.to(DEVICE)
    y = torch.as_tensor(label.astype(np.float32), device=DEVICE)
    # Balance the three safe categories against the imminent category.
    weight = torch.tensor([(label[train] == 0).sum() / max(1, (label[train] == 1).sum())], dtype=torch.float32, device=DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=5e-4)
    best: tuple[float, dict[str, torch.Tensor]] | None = None
    for _ in range(epochs):
        model.train(); optimizer.zero_grad()
        output = model(*[value[train] for value in tensors])
        loss = nn.functional.binary_cross_entropy_with_logits(output, y[train], pos_weight=weight)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        model.eval()
        with torch.no_grad(): score = torch.sigmoid(model(*[value[validation] for value in tensors])).detach().cpu().numpy()
        value = _auc(label[validation], score)
        if best is None or value > best[0]:
            best = (value, {name: item.detach().cpu().clone() for name, item in model.state_dict().items()})
    assert best is not None
    model.load_state_dict(best[1]); model.eval()
    with torch.no_grad():
        val = torch.sigmoid(model(*[value[validation] for value in tensors])).detach().cpu().numpy()
        score = torch.sigmoid(model(*[value[test] for value in tensors])).detach().cpu().numpy()
    return val, score, _threshold(label[validation], val)


def _flat_scores(values: np.ndarray, label: np.ndarray, split: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[float]]:
    train = split == "train"
    values = _standardize(values[train], values)
    tensor = torch.as_tensor(values, device=DEVICE)
    val_scores, test_scores, thresholds = [], [], []
    for seed in SEEDS:
        val, test, threshold = _train(FlatHead(values.shape[1]), (tensor,), label, split, seed)
        val_scores.append(val); test_scores.append(test); thresholds.append(threshold)
    return np.stack(val_scores), np.stack(test_scores), thresholds


def _sequence_scores(kind: str, visual: np.ndarray, history: np.ndarray, action: np.ndarray, label: np.ndarray, split: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[float]]:
    train = split == "train"
    visual = _standardize(visual[train].reshape(train.sum(), -1), visual.reshape(len(visual), -1)).reshape(visual.shape)
    history = _standardize(history[train].reshape(train.sum(), -1), history.reshape(len(history), -1)).reshape(history.shape)
    action = _standardize(action[train].reshape(train.sum(), -1), action.reshape(len(action), -1)).reshape(action.shape)
    vt, ht, at = (torch.as_tensor(value, device=DEVICE) for value in (visual, history, action))
    if kind == "state_action_gru":
        model_factory: Callable[[], nn.Module] = StateActionGRU
        tensors = (ht, at)
    elif kind == "visual_history_gru":
        model_factory = VisualGRU
        tensors = (vt,)
    elif kind == "visual_history_tcn":
        model_factory = VisualTCN
        tensors = (vt,)
    elif kind == "visual_state_gru":
        static = torch.as_tensor(history.reshape(len(history), -1), device=DEVICE)
        model_factory = lambda: VisualGRU(static.shape[1])
        tensors = (vt, static)
    elif kind == "visual_action_gru":
        static = torch.as_tensor(action.reshape(len(action), -1), device=DEVICE)
        model_factory = lambda: VisualGRU(static.shape[1])
        tensors = (vt, static)
    elif kind == "visual_state_action_gru":
        static = torch.as_tensor(np.concatenate((history.reshape(len(history), -1), action.reshape(len(action), -1)), axis=1), device=DEVICE)
        model_factory = lambda: VisualGRU(static.shape[1])
        tensors = (vt, static)
    else:
        raise ValueError(kind)
    val_scores, test_scores, thresholds = [], [], []
    for seed in SEEDS:
        val, test, threshold = _train(model_factory(), tensors, label, split, seed)
        val_scores.append(val); test_scores.append(test); thresholds.append(threshold)
    return np.stack(val_scores), np.stack(test_scores), thresholds


def _nearest_neighbor(visual: np.ndarray, label: np.ndarray, split: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[float]]:
    train, validation, test = (split == name for name in ("train", "validation", "test"))
    values = visual.mean(axis=1)
    values = values / np.maximum(1e-6, np.linalg.norm(values, axis=1, keepdims=True))
    def score(mask: np.ndarray) -> np.ndarray:
        similarity = values[mask] @ values[train].T
        nearest = np.argsort(-similarity, axis=1)[:, :7]
        return label[train][nearest].mean(axis=1).astype(np.float32)
    val, result = score(validation), score(test)
    return val[None], result[None], [_threshold(label[validation], val)]


def _task_identity(records: list[Mapping[str, Any]], split: np.ndarray) -> np.ndarray:
    names = sorted({str(row["task_name"]) for row, owner in zip(records, split) if owner == "train"})
    lookup = {name: index for index, name in enumerate(names)}
    result = np.zeros((len(records), len(names)), np.float32)
    for index, row in enumerate(records):
        if str(row["task_name"]) in lookup:
            result[index, lookup[str(row["task_name"])]] = 1.0
    return result


def _record_scores(name: str, records: list[Mapping[str, Any]], labels: np.ndarray, split: np.ndarray, val: np.ndarray, test: np.ndarray, thresholds: list[float], raw_dir: Path) -> dict[str, Any]:
    test_records = [row for row, owner in zip(records, split) if owner == "test"]
    test_label = labels[split == "test"]
    mean = test.mean(axis=0)
    threshold = float(np.mean(thresholds))
    metric = _metrics(test_label, mean, threshold)
    validation_label = labels[split == "validation"]
    metric["validation_metric"] = _metrics(validation_label, val.mean(axis=0), threshold)
    metric["validation_thresholds"] = [float(value) for value in thresholds]
    metric["per_seed_balanced_accuracy"] = [_metrics(test_label, value, threshold)["balanced_accuracy"] for value in test]
    metric["seed_range"] = float(max(metric["per_seed_balanced_accuracy"]) - min(metric["per_seed_balanced_accuracy"]))
    rows = []
    for index, (record, target, score) in enumerate(zip(test_records, test_label, mean)):
        rows.append({"model": name, "window_id": record["window_id"], "episode_id": record["episode_id"], "task_name": record["task_name"], "horizon_steps": record["horizon_steps"], "label": int(target), "score_mean": float(score), "seed_scores": [float(value) for value in test[:, index]]})
    write_jsonl(raw_dir / f"{name}.jsonl", rows)
    return metric


def _run_horizon(horizon: int, records: list[Mapping[str, Any]], history: np.ndarray, action: np.ndarray, labels: np.ndarray, mode_a: np.ndarray, visual: np.ndarray, future_visual: np.ndarray, *, split_key: str) -> dict[str, Any]:
    mask = np.asarray([row["horizon_steps"] == horizon for row in records])
    subset = [row for row, use in zip(records, mask) if use]
    split = np.asarray([str(row[split_key]) for row in subset])
    history, action, labels, mode_a, visual, future_visual = (value[mask] for value in (history, action, labels, mode_a, visual, future_visual))
    raw_dir = RESULT_DIR / "raw_scores" / split_key / f"h{horizon}"
    raw_dir.mkdir(parents=True, exist_ok=True)
    flat = {
        "shortcut_task_identity": _task_identity(subset, split),
        "shortcut_episode_time_proxy": np.asarray([[row["anchor"], row["anchor"] / max(1, row["target_end_exclusive"])] for row in subset], np.float32),
        "shortcut_action_only": action.reshape(len(action), -1),
        "shortcut_action_magnitude_smoothness": np.asarray([[row["matching"]["window_metrics"]["action_magnitude"], row["matching"]["window_metrics"]["action_smoothness"]] for row in subset], np.float32),
        "shortcut_robot_state_only": history.reshape(len(history), -1),
        "shortcut_current_rgb_only": visual[:, -1],
        "shortcut_brightness_motion_only": mode_a,
    }
    results: dict[str, Any] = {}
    # Constants/source/view are legal diagnostics but not learned models.
    test_count = int((split == "test").sum())
    constant = np.full(test_count, labels[split == "train"].mean(), np.float32)
    for name, score in {"shortcut_constant": constant, "shortcut_source_identity": constant, "shortcut_camera_view": constant}.items():
        threshold = 0.5
        results[name] = _metrics(labels[split == "test"], score, threshold) | {"validation_thresholds": [threshold], "per_seed_balanced_accuracy": [_metrics(labels[split == "test"], score, threshold)["balanced_accuracy"]], "seed_range": 0.0}
        write_jsonl(raw_dir / f"{name}.jsonl", [{"model": name, "window_id": row["window_id"], "episode_id": row["episode_id"], "task_name": row["task_name"], "horizon_steps": horizon, "label": int(label), "score_mean": float(value), "seed_scores": [float(value)]} for row, label, value in zip([row for row, owner in zip(subset, split) if owner == "test"], labels[split == "test"], score)])
    for name, value in flat.items():
        val, test, thresholds = _flat_scores(value, labels, split)
        results[name] = _record_scores(name, subset, labels, split, val, test, thresholds, raw_dir)
    val, test, thresholds = _flat_scores(future_visual, labels, split)
    leakage_metric = _record_scores("privileged_future_rgb_leakage_oracle", subset, labels, split, val, test, thresholds, raw_dir)
    leakage_metric["leakage_oracle"] = True
    results["privileged_future_rgb_leakage_oracle"] = leakage_metric
    for name in ("state_action_gru", "visual_history_gru", "visual_history_tcn", "visual_state_gru", "visual_action_gru", "visual_state_action_gru"):
        val, test, thresholds = _sequence_scores(name, visual, history, action, labels, split)
        results[f"standard_{name}"] = _record_scores(f"standard_{name}", subset, labels, split, val, test, thresholds, raw_dir)
    val, test, thresholds = _nearest_neighbor(visual, labels, split)
    results["standard_nearest_neighbor"] = _record_scores("standard_nearest_neighbor", subset, labels, split, val, test, thresholds, raw_dir)
    return {"horizon": horizon, "split_key": split_key, "n": len(subset), "split_counts": {name: int((split == name).sum()) for name in ("train", "validation", "test")}, "metrics": results}


def run() -> dict[str, Any]:
    feature_path = FEATURE_DIR / "visual_features.npz"
    if not feature_path.is_file():
        raise FileNotFoundError("extract frozen visual features before running R3 baselines")
    task = build_task()
    feature = np.load(feature_path)
    records, history, action, labels = task["windows"], task["history"], task["action"], task["label"]
    mode_a, visual, future_visual = feature["mode_a"], feature["resnet_history"], feature["future_resnet_diagnostic"]
    reports_by_split = {
        "task_heldout": {str(horizon): _run_horizon(horizon, records, history, action, labels, mode_a, visual, future_visual, split_key="task_split") for horizon in (8, 16, 32)},
        "episode_heldout": {str(horizon): _run_horizon(horizon, records, history, action, labels, mode_a, visual, future_visual, split_key="episode_split") for horizon in (8, 16, 32)},
    }
    reports = reports_by_split["task_heldout"]
    # Choose the primary horizon/model on validation only.  We reconstruct this
    # decision from the seed-mean test report only for reporting; the model
    # families and horizons were all preregistered, so no additional training
    # or test-driven hyperparameter selection occurs.
    fair_names = ["standard_visual_history_gru", "standard_visual_history_tcn", "standard_visual_state_gru", "standard_visual_action_gru", "standard_visual_state_action_gru", "standard_nearest_neighbor"]
    best_entries = [(float(report["metrics"][name]["validation_metric"]["balanced_accuracy"]), horizon, name) for horizon, report in reports.items() for name in fair_names]
    # Horizon/model selection uses validation only. Test metrics below are
    # subsequently reported once for that frozen validation choice.
    _, selected_horizon, selected_model = max(best_entries)
    selected = reports[selected_horizon]["metrics"]
    best_visual = float(selected[selected_model]["balanced_accuracy"])
    numeric_controls = ["shortcut_action_only", "shortcut_action_magnitude_smoothness", "shortcut_robot_state_only"]
    numeric_best = max(float(selected[name]["balanced_accuracy"]) for name in numeric_controls)
    current = float(selected["shortcut_current_rgb_only"]["balanced_accuracy"])
    standard_names = [name for name in selected if name.startswith("standard_")]
    standard_best = max(float(selected[name]["balanced_accuracy"]) for name in standard_names)
    stable = float(selected[selected_model]["seed_range"]) <= 0.10
    if all(max(float(report["metrics"][name]["balanced_accuracy"]) for name in fair_names) <= 0.60 for report in reports.values()):
        decision = "R3_FAILURE_NOT_PREDICTABLE_FROM_AVAILABLE_PREHISTORY"
    elif standard_best >= 0.90:
        decision = "R3_STANDARD_VISUAL_BASELINES_SATURATE"
    elif numeric_best <= 0.65 and current <= 0.70 and best_visual >= 0.65 and standard_best < 0.90 and stable and best_visual > 0.50:
        decision = "R3_VISUAL_HEADROOM_FOUND"
    else:
        decision = "R3_REAL_VISUAL_BENCHMARK_SUPPORTED_METHOD_NOT_YET"
    gate = {"decision": decision, "selected_horizon": int(selected_horizon), "selected_model": selected_model, "numeric_shortcut_best": numeric_best, "current_rgb_only": current, "best_fair_visual": best_visual, "best_standard": standard_best, "selected_seed_range": float(selected[selected_model]["seed_range"]), "gate_pass": decision == "R3_VISUAL_HEADROOM_FOUND"}
    report = {"schema": "r3-visual-baseline-report-v1", "device": str(DEVICE), "seeds": list(SEEDS), "reports_by_split": reports_by_split, "recoverability_gate": gate, "note": "All model candidates and horizons were preregistered. Source/view controls are constant because every R3 sample uses one test-source camera view."}
    write_json(RESULT_DIR / "visual_baseline_report.json", report)
    r2_append_log(OUT / "run_log.jsonl", event="r3_visual_baselines_complete", payload=gate)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        parser.error("choose --run")
    report = run()
    print(json.dumps(report["recoverability_gate"], sort_keys=True))


if __name__ == "__main__":
    main()
