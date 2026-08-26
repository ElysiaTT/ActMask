"""The single authorized R2 method: RealRelDynVerifier and fixed ablations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from actmask.experiments.r2_botfails_verification import DEVICE, OUT, SEED, _metrics, build_formulation
from actmask.data.r2_real_outcome_adapter import r2_append_log
from actmask.data.real_robot_dataset_adapter import write_json, write_jsonl


METHOD_ROOT = OUT / "method_results" / "real_rel_dyn_verifier"
METHOD_SEEDS = (20260725, 20260726, 20260727)


class RealRelDynVerifier(nn.Module):
    """Condition a candidate control embedding on observed state dynamics.

    Both streams are historical or proposed-control inputs.  The multiplicative
    state/action interaction is the only relational component; no target-time
    robot state or failure label is supplied to this network.
    """

    def __init__(self, ablation: str) -> None:
        super().__init__()
        self.ablation = ablation
        self.state_encoder = nn.GRU(6, 32, batch_first=True)
        self.action_encoder = nn.GRU(6, 32, batch_first=True)
        if ablation == "full":
            width = 32 * 4  # state, action, state*action, state-action
        elif ablation == "no_relational_interaction":
            width = 64
        elif ablation == "no_action":
            width = 32
        elif ablation == "no_history":
            width = 32
        else:
            raise ValueError(ablation)
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 64), nn.GELU(), nn.Dropout(0.1), nn.Linear(64, 1))

    def forward(self, history: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        state = self.state_encoder(history)[1][-1]
        control = self.action_encoder(action)[1][-1]
        if self.ablation == "full":
            value = torch.cat((state, control, state * control, state - control), dim=1)
        elif self.ablation == "no_relational_interaction":
            value = torch.cat((state, control), dim=1)
        elif self.ablation == "no_action":
            value = state
        else:
            value = control
        return self.head(value).squeeze(-1)


def _standardize(history: np.ndarray, action: np.ndarray, train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    joined = np.concatenate((history.reshape(len(history), -1), action.reshape(len(action), -1)), axis=1)
    mean, std = joined[train].mean(axis=0), joined[train].std(axis=0)
    std = np.where(std < 1e-5, 1.0, std)
    joined = (joined - mean) / std
    hwidth = history.shape[1] * history.shape[2]
    return joined[:, :hwidth].reshape(history.shape).astype(np.float32), joined[:, hwidth:].reshape(action.shape).astype(np.float32)


def _fit_one(ablation: str, history: np.ndarray, action: np.ndarray, label: np.ndarray, split: np.ndarray, seed: int) -> np.ndarray:
    train, validation, test = split == "train", split == "validation", split == "test"
    history, action = _standardize(history, action, train)
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = RealRelDynVerifier(ablation).to(DEVICE)
    h = torch.as_tensor(history, device=DEVICE)
    a = torch.as_tensor(action, device=DEVICE)
    y = torch.as_tensor(label.astype(np.float32), device=DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=5e-4)
    best: tuple[float, dict[str, torch.Tensor]] | None = None
    for _ in range(240):
        model.train()
        optimizer.zero_grad()
        loss = nn.functional.binary_cross_entropy_with_logits(model(h[train], a[train]), y[train])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            value = torch.sigmoid(model(h[validation], a[validation])).detach().cpu().numpy()
        metric = _metrics(label[validation], value)["balanced_accuracy"]
        if best is None or metric > best[0]:
            best = (metric, {key: item.detach().cpu().clone() for key, item in model.state_dict().items()})
    assert best is not None
    model.load_state_dict(best[1])
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(h[test], a[test])).detach().cpu().numpy()


def _pair_order(records: list[dict[str, Any]], labels: np.ndarray, scores: np.ndarray) -> float:
    by_pair: dict[str, list[tuple[int, float]]] = {}
    for row, label, score in zip(records, labels, scores):
        by_pair.setdefault(str(row["pair_id"]), []).append((int(label), float(score)))
    outcomes = []
    for pair in by_pair.values():
        if len(pair) != 2:
            raise RuntimeError("pair split unexpectedly")
        positive = next(score for label, score in pair if label == 1)
        negative = next(score for label, score in pair if label == 0)
        outcomes.append(1.0 if positive > negative else 0.5 if positive == negative else 0.0)
    return float(np.mean(outcomes))


def _bootstrap_pair_ci(records: list[dict[str, Any]], labels: np.ndarray, scores: np.ndarray, *, samples: int = 2000) -> list[float]:
    by_pair: dict[str, list[int]] = {}
    for index, row in enumerate(records):
        by_pair.setdefault(str(row["pair_id"]), []).append(index)
    groups = list(by_pair.values())
    rng = np.random.default_rng(SEED)
    values = []
    for _ in range(samples):
        indices = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        values.append(_metrics(labels[indices], scores[indices])["balanced_accuracy"])
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def run() -> dict[str, Any]:
    formulation = build_formulation()
    records = formulation["records"]
    history, action, labels = formulation["history"], formulation["action"], formulation["label"]
    split = np.asarray([str(row["hard_split"]) for row in records])
    test = split == "test"
    test_records = [row for row, include in zip(records, test) if include]
    test_labels = labels[test]
    results: dict[str, Any] = {}
    raw_dir = METHOD_ROOT / "raw_scores"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for ablation in ("full", "no_relational_interaction", "no_action", "no_history"):
        all_scores = np.stack([_fit_one(ablation, history, action, labels, split, seed) for seed in METHOD_SEEDS])
        score = all_scores.mean(axis=0)
        metrics = _metrics(test_labels, score)
        metrics["pair_order_accuracy"] = _pair_order(test_records, test_labels, score)
        metrics["pair_bootstrap_95ci_balanced_accuracy"] = _bootstrap_pair_ci(test_records, test_labels, score)
        metrics["per_seed_balanced_accuracy"] = [_metrics(test_labels, row)["balanced_accuracy"] for row in all_scores]
        results[ablation] = metrics
        rows = []
        for index, (record, target, averaged) in enumerate(zip(test_records, test_labels, score)):
            rows.append({
                "variant": ablation, "sample_id": record["sample_id"], "pair_id": record["pair_id"], "task_name": record["task_name"],
                "label": int(target), "score_mean": float(averaged), "seed_scores": [float(value) for value in all_scores[:, index]],
            })
        write_jsonl(raw_dir / f"{ablation}.jsonl", rows)
    baseline_path = OUT / "baseline_results" / "formulation_a_baseline_report.json"
    baseline = json.loads(baseline_path.read_text())
    best_standard = float(baseline["gate"]["standard_best"])
    full = results["full"]
    improvement = float(full["balanced_accuracy"] - best_standard)
    authorized = improvement >= 0.08
    report = {
        "schema": "r2-real-rel-dyn-verifier-v1",
        "device": str(DEVICE),
        "method_trial_count": 1,
        "seeds": list(METHOD_SEEDS),
        "primary_split": "whole-task held-out",
        "method": "state GRU + action GRU + multiplicative/difference relational interaction",
        "inputs": ["history_robot_state", "candidate_action_chunk"],
        "target": "official failure onset within candidate chunk",
        "results": results,
        "best_standard_balanced_accuracy": best_standard,
        "full_method_improvement": improvement,
        "method_gate": {"minimum_improvement": 0.08, "pass": authorized, "decision": "method_supported" if authorized else "method_not_supported"},
        "raw_score_dir": str(raw_dir.relative_to(OUT)),
    }
    write_json(METHOD_ROOT / "method_report.json", report)
    r2_append_log(OUT / "run_log.jsonl", event="r2_real_rel_dyn_verifier_complete", payload={"improvement": improvement, "gate_pass": authorized, "full": full})
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        parser.error("choose --run")
    report = run()
    print(json.dumps(report["method_gate"], sort_keys=True))


if __name__ == "__main__":
    main()
