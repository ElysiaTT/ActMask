"""R2 Formulation A: real logged failure-onset verification on BotFails.

Each candidate is a 32-step state history plus a 16-step logged action chunk.
For every official failure trajectory, the positive chunk is the one containing
the first non-zero official failure label; its negative is the immediately
preceding, wholly zero-labelled chunk from the *same trajectory*.  This makes
source, robot, task and episode provenance constant within a pair and keeps
all labels and all post-chunk observations out of model inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from torch import nn

from actmask.data.r2_real_outcome_adapter import BOTFAILS_DATASET, R2_ROOT, r2_append_log
from actmask.data.real_robot_dataset_adapter import forbidden_input_audit, read_jsonl, sha256_file, write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "actmask" / "r2_series_real_outcome_search"
PROCESSED = OUT / "processed_subset" / "botfails"
TASK_ROOT = OUT / "task_formulation" / "formulation_a_failure_onset"
HISTORY = 32
ACTION = 16
SEED = 20260725
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TaskConstructionError(RuntimeError):
    pass


def _load_array(row: Mapping[str, Any], name: str) -> np.ndarray:
    return np.load(PROCESSED / str(row[name]))["value"].astype(np.float32)


def _hard_split(row: Mapping[str, Any]) -> str:
    return str(dict(row["split_metadata"])["hard_task_heldout"])


def _formulation_config() -> dict[str, Any]:
    return {
        "schema": "r2-botfails-formulation-a-v1",
        "dataset": BOTFAILS_DATASET,
        "formulation": "prospective logged failure-onset verification",
        "history_steps": HISTORY,
        "candidate_action_steps": ACTION,
        "positive": "official binary label first becomes one inside candidate action chunk",
        "negative": "immediately preceding same-episode action chunk whose official labels are all zero",
        "primary_split": "whole-task held-out split from adapter metadata",
        "model_inputs": ["history_robot_state", "candidate_action_chunk"],
        "target_only": ["official_failure_label_in_candidate_chunk", "post_candidate_observation"],
        "excluded_features": ["episode_id", "source_split", "task_id", "language_instruction", "timestamp", "failure_label", "future_observation"],
        "claim": "logged real-robot action-conditioned failure-onset verification, not a physical counterfactual intervention claim",
    }


def build_formulation() -> dict[str, Any]:
    """Build the paired task and emit mandatory leakage/grouping audits."""

    manifest = PROCESSED / "data_manifest.json"
    episodes_path = PROCESSED / "episodes.jsonl"
    if not (manifest.is_file() and episodes_path.is_file()):
        raise FileNotFoundError("BotFails normalized subset is absent; run the R2 adapter first")
    rows = read_jsonl(episodes_path)
    failure_rows = [row for row in rows if row["success_or_failure"] == "failure"]
    if len(failure_rows) != 50:
        raise TaskConstructionError(f"expected 50 selected failure trajectories, got {len(failure_rows)}")
    records: list[dict[str, Any]] = []
    histories: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    labels: list[int] = []
    for episode in sorted(failure_rows, key=lambda row: str(row["episode_id"])):
        state, action, failure = (_load_array(episode, name) for name in ("robot_state_path", "action_path", "failure_label_path"))
        first = np.flatnonzero(failure)
        if not len(first):
            raise TaskConstructionError(f"failure row lacks a positive label: {episode['episode_id']}")
        onset = int(first[0])
        positive_anchor = onset - ACTION // 2
        negative_anchor = positive_anchor - ACTION
        if negative_anchor < HISTORY:
            raise TaskConstructionError(f"onset is too early for causal paired window: {episode['episode_id']} at {onset}")
        if failure[positive_anchor - HISTORY:positive_anchor].any():
            raise TaskConstructionError(f"positive history already contains a failure label: {episode['episode_id']}")
        if failure[negative_anchor - HISTORY:negative_anchor + ACTION].any():
            raise TaskConstructionError(f"negative window is not wholly pre-failure: {episode['episode_id']}")
        if not failure[positive_anchor:positive_anchor + ACTION].any():
            raise TaskConstructionError(f"positive candidate misses onset: {episode['episode_id']}")
        for kind, anchor, label in (("pre_onset", negative_anchor, 0), ("onset", positive_anchor, 1)):
            histories.append(state[anchor - HISTORY:anchor])
            actions.append(action[anchor:anchor + ACTION])
            labels.append(label)
            records.append({
                "sample_id": f"{episode['episode_id']}:{kind}",
                "episode_id": episode["episode_id"],
                "pair_id": str(episode["episode_id"]),
                "task_name": episode["task_name"],
                "hard_split": _hard_split(episode),
                "anchor": int(anchor),
                "history_start": int(anchor - HISTORY),
                "candidate_end_exclusive": int(anchor + ACTION),
                "observed_current_timestamp_step": int(anchor),
                "episode_length": int(len(state)),
                "candidate_kind": kind,
                "target_failure_in_candidate": label,
                "source_partition": "test",  # retained for an audit only, never features
                "official_first_positive_step": onset,
            })
    history_array = np.stack(histories).astype(np.float32)
    action_array = np.stack(actions).astype(np.float32)
    label_array = np.asarray(labels, dtype=np.int64)
    if history_array.shape != (100, HISTORY, 6) or action_array.shape != (100, ACTION, 6):
        raise TaskConstructionError(f"unexpected formulation shapes {history_array.shape} / {action_array.shape}")
    if label_array.sum() != 50:
        raise TaskConstructionError("paired outcome labels are not balanced")
    task_counts = Counter(str(row["task_name"]) for row in records)
    split_counts = Counter(str(row["hard_split"]) for row in records)
    pair_counts = Counter(str(row["pair_id"]) for row in records)
    leakage_audit = {
        "pass": True,
        "future_observation_used": False,
        "failure_label_used_as_input": False,
        "source_partition_used_as_input": False,
        "task_or_language_used_as_input": False,
        "pair_grouping": "both candidates from the same episode; every pair is assigned to exactly one whole-task split",
        "input_audit": forbidden_input_audit(["history_robot_state", "candidate_action_chunk"]),
        "per_record_checks": {
            "history_strictly_before_candidate": all(int(row["history_start"]) < int(row["anchor"]) for row in records),
            "positive_has_future_target": all(row["candidate_kind"] != "onset" or row["target_failure_in_candidate"] == 1 for row in records),
            "negative_is_pre_failure": all(row["candidate_kind"] != "pre_onset" or row["target_failure_in_candidate"] == 0 for row in records),
        },
    }
    leakage_audit["pass"] = leakage_audit["input_audit"]["pass"] and all(leakage_audit["per_record_checks"].values())
    if not leakage_audit["pass"]:
        raise TaskConstructionError(f"leakage audit failed: {leakage_audit}")
    TASK_ROOT.mkdir(parents=True, exist_ok=True)
    config = _formulation_config()
    write_json(TASK_ROOT / "config.json", config)
    write_jsonl(TASK_ROOT / "samples.jsonl", records)
    np.savez_compressed(TASK_ROOT / "segments.npz", history=history_array, action=action_array, label=label_array)
    audit = {
        "schema": "r2-botfails-formulation-a-audit-v1",
        "pass": True,
        "samples": len(records),
        "pairs": len(pair_counts),
        "labels": {"pre_onset": int((label_array == 0).sum()), "onset": int((label_array == 1).sum())},
        "task_counts": dict(sorted(task_counts.items())),
        "hard_split_counts": dict(split_counts),
        "source_partition_counts": dict(Counter(str(row["source_partition"]) for row in records)),
        "leakage_audit": leakage_audit,
        "repeated_attempt_support": {task: count // 2 for task, count in sorted(task_counts.items())},
        "negative_policy": "one exact same-episode pre-onset chunk; no pool fallback and no cross-source negative",
    }
    write_json(TASK_ROOT / "task_audit.json", audit)
    write_json(TASK_ROOT / "task_manifest.json", {
        "schema": "r2-botfails-formulation-a-manifest-v1",
        "processed_manifest_sha256": sha256_file(manifest),
        "config_sha256": sha256_file(TASK_ROOT / "config.json"),
        "samples_sha256": sha256_file(TASK_ROOT / "samples.jsonl"),
        "segments_sha256": sha256_file(TASK_ROOT / "segments.npz"),
    })
    r2_append_log(OUT / "run_log.jsonl", event="r2_botfails_formulation_a_constructed", payload={"samples": len(records), "pairs": len(pair_counts), "hard_splits": dict(split_counts)})
    return {"records": records, "history": history_array, "action": action_array, "label": label_array, "audit": audit}


def _auc(labels: np.ndarray, score: np.ndarray) -> float:
    pos, neg = score[labels == 1], score[labels == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    return float((np.greater(pos[:, None], neg[None, :]).mean()) + 0.5 * np.equal(pos[:, None], neg[None, :]).mean())


def _metrics(label: np.ndarray, score: np.ndarray) -> dict[str, float]:
    pred = (score >= 0.5).astype(np.int64)
    tp = int(((pred == 1) & (label == 1)).sum())
    tn = int(((pred == 0) & (label == 0)).sum())
    fp = int(((pred == 1) & (label == 0)).sum())
    fn = int(((pred == 0) & (label == 1)).sum())
    tpr = tp / max(1, tp + fn)
    tnr = tn / max(1, tn + fp)
    return {"accuracy": float((pred == label).mean()), "balanced_accuracy": float((tpr + tnr) / 2), "auroc": _auc(label, score), "n": int(len(label))}


def _normalize(train: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean, std = train.mean(axis=0), train.std(axis=0)
    std = np.where(std < 1e-5, 1.0, std)
    return (values - mean) / std, mean, std


class FlatMLP(nn.Module):
    def __init__(self, features: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(features, 64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, 1))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value).squeeze(-1)


class SequenceGRU(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gru = nn.GRU(12, 32, batch_first=True)
        self.head = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, history: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        summary = action.mean(dim=1, keepdim=True).expand(-1, history.shape[1], -1)
        output, _ = self.gru(torch.cat((history, summary), dim=-1))
        return self.head(output[:, -1]).squeeze(-1)


class SequenceTCN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(12, 32, 3, padding=1), nn.ReLU(), nn.Conv1d(32, 32, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(1))
        self.head = nn.Linear(32, 1)

    def forward(self, history: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        summary = action.mean(dim=1, keepdim=True).expand(-1, history.shape[1], -1)
        return self.head(self.net(torch.cat((history, summary), dim=-1).transpose(1, 2)).squeeze(-1)).squeeze(-1)


def _train_flat(name: str, values: np.ndarray, labels: np.ndarray, split: np.ndarray, *, epochs: int = 160) -> np.ndarray:
    train, validation, test = split == "train", split == "validation", split == "test"
    all_values, _, _ = _normalize(values[train], values)
    torch.manual_seed(SEED)
    model = FlatMLP(all_values.shape[1]).to(DEVICE)
    x, y = torch.as_tensor(all_values, device=DEVICE), torch.as_tensor(labels.astype(np.float32), device=DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    best: tuple[float, dict[str, torch.Tensor]] | None = None
    for _ in range(epochs):
        model.train(); optimizer.zero_grad(); loss = nn.functional.binary_cross_entropy_with_logits(model(x[train]), y[train]); loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad():
            score = torch.sigmoid(model(x[validation])).detach().cpu().numpy()
        value = _metrics(labels[validation], score)["balanced_accuracy"]
        if best is None or value > best[0]:
            best = (value, {key: item.detach().cpu().clone() for key, item in model.state_dict().items()})
    assert best is not None
    model.load_state_dict(best[1]); model.eval()
    with torch.no_grad():
        score = torch.sigmoid(model(x[test])).detach().cpu().numpy()
    return score


def _train_sequence(kind: str, history: np.ndarray, action: np.ndarray, labels: np.ndarray, split: np.ndarray, *, epochs: int = 180) -> np.ndarray:
    train, validation, test = split == "train", split == "validation", split == "test"
    joined = np.concatenate((history.reshape(len(history), -1), action.reshape(len(action), -1)), axis=1)
    normalized, mean, std = _normalize(joined[train], joined)
    hwidth = history.shape[1] * history.shape[2]
    history_n = normalized[:, :hwidth].reshape(history.shape)
    action_n = normalized[:, hwidth:].reshape(action.shape)
    torch.manual_seed(SEED)
    model = SequenceGRU() if kind == "gru" else SequenceTCN()
    model = model.to(DEVICE)
    h, a, y = (torch.as_tensor(history_n, device=DEVICE), torch.as_tensor(action_n, device=DEVICE), torch.as_tensor(labels.astype(np.float32), device=DEVICE))
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    best: tuple[float, dict[str, torch.Tensor]] | None = None
    for _ in range(epochs):
        model.train(); optimizer.zero_grad(); loss = nn.functional.binary_cross_entropy_with_logits(model(h[train], a[train]), y[train]); loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad(): score = torch.sigmoid(model(h[validation], a[validation])).detach().cpu().numpy()
        value = _metrics(labels[validation], score)["balanced_accuracy"]
        if best is None or value > best[0]:
            best = (value, {key: item.detach().cpu().clone() for key, item in model.state_dict().items()})
    assert best is not None
    model.load_state_dict(best[1]); model.eval()
    with torch.no_grad(): return torch.sigmoid(model(h[test], a[test])).detach().cpu().numpy()


def _task_one_hot(records: Iterable[Mapping[str, Any]], split: np.ndarray) -> np.ndarray:
    names = sorted({str(row["task_name"]) for row, owner in zip(records, split) if owner == "train"})
    lookup = {name: index for index, name in enumerate(names)}
    value = np.zeros((len(split), len(names)), dtype=np.float32)
    for index, row in enumerate(records):
        if str(row["task_name"]) in lookup:
            value[index, lookup[str(row["task_name"])]] = 1.0
    return value


def run_baselines() -> dict[str, Any]:
    """Run required shortcut controls and standard numeric baselines on GPU."""

    result = build_formulation()
    records, history, action, labels = result["records"], result["history"], result["action"], result["label"]
    split = np.asarray([str(row["hard_split"]) for row in records])
    if Counter(split) != {"train": 60, "validation": 20, "test": 20}:
        raise TaskConstructionError(f"unexpected primary hard split distribution {Counter(split)}")
    test_mask = split == "test"
    flat_state = history.reshape(len(history), -1)
    flat_action = action.reshape(len(action), -1)
    # A deliberately weak, but legal, timestamp-only control: it sees only the
    # current recorded step, never the failure onset or final episode outcome.
    timestamp_proxy = np.asarray([float(row["observed_current_timestamp_step"]) for row in records], dtype=np.float32)[:, None]
    current_smoothness = np.linalg.norm(history[:, -1] - history[:, -2], axis=1, keepdims=True).astype(np.float32)
    task_id = _task_one_hot(records, split)
    rng = np.random.default_rng(SEED)
    shuffled_action = flat_action[rng.permutation(len(flat_action))]
    models: list[tuple[str, str, Any]] = [
        ("shortcut_action_only", "flat", flat_action),
        ("shortcut_task_identity", "flat", task_id),
        ("shortcut_timestamp_proxy", "flat", timestamp_proxy),
        ("shortcut_current_smoothness", "flat", current_smoothness),
        ("shortcut_source_identity", "constant", None),
        ("shortcut_shuffled_action", "flat", np.concatenate((flat_state, shuffled_action), axis=1)),
        ("standard_mlp_state_action", "flat", np.concatenate((flat_state, flat_action), axis=1)),
        ("standard_gru_state_action", "gru", None),
        ("standard_tcn_state_action", "tcn", None),
        ("upper_oracle_future_label", "oracle", None),
    ]
    output_rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    raw_dir = OUT / "baseline_results" / "formulation_a_raw_scores"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name, kind, values in models:
        if kind == "constant":
            score = np.full(int(test_mask.sum()), 0.5, np.float32)
        elif kind == "oracle":
            score = labels[test_mask].astype(np.float32)
        elif kind == "flat":
            score = _train_flat(name, np.asarray(values, np.float32), labels, split)
        else:
            score = _train_sequence(kind, history, action, labels, split)
        metric = _metrics(labels[test_mask], score)
        summaries[name] = metric
        raw = []
        for record, target, value in zip(np.asarray(records, dtype=object)[test_mask], labels[test_mask], score):
            raw.append({"model": name, "sample_id": record["sample_id"], "pair_id": record["pair_id"], "task_name": record["task_name"], "label": int(target), "score": float(value)})
        write_jsonl(raw_dir / f"{name}.jsonl", raw)
        output_rows.extend(raw)
    unavailable = {
        "visual_baseline": "not run: official RGB videos are deliberately unacquired during numeric-first validation",
        "contact_force_baseline": "not run: BotFails Parquet schema exposes no force/tactile/contact channel",
    }
    shortcut_names = [name for name in summaries if name.startswith("shortcut_")]
    standard_names = ["standard_mlp_state_action", "standard_gru_state_action", "standard_tcn_state_action"]
    shortcut_best = max(float(summaries[name]["balanced_accuracy"]) for name in shortcut_names)
    standard_best = max(float(summaries[name]["balanced_accuracy"]) for name in standard_names)
    oracle = float(summaries["upper_oracle_future_label"]["balanced_accuracy"])
    gate = {
        "shortcut_max": 0.65,
        "standard_hard_split_max": 0.88,
        "oracle_min": 0.95,
        "shortcut_best": shortcut_best,
        "standard_best": standard_best,
        "oracle": oracle,
        "pass": shortcut_best <= 0.65 and standard_best < 0.88 and oracle >= 0.95,
        "reason": "eligible_for_one_method" if shortcut_best <= 0.65 and standard_best < 0.88 and oracle >= 0.95 else "reject_or_reformulate_before_method",
    }
    report = {
        "schema": "r2-botfails-baseline-report-v1",
        "device": str(DEVICE),
        "primary_split": "whole-task held-out",
        "metrics": summaries,
        "unavailable_standard_controls": unavailable,
        "raw_score_dir": str(raw_dir.relative_to(OUT)),
        "gate": gate,
        "fairness": {"source_partition_constant": ["test"], "future_label_only_oracle": True, "task_identity_is_control_only": True},
    }
    write_json(OUT / "baseline_results" / "formulation_a_baseline_report.json", report)
    r2_append_log(OUT / "run_log.jsonl", event="r2_botfails_formulation_a_baselines_complete", payload={"gate": gate, "device": str(DEVICE)})
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--run-baselines", action="store_true")
    args = parser.parse_args()
    if not args.build and not args.run_baselines:
        parser.error("choose --build and/or --run-baselines")
    if args.build and not args.run_baselines:
        print(json.dumps({"samples": len(build_formulation()["records"])}))
    if args.run_baselines:
        print(json.dumps(run_baselines()["gate"], sort_keys=True))


if __name__ == "__main__":
    main()
