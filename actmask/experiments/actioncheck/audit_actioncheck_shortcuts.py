"""Train proposal-level shortcut controls without relational interactions."""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn

from .common import read_jsonl, stable_bucket, write_json, write_jsonl
from .evaluate_actioncheck import (
    best_balanced_threshold,
    classification_metrics,
)


CONTROL_NAMES = (
    "random",
    "task_only",
    "progress_only",
    "context_only",
    "action_only",
    "action_summary_only",
    "state_only",
    "visual_only",
    "policy_source_only",
    "evidence_type_only",
    "reason_code_only",
    "metadata_only",
    "combined_non_relational",
)


class _Classifier(nn.Module):
    def __init__(self, width: int, nonlinear: bool) -> None:
        super().__init__()
        hidden = min(128, max(32, width // 2))
        self.network = (
            nn.Sequential(
                nn.Linear(width, hidden),
                nn.GELU(),
                nn.Linear(hidden, hidden // 2),
                nn.GELU(),
                nn.Linear(hidden // 2, 1),
            )
            if nonlinear
            else nn.Linear(width, 1)
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value).squeeze(1)


def _one_hot(values: Sequence[str], categories: Sequence[str]) -> np.ndarray:
    lookup = {value: index for index, value in enumerate(categories)}
    result = np.zeros((len(values), len(categories) + 1), dtype=np.float32)
    for row, value in enumerate(values):
        result[row, lookup.get(value, len(categories))] = 1.0
    return result


def _action_summary(action: np.ndarray, mask: np.ndarray) -> np.ndarray:
    values = action.copy()
    valid = mask.any(axis=2)
    lengths = valid.sum(axis=1).clip(min=1)
    features = []
    for row in range(len(values)):
        selected = values[row, valid[row]]
        delta = np.diff(selected, axis=0)
        features.append(
            np.concatenate(
                (
                    selected.mean(axis=0),
                    selected.std(axis=0),
                    selected[0],
                    selected[-1],
                    selected.sum(axis=0),
                    np.asarray(
                        [
                            np.linalg.norm(selected, axis=1).mean(),
                            np.linalg.norm(selected, axis=1).std(),
                            np.linalg.norm(delta, axis=1).mean()
                            if len(delta)
                            else 0.0,
                            np.linalg.norm(np.diff(delta, axis=0), axis=1).mean()
                            if len(delta) > 1
                            else 0.0,
                            float(lengths[row]),
                        ],
                        dtype=np.float32,
                    ),
                )
            )
        )
    return np.asarray(features, dtype=np.float32)


def _fit(
    features: np.ndarray,
    labels: np.ndarray,
    split: np.ndarray,
    *,
    nonlinear: bool,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    train = split == "train"
    validation = split == "validation"
    mean = features[train].mean(axis=0, keepdims=True)
    scale = features[train].std(axis=0, keepdims=True)
    scale[scale < 1e-6] = 1.0
    normalized = np.nan_to_num((features - mean) / scale).astype(np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    value = torch.as_tensor(normalized, device=device)
    target = torch.as_tensor(labels.astype(np.float32), device=device)
    model = _Classifier(features.shape[1], nonlinear).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.005 if nonlinear else 0.02, weight_decay=2e-4
    )
    train_index = torch.as_tensor(np.flatnonzero(train), device=device)
    positive = float(labels[train].sum())
    pos_weight = torch.tensor(
        [(int(train.sum()) - positive) / max(positive, 1.0)], device=device
    )
    best_state = None
    best = -1.0
    patience = 0
    history = []
    for epoch in range(160):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(value[train_index])
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits, target[train_index], pos_weight=pos_weight
        )
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        if epoch % 5 == 4:
            model.eval()
            with torch.no_grad():
                score = torch.sigmoid(model(value)).detach().cpu().numpy()
            threshold = best_balanced_threshold(labels[validation], score[validation])
            metric = classification_metrics(
                labels[validation], score[validation], threshold
            )["balanced_accuracy"]
            history.append(
                {
                    "epoch": epoch + 1,
                    "loss": float(loss.detach().cpu()),
                    "validation_balanced_accuracy": metric,
                }
            )
            if metric > best + 1e-5:
                best = float(metric)
                best_state = {
                    name: parameter.detach().cpu().clone()
                    for name, parameter in model.state_dict().items()
                }
                patience = 0
            else:
                patience += 1
            if patience >= 10:
                break
    if best_state is None:
        raise RuntimeError("shortcut control failed to train")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        score = torch.sigmoid(model(value)).detach().cpu().numpy()
    threshold = best_balanced_threshold(labels[validation], score[validation])
    return score, {
        "seed": seed,
        "device": str(device),
        "nonlinear": nonlinear,
        "feature_dimension": int(features.shape[1]),
        "epochs": history[-1]["epoch"],
        "threshold_selected_on_validation": threshold,
        "best_validation_balanced_accuracy": best,
        "history": history,
    }


def _load_features(
    benchmark_dir: str | Path, scheme: str
) -> tuple[
    list[dict[str, Any]],
    np.ndarray,
    np.ndarray,
    dict[str, np.ndarray],
]:
    root = Path(benchmark_dir)
    candidates = [
        row
        for row in read_jsonl(root / "candidate_manifest.jsonl")
        if row["primary_eligible"]
    ]
    candidates.sort(key=lambda row: int(row["primary_array_index"]))
    with np.load(root / "benchmark_arrays.npz") as stored:
        arrays = {name: stored[name].copy() for name in stored.files}
    split_manifest = json.loads(
        (root / "split_manifest.json").read_text(encoding="utf-8")
    )
    assignments = split_manifest["schemes"][scheme]["assignments"]
    split = np.asarray(
        [assignments[row["context_id"]] for row in candidates], dtype="<U10"
    )
    labels = arrays["labels"].astype(np.int64)
    groups = arrays["group_index"].astype(np.int64)
    action = arrays["candidate_actions"].astype(np.float32)
    action_mask = arrays["candidate_action_mask"].astype(bool)
    state = arrays["state_history"][groups].reshape(len(candidates), -1)
    visual = arrays["visual_history"][groups].reshape(len(candidates), -1)
    gripper = arrays["gripper_history"][groups].reshape(len(candidates), -1)
    action_flat = action.reshape(len(candidates), -1)
    summary = _action_summary(action, action_mask)
    tasks = sorted({row["task_id"] for row in candidates})
    policies = sorted({row["source_policy"] for row in candidates})
    evidences = sorted({row["evidence_type_audit_only"] for row in candidates})
    reasons = sorted(
        {
            reason
            for row in candidates
            for reason in row["reason_codes_audit_only"]
        }
    )
    reason_values = [
        "|".join(row["reason_codes_audit_only"]) or "none" for row in candidates
    ]
    task = _one_hot([row["task_id"] for row in candidates], tasks)
    policy = _one_hot([row["source_policy"] for row in candidates], policies)
    evidence = _one_hot(
        [row["evidence_type_audit_only"] for row in candidates], evidences
    )
    reason = _one_hot(reason_values, sorted(set(reason_values)))
    progress = arrays["normalized_progress"][:, None].astype(np.float32)
    horizon = np.asarray(
        [[row["action_horizon"], row["action_dim"]] for row in candidates],
        dtype=np.float32,
    )
    context = np.concatenate((state, visual, gripper), axis=1)
    features = {
        "task_only": task,
        "progress_only": progress,
        "context_only": context,
        "action_only": action_flat,
        "action_summary_only": summary,
        "state_only": np.concatenate((state, gripper), axis=1),
        "visual_only": visual,
        "policy_source_only": policy,
        "evidence_type_only": evidence,
        "reason_code_only": reason,
        "metadata_only": np.concatenate((horizon, progress), axis=1),
        "combined_non_relational": np.concatenate(
            (context, summary, task, progress), axis=1
        ),
    }
    # Linear layers cannot form an explicit context-action interaction.
    return candidates, labels, split, features


def audit_shortcuts(
    benchmark_dir: str | Path,
    output_dir: str | Path,
    *,
    scheme: str = "episode_held_out",
    seed: int = 1701,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    candidates, labels, split, features = _load_features(benchmark_dir, scheme)
    reports: dict[str, Any] = {}
    predictions = []
    for name in CONTROL_NAMES:
        if name == "random":
            scores = np.asarray(
                [
                    stable_bucket(row["candidate_id"] + ":random", 10**9)
                    / float(10**9)
                    for row in candidates
                ],
                dtype=np.float32,
            )
            validation = split == "validation"
            training = {
                "seed": seed,
                "device": "deterministic_hash",
                "nonlinear": False,
                "feature_dimension": 0,
                "threshold_selected_on_validation": best_balanced_threshold(
                    labels[validation], scores[validation]
                ),
            }
        else:
            nonlinear = name in {
                "context_only",
                "action_only",
                "action_summary_only",
                "state_only",
                "visual_only",
            }
            # A zero-width absent modality is represented by one all-zero feature.
            value = features[name]
            if value.shape[1] == 0:
                value = np.zeros((len(value), 1), dtype=np.float32)
            scores, training = _fit(
                value, labels, split, nonlinear=nonlinear, seed=seed
            )
        threshold = float(training["threshold_selected_on_validation"])
        reports[name] = {
            "control": name,
            "training": training,
            "validation": classification_metrics(
                labels[split == "validation"],
                scores[split == "validation"],
                threshold,
            ),
            "test": classification_metrics(
                labels[split == "test"], scores[split == "test"], threshold
            ),
            "audit_only_metadata_input": name
            in {
                "policy_source_only",
                "evidence_type_only",
                "reason_code_only",
                "metadata_only",
            },
            "explicit_context_action_interaction": False,
        }
        for index, (candidate, label, score_value, owner) in enumerate(
            zip(candidates, labels, scores, split)
        ):
            if owner not in {"validation", "test"}:
                continue
            predictions.append(
                {
                    "control": name,
                    "candidate_id": candidate["candidate_id"],
                    "context_id": candidate["context_id"],
                    "episode_id": candidate["episode_id"],
                    "task_id": candidate["task_id"],
                    "label": int(label),
                    "score": float(score_value),
                    "split": str(owner),
                    "scheme": scheme,
                    "seed": seed,
                }
            )
    source_label: dict[str, Counter[int]] = defaultdict(Counter)
    reason_count: Counter[str] = Counter()
    rejects = 0
    for row, label in zip(candidates, labels):
        source_label[row["source_policy"]][int(label)] += 1
        if not label:
            rejects += 1
            reason_count.update(row["reason_codes_audit_only"])
    label_totals = Counter(labels.tolist())
    maximum_source_label_contribution = max(
        count / label_totals[label]
        for values in source_label.values()
        for label, count in values.items()
    )
    maximum_reason_fraction = max(reason_count.values(), default=0) / max(1, rejects)
    gates = {
        "policy_source_only_BA_le_0_53": reports["policy_source_only"]["test"][
            "balanced_accuracy"
        ]
        <= 0.53,
        "metadata_only_BA_le_0_53": reports["metadata_only"]["test"][
            "balanced_accuracy"
        ]
        <= 0.53,
        "action_only_BA_le_0_55": reports["action_only"]["test"][
            "balanced_accuracy"
        ]
        <= 0.55,
        "context_only_BA_le_0_55": reports["context_only"]["test"][
            "balanced_accuracy"
        ]
        <= 0.55,
        "action_summary_only_BA_le_0_56": reports["action_summary_only"]["test"][
            "balanced_accuracy"
        ]
        <= 0.56,
        "combined_non_relational_BA_le_0_60": reports[
            "combined_non_relational"
        ]["test"]["balanced_accuracy"]
        <= 0.60,
        "no_candidate_source_dominates_labels": maximum_source_label_contribution
        <= 0.80,
        "single_reason_code_fraction_le_0_40": maximum_reason_fraction <= 0.40,
    }
    result = {
        "schema": "actioncheck-shortcut-audit-v1",
        "scheme": scheme,
        "controls": reports,
        "gates": gates,
        "pass": all(gates.values()),
        "candidate_source_label_counts": {
            source: {str(label): count for label, count in sorted(values.items())}
            for source, values in sorted(source_label.items())
        },
        "maximum_candidate_source_label_contribution": maximum_source_label_contribution,
        "reject_reason_distribution": dict(sorted(reason_count.items())),
        "maximum_reason_code_fraction_of_rejects": maximum_reason_fraction,
        "reason_code_and_evidence_type_predictive_use_forbidden": True,
    }
    write_json(output / "shortcut_audit.json", result)
    write_jsonl(output / "shortcut_predictions.jsonl", predictions)
    return result


def shortcut_gate_config() -> dict[str, Any]:
    return {
        "schema": "actioncheck-shortcut-gate-config-v1",
        "gates": {
            "policy_source_only_BA_max": 0.53,
            "metadata_only_BA_max": 0.53,
            "action_only_BA_max": 0.55,
            "context_only_BA_max": 0.55,
            "action_summary_only_BA_max": 0.56,
            "combined_non_relational_BA_max": 0.60,
            "candidate_source_label_contribution_max": 0.80,
            "single_reason_code_reject_fraction_max": 0.40,
        },
        "reason_code_model_input": False,
        "evidence_type_model_input": False,
        "combined_control_interaction": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark_dir")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scheme", default="episode_held_out")
    args = parser.parse_args()
    result = audit_shortcuts(
        args.benchmark_dir, args.output_dir, scheme=args.scheme
    )
    print(json.dumps(result, indent=2, sort_keys=True))
