"""Lightweight fair baselines for an admitted ActionCheck pilot dataset."""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch import nn

from .common import read_jsonl, write_json, write_jsonl
from .evaluate_actioncheck import (
    best_balanced_threshold,
    classification_metrics,
    evaluate_prediction_rows,
)


BASELINES = (
    "action_only",
    "context_only",
    "state_action_TCN",
    "task_conditioned_action_Transformer",
    "visual_state_action_Transformer",
    "contrastive_context_action_encoder",
)


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.float().unsqueeze(2)
    return (value * weight).sum(1) / weight.sum(1).clamp_min(1.0)


class ActionOnly(nn.Module):
    def __init__(self, action_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(action_dim, 96),
            nn.GELU(),
            nn.Linear(96, 1),
        )

    def forward(self, context, action, task, context_mask, action_mask):
        return self.network(action.flatten(1)).squeeze(1)


class ContextOnly(nn.Module):
    def __init__(self, context_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(context_dim, 96),
            nn.GELU(),
            nn.Linear(96, 1),
        )

    def forward(self, context, action, task, context_mask, action_mask):
        return self.network(context.flatten(1)).squeeze(1)


class StateActionTCN(nn.Module):
    def __init__(self, context_width: int, action_width: int) -> None:
        super().__init__()
        self.context = nn.Sequential(
            nn.Conv1d(context_width, 48, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(48, 48, 3, padding=2, dilation=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.action = nn.Sequential(
            nn.Conv1d(action_width, 48, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(48, 48, 3, padding=2, dilation=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(nn.Linear(48 * 4, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, context, action, task, context_mask, action_mask):
        c = self.context(context.transpose(1, 2)).squeeze(2)
        a = self.action(action.transpose(1, 2)).squeeze(2)
        return self.head(torch.cat((c, a, c * a, torch.abs(c - a)), 1)).squeeze(1)


class TaskActionTransformer(nn.Module):
    def __init__(self, action_width: int, tasks: int) -> None:
        super().__init__()
        width = 64
        self.action = nn.Linear(action_width, width)
        self.task = nn.Embedding(tasks, width)
        layer = nn.TransformerEncoderLayer(
            width, 4, 128, dropout=0.0, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, 2)
        self.head = nn.Linear(width, 1)

    def forward(self, context, action, task, context_mask, action_mask):
        token = self.action(action) + self.task(task)[:, None]
        encoded = self.encoder(token, src_key_padding_mask=~action_mask)
        return self.head(_masked_mean(encoded, action_mask)).squeeze(1)


class VisualStateActionTransformer(nn.Module):
    def __init__(self, context_width: int, action_width: int) -> None:
        super().__init__()
        width = 64
        self.context = nn.Linear(context_width, width)
        self.action = nn.Linear(action_width, width)
        self.type_embedding = nn.Parameter(torch.randn(2, width) * 0.02)
        layer = nn.TransformerEncoderLayer(
            width, 4, 128, dropout=0.0, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, 2)
        self.head = nn.Sequential(nn.Linear(width, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, context, action, task, context_mask, action_mask):
        c = self.context(context) + self.type_embedding[0]
        a = self.action(action) + self.type_embedding[1]
        token = torch.cat((c, a), 1)
        mask = torch.cat((context_mask, action_mask), 1)
        encoded = self.encoder(token, src_key_padding_mask=~mask)
        return self.head(_masked_mean(encoded, mask)).squeeze(1)


class ContrastiveContextAction(nn.Module):
    def __init__(self, context_width: int, action_width: int) -> None:
        super().__init__()
        width = 64
        self.context = nn.Sequential(
            nn.Linear(context_width, 96), nn.GELU(), nn.Linear(96, width)
        )
        self.action = nn.Sequential(
            nn.Linear(action_width, 96), nn.GELU(), nn.Linear(96, width)
        )
        self.scale = nn.Parameter(torch.tensor(2.0))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, context, action, task, context_mask, action_mask):
        c = nn.functional.normalize(
            self.context(_masked_mean(context, context_mask)), dim=1
        )
        a = nn.functional.normalize(
            self.action(_masked_mean(action, action_mask)), dim=1
        )
        return self.scale.exp().clamp(max=100.0) * (c * a).sum(1) + self.bias


def _load(benchmark_dir: Path) -> dict[str, Any]:
    candidates = [
        row
        for row in read_jsonl(benchmark_dir / "candidate_manifest.jsonl")
        if row["primary_eligible"]
    ]
    candidates.sort(key=lambda row: int(row["primary_array_index"]))
    contexts = read_jsonl(benchmark_dir / "processed_contexts.jsonl")
    split = json.loads(
        (benchmark_dir / "split_manifest.json").read_text(encoding="utf-8")
    )
    with np.load(benchmark_dir / "benchmark_arrays.npz") as stored:
        arrays = {name: stored[name].copy() for name in stored.files}
    groups = arrays["group_index"].astype(np.int64)
    state = arrays["state_history"].astype(np.float32)
    visual = arrays["visual_history"].astype(np.float32)
    gripper = arrays["gripper_history"].astype(np.float32)
    state_present = (np.abs(state).sum((1, 2)) > 0).astype(np.float32)
    visual_present = (np.abs(visual).sum((1, 2)) > 0).astype(np.float32)
    flags = np.stack((state_present, visual_present), axis=1)
    flags = np.tile(flags[:, None], (1, state.shape[1], 1))
    context = np.concatenate((state, visual, gripper, flags), axis=2)
    context_mask = np.logical_or(
        arrays["state_history_mask"], arrays["visual_history_mask"]
    )
    action = arrays["candidate_actions"].astype(np.float32)
    action_mask = arrays["candidate_action_mask"].any(axis=2)
    return {
        "candidates": candidates,
        "contexts": contexts,
        "split_manifest": split,
        "context": context[groups],
        "context_mask": context_mask[groups],
        "action": action,
        "action_mask": action_mask,
        "task": arrays["task_index"].astype(np.int64),
        "labels": arrays["labels"].astype(np.int64),
        "task_count": len({row["task_id"] for row in candidates}),
    }


def _normalization(
    context: np.ndarray, action: np.ndarray, train: np.ndarray
) -> tuple[np.ndarray, np.ndarray, str]:
    context_mean = context[train].mean((0, 1), keepdims=True)
    context_std = context[train].std((0, 1), keepdims=True)
    context_std[context_std < 1e-5] = 1.0
    action_mean = action[train].mean((0, 1), keepdims=True)
    action_std = action[train].std((0, 1), keepdims=True)
    action_std[action_std < 1e-5] = 1.0
    normalized_context = np.nan_to_num((context - context_mean) / context_std).astype(
        np.float32
    )
    normalized_action = np.nan_to_num((action - action_mean) / action_std).astype(
        np.float32
    )
    digest = __import__("hashlib").sha256(
        context_mean.tobytes()
        + context_std.tobytes()
        + action_mean.tobytes()
        + action_std.tobytes()
    ).hexdigest()
    return normalized_context, normalized_action, digest


def _factory(
    name: str, context_width: int, action_width: int, action_steps: int, tasks: int
) -> nn.Module:
    if name == "action_only":
        return ActionOnly(action_steps * action_width)
    if name == "context_only":
        return ContextOnly(6 * context_width)
    if name == "state_action_TCN":
        return StateActionTCN(context_width, action_width)
    if name == "task_conditioned_action_Transformer":
        return TaskActionTransformer(action_width, tasks)
    if name == "visual_state_action_Transformer":
        return VisualStateActionTransformer(context_width, action_width)
    if name == "contrastive_context_action_encoder":
        return ContrastiveContextAction(context_width, action_width)
    raise KeyError(name)


def _train_one(
    name: str,
    data: dict[str, Any],
    split: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    train = split == "train"
    validation = split == "validation"
    context, action, normalization_hash = _normalization(
        data["context"], data["action"], train
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensors = {
        "context": torch.as_tensor(context, device=device),
        "action": torch.as_tensor(action, device=device),
        "task": torch.as_tensor(data["task"], device=device),
        "context_mask": torch.as_tensor(data["context_mask"], device=device),
        "action_mask": torch.as_tensor(data["action_mask"], device=device),
        "label": torch.as_tensor(data["labels"].astype(np.float32), device=device),
    }
    model = _factory(
        name,
        context.shape[2],
        action.shape[2],
        action.shape[1],
        data["task_count"],
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=3e-4)
    train_indices = np.flatnonzero(train)
    positive = float(data["labels"][train].sum())
    pos_weight = torch.tensor(
        [(len(train_indices) - positive) / max(positive, 1.0)], device=device
    )
    best_state = None
    best = -1.0
    patience = 0
    history = []
    rng = np.random.default_rng(seed)
    batch_size = 512
    for epoch in range(50):
        model.train()
        order = rng.permutation(train_indices)
        losses = []
        for start in range(0, len(order), batch_size):
            index = torch.as_tensor(order[start : start + batch_size], device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(
                tensors["context"][index],
                tensors["action"][index],
                tensors["task"][index],
                tensors["context_mask"][index],
                tensors["action_mask"][index],
            )
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits, tensors["label"][index], pos_weight=pos_weight
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            scores = []
            for start in range(0, len(data["labels"]), 1024):
                sl = slice(start, start + 1024)
                scores.append(
                    torch.sigmoid(
                        model(
                            tensors["context"][sl],
                            tensors["action"][sl],
                            tensors["task"][sl],
                            tensors["context_mask"][sl],
                            tensors["action_mask"][sl],
                        )
                    )
                    .cpu()
                    .numpy()
                )
            score = np.concatenate(scores)
        threshold = best_balanced_threshold(
            data["labels"][validation], score[validation]
        )
        metric = float(
            classification_metrics(
                data["labels"][validation], score[validation], threshold
            )["balanced_accuracy"]
        )
        history.append(
            {
                "epoch": epoch + 1,
                "loss": float(np.mean(losses)),
                "validation_balanced_accuracy": metric,
            }
        )
        if metric > best + 1e-4:
            best = metric
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            patience = 0
        else:
            patience += 1
        if patience >= 8:
            break
    if best_state is None:
        raise RuntimeError(f"{name} did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        scores = []
        for start in range(0, len(data["labels"]), 1024):
            sl = slice(start, start + 1024)
            scores.append(
                torch.sigmoid(
                    model(
                        tensors["context"][sl],
                        tensors["action"][sl],
                        tensors["task"][sl],
                        tensors["context_mask"][sl],
                        tensors["action_mask"][sl],
                    )
                )
                .cpu()
                .numpy()
            )
    score = np.concatenate(scores)
    threshold = best_balanced_threshold(
        data["labels"][validation], score[validation]
    )
    return score, {
        "seed": seed,
        "device": str(device),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "epochs": history[-1]["epoch"],
        "threshold_selected_on_validation": threshold,
        "best_validation_balanced_accuracy": best,
        "normalization_sha256": normalization_hash,
        "history": history,
    }


def run_pilot_baselines(
    benchmark_dir: str | Path,
    output_dir: str | Path,
    *,
    seed: int = 1701,
) -> dict[str, Any]:
    benchmark = Path(benchmark_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    data = _load(benchmark)
    all_predictions = []
    training_reports: dict[str, Any] = {}
    evaluations: dict[str, Any] = {}
    for scheme in ("episode_held_out", "task_held_out"):
        assignments = data["split_manifest"]["schemes"][scheme]["assignments"]
        split = np.asarray(
            [assignments[row["context_id"]] for row in data["candidates"]],
            dtype="<U10",
        )
        for model_name in BASELINES:
            score, training = _train_one(
                model_name, data, split, seed=seed
            )
            rows = []
            for candidate, label, value, owner in zip(
                data["candidates"], data["labels"], score, split
            ):
                if owner not in {"validation", "test"}:
                    continue
                row = {
                    "model": model_name,
                    "scheme": scheme,
                    "candidate_id": candidate["candidate_id"],
                    "context_id": candidate["context_id"],
                    "episode_id": candidate["episode_id"],
                    "task_id": candidate["task_id"],
                    "source_policy": candidate["source_policy"],
                    "evidence_type_audit_only": candidate[
                        "evidence_type_audit_only"
                    ],
                    "reason_codes_audit_only": candidate[
                        "reason_codes_audit_only"
                    ],
                    "label": int(label),
                    "score": float(value),
                    "split": str(owner),
                    "seed": seed,
                }
                rows.append(row)
                all_predictions.append(row)
            key = f"{scheme}/{model_name}"
            training_reports[key] = training
            evaluations[key] = evaluate_prediction_rows(rows)
            print(
                json.dumps(
                    {
                        "baseline": key,
                        "test_BA": evaluations[key]["classification"][
                            "balanced_accuracy"
                        ],
                        "Recall@1": evaluations[key]["grouped_ranking"][
                            "recall_at_1"
                        ],
                        "epochs": training["epochs"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    write_jsonl(output / "raw_baseline_predictions.jsonl", all_predictions)
    result = {
        "schema": "actioncheck-pilot-baseline-report-v1",
        "models": list(BASELINES),
        "schemes": ["episode_held_out", "task_held_out"],
        "seed": seed,
        "training": training_reports,
        "evaluation": evaluations,
        "raw_predictions": len(all_predictions),
        "phaseaware_temporal_energy_verifier_trained": False,
    }
    write_json(output / "pilot_baseline_results.json", result)
    return result
