"""Shared contracts for the independently preregistered RM-ACV-PILOT166.

The module contains deterministic source/split utilities, metrics, and a small
PyTorch classifier used by the shortcut audit.  Provenance identifiers are
kept in manifests for leakage checks but are never assembled into fair model
inputs.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch import nn

from actmask.data.robomind_adapter import RoboMINDRecord


ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path("/data/projects/tzh/RoboMIND2")
OUT = ROOT / "outputs" / "actmask" / "rm_acv_pilot166"
PYTHON = Path("/home/tzh/conda_envs/actmask/bin/python")
PRIMARY_EMBODIMENT = "franka_fr3_dualArm-gripper-6cameras_4"
EXCLUDED_EMBODIMENT = "franka_fr3_dualArm-gripper-6cameras_2"
HISTORY = 8
ACTION_HORIZON = 16
MAX_ANCHORS = 6
MIN_ANCHOR_SPACING = 16
GROUPED_NEGATIVES = 3
SEEDS = (17, 29, 43)
TASK_CV_FOLDS = 5

FAIR_INPUTS = (
    "causal_rgb_history",
    "causal_proprioceptive_history",
    "task_instruction",
    "candidate_action_chunk",
)
FORBIDDEN_PREDICTIVE_INPUTS = (
    "future_rgb",
    "future_state",
    "SAM3",
    "robot_masks",
    "episode_id",
    "source_path",
    "file_ordering",
    "negative_family_identity",
)
NEGATIVE_FAMILIES = (
    "N1_same_task_near_state",
    "N2_local_temporal_permutation",
    "N3_arm_gripper_desynchronization",
    "N4_phase_shifted_same_task",
    "N5_joint_coordination_corruption",
    "N6_endpoint_matched_path_corruption",
)


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=json_default) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, default=json_default) + "\n"
            )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=json_default).encode()
    ).hexdigest()


def stable_int(value: str, modulo: int | None = None) -> int:
    number = int(hashlib.sha256(value.encode()).hexdigest()[:16], 16)
    return number % modulo if modulo else number


def append_log(event: str, **payload: Any) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "run_log.jsonl"
    previous = ""
    seq = 0
    if path.is_file() and path.stat().st_size:
        last = json.loads(path.read_text(encoding="utf-8").splitlines()[-1])
        previous = str(last["event_sha256"])
        seq = int(last["seq"])
    row = {
        "seq": seq + 1,
        "utc": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "payload": payload,
        "previous_event_sha256": previous,
    }
    row["event_sha256"] = sha256_json(row)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=json_default) + "\n")


def _episode_number(path: Path) -> int:
    return int(path.stem.removeprefix("episode_"))


def scan_records() -> tuple[list[RoboMINDRecord], list[RoboMINDRecord]]:
    """Resolve the exact primary and excluded embodiments from official metadata."""

    primary: list[RoboMINDRecord] = []
    excluded: list[RoboMINDRecord] = []
    franka = SOURCE / "camera_extrinsics_intrinsics" / "franka"
    for info_path in sorted(franka.rglob("meta/info.json")):
        task_root = info_path.parent.parent
        info = json.loads(info_path.read_text(encoding="utf-8"))
        robot_type = str(info.get("robot_type", "unknown"))
        episodes_path = task_root / "meta" / "episodes.jsonl"
        if not episodes_path.is_file():
            continue
        episode_rows = {
            int(row["episode_index"]): row
            for row in (json.loads(line) for line in episodes_path.read_text(encoding="utf-8").splitlines() if line)
        }
        for parquet in sorted((task_root / "data").glob("chunk-*/*.parquet")):
            episode_index = _episode_number(parquet)
            metadata = episode_rows.get(episode_index, {})
            tasks = metadata.get("tasks") or []
            front = (
                task_root
                / "videos"
                / "chunk-000"
                / "observation.rgb_images.camera_front"
                / f"{parquet.stem}.mp4"
            )
            record = RoboMINDRecord(
                episode_id=f"{task_root.name}:{episode_index:06d}",
                task_id=task_root.name,
                robot_embodiment=robot_type,
                language_instruction=str(tasks[0]) if tasks else "",
                parquet_path=str(parquet),
                front_video_path=str(front) if front.is_file() else None,
                source_bytes=int(parquet.stat().st_size),
                episode_index=episode_index,
                metadata_path=str(episodes_path),
            )
            if robot_type == PRIMARY_EMBODIMENT:
                primary.append(record)
            elif robot_type == EXCLUDED_EMBODIMENT:
                excluded.append(record)
    primary.sort(key=lambda row: row.episode_id)
    excluded.sort(key=lambda row: row.episode_id)
    if len(primary) != 166 or len({row.task_id for row in primary}) != 11:
        raise RuntimeError(
            f"primary source drift: expected 166 trajectories/11 tasks, got "
            f"{len(primary)}/{len({row.task_id for row in primary})}"
        )
    if len(excluded) != 72:
        raise RuntimeError(f"excluded 16D source drift: expected 72 trajectories, got {len(excluded)}")
    if any(not row.language_instruction or not row.front_video_path for row in primary):
        raise RuntimeError("primary source has missing task language or front RGB")
    return primary, excluded


def episode_split(records: Sequence[RoboMINDRecord]) -> dict[str, list[str]]:
    """Deterministic within-task split with at least two validation/test episodes."""

    by_task: dict[str, list[RoboMINDRecord]] = defaultdict(list)
    for record in records:
        by_task[record.task_id].append(record)
    split = {"train": [], "validation": [], "test": []}
    for task, group in sorted(by_task.items()):
        ordered = sorted(group, key=lambda row: (stable_int(row.episode_id), row.episode_id))
        n = len(ordered)
        n_validation = max(2, int(round(0.15 * n)))
        n_test = max(2, int(round(0.15 * n)))
        if n - n_validation - n_test < 6:
            raise RuntimeError(f"task {task} cannot retain six training episodes")
        split["test"].extend(row.episode_id for row in ordered[:n_test])
        split["validation"].extend(row.episode_id for row in ordered[n_test : n_test + n_validation])
        split["train"].extend(row.episode_id for row in ordered[n_test + n_validation :])
    return {name: sorted(values) for name, values in split.items()}


def task_cv_folds(tasks: Sequence[str]) -> list[dict[str, list[str]]]:
    ordered = sorted(set(tasks), key=lambda task: (stable_int(task), task))
    buckets: list[list[str]] = [[] for _ in range(TASK_CV_FOLDS)]
    for index, task in enumerate(ordered):
        buckets[index % TASK_CV_FOLDS].append(task)
    folds: list[dict[str, list[str]]] = []
    for test_index in range(TASK_CV_FOLDS):
        validation_index = (test_index + 1) % TASK_CV_FOLDS
        folds.append(
            {
                "fold": [f"fold_{test_index}"],
                "train_tasks": sorted(
                    task for index, bucket in enumerate(buckets)
                    if index not in {test_index, validation_index} for task in bucket
                ),
                "validation_tasks": sorted(buckets[validation_index]),
                "test_tasks": sorted(buckets[test_index]),
            }
        )
    held_out = [task for fold in folds for task in fold["test_tasks"]]
    if sorted(held_out) != sorted(set(tasks)):
        raise RuntimeError("task CV does not hold out every task exactly once")
    return folds


def anchor_steps(length: int) -> list[int]:
    first = HISTORY - 1
    last = length - ACTION_HORIZON - 1
    if last < first:
        return []
    capacity = 1 + (last - first) // MIN_ANCHOR_SPACING
    count = min(MAX_ANCHORS, capacity)
    if count <= 0:
        return []
    if count == 1:
        return [first]
    anchors = sorted({int(round(value)) for value in np.linspace(first, last, count)})
    if any(right - left < MIN_ANCHOR_SPACING for left, right in zip(anchors, anchors[1:])):
        raise RuntimeError(f"anchor spacing construction failed for length {length}: {anchors}")
    return anchors


def instruction_features(text: str, width: int = 128) -> np.ndarray:
    """Stable signed hashing of instruction word and character n-grams."""

    result = np.zeros(width, dtype=np.float32)
    normalized = " ".join(text.lower().replace("_", " ").split())
    tokens = normalized.split()
    grams = tokens + [" ".join(tokens[i : i + 2]) for i in range(max(0, len(tokens) - 1))]
    grams += [normalized[i : i + 3] for i in range(max(0, len(normalized) - 2))]
    for gram in grams:
        digest = hashlib.sha256(gram.encode()).digest()
        index = int.from_bytes(digest[:4], "little") % width
        sign = 1.0 if digest[4] & 1 else -1.0
        result[index] += sign
    norm = float(np.linalg.norm(result))
    return result / norm if norm > 0 else result


def action_summary(action: np.ndarray) -> dict[str, np.ndarray]:
    """Strong non-context action statistics for shortcut auditing."""

    if action.ndim != 3:
        raise ValueError(f"expected [N,H,D] action, got {action.shape}")
    velocity = np.diff(action, axis=1)
    acceleration = np.diff(velocity, axis=1)
    jerk = np.diff(acceleration, axis=1)
    norm = np.linalg.norm(action, axis=2)
    smooth = np.linalg.norm(velocity, axis=2)
    accel = np.linalg.norm(acceleration, axis=2)
    jerk_norm = np.linalg.norm(jerk, axis=2)
    displacement = action[:, -1] - action[:, 0]
    groups = {
        "action_magnitude_only": np.stack([norm.mean(1), norm.std(1), np.linalg.norm(action, axis=(1, 2))], axis=1),
        "action_mean_variance": np.concatenate([action.mean(1), action.var(1)], axis=1),
        "smoothness_jerk": np.stack(
            [smooth.mean(1), smooth.std(1), accel.mean(1), jerk_norm.mean(1)], axis=1
        ),
        "start_end_action": np.concatenate([action[:, 0], action[:, -1]], axis=1),
        "total_displacement": np.concatenate(
            [displacement, np.linalg.norm(displacement, axis=1, keepdims=True)], axis=1
        ),
    }
    groups["action_summary_combined"] = np.concatenate(list(groups.values()), axis=1)
    return {key: value.astype(np.float32) for key, value in groups.items()}


def _auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    if not len(positive) or not len(negative):
        return float("nan")
    total = 0.0
    block = 512
    for start in range(0, len(positive), block):
        comparison = positive[start : start + block, None] - negative[None, :]
        total += float((comparison > 0).sum() + 0.5 * (comparison == 0).sum())
    return total / (len(positive) * len(negative))


def _average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(-scores, kind="stable")
    ranked = labels[order]
    positives = int(ranked.sum())
    if not positives:
        return float("nan")
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float((precision * ranked).sum() / positives)


def binary_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> dict[str, float | int]:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    prediction = scores >= threshold
    positive = labels == 1
    negative = labels == 0
    tpr = float(prediction[positive].mean()) if positive.any() else float("nan")
    tnr = float((~prediction[negative]).mean()) if negative.any() else float("nan")
    bins = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    for left, right in zip(bins[:-1], bins[1:]):
        mask = (scores >= left) & (scores < right if right < 1.0 else scores <= right)
        if mask.any():
            ece += float(mask.mean()) * abs(float(scores[mask].mean()) - float(labels[mask].mean()))
    return {
        "balanced_accuracy": (tpr + tnr) / 2.0,
        "auroc": _auc(labels, scores),
        "auprc": _average_precision(labels, scores),
        "false_accept_rate": 1.0 - tnr,
        "false_reject_rate": 1.0 - tpr,
        "brier": float(np.mean((scores - labels) ** 2)),
        "ece_10_bin": ece,
        "threshold": float(threshold),
        "n": int(len(labels)),
        "positive_n": int(positive.sum()),
        "negative_n": int(negative.sum()),
    }


def best_balanced_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    candidates = np.unique(np.concatenate(([0.0], np.asarray(scores), [1.0])))
    if len(candidates) > 501:
        candidates = np.quantile(candidates, np.linspace(0, 1, 501))
    best = (float("-inf"), 0.5)
    for threshold in candidates:
        metric = float(binary_metrics(labels, scores, float(threshold))["balanced_accuracy"])
        if metric > best[0] + 1e-12 or (abs(metric - best[0]) <= 1e-12 and abs(threshold - 0.5) < abs(best[1] - 0.5)):
            best = (metric, float(threshold))
    return best[1]


def ranking_metrics(
    group_ids: Sequence[str],
    labels: np.ndarray,
    scores: np.ndarray,
) -> dict[str, float | int]:
    groups: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for group, label, score in zip(group_ids, labels, scores):
        groups[str(group)].append((int(label), float(score)))
    recalls: list[float] = []
    reciprocal: list[float] = []
    ndcg: list[float] = []
    margins: list[float] = []
    candidate_counts: list[int] = []
    for rows in groups.values():
        if sum(label for label, _ in rows) != 1:
            continue
        ordered = sorted(rows, key=lambda item: item[1], reverse=True)
        rank = next(index + 1 for index, (label, _) in enumerate(ordered) if label == 1)
        positive_score = next(score for label, score in rows if label == 1)
        negative_scores = [score for label, score in rows if label == 0]
        recalls.append(float(rank == 1))
        reciprocal.append(1.0 / rank)
        ndcg.append(1.0 / math.log2(rank + 1))
        margins.append(positive_score - max(negative_scores))
        candidate_counts.append(len(rows))
    chance = float(np.mean([1.0 / count for count in candidate_counts])) if candidate_counts else float("nan")
    return {
        "recall_at_1": float(np.mean(recalls)) if recalls else float("nan"),
        "mrr": float(np.mean(reciprocal)) if reciprocal else float("nan"),
        "ndcg": float(np.mean(ndcg)) if ndcg else float("nan"),
        "positive_negative_score_margin": float(np.mean(margins)) if margins else float("nan"),
        "groups": len(recalls),
        "chance_recall_at_1": chance,
    }


def grouped_bootstrap_balanced_accuracy(
    labels: np.ndarray,
    scores: np.ndarray,
    groups: Sequence[str],
    threshold: float,
    *,
    seed: int = 20260728,
    replicates: int = 1000,
) -> dict[str, float | int]:
    unique = sorted(set(str(group) for group in groups))
    indices = {group: np.asarray([i for i, value in enumerate(groups) if str(value) == group]) for group in unique}
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(replicates):
        sample = rng.choice(unique, size=len(unique), replace=True)
        selected = np.concatenate([indices[group] for group in sample])
        values.append(float(binary_metrics(labels[selected], scores[selected], threshold)["balanced_accuracy"]))
    low, high = np.quantile(values, [0.025, 0.975])
    return {
        "grouping": "episode",
        "groups": len(unique),
        "replicates": replicates,
        "balanced_accuracy_ci95_low": float(low),
        "balanced_accuracy_ci95_high": float(high),
    }


class ShortcutClassifier(nn.Module):
    def __init__(self, width: int, hidden: int) -> None:
        super().__init__()
        if hidden:
            self.network = nn.Sequential(
                nn.Linear(width, hidden),
                nn.GELU(),
                nn.LayerNorm(hidden),
                nn.Dropout(0.1),
                nn.Linear(hidden, hidden // 2),
                nn.GELU(),
                nn.Linear(hidden // 2, 1),
            )
        else:
            self.network = nn.Linear(width, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value).squeeze(1)


def train_shortcut_classifier(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    x_test: np.ndarray,
    *,
    seed: int,
    nonlinear: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], dict[str, np.ndarray]]:
    """Train a frozen linear/MLP classifier and select threshold on validation."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    mean = x_train.mean(0, keepdims=True)
    std = x_train.std(0, keepdims=True)
    std[std < 1e-6] = 1.0
    standardized = [
        np.nan_to_num((value - mean) / std, nan=0.0, posinf=8.0, neginf=-8.0).astype(np.float32)
        for value in (x_train, x_validation, x_test)
    ]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_x = torch.as_tensor(standardized[0], device=device)
    train_y = torch.as_tensor(y_train.astype(np.float32), device=device)
    validation_x = torch.as_tensor(standardized[1], device=device)
    test_x = torch.as_tensor(standardized[2], device=device)
    hidden = min(128, max(32, x_train.shape[1] // 2)) if nonlinear else 0
    model = ShortcutClassifier(x_train.shape[1], hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.006 if nonlinear else 0.02, weight_decay=2e-4)
    positives = float(y_train.sum())
    pos_weight = torch.tensor([(len(y_train) - positives) / max(positives, 1.0)], device=device)
    best_state: dict[str, torch.Tensor] | None = None
    best_validation = float("-inf")
    patience = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(250):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(train_x)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, train_y, pos_weight=pos_weight)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        if epoch % 5 == 4:
            model.eval()
            with torch.no_grad():
                validation_scores = torch.sigmoid(model(validation_x)).cpu().numpy()
            threshold = best_balanced_threshold(y_validation, validation_scores)
            score = float(binary_metrics(y_validation, validation_scores, threshold)["balanced_accuracy"])
            history.append({"epoch": epoch + 1, "loss": float(loss.detach().cpu()), "validation_balanced_accuracy": score})
            if score > best_validation + 1e-5:
                best_validation = score
                best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
            if patience >= 12:
                break
    if best_state is None:
        raise RuntimeError("shortcut classifier failed to produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        validation_scores = torch.sigmoid(model(validation_x)).cpu().numpy()
        test_scores = torch.sigmoid(model(test_x)).cpu().numpy()
    threshold = best_balanced_threshold(y_validation, validation_scores)
    info = {
        "seed": seed,
        "device": str(device),
        "nonlinear": nonlinear,
        "hidden": hidden,
        "epochs": history[-1]["epoch"],
        "threshold_selected_on_validation": threshold,
        "best_validation_balanced_accuracy": best_validation,
        "history": history,
    }
    normalization = {"mean": mean.squeeze(0).astype(np.float32), "std": std.squeeze(0).astype(np.float32)}
    return validation_scores, test_scores, info, normalization


def environment_manifest() -> dict[str, Any]:
    return {
        "python": str(PYTHON),
        "python_version": os.sys.version,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


__all__ = [
    "ACTION_HORIZON",
    "EXCLUDED_EMBODIMENT",
    "FAIR_INPUTS",
    "FORBIDDEN_PREDICTIVE_INPUTS",
    "GROUPED_NEGATIVES",
    "HISTORY",
    "MAX_ANCHORS",
    "MIN_ANCHOR_SPACING",
    "NEGATIVE_FAMILIES",
    "OUT",
    "PRIMARY_EMBODIMENT",
    "PYTHON",
    "ROOT",
    "SEEDS",
    "SOURCE",
    "TASK_CV_FOLDS",
    "action_summary",
    "anchor_steps",
    "append_log",
    "best_balanced_threshold",
    "binary_metrics",
    "environment_manifest",
    "episode_split",
    "grouped_bootstrap_balanced_accuracy",
    "instruction_features",
    "json_default",
    "ranking_metrics",
    "read_jsonl",
    "scan_records",
    "sha256_file",
    "sha256_json",
    "stable_int",
    "task_cv_folds",
    "train_shortcut_classifier",
    "write_json",
    "write_jsonl",
]
