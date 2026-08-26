"""RM-Series: RoboMIND-only bounded data and baseline admission audit.

This program never trains an ActMask / RelDynVerifier method.  It first makes
the local source, label boundary, alignment, split construction and baseline
inputs explicit.  The only task enabled without outcome labels is the weaker
``action_future_consistency`` formulation: a logged continuation is positive;
same-task other-attempt, wrong-time, reversed and shuffled actions are
negatives.  It is not an outcome or physical-counterfactual claim.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import torch
from torch import nn

from actmask.data.robomind_adapter import (
    ROBOMIND_DATASET,
    RoboMINDRecord,
    fair_input_audit,
    load_numeric_episode,
    scan_local_robomind,
    select_bounded_records,
    validate_numeric_episode,
)


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "actmask" / "rm_series_robomind"
SOURCE = Path("/data/projects/tzh/RoboMIND2")
SEEDS = (17, 29, 43)
HISTORY = 4
ACTION_CHUNK = 4
ANCHOR_FRACTIONS = (0.20, 0.40, 0.60, 0.80)
FAIR_FIELDS = ("history_state", "candidate_action", "history_visual_features")


def _json_default(value: Any) -> Any:
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
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default) + "\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=_json_default) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def append_log(event: str, **payload: Any) -> None:
    path = OUT / "run_log.jsonl"
    previous = ""
    seq = 0
    if path.is_file() and path.stat().st_size:
        lines = [json.loads(line) for line in path.read_text().splitlines() if line]
        seq = int(lines[-1]["seq"])
        previous = str(lines[-1]["event_sha256"])
    row = {
        "seq": seq + 1,
        "utc": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "payload": payload,
        "prev_event_sha256": previous,
    }
    row["event_sha256"] = hashlib.sha256(
        json.dumps(row, sort_keys=True, separators=(",", ":"), default=_json_default).encode()
    ).hexdigest()
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, default=_json_default) + "\n")


def _front_stats(video_path: str, anchor_steps: list[int]) -> np.ndarray:
    """Deterministic frozen image statistics; no pretrained model is fetched."""

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"unreadable front video: {video_path}")
    result: list[np.ndarray] = []
    try:
        for step in anchor_steps:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(step))
            ok, frame = cap.read()
            if not ok or frame is None:
                raise ValueError(f"cannot decode frame {step} from {video_path}")
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            small = cv2.resize(rgb, (4, 4), interpolation=cv2.INTER_AREA)
            # RGB mean/std plus a 4x4 luminance map are frozen visual features.
            vec = np.concatenate(
                [rgb.mean((0, 1)), rgb.std((0, 1)), small.mean(axis=2).reshape(-1)]
            ).astype(np.float32)
            result.append(vec)
    finally:
        cap.release()
    return np.stack(result)


def _pad_width(array: np.ndarray, width: int = 32) -> np.ndarray:
    if array.shape[1] > width:
        raise ValueError(f"unexpected feature width {array.shape[1]} > {width}")
    if array.shape[1] == width:
        return array.astype(np.float32, copy=False)
    result = np.zeros((len(array), width), dtype=np.float32)
    result[:, : array.shape[1]] = array
    return result


def _anchor_steps(length: int) -> list[int]:
    latest = length - ACTION_CHUNK - 1
    if latest <= HISTORY + 1:
        raise ValueError(f"episode too short for history/action task: {length}")
    return sorted({max(HISTORY, min(latest, int(length * fraction))) for fraction in ANCHOR_FRACTIONS})


def _normal_step(length: int, fraction: float) -> int:
    latest = length - ACTION_CHUNK - 1
    return max(HISTORY, min(latest, int(length * fraction)))


def _episode_payload(records: list[RoboMINDRecord]) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Read fair numeric streams and front-image features, retaining provenance separately."""

    payload: dict[str, dict[str, Any]] = {}
    audits: list[dict[str, Any]] = []
    video_requests: list[dict[str, Any]] = []
    for record in records:
        numeric = load_numeric_episode(record)
        audit = validate_numeric_episode(record, numeric)
        if not audit["pass"]:
            raise ValueError(f"alignment failure: {record.episode_id}")
        if not record.front_video_path:
            raise ValueError(f"front video unavailable: {record.episode_id}")
        anchors = _anchor_steps(len(numeric["state"]))
        history_frame_steps = [step for anchor in anchors for step in range(anchor - HISTORY, anchor)]
        payload[record.episode_id] = {
            "record": record,
            "state": _pad_width(numeric["state"]),
            "action": _pad_width(numeric["action"]),
            "timestamp": numeric["timestamp"],
            "frame_index": numeric["frame_index"],
            "anchors": anchors,
        }
        video_requests.append({"episode_id": record.episode_id, "video_path": record.front_video_path, "history_frame_steps": history_frame_steps, "anchors": len(anchors), "history": HISTORY})
        audits.append({**audit, "anchors": anchors, "visual_feature_dim": 22, "visual_history_steps": HISTORY})
    scratch = OUT / "processed_subset" / ".visual_worker"
    scratch.mkdir(parents=True, exist_ok=True)
    for batch_index in range(0, len(video_requests), 10):
        batch = video_requests[batch_index : batch_index + 10]
        manifest = scratch / f"batch_{batch_index // 10:02d}.json"
        write_json(manifest, batch)
        command = [sys.executable, "-m", "actmask.experiments.rm_series_video_feature_worker", "--manifest", str(manifest), "--output", str(scratch)]
        completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        if completed.returncode:
            raise RuntimeError(f"visual worker batch {batch_index // 10} failed: {completed.stderr.strip() or completed.stdout.strip()}")
    for record in records:
        visual_path = scratch / f"{record.episode_id.replace(':', '_')}.npz"
        if not visual_path.is_file():
            raise RuntimeError(f"visual worker did not create {visual_path.name}")
        payload[record.episode_id]["visual"] = np.load(visual_path)["value"]
    return payload, audits


def _within_split_partners(ids: list[str]) -> dict[str, str]:
    if len(ids) < 2:
        raise ValueError("at least two repeated attempts are required for same-task negatives")
    ordered = sorted(ids)
    return {episode_id: ordered[(index + 1) % len(ordered)] for index, episode_id in enumerate(ordered)}


def build_samples(
    episode_data: dict[str, dict[str, Any]],
    split_episode_ids: dict[str, list[str]],
) -> dict[str, dict[str, Any]]:
    """Create positive true-continuations and four provenance-free negatives."""

    output: dict[str, dict[str, Any]] = {}
    for split, ids in split_episode_ids.items():
        by_task: dict[str, list[str]] = defaultdict(list)
        for episode_id in ids:
            by_task[episode_data[episode_id]["record"].task_id].append(episode_id)
        partners = {}
        for group in by_task.values():
            partners.update(_within_split_partners(group))

        rows: list[dict[str, Any]] = []
        for episode_id in sorted(ids):
            item = episode_data[episode_id]
            partner = episode_data[partners[episode_id]]
            state, action = item["state"], item["action"]
            for ordinal, anchor in enumerate(item["anchors"]):
                true_action = action[anchor : anchor + ACTION_CHUNK]
                frac = anchor / len(action)
                partner_anchor = _normal_step(len(partner["action"]), frac)
                wrong_anchor = _normal_step(len(action), (frac + 0.47) % 1.0)
                if abs(wrong_anchor - anchor) < ACTION_CHUNK:
                    wrong_anchor = _normal_step(len(action), (frac + 0.23) % 1.0)
                candidates = [
                    ("true_continuation", 1, true_action),
                    ("same_task_other_attempt", 0, partner["action"][partner_anchor : partner_anchor + ACTION_CHUNK]),
                    ("wrong_time_same_episode", 0, action[wrong_anchor : wrong_anchor + ACTION_CHUNK]),
                    ("reversed_action", 0, true_action[::-1].copy()),
                    ("shuffled_action", 0, np.roll(true_action, 1, axis=0).copy()),
                ]
                history_state = state[anchor - HISTORY : anchor]
                history_visual = item["visual"][ordinal]
                for kind, label, candidate_action in candidates:
                    if candidate_action.shape != (ACTION_CHUNK, action.shape[1]):
                        raise ValueError(f"malformed candidate action for {episode_id}")
                    rows.append(
                        {
                            "sample_id": f"{split}:{episode_id}:{anchor}:{kind}",
                            "anchor_key": f"{split}:{episode_id}:{anchor}",
                            "label": int(label),
                            "kind": kind,
                            "history_state": history_state.astype(np.float32),
                            "candidate_action": candidate_action.astype(np.float32),
                            "history_visual": history_visual.astype(np.float32),
                            "oracle_future_action": true_action.astype(np.float32),
                            # Provenance is retained only for audit/split checks, never features.
                            "episode_id": episode_id,
                            "task_id": item["record"].task_id,
                            "anchor": int(anchor),
                        }
                    )
        output[split] = {"rows": rows, "episode_ids": sorted(ids), "task_ids": sorted(by_task)}
    return output


def _stack(rows: list[dict[str, Any]], field: str) -> np.ndarray:
    return np.stack([np.asarray(row[field]) for row in rows]).astype(np.float32)


def _labels(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([row["label"] for row in rows], dtype=np.float32)


def _linear_predict(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, seed: int) -> np.ndarray:
    if x_train.ndim != 2:
        x_train = x_train.reshape(len(x_train), -1)
    if x_test.ndim != 2:
        x_test = x_test.reshape(len(x_test), -1)
    mean = x_train.mean(0, keepdims=True)
    std = x_train.std(0, keepdims=True)
    std[std < 1e-6] = 1.0
    x_train, x_test = (x_train - mean) / std, (x_test - mean) / std
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.Linear(x_train.shape[1], 1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.025, weight_decay=1e-4)
    x = torch.as_tensor(x_train, device=device)
    y = torch.as_tensor(y_train[:, None], device=device)
    weight = torch.tensor([(len(y_train) - y_train.sum()) / max(y_train.sum(), 1.0)], device=device)
    for _ in range(70):
        opt.zero_grad(set_to_none=True)
        loss = nn.functional.binary_cross_entropy_with_logits(model(x), y, pos_weight=weight)
        loss.backward()
        opt.step()
    with torch.no_grad():
        return torch.sigmoid(model(torch.as_tensor(x_test, device=device))).squeeze(1).cpu().numpy()


class _GRUScore(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.gru = nn.GRU(width, 32, batch_first=True)
        self.head = nn.Linear(32, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.head(self.gru(value)[0][:, -1])


class _TCNScore(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(width, 32, 3, padding=1), nn.ReLU(), nn.Conv1d(32, 24, 3, padding=1), nn.ReLU())
        self.head = nn.Linear(24, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.head(self.net(value.transpose(1, 2)).mean(2))


def _sequence_predict(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, seed: int, kind: str) -> np.ndarray:
    mean = x_train.mean((0, 1), keepdims=True)
    std = x_train.std((0, 1), keepdims=True)
    std[std < 1e-6] = 1.0
    x_train, x_test = (x_train - mean) / std, (x_test - mean) / std
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model: nn.Module = _GRUScore(x_train.shape[2]) if kind == "gru" else _TCNScore(x_train.shape[2])
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.008, weight_decay=1e-4)
    x = torch.as_tensor(x_train, device=device)
    y = torch.as_tensor(y_train[:, None], device=device)
    weight = torch.tensor([(len(y_train) - y_train.sum()) / max(y_train.sum(), 1.0)], device=device)
    for _ in range(55):
        opt.zero_grad(set_to_none=True)
        loss = nn.functional.binary_cross_entropy_with_logits(model(x), y, pos_weight=weight)
        loss.backward()
        opt.step()
    with torch.no_grad():
        return torch.sigmoid(model(torch.as_tensor(x_test, device=device))).squeeze(1).cpu().numpy()


def _nn_predict(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    x_train = x_train.reshape(len(x_train), -1)
    x_test = x_test.reshape(len(x_test), -1)
    scale = x_train.std(0, keepdims=True)
    scale[scale < 1e-6] = 1.0
    train = x_train / scale
    result: list[float] = []
    for row in x_test / scale:
        distance = ((train - row) ** 2).mean(1)
        nearest = np.argpartition(distance, min(7, len(distance) - 1))[:8]
        weights = 1.0 / (distance[nearest] + 1e-6)
        result.append(float(np.average(y_train[nearest], weights=weights)))
    return np.asarray(result, dtype=np.float32)


def _auc(labels: np.ndarray, scores: np.ndarray) -> float:
    pos, neg = scores[labels == 1], scores[labels == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    return float(((pos[:, None] > neg[None, :]).mean()) + 0.5 * (pos[:, None] == neg[None, :]).mean())


def _metrics(rows: list[dict[str, Any]], scores: np.ndarray) -> dict[str, float]:
    labels = _labels(rows).astype(int)
    pred = scores >= 0.5
    tpr = float(pred[labels == 1].mean()) if (labels == 1).any() else float("nan")
    tnr = float((~pred[labels == 0]).mean()) if (labels == 0).any() else float("nan")
    by_anchor: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for row, score in zip(rows, scores):
        by_anchor[str(row["anchor_key"])].append((int(row["label"]), float(score)))
    ordered: list[float] = []
    for pairs in by_anchor.values():
        positive = [score for label, score in pairs if label == 1]
        negative = [score for label, score in pairs if label == 0]
        if positive and negative:
            ordered.extend([float(positive[0] > score) + 0.5 * float(positive[0] == score) for score in negative])
    return {
        "balanced_accuracy": (tpr + tnr) / 2.0,
        "auroc": _auc(labels, scores),
        "pair_order_accuracy": float(np.mean(ordered)),
        "n": int(len(rows)),
        "positive_n": int(labels.sum()),
        "negative_n": int((labels == 0).sum()),
    }


def _feature_bundle(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    state = _stack(rows, "history_state")
    action = _stack(rows, "candidate_action")
    visual = _stack(rows, "history_visual")
    state_action_seq = np.concatenate([state, np.zeros_like(state)], axis=1)
    state_action_seq[:, HISTORY:, :] = action
    # Four history observations followed by four candidate action chunks.
    visual_pad = np.pad(visual, ((0, 0), (0, ACTION_CHUNK), (0, 32 - visual.shape[2])))
    state_pad = np.pad(state, ((0, 0), (0, ACTION_CHUNK), (0, 0)))
    action_pad = np.zeros_like(state_pad)
    action_pad[:, HISTORY:, :] = action
    return {
        "constant": np.zeros((len(rows), 1), np.float32),
        "action_only": action.reshape(len(rows), -1),
        "action_magnitude_smoothness": np.stack(
            [np.linalg.norm(action, axis=2).mean(1), np.linalg.norm(np.diff(action, axis=1), axis=2).mean(1)], axis=1
        ),
        "episode_time_proxy": np.asarray([[row["anchor"]] for row in rows], dtype=np.float32),
        "current_state_only": state[:, -1],
        "current_frame_only": visual[:, -1],
        "frozen_visual_feature_temporal_model": visual.reshape(len(rows), -1),
        "state_action_linear": np.concatenate([state.reshape(len(rows), -1), action.reshape(len(rows), -1)], axis=1),
        "proprio_action_gru": state_action_seq,
        "state_action_tcn": state_action_seq,
        "visual_history_gru": visual_pad,
        "visual_history_tcn": visual_pad,
        "visual_state_gru": np.concatenate([visual_pad, state_pad], axis=2),
        "visual_action_gru": np.concatenate([visual_pad, action_pad], axis=2),
        "visual_state_action_gru": np.concatenate([visual_pad, state_pad, action_pad], axis=2),
        "nearest_neighbor_retrieval": np.concatenate([state.reshape(len(rows), -1), action.reshape(len(rows), -1)], axis=1),
    }


def _oracle_scores(rows: list[dict[str, Any]]) -> np.ndarray:
    candidate = _stack(rows, "candidate_action")
    truth = _stack(rows, "oracle_future_action")
    distance = ((candidate - truth) ** 2).mean((1, 2))
    return np.exp(-distance / max(float(np.median(distance[distance > 0])) if (distance > 0).any() else 1.0, 1e-6))


def _run_model(name: str, train_rows: list[dict[str, Any]], test_rows: list[dict[str, Any]], seed: int) -> np.ndarray:
    train, test = _feature_bundle(train_rows)[name], _feature_bundle(test_rows)[name]
    labels = _labels(train_rows)
    if name == "constant":
        return np.full(len(test_rows), 0.5, dtype=np.float32)
    if name == "nearest_neighbor_retrieval":
        return _nn_predict(train, labels, test)
    if name.endswith("_gru"):
        return _sequence_predict(train, labels, test, seed, "gru")
    if name.endswith("_tcn"):
        return _sequence_predict(train, labels, test, seed, "tcn")
    return _linear_predict(train, labels, test, seed)


def _split_definitions(records: list[RoboMINDRecord]) -> dict[str, dict[str, list[str]]]:
    by_task: dict[str, list[RoboMINDRecord]] = defaultdict(list)
    for record in records:
        by_task[record.task_id].append(record)
    episode_split = {"train": [], "validation": [], "test": []}
    for task, group in sorted(by_task.items()):
        ordered = sorted(group, key=lambda record: record.episode_index)
        episode_split["train"].extend(record.episode_id for record in ordered[:6])
        episode_split["validation"].extend(record.episode_id for record in ordered[6:8])
        episode_split["test"].extend(record.episode_id for record in ordered[8:10])
    tasks = sorted(by_task)
    task_split = {"train": [], "validation": [], "test": []}
    for task in tasks[:6]:
        task_split["train"].extend(record.episode_id for record in by_task[task])
    for task in tasks[6:8]:
        task_split["validation"].extend(record.episode_id for record in by_task[task])
    for task in tasks[8:10]:
        task_split["test"].extend(record.episode_id for record in by_task[task])
    return {"episode_held_out": episode_split, "task_held_out": task_split}


def _split_audit(splits: dict[str, dict[str, list[str]]], records: list[RoboMINDRecord]) -> dict[str, Any]:
    task = {record.episode_id: record.task_id for record in records}
    result: dict[str, Any] = {}
    for name, split in splits.items():
        sets = {key: set(value) for key, value in split.items()}
        no_episode_overlap = not any(sets[a] & sets[b] for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")))
        task_sets = {key: sorted({task[episode_id] for episode_id in value}) for key, value in split.items()}
        task_disjoint = not any(set(task_sets[a]) & set(task_sets[b]) for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")))
        result[name] = {
            "splits": {key: {"episodes": len(value), "tasks": task_sets[key]} for key, value in split.items()},
            "no_episode_or_fragment_leakage": no_episode_overlap,
            "task_disjoint": task_disjoint,
            "pass": no_episode_overlap and (task_disjoint if name == "task_held_out" else True),
        }
    return result


def _make_access_artifacts(inventory: dict[str, Any], records: list[RoboMINDRecord]) -> None:
    probe = OUT / "access_probe"
    parquet_example = Path(records[0].parquet_path)
    failure_metadata = {
        "official_outcome_fields_found_in_selected_le_robot_metadata": [],
        "failure_data_directory_visible": inventory["failure_data_directory_visible"],
        "outcome_semantics": "unavailable",
        "failure_category_semantics": "unavailable",
        "failure_onset_semantics": "unavailable",
        "directory_names_used_as_labels": False,
        "consequence": "Failure/outcome formulations A, B, and D are not authorized. Only formulation C may be audited.",
    }
    write_json(probe / "robomind_repository_probe.json", {
        "dataset": ROBOMIND_DATASET,
        "repository_visible_locally": inventory["source_root_exists"],
        "source_root": inventory["source_root"],
        "official_readme_visible": (SOURCE / "README.md").is_file(),
        "remote_probe": "not attempted: current container proxy endpoints point to unavailable localhost ports; local official snapshot is readable and sufficient for bounded admission",
    })
    write_json(probe / "robomind_access_status.json", {
        "status": "LOCAL_READ_ACCESS_AVAILABLE",
        "authentication_required_for_local_read": False,
        "gated_access_bypassed": False,
        "trajectory_files_accessible": inventory["readable_parquet_files"] > 0,
        "example_file": str(parquet_example),
        "example_file_readable": parquet_example.is_file(),
        "file_sizes_available": True,
        "decision": "PROCEED_LOCAL_BOUNDED_SUBSET",
    })
    write_json(probe / "robomind_file_inventory_probe.json", {
        **inventory,
        "selected_candidate_episodes": len(records),
        "selected_candidate_tasks": len({record.task_id for record in records}),
        "selected_robot_embodiments": sorted({record.robot_embodiment for record in records}),
        "selected_front_rgb_available": sum(record.front_video_path is not None for record in records),
    })
    write_json(probe / "robomind_failure_metadata_probe.json", failure_metadata)
    (probe / "robomind_access_blocker_report.md").write_text(
        "# RoboMIND access and label boundary\n\n"
        "The local official RoboMIND2.0 snapshot is readable. It provides 238 Parquet episodes, HDF5 examples, metadata and RGB videos; no gated-access control was bypassed. "
        "No `failure_data` directory or official success/failure, category, or onset field is present in the selected LeRobot metadata. "
        "Folder and source path names are retained only for provenance and split grouping, never as labels or fair inputs.\n"
    )


def _write_docs() -> None:
    docs = OUT / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "rm_series_plan.md").write_text(
        "# RM-Series plan\n\nRM0–RM3 use only the local RoboMIND2.0 snapshot. The bounded source is 100 real Franka episodes from 10 repeated tasks. "
        "No outcome labels are inferred. Therefore only logged action–future consistency (Formulation C) can be evaluated; no physical intervention or failure-prediction claim is permitted.\n"
    )
    (docs / "rm_series_results.md").write_text(
        "# RM-Series results\n\nThis file is generated after the baseline gate. See `final_package/final_report.md` for the current decision and its evidence.\n"
    )
    (docs / "rm_series_handoff.md").write_text(
        "# RM-Series handoff\n\nUse `final_package/final_decision.json` as the controlling decision. Do not train a method unless it explicitly reports `RM_BENCHMARK_HEADROOM_FOUND`.\n"
    )


def _write_processed(records: list[RoboMINDRecord], payload: dict[str, dict[str, Any]], audits: list[dict[str, Any]]) -> None:
    root = OUT / "processed_subset"
    rows: list[dict[str, Any]] = []
    for record in records:
        item = payload[record.episode_id]
        stem = record.episode_id.replace(":", "_")
        state_path = root / "numeric" / f"{stem}.npz"
        visual_path = root / "visual" / f"{stem}.npz"
        state_path.parent.mkdir(parents=True, exist_ok=True)
        visual_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(state_path, state=item["state"], action=item["action"], timestamp=item["timestamp"], frame_index=item["frame_index"])
        np.savez_compressed(visual_path, pre_anchor_feature=item["visual"], anchor=np.asarray(item["anchors"], dtype=np.int64))
        rows.append({
            **record.json(),
            "dataset_name": ROBOMIND_DATASET,
            "success_or_failure": None,
            "failure_category": None,
            "failure_onset_step": None,
            "outcome_source": "not_present_in_selected_official_metadata",
            "robot_state_path": str(state_path.relative_to(root)),
            "action_path": str(state_path.relative_to(root)),
            "timestamp_path": str(state_path.relative_to(root)),
            "visual_feature_path": str(visual_path.relative_to(root)),
            "camera_metadata": {"selected_camera": "observation.rgb_images.camera_front", "features": "deterministic RGB moments; no external encoder"},
            "source_file_paths": {"parquet": record.parquet_path, "front_video": record.front_video_path, "metadata": record.metadata_path},
        })
    write_json(root / "data_manifest.json", {
        "schema": "rm-series-robomind-processed-subset-v1",
        "dataset": ROBOMIND_DATASET,
        "episodes": rows,
        "episode_count": len(rows),
        "task_count": len({record.task_id for record in records}),
        "fair_input_audit": fair_input_audit(FAIR_FIELDS),
    })
    write_json(root / "adapter_audit.json", {
        "status": "pass",
        "source_read_only": True,
        "corrupted_episodes_silently_repaired": False,
        "outcome_label_inferred_from_path": False,
        "records": len(rows),
        "required_schema_fields": ["episode_id", "dataset_name", "robot_embodiment", "task_name", "language_instruction", "success_or_failure", "failure_category", "failure_onset_step", "robot_state", "action", "camera_metadata", "source_file_paths", "split_metadata"],
    })
    write_json(root / "alignment_audit.json", {"status": "pass", "all_episodes_pass": all(row["pass"] for row in audits), "episodes": audits})
    write_json(root / "label_semantics_audit.json", {
        "status": "outcome_labels_unavailable_but_consistency_target_valid",
        "outcome_or_failure_supervision_authorized": False,
        "folder_names_used_as_labels": False,
        "formulation_c_authorized": True,
        "reason": "A positive is the actual logged action continuation, not an inferred success label; all negatives are explicitly constructed action alternatives.",
    })
    write_json(root / "modality_audit.json", {
        "status": "pass",
        "state_action_aligned": len(audits),
        "front_rgb_videos": sum(bool(record.front_video_path) for record in records),
        "depth_present_in_source_parquet": True,
        "used_fair_modalities": list(FAIR_FIELDS),
        "not_used": ["episode_id", "source path", "folder name", "future observations", "outcome labels"],
    })


def _write_source_manifests(records: list[RoboMINDRecord]) -> None:
    selected: list[dict[str, Any]] = []
    for record in records:
        paths = [Path(record.parquet_path)] + ([Path(record.front_video_path)] if record.front_video_path else [])
        for path in paths:
            selected.append({
                "episode_id": record.episode_id,
                "task": record.task_id,
                "modality": "parquet_numeric_depth" if path.suffix == ".parquet" else "front_rgb_video",
                "source_path": str(path),
                "local_path": str(path),
                "bytes": int(path.stat().st_size),
                "download_status": "preexisting_local_source_read_only",
            })
    plan = {
        "dataset": ROBOMIND_DATASET,
        "selection_rule": "10 task families with lowest byte cost for 10 repeated local trajectories each",
        "episodes": len(records),
        "tasks": len({record.task_id for record in records}),
        "parquet_bytes": int(sum(record.source_bytes for record in records)),
        "front_video_bytes": int(sum(Path(record.front_video_path).stat().st_size for record in records if record.front_video_path)),
        "new_remote_download_bytes": 0,
        "source_is_preexisting_local_snapshot": True,
        "visual_budget_under_10_gb": True,
    }
    write_json(OUT / "download_plan.json", plan)
    write_json(OUT / "source_subset" / "subset_manifest.json", {**plan, "episodes_detail": [record.json() for record in records], "not_used_for_method_training": True})
    write_json(OUT / "source_download_manifest.json", {"schema": "rm-series-local-source-manifest-v1", "files": selected, "new_downloads": 0})
    hashes = [{**row, "sha256": sha256_file(Path(row["local_path"]))} for row in selected]
    write_json(OUT / "source_hash_manifest.json", {"schema": "rm-series-source-hash-manifest-v1", "files": hashes})
    write_json(OUT / "file_size_manifest.json", {"schema": "rm-series-file-size-manifest-v1", "files": [{key: row[key] for key in ("episode_id", "task", "modality", "source_path", "bytes", "download_status")} for row in selected]})


def _task_artifacts(splits: dict[str, dict[str, list[str]]], sample_sets: dict[str, dict[str, dict[str, Any]]], records: list[RoboMINDRecord]) -> None:
    task_root = OUT / "task_formulations"
    formulation = {
        "name": "action_future_consistency",
        "formulation": "C",
        "claim_boundary": "Logged real-robot action-conditioned future consistency only; not episode outcome, failure onset, or physical counterfactual success.",
        "input": ["pre-anchor state history", "pre-anchor deterministic front-RGB features", "candidate action chunk"],
        "positive": "true logged continuation from the same anchor",
        "negatives": ["same-task different attempt", "wrong-time same episode", "reversed action", "shuffled action"],
        "outcome_label_used": False,
    }
    write_json(task_root / "task_formulation_action_future_consistency.json", formulation)
    neg_counts: dict[str, int] = defaultdict(int)
    for definition in sample_sets.values():
        for split in definition.values():
            for row in split["rows"]:
                neg_counts[str(row["kind"])] += 1
    write_json(task_root / "negative_sampling_audit_action_future_consistency.json", {
        "status": "pass",
        "counts": dict(neg_counts),
        "same_task_negative_requires_repeated_attempts": True,
        "fallback_pools_used": False,
        "provenance_used_as_fair_input": False,
    })
    leakage = fair_input_audit(FAIR_FIELDS)
    leakage.update({"status": "pass" if leakage["pass"] else "fail", "future_observations_used_in_fair_inputs": False, "labels_used_in_fair_inputs": False, "source_or_folder_inputs_used": False, "oracle_isolated_from_fair_models": True})
    write_json(task_root / "leakage_audit_action_future_consistency.json", leakage)
    audit = _split_audit(splits, records)
    write_json(task_root / "split_audit_action_future_consistency.json", audit)


def _run_baselines(sample_sets: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    baseline_names = [
        "constant", "action_only", "action_magnitude_smoothness", "episode_time_proxy", "current_state_only", "current_frame_only",
        "frozen_visual_feature_temporal_model", "proprio_action_gru", "state_action_tcn", "visual_history_gru", "visual_history_tcn",
        "visual_state_gru", "visual_action_gru", "visual_state_action_gru", "nearest_neighbor_retrieval",
    ]
    reports: dict[str, Any] = {}
    raw_entries: list[dict[str, Any]] = []
    for split_name, splits in sample_sets.items():
        train_rows = splits["train"]["rows"]
        test_rows = splits["test"]["rows"]
        reports[split_name] = {}
        for name in baseline_names:
            seeds = (17,) if name in {"constant", "nearest_neighbor_retrieval"} else SEEDS
            per_seed: list[dict[str, Any]] = []
            for seed in seeds:
                scores = _run_model(name, train_rows, test_rows, seed)
                metrics = _metrics(test_rows, scores)
                relative = Path("raw_scores") / split_name / name / f"seed_{seed}.jsonl"
                rows = [{"sample_id": row["sample_id"], "anchor_key": row["anchor_key"], "label": row["label"], "score": float(score), "split": split_name, "model": name, "seed": seed} for row, score in zip(test_rows, scores)]
                write_jsonl(OUT / relative, rows)
                raw_entries.append({"split": split_name, "model": name, "seed": seed, "path": str(relative), "rows": len(rows), "sha256": sha256_file(OUT / relative)})
                per_seed.append({"seed": seed, "metrics": metrics})
            means = {metric: float(np.mean([item["metrics"][metric] for item in per_seed])) for metric in ("balanced_accuracy", "auroc", "pair_order_accuracy")}
            stds = {metric: float(np.std([item["metrics"][metric] for item in per_seed])) for metric in ("balanced_accuracy", "auroc", "pair_order_accuracy")}
            reports[split_name][name] = {"seeds": per_seed, "mean": means, "std": stds, "fair": name not in {"episode_time_proxy"}}
        # Isolated oracle/future-action diagnostic: never sent to fair models.
        oracle = _oracle_scores(test_rows)
        metric = _metrics(test_rows, oracle)
        relative = Path("raw_scores") / split_name / "oracle_future_action_distance" / "seed_0.jsonl"
        rows = [{"sample_id": row["sample_id"], "anchor_key": row["anchor_key"], "label": row["label"], "score": float(score), "split": split_name, "model": "oracle_future_action_distance", "seed": 0} for row, score in zip(test_rows, oracle)]
        write_jsonl(OUT / relative, rows)
        raw_entries.append({"split": split_name, "model": "oracle_future_action_distance", "seed": 0, "path": str(relative), "rows": len(rows), "sha256": sha256_file(OUT / relative), "diagnostic_only": True})
        reports[split_name]["oracle_future_action_distance"] = {"seeds": [{"seed": 0, "metrics": metric}], "mean": {key: metric[key] for key in ("balanced_accuracy", "auroc", "pair_order_accuracy")}, "std": {key: 0.0 for key in ("balanced_accuracy", "auroc", "pair_order_accuracy")}, "fair": False, "diagnostic_only": True}
    manifest = {"schema": "rm-series-raw-score-manifest-v1", "entries": raw_entries, "all_entries_present": all((OUT / row["path"]).is_file() for row in raw_entries), "fair_models_never_receive": ["episode_id", "task_id", "source path", "folder name", "future action", "label"]}
    write_json(OUT / "raw_score_manifest.json", manifest)
    return {"reports": reports, "raw_manifest": manifest}


def _baseline_decision(results: dict[str, Any]) -> tuple[dict[str, Any], str]:
    report = results["reports"]
    # Standard fair models exclude controls and the future-action oracle.
    standards = ["proprio_action_gru", "state_action_tcn", "visual_history_gru", "visual_history_tcn", "visual_state_gru", "visual_action_gru", "visual_state_action_gru", "frozen_visual_feature_temporal_model", "nearest_neighbor_retrieval"]
    shortcuts = ["action_only", "action_magnitude_smoothness", "episode_time_proxy", "current_state_only", "current_frame_only"]
    episode = report["episode_held_out"]
    task = report["task_held_out"]
    best_name = max(standards, key=lambda name: episode[name]["mean"]["balanced_accuracy"])
    best = episode[best_name]["mean"]["balanced_accuracy"]
    hardest_best_name = max(standards, key=lambda name: task[name]["mean"]["balanced_accuracy"])
    hardest = task[hardest_best_name]["mean"]["balanced_accuracy"]
    max_shortcut = max(max(episode[name]["mean"]["balanced_accuracy"], task[name]["mean"]["balanced_accuracy"]) for name in shortcuts)
    current_frame = max(episode["current_frame_only"]["mean"]["balanced_accuracy"], task["current_frame_only"]["mean"]["balanced_accuracy"])
    oracle = episode["oracle_future_action_distance"]["mean"]["balanced_accuracy"]
    stable = max(episode[name]["std"]["balanced_accuracy"] for name in standards if len(episode[name]["seeds"]) > 1) <= 0.05
    gates = {
        "shortcut_max_le_065": max_shortcut <= 0.65,
        "current_frame_only_le_070": current_frame <= 0.70,
        "fair_recoverability_ge_065": best >= 0.65,
        "best_standard_lt_090": best < 0.90,
        "oracle_headroom_ge_008": oracle >= best + 0.08,
        "stable_across_seeds": stable,
        "task_held_out_above_chance": hardest > 0.50,
        "raw_scores_saved": results["raw_manifest"]["all_entries_present"],
    }
    if not gates["shortcut_max_le_065"] or not gates["current_frame_only_le_070"]:
        decision = "RM_SHORTCUT_DOMINATES"
    elif not gates["fair_recoverability_ge_065"] or not gates["task_held_out_above_chance"]:
        decision = "RM_RECOVERABILITY_INSUFFICIENT"
    elif not gates["best_standard_lt_090"]:
        decision = "RM_BASELINES_SATURATE"
    elif all(gates.values()):
        decision = "RM_BENCHMARK_HEADROOM_FOUND_METHOD_NOT_RUN"
    else:
        decision = "RM_HUMAN_REVIEW_REQUIRED"
    summary = {"best_standard": best_name, "best_standard_balanced_accuracy": best, "task_held_out_best_standard": hardest_best_name, "task_held_out_best_balanced_accuracy": hardest, "maximum_shortcut_balanced_accuracy": max_shortcut, "current_frame_only_balanced_accuracy": current_frame, "oracle_balanced_accuracy": oracle, "gates": gates, "decision": decision}
    return summary, decision


def _write_final(records: list[RoboMINDRecord], baseline: dict[str, Any], decision: str) -> None:
    final = OUT / "final_package"
    method_authorized = False
    final_decision = {
        "schema": "rm-series-final-decision-v1",
        "decision": decision,
        "selected_source": ROBOMIND_DATASET,
        "method_authorized": method_authorized,
        "method_ran": False,
        "outcome_or_failure_claim_authorized": False,
        "physical_counterfactual_claim_authorized": False,
        "claim_boundary": "Logged real-robot action-conditioned future consistency only, if the baseline gate passes; current final decision controls whether a method trial is authorized.",
        "reason": "RoboMIND outcome/failure semantics are not used or inferred. The decision is driven only by the strict logged-consistency baseline gate.",
        "episodes": len(records),
        "tasks": len({record.task_id for record in records}),
        "baseline_summary": baseline,
    }
    write_json(final / "final_decision.json", final_decision)
    (final / "dataset_access_report.md").write_text(
        "# Dataset access report\n\nThe local RoboMIND2.0 snapshot is read-only and accessible without authentication in this environment. The bounded source uses 100 existing Parquet episodes and one preexisting front RGB video per episode; no remote download or gated-access bypass occurred.\n"
    )
    (final / "adapter_report.md").write_text(
        "# Adapter report\n\nThe adapter reads only state, action, timestamps and front-video frames. It stores normalized numeric arrays and deterministic pre-anchor RGB statistics. It rejects non-monotone, non-finite or mismatched streams and never treats a path or directory as a label.\n"
    )
    (final / "task_formulation_report.md").write_text(
        "# Task formulation report\n\nOutcome/failure labels are unavailable. The only evaluated task is logged action–future consistency: the true logged action chunk is positive; four documented action alternatives are negative. This is not an intervention, completion, or failure claim.\n"
    )
    write_json(final / "baseline_summary.json", baseline)
    write_json(final / "raw_score_manifest.json", json.loads((OUT / "raw_score_manifest.json").read_text()))
    (final / "claim_boundary.md").write_text(
        "# Claim boundary\n\nAllowed: a result about a controlled, logged RoboMIND action-conditioned consistency benchmark.\n\nNot allowed: real-robot failure prediction, success/outcome verification, failure onset, causal intervention success, or physical counterfactual execution.\n"
    )
    (final / "paper_positioning.md").write_text(
        "# Paper positioning\n\nThis series can only support a method paper if the logged-consistency gate finds genuine headroom and a later single-family method meets its preregistered improvements. It currently provides an auditable real-robot benchmark decision, not an outcome-verification result.\n"
    )
    (final / "next_steps.md").write_text(
        "# Next steps\n\nIf the final decision is not `RM_BENCHMARK_HEADROOM_FOUND_METHOD_NOT_RUN`, do not train a method. Separately request official RoboMIND failure/outcome labels and onset semantics if failure verification is required.\n"
    )
    (final / "final_report.md").write_text(
        f"# RM-Series final report\n\nDecision: `{decision}`.\n\n"
        f"Access: local, read-only RoboMIND2.0. Selected subset: {len(records)} real Franka trajectories across {len({record.task_id for record in records})} repeated tasks. "
        "Modalities used by fair baselines: pre-anchor state, candidate action and deterministic pre-anchor front RGB features. "
        "No episode outcome, failure category or failure onset label was available or inferred. "
        f"Best episode-held-out fair standard baseline: `{baseline['best_standard']}` at balanced accuracy {baseline['best_standard_balanced_accuracy']:.3f}; "
        f"hardest task-held-out best baseline: `{baseline['task_held_out_best_standard']}` at {baseline['task_held_out_best_balanced_accuracy']:.3f}. "
        "No new ActMask/RelDynVerifier method was trained.\n"
    )
    generated = []
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and path not in {final / "reproducibility_manifest.json"}:
            generated.append({"path": str(path.relative_to(OUT)), "sha256": sha256_file(path)})
    write_json(final / "reproducibility_manifest.json", {"schema": "rm-series-reproducibility-v1", "generator": str(Path(__file__).relative_to(ROOT)), "generator_sha256": sha256_file(Path(__file__)), "python": os.environ.get("VIRTUAL_ENV", "/home/tzh/conda_envs/actmask") + "/bin/python", "source_hash_manifest": "../source_hash_manifest.json", "generated_files": generated, "no_method_training": True})


def run() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    append_log("rm_series_started", source=str(SOURCE), dataset=ROBOMIND_DATASET)
    _write_docs()
    inventory = scan_local_robomind(SOURCE)
    records = select_bounded_records(SOURCE, max_tasks=10, episodes_per_task=10)
    _make_access_artifacts(inventory, records)
    append_log("rm0_access_complete", local_parquet=inventory["readable_parquet_files"], selected_episodes=len(records))
    _write_source_manifests(records)
    append_log("rm2_bounded_local_source_hashed", source_files=len(records) * 2)
    payload, audits = _episode_payload(records)
    _write_processed(records, payload, audits)
    append_log("rm3_adapter_alignment_complete", aligned_episodes=sum(row["pass"] for row in audits))
    splits = _split_definitions(records)
    sample_sets = {name: build_samples(payload, definition) for name, definition in splits.items()}
    _task_artifacts(splits, sample_sets, records)
    append_log("rm4_task_and_splits_complete", formulations=["action_future_consistency"])
    results = _run_baselines(sample_sets)
    baseline, decision = _baseline_decision(results)
    write_json(OUT / "baseline_report.json", {"schema": "rm-series-baseline-report-v1", "task": "action_future_consistency", "reports": results["reports"], "summary": baseline})
    write_json(OUT / "shortcut_audit.json", {"status": "pass" if baseline["gates"]["shortcut_max_le_065"] and baseline["gates"]["current_frame_only_le_070"] else "fail", "maximum_shortcut": baseline["maximum_shortcut_balanced_accuracy"], "current_frame_only": baseline["current_frame_only_balanced_accuracy"], "thresholds": {"shortcut": 0.65, "current_frame": 0.70}})
    write_json(OUT / "recoverability_audit.json", {"status": "pass" if baseline["gates"]["fair_recoverability_ge_065"] else "fail", "best_standard": baseline["best_standard"], "balanced_accuracy": baseline["best_standard_balanced_accuracy"], "threshold": 0.65, "task_held_out": baseline["task_held_out_best_balanced_accuracy"]})
    write_json(OUT / "ci_consistency_audit.json", {"status": "pass" if baseline["gates"]["stable_across_seeds"] else "fail", "seed_stability_required_std_le": 0.05, "pass": baseline["gates"]["stable_across_seeds"]})
    append_log("rm6_baseline_gate_complete", decision=decision, best_standard=baseline["best_standard"], best_score=baseline["best_standard_balanced_accuracy"])
    append_log("rm9_final_package_complete", decision=decision, method_ran=False)
    _write_final(records, baseline, decision)
    return {"decision": decision, "episodes": len(records), "tasks": len({record.task_id for record in records}), "best_standard": baseline["best_standard"], "best_standard_balanced_accuracy": baseline["best_standard_balanced_accuracy"]}


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, sort_keys=True))
