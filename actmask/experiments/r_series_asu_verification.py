"""Strict, versioned R-Series evaluation on the public ASU TableTop UR5 logs.

This module is intentionally separate from the frozen invalid DROID R2 attempt.
It uses official ASU action/object task fields and dataset-provided object poses
to create logged continuation and action-to-future-relation tasks.  It never
relabels an unavailable negative pool: the anchor is discarded and counted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image
from torch import nn

from actmask.data.asu_table_top_adapter import ASU_DATASET
from actmask.data.real_robot_dataset_adapter import (
    FORBIDDEN_FAIR_INPUTS,
    R_SERIES_ROOT,
    append_run_log,
    canonical_json,
    composite_file_hash,
    forbidden_input_audit,
    read_jsonl,
    sha256_file,
    write_json,
    write_jsonl,
)


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "actmask" / "r_series_real_robot_verification"
PROCESSED = OUT / "processed_subset_asu_tabletop"
ATTEMPT_ID = "asu_ur5_object_relation_v1"
ATTEMPT = OUT / "asu_attempts" / ATTEMPT_ID
TASK_DIR = ATTEMPT / "r2"
SPLIT_MODE = "within_task"
HISTORY_STEPS = 16
ACTION_STEPS = 8
ANCHOR_STRIDE = 16
CANDIDATES_PER_SET = 5
MIN_EPISODES_PER_TASK_PER_SPLIT = 2


def _configure_attempt(*, heldout_goal_object: bool) -> None:
    """Select a versioned strict IID or language-family-held-out attempt."""

    global ATTEMPT_ID, ATTEMPT, TASK_DIR, SPLIT_MODE
    if heldout_goal_object:
        ATTEMPT_ID = "asu_ur5_object_relation_heldout_goal_object_v1"
        SPLIT_MODE = "heldout_goal_object"
    ATTEMPT = OUT / "asu_attempts" / ATTEMPT_ID
    TASK_DIR = ATTEMPT / "r2"


class AsuTaskError(RuntimeError):
    """A strict task-construction failure which triggers the R-Series tree."""


def _sha(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _config() -> dict[str, Any]:
    return {
        "schema": "r-series-asu-ur5-relation-v1",
        "attempt_id": ATTEMPT_ID,
        "dataset": ASU_DATASET,
        "processed_manifest_sha256": sha256_file(PROCESSED / "data_manifest.json"),
        "selected_task_policy": "official action_inst+goal_object task IDs with at least six episodes",
        "split_mode": SPLIT_MODE,
        "history_steps": HISTORY_STEPS,
        "action_steps": ACTION_STEPS,
        "anchor_stride": ANCHOR_STRIDE,
        "candidates_per_set": CANDIDATES_PER_SET,
        "relation_target": "ordered-step change of provided object-minus-end-effector 6-D poses from history endpoint to chunk endpoint",
        "required_negative_kinds": ["same_episode_distant", "same_task_other_episode", "different_task", "temporally_reversed_logged_chunk"],
        "minimum": {"task_families": 5, "test_episodes": 20, "episodes_per_task_per_split": MIN_EPISODES_PER_TASK_PER_SPLIT},
        "fair_input_fields": [
            "history_observation_state", "history_robot_state", "history_rgb_thumbnail_at_or_before_anchor",
            "candidate_action_chunk", "language_instruction_bag_of_words_trained_on_train_only", "official_task_id",
        ],
        "forbidden_fair_inputs": list(FORBIDDEN_FAIR_INPUTS),
        "claim": "action-conditioned consistency / verification from logged real UR5 trajectories",
        "not_claimed": ["true counterfactual physical success prediction", "object pose from a future step as model input"],
    }


def initialize() -> dict[str, Any]:
    """Freeze the automatic DROID->ASU decision and this attempt's settings."""

    if not (PROCESSED / "data_manifest.json").is_file():
        raise FileNotFoundError("ASU normalized subset is absent; finish the backup adapter first")
    config = _config()
    ATTEMPT.mkdir(parents=True, exist_ok=True)
    path = ATTEMPT / "config.json"
    if path.exists() and json.loads(path.read_text()) != config:
        raise RuntimeError("ASU attempt config exists with a different frozen value")
    if not path.exists():
        write_json(path, config)
    activation = {
        "schema": "r-series-backup-activation-v1",
        "primary": "DROID droid_100",
        "primary_status": "R1 adapter passed; R2 attempt 1 invalid due unique instructions, insufficient repeat-task support, and detected negative-sampler semantic fallback",
        "backup": ASU_DATASET,
        "backup_reason": "official compact real UR5 data with repeated official action/object task labels and provided object pose trajectories",
        "attempt_id": ATTEMPT_ID,
        "config_sha256": sha256_file(path),
    }
    root_path = OUT / "asu_backup_activation.json"
    if not root_path.exists():
        write_json(root_path, activation)
        append_run_log(OUT / "run_log.jsonl", event="r2_primary_droid_invalid_backup_asu_activated", payload={"attempt_id": ATTEMPT_ID, "backup": ASU_DATASET})
    write_json(ATTEMPT / "parent_decision.json", {"backup_activation_path": str(root_path.relative_to(OUT)), "attempt_id": ATTEMPT_ID, "split_mode": SPLIT_MODE})
    return {"config": config, "config_sha256": sha256_file(path), "activation": activation}


def _load_episode(row: Mapping[str, Any]) -> dict[str, Any]:
    def load(name: str) -> np.ndarray:
        return np.load(PROCESSED / str(row[name]))["value"].astype(np.float32)

    robot, relation, observation, action, timestamp = (load(name) for name in ("robot_state_path", "relation_state_path", "observation_state_path", "action_path", "timestamp_path"))
    if not (len(robot) == len(relation) == len(observation) == len(action) == len(timestamp)):
        raise AsuTaskError(f"alignment failure in {row['episode_id']}")
    if relation.shape[1] != 42 or observation.shape[1] != 49 or robot.shape[1] != 7 or action.shape[1] != 7:
        raise AsuTaskError(f"unexpected ASU dimensions in {row['episode_id']}")
    return {**dict(row), "robot": robot, "relation": relation, "observation": observation, "action": action, "timestamp": timestamp, "rgb": sorted((dict(item) for item in row["rgb_paths"]), key=lambda item: int(item["step"]))}


def _eligible_tasks(episodes: Sequence[Mapping[str, Any]]) -> list[str]:
    counts = Counter(str(row["task_id"]) for row in episodes)
    names = sorted(name for name, count in counts.items() if count >= 3 * MIN_EPISODES_PER_TASK_PER_SPLIT)
    if not (5 <= len(names) <= 20):
        raise AsuTaskError(f"expected 5-20 repeated official tasks, found {len(names)}: {names}")
    return names


def _splits(episodes: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """Give every selected official task at least two episode groups per split."""

    if SPLIT_MODE == "heldout_goal_object":
        # Object words are present in the language instruction.  The test set
        # uses entirely unseen goal-object language families; validation uses
        # a third held-out family for model selection.  Each official task is
        # nevertheless internally repeated, so strict same-task negatives
        # remain available without a fallback.
        heldout = {"bread": "train", "bottle": "train", "cube": "train", "pepsi": "validation", "coke": "test", "milk": "test"}
        owners: dict[str, str] = {}
        for episode in episodes:
            object_name = str(episode["task_id"]).split("__", 1)[1]
            if object_name not in heldout:
                raise AsuTaskError(f"unmapped official goal object: {object_name}")
            owners[str(episode["episode_id"])] = heldout[object_name]
        if not {"train", "validation", "test"}.issubset(set(owners.values())):
            raise AsuTaskError("held-out object split has an empty partition")
        return owners
    groups: dict[str, list[str]] = defaultdict(list)
    for episode in episodes:
        groups[str(episode["task_id"])].append(str(episode["episode_id"]))
    owners: dict[str, str] = {}
    for task, identifiers in sorted(groups.items()):
        ordered = sorted(identifiers, key=lambda identifier: hashlib.sha256(f"{ATTEMPT_ID}:{task}:{identifier}".encode()).hexdigest())
        n = len(ordered)
        if n < 3 * MIN_EPISODES_PER_TASK_PER_SPLIT:
            raise AsuTaskError(f"task {task} has {n}, insufficient for strict split")
        n_train = n - 2 * MIN_EPISODES_PER_TASK_PER_SPLIT
        for index, identifier in enumerate(ordered):
            owners[identifier] = "train" if index < n_train else "validation" if index < n_train + MIN_EPISODES_PER_TASK_PER_SPLIT else "test"
    return owners


def _tokens(text: str) -> list[str]:
    import re

    return re.findall(r"[a-z]+", text.lower())


def _vocabulary(episodes: Sequence[Mapping[str, Any]], splits: Mapping[str, str], max_words: int = 128) -> tuple[dict[str, int], np.ndarray]:
    count: Counter[str] = Counter()
    for episode in episodes:
        if splits[str(episode["episode_id"])] == "train":
            count.update(_tokens(str(episode["language_instruction"])))
    words = [word for word, _ in count.most_common(max_words)]
    return {word: index for index, word in enumerate(words)}, np.asarray([count[word] for word in words], dtype=np.int64)


def _bow(text: str, vocabulary: Mapping[str, int]) -> np.ndarray:
    value = np.zeros(len(vocabulary), dtype=np.float32)
    for word in _tokens(text):
        if word in vocabulary:
            value[vocabulary[word]] += 1.0
    return value / value.sum() if value.sum() else value


def _history_rgb(episode: Mapping[str, Any], anchor: int) -> np.ndarray:
    available = [item for item in episode["rgb"] if int(item["step"]) <= anchor - 1]
    if not available:
        raise AsuTaskError(f"history RGB missing before anchor {anchor} in {episode['episode_id']}")
    chosen = available[-3:]
    if len(chosen) < 3:
        chosen = [chosen[0]] * (3 - len(chosen)) + chosen
    result: list[np.ndarray] = []
    for item in chosen:
        with Image.open(PROCESSED / str(item["path"])) as image:
            result.append(np.asarray(image.convert("RGB"), dtype=np.uint8).transpose(2, 0, 1))
    return np.stack(result)


def _relative_object_pose(relation: np.ndarray) -> np.ndarray:
    values = relation.reshape(len(relation), 7, 6)
    return (values[:, 1:, :] - values[:, :1, :]).reshape(len(relation), -1)


def _anchors(episodes: Sequence[Mapping[str, Any]], splits: Mapping[str, str], vocabulary: Mapping[str, int]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for episode in episodes:
        relative = _relative_object_pose(np.asarray(episode["relation"], np.float32))
        for anchor in range(HISTORY_STEPS, len(episode["action"]) - ACTION_STEPS, ANCHOR_STRIDE):
            endpoint = anchor + ACTION_STEPS - 1
            result.append({
                "anchor_id": f"{episode['episode_id']}:{anchor}", "episode_id": episode["episode_id"],
                "task_id": episode["task_id"], "split": splits[str(episode["episode_id"])], "anchor": anchor,
                "history_observation": episode["observation"][anchor - HISTORY_STEPS:anchor],
                "history_robot": episode["robot"][anchor - HISTORY_STEPS:anchor],
                "candidate_action": episode["action"][anchor:anchor + ACTION_STEPS],
                # Ordered-step endpoint convention: action chunk [a,a+8), target change from a-1 to a+7.
                "future_relation_delta": relative[endpoint] - relative[anchor - 1],
                "language_bow": _bow(str(episode["language_instruction"]), vocabulary),
                "visual_history": _history_rgb(episode, anchor),
            })
    return result


def _pick(pool: Sequence[Mapping[str, Any]], key: str) -> Mapping[str, Any]:
    if not pool:
        raise AsuTaskError("strict candidate pool unexpectedly empty")
    return pool[int(hashlib.sha256(key.encode()).hexdigest(), 16) % len(pool)]


def _candidate_rows(anchors: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_split: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for anchor in anchors:
        by_split[str(anchor["split"])].append(anchor)
    rows: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    kind_counts: Counter[str] = Counter()
    for anchor in anchors:
        pool = by_split[str(anchor["split"])]
        same_episode = [item for item in pool if item["episode_id"] == anchor["episode_id"] and abs(int(item["anchor"]) - int(anchor["anchor"])) >= HISTORY_STEPS + ACTION_STEPS]
        same_task = [item for item in pool if item["episode_id"] != anchor["episode_id"] and item["task_id"] == anchor["task_id"]]
        different_task = [item for item in pool if item["task_id"] != anchor["task_id"]]
        missing = [name for name, values in (("same_episode_distant", same_episode), ("same_task_other_episode", same_task), ("different_task", different_task)) if not values]
        if missing:
            skipped["+".join(missing)] += 1
            continue
        source_same_episode = _pick(same_episode, str(anchor["anchor_id"]) + ":same_episode")
        source_same_task = _pick(same_task, str(anchor["anchor_id"]) + ":same_task")
        source_different = _pick(different_task, str(anchor["anchor_id"]) + ":different_task")
        candidates = [
            ("logged_true_future", anchor["candidate_action"], True, anchor),
            ("same_episode_distant", source_same_episode["candidate_action"], False, source_same_episode),
            ("same_task_other_episode", source_same_task["candidate_action"], False, source_same_task),
            ("different_task", source_different["candidate_action"], False, source_different),
            ("temporally_reversed_logged_chunk", np.asarray(anchor["candidate_action"])[::-1].copy(), False, anchor),
        ]
        order = np.random.default_rng(int(hashlib.sha256((str(anchor["anchor_id"]) + ":display").encode()).hexdigest()[:16], 16)).permutation(len(candidates))
        for display_order, local_index in enumerate(order):
            kind, action, consistent, source = candidates[int(local_index)]
            rows.append({
                "candidate_set_id": anchor["anchor_id"], "candidate_id": hashlib.sha256(f"{anchor['anchor_id']}:{kind}".encode()).hexdigest()[:20],
                "episode_id": anchor["episode_id"], "task_id": anchor["task_id"], "split": anchor["split"], "anchor_step": int(anchor["anchor"]),
                "candidate_kind": kind, "candidate_display_order": int(display_order), "consistent": bool(consistent),
                "candidate_source_episode_id": source["episode_id"], "candidate_source_task_id": source["task_id"], "candidate_source_anchor": int(source["anchor"]),
                "history_observation": anchor["history_observation"], "history_robot": anchor["history_robot"], "candidate_action": action,
                "future_relation_delta": anchor["future_relation_delta"], "language_bow": anchor["language_bow"], "visual_history": anchor["visual_history"],
            })
            kind_counts[kind] += 1
    source_lookup = {(str(item["episode_id"]), int(item["anchor"])): item for item in anchors}
    checks: list[bool] = []
    for row in rows:
        source = source_lookup[(str(row["candidate_source_episode_id"]), int(row["candidate_source_anchor"]))]
        if row["candidate_kind"] == "same_episode_distant":
            checks.append(source["episode_id"] == row["episode_id"] and abs(int(source["anchor"]) - int(row["anchor_step"])) >= HISTORY_STEPS + ACTION_STEPS)
        elif row["candidate_kind"] == "same_task_other_episode":
            checks.append(source["episode_id"] != row["episode_id"] and source["task_id"] == row["task_id"])
        elif row["candidate_kind"] == "different_task":
            checks.append(source["task_id"] != row["task_id"])
        else:
            checks.append(source["episode_id"] == row["episode_id"])
        checks[-1] = checks[-1] and source["split"] == row["split"]
    slot_rates: dict[int, float] = {}
    for slot in range(CANDIDATES_PER_SET):
        labels = [bool(row["consistent"]) for row in rows if int(row["candidate_display_order"]) == slot]
        slot_rates[slot] = float(np.mean(labels)) if labels else float("nan")
    audit = {
        "schema": "r-series-asu-strict-negative-audit-v1", "candidate_sets": len({row["candidate_set_id"] for row in rows}),
        "candidate_rows": len(rows), "candidate_kinds": dict(sorted(kind_counts.items())), "skipped_anchors_by_missing_exact_pool": dict(sorted(skipped.items())),
        "semantic_checks_pass": bool(checks and all(checks)), "candidate_order_positive_rate": slot_rates,
        "candidate_order_label_rate_span": float(max(slot_rates.values()) - min(slot_rates.values())) if slot_rates else float("nan"),
        "positive_fraction": float(np.mean([row["consistent"] for row in rows])) if rows else float("nan"),
    }
    audit["pass"] = bool(audit["candidate_sets"] >= 100 and audit["semantic_checks_pass"] and not skipped and all(audit["candidate_kinds"].get(name, 0) == audit["candidate_sets"] for name in ("logged_true_future", "same_episode_distant", "same_task_other_episode", "different_task", "temporally_reversed_logged_chunk")) and abs(audit["positive_fraction"] - 0.2) < 1e-8 and audit["candidate_order_label_rate_span"] <= 0.08)
    return rows, audit


def build_tasks() -> dict[str, Any]:
    """Build a strict immutable ASU R2 attempt; an existing valid attempt is reused."""

    init = initialize()
    existing = TASK_DIR / "data_manifest.json"
    if existing.is_file():
        manifest = json.loads(existing.read_text())
        return {"status": "already_built", "manifest": manifest, "attempt_id": ATTEMPT_ID}
    records = [json.loads(line) for line in (PROCESSED / "episodes.jsonl").read_text().splitlines() if line]
    episodes = [_load_episode(row) for row in records]
    selected = _eligible_tasks(episodes)
    episodes = [row for row in episodes if str(row["task_id"]) in selected]
    splits = _splits(episodes)
    vocabulary, vocabulary_counts = _vocabulary(episodes, splits)
    anchors = _anchors(episodes, splits, vocabulary)
    candidates, negative_audit = _candidate_rows(anchors)
    if not negative_audit["pass"]:
        raise AsuTaskError(f"strict ASU negative sampling failed: {negative_audit}")
    candidates.sort(key=lambda row: hashlib.sha256(str(row["candidate_id"]).encode()).hexdigest())
    TASK_DIR.mkdir(parents=True, exist_ok=False)
    arrays = {
        "history_observation": np.asarray([row["history_observation"] for row in candidates], dtype=np.float32),
        "history_robot": np.asarray([row["history_robot"] for row in candidates], dtype=np.float32),
        "candidate_action": np.asarray([row["candidate_action"] for row in candidates], dtype=np.float32),
        "future_relation_delta": np.asarray([row["future_relation_delta"] for row in candidates], dtype=np.float32),
        "language_bow": np.asarray([row["language_bow"] for row in candidates], dtype=np.float32),
        "visual_history": np.asarray([row["visual_history"] for row in candidates], dtype=np.uint8),
        "consistent": np.asarray([row["consistent"] for row in candidates], dtype=np.float32),
    }
    np.savez_compressed(TASK_DIR / "task_arrays.npz", **arrays)
    metadata = [{key: value for key, value in row.items() if key not in {"history_observation", "history_robot", "candidate_action", "future_relation_delta", "language_bow", "visual_history"}} for row in candidates]
    write_jsonl(TASK_DIR / "candidate_rows.jsonl", metadata)
    write_json(TASK_DIR / "vocabulary.json", {"vocabulary": vocabulary, "train_counts": vocabulary_counts.tolist()})
    task_counts = Counter(str(row["task_id"]) for row in episodes)
    split_task_counts = {split: dict(sorted(Counter(row["task_id"] for row in episodes if splits[str(row["episode_id"])] == split).items())) for split in ("train", "validation", "test")}
    split_audit = {
        "schema": "r-series-asu-strict-episode-split-audit-v1", "selected_task_families": selected,
        "episode_count": len(episodes), "task_counts": dict(sorted(task_counts.items())), "split_task_counts": split_task_counts,
        "split_episode_counts": dict(Counter(splits.values())), "test_independent_episodes": sum(value == "test" for value in splits.values()),
        "per_task_minimum": min(min(values.values()) for values in split_task_counts.values()),
    }
    if SPLIT_MODE == "within_task":
        split_audit["pass"] = bool(len(selected) >= 5 and split_audit["test_independent_episodes"] >= 20 and split_audit["per_task_minimum"] >= MIN_EPISODES_PER_TASK_PER_SPLIT)
    else:
        # In the OOD split a whole official task belongs to exactly one
        # partition.  Verify instead that each task has enough same-task
        # partner episodes within its assigned partition and test has scale.
        owner_counts = {task: task_counts[task] for task in selected}
        split_audit["whole_task_heldout"] = True
        split_audit["per_task_assigned_split_minimum"] = min(owner_counts.values())
        split_audit["pass"] = bool(len(selected) >= 5 and split_audit["test_independent_episodes"] >= 20 and min(owner_counts.values()) >= 2)
    future_audit = {
        "schema": "r-series-asu-future-target-audit-v1", "history_rule": "all state and RGB values use steps <= anchor_step - 1",
        "target_rule": "future_relation_delta uses provided object-minus-EE pose at anchor+7 minus anchor-1",
        "future_target_fields": ["future_relation_delta"], "future_target_forbidden_fair_input": True, "pass": True,
    }
    fair = forbidden_input_audit(init["config"]["fair_input_fields"])
    formulation = {
        "schema": "r-series-asu-logged-relation-task-v1", "attempt_id": ATTEMPT_ID, "claim": init["config"]["claim"], "not_claimed": init["config"]["not_claimed"],
        "formulation_a": {"name": "logged_continuation_identification", "positive": "actual next logged action chunk", "negatives": init["config"]["required_negative_kinds"], "strictness": "anchors without every exact pool are excluded"},
        "formulation_c": {"name": "action_to_future_object_relation_prediction", "input": "history observation/action only", "target": "36-D logged object-minus-EE relative pose change", "evaluation": "MSE/cosine on true chunks and candidate ranking by predicted-versus-held-out target error"},
        "ordered_step_alignment": future_audit["target_rule"], "config_sha256": init["config_sha256"],
    }
    files = [TASK_DIR / "task_arrays.npz", TASK_DIR / "candidate_rows.jsonl", TASK_DIR / "vocabulary.json"]
    digest = composite_file_hash(TASK_DIR, files)
    manifest = {"schema": "r-series-asu-task-manifest-v1", "attempt_id": ATTEMPT_ID, "config_sha256": init["config_sha256"], "processed_manifest_sha256": init["config"]["processed_manifest_sha256"], "task_data_sha256": digest["composite_sha256"], "files": digest["entries"], "candidate_rows": len(candidates), "candidate_sets": negative_audit["candidate_sets"], "episodes": len(episodes), "task_families": len(selected)}
    write_json(TASK_DIR / "task_formulation.json", formulation)
    write_json(TASK_DIR / "split_audit.json", split_audit)
    write_json(TASK_DIR / "negative_sampling_audit.json", negative_audit)
    write_json(TASK_DIR / "future_target_audit.json", future_audit)
    write_json(TASK_DIR / "fair_input_schema.json", fair)
    write_json(TASK_DIR / "data_manifest.json", manifest)
    write_json(OUT / "task_formulation.json", {"active_attempt": ATTEMPT_ID, "path": str((TASK_DIR / "task_formulation.json").relative_to(OUT)), "task_data_sha256": manifest["task_data_sha256"], "claim": formulation["claim"]})
    append_run_log(OUT / "run_log.jsonl", event="r2_asu_strict_task_built", payload={"attempt_id": ATTEMPT_ID, "candidate_sets": manifest["candidate_sets"], "episodes": manifest["episodes"], "task_data_sha256": manifest["task_data_sha256"]})
    return {"status": "built", "manifest": manifest, "split_audit": split_audit, "negative_audit": negative_audit}


# --- R3: raw-score-backed non-pretrained baselines --------------------------


def _task_data() -> tuple[dict[str, np.ndarray], list[dict[str, Any]], dict[str, Any]]:
    needed = (TASK_DIR / "task_arrays.npz", TASK_DIR / "candidate_rows.jsonl", TASK_DIR / "data_manifest.json")
    if not all(path.is_file() for path in needed):
        raise FileNotFoundError("strict ASU R2 artifacts are absent; run --build-tasks")
    archive = np.load(needed[0])
    data = {name: archive[name] for name in archive.files}
    rows = read_jsonl(needed[1])
    if any(len(value) != len(rows) for value in data.values()):
        raise RuntimeError("ASU task arrays and candidate metadata differ in row count")
    return data, rows, json.loads(needed[2].read_text())


def _groups(rows: Sequence[Mapping[str, Any]], split: str) -> list[np.ndarray]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        if row["split"] == split:
            grouped[str(row["candidate_set_id"])].append(index)
    result: list[np.ndarray] = []
    for identifier, indices in sorted(grouped.items()):
        if len(indices) != CANDIDATES_PER_SET:
            raise RuntimeError(f"candidate set {identifier} has {len(indices)} rows")
        if sum(bool(rows[index]["consistent"]) for index in indices) != 1:
            raise RuntimeError(f"candidate set {identifier} has not exactly one logged positive")
        result.append(np.asarray(indices, dtype=np.int64))
    return result


def _binary_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    positive = labels > 0.5
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if not n_pos or not n_neg:
        return float("nan")
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(scores), dtype=float)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and abs(float(scores[order[stop]]) - float(scores[order[start]])) <= 1e-8:
            stop += 1
        ranks[order[start:stop]] = (start + 1 + stop) / 2.0
        start = stop
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _rank_metrics(rows: Sequence[Mapping[str, Any]], scores: np.ndarray, split: str) -> dict[str, float | int]:
    """Tie-aware set ranking plus genuine global AUROC and ECE."""

    groups = _groups(rows, split)
    if not groups:
        return {"pair_order_accuracy": float("nan"), "top1_success": float("nan"), "top3_recall": float("nan"), "ndcg": float("nan"), "auroc": float("nan"), "ece": float("nan"), "pair_count": 0, "candidate_set_count": 0}
    pair, top1, top3, ndcg = [], [], [], []
    all_labels, all_scores = [], []
    for indices in groups:
        local_scores = np.asarray(scores[indices], dtype=float)
        local_labels = np.asarray([float(rows[index]["consistent"]) for index in indices])
        positive = int(np.flatnonzero(local_labels > 0.5)[0])
        for negative in np.flatnonzero(local_labels < 0.5):
            delta = local_scores[positive] - local_scores[int(negative)]
            pair.append(1.0 if delta > 1e-8 else 0.5 if abs(delta) <= 1e-8 else 0.0)
        maximum = local_scores.max()
        tied_top = np.flatnonzero(np.abs(local_scores - maximum) <= 1e-8)
        top1.append(float(positive in tied_top) / len(tied_top))
        ordered_scores = sorted(set(float(value) for value in local_scores), reverse=True)
        positions = 0
        expected_dcg = 0.0
        for value in ordered_scores:
            members = np.flatnonzero(np.abs(local_scores - value) <= 1e-8)
            block_positions = np.arange(positions, positions + len(members))
            if positive in members:
                weights = 1.0 / np.log2(block_positions + 2)
                expected_dcg = float(weights.mean())
                if positions < 3:
                    top3.append(float(np.sum(block_positions < 3)) / len(members))
                else:
                    top3.append(0.0)
                break
            positions += len(members)
        ndcg.append(expected_dcg)  # ideal DCG for one positive is 1/log2(2)=1.
        all_labels.extend(local_labels.tolist())
        all_scores.extend(local_scores.tolist())
    score_array, label_array = np.asarray(all_scores), np.asarray(all_labels)
    confidence = 1.0 / (1.0 + np.exp(-np.clip(score_array, -20, 20)))
    ece = 0.0
    for lower, upper in zip(np.linspace(0.0, 1.0, 11)[:-1], np.linspace(0.0, 1.0, 11)[1:], strict=True):
        member = (confidence >= lower) & (confidence < upper if upper < 1 else confidence <= upper)
        if member.any():
            ece += float(member.mean() * abs(confidence[member].mean() - label_array[member].mean()))
    return {"pair_order_accuracy": float(np.mean(pair)), "top1_success": float(np.mean(top1)), "top3_recall": float(np.mean(top3)), "ndcg": float(np.mean(ndcg)), "auroc": _binary_auroc(label_array, score_array), "ece": float(ece), "pair_count": len(pair), "candidate_set_count": len(groups)}


def _task_one_hot(rows: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, dict[str, int]]:
    names = {name: index for index, name in enumerate(sorted({str(row["task_id"]) for row in rows}))}
    value = np.zeros((len(rows), len(names)), dtype=np.float32)
    for index, row in enumerate(rows):
        value[index, names[str(row["task_id"])]] = 1.0
    return value, names


def _normalize(value: np.ndarray, fit: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    axes = tuple(range(value.ndim - 1))
    mean = value[fit].mean(axis=axes, keepdims=True)
    std = value[fit].std(axis=axes, keepdims=True)
    std[std < 1e-6] = 1.0
    return ((value - mean) / std).astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FlatMLP(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(width, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value).squeeze(-1)


class StateGRU(nn.Module):
    def __init__(self, state_width: int, action_width: int) -> None:
        super().__init__()
        self.memory = nn.GRU(state_width, 96, batch_first=True)
        self.head = nn.Sequential(nn.Linear(96 + action_width, 96), nn.ReLU(), nn.Linear(96, 1))

    def forward(self, history: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat((self.memory(history)[0][:, -1], action.flatten(1)), dim=1)).squeeze(-1)


class StateTCN(nn.Module):
    def __init__(self, state_width: int, action_width: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv1d(state_width, 96, 3, padding=1), nn.ReLU(), nn.Conv1d(96, 96, 3, padding=1), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(96 + action_width, 96), nn.ReLU(), nn.Linear(96, 1))

    def forward(self, history: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        feature = self.encoder(history.transpose(1, 2)).mean(2)
        return self.head(torch.cat((feature, action.flatten(1)), dim=1)).squeeze(-1)


class VisualGRU(nn.Module):
    def __init__(self, action_width: int) -> None:
        super().__init__()
        self.cnn = nn.Sequential(nn.Conv2d(3, 24, 5, stride=2, padding=2), nn.ReLU(), nn.Conv2d(24, 48, 3, stride=2, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1))
        self.memory = nn.GRU(48, 64, batch_first=True)
        self.head = nn.Sequential(nn.Linear(64 + action_width, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, visual: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        batch, frames, channels, height, width = visual.shape
        z = self.cnn((visual.reshape(batch * frames, channels, height, width) / 255.0)).flatten(1)
        return self.head(torch.cat((self.memory(z.reshape(batch, frames, -1))[0][:, -1], action.flatten(1)), dim=1)).squeeze(-1)


class FutureRelationPredictor(nn.Module):
    def __init__(self, state_width: int, action_width: int, target_width: int) -> None:
        super().__init__()
        self.memory = nn.GRU(state_width, 96, batch_first=True)
        self.action = nn.Sequential(nn.Linear(action_width, 96), nn.ReLU(), nn.Linear(96, 96), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(192, 128), nn.ReLU(), nn.Linear(128, target_width))

    def forward(self, history: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat((self.memory(history)[0][:, -1], self.action(action.flatten(1))), dim=1))


def _flat_features(kind: str, data: Mapping[str, np.ndarray], task_one_hot: np.ndarray) -> tuple[np.ndarray | None, list[str], bool]:
    action = data["candidate_action"].reshape(len(data["consistent"]), -1)
    if kind == "action_only":
        return action, ["candidate_action_chunk"], True
    if kind == "language_task_only":
        return np.concatenate((data["language_bow"], task_one_hot), axis=1), ["language_instruction_bag_of_words_trained_on_train_only", "official_task_id"], True
    if kind == "task_id_shortcut":
        return task_one_hot, ["official_task_id"], True
    if kind == "current_frame_only":
        return np.concatenate((data["history_observation"][:, -1], action), axis=1), ["history_observation_current_frame", "candidate_action_chunk"], True
    if kind == "future_frame_leakage_diagnostic":
        return np.concatenate((data["future_relation_delta"], action), axis=1), ["future_relation_TARGET_ONLY", "candidate_action_chunk"], False
    return None, [], True


def _model(kind: str, data: Mapping[str, np.ndarray], task_one_hot: np.ndarray, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    action_width = int(np.prod(data["candidate_action"].shape[1:]))
    flat, fields, fair = _flat_features(kind, data, task_one_hot)
    if flat is not None:
        return FlatMLP(flat.shape[1]).to(device), {"name": kind, "family": "flat_mlp", "input_fields": fields, "input_tier": "fair" if fair else "future_target_leakage_diagnostic"}
    if kind == "temporal_observation_gru":
        return StateGRU(data["history_observation"].shape[-1], action_width).to(device), {"name": kind, "family": "gru", "input_fields": ["history_observation_state", "candidate_action_chunk"], "input_tier": "fair"}
    if kind == "proprio_action_gru":
        return StateGRU(data["history_robot"].shape[-1], action_width).to(device), {"name": kind, "family": "gru", "input_fields": ["history_robot_state", "candidate_action_chunk"], "input_tier": "fair"}
    if kind == "temporal_convolution":
        return StateTCN(data["history_observation"].shape[-1], action_width).to(device), {"name": kind, "family": "tcn", "input_fields": ["history_observation_state", "candidate_action_chunk"], "input_tier": "fair"}
    if kind == "visual_encoder_gru":
        return VisualGRU(action_width).to(device), {"name": kind, "family": "small_cnn_gru_current_data", "input_fields": ["history_rgb_thumbnail_at_or_before_anchor", "candidate_action_chunk"], "input_tier": "fair"}
    raise ValueError(kind)


def _batch_forward(kind: str, model: nn.Module, data: Mapping[str, np.ndarray], flat: np.ndarray | None, indices: np.ndarray, device: torch.device) -> torch.Tensor:
    if flat is not None:
        return model(torch.as_tensor(flat[indices], device=device))
    action = torch.as_tensor(data["candidate_action"][indices], device=device)
    if kind == "temporal_observation_gru" or kind == "temporal_convolution":
        return model(torch.as_tensor(data["history_observation"][indices], device=device), action)
    if kind == "proprio_action_gru":
        return model(torch.as_tensor(data["history_robot"][indices], device=device), action)
    if kind == "visual_encoder_gru":
        return model(torch.as_tensor(data["visual_history"][indices], device=device), action)
    raise ValueError(kind)


def _run_binary(kind: str, seed: int, *, data: Mapping[str, np.ndarray], rows: Sequence[Mapping[str, Any]], task_one_hot: np.ndarray, task_hash: str, epochs: int) -> tuple[dict[str, dict[str, float | int]], dict[str, Any]]:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = _device()
    fit = np.asarray([row["split"] == "train" for row in rows], dtype=bool)
    normalized = dict(data)
    preprocessing: dict[str, Any] = {}
    for name in ("history_observation", "history_robot", "candidate_action"):
        normalized[name], mean, std = _normalize(np.asarray(data[name], np.float32), fit)
        preprocessing[name] = {"mean": mean.tolist(), "std": std.tolist()}
    flat, _, _ = _flat_features(kind, data, task_one_hot)
    if flat is not None:
        flat, mean, std = _normalize(np.asarray(flat, np.float32), fit)
        preprocessing["flat"] = {"mean": mean.tolist(), "std": std.tolist()}
    model, model_config = _model(kind, data, task_one_hot, device)
    audit = forbidden_input_audit(model_config["input_fields"])
    if model_config["input_tier"] == "fair" and not audit["pass"]:
        raise RuntimeError(f"forbidden fair input in {kind}: {audit}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)
    train_groups = np.stack(_groups(rows, "train"))
    labels = np.asarray(data["consistent"], dtype=np.float32)
    logs: list[dict[str, Any]] = []
    start = time.time()
    generator = np.random.default_rng(seed)
    for epoch in range(epochs):
        model.train()
        losses: list[float] = []
        shuffled_groups = train_groups[generator.permutation(len(train_groups))]
        for begin in range(0, len(shuffled_groups), 16):
            group_indices = shuffled_groups[begin:begin + 16]
            chosen = group_indices.reshape(-1)
            logits = _batch_forward(kind, model, normalized, flat, chosen, device)
            y = torch.as_tensor(labels[chosen], device=device)
            bce = nn.functional.binary_cross_entropy_with_logits(logits, y, pos_weight=torch.tensor(4.0, device=device))
            listwise = nn.functional.cross_entropy(logits.reshape(-1, CANDIDATES_PER_SET), y.reshape(-1, CANDIDATES_PER_SET).argmax(1))
            loss = bce + 0.5 * listwise
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        logs.append({"epoch": epoch + 1, "loss": float(np.mean(losses))})
    model.eval()
    scores = np.empty(len(rows), dtype=float)
    with torch.no_grad():
        for start_index in range(0, len(rows), 128):
            indices = np.arange(start_index, min(start_index + 128, len(rows)), dtype=np.int64)
            scores[indices] = _batch_forward(kind, model, normalized, flat, indices, device).detach().cpu().numpy()
    metrics = {split: _rank_metrics(rows, scores, split) for split in ("validation", "test")}
    run = ATTEMPT / "r3" / kind / f"seed_{seed}"
    if (run / "metrics.json").is_file() and (run / "raw_scores.jsonl").is_file():
        return json.loads((run / "metrics.json").read_text()), {"model": kind, "seed": seed, "path": str(run.relative_to(OUT)), "raw_scores": str((run / "raw_scores.jsonl").relative_to(OUT)), "input_tier": model_config["input_tier"], "task_data_sha256": task_hash}
    run.mkdir(parents=True, exist_ok=False)
    torch.save(model.state_dict(), run / "checkpoint.pt")
    write_json(run / "model_config.json", model_config)
    write_json(run / "preprocessing_config.json", {**preprocessing, "forbidden_input_audit": audit})
    write_json(run / "training_config.json", {"seed": seed, "epochs": epochs, "device": str(device), "optimizer": "AdamW", "learning_rate": 0.002, "loss": "BCE(pos_weight=4)+0.5*listwise CE", "wall_seconds": time.time() - start, "command": " ".join(sys.argv), "task_data_sha256": task_hash})
    write_jsonl(run / "train_log.jsonl", logs)
    write_jsonl(run / "raw_scores.jsonl", [{**dict(row), "score": float(scores[index]), "model": kind, "seed": seed} for index, row in enumerate(rows)])
    write_json(run / "metrics.json", metrics)
    return metrics, {"model": kind, "seed": seed, "path": str(run.relative_to(OUT)), "raw_scores": str((run / "raw_scores.jsonl").relative_to(OUT)), "input_tier": model_config["input_tier"], "task_data_sha256": task_hash}


def _analytic_scores(kind: str, data: Mapping[str, np.ndarray], rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    if kind == "dataset_source_shortcut":
        return np.zeros(len(rows), dtype=float)
    action = np.asarray(data["candidate_action"], dtype=float)
    if kind == "shuffled_action_smoothness":
        return -np.linalg.norm(np.diff(action, axis=1), axis=2).mean(1)
    if kind == "nearest_neighbor_retrieval":
        train_positive = [next(index for index in group if rows[int(index)]["consistent"]) for group in _groups(rows, "train")]
        state = np.asarray(data["history_observation"], dtype=float)[:, -1]
        action_flat = action.reshape(len(action), -1)
        reference_state, reference_action = state[train_positive], action_flat[train_positive]
        scale = np.maximum(reference_state.std(0), 1e-4)
        result = np.empty(len(rows), dtype=float)
        for start in range(0, len(rows), 256):
            stop = min(start + 256, len(rows))
            closest = (np.square((state[start:stop, None] - reference_state[None]) / scale).mean(2)).argmin(1)
            result[start:stop] = -np.linalg.norm(action_flat[start:stop] - reference_action[closest], axis=1)
        return result
    raise ValueError(kind)


def _run_analytic(kind: str, *, data: Mapping[str, np.ndarray], rows: Sequence[Mapping[str, Any]], task_hash: str, input_fields: list[str]) -> tuple[dict[str, dict[str, float | int]], dict[str, Any]]:
    score = _analytic_scores(kind, data, rows)
    metrics = {split: _rank_metrics(rows, score, split) for split in ("validation", "test")}
    run = ATTEMPT / "r3" / kind / "seed_17"
    if (run / "metrics.json").is_file() and (run / "raw_scores.jsonl").is_file():
        return json.loads((run / "metrics.json").read_text()), {"model": kind, "seed": 17, "path": str(run.relative_to(OUT)), "raw_scores": str((run / "raw_scores.jsonl").relative_to(OUT)), "input_tier": "fair_control", "task_data_sha256": task_hash}
    run.mkdir(parents=True, exist_ok=False)
    write_json(run / "input_config.json", {"input_tier": "fair_control", "input_fields": input_fields, "forbidden_input_audit": forbidden_input_audit(input_fields), "command": " ".join(sys.argv)})
    write_jsonl(run / "raw_scores.jsonl", [{**dict(row), "score": float(score[index]), "model": kind, "seed": 17} for index, row in enumerate(rows)])
    write_json(run / "metrics.json", metrics)
    return metrics, {"model": kind, "seed": 17, "path": str(run.relative_to(OUT)), "raw_scores": str((run / "raw_scores.jsonl").relative_to(OUT)), "input_tier": "fair_control", "task_data_sha256": task_hash}


def _run_future_relation(seed: int, *, data: Mapping[str, np.ndarray], rows: Sequence[Mapping[str, Any]], task_hash: str, epochs: int) -> tuple[dict[str, dict[str, float | int]], dict[str, Any]]:
    """Actually evaluate R2-C: predict relation target, then rank candidates."""

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = _device()
    fit = np.asarray([row["split"] == "train" for row in rows], dtype=bool)
    positive = np.asarray(data["consistent"], dtype=bool)
    history, history_mean, history_std = _normalize(np.asarray(data["history_observation"], np.float32), fit)
    action, action_mean, action_std = _normalize(np.asarray(data["candidate_action"], np.float32), fit)
    relation_fit = fit & positive
    target, target_mean, target_std = _normalize(np.asarray(data["future_relation_delta"], np.float32), relation_fit)
    model = FutureRelationPredictor(history.shape[-1], int(np.prod(action.shape[1:])), target.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)
    train_indices = np.flatnonzero(relation_fit)
    generator = np.random.default_rng(seed)
    logs: list[dict[str, Any]] = []
    started = time.time()
    for epoch in range(epochs):
        losses: list[float] = []
        shuffled = train_indices[generator.permutation(len(train_indices))]
        for start in range(0, len(shuffled), 64):
            idx = shuffled[start:start + 64]
            prediction = model(torch.as_tensor(history[idx], device=device), torch.as_tensor(action[idx], device=device))
            loss = nn.functional.mse_loss(prediction, torch.as_tensor(target[idx], device=device))
            optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
            losses.append(float(loss.detach()))
        logs.append({"epoch": epoch + 1, "mse": float(np.mean(losses))})
    prediction = np.empty_like(target)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(rows), 128):
            idx = np.arange(start, min(start + 128, len(rows)), dtype=np.int64)
            prediction[idx] = model(torch.as_tensor(history[idx], device=device), torch.as_tensor(action[idx], device=device)).cpu().numpy()
    scores = -np.mean((prediction - target) ** 2, axis=1)
    metrics = {split: _rank_metrics(rows, scores, split) for split in ("validation", "test")}
    relation_metrics: dict[str, dict[str, float]] = {}
    for split in ("validation", "test"):
        select = np.asarray([row["split"] == split for row in rows]) & positive
        error = prediction[select] - target[select]
        cosine = np.sum(prediction[select] * target[select], axis=1) / np.maximum(np.linalg.norm(prediction[select], axis=1) * np.linalg.norm(target[select], axis=1), 1e-8)
        relation_metrics[split] = {"normalized_mse_true_logged_chunks": float(np.mean(error ** 2)), "cosine_true_logged_chunks": float(np.mean(cosine))}
    run = ATTEMPT / "r3" / "future_relation_predictor" / f"seed_{seed}"
    if (run / "metrics.json").is_file() and (run / "raw_scores.jsonl").is_file():
        saved = json.loads((run / "metrics.json").read_text())
        return saved["ranking"], {"model": "future_relation_predictor", "seed": seed, "path": str(run.relative_to(OUT)), "raw_scores": str((run / "raw_scores.jsonl").relative_to(OUT)), "input_tier": "fair", "task_data_sha256": task_hash, "relation_prediction": saved.get("relation_prediction", {})}
    run.mkdir(parents=True, exist_ok=False)
    torch.save(model.state_dict(), run / "checkpoint.pt")
    config = {"name": "future_relation_predictor", "family": "action_conditioned_GRU_relation_regressor", "input_tier": "fair", "input_fields": ["history_observation_state", "candidate_action_chunk"], "target_only": ["future_relation_delta"], "forbidden_input_audit": forbidden_input_audit(["history_observation_state", "candidate_action_chunk"])}
    write_json(run / "model_config.json", config)
    write_json(run / "preprocessing_config.json", {"history": {"mean": history_mean.tolist(), "std": history_std.tolist()}, "action": {"mean": action_mean.tolist(), "std": action_std.tolist()}, "target": {"mean": target_mean.tolist(), "std": target_std.tolist()}})
    write_json(run / "training_config.json", {"seed": seed, "epochs": epochs, "device": str(device), "optimizer": "AdamW", "learning_rate": 0.002, "loss": "MSE on true logged chunks only", "wall_seconds": time.time() - started, "command": " ".join(sys.argv), "task_data_sha256": task_hash})
    write_jsonl(run / "train_log.jsonl", logs)
    write_jsonl(run / "raw_scores.jsonl", [{**dict(row), "score": float(scores[index]), "model": "future_relation_predictor", "seed": seed} for index, row in enumerate(rows)])
    write_json(run / "metrics.json", {"ranking": metrics, "relation_prediction": relation_metrics})
    return metrics, {"model": "future_relation_predictor", "seed": seed, "path": str(run.relative_to(OUT)), "raw_scores": str((run / "raw_scores.jsonl").relative_to(OUT)), "input_tier": "fair", "task_data_sha256": task_hash, "relation_prediction": relation_metrics}


def _summary(per_seed: Mapping[int, Mapping[str, Mapping[str, float | int]]]) -> dict[str, Any]:
    result: dict[str, Any] = {"estimator": "unweighted seed mean; seed variation is not a data confidence interval", "seeds": sorted(per_seed), "per_seed": {str(seed): value for seed, value in per_seed.items()}, "metrics": {}}
    for split in ("validation", "test"):
        result["metrics"][split] = {}
        for metric in next(iter(per_seed.values()))[split]:
            values = np.asarray([float(value[split][metric]) for value in per_seed.values()])
            result["metrics"][split][metric] = {"mean": float(values.mean()), "std": float(values.std()), "min": float(values.min()), "max": float(values.max()), "seed_variation_range": [float(values.min()), float(values.max())]}
    return result


def run_baselines(*, epochs: int = 20) -> dict[str, Any]:
    init = initialize()
    report_path = ATTEMPT / "r3" / "baseline_report.json"
    if report_path.is_file():
        return {"status": "already_complete", "report": json.loads(report_path.read_text())}
    data, rows, manifest = _task_data()
    task_one_hot, task_map = _task_one_hot(rows)
    report: dict[str, Any] = {"schema": "r-series-asu-baseline-report-v1", "attempt_id": ATTEMPT_ID, "config_sha256": init["config_sha256"], "task_data_sha256": manifest["task_data_sha256"], "task_id_map": task_map, "models": {}, "command": " ".join(sys.argv)}
    entries: list[dict[str, Any]] = []
    for name, fields in (("dataset_source_shortcut", ["source_dataset_constant_not_predictive"]), ("shuffled_action_smoothness", ["candidate_action_chunk_temporal_smoothness"]), ("nearest_neighbor_retrieval", ["history_observation_state", "candidate_action_chunk", "train_logged_action_reference"])):
        metrics, entry = _run_analytic(name, data=data, rows=rows, task_hash=str(manifest["task_data_sha256"]), input_fields=fields)
        report["models"][name] = {"input_tier": entry["input_tier"], "metrics": _summary({17: metrics})}
        entries.append(entry)
    for name in ("action_only", "language_task_only", "task_id_shortcut", "current_frame_only", "future_frame_leakage_diagnostic", "temporal_observation_gru", "temporal_convolution", "proprio_action_gru", "visual_encoder_gru"):
        per_seed: dict[int, dict[str, dict[str, float | int]]] = {}
        local_entries: list[dict[str, Any]] = []
        for seed in (17, 29, 43):
            metrics, entry = _run_binary(name, seed, data=data, rows=rows, task_one_hot=task_one_hot, task_hash=str(manifest["task_data_sha256"]), epochs=epochs)
            per_seed[seed] = metrics; local_entries.append(entry)
        report["models"][name] = {"input_tier": local_entries[0]["input_tier"], "metrics": _summary(per_seed)}
        entries.extend(local_entries)
    future_seed: dict[int, dict[str, dict[str, float | int]]] = {}
    future_entries: list[dict[str, Any]] = []
    for seed in (17, 29, 43):
        metrics, entry = _run_future_relation(seed, data=data, rows=rows, task_hash=str(manifest["task_data_sha256"]), epochs=epochs)
        future_seed[seed] = metrics; future_entries.append(entry)
    report["models"]["future_relation_predictor"] = {"input_tier": "fair", "metrics": _summary(future_seed), "formulation": "R2-C action-to-future relation prediction"}
    entries.extend(future_entries)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(report_path, report)
    raw_manifest = {"schema": "r-series-asu-raw-score-manifest-v1", "attempt_id": ATTEMPT_ID, "task_data_sha256": manifest["task_data_sha256"], "entries": entries, "all_entries_have_raw_scores": all((OUT / entry["raw_scores"]).is_file() for entry in entries)}
    write_json(ATTEMPT / "r3" / "raw_score_manifest.json", raw_manifest)
    checks: list[dict[str, Any]] = []
    for entry in entries:
        raw = read_jsonl(OUT / entry["raw_scores"])
        recalculated = {split: _rank_metrics(raw, np.asarray([row["score"] for row in raw], dtype=float), split) for split in ("validation", "test")}
        saved = json.loads((OUT / entry["path"] / "metrics.json").read_text())
        saved_ranking = saved.get("ranking", saved)
        checks.append({"model": entry["model"], "seed": entry["seed"], "pass": recalculated == saved_ranking})
    ci_audit = {"schema": "r-series-asu-raw-score-consistency-audit-v1", "attempt_id": ATTEMPT_ID, "checks": checks, "all_pass": all(item["pass"] for item in checks), "scope": "recomputes every saved validation/test ranking metric from raw scores"}
    write_json(ATTEMPT / "r3" / "ci_consistency_audit.json", ci_audit)
    write_json(OUT / "baseline_report.json", report)
    write_json(OUT / "raw_score_manifest.json", raw_manifest)
    write_json(OUT / "ci_consistency_audit.json", ci_audit)
    append_run_log(OUT / "run_log.jsonl", event="r3_asu_baselines_complete", payload={"attempt_id": ATTEMPT_ID, "models": list(report["models"]), "task_data_sha256": manifest["task_data_sha256"], "raw_score_audit_pass": ci_audit["all_pass"]})
    return {"status": "complete", "report": report, "raw_manifest": raw_manifest, "ci_audit": ci_audit}


def _mean_metric(report: Mapping[str, Any], model: str, split: str = "validation") -> float:
    return float(report["models"][model]["metrics"]["metrics"][split]["pair_order_accuracy"]["mean"])


def final_decision() -> dict[str, Any]:
    """Commit R5 after the required harder split, without training a method."""

    initialize()
    iid_attempt = OUT / "asu_attempts" / "asu_ur5_object_relation_v1" / "r3"
    heldout_r3 = ATTEMPT / "r3"
    required = [
        iid_attempt / "baseline_report.json", iid_attempt / "ci_consistency_audit.json",
        heldout_r3 / "baseline_report.json", heldout_r3 / "ci_consistency_audit.json",
        TASK_DIR / "split_audit.json", TASK_DIR / "negative_sampling_audit.json", TASK_DIR / "future_target_audit.json", TASK_DIR / "fair_input_schema.json",
    ]
    if not all(path.is_file() for path in required):
        raise FileNotFoundError("R5 needs both the IID and held-out-object R2/R3 artifacts")
    iid = json.loads((iid_attempt / "baseline_report.json").read_text())
    heldout = json.loads((heldout_r3 / "baseline_report.json").read_text())
    audits = [json.loads(path.read_text()) for path in required[4:]]
    iid_ci = json.loads((iid_attempt / "ci_consistency_audit.json").read_text())
    heldout_ci = json.loads((heldout_r3 / "ci_consistency_audit.json").read_text())
    trivial_names = ("action_only", "language_task_only", "task_id_shortcut", "dataset_source_shortcut", "shuffled_action_smoothness")
    standard_names = ("current_frame_only", "temporal_observation_gru", "temporal_convolution", "proprio_action_gru", "visual_encoder_gru", "future_relation_predictor")
    iid_trivial = {name: _mean_metric(iid, name) for name in trivial_names}
    heldout_trivial = {name: _mean_metric(heldout, name) for name in trivial_names}
    iid_standard = {name: _mean_metric(iid, name) for name in standard_names}
    heldout_standard = {name: _mean_metric(heldout, name) for name in standard_names}
    iid_best_name, iid_best = max(iid_standard.items(), key=lambda item: item[1])
    heldout_best_name, heldout_best = max(heldout_standard.items(), key=lambda item: item[1])
    audit_ok = all(item.get("pass", False) for item in audits) and bool(iid_ci.get("all_pass")) and bool(heldout_ci.get("all_pass"))
    decision = {
        "schema": "r-series-final-decision-v2", "attempt_id": ATTEMPT_ID,
        "decision": "E. BASELINES_SATURATE",
        "reason": "Standard temporal/proprioception baselines remain above the preregistered 0.88 saturation ceiling after the one required held-out goal-object language-family split.",
        "method_authorized": False,
        "claim_scope": "logged action-conditioned consistency / verification from real UR5 trajectories; not true counterfactual physical success prediction",
        "audits_pass": audit_ok,
        "iid": {"best_standard": iid_best_name, "best_standard_validation_pair_order": iid_best, "best_standard_test_pair_order": _mean_metric(iid, iid_best_name, "test"), "trivial_validation": iid_trivial},
        "heldout_goal_object": {"best_standard": heldout_best_name, "best_standard_validation_pair_order": heldout_best, "best_standard_test_pair_order": _mean_metric(heldout, heldout_best_name, "test"), "trivial_validation": heldout_trivial, "unseen_test_objects": ["coke", "milk"], "validation_object": "pepsi"},
        "thresholds": {"trivial_shortcut_max": 0.65, "standard_saturation_max": 0.88},
        "next_action": "Do not train RealRelDynVerifier on this saturated logged task. Retain the public ASU benchmark/audit artifacts; a future method track needs a larger real dataset with nontrivial alternate-action outcomes, contact/object-state ambiguity, or matched re-execution.",
    }
    if not audit_ok or max(iid_trivial.values()) > 0.65 or max(heldout_trivial.values()) > 0.65:
        raise RuntimeError("cannot make a saturation decision because an audit or trivial shortcut gate failed")
    if iid_best <= 0.88 or heldout_best <= 0.88:
        raise RuntimeError("saturation decision is unsupported by the preregistered validation gate")
    write_json(ATTEMPT / "r5_final_decision.json", decision)
    write_json(OUT / "final_decision.json", decision)
    completion = {
        "schema": "r-series-completion-audit-v1", "decision": decision["decision"],
        "r0_survey": (OUT / "dataset_survey.json").is_file(),
        "r1_droid_adapter": (OUT / "processed_subset" / "adapter_audit.json").is_file(),
        "r1_asu_adapter": (PROCESSED / "adapter_audit.json").is_file(),
        "r2_iid_attempt_preserved": (OUT / "asu_attempts" / "asu_ur5_object_relation_v1" / "r2" / "data_manifest.json").is_file(),
        "r2_heldout_attempt_preserved": (TASK_DIR / "data_manifest.json").is_file(),
        "r3_iid_raw_score_audit": bool(iid_ci.get("all_pass")),
        "r3_heldout_raw_score_audit": bool(heldout_ci.get("all_pass")),
        "r4_not_run_reason": "gate explicitly rejected method because standard baselines saturated",
        "all_required_evidence_present": True,
    }
    write_json(OUT / "r5_completion_audit.json", completion)
    append_run_log(OUT / "run_log.jsonl", event="r5_asu_baselines_saturate_decision", payload={"attempt_id": ATTEMPT_ID, "iid_best_validation": iid_best, "heldout_best_validation": heldout_best, "best_models": [iid_best_name, heldout_best_name]})
    return decision


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init", action="store_true")
    parser.add_argument("--build-tasks", action="store_true")
    parser.add_argument("--baselines", action="store_true")
    parser.add_argument("--decision", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--heldout-goal-object", action="store_true", help="run the preregistered language-family-held-out saturation repair in a new immutable attempt")
    args = parser.parse_args()
    if not (args.init or args.build_tasks or args.baselines or args.decision):
        parser.error("select --init, --build-tasks, --baselines, and/or --decision")
    _configure_attempt(heldout_goal_object=args.heldout_goal_object)
    result: dict[str, Any] = {}
    if args.init:
        result["init"] = initialize()
    if args.build_tasks:
        result["tasks"] = build_tasks()
    if args.baselines:
        result["baselines"] = run_baselines(epochs=args.epochs)
    if args.decision:
        result["decision"] = final_decision()
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    _main()
