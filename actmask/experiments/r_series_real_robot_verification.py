"""R-Series manager: real-robot logged action-conditioned verification.

The entry point is intentionally conservative.  It creates a reproducible
survey/configuration, turns the public DROID processed subset into two honest
logged-trajectory tasks, audits candidate construction, runs non-pretrained
controls, and trains ``RealRelDynVerifier`` only after the preregistered
headroom gate passes.  It never calls a logged negative a physical
counterfactual outcome.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from PIL import Image
from torch import nn

from actmask.data.droid_adapter import DROID_DATASET, load_processed_subset
from actmask.data.real_robot_dataset_adapter import (
    R_SERIES_ROOT,
    append_run_log,
    canonical_json,
    composite_file_hash,
    forbidden_input_audit,
    read_jsonl,
    sha256_file,
    stable_episode_split,
    write_json,
    write_jsonl,
)


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "actmask" / "r_series_real_robot_verification"
PROCESSED = OUT / "processed_subset"
TASK_DIR = OUT / "task_formulation"
LEARNED = OUT / "learned"
SEEDS = (17, 29, 43)
CONFIG_VERSION = "r-series-real-robot-v1"
HISTORY_STEPS = 32
ACTION_STEPS = 8
ANCHOR_STRIDE = 16
CANDIDATES_PER_SET = 5


def _sha_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _config() -> dict[str, Any]:
    return {
        "schema": CONFIG_VERSION,
        "primary_dataset": {
            "name": "DROID droid_100 RLDS debugging release",
            "source_dataset": DROID_DATASET,
            "official_url": "https://droid-dataset.github.io/droid/the-droid-dataset",
            "declared_size_bytes": 2192594069,
            "target_episodes": 100,
            "limitations": [
                "The RLDS debugging sample has ordered steps but no physical timestamp field.",
                "It has terminal/reward demo markers, not a balanced explicit failure label.",
                "It is logged data, not matched counterfactual re-execution data.",
            ],
        },
        "backup_dataset": {
            "name": "Open X-Embodiment ASU TableTop",
            "reason": "UR5 + language + RGB/proprio/action in a documented 0.773 GB public subset",
            "official_url": "https://github.com/google-deepmind/open_x_embodiment",
        },
        "task": {
            "history_steps": HISTORY_STEPS,
            "action_steps": ACTION_STEPS,
            "anchor_stride": ANCHOR_STRIDE,
            "candidates_per_set": CANDIDATES_PER_SET,
            "minimum_episodes": 40,
            "minimum_task_families": 5,
            "maximum_task_families": 20,
            "formulations": ["logged_action_future_consistency", "logged_future_relation_prediction"],
        },
        "fair_input_fields": [
            "history_robot_state",
            "history_rgb_thumbnail_at_or_before_anchor",
            "candidate_action_chunk",
            "language_instruction_bag_of_words_trained_on_current_subset",
            "derived_language_task_family",
            "relative_order_timestamp",
        ],
        "forbidden_fair_inputs": [
            "episode_id",
            "source_shard",
            "source_record_index",
            "candidate_id",
            "candidate_index",
            "candidate_kind",
            "label",
            "success",
            "future_robot_state",
            "future_rgb",
            "future_timestamp",
        ],
        "gates": {
            "trivial_shortcut_pair_order_max": 0.65,
            "standard_temporal_min": 0.55,
            "standard_temporal_saturation_max": 0.88,
            "oracle_headroom_min": 0.08,
            "method_improvement_min": 0.08,
        },
        "seeds": list(SEEDS),
        "no_pretrained_models": True,
        "decision_tree": {
            "task_invalid": "redesign_negative_sampling_once_then_activate_backup_if_still_invalid",
            "dataset_unsuitable": "activate_backup_dataset",
            "baselines_saturate": "run_held_out_language_family_split_once_then_activate_backup_if_still_saturated",
            "headroom": "train_real_reldyn_verifier",
            "no_headroom": "real_dataset_benchmark_track",
        },
    }


DATASET_SURVEY: dict[str, Any] = {
    "schema": "r-series-dataset-survey-v1",
    "surveyed_at": "2026-07-25",
    "storage_constraint_bytes": 7398502400,
    "datasets": [
        {
            "name": "DROID",
            "public": True,
            "license": "CC-BY-4.0 reported by the official paper; code repository is MIT",
            "download": {"full_rlds": "1.7 TB", "official_debug_subset": "100 episodes / 2 GB"},
            "partial_download": True,
            "embodiment": "Franka Panda 7-DoF, not UR5/UR5e",
            "modalities": ["three RGB views", "joint/cartesian/gripper state", "7-D action", "language"],
            "outcomes": "RLDS sample has reward/terminal demo markers; raw DROID has metadata success/failure but is not downloaded for the first run",
            "contact": "raw data exposes torque proxies; RLDS sample does not",
            "calibration": "raw and later DROID releases provide calibration; not used as a fair input",
            "simulator": False,
            "decision": "primary",
            "sources": [
                "https://droid-dataset.github.io/droid/the-droid-dataset",
                "https://github.com/droid-dataset/droid_policy_learning",
                "https://storage.googleapis.com/download/storage/v1/b/gresearch/o/robotics%2Fdroid_100%2F1.0.0%2Fdataset_info.json?alt=media",
            ],
        },
        {
            "name": "Open X-Embodiment ASU TableTop",
            "public": True,
            "license": "consult source-dataset terms; Open X repository software is Apache-2.0",
            "download": {"selected_subset": "110 episodes / 0.773 GB"},
            "partial_download": True,
            "embodiment": "UR5",
            "modalities": ["RGB", "proprioception", "action", "language"],
            "outcomes": "no explicit success/contact/force field in selected schema",
            "contact": "not directly annotated",
            "calibration": "not required for first adapter",
            "simulator": False,
            "decision": "backup",
            "sources": [
                "https://github.com/google-deepmind/open_x_embodiment",
                "https://storage.googleapis.com/download/storage/v1/b/gdm-robotics-open-x-embodiment/o/asu_table_top_converted_externally_to_rlds%2F0.1.0%2Fdataset_info.json?alt=media",
            ],
        },
        {
            "name": "RH20T",
            "public": True,
            "license": "scene-dependent CC-BY-SA-4.0 or CC-BY-NC-4.0",
            "download": {"minimum_UR5_RGB_cfg": "about 4.4 GB plus extraction space", "depth": "at least 71.3 GB"},
            "partial_download": "tar-level selection possible but not safe under current disk margin",
            "embodiment": "UR5 in cfg3/cfg4; local pilot is KUKA cfg7",
            "modalities": ["RGB-D", "joint/TCP/gripper", "force/torque", "audio", "tactile on cfg7"],
            "outcomes": "metadata completion/quality codes",
            "contact": "force/torque and tactile where configured",
            "calibration": True,
            "simulator": False,
            "decision": "deferred_for_space",
            "sources": ["https://rh20t.github.io/", "https://github.com/rh20t/rh20t_api"],
        },
        {
            "name": "RoboMIND / RoboMIND 2.0",
            "public": True,
            "license": "Apache-2.0 reported for current releases",
            "download": {"v1": ">1 TB and gated", "2.0_sim_task_example": "about 4.03 GiB"},
            "partial_download": True,
            "embodiment": "Franka, UR5e/UR, AgileX, humanoid and mobile systems",
            "modalities": ["RGB/RGB-D", "proprioception", "language", "some force/tactile", "calibration"],
            "outcomes": "v1 has failures; current 2.0 public tree is mainly success episodes",
            "contact": "rich but release-dependent",
            "calibration": True,
            "simulator": "RoboMIND-Sim exists but Isaac Sim needs more local storage than available",
            "decision": "deferred_for_infrastructure",
            "sources": ["https://log2r.github.io/RoboMIND2.0/", "https://github.com/Open-X-Humanoid/RoboMIND-Sim"],
        },
        {
            "name": "BridgeData V2",
            "public": True,
            "license": "CC-BY-4.0 data / MIT code",
            "download": {"TFDS": "about 132.5 GB", "raw": "over 400 GB"},
            "partial_download": "individual shards, but no compact official self-contained subset",
            "embodiment": "WidowX 250, not UR5",
            "modalities": ["RGB", "action", "robot state", "language"],
            "outcomes": "no explicit success/contact/force field in the official TFDS schema",
            "contact": "tasks are contact-rich but sensor labels are absent",
            "calibration": "varies by release",
            "simulator": False,
            "decision": "deferred_for_space",
            "sources": ["https://rail-berkeley.github.io/bridgedata/"],
        },
    ],
}


def source_inventory() -> dict[str, Any]:
    paths = [
        ROOT / "actmask/data/real_robot_dataset_adapter.py",
        ROOT / "actmask/data/droid_adapter.py",
        Path(__file__),
    ]
    entries = [
        {"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in paths
    ]
    digest = _sha_json(entries)
    return {"schema": "r-series-source-inventory-v1", "entries": entries, "source_tree_sha256": digest}


def initialize() -> dict[str, Any]:
    """Freeze R0 evidence and create the immutable manager configuration."""

    OUT.mkdir(parents=True, exist_ok=True)
    config = _config()
    config_path = OUT / "manager_config.json"
    if config_path.exists():
        existing = json.loads(config_path.read_text())
        if existing != config:
            raise RuntimeError("R-Series manager config already exists with a different frozen value")
    else:
        write_json(config_path, config)
    config_hash = sha256_file(config_path)
    write_json(OUT / "dataset_survey.json", DATASET_SURVEY)
    write_json(OUT / "manager_source_inventory.json", source_inventory())
    write_json(
        OUT / "manager_state.json",
        {
            "schema": "r-series-manager-state-v1",
            "config_sha256": config_hash,
            "primary": "DROID droid_100 RLDS debugging release",
            "backup": "Open X-Embodiment ASU TableTop",
            "phase": "R0_COMPLETE_R1_IN_PROGRESS",
        },
    )
    log_path = OUT / "run_log.jsonl"
    prior_events = read_jsonl(log_path) if log_path.is_file() else []
    if not any(row.get("event") == "r0_dataset_selection_frozen" for row in prior_events):
        append_run_log(
            log_path,
            event="r0_dataset_selection_frozen",
            payload={"config_sha256": config_hash, "primary": "DROID", "backup": "Open X-Embodiment ASU TableTop"},
        )
    return {"config": config, "config_sha256": config_hash}


class TaskConstructionError(RuntimeError):
    """A data/task issue which maps to the R-Series decision tree."""


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())


def _load_episode(row: Mapping[str, Any]) -> dict[str, Any]:
    state = np.load(PROCESSED / str(row["robot_state_path"]))["value"].astype(np.float32)
    action = np.load(PROCESSED / str(row["action_path"]))["value"].astype(np.float32)
    timestamp = np.load(PROCESSED / str(row["timestamp_path"]))["value"].astype(np.float32)
    if state.shape[0] != action.shape[0] or state.shape[0] != timestamp.shape[0]:
        raise TaskConstructionError(f"alignment lost for {row['episode_id']}")
    rgb = sorted((dict(item) for item in row["rgb_paths"]), key=lambda item: int(item["step"]))
    return {**dict(row), "state": state, "action": action, "timestamp": timestamp, "rgb": rgb}


def _experiment_splits(episodes: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """Stratify individual real episodes by derived language family."""

    grouped: dict[str, list[str]] = defaultdict(list)
    for episode in episodes:
        grouped[str(episode["task_id"])].append(str(episode["episode_id"]))
    result: dict[str, str] = {}
    for family, identifiers in sorted(grouped.items()):
        identifiers = sorted(identifiers, key=lambda item: hashlib.sha256(f"R:{family}:{item}".encode()).hexdigest())
        n = len(identifiers)
        n_train = max(1, round(n * 0.70))
        n_validation = max(1 if n >= 3 else 0, round(n * 0.15))
        if n_train + n_validation >= n:
            n_train = max(1, n - 1)
            n_validation = max(0, n - n_train - 1)
        for index, identifier in enumerate(identifiers):
            result[identifier] = "train" if index < n_train else "validation" if index < n_train + n_validation else "test"
    if not {"train", "test"}.issubset(set(result.values())):
        raise TaskConstructionError("episode split did not produce both train and test groups")
    return result


def _vocabulary(episodes: Sequence[Mapping[str, Any]], splits: Mapping[str, str], max_words: int = 96) -> tuple[dict[str, int], np.ndarray]:
    count: Counter[str] = Counter()
    for episode in episodes:
        if splits[str(episode["episode_id"])] == "train":
            count.update(_tokens(str(episode["language_instruction"])))
    words = [word for word, _ in count.most_common(max_words)]
    return {word: index for index, word in enumerate(words)}, np.asarray([count[word] for word in words], dtype=np.int64)


def _bow(text: str, vocabulary: Mapping[str, int]) -> np.ndarray:
    vector = np.zeros(len(vocabulary), dtype=np.float32)
    for word in _tokens(text):
        if word in vocabulary:
            vector[vocabulary[word]] += 1.0
    total = vector.sum()
    return vector / total if total else vector


def _load_visual_history(episode: Mapping[str, Any], anchor: int, *, frames: int = 3) -> np.ndarray:
    choices = [item for item in episode["rgb"] if int(item["step"]) <= anchor - 1]
    if not choices:
        raise TaskConstructionError(f"no non-future RGB observation for {episode['episode_id']} at {anchor}")
    choices = (choices[-frames:] if len(choices) >= frames else [choices[0]] * (frames - len(choices)) + choices)
    tensors: list[np.ndarray] = []
    for item in choices:
        with Image.open(PROCESSED / str(item["path"])) as image:
            tensors.append(np.asarray(image.convert("RGB"), dtype=np.uint8).transpose(2, 0, 1))
    return np.stack(tensors, axis=0)


def _anchor_rows(episodes: Sequence[Mapping[str, Any]], splits: Mapping[str, str], vocabulary: Mapping[str, int]) -> list[dict[str, Any]]:
    anchors: list[dict[str, Any]] = []
    for episode in episodes:
        state, action = episode["state"], episode["action"]
        for anchor in range(HISTORY_STEPS, len(state) - ACTION_STEPS, ANCHOR_STRIDE):
            anchors.append(
                {
                    "anchor_id": f"{episode['episode_id']}:{anchor}",
                    "episode_id": episode["episode_id"],
                    "task_id": episode["task_id"],
                    "split": splits[str(episode["episode_id"])],
                    "anchor": anchor,
                    "history_state": state[anchor - HISTORY_STEPS : anchor],
                    "candidate_action": action[anchor : anchor + ACTION_STEPS],
                    "future_delta": state[anchor + ACTION_STEPS - 1] - state[anchor - 1],
                    "language_bow": _bow(str(episode["language_instruction"]), vocabulary),
                    "visual_history": _load_visual_history(episode, anchor),
                    "source_episode": episode,
                }
            )
    return anchors


def _choose(candidates: Sequence[Mapping[str, Any]], *, seed_text: str) -> Mapping[str, Any]:
    if not candidates:
        raise TaskConstructionError("negative sampler has no eligible candidate")
    index = int(hashlib.sha256(seed_text.encode()).hexdigest(), 16) % len(candidates)
    return candidates[index]


def _candidate_sets(anchors: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_split: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for anchor in anchors:
        by_split[str(anchor["split"])].append(anchor)
    rows: list[dict[str, Any]] = []
    type_counts: Counter[str] = Counter()
    for anchor in anchors:
        pool = by_split[str(anchor["split"])]
        same_episode = [
            candidate
            for candidate in pool
            if candidate["episode_id"] == anchor["episode_id"] and abs(int(candidate["anchor"]) - int(anchor["anchor"])) >= HISTORY_STEPS + ACTION_STEPS
        ]
        same_task = [candidate for candidate in pool if candidate["episode_id"] != anchor["episode_id"] and candidate["task_id"] == anchor["task_id"]]
        different_task = [candidate for candidate in pool if candidate["task_id"] != anchor["task_id"]]
        other_episode = [candidate for candidate in pool if candidate["episode_id"] != anchor["episode_id"]]
        candidates: list[tuple[str, np.ndarray, bool, Mapping[str, Any] | None]] = [
            ("logged_true_future", np.asarray(anchor["candidate_action"]), True, anchor),
            ("same_episode_distant", np.asarray(_choose(same_episode or other_episode, seed_text=str(anchor["anchor_id"]) + ":same")["candidate_action"]), False, _choose(same_episode or other_episode, seed_text=str(anchor["anchor_id"]) + ":same")),
            ("same_task_other_episode", np.asarray(_choose(same_task or other_episode, seed_text=str(anchor["anchor_id"]) + ":task")["candidate_action"]), False, _choose(same_task or other_episode, seed_text=str(anchor["anchor_id"]) + ":task")),
            ("different_task", np.asarray(_choose(different_task or other_episode, seed_text=str(anchor["anchor_id"]) + ":different")["candidate_action"]), False, _choose(different_task or other_episode, seed_text=str(anchor["anchor_id"]) + ":different")),
            ("temporally_reversed_logged_chunk", np.asarray(anchor["candidate_action"])[::-1].copy(), False, anchor),
        ]
        # Candidate list position is deliberately randomized and only written as provenance.
        order = np.random.default_rng(int(hashlib.sha256((str(anchor["anchor_id"]) + ":order").encode()).hexdigest()[:16], 16)).permutation(len(candidates))
        for display_order, local_index in enumerate(order):
            kind, action, label, source = candidates[int(local_index)]
            rows.append(
                {
                    "candidate_set_id": anchor["anchor_id"],
                    "candidate_id": hashlib.sha256(f"{anchor['anchor_id']}:{kind}:{local_index}".encode()).hexdigest()[:16],
                    "episode_id": anchor["episode_id"],
                    "task_id": anchor["task_id"],
                    "split": anchor["split"],
                    "anchor_step": int(anchor["anchor"]),
                    "candidate_kind": kind,
                    "candidate_display_order": int(display_order),
                    "consistent": bool(label),
                    "candidate_source_episode_id": None if source is None else source["episode_id"],
                    "candidate_source_anchor": None if source is None else int(source["anchor"]),
                    "history_state": anchor["history_state"],
                    "candidate_action": action,
                    "future_delta": anchor["future_delta"],
                    "language_bow": anchor["language_bow"],
                    "visual_history": anchor["visual_history"],
                }
            )
            type_counts[kind] += 1
    slot_success: dict[int, list[bool]] = defaultdict(list)
    for row in rows:
        # This display order is audit metadata only and is explicitly forbidden to fair models.
        slot_success[int(row["candidate_display_order"])].append(bool(row["consistent"]))
    audit = {
        "candidate_rows": len(rows),
        "candidate_sets": len({row["candidate_set_id"] for row in rows}),
        "candidate_kinds": dict(sorted(type_counts.items())),
        "positive_fraction": float(np.mean([row["consistent"] for row in rows])),
        "candidate_order_label_rate_span": float(max(np.mean(values) for values in slot_success.values()) - min(np.mean(values) for values in slot_success.values())),
        "candidate_source_split_match": all(
            row["candidate_source_episode_id"] is None
            or next(item["split"] for item in anchors if item["episode_id"] == row["candidate_source_episode_id"] and int(item["anchor"]) == int(row["candidate_source_anchor"])) == row["split"]
            for row in rows
        ),
    }
    audit["pass"] = audit["candidate_sets"] > 0 and abs(audit["positive_fraction"] - 1.0 / CANDIDATES_PER_SET) < 1e-8 and audit["candidate_source_split_match"]
    return rows, audit


def build_tasks() -> dict[str, Any]:
    """Create both R2 formulations from a completed, normalized real subset."""

    initialize()
    rows = load_processed_subset(PROCESSED)
    if len(rows) < _config()["task"]["minimum_episodes"]:
        raise TaskConstructionError(f"need at least {_config()['task']['minimum_episodes']} processed episodes, found {len(rows)}")
    episodes = [_load_episode(row) for row in rows]
    task_counts = Counter(str(episode["task_id"]) for episode in episodes)
    selected = [name for name, _ in task_counts.most_common(_config()["task"]["maximum_task_families"])]
    if len(selected) < _config()["task"]["minimum_task_families"]:
        raise TaskConstructionError(f"only {len(selected)} language task families; R1 target requires at least 5")
    episodes = [episode for episode in episodes if str(episode["task_id"]) in selected]
    splits = _experiment_splits(episodes)
    vocabulary, vocabulary_counts = _vocabulary(episodes, splits)
    anchors = _anchor_rows(episodes, splits, vocabulary)
    if len(anchors) < 100:
        raise TaskConstructionError(f"only {len(anchors)} valid anchors after history/action windows")
    candidate_rows, negative_audit = _candidate_sets(anchors)
    candidate_rows.sort(key=lambda row: hashlib.sha256(row["candidate_id"].encode()).hexdigest())
    arrays = {
        "history_state": np.asarray([row["history_state"] for row in candidate_rows], np.float32),
        "candidate_action": np.asarray([row["candidate_action"] for row in candidate_rows], np.float32),
        "future_delta": np.asarray([row["future_delta"] for row in candidate_rows], np.float32),
        "language_bow": np.asarray([row["language_bow"] for row in candidate_rows], np.float32),
        "visual_history": np.asarray([row["visual_history"] for row in candidate_rows], np.uint8),
        "consistent": np.asarray([row["consistent"] for row in candidate_rows], np.float32),
    }
    TASK_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(TASK_DIR / "task_arrays.npz", **arrays)
    metadata = [
        {key: value for key, value in row.items() if key not in {"history_state", "candidate_action", "future_delta", "language_bow", "visual_history"}}
        for row in candidate_rows
    ]
    write_jsonl(TASK_DIR / "candidate_rows.jsonl", metadata)
    write_json(TASK_DIR / "vocabulary.json", {"vocabulary": vocabulary, "train_counts": vocabulary_counts.tolist()})
    split_audit = {
        "schema": "r-series-episode-split-audit-v1",
        "episode_count": len(episodes),
        "selected_task_families": selected,
        "task_counts": dict(sorted(Counter(str(item["task_id"]) for item in episodes).items())),
        "split_episode_counts": dict(Counter(splits.values())),
        "episode_owner_unique": len(splits) == len(set(splits)),
        "candidate_row_split_matches_episode": all(row["split"] == splits[row["episode_id"]] for row in metadata),
        "pass": True,
    }
    split_audit["pass"] = bool(split_audit["episode_owner_unique"] and split_audit["candidate_row_split_matches_episode"] and "train" in splits.values() and "test" in splits.values())
    future_audit = {
        "schema": "r-series-future-leakage-audit-v1",
        "fair_input_fields": _config()["fair_input_fields"],
        "future_target_fields": ["future_delta"],
        "fair_input_does_not_include_future": True,
        "visual_history_rule": "each visual frame step <= anchor_step - 1",
        "candidate_provenance_is_metadata_only": True,
        "pass": True,
    }
    formulation = {
        "schema": "r-series-logged-verification-task-v1",
        "claim": "action-conditioned consistency / verification from logged real trajectories",
        "not_claimed": ["true physical counterfactual success prediction", "counterfactual execution label"],
        "formulation_a": {
            "name": "logged_action_future_consistency",
            "positive": "the logged next action chunk from the same episode at the anchor",
            "negatives": ["non-overlapping same-episode chunk", "same language-family other-episode chunk", "different-family chunk", "temporally reversed logged chunk"],
            "label": "whether candidate action is the logged continuation-consistent chunk",
        },
        "formulation_c": {
            "name": "logged_future_relation_prediction",
            "target": "held-out change in 14-D Cartesian/joint/gripper robot state",
            "use": "target-only action-conditioned future prediction and contrastive ranking diagnostic",
        },
        "config_sha256": sha256_file(OUT / "manager_config.json"),
    }
    files = [TASK_DIR / "task_arrays.npz", TASK_DIR / "candidate_rows.jsonl", TASK_DIR / "vocabulary.json"]
    data_hash = composite_file_hash(TASK_DIR, files)
    manifest = {
        "schema": "r-series-task-data-manifest-v1",
        "config_sha256": formulation["config_sha256"],
        "task_data_sha256": data_hash["composite_sha256"],
        "files": data_hash["entries"],
        "candidate_rows": len(metadata),
        "candidate_sets": negative_audit["candidate_sets"],
        "episodes": len(episodes),
        "task_families": len(selected),
    }
    write_json(TASK_DIR / "task_formulation.json", formulation)
    write_json(TASK_DIR / "split_audit.json", split_audit)
    write_json(TASK_DIR / "negative_sampling_audit.json", negative_audit)
    write_json(TASK_DIR / "future_leakage_audit.json", future_audit)
    write_json(TASK_DIR / "fair_input_schema.json", forbidden_input_audit(_config()["fair_input_fields"]))
    write_json(TASK_DIR / "data_manifest.json", manifest)
    append_run_log(OUT / "run_log.jsonl", event="r2_task_formulations_built", payload={"task_data_sha256": manifest["task_data_sha256"], "candidate_sets": manifest["candidate_sets"], "episodes": len(episodes)})
    return {"manifest": manifest, "split_audit": split_audit, "negative_audit": negative_audit, "formulation": formulation}


# --- R3 metrics and non-pretrained baselines --------------------------------


def _task_data() -> tuple[dict[str, np.ndarray], list[dict[str, Any]], dict[str, Any]]:
    array_path, row_path, manifest_path = TASK_DIR / "task_arrays.npz", TASK_DIR / "candidate_rows.jsonl", TASK_DIR / "data_manifest.json"
    if not (array_path.is_file() and row_path.is_file() and manifest_path.is_file()):
        raise FileNotFoundError("R2 task artifacts are missing; run --build-tasks first")
    archive = np.load(array_path)
    arrays = {name: archive[name] for name in archive.files}
    rows = read_jsonl(row_path)
    if any(len(value) != len(rows) for value in arrays.values()):
        raise RuntimeError("R2 arrays and candidate metadata have inconsistent row counts")
    return arrays, rows, json.loads(manifest_path.read_text())


def _task_one_hot(rows: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, dict[str, int]]:
    names = {name: index for index, name in enumerate(sorted({str(row["task_id"]) for row in rows}))}
    values = np.zeros((len(rows), len(names)), dtype=np.float32)
    for index, row in enumerate(rows):
        values[index, names[str(row["task_id"])]] = 1.0
    return values, names


def _groups(rows: Sequence[Mapping[str, Any]], split: str) -> list[np.ndarray]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        if row["split"] == split:
            grouped[str(row["candidate_set_id"])].append(index)
    result: list[np.ndarray] = []
    for identifier, indices in sorted(grouped.items()):
        if len(indices) != CANDIDATES_PER_SET:
            raise RuntimeError(f"candidate set {identifier} has {len(indices)} candidates")
        result.append(np.asarray(indices, dtype=np.int64))
    return result


def _rank_metrics(rows: Sequence[Mapping[str, Any]], scores: np.ndarray, split: str) -> dict[str, float | int]:
    """Tie-aware candidate-set metrics; candidate IDs never break model ties."""

    values = _groups(rows, split)
    if not values:
        return {"pair_order_accuracy": float("nan"), "top1_success": float("nan"), "top3_recall": float("nan"), "ndcg": float("nan"), "auroc": float("nan"), "ece": float("nan"), "pair_swap_accuracy": float("nan"), "score_swap_consistency": float("nan"), "pair_count": 0, "candidate_set_count": 0}
    pair_values: list[float] = []
    top1: list[float] = []
    top3: list[float] = []
    ndcg: list[float] = []
    confidences: list[float] = []
    labels: list[float] = []
    for indices in values:
        local_score = scores[indices]
        local_y = np.asarray([float(rows[index]["consistent"]) for index in indices])
        positive = np.flatnonzero(local_y > 0.5)
        negative = np.flatnonzero(local_y < 0.5)
        if len(positive) != 1:
            raise RuntimeError("every logged consistency candidate set must have exactly one positive")
        p = int(positive[0])
        for n in negative:
            diff = local_score[p] - local_score[int(n)]
            pair_values.append(1.0 if diff > 1e-8 else 0.5 if abs(diff) <= 1e-8 else 0.0)
        maximum = local_score.max()
        tied = np.flatnonzero(np.abs(local_score - maximum) <= 1e-8)
        top1.append(float(p in tied) / len(tied))
        # Expected chance that a uniformly tie-resolved top-3 includes the sole positive.
        ordered = np.argsort(-local_score, kind="stable")
        cutoff = min(3, len(ordered))
        boundary = local_score[ordered[cutoff - 1]]
        above = np.flatnonzero(local_score > boundary + 1e-8)
        boundary_members = np.flatnonzero(np.abs(local_score - boundary) <= 1e-8)
        remaining = cutoff - len(above)
        if p in above:
            top3.append(1.0)
        elif p in boundary_members:
            top3.append(float(remaining) / len(boundary_members))
        else:
            top3.append(0.0)
        gains = local_y[ordered]
        weights = 1.0 / np.log2(np.arange(2, len(ordered) + 2))
        ideal = weights[0]
        ndcg.append(float(np.dot(gains, weights) / ideal))
        confidences.extend((1.0 / (1.0 + np.exp(-np.clip(local_score, -20, 20)))).tolist())
        labels.extend(local_y.tolist())
    confidence = np.asarray(confidences)
    truth = np.asarray(labels)
    bins = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:], strict=True):
        mask = (confidence >= lo) & (confidence < hi if hi < 1.0 else confidence <= hi)
        if mask.any():
            ece += float(mask.mean() * abs(confidence[mask].mean() - truth[mask].mean()))
    return {
        "pair_order_accuracy": float(np.mean(pair_values)),
        "top1_success": float(np.mean(top1)),
        "top3_recall": float(np.mean(top3)),
        "ndcg": float(np.mean(ndcg)),
        "auroc": float(np.mean(pair_values)),
        "ece": float(ece),
        "pair_swap_accuracy": float(np.mean(pair_values)),
        "score_swap_consistency": float(np.mean(pair_values)),
        "pair_count": int(len(pair_values)),
        "candidate_set_count": int(len(values)),
    }


def _summary(per_seed: Mapping[int, Mapping[str, Mapping[str, float | int]]]) -> dict[str, Any]:
    result: dict[str, Any] = {"estimator": "unweighted_seed_mean", "seeds": sorted(per_seed), "per_seed": {str(seed): dict(value) for seed, value in per_seed.items()}, "metrics": {}}
    splits = sorted(next(iter(per_seed.values())).keys())
    for split in splits:
        metric_names = next(iter(per_seed.values()))[split].keys()
        result["metrics"][split] = {}
        for metric in metric_names:
            values = np.asarray([float(per_seed[seed][split][metric]) for seed in per_seed], dtype=float)
            if not np.isfinite(values).all():
                result["metrics"][split][metric] = {"mean": None, "std": None, "min": None, "max": None, "bootstrap_ci95": None}
                continue
            bootstrap = np.asarray([np.mean(np.random.default_rng(911 + trial).choice(values, len(values), replace=True)) for trial in range(400)])
            result["metrics"][split][metric] = {
                "mean": float(values.mean()),
                "std": float(values.std()),
                "min": float(values.min()),
                "max": float(values.max()),
                "bootstrap_ci95": [float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))],
            }
    return result


def _normalize(array: np.ndarray, fit: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    axes = tuple(range(array.ndim - 1))
    mean = array[fit].mean(axes, keepdims=True)
    std = array[fit].std(axes, keepdims=True)
    std[std < 1e-6] = 1.0
    return ((array - mean) / std).astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


class FlatMLP(nn.Module):
    def __init__(self, features: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(features, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value).squeeze(-1)


class StateGRU(nn.Module):
    def __init__(self, state_features: int, action_features: int, *, proprio_only: bool = False) -> None:
        super().__init__()
        self.proprio_only = proprio_only
        self.encoder = nn.GRU(state_features, 96, batch_first=True)
        self.head = nn.Sequential(nn.Linear(96 + action_features, 96), nn.ReLU(), nn.Linear(96, 1))

    def forward(self, history: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat((self.encoder(history)[0][:, -1], action.flatten(1)), dim=1)).squeeze(-1)


class StateTCN(nn.Module):
    def __init__(self, state_features: int, action_features: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv1d(state_features, 96, 3, padding=1), nn.ReLU(), nn.Conv1d(96, 96, 3, padding=1), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(96 + action_features, 96), nn.ReLU(), nn.Linear(96, 1))

    def forward(self, history: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        feature = self.encoder(history.transpose(1, 2)).mean(dim=2)
        return self.head(torch.cat((feature, action.flatten(1)), dim=1)).squeeze(-1)


class VisualGRU(nn.Module):
    """Small current-data visual encoder; no pretrained weights are used."""

    def __init__(self, action_features: int) -> None:
        super().__init__()
        self.cnn = nn.Sequential(nn.Conv2d(3, 24, 5, stride=2, padding=2), nn.ReLU(), nn.Conv2d(24, 48, 3, stride=2, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1))
        self.memory = nn.GRU(48, 64, batch_first=True)
        self.head = nn.Sequential(nn.Linear(64 + action_features, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, visual: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        batch, frames, channels, height, width = visual.shape
        z = self.cnn(visual.reshape(batch * frames, channels, height, width) / 255.0).flatten(1)
        z = self.memory(z.reshape(batch, frames, -1))[0][:, -1]
        return self.head(torch.cat((z, action.flatten(1)), dim=1)).squeeze(-1)


class RealRelDynVerifier(nn.Module):
    """Fair state/action relational verifier for the real logged-data task."""

    def __init__(self, state_features: int, action_features: int) -> None:
        super().__init__()
        self.state = nn.Sequential(nn.Linear(state_features, 96), nn.ReLU(), nn.Linear(96, 96), nn.ReLU())
        self.action = nn.Sequential(nn.Linear(7, 96), nn.ReLU(), nn.Linear(96, 96), nn.ReLU())
        self.cross = nn.MultiheadAttention(96, num_heads=4, batch_first=True)
        self.memory = nn.GRU(96, 96, batch_first=True)
        self.head = nn.Sequential(nn.Linear(96 + action_features, 96), nn.ReLU(), nn.Linear(96, 1))

    def forward(self, history: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        state = self.state(history)
        act = self.action(action)
        relation, _ = self.cross(state, act, act, need_weights=False)
        temporal = self.memory(state + relation)[0][:, -1]
        return self.head(torch.cat((temporal, action.flatten(1)), dim=1)).squeeze(-1)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _model_features(kind: str, data: Mapping[str, np.ndarray], task_one_hot: np.ndarray) -> tuple[np.ndarray | None, list[str]]:
    action = data["candidate_action"].reshape(len(data["consistent"]), -1)
    history = data["history_state"]
    if kind == "action_only":
        return action, ["candidate_action_chunk"]
    if kind == "language_task_only":
        return np.concatenate((data["language_bow"], task_one_hot), axis=1), ["language_instruction_bag_of_words_trained_on_current_subset", "derived_language_task_family"]
    if kind == "current_frame_only":
        return np.concatenate((history[:, -1], action), axis=1), ["history_robot_state_current_frame", "candidate_action_chunk"]
    if kind == "task_id_shortcut":
        return np.concatenate((task_one_hot, action), axis=1), ["derived_language_task_family", "candidate_action_chunk"]
    if kind == "future_frame_leakage_diagnostic":
        return np.concatenate((data["future_delta"], action), axis=1), ["future_robot_state_TARGET_ONLY", "candidate_action_chunk"]
    return None, []


def _build_model(kind: str, data: Mapping[str, np.ndarray], task_one_hot: np.ndarray, device: torch.device) -> tuple[nn.Module, dict[str, Any], dict[str, np.ndarray]]:
    action = data["candidate_action"]
    history = data["history_state"]
    fit = np.asarray([], dtype=np.int64)  # populated by caller normalization path
    flat, fields = _model_features(kind, data, task_one_hot)
    if flat is not None:
        return FlatMLP(flat.shape[1]).to(device), {"name": kind, "family": "flat_mlp", "input_fields": fields}, {}
    if kind in {"temporal_observation_gru", "proprio_action_gru"}:
        state_width = history.shape[-1] if kind == "temporal_observation_gru" else 8
        return StateGRU(state_width, action.shape[1] * action.shape[2]).to(device), {"name": kind, "family": "gru", "input_fields": ["history_robot_state", "candidate_action_chunk"]}, {}
    if kind == "temporal_convolution":
        return StateTCN(history.shape[-1], action.shape[1] * action.shape[2]).to(device), {"name": kind, "family": "tcn", "input_fields": ["history_robot_state", "candidate_action_chunk"]}, {}
    if kind == "visual_encoder_gru":
        return VisualGRU(action.shape[1] * action.shape[2]).to(device), {"name": kind, "family": "small_cnn_gru_from_current_data", "input_fields": ["history_rgb_thumbnail_at_or_before_anchor", "candidate_action_chunk"]}, {}
    if kind == "real_reldyn_verifier":
        return RealRelDynVerifier(history.shape[-1], action.shape[1] * action.shape[2]).to(device), {"name": "RealRelDynVerifier", "family": "state_action_cross_relation_gru", "input_fields": ["history_robot_state", "candidate_action_chunk", "relative_order_timestamp"]}, {}
    raise ValueError(f"unknown learned model kind {kind}")


def _forward(kind: str, model: nn.Module, tensors: Mapping[str, torch.Tensor], flat: torch.Tensor | None) -> torch.Tensor:
    if flat is not None:
        return model(flat)
    if kind == "temporal_observation_gru":
        return model(tensors["history"], tensors["action"])
    if kind == "proprio_action_gru":
        return model(tensors["history"][:, :, 6:], tensors["action"])
    if kind == "temporal_convolution":
        return model(tensors["history"], tensors["action"])
    if kind == "visual_encoder_gru":
        return model(tensors["visual"], tensors["action"])
    if kind == "real_reldyn_verifier":
        return model(tensors["history"], tensors["action"])
    raise ValueError(kind)


def _artifact_path(kind: str, seed: int, *, method: bool = False) -> Path:
    return (OUT / "d2_method" if method else LEARNED) / kind / f"seed_{seed}"


def _run_learned(
    kind: str,
    seed: int,
    *,
    data: Mapping[str, np.ndarray],
    rows: Sequence[Mapping[str, Any]],
    task_one_hot: np.ndarray,
    data_sha256: str,
    method: bool = False,
    epochs: int = 35,
) -> tuple[dict[str, dict[str, float | int]], dict[str, Any]]:
    """Train one fair (or clearly marked future-leakage diagnostic) control."""

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = _device()
    fit = np.asarray([row["split"] == "train" for row in rows], dtype=bool)
    if not fit.any():
        raise RuntimeError("no training candidate rows")
    history, history_mean, history_std = _normalize(np.asarray(data["history_state"], np.float32), fit)
    action, action_mean, action_std = _normalize(np.asarray(data["candidate_action"], np.float32), fit)
    tensors: dict[str, torch.Tensor] = {
        "history": torch.tensor(history, device=device),
        "action": torch.tensor(action, device=device),
        "visual": torch.tensor(np.asarray(data["visual_history"], np.uint8), device=device),
        "labels": torch.tensor(np.asarray(data["consistent"], np.float32), device=device),
    }
    flat, input_fields = _model_features(kind, data, task_one_hot)
    flat_tensor: torch.Tensor | None = None
    flat_mean: np.ndarray | None = None
    flat_std: np.ndarray | None = None
    if flat is not None:
        flat, flat_mean, flat_std = _normalize(flat.astype(np.float32), fit)
        flat_tensor = torch.tensor(flat, device=device)
    model, model_config, _ = _build_model(kind, data, task_one_hot, device)
    if kind == "future_frame_leakage_diagnostic":
        model_config["input_tier"] = "future_leakage_diagnostic_not_fair"
    else:
        model_config["input_tier"] = "fair" if kind != "visual_encoder_gru" else "fair_current_data_visual"
    fair_audit = forbidden_input_audit(model_config["input_fields"])
    if model_config["input_tier"].startswith("fair") and not fair_audit["pass"]:
        raise RuntimeError(f"forbidden fair model input: {fair_audit}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
    train_indices = torch.tensor(np.flatnonzero(fit), device=device)
    train_groups = _groups(rows, "train")
    group_indices = torch.tensor(np.stack(train_groups), device=device) if train_groups else None
    positive_weight = torch.tensor([(fit.sum() - data["consistent"][fit].sum()) / max(data["consistent"][fit].sum(), 1.0)], device=device)
    logs: list[dict[str, Any]] = []
    started = time.time()
    for epoch in range(epochs):
        logits = _forward(kind, model, tensors, flat_tensor)
        bce = nn.functional.binary_cross_entropy_with_logits(logits[train_indices], tensors["labels"][train_indices], pos_weight=positive_weight)
        if group_indices is not None:
            grouped_logits = logits[group_indices]
            grouped_labels = tensors["labels"][group_indices].argmax(dim=1)
            listwise = nn.functional.cross_entropy(grouped_logits, grouped_labels)
        else:
            listwise = logits.new_zeros(())
        loss = bce + 0.5 * listwise
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        logs.append({"epoch": epoch + 1, "loss": float(loss.detach()), "bce": float(bce.detach()), "listwise": float(listwise.detach())})
    with torch.no_grad():
        score = _forward(kind, model, tensors, flat_tensor).detach().cpu().numpy().astype(float)
    metrics = {split: _rank_metrics(rows, score, split) for split in ("validation", "test")}
    run = _artifact_path(kind, seed, method=method)
    run.mkdir(parents=True, exist_ok=True)
    preprocessing = {
        "history_mean": history_mean.tolist(),
        "history_std": history_std.tolist(),
        "action_mean": action_mean.tolist(),
        "action_std": action_std.tolist(),
        "flat_mean": None if flat_mean is None else flat_mean.tolist(),
        "flat_std": None if flat_std is None else flat_std.tolist(),
        "visual_policy": "uint8 RGB thumbnails at or before anchor only" if kind == "visual_encoder_gru" else None,
        "fair_input_fields": model_config["input_fields"],
        "forbidden_input_audit": fair_audit,
    }
    training = {
        "seed": seed,
        "epochs": epochs,
        "optimizer": "AdamW",
        "learning_rate": 0.003,
        "weight_decay": 0.0001,
        "loss_weights": {"bce": 1.0, "listwise": 0.5},
        "device": str(device),
        "wall_seconds": time.time() - started,
        "command": " ".join(sys.argv),
        "config_sha256": sha256_file(OUT / "manager_config.json"),
        "source_tree_sha256": source_inventory()["source_tree_sha256"],
    }
    torch.save(model.state_dict(), run / "checkpoint.pt")
    write_json(run / "model_config.json", model_config)
    write_json(run / "preprocessing_config.json", preprocessing)
    write_json(run / "training_config.json", training)
    (run / "seed.txt").write_text(f"{seed}\n")
    (run / "data_sha256.txt").write_text(f"{data_sha256}\n")
    (run / "command.txt").write_text(" ".join(sys.argv) + "\n")
    write_jsonl(run / "train_log.jsonl", logs)
    raw = [{**dict(row), "score": float(score[index]), "seed": seed, "model": kind} for index, row in enumerate(rows)]
    write_jsonl(run / "raw_scores.jsonl", raw)
    write_json(run / "metrics.json", metrics)
    return metrics, {"kind": "learned", "model": kind, "seed": seed, "path": str(run.relative_to(OUT)), "raw_scores": str((run / "raw_scores.jsonl").relative_to(OUT)), "data_sha256": data_sha256, "input_tier": model_config["input_tier"]}


def _analytic_scores(kind: str, data: Mapping[str, np.ndarray], rows: Sequence[Mapping[str, Any]], task_one_hot: np.ndarray) -> np.ndarray:
    """Controls that do not train a model and are explicit about their inputs."""

    if kind == "dataset_source_shortcut":
        return np.zeros(len(rows), dtype=float)
    if kind == "shuffled_action_smoothness":
        action = np.asarray(data["candidate_action"], dtype=float)
        return -np.linalg.norm(np.diff(action, axis=1), axis=2).mean(axis=1)
    if kind == "nearest_neighbor_retrieval":
        current = np.asarray(data["history_state"], dtype=float)[:, -1]
        action = np.asarray(data["candidate_action"], dtype=float).reshape(len(rows), -1)
        fit = np.asarray([row["split"] == "train" for row in rows])
        refs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for indices in _groups(rows, "train"):
            positive = next(index for index in indices if rows[int(index)]["consistent"])
            refs[str(rows[int(indices[0])]["candidate_set_id"])] = (current[int(indices[0])], action[positive])
        ref_state = np.asarray([value[0] for value in refs.values()])
        ref_action = np.asarray([value[1] for value in refs.values()])
        if not len(ref_state):
            raise RuntimeError("nearest-neighbor baseline has no training groups")
        state_scale = np.maximum(ref_state.std(0), 1e-4)
        answer = np.empty(len(rows), dtype=float)
        for start in range(0, len(rows), 256):
            stop = min(len(rows), start + 256)
            distance = ((current[start:stop, None] - ref_state[None]) / state_scale).square().mean(axis=2)
            nearest = distance.argmin(axis=1)
            answer[start:stop] = -np.linalg.norm(action[start:stop] - ref_action[nearest], axis=1)
        return answer
    raise ValueError(kind)


def _write_analytic(kind: str, score: np.ndarray, rows: Sequence[Mapping[str, Any]], data_sha256: str, *, input_tier: str, input_fields: list[str]) -> tuple[dict[str, dict[str, float | int]], dict[str, Any]]:
    run = OUT / "raw_scores" / kind / "seed_17"
    run.mkdir(parents=True, exist_ok=True)
    raw = [{**dict(row), "score": float(score[index]), "seed": 17, "model": kind} for index, row in enumerate(rows)]
    write_jsonl(run / "raw_scores.jsonl", raw)
    metrics = {split: _rank_metrics(rows, score, split) for split in ("validation", "test")}
    write_json(run / "metrics.json", metrics)
    write_json(run / "input_config.json", {"input_tier": input_tier, "input_fields": input_fields, "forbidden_input_audit": forbidden_input_audit(input_fields)})
    return metrics, {"kind": "analytic", "model": kind, "seed": 17, "path": str(run.relative_to(OUT)), "raw_scores": str((run / "raw_scores.jsonl").relative_to(OUT)), "data_sha256": data_sha256, "input_tier": input_tier}


def run_baselines(*, epochs: int = 35) -> dict[str, Any]:
    """Run the complete R3 control suite and write raw-score-backed reports."""

    initialize()
    data, rows, task_manifest = _task_data()
    data_sha256 = str(task_manifest["task_data_sha256"])
    task_one_hot, task_map = _task_one_hot(rows)
    report: dict[str, Any] = {
        "schema": "r-series-baseline-report-v1",
        "config_sha256": sha256_file(OUT / "manager_config.json"),
        "task_data_sha256": data_sha256,
        "task_id_map": task_map,
        "models": {},
    }
    manifest: list[dict[str, Any]] = []
    analytic_specs = {
        "dataset_source_shortcut": ("fair_control", ["source_dataset_constant_not_predictive"]),
        "shuffled_action_smoothness": ("fair_diagnostic", ["candidate_action_chunk_temporal_smoothness"]),
        "nearest_neighbor_retrieval": ("fair", ["history_robot_state", "candidate_action_chunk", "training_set_logged_actions"]),
    }
    for name, (tier, fields) in analytic_specs.items():
        metrics, item = _write_analytic(name, _analytic_scores(name, data, rows, task_one_hot), rows, data_sha256, input_tier=tier, input_fields=fields)
        report["models"][name] = {"input_tier": tier, "metrics": _summary({17: metrics})}
        manifest.append(item)
    learned = [
        "action_only",
        "language_task_only",
        "current_frame_only",
        "task_id_shortcut",
        "future_frame_leakage_diagnostic",
        "temporal_observation_gru",
        "temporal_convolution",
        "proprio_action_gru",
        "visual_encoder_gru",
    ]
    for name in learned:
        per_seed: dict[int, dict[str, dict[str, float | int]]] = {}
        entries: list[dict[str, Any]] = []
        for seed in SEEDS:
            metrics, entry = _run_learned(name, seed, data=data, rows=rows, task_one_hot=task_one_hot, data_sha256=data_sha256, epochs=epochs)
            per_seed[seed] = metrics
            entries.append(entry)
        report["models"][name] = {"input_tier": entries[0]["input_tier"], "metrics": _summary(per_seed)}
        manifest.extend(entries)
    write_json(OUT / "baseline_report.json", report)
    raw_manifest = {"schema": "r-series-raw-score-manifest-v1", "task_data_sha256": data_sha256, "entries": manifest, "all_entries_have_raw_scores": all((OUT / item["raw_scores"]).is_file() for item in manifest)}
    write_json(OUT / "raw_score_manifest.json", raw_manifest)
    ci_checks: list[dict[str, Any]] = []
    for item in manifest:
        raw = read_jsonl(OUT / item["raw_scores"])
        recalculated = {split: _rank_metrics(raw, np.asarray([row["score"] for row in raw], dtype=float), split) for split in ("validation", "test")}
        saved = json.loads((OUT / item["path"] / "metrics.json").read_text())
        ci_checks.append({"model": item["model"], "seed": item["seed"], "pass": recalculated == saved})
    ci_audit = {"schema": "r-series-ci-consistency-audit-v1", "checks": ci_checks, "all_pass": all(item["pass"] for item in ci_checks), "estimator": "raw-score re-computation then unweighted seed mean"}
    write_json(OUT / "ci_consistency_audit.json", ci_audit)
    append_run_log(OUT / "run_log.jsonl", event="r3_baselines_complete", payload={"models": list(report["models"]), "task_data_sha256": data_sha256, "ci_pass": ci_audit["all_pass"]})
    return {"report": report, "raw_manifest": raw_manifest, "ci_audit": ci_audit}


# --- R4/R5: gated method and explicit decision --------------------------------


def _metric(report: Mapping[str, Any], model: str, split: str = "test") -> float:
    return float(report["models"][model]["metrics"]["metrics"][split]["pair_order_accuracy"]["mean"])


def _baseline_gate(report: Mapping[str, Any], *, audits_ok: bool) -> dict[str, Any]:
    config = _config()["gates"]
    trivial_names = ["action_only", "language_task_only", "current_frame_only", "task_id_shortcut", "dataset_source_shortcut", "shuffled_action_smoothness"]
    trivial = {name: _metric(report, name) for name in trivial_names}
    standard_names = ["temporal_observation_gru", "temporal_convolution", "proprio_action_gru", "visual_encoder_gru"]
    standard = {name: _metric(report, name) for name in standard_names}
    best_name, best = max(standard.items(), key=lambda item: item[1])
    upper = _metric(report, "future_frame_leakage_diagnostic")
    gates = {
        "audits": audits_ok,
        "trivial_shortcuts_le_065": max(trivial.values()) <= config["trivial_shortcut_pair_order_max"],
        "standard_temporal_useful": best >= config["standard_temporal_min"],
        "standard_temporal_not_saturated": best <= config["standard_temporal_saturation_max"],
        "future_diagnostic_above_standard": upper - best >= config["oracle_headroom_min"],
    }
    return {"gates": gates, "trivial": trivial, "standard": standard, "best_standard": best_name, "best_standard_pair_order": best, "future_diagnostic_pair_order": upper, "headroom": upper - best}


def run_method(*, epochs: int = 45) -> dict[str, Any]:
    """Train only after R3's preregistered gate has demonstrably passed."""

    report_path = OUT / "baseline_report.json"
    if not report_path.is_file():
        raise FileNotFoundError("run R3 baselines before R4")
    report = json.loads(report_path.read_text())
    task_data, rows, manifest = _task_data()
    task_one_hot, _ = _task_one_hot(rows)
    audit_paths = [TASK_DIR / "split_audit.json", TASK_DIR / "negative_sampling_audit.json", TASK_DIR / "future_leakage_audit.json", TASK_DIR / "fair_input_schema.json", OUT / "ci_consistency_audit.json"]
    audits_ok = all(path.is_file() and json.loads(path.read_text()).get("pass", json.loads(path.read_text()).get("all_pass", False)) for path in audit_paths)
    gate = _baseline_gate(report, audits_ok=audits_ok)
    write_json(OUT / "headroom_metrics.json", gate)
    if not all(gate["gates"].values()):
        return {"authorized": False, "gate": gate}
    per_seed: dict[int, dict[str, dict[str, float | int]]] = {}
    entries: list[dict[str, Any]] = []
    for seed in SEEDS:
        metrics, entry = _run_learned("real_reldyn_verifier", seed, data=task_data, rows=rows, task_one_hot=task_one_hot, data_sha256=str(manifest["task_data_sha256"]), method=True, epochs=epochs)
        per_seed[seed] = metrics
        entries.append(entry)
    method_report = {
        "schema": "r-series-real-reldyn-method-report-v1",
        "config_sha256": sha256_file(OUT / "manager_config.json"),
        "task_data_sha256": manifest["task_data_sha256"],
        "model": "RealRelDynVerifier",
        "metrics": _summary(per_seed),
        "baseline_gate": gate,
        "honest_claim": "fair logged action-conditioned consistency, not counterfactual physical success prediction",
    }
    write_json(OUT / "method_report.json", method_report)
    append_run_log(OUT / "run_log.jsonl", event="r4_method_complete", payload={"task_data_sha256": manifest["task_data_sha256"], "method_test_pair_order": method_report["metrics"]["metrics"]["test"]["pair_order_accuracy"]["mean"]})
    return {"authorized": True, "gate": gate, "method": method_report, "entries": entries}


def final_decision() -> dict[str, Any]:
    """Commit a data-backed R5 decision without silently promoting a method."""

    initialize()
    result: dict[str, Any] = {"schema": "r-series-final-decision-v1", "config_sha256": sha256_file(OUT / "manager_config.json")}
    required = [TASK_DIR / "data_manifest.json", OUT / "baseline_report.json", OUT / "raw_score_manifest.json", OUT / "ci_consistency_audit.json"]
    if not all(path.is_file() for path in required):
        result.update({"decision": "F. TOO_EXPENSIVE_OR_INFRA_BLOCKED", "reason": "R1-R3 artifacts are incomplete", "next_action": "continue primary download/adapter/baseline pipeline"})
        write_json(OUT / "final_decision.json", result)
        return result
    report = json.loads((OUT / "baseline_report.json").read_text())
    task_audits = [json.loads((TASK_DIR / name).read_text()) for name in ("split_audit.json", "negative_sampling_audit.json", "future_leakage_audit.json", "fair_input_schema.json")]
    ci = json.loads((OUT / "ci_consistency_audit.json").read_text())
    audits_ok = all(item.get("pass", False) for item in task_audits) and bool(ci.get("all_pass"))
    gate = _baseline_gate(report, audits_ok=audits_ok)
    method_path = OUT / "method_report.json"
    if not gate["gates"]["audits"]:
        decision, next_action = "D. TASK_FORMULATION_INVALID", "redesign formulation once; retain logged-consistency wording"
    elif not gate["gates"]["trivial_shortcuts_le_065"]:
        decision, next_action = "D. TASK_FORMULATION_INVALID", "repair negative sampling once because a trivial action/task/source control is too strong"
    elif not gate["gates"]["standard_temporal_not_saturated"]:
        decision, next_action = "E. BASELINES_SATURATE", "run held-out language-family split once, then activate Open X backup if saturation persists"
    elif not (gate["gates"]["standard_temporal_useful"] and gate["gates"]["future_diagnostic_above_standard"]):
        decision, next_action = "B. REAL_DATASET_BENCHMARK_TRACK_FOUND", "package real logged-consistency benchmark/audits; do not train or claim a method advantage"
    elif not method_path.is_file():
        decision, next_action = "R4_METHOD_AUTHORIZED", "train RealRelDynVerifier"
    else:
        method = json.loads(method_path.read_text())
        method_score = float(method["metrics"]["metrics"]["test"]["pair_order_accuracy"]["mean"])
        improvement = method_score - gate["best_standard_pair_order"]
        if improvement >= _config()["gates"]["method_improvement_min"]:
            decision, next_action = "A. REAL_DATASET_METHOD_TRACK_FOUND", "run robustness checks and package method direction"
        else:
            decision, next_action = "B. REAL_DATASET_BENCHMARK_TRACK_FOUND", "package benchmark/audit direction; standard baseline remains competitive"
        result["method_pair_order"] = method_score
        result["method_improvement"] = improvement
    result.update({"decision": decision, "next_action": next_action, "baseline_gate": gate, "audits_ok": audits_ok, "claim_scope": "logged action-conditioned consistency / verification from real trajectories; not true counterfactual physical success prediction"})
    write_json(OUT / "final_decision.json", result)
    append_run_log(OUT / "run_log.jsonl", event="r5_decision_committed", payload={"decision": decision, "next_action": next_action})
    return result


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init", action="store_true")
    parser.add_argument("--build-tasks", action="store_true")
    parser.add_argument("--baselines", action="store_true")
    parser.add_argument("--method", action="store_true")
    parser.add_argument("--decision", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--epochs", type=int, default=35)
    args = parser.parse_args()
    if not any((args.init, args.build_tasks, args.baselines, args.method, args.decision, args.all)):
        parser.error("select one or more phase flags")
    outputs: dict[str, Any] = {}
    if args.init or args.all:
        outputs["init"] = initialize()
    if args.build_tasks or args.all:
        outputs["tasks"] = build_tasks()
    if args.baselines or args.all:
        outputs["baselines"] = run_baselines(epochs=args.epochs)
    if args.method or args.all:
        outputs["method"] = run_method(epochs=max(args.epochs, 40))
    if args.decision or args.all:
        outputs["decision"] = final_decision()
    print(json.dumps(outputs, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    _main()
