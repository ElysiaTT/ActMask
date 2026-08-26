"""V3 immutable unified audit table from frozen N1--N8 artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .common import (
    HISTORICAL,
    OUT,
    append_log,
    read_json,
    read_jsonl,
    sha256_file,
    write_json,
    write_jsonl,
)
from .preregister import FAMILIES


PILOT = HISTORICAL / "rm_acv_pilot166"
REAL = HISTORICAL / "rm_acv_realneg_v1"


def _action_hash(value: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(value, dtype=np.float32).tobytes()).hexdigest()


def _source_lookup() -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    rows = read_jsonl(PILOT / "source_manifest.jsonl")
    by_episode = {
        row["episode_id_provenance_only"]: row
        for row in rows
    }
    by_safe = {row["safe_episode_key"]: row for row in rows}
    return by_episode, {
        key: row["language_instruction"] for key, row in by_safe.items()
    }


def _base_row(
    *,
    sample_id: str,
    anchor: dict[str, Any],
    family: str,
    label: int,
    action_id: str,
    action_sha256: str,
    source_anchor: dict[str, Any],
    source_episode: str,
    manifest_order: int,
    imported_artifacts: dict[str, str],
    pair: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source_split = source_anchor["episode_split"]
    return {
        "schema": "vsa-unified-audit-sample-v1",
        "sample_id": sample_id,
        "anchor_id": anchor["anchor_id"],
        "anchor_array_index": int(anchor["array_index"]),
        "context_id": anchor["anchor_id"],
        "action_id": action_id,
        "action_sha256": action_sha256,
        "task": anchor["task_id_provenance_only"],
        "episode": anchor["episode_id_provenance_only"],
        "family": family,
        "family_group": FAMILIES[family],
        "construction_label": int(label),
        "label_semantics": (
            "logged continuation construction label"
            if label == 1
            else "historical negative-construction label"
        ),
        "physical_ground_truth": False,
        "context_history": {
            "array": "unified_audit_arrays.npz#history_state",
            "history_steps": 8,
            "state_dimension": 32,
            "causal": True,
        },
        "candidate_action_chunk": {
            "array": "unified_audit_arrays.npz#candidate_action",
            "horizon": 16,
            "action_dimension": 32,
            "values_modified": False,
        },
        "source_anchor_id": source_anchor["anchor_id"],
        "source_episode": source_episode,
        "source_episode_split": source_split,
        "source_trajectory_length": int(source_anchor["episode_steps"]),
        "normalized_progress": float(anchor["normalized_progress"]),
        "episode_duration_seconds": float(anchor["episode_duration_seconds"]),
        "visual_source": {
            "array": "unified_audit_arrays.npz#visual_history",
            "camera": "observation.rgb_images.camera_front",
            "feature": "RGB mean/std plus 4x4 luminance map",
            "causal": True,
        },
        "split_group": {
            "episode_split": anchor["episode_split"],
            "context_episode": anchor["episode_id_provenance_only"],
            "source_episode": source_episode,
            "task": anchor["task_id_provenance_only"],
            "source_safe_for_split": source_split == anchor["episode_split"],
        },
        "provenance": {
            "historical_branch": (
                "rm_acv_realneg_v1" if FAMILIES[family] == "B"
                else "rm_acv_pilot166"
            ),
            "manifest_order": manifest_order,
            "negative_generator": family,
            "source_policy": "logged_demonstration",
            "camera_identity": "camera_front",
            "scene_environment_identifier": None,
            "pair_id": pair["pair_id"] if pair else None,
            "pair_mining_rank": pair["pair_selection_rank"] if pair else None,
            "matching_distances_audit_only": (
                pair["matching_distances"] if pair else None
            ),
            "imported_source_hashes": imported_artifacts,
        },
    }


def build() -> dict[str, Any]:
    anchors = read_jsonl(PILOT / "anchor_manifest.jsonl")
    anchors.sort(key=lambda row: int(row["array_index"]))
    if len(anchors) != 992 or any(
        int(row["array_index"]) != index for index, row in enumerate(anchors)
    ):
        raise RuntimeError("frozen anchor manifest is not contiguous 0..991")
    anchor_by_id = {row["anchor_id"]: row for row in anchors}
    source_by_episode, instruction_by_safe = _source_lookup()
    with np.load(PILOT / "benchmark_arrays.npz") as stored:
        pilot_arrays = {name: stored[name].copy() for name in stored.files}
    with np.load(REAL / "causal_visual_features.npz") as stored:
        visual_unique = stored["visual"].copy().astype(np.float32)
    if visual_unique.shape != (992, 8, 22):
        raise RuntimeError(f"unexpected visual shape {visual_unique.shape}")
    positive_rows = {
        row["anchor_id"]: row
        for row in read_jsonl(PILOT / "positive_candidate_manifest.jsonl")
    }
    negative_rows = read_jsonl(PILOT / "negative_candidate_manifest.jsonl")
    imported_hashes = {
        "pilot_benchmark_arrays": sha256_file(PILOT / "benchmark_arrays.npz"),
        "pilot_anchor_manifest": sha256_file(PILOT / "anchor_manifest.jsonl"),
        "pilot_positive_manifest": sha256_file(
            PILOT / "positive_candidate_manifest.jsonl"
        ),
        "pilot_negative_manifest": sha256_file(
            PILOT / "negative_candidate_manifest.jsonl"
        ),
        "real_N7_manifest": sha256_file(REAL / "n7_pair_manifest.jsonl"),
        "real_N8_manifest": sha256_file(REAL / "n8_pair_manifest.jsonl"),
        "real_visual_features": sha256_file(REAL / "causal_visual_features.npz"),
    }
    rows: list[dict[str, Any]] = []
    context_indices: list[int] = []
    source_indices: list[int] = []
    actions: list[np.ndarray] = []
    labels: list[int] = []
    families: list[str] = []
    action_hash_failures = []

    def add(
        row: dict[str, Any],
        context_index: int,
        source_index: int,
        action: np.ndarray,
    ) -> None:
        actual = _action_hash(action)
        if actual != row["action_sha256"]:
            action_hash_failures.append(
                {
                    "sample_id": row["sample_id"],
                    "expected": row["action_sha256"],
                    "actual": actual,
                }
            )
        row["unified_array_index"] = len(rows)
        rows.append(row)
        context_indices.append(context_index)
        source_indices.append(source_index)
        actions.append(np.asarray(action, dtype=np.float32))
        labels.append(int(row["construction_label"]))
        families.append(row["family"])

    for negative in sorted(
        negative_rows,
        key=lambda row: (row["family_index"], row["anchor_array_index"]),
    ):
        context_index = int(negative["anchor_array_index"])
        anchor = anchors[context_index]
        family = negative["family"]
        family_index = int(negative["family_index"])
        source_anchor_id = negative["source_anchor_id_provenance_only"]
        source_anchor = anchor_by_id[source_anchor_id]
        source_index = int(source_anchor["array_index"])
        source_episode = negative["source_episode_provenance_only"]
        positive = positive_rows[anchor["anchor_id"]]
        positive_action = pilot_arrays["positive_action"][context_index]
        negative_action = pilot_arrays["negative_action_all"][
            context_index, family_index
        ]
        add(
            _base_row(
                sample_id=f"{family}:{anchor['anchor_id']}:positive",
                anchor=anchor,
                family=family,
                label=1,
                action_id=positive["candidate_id"],
                action_sha256=positive["candidate_sha256"],
                source_anchor=anchor,
                source_episode=anchor["episode_id_provenance_only"],
                manifest_order=len(rows),
                imported_artifacts=imported_hashes,
            ),
            context_index,
            context_index,
            positive_action,
        )
        add(
            _base_row(
                sample_id=f"{family}:{anchor['anchor_id']}:negative",
                anchor=anchor,
                family=family,
                label=0,
                action_id=negative["candidate_id"],
                action_sha256=negative["candidate_sha256"],
                source_anchor=source_anchor,
                source_episode=source_episode,
                manifest_order=len(rows),
                imported_artifacts=imported_hashes,
            ),
            context_index,
            source_index,
            negative_action,
        )

    for family, path in (
        (
            "N7_reciprocal_near_state_context_swap",
            REAL / "n7_pair_manifest.jsonl",
        ),
        (
            "N8_visual_substate_reciprocal_context_swap",
            REAL / "n8_pair_manifest.jsonl",
        ),
    ):
        for pair in read_jsonl(path):
            left, right = [int(value) for value in pair["context_indices"]]
            combinations = (
                (left, left, 1, "left_positive", pair["positive_sample_hashes"][0]),
                (right, right, 1, "right_positive", pair["positive_sample_hashes"][1]),
                (left, right, 0, "left_context_right_action", pair["negative_sample_hashes"][0]),
                (right, left, 0, "right_context_left_action", pair["negative_sample_hashes"][1]),
            )
            for context_index, action_index, label, role, sample_hash in combinations:
                anchor = anchors[context_index]
                source_anchor = anchors[action_index]
                action_slot = 0 if action_index == left else 1
                action = pilot_arrays["positive_action"][action_index]
                row = _base_row(
                    sample_id=f"{family}:{pair['pair_id']}:{role}",
                    anchor=anchor,
                    family=family,
                    label=label,
                    action_id=pair["action_ids"][action_slot],
                    action_sha256=pair["action_hashes"][action_slot],
                    source_anchor=source_anchor,
                    source_episode=source_anchor["episode_id_provenance_only"],
                    manifest_order=len(rows),
                    imported_artifacts=imported_hashes,
                    pair=pair,
                )
                row["provenance"]["historical_sample_sha256"] = sample_hash
                add(row, context_index, action_index, action)

    if action_hash_failures:
        raise RuntimeError(f"action hashes failed: {action_hash_failures[:3]}")
    context_index_array = np.asarray(context_indices, dtype=np.int32)
    source_index_array = np.asarray(source_indices, dtype=np.int32)
    family_names = sorted(FAMILIES)
    family_lookup = {name: index for index, name in enumerate(family_names)}
    task_names = sorted({row["task_id_provenance_only"] for row in anchors})
    task_lookup = {name: index for index, name in enumerate(task_names)}
    instructions = pilot_arrays["instruction"].astype(np.float32)
    instruction_text = [
        instruction_by_safe.get(anchor["safe_episode_key"], "")
        for anchor in anchors
    ]
    for row, text in zip(anchors, instruction_text):
        if not text:
            # Text is metadata only; the imported 128D embedding remains authoritative.
            text = row["task_id_provenance_only"].replace("_", " ")
    np.savez_compressed(
        OUT / "unified_audit_arrays.npz",
        history_state=pilot_arrays["history_state"][context_index_array].astype(
            np.float32
        ),
        current_state=pilot_arrays["current_state"][context_index_array].astype(
            np.float32
        ),
        candidate_action=np.stack(actions).astype(np.float32),
        visual_history=visual_unique[context_index_array].astype(np.float32),
        language_embedding=instructions[context_index_array].astype(np.float32),
        normalized_progress=pilot_arrays["progress"][context_index_array].astype(
            np.float32
        ),
        episode_duration=pilot_arrays["episode_duration"][
            context_index_array
        ].astype(np.float32),
        context_episode_index=pilot_arrays["episode_index_internal"][
            context_index_array
        ].astype(np.int32),
        source_episode_index=pilot_arrays["episode_index_internal"][
            source_index_array
        ].astype(np.int32),
        task_index=np.asarray(
            [task_lookup[row["task"]] for row in rows], dtype=np.int16
        ),
        family_index=np.asarray(
            [family_lookup[name] for name in families], dtype=np.int8
        ),
        label=np.asarray(labels, dtype=np.int8),
        context_anchor_index=context_index_array,
        source_anchor_index=source_index_array,
    )
    write_jsonl(OUT / "unified_audit_manifest.jsonl", rows)
    family_label: dict[str, Counter[int]] = defaultdict(Counter)
    split_family: dict[str, Counter[str]] = defaultdict(Counter)
    task_family: dict[str, Counter[str]] = defaultdict(Counter)
    safe_source = Counter()
    for row in rows:
        family_label[row["family"]][row["construction_label"]] += 1
        split_family[row["split_group"]["episode_split"]][row["family"]] += 1
        task_family[row["task"]][row["family"]] += 1
        safe_source[row["split_group"]["source_safe_for_split"]] += 1
    statistics = {
        "schema": "vsa-unified-audit-statistics-v1",
        "samples": len(rows),
        "anchors": len(anchors),
        "tasks": len(task_names),
        "families": family_names,
        "family_label_counts": {
            family: {str(label): count for label, count in sorted(counts.items())}
            for family, counts in sorted(family_label.items())
        },
        "split_family_counts": {
            split: dict(sorted(counts.items()))
            for split, counts in sorted(split_family.items())
        },
        "task_family_counts": {
            task: dict(sorted(counts.items()))
            for task, counts in sorted(task_family.items())
        },
        "source_safe_for_split": {
            str(key).lower(): value for key, value in sorted(safe_source.items())
        },
        "action_values_modified": False,
        "historical_labels_modified": False,
        "physical_ground_truth": False,
    }
    write_json(OUT / "unified_audit_statistics.json", statistics)
    write_json(
        OUT / "unified_audit_hash_manifest.json",
        {
            "schema": "vsa-unified-audit-hash-manifest-v1",
            "imported_sources": imported_hashes,
            "generated": [
                {
                    "path": "unified_audit_manifest.jsonl",
                    "bytes": (OUT / "unified_audit_manifest.jsonl").stat().st_size,
                    "sha256": sha256_file(OUT / "unified_audit_manifest.jsonl"),
                },
                {
                    "path": "unified_audit_arrays.npz",
                    "bytes": (OUT / "unified_audit_arrays.npz").stat().st_size,
                    "sha256": sha256_file(OUT / "unified_audit_arrays.npz"),
                },
            ],
            "action_hashes_recomputed": len(rows),
            "action_hash_failures": [],
        },
    )
    append_log(
        "V3_UNIFIED_AUDIT_TABLE_COMPLETE",
        samples=len(rows),
        anchors=len(anchors),
        families=len(family_names),
        action_hash_failures=0,
        manifest_sha256=sha256_file(OUT / "unified_audit_manifest.jsonl"),
        arrays_sha256=sha256_file(OUT / "unified_audit_arrays.npz"),
    )
    return statistics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.parse_args()
    print(json.dumps(build(), indent=2, sort_keys=True))
