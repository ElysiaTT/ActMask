"""Build an atomic context-group ActionCheck benchmark."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .common import read_jsonl, sha256_file, write_json, write_jsonl
from .create_actioncheck_splits import create_splits
from .validate_actioncheck_schema import validate_group
from .validate_candidate_balance import candidate_balance
from .validate_evidence import validate_evidence
from .validate_group_structure import validate_groups
from .validate_splits import validate_splits


def _matrix(value: Any, history_length: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.size == 0:
        return np.zeros((history_length, 0), dtype=np.float32)
    if array.ndim == 1:
        if len(array) == history_length:
            return array[:, None]
        return np.tile(array[None], (history_length, 1))
    if array.ndim != 2:
        return array.reshape(history_length, -1)
    return array


def _pad_history(
    groups: Sequence[dict[str, Any]],
    key: str,
) -> tuple[np.ndarray, np.ndarray]:
    matrices = [
        _matrix(group["history"].get(key, []), group["history"]["history_length"])
        for group in groups
    ]
    maximum_steps = max(group["history"]["history_length"] for group in groups)
    maximum_width = max((matrix.shape[1] for matrix in matrices), default=0)
    result = np.zeros(
        (len(groups), maximum_steps, maximum_width), dtype=np.float32
    )
    mask = np.zeros((len(groups), maximum_steps), dtype=np.bool_)
    for index, matrix in enumerate(matrices):
        steps = min(matrix.shape[0], maximum_steps)
        result[index, :steps, : matrix.shape[1]] = matrix[:steps]
        mask[index, :steps] = True
    return result, mask


def build_benchmark(
    input_paths: Sequence[str | Path],
    output_dir: str | Path,
    *,
    schema_config: str | Path | None = None,
    split_config: str | Path | None = None,
    metric_config: str | Path | None = None,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    groups: list[dict[str, Any]] = []
    for path in input_paths:
        groups.extend(read_jsonl(path))
    groups.sort(key=lambda row: row["context_id"])
    context_ids = [group["context_id"] for group in groups]
    if len(context_ids) != len(set(context_ids)):
        raise ValueError("context_id values must be globally unique")
    schema_failures = [
        {"context_id": group.get("context_id"), "errors": errors}
        for group in groups
        if (errors := validate_group(group))
    ]
    if schema_failures:
        raise ValueError(f"schema validation failed: {schema_failures[:3]}")
    evidence = validate_evidence(groups)
    if not evidence["pass"]:
        raise ValueError(f"evidence validation failed: {evidence['failures'][:3]}")
    if split_config:
        splits = json.loads(Path(split_config).read_text(encoding="utf-8"))
    else:
        splits = create_splits(groups)
    split_audit = validate_splits(groups, splits)
    if not split_audit["pass"]:
        raise ValueError("split audit failed")
    primary = splits["schemes"][splits["primary_scheme"]]["assignments"]
    state, state_mask = _pad_history(groups, "state")
    visual, visual_mask = _pad_history(groups, "visual_features")
    gripper, gripper_mask = _pad_history(groups, "gripper")
    primary_candidates = [
        (group_index, group, candidate)
        for group_index, group in enumerate(groups)
        for candidate in group["candidates"]
        if candidate["label"] in {"accept", "reject"}
    ]
    maximum_horizon = max(
        candidate["action_horizon"] for _, _, candidate in primary_candidates
    )
    maximum_action_dim = max(
        candidate["action_dim"] for _, _, candidate in primary_candidates
    )
    actions = np.zeros(
        (len(primary_candidates), maximum_horizon, maximum_action_dim),
        dtype=np.float32,
    )
    action_mask = np.zeros(
        (len(primary_candidates), maximum_horizon, maximum_action_dim),
        dtype=np.bool_,
    )
    labels = np.zeros(len(primary_candidates), dtype=np.int8)
    group_indices = np.zeros(len(primary_candidates), dtype=np.int32)
    progress = np.zeros(len(primary_candidates), dtype=np.float32)
    tasks = sorted({group["task_id"] for group in groups})
    task_lookup = {value: index for index, value in enumerate(tasks)}
    sources = sorted(
        {
            candidate["source_policy"]
            for _, _, candidate in primary_candidates
        }
    )
    source_lookup = {value: index for index, value in enumerate(sources)}
    task_index = np.zeros(len(primary_candidates), dtype=np.int16)
    source_index = np.zeros(len(primary_candidates), dtype=np.int16)
    candidate_manifest = []
    grouped_manifest = []
    all_candidate_index = 0
    primary_index = 0
    for group_index, group in enumerate(groups):
        primary_indices = []
        uncertain_ids = []
        for candidate in group["candidates"]:
            base = {
                "schema": "actioncheck-candidate-manifest-v1",
                "candidate_manifest_index": all_candidate_index,
                "context_id": group["context_id"],
                "episode_id": group["episode_id"],
                "task_id": group["task_id"],
                "candidate_id": candidate["candidate_id"],
                "action_hash": candidate["hash"],
                "action_horizon": candidate["action_horizon"],
                "action_dim": candidate["action_dim"],
                "source": candidate["source"],
                "source_policy": candidate["source_policy"],
                "label": candidate["label"],
                "label_source": candidate["label_source"],
                "evidence_type_audit_only": candidate["evidence_type"],
                "reason_codes_audit_only": candidate["reason_codes"],
                "primary_eligible": candidate["label"] in {"accept", "reject"},
                "split_assignments": {
                    name: scheme["assignments"][group["context_id"]]
                    for name, scheme in splits["schemes"].items()
                    if scheme.get("status") == "available"
                },
            }
            if base["primary_eligible"]:
                action = np.asarray(candidate["action_chunk"], dtype=np.float32)
                actions[
                    primary_index, : action.shape[0], : action.shape[1]
                ] = action
                action_mask[
                    primary_index, : action.shape[0], : action.shape[1]
                ] = True
                labels[primary_index] = 1 if candidate["label"] == "accept" else 0
                group_indices[primary_index] = group_index
                task_index[primary_index] = task_lookup[group["task_id"]]
                source_index[primary_index] = source_lookup[
                    candidate["source_policy"]
                ]
                progress[primary_index] = float(
                    candidate["sampling_metadata"].get("normalized_progress", 0.0)
                )
                base["primary_array_index"] = primary_index
                primary_indices.append(primary_index)
                primary_index += 1
            else:
                base["primary_array_index"] = None
                uncertain_ids.append(candidate["candidate_id"])
            candidate_manifest.append(base)
            all_candidate_index += 1
        grouped_manifest.append(
            {
                "schema": "actioncheck-grouped-candidate-manifest-v1",
                "context_id": group["context_id"],
                "episode_id": group["episode_id"],
                "task_id": group["task_id"],
                "candidate_ids": [
                    candidate["candidate_id"] for candidate in group["candidates"]
                ],
                "primary_array_indices": primary_indices,
                "uncertain_candidate_ids": uncertain_ids,
                "primary_split": primary[group["context_id"]],
            }
        )
    np.savez_compressed(
        output / "benchmark_arrays.npz",
        state_history=state,
        state_history_mask=state_mask,
        visual_history=visual,
        visual_history_mask=visual_mask,
        gripper_history=gripper,
        gripper_history_mask=gripper_mask,
        candidate_actions=actions,
        candidate_action_mask=action_mask,
        labels=labels,
        group_index=group_indices,
        task_index=task_index,
        source_index=source_index,
        normalized_progress=progress,
    )
    write_jsonl(output / "processed_contexts.jsonl", groups)
    write_jsonl(output / "candidate_manifest.jsonl", candidate_manifest)
    write_jsonl(output / "grouped_candidate_manifest.jsonl", grouped_manifest)
    write_json(output / "split_manifest.json", splits)
    source_files = [Path(path) for path in input_paths]
    for optional in (schema_config, split_config, metric_config):
        if optional:
            source_files.append(Path(optional))
    source_hashes = [
        {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in source_files
    ]
    write_json(
        output / "source_hash_manifest.json",
        {
            "schema": "actioncheck-source-hash-manifest-v1",
            "files": source_hashes,
        },
    )
    structure = validate_groups(groups, minimum_candidates=4)
    balance = candidate_balance(groups)
    report = {
        "schema": "actioncheck-benchmark-build-report-v1",
        "context_groups": len(groups),
        "all_candidates": len(candidate_manifest),
        "primary_candidates": len(primary_candidates),
        "uncertain_candidates": len(candidate_manifest) - len(primary_candidates),
        "tasks": tasks,
        "candidate_sources": sources,
        "maximum_history_steps": int(state.shape[1]),
        "maximum_state_dimension": int(state.shape[2]),
        "maximum_visual_dimension": int(visual.shape[2]),
        "maximum_action_horizon": maximum_horizon,
        "maximum_action_dimension": maximum_action_dim,
        "atomic_context_groups": True,
        "candidate_level_random_split": False,
        "uncertain_excluded_from_primary": True,
        "reject_labels_synthesized": False,
        "logged_action_swaps_used": False,
        "schema_validation": {"pass": True, "failures": []},
        "evidence_validation": evidence,
        "group_structure": structure,
        "candidate_balance": balance,
        "split_audit": split_audit,
    }
    write_json(output / "benchmark_build_report.json", report)
    (output / "benchmark_build_report.md").write_text(
        "# ActionCheck benchmark build report\n\n"
        f"Built {len(groups)} atomic context groups and {len(primary_candidates)} "
        f"primary accept/reject candidates from {len(input_paths)} source file(s). "
        "Uncertain candidates are retained in manifests but excluded from primary arrays. "
        "No reject labels were synthesized and no candidate-level random split was used.\n",
        encoding="utf-8",
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--schema-config")
    parser.add_argument("--split-config")
    parser.add_argument("--metric-config")
    args = parser.parse_args()
    result = build_benchmark(
        args.input,
        args.output_dir,
        schema_config=args.schema_config,
        split_config=args.split_config,
        metric_config=args.metric_config,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
