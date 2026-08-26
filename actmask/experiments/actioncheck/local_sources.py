"""Read-only inventory and adapters for local proposal-level simulator data."""
from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .common import (
    ROOT,
    action_hash,
    canonical_json,
    sha256_file,
    sha256_json,
)


ACTMASK_OUTPUTS = ROOT / "outputs" / "actmask"
M4R = (
    ACTMASK_OUTPUTS
    / "milestone4r_v3_all_family_candidate_diversity"
    / "full_probe"
)
M5B = (
    ACTMASK_OUTPUTS
    / "milestone5b_data_v3_compositional_ood"
    / "state_probe"
)
M4R_GENERATOR = ROOT / "actmask" / "data" / "milestone4r_robust_pilot.py"
M5B_GENERATOR = ROOT / "actmask" / "data" / "milestone5b_data_v3_probe.py"
M4R_TASKS = (
    "MovingCubeIntercept",
    "SignedMovingWindowPlacement",
    "FixedPhaseRotatingCaptureWindow",
)
M5B_TASKS = (
    "CompositionalPartialCueIdentitySwap",
    "CompositionalIntermittentContactTiming",
)
INSTRUCTIONS = {
    "MovingCubeIntercept": "Intercept the moving cube with the proxy end effector.",
    "SignedMovingWindowPlacement": "Place the payload through the moving signed window.",
    "FixedPhaseRotatingCaptureWindow": "Reach the rotating capture window at the compatible phase.",
    "CompositionalPartialCueIdentitySwap": "Choose an action compatible with the observed target identity and motion.",
    "CompositionalIntermittentContactTiming": "Choose an action compatible with intermittent contact timing.",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _visual_summary(bundle: dict[str, np.ndarray], reference: int) -> list[list[float]]:
    rgb = bundle["rgb"][reference].astype(np.float32) / 255.0
    depth = bundle["depth_mm"][reference].astype(np.float32) / 1000.0
    rgb_mask = bundle["rgb_mask"][reference].astype(np.float32)
    depth_mask = bundle["depth_mask"][reference].astype(np.float32)
    visibility = bundle["visibility"][reference].astype(np.float32)
    confidence = bundle["confidence"][reference].astype(np.float32)
    timestamps = bundle["timestamps"][reference].astype(np.float32)
    features = []
    for frame in range(len(rgb)):
        valid_depth = depth[frame][depth[frame] > 0]
        depth_values = (
            [
                float(valid_depth.mean()),
                float(valid_depth.std()),
                float(valid_depth.min()),
                float(valid_depth.max()),
            ]
            if len(valid_depth)
            else [0.0, 0.0, 0.0, 0.0]
        )
        features.append(
            [
                *rgb[frame].mean(axis=(0, 1)).astype(float).tolist(),
                *rgb[frame].std(axis=(0, 1)).astype(float).tolist(),
                *depth_values,
                float(rgb_mask[frame]),
                float(depth_mask[frame]),
                float(visibility[frame].mean()),
                float(confidence[frame].mean()),
                float(timestamps[frame]),
            ]
        )
    return features


def _sim_candidate(
    *,
    candidate_id: str,
    action: list[list[float]],
    source_policy: str,
    sampling_metadata: dict[str, Any],
    success: bool,
    label_source: str,
    sim_result: dict[str, Any],
) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "action_chunk": action,
        "action_horizon": len(action),
        "action_dim": len(action[0]),
        "source": "simulator_proposal",
        "source_policy": source_policy,
        "sampling_metadata": {
            **sampling_metadata,
            "label_basis": "simulator_execution_outcome",
        },
        "label": "accept" if success else "reject",
        "label_source": label_source,
        "evidence_type": (
            "sim_execution_success" if success else "sim_execution_failure"
        ),
        "reason_codes": [] if success else ["simulator_failure"],
        "execution_result": None,
        "sim_result": {
            **sim_result,
            "success": success,
            "claim_boundary": "simulated compatibility outcome; not physical safety",
        },
        "expert_review": None,
        "rule_check": None,
        "hash": action_hash(action),
    }


def adapt_m4r() -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    generator_hash = sha256_file(M4R_GENERATOR)
    for task in M4R_TASKS:
        rows = _read_jsonl(M4R / f"{task}_candidates.jsonl")
        with np.load(M4R / f"{task}_labels.npz") as stored:
            labels = stored["success"].astype(bool)
        with np.load(M4R / f"{task}_bundles.npz") as stored:
            bundle = {name: stored[name].copy() for name in stored.files}
        if len(rows) != len(labels):
            raise RuntimeError(f"{task}: candidate/label length mismatch")
        by_history: dict[int, list[tuple[dict[str, Any], bool, int]]] = defaultdict(list)
        for row_index, (row, label) in enumerate(zip(rows, labels)):
            by_history[int(row["history_ref"])].append(
                (row, bool(label), row_index)
            )
        for reference, values in sorted(by_history.items()):
            values.sort(key=lambda item: int(item[0]["candidate_slot"]))
            first = values[0][0]
            world = int(first["world_id"])
            branch = reference % 2
            context_id = f"m4r-v3:{task}:world-{world:04d}:branch-{branch}"
            episode_id = f"m4r-v3:{task}:world-{world:04d}"
            visual = _visual_summary(bundle, reference)
            candidates = []
            for row, success, row_index in values:
                candidate_id = (
                    f"{context_id}:candidate-{int(row['candidate_slot']):02d}"
                )
                candidates.append(
                    _sim_candidate(
                        candidate_id=candidate_id,
                        action=row["candidate_actions"],
                        source_policy="milestone4r_v3_candidate_spec_bank",
                        sampling_metadata={
                            "candidate_slot": int(row["candidate_slot"]),
                            "observation_condition": row["condition"],
                            "normalized_progress": 0.0,
                        },
                        success=success,
                        label_source="ManiSkill 3 GPU PhysX candidate rollout",
                        sim_result={
                            "simulator": "ManiSkill 3 GPU PhysX",
                            "source_candidate_row": row_index,
                            "source_history_ref": reference,
                            "source_world_id": world,
                            "execution_generator_sha256": generator_hash,
                            "executed_action_horizon": len(row["candidate_actions"]),
                        },
                    )
                )
            groups.append(
                {
                    "context_id": context_id,
                    "task_id": task,
                    "task_instruction": INSTRUCTIONS[task],
                    "episode_id": episode_id,
                    "timestamp": 0.0,
                    "history": {
                        "rgb_refs": [
                            f"{(M4R / f'{task}_bundles.npz').resolve()}#rgb[{reference},{frame}]"
                            for frame in range(6)
                        ],
                        "state": [],
                        "gripper": [],
                        "action_history": None,
                        "visual_features": visual,
                        "history_length": 6,
                        "frequency_hz": 20.0,
                    },
                    "candidates": candidates,
                    "split_group": {
                        "episode_id": episode_id,
                        "task_id": task,
                        "object_composition_id": f"m4r-object-composition:{task}",
                        "scene_id": f"m4r-scene:{task}:world-{world:04d}",
                        "policy_source": "milestone4r_v3_candidate_spec_bank",
                        "source_split": first["split"],
                    },
                    "provenance": {
                        "adapter": "actioncheck.local_sources.adapt_m4r",
                        "source_candidate_file": str(
                            (M4R / f"{task}_candidates.jsonl").resolve()
                        ),
                        "source_label_file": str(
                            (M4R / f"{task}_labels.npz").resolve()
                        ),
                        "source_bundle_file": str(
                            (M4R / f"{task}_bundles.npz").resolve()
                        ),
                        "history_ref": reference,
                        "world_id": world,
                        "branch": branch,
                        "causal_context_only": True,
                        "post_execution_tcp_state_excluded": True,
                        "generator_private_segmentation_excluded": True,
                    },
                }
            )
    return groups


def _m5b_state_tokens(
    arrays: dict[str, np.ndarray], history_id: int
) -> list[list[float]]:
    positions = arrays["positions"][history_id].reshape(6, -1)
    appearance = arrays["appearance_cues"][history_id].reshape(6, -1)
    appearance_visibility = arrays["appearance_visibility"][history_id].reshape(
        6, -1
    )
    gap = arrays["gap_proxy"][history_id].reshape(6, -1)
    gap_visibility = arrays["gap_visibility"][history_id].reshape(6, -1)
    goal = np.tile(arrays["goal_cues"][history_id][None], (6, 1))
    return np.concatenate(
        (positions, appearance, appearance_visibility, gap, gap_visibility, goal),
        axis=1,
    ).astype(float).tolist()


def adapt_m5b() -> list[dict[str, Any]]:
    rows = _read_jsonl(M5B / "candidate_labels.jsonl")
    actions = np.load(M5B / "candidate_actions.npy").astype(np.float32)
    with np.load(M5B / "histories.npz") as stored:
        arrays = {name: stored[name].copy() for name in stored.files}
    task_lookup = {task: index for index, task in enumerate(M5B_TASKS)}
    by_history: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_history[(row["task"], int(row["history_id"]))].append(row)
    generator_hash = sha256_file(M5B_GENERATOR)
    execution_log_hash = sha256_file(M5B / "execution_logs.json")
    groups = []
    for (task, history_id), values in sorted(by_history.items()):
        values.sort(key=lambda row: int(row["candidate_slot"]))
        first = values[0]
        world = int(first["world_id"])
        branch = int(first["branch"])
        context_id = f"m5b-v3:{task}:world-{world:04d}:branch-{branch}"
        episode_id = f"m5b-v3:{task}:world-{world:04d}"
        candidates = []
        for row in values:
            slot = int(row["candidate_slot"])
            action = actions[task_lookup[task], world, slot].astype(float).tolist()
            candidate_id = f"{context_id}:candidate-{slot:02d}"
            candidates.append(
                _sim_candidate(
                    candidate_id=candidate_id,
                    action=action,
                    source_policy="milestone5b_v3_compositional_planner_bank",
                    sampling_metadata={
                        "candidate_slot": slot,
                        "factor_combination": row["factor_combination"],
                        "normalized_progress": 0.0,
                    },
                    success=bool(row["success"]),
                    label_source=row["label_source"],
                    sim_result={
                        "simulator": "ManiSkill 3 GPU PhysX",
                        "source_world_id": world,
                        "source_branch": branch,
                        "source_candidate_slot": slot,
                        "execution_generator_sha256": generator_hash,
                        "execution_log_sha256": execution_log_hash,
                        "executed_action_horizon": len(action),
                    },
                )
            )
        factor = dict(first["factor_combination"])
        factor.pop("candidate_direction", None)
        groups.append(
            {
                "context_id": context_id,
                "task_id": task,
                "task_instruction": INSTRUCTIONS[task],
                "episode_id": episode_id,
                "timestamp": 0.0,
                "history": {
                    "rgb_refs": [],
                    "state": _m5b_state_tokens(arrays, history_id),
                    "gripper": [],
                    "action_history": None,
                    "visual_features": [],
                    "history_length": 6,
                    "frequency_hz": 20.0,
                },
                "candidates": candidates,
                "split_group": {
                    "episode_id": episode_id,
                    "task_id": task,
                    "object_composition_id": "m5b-composition:"
                    + sha256_json(factor)[:16],
                    "scene_id": f"m5b-scene:{task}:world-{world:04d}",
                    "policy_source": "milestone5b_v3_compositional_planner_bank",
                    "source_split": first["iid_split"],
                },
                "provenance": {
                    "adapter": "actioncheck.local_sources.adapt_m5b",
                    "source_candidate_file": str(
                        (M5B / "candidate_labels.jsonl").resolve()
                    ),
                    "source_action_file": str(
                        (M5B / "candidate_actions.npy").resolve()
                    ),
                    "source_history_file": str((M5B / "histories.npz").resolve()),
                    "source_execution_log": str(
                        (M5B / "execution_logs.json").resolve()
                    ),
                    "history_id": history_id,
                    "world_id": world,
                    "branch": branch,
                    "factor_combination": factor,
                    "causal_context_only": True,
                },
            }
        )
    return groups


def build_local_pilot_groups() -> list[dict[str, Any]]:
    groups = adapt_m4r() + adapt_m5b()
    groups.sort(key=lambda row: row["context_id"])
    if len(groups) != 1024:
        raise RuntimeError(f"expected 1024 local proposal groups, found {len(groups)}")
    return groups


def inventory_local_data() -> dict[str, Any]:
    project_root = ROOT.parent.parent
    data_roots = [
        ROOT,
        project_root / "RoboMIND2",
        project_root,
    ]
    root_status = []
    for path in data_roots:
        root_status.append(
            {
                "path": str(path),
                "exists": path.exists(),
                "readable": path.exists() and path.is_dir(),
            }
        )
    name_terms = ("tb6", "ur5", "pi0", "openpi")
    named_matches = []
    pruned = {
        ".git",
        ".venv",
        "__pycache__",
        "pip_cache",
        "wheels",
        "node_modules",
    }
    for current, directories, files in os.walk(project_root):
        directories[:] = [name for name in directories if name not in pruned]
        for name in directories + files:
            if any(term in name.lower() for term in name_terms):
                named_matches.append(str(Path(current) / name))
            if len(named_matches) >= 500:
                break
        if len(named_matches) >= 500:
            break
    selected = build_local_pilot_groups()
    labels = Counter(
        candidate["label"]
        for group in selected
        for candidate in group["candidates"]
    )
    return {
        "schema": "actioncheck-local-data-inventory-v1",
        "roots": root_status,
        "name_search_terms": list(name_terms),
        "name_matches": sorted(named_matches),
        "name_matches_are_not_labels": True,
        "selected_candidate_summary": {
            "context_groups": len(selected),
            "candidates": sum(len(group["candidates"]) for group in selected),
            "tasks": sorted({group["task_id"] for group in selected}),
            "candidate_sources": sorted(
                {
                    candidate["source_policy"]
                    for group in selected
                    for candidate in group["candidates"]
                }
            ),
            "label_distribution": dict(sorted(labels.items())),
            "evidence_types": sorted(
                {
                    candidate["evidence_type"]
                    for group in selected
                    for candidate in group["candidates"]
                }
            ),
        },
        "inspection_only": True,
        "files_modified_during_inventory": 0,
    }


def source_inventory_records() -> list[dict[str, Any]]:
    return [
        {
            "source_id": "milestone4r_v3_gpu_physx",
            "path": str(M4R),
            "status": "valid_proposal_data_candidate_selected",
            "context_groups": 768,
            "candidates": 7680,
            "tasks": list(M4R_TASKS),
            "numeric_action_chunks": True,
            "same_context_multiple_candidates": True,
            "evidence": "per-candidate ManiSkill 3 GPU PhysX execution outcome",
            "label_inferred_from_filename": False,
        },
        {
            "source_id": "milestone5b_v3_gpu_physx",
            "path": str(M5B),
            "status": "valid_proposal_data_candidate_selected",
            "context_groups": 256,
            "candidates": 2560,
            "tasks": list(M5B_TASKS),
            "numeric_action_chunks": True,
            "same_context_multiple_candidates": True,
            "evidence": "per-candidate ManiSkill 3 GPU PhysX execution log",
            "label_inferred_from_filename": False,
        },
        {
            "source_id": "milestone3q_signed_dynamics",
            "path": str(ACTMASK_OUTPUTS / "milestone3q_signed_dynamics"),
            "status": "proposal_execution_source_not_selected_for_pilot",
            "reason": "only 20% of exact causal context groups contain both labels; later v3 sources provide stronger candidate diversity",
            "labels_fabricated": False,
        },
        {
            "source_id": "milestone3p_maniskill_pilot",
            "path": str(ACTMASK_OUTPUTS / "milestone3p_maniskill_pilot"),
            "status": "proposal_execution_source_not_selected_for_pilot",
            "reason": "only 20% of exact causal context groups contain both labels; superseded by audited v3 sources",
            "labels_fabricated": False,
        },
        {
            "source_id": "asu_logged_candidate_rows",
            "path": str(ACTMASK_OUTPUTS / "r_series_real_robot_verification"),
            "status": "proposals_found_no_allowed_labels",
            "reason": "consistent flags include logged swaps/reversals and are not execution-backed proposal outcomes",
        },
        {
            "source_id": "robomind2",
            "path": str(ROOT.parent.parent / "RoboMIND2"),
            "status": "logged_demonstrations_no_grouped_proposals",
            "reason": "demonstrations and segmentation annotation do not provide multiple executed proposals per context",
        },
        {
            "source_id": "botfails",
            "path": str(ACTMASK_OUTPUTS / "r2_series_real_outcome_search"),
            "status": "execution_labels_found_no_grouped_contexts",
            "reason": "failure trajectories do not contain multiple candidate actions executed from one context",
        },
        {
            "source_id": "embodied_agent_symbolic",
            "path": str(ROOT.parent.parent / "embodied_agent"),
            "status": "not_robot_action_chunk_proposal_data",
            "reason": "symbolic/high-level planner decisions are not numeric robot action chunks; smoke results explicitly say not simulator-backed",
        },
        {
            "source_id": "sam3_annotation",
            "path": str(
                ACTMASK_OUTPUTS / "rm_spd_sam3_prior_dynamics"
            ),
            "status": "not_action_proposal_data",
            "reason": "segmentation proposals are neither action chunks nor compatibility labels",
        },
    ]
