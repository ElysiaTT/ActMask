"""RM-ACV Phase A0/A1: full local RoboMIND inventory and data admission.

This module intentionally does not construct candidates or train a model when
the preregistered trajectory/task floor is not met.  It is a read-only audit of
the local RoboMIND 2.0 release; file paths are provenance only and are never a
predictive feature.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pyarrow.parquet as pq

from actmask.data.robomind_adapter import RoboMINDRecord, load_numeric_episode, validate_numeric_episode


ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path("/data/projects/tzh/RoboMIND2")
OUT = ROOT / "outputs" / "actmask" / "rm_acv_complete"
DOCS = OUT / "paper_reframe"
CONTEXT_STEPS = 8
ACTION_HORIZON = 16
MIN_TRAJECTORIES = 300
MIN_TASKS = 20
MIN_TASK_TRAJECTORIES = 5
MIN_HELDOUT_TASKS = 5
REQUIRED = (
    "preregistered_config.json", "preregistered_config.sha256", "dataset_inventory.json",
    "task_inventory.json", "trajectory_inventory.jsonl", "eligibility_report.md",
    "fresh_benchmark_manifest.jsonl", "source_hash_manifest.json", "anchor_manifest.jsonl",
    "positive_candidate_manifest.jsonl", "alignment_audit.json", "anchor_statistics.json",
    "negative_candidate_manifest.jsonl", "negative_generation_report.json",
    "negative_matching_statistics.json", "negative_failure_log.jsonl", "shortcut_feature_report.json",
    "shortcut_predictions.jsonl", "negative_quality_report.json", "matching_balance_report.json",
    "negative_family_decision.json", "split_manifest.json", "split_audit.json",
    "task_heldout_audit.json", "leakage_audit.json", "baseline_config.json", "baseline_summary.json",
    "baseline_per_family.json", "baseline_task_heldout.json", "raw_prediction_manifest.json",
    "phase_a9_decision.json", "final_decision.json", "final_report.md", "dataset_report.md",
    "negative_construction_report.md", "shortcut_audit_report.md", "baseline_report.md",
    "task_heldout_report.md", "candidate_ranking_report.md", "calibration_report.md",
    "error_analysis_report.md", "claim_boundary.md", "next_steps.md", "reproducibility_manifest.json",
    "run_log.jsonl", "paper_reframe/title_candidates.md", "paper_reframe/abstract_zh.md",
    "paper_reframe/abstract_en.md", "paper_reframe/introduction_outline.md",
    "paper_reframe/problem_formulation.md", "paper_reframe/contribution_boundary.md",
    "paper_reframe/related_work_plan.md", "paper_reframe/method.md", "paper_reframe/experiments.md",
    "paper_reframe/results_tables.md", "paper_reframe/figure_plan.md", "paper_reframe/appendix_audit_plan.md",
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _log(event: str, **payload: Any) -> None:
    path = OUT / "run_log.jsonl"
    previous = ""
    if path.is_file():
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
        if lines:
            previous = json.loads(lines[-1])["event_sha256"]
    event_payload = {"event": event, "utc": datetime.now(timezone.utc).isoformat(), "previous_event_sha256": previous, "payload": payload}
    event_payload["event_sha256"] = hashlib.sha256(json.dumps(event_payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event_payload, ensure_ascii=False, sort_keys=True) + "\n")


def _episode_number(path: Path) -> int:
    return int(path.stem.removeprefix("episode_"))


def _task_root(path: Path) -> Path:
    if path.parent.name != "chunk-000" or path.parent.parent.name != "data":
        raise ValueError(f"unexpected source layout: {path}")
    return path.parents[2]


def _metadata(task_root: Path, cache: dict[Path, tuple[dict[str, Any], dict[int, dict[str, Any]]]]) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    if task_root not in cache:
        info_path, episodes_path = task_root / "meta" / "info.json", task_root / "meta" / "episodes.jsonl"
        info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.is_file() else {}
        episodes = {}
        if episodes_path.is_file():
            for line in episodes_path.read_text(encoding="utf-8").splitlines():
                if line:
                    row = json.loads(line); episodes[int(row["episode_index"])] = row
        cache[task_root] = (info, episodes)
    return cache[task_root]


def _schema_fields(path: Path) -> list[str]:
    return [field.name for field in pq.ParquetFile(path).schema_arrow]


def _numeric_probe(record: RoboMINDRecord) -> tuple[dict[str, Any], dict[str, Any]]:
    """Full numeric-only validation; depth/RGB bytes are never decoded here."""
    numeric = load_numeric_episode(record)
    audit = validate_numeric_episode(record, numeric)
    duration = float(numeric["timestamp"][-1] - numeric["timestamp"][0]) if len(numeric["timestamp"]) > 1 else 0.0
    dt = np.diff(numeric["timestamp"])
    frequency = float(1.0 / np.median(dt)) if len(dt) and np.median(dt) > 0 else None
    detail = {
        "state_dim": int(numeric["state"].shape[1]), "action_dim": int(numeric["action"].shape[1]),
        "duration_seconds": duration, "control_frequency_hz": frequency,
        "action_finite": bool(np.isfinite(numeric["action"]).all()), "state_finite": bool(np.isfinite(numeric["state"]).all()),
    }
    return audit, detail


def _configuration() -> dict[str, Any]:
    return {
        "schema": "rm-acv-preregistered-config-v1",
        "branch": "RM-ACV complete local benchmark",
        "source": {"dataset": "RoboMIND2.0", "root": str(SOURCE), "policy": "complete local release; read-only; no external data"},
        "primary_embodiment_selection": "largest single complete embodiment after A1 numeric eligibility inventory; ties lexicographic; selection cannot depend on candidate labels, model outputs, or performance",
        "eligibility": {"minimum_usable_trajectories": MIN_TRAJECTORIES, "target_trajectories": [500, 2000], "minimum_tasks": MIN_TASKS, "minimum_trajectories_per_task": MIN_TASK_TRAJECTORIES, "minimum_task_heldout_tasks": MIN_HELDOUT_TASKS},
        "causal_sample_protocol": {"context_steps": CONTEXT_STEPS, "action_horizon_steps": ACTION_HORIZON, "action_representation": "raw logged action vector; no future state or outcome", "normalization": "train-split per-dimension z-score frozen after A5", "anchor_sampling": "four evenly spaced legal anchors per verified eligible trajectory"},
        "positive_definition": "logged action chunk immediately after causal anchor; compatible distributional example only, not an optimal/success/safe label",
        "negative_families": ["N1_same_task_near_state", "N2_local_temporal_corruption", "N3_arm_gripper_desynchronization", "N4_phase_shifted_same_task", "N5_coordination_corruption", "N6_endpoint_matched_path_corruption"],
        "negative_matching_tolerances": {"state_normalized_l2_max": 0.35, "end_effector_normalized_l2_max_when_available": 0.35, "normalized_progress_absolute_delta_max": 0.15, "action_magnitude_relative_delta_max": 0.15, "action_smoothness_relative_delta_max": 0.20, "chunk_displacement_relative_delta_max": 0.20, "policy": "a family is rejected wholesale after at most one redesign; individual samples are never hand-repaired"},
        "splits": {"development_validation_episode_test": [0.70, 0.15, 0.15], "episode_assignment": "deterministic SHA-256 of provenance episode identifier, grouped so every anchor and every retrieved source from an episode stays within one split", "selection_before_evaluation": True},
        "task_heldout_protocol": {"heldout_task_count": MIN_HELDOUT_TASKS, "selection": "deterministic SHA-256 ordering of eligible primary-embodiment task identifiers before candidate construction; never selected using model performance", "source_episode_boundary": "retrieved negative source episodes must remain inside the destination split"},
        "shortcut_baselines": ["random", "task_only", "progress_only", "action_magnitude_only", "action_summary_only", "source_metadata_diagnostic_only", "negative_family_identity_diagnostic_only", "combined_non_context_features"],
        "standard_baselines": ["state_action_tcn", "task_conditioned_temporal_transformer", "visual_state_action_transformer", "maintained_ssm_verifier", "autoregressive_behavior_cloning_likelihood", "masked_action_transformer_score", "contrastive_context_action_encoder", "diffusion_or_flow_action_denoising_score_when_practical"],
        "metrics": {"classification": ["balanced_accuracy", "auroc", "auprc", "false_accept_rate", "false_reject_rate"], "ranking": ["recall_at_1", "mrr", "ndcg", "positive_negative_energy_margin"], "calibration": ["ece", "brier", "risk_coverage", "selective_accuracy_100_90_80_70"], "localization": ["temporal_iou", "corrupted_window_recall"], "uncertainty": "trajectory-grouped bootstrap confidence intervals"},
        "fair_inputs": ["causal_rgb_history", "causal_proprioceptive_history", "task_instruction", "candidate_action_chunk"],
        "forbidden_inputs": ["SAM3", "robot_masks", "future_rgb", "future_visual_features", "future_robot_state", "episode_id", "file_path", "source_ordering", "negative_family_identity", "candidate_source_metadata"],
        "admission_gates": {"shortcut_balanced_accuracy_max": 0.60, "fair_episode_balanced_accuracy_min": 0.62, "fair_task_balanced_accuracy_min": 0.57, "minimum_surviving_negative_families": 3},
        "method_if_authorized": {"family": "PhaseAwareTemporalEnergyVerifier", "seeds": [17, 29, 43], "method_training_prohibited_before_new_a9_ready": True},
        "resource_limits": {"no_unrelated_process_termination": True, "source_hashing": "SHA-256 all local Parquet and required task metadata; RGB presence is audited without decoding video bytes"},
    }


def _not_run(name: str, reason: str) -> dict[str, Any]:
    return {"schema": "rm-acv-terminal-not-run-v1", "artifact": name, "status": "not_run", "reason": reason, "method_trained": False, "learned_baseline_trained": False}


def run() -> dict[str, Any]:
    if not SOURCE.is_dir():
        raise RuntimeError(f"local RoboMIND source missing: {SOURCE}")
    OUT.mkdir(parents=True, exist_ok=True)
    # A run log is append-only within one invocation.  Reset only this branch's
    # generated log before a full fresh inventory, so its final hash is stable.
    (OUT / "run_log.jsonl").unlink(missing_ok=True)
    config = _configuration()
    _write_json(OUT / "preregistered_config.json", config)
    (OUT / "preregistered_config.sha256").write_text(_sha(OUT / "preregistered_config.json") + "\n", encoding="utf-8")
    _log("A0_PREREGISTRATION_FROZEN", config_sha256=_sha(OUT / "preregistered_config.json"))

    candidates = sorted(path for path in SOURCE.rglob("episode_*.parquet") if path.parent.name == "chunk-000" and path.parent.parent.name == "data")
    metadata_cache: dict[Path, tuple[dict[str, Any], dict[int, dict[str, Any]]]] = {}
    rows: list[dict[str, Any]] = []
    hash_rows: list[dict[str, Any]] = []
    for ordinal, path in enumerate(candidates, start=1):
        root = _task_root(path); task_id, episode_index = root.name, _episode_number(path)
        info, episode_rows = _metadata(root, metadata_cache); episode = episode_rows.get(episode_index, {})
        language = str((episode.get("tasks") or [""])[0])
        front = root / "videos" / "chunk-000" / "observation.rgb_images.camera_front" / f"{path.stem}.mp4"
        camera_videos = sorted((root / "videos" / "chunk-000").glob(f"*/{path.stem}.mp4"))
        record = RoboMINDRecord(episode_id=f"{task_id}:{episode_index:06d}", task_id=task_id, robot_embodiment=str(info.get("robot_type", "unknown")), language_instruction=language, parquet_path=str(path), front_video_path=str(front) if front.is_file() else None, source_bytes=int(path.stat().st_size), episode_index=episode_index, metadata_path=str(root / "meta" / "episodes.jsonl"))
        try:
            alignment, detail = _numeric_probe(record)
            numeric_pass = bool(alignment["pass"] and detail["action_finite"] and detail["state_finite"])
            failure = None
        except Exception as exc:  # fail closed; never repair a source trajectory
            alignment, detail, numeric_pass, failure = {"pass": False}, {}, False, f"{type(exc).__name__}: {exc}"
        schema = _schema_fields(path)
        required_fields = {"action.arm_joint_position", "action.hand_joint_position", "action.end_effector", "observation.state.arm_joint_position", "observation.state.hand_joint_position", "observation.state.end_effector", "timestamp", "frame_index", "episode_index", "task_index"}
        metadata_pass = bool(language and record.robot_embodiment != "unknown" and required_fields.issubset(schema))
        row = {
            "schema": "rm-acv-trajectory-inventory-v1", "inventory_ordinal": ordinal,
            "episode_id_provenance_only": record.episode_id, "task_id_provenance_only": task_id,
            "episode_index": episode_index, "embodiment": record.robot_embodiment,
            "language_available": bool(language), "front_rgb_available": bool(record.front_video_path),
            "rgb_camera_count": len(camera_videos), "all_six_rgb_views_available": len(camera_videos) >= 6,
            "parquet_bytes": record.source_bytes, "numeric_alignment": alignment, "numeric_detail": detail,
            "schema_fields": schema, "metadata_pass": metadata_pass, "numeric_pass": numeric_pass,
            "usable_for_candidate_benchmark": bool(metadata_pass and numeric_pass and record.front_video_path),
            "failure": failure, "source_provenance": {"parquet": str(path), "metadata": record.metadata_path, "front_video": str(front) if front.is_file() else None},
        }
        rows.append(row)
        hash_rows.append({"kind": "parquet", "path": str(path.relative_to(SOURCE)), "bytes": record.source_bytes, "sha256": _sha(path)})
        if ordinal % 25 == 0:
            print(f"RM_ACV_A1={ordinal}/{len(candidates)}", flush=True)
    metadata_task_roots = sorted({path.parent.parent for path in SOURCE.rglob("meta/info.json")})
    for root in metadata_task_roots:
        for name in ("info.json", "episodes.jsonl"):
            path = root / "meta" / name
            if path.is_file(): hash_rows.append({"kind": "metadata", "path": str(path.relative_to(SOURCE)), "bytes": path.stat().st_size, "sha256": _sha(path)})
    auxiliary_hdf5 = sorted((SOURCE / "data").rglob("*.hdf5")) if (SOURCE / "data").is_dir() else []
    for ordinal, path in enumerate(auxiliary_hdf5, start=1):
        hash_rows.append({"kind": "auxiliary_hdf5_unusable_for_primary", "path": str(path.relative_to(SOURCE)), "bytes": path.stat().st_size, "sha256": _sha(path)})
        print(f"RM_ACV_A1_HDF5_HASH={ordinal}/{len(auxiliary_hdf5)}", flush=True)
    hash_rows.sort(key=lambda row: row["path"])
    aggregate = hashlib.sha256("\n".join(f"{row['path']}:{row['sha256']}" for row in hash_rows).encode()).hexdigest()
    _write_json(OUT / "source_hash_manifest.json", {"schema": "rm-acv-source-hash-manifest-v1", "source_root": str(SOURCE), "entries": hash_rows, "aggregate_sha256": aggregate, "content_hash_policy": "SHA-256 every discovered Parquet, auxiliary HDF5, and task info/episodes metadata; video files are not decoded because A1 ends before visual model use"})
    _write_jsonl(OUT / "trajectory_inventory.jsonl", rows)

    usable = [row for row in rows if row["usable_for_candidate_benchmark"]]
    by_embodiment = Counter(row["embodiment"] for row in usable)
    primary = sorted(by_embodiment, key=lambda value: (-by_embodiment[value], value))[0] if by_embodiment else None
    primary_rows = [row for row in usable if row["embodiment"] == primary]
    task_rows: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in usable:
        task_rows[(str(row["embodiment"]), str(row["task_id_provenance_only"]))].append(row)
    task_inventory = []
    for (embodiment, task), members in sorted(task_rows.items()):
        frequencies = [row["numeric_detail"].get("control_frequency_hz") for row in members if row["numeric_detail"].get("control_frequency_hz") is not None]
        task_inventory.append({
            "task_id_provenance_only": task, "embodiment": embodiment, "is_primary_embodiment": embodiment == primary,
            "usable_trajectories": len(members), "minimum_trajectory_requirement_pass": len(members) >= MIN_TASK_TRAJECTORIES,
            "language_available_all": all(row["language_available"] for row in members),
            "front_rgb_available_all": all(row["front_rgb_available"] for row in members),
            "all_six_rgb_views_available_all": all(row["all_six_rgb_views_available"] for row in members),
            "state_dimensions": sorted({int(row["numeric_detail"]["state_dim"]) for row in members}),
            "action_dimensions": sorted({int(row["numeric_detail"]["action_dim"]) for row in members}),
            "control_frequency_hz_median": float(np.median(frequencies)) if frequencies else None,
            "duration_seconds_median": float(np.median([row["numeric_detail"].get("duration_seconds", 0.0) for row in members])),
        })
    primary_task_inventory = [row for row in task_inventory if row["is_primary_embodiment"]]
    tasks_meeting_floor = [row for row in primary_task_inventory if row["minimum_trajectory_requirement_pass"]]
    task_to_embodiments: dict[str, set[str]] = defaultdict(set)
    for row in usable:
        task_to_embodiments[str(row["task_id_provenance_only"])].add(str(row["embodiment"]))
    incomplete_metadata_tasks = sorted(
        root.name for root in metadata_task_roots
        if not any((root / "data").glob("chunk-*/*.parquet"))
    )
    frequencies_all = [row["numeric_detail"].get("control_frequency_hz") for row in usable if row["numeric_detail"].get("control_frequency_hz") is not None]
    durations_all = [row["numeric_detail"].get("duration_seconds", 0.0) for row in usable]
    requirements = {
        "one_consistent_primary_embodiment": primary is not None,
        "minimum_300_usable_trajectories": len(primary_rows) >= MIN_TRAJECTORIES,
        "minimum_20_tasks": len(tasks_meeting_floor) >= MIN_TASKS,
        # The five held-out tasks are a subset of the required 20 tasks; they
        # are not five additional task families.  Keep the independent gates
        # separate so the admission report identifies the real shortfall.
        "minimum_5_trajectories_per_included_task": bool(tasks_meeting_floor) and all(row["minimum_trajectory_requirement_pass"] for row in task_inventory),
        "minimum_5_entirely_held_out_tasks": len(tasks_meeting_floor) >= MIN_HELDOUT_TASKS,
    }
    admitted = all(requirements.values())
    inventory = {
        "schema": "rm-acv-dataset-inventory-v1", "dataset": "RoboMIND2.0", "source_root": str(SOURCE),
        "total_discovered_trajectories": len(rows), "numeric_and_metadata_usable_trajectories": len(usable),
        "corrupt_or_incomplete_parquet_trajectories": len(rows) - len(usable),
        "embodiment_counts": dict(sorted(by_embodiment.items())), "primary_embodiment": primary,
        "primary_embodiment_usable_trajectories": len(primary_rows), "all_embodiment_task_count": len(task_inventory),
        "tasks_with_primary_embodiment": len(primary_task_inventory), "tasks_meeting_minimum_5_trajectories": len(tasks_meeting_floor),
        "state_action_dimension_pairs": dict(sorted(Counter(f"{row['numeric_detail']['state_dim']}/{row['numeric_detail']['action_dim']}" for row in usable).items())),
        "control_frequency_hz": {"median": float(np.median(frequencies_all)) if frequencies_all else None, "minimum": float(min(frequencies_all)) if frequencies_all else None, "maximum": float(max(frequencies_all)) if frequencies_all else None},
        "duration_seconds": {"median": float(np.median(durations_all)) if durations_all else None, "minimum": float(min(durations_all)) if durations_all else None, "maximum": float(max(durations_all)) if durations_all else None},
        "rgb_availability": {"front_rgb_trajectories": sum(bool(row["front_rgb_available"]) for row in rows), "all_six_rgb_view_trajectories": sum(bool(row["all_six_rgb_views_available"]) for row in rows)},
        "task_language_availability": {"trajectories_with_language": sum(bool(row["language_available"]) for row in rows)},
        "repeated_task_ids_across_embodiments": sorted(task for task, embodiments in task_to_embodiments.items() if len(embodiments) > 1),
        "environment_metadata": {"dedicated_environment_identifier_available": False, "policy": "no environment name is inferred from file paths or task directory names"},
        "incomplete_metadata_only_tasks": incomplete_metadata_tasks,
        "auxiliary_hdf5_inventory": {"file_count": len(auxiliary_hdf5), "total_bytes": sum(path.stat().st_size for path in auxiliary_hdf5), "admission": "excluded: each is a heterogeneous single-trajectory HDF5 source without the repeated task scale or shared Parquet action/state schema needed for this one-embodiment benchmark"},
        "required": {"min_trajectories": MIN_TRAJECTORIES, "min_tasks": MIN_TASKS, "min_per_task": MIN_TASK_TRAJECTORIES, "min_heldout_tasks": MIN_HELDOUT_TASKS}, "requirements": requirements, "admitted": admitted, "source_hash_aggregate": aggregate,
    }
    _write_json(OUT / "dataset_inventory.json", inventory); _write_json(OUT / "task_inventory.json", {"schema": "rm-acv-task-inventory-v1", "tasks": task_inventory})
    # This is a full fresh-source manifest, not a silently recycled 100-trajectory pilot.
    _write_jsonl(OUT / "fresh_benchmark_manifest.jsonl", primary_rows)
    (OUT / "eligibility_report.md").write_text(
        "# RM-ACV A1 eligibility\n\n"
        f"The complete local release contains {len(rows)} structured Parquet trajectories and {len(auxiliary_hdf5)} auxiliary HDF5 files. The HDF5 files are audited and hashed but excluded from the primary benchmark because they are heterogeneous single-trajectory sources without a shared repeated-task schema. The largest single embodiment is `{primary}` with {len(primary_rows)} usable trajectories across {len(tasks_meeting_floor)} tasks having at least {MIN_TASK_TRAJECTORIES} trajectories.\n\n"
        f"Required: ≥{MIN_TRAJECTORIES} trajectories and ≥{MIN_TASKS} tasks under one embodiment. Result: **{'ADMITTED' if admitted else 'NOT ADMITTED'}**.\n",
        encoding="utf-8",
    )
    _log("A1_COMPLETE_LOCAL_DATASET_INVENTORY", total=len(rows), usable=len(usable), primary_embodiment=primary, primary_usable=len(primary_rows), eligible_tasks=len(tasks_meeting_floor), admitted=admitted, source_hash_aggregate=aggregate)

    if admitted:
        raise RuntimeError("A1 unexpectedly admitted; this A0/A1-only runner must be replaced by the full candidate-construction phase before emitting A9.")
    reason = "The complete local RoboMIND 2.0 release cannot meet the frozen single-embodiment minimum of 300 usable trajectories and 20 tasks; constructing additional anchors cannot increase trajectory or task count."
    decision = {"schema": "rm-acv-phase-a9-decision-v1", "decision": "RM_ACV_DATA_INSUFFICIENT", "terminal": True, "stage_reached": "A1", "reason": reason, "requirements": requirements, "observed": {"complete_local_parquet_trajectories": len(rows), "auxiliary_hdf5_files_excluded_as_heterogeneous": len(auxiliary_hdf5), "largest_single_embodiment_usable_trajectories": len(primary_rows), "tasks_meeting_minimum_5_trajectories": len(tasks_meeting_floor)}, "learned_baselines_run": False, "method_training_authorized": False}
    _write_json(OUT / "phase_a9_decision.json", decision); _write_json(OUT / "final_decision.json", decision)
    for filename in ("anchor_manifest.jsonl", "positive_candidate_manifest.jsonl", "negative_candidate_manifest.jsonl", "negative_failure_log.jsonl", "shortcut_predictions.jsonl", "raw_prediction_manifest.json"):
        _write_jsonl(OUT / filename, [])
    for filename in ("alignment_audit.json", "anchor_statistics.json", "negative_generation_report.json", "negative_matching_statistics.json", "shortcut_feature_report.json", "negative_quality_report.json", "matching_balance_report.json", "negative_family_decision.json", "split_manifest.json", "split_audit.json", "task_heldout_audit.json", "leakage_audit.json", "baseline_config.json", "baseline_summary.json", "baseline_per_family.json", "baseline_task_heldout.json"):
        _write_json(OUT / filename, _not_run(filename, reason))
    reports = {
        "final_report.md": "# RM-ACV final report\n\n**RM_ACV_DATA_INSUFFICIENT**. No learned baseline or proposed method was trained.\n",
        "dataset_report.md": f"# Dataset report\n\n{reason}\n\nObserved complete local release: {len(rows)} numerically validated Parquet trajectories, {len(auxiliary_hdf5)} hashed but heterogeneous auxiliary HDF5 files, and {len(incomplete_metadata_tasks)} metadata-only task directories. Largest consistent embodiment: {len(primary_rows)} trajectories; task floor-qualified primary tasks: {len(tasks_meeting_floor)}. All {len(usable)} Parquet trajectories have front RGB, task language, finite aligned numeric state/action fields, and were audited without using source provenance as a model input.\n",
        "negative_construction_report.md": "# Negative construction\n\nNot run: A1 data admission failed before candidate construction.\n",
        "shortcut_audit_report.md": "# Shortcut audit\n\nNot run: no admissible benchmark exists.\n",
        "baseline_report.md": "# Baselines\n\nNot run: data-admission stop occurred before learned baseline training.\n",
        "task_heldout_report.md": "# Task-held-out evaluation\n\nNot run: the frozen requirement of at least 20 tasks was not met.\n",
        "candidate_ranking_report.md": "# Candidate ranking\n\nNot run: no candidate groups were constructed after A1 stop.\n",
        "calibration_report.md": "# Calibration\n\nNot run: no model scores exist.\n",
        "error_analysis_report.md": "# Error analysis\n\nThe terminal cause is data insufficiency, not a model error.\n",
        "claim_boundary.md": "# Claim boundary\n\nThis branch reports only local-data admission failure. It makes no compatibility, safety, success, physical outcome, or robot-performance claim.\n",
        "next_steps.md": "# Next steps\n\nAcquire a versioned local RoboMIND release with at least 300 usable trajectories from one consistent embodiment and at least 20 tasks (including five preselected held-out tasks). Re-run A0 from a new frozen configuration; do not pad the dataset with anchors or reuse the 100-trajectory pilot as final evidence.\n",
    }
    for filename, text in reports.items(): (OUT / filename).write_text(text, encoding="utf-8")
    DOCS.mkdir(exist_ok=True)
    paper = {
        "title_candidates.md": "# Title candidates\n\nNo RM-ACV method title is supported while A1 data admission fails.\n",
        "abstract_zh.md": "# 摘要\n\n当前本地数据未达到预注册的 RM-ACV 基准规模，不报告方法实验结果。\n",
        "abstract_en.md": "# Abstract\n\nThe local data release does not meet the preregistered RM-ACV benchmark scale; no method result is reported.\n",
        "introduction_outline.md": "# Introduction outline\n\nNot developed into a method claim: benchmark admission failed at A1.\n",
        "problem_formulation.md": "# Problem formulation\n\nPre-execution action compatibility is a conditional-distribution verification task, not success or safety prediction. No experiment is reported here.\n",
        "contribution_boundary.md": "# Contribution boundary\n\nNo method contribution is claimed.\n",
        "related_work_plan.md": "# Related work plan\n\nDeferred until an admissible benchmark exists.\n",
        "method.md": "# Method\n\nNot run: proposed method training is prohibited after A1 data-insufficiency decision.\n",
        "experiments.md": "# Experiments\n\nNo candidate construction, baseline, or method experiment was run.\n",
        "results_tables.md": "# Results tables\n\nNo result table is valid under the current data-admission stop.\n",
        "figure_plan.md": "# Figure plan\n\nOnly the data-admission inventory may be visualized; no performance figure is supported.\n",
        "appendix_audit_plan.md": "# Appendix audit plan\n\nPrior RM-OD, RM-ACD, mask, and RM-SPD branches remain unchanged and are not evidence for RM-ACV.\n",
    }
    for filename, text in paper.items(): (DOCS / filename).write_text(text, encoding="utf-8")
    _log("A9_DATA_INSUFFICIENT_TERMINAL", decision=decision["decision"], reason=reason, method_training_authorized=False)
    tracked = [OUT / name for name in REQUIRED if name != "reproducibility_manifest.json"]
    _write_json(OUT / "reproducibility_manifest.json", {"schema": "rm-acv-reproducibility-manifest-v1", "decision": decision["decision"], "generator": {"path": str(Path(__file__).relative_to(ROOT)), "sha256": _sha(Path(__file__))}, "source_hash_aggregate": aggregate, "files": [{"path": str(path.relative_to(OUT)), "sha256": _sha(path), "bytes": path.stat().st_size} for path in tracked]})
    return decision


def verify(*, content_rehash: bool = False) -> dict[str, Any]:
    missing = [name for name in REQUIRED if not (OUT / name).is_file()]
    decision = json.loads((OUT / "phase_a9_decision.json").read_text()) if not missing else {}
    inventory = json.loads((OUT / "dataset_inventory.json").read_text()) if not missing else {}
    manifest = json.loads((OUT / "reproducibility_manifest.json").read_text()) if not missing else {}
    hash_failures = []
    for item in manifest.get("files", []):
        path = OUT / item["path"]
        if not path.is_file() or _sha(path) != item["sha256"]: hash_failures.append(item["path"])
    if manifest.get("generator", {}).get("sha256") != _sha(Path(__file__)): hash_failures.append("generator")
    source_result: dict[str, Any] = {"content_rehash_performed": content_rehash, "pass": True}
    if content_rehash:
        source_manifest = json.loads((OUT / "source_hash_manifest.json").read_text())
        mismatches = []
        for item in source_manifest["entries"]:
            path = SOURCE / item["path"]
            if not path.is_file() or _sha(path) != item["sha256"]: mismatches.append(item["path"])
        source_result.update({"entries": len(source_manifest["entries"]), "mismatches": mismatches, "pass": not mismatches})
    passed = not missing and not hash_failures and source_result["pass"] and decision.get("decision") == "RM_ACV_DATA_INSUFFICIENT" and inventory.get("primary_embodiment_usable_trajectories", MIN_TRAJECTORIES) < MIN_TRAJECTORIES and inventory.get("tasks_meeting_minimum_5_trajectories", MIN_TASKS) < MIN_TASKS
    return {"passed": passed, "missing": missing, "hash_failures": hash_failures, "decision": decision.get("decision"), "primary_trajectory_count": inventory.get("primary_embodiment_usable_trajectories"), "task_count": inventory.get("tasks_meeting_minimum_5_trajectories"), "source_hash_verification": source_result}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--verify", action="store_true"); parser.add_argument("--rehash-source", action="store_true"); args = parser.parse_args()
    print(json.dumps(verify(content_rehash=args.rehash_source) if args.verify else run(), ensure_ascii=False, sort_keys=True))
