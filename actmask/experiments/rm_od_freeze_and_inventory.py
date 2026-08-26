"""Freeze RM-OD before any future-object-dynamics result is inspected."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "actmask" / "rm_od_future_object_dynamics"
RM = ROOT / "outputs" / "actmask" / "rm_series_robomind"
SOURCE = Path("/data/projects/tzh/RoboMIND2")


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_json(value))


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def append_log(event: str, **payload: Any) -> None:
    path = OUT / "run_log.jsonl"
    previous = ""
    sequence = 0
    if path.is_file() and path.stat().st_size:
        lines = [json.loads(line) for line in path.read_text().splitlines() if line]
        sequence, previous = int(lines[-1]["seq"]), str(lines[-1]["event_sha256"])
    row = {"seq": sequence + 1, "utc": datetime.now(timezone.utc).isoformat(), "event": event, "payload": payload, "prev_event_sha256": previous}
    row["event_sha256"] = hashlib.sha256(json.dumps(row, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    with path.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def run() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    prior_subset = json.loads((RM / "source_subset" / "subset_manifest.json").read_text())
    prior_hashes = json.loads((RM / "source_hash_manifest.json").read_text())["files"]
    selected = list(prior_subset["episodes_detail"])
    selected_tasks = sorted({row["task_id"] for row in selected})
    by_task: dict[str, list[Path]] = {task: [] for task in selected_tasks}
    for path in SOURCE.rglob("episode_*.parquet"):
        if path.parent.name == "chunk-000" and path.parent.parent.name == "data" and path.parents[2].name in by_task:
            by_task[path.parents[2].name].append(path)
    local_counts = {task: len(paths) for task, paths in sorted(by_task.items())}
    all_same_task = sum(local_counts.values())
    hash_by_path = {str(row["local_path"]): row for row in prior_hashes}
    source_entries = []
    for episode in selected:
        for modality, path in (("parquet_numeric_depth", episode["parquet_path"]), ("front_rgb_video", episode["front_video_path"])):
            old = hash_by_path.get(str(path))
            if old is None:
                raise RuntimeError(f"RM-Series hash is missing selected RM-OD source: {path}")
            source_entries.append({"episode_id": episode["episode_id"], "task": episode["task_id"], "modality": modality, "local_path": str(path), "bytes": int(old["bytes"]), "sha256": str(old["sha256"]), "hash_origin": "rm_series_robomind/source_hash_manifest.json"})
    config = {
        "schema": "rm-od-preregistered-config-v1",
        "source": {"dataset": "RoboMIND2.0", "source_root": str(SOURCE), "selected_episodes": 100, "selected_tasks": selected_tasks, "source_hash_manifest": "source_hash_manifest.json", "no_new_download": True},
        "views": {"fair_history_camera": "observation.rgb_images.camera_front", "candidate_future_camera": "observation.rgb_images.camera_front", "camera_mixing_allowed": False},
        "temporal": {"history_steps": 4, "candidate_action_steps": 4, "future_clip_steps": 4, "anchors_per_episode": 8, "anchor_fractions": [0.12, 0.22, 0.32, 0.42, 0.52, 0.62, 0.72, 0.82]},
        "candidate_retrieval": {"k": 8, "optional_k16_only_after_k8_pass": True, "positive": "same-trajectory post-action future clip", "hard_negative_policy": {"same_official_task": True, "same_embodiment": True, "same_camera": True, "match_pre_anchor_state": True, "match_pre_anchor_visual": True, "match_action_magnitude": True, "match_action_trajectory": True, "match_end_effector_displacement": True, "match_episode_progress": True, "require_future_nonrobot_visual_difference": True, "forbidden_primary_selection": ["different task", "different embodiment", "different resolution", "different camera", "radically different action", "source folder convention"]}},
        "splits": {"episode_held_out": "6/2/2 episodes per task", "task_held_out": "6/2/2 task groups", "candidate_sources_stay_inside_split": True, "no_test_episode_candidate_in_train": True},
        "fair_inputs": ["pre-anchor front RGB history", "pre-anchor robot state history", "candidate action chunk", "candidate future RGB clip as symmetric retrieval candidate"],
        "forbidden_fair_inputs": ["source path", "folder name", "episode id", "candidate index", "future action", "object id", "segmentation id", "true-positive indicator"],
        "shortcut_controls": ["constant", "candidate index", "source path", "episode progress", "task/language only", "action only", "state only", "state+action", "current RGB only", "future clip low-level diagnostic", "camera/background diagnostic"],
        "benchmark_gates": {"random_recall_at_1": 0.125, "shortcut_recall_at_1_max": 0.225, "fair_visual_recall_at_1_min": 0.275, "fair_visual_recall_at_1_max": 0.75, "privileged_future_diagnostic_margin": 0.15, "task_held_out_above_random": True, "raw_score_hash_recompute_required": True},
        "method_gates": {"family": "ObjectRelationalFutureRetriever", "seeds": [17, 29, 43], "recall_at_1_improvement": 0.05, "pair_order_improvement": 0.05, "headroom_closure_min": 0.30, "meaningful_ablation_drop": 0.03, "minimum_ablations": 2, "one_implementation_repair_only": True},
        "artifact_policy": {"all_raw_scores": True, "source_hashes": True, "metric_recomputation": True, "configs_checkpoints_logs": True, "no_frozen_rm_series_artifact_modification": True},
        "claim_boundary": "action-conditioned future visual/object dynamics retrieval from logged real-robot trajectories only; no failure, success, or physical counterfactual claim",
    }
    config_path = OUT / "preregistered_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise RuntimeError("refusing to change frozen RM-OD preregistered configuration")
    write_json(config_path, config)
    (OUT / "preregistered_config.sha256").write_text(sha(config_path) + "\n")
    write_json(OUT / "source_inventory.json", {"schema": "rm-od-source-inventory-v1", "dataset": "RoboMIND2.0", "existing_rm_series_episodes": len(selected), "same_task_local_trajectory_counts": local_counts, "same_task_local_total": all_same_task, "expanded": False, "expansion_decision": "Retain exactly the frozen 100 source trajectories: only 151 local trajectories exist in these task families, below the configured 200–500 expansion range; unrelated tasks/embodiments were not added.", "storage": {"selected_parquet_bytes": sum(row["bytes"] for row in source_entries if row["modality"] == "parquet_numeric_depth"), "selected_front_video_bytes": sum(row["bytes"] for row in source_entries if row["modality"] == "front_rgb_video")}})
    write_json(OUT / "selected_trajectory_manifest.json", {"schema": "rm-od-selected-trajectories-v1", "source": "frozen RM-Series 100-trajectory selection", "episodes": selected, "episode_count": len(selected), "task_count": len(selected_tasks), "not_used_for_prior_action_consistency_tuning": True})
    write_json(OUT / "source_hash_manifest.json", {"schema": "rm-od-source-hash-manifest-v1", "files": source_entries, "all_hashes_inherited_from_verified_rm_series": True})
    docs = OUT / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "rm_od_plan.md").write_text("# RM-OD plan\n\nThe frozen 100-trajectory RoboMIND2.0 Franka source will be used to build K=8 same-task future visual-consequence retrieval groups. Existing RM-Series logged-action-consistency artifacts are not inputs and will not be modified.\n")
    (docs / "rm_od_results.md").write_text("# RM-OD results\n\nPending task construction and audits. This file will only summarize pre-registered RM-OD results.\n")
    (docs / "rm_od_handoff.md").write_text("# RM-OD handoff\n\nConsult `preregistered_config.json` before running any baseline or method. Method training is prohibited until every K=8 benchmark gate passes.\n")
    append_log("phase0_phase1_frozen", selected_episodes=len(selected), selected_tasks=len(selected_tasks), same_task_local_total=all_same_task, config_sha256=sha(config_path))
    return {"config_sha256": sha(config_path), "episodes": len(selected), "tasks": len(selected_tasks), "same_task_local_total": all_same_task}


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, sort_keys=True))
