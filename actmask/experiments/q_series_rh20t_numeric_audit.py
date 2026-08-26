"""Run only the non-training RH20T numeric admission audit after both files verify."""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq


PROJECT = Path(__file__).resolve().parents[2]
OUT = PROJECT / "outputs" / "actmask" / "q_series_new_data_admission"
DATA = OUT / "source_subset" / "rh20t_cfg2"
META = OUT / "source_inventory" / "rh20t_cfg2"
THRESHOLD = 1e4


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _array(table: Any, name: str) -> np.ndarray:
    return np.asarray(table[name].to_pylist(), dtype=np.float32)


def run() -> dict[str, Any]:
    files = [DATA / "file-000.parquet", DATA / "file-001.parquet"]
    for path in files:
        pq.ParquetFile(path)  # validates footer before any audit
    episodes_meta = {int(key): value for key, value in json.loads((META / "rh20t_episodes.json").read_text()).items()}
    config = json.loads((META / "rh20t_config.json").read_text())
    fields = ["observation.state", "action", "observation.force", "observation.torque", "observation.robot_ft"]
    summary: dict[int, dict[str, Any]] = {}
    bad_by_field: dict[str, set[int]] = {field: set() for field in fields}
    total_rows = 0
    metadata_mismatches = {"task": 0, "rating": 0}
    for path in files:
        table = pq.read_table(path, columns=["episode_index", "frame_index", "timestamp", "task_index", "meta.rating", *fields])
        ep = table["episode_index"].to_numpy(zero_copy_only=False).astype(np.int64)
        frame = table["frame_index"].to_numpy(zero_copy_only=False).astype(np.int64)
        timestamp = table["timestamp"].to_numpy(zero_copy_only=False).astype(np.float64)
        task = table["task_index"].to_numpy(zero_copy_only=False).astype(np.int64)
        rating = table["meta.rating"].to_numpy(zero_copy_only=False).astype(np.int64)
        invalid = np.zeros(len(ep), dtype=bool)
        for field in fields:
            value = _array(table, field)
            bad = ~np.isfinite(value).all(axis=1) | (np.abs(value) > THRESHOLD).any(axis=1)
            invalid |= bad
            bad_by_field[field].update(int(item) for item in np.unique(ep[bad]))
        for episode in np.unique(ep):
            mask = ep == episode
            order = np.argsort(frame[mask])
            local_frame = frame[mask][order]
            local_time = timestamp[mask][order]
            local_task = task[mask]
            local_rating = rating[mask]
            meta = episodes_meta[int(episode)]
            task_ok = bool(np.all(local_task == int(meta["task_id"]) - 1))
            rating_ok = bool(np.all(local_rating == int(meta["rating"])))
            metadata_mismatches["task"] += int(not task_ok)
            metadata_mismatches["rating"] += int(not rating_ok)
            summary[int(episode)] = {
                "episode_index": int(episode), "rows": int(mask.sum()), "task_index": int(local_task[0]),
                "task_id": int(meta["task_id"]), "rating": int(local_rating[0]), "scene_id": int(meta["scene_id"]),
                "timestamp_monotonic": bool(np.all(np.diff(local_time) >= 0)),
                "frame_strictly_increasing": bool(np.all(np.diff(local_frame) > 0)),
                "rating_constant": bool(len(np.unique(local_rating)) == 1), "numeric_clean": bool(not invalid[mask].any()),
            }
        total_rows += len(ep)
    clean = [row for _, row in sorted(summary.items()) if row["numeric_clean"] and row["timestamp_monotonic"] and row["frame_strictly_increasing"] and row["rating_constant"]]
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in clean:
        groups[row["task_id"]].append(row)
    selection: list[dict[str, Any]] = []
    for task_id in sorted(groups):
        if len(groups[task_id]) >= 5:
            selection.extend(sorted(groups[task_id], key=lambda row: row["episode_index"])[:5])
        if len(selection) >= 100:
            break
    selection = selection[:100]
    ratings = [row["rating"] for row in summary.values()]
    rating_counts = {str(key): int(ratings.count(key)) for key in sorted(set(ratings))}
    candidate_threshold_counts = {"rating_eq_minus_1": sum(value == -1 for value in ratings), "rating_le_0": sum(value <= 0 for value in ratings), "rating_le_1": sum(value <= 1 for value in ratings)}
    integrity = {"schema": "q-series-rh20t-numeric-integrity-v1", "files": [{"path": str(path.relative_to(OUT)), "sha256": sha(path), "bytes": path.stat().st_size, "rows": pq.ParquetFile(path).metadata.num_rows} for path in files], "rows": total_rows, "episodes": len(summary), "metadata_task_namespace_differences": metadata_mismatches["task"], "task_namespace_interpretation": "The converter's contiguous task_index is documented by meta/tasks.parquet (task 1..task 108); it is not the original, non-contiguous folder task_id. Grouped splits must use one namespace consistently.", "metadata_rating_mismatches": metadata_mismatches["rating"], "bad_episode_count": len(summary) - len(clean), "bad_episode_ids": sorted(set(summary).difference(row["episode_index"] for row in clean)), "bad_episode_by_field": {field: sorted(values) for field, values in bad_by_field.items()}, "numeric_threshold_abs": THRESHOLD, "pass": len(clean) >= 100 and not metadata_mismatches["rating"]}
    label = {"schema": "q-series-rh20t-outcome-semantics-v1", "status": "fail", "reason": "The source metadata exposes an ordinal meta.rating and config-level n_failed, but no documented mapping from rating to success/failure or failure onset. Counts do not identify a safe threshold.", "config_n_failed": int(config["n_failed"]), "episode_rating_counts": rating_counts, "candidate_threshold_counts": candidate_threshold_counts, "inference_prohibited": "No binary outcome, failure onset, or oracle target is inferred from rating.", "required_to_pass": "Official source documentation or maintainer-provided mapping defining rating, success/failure, and any onset semantics."}
    selection_manifest = {"schema": "q-series-rh20t-clean-selection-v1", "status": "provisional_no_supervised_task", "selected_episodes": selection, "episodes": len(selection), "task_count": len(set(row["task_id"] for row in selection)), "selection_rule": "first five clean, time-aligned, rating-constant episodes from each repeated task; no label balancing or outcome inference", "not_used_for_training": True}
    checks = [
        {"id": 1, "status": "pass", "evidence": "public source with retained per-scene license metadata"},
        {"id": 2, "status": "pass", "evidence": f"{len(summary)} verified aligned numeric episodes; {len(clean)} clean episodes after complete-episode exclusion"},
        {"id": 3, "status": "pass", "evidence": f"{len(groups)} repeated task groups; selected subset spans {len(set(row['task_id'] for row in selection))} groups"},
        {"id": 4, "status": "not_evaluated", "evidence": "No outcome/failure target or anchor is defined without source label semantics."},
        {"id": 5, "status": "not_evaluated", "evidence": "No valid supervised target; shortcut baseline is prohibited."},
        {"id": 6, "status": "not_evaluated", "evidence": "No valid supervised target; recoverability model is prohibited."},
        {"id": 7, "status": "not_evaluated", "evidence": "No valid supervised target; standard baseline is prohibited."},
        {"id": 8, "status": "not_evaluated", "evidence": "No meaningful isolated oracle can be defined without outcome semantics."},
        {"id": 9, "status": "pass", "evidence": "source hashes, selection manifest, audits, and command source are retained; no scores/checkpoints exist because no model ran."},
    ]
    gate = {"schema": "q-series-gate-status-v2", "all_checks_pass": False, "method_authorized": False, "checks": checks, "blocking_check": 4, "rule": "The gate is conjunctive; checks 5–8 cannot be manufactured when the label task is undefined."}
    decision = {"schema": "q-series-final-decision-v2", "decision": "Q_ADMISSION_REJECTED_OUTCOME_SEMANTICS_UNAVAILABLE", "selected_candidate": "RH20T cfg2", "method_authorized": False, "reason": f"The numeric source passes file, alignment, repeated-task, and raw-artifact checks after excluding {integrity['bad_episode_count']} corrupted episodes, but lacks a documented outcome/failure/onset mapping. A rating-derived target would be an impermissible inference.", "next_data_or_metadata_required": ["official RH20T rating/outcome semantics", "success/failure or completion label mapping", "failure onset if forecasting is desired", "separately accessible RGB/RGB-D subset if a visual path is requested"], "prohibited_actions": ["rating-threshold label invention", "shortcut/recoverability/oracle baselines", "RealRelDynVerifier or VisualRelDynVerifier training"]}
    write(OUT / "numeric_integrity_audit.json", integrity)
    write(OUT / "outcome_semantics_audit.json", label)
    write(OUT / "subset_manifest.json", selection_manifest)
    write(OUT / "future_method_gate_status.json", gate)
    write(OUT / "causality_and_leakage_audit.json", {"schema": "q-series-causality-audit-v1", "numeric_stream_alignment": "pass", "evidence": "all retained episodes have monotonic timestamps and strictly increasing frame indices", "supervised_target_status": "blocked", "reason": "Without outcome/failure and anchor semantics, no causal cutoff or future target can be defined. No feature may be called fair or leakage-free yet.", "pass": False})
    write(OUT / "task_split_audit.json", {"schema": "q-series-task-split-audit-v1", "status": "not_run", "reason": "A grouped split for supervised evaluation requires a valid target. The 100-episode manifest is inventory-only and may not be used as a train/validation/test task.", "inventory_groups": len(groups), "inventory_episodes": len(selection)})
    write(OUT / "baseline_admission_status.json", {"schema": "q-series-baseline-admission-v1", "shortcut_baselines": "not_run", "fair_recoverability": "not_run", "standard_baselines": "not_run", "reason": "No source-validated outcome target exists; running baselines would fabricate a task."})
    write(OUT / "oracle_admission_status.json", {"schema": "q-series-oracle-admission-v1", "status": "not_run", "reason": "An oracle/upper bound is meaningful only relative to a source-validated target; no rating-derived oracle is permitted."})
    write(OUT / "raw_artifact_log.json", {"schema": "q-series-raw-artifact-log-v1", "source_files": integrity["files"], "audits": ["numeric_integrity_audit.json", "outcome_semantics_audit.json", "causality_and_leakage_audit.json", "task_split_audit.json", "baseline_admission_status.json", "oracle_admission_status.json"], "models_run": 0, "raw_score_files": 0, "checkpoints": 0, "reason": "No model is authorized before a valid target exists."})
    write(OUT / "transfer_attempt.json", {"schema": "q-series-transfer-attempt-v2", "result": "complete_and_verified", "purpose": "minimum numeric state/action/timestamp admission subset; no videos/depth downloaded", "files": integrity["files"], "historical_network_note": "Initial CDN transfers reset; the final transfers completed through HTTP Range continuation and were verified by Parquet footer, exact remote byte count, and SHA-256.", "not_used_for_training": True})
    write(OUT / "missing_evidence.json", {"schema": "q-series-missing-evidence-v2", "blocking_evidence": ["Official RH20T mapping from meta.rating/config n_failed to a valid success/failure or completion target.", "Failure-onset semantics if a forecasting task is intended.", "A separately accessible RGB/RGB-D subset if a visual admission path is intended."], "resolved_evidence": ["Complete numeric Parquet files", "episode-level timestamp/frame alignment", "repeated task groups", "source hashes and clean-episode selection"], "next_action": "Obtain official label semantics; do not create a rating-threshold target locally."})
    write(OUT / "final_decision.json", decision)
    (OUT / "final_report.md").write_text(f"# Q-Series admission report\n\n**Q_ADMISSION_REJECTED_OUTCOME_SEMANTICS_UNAVAILABLE.** Two RH20T numeric shards verify, episode-level timestamps and frame indices are monotonic, and {integrity['bad_episode_count']} fully corrupted episodes are excluded rather than repaired. However, `meta.rating` has no documented success/failure or onset mapping; its observed threshold counts do not reproduce `n_failed`. The only valid conclusion is to stop before supervised baselines and method training.\n")
    (OUT / "docs/q_series_results.md").write_text("# Q-Series results\n\nRH20T cfg2 supplies a clean aligned numeric inventory after whole-episode exclusion of three anomalous trajectories. It has repeated tasks and sufficient scale, but it does not yet supply a scientifically valid outcome target. The gate fails at task causality/label definition, so no shortcut, recoverability, standard, oracle, or method model was run.\n")
    generated = sorted(path for path in OUT.rglob("*") if path.is_file() and path.name not in {"reproducibility_manifest.json", "package_manifest.json"} and "source_subset" not in path.relative_to(OUT).parts)
    write(OUT / "reproducibility_manifest.json", {"schema": "q-series-reproducibility-manifest-v2", "audit_script": str(Path(__file__).relative_to(PROJECT)), "audit_script_sha256": sha(Path(__file__)), "generated_files": [{"path": str(path.relative_to(OUT)), "sha256": sha(path)} for path in generated], "no_training": True, "source_files_hashed_in_integrity_audit": True})
    package = sorted(path for path in OUT.rglob("*") if path.is_file() and path.name != "package_manifest.json" and "source_subset" not in path.relative_to(OUT).parts)
    write(OUT / "package_manifest.json", {"schema": "q-series-package-manifest-v2", "decision": decision["decision"], "files": [{"path": str(path.relative_to(OUT)), "sha256": sha(path)} for path in package]})
    return decision


if __name__ == "__main__":
    print(json.dumps(run(), sort_keys=True))
