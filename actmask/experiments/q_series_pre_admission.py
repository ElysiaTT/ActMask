"""Build a non-mutating Q-Series pre-admission package for the RH20T candidate."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


PROJECT = Path(__file__).resolve().parents[2]
OUT = PROJECT / "outputs" / "actmask" / "q_series_new_data_admission"
META = OUT / "source_inventory" / "rh20t_cfg2"
PARTIAL = OUT / "source_subset" / "rh20t_cfg2" / "file-000.parquet"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.strip() + "\n")


def build() -> dict[str, Any]:
    info = json.loads((META / "info.json").read_text())
    config = json.loads((META / "rh20t_config.json").read_text())
    episodes = json.loads((META / "rh20t_episodes.json").read_text())
    rows = list(episodes.values())
    task_counts: dict[int, int] = {}
    rating_counts: dict[int, int] = {}
    for row in rows:
        task_counts[int(row["task_id"])] = task_counts.get(int(row["task_id"]), 0) + 1
        rating_counts[int(row["rating"])] = rating_counts.get(int(row["rating"]), 0) + 1
    metadata = []
    for path in sorted(META.glob("*")):
        if path.is_file():
            metadata.append({"path": str(path.relative_to(OUT)), "bytes": path.stat().st_size, "sha256": sha(path)})
    candidate = {
        "schema": "q-series-candidate-inventory-v1",
        "selected_candidate": "RH20T cfg2 LeRobot v3 conversion (robot-lev/rh20t_cfg2)",
        "source": {"repository": "https://huggingface.co/datasets/robot-lev/rh20t_cfg2", "revision": "91ebaf641f1d99837170e2338a25e7aa6fa91ff2", "public": True, "gated": False, "license": "per-scene RH20T dual CC BY-SA 4.0 / CC BY-NC 4.0; scene license must be retained per selected episode"},
        "inventory": {"episodes": int(info["total_episodes"]), "tasks": int(info["total_tasks"]), "frames": int(info["total_frames"]), "fps": int(info["fps"]), "robot": info["robot_type"], "cameras": len(config["cameras"]), "state_dim": 15, "action_dim": 8, "force_torque": True, "ratings": rating_counts, "config_reported_failed": int(config["n_failed"]), "task_repeat_min": min(task_counts.values()), "task_repeat_max": max(task_counts.values())},
        "available_fields": sorted(info["features"]),
        "metadata_artifacts": metadata,
        "caution": "The config-level failed count (183) is not equivalent to the 12 episodes whose rating is -1. No binary failure label is inferred until raw numeric rows and upstream semantics are audited.",
    }
    comparison = {"schema": "q-series-candidate-comparison-v1", "candidates": [
        {"name": "RoboMIND", "status": "not_selected", "reason": "Hugging Face metadata is visible but file access remains gated pending user account/contact-condition acceptance; no access was requested or bypassed.", "public_or_permissioned": "permissioned but not granted", "episodes": 107000, "tasks": 479},
        {"name": "RH20T cfg2", "status": "selected_for_pre_admission", "reason": "Public inventory gives 1,789 trajectories, 109 repeated tasks, RGB, state/action, force/torque, ratings, and per-scene licensing.", "public_or_permissioned": "public", "episodes": 1789, "tasks": 109},
    ]}
    transfer = {"schema": "q-series-transfer-attempt-v1", "requested": ["data/chunk-000/file-000.parquet (69,352,058 bytes)", "data/chunk-000/file-001.parquet (53,751,740 bytes)"], "purpose": "minimum numeric state/action/timestamp subset; no videos/depth requested", "result": "incomplete", "partial_file": {"path": str(PARTIAL.relative_to(OUT)), "bytes": PARTIAL.stat().st_size if PARTIAL.exists() else 0, "valid_parquet": False}, "observed_failures": ["direct CDN transfer reset before footer", "Hugging Face downloader stalled", "HTTP Range probe failed with SSL_ERROR_SYSCALL to us.aws.cdn.hf.co", "the official datasets-server rows probe exposes only the repository's audio viewer config, not the LeRobot numeric table", "a separate wget 1 MiB Range probe completed with a zero-byte output"], "safe_retry": "Use a stable network or manually supplied complete checksummed Parquet files; retain source revision and verify PAR1 footer plus SHA-256 before parsing."}
    checks = [
        {"id": 1, "requirement": "public or permissioned real robot source", "status": "pass", "evidence": "public RH20T cfg2 repository and per-scene license metadata"},
        {"id": 2, "requirement": "at least 100 aligned trajectories", "status": "provisional_pass", "evidence": "remote metadata lists 1,789 episodes; alignment needs raw-row audit"},
        {"id": 3, "requirement": "repeated tasks or outcome/failure labels", "status": "provisional_pass", "evidence": "109 tasks, each repeated 4–50 times; rating field exists"},
        {"id": 4, "requirement": "no future leakage", "status": "not_evaluated", "evidence": "requires complete numeric/video subset"},
        {"id": 5, "requirement": "shortcut baselines <= 0.65", "status": "not_evaluated", "evidence": "requires held-out raw subset"},
        {"id": 6, "requirement": "fair recoverability model >= 0.65", "status": "not_evaluated", "evidence": "requires complete causal features"},
        {"id": 7, "requirement": "best standard baseline < 0.90", "status": "not_evaluated", "evidence": "requires held-out raw subset"},
        {"id": 8, "requirement": "meaningful oracle/upper bound exceeds standard by >= 0.08", "status": "not_evaluated", "evidence": "requires an isolated, non-fair diagnostic and complete labels"},
        {"id": 9, "requirement": "raw-score and artifact logging passes", "status": "not_evaluated", "evidence": "no training or score generation is authorized before checks 1–8"},
    ]
    gate = {"schema": "q-series-gate-status-v1", "all_checks_pass": False, "method_authorized": False, "checks": checks, "rule": "No RealRelDynVerifier or VisualRelDynVerifier trial may run while any check is provisional, not evaluated, or failed."}
    missing = {"schema": "q-series-missing-evidence-v1", "blocking_evidence": [
        "Complete, checksummed numeric Parquet files for the selected source; the local partial file has no valid Parquet footer.",
        "A deterministic selection of >=100 episodes retaining per-scene licensing and repeated-task groups.",
        "Raw-image/video access for a visual admission path, or an explicit decision to audit the numeric/force path only.",
        "Upstream semantics mapping meta.rating/config n_failed to a valid outcome or failure target; no label is inferred from the current metadata.",
    ], "not_missing": ["Public source inventory", "trajectory/task count", "modality schema", "state/action/force presence", "license boundary"], "next_action": "Retry only the two numeric files from a stable network, verify them, then select a bounded >=100-episode subset before running any admission baseline."}
    write_json(OUT / "candidate_inventory.json", candidate)
    write_json(OUT / "candidate_comparison.json", comparison)
    write_json(OUT / "transfer_attempt.json", transfer)
    write_json(OUT / "future_method_gate_status.json", gate)
    write_json(OUT / "missing_evidence.json", missing)
    write_text(OUT / "missing_evidence.md", "# Missing evidence\n\nThe selected RH20T source is qualified for pre-admission, but not for a method trial. Complete numeric files and verified outcome semantics are required before causality, splitting, controls, recoverability, oracle headroom, or raw-score logging can be evaluated. The partial Parquet is intentionally excluded from every result.\n")
    write_text(OUT / "docs/q_series_plan.md", "# Q-Series plan\n\nQ-Series selects a public RH20T candidate, acquires only the smallest numeric subset needed for admission, then applies the nine fixed P-Series checks. No method training, BotFails tuning, external request submission, or frozen-artifact modification is permitted.\n")
    write_text(OUT / "docs/q_series_results.md", "# Q-Series results\n\nRH20T cfg2 is selected over currently gated RoboMIND: its remote inventory exposes 1,789 episodes, 109 repeated tasks, RGB, state/action, force/torque, ratings, and per-scene licenses. The local RH20T pilot remains three trajectories only and is not used. Minimal numeric-file transfer is incomplete because the CDN connection resets; no causal, baseline, recoverability, oracle, or method result exists.\n")
    write_text(OUT / "docs/q_series_handoff.md", "# Q-Series handoff\n\nResume from `transfer_attempt.json` only after a stable network is available. Verify both Parquet files before parsing, derive an explicit rating/outcome target from source semantics, select >=100 grouped episodes, and then run the fixed gate in order. Do not use the partial file or authorize a method.\n")
    write_text(OUT / "data_selection.md", "# Candidate data selection\n\nSelect RH20T cfg2 after complete numeric files are available. Retain only episodes from a license-compatible scene set, at least 100 total, with multiple repeated task IDs and grouped train/validation/test splits by episode. Preserve task, scene, user, rating, source folder/license, timestamps, state, action, force/torque, and camera references. Prefer a rating/outcome-balanced subset only after verifying the source's rating semantics; do not manufacture failure-onset labels.\n")
    final = {"schema": "q-series-final-decision-v1", "decision": "Q_ADMISSION_PENDING_VERIFIED_MINIMUM_SUBSET", "selected_candidate": "RH20T cfg2", "method_authorized": False, "reason": "The remote inventory satisfies basic source-scale and modality criteria, but a complete verified numeric subset and outcome semantics are unavailable locally; six gate checks are therefore not evaluable.", "prohibited_actions": ["method training", "BotFails tuning", "using the partial Parquet", "inferring a binary failure label from rating without source validation"]}
    write_json(OUT / "final_decision.json", final)
    write_text(OUT / "final_report.md", "# Q-Series pre-admission report\n\n**Q_ADMISSION_PENDING_VERIFIED_MINIMUM_SUBSET.** RH20T cfg2 is the selected source. It is public, has sufficient scale and repeated tasks, and exposes the needed state/action/force/RGB schema. However, the two smallest numeric files could not be fully transferred from the current network and the rating-to-outcome mapping is not yet verified. The strict gate is therefore incomplete and no method claim or method trial is authorized.\n")
    generated = sorted(path for path in OUT.rglob("*") if path.is_file() and path.name not in {"reproducibility_manifest.json", "package_manifest.json"} and "source_subset" not in path.relative_to(OUT).parts)
    repro = {"schema": "q-series-reproducibility-manifest-v1", "generator": str(Path(__file__).relative_to(PROJECT)), "generator_sha256": sha(Path(__file__)), "metadata_files": metadata, "generated_files": [{"path": str(path.relative_to(OUT)), "sha256": sha(path)} for path in generated], "no_training": True, "partial_file_excluded": True}
    write_json(OUT / "reproducibility_manifest.json", repro)
    package = sorted(path for path in OUT.rglob("*") if path.is_file() and path.name != "package_manifest.json" and "source_subset" not in path.relative_to(OUT).parts)
    write_json(OUT / "package_manifest.json", {"schema": "q-series-package-manifest-v1", "decision": final["decision"], "files": [{"path": str(path.relative_to(OUT)), "sha256": sha(path)} for path in package]})
    return {"selected": "RH20T cfg2", "gate_pass": False, "output": str(OUT)}


if __name__ == "__main__":
    print(json.dumps(build(), sort_keys=True))
