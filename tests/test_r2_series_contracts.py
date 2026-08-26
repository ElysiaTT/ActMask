"""Artifact-level R2 contracts: alignment, leakage, scores, and final schema."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from actmask.data.r2_real_outcome_adapter import R2_ROOT, _validate_row
from actmask.data.real_robot_dataset_adapter import canonical_json, sha256_bytes
from actmask.experiments.r2_botfails_verification import _metrics


ROOT = R2_ROOT
PROCESSED = ROOT / "processed_subset" / "botfails"


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def test_r2_adapter_loads_every_episode_with_exact_labels_and_alignment() -> None:
    rows = _rows(PROCESSED / "episodes.jsonl")
    assert len(rows) == 100
    audited = [_validate_row(row, PROCESSED) for row in rows]
    assert sum(row["positive_steps"] > 0 for row in audited) == 50
    assert {row["success_or_failure"] for row in rows} == {"success", "failure"}
    assert all(row["rgb_paths"] == [] for row in rows)


def test_r2_formulation_keeps_pairs_groups_negatives_and_future_separate() -> None:
    task = ROOT / "task_formulation" / "formulation_a_failure_onset"
    rows = _rows(task / "samples.jsonl")
    audit = json.loads((task / "task_audit.json").read_text())
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["pair_id"]].append(row)
        assert row["history_start"] < row["anchor"] < row["candidate_end_exclusive"]
    assert len(grouped) == 50
    assert all(len(pair) == 2 and {row["target_failure_in_candidate"] for row in pair} == {0, 1} for pair in grouped.values())
    assert all(len({row["hard_split"] for row in pair}) == 1 for pair in grouped.values())
    assert audit["leakage_audit"]["pass"]
    assert audit["source_partition_counts"] == {"test": 100}


def test_r2_raw_scores_recompile_to_reported_metrics_and_shortcut_gate() -> None:
    report = json.loads((ROOT / "baseline_results" / "formulation_a_baseline_report.json").read_text())
    raw_dir = ROOT / report["raw_score_dir"]
    for name, expected in report["metrics"].items():
        rows = _rows(raw_dir / f"{name}.jsonl")
        labels = np.asarray([row["label"] for row in rows], dtype=np.int64)
        scores = np.asarray([row["score"] for row in rows], dtype=np.float32)
        actual = _metrics(labels, scores)
        assert actual["n"] == expected["n"] == 20
        assert actual["balanced_accuracy"] == expected["balanced_accuracy"]
        assert actual["auroc"] == expected["auroc"]
    assert report["gate"]["pass"]
    assert report["gate"]["shortcut_best"] <= report["gate"]["shortcut_max"]


def test_r2_method_summary_and_final_package_have_rejection_schema() -> None:
    package = ROOT / "final_package"
    decision = json.loads((package / "final_decision.json").read_text())
    method = json.loads((package / "method_summary.json").read_text())
    repro = json.loads((package / "reproducibility_manifest.json").read_text())
    assert decision["decision"] == "REJECT_CURRENT_PAPER_METHOD_CLAIM"
    assert method["method_trial_count"] == 1
    assert not method["authorized"]
    assert method["improvement"] < method["required_improvement"]
    for relative, digest in repro["artifact_sha256"].items():
        import hashlib

        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest


def test_r2_hash_chained_log_and_backup_block_reason_are_preserved() -> None:
    rows = _rows(ROOT / "run_log.jsonl")
    assert [row["seq"] for row in rows] == list(range(1, len(rows) + 1))
    previous = ""
    for row in rows:
        assert row["prev_event_sha256"] == previous
        unsigned = {key: value for key, value in row.items() if key != "event_sha256"}
        assert sha256_bytes(canonical_json(unsigned).encode()) == row["event_sha256"]
        previous = row["event_sha256"]
    backup = json.loads((ROOT / "backup_probe.json").read_text())
    assert backup["download_probe"]["http_status"] == 401
    assert backup["download_probe"]["error_code"] == "GatedRepo"
