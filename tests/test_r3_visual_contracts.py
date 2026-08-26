"""R3 artifact contracts for video identity, causality, features, and decision."""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

from actmask.data.r3_botfails_video_adapter import _remote_video_path
from actmask.experiments.r3_botfails_visual_baselines import _metrics


ROOT = Path(__file__).resolve().parents[1] / "outputs" / "actmask" / "r3_botfails_visual_failure"
R2 = Path(__file__).resolve().parents[1] / "outputs" / "actmask" / "r2_series_real_outcome_search" / "processed_subset" / "botfails"


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def test_r3_preregistration_hash_and_full_video_identity_alignment() -> None:
    config = ROOT / "preregistered_config.json"
    digest = (ROOT / "preregistered_config.sha256").read_text().split()[0]
    assert hashlib.sha256(config.read_bytes()).hexdigest() == digest
    source = json.loads((ROOT / "source_video_manifest.json").read_text())
    audit = json.loads((ROOT / "video_alignment_audit.json").read_text())
    r2 = {row["episode_id"]: row for row in _rows(R2 / "episodes.jsonl")}
    assert source["episodes"] == 100 and audit["pass"] and len(audit["rows"]) == 100
    for row in audit["rows"]:
        assert row["decoded_frames"] == row["numeric_timestamp_count"]
        assert abs(row["decoded_fps"] - row["numeric_rate_hz"]) < 0.25
        assert _remote_video_path(r2[row["episode_id"]]) == row["remote_video_path"]
        assert (ROOT / row["local_video_path"]).is_file()


def test_r3_windows_are_causal_grouped_and_match_horizon_labels() -> None:
    task = ROOT / "task_formulation"
    rows = _rows(task / "windows.jsonl")
    labels = np.load(task / "numeric_windows.npz")["label"]
    leakage = json.loads((task / "leakage_audit.json").read_text())
    split = json.loads((task / "split_audit.json").read_text())
    matching = json.loads((task / "window_matching_audit.json").read_text())
    assert len(rows) == len(labels) == 600
    assert leakage["pass"] and split["pass"] and matching["pass"]
    groups: dict[str, list[dict]] = defaultdict(list)
    for row, label in zip(rows, labels):
        groups[row["episode_id"]].append(row)
        assert max(row["rgb_indices"]) < row["anchor"]
        assert row["history_start"] < min(row["rgb_indices"])
        assert row["candidate_action_end_exclusive"] >= row["anchor"]
        assert int(row["target_failure_within_horizon"]) == int(label)
    assert len(groups) == 50
    assert all(len({row["episode_split"] for row in value}) == len({row["task_split"] for row in value}) == 1 for value in groups.values())
    assert sum(labels) == 150


def test_r3_frozen_feature_manifest_and_hashes_are_reproducible() -> None:
    manifest = json.loads((ROOT / "processed_features" / "feature_manifest.json").read_text())
    feature = ROOT / "processed_features" / "visual_features.npz"
    values = np.load(feature)
    assert manifest["samples"] == 600
    assert values["mode_a"].shape == (600, 5)
    assert values["resnet_history"].shape == (600, 4, 512)
    assert hashlib.sha256(feature.read_bytes()).hexdigest() == manifest["feature_file_sha256"]
    assert len(set(values["frame_hash"].tolist())) > 100


def test_r3_raw_scores_recompute_and_final_decision_schema() -> None:
    report = json.loads((ROOT / "baseline_results" / "visual_baseline_report.json").read_text())
    assert set(report["reports_by_split"]) == {"task_heldout", "episode_heldout"}
    # Each reported raw-score file must contain exactly the held-out rows and
    # only score/label metadata, never a post-anchor frame reference.
    for split_name, horizons in report["reports_by_split"].items():
        for horizon, record in horizons.items():
            for name, metric in record["metrics"].items():
                path = ROOT / "baseline_results" / "raw_scores" / record["split_key"] / f"h{horizon}" / f"{name}.jsonl"
                rows = _rows(path)
                assert len(rows) == metric["n"]
                assert all("score_mean" in row and "label" in row and "future" not in " ".join(row) for row in rows)
                actual = _metrics(
                    np.asarray([row["label"] for row in rows], dtype=np.int64),
                    np.asarray([row["score_mean"] for row in rows], dtype=np.float64),
                    metric["threshold"],
                )
                for field in ("accuracy", "balanced_accuracy", "auroc", "threshold"):
                    assert actual[field] == pytest.approx(metric[field])
    decision = json.loads((ROOT / "final_package" / "final_decision.json").read_text())
    assert decision["decision"] in {
        "R3_REAL_VISUAL_METHOD_SUPPORTED", "R3_REAL_VISUAL_BENCHMARK_SUPPORTED_METHOD_NOT_YET",
        "R3_FAILURE_NOT_PREDICTABLE_FROM_AVAILABLE_PREHISTORY", "R3_STANDARD_VISUAL_BASELINES_SATURATE",
        "R3_RELATIONAL_METHOD_NOT_SUPPORTED", "R3_VIDEO_ALIGNMENT_INVALID", "R3_DATA_ACCESS_BLOCKED",
    }
    status = json.loads((ROOT / "final_package" / "method_status.json").read_text())
    assert status["status"] == "not_run"
