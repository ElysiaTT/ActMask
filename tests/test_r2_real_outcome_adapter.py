"""Offline contracts for the R2 BotFails real-outcome adapter."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from actmask.data.r2_real_outcome_adapter import (
    EXPECTED_TASKS,
    R2FormatError,
    _parquet_arrays,
    _validate_row,
    _within_task_splits,
    label_inventory,
    select_subset,
)


def _labels(root: Path) -> None:
    for task in EXPECTED_TASKS:
        directory = root / "labels" / f"{task}_anomaly"
        directory.mkdir(parents=True)
        for index in range(5):
            values = ["0"] * 80 + ["1"] * 3
            (directory / f"episode_{index:06d}_labels.csv").write_text("\n".join(values) + "\n")


def test_selection_uses_official_binary_csv_and_has_balanced_100_episode_contract(tmp_path: Path) -> None:
    _labels(tmp_path)
    inventory = label_inventory(tmp_path)
    selected = select_subset(tmp_path)
    assert len(inventory) == len(EXPECTED_TASKS)
    assert len(selected) == 100
    assert sum(row["outcome"] == "success" for row in selected) == 50
    assert sum(row["outcome"] == "failure" for row in selected) == 50
    assert all(row["label_rel"] is not None for row in selected if row["outcome"] == "failure")


def test_within_task_split_is_episode_level_and_exactly_3_1_1_per_outcome() -> None:
    rows = [
        {"task_name": task, "outcome": outcome, "episode_index": index}
        for task in EXPECTED_TASKS
        for outcome in ("success", "failure")
        for index in range(5)
    ]
    splits = _within_task_splits(rows)
    assert len(splits) == 100
    for task in EXPECTED_TASKS:
        for outcome in ("success", "failure"):
            assigned = [splits[(task, outcome, index)] for index in range(5)]
            assert assigned.count("train") == 3
            assert assigned.count("validation") == 1
            assert assigned.count("test") == 1


def test_parquet_schema_and_alignment_are_read_without_a_future_column(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    table = pa.table({
        "action": [[0.0] * 6, [1.0] * 6, [2.0] * 6, [3.0] * 6],
        "observation.state": [[0.5] * 6, [1.5] * 6, [2.5] * 6, [3.5] * 6],
        "timestamp": [0.0, 0.1, 0.2, 0.3],
        "frame_index": [0, 1, 2, 3], "episode_index": [2, 2, 2, 2],
        "index": [8, 9, 10, 11], "task_index": [0, 0, 0, 0],
    })
    path = tmp_path / "episode.parquet"
    pq.write_table(table, path)
    state, action, timestamp, info = _parquet_arrays(path)
    assert state.shape == action.shape == (4, 6)
    assert timestamp.tolist() == pytest.approx([0.0, 0.1, 0.2, 0.3])
    assert info["rows"] == 4


def test_normalized_audit_rejects_unaligned_or_mislabeled_episode(tmp_path: Path) -> None:
    root = tmp_path / "processed"
    root.mkdir()
    for name, value in {
        "observation_path": np.zeros((4, 6), np.float32), "robot_state_path": np.zeros((4, 6), np.float32),
        "action_path": np.zeros((4, 6), np.float32), "timestamp_path": np.asarray([0, 1, 2, 3], np.float32),
        "failure_label_path": np.asarray([0, 0, 0, 0], np.int8),
    }.items():
        np.savez_compressed(root / f"{name}.npz", value=value)
    row = {
        "episode_id": "normal", "success_or_failure": "success",
        **{name: f"{name}.npz" for name in ("observation_path", "robot_state_path", "action_path", "timestamp_path", "failure_label_path")},
    }
    assert _validate_row(row, root)["steps"] == 4
    row["success_or_failure"] = "failure"
    with pytest.raises(R2FormatError, match="lacks official positive"):
        _validate_row(row, root)
