"""Small, offline tests for the R-Series real-robot adapter contracts."""
from __future__ import annotations

import struct

import numpy as np
import pytest

from actmask.data.droid_adapter import (
    DroidFormatError,
    _masked_crc32c,
    decode_droid_episode,
    iter_tfrecord,
    parse_tf_example,
)
from actmask.data.real_robot_dataset_adapter import append_run_log, forbidden_input_audit, read_jsonl, stable_episode_split
from actmask.experiments.r_series_real_robot_verification import _candidate_sets, _experiment_splits, _rank_metrics
from actmask.experiments.r_series_asu_verification import _candidate_rows as strict_asu_candidate_rows
from actmask.experiments.r_series_asu_verification import _rank_metrics as strict_asu_rank_metrics


def _varint(value: int) -> bytes:
    result = bytearray()
    while value > 127:
        result.append((value & 127) | 128)
        value >>= 7
    result.append(value)
    return bytes(result)


def _field(number: int, wire: int, value: bytes | int) -> bytes:
    key = _varint((number << 3) | wire)
    if wire == 0:
        return key + _varint(int(value))
    if wire == 2:
        assert isinstance(value, bytes)
        return key + _varint(len(value)) + value
    if wire == 5:
        assert isinstance(value, bytes) and len(value) == 4
        return key + value
    raise ValueError(wire)


def _feature_bytes(values: list[bytes]) -> bytes:
    return _field(1, 2, b"".join(_field(1, 2, value) for value in values))


def _feature_float(values: np.ndarray) -> bytes:
    return _field(2, 2, _field(1, 2, np.asarray(values, dtype="<f4").tobytes()))


def _feature_int(values: list[int]) -> bytes:
    return _field(3, 2, b"".join(_field(1, 0, value) for value in values))


def _example(features: dict[str, bytes]) -> bytes:
    entries = []
    for key, feature in sorted(features.items()):
        entries.append(_field(1, 2, _field(1, 2, key.encode()) + _field(2, 2, feature)))
    return _field(1, 2, b"".join(entries))


def _fake_droid_record(*, language: bytes = b"Put a marker in the pot") -> bytes:
    steps = 4
    features = {
        "episode_metadata/file_path": _feature_bytes([b"episode/trajectory.h5"]),
        "episode_metadata/recording_folderpath": _feature_bytes([b"episode"]),
        "steps/action": _feature_float(np.arange(steps * 7)),
        "steps/observation/cartesian_position": _feature_float(np.arange(steps * 6)),
        "steps/observation/joint_position": _feature_float(np.arange(steps * 7)),
        "steps/observation/gripper_position": _feature_float(np.arange(steps)),
        "steps/observation/exterior_image_1_left": _feature_bytes([b"fake-jpeg"] * steps),
        "steps/language_instruction": _feature_bytes([language] * steps),
        "steps/is_terminal": _feature_int([0, 0, 0, 1]),
        "steps/reward": _feature_float(np.asarray([0, 0, 0, 1], dtype=np.float32)),
    }
    return _example(features)


def test_minimal_tf_example_decodes_real_adapter_contract() -> None:
    record = _fake_droid_record()
    parsed = parse_tf_example(record)
    assert parsed["steps/action"][0] == "float"
    episode = decode_droid_episode(record, source_shard="unit.tfrecord", source_record_index=2)
    assert episode["robot_state"].shape == (4, 14)
    assert episode["action"].shape == (4, 7)
    assert episode["task_id"] == "language_verb_put"
    assert episode["success"] is True


def test_missing_language_fails_closed() -> None:
    with pytest.raises(DroidFormatError, match="no usable language"):
        decode_droid_episode(_fake_droid_record(language=b""), source_shard="unit", source_record_index=0)


def test_tfrecord_crc_validation(tmp_path) -> None:
    record = _fake_droid_record()
    payload = struct.pack("<Q", len(record))
    path = tmp_path / "sample.tfrecord"
    path.write_bytes(payload + struct.pack("<I", _masked_crc32c(payload)) + record + struct.pack("<I", _masked_crc32c(record)))
    assert list(iter_tfrecord(path)) == [record]
    broken = tmp_path / "broken.tfrecord"
    broken.write_bytes(payload + struct.pack("<I", 0) + record + struct.pack("<I", _masked_crc32c(record)))
    with pytest.raises(DroidFormatError, match="length CRC"):
        list(iter_tfrecord(broken))


def test_fair_contract_and_hash_chained_log(tmp_path) -> None:
    audit = forbidden_input_audit(["history_robot_state", "candidate_action_chunk"])
    assert audit["pass"]
    assert not forbidden_input_audit(["candidate_id"])["pass"]
    path = tmp_path / "run_log.jsonl"
    first = append_run_log(path, event="first", payload={"x": 1})
    second = append_run_log(path, event="second", payload={"x": 2})
    rows = read_jsonl(path)
    assert [row["seq"] for row in rows] == [1, 2]
    assert second["prev_event_sha256"] == first["event_sha256"]
    assert stable_episode_split("episode-a") == stable_episode_split("episode-a")


def test_ranking_metrics_are_tie_aware_and_candidate_sets_keep_sources_in_split() -> None:
    rows = []
    for candidate in range(5):
        rows.append({"candidate_set_id": "set-a", "candidate_id": str(candidate), "split": "test", "consistent": candidate == 0})
    metric = _rank_metrics(rows, np.zeros(5), "test")
    assert metric["pair_order_accuracy"] == 0.5
    assert metric["top1_success"] == pytest.approx(0.2)

    def anchor(episode: str, task: str, step: int) -> dict:
        return {
            "anchor_id": f"{episode}:{step}",
            "episode_id": episode,
            "task_id": task,
            "split": "train",
            "anchor": step,
            "history_state": np.zeros((32, 14), np.float32),
            "candidate_action": np.full((8, 7), step, np.float32),
            "future_delta": np.zeros(14, np.float32),
            "language_bow": np.zeros(2, np.float32),
            "visual_history": np.zeros((3, 3, 4, 4), np.uint8),
            "source_episode": {},
        }

    anchors = [anchor("a", "put", 32), anchor("a", "put", 96), anchor("b", "put", 32), anchor("c", "pick", 32)]
    candidates, audit = _candidate_sets(anchors)
    assert audit["pass"]
    assert len(candidates) == len(anchors) * 5
    for identifier in {row["candidate_set_id"] for row in candidates}:
        group = [row for row in candidates if row["candidate_set_id"] == identifier]
        assert sum(row["consistent"] for row in group) == 1
        assert sorted(row["candidate_display_order"] for row in group) == list(range(5))


def test_episode_splits_never_split_a_real_episode() -> None:
    episodes = [
        {"episode_id": "a", "task_id": "put"},
        {"episode_id": "b", "task_id": "put"},
        {"episode_id": "c", "task_id": "put"},
        {"episode_id": "d", "task_id": "pick"},
        {"episode_id": "e", "task_id": "pick"},
        {"episode_id": "f", "task_id": "pick"},
    ]
    owners = _experiment_splits(episodes)
    assert set(owners) == {row["episode_id"] for row in episodes}
    assert set(owners.values()).issuperset({"train", "test"})


def test_asu_strict_sampler_never_falls_back_from_missing_exact_pool() -> None:
    def anchor(episode: str, task: str, step: int) -> dict:
        return {
            "anchor_id": f"{episode}:{step}", "episode_id": episode, "task_id": task, "split": "train", "anchor": step,
            "history_observation": np.zeros((16, 49), np.float32), "history_robot": np.zeros((16, 7), np.float32),
            "candidate_action": np.full((8, 7), step, np.float32), "future_relation_delta": np.zeros(36, np.float32),
            "language_bow": np.zeros(2, np.float32), "visual_history": np.zeros((3, 3, 4, 4), np.uint8),
        }

    # One episode cannot supply an exact different-episode same-task negative.
    rows, audit = strict_asu_candidate_rows([anchor("a", "pick_cube", 16), anchor("a", "pick_cube", 48), anchor("b", "push_cube", 16), anchor("b", "push_cube", 48)])
    assert not rows
    assert audit["skipped_anchors_by_missing_exact_pool"]

    anchors = []
    for task in ("pick_cube", "push_cube"):
        for episode in (f"{task}_1", f"{task}_2"):
            anchors.extend((anchor(episode, task, 16), anchor(episode, task, 48)))
    rows, audit = strict_asu_candidate_rows(anchors)
    assert audit["semantic_checks_pass"]
    for row in rows:
        if row["candidate_kind"] == "same_task_other_episode":
            assert row["candidate_source_episode_id"] != row["episode_id"]
            assert row["candidate_source_task_id"] == row["task_id"]


def test_asu_ranking_metrics_are_tie_safe_and_auroc_is_global() -> None:
    rows = [{"candidate_set_id": "set", "split": "test", "consistent": index == 0} for index in range(5)]
    metric = strict_asu_rank_metrics(rows, np.zeros(5), "test")
    assert metric["top1_success"] == pytest.approx(0.2)
    assert metric["top3_recall"] == pytest.approx(0.6)
    assert metric["ndcg"] == pytest.approx(np.mean(1.0 / np.log2(np.arange(2, 7))))
    assert metric["auroc"] == pytest.approx(0.5)
