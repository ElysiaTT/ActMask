"""Strict, read-only adapter for the locally available RoboMIND 2.0 subset.

The RoboMIND snapshot in this environment is a LeRobot-style export.  Its
per-episode Parquet files contain synchronized numeric streams while RGB is
stored separately as MP4.  This module deliberately treats a directory name
as *provenance only*: it is never an outcome label or a fair model feature.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow.parquet as pq


ROBOMIND_DATASET = "RoboMIND2.0"
PARQUET_COLUMNS = (
    "action.arm_joint_position",
    "action.hand_joint_position",
    "action.end_effector",
    "observation.state.arm_joint_position",
    "observation.state.hand_joint_position",
    "observation.state.end_effector",
    "timestamp",
    "frame_index",
    "episode_index",
    "task_index",
)
FORBIDDEN_FAIR_FIELDS = {
    "episode_id",
    "task_id",
    "task_name",
    "language_instruction",  # not used in the first, language-free audit
    "source_file_paths",
    "source_path",
    "folder_name",
    "candidate_origin_episode_id",
    "candidate_origin_path",
    "future_state",
    "future_rgb",
    "future_action",
    "label",
    "success_or_failure",
    "failure_category",
}


class RoboMINDFormatError(ValueError):
    """Raised when a selected trajectory cannot satisfy the adapter contract."""


@dataclass(frozen=True)
class RoboMINDRecord:
    """One source episode plus official, non-label metadata."""

    episode_id: str
    task_id: str
    robot_embodiment: str
    language_instruction: str
    parquet_path: str
    front_video_path: str | None
    source_bytes: int
    episode_index: int
    metadata_path: str

    def json(self) -> dict[str, Any]:
        return asdict(self)


def _task_root(parquet_path: Path) -> Path:
    # .../<task>/data/chunk-000/episode_000000.parquet
    if parquet_path.parent.name != "chunk-000" or parquet_path.parent.parent.name != "data":
        raise RoboMINDFormatError(f"unexpected RoboMIND Parquet layout: {parquet_path}")
    return parquet_path.parents[2]


def _read_episode_metadata(task_root: Path) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    info_path = task_root / "meta" / "info.json"
    episodes_path = task_root / "meta" / "episodes.jsonl"
    if not info_path.is_file() or not episodes_path.is_file():
        raise RoboMINDFormatError(f"missing official LeRobot metadata below {task_root}")
    info = json.loads(info_path.read_text())
    episodes: dict[int, dict[str, Any]] = {}
    for line in episodes_path.read_text().splitlines():
        if line:
            row = json.loads(line)
            episodes[int(row["episode_index"])] = row
    return info, episodes


def _episode_number(path: Path) -> int:
    prefix = "episode_"
    if not path.stem.startswith(prefix):
        raise RoboMINDFormatError(f"unexpected episode filename: {path.name}")
    return int(path.stem[len(prefix) :])


def scan_local_robomind(source_root: str | Path) -> dict[str, Any]:
    """Return a read-only inventory of actual local RoboMIND source files."""

    root = Path(source_root)
    parquet_files = sorted(root.rglob("*.parquet"))
    hdf5_files = sorted(root.rglob("*.hdf5"))
    metadata_files = sorted(root.rglob("meta/info.json"))
    video_files = sorted(root.rglob("*.mp4"))
    task_dirs = sorted({str(_task_root(path).relative_to(root)) for path in parquet_files})
    return {
        "dataset": ROBOMIND_DATASET,
        "source_root": str(root),
        "source_root_exists": root.is_dir(),
        "readable_parquet_files": len(parquet_files),
        "readable_hdf5_files": len(hdf5_files),
        "metadata_info_files": len(metadata_files),
        "readable_rgb_video_files": len(video_files),
        "parquet_bytes": int(sum(path.stat().st_size for path in parquet_files)),
        "hdf5_bytes": int(sum(path.stat().st_size for path in hdf5_files)),
        "task_directories": task_dirs,
        "failure_data_directory_visible": any("failure" in part.lower() for path in root.rglob("*") for part in path.parts),
    }


def select_bounded_records(
    source_root: str | Path,
    *,
    max_tasks: int = 10,
    episodes_per_task: int = 10,
) -> list[RoboMINDRecord]:
    """Pick a bounded, balanced source subset without moving any source file.

    Task families are chosen by the byte cost of their smallest requested
    episode count.  Selection is therefore deterministic, contains repeated
    attempts per task, and remains below the specified first-pass budget.
    """

    root = Path(source_root)
    candidates: dict[Path, list[Path]] = defaultdict(list)
    for path in sorted(root.rglob("episode_*.parquet")):
        if path.parent.name == "chunk-000" and path.parent.parent.name == "data":
            candidates[_task_root(path)].append(path)
    if not candidates:
        raise RoboMINDFormatError(f"no LeRobot episode Parquet files below {root}")

    affordable: list[tuple[int, str, Path, list[Path]]] = []
    for task_root, paths in candidates.items():
        chosen = sorted(paths, key=lambda item: (item.stat().st_size, item.name))[:episodes_per_task]
        if len(chosen) == episodes_per_task:
            affordable.append((sum(item.stat().st_size for item in chosen), task_root.name, task_root, chosen))
    affordable.sort(key=lambda item: (item[0], item[1]))
    selected = affordable[:max_tasks]
    if len(selected) != max_tasks:
        raise RoboMINDFormatError(
            f"only {len(selected)} task families have {episodes_per_task} local episodes; need {max_tasks}"
        )

    records: list[RoboMINDRecord] = []
    for _, task_name, task_root, paths in selected:
        info, episode_rows = _read_episode_metadata(task_root)
        robot = str(info.get("robot_type", "unknown"))
        for path in sorted(paths, key=_episode_number):
            number = _episode_number(path)
            row = episode_rows.get(number)
            if row is None or not row.get("tasks"):
                raise RoboMINDFormatError(f"missing language metadata for {path}")
            front = task_root / "videos" / "chunk-000" / "observation.rgb_images.camera_front" / f"{path.stem}.mp4"
            records.append(
                RoboMINDRecord(
                    episode_id=f"{task_name}:{number:06d}",
                    task_id=task_name,
                    robot_embodiment=robot,
                    language_instruction=str(row["tasks"][0]),
                    parquet_path=str(path),
                    front_video_path=str(front) if front.is_file() else None,
                    source_bytes=int(path.stat().st_size),
                    episode_index=number,
                    metadata_path=str(task_root / "meta" / "episodes.jsonl"),
                )
            )
    return records


def _column_matrix(table: Any, name: str) -> np.ndarray:
    if name not in table.column_names:
        raise RoboMINDFormatError(f"required column missing: {name}")
    values = np.asarray(table[name].to_pylist(), dtype=np.float32)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2:
        raise RoboMINDFormatError(f"{name} must be a rank-1 or rank-2 numeric stream, got {values.shape}")
    return values


def load_numeric_episode(record: RoboMINDRecord) -> dict[str, np.ndarray]:
    """Read only fair numeric columns, never image/depth bytes or provenance."""

    table = pq.read_table(record.parquet_path, columns=list(PARQUET_COLUMNS))
    action = np.concatenate(
        [_column_matrix(table, name) for name in PARQUET_COLUMNS[:3]], axis=1
    ).astype(np.float32, copy=False)
    state = np.concatenate(
        [_column_matrix(table, name) for name in PARQUET_COLUMNS[3:6]], axis=1
    ).astype(np.float32, copy=False)
    timestamp = _column_matrix(table, "timestamp").reshape(-1).astype(np.float64, copy=False)
    frame_index = _column_matrix(table, "frame_index").reshape(-1).astype(np.int64, copy=False)
    episode_index = _column_matrix(table, "episode_index").reshape(-1).astype(np.int64, copy=False)
    task_index = _column_matrix(table, "task_index").reshape(-1).astype(np.int64, copy=False)
    return {
        "state": state,
        "action": action,
        "timestamp": timestamp,
        "frame_index": frame_index,
        "episode_index": episode_index,
        "task_index": task_index,
    }


def validate_numeric_episode(record: RoboMINDRecord, numeric: dict[str, np.ndarray]) -> dict[str, Any]:
    """Fail closed on bad alignment instead of repairing a source trajectory."""

    state, action = numeric["state"], numeric["action"]
    timestamp, frame = numeric["timestamp"], numeric["frame_index"]
    lengths = {"state": len(state), "action": len(action), "timestamp": len(timestamp), "frame_index": len(frame)}
    aligned = len(set(lengths.values())) == 1 and len(state) >= 16
    finite = bool(np.isfinite(state).all() and np.isfinite(action).all() and np.isfinite(timestamp).all())
    timestamp_strict = bool(len(timestamp) > 1 and np.all(np.diff(timestamp) > 0))
    frame_strict = bool(len(frame) > 1 and np.all(np.diff(frame) > 0))
    episode_constant = bool(len(set(numeric["episode_index"].tolist())) == 1)
    task_constant = bool(len(set(numeric["task_index"].tolist())) == 1)
    valid = aligned and finite and timestamp_strict and frame_strict and episode_constant and task_constant
    return {
        "episode_id": record.episode_id,
        "steps": int(len(state)),
        "state_dim": int(state.shape[1]),
        "action_dim": int(action.shape[1]),
        "lengths": lengths,
        "finite": finite,
        "timestamp_strictly_increasing": timestamp_strict,
        "frame_index_strictly_increasing": frame_strict,
        "episode_index_constant": episode_constant,
        "task_index_constant": task_constant,
        "front_rgb_available": bool(record.front_video_path),
        "pass": valid,
    }


def fair_input_audit(fields: Iterable[str]) -> dict[str, Any]:
    fields = [str(item) for item in fields]
    forbidden = sorted(set(fields).intersection(FORBIDDEN_FAIR_FIELDS))
    return {
        "fair_input_fields": fields,
        "forbidden_fields": sorted(FORBIDDEN_FAIR_FIELDS),
        "forbidden_fields_present": forbidden,
        "pass": not forbidden,
    }


__all__ = [
    "FORBIDDEN_FAIR_FIELDS",
    "PARQUET_COLUMNS",
    "ROBOMIND_DATASET",
    "RoboMINDFormatError",
    "RoboMINDRecord",
    "fair_input_audit",
    "load_numeric_episode",
    "scan_local_robomind",
    "select_bounded_records",
    "validate_numeric_episode",
]
