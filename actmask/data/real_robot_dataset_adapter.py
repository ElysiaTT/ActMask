"""Shared, dependency-light contracts for real-robot dataset adapters.

The R-Series deliberately keeps the first feasibility run independent of a
large framework such as TensorFlow, LeRobot, or a pretrained encoder.  This
module is the small contract shared by public-dataset adapters and the
logged-trajectory verification experiments.  It makes the provenance and the
episode-level split explicit so that a candidate action never gets separated
from the trajectory that supplied it by accident.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
R_SERIES_ROOT = ROOT / "outputs" / "actmask" / "r_series_real_robot_verification"

NORMALIZED_SCHEMA_VERSION = "r-series-real-robot-episode-v1"
REQUIRED_EPISODE_FIELDS = (
    "episode_id",
    "task_id",
    "language_instruction",
    "robot_state_path",
    "action_path",
    "rgb_paths",
    "timestamp_path",
    "success",
    "outcome_source",
    "source_dataset",
    "split",
)
FORBIDDEN_FAIR_INPUTS = (
    "episode_id",
    "source_episode_id",
    "source_shard",
    "source_record_index",
    "candidate_id",
    "candidate_index",
    "candidate_display_order",
    "candidate_kind",
    "label",
    "success",
    "outcome",
    "future_robot_state",
    "future_rgb",
    "future_timestamp",
    "source_split",
)


def canonical_json(value: Any) -> str:
    """Stable JSON representation used for hashes and manifests."""

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path, *, chunk_bytes: int = 1024 * 1024) -> str:
    """Return a streaming SHA-256 without reading a data shard into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(block)
    return digest.hexdigest()


def composite_file_hash(root: str | Path, paths: Iterable[str | Path]) -> dict[str, Any]:
    """Hash a sorted set of files without a self-referential manifest hash."""

    root = Path(root).resolve()
    entries: list[dict[str, Any]] = []
    for candidate in sorted({Path(value).resolve() for value in paths}):
        relative = candidate.relative_to(root).as_posix()
        entries.append(
            {
                "path": relative,
                "bytes": candidate.stat().st_size,
                "sha256": sha256_file(candidate),
            }
        )
    payload = "".join(
        f"{row['path']}\0{row['sha256']}\0{row['bytes']}\n" for row in entries
    ).encode("utf-8")
    return {"entries": entries, "composite_sha256": sha256_bytes(payload)}


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(canonical_json(dict(row)) + "\n" for row in rows))


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def stable_episode_split(episode_id: str, *, seed: int = 20260725) -> str:
    """Assign an episode, never an individual window, to one stable split."""

    word = hashlib.sha256(f"{seed}:{episode_id}".encode("utf-8")).digest()[0]
    return "train" if word < 180 else "validation" if word < 218 else "test"


def validate_normalized_episode(
    row: Mapping[str, Any],
    *,
    processed_root: str | Path,
    require_rgb: bool = True,
) -> dict[str, Any]:
    """Validate one saved normalized episode and its time alignment.

    The adapter stores large arrays in ``.npz`` and only a JSONL index in the
    manifest.  This audit validates both layers, including that every RGB
    reference points to a genuine sampled step rather than a future frame.
    """

    missing = [field for field in REQUIRED_EPISODE_FIELDS if field not in row]
    if missing:
        raise ValueError(f"normalized episode misses fields: {missing}")
    root = Path(processed_root)
    state_path = root / str(row["robot_state_path"])
    action_path = root / str(row["action_path"])
    time_path = root / str(row["timestamp_path"])
    if not (state_path.is_file() and action_path.is_file() and time_path.is_file()):
        raise ValueError(f"missing numeric artifact for episode {row['episode_id']}")
    state = np.load(state_path)["value"]
    action = np.load(action_path)["value"]
    timestamp = np.load(time_path)["value"]
    if state.ndim != 2 or action.ndim != 2 or timestamp.ndim != 1:
        raise ValueError(f"invalid numeric ranks for episode {row['episode_id']}")
    if not (len(state) == len(action) == len(timestamp)) or len(state) < 4:
        raise ValueError(f"time alignment failure for episode {row['episode_id']}")
    if not np.isfinite(state).all() or not np.isfinite(action).all() or not np.isfinite(timestamp).all():
        raise ValueError(f"non-finite numeric values for episode {row['episode_id']}")
    if np.any(np.diff(timestamp) <= 0):
        raise ValueError(f"timestamps are not strictly increasing for episode {row['episode_id']}")
    rgb_rows = list(row["rgb_paths"])
    if require_rgb and not rgb_rows:
        raise ValueError(f"episode {row['episode_id']} has no sampled RGB")
    seen_steps: set[int] = set()
    for item in rgb_rows:
        if not isinstance(item, Mapping) or "step" not in item or "path" not in item:
            raise ValueError(f"invalid RGB reference for episode {row['episode_id']}")
        step = int(item["step"])
        image_path = root / str(item["path"])
        if step < 0 or step >= len(state) or not image_path.is_file():
            raise ValueError(f"bad RGB reference for episode {row['episode_id']}")
        if step in seen_steps:
            raise ValueError(f"duplicate RGB step for episode {row['episode_id']}")
        seen_steps.add(step)
    return {
        "episode_id": str(row["episode_id"]),
        "steps": int(len(state)),
        "state_dim": int(state.shape[1]),
        "action_dim": int(action.shape[1]),
        "sampled_rgb_frames": int(len(rgb_rows)),
        "split": str(row["split"]),
    }


def append_run_log(
    path: str | Path,
    *,
    event: str,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Append a hash-chained live event without silently rewriting history."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    previous_hash = ""
    sequence = 0
    if path.exists() and path.stat().st_size:
        previous = read_jsonl(path)
        sequence = int(previous[-1]["seq"])
        previous_hash = str(previous[-1]["event_sha256"])
    row: dict[str, Any] = {
        "seq": sequence + 1,
        "utc": datetime.now(timezone.utc).isoformat(),
        "event": str(event),
        "payload": dict(payload),
        "prev_event_sha256": previous_hash,
    }
    row["event_sha256"] = sha256_bytes(canonical_json(row).encode("utf-8"))
    with path.open("a", encoding="utf-8") as handle:
        handle.write(canonical_json(row) + "\n")
    return row


def forbidden_input_audit(feature_fields: Iterable[str]) -> dict[str, Any]:
    fields = tuple(str(field) for field in feature_fields)
    present = sorted(set(fields).intersection(FORBIDDEN_FAIR_INPUTS))
    return {
        "fair_input_fields": list(fields),
        "forbidden_fields": list(FORBIDDEN_FAIR_INPUTS),
        "forbidden_fields_present": present,
        "pass": not present,
    }


__all__ = [
    "FORBIDDEN_FAIR_INPUTS",
    "NORMALIZED_SCHEMA_VERSION",
    "REQUIRED_EPISODE_FIELDS",
    "R_SERIES_ROOT",
    "append_run_log",
    "canonical_json",
    "composite_file_hash",
    "forbidden_input_audit",
    "read_jsonl",
    "sha256_bytes",
    "sha256_file",
    "stable_episode_split",
    "validate_normalized_episode",
    "write_json",
    "write_jsonl",
]
