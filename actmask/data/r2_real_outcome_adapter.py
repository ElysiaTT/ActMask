"""R2 BotFails adapter: audited real-robot outcome/failure trajectories.

The R2 series must not quietly turn a dataset directory name into a target or
an input feature.  This module therefore consumes the release's per-step CSV
labels, pairs them with the identically indexed LeRobot Parquet trajectory,
and preserves the resulting provenance in a compact normalized manifest.

Only a deliberately bounded 100-episode numeric subset is acquired: five
author-provided normal demonstrations and five trajectories with an official
non-zero failure label for each of the ten released tasks.  RGB videos remain
available upstream but are not downloaded before the numeric alignment gate;
the absence is explicit in every normalized row rather than silently faked.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import quote

import numpy as np

from actmask.data.real_robot_dataset_adapter import (
    append_run_log,
    composite_file_hash,
    forbidden_input_audit,
    sha256_file,
    write_json,
    write_jsonl,
)


ROOT = Path(__file__).resolve().parents[2]
R2_ROOT = ROOT / "outputs" / "actmask" / "r2_series_real_outcome_search"
BOTFAILS_DATASET = "BotFails"
BOTFAILS_REPO = "kantine/BotFails"
BOTFAILS_REVISION = "main"
BOTFAILS_API = f"https://huggingface.co/api/datasets/{BOTFAILS_REPO}/tree/{BOTFAILS_REVISION}"
BOTFAILS_RESOLVE = f"https://huggingface.co/datasets/{BOTFAILS_REPO}/resolve/{BOTFAILS_REVISION}"
NORMAL_PER_TASK = 5
FAILURE_PER_TASK = 5
# The formulation uses a 32-step history.  Requiring an official onset after
# that history prevents the target event from already residing in the observed
# state history of every selected failure episode.
MIN_FAILURE_ONSET_STEP = 64
EXPECTED_TASKS = (
    "domotic_dishTidyUp",
    "domotic_groceriesSorting",
    "domotic_makingCoffee",
    "domotic_pouringCoffee",
    "domotic_setTheTable",
    "domotic_vegetablesAndFruitsSorting",
    "industrial_robothon_buttons",
    "industrial_robothon_hatchAndProbe",
    "industrial_screws_sorting",
    "industrial_soldering",
)
REQUIRED_PARQUET_COLUMNS = (
    "action",
    "observation.state",
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
)
MODEL_FEATURE_FIELDS = (
    "history_robot_state",
    "candidate_action_chunk",
    "optional_history_rgb_at_or_before_anchor",
)


class R2FormatError(ValueError):
    """A BotFails file violates a requirement required for a fair R2 task."""


def _curl_json(url: str) -> Any:
    if shutil.which("curl") is None:
        raise RuntimeError("curl is required for verified HTTPS access")
    result = subprocess.run(
        ["curl", "--fail", "--location", "--retry", "4", "--retry-all-errors", "--retry-delay", "1", url],
        check=False,
        capture_output=True,
    )
    if result.returncode:
        raise RuntimeError(f"curl metadata request failed: {result.stderr.decode('utf-8', errors='replace').strip()}")
    return json.loads(result.stdout)


def r2_append_log(path: str | Path, *, event: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Append to R2's hash chain, preserving its one pre-chain survey record.

    The initial survey was intentionally written before the common R-Series
    logger was adopted.  On first mutable event it is moved, not discarded,
    and a fresh, internally hash-chained R2 log begins with a migration event.
    """

    path = Path(path)
    if path.is_file() and path.stat().st_size:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        if rows and ("seq" not in rows[-1] or "event_sha256" not in rows[-1]):
            legacy = path.with_name("run_log_pre_hash_migration.jsonl")
            if legacy.exists():
                legacy = path.with_name(f"run_log_pre_hash_migration_{hashlib.sha256(path.read_bytes()).hexdigest()[:12]}.jsonl")
            path.replace(legacy)
            append_run_log(path, event="r2_run_log_migrated", payload={"legacy_log": legacy.name, "legacy_events": len(rows)})
    return append_run_log(path, event=event, payload=payload)


def _remote_url(remote_path: str) -> str:
    return f"{BOTFAILS_RESOLVE}/{quote(remote_path, safe='/')}?download=true"


def _download_file(remote_path: str, destination: Path) -> dict[str, Any]:
    """Download an individual public file with retries and an atomic rename."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size > 0:
        return {"remote_path": remote_path, "local_path": str(destination), "bytes": destination.stat().st_size, "status": "reused"}
    partial = destination.with_name(destination.name + ".part")
    command = [
        "curl", "--fail", "--location", "--retry", "6", "--retry-all-errors", "--retry-delay", "1",
        "--connect-timeout", "30", "--speed-limit", "1", "--speed-time", "120", "--output", str(partial), _remote_url(remote_path),
    ]
    result = subprocess.run(command, check=False, capture_output=True)
    if result.returncode or not partial.is_file() or partial.stat().st_size == 0:
        message = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"BotFails download failed for {remote_path}: {message}")
    partial.replace(destination)
    return {"remote_path": remote_path, "local_path": str(destination), "bytes": destination.stat().st_size, "status": "downloaded"}


def _label_remote_paths() -> list[str]:
    rows = _curl_json(f"{BOTFAILS_API}/{quote('BotFails/labels', safe='')}?recursive=true&expand=false")
    paths = sorted(str(row.get("path", "")) for row in rows if str(row.get("path", "")).endswith("_labels.csv"))
    if len(paths) != 144:
        raise R2FormatError(f"expected 144 BotFails label CSVs, found {len(paths)}")
    return paths


def acquire_labels(source_root: str | Path) -> list[Path]:
    """Acquire only the 144 small official label files before any trajectories."""

    source_root = Path(source_root)
    result: list[Path] = []
    for remote in _label_remote_paths():
        local = source_root / "labels" / Path(remote).relative_to("BotFails/labels")
        _download_file(remote, local)
        result.append(local)
    return result


def _read_label(path: Path) -> np.ndarray:
    try:
        value = np.loadtxt(path, dtype=np.int8)
    except Exception as exc:  # numpy reports a useful parse error
        raise R2FormatError(f"cannot parse label CSV {path}: {exc}") from exc
    value = np.atleast_1d(value)
    if value.ndim != 1 or not len(value) or not np.isin(value, (0, 1)).all():
        raise R2FormatError(f"label CSV must be a nonempty binary sequence: {path}")
    return value


def label_inventory(source_root: str | Path) -> dict[str, list[dict[str, int | None]]]:
    """Return per-task official labels without inferring labels from paths."""

    source_root = Path(source_root)
    root = source_root / "labels"
    inventory: dict[str, list[dict[str, int | None]]] = {}
    for directory in sorted(root.glob("*_anomaly")):
        rows: list[dict[str, int | None]] = []
        for path in sorted(directory.glob("episode_*_labels.csv")):
            label = _read_label(path)
            episode = int(path.name.split("_")[1])
            positives = np.flatnonzero(label)
            rows.append({
                "episode_index": episode,
                "steps": int(len(label)),
                "positive_steps": int(label.sum()),
                "first_positive_step": int(positives[0]) if len(positives) else None,
            })
        inventory[directory.name] = rows
    expected = {f"{task}_anomaly" for task in EXPECTED_TASKS}
    if set(inventory) != expected:
        raise R2FormatError(f"unexpected label task directories: {sorted(inventory)}")
    return inventory


def select_subset(source_root: str | Path, *, normal_per_task: int = NORMAL_PER_TASK, failure_per_task: int = FAILURE_PER_TASK, minimum_failure_onset_step: int = MIN_FAILURE_ONSET_STEP) -> list[dict[str, Any]]:
    """Select a balanced, deterministic subset using only official CSV labels.

    A selected failure episode has at least one non-zero CSV label.  The
    source path is never included in the normalized model feature set.
    """

    if normal_per_task < 1 or failure_per_task < 1:
        raise ValueError("per-task sample counts must be positive")
    inventory = label_inventory(source_root)
    rows: list[dict[str, Any]] = []
    for task in EXPECTED_TASKS:
        anomaly_name = f"{task}_anomaly"
        positives = [
            row for row in inventory[anomaly_name]
            if int(row["positive_steps"]) > 0 and int(row["first_positive_step"] or -1) >= minimum_failure_onset_step
        ]
        if len(positives) < failure_per_task:
            raise R2FormatError(f"{task} has only {len(positives)} labeled failure episodes with onset after step {minimum_failure_onset_step}")
        for episode in range(normal_per_task):
            rows.append({"task_name": task, "episode_index": episode, "outcome": "success", "label_rel": None})
        for info in positives[:failure_per_task]:
            rows.append({
                "task_name": task,
                "episode_index": int(info["episode_index"]),
                "outcome": "failure",
                "label_rel": f"labels/{anomaly_name}/episode_{int(info['episode_index']):06d}_labels.csv",
                "first_positive_step": info["first_positive_step"],
                "positive_steps": info["positive_steps"],
            })
    if len(rows) != len(EXPECTED_TASKS) * (normal_per_task + failure_per_task):
        raise R2FormatError("subset cardinality mismatch")
    return rows


def _normal_remote(task_name: str, episode_index: int) -> str:
    return f"BotFails/normal_train/{task_name}_expert/data/chunk-000/episode_{episode_index:06d}.parquet"


def _failure_remote(task_name: str, episode_index: int) -> str:
    return f"BotFails/test/{task_name}_anomaly/data/chunk-000/episode_{episode_index:06d}.parquet"


def _normal_meta_remote(task_name: str) -> str:
    return f"BotFails/normal_train/{task_name}_expert/meta/episodes.jsonl"


def acquire_subset(source_root: str | Path, *, log_path: str | Path | None = None) -> dict[str, Any]:
    """Download exactly the numeric files needed for the bounded 100-episode run."""

    source_root = Path(source_root)
    acquire_labels(source_root)
    selected = select_subset(source_root)
    downloads: list[dict[str, Any]] = []
    for task in EXPECTED_TASKS:
        remote = _normal_meta_remote(task)
        local = source_root / "raw" / Path(remote).relative_to("BotFails")
        downloads.append(_download_file(remote, local))
    for row in selected:
        remote = _normal_remote(str(row["task_name"]), int(row["episode_index"])) if row["outcome"] == "success" else _failure_remote(str(row["task_name"]), int(row["episode_index"]))
        local = source_root / "raw" / Path(remote).relative_to("BotFails")
        downloads.append(_download_file(remote, local))
    manifest = {
        "schema": "r2-botfails-source-subset-v1",
        "dataset": BOTFAILS_DATASET,
        "repository": BOTFAILS_REPO,
        "revision": BOTFAILS_REVISION,
        "selected_episodes": selected,
        "downloads": downloads,
        "downloaded_bytes": int(sum(int(item["bytes"]) for item in downloads)),
        "unacquired_media": "Official RGB videos are available upstream but intentionally excluded until numeric label/alignment validation passes.",
    }
    write_json(source_root / "source_download_manifest.json", manifest)
    if log_path is not None:
        r2_append_log(log_path, event="r2_botfails_bounded_source_acquired", payload={"episodes": len(selected), "bytes": manifest["downloaded_bytes"]})
    return manifest


def _parquet_arrays(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise RuntimeError("pyarrow is required in the ActMask environment to read BotFails Parquet files") from exc
    table = pq.read_table(path)
    missing = [name for name in REQUIRED_PARQUET_COLUMNS if name not in table.column_names]
    if missing:
        raise R2FormatError(f"{path} misses required columns: {missing}")
    action = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    timestamp = np.asarray(table["timestamp"].to_numpy(), dtype=np.float64)
    if action.ndim != 2 or state.ndim != 2 or action.shape[1] != 6 or state.shape[1] != 6:
        raise R2FormatError(f"{path} expected 6-D action/state, found action={action.shape}, state={state.shape}")
    if not (len(action) == len(state) == len(timestamp)) or len(action) < 4:
        raise R2FormatError(f"{path} action/state/timestamp lengths are not aligned")
    if not np.isfinite(action).all() or not np.isfinite(state).all() or not np.isfinite(timestamp).all():
        raise R2FormatError(f"{path} contains non-finite values")
    if np.any(np.diff(timestamp) <= 0):
        raise R2FormatError(f"{path} timestamps are not strictly increasing")
    metadata = {
        "rows": int(len(action)),
        "action_dim": int(action.shape[1]),
        "state_dim": int(state.shape[1]),
        "timestamp_start": float(timestamp[0]),
        "timestamp_end": float(timestamp[-1]),
        "sample_period_median": float(np.median(np.diff(timestamp))),
        "parquet_columns": list(table.column_names),
    }
    return state, action, timestamp.astype(np.float32), metadata


def _normal_language(source_root: Path, task_name: str, episode_index: int) -> str:
    path = source_root / "raw" / Path(_normal_meta_remote(task_name)).relative_to("BotFails")
    if not path.is_file():
        raise R2FormatError(f"normal metadata missing: {path}")
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if int(row.get("episode_index", -1)) == episode_index:
            tasks = row.get("tasks", [])
            if isinstance(tasks, list) and len(tasks) == 1 and str(tasks[0]).strip():
                return str(tasks[0]).strip()
    raise R2FormatError(f"no neutral normal-task language for {task_name} episode {episode_index}")


def _hard_task_split(task_name: str) -> str:
    """Deterministically hold out whole tasks without inspecting outcomes."""

    order = sorted(EXPECTED_TASKS, key=lambda task: hashlib.sha256(f"r2-hard:{task}".encode()).hexdigest())
    return "train" if task_name in order[:6] else "validation" if task_name in order[6:8] else "test"


def _within_task_splits(selected: Iterable[Mapping[str, Any]]) -> dict[tuple[str, str, int], str]:
    """Allocate exactly 3/1/1 samples per task/outcome, never by time step."""

    groups: dict[tuple[str, str], list[int]] = {}
    for row in selected:
        groups.setdefault((str(row["task_name"]), str(row["outcome"])), []).append(int(row["episode_index"]))
    result: dict[tuple[str, str, int], str] = {}
    for (task, outcome), indices in groups.items():
        if len(indices) != 5:
            raise R2FormatError(f"expected five selected episodes for {task}/{outcome}, got {len(indices)}")
        ordered = sorted(indices, key=lambda index: hashlib.sha256(f"r2-within:{task}:{outcome}:{index}".encode()).hexdigest())
        for rank, index in enumerate(ordered):
            result[(task, outcome, index)] = "train" if rank < 3 else "validation" if rank == 3 else "test"
    return result


def _save_array(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, value=value)


def _validate_row(row: Mapping[str, Any], processed_root: Path) -> dict[str, Any]:
    paths = {name: processed_root / str(row[name]) for name in ("observation_path", "robot_state_path", "action_path", "timestamp_path", "failure_label_path")}
    if not all(path.is_file() for path in paths.values()):
        raise R2FormatError(f"normalized arrays missing for {row['episode_id']}")
    observation, robot, action, timestamp, labels = (np.load(paths[name])["value"] for name in paths)
    if not (observation.ndim == robot.ndim == action.ndim == 2 and timestamp.ndim == labels.ndim == 1):
        raise R2FormatError(f"normalized ranks invalid for {row['episode_id']}")
    if not (len(observation) == len(robot) == len(action) == len(timestamp) == len(labels)):
        raise R2FormatError(f"normalized lengths invalid for {row['episode_id']}")
    if observation.shape[1] != 6 or robot.shape[1] != 6 or action.shape[1] != 6:
        raise R2FormatError(f"normalized dimensions invalid for {row['episode_id']}")
    if np.any(np.diff(timestamp) <= 0) or not np.isin(labels, (0, 1)).all():
        raise R2FormatError(f"normalized time/label values invalid for {row['episode_id']}")
    if row["success_or_failure"] == "success" and int(labels.sum()) != 0:
        raise R2FormatError(f"normal episode has a failure label: {row['episode_id']}")
    if row["success_or_failure"] == "failure" and int(labels.sum()) == 0:
        raise R2FormatError(f"failure episode lacks official positive label: {row['episode_id']}")
    return {"episode_id": str(row["episode_id"]), "steps": int(len(action)), "positive_steps": int(labels.sum())}


def prepare_subset(source_root: str | Path, processed_root: str | Path, *, log_path: str | Path | None = None) -> dict[str, Any]:
    """Normalize the bounded BotFails numeric subset and write alignment audits."""

    source_root, processed_root = Path(source_root), Path(processed_root)
    selected = select_subset(source_root)
    within_splits = _within_task_splits(selected)
    normalized: list[dict[str, Any]] = []
    for spec in selected:
        task, index, outcome = str(spec["task_name"]), int(spec["episode_index"]), str(spec["outcome"])
        remote = _normal_remote(task, index) if outcome == "success" else _failure_remote(task, index)
        parquet_path = source_root / "raw" / Path(remote).relative_to("BotFails")
        if not parquet_path.is_file():
            raise R2FormatError(f"selected Parquet is missing: {parquet_path}")
        state, action, timestamp, source_info = _parquet_arrays(parquet_path)
        labels = np.zeros(len(action), dtype=np.int8) if outcome == "success" else _read_label(source_root / str(spec["label_rel"]))
        if len(labels) != len(action):
            raise R2FormatError(f"official label/trajectory length mismatch for {task}:{index}: {len(labels)} != {len(action)}")
        if outcome == "failure" and not labels.any():
            raise R2FormatError(f"selected failure has no positive official label: {task}:{index}")
        episode_id = f"botfails_{task}_{outcome}_{index:06d}"
        observation_rel = f"arrays/{episode_id}_observation.npz"
        robot_rel = f"arrays/{episode_id}_robot_state.npz"
        action_rel = f"arrays/{episode_id}_action.npz"
        time_rel = f"arrays/{episode_id}_timestamp.npz"
        label_rel = f"arrays/{episode_id}_failure_label.npz"
        _save_array(processed_root / observation_rel, state)
        _save_array(processed_root / robot_rel, state)
        _save_array(processed_root / action_rel, action)
        _save_array(processed_root / time_rel, timestamp)
        _save_array(processed_root / label_rel, labels)
        within_split = within_splits[(task, outcome, index)]
        hard_split = _hard_task_split(task)
        # Always use the normal demonstration's task text, including for failures;
        # anomaly-description metadata is never exposed as language input.
        language = _normal_language(source_root, task, min(index, NORMAL_PER_TASK - 1))
        normalized.append({
            "schema": "r2-real-robot-outcome-episode-v1",
            "episode_id": episode_id,
            "dataset_name": BOTFAILS_DATASET,
            "robot": "LeRobot teleoperation platform (release metadata; exact arm model not asserted by this adapter)",
            "task_name": task,
            "task_id": task,
            "language_instruction": language,
            "success_or_failure": outcome,
            "outcome_source": "official BotFails per-step binary CSV label" if outcome == "failure" else "BotFails normal_train author-provided expert demonstration",
            "observation_path": observation_rel,
            "robot_state_path": robot_rel,
            "action_path": action_rel,
            "timestamp_path": time_rel,
            "failure_label_path": label_rel,
            "rgb_paths": [],
            "rgb_availability": "official videos available remotely but intentionally not downloaded before numeric validation",
            "object_state_or_contact_path": None,
            "source_parquet": str(parquet_path.relative_to(source_root)),
            "source_label": None if outcome == "success" else str(spec["label_rel"]),
            "split": within_split,
            "split_metadata": {"within_task_stratified": within_split, "hard_task_heldout": hard_split, "source_split_not_model_input": "normal_train" if outcome == "success" else "test"},
            "alignment": source_info,
        })
    rows = sorted(normalized, key=lambda row: str(row["episode_id"]))
    write_jsonl(processed_root / "episodes.jsonl", rows)
    audits = [_validate_row(row, processed_root) for row in rows]
    outcome_counts = Counter(str(row["success_or_failure"]) for row in rows)
    task_counts = Counter(str(row["task_name"]) for row in rows)
    hard_counts = Counter(str(row["split_metadata"]["hard_task_heldout"]) for row in rows)
    audit = {
        "schema": "r2-botfails-adapter-audit-v1",
        "pass": len(rows) == 100 and set(outcome_counts) == {"success", "failure"} and all(value == 10 for value in task_counts.values()),
        "episodes": len(rows),
        "outcomes": dict(outcome_counts),
        "task_counts": dict(sorted(task_counts.items())),
        "hard_split_episode_counts": dict(hard_counts),
        "alignment_rows": audits,
        "nonempty_failure_episodes": int(sum(item["positive_steps"] > 0 for item in audits)),
        "fair_input_audit": forbidden_input_audit(MODEL_FEATURE_FIELDS),
        "known_limitations": [
            "RGB is available in the source release but not included in this numeric-first subset.",
            "Normal and anomaly source partitions must be evaluated as an explicit source-identity shortcut audit; source metadata is not a model feature.",
            "The task formulation may claim logged failure/outcome verification, not a counterfactual physical intervention result.",
        ],
    }
    if not audit["pass"] or not audit["fair_input_audit"]["pass"]:
        raise R2FormatError(f"normalized subset audit failed: {audit}")
    write_json(processed_root / "adapter_audit.json", audit)
    source_files = [source_root / str(row["source_parquet"]) for row in rows]
    source_files.extend(source_root / str(row["source_label"]) for row in rows if row["source_label"])
    source_hash = composite_file_hash(source_root, source_files)
    manifest = {
        "schema": "r2-botfails-normalized-manifest-v1",
        "dataset": BOTFAILS_DATASET,
        "episodes_path": "episodes.jsonl",
        "adapter_audit_path": "adapter_audit.json",
        "source_hash": source_hash,
        "normalized_episode_count": len(rows),
        "normalization": "observation.state is used as both observation and robot_state; 6-D action/timestamp copied without resampling",
    }
    write_json(processed_root / "data_manifest.json", manifest)
    if log_path is not None:
        r2_append_log(log_path, event="r2_botfails_adapter_alignment_pass", payload={"episodes": len(rows), "manifest_sha256": sha256_file(processed_root / "data_manifest.json")})
    return {"audit": audit, "manifest": manifest}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=R2_ROOT / "source_subset" / "botfails")
    parser.add_argument("--processed-root", type=Path, default=R2_ROOT / "processed_subset" / "botfails")
    parser.add_argument("--acquire", action="store_true", help="download bounded labels/metadata/100 Parquet trajectories")
    parser.add_argument("--prepare", action="store_true", help="normalize locally available source files")
    args = parser.parse_args()
    if not args.acquire and not args.prepare:
        parser.error("choose --acquire, --prepare, or both")
    log = R2_ROOT / "run_log.jsonl"
    if args.acquire:
        acquire_subset(args.source_root, log_path=log)
    if args.prepare:
        result = prepare_subset(args.source_root, args.processed_root, log_path=log)
        print(json.dumps({"episodes": result["audit"]["episodes"], "pass": result["audit"]["pass"]}, sort_keys=True))


if __name__ == "__main__":
    main()
