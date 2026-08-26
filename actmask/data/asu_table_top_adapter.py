"""Dependency-light adapter for the public Open X ASU TableTop RLDS release.

This is the R-Series fallback after the tiny DROID debugging release proved
insufficiently supported for a repeated-task logged-consistency evaluation.
The official ASU TableTop release contains 110 real UR5 tabletop trajectories
and, unusually for a compact public release, dataset-provided 6-D poses for
the end effector and six tabletop objects.  Those poses are observable inputs
only up to the history boundary; their future values are target-only.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np
from PIL import Image

from actmask.data.droid_adapter import (
    DroidFormatError,
    _feature_bytes,
    _feature_float,
    _feature_int,
    _get_json,
    _metadata_text,
    download_remote_object,
    iter_tfrecord,
    parse_tf_example,
)
from actmask.data.real_robot_dataset_adapter import (
    NORMALIZED_SCHEMA_VERSION,
    R_SERIES_ROOT,
    append_run_log,
    composite_file_hash,
    sha256_file,
    stable_episode_split,
    validate_normalized_episode,
    write_json,
    write_jsonl,
)


ASU_DATASET = "Open X-Embodiment ASU TableTop"
ASU_BUCKET = "gdm-robotics-open-x-embodiment"
ASU_PREFIX = "asu_table_top_converted_externally_to_rlds/0.1.0"
ASU_STORAGE_API = f"https://storage.googleapis.com/storage/v1/b/{ASU_BUCKET}/o"
ASU_DOWNLOAD_API = f"https://storage.googleapis.com/download/storage/v1/b/{ASU_BUCKET}/o"
OBJECT_NAMES = ("bottle", "bread", "coke", "cube", "milk", "pepsi")
RELATION_NAMES = ("EE",) + OBJECT_NAMES
REQUIRED_ASU_FEATURES = (
    "steps/action",
    "steps/observation/state",
    "steps/observation/image",
    "steps/language_instruction",
    *(f"steps/ground_truth_states/{name}" for name in RELATION_NAMES),
)


class AsuFormatError(ValueError):
    """Raised when an ASU RLDS record violates the explicit adapter contract."""


def list_asu_remote_objects() -> list[dict[str, Any]]:
    """List the fixed, public ASU TableTop release with expected byte sizes."""

    payload = _get_json(f"{ASU_STORAGE_API}?prefix={quote(ASU_PREFIX + '/', safe='')}&maxResults=1000")
    result: list[dict[str, Any]] = []
    for row in payload.get("items", []):
        name = str(row.get("name", ""))
        if name.endswith(("dataset_info.json", "features.json")) or ".tfrecord-" in name:
            result.append({"name": name, "bytes": int(row["size"]), "md5_base64": row.get("md5Hash")})
    if len([row for row in result if ".tfrecord-" in str(row["name"])]) != 8:
        raise RuntimeError("ASU TableTop public listing did not contain its expected eight train shards")
    return sorted(result, key=lambda item: str(item["name"]))


def _media_url(remote_name: str) -> str:
    return f"{ASU_DOWNLOAD_API}/{quote(remote_name, safe='')}?alt=media"


def download_asu_table_top(source_root: str | Path, *, log_path: str | Path | None = None) -> dict[str, Any]:
    """Resumably acquire the complete documented 0.773-GB public subset."""

    source_root = Path(source_root)
    source_root.mkdir(parents=True, exist_ok=True)
    inventory = list_asu_remote_objects()
    downloaded: list[dict[str, Any]] = []
    for remote in inventory:
        relative = Path(str(remote["name"])).relative_to(ASU_PREFIX)
        local_remote = dict(remote)
        # Reuse the hardened curl downloader while substituting this public object URL.
        # It accepts a remote name; monkey-free direct implementation keeps provenance explicit.
        destination = source_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        expected = int(remote["bytes"])
        partial = destination.with_name(destination.name + ".part")
        if not destination.is_file() and partial.is_file() and partial.stat().st_size == expected:
            partial.replace(destination)
        if not (destination.is_file() and destination.stat().st_size == expected):
            import shutil
            import subprocess

            if shutil.which("curl") is None:
                raise RuntimeError("ASU adapter requires host curl for verified HTTPS transport")
            command = [
                "curl", "--fail", "--location", "--retry", "4", "--retry-delay", "2",
                "--connect-timeout", "30", "--speed-limit", "1", "--speed-time", "120",
            ]
            # GCS accepts ranges, but ``curl --continue-at -`` against a new
            # zero-byte output can complete without transferring a body on
            # some curl/GCS combinations.  Resume only a genuinely partial
            # object; otherwise start a normal, size-checked transfer.
            if partial.exists() and partial.stat().st_size > 0:
                command.extend(("--continue-at", "-"))
            elif partial.exists():
                partial.unlink()
            command.extend(("--output", str(partial), _media_url(str(remote["name"]))))
            completed = subprocess.run(command, check=False, capture_output=True)
            if completed.returncode:
                raise RuntimeError(f"ASU curl download failed for {remote['name']}: {completed.stderr.decode('utf-8', errors='replace').strip()}")
            if not partial.is_file() or partial.stat().st_size != expected:
                size = partial.stat().st_size if partial.exists() else 0
                raise RuntimeError(f"ASU object has wrong final size: {remote['name']} ({size} != {expected})")
            partial.replace(destination)
        digest = base64.b64encode(hashlib.md5(destination.read_bytes()).digest()).decode("ascii")
        if remote.get("md5_base64") and digest != remote["md5_base64"]:
            raise RuntimeError(f"ASU object MD5 mismatch for {remote['name']}")
        downloaded.append({"path": str(destination), "status": "present", "bytes": expected, "md5_base64": digest})
    manifest = {
        "schema": "r-series-asu-public-download-v1",
        "dataset": ASU_DATASET,
        "remote_prefix": f"gs://{ASU_BUCKET}/{ASU_PREFIX}",
        "remote_inventory": inventory,
        "downloaded": downloaded,
        "downloaded_bytes": int(sum(row["bytes"] for row in downloaded)),
    }
    write_json(source_root / "download_manifest.json", manifest)
    if log_path is not None:
        append_run_log(log_path, event="asu_tabletop_backup_download", payload={"bytes": manifest["downloaded_bytes"], "shards": 8})
    return manifest


def _scalar_text(features: Mapping[str, tuple[str, np.ndarray | list[bytes]]], key: str, steps: int) -> str:
    values = _feature_bytes(features, key)
    decoded = [item.decode("utf-8", errors="replace").strip() for item in values]
    nonempty = next((item for item in decoded if item), "")
    if not nonempty:
        raise AsuFormatError(f"ASU feature {key!r} has no usable text")
    if len(values) not in (1, steps):
        raise AsuFormatError(f"ASU feature {key!r} has unexpected cardinality {len(values)}")
    return nonempty


def _task_id(action_instruction: str, goal_object: str) -> str:
    """Dataset-provided action/object labels, never a path-derived surrogate."""

    action = "_".join(action_instruction.lower().split()) or "unknown_action"
    object_name = "_".join(goal_object.lower().split()) or "unknown_object"
    return f"asu_{action}__{object_name}"


def decode_asu_episode(record: bytes, *, source_shard: str, source_record_index: int) -> dict[str, Any]:
    features = parse_tf_example(record)
    missing = [key for key in REQUIRED_ASU_FEATURES if key not in features]
    if missing:
        raise AsuFormatError(f"ASU record misses required features: {missing}")
    action = _feature_float(features, "steps/action", 7)
    robot = _feature_float(features, "steps/observation/state", 7)
    n = len(action)
    if len(robot) != n:
        raise AsuFormatError("ASU action/robot state lengths disagree")
    relation = np.concatenate([_feature_float(features, f"steps/ground_truth_states/{name}", 6) for name in RELATION_NAMES], axis=1)
    if len(relation) != n:
        raise AsuFormatError("ASU relation-state/action lengths disagree")
    images = _feature_bytes(features, "steps/observation/image")
    if len(images) != n:
        raise AsuFormatError("ASU RGB/action lengths disagree")
    language = _scalar_text(features, "steps/language_instruction", n)
    action_instruction = _scalar_text(features, "steps/action_inst", n) if "steps/action_inst" in features else language
    goal_object = _scalar_text(features, "steps/goal_object", n) if "steps/goal_object" in features else "unknown_object"
    file_path = _metadata_text(features, "episode_metadata/file_path")
    terminal = _feature_int(features, "steps/is_terminal") if "steps/is_terminal" in features else np.zeros(n, np.int64)
    reward = _feature_float(features, "steps/reward", 1).reshape(-1) if "steps/reward" in features else np.zeros(n, np.float32)
    if len(terminal) not in (1, n) or len(reward) not in (1, n):
        raise AsuFormatError("ASU terminal/reward cardinality is invalid")
    identifier = hashlib.sha256(f"{source_shard}:{source_record_index}:{file_path}".encode()).hexdigest()[:24]
    return {
        "episode_id": f"asu_{identifier}",
        "task_id": _task_id(action_instruction, goal_object),
        "task_id_source": "dataset-provided action_inst and goal_object fields",
        "language_instruction": language,
        "robot_state": robot,
        "relation_state": relation,
        "observation_state": np.concatenate((robot, relation), axis=1),
        "action": action,
        "timestamp": np.arange(n, dtype=np.float32),
        "timestamp_source": "ordered_rlds_step_index_no_declared_frame_rate",
        "rgb_png": images,
        "success": bool((terminal[-1] if len(terminal) else 0) or (reward[-1] if len(reward) else 0.0) > 0.5),
        "outcome_source": "ASU RLDS terminal/reward demonstration marker; no failure class inferred",
        "source_shard": source_shard,
        "source_record_index": int(source_record_index),
        "source_file_path": file_path,
    }


def _save_value(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, value=value)


def _save_thumbnail(image_bytes: bytes, path: Path, *, max_width: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(io.BytesIO(image_bytes)) as image:
        image = image.convert("RGB")
        ratio = min(1.0, float(max_width) / image.width)
        image.resize((max(1, round(image.width * ratio)), max(1, round(image.height * ratio))), Image.Resampling.BILINEAR).save(path, format="JPEG", quality=80, optimize=True)


def _sample_steps(length: int, stride: int) -> tuple[int, ...]:
    result = list(range(0, length, stride))
    if result[-1] != length - 1:
        result.append(length - 1)
    return tuple(result)


def prepare_processed_subset(source_root: str | Path, processed_root: str | Path, *, image_stride: int = 8, thumbnail_width: int = 96, max_new_episodes: int | None = None, log_path: str | Path | None = None) -> dict[str, Any]:
    """Convert ASU episodes with an interruption-safe per-episode checkpoint.

    A constrained worker can call this repeatedly with ``max_new_episodes``.
    The checkpoint is written only after an episode's arrays and RGB samples
    are complete, so an interrupted invocation never creates a partial row.
    """

    source_root, processed_root = Path(source_root), Path(processed_root)
    # A fully downloaded recovery ``.part`` can remain if an outer job was
    # interrupted after it had already promoted an identical final object.
    # It is never a source shard and is deliberately excluded from the source
    # manifest and parser.
    shards = sorted(path for path in source_root.glob("*.tfrecord-*") if not path.name.endswith(".part"))
    if len(shards) != 8:
        raise FileNotFoundError(f"expected eight ASU shards below {source_root}, found {len(shards)}")
    if max_new_episodes is not None and max_new_episodes <= 0:
        raise ValueError("max_new_episodes must be positive when supplied")
    processed_root.mkdir(parents=True, exist_ok=True)
    final_index = processed_root / "episodes.jsonl"
    final_manifest = processed_root / "data_manifest.json"
    final_audit = processed_root / "adapter_audit.json"
    if final_index.is_file() and final_manifest.is_file() and final_audit.is_file():
        prior_rows = [json.loads(line) for line in final_index.read_text().splitlines() if line]
        prior_manifest = json.loads(final_manifest.read_text())
        prior_audit = json.loads(final_audit.read_text())
        if len(prior_rows) == 110 and prior_manifest.get("episodes") == 110 and prior_audit.get("all_pass") is True:
            return {"complete": True, "manifest": prior_manifest, "audit": prior_audit, "rows": prior_rows, "status": "already_complete"}
    progress_path = processed_root / "episodes_in_progress.jsonl"
    if progress_path.is_file():
        rows = [json.loads(line) for line in progress_path.read_text().splitlines() if line]
    else:
        rows = []
    completed = {str(row["episode_id"]) for row in rows}
    if len(completed) != len(rows):
        raise RuntimeError("ASU progress checkpoint contains duplicate episode IDs")
    new_rows = 0
    total_records = 0
    for shard in shards:
        for record_index, record in enumerate(iter_tfrecord(shard)):
            total_records += 1
            source = decode_asu_episode(record, source_shard=shard.name, source_record_index=record_index)
            episode_id = str(source["episode_id"])
            if episode_id in completed:
                continue
            numeric = processed_root / "numeric"
            state_rel = Path("numeric") / f"{episode_id}_robot_state.npz"
            relation_rel = Path("numeric") / f"{episode_id}_relation_state.npz"
            observation_rel = Path("numeric") / f"{episode_id}_observation_state.npz"
            action_rel = Path("numeric") / f"{episode_id}_action.npz"
            time_rel = Path("numeric") / f"{episode_id}_timestamp.npz"
            _save_value(processed_root / state_rel, source["robot_state"])
            _save_value(processed_root / relation_rel, source["relation_state"])
            _save_value(processed_root / observation_rel, source["observation_state"])
            _save_value(processed_root / action_rel, source["action"])
            _save_value(processed_root / time_rel, source["timestamp"])
            image_rows: list[dict[str, Any]] = []
            for step in _sample_steps(len(source["timestamp"]), image_stride):
                relative = Path("rgb") / episode_id / f"{step:05d}.jpg"
                _save_thumbnail(source["rgb_png"][step], processed_root / relative, max_width=thumbnail_width)
                image_rows.append({"step": step, "path": relative.as_posix(), "camera": "main_camera"})
            rows.append({
                "schema": NORMALIZED_SCHEMA_VERSION,
                "episode_id": episode_id,
                "task_id": source["task_id"],
                "task_id_source": source["task_id_source"],
                "language_instruction": source["language_instruction"],
                "robot_state_path": state_rel.as_posix(),
                "relation_state_path": relation_rel.as_posix(),
                "observation_state_path": observation_rel.as_posix(),
                "relation_state_names": list(RELATION_NAMES),
                "action_path": action_rel.as_posix(),
                "rgb_paths": image_rows,
                "timestamp_path": time_rel.as_posix(),
                "timestamp_source": source["timestamp_source"],
                "success": source["success"],
                "outcome_source": source["outcome_source"],
                "source_dataset": ASU_DATASET,
                "split": stable_episode_split(episode_id),
                "source_shard": source["source_shard"],
                "source_record_index": source["source_record_index"],
                "source_file_path": source["source_file_path"],
            })
            completed.add(episode_id)
            new_rows += 1
            rows.sort(key=lambda row: str(row["episode_id"]))
            write_jsonl(progress_path, rows)
            if max_new_episodes is not None and new_rows >= max_new_episodes:
                return {
                    "complete": False,
                    "progress": {
                        "episodes_checkpointed": len(rows), "new_episodes": new_rows,
                        "expected_episodes": 110, "image_stride": image_stride, "thumbnail_width": thumbnail_width,
                    },
                }
    rows.sort(key=lambda row: str(row["episode_id"]))
    if total_records != 110 or len(rows) != total_records:
        raise RuntimeError(f"ASU conversion incomplete: parsed {total_records} records, checkpoint has {len(rows)} episodes")
    index_path = processed_root / "episodes.jsonl"
    write_jsonl(index_path, rows)
    audits = [validate_normalized_episode(row, processed_root=processed_root) for row in rows]
    required_files = [index_path]
    for row in rows:
        required_files.extend(processed_root / row[key] for key in ("robot_state_path", "relation_state_path", "observation_state_path", "action_path", "timestamp_path"))
        required_files.extend(processed_root / image["path"] for image in row["rgb_paths"])
    data_hash = composite_file_hash(processed_root, required_files)
    source_hash = composite_file_hash(source_root, shards)
    manifest = {
        "schema": "r-series-asu-processed-subset-v1", "dataset": ASU_DATASET,
        "episodes": len(rows), "tasks": len({row["task_id"] for row in rows}),
        "splits": {split: sum(row["split"] == split for row in rows) for split in ("train", "validation", "test")},
        "relation_state": {"names": list(RELATION_NAMES), "dimensions_per_entity": 6, "source": "ASU dataset-provided ground_truth_states"},
        "source_sha256": source_hash["composite_sha256"], "processed_data_sha256": data_hash["composite_sha256"],
        "source_file_hashes": source_hash["entries"], "processed_file_hashes": data_hash["entries"],
        "image_stride": image_stride, "thumbnail_width": thumbnail_width,
        "timestamp_policy": "ordered step index because the public schema has no physical timestamp field",
    }
    audit = {
        "schema": "r-series-asu-adapter-audit-v1", "required_features": list(REQUIRED_ASU_FEATURES),
        "episodes_audited": audits, "all_pass": len(audits) == len(rows),
        "relation_alignment": "all seven 6-D provided entity poses have one sample per action and observation step",
        "source_shard_sha256": {path.name: sha256_file(path) for path in shards},
    }
    write_json(processed_root / "data_manifest.json", manifest)
    write_json(processed_root / "adapter_audit.json", audit)
    if log_path is not None:
        append_run_log(log_path, event="asu_tabletop_backup_processed", payload={"episodes": len(rows), "tasks": manifest["tasks"], "data_sha256": manifest["processed_data_sha256"]})
    return {"complete": True, "manifest": manifest, "audit": audit, "rows": rows}


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=R_SERIES_ROOT)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--image-stride", type=int, default=8)
    parser.add_argument("--thumbnail-width", type=int, default=96)
    parser.add_argument("--max-new-episodes", type=int, default=None)
    args = parser.parse_args()
    source = args.root / "source_subset" / "asu_tabletop_rlds"
    processed = args.root / "processed_subset_asu_tabletop"
    if args.download:
        download_asu_table_top(source, log_path=args.root / "run_log.jsonl")
    if args.prepare:
        result = prepare_processed_subset(source, processed, image_stride=args.image_stride, thumbnail_width=args.thumbnail_width, max_new_episodes=args.max_new_episodes, log_path=args.root / "run_log.jsonl")
        print(json.dumps(result.get("manifest", result), indent=2, sort_keys=True))
    if not (args.download or args.prepare):
        parser.error("select --download and/or --prepare")


if __name__ == "__main__":
    _main()
