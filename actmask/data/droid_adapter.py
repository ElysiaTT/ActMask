"""A small, auditable DROID RLDS adapter with no TensorFlow dependency.

DROID's official debugging release contains 100 real Franka demonstrations in
TFRecord/RLDS format.  The project environment intentionally does not install
TensorFlow just to read that small subset, so this adapter implements the tiny
well-specified portion of TFRecord and ``tf.train.Example`` needed by the
official flat RLDS records.  It validates record CRCs and feature dimensions,
then writes an indexed, disk-conscious normalized subset.

This is not a general protobuf or TensorFlow implementation.  Unsupported
wire types and malformed feature payloads fail closed rather than being
silently skipped.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import shutil
import struct
import subprocess
import time
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np
from PIL import Image

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


DROID_DATASET = "DROID"
DROID_SAMPLE_PREFIX = "robotics/droid_100/1.0.0"
DROID_BUCKET = "gresearch"
STORAGE_API = f"https://storage.googleapis.com/storage/v1/b/{DROID_BUCKET}/o"
DOWNLOAD_API = f"https://storage.googleapis.com/download/storage/v1/b/{DROID_BUCKET}/o"
EXPECTED_IMAGE_KEY = "steps/observation/exterior_image_1_left"
REQUIRED_DROID_FEATURES = (
    "steps/action",
    "steps/observation/cartesian_position",
    "steps/observation/joint_position",
    "steps/observation/gripper_position",
    "steps/language_instruction",
    EXPECTED_IMAGE_KEY,
)


class DroidFormatError(ValueError):
    """Raised when an official data record does not satisfy the adapter contract."""


def _get_json(url: str, *, timeout: int = 60) -> dict[str, Any]:
    """Fetch fixed public GCS metadata through the host's trusted ``curl``.

    The current container's Python CA bundle is incomplete while the system
    curl trust store is correctly configured.  Calling curl with a fixed URL
    list is safer than disabling certificate validation in Python.
    """

    if shutil.which("curl") is None:
        raise RuntimeError("DROID adapter requires the host curl executable for HTTPS transport")
    completed = subprocess.run(
        ["curl", "--fail", "--location", "--silent", "--show-error", "--max-time", str(timeout), url],
        check=False,
        capture_output=True,
    )
    if completed.returncode:
        raise RuntimeError(
            f"curl metadata fetch failed ({completed.returncode}): {completed.stderr.decode('utf-8', errors='replace').strip()}"
        )
    return json.loads(completed.stdout.decode("utf-8"))


def list_droid_100_remote_objects() -> list[dict[str, Any]]:
    """List the public sample's metadata and TFRecord objects from GCS."""

    token: str | None = None
    result: list[dict[str, Any]] = []
    while True:
        suffix = f"?prefix={quote(DROID_SAMPLE_PREFIX + '/', safe='')}&maxResults=1000"
        if token:
            suffix += f"&pageToken={quote(token, safe='')}"
        payload = _get_json(STORAGE_API + suffix)
        for row in payload.get("items", []):
            name = str(row.get("name", ""))
            if name.endswith(("dataset_info.json", "features.json")) or ".tfrecord-" in name:
                result.append(
                    {
                        "name": name,
                        "bytes": int(row["size"]),
                        "md5_base64": row.get("md5Hash"),
                        "etag": row.get("etag"),
                        "updated": row.get("updated"),
                    }
                )
        token = payload.get("nextPageToken")
        if not token:
            break
    if not result:
        raise RuntimeError("DROID droid_100 remote listing was unexpectedly empty")
    return sorted(result, key=lambda item: str(item["name"]))


def remote_media_url(remote_name: str) -> str:
    return f"{DOWNLOAD_API}/{quote(remote_name, safe='')}?alt=media"


def download_remote_object(
    remote: Mapping[str, Any],
    destination: str | Path,
    *,
    timeout: int = 120,
) -> dict[str, Any]:
    """Resumably download one known-size public object and validate its size."""

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected = int(remote["bytes"])
    if destination.is_file() and destination.stat().st_size == expected:
        return {"path": str(destination), "status": "already_present", "bytes": expected}
    partial = destination.with_name(destination.name + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > expected:
        raise RuntimeError(f"partial download exceeds known remote size: {partial}")
    if shutil.which("curl") is None:
        raise RuntimeError("DROID adapter requires the host curl executable for HTTPS transport")
    command = [
        "curl",
        "--fail",
        "--location",
        "--retry",
        "4",
        "--retry-delay",
        "2",
        "--connect-timeout",
        "30",
        "--speed-limit",
        "1",
        "--speed-time",
        str(timeout),
        "--continue-at",
        "-",
        "--output",
        str(partial),
        remote_media_url(str(remote["name"])),
    ]
    completed = subprocess.run(command, check=False, capture_output=True)
    if completed.returncode:
        raise RuntimeError(
            f"curl download failed for {remote['name']} ({completed.returncode}): "
            f"{completed.stderr.decode('utf-8', errors='replace').strip()}"
        )
    if partial.stat().st_size != expected:
        raise RuntimeError(
            f"incomplete download for {remote['name']}: {partial.stat().st_size} != {expected}"
        )
    partial.replace(destination)
    return {"path": str(destination), "status": "downloaded", "bytes": expected}


def download_droid_100(
    source_root: str | Path,
    *,
    max_shards: int | None = None,
    log_path: str | Path | None = None,
) -> dict[str, Any]:
    """Download only the official metadata plus a bounded number of shards.

    ``max_shards=None`` means all 31 shards (the documented 100-episode,
    approximately 2 GB debugging release).  The caller can use a smaller
    number for a format or adapter smoke test.
    """

    source_root = Path(source_root)
    source_root.mkdir(parents=True, exist_ok=True)
    remote = list_droid_100_remote_objects()
    metadata = [item for item in remote if str(item["name"]).endswith(".json")]
    shards = [item for item in remote if ".tfrecord-" in str(item["name"])]
    if max_shards is not None:
        if max_shards <= 0:
            raise ValueError("max_shards must be positive when provided")
        shards = shards[: int(max_shards)]
    selected = metadata + shards
    downloaded: list[dict[str, Any]] = []
    for item in selected:
        relative = Path(str(item["name"])).relative_to(DROID_SAMPLE_PREFIX)
        downloaded.append(download_remote_object(item, source_root / relative))
    manifest = {
        "schema": "r-series-droid-public-download-v1",
        "dataset": DROID_DATASET,
        "remote_prefix": f"gs://{DROID_BUCKET}/{DROID_SAMPLE_PREFIX}",
        "remote_inventory": selected,
        "requested_max_shards": max_shards,
        "downloaded": downloaded,
        "downloaded_bytes": int(sum(item["bytes"] for item in downloaded)),
    }
    write_json(source_root / "download_manifest.json", manifest)
    if log_path is not None:
        append_run_log(
            log_path,
            event="droid_sample_download",
            payload={"source_root": str(source_root), "shards": len(shards), "bytes": manifest["downloaded_bytes"]},
        )
    return manifest


# --- Minimal, strict TFRecord / tf.train.Example decoder -----------------


def _read_varint(data: bytes, position: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while True:
        if position >= len(data):
            raise DroidFormatError("truncated protobuf varint")
        byte = data[position]
        position += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, position
        shift += 7
        if shift > 70:
            raise DroidFormatError("overlong protobuf varint")


def _protobuf_fields(data: bytes) -> list[tuple[int, int, bytes | int]]:
    position = 0
    fields: list[tuple[int, int, bytes | int]] = []
    while position < len(data):
        key, position = _read_varint(data, position)
        number, wire_type = key >> 3, key & 0x07
        if number <= 0:
            raise DroidFormatError("invalid protobuf field number")
        if wire_type == 0:
            value, position = _read_varint(data, position)
        elif wire_type == 1:
            if position + 8 > len(data):
                raise DroidFormatError("truncated fixed64 field")
            value = data[position : position + 8]
            position += 8
        elif wire_type == 2:
            length, position = _read_varint(data, position)
            if length < 0 or position + length > len(data):
                raise DroidFormatError("truncated length-delimited field")
            value = data[position : position + length]
            position += length
        elif wire_type == 5:
            if position + 4 > len(data):
                raise DroidFormatError("truncated fixed32 field")
            value = data[position : position + 4]
            position += 4
        else:
            raise DroidFormatError(f"unsupported protobuf wire type {wire_type}")
        fields.append((number, wire_type, value))
    return fields


def _single_message(fields: Iterable[tuple[int, int, bytes | int]], number: int, label: str) -> bytes:
    values = [value for field, wire, value in fields if field == number and wire == 2]
    if len(values) != 1 or not isinstance(values[0], bytes):
        raise DroidFormatError(f"{label} requires exactly one length-delimited field {number}")
    return values[0]


def _decode_bytes_list(data: bytes) -> list[bytes]:
    return [value for number, wire, value in _protobuf_fields(data) if number == 1 and wire == 2 and isinstance(value, bytes)]


def _decode_float_list(data: bytes) -> np.ndarray:
    values: list[np.ndarray] = []
    for number, wire, value in _protobuf_fields(data):
        if number != 1:
            continue
        if wire == 2:
            if not isinstance(value, bytes) or len(value) % 4:
                raise DroidFormatError("packed FloatList payload is not float32-aligned")
            values.append(np.frombuffer(value, dtype="<f4"))
        elif wire == 5:
            if not isinstance(value, bytes):
                raise DroidFormatError("invalid fixed32 FloatList field")
            values.append(np.frombuffer(value, dtype="<f4"))
        else:
            raise DroidFormatError("FloatList used an unsupported wire type")
    return np.concatenate(values) if values else np.empty((0,), dtype=np.float32)


def _signed_int64(value: int) -> int:
    return value - (1 << 64) if value >= (1 << 63) else value


def _decode_int64_list(data: bytes) -> np.ndarray:
    result: list[int] = []
    for number, wire, value in _protobuf_fields(data):
        if number != 1:
            continue
        if wire == 0:
            assert isinstance(value, int)
            result.append(_signed_int64(value))
        elif wire == 2:
            assert isinstance(value, bytes)
            position = 0
            while position < len(value):
                scalar, position = _read_varint(value, position)
                result.append(_signed_int64(scalar))
        else:
            raise DroidFormatError("Int64List used an unsupported wire type")
    return np.asarray(result, dtype=np.int64)


def parse_tf_example(data: bytes) -> dict[str, tuple[str, np.ndarray | list[bytes]]]:
    """Decode the flat feature map written by TFDS for the DROID RLDS sample."""

    top = _protobuf_fields(data)
    features_blob = _single_message(top, 1, "Example")
    result: dict[str, tuple[str, np.ndarray | list[bytes]]] = {}
    for field, wire, entry in _protobuf_fields(features_blob):
        if field != 1 or wire != 2:
            continue
        assert isinstance(entry, bytes)
        entry_fields = _protobuf_fields(entry)
        key_blob = _single_message(entry_fields, 1, "Features map entry key")
        value_blob = _single_message(entry_fields, 2, "Features map entry value")
        try:
            key = key_blob.decode("utf-8")
        except UnicodeDecodeError as error:
            raise DroidFormatError("non-UTF8 TFDS feature name") from error
        if key in result:
            raise DroidFormatError(f"duplicate TFDS feature name: {key}")
        feature_fields = _protobuf_fields(value_blob)
        kinds = [(field_number, value) for field_number, wire_type, value in feature_fields if wire_type == 2]
        if len(kinds) != 1 or not isinstance(kinds[0][1], bytes):
            raise DroidFormatError(f"feature {key!r} has no unique list payload")
        kind, payload = kinds[0]
        if kind == 1:
            result[key] = ("bytes", _decode_bytes_list(payload))
        elif kind == 2:
            result[key] = ("float", _decode_float_list(payload))
        elif kind == 3:
            result[key] = ("int64", _decode_int64_list(payload))
        else:
            raise DroidFormatError(f"feature {key!r} uses unsupported Feature oneof {kind}")
    return result


_CRC32C_TABLE: tuple[int, ...] | None = None


def _crc32c(data: bytes) -> int:
    global _CRC32C_TABLE
    if _CRC32C_TABLE is None:
        table: list[int] = []
        for value in range(256):
            word = value
            for _ in range(8):
                word = (word >> 1) ^ (0x82F63B78 if word & 1 else 0)
            table.append(word & 0xFFFFFFFF)
        _CRC32C_TABLE = tuple(table)
    crc = 0xFFFFFFFF
    for byte in data:
        crc = _CRC32C_TABLE[(crc ^ byte) & 0xFF] ^ (crc >> 8)
    return (~crc) & 0xFFFFFFFF


def _masked_crc32c(data: bytes) -> int:
    value = _crc32c(data)
    return (((value >> 15) | (value << 17)) + 0xA282EAD8) & 0xFFFFFFFF


def iter_tfrecord(path: str | Path, *, verify_crc: bool = True) -> Iterator[bytes]:
    """Yield complete records, rejecting bad length or CRC fields."""

    with Path(path).open("rb") as handle:
        index = 0
        while True:
            header = handle.read(12)
            if not header:
                return
            if len(header) != 12:
                raise DroidFormatError(f"truncated TFRecord header at record {index}")
            length = struct.unpack("<Q", header[:8])[0]
            if verify_crc and struct.unpack("<I", header[8:])[0] != _masked_crc32c(header[:8]):
                raise DroidFormatError(f"bad TFRecord length CRC at record {index}")
            record = handle.read(length)
            trailer = handle.read(4)
            if len(record) != length or len(trailer) != 4:
                raise DroidFormatError(f"truncated TFRecord payload at record {index}")
            if verify_crc and struct.unpack("<I", trailer)[0] != _masked_crc32c(record):
                raise DroidFormatError(f"bad TFRecord payload CRC at record {index}")
            yield record
            index += 1


def _feature_float(features: Mapping[str, tuple[str, np.ndarray | list[bytes]]], key: str, width: int) -> np.ndarray:
    kind, value = features.get(key, (None, None))  # type: ignore[arg-type]
    if kind != "float" or not isinstance(value, np.ndarray) or value.size == 0 or value.size % width:
        raise DroidFormatError(f"DROID feature {key!r} is not a nonempty float array of width {width}")
    return value.reshape(-1, width).astype(np.float32, copy=False)


def _feature_bytes(features: Mapping[str, tuple[str, np.ndarray | list[bytes]]], key: str) -> list[bytes]:
    kind, value = features.get(key, (None, None))  # type: ignore[arg-type]
    if kind != "bytes" or not isinstance(value, list) or not value:
        raise DroidFormatError(f"DROID feature {key!r} is not a nonempty bytes list")
    return value


def _feature_int(features: Mapping[str, tuple[str, np.ndarray | list[bytes]]], key: str) -> np.ndarray:
    kind, value = features.get(key, (None, None))  # type: ignore[arg-type]
    if kind != "int64" or not isinstance(value, np.ndarray) or value.size == 0:
        raise DroidFormatError(f"DROID feature {key!r} is not a nonempty int64 list")
    return value


def _language_from_feature(features: Mapping[str, tuple[str, np.ndarray | list[bytes]]], steps: int) -> str:
    values = _feature_bytes(features, "steps/language_instruction")
    decoded = [value.decode("utf-8", errors="replace").strip() for value in values]
    nonempty = next((value for value in decoded if value), "")
    if not nonempty:
        for alternate in ("steps/language_instruction_2", "steps/language_instruction_3"):
            if alternate in features:
                candidate = _feature_bytes(features, alternate)
                nonempty = next((value.decode("utf-8", errors="replace").strip() for value in candidate if value.strip()), "")
                if nonempty:
                    break
    if not nonempty:
        raise DroidFormatError("DROID episode has no usable language instruction")
    if len(values) not in (1, steps):
        raise DroidFormatError("DROID language feature has an unexpected step count")
    return nonempty


def _metadata_text(features: Mapping[str, tuple[str, np.ndarray | list[bytes]]], key: str) -> str:
    values = _feature_bytes(features, key)
    if len(values) != 1:
        raise DroidFormatError(f"DROID metadata {key!r} must be scalar")
    return values[0].decode("utf-8", errors="replace")


def language_task_family(language: str) -> str:
    """Make an explicit, non-pretrained coarse task family from language.

    DROID's public RLDS schema exposes natural-language instructions, not a
    canonical task-ID field.  A transparent first-content-word family gives
    the R-Series a bounded task grouping without pretending that it is an
    official DROID annotation.  The original instruction remains intact.
    """

    ignored = {"please", "can", "could", "would", "you", "the", "a", "an", "to", "my", "your"}
    words = re.findall(r"[a-z]+", language.lower())
    token = next((word for word in words if word not in ignored), "unknown")
    aliases = {"place": "put", "move": "put", "grab": "pick", "take": "pick"}
    return "language_verb_" + aliases.get(token, token)


def decode_droid_episode(
    record: bytes,
    *,
    source_shard: str,
    source_record_index: int,
) -> dict[str, Any]:
    """Return one in-memory normalized episode plus raw JPEG bytes."""

    features = parse_tf_example(record)
    missing = [key for key in REQUIRED_DROID_FEATURES if key not in features]
    if missing:
        raise DroidFormatError(f"DROID record is missing required features: {missing}")
    action = _feature_float(features, "steps/action", 7)
    cartesian = _feature_float(features, "steps/observation/cartesian_position", 6)
    joint = _feature_float(features, "steps/observation/joint_position", 7)
    gripper = _feature_float(features, "steps/observation/gripper_position", 1)
    n = len(action)
    if any(len(value) != n for value in (cartesian, joint, gripper)):
        raise DroidFormatError("DROID robot-state/action feature lengths disagree")
    images = _feature_bytes(features, EXPECTED_IMAGE_KEY)
    if len(images) != n:
        raise DroidFormatError(f"DROID RGB step count {len(images)} disagrees with numeric count {n}")
    language = _language_from_feature(features, n)
    terminal = _feature_int(features, "steps/is_terminal") if "steps/is_terminal" in features else np.zeros(n, np.int64)
    reward = _feature_float(features, "steps/reward", 1).reshape(-1) if "steps/reward" in features else np.zeros(n, np.float32)
    if len(terminal) not in (1, n) or len(reward) not in (1, n):
        raise DroidFormatError("DROID terminal/reward feature length is invalid")
    file_path = _metadata_text(features, "episode_metadata/file_path")
    identifier = hashlib.sha256(f"{source_shard}:{source_record_index}:{file_path}".encode("utf-8")).hexdigest()[:24]
    task_id = language_task_family(language)
    success = bool((terminal[-1] if len(terminal) else 0) or (reward[-1] if len(reward) else 0.0) > 0.5)
    return {
        "episode_id": f"droid_{identifier}",
        "task_id": task_id,
        "language_instruction": language,
        "robot_state": np.concatenate((cartesian, joint, gripper), axis=1),
        "action": action,
        # The official RLDS schema has ordered steps but no physical clock.
        "timestamp": np.arange(n, dtype=np.float32),
        "timestamp_source": "ordered_rlds_step_index_no_declared_frame_rate",
        "rgb_jpeg": images,
        "success": success,
        "outcome_source": "DROID RLDS terminal/reward demo marker; no failure class inferred",
        "source_shard": source_shard,
        "source_record_index": int(source_record_index),
        "source_file_path": file_path,
    }


def _save_value(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, value=value)


def _save_thumbnail(jpeg: bytes, path: Path, *, max_width: int) -> None:
    with Image.open(io.BytesIO(jpeg)) as image:
        image = image.convert("RGB")
        ratio = min(1.0, float(max_width) / float(image.width))
        size = (max(1, round(image.width * ratio)), max(1, round(image.height * ratio)))
        image.resize(size, Image.Resampling.BILINEAR).save(path, format="JPEG", quality=80, optimize=True)


def _sample_steps(length: int, stride: int) -> tuple[int, ...]:
    values = list(range(0, length, stride))
    if values[-1] != length - 1:
        values.append(length - 1)
    return tuple(values)


def prepare_processed_subset(
    source_root: str | Path,
    processed_root: str | Path,
    *,
    max_episodes: int = 100,
    image_stride: int = 8,
    thumbnail_width: int = 96,
    log_path: str | Path | None = None,
) -> dict[str, Any]:
    """Parse local public shards into the R-Series normalized small subset."""

    if max_episodes <= 0 or image_stride <= 0 or thumbnail_width <= 0:
        raise ValueError("max_episodes, image_stride, and thumbnail_width must be positive")
    source_root, processed_root = Path(source_root), Path(processed_root)
    shards = sorted(source_root.glob("*.tfrecord-*"))
    if not shards:
        raise FileNotFoundError(f"no DROID TFRecord shards found under {source_root}")
    processed_root.mkdir(parents=True, exist_ok=True)
    index_path = processed_root / "episodes.jsonl"
    existing: dict[str, dict[str, Any]] = {}
    if index_path.exists():
        existing = {row["episode_id"]: row for row in json.loads("[" + ",".join(index_path.read_text().splitlines()) + "]") if row}
        for row in existing.values():
            row["task_id"] = language_task_family(str(row["language_instruction"]))
            row["task_id_source"] = "deterministic first-content-word language family; DROID RLDS has no canonical task_id"
    rows = list(existing.values())
    decoded_count = 0
    skipped_missing_language: list[dict[str, Any]] = []
    for shard in shards:
        for record_index, record in enumerate(iter_tfrecord(shard)):
            if len(rows) >= max_episodes:
                break
            try:
                episode = decode_droid_episode(record, source_shard=shard.name, source_record_index=record_index)
            except DroidFormatError as error:
                if str(error) != "DROID episode has no usable language instruction":
                    raise
                skipped_missing_language.append({"source_shard": shard.name, "source_record_index": record_index, "reason": str(error)})
                continue
            if episode["episode_id"] in existing:
                continue
            episode_id = str(episode["episode_id"])
            numeric = processed_root / "numeric"
            state_rel = Path("numeric") / f"{episode_id}_robot_state.npz"
            action_rel = Path("numeric") / f"{episode_id}_action.npz"
            time_rel = Path("numeric") / f"{episode_id}_timestamp.npz"
            _save_value(processed_root / state_rel, episode["robot_state"])
            _save_value(processed_root / action_rel, episode["action"])
            _save_value(processed_root / time_rel, episode["timestamp"])
            image_rows: list[dict[str, Any]] = []
            image_dir = processed_root / "rgb" / episode_id
            image_dir.mkdir(parents=True, exist_ok=True)
            for step in _sample_steps(len(episode["timestamp"]), image_stride):
                relative = Path("rgb") / episode_id / f"{step:05d}.jpg"
                _save_thumbnail(episode["rgb_jpeg"][step], processed_root / relative, max_width=thumbnail_width)
                image_rows.append({"step": step, "path": relative.as_posix(), "camera": "exterior_image_1_left"})
            row = {
                "schema": NORMALIZED_SCHEMA_VERSION,
                "episode_id": episode_id,
                "task_id": episode["task_id"],
                "task_id_source": "deterministic first-content-word language family; DROID RLDS has no canonical task_id",
                "language_instruction": episode["language_instruction"],
                "robot_state_path": state_rel.as_posix(),
                "action_path": action_rel.as_posix(),
                "rgb_paths": image_rows,
                "timestamp_path": time_rel.as_posix(),
                "timestamp_source": episode["timestamp_source"],
                "success": episode["success"],
                "outcome_source": episode["outcome_source"],
                "source_dataset": DROID_DATASET,
                "split": stable_episode_split(episode_id),
                # Provenance only; the experiment explicitly excludes these from fair inputs.
                "source_shard": episode["source_shard"],
                "source_record_index": episode["source_record_index"],
                "source_file_path": episode["source_file_path"],
            }
            rows.append(row)
            existing[episode_id] = row
            decoded_count += 1
        if len(rows) >= max_episodes:
            break
    rows.sort(key=lambda row: str(row["episode_id"]))
    write_jsonl(index_path, rows)
    audits = [validate_normalized_episode(row, processed_root=processed_root) for row in rows]
    all_files = [index_path]
    for row in rows:
        all_files += [
            processed_root / row["robot_state_path"],
            processed_root / row["action_path"],
            processed_root / row["timestamp_path"],
        ]
        all_files += [processed_root / image["path"] for image in row["rgb_paths"]]
    data_hash = composite_file_hash(processed_root, all_files)
    source_hash = composite_file_hash(source_root, [path for path in shards if path.is_file()])
    split_counts = {split: sum(row["split"] == split for row in rows) for split in ("train", "validation", "test")}
    manifest = {
        "schema": "r-series-droid-processed-subset-v1",
        "dataset": DROID_DATASET,
        "source_root": str(source_root),
        "source_shards": [path.name for path in shards],
        "source_sha256": source_hash["composite_sha256"],
        "processed_data_sha256": data_hash["composite_sha256"],
        "episodes": len(rows),
        "skipped_missing_language_episodes": len(skipped_missing_language),
        "tasks": len({row["task_id"] for row in rows}),
        "splits": split_counts,
        "image_stride": image_stride,
        "thumbnail_width": thumbnail_width,
        "timestamp_policy": "ordered step index because DROID RLDS schema contains no physical timestamp field",
        "source_file_hashes": source_hash["entries"],
        "processed_file_hashes": data_hash["entries"],
    }
    audit = {
        "schema": "r-series-droid-adapter-audit-v1",
        "required_features": list(REQUIRED_DROID_FEATURES),
        "episodes_audited": audits,
        "all_pass": len(audits) == len(rows) and all(item["steps"] >= 4 for item in audits),
        "newly_decoded_episodes": decoded_count,
        "skipped_missing_language": skipped_missing_language,
        "source_shard_sha256": {path.name: sha256_file(path) for path in shards},
        "timestamp_alignment": "one ordered index per action and robot-state sample; no physical clock claimed",
    }
    write_json(processed_root / "data_manifest.json", manifest)
    write_json(processed_root / "adapter_audit.json", audit)
    if log_path is not None:
        append_run_log(
            log_path,
            event="droid_subset_processed",
            payload={"episodes": len(rows), "tasks": manifest["tasks"], "data_sha256": manifest["processed_data_sha256"]},
        )
    return {"manifest": manifest, "audit": audit, "rows": rows}


def load_processed_subset(processed_root: str | Path) -> list[dict[str, Any]]:
    """Load the JSONL index only; numeric arrays remain lazy on disk."""

    path = Path(processed_root) / "episodes.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=R_SERIES_ROOT)
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--max-episodes", type=int, default=100)
    parser.add_argument("--image-stride", type=int, default=8)
    parser.add_argument("--thumbnail-width", type=int, default=96)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    source = args.root / "source_subset" / "droid_100_rlds"
    processed = args.root / "processed_subset"
    log = args.root / "run_log.jsonl"
    if args.download:
        download_droid_100(source, max_shards=args.max_shards, log_path=log)
    if args.prepare:
        result = prepare_processed_subset(
            source,
            processed,
            max_episodes=args.max_episodes,
            image_stride=args.image_stride,
            thumbnail_width=args.thumbnail_width,
            log_path=log,
        )
        print(json.dumps(result["manifest"], indent=2, sort_keys=True))
    if not args.download and not args.prepare:
        parser.error("specify --download, --prepare, or both")


if __name__ == "__main__":
    _main()
