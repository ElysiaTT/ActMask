"""Shared, deliberately small utilities for the RM-SPD research branch.

This module owns only files below ``rm_spd_sam3_prior_dynamics``.  In
particular, it must never write to the previous RM/SAM3/mask branches.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "actmask" / "rm_spd_sam3_prior_dynamics"
RM_SERIES = ROOT / "outputs" / "actmask" / "rm_series_robomind"
PYTHON = Path("/home/tzh/conda_envs/actmask/bin/python")


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_value(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def append_log(event: str, **payload: Any) -> None:
    """Append a hash-chained event to the sole permitted branch log."""

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "run_log.jsonl"
    prior_hash, sequence = "", 0
    if path.is_file() and path.stat().st_size:
        for line in path.read_text().splitlines():
            if line.strip():
                last = json.loads(line)
                sequence = int(last["seq"])
                prior_hash = str(last["event_sha256"])
    row: dict[str, Any] = {
        "seq": sequence + 1,
        "utc": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "payload": payload,
        "prev_event_sha256": prior_hash,
    }
    row["event_sha256"] = sha256_value(row)
    with path.open("a") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def config() -> dict[str, Any]:
    path = OUT / "preregistered_config.json"
    if not path.is_file():
        raise RuntimeError("RM-SPD S0 is missing; run rm_spd_freeze.py first")
    value = read_json(path)
    expected = (OUT / "preregistered_config.sha256").read_text().strip()
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError("RM-SPD preregistered config hash mismatch")
    return value


def require_preregistered_config() -> dict[str, Any]:
    value = config()
    if value.get("branch") != "RM-SPD":
        raise RuntimeError("wrong branch configuration")
    return value
