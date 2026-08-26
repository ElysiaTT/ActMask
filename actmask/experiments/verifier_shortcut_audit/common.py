"""Shared deterministic utilities for VERIFIER-SHORTCUT-AUDIT-V1."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "outputs" / "actmask" / "verifier_shortcut_audit_v1"
HISTORICAL = ROOT / "outputs" / "actmask"
PYTHON = Path("/home/tzh/conda_envs/actmask/bin/python")
STUDY_ID = "VERIFIER-SHORTCUT-AUDIT-V1"
SEEDS = (17, 29, 43)


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def stable_int(value: str, modulus: int = 2**31 - 1) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:16], 16) % modulus


def append_log(event: str, **payload: Any) -> None:
    path = OUT / "run_log.jsonl"
    rows = read_jsonl(path) if path.exists() else []
    previous = rows[-1]["record_sha256"] if rows else None
    record = {
        "sequence": len(rows),
        "study_id": STUDY_ID,
        "event": event,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "previous_record_sha256": previous,
        **payload,
    }
    record["record_sha256"] = sha256_json(record)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

