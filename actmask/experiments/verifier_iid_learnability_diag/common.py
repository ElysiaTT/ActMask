"""Shared utilities for VERIFIER-IID-LEARNABILITY-DIAG-V1."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "outputs" / "actmask" / "verifier_shortcut_audit_v1"
OUT = ROOT / "outputs" / "actmask" / "verifier_iid_learnability_diag_v1"
PYTHON = Path("/home/tzh/conda_envs/actmask/bin/python")
STUDY_ID = "VERIFIER-IID-LEARNABILITY-DIAG-V1"
SEEDS = (17, 29, 43)
EASY_FAMILIES = ("N2", "N3", "N5", "N6")
SPECS = tuple(f"within_{family.lower()}" for family in EASY_FAMILIES)
ARCHITECTURES = (
    "StateActionTCN",
    "TaskConditionedActionTransformer",
    "VisualStateActionTransformer",
    "ContrastiveContextActionVerifier",
    "CrossAttentionEnergyVerifier",
)


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


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)


def with_payload_sha256(value: dict[str, Any]) -> dict[str, Any]:
    payload = dict(value)
    payload.pop("record_payload_sha256", None)
    payload["record_payload_sha256"] = sha256_json(payload)
    return payload


def verify_payload_sha256(value: dict[str, Any]) -> bool:
    recorded = value.get("record_payload_sha256")
    payload = dict(value)
    payload.pop("record_payload_sha256", None)
    return isinstance(recorded, str) and recorded == sha256_json(payload)


def write_hashed_json(
    path: str | Path, value: dict[str, Any]
) -> dict[str, Any]:
    payload = with_payload_sha256(value)
    write_json(path, payload)
    return payload


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    temporary.replace(target)


def directory_snapshot(root: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": str(path.relative_to(root)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]


def verify_snapshot(root: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    expected = {row["path"]: row for row in rows}
    current = {
        str(path.relative_to(root)): path
        for path in root.rglob("*")
        if path.is_file()
    }
    missing = sorted(set(expected) - set(current))
    extra = sorted(set(current) - set(expected))
    mismatches = []
    for name in sorted(set(expected) & set(current)):
        row = expected[name]
        path = current[name]
        if path.stat().st_size != row["bytes"] or sha256_file(path) != row["sha256"]:
            mismatches.append(name)
    return {
        "pass": not missing and not extra and not mismatches,
        "expected_files": len(expected),
        "current_files": len(current),
        "missing": missing,
        "extra": extra,
        "content_or_size_mismatches": mismatches,
        "snapshot_sha256": sha256_json(rows),
    }


def verify_bound_source_snapshot(
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Verify both the frozen snapshot identity and every source-study file."""
    if config is None:
        config = read_json(OUT / "preregistered_config.json")
    frozen = config["source_study"]
    snapshot = read_json(OUT / "source_preservation_snapshot.json")
    rows = snapshot["files"]
    rows_sha256 = sha256_json(rows)
    rows_bytes = sum(int(row["bytes"]) for row in rows)
    binding_failures = []
    if rows_sha256 != frozen["snapshot_sha256"]:
        binding_failures.append("snapshot_sha256")
    if len(rows) != int(frozen["snapshot_records"]):
        binding_failures.append("snapshot_records")
    if rows_bytes != int(frozen["snapshot_bytes"]):
        binding_failures.append("snapshot_bytes")
    embedded_sha256 = snapshot.get(
        "snapshot_sha256", snapshot.get("tree_sha256")
    )
    embedded_count = snapshot.get(
        "files_count", snapshot.get("file_count")
    )
    if embedded_sha256 != rows_sha256:
        binding_failures.append("snapshot_file_embedded_sha256")
    if embedded_count != len(rows):
        binding_failures.append("snapshot_file_embedded_count")
    if snapshot.get("total_bytes") != rows_bytes:
        binding_failures.append("snapshot_file_embedded_bytes")
    content = verify_snapshot(SOURCE, rows)
    result = {
        **content,
        "binding_failures": binding_failures,
        "snapshot_binding_pass": not binding_failures,
        "pass": content["pass"] and not binding_failures,
        "snapshot_records": len(rows),
        "snapshot_bytes": rows_bytes,
    }
    return result


def verify_log_chain(path: Path | None = None) -> dict[str, Any]:
    target = OUT / "run_log.jsonl" if path is None else path
    rows = read_jsonl(target) if target.exists() else []
    failures = []
    previous = None
    for index, row in enumerate(rows):
        payload = dict(row)
        recorded = payload.pop("record_sha256", None)
        expected = sha256_json(payload)
        if row.get("sequence") != index:
            failures.append(
                {
                    "row": index,
                    "field": "sequence",
                    "expected": index,
                    "actual": row.get("sequence"),
                }
            )
        if row.get("study_id") != STUDY_ID:
            failures.append(
                {
                    "row": index,
                    "field": "study_id",
                    "expected": STUDY_ID,
                    "actual": row.get("study_id"),
                }
            )
        if row.get("previous_record_sha256") != previous:
            failures.append(
                {
                    "row": index,
                    "field": "previous_record_sha256",
                    "expected": previous,
                    "actual": row.get("previous_record_sha256"),
                }
            )
        if recorded != expected:
            failures.append(
                {
                    "row": index,
                    "field": "record_sha256",
                    "expected": expected,
                    "actual": recorded,
                }
            )
        previous = recorded
    return {
        "pass": not failures,
        "records": len(rows),
        "head_record_sha256": previous,
        "failures": failures,
    }


def append_log(event: str, **payload: Any) -> None:
    path = OUT / "run_log.jsonl"
    chain = verify_log_chain(path)
    if not chain["pass"]:
        raise RuntimeError(f"run-log hash chain is invalid: {chain['failures']}")
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
