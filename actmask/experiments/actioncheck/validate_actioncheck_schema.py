"""Validate canonical ActionCheck context-group objects."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .common import (
    EVIDENCE_TYPES,
    LABELS,
    REJECT_REASON_CODES,
    action_hash,
    finite_numeric_array,
    read_jsonl,
)


TOP_LEVEL = {
    "context_id": str,
    "task_id": str,
    "task_instruction": str,
    "episode_id": str,
    "timestamp": (int, float),
    "history": dict,
    "candidates": list,
    "split_group": dict,
    "provenance": dict,
}
HISTORY_FIELDS = {
    "rgb_refs": list,
    "state": list,
    "gripper": list,
    "history_length": int,
    "frequency_hz": (int, float),
}
CANDIDATE_FIELDS = {
    "candidate_id": str,
    "action_chunk": list,
    "action_horizon": int,
    "action_dim": int,
    "source": str,
    "source_policy": str,
    "sampling_metadata": dict,
    "label": str,
    "label_source": str,
    "evidence_type": str,
    "reason_codes": list,
    "hash": str,
}


def validate_group(group: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for key, expected in TOP_LEVEL.items():
        if key not in group:
            errors.append(f"missing:{key}")
        elif not isinstance(group[key], expected):
            errors.append(f"type:{key}")
    if errors:
        return errors
    if not np.isfinite(float(group["timestamp"])):
        errors.append("timestamp:not_finite")
    history = group["history"]
    for key, expected in HISTORY_FIELDS.items():
        if key not in history:
            errors.append(f"history_missing:{key}")
        elif not isinstance(history[key], expected):
            errors.append(f"history_type:{key}")
    if any(error.startswith("history_") for error in errors):
        return errors
    if history["history_length"] < 1 or float(history["frequency_hz"]) <= 0:
        errors.append("history:invalid_length_or_frequency")
    if not all(isinstance(value, str) for value in history["rgb_refs"]):
        errors.append("history:rgb_refs")
    for name in ("state", "gripper"):
        if history[name] and not finite_numeric_array(history[name]):
            errors.append(f"history:{name}_not_finite_numeric")
    if history.get("visual_features") and not finite_numeric_array(
        history["visual_features"], dimensions=2
    ):
        errors.append("history:visual_features_not_2d_finite")
    if len(group["candidates"]) == 0:
        errors.append("candidates:empty")
    candidate_ids: set[str] = set()
    for index, candidate in enumerate(group["candidates"]):
        prefix = f"candidate[{index}]"
        for key, expected in CANDIDATE_FIELDS.items():
            if key not in candidate:
                errors.append(f"{prefix}:missing:{key}")
            elif not isinstance(candidate[key], expected):
                errors.append(f"{prefix}:type:{key}")
        if any(error.startswith(prefix) for error in errors):
            continue
        if not candidate["candidate_id"] or candidate["candidate_id"] in candidate_ids:
            errors.append(f"{prefix}:candidate_id_not_unique")
        candidate_ids.add(candidate["candidate_id"])
        if candidate["label"] not in LABELS:
            errors.append(f"{prefix}:label")
        if candidate["evidence_type"] not in EVIDENCE_TYPES:
            errors.append(f"{prefix}:evidence_type")
        if (
            len(candidate["reason_codes"]) != len(set(candidate["reason_codes"]))
            or any(code not in REJECT_REASON_CODES for code in candidate["reason_codes"])
        ):
            errors.append(f"{prefix}:reason_codes")
        if not finite_numeric_array(candidate["action_chunk"], dimensions=2):
            errors.append(f"{prefix}:action_chunk_not_2d_finite")
            continue
        array = np.asarray(candidate["action_chunk"], dtype=np.float32)
        if tuple(array.shape) != (
            candidate["action_horizon"],
            candidate["action_dim"],
        ):
            errors.append(f"{prefix}:action_shape")
        if action_hash(candidate["action_chunk"]) != candidate["hash"]:
            errors.append(f"{prefix}:action_hash")
    return errors


def validate_file(path: str | Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    failures = []
    for index, row in enumerate(rows):
        errors = validate_group(row)
        if errors:
            failures.append(
                {
                    "row": index,
                    "context_id": row.get("context_id"),
                    "errors": errors,
                }
            )
    return {
        "schema": "actioncheck-schema-validation-report-v1",
        "path": str(Path(path)),
        "context_groups": len(rows),
        "valid_context_groups": len(rows) - len(failures),
        "failures": failures,
        "pass": not failures,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    args = parser.parse_args()
    result = validate_file(args.path)
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["pass"] else 1)
