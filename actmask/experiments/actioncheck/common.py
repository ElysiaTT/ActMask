"""Shared constants and deterministic I/O for ActionCheck."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "outputs" / "actmask" / "actioncheck_proposal_data_v1"
PYTHON = Path("/home/tzh/conda_envs/actmask/bin/python")

LABELS = ("accept", "reject", "uncertain")
EVIDENCE_TYPES = (
    "human_demo",
    "real_execution_success",
    "real_execution_failure",
    "operator_intervention",
    "sim_execution_success",
    "sim_execution_failure",
    "expert_review",
    "rule_checker",
    "policy_timeout",
    "ambiguous",
)
REJECT_REASON_CODES = (
    "wrong_phase",
    "wrong_object_or_direction",
    "gripper_timing_error",
    "no_progress",
    "task_regression",
    "collision_or_near_collision",
    "joint_limit",
    "workspace_limit",
    "unstable_contact",
    "no_op_or_timeout",
    "human_intervention",
    "simulator_failure",
    "rule_violation",
    "ambiguous",
)
FORBIDDEN_LABEL_BASES = (
    "logged_trajectory_mismatch",
    "source_path",
    "action_distance",
    "synthetic_perturbation",
    "file_name",
    "episode_id",
)


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def action_hash(action: Sequence[Sequence[float]]) -> str:
    array = np.asarray(action, dtype=np.float32)
    shape = np.asarray(array.shape, dtype=np.int64)
    return hashlib.sha256(shape.tobytes() + array.tobytes()).hexdigest()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


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


def stable_bucket(value: str, modulus: int) -> int:
    if modulus <= 0:
        raise ValueError("modulus must be positive")
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:16], 16) % modulus


def finite_numeric_array(value: Any, *, dimensions: int | None = None) -> bool:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(
        (dimensions is None or array.ndim == dimensions)
        and array.size >= 0
        and np.isfinite(array).all()
    )


def actioncheck_schema() -> dict[str, Any]:
    """Return the frozen canonical schema; extra provenance fields are allowed."""
    candidate = {
        "type": "object",
        "required": [
            "candidate_id",
            "action_chunk",
            "action_horizon",
            "action_dim",
            "source",
            "source_policy",
            "sampling_metadata",
            "label",
            "label_source",
            "evidence_type",
            "reason_codes",
            "hash",
        ],
        "properties": {
            "candidate_id": {"type": "string", "minLength": 1},
            "action_chunk": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "array",
                    "minItems": 1,
                    "items": {"type": "number"},
                },
            },
            "action_horizon": {"type": "integer", "minimum": 1},
            "action_dim": {"type": "integer", "minimum": 1},
            "source": {"type": "string", "minLength": 1},
            "source_policy": {"type": "string", "minLength": 1},
            "sampling_metadata": {"type": "object"},
            "label": {"enum": list(LABELS)},
            "label_source": {"type": "string", "minLength": 1},
            "evidence_type": {"enum": list(EVIDENCE_TYPES)},
            "reason_codes": {
                "type": "array",
                "items": {"enum": list(REJECT_REASON_CODES)},
                "uniqueItems": True,
            },
            "execution_result": {"type": ["object", "null"]},
            "sim_result": {"type": ["object", "null"]},
            "expert_review": {"type": ["object", "null"]},
            "rule_check": {"type": ["object", "null"]},
            "hash": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        },
        "additionalProperties": True,
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "actioncheck_schema_v1.json",
        "title": "ActionCheck context-group schema v1",
        "type": "object",
        "required": [
            "context_id",
            "task_id",
            "task_instruction",
            "episode_id",
            "timestamp",
            "history",
            "candidates",
            "split_group",
            "provenance",
        ],
        "properties": {
            "context_id": {"type": "string", "minLength": 1},
            "task_id": {"type": "string", "minLength": 1},
            "task_instruction": {"type": "string"},
            "episode_id": {"type": "string", "minLength": 1},
            "timestamp": {"type": "number"},
            "history": {
                "type": "object",
                "required": [
                    "rgb_refs",
                    "state",
                    "gripper",
                    "history_length",
                    "frequency_hz",
                ],
                "properties": {
                    "rgb_refs": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "state": {"type": "array"},
                    "gripper": {"type": "array"},
                    "action_history": {"type": ["array", "null"]},
                    "visual_features": {"type": "array"},
                    "history_length": {"type": "integer", "minimum": 1},
                    "frequency_hz": {"type": "number", "exclusiveMinimum": 0},
                },
                "additionalProperties": True,
            },
            "candidates": {
                "type": "array",
                "minItems": 1,
                "items": candidate,
            },
            "split_group": {"type": "object"},
            "provenance": {"type": "object"},
        },
        "additionalProperties": True,
    }


def label_taxonomy() -> dict[str, Any]:
    return {
        "schema": "actioncheck-label-taxonomy-v1",
        "labels": {
            "accept": {
                "primary_training": True,
                "meaning": "acceptable under the current context according to recorded evidence",
            },
            "reject": {
                "primary_training": True,
                "meaning": "unacceptable under the current context according to recorded evidence",
            },
            "uncertain": {
                "primary_training": False,
                "meaning": "evidence is ambiguous, incomplete, or conflicting",
            },
        },
        "allowed_evidence_types": list(EVIDENCE_TYPES),
        "claim_boundary": "compatibility, not guaranteed safety or task success",
    }


def reason_taxonomy() -> dict[str, Any]:
    descriptions = {
        "wrong_phase": "proposal is incompatible with the demonstrated or executed phase",
        "wrong_object_or_direction": "proposal targets the wrong object or direction",
        "gripper_timing_error": "open/close timing is incompatible",
        "no_progress": "execution makes no measurable task progress",
        "task_regression": "execution reverses prior progress",
        "collision_or_near_collision": "execution collides or violates a frozen clearance",
        "joint_limit": "joint-limit rule is violated",
        "workspace_limit": "workspace rule is violated",
        "unstable_contact": "contact is unstable under the evidence source",
        "no_op_or_timeout": "proposal is a no-op or times out",
        "human_intervention": "operator had to intervene",
        "simulator_failure": "simulator reports failure without a finer verified cause",
        "rule_violation": "an explicit frozen rule checker rejects the proposal",
        "ambiguous": "a reliable reject subtype cannot be established",
    }
    return {
        "schema": "actioncheck-reason-code-taxonomy-v1",
        "reason_codes": {
            code: {"description": descriptions[code], "audit_only": True}
            for code in REJECT_REASON_CODES
        },
        "model_input_forbidden": True,
    }


def nested_list_shape(value: Any) -> tuple[int, ...]:
    array = np.asarray(value)
    return tuple(int(size) for size in array.shape)


def safe_rate(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator / denominator) if denominator else None
