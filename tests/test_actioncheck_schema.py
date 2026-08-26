"""Canonical ActionCheck schema contracts."""
from __future__ import annotations

from copy import deepcopy

from actmask.experiments.actioncheck.common import (
    EVIDENCE_TYPES,
    LABELS,
    REJECT_REASON_CODES,
    action_hash,
    actioncheck_schema,
)
from actmask.experiments.actioncheck.validate_actioncheck_schema import (
    validate_group,
)


def example_group() -> dict:
    accept = [[0.1, 0.2], [0.2, 0.3]]
    reject = [[-0.1, -0.2], [-0.2, -0.3]]
    return {
        "context_id": "context-001",
        "task_id": "task-a",
        "task_instruction": "move safely",
        "episode_id": "episode-001",
        "timestamp": 1.5,
        "history": {
            "rgb_refs": [],
            "state": [[0.0, 1.0], [0.1, 1.1]],
            "gripper": [],
            "action_history": None,
            "visual_features": [],
            "history_length": 2,
            "frequency_hz": 20.0,
        },
        "candidates": [
            {
                "candidate_id": "accept-1",
                "action_chunk": accept,
                "action_horizon": 2,
                "action_dim": 2,
                "source": "simulator_proposal",
                "source_policy": "policy-a",
                "sampling_metadata": {
                    "label_basis": "simulator_execution_outcome"
                },
                "label": "accept",
                "label_source": "simulator",
                "evidence_type": "sim_execution_success",
                "reason_codes": [],
                "sim_result": {"success": True},
                "hash": action_hash(accept),
            },
            {
                "candidate_id": "reject-1",
                "action_chunk": reject,
                "action_horizon": 2,
                "action_dim": 2,
                "source": "simulator_proposal",
                "source_policy": "policy-a",
                "sampling_metadata": {
                    "label_basis": "simulator_execution_outcome"
                },
                "label": "reject",
                "label_source": "simulator",
                "evidence_type": "sim_execution_failure",
                "reason_codes": ["simulator_failure"],
                "sim_result": {"success": False},
                "hash": action_hash(reject),
            },
        ],
        "split_group": {
            "object_composition_id": "object-a",
            "scene_id": "scene-a",
        },
        "provenance": {"source": "unit-test"},
    }


def test_taxonomies_and_json_schema_are_complete() -> None:
    schema = actioncheck_schema()
    assert schema["$id"] == "actioncheck_schema_v1.json"
    assert set(LABELS) == {"accept", "reject", "uncertain"}
    assert "sim_execution_failure" in EVIDENCE_TYPES
    assert "simulator_failure" in REJECT_REASON_CODES
    assert set(schema["required"]) == {
        "context_id",
        "task_id",
        "task_instruction",
        "episode_id",
        "timestamp",
        "history",
        "candidates",
        "split_group",
        "provenance",
    }


def test_valid_group_and_tensor_hash_pass() -> None:
    assert validate_group(example_group()) == []


def test_shape_label_reason_and_hash_failures_are_detected() -> None:
    group = deepcopy(example_group())
    group["candidates"][0]["action_horizon"] = 3
    group["candidates"][0]["hash"] = "0" * 64
    group["candidates"][1]["label"] = "bad"
    group["candidates"][1]["reason_codes"] = ["invented"]
    errors = validate_group(group)
    assert "candidate[0]:action_shape" in errors
    assert "candidate[0]:action_hash" in errors
    assert "candidate[1]:label" in errors
    assert "candidate[1]:reason_codes" in errors
