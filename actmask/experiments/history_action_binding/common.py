"""Frozen protocol constants and evidence helpers for the binding MVP."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = PROJECT_ROOT / "outputs/actmask/history_action_binding_mvp"
PROTECTED_GLOBS = (
    "outputs/actmask/milestone3r_nl_v2*",
    "audit",
    "paper_icra2027",
)

PROTOCOL = {
    "protocol_version": "history_action_binding_mvp_v1",
    "seed": 731_903,
    "smoke_twins_per_task": 8,
    "pilot_twins_per_task": 64,
    "candidates_per_twin": 8,
    "candidate_horizon": 4,
    "probe_horizon": 4,
    "matching_steps": 96,
    "matching_kp": 7.0,
    "matching_kd": 4.5,
    "matching_position_tolerance": 5.0e-4,
    "matching_velocity_tolerance": 1.5e-3,
    "pair_current_tolerance": 1.5e-3,
    "public_observation_resolution": 1.0e-3,
    "trace_recompute_tolerance": 1.0e-7,
    "candidate_equality_tolerance": 0.0,
    "success_radius": 0.012,
    "goal_x": 0.055,
    "probe_profiles": {
        "task_a_discrete_mode": [-0.8, -0.4, 0.4, 0.8],
        "task_b_continuous_response": [-0.8, -0.5, -0.2, 0.2, 0.5, 0.8],
    },
    "candidate_amplitudes": [-0.95, -0.65, -0.35, -0.15, 0.15, 0.35, 0.65, 0.95],
    "pulse_profile": [0.5, 1.0, 1.0, 0.5],
    "candidate_execution_budget": 5000,
    "output_byte_budget": 1_000_000_000,
    "saturation_threshold": 0.90,
    "binding_drop_threshold": 0.10,
    "pair_shuffle_max_change": 0.02,
    "novel_candidate_min_twin_accuracy": 0.70,
    "branch_predictability_max_balanced_accuracy": 0.65,
    "bootstrap_replicates": 1000,
    "bootstrap_seed": 9973,
    "split_modulus": 10,
    "split_train_values": [0, 1, 2, 3, 4, 5],
    "split_validation_values": [6, 7],
    "split_test_values": [8, 9],
}

SOURCE_FILES = (
    "actmask/data/history_action_binding_tasks.py",
    "actmask/experiments/history_action_binding/__init__.py",
    "actmask/experiments/history_action_binding/common.py",
    "actmask/experiments/history_action_binding/generate.py",
    "actmask/experiments/history_action_binding/audit.py",
    "actmask/experiments/history_action_binding/run_mvp.py",
    "tests/test_history_action_binding_mvp.py",
    "docs/history_action_binding_mvp_preregister.md",
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def protocol_sha256() -> str:
    return sha256_bytes(canonical_json(PROTOCOL).encode("utf-8"))


def source_manifest() -> dict:
    entries = {}
    for relative in SOURCE_FILES:
        path = PROJECT_ROOT / relative
        if not path.exists():
            continue
        entries[relative] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    return {"files": entries, "combined_sha256": sha256_bytes(canonical_json(entries).encode())}


def state_hash(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values, dtype=np.float32)
    return sha256_bytes(array.view(np.uint8).tobytes())


def split_for_twin(twin_id: int) -> str:
    value = twin_id % int(PROTOCOL["split_modulus"])
    if value in PROTOCOL["split_train_values"]:
        return "train"
    if value in PROTOCOL["split_validation_values"]:
        return "validation"
    return "test"


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
