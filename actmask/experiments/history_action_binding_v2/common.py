"""Frozen protocol and evidence helpers for History--Action Binding v2."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = PROJECT_ROOT / "outputs/actmask/history_action_binding_v2"
V1_ROOT = PROJECT_ROOT / "outputs/actmask/history_action_binding_mvp"
TASKS = ("task_a_discrete_higher_order", "task_b_continuous_nonlinear")

PROTECTED_GLOBS = (
    "outputs/actmask/history_action_binding_mvp",
    "outputs/actmask/milestone3r_nl_v2*",
    "audit",
    "paper_icra2027",
)

PROTOCOL = {
    "protocol_version": "history_action_binding_v2_v1",
    "seed": 2_731_903,
    "smoke_twins_per_task": 8,
    "pilot_twins_per_task": 64,
    "candidates_per_twin": 16,
    "candidate_horizon": 4,
    "pulse_profile": [0.5, 1.0, 1.0, 0.5],
    "probe_directions_degrees": [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0],
    "candidate_direction_offset_degrees": 15.0,
    "candidate_magnitudes": [0.35, 0.65],
    "task_a_probe_magnitudes": [0.80],
    "task_b_probe_magnitudes": [0.20, 0.50, 0.80],
    "twin_phase_offset_degrees": 45.0,
    "task_b_alpha_range": [1.05, 1.25],
    "task_b_beta_range": [0.18, 0.32],
    "task_b_hidden_phase_jitter_degrees": [0.0, 45.0],
    "symbolic_displacement_gain": 0.0578,
    "goal_x": 0.030,
    "success_radius": 0.005,
    "matching_steps": 96,
    "matching_kp": 7.0,
    "matching_kd": 4.5,
    "matching_position_tolerance": 5.0e-4,
    "matching_velocity_tolerance": 1.5e-3,
    "physical_pair_current_tolerance": 1.5e-3,
    "public_observation_resolution": 1.0e-3,
    "public_probe_resolution": 1.0e-3,
    "physical_probe_multiset_tolerance": 1.5e-3,
    "trace_recompute_tolerance": 1.0e-7,
    "candidate_execution_budget": 5000,
    "output_byte_budget": 1_000_000_000,
    "symbolic_marginal_max": 0.60,
    "symbolic_arrow_max": 0.70,
    "symbolic_sysid_min": 0.90,
    "marginal_shortcut_threshold": 0.70,
    "interaction_shortcut_threshold": 0.90,
    "sysid_novel_min": 0.85,
    "binding_drop_min": 0.20,
    "pair_shuffle_max_change": 0.02,
    "invariance_tolerance": 1.0e-12,
    "candidate_top1_crossing_min": 0.90,
    "branch_predictability_max": 0.65,
    "bootstrap_replicates": 1000,
    "bootstrap_seed": 17_357,
    "split_modulus": 10,
    "split_train_values": [0, 1, 2, 3, 4, 5],
    "split_validation_values": [6, 7],
    "split_test_values": [8, 9],
}

SOURCE_FILES = (
    "actmask/data/history_action_binding_v2_tasks.py",
    "actmask/experiments/history_action_binding_v2/__init__.py",
    "actmask/experiments/history_action_binding_v2/common.py",
    "actmask/experiments/history_action_binding_v2/postmortem.py",
    "actmask/experiments/history_action_binding_v2/symbolic.py",
    "actmask/experiments/history_action_binding_v2/generate.py",
    "actmask/experiments/history_action_binding_v2/audit.py",
    "actmask/experiments/history_action_binding_v2/run_v2.py",
    "tests/test_history_action_binding_v2.py",
    "docs/history_action_binding_v2_preregister.md",
)


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def protocol_sha256() -> str:
    return sha256_bytes(canonical_json(PROTOCOL).encode("utf-8"))


def source_manifest() -> dict:
    files = {}
    for relative in SOURCE_FILES:
        path = PROJECT_ROOT / relative
        if path.exists():
            files[relative] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    return {"files": files, "combined_sha256": sha256_bytes(canonical_json(files).encode("utf-8"))}


def split_for_twin(twin_id: int) -> str:
    value = twin_id % int(PROTOCOL["split_modulus"])
    if value in PROTOCOL["split_train_values"]:
        return "train"
    if value in PROTOCOL["split_validation_values"]:
        return "validation"
    return "test"


def quantize(value: np.ndarray, resolution_key: str) -> np.ndarray:
    resolution = float(PROTOCOL[resolution_key])
    return (np.round(np.asarray(value) / resolution) * resolution).astype(np.float32)


def state_hash(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value, dtype=np.float32)
    return sha256_bytes(array.view(np.uint8).tobytes())


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

