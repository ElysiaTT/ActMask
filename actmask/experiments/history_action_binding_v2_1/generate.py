"""Conditional state-only GPU PhysX generator for frozen v2.1."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import platform
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from actmask.data import history_action_binding_v2_1_tasks as _task_registration
from actmask.data.history_action_binding_v2_1_tasks import TASK_IDS
from actmask.experiments.history_action_binding_v2_1.audit import slot_balance_table
from actmask.experiments.history_action_binding_v2_1.common import (
    OUTPUT_ROOT,
    PROTOCOL,
    TASKS,
    canonical_json,
    protocol_sha256,
    quantize,
    scientific_core_manifest,
    sha256_bytes,
    sha256_file,
    split_for_twin,
    state_hash,
    write_json,
)
from actmask.experiments.history_action_binding_v2_1.symbolic import (
    candidate_arrays,
    ideal_displacement,
    probe_action_set,
    world_parameters,
)


class GenerationBlocked(RuntimeError):
    """Raised when a preregistered predecessor gate is not satisfied."""


def _prepare_runtime_home() -> Path:
    """Route SAPIEN to the audited local PhysX payload without network writes."""
    runtime_home = OUTPUT_ROOT / "runtime_home"
    relative = Path(".sapien/physx/105.1-physx-5.3.1.patch0")
    source = PROJECT_ROOT / "audit/environment/home" / relative
    target = runtime_home / relative
    if not source.is_dir():
        raise GenerationBlocked(f"audited local PhysX payload is missing: {source}")
    target.mkdir(parents=True, exist_ok=True)
    for name in ("libPhysXGpu_64.so", "linux-so.zip"):
        link = target / name
        expected = source / name
        if link.exists() or link.is_symlink():
            if not link.is_symlink() or link.resolve() != expected.resolve():
                raise GenerationBlocked(f"unexpected runtime PhysX asset: {link}")
        else:
            link.symlink_to(expected)
    os.environ["HOME"] = str(runtime_home)
    return runtime_home


def _numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy().astype(np.float32, copy=False)


def _clone(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.clone() for key, value in state.items()}


def _trace_frame(base) -> dict[str, np.ndarray]:
    public = _numpy(base.public_state())
    response_pos = _numpy(base.response.pose.p)
    response_vel = _numpy(base.response.get_linear_velocity())
    actuator_pos = _numpy(base.actuator.pose.p)
    target_pos = _numpy(base.goal_pos)
    distance = np.abs(response_pos[:, 0] - target_pos[:, 0]).astype(np.float32)
    try:
        contact_force = _numpy(base.response.get_net_contact_forces())
        contact_magnitude = np.linalg.norm(contact_force, axis=1).astype(np.float32)
    except Exception:
        contact_magnitude = np.zeros((base.num_envs,), dtype=np.float32)
    return {
        "public": public,
        "response_pos": response_pos,
        "response_vel": response_vel,
        "actuator_pos": actuator_pos,
        "target_pos": target_pos,
        "distance": distance,
        "contact_magnitude": contact_magnitude,
        "contact": contact_magnitude > 1.0e-7,
    }


def _full_state_frame(base) -> np.ndarray:
    pieces = (
        _numpy(base.response.get_state()),
        _numpy(base.actuator.get_state()),
        _numpy(base.goal_marker.get_state()),
        _numpy(base.goal_pos),
        _numpy(base.command),
        _numpy(base.public_rotation)[:, None],
        _numpy(base.latent_phi)[:, None],
        _numpy(base.latent_alpha)[:, None],
        _numpy(base.latent_beta)[:, None],
    )
    return np.concatenate(pieces, axis=1).astype(np.float32, copy=False)


def _append_frame(store: dict[str, list[np.ndarray]], frame: dict[str, np.ndarray]) -> None:
    for key, value in frame.items():
        store.setdefault(key, []).append(value.copy())


def _stack_frames(store: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
    return {key: np.stack(value, axis=1) for key, value in store.items()}


def _matching_commands(base) -> torch.Tensor:
    position = base.response.pose.p[:, 0]
    velocity = base.response.get_linear_velocity()[:, 0]
    desired = (
        -float(PROTOCOL["matching_kp"]) * position
        - float(PROTOCOL["matching_kd"]) * velocity
    ) / float(base.force_scale)
    desired = torch.clamp(desired, -0.55, 0.55)
    return base.inverse_physical_response(desired)


def _reshape_frame(value: np.ndarray, twins: int) -> np.ndarray:
    return value.reshape(twins, 2, *value.shape[1:])


def _run_history(env, base, task: str, twins: int, public_rotation: np.ndarray) -> dict[str, np.ndarray]:
    probes = probe_action_set(task, public_rotation)
    probes_by_branch = np.repeat(probes[:, None], 2, axis=1)
    pulse = np.asarray(PROTOCOL["pulse_profile"], dtype=np.float32)
    probe_count = probes.shape[1]
    public_dim = int(base.public_state().shape[1])
    before = np.zeros((twins, 2, probe_count, public_dim), dtype=np.float32)
    after = np.zeros_like(before)
    delta_physical = np.zeros((twins, 2, probe_count), dtype=np.float32)
    utility_physical = np.zeros_like(delta_physical)
    success = np.zeros((twins, 2, probe_count), dtype=bool)
    trace_store: dict[str, list[np.ndarray]] = {}
    trace_actions: list[np.ndarray] = []
    trace_phase: list[int] = []
    trace_probe: list[int] = []
    _append_frame(trace_store, _trace_frame(base))
    for probe_index in range(probe_count):
        start = _reshape_frame(_numpy(base.public_state()), twins)
        before[:, :, probe_index] = start
        command_peak = probes_by_branch[:, :, probe_index].reshape(twins * 2, 2)
        for profile_value in pulse:
            command = command_peak * profile_value
            env.step(torch.from_numpy(command).to(base.device))
            trace_actions.append(command.copy())
            trace_phase.append(0)
            trace_probe.append(probe_index)
            _append_frame(trace_store, _trace_frame(base))
        endpoint = _reshape_frame(_numpy(base.public_state()), twins)
        after[:, :, probe_index] = endpoint
        delta_physical[:, :, probe_index] = endpoint[:, :, 0] - start[:, :, 0]
        start_distance = np.abs(float(PROTOCOL["goal_x"]) - start[:, :, 0])
        endpoint_distance = np.abs(float(PROTOCOL["goal_x"]) - endpoint[:, :, 0])
        minimum = np.minimum(start_distance, endpoint_distance)
        utility_physical[:, :, probe_index] = -minimum
        success[:, :, probe_index] = minimum <= float(PROTOCOL["success_radius"])
        for match_step in range(int(PROTOCOL["matching_steps"])):
            if match_step == int(PROTOCOL["matching_steps"]) - 1:
                command_t = torch.zeros((twins * 2, 2), device=base.device)
            else:
                command_t = _matching_commands(base)
            command = _numpy(command_t)
            env.step(command_t)
            trace_actions.append(command.copy())
            trace_phase.append(1)
            trace_probe.append(probe_index)
            _append_frame(trace_store, _trace_frame(base))
    traces = _stack_frames(trace_store)
    traces["action"] = np.stack(trace_actions, axis=1).astype(np.float32)
    traces["phase"] = np.asarray(trace_phase, dtype=np.int8)
    traces["probe_index"] = np.asarray(trace_probe, dtype=np.int16)
    shaped_traces = {
        f"history_trace_{key}": (
            value.reshape(twins, 2, *value.shape[1:])
            if value.ndim >= 2 and value.shape[0] == twins * 2 else value
        )
        for key, value in traces.items()
    }
    return {
        "probe_observation_before": before,
        "probe_actions": probes_by_branch,
        "probe_observation_after": after,
        "probe_delta_physical": delta_physical,
        "probe_delta": quantize(delta_physical, "public_probe_resolution"),
        "probe_utility_physical": utility_physical,
        "probe_utility": quantize(utility_physical, "public_probe_resolution"),
        "probe_success": success,
        **shaped_traces,
    }


def _run_candidates(
    env,
    base,
    decision: dict,
    task: str,
    twins: int,
    seed: int,
    public_rotation: np.ndarray,
) -> dict[str, np.ndarray]:
    task_offset = 0 if task == TASKS[0] else 1_000_000
    candidate_actions, template_id, candidate_seeds = candidate_arrays(
        task, public_rotation, int(seed) + task_offset + 100_000
    )
    candidates = candidate_actions.shape[1]
    horizon = candidate_actions.shape[2]
    public_dim = int(base.public_state().shape[1])
    base.set_state_dict(decision)
    if base.gpu_sim_enabled:
        base.scene._gpu_apply_all()
        base.scene._gpu_fetch_all()
    full_state_dim = int(_full_state_frame(base).shape[1])
    decision_state_vector = _full_state_frame(base).reshape(twins, 2, full_state_dim)
    trace_shape = (twins, 2, candidates, horizon + 1)
    public = np.zeros((*trace_shape, public_dim), dtype=np.float32)
    response_pos = np.zeros((*trace_shape, 3), dtype=np.float32)
    response_vel = np.zeros((*trace_shape, 3), dtype=np.float32)
    actuator_pos = np.zeros((*trace_shape, 3), dtype=np.float32)
    target_pos = np.zeros((*trace_shape, 3), dtype=np.float32)
    distance = np.zeros(trace_shape, dtype=np.float32)
    contact_magnitude = np.zeros(trace_shape, dtype=np.float32)
    contact = np.zeros(trace_shape, dtype=bool)
    final_state_vector = np.zeros((twins, 2, candidates, full_state_dim), dtype=np.float32)
    final_hash = np.empty((twins, 2, candidates), dtype="U64")
    for candidate in range(candidates):
        base.set_state_dict(decision)
        base._elapsed_steps.zero_()
        if base.gpu_sim_enabled:
            base.scene._gpu_apply_all()
            base.scene._gpu_fetch_all()
        for step in range(horizon + 1):
            frame = _trace_frame(base)
            shaped = {key: _reshape_frame(value, twins) for key, value in frame.items()}
            public[:, :, candidate, step] = shaped["public"]
            response_pos[:, :, candidate, step] = shaped["response_pos"]
            response_vel[:, :, candidate, step] = shaped["response_vel"]
            actuator_pos[:, :, candidate, step] = shaped["actuator_pos"]
            target_pos[:, :, candidate, step] = shaped["target_pos"]
            distance[:, :, candidate, step] = shaped["distance"]
            contact_magnitude[:, :, candidate, step] = shaped["contact_magnitude"]
            contact[:, :, candidate, step] = shaped["contact"]
            if step < horizon:
                command = np.repeat(candidate_actions[:, candidate, step], 2, axis=0)
                env.step(torch.from_numpy(command).to(base.device))
        endpoint = _full_state_frame(base).reshape(twins, 2, full_state_dim)
        final_state_vector[:, :, candidate] = endpoint
        for twin in range(twins):
            for branch in range(2):
                final_hash[twin, branch, candidate] = state_hash(endpoint[twin, branch])
    within = distance <= float(PROTOCOL["success_radius"])
    success = within.any(axis=3)
    first_success = np.full((twins, 2, candidates), -1, dtype=np.int16)
    for step in range(horizon + 1):
        selected = within[:, :, :, step] & (first_success < 0)
        first_success[selected] = step
    min_distance = distance.min(axis=3)
    utility = -min_distance
    decision_physical = public[:, :, 0, 0].copy()
    decision_public = quantize(decision_physical, "public_observation_resolution")
    decision_hash = np.empty((twins, 2), dtype="U64")
    for twin in range(twins):
        for branch in range(2):
            decision_hash[twin, branch] = state_hash(decision_state_vector[twin, branch])
    candidate_hash = np.asarray([
        hashlib.sha256(np.ascontiguousarray(value).view(np.uint8).tobytes()).hexdigest()
        for value in candidate_actions
    ], dtype="U64")
    peak = candidate_actions[:, :, int(np.argmax(PROTOCOL["pulse_profile"]))]
    return {
        "candidate_actions": candidate_actions,
        "candidate_actions_by_branch": np.repeat(candidate_actions[:, None], 2, axis=1),
        "candidate_template_id": template_id,
        "candidate_seed": candidate_seeds,
        "candidate_hash": candidate_hash,
        "decision_public_state": decision_public,
        "decision_physical_state": decision_physical,
        "decision_state_vector": decision_state_vector,
        "decision_state_hash": decision_hash,
        "candidate_trace_public": public,
        "candidate_trace_response_pos": response_pos,
        "candidate_trace_response_vel": response_vel,
        "candidate_trace_actuator_pos": actuator_pos,
        "candidate_trace_target_pos": target_pos,
        "candidate_trace_distance": distance,
        "candidate_trace_contact_magnitude": contact_magnitude,
        "candidate_trace_contact": contact,
        "first_success_step": first_success,
        "min_distance": min_distance,
        "final_error": distance[:, :, :, -1],
        "utility": utility,
        "success": success,
        "final_state_hash": final_hash,
        "final_state_vector": final_state_vector,
        "candidate_peak_action": peak,
    }


def _integrity(dataset: dict[str, np.ndarray]) -> dict:
    candidate_error = float(np.max(np.abs(
        dataset["candidate_actions_by_branch"][:, 0]
        - dataset["candidate_actions_by_branch"][:, 1]
    )))
    current = dataset["decision_public_state"]
    physical = dataset["decision_physical_state"]
    pair_current_error = float(np.max(np.abs(physical[:, 0] - physical[:, 1])))
    observable_current_error = float(np.max(np.abs(current[:, 0] - current[:, 1])))
    position_error = float(np.max(np.abs(physical[:, :, 0])))
    velocity_error = float(np.max(np.abs(physical[:, :, 1])))
    recomputed_min = dataset["candidate_trace_distance"].min(axis=3)
    recomputed_success = recomputed_min <= float(PROTOCOL["success_radius"])
    utility_error = float(np.max(np.abs(dataset["utility"] + recomputed_min)))
    trace_label_match = bool(np.array_equal(recomputed_success, dataset["success"]))
    public_result_multiset = bool(np.array_equal(
        np.sort(dataset["probe_delta"][:, 0], axis=1),
        np.sort(dataset["probe_delta"][:, 1], axis=1),
    ))
    physical_sorted = np.sort(dataset["probe_delta_physical"], axis=2)
    physical_multiset_error = float(np.max(np.abs(physical_sorted[:, 0] - physical_sorted[:, 1])))
    public_utility_multiset = bool(np.array_equal(
        np.sort(dataset["probe_utility"][:, 0], axis=1),
        np.sort(dataset["probe_utility"][:, 1], axis=1),
    ))
    table = slot_balance_table(dataset["candidate_template_id"])
    expected = len(dataset["candidate_template_id"]) // dataset["candidate_template_id"].shape[1]
    symbolic = ideal_displacement(
        dataset["candidate_peak_action"][:, None],
        dataset["latent_phi"][:, :, None],
        dataset["latent_alpha"][:, :, None],
        dataset["latent_beta"][:, :, None],
    )
    physical_displacement = (
        dataset["candidate_trace_response_pos"][:, :, :, -1, 0]
        - dataset["candidate_trace_response_pos"][:, :, :, 0, 0]
    )
    direction_mask = np.abs(symbolic) > 1.0e-6
    direction_match = bool(np.all(np.sign(symbolic[direction_mask]) == np.sign(physical_displacement[direction_mask])))
    probe_magnitudes = set(PROTOCOL["task_a_probe_magnitudes"]) | set(PROTOCOL["task_b_probe_magnitudes"])
    candidate_magnitudes = set(PROTOCOL["candidate_magnitudes"])
    novel_magnitudes = probe_magnitudes.isdisjoint(candidate_magnitudes)
    probe_angles = set(float(value) % 360.0 for value in PROTOCOL["probe_directions_degrees"])
    if str(dataset["task"]) == TASKS[0]:
        candidate_angles = set(float(value) % 360.0 for value in PROTOCOL["task_a_candidate_directions_degrees"])
    else:
        offset = float(PROTOCOL["task_b_candidate_direction_offset_degrees"])
        candidate_angles = set((float(value) + offset) % 360.0 for value in PROTOCOL["probe_directions_degrees"])
    novel_angles = probe_angles.isdisjoint(candidate_angles)
    initial_candidate_public = dataset["candidate_trace_public"][:, :, :, 0]
    candidate_initial_exact = bool(np.array_equal(
        initial_candidate_public,
        np.repeat(initial_candidate_public[:, :, :1], initial_candidate_public.shape[2], axis=2),
    ))
    history_source_has_no_pose_edit = "set_pose" not in inspect.getsource(_run_history)
    outcome_counts = {}
    for left, right in ((0, 0), (0, 1), (1, 0), (1, 1)):
        outcome_counts[f"{left}{right}"] = int(np.sum(
            (dataset["success"][:, 0] == bool(left))
            & (dataset["success"][:, 1] == bool(right))
        ))
    checks = {
        "candidate_branch_exact": candidate_error == 0.0,
        "physical_pair_current_within_tolerance": pair_current_error <= float(PROTOCOL["physical_pair_current_tolerance"]),
        "public_pair_current_exact": observable_current_error == 0.0,
        "matched_position_within_tolerance": position_error <= float(PROTOCOL["matching_position_tolerance"]),
        "matched_velocity_within_tolerance": velocity_error <= float(PROTOCOL["matching_velocity_tolerance"]),
        "trace_utility_recomputes": utility_error <= float(PROTOCOL["trace_recompute_tolerance"]),
        "trace_success_recomputes": trace_label_match,
        "probe_action_marginal_exact": bool(np.array_equal(dataset["probe_actions"][:, 0], dataset["probe_actions"][:, 1])),
        "probe_result_public_multiset_exact": public_result_multiset,
        "probe_utility_public_multiset_exact": public_utility_multiset,
        "probe_result_physical_multiset_within_tolerance": physical_multiset_error <= float(PROTOCOL["physical_probe_multiset_tolerance"]),
        "candidate_slots_exact": bool(np.all(table == expected)),
        "candidate_magnitudes_novel": novel_magnitudes,
        "candidate_angles_novel": novel_angles,
        "candidate_branching_same_public_state": candidate_initial_exact,
        "history_rollout_source_has_no_set_pose": history_source_has_no_pose_edit,
        "symbolic_physx_direction_consistent": direction_match,
        "all_candidate_executions_saved": dataset["candidate_trace_distance"].shape[:3] == dataset["success"].shape,
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "candidate_max_error": candidate_error,
        "pair_current_max_error": pair_current_error,
        "observable_pair_current_max_error": observable_current_error,
        "max_position_error_from_match_target": position_error,
        "max_velocity_error_from_match_target": velocity_error,
        "trace_utility_recompute_max_error": utility_error,
        "physical_probe_multiset_max_error": physical_multiset_error,
        "slot_balance_table": table.tolist(),
        "slot_expected_count": int(expected),
        "outcome_counts": outcome_counts,
        "saved_candidate_rows": int(np.prod(dataset["success"].shape)),
    }


def _environment_record() -> dict:
    import mani_skill
    import sapien

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "sapien": sapien.__version__,
        "mani_skill": getattr(mani_skill, "__version__", "3.0.1"),
        "simulation": "state-only GPU PhysX",
    }


def symbolic_allows_gpu(summary: dict) -> bool:
    return bool(
        summary.get("passed")
        and all(summary.get("tasks", {}).get(task, {}).get("passed") for task in TASKS)
    )


def _predecessor_gate(phase: str) -> dict:
    symbolic_path = OUTPUT_ROOT / "official_symbolic_preflight" / "symbolic_preflight_summary.json"
    if not symbolic_path.exists():
        raise GenerationBlocked("official symbolic summary is missing")
    symbolic = json.loads(symbolic_path.read_text())
    if not symbolic_allows_gpu(symbolic):
        raise GenerationBlocked("official symbolic preflight failed")
    if phase == "pilot":
        missing = []
        failed = []
        for task in TASKS:
            path = OUTPUT_ROOT / "smoke" / task / "generation_report.json"
            if not path.exists():
                missing.append(task)
            elif json.loads(path.read_text()).get("status") != "passed":
                failed.append(task)
        if missing or failed:
            raise GenerationBlocked(f"pilot requires both smoke tasks to pass; missing={missing}, failed={failed}")
    return symbolic


def generation_refusal(phase: str, task: str, reason: str) -> dict:
    record = {
        "status": "GENERATION_BLOCKED",
        "generation_started": False,
        "gpu_environment_created": False,
        "candidate_executions": 0,
        "phase": phase,
        "task": task,
        "reason": reason,
        "learned_training_authorized": False,
    }
    write_json(OUTPUT_ROOT / "generator_refusal.json", record)
    return record


def generate_task(output: Path, *, phase: str, task: str, twins: int, seed: int) -> dict:
    _predecessor_gate(phase)
    if task not in TASK_IDS:
        raise ValueError(task)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing generation directory: {output}")
    existing_executions = 0
    for report_path in OUTPUT_ROOT.glob("*/*/generation_report.json"):
        existing_executions += int(json.loads(report_path.read_text()).get("candidate_executions", 0))
    requested_executions = twins * 2 * int(PROTOCOL["candidates_per_twin"])
    if existing_executions + requested_executions > int(PROTOCOL["candidate_execution_budget"]):
        raise GenerationBlocked("candidate execution budget would be exceeded")
    runtime_home = _prepare_runtime_home()
    output.mkdir(parents=True)
    started = time.perf_counter()
    env = gym.make(
        TASK_IDS[task], obs_mode="state", num_envs=twins * 2,
        sim_backend="gpu", render_backend="none",
    )
    try:
        reset_seeds = [seed + twin * 2 + branch for twin in range(twins) for branch in range(2)]
        env.reset(seed=reset_seeds)
        base = env.unwrapped
        task_offset = 0 if task == TASKS[0] else 1_000_000
        parameters = world_parameters(task, twins, int(seed) + task_offset)
        base.set_parameters(
            public_rotation=np.repeat(parameters["public_rotation"], 2),
            phi=parameters["latent_phi"].reshape(-1),
            alpha=parameters["latent_alpha"].reshape(-1),
            beta=parameters["latent_beta"].reshape(-1),
        )
        initial_public = _reshape_frame(_numpy(base.public_state()), twins)
        initial_full = _full_state_frame(base).reshape(twins, 2, -1)
        initial_hash = np.empty((twins, 2), dtype="U64")
        for twin in range(twins):
            for branch in range(2):
                initial_hash[twin, branch] = state_hash(initial_full[twin, branch])
        history = _run_history(env, base, task, twins, parameters["public_rotation"])
        decision = _clone(base.get_state_dict())
        candidates = _run_candidates(
            env, base, decision, task, twins, seed, parameters["public_rotation"]
        )
    finally:
        env.close()

    environment = _environment_record()
    core = scientific_core_manifest()
    dataset = {
        "task": np.asarray(task),
        "phase": np.asarray(phase),
        "twin_id": np.arange(twins, dtype=np.int32),
        "split": np.asarray([split_for_twin(index) for index in range(twins)]),
        "reset_seed": np.asarray(reset_seeds, dtype=np.int64).reshape(twins, 2),
        "public_rotation": parameters["public_rotation"],
        "latent_phi": parameters["latent_phi"],
        "latent_alpha": parameters["latent_alpha"],
        "latent_beta": parameters["latent_beta"],
        "latent_branch_swapped": parameters["latent_branch_swapped"],
        "hidden_phase_jitter": parameters["hidden_phase_jitter"],
        "initial_public_state": initial_public,
        "initial_state_vector": initial_full,
        "initial_state_hash": initial_hash,
        "protocol_sha256": np.asarray(protocol_sha256()),
        "source_combined_sha256": np.asarray(core["combined_sha256"]),
        "environment_sha256": np.asarray(sha256_bytes(canonical_json(environment).encode("utf-8"))),
        **history,
        **candidates,
    }
    integrity = _integrity(dataset)
    np.savez_compressed(output / "dataset.npz", **dataset)
    with (output / "twins.jsonl").open("w") as handle:
        for twin in range(twins):
            row = {
                "task": task,
                "phase": phase,
                "twin_id": twin,
                "split": str(dataset["split"][twin]),
                "reset_seeds": dataset["reset_seed"][twin].tolist(),
                "candidate_seed": int(dataset["candidate_seed"][twin]),
                "candidate_hash": str(dataset["candidate_hash"][twin]),
                "initial_state_hash": dataset["initial_state_hash"][twin].tolist(),
                "decision_state_hash": dataset["decision_state_hash"][twin].tolist(),
                "latent_oracle": {
                    "phi": dataset["latent_phi"][twin].tolist(),
                    "alpha": dataset["latent_alpha"][twin].tolist(),
                    "beta": dataset["latent_beta"][twin].tolist(),
                },
                "outcome_counts": {
                    key: int(np.sum(
                        (dataset["success"][twin, 0] == bool(int(key[0])))
                        & (dataset["success"][twin, 1] == bool(int(key[1])))
                    ))
                    for key in ("00", "01", "10", "11")
                },
            }
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    report = {
        "status": "passed" if integrity["passed"] else "DATA_INTEGRITY_FAIL",
        "phase": phase,
        "task": task,
        "twins": twins,
        "branches": 2,
        "candidates": int(dataset["candidate_actions"].shape[1]),
        "candidate_executions": int(np.prod(dataset["success"].shape)),
        "protocol_sha256": protocol_sha256(),
        "source_combined_sha256": core["combined_sha256"],
        "environment_sha256": sha256_bytes(canonical_json(environment).encode("utf-8")),
        "environment": environment,
        "runtime_home": str(runtime_home),
        "integrity": integrity,
        "dataset_sha256": sha256_file(output / "dataset.npz"),
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(output / "generation_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("smoke", "pilot"), required=True)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--twins", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        summary = json.loads((OUTPUT_ROOT / "freeze_record.json").read_text())
        seed = int(summary["official_seed"]) if args.seed is None else int(args.seed)
        expected_seed = int(summary["official_seed"])
        if seed != expected_seed:
            raise GenerationBlocked("GPU stages must use the frozen official seed")
        twins = int(PROTOCOL[f"{args.phase}_twins_per_task"]) if args.twins is None else args.twins
        if twins != int(PROTOCOL[f"{args.phase}_twins_per_task"]):
            raise GenerationBlocked("GPU stages must use the preregistered twin count")
        output = args.output or OUTPUT_ROOT / args.phase / args.task
        result = generate_task(output, phase=args.phase, task=args.task, twins=twins, seed=seed)
    except (GenerationBlocked, FileNotFoundError) as exc:
        result = generation_refusal(args.phase, args.task, str(exc))
        print(json.dumps(result, indent=2, sort_keys=True))
        raise SystemExit(3)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
