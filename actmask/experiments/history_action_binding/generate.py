"""Generate real state-only PhysX traces for the binding-audit MVP."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from actmask.data import history_action_binding_tasks as _task_registration
from actmask.data.history_action_binding_tasks import TASK_IDS
from actmask.experiments.history_action_binding.common import (
    OUTPUT_ROOT,
    PROTOCOL,
    canonical_json,
    protocol_sha256,
    sha256_bytes,
    source_manifest,
    split_for_twin,
    state_hash,
    write_json,
)


def _numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy().astype(np.float32, copy=False)


def _clone(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.clone() for key, value in state.items()}


def candidate_sampler(candidate_seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return candidates using an opaque RNG seed as the sole input."""
    rng = np.random.default_rng(int(candidate_seed))
    amplitudes = np.asarray(PROTOCOL["candidate_amplitudes"], dtype=np.float32)
    order = rng.permutation(len(amplitudes))
    amplitudes = amplitudes[order]
    profile = np.asarray(PROTOCOL["pulse_profile"], dtype=np.float32)
    actions = amplitudes[:, None, None] * profile[None, :, None]
    return actions.astype(np.float32), order.astype(np.int64)


def _latents(task: str, twins: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    mode = np.ones((twins, 2), dtype=np.float32)
    alpha = np.ones((twins, 2), dtype=np.float32)
    beta = np.zeros((twins, 2), dtype=np.float32)
    swap = rng.integers(0, 2, size=twins).astype(bool)
    if task == "task_a_discrete_mode":
        mode[:, 0], mode[:, 1] = -1.0, 1.0
    elif task == "task_b_continuous_response":
        fraction = rng.uniform(0.0, 1.0, size=twins).astype(np.float32)
        alpha[:, 0] = 0.72 + 0.10 * fraction
        beta[:, 0] = 0.30 + 0.07 * fraction
        alpha[:, 1] = 1.30 + 0.12 * fraction
        beta[:, 1] = 0.08 + 0.04 * fraction
    else:
        raise ValueError(f"unknown task: {task}")
    for index in np.flatnonzero(swap):
        mode[index] = mode[index, ::-1]
        alpha[index] = alpha[index, ::-1]
        beta[index] = beta[index, ::-1]
    return {"mode": mode, "alpha": alpha, "beta": beta, "swap": swap}


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
    return dict(
        public=public,
        response_pos=response_pos,
        response_vel=response_vel,
        actuator_pos=actuator_pos,
        target_pos=target_pos,
        distance=distance,
        contact_magnitude=contact_magnitude,
        contact=(contact_magnitude > 1.0e-7),
    )


def _full_state_frame(base) -> np.ndarray:
    """Full reproducibility state, including oracle-only latent buffers."""
    pieces = (
        _numpy(base.response.get_state()),
        _numpy(base.actuator.get_state()),
        _numpy(base.goal_marker.get_state()),
        _numpy(base.goal_pos),
        _numpy(base.command)[:, None],
        _numpy(base.latent_mode)[:, None],
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
    desired_force_response = (
        -float(PROTOCOL["matching_kp"]) * position
        - float(PROTOCOL["matching_kd"]) * velocity
    ) / float(base.force_scale)
    # Stay inside every preregistered response law's invertible range.
    desired_force_response = torch.clamp(desired_force_response, -0.55, 0.55)
    return base.inverse_physical_response(desired_force_response)


def _run_history(env, base, task: str, twins: int) -> dict[str, np.ndarray]:
    probes = np.asarray(PROTOCOL["probe_profiles"][task], dtype=np.float32)
    pulse = np.asarray(PROTOCOL["pulse_profile"], dtype=np.float32)
    branches = 2
    public_dim = int(base.public_state().shape[1])
    before = np.zeros((twins, branches, len(probes), public_dim), dtype=np.float32)
    after = np.zeros_like(before)
    delta = np.zeros_like(before)
    utility = np.zeros((twins, branches, len(probes)), dtype=np.float32)
    success = np.zeros((twins, branches, len(probes)), dtype=bool)
    trace_stores: dict[str, list[np.ndarray]] = {}
    trace_actions: list[np.ndarray] = []
    trace_phase: list[int] = []
    trace_probe: list[int] = []
    _append_frame(trace_stores, _trace_frame(base))
    for probe_index, amplitude in enumerate(probes):
        start = _numpy(base.public_state()).reshape(twins, branches, public_dim)
        before[:, :, probe_index] = start
        for profile_value in pulse:
            command = np.full((twins * branches, 1), amplitude * profile_value, dtype=np.float32)
            env.step(torch.from_numpy(command).to(base.device))
            trace_actions.append(command[:, 0].copy())
            trace_phase.append(0)
            trace_probe.append(probe_index)
            _append_frame(trace_stores, _trace_frame(base))
        endpoint = _numpy(base.public_state()).reshape(twins, branches, public_dim)
        after[:, :, probe_index] = endpoint
        delta[:, :, probe_index] = endpoint - start
        endpoint_error = np.abs(endpoint[:, :, 0] - float(PROTOCOL["goal_x"]))
        utility[:, :, probe_index] = -endpoint_error
        success[:, :, probe_index] = endpoint_error <= float(PROTOCOL["success_radius"])
        for match_step in range(int(PROTOCOL["matching_steps"])):
            if match_step == int(PROTOCOL["matching_steps"]) - 1:
                command_t = torch.zeros((twins * branches,), device=base.device)
            else:
                command_t = _matching_commands(base)
            command = _numpy(command_t)[:, None]
            env.step(command_t[:, None])
            trace_actions.append(command[:, 0].copy())
            trace_phase.append(1)
            trace_probe.append(probe_index)
            _append_frame(trace_stores, _trace_frame(base))
    traces = _stack_frames(trace_stores)
    traces["action"] = np.stack(trace_actions, axis=1).astype(np.float32)
    traces["phase"] = np.asarray(trace_phase, dtype=np.int8)
    traces["probe_index"] = np.asarray(trace_probe, dtype=np.int8)
    return dict(
        probe_observation_before=before,
        probe_action=np.broadcast_to(probes[None, None, :], (twins, branches, len(probes))).copy(),
        probe_observation_after=after,
        probe_delta=delta,
        probe_utility=utility,
        probe_success=success,
        **{f"history_trace_{key}": value.reshape(twins, branches, *value.shape[1:]) if value.ndim >= 2 and value.shape[0] == twins * branches else value for key, value in traces.items()},
    )


def _candidate_arrays(task: str, twins: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    actions = []
    orders = []
    seeds = []
    task_offset = 0 if task == "task_a_discrete_mode" else 10_000_000
    for twin in range(twins):
        candidate_seed = seed + task_offset + 100_000 + twin
        action, order = candidate_sampler(candidate_seed)
        actions.append(action)
        orders.append(order)
        seeds.append(candidate_seed)
    return np.stack(actions), np.stack(orders), np.asarray(seeds, dtype=np.int64)


def _quantize_observation(value: np.ndarray) -> np.ndarray:
    resolution = float(PROTOCOL["public_observation_resolution"])
    return (np.round(value / resolution) * resolution).astype(np.float32)


def _run_candidates(env, base, decision: dict, task: str, twins: int, seed: int) -> dict[str, np.ndarray]:
    candidate_actions, candidate_template_order, candidate_seeds = _candidate_arrays(task, twins, seed)
    branches = 2
    candidates = candidate_actions.shape[1]
    horizon = candidate_actions.shape[2]
    public_dim = int(base.public_state().shape[1])
    base.set_state_dict(decision)
    if base.gpu_sim_enabled:
        base.scene._gpu_apply_all()
        base.scene._gpu_fetch_all()
    full_state_dim = int(_full_state_frame(base).shape[1])
    decision_state_vector = _full_state_frame(base).reshape(twins, branches, full_state_dim)
    trace_shape = (twins, branches, candidates, horizon + 1)
    public = np.zeros((*trace_shape, public_dim), dtype=np.float32)
    response_pos = np.zeros((*trace_shape, 3), dtype=np.float32)
    response_vel = np.zeros((*trace_shape, 3), dtype=np.float32)
    actuator_pos = np.zeros((*trace_shape, 3), dtype=np.float32)
    target_pos = np.zeros((*trace_shape, 3), dtype=np.float32)
    distance = np.zeros(trace_shape, dtype=np.float32)
    contact_magnitude = np.zeros(trace_shape, dtype=np.float32)
    contact = np.zeros(trace_shape, dtype=bool)
    final_state_vector = np.zeros((twins, branches, candidates, full_state_dim), dtype=np.float32)
    final_hash = np.empty((twins, branches, candidates), dtype="U64")
    for candidate in range(candidates):
        base.set_state_dict(decision)
        base._elapsed_steps.zero_()
        if base.gpu_sim_enabled:
            base.scene._gpu_apply_all()
            base.scene._gpu_fetch_all()
        for step in range(horizon + 1):
            frame = _trace_frame(base)
            shaped = {key: value.reshape(twins, branches, *value.shape[1:]) for key, value in frame.items()}
            public[:, :, candidate, step] = shaped["public"]
            response_pos[:, :, candidate, step] = shaped["response_pos"]
            response_vel[:, :, candidate, step] = shaped["response_vel"]
            actuator_pos[:, :, candidate, step] = shaped["actuator_pos"]
            target_pos[:, :, candidate, step] = shaped["target_pos"]
            distance[:, :, candidate, step] = shaped["distance"]
            contact_magnitude[:, :, candidate, step] = shaped["contact_magnitude"]
            contact[:, :, candidate, step] = shaped["contact"]
            if step < horizon:
                command = np.repeat(candidate_actions[:, candidate, step, :], branches, axis=0)
                env.step(torch.from_numpy(command).to(base.device))
        endpoint = _full_state_frame(base).reshape(twins, branches, full_state_dim)
        final_state_vector[:, :, candidate] = endpoint
        for twin in range(twins):
            for branch in range(branches):
                final_hash[twin, branch, candidate] = state_hash(endpoint[twin, branch])
    within = distance <= float(PROTOCOL["success_radius"])
    success = within.any(axis=3)
    first_success = np.full((twins, branches, candidates), -1, dtype=np.int16)
    for step in range(horizon + 1):
        select = within[:, :, :, step] & (first_success < 0)
        first_success[select] = step
    min_distance = distance.min(axis=3)
    final_error = distance[:, :, :, -1]
    utility = -min_distance
    decision_physical = public[:, :, 0, 0].copy()
    decision_public = _quantize_observation(decision_physical)
    decision_hash = np.empty((twins, branches), dtype="U64")
    for twin in range(twins):
        for branch in range(branches):
            decision_hash[twin, branch] = state_hash(decision_state_vector[twin, branch])
    candidate_hash = np.asarray(
        [hashlib.sha256(np.ascontiguousarray(value).view(np.uint8).tobytes()).hexdigest() for value in candidate_actions],
        dtype="U64",
    )
    return dict(
        candidate_actions=candidate_actions,
        candidate_actions_by_branch=np.repeat(candidate_actions[:, None], branches, axis=1),
        candidate_template_order=candidate_template_order,
        candidate_seed=candidate_seeds,
        candidate_hash=candidate_hash,
        decision_public_state=decision_public,
        decision_physical_state=decision_physical,
        decision_state_vector=decision_state_vector,
        decision_state_hash=decision_hash,
        candidate_trace_public=public,
        candidate_trace_response_pos=response_pos,
        candidate_trace_response_vel=response_vel,
        candidate_trace_actuator_pos=actuator_pos,
        candidate_trace_target_pos=target_pos,
        candidate_trace_distance=distance,
        candidate_trace_contact_magnitude=contact_magnitude,
        candidate_trace_contact=contact,
        first_success_step=first_success,
        min_distance=min_distance,
        final_error=final_error,
        utility=utility,
        success=success,
        final_state_hash=final_hash,
        final_state_vector=final_state_vector,
    )


def _integrity(dataset: dict[str, np.ndarray]) -> dict:
    actions = dataset["candidate_actions_by_branch"]
    candidate_max_error = float(np.max(np.abs(actions[:, 0] - actions[:, 1])))
    current = dataset["decision_public_state"]
    physical = dataset["decision_physical_state"]
    pair_current_max_error = float(np.max(np.abs(physical[:, 0] - physical[:, 1])))
    observable_pair_current_max_error = float(np.max(np.abs(current[:, 0] - current[:, 1])))
    position_error = float(np.max(np.abs(physical[:, :, 0])))
    velocity_error = float(np.max(np.abs(physical[:, :, 1])))
    recomputed_min = dataset["candidate_trace_distance"].min(axis=3)
    recomputed_success = recomputed_min <= float(PROTOCOL["success_radius"])
    utility_error = float(np.max(np.abs(dataset["utility"] + recomputed_min)))
    trace_label_match = bool(np.array_equal(recomputed_success, dataset["success"]))
    novel_disjoint = set(np.abs(PROTOCOL["candidate_amplitudes"])).isdisjoint(
        set(np.abs(np.concatenate(list(PROTOCOL["probe_profiles"].values()))))
    )
    outcome_counts = {}
    for left, right in ((0, 0), (0, 1), (1, 0), (1, 1)):
        key = f"{left}{right}"
        outcome_counts[key] = int(np.sum((dataset["success"][:, 0] == bool(left)) & (dataset["success"][:, 1] == bool(right))))
    passed = bool(
        candidate_max_error == 0.0
        and pair_current_max_error <= float(PROTOCOL["pair_current_tolerance"])
        and observable_pair_current_max_error == 0.0
        and position_error <= float(PROTOCOL["matching_position_tolerance"])
        and velocity_error <= float(PROTOCOL["matching_velocity_tolerance"])
        and utility_error <= float(PROTOCOL["trace_recompute_tolerance"])
        and trace_label_match
        and novel_disjoint
    )
    return dict(
        passed=passed,
        candidate_byte_identical=candidate_max_error == 0.0,
        candidate_max_error=candidate_max_error,
        pair_current_max_error=pair_current_max_error,
        observable_pair_current_max_error=observable_pair_current_max_error,
        max_position_error_from_match_target=position_error,
        max_velocity_error_from_match_target=velocity_error,
        trace_label_recompute_match=trace_label_match,
        trace_utility_recompute_max_error=utility_error,
        novel_candidate_magnitudes_disjoint=novel_disjoint,
        outcome_counts=outcome_counts,
    )


def _environment_record() -> dict:
    import mani_skill
    import sapien

    return dict(
        python=platform.python_version(),
        numpy=np.__version__,
        torch=torch.__version__,
        torch_cuda=torch.version.cuda,
        cuda_available=torch.cuda.is_available(),
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        sapien=sapien.__version__,
        mani_skill=getattr(mani_skill, "__version__", "3.0.1"),
        simulation="state-only GPU PhysX",
    )


def generate_task(output: Path, *, phase: str, task: str, twins: int, seed: int) -> dict:
    if task not in TASK_IDS:
        raise ValueError(f"unknown task: {task}")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    branches = 2
    env = gym.make(
        TASK_IDS[task], obs_mode="state", num_envs=twins * branches,
        sim_backend="gpu", render_backend="none",
    )
    try:
        reset_seeds = [seed + twin * 2 + branch for twin in range(twins) for branch in range(branches)]
        env.reset(seed=reset_seeds)
        base = env.unwrapped
        latents = _latents(task, twins, seed + 17)
        base.set_hidden_parameters(
            mode=torch.from_numpy(latents["mode"].reshape(-1)).to(base.device),
            alpha=torch.from_numpy(latents["alpha"].reshape(-1)).to(base.device),
            beta=torch.from_numpy(latents["beta"].reshape(-1)).to(base.device),
        )
        initial_public = _numpy(base.public_state()).reshape(twins, branches, -1)
        initial_full = _full_state_frame(base).reshape(twins, branches, -1)
        initial_hash = np.empty((twins, branches), dtype="U64")
        for twin in range(twins):
            for branch in range(branches):
                initial_hash[twin, branch] = state_hash(initial_full[twin, branch])
        history = _run_history(env, base, task, twins)
        decision = _clone(base.get_state_dict())
        candidates = _run_candidates(env, base, decision, task, twins, seed)
    finally:
        env.close()
    environment = _environment_record()
    sources = source_manifest()
    dataset = dict(
        task=np.asarray(task),
        phase=np.asarray(phase),
        twin_id=np.arange(twins, dtype=np.int32),
        split=np.asarray([split_for_twin(index) for index in range(twins)]),
        reset_seed=np.asarray(reset_seeds, dtype=np.int64).reshape(twins, branches),
        latent_mode=latents["mode"],
        latent_alpha=latents["alpha"],
        latent_beta=latents["beta"],
        latent_branch_swapped=latents["swap"],
        initial_public_state=initial_public,
        initial_state_vector=initial_full,
        initial_state_hash=initial_hash,
        protocol_sha256=np.asarray(protocol_sha256()),
        source_combined_sha256=np.asarray(sources["combined_sha256"]),
        environment_sha256=np.asarray(sha256_bytes(canonical_json(environment).encode("utf-8"))),
        **history,
        **candidates,
    )
    integrity = _integrity(dataset)
    np.savez_compressed(output / "dataset.npz", **dataset)
    with (output / "twins.jsonl").open("w") as handle:
        for twin in range(twins):
            row = dict(
                task=task,
                phase=phase,
                twin_id=twin,
                split=str(dataset["split"][twin]),
                reset_seeds=dataset["reset_seed"][twin].tolist(),
                candidate_seed=int(dataset["candidate_seed"][twin]),
                candidate_hash=str(dataset["candidate_hash"][twin]),
                protocol_sha256=str(dataset["protocol_sha256"]),
                source_combined_sha256=str(dataset["source_combined_sha256"]),
                environment_sha256=str(dataset["environment_sha256"]),
                latent_oracle=dict(
                    mode=dataset["latent_mode"][twin].tolist(),
                    alpha=dataset["latent_alpha"][twin].tolist(),
                    beta=dataset["latent_beta"][twin].tolist(),
                ),
                decision_state_hash=dataset["decision_state_hash"][twin].tolist(),
                outcome_counts={key: int(np.sum((dataset["success"][twin, 0] == bool(int(key[0]))) & (dataset["success"][twin, 1] == bool(int(key[1]))))) for key in ("00", "01", "10", "11")},
            )
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    report = dict(
        status="passed" if integrity["passed"] else "DATA_INTEGRITY_FAIL",
        phase=phase,
        task=task,
        twins=twins,
        branches=branches,
        candidates=int(dataset["candidate_actions"].shape[1]),
        candidate_executions=twins * branches * int(dataset["candidate_actions"].shape[1]),
        protocol_sha256=protocol_sha256(),
        source_combined_sha256=sources["combined_sha256"],
        environment_sha256=sha256_bytes(canonical_json(environment).encode("utf-8")),
        environment=environment,
        integrity=integrity,
        elapsed_seconds=time.perf_counter() - started,
    )
    write_json(output / "generation_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("smoke", "pilot"), required=True)
    parser.add_argument("--task", choices=tuple(TASK_IDS), required=True)
    parser.add_argument("--twins", type=int)
    parser.add_argument("--seed", type=int, default=int(PROTOCOL["seed"]))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    default_twins = int(PROTOCOL[f"{args.phase}_twins_per_task"])
    twins = default_twins if args.twins is None else args.twins
    if twins <= 0 or twins > int(PROTOCOL["pilot_twins_per_task"]):
        raise ValueError("twins must be in [1, pilot_twins_per_task]")
    output = args.output or OUTPUT_ROOT / args.phase / args.task
    report = generate_task(output, phase=args.phase, task=args.task, twins=twins, seed=args.seed)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
