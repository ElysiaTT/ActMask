"""GPU-PhysX signed-order counterfactual generator for Milestone 3Q."""

from __future__ import annotations

import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from mani_skill.utils.structs import Pose

from actmask.data import maniskill_state_tasks as _tasks  # registers environments
from actmask.data import maniskill_3q_tasks as _window_task  # registers 3Q variant
from actmask.data.maniskill_pilot import MODEL_INPUT_KEYS, load_model_inputs
from actmask.experiments.milestone3q_order_invariant_audit import (
    multiset_features,
    order_invariant_features,
    sorted_frame_features,
    velocity_magnitude_features,
)


TASK_IDS = {
    "moving_cube_intercept": "ActMaskMovingCubeIntercept-v1",
    "signed_moving_window_placement": "ActMaskSignedMovingWindowPlacement-v1",
    "fixed_phase_rotating_capture_window": "ActMaskFixedPhaseRotatingCaptureWindow-v1",
    "crossing_obstacle_timing": "ActMaskCrossingObstacleTiming-v1",
}
REGIMES = ("original", "static_balanced", "matched_counterfactual", "signed_order")
MECHANISMS = ("loop_reversal", "opposite_velocity", "acceleration_sign", "phase_delay")


def _numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy().astype(np.float32, copy=False)


def _clone(state: dict) -> dict:
    return {key: value.clone() for key, value in state.items()}


def _loop_history(base, decision: dict, sign: int, mode: str) -> np.ndarray:
    """Return position-only closed loops; future mode is not yet selected."""
    base.set_state_dict(decision)
    base.scene._gpu_apply_all()
    base.scene._gpu_fetch_all()
    first = _numpy(base.get_obs())
    if mode == "flat":
        return np.stack((first, first, first, first), axis=1)
    a = base.payload.pose.p.clone()
    # Equilateral interior states make every per-frame displacement magnitude
    # exactly equal under [A,B,C,A] versus [A,C,B,A].
    radius = 0.075
    b = a + torch.tensor([0.0, radius, 0.0], device=base.device)
    c = a + torch.tensor([0.0649519, 0.0375, 0.0], device=base.device)
    interior = (b, c) if sign > 0 else (c, b)
    frames = [first]
    for point in interior:
        base.payload.set_pose(Pose.create_from_pq(p=point))
        base.payload.set_linear_velocity(torch.zeros_like(point))
        base.payload.set_angular_velocity(torch.zeros_like(point))
        base.scene._gpu_apply_all()
        base.scene._gpu_fetch_all()
        frames.append(_numpy(base.get_obs()))
    base.set_state_dict(decision)
    base.scene._gpu_apply_all()
    base.scene._gpu_fetch_all()
    frames.append(_numpy(base.get_obs()))
    return np.stack(frames, axis=1)


def _action_plan(base, task: str, mechanism: str, horizon: int, *, success_plan: bool) -> np.ndarray:
    proxy = base.proxy.pose.p.clone()
    payload = base.payload.pose.p.clone()
    # The window mechanisms differ in *window* signed dynamics, not a
    # candidate-specific TCP delay. Keeping this zero makes all pre-registered
    # positive windows physically reachable with the common action horizon.
    delay = 0 if task == "signed_moving_window_placement" else MECHANISMS.index(mechanism)
    if task == "moving_cube_intercept":
        target = payload + torch.tensor([0.0, 0.42 if success_plan else -0.42, 0.0], device=base.device)
        acquire = 0
    elif task == "signed_moving_window_placement":
        target = base.window_base.clone() + torch.tensor([0.0, 0.24 if success_plan else -0.24, 0.04], device=base.device)
        acquire = 4
    elif task == "fixed_phase_rotating_capture_window":
        mechanism_id = MECHANISMS.index(mechanism)
        planned = proxy
        actions = []
        direction = 1.0 if success_plan else -1.0
        for step in range(horizon):
            n = float(step + 1)
            magnitude = (0.22 * n if mechanism_id in (0, 3) else (0.028 * n * n if mechanism_id == 1 else 0.32 + 0.30 * max(n - 4.0, 0.0)))
            angle = direction * magnitude
            goal = base.rotation_center + torch.stack((0.13 * torch.cos(torch.full((base.num_envs,), angle, device=base.device)), 0.13 * torch.sin(torch.full((base.num_envs,), angle, device=base.device)), torch.full((base.num_envs,), 0.05, device=base.device)), dim=1)
            action = torch.clamp((goal - planned) / base.proxy_step_size, -1.0, 1.0)
            planned = planned + action * base.proxy_step_size
            actions.append(_numpy(action))
        return np.stack(actions, axis=1)
    elif task == "rotating_target_interaction":
        radial = base.goal_pos - payload
        base_angle = torch.atan2(radial[:, 1], radial[:, 0])
        planned = proxy
        actions = []
        direction = 1.0 if success_plan else -1.0
        # Track the observable positive-sign finite-difference continuation.
        # The paired negative branch receives this identical plan, while its
        # target rotates in the opposite direction during GPU execution.
        for step in range(horizon):
            angle = base_angle + direction * 0.26 * float(step + 1)
            goal = payload + torch.stack((0.13 * torch.cos(angle), 0.13 * torch.sin(angle), torch.zeros_like(angle)), dim=1)
            action = torch.clamp((goal - planned) / base.proxy_step_size, -1.0, 1.0)
            planned = planned + action * base.proxy_step_size
            actions.append(_numpy(action))
        return np.stack(actions, axis=1)
    else:
        target = base.gate.pose.p.clone() + torch.tensor([0.0, 0.22 if success_plan else -0.22, 0.0], device=base.device)
        acquire = 0
    planned = proxy
    actions = []
    for step in range(horizon):
        if step < delay:
            action = torch.zeros_like(planned)
        else:
            goal = payload if task == "signed_moving_window_placement" and step < acquire else target
            action = torch.clamp((goal - planned) / base.proxy_step_size, -1.0, 1.0)
        planned = planned + action * base.proxy_step_size
        actions.append(_numpy(action))
    return np.stack(actions, axis=1)


def _execute(env, base, task: str, actions: np.ndarray, hidden_mode: int) -> np.ndarray:
    base.set_hidden_future_mode(float(hidden_mode))
    info = None
    completed = torch.zeros(base.num_envs, dtype=torch.bool, device=base.device)
    for step in range(actions.shape[1]):
        _, _, _, _, info = env.step(torch.from_numpy(actions[:, step]).to(base.device))
        completed = torch.logical_or(completed, info["success"].bool())
    assert info is not None
    # Task completion is monotone at the benchmark level: a container release
    # that is valid at step t remains a successful execution even if later
    # unneeded control lets the released object fall away.
    return completed.detach().cpu().numpy().astype(bool)


def _static(history: np.ndarray, actions: np.ndarray, tcp: np.ndarray) -> np.ndarray:
    return np.concatenate((history[:, -1], actions.mean(axis=1), actions.sum(axis=1), tcp), axis=1).astype(np.float32)


def _pair_audit(rows: list[dict], history: np.ndarray, actions: np.ndarray) -> dict:
    groups: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        if row["signed_group"]:
            groups.setdefault(row["signed_group"], []).append(index)
    checks = []
    for group, members in groups.items():
        if len(members) != 2:
            raise AssertionError(f"{group} has {len(members)} members")
        left, right = members
        current = float(np.abs(history[left, -1] - history[right, -1]).max())
        action = float(np.abs(actions[left] - actions[right]).max())
        unordered = float(np.abs(order_invariant_features(history[[left]]) - order_invariant_features(history[[right]])).max())
        velocity = float(np.abs(velocity_magnitude_features(history[[left]]) - velocity_magnitude_features(history[[right]])).max())
        sorted_frames = float(np.abs(sorted_frame_features(history[[left]]) - sorted_frame_features(history[[right]])).max())
        multiset = float(np.abs(multiset_features(history[[left]]) - multiset_features(history[[right]])).max())
        ordered_difference = float(np.abs(history[left] - history[right]).max())
        if max(current, action, unordered, velocity, sorted_frames, multiset) > 1.0e-6:
            raise AssertionError(f"{group} failed signed matching")
        if ordered_difference <= 1.0e-6:
            raise AssertionError(f"{group} lacks an ordered temporal difference")
        # These diagnostic-only fields are supplied by the rotating task.  They
        # are deliberately absent from the fair NPZ input schema.
        phase_values = [rows[index].get("current_phase") for index in (left, right)]
        if all(value is not None for value in phase_values) and abs(float(phase_values[0]) - float(phase_values[1])) > 1.0e-6:
            raise AssertionError(f"{group} changed current phase")
        speed_values = [rows[index].get("angular_speed_magnitude") for index in (left, right)]
        if all(value is not None for value in speed_values) and not np.allclose(speed_values[0], speed_values[1], atol=1.0e-6, rtol=0.0):
            raise AssertionError(f"{group} changed angular-speed magnitude")
        horizons = [rows[index].get("action_horizon", actions.shape[1]) for index in (left, right)]
        if horizons[0] != horizons[1] or int(horizons[0]) != actions.shape[1]:
            raise AssertionError(f"{group} changed action horizon")
        if bool(rows[left]["success"]) == bool(rows[right]["success"]):
            raise AssertionError(f"{group} did not flip simulator success")
        checks.append(dict(group=group, current=current, action=action, unordered=unordered, velocity=velocity, sorted=sorted_frames, multiset=multiset))
    return dict(
        groups=len(checks),
        tolerance=1.0e-6,
        max_error=max((max(item[key] for key in ("current", "action", "unordered", "velocity", "sorted", "multiset")) for item in checks), default=0.0),
    )


def generate(output_dir: str | Path, *, worlds_per_regime: int = 250, horizon: int = 20, seed: int = 3907, task_names: tuple[str, ...] | None = None) -> dict:
    """Generate four regimes; a 250-world regime gives 1,000 worlds/task."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary = dict(worlds_per_regime=worlds_per_regime, worlds_per_task=4 * worlds_per_regime, candidates_per_world=20, horizon=horizon, tasks={})
    selected = tuple(TASK_IDS) if task_names is None else tuple(task_names)
    if not selected or any(task not in TASK_IDS for task in selected):
        raise ValueError("task_names must be non-empty keys from TASK_IDS")
    for task_index, task in enumerate(selected):
        task_id = TASK_IDS[task]
        histories: list[np.ndarray] = []
        plans: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        tcp_values: list[np.ndarray] = []
        rows: list[dict] = []
        task_horizon = 13 if task == "fixed_phase_rotating_capture_window" else horizon
        env = gym.make(task_id, obs_mode="state", num_envs=worlds_per_regime, sim_backend="gpu", render_backend="none")
        try:
            for regime_index, regime in enumerate(REGIMES):
                world_ids = np.arange(regime_index * worlds_per_regime, (regime_index + 1) * worlds_per_regime)
                seeds = (seed + task_index * 100_000 + world_ids).tolist()
                for candidate in range(20):
                    mechanism = MECHANISMS[(candidate // 4) % len(MECHANISMS)]
                    pair_member = candidate % 2
                    signed = regime == "signed_order"
                    # The two pair members share the same candidate action; only
                    # observable loop order and oracle future sign differ.
                    success_plan = True if regime != "original" else pair_member == 0
                    hidden_mode = 1 if (not signed or pair_member == 0) else -1
                    history_kind = "loop" if signed else ("flat" if regime == "static_balanced" else "loop")
                    history_sign = 1 if (not signed or pair_member == 0) else -1
                    env.reset(seed=seeds)
                    base = env.unwrapped
                    if task == "signed_moving_window_placement":
                        base.set_window_mechanism(MECHANISMS.index(mechanism))
                    if task == "fixed_phase_rotating_capture_window":
                        base.set_rotation_mechanism(MECHANISMS.index(mechanism))
                    decision = _clone(base.get_state_dict())
                    history = _loop_history(base, decision, history_sign, history_kind)
                    base.set_state_dict(decision)
                    base.scene._gpu_apply_all()
                    base.scene._gpu_fetch_all()
                    # Pair actions must be identical. Candidate // 2 supplies
                    # one frozen template for both signed branches.
                    plan = _action_plan(base, task, mechanism, task_horizon, success_plan=(True if regime != "original" else success_plan))
                    success = _execute(env, base, task, plan, hidden_mode)
                    histories.append(history)
                    plans.append(plan)
                    labels.append(success)
                    tcp_values.append(_numpy(base.proxy.pose.p))
                    for local, world in enumerate(world_ids):
                        group = f"{task}:{regime}:{mechanism}:{int(world)}:{candidate // 2}" if signed else None
                        row = dict(task=task, regime=regime, mechanism=mechanism, world_id=int(world), candidate_id=candidate, seed=int(seeds[local]), signed_group=group, success=bool(success[local]), split=("train" if int(world) % 5 < 3 else ("val" if int(world) % 5 == 3 else "test")), action_horizon=task_horizon)
                        if task == "fixed_phase_rotating_capture_window":
                            n = np.arange(1, task_horizon + 1, dtype=np.float32)
                            mechanism_id = MECHANISMS.index(mechanism)
                            angles = (0.22 * n if mechanism_id in (0, 3) else (0.028 * np.square(n) if mechanism_id == 1 else 0.32 + 0.30 * np.maximum(n - 4.0, 0.0)))
                            row.update(current_phase=float(base.fixed_initial_phase), angular_speed_magnitude=np.diff(np.concatenate((np.zeros(1, dtype=np.float32), angles))).astype(float).tolist())
                        rows.append(row)
        finally:
            env.close()
        history = np.concatenate(histories, axis=0)
        action = np.concatenate(plans, axis=0)
        label = np.concatenate(labels, axis=0)
        tcp = np.concatenate(tcp_values, axis=0)
        timestamps = np.tile(np.linspace(-0.15, 0.0, history.shape[1], dtype=np.float32), (len(history), 1))
        np.savez_compressed(output / f"{task}_model_inputs.npz", history=history, timestamps=timestamps, visibility=np.ones((len(history), history.shape[1], 1), dtype=np.float32), observation_confidence=np.ones((len(history), history.shape[1], 1), dtype=np.float32), candidate_actions=action, nominal_action_timing=np.tile(np.arange(task_horizon, dtype=np.float32) / 20.0, (len(history), 1)), tcp_state=tcp)
        np.savez_compressed(output / f"{task}_labels.npz", success=label, static_features=_static(history, action, tcp))
        with (output / f"{task}_metadata.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        load_model_inputs(output / f"{task}_model_inputs.npz")
        summary["tasks"][task] = dict(examples=int(len(label)), success_rate=float(label.mean()), action_horizon=task_horizon, signed_audit=_pair_audit(rows, history, action))
    (output / "signed_generation_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary
