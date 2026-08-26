"""Separate, renderer-qualified RGB-D counterfactual pilot generation.

This module deliberately does not read or alter frozen state-only artifacts.
It writes each world/branch RGB-D history once and candidate rows reference it.
"""

from __future__ import annotations

import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from actmask.data import maniskill_state_tasks as _registered_tasks  # noqa: F401
from actmask.data.maniskill_pilot import CandidateSpec, _candidate_action, candidate_specs
from mani_skill.utils.structs import Pose


TASKS = {
    "MovingCubeIntercept": "ActMaskMovingCubeIntercept-v1",
    "SignedMovingWindowPlacement": "ActMaskMovingContainerPlacement-v1",
    "FixedPhaseRotatingCaptureWindow": "ActMaskRotatingTargetInteraction-v1",
}
FAIR_INPUT_KEYS = ("history_ref", "timestamps", "visibility", "observation_confidence", "candidate_actions", "nominal_action_timing", "tcp_state")
FORBIDDEN_FAIR_KEYS = {"success", "object_id", "segmentation_id", "future_state", "branch", "mechanism", "world_id", "candidate_id"}


def load_fair_visual_inputs(output_dir: str | Path, task_name: str) -> dict[str, np.ndarray]:
    """Resolve shared histories while exposing no IDs, labels, or segmentation.

    ``history_ref`` is an on-disk storage index only and is deliberately not
    returned to a fair baseline. Depth is converted from millimetres to metres.
    """
    root = Path(output_dir)
    histories = np.load(root / f"{task_name}_histories.npz")
    rows = [json.loads(line) for line in (root / f"{task_name}_candidates.jsonl").read_text().splitlines()]
    refs = np.asarray([row["history_ref"] for row in rows], dtype=np.int64)
    return {
        "rgbd_history": np.concatenate((histories["rgb"][refs].astype(np.float32) / 255.0, histories["depth_mm"][refs, ..., None].astype(np.float32) / 1000.0), axis=-1),
        "timestamps": np.tile(np.linspace(-0.25, 0.0, histories["rgb"].shape[1], dtype=np.float32), (len(rows), 1)),
        "visibility": np.ones((len(rows), histories["rgb"].shape[1], 1), dtype=np.float32),
        "observation_confidence": np.ones((len(rows), histories["rgb"].shape[1], 1), dtype=np.float32),
        "candidate_actions": np.asarray([row["candidate_actions"] for row in rows], dtype=np.float32),
        "nominal_action_timing": np.tile(np.arange(len(rows[0]["candidate_actions"]), dtype=np.float32) / 20.0, (len(rows), 1)),
        "tcp_state": np.asarray([row["tcp_state"] for row in rows], dtype=np.float32),
    }


def _clone(state: dict) -> dict:
    return {key: value.clone() for key, value in state.items()}


def _frame(observation: dict) -> tuple[np.ndarray, np.ndarray]:
    sensor = observation["sensor_data"]["actmask_fixed_camera"]
    rgb = sensor["Color"][0, ..., :3].detach().cpu().numpy().astype(np.uint8)
    # ManiSkill PositionSegmentation encodes camera-space depth in millimetres
    # in its Z channel. Segmentation is intentionally discarded here.
    depth_mm = np.clip(-sensor["PositionSegmentation"][0, ..., 2].detach().cpu().numpy(), 0, 5000).astype(np.uint16)
    return rgb, depth_mm


def _history(env, seed: int, branch: int, frames: int) -> tuple[np.ndarray, np.ndarray, dict]:
    observation, _ = env.reset(seed=[seed])
    base = env.unwrapped
    reference = None
    if branch:
        # The paired branch shares the exact decision state with the clean
        # branch. Its earlier frames are perturbed, then this state is restored
        # before the final/current observation is serialized.
        zero = torch.zeros((1, 3), dtype=torch.float32, device=base.device)
        for _ in range(frames - 1):
            observation, _, _, _, _ = env.step(zero)
        reference = _clone(base.get_state_dict())
        observation, _ = env.reset(seed=[seed])
    values = [_frame(observation)]
    zero = torch.zeros((1, 3), dtype=torch.float32, device=base.device)
    for step in range(frames - 1):
        if branch:
            shifted = base.payload.pose.p.clone()
            shifted[:, 1] += 0.04 * float(frames - step - 1)
            base.payload.set_pose(Pose.create_from_pq(p=shifted))
            base.payload.set_linear_velocity(torch.zeros_like(shifted))
            base.payload.set_angular_velocity(torch.zeros_like(shifted))
            base.scene._gpu_apply_all()
            base.scene._gpu_fetch_all()
        observation, _, _, _, _ = env.step(zero)
        values.append(_frame(observation))
    decision = _clone(base.get_state_dict())
    if reference is not None:
        base.set_state_dict(reference)
        base.scene._gpu_apply_all(); base.scene._gpu_fetch_all()
        decision = reference
        values[-1] = _frame(base.get_obs())
    return np.stack([item[0] for item in values]), np.stack([item[1] for item in values]), decision


def _candidate_rollout(env, decision: dict, task_name: str, spec: CandidateSpec, slot: int, branch: int, horizon: int) -> tuple[np.ndarray, bool, np.ndarray]:
    base = env.unwrapped
    base.set_state_dict(decision)
    base.scene._gpu_apply_all(); base.scene._gpu_fetch_all()
    base._elapsed_steps.zero_()
    base.set_hidden_future_mode(float(branch))
    context = {
        "initial_goal": base.goal_pos.clone(), "initial_proxy": base.proxy.pose.p.clone(),
        "initial_payload": base.payload.pose.p.clone(), "planned_proxy": base.proxy.pose.p.clone(),
        "world_phase": torch.zeros(1, device=base.device),
    }
    if hasattr(base, "container"):
        context["initial_container"] = base.container.pose.p.clone()
    actions = []
    info = None
    for step in range(horizon):
        action = _candidate_action(base, task_name, spec, step, horizon, "matched_counterfactual", context, True)
        # Ten distinct candidate templates without using branch/label state.
        bias = torch.tensor([0.0, ((slot % 5) - 2) * 0.035, ((slot // 5) - 0.5) * 0.03], device=base.device)
        action = torch.clamp(action + bias, -1.0, 1.0)
        _, _, _, _, info = env.step(action)
        actions.append(action.detach().cpu().numpy()[0].astype(np.float32))
    assert info is not None
    success = bool(info["success"][0].detach().cpu())
    tcp = decision["tcp_proxy"][0, :3].detach().cpu().numpy().astype(np.float32)
    return np.stack(actions), success, tcp


def generate(output_dir: str | Path, *, worlds_per_task: int = 128, history_frames: int = 6, horizon: int = 12, seed: int = 4400) -> dict:
    """Generate the protocol-scale pilot, or a smaller deterministic test pilot."""
    if worlds_per_task < 2:
        raise ValueError("worlds_per_task must be at least two")
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    summary = {"schema": "milestone4v-visual-pilot-v1", "worlds_per_task": worlds_per_task, "history_frames": history_frames, "horizon": horizon, "tasks": {}}
    for task_offset, (task_name, task_id) in enumerate(TASKS.items()):
        env = gym.make(task_id, obs_mode="sensor_data", num_envs=1, sim_backend="gpu", sensor_configs={"width": 128, "height": 128})
        rgbs, depths, rows, labels = [], [], [], []
        try:
            specs = list(candidate_specs(task_name.lower().replace("signedmovingwindowplacement", "moving_container_placement").replace("fixedphaserotatingcapturewindow", "rotating_target_interaction").replace("movingcubeintercept", "moving_cube_intercept")))
            for world in range(worlds_per_task):
                for branch in (0, 1):
                    rgb, depth, decision = _history(env, seed + task_offset * 10000 + world, branch, history_frames)
                    history_index = len(rgbs); rgbs.append(rgb); depths.append(depth)
                    for slot in range(10):
                        spec = specs[slot % len(specs)]
                        actions, success, tcp = _candidate_rollout(env, decision, task_name.lower().replace("signedmovingwindowplacement", "moving_container_placement").replace("fixedphaserotatingcapturewindow", "rotating_target_interaction").replace("movingcubeintercept", "moving_cube_intercept"), spec, slot, branch, horizon)
                        rows.append({"history_ref": history_index, "world_id": world, "candidate_slot": slot, "split": ("train" if world % 5 < 3 else "validation" if world % 5 == 3 else "test"), "pair_group": f"{task_name}:{world}:{slot}", "candidate_actions": actions.tolist(), "tcp_state": tcp.tolist()})
                        labels.append(success)
        finally:
            env.close()
        # Candidate records are storage metadata, not an observable. A fixed
        # global permutation prevents pair-member order or candidate-file
        # position from encoding the branch/success label.
        order = np.random.default_rng(5081 + task_offset).permutation(len(rows))
        rows = [rows[int(index)] for index in order]
        labels = [labels[int(index)] for index in order]
        np.savez_compressed(output / f"{task_name}_histories.npz", rgb=np.stack(rgbs), depth_mm=np.stack(depths))
        (output / f"{task_name}_candidates.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
        np.savez_compressed(output / f"{task_name}_labels.npz", success=np.asarray(labels, dtype=np.bool_))
        summary["tasks"][task_name] = {"history_count": len(rgbs), "candidate_count": len(rows), "label_mean": float(np.mean(labels)), "shared_history": True}
    (output / "visual_pilot_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary
