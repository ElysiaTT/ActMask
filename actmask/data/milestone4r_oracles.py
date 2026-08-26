"""Generator-private state and segmentation diagnostics for 4R.

These files are produced separately from the fair visual bundles.  They are
strictly for the preregistered state/object-pose/segmentation oracle controls
and never enter a fair model adapter.
"""
from __future__ import annotations

import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from mani_skill.utils import sapien_utils

from actmask.data import maniskill_state_tasks as _registered  # noqa: F401
from actmask.data.milestone4r_robust_pilot import CAMERAS, TASKS, TEMPORAL_ORDER, _camera_local_distractor_offset
from actmask.data.milestone4v_visual_pilot import _frame


def _capture(env, seed: int, frames: int = 6):
    observation, _ = env.reset(seed=[seed])
    base = env.unwrapped
    target_id = int(base.payload.per_scene_id[0].detach().cpu())
    state, object_pose, depth, target_mask = [], [], [], []

    def record(value):
        _, current_depth = _frame(value)
        segmentation = value["sensor_data"]["actmask_fixed_camera"]["PositionSegmentation"][0, ..., 3].detach().cpu().numpy()
        state.append(np.concatenate((base.payload.pose.p[0].detach().cpu().numpy(), base.proxy.pose.p[0].detach().cpu().numpy(), base.goal_pos[0].detach().cpu().numpy())).astype(np.float32))
        object_pose.append(base.payload.pose.p[0].detach().cpu().numpy().astype(np.float32))
        depth.append(current_depth)
        target_mask.append((segmentation == target_id).astype(np.uint8))

    record(observation)
    zero = torch.zeros((1, 3), dtype=torch.float32, device=base.device)
    for _ in range(frames - 1):
        observation, _, _, _, _ = env.step(zero)
        record(observation)
    return np.stack(state), np.stack(object_pose), np.stack(depth), np.stack(target_mask)


def generate_oracle_diagnostics(root: str | Path, *, worlds_per_task: int = 128, seed: int = 47000) -> dict:
    """Write aligned private oracle histories without candidate rollouts."""
    root = Path(root)
    result = {"schema": "milestone4r-oracle-diagnostics-v1", "generator_private": True, "tasks": {}}
    for task_offset, (task, task_id) in enumerate(TASKS.items()):
        calibration = json.loads((root / f"{task}_camera_calibration.json").read_text())
        if len(calibration) != worlds_per_task:
            raise ValueError(f"Expected {worlds_per_task} calibration worlds for {task}, found {len(calibration)}")
        calibration_by_world = {entry["world_id"]: entry for entry in calibration}
        state, pose, depth, target_mask = [], [], [], []
        for world in range(worlds_per_task):
            camera = calibration_by_world[world]["camera"]
            env = gym.make(task_id, obs_mode="sensor_data", num_envs=1, sim_backend="gpu", robust_distractor=True, robust_distractor_camera_offset=_camera_local_distractor_offset(camera), sensor_configs={"pose": sapien_utils.look_at(eye=CAMERAS[camera], target=[0, 0, .04]), "width": 128, "height": 128})
            try:
                captured = _capture(env, seed + task_offset * 10000 + world)
                for destination, values in zip((state, pose, depth, target_mask), captured):
                    destination.extend((values, values[TEMPORAL_ORDER]))
            finally:
                env.close()
        np.savez_compressed(root / f"{task}_oracle_diagnostics.npz", state_history=np.stack(state), object_pose_history=np.stack(pose), target_depth_mm=np.stack(depth), target_mask=np.stack(target_mask))
        result["tasks"][task] = {"histories": len(state), "state_dim": int(state[0].shape[-1]), "segmentation_private": True}
    (root / "oracle_diagnostic_manifest.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


if __name__ == "__main__":
    project = Path(__file__).resolve().parents[2]
    default = project / "outputs" / "actmask" / "milestone4r_robust_visual" / "robust_probe_v1"
    print(json.dumps(generate_oracle_diagnostics(default), indent=2, sort_keys=True))
