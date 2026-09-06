"""Asset-free ManiSkill/PhysX force fixture, deliberately limited to CPU smoke.

The hidden variable is a real rigid body's center of mass, not an equation
used to synthesize outcomes. No robot, contact, sensor, or learned policy is
claimed. Snapshots are supported at control-step boundaries only; there are
no contacts, queued forces, stochastic steps, or stateful controllers.
"""
from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
import sapien
import torch
from mani_skill.envs.sapien_env import BaseEnv
from .contracts import validate_snapshot


class RodCOMEnv(BaseEnv):
    SUPPORTED_ROBOTS = ["none"]

    def __init__(self):
        super().__init__(robot_uids="none", num_envs=1, sim_backend="physx_cpu",
                         render_backend="none", obs_mode="state_dict", reward_mode="none")
        self.single_action_space = gym.spaces.Box(-1, 1, (1,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1, 1, (1, 1), dtype=np.float32)
        assert self.device.type == "cpu" and not self.gpu_sim_enabled

    def _load_scene(self, options):
        builder = self.scene.create_actor_builder()
        builder.add_box_collision(half_size=[0.18, 0.025, 0.025])
        builder.set_mass_and_inertia(0.5, sapien.Pose(), [0.001, 0.008, 0.008])
        builder.initial_pose = sapien.Pose([0, 0, 0.5])
        self.rod = builder.build(name="rod")
        self.body = self.rod._bodies[0]

    def _initialize_episode(self, env_idx, options):
        self.set_anchor(0.06)

    def set_anchor(self, com: float):
        if not np.isfinite(com) or abs(com) > 0.1:
            raise ValueError("COM must be finite and inside the rod")
        self.body.cmass_local_pose = sapien.Pose([float(com), 0, 0])
        self.rod.set_state(torch.tensor([[0., 0., .5, 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]))
        self._elapsed_steps.zero_()

    def _get_obs_agent(self):
        return {}

    def _get_obs_extra(self, info):
        # Never expose mass properties, native get_state_dict, or branch IDs.
        return {"body": self.rod.get_state().clone()}

    def evaluate(self):
        return {}

    def _step_action(self, action):
        value = np.asarray(action, dtype=np.float64)
        if value.size != 1 or not np.isfinite(value).all() or np.max(np.abs(value)) > 1:
            raise ValueError("action must be one finite scalar in [-1,1]")
        local = sapien.Pose([float(value.reshape(-1)[0]) * .15, 0, 0])
        for _ in range(self._sim_steps_per_control):
            point = (self.body.entity.pose * local).p
            self.body.add_force_at_point(np.array([0, 0, 6.], np.float32), point)
            self.scene.step()
        return action

    def get_state_dict(self):
        return {"body": self.rod.get_state().clone(),
                "com": torch.tensor(self.body.cmass_local_pose.p.copy()),
                "mass": torch.tensor([self.body.mass]),
                "inertia": torch.tensor(self.body.inertia.copy()),
                "elapsed": self._elapsed_steps.clone()}

    def set_state_dict(self, state, env_idx=None):
        self.body.mass = float(np.asarray(state["mass"]).reshape(-1)[0])
        self.body.inertia = np.asarray(state["inertia"], dtype=np.float32)
        self.body.cmass_local_pose = sapien.Pose(np.asarray(state["com"], dtype=np.float32))
        self.rod.set_state(torch.as_tensor(state["body"], dtype=torch.float32))
        self._elapsed_steps[:] = torch.as_tensor(state["elapsed"], dtype=torch.int32)

    def snapshot(self):
        return {"version": np.asarray(1, dtype=np.int64),
                **{key: value.cpu().numpy().copy() for key, value in self.get_state_dict().items()}}

    def restore(self, state):
        validate_snapshot(state)
        self.set_state_dict(state)

    def rollout(self, snapshot, action: float, steps: int = 8):
        if type(steps) is not int or not 1 <= steps <= 8:
            raise ValueError("rollout steps must be an integer in [1,8]")
        self.restore(snapshot)
        trajectory = []
        for _ in range(steps):
            self.step(np.asarray([action], dtype=np.float32))
            trajectory.append(self.rod.get_state().cpu().numpy()[0].copy())
        trajectory = np.asarray(trajectory)
        q = trajectory[-1, 3:7]
        tilt = np.arccos(np.clip(1 - 2 * (q[1] ** 2 + q[2] ** 2), -1, 1))
        effect = np.asarray([trajectory[-1, 2] - snapshot["body"][0, 2], tilt])
        return effect, trajectory


def save_snapshot(path: Path, state):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **state)


def load_snapshot(path: Path):
    with np.load(path, allow_pickle=False) as raw:
        return {key: raw[key].copy() for key in raw.files}
