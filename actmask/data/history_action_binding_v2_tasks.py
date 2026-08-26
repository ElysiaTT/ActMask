"""State-only GPU-PhysX tasks for History--Action Binding Audit v2.

The public command is two-dimensional while the response body moves on a
one-dimensional PhysX rail.  A hidden phase maps command angle to signed force
through a second angular harmonic.  The phase and continuous response
parameters are oracle-only; they never appear in fair observations.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import sapien
import torch

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs import Pose


def _build_actor(scene, name: str, *, dynamic: bool, collision: bool):
    builder = scene.create_actor_builder()
    if collision:
        builder.add_box_collision(half_size=(0.025, 0.018, 0.018))
    builder.set_initial_pose(sapien.Pose())
    actor = builder.build_dynamic(name) if dynamic else builder.build_kinematic(name)
    if dynamic:
        actor.set_locked_motion_axes([False, True, True, True, True, True])
        actor.set_mass(1.0)
        actor.set_linear_damping(1.25)
        actor.set_angular_damping(2.0)
    return actor


class _BindingV2BaseEnv(BaseEnv):
    SUPPORTED_OBS_MODES = ("state", "state_dict", "none")
    SUPPORTED_REWARD_MODES = ("sparse", "none")
    force_scale = 4.0
    goal_x = 0.030
    success_radius = 0.005
    task_kind = "base"

    def __init__(self, *args, robot_uids: str = "none", **kwargs):
        kwargs.setdefault("render_backend", "none")
        self.goal_pos: torch.Tensor | None = None
        self.command: torch.Tensor | None = None
        self.public_rotation: torch.Tensor | None = None
        self.latent_phi: torch.Tensor | None = None
        self.latent_alpha: torch.Tensor | None = None
        self.latent_beta: torch.Tensor | None = None
        super().__init__(*args, robot_uids=robot_uids, **kwargs)
        self.single_action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self._orig_single_action_space = self.single_action_space
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(self.num_envs, 2), dtype=np.float32)

    def _load_scene(self, options: dict):
        del options
        self.response = _build_actor(self.scene, "binding_v2_response", dynamic=True, collision=True)
        self.actuator = _build_actor(self.scene, "binding_v2_actuator", dynamic=False, collision=False)
        self.goal_marker = _build_actor(self.scene, "binding_v2_goal", dynamic=False, collision=False)

    def _ensure_buffers(self):
        if self.goal_pos is None or self.goal_pos.shape[0] != self.num_envs:
            self.goal_pos = torch.zeros((self.num_envs, 3), device=self.device)
            self.command = torch.zeros((self.num_envs, 2), device=self.device)
            self.public_rotation = torch.zeros((self.num_envs,), device=self.device)
            self.latent_phi = torch.zeros((self.num_envs,), device=self.device)
            self.latent_alpha = torch.ones((self.num_envs,), device=self.device)
            self.latent_beta = torch.zeros((self.num_envs,), device=self.device)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        del options
        self._ensure_buffers()
        count = len(env_idx)
        zeros = torch.zeros((count, 3), device=self.device)
        response_pos = zeros.clone(); response_pos[:, 2] = 0.10
        actuator_pos = zeros.clone(); actuator_pos[:, 1] = -0.10; actuator_pos[:, 2] = 0.10
        goal_pos = zeros.clone(); goal_pos[:, 0] = self.goal_x; goal_pos[:, 2] = 0.10
        self.response.set_pose(Pose.create_from_pq(p=response_pos))
        self.response.set_linear_velocity(zeros)
        self.response.set_angular_velocity(zeros)
        self.actuator.set_pose(Pose.create_from_pq(p=actuator_pos))
        self.goal_marker.set_pose(Pose.create_from_pq(p=goal_pos))
        self.goal_pos[env_idx] = goal_pos
        self.command[env_idx] = 0.0
        self.public_rotation[env_idx] = 0.0
        self.latent_phi[env_idx] = 0.0
        self.latent_alpha[env_idx] = 1.0
        self.latent_beta[env_idx] = 0.0

    def _get_obs_agent(self):
        return dict()

    def _get_obs_extra(self, info: dict):
        del info
        return dict(
            response_pose=self.response.pose.raw_pose,
            response_linear_velocity=self.response.get_linear_velocity(),
            actuator_pose=self.actuator.pose.raw_pose,
            goal_pos=self.goal_pos,
            public_frame=torch.stack((torch.cos(self.public_rotation), torch.sin(self.public_rotation)), dim=1),
        )

    def public_state(self) -> torch.Tensor:
        return torch.stack(
            (
                self.response.pose.p[:, 0],
                self.response.get_linear_velocity()[:, 0],
                self.goal_pos[:, 0],
                torch.cos(self.public_rotation),
                torch.sin(self.public_rotation),
            ), dim=1,
        )

    def get_state_dict(self):
        return dict(
            response=self.response.get_state(), actuator=self.actuator.get_state(),
            goal_marker=self.goal_marker.get_state(), goal_pos=self.goal_pos.clone(),
            command=self.command.clone(), public_rotation=self.public_rotation.clone(),
            latent_phi=self.latent_phi.clone(), latent_alpha=self.latent_alpha.clone(),
            latent_beta=self.latent_beta.clone(),
        )

    def set_state_dict(self, state: dict, env_idx: torch.Tensor | None = None):
        self.response.set_state(state["response"], env_idx=env_idx)
        self.actuator.set_state(state["actuator"], env_idx=env_idx)
        self.goal_marker.set_state(state["goal_marker"], env_idx=env_idx)
        for key in ("goal_pos", "command", "public_rotation", "latent_phi", "latent_alpha", "latent_beta"):
            target = getattr(self, key)
            if env_idx is None:
                target.copy_(state[key])
            else:
                target[env_idx] = state[key]

    def set_parameters(self, *, public_rotation, phi, alpha, beta):
        for name, raw in (
            ("public_rotation", public_rotation), ("latent_phi", phi),
            ("latent_alpha", alpha), ("latent_beta", beta),
        ):
            value = torch.as_tensor(raw, dtype=torch.float32, device=self.device)
            if tuple(value.shape) != (self.num_envs,):
                raise ValueError(f"{name} must have shape {(self.num_envs,)}, got {tuple(value.shape)}")
            getattr(self, name).copy_(value)

    def physical_response(self, command: torch.Tensor) -> torch.Tensor:
        radius = torch.linalg.vector_norm(command, dim=1)
        angle = torch.atan2(command[:, 1], command[:, 0])
        radial = self.radial_response(radius)
        return radial * torch.cos(2.0 * (angle - self.latent_phi))

    def radial_response(self, radius: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def inverse_radial_response(self, magnitude: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def inverse_physical_response(self, desired: torch.Tensor) -> torch.Tensor:
        magnitude = self.inverse_radial_response(torch.abs(desired))
        angle = self.latent_phi + torch.where(desired >= 0.0, 0.0, torch.pi / 2.0)
        return torch.stack((magnitude * torch.cos(angle), magnitude * torch.sin(angle)), dim=1)

    def _before_simulation_step(self):
        force = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        force[:, 0] = self.force_scale * self.physical_response(self.command)
        self.response.apply_force(force)

    def _step_action(self, action):
        if action is None:
            action = torch.zeros((self.num_envs, 2), device=self.device)
        action = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        if action.ndim == 1:
            action = action[None].expand(self.num_envs, -1)
        if tuple(action.shape) != (self.num_envs, 2):
            raise ValueError(f"action must have shape {(self.num_envs, 2)}, got {tuple(action.shape)}")
        self.command.copy_(torch.clamp(action, -1.0, 1.0))
        actuator_pos = torch.zeros((self.num_envs, 3), device=self.device)
        actuator_pos[:, :2] = 0.10 * self.command
        actuator_pos[:, 1] -= 0.10
        actuator_pos[:, 2] = 0.10
        self.actuator.set_pose(Pose.create_from_pq(p=actuator_pos))
        if self.gpu_sim_enabled:
            self.scene._gpu_apply_all()
        self._before_control_step()
        for _ in range(self._sim_steps_per_control):
            self._before_simulation_step()
            self.scene.step()
            self._after_simulation_step()
        self._after_control_step()
        if self.gpu_sim_enabled:
            self.scene._gpu_fetch_all()
        return action

    def evaluate(self):
        distance = torch.abs(self.response.pose.p[:, 0] - self.goal_pos[:, 0])
        return dict(success=distance <= self.success_radius, target_distance=distance)

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        del obs, action
        return -info["target_distance"]


@register_env("ActMaskHistoryBindingV2Discrete-v0", max_episode_steps=4096)
class BindingV2DiscreteEnv(_BindingV2BaseEnv):
    task_kind = "task_a_discrete_higher_order"

    def radial_response(self, radius: torch.Tensor) -> torch.Tensor:
        return radius

    def inverse_radial_response(self, magnitude: torch.Tensor) -> torch.Tensor:
        return torch.clamp(magnitude, 0.0, 1.0)


@register_env("ActMaskHistoryBindingV2Continuous-v0", max_episode_steps=8192)
class BindingV2ContinuousEnv(_BindingV2BaseEnv):
    task_kind = "task_b_continuous_nonlinear"

    def radial_response(self, radius: torch.Tensor) -> torch.Tensor:
        return self.latent_alpha * radius - self.latent_beta * radius.square()

    def inverse_radial_response(self, magnitude: torch.Tensor) -> torch.Tensor:
        beta = torch.clamp(self.latent_beta, min=1.0e-6)
        discriminant = torch.clamp(self.latent_alpha.square() - 4.0 * beta * magnitude, min=0.0)
        root = (self.latent_alpha - torch.sqrt(discriminant)) / (2.0 * beta)
        linear = magnitude / torch.clamp(self.latent_alpha, min=1.0e-6)
        return torch.clamp(torch.where(self.latent_beta > 1.0e-6, root, linear), 0.0, 1.0)


TASK_IDS = {
    "task_a_discrete_higher_order": "ActMaskHistoryBindingV2Discrete-v0",
    "task_b_continuous_nonlinear": "ActMaskHistoryBindingV2Continuous-v0",
}

