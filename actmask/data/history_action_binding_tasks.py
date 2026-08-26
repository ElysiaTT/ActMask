"""State-only GPU-PhysX tasks for the History--Action Binding MVP.

The response body is a dynamic PhysX actor constrained to a one-dimensional
rail.  Commands are converted to external forces by an oracle-only latent
response law.  Public observations never contain the latent mode/parameters.
Unlike the retired TimeArrow construction, response histories are produced by
continuous simulator steps; response poses are only written during episode
initialization or state restoration for candidate branching.
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


def _empty_actor(scene, name: str, *, dynamic: bool):
    builder = scene.create_actor_builder()
    builder.set_initial_pose(sapien.Pose())
    return builder.build_dynamic(name) if dynamic else builder.build_kinematic(name)


def _response_actor(scene, name: str):
    builder = scene.create_actor_builder()
    builder.add_box_collision(half_size=(0.025, 0.018, 0.018))
    builder.set_initial_pose(sapien.Pose())
    actor = builder.build_dynamic(name)
    # True means locked in the runtime API.  The body may translate in X only.
    actor.set_locked_motion_axes([False, True, True, True, True, True])
    actor.set_mass(1.0)
    actor.set_linear_damping(1.25)
    actor.set_angular_damping(2.0)
    return actor


class _HistoryActionBindingBaseEnv(BaseEnv):
    """A rail-constrained response system controlled by latent force mapping."""

    SUPPORTED_OBS_MODES = ("state", "state_dict", "none")
    SUPPORTED_REWARD_MODES = ("sparse", "none")
    response_mass = 1.0
    response_damping = 1.25
    force_scale = 4.0
    success_radius = 0.012
    goal_x = 0.055
    task_kind = "base"

    def __init__(self, *args, robot_uids: str = "none", **kwargs):
        kwargs.setdefault("render_backend", "none")
        self.goal_pos: torch.Tensor | None = None
        self.command: torch.Tensor | None = None
        self.latent_mode: torch.Tensor | None = None
        self.latent_alpha: torch.Tensor | None = None
        self.latent_beta: torch.Tensor | None = None
        super().__init__(*args, robot_uids=robot_uids, **kwargs)
        self.single_action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        self._orig_single_action_space = self.single_action_space
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.num_envs, 1), dtype=np.float32
        )

    def _load_scene(self, options: dict):
        del options
        self.response = _response_actor(self.scene, "binding_response")
        # These shape-free kinematic actors expose an actuator/goal state for
        # trace completeness without creating contacts or requiring rendering.
        self.actuator = _empty_actor(self.scene, "binding_actuator", dynamic=False)
        self.goal_marker = _empty_actor(self.scene, "binding_goal", dynamic=False)

    def _ensure_buffers(self):
        if self.goal_pos is None or self.goal_pos.shape[0] != self.num_envs:
            self.goal_pos = torch.zeros((self.num_envs, 3), device=self.device)
            self.command = torch.zeros((self.num_envs,), device=self.device)
            self.latent_mode = torch.ones((self.num_envs,), device=self.device)
            self.latent_alpha = torch.ones((self.num_envs,), device=self.device)
            self.latent_beta = torch.zeros((self.num_envs,), device=self.device)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        del options
        self._ensure_buffers()
        count = len(env_idx)
        zeros = torch.zeros((count, 3), device=self.device)
        response_pos = zeros.clone()
        response_pos[:, 2] = 0.10
        actuator_pos = zeros.clone()
        actuator_pos[:, 1] = -0.10
        actuator_pos[:, 2] = 0.10
        goal_pos = zeros.clone()
        goal_pos[:, 0] = self.goal_x
        goal_pos[:, 2] = 0.10
        self.response.set_pose(Pose.create_from_pq(p=response_pos))
        self.response.set_linear_velocity(zeros)
        self.response.set_angular_velocity(zeros)
        self.actuator.set_pose(Pose.create_from_pq(p=actuator_pos))
        self.goal_marker.set_pose(Pose.create_from_pq(p=goal_pos))
        self.goal_pos[env_idx] = goal_pos
        self.command[env_idx] = 0.0
        self.latent_mode[env_idx] = 1.0
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
        )

    def public_state(self) -> torch.Tensor:
        """Compact fair state: response x/v, actuator x, and goal x."""
        return torch.stack(
            (
                self.response.pose.p[:, 0],
                self.response.get_linear_velocity()[:, 0],
                self.actuator.pose.p[:, 0],
                self.goal_pos[:, 0],
            ),
            dim=1,
        )

    def get_state_dict(self):
        return dict(
            response=self.response.get_state(),
            actuator=self.actuator.get_state(),
            goal_marker=self.goal_marker.get_state(),
            goal_pos=self.goal_pos.clone(),
            command=self.command.clone(),
            latent_mode=self.latent_mode.clone(),
            latent_alpha=self.latent_alpha.clone(),
            latent_beta=self.latent_beta.clone(),
        )

    def set_state_dict(self, state: dict, env_idx: torch.Tensor | None = None):
        self.response.set_state(state["response"], env_idx=env_idx)
        self.actuator.set_state(state["actuator"], env_idx=env_idx)
        self.goal_marker.set_state(state["goal_marker"], env_idx=env_idx)
        keys = ("goal_pos", "command", "latent_mode", "latent_alpha", "latent_beta")
        for key in keys:
            target = getattr(self, key)
            if env_idx is None:
                target.copy_(state[key])
            else:
                target[env_idx] = state[key]

    def set_hidden_parameters(
        self,
        *,
        mode: torch.Tensor | np.ndarray | None = None,
        alpha: torch.Tensor | np.ndarray | None = None,
        beta: torch.Tensor | np.ndarray | None = None,
    ) -> None:
        """Set oracle-only latent parameters; they remain outside observations."""
        for name, raw in (("latent_mode", mode), ("latent_alpha", alpha), ("latent_beta", beta)):
            if raw is None:
                continue
            value = torch.as_tensor(raw, dtype=torch.float32, device=self.device)
            if tuple(value.shape) != (self.num_envs,):
                raise ValueError(f"{name} must have shape {(self.num_envs,)}, got {tuple(value.shape)}")
            getattr(self, name).copy_(value)

    def physical_response(self, command: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def inverse_physical_response(self, desired: torch.Tensor) -> torch.Tensor:
        """Oracle construction-only inverse used by the executed matching tail."""
        raise NotImplementedError

    def _before_simulation_step(self):
        force_x = self.force_scale * self.physical_response(self.command)
        force = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        force[:, 0] = force_x
        self.response.apply_force(force)

    def _step_action(self, action):
        if action is None:
            action = torch.zeros((self.num_envs, 1), device=self.device)
        action = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        if action.ndim == 1:
            action = action[:, None] if action.shape[0] == self.num_envs else action[None, :]
        if tuple(action.shape) != (self.num_envs, 1):
            raise ValueError(f"action must have shape {(self.num_envs, 1)}, got {tuple(action.shape)}")
        self.command.copy_(torch.clamp(action[:, 0], -1.0, 1.0))
        actuator_pos = torch.zeros((self.num_envs, 3), device=self.device)
        actuator_pos[:, 0] = 0.10 * self.command
        actuator_pos[:, 1] = -0.10
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


@register_env("ActMaskHistoryBindingDiscrete-v0", max_episode_steps=512)
class HistoryBindingDiscreteEnv(_HistoryActionBindingBaseEnv):
    """Hidden push/pull mode: the same command produces opposite force."""

    task_kind = "discrete_mode"

    def physical_response(self, command: torch.Tensor) -> torch.Tensor:
        return self.latent_mode * command

    def inverse_physical_response(self, desired: torch.Tensor) -> torch.Tensor:
        return torch.clamp(desired / self.latent_mode, -1.0, 1.0)


@register_env("ActMaskHistoryBindingContinuous-v0", max_episode_steps=512)
class HistoryBindingContinuousEnv(_HistoryActionBindingBaseEnv):
    """Hidden continuous response: g(u)=alpha*u-beta*u*abs(u)."""

    task_kind = "continuous_response"

    def physical_response(self, command: torch.Tensor) -> torch.Tensor:
        return self.latent_alpha * command - self.latent_beta * command * torch.abs(command)

    def inverse_physical_response(self, desired: torch.Tensor) -> torch.Tensor:
        # Monotone on the preregistered parameter/action domain.  A fixed-count
        # bisection keeps the construction deterministic and vectorized.
        low = torch.full_like(desired, -1.0)
        high = torch.full_like(desired, 1.0)
        for _ in range(28):
            middle = (low + high) * 0.5
            response = self.latent_alpha * middle - self.latent_beta * middle * torch.abs(middle)
            low = torch.where(response < desired, middle, low)
            high = torch.where(response >= desired, middle, high)
        return torch.clamp((low + high) * 0.5, -1.0, 1.0)


TASK_IDS = {
    "task_a_discrete_mode": "ActMaskHistoryBindingDiscrete-v0",
    "task_b_continuous_response": "ActMaskHistoryBindingContinuous-v0",
}
