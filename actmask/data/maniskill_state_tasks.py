"""State-only ManiSkill task families for the Milestone 3P simulator pilot.

The host used for this pilot has CUDA/PhysX support but no usable Vulkan render
ICD.  These environments therefore deliberately contain collision shapes only:
they run vectorized PhysX on CUDA and expose state observations, without
constructing render materials, cameras, or dataset assets.  The three families
use a kinematic TCP proxy as their control interface; this keeps the task
mechanism explicit while avoiding an external robot/asset dependency.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import sapien
import torch

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs import Pose


def _collision_box(scene, name: str, half_size: tuple[float, float, float], body_type: str):
    """Build a procedural rigid box with both collision and render geometry."""
    builder = scene.create_actor_builder()
    builder.add_box_collision(half_size=half_size)
    builder.add_box_visual(half_size=half_size)
    builder.set_initial_pose(sapien.Pose())
    if body_type == "dynamic":
        return builder.build_dynamic(name)
    if body_type == "kinematic":
        return builder.build_kinematic(name)
    if body_type == "static":
        return builder.build_static(name)
    raise ValueError(f"Unsupported body type: {body_type}")


def _visual_box(scene, name: str, half_size: tuple[float, float, float]):
    """Kinematic render-only distractor; it has no collision/contact effect."""
    builder = scene.create_actor_builder()
    builder.add_box_visual(half_size=half_size)
    builder.set_initial_pose(sapien.Pose())
    return builder.build_kinematic(name)


class _StateOnlyProxyEnv(BaseEnv):
    """Common GPU-PhysX base with a kinematic TCP proxy and fixed RGB-D camera."""

    SUPPORTED_OBS_MODES = ("state", "state_dict", "none", "sensor_data", "pointcloud")
    SUPPORTED_REWARD_MODES = ("sparse", "none")
    proxy_step_size = 0.04
    success_radius = 0.065

    def __init__(
        self,
        *args,
        robot_uids: str = "none",
        robust_distractor: bool = False,
        robust_distractor_camera_offset: tuple[float, float, float] | None = None,
        **kwargs,
    ):
        # State-only runs retain their original no-render path. Visual runs use
        # the separately qualified Vulkan backend and fixed sensor protocol.
        if kwargs.get("obs_mode") in {"sensor_data", "pointcloud"}:
            kwargs.setdefault("render_backend", "gpu")
        else:
            kwargs.setdefault("render_backend", "none")
        self.robust_distractor = robust_distractor
        # This optional vector is expressed in the world frame, but is derived
        # from the camera's lateral/up axes by the 4R generator.  It is a
        # render-only placement control, never an observation feature.
        self.robust_distractor_camera_offset = robust_distractor_camera_offset
        self.goal_pos: torch.Tensor | None = None
        self._task_phase: torch.Tensor | None = None
        super().__init__(*args, robot_uids=robot_uids, **kwargs)
        self.single_action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=float
        )
        self._orig_single_action_space = self.single_action_space
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.num_envs, 3), dtype=float
        )

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.55, -0.55, 0.50], target=[0.0, 0.0, 0.04])
        return [
            CameraConfig(
                "actmask_fixed_camera", pose=pose, width=128, height=128,
                fov=np.pi / 3, near=0.01, far=5.0,
            )
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=[0.65, -0.65, 0.55], target=[0.0, 0.0, 0.04])
        return CameraConfig("render_camera", pose=pose, width=512, height=512, fov=np.pi / 3)

    def _load_scene(self, options: dict):
        # No visual record/material is ever attached to any of these actors.
        # A finite static box is used instead of a PhysX plane because the GPU
        # path on this host does not retain the plane contact reliably.
        floor = self.scene.create_actor_builder()
        floor.add_box_collision(half_size=(1.0, 1.0, 0.05))
        floor.add_box_visual(half_size=(1.0, 1.0, 0.05))
        floor.set_initial_pose(sapien.Pose(p=[0, 0, -0.05]))
        floor.build_static("ground")
        self.payload = _collision_box(
            self.scene, "payload", (0.025, 0.025, 0.025), "dynamic"
        )
        self.proxy = _collision_box(
            self.scene, "tcp_proxy", (0.035, 0.035, 0.035), "kinematic"
        )
        if self.robust_distractor:
            self.distractor = _visual_box(self.scene, "visual_distractor", (0.025, 0.025, 0.025))

    def _ensure_task_buffers(self):
        if self.goal_pos is None or self.goal_pos.shape[0] != self.num_envs:
            self.goal_pos = torch.zeros((self.num_envs, 3), device=self.device)
            self._task_phase = torch.zeros(self.num_envs, device=self.device)
            self._hidden_future_mode = torch.zeros(self.num_envs, device=self.device)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        self._ensure_task_buffers()
        b = len(env_idx)
        with torch.device(self.device):
            payload_pos = torch.zeros((b, 3))
            payload_pos[:, 0] = torch.rand(b) * 0.24 - 0.12
            payload_pos[:, 1] = torch.rand(b) * 0.16 - 0.08
            payload_pos[:, 2] = 0.08
            proxy_pos = payload_pos + torch.tensor([-0.20, 0.0, 0.05])
            self.payload.set_pose(Pose.create_from_pq(p=payload_pos))
            self.payload.set_linear_velocity(torch.zeros((b, 3)))
            self.payload.set_angular_velocity(torch.zeros((b, 3)))
            self.proxy.set_pose(Pose.create_from_pq(p=proxy_pos))
            if self.robust_distractor:
                # Keep the visually similar distractor beside (not on top of)
                # the target during the preregistered occlusion interval.
                # It later approaches the target, creating association
                # ambiguity without changing any physical contact dynamics.
                distractor_pos = payload_pos + self._robust_distractor_offset()
                self.distractor.set_pose(Pose.create_from_pq(p=distractor_pos))
            self.goal_pos[env_idx] = payload_pos + torch.tensor([0.18, 0.0, 0.0])
            self._task_phase[env_idx] = 0
            self._hidden_future_mode[env_idx] = 0
            self._initialize_family(env_idx, payload_pos)

    def _initialize_family(self, env_idx: torch.Tensor, payload_pos: torch.Tensor):
        """Family-specific hidden state and motion setup."""

    def _get_obs_agent(self):
        # There is deliberately no robot asset. State observations still retain
        # the ManiSkill ``agent``/``extra`` structure for downstream adapters.
        return dict()

    def get_state_dict(self):
        """State serialization for a task with no robot controller."""
        state = dict(
            payload=self.payload.get_state(),
            tcp_proxy=self.proxy.get_state(),
            goal_pos=self.goal_pos.clone(),
            task_phase=self._task_phase.clone(),
        )
        if self.robust_distractor:
            state["visual_distractor"] = self.distractor.get_state()
        state.update(self._get_task_state())
        return state

    def _get_task_state(self) -> dict:
        return dict(hidden_future_mode=self._hidden_future_mode.clone())

    def _set_task_state(self, state: dict, env_idx: torch.Tensor | None):
        if env_idx is None:
            self._hidden_future_mode.copy_(state["hidden_future_mode"])
        else:
            self._hidden_future_mode[env_idx] = state["hidden_future_mode"]

    def set_hidden_future_mode(self, mode: torch.Tensor | float | int):
        """Set oracle-only post-decision dynamics; never part of observations."""
        mode = torch.as_tensor(mode, dtype=torch.float32, device=self.device)
        if mode.ndim == 0:
            mode = mode.expand(self.num_envs)
        if tuple(mode.shape) != (self.num_envs,):
            raise ValueError(f"Expected hidden mode {(self.num_envs,)}, got {tuple(mode.shape)}")
        self._hidden_future_mode.copy_(mode)

    def set_state_dict(self, state, env_idx: torch.Tensor | None = None):
        """Restore the state-only task fields without calling agent methods."""
        self.payload.set_state(state["payload"], env_idx=env_idx)
        self.proxy.set_state(state["tcp_proxy"], env_idx=env_idx)
        if self.robust_distractor:
            self.distractor.set_state(state["visual_distractor"], env_idx=env_idx)
        if env_idx is None:
            self.goal_pos.copy_(state["goal_pos"])
            self._task_phase.copy_(state["task_phase"])
        else:
            self.goal_pos[env_idx] = state["goal_pos"]
            self._task_phase[env_idx] = state["task_phase"]
        self._set_task_state(state, env_idx)

    def _get_obs_extra(self, info: dict):
        # Do not add _hidden_future_mode or _task_phase here. They are label and
        # replay diagnostics only, never fair observation features.
        return dict(
            payload_pose=self.payload.pose.raw_pose,
            payload_linear_velocity=self.payload.get_linear_velocity(),
            tcp_proxy_pose=self.proxy.pose.raw_pose,
            goal_pos=self.goal_pos,
        )

    def set_proxy_target(self, position: torch.Tensor | list[float], *, apply: bool = True):
        """Set the controlled TCP-proxy pose for all vectorized worlds."""
        position = torch.as_tensor(position, dtype=torch.float32, device=self.device)
        if position.ndim == 1:
            position = position.expand(self.num_envs, -1)
        if position.shape != (self.num_envs, 3):
            raise ValueError(
                f"Expected proxy position {(self.num_envs, 3)}, got {tuple(position.shape)}"
            )
        self.proxy.set_pose(Pose.create_from_pq(p=position))
        if apply and self.gpu_sim_enabled:
            self.scene._gpu_apply_all()

    def _before_proxy_step(self):
        """Family hook invoked after the proxy target is written, before GPU apply."""

    def _robust_distractor_offset(self) -> torch.Tensor:
        if self.robust_distractor_camera_offset is None:
            return torch.tensor([-0.16, 0.0, 0.14], dtype=torch.float32, device=self.device)
        return torch.as_tensor(
            self.robust_distractor_camera_offset, dtype=torch.float32, device=self.device
        )

    def _step_action(self, action):
        # BaseEnv delegates actions to a robot controller.  Here, the action is
        # a bounded Cartesian delta for the collision-only TCP proxy.
        if action is not None:
            action = torch.as_tensor(action, dtype=torch.float32, device=self.device)
            if action.ndim == 1:
                action = action.expand(self.num_envs, -1)
            if action.shape != (self.num_envs, 3):
                raise ValueError(
                    f"Expected action {(self.num_envs, 3)}, got {tuple(action.shape)}"
                )
            action = torch.clamp(action, -1.0, 1.0)
            self.set_proxy_target(
                self.proxy.pose.p + action * self.proxy_step_size, apply=False
            )

        if self.robust_distractor:
            # The render-only distractor stays lateral through frames 2--3,
            # then approaches/crosses the target after the target-only
            # occlusion.  It remains independent of hidden dynamics/labels.
            pos = self.payload.pose.p.clone()
            post_occlusion = torch.clamp(self._elapsed_steps.float() - 3.0, min=0.0, max=2.0)
            approach = (1.0 - 0.5 * post_occlusion).unsqueeze(-1)
            pos += approach * self._robust_distractor_offset().unsqueeze(0)
            self.distractor.set_pose(Pose.create_from_pq(p=pos))
        self._before_proxy_step()
        if self.gpu_sim_enabled:
            # All state writes for a control step are applied exactly once.
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

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        distance = torch.linalg.vector_norm(
            self.payload.pose.p - self.goal_pos, dim=1
        )
        return 1.0 - torch.tanh(5.0 * distance)


@register_env("ActMaskMovingCubeIntercept-v1", max_episode_steps=80)
class MovingCubeInterceptEnv(_StateOnlyProxyEnv):
    """Intercept a laterally moving cube with the TCP proxy."""

    def _initialize_family(self, env_idx: torch.Tensor, payload_pos: torch.Tensor):
        self.payload.set_linear_velocity(torch.zeros((len(env_idx), 3), device=self.device))
        self.goal_pos[env_idx] = payload_pos

    def _before_proxy_step(self):
        # At the same observable decision state, a hidden future kick either
        # leaves the cube interceptable (mode=0) or moves it after four steps.
        event_now = self._elapsed_steps == 4
        if bool(event_now.any()):
            displaced = self.payload.pose.p.clone()
            # Modes 0/1 preserve the frozen 3P behavior.  Signed modes are
            # additionally used by the separately versioned 3Q benchmark.
            displaced[:, 1] += 0.42 * self._hidden_future_mode
            self.payload.set_pose(Pose.create_from_pq(p=displaced))
            self.payload.set_linear_velocity(torch.zeros_like(displaced))

    def evaluate(self):
        intercept_distance = torch.linalg.vector_norm(
            self.payload.pose.p - self.proxy.pose.p, dim=1
        )
        success = intercept_distance < self.success_radius
        return dict(success=success, intercept_distance=intercept_distance)


@register_env("ActMaskMovingContainerPlacement-v1", max_episode_steps=100)
class MovingContainerPlacementEnv(_StateOnlyProxyEnv):
    """Place a moving payload into a translating collision-only container proxy."""

    def _load_scene(self, options: dict):
        super()._load_scene(options)
        self.container = _collision_box(
            self.scene, "container_proxy", (0.09, 0.09, 0.03), "kinematic"
        )

    def _initialize_family(self, env_idx: torch.Tensor, payload_pos: torch.Tensor):
        container_pos = payload_pos + torch.tensor([0.18, 0.0, 0.04], device=self.device)
        self.container.set_pose(Pose.create_from_pq(p=container_pos))
        self.goal_pos[env_idx] = container_pos

    def _get_task_state(self) -> dict:
        state = super()._get_task_state()
        state["container_proxy"] = self.container.get_state()
        return state

    def _set_task_state(self, state: dict, env_idx: torch.Tensor | None):
        super()._set_task_state(state, env_idx)
        self.container.set_state(state["container_proxy"], env_idx=env_idx)

    def _before_proxy_step(self):
        # A deterministic, observable translation of the container creates a
        # moving-placement timing requirement without hidden camera shortcuts.
        offset = torch.zeros_like(self.goal_pos)
        event_active = self._elapsed_steps >= 6
        offset[:, 1] = torch.where(
            event_active,
            0.32 * self._hidden_future_mode,
            torch.zeros(self.num_envs, device=self.device),
        )
        container_pos = self.goal_pos + offset
        self.container.set_pose(Pose.create_from_pq(p=container_pos))

        # This is a controlled TCP proxy rather than a full robot hand.  A
        # near-contact acquires the payload; carrying and release both update
        # the PhysX actor state before the single per-step GPU apply.
        proxy_pos = self.proxy.pose.p
        payload_pos = self.payload.pose.p
        acquire = torch.linalg.vector_norm(proxy_pos - payload_pos, dim=1) < 0.07
        grasped = torch.logical_or(self._task_phase > 0.5, acquire)
        release = grasped & (
            torch.linalg.vector_norm(proxy_pos - container_pos, dim=1) < 0.08
        )
        carried_pos = proxy_pos + torch.tensor([0.0, 0.0, -0.045], device=self.device)
        released_pos = container_pos + torch.tensor([0.0, 0.0, 0.055], device=self.device)
        next_pos = torch.where(grasped[:, None], carried_pos, payload_pos)
        next_pos = torch.where(release[:, None], released_pos, next_pos)
        self.payload.set_pose(Pose.create_from_pq(p=next_pos))
        self.payload.set_linear_velocity(torch.zeros_like(next_pos))
        self.payload.set_angular_velocity(torch.zeros_like(next_pos))
        self._task_phase = torch.where(
            release,
            torch.full_like(self._task_phase, 2.0),
            torch.where(grasped, torch.ones_like(self._task_phase), self._task_phase),
        )

    def evaluate(self):
        container_pos = self.container.pose.p
        horizontal = torch.linalg.vector_norm(
            self.payload.pose.p[:, :2] - container_pos[:, :2], dim=1
        )
        height_ok = torch.abs(self.payload.pose.p[:, 2] - container_pos[:, 2]) < 0.09
        speed_ok = torch.linalg.vector_norm(self.payload.get_linear_velocity(), dim=1) < 0.45
        released = self._task_phase >= 2.0
        return dict(success=released & (horizontal < 0.075) & height_ok & speed_ok, container_distance=horizontal)

    def _get_obs_extra(self, info: dict):
        obs = super()._get_obs_extra(info)
        obs["container_pose"] = self.container.pose.raw_pose
        return obs


@register_env("ActMaskRotatingTargetInteraction-v1", max_episode_steps=80)
class RotatingTargetInteractionEnv(_StateOnlyProxyEnv):
    """Reach a target whose required contact point rotates during the episode."""

    proxy_step_size = 0.07

    def _ensure_task_buffers(self):
        super()._ensure_task_buffers()
        if not hasattr(self, "target_angle") or self.target_angle.shape[0] != self.num_envs:
            self.target_angle = torch.zeros(self.num_envs, device=self.device)

    def _initialize_family(self, env_idx: torch.Tensor, payload_pos: torch.Tensor):
        self.target_angle[env_idx] = torch.rand(len(env_idx), device=self.device) * 6.283185307
        angle = self.target_angle[env_idx]
        radial = torch.stack(
            [0.13 * torch.cos(angle), 0.13 * torch.sin(angle), torch.zeros_like(angle)],
            dim=1,
        )
        self.goal_pos[env_idx] = payload_pos + radial

    def _get_task_state(self) -> dict:
        state = super()._get_task_state()
        state["target_angle"] = self.target_angle.clone()
        return state

    def _set_task_state(self, state: dict, env_idx: torch.Tensor | None):
        super()._set_task_state(state, env_idx)
        if env_idx is None:
            self.target_angle.copy_(state["target_angle"])
        else:
            self.target_angle[env_idx] = state["target_angle"]

    def _before_proxy_step(self):
        angular_velocity = 0.26 * self._hidden_future_mode
        self.target_angle = torch.remainder(
            self.target_angle + angular_velocity, 6.283185307
        )
        radial = torch.stack(
            [0.13 * torch.cos(self.target_angle), 0.13 * torch.sin(self.target_angle), torch.zeros_like(self.target_angle)],
            dim=1,
        )
        self.goal_pos = self.payload.pose.p + radial

    def evaluate(self):
        interaction_distance = torch.linalg.vector_norm(
            self.proxy.pose.p - self.goal_pos, dim=1
        )
        return dict(success=interaction_distance < self.success_radius, interaction_distance=interaction_distance)

    def _get_obs_extra(self, info: dict):
        obs = super()._get_obs_extra(info)
        obs["target_phase"] = torch.stack(
            [torch.cos(self.target_angle), torch.sin(self.target_angle)], dim=1
        )
        return obs


@register_env("ActMaskCrossingObstacleTiming-v1", max_episode_steps=80)
class CrossingObstacleTimingEnv(_StateOnlyProxyEnv):
    """Cross a moving gate whose signed motion selects the contact window."""

    def _load_scene(self, options: dict):
        super()._load_scene(options)
        self.gate = _collision_box(
            self.scene, "timing_gate", (0.035, 0.09, 0.09), "kinematic"
        )

    def _initialize_family(self, env_idx: torch.Tensor, payload_pos: torch.Tensor):
        gate_pos = payload_pos + torch.tensor([0.18, 0.0, 0.06], device=self.device)
        self.gate.set_pose(Pose.create_from_pq(p=gate_pos))
        self.goal_pos[env_idx] = gate_pos

    def _get_task_state(self) -> dict:
        state = super()._get_task_state()
        state["timing_gate"] = self.gate.get_state()
        return state

    def _set_task_state(self, state: dict, env_idx: torch.Tensor | None):
        super()._set_task_state(state, env_idx)
        self.gate.set_state(state["timing_gate"], env_idx=env_idx)

    def _before_proxy_step(self):
        # At the fixed decision pose, the gate later moves through (+) or away
        # from (-) the same TCP contact window.  The evaluated label is read
        # after the GPU-PhysX rollout in ``evaluate``.
        event_active = self._elapsed_steps >= 4
        gate_pos = self.goal_pos.clone()
        gate_pos[:, 1] += torch.where(
            event_active,
            0.22 * self._hidden_future_mode,
            torch.zeros(self.num_envs, device=self.device),
        )
        self.gate.set_pose(Pose.create_from_pq(p=gate_pos))

    def evaluate(self):
        crossing_distance = torch.linalg.vector_norm(
            self.proxy.pose.p - self.gate.pose.p, dim=1
        )
        return dict(success=crossing_distance < self.success_radius, crossing_distance=crossing_distance)

    def _get_obs_extra(self, info: dict):
        obs = super()._get_obs_extra(info)
        obs["timing_gate_pose"] = self.gate.pose.raw_pose
        return obs
