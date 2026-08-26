"""Separate 3Q signed moving-window task; it never changes frozen 3P tasks."""

from __future__ import annotations

import torch
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs import Pose

from actmask.data.maniskill_state_tasks import _StateOnlyProxyEnv, _collision_box


@register_env("ActMaskSignedMovingWindowPlacement-v1", max_episode_steps=100)
class SignedMovingWindowPlacementEnv(_StateOnlyProxyEnv):
    """Place a carried payload into a signed, moving capture window.

    Frozen 3Q constants: capture half-width 0.075 m, window excursion 0.24 m,
    relative-speed limit 0.45 m/s, and two consecutive valid control steps.
    """

    window_excursion = 0.24
    capture_half_width = 0.075
    relative_speed_limit = 0.45
    capture_persistence_steps = 2

    def _load_scene(self, options: dict):
        super()._load_scene(options)
        self.window = _collision_box(self.scene, "signed_capture_window", (0.09, 0.09, 0.03), "kinematic")

    def _ensure_task_buffers(self):
        super()._ensure_task_buffers()
        size = self.num_envs
        device = self.device
        if not hasattr(self, "window_base") or self.window_base.shape[0] != size:
            self.window_base = torch.zeros((size, 3), device=device)
            self.window_mechanism = torch.zeros(size, dtype=torch.int64, device=device)
            self.capture_streak = torch.zeros(size, dtype=torch.int64, device=device)
            self.capture_latched = torch.zeros(size, dtype=torch.bool, device=device)

    def _initialize_family(self, env_idx: torch.Tensor, payload_pos: torch.Tensor):
        base = payload_pos + torch.tensor([0.18, 0.0, 0.04], device=self.device)
        self.window_base[env_idx] = base
        self.window.set_pose(Pose.create_from_pq(p=base))
        self.goal_pos[env_idx] = base
        self.window_mechanism[env_idx] = 0
        self.capture_streak[env_idx] = 0
        self.capture_latched[env_idx] = False

    def _get_task_state(self) -> dict:
        state = super()._get_task_state()
        state.update(
            signed_window=self.window.get_state(),
            window_base=self.window_base.clone(),
            window_mechanism=self.window_mechanism.clone(),
            capture_streak=self.capture_streak.clone(),
            capture_latched=self.capture_latched.clone(),
        )
        return state

    def _set_task_state(self, state: dict, env_idx: torch.Tensor | None):
        super()._set_task_state(state, env_idx)
        self.window.set_state(state["signed_window"], env_idx=env_idx)
        if env_idx is None:
            self.window_base.copy_(state["window_base"])
            self.window_mechanism.copy_(state["window_mechanism"])
            self.capture_streak.copy_(state["capture_streak"])
            self.capture_latched.copy_(state["capture_latched"])
        else:
            self.window_base[env_idx] = state["window_base"]
            self.window_mechanism[env_idx] = state["window_mechanism"]
            self.capture_streak[env_idx] = state["capture_streak"]
            self.capture_latched[env_idx] = state["capture_latched"]

    def set_window_mechanism(self, mechanism: int | torch.Tensor):
        mechanism = torch.as_tensor(mechanism, dtype=torch.int64, device=self.device)
        if mechanism.ndim == 0:
            mechanism = mechanism.expand(self.num_envs)
        if tuple(mechanism.shape) != (self.num_envs,) or bool(((mechanism < 0) | (mechanism > 3)).any()):
            raise ValueError("window mechanism must be an integer [0, 3] per environment")
        self.window_mechanism.copy_(mechanism)

    def _window_offset(self) -> torch.Tensor:
        # 0: signed constant velocity; 1: signed acceleration; 2: phase/delay;
        # 3: early signed crossing. All converge to the frozen 0.24 m excursion.
        elapsed = self._elapsed_steps.to(torch.float32)
        start = torch.tensor([5.0, 4.0, 8.0, 3.0], device=self.device)[self.window_mechanism]
        local = torch.clamp(elapsed - start + 1.0, min=0.0)
        linear = torch.clamp(0.04 * local, max=self.window_excursion)
        accelerated = torch.clamp(0.012 * local.square(), max=self.window_excursion)
        delayed = torch.clamp(0.06 * local, max=self.window_excursion)
        early = torch.clamp(0.03 * local, max=self.window_excursion)
        magnitude = torch.where(self.window_mechanism == 0, linear, torch.where(self.window_mechanism == 1, accelerated, torch.where(self.window_mechanism == 2, delayed, early)))
        return magnitude * self._hidden_future_mode

    def _before_proxy_step(self):
        window_pos = self.window_base.clone()
        window_pos[:, 1] += self._window_offset()
        self.window.set_pose(Pose.create_from_pq(p=window_pos))
        proxy = self.proxy.pose.p
        payload = self.payload.pose.p
        acquire = torch.linalg.vector_norm(proxy - payload, dim=1) < 0.07
        grasped = torch.logical_or(self._task_phase > 0.5, acquire)
        capture_ready = grasped & (torch.linalg.vector_norm(proxy - window_pos, dim=1) < self.capture_half_width)
        carrying = proxy + torch.tensor([0.0, 0.0, -0.045], device=self.device)
        captured = window_pos + torch.tensor([0.0, 0.0, 0.055], device=self.device)
        next_pos = torch.where(grasped[:, None], carrying, payload)
        next_pos = torch.where(capture_ready[:, None], captured, next_pos)
        next_pos = torch.where(self.capture_latched[:, None], captured, next_pos)
        self.payload.set_pose(Pose.create_from_pq(p=next_pos))
        self.payload.set_linear_velocity(torch.zeros_like(next_pos))
        self.payload.set_angular_velocity(torch.zeros_like(next_pos))
        self._task_phase = torch.where(capture_ready | self.capture_latched, torch.full_like(self._task_phase, 2.0), torch.where(grasped, torch.ones_like(self._task_phase), self._task_phase))

    def _after_control_step(self):
        window_pos = self.window.pose.p
        horizontal = torch.linalg.vector_norm(self.payload.pose.p[:, :2] - window_pos[:, :2], dim=1)
        height = torch.abs(self.payload.pose.p[:, 2] - window_pos[:, 2]) < 0.09
        speed = torch.linalg.vector_norm(self.payload.get_linear_velocity(), dim=1) < self.relative_speed_limit
        valid = (self._task_phase >= 2.0) & (horizontal < self.capture_half_width) & height & speed
        self.capture_streak = torch.where(valid, self.capture_streak + 1, torch.zeros_like(self.capture_streak))
        self.capture_latched = torch.logical_or(self.capture_latched, self.capture_streak >= self.capture_persistence_steps)

    def evaluate(self):
        return dict(success=self.capture_latched, capture_streak=self.capture_streak.to(torch.float32))

    def _get_obs_extra(self, info: dict):
        obs = super()._get_obs_extra(info)
        obs["signed_window_pose"] = self.window.pose.raw_pose
        return obs


@register_env("ActMaskFixedPhaseRotatingCaptureWindow-v1", max_episode_steps=100)
class FixedPhaseRotatingCaptureWindowEnv(_StateOnlyProxyEnv):
    """Fixed-phase capture window with hidden clockwise/counter-clockwise future."""

    fixed_initial_phase = 0.0
    capture_radius = 0.06
    rotation_radius = 0.13
    angular_speed = 0.22
    persistence_steps = 2
    # Frozen after the first probe: it excludes the later, unintended
    # near-full-orbit re-intersection of the common TCP plan with the opposite
    # branch while retaining the early positive capture window.
    action_horizon = 13

    def _load_scene(self, options: dict):
        super()._load_scene(options)
        self.capture_window = _collision_box(self.scene, "fixed_phase_rotating_window", (0.04, 0.04, 0.04), "kinematic")

    def _ensure_task_buffers(self):
        super()._ensure_task_buffers()
        if not hasattr(self, "rotation_center") or self.rotation_center.shape[0] != self.num_envs:
            self.rotation_center = torch.zeros((self.num_envs, 3), device=self.device)
            self.rotation_mechanism = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
            self.capture_streak = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
            self.capture_latched = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _initialize_family(self, env_idx: torch.Tensor, payload_pos: torch.Tensor):
        self.rotation_center[env_idx] = payload_pos
        window_pos = payload_pos + torch.tensor([self.rotation_radius, 0.0, 0.05], device=self.device)
        self.capture_window.set_pose(Pose.create_from_pq(p=window_pos))
        self.goal_pos[env_idx] = window_pos
        self.rotation_mechanism[env_idx] = 0
        self.capture_streak[env_idx] = 0
        self.capture_latched[env_idx] = False

    def _get_task_state(self) -> dict:
        state = super()._get_task_state()
        state.update(
            fixed_phase_window=self.capture_window.get_state(),
            rotation_center=self.rotation_center.clone(),
            rotation_mechanism=self.rotation_mechanism.clone(),
            capture_streak=self.capture_streak.clone(),
            capture_latched=self.capture_latched.clone(),
        )
        return state

    def _set_task_state(self, state: dict, env_idx: torch.Tensor | None):
        super()._set_task_state(state, env_idx)
        self.capture_window.set_state(state["fixed_phase_window"], env_idx=env_idx)
        if env_idx is None:
            self.rotation_center.copy_(state["rotation_center"])
            self.rotation_mechanism.copy_(state["rotation_mechanism"])
            self.capture_streak.copy_(state["capture_streak"])
            self.capture_latched.copy_(state["capture_latched"])
        else:
            self.rotation_center[env_idx] = state["rotation_center"]
            self.rotation_mechanism[env_idx] = state["rotation_mechanism"]
            self.capture_streak[env_idx] = state["capture_streak"]
            self.capture_latched[env_idx] = state["capture_latched"]

    def set_rotation_mechanism(self, mechanism: int | torch.Tensor):
        value = torch.as_tensor(mechanism, dtype=torch.int64, device=self.device)
        if value.ndim == 0:
            value = value.expand(self.num_envs)
        if tuple(value.shape) != (self.num_envs,) or bool(((value < 0) | (value > 3)).any()):
            raise ValueError("rotation mechanism must be an integer [0, 3] per environment")
        self.rotation_mechanism.copy_(value)

    def _angle(self) -> torch.Tensor:
        n = self._elapsed_steps.to(torch.float32) + 1.0
        constant = self.angular_speed * n
        accelerated = 0.028 * n.square()
        delayed = 0.32 + 0.30 * torch.clamp(n - 4.0, min=0.0)
        reversal = self.angular_speed * n
        magnitude = torch.where(self.rotation_mechanism == 0, constant, torch.where(self.rotation_mechanism == 1, accelerated, torch.where(self.rotation_mechanism == 2, delayed, reversal)))
        return self.fixed_initial_phase + magnitude * self._hidden_future_mode

    def _before_proxy_step(self):
        angle = self._angle()
        window_pos = self.rotation_center + torch.stack((self.rotation_radius * torch.cos(angle), self.rotation_radius * torch.sin(angle), torch.full_like(angle, 0.05)), dim=1)
        self.capture_window.set_pose(Pose.create_from_pq(p=window_pos))

    def _after_control_step(self):
        in_window = torch.linalg.vector_norm(self.proxy.pose.p - self.capture_window.pose.p, dim=1) < self.capture_radius
        self.capture_streak = torch.where(in_window, self.capture_streak + 1, torch.zeros_like(self.capture_streak))
        self.capture_latched = torch.logical_or(self.capture_latched, self.capture_streak >= self.persistence_steps)

    def evaluate(self):
        return dict(success=self.capture_latched, capture_streak=self.capture_streak.to(torch.float32))

    def _get_obs_extra(self, info: dict):
        obs = super()._get_obs_extra(info)
        obs["fixed_phase_window_pose"] = self.capture_window.pose.raw_pose
        return obs
