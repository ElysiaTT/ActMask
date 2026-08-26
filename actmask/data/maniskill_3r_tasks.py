"""Separate nonlinear GPU-PhysX signed-dynamics task for Milestone 3R."""

from __future__ import annotations

import torch
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs import Pose

from actmask.data.maniskill_state_tasks import _StateOnlyProxyEnv


@register_env("ActMaskNonlinearSignedIntercept-v1", max_episode_steps=80)
class NonlinearSignedInterceptEnv(_StateOnlyProxyEnv):
    """A common-action interception task with nonlinear signed future motion.

    GPU-PhysX advances after every kinematic payload write. The branch sign and
    mechanism are oracle-only and excluded from state observations.
    """

    mechanism_count = 8
    min_execution_step = 6

    def _ensure_task_buffers(self):
        super()._ensure_task_buffers()
        if not hasattr(self, "payload_base") or self.payload_base.shape[0] != self.num_envs:
            self.payload_base = torch.zeros((self.num_envs, 3), device=self.device)
            self.nonlinear_mechanism = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)

    def _initialize_family(self, env_idx: torch.Tensor, payload_pos: torch.Tensor):
        self.payload_base[env_idx] = payload_pos
        self.goal_pos[env_idx] = payload_pos
        self.nonlinear_mechanism[env_idx] = 0

    def _get_task_state(self) -> dict:
        state = super()._get_task_state()
        state.update(payload_base=self.payload_base.clone(), nonlinear_mechanism=self.nonlinear_mechanism.clone())
        return state

    def _set_task_state(self, state: dict, env_idx: torch.Tensor | None):
        super()._set_task_state(state, env_idx)
        if env_idx is None:
            self.payload_base.copy_(state["payload_base"])
            self.nonlinear_mechanism.copy_(state["nonlinear_mechanism"])
        else:
            self.payload_base[env_idx] = state["payload_base"]
            self.nonlinear_mechanism[env_idx] = state["nonlinear_mechanism"]

    def set_nonlinear_mechanism(self, mechanism: int | torch.Tensor):
        value = torch.as_tensor(mechanism, dtype=torch.int64, device=self.device)
        if value.ndim == 0:
            value = value.expand(self.num_envs)
        if tuple(value.shape) != (self.num_envs,) or bool(((value < 0) | (value >= self.mechanism_count)).any()):
            raise ValueError("mechanism must be [0, 7]")
        self.nonlinear_mechanism.copy_(value)

    def _offset(self) -> torch.Tensor:
        n = self._elapsed_steps.to(torch.float32) + 1.0
        linear = 0.0025 * n.square()  # constant acceleration
        jerk = 0.00030 * n.pow(3)  # changing acceleration / jerk
        piecewise = 0.018 * torch.clamp(n - 3.0, min=0.0)
        delayed = 0.026 * torch.clamp(n - 5.0, min=0.0)
        damping = 0.18 * (1.0 - torch.exp(-0.27 * n))
        collision = torch.where(n < 7.0, 0.018 * n, 0.126 + 0.008 * (n - 7.0))
        curved = 0.20 * torch.sin(0.07 * n)
        switch = torch.where(n < 6.0, 0.008 * n, 0.048 + 0.032 * (n - 6.0))
        values = torch.stack((linear, jerk, piecewise, delayed, damping, collision, curved, switch), dim=1)
        selected = values.gather(1, self.nonlinear_mechanism[:, None]).squeeze(1)
        offset = torch.zeros_like(self.payload_base)
        offset[:, 1] = selected * self._hidden_future_mode
        # Curved motion has an observable non-linear x component but retains
        # the same current decision state before execution.
        curved_mask = self.nonlinear_mechanism == 6
        offset[:, 0] = torch.where(curved_mask, 0.08 * (1.0 - torch.cos(0.07 * n)), torch.zeros_like(n))
        return offset

    def _before_proxy_step(self):
        next_pos = self.payload_base + self._offset()
        self.payload.set_pose(Pose.create_from_pq(p=next_pos))
        self.payload.set_linear_velocity(torch.zeros_like(next_pos))
        self.payload.set_angular_velocity(torch.zeros_like(next_pos))

    def evaluate(self):
        distance = torch.linalg.vector_norm(self.proxy.pose.p - self.payload.pose.p, dim=1)
        return dict(success=(distance < self.success_radius) & (self._elapsed_steps >= self.min_execution_step), intercept_distance=distance)
