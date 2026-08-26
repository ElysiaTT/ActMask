"""Versioned matched-final nonlinear GPU-PhysX tasks for 3R-NL v1."""
from __future__ import annotations

import torch
import sapien
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs import Pose

from actmask.data.maniskill_state_tasks import _StateOnlyProxyEnv, _collision_box


class _MatchedFinalNonlinearEnv(_StateOnlyProxyEnv):
    """A target whose hidden dynamics start only after a common decision state.

    The dynamics buffers are oracle-only task state.  ``get_obs`` deliberately
    inherits the state-only proxy schema and never exposes any of them.
    """

    mechanism_count = 2
    min_execution_step = 6
    success_radius = 0.055
    family_index = 0

    def _load_scene(self, options: dict):
        # Correction NL-C1 (common to both mechanisms): the old 0.025/0.035
        # collision radii made the frozen 0.055 interception criterion
        # physically unreachable (minimum 0.061).  Smaller collision-only
        # primitives preserve the criterion and allow actual GPU contact range.
        floor = self.scene.create_actor_builder()
        floor.add_box_collision(half_size=(1.0, 1.0, 0.05))
        floor.set_initial_pose(sapien.Pose(p=[0, 0, -0.05]))
        floor.build_static("ground")
        self.payload = _collision_box(self.scene, "payload", (0.015, 0.015, 0.015), "dynamic")
        self.proxy = _collision_box(self.scene, "tcp_proxy", (0.020, 0.020, 0.020), "kinematic")

    def _ensure_task_buffers(self):
        super()._ensure_task_buffers()
        if not hasattr(self, "nl_anchor") or self.nl_anchor.shape[0] != self.num_envs:
            self.nl_anchor = torch.zeros((self.num_envs, 3), device=self.device)
            self.nl_mechanism = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
            self.nl_branch = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
            self.nl_damping = torch.zeros(self.num_envs, device=self.device)
            self.nl_frequency = torch.zeros(self.num_envs, device=self.device)
            self.nl_threshold = torch.zeros(self.num_envs, device=self.device)
            self.nl_delay = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)

    def _initialize_family(self, env_idx: torch.Tensor, payload_pos: torch.Tensor):
        self.nl_anchor[env_idx] = payload_pos
        self.nl_mechanism[env_idx] = 0
        self.nl_branch[env_idx] = 0
        self.nl_damping[env_idx] = 0.12
        self.nl_frequency[env_idx] = 1.15
        self.nl_threshold[env_idx] = -0.06
        self.nl_delay[env_idx] = 0
        self.goal_pos[env_idx] = payload_pos

    def _get_task_state(self) -> dict:
        state = super()._get_task_state()
        state.update(
            nl_anchor=self.nl_anchor.clone(), nl_mechanism=self.nl_mechanism.clone(),
            nl_branch=self.nl_branch.clone(), nl_damping=self.nl_damping.clone(),
            nl_frequency=self.nl_frequency.clone(), nl_threshold=self.nl_threshold.clone(),
            nl_delay=self.nl_delay.clone(),
        )
        return state

    def _set_task_state(self, state: dict, env_idx: torch.Tensor | None):
        super()._set_task_state(state, env_idx)
        keys = ("nl_anchor", "nl_mechanism", "nl_branch", "nl_damping", "nl_frequency", "nl_threshold", "nl_delay")
        for key in keys:
            value = getattr(self, key)
            if env_idx is None:
                value.copy_(state[key])
            else:
                value[env_idx] = state[key]

    def set_matched_dynamics(self, mechanism, branch, damping, frequency, threshold, delay):
        values = dict(
            nl_mechanism=(mechanism, torch.int64), nl_branch=(branch, torch.int64),
            nl_damping=(damping, torch.float32), nl_frequency=(frequency, torch.float32),
            nl_threshold=(threshold, torch.float32), nl_delay=(delay, torch.int64),
        )
        for name, (raw, dtype) in values.items():
            value = torch.as_tensor(raw, dtype=dtype, device=self.device)
            if value.ndim == 0:
                value = value.expand(self.num_envs)
            if tuple(value.shape) != (self.num_envs,):
                raise ValueError(f"{name} must be shaped [num_envs]")
            getattr(self, name).copy_(value)
        if bool(((self.nl_mechanism < 0) | (self.nl_mechanism >= self.mechanism_count)).any()):
            raise ValueError("mechanism must be 0 or 1")
        if bool(((self.nl_branch < 0) | (self.nl_branch > 1)).any()):
            raise ValueError("branch must be 0 or 1")

    def _response_at(self, n: torch.Tensor) -> torch.Tensor:
        """Post-decision displacement inferred from hidden, historical dynamics."""
        n = n.to(torch.float32)
        drive_phase = self.nl_branch.to(torch.float32) * torch.pi
        # Correction NL-C2 for the damping-drive mechanism: a common driven
        # displacement makes phase/damping branches separate before the fixed
        # success window, while damping/frequency still control the nonlinear
        # component and are visible only in earlier history.
        sign = torch.where(self.nl_branch == 0, torch.ones_like(n), -torch.ones_like(n))
        damping_drive = sign * (0.025 * n + 0.020 * (1.0 - torch.exp(-self.nl_damping * n)) * torch.sin(self.nl_frequency * n + drive_phase))
        active = torch.clamp(n - self.nl_delay.to(torch.float32), min=0.0)
        mode = torch.where(self.nl_branch == 0, torch.ones_like(n), -torch.ones_like(n))
        # Correction NL-C3: the initial 0.0065 quadratic response moved the
        # positive target beyond the common proxy speed as well.  0.001 keeps
        # it reachable while its two modes differ by 0.072 at step six.
        hysteresis = mode * 0.0010 * active.square()
        return torch.where(self.nl_mechanism == 0, damping_drive, hysteresis)

    def target_at(self, step: torch.Tensor | float | int) -> torch.Tensor:
        n = torch.as_tensor(step, dtype=torch.float32, device=self.device)
        if n.ndim == 0:
            n = n.expand(self.num_envs)
        response = self._response_at(n)
        target = self.nl_anchor.clone()
        if self.family_index == 0:
            target[:, 1] += response
        elif self.family_index == 1:
            target[:, 0] += response
            target[:, 2] += 0.04
        else:
            angle = response / 0.13
            target[:, 0] += 0.13 * torch.cos(angle)
            target[:, 1] += 0.13 * torch.sin(angle)
        return target

    def _before_proxy_step(self):
        n = self._elapsed_steps.to(torch.float32) + 1.0
        target = self.target_at(n)
        if self.family_index == 2:
            self.goal_pos.copy_(target)
        else:
            self.payload.set_pose(Pose.create_from_pq(p=target))
            self.payload.set_linear_velocity(torch.zeros_like(target))
            self.payload.set_angular_velocity(torch.zeros_like(target))
            self.goal_pos.copy_(target)

    def evaluate(self):
        target = self.goal_pos if self.family_index == 2 else self.payload.pose.p
        distance = torch.linalg.vector_norm(self.proxy.pose.p - target, dim=1)
        return dict(success=(distance < self.success_radius) & (self._elapsed_steps >= self.min_execution_step), matched_target_distance=distance)


@register_env("ActMaskNLMatchedDampedCapture-v1", max_episode_steps=80)
class MatchedDampedCaptureEnv(_MatchedFinalNonlinearEnv):
    family_index = 0


@register_env("ActMaskNLMatchedHystereticContainer-v1", max_episode_steps=80)
class MatchedHystereticContainerEnv(_MatchedFinalNonlinearEnv):
    family_index = 1


@register_env("ActMaskNLMatchedDampedRotatingSlot-v1", max_episode_steps=80)
class MatchedDampedRotatingSlotEnv(_MatchedFinalNonlinearEnv):
    family_index = 2
