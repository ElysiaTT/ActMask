"""v2.1 registrations using the protected v2 PhysX response dynamics."""

from __future__ import annotations

import torch

from mani_skill.utils.registration import register_env

from actmask.data.history_action_binding_v2_tasks import _BindingV2BaseEnv


@register_env("ActMaskHistoryBindingV21Discrete-v0", max_episode_steps=8192)
class BindingV21DiscreteEnv(_BindingV2BaseEnv):
    task_kind = "task_a_discrete_higher_order"

    def radial_response(self, radius: torch.Tensor) -> torch.Tensor:
        return radius

    def inverse_radial_response(self, magnitude: torch.Tensor) -> torch.Tensor:
        return torch.clamp(magnitude, 0.0, 1.0)


@register_env("ActMaskHistoryBindingV21Continuous-v0", max_episode_steps=12288)
class BindingV21ContinuousEnv(_BindingV2BaseEnv):
    task_kind = "task_b_continuous_nonlinear"

    def radial_response(self, radius: torch.Tensor) -> torch.Tensor:
        return self.latent_alpha * radius - self.latent_beta * radius.square()

    def inverse_radial_response(self, magnitude: torch.Tensor) -> torch.Tensor:
        beta = torch.clamp(self.latent_beta, min=1.0e-6)
        discriminant = torch.clamp(self.latent_alpha.square() - 4.0 * beta * magnitude, min=0.0)
        root = (self.latent_alpha - torch.sqrt(discriminant)) / (2.0 * beta)
        linear = magnitude / torch.clamp(self.latent_alpha, min=1.0e-6)
        return torch.clamp(
            torch.where(self.latent_beta > 1.0e-6, root, linear), 0.0, 1.0
        )


TASK_IDS = {
    "task_a_discrete_higher_order": "ActMaskHistoryBindingV21Discrete-v0",
    "task_b_continuous_nonlinear": "ActMaskHistoryBindingV21Continuous-v0",
}
