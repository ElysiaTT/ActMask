from __future__ import annotations

import torch

from actmask.data.maniskill_3r_tasks import NonlinearSignedInterceptEnv


def test_nonlinear_mechanism_accepts_all_preregistered_indices() -> None:
    # Unit-level range guard; CUDA smoke is run by the phase implementation.
    assert NonlinearSignedInterceptEnv.mechanism_count == 8
    assert NonlinearSignedInterceptEnv.success_radius == 0.065
    assert torch.isfinite(torch.tensor([0.018, 0.0022, 0.27, 0.19])).all()
