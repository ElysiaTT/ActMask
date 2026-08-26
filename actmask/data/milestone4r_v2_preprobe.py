"""GPU-PhysX, no-render candidate-diversity preprobe for 4R-v2.

This module is deliberately separate from frozen 4R generation.  Its C10
placement plans are functions only of the common decision state and slot; they
never read the hidden branch or an observed outcome.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from actmask.data import maniskill_state_tasks as _registered_tasks  # noqa: F401


@dataclass(frozen=True)
class PlacementCandidate:
    slot: int
    acquire_steps: int
    endpoint_lateral_y_m: float
    endpoint_lateral_x_m: float
    approach_sign_x: int


C10: tuple[PlacementCandidate, ...] = (
    PlacementCandidate(0, 5, -0.06, -0.020, -1),
    PlacementCandidate(1, 5, -0.03, 0.020, 1),
    PlacementCandidate(2, 5, 0.00, -0.015, -1),
    PlacementCandidate(3, 5, 0.03, 0.015, 1),
    PlacementCandidate(4, 5, 0.06, 0.000, -1),
    PlacementCandidate(5, 4, 0.26, -0.020, 1),
    PlacementCandidate(6, 4, 0.29, 0.020, -1),
    PlacementCandidate(7, 4, 0.32, -0.015, 1),
    PlacementCandidate(8, 4, 0.35, 0.015, -1),
    PlacementCandidate(9, 4, 0.38, 0.000, 1),
)


def _clone(state: dict) -> dict:
    return {name: value.clone() for name, value in state.items()}


def _state_digest(state: dict) -> str:
    """Hash decision state excluding the intentionally selected future branch."""
    digest = hashlib.sha256()
    for name in sorted(state):
        if name == "hidden_future_mode":
            continue
        value = state[name].detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _planned_actions(base, state: dict, candidate: PlacementCandidate, horizon: int) -> np.ndarray:
    """Create a physical TCP plan from only the common decision state."""
    proxy = state["tcp_proxy"][:, :3].clone()
    payload = state["payload"][:, :3].clone()
    container = state["container_proxy"][:, :3].clone()
    approach = payload.clone()
    approach[:, 0] += 0.015 * candidate.approach_sign_x
    endpoint = container.clone()
    endpoint[:, 0] += candidate.endpoint_lateral_x_m
    endpoint[:, 1] += candidate.endpoint_lateral_y_m
    values = []
    for step in range(horizon):
        target = approach if step < candidate.acquire_steps else endpoint
        action = torch.clamp((target - proxy) / base.proxy_step_size, -1.0, 1.0)
        proxy = proxy + action * base.proxy_step_size
        values.append(action.detach().cpu().numpy().astype(np.float32, copy=False))
    return np.stack(values, axis=1)


def _rollout(env, decision: dict, actions: np.ndarray, branch: int) -> tuple[np.ndarray, str]:
    base = env.unwrapped
    base.set_state_dict(decision)
    base.scene._gpu_apply_all(); base.scene._gpu_fetch_all()
    restored_digest = _state_digest(base.get_state_dict())
    base._elapsed_steps.zero_()
    base.set_hidden_future_mode(float(branch))
    info = None
    for step in range(actions.shape[1]):
        _, _, _, _, info = env.step(torch.as_tensor(actions[:, step], device=base.device))
    assert info is not None
    return info["success"].detach().cpu().numpy().astype(bool), restored_digest


def run(output_dir: str | Path, *, worlds: int = 64, seed: int = 54000, horizon: int = 12) -> dict:
    if worlds < 64:
        raise ValueError("4R-v2 preregistration requires at least 64 base worlds")
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    env = gym.make("ActMaskMovingContainerPlacement-v1", obs_mode="state", num_envs=worlds, sim_backend="gpu", render_backend="none")
    try:
        _, _ = env.reset(seed=list(range(seed, seed + worlds)))
        base = env.unwrapped
        zero = torch.zeros((worlds, 3), dtype=torch.float32, device=base.device)
        # Match the six-frame 4R decision point without rendering frames.
        for _ in range(5):
            env.step(zero)
        decision = _clone(base.get_state_dict())
        digest = _state_digest(decision)
        records = []
        action_hashes = {}
        restored_state_digests = []
        for candidate in C10:
            actions = _planned_actions(base, decision, candidate, horizon)
            action_hashes[str(candidate.slot)] = hashlib.sha256(actions.tobytes()).hexdigest()
            rollouts = [_rollout(env, decision, actions, branch) for branch in (0, 1)]
            labels = [value[0] for value in rollouts]
            restored_state_digests.extend(value[1] for value in rollouts)
            # Recompute after both rollouts to prove branch selection cannot
            # influence candidate construction.
            identical_actions = np.array_equal(actions, _planned_actions(base, decision, candidate, horizon))
            for branch, success in enumerate(labels):
                for world in range(worlds):
                    records.append({"world_id": world, "branch": branch, "candidate_slot": candidate.slot, "success": bool(success[world])})
            if not identical_actions:
                raise AssertionError("candidate plan changed after branch rollouts")
        result = {
            "schema": "milestone4r-v2-placement-preprobe-v1",
            "worlds": worlds,
            "candidate_count": len(C10),
            "candidate_executions": worlds * len(C10) * 2,
            "rendering": False,
            "simulator": "ManiSkill 3 GPU PhysX",
            "decision_state_digest": digest,
            "restored_state_digests": restored_state_digests,
            "candidate_grid": [asdict(value) for value in C10],
            "candidate_action_hashes": action_hashes,
            "records": records
        }
        (output / "placement_preprobe_raw.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        return result
    finally:
        env.close()


if __name__ == "__main__":
    project = Path(__file__).resolve().parents[2]
    print(json.dumps(run(project / "outputs" / "actmask" / "milestone4r_v2_candidate_diversity" / "preprobe_attempt_1"), indent=2))
