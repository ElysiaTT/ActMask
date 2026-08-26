"""All-family, branch-independent GPU-PhysX C10 preprobes for 4R-v3."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from actmask.data import maniskill_state_tasks as _registered  # noqa: F401
from actmask.data.milestone4r_v2_preprobe import C10 as PLACEMENT_C10
from actmask.data.milestone4r_v2_preprobe import _clone, _state_digest, _planned_actions as _placement_actions


@dataclass(frozen=True)
class CubeCandidate:
    slot: int; wait_steps: int; lateral_y_m: float; endpoint_x_m: float; endpoint_z_m: float; approach_sign_x: int


@dataclass(frozen=True)
class PhaseCandidate:
    slot: int; wait_steps: int; phase_offset_rad: float; radial_offset_m: float; tangential_offset_m: float; approach_sign_x: int


# Five early/current and five delayed/lateral-intercept hypotheses.  Every
# field is fixed before execution; slots do not encode a branch or outcome.
CUBE_C10 = (
    CubeCandidate(0, 0, -0.025, 0.000, 0.000, -1), CubeCandidate(1, 0, -0.012, 0.000, 0.010, 1),
    CubeCandidate(2, 0, 0.000, 0.015, 0.000, -1), CubeCandidate(3, 0, 0.020, 0.000, -0.010, 1),
    CubeCandidate(4, 0, 0.025, 0.000, 0.000, -1), CubeCandidate(5, 3, 0.395, 0.000, 0.000, 1),
    CubeCandidate(6, 3, 0.400, 0.000, 0.010, -1), CubeCandidate(7, 3, 0.420, 0.015, 0.000, 1),
    CubeCandidate(8, 3, 0.445, 0.005, 0.000, -1), CubeCandidate(9, 3, 0.445, -0.005, 0.000, 1),
)

# Five initial-phase and five early-rotating-phase hypotheses.  The nominal
# early rotation is 0.26*5 radians, derived from the frozen task dynamics and
# the observable current window pose; candidates remain branch-independent.
PHASE_C10 = (
    PhaseCandidate(0, 0, -0.12, -0.005, -0.005, -1), PhaseCandidate(1, 0, -0.06, 0.000, 0.005, 1),
    PhaseCandidate(2, 0, 0.00, 0.005, 0.000, -1), PhaseCandidate(3, 0, 0.06, 0.000, -0.005, 1),
    PhaseCandidate(4, 0, 0.12, -0.005, 0.005, -1), PhaseCandidate(5, 1, 2.76, -0.005, -0.005, 1),
    PhaseCandidate(6, 1, 2.76, 0.000, 0.005, -1), PhaseCandidate(7, 1, 2.76, 0.005, 0.000, 1),
    PhaseCandidate(8, 1, 2.76, 0.000, -0.005, -1), PhaseCandidate(9, 1, 2.76, -0.005, 0.005, 1),
)


def _bounded(target: torch.Tensor, proxy: torch.Tensor, step: float) -> torch.Tensor:
    return torch.clamp((target - proxy) / step, -1.0, 1.0)


def _cube_actions(base, state: dict, candidate: CubeCandidate, horizon: int) -> np.ndarray:
    proxy=state["tcp_proxy"][:, :3].clone(); endpoint=state["payload"][:, :3].clone()
    endpoint[:, 0] += candidate.endpoint_x_m; endpoint[:, 1] += candidate.lateral_y_m; endpoint[:, 2] += candidate.endpoint_z_m
    values=[]
    for step in range(horizon):
        action=torch.zeros_like(proxy) if step < candidate.wait_steps else _bounded(endpoint, proxy, base.proxy_step_size)
        proxy=proxy+action*base.proxy_step_size; values.append(action.detach().cpu().numpy().astype(np.float32,copy=False))
    return np.stack(values,axis=1)


def _phase_actions(base, state: dict, candidate: PhaseCandidate, horizon: int) -> np.ndarray:
    proxy=state["tcp_proxy"][:, :3].clone(); payload=state["payload"][:, :3].clone(); radial=state["goal_pos"][:, :2]-payload[:, :2]
    radius=torch.linalg.vector_norm(radial,dim=1)+candidate.radial_offset_m; angle=torch.atan2(radial[:,1],radial[:,0])+candidate.phase_offset_rad
    endpoint=payload.clone(); endpoint[:,0]+=radius*torch.cos(angle); endpoint[:,1]+=radius*torch.sin(angle)
    tangent=torch.stack((-torch.sin(angle),torch.cos(angle)),dim=1); endpoint[:,:2]+=candidate.tangential_offset_m*tangent; endpoint[:,2]=state["goal_pos"][:,2]
    values=[]
    for step in range(horizon):
        action=torch.zeros_like(proxy) if step < candidate.wait_steps else _bounded(endpoint, proxy, base.proxy_step_size)
        proxy=proxy+action*base.proxy_step_size; values.append(action.detach().cpu().numpy().astype(np.float32,copy=False))
    return np.stack(values,axis=1)


def _rollout(env, decision: dict, actions: np.ndarray, branch: int) -> tuple[np.ndarray,str]:
    base=env.unwrapped; base.set_state_dict(decision); base.scene._gpu_apply_all(); base.scene._gpu_fetch_all(); digest=_state_digest(base.get_state_dict()); base._elapsed_steps.zero_(); base.set_hidden_future_mode(float(branch)); info=None
    for step in range(actions.shape[1]): _,_,_,_,info=env.step(torch.as_tensor(actions[:,step],device=base.device))
    assert info is not None
    return info["success"].detach().cpu().numpy().astype(bool), digest


def _run_family(output: Path, *, family: str, task_id: str, candidates, planner, worlds: int, seed: int, horizon: int=12) -> dict:
    env=gym.make(task_id,obs_mode="state",num_envs=worlds,sim_backend="gpu",render_backend="none")
    try:
        env.reset(seed=list(range(seed,seed+worlds))); base=env.unwrapped; zero=torch.zeros((worlds,3),device=base.device)
        for _ in range(5): env.step(zero)
        decision=_clone(base.get_state_dict()); decision_digest=_state_digest(decision); records=[]; hashes={}; restored=[]
        for candidate in candidates:
            actions=planner(base,decision,candidate,horizon); hashes[str(candidate.slot)]=hashlib.sha256(actions.tobytes()).hexdigest(); values=[_rollout(env,decision,actions,b) for b in (0,1)]; restored.extend(v[1] for v in values)
            if not np.array_equal(actions,planner(base,decision,candidate,horizon)): raise AssertionError("branch replay changed action plan")
            for branch,(success,_) in enumerate(values):
                records.extend({"world_id":w,"branch":branch,"candidate_slot":candidate.slot,"success":bool(success[w])} for w in range(worlds))
        result={"schema":"milestone4r-v3-family-preprobe-v1","family":family,"worlds":worlds,"candidate_count":10,"candidate_executions":worlds*20,"rendering":False,"simulator":"ManiSkill 3 GPU PhysX","decision_state_digest":decision_digest,"restored_state_digests":restored,"candidate_grid":[asdict(x) for x in candidates],"candidate_action_hashes":hashes,"records":records}
        (output/f"{family}_preprobe_raw.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n"); return result
    finally: env.close()


def run(output_dir: str|Path) -> dict:
    output=Path(output_dir); output.mkdir(parents=True,exist_ok=True)
    results={
      "MovingCubeIntercept":_run_family(output,family="MovingCubeIntercept",task_id="ActMaskMovingCubeIntercept-v1",candidates=CUBE_C10,planner=_cube_actions,worlds=64,seed=65000),
      "FixedPhaseRotatingCaptureWindow":_run_family(output,family="FixedPhaseRotatingCaptureWindow",task_id="ActMaskRotatingTargetInteraction-v1",candidates=PHASE_C10,planner=_phase_actions,worlds=64,seed=75000),
      "SignedMovingWindowPlacement":_run_family(output,family="SignedMovingWindowPlacement",task_id="ActMaskMovingContainerPlacement-v1",candidates=PLACEMENT_C10,planner=_placement_actions,worlds=64,seed=85000),
    }
    (output/"preprobe_summary.json").write_text(json.dumps({key:{"executions":value["candidate_executions"],"worlds":value["worlds"]} for key,value in results.items()},indent=2,sort_keys=True)+"\n"); return results


if __name__ == "__main__":
    project=Path(__file__).resolve().parents[2]; print(json.dumps(run(project/"outputs"/"actmask"/"milestone4r_v3_all_family_candidate_diversity"/"preprobes_attempt_1"),indent=2))
