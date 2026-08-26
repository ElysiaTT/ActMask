"""Strict matched-final GPU pair generator for the frozen 3R-NL v1 config."""
from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from mani_skill.utils.structs import Pose

from actmask.data import maniskill_3r_nl_tasks  # registers v1 tasks
from actmask.data.maniskill_pilot import MODEL_INPUT_KEYS, load_model_inputs
from actmask.data.milestone3q_signed import _clone, _numpy


CONFIG = Path("outputs/actmask/milestone3r_nl_v1/preregistered_config.json")
TASKS = {
    "damped_moving_capture": "ActMaskNLMatchedDampedCapture-v1",
    "hysteretic_moving_container": "ActMaskNLMatchedHystereticContainer-v1",
    "damped_rotating_slot": "ActMaskNLMatchedDampedRotatingSlot-v1",
}
MECHANISMS = ("history_identifiable_damping_drive", "hysteretic_mode_memory")
# This is a physical-validity result from the pre-model probe, not a learned
# filtering rule.  With the two allowed common physical corrections already
# consumed, the rotating-slot damping/drive combination still has no strict
# label-flipping pairs.  The remaining five task/mechanism cells are fixed
# before full generation: 500 worlds * 20 candidates * 2 branches = 100,000
# GPU-PhysX executions exactly.
FULL_ACTIVE_CELLS = frozenset({
    ("damped_moving_capture", "history_identifiable_damping_drive"),
    ("damped_moving_capture", "hysteretic_mode_memory"),
    ("hysteretic_moving_container", "history_identifiable_damping_drive"),
    ("hysteretic_moving_container", "hysteretic_mode_memory"),
    ("damped_rotating_slot", "hysteretic_mode_memory"),
})


def config_hash(path: str | Path = CONFIG) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _parameters(worlds: int, mechanism: int, branch: int, *, ood: bool = False):
    base = torch.arange(worlds, dtype=torch.float32)
    if mechanism == 0:
        damping = (0.26 if branch else 0.12) + 0.06 * torch.remainder(base * 0.618, 1.0)
        frequency = (1.85 if branch else 1.15) + 0.30 * torch.remainder(base * 0.414, 1.0)
        threshold = torch.full((worlds,), 0.0)
        delay = torch.zeros(worlds, dtype=torch.int64)
        if ood:
            damping += 0.14
            frequency += 0.70
    else:
        damping = torch.full((worlds,), 0.16)
        frequency = torch.full((worlds,), 1.30)
        threshold = (-0.08 if branch == 0 else 0.04) + 0.04 * torch.remainder(base * 0.414, 1.0)
        delay = torch.full((worlds,), branch, dtype=torch.int64)
        if ood:
            threshold += 0.12
            delay += 2
    return damping, frequency, threshold, delay


def _history_offsets(worlds: int, mechanism: int, branch: int):
    # NL-F1 fairness correction before *any* model evaluation: both branches
    # must have the exact same unordered early-frame multiset.  Their temporal
    # order encodes drive phase / threshold-crossing direction, while the
    # hidden post-decision dynamics remain distinct.
    if mechanism == 0:
        common = torch.tensor([-0.066, -0.020, 0.024, 0.062], dtype=torch.float32)
    else:
        common = torch.tensor([-0.060, -0.018, 0.022, 0.058], dtype=torch.float32)
    values = common if branch == 0 else torch.flip(common, dims=(0,))
    return values[None, :].expand(worlds, -1).clone()


def _set_history_target(base, decision, values: torch.Tensor):
    base.set_state_dict(decision)
    target = base.nl_anchor.clone()
    if base.family_index == 0:
        target[:, 1] += values.to(base.device)
        base.payload.set_pose(Pose.create_from_pq(p=target))
    elif base.family_index == 1:
        target[:, 0] += values.to(base.device)
        target[:, 2] += 0.04
        base.payload.set_pose(Pose.create_from_pq(p=target))
    else:
        angle = values.to(base.device) / 0.13
        base.goal_pos[:, 0] = base.nl_anchor[:, 0] + .13 * torch.cos(angle)
        base.goal_pos[:, 1] = base.nl_anchor[:, 1] + .13 * torch.sin(angle)
    base.payload.set_linear_velocity(torch.zeros_like(base.payload.pose.p))
    base.payload.set_angular_velocity(torch.zeros_like(base.payload.pose.p))
    base.scene._gpu_apply_all(); base.scene._gpu_fetch_all()
    return _numpy(base.get_obs())


def _history(base, decision, worlds: int, mechanism: int, branch: int):
    early = _history_offsets(worlds, mechanism, branch)
    frames = [_set_history_target(base, decision, early[:, frame]) for frame in range(4)]
    # The synchronization segment is a common exact state, velocity, and time.
    base.set_state_dict(decision); base.scene._gpu_apply_all(); base.scene._gpu_fetch_all(); common = _numpy(base.get_obs())
    frames.extend((common, common))
    return np.stack(frames, axis=1)


def _plan(base, mechanism: int, worlds: int, *, candidate: int = 0):
    damping, frequency, threshold, delay = _parameters(worlds, mechanism, 0)
    base.set_matched_dynamics(mechanism, 0, damping.to(base.device), frequency.to(base.device), threshold.to(base.device), delay.to(base.device))
    current = base.proxy.pose.p.clone(); actions=[]
    pulse = torch.tensor([
        0.004 * ((candidate % 5) - 2),
        0.004 * (((candidate // 5) % 4) - 1.5),
        0.003 * ((candidate % 2) - .5),
    ], device=base.device)
    for step in range(20):
        if base.family_index == 2:
            # Rotating-slot templates deliberately avoid the pre-divergence
            # contact window. Both are fixed and shared by hidden branches.
            delay = 4
            target = current if step < delay else base.target_at(step + 1)
        else:
            target = base.target_at(step + 1)
        action = torch.clamp((target-current) / base.proxy_step_size, -1.0, 1.0)
        # Fixed candidate identity is represented solely by this common short
        # control pulse; subsequent feedback returns to the same target path.
        if step == 2:
            action = torch.clamp(action + pulse, -1.0, 1.0)
        current = current + action * base.proxy_step_size
        actions.append(_numpy(action))
    return np.stack(actions, axis=1)


def _execute(env, base, decision, action, mechanism: int, branch: int, worlds: int, *, ood: bool = False):
    base.set_state_dict(decision)
    base._elapsed_steps.zero_()
    damping, frequency, threshold, delay = _parameters(worlds, mechanism, branch, ood=ood)
    base.set_matched_dynamics(mechanism, branch, damping.to(base.device), frequency.to(base.device), threshold.to(base.device), delay.to(base.device))
    base.scene._gpu_apply_all(); base.scene._gpu_fetch_all(); accepted=torch.zeros(worlds,dtype=torch.bool,device=base.device)
    for step in range(action.shape[1]):
        _, _, _, _, info = env.step(torch.from_numpy(action[:,step]).to(base.device))
        accepted |= info['success'].bool()
    return accepted.cpu().numpy()


def _static(history, action, tcp):
    return np.concatenate((history[:, -1], action.mean(1), action.sum(1), tcp), axis=1).astype(np.float32)


def _audit(rows, history, timestamps, actions, tcp, static, labels, tolerance=1e-6):
    grouped=defaultdict(list)
    for index,row in enumerate(rows): grouped[row['signed_group']].append(index)
    accepted=[]
    for group,members in grouped.items():
        if len(members)!=2: raise AssertionError(f'{group}: not a pair')
        left,right=members
        checks=dict(
            final_two=float(np.abs(history[left,-2:]-history[right,-2:]).max()),
            current=float(np.abs(history[left,-1]-history[right,-1]).max()),
            action=float(np.abs(actions[left]-actions[right]).max()),
            timestamps=float(np.abs(timestamps[left,-2:]-timestamps[right,-2:]).max()),
            tcp=float(np.abs(tcp[left]-tcp[right]).max()),
            static=float(np.abs(static[left]-static[right]).max()),
            last_two_input=float(np.abs(history[left,-2:]-history[right,-2:]).max()),
            earlier=float(np.abs(history[left,:-2]-history[right,:-2]).max()),
        )
        if any(checks[key]>tolerance for key in ('final_two','current','action','timestamps','tcp','static','last_two_input')): raise AssertionError(f'{group}: matched-final audit failed {checks}')
        if checks['earlier']<=tolerance: raise AssertionError(f'{group}: history lacks early difference')
        if labels[left]==labels[right]: raise AssertionError(f'{group}: execution did not flip')
        accepted.append(checks)
    return dict(groups=len(accepted),tolerance=tolerance,max_matching_error=max((max(x[k] for k in ('final_two','current','action','timestamps','tcp','static','last_two_input')) for x in accepted),default=0.0),min_earlier_history_difference=min((x['earlier'] for x in accepted),default=0.0))


def generate(output_root: str | Path, *, worlds: int, seed: int, candidates: int = 2, ood: bool = False, active_cells=None):
    """Generate strict matched-final pairs. ``worlds`` is per task/mechanism."""
    if config_hash() != '79d04d3204af01386d1a178a874d537a345bd2a8ca94545e750fcb4d50209b9f':
        raise RuntimeError('preregistered config hash changed')
    output=Path(output_root);output.mkdir(parents=True,exist_ok=True);started=time.perf_counter();rows=[];histories=[];actions=[];labels=[];tcps=[];attempted=0
    active_cells = None if active_cells is None else frozenset(active_cells)
    rejection_reasons=defaultdict(int)
    for family_index,(family,task_id) in enumerate(TASKS.items()):
        for mechanism,name in enumerate(MECHANISMS):
            if active_cells is not None and (family, name) not in active_cells:
                continue
            env=gym.make(task_id,obs_mode='state',num_envs=worlds,sim_backend='gpu',render_backend='none')
            try:
                seeds=list(range(seed+family_index*10000+mechanism*1000,seed+family_index*10000+mechanism*1000+worlds));env.reset(seed=seeds);base=env.unwrapped;decision=_clone(base.get_state_dict());decision_tcp=_numpy(base.proxy.pose.p)
                for candidate in range(candidates):
                    # NL-F2: candidate templates must all start from the same
                    # decision state; otherwise prior execution leaks into the
                    # next candidate's common action construction.
                    base.set_state_dict(decision);base.scene._gpu_apply_all();base.scene._gpu_fetch_all()
                    base._elapsed_steps.zero_()
                    plan=_plan(base,mechanism,worlds,candidate=candidate);pair_history=[];pair_success=[]
                    for branch in (0,1):
                        pair_history.append(_history(base,decision,worlds,mechanism,branch))
                        pair_success.append(_execute(env,base,decision,plan,mechanism,branch,worlds,ood=ood))
                    attempted += worlds
                    for world in range(worlds):
                        if pair_success[0][world] == pair_success[1][world]:
                            rejection_reasons['label_nonflip'] += 1
                            continue
                        for branch in (0,1):
                            histories.append(pair_history[branch][world:world+1]);actions.append(plan[world:world+1]);labels.append(pair_success[branch][world:world+1]);tcps.append(decision_tcp[world:world+1])
                            rows.append(dict(family=family,mechanism=name,world_id=world,seed=seeds[world],candidate_id=candidate,signed_group=f'{family}:{name}:{world}:{candidate}',branch=branch,split=('train' if world%5<3 else 'val' if world%5==3 else 'test'),success=bool(pair_success[branch][world])))
            finally:
                env.close()
    if not rows:
        raise AssertionError('no strict matched-final groups were accepted')
    history=np.concatenate(histories);action=np.concatenate(actions);label=np.concatenate(labels);tcp=np.concatenate(tcps);timestamps=np.tile(np.linspace(-.25,0.,6,dtype=np.float32),(len(history),1));static=_static(history,action,tcp);audit=_audit(rows,history,timestamps,action,tcp,static,label)
    np.savez_compressed(output/'model_inputs.npz',history=history,timestamps=timestamps,visibility=np.ones((len(history),6,1),np.float32),observation_confidence=np.ones((len(history),6,1),np.float32),candidate_actions=action,nominal_action_timing=np.tile(np.arange(20,dtype=np.float32)/20,(len(history),1)),tcp_state=tcp)
    np.savez_compressed(output/'labels.npz',success=label,static_features=static)
    (output/'metadata.jsonl').write_text(''.join(json.dumps(row,sort_keys=True)+'\n' for row in rows))
    load_model_inputs(output/'model_inputs.npz')
    report=dict(config_sha256=config_hash(),worlds_per_task_mechanism=worlds,candidates_per_world=candidates,attempted_groups=attempted,attempted_candidate_executions=attempted*2,accepted_groups=audit['groups'],rejected_groups=attempted-audit['groups'],rejection_reasons=dict(rejection_reasons),examples=int(len(label)),label_balance=float(label.mean()),audit=audit,generation_seconds=time.perf_counter()-started,groups_by_family_mechanism={f'{family}:{mechanism}':sum(1 for row in rows if row['family']==family and row['mechanism']==mechanism and row['branch']==0) for family in TASKS for mechanism in MECHANISMS},active_cells=sorted(':'.join(x) for x in (active_cells or {(family, mechanism) for family in TASKS for mechanism in MECHANISMS})),ood=ood)
    (output/'generation_report.json').write_text(json.dumps(report,indent=2));return report
