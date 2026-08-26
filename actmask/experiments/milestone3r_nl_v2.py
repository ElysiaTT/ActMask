"""Bounded, strict matched-pair GPU generator for 3R-NL-v2."""
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

from actmask.data import maniskill_3r_nl_tasks  # v1 task definitions, never modified here
from actmask.data.maniskill_pilot import MODEL_INPUT_KEYS, load_model_inputs
from actmask.data.milestone3q_signed import _clone, _numpy


CONFIG = Path("outputs/actmask/milestone3r_nl_v2/preregistered_config.json")
CONFIG_HASH = "9dc510bdec94e4715df96f6cc6f960f932d4dd37babea957e0f14325cf98e564"
TASKS = {
    "damped_moving_capture": "ActMaskNLMatchedDampedCapture-v1",
    "hysteretic_moving_container": "ActMaskNLMatchedHystereticContainer-v1",
    "damped_rotating_slot": "ActMaskNLMatchedDampedRotatingSlot-v1",
}
MECHANISMS = ("history_identifiable_damping_drive", "hysteretic_mode_memory")


def config_hash():
    return hashlib.sha256(CONFIG.read_bytes()).hexdigest()


def _parameters(worlds, mechanism, branch, condition):
    """Hidden task parameters; metadata-only and disjoint where predeclared."""
    base = torch.arange(worlds, dtype=torch.float32)
    frac = torch.remainder(base * .6180339, 1.)
    if mechanism == 0:
        if condition == "physical_ood":
            damping = .26 + .06 * frac
            frequency = 1.85 + .30 * torch.remainder(base * .4142135, 1.)
        else:
            damping = .12 + .06 * frac
            frequency = 1.15 + .30 * torch.remainder(base * .4142135, 1.)
        threshold = torch.zeros(worlds)
        delay = torch.zeros(worlds, dtype=torch.int64)
    else:
        damping = torch.full((worlds,), .16)
        frequency = torch.full((worlds,), 1.30)
        threshold = -.08 + .04 * frac
        # The two observable early-order modes use the frozen ID delays 0/1;
        # temporal OOD shifts the same paired modes to the disjoint 2/3 range.
        delay = torch.full((worlds,), (2 if condition == "temporal_ood" else 0) + int(branch), dtype=torch.int64)
    return damping, frequency, threshold, delay


def _early_offsets(worlds, mechanism, branch, condition):
    """Observable mode evidence; branch 1 is the exact early-frame swap."""
    base = torch.arange(worlds, dtype=torch.float32)
    scale = (.050 + .010 * torch.remainder(base * .618, 1.)) if mechanism == 0 else (.044 + .008 * torch.remainder(base * .414, 1.))
    minor = scale * (.30 if mechanism == 0 else .36)
    values = torch.stack((-scale, -minor, minor, scale), 1)
    return values if branch == 0 else torch.flip(values, (1,))


def swap_history(history):
    """Pre-registered observable pair-swap: reverse only earlier mode frames."""
    return np.concatenate((history[:, :4][:, ::-1], history[:, 4:]), axis=1).copy()


def _set_history_target(base, state, values):
    base.set_state_dict(state)
    target = base.nl_anchor.clone()
    if base.family_index == 0:
        target[:, 1] += values.to(base.device)
        base.payload.set_pose(Pose.create_from_pq(p=target))
    elif base.family_index == 1:
        target[:, 0] += values.to(base.device); target[:, 2] += .04
        base.payload.set_pose(Pose.create_from_pq(p=target))
    else:
        angle = values.to(base.device) / .13
        base.goal_pos[:, 0] = base.nl_anchor[:, 0] + .13 * torch.cos(angle)
        base.goal_pos[:, 1] = base.nl_anchor[:, 1] + .13 * torch.sin(angle)
    base.payload.set_linear_velocity(torch.zeros_like(base.payload.pose.p))
    base.payload.set_angular_velocity(torch.zeros_like(base.payload.pose.p))
    base.scene._gpu_apply_all(); base.scene._gpu_fetch_all()
    return _numpy(base.get_obs())


def _history(base, state, worlds, mechanism, branch, condition):
    early = _early_offsets(worlds, mechanism, branch, condition)
    frames = [_set_history_target(base, state, early[:, index]) for index in range(4)]
    base.set_state_dict(state); base.scene._gpu_apply_all(); base.scene._gpu_fetch_all()
    common = _numpy(base.get_obs())
    return np.stack((*frames, common, common), 1)


def _desired_target(base, step, mechanism, direction, variation):
    n = torch.full((base.num_envs,), float(step), device=base.device)
    sign = 1. if direction > 0 else -1.
    if mechanism == 0:
        response = sign * (.025 * n)
    else:
        # The candidate does not observe the actual delay; it uses a fixed
        # nominal response and must infer the compatible branch from history.
        response = sign * (.001 * n.square())
    offset = (.002 * ((variation % 3) - 1))
    target = base.nl_anchor.clone()
    if base.family_index == 0:
        target[:, 1] += response + offset
    elif base.family_index == 1:
        target[:, 0] += response + offset; target[:, 2] += .04
    else:
        angle = (response + offset) / .13
        target[:, 0] += .13 * torch.cos(angle); target[:, 1] += .13 * torch.sin(angle)
    return target


def _candidate_plan(base, state, mechanism, candidate, condition, total_candidates=10):
    """Ten shared, physically meaningful signed interception trajectories."""
    base.set_state_dict(state); base.scene._gpu_apply_all(); base.scene._gpu_fetch_all(); base._elapsed_steps.zero_()
    if total_candidates < 2 or total_candidates % 2:
        raise ValueError("candidate count must be an even integer >= 2")
    per_direction = total_candidates // 2
    # Interleave directions so aligned C5/C10 prefixes themselves remain
    # action-diverse (3:2, 5:5, and 10:10 positive:negative templates).
    direction = 1 if candidate % 2 == 0 else -1
    variation = candidate // 2
    start_delay = 0
    # For the rotating family, construct each shared candidate from its
    # explicit proposed direction.  This is a candidate template, not the
    # executed branch: both branches receive the exact same resulting action.
    if base.family_index == 2:
        damping, frequency, threshold, delay = _parameters(base.num_envs, mechanism, 0 if direction > 0 else 1, condition)
        base.set_matched_dynamics(mechanism, 0 if direction > 0 else 1, damping.to(base.device), frequency.to(base.device), threshold.to(base.device), delay.to(base.device))
    current = base.proxy.pose.p.clone(); actions = []
    for step in range(20):
        if base.family_index == 2:
            target = current if step < 4 else base.target_at(step + 1)
        else:
            target = _desired_target(base, step + 1, mechanism, direction, variation)
        action = torch.clamp((target - current) / base.proxy_step_size, -1., 1.)
        # Duration/pulse variation is common across branches and candidate IDs.
        if step == 2 + (variation % 4):
            axis = 1 if base.family_index == 0 else 0
            action[:, axis] = torch.clamp(action[:, axis] + (.006 + .001 * (variation // 5)) * direction, -1., 1.)
        current = current + action * base.proxy_step_size
        actions.append(_numpy(action))
    return np.stack(actions, 1)


def _execute(env, base, state, action, mechanism, branch, worlds, condition):
    base.set_state_dict(state); base._elapsed_steps.zero_()
    damping, frequency, threshold, delay = _parameters(worlds, mechanism, branch, condition)
    base.set_matched_dynamics(mechanism, branch, damping.to(base.device), frequency.to(base.device), threshold.to(base.device), delay.to(base.device))
    base.scene._gpu_apply_all(); base.scene._gpu_fetch_all(); success = torch.zeros(worlds, dtype=torch.bool, device=base.device)
    for step in range(action.shape[1]):
        _, _, _, _, info = env.step(torch.from_numpy(action[:, step]).to(base.device))
        success |= info["success"].bool()
    return success.cpu().numpy()


def _static(history, action, tcp):
    return np.concatenate((history[:, -1], action.mean(1), action.sum(1), tcp), 1).astype(np.float32)


def _audit(rows, history, timestamps, actions, tcp, static, labels):
    grouped = defaultdict(list)
    for index, row in enumerate(rows): grouped[row["pair_id"]].append(index)
    max_error = 0.; swap_error = 0.; differing = []
    for pair, indices in grouped.items():
        if len(indices) != 2: raise AssertionError(f"{pair}: incomplete pair")
        left, right = indices
        if rows[left]["branch"] == 1: left, right = right, left
        values = (history[left, -2:], history[right, -2:], history[left, -1], history[right, -1], actions[left], actions[right], timestamps[left], timestamps[right], tcp[left], tcp[right], static[left], static[right])
        error = max(float(np.abs(a - b).max()) for a, b in zip(values[::2], values[1::2]))
        max_error = max(max_error, error)
        swap_error = max(swap_error, float(np.abs(swap_history(history[left:left+1])[0] - history[right]).max()))
        differing.append(bool(labels[left] != labels[right]))
    if max_error > 1e-6 or swap_error > 1e-6 or not all(differing):
        raise AssertionError(dict(max_matching_error=max_error, swap_error=swap_error, label_flip_rate=float(np.mean(differing))))
    return dict(pairs=len(grouped), max_matching_error=max_error, pair_swap_max_error=swap_error, label_flip_rate=float(np.mean(differing)))


def generate(output, *, worlds, seed, condition="id", mechanisms=(0, 1), candidates=10, families=None):
    if config_hash() != CONFIG_HASH: raise RuntimeError("v2 preregistered config hash changed")
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter(); rows=[]; outcome_rows=[]; histories=[]; actions=[]; labels=[]; tcps=[]; executions=0
    allowed_families = set(TASKS if families is None else families)
    for family_index, (family, task_id) in enumerate(TASKS.items()):
        if family not in allowed_families:
            continue
        for mechanism in mechanisms:
            env = gym.make(task_id, obs_mode="state", num_envs=worlds, sim_backend="gpu", render_backend="none")
            try:
                seeds=list(range(seed + family_index * 10000 + mechanism * 1000, seed + family_index * 10000 + mechanism * 1000 + worlds)); env.reset(seed=seeds)
                base=env.unwrapped; state=_clone(base.get_state_dict()); tcp=_numpy(base.proxy.pose.p)
                for candidate in range(candidates):
                    plan=_candidate_plan(base, state, mechanism, candidate, condition, candidates); pair_history=[]; pair_labels=[]
                    for branch in (0, 1):
                        pair_history.append(_history(base, state, worlds, mechanism, branch, condition))
                        pair_labels.append(_execute(env, base, state, plan, mechanism, branch, worlds, condition)); executions += worlds
                    for world in range(worlds):
                        outcome_rows.append(dict(family=family, mechanism=MECHANISMS[mechanism], world_id=world, candidate_id=candidate, condition=condition, branch0_success=bool(pair_labels[0][world]), branch1_success=bool(pair_labels[1][world]), pair_flip=bool(pair_labels[0][world] != pair_labels[1][world])))
                        if pair_labels[0][world] == pair_labels[1][world]:
                            # A v2 counterfactual pair requires an execution flip.
                            continue
                        for branch in (0, 1):
                            histories.append(pair_history[branch][world:world+1]); actions.append(plan[world:world+1]); labels.append(pair_labels[branch][world:world+1]); tcps.append(tcp[world:world+1])
                            damping, frequency, threshold, delay = _parameters(worlds, mechanism, branch, condition)
                            rows.append(dict(family=family, mechanism=MECHANISMS[mechanism], world_id=world, seed=seeds[world], candidate_id=candidate, branch=branch, pair_id=f"{condition}:{family}:{mechanism}:{world}:{candidate}", split=("train" if world % 5 < 3 else "validation" if world % 5 == 3 else "test"), condition=condition, success=bool(pair_labels[branch][world]), hidden_parameters=dict(damping=float(damping[world]), drive_frequency=float(frequency[world]), threshold=float(threshold[world]), execution_delay=int(delay[world]))))
            finally:
                env.close()
    if not rows: raise AssertionError("no valid v2 pairs")
    history=np.concatenate(histories); action=np.concatenate(actions); label=np.concatenate(labels); tcp=np.concatenate(tcps)
    timestamps=np.tile(np.linspace(-.25, 0., 6, dtype=np.float32), (len(history), 1)); static=_static(history, action, tcp); audit=_audit(rows, history, timestamps, action, tcp, static, label)
    np.savez_compressed(output / "model_inputs.npz", history=history, timestamps=timestamps, visibility=np.ones((len(history), 6, 1), np.float32), observation_confidence=np.ones((len(history), 6, 1), np.float32), candidate_actions=action, nominal_action_timing=np.tile(np.arange(20, dtype=np.float32) / 20, (len(history), 1)), tcp_state=tcp)
    np.savez_compressed(output / "labels.npz", success=label, static_features=static)
    (output / "metadata.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    (output / "candidate_outcomes.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in outcome_rows)); load_model_inputs(output / "model_inputs.npz")
    report=dict(config_sha256=config_hash(), condition=condition, worlds=worlds, candidates=candidates, mechanisms=[MECHANISMS[x] for x in mechanisms], candidate_executions=executions, accepted_pairs=audit["pairs"], examples=int(len(label)), label_balance=float(label.mean()), audit=audit, generation_seconds=time.perf_counter()-started)
    (output / "generation_report.json").write_text(json.dumps(report, indent=2)); return report
