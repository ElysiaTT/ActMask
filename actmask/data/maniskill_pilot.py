"""GPU state-only ManiSkill pilot generation and input-isolation helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import gymnasium as gym
import numpy as np
import torch
from mani_skill.utils.structs import Pose

# Importing this module registers the three custom environments with ManiSkill.
from actmask.data import maniskill_state_tasks as _registered_tasks  # noqa: F401


TASK_IDS = {
    "moving_cube_intercept": "ActMaskMovingCubeIntercept-v1",
    "moving_container_placement": "ActMaskMovingContainerPlacement-v1",
    "rotating_target_interaction": "ActMaskRotatingTargetInteraction-v1",
}
REGIMES = ("original", "static_balanced", "matched_counterfactual")
MECHANISMS = {
    "moving_cube_intercept": ("timing", "path", "phase"),
    "moving_container_placement": ("acquire", "transport", "release"),
    "rotating_target_interaction": ("phase_lock", "stale_phase", "anti_phase"),
}
MODEL_INPUT_KEYS = (
    "history",
    "timestamps",
    "visibility",
    "observation_confidence",
    "candidate_actions",
    "nominal_action_timing",
    "tcp_state",
)
FORBIDDEN_MODEL_INPUT_KEYS = {
    "success",
    "label",
    "mechanism",
    "regime",
    "candidate_id",
    "world_id",
    "seed",
    "task_phase",
    "target_angle",
    "future_trajectory",
    "future_contact",
    "future_dynamics",
}


@dataclass(frozen=True)
class CandidateSpec:
    mechanism: str
    intended_success: bool
    candidate_id: int


def candidate_specs(task_name: str) -> tuple[CandidateSpec, ...]:
    """Two matched-budget candidates for each of three causal mechanisms."""
    specs = []
    for mechanism in MECHANISMS[task_name]:
        specs.append(CandidateSpec(mechanism, True, len(specs)))
        specs.append(CandidateSpec(mechanism, False, len(specs)))
    return tuple(specs)


def load_model_inputs(path: str | Path) -> dict[str, np.ndarray]:
    """Load only fields that a predictor is allowed to consume."""
    with np.load(path) as arrays:
        keys = set(arrays.files)
        if keys != set(MODEL_INPUT_KEYS):
            raise ValueError(f"Unexpected model-input fields: {sorted(keys)}")
        if keys & FORBIDDEN_MODEL_INPUT_KEYS:
            raise ValueError("A hidden field was included in model input")
        return {key: arrays[key] for key in MODEL_INPUT_KEYS}


def _bounded_delta(target: torch.Tensor, position: torch.Tensor, step_size: float) -> torch.Tensor:
    return torch.clamp((target - position) / step_size, -1.0, 1.0)


def _failure_target(task_name: str, mechanism: str, context: dict[str, torch.Tensor]) -> torch.Tensor:
    payload = context["initial_payload"]
    proxy = context["initial_proxy"]
    device = payload.device
    if task_name == "moving_cube_intercept":
        if mechanism == "timing":
            # Remain on the approach side rather than crossing and pushing the
            # moving payload along with the proxy.
            return proxy + torch.tensor(
                [-0.42, 0.0, 0.0], device=device
            )
        offsets = {
            "path": torch.tensor([0.0, 0.42, 0.0], device=device),
            "phase": torch.tensor([0.0, 0.0, 0.42], device=device),
        }
        return payload + offsets[mechanism]
    if task_name == "moving_container_placement":
        if mechanism == "acquire":
            return context["initial_container"]
        if mechanism == "transport":
            return payload
        return context["initial_container"] + torch.tensor([0.0, 0.34, 0.0], device=device)
    if mechanism == "stale_phase":
        return context["initial_goal"] + torch.tensor([0.42, 0.0, 0.0], device=device)
    if mechanism == "anti_phase":
        return 2 * payload - context["initial_goal"]
    return context["initial_goal"] + torch.tensor([0.42, 0.0, 0.0], device=device)


def _candidate_action(
    base,
    task_name: str,
    spec: CandidateSpec,
    step: int,
    horizon: int,
    regime: str,
    context: dict[str, torch.Tensor],
    policy_success: bool,
) -> torch.Tensor:
    """A controlled-policy family with matched horizon and action budget."""
    payload = context["initial_payload"]
    proxy = context["planned_proxy"]
    if policy_success:
        if task_name == "moving_container_placement":
            # All positive candidates execute acquire -> carry -> release. Their
            # acquire timing differs by mechanism but the action budget is fixed.
            acquire_steps = {"acquire": 5, "transport": 4, "release": 6}[spec.mechanism]
            target = payload if step < acquire_steps else context["initial_container"]
        elif task_name == "rotating_target_interaction":
            # The delayed mechanism is still feasible; the stale/anti variants
            # are represented by the corresponding matched negative candidate.
            delay = 2 if spec.mechanism == "stale_phase" else 0
            target = proxy if step < delay else context["initial_goal"]
        else:
            delay = 2 if spec.mechanism == "timing" else 0
            target = proxy if step < delay else payload
    else:
        target = _failure_target(task_name, spec.mechanism, context)

    action = _bounded_delta(target, proxy, base.proxy_step_size)
    if regime == "original":
        # Deterministic world-level variation gives several causes of outcomes
        # while preserving the same candidate/action interface.
        noise = 0.035 * torch.sin(
            context["world_phase"][:, None] + float(step) * torch.tensor([0.7, 1.1, 1.3], device=base.device)
        )
        action = torch.clamp(action + noise, -1.0, 1.0)
    context["planned_proxy"] = proxy + action * base.proxy_step_size
    return action


def _as_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().to("cpu").numpy().astype(np.float32, copy=False)


def _clone_state(state: dict) -> dict:
    """The custom tasks expose a flat tensor-only replay state."""
    return {key: value.clone() for key, value in state.items()}


def _run_candidate(
    env,
    task_name: str,
    spec: CandidateSpec,
    regime: str,
    seeds: list[int],
    horizon: int,
    *,
    policy_success: bool,
    hidden_future_mode: float,
    history_mode: int,
    history_frames: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    observation, _ = env.reset(seed=seeds)
    # Build an observable pre-decision history before any oracle dynamics are
    # selected. The future event is scheduled only after this decision point.
    history = [_as_numpy(observation)]
    zero_action = torch.zeros((len(seeds), 3), dtype=torch.float32, device=env.unwrapped.device)
    for _ in range(history_frames - 1):
        observation, _, _, _, _ = env.step(zero_action)
        history.append(_as_numpy(observation))
    base = env.unwrapped
    decision_state = _clone_state(base.get_state_dict())
    if history_mode:
        # Re-simulate a visibly different preceding trajectory, then restore
        # the exact same state at the decision time. The history is fair model
        # input; the restored state and candidate plan are the static match.
        observation, _ = env.reset(seed=seeds)
        history = [_as_numpy(observation)]
        for frame in range(history_frames - 1):
            offset = 0.08 * float(history_frames - frame - 1)
            shifted = base.payload.pose.p.clone()
            shifted[:, 1] += offset
            base.payload.set_pose(Pose.create_from_pq(p=shifted))
            base.payload.set_linear_velocity(torch.zeros_like(shifted))
            base.payload.set_angular_velocity(torch.zeros_like(shifted))
            observation, _, _, _, _ = env.step(zero_action)
            history.append(_as_numpy(observation))
        base.set_state_dict(decision_state)
        if base.gpu_sim_enabled:
            base.scene._gpu_apply_all()
            base.scene._gpu_fetch_all()
        history[-1] = _as_numpy(base.get_obs())
    base.set_hidden_future_mode(hidden_future_mode)
    context = dict(
        initial_goal=base.goal_pos.clone(),
        initial_proxy=base.proxy.pose.p.clone(),
        initial_payload=base.payload.pose.p.clone(),
        planned_proxy=base.proxy.pose.p.clone(),
        world_phase=torch.tensor(seeds, dtype=torch.float32, device=base.device) * 0.013,
    )
    if hasattr(base, "container"):
        context["initial_container"] = base.container.pose.p.clone()
    actions = []
    info = None
    for step in range(horizon):
        action = _candidate_action(
            base, task_name, spec, step, horizon, regime, context, policy_success
        )
        observation, _, _, _, info = env.step(action)
        actions.append(_as_numpy(action))
    assert info is not None
    success = info["success"].detach().to("cpu").numpy().astype(np.bool_)
    final_proxy_radius = torch.linalg.vector_norm(
        base.proxy.pose.p - base.payload.pose.p, dim=1
    )
    stats = dict(
        success_rate=float(success.mean()),
        final_proxy_radius=float(final_proxy_radius.float().mean().cpu()),
        mean_action_norm=float(np.linalg.norm(np.stack(actions), axis=-1).mean()),
    )
    static_features = torch.stack(
        [
            torch.linalg.vector_norm(context["initial_payload"] - context["initial_proxy"], dim=1),
            torch.linalg.vector_norm(context["initial_goal"] - context["initial_proxy"], dim=1),
            torch.linalg.vector_norm(context["planned_proxy"] - context["initial_goal"], dim=1),
            torch.full((len(seeds),), float(horizon), device=base.device),
        ],
        dim=1,
    )
    return (
        np.stack(history, axis=1),
        np.stack(actions, axis=1),
        success,
        _as_numpy(static_features),
        _as_numpy(context["initial_proxy"]),
        stats,
    )


def generate_pilot(
    output_dir: str | Path,
    *,
    worlds_per_task: int = 108,
    horizon: int = 12,
    seed: int = 3107,
) -> dict:
    """Generate the <=10k-example, GPU-vectorized simulator pilot.

    ``worlds_per_task`` must divide equally across the three requested regimes.
    Six candidates per base world give one intended success and one intended
    failure for every task mechanism.
    """
    if worlds_per_task < 100 or worlds_per_task % len(REGIMES):
        raise ValueError("worlds_per_task must be >=100 and divisible by three")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    worlds_per_regime = worlds_per_task // len(REGIMES)
    total_examples = worlds_per_task * len(candidate_specs("moving_cube_intercept")) * len(TASK_IDS)
    if total_examples > 10_000:
        raise ValueError(f"Pilot would contain {total_examples} examples (>10k cap)")

    summary = dict(
        schema_version=1,
        worlds_per_task=worlds_per_task,
        worlds_per_regime=worlds_per_regime,
        candidates_per_world=6,
        horizon=horizon,
        total_examples=total_examples,
        visual_observations=False,
        tasks={},
    )
    for task_index, (task_name, task_id) in enumerate(TASK_IDS.items()):
        history_all: list[np.ndarray] = []
        actions_all: list[np.ndarray] = []
        labels_all: list[np.ndarray] = []
        static_features_all: list[np.ndarray] = []
        tcp_states_all: list[np.ndarray] = []
        metadata: list[dict] = []
        task_summary = dict(regimes={})
        env = gym.make(
            task_id,
            obs_mode="state",
            num_envs=worlds_per_regime,
            sim_backend="gpu",
            render_backend="none",
        )
        try:
            for regime_index, regime in enumerate(REGIMES):
                world_start = regime_index * worlds_per_regime
                world_ids = np.arange(world_start, world_start + worlds_per_regime)
                seeds = (seed + task_index * 10_000 + world_ids).tolist()
                regime_summary = dict()
                for spec in candidate_specs(task_name):
                    paired = regime == "matched_counterfactual"
                    matched_static = regime == "static_balanced"
                    policy_success = spec.intended_success if regime == "original" else True
                    # The original distribution preserves naturally correlated
                    # candidate plans without an oracle future intervention.
                    # Static-balanced and matched distributions use the hidden
                    # post-decision event to balance/flip labels fairly.
                    hidden_future_mode = (
                        0.0
                        if regime == "original" or spec.intended_success
                        else 1.0
                    )
                    # Negative examples retain a different *observable*
                    # pre-decision motion history in every regime.  The
                    # decision state and frozen candidate plan are restored
                    # exactly before rollout, so this does not create a
                    # static/action leak while letting a temporal method learn
                    # from the original distribution as well.
                    history_mode = 0 if spec.intended_success else 1
                    history, actions, success, static_features, tcp_state, stats = _run_candidate(
                        env,
                        task_name,
                        spec,
                        regime,
                        seeds,
                        horizon,
                        policy_success=policy_success,
                        hidden_future_mode=hidden_future_mode,
                        history_mode=history_mode,
                    )
                    history_all.append(history)
                    actions_all.append(actions)
                    labels_all.append(success)
                    static_features_all.append(static_features)
                    tcp_states_all.append(tcp_state)
                    regime_summary[spec.mechanism + ("_positive" if spec.intended_success else "_negative")] = stats
                    for batch_idx, world_id in enumerate(world_ids):
                        metadata.append(
                            dict(
                                task=task_name,
                                regime=regime,
                                mechanism=spec.mechanism,
                                intended_success=spec.intended_success,
                                world_id=int(world_id),
                                candidate_id=spec.candidate_id,
                                seed=int(seeds[batch_idx]),
                                label_success=bool(success[batch_idx]),
                                pair_group=(
                                    f"{task_name}:{spec.mechanism}:{int(world_id)}"
                                    if paired
                                    else None
                                ),
                                static_match_group=(
                                    f"{task_name}:{spec.mechanism}:{int(world_id)}"
                                    if matched_static else None
                                ),
                                split="pilot",
                                static_features=static_features[batch_idx].round(7).tolist(),
                            )
                        )
                task_summary["regimes"][regime] = regime_summary
        finally:
            env.close()

        history_np = np.concatenate(history_all, axis=0)
        actions_np = np.concatenate(actions_all, axis=0)
        labels_np = np.concatenate(labels_all, axis=0)
        static_features_np = np.concatenate(static_features_all, axis=0)
        tcp_state_np = np.concatenate(tcp_states_all, axis=0)
        model_path = output_dir / f"{task_name}_model_inputs.npz"
        label_path = output_dir / f"{task_name}_labels.npz"
        metadata_path = output_dir / f"{task_name}_metadata.jsonl"
        timestamps = np.tile(
            np.linspace(-0.15, 0.0, history_np.shape[1], dtype=np.float32),
            (history_np.shape[0], 1),
        )
        visibility = np.ones((history_np.shape[0], history_np.shape[1], 1), dtype=np.float32)
        confidence = np.ones_like(visibility)
        action_timing = np.tile(
            np.arange(horizon, dtype=np.float32) / 20.0,
            (history_np.shape[0], 1),
        )
        np.savez_compressed(
            model_path,
            history=history_np,
            timestamps=timestamps,
            visibility=visibility,
            observation_confidence=confidence,
            candidate_actions=actions_np,
            nominal_action_timing=action_timing,
            tcp_state=tcp_state_np,
        )
        np.savez_compressed(label_path, success=labels_np, static_features=static_features_np)
        with metadata_path.open("w", encoding="utf-8") as handle:
            for item in metadata:
                handle.write(json.dumps(item, sort_keys=True) + "\n")
        # Re-open through the strict adapter as a generation-time leakage gate.
        inputs = load_model_inputs(model_path)
        if inputs["history"].shape[0] != len(metadata) or labels_np.shape[0] != len(metadata):
            raise RuntimeError("Pilot array/metadata alignment failed")
        task_summary.update(
            examples=int(labels_np.shape[0]),
            observed_success_rate=float(labels_np.mean()),
            model_input_path=model_path.name,
            label_path=label_path.name,
            metadata_path=metadata_path.name,
            history_shape=list(history_np.shape[1:]),
            action_shape=list(actions_np.shape[1:]),
        )
        summary["tasks"][task_name] = task_summary

    summary_path = output_dir / "pilot_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary
