"""GPU-PhysX, no-render generator for the 5B-0 relational dynamics probe.

The simulator determines every C10 success label.  The paired observation
histories are constructed from the same two physical object trajectories with
only their (unobserved) identity association altered.  Sorting object tokens
therefore makes every preregistered global moment exactly identical across a
pair, while an identity tracker can still select the relation-appropriate
candidate family.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from actmask.data import maniskill_state_tasks as _registered  # noqa: F401
from actmask.data.milestone4r_v2_preprobe import C10 as CONTACT_C10
from actmask.data.milestone4r_v2_preprobe import _planned_actions as contact_actions
from actmask.data.milestone4r_v2_preprobe import run as contact_preprobe
from actmask.data.milestone4r_v3_preprobe import CUBE_C10, _cube_actions, _run_family


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "actmask" / "milestone5b0_relational_probe"
STATE = OUT / "state_probe"
CONFIG = OUT / "preregistered_config.json"
TASKS = (
    ("TwoObjectIdentitySwapIntercept", "ActMaskMovingCubeIntercept-v1", CUBE_C10, _cube_actions, 11000),
    ("CounterMovingRelationalContact", "ActMaskMovingContainerPlacement-v1", CONTACT_C10, contact_actions, 12000),
)


def _clone(state: dict) -> dict:
    return {key: value.clone() for key, value in state.items()}


def _plans(task_id: str, candidates, planner, seed: int, worlds: int = 64) -> np.ndarray:
    """Recreate only the branch-independent plans used by the logged rollout."""
    env = gym.make(task_id, obs_mode="state", num_envs=worlds, sim_backend="gpu", render_backend="none")
    try:
        env.reset(seed=list(range(seed, seed + worlds)))
        base = env.unwrapped
        zero = torch.zeros((worlds, 3), device=base.device)
        for _ in range(5):
            env.step(zero)
        decision = _clone(base.get_state_dict())
        values = [planner(base, decision, candidate, 12) for candidate in candidates]
        return np.stack(values, axis=1).astype(np.float32)  # world, C10, time, xyz
    finally:
        env.close()


def _trajectory(task_index: int, world: int) -> np.ndarray:
    """Two physically plausible, crossing planar trajectories, T x 2 x 3."""
    t = np.linspace(-1.0, 1.0, 6, dtype=np.float32)
    centre = np.array([(world % 8 - 3.5) * .012, (world // 8 - 3.5) * .009, .08], np.float32)
    scale = .065 + .002 * (world % 5)
    if task_index == 0:
        delta = np.stack((scale * t, .035 * np.sin(np.pi * t), np.zeros_like(t)), axis=1)
    else:
        delta = np.stack((.045 * np.sin(np.pi * t), scale * t, .012 * np.cos(np.pi * t)), axis=1)
    first, second = centre + delta, centre - delta
    # The final decision frame is exactly the same canonical presentation.
    return np.stack((first, second), axis=1).astype(np.float32)


def _observed_pair(base: np.ndarray, branch: int, cue_visible: bool) -> np.ndarray:
    """Fair tokens: association is visible for exactly half of base worlds.

    Token ordering is a detector ordering, not an object identity.  At the
    final frame it is canonicalized for both branches.  The oracle-only field
    stores the true association, which is never included in fair features.
    """
    result = base.copy()
    if branch and cue_visible:
        result[:-1] = result[:-1, ::-1]
    return result


def _sorted_tokens(history: np.ndarray) -> np.ndarray:
    # Stable lexicographic order gives an explicit unordered-token canonical form.
    result = np.empty_like(history)
    for i, frame in enumerate(history):
        result[i] = frame[np.lexsort((frame[:, 2], frame[:, 1], frame[:, 0]))]
    return result


def _moments(history: np.ndarray) -> dict[str, np.ndarray]:
    unordered = _sorted_tokens(history)
    centroid = unordered.mean(1)
    velocity = np.diff(centroid, axis=0, prepend=centroid[:1])
    acceleration = np.diff(velocity, axis=0, prepend=velocity[:1])
    centered = unordered - centroid[:, None]
    covariance = np.einsum("tni,tnj->tij", centered, centered) / 2.0
    pair_distance = np.linalg.norm(unordered[:, 0] - unordered[:, 1], axis=1, keepdims=True)
    token_velocity = np.diff(unordered, axis=0, prepend=unordered[:1])
    speed = np.sort(np.linalg.norm(token_velocity, axis=2), axis=1)
    displacement = np.sort(np.linalg.norm(unordered[-1] - unordered[0], axis=1))
    return {"current_unordered_state": unordered[-1], "centroid": centroid, "centroid_velocity": velocity,
            "centroid_acceleration": acceleration, "covariance": covariance,
            "pairwise_distance_multiset": pair_distance, "speed_magnitude_multiset": speed,
            "total_displacement": displacement, "unordered_token_multiset": unordered}


def _audit_pair(first: np.ndarray, second: np.ndarray, actions_equal: bool) -> dict:
    one, two = _moments(first), _moments(second)
    errors = {key: float(np.max(np.abs(one[key] - two[key]))) for key in one}
    errors["candidate_actions"] = 0.0 if actions_equal else float("inf")
    return {"max_abs_error": errors, "pass": all(value == 0.0 for value in errors.values())}


def _split(world: int) -> str:
    return "train" if world < 39 else "validation" if world < 52 else "test"


def _run_labels(task: tuple, output: Path) -> dict:
    name, task_id, candidates, planner, seed = task
    output.mkdir(parents=True, exist_ok=True)
    if name == "TwoObjectIdentitySwapIntercept":
        return _run_family(output, family=name, task_id=task_id, candidates=candidates, planner=planner, worlds=64, seed=seed)
    # This preprobe's output directory is nested; normalize its payload below.
    return contact_preprobe(output / name, worlds=64, seed=seed)


def run(output_dir: str | Path = OUT) -> dict:
    output = Path(output_dir); state_dir = output / "state_probe"
    state_dir.mkdir(parents=True, exist_ok=True)
    config_hash = hashlib.sha256((output / "preregistered_config.json").read_bytes()).hexdigest()
    histories: list[np.ndarray] = []; oracle_histories: list[np.ndarray] = []; actions_all: list[np.ndarray] = []
    worlds_json: list[dict] = []; labels: list[dict] = []; pairs: list[dict] = []; audits: list[dict] = []; raw_logs: dict[str, dict] = {}
    for task_index, task in enumerate(TASKS):
        name, task_id, candidates, planner, seed = task
        raw = _run_labels(task, state_dir / "raw_candidate_execution_logs")
        raw_logs[name] = raw
        plans = _plans(task_id, candidates, planner, seed)
        actions_all.append(plans)
        by_key = {(r["world_id"], r["branch"], r["candidate_slot"]): bool(r["success"]) for r in raw["records"]}
        for world in range(64):
            base = _trajectory(task_index, world)
            # Alternating worlds creates a preregistered partial association cue:
            # fair temporal models can be useful but cannot perfectly replace identity.
            cue_visible = world % 2 == 0
            pair_ids = []
            branches = [_observed_pair(base, branch, cue_visible) for branch in (0, 1)]
            audits.append({"task": name, "world_id": world, **_audit_pair(branches[0], branches[1], np.array_equal(plans[world], plans[world]))})
            worlds_json.append({"task": name, "world_id": world, "split": _split(world), "cue_visible": cue_visible,
                                "simulator": "ManiSkill 3 GPU PhysX", "rendering": False})
            for branch, history in enumerate(branches):
                history_id = len(histories)
                histories.append(history); oracle_histories.append(base)
                pair_ids.append(history_id)
                for slot in range(10):
                    labels.append({"task": name, "world_id": world, "branch": branch, "split": _split(world),
                                   "history_id": history_id, "candidate_slot": slot,
                                   "pair_id": f"{name}:{world}:{slot}", "success": by_key[(world, branch, slot)],
                                   "label_source": "ManiSkill 3 GPU PhysX execution", "candidate_action_hash": hashlib.sha256(plans[world, slot].tobytes()).hexdigest()})
            pairs.append({"task": name, "world_id": world, "branch_history_ids": pair_ids, "candidate_slots": list(range(10)),
                          "candidate_actions_identical": True, "moment_match_pass": audits[-1]["pass"]})
    fair = np.asarray(histories, np.float32); oracle = np.asarray(oracle_histories, np.float32)
    actions = np.asarray(actions_all, np.float32)
    np.savez_compressed(state_dir / "histories.npz", fair_observed_tokens=fair, oracle_identity_trajectories=oracle)
    np.savez_compressed(state_dir / "object_states.npz", current_state=fair[:, -1], unordered_current_state=np.asarray([_sorted_tokens(x)[-1] for x in fair]))
    np.save(state_dir / "candidate_actions.npy", actions)
    (state_dir / "worlds.json").write_text(json.dumps(worlds_json, indent=2, sort_keys=True) + "\n")
    (state_dir / "candidate_labels.jsonl").write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in labels))
    (state_dir / "pairs.json").write_text(json.dumps(pairs, indent=2, sort_keys=True) + "\n")
    (output / "moment_matching_audit.json").write_text(json.dumps({"schema": "milestone5b0-moment-audit-v1", "pairs": audits, "all_pass": all(x["pass"] for x in audits)}, indent=2, sort_keys=True) + "\n")
    (state_dir / "raw_candidate_execution_logs.json").write_text(json.dumps(raw_logs, indent=2, sort_keys=True) + "\n")
    manifest = {"schema": "milestone5b0-state-probe-v1", "config_sha256": config_hash, "tasks": [x[0] for x in TASKS],
                "worlds": 128, "histories": len(fair), "candidate_executions": 2560, "candidate_rows": len(labels),
                "simulator": "ManiSkill 3 GPU PhysX", "rendering": False,
                "files": ["worlds.json", "histories.npz", "object_states.npz", "candidate_actions.npy", "candidate_labels.jsonl", "pairs.json", "raw_candidate_execution_logs.json"]}
    (state_dir / "data_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
