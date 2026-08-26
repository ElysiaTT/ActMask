"""Separate 4R-v2 full visual regeneration after the approved C10 preprobe."""
from __future__ import annotations

import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from mani_skill.utils import sapien_utils

from actmask.data import maniskill_state_tasks as _registered  # noqa: F401
from actmask.data.maniskill_pilot import CandidateSpec, candidate_specs
from actmask.data.milestone4r_robust_pilot import (
    CAMERAS, CONDITIONS, TASKS, TASK_KEY, _appearance_transform,
    _camera_local_distractor_offset, _camera_schedule, _condition_schedule,
    _history_with_oracle_visibility, _mask, _reordered_pair,
)
from actmask.data.milestone4v_visual_pilot import _candidate_rollout
from actmask.data.milestone4r_v2_preprobe import C10, _planned_actions


def _placement_rollout(env, decision: dict, slot: int, branch: int, horizon: int) -> tuple[np.ndarray, bool, np.ndarray]:
    """Run a C10 plan calculated before and independently of future branch."""
    base = env.unwrapped
    actions = _planned_actions(base, decision, C10[slot], horizon)
    base.set_state_dict(decision)
    base.scene._gpu_apply_all(); base.scene._gpu_fetch_all()
    base._elapsed_steps.zero_()
    base.set_hidden_future_mode(float(branch))
    info = None
    for step in range(horizon):
        _, _, _, _, info = env.step(torch.as_tensor(actions[:, step], device=base.device))
    assert info is not None
    success = bool(info["success"][0].detach().cpu())
    tcp = decision["tcp_proxy"][0, :3].detach().cpu().numpy().astype(np.float32)
    return actions[0], success, tcp


def generate(output_dir: str | Path, *, worlds_per_task: int = 128, horizon: int = 12, seed: int = 47000) -> dict:
    """Regenerate all three 4R families; only placement C10 is versioned here."""
    if worlds_per_task != 128:
        raise ValueError("The preregistered full 4R-v2 probe requires 128 worlds per family")
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    summary = {"schema": "milestone4r-v2-full-visual-probe-v1", "worlds_per_task": worlds_per_task, "candidate_executions": 0, "tasks": {}}
    for task_offset, (task, task_id) in enumerate(TASKS.items()):
        bundles_rgb=[]; bundles_depth=[]; masks_rgb=[]; masks_depth=[]; visibility=[]; stamps=[]; rows=[]; labels=[]; camera_calibration=[]; association_diagnostics=[]
        generic_specs = list(candidate_specs(TASK_KEY[task]))
        camera_schedule = _camera_schedule(worlds_per_task)
        condition_schedule = _condition_schedule(worlds_per_task)
        for world in range(worlds_per_task):
            split, camera = camera_schedule[world]; condition = condition_schedule[world]
            pose = sapien_utils.look_at(eye=CAMERAS[camera], target=[0, 0, .04])
            env = gym.make(task_id, obs_mode="sensor_data", num_envs=1, sim_backend="gpu", robust_distractor=True, robust_distractor_camera_offset=_camera_local_distractor_offset(camera), sensor_configs={"pose": pose, "width":128, "height":128})
            try:
                rgb, depth, target_visible, distractor_visible, decision = _history_with_oracle_visibility(env, seed + task_offset * 10000 + world, 6)
                rgb, depth = _appearance_transform(rgb, task_offset=task_offset, world=world), depth.copy()
                pixel_visibility = np.ones(depth.shape, dtype=np.uint8)
                if condition == "target_occlusion":
                    pixel_visibility[2:4] = np.where(target_visible[2:4], 0, pixel_visibility[2:4])
                    rgb = np.where(pixel_visibility[..., None] > 0, rgb, 0).astype(np.uint8)
                    depth = np.where(pixel_visibility > 0, depth, 0).astype(np.uint16)
                    frames=[]
                    for frame in (2, 3):
                        before=int(target_visible[frame].sum()); distractor=int((distractor_visible[frame] & (pixel_visibility[frame] > 0)).sum())
                        if distractor == 0: raise RuntimeError(f"{task} world={world}: distractor absent during target occlusion")
                        frames.append({"frame":frame,"target_visible_before":before,"target_visible_after":int((target_visible[frame] & (pixel_visibility[frame] > 0)).sum()),"distractor_visible_after":distractor,"natural_target_occlusion":before == 0})
                    association_diagnostics.append({"world_id":world,"camera":camera,"frames":frames})
                branch_histories=((rgb, depth, pixel_visibility), _reordered_pair(rgb, depth, pixel_visibility))
                rgb_mask, depth_mask, timestamps = _mask(world, condition)
                for branch, (brgb, bdepth, bvisibility) in enumerate(branch_histories):
                    ref=len(bundles_rgb); bundles_rgb.append(brgb.copy()); bundles_depth.append(bdepth.copy()); visibility.append(bvisibility.copy()); masks_rgb.append(rgb_mask); masks_depth.append(depth_mask); stamps.append(timestamps)
                    for slot in range(10):
                        if task == "SignedMovingWindowPlacement":
                            actions, success, tcp = _placement_rollout(env, decision, slot, branch, horizon)
                        else:
                            spec: CandidateSpec = generic_specs[slot % len(generic_specs)]
                            actions, success, tcp = _candidate_rollout(env, decision, TASK_KEY[task], spec, slot, branch, horizon)
                        rows.append({"history_ref":ref,"world_id":world,"candidate_slot":slot,"split":split,"pair_group":f"{task}:{world}:{slot}","condition":condition,"candidate_actions":actions.tolist(),"tcp_state":tcp.tolist()})
                        labels.append(success)
                param=env.unwrapped.get_sensor_params()["actmask_fixed_camera"]
                camera_calibration.append({"world_id":world,"camera":camera,"intrinsic":param["intrinsic_cv"][0].detach().cpu().numpy().tolist(),"extrinsic":param["extrinsic_cv"][0].detach().cpu().numpy().tolist()})
            finally:
                env.close()
        order=np.random.default_rng(8301+task_offset).permutation(len(rows)); rows=[rows[int(index)] for index in order]; labels=[labels[int(index)] for index in order]
        np.savez_compressed(output/f"{task}_bundles.npz", rgb=np.stack(bundles_rgb), depth_mm=np.stack(bundles_depth), rgb_mask=np.stack(masks_rgb), depth_mask=np.stack(masks_depth), visibility=np.stack(visibility), confidence=np.stack(visibility), timestamps=np.stack(stamps))
        (output/f"{task}_candidates.jsonl").write_text("".join(json.dumps(row,sort_keys=True)+"\n" for row in rows))
        np.savez_compressed(output/f"{task}_labels.npz", success=np.asarray(labels,dtype=np.bool_))
        (output/f"{task}_camera_calibration.json").write_text(json.dumps(camera_calibration,indent=2,sort_keys=True)+"\n")
        (output/f"{task}_oracle_association_diagnostic.json").write_text(json.dumps({"schema":"milestone4r-v2-oracle-association-diagnostic-v1","generator_private":True,"target_occlusions":association_diagnostics},indent=2,sort_keys=True)+"\n")
        count=len(rows); summary["candidate_executions"] += count
        summary["tasks"][task]={"bundles":len(bundles_rgb),"candidates":count,"conditions":sorted(set(row["condition"] for row in rows)),"candidate_grid":"v2_placement_grid" if task == "SignedMovingWindowPlacement" else "frozen_4r_grid"}
    (output/"probe_summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
    return summary


if __name__ == "__main__":
    project=Path(__file__).resolve().parents[2]
    print(json.dumps(generate(project/"outputs"/"actmask"/"milestone4r_v2_candidate_diversity"/"full_probe"),indent=2,sort_keys=True))
