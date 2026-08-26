"""4R multi-camera, masked, order-sensitive visual probe generator.

This is separate from all frozen 4V outputs. It stores each paired bundle once;
candidate rows refer to the bundle and never receive camera IDs or branch IDs
as fair inputs.
"""
from __future__ import annotations

import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from mani_skill.utils import sapien_utils

from actmask.data import maniskill_state_tasks as _registered  # noqa: F401
from actmask.data.maniskill_pilot import CandidateSpec, _candidate_action, candidate_specs
from actmask.data.milestone4v_visual_pilot import _candidate_rollout, _clone, _frame


TASKS = {"MovingCubeIntercept": "ActMaskMovingCubeIntercept-v1", "SignedMovingWindowPlacement": "ActMaskMovingContainerPlacement-v1", "FixedPhaseRotatingCaptureWindow": "ActMaskRotatingTargetInteraction-v1"}
TASK_KEY = {"MovingCubeIntercept": "moving_cube_intercept", "SignedMovingWindowPlacement": "moving_container_placement", "FixedPhaseRotatingCaptureWindow": "rotating_target_interaction"}
CAMERAS = {"cam_train_0": [0.55,-0.55,.50], "cam_train_1": [0.55,.35,.46], "cam_train_2": [.38,-.62,.60], "cam_train_3": [-.35,-.55,.52], "cam_val_0": [-.55,.35,.52], "cam_test_held": [-.62,-.35,.62], "cam_test_perturbed": [.68,-.40,.58]}
CONDITIONS = ("complete", "one_random_early", "two_random", "burst_two", "burst_three", "missing_final", "target_occlusion", "async_rgb", "async_depth", "irregular_timestamps", "latency")
TEMPORAL_ORDER = np.array([2, 0, 3, 1, 4, 5])


def _camera_schedule(worlds: int) -> list[tuple[str, str]]:
    """Independent, deterministic world-level camera assignment.

    The schedule is sampled before simulator worlds exist.  It contains every
    preregistered camera (four train, one validation, two test) and is shuffled
    independently of the world seed, temporal branch, and candidate outcome.
    """
    train_count = int(worlds * 0.60)
    val_count = int(worlds * 0.20)
    test_count = worlds - train_count - val_count
    entries = [
        ("train", f"cam_train_{index % 4}") for index in range(train_count)
    ]
    entries += [("validation", "cam_val_0") for _ in range(val_count)]
    entries += [
        ("test", ("cam_test_held", "cam_test_perturbed")[index % 2])
        for index in range(test_count)
    ]
    order = np.random.default_rng(9413 + worlds).permutation(worlds)
    return [entries[int(index)] for index in order]


def _condition_schedule(worlds: int) -> list[str]:
    """Counterbalanced condition assignment independent of simulator worlds."""
    rng = np.random.default_rng(9479 + worlds)
    schedule: list[str] = []
    while len(schedule) < worlds:
        schedule.extend(np.asarray(CONDITIONS)[rng.permutation(len(CONDITIONS))].tolist())
    return schedule[:worlds]


def _mask(world: int, condition: str, frames: int = 6):
    rgb = np.ones(frames, dtype=np.uint8); depth = np.ones(frames, dtype=np.uint8)
    if condition == "one_random_early": rgb[1 + world % 3] = depth[1 + world % 3] = 0
    elif condition == "two_random": rgb[[1,3]] = depth[[1,3]] = 0
    elif condition == "burst_two": rgb[1:3] = depth[1:3] = 0
    elif condition == "burst_three": rgb[1:4] = depth[1:4] = 0
    elif condition == "missing_final": rgb[-1] = depth[-1] = 0
    elif condition == "async_rgb": rgb[[1,3]] = 0
    elif condition == "async_depth": depth[[2,4]] = 0
    ts = np.linspace(-.25, 0, frames, dtype=np.float32)
    if condition == "irregular_timestamps": ts += np.array([-.02,.01,-.01,.015,-.005,0], dtype=np.float32)
    if condition == "latency": ts += .04
    return rgb, depth, ts


def _reordered_pair(*arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    # Exact unordered frame and centroid multisets; final two/current match.
    return tuple(array[TEMPORAL_ORDER] for array in arrays)


def _camera_local_distractor_offset(camera: str) -> tuple[float, float, float]:
    """Place the distractor laterally/upward in this camera's image plane.

    This is generator-only scene setup.  The resulting calibration remains a
    legal observable, but neither this offset nor a camera ID is serialized as
    a fair model feature.
    """
    eye = np.asarray(CAMERAS[camera], dtype=np.float32)
    forward = np.asarray([0.0, 0.0, 0.04], dtype=np.float32) - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.asarray([0.0, 0.0, 1.0], dtype=np.float32))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    up /= np.linalg.norm(up)
    return tuple((0.24 * right + 0.14 * up).astype(np.float32).tolist())


def _appearance_transform(rgb: np.ndarray, *, task_offset: int, world: int) -> np.ndarray:
    """Counterfactually shared, world-randomized RGB appearance/background.

    The seed is separate from physical-world, temporal-branch, and candidate
    seeds.  It changes global material-like colour plus a fixed spatial
    background gradient, while preserving an identical transformation for both
    paired branches.
    """
    rng = np.random.default_rng(9613 + task_offset * 1000 + world)
    gain = rng.uniform(.82, 1.18, size=(1, 1, 1, 3))
    bias = rng.uniform(-14.0, 14.0, size=(1, 1, 1, 3))
    height, width = rgb.shape[1:3]
    grid_y, grid_x = np.mgrid[-1:1:complex(height), -1:1:complex(width)]
    gradient_direction = rng.uniform(-1.0, 1.0, size=(1, 1, 1, 3))
    gradient = (grid_x + rng.uniform(-1.0, 1.0) * grid_y)[None, ..., None] * gradient_direction * 8.0
    return np.clip(rgb.astype(np.float32) * gain + bias + gradient, 0, 255).astype(np.uint8)


def _history_with_oracle_visibility(env, seed: int, frames: int = 6):
    """Render a fair RGB-D history plus generator-private visibility masks.

    Segmentation is read only to construct and audit the target-only synthetic
    occlusion.  It is not stored in the benchmark and is never returned by a
    fair data loader.
    """
    observation, _ = env.reset(seed=[seed])
    base = env.unwrapped
    target_id = int(base.payload.per_scene_id[0].detach().cpu())
    distractor_id = int(base.distractor.per_scene_id[0].detach().cpu())
    rgb_values = []
    depth_values = []
    target_visible = []
    distractor_visible = []

    def capture(value):
        rgb, depth = _frame(value)
        segmentation = value["sensor_data"]["actmask_fixed_camera"]["PositionSegmentation"][0, ..., 3]
        ids = segmentation.detach().cpu().numpy()
        rgb_values.append(rgb)
        depth_values.append(depth)
        target_visible.append(ids == target_id)
        distractor_visible.append(ids == distractor_id)

    capture(observation)
    zero = torch.zeros((1, 3), dtype=torch.float32, device=base.device)
    for _ in range(frames - 1):
        observation, _, _, _, _ = env.step(zero)
        capture(observation)
    return (
        np.stack(rgb_values),
        np.stack(depth_values),
        np.stack(target_visible),
        np.stack(distractor_visible),
        _clone(base.get_state_dict()),
    )


def generate(output_dir: str | Path, *, worlds_per_task: int = 128, horizon: int = 12, seed: int = 47000):
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    summary = {"schema": "milestone4r-robust-pilot-v1", "worlds_per_task": worlds_per_task, "tasks": {}}
    for task_offset, (task, task_id) in enumerate(TASKS.items()):
        bundles_rgb=[]; bundles_depth=[]; masks_rgb=[]; masks_depth=[]; visibility=[]; stamps=[]; rows=[]; labels=[]; camera_calibration=[]; association_diagnostics=[]
        specs = list(candidate_specs(TASK_KEY[task]))
        camera_schedule = _camera_schedule(worlds_per_task)
        condition_schedule = _condition_schedule(worlds_per_task)
        for world in range(worlds_per_task):
            split, camera = camera_schedule[world]; condition = condition_schedule[world]
            pose=sapien_utils.look_at(eye=CAMERAS[camera], target=[0,0,.04])
            env = gym.make(task_id, obs_mode="sensor_data", num_envs=1, sim_backend="gpu", robust_distractor=True, robust_distractor_camera_offset=_camera_local_distractor_offset(camera), sensor_configs={"pose":pose,"width":128,"height":128})
            try:
                rgb, depth, target_visible, distractor_visible, decision = _history_with_oracle_visibility(env, seed + task_offset * 10000 + world, 6)
                rgb, depth = _appearance_transform(rgb, task_offset=task_offset, world=world), depth.copy()
                pixel_visibility = np.ones(depth.shape, dtype=np.uint8)
                if condition == "target_occlusion":
                    # Target-only occlusion is derived from renderer-private
                    # segmentation in source-time coordinates.  The masks are
                    # retained only as a legal visibility/confidence input;
                    # segmentation IDs/masks themselves are discarded.
                    pixel_visibility[2:4] = np.where(target_visible[2:4], 0, pixel_visibility[2:4])
                    rgb = np.where(pixel_visibility[..., None] > 0, rgb, 0).astype(np.uint8)
                    depth = np.where(pixel_visibility > 0, depth, 0).astype(np.uint16)
                    diagnostic_frames = []
                    for frame in (2, 3):
                        target_before = int(target_visible[frame].sum())
                        distractor_after = int((distractor_visible[frame] & (pixel_visibility[frame] > 0)).sum())
                        if distractor_after == 0:
                            raise RuntimeError(f"{task} world={world} camera={camera}: distractor absent during target occlusion")
                        diagnostic_frames.append({"frame": frame, "target_visible_before": target_before, "target_visible_after": int((target_visible[frame] & (pixel_visibility[frame] > 0)).sum()), "distractor_visible_after": distractor_after, "natural_target_occlusion": target_before == 0})
                    association_diagnostics.append({"world_id": world, "camera": camera, "frames": diagnostic_frames})
                # Both branches have identical observations as unordered sets,
                # final two frames, masks, camera, action set, and decision state.
                branch_histories = ((rgb, depth, pixel_visibility), _reordered_pair(rgb, depth, pixel_visibility))
                rm, dm, ts = _mask(world, condition)
                for branch, (brgb, bdepth, bvisibility) in enumerate(branch_histories):
                    brgb, bdepth, bvisibility = brgb.copy(), bdepth.copy(), bvisibility.copy()
                    ref=len(bundles_rgb); bundles_rgb.append(brgb); bundles_depth.append(bdepth); visibility.append(bvisibility); masks_rgb.append(rm); masks_depth.append(dm); stamps.append(ts)
                    for slot in range(10):
                        spec: CandidateSpec = specs[slot % len(specs)]
                        actions, success, tcp = _candidate_rollout(env, decision, TASK_KEY[task], spec, slot, branch, horizon)
                        rows.append({"history_ref": ref, "world_id": world, "candidate_slot": slot, "split": split, "pair_group": f"{task}:{world}:{slot}", "condition": condition, "candidate_actions": actions.tolist(), "tcp_state": tcp.tolist()}); labels.append(success)
                param = env.unwrapped.get_sensor_params()["actmask_fixed_camera"]
                camera_calibration.append({"world_id": world, "camera": camera, "intrinsic": param["intrinsic_cv"][0].detach().cpu().numpy().tolist(), "extrinsic": param["extrinsic_cv"][0].detach().cpu().numpy().tolist()})
            finally: env.close()
        order=np.random.default_rng(7301+task_offset).permutation(len(rows)); rows=[rows[int(i)] for i in order]; labels=[labels[int(i)] for i in order]
        np.savez_compressed(output/f"{task}_bundles.npz", rgb=np.stack(bundles_rgb), depth_mm=np.stack(bundles_depth), rgb_mask=np.stack(masks_rgb), depth_mask=np.stack(masks_depth), visibility=np.stack(visibility), confidence=np.stack(visibility), timestamps=np.stack(stamps))
        (output/f"{task}_candidates.jsonl").write_text("".join(json.dumps(r,sort_keys=True)+"\n" for r in rows))
        np.savez_compressed(output/f"{task}_labels.npz", success=np.asarray(labels,dtype=np.bool_))
        (output/f"{task}_camera_calibration.json").write_text(json.dumps(camera_calibration,indent=2,sort_keys=True)+"\n")
        (output/f"{task}_oracle_association_diagnostic.json").write_text(json.dumps({"schema":"milestone4r-oracle-association-diagnostic-v1","generator_private":True,"target_occlusions":association_diagnostics},indent=2,sort_keys=True)+"\n")
        summary["tasks"][task]={"bundles":len(bundles_rgb),"candidates":len(rows),"executions":len(rows),"conditions":sorted(set(r["condition"] for r in rows)),"target_occlusion_worlds":len(association_diagnostics),"appearance_randomized_worlds":worlds_per_task}
    (output/"probe_summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
    return summary
