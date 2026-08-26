"""Fair-input adapter for the separately versioned Milestone 4R probe.

The storage records contain world/pair references for grouping and audit.  This
module resolves those references internally, then returns only the frozen fair
input whitelist to learning code.  It deliberately does not load any oracle
segmentation diagnostic.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


FAIR_KEYS = {
    "rgbd_history", "world_pointcloud_history", "camera_intrinsics",
    "camera_extrinsics", "timestamps", "rgb_mask", "depth_mask",
    "visibility", "confidence", "candidate_actions", "tcp_state",
}
PRIVATE_METADATA_KEYS = {"history_ref", "world_id", "candidate_slot", "split", "pair_group", "condition"}


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def _block_mean(values: np.ndarray, block: int = 8) -> np.ndarray:
    """Deterministic 8×8 pooling used by all standard fair controls."""
    *leading, height, width, channels = values.shape
    if height % block or width % block:
        raise ValueError(f"Expected spatial dimensions divisible by {block}, got {(height, width)}")
    return values.reshape(*leading, height // block, block, width // block, block, channels).mean(axis=(-4, -2))


def backproject_world(
    depth_m: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic_world_to_cv: np.ndarray,
) -> np.ndarray:
    """Backproject CV depth to the common world frame using legal calibration.

    ``extrinsic_world_to_cv`` is the 3×4 ``extrinsic_cv`` matrix supplied by
    ManiSkill.  The returned coordinates use no segmentation or identity.
    """
    depth = np.asarray(depth_m, dtype=np.float32)
    if depth.ndim != 3:
        raise ValueError(f"Expected [frames,height,width] depth, got {depth.shape}")
    intr = np.asarray(intrinsic, dtype=np.float32)
    extr = np.asarray(extrinsic_world_to_cv, dtype=np.float32)
    if intr.shape != (3, 3) or extr.shape != (3, 4):
        raise ValueError("Expected intrinsic [3,3] and extrinsic [3,4]")
    frames, height, width = depth.shape
    u, v = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    x = (u - intr[0, 2]) * depth / intr[0, 0]
    y = (v - intr[1, 2]) * depth / intr[1, 1]
    camera = np.stack((x, y, depth), axis=-1)
    rotation = extr[:, :3]
    translation = extr[:, 3]
    # extrinsic_cv maps world→CV, so apply its rigid inverse pointwise.
    world = (camera - translation.reshape(1, 1, 1, 3)) @ rotation
    return np.where(depth[..., None] > 0, world, 0.0).astype(np.float32)


def load_fair_inputs(root: str | Path, task: str, *, pool_block: int = 8) -> dict[str, np.ndarray]:
    """Return only fair 4R model inputs, aligned one-to-one with candidate rows."""
    root = Path(root)
    bundle = np.load(root / f"{task}_bundles.npz")
    rows = _rows(root / f"{task}_candidates.jsonl")
    calibration = json.loads((root / f"{task}_camera_calibration.json").read_text())
    refs = np.asarray([row["history_ref"] for row in rows], dtype=np.int64)
    world_for_ref: dict[int, int] = {}
    for row in rows:
        reference = row["history_ref"]
        if reference in world_for_ref and world_for_ref[reference] != row["world_id"]:
            raise ValueError("A history reference must belong to exactly one world")
        world_for_ref[reference] = row["world_id"]
    calibration_by_world = {entry["world_id"]: entry for entry in calibration}
    intrinsics = np.stack([np.asarray(calibration_by_world[world_for_ref[reference]]["intrinsic"], dtype=np.float32) for reference in range(len(bundle["rgb"]))])
    extrinsics = np.stack([np.asarray(calibration_by_world[world_for_ref[reference]]["extrinsic"], dtype=np.float32) for reference in range(len(bundle["rgb"]))])

    rgb = bundle["rgb"].astype(np.float32) / 255.0
    depth = bundle["depth_mm"].astype(np.float32) / 1000.0
    visibility = bundle["visibility"].astype(np.float32)
    confidence = bundle["confidence"].astype(np.float32)
    rgb_mask = bundle["rgb_mask"].astype(np.float32)
    depth_mask = bundle["depth_mask"].astype(np.float32)
    rgb = rgb * visibility[..., None] * rgb_mask[..., None, None, None]
    depth = depth * visibility * depth_mask[..., None, None]
    rgbd = np.concatenate((rgb, depth[..., None]), axis=-1)
    rgbd = _block_mean(rgbd, pool_block).reshape(len(rgb), rgb.shape[1], -1).astype(np.float32)
    world_cloud = np.stack([backproject_world(depth[index], intrinsics[index], extrinsics[index]) for index in range(len(depth))])
    world_cloud = _block_mean(world_cloud, pool_block).reshape(len(depth), depth.shape[1], -1).astype(np.float32)

    fair = {
        "rgbd_history": rgbd[refs],
        "world_pointcloud_history": world_cloud[refs],
        "camera_intrinsics": intrinsics[refs],
        "camera_extrinsics": extrinsics[refs],
        "timestamps": bundle["timestamps"][refs].astype(np.float32),
        "rgb_mask": rgb_mask[refs],
        "depth_mask": depth_mask[refs],
        "visibility": visibility[refs],
        "confidence": confidence[refs],
        "candidate_actions": np.asarray([row["candidate_actions"] for row in rows], dtype=np.float32),
        "tcp_state": np.asarray([row["tcp_state"] for row in rows], dtype=np.float32),
    }
    if set(fair) != FAIR_KEYS:
        raise AssertionError(f"Fair interface mismatch: {set(fair) ^ FAIR_KEYS}")
    return fair


def load_evaluation_metadata(root: str | Path, task: str) -> tuple[list[dict], np.ndarray]:
    """Private grouping/label metadata for evaluation only; never pass to a model."""
    root = Path(root)
    return _rows(root / f"{task}_candidates.jsonl"), np.load(root / f"{task}_labels.npz")["success"].astype(np.float32)
