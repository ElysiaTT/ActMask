"""Qualified RGB-D/point-cloud smoke test for the three ActMask task families."""

from __future__ import annotations

import json
import os
from pathlib import Path

import gymnasium as gym
import torch

from actmask.data import maniskill_state_tasks as _registered_tasks  # noqa: F401


TASK_IDS = (
    "ActMaskMovingCubeIntercept-v1",
    "ActMaskMovingContainerPlacement-v1",
    "ActMaskRotatingTargetInteraction-v1",
)
VALIDATED_ICD = "/etc/vulkan/icd.d/test_nvidia_icd.json"


def run(output_dir: str | Path) -> dict:
    """Render only one fixed-seed observation per family; generate no pilot data."""
    if os.environ.get("VK_ICD_FILENAMES") != VALIDATED_ICD:
        raise RuntimeError(f"Set VK_ICD_FILENAMES={VALIDATED_ICD} before visual execution")
    if os.environ.get("XDG_RUNTIME_DIR") != "/tmp":
        raise RuntimeError("Set XDG_RUNTIME_DIR=/tmp before visual execution")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}
    for offset, task_id in enumerate(TASK_IDS):
        env = gym.make(
            task_id, obs_mode="sensor_data", num_envs=1, sim_backend="gpu",
            sensor_configs={"width": 128, "height": 128},
        )
        try:
            first, _ = env.reset(seed=[731 + offset])
            color = first["sensor_data"]["actmask_fixed_camera"]["Color"]
            position_segmentation = first["sensor_data"]["actmask_fixed_camera"]["PositionSegmentation"]
            repeated, _ = env.reset(seed=[731 + offset])
            deterministic = bool(torch.equal(color, repeated["sensor_data"]["actmask_fixed_camera"]["Color"]))
            results[task_id] = {
                "color_shape": list(color.shape),
                "color_dtype": str(color.dtype),
                "position_segmentation_shape": list(position_segmentation.shape),
                "fixed_seed_rgb_deterministic": deterministic,
            }
            if not deterministic:
                raise AssertionError(f"{task_id}: fixed-seed RGB observation changed")
        finally:
            env.close()
    pointcloud = gym.make(
        TASK_IDS[0], obs_mode="pointcloud", num_envs=1, sim_backend="gpu",
        sensor_configs={"width": 128, "height": 128},
    )
    try:
        observation, _ = pointcloud.reset(seed=[991])
        pointcloud_data = observation["pointcloud"]
        pointcloud_shapes = {key: list(value.shape) for key, value in pointcloud_data.items()}
    finally:
        pointcloud.close()
    report = {
        "schema": "milestone4v-visual-smoke-v1",
        "visual_data_generated": False,
        "camera": {"uid": "actmask_fixed_camera", "width": 128, "height": 128},
        "tasks": results,
        "pointcloud": pointcloud_shapes,
    }
    (output / "visual_smoke.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    print(json.dumps(run(root / "outputs" / "actmask" / "milestone4v_visual_pilot"), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
