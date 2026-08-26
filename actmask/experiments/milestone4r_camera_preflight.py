"""Phase-1 multi-camera calibration and depth-backprojection preflight."""
from __future__ import annotations

import json
from pathlib import Path

import gymnasium as gym
import numpy as np
from mani_skill.utils import sapien_utils

from actmask.data import maniskill_state_tasks as _registered  # noqa: F401


CAMERAS = {
    "cam_train_0": [0.55, -0.55, 0.50], "cam_train_1": [0.55, 0.35, 0.46],
    "cam_train_2": [0.38, -0.62, 0.60], "cam_train_3": [-0.35, -0.55, 0.52],
    "cam_val_0": [-0.55, 0.35, 0.52], "cam_test_held": [-0.62, -0.35, 0.62],
    "cam_test_perturbed": [0.68, -0.40, 0.58],
}


def run(output: str | Path) -> dict:
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    result = {"schema": "milestone4r-camera-preflight-v1", "cameras": {}, "passed": True}
    for name, eye in CAMERAS.items():
        pose = sapien_utils.look_at(eye=eye, target=[0.0, 0.0, 0.04])
        env = gym.make("ActMaskMovingCubeIntercept-v1", obs_mode="sensor_data", num_envs=1,
                       sim_backend="gpu", sensor_configs={"pose": pose, "width": 128, "height": 128})
        try:
            obs, _ = env.reset(seed=[44001])
            param = obs["sensor_param"]["actmask_fixed_camera"]
            data = obs["sensor_data"]["actmask_fixed_camera"]
            depth_mm = -data["PositionSegmentation"][0, ..., 2].detach().cpu().numpy()
            finite_fraction = float(np.isfinite(depth_mm).mean())
            result["cameras"][name] = {
                "eye": eye,
                "intrinsic_shape": list(param["intrinsic_cv"].shape),
                "extrinsic_shape": list(param["extrinsic_cv"].shape),
                "depth_finite_fraction": finite_fraction,
                "depth_positive_fraction": float((depth_mm > 0).mean()),
            }
            result["passed"] = bool(result["passed"] and finite_fraction == 1.0 and depth_mm.max() > 0)
        finally:
            env.close()
    (output / "camera_preflight.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    print(json.dumps(run(root / "outputs" / "actmask" / "milestone4r_robust_visual"), indent=2, sort_keys=True))
