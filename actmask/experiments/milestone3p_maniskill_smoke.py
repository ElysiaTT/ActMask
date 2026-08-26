"""Record the bounded B1 GPU/state qualification for the custom tasks."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import gymnasium as gym
import torch

from actmask.data import maniskill_state_tasks as _registered_tasks  # noqa: F401


TASK_ID = "ActMaskMovingCubeIntercept-v1"


def _array(value) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Expected tensor state observation, got {type(value)!r}")
    return value


def run(output_dir: str | Path) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("ManiSkill GPU smoke requires CUDA")
    device = torch.device("cuda:0")
    # The linked CUDA build accepts only the current-device form of these
    # memory-stat APIs (device arguments raise ``Invalid device argument``).
    torch.cuda.reset_peak_memory_stats()
    single = gym.make(TASK_ID, obs_mode="state", num_envs=1, sim_backend="gpu", render_backend="none")
    try:
        first, _ = single.reset(seed=[20260723])
        first = _array(first)
        _, _, _, _, info = single.step(torch.zeros((1, 3), device=device))
        repeated, _ = single.reset(seed=[20260723])
        deterministic = bool(torch.equal(first, _array(repeated)))
        single_result = dict(
            observation_shape=list(first.shape),
            success_field_present="success" in info,
            deterministic_reset=deterministic,
        )
    finally:
        single.close()
    vector = gym.make(TASK_ID, obs_mode="state", num_envs=8, sim_backend="gpu", render_backend="none")
    try:
        observation, _ = vector.reset(seed=list(range(3107, 3115)))
        observation = _array(observation)
        action = torch.zeros((8, 3), device=device)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        for _ in range(40):
            vector.step(action)
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        vector_result = dict(
            num_envs=8,
            observation_shape=list(observation.shape),
            control_steps=40,
            aggregate_steps_per_second=float(8 * 40 / elapsed),
        )
    finally:
        vector.close()
    result = dict(
        task_id=TASK_ID,
        torch=torch.__version__,
        cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0),
        state_observation_smoke=single_result,
        gpu_vector_smoke=vector_result,
        vulkaninfo_available=shutil.which("vulkaninfo") is not None,
        visual_rgbd_pointcloud_status=(
            "blocked: vulkaninfo is unavailable and prior SAPIEN renderer probe failed to find a rendering device; no system package was installed"
        ),
        peak_torch_gpu_memory_bytes=int(torch.cuda.max_memory_allocated()),
    )
    if not deterministic:
        raise AssertionError("fixed-seed reset was not deterministic")
    (output / "maniskill_state_gpu_smoke.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    result = run(root / "outputs" / "actmask" / "milestone3p_gpu_benchmark")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
