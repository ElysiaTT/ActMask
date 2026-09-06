"""G1: real CPU replay, including disk/fresh-environment and mechanism restore."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import importlib.metadata

import numpy as np
from .rod import RodCOMEnv, load_snapshot, save_snapshot


def run(output: Path):
    output.mkdir(parents=True, exist_ok=True)
    env = RodCOMEnv()
    try:
        env.set_anchor(.06)
        anchor = env.snapshot()
        save_snapshot(output / "anchor.npz", anchor)
        effect, reference = env.rollout(anchor, .5)
        errors = []
        for _ in range(3):
            env.set_anchor(-.06)
            env.step(np.array([-.8]))
            _, replay = env.rollout(load_snapshot(output / "anchor.npz"), .5)
            errors.append(float(np.max(np.abs(reference - replay))))
        env.set_anchor(-.06)
        other, _ = env.rollout(env.snapshot(), .5)
        # Also test a moving, nonzero-time state, not only reset snapshots.
        moving = env.snapshot()
        _, moving_reference = env.rollout(moving, -.3)
        _, moving_replay = env.rollout(moving, -.3)
        errors.append(float(np.max(np.abs(moving_reference - moving_replay))))
        restored_time = int(env._elapsed_steps.item()) == int(moving["elapsed"][0]) + 8
    finally:
        env.close()
    fresh = RodCOMEnv()
    try:
        _, replay = fresh.rollout(load_snapshot(output / "anchor.npz"), .5)
        errors.append(float(np.max(np.abs(reference - replay))))
    finally:
        fresh.close()
    checks = {"replay": max(errors) <= 1e-5,
              "elapsed_restored": restored_time,
              "nonzero_motion": float(np.max(np.abs(reference[-1] - anchor["body"][0]))) > .01,
              "mechanism_changes_outcome": float(np.max(np.abs(effect - other))) > .01}
    result = {"schema_version": "actmask-cpu-g1-v1", "passed": all(checks.values()),
              "checks": checks, "maximum_replay_error": max(errors), "replay_errors": errors,
              "mechanism_effect_difference": float(np.max(np.abs(effect - other))),
              "backend": "physx_cpu", "render_backend": "none",
              "versions": {name: importlib.metadata.version(name) for name in ("torch", "mani-skill", "sapien", "numpy")}}
    (output / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if not result["passed"]:
        raise RuntimeError(f"G1 failed: {checks}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    print(json.dumps(run(parser.parse_args().output_dir), indent=2))
