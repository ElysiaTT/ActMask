"""Required real-simulator tests: missing ManiSkill must fail, never skip."""
import json
from dataclasses import replace

import numpy as np
import pytest

from actmask.experiments.binding_gpu.rod import RodCOMEnv
from actmask.experiments.binding_gpu.snapshot_smoke import run
from actmask.experiments.binding_gpu.generate import generate, sha256
from actmask.experiments.binding_gpu.audit_dataset import audit


def test_g1_real_replay(tmp_path):
    assert run(tmp_path)["passed"]


def test_invalid_snapshot_and_actions():
    env = RodCOMEnv()
    try:
        state = env.snapshot()
        with pytest.raises(ValueError, match="schema"):
            env.restore({k: v for k, v in state.items() if k != "com"})
        with pytest.raises(ValueError, match="snapshot body"):
            env.restore({**state, "body": np.full((1, 13), np.nan)})
        with pytest.raises(ValueError, match="action"):
            env.step(np.array([np.nan]))
    finally:
        env.close()


def test_g2_replay_benchmark_and_tamper(tmp_path):
    directory = tmp_path / "data"
    manifest = generate(directory, twins_per_seed=2)
    assert manifest["attempts"] == 4
    report = audit(directory, tmp_path / "reports")
    assert report["passed"] and report["maximum_replay_error"] <= 1e-5
    assert report["method_decision"] == "METHOD_NO_GO"
    assert not report["admission_evaluable"]
    with pytest.raises(ValueError, match="overwrite"):
        generate(directory, twins_per_seed=2)
    public = directory / "public/inputs.npz"
    with np.load(public, allow_pickle=False) as raw:
        arrays = {k: raw[k].copy() for k in raw.files}
    np.savez_compressed(public, **arrays, hidden_com=np.zeros(4))
    with pytest.raises(ValueError, match="digest"):
        audit(directory, tmp_path / "bad")
    # Even a newly signed manifest must not allow extra hidden public fields.
    manifest["files"]["public/inputs.npz"] = sha256(public)
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="public schema"):
        audit(directory, tmp_path / "bad")
