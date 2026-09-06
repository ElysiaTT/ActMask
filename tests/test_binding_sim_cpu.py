"""Required real-simulator tests: missing ManiSkill must fail, never skip."""
import json
from dataclasses import replace
import shutil

import numpy as np
import pytest

from actmask.experiments.binding_gpu.rod import RodCOMEnv
from actmask.experiments.binding_gpu.snapshot_smoke import run
from actmask.experiments.binding_gpu.generate import generate, sha256
from actmask.experiments.binding_gpu.audit_dataset import audit
from actmask.experiments.binding_gpu.contracts import joint_marginal_error, validate_artifacts
from binding_bench.schema import dataset_digest, load_dataset
from binding_bench.interventions import build_intervention_suite


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
    # Even a rehashed manifest must not allow extra hidden public fields.
    manifest["files"]["public/inputs.npz"] = sha256(public)
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="public schema"):
        audit(directory, tmp_path / "bad")


def test_joint_effect_matching_preserves_height_tilt_pairing():
    left = np.array([[0., 0.], [1., 1.]])
    crossed = np.array([[0., 1.], [1., 0.]])
    assert np.array_equal(np.sort(left, axis=0), np.sort(crossed, axis=0))
    assert joint_marginal_error(left, crossed) == 1.
    assert joint_marginal_error(left, left[::-1]) == 0.
    assert joint_marginal_error(left, left[::-1] + 1e-6) == pytest.approx(1e-6)
    with pytest.raises(ValueError, match="non-finite"):
        joint_marginal_error(left, np.full_like(left, np.nan))


@pytest.fixture(scope="module")
def generated_artifact(tmp_path_factory):
    directory = tmp_path_factory.mktemp("physical") / "dataset"
    generate(directory, twins_per_seed=2)
    return directory


def rehash(directory):
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["files"] = {p.relative_to(directory).as_posix(): sha256(p)
                         for p in directory.rglob("*") if p.is_file() and p != path}
    path.write_text(json.dumps(manifest), encoding="utf-8")


@pytest.mark.parametrize("corruption,message", [
    ("omitted_file", "inventory"), ("unlisted_file", "inventory"),
    ("public_extra", "public inventory"), ("nan_trajectory", "non-finite raw trajectories"),
    ("wrong_shape", "raw trajectories"), ("raw_action", "action mismatch"),
    ("retention", "accounting"), ("duplicate_attempt", "duplicate attempt"),
    ("wrong_suite", "suite original"), ("suite_escape", "suite path"),
    ("false_rejection", "admission ledger"), ("missing_snapshot", "inventory"),
    ("extra_mechanism", "beyond hidden COM"),
])
def test_rehashed_corrupt_artifacts_fail_before_physics(generated_artifact, tmp_path, corruption, message):
    directory = tmp_path / "data"
    shutil.copytree(generated_artifact, directory)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_path = next((directory / "evaluator").glob("raw-*.npz"))
    if corruption == "omitted_file":
        manifest["files"].pop("public/inputs.npz")
    elif corruption in ("unlisted_file", "public_extra"):
        (directory / "public/hidden.txt").write_text("hidden COM", encoding="utf-8")
    elif corruption in ("nan_trajectory", "wrong_shape", "raw_action"):
        with np.load(raw_path, allow_pickle=False) as raw:
            arrays = {key: raw[key].copy() for key in raw.files}
        if corruption == "nan_trajectory":
            arrays["trajectories"][0, 0, 0, 0] = np.nan
        elif corruption == "wrong_shape":
            arrays["trajectories"] = arrays["trajectories"][:, :, :1]
        else:
            arrays["history"] = arrays["history"][::-1]
        np.savez_compressed(raw_path, **arrays)
    elif corruption == "retention":
        manifest["retention"] = .5
    elif corruption in ("duplicate_attempt", "false_rejection"):
        ledger_path = directory / "evaluator/attempts.json"
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        if corruption == "duplicate_attempt":
            ledger[1] = ledger[0]
        else:
            ledger[0]["accepted"] = False
        ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
    elif corruption == "wrong_suite":
        dataset = load_dataset(directory / "evaluator/dataset.npz")
        build_intervention_suite(replace(dataset, true_utility=dataset.true_utility + .25),
                                 directory / "evaluator/suite", seed=44019, source={"kind": "test"})
    elif corruption == "suite_escape":
        path = directory / "evaluator/suite/suite.json"
        suite = json.loads(path.read_text(encoding="utf-8"))
        suite["cases"]["original"]["path"] = "../../dataset.npz"
        path.write_text(json.dumps(suite), encoding="utf-8")
    elif corruption == "missing_snapshot":
        # Move within a disposable test fixture, never delete user evidence.
        snapshot = next((directory / "evaluator/snapshots").glob("*.npz"))
        snapshot.rename(tmp_path / "removed-snapshot.npz")
    elif corruption == "extra_mechanism":
        snapshot = next((directory / "evaluator/snapshots").glob("*.npz"))
        with np.load(snapshot, allow_pickle=False) as raw:
            arrays = {key: raw[key].copy() for key in raw.files}
        arrays["mass"] *= 2
        np.savez_compressed(snapshot, **arrays)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    if corruption not in ("omitted_file", "unlisted_file"):
        rehash(directory)
    with pytest.raises(ValueError, match=message):
        validate_artifacts(directory)


def test_deterministic_dataset_regeneration(generated_artifact, tmp_path):
    generate(tmp_path / "again", twins_per_seed=2)
    _, first = validate_artifacts(generated_artifact)
    _, second = validate_artifacts(tmp_path / "again")
    assert dataset_digest(first) == dataset_digest(second)


def test_report_cannot_modify_dataset(generated_artifact):
    with pytest.raises(ValueError, match="immutable dataset"):
        audit(generated_artifact, generated_artifact / "reports")


def test_nonfinite_live_replay_cannot_emit_pass(generated_artifact, tmp_path, monkeypatch):
    def broken_rollout(self, snapshot, action, steps=8):
        return np.full(2, np.nan), np.full((8, 13), np.nan)

    monkeypatch.setattr(RodCOMEnv, "rollout", broken_rollout)
    with pytest.raises(ValueError, match="non-finite simulator replay"):
        audit(generated_artifact, tmp_path / "report")
    assert not (tmp_path / "report/report.json").exists()


@pytest.mark.parametrize("field,value,message", [
    ("version", np.array(1.5), "schema"),
    ("elapsed", np.array([-.5]), "elapsed"),
    ("elapsed", np.array([2 ** 32]), "elapsed"),
    ("com", np.array([.05, .1, 0]), "mass properties"),
    ("inertia", np.array([1., .01, .01]), "mass properties"),
    ("body", np.zeros((1, 13)), "quaternion"),
])
def test_snapshot_rejects_invalid_state_without_mutation(field, value, message):
    env = RodCOMEnv()
    try:
        before = env.snapshot()
        with pytest.raises(ValueError, match=message):
            env.restore({**before, field: value})
        after = env.snapshot()
        assert all(np.array_equal(before[key], after[key]) for key in before)
        with pytest.raises(ValueError, match="steps"):
            env.rollout(before, .1, steps=0)
    finally:
        env.close()
