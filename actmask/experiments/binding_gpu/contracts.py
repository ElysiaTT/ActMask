"""Fail-closed, NumPy-only contracts for the bounded physical smoke artifact."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
from binding_bench.interventions import CASES, load_suite
from binding_bench.schema import dataset_arrays, dataset_digest, load_dataset

PUBLIC_KEYS = {"schema_version", "dataset_id", "split", "history_actions", "history_effects",
               "history_times", "history_mask", "context", "candidates", "candidate_ids", "twin_ids"}
RAW_SHAPES = {"mechanisms": (2,), "history": (8,), "candidates": (12,),
              "history_effects": (2, 8, 2), "candidate_effects": (2, 12, 2),
              "context": (2, 13), "trajectories": (2, 20, 8, 13)}
SNAPSHOT_KEYS = {"version", "body", "com", "mass", "inertia", "elapsed"}


def validate_snapshot(state):
    """Validate before mutating the simulator, including native float/int limits."""
    if set(state) != SNAPSHOT_KEYS:
        raise ValueError("unsupported snapshot schema")
    version = np.asarray(state["version"])
    if version.shape != () or version.dtype.kind not in "iu" or version.item() != 1:
        raise ValueError("unsupported snapshot schema")
    shapes = {"body": (1, 13), "com": (3,), "mass": (1,), "inertia": (3,), "elapsed": (1,)}
    for key, shape in shapes.items():
        value = np.asarray(state[key])
        if value.shape != shape or value.dtype.kind not in "fiu" or not np.isfinite(value).all() or np.any(np.abs(value.astype(np.float64)) > np.finfo(np.float32).max):
            raise ValueError(f"invalid snapshot {key}")
    elapsed = np.asarray(state["elapsed"])
    if elapsed.dtype.kind not in "iu" or elapsed[0] < 0 or elapsed[0] > np.iinfo(np.int32).max - 8:
        raise ValueError("invalid snapshot elapsed")
    if not np.isclose(np.linalg.norm(state["body"][0, 3:7]), 1., atol=1e-5, rtol=0):
        raise ValueError("invalid snapshot quaternion")
    if state["mass"][0] <= 0 or np.any(state["inertia"] <= 0) or abs(state["com"][0]) > .1:
        raise ValueError("invalid snapshot mass properties")
    if np.any(state["com"][1:] != 0) or 2 * np.max(state["inertia"]) > np.sum(state["inertia"]) + 1e-8:
        raise ValueError("invalid snapshot mass properties")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utility(effects):
    return effects[..., 0] - .1 * effects[..., 1]


def joint_marginal_error(left, right):
    """Minimum bottleneck matching error between whole effect vectors.

    Matching rows preserves correlations; sorting each coordinate does not.
    Tiny smoke histories permit an exact augmenting-path matching algorithm.
    """
    left, right = np.asarray(left), np.asarray(right)
    if left.shape != right.shape or left.ndim != 2 or not left.size:
        raise ValueError("invalid effect marginal shape")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("non-finite effect marginal")
    distances = np.max(np.abs(left[:, None] - right[None, :]), axis=2)
    for tolerance in np.unique(distances):
        assigned = {}

        def augment(row, visited):
            for column in np.flatnonzero(distances[row] <= tolerance):
                if column in visited:
                    continue
                visited.add(column)
                if column not in assigned or augment(assigned[column], visited):
                    assigned[column] = row
                    return True
            return False

        if all(augment(row, set()) for row in range(len(left))):
            return float(tolerance)
    raise ValueError("cannot match effect marginal")


def attempt_checks(context, effects, outcomes):
    error = joint_marginal_error(effects[0], effects[1])
    checks = {"public_context_equal": np.array_equal(context[0], context[1]),
              "effect_marginal_matched": error <= 1e-4,
              "utility_span": np.min(np.ptp(utility(outcomes), axis=1)) > .05,
              "different_best_action": np.argmax(utility(outcomes)[0]) != np.argmax(utility(outcomes)[1])}
    return {key: bool(value) for key, value in checks.items()}, error


def validate_artifacts(directory: Path):
    """Validate coverage, content, provenance and suite binding before physics."""
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "actmask-cpu-g2-v1":
        raise ValueError("unsupported G2 manifest")
    paths = list(directory.rglob("*"))
    if any(path.is_symlink() or not path.resolve().is_relative_to(directory.resolve()) for path in paths):
        raise ValueError("artifact paths must not escape or use symbolic links")
    actual = {path.relative_to(directory).as_posix() for path in paths if path.is_file()} - {"manifest.json"}
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != actual:
        raise ValueError("artifact inventory must cover every file exactly")
    for relative, expected in files.items():
        path = (directory / relative).resolve()
        if not path.is_relative_to(directory.resolve()) or sha256(path) != expected:
            raise ValueError(f"artifact digest mismatch or unsafe path: {relative}")
    if {p.relative_to(directory / "public").as_posix() for p in (directory / "public").rglob("*") if p.is_file()} != {"inputs.npz"}:
        raise ValueError("public inventory must contain only inputs.npz")
    dataset = load_dataset(directory / "evaluator/dataset.npz")
    if dataset.history_actions.shape != (dataset.twins, 2, 8, 1) or dataset.candidates.shape != (dataset.twins, 2, 12, 1):
        raise ValueError("unexpected smoke action dimensions")
    digest = dataset_digest(dataset)
    if manifest.get("dataset_audit", {}).get("dataset_digest") != digest:
        raise ValueError("dataset provenance digest mismatch")
    if manifest.get("public_keys") != sorted(PUBLIC_KEYS):
        raise ValueError("public schema manifest mismatch")
    with np.load(directory / "public/inputs.npz", allow_pickle=False) as raw:
        if set(raw.files) != PUBLIC_KEYS:
            raise ValueError("public schema contains missing or hidden fields")
        for key, value in dataset_arrays(dataset).items():
            if key in PUBLIC_KEYS and not np.array_equal(raw[key], value):
                raise ValueError(f"public/evaluator mismatch: {key}")
    attempts = json.loads((directory / "evaluator/attempts.json").read_text(encoding="utf-8"))
    if not isinstance(attempts, list) or not attempts:
        raise ValueError("attempt ledger must be nonempty")
    seeds = manifest.get("seeds", [])
    if not 2 <= len(seeds) <= 4 or any(type(seed) is not int for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("invalid bounded seed accounting")
    expected_files = {"public/inputs.npz", "evaluator/dataset.npz", "evaluator/attempts.json", "evaluator/suite/suite.json"}
    expected_files.update(f"evaluator/suite/cases/{case}.npz" for case in CASES)
    ids, accepted, counts = set(), [], dict.fromkeys(seeds, 0)
    for attempt in attempts:
        uid = attempt.get("twin_id", "")
        if not isinstance(uid, str) or not re.fullmatch(r"[0-9a-f]{20}", uid) or uid in ids:
            raise ValueError("invalid or duplicate attempt ID")
        ids.add(uid)
        seed = attempt.get("seed")
        if type(seed) is not int or seed not in counts or type(attempt.get("accepted")) is not bool:
            raise ValueError("invalid attempt seed/acceptance")
        counts[seed] += 1
        expected_files.add(f"evaluator/raw-{uid}.npz")
        expected_files.update(f"evaluator/snapshots/{uid}-{b}.npz" for b in range(2))
        with np.load(directory / f"evaluator/raw-{uid}.npz", allow_pickle=False) as raw:
            if set(raw.files) != set(RAW_SHAPES):
                raise ValueError("invalid raw schema")
            for key, shape in RAW_SHAPES.items():
                value = raw[key]
                if value.shape != shape or value.dtype.kind not in "fiu" or not np.isfinite(value).all():
                    raise ValueError(f"invalid or non-finite raw {key}")
            mechanisms = raw["mechanisms"]
            if not np.isclose(mechanisms.sum(), 0, atol=1e-12, rtol=0) or not np.all((abs(mechanisms) >= .035) & (abs(mechanisms) <= .075)):
                raise ValueError("invalid mirrored mechanisms")
            snapshots = []
            for branch in range(2):
                snapshot_path = directory / f"evaluator/snapshots/{uid}-{branch}.npz"
                if not snapshot_path.is_file():
                    raise ValueError("missing required snapshot inventory")
                with np.load(snapshot_path, allow_pickle=False) as saved:
                    state = {key: saved[key].copy() for key in saved.files}
                validate_snapshot(state)
                if not np.array_equal(state["body"][0], raw["context"][branch]) or state["elapsed"][0] != 0:
                    raise ValueError("snapshot anchor/context mismatch")
                if not np.isclose(state["com"][0], mechanisms[branch], atol=1e-7, rtol=0):
                    raise ValueError("snapshot mechanism mismatch")
                snapshots.append(state)
            if any(not np.array_equal(snapshots[0][key], snapshots[1][key]) for key in ("body", "mass", "inertia", "elapsed")):
                raise ValueError("twin snapshots differ beyond hidden COM")
            checks, error = attempt_checks(raw["context"], raw["history_effects"], raw["candidate_effects"])
            if checks != attempt.get("checks") or all(checks.values()) != attempt["accepted"] or not np.isclose(error, attempt.get("effect_marginal_error", np.nan), atol=1e-12, rtol=0):
                raise ValueError("attempt admission ledger disagrees with raw outcomes")
            if attempt["accepted"]:
                row = len(accepted)
                if row >= dataset.twins or uid != dataset.twin_ids[row] or seed != dataset.seed_ids[row]:
                    raise ValueError("accepted attempt IDs/seeds disagree with dataset")
                for field in ("history_effects", "candidate_effects", "context"):
                    if not np.array_equal(raw[field], getattr(dataset, field)[row]):
                        raise ValueError(f"raw/dataset {field} mismatch")
                for branch in range(2):
                    if not np.array_equal(raw["history"], dataset.history_actions[row, branch, :, 0]) or not np.array_equal(raw["candidates"], dataset.candidates[row, branch, :, 0]):
                        raise ValueError("raw/dataset action mismatch")
                accepted.append(uid)
    if set(files) != expected_files:
        raise ValueError("artifact inventory differs from required attempt/suite files")
    if any(not 2 <= value <= 16 for value in counts.values()) or set(dataset.seed_ids) != set(seeds):
        raise ValueError("bounded per-seed attempt accounting mismatch")
    if len(accepted) != dataset.twins or manifest.get("accepted") != dataset.twins or manifest.get("attempts") != len(attempts) or manifest.get("retention") != len(accepted) / len(attempts):
        raise ValueError("attempt/retention accounting mismatch")
    if not dataset.history_mask.all() or np.any(dataset.history_times != 0):
        raise ValueError("smoke history must contain independent zero-time anchor probes")
    suite = directory / "evaluator/suite"
    suite_manifest = json.loads((suite / "suite.json").read_text(encoding="utf-8"))
    if any(suite_manifest.get("cases", {}).get(case, {}).get("path") != f"cases/{case}.npz" for case in CASES):
        raise ValueError("noncanonical suite path")
    _, cases = load_suite(suite)
    if dataset_digest(cases["original"]) != digest:
        raise ValueError("suite original differs from replay dataset")
    return manifest, dataset
