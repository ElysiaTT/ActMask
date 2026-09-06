"""Independent G2 artifact/replay audit and existing evaluator plumbing check."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from binding_bench.adapters import CallableAdapter, run_adapter
from binding_bench.evaluator import evaluate_suite, write_evaluation
from binding_bench.schema import dataset_arrays, load_dataset
from .generate import PUBLIC_KEYS, sha256, utility
from .rod import RodCOMEnv, load_snapshot


def probe_nearest(inputs):
    """Frozen, non-learned nearest probe; only a plumbing fixture."""
    distances = np.abs(inputs.candidates[..., 0, None] - inputs.history_actions[..., None, :, 0])
    nearest = distances.argmin(axis=-1)
    effects = np.take_along_axis(inputs.history_effects, nearest[..., None], axis=2)
    return {"scores": utility(effects), "predicted_effects": effects}


def uniform(inputs):
    return np.zeros(inputs.candidates.shape[:3])


def audit(directory: Path, report_dir: Path):
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "actmask-cpu-g2-v1":
        raise ValueError("unsupported G2 manifest")
    for relative, expected in manifest["files"].items():
        path = (directory / relative).resolve()
        if not path.is_relative_to(directory.resolve()) or sha256(path) != expected:
            raise ValueError(f"artifact digest mismatch or unsafe path: {relative}")
    dataset = load_dataset(directory / "evaluator/dataset.npz")
    with np.load(directory / "public/inputs.npz", allow_pickle=False) as raw:
        if set(raw.files) != PUBLIC_KEYS:
            raise ValueError("public schema contains missing or hidden fields")
        arrays = dataset_arrays(dataset)
        for key in PUBLIC_KEYS:
            if not np.array_equal(raw[key], arrays[key]):
                raise ValueError(f"public/evaluator mismatch: {key}")
    attempts = json.loads((directory / "evaluator/attempts.json").read_text(encoding="utf-8"))
    if len(attempts) != manifest["attempts"] or sum(a["accepted"] for a in attempts) != dataset.twins:
        raise ValueError("attempt/retention accounting mismatch")
    env = RodCOMEnv()
    max_replay, marginal_error = 0., 0.
    try:
        for twin, uid in enumerate(dataset.twin_ids):
            with np.load(directory / f"evaluator/raw-{uid}.npz", allow_pickle=False) as raw:
                for branch in range(2):
                    state = load_snapshot(directory / f"evaluator/snapshots/{uid}-{branch}.npz")
                    if not np.isclose(state["com"][0], raw["mechanisms"][branch], atol=1e-7, rtol=0):
                        raise ValueError("snapshot mechanism mismatch")
                    if not np.array_equal(state["body"][0], dataset.context[twin, branch]):
                        raise ValueError("anchor/context mismatch")
                    # Reverse execution order catches missing restore between candidates.
                    actions = np.concatenate([raw["history"], raw["candidates"]])
                    expected_effects = np.concatenate([dataset.history_effects[twin, branch], dataset.candidate_effects[twin, branch]])
                    for index in reversed(range(len(actions))):
                        effect, trajectory = env.rollout(state, float(actions[index]))
                        max_replay = max(max_replay, float(np.max(np.abs(trajectory - raw["trajectories"][branch, index]))),
                                         float(np.max(np.abs(effect - expected_effects[index]))))
            marginal_error = max(marginal_error, float(np.max(np.abs(
                np.sort(dataset.history_effects[twin, 0], axis=0) - np.sort(dataset.history_effects[twin, 1], axis=0)))))
    finally:
        env.close()
    if max_replay > 1e-5 or marginal_error > 1e-4:
        raise ValueError("replay or hidden-mechanism marginal isolation failed")
    if not np.allclose(utility(dataset.candidate_effects), dataset.true_utility, atol=1e-12, rtol=0):
        raise ValueError("utility must be derived from replayed effects")
    suite = directory / "evaluator/suite"
    methods = {"probe_nearest": probe_nearest, "uniform": uniform}
    for name, function in methods.items():
        run_adapter(CallableAdapter(name, function), suite, report_dir / name)
    result = evaluate_suite(suite, {name: report_dir / name for name in methods},
                            candidate_method="probe_nearest", baseline_methods=("uniform",))
    write_evaluation(report_dir / "evaluation.json", result)
    # No threshold relaxation and no METHOD_GO required for a transport smoke.
    report = {"schema_version": "actmask-cpu-g2-audit-v1", "passed": True,
              "twins": dataset.twins, "attempts": manifest["attempts"], "retention": manifest["retention"],
              "maximum_replay_error": max_replay, "effect_marginal_error": marginal_error,
              "public_allowlist_passed": True, "benchmark_cases": list(result["scores"]["uniform"]),
              "method_decision": result["decision"], "admission_evaluable": result["admission_evaluable"],
              "scope": "G1/G2 plumbing only; no model or scientific dataset admission"}
    (report_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.dataset_dir, args.output_dir), indent=2))
