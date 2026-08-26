#!/usr/bin/env python3
"""Verify the frozen WAV MiniGrid external-audit report without retraining."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = ROOT / "outputs/actmask/icra2027_wav_external_audit/audit.json"
EXPECTED_REPORT_SHA256 = "e9e1aaa2f37cc3b705b70f71f8e1439fdeb1d0f426e1d50b58ca331c6e6091f0"
EXPECTED_PROTOCOL_SHA256 = "4a76b946de840b6b8f0357d0c24da4034bb3216ed3a635598ac665c13807ea22"
EXPECTED_RUNNER_SHA256 = "fa368befc6f300bcae95db8462a7f3c4d2b2631aeeea0ae26a6be8546466d9d4"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_runner():
    path = ROOT / "scripts/run_icra2027_wav_external_audit.py"
    spec = importlib.util.spec_from_file_location("icra2027_wav_external_runner", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def close(left: float, right: float, tolerance: float = 1e-12) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)


def verify(path: Path) -> dict:
    runner = load_runner()
    data = json.loads(path.read_text(encoding="utf-8"))
    checks = []

    def add(name: str, passed: bool, detail):
        checks.append({"name": name, "status": "PASS" if passed else "FAIL", "detail": detail})

    add("report_hash", sha256(path) == EXPECTED_REPORT_SHA256, sha256(path))
    protocol_path = ROOT / data["protocol"]
    add(
        "protocol_hash",
        protocol_path.is_file()
        and sha256(protocol_path) == EXPECTED_PROTOCOL_SHA256
        and data["protocol_sha256"] == EXPECTED_PROTOCOL_SHA256,
        data.get("protocol_sha256"),
    )
    runner_path = ROOT / "scripts/run_icra2027_wav_external_audit.py"
    add("runner_hash", sha256(runner_path) == EXPECTED_RUNNER_SHA256, sha256(runner_path))
    add(
        "frozen_config",
        data["schema_version"] == 1
        and data["config"]["seeds"] == list(runner.SEEDS)
        and data["config"]["complexities"] == list(runner.COMPLEXITIES)
        and data["config"]["max_epochs"] == 200
        and data["config"]["patience"] == 25
        and data["config"]["batch_size"] == 64
        and data["config"]["learning_rate"] == 1e-3
        and "runtime_seconds" not in data,
        data["config"],
    )
    add(
        "upstream_provenance",
        data["upstream"]["commit"] == runner.UPSTREAM_COMMIT
        and data["upstream"]["sha256"] == runner.EXPECTED_HASHES,
        data["upstream"],
    )
    expected_seed_keys = {str(seed) for seed in runner.SEEDS}
    method_keys_ok = (
        set(data["methods"]) == {
            "train_prior", "current_only", "next_only", "paired",
            "sparse_idm", "transition_rule",
        }
        and all(
            set(data["methods"][method]) == expected_seed_keys
            for method in ("train_prior", "current_only", "next_only", "paired", "sparse_idm")
        )
        and set(data["methods"]["transition_rule"]) == {"fixed"}
    )
    add("method_seed_matrix", method_keys_ok, {k: sorted(v) for k, v in data["methods"].items()})

    cell_failures = []
    for method, seeds in data["methods"].items():
        for seed, entry in seeds.items():
            for complexity in runner.COMPLEXITIES:
                key = str(complexity)
                cell = entry["tests"][key]
                confusion = np.asarray(cell["confusion"], dtype=np.int64)
                expected_count = int(data["dataset_counts"]["tests"][key])
                conditions = [
                    confusion.shape == (7, 7),
                    int(confusion.sum()) == expected_count == int(cell["count"]),
                    int(np.trace(confusion)) == int(cell["action_correct"]),
                    close(cell["action_accuracy"], int(cell["action_correct"]) / expected_count),
                    close(cell["dynamic_accuracy"], int(cell["dynamic_correct"]) / int(cell["dynamic_total"])),
                    len(cell["prediction_sha256"]) == 64,
                ]
                if not all(conditions):
                    cell_failures.append(f"{method}/{seed}/{key}")
    add("raw_confusion_and_counts", not cell_failures, cell_failures)

    recomputed = {
        method: runner.summarize_method(per_seed)
        for method, per_seed in data["methods"].items()
    }
    add("summary_recomputation", recomputed == data["summaries"], "exact JSON equality")
    decision = runner.evaluate_decision(recomputed)
    add(
        "preregistered_decision",
        decision == data["decision"]
        and decision["pass"] is True
        and len(decision["gates"]) == 15
        and all(gate["pass"] for gate in decision["gates"]),
        decision,
    )
    return {
        "schema": "actmask-icra2027-wav-external-audit-verification-v1",
        "passed": all(check["status"] == "PASS" for check in checks),
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(args.report.resolve())
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"passed": result["passed"], "checks": len(result["checks"])}, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
