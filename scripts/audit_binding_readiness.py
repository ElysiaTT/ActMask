"""Evidence-based readiness audit for the counterfactual binding project.

The audit is deliberately non-promotional: benchmark infrastructure, current
method evidence, source-data feasibility, GPU protocol, GPU implementation,
and permission to start compute are separate decisions.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_SPLITS = {
    "id_test",
    "parameter_ood",
    "sparse_history",
    "observation_noise_ood",
    "held_mechanism",
}

REQUIRED_FILES = (
    "binding_bench/schema.py",
    "binding_bench/interventions.py",
    "binding_bench/adapters.py",
    "binding_bench/metrics.py",
    "binding_bench/evaluator.py",
    "binding_bench/benchmark.py",
    "tests/test_binding_bench.py",
    "docs/counterfactual_binding_benchmark_v1.md",
    "docs/binding_dataset_selection_2026-09.md",
    "docs/binding_gpu_preregister.json",
    "docs/binding_gpu_preregister.md",
    "docs/binding_gpu_environment_checklist.md",
    "configs/binding_gpu_readiness.json",
    "audit/binding_cpu_learned_v1_saturation_diagnosis.md",
    "audit/binding_cpu_v2_antisaturation_report.md",
    "audit/binding_cpu_v3_sysid_residual_report.md",
    "audit/results/binding_bench_example/evaluation.json",
    "audit/results/binding_cpu_learned_v1.json",
    "audit/results/binding_cpu_v2_antisaturation.json",
    "audit/results/binding_cpu_v3_sysid_residual.json",
    "audit/results/binding_dataset_probe.json",
    "audit/results/binding_gpu_readiness_local.json",
)

PLANNED_GPU_MODULES = (
    "actmask/experiments/binding_gpu/snapshot_smoke.py",
    "actmask/experiments/binding_gpu/generate.py",
    "actmask/experiments/binding_gpu/audit_dataset.py",
    "actmask/experiments/binding_gpu/run_state_methods.py",
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_tests(root: Path) -> dict[str, Any]:
    command = [sys.executable, "-m", "pytest", "tests/test_binding_bench.py", "-q"]
    completed = subprocess.run(
        command,
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
        "passed": completed.returncode == 0,
    }


def audit(root: Path, *, run_tests: bool) -> dict[str, Any]:
    missing_files = [relative for relative in REQUIRED_FILES if not (root / relative).is_file()]
    tests = _run_tests(root) if run_tests else {"passed": None, "skipped": True}

    example = _load_json(root / "audit/results/binding_bench_example/evaluation.json")
    example_original = example["scores"][example["candidate_method"]]["original"]
    example_checks = {
        "schema": example.get("schema_version") == "actmask-binding-evaluation-v1",
        "self_test_decision": example.get("decision") == "METHOD_GO",
        "anti_saturation_tpa": 0.5 < float(example_original["tpa"]) < 0.95,
        "anti_saturation_regret": float(example_original["normalized_regret"]) >= 0.01,
        "all_frozen_checks": all(bool(value) for value in example.get("checks", {}).values()),
    }

    historical = {
        name: _load_json(root / relative)["aggregate"]
        for name, relative in {
            "learned_v1": "audit/results/binding_cpu_learned_v1.json",
            "learned_v2": "audit/results/binding_cpu_v2_antisaturation.json",
            "learned_v3": "audit/results/binding_cpu_v3_sysid_residual.json",
        }.items()
    }
    historical_checks = {
        "v1_is_diagnostic_only": historical["learned_v1"]["decision"]
        == "LEARNED_CPU_AUDIT_TRANSPORTS",
        "v2_method_no_go": historical["learned_v2"]["decision"] == "CPU_V2_NO_GO",
        "v3_method_no_go": historical["learned_v3"]["decision"] == "CPU_V3_NO_GO",
    }

    probe = _load_json(root / "audit/results/binding_dataset_probe.json")
    probe_decision = probe["decision"]
    data_checks = {
        "metadata_probe": bool(probe_decision["metadata_passed"]),
        "bounded_file_probe": bool(probe_decision["data_probe_passed"]),
        "primary_maniskill": probe_decision["primary"] == "maniskill",
        "backup_robomimic": probe_decision["backup"] == "robomimic",
        "maniskill_sha_verified": bool(
            probe["datasets"]["maniskill"]["download_verification"]["matches_official_lfs"]
        ),
        "robomimic_sha_verified": bool(
            probe["datasets"]["robomimic"]["download_verification"]["matches_official_lfs"]
        ),
        "temporary_downloads_deleted": bool(probe["policy"]["temporary_downloads_deleted"]),
    }

    prereg = _load_json(root / "docs/binding_gpu_preregister.json")
    readiness_config = _load_json(root / "configs/binding_gpu_readiness.json")
    protocol_checks = {
        "frozen_before_gpu": prereg.get("status") == "FROZEN_BEFORE_GPU_ACCESS",
        "all_required_splits": set(
            prereg["method_admission_gates"]["required_splits"]
        )
        == EXPECTED_SPLITS,
        "five_seeds": len(prereg["sampling"]["official_seeds"]) == 5,
        "candidate_rollout_restores_anchor": bool(
            prereg["sampling"]["restore_anchor_before_every_probe_and_candidate"]
        ),
        "compute_bounded": (
            float(prereg["compute_boundaries"]["state_pilot"]["maximum_gpu_hours"]) > 0
            and float(prereg["compute_boundaries"]["state_pilot"]["maximum_output_gib"]) > 0
        ),
        "cost_fail_closed": (
            prereg["compute_boundaries"]["approved_cost_amount"] is None
            and not prereg["compute_boundaries"]["cost_authorized"]
            and readiness_config["approved_cost_amount"] is None
            and not readiness_config["cost_authorized"]
            and not readiness_config["remote_host_approved"]
        ),
    }

    gpu_host = _load_json(root / "audit/results/binding_gpu_readiness_local.json")
    gpu_module_presence = {
        relative: (root / relative).is_file() for relative in PLANNED_GPU_MODULES
    }
    transport_path = root / "audit/results/binding_cpu_transport.json"
    transport = _load_json(transport_path) if transport_path.is_file() else {}
    cpu_transport_ready = (
        transport.get("g1", {}).get("passed") is True
        and transport.get("g2", {}).get("passed") is True
        and all(gpu_module_presence[path] for path in PLANNED_GPU_MODULES[:3])
    )
    harness_ready = (
        not missing_files
        and tests.get("passed") is not False
        and all(example_checks.values())
    )
    current_method_ready = (
        all(historical_checks.values()) and historical["learned_v3"]["passed"]
    )
    data_source_ready = all(data_checks.values())
    gpu_protocol_ready = all(protocol_checks.values())
    gpu_generator_ready = all(gpu_module_presence.values())
    gpu_host_ready = gpu_host.get("decision") == "GPU_START_GO"
    gpu_start_ready = gpu_protocol_ready and gpu_generator_ready and gpu_host_ready

    return {
        "schema_version": "actmask-binding-project-readiness-v1",
        "utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "components": {
            "cpu_g1_g2_transport": {
                "decision": "GO" if cpu_transport_ready else "NO_GO",
                "evidence": "audit/results/binding_cpu_transport.json",
                "scope": "Archived CPU replay/schema/isolation/plumbing only; not scientific admission or GPU execution.",
            },
            "cpu_benchmark_harness": {
                "decision": "GO" if harness_ready else "NO_GO",
                "missing_files": missing_files,
                "tests": tests,
                "example_checks": example_checks,
                "example_metrics": {
                    "tpa": example_original["tpa"],
                    "normalized_regret": example_original["normalized_regret"],
                    "top1_accuracy": example_original["top1_accuracy"],
                },
            },
            "current_learned_method_claim": {
                "decision": "GO" if current_method_ready else "NO_GO",
                "historical_checks": historical_checks,
                "decisions": {
                    name: value["decision"] for name, value in historical.items()
                },
                "reason": (
                    "The benchmark plumbing is usable, but v2 and v3 fail frozen method gates; "
                    "v1 remains a saturation diagnostic."
                ),
            },
            "public_data_source_selection": {
                "decision": "GO" if data_source_ready else "NO_GO",
                "checks": data_checks,
                "primary": probe_decision["primary"],
                "backup": probe_decision["backup"],
                "scope": "File authenticity and state/action structure only; not simulator replay.",
            },
            "gpu_protocol": {
                "decision": "GO" if gpu_protocol_ready else "NO_GO",
                "checks": protocol_checks,
            },
            "gpu_generator_implementation": {
                "decision": "GO" if gpu_generator_ready else "NO_GO",
                "module_presence": gpu_module_presence,
                "reason": "CPU G1/G2 fixture implemented; full GPU backend, G3 scientific admission and G4 methods remain pending.",
            },
            "gpu_host_and_authorization": {
                "decision": "GO" if gpu_host_ready else "NO_GO",
                "local_preflight_decision": gpu_host.get("decision"),
                "failures": gpu_host.get("failures", []),
            },
        },
        "overall_decision": "GPU_START_GO" if gpu_start_ready else "GPU_START_NO_GO",
        "next_gate": (
            "Pull the frozen GitHub commit on Linux, reproduce CPU G1/G2, then pass remote G0 "
            "before separately implementing GPU backend and scientific G3/G4."
        ),
        "scope_note": (
            "GO for the CPU harness or source selection is not evidence of a learned-method gain "
            "and is not permission to start GPU compute."
        ),
    }


def render_markdown(result: dict[str, Any]) -> str:
    components = result["components"]
    harness = components["cpu_benchmark_harness"]
    data = components["public_data_source_selection"]
    gpu = components["gpu_host_and_authorization"]
    rows = []
    labels = {
        "cpu_g1_g2_transport": "CPU G1/G2 transport smoke (archived evidence)",
        "cpu_benchmark_harness": "CPU benchmark harness",
        "current_learned_method_claim": "Current learned-method claim",
        "public_data_source_selection": "Public data/source selection",
        "gpu_protocol": "Frozen GPU protocol",
        "gpu_generator_implementation": "G1-G4 generator/method modules",
        "gpu_host_and_authorization": "GPU host and authorization",
    }
    for key, label in labels.items():
        rows.append(f"| {label} | **{components[key]['decision']}** |")
    failures = ", ".join(gpu["failures"]) or "none"
    tests = harness["tests"]
    test_evidence = tests.get("stdout", "tests skipped").replace("\n", " ")
    return f"""# Binding benchmark readiness audit

Generated: {result['utc']}
Overall decision: **{result['overall_decision']}**

| Layer | Decision |
|---|---|
{chr(10).join(rows)}

## Evidence

- Targeted CPU tests: `{test_evidence}`
- Anti-saturation example: TPA `{harness['example_metrics']['tpa']:.4f}`, normalized regret `{harness['example_metrics']['normalized_regret']:.4f}`, Top-1 `{harness['example_metrics']['top1_accuracy']:.4f}`.
- Public source choice: `{data['primary']}` primary, `{data['backup']}` backup; both bounded files passed official LFS SHA and structural inspection.
- Current learned result remains `CPU_V3_NO_GO`; the synthetic harness `METHOD_GO` is only a self-test.
- Local/authorization preflight failures: `{failures}`.

## Interpretation

The reusable CPU benchmark and the public-source decision are ready. Actual GPU
execution is not ready: the current host is unsupported, spending and the exact
remote host are unapproved. CPU G1/G2 is a limited physical transport fixture;
full GPU backend, G3 scientific admission and G4 methods remain pending.
Archived smoke reports are not evidence that the current checkout was rerun.

Next gate: {result['next_gate']}
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    result = audit(root, run_tests=not args.skip_tests)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.markdown_output is not None:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(render_markdown(result), encoding="utf-8")
    print(
        json.dumps(
            {
                "overall_decision": result["overall_decision"],
                "component_decisions": {
                    name: value["decision"] for name, value in result["components"].items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
