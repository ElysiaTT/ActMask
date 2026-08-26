"""Conditional orchestration, audit, and terminal packaging for v2.1."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from actmask.experiments.history_action_binding_v2_1.audit import audit_dataset
from actmask.experiments.history_action_binding_v2_1.common import (
    OUTPUT_ROOT,
    PROJECT_ROOT,
    PROTOCOL,
    SOURCE_FILES,
    TASKS,
    V2_ROOT,
    canonical_json,
    scientific_core_manifest,
    sha256_bytes,
    sha256_file,
    source_manifest,
    write_json,
)
from actmask.experiments.history_action_binding_v2_1.freeze import verify_protected_boundaries
from actmask.experiments.history_action_binding_v2_1.generate import generation_refusal


def _record(path: Path) -> dict:
    if path.is_symlink():
        target = os.readlink(path)
        return {
            "kind": "symlink",
            "bytes": len(target.encode("utf-8")),
            "sha256": sha256_bytes(target.encode("utf-8")),
            "target": target,
        }
    return {"kind": "file", "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def generated_manifest() -> dict:
    generated_path = OUTPUT_ROOT / "generated_manifest.json"
    paths = [
        path for path in OUTPUT_ROOT.rglob("*")
        if (path.is_file() or path.is_symlink()) and path != generated_path
    ]
    report = PROJECT_ROOT / "docs/history_action_binding_v2_1_report.md"
    if report.exists():
        paths.append(report)
    files = {str(path.relative_to(PROJECT_ROOT)): _record(path) for path in sorted(paths)}
    return {
        "files": files,
        "combined_sha256": sha256_bytes(canonical_json(files).encode("utf-8")),
    }


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def verify_frozen_core() -> dict:
    freeze_path = OUTPUT_ROOT / "freeze_record.json"
    if not freeze_path.exists():
        raise FileNotFoundError(freeze_path)
    freeze = json.loads(freeze_path.read_text())
    current = scientific_core_manifest()
    return {
        "passed": current == freeze["scientific_core_manifest"],
        "expected_combined_sha256": freeze["scientific_core_manifest"]["combined_sha256"],
        "current_combined_sha256": current["combined_sha256"],
    }


def run_formal_audits() -> dict:
    output = OUTPUT_ROOT / "formal_audit"
    output.mkdir(parents=True, exist_ok=True)
    tasks = {}
    for task in TASKS:
        dataset_path = OUTPUT_ROOT / "pilot" / task / "dataset.npz"
        report_path = OUTPUT_ROOT / "pilot" / task / "generation_report.json"
        if not dataset_path.exists() or not report_path.exists():
            raise FileNotFoundError(f"formal pilot missing for {task}")
        generation = json.loads(report_path.read_text())
        if generation.get("status") != "passed":
            raise RuntimeError(f"formal pilot integrity failed for {task}")
        dataset = _load_npz(dataset_path)
        task_output = output / task
        task_output.mkdir(parents=True, exist_ok=True)
        tasks[task] = audit_dataset(dataset, task_output, generation["integrity"])
    record = {
        "status": "BINDING_AUDIT_PASS" if all(item["gate"] == "BINDING_AUDIT_PASS" for item in tasks.values()) else "FORMAL_BINDING_AUDIT_FAIL",
        "passed": bool(all(item["gate"] == "BINDING_AUDIT_PASS" for item in tasks.values())),
        "tasks": tasks,
    }
    write_json(output / "formal_audit_summary.json", record)
    return record


def _npz_schema(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        return {
            key: {"dtype": str(archive[key].dtype), "shape": list(archive[key].shape)}
            for key in archive.files
        }


def write_schemas() -> tuple[dict, dict]:
    symbolic = {}
    simulator = {"smoke": {}, "pilot": {}}
    for task in TASKS:
        path = OUTPUT_ROOT / "official_symbolic_preflight" / task / "symbolic_samples.npz"
        if path.exists():
            symbolic[task] = _npz_schema(path)
        for phase in ("smoke", "pilot"):
            path = OUTPUT_ROOT / phase / task / "dataset.npz"
            if path.exists():
                simulator[phase][task] = _npz_schema(path)
    dataset_schema = {
        "schema_version": "history_action_binding_v2_1_v1",
        "symbolic": symbolic,
        "simulator": simulator,
    }
    trace_fields = {
        "history_trace_public": "per-step public state for probes and matching tail",
        "history_trace_action": "per-step 2D command",
        "history_trace_phase": "0=probe pulse, 1=matching controller",
        "history_trace_probe_index": "probe assignment for every history step",
        "candidate_trace_public": "public decision trajectory",
        "candidate_trace_response_pos": "per-step PhysX response position",
        "candidate_trace_response_vel": "per-step PhysX response velocity",
        "candidate_trace_actuator_pos": "per-step command proxy position",
        "candidate_trace_target_pos": "per-step target position",
        "candidate_trace_distance": "per-step target distance used to recompute utility/success",
        "candidate_trace_contact_magnitude": "per-step net contact magnitude",
        "candidate_trace_contact": "per-step contact boolean",
        "first_success_step": "first trace index within success radius, else -1",
        "initial_state_hash": "full reset state SHA-256",
        "decision_state_hash": "full matched decision state SHA-256",
        "final_state_hash": "full final state SHA-256 per candidate",
    }
    trace_schema = {
        "status": "instantiated" if simulator["smoke"] or simulator["pilot"] else "not_instantiated_before_gpu_rollout",
        "fields": trace_fields,
        "trace_label_rule": "success = any(candidate_trace_distance <= success_radius)",
        "trace_utility_rule": "utility = -min(candidate_trace_distance)",
    }
    write_json(OUTPUT_ROOT / "dataset_schema.json", dataset_schema)
    write_json(OUTPUT_ROOT / "trace_schema.json", trace_schema)
    return dataset_schema, trace_schema


def _reports(phase: str) -> dict[str, dict | None]:
    records = {}
    for task in TASKS:
        path = OUTPUT_ROOT / phase / task / "generation_report.json"
        records[task] = json.loads(path.read_text()) if path.exists() else None
    return records


def write_comparison(symbolic: dict) -> dict:
    v2 = json.loads((V2_ROOT / "summary.json").read_text())
    rows = []
    for task in TASKS:
        old = v2["symbolic_preflight"]["tasks"][task]
        new = symbolic["tasks"][task]
        rows.extend((
            {
                "version": "v2",
                "task": task,
                "slot_min": min(min(row) for row in old["slot_balance_table"]),
                "slot_max": max(max(row) for row in old["slot_balance_table"]),
                "fourier": old["fourier_score"],
                "passed": old["passed"],
            },
            {
                "version": "v2.1",
                "task": task,
                "slot_min": new["details"]["slot_min"],
                "slot_max": new["details"]["slot_max"],
                "fourier": new["details"]["fourier_score"],
                "passed": new["passed"],
            },
        ))
    comparison = {
        "v2_gate": v2["gate"],
        "v2_1_symbolic_gate": symbolic["status"],
        "rows": rows,
        "task_a_repair": "two probe radii + public-quadrant novel candidate directions + cyclic slots",
        "task_b_change": "cyclic slots only",
    }
    write_json(OUTPUT_ROOT / "v2_v2_1_comparison.json", comparison)
    return comparison


def _commands(symbolic_passed: bool, gate: str) -> list[str]:
    python = "/data/envs/actmask-audit/bin/python"
    prefix = f"PYTHONDONTWRITEBYTECODE=1 PYTHONPATH={PROJECT_ROOT} {python} -m"
    commands = [
        f"{prefix} actmask.experiments.history_action_binding_v2_1.diagnose_v2",
        f"{prefix} actmask.experiments.history_action_binding_v2_1.development",
        f"{prefix} unittest tests.test_history_action_binding_v2_1 -v",
        f"{prefix} actmask.experiments.history_action_binding_v2_1.freeze",
        f"{prefix} actmask.experiments.history_action_binding_v2_1.symbolic --official",
    ]
    if symbolic_passed and gate == "GPU_SMOKE_EXECUTION_ERROR":
        runtime_home = OUTPUT_ROOT / "runtime_home"
        commands.append(
            f"HOME={runtime_home} {prefix} actmask.experiments.history_action_binding_v2_1.generate "
            f"--phase smoke --task {TASKS[0]}  # exited 1 before gym.make"
        )
        commands.append(
            f"{prefix} actmask.experiments.history_action_binding_v2_1.run_v2_1 --stage record-smoke-error"
        )
    elif symbolic_passed:
        runtime_home = OUTPUT_ROOT / "runtime_home"
        for phase in ("smoke", "pilot"):
            for task in TASKS:
                commands.append(f"HOME={runtime_home} {prefix} actmask.experiments.history_action_binding_v2_1.generate --phase {phase} --task {task}")
        commands.append(f"{prefix} actmask.experiments.history_action_binding_v2_1.run_v2_1 --stage audit")
    commands.extend((
        f"{prefix} actmask.experiments.history_action_binding_v2_1.run_v2_1 --stage finalize",
        f"{prefix} unittest tests.test_history_action_binding_v2_1 -v",
    ))
    return commands


def _terminal_state(symbolic: dict) -> tuple[str, dict | None, dict, dict]:
    smoke = _reports("smoke")
    pilot = _reports("pilot")
    if not symbolic["passed"]:
        if (OUTPUT_ROOT / "smoke").exists() or (OUTPUT_ROOT / "pilot").exists():
            raise RuntimeError("GPU directories exist despite failed symbolic preflight")
        return "SYMBOLIC_PREFLIGHT_FAIL", None, smoke, pilot
    execution_error = OUTPUT_ROOT / "smoke_execution_error.json"
    if execution_error.exists():
        if (OUTPUT_ROOT / "smoke").exists() or (OUTPUT_ROOT / "pilot").exists():
            raise RuntimeError("GPU directories exist despite recorded pre-environment smoke failure")
        return "GPU_SMOKE_EXECUTION_ERROR", None, smoke, pilot
    if any(record is None for record in smoke.values()):
        raise RuntimeError("symbolic passed; both smoke tasks must be executed before finalize")
    if any(record["status"] != "passed" for record in smoke.values()):
        if (OUTPUT_ROOT / "pilot").exists():
            raise RuntimeError("pilot exists despite failed smoke")
        return "GPU_SMOKE_DATA_INTEGRITY_FAIL", None, smoke, pilot
    if any(record is None for record in pilot.values()):
        raise RuntimeError("smoke passed; both formal pilot tasks must be executed before finalize")
    if any(record["status"] != "passed" for record in pilot.values()):
        return "GPU_FORMAL_DATA_INTEGRITY_FAIL", None, smoke, pilot
    audit_path = OUTPUT_ROOT / "formal_audit" / "formal_audit_summary.json"
    audit = json.loads(audit_path.read_text()) if audit_path.exists() else run_formal_audits()
    return audit["status"], audit, smoke, pilot


def _report(summary: dict) -> str:
    symbolic = summary["symbolic_preflight"]
    lines = [
        "# History–Action Binding Audit v2.1 最终报告", "",
        f"终止状态：**{summary['gate']}**。本 Goal 没有训练任何 learned model。", "",
        "## 最终五个问题", "",
        f"1. slot balance 是否严格修复：**{'是' if summary['slot_balance_strictly_fixed'] else '否'}**。",
        f"2. Task A/Task B symbolic 是否通过：**{symbolic['tasks'][TASKS[0]]['passed']} / {symbolic['tasks'][TASKS[1]]['passed']}**。",
        f"3. GPU 数据是否真实、完整、可由 trace 重算：**{summary['gpu_trace_conclusion']}**。",
        f"4. binding 与 novel-candidate SysID 是否成立：**{summary['binding_sysid_conclusion']}**。",
        f"5. 是否授权 learned verifier Goal：**{'授权' if summary['learned_training_authorized'] else '不授权'}**。", "",
        "## Official symbolic preflight", "",
        "| Task | Slot min/max | Marginal max | Arrow | Nearest | kNN | Fourier | Binding drop | Pass |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for task in TASKS:
        item = symbolic["tasks"][task]
        detail = item["details"]
        lines.append(
            f"| {task} | {detail['slot_min']}/{detail['slot_max']} | {max(detail['marginal_scores'].values()):.4f} | "
            f"{detail['interaction_scores']['arrow_action_compatibility']:.4f} | {detail['interaction_scores']['nearest_historical_action']:.4f} | "
            f"{detail['interaction_scores']['knn_action_result_lookup']:.4f} | {detail['fourier_score']:.4f} | "
            f"{detail['binding_drop']:.4f} | {item['passed']} |"
        )
    lines.extend((
        "", "## v2 失败诊断与 v2.1 修复", "",
        "v2 的124个 Task A miss全部位于105°/285°的四个理论 utility-tie templates；float32残差被固定metric当成有方向样本。"
        "v2.1 保持metric与0.90阈值不变，增加第二个probe半径、移除裁剪tie candidates，并用cyclic Latin square把每格固定为4。", "",
        "Task B 的科学设计没有依据开发结果修改，official seed由冻结哈希导出。", "",
        "## GPU 与正式审计", "",
        summary["gpu_detail"], "",
        "## 证据边界", "",
        f"Protected project/environment unchanged: `{summary['protected_boundary_unchanged']}`；scientific core frozen: `{summary['frozen_core_unchanged']}`。",
        f"Candidate executions: `{summary['candidate_executions']}/{PROTOCOL['candidate_execution_budget']}`；output bytes: `{summary['output_bytes']}/{PROTOCOL['output_byte_budget']}`。", "",
        "## 可复制命令", "", "```bash", *summary["commands"], "```", "",
    ))
    return "\n".join(lines)


def finalize() -> dict:
    symbolic_path = OUTPUT_ROOT / "official_symbolic_preflight" / "symbolic_preflight_summary.json"
    if not symbolic_path.exists():
        raise FileNotFoundError("official symbolic preflight is missing")
    symbolic = json.loads(symbolic_path.read_text())
    gate, formal_audit, smoke, pilot = _terminal_state(symbolic)
    if not symbolic["passed"]:
        generation_refusal("smoke", TASKS[0], "official symbolic preflight failed")
    frozen_core = verify_frozen_core()
    protected = verify_protected_boundaries()
    if not frozen_core["passed"] or not protected["passed"]:
        raise RuntimeError("frozen source or protected boundary changed")
    write_json(OUTPUT_ROOT / "protected_boundary_verification.json", protected)
    prereg_path = PROJECT_ROOT / "docs/history_action_binding_v2_1_preregister.md"
    (OUTPUT_ROOT / "preregister.sha256").write_text(
        sha256_file(prereg_path) + "  docs/history_action_binding_v2_1_preregister.md\n"
    )
    dataset_schema, trace_schema = write_schemas()
    comparison = write_comparison(symbolic)
    source = source_manifest()
    write_json(OUTPUT_ROOT / "source_manifest.json", source)
    candidate_executions = sum(
        int(record.get("candidate_executions", 0))
        for group in (smoke, pilot) for record in group.values() if record is not None
    )
    if candidate_executions > int(PROTOCOL["candidate_execution_budget"]):
        raise RuntimeError("candidate execution budget exceeded")
    simulator_generated = candidate_executions > 0
    gpu_reports = [record for group in (smoke, pilot) for record in group.values() if record]
    gpu_all_integrity = bool(gpu_reports) and all(record["status"] == "passed" for record in gpu_reports)
    formal_pass = formal_audit is not None and formal_audit["passed"]
    authorized = bool(gate == "BINDING_AUDIT_PASS" and formal_pass and gpu_all_integrity)
    binding_pass = bool(formal_pass and all(
        item["pass_conditions"]["binding_drop"] and item["pass_conditions"]["sysid_novel"]
        for item in formal_audit["tasks"].values()
    )) if formal_audit else False
    commands = _commands(symbolic["passed"], gate)
    write_json(OUTPUT_ROOT / "commands.json", {
        "cwd": str(PROJECT_ROOT),
        "python": "/data/envs/actmask-audit/bin/python",
        "environment_modified": False,
        "commands": commands,
    })
    slot_fixed = all(
        item["checks"]["candidate_slots_exact"]
        and item["details"]["slot_min"] == 4 and item["details"]["slot_max"] == 4
        for item in symbolic["tasks"].values()
    )
    smoke_execution_error = None
    error_path = OUTPUT_ROOT / "smoke_execution_error.json"
    if error_path.exists():
        smoke_execution_error = json.loads(error_path.read_text())
    if not simulator_generated and gate == "GPU_SMOKE_EXECUTION_ERROR":
        gpu_conclusion = "未生成；冻结 generator 在创建环境前发生执行错误，不能声称真实GPU数据成立"
        gpu_detail = (
            "Official symbolic 已通过，但 frozen generate.py 在 gym.make 前因 PROJECT_ROOT 未导入而 NameError；"
            "按 smoke failure 规则停止，未运行 Task B smoke 或 formal pilot。"
        )
    elif not simulator_generated:
        gpu_conclusion = "未生成；symbolic gate失败后按预注册停止，不能声称真实GPU数据成立"
        gpu_detail = "Official symbolic 未全通过，因此没有创建 GPU smoke/pilot，candidate executions=0。"
    else:
        gpu_conclusion = "是" if gpu_all_integrity else "否"
        gpu_detail = f"GPU reports integrity all passed={gpu_all_integrity}；formal audit gate={gate}。"
    summary = {
        "gate": gate,
        "slot_balance_strictly_fixed": slot_fixed,
        "symbolic_preflight": symbolic,
        "simulator_data_generated": simulator_generated,
        "gpu_data_integrity_passed": gpu_all_integrity,
        "gpu_trace_conclusion": gpu_conclusion,
        "gpu_detail": gpu_detail,
        "formal_audit": formal_audit,
        "binding_and_novel_candidate_sysid_established": binding_pass,
        "binding_sysid_conclusion": "成立" if binding_pass else "未在两个正式GPU任务上同时成立",
        "learned_training_authorized": authorized,
        "learned_models_trained": [],
        "candidate_executions": candidate_executions,
        "candidate_execution_budget": PROTOCOL["candidate_execution_budget"],
        "output_bytes": 0,
        "output_byte_budget": PROTOCOL["output_byte_budget"],
        "smoke_reports": smoke,
        "pilot_reports": pilot,
        "smoke_execution_error": smoke_execution_error,
        "dataset_schema_status": trace_schema["status"],
        "comparison": comparison,
        "frozen_core_unchanged": frozen_core["passed"],
        "protected_boundary_unchanged": protected["passed"],
        "source_combined_sha256": source["combined_sha256"],
        "commands": commands,
    }
    summary_path = OUTPUT_ROOT / "summary.json"
    report_path = OUTPUT_ROOT / "audit_report.md"
    docs_path = PROJECT_ROOT / "docs/history_action_binding_v2_1_report.md"
    for _ in range(8):
        write_json(summary_path, summary)
        report = _report(summary)
        report_path.write_text(report)
        docs_path.write_text(report)
        actual = sum(path.lstat().st_size for path in OUTPUT_ROOT.rglob("*") if path.is_file() or path.is_symlink())
        if actual == summary["output_bytes"]:
            break
        summary["output_bytes"] = actual
    write_json(summary_path, summary)
    report = _report(summary)
    report_path.write_text(report)
    docs_path.write_text(report)
    write_json(OUTPUT_ROOT / "terminal_status.json", {
        "gate": gate,
        "learned_training_authorized": authorized,
        "candidate_executions": candidate_executions,
        "official_symbolic_passed": symbolic["passed"],
    })
    final_manifest = generated_manifest()
    write_json(OUTPUT_ROOT / "generated_manifest.json", final_manifest)
    return summary


def record_smoke_execution_error() -> dict:
    symbolic_path = OUTPUT_ROOT / "official_symbolic_preflight" / "symbolic_preflight_summary.json"
    symbolic = json.loads(symbolic_path.read_text())
    if not symbolic.get("passed"):
        raise RuntimeError("this error record is only valid after a passed official symbolic preflight")
    if (OUTPUT_ROOT / "smoke").exists() or (OUTPUT_ROOT / "pilot").exists():
        raise RuntimeError("cannot record pre-environment error after GPU output exists")
    frozen_core = verify_frozen_core()
    if not frozen_core["passed"]:
        raise RuntimeError("scientific core changed after official preflight")
    record = {
        "status": "GPU_SMOKE_EXECUTION_ERROR",
        "phase": "smoke",
        "task": TASKS[0],
        "command_exit_code": 1,
        "generation_started": False,
        "gpu_environment_created": False,
        "gym_make_reached": False,
        "candidate_executions": 0,
        "error_type": "NameError",
        "error_message": "name 'PROJECT_ROOT' is not defined",
        "source_location": "actmask/experiments/history_action_binding_v2_1/generate.py::_prepare_runtime_home",
        "call_order_evidence": "The exception occurs in _prepare_runtime_home before output.mkdir and gym.make in generate_task.",
        "scientific_core_frozen_unchanged": True,
        "scientific_core_combined_sha256": frozen_core["current_combined_sha256"],
        "formal_pilot_allowed": False,
        "learned_training_authorized": False,
        "remediation": "Create a separately versioned v2.2; do not patch or rerun the frozen v2.1 scientific core.",
    }
    write_json(OUTPUT_ROOT / "smoke_execution_error.json", record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("audit", "record-smoke-error", "finalize"), required=True)
    args = parser.parse_args()
    if args.stage == "audit":
        result = run_formal_audits()
    elif args.stage == "record-smoke-error":
        result = record_smoke_execution_error()
    else:
        result = finalize()
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.stage == "audit" and not result["passed"]:
        raise SystemExit(4)


if __name__ == "__main__":
    main()
