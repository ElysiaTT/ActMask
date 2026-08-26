"""Terminal packaging and verification for the stopped v2 experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from actmask.experiments.history_action_binding_v2.common import (
    OUTPUT_ROOT, PROJECT_ROOT, PROTOCOL, SOURCE_FILES, TASKS, V1_ROOT,
    canonical_json, protocol_sha256, sha256_bytes, sha256_file, source_manifest, write_json,
)
from actmask.experiments.history_action_binding_v2.generate import generation_refusal
from actmask.experiments.history_action_binding_v2.postmortem import build_postmortem
from actmask.experiments.history_action_binding_v2.symbolic import run_symbolic_preflight
from actmask.experiments.history_action_binding_v2.audit import (
    binding_breaking_shuffle, candidate_slot_permutation, coordinate_rotation,
    pair_preserving_shuffle, twin_history_swap,
)


def _record(path: Path) -> dict:
    if path.is_symlink():
        target = os.readlink(path)
        return {"kind": "symlink", "bytes": len(target.encode()), "sha256": sha256_bytes(target.encode()), "target": target}
    return {"kind": "file", "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def tree_manifest(root: Path, *, exclude: set[Path] | None = None) -> dict:
    excluded = set() if exclude is None else {path.resolve() for path in exclude}
    files = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file() or item.is_symlink()):
        if path.resolve() in excluded:
            continue
        files[str(path.relative_to(root))] = _record(path)
    return {"files": files, "combined_sha256": sha256_bytes(canonical_json(files).encode("utf-8"))}


def verify_v1_tree() -> dict:
    expected = json.loads((V1_ROOT / "generated_manifest.json").read_text())
    missing, mismatched = [], []
    for relative, record in expected["files"].items():
        path = V1_ROOT / relative
        if not path.exists() and not path.is_symlink():
            missing.append(relative); continue
        if _record(path) != record:
            mismatched.append(relative)
    extra = sorted(
        str(path.relative_to(V1_ROOT)) for path in V1_ROOT.rglob("*")
        if (path.is_file() or path.is_symlink()) and str(path.relative_to(V1_ROOT)) not in expected["files"]
        and path.name != "generated_manifest.json"
    )
    return {"passed": not missing and not mismatched and not extra, "expected_files": len(expected["files"]), "missing": missing, "mismatched": mismatched, "extra": extra}


def verify_other_protected() -> dict:
    expected = json.loads((V1_ROOT / "protected_boundary_after.json").read_text())
    missing, mismatched = [], []
    for relative, record in expected["files"].items():
        path = PROJECT_ROOT / relative
        if not path.exists() and not path.is_symlink():
            missing.append(relative); continue
        if _record(path) != record:
            mismatched.append(relative)
    return {
        "passed": not missing and not mismatched,
        "reference": "outputs/actmask/history_action_binding_mvp/protected_boundary_after.json",
        "reference_combined_sha256": expected["combined_sha256"], "expected_files": len(expected["files"]),
        "missing": missing, "mismatched": mismatched,
    }


def write_config_and_record(preflight: dict) -> dict:
    core_files = (
        "actmask/data/history_action_binding_v2_tasks.py",
        "actmask/experiments/history_action_binding_v2/common.py",
        "actmask/experiments/history_action_binding_v2/audit.py",
        "actmask/experiments/history_action_binding_v2/symbolic.py",
        "docs/history_action_binding_v2_preregister.md",
    )
    core = {relative: {"bytes": (PROJECT_ROOT / relative).stat().st_size, "sha256": sha256_file(PROJECT_ROOT / relative)} for relative in core_files}
    config = {
        "status": "official_symbolic_preflight_executed; GPU stages blocked",
        "protocol": PROTOCOL, "protocol_sha256": protocol_sha256(),
        "thresholds_frozen_before_formal_pilot": True,
        "formal_pilot_results_seen": False, "formal_simulator_data_generated": False,
        "core_preflight_sources": core,
        "preflight_passed": preflight["passed"],
    }
    path = OUTPUT_ROOT / "preregistered_config.json"; write_json(path, config)
    digest = sha256_file(path)
    (OUTPUT_ROOT / "preregistered_config.sha256").write_text(digest + "  preregistered_config.json\n")
    record = {
        "preregistered_config_sha256": digest, "protocol_sha256": protocol_sha256(),
        "symbolic_summary_sha256": sha256_file(OUTPUT_ROOT / "symbolic_preflight/summary.json"),
        "core_preflight_sources_combined_sha256": sha256_bytes(canonical_json(core).encode("utf-8")),
        "gpu_stage_started": False,
    }
    write_json(OUTPUT_ROOT / "preflight_record.json", record)
    return record


def write_schema() -> dict:
    tasks = {}
    for task in TASKS:
        path = OUTPUT_ROOT / "symbolic_preflight" / task / "symbolic_samples.npz"
        with np.load(path, allow_pickle=False) as archive:
            tasks[task] = {key: {"dtype": str(archive[key].dtype), "shape": list(archive[key].shape)} for key in archive.files}
    planned_trace_fields = {
        "probe_observation_before": "public state before each executed probe",
        "probe_observation_after": "public state after each pulse",
        "history_trace_public": "per-step public state including matching tail",
        "history_trace_action": "per-step 2D probe or oracle-construction command",
        "candidate_trace_response_pos": "per-step PhysX response position",
        "candidate_trace_response_vel": "per-step PhysX response velocity",
        "candidate_trace_actuator_pos": "per-step proxy state",
        "candidate_trace_target_pos": "per-step target state",
        "candidate_trace_distance": "per-step target distance",
        "candidate_trace_contact": "per-step contact boolean",
        "initial_state_hash/decision_state_hash/final_state_hash": "full simulator/task state SHA-256",
    }
    schema = {
        "schema_version": "history_action_binding_v2_v1",
        "symbolic_datasets": tasks,
        "simulator_trace_schema_status": "not_instantiated_because_symbolic_preflight_failed",
        "planned_simulator_trace_fields": planned_trace_fields,
    }
    write_json(OUTPUT_ROOT / "dataset_schema.json", schema)
    return schema


def write_symbolic_interventions() -> dict:
    records = {}
    for task in TASKS:
        source = OUTPUT_ROOT / "symbolic_preflight" / task / "symbolic_samples.npz"
        with np.load(source, allow_pickle=False) as archive:
            dataset = {key: archive[key] for key in archive.files}
        actions = dataset["probe_actions"]; results = dataset["probe_delta"]
        pair_actions, pair_results = pair_preserving_shuffle(actions, results)
        broken_actions, broken_results = binding_breaking_shuffle(actions, results)
        swapped_actions, swapped_results = twin_history_swap(actions, results)
        permuted = candidate_slot_permutation(dataset); rotated = coordinate_rotation(dataset)
        target = source.parent / "intervention_samples.npz"
        np.savez_compressed(
            target,
            original_probe_actions=actions, original_probe_results=results,
            pair_preserving_probe_actions=pair_actions, pair_preserving_probe_results=pair_results,
            binding_break_probe_actions=broken_actions, binding_break_probe_results=broken_results,
            twin_swap_probe_actions=swapped_actions, twin_swap_probe_results=swapped_results,
            fixed_current_observation=dataset["decision_public_state"],
            fixed_candidate_actions=dataset["candidate_actions"],
            original_success=dataset["success"], original_utility=dataset["utility"],
            candidate_template_id=dataset["candidate_template_id"],
            slot_permuted_candidates=permuted["candidate_actions"],
            slot_permuted_template_id=permuted["candidate_template_id"],
            slot_permuted_success=permuted["success"], slot_permuted_utility=permuted["utility"],
            rotated_probe_actions=rotated["probe_actions"], rotated_candidates=rotated["candidate_actions"],
            rotated_current=rotated["decision_public_state"], rotated_latent_phi=rotated["latent_phi"],
            simulator_expected_preference=np.sign(dataset["utility"][:, 1] - dataset["utility"][:, 0]).astype(np.int8),
            expected_preference_after_twin_swap=-np.sign(dataset["utility"][:, 1] - dataset["utility"][:, 0]).astype(np.int8),
        )
        records[task] = {"path": str(target.relative_to(OUTPUT_ROOT)), "bytes": target.stat().st_size, "sha256": sha256_file(target)}
    write_json(OUTPUT_ROOT / "symbolic_intervention_manifest.json", records)
    return records


def write_comparison(preflight: dict) -> dict:
    v1 = json.loads((V1_ROOT / "summary.json").read_text())
    rows = []
    for task in TASKS:
        item = preflight["tasks"][task]
        rows.append({
            "version": "v2 symbolic", "task": task, "gate": "PASS" if item["passed"] else "SYMBOLIC_PREFLIGHT_FAIL",
            "marginal_max": max(item["marginal_scores"].values()), "arrow": item["arrow_score"],
            "sysid": item["fourier_score"], "binding_drop": item["binding_drop"],
            "formal_gpu_data": False,
        })
    comparison = {
        "v1": {"gate": v1["gate"], "training_authorized": v1["learned_training_authorized"], "candidate_executions": v1["candidate_executions"]},
        "v2": {"gate": "SYMBOLIC_PREFLIGHT_FAIL", "training_authorized": False, "candidate_executions": 0},
        "rows": rows,
        "interpretation": "v2 removes the demonstrated marginal/Arrow cues symbolically, but Task A misses the preregistered SysID threshold and slot balance is not exact; the protocol is stopped rather than tuned after inspection.",
    }
    write_json(OUTPUT_ROOT / "v1_v2_comparison.json", comparison)
    return comparison


def _report(summary: dict) -> str:
    lines = [
        "# History–Action Binding Audit v2 报告", "",
        "终止状态：**SYMBOLIC_PREFLIGHT_FAIL**。按照预注册，没有运行 GPU smoke/pilot，没有生成 candidate rollout，也没有训练 learned verifier。", "",
        "## 最终五个问题", "",
        "- v2 数据是否可信：**symbolic evidence 完整可复算；真实 simulator dataset 未生成，因此不能声称真实数据已通过。**",
        "- marginal shortcut 是否排除：**在 symbolic preflight 中是。** current/action/result/index corrected TPA 均为0.5；但这不是正式 GPU 结论。",
        "- action–result binding 是否必要：**尚未建立。** Task A Fourier SysID 只有0.8789，低于冻结0.90，且 slot balance 未通过。",
        "- novel candidate SysID 是否成立：**未同时成立。** Task A未达阈值；Task B为0.9082。",
        "- 是否授权下一个 learned verifier Goal：**不授权。**", "",
        "## Symbolic preflight", "",
        "| Task | Marginal max | Arrow | Fourier SysID | Binding-break | Drop | Slot balance | Result multiset |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for task in TASKS:
        item = summary["symbolic_preflight"]["tasks"][task]
        lines.append(
            f"| {task} | {max(item['marginal_scores'].values()):.4f} | {item['arrow_score']:.4f} | {item['fourier_score']:.4f} | "
            f"{item['binding_break_score']:.4f} | {item['binding_drop']:.4f} | {item['checks']['candidate_slots_balanced']} | {item['checks']['result_multiset_exact']} |"
        )
    lines.extend([
        "", "slot table 的冻结期望是每个 cell=4；实际两任务均为 min=3/max=5。没有改 sampler 后重跑。Task A Fourier corrected TPA=0.87890625；没有降低0.90阈值。", "",
        "其余设计信号正确：两个任务 action/result/utility marginals 精确相同，Arrow=0.5，top-1 crossing=1.0，pair-preserving变化=0，binding-break drop分别0.3594/0.4193。", "",
        "## v1 到 v2", "",
        "| Version | Stage/Gate | Marginal | Arrow | SysID | GPU candidates | Training |",
        "|---|---|---:|---:|---:|---:|---:|",
        "| v1 | SHORTCUT_SATURATED | result/action shortcuts saturated | Task A=1.0 | explicit fits saturated | 2336 | no |",
        f"| v2 Task A | SYMBOLIC_PREFLIGHT_FAIL | {max(summary['symbolic_preflight']['tasks'][TASKS[0]]['marginal_scores'].values()):.4f} | {summary['symbolic_preflight']['tasks'][TASKS[0]]['arrow_score']:.4f} | {summary['symbolic_preflight']['tasks'][TASKS[0]]['fourier_score']:.4f} | 0 | no |",
        f"| v2 Task B | SYMBOLIC_PREFLIGHT_FAIL | {max(summary['symbolic_preflight']['tasks'][TASKS[1]]['marginal_scores'].values()):.4f} | {summary['symbolic_preflight']['tasks'][TASKS[1]]['arrow_score']:.4f} | {summary['symbolic_preflight']['tasks'][TASKS[1]]['fourier_score']:.4f} | 0 | no |", "",
        "v1 的 raw TPA 混入 branch总体容易度，NDCG又奖励两个branch共享的候选幅值排序。v2 corrected TPA先在branch内center，再逐candidate比较 twin；branch常数分数严格为0.5。量化复盘见 `v1_failure_postmortem.json`。", "",
        "## 为什么没有 trace", "",
        "目标明确规定 symbolic preflight 失败后立即停止且不得运行 GPU pilot。`generate.py` 是 fail-closed 可执行入口；调用会写 refusal evidence 并以非零状态退出。缺少真实 trace 是遵守终止条件的结果，而不是把 symbolic 结果冒充 simulator evidence。", "",
        "## 修复建议（只能新建 v2.1）", "",
        "1. 用预先证明的 Latin-square slot permutation 取代当前 affine seed mapping；在新版本运行前符号证明每个slot/template cell严格相等。",
        "2. 不降低SysID阈值；重新设计 Task A probe/candidate geometry或提高可识别性，并在新版本预注册后只运行一次 official preflight。",
        "3. 保留本 v2 全部结果作为失败证据，v2.1 使用新路径，不能覆盖。", "",
        "## 可复制命令", "", "```bash",
        *summary["commands"], "```", "",
    ])
    return "\n".join(lines)


def finalize() -> dict:
    preflight_path = OUTPUT_ROOT / "symbolic_preflight/summary.json"
    if not preflight_path.exists():
        build_postmortem(); run_symbolic_preflight()
    preflight = json.loads(preflight_path.read_text())
    if preflight["passed"]:
        raise RuntimeError("This terminal packager is only for the recorded failed v2 preflight")
    if not (OUTPUT_ROOT / "v1_failure_postmortem.json").exists():
        build_postmortem()
    refusal = generation_refusal("smoke", TASKS[0])
    config_record = write_config_and_record(preflight)
    schema = write_schema(); comparison = write_comparison(preflight)
    intervention_records = write_symbolic_interventions()
    v1_check = verify_v1_tree(); protected_check = verify_other_protected()
    if not v1_check["passed"] or not protected_check["passed"]:
        raise RuntimeError("protected evidence changed")
    write_json(OUTPUT_ROOT / "protected_boundary_verification.json", {"passed": True, "v1": v1_check, "other_protected": protected_check})
    sources = source_manifest(); write_json(OUTPUT_ROOT / "source_manifest.json", sources)
    commands = [
        f"PYTHONPATH={PROJECT_ROOT} /data/envs/actmask-audit/bin/python -m actmask.experiments.history_action_binding_v2.postmortem",
        f"PYTHONPATH={PROJECT_ROOT} /data/envs/actmask-audit/bin/python -m actmask.experiments.history_action_binding_v2.symbolic",
        f"PYTHONPATH={PROJECT_ROOT} /data/envs/actmask-audit/bin/python -m actmask.experiments.history_action_binding_v2.generate --phase smoke --task {TASKS[0]}  # expected exit 3",
        f"PYTHONPATH={PROJECT_ROOT} /data/envs/actmask-audit/bin/python -m actmask.experiments.history_action_binding_v2.run_v2 --stage finalize",
        f"PYTHONPATH={PROJECT_ROOT} /data/envs/actmask-audit/bin/python -m unittest tests.test_history_action_binding_v2 -v",
    ]
    summary = {
        "gate": "SYMBOLIC_PREFLIGHT_FAIL", "symbolic_evidence_trustworthy": True,
        "simulator_data_generated": False, "simulator_data_trustworthy": None,
        "marginal_shortcut_excluded_symbolically": True,
        "binding_necessity_established": False, "novel_candidate_sysid_established_both_tasks": False,
        "learned_training_authorized": False, "learned_models_trained": [],
        "candidate_executions": 0, "candidate_execution_budget": PROTOCOL["candidate_execution_budget"],
        "output_bytes": 0, "output_byte_budget": PROTOCOL["output_byte_budget"],
        "symbolic_preflight": preflight, "generator_refusal": refusal,
        "preregistered_config": config_record, "schema_status": schema["simulator_trace_schema_status"],
        "v1_v2_comparison": comparison, "protected_boundary_unchanged": True,
        "symbolic_intervention_samples": intervention_records,
        "source_combined_sha256": sources["combined_sha256"], "commands": commands,
    }
    summary_path = OUTPUT_ROOT / "summary.json"; report_path = OUTPUT_ROOT / "audit_report.md"
    write_json(OUTPUT_ROOT / "commands.json", {"commands": commands})
    for _ in range(10):
        write_json(summary_path, summary)
        report = _report(summary); report_path.write_text(report); (PROJECT_ROOT / "docs/history_action_binding_v2_report.md").write_text(report)
        actual = sum(path.lstat().st_size for path in OUTPUT_ROOT.rglob("*") if path.is_file() or path.is_symlink())
        if actual == summary["output_bytes"]:
            break
        summary["output_bytes"] = actual
    write_json(summary_path, summary)
    report = _report(summary); report_path.write_text(report); (PROJECT_ROOT / "docs/history_action_binding_v2_report.md").write_text(report)
    terminal = {
        "gate": summary["gate"], "stage": "symbolic_preflight", "gpu_stages_started": False,
        "failed_checks": {task: [key for key, passed in preflight["tasks"][task]["checks"].items() if not passed] for task in TASKS},
        "learned_training_authorized": False,
    }
    write_json(OUTPUT_ROOT / "terminal_stop.json", terminal)
    generated_path = OUTPUT_ROOT / "generated_manifest.json"
    generated = tree_manifest(OUTPUT_ROOT, exclude={generated_path}); write_json(generated_path, generated)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("postmortem", "symbolic", "smoke", "pilot", "audit", "finalize"), required=True)
    args = parser.parse_args()
    if args.stage == "postmortem":
        result = build_postmortem()
    elif args.stage == "symbolic":
        result = run_symbolic_preflight()
    elif args.stage in ("smoke", "pilot"):
        result = generation_refusal(args.stage, TASKS[0])
    elif args.stage == "audit":
        result = json.loads((OUTPUT_ROOT / "symbolic_preflight/summary.json").read_text())
    else:
        result = finalize()
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.stage in ("smoke", "pilot") or (args.stage == "symbolic" and not result["passed"]):
        raise SystemExit(3)


if __name__ == "__main__":
    main()
