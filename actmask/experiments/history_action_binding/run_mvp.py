"""Stage runner, freeze logic, reports, schemas, and evidence manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from actmask.data.history_action_binding_tasks import TASK_IDS
from actmask.experiments.history_action_binding.audit import BASELINES, audit_dataset
from actmask.experiments.history_action_binding.common import (
    OUTPUT_ROOT,
    PROJECT_ROOT,
    PROTECTED_GLOBS,
    PROTOCOL,
    canonical_json,
    protocol_sha256,
    sha256_bytes,
    sha256_file,
    source_manifest,
    write_json,
)
from actmask.experiments.history_action_binding.generate import generate_task


TASKS = tuple(TASK_IDS)
PHASE_SEEDS = {
    ("smoke", "task_a_discrete_mode"): int(PROTOCOL["seed"]),
    ("smoke", "task_b_continuous_response"): int(PROTOCOL["seed"]) + 100_000,
    ("pilot", "task_a_discrete_mode"): int(PROTOCOL["seed"]) + 1_000_000,
    ("pilot", "task_b_continuous_response"): int(PROTOCOL["seed"]) + 1_100_000,
}


def _file_record(path: Path) -> dict:
    if path.is_symlink():
        target = os.readlink(path)
        return {"kind": "symlink", "bytes": len(target.encode()), "sha256": sha256_bytes(target.encode()), "target": target}
    return {"kind": "file", "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def tree_manifest(roots: list[Path], *, relative_to: Path, exclude: set[Path] | None = None) -> dict:
    exclude = set() if exclude is None else {path.resolve() for path in exclude}
    files: dict[str, dict] = {}
    for root in roots:
        if not root.exists():
            continue
        paths = [root] if root.is_file() or root.is_symlink() else sorted(path for path in root.rglob("*") if path.is_file() or path.is_symlink())
        for path in paths:
            if path.resolve() in exclude:
                continue
            relative = str(path.relative_to(relative_to))
            files[relative] = _file_record(path)
    return {"files": files, "combined_sha256": sha256_bytes(canonical_json(files).encode("utf-8"))}


def protected_snapshot() -> dict:
    roots: list[Path] = []
    for pattern in PROTECTED_GLOBS:
        roots.extend(sorted(PROJECT_ROOT.glob(pattern)))
    snapshot = tree_manifest(roots, relative_to=PROJECT_ROOT)
    snapshot["roots"] = [str(path.relative_to(PROJECT_ROOT)) for path in roots]
    return snapshot


def _generation_report(phase: str, task: str) -> dict:
    path = OUTPUT_ROOT / phase / task / "generation_report.json"
    if not path.exists():
        raise RuntimeError(f"missing generation report: {path}")
    return json.loads(path.read_text())


def run_smoke() -> dict:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT_ROOT / "protected_boundary_before.json", protected_snapshot())
    reports = {}
    for task in TASKS:
        report = generate_task(
            OUTPUT_ROOT / "smoke" / task,
            phase="smoke", task=task,
            twins=int(PROTOCOL["smoke_twins_per_task"]), seed=PHASE_SEEDS[("smoke", task)],
        )
        reports[task] = report
        if report["status"] != "passed":
            write_json(OUTPUT_ROOT / "terminal_stop.json", {"gate": "DATA_INTEGRITY_FAIL", "stage": "smoke", "task": task, "report": report})
            raise RuntimeError(f"{task} smoke integrity failed")

    determinism_root = OUTPUT_ROOT / "determinism"
    deterministic_reports = []
    for repeat in ("run1", "run2"):
        deterministic_reports.append(generate_task(
            determinism_root / repeat,
            phase="smoke", task="task_a_discrete_mode", twins=1,
            seed=int(PROTOCOL["seed"]) + 2_000_000,
        ))
    arrays_equal = True
    mismatches = []
    with np.load(determinism_root / "run1/dataset.npz", allow_pickle=False) as left, np.load(determinism_root / "run2/dataset.npz", allow_pickle=False) as right:
        if left.files != right.files:
            arrays_equal = False
            mismatches.append("field_set")
        for key in sorted(set(left.files) & set(right.files)):
            if not np.array_equal(left[key], right[key]):
                arrays_equal = False
                mismatches.append(key)
    determinism = {
        "passed": arrays_equal and all(report["status"] == "passed" for report in deterministic_reports),
        "same_seed_same_batch_all_arrays_exact": arrays_equal,
        "mismatched_fields": mismatches,
        "candidate_executions": sum(report["candidate_executions"] for report in deterministic_reports),
    }
    write_json(determinism_root / "verification.json", determinism)
    if not determinism["passed"]:
        write_json(OUTPUT_ROOT / "terminal_stop.json", {"gate": "DATA_INTEGRITY_FAIL", "stage": "determinism", "detail": determinism})
        raise RuntimeError("deterministic regeneration failed")

    sources = source_manifest()
    frozen = {
        "freeze_stage": "after both 8-twin smoke integrity checks and exact deterministic regeneration",
        "formal_pilot_results_seen": False,
        "protocol": PROTOCOL,
        "protocol_sha256": protocol_sha256(),
        "source_manifest": sources,
        "smoke_integrity": {task: reports[task]["integrity"] for task in TASKS},
        "determinism": determinism,
    }
    config_path = OUTPUT_ROOT / "preregistered_config.json"
    write_json(config_path, frozen)
    config_sha = sha256_file(config_path)
    (OUTPUT_ROOT / "preregistered_config.sha256").write_text(config_sha + "  preregistered_config.json\n")
    write_json(OUTPUT_ROOT / "source_manifest.json", sources)
    write_json(OUTPUT_ROOT / "freeze_record.json", {"preregistered_config_sha256": config_sha, "protocol_sha256": protocol_sha256(), "source_combined_sha256": sources["combined_sha256"]})
    return frozen


def verify_freeze() -> dict:
    config_path = OUTPUT_ROOT / "preregistered_config.json"
    record_path = OUTPUT_ROOT / "freeze_record.json"
    if not config_path.exists() or not record_path.exists():
        raise RuntimeError("smoke/freeze stage has not completed")
    record = json.loads(record_path.read_text())
    checks = {
        "config_sha256": sha256_file(config_path) == record["preregistered_config_sha256"],
        "protocol_sha256": protocol_sha256() == record["protocol_sha256"],
        "source_combined_sha256": source_manifest()["combined_sha256"] == record["source_combined_sha256"],
    }
    checks["passed"] = all(checks.values())
    if not checks["passed"]:
        write_json(OUTPUT_ROOT / "terminal_stop.json", {"gate": "DATA_INTEGRITY_FAIL", "stage": "freeze_verification", "checks": checks})
        raise RuntimeError(f"frozen protocol/source changed: {checks}")
    return checks


def run_pilot() -> dict:
    freeze = verify_freeze()
    reports = {}
    # Both datasets are generated before any formal outcome audit is opened.
    for task in TASKS:
        report = generate_task(
            OUTPUT_ROOT / "pilot" / task,
            phase="pilot", task=task,
            twins=int(PROTOCOL["pilot_twins_per_task"]), seed=PHASE_SEEDS[("pilot", task)],
        )
        reports[task] = report
        if report["status"] != "passed":
            write_json(OUTPUT_ROOT / "terminal_stop.json", {"gate": "DATA_INTEGRITY_FAIL", "stage": "pilot", "task": task, "report": report})
            raise RuntimeError(f"{task} pilot integrity failed")
    write_json(OUTPUT_ROOT / "pilot_generation_summary.json", {
        "formal_outcomes_not_used_for_resampling_or_configuration": True,
        "freeze_verification": freeze,
        "tasks": reports,
    })
    return reports


def run_audits() -> dict:
    verify_freeze()
    summaries = {}
    for task in TASKS:
        dataset = OUTPUT_ROOT / "pilot" / task / "dataset.npz"
        summaries[task] = audit_dataset(dataset, OUTPUT_ROOT / "pilot" / task / "audit")
    write_json(OUTPUT_ROOT / "formal_audit_summaries.json", summaries)
    return summaries


FIELD_DESCRIPTIONS = {
    "probe_observation_before": "public o_t at each executed probe start",
    "probe_action": "public probe command u_t; matching-tail commands excluded",
    "probe_observation_after": "public o_(t+1) after the probe pulse",
    "probe_delta": "probe_observation_after - probe_observation_before",
    "probe_utility": "negative endpoint distance for the probe transition",
    "probe_success": "probe endpoint within frozen success radius",
    "decision_public_state": "fixed-resolution fair observable shared exactly within each twin",
    "decision_physical_state": "unquantized PhysX x/v and task state used for matching gates",
    "candidate_actions": "K candidates sampled without latent/task-outcome access",
    "candidate_actions_by_branch": "byte-identical candidate copy for twin equality audit",
    "candidate_trace_distance": "per-step response-to-goal absolute distance",
    "candidate_trace_contact": "per-step contact-force threshold boolean",
    "success": "any trace distance at or below success radius",
    "utility": "negative minimum trace distance",
    "latent_mode": "oracle-only discrete hidden mode",
    "latent_alpha": "oracle-only continuous response alpha",
    "latent_beta": "oracle-only continuous response beta",
    "initial_state_hash": "SHA-256 of complete simulator/task state vector",
    "decision_state_hash": "SHA-256 of complete matched simulator/task state vector",
    "final_state_hash": "SHA-256 of complete final simulator/task state vector",
}


def write_schema() -> dict:
    schema = {"schema_version": "history_action_binding_mvp_v1", "tasks": {}}
    for task in TASKS:
        path = OUTPUT_ROOT / "pilot" / task / "dataset.npz"
        with np.load(path, allow_pickle=False) as archive:
            schema["tasks"][task] = {
                key: {"dtype": str(archive[key].dtype), "shape": list(archive[key].shape), "description": FIELD_DESCRIPTIONS.get(key, "saved execution evidence")}
                for key in archive.files
            }
    write_json(OUTPUT_ROOT / "dataset_schema.json", schema)
    return schema


def _global_gate(audits: dict) -> str:
    gates = [audits[task]["gate"] for task in TASKS]
    for gate in ("DATA_INTEGRITY_FAIL", "SHORTCUT_SATURATED", "NO_IDENTIFIABLE_SIGNAL"):
        if gate in gates:
            return gate
    if all(gate == "BINDING_AUDIT_PASS" for gate in gates):
        return "BINDING_AUDIT_PASS"
    return "BINDING_AUDIT_FAIL"


def _actual_output_bytes() -> int:
    total = 0
    for path in OUTPUT_ROOT.rglob("*"):
        if path.is_file() or path.is_symlink():
            total += path.lstat().st_size
    return total


def _report_markdown(summary: dict) -> str:
    lines = [
        "# History–Action Binding Audit MVP 报告",
        "",
        f"终止判定：**{summary['gate']}**；learned verifier training：**{'授权' if summary['learned_training_authorized'] else '不授权'}**。",
        "",
        "## 四个最终问题",
        "",
        f"- 数据完整可信：**{'是' if summary['data_complete_and_trustworthy'] else '否'}**。两个 task 的 trace label/utility 100% 重算、twin candidates 逐字节相同、matched current observable 逐字节相同，且同 seed/same batch 复跑全部数组一致。这里的“可信”指本地执行证据和内部可重算性，不扩大为跨机器 bit-exact。",
        f"- action–result binding 必要：**{summary['binding_necessity_conclusion']}**",
        f"- novel-candidate generalization：**{summary['novel_candidate_conclusion']}**",
        f"- 下一阶段 learned model training：**{'授权' if summary['learned_training_authorized'] else '不授权'}**。",
        "",
        "## 正式 pilot",
        "",
        "两个 64-twin pilot 在统一打开 outcome 之前已全部生成；没有 outcome-based resampling、过滤 00/01/10/11、改阈值或改候选分布。",
        "",
        "| Task | Gate | 00/01/10/11 | flip ratio | best marginal | best explicit fit | fit TPA | binding-break drop |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for task in TASKS:
        item = summary["tasks"][task]
        counts = item["outcome_counts"]
        fit = item["best_explicit_fit"]
        lines.append(
            f"| {task} | {item['gate']} | {counts['00']}/{counts['01']}/{counts['10']}/{counts['11']} | "
            f"{item['flip_ratio']:.3f} | {item['best_marginal_baseline']} | {fit} | "
            f"{item['metrics']['original'][fit]['twin_preference_accuracy']:.3f} | {item['binding_drop']:.3f} |"
        )
    lines.extend(["", "## 全部无训练 baseline（正式 original binding）", ""])
    for task in TASKS:
        lines.extend([
            f"### {task}", "",
            "| Baseline | Balanced acc. | Twin pref. acc. | NDCG | Regret | Utility MAE |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for baseline in BASELINES:
            metric = summary["tasks"][task]["metrics"]["original"][baseline]
            lines.append(
                f"| {baseline} | {metric['balanced_accuracy']:.3f} | {metric['twin_preference_accuracy']:.3f} | "
                f"{metric['ndcg']:.3f} | {metric['normalized_regret']:.3f} | {metric['utility_mae']:.5f} |"
            )
        lines.append("")
    lines.extend([
        "## Gate 解释",
        "",
        "`SHORTCUT_SATURATED` 是预注册的终止状态：任一 simple no-training shortcut 的 balanced accuracy、Twin Preference Accuracy 或等价 NDCG ≥0.90。达到它以后没有训练 learned verifier。explicit fit 的 binding-drop 即使为正，也不能证明 binding 是唯一必要机制，因为更简单的边缘/方向/nearest 规则已经做满同一任务。",
        "",
        "所有六类干预的逐样本变换保存在各 task 的 `audit/intervention_samples.npz`；original、pair-preserving、binding-breaking、twin-swap、slot permutation、coordinate reflection 的每个 baseline 逐样本预测保存在 `baseline_predictions.csv`。",
        "",
        "## 数据与执行限制",
        "",
        "这是小型一维 state-only PhysX MVP，而不是机械臂/RGB benchmark。matching tail 可访问 latent，但仅用于数据构造，完整记录且不进入 verifier probe pairs。公共 current observation 使用冻结的 1e-3 sensor resolution 达到 byte identity，同时原始 PhysX state 仍接受更严格的物理误差 gate。oracle rollout 是上限，不是可部署输入。",
        "",
        "## 可复制命令",
        "",
        "```bash",
        *summary["commands"],
        "```",
        "",
    ])
    return "\n".join(lines)


def finalize() -> dict:
    freeze_checks = verify_freeze()
    audits = json.loads((OUTPUT_ROOT / "formal_audit_summaries.json").read_text())
    schema = write_schema()
    after = protected_snapshot()
    before = json.loads((OUTPUT_ROOT / "protected_boundary_before.json").read_text())
    protected_unchanged = before["files"] == after["files"] and before["combined_sha256"] == after["combined_sha256"]
    write_json(OUTPUT_ROOT / "protected_boundary_after.json", after)
    write_json(OUTPUT_ROOT / "protected_boundary_verification.json", {
        "passed": protected_unchanged,
        "before_combined_sha256": before["combined_sha256"],
        "after_combined_sha256": after["combined_sha256"],
        "protected_roots": before["roots"],
    })
    if not protected_unchanged:
        raise RuntimeError("protected TimeArrow/audit/paper paths changed during MVP run")
    reports = {phase: {task: _generation_report(phase, task) for task in TASKS} for phase in ("smoke", "pilot")}
    determinism = json.loads((OUTPUT_ROOT / "determinism/verification.json").read_text())
    candidate_executions = sum(reports[phase][task]["candidate_executions"] for phase in reports for task in TASKS) + determinism["candidate_executions"]
    gate = _global_gate(audits)
    integrity_pass = all(reports[phase][task]["integrity"]["passed"] for phase in reports for task in TASKS)
    learned_authorized = gate == "BINDING_AUDIT_PASS"
    fit_novel = {
        task: audits[task]["metrics"]["original"][audits[task]["best_explicit_fit"]]["twin_preference_accuracy"]
        for task in TASKS
    }
    binding_unique = all(
        audits[task]["binding_drop"] >= float(PROTOCOL["binding_drop_threshold"])
        and not audits[task]["shortcut_saturated_baselines"] for task in TASKS
    )
    runtime_home = OUTPUT_ROOT / "runtime_home"
    commands = [
        f"HOME={runtime_home} PYTHONPATH={PROJECT_ROOT} {Path(os.sys.executable)} -m actmask.experiments.history_action_binding.run_mvp --stage smoke",
        f"HOME={runtime_home} PYTHONPATH={PROJECT_ROOT} {Path(os.sys.executable)} -m actmask.experiments.history_action_binding.run_mvp --stage pilot",
        f"HOME={runtime_home} PYTHONPATH={PROJECT_ROOT} {Path(os.sys.executable)} -m actmask.experiments.history_action_binding.run_mvp --stage audit",
        f"HOME={runtime_home} PYTHONPATH={PROJECT_ROOT} {Path(os.sys.executable)} -m actmask.experiments.history_action_binding.run_mvp --stage finalize",
        f"PYTHONPATH={PROJECT_ROOT} {Path(os.sys.executable)} -m unittest tests.test_history_action_binding_mvp -v",
    ]
    summary = {
        "gate": gate,
        "data_complete_and_trustworthy": integrity_pass and determinism["passed"] and protected_unchanged,
        "binding_necessity_conclusion": "not uniquely necessary; simple shortcuts saturate at least one formal task" if not binding_unique else "supported on both tasks under the frozen binding interventions",
        "novel_candidate_generalization": fit_novel,
        "novel_candidate_conclusion": "explicit fits interpolate novel magnitudes, but this does not establish learned headroom" if all(value >= float(PROTOCOL["novel_candidate_min_twin_accuracy"]) for value in fit_novel.values()) else "not established on both tasks",
        "learned_training_authorized": learned_authorized,
        "learned_models_trained": [],
        "tasks": audits,
        "candidate_executions": candidate_executions,
        "candidate_execution_budget": int(PROTOCOL["candidate_execution_budget"]),
        "output_bytes": 0,
        "output_byte_budget": int(PROTOCOL["output_byte_budget"]),
        "independent_statistical_unit": "twin/world",
        "protocol_sha256": protocol_sha256(),
        "preregistered_config_sha256": sha256_file(OUTPUT_ROOT / "preregistered_config.json"),
        "source_combined_sha256": source_manifest()["combined_sha256"],
        "freeze_verification": freeze_checks,
        "deterministic_regeneration": determinism,
        "protected_boundary_unchanged": protected_unchanged,
        "schema_tasks": list(schema["tasks"]),
        "commands": commands,
        "no_outcome_filtering_or_resampling": True,
        "formal_datasets_generated_before_outcome_audit": True,
    }
    summary_path = OUTPUT_ROOT / "summary.json"
    for _ in range(10):
        write_json(summary_path, summary)
        actual = _actual_output_bytes()
        if actual == summary["output_bytes"]:
            break
        summary["output_bytes"] = actual
    write_json(summary_path, summary)
    report = _report_markdown(summary)
    (OUTPUT_ROOT / "audit_report.md").write_text(report)
    (PROJECT_ROOT / "docs/history_action_binding_mvp_report.md").write_text(report)
    write_json(OUTPUT_ROOT / "commands.json", {"commands": commands})
    # Re-stabilize the byte total after report/commands are present.
    for _ in range(10):
        actual = _actual_output_bytes()
        if actual == summary["output_bytes"]:
            break
        summary["output_bytes"] = actual
        write_json(summary_path, summary)
        report = _report_markdown(summary)
        (OUTPUT_ROOT / "audit_report.md").write_text(report)
        (PROJECT_ROOT / "docs/history_action_binding_mvp_report.md").write_text(report)
    if summary["candidate_executions"] > int(PROTOCOL["candidate_execution_budget"]) or summary["output_bytes"] > int(PROTOCOL["output_byte_budget"]):
        raise RuntimeError("execution or output budget exceeded")
    generated_path = OUTPUT_ROOT / "generated_manifest.json"
    manifest = tree_manifest([OUTPUT_ROOT], relative_to=OUTPUT_ROOT, exclude={generated_path})
    write_json(generated_path, manifest)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("smoke", "pilot", "audit", "finalize", "all"), required=True)
    args = parser.parse_args()
    result = None
    stages = ("smoke", "pilot", "audit", "finalize") if args.stage == "all" else (args.stage,)
    for stage in stages:
        if stage == "smoke":
            result = run_smoke()
        elif stage == "pilot":
            result = run_pilot()
        elif stage == "audit":
            result = run_audits()
        else:
            result = finalize()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
