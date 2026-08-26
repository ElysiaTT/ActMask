"""Terminal authorization audit for the conditional RM-ACV method stage.

The M0--M7 implementation is deliberately unreachable unless an existing,
frozen Phase A9 artifact contains exactly ``RM_ACV_BENCHMARK_READY``.  This
script does not reinterpret similarly named historical milestones as RM-ACV.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "actmask" / "rm_acv_method_authorization"
OUTPUTS = ROOT / "outputs" / "actmask"
REQUIRED = (
    "authorization_audit.json",
    "final_method_decision.json",
    "method_summary.json",
    "method_report.md",
    "method_per_seed.json",
    "ablation_summary.json",
    "localization_report.json",
    "candidate_reranking_report.json",
    "selective_rejection_report.json",
    "calibration_report.json",
    "raw_method_prediction_manifest.json",
    "checkpoint_manifest.json",
    "method_reproducibility_manifest.json",
    "paper_reframe/abstract_zh.md",
    "paper_reframe/abstract_en.md",
    "paper_reframe/method.md",
    "paper_reframe/experiments.md",
    "paper_reframe/results_tables.md",
    "paper_reframe/figure_plan.md",
    "paper_reframe/claim_boundary.md",
)


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _iter_decisions(value: Any, origin: str = "") -> Iterator[dict[str, str]]:
    if isinstance(value, dict):
        for key, item in value.items():
            if key.lower() in {"decision", "final_decision", "a9_decision"} and isinstance(item, str):
                yield {"key": key, "value": item, "origin": origin}
            yield from _iter_decisions(item, origin)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_decisions(item, origin)


def _read_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None


def _audit() -> dict[str, Any]:
    decision_files = sorted(
        path for path in OUTPUTS.rglob("*.json")
        if OUT not in path.parents and any(token in path.name.lower() for token in ("decision", "summary", "audit"))
    )
    known_decisions: list[dict[str, str]] = []
    exact_ready_paths: list[str] = []
    rm_acv_named_artifacts: list[str] = []
    for path in decision_files:
        payload = _read_json(path)
        if payload is None:
            continue
        relative = str(path.relative_to(ROOT))
        text = json.dumps(payload, ensure_ascii=False)
        if "RM_ACV" in text or "PhaseAwareTemporalEnergyVerifier" in text:
            rm_acv_named_artifacts.append(relative)
        for item in _iter_decisions(payload, relative):
            if item["value"] == "RM_ACV_BENCHMARK_READY":
                exact_ready_paths.append(relative)
            if "ACV" in item["value"] or "method" in item["value"].lower() or "stage_b" in item["value"].lower():
                known_decisions.append(item)
    # This historic artifact is deliberately recorded only as a non-equivalent
    # precedent, so its GO result cannot authorize a new RM-ACV method.
    historic = OUTPUTS / "milestone3p_cpu_shortcut_audit" / "cpu_summary.json"
    historic_payload = _read_json(historic)
    historic_decision = None
    if isinstance(historic_payload, dict):
        historic_decision = historic_payload.get("decision", {}).get("decision")
    forbidden_method_outputs = [
        str(path.relative_to(ROOT))
        for path in OUTPUTS.rglob("method_preregistered_config.json")
        if OUT not in path.parents
    ]
    return {
        "schema": "rm-acv-method-authorization-audit-v1",
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "required_a9_value": "RM_ACV_BENCHMARK_READY",
        "a9_artifact_found": bool(rm_acv_named_artifacts),
        "rm_acv_named_artifacts": rm_acv_named_artifacts,
        "exact_ready_paths": exact_ready_paths,
        "method_authorized": len(exact_ready_paths) > 0,
        "other_observed_decisions": known_decisions,
        "historic_non_equivalent_milestone3p": {
            "path": str(historic.relative_to(ROOT)),
            "exists": historic.is_file(),
            "decision": historic_decision,
            "why_not_authorization": "It belongs to the older Milestone 3P shortcut-audit protocol and is not the exact RM_ACV A9 decision required by the conditional method authorization.",
        },
        "preexisting_method_configs": forbidden_method_outputs,
        "conclusion": "RM_ACV_A9_DECISION_MISSING" if not exact_ready_paths else "RM_ACV_BENCHMARK_READY",
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _not_run(name: str, reason: str) -> dict[str, Any]:
    return {"schema": "rm-acv-terminal-not-run-v1", "artifact": name, "status": "not_run", "reason": reason, "method_trained": False}


def run() -> dict[str, Any]:
    audit = _audit()
    if audit["method_authorized"]:
        raise RuntimeError("This terminal-only script refuses to run when RM_ACV is authorized; proceed through a separately frozen M0 configuration.")
    OUT.mkdir(parents=True, exist_ok=True)
    reason = "No current RM-ACV Phase A9 artifact returns exactly RM_ACV_BENCHMARK_READY; M0--M7 are prohibited."
    _write_json(OUT / "authorization_audit.json", audit)
    final = {
        "schema": "rm-acv-final-method-decision-v1",
        "terminal": True,
        "authorization_status": "NOT_AUTHORIZED",
        "a9_decision": audit["conclusion"],
        "required_a9_decision": audit["required_a9_value"],
        "method_decision": None,
        "method_trained": False,
        "reason": reason,
        "scope": "This is an authorization terminal record, not a claim about method quality or robot performance.",
    }
    _write_json(OUT / "final_method_decision.json", final)
    _write_json(OUT / "method_summary.json", {**_not_run("method_summary", reason), "authorization_audit": "authorization_audit.json"})
    for filename in (
        "method_per_seed.json", "ablation_summary.json", "localization_report.json",
        "candidate_reranking_report.json", "selective_rejection_report.json",
        "calibration_report.json", "raw_method_prediction_manifest.json", "checkpoint_manifest.json",
    ):
        _write_json(OUT / filename, _not_run(filename, reason))
    (OUT / "method_report.md").write_text(
        "# RM-ACV 方法授权终止报告\n\n"
        "结论：**未授权训练**。条件授权要求已有 Phase A9 决策精确等于 `RM_ACV_BENCHMARK_READY`。"
        "当前仓库没有该 RM-ACV A9 工件或该精确决策；旧 Milestone 3P 的 Stage-B GO 不是等价协议。"
        "因此没有创建 M0 配置、没有训练模型、没有 checkpoint、没有候选能量或校准预测。\n",
        encoding="utf-8",
    )
    paper = OUT / "paper_reframe"
    paper.mkdir(exist_ok=True)
    texts = {
        "abstract_zh.md": "# 摘要\n\n尚无经授权的 RM-ACV 方法实验结果，不能报告方法性能或机器人成功结论。\n",
        "abstract_en.md": "# Abstract\n\nNo RM-ACV method experiment has been authorized; no method-performance or robot-success claim is reported.\n",
        "method.md": "# Method\n\nNot run: the required RM-ACV A9 authorization artifact is absent.\n",
        "experiments.md": "# Experiments\n\nNo M0--M7 experiment was run. See `../authorization_audit.json`.\n",
        "results_tables.md": "# Results\n\nNo verified method results are available.\n",
        "figure_plan.md": "# Figure plan\n\nNo result figure is planned until the benchmark authorization condition is satisfied.\n",
        "claim_boundary.md": "# Claim boundary\n\nThis package makes only an authorization-status claim. It makes no online safety, physical execution, or robot-success claim.\n",
    }
    for filename, content in texts.items():
        (paper / filename).write_text(content, encoding="utf-8")
    tracked = [OUT / relative for relative in REQUIRED if relative != "method_reproducibility_manifest.json"]
    manifest = {
        "schema": "rm-acv-terminal-reproducibility-manifest-v1",
        "terminal_status": "NOT_AUTHORIZED",
        "method_trained": False,
        "generator": {"path": str(Path(__file__).relative_to(ROOT)), "sha256": _sha(Path(__file__))},
        "files": [{"path": str(path.relative_to(OUT)), "sha256": _sha(path), "bytes": path.stat().st_size} for path in tracked],
    }
    _write_json(OUT / "method_reproducibility_manifest.json", manifest)
    return final


def verify() -> dict[str, Any]:
    missing = [name for name in REQUIRED if not (OUT / name).is_file()]
    final = _read_json(OUT / "final_method_decision.json")
    audit = _read_json(OUT / "authorization_audit.json")
    manifest = _read_json(OUT / "method_reproducibility_manifest.json")
    hash_failures: list[str] = []
    if isinstance(manifest, dict):
        for item in manifest.get("files", []):
            path = OUT / item["path"]
            if not path.is_file() or _sha(path) != item.get("sha256"):
                hash_failures.append(item["path"])
        generator = manifest.get("generator", {})
        if generator.get("sha256") != _sha(Path(__file__)):
            hash_failures.append("generator")
    else:
        hash_failures.append("method_reproducibility_manifest.json")
    current = _audit()
    passed = not missing and not hash_failures and isinstance(final, dict) and final.get("authorization_status") == "NOT_AUTHORIZED" and final.get("method_trained") is False and isinstance(audit, dict) and audit.get("method_authorized") is False and current.get("method_authorized") is False
    return {"passed": passed, "missing": missing, "hash_failures": hash_failures, "authorization_status": final.get("authorization_status") if isinstance(final, dict) else None, "a9_conclusion": audit.get("conclusion") if isinstance(audit, dict) else None, "current_a9_conclusion": current.get("conclusion")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    print(json.dumps(verify() if args.verify else run(), ensure_ascii=False, sort_keys=True))
