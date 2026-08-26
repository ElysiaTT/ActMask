"""V13 terminal package for VERIFIER-SHORTCUT-AUDIT-V1.

This module is intentionally report-only.  It consumes the frozen V10
decision and completed analysis artifacts, writes the always-required V13
documents, and builds a recursive content-hash inventory.  It never launches
mitigation, changes V8/V9/V10 artifacts, or appends to ``run_log.jsonl``.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from .common import OUT, PYTHON, ROOT, SEEDS, STUDY_ID, read_json, sha256_file, write_json


ALWAYS_OUTPUTS = (
    "final_decision.json",
    "final_report.md",
    "executive_summary_zh.md",
    "executive_summary_en.md",
    "historical_branch_inventory.json",
    "family_role_manifest.json",
    "shortcut_control_summary.json",
    "verifier_summary.json",
    "verifier_per_environment.json",
    "shortcut_reliance_summary.json",
    "generator_transfer_matrix.csv",
    "matching_gap_report.json",
    "nuisance_probe_report.json",
    "calibration_shift_report.json",
    "failure_taxonomy.md",
    "claim_boundary.md",
    "next_steps.md",
    "reproducibility_manifest.json",
)

MITIGATION_OUTPUTS = (
    "final_mitigation_decision.json",
    "mitigation_summary.json",
    "mitigation_per_seed.json",
    "mitigation_ablation_summary.json",
    "mitigation_report.md",
    "mitigation_raw_prediction_manifest.json",
    "mitigation_checkpoint_manifest.json",
)

PREREQUISITES = (
    "phase_v10_decision.json",
    "preregistered_config.json",
    "imported_artifact_manifest.json",
    "imported_hash_verification.json",
    "unified_audit_hash_manifest.json",
    "audit_environment_manifest.json",
    "shortcut_control_config.json",
    "shortcut_control_summary.json",
    "verifier_config.json",
    "verifier_summary.json",
    "verifier_per_environment.json",
    "shortcut_reliance_summary.json",
    "generator_transfer_matrix.csv",
    "generator_transfer_matrix.json",
    "matching_gap_report.json",
    "nuisance_probe_report.json",
    "calibration_shift_report.json",
    "relational_gain_report.json",
    "representation_probe_summary.json",
    "independent_metric_recomputation.json",
    "failure_taxonomy.md",
)

INPUT_HASH_FILES = (
    "preregistered_config.json",
    "imported_artifact_manifest.json",
    "imported_hash_verification.json",
    "unified_audit_hash_manifest.json",
    "audit_environment_manifest.json",
    "shortcut_control_config.json",
    "verifier_config.json",
    "shortcut_reliance_summary.json",
    "generator_transfer_matrix.json",
    "matching_gap_report.json",
    "nuisance_probe_report.json",
    "calibration_shift_report.json",
    "relational_gain_report.json",
    "representation_probe_summary.json",
    "independent_metric_recomputation.json",
    "phase_v10_decision.json",
    "failure_taxonomy.md",
)

# The manifest cannot contain its own digest.  The run log is append-only and
# the orchestrating agent adds the V13 event after this module returns, so the
# log is represented by an explicitly labelled pre-terminal snapshot rather
# than treated as an immutable inventory member.
MANIFEST_EXCLUSIONS = (
    "reproducibility_manifest.json",
    "run_log.jsonl",
)


def _text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.strip() + "\n", encoding="utf-8")


def _require_prerequisites() -> None:
    missing = [
        name
        for name in PREREQUISITES
        if not (OUT / name).is_file() or (OUT / name).stat().st_size == 0
    ]
    if missing:
        raise RuntimeError(
            "V13 prerequisites missing or empty: " + ", ".join(missing)
        )


def _number(value: float, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def _pooled_control_rows(control: dict[str, Any]) -> list[dict[str, Any]]:
    return sorted(
        (
            row
            for row in control["summary_rows"]
            if row["environment"] == "pooled_reference"
        ),
        key=lambda row: row["balanced_accuracy_mean"],
        reverse=True,
    )


def _pooled_verifier_rows(verifier: dict[str, Any]) -> list[dict[str, Any]]:
    return sorted(
        (
            row
            for row in verifier["environment_summary"]
            if row["environment"] == "pooled_reference"
        ),
        key=lambda row: row["architecture"],
    )


def _easy_gate_rows(decision: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for architecture in decision["easy_IID_learnability_gate"][
        "per_architecture"
    ]:
        for family in architecture["families"]:
            rows.append(
                {
                    "architecture": architecture["architecture"],
                    "family": family["family"],
                    "seed_balanced_accuracy": [
                        float(row["balanced_accuracy"])
                        for row in family["seed_rows"]
                    ],
                    "passing_seeds": family["passing_seeds"],
                    "holds": family[
                        "holds_across_at_least_two_seeds"
                    ],
                }
            )
    return rows


def _gate_summary(decision: dict[str, Any]) -> dict[str, Any]:
    independent = decision["independent_recomputation_gate"]
    return {
        "historical_artifacts_complete": bool(
            decision["historical_artifact_gate"]["pass"]
        ),
        "independent_metric_recomputation": {
            "pass": bool(independent["pass"]),
            "standard_metric_rows": independent[
                "total_metric_rows_recomputed"
            ],
            "verifier_environment_rows": independent[
                "verifier_environment_predictions"
            ]["rows_checked"],
            "shortcut_control_rows": independent[
                "shortcut_control_predictions"
            ]["rows_checked"],
            "family_transfer_rows": independent[
                "family_transfer_predictions"
            ]["rows_checked"],
            "representation_probe_runs": independent[
                "representation_probe_predictions"
            ]["probe_runs_verified"],
            "nuisance_probe_files": independent[
                "nuisance_probe_predictions"
            ]["probe_files"],
        },
        "easy_IID_learnability": {
            "pass": bool(
                decision["easy_IID_learnability_gate"]["pass"]
            ),
            "qualifying_architectures": decision[
                "easy_IID_learnability_gate"
            ]["qualifying_architectures"],
            "qualifying_architecture_count": decision[
                "easy_IID_learnability_gate"
            ]["qualifying_architecture_count"],
            "required_architectures": decision[
                "easy_IID_learnability_gate"
            ]["required_architectures"],
            "threshold_balanced_accuracy": decision[
                "easy_IID_learnability_gate"
            ]["threshold"],
            "required_seed_count_per_family": 2,
        },
        "shortcut_indicators": {
            "pass": bool(decision["shortcut_indicator_gate"]["pass"]),
            "qualifying_architectures": decision[
                "shortcut_indicator_gate"
            ]["qualifying_architectures"],
            "qualifying_architecture_count": decision[
                "shortcut_indicator_gate"
            ]["qualifying_architecture_count"],
            "required_architectures": decision[
                "shortcut_indicator_gate"
            ]["required_architectures"],
            "best_shortcut_control": decision[
                "shortcut_indicator_gate"
            ]["best_shortcut_control"],
            "best_shortcut_control_pooled_balanced_accuracy": decision[
                "shortcut_indicator_gate"
            ]["best_shortcut_control_pooled_BA"],
            "interpretation": (
                "diagnostic only after the earlier easy-IID learnability "
                "prerequisite failed"
            ),
        },
        "task_span": {
            "pass": bool(decision["task_span_gate"]["pass"]),
            "qualifying_task_count": decision["task_span_gate"][
                "qualifying_task_count"
            ],
            "required_tasks": decision["task_span_gate"][
                "required_tasks"
            ],
            "interpretation": (
                "diagnostic only after the earlier easy-IID learnability "
                "prerequisite failed"
            ),
        },
    }


def _final_decision(
    decision: dict[str, Any],
    prereg: dict[str, Any],
) -> dict[str, Any]:
    if decision["decision"] != "VSA_NO_LEARNABLE_IID_TASK":
        raise RuntimeError(
            "This terminal branch expects VSA_NO_LEARNABLE_IID_TASK, got "
            f"{decision['decision']}"
        )
    if decision["mitigation_authorized"] or not decision[
        "must_stop_before_mitigation"
    ]:
        raise RuntimeError("V10 unexpectedly authorizes mitigation")
    return {
        "schema": "vsa-final-decision-v1",
        "study_id": STUDY_ID,
        "decision": decision["decision"],
        "status": "TERMINAL_NON_SUPPORTED",
        "terminal": True,
        "terminal_phase": "V13",
        "source_phase": "V10",
        "source_phase_decision_path": "phase_v10_decision.json",
        "source_phase_decision_sha256": sha256_file(
            OUT / "phase_v10_decision.json"
        ),
        "scientific_question": prereg["scientific_question"],
        "systematic_shortcut_supported": False,
        "mitigation_authorized": False,
        "mitigation_executed": False,
        "must_stop_before_mitigation": True,
        "decision_precedence": list(decision["gate_order"]),
        "gate_outcomes": _gate_summary(decision),
        "decision_interpretation": (
            "None of the five fair verifier architectures learned any "
            "preregistered easy family at balanced accuracy >= 0.65 in at "
            "least two of seeds 17/29/43. Therefore the audit cannot attribute "
            "the downstream diagnostic shortcut signals to a systematically "
            "learned verifier shortcut. This is not evidence that shortcuts "
            "are absent."
        ),
        "why_mitigation_is_forbidden": (
            "ShortcutResistantVerifierTraining was preregistered as conditional "
            "on VSA_SYSTEMATIC_SHORTCUT_SUPPORTED. The condition is false at "
            "an earlier gate. Running or selecting mitigation now would violate "
            "the frozen protocol and would confound inability to learn the IID "
            "construction with mitigation effectiveness."
        ),
        "label_semantics": prereg["claim_boundary"]["labels"],
        "physical_ground_truth": False,
        "robot_safety_claim": False,
        "historical_inputs_modified": False,
        "new_physical_validity_labels_generated": False,
        "N7_N8_role": "behaviorally ambiguous reciprocal diagnostics only",
        "allowed_claims": prereg["claim_boundary"]["allowed"],
        "forbidden_claims": prereg["claim_boundary"]["forbidden"],
    }


def _claim_boundary(prereg: dict[str, Any]) -> str:
    allowed = "\n".join(
        f"- {item}." for item in prereg["claim_boundary"]["allowed"]
    )
    forbidden = "\n".join(
        f"- {item}." for item in prereg["claim_boundary"]["forbidden"]
    )
    return f"""
# Claim boundary

## Evidence semantics

All labels in this study are historical benchmark-construction labels. They
are **not physical failure truth**, task-success truth, or real-robot safety
labels. The audit asks whether a pre-execution scoring model learns the frozen
constructed classification problems and whether its scores vary with
nuisances or construction shifts.

`N1`–`N6` are audit environments, not a validated final compatibility
benchmark. `N7` and `N8` are behaviorally ambiguous reciprocal diagnostics
only; their exact action/context balancing does not turn them into physical
counterfactual evidence.

## Claims the artifacts can support

{allowed}

Each allowed statement must remain tied to the frozen families, splits,
metrics, and historical construction labels used here.

## Claims the artifacts cannot support

{forbidden}

In particular:

- High IID accuracy would not by itself establish real-world action
  verification.
- `VSA_NO_LEARNABLE_IID_TASK` does not establish that shortcuts are absent or
  that the tested models are robust. It says the prerequisite needed to
  interpret shortcut-reliance gates was not met.
- Diagnostic transfer gaps, nuisance probes, reciprocal gains, and calibration
  shifts cannot be re-described as physical correctness, causal validity, or
  safety.
- No mitigation was authorized or executed. A future mitigation result, even
  if separately authorized, could at most support construction-shift
  robustness—not safety certification.
"""


def _next_steps(decision: dict[str, Any]) -> str:
    threshold = decision["easy_IID_learnability_gate"]["threshold"]
    return f"""
# Next steps

The frozen study is terminal at `VSA_NO_LEARNABLE_IID_TASK`. Preserve it
unchanged; do not post-hoc tune this branch and do not run the conditional
mitigation.

## Recommended next branch

1. Preregister a separate **IID learnability diagnosis** before any new
   training. Its primary question should be why all five fair architectures
   stayed below the frozen `{threshold:.2f}` balanced-accuracy criterion on
   `N2/N3/N5/N6` despite strong audit-only summary controls.
2. Recheck sample-to-tensor-to-label alignment, train-only normalization,
   candidate pairing, batching, calibration/threshold selection, gradient
   flow, and per-family class balance without consulting held-out test scores
   for model selection.
3. Include negative controls (label permutation and constant input), positive
   controls with a deliberately observable synthetic target, training-set
   memorization checks, and raw-logit/gradient/parameter-update traces. Freeze
   acceptance thresholds and seeds before running them.
4. Distinguish three outcomes: implementation fault, optimization/capacity
   failure, or a representation mismatch between summary controls and fair
   verifier inputs. Any confirmed code repair must be isolated, documented,
   and followed by a fresh preregistered rerun rather than rewriting this
   audit.
5. Only if at least two fair architectures newly satisfy the easy-IID
   prerequisite across at least two frozen seeds should a fresh full shortcut
   gate be considered. Mitigation may be requested only if that new audit
   reaches `VSA_SYSTEMATIC_SHORTCUT_SUPPORTED`.
6. If the scientific target is physical action validity or robot safety,
   collect independent behavioral evidence: executed alternatives or
   appropriately controlled simulator/real-robot outcomes, explicit
   uncertainty, synchronized context/action traces, and blinded adjudication.
   Do not relabel `N1`–`N8` as physical truth.

## Ready-to-use Codex goal

```text
/goal Continue /data/projects/tzh/papers/ActMask in a new separately
preregistered VERIFIER-IID-LEARNABILITY-DIAG-V1 branch. Preserve
outputs/actmask/verifier_shortcut_audit_v1 unchanged. Diagnose why the five
fair verifier architectures failed the frozen N2/N3/N5/N6 IID BA>=0.65 in two
of seeds 17/29/43 while audit-only action/context summary controls were
predictive. Before training, freeze input/label alignment checks, train-only
normalization checks, positive and negative sanity controls, memorization and
gradient-flow tests, selection rules, seeds, resource limits, and terminal
decisions. Do not run ShortcutResistantVerifierTraining, do not use held-out
tests for selection, do not create physical-validity labels, and do not claim
robot safety. If a confirmed implementation fault is found, document one
isolated repair and rerun as a new version; otherwise stop with the diagnosed
learnability failure.
```
"""


def _executive_zh(
    decision: dict[str, Any],
    controls: dict[str, Any],
    verifier: dict[str, Any],
) -> str:
    best_control = _pooled_control_rows(controls)[0]
    pooled = _pooled_verifier_rows(verifier)
    low = min(row["balanced_accuracy_mean"] for row in pooled)
    high = max(row["balanced_accuracy_mean"] for row in pooled)
    gate = decision["easy_IID_learnability_gate"]
    return f"""
# 执行摘要

最终决定：**`{decision['decision']}`**。5 个公平 verifier 架构、3 个冻结种子
和 19 种训练规格均已运行，历史输入哈希和 1,833 行标准指标已独立复算通过。
但在预先指定为 “容易” 的 `N2/N3/N5/N6` 上，没有任何架构能在至少两个种子中
达到 BA≥{gate['threshold']:.2f}；要求是至少
{gate['required_architectures']} 个架构，实际为
{gate['qualifying_architecture_count']} 个。因此按冻结门禁顺序，研究必须在
mitigation 前停止。

这不是 “没有捷径” 的结论。相反，审计型 `current_state_action_summary_MLP`
在 pooled construction labels 上达到 BA={best_control['balanced_accuracy_mean']:.3f}，
而五个完整 verifier 的 pooled 三种子均值仅为 {low:.3f}–{high:.3f}。后续
shortcut 指标门有 5 个架构通过、任务跨度门覆盖 11 个任务，但因为更早的
IID 可学习性前提失败，这些只能作为诊断现象，不能证明 verifier 系统性地学会了
捷径。

所有标签都是历史构造标签，不是物理失败真值；`N7/N8` 仍是行为含义不确定的
互换诊断。高 IID 分数不等于真实动作验证。本轮没有运行
`ShortcutResistantVerifierTraining`，也没有生成任何 mitigation 结果，因为其
授权条件 `VSA_SYSTEMATIC_SHORTCUT_SUPPORTED` 未满足。下一步应另开预注册分支，
诊断输入/标签对齐、优化、表示与训练协议的 IID 可学习性；若目标是安全或物理
有效性，还必须重新收集独立行为证据。
"""


def _executive_en(
    decision: dict[str, Any],
    controls: dict[str, Any],
    verifier: dict[str, Any],
) -> str:
    best_control = _pooled_control_rows(controls)[0]
    pooled = _pooled_verifier_rows(verifier)
    low = min(row["balanced_accuracy_mean"] for row in pooled)
    high = max(row["balanced_accuracy_mean"] for row in pooled)
    gate = decision["easy_IID_learnability_gate"]
    return f"""
# Executive summary

Final decision: **`{decision['decision']}`**. All five fair verifier
architectures were run with three frozen seeds and 19 training
specifications. Historical input hashes passed and 1,833 standard-metric rows
were independently recomputed. However, no architecture reached balanced
accuracy >= {gate['threshold']:.2f} on any preregistered easy family
(`N2/N3/N5/N6`) in at least two seeds; the gate required
{gate['required_architectures']} qualifying architectures and observed
{gate['qualifying_architecture_count']}. The frozen gate order therefore
requires stopping before mitigation.

This does **not** show that shortcuts are absent. The audit-only
`current_state_action_summary_MLP` reached pooled construction-label
BA={best_control['balanced_accuracy_mean']:.3f}, whereas the five full
verifiers' three-seed pooled means were only {low:.3f}–{high:.3f}. All five
architectures passed the later shortcut-indicator diagnostic and evidence
spanned 11 tasks, but those observations cannot establish systematic learned
shortcut reliance after the earlier IID-learnability prerequisite failed.

The labels are historical construction labels, not physical failure truth.
`N7/N8` remain behaviorally ambiguous reciprocal diagnostics, and high IID
accuracy would not equal real-world action verification. No
`ShortcutResistantVerifierTraining` run or mitigation artifact was produced:
its prerequisite, `VSA_SYSTEMATIC_SHORTCUT_SUPPORTED`, was false. The next
scientific step is a separately preregistered IID-learnability diagnosis; a
physical-validity or safety claim would additionally require independent
behavioral evidence.
"""


def _final_report(
    decision: dict[str, Any],
    prereg: dict[str, Any],
    controls: dict[str, Any],
    verifier: dict[str, Any],
    shortcut: dict[str, Any],
    matching: dict[str, Any],
    calibration: dict[str, Any],
    relational: dict[str, Any],
    nuisance: dict[str, Any],
    imported: dict[str, Any],
) -> str:
    control_rows = _pooled_control_rows(controls)
    verifier_rows = _pooled_verifier_rows(verifier)
    easy_rows = _easy_gate_rows(decision)
    transfer_rows = shortcut["generator_and_easy_to_real"][
        "architecture_aggregate"
    ]

    control_table = "\n".join(
        "| `{}` | {:.3f} |".format(
            row["control"], row["balanced_accuracy_mean"]
        )
        for row in control_rows
    )
    verifier_table = "\n".join(
        "| `{}` | {:.3f} | {:.3f}–{:.3f} |".format(
            row["architecture"],
            row["balanced_accuracy_mean"],
            row["balanced_accuracy_minimum"],
            row["balanced_accuracy_maximum"],
        )
        for row in verifier_rows
    )
    easy_table = "\n".join(
        "| `{}` | {} | {} | {} |".format(
            row["architecture"],
            row["family"],
            ", ".join(_number(value, 3) for value in row[
                "seed_balanced_accuracy"
            ]),
            ", ".join(str(seed) for seed in row["passing_seeds"]) or "none",
        )
        for row in easy_rows
    )
    transfer_table = "\n".join(
        "| `{}` | {:.3f} | {:.3f} | {:.3f} |".format(
            row["architecture"],
            row["generator_transfer_gap_mean"],
            row["generator_transfer_gap_maximum"],
            row["easy_to_real_gap_mean"],
        )
        for row in transfer_rows
    )
    matching_table = "\n".join(
        "| `{}` | {} | {:.3f} |".format(
            row["architecture"],
            row["environment"],
            row.get(
                "action_matching_gap_mean",
                row.get("progress_matching_gap_mean"),
            ),
        )
        for row in matching["architecture_aggregate"]
    )
    calibration_table = "\n".join(
        "| `{}` | {} | {:.3f} | {:.3f} |".format(
            row["architecture"],
            row["setting"],
            row["ECE_shift_mean"],
            row["Brier_shift_mean"],
        )
        for row in calibration["architecture_aggregate"]
    )
    relational_table = "\n".join(
        "| `{}` | {:.3f} | {:.3f} |".format(
            row["architecture"],
            row["pooled_relational_gain_mean"],
            row["reciprocal_relational_gain_mean"],
        )
        for row in relational["architecture_aggregate"]
    )

    independent = decision["independent_recomputation_gate"]
    source = imported["source_content_rehash"]
    total_gpu_hours = (
        float(controls["total_GPU_hours"])
        + float(verifier["total_GPU_hours"])
    )
    nuisance_qualifying = nuisance["aggregates"][
        "architectures_qualifying_nuisance_score_probe_indicator"
    ]
    best_control = control_rows[0]
    return f"""
# VERIFIER-SHORTCUT-AUDIT-V1 final report

## Terminal outcome

**Final decision: `{decision['decision']}`.**

The study reaches a terminal non-supported decision at the second
preregistered evidence gate. Historical completeness and independent
recomputation pass, but no fair verifier architecture demonstrates the
required easy-family IID learnability. Consequently, later shortcut
diagnostics cannot be promoted to a systematic learned-shortcut claim and the
conditional mitigation is not authorized.

The decision is deliberately narrower than “shortcuts do not exist.” It says
the tested verifier/training combinations did not first establish a learnable
IID task under the frozen criterion, so a causal interpretation of their
cross-environment gaps is not identified.

## Scientific question and evidence

The frozen question was:

> {prereg['scientific_question']}

The unified audit contains historical constructed positive-versus-negative
labels. These are benchmark-construction labels—not physical failure truth,
task-success truth, or safety ground truth. `N1`–`N6` are grouped audit environments:
the easy synthetic group is `N2/N3/N5/N6`, and the real-action retrieval group
is `N1/N4`. `N7/N8` are reciprocal-balance diagnostics only and remain
behaviorally ambiguous.

The historical import was read-only (`copy_performed=false`). All 61 frozen
historical artifacts passed verification. The content audit rehashed
{source['unique_source_files']} unique source files
({source['declared_bytes'] / 1024**3:.2f} GiB) with zero failures. No
historical action tensor or label was changed, no future observation was used,
and no physical-validity label was generated.

## Experiment coverage

- Five preregistered learned architectures (the frozen maximum).
- Seeds: {", ".join(str(seed) for seed in SEEDS)}.
- 19 training specifications per architecture/seed and 285 verifier training
  runs.
- Six shortcut controls over 19 specifications and 114 control training runs.
- Environments `E1`–`E12`, including within-family, held-out family, E↔R
  transfer, action/progress matching, source/task holdout, reciprocal
  diagnostics, language/visual ablations, and prior shifts.
- Raw verifier predictions for 795 environment rows, raw control predictions
  for 318 rows, and raw family-transfer predictions for 720 rows.
- Validation-only scalar calibration remained frozen under shifts.
- Recorded training GPU time: {total_gpu_hours:.4f} hours
  ({verifier['total_GPU_hours']:.4f} verifier +
  {controls['total_GPU_hours']:.4f} controls), below the 20-hour cap.

## Preregistered gate audit

Gate order is binding; a later passing diagnostic cannot override an earlier
failure.

1. **Historical completeness and independent recomputation — PASS.**
   Historical hashes pass. Independent recomputation checked
   {independent['total_metric_rows_recomputed']} standard-metric rows:
   {independent['verifier_environment_predictions']['rows_checked']} verifier,
   {independent['shortcut_control_predictions']['rows_checked']} control, and
   {independent['family_transfer_predictions']['rows_checked']} transfer rows.
   It also verified
   {independent['representation_probe_predictions']['probe_runs_verified']}
   representation-probe runs and
   {independent['nuisance_probe_predictions']['probe_files']} nuisance-probe
   files. No recorded hash or metric mismatch remained.
2. **Easy-family IID learnability — FAIL.** At least two architectures had to
   reach BA >=
   {decision['easy_IID_learnability_gate']['threshold']:.2f} on at least one
   of `N2/N3/N5/N6` in two or more seeds. Observed qualifying architectures:
   **{decision['easy_IID_learnability_gate']['qualifying_architecture_count']}**
   of the required
   **{decision['easy_IID_learnability_gate']['required_architectures']}**.
3. **Shortcut indicators — diagnostic PASS, not decision-authorizing.**
   {decision['shortcut_indicator_gate']['qualifying_architecture_count']}
   architectures met at least two indicator definitions. This evidence occurs
   downstream of gate 2 and therefore cannot establish systematic learned
   shortcut reliance in this terminal branch.
4. **Task span — diagnostic PASS, not decision-authorizing.** Evidence was
   present for {decision['task_span_gate']['qualifying_task_count']} tasks
   versus the requirement of {decision['task_span_gate']['required_tasks']},
   but it has the same prerequisite limitation.

### Easy-family seed results

The seed columns are 17, 29, and 43. “Passing seeds” means BA >=
{decision['easy_IID_learnability_gate']['threshold']:.2f}.

| Architecture | Family | BA by seed | Passing seeds |
|---|---:|---:|---:|
{easy_table}

## Shortcut controls and fair verifier results

The audit-only controls demonstrate that construction-label signal exists in
summary/nuisance views. The strongest pooled control was
`{best_control['control']}` with BA={best_control['balanced_accuracy_mean']:.3f}.
That does not rescue the failed verifier learnability gate: controls are
diagnostics, not substitutes for the five fair architectures.

| Pooled shortcut control | Mean BA |
|---|---:|
{control_table}

| Fair verifier | Pooled mean BA | Seed range |
|---|---:|---:|
{verifier_table}

The contrast between a strong summary control and near-chance-to-weak full
verifier performance is a key diagnosis target. Plausible explanations include
input/label alignment, optimization, inductive-bias, or representation
mismatch; the terminal audit does not choose among them post hoc.

## Transfer, matching, representation, and calibration diagnostics

These results are retained for audit transparency but are secondary after the
learnability failure.

| Architecture | Mean generator gap | Maximum generator gap | Mean E→R gap |
|---|---:|---:|---:|
{transfer_table}

Generator-held-out scores themselves are close to chance in the aggregate
transfer matrix, so a large gap can arise from a high source-family score
rather than robust target-family behavior. It must not be interpreted as
physical rejection ability.

| Architecture | Matched environment | Mean BA drop |
|---|---:|---:|
{matching_table}

Average matching drops are small; individual task/family rows are preserved in
`matching_gap_report.json`. Nuisance score probes passed their coverage and
recomputation audits, but the preregistered R² indicator qualified
{len(nuisance_qualifying)} architectures. Representation probes still show
that task, progress, magnitude, or background information can be linearly
encoded in particular representations; encoding is not proof that it caused a
decision.

| Architecture | Pooled relational gain | Reciprocal diagnostic gain |
|---|---:|---:|
{relational_table}

Reciprocal gains on `N7/N8` remain construction diagnostics because both
families have unresolved behavioral ambiguity.

| Architecture | Shift | Mean ECE shift | Mean Brier shift |
|---|---:|---:|---:|
{calibration_table}

Calibration shifts are reported relative to validation-only scalar
temperatures and do not support a safety or false-accept guarantee.

## Why mitigation was not run

`ShortcutResistantVerifierTraining` was frozen as conditional on exactly
`VSA_SYSTEMATIC_SHORTCUT_SUPPORTED`. The actual terminal decision is
`VSA_NO_LEARNABLE_IID_TASK`; therefore:

- mitigation selection, training, ablation, and result files are absent;
- no mitigation architecture or checkpoint was created;
- running mitigation would violate the preregistered gate order;
- any apparent gain would be confounded with repairing or replacing a model
  that did not learn the prerequisite IID task.

The scientifically valid action is to stop, preserve the audit, and open a
separately preregistered learnability-diagnosis branch.

## Integrity and repair record

One confirmed implementation repair occurred within the frozen allowance: a
serialization-precision defect stored validation-selected float64 threshold
and temperature values as float32. The repair changed serialization precision
only; checkpoints, logits, calibrated probabilities, labels, samples, splits,
and weights were unchanged, and no retraining occurred. Independent metric
recomputation then passed. This is documented in
`prediction_serialization_precision_repair.json`.

The five-architecture limit, seeds, candidate groups, train-only
normalization, no-future-information rule, and historical preservation flags
are recorded in `verifier_config.json`. The recursive reproducibility manifest
hashes all immutable study output files and the complete audit implementation.

## Claim boundary and limitations

Supported statements are limited to shortcut solvability, nuisance
dependence, construction-family/task/source transfer behavior, matching gaps,
representation diagnostics, and calibration shift under the frozen
construction labels. This package does not validate physical failure,
counterfactual correctness, task success, real-robot safety, or a new
verification method. High IID accuracy—had it occurred—would still not equal
real-world action verification.

The study is also limited by historical constructed families, only 11 tasks,
behavioral ambiguity in `N7/N8`, reused cached representations, and a training
protocol that did not establish easy-family learnability. Consult
`failure_taxonomy.md`, `per_family_analysis.md`,
`per_task_analysis.md`, `per_shortcut_analysis.md`, and
`false_accept_analysis.md` for the detailed error analysis.

## Final disposition

The required terminal package is complete. Do not continue into 4D modeling,
world-model training, mitigation, or new physical-validity labels in this
branch. The exact bounded next study is specified in `next_steps.md`.
"""


def _package_versions() -> dict[str, str | None]:
    packages = (
        "numpy",
        "torch",
        "scipy",
        "pandas",
        "pyarrow",
        "opencv-python",
        "Pillow",
    )
    result: dict[str, str | None] = {}
    for name in packages:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def _gpu_environment() -> dict[str, Any]:
    result: dict[str, Any] = {
        "torch_imported": False,
        "cuda_available": False,
        "devices": [],
    }
    try:
        import torch

        result.update(
            {
                "torch_imported": True,
                "torch_version": torch.__version__,
                "torch_cuda_build": torch.version.cuda,
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_device_count": int(torch.cuda.device_count()),
                "cudnn_version": (
                    int(torch.backends.cudnn.version())
                    if torch.backends.cudnn.version() is not None
                    else None
                ),
            }
        )
        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                properties = torch.cuda.get_device_properties(index)
                result["devices"].append(
                    {
                        "index": index,
                        "name": properties.name,
                        "total_memory_bytes": int(
                            properties.total_memory
                        ),
                        "compute_capability": (
                            f"{properties.major}.{properties.minor}"
                        ),
                        "multiprocessor_count": int(
                            properties.multi_processor_count
                        ),
                    }
                )
    except Exception as exc:  # pragma: no cover - environment fallback
        result["torch_probe_error"] = f"{type(exc).__name__}: {exc}"

    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        result["nvidia_smi_query"] = [
            line.strip()
            for line in completed.stdout.splitlines()
            if line.strip()
        ]
    except Exception as exc:  # pragma: no cover - environment fallback
        result["nvidia_smi_error"] = f"{type(exc).__name__}: {exc}"
    return result


def _inventory_entry(path: Path, base: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(base)),
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def _build_reproducibility_manifest(
    prereg: dict[str, Any],
    controls: dict[str, Any],
    verifier: dict[str, Any],
) -> dict[str, Any]:
    excluded = set(MANIFEST_EXCLUSIONS)
    output_paths = [
        path
        for path in sorted(OUT.rglob("*"))
        if path.is_file() and str(path.relative_to(OUT)) not in excluded
    ]
    output_files = [_inventory_entry(path, OUT) for path in output_paths]
    code_root = ROOT / "actmask" / "experiments" / "verifier_shortcut_audit"
    code_paths = sorted(code_root.rglob("*.py"))
    implementation_files = [
        _inventory_entry(path, ROOT) for path in code_paths
    ]
    input_hashes = [
        {
            "path": name,
            "bytes": int((OUT / name).stat().st_size),
            "sha256": sha256_file(OUT / name),
        }
        for name in INPUT_HASH_FILES
    ]
    output_bytes = sum(row["bytes"] for row in output_files)
    total_gpu_hours = float(controls["total_GPU_hours"]) + float(
        verifier["total_GPU_hours"]
    )
    resource_limits = prereg["resource_limits"]
    log_path = OUT / "run_log.jsonl"
    log_snapshot = (
        {
            "path": "run_log.jsonl",
            "exists_at_manifest_creation": True,
            "bytes_before_terminal_append": int(log_path.stat().st_size),
            "sha256_before_terminal_append": sha256_file(log_path),
            "mutable_policy": (
                "excluded from immutable inventory because the orchestrator "
                "appends the V13 completion record after this manifest is "
                "created"
            ),
        }
        if log_path.is_file()
        else {
            "path": "run_log.jsonl",
            "exists_at_manifest_creation": False,
            "mutable_policy": "append-only log excluded from immutable inventory",
        }
    )
    always_nonself = [
        name for name in ALWAYS_OUTPUTS if name != "reproducibility_manifest.json"
    ]
    always_rows = [
        {
            "path": name,
            "exists": (OUT / name).is_file(),
            "nonempty": (
                (OUT / name).is_file() and (OUT / name).stat().st_size > 0
            ),
            "sha256": (
                sha256_file(OUT / name) if (OUT / name).is_file() else None
            ),
        }
        for name in always_nonself
    ]
    mitigation_absence = {
        name: not (OUT / name).exists() for name in MITIGATION_OUTPUTS
    }
    return {
        "schema": "vsa-reproducibility-manifest-v1",
        "study_id": STUDY_ID,
        "terminal_decision": "VSA_NO_LEARNABLE_IID_TASK",
        "inventory_policy": {
            "output_root": str(OUT),
            "recursive": True,
            "regular_files_only": True,
            "exclusions": {
                "reproducibility_manifest.json": (
                    "self-hash recursion has no stable representation"
                ),
                "run_log.jsonl": (
                    "append-only provenance log receives the V13 completion "
                    "record after manifest creation"
                ),
            },
            "all_other_output_files_hashed": True,
            "manifest_self_validation": (
                "self is validated by existence, non-empty JSON parsing, "
                "schema, and the verification routine; no self digest claimed"
            ),
        },
        "output_files": output_files,
        "output_file_count": len(output_files),
        "output_bytes": output_bytes,
        "implementation_files": implementation_files,
        "implementation_file_count": len(implementation_files),
        "input_hashes": input_hashes,
        "historical_source_hash_authority": {
            "path": "imported_hash_verification.json",
            "sha256": sha256_file(OUT / "imported_hash_verification.json"),
            "note": (
                "contains per-file hashes and byte counts for all imported "
                "historical artifacts and 354 unique raw source files"
            ),
        },
        "append_only_log_snapshot": log_snapshot,
        "environment": {
            "python_executable_expected": str(PYTHON),
            "python_executable_observed": sys.executable,
            "python_version": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "package_versions": _package_versions(),
            "gpu": _gpu_environment(),
        },
        "resources": {
            "preregistered_limits": resource_limits,
            "observed": {
                "verifier_training_GPU_hours": verifier[
                    "total_GPU_hours"
                ],
                "shortcut_control_training_GPU_hours": controls[
                    "total_GPU_hours"
                ],
                "total_recorded_training_GPU_hours": total_gpu_hours,
                "GPU_hour_limit_fraction": (
                    total_gpu_hours / resource_limits["GPU_hours_max"]
                ),
                "new_artifact_bytes_excluding_manifest_and_append_only_log": (
                    output_bytes
                ),
                "new_artifact_GiB_excluding_manifest_and_append_only_log": (
                    output_bytes / 1024**3
                ),
                "learned_verifier_architectures": verifier[
                    "learned_architectures"
                ],
                "seeds": list(verifier["seeds"]),
            },
            "within_limits": {
                "GPU_hours": (
                    total_gpu_hours <= resource_limits["GPU_hours_max"]
                ),
                "new_artifact_GiB": (
                    output_bytes / 1024**3
                    <= resource_limits["new_artifacts_GiB_max"]
                ),
                "learned_verifier_architectures": (
                    verifier["learned_architectures"]
                    <= resource_limits["learned_verifier_architectures_max"]
                ),
            },
        },
        "V13_always_outputs": {
            "required": list(ALWAYS_OUTPUTS),
            "nonself_rows": always_rows,
            "nonself_all_present_and_nonempty": all(
                row["exists"] and row["nonempty"] for row in always_rows
            ),
            "self_required_and_written_after_payload_construction": True,
        },
        "mitigation": {
            "authorized": False,
            "executed": False,
            "required_only_if_authorized": list(MITIGATION_OUTPUTS),
            "absence_checks": mitigation_absence,
            "all_conditional_outputs_absent": all(
                mitigation_absence.values()
            ),
        },
        "preservation": {
            "historical_branches_modified": False,
            "historical_action_tensors_modified": False,
            "historical_labels_modified": False,
            "new_physical_validity_labels_generated": False,
            "future_information_used": False,
        },
    }


def _write_terminal_outputs() -> None:
    _require_prerequisites()
    decision = read_json(OUT / "phase_v10_decision.json")
    prereg = read_json(OUT / "preregistered_config.json")
    controls = read_json(OUT / "shortcut_control_summary.json")
    verifier = read_json(OUT / "verifier_summary.json")
    shortcut = read_json(OUT / "shortcut_reliance_summary.json")
    matching = read_json(OUT / "matching_gap_report.json")
    calibration = read_json(OUT / "calibration_shift_report.json")
    relational = read_json(OUT / "relational_gain_report.json")
    nuisance = read_json(OUT / "nuisance_probe_report.json")
    imported = read_json(OUT / "imported_hash_verification.json")

    if any((OUT / name).exists() for name in MITIGATION_OUTPUTS):
        present = [name for name in MITIGATION_OUTPUTS if (OUT / name).exists()]
        raise RuntimeError(
            "Mitigation is unauthorized but conditional artifacts exist: "
            + ", ".join(present)
        )

    write_json(OUT / "final_decision.json", _final_decision(decision, prereg))
    _text(OUT / "claim_boundary.md", _claim_boundary(prereg))
    _text(OUT / "next_steps.md", _next_steps(decision))
    _text(
        OUT / "executive_summary_zh.md",
        _executive_zh(decision, controls, verifier),
    )
    _text(
        OUT / "executive_summary_en.md",
        _executive_en(decision, controls, verifier),
    )
    _text(
        OUT / "final_report.md",
        _final_report(
            decision,
            prereg,
            controls,
            verifier,
            shortcut,
            matching,
            calibration,
            relational,
            nuisance,
            imported,
        ),
    )
    manifest = _build_reproducibility_manifest(
        prereg,
        controls,
        verifier,
    )
    write_json(OUT / "reproducibility_manifest.json", manifest)


def verify() -> dict[str, Any]:
    """Independently re-open and verify the V13 terminal package."""
    missing_or_empty = [
        name
        for name in ALWAYS_OUTPUTS
        if not (OUT / name).is_file() or (OUT / name).stat().st_size == 0
    ]
    result: dict[str, Any] = {
        "schema": "vsa-v13-terminal-verification-v1",
        "study_id": STUDY_ID,
        "required_outputs": len(ALWAYS_OUTPUTS),
        "missing_or_empty": missing_or_empty,
        "checks": {},
    }
    if missing_or_empty:
        result["pass"] = False
        return result

    phase = read_json(OUT / "phase_v10_decision.json")
    final = read_json(OUT / "final_decision.json")
    manifest = read_json(OUT / "reproducibility_manifest.json")
    manifest_failures = []
    for row in manifest["output_files"]:
        path = OUT / row["path"]
        if (
            not path.is_file()
            or path.stat().st_size != row["bytes"]
            or sha256_file(path) != row["sha256"]
        ):
            manifest_failures.append(row["path"])
    for row in manifest["implementation_files"]:
        path = ROOT / row["path"]
        if (
            not path.is_file()
            or path.stat().st_size != row["bytes"]
            or sha256_file(path) != row["sha256"]
        ):
            manifest_failures.append(row["path"])
    input_failures = []
    for row in manifest["input_hashes"]:
        path = OUT / row["path"]
        if (
            not path.is_file()
            or path.stat().st_size != row["bytes"]
            or sha256_file(path) != row["sha256"]
        ):
            input_failures.append(row["path"])
    mitigation_present = [
        name for name in MITIGATION_OUTPUTS if (OUT / name).exists()
    ]
    excluded = set(manifest["inventory_policy"]["exclusions"])
    current_immutable_paths = {
        str(path.relative_to(OUT))
        for path in OUT.rglob("*")
        if path.is_file() and str(path.relative_to(OUT)) not in excluded
    }
    manifest_paths = {row["path"] for row in manifest["output_files"]}
    untracked = sorted(current_immutable_paths - manifest_paths)
    stale = sorted(manifest_paths - current_immutable_paths)
    narratives = {
        name: (OUT / name).read_text(encoding="utf-8").lower()
        for name in (
            "final_report.md",
            "executive_summary_zh.md",
            "executive_summary_en.md",
            "claim_boundary.md",
            "next_steps.md",
        )
    }
    boundary_ok = (
        "not physical failure truth" in narratives["final_report.md"]
        and "不是物理失败真值" in narratives["executive_summary_zh.md"]
        and "not physical failure truth"
        in narratives["executive_summary_en.md"]
        and "not physical failure truth" in narratives["claim_boundary.md"]
        and "do not run the conditional" in narratives["next_steps.md"]
    )
    result["checks"] = {
        "final_matches_frozen_V10": (
            final["decision"] == phase["decision"]
            and final["source_phase_decision_sha256"]
            == sha256_file(OUT / "phase_v10_decision.json")
        ),
        "terminal_non_supported": (
            final["terminal"] is True
            and final["decision"] == "VSA_NO_LEARNABLE_IID_TASK"
            and final["systematic_shortcut_supported"] is False
        ),
        "mitigation_not_authorized_or_executed": (
            final["mitigation_authorized"] is False
            and final["mitigation_executed"] is False
            and not mitigation_present
        ),
        "manifest_schema_and_self_policy": (
            manifest["schema"] == "vsa-reproducibility-manifest-v1"
            and "reproducibility_manifest.json" in excluded
            and "run_log.jsonl" in excluded
        ),
        "manifest_hashes_and_bytes": not manifest_failures,
        "input_hashes_and_bytes": not input_failures,
        "recursive_inventory_complete": not untracked and not stale,
        "claim_boundary_language_present": boundary_ok,
        "resource_limits_pass": all(
            manifest["resources"]["within_limits"].values()
        ),
        "V13_always_outputs_present": True,
    }
    result["details"] = {
        "manifest_files_checked": len(manifest["output_files"]),
        "implementation_files_checked": len(
            manifest["implementation_files"]
        ),
        "input_hashes_checked": len(manifest["input_hashes"]),
        "manifest_failures": manifest_failures,
        "input_failures": input_failures,
        "untracked_immutable_files": untracked,
        "stale_manifest_files": stale,
        "mitigation_artifacts_present": mitigation_present,
        "reproducibility_manifest_sha256": sha256_file(
            OUT / "reproducibility_manifest.json"
        ),
    }
    result["pass"] = all(result["checks"].values())
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="verify existing outputs without regenerating them",
    )
    args = parser.parse_args()
    if not args.verify_only:
        _write_terminal_outputs()
    result = verify()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
