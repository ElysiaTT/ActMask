"""Generate the V12 paper package from frozen V0--V10 audit artifacts.

This script deliberately writes only ``outputs/.../paper/*.md``.  It does not
touch the run log, V8--V11 analyses, V13 terminal outputs, or model artifacts.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "outputs" / "actmask" / "verifier_shortcut_audit_v1"
PAPER = OUT / "paper"

EXPECTED = (
    "title_candidates.md",
    "abstract_zh.md",
    "abstract_en.md",
    "introduction_outline.md",
    "problem_formulation.md",
    "audit_taxonomy.md",
    "audit_protocol.md",
    "negative_generator_analysis.md",
    "shortcut_metrics.md",
    "verifier_zoo.md",
    "mitigation_method.md",
    "experiments.md",
    "results_tables.md",
    "figure_plan.md",
    "claim_boundary.md",
    "limitations.md",
    "related_work_plan.md",
)

ARCHITECTURES = (
    "StateActionTCN",
    "TaskConditionedActionTransformer",
    "VisualStateActionTransformer",
    "ContrastiveContextActionVerifier",
    "CrossAttentionEnergyVerifier",
)

ARCH_SHORT = {
    "StateActionTCN": "State–Action TCN",
    "TaskConditionedActionTransformer": "Task-Conditioned Action Transformer",
    "VisualStateActionTransformer": "Visual State–Action Transformer",
    "ContrastiveContextActionVerifier": "Contrastive Context–Action Verifier",
    "CrossAttentionEnergyVerifier": "Cross-Attention Energy Verifier",
}

FAMILY_INFO = (
    (
        "N1",
        "N1_same_task_near_state",
        "R",
        "same-task near-state real-action retrieval",
        "real-action retrieval audit",
    ),
    (
        "N2",
        "N2_local_temporal_permutation",
        "E",
        "local temporal permutation",
        "easy synthetic construction",
    ),
    (
        "N3",
        "N3_arm_gripper_desynchronization",
        "E",
        "arm–gripper desynchronization",
        "easy synthetic construction",
    ),
    (
        "N4",
        "N4_phase_shifted_same_task",
        "R",
        "phase-shifted same-task real action",
        "real-action retrieval audit",
    ),
    (
        "N5",
        "N5_joint_coordination_corruption",
        "E",
        "joint-coordination corruption",
        "easy synthetic construction",
    ),
    (
        "N6",
        "N6_endpoint_matched_path_corruption",
        "E",
        "endpoint-matched path corruption",
        "easy synthetic construction",
    ),
    (
        "N7",
        "N7_reciprocal_near_state_context_swap",
        "B",
        "reciprocal near-state context swap",
        "ambiguous reciprocal diagnostic only",
    ),
    (
        "N8",
        "N8_visual_substate_reciprocal_context_swap",
        "B",
        "visual-substate reciprocal context swap",
        "ambiguous reciprocal diagnostic only",
    ),
)

CONTROL_NAMES = {
    "combined_non_relational_nuisance_classifier": "Combined non-relational nuisance classifier",
    "current_state_action_summary_MLP": "Current-state + action-summary MLP",
    "logistic_regression_combined_nuisance": "Logistic regression on combined nuisances",
    "provenance_only_classifier": "Provenance-only classifier",
    "shallow_mlp_combined_nuisance": "Shallow MLP on combined nuisances",
    "temporal_action_only_TCN": "Temporal action-only TCN",
}


def load(name: str) -> dict[str, Any]:
    with (OUT / name).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def f3(value: float) -> str:
    return f"{value:.3f}"


def pm(mean: float, std: float) -> str:
    return f"{mean:.3f} ± {std:.3f}"


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    line1 = "| " + " | ".join(headers) + " |"
    line2 = "|" + "|".join("---" for _ in headers) + "|"
    body = ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join((line1, line2, *body))


def main() -> None:
    prereg = load("preregistered_config.json")
    stats = load("unified_audit_statistics.json")
    history = load("historical_branch_inventory.json")
    roles = load("family_role_manifest.json")
    environments = load("audit_environment_manifest.json")
    controls = load("shortcut_control_summary.json")
    verifier_config = load("verifier_config.json")
    verifiers = load("verifier_summary.json")
    transfer = load("generator_transfer_matrix.json")
    matching = load("matching_gap_report.json")
    matching_quality = load("matching_quality_report.json")
    reliance = load("shortcut_reliance_summary.json")
    calibration = load("calibration_shift_report.json")
    relational = load("relational_gain_report.json")
    nuisance = load("nuisance_probe_report.json")
    representation = load("representation_probe_summary.json")
    decision = load("phase_v10_decision.json")
    recompute = load("independent_metric_recomputation.json")

    assert decision["decision"] == "VSA_NO_LEARNABLE_IID_TASK"
    assert decision["mitigation_authorized"] is False
    assert decision["must_stop_before_mitigation"] is True
    assert stats["physical_ground_truth"] is False
    assert recompute["pass"] is True

    pilot = next(
        branch for branch in history["branches"] if branch["branch"] == "rm_acv_pilot166"
    )
    realneg = next(
        branch for branch in history["branches"] if branch["branch"] == "rm_acv_realneg_v1"
    )
    historical_ba = pilot["known_shortcut"]

    control_rows = [
        row
        for row in controls["summary_rows"]
        if row["environment"] == "pooled_reference"
    ]
    control_rows.sort(key=lambda row: row["control"])
    best_control = max(control_rows, key=lambda row: row["balanced_accuracy_mean"])

    verifier_rows = {
        (row["architecture"], row["environment"]): row
        for row in verifiers["environment_summary"]
    }
    match_rows = {
        (row["architecture"], row["environment"]): row
        for row in matching["architecture_aggregate"]
    }
    reliance_rows = {
        row["architecture"]: row
        for row in reliance["generator_and_easy_to_real"]["architecture_aggregate"]
    }
    relational_rows = {
        row["architecture"]: row
        for row in relational["architecture_aggregate"]
    }
    representation_rows = {
        (row["architecture"], row["target"]): row
        for row in representation["architecture_target_summary"]
    }

    easy_values = [
        seed_row["balanced_accuracy"]
        for architecture in decision["easy_IID_learnability_gate"]["per_architecture"]
        for family in architecture["families"]
        for seed_row in family["seed_rows"]
    ]
    easy_min = min(easy_values)
    easy_max = max(easy_values)
    easy_mean = sum(easy_values) / len(easy_values)

    family_rows: list[list[str]] = []
    for family_id, full_name, group, construction, use in FAMILY_INFO:
        counts = stats["family_label_counts"][full_name]
        if family_id in {"N7", "N8"}:
            pairs = realneg[f"{family_id}_pairs"]
            ambiguity = realneg["known_ambiguity"][full_name]
            diagnostic = (
                f"{pairs} reciprocal pairs; "
                f"{100 * ambiguity['ambiguous_fraction']:.0f}% ambiguous, "
                f"{100 * ambiguity['distinct_mode_fraction']:.0f}% distinct-mode"
            )
        else:
            pairs = "—"
            h = historical_ba[full_name]
            diagnostic = (
                f"historical action-summary BA {h['action_summary_test_BA']:.3f}; "
                f"historical gate {'pass' if h['survives_historical_gate'] else 'fail'}"
            )
        family_rows.append(
            [
                family_id,
                group,
                str(counts["1"]),
                str(counts["0"]),
                str(pairs),
                construction,
                diagnostic,
                use,
            ]
        )

    table1 = markdown_table(
        [
            "Family",
            "Group",
            "Positive rows",
            "Constructed-negative rows",
            "Pairs",
            "Construction",
            "Observed historical property",
            "Permitted use",
        ],
        family_rows,
    )

    table2 = markdown_table(
        ["Shortcut control", "Inputs / role", "Pooled test BA", "Seed / status"],
        [
            [
                CONTROL_NAMES[row["control"]],
                {
                    "combined_non_relational_nuisance_classifier": "all audit-only nuisance groups; non-relational",
                    "current_state_action_summary_MLP": "current state + action summaries",
                    "logistic_regression_combined_nuisance": "combined nuisance vector, linear",
                    "provenance_only_classifier": "source/provenance metadata only",
                    "shallow_mlp_combined_nuisance": "combined nuisance vector, shallow nonlinear",
                    "temporal_action_only_TCN": "candidate action sequence only",
                }[row["control"]],
                f3(row["balanced_accuracy_mean"]),
                "17; audit-only",
            ]
            for row in control_rows
        ],
    )

    table3 = markdown_table(
        [
            "Verifier",
            "Pooled IID BA",
            "E1 within-family BA",
            "E2 generator-held-out BA",
            "E3 E→R BA",
            "E8 task-held-out BA",
            "E9 reciprocal BA*",
        ],
        [
            [
                ARCH_SHORT[architecture],
                pm(
                    verifier_rows[(architecture, "pooled_reference")][
                        "balanced_accuracy_mean"
                    ],
                    verifier_rows[(architecture, "pooled_reference")][
                        "balanced_accuracy_standard_deviation"
                    ],
                ),
                pm(
                    verifier_rows[(architecture, "E1")]["balanced_accuracy_mean"],
                    verifier_rows[(architecture, "E1")][
                        "balanced_accuracy_standard_deviation"
                    ],
                ),
                pm(
                    verifier_rows[(architecture, "E2")]["balanced_accuracy_mean"],
                    verifier_rows[(architecture, "E2")][
                        "balanced_accuracy_standard_deviation"
                    ],
                ),
                pm(
                    verifier_rows[(architecture, "E3")]["balanced_accuracy_mean"],
                    verifier_rows[(architecture, "E3")][
                        "balanced_accuracy_standard_deviation"
                    ],
                ),
                pm(
                    verifier_rows[(architecture, "E8")]["balanced_accuracy_mean"],
                    verifier_rows[(architecture, "E8")][
                        "balanced_accuracy_standard_deviation"
                    ],
                ),
                pm(
                    verifier_rows[(architecture, "E9")]["balanced_accuracy_mean"],
                    verifier_rows[(architecture, "E9")][
                        "balanced_accuracy_standard_deviation"
                    ],
                ),
            ]
            for architecture in ARCHITECTURES
        ],
    )

    table4 = markdown_table(
        [
            "Verifier",
            "E5 action-match gap",
            "E6 progress-match gap",
            "Generator-transfer gap (mean / max)",
            "E→R gap",
            "Relational gain",
        ],
        [
            [
                ARCH_SHORT[architecture],
                pm(
                    match_rows[(architecture, "E5")]["action_matching_gap_mean"],
                    match_rows[(architecture, "E5")][
                        "action_matching_gap_standard_deviation"
                    ],
                ),
                pm(
                    match_rows[(architecture, "E6")]["progress_matching_gap_mean"],
                    match_rows[(architecture, "E6")][
                        "progress_matching_gap_standard_deviation"
                    ],
                ),
                (
                    f"{reliance_rows[architecture]['generator_transfer_gap_mean']:.3f}"
                    f" / {reliance_rows[architecture]['generator_transfer_gap_maximum']:.3f}"
                ),
                pm(
                    reliance_rows[architecture]["easy_to_real_gap_mean"],
                    reliance_rows[architecture][
                        "easy_to_real_gap_standard_deviation"
                    ],
                ),
                pm(
                    relational_rows[architecture]["pooled_relational_gain_mean"],
                    relational_rows[architecture][
                        "pooled_relational_gain_standard_deviation"
                    ],
                ),
            ]
            for architecture in ARCHITECTURES
        ],
    )

    table5 = markdown_table(
        [
            "Verifier representation",
            "Family BA",
            "Action-mag. BA",
            "Progress BA",
            "Task BA",
            "Source-group BA",
            "Visual-stat. BA",
            "Label BA",
        ],
        [
            [
                ARCH_SHORT[architecture],
                *[
                    f3(
                        representation_rows[(architecture, target)][
                            "test_balanced_accuracy_mean"
                        ]
                    )
                    for target in (
                        "negative_family",
                        "action_magnitude_bin",
                        "progress_bin",
                        "task_identity",
                        "source_episode_group",
                        "visual_background_statistics_bin",
                        "label",
                    )
                ],
            ]
            for architecture in ARCHITECTURES
        ],
    )

    model_table = markdown_table(
        ["Verifier", "Inputs", "Parameters", "Relational mechanism"],
        [
            [
                ARCH_SHORT[architecture],
                ", ".join(
                    verifier_config["learned_architectures"][architecture][
                        "required_inputs"
                    ]
                ),
                f"{verifier_config['learned_architectures'][architecture]['learned_parameters']:,}",
                {
                    "StateActionTCN": "state/action encoders + product and absolute difference",
                    "TaskConditionedActionTransformer": "joint state/language/action self-attention",
                    "VisualStateActionTransformer": "joint state/visual/language/action self-attention",
                    "ContrastiveContextActionVerifier": "normalized context–action compatibility",
                    "CrossAttentionEnergyVerifier": "action self-attention then action→context cross-attention",
                }[architecture],
            ]
            for architecture in ARCHITECTURES
        ],
    )

    environment_table = markdown_table(
        ["ID", "Audit environment", "Frozen role"],
        [
            [
                environment_id,
                environments["environments"][environment_id]["name"],
                prereg["audit_environments"][environment_id],
            ]
            for environment_id in sorted(
                environments["environments"], key=lambda value: int(value[1:])
            )
        ],
    )

    transfer_matrix = transfer["balanced_accuracy_matrices"][
        "all_architectures_and_seeds_mean"
    ]
    transfer_table = markdown_table(
        ["Training source", "N1", "N2", "N3", "N4", "N5", "N6"],
        [
            [
                source,
                *[f3(transfer_matrix[source][family]) for family in ("N1", "N2", "N3", "N4", "N5", "N6")],
            ]
            for source in transfer["balanced_accuracy_matrices"][
                "row_order_source_training_specs"
            ]
        ],
    )

    nuisance_feature_rows = []
    for feature_set, row in nuisance["aggregates"]["by_feature_set"].items():
        nuisance_feature_rows.append(
            [
                feature_set,
                f3(row["heldout_decision_BA"]["mean"]),
                f3(row["heldout_decision_AUROC"]["mean"]),
                f3(row["heldout_score_R2"]["mean"]),
                str(row["runs_R2_at_least_0_30"]),
            ]
        )
    nuisance_table = markdown_table(
        [
            "Audit-only nuisance set",
            "Decision BA",
            "Decision AUROC",
            "Score R²",
            "Runs with R² ≥ 0.30",
        ],
        nuisance_feature_rows,
    )

    documents: dict[str, str] = {}

    documents["title_candidates.md"] = f"""# Title candidates

## Working title

**Are Robot Action Verifiers Actually Verifying? A Shortcut Audit of Pre-Execution Action Scoring**

This is the frozen working title from `VERIFIER-SHORTCUT-AUDIT-V1`.

## Alternative titles

1. **VerifierBench: Stress-Testing Shortcut Reliance in Robot Action Verification**
2. **ShortcutCheck: Auditing Pre-Execution Robot Action Verifiers**
3. **When Bad Actions Are Too Easy: Shortcut Learning in Robot Action Verification**

## Decision-aware variants

- **When “Invalid” Means Constructed: Auditing Pre-Execution Robot Action Verifier Benchmarks**
- **Can Robot Action Verifiers Learn the Intended Task? A Preregistered Construction-Label Audit**

## Recommended paper positioning

Use the working title, but qualify the contribution in the subtitle, abstract, and conclusion as a **benchmark-validity and model-reliance audit**. The preregistered terminal decision is `{decision['decision']}`: none of the five verifier architectures passed the easy-family IID learnability gate. The title must therefore not imply that systematic shortcut learning, physical invalidity detection, or safety certification was demonstrated.
"""

    documents["abstract_zh.md"] = f"""# 中文摘要

在机器人执行候选动作之前进行打分，常被描述为“动作验证”；但若负例由固定扰动或检索规则构造，高测试准确率可能只反映对生成器痕迹的识别。本文开展预注册研究 `VERIFIER-SHORTCUT-AUDIT-V1`，审计此类基准的构造有效性与模型依赖，而不是验证动作的物理可行性或机器人安全。统一数据包含 {stats['samples']:,} 个构造标签样本、{stats['anchors']:,} 个锚点和 {stats['tasks']} 个任务，覆盖 N1–N6 六类主审计负例及仅用于诊断的 N7/N8 互换集合。我们比较 6 个 shortcut controls、5 类公平验证器、12 个 IID/生成器留出/任务留出/匹配/模态扰动环境，并在 3 个种子上重算原始预测指标。

审计得到一个关键否定结果：最强的当前状态与动作摘要控制器在 pooled test 上达到 BA={best_control['balanced_accuracy_mean']:.3f}，部分家族迁移差距也较大；然而五类验证器在预注册的 N2/N3/N5/N6 easy-family IID 门槛上，60 个架构–家族–种子结果仅为 {easy_min:.3f}–{easy_max:.3f}（均值 {easy_mean:.3f}），没有任何架构满足 BA≥0.65 且至少两个种子成立的要求。因此终止决策为 `{decision['decision']}`，不能进入系统性 shortcut 证据判定，也未授权 mitigation。N7/N8 的行为歧义率均为 100%，只支持平衡性和一致性诊断。本文的可支持结论是：construction-label benchmark 需要生成器留出、匹配和表征探针等审计；IID 表现不等于真实世界动作验证，更不构成安全认证。
"""

    documents["abstract_en.md"] = f"""# Abstract

Pre-execution scoring is often described as robot action *verification*, yet a verifier trained against rule-generated negatives may recognize construction artifacts rather than physical invalidity. We conduct `VERIFIER-SHORTCUT-AUDIT-V1`, a preregistered audit of benchmark validity and model reliance—not a test of physical failure truth or robot safety. The unified corpus contains {stats['samples']:,} construction-labelled rows from {stats['anchors']:,} anchors and {stats['tasks']} tasks. It covers six primary construction families (N1–N6), two behaviorally ambiguous reciprocal diagnostics (N7/N8), six shortcut controls, five fair verifier architectures, twelve IID, held-out, matched, and modality-perturbation environments, and three final seeds with metrics independently recomputed from raw predictions.

The strongest current-state/action-summary control reaches pooled balanced accuracy {best_control['balanced_accuracy_mean']:.3f}, and some family-transfer gaps are large. However, the five fair verifiers score only {easy_min:.3f}–{easy_max:.3f} (mean {easy_mean:.3f}) over the 60 preregistered architecture–easy-family–seed IID checks. No architecture reaches balanced accuracy 0.65 on N2/N3/N5/N6 across at least two seeds. The terminal decision is therefore `{decision['decision']}`; the protocol stops before a systematic-shortcut conclusion, and mitigation is not authorized. N7/N8 are 100% behaviorally ambiguous and remain diagnostic only. These results establish an audit package and a negative learnability finding: construction-label performance, including high IID performance when it occurs, is not equivalent to real-world action verification and cannot certify safety.
"""

    documents["introduction_outline.md"] = f"""# Introduction outline

## 1. Motivation

- Pre-execution action scorers promise to reject bad robot actions before execution.
- A benchmark label may instead mean “logged continuation” versus “output of generator Nᵢ.” This is a construction label, not an observed success/failure outcome.
- If negative generators leave temporal, action-statistical, progress, visual, language, or provenance signatures, a model can optimize the benchmark without learning context–action validity.

## 2. Missing evaluation layer

Standard episode-disjoint IID evaluation does not isolate generator identity, task transfer, source reuse, matched nuisance factors, or calibration under shift. Consequently, high IID accuracy—where observed—cannot by itself establish real-world verification.

## 3. Scientific question

> {prereg['scientific_question']}

The question is deliberately conditional: the intended IID construction-label task must first be learnable before systematic shortcut reliance can be claimed under the preregistered gate.

## 4. Audit design

- Reuse immutable historical N1–N8 artifacts and their hashes.
- Separate easy synthetic families E={{N2,N3,N5,N6}}, real-action retrieval families R={{N1,N4}}, and ambiguous reciprocal diagnostics B={{N7,N8}}.
- Compare six shortcut-only controls with five fair verifier architectures.
- Evaluate twelve environments spanning within-family IID, generator transfer, E↔R transfer, action/progress matching, source/task holdout, reciprocal diagnostics, language/visual perturbations, and prior shifts.
- Recompute {recompute['total_metric_rows_recomputed']:,} metric rows from raw predictions.

## 5. Results narrative

1. Shortcut controls expose benchmark structure: the current-state/action-summary MLP reaches pooled BA {best_control['balanced_accuracy_mean']:.3f}.
2. Family transfer is highly uneven; the all-model/all-seed matrix reaches BA {transfer_matrix['group_r_to_e']['N4']:.3f} on N4 but stays near 0.50 on N2/N3/N5/N6.
3. Matched pooled drops are small (E5 means {min(match_rows[(a, 'E5')]['action_matching_gap_mean'] for a in ARCHITECTURES):.3f}–{max(match_rows[(a, 'E5')]['action_matching_gap_mean'] for a in ARCHITECTURES):.3f}), so matching alone does not establish shortcut reliance.
4. The prerequisite learnability gate fails: all five architectures remain near chance on the nominally easy IID families.
5. The correct terminal interpretation is `{decision['decision']}`, not “systematic shortcuts proven.”

## 6. Contributions that remain supportable

- A reproducible taxonomy and twelve-environment audit protocol for construction-labelled robot action scorers.
- A controlled five-architecture verifier zoo and six diagnostic shortcut controls.
- Raw-prediction-backed transfer, matching, calibration, nuisance, and representation analyses.
- A decision-calibrated negative result and explicit claim boundary separating benchmark validity from physical validity and safety.
"""

    documents["problem_formulation.md"] = f"""# Problem formulation

## Data and label semantics

For sample *i*, let:

- \(c_i=(s_{{i,1:8}}, v_{{i,1:8}}, \ell_i)\) be the available pre-execution context: eight 32-D state tokens, eight 22-D visual tokens, and a 128-D language embedding;
- \(a_i\\in\\mathbb{{R}}^{{16\\times32}}\) be a candidate action sequence;
- \(y_i\\in\\{{0,1\\}}\) be the historical **construction label**;
- \(g_i\\in\\{{N1,\\ldots,N8\\}}\), \(t_i\), and \(u_i\) denote generator family, task, and source episode.

Here, \(y=1\) marks a logged continuation paired by the historical construction, while \(y=0\) marks a candidate produced or retrieved by the corresponding negative family. Neither value is physical outcome truth. In particular, \(y=0\) does not guarantee failure and \(y=1\) does not guarantee task success.

## Verifier objective

A fair verifier \(f_\\theta(c,a)\) outputs a scalar logit. Models minimize binary cross-entropy on construction labels, select checkpoints by validation balanced accuracy, fit a scalar temperature on validation only, and choose a validation-only balanced-accuracy threshold. Family identity, source path, progress, mining metadata, and all future observations are forbidden fair-model inputs.

## Audit rather than certification

The standard metrics are balanced accuracy, AUROC, AUPRC, ECE, Brier score, false-accept rate, and false-reject rate. They quantify discrimination and calibration for the construction-labelled benchmark. They do not measure collision probability, task completion, physical feasibility, or safety.

## Distribution-indexed risk

Let \(e\\in\\{{E1,\\ldots,E12\\}}\) denote an audit environment. We report \(M_e(f)\) and shifts such as:

\[
\\Delta_{{gen}} = BA_{{source\ family}} - BA_{{held\!-\!out\ family}},
\\qquad
\\Delta_{{match}} = BA_{{unmatched}} - BA_{{matched}}.
\]

Easy-to-real, source-held-out, calibration, language, and relational quantities are defined analogously in `shortcut_metrics.md`. Evidence is aggregated over seeds 17, 29, and 43; confidence units are episodes or task groups, never duplicated sample rows.

## Preregistered decision logic

The first substantive gate requires at least two architectures to reach BA≥0.65 on at least one easy family among N2/N3/N5/N6 across at least two seeds. The observed 60 checks span {easy_min:.3f}–{easy_max:.3f} (mean {easy_mean:.3f}); zero architectures qualify. The terminal decision is `{decision['decision']}`. Subsequent shortcut-support and mitigation claims are therefore not authorized, even though some secondary indicators and audit-only controls show benchmark structure.
"""

    documents["audit_taxonomy.md"] = f"""# Audit taxonomy

## Axis A: negative-construction role

{markdown_table(
    ['Group', 'Families', 'Meaning in this audit', 'Physical-invalidity claim?'],
    [
        ['E', 'N2, N3, N5, N6', 'easy synthetic construction environments', 'No'],
        ['R', 'N1, N4', 'real-action retrieval audit environments', 'No'],
        ['B', 'N7, N8', 'reciprocal balance/consistency diagnostics only', 'No'],
    ],
)}

N7/N8 are not upgraded to primary negatives: they contain {realneg['N7_pairs']} and {realneg['N8_pairs']} reciprocal pairs respectively, both with 100% behavioral ambiguity and 0% distinct-mode evidence.

## Axis B: shortcut surface

1. **Action statistics:** magnitude, variance, smoothness, jerk, displacement, start/end values, and gripper events.
2. **Temporal context:** normalized progress, episode duration, action horizon, and history-change statistics.
3. **Generator identity:** family membership and generator-specific construction patterns.
4. **Provenance:** source episode, source path/directory, policy, camera, and split membership.
5. **Language:** correct, shuffled, empty, or task-ID-only conditioning.
6. **Visual nuisance:** full history, current frame, history difference, static border, center workspace, and global statistics.
7. **Prior structure:** label, family, and task prevalence.
8. **Relational evidence:** improvement of a context–action model over the stronger action-only or context-only contribution.

## Axis C: evaluation shift

{environment_table}

## Interpretation levels

- **Associational diagnostic:** a nuisance control or probe predicts construction labels/scores.
- **Robustness diagnostic:** performance changes after matching or held-out evaluation.
- **Systematic shortcut evidence:** only available if every preregistered gate passes.
- **Physical validity/safety:** unavailable in this dataset because no alternate action was executed and no physical outcome label exists.

The present study ends at `{decision['decision']}`. It supplies the first two kinds of evidence but does not authorize the third or fourth.
"""

    documents["audit_protocol.md"] = f"""# Audit protocol

## Frozen inputs

- Study: `VERIFIER-SHORTCUT-AUDIT-V1`
- Preregistered configuration hash: `{(OUT / 'preregistered_config.sha256').read_text(encoding='utf-8').strip()}`
- Unified corpus: {stats['samples']:,} rows, {stats['anchors']:,} anchors, {stats['tasks']} complete tasks
- Families: N1–N6 primary; N7/N8 diagnostic only
- Imported action tensors and historical labels: unchanged
- Future state/RGB: forbidden

## Split discipline

The imported episode split remains unchanged. Candidate source and context episodes must share the same split; candidates never cross splits. Five task-CV folds use complete task groups. Normalizers, checkpoint selection, calibration temperatures, and thresholds use train/validation only; test data are not used for fitting or selection.

## Matching

E5 matches opposite construction labels by standardized action magnitude, variance, smoothness, jerk, displacement, endpoints, and gripper events. E6 matches normalized progress, duration, and horizon. Both use deterministic one-to-one greedy minimum distance with lexical sample-ID tie breaks; the caliper is the 90th percentile of development/train opposite-label nearest-neighbor distances and is frozen before model evaluation. Matching retained {100 * matching_quality['coverage']['mean']:.1f}% of samples on average (range {100 * matching_quality['coverage']['minimum']:.1f}%–{100 * matching_quality['coverage']['maximum']:.1f}%).

## Model protocol

Five preregistered learned architectures share AdamW (learning rate 0.002, weight decay 0.0003), batch size 512, at most 40 epochs, early-stop patience 6, gradient clipping at 2.0, and CUDA bfloat16 autocast. Seeds are 17/29/43. Training comprises {verifiers['training_runs']} verifier runs over {verifiers['training_specifications']} specifications. Six audit-only controls use seed 17 across {controls['training_runs']} runs.

## Metric integrity

All standard and shortcut metrics are recomputed from serialized raw predictions. The independent audit checked {recompute['total_metric_rows_recomputed']:,} standard-metric rows: {recompute['verifier_environment_predictions']['rows']} verifier-environment, {recompute['family_transfer_predictions']['rows']} family-transfer, and {recompute['shortcut_control_predictions']['rows']} control rows. Separately, it verified 75 nuisance-probe files and 105 representation-probe runs. Hash failures and metric mismatches were zero.

## Ordered gate

1. Historical artifact integrity — passed.
2. Easy-family IID learnability — failed (0 qualifying architectures; 2 required).
3. Shortcut indicator and task-span gates — recorded diagnostically but cannot override step 2.
4. Conditional mitigation — not authorized; stop before training.

This order prevents a post hoc shortcut claim from secondary patterns when the intended easy-family task itself was not learned.
"""

    documents["negative_generator_analysis.md"] = f"""# Negative-generator analysis

## Main inventory

{table1}

All counts and historical properties come from the hash-verified imported branches and the unified manifest. “Positive” and “negative” mean logged-versus-constructed rows, not physically valid-versus-invalid actions.

## N1: same-task near-state retrieval

N1 uses a real logged action retrieved from a nearby state in the same task. Its historical action-summary BA was {historical_ba['N1_same_task_near_state']['action_summary_test_BA']:.3f}, below the original shortcut rejection threshold, so it survived as a Group-R audit family. This survival is not behavioral validation: the retrieved action was never executed in the target context.

## N2: local temporal permutation

N2 perturbs local action order. Historical action summaries reached BA {historical_ba['N2_local_temporal_permutation']['action_summary_test_BA']:.3f}; it failed the historical shortcut gate. It remains useful only as an easy synthetic audit construction.

## N3: arm–gripper desynchronization

N3 offsets arm and gripper timing. Historical action-summary BA was {historical_ba['N3_arm_gripper_desynchronization']['action_summary_test_BA']:.3f}. Coverage is smaller ({stats['family_label_counts']['N3_arm_gripper_desynchronization']['0']} constructed negatives) than the other N1–N6 families.

## N4: phase-shifted same-task action

N4 retrieves/phase-shifts real same-task motion. Its historical action-summary BA was {historical_ba['N4_phase_shifted_same_task']['action_summary_test_BA']:.3f}, and it survived the historical gate. Nevertheless, the all-architecture/all-seed transfer matrix shows strong construction specificity: training on R scores N4 at BA {transfer_matrix['group_r_to_e']['N4']:.3f}, while training on E transfers to N4 at only {transfer_matrix['group_e_to_r']['N4']:.3f}.

## N5: joint-coordination corruption

N5 disrupts cross-joint coordination. Historical action-summary BA reached {historical_ba['N5_joint_coordination_corruption']['action_summary_test_BA']:.3f}, the strongest of N1–N6, so it is explicitly categorized as an easy synthetic construction rather than behavioral truth.

## N6: endpoint-matched path corruption

N6 preserves endpoints while changing the intermediate path. Historical action-summary BA was {historical_ba['N6_endpoint_matched_path_corruption']['action_summary_test_BA']:.3f}. Endpoint matching therefore does not remove broader path-statistical cues.

## N7 and N8: reciprocal diagnostics

N7 contains {realneg['N7_pairs']} pairs ({stats['family_label_counts']['N7_reciprocal_near_state_context_swap']['0'] + stats['family_label_counts']['N7_reciprocal_near_state_context_swap']['1']} rows); N8 contains {realneg['N8_pairs']} pairs ({stats['family_label_counts']['N8_visual_substate_reciprocal_context_swap']['0'] + stats['family_label_counts']['N8_visual_substate_reciprocal_context_swap']['1']} rows). Both attain exact action/context reuse balance, yet both are 100% behaviorally ambiguous and have 0% distinct-mode evidence. Their E9 performance can measure reciprocal score consistency and source dependence only. It cannot be reported as action-validity accuracy.

## Cross-family finding

{transfer_table}

The matrix is mean balanced accuracy over all five architectures and three seeds. Strong N4 diagonal/related-source behavior coexists with near-chance performance across most other cells. This is evidence that the benchmark families induce different learnability regimes; because the easy-family IID prerequisite failed, it is not sufficient for the terminal claim “systematic shortcut learning.”
"""

    documents["shortcut_metrics.md"] = f"""# Shortcut metrics

## Standard construction-label metrics

Balanced accuracy, AUROC, AUPRC, ECE, Brier score, false-accept rate, and false-reject rate are computed from raw predictions. A “false accept” is a constructed-negative row scored positive under the historical label; it is not an unsafe action proven acceptable.

## Generator-transfer gap

\[
\\Delta_{{gen}}=BA_{{source-family\ IID}}-BA_{{target-family}}.
\]

Across architectures, mean generator-transfer gaps are {min(reliance_rows[a]['generator_transfer_gap_mean'] for a in ARCHITECTURES):.3f}–{max(reliance_rows[a]['generator_transfer_gap_mean'] for a in ARCHITECTURES):.3f}; maximum observed aggregate gaps are {min(reliance_rows[a]['generator_transfer_gap_maximum'] for a in ARCHITECTURES):.3f}–{max(reliance_rows[a]['generator_transfer_gap_maximum'] for a in ARCHITECTURES):.3f}. The preregistered generator-held-out indicator fires for all five architectures, but the terminal gate still stops earlier at failed IID learnability.

## Easy-to-real gap

\[
\\Delta_{{E\\rightarrow R}}=BA_{{Group\ E\ IID}}-BA_{{train\ E,test\ R}}.
\]

Architecture means range from {min(reliance_rows[a]['easy_to_real_gap_mean'] for a in ARCHITECTURES):.3f} to {max(reliance_rows[a]['easy_to_real_gap_mean'] for a in ARCHITECTURES):.3f}. These small values are not proof of successful transfer: both sides are approximately chance because Group E was not learned.

## Matched gaps

E5 action matching and E6 progress/duration/horizon matching use \(\\Delta=BA_{{unmatched}}-BA_{{matched}}\). Mean E5 gaps are {min(match_rows[(a, 'E5')]['action_matching_gap_mean'] for a in ARCHITECTURES):.3f}–{max(match_rows[(a, 'E5')]['action_matching_gap_mean'] for a in ARCHITECTURES):.3f}; E6 gaps are {min(match_rows[(a, 'E6')]['progress_matching_gap_mean'] for a in ARCHITECTURES):.3f}–{max(match_rows[(a, 'E6')]['progress_matching_gap_mean'] for a in ARCHITECTURES):.3f}. None reaches the preregistered 0.08 shortcut indicator threshold across two seeds.

## Source-held-out gap

E7 jointly holds out context/source episodes under the imported split. Its score equals the pooled-reference score in this corpus because camera and policy have only one observed value and source holding is already enforced by the episode split. A zero gap here is a coverage limitation, not invariance to arbitrary deployment sources.

## Relational gain

\[
G_{{rel}}=BA_{{full}}-\\max(BA_{{action-only}},BA_{{context-only}}).
\]

Pooled mean gains range {min(relational_rows[a]['pooled_relational_gain_mean'] for a in ARCHITECTURES):.3f}–{max(relational_rows[a]['pooled_relational_gain_mean'] for a in ARCHITECTURES):.3f}. This is construction-label complementarity, not causal context–action validity.

## Nuisance score predictability

An audit-only ridge probe predicts continuous verifier scores from nuisance groups, and a logistic decision probe predicts thresholded decisions. Held-out aggregates are:

{nuisance_table}

No architecture has nuisance-score R²≥0.30 across the required seeds; predictability is associational and cannot establish a causal shortcut.

## Calibration shift

The validation-only temperature is fixed under shift. Generator-held-out mean Brier increases span {min(row['Brier_shift_mean'] for row in calibration['architecture_aggregate'] if row['setting']=='generator_heldout'):.3f}–{max(row['Brier_shift_mean'] for row in calibration['architecture_aggregate'] if row['setting']=='generator_heldout'):.3f}. ECE changes may be signed and do not imply improved safety calibration.
"""

    documents["verifier_zoo.md"] = f"""# Verifier zoo

## Architecture inventory

{model_table}

All five are the complete preregistered learned-model budget. They receive identical candidate groups and never receive family identity, source episode/path, progress, matching distance, mining rank, reason/evidence codes, or future state/RGB.

## Common training and calibration

- AdamW, learning rate 0.002, weight decay 0.0003
- batch size 512, at most 40 epochs, early-stop patience 6
- validation balanced-accuracy checkpoint selection
- validation-only scalar temperature and threshold
- seeds 17, 29, 43
- 19 frozen training specifications, totaling {verifiers['training_runs']} runs

## Observed performance

{table3}

\*E9 is an N7/N8 reciprocal diagnostic over behaviorally ambiguous pairs; it is not physical-validity accuracy.

Pooled IID means fall between {min(verifier_rows[(a, 'pooled_reference')]['balanced_accuracy_mean'] for a in ARCHITECTURES):.3f} and {max(verifier_rows[(a, 'pooled_reference')]['balanced_accuracy_mean'] for a in ARCHITECTURES):.3f}. Generator-held-out and task-held-out means are closer to chance. Most importantly, V10 does not use the pooled average to declare learnability: it requires at least two architectures to reach BA≥0.65 on at least one easy family among N2/N3/N5/N6 across at least two seeds. No architecture qualifies.

## Relational architectures

The contrastive and cross-attention designs were included to test whether explicit context–action interaction improves beyond action-only/context-only controls. Their pooled relational gains are {relational_rows['ContrastiveContextActionVerifier']['pooled_relational_gain_mean']:.3f} and {relational_rows['CrossAttentionEnergyVerifier']['pooled_relational_gain_mean']:.3f}, respectively. These modest gains do not rescue the failed learnability gate and must not be interpreted as physical counterfactual reasoning.
"""

    documents["mitigation_method.md"] = f"""# Conditional mitigation method — not authorized

## Authorization status

**NOT AUTHORIZED / 未授权。 No mitigation training, checkpoint, prediction, ablation, or mitigation result exists in this study.** The preregistered condition was `VSA_SYSTEMATIC_SHORTCUT_SUPPORTED`; the observed terminal decision is `{decision['decision']}`. `phase_v10_decision.json` sets `mitigation_authorized=false` and `must_stop_before_mitigation=true`.

## Preregistered conditional method

Had authorization been reached, `ShortcutResistantVerifierTraining` would have started from the highest-validation-BA fair cross-attention or Transformer model (lexical tie-break) and combined:

1. family- and task-balanced sampling;
2. Group DRO over family × task × source groups;
3. gradient-reversal heads for family, action magnitude, progress, and source;
4. matched-pair prediction consistency.

The frozen coefficients were \(\\lambda_{{adv}}=0.1\), \(\\lambda_{{group}}=0.2\), and \(\\lambda_{{match}}=0.1\), evaluated with seeds 17/29/43. Planned ablations were ERM, family-balanced only, task-balanced only, Group-DRO only, nuisance-adversarial only, matched-training only, and the full method.

## Conditional acceptance rule

Authorization alone would not imply success. Acceptance required all of the following: ≥0.04 generator-held-out BA gain; ≥0.04 E→R gain; ≥0.04 gain in at least one matched environment; improvement in at least three families and three tasks; ≤0.03 IID regression; ≤0.02 task-held-out regression; decreased nuisance predictability; improvement in calibration or false-accept rate; and exact three-seed independent recomputation.

## Permitted interpretation if a future run succeeds

A successful future mitigation could support only **improved robustness to the measured construction shifts**. It would not show that rejected actions physically fail, accepted actions succeed, or the verifier is safety-certified. No sentence in the current paper may use present-tense language such as “our mitigation improves,” because mitigation was not run.
"""

    documents["experiments.md"] = f"""# Experiments

## Data

The unified audit has {stats['samples']:,} rows from {stats['anchors']:,} anchors and {stats['tasks']} tasks. N1/N2/N4/N5/N6 each contribute 992 positive and 992 constructed-negative rows; N3 contributes 640+640; N7 contributes 578+578; and N8 contributes 160+160. All {stats['samples']:,} rows pass the source-safe split check. No action value or historical label was modified.

## Environments

{environment_table}

N1–N6 are used for fair training and primary evaluation. N7/N8 appear only in E9 diagnostics because their labels are behaviorally ambiguous. E5/E6 matching was frozen before held-out model performance and retained 91.3% of samples on average.

## Shortcut controls

Six audit-only controls test whether labels can be predicted from provenance, non-relational nuisances, current state/action summaries, or action sequence alone. They use the same 19 training specifications, seed 17, validation-only calibration/thresholding, and total {controls['training_runs']} training runs ({controls['total_GPU_hours']:.3f} reported GPU-hours).

## Fair verifier zoo

Five architectures share the protocol in `verifier_zoo.md`. The study trains {verifiers['training_runs']} runs (5 architectures × 19 specifications × 3 seeds), reporting {verifiers['total_GPU_hours']:.3f} GPU-hours. The small runtime reflects compact frozen representations and models with 62k–96k parameters; it does not include historical data production.

## Transfer and matching

Family-to-family transfer reloads each frozen checkpoint/normalizer and tests all N1–N6 targets, yielding {transfer['per_seed_evaluations']} per-seed evaluations. E5 and E6 compare pooled checkpoints on unmatched versus deterministically matched tests. E3/E4 distinguish E→R from diagnostic R→E transfer.

## Probes and perturbations

- 75 nuisance-score probes: five feature sets × five architectures × three seeds.
- 105 frozen-representation probes: seven targets × five architectures × three seeds.
- E10 language variants: correct/shuffled/empty/task-ID-only.
- E11 visual variants: full/current/difference/border/center/global statistics.
- E12 balanced and shifted label/family/task weights.

## Metrics and terminal rule

The seven standard metrics and nine shortcut metrics are recomputed from raw predictions. The primary learnability gate fails before the systematic-shortcut gate: easy-family BA is {easy_min:.3f}–{easy_max:.3f}, with zero qualifying architectures. Mitigation experiments are therefore absent by design, not missing due to runtime failure.
"""

    documents["results_tables.md"] = f"""# Results tables

All values below are computed from existing V0–V10 artifacts. BA means balanced accuracy on construction labels. Unless stated otherwise, verifier entries are mean ± population standard deviation over the evaluations represented by the frozen summary. These tables do not report physical-validity or safety accuracy.

## Table 1. Historical negative families and known shortcut properties

{table1}

N7/N8 labels are behaviorally ambiguous; their counts and pair diagnostics must not be interpreted as failure labels.

## Table 2. Shortcut controls

{table2}

The strongest control is `{best_control['control']}` at BA {best_control['balanced_accuracy_mean']:.3f}. This is benchmark-structure evidence, not physical verification.

## Table 3. Verifier zoo across IID and held-out environments

{table3}

\*E9 is diagnostic only. E1 aggregates six family-specific models and three seeds, so its standard deviation includes family heterogeneity. The V10 easy-family gate is stricter than the aggregate: no architecture reaches BA≥0.65 for N2/N3/N5/N6 across two seeds.

## Table 4. Matched evaluation and transfer gaps

{table4}

Positive matching/transfer gaps denote lower BA after the specified shift. Near-zero E→R gaps coexist with chance-level endpoints and therefore do not mean robust transfer.

## Table 5. Frozen representation probes

{table5}

Values are held-out-test BA averaged over seeds. Family chance BA is 1/6=0.167, action/progress/visual-bin chance is 0.25, task chance is 1/11≈0.091, and label chance is 0.50. Family BA never reaches the preregistered 0.70 representation indicator in two seeds. Probe decodability is associational, not causal proof.

## Table 6. Mitigation and ablations — not authorized

| Authorization condition | Observed decision | Training runs | Checkpoints | Result claim |
|---|---|---:|---:|---|
| `VSA_SYSTEMATIC_SHORTCUT_SUPPORTED` required | `{decision['decision']}` | 0 | 0 | Not authorized; no mitigation result |

Table 6 is a status record only. It contains no fabricated method or ablation measurements.
"""

    documents["figure_plan.md"] = f"""# Figure plan

The package plans exactly six main-figure slots. Figures 1–5 may be generated from frozen V0–V10 data; Figure 6 is explicitly not authorized and must not be generated.

## Figure 1 — Negative construction families and visible shortcut surfaces

- **Panels:** N1–N8 schematic; E/R/B role bands; action, temporal, visual, language, and provenance cue icons.
- **Data:** `family_role_manifest.json`, `historical_branch_inventory.json`, `unified_audit_statistics.json`.
- **Annotations:** N1–N6 construction-label semantics; N7/N8 “100% ambiguous, diagnostic only.”
- **Caption claim:** the generators create distinct audit surfaces; no panel should label any candidate “physically invalid.”

## Figure 2 — Verifier training and audit environments

- **Layout:** training inputs → five fair verifiers → E1–E12 evaluation branches.
- **Mark:** validation-only selection/calibration, episode/task split barriers, forbidden future and metadata inputs.
- **Data:** `audit_environment_manifest.json`, `verifier_config.json`.
- **Caption claim:** a preregistered environment suite separates IID, generator, task, matching, source, modality, and prior shifts.

## Figure 3 — Family-to-family transfer matrix

- **Plot:** 8×6 heat map using the all-architecture/all-seed matrix in `generator_transfer_matrix.json`.
- **Range:** common BA scale centered at 0.50; annotate N4 values {transfer_matrix['within_n4']['N4']:.3f} (within-N4) and {transfer_matrix['group_r_to_e']['N4']:.3f} (R-trained).
- **Overlay:** E/R source and target brackets.
- **Caption claim:** transfer is generator-specific and uneven; the matrix does not by itself pass the ordered shortcut gate.

## Figure 4 — IID versus matched and held-out performance

- **Plot:** architecture-level points for pooled, E2 generator-held-out, E3 E→R, E5 action-matched, E6 progress-matched, and E8 task-held-out.
- **Show:** three-seed points plus aggregate; do not hide near-chance endpoints behind “small gap” bars.
- **Data:** `verifier_per_environment.json`, `matching_gap_report.json`.
- **Caption claim:** matched drops are only {min(match_rows[(a, 'E5')]['action_matching_gap_mean'] for a in ARCHITECTURES):.3f}–{max(match_rows[(a, 'E5')]['action_matching_gap_mean'] for a in ARCHITECTURES):.3f}, while the learnability prerequisite fails.

## Figure 5 — Nuisance/representation signal versus verifier performance

- **Plot:** x-axis held-out probe BA or score R²; y-axis pooled and generator-held-out BA; color by architecture and target.
- **Data:** `nuisance_probe_report.json`, `representation_probe_summary.json`, `verifier_summary.json`.
- **Reference lines:** score R²=0.30 and family-probe BA=0.70 preregistered thresholds.
- **Caption claim:** action magnitude, progress, task, and visual statistics can be decoded to varying degrees, but no nuisance score-probe or family-probe gate qualifies.

## Figure 6 — Mitigation results

**NOT AUTHORIZED — DO NOT GENERATE.**

The required condition was `VSA_SYSTEMATIC_SHORTCUT_SUPPORTED`; the terminal decision is `{decision['decision']}`. No mitigation checkpoints, predictions, or ablations exist. Keep the slot as a clearly crossed-out planning placeholder or omit it from the submitted figure set; never draw hypothetical bars.
"""

    documents["claim_boundary.md"] = f"""# Claim boundary

## What the labels mean

The labels are **construction labels**: logged continuation versus a candidate produced/retrieved by N1–N8. They are not physical failure truth. A negative does not guarantee physical failure; a positive does not guarantee success.

## What this study audits

- validity and learnability of a construction-labelled benchmark;
- reliance of model scores/decisions on generator, nuisance, task, source, language, and visual structure;
- transfer, matching, calibration, and representation behavior under frozen environments.

## Claims supported by the evidence

1. The corpus and its historical artifacts are internally reproducible and hash-verified.
2. Audit-only controls find construction structure; the best pooled control reaches BA {best_control['balanced_accuracy_mean']:.3f}.
3. Family transfer is uneven, matching effects are small on average, and several nuisance/task properties are linearly decodable.
4. The five fair verifier architectures do **not** learn the preregistered easy-family IID task: {easy_min:.3f}–{easy_max:.3f}, zero qualifying architectures.
5. The terminal decision is `{decision['decision']}`.

## Claims not supported

- “The constructed negatives are physically invalid.”
- “The logged positives are guaranteed successful.”
- “IID benchmark accuracy equals real-world action verification.”
- “The verifier certifies safe execution.”
- “Systematic shortcut learning is established.” The ordered gate stops before that conclusion.
- “N7/N8 are counterfactual failure examples.” Both are 100% behaviorally ambiguous.
- “The proposed mitigation improves robustness.” Mitigation was not authorized or run.

## Conditional future mitigation language

If a future authorized experiment meets every preregistered acceptance criterion, the strongest permissible statement would be “improves robustness to the measured construction shifts.” It still would not amount to safety certification or physical-validity evidence.
"""

    documents["limitations.md"] = f"""# Limitations

1. **No physical outcome truth.** Alternate candidates were not executed, so construction labels cannot establish success, failure, feasibility, or safety.
2. **Failed prerequisite learnability.** The nominally easy N2/N3/N5/N6 task remains near chance for all five fair architectures. This prevents the intended systematic-shortcut conclusion and makes some gap metrics hard to interpret because both endpoints are weak.
3. **Single historical corpus.** The audit covers {stats['tasks']} RoboMIND-derived tasks and {stats['anchors']:,} anchors; it does not establish generality to other robots, embodiments, controllers, or deployment policies.
4. **Ambiguous reciprocal data.** N7/N8 have 100% behavioral ambiguity and 0% distinct-mode evidence. E9 is consistency diagnostics only.
5. **Uneven family coverage.** N3 has 640 negatives versus 992 for most primary families. N8 has only four imported test rows, making test summaries particularly unstable.
6. **Restricted source diversity.** E7 has one available camera (`camera_front`) and one logged-demonstration policy; scene metadata are unavailable. Its zero source gap cannot be generalized to unseen cameras, policies, or scenes.
7. **Matching is feature-limited.** E5/E6 match measured summaries, not all latent context variables. Some action-family standardized differences remain after matching (especially synthetic families), and coverage ranges from 77.8% to 95.7%.
8. **Compact frozen representations.** Visual inputs are 22-D tokens rather than raw images, language is a frozen 128-D embedding, and state/action dimensions are fixed at 32. Conclusions concern this interface.
9. **Small verifier zoo.** Five compact architectures (62k–96k parameters) satisfy the preregistered budget but do not span all possible models. The negative learnability decision is scoped to this audit protocol.
10. **Probe limits.** Linear decodability and nuisance score predictability are associational. A low R² can reflect model nonlinearity; a high probe BA does not prove causal use.
11. **Calibration semantics.** ECE and Brier scores calibrate construction labels, not physical risk probabilities.
12. **No mitigation evidence.** The ordered gate withheld authorization. Figure 6 and Table 6 therefore have no experimental values.
"""

    documents["related_work_plan.md"] = """# Related-work plan

This file is a citation and positioning plan, not a fabricated bibliography. Final writing should verify every claim against primary papers or official dataset documentation.

## 1. Robot action feasibility, affordance, and pre-execution scoring

- Survey action feasibility/affordance classifiers, critic or value models, and learned failure predictors.
- Separate methods trained on executed outcomes from methods trained on synthetic or retrieved negatives.
- Record whether evaluation is offline discrimination, online task success, collision avoidance, or calibrated risk.

## 2. Offline robot datasets and counterfactual limitations

- Cover RoboMIND and comparable large-scale manipulation datasets using official dataset papers/docs.
- Explain why demonstrations provide logged outcomes but usually not the outcome of an unexecuted alternate action.
- Distinguish negative mining from physical counterfactual supervision.

## 3. Shortcut learning and dataset artifacts

- Review shortcut learning, dataset bias, Clever Hans behavior, and spurious correlations.
- Prioritize work with generator-held-out, domain-held-out, matched, or causal intervention evaluations.
- Position this audit around construction-family transfer rather than generic “bias.”

## 4. Out-of-distribution and group robustness

- Cover Group DRO, invariant learning, adversarial nuisance removal, and balanced sampling.
- State that these methods motivate the *conditional* mitigation design; none was run here.
- Do not cite robustness to a measured construction shift as safety robustness.

## 5. Contrastive and cross-attention compatibility models

- Review contrastive context–action alignment and cross-attention energy scoring.
- Compare the relation modeled (state/action, vision/action, language/action) and supervision source.
- Position the five-model zoo as controlled audit instruments, not a state-of-the-art architecture claim.

## 6. Evaluation under matching, calibration, and prevalence shift

- Survey propensity/matching diagnostics, calibration under distribution shift, and label-prior shift.
- Explain the limits of matching observed features and of ECE on non-physical labels.
- Identify precedents for episode/task-group uncertainty rather than row-level confidence.

## 7. Representation and nuisance probes

- Review linear probes and cautions against causal interpretations.
- Separate “information is decodable” from “the decision uses this information.”
- Connect representation probes to raw-score nuisance probes and held-out performance.

## 8. Safety assurance and runtime monitors

- Review formal/runtime assurance, control barrier or reachability methods, and calibrated safety monitoring.
- Use this literature to sharpen the boundary: construction-label classification is not safety certification.

## Citation audit checklist

- Use primary sources for method claims and official sources for dataset facts.
- Record version/year, robot/task scope, label source, negative construction, and evaluation split.
- Do not cite a paper as physical validation unless alternate actions were executed or otherwise physically adjudicated.
- Verify current benchmarks and dataset versions before submission.
- Add no claim that N7/N8 are true counterfactuals.
- Describe mitigation only as preregistered future work because it was not authorized.
"""

    PAPER.mkdir(parents=True, exist_ok=True)
    unexpected = sorted(
        path.name for path in PAPER.iterdir() if path.is_file() and path.name not in EXPECTED
    )
    if unexpected:
        raise RuntimeError(
            "Refusing to make a non-exact paper package; unexpected files: "
            + ", ".join(unexpected)
        )
    for name in EXPECTED:
        content = documents[name].strip() + "\n"
        (PAPER / name).write_text(content, encoding="utf-8")

    actual = sorted(path.name for path in PAPER.iterdir() if path.is_file())
    if actual != sorted(EXPECTED):
        raise RuntimeError(f"paper package mismatch: expected={sorted(EXPECTED)} actual={actual}")
    if len(actual) != 17:
        raise RuntimeError(f"expected exactly 17 documents, found {len(actual)}")
    for name in EXPECTED:
        text = (PAPER / name).read_text(encoding="utf-8")
        if len(text) < 400:
            raise RuntimeError(f"{name} is unexpectedly short")
    if "NOT AUTHORIZED — DO NOT GENERATE" not in documents["figure_plan.md"]:
        raise RuntimeError("Figure 6 authorization boundary missing")
    if "no mitigation result" not in documents["results_tables.md"].lower():
        raise RuntimeError("Table 6 authorization boundary missing")

    print(
        json.dumps(
            {
                "paper_directory": str(PAPER),
                "documents": actual,
                "document_count": len(actual),
                "decision": decision["decision"],
                "mitigation_authorized": decision["mitigation_authorized"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
