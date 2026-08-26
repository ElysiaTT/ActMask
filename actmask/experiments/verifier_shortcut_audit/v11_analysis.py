"""Generate the frozen V11 error and failure analysis.

This phase is descriptive only.  It recomputes subgroup summaries from the
frozen V8/V9 prediction artifacts and consumes the V10 decision; it does not
train, tune, mitigate, or append to the study run log.
"""
from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .common import OUT, STUDY_ID, read_json, read_jsonl, sha256_file, write_json
from .metrics import binary_metrics


ARCHITECTURES = (
    "StateActionTCN",
    "TaskConditionedActionTransformer",
    "VisualStateActionTransformer",
    "ContrastiveContextActionVerifier",
    "CrossAttentionEnergyVerifier",
)
FAMILY_NAMES = {
    "N1": "same-task near-state retrieval",
    "N2": "local temporal permutation",
    "N3": "arm/gripper desynchronization",
    "N4": "phase-shifted same-task retrieval",
    "N5": "joint-coordination corruption",
    "N6": "endpoint-matched path corruption",
    "N7": "reciprocal near-state context swap",
    "N8": "visual-substate reciprocal context swap",
}
FAMILY_GROUPS = {
    "N1": "R",
    "N2": "E",
    "N3": "E",
    "N4": "R",
    "N5": "E",
    "N6": "E",
    "N7": "B",
    "N8": "B",
}
CONTROL_LABELS = {
    "combined_non_relational_nuisance_classifier": "combined non-relational",
    "current_state_action_summary_MLP": "current-state + action-summary MLP",
    "logistic_regression_combined_nuisance": "combined nuisance logistic",
    "provenance_only_classifier": "provenance only",
    "shallow_mlp_combined_nuisance": "combined nuisance shallow MLP",
    "temporal_action_only_TCN": "action-only temporal TCN",
}
INPUT_FILES = (
    "phase_v10_decision.json",
    "independent_metric_recomputation.json",
    "verifier_per_environment.json",
    "shortcut_control_per_environment.json",
    "shortcut_control_summary.json",
    "shortcut_reliance_summary.json",
    "matching_gap_report.json",
    "matching_quality_report.json",
    "nuisance_probe_report.json",
    "representation_probe_summary.json",
    "relational_gain_report.json",
    "generator_transfer_matrix.json",
    "calibration_shift_report.json",
    "family_claim_boundary.json",
    "family_role_manifest.json",
    "unified_audit_manifest.jsonl",
)


def mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        raise ValueError("cannot average an empty collection")
    return float(statistics.fmean(values))


def std(values: Iterable[float]) -> float:
    values = list(values)
    return float(statistics.pstdev(values)) if len(values) > 1 else 0.0


def summary(values: Iterable[float]) -> dict[str, float | int]:
    values = [float(value) for value in values]
    if not values:
        raise ValueError("cannot summarize an empty collection")
    return {
        "count": len(values),
        "mean": mean(values),
        "standard_deviation": std(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def optional_summary(
    values: Iterable[float],
) -> dict[str, float | int] | None:
    values = list(values)
    return summary(values) if values else None


def metric_summary(rows: list[dict[str, Any]], key: str) -> dict[str, float | int]:
    return summary(row["metrics"][key] for row in rows)


def markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    return [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
        *["| " + " | ".join(row) + " |" for row in rows],
    ]


def pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def f3(value: float) -> str:
    return f"{value:.3f}"


def aggregate_group_metrics(
    prediction_rows: list[dict[str, Any]],
    index_values: np.ndarray,
    selected_value: str,
) -> dict[str, Any]:
    rows = []
    for row in prediction_rows:
        with np.load(row["raw_prediction_path"], allow_pickle=False) as arrays:
            index = arrays["test_index"].astype(np.int64)
            selected = index_values[index] == selected_value
            if not np.any(selected):
                continue
            rows.append(
                binary_metrics(
                    arrays["test_calibrated_probability"][selected],
                    arrays["test_label"][selected],
                    float(arrays["threshold"][0]),
                )
            )
    return {
        "evaluations": len(rows),
        "samples_per_evaluation": sorted({row["samples"] for row in rows}),
        **{
            key: summary(row[key] for row in rows)
            for key in (
                "balanced_accuracy",
                "AUROC",
                "AUPRC",
                "Brier",
                "ECE",
                "false_accept_rate",
                "false_reject_rate",
            )
        },
    }


def main() -> None:
    for name in INPUT_FILES:
        if not (OUT / name).is_file():
            raise FileNotFoundError(OUT / name)

    decision = read_json(OUT / "phase_v10_decision.json")
    independent = read_json(OUT / "independent_metric_recomputation.json")
    if decision["decision"] != "VSA_NO_LEARNABLE_IID_TASK":
        raise RuntimeError("V11 generator is frozen to VSA_NO_LEARNABLE_IID_TASK")
    if decision["mitigation_authorized"] or not decision["must_stop_before_mitigation"]:
        raise RuntimeError("V10 did not forbid mitigation as expected")
    if not independent["pass"]:
        raise RuntimeError("independent metric recomputation did not pass")

    verifier = read_json(OUT / "verifier_per_environment.json")
    verifier_rows = verifier["rows"]
    controls = read_json(OUT / "shortcut_control_per_environment.json")
    control_rows = controls["rows"]
    reliance = read_json(OUT / "shortcut_reliance_summary.json")
    matching_gap = read_json(OUT / "matching_gap_report.json")
    matching_quality = read_json(OUT / "matching_quality_report.json")
    nuisance = read_json(OUT / "nuisance_probe_report.json")
    representation = read_json(OUT / "representation_probe_summary.json")
    relational = read_json(OUT / "relational_gain_report.json")
    transfer = read_json(OUT / "generator_transfer_matrix.json")
    calibration = read_json(OUT / "calibration_shift_report.json")
    manifest = read_jsonl(OUT / "unified_audit_manifest.jsonl")
    manifest.sort(key=lambda row: row["unified_array_index"])
    if [row["unified_array_index"] for row in manifest] != list(range(len(manifest))):
        raise RuntimeError("unified manifest indices are not contiguous")
    index_family = np.asarray(
        [row["family"].split("_", 1)[0] for row in manifest], dtype="U2"
    )
    index_task = np.asarray([row["task"] for row in manifest], dtype="U128")

    evidence = {
        name: {"path": str(OUT / name), "sha256": sha256_file(OUT / name)}
        for name in INPUT_FILES
    }

    pooled_rows = [
        row
        for row in verifier_rows
        if row["environment"] == "pooled_reference"
        and row["condition"] == "default_test"
    ]
    if len(pooled_rows) != 15:
        raise RuntimeError(f"expected 15 pooled verifier rows, found {len(pooled_rows)}")

    pooled_family: dict[str, dict[str, Any]] = {}
    pooled_task: dict[str, dict[str, Any]] = {}
    for family in FAMILY_NAMES:
        if family in {"N7", "N8"}:
            continue
        pooled_family[family] = aggregate_group_metrics(
            pooled_rows, index_family, family
        )
    for task in sorted(set(index_task)):
        pooled_task[task] = aggregate_group_metrics(pooled_rows, index_task, task)

    negative_score_by_family: dict[str, list[float]] = defaultdict(list)
    score_range_by_run: list[float] = []
    for row in pooled_rows:
        with np.load(row["raw_prediction_path"], allow_pickle=False) as arrays:
            index = arrays["test_index"].astype(np.int64)
            labels = arrays["test_label"]
            probability = arrays["test_calibrated_probability"]
            run_means = []
            for family in ("N1", "N2", "N3", "N4", "N5", "N6"):
                selected = (index_family[index] == family) & (labels == 0)
                family_mean = float(np.mean(probability[selected]))
                negative_score_by_family[family].append(family_mean)
                run_means.append(family_mean)
            score_range_by_run.append(max(run_means) - min(run_means))

    match_records = {
        (row["environment"], row["family"].split("_", 1)[0]): row
        for row in matching_quality["matching_records"]
    }
    family_rows: list[dict[str, Any]] = []
    for family in ("N1", "N2", "N3", "N4", "N5", "N6"):
        spec = f"within_{family.lower()}"
        fair_rows = [
            row
            for row in verifier_rows
            if row["environment"] == "E1" and row["training_spec"] == spec
        ]
        family_controls = [
            row
            for row in control_rows
            if row["environment"] == "E1" and row["training_spec"] == spec
        ]
        if len(fair_rows) != 15 or len(family_controls) != 6:
            raise RuntimeError(f"incomplete E1 rows for {family}")
        best_control = max(
            family_controls, key=lambda row: row["metrics"]["balanced_accuracy"]
        )
        action_record = match_records[("E5_action_summary", family)]
        action_gap_rows = [
            row
            for row in matching_gap["rows"]
            if row["environment"] == "E5"
            and row["grouping"] == "family"
            and row["group"] == family
        ]
        progress_gap_rows = [
            row
            for row in matching_gap["rows"]
            if row["environment"] == "E6"
            and row["grouping"] == "family"
            and row["group"] == family
        ]
        generator_rows = [
            row
            for row in reliance["generator_and_easy_to_real"]["generator_rows"]
            if row["heldout_family"] == family
        ]
        family_rows.append(
            {
                "family": family,
                "name": FAMILY_NAMES[family],
                "frozen_group": FAMILY_GROUPS[family],
                "empirical_rank_criterion": "mean within-family E1 balanced accuracy over 5 architectures x 3 seeds",
                "within_family_fair_verifier": {
                    "balanced_accuracy": metric_summary(
                        fair_rows, "balanced_accuracy"
                    ),
                    "false_accept_rate": metric_summary(
                        fair_rows, "false_accept_rate"
                    ),
                    "false_reject_rate": metric_summary(
                        fair_rows, "false_reject_rate"
                    ),
                },
                "pooled_verifier": pooled_family[family],
                "best_audit_only_control": {
                    "control": best_control["control"],
                    "balanced_accuracy": best_control["metrics"][
                        "balanced_accuracy"
                    ],
                },
                "all_audit_only_controls": {
                    row["control"]: row["metrics"]["balanced_accuracy"]
                    for row in sorted(
                        family_controls, key=lambda item: item["control"]
                    )
                },
                "action_summary_construction_association": {
                    "top_features_before_matching": action_record[
                        "standardized_mean_difference_before"
                    ]["features"][:5],
                    "mean_absolute_SMD_before": action_record[
                        "standardized_mean_difference_before"
                    ]["mean_absolute"],
                    "maximum_absolute_SMD_before": action_record[
                        "standardized_mean_difference_before"
                    ]["maximum_absolute"],
                    "matching_coverage": action_record["coverage"],
                    "mean_fair_verifier_BA_drop_after_matching": mean(
                        row["action_matching_gap"] for row in action_gap_rows
                    ),
                },
                "progress_association": {
                    "construction_SMD_before_matching": match_records[
                        ("E6_progress", family)
                    ]["standardized_mean_difference_before"][
                        "maximum_absolute"
                    ],
                    "mean_fair_verifier_BA_drop_after_matching": mean(
                        row["progress_matching_gap"] for row in progress_gap_rows
                    ),
                },
                "generator_heldout": {
                    "available": bool(generator_rows),
                    "IID_balanced_accuracy": optional_summary(
                        row["IID_balanced_accuracy"] for row in generator_rows
                    ),
                    "heldout_balanced_accuracy": optional_summary(
                        row["generator_heldout_balanced_accuracy"]
                        for row in generator_rows
                    ),
                    "transfer_gap": optional_summary(
                        row["generator_transfer_gap"] for row in generator_rows
                    ),
                },
                "pooled_negative_score": summary(
                    negative_score_by_family[family]
                ),
            }
        )
    family_rows.sort(
        key=lambda row: row["within_family_fair_verifier"][
            "balanced_accuracy"
        ]["mean"],
        reverse=True,
    )
    for rank, row in enumerate(family_rows, 1):
        row["empirical_easiness_rank"] = rank

    reciprocal_rows = []
    for family in ("N7", "N8"):
        rows = [
            row
            for row in verifier_rows
            if row["environment"] == "E9" and row["condition"] == family
        ]
        relation_rows = [
            row for row in relational["reciprocal_rows"] if row["family"] == family
        ]
        reciprocal_rows.append(
            {
                "family": family,
                "name": FAMILY_NAMES[family],
                "frozen_group": "B",
                "diagnostic_only": True,
                "behaviorally_ambiguous": True,
                "balanced_accuracy": metric_summary(rows, "balanced_accuracy"),
                "false_accept_rate_under_construction_labels": metric_summary(
                    rows, "false_accept_rate"
                ),
                "relational_gain": summary(
                    row["relational_gain"] for row in relation_rows
                ),
                "strict_two_direction_pair_consistency": summary(
                    row["strict_two_direction_pair_consistency"]
                    for row in relation_rows
                ),
            }
        )

    task_fold_rows = [row for row in verifier_rows if row["environment"] == "E8"]
    task_metric_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in task_fold_rows:
        with np.load(row["raw_prediction_path"], allow_pickle=False) as arrays:
            index = arrays["test_index"].astype(np.int64)
            tasks = index_task[index]
            for task in sorted(set(tasks)):
                selected = tasks == task
                task_metric_rows[task].append(
                    binary_metrics(
                        arrays["test_calibrated_probability"][selected],
                        arrays["test_label"][selected],
                        float(arrays["threshold"][0]),
                    )
                )
    generator_task = reliance["generator_and_easy_to_real"]["generator_task_rows"]
    easy_real_task = reliance["generator_and_easy_to_real"][
        "easy_to_real_task_rows"
    ]
    task_rows = []
    for task in sorted(task_metric_rows):
        heldout = task_metric_rows[task]
        generator = [row for row in generator_task if row["task"] == task]
        easy_real = [row for row in easy_real_task if row["task"] == task]
        task_rows.append(
            {
                "task": task,
                "task_heldout_evaluations": len(heldout),
                "task_heldout_balanced_accuracy": summary(
                    row["balanced_accuracy"] for row in heldout
                ),
                "task_heldout_false_accept_rate": summary(
                    row["false_accept_rate"] for row in heldout
                ),
                "pooled_episode_split_balanced_accuracy": pooled_task[task][
                    "balanced_accuracy"
                ],
                "pooled_episode_split_false_accept_rate": pooled_task[task][
                    "false_accept_rate"
                ],
                "generator_heldout_gap": summary(
                    row["generator_transfer_gap"] for row in generator
                ),
                "easy_to_real_gap": summary(
                    row["easy_to_real_gap"] for row in easy_real
                ),
            }
        )

    nuisance_arch = nuisance["aggregates"]["by_architecture_and_feature_set"]
    rep_arch = representation["architecture_target_summary"]
    language_rows = reliance["language_reliance"]["architecture_aggregate"]
    visual_rows = []
    for architecture in ARCHITECTURES:
        conditions = {}
        for condition in (
            "full",
            "static_border_only",
            "center_workspace_only",
            "global_statistics_only",
            "current_only",
            "difference_only",
        ):
            rows = [
                row
                for row in verifier_rows
                if row["environment"] == "E11"
                and row["architecture"] == architecture
                and row["condition"] == condition
            ]
            conditions[condition] = {
                "balanced_accuracy": metric_summary(rows, "balanced_accuracy"),
                "Brier": metric_summary(rows, "Brier"),
            }
        full_ba = conditions["full"]["balanced_accuracy"]["mean"]
        border_ba = conditions["static_border_only"]["balanced_accuracy"]["mean"]
        visual_probe = next(
            row
            for row in nuisance_arch
            if row["architecture"] == architecture
            and row["feature_set"] == "visual_global_statistics"
        )
        visual_rows.append(
            {
                "architecture": architecture,
                "consumes_visual_input": architecture
                in {
                    "VisualStateActionTransformer",
                    "ContrastiveContextActionVerifier",
                    "CrossAttentionEnergyVerifier",
                },
                "conditions": conditions,
                "static_border_BA_drop_from_full": full_ba - border_ba,
                "static_border_fraction_of_above_chance_BA_retained": (
                    (border_ba - 0.5) / (full_ba - 0.5)
                    if full_ba > 0.5
                    else None
                ),
                "visual_global_statistics_score_probe_heldout_R2": visual_probe[
                    "heldout_score_R2"
                ],
            }
        )

    progress_rows = []
    for architecture in ARCHITECTURES:
        probe = next(
            row
            for row in nuisance_arch
            if row["architecture"] == architecture
            and row["feature_set"] == "progress"
        )
        encoded = next(
            row
            for row in rep_arch
            if row["architecture"] == architecture
            and row["target"] == "progress_bin"
        )
        matched = next(
            row
            for row in matching_gap["architecture_aggregate"]
            if row["architecture"] == architecture and row["environment"] == "E6"
        )
        progress_rows.append(
            {
                "architecture": architecture,
                "representation_progress_bin_BA_mean": encoded[
                    "test_balanced_accuracy_mean"
                ],
                "progress_score_probe_heldout_R2_mean": probe[
                    "heldout_score_R2"
                ]["mean"],
                "progress_matched_BA_drop_mean": matched[
                    "progress_matching_gap_mean"
                ],
            }
        )

    relation_rows = relational["architecture_aggregate"]
    cross_relation = next(
        row
        for row in relation_rows
        if row["architecture"] == "CrossAttentionEnergyVerifier"
    )
    cross_attention_comparison = {
        "cross_attention": cross_relation,
        "other_architectures_pooled_gain_mean": mean(
            row["pooled_relational_gain_mean"]
            for row in relation_rows
            if row["architecture"] != "CrossAttentionEnergyVerifier"
        ),
        "other_architectures_reciprocal_gain_mean": mean(
            row["reciprocal_relational_gain_mean"]
            for row in relation_rows
            if row["architecture"] != "CrossAttentionEnergyVerifier"
        ),
        "best_architecture": max(
            relation_rows, key=lambda row: row["reciprocal_relational_gain_mean"]
        )["architecture"],
        "conclusion": "NO_UNIQUE_CROSS_ATTENTION_RELATIONAL_GAIN",
    }

    matrices = transfer["balanced_accuracy_matrices"][
        "all_architectures_and_seeds_mean"
    ]
    diagonal = [
        matrices[f"within_{family.lower()}"][family]
        for family in ("N1", "N2", "N3", "N4", "N5", "N6")
    ]
    off_diagonal = [
        value
        for source, targets in matrices.items()
        if source.startswith("within_")
        for family, value in targets.items()
        if family.lower() != source.removeprefix("within_")
    ]
    negative_family_probe = [
        row for row in rep_arch if row["target"] == "negative_family"
    ]
    family_score_probe = [
        row
        for row in nuisance_arch
        if row["feature_set"] == "family_identity"
    ]
    generator_clustering = {
        "within_generator_diagonal_BA": summary(diagonal),
        "within_generator_off_diagonal_BA": summary(off_diagonal),
        "pooled_negative_score_by_generator": {
            family: summary(values)
            for family, values in sorted(negative_score_by_family.items())
        },
        "mean_within_run_range_of_generator_negative_scores": mean(
            score_range_by_run
        ),
        "negative_family_representation_probe_BA_by_architecture": {
            row["architecture"]: row["test_balanced_accuracy_mean"]
            for row in negative_family_probe
        },
        "negative_family_probe_chance_BA": 1.0 / 6.0,
        "family_identity_score_probe_R2_by_architecture": {
            row["architecture"]: row["heldout_score_R2"]["mean"]
            for row in family_score_probe
        },
        "family_identity_score_probe_gate_R2": nuisance["aggregates"][
            "gate_threshold_R2"
        ],
        "interpretation": (
            "Generator-specific score/performance clustering is descriptive and "
            "strongest for N4, but family-identity score probes remain below the "
            "pre-registered R2 gate and are not causal proof."
        ),
    }

    calibration_rows = []
    for setting in ("generator_heldout", "task_heldout", "source_heldout"):
        rows = [row for row in calibration["rows"] if row["setting"] == setting]
        calibration_rows.append(
            {
                "setting": setting,
                "IID_ECE": summary(row["IID_ECE"] for row in rows),
                "shifted_ECE": summary(row["shifted_ECE"] for row in rows),
                "ECE_shift": summary(row["ECE_shift"] for row in rows),
                "IID_Brier": summary(row["IID_Brier"] for row in rows),
                "shifted_Brier": summary(row["shifted_Brier"] for row in rows),
                "Brier_shift": summary(row["Brier_shift"] for row in rows),
            }
        )

    task_representation = {
        row["architecture"]: row["test_balanced_accuracy_mean"]
        for row in rep_arch
        if row["target"] == "task_identity"
    }
    mitigation = {
        "authorized": False,
        "run": False,
        "reason": decision["decision"],
        "baseline_task_identity_probe_BA": task_representation,
        "post_mitigation_task_identity_probe_BA": None,
        "task_information_removed": "NOT_ASSESSABLE",
        "interpretation": (
            "V10 stopped the study before M0--M4. Baseline representations do "
            "encode task identity, but no post-mitigation representation exists; "
            "therefore no claim about retaining or removing useful task information "
            "is permitted."
        ),
    }

    family_by_id = {row["family"]: row for row in family_rows}
    easy_group = ("N2", "N3", "N5", "N6")
    r_group = ("N1", "N4")
    pooled_easy_far = mean(
        family_by_id[family]["pooled_verifier"]["false_accept_rate"]["mean"]
        for family in easy_group
    )
    pooled_r_far = mean(
        family_by_id[family]["pooled_verifier"]["false_accept_rate"]["mean"]
        for family in r_group
    )
    false_accept_conclusion = {
        "N1_N4_mean_pooled_false_accept_rate": pooled_r_far,
        "N2_N3_N5_N6_mean_pooled_false_accept_rate": pooled_easy_far,
        "difference_N1_N4_minus_other_families": pooled_r_far - pooled_easy_far,
        "N1_pooled_false_accept_rate": family_by_id["N1"]["pooled_verifier"][
            "false_accept_rate"
        ]["mean"],
        "N4_pooled_false_accept_rate": family_by_id["N4"]["pooled_verifier"][
            "false_accept_rate"
        ]["mean"],
        "conclusion": "NO_N1_N4_FALSE_ACCEPT_INCREASE",
        "explanation": (
            "The pooled N1/N4 mean is lower, not higher. N1 remains high in "
            "absolute terms, while N4 is the lowest-FAR and easiest learned family. "
            "All rates refer to accepting construction negatives, not physical failures."
        ),
    }

    shortcut_findings = {
        "language": {
            "rows": language_rows,
            "conclusion": "BEHAVIORALLY_LARGELY_IGNORED",
            "basis": (
                "Across language-consuming architectures, mean absolute BA changes "
                "under shuffled/empty/task-ID-only language are at most 0.007, even "
                "though representations encode task identity."
            ),
        },
        "visual_static_background": {
            "rows": visual_rows,
            "conclusion": "WEAK_ABLATION_SIGNAL_NOT_GATE_LEVEL_SCORE_RELIANCE",
            "basis": (
                "Static-border input retains some above-chance BA in two visual "
                "models, but visual-statistic score-probe held-out R2 is below zero "
                "on average for every architecture and no nuisance-score gate passes."
            ),
        },
        "progress": {
            "rows": progress_rows,
            "conclusion": "ENCODED_BUT_NOT_SUPPORTED_AS_SCORE_RELIANCE",
            "basis": (
                "StateActionTCN progress-bin representation BA is about 0.596, "
                "but its progress score-probe held-out R2 is about -0.003 and E6 "
                "matching changes BA by only about 0.002."
            ),
        },
        "relational": {
            "rows": relation_rows,
            "cross_attention_comparison": cross_attention_comparison,
        },
        "generator_clustering": generator_clustering,
        "calibration_shift": {
            "rows": calibration_rows,
            "conclusion": "NO_ECE_COLLAPSE_BUT_BRIER_DEGRADATION",
            "basis": (
                "Mean ECE rises only modestly under generator/task shift, while "
                "Brier worsens. Near-chance shifted predictions can remain apparently "
                "calibrated, so stable ECE does not establish transfer."
            ),
            "source_heldout_caveat": (
                "E7 reuses the imported source-episode split; only one policy and "
                "camera are available and scene identity is unavailable, so its zero "
                "shift is not a new policy/camera/scene generalization result."
            ),
        },
    }

    taxonomy_entries = [
        {
            "id": "F01",
            "name": "fair-verifier IID non-learnability on the frozen E group",
            "severity": "terminal",
            "evidence": {
                family: family_by_id[family]["within_family_fair_verifier"][
                    "balanced_accuracy"
                ]["mean"]
                for family in easy_group
            },
            "interpretation": (
                "N2/N3/N5/N6 are easy for hand-engineered audit controls but remain "
                "at chance for all five fair-verifier architectures. This blocks the "
                "systematic-shortcut claim about learned verifiers."
            ),
        },
        {
            "id": "F02",
            "name": "generator-specific discrimination dominated by N4",
            "severity": "major",
            "evidence": generator_clustering,
            "interpretation": (
                "The learned signal clusters by generator construction rather than "
                "forming a generator-invariant verifier."
            ),
        },
        {
            "id": "F03",
            "name": "high construction-negative acceptance on unlearned families",
            "severity": "major",
            "evidence": false_accept_conclusion,
            "interpretation": (
                "The issue is broad failure on N2/N3/N5/N6, not a selective N1/N4 "
                "false-accept increase."
            ),
        },
        {
            "id": "F04",
            "name": "language changes scores slightly but not decisions",
            "severity": "diagnostic",
            "evidence": shortcut_findings["language"],
            "interpretation": "Language is behaviorally largely ignored.",
        },
        {
            "id": "F05",
            "name": "static visual statistics retain weak above-chance signal",
            "severity": "diagnostic",
            "evidence": shortcut_findings["visual_static_background"],
            "interpretation": (
                "This is weak associational ablation evidence, not a passed shortcut gate."
            ),
        },
        {
            "id": "F06",
            "name": "progress is encoded but does not explain verifier score",
            "severity": "diagnostic",
            "evidence": shortcut_findings["progress"],
            "interpretation": (
                "Do not equate a linearly decodable representation with score reliance."
            ),
        },
        {
            "id": "F07",
            "name": "cross-attention adds no unique relational gain",
            "severity": "diagnostic",
            "evidence": shortcut_findings["relational"],
            "interpretation": (
                "The contrastive verifier, not cross-attention, has the largest "
                "reciprocal diagnostic gain."
            ),
        },
        {
            "id": "F08",
            "name": "shift degrades Brier without an ECE collapse",
            "severity": "diagnostic",
            "evidence": shortcut_findings["calibration_shift"],
            "interpretation": (
                "Stable low ECE near chance is not evidence of useful transfer."
            ),
        },
        {
            "id": "F09",
            "name": "reciprocal sets are behaviorally ambiguous",
            "severity": "claim-boundary",
            "evidence": reciprocal_rows,
            "interpretation": (
                "N7/N8 support consistency diagnostics only and cannot validate "
                "physical failure detection."
            ),
        },
        {
            "id": "F10",
            "name": "mitigation task-information effect is unavailable",
            "severity": "not-run",
            "evidence": mitigation,
            "interpretation": (
                "Mitigation was correctly not run after the V10 terminal decision."
            ),
        },
    ]

    payload = {
        "schema": "vsa-failure-taxonomy-v1",
        "study_id": STUDY_ID,
        "phase": "V11",
        "decision": decision["decision"],
        "systematic_shortcut_supported": False,
        "mitigation_authorized": False,
        "independent_metric_recomputation_passed": True,
        "construction_labels_are_physical_truth": False,
        "claim_boundary": (
            "All positives and negatives are historical construction labels. "
            "No balanced accuracy, false-accept rate, reciprocal score, or taxonomy "
            "entry establishes physical validity, physical failure, success, or safety."
        ),
        "reciprocal_boundary": (
            "N7/N8 are behaviorally ambiguous reciprocal diagnostics only."
        ),
        "source_evidence": evidence,
        "family_analysis": family_rows,
        "reciprocal_family_analysis": reciprocal_rows,
        "task_analysis": task_rows,
        "shortcut_findings": shortcut_findings,
        "false_accept_conclusion": false_accept_conclusion,
        "mitigation_task_information_analysis": mitigation,
        "taxonomy": taxonomy_entries,
    }
    write_json(OUT / "failure_taxonomy.json", payload)

    # failure_taxonomy.md
    lines = [
        "# V11 Failure Taxonomy",
        "",
        f"Terminal decision: **{decision['decision']}**. The V10 gate found no "
        "learnable IID task on the pre-registered E families, so mitigation was "
        "not authorized and was not run.",
        "",
        "## Claim boundary",
        "",
        payload["claim_boundary"],
        "",
        "N7 and N8 are behaviorally ambiguous reciprocal sets. Their labels and "
        '“false accepts” are diagnostic construction labels, not physical truth.',
        "",
        "## Taxonomy",
        "",
    ]
    for entry in taxonomy_entries:
        lines.extend(
            [
                f"### {entry['id']} — {entry['name']}",
                "",
                f"Severity: `{entry['severity']}`. {entry['interpretation']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Gate-level conclusion",
            "",
            "Hand-engineered controls expose construction shortcuts, and all five "
            "architectures show generator-heldout degradation. However, the learned "
            "fair verifiers cannot learn N2/N3/N5/N6 even IID (all family means "
            "0.500–0.503 BA). The correct conclusion is therefore not that successful "
            "verifiers systematically exploit shortcuts; it is that the proposed "
            "learned-verifier task is not established on the frozen benchmark.",
            "",
            "Mitigation task-information removal is `NOT_ASSESSABLE`: baseline task "
            "identity is decodable (architecture means 0.758–0.977 BA), but there is "
            "no authorized post-mitigation representation.",
            "",
        ]
    )
    (OUT / "failure_taxonomy.md").write_text("\n".join(lines), encoding="utf-8")

    # per_family_analysis.md
    lines = [
        "# V11 Per-Family Analysis",
        "",
        "Easiness is ranked by mean E1 within-family balanced accuracy over five "
        "architectures and three seeds. The frozen group name `E` means a "
        "synthetic audit construction; it does not guarantee empirical learnability.",
        "",
    ]
    table_rows = []
    for row in family_rows:
        ba = row["within_family_fair_verifier"]["balanced_accuracy"]["mean"]
        far = row["pooled_verifier"]["false_accept_rate"]["mean"]
        best = row["best_audit_only_control"]
        assoc = row["action_summary_construction_association"]
        top = assoc["top_features_before_matching"][0]
        table_rows.append(
            [
                str(row["empirical_easiness_rank"]),
                row["family"],
                row["frozen_group"],
                f3(ba),
                f3(far),
                f"{CONTROL_LABELS[best['control']]} ({f3(best['balanced_accuracy'])})",
                f"{top['feature']} ({f3(top['absolute_standardized_mean_difference'])})",
                f3(assoc["mean_fair_verifier_BA_drop_after_matching"]),
            ]
        )
    lines.extend(
        markdown_table(
            [
                "Rank",
                "Family",
                "Group",
                "E1 fair BA",
                "Pooled FAR",
                "Best audit-only control BA",
                "Largest pre-match action SMD",
                "E5 BA drop",
            ],
            table_rows,
        )
    )
    lines.extend(["", "## Findings by family", ""])
    for row in sorted(family_rows, key=lambda item: item["family"]):
        family = row["family"]
        assoc = row["action_summary_construction_association"]
        top = ", ".join(
            f"{feature['feature']}={feature['absolute_standardized_mean_difference']:.3f}"
            for feature in assoc["top_features_before_matching"][:3]
        )
        lines.extend(
            [
                f"### {family} — {row['name']}",
                "",
                f"Fair-verifier E1 BA is {row['within_family_fair_verifier']['balanced_accuracy']['mean']:.3f} "
                f"± {row['within_family_fair_verifier']['balanced_accuracy']['standard_deviation']:.3f}; "
                f"the best audit-only control is {CONTROL_LABELS[row['best_audit_only_control']['control']]} "
                f"at {row['best_audit_only_control']['balanced_accuracy']:.3f} BA. "
                f"Leading construction-associated action summaries are {top}. "
                f"After E5 matching, fair-verifier BA changes by only "
                f"{assoc['mean_fair_verifier_BA_drop_after_matching']:.3f} on average.",
                "",
            ]
        )
    lines.extend(
        [
            "N4 is empirically easiest (0.827 mean BA), followed by N1 (0.552). "
            "N2/N3/N5/N6 remain essentially chance (0.500–0.503), despite audit-only "
            "controls reaching 0.823–0.885 BA. This mismatch is the V10 terminal issue.",
            "",
            "## Reciprocal diagnostic families",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            ["Family", "Diagnostic BA", "Construction-label FAR", "Relational gain", "Pair consistency"],
            [
                [
                    row["family"],
                    f3(row["balanced_accuracy"]["mean"]),
                    f3(row["false_accept_rate_under_construction_labels"]["mean"]),
                    f3(row["relational_gain"]["mean"]),
                    f3(row["strict_two_direction_pair_consistency"]["mean"]),
                ]
                for row in reciprocal_rows
            ],
        )
    )
    lines.extend(
        [
            "",
            "N7/N8 are behaviorally ambiguous by construction and are not physical "
            "failure labels. Their numbers are consistency diagnostics only.",
            "",
        ]
    )
    (OUT / "per_family_analysis.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )

    # per_task_analysis.md
    lines = [
        "# V11 Per-Task Analysis",
        "",
        "Each task-heldout value averages 15 evaluations (five architectures × "
        "three frozen seeds). Generator gaps average the available family-heldout "
        "evaluations. All labels remain construction labels.",
        "",
    ]
    lines.extend(
        markdown_table(
            [
                "Task",
                "Task-heldout BA",
                "Task-heldout FAR",
                "Episode-split BA",
                "Generator gap",
                "E→R gap",
            ],
            [
                [
                    row["task"].removeprefix("robogene_twoArm_franka_"),
                    f3(row["task_heldout_balanced_accuracy"]["mean"]),
                    f3(row["task_heldout_false_accept_rate"]["mean"]),
                    f3(
                        row["pooled_episode_split_balanced_accuracy"]["mean"]
                    ),
                    f3(row["generator_heldout_gap"]["mean"]),
                    f3(row["easy_to_real_gap"]["mean"]),
                ]
                for row in task_rows
            ],
        )
    )
    task_bas = [row["task_heldout_balanced_accuracy"]["mean"] for row in task_rows]
    generator_gaps = [row["generator_heldout_gap"]["mean"] for row in task_rows]
    lines.extend(
        [
            "",
            f"Task-heldout BA ranges from {min(task_bas):.3f} to {max(task_bas):.3f}; "
            f"the mean across tasks is {mean(task_bas):.3f}. Generator-heldout gaps "
            f"range from {min(generator_gaps):.3f} to {max(generator_gaps):.3f}. "
            "Thus generator-specific evidence spans tasks, but task-heldout verifier "
            "performance itself remains near chance.",
            "",
            "Baseline task identity is nevertheless linearly decodable from every "
            "architecture’s representation:",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            ["Architecture", "Task-identity probe BA"],
            [[architecture, f3(value)] for architecture, value in task_representation.items()],
        )
    )
    lines.extend(
        [
            "",
            "There is no post-mitigation task probe. V10 forbade mitigation, so whether "
            "mitigation removes useful task information is `NOT_ASSESSABLE`.",
            "",
        ]
    )
    (OUT / "per_task_analysis.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )

    # per_shortcut_analysis.md
    lines = [
        "# V11 Per-Shortcut Analysis",
        "",
        "## Language",
        "",
    ]
    lines.extend(
        markdown_table(
            [
                "Architecture",
                "Variant",
                "Correct−variant BA",
                r"Mean \|score Δ\|",
                "Rank correlation",
            ],
            [
                [
                    row["architecture"],
                    row["condition"],
                    f3(row["balanced_accuracy_delta_mean"]),
                    f3(row["mean_absolute_score_delta"]),
                    f3(row["rank_correlation_mean"]),
                ]
                for row in language_rows
            ],
        )
    )
    lines.extend(
        [
            "",
            "Language is behaviorally largely ignored: the largest mean absolute BA "
            "change is 0.007. Score rankings move somewhat, but decisions do not "
            "depend materially on correct language.",
            "",
            "## Visual/static background",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            ["Architecture", "Uses visual", "Full BA", "Border-only BA", "BA drop", "Visual-score probe R²"],
            [
                [
                    row["architecture"],
                    "yes" if row["consumes_visual_input"] else "no",
                    f3(row["conditions"]["full"]["balanced_accuracy"]["mean"]),
                    f3(
                        row["conditions"]["static_border_only"][
                            "balanced_accuracy"
                        ]["mean"]
                    ),
                    f3(row["static_border_BA_drop_from_full"]),
                    f3(
                        row["visual_global_statistics_score_probe_heldout_R2"][
                            "mean"
                        ]
                    ),
                ]
                for row in visual_rows
            ],
        )
    )
    lines.extend(
        [
            "",
            "The explicit visual transformer retains 0.532 BA from static border only "
            "versus 0.562 with full visual history; the contrastive verifier retains "
            "0.556 versus 0.572. Cross-attention falls to 0.503. Because held-out "
            "visual-statistics score-probe R² is negative for every architecture and "
            "no nuisance-score gate passes, this is weak ablation evidence—not a "
            "systematic static-background shortcut finding.",
            "",
            "## State/action progress",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            ["Architecture", "Progress representation BA", "Progress score-probe R²", "E6 BA drop"],
            [
                [
                    row["architecture"],
                    f3(row["representation_progress_bin_BA_mean"]),
                    f3(row["progress_score_probe_heldout_R2_mean"]),
                    f3(row["progress_matched_BA_drop_mean"]),
                ]
                for row in progress_rows
            ],
        )
    )
    lines.extend(
        [
            "",
            "StateActionTCN encodes progress (0.596 probe BA), but progress does not "
            "explain its held-out score (R² −0.003) and E6 matching changes BA by "
            "0.002. Encoding is not evidence of decision reliance.",
            "",
            "## Relational gain",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            ["Architecture", "Pooled relational gain", "Reciprocal relational gain"],
            [
                [
                    row["architecture"],
                    f3(row["pooled_relational_gain_mean"]),
                    f3(row["reciprocal_relational_gain_mean"]),
                ]
                for row in relation_rows
            ],
        )
    )
    lines.extend(
        [
            "",
            "Cross-attention does not uniquely improve relational gain: it reaches "
            f"{cross_relation['pooled_relational_gain_mean']:.3f} pooled and "
            f"{cross_relation['reciprocal_relational_gain_mean']:.3f} reciprocal, "
            "while the contrastive verifier is best on the reciprocal diagnostic "
            f"({max(row['reciprocal_relational_gain_mean'] for row in relation_rows):.3f}).",
            "",
            "## Generator clustering",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            ["Generator", "Mean calibrated score on construction negatives"],
            [
                [
                    family,
                    f3(generator_clustering["pooled_negative_score_by_generator"][family]["mean"]),
                ]
                for family in ("N1", "N2", "N3", "N4", "N5", "N6")
            ],
        )
    )
    lines.extend(
        [
            "",
            f"The within-run family score range averages "
            f"{generator_clustering['mean_within_run_range_of_generator_negative_scores']:.3f}; "
            "N4 negatives score 0.246 on average versus 0.513–0.517 for N2/N3/N5/N6. "
            "Negative-family representation probes reach 0.261–0.298 BA versus "
            "1/6 chance. Generator clustering is therefore descriptive, especially "
            "for N4, but the family-identity verifier-score R² (architecture means "
            "0.158–0.184) remains below the pre-registered 0.30 gate.",
            "",
            "## Calibration shift",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            ["Shift", "IID ECE", "Shifted ECE", "ΔECE", "ΔBrier"],
            [
                [
                    row["setting"],
                    f3(row["IID_ECE"]["mean"]),
                    f3(row["shifted_ECE"]["mean"]),
                    f3(row["ECE_shift"]["mean"]),
                    f3(row["Brier_shift"]["mean"]),
                ]
                for row in calibration_rows
            ],
        )
    )
    lines.extend(
        [
            "",
            "Calibration does not collapse by ECE: generator-heldout ΔECE is 0.006 "
            "and task-heldout ΔECE is 0.002. Brier worsens by 0.015 and 0.013, "
            "respectively. Near-chance predictions can remain apparently calibrated, "
            "so this is not evidence of useful transfer. E7’s zero shift is expected "
            "because it reuses the imported source-episode split and has no independent "
            "policy/camera/scene diversity.",
            "",
        ]
    )
    (OUT / "per_shortcut_analysis.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )

    # false_accept_analysis.md
    lines = [
        "# V11 False-Accept Analysis",
        "",
        "A “false accept” here means accepting a historical construction negative. "
        "It does **not** mean accepting an action proven to fail physically.",
        "",
    ]
    lines.extend(
        markdown_table(
            ["Family", "Pooled BA", "Pooled FAR", "Within-family FAR", "Mean negative score"],
            [
                [
                    family,
                    f3(family_by_id[family]["pooled_verifier"]["balanced_accuracy"]["mean"]),
                    f3(family_by_id[family]["pooled_verifier"]["false_accept_rate"]["mean"]),
                    f3(
                        family_by_id[family]["within_family_fair_verifier"][
                            "false_accept_rate"
                        ]["mean"]
                    ),
                    f3(family_by_id[family]["pooled_negative_score"]["mean"]),
                ]
                for family in ("N1", "N2", "N3", "N4", "N5", "N6")
            ],
        )
    )
    lines.extend(
        [
            "",
            f"N1/N4 do not show an aggregate FAR increase. Their pooled mean FAR is "
            f"{pooled_r_far:.3f}, versus {pooled_easy_far:.3f} for N2/N3/N5/N6 "
            f"(difference {pooled_r_far - pooled_easy_far:+.3f}). N1 is high in "
            f"absolute terms ({false_accept_conclusion['N1_pooled_false_accept_rate']:.3f}), "
            f"but N4 is much lower ({false_accept_conclusion['N4_pooled_false_accept_rate']:.3f}) "
            "because N4 is the only consistently easy learned family. The dominant "
            "false-accept problem is instead N2/N3/N5/N6, where pooled FAR is about "
            "0.81–0.83 and balanced accuracy is at chance.",
            "",
            "For N7/N8, any FAR is a reciprocal construction-label diagnostic only. "
            "Those sets are behaviorally ambiguous, so their FAR cannot be read as "
            "a physical-risk rate.",
            "",
        ]
    )
    (OUT / "false_accept_analysis.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )

    output_names = (
        "failure_taxonomy.json",
        "failure_taxonomy.md",
        "per_family_analysis.md",
        "per_task_analysis.md",
        "per_shortcut_analysis.md",
        "false_accept_analysis.md",
    )
    print(
        json.dumps(
            {
                "phase": "V11",
                "decision": decision["decision"],
                "outputs": {
                    name: {
                        "bytes": (OUT / name).stat().st_size,
                        "sha256": sha256_file(OUT / name),
                    }
                    for name in output_names
                },
                "run_log_modified": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
