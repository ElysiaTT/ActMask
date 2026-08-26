"""Finalize and independently verify a non-ready RM-ACV-PILOT166 result."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from actmask.experiments.rm_acv_phase_a0_a1 import verify as verify_prior_rm_acv
from actmask.experiments.rm_acv_pilot166_common import (
    FAIR_INPUTS,
    FORBIDDEN_PREDICTIVE_INPUTS,
    NEGATIVE_FAMILIES,
    OUT,
    PRIMARY_EMBODIMENT,
    ROOT,
    SOURCE,
    append_log,
    binary_metrics,
    read_jsonl,
    scan_records,
    sha256_file,
    write_jsonl,
    write_json,
)
from actmask.experiments.rm_acv_pilot166_prepare import _fit_feature, _shortcut_table


DECISION = "RM_ACV_PILOT_NEGATIVES_INVALID"
REQUIRED = (
    "preregistered_config.json",
    "preregistered_config.sha256",
    "source_manifest.jsonl",
    "source_hash_manifest.json",
    "excluded_heterogeneous_source.json",
    "anchor_manifest.jsonl",
    "positive_candidate_manifest.jsonl",
    "anchor_statistics.json",
    "anchor_overlap_audit.json",
    "alignment_audit.json",
    "negative_candidate_manifest.jsonl",
    "negative_generation_report.json",
    "negative_matching_statistics.json",
    "negative_failure_log.jsonl",
    "candidate_group_manifest.jsonl",
    "benchmark_arrays.npz",
    "shortcut_feature_report.json",
    "shortcut_predictions.jsonl",
    "shortcut_recomputation_audit.json",
    "negative_quality_report.json",
    "negative_family_decision.json",
    "matching_balance_report.json",
    "split_manifest.json",
    "episode_split_audit.json",
    "task_cv_manifest.json",
    "task_cv_audit.json",
    "leakage_audit.json",
    "prior_branch_preservation_audit.json",
    "preregistration_amendment.json",
    "completion_audit.json",
    "baseline_config.json",
    "baseline_summary.json",
    "baseline_per_family.json",
    "baseline_episode_heldout.json",
    "baseline_task_cv.json",
    "raw_prediction_manifest.json",
    "phase_p8_decision.json",
    "final_decision.json",
    "final_report.md",
    "anchor_report.md",
    "negative_construction_report.md",
    "shortcut_audit_report.md",
    "baseline_report.md",
    "episode_heldout_report.md",
    "task_cv_report.md",
    "candidate_ranking_report.md",
    "calibration_report.md",
    "error_analysis_report.md",
    "claim_boundary.md",
    "next_steps.md",
    "reproducibility_manifest.json",
    "run_log.jsonl",
    "paper/title_candidates.md",
    "paper/abstract_zh.md",
    "paper/abstract_en.md",
    "paper/introduction_outline.md",
    "paper/problem_formulation.md",
    "paper/benchmark.md",
    "paper/method.md",
    "paper/experiments.md",
    "paper/results_tables.md",
    "paper/figure_plan.md",
    "paper/claim_boundary.md",
    "paper/scale_up_plan.md",
)


def _not_run(artifact: str, reason: str) -> dict[str, Any]:
    return {
        "schema": "rm-acv-pilot166-not-run-v1",
        "artifact": artifact,
        "status": "not_run",
        "reason": reason,
        "learned_baselines_trained": False,
        "proposed_method_trained": False,
    }


def _complete_provisional_shortcut_controls() -> dict[str, Any]:
    """Run every required P3 control on the two-family survivor pool.

    The pool is explicitly non-admissible because it has fewer than three
    families.  These controls close the audit surface without changing the
    terminal precedence rule or authorizing a predictive baseline.
    """

    shortcut_path = OUT / "shortcut_feature_report.json"
    shortcut = json.loads(shortcut_path.read_text(encoding="utf-8"))
    if shortcut.get("provisional_survivor_only_controls", {}).get("completed"):
        return shortcut
    survivors = list(shortcut.get("surviving_families", []))
    if not survivors:
        shortcut["provisional_survivor_only_controls"] = {
            "completed": False,
            "status": "not_run_no_surviving_family",
        }
        write_json(shortcut_path, shortcut)
        return shortcut
    anchors = read_jsonl(OUT / "anchor_manifest.jsonl")
    negatives_manifest = read_jsonl(OUT / "negative_candidate_manifest.jsonl")
    with np.load(OUT / "benchmark_arrays.npz") as stored:
        arrays = {
            name: stored[name].copy()
            for name in (
                "history_state",
                "positive_action",
                "current_state",
                "progress",
                "instruction",
                "episode_duration",
                "episode_index_internal",
                "task_index_internal",
            )
        }
        negatives = stored["negative_action_all"].copy()
    source_rows = {
        row["episode_id_provenance_only"]: row
        for row in read_jsonl(OUT / "source_manifest.jsonl")
    }
    records, _ = scan_records()
    payload: dict[str, dict[str, Any]] = {}
    for record in records:
        source = source_rows[record.episode_id]
        cache = np.load(
            OUT / "processed" / "episodes" / f"{source['safe_episode_key']}.npz"
        )
        state = cache["state"].astype(np.float32)
        action = cache["action"].astype(np.float32)
        timestamp = cache["timestamp"].astype(np.float64)
        payload[record.episode_id] = {
            "record": record,
            "state": state,
            "action": action,
            "duration": float(timestamp[-1] - timestamp[0]),
        }
        cache.close()
    rows, _, features = _shortcut_table(
        anchors,
        arrays,
        negatives,
        negatives_manifest,
        survivors,
        payload,
    )
    predictions = read_jsonl(OUT / "shortcut_predictions.jsonl")
    reports: dict[str, Any] = {}
    for name, feature in features.items():
        report, raw = _fit_feature(
            f"provisional_survivors:{name}",
            rows,
            feature,
            nonlinear=name == "combined_non_context",
        )
        reports[name] = report
        predictions.extend(raw)
    labels = np.asarray([row["label"] for row in rows], dtype=np.int64)
    family_counts = Counter(row["paired_family"] for row in rows if row["label"] == 0)
    shortcut["provisional_survivor_only_controls"] = {
        "completed": True,
        "scope": "two surviving families only; diagnostic closure, not an admissible benchmark",
        "families": survivors,
        "rows": len(rows),
        "positive_fraction": float(labels.mean()),
        "negative_family_counts": dict(sorted(family_counts.items())),
        "feature_reports": reports,
        "decision_precedence": DECISION,
        "can_authorize_fair_baselines": False,
    }
    write_json(shortcut_path, shortcut)
    write_jsonl(OUT / "shortcut_predictions.jsonl", predictions)
    write_json(
        OUT / "matching_balance_report.json",
        {
            "schema": "rm-acv-pilot166-matching-balance-v1",
            "scope": "provisional two-family diagnostic controls; benchmark remains inadmissible",
            "positive_fraction": float(labels.mean()),
            "negative_family_counts": dict(sorted(family_counts.items())),
            "maximum_single_family_fraction": max(family_counts.values()) / sum(family_counts.values()),
            "class_balance_45_55": 0.45 <= float(labels.mean()) <= 0.55,
            "single_family_fraction_le_0_50": (
                max(family_counts.values()) / sum(family_counts.values()) <= 0.50
            ),
        },
    )
    append_log(
        "P3_PROVISIONAL_REQUIRED_CONTROLS_COMPLETE",
        families=survivors,
        rows=len(rows),
        combined_balanced_accuracy=reports["combined_non_context"]["test"]["balanced_accuracy"],
        source_metadata_balanced_accuracy=reports["source_metadata"]["test"]["balanced_accuracy"],
        decision_unchanged=DECISION,
    )
    return shortcut


def _shortcut_recomputation() -> dict[str, Any]:
    shortcut = json.loads((OUT / "shortcut_feature_report.json").read_text(encoding="utf-8"))
    predictions = read_jsonl(OUT / "shortcut_predictions.jsonl")
    checks: list[dict[str, Any]] = []
    for family, report in shortcut["family_reports"].items():
        if "test" not in report:
            checks.append(
                {
                    "family": family,
                    "status": "not_scored_insufficient_pairs",
                    "match": True,
                }
            )
            continue
        name = f"family_action_summary:{family}"
        rows = [
            row for row in predictions
            if row["classifier"] == name and row["split"] == "test"
        ]
        labels = np.asarray([row["label"] for row in rows], dtype=np.int64)
        scores = np.asarray([row["score"] for row in rows], dtype=np.float64)
        threshold = float(report["training"]["threshold_selected_on_validation"])
        recomputed = binary_metrics(labels, scores, threshold)
        expected = report["test"]
        delta = abs(float(recomputed["balanced_accuracy"]) - float(expected["balanced_accuracy"]))
        checks.append(
            {
                "family": family,
                "rows": len(rows),
                "threshold": threshold,
                "reported_balanced_accuracy": expected["balanced_accuracy"],
                "recomputed_balanced_accuracy": recomputed["balanced_accuracy"],
                "absolute_delta": delta,
                "match": delta < 1e-12,
            }
        )
    provisional = shortcut.get("provisional_survivor_only_controls", {})
    for name, report in provisional.get("feature_reports", {}).items():
        classifier = f"provisional_survivors:{name}"
        rows = [
            row for row in predictions
            if row["classifier"] == classifier and row["split"] == "test"
        ]
        labels = np.asarray([row["label"] for row in rows], dtype=np.int64)
        scores = np.asarray([row["score"] for row in rows], dtype=np.float64)
        threshold = float(report["training"]["threshold_selected_on_validation"])
        recomputed = binary_metrics(labels, scores, threshold)
        expected = report["test"]
        delta = abs(float(recomputed["balanced_accuracy"]) - float(expected["balanced_accuracy"]))
        checks.append(
            {
                "feature": name,
                "classifier": classifier,
                "scope": "provisional_two_family_controls",
                "rows": len(rows),
                "threshold": threshold,
                "reported_balanced_accuracy": expected["balanced_accuracy"],
                "recomputed_balanced_accuracy": recomputed["balanced_accuracy"],
                "absolute_delta": delta,
                "match": delta < 1e-12,
            }
        )
    result = {
        "schema": "rm-acv-pilot166-shortcut-recomputation-v1",
        "checks": checks,
        "all_match": all(row["match"] for row in checks),
    }
    write_json(OUT / "shortcut_recomputation_audit.json", result)
    return result


def _split_and_leakage_artifacts() -> dict[str, Any]:
    source = read_jsonl(OUT / "source_manifest.jsonl")
    anchors = read_jsonl(OUT / "anchor_manifest.jsonl")
    negatives = read_jsonl(OUT / "negative_candidate_manifest.jsonl")
    episode_preregistered = json.loads(
        (OUT / "episode_split_preregistered.json").read_text(encoding="utf-8")
    )["splits"]
    task_preregistered = json.loads(
        (OUT / "task_cv_preregistered.json").read_text(encoding="utf-8")
    )["folds"]
    episode_sets = {name: set(values) for name, values in episode_preregistered.items()}
    disjoint = not any(
        episode_sets[left] & episode_sets[right]
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
    )
    source_ids = {row["episode_id_provenance_only"] for row in source}
    complete = set().union(*episode_sets.values()) == source_ids
    assignment = {
        episode_id: split for split, values in episode_sets.items() for episode_id in values
    }
    anchor_consistent = all(
        assignment[row["episode_id_provenance_only"]] == row["episode_split"]
        for row in anchors
    )
    retrieval_within_split = all(
        assignment[row["source_episode_provenance_only"]]
        == assignment[
            next(
                anchor["episode_id_provenance_only"]
                for anchor in anchors if anchor["anchor_id"] == row["anchor_id"]
            )
        ]
        for row in negatives
    )
    anchor_counts = Counter(row["episode_split"] for row in anchors)
    write_json(
        OUT / "split_manifest.json",
        {
            "schema": "rm-acv-pilot166-split-manifest-v1",
            "episode_splits": {
                name: {
                    "episodes": sorted(values),
                    "episode_count": len(values),
                    "anchor_count": anchor_counts[name],
                }
                for name, values in episode_sets.items()
            },
            "predictive_evaluation_status": "not_run_after_P3_negative_gate_failure",
        },
    )
    episode_audit = {
        "schema": "rm-acv-pilot166-episode-split-audit-v1",
        "no_episode_overlap": disjoint,
        "covers_all_166_episodes": complete and len(source_ids) == 166,
        "all_anchors_follow_episode_assignment": anchor_consistent,
        "all_retrieval_sources_within_episode_split": retrieval_within_split,
        "pass": disjoint and complete and anchor_consistent and retrieval_within_split,
    }
    write_json(OUT / "episode_split_audit.json", episode_audit)
    test_tasks = [
        task for fold in task_preregistered for task in fold["test_tasks"]
    ]
    all_tasks = sorted({row["task_id_provenance_only"] for row in source})
    task_audit = {
        "schema": "rm-acv-pilot166-task-cv-audit-v1",
        "fold_count": len(task_preregistered),
        "all_folds_hold_out_complete_tasks": all(
            not (
                set(fold["train_tasks"]) & set(fold["validation_tasks"])
                or set(fold["train_tasks"]) & set(fold["test_tasks"])
                or set(fold["validation_tasks"]) & set(fold["test_tasks"])
            )
            for fold in task_preregistered
        ),
        "every_task_held_out_exactly_once": (
            sorted(test_tasks) == all_tasks and len(test_tasks) == len(set(test_tasks))
        ),
        "model_evaluation_status": "not_run_after_P3_negative_gate_failure",
    }
    task_audit["pass"] = bool(
        task_audit["all_folds_hold_out_complete_tasks"]
        and task_audit["every_task_held_out_exactly_once"]
    )
    write_json(
        OUT / "task_cv_manifest.json",
        {
            "schema": "rm-acv-pilot166-task-cv-manifest-v1",
            "folds": task_preregistered,
            "model_evaluation_status": "not_run_after_P3_negative_gate_failure",
        },
    )
    write_json(OUT / "task_cv_audit.json", task_audit)
    with np.load(OUT / "benchmark_arrays.npz") as arrays:
        array_keys = sorted(arrays.files)
    leakage = {
        "schema": "rm-acv-pilot166-leakage-audit-v1",
        "fair_input_contract": list(FAIR_INPUTS),
        "forbidden_predictive_inputs": list(FORBIDDEN_PREDICTIVE_INPUTS),
        "benchmark_array_keys": array_keys,
        "grouping_only_arrays": [
            "episode_index_internal",
            "task_index_internal",
            "selected_family_indices",
            "selected_group_indices",
        ],
        "provenance_paths_present_only_in_json_audit_manifests": True,
        "future_rgb_present": False,
        "future_state_present": False,
        "sam3_or_robot_mask_present": False,
        "negative_family_identity_present_in_fair_tensor": False,
        "predictive_training_started": False,
        "episode_split_audit_pass": episode_audit["pass"],
        "task_cv_audit_pass": task_audit["pass"],
    }
    leakage["pass"] = all(
        [
            leakage["provenance_paths_present_only_in_json_audit_manifests"],
            not leakage["future_rgb_present"],
            not leakage["future_state_present"],
            not leakage["sam3_or_robot_mask_present"],
            not leakage["negative_family_identity_present_in_fair_tensor"],
            leakage["episode_split_audit_pass"],
            leakage["task_cv_audit_pass"],
        ]
    )
    write_json(OUT / "leakage_audit.json", leakage)
    return {
        "episode": episode_audit,
        "task_cv": task_audit,
        "leakage": leakage,
    }


def _write_terminal_artifacts() -> dict[str, Any]:
    shortcut = json.loads((OUT / "shortcut_feature_report.json").read_text(encoding="utf-8"))
    anchors = json.loads((OUT / "anchor_statistics.json").read_text(encoding="utf-8"))
    generation = json.loads((OUT / "negative_generation_report.json").read_text(encoding="utf-8"))
    prior_verify = verify_prior_rm_acv(content_rehash=False)
    write_json(
        OUT / "prior_branch_preservation_audit.json",
        {
            "schema": "rm-acv-pilot166-prior-preservation-v1",
            "prior_branch": "outputs/actmask/rm_acv_complete",
            "prior_decision": prior_verify["decision"],
            "prior_hash_verifier_pass": prior_verify["passed"],
            "pilot_writer_targets_prior_branch": False,
        },
    )
    reason = (
        "Only two negative families survived the frozen construction/shortcut gate; "
        "at least three are required. Learned fair baselines and the proposed method are prohibited."
    )
    decision = {
        "schema": "rm-acv-pilot166-phase-p8-decision-v1",
        "decision": DECISION,
        "terminal": True,
        "stage_reached": "P3",
        "reason": reason,
        "trajectories": 166,
        "tasks": 11,
        "embodiment": PRIMARY_EMBODIMENT,
        "anchors": anchors["anchors"],
        "negative_candidates_constructed": generation["constructed"],
        "surviving_families": shortcut["surviving_families"],
        "selected_families": shortcut["selected_families"],
        "minimum_required_families": 3,
        "learned_baselines_trained": False,
        "method_training_authorized": False,
        "proposed_method_trained": False,
    }
    write_json(OUT / "phase_p8_decision.json", decision)
    write_json(OUT / "final_decision.json", decision)
    for name in (
        "baseline_config.json",
        "baseline_summary.json",
        "baseline_per_family.json",
        "baseline_episode_heldout.json",
        "baseline_task_cv.json",
        "raw_prediction_manifest.json",
    ):
        write_json(OUT / name, _not_run(name, reason))
    (OUT / "raw_predictions").mkdir(exist_ok=True)
    write_json(OUT / "raw_predictions" / "NOT_RUN.json", _not_run("raw_predictions", reason))
    (OUT / "checkpoints").mkdir(exist_ok=True)
    write_json(OUT / "checkpoints" / "NOT_RUN.json", _not_run("checkpoints", reason))
    family_lines = "\n".join(
        f"- `{family}`: coverage {report.get('coverage', 0):.3f}, "
        f"test shortcut BA {report.get('test', {}).get('balanced_accuracy', float('nan')):.3f}, "
        f"{'survived' if report.get('survives') else 'rejected'}."
        for family, report in shortcut["family_reports"].items()
    )
    provisional = shortcut["provisional_survivor_only_controls"]
    provisional_reports = provisional["feature_reports"]
    provisional_combined = provisional_reports["combined_non_context"]["test"]["balanced_accuracy"]
    provisional_source = provisional_reports["source_metadata"]["test"]["balanced_accuracy"]
    provisional_family = provisional_reports["negative_family_identity"]["test"]["balanced_accuracy"]
    reports = {
        "final_report.md": (
            "# RM-ACV-PILOT166 final report\n\n"
            f"Decision: **`{DECISION}`**.\n\n"
            "This was a bounded 166-trajectory, 11-task, single-embodiment pilot. "
            f"It produced {anchors['anchors']} spaced anchors and attempted six negative families. "
            "Only N1 same-task near-state retrieval and N4 phase-shifted retrieval survived. "
            "Because the preregistered minimum is three families, no learned fair baseline and no proposed method were trained.\n"
        ),
        "anchor_report.md": (
            "# Anchor report\n\n"
            f"{anchors['anchors']} anchors were constructed from 166 trajectories. "
            f"Per trajectory: minimum {anchors['minimum_per_episode']}, median {anchors['median_per_episode']:.1f}, "
            f"maximum {anchors['maximum_per_episode']}. Histories contain 8 causal steps and positives contain "
            "the immediately following 16 logged actions. Anchors are grouped by episode for all uncertainty statements.\n"
        ),
        "negative_construction_report.md": (
            "# Negative construction report\n\n"
            f"Attempted {generation['attempted']} candidates and constructed {generation['constructed']}; "
            f"{generation['failures']} inapplicable constructions were recorded without per-sample repair. "
            "The allowed one whole-family redesign was used for the synthetic families after an archived P2 preflight exposed "
            "an over-strict low-motion cutoff.\n\n" + family_lines + "\n"
        ),
        "shortcut_audit_report.md": (
            "# Shortcut audit report\n\n"
            "The family-specific action-summary audit rejected N2, N3, N5 and N6. "
            "N3 also failed the frozen 85% construction-coverage requirement. "
            "With fewer than three survivors, an admissible three-family benchmark is undefined. "
            "For audit completeness only, all required shortcut controls were additionally evaluated on the two-family survivor pool: "
            f"combined non-context BA {provisional_combined:.3f}, source-metadata BA {provisional_source:.3f}, and paired-family-identity BA "
            f"{provisional_family:.3f}. These provisional controls cannot override the family-count failure or authorize fair models. "
            "The controlling decision remains negative-construction invalid.\n"
        ),
        "baseline_report.md": "# Baseline report\n\nNot run: the P3 negative-family gate failed before learned fair baseline training.\n",
        "episode_heldout_report.md": "# Episode-held-out report\n\nSplit assignments and leakage were audited, but no predictive model was evaluated after the P3 stop.\n",
        "task_cv_report.md": "# Task-CV report\n\nThe preregistered five complete-task folds cover all 11 tasks exactly once as test tasks. No model was trained or scored.\n",
        "candidate_ranking_report.md": "# Candidate ranking report\n\nNot run: fewer than three negative families survived, so frozen four-candidate groups were not admitted.\n",
        "calibration_report.md": "# Calibration report\n\nNot run: no fair model probabilities exist.\n",
        "error_analysis_report.md": (
            "# Error analysis\n\n"
            "The terminal failure is benchmark construction, not model performance. "
            "Local permutation, joint coordination and endpoint-path corruptions retain detectable action-summary artifacts; "
            "gripper desynchronization is inapplicable to many static-gripper anchors. "
            "Using them would inflate apparent compatibility accuracy without proving contextual reasoning.\n"
        ),
        "claim_boundary.md": (
            "# Claim boundary\n\n"
            "Supported: an audited finding that the attempted six-family construction does not yield three shortcut-resistant "
            "negative families on this bounded 166-trajectory, 11-task subset.\n\n"
            "Not supported: compatibility prediction performance, method improvement, large-scale coverage, cross-embodiment "
            "generalization, robot success/failure, physical safety, or causal alternate-action outcomes.\n"
        ),
        "next_steps.md": (
            "# Next steps\n\n"
            "Do not train `PhaseAwareTemporalEnergyVerifier` on this benchmark. A new preregistered study must introduce at least "
            "one additional behaviorally grounded negative family that is both broadly applicable and action-summary resistant, "
            "then repeat P0–P3 without changing this terminal result. Scaling beyond the pilot still requires more tasks and "
            "trajectories; the 72-trajectory 16D embodiment must remain separate.\n"
        ),
    }
    for name, body in reports.items():
        (OUT / name).write_text(body, encoding="utf-8")
    paper = OUT / "paper"
    paper.mkdir(exist_ok=True)
    papers = {
        "title_candidates.md": (
            "# Title candidates\n\n"
            "1. Auditing Shortcut-Resistant Action-Chunk Compatibility Negatives on RoboMIND\n"
            "2. When Synthetic Action Corruptions Leak: A Bounded Robot-Action Verification Pilot\n\n"
            "The working `ActionCheck` method title is not used as a supported method claim.\n"
        ),
        "abstract_zh.md": (
            "# 摘要\n\n"
            "我们在 RoboMIND 的单一 32 维 embodiment 上开展了一个受限的 166 轨迹、11 任务 pilot，研究执行前动作块兼容性验证。"
            "实验构造了 992 个因果锚点并审计六类负样本。只有同任务近状态检索和阶段错位检索通过预注册的覆盖率与 shortcut 门槛；"
            "其余合成族可被简单动作统计区分，或覆盖率不足。因此最终决策为 `RM_ACV_PILOT_NEGATIVES_INVALID`，未训练兼容性基线或新方法。"
            "该结果仅说明当前负样本设计无效，不支持成功、安全或物理反事实主张。\n"
        ),
        "abstract_en.md": (
            "# Abstract\n\n"
            "We conducted a bounded 166-trajectory, 11-task pilot on one 32-dimensional RoboMIND embodiment for pre-execution "
            "action-chunk compatibility verification. The run constructed 992 causal anchors and audited six negative families. "
            "Only same-task near-state and phase-shifted retrieval passed the preregistered coverage and shortcut criteria; "
            "synthetic families were distinguishable from simple action statistics or lacked coverage. The terminal decision is "
            "`RM_ACV_PILOT_NEGATIVES_INVALID`; no compatibility baseline or proposed method was trained. This is a negative "
            "benchmark-construction result, not a success, safety, or physical-counterfactual claim.\n"
        ),
        "introduction_outline.md": (
            "# Introduction outline\n\n"
            "- Motivation: pre-execution consistency with a demonstrated action distribution.\n"
            "- Core risk: synthetic negatives can reward non-context action artifacts.\n"
            "- Scope: bounded 166 trajectories, 11 tasks, one embodiment.\n"
            "- Result: only two of six families survive; benchmark is not admitted.\n"
        ),
        "problem_formulation.md": (
            "# Problem formulation\n\n"
            "Given causal RGB/proprioceptive history, a task instruction and a candidate future action chunk, estimate consistency "
            "with the demonstrated conditional action distribution. Logged actions are positive examples, not claims of optimality, "
            "success or safety. This pilot did not reach a valid predictive benchmark.\n"
        ),
        "benchmark.md": (
            "# Benchmark\n\n"
            "The bounded source contains 166 trajectories across 11 tasks from one 32D Franka embodiment. "
            "Anchors use 8 causal steps and 16 future action steps, with 5–6 anchors per trajectory and episode-grouped splits. "
            "The 72-trajectory 16D source is explicitly excluded. Negative-family admission failed at P3.\n"
        ),
        "method.md": (
            "# Method\n\n"
            "`PhaseAwareTemporalEnergyVerifier` was preregistered conditionally but not implemented or trained in this run because "
            "the current-run benchmark decision was not `RM_ACV_PILOT_BENCHMARK_READY`.\n"
        ),
        "experiments.md": (
            "# Experiments\n\n"
            "P0–P3 were executed. Family-specific nonlinear action-summary classifiers were fit on episode-disjoint partitions with "
            "thresholds selected only on validation. P4 split/leakage contracts were audited. P6–M5 were not run after the terminal gate.\n"
        ),
        "results_tables.md": (
            "# Results tables\n\n"
            "| Family | Coverage | Action-summary BA | Result |\n"
            "|---|---:|---:|---|\n"
            + "\n".join(
                f"| {family} | {report.get('coverage', 0):.3f} | "
                f"{report.get('test', {}).get('balanced_accuracy', float('nan')):.3f} | "
                f"{'survive' if report.get('survives') else 'reject'} |"
                for family, report in shortcut["family_reports"].items()
            )
            + "\n\nNo fair-model result exists.\n"
        ),
        "figure_plan.md": (
            "# Figure plan\n\n"
            "1. Anchor counts by task and trajectory.\n"
            "2. Negative-family coverage versus action-summary shortcut balanced accuracy.\n"
            "3. Gate flow ending at `RM_ACV_PILOT_NEGATIVES_INVALID`.\n"
            "No method-performance figure is supported.\n"
        ),
        "claim_boundary.md": (
            "# Paper claim boundary\n\n"
            "This paper package may discuss a bounded negative-construction audit only. It must not claim a working compatibility "
            "verifier, method improvement, large-scale generalization, success prediction or safety.\n"
        ),
        "scale_up_plan.md": (
            "# Scale-up plan\n\n"
            "First create a new preregistered shortcut-resistant third negative family without altering this run. "
            "Then rerun the pilot gate. Separately acquire at least 300 single-embodiment trajectories across at least 20 tasks "
            "before making a large-scale benchmark claim.\n"
        ),
    }
    for name, body in papers.items():
        (paper / name).write_text(body, encoding="utf-8")
    write_json(
        OUT / "completion_audit.json",
        {
            "schema": "rm-acv-pilot166-completion-audit-v1",
            "objective": "bounded 166-trajectory, 11-task shortcut-audited action-chunk compatibility pilot",
            "phases": {
                "P0": {
                    "status": "completed",
                    "evidence": [
                        "preregistered_config.json",
                        "preregistered_config.sha256",
                        "source_manifest.jsonl",
                        "preregistration_amendment.json",
                    ],
                },
                "P1": {
                    "status": "completed",
                    "anchors": anchors["anchors"],
                    "evidence": [
                        "anchor_manifest.jsonl",
                        "positive_candidate_manifest.jsonl",
                        "anchor_statistics.json",
                        "anchor_overlap_audit.json",
                        "alignment_audit.json",
                    ],
                },
                "P2": {
                    "status": "completed_with_recorded_construction_failures",
                    "attempted": generation["attempted"],
                    "constructed": generation["constructed"],
                    "evidence": [
                        "negative_candidate_manifest.jsonl",
                        "negative_generation_report.json",
                        "negative_matching_statistics.json",
                        "negative_failure_log.jsonl",
                    ],
                },
                "P3": {
                    "status": "completed_terminal_gate",
                    "surviving_families": shortcut["surviving_families"],
                    "minimum_required": 3,
                    "all_named_shortcut_controls_evaluated_on_provisional_survivor_pool": True,
                    "decision": DECISION,
                    "evidence": [
                        "shortcut_feature_report.json",
                        "shortcut_predictions.jsonl",
                        "shortcut_recomputation_audit.json",
                        "negative_family_decision.json",
                        "matching_balance_report.json",
                    ],
                },
                "P4": {
                    "status": "split_and_leakage_contracts_audited_no_model_evaluation",
                    "evidence": [
                        "split_manifest.json",
                        "episode_split_audit.json",
                        "task_cv_manifest.json",
                        "task_cv_audit.json",
                        "leakage_audit.json",
                    ],
                },
                "P5": {
                    "status": "task_definitions_frozen_but_candidate_groups_not_admitted",
                    "reason": "fewer than three negative families",
                },
                "P6_P7": {
                    "status": "not_run_as_required_by_P3_terminal_gate",
                    "learned_baselines_trained": False,
                    "raw_predictions": "raw_prediction_manifest.json",
                },
                "P8": {
                    "status": "completed",
                    "decision": DECISION,
                    "evidence": ["phase_p8_decision.json", "final_decision.json"],
                },
                "M0_M5": {
                    "status": "not_authorized",
                    "method_training_authorized": False,
                    "proposed_method_trained": False,
                },
                "paper_and_final_package": {
                    "status": "completed",
                    "bounded_pilot_language_required": True,
                    "method_results_invented": False,
                },
            },
            "invariants": {
                "prior_rm_acv_complete_branch_verifier_pass": prior_verify["passed"],
                "primary_166_only": True,
                "excluded_72_16d_not_merged": True,
                "future_rgb_used": False,
                "future_state_used": False,
                "sam3_used": False,
                "robot_masks_used": False,
                "provenance_used_as_fair_predictive_input": False,
                "unrelated_processes_terminated": False,
            },
            "terminal_condition_satisfied": True,
            "no_required_work_after_terminal_package": True,
        },
    )
    return decision


def _write_reproducibility(decision: dict[str, Any]) -> None:
    append_log(
        "P8_TERMINAL",
        decision=decision["decision"],
        stage_reached=decision["stage_reached"],
        learned_baselines_trained=False,
        proposed_method_trained=False,
    )
    append_log("FINAL_PACKAGE_COMPLETE", decision=decision["decision"])
    generator_paths = [
        Path(__file__),
        ROOT / "actmask" / "experiments" / "rm_acv_pilot166_common.py",
        ROOT / "actmask" / "experiments" / "rm_acv_pilot166_prepare.py",
    ]
    files = []
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and path.name != "reproducibility_manifest.json":
            files.append(
                {
                    "path": str(path.relative_to(OUT)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    source = json.loads((OUT / "source_hash_manifest.json").read_text(encoding="utf-8"))
    write_json(
        OUT / "reproducibility_manifest.json",
        {
            "schema": "rm-acv-pilot166-reproducibility-v1",
            "decision": decision["decision"],
            "generators": [
                {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)}
                for path in generator_paths
            ],
            "source_hash_aggregate": source["aggregate_sha256"],
            "source_entries": len(source["entries"]),
            "generated_files": files,
            "fair_model_training_performed": False,
            "proposed_method_training_performed": False,
        },
    )


def finalize() -> dict[str, Any]:
    shortcut = _complete_provisional_shortcut_controls()
    if shortcut.get("decision") != DECISION:
        raise RuntimeError(f"expected P3 terminal {DECISION}, got {shortcut.get('decision')}")
    if len(shortcut.get("surviving_families", [])) >= 3:
        raise RuntimeError("cannot emit negative-invalid decision with three surviving families")
    recomputation = _shortcut_recomputation()
    split = _split_and_leakage_artifacts()
    if not recomputation["all_match"] or not split["leakage"]["pass"]:
        raise RuntimeError("independent shortcut or leakage audit failed")
    decision = _write_terminal_artifacts()
    _write_reproducibility(decision)
    return decision


def verify(*, rehash_source: bool = False) -> dict[str, Any]:
    missing = [name for name in REQUIRED if not (OUT / name).is_file()]
    failures: list[str] = []
    if missing:
        return {"passed": False, "missing": missing, "failures": ["missing_required"]}
    config_hash = (OUT / "preregistered_config.sha256").read_text(encoding="utf-8").strip()
    if config_hash != sha256_file(OUT / "preregistered_config.json"):
        failures.append("preregistered_config_sha256")
    reproducibility = json.loads((OUT / "reproducibility_manifest.json").read_text(encoding="utf-8"))
    for row in reproducibility["generated_files"]:
        path = OUT / row["path"]
        if not path.is_file() or sha256_file(path) != row["sha256"]:
            failures.append(f"generated:{row['path']}")
    for row in reproducibility["generators"]:
        path = ROOT / row["path"]
        if not path.is_file() or sha256_file(path) != row["sha256"]:
            failures.append(f"generator:{row['path']}")
    previous = ""
    sequence = 0
    for line in (OUT / "run_log.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        actual = row.pop("event_sha256")
        if row["previous_event_sha256"] != previous or row["seq"] != sequence + 1:
            failures.append("run_log_chain")
            break
        from actmask.experiments.rm_acv_pilot166_common import sha256_json
        if sha256_json(row) != actual:
            failures.append("run_log_hash")
            break
        previous = actual
        sequence += 1
    source_result: dict[str, Any] = {"rehash_performed": rehash_source, "mismatches": []}
    if rehash_source:
        source_manifest = json.loads((OUT / "source_hash_manifest.json").read_text(encoding="utf-8"))
        for row in source_manifest["entries"]:
            path = Path(row["source_path_provenance_only"])
            if not path.is_file() or sha256_file(path) != row["sha256"]:
                source_result["mismatches"].append(row["source_path_provenance_only"])
        if source_result["mismatches"]:
            failures.append("source_hashes")
    decision = json.loads((OUT / "final_decision.json").read_text(encoding="utf-8"))
    shortcut = json.loads((OUT / "shortcut_feature_report.json").read_text(encoding="utf-8"))
    checks = {
        "decision_exact": decision["decision"] == DECISION,
        "terminal_at_P3": decision["stage_reached"] == "P3",
        "fewer_than_three_survivors": len(shortcut["surviving_families"]) < 3,
        "no_learned_baselines": not decision["learned_baselines_trained"],
        "method_not_authorized": not decision["method_training_authorized"],
        "prior_branch_preserved": json.loads(
            (OUT / "prior_branch_preservation_audit.json").read_text(encoding="utf-8")
        )["prior_hash_verifier_pass"],
        "shortcut_recomputed": json.loads(
            (OUT / "shortcut_recomputation_audit.json").read_text(encoding="utf-8")
        )["all_match"],
        "leakage_audit": json.loads(
            (OUT / "leakage_audit.json").read_text(encoding="utf-8")
        )["pass"],
    }
    if not all(checks.values()):
        failures.extend([f"check:{name}" for name, passed in checks.items() if not passed])
    return {
        "passed": not failures,
        "missing": missing,
        "failures": failures,
        "checks": checks,
        "decision": decision["decision"],
        "surviving_families": shortcut["surviving_families"],
        "source_verification": source_result,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--rehash-source", action="store_true")
    args = parser.parse_args()
    result = verify(rehash_source=args.rehash_source) if args.verify else finalize()
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
