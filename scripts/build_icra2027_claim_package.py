#!/usr/bin/env python3
"""Build deterministic ICRA 2027 claim tables from frozen ActMask reports.

The script is deliberately downstream-only: it reads frozen JSON files,
checks their byte hashes and scientific invariants, and writes a compact claim
ledger plus LaTeX/JSON tables. It never imports training or evaluation code.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "outputs/actmask/icra2027_submission"

SOURCES = {
    "state_config": {
        "path": "outputs/actmask/milestone3r_nl_v2/full_run_config.json",
        "sha256": "12c692d032ba37bdb14761a1c63ff8ceccff72af5f037d936ebaed8748016a47",
    },
    "state_evaluation": {
        "path": "outputs/actmask/milestone3r_nl_v2/full_evaluation/full_evaluation.json",
        "sha256": "abc371e50fda1e4969f37683201ef0aa9a01e6e5d00ac828521adbceaf7a96e4",
    },
    "state_ranking_correction": {
        "path": "outputs/actmask/milestone3r_nl_v2_rankfix/rankfix_report.json",
        "sha256": "e8b2e813ee5dfe07463bc544d6c39d6fed1658eb44051116f14db05e1734e982",
    },
    "state_world_ci_correction": {
        "path": "outputs/actmask/milestone3r_nl_v2_statsfix/world_cluster_bootstrap.json",
        "sha256": "bb6e3189f0f56160193bc3694f98becb53740c30f199fe5d13f954cb79a54b31",
    },
    "state_red_team": {
        "path": "outputs/actmask/milestone3s_paper_package/red_team_audit.json",
        "sha256": "b9ae305ddf58f5bff185717e886af56ccc5d447a9a401aab622aa97fb3a00098",
    },
    "visual_baselines": {
        "path": "outputs/actmask/milestone4r_v3_all_family_candidate_diversity/full_probe/standard_baseline_report.json",
        "sha256": "fcc9f296d20a5333b0293e1c2c590a79f2be8b598682400028ded08d941f8557",
    },
    "visual_headroom": {
        "path": "outputs/actmask/milestone4r_v3_all_family_candidate_diversity/full_probe/derived_headroom_metrics.json",
        "sha256": "8db5f49591ebd433d823c6eb81931e5a87c36311fc76c75335c899c3e60f94a7",
    },
    "visual_ci_correction": {
        "path": "outputs/actmask/milestone5a_relational_probe/corrected_4r_v3_reporting_note.json",
        "sha256": "34f16e6e07f6de9f48e87f1d33c606f314ef4e5aa90a9ab49632f61b2f2e2a42",
    },
    "visual_summary": {
        "path": "outputs/actmask/milestone4r_v3_all_family_candidate_diversity/full_probe/probe_summary.json",
        "sha256": "3accfb377fdd1b622b995b520be97bde09f680252804776c15c788d5744114b3",
    },
    "visual_audit": {
        "path": "outputs/actmask/milestone4r_v3_all_family_candidate_diversity/full_probe/robust_audit.json",
        "sha256": "2a2a17c98dea9df8552186f3b96bf93076ca34c6de11e4f81f9ae1f3150f6b43",
    },
    "real_manifest": {
        "path": "outputs/actmask/r_series_real_robot_verification/asu_attempts/asu_ur5_object_relation_heldout_goal_object_v1/r2/data_manifest.json",
        "sha256": "68e220d39f1b0ab50a912d1955370c1cbe652340de641176d659594201b842c0",
    },
    "real_task_formulation": {
        "path": "outputs/actmask/r_series_real_robot_verification/asu_attempts/asu_ur5_object_relation_heldout_goal_object_v1/r2/task_formulation.json",
        "sha256": "a896a2f26f940f4d48d96f7eebb715b5afcea05925e3678762d9f3393235bcd2",
    },
    "real_split_audit": {
        "path": "outputs/actmask/r_series_real_robot_verification/asu_attempts/asu_ur5_object_relation_heldout_goal_object_v1/r2/split_audit.json",
        "sha256": "4969e8235836a62263d191afdc3189a5148e0ad57f2cc2777195917bd1a1a123",
    },
    "real_negative_audit": {
        "path": "outputs/actmask/r_series_real_robot_verification/asu_attempts/asu_ur5_object_relation_heldout_goal_object_v1/r2/negative_sampling_audit.json",
        "sha256": "ecb1fa05b71c3f17e8a36a30d1a8cf118e85d3a9cdfdcf3a27f9991e57890c4c",
    },
    "real_decision": {
        "path": "outputs/actmask/r_series_real_robot_verification/asu_attempts/asu_ur5_object_relation_heldout_goal_object_v1/r5_final_decision.json",
        "sha256": "3486a3fd8f1183b6d74968c1a39e00e61dbda23023a674d91060be4774fea05f",
    },
    "wav_external_audit": {
        "path": "outputs/actmask/icra2027_wav_external_audit/audit.json",
        "sha256": "e9e1aaa2f37cc3b705b70f71f8e1439fdeb1d0f426e1d50b58ca331c6e6091f0",
    },
}

TASKS = (
    ("Moving cube", "MovingCubeIntercept"),
    ("Moving window", "SignedMovingWindowPlacement"),
    ("Rotating window", "FixedPhaseRotatingCaptureWindow"),
)

STATE_METHODS = (
    ("Analytic", "Last-two finite difference", "LastTwoFrameFiniteDifference"),
    ("Analytic", "Multi-frame linear velocity", "MultiFrameLinearVelocity"),
    ("Analytic", "Robust linear velocity", "RobustLinearVelocity"),
    ("Analytic", "Constant-acceleration fit", "ConstantAccelerationFit"),
    ("Analytic", "Alpha--beta filter", "AlphaBetaFilter"),
    ("Analytic", "Kalman constant velocity", "KalmanFilterConstantVelocity"),
    ("Analytic", "Kalman constant acceleration", "KalmanFilterConstantAcceleration"),
    ("Analytic", "Exponential smoothing", "ExponentialSmoothingVelocity"),
    ("Learned (3)", "Static MLP", "StaticMLP"),
    ("Learned (3)", "Action-only MLP", "ActionMLP"),
    ("Learned (3)", "Unordered-history MLP", "UnorderedHistoryMLP"),
    ("Learned (3)", "Temporal convolution", "TemporalConv1D"),
    ("Learned (3)", "Ordered temporal MLP", "OrderedTemporalMLP"),
    ("Learned (3)", "GRU selection run", "GRU"),
    ("Confirm. (5)", "GRU confirmation", "GRUFiveSeed"),
)

STATE_FAMILY_NAMES = {
    "damped_moving_capture": "Damped capture",
    "hysteretic_moving_container": "Moving container",
    "damped_rotating_slot": "Rotating slot",
}

STATE_MECHANISM_NAMES = {
    "history_identifiable_damping_drive": "Damping/drive",
    "hysteretic_mode_memory": "Hysteresis",
}

STATE_ANALYTIC_NAMES = {
    "LastTwoFrameFiniteDifference": "Last-two FD",
    "MultiFrameLinearVelocity": "Multi-frame linear",
    "RobustLinearVelocity": "Robust linear",
    "ConstantAccelerationFit": "Const.-accel.",
    "AlphaBetaFilter": "Alpha--beta",
    "KalmanFilterConstantVelocity": "Kalman CV",
    "KalmanFilterConstantAcceleration": "Kalman CA",
    "ExponentialSmoothingVelocity": "Exp. smoothing",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_sources() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    data: dict[str, Any] = {}
    inventory = []
    for name, frozen in SOURCES.items():
        path = ROOT / frozen["path"]
        if not path.is_file():
            raise RuntimeError(f"missing frozen source: {frozen['path']}")
        observed = _sha256(path)
        if observed != frozen["sha256"]:
            raise RuntimeError(
                f"frozen source hash mismatch for {name}: "
                f"expected {frozen['sha256']}, observed {observed}"
            )
        with path.open(encoding="utf-8") as handle:
            data[name] = json.load(handle)
        inventory.append(
            {
                "name": name,
                "path": frozen["path"],
                "bytes": path.stat().st_size,
                "sha256": observed,
            }
        )
    return data, inventory


def _close(left: float, right: float, tolerance: float = 1e-12) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=tolerance)


def _method_mean(report: dict[str, Any], method: str) -> float:
    values = [
        float(row["pair_order_accuracy"])
        for row in report["records"]
        if row["method"] == method and row["split"] == "test"
    ]
    if not values:
        raise RuntimeError(f"missing test records for {method}")
    return mean(values)


def _method_split_mean(report: dict[str, Any], method: str, split: str) -> float:
    values = [
        float(row["pair_order_accuracy"])
        for row in report["records"]
        if row["method"] == method and row["split"] == split
    ]
    if not values:
        raise RuntimeError(f"missing {split} records for {method}")
    return mean(values)


def _check(data: dict[str, Any]) -> list[dict[str, Any]]:
    state_config = data["state_config"]
    state = data["state_evaluation"]
    state_ranking = data["state_ranking_correction"]
    state_ci = data["state_world_ci_correction"]
    visual = data["visual_baselines"]
    headroom = data["visual_headroom"]
    correction = data["visual_ci_correction"]
    visual_summary = data["visual_summary"]
    visual_audit = data["visual_audit"]
    red_team = data["state_red_team"]
    real_manifest = data["real_manifest"]
    real_task = data["real_task_formulation"]
    real_split = data["real_split_audit"]
    real_negative = data["real_negative_audit"]
    real_decision = data["real_decision"]
    wav = data["wav_external_audit"]

    checks: list[tuple[str, bool, Any]] = [
        (
            "state_config_link",
            state["full_config_sha256"] == SOURCES["state_config"]["sha256"],
            state["full_config_sha256"],
        ),
        (
            "state_execution_budget",
            state_config["candidate_execution_budget"] == 92000
            and len(state_config["active_id_cells"]) == 5
            and state_config["candidate_counts"] == [5, 10, 20],
            {
                "candidate_execution_budget": state_config["candidate_execution_budget"],
                "active_id_cells": len(state_config["active_id_cells"]),
                "candidate_counts": state_config["candidate_counts"],
            },
        ),
        (
            "state_split_contract",
            state_config["split"]
            == {
                "train_world_modulo": [0, 1, 2],
                "validation_world_modulo": [3],
                "test_world_modulo": [4],
            },
            state_config["split"],
        ),
        (
            "state_nonranking_gates",
            all(value for name, value in state["gate"].items() if name != "ranking_dynamic_advantage"),
            state["gate"],
        ),
        (
            "state_ranking_correction",
            state_ranking["pass"] is True
            and state_ranking["correction"]["correct_group_key"]
            == ["family", "mechanism", "world_id", "branch"]
            and state_ranking["pair_order_recheck"]["matches_parent_gru"] is True
            and all(
                cell["contexts"] == 600
                and cell["mixed_context_fraction"] == 1.0
                for table in state_ranking["rankings"].values()
                for cell in table.values()
            ),
            {
                "pass": state_ranking["pass"],
                "correction": state_ranking["correction"],
                "gates": state_ranking["gates"],
            },
        ),
        (
            "state_ranking_raw_scores",
            _sha256(ROOT / "outputs/actmask/milestone3r_nl_v2_rankfix/raw_scores.npz")
            == state_ranking["raw_scores"]["sha256"],
            state_ranking["raw_scores"],
        ),
        ("state_selected_learned", state["selected_learned"] == "GRU", state["selected_learned"]),
        (
            "state_selected_analytic",
            state["selected_analytic"] == "MultiFrameLinearVelocity",
            state["selected_analytic"],
        ),
        ("state_gru_test", _close(_method_mean(state, "GRUFiveSeed"), 1.0), _method_mean(state, "GRUFiveSeed")),
        (
            "state_analytic_test",
            _close(_method_mean(state, "MultiFrameLinearVelocity"), 0.6),
            _method_mean(state, "MultiFrameLinearVelocity"),
        ),
        ("state_delta", _close(state["bootstrap"]["delta"], 0.4), state["bootstrap"]),
        (
            "state_world_cluster_ci",
            state_ci["pass"] is True
            and state_ci["schema"] == "actmask-state-world-cluster-bootstrap-v1"
            and state_ci["test_units"]
            == {
                "base_worlds": 300,
                "candidate_pairs": 6000,
                "candidate_pairs_per_world": 20,
                "family_mechanism_cells": 5,
            }
            and state_ci["pair_order"]["delta"] == 0.4
            and state_ci["pair_order"]["world_cluster_ci95"]
            == [0.37666666666666665, 0.4216666666666667],
            state_ci,
        ),
        ("state_ood_count", len(state["ood"]) == 3, state["ood"]),
        (
            "state_ood_positive",
            all(row["learned"] > row["analytic"] for row in state["ood"]),
            state["ood"],
        ),
        (
            "state_controls",
            state["interventions"]["correct"] == 1.0
            and state["interventions"]["last_two_only"] == 0.5
            and state["swap"]["counterfactual_swap_accuracy"] == 1.0
            and state["swap"]["score_swap_consistency"] == 1.0,
            {"interventions": state["interventions"], "swap": state["swap"]},
        ),
        (
            "state_candidate_diversity",
            state["candidate_diversity"]["nondegenerate_world_fraction"] == 1.0
            and state["candidate_diversity"]["mean_branch_outcome_change_rate"] == 1.0
            and state["candidate_diversity"]["max_candidate_index_leakage"] == 0.0,
            state["candidate_diversity"],
        ),
        (
            "state_red_team",
            red_team["blocking_fail_count"] == 0,
            {
                "blocking_fail_count": red_team["blocking_fail_count"],
                "warning_count": red_team.get("warning_count"),
            },
        ),
        (
            "visual_schema",
            visual["schema"] == "milestone4r-standard-baselines-v1",
            visual.get("schema"),
        ),
        (
            "visual_ci_suppressed",
            correction["pair_order_ci95"] is None
            and correction["ci_status"].startswith("suppressed"),
            correction,
        ),
        (
            "visual_execution_budget",
            visual_summary["candidate_executions"] == 7680
            and visual_summary["worlds_per_task"] == 128
            and all(
                row["bundles"] == 256 and row["candidates"] == 2560
                for row in visual_summary["tasks"].values()
            ),
            visual_summary,
        ),
        ("visual_integrity_audit", visual_audit["passed"] is True, visual_audit["passed"]),
        (
            "real_log_contract",
            real_manifest["episodes"] == 84
            and real_manifest["task_families"] == 11
            and real_manifest["candidate_sets"] == 1151
            and real_manifest["candidate_rows"] == 5755
            and real_split["pass"] is True
            and real_split["whole_task_heldout"] is True
            and real_negative["pass"] is True
            and real_negative["semantic_checks_pass"] is True
            and real_task["formulation_a"]["positive"]
            == "actual next logged action chunk"
            and "true counterfactual physical success prediction"
            in real_task["not_claimed"],
            {
                "manifest": real_manifest,
                "split": real_split,
                "negative_audit": real_negative,
                "not_claimed": real_task["not_claimed"],
            },
        ),
        (
            "real_log_rejection",
            real_decision["audits_pass"] is True
            and real_decision["decision"] == "E. BASELINES_SATURATE"
            and real_decision["method_authorized"] is False
            and real_decision["heldout_goal_object"]["best_standard"]
            == "proprio_action_gru"
            and real_decision["heldout_goal_object"]["best_standard_validation_pair_order"]
            > real_decision["thresholds"]["standard_saturation_max"]
            and max(
                real_decision["heldout_goal_object"]["trivial_validation"].values()
            )
            <= real_decision["thresholds"]["trivial_shortcut_max"],
            real_decision,
        ),
        (
            "wav_external_transition_necessity",
            wav["schema_version"] == 1
            and wav["protocol_sha256"]
            == "4a76b946de840b6b8f0357d0c24da4034bb3216ed3a635598ac665c13807ea22"
            and wav["upstream"]["commit"]
            == "527159b06149beacfb3b2d77af7d938ca4efa32d"
            and wav["config"]["seeds"] == [9102, 9103, 9104, 9105, 9106]
            and wav["config"]["complexities"] == [6, 8, 10, 12, 14]
            and wav["dataset_counts"]
            == {
                "train": 2000,
                "tests": {"6": 1000, "8": 2500, "10": 2500, "12": 2500, "14": 2500},
            }
            and wav["decision"]["pass"] is True
            and len(wav["decision"]["gates"]) == 15
            and all(gate["pass"] is True for gate in wav["decision"]["gates"])
            and all(
                wav["summaries"]["paired"]["by_complexity"][str(complexity)]["dynamic_accuracy"]["mean"]
                - max(
                    wav["summaries"]["current_only"]["by_complexity"][str(complexity)]["dynamic_accuracy"]["mean"],
                    wav["summaries"]["next_only"]["by_complexity"][str(complexity)]["dynamic_accuracy"]["mean"],
                )
                >= 0.10
                and wav["summaries"]["sparse_idm"]["by_complexity"][str(complexity)]["dynamic_accuracy"]["mean"]
                >= 0.90
                and wav["summaries"]["transition_rule"]["by_complexity"][str(complexity)]["action_accuracy"]["mean"]
                == 1.0
                for complexity in (6, 8, 10, 12, 14)
            ),
            {
                "upstream_commit": wav["upstream"]["commit"],
                "dataset_counts": wav["dataset_counts"],
                "decision": wav["decision"],
            },
        ),
    ]

    for _, task in TASKS:
        fair = visual["tasks"][task]["fair"]
        derived = headroom["tasks"][task]
        audit = visual_audit["tasks"][task]
        checks.extend(
            [
                (
                    f"{task}_matching_contract",
                    audit["malformed_groups"] == 0
                    and all(value == 1.0 for value in audit["pair_matching"].values())
                    and audit["candidate_diversity"]["c10_aligned"] is True
                    and audit["candidate_diversity"]["mixed_success_fraction"] > 0.0,
                    {
                        "malformed_groups": audit["malformed_groups"],
                        "pair_matching": audit["pair_matching"],
                        "candidate_diversity": audit["candidate_diversity"],
                    },
                ),
                (
                    f"{task}_static_controls",
                    fair["action_only"]["combined_held_dropout"]["pair_order_accuracy"] == 0.5
                    and fair["current_rgbd"]["combined_held_dropout"]["pair_order_accuracy"] == 0.5,
                    {
                        "action_only": fair["action_only"]["combined_held_dropout"],
                        "current_rgbd": fair["current_rgbd"]["combined_held_dropout"],
                    },
                ),
                (
                    f"{task}_temporal_necessity_controls",
                    fair["last_two_rgbd_difference"]["combined_held_dropout"]["pair_order_accuracy"] == 0.5
                    and max(
                        fair["unordered_rgbd"]["combined_held_dropout"]["pair_order_accuracy"],
                        fair["unordered_pointcloud"]["combined_held_dropout"]["pair_order_accuracy"],
                    )
                    <= 0.55,
                    {
                        "last_two_rgbd_difference": fair["last_two_rgbd_difference"]["combined_held_dropout"],
                        "unordered_rgbd": fair["unordered_rgbd"]["combined_held_dropout"],
                        "unordered_pointcloud": fair["unordered_pointcloud"]["combined_held_dropout"],
                        "frozen_failure_threshold": 0.55,
                    },
                ),
                (
                    f"{task}_centroid_saturation",
                    fair["centroid_velocity"]["combined_held_dropout"]["pair_order_accuracy"] == 1.0
                    and derived["best_standard_fair"] == "centroid_velocity"
                    and derived["model_headroom_state_minus_best_standard_fair"] == 0.0,
                    {
                        "centroid": fair["centroid_velocity"]["combined_held_dropout"],
                        "headroom": derived,
                    },
                ),
            ]
        )

    rendered = [
        {"name": name, "status": "PASS" if passed else "FAIL", "observed": observed}
        for name, passed, observed in checks
    ]
    failures = [row["name"] for row in rendered if row["status"] != "PASS"]
    if failures:
        raise RuntimeError(f"scientific invariant checks failed: {failures}")
    return rendered


def _fmt(value: float) -> str:
    return f"{value:.3f}"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_table(path: Path, columns: list[str], rows: list[list[str]]) -> None:
    alignment = "l" + "c" * (len(columns) - 1)
    lines = [
        f"\\begin{{tabular}}{{{alignment}}}",
        "\\toprule",
        " & ".join(columns) + r" \\",
        "\\midrule",
    ]
    lines.extend(" & ".join(row) + r" \\" for row in rows)
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _state_rows(state: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [
        {
            "split": "ID",
            "learned": _method_mean(state, "GRUFiveSeed"),
            "analytic": _method_mean(state, "MultiFrameLinearVelocity"),
            "analytic_method": "MultiFrameLinearVelocity",
        }
    ]
    label = {
        "physical_parameter_ood": "Physical OOD",
        "temporal_delay_ood": "Delay OOD",
        "held_mechanism_ood": "Held mechanism",
    }
    rows.extend(
        {
            "split": label[row["axis"]],
            "learned": float(row["learned"]),
            "analytic": float(row["analytic"]),
            "analytic_method": row["best_analytic"],
        }
        for row in state["ood"]
    )
    for row in rows:
        row["delta"] = row["learned"] - row["analytic"]
    return rows


def _visual_rows(visual: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for display, task in TASKS:
        fair = visual["tasks"][task]["fair"]
        condition = "combined_held_dropout"
        rows.append(
            {
                "family": display,
                "action_only": fair["action_only"][condition]["pair_order_accuracy"],
                "current_rgbd": fair["current_rgbd"][condition]["pair_order_accuracy"],
                "ordered_rgbd_gru": fair["ordered_rgbd_gru"][condition]["pair_order_accuracy"],
                "world_pointcloud_gru": fair["world_pointcloud_gru"][condition]["pair_order_accuracy"],
                "centroid_velocity": fair["centroid_velocity"][condition]["pair_order_accuracy"],
                "centroid_top1": fair["centroid_velocity"][condition]["top1_success"],
            }
        )
    return rows


def _state_baseline_rows(state: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "category": category,
            "method": display,
            "source_method": method,
            "validation": _method_split_mean(state, method, "validation"),
            "test": _method_split_mean(state, method, "test"),
        }
        for category, display, method in STATE_METHODS
    ]


def _state_cell_rows(state: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "family": STATE_FAMILY_NAMES[row["family"]],
            "mechanism": STATE_MECHANISM_NAMES[row["mechanism"]],
            "learned": float(row["learned"]),
            "analytic_method": STATE_ANALYTIC_NAMES[row["best_analytic"]],
            "analytic": float(row["analytic"]),
        }
        for row in state["per_task_mechanism"]
    ]


def _state_ranking_rows(state_ranking: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "candidates": name,
            "contexts": int(value["dynamic"]["contexts"]),
            "mixed_context_fraction": float(value["dynamic"]["mixed_context_fraction"]),
            "dynamic_top1": float(value["dynamic"]["top1_success"]),
            "static_top1": float(value["StaticMLP"]["top1_success"]),
            "action_top1": float(value["ActionMLP"]["top1_success"]),
            "dynamic_ndcg": float(value["dynamic"]["corrected_utility_gain_ndcg"]),
            "dynamic_regret": float(value["dynamic"]["normalized_regret"]),
        }
        for name in ("C5", "C10", "C20")
        for value in (state_ranking["rankings"][name],)
    ]


def _wav_rows(wav: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for complexity in (6, 8, 10, 12, 14):
        key = str(complexity)
        current = wav["summaries"]["current_only"]["by_complexity"][key]["dynamic_accuracy"]
        nxt = wav["summaries"]["next_only"]["by_complexity"][key]["dynamic_accuracy"]
        single = current if current["mean"] >= nxt["mean"] else nxt
        rows.append(
            {
                "complexity": complexity,
                "test_count": wav["dataset_counts"]["tests"][key],
                "strongest_single_mean": float(single["mean"]),
                "strongest_single_std": float(single["sample_std"]),
                "paired_mean": float(wav["summaries"]["paired"]["by_complexity"][key]["dynamic_accuracy"]["mean"]),
                "paired_std": float(wav["summaries"]["paired"]["by_complexity"][key]["dynamic_accuracy"]["sample_std"]),
                "sparse_idm_mean": float(wav["summaries"]["sparse_idm"]["by_complexity"][key]["dynamic_accuracy"]["mean"]),
                "sparse_idm_std": float(wav["summaries"]["sparse_idm"]["by_complexity"][key]["dynamic_accuracy"]["sample_std"]),
                "transition_rule": float(wav["summaries"]["transition_rule"]["by_complexity"][key]["dynamic_accuracy"]["mean"]),
            }
        )
    return rows


def _admission_rows(
    state: dict[str, Any], visual: dict[str, Any], real: dict[str, Any], wav: dict[str, Any]
) -> list[dict[str, str]]:
    visual_fair = [visual["tasks"][task]["fair"] for _, task in TASKS]
    visual_last_two = max(
        fair["last_two_rgbd_difference"]["combined_held_dropout"]["pair_order_accuracy"]
        for fair in visual_fair
    )
    visual_unordered = max(
        fair[method]["combined_held_dropout"]["pair_order_accuracy"]
        for fair in visual_fair
        for method in ("unordered_rgbd", "unordered_pointcloud")
    )
    return [
        {
            "benchmark": "State observations",
            "integrity": "Pass",
            "temporal_evidence": f"Final-two {_fmt(state['interventions']['last_two_only'])}",
            "fair_control": (
                f"Analytic {_fmt(_method_mean(state, 'MultiFrameLinearVelocity'))} "
                f"vs GRU {_fmt(_method_mean(state, 'GRUFiveSeed'))}"
            ),
            "decision": "Admit controlled comparison",
        },
        {
            "benchmark": "Public WAV MiniGrid",
            "integrity": "Pinned upstream",
            "temporal_evidence": (
                "Single $\\leq$"
                f"{_fmt(max(wav['summaries'][method]['macro_dynamic_accuracy']['mean'] for method in ('current_only', 'next_only')))}"
            ),
            "fair_control": (
                "SparseIDM "
                f"{_fmt(wav['summaries']['sparse_idm']['macro_dynamic_accuracy']['mean'])}"
            ),
            "decision": "Admit transition evidence",
        },
        {
            "benchmark": "Visual RGB-D stress test",
            "integrity": "Pass",
            "temporal_evidence": (
                f"Final-two {_fmt(visual_last_two)}; unordered $\\leq${_fmt(visual_unordered)}"
            ),
            "fair_control": "Centroid 1.000",
            "decision": "Reject method claim",
        },
        {
            "benchmark": "Public UR5 logs",
            "integrity": "Logged groups pass",
            "temporal_evidence": (
                "Trivial val. $\\leq$"
                f"{_fmt(max(real['heldout_goal_object']['trivial_validation'].values()))}"
            ),
            "fair_control": (
                "Proprio GRU "
                f"{_fmt(real['heldout_goal_object']['best_standard_validation_pair_order'])} val."
            ),
            "decision": "Reject: no re-execution",
        },
    ]


def build(output: Path) -> dict[str, Any]:
    data, sources = _read_sources()
    checks = _check(data)
    state = data["state_evaluation"]
    state_ci = data["state_world_ci_correction"]
    state_rows = _state_rows(state)
    state_baseline_rows = _state_baseline_rows(state)
    state_cell_rows = _state_cell_rows(state)
    state_ranking_rows = _state_ranking_rows(data["state_ranking_correction"])
    visual_rows = _visual_rows(data["visual_baselines"])
    wav_rows = _wav_rows(data["wav_external_audit"])
    admission_rows = _admission_rows(
        state, data["visual_baselines"], data["real_decision"], data["wav_external_audit"]
    )

    claims = [
        {
            "id": "ICRA-C1",
            "claim": "The final state benchmark uses execution-grounded, action-diverse matched pairs.",
            "source": SOURCES["state_evaluation"]["path"],
            "selectors": ["candidate_diversity", "rankings.C20", "full_gate_passed"],
            "allowed_scope": "controlled state-observation benchmark",
            "forbidden_scope": "visual, real-robot, safety, or policy claim",
        },
        {
            "id": "ICRA-C2",
            "claim": "GRU exceeds the validation-selected fair analytic baseline by 0.400 ID pair-order accuracy.",
            "source": SOURCES["state_world_ci_correction"]["path"],
            "selectors": ["pair_order", "test_units", "correction"],
            "value": {
                "delta": state_ci["pair_order"]["delta"],
                "ci95": state_ci["pair_order"]["world_cluster_ci95"],
                "base_worlds": state_ci["test_units"]["base_worlds"],
            },
            "allowed_scope": "base-world-clustered frozen state benchmark effect",
            "forbidden_scope": "open-world population or physical-system interval",
        },
        {
            "id": "ICRA-C3",
            "claim": "The state result has positive advantage on three frozen OOD axes.",
            "source": SOURCES["state_evaluation"]["path"],
            "selectors": ["ood"],
            "value": state["ood"],
            "allowed_scope": "named procedural OOD cells",
            "forbidden_scope": "general OOD or sim-to-real claim",
        },
        {
            "id": "ICRA-C4",
            "claim": "The RGB-D benchmark admission case is saturated by a fair centroid-velocity control.",
            "source": SOURCES["visual_baselines"]["path"],
            "selectors": ["tasks.*.fair.*.combined_held_dropout"],
            "value": visual_rows,
            "allowed_scope": "negative visual benchmark-admission result",
            "forbidden_scope": "positive visual method or hard-benchmark claim",
        },
        {
            "id": "ICRA-C5",
            "claim": "Visual pair-order confidence intervals are suppressed because they target a different estimator.",
            "source": SOURCES["visual_ci_correction"]["path"],
            "selectors": ["pair_order_point_estimate", "pair_order_ci95", "ci_status"],
            "value": data["visual_ci_correction"],
            "allowed_scope": "three-seed point estimates only",
            "forbidden_scope": "the frozen non-comparable visual intervals",
        },
        {
            "id": "ICRA-C6",
            "claim": "The frozen state protocol accounts for 92,000 candidate executions and aligned C5/C10/C20 views.",
            "source": SOURCES["state_config"]["path"],
            "selectors": ["candidate_execution_budget", "candidate_counts", "world_counts"],
            "value": {
                "candidate_execution_budget": data["state_config"]["candidate_execution_budget"],
                "candidate_counts": data["state_config"]["candidate_counts"],
            },
            "allowed_scope": "frozen protocol accounting",
            "forbidden_scope": "claim that all 92,000 executions are independent worlds",
        },
        {
            "id": "ICRA-C7",
            "claim": "The visual pilot passes its declared matching, schema, and candidate-diversity audits before model admission is considered.",
            "source": SOURCES["visual_audit"]["path"],
            "selectors": ["passed", "tasks.*.pair_matching", "tasks.*.candidate_diversity"],
            "allowed_scope": "three procedural visual families",
            "forbidden_scope": "real-camera or general visual benchmark validity",
        },
        {
            "id": "ICRA-C8",
            "claim": "History-keyed C5/C10/C20 ranking uses 600 mixed-outcome test contexts and retains a 1.000 versus 0.500 Top-1 separation.",
            "source": SOURCES["state_ranking_correction"]["path"],
            "selectors": ["correction", "rankings", "raw_scores"],
            "allowed_scope": "corrected within-history candidate ranking",
            "forbidden_scope": "the superseded branch-pooled ranking table",
        },
        {
            "id": "ICRA-C9",
            "claim": "The public UR5 logged-continuation task is rejected because alternate candidates lack matched physical re-execution and a standard baseline saturates validation.",
            "source": SOURCES["real_decision"]["path"],
            "selectors": ["claim_scope", "heldout_goal_object", "decision"],
            "value": {
                "episodes": data["real_manifest"]["episodes"],
                "candidate_sets": data["real_manifest"]["candidate_sets"],
                "proprio_gru_validation": data["real_decision"]["heldout_goal_object"]["best_standard_validation_pair_order"],
                "proprio_gru_test": data["real_decision"]["heldout_goal_object"]["best_standard_test_pair_order"],
            },
            "allowed_scope": "negative admission result for logged continuation identification",
            "forbidden_scope": "counterfactual real-robot success prediction or real-robot transfer",
        },
        {
            "id": "ICRA-C10",
            "claim": "On the pinned public WAV MiniGrid state-complexity split, paired observations beat the strongest single-state control by at least 0.459 dynamic accuracy at every tested complexity.",
            "source": SOURCES["wav_external_audit"]["path"],
            "selectors": ["upstream", "summaries", "decision"],
            "value": wav_rows,
            "allowed_scope": "public WAV MiniGrid inverse-dynamics state-complexity benchmark under frozen evidence controls",
            "forbidden_scope": "WAV robot experiments, Action Following Score, real-robot verification, or general benchmark validity",
        },
    ]

    _write_json(output / "claim_ledger.json", {"schema": "actmask-icra2027-claim-ledger-v1", "claims": claims})
    _write_json(output / "tables/state_results.json", {"schema": "actmask-icra2027-state-results-v1", "rows": state_rows})
    _write_json(output / "tables/state_baselines.json", {"schema": "actmask-icra2027-state-baselines-v1", "rows": state_baseline_rows})
    _write_json(output / "tables/state_family_mechanism.json", {"schema": "actmask-icra2027-state-family-mechanism-v1", "selection": "strongest fair analytic control within each frozen test cell", "rows": state_cell_rows})
    _write_json(output / "tables/state_ranking.json", {"schema": "actmask-icra2027-state-ranking-v1", "rows": state_ranking_rows})
    _write_json(output / "tables/visual_admission.json", {"schema": "actmask-icra2027-visual-admission-v1", "condition": "held camera and all non-complete history corruptions (frozen key: combined_held_dropout)", "confidence_intervals_reported": False, "rows": visual_rows})
    _write_json(output / "tables/wav_external_audit.json", {"schema": "actmask-icra2027-wav-external-audit-v1", "metric": "official dynamic accuracy", "rows": wav_rows})
    _write_json(
        output / "tables/admission_decision.json",
        {
            "schema": "actmask-icra2027-admission-decision-v1",
            "state_headroom_scope": "aggregate ID and named OOD cells; not every family-mechanism cell",
            "visual_temporal_failure_threshold": 0.55,
            "rows": admission_rows,
        },
    )

    _write_table(
        output / "tables/state_results.tex",
        ["Split", "GRU", "Fair analytic", "$\\Delta$"],
        [[row["split"], _fmt(row["learned"]), _fmt(row["analytic"]), f"+{_fmt(row['delta'])}"] for row in state_rows],
    )
    _write_table(
        output / "tables/state_controls.tex",
        ["Audit", "Pair-order / rate"],
        [
            ["Correct history", _fmt(state["interventions"]["correct"])],
            ["Final two only", _fmt(state["interventions"]["last_two_only"])],
            ["Valid pair swap", _fmt(state["swap"]["counterfactual_swap_accuracy"])],
            ["Score-swap consistency", _fmt(state["swap"]["score_swap_consistency"])],
            ["Nondegenerate C20 worlds", _fmt(state["candidate_diversity"]["nondegenerate_world_fraction"])],
            ["C20 dynamic Top-1", _fmt(data["state_ranking_correction"]["rankings"]["C20"]["dynamic"]["top1_success"])],
        ],
    )
    _write_table(
        output / "tables/state_baselines.tex",
        ["Category", "Method", "Val.", "Test"],
        [
            [row["category"], row["method"], _fmt(row["validation"]), _fmt(row["test"])]
            for row in state_baseline_rows
        ],
    )
    _write_table(
        output / "tables/state_family_mechanism.tex",
        ["Family", "Mechanism", "GRU", "Strongest analytic", "Score"],
        [
            [
                row["family"],
                row["mechanism"],
                _fmt(row["learned"]),
                row["analytic_method"],
                _fmt(row["analytic"]),
            ]
            for row in state_cell_rows
        ],
    )
    _write_table(
        output / "tables/state_ranking.tex",
        ["Set", "Contexts", "Mixed", "GRU Top-1", "Static", "Action", "NDCG", "Regret"],
        [
            [
                row["candidates"],
                str(row["contexts"]),
                _fmt(row["mixed_context_fraction"]),
                _fmt(row["dynamic_top1"]),
                _fmt(row["static_top1"]),
                _fmt(row["action_top1"]),
                _fmt(row["dynamic_ndcg"]),
                _fmt(row["dynamic_regret"]),
            ]
            for row in state_ranking_rows
        ],
    )
    _write_table(
        output / "tables/visual_admission.tex",
        ["Family", "Action", "Current", "RGB-D", "World-PC", "Centroid", "Cent. Top-1"],
        [
            [
                row["family"],
                _fmt(row["action_only"]),
                _fmt(row["current_rgbd"]),
                _fmt(row["ordered_rgbd_gru"]),
                _fmt(row["world_pointcloud_gru"]),
                _fmt(row["centroid_velocity"]),
                _fmt(row["centroid_top1"]),
            ]
            for row in visual_rows
        ],
    )
    _write_table(
        output / "tables/wav_external_audit.tex",
        ["Objects", "$N$", "Strongest single", "Paired CNN", "SparseIDM", "Transition rule"],
        [
            [
                str(row["complexity"]),
                str(row["test_count"]),
                f"{_fmt(row['strongest_single_mean'])}$\\pm${_fmt(row['strongest_single_std'])}",
                f"{_fmt(row['paired_mean'])}$\\pm${_fmt(row['paired_std'])}",
                f"{_fmt(row['sparse_idm_mean'])}$\\pm${_fmt(row['sparse_idm_std'])}",
                _fmt(row["transition_rule"]),
            ]
            for row in wav_rows
        ],
    )
    _write_table(
        output / "tables/admission_decision.tex",
        ["Benchmark", "Integrity", "Temporal evidence", "Fair-control test", "Decision"],
        [
            [
                row["benchmark"],
                row["integrity"],
                row["temporal_evidence"],
                row["fair_control"],
                row["decision"],
            ]
            for row in admission_rows
        ],
    )

    tracked = sorted(
        path
        for path in output.rglob("*")
        if path.is_file() and path != output / "manifest.json"
    )
    outputs = [
        {
            "path": str(path.relative_to(output)),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in tracked
    ]
    manifest = {
        "schema": "actmask-icra2027-claim-package-manifest-v1",
        "source_policy": "Sixteen frozen JSON sources plus the ranking raw-score hash; exact byte hashes required.",
        "statistics_policy": "State CI resamples 300 base worlds; the superseded candidate-pair CI and invalid visual CI are not reported.",
        "sources": sources,
        "checks": checks,
        "outputs": outputs,
        "pass": True,
    }
    _write_json(output / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    arguments = parser.parse_args()
    output = arguments.output_dir.resolve()
    manifest = build(output)
    print(
        json.dumps(
            {
                "pass": manifest["pass"],
                "sources": len(manifest["sources"]),
                "checks": len(manifest["checks"]),
                "outputs": len(manifest["outputs"]),
                "output_dir": str(output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
