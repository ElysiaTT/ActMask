from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import yaml

from actmask.experiments.milestone2 import (
    OOD_DOMAIN_BY_AXIS,
    REQUIRED_OOD_AXES,
    SHUFFLED_VARIANTS,
    TRAINED_VARIANTS,
    run_milestone2,
)
from actmask.eval.ranking import RANKING_SCORE_TIE_TOLERANCE, RANKING_TIE_POLICY
from actmask.visualization.milestone2_failures import FAILURE_CATEGORIES


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _tiny_config(output_dir: Path) -> dict:
    return {
        "experiment": {
            "name": "milestone2_test",
            "output_dir": str(output_dir),
            "seeds": [17, 31],
            "device": "cpu",
            "deterministic": True,
            "resume": True,
            "stages": ["prepare", "train", "evaluate", "report"],
        },
        "dataset": {
            "seed": 991,
            "num_points": 32,
            "trajectory_steps": 8,
            "groups": {"train": 6, "val": 4, "test": 4},
            "variants_per_group": 2,
            "ood_groups_per_axis": 2,
            "ood_axes": list(REQUIRED_OOD_AXES),
            "position_noise_std": 0.002,
            "velocity_noise_std": 0.004,
            "temporal_velocity_delay": 0.02,
            "point_dropout_rate": 0.0,
        },
        "model": {
            "action_dim": 8,
            "point_hidden_dim": 8,
            "action_hidden_dim": 6,
            "fusion_hidden_dim": 8,
        },
        "training": {
            "variants": list(TRAINED_VARIANTS),
            "epochs": 1,
            "batch_size": 4,
            "learning_rate": 0.003,
            "weight_decay": 0.0,
            "optimizer": "adamw",
            "compute_pos_weight": True,
            "max_pos_weight": 10.0,
            "num_workers": 0,
            "num_threads": 1,
        },
        "evaluation": {
            "batch_size": 8,
            "num_workers": 0,
            "threshold_grid_size": 51,
            "shuffled_variants": list(SHUFFLED_VARIANTS),
            "baselines": [
                "NoMask",
                "MotionMagnitudeMask",
                "CurrentPositionProximity",
                "FuturePositionProximity",
                "TimeAlignedTrajectoryProximity",
                "GeometricOracle",
            ],
            "oracle_name": "GeometricOracle",
        },
        # Exercise the full evidence path: failures must be selected from the
        # real predictions and rendered for every seed, not merely supported
        # by an isolated visualization unit test.
        "reporting": {
            "save_pr_plots": False,
            "save_failure_visualizations": True,
            "float_precision": 6,
        },
    }


def _load(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def test_production_config_has_exact_five_cpu_seeds_and_required_workload() -> None:
    path = PROJECT_ROOT / "configs/actmask/milestone2_cpu.yaml"
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    assert config["experiment"]["seeds"] == [1301, 2303, 3307, 4327, 5347]
    assert config["experiment"]["device"] == "cpu"
    assert config["experiment"]["output_dir"] == "outputs/actmask/milestone2_evaluation"
    assert config["training"]["variants"] == list(TRAINED_VARIANTS)
    assert config["evaluation"]["shuffled_variants"] == list(SHUFFLED_VARIANTS)
    assert "TimeShuffled" in config["evaluation"]["shuffled_variants"]
    assert {
        "CurrentPositionProximity",
        "FuturePositionProximity",
        "TimeAlignedTrajectoryProximity",
        "GeometricOracle",
    }.issubset(config["evaluation"]["baselines"])
    assert config["evaluation"]["oracle_name"] == "GeometricOracle"
    assert set(config["dataset"]["ood_axes"]) == set(REQUIRED_OOD_AXES)
    assert config["dataset"]["num_points"] == 96
    assert config["dataset"]["temporal_velocity_delay"] == 0.04
    assert config["reporting"]["save_failure_visualizations"] is True


def test_tiny_two_seed_run_is_deterministic_and_writes_complete_schema(tmp_path: Path) -> None:
    output_dir = tmp_path / "milestone2"
    config = _tiny_config(output_dir)
    first = run_milestone2(deepcopy(config), force=True)
    assert first["status"] == "complete"
    assert first["completion_assertions"]["cpu_only"] is True
    assert first["completion_assertions"]["seed_count"] == 2

    required = (
        "run_manifest.json",
        "resolved_config.json",
        "split_manifest.json",
        "scenario_metrics.json",
        "scenario_metrics.csv",
        "counterfactual_metrics.json",
        "counterfactual_metrics.csv",
        "counterfactual_multi_seed_summary.json",
        "counterfactual_multi_seed_summary.csv",
        "action_ranking_metrics.json",
        "action_ranking_metrics.csv",
        "action_ranking_multi_seed_summary.json",
        "action_ranking_multi_seed_summary.csv",
        "id_ood_generalization_gap.json",
        "id_ood_generalization_gap.csv",
        "id_ood_generalization_gap_summary.json",
        "id_ood_generalization_gap_summary.csv",
        "validation_thresholds.json",
        "strongest_fair_baseline_selection.json",
        "feature_ablation.csv",
        "feature_ablation.md",
        "multi_seed_summary.json",
        "multi_seed_summary.csv",
        "summary.md",
    )
    assert all((output_dir / name).is_file() for name in required)
    first_manifest = _load(output_dir / "run_manifest.json")
    summary_text = (output_dir / "summary.md").read_text(encoding="utf-8")
    assert f"Runtime: {first_manifest['runtime_seconds']:.3f} seconds" in summary_text
    assert "GeometricOracle uses exact candidate utility" in summary_text
    for seed in config["experiment"]["seeds"]:
        for variant in TRAINED_VARIANTS:
            variant_dir = output_dir / "runs" / f"seed_{seed}" / variant
            assert (variant_dir / "checkpoint.pt").is_file()
            assert (variant_dir / "training_metrics.json").is_file()
            assert (variant_dir / "validation_threshold.json").is_file()
            assert (variant_dir / "evaluation_metrics.json").is_file()

    split_manifest = _load(output_dir / "split_manifest.json")
    assert split_manifest["group_leakage_check"]["passed"] is True
    assert split_manifest["id"]["train"]["num_groups"] == 6
    assert split_manifest["id"]["validation"]["num_groups"] == 4
    assert split_manifest["id"]["test"]["num_groups"] == 4
    for axis in REQUIRED_OOD_AXES:
        ood_stats = split_manifest["ood"][axis]
        assert ood_stats["domains"] == {
            OOD_DOMAIN_BY_AXIS[axis]: ood_stats["num_samples"]
        }

    scenarios = _load(output_dir / "scenario_metrics.json")["rows"]
    assert {row["seed"] for row in scenarios} == {17, 31}
    assert {row["ood_axis"] for row in scenarios if row["distribution"] == "OOD"} == set(
        REQUIRED_OOD_AXES
    )
    assert all(row["threshold_source"] == "id_validation" for row in scenarios)
    assert any(row["method_kind"] == "oracle_upper_bound_excluded" for row in scenarios)
    first_scenario_rows = deepcopy(scenarios)

    counterfactuals = _load(output_dir / "counterfactual_metrics.json")["rows"]
    assert counterfactuals
    assert all(
        {"matching_accuracy", "matching_pair_count", "matching_margin"}.issubset(row)
        for row in counterfactuals
    )
    counterfactual_summary = _load(output_dir / "counterfactual_multi_seed_summary.json")
    assert counterfactual_summary["seed_count"] == 2
    assert counterfactual_summary["statistics"] == ["mean", "std", "min", "max"]
    assert any(
        row["method_id"] == "Full"
        and row["metric"] == "matching_accuracy"
        and row["seed_count"] == 2
        for row in counterfactual_summary["rows"]
    )

    ranking_payload = _load(output_dir / "action_ranking_metrics.json")
    assert ranking_payload["tie_policy"] == RANKING_TIE_POLICY
    assert ranking_payload["score_tie_tolerance"] == RANKING_SCORE_TIE_TOLERANCE
    assert "candidate_id is display provenance only" in ranking_payload["tie_handling"]
    ranking = ranking_payload["rows"]
    ranking_aggregates = [
        row for row in ranking if row["row_type"] == "action_ranking_aggregate"
    ]
    expected_ranking_methods = {
        *TRAINED_VARIANTS,
        *SHUFFLED_VARIANTS,
        "CurrentPositionProximity",
        "FuturePositionProximity",
        "TimeAlignedTrajectoryProximity",
        "GeometricOracle",
    }
    assert {row["method_id"] for row in ranking_aggregates}.issuperset(
        expected_ranking_methods
    )
    assert {row["seed"] for row in ranking_aggregates} == {17, 31}
    assert all(row["split"] == "id_candidate_ranking" for row in ranking_aggregates)
    assert all(row["ranking_scene_count"] == 4 for row in ranking_aggregates)
    assert all(row["candidate_count"] == 20 for row in ranking_aggregates)
    assert all(
        {
            "top1_successful_action_accuracy",
            "top3_success_recall",
            "pairwise_ranking_accuracy",
            "binary_ranking_auc",
            "oracle_regret",
        }.issubset(row)
        for row in ranking_aggregates
    )

    ranking_summary = _load(output_dir / "action_ranking_multi_seed_summary.json")
    assert ranking_summary["seed_count"] == 2
    assert ranking_summary["statistics"] == ["mean", "std", "min", "max"]
    assert any(
        row["method_id"] == "Full"
        and row["metric"] == "top1_successful_action_accuracy"
        and row["seed_count"] == 2
        for row in ranking_summary["rows"]
    )

    gaps = _load(output_dir / "id_ood_generalization_gap.json")["rows"]
    assert gaps
    assert {row["ood_axis"] for row in gaps} == set(REQUIRED_OOD_AXES)
    assert all(
        {"id_average_precision", "ood_average_precision", "id_minus_ood_average_precision"}
        .issubset(row)
        for row in gaps
    )
    gap_summary = _load(output_dir / "id_ood_generalization_gap_summary.json")
    assert gap_summary["seed_count"] == 2
    assert gap_summary["statistics"] == ["mean", "std", "min", "max"]
    assert any(
        row["method_id"] == "Full"
        and row["metric"] == "id_minus_ood_average_precision"
        and row["seed_count"] == 2
        for row in gap_summary["rows"]
    )

    thresholds = _load(output_dir / "validation_thresholds.json")["rows"]
    for seed in (17, 31):
        full = next(row for row in thresholds if row.get("seed") == seed and row.get("variant") == "Full")
        for shuffled in SHUFFLED_VARIANTS:
            row = next(
                item
                for item in thresholds
                if item.get("seed") == seed and item.get("method_id") == shuffled
            )
            assert row["inherited_from"] == "Full"
            assert row["threshold"] == full["threshold"]

    feature_lines = (output_dir / "feature_ablation.csv").read_text(encoding="utf-8").splitlines()
    assert len(feature_lines) == 1 + len(TRAINED_VARIANTS) + len(SHUFFLED_VARIANTS)
    multi_seed = _load(output_dir / "multi_seed_summary.json")
    assert multi_seed["seed_count"] == 2
    assert multi_seed["statistics"] == ["mean", "std", "min", "max"]
    assert all(
        {"mean", "std", "min", "max", "seed_count"}.issubset(row)
        for row in multi_seed["rows"]
    )
    selections = _load(output_dir / "strongest_fair_baseline_selection.json")
    assert selections["oracle_excluded"] is True
    assert all(row["test_metrics_were_not_used"] is True for row in selections["rows"])

    for seed in config["experiment"]["seeds"]:
        seed_dir = output_dir / "runs" / f"seed_{seed}"
        seed_ranking = _load(seed_dir / "action_ranking_metrics.json")
        assert seed_ranking["seed"] == seed
        assert seed_ranking["tie_policy"] == RANKING_TIE_POLICY
        assert seed_ranking["score_tie_tolerance"] == RANKING_SCORE_TIE_TOLERANCE
        assert any(
            row["row_type"] == "action_ranking_aggregate"
            and row["method_id"] == "Full"
            for row in seed_ranking["rows"]
        )

        run_metrics = _load(seed_dir / "run_metrics.json")
        failure_artifacts = run_metrics["failure_artifacts"]
        assert failure_artifacts is not None
        manifest_path = Path(failure_artifacts["manifest"])
        assert manifest_path.is_file()
        failure_manifest = _load(manifest_path)
        assert failure_manifest["metadata"]["seed"] == seed
        assert [row["category"] for row in failure_manifest["selections"]] == list(
            FAILURE_CATEGORIES
        )
        assert all(isinstance(row["qualifying"], bool) for row in failure_manifest["selections"])
        for row in failure_manifest["selections"]:
            figure = Path(row["artifact_path"])
            assert figure.is_file()
            assert figure.stat().st_size > 500

    # Derived report artifacts must be reproducible from raw evaluation data.
    # This catches report-only implementations that accidentally omit ranking
    # rows or ID--OOD gaps after a process restart.
    for name in (
        "counterfactual_multi_seed_summary.json",
        "counterfactual_multi_seed_summary.csv",
        "action_ranking_multi_seed_summary.json",
        "action_ranking_multi_seed_summary.csv",
        "id_ood_generalization_gap.json",
        "id_ood_generalization_gap.csv",
        "id_ood_generalization_gap_summary.json",
        "id_ood_generalization_gap_summary.csv",
    ):
        (output_dir / name).unlink()
    report_only = run_milestone2(deepcopy(config), stages=("report",))
    assert report_only["status"] == "stage_complete"
    assert all((output_dir / name).is_file() for name in required)
    report_manifest = _load(output_dir / "run_manifest.json")
    assert f"Runtime: {report_manifest['runtime_seconds']:.3f} seconds" in (
        output_dir / "summary.md"
    ).read_text(encoding="utf-8")
    assert _load(output_dir / "action_ranking_metrics.json")["rows"] == ranking

    # Force a second from-scratch execution with the same seeds/data. Runtime
    # metadata may differ, but every computed metric row must be bit-identical.
    second = run_milestone2(deepcopy(config), force=True)
    assert second["status"] == "complete"
    assert _load(output_dir / "scenario_metrics.json")["rows"] == first_scenario_rows
