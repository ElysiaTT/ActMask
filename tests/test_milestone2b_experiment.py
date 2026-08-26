"""Fast CPU integration tests for the Milestone 2B experiment pipeline."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import yaml

from actmask.data.milestone2b_dataset import MILESTONE2B_OOD_AXES
from actmask.experiments.milestone2b import (
    ABLATION_METHODS,
    LEARNED_VARIANTS,
    _load_config,
    build_datasets,
    evaluate_seed,
    mask_metrics,
    predict_learned,
    train_variant,
)
from actmask.models.milestone2b_baselines import FAIR_BASELINES


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _tiny_config(output_dir: Path) -> dict[str, Any]:
    """Use complete 11-sample causal groups while keeping runtime very small."""

    return {
        "experiment": {
            "name": "milestone2b_test",
            "output_dir": str(output_dir),
            "seeds": [11, 13, 17, 19, 23],
            "device": "cpu",
            "deterministic": True,
        },
        "dataset": {
            "seed": 731,
            "observation_frames": 4,
            "future_steps": 32,
            "num_points": 48,
            "groups": {"train": 2, "val": 1, "test": 1},
            "ood_groups_per_axis": 1,
            "observation_noise_std": 0.006,
            "point_dropout_rate": 0.04,
            "occlusion_rate": 0.08,
        },
        "model": {
            "history_hidden_dim": 8,
            "action_hidden_dim": 6,
            "fusion_hidden_dim": 12,
            "contact_hidden_dim": 8,
            "utility_hidden_dim": 8,
        },
        "training": {
            "epochs": 1,
            "batch_size": 22,
            "learning_rate": 0.003,
            "weight_decay": 0.0,
            "max_pos_weight": 12.0,
            "num_threads": 1,
            "loss_coefficients": {
                "mask_bce": 1.0,
                "contact_time_bce": 0.20,
                "success_bce": 0.45,
                "pairwise_ranking": 0.30,
                "velocity_counterfactual": 0.25,
                "action_counterfactual": 0.25,
                "irrelevant_stability": 0.10,
            },
        },
        "evaluation": {
            "batch_size": 11,
            "primary_model": "TemporalActMask",
            "previous_full_top1": 0.50,
            "gates": {
                "action_ap_drop": 0.05,
                "history_removal_ap_drop": 0.05,
                "history_shuffle_ap_drop": 0.05,
                "velocity_counterfactual_accuracy": 0.80,
                "action_counterfactual_accuracy": 0.80,
                "irrelevant_stability": 0.85,
                "top1_improvement": 0.15,
            },
        },
    }


@pytest.fixture(scope="module")
def tiny_evaluation(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    root = tmp_path_factory.mktemp("milestone2b_experiment")
    output_dir = root / "outputs" / "actmask" / "milestone2b"
    seed_dir = output_dir / "runs" / "seed_11"
    config = _tiny_config(output_dir)
    datasets = build_datasets(config)
    models = {}
    training = {}
    for variant in LEARNED_VARIANTS:
        models[variant], training[variant] = train_variant(
            variant=variant,
            seed=11,
            config=config,
            datasets=datasets,
            seed_dir=seed_dir,
        )
    evaluation = evaluate_seed(
        seed=11,
        config=config,
        datasets=datasets,
        models=models,
        seed_dir=seed_dir,
    )
    return {
        "root": root,
        "output_dir": output_dir,
        "seed_dir": seed_dir,
        "config": config,
        "datasets": datasets,
        "models": models,
        "training": training,
        "evaluation": evaluation,
    }


def test_canonical_config_requires_five_unique_cpu_seeds(tmp_path: Path) -> None:
    production_path = PROJECT_ROOT / "configs" / "actmask" / "milestone2b_cpu.yaml"
    config = _load_config(production_path)
    assert config["experiment"]["seeds"] == [1401, 2403, 3407, 4411, 5413]
    assert len(set(config["experiment"]["seeds"])) == 5
    assert config["experiment"]["device"] == "cpu"
    assert config["experiment"]["output_dir"] == "outputs/actmask/milestone2b"
    assert "milestone2_evaluation" not in config["experiment"]["output_dir"]

    invalid = deepcopy(config)
    invalid["experiment"]["seeds"] = [2, 3, 5, 7]
    invalid_path = tmp_path / "invalid_four_seed.yaml"
    invalid_path.write_text(yaml.safe_dump(invalid), encoding="utf-8")
    with pytest.raises(ValueError, match="at least five unique seeds"):
        _load_config(invalid_path)


def test_tiny_training_and_full_seed_evaluation_run_on_cpu(
    tiny_evaluation: dict[str, Any],
) -> None:
    evaluation = tiny_evaluation["evaluation"]
    seed_dir = tiny_evaluation["seed_dir"]
    datasets = tiny_evaluation["datasets"]

    assert set(tiny_evaluation["models"]) == set(LEARNED_VARIANTS)
    assert all(
        next(model.parameters()).device.type == "cpu"
        for model in tiny_evaluation["models"].values()
    )
    assert {row["dataset"] for row in evaluation["mask_rows"]} == {
        "test",
        *(f"ood/{axis}" for axis in MILESTONE2B_OOD_AXES),
    }
    expected_methods = {
        *ABLATION_METHODS,
        *FAIR_BASELINES,
        "ExactHiddenTrajectoryOracle",
    }
    assert {row["method"] for row in evaluation["mask_rows"]} == expected_methods
    assert all(len(dataset) in {11, 22} for dataset in datasets.values())

    for variant in LEARNED_VARIANTS:
        assert (seed_dir / "checkpoints" / f"{variant}.pt").is_file()
        assert (seed_dir / "training" / f"{variant}.csv").is_file()
    assert (seed_dir / "thresholds.json").is_file()
    for name in (
        "mask_metrics.csv",
        "scenario_metrics.csv",
        "condition_metrics.csv",
        "counterfactual_metrics.csv",
        "hard_case_metrics.csv",
        "ranking_metrics.csv",
    ):
        assert (seed_dir / "evaluation" / name).is_file()


def test_retraining_same_seed_is_prediction_and_metric_deterministic(
    tiny_evaluation: dict[str, Any], tmp_path: Path
) -> None:
    config = tiny_evaluation["config"]
    datasets = tiny_evaluation["datasets"]
    first_model = tiny_evaluation["models"]["TemporalActMask"]
    second_model, second_metadata = train_variant(
        variant="TemporalActMask",
        seed=11,
        config=config,
        datasets=datasets,
        seed_dir=tmp_path / "repeat_seed_11",
    )

    for name, first_value in first_model.state_dict().items():
        assert torch.equal(first_value, second_model.state_dict()[name])
    assert second_metadata["best_validation_ap"] == pytest.approx(
        tiny_evaluation["training"]["TemporalActMask"]["best_validation_ap"],
        abs=0.0,
    )

    first = predict_learned(first_model, datasets["test"], batch_size=11)
    second = predict_learned(second_model, datasets["test"], batch_size=11)
    assert np.array_equal(first.probabilities, second.probabilities)
    assert np.array_equal(first.utility, second.utility)
    threshold = tiny_evaluation["evaluation"]["thresholds"]["TemporalActMask"]
    assert mask_metrics(first, threshold) == mask_metrics(second, threshold)


def test_manifests_are_disjoint_and_writes_stay_out_of_milestone2_outputs(
    tiny_evaluation: dict[str, Any]
) -> None:
    partitions = {
        name: {int(record.group_id) for record in dataset.group_records}
        for name, dataset in tiny_evaluation["datasets"].items()
    }
    names = sorted(partitions)
    for index, first in enumerate(names):
        for second in names[index + 1 :]:
            assert partitions[first].isdisjoint(partitions[second])

    root = tiny_evaluation["root"]
    assert tiny_evaluation["output_dir"].name == "milestone2b"
    assert not (root / "outputs" / "actmask" / "milestone2_evaluation").exists()
    assert all("milestone2_evaluation" not in path.parts for path in root.rglob("*"))
