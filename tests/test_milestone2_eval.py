"""Exact CPU tests for Milestone-2 precision/recall evaluation."""

from __future__ import annotations

import math
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from actmask.eval.pr_metrics import (
    apply_probability_threshold,
    average_precision,
    pr_auc,
    precision_recall_curve,
    select_validation_threshold,
)
from actmask.eval.aggregation import aggregate_scenario_metrics
from actmask.eval.counterfactual import aggregate_counterfactual_metrics
from actmask.eval.prediction import PredictionRecord, collect_predictions
from actmask.eval.reporting import (
    seed_summary_records,
    write_csv_records,
    write_json_report,
    write_summary_markdown,
)
from actmask.models.ablations import (
    AblatedActMask,
    ShuffledActionDataset,
    ShuffledVelocityDataset,
    deterministic_derangement,
)
from actmask.models.actmask_v2 import ActMaskV2


def test_precision_recall_curve_ap_and_auc_are_exact() -> None:
    # Descending labels are 1, 0, 1, 0. Precision at the two positive recall
    # steps is 1 and 2/3, so AP = (1/2)*1 + (1/2)*(2/3) = 5/6.
    scores = torch.tensor([0.9, 0.8, 0.7, 0.1], dtype=torch.float32)
    targets = torch.tensor([1, 0, 1, 0], dtype=torch.int64)
    curve = precision_recall_curve(scores, targets)

    assert curve.thresholds == pytest.approx([0.9, 0.8, 0.7, 0.1])
    assert curve.recall == pytest.approx([0.0, 0.5, 0.5, 1.0, 1.0])
    assert curve.precision == pytest.approx([1.0, 1.0, 0.5, 2.0 / 3.0, 0.5])
    assert curve.average_precision == pytest.approx(5.0 / 6.0)
    assert curve.pr_auc == pytest.approx(19.0 / 24.0)
    assert curve.has_positive_labels is True
    assert average_precision(scores, targets) == pytest.approx(5.0 / 6.0)
    assert pr_auc(scores, targets) == pytest.approx(19.0 / 24.0)


def test_tied_scores_are_grouped_and_input_order_independent() -> None:
    first = precision_recall_curve([0.8, 0.8, 0.2], [1, 0, 1])
    reordered = precision_recall_curve([0.8, 0.2, 0.8], [0, 1, 1])

    assert first.thresholds == pytest.approx([0.8, 0.2])
    assert first.precision == pytest.approx([1.0, 0.5, 2.0 / 3.0])
    assert first.recall == pytest.approx([0.0, 0.5, 1.0])
    assert first.average_precision == pytest.approx(7.0 / 12.0)
    assert np.array_equal(first.thresholds, reordered.thresholds)
    assert np.array_equal(first.precision, reordered.precision)
    assert np.array_equal(first.recall, reordered.recall)


def test_pr_edge_cases_are_explicit() -> None:
    no_positives = precision_recall_curve([0.9, 0.1], [0, 0])
    assert no_positives.has_positive_labels is False
    assert no_positives.average_precision == 0.0
    assert no_positives.pr_auc == 0.0
    assert no_positives.recall == pytest.approx([0.0, 0.0, 0.0])

    all_positives = precision_recall_curve([0.7, 0.3], [1, 1])
    assert all_positives.has_positive_labels is True
    assert all_positives.average_precision == pytest.approx(1.0)
    assert all_positives.pr_auc == pytest.approx(1.0)


def test_validation_threshold_has_exact_tie_break_and_is_frozen_for_test() -> None:
    # Thresholds 0.4 and 0.6 both give perfect validation F1. They are equally
    # distant from 0.5, so the documented final tie break selects 0.6.
    validation_scores = [0.8, 0.6, 0.4, 0.2]
    validation_targets = [1, 1, 0, 0]
    selected = select_validation_threshold(validation_scores, validation_targets)

    assert selected.threshold == pytest.approx(0.6)
    assert selected.validation_f1 == pytest.approx(1.0)
    assert selected.validation_iou == pytest.approx(1.0)
    assert (selected.tp, selected.fp, selected.fn, selected.tn) == (2, 0, 0, 2)
    assert selected.selection_split == "validation"
    assert selected.tie_break == "closest_to_0.5_then_higher"

    test_scores = [0.7, 0.5, 0.1]
    first_test_predictions = apply_probability_threshold(test_scores, selected.threshold)
    # Test targets are never an argument to threshold selection or application;
    # changing them therefore cannot refit or alter the frozen predictions.
    changed_test_targets = [0, 0, 1]
    assert changed_test_targets != [1, 0, 0]
    second_test_predictions = apply_probability_threshold(test_scores, selected.threshold)
    assert first_test_predictions.tolist() == [True, False, False]
    assert np.array_equal(first_test_predictions, second_test_predictions)


@pytest.mark.parametrize(
    ("scores", "targets", "match"),
    [
        ([], [], "cannot be empty"),
        ([0.2, 1.1], [0, 1], r"scores must lie in \[0, 1\]"),
        ([0.2, 0.8], [0, 2], "targets must be binary"),
        ([0.2], [0, 1], "same number"),
    ],
)
def test_pr_inputs_are_validated(scores: list[float], targets: list[int], match: str) -> None:
    with pytest.raises((ValueError, TypeError), match=match):
        precision_recall_curve(scores, targets)


def _v2_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    points = torch.linspace(0.05, 0.95, 2 * 7 * 3).reshape(2, 7, 3)
    velocities = torch.linspace(0.01, 0.41, 2 * 7 * 3).reshape(2, 7, 3)
    action = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.8, 0.2, 0.1, 1.0, 0.09],
            [0.2, 0.1, 0.0, 0.4, 0.9, 0.2, 0.7, 0.12],
        ],
        dtype=torch.float32,
    )
    return points, velocities, action


def test_actmask_v2_branch_embeddings_are_exactly_ablated() -> None:
    model = ActMaskV2(
        action_dim=8,
        point_hidden_dim=12,
        action_hidden_dim=8,
        fusion_hidden_dim=16,
    ).cpu()
    # Positive parameters and inputs make all active ReLU branch embeddings
    # nonzero, including biased encoders. Disabled embeddings must nevertheless
    # be exact zeros because masking happens after/around the complete branch.
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(0.05)
    points, velocities, action = _v2_inputs()

    full = model.encode_branches(points, velocities, action, "full")
    no_velocity = model.encode_branches(points, velocities, action, "no_velocity")
    no_action = model.encode_branches(points, velocities, action, "no_action")
    position_only = model.encode_branches(points, velocities, action, "position_only")

    assert torch.count_nonzero(full["velocity"]) > 0
    assert torch.count_nonzero(full["action_relative"]) > 0
    assert torch.count_nonzero(no_velocity["velocity"]) == 0
    assert torch.count_nonzero(no_action["action_relative"]) == 0
    assert torch.count_nonzero(position_only["velocity"]) == 0
    assert torch.count_nonzero(position_only["action_relative"]) == 0
    for branches in (no_velocity, no_action, position_only):
        assert torch.equal(branches["position"], full["position"])
    assert torch.equal(no_velocity["action_relative"], full["action_relative"])
    assert torch.equal(no_action["velocity"], full["velocity"])

    for feature_mode in ("full", "no_velocity", "no_action", "position_only"):
        logits = model(points, velocities, action, feature_mode=feature_mode)
        assert logits.shape == (2, 7)
        assert logits.device.type == "cpu"
        assert torch.isfinite(logits).all()


def test_actmask_v2_disabled_features_cannot_affect_outputs() -> None:
    torch.manual_seed(3)
    model = ActMaskV2(
        action_dim=8,
        point_hidden_dim=10,
        action_hidden_dim=6,
        fusion_hidden_dim=12,
        use_global_context=True,
    ).eval()
    points, velocities, action = _v2_inputs()
    changed_velocity = velocities + 10.0
    changed_action = action.clone()
    changed_action[:, :6] += 5.0
    changed_action[:, 6:] *= 2.0

    with torch.inference_mode():
        no_velocity_a = model(points, velocities, action, "no_velocity")
        no_velocity_b = model(points, changed_velocity, action, "no_velocity")
        no_action_a = model(points, velocities, action, "no_action")
        no_action_b = model(points, velocities, changed_action, "no_action")
        position_a = model(points, velocities, action, "position_only")
        position_b = model(points, changed_velocity, changed_action, "position_only")

    assert torch.equal(no_velocity_a, no_velocity_b)
    assert torch.equal(no_action_a, no_action_b)
    assert torch.equal(position_a, position_b)

    wrapped = AblatedActMask(model, "no-action")
    with torch.inference_mode():
        assert torch.equal(
            wrapped(points, velocities, action),
            model(points, velocities, action, "no_action"),
        )


class _ShuffleFixture(Dataset[dict[str, object]]):
    def __init__(self, count: int = 7) -> None:
        self.samples: list[dict[str, object]] = []
        for index in range(count):
            self.samples.append(
                {
                    "points": torch.full((4, 3), float(index)),
                    "velocities": torch.full((4, 3), float(index + 10)),
                    "action": torch.full((8,), float(index + 20)),
                    "mask": torch.tensor([index % 2, 0, 1, 0], dtype=torch.float32),
                    "sample_id": torch.tensor(1000 + index, dtype=torch.int64),
                    "pair_id": torch.tensor(2000 + index, dtype=torch.int64),
                    "scenario": f"scenario-{index}",
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        return {
            key: value.clone() if isinstance(value, torch.Tensor) else value
            for key, value in self.samples[index].items()
        }


def _batched_donor_ids(dataset: Dataset[dict[str, object]], batch_size: int) -> list[int]:
    result: list[int] = []
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        result.extend(int(value) for value in batch["shuffle_donor_sample_id"].tolist())
    return result


def test_global_derangement_is_deterministic_and_batch_size_independent() -> None:
    first = deterministic_derangement(7, seed=91)
    second = deterministic_derangement(7, seed=91)
    assert first == second
    assert sorted(first) == list(range(7))
    assert all(recipient != donor for recipient, donor in enumerate(first))

    base = _ShuffleFixture()
    shuffled = ShuffledVelocityDataset(base, seed=91)
    assert shuffled.donor_indices == first
    assert _batched_donor_ids(shuffled, batch_size=1) == _batched_donor_ids(
        shuffled, batch_size=4
    )


def test_shuffle_replaces_only_requested_input_and_records_donor() -> None:
    base = _ShuffleFixture()
    velocity_view = ShuffledVelocityDataset(base, seed=17)
    action_view = ShuffledActionDataset(base, seed=17)

    for index, donor_index in enumerate(velocity_view.donor_indices):
        original = base[index]
        donor = base[donor_index]
        velocity_sample = velocity_view[index]
        action_sample = action_view[index]

        assert torch.equal(velocity_sample["velocities"], donor["velocities"])
        assert torch.equal(velocity_sample["action"], original["action"])
        assert torch.equal(action_sample["action"], donor["action"])
        assert torch.equal(action_sample["velocities"], original["velocities"])
        for key in ("points", "mask", "sample_id", "pair_id"):
            assert torch.equal(velocity_sample[key], original[key])
            assert torch.equal(action_sample[key], original[key])
        assert velocity_sample["scenario"] == original["scenario"]
        assert action_sample["scenario"] == original["scenario"]
        assert velocity_sample["velocity_donor_sample_id"] == int(donor["sample_id"])
        assert action_sample["action_donor_sample_id"] == int(donor["sample_id"])

    # Accessing either view must not mutate the underlying dataset.
    assert torch.equal(base[0]["velocities"], torch.full((4, 3), 10.0))
    assert torch.equal(base[0]["action"], torch.full((8,), 20.0))


class _RecordedPredictionDataset(Dataset[dict[str, object]]):
    def __init__(self) -> None:
        rows = [
            ("alpha", 10, 0, 1.0, [0.9, 0.2], [1.0, 0.0]),
            ("alpha", 10, 1, 0.0, [0.4, 0.8], [0.0, 1.0]),
            ("beta", 11, 0, 0.0, [0.6, 0.3], [0.0, 0.0]),
            ("beta", 11, 1, 0.0, [0.2, 0.7], [0.0, 1.0]),
        ]
        self.samples: list[dict[str, object]] = []
        for index, (scenario, pair_id, variant_id, success, probabilities, targets) in enumerate(rows):
            probability_tensor = torch.tensor(probabilities, dtype=torch.float32)
            points = torch.zeros((2, 3), dtype=torch.float32)
            points[:, 0] = probability_tensor
            self.samples.append(
                {
                    "points": points,
                    "velocities": torch.full((2, 3), float(index), dtype=torch.float32),
                    "action": torch.tensor(
                        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.1],
                        dtype=torch.float32,
                    ),
                    "mask": torch.tensor(targets, dtype=torch.float32),
                    "success": torch.tensor(success, dtype=torch.float32),
                    "sample_id": torch.tensor(100 + index, dtype=torch.int64),
                    "group_id": torch.tensor(200 + pair_id, dtype=torch.int64),
                    "base_scene_id": torch.tensor(300 + pair_id, dtype=torch.int64),
                    "geometry_seed": torch.tensor(400 + pair_id, dtype=torch.int64),
                    "scenario_seed": torch.tensor(500 + pair_id, dtype=torch.int64),
                    "pair_id": torch.tensor(pair_id, dtype=torch.int64),
                    "variant_id": torch.tensor(variant_id, dtype=torch.int64),
                    "scenario": scenario,
                    "counterfactual_type": "action" if scenario == "alpha" else "velocity",
                    "domain": "id",
                    "ood_axis": "none",
                    "trajectory_family": "linear" if scenario == "alpha" else "curved",
                    "parameters": torch.tensor([float(index), 0.25], dtype=torch.float32),
                    "changed_input_mask": torch.tensor([False, index % 2 == 0]),
                    "invariant_mask": torch.tensor([True, index % 2 != 0]),
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        return {
            key: value.clone() if isinstance(value, torch.Tensor) else value
            for key, value in self.samples[index].items()
        }


class _PointProbabilityModel(torch.nn.Module):
    def forward(
        self, points: torch.Tensor, velocities: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        del velocities, action
        probabilities = points[..., 0].clamp(1.0e-5, 1.0 - 1.0e-5)
        return torch.logit(probabilities)


def test_prediction_collection_preserves_cpu_outputs_and_m2_metadata() -> None:
    dataset = _RecordedPredictionDataset()
    model = _PointProbabilityModel().train()
    records = collect_predictions(
        model,
        DataLoader(dataset, batch_size=3, shuffle=False),
        device="cpu",
        distribution="ID",
        retain_inputs=True,
    )

    assert len(records) == 4
    assert model.training is True  # Collection restores the caller's mode.
    first = records[0]
    assert first.probabilities.device.type == "cpu"
    assert first.probabilities == pytest.approx([0.9, 0.2], abs=1.0e-6)
    assert first.targets.dtype == torch.bool
    assert first.targets.tolist() == [True, False]
    assert first.sample_id == 100
    assert first.group_id == 210
    assert first.base_scene_id == 310
    assert first.geometry_seed == 410
    assert first.scenario_seed == 510
    assert first.pair_id == 10
    assert first.variant_id == 0
    assert first.scenario == "alpha"
    assert first.counterfactual_type == "action"
    assert first.distribution == "ID"
    assert first.ood_axis == "none"
    assert first.changed_input_mask.tolist() == [False, True]
    assert first.invariant_mask.tolist() == [True, False]
    assert first.metadata["trajectory_family"] == "linear"
    assert torch.equal(first.metadata["parameters"], torch.tensor([0.0, 0.25]))
    assert first.points is not None and first.velocities is not None and first.action is not None

    # Dataset and DataLoader entry paths retain identical, batch-independent records.
    direct = collect_predictions(model, dataset, batch_size=1)
    assert [record.sample_id for record in direct] == [record.sample_id for record in records]
    for expected, actual in zip(records, direct, strict=True):
        assert torch.allclose(expected.probabilities, actual.probabilities)
        assert torch.equal(expected.targets, actual.targets)


def test_scenario_aggregation_uses_global_counts_and_frozen_threshold() -> None:
    records = collect_predictions(_PointProbabilityModel(), _RecordedPredictionDataset(), batch_size=2)
    rows = aggregate_scenario_metrics(
        records,
        threshold=0.5,
        method_id="full",
        method_kind="learned",
        seed=2026,
        split="id_test",
        threshold_source="id_validation",
        training_variant="full",
    )

    assert [row["scenario"] for row in rows] == ["__all__", "alpha", "beta"]
    global_row, alpha, beta = rows
    assert (global_row["tp"], global_row["fp"], global_row["fn"], global_row["tn"]) == (
        3,
        1,
        0,
        4,
    )
    assert global_row["precision"] == pytest.approx(0.75)
    assert global_row["recall"] == pytest.approx(1.0)
    assert global_row["f1"] == pytest.approx(6.0 / 7.0)
    assert global_row["iou"] == pytest.approx(0.75)
    assert global_row["positive_prevalence"] == pytest.approx(3.0 / 8.0)
    assert global_row["average_precision"] == pytest.approx(1.0)
    assert global_row["pr_auc"] == pytest.approx(1.0)
    assert global_row["samples"] == 4
    assert global_row["points"] == 8
    assert global_row["positives"] == 3
    assert global_row["pair_count"] == 2
    assert global_row["paired_consistency"] == pytest.approx(0.75)
    assert global_row["ranking_pair_count"] == 1
    assert global_row["success_ranking_accuracy"] == pytest.approx(0.0)
    assert global_row["threshold"] == pytest.approx(0.5)
    assert global_row["threshold_source"] == "id_validation"
    assert global_row["method_id"] == "full"
    assert global_row["seed"] == 2026

    assert (alpha["tp"], alpha["fp"], alpha["fn"], alpha["tn"]) == (2, 0, 0, 2)
    assert alpha["paired_consistency"] == pytest.approx(1.0)
    assert (beta["tp"], beta["fp"], beta["fn"], beta["tn"]) == (1, 1, 0, 2)
    assert beta["paired_consistency"] == pytest.approx(0.5)

    # A different supplied (already-frozen) threshold changes test counts; the
    # aggregator never sees validation or test labels for threshold fitting.
    high_threshold = aggregate_scenario_metrics(records, 0.75)[0]
    assert high_threshold["threshold"] == pytest.approx(0.75)
    assert (high_threshold["tp"], high_threshold["fp"], high_threshold["fn"]) == (2, 0, 1)


def test_prediction_record_validates_point_shapes_and_probability_range() -> None:
    with pytest.raises(ValueError, match="identical point shapes"):
        PredictionRecord(probabilities=torch.ones(2), targets=torch.ones(3))
    with pytest.raises(ValueError, match=r"lie in \[0, 1\]"):
        PredictionRecord(probabilities=torch.tensor([1.2]), targets=torch.tensor([1]))


def _counterfactual_record(
    probabilities: list[float],
    targets: list[int],
    *,
    sample_id: int,
    pair_id: int,
    variant_id: int,
    counterfactual_type: str,
    scenario: str,
    changed: list[bool] | None = None,
    invariant: list[bool] | None = None,
    metadata: dict[str, object] | None = None,
) -> PredictionRecord:
    return PredictionRecord(
        probabilities=torch.tensor(probabilities),
        targets=torch.tensor(targets),
        sample_id=sample_id,
        pair_id=pair_id,
        variant_id=variant_id,
        scenario=scenario,
        counterfactual_type=counterfactual_type,
        distribution="ID",
        ood_axis="none",
        changed_input_mask=torch.tensor(changed) if changed is not None else None,
        invariant_mask=torch.tensor(invariant) if invariant is not None else None,
        metadata={} if metadata is None else dict(metadata),
    )


@pytest.mark.parametrize(
    ("counterfactual_type", "scenario"),
    [
        ("velocity", "velocity_counterfactual"),
        ("action", "action_counterfactual"),
        ("action_timing", "timing_counterfactual"),
        ("future_dynamics", "constant_acceleration"),
    ],
)
def test_response_counterfactual_metrics_are_exact(
    counterfactual_type: str, scenario: str
) -> None:
    records = [
        _counterfactual_record(
            [0.9, 0.2, 0.4, 0.1],
            [1, 0, 0, 0],
            sample_id=1,
            pair_id=50,
            variant_id=0,
            counterfactual_type=counterfactual_type,
            scenario=scenario,
        ),
        _counterfactual_record(
            [0.2, 0.8, 0.45, 0.1],
            [0, 1, 0, 0],
            sample_id=2,
            pair_id=50,
            variant_id=1,
            counterfactual_type=counterfactual_type,
            scenario=scenario,
        ),
    ]
    rows = aggregate_counterfactual_metrics(
        records, 0.5, method_id="full", seed=2026, training_variant="full"
    )
    assert [(row["counterfactual_type"], row["scenario"]) for row in rows] == [
        (counterfactual_type, "__all__"),
        (counterfactual_type, scenario),
    ]
    row = rows[0]
    assert row["pair_count"] == 1
    assert row["change_iou"] == pytest.approx(1.0)
    assert row["change_ap"] == pytest.approx(1.0)
    assert row["signed_direction_accuracy"] == pytest.approx(1.0)
    assert row["ground_truth_changed_points"] == 2
    assert row["changed_region_probability_delta"] == pytest.approx(0.65)
    assert row["invariant_points"] == 2
    assert row["invariant_probability_mae"] == pytest.approx(0.025, abs=1e-7)
    assert row["invariant_probability_stability"] == pytest.approx(0.975, abs=1e-7)
    assert row["invariant_binary_agreement"] == pytest.approx(1.0)
    assert row["matching_pair_count"] == 1
    assert row["matching_accuracy"] == pytest.approx(1.0)

    def log_similarity(probabilities: list[float], targets: list[int]) -> float:
        return sum(
            math.log(probability) if target else math.log1p(-probability)
            for probability, target in zip(probabilities, targets, strict=True)
        ) / len(probabilities)

    first_probabilities = [0.9, 0.2, 0.4, 0.1]
    second_probabilities = [0.2, 0.8, 0.45, 0.1]
    first_targets = [1, 0, 0, 0]
    second_targets = [0, 1, 0, 0]
    expected_margin = 0.5 * (
        log_similarity(first_probabilities, first_targets)
        + log_similarity(second_probabilities, second_targets)
        - log_similarity(first_probabilities, second_targets)
        - log_similarity(second_probabilities, first_targets)
    )
    assert row["matching_margin"] == pytest.approx(expected_margin, abs=1e-7)


def test_counterfactual_matching_penalizes_swapped_prediction_assignments() -> None:
    records = [
        _counterfactual_record(
            [0.2, 0.8, 0.45, 0.1],
            [1, 0, 0, 0],
            sample_id=1,
            pair_id=70,
            variant_id=0,
            counterfactual_type="velocity",
            scenario="velocity_counterfactual",
        ),
        _counterfactual_record(
            [0.9, 0.2, 0.4, 0.1],
            [0, 1, 0, 0],
            sample_id=2,
            pair_id=70,
            variant_id=1,
            counterfactual_type="velocity",
            scenario="velocity_counterfactual",
        ),
    ]
    row = aggregate_counterfactual_metrics(records, 0.5)[0]
    assert row["matching_pair_count"] == 1
    assert row["matching_accuracy"] == pytest.approx(0.0)
    assert row["matching_margin"] < 0.0


def test_irrelevant_background_metrics_pair_preserved_sample_ids() -> None:
    original = _counterfactual_record(
        [0.8, 0.2, 0.4, 0.1],
        [1, 0, 0, 0],
        sample_id=9,
        pair_id=60,
        variant_id=0,
        counterfactual_type="action",
        scenario="near_miss_action",
    )
    changed = _counterfactual_record(
        [0.75, 0.6, 0.45, 0.1],
        [1, 0, 0, 0],
        sample_id=9,
        pair_id=60,
        variant_id=0,
        counterfactual_type="irrelevant_background",
        scenario="near_miss_action",
        changed=[False, True, False, False],
        invariant=[True, False, True, True],
    )
    rows = aggregate_counterfactual_metrics([original, changed], 0.5, method_id="full")
    assert len(rows) == 2
    row = rows[0]
    assert row["counterfactual_type"] == "irrelevant_background"
    assert row["pair_count"] == 1
    assert row["change_iou"] is None
    assert row["invariant_points"] == 3
    assert row["invariant_probability_mae"] == pytest.approx(1.0 / 30.0, abs=1e-7)
    assert row["invariant_probability_stability"] == pytest.approx(29.0 / 30.0, abs=1e-7)
    assert row["invariant_binary_agreement"] == pytest.approx(1.0)
    assert row["invariant_prediction_iou"] == pytest.approx(1.0)
    # One of the two predictions on the changed negative background point is positive.
    assert row["changed_background_false_positive_rate"] == pytest.approx(0.5)
    # Identical ground truth makes correct and swapped assignments
    # indistinguishable by design, so the prescribed tie score is 0.5.
    assert row["matching_pair_count"] == 1
    assert row["matching_accuracy"] == pytest.approx(0.5)
    assert row["matching_margin"] == pytest.approx(0.0, abs=1e-12)

    invalid = _counterfactual_record(
        [0.75, 0.6, 0.45, 0.1],
        [1, 1, 0, 0],
        sample_id=9,
        pair_id=60,
        variant_id=0,
        counterfactual_type="irrelevant_background",
        scenario="near_miss_action",
        changed=[False, True, False, False],
        invariant=[True, False, True, True],
    )
    with pytest.raises(ValueError, match="must preserve ground truth"):
        aggregate_counterfactual_metrics([original, invalid], 0.5)


def test_irrelevant_background_pairs_by_source_sample_id_metadata() -> None:
    """A materialized perturbation may have a new sample ID but the same source."""

    original = _counterfactual_record(
        [0.8, 0.2, 0.4, 0.1],
        [1, 0, 0, 0],
        sample_id=901,
        pair_id=61,
        variant_id=0,
        counterfactual_type="irrelevant_background",
        scenario="irrelevant_background_change",
        changed=[False, False, False, False],
        invariant=[True, True, True, True],
        metadata={"source_sample_id": 77},
    )
    changed = _counterfactual_record(
        [0.75, 0.6, 0.45, 0.1],
        [1, 0, 0, 0],
        sample_id=902,
        pair_id=61,
        variant_id=1,
        counterfactual_type="irrelevant_background",
        scenario="irrelevant_background_change",
        changed=[False, True, False, False],
        invariant=[True, False, True, True],
        metadata={"source_sample_id": 77},
    )

    rows = aggregate_counterfactual_metrics([original, changed], 0.5)
    assert len(rows) == 2
    row = rows[0]
    assert row["counterfactual_type"] == "irrelevant_background"
    assert row["pair_count"] == 1
    assert row["matching_pair_count"] == 1
    assert row["matching_accuracy"] == pytest.approx(0.5)
    assert row["matching_margin"] == pytest.approx(0.0, abs=1e-12)


def test_counterfactual_matching_requires_exactly_two_samples_per_group() -> None:
    records = [
        _counterfactual_record(
            [0.9, 0.1],
            [1, 0],
            sample_id=index,
            pair_id=88,
            variant_id=index,
            counterfactual_type="action",
            scenario="action_counterfactual",
        )
        for index in range(3)
    ]
    with pytest.raises(ValueError, match="exactly two samples"):
        aggregate_counterfactual_metrics(records, 0.5)


def _five_seed_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, seed in enumerate((2026, 2027, 2028, 2029, 2030), start=1):
        rows.append(
            {
                "row_type": "run",
                "statistic": "single",
                "seed": seed,
                "method_id": "full",
                "method_kind": "learned",
                "training_variant": "full",
                "eval_perturbation": "none",
                "split": "id_test",
                "distribution": "ID",
                "ood_axis": "none",
                "scenario": "__all__",
                "threshold": 0.4 + index * 0.01,
                "threshold_source": "id_validation",
                "samples": 10,
                "points": 100,
                "positives": 25,
                "tp": 20,
                "fp": 5,
                "fn": 5,
                "tn": 70,
                "iou": float(index),
                "f1": float(index) / 10.0,
                "average_precision": float(index) / 5.0,
            }
        )
    return rows


def test_five_seed_summary_uses_sample_std_and_null_supports() -> None:
    summaries = seed_summary_records(_five_seed_rows())
    assert [row["statistic"] for row in summaries] == ["mean", "std", "min", "max"]
    by_statistic = {row["statistic"]: row for row in summaries}
    assert by_statistic["mean"]["iou"] == pytest.approx(3.0)
    assert by_statistic["std"]["iou"] == pytest.approx(np.sqrt(2.5))
    assert by_statistic["min"]["iou"] == pytest.approx(1.0)
    assert by_statistic["max"]["iou"] == pytest.approx(5.0)
    for row in summaries:
        assert row["seed"] is None
        assert row["seed_count"] == 5
        assert row["threshold"] is None
        assert row["samples"] is None
        assert row["points"] is None
        assert row["tp"] is None

    with pytest.raises(ValueError, match="expected 5 seeds"):
        seed_summary_records(_five_seed_rows()[:4])


def test_report_writers_are_deterministic_and_stably_ordered(tmp_path: object) -> None:
    from pathlib import Path

    root = Path(str(tmp_path))
    records = list(reversed(_five_seed_rows()[:2]))
    json_path = write_json_report(
        root / "metrics.json",
        records,
        metadata={"tensor": torch.tensor([2, 1]), "nonfinite": float("nan")},
    )
    first_json = json_path.read_bytes()
    write_json_report(json_path, records, metadata={"tensor": torch.tensor([2, 1]), "nonfinite": float("nan")})
    assert json_path.read_bytes() == first_json
    payload = __import__("json").loads(json_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "2.0"
    assert payload["experiment"]["tensor"] == [2, 1]
    assert payload["experiment"]["nonfinite"] is None
    assert [row["seed"] for row in payload["records"]] == [2026, 2027]

    csv_path = write_csv_records(root / "metrics.csv", records)
    first_csv = csv_path.read_bytes()
    write_csv_records(csv_path, records)
    assert csv_path.read_bytes() == first_csv
    header = csv_path.read_text(encoding="utf-8").splitlines()[0]
    assert header.startswith("row_type,statistic,seed,method_id")

    markdown_path = write_summary_markdown(
        root / "summary.md", records, strongest_baseline="time_aligned_future"
    )
    first_markdown = markdown_path.read_bytes()
    write_summary_markdown(
        markdown_path, records, strongest_baseline="time_aligned_future"
    )
    assert markdown_path.read_bytes() == first_markdown
    text = markdown_path.read_text(encoding="utf-8")
    assert "Scenario-wise metrics" in text
    assert "time_aligned_future" in text
