"""Fast CPU invariants for the Milestone 2C closure components."""

from __future__ import annotations

import copy
from itertools import permutations, product

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from actmask.data.milestone2b_dataset import HIDDEN_STATE_FIELDS, Milestone2BDataset
from actmask.data.milestone2c_dataset import (
    CORRESPONDENCE_MODES,
    CorrespondenceCorruptionDataset,
    HardCandidateActionDataset,
    RigidTransformDataset,
    _orthogonal_transform,
    candidate_set_statistics,
)
from actmask.eval.calibration import apply_platt, fit_platt_scaling, reliability_metrics
from actmask.eval.statistics import paired_summary
from actmask.experiments.milestone2c import _coefficients, _runtime_case, predict_model, ranking_metrics
from actmask.models.correspondence import (
    InvariantFeatureMotionTemporalActMask,
    LegacyAxisBiasedNeighborhoodTemporalActMask,
    LocalNeighborhoodTemporalActMask,
    MotionConsistentMutualTemporalActMask,
    _euclidean_cost,
    _legacy_axis_biased_cost,
    SetHistoryTemporalActMask,
)
from actmask.models.milestone2c_baselines import FAIR_BASELINES_2C


COUNTS = {"train": 2, "val": 2, "test": 2}


def _dataset() -> Milestone2BDataset:
    return Milestone2BDataset(
        split="test", master_seed=20260719, split_counts=COUNTS, ood_groups_per_axis=1
    )


def test_loss_specs_explicitly_disable_unspecified_terms() -> None:
    only_mask = _coefficients("mask_only").to_dict()
    assert only_mask["mask_bce"] == 1.0
    assert all(value == 0.0 for name, value in only_mask.items() if name != "mask_bce")
    full = _coefficients("full_2b").to_dict()
    assert all(value > 0.0 for value in full.values())


def test_paired_statistics_require_group_evidence() -> None:
    result = paired_summary([0.9, 0.8, 0.7, 0.8], [0.1, 0.2, 0.3, 0.2], seed=7)
    assert result["groups"] == 4
    assert result["ci_low"] > 0.0
    # With four independent pairs, the exact sign-flip resolution is coarse.
    assert result["p_value"] <= 0.13
    with pytest.raises(ValueError):
        paired_summary([0.5], [0.4], seed=7)


def test_calibration_is_bounded_and_uses_binary_targets() -> None:
    scores = np.asarray([0.05, 0.15, 0.75, 0.95])
    targets = np.asarray([0.0, 0.0, 1.0, 1.0])
    parameters = fit_platt_scaling(scores, targets)
    calibrated = apply_platt(scores, parameters)
    assert np.all((0.0 <= calibrated) & (calibrated <= 1.0))
    metrics = reliability_metrics(calibrated, targets, bins=2)
    assert metrics["count"] == 4


def test_corruption_contract_preserves_point_target_alignment_and_variable_n() -> None:
    source = _dataset()
    for mode in CORRESPONDENCE_MODES:
        sample = CorrespondenceCorruptionDataset(source, mode=mode, seed=31)[0]
        points = sample["observable"]["points_history"].shape[1]
        assert sample["targets"]["ground_truth_future_mask"].shape == (points,)
        assert sample["targets"]["contact_matrix"].shape[1] == points
        assert sample["metadata"]["correspondence_mode"] == mode
    small = CorrespondenceCorruptionDataset(
        source, mode="per_frame_permutation", point_count=32, seed=31
    )[0]
    large = CorrespondenceCorruptionDataset(
        source, mode="per_frame_permutation", point_count=64, seed=31
    )[0]
    assert small["observable"]["points_history"].shape[1] == 32
    assert large["observable"]["points_history"].shape[1] == 64


def test_spatial_models_accept_independently_permuted_frames_without_hidden_inputs() -> None:
    batch = next(iter(DataLoader(CorrespondenceCorruptionDataset(_dataset(), mode="per_frame_permutation"), batch_size=2)))
    for constructor in (LocalNeighborhoodTemporalActMask, SetHistoryTemporalActMask):
        model = constructor()
        output = model(batch["observable"])
        assert output["mask_logits"].shape == (2, 48)
        polluted = {**batch["observable"], next(iter(HIDDEN_STATE_FIELDS)): batch["hidden_state"][next(iter(HIDDEN_STATE_FIELDS))]}
        with pytest.raises(ValueError, match="hidden state"):
            model(polluted)


def test_hard_candidate_sets_have_independent_ids_and_physical_labels() -> None:
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    dataset = HardCandidateActionDataset(
        candidate_count=10, split="test", master_seed=101, split_counts=COUNTS,
        ood_groups_per_axis=1,
    )
    group = [dataset[index] for index in range(10)]
    assert sorted(int(sample["metadata"]["candidate_id"]) for sample in group) == list(range(10))
    assert {sample["metadata"]["candidate_slot"] for sample in group} == set(range(10))
    assert any(float(sample["targets"]["success"]) > 0.5 for sample in group)
    stats = candidate_set_statistics(dataset)
    assert stats["candidate_count"] == 10
    assert stats["worlds_with_success_fraction"] == 1.0


def test_all_fair_baselines_reject_hidden_state_contamination() -> None:
    batch = next(iter(DataLoader(_dataset(), batch_size=2)))
    polluted = {**batch["observable"], "exact_contact_state": batch["hidden_state"]["exact_contact_state"]}
    for name, constructor in FAIR_BASELINES_2C.items():
        model = constructor()
        with pytest.raises(ValueError, match="hidden state"):
            model(polluted)
        assert model(batch["observable"])["mask_logits"].shape == (2, 48)


def test_runtime_case_is_cpu_and_reports_all_scaling_fields() -> None:
    from actmask.experiments.milestone2c import _load_config

    config = copy.deepcopy(_load_config())
    config["runtime"]["warmup"] = 1
    config["runtime"]["repeats"] = 2
    result = _runtime_case(config=config, points=32, frames=2, horizon=4, candidates=2)
    assert result["latency_p95_ms"] >= result["latency_p50_ms"] >= 0.0
    assert result["throughput_candidates_per_second"] > 0.0
    assert result["parameter_count"] > 0



def _rotated_expected(value: torch.Tensor, group_ids: torch.Tensor, mode: str) -> torch.Tensor:
    rotation = torch.stack([
        _orthogonal_transform(20260911, int(group_id), mode)
        for group_id in group_ids
    ])
    return torch.einsum("b...i,bij->b...j", value, rotation.transpose(-1, -2))


def _transform_observable(observable: dict[str, torch.Tensor], matrix: torch.Tensor) -> dict[str, torch.Tensor]:
    transformed = {key: value.clone() for key, value in observable.items()}
    transformed["points_history"] = transformed["points_history"] @ matrix.T
    transformed["estimated_velocity"] = transformed["estimated_velocity"] @ matrix.T
    action = transformed["action_command"].clone()
    if action.ndim == 1:
        action[:6] = (action[:6].reshape(2, 3) @ matrix.T).reshape(6)
    else:
        action[:, :6] = (action[:, :6].reshape(-1, 2, 3) @ matrix.T).reshape(-1, 6)
    transformed["action_command"] = action
    return transformed


def test_primary_correspondence_is_equivariant_for_rotations_permutations_and_corruptions() -> None:
    base = _dataset()
    model = MotionConsistentMutualTemporalActMask().eval()
    for corruption in ("per_frame_permutation", "per_frame_resampling", "full_part_occlusion"):
        source = CorrespondenceCorruptionDataset(base, mode=corruption, seed=67)
        ordinary = next(iter(DataLoader(source, batch_size=2)))
        ordinary_result = model.associate(ordinary["observable"])
        keys = ordinary["metadata"]["candidate_set_id"]
        for mode in ("random_so3", "axis_permutation", "sign_flip"):
            rotated = next(iter(DataLoader(RigidTransformDataset(source, mode=mode), batch_size=2)))
            transformed_result = model.associate(rotated["observable"])
            assert torch.equal(ordinary_result.source_index, transformed_result.source_index)
            assert torch.equal(ordinary_result.aligned_visibility, transformed_result.aligned_visibility)
            finite = torch.isfinite(ordinary_result.normalized_distance)
            assert torch.allclose(
                ordinary_result.normalized_distance[finite],
                transformed_result.normalized_distance[finite],
                atol=2.0e-5,
                rtol=2.0e-5,
            )
            assert torch.allclose(
                transformed_result.aligned_history,
                _rotated_expected(ordinary_result.aligned_history, keys, mode),
                atol=2.0e-5,
                rtol=2.0e-5,
            )
            assert torch.allclose(
                transformed_result.velocity,
                _rotated_expected(ordinary_result.velocity, keys, mode),
                atol=2.0e-5,
                rtol=2.0e-5,
            )


def test_primary_correspondence_covers_all_axis_permutations_and_sign_flips() -> None:
    source = CorrespondenceCorruptionDataset(
        _dataset(), mode="per_frame_resampling", seed=68
    )
    batch = next(iter(DataLoader(source, batch_size=2)))
    observable = batch["observable"]
    model = MotionConsistentMutualTemporalActMask().eval()
    reference = model.associate(observable)
    matrices = []
    for order in permutations(range(3)):
        permutation = torch.zeros(3, 3)
        permutation[torch.arange(3), torch.as_tensor(order)] = 1.0
        matrices.append(permutation)
    for signs in product((-1.0, 1.0), repeat=3):
        matrices.append(torch.diag(torch.as_tensor(signs)))
    for matrix in matrices:
        transformed = model.associate(_transform_observable(observable, matrix))
        assert torch.equal(reference.source_index, transformed.source_index)
        assert torch.equal(reference.aligned_visibility, transformed.aligned_visibility)
        finite = torch.isfinite(reference.normalized_distance)
        assert torch.allclose(
            reference.normalized_distance[finite], transformed.normalized_distance[finite],
            atol=2.0e-5, rtol=2.0e-5,
        )
        assert torch.allclose(
            transformed.aligned_history, reference.aligned_history @ matrix.T,
            atol=2.0e-5, rtol=2.0e-5,
        )
        assert torch.allclose(
            transformed.velocity, reference.velocity @ matrix.T,
            atol=2.0e-5, rtol=2.0e-5,
        )


def test_axis_biased_negative_control_is_detected_by_axis_permutation() -> None:
    anchors = torch.tensor([[[0.0, 0.0, 0.0]]])
    source = torch.tensor([[[0.15, 0.0, 0.0], [0.0, 0.20, 0.0]]])
    swap = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    original_legacy = _legacy_axis_biased_cost(anchors, source).argmin(dim=-1)
    transformed_legacy = _legacy_axis_biased_cost(anchors @ swap.T, source @ swap.T).argmin(dim=-1)
    original_primary = _euclidean_cost(anchors, source).argmin(dim=-1)
    transformed_primary = _euclidean_cost(anchors @ swap.T, source @ swap.T).argmin(dim=-1)
    assert not torch.equal(original_legacy, transformed_legacy)
    assert torch.equal(original_primary, transformed_primary)
    assert LegacyAxisBiasedNeighborhoodTemporalActMask.__name__.startswith("LegacyAxisBiased")
    proper_rotation = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    rotated_legacy = _legacy_axis_biased_cost(anchors, source).argmin(dim=-1)
    rotated_legacy = _legacy_axis_biased_cost(
        anchors @ proper_rotation.T, source @ proper_rotation.T
    ).argmin(dim=-1)
    assert not torch.equal(original_legacy, rotated_legacy)


def test_final_ranking_invariance_requires_scalar_backbone() -> None:
    dataset = HardCandidateActionDataset(
        candidate_count=10, split="test", master_seed=20260912,
        split_counts=COUNTS, ood_groups_per_axis=1,
    )
    rotated = RigidTransformDataset(dataset, mode="random_so3")
    torch.manual_seed(101)
    vector = MotionConsistentMutualTemporalActMask().eval()
    vector_clean = predict_model(vector, dataset, batch_size=10)
    vector_rotated = predict_model(vector, rotated, batch_size=10)
    assert not np.allclose(vector_clean.logits, vector_rotated.logits, atol=1.0e-5, rtol=1.0e-5)
    torch.manual_seed(101)
    invariant = InvariantFeatureMotionTemporalActMask().eval()
    clean = predict_model(invariant, dataset, batch_size=10)
    transformed = predict_model(invariant, rotated, batch_size=10)
    assert np.allclose(clean.logits, transformed.logits, atol=5.0e-4, rtol=5.0e-4)
    assert np.allclose(clean.utility, transformed.utility, atol=5.0e-4, rtol=5.0e-4)
    original_metrics = ranking_metrics(clean, clean.utility, score_name="learned_utility")
    transformed_metrics = ranking_metrics(transformed, transformed.utility, score_name="learned_utility")
    for key in ("top1_success", "top3_success", "ranking_auc", "mrr", "ndcg", "regret"):
        assert original_metrics[key] == pytest.approx(transformed_metrics[key], abs=1.0e-8)


def test_invariant_ranking_covers_all_axis_permutations_and_sign_flips() -> None:
    dataset = HardCandidateActionDataset(
        candidate_count=5, split="test", master_seed=20260913,
        split_counts={"train": 1, "val": 1, "test": 1}, ood_groups_per_axis=1,
    )

    torch.manual_seed(102)
    model = InvariantFeatureMotionTemporalActMask().eval()
    batch = next(iter(DataLoader(dataset, batch_size=5)))
    clean_output = model(batch["observable"])
    clean_order = torch.argsort(clean_output["utility_logits"], descending=True)
    matrices = []
    for order in permutations(range(3)):
        matrix = torch.zeros(3, 3)
        matrix[torch.arange(3), torch.as_tensor(order)] = 1.0
        matrices.append(matrix)
    matrices.extend(torch.diag(torch.as_tensor(signs)) for signs in product((-1.0, 1.0), repeat=3))
    for matrix in matrices:
        transformed_output = model(_transform_observable(batch["observable"], matrix))
        assert torch.allclose(
            clean_output["mask_logits"], transformed_output["mask_logits"],
            atol=5.0e-4, rtol=5.0e-4,
        )
        assert torch.allclose(
            clean_output["utility_logits"], transformed_output["utility_logits"],
            atol=5.0e-4, rtol=5.0e-4,
        )
        assert torch.equal(clean_order, torch.argsort(transformed_output["utility_logits"], descending=True))
