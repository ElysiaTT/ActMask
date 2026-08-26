"""Blocking integrity tests for the Milestone 2E factorial study."""

from __future__ import annotations

import copy
from itertools import permutations, product

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from actmask.data.milestone2b_dataset import Milestone2BDataset
from actmask.data.milestone2c_dataset import CorrespondenceCorruptionDataset, HardCandidateActionDataset, RigidTransformDataset
from actmask.experiments.milestone2e import (
    RankingBundle,
    _load_config,
    audit_scoring_and_indexing,
    ranking_metrics_tie_aware,
    validate_checkpoint_metadata,
)
from actmask.experiments.milestone2e_completion_extensions import PointPermutationDataset
from actmask.experiments.milestone2e_full import DiagnosticIdentityDataset
from actmask.experiments.milestone2e_temporal_diagnostics import (
    PRIMARY_HEAD,
    _validation_promising_from_records,
)
from actmask.models.correspondence import _euclidean_cost, _legacy_axis_biased_cost
from actmask.models.milestone2e import FactorialTemporalActMask


def _transform_observable(observable: dict[str, torch.Tensor], matrix: torch.Tensor) -> dict[str, torch.Tensor]:
    """Apply one exact orthogonal transform to all vector-valued observables."""

    changed = {key: value.clone() for key, value in observable.items()}
    changed["points_history"] = changed["points_history"] @ matrix.T
    changed["estimated_velocity"] = changed["estimated_velocity"] @ matrix.T
    action = changed["action_command"].clone()
    action[:, :6] = (action[:, :6].reshape(-1, 2, 3) @ matrix.T).reshape(-1, 6)
    changed["action_command"] = action
    return changed


def _all_signed_axis_transforms() -> list[torch.Tensor]:
    matrices: list[torch.Tensor] = []
    for order in permutations(range(3)):
        matrix = torch.zeros(3, 3)
        matrix[torch.arange(3), torch.as_tensor(order)] = 1.0
        matrices.append(matrix)
    matrices.extend(torch.diag(torch.as_tensor(signs)) for signs in product((-1.0, 1.0), repeat=3))
    return matrices


def test_handcrafted_score_direction_and_ties_ignore_candidate_ids() -> None:
    bundle = RankingBundle(
        scores=np.asarray([0.2, 0.9, 0.9]),
        success=np.asarray([0.0, 1.0, 0.0]),
        utility=np.asarray([0.0, 1.0, 0.0]),
        metadata=[
            {"candidate_set_id": 41, "candidate_id": candidate_id}
            for candidate_id in (200, 5, 1)
        ],
    )
    result = ranking_metrics_tie_aware(bundle, score_name="known")
    assert result["top1_success"] == pytest.approx(0.5)
    assert result["top3_success"] == pytest.approx(1.0)
    reordered = RankingBundle(
        scores=bundle.scores[[2, 0, 1]], success=bundle.success[[2, 0, 1]],
        utility=bundle.utility[[2, 0, 1]], metadata=[bundle.metadata[index] for index in (2, 0, 1)],
    )
    assert ranking_metrics_tie_aware(reordered, score_name="known")["top1_success"] == pytest.approx(0.5)


def test_standard_ndcg_uses_graded_utility_and_expected_tie_resolution() -> None:
    metadata = [
        {"candidate_set_id": 77, "candidate_id": candidate_id}
        for candidate_id in (41, 3, 29)
    ]
    utility = np.asarray([1.0, 0.6, 0.1])
    success = np.asarray([1.0, 1.0, 0.0])
    ideal = ranking_metrics_tie_aware(
        RankingBundle(np.asarray([0.9, 0.8, 0.7]), success, utility, metadata),
        score_name="graded",
    )
    assert ideal["ndcg"] == pytest.approx(1.0)
    assert "standard DCG/IDCG" in ideal["ndcg_definition"]

    worst = ranking_metrics_tie_aware(
        RankingBundle(np.asarray([0.7, 0.8, 0.9]), success, utility, metadata),
        score_name="graded",
    )
    discounts = 1.0 / np.log2(np.arange(1, 4, dtype=np.float64) + 1.0)
    expected_worst = float(np.dot(np.asarray([0.1, 0.6, 1.0]), discounts) / np.dot(np.asarray([1.0, 0.6, 0.1]), discounts))
    assert worst["ndcg"] == pytest.approx(expected_worst)

    tied = ranking_metrics_tie_aware(
        RankingBundle(
            np.asarray([0.9, 0.9, 0.1]), np.asarray([1.0, 0.0, 0.0]),
            np.asarray([1.0, 0.0, 0.0]), metadata,
        ),
        score_name="graded_tie",
    )
    expected_tie = 0.5 * (discounts[0] + discounts[1])
    assert tied["ndcg"] == pytest.approx(expected_tie)


def test_c50_candidate_sets_are_labeled_and_transform_safe() -> None:
    dataset = HardCandidateActionDataset(
        candidate_count=50, split="test", master_seed=20260922,
        split_counts={"train": 1, "val": 1, "test": 1}, ood_groups_per_axis=1,
    )
    members = [dataset[index] for index in range(50)]
    assert sorted(int(sample["metadata"]["candidate_id"]) for sample in members) == list(range(50))
    assert any(float(sample["targets"]["success"]) >= 0.5 for sample in members)
    transformed = RigidTransformDataset(dataset, mode="random_so3")
    for index, sample in enumerate(members):
        changed = transformed[index]
        assert float(changed["targets"]["success"]) == float(sample["targets"]["success"])
        assert float(changed["targets"]["candidate_utility"]) == pytest.approx(float(sample["targets"]["candidate_utility"]))
        assert int(changed["metadata"]["candidate_id"]) == int(sample["metadata"]["candidate_id"])


def test_stale_checkpoint_metadata_is_rejected() -> None:
    checkpoint = {
        "architecture": "factorial", "correspondence": "mutual", "backbone": "scalar",
        "config_digest": "abc", "state_dict": {},
    }
    validate_checkpoint_metadata(checkpoint, {
        "architecture": "factorial", "correspondence": "mutual", "backbone": "scalar", "config_digest": "abc",
    })
    with pytest.raises(ValueError, match="mismatch"):
        validate_checkpoint_metadata(checkpoint, {
            "architecture": "factorial", "correspondence": "soft", "backbone": "scalar", "config_digest": "abc",
        })


def test_scoring_and_indexing_audit_is_blocking_and_passes() -> None:
    config = copy.deepcopy(_load_config())
    # Exercise the same every-world audit logic with a small deterministic
    # frozen protocol; production config still audits all 80/500 worlds.
    config["ranking"]["validation_groups"] = 2
    config["ranking"]["test_groups"] = 3
    report = audit_scoring_and_indexing(config)
    assert report["passed"]
    assert report["checkpoint_mismatch_rejected"]
    assert set(report["candidate_availability"]) == {"C5", "C10", "C20", "C50"}
    assert all(row["labels_preserved"] for row in report["transform_checks"])
    assert report["handcrafted_cases"]["global_so3_copy"]["ranking_metrics_identical"]
    assert report["handcrafted_cases"]["temporal_order_changes_correct_candidate"] == {
        "forward_correct_candidate_id": 73,
        "reversed_correct_candidate_id": 8,
        "forward_top1_success": 1.0,
        "reversed_top1_success": 1.0,
    }
    assert all(
        value["scope"] == "every frozen validation and test ranking world"
        and value["mask_to_candidate_index_checks"] > 0
        for value in report["candidate_availability"].values()
    )


def test_factorial_correspondence_variants_are_explicit_about_identity() -> None:
    source = CorrespondenceCorruptionDataset(
        Milestone2BDataset(split="test", master_seed=302, split_counts={"train": 1, "val": 1, "test": 2}, ood_groups_per_axis=1),
        mode="per_frame_permutation", seed=17,
    )
    batch = next(iter(DataLoader(source, batch_size=2)))
    for correspondence in ("CoordinateInvariantMutualNN", "SoftCoordinateInvariantCorrespondence", "NoExplicitCorrespondence"):
        model = FactorialTemporalActMask(correspondence=correspondence, backbone="ScalarInvariantFeatureBackbone")
        assert model(batch["observable"])["mask_logits"].shape == (2, 48)
        with pytest.raises(ValueError, match="only allowed"):
            model(batch["observable"], identity_history=batch["hidden_state"]["evaluation_identity_history"])
    diagnostic = FactorialTemporalActMask(
        correspondence="GroundTruthIdentityCorrespondence", backbone="ScalarInvariantFeatureBackbone"
    )
    with pytest.raises(ValueError, match="requires diagnostic"):
        diagnostic(batch["observable"])
    assert diagnostic(
        batch["observable"], identity_history=batch["hidden_state"]["evaluation_identity_history"]
    )["utility_logits"].shape == (2,)


def test_action_frame_backbone_is_invariant_without_global_axes() -> None:
    dataset = HardCandidateActionDataset(
        candidate_count=5, split="test", master_seed=303,
        split_counts={"train": 1, "val": 1, "test": 1}, ood_groups_per_axis=1,
    )
    torch.manual_seed(12)
    model = FactorialTemporalActMask(
        correspondence="CoordinateInvariantMutualNN", backbone="ActionFrameRelativeBackbone"
    ).eval()
    clean = model(next(iter(DataLoader(dataset, batch_size=5)))["observable"])
    for mode in ("random_so3", "axis_permutation", "sign_flip"):
        transformed = model(next(iter(DataLoader(RigidTransformDataset(dataset, mode=mode), batch_size=5)))["observable"])
        # Full-precision GEMM reduction order changes under a rotation.  The
        # observed numerical envelope is far below any ranking-relevant value.
        assert torch.allclose(clean["mask_logits"], transformed["mask_logits"], atol=3.0e-5, rtol=3.0e-5)
        assert torch.allclose(clean["utility_logits"], transformed["utility_logits"], atol=3.0e-5, rtol=3.0e-5)


def test_action_frame_backbone_handles_zero_and_near_zero_length_actions() -> None:
    dataset = HardCandidateActionDataset(
        candidate_count=5, split="test", master_seed=304,
        split_counts={"train": 1, "val": 1, "test": 1}, ood_groups_per_axis=1,
    )
    batch = next(iter(DataLoader(dataset, batch_size=5)))
    torch.manual_seed(13)
    model = FactorialTemporalActMask(
        correspondence="CoordinateInvariantMutualNN", backbone="ActionFrameRelativeBackbone"
    ).eval()

    zero = {key: value.clone() for key, value in batch["observable"].items()}
    zero_action = zero["action_command"].clone()
    zero_action[:, 3:6] = zero_action[:, :3]
    zero["action_command"] = zero_action
    zero_output = model(zero)
    assert all(torch.isfinite(value).all() for value in zero_output.values() if isinstance(value, torch.Tensor))

    near_zero = {key: value.clone() for key, value in zero.items()}
    near_action = near_zero["action_command"].clone()
    near_action[:, 3:6] = near_action[:, :3] + torch.tensor([1.0e-8, -1.0e-8, 1.0e-8])
    near_zero["action_command"] = near_action
    near_output = model(near_zero)
    assert all(torch.isfinite(value).all() for value in near_output.values() if isinstance(value, torch.Tensor))

    rotation = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    rotated_zero = _transform_observable(zero, rotation)
    rotated_output = model(rotated_zero)
    assert torch.allclose(zero_output["mask_logits"], rotated_output["mask_logits"], atol=5.0e-5, rtol=5.0e-5)
    assert torch.allclose(zero_output["utility_logits"], rotated_output["utility_logits"], atol=5.0e-5, rtol=5.0e-5)


def test_factorial_mutual_correspondence_is_equivariant_under_all_axis_shortcut_probes() -> None:
    """Correspondence-level test: indices and scalar cost cannot see an axis."""

    source = CorrespondenceCorruptionDataset(
        Milestone2BDataset(split="test", master_seed=404, split_counts={"train": 1, "val": 1, "test": 2}, ood_groups_per_axis=1),
        mode="per_frame_resampling", seed=77,
    )
    batch = next(iter(DataLoader(source, batch_size=2)))
    model = FactorialTemporalActMask(correspondence="CoordinateInvariantMutualNN", backbone="ScalarInvariantFeatureBackbone").eval()
    reference = model.associate(batch["observable"])
    generator = torch.Generator().manual_seed(9)
    random_matrix = torch.linalg.qr(torch.randn(3, 3, generator=generator)).Q
    if torch.linalg.det(random_matrix) < 0.0:
        random_matrix[:, 0] *= -1.0
    matrices = [random_matrix, *_all_signed_axis_transforms()]
    for matrix in matrices:
        changed = model.associate(_transform_observable(batch["observable"], matrix))
        assert torch.equal(reference.source_index, changed.source_index)
        assert torch.equal(reference.aligned_visibility, changed.aligned_visibility)
        finite = torch.isfinite(reference.normalized_distance)
        assert torch.allclose(reference.normalized_distance[finite], changed.normalized_distance[finite], atol=3.0e-5, rtol=3.0e-5)
        assert torch.allclose(changed.aligned_history, reference.aligned_history @ matrix.T, atol=5.0e-5, rtol=5.0e-5)
        assert torch.allclose(changed.velocity, reference.velocity @ matrix.T, atol=5.0e-5, rtol=5.0e-5)


def test_axis_biased_negative_control_fails_invariance_probe() -> None:
    anchors = torch.tensor([[[0.0, 0.0, 0.0]]])
    source = torch.tensor([[[0.15, 0.0, 0.0], [0.0, 0.20, 0.0]]])
    swap = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    legacy = _legacy_axis_biased_cost(anchors, source).argmin(dim=-1)
    legacy_swapped = _legacy_axis_biased_cost(anchors @ swap.T, source @ swap.T).argmin(dim=-1)
    euclidean = _euclidean_cost(anchors, source).argmin(dim=-1)
    euclidean_swapped = _euclidean_cost(anchors @ swap.T, source @ swap.T).argmin(dim=-1)
    assert not torch.equal(legacy, legacy_swapped)
    assert torch.equal(euclidean, euclidean_swapped)


def test_final_ranking_invariance_is_a_separate_action_frame_test() -> None:
    """Ranking-level test: correspondence passing alone is not the claim."""

    dataset = HardCandidateActionDataset(
        candidate_count=5, split="test", master_seed=505,
        split_counts={"train": 1, "val": 1, "test": 1}, ood_groups_per_axis=1,
    )
    torch.manual_seed(14)
    model = FactorialTemporalActMask(correspondence="SoftCoordinateInvariantCorrespondence", backbone="ActionFrameRelativeBackbone").eval()
    batch = next(iter(DataLoader(dataset, batch_size=5)))
    clean = model(batch["observable"])
    metadata = [
        {key: int(value[index]) if isinstance(value, torch.Tensor) and value.ndim == 1 else value[index] for key, value in batch["metadata"].items()}
        for index in range(5)
    ]
    success = batch["targets"]["success"].numpy()
    utility = batch["targets"]["candidate_utility"].numpy()
    clean_bundle = RankingBundle(torch.sigmoid(clean["utility_logits"]).detach().numpy(), success, utility, metadata)
    clean_metrics = ranking_metrics_tie_aware(clean_bundle, score_name="unit")
    for matrix in _all_signed_axis_transforms():
        changed = model(_transform_observable(batch["observable"], matrix))
        changed_bundle = RankingBundle(torch.sigmoid(changed["utility_logits"]).detach().numpy(), success, utility, metadata)
        changed_metrics = ranking_metrics_tie_aware(changed_bundle, score_name="unit")
        assert torch.allclose(clean["utility_logits"], changed["utility_logits"], atol=5.0e-5, rtol=5.0e-5)
        for key in ("top1_success", "top3_success", "ranking_roc_auc", "mrr", "ndcg", "normalized_regret"):
            assert clean_metrics[key] == pytest.approx(changed_metrics[key], abs=1.0e-10)


def test_ranking_point_permutation_keeps_targets_and_oracle_identity_aligned() -> None:
    """A ranking-level order view must not turn into an indexing corruption."""

    source = DiagnosticIdentityDataset(HardCandidateActionDataset(
        candidate_count=5, split="test", master_seed=606,
        split_counts={"train": 1, "val": 1, "test": 1}, ood_groups_per_axis=1,
    ))
    changed = PointPermutationDataset(source)
    original_sample, permuted_sample = source[0], changed[0]
    order = permuted_sample["hidden_state"]["evaluation_identity_history"][-1]
    assert not torch.equal(order, torch.arange(order.numel()))
    assert int(original_sample["metadata"]["candidate_id"]) == int(permuted_sample["metadata"]["candidate_id"])
    assert float(original_sample["targets"]["success"]) == float(permuted_sample["targets"]["success"])
    assert float(original_sample["targets"]["candidate_utility"]) == pytest.approx(
        float(permuted_sample["targets"]["candidate_utility"])
    )
    assert torch.equal(
        permuted_sample["targets"]["ground_truth_future_mask"],
        original_sample["targets"]["ground_truth_future_mask"][order],
    )
    for frame in range(permuted_sample["observable"]["points_history"].shape[0]):
        assert torch.equal(permuted_sample["hidden_state"]["evaluation_identity_history"][frame], order)


def test_validation_promising_phase4_arms_are_fixed_from_complete_five_seed_records() -> None:
    """The expansion rule is validation-only and includes both near-best fair families."""

    seeds = (1601, 2603, 3607, 4611, 5613)
    means = {
        ("SoftCoordinateInvariantCorrespondence", "ScalarInvariantFeatureBackbone"): (0.2115, 0.8200),
        ("SoftCoordinateInvariantCorrespondence", "ActionFrameRelativeBackbone"): (0.21175, 0.8175),
        ("CoordinateInvariantMutualNN", "ScalarInvariantFeatureBackbone"): (0.2680, 0.7700),
        ("CoordinateInvariantMutualNN", "ActionFrameRelativeBackbone"): (0.26575, 0.7850),
        ("NoExplicitCorrespondence", "ActionFrameRelativeBackbone"): (0.3295, 0.7125),
    }
    records = [
        {
            "correspondence": correspondence,
            "backbone": backbone,
            "ranking_head": PRIMARY_HEAD,
            "seed": seed,
            "validation_normalized_regret": regret,
            "validation_top1": top1,
        }
        for (correspondence, backbone), (regret, top1) in means.items()
        for seed in seeds
    ]
    promising, evidence = _validation_promising_from_records(
        records,
        expected_seeds=seeds,
        fair_variants=tuple(means),
        selected_primary={
            "correspondence": "SoftCoordinateInvariantCorrespondence",
            "backbone": "ScalarInvariantFeatureBackbone",
        },
    )
    assert {variant.key for variant in promising} == {
        ("SoftCoordinateInvariantCorrespondence", "ScalarInvariantFeatureBackbone"),
        ("SoftCoordinateInvariantCorrespondence", "ActionFrameRelativeBackbone"),
        ("CoordinateInvariantMutualNN", "ScalarInvariantFeatureBackbone"),
        ("CoordinateInvariantMutualNN", "ActionFrameRelativeBackbone"),
    }
    assert all(variant.seeds == seeds for variant in promising)
    rejected = next(
        row for row in evidence["all_fair_primary_arms"]
        if row["correspondence"] == "NoExplicitCorrespondence"
    )
    assert not rejected["passes_fixed_validation_promising_rule"]
