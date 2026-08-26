"""Focused provenance and candidate-mixture tests for final-v2 evaluation."""

from __future__ import annotations

import pytest
import torch

from actmask.data.milestone2c_dataset import HardCandidateActionDataset
from actmask.experiments.milestone2e_final_v2 import (
    CHANGED_CANDIDATE_DISTRIBUTION_ID,
    CHANGED_CANDIDATE_TEMPLATE_INDICES,
    INVARIANCE_VIEWS,
    _parameter_free_fair_baseline_names,
)


def test_predeclared_candidate_template_mixture_preserves_labels_and_observable_contract() -> None:
    """The v2 mixture is explicit evaluator provenance, never a model input."""

    assert len(CHANGED_CANDIDATE_TEMPLATE_INDICES) == 20
    assert len(set(CHANGED_CANDIDATE_TEMPLATE_INDICES)) == 20
    dataset = HardCandidateActionDataset(
        candidate_count=20,
        split="test",
        master_seed=20268119,
        split_counts={"train": 1, "val": 1, "test": 1},
        ood_groups_per_axis=1,
        template_indices=CHANGED_CANDIDATE_TEMPLATE_INDICES,
        candidate_distribution_id=CHANGED_CANDIDATE_DISTRIBUTION_ID,
    )
    samples = [dataset[index] for index in range(20)]
    assert sorted(int(sample["metadata"]["candidate_id"]) for sample in samples) == list(range(20))
    assert [int(sample["metadata"]["candidate_template_index"]) for sample in samples] == list(
        CHANGED_CANDIDATE_TEMPLATE_INDICES
    )
    assert {sample["metadata"]["candidate_distribution_id"] for sample in samples} == {
        CHANGED_CANDIDATE_DISTRIBUTION_ID
    }
    for sample in samples:
        assert torch.equal(
            sample["targets"]["ground_truth_future_mask"].bool(),
            sample["hidden_state"]["exact_contact_state"].bool(),
        )
        assert "candidate_distribution_id" not in sample["observable"]
        assert "candidate_template_index" not in sample["observable"]


def test_legacy_prefix_candidate_construction_remains_the_default() -> None:
    dataset = HardCandidateActionDataset(
        candidate_count=5,
        split="test",
        master_seed=20268120,
        split_counts={"train": 1, "val": 1, "test": 1},
        ood_groups_per_axis=1,
    )
    assert dataset.template_indices == (0, 1, 2, 3, 4)
    assert dataset.candidate_distribution_id == "legacy_prefix"
    for index in range(5):
        metadata = dataset[index]["metadata"]
        assert "candidate_template_index" not in metadata
        assert "candidate_distribution_id" not in metadata


def test_candidate_template_mixture_requires_a_complete_unique_c20_specification() -> None:
    arguments = {
        "candidate_count": 20,
        "split": "test",
        "master_seed": 20268121,
        "split_counts": {"train": 1, "val": 1, "test": 1},
        "ood_groups_per_axis": 1,
    }
    with pytest.raises(ValueError, match="length"):
        HardCandidateActionDataset(template_indices=(0, 1), **arguments)
    with pytest.raises(ValueError, match="duplicates"):
        HardCandidateActionDataset(template_indices=tuple([0] * 20), **arguments)


def test_v2_protocol_declares_all_required_ranking_views_and_excludes_learned_baseline() -> None:
    assert INVARIANCE_VIEWS == (
        "original",
        "random_so3",
        "axis_permutation",
        "sign_flip",
        "group_shared_all_frame_point_permutation",
        "group_shared_per_frame_resampling",
        "group_shared_full_part_occlusion",
    )
    names = _parameter_free_fair_baseline_names()
    assert "GeometryWithLearnedResidual" not in names
    assert "MultiHypothesisTrajectoryProximity" in names
