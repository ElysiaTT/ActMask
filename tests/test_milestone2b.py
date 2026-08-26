"""Focused CPU invariants for Milestone 2B dynamics/history support."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from actmask.data.milestone2b_dataset import (
    HIDDEN_STATE_FIELDS,
    OBSERVABLE_FIELDS,
    Milestone2BDataset,
    assert_milestone2b_no_group_leakage,
    build_milestone2b_group_manifest,
    estimate_velocity_from_observation_history,
    milestone2b_dataset_statistics,
)
from actmask.models.milestone2b_baselines import (
    FAIR_BASELINES,
    ExactHiddenTrajectoryOracle,
)
from actmask.models.temporal_actmask import ActMaskMLP, TemporalActMask
from actmask.training.milestone2b_losses import (
    base_supervised_losses,
    counterfactual_assignment_loss,
    irrelevant_background_stability_loss,
    pairwise_candidate_ranking_loss,
)


SMALL_COUNTS = {"train": 8, "val": 4, "test": 4}


def _dataset(*, domain: str = "id", split: str = "test") -> Milestone2BDataset:
    return Milestone2BDataset(
        split=split,
        domain=domain,
        split_counts=SMALL_COUNTS,
        ood_groups_per_axis=2,
    )


def test_hidden_and_observable_fields_are_isolated_from_fair_methods() -> None:
    sample = _dataset()[0]
    assert set(sample["observable"]) == set(OBSERVABLE_FIELDS)
    assert set(sample["hidden_state"]) == set(HIDDEN_STATE_FIELDS)
    assert not set(sample["observable"]).intersection(HIDDEN_STATE_FIELDS)
    batch = next(iter(DataLoader(_dataset(), batch_size=2)))
    for baseline in FAIR_BASELINES.values():
        output = baseline()(batch["observable"])
        assert output["mask_logits"].shape == (2, 48)
        polluted = {
            **batch["observable"],
            "exact_contact_state": batch["hidden_state"]["exact_contact_state"],
        }
        with pytest.raises(ValueError, match="hidden state"):
            baseline()(polluted)
    oracle = ExactHiddenTrajectoryOracle()(batch["hidden_state"])
    assert oracle["mask_logits"].shape == (2, 48)


def test_hard_positive_and_negative_proportions_are_above_fixed_gate() -> None:
    statistics = milestone2b_dataset_statistics(_dataset())
    assert statistics["hard_positive_fraction"] >= 0.40
    assert statistics["hard_negative_fraction"] >= 0.40
    assert statistics["candidate_sets"] == 2 * statistics["groups"]


def test_observation_history_velocity_estimation_is_exact_without_corruption() -> None:
    timestamps = np.asarray([-0.3, -0.2, -0.1, 0.0], dtype=np.float32)
    initial = np.asarray([[0.1, -0.2, 0.0], [0.0, 0.2, -0.1]], dtype=np.float32)
    velocity = np.asarray([[0.4, -0.1, 0.2], [-0.2, 0.3, 0.1]], dtype=np.float32)
    history = initial[None, :, :] + timestamps[:, None, None] * velocity[None, :, :]
    visibility = np.ones(history.shape[:2], dtype=np.bool_)
    estimated, confidence = estimate_velocity_from_observation_history(
        history, visibility, timestamps
    )
    assert np.allclose(estimated, velocity, atol=1.0e-6)
    assert np.allclose(confidence, 1.0)


def test_occlusion_delay_and_nonlinear_trajectories_are_real() -> None:
    occluded = _dataset(domain="unseen_occlusion_rate")[0]
    delayed = _dataset(domain="unseen_observation_delay")[0]
    assert float(occluded["observable"]["visibility_history"].float().mean()) < 0.70
    assert float(delayed["observable"]["observation_delay"]) >= 0.16

    nonlinear = _dataset(domain="unseen_acceleration")[0]
    trajectory = nonlinear["hidden_state"]["exact_future_point_trajectories"][:, :10]
    second_difference = trajectory[2:] - 2.0 * trajectory[1:-1] + trajectory[:-2]
    assert float(second_difference.abs().max()) > 1.0e-5
    assert float(
        nonlinear["hidden_state"]["exact_point_acceleration"].abs().max()
    ) > 1.0e-3


def test_temporal_models_have_mask_contact_future_and_utility_outputs() -> None:
    batch = next(iter(DataLoader(_dataset(), batch_size=3)))
    temporal = TemporalActMask()
    output = temporal(batch["observable"])
    assert output["mask_logits"].shape == (3, 48)
    assert output["contact_logits"].shape == (3, 48, 32)
    assert output["future_hypotheses"].shape == (3, 48, 32, 3)
    assert output["utility_logits"].shape == (3,)
    for mode in ("no_history", "no_action", "no_visibility"):
        assert temporal(batch["observable"], feature_mode=mode)["mask_logits"].shape == (
            3,
            48,
        )
    mlp = ActMaskMLP()(batch["observable"])
    assert mlp["mask_logits"].shape == (3, 48)
    assert mlp["utility_logits"].shape == (3,)


def test_counterfactual_and_utility_losses_respect_causal_semantics() -> None:
    correct_first = torch.tensor([[4.0, -4.0]])
    correct_second = torch.tensor([[-4.0, 4.0]])
    first_target = torch.tensor([[1.0, 0.0]])
    second_target = torch.tensor([[0.0, 1.0]])
    assert counterfactual_assignment_loss(
        correct_first, correct_second, first_target, second_target
    ).item() == pytest.approx(0.0)
    assert counterfactual_assignment_loss(
        correct_second, correct_first, first_target, second_target
    ).item() > 1.0
    # Unchanged GT returns exact zero rather than forcing prediction difference.
    assert counterfactual_assignment_loss(
        correct_first, correct_second, first_target, first_target
    ).item() == pytest.approx(0.0)
    assert irrelevant_background_stability_loss(
        correct_first, correct_first, first_target, first_target
    ).item() == pytest.approx(0.0)

    utility = torch.tensor([1.0, -1.0, 0.2], requires_grad=True)
    ranking = pairwise_candidate_ranking_loss(
        utility, torch.tensor([1.0, 0.0, 0.0]), ["scene", "scene", "scene"]
    )
    ranking.backward()
    assert utility.grad is not None


def test_supervised_loss_shapes_and_utility_head_receive_gradients() -> None:
    batch = next(iter(DataLoader(_dataset(), batch_size=2)))
    model = TemporalActMask()
    output = model(batch["observable"])
    losses = base_supervised_losses(
        mask_logits=output["mask_logits"],
        contact_logits=output["contact_logits"],
        utility_logits=output["utility_logits"],
        mask_targets=batch["targets"]["ground_truth_future_mask"],
        contact_targets=batch["targets"]["contact_matrix"],
        success_targets=batch["targets"]["success"],
    )
    sum(losses.values()).backward()
    assert all(torch.isfinite(value) for value in losses.values())
    assert any(
        parameter.grad is not None
        for name, parameter in model.named_parameters()
        if "utility_head" in name
    )


def test_grouped_manifest_is_deterministic_and_detects_family_leakage() -> None:
    first = build_milestone2b_group_manifest(
        master_seed=91, split_counts=SMALL_COUNTS, ood_groups_per_axis=2
    )
    second = build_milestone2b_group_manifest(
        master_seed=91, split_counts=SMALL_COUNTS, ood_groups_per_axis=2
    )
    assert first == second
    assert_milestone2b_no_group_leakage(first)
    source = first["train"][0]
    leaked = replace(
        first["val"][0], trajectory_family_id=source.trajectory_family_id
    )
    broken = dict(first)
    broken["val"] = (leaked, *first["val"][1:])
    with pytest.raises(ValueError, match="trajectory_family_id"):
        assert_milestone2b_no_group_leakage(broken)


def test_dataset_generation_is_byte_deterministic_for_same_seed() -> None:
    first, second = _dataset(), _dataset()
    for index in (0, 5, 10, len(first) - 1):
        left, right = first[index], second[index]
        assert left["metadata"] == right["metadata"]
        for section in ("observable", "targets", "hidden_state"):
            for key in left[section]:
                assert torch.equal(left[section][key], right[section][key])
