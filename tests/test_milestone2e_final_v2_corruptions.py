"""Focused contracts for final-v2 group-shared ranking corruptions.

These tests intentionally exercise the new versioned wrapper only.  Legacy
``CorrespondenceCorruptionDataset`` remains a per-sample diagnostic and is
not changed or reinterpreted here.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from actmask.data.milestone2b_dataset import estimate_velocity_from_observation_history
from actmask.data.milestone2c_dataset import HardCandidateActionDataset
from actmask.data.milestone2e_final_v2_corruptions import (
    GROUP_SHARED_RANKING_CORRUPTION_MODES,
    GroupSharedRankingCorruptionDataset,
)


def _ranking_dataset() -> HardCandidateActionDataset:
    return HardCandidateActionDataset(
        candidate_count=5,
        split="test",
        master_seed=20262091,
        split_counts={"train": 1, "val": 1, "test": 2},
        ood_groups_per_axis=1,
    )


def _scene_fields(sample: dict[str, object]) -> dict[str, torch.Tensor]:
    observable = sample["observable"]
    assert isinstance(observable, dict)
    return {
        name: observable[name]
        for name in (
            "points_history",
            "visibility_history",
            "timestamps",
            "estimated_velocity",
            "velocity_confidence",
        )
    }


@pytest.mark.parametrize("mode", GROUP_SHARED_RANKING_CORRUPTION_MODES)
def test_group_shared_ranking_corruption_keeps_one_scene_per_world_and_aligns_labels(
    mode: str,
) -> None:
    """One candidate-set key, not five sample IDs, determines the scene view."""

    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        source = _ranking_dataset()
        original = [source[index] for index in range(source.candidate_count)]
        original_snapshot = copy.deepcopy(original)
        assert len({int(sample["metadata"]["sample_id"]) for sample in original}) == source.candidate_count
        assert len({int(sample["metadata"]["candidate_set_id"]) for sample in original}) == 1

        corrupted = GroupSharedRankingCorruptionDataset(source, mode=mode, seed=751)
        changed = [corrupted[index] for index in range(source.candidate_count)]

        # All candidates share the same corrupted observable scene despite
        # having distinct sample IDs and candidate-specific actions.
        reference_scene = _scene_fields(changed[0])
        for candidate in changed[1:]:
            for field, reference in reference_scene.items():
                assert torch.equal(_scene_fields(candidate)[field], reference), field

        # Candidate actions/timing, scalar labels, utility, and IDs never
        # change.  Point-indexed labels may move only with the final-frame
        # permutation, verified below against identity provenance.
        for before, after in zip(original, changed, strict=True):
            before_observable = before["observable"]
            after_observable = after["observable"]
            assert isinstance(before_observable, dict)
            assert isinstance(after_observable, dict)
            for field in ("action_command", "nominal_action_delay", "observation_delay"):
                assert torch.equal(after_observable[field], before_observable[field]), field
            for field in ("success", "candidate_utility"):
                assert torch.equal(after["targets"][field], before["targets"][field]), field
            for field in ("sample_id", "candidate_id", "candidate_set_id", "oracle_candidate_id"):
                assert after["metadata"][field] == before["metadata"][field], field
            assert after["metadata"]["ranking_corruption_mode"] == mode
            assert after["metadata"]["ranking_corruption_group_key_field"] == "candidate_set_id"
            assert after["metadata"]["ranking_corruption_group_key"] == str(
                before["metadata"]["candidate_set_id"]
            )

            identity = after["hidden_state"]["evaluation_identity_history"]
            assert identity.shape == before_observable["points_history"].shape[:2]
            assert torch.equal(identity, changed[0]["hidden_state"]["evaluation_identity_history"])
            final_identity = identity[-1]
            # The final-frame output mask/contact labels and all evaluator
            # point fields follow precisely the same final-frame ordering.
            for field in (
                "ground_truth_future_mask",
                "hard_positive_mask",
                "hard_negative_mask",
                "target_object_mask",
            ):
                assert torch.equal(after["targets"][field], before["targets"][field][final_identity]), field
            assert torch.equal(
                after["targets"]["contact_matrix"], before["targets"]["contact_matrix"][:, final_identity]
            )
            assert torch.equal(
                after["hidden_state"]["exact_future_point_trajectories"],
                before["hidden_state"]["exact_future_point_trajectories"][:, final_identity],
            )
            assert torch.equal(
                after["hidden_state"]["exact_point_velocity"],
                before["hidden_state"]["exact_point_velocity"][final_identity],
            )
            assert torch.equal(
                after["hidden_state"]["exact_point_acceleration"],
                before["hidden_state"]["exact_point_acceleration"][final_identity],
            )
            assert torch.equal(
                after["hidden_state"]["exact_contact_matrix"],
                before["hidden_state"]["exact_contact_matrix"][:, final_identity],
            )
            assert torch.equal(
                after["hidden_state"]["exact_contact_state"],
                before["hidden_state"]["exact_contact_state"][final_identity],
            )

            expected_velocity, expected_confidence = estimate_velocity_from_observation_history(
                after_observable["points_history"].numpy(),
                after_observable["visibility_history"].numpy(),
                after_observable["timestamps"].numpy(),
            )
            assert np.array_equal(after_observable["estimated_velocity"].numpy(), expected_velocity)
            assert np.array_equal(after_observable["velocity_confidence"].numpy(), expected_confidence)

        # The wrapper works on clones; accessing it cannot mutate legacy/base
        # samples or their historical artifacts.
        for before, untouched in zip(original_snapshot, (source[index] for index in range(source.candidate_count)), strict=True):
            assert torch.equal(
                before["observable"]["points_history"], untouched["observable"]["points_history"]
            )
            assert torch.equal(
                before["targets"]["ground_truth_future_mask"],
                untouched["targets"]["ground_truth_future_mask"],
            )
            assert "evaluation_identity_history" not in untouched["hidden_state"]
    finally:
        torch.set_num_threads(old_threads)


def test_group_shared_ranking_corruption_is_repeatable_and_rejects_unknown_mode() -> None:
    source = _ranking_dataset()
    first = GroupSharedRankingCorruptionDataset(source, mode="per_frame_resampling", seed=752)
    second = GroupSharedRankingCorruptionDataset(source, mode="per_frame_resampling", seed=752)
    for index in range(source.candidate_count):
        left, right = first[index], second[index]
        for field in _scene_fields(left):
            assert torch.equal(_scene_fields(left)[field], _scene_fields(right)[field])
        assert torch.equal(
            left["hidden_state"]["evaluation_identity_history"],
            right["hidden_state"]["evaluation_identity_history"],
        )
    with pytest.raises(ValueError, match="unknown group-shared ranking corruption mode"):
        GroupSharedRankingCorruptionDataset(source, mode="not-a-view")  # type: ignore[arg-type]
