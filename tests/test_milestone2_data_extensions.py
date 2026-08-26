"""Focused invariants for Milestone-2 data corruption and ranking views."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from actmask.data import (
    ActionRankingDataset,
    Milestone2DynamicDataset,
    compute_time_aligned_contact,
)
from actmask.data.grouped_splits import assert_no_group_leakage, build_group_manifest
from actmask.data.milestone2_dataset import ROLE_BACKGROUND
from actmask.models.ablations import ShuffledTimeDataset, deterministic_derangement


def _m2_dataset(
    *, point_dropout_rate: float = 0.0, temporal_velocity_delay: float = 0.0
) -> Milestone2DynamicDataset:
    return Milestone2DynamicDataset(
        split="train",
        master_seed=812,
        total_id_groups=24,
        split_counts={"train": 12, "val": 6, "test": 6},
        ood_groups_per_domain=1,
        num_points=48,
        trajectory_steps=16,
        point_noise_std=0.003,
        velocity_noise_std=0.005,
        temporal_velocity_delay=temporal_velocity_delay,
        point_dropout_rate=point_dropout_rate,
    )


def test_trajectory_family_identity_is_grouped_and_checked_for_leakage() -> None:
    manifest = build_group_manifest(
        master_seed=71,
        total_id_groups=12,
        split_counts={"train": 6, "val": 3, "test": 3},
        ood_groups_per_domain=1,
    )
    assert_no_group_leakage(manifest)
    for records in manifest.values():
        for record in records:
            assert record.trajectory_family_id == record.group_id

    all_records = [record for records in manifest.values() for record in records]
    source = manifest["train"][0]
    leaked_record = replace(
        source,
        group_id=max(record.group_id for record in all_records) + 1,
        base_scene_id=max(record.base_scene_id for record in all_records) + 1,
        pair_id=max(record.pair_id for record in all_records) + 1,
        geometry_seed=max(record.geometry_seed for record in all_records) + 1,
        scenario_seed=max(record.scenario_seed for record in all_records) + 1,
        split="val",
        # Keep the original trajectory-family identity: this is the exact
        # leakage case that group/base/pair checks alone cannot see.
        trajectory_family_id=source.trajectory_family_id,
    )
    leaked = dict(manifest)
    leaked["val"] = (leaked_record, *manifest["val"])
    with pytest.raises(ValueError, match="trajectory_family_id"):
        assert_no_group_leakage(leaked)

    dataset = _m2_dataset()
    first, second = dataset[0], dataset[1]
    assert first["trajectory_family_id"].item() == second["trajectory_family_id"].item()


def test_safe_point_dropout_is_fixed_shape_deterministic_and_preserves_labels() -> None:
    clean = _m2_dataset(point_dropout_rate=0.0)
    dropped = _m2_dataset(point_dropout_rate=0.25)
    repeated = _m2_dataset(point_dropout_rate=0.25)

    for index in range(0, len(dropped), 2):
        clean_first, clean_second = clean[index], clean[index + 1]
        first, second = dropped[index], dropped[index + 1]
        again = repeated[index]
        dropout = first["point_dropout_mask"]

        assert dropout.dtype is torch.bool
        assert dropout.shape == first["mask"].shape
        assert torch.equal(dropout, ~first["valid_point_mask"])
        assert torch.equal(dropout, second["point_dropout_mask"])
        assert torch.equal(dropout, again["point_dropout_mask"])
        assert int(dropout.sum()) > 0
        assert torch.equal(first["mask"], clean_first["mask"])
        assert torch.equal(second["mask"], clean_second["mask"])
        assert torch.equal(first["point_trajectory"], clean_first["point_trajectory"])
        assert torch.equal(second["point_trajectory"], clean_second["point_trajectory"])
        assert torch.equal(first["points"], second["points"])
        assert torch.all(first["point_roles"][dropout] == ROLE_BACKGROUND)
        assert torch.all(first["mask"][dropout] == 0.0)
        assert torch.all(second["mask"][dropout] == 0.0)
        assert torch.all(first["velocities"][dropout] == 0.0)
        assert torch.all(second["velocities"][dropout] == 0.0)
        assert torch.all(first["points"][dropout] != clean_first["points"][dropout])
        assert first["parameter_values"].numel() == 11

    with pytest.raises(ValueError, match="point_dropout_rate"):
        _m2_dataset(point_dropout_rate=1.01)


def test_temporal_velocity_delay_changes_only_observation_not_physical_labels() -> None:
    common = dict(
        split="train",
        master_seed=812,
        total_id_groups=24,
        split_counts={"train": 12, "val": 6, "test": 6},
        ood_groups_per_domain=1,
        num_points=48,
        trajectory_steps=16,
        point_noise_std=0.0,
        velocity_noise_std=0.0,
        point_dropout_rate=0.0,
    )
    current = Milestone2DynamicDataset(**common, temporal_velocity_delay=0.0)
    delayed = Milestone2DynamicDataset(**common, temporal_velocity_delay=0.04)
    index = next(
        index
        for index in range(len(current))
        if current[index]["scenario"] == "constant_acceleration"
    )
    baseline, stale = current[index], delayed[index]

    # The physical scene, action, exact trajectory, and label stay unchanged;
    # only the model-facing velocity observation is temporally stale.
    for key in ("points", "action", "mask", "point_trajectory", "true_velocities"):
        assert torch.equal(stale[key], baseline[key])
    assert stale["temporal_velocity_delay"].item() == pytest.approx(0.04)
    assert stale["parameter_values"].numel() == 11
    assert torch.allclose(
        stale["velocities"] - baseline["velocities"],
        -stale["accelerations"] * 0.04,
        atol=1.0e-7,
        rtol=0.0,
    )
    assert not torch.equal(stale["velocities"], baseline["velocities"])

    with pytest.raises(ValueError, match="temporal_velocity_delay"):
        Milestone2DynamicDataset(**common, temporal_velocity_delay=-0.01)


class _TimeShuffleFixture(Dataset[dict[str, object]]):
    def __init__(self, count: int = 6) -> None:
        self.rows = []
        for index in range(count):
            action = torch.tensor(
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, float(index + 10), 0.1],
                dtype=torch.float32,
            )
            self.rows.append(
                {
                    "points": torch.full((3, 3), float(index)),
                    "velocities": torch.full((3, 3), float(index + 20)),
                    "action": action,
                    "mask": torch.tensor([1.0, 0.0, 0.0]),
                    "sample_id": torch.tensor(100 + index),
                    "pair_id": torch.tensor(200 + index),
                }
            )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        return {
            key: value.clone() if isinstance(value, torch.Tensor) else value
            for key, value in self.rows[index].items()
        }


def test_time_shuffle_replaces_only_duration_with_global_donor_mapping() -> None:
    base = _TimeShuffleFixture()
    shuffled = ShuffledTimeDataset(base, seed=73)
    assert shuffled.donor_indices == deterministic_derangement(len(base), seed=73)
    donor_ids_by_batch: list[int] = []
    for batch in DataLoader(shuffled, batch_size=3, shuffle=False):
        donor_ids_by_batch.extend(int(value) for value in batch["time_donor_sample_id"].tolist())

    for index, donor_index in enumerate(shuffled.donor_indices):
        original, donor, changed = base[index], base[donor_index], shuffled[index]
        assert torch.equal(changed["action"][:6], original["action"][:6])
        assert torch.equal(changed["action"][7:], original["action"][7:])
        assert changed["action"][6].item() == donor["action"][6].item()
        for key in ("points", "velocities", "mask", "sample_id", "pair_id"):
            assert torch.equal(changed[key], original[key])
        assert changed["time_donor_sample_id"] == int(donor["sample_id"])
        assert changed["action_duration_donor_sample_id"] == int(donor["sample_id"])
    assert donor_ids_by_batch == [shuffled[index]["time_donor_sample_id"] for index in range(len(base))]


def test_action_ranking_dataset_has_five_recomputed_candidates_per_scene() -> None:
    source = _m2_dataset(point_dropout_rate=0.15)
    ranking = ActionRankingDataset(source)
    repeated = ActionRankingDataset(_m2_dataset(point_dropout_rate=0.15))
    assert len(ranking) == len(ranking.source_indices) * 5
    assert len(ranking.source_indices) == len(source) // 2

    for scene_index in range(len(ranking.source_indices)):
        candidates = [ranking[scene_index * 5 + offset] for offset in range(5)]
        assert [candidate["candidate_type"] for candidate in candidates] == list(
            ranking.candidate_types
        )
        assert [candidate["candidate_id"].item() for candidate in candidates] == list(range(5))
        assert all(
            candidate["ranking_scene_id"].item()
            == candidates[0]["ranking_scene_id"].item()
            for candidate in candidates
        )
        assert all(
            candidate["candidate_set_id"].item()
            == candidates[0]["ranking_scene_id"].item()
            for candidate in candidates
        )
        assert len({candidate["sample_id"].item() for candidate in candidates}) == 5
        assert len({candidate["source_sample_id"].item() for candidate in candidates}) == 1
        assert candidates[0]["success"].item() == 1.0
        assert candidates[1]["success"].item() == 0.0
        assert candidates[3]["success"].item() == 0.0
        assert candidates[4]["success"].item() == 0.0
        assert candidates[2]["action"][6].item() != candidates[0]["action"][6].item()

        utilities = [candidate["target_utility"].item() for candidate in candidates]
        oracle_id = candidates[0]["oracle_candidate_id"].item()
        assert oracle_id == max(range(5), key=lambda item: (utilities[item], -item))
        assert sum(bool(candidate["is_oracle_candidate"].item()) for candidate in candidates) == 1
        assert all(
            candidate["candidate_utility"].item()
            == pytest.approx(candidate["target_utility"].item())
            for candidate in candidates
        )
        assert all(
            candidate["oracle_utility"].item() == pytest.approx(utilities[oracle_id])
            for candidate in candidates
        )
        for candidate in candidates:
            mask, minimum = compute_time_aligned_contact(
                candidate["point_trajectory"].numpy(),
                candidate["gripper_trajectory"].numpy(),
                float(candidate["action"][7].item()),
            )
            assert torch.equal(candidate["mask"], torch.from_numpy(mask))
            assert torch.allclose(
                candidate["min_contact_distance"], torch.from_numpy(minimum)
            )

    # Source construction and candidate labels are deterministic, not cached
    # artifacts depending on call order.
    for index in (0, 2, len(ranking) - 1):
        first, second = ranking[index], repeated[index]
        assert first["candidate_type"] == second["candidate_type"]
        assert torch.equal(first["action"], second["action"])
        assert torch.equal(first["mask"], second["mask"])
        assert torch.equal(first["ranking_scene_id"], second["ranking_scene_id"])
