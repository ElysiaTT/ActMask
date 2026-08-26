"""Focused group-splitting tests for the hardened ActMask benchmark."""

from __future__ import annotations

from dataclasses import replace
import inspect

import pytest
import torch
from torch.utils.data import DataLoader

from actmask.data import Milestone2DynamicDataset, ToyDynamicDataset
from actmask.data.grouped_splits import (
    OOD_DOMAINS,
    assert_no_group_leakage,
    build_group_manifest,
    split_statistics,
)
from actmask.models.baselines import (
    AnalyticClosestApproachMask,
    FutureEndpointProximityMask,
    MinFutureSpatialDistanceMask,
    TimeAlignedFutureProximityMask,
)


IDENTITY_FIELDS = (
    "group_id",
    "base_scene_id",
    "pair_id",
    "geometry_seed",
    "scenario_seed",
)


def test_group_manifest_has_exact_counts_and_no_identity_overlap() -> None:
    manifest = build_group_manifest(
        master_seed=91,
        total_id_groups=76,
        split_counts={"train": 40, "val": 16, "test": 20},
        ood_groups_per_domain=5,
    )

    assert [len(manifest[name]) for name in ("train", "val", "test")] == [40, 16, 20]
    assert all(len(manifest[name]) == 5 for name in OOD_DOMAINS)
    assert_no_group_leakage(manifest)

    for field in IDENTITY_FIELDS:
        sets = {
            partition: {getattr(record, field) for record in records}
            for partition, records in manifest.items()
        }
        partitions = list(sets)
        for index, first in enumerate(partitions):
            for second in partitions[index + 1 :]:
                assert sets[first].isdisjoint(sets[second])

    all_records = [record for records in manifest.values() for record in records]
    for field in IDENTITY_FIELDS:
        values = [getattr(record, field) for record in all_records]
        assert len(values) == len(set(values))
        assert all(isinstance(value, int) and 0 <= value < 2**63 for value in values)


def test_manifest_is_deterministic_and_seed_sensitive() -> None:
    kwargs = dict(
        total_id_groups=31,
        split_fractions=(0.5, 0.2, 0.3),
        ood_groups_per_domain=3,
    )
    first = build_group_manifest(master_seed=7, **kwargs)
    second = build_group_manifest(master_seed=7, **kwargs)
    changed = build_group_manifest(master_seed=8, **kwargs)

    assert first == second
    assert first != changed
    assert [len(first[name]) for name in ("train", "val", "test")] == [16, 6, 9]
    stats = split_statistics(first)
    assert stats["train"]["groups"] == 16
    assert stats["train"]["samples"] == 32


def test_leakage_checker_rejects_cross_partition_scene_identity() -> None:
    manifest = build_group_manifest(
        master_seed=3,
        total_id_groups=6,
        split_counts={"train": 2, "val": 2, "test": 2},
        ood_groups_per_domain=0,
    )
    leaked = dict(manifest)
    source = manifest["train"][0]
    leaked["val"] = (replace(source, split="val"), *manifest["val"])

    with pytest.raises(ValueError, match="leaks across"):
        assert_no_group_leakage(leaked)


@pytest.mark.parametrize(
    "bad_counts",
    [
        {"train": 2, "val": 2, "test": 1},
        {"train": 2, "val": 2},
        {"train": 2, "val": -1, "test": 5},
    ],
)
def test_manifest_rejects_invalid_exact_counts(bad_counts: dict[str, int]) -> None:
    with pytest.raises((ValueError, TypeError)):
        build_group_manifest(total_id_groups=6, split_counts=bad_counts)


def _small_milestone2_dataset() -> Milestone2DynamicDataset:
    return Milestone2DynamicDataset(
        split="train",
        master_seed=41,
        total_id_groups=6,
        split_counts={"train": 2, "val": 2, "test": 2},
        ood_groups_per_domain=1,
        num_points=32,
        trajectory_steps=12,
    )


@pytest.mark.parametrize(
    "model",
    [
        FutureEndpointProximityMask(),
        MinFutureSpatialDistanceMask(trajectory_steps=11),
        TimeAlignedFutureProximityMask(trajectory_steps=11),
        AnalyticClosestApproachMask(),
    ],
    ids=("future-endpoint", "minimum-spatial", "time-aligned", "analytic"),
)
def test_strong_geometric_baseline_shapes_are_finite_on_cpu(model: torch.nn.Module) -> None:
    batch = next(iter(DataLoader(_small_milestone2_dataset(), batch_size=3)))
    logits = model(batch["points"], batch["velocities"], batch["action"])

    assert logits.shape == (3, 32)
    assert logits.dtype == torch.float32
    assert logits.device.type == "cpu"
    assert torch.isfinite(logits).all()


def test_analytic_closest_approach_has_exact_hand_computable_logits() -> None:
    # The gripper moves from x=0 to x=1 in one second.  Point 0 meets it at
    # t=1; point 1 has the same x motion but remains 0.3 units away in y.
    points = torch.tensor([[[0.5, 0.2, 0.0], [0.5, 0.3, 0.0]]])
    velocities = torch.tensor([[[0.5, -0.2, 0.0], [0.5, 0.0, 0.0]]])
    action = torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.1]])

    logits = AnalyticClosestApproachMask(temperature=0.1)(points, velocities, action)
    assert logits[0, 0].item() == pytest.approx(1.0, abs=1.0e-6)
    assert logits[0, 1].item() == pytest.approx(-2.0, abs=1.0e-6)


def test_time_aligned_baseline_exactly_reproduces_legacy_toy_masks() -> None:
    trajectory_steps = 17
    dataset = ToyDynamicDataset(
        num_samples=16,
        num_points=48,
        seed=73,
        trajectory_steps=trajectory_steps,
    )
    batch = next(iter(DataLoader(dataset, batch_size=len(dataset), shuffle=False)))
    logits = TimeAlignedFutureProximityMask(
        trajectory_steps=trajectory_steps, temperature=0.05
    )(batch["points"], batch["velocities"], batch["action"])

    assert torch.equal((logits >= 0.0).to(torch.float32), batch["mask"])


def test_strong_baselines_have_no_oracle_metadata_input() -> None:
    sample = _small_milestone2_dataset()[0]
    models = (
        FutureEndpointProximityMask(),
        MinFutureSpatialDistanceMask(),
        TimeAlignedFutureProximityMask(),
        AnalyticClosestApproachMask(),
    )
    points = sample["points"].unsqueeze(0)
    velocities = sample["velocities"].unsqueeze(0)
    action = sample["action"].unsqueeze(0)

    for model in models:
        assert tuple(inspect.signature(model.forward).parameters) == (
            "points",
            "velocities",
            "action",
        )
        before = model(points, velocities, action)
        # Oracle-only metadata may be arbitrarily corrupted; it is not accepted
        # by, captured by, or forwarded to any deployable baseline.
        altered_oracle = torch.randn_like(sample["point_trajectory"]) * 1000.0
        altered_distance = torch.zeros_like(sample["min_contact_distance"])
        assert not torch.equal(altered_oracle, sample["point_trajectory"])
        assert not torch.equal(altered_distance, sample["min_contact_distance"])
        after = model(points, velocities, action)
        assert torch.equal(before, after)
