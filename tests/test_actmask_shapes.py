"""Fast, deterministic CPU tests for the ActMask milestone-1 contract."""

from __future__ import annotations

from collections import defaultdict

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from actmask.data import ToyDynamicDataset
from actmask.eval.evaluate import paired_scene_consistency, success_ranking_accuracy
from actmask.models import (
    ActMaskModel,
    ActionProximityMask,
    MotionMagnitudeMask,
    NoMask,
)
from actmask.utils import BinaryMaskMetrics, binary_mask_metrics


def _paired_dataset(seed: int = 17) -> ToyDynamicDataset:
    # Eight samples produce one complete pair for each of the four scenarios.
    return ToyDynamicDataset(num_samples=8, num_points=48, seed=seed, trajectory_steps=12)


def test_dataset_shapes_dtypes_and_action_schema() -> None:
    dataset = _paired_dataset()
    assert len(dataset) == 8
    sample = dataset[0]

    assert set(sample) == {
        "points",
        "velocities",
        "action",
        "mask",
        "success",
        "pair_id",
        "variant_id",
        "scenario",
    }
    assert sample["points"].shape == (48, 3)
    assert sample["velocities"].shape == (48, 3)
    assert sample["action"].shape == (8,)
    assert sample["mask"].shape == (48,)
    assert sample["success"].shape == torch.Size([])
    assert sample["pair_id"].shape == torch.Size([])
    assert sample["variant_id"].shape == torch.Size([])

    for key in ("points", "velocities", "action", "mask", "success"):
        assert sample[key].dtype == torch.float32
        assert sample[key].device.type == "cpu"
        assert torch.isfinite(sample[key]).all()
    assert sample["pair_id"].dtype == torch.int64
    assert sample["variant_id"].dtype == torch.int64
    assert isinstance(sample["scenario"], str)
    assert float(sample["action"][6]) > 0.0  # duration in seconds
    assert float(sample["action"][7]) > 0.0  # gripper radius
    assert set(torch.unique(sample["mask"]).tolist()).issubset({0.0, 1.0})


def test_dataset_is_deterministic_without_shared_mutable_samples() -> None:
    first = _paired_dataset(seed=123)
    second = _paired_dataset(seed=123)
    different_seed = _paired_dataset(seed=124)

    tensor_keys = (
        "points",
        "velocities",
        "action",
        "mask",
        "success",
        "pair_id",
        "variant_id",
    )
    for index in range(len(first)):
        for key in tensor_keys:
            assert torch.equal(first[index][key], second[index][key])
        assert first[index]["scenario"] == second[index]["scenario"]

    assert not torch.equal(first[0]["points"], different_seed[0]["points"])
    returned = first[0]
    original_value = first[0]["points"][0, 0].item()
    returned["points"][0, 0] += 100.0
    assert first[0]["points"][0, 0].item() == original_value


def test_all_four_counterfactual_pair_scenarios() -> None:
    dataset = _paired_dataset()
    pairs: dict[int, list[dict[str, object]]] = defaultdict(list)
    for index in range(len(dataset)):
        sample = dataset[index]
        pairs[int(sample["pair_id"].item())].append(sample)

    assert len(pairs) == 4
    expected_scenarios = {
        "opposite_velocity",
        "different_action_direction",
        "different_timing",
        "interception_success",
    }
    assert {str(samples[0]["scenario"]) for samples in pairs.values()} == expected_scenarios

    by_scenario: dict[str, tuple[dict[str, object], dict[str, object]]] = {}
    for samples in pairs.values():
        assert len(samples) == 2
        samples.sort(key=lambda item: int(item["variant_id"].item()))  # type: ignore[union-attr]
        first, second = samples
        assert first["scenario"] == second["scenario"]
        assert {int(first["variant_id"].item()), int(second["variant_id"].item())} == {0, 1}  # type: ignore[union-attr]
        assert torch.equal(first["points"], second["points"])  # type: ignore[arg-type]
        assert not torch.equal(first["mask"], second["mask"])  # type: ignore[arg-type]
        by_scenario[str(first["scenario"])] = (first, second)

    first, second = by_scenario["opposite_velocity"]
    assert torch.equal(first["action"], second["action"])  # type: ignore[arg-type]
    assert not torch.equal(first["velocities"], second["velocities"])  # type: ignore[arg-type]

    first, second = by_scenario["different_action_direction"]
    assert torch.equal(first["velocities"], second["velocities"])  # type: ignore[arg-type]
    assert torch.equal(first["action"][:3], second["action"][:3])  # type: ignore[index]
    assert not torch.equal(first["action"][3:6], second["action"][3:6])  # type: ignore[index]
    assert torch.equal(first["action"][6:], second["action"][6:])  # type: ignore[index]

    first, second = by_scenario["different_timing"]
    assert torch.equal(first["velocities"], second["velocities"])  # type: ignore[arg-type]
    assert torch.equal(first["action"][:6], second["action"][:6])  # type: ignore[index]
    assert not torch.equal(first["action"][6:7], second["action"][6:7])  # type: ignore[index]
    assert torch.equal(first["action"][7:], second["action"][7:])  # type: ignore[index]

    first, second = by_scenario["interception_success"]
    assert torch.equal(first["velocities"], second["velocities"])  # type: ignore[arg-type]
    assert not torch.equal(first["action"], second["action"])  # type: ignore[arg-type]
    assert {float(first["success"].item()), float(second["success"].item())} == {0.0, 1.0}  # type: ignore[union-attr]


@pytest.mark.parametrize(
    "model_factory",
    [
        NoMask,
        MotionMagnitudeMask,
        ActionProximityMask,
        lambda: ActMaskModel(
            action_dim=8,
            point_hidden_dim=16,
            action_hidden_dim=8,
            fusion_hidden_dim=16,
        ),
    ],
    ids=("no-mask", "motion-magnitude", "action-proximity", "learned-actmask"),
)
def test_all_model_forward_shapes_on_cpu(model_factory: object) -> None:
    batch = next(iter(DataLoader(_paired_dataset(), batch_size=3, shuffle=False)))
    model = model_factory()  # type: ignore[operator]
    logits = model(batch["points"], batch["velocities"], batch["action"])

    assert logits.shape == (3, 48)
    assert logits.dtype == torch.float32
    assert logits.device.type == "cpu"
    assert torch.isfinite(logits).all()


def test_bce_loss_and_one_cpu_optimizer_step() -> None:
    torch.manual_seed(9)
    batch = next(iter(DataLoader(_paired_dataset(), batch_size=2, shuffle=False)))
    model = ActMaskModel(
        action_dim=8,
        point_hidden_dim=16,
        action_hidden_dim=8,
        fusion_hidden_dim=16,
    ).cpu()
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-2)
    criterion = nn.BCEWithLogitsLoss()
    before = [parameter.detach().clone() for parameter in model.parameters()]

    optimizer.zero_grad(set_to_none=True)
    logits = model(batch["points"], batch["velocities"], batch["action"])
    loss = criterion(logits, batch["mask"])
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss.item() > 0.0
    loss.backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    optimizer.step()

    assert all(parameter.device.type == "cpu" for parameter in model.parameters())
    assert any(
        not torch.equal(old, new.detach())
        for old, new in zip(before, model.parameters(), strict=True)
    )


def test_binary_metrics_have_exact_confusion_counts_and_ratios() -> None:
    probabilities = torch.tensor([[0.9, 0.8, 0.2, 0.1]], dtype=torch.float32)
    targets = torch.tensor([[1.0, 0.0, 1.0, 0.0]], dtype=torch.float32)
    expected = {
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
        "iou": 1.0 / 3.0,
        "accuracy": 0.5,
        "tp": 1,
        "fp": 1,
        "fn": 1,
        "tn": 1,
    }

    metrics = binary_mask_metrics(probabilities, targets, threshold=0.5, from_logits=False)
    for key, expected_value in expected.items():
        if isinstance(expected_value, float):
            assert metrics[key] == pytest.approx(expected_value, abs=1.0e-12)
        else:
            assert metrics[key] == expected_value

    accumulator = BinaryMaskMetrics(threshold=0.5, from_logits=False)
    accumulator.update(probabilities[:, :2], targets[:, :2])
    accumulator.update(probabilities[:, 2:], targets[:, 2:])
    accumulated = accumulator.compute()
    assert accumulated == pytest.approx(expected, abs=1.0e-12)

    empty_positive_case = binary_mask_metrics(
        torch.zeros(4), torch.zeros(4), threshold=0.5, from_logits=False
    )
    assert empty_positive_case == {
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "iou": 0.0,
        "accuracy": 1.0,
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "tn": 4,
    }


def test_paired_consistency_and_success_ranking_are_exact() -> None:
    records = [
        {
            "scenario": "pair-test",
            "pair_id": 0,
            "variant_id": 0,
            "success": 1.0,
            "score": 0.8,
            "prediction": torch.tensor([True, False]),
            "target": torch.tensor([True, False]),
        },
        {
            "scenario": "pair-test",
            "pair_id": 0,
            "variant_id": 1,
            "success": 0.0,
            "score": 0.2,
            "prediction": torch.tensor([False, True]),
            "target": torch.tensor([False, True]),
        },
        {
            "scenario": "pair-test",
            "pair_id": 1,
            "variant_id": 0,
            "success": 1.0,
            "score": 0.4,
            "prediction": torch.tensor([False, False]),
            "target": torch.tensor([True, False]),
        },
        {
            "scenario": "pair-test",
            "pair_id": 1,
            "variant_id": 1,
            "success": 0.0,
            "score": 0.4,
            "prediction": torch.tensor([False, False]),
            "target": torch.tensor([False, False]),
        },
    ]

    # Pair 0 has perfect predicted/target change-mask IoU; pair 1 has zero.
    assert paired_scene_consistency(records) == pytest.approx(0.5, abs=1.0e-12)
    # Pair 0 ranks correctly (1.0); pair 1 is tied (0.5).
    assert success_ranking_accuracy(records) == pytest.approx(0.75, abs=1.0e-12)
