"""Unit tests for device-independent GPU-equivalence helpers."""

from __future__ import annotations

import torch

from actmask.experiments.milestone2e_gpu_equivalence import _complete_group_rankings, _to_device


def test_to_device_preserves_nested_mapping_on_cpu() -> None:
    payload = {"x": torch.tensor([1.0]), "nested": (torch.tensor([2]), "metadata")}
    moved = _to_device(payload, torch.device("cpu"))
    assert moved["x"].device.type == "cpu"
    assert moved["nested"][0].device.type == "cpu"
    assert moved["nested"][1] == "metadata"


def test_complete_group_rankings_ignores_partial_group() -> None:
    scores = torch.arange(25, dtype=torch.float32).numpy()
    groups = torch.tensor([1] * 20 + [2] * 5)
    result = _complete_group_rankings(scores, groups)
    assert list(result) == ["1"]
    assert result["1"] == list(range(19, -1, -1))
