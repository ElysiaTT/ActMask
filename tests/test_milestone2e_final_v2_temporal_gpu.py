"""Focused non-CUDA checks for the temporal GPU qualification gate."""

from __future__ import annotations

import json

import pytest

from actmask.experiments.milestone2e_final_v2_temporal_gpu import _utility_qualification


def test_utility_qualification_rejects_missing_artifact(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "actmask.experiments.milestone2e_final_v2_temporal_gpu.UTILITY_QUALIFICATION_PATH",
        tmp_path / "missing.json",
    )
    with pytest.raises(FileNotFoundError):
        _utility_qualification()


def test_utility_qualification_requires_all_utility_and_ranking_checks(monkeypatch, tmp_path) -> None:
    path = tmp_path / "qualification.json"
    path.write_text(json.dumps({
        "artifact_schema": "milestone2e-final-v2-cpu-cuda-equivalence-v1",
        "per_seed": [
            {"outputs": {"utility_logits": {"allclose_atol_1e-5_rtol_1e-4": True}}, "rankings_identical_for_complete_groups": True}
            for _ in range(5)
        ],
    }))
    monkeypatch.setattr("actmask.experiments.milestone2e_final_v2_temporal_gpu.UTILITY_QUALIFICATION_PATH", path)
    assert _utility_qualification()["per_seed"]
