from __future__ import annotations

import numpy as np
import pytest

from actmask.data.maniskill_pilot import MODEL_INPUT_KEYS, load_model_inputs
from actmask.experiments.milestone3p_maniskill_baselines import _corrupt_history


def _observable_arrays() -> dict[str, np.ndarray]:
    return {
        "history": np.zeros((2, 4, 20), dtype=np.float32),
        "timestamps": np.zeros((2, 4), dtype=np.float32),
        "visibility": np.ones((2, 4, 1), dtype=np.float32),
        "observation_confidence": np.ones((2, 4, 1), dtype=np.float32),
        "candidate_actions": np.zeros((2, 12, 3), dtype=np.float32),
        "nominal_action_timing": np.zeros((2, 12), dtype=np.float32),
        "tcp_state": np.zeros((2, 3), dtype=np.float32),
    }


def test_maniskill_model_input_loader_allows_only_observable_contract(tmp_path) -> None:
    path = tmp_path / "observable.npz"
    arrays = _observable_arrays()
    np.savez(path, **arrays)
    loaded = load_model_inputs(path)
    assert tuple(loaded) == MODEL_INPUT_KEYS
    assert set(loaded) == set(arrays)

    leaked = tmp_path / "leaked.npz"
    np.savez(leaked, **arrays, future_dynamics=np.zeros(2, dtype=np.float32))
    with pytest.raises(ValueError, match="Unexpected model-input fields"):
        load_model_inputs(leaked)


def test_cross_scene_corruption_is_not_a_candidate_major_local_roll() -> None:
    history = np.arange(8 * 4 * 3, dtype=np.float32).reshape(8, 4, 3)
    mismatched = _corrupt_history(history, "history_mismatched_across_scenes")
    assert not np.array_equal(mismatched, history)
    assert {tuple(row.reshape(-1)) for row in mismatched} == {tuple(row.reshape(-1)) for row in history}
    zeroed = _corrupt_history(history, "zeroed_motion")
    assert np.array_equal(zeroed[:, 0], history[:, -1])
    assert np.array_equal(zeroed[:, -1], history[:, -1])
