from __future__ import annotations

import numpy as np

from actmask.data.milestone3r_observations import PREREGISTERED_SPECS, apply_corruption


def _values() -> dict[str, np.ndarray]:
    history = np.arange(4 * 4 * 6, dtype=np.float32).reshape(4, 4, 6) / 100.0
    return dict(
        history=history,
        timestamps=np.tile(np.asarray([-0.15, -0.10, -0.05, 0.0], dtype=np.float32), (4, 1)),
        visibility=np.ones((4, 4, 1), dtype=np.float32),
        observation_confidence=np.ones((4, 4, 1), dtype=np.float32),
        candidate_actions=np.ones((4, 3, 3), dtype=np.float32),
        nominal_action_timing=np.ones((4, 3), dtype=np.float32),
        tcp_state=np.ones((4, 3), dtype=np.float32),
    )


def test_corruptions_are_deterministic_and_preserve_nonhistory_inputs() -> None:
    source = _values()
    for spec in PREREGISTERED_SPECS:
        first, audit = apply_corruption(source, spec)
        second, _ = apply_corruption(source, spec)
        for key in first:
            assert np.array_equal(first[key], second[key])
        assert np.array_equal(first["candidate_actions"], source["candidate_actions"])
        assert np.array_equal(first["tcp_state"], source["tcp_state"])
        assert audit["timestamp_monotonic"]


def test_dropout_masks_are_explicit() -> None:
    source = _values()
    spec = next(item for item in PREREGISTERED_SPECS if item.family == "last_dropout")
    corrupted, _ = apply_corruption(source, spec)
    assert np.all(corrupted["visibility"][:, -1] == 0)
    assert np.all(corrupted["observation_confidence"][:, -1] == 0)
    assert np.all(corrupted["history"][:, -1] == 0)
