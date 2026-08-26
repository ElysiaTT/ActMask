from __future__ import annotations

import numpy as np
import pytest

from actmask.data.milestone3q_signed import _pair_audit
from actmask.experiments.milestone3q_order_invariant_audit import (
    multiset_features,
    order_invariant_features,
    sorted_frame_features,
    velocity_magnitude_features,
)


def test_closed_loop_reversal_matches_all_unordered_statistics() -> None:
    # [A, B, C, A] and [A, C, B, A] keep the decision state A at the
    # final frame while exactly reversing the interior temporal direction.
    first = np.asarray([[[0.0, 0.0], [1.0, 0.0], [0.5, 0.8660254], [0.0, 0.0]]], dtype=np.float32)
    reverse = np.asarray([[[0.0, 0.0], [0.5, 0.8660254], [1.0, 0.0], [0.0, 0.0]]], dtype=np.float32)
    assert np.array_equal(first[:, -1], reverse[:, -1])
    assert np.array_equal(order_invariant_features(first), order_invariant_features(reverse))
    assert np.array_equal(velocity_magnitude_features(first), velocity_magnitude_features(reverse))
    assert np.array_equal(sorted_frame_features(first), sorted_frame_features(reverse))
    assert np.array_equal(multiset_features(first), multiset_features(reverse))
    assert not np.array_equal(first, reverse)


def test_signed_pair_audit_rejects_nonflipped_execution_label() -> None:
    history = np.asarray(
        [
            [[0.0], [1.0], [0.5], [0.0]],
            [[0.0], [0.5], [1.0], [0.0]],
        ],
        dtype=np.float32,
    )
    actions = np.zeros((2, 3, 3), dtype=np.float32)
    rows = [
        {"signed_group": "g", "success": False},
        {"signed_group": "g", "success": False},
    ]
    with pytest.raises(AssertionError, match="did not flip simulator success"):
        _pair_audit(rows, history, actions)
