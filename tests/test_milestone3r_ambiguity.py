from __future__ import annotations

import numpy as np

from actmask.data.milestone3r_ambiguity import build_ambiguity


def test_ambiguity_is_deterministic_and_hides_identity() -> None:
    history = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5) / 20.0
    visibility = np.ones((3, 4, 1), dtype=np.float32)
    first, diagnostic = build_ambiguity(history, visibility, seed=17)
    second, _ = build_ambiguity(history, visibility, seed=17)
    assert all(np.array_equal(first[key], second[key]) for key in first)
    assert not any("id" in key.lower() for key in first)
    assert diagnostic["oracle_identity_available_only_for_diagnostics"]
    assert np.all(first["object_visibility"][:, 1:-1, 0] == 0)
