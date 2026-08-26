"""Deterministic paired statistical inference for Milestone 2C."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def _paired_arrays(first: Sequence[float], second: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    left = np.asarray(first, dtype=np.float64).reshape(-1)
    right = np.asarray(second, dtype=np.float64).reshape(-1)
    if left.shape != right.shape or left.size < 2:
        raise ValueError("paired statistics require equally sized arrays with at least two groups")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("paired values must be finite")
    return left, right


def bootstrap_ci(
    values: Sequence[float], *, seed: int, resamples: int = 2000, confidence: float = 0.95
) -> dict[str, float | int]:
    """Nonparametric group bootstrap interval for a macro metric."""

    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size < 2 or not np.isfinite(array).all():
        raise ValueError("bootstrap requires at least two finite values")
    if resamples < 100:
        raise ValueError("resamples must be at least 100")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0,1)")
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, array.size, size=(int(resamples), array.size))
    estimates = array[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return {
        "mean": float(array.mean()),
        "ci_low": float(np.quantile(estimates, alpha)),
        "ci_high": float(np.quantile(estimates, 1.0 - alpha)),
        "groups": int(array.size),
        "resamples": int(resamples),
    }


def paired_bootstrap_ci(
    first: Sequence[float],
    second: Sequence[float],
    *,
    seed: int,
    resamples: int = 2000,
    confidence: float = 0.95,
) -> dict[str, float | int]:
    """Paired bootstrap for first-minus-second group-macro metric deltas."""

    left, right = _paired_arrays(first, second)
    difference = left - right
    result = bootstrap_ci(
        difference, seed=seed, resamples=resamples, confidence=confidence
    )
    result.update(
        {
            "first_mean": float(left.mean()),
            "second_mean": float(right.mean()),
            "delta": float(difference.mean()),
            "ci_excludes_zero": bool(result["ci_low"] > 0.0 or result["ci_high"] < 0.0),
        }
    )
    return result


def paired_permutation_test(
    first: Sequence[float],
    second: Sequence[float],
    *,
    seed: int,
    permutations: int = 4096,
) -> dict[str, float | int]:
    """Two-sided sign-flip paired permutation test for mean delta."""

    left, right = _paired_arrays(first, second)
    difference = left - right
    observed = float(difference.mean())
    rng = np.random.default_rng(int(seed))
    signs = rng.integers(0, 2, size=(int(permutations), difference.size), dtype=np.int8)
    signs = signs * 2 - 1
    null = (signs * difference[None]).mean(axis=1)
    exceed = int(np.count_nonzero(np.abs(null) >= abs(observed)))
    return {
        "delta": observed,
        "p_value": float((exceed + 1) / (int(permutations) + 1)),
        "permutations": int(permutations),
        "groups": int(difference.size),
    }


def paired_summary(
    first: Sequence[float],
    second: Sequence[float],
    *,
    seed: int,
    resamples: int = 2000,
    permutations: int = 4096,
) -> dict[str, Any]:
    """Combine bootstrap and permutation evidence with win/tie/loss counts."""

    left, right = _paired_arrays(first, second)
    difference = left - right
    result: dict[str, Any] = paired_bootstrap_ci(
        left, right, seed=seed, resamples=resamples
    )
    result.update(
        paired_permutation_test(left, right, seed=seed + 1, permutations=permutations)
    )
    result.update(
        {
            "wins": int(np.count_nonzero(difference > 1.0e-12)),
            "ties": int(np.count_nonzero(np.abs(difference) <= 1.0e-12)),
            "losses": int(np.count_nonzero(difference < -1.0e-12)),
        }
    )
    return result


__all__ = [
    "bootstrap_ci",
    "paired_bootstrap_ci",
    "paired_permutation_test",
    "paired_summary",
]
