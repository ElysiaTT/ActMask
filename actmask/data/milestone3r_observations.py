"""Deterministic, observable-only corruptions for Milestone 3R."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class CorruptionSpec:
    corruption_id: str
    family: str
    severity: float
    seed: int


PREREGISTERED_SPECS = (
    CorruptionSpec("gaussian_mild", "gaussian", 0.002, 301),
    CorruptionSpec("gaussian_medium", "gaussian", 0.010, 302),
    CorruptionSpec("gaussian_severe", "gaussian", 0.030, 303),
    CorruptionSpec("heteroscedastic_medium", "heteroscedastic", 0.015, 304),
    CorruptionSpec("sparse_outliers", "outliers", 0.040, 305),
    CorruptionSpec("quantized", "quantization", 0.010, 306),
    CorruptionSpec("timestamp_jitter", "timestamp_jitter", 0.012, 307),
    CorruptionSpec("irregular_intervals", "irregular", 0.040, 308),
    CorruptionSpec("random_dropout", "dropout", 0.35, 309),
    CorruptionSpec("last_frame_dropout", "last_dropout", 1.0, 310),
    CorruptionSpec("burst_occlusion", "burst", 2.0, 311),
    CorruptionSpec("observation_latency", "latency", 1.0, 312),
    CorruptionSpec("partial_coordinates", "partial_coordinates", 0.34, 313),
)


def _copy(values: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {key: value.copy() for key, value in values.items()}


def _monotonic(timestamps: np.ndarray) -> np.ndarray:
    # Preserve frame order; the epsilon gives strict order even after equal
    # jittered samples. Absolute times stay observable to the estimator.
    ordered = np.maximum.accumulate(timestamps, axis=1)
    epsilon = np.arange(ordered.shape[1], dtype=np.float32)[None] * 1.0e-5
    return (ordered + epsilon).astype(np.float32)


def apply_corruption(values: dict[str, np.ndarray], spec: CorruptionSpec) -> tuple[dict[str, np.ndarray], dict]:
    """Return a corrupted fair-input dict and an audit-only manifest.

    The function never receives labels, metadata, hidden state, or simulator IDs.
    """
    result = _copy(values)
    rng = np.random.default_rng(spec.seed)
    history = result["history"]
    visibility = result["visibility"]
    confidence = result["observation_confidence"]
    timestamps = result["timestamps"]
    n, frames, dimensions = history.shape
    family = spec.family
    details: dict[str, float | int | list[int]] = {}
    if family == "gaussian":
        history += rng.normal(0.0, spec.severity, history.shape).astype(np.float32)
    elif family == "heteroscedastic":
        speed = np.linalg.norm(np.diff(history, axis=1, prepend=history[:, :1]), axis=2, keepdims=True)
        scale = spec.severity * (0.35 + speed / np.maximum(speed.mean(axis=1, keepdims=True), 1.0e-6))
        history += rng.normal(0.0, 1.0, history.shape).astype(np.float32) * scale.astype(np.float32)
        details["observable_speed_scaled"] = 1
    elif family == "outliers":
        mask = rng.random((n, frames, 1)) < spec.severity
        magnitude = 0.12
        history += mask * rng.normal(0.0, magnitude, history.shape).astype(np.float32)
        details.update(outlier_rate=float(mask.mean()), outlier_magnitude=magnitude)
    elif family == "quantization":
        history[:] = np.round(history / spec.severity) * spec.severity
        details["bin_width"] = spec.severity
    elif family == "timestamp_jitter":
        timestamps[:] = _monotonic(timestamps + rng.normal(0.0, spec.severity, timestamps.shape).astype(np.float32))
    elif family == "irregular":
        increments = rng.uniform(0.35, 1.65, (n, frames)).astype(np.float32)
        increments[:, 0] = 0.0
        increments *= spec.severity / np.maximum(increments[:, -1:], 1.0e-6)
        timestamps[:] = _monotonic(timestamps[:, :1] + np.cumsum(increments, axis=1))
    elif family in {"dropout", "last_dropout", "burst", "latency", "partial_coordinates"}:
        mask = np.zeros((n, frames), dtype=bool)
        if family == "dropout":
            mask = rng.random((n, frames)) < spec.severity
            mask[:, 0] = False
        elif family == "last_dropout":
            mask[:, -1] = True
        elif family == "burst":
            width = int(spec.severity)
            start = rng.integers(0, max(1, frames - width + 1), n)
            for row, begin in enumerate(start):
                mask[row, begin : begin + width] = True
            details["burst_width"] = width
        elif family == "latency":
            # Retain a visible historical end-state but make it explicitly old.
            history[:, -1] = history[:, -2]
            timestamps[:, -1] = timestamps[:, -2] - 0.05 * spec.severity
            timestamps[:] = _monotonic(timestamps)
            mask[:, -1] = True
        else:
            coordinate_count = max(1, int(round(dimensions * spec.severity)))
            axes = np.stack([rng.choice(dimensions, coordinate_count, replace=False) for _ in range(n)])
            for row, selected in enumerate(axes):
                history[row, :, selected] = 0.0
            details["coordinates_removed"] = coordinate_count
            details["axes_seeded_per_example"] = 1
        if family != "partial_coordinates":
            history[mask] = 0.0
            visibility[mask] = 0.0
            confidence[mask] = 0.0
            details["dropped_frame_rate"] = float(mask.mean())
    else:
        raise ValueError(f"Unknown corruption family: {family}")
    result["history"] = history.astype(np.float32)
    result["timestamps"] = timestamps.astype(np.float32)
    result["visibility"] = visibility.astype(np.float32)
    result["observation_confidence"] = confidence.astype(np.float32)
    audit = dict(**asdict(spec), details=details, timestamp_monotonic=bool(np.all(np.diff(result["timestamps"], axis=1) > 0)))
    return result, audit
