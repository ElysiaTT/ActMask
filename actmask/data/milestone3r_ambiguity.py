"""Observable multi-object/correspondence ambiguity layer for 3R."""

from __future__ import annotations

import numpy as np


AMBIGUITY_TASKS = (
    "moving_cube_intercept",
    "signed_moving_window_placement",
    "fixed_phase_rotating_capture_window",
)


def build_ambiguity(history: np.ndarray, visibility: np.ndarray, *, seed: int = 401) -> tuple[dict[str, np.ndarray], dict]:
    """Create fair two-object observations without storing any object identity.

    Slot zero is used only to synthesize the diagnostic oracle outside fair
    inputs. Fair variants either estimate correspondence, use soft weights, or
    consume the unordered object set directly.
    """
    rng = np.random.default_rng(seed)
    target = history.astype(np.float32, copy=True)
    n, frames, dimensions = target.shape
    phase = np.linspace(-1.0, 1.0, frames, dtype=np.float32)[None, :, None]
    offset = rng.uniform(-0.06, 0.06, (n, 1, dimensions)).astype(np.float32)
    # Similar magnitude, crossing trajectories, and late divergence. The
    # construction uses only observed target coordinates; it does not inspect
    # labels, simulator IDs, or futures.
    distractor = target[:, ::-1].copy() + offset + phase * rng.uniform(0.015, 0.045, (n, 1, dimensions)).astype(np.float32)
    distractor[:, -1] += rng.uniform(0.025, 0.055, (n, dimensions)).astype(np.float32)
    objects = np.stack((target, distractor), axis=2)
    object_visibility = np.repeat(visibility[:, :, None, :], 2, axis=2).astype(np.float32)
    # Target is temporarily occluded during the intermediate crossing while
    # the distractor remains visible. This is a fair missing observation.
    object_visibility[:, 1:-1, 0] = 0.0
    objects[:, 1:-1, 0] = 0.0

    # Nearest-neighbour correspondence starts from the final candidate track;
    # at crossing it can select the distractor. No identity is supplied.
    estimated = np.empty_like(target)
    estimated[:, -1] = objects[:, -1, 0]
    for t in range(frames - 2, -1, -1):
        distance = np.linalg.norm(objects[:, t] - estimated[:, t + 1, None], axis=2)
        choice = distance.argmin(axis=1)
        estimated[:, t] = objects[np.arange(n), t, choice]
    distance = np.linalg.norm(objects - estimated[:, :, None], axis=3)
    weights = np.exp(-distance / 0.03)
    weights /= weights.sum(axis=2, keepdims=True)
    soft = (objects * weights[..., None]).sum(axis=2)
    fair = dict(
        # no object/track/simulator identity field exists in the fair schema
        object_history=objects,
        object_visibility=object_visibility,
        nearest_neighbor_history=estimated,
        soft_correspondence_history=soft,
        soft_correspondence_weights=weights.astype(np.float32),
    )
    diagnostic = dict(
        seed=seed,
        target_occluded_frames=list(range(1, frames - 1)),
        mechanisms=["similar_distractor", "crossing_trajectories", "temporary_occlusion", "near_parallel_late_divergence", "identity_swap_diagnostic"],
        oracle_identity_available_only_for_diagnostics=True,
    )
    return fair, diagnostic
