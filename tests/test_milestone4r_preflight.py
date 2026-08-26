"""Frozen-protocol and generated-smoke integrity checks for Milestone 4R."""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from actmask.data.milestone4r_robust_pilot import CAMERAS, CONDITIONS, _camera_schedule, _condition_schedule, _mask
from actmask.data.milestone4r_inputs import FAIR_KEYS, backproject_world, load_fair_inputs


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "actmask" / "milestone4r_robust_visual"
SMOKE = OUTPUT / "generation_condition_smoke_v4"
FINAL = OUTPUT / "robust_probe_v2"
EXPECTED_HASH = "8b07ab21338da9bf9d39c8684b9bfb10e348ebe9a0bb4b4d27af17db1af47422"
TASKS = ("MovingCubeIntercept", "SignedMovingWindowPlacement", "FixedPhaseRotatingCaptureWindow")


def _rows(task: str) -> list[dict]:
    return [json.loads(line) for line in (SMOKE / f"{task}_candidates.jsonl").read_text().splitlines()]


def test_preregistered_config_hash_and_balanced_protocol_schedules():
    assert hashlib.sha256((OUTPUT / "preregistered_config.json").read_bytes()).hexdigest() == EXPECTED_HASH
    cameras = _camera_schedule(128)
    assert {camera for _, camera in cameras} == set(CAMERAS)
    assert {split for split, _ in cameras} == {"train", "validation", "test"}
    conditions = _condition_schedule(128)
    counts = [conditions.count(condition) for condition in CONDITIONS]
    assert set(conditions) == set(CONDITIONS)
    assert max(counts) - min(counts) <= 1


def test_missingness_is_deterministic_and_async_by_modality():
    for condition in CONDITIONS:
        first = _mask(9, condition)
        second = _mask(9, condition)
        assert all(np.array_equal(a, b) for a, b in zip(first, second))
    rgb, depth, _ = _mask(9, "async_rgb")
    assert not np.array_equal(rgb, depth)
    rgb, depth, _ = _mask(9, "async_depth")
    assert not np.array_equal(rgb, depth)


def test_smoke_audit_covers_pairing_camera_association_hidden_fields_and_c10_degeneracy():
    audit = json.loads((SMOKE / "robust_audit.json").read_text())
    for task in TASKS:
        result = audit["tasks"][task]
        assert result["candidate_count"] == 220
        assert result["pairs"] == 110
        assert result["condition_coverage"] is True
        assert result["calibration"] == {"camera_count": 7, "passed": True}
        assert result["association"]["passed"] is True
        assert result["appearance"]["world_randomization_observed"] is True
        assert result["appearance"]["metadata_excluded"] is True
        assert result["asynchronous_masks"] == {"async_depth": True, "async_rgb": True}
        assert all(value == 1.0 for value in result["pair_matching"].values())
        assert all(result["schema"].values())
    # The smoke also exposes the eventual preregistered stop condition: the
    # placement family has no within-history C10 outcome diversity.  The audit
    # must flag this rather than allowing an invalid ranking benchmark through.
    assert audit["passed"] is False
    diversity = audit["tasks"]["SignedMovingWindowPlacement"]["candidate_diversity"]
    assert diversity["c10_aligned"] is True
    assert diversity["mixed_success_fraction"] == 0.0


def test_c10_alignment_candidate_diversity_and_no_private_ids_in_storage():
    forbidden = {"branch", "camera", "camera_id", "segmentation", "segmentation_id", "object_id", "future_state", "success", "candidate_outcome"}
    for task in TASKS:
        rows = _rows(task)
        labels = np.load(SMOKE / f"{task}_labels.npz")["success"]
        bundles = np.load(SMOKE / f"{task}_bundles.npz")
        assert not (set(bundles.files) & {"segmentation", "object_id", "branch_id"})
        assert {"visibility", "confidence"} <= set(bundles.files)
        assert np.array_equal(bundles["visibility"], bundles["confidence"])
        groups: dict[tuple[int, int], list[int]] = defaultdict(list)
        for index, row in enumerate(rows):
            assert not (set(row) & forbidden)
            groups[(row["world_id"], row["candidate_slot"])].append(index)
        assert len(groups) == 11 * 10
        assert all(len(indices) == 2 for indices in groups.values())
        for world in range(11):
            candidate_actions = {
                tuple(np.asarray(rows[index]["candidate_actions"], dtype=np.float32).reshape(-1).round(6))
                for index, row in enumerate(rows)
                if row["world_id"] == world
            }
            assert len(candidate_actions) == 10
        assert labels.shape == (220,)


def test_target_occlusion_diagnostics_are_generator_private_and_keep_distractor_visible():
    for task in TASKS:
        diagnostic = json.loads((SMOKE / f"{task}_oracle_association_diagnostic.json").read_text())
        assert diagnostic["generator_private"] is True
        assert len(diagnostic["target_occlusions"]) == 1
        for frame in diagnostic["target_occlusions"][0]["frames"]:
            assert frame["target_visible_after"] == 0
            assert frame["distractor_visible_after"] > 0


def test_fair_loader_excludes_storage_metadata_and_backprojects_world_coordinates():
    fair = load_fair_inputs(SMOKE, "MovingCubeIntercept")
    assert set(fair) == FAIR_KEYS
    assert fair["rgbd_history"].shape == (220, 6, 1024)
    assert fair["world_pointcloud_history"].shape == (220, 6, 768)
    assert all(np.isfinite(value).all() for value in fair.values())
    depth = np.ones((1, 2, 2), dtype=np.float32)
    extrinsic = np.c_[np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32)]
    world = backproject_world(depth, np.eye(3, dtype=np.float32), extrinsic)
    assert np.allclose(world[0, 0, 0], [0.0, 0.0, 1.0])


def test_final_preregistered_integrity_gate_rejects_degenerate_c10_candidate_ranking():
    audit = json.loads((FINAL / "robust_audit.json").read_text())
    decision = json.loads((FINAL / "milestone4r_final_decision.json").read_text())
    runtime = json.loads((FINAL / "runtime_and_storage_report.json").read_text())
    diversity = audit["tasks"]["SignedMovingWindowPlacement"]["candidate_diversity"]
    assert audit["passed"] is False
    assert diversity == {"c10_aligned": True, "histories": 256, "mixed_success_fraction": 0.0}
    assert decision["decision"] == "E. BENCHMARK INVALID"
    assert decision["milestone4m_authorized"] is False
    assert runtime["final_candidate_executions"] == 7680
    assert runtime["nonfinal_v1_accounting"]["combined_candidate_execution_upper_bound"] <= 20000
