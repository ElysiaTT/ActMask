"""Integrity, shortcut, and association audit for the 4R visual probe.

This reads storage metadata and generator-private diagnostics only.  It never
constructs fair learning inputs, so internal IDs cannot accidentally become a
baseline feature through this audit.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from actmask.data.milestone4r_robust_pilot import CONDITIONS, TASKS


FORBIDDEN_CANDIDATE_FIELDS = {
    "branch", "branch_id", "camera", "camera_id", "mechanism",
    "object_id", "segmentation", "segmentation_id", "future_state",
    "candidate_outcome", "success", "appearance", "appearance_seed",
}


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def _frame_multiset(frames: np.ndarray) -> list[str]:
    """Exact frame multiset, stronger than a centroid or sum comparison."""
    return sorted(hashlib.sha256(np.ascontiguousarray(frame).tobytes()).hexdigest() for frame in frames)


def _digest(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _history_fingerprints(rgb_hist: np.ndarray, depth_hist: np.ndarray, visibility_hist: np.ndarray, rgb_masks: np.ndarray, depth_masks: np.ndarray, timestamps: np.ndarray, reference: int) -> dict[str, object]:
    """Cache exact content fingerprints once per shared history.

    Each history is referenced by ten C10 candidates.  Caching preserves the
    exact same SHA-256 checks while preventing ten duplicate scans of identical
    RGB-D arrays in the full audit.
    """
    rgb = rgb_hist[reference]
    depth = depth_hist[reference]
    visibility = visibility_hist[reference]
    return {
        "current": (_digest(rgb[-1]), _digest(depth[-1]), _digest(visibility[-1])),
        "last_two": (_digest(rgb[-2:]), _digest(depth[-2:]), _digest(visibility[-2:])),
        "unordered": (_frame_multiset(rgb), _frame_multiset(depth), _frame_multiset(visibility)),
        "whole_depth": _digest(depth),
        "temporal": (_digest(rgb_masks[reference]), _digest(depth_masks[reference]), _digest(timestamps[reference])),
    }


def _rate(checks: list[dict], key: str) -> float:
    return float(np.mean([check[key] for check in checks]))


def _label_rates(rows: list[dict], labels: np.ndarray, field: str) -> dict[str, float]:
    values: dict[str, list[bool]] = defaultdict(list)
    for row, label in zip(rows, labels):
        values[str(row[field])].append(bool(label))
    return {key: float(np.mean(value)) for key, value in sorted(values.items())}


def _camera_label_rates(rows: list[dict], labels: np.ndarray, calibration: list[dict]) -> dict[str, float]:
    camera_for_world = {item["world_id"]: item["camera"] for item in calibration}
    values: dict[str, list[bool]] = defaultdict(list)
    for row, label in zip(rows, labels):
        values[camera_for_world[row["world_id"]]].append(bool(label))
    return {key: float(np.mean(value)) for key, value in sorted(values.items())}


def run(root: str | Path) -> dict:
    root = Path(root)
    result: dict = {"schema": "milestone4r-audit-v2", "tasks": {}, "passed": True}
    for task in TASKS:
        bundle = np.load(root / f"{task}_bundles.npz")
        # NpzFile lazily decompresses each key access. Materialize each array
        # once so the full exact audit is bounded by file size, not C10 reuse.
        rgb_hist = bundle["rgb"]
        depth_hist = bundle["depth_mm"]
        visibility_hist = bundle["visibility"]
        confidence_hist = bundle["confidence"]
        rgb_masks = bundle["rgb_mask"]
        depth_masks = bundle["depth_mask"]
        timestamps = bundle["timestamps"]
        rows = _rows(root / f"{task}_candidates.jsonl")
        labels = np.load(root / f"{task}_labels.npz")["success"]
        calibration = json.loads((root / f"{task}_camera_calibration.json").read_text())
        association = json.loads((root / f"{task}_oracle_association_diagnostic.json").read_text())
        groups: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(rows):
            groups[row["pair_group"]].append(index)

        bundle_keys = set(bundle.files)
        expected_bundle_keys = {"rgb", "depth_mm", "rgb_mask", "depth_mask", "visibility", "confidence", "timestamps"}
        has_no_segmentation = not any("seg" in key.lower() or "object" in key.lower() or key.lower().endswith("_id") for key in bundle_keys)
        shape_ok = (
            expected_bundle_keys <= bundle_keys
            and rgb_hist.shape[0] == depth_hist.shape[0]
            and visibility_hist.shape == depth_hist.shape
            and confidence_hist.shape == visibility_hist.shape
            and rgb_hist.shape[1] == 6
        )
        visibility_ok = bool(
            np.array_equal(visibility_hist, confidence_hist)
            and np.isin(visibility_hist, [0, 1]).all()
        )
        forbidden_fields_absent = all(not (set(row) & FORBIDDEN_CANDIDATE_FIELDS) for row in rows)
        fingerprints = {reference: _history_fingerprints(rgb_hist, depth_hist, visibility_hist, rgb_masks, depth_masks, timestamps, reference) for reference in range(len(rgb_hist))}
        content_pair_checks: dict[tuple[int, int], dict[str, bool]] = {}
        checks = []
        malformed_groups = 0
        for pair in groups.values():
            if len(pair) != 2:
                malformed_groups += 1
                continue
            first, second = pair
            a, b = rows[first], rows[second]
            ai, bi = a["history_ref"], b["history_ref"]
            content_key = tuple(sorted((ai, bi)))
            if content_key not in content_pair_checks:
                left, right = fingerprints[ai], fingerprints[bi]
                content_pair_checks[content_key] = {
                    "current": left["current"] == right["current"],
                    "last_two": left["last_two"] == right["last_two"],
                    "unordered": left["unordered"] == right["unordered"],
                    "temporal_masks": left["temporal"] == right["temporal"],
                    "chronology_differs": left["whole_depth"] != right["whole_depth"],
                }
            content = content_pair_checks[content_key]
            checks.append({
                "current": content["current"],
                "last_two": content["last_two"],
                "unordered": content["unordered"],
                "temporal_masks": content["temporal_masks"],
                "action": np.array_equal(a["candidate_actions"], b["candidate_actions"]) and np.array_equal(a["tcp_state"], b["tcp_state"]),
                "metadata": a["condition"] == b["condition"] and a["split"] == b["split"] and a["world_id"] == b["world_id"] and a["candidate_slot"] == b["candidate_slot"],
                "chronology_differs": content["chronology_differs"],
            })

        conditions_by_history: dict[int, set[str]] = defaultdict(set)
        labels_by_history: dict[int, list[bool]] = defaultdict(list)
        for row, label in zip(rows, labels):
            conditions_by_history[row["history_ref"]].add(row["condition"])
            labels_by_history[row["history_ref"]].append(bool(label))
        condition_count = Counter(next(iter(value)) for value in conditions_by_history.values() if len(value) == 1)
        temporal_mask_checks = {
            "async_rgb": any(not np.array_equal(rgb_masks[reference], depth_masks[reference]) for reference, value in conditions_by_history.items() if value == {"async_rgb"}),
            "async_depth": any(not np.array_equal(rgb_masks[reference], depth_masks[reference]) for reference, value in conditions_by_history.items() if value == {"async_depth"}),
        }
        calibration_ok = bool(
            len(calibration) == len({row["world_id"] for row in rows})
            and all(np.asarray(item["intrinsic"]).shape == (3, 3) and np.asarray(item["extrinsic"]).shape == (3, 4) and np.isfinite(np.asarray(item["intrinsic"])).all() and np.isfinite(np.asarray(item["extrinsic"])).all() for item in calibration)
            and {item["camera"] for item in calibration} >= {"cam_train_0", "cam_train_1", "cam_train_2", "cam_train_3", "cam_val_0", "cam_test_held", "cam_test_perturbed"}
        )
        association_frames = [frame for world in association["target_occlusions"] for frame in world["frames"]]
        association_ok = bool(
            association.get("generator_private") is True
            and len(association_frames) > 0
            and all(frame["distractor_visible_after"] > 0 and frame["target_visible_after"] == 0 for frame in association_frames)
        )
        # Corner RGB statistics are a renderer-observable proxy for the shared
        # world background. They must vary across worlds but are not serialized
        # as candidate/model metadata. Paired branches are already checked for
        # an exact RGB frame multiset above.
        corner = rgb_hist[:, :, :16, :16].mean(axis=(1, 2, 3, 4))
        appearance_unique = int(len(np.unique(corner.round(3))))
        corner_by_row = corner[np.asarray([row["history_ref"] for row in rows], dtype=np.int64)]
        background_label_correlation = float(np.corrcoef(corner_by_row, labels)[0, 1]) if np.std(corner_by_row) > 0 and np.std(labels) > 0 else 0.0
        task_result = {
            "candidate_count": len(rows),
            "pairs": len(checks),
            "malformed_groups": malformed_groups,
            "schema": {"bundle_shape": shape_ok, "visibility_confidence": visibility_ok, "no_segmentation_in_bundle": has_no_segmentation, "forbidden_candidate_fields_absent": forbidden_fields_absent},
            "pair_matching": {key: _rate(checks, key) for key in checks[0]} if checks else {},
            "condition_history_counts": dict(sorted(condition_count.items())),
            "condition_coverage": sorted(condition_count) == sorted(CONDITIONS),
            "asynchronous_masks": temporal_mask_checks,
            "calibration": {"passed": calibration_ok, "camera_count": len({item["camera"] for item in calibration})},
            "association": {"passed": association_ok, "target_occlusion_frames": len(association_frames), "natural_target_occlusion_frames": int(sum(frame["natural_target_occlusion"] for frame in association_frames))},
            "appearance": {"world_randomization_observed": appearance_unique > 2, "corner_signature_count": appearance_unique, "metadata_excluded": all("appearance" not in row for row in rows), "background_label_correlation": background_label_correlation},
            "label_rate_by_condition": _label_rates(rows, labels, "condition"),
            "label_rate_by_camera_diagnostic_only": _camera_label_rates(rows, labels, calibration),
            "label_mean": float(np.mean(labels)),
            "pair_label_disagreement": float(np.mean([labels[pair[0]] != labels[pair[1]] for pair in groups.values() if len(pair) == 2])),
            "candidate_diversity": {"histories": len(labels_by_history), "c10_aligned": all(len(value) == 10 for value in labels_by_history.values()), "mixed_success_fraction": float(np.mean([any(value) and not all(value) for value in labels_by_history.values()]))},
        }
        pair_ok = bool(checks) and all(value == 1.0 for value in task_result["pair_matching"].values())
        task_result["passed"] = bool(
            malformed_groups == 0
            and pair_ok
            and shape_ok
            and visibility_ok
            and has_no_segmentation
            and forbidden_fields_absent
            and task_result["condition_coverage"]
            and all(temporal_mask_checks.values())
            and calibration_ok
            and association_ok
            and task_result["appearance"]["world_randomization_observed"]
            and task_result["appearance"]["metadata_excluded"]
            and task_result["candidate_diversity"]["c10_aligned"]
            and task_result["candidate_diversity"]["mixed_success_fraction"] > 0
        )
        result["tasks"][task] = task_result
        result["passed"] &= task_result["passed"]
    (root / "robust_audit.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


if __name__ == "__main__":
    project = Path(__file__).resolve().parents[2]
    default = project / "outputs" / "actmask" / "milestone4r_robust_visual" / "generation_condition_smoke_v3"
    print(json.dumps(run(default), indent=2, sort_keys=True))
