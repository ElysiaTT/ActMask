#!/usr/bin/env python3
"""Compare a bounded fresh TimeArrow execution with matching frozen records."""
from __future__ import annotations

import argparse
import hashlib
import json
import shlex
from pathlib import Path

import numpy as np


INPUT_KEYS = (
    "history",
    "timestamps",
    "visibility",
    "observation_confidence",
    "candidate_actions",
    "nominal_action_timing",
    "tcp_state",
)
LABEL_KEYS = ("success", "static_features")
FAMILY_INDEX = {
    "damped_moving_capture": 0,
    "hysteretic_moving_container": 1,
    "damped_rotating_slot": 2,
}


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _row_key(row: dict) -> tuple[str, int]:
    return row["pair_id"], int(row["branch"])


def _outcome_key(row: dict) -> tuple:
    return (
        row["condition"], row["family"], row["mechanism"],
        int(row["world_id"]), int(row["candidate_id"]),
    )


def _array_diff(left: np.ndarray, right: np.ndarray) -> dict:
    same_shape = left.shape == right.shape
    same_dtype = left.dtype == right.dtype
    exact = bool(same_shape and same_dtype and np.array_equal(left, right))
    max_abs = None
    if same_shape and left.size and (
        np.issubdtype(left.dtype, np.number)
        and np.issubdtype(right.dtype, np.number)
    ):
        max_abs = float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64))))
    return {
        "shape": list(left.shape),
        "dtype_fresh": str(left.dtype),
        "dtype_frozen": str(right.dtype),
        "dtype_exact": bool(same_dtype),
        "exact": exact,
        "max_abs_error": max_abs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh", type=Path, required=True)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-name", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--mechanism", type=int, choices=(0, 1), required=True)
    parser.add_argument("--candidates", type=int, required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--require-nonflip", action="store_true")
    args = parser.parse_args()

    generation_report = json.loads((args.fresh / "generation_report.json").read_text())
    frozen_parent_report = json.loads((args.frozen / "generation_report.json").read_text())
    expected_mechanism = ("history_identifiable_damping_drive", "hysteretic_mode_memory")[args.mechanism]
    expected_report = {
        "condition": args.condition,
        "worlds": 1,
        "candidates": args.candidates,
        "mechanisms": [expected_mechanism],
        "candidate_executions": 2 * args.candidates,
    }
    report_mismatches = {
        key: {"expected": value, "observed": generation_report.get(key)}
        for key, value in expected_report.items()
        if generation_report.get(key) != value
    }
    if report_mismatches:
        raise AssertionError(f"fresh generation configuration mismatch: {report_mismatches}")
    if generation_report.get("config_sha256") != frozen_parent_report.get("config_sha256"):
        raise AssertionError({
            "fresh_config_sha256": generation_report.get("config_sha256"),
            "frozen_config_sha256": frozen_parent_report.get("config_sha256"),
        })
    if args.family not in FAMILY_INDEX:
        raise AssertionError(f"unknown family argument: {args.family}")

    fresh_rows = _jsonl(args.fresh / "metadata.jsonl")
    frozen_rows = _jsonl(args.frozen / "metadata.jsonl")
    if len({_row_key(row) for row in fresh_rows}) != len(fresh_rows):
        raise AssertionError("fresh metadata contains duplicate (pair_id, branch) keys")
    if len({_row_key(row) for row in frozen_rows}) != len(frozen_rows):
        raise AssertionError("frozen metadata contains duplicate (pair_id, branch) keys")
    fresh_index = {_row_key(row): i for i, row in enumerate(fresh_rows)}
    frozen_all = {_row_key(row): (i, row) for i, row in enumerate(frozen_rows)}
    missing_frozen_rows = [list(key) for key in fresh_index if key not in frozen_all]
    if missing_frozen_rows:
        raise AssertionError(f"fresh rows absent from frozen data: {missing_frozen_rows}")

    ordered_keys = sorted(fresh_index)
    fresh_order = [fresh_index[key] for key in ordered_keys]
    frozen_order = [frozen_all[key][0] for key in ordered_keys]
    metadata_exact = {
        f"{pair_id}|branch={branch}": fresh_rows[fresh_index[(pair_id, branch)]]
        == frozen_all[(pair_id, branch)][1]
        for pair_id, branch in ordered_keys
    }

    fresh_outcome_rows = _jsonl(args.fresh / "candidate_outcomes.jsonl")
    frozen_outcome_rows = _jsonl(args.frozen / "candidate_outcomes.jsonl")
    if len({_outcome_key(row) for row in fresh_outcome_rows}) != len(fresh_outcome_rows):
        raise AssertionError("fresh outcomes contain duplicate join keys")
    if len({_outcome_key(row) for row in frozen_outcome_rows}) != len(frozen_outcome_rows):
        raise AssertionError("frozen outcomes contain duplicate join keys")
    fresh_outcomes = {_outcome_key(row): row for row in fresh_outcome_rows}
    frozen_outcomes_all = {_outcome_key(row): row for row in frozen_outcome_rows}
    expected_outcome_keys = {
        (args.condition, args.family, expected_mechanism, 0, candidate)
        for candidate in range(args.candidates)
    }
    if set(fresh_outcomes) != expected_outcome_keys:
        raise AssertionError({
            "expected_fresh_outcome_keys": [list(key) for key in sorted(expected_outcome_keys)],
            "observed_fresh_outcome_keys": [list(key) for key in sorted(fresh_outcomes)],
        })
    missing_frozen_outcomes = [list(key) for key in fresh_outcomes if key not in frozen_outcomes_all]
    if missing_frozen_outcomes:
        raise AssertionError(f"fresh outcomes absent from frozen data: {missing_frozen_outcomes}")
    expected_metadata_seed = args.seed + FAMILY_INDEX[args.family] * 10000 + args.mechanism * 1000
    metadata_argument_consistency = all(
        row["condition"] == args.condition
        and row["family"] == args.family
        and row["mechanism"] == expected_mechanism
        and int(row["world_id"]) == 0
        and 0 <= int(row["candidate_id"]) < args.candidates
        and int(row["seed"]) == expected_metadata_seed
        for row in fresh_rows
    )
    if not metadata_argument_consistency:
        raise AssertionError("fresh metadata does not match verifier/generator arguments")
    outcome_exact = {
        "|".join(map(str, key)): row == frozen_outcomes_all[key]
        for key, row in sorted(fresh_outcomes.items())
    }
    fresh_outcome_metadata_exact = all(
        bool(row["success"])
        == bool(fresh_outcomes[(
            row["condition"], row["family"], row["mechanism"],
            int(row["world_id"]), int(row["candidate_id"]),
        )][f"branch{int(row['branch'])}_success"])
        for row in fresh_rows
    )

    fresh_retained = {
        (row["condition"], row["family"], row["mechanism"], int(row["world_id"]), int(row["candidate_id"]))
        for row in fresh_rows
    }
    expected_retained = {key for key, row in fresh_outcomes.items() if bool(row["pair_flip"])}
    outcome_flip_invariant = all(
        bool(row["pair_flip"]) == (bool(row["branch0_success"]) != bool(row["branch1_success"]))
        for row in fresh_outcomes.values()
    )
    branches_by_pair: dict[str, set[int]] = {}
    for row in fresh_rows:
        branches_by_pair.setdefault(row["pair_id"], set()).add(int(row["branch"]))
    retained_pair_branches_exact = all(branches == {0, 1} for branches in branches_by_pair.values())
    retention = {
        "attempted_pairs": len(fresh_outcomes),
        "flip_pairs": len(expected_retained),
        "nonflip_pairs": len(fresh_outcomes) - len(expected_retained),
        "retained_pairs": len(fresh_retained),
        "retained_exactly_equals_flip_pairs": fresh_retained == expected_retained,
        "fresh_and_frozen_flip_status_exact": all(outcome_exact.values()),
        "outcome_flip_invariant": outcome_flip_invariant,
        "retained_pair_branches_exact": retained_pair_branches_exact,
        "required_nonflip_present": (not args.require_nonflip) or (len(fresh_outcomes) - len(expected_retained) >= 1),
    }

    arrays: dict[str, dict] = {}
    schema_detail: dict[str, dict] = {}
    with np.load(args.fresh / "model_inputs.npz") as fresh_npz, np.load(args.frozen / "model_inputs.npz") as frozen_npz:
        schema_detail["model_inputs.npz"] = {
            "fresh_keys": sorted(fresh_npz.files),
            "frozen_keys": sorted(frozen_npz.files),
            "expected_keys": sorted(INPUT_KEYS),
            "key_sets_exact": set(fresh_npz.files) == set(frozen_npz.files) == set(INPUT_KEYS),
            "fields": {},
        }
        for key in INPUT_KEYS:
            arrays[f"model_inputs.{key}"] = _array_diff(
                fresh_npz[key][fresh_order], frozen_npz[key][frozen_order]
            )
            schema_detail["model_inputs.npz"]["fields"][key] = {
                "fresh_dtype": str(fresh_npz[key].dtype),
                "frozen_dtype": str(frozen_npz[key].dtype),
                "dtype_exact": fresh_npz[key].dtype == frozen_npz[key].dtype,
                "fresh_shape": list(fresh_npz[key].shape),
                "frozen_shape": list(frozen_npz[key].shape),
                "non_batch_shape_exact": fresh_npz[key].shape[1:] == frozen_npz[key].shape[1:],
                "fresh_batch_matches_metadata": fresh_npz[key].shape[0] == len(fresh_rows),
                "frozen_batch_matches_metadata": frozen_npz[key].shape[0] == len(frozen_rows),
            }
    label_metadata_success_exact = False
    with np.load(args.fresh / "labels.npz") as fresh_npz, np.load(args.frozen / "labels.npz") as frozen_npz:
        schema_detail["labels.npz"] = {
            "fresh_keys": sorted(fresh_npz.files),
            "frozen_keys": sorted(frozen_npz.files),
            "expected_keys": sorted(LABEL_KEYS),
            "key_sets_exact": set(fresh_npz.files) == set(frozen_npz.files) == set(LABEL_KEYS),
            "fields": {},
        }
        for key in LABEL_KEYS:
            arrays[f"labels.{key}"] = _array_diff(
                fresh_npz[key][fresh_order], frozen_npz[key][frozen_order]
            )
            schema_detail["labels.npz"]["fields"][key] = {
                "fresh_dtype": str(fresh_npz[key].dtype),
                "frozen_dtype": str(frozen_npz[key].dtype),
                "dtype_exact": fresh_npz[key].dtype == frozen_npz[key].dtype,
                "fresh_shape": list(fresh_npz[key].shape),
                "frozen_shape": list(frozen_npz[key].shape),
                "non_batch_shape_exact": fresh_npz[key].shape[1:] == frozen_npz[key].shape[1:],
                "fresh_batch_matches_metadata": fresh_npz[key].shape[0] == len(fresh_rows),
                "frozen_batch_matches_metadata": frozen_npz[key].shape[0] == len(frozen_rows),
            }
        label_metadata_success_exact = bool(np.array_equal(
            fresh_npz["success"],
            np.asarray([row["success"] for row in fresh_rows], dtype=fresh_npz["success"].dtype),
        ))

    required = ("model_inputs.npz", "labels.npz", "metadata.jsonl", "candidate_outcomes.jsonl", "generation_report.json")
    files = {
        scope: {
            name: {"sha256": _sha256(root / name), "bytes": (root / name).stat().st_size}
            for name in required
        }
        for scope, root in (("fresh", args.fresh), ("frozen", args.frozen))
    }
    schema_matches = bool(all(
        detail["key_sets_exact"]
        and all(
            field["dtype_exact"] and field["non_batch_shape_exact"]
            and field["fresh_batch_matches_metadata"]
            and field["frozen_batch_matches_metadata"]
            for field in detail["fields"].values()
        )
        for detail in schema_detail.values()
    ))
    success_consistency = bool(
        fresh_outcome_metadata_exact and label_metadata_success_exact
        and all(outcome_exact.values())
        and arrays["labels.success"]["exact"]
    )
    filter_consistency = bool(
        retention["retained_exactly_equals_flip_pairs"]
        and retention["fresh_and_frozen_flip_status_exact"]
        and retention["outcome_flip_invariant"]
        and retention["retained_pair_branches_exact"]
        and retention["required_nonflip_present"]
    )
    all_arrays_exact = all(item["exact"] for item in arrays.values())
    comparable_errors = {
        key: item["max_abs_error"]
        for key, item in arrays.items()
        if item["max_abs_error"] is not None
    }
    not_numerically_comparable = [
        key for key, item in arrays.items() if item["max_abs_error"] is None
    ]
    max_array_abs_error = max(comparable_errors.values()) if comparable_errors else None
    verification_passed = bool(
        schema_matches and success_consistency and filter_consistency
        and all(metadata_exact.values()) and all(outcome_exact.values())
        and metadata_argument_consistency
        and all_arrays_exact and max_array_abs_error == 0.0
    )
    generate_argv = [
        "/data/envs/actmask-audit/bin/python", "-c",
        "from actmask.experiments.milestone3r_nl_v2 import generate; "
        f"generate('{args.fresh}', worlds=1, seed={args.seed}, "
        f"condition='{args.condition}', mechanisms=({args.mechanism},), candidates={args.candidates}, "
        f"families=('{args.family}',))",
    ]
    generation_count_binding = {
        "candidate_executions_matches_outcome_branches": generation_report.get("candidate_executions") == 2 * len(fresh_outcomes),
        "accepted_pairs_matches_retained_pairs": generation_report.get("accepted_pairs") == len(fresh_retained),
        "examples_matches_metadata_rows": generation_report.get("examples") == len(fresh_rows),
        "audit_pairs_matches_retained_pairs": generation_report.get("audit", {}).get("pairs") == len(fresh_retained),
    }
    generation_counts_exact = all(generation_count_binding.values())
    verification_passed = verification_passed and generation_counts_exact
    shell_command = (
        "cd /data/project/tzh/papers/ActMask && "
        "HOME=/data/project/tzh/papers/ActMask/audit/environment/home "
        "PYTHONPATH=/data/project/tzh/papers/ActMask "
        + shlex.join(generate_argv)
    )
    result = {
        "schema_version": 1,
        "status": "completed_verified_exact" if verification_passed else "completed_with_mismatch",
        "verification_passed": verification_passed,
        "scope": "bounded_frozen_subset_reexecution_under_reduced_batch_not_same_parent_configuration_or_snapshot_replay",
        "case_name": args.case_name,
        "command": generate_argv,
        "shell_command": shell_command,
        "cwd": "/data/project/tzh/papers/ActMask",
        "environment": {
            "HOME": "/data/project/tzh/papers/ActMask/audit/environment/home",
            "PYTHONPATH": "/data/project/tzh/papers/ActMask",
        },
        "join_key": "metadata: (pair_id, branch); outcomes: (condition, family, mechanism, world_id, candidate_id)",
        "configuration": {
            "worlds": 1,
            "generate_seed": args.seed,
            "condition": args.condition,
            "mechanisms": [args.mechanism],
            "candidates": args.candidates,
            "families": [args.family],
            "expected_branch_executions": 2 * args.candidates,
        },
        "fresh_generation_report": generation_report,
        "frozen_parent_generation_report": frozen_parent_report,
        "argument_data_binding": {
            "expected_outcome_keys_exact": set(fresh_outcomes) == expected_outcome_keys,
            "metadata_argument_consistency": metadata_argument_consistency,
            "expected_metadata_seed": expected_metadata_seed,
            "config_sha256_matches_frozen_parent": generation_report.get("config_sha256") == frozen_parent_report.get("config_sha256"),
            "generation_count_binding": generation_count_binding,
            "generation_counts_exact": generation_counts_exact,
        },
        "examples": len(ordered_keys),
        "pairs": len(fresh_retained),
        "attempted": sum(2 for _ in fresh_outcomes),
        "matched_rows": len(ordered_keys),
        "matched_outcomes": len(fresh_outcomes),
        "metadata_exact_by_row": metadata_exact,
        "metadata_all_exact": all(metadata_exact.values()),
        "outcomes_exact_by_pair": outcome_exact,
        "outcomes_all_exact": all(outcome_exact.values()),
        "retention_filter": retention,
        "schema_matches": schema_matches,
        "schema_detail": schema_detail,
        "success_consistency": success_consistency,
        "filter_consistency": filter_consistency,
        "array_comparisons": arrays,
        "all_arrays_exact": all_arrays_exact,
        "max_array_abs_error": max_array_abs_error,
        "numeric_array_max_abs_error_by_field": comparable_errors,
        "not_numerically_comparable_fields": not_numerically_comparable,
        "files": files,
        "limitations": [
            "Fresh worlds/candidates are reduced versus the frozen parent generation; this is a seed/world/candidate subset re-execution, not the same parent batch configuration.",
            "No historical simulator snapshot or per-step trace exists.",
            "A match supports repeatability under the reconstructed environment, not independent proof of how every frozen result was produced.",
            "Only one world, one family, one mechanism, and two candidates were re-executed; no full dataset or training run was attempted.",
            "Filtered nonflip observations/actions are absent from both fresh and frozen NPZ files, so nonflip comparison is limited to exact outcome booleans and absence from retained metadata/arrays.",
        ],
    }
    result["scope_limitations"] = list(result["limitations"])
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": result["status"],
        "matched_rows": result["matched_rows"],
        "matched_outcomes": result["matched_outcomes"],
        "metadata_all_exact": result["metadata_all_exact"],
        "outcomes_all_exact": result["outcomes_all_exact"],
        "all_arrays_exact": result["all_arrays_exact"],
        "max_array_abs_error": result["max_array_abs_error"],
        "retention_filter": result["retention_filter"],
    }, indent=2))
    if not verification_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
