#!/usr/bin/env python3
"""Aggregate the two bounded fresh-vs-frozen replay comparisons."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cases = [json.loads(path.read_text()) for path in args.case]
    if any(not str(case.get("status", "")).startswith("completed") for case in cases):
        raise AssertionError("all replay cases must complete before aggregation")
    retention = [case["retention_filter"] for case in cases]
    verification_passed = all(bool(case.get("verification_passed")) for case in cases)
    result = {
        "schema_version": 1,
        "status": "completed_verified_exact" if verification_passed else "completed_with_mismatch",
        "verification_passed": verification_passed,
        "command": [case["command"] for case in cases],
        "shell_command": [case["shell_command"] for case in cases],
        "cwd": "/data/project/tzh/papers/ActMask",
        "environment": {
            "HOME": "/data/project/tzh/papers/ActMask/audit/environment/home",
            "PYTHONPATH": "/data/project/tzh/papers/ActMask",
        },
        "examples": sum(int(case["examples"]) for case in cases),
        "pairs": sum(int(case["pairs"]) for case in cases),
        "attempted": sum(int(case["attempted"]) for case in cases),
        "schema_matches": all(bool(case["schema_matches"]) for case in cases),
        "success_consistency": all(bool(case["success_consistency"]) for case in cases),
        "filter_consistency": all(bool(case["filter_consistency"]) for case in cases),
        "direct_frozen_comparison": {
            "metadata_all_exact": all(bool(case["metadata_all_exact"]) for case in cases),
            "outcomes_all_exact": all(bool(case["outcomes_all_exact"]) for case in cases),
            "all_retained_arrays_exact": all(bool(case["all_arrays_exact"]) for case in cases),
            "max_retained_array_abs_error": max(float(case["max_array_abs_error"]) for case in cases),
            "join_key": cases[0]["join_key"],
            "per_case_array_comparisons": {
                case["case_name"]: case["array_comparisons"] for case in cases
            },
        },
        "filter_summary": {
            "attempted_pairs": sum(int(item["attempted_pairs"]) for item in retention),
            "flip_pairs": sum(int(item["flip_pairs"]) for item in retention),
            "nonflip_pairs": sum(int(item["nonflip_pairs"]) for item in retention),
            "retained_pairs": sum(int(item["retained_pairs"]) for item in retention),
            "retained_exactly_equals_flip_pairs": all(
                bool(item["retained_exactly_equals_flip_pairs"]) for item in retention
            ),
            "fresh_and_frozen_flip_status_exact": all(
                bool(item["fresh_and_frozen_flip_status_exact"]) for item in retention
            ),
        },
        "cases": cases,
        "scope_limitations": [
            "Fresh worlds/candidates are reduced versus each frozen parent; these are subset re-executions, not the same parent configurations or historical simulator snapshot replays.",
            "The comparison covers two worlds/configurations, one family, two mechanisms, and four candidates (eight branch executions).",
            "Arrays can only be compared for retained flip pairs because neither frozen nor fresh model-input files contain filtered nonflip observations/actions.",
            "Nonflip validation is limited to exact candidate_outcomes booleans plus absence from retained metadata/NPZ data.",
            "No training or full dataset generation was run.",
        ],
        "mismatch_details": [],
        "blockers": [],
    }
    expected = {
        "examples": 6,
        "pairs": 3,
        "attempted": 8,
        "attempted_pairs": 4,
        "flip_pairs": 3,
        "nonflip_pairs": 1,
        "retained_pairs": 3,
    }
    observed = {
        "examples": result["examples"],
        "pairs": result["pairs"],
        "attempted": result["attempted"],
        **{key: result["filter_summary"][key] for key in (
            "attempted_pairs", "flip_pairs", "nonflip_pairs", "retained_pairs"
        )},
    }
    if observed != expected:
        raise AssertionError(f"bounded replay count mismatch: expected={expected}, observed={observed}")
    if not verification_passed:
        for case in cases:
            if not case.get("verification_passed"):
                result["mismatch_details"].append({
                    "case_name": case["case_name"],
                    "status": case["status"],
                    "metadata_all_exact": case["metadata_all_exact"],
                    "outcomes_all_exact": case["outcomes_all_exact"],
                    "schema_matches": case["schema_matches"],
                    "success_consistency": case["success_consistency"],
                    "filter_consistency": case["filter_consistency"],
                    "all_arrays_exact": case["all_arrays_exact"],
                    "max_array_abs_error": case["max_array_abs_error"],
                })
        result["blockers"].append(
            "Replay mismatch/limitation: the reduced-batch current-runtime subset preserves exact schemas, metadata, outcomes, labels, and filtering, but retained float arrays are not bit-exact; inspect mismatch_details and per_case_array_comparisons."
        )
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": result["status"],
        "examples": result["examples"],
        "pairs": result["pairs"],
        "attempted_branch_executions": result["attempted"],
        "direct_frozen_comparison": result["direct_frozen_comparison"],
        "filter_summary": result["filter_summary"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
