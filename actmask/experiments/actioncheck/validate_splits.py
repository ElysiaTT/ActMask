"""Leakage and distribution audits for ActionCheck split manifests."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from .common import read_jsonl


def _group_values(group: dict[str, Any], field: str) -> set[str]:
    if field == "context_id":
        return {str(group["context_id"])}
    if field == "episode_id":
        return {str(group["episode_id"])}
    if field == "task_id":
        return {str(group["task_id"])}
    if field == "policy_source":
        return {
            str(candidate["source_policy"]) for candidate in group["candidates"]
        }
    value = group.get("split_group", {}).get(field)
    return set() if value in (None, "") else {str(value)}


def _scheme_audit(
    groups: list[dict[str, Any]],
    scheme: dict[str, Any],
) -> dict[str, Any]:
    assignments = scheme.get("assignments", {})
    field = scheme["group_field"]
    expected = {group["context_id"] for group in groups}
    assignment_complete = set(assignments) == expected
    owner_by_value: dict[str, set[str]] = defaultdict(set)
    label_by_split: dict[str, Counter[str]] = defaultdict(Counter)
    reason_by_split: dict[str, Counter[str]] = defaultdict(Counter)
    evidence_by_split: dict[str, Counter[str]] = defaultdict(Counter)
    for group in groups:
        split = assignments.get(group["context_id"])
        if split is None:
            continue
        for value in _group_values(group, field):
            owner_by_value[value].add(split)
        for candidate in group["candidates"]:
            label_by_split[split][candidate["label"]] += 1
            evidence_by_split[split][candidate["evidence_type"]] += 1
            reason_by_split[split].update(candidate["reason_codes"])
    crossing = {
        key: sorted(values)
        for key, values in owner_by_value.items()
        if len(values) > 1
    }
    return {
        "assignment_complete": assignment_complete,
        "context_atomic": assignment_complete,
        "group_field": field,
        "group_leakage": crossing,
        "label_distribution": {
            split: dict(sorted(counts.items()))
            for split, counts in sorted(label_by_split.items())
        },
        "reason_code_distribution": {
            split: dict(sorted(counts.items()))
            for split, counts in sorted(reason_by_split.items())
        },
        "evidence_type_distribution": {
            split: dict(sorted(counts.items()))
            for split, counts in sorted(evidence_by_split.items())
        },
        "pass": assignment_complete and not crossing,
    }


def validate_splits(
    groups: Iterable[dict[str, Any]],
    split_manifest: dict[str, Any],
) -> dict[str, Any]:
    rows = list(groups)
    schemes = {}
    for name, scheme in split_manifest.get("schemes", {}).items():
        if scheme.get("status") != "available" or "assignments" not in scheme:
            schemes[name] = {
                "status": scheme.get("status", "unavailable"),
                "reason": scheme.get("reason"),
                "pass": True,
            }
        else:
            schemes[name] = {"status": "available", **_scheme_audit(rows, scheme)}
    source_folds = []
    for fold in split_manifest.get("policy_source_folds", []):
        train = set(fold["train_sources"])
        test = set(fold["test_sources"])
        source_folds.append(
            {
                "fold": fold["fold"],
                "disjoint": not (train & test),
                "train_sources": sorted(train),
                "test_sources": sorted(test),
                "pass": bool(train and test and not (train & test)),
            }
        )
    return {
        "schema": "actioncheck-split-validation-report-v1",
        "context_groups": len(rows),
        "schemes": schemes,
        "policy_source_folds": source_folds,
        "no_context_leakage": all(
            row.get("context_atomic", True) for row in schemes.values()
        ),
        "no_episode_leakage": schemes.get("episode_held_out", {}).get("pass", False),
        "task_held_out_complete": schemes.get("task_held_out", {}).get("pass", False),
        "policy_source_held_out_complete": bool(source_folds)
        and all(row["pass"] for row in source_folds),
        "pass": all(row["pass"] for row in schemes.values())
        and all(row["pass"] for row in source_folds),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("context_groups")
    parser.add_argument("split_manifest")
    args = parser.parse_args()
    result = validate_splits(
        read_jsonl(Path(args.context_groups)),
        json.loads(Path(args.split_manifest).read_text(encoding="utf-8")),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["pass"] else 1)
