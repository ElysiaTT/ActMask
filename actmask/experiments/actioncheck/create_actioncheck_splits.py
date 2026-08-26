"""Generate leakage-safe ActionCheck context-group split schemes."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

from .common import read_jsonl, stable_bucket, write_json
from .validate_splits import validate_splits


def _partition(values: set[str]) -> dict[str, str]:
    ordered = sorted(values, key=lambda value: (stable_bucket(value, 2**31 - 1), value))
    count = len(ordered)
    if count < 3:
        raise ValueError("at least three groups are required for train/validation/test")
    validation_count = max(1, round(count * 0.15))
    test_count = max(1, round(count * 0.15))
    if validation_count + test_count >= count:
        validation_count = test_count = 1
    train_count = count - validation_count - test_count
    return {
        value: (
            "train"
            if index < train_count
            else "validation"
            if index < train_count + validation_count
            else "test"
        )
        for index, value in enumerate(ordered)
    }


def _make_scheme(
    groups: list[dict[str, Any]],
    name: str,
    field: str,
    getter: Callable[[dict[str, Any]], str | None],
) -> dict[str, Any]:
    values = {value for group in groups if (value := getter(group))}
    covered = [group for group in groups if getter(group)]
    if len(covered) != len(groups):
        return {
            "status": "unavailable",
            "reason": f"{field} missing for {len(groups) - len(covered)} contexts",
            "covered_contexts": len(covered),
            "total_contexts": len(groups),
            "group_field": field,
        }
    if len(values) < 3:
        return {
            "status": "unavailable",
            "reason": f"only {len(values)} unique {field} groups",
            "covered_contexts": len(covered),
            "total_contexts": len(groups),
            "group_field": field,
        }
    owner = _partition(values)
    return {
        "status": "available",
        "group_field": field,
        "unique_groups": len(values),
        "assignments": {
            group["context_id"]: owner[str(getter(group))] for group in groups
        },
    }


def create_splits(groups: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(groups)
    schemes = {
        "episode_held_out": _make_scheme(
            rows, "episode_held_out", "episode_id", lambda group: group["episode_id"]
        ),
        "task_held_out": _make_scheme(
            rows, "task_held_out", "task_id", lambda group: group["task_id"]
        ),
        "object_composition_held_out": _make_scheme(
            rows,
            "object_composition_held_out",
            "object_composition_id",
            lambda group: group["split_group"].get("object_composition_id"),
        ),
        "scene_environment_held_out": _make_scheme(
            rows,
            "scene_environment_held_out",
            "scene_id",
            lambda group: group["split_group"].get("scene_id"),
        ),
    }
    sources = sorted(
        {
            candidate["source_policy"]
            for group in rows
            for candidate in group["candidates"]
        }
    )
    policy_source_folds = [
        {
            "fold": index,
            "test_sources": [source],
            "train_sources": [other for other in sources if other != source],
            "validation_sources": [],
            "note": "source-held-out fold; validation must be selected within train sources by episode",
        }
        for index, source in enumerate(sources)
        if len(sources) >= 2
    ]
    manifest = {
        "schema": "actioncheck-split-manifest-v1",
        "primary_scheme": "episode_held_out",
        "schemes": schemes,
        "policy_source_folds": policy_source_folds,
        "candidate_level_random_splitting": False,
        "context_group_atomic": True,
    }
    manifest["audit"] = validate_splits(rows, manifest)
    return manifest


def split_audit_schema() -> dict[str, Any]:
    return {
        "schema": "actioncheck-split-audit-schema-v1",
        "required_checks": [
            "no context leakage",
            "no episode leakage",
            "no task leakage for task-held-out",
            "no candidate-source leakage for source-held-out",
            "label balance by split",
            "reason-code balance by split",
            "evidence-type balance by split",
        ],
        "atomic_unit": "context_id",
        "candidate_random_split_forbidden": True,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("context_groups")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    groups = read_jsonl(Path(args.context_groups))
    result = create_splits(groups)
    write_json(args.output, result)
    print(json.dumps(result["audit"], indent=2, sort_keys=True))
    raise SystemExit(0 if result["audit"]["pass"] else 1)
