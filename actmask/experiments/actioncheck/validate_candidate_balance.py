"""Report ActionCheck proposal, label, reason, task and group balance."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .common import read_jsonl


def candidate_balance(groups: Iterable[dict[str, Any]]) -> dict[str, Any]:
    source: Counter[str] = Counter()
    label: Counter[str] = Counter()
    reason: Counter[str] = Counter()
    task: Counter[str] = Counter()
    evidence: Counter[str] = Counter()
    count: Counter[int] = Counter()
    context_groups = 0
    candidates = 0
    for group in groups:
        context_groups += 1
        count[len(group["candidates"])] += 1
        task[group["task_id"]] += 1
        for candidate in group["candidates"]:
            candidates += 1
            source[candidate["source_policy"]] += 1
            label[candidate["label"]] += 1
            evidence[candidate["evidence_type"]] += 1
            reason.update(candidate["reason_codes"])
    rejects = label["reject"]
    max_reason_fraction = (
        max(reason.values(), default=0) / rejects if rejects else None
    )
    return {
        "schema": "actioncheck-candidate-balance-report-v1",
        "context_groups": context_groups,
        "candidates": candidates,
        "candidate_source_distribution": dict(sorted(source.items())),
        "label_distribution": dict(sorted(label.items())),
        "reason_code_distribution": dict(sorted(reason.items())),
        "task_context_distribution": dict(sorted(task.items())),
        "evidence_type_distribution": dict(sorted(evidence.items())),
        "per_context_candidate_count": {
            str(key): value for key, value in sorted(count.items())
        },
        "maximum_reason_code_fraction_of_rejects": max_reason_fraction,
        "single_reason_code_le_0_40": bool(
            max_reason_fraction is not None and max_reason_fraction <= 0.40
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    args = parser.parse_args()
    print(
        json.dumps(
            candidate_balance(read_jsonl(Path(args.path))),
            indent=2,
            sort_keys=True,
        )
    )
