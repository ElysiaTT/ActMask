"""Audit proposal multiplicity and label support per context."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .common import read_jsonl


def validate_groups(
    groups: Iterable[dict[str, Any]],
    *,
    minimum_candidates: int = 4,
) -> dict[str, Any]:
    failures = []
    count_histogram: Counter[int] = Counter()
    mixed = 0
    total = 0
    for group in groups:
        total += 1
        labels = Counter(
            candidate["label"]
            for candidate in group["candidates"]
            if candidate["label"] in {"accept", "reject"}
        )
        count = len(group["candidates"])
        count_histogram[count] += 1
        reasons = []
        if count < minimum_candidates:
            reasons.append("candidate_count_below_K")
        if labels["accept"] < 1:
            reasons.append("no_accept_candidate")
        if labels["reject"] < 1:
            reasons.append("no_reject_candidate")
        if not reasons:
            mixed += 1
        else:
            failures.append(
                {
                    "context_id": group["context_id"],
                    "candidate_count": count,
                    "accept": labels["accept"],
                    "reject": labels["reject"],
                    "reasons": reasons,
                }
            )
    return {
        "schema": "actioncheck-group-structure-report-v1",
        "minimum_candidates": minimum_candidates,
        "context_groups": total,
        "fully_supported_contexts": mixed,
        "fully_supported_fraction": mixed / total if total else None,
        "candidate_count_histogram": {
            str(key): value for key, value in sorted(count_histogram.items())
        },
        "failures": failures,
        "all_contexts_pass": not failures,
        "pilot_80_percent_gate": bool(total and mixed / total >= 0.80),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--minimum-candidates", type=int, default=4)
    args = parser.parse_args()
    result = validate_groups(
        read_jsonl(Path(args.path)),
        minimum_candidates=args.minimum_candidates,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["pilot_80_percent_gate"] else 1)
