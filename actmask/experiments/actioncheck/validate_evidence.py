"""Validate that ActionCheck labels are backed by permitted evidence."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .common import EVIDENCE_TYPES, FORBIDDEN_LABEL_BASES, read_jsonl


def _candidate_errors(candidate: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    label = candidate["label"]
    evidence = candidate["evidence_type"]
    basis = str(candidate.get("sampling_metadata", {}).get("label_basis", ""))
    if label in {"accept", "reject"}:
        if evidence not in EVIDENCE_TYPES or evidence == "ambiguous":
            errors.append("primary_label_without_allowed_evidence_type")
        if not candidate.get("label_source"):
            errors.append("primary_label_without_label_source")
        if basis in FORBIDDEN_LABEL_BASES:
            errors.append(f"forbidden_label_basis:{basis}")
    if label == "uncertain" and evidence != "ambiguous":
        errors.append("uncertain_without_ambiguous_evidence")
    if label == "reject" and not candidate["reason_codes"]:
        errors.append("reject_without_reason_code")
    if evidence.startswith("sim_execution_"):
        result = candidate.get("sim_result")
        if not isinstance(result, dict) or not isinstance(result.get("success"), bool):
            errors.append("sim_evidence_without_boolean_result")
        elif (label == "accept") != bool(result["success"]):
            errors.append("sim_result_label_mismatch")
    if evidence.startswith("real_execution_"):
        result = candidate.get("execution_result")
        if not isinstance(result, dict) or not isinstance(result.get("success"), bool):
            errors.append("real_execution_without_boolean_result")
    if evidence == "expert_review" and not isinstance(
        candidate.get("expert_review"), dict
    ):
        errors.append("expert_review_missing")
    if evidence == "rule_checker" and not isinstance(candidate.get("rule_check"), dict):
        errors.append("rule_check_missing")
    return errors


def validate_evidence(groups: Iterable[dict[str, Any]]) -> dict[str, Any]:
    failures = []
    counts: Counter[str] = Counter()
    labels: Counter[str] = Counter()
    candidates = 0
    for group in groups:
        for candidate in group["candidates"]:
            candidates += 1
            counts[candidate["evidence_type"]] += 1
            labels[candidate["label"]] += 1
            errors = _candidate_errors(candidate)
            if errors:
                failures.append(
                    {
                        "context_id": group["context_id"],
                        "candidate_id": candidate["candidate_id"],
                        "errors": errors,
                    }
                )
    return {
        "schema": "actioncheck-evidence-validation-report-v1",
        "candidates": candidates,
        "evidence_type_distribution": dict(sorted(counts.items())),
        "label_distribution": dict(sorted(labels.items())),
        "failures": failures,
        "pass": not failures,
        "forbidden_label_bases": list(FORBIDDEN_LABEL_BASES),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    args = parser.parse_args()
    result = validate_evidence(read_jsonl(Path(args.path)))
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["pass"] else 1)
