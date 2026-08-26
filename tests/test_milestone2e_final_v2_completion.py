"""Pure-structure regression tests for corrected final-v2 A6 reporting."""

from __future__ import annotations

import pytest

from actmask.experiments.milestone2e_final_v2_completion import (
    POINT_PERMUTATION_VIEW,
    _point_permutation_audit,
)


def _condition(*, paired: bool = False) -> dict[str, object]:
    result: dict[str, object] = {"condition_id": "original", "per_seed": []}
    if paired:
        result["paired_view_minus_original_group_mean"] = {"metrics": {"top1_success": {"mean": 0.0}}}
    return result


def test_point_permutation_audit_requires_all_corrected_candidate_counts() -> None:
    artifact = {
        "candidate_counts": {
            f"C{count}": {
                "views": {
                    "original": _condition(),
                    POINT_PERMUTATION_VIEW: _condition(paired=True),
                }
            }
            for count in (5, 10, 20, 50)
        }
    }
    audit = _point_permutation_audit(artifact)
    assert tuple(audit["candidate_counts"]) == ("C5", "C10", "C20", "C50")
    assert audit["view"] == POINT_PERMUTATION_VIEW


def test_point_permutation_audit_rejects_missing_paired_corrected_evidence() -> None:
    artifact = {
        "candidate_counts": {
            f"C{count}": {
                "views": {
                    "original": _condition(),
                    POINT_PERMUTATION_VIEW: _condition(paired=count != 20),
                }
            }
            for count in (5, 10, 20, 50)
        }
    }
    with pytest.raises(ValueError, match="paired"):
        _point_permutation_audit(artifact)
