"""Regression checks for final-v2 temporal-envelope validation."""

from __future__ import annotations

import pytest

from actmask.experiments.milestone2e_final_v2_temporal import (
    TEMPORAL_SHORTCUT_VIEWS,
    _validate_established_report,
)


def test_temporal_envelope_requires_complete_all_validation_promising_report() -> None:
    report = {
        "status": "completed_frozen_c20",
        "all_validation_promising": True,
        "phase4_temporal_shortcut": {
            "view_order": list(TEMPORAL_SHORTCUT_VIEWS),
            "per_variant": {"selected": {}},
        },
    }
    _validate_established_report(report)


def test_temporal_envelope_rejects_partial_or_nonpromising_report() -> None:
    with pytest.raises(ValueError, match="all-validation-promising"):
        _validate_established_report({"status": "completed_frozen_c20", "all_validation_promising": False})
