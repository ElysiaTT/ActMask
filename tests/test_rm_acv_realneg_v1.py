"""Terminal contracts for RM-ACV-REALNEG-V1."""
from __future__ import annotations

import json
from pathlib import Path

from actmask.experiments.rm_acv_realneg_v1_verify import (
    EXPECTED_DECISION,
    OUT,
    PAPER_FILES,
    verify,
)


def load(relative: str) -> dict:
    return json.loads((OUT / relative).read_text(encoding="utf-8"))


def test_independent_terminal_verifier_passes() -> None:
    result = verify(rehash_source=False)
    assert result["passed"], result
    assert result["decision"] == EXPECTED_DECISION
    assert all(result["checks"].values())


def test_pair_scale_balance_and_coverage_decision_are_exact() -> None:
    result = verify(rehash_source=False)
    assert result["pair_details"] == {
        "anchors": 992,
        "samples": 1476,
        "N7_pairs": 289,
        "N8_pairs": 80,
        "N7_unique_anchors": 578,
        "N8_unique_episode_pairs": 80,
    }
    family = load("real_family_decision.json")["families"]
    assert all(row["exact_action_balance"] for row in family.values())
    assert all(row["exact_context_balance"] for row in family.values())
    assert all(row["leakage_pass"] for row in family.values())
    assert all(not row["coverage"]["pass"] for row in family.values())


def test_shortcut_predictions_and_behavior_are_independently_recomputed() -> None:
    result = verify(rehash_source=False)
    assert result["checks"]["raw_shortcut_prediction_recomputation"]
    assert result["prediction_details"]["raw_predictions"] == 7252
    assert result["checks"]["behavior_and_terminal_decision_recomputation"]
    behavior = load("behavioral_validity_report.json")
    assert not behavior["pass"]
    assert all(not row["pass"] for row in behavior["families"].values())


def test_conditional_training_stopped_at_r6() -> None:
    decision = load("phase_r6_decision.json")
    assert decision["decision"] == EXPECTED_DECISION
    assert decision["stage_reached"] == "R6" and decision["terminal"]
    assert not decision["combined_benchmark_construction_authorized"]
    assert not decision["learned_fair_baselines_trained"]
    assert not decision["proposed_method_authorized"]
    assert not decision["valid_new_families"]


def test_paper_package_and_claim_boundary_are_complete() -> None:
    assert all((OUT / "paper" / name).is_file() for name in PAPER_FILES)
    assert len(list((OUT / "paper").glob("*.md"))) == 14
    boundary = (OUT / "claim_boundary.md").read_text(encoding="utf-8").lower()
    method = (OUT / "paper" / "method.md").read_text(encoding="utf-8").lower()
    assert "not supported" in boundary
    assert "not authorized" in method


def test_resource_and_reproducibility_budgets_hold() -> None:
    result = verify(rehash_source=False)
    assert result["checks"]["generated_artifact_and_generator_hashes"]
    assert result["checks"]["historical_N1_N4_branch_unchanged"]
    assert result["checks"]["resource_budget"]
    assert result["checks"]["unauthorized_outputs_absent"]
