"""Exact ActionCheck classification, ranking and calibration metrics."""
from __future__ import annotations

import numpy as np

from actmask.experiments.actioncheck.evaluate_actioncheck import (
    auprc,
    auroc,
    best_balanced_threshold,
    calibration_metrics,
    classification_metrics,
    evaluate_prediction_rows,
    grouped_bootstrap_ci,
    grouped_ranking_metrics,
    risk_coverage_metrics,
)


def test_perfect_binary_metrics() -> None:
    labels = [0, 0, 1, 1]
    scores = [0.1, 0.2, 0.8, 0.9]
    assert auroc(labels, scores) == 1.0
    assert auprc(labels, scores) == 1.0
    threshold = best_balanced_threshold(labels, scores)
    result = classification_metrics(labels, scores, threshold)
    assert result["balanced_accuracy"] == 1.0
    assert result["false_accept_rate"] == 0.0
    assert result["false_reject_rate"] == 0.0


def test_grouped_ranking_is_context_atomic() -> None:
    labels = [1, 0, 0, 1, 0, 0]
    scores = [0.9, 0.2, 0.1, 0.8, 0.7, 0.3]
    contexts = ["a", "a", "a", "b", "b", "b"]
    result = grouped_ranking_metrics(labels, scores, contexts)
    assert result["eligible_context_groups"] == 2
    assert result["recall_at_1"] == 1.0
    assert result["mrr"] == 1.0
    assert result["best_accept_rank"] == 1.0
    assert np.isclose(result["chance_recall_at_1"], 1 / 3)


def test_calibration_risk_coverage_and_grouped_bootstrap() -> None:
    labels = [0, 0, 1, 1]
    scores = [0.1, 0.3, 0.7, 0.9]
    calibration = calibration_metrics(labels, scores)
    assert np.isclose(calibration["brier_score"], 0.05)
    risk = risk_coverage_metrics(labels, scores, threshold=0.5)
    assert [row["requested_coverage"] for row in risk] == [1.0, 0.9, 0.8, 0.7]
    rows = [
        {"context_id": "a", "value": 1.0},
        {"context_id": "a", "value": 1.0},
        {"context_id": "b", "value": 0.0},
        {"context_id": "b", "value": 0.0},
    ]
    interval = grouped_bootstrap_ci(
        rows,
        lambda sample: float(np.mean([row["value"] for row in sample])),
        draws=100,
        seed=3,
    )
    assert 0.0 <= interval[0] <= interval[1] <= 1.0


def test_evaluation_reports_task_source_reason_and_evidence_strata() -> None:
    rows = []
    for split, offset in (("validation", 0), ("test", 4)):
        for index, (label, score) in enumerate(
            ((0, 0.1), (1, 0.9), (0, 0.2), (1, 0.8))
        ):
            rows.append(
                {
                    "split": split,
                    "label": label,
                    "score": score,
                    "context_id": f"c{offset + index // 2}",
                    "task_id": "task-a" if index < 2 else "task-b",
                    "source_policy": "policy-a" if index % 2 else "policy-b",
                    "reason_codes_audit_only": (
                        ["simulator_failure"] if label == 0 else []
                    ),
                    "evidence_type_audit_only": (
                        "sim_execution_success"
                        if label == 1
                        else "sim_execution_failure"
                    ),
                }
            )
    result = evaluate_prediction_rows(rows)
    strata = result["generalization_strata"]
    assert set(strata) == {
        "per_task",
        "per_policy_source",
        "per_reason_code_audit_only",
        "per_evidence_type_audit_only",
    }
    assert strata["per_reason_code_audit_only"]["simulator_failure"][
        "false_accept_rate"
    ] == 0.0
    assert result["reason_code_and_evidence_type_used_as_model_inputs"] is False
