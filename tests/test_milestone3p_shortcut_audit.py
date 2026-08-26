from __future__ import annotations

import numpy as np

from actmask.data.milestone3p_counterfactual import CounterfactualTemporalDataset
from actmask.eval.statistics import paired_bootstrap_ci
from actmask.experiments.milestone3p_shortcut_audit import (
    _counterfactual_pair_accuracy,
    _feature_identity_audit,
    _materialize,
    _metric_report,
    _prior_audit,
    _score_for_observable_baseline,
    _temporal_corruption_report,
    SCHEMA_VERSION,
)


_COUNTS = {"train": 4, "val": 2, "test": 2}


def _matched_rows():
    dataset = CounterfactualTemporalDataset(
        split="test", regime="matched_counterfactual", split_counts=_COUNTS, include_hidden_state=False
    )
    return dataset, _materialize(dataset)


def test_milestone3p_matching_audit_proves_static_chance_on_flip_pairs() -> None:
    dataset, _ = _matched_rows()
    report = _feature_identity_audit(dataset)
    assert report["passed"]
    assert report["static_leakage_bayes_accuracy_on_label_flip_pairs"] == 0.5


def test_milestone3p_time_aligned_baseline_beats_static_on_matched_pairs() -> None:
    _, rows = _matched_rows()
    static = _score_for_observable_baseline(rows, "CurrentPositionProximity")
    temporal = _score_for_observable_baseline(rows, "TimeAlignedRelativeFeatures")
    static_pair = _counterfactual_pair_accuracy(rows, static)
    temporal_pair = _counterfactual_pair_accuracy(rows, temporal)
    assert static_pair["pair_accuracy"] == 0.5
    assert temporal_pair["pair_accuracy"] >= 0.9


def test_milestone3p_positive_dynamic_control_requires_history() -> None:
    _, rows = _matched_rows()
    static = _score_for_observable_baseline(rows, "CurrentPositionProximity")
    temporal = _score_for_observable_baseline(rows, "TimeAlignedRelativeFeatures")
    static_top1 = _metric_report(rows, static, score_name="positive_control_static")["top1_success"]
    temporal_top1 = _metric_report(rows, temporal, score_name="positive_control_temporal")["top1_success"]
    assert temporal_top1 - static_top1 >= 0.2


def test_milestone3p_temporal_corruptions_preserve_static_rows_and_reduce_temporal_score() -> None:
    _, rows = _matched_rows()
    report = _temporal_corruption_report(rows)
    assert set(report["temporal_causal_drop"]) == {
        "reversed", "frame_permutation", "mismatched", "zeroed_motion", "shuffled_motion", "removed"
    }
    assert report["correct"]["metrics"]["top1_success"] >= 0.9
    assert any(value["top1_drop"] >= 0.1 for value in report["temporal_causal_drop"].values())
    assert np.isfinite([value["top1_drop"] for value in report["temporal_causal_drop"].values()]).all()


def test_milestone3p_prior_audit_keeps_future_tensors_out_of_observable_inputs() -> None:
    report = _prior_audit()
    assert not report["static_clue_audit"]["source_contains_future_label_as_observable"]
    assert not report["static_clue_audit"]["future_field_is_in_observable_whitelist"]


def test_milestone3p_negative_shortcut_control_is_solved_statically() -> None:
    dataset = CounterfactualTemporalDataset(
        split="test", regime="negative_shortcut_control", split_counts={"train": 48, "val": 16, "test": 24}, include_hidden_state=False
    )
    rows = _materialize(dataset)
    score = _score_for_observable_baseline(rows, "CurrentPositionProximity")
    assert _metric_report(rows, score, score_name="negative_static")["top1_success"] >= 0.9


def test_milestone3p_paired_bootstrap_and_new_metric_schema_are_correct() -> None:
    bootstrap = paired_bootstrap_ci([1.0, 1.0, 1.0], [0.5, 0.5, 0.5], seed=17)
    assert bootstrap["delta"] == 0.5
    assert bootstrap["ci_low"] == 0.5
    assert bootstrap["ci_high"] == 0.5
    _, rows = _matched_rows()
    score = _score_for_observable_baseline(rows, "CurrentPositionProximity")
    report = _metric_report(rows, score, score_name="fresh_milestone3p_metric")
    assert report["evaluation_artifact_schema"] == SCHEMA_VERSION
    assert report["score_name"] == "fresh_milestone3p_metric"
