"""CPU-only contracts for the reusable binding audit."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from binding_bench.adapters import CallableAdapter, run_adapter
from binding_bench.benchmark import (
    EVALUATION_PLAN_SCHEMA,
    build_benchmark_manifest,
    evaluate_benchmark_plan,
    split_leakage_audit,
)
from binding_bench.evaluator import evaluate_suite
from binding_bench.example import _linear_binding, _result_mean, make_example_dataset
from binding_bench.interventions import CASES, build_intervention_suite, load_suite, make_cases
from binding_bench.metrics import align_scores, score_prediction
from binding_bench.schema import (
    PredictionBundle,
    dataset_digest,
    load_dataset,
    save_dataset,
    validate_dataset,
    validate_prediction,
)


def small_dataset():
    return make_example_dataset(seeds=(31, 32), twins_per_seed=6, history_length=8, candidate_count=7)


def test_schema_round_trip_and_digest_rejects_mismatched_truth(tmp_path) -> None:
    dataset = small_dataset()
    audit = save_dataset(tmp_path / "dataset.npz", dataset)
    loaded = load_dataset(tmp_path / "dataset.npz")
    assert audit["dataset_digest"] == dataset_digest(loaded)
    assert audit["twins"] == 12
    changed = replace(loaded, true_utility=loaded.true_utility + 0.01)
    assert dataset_digest(changed) != audit["dataset_digest"]


def test_schema_rejects_candidate_mismatch_and_training_leak() -> None:
    dataset = small_dataset()
    invalid_ids = dataset.candidate_ids.copy()
    invalid_ids[0, 1, 0] = "not-a-matched-candidate"
    with pytest.raises(ValueError, match="same candidate ID set"):
        validate_dataset(replace(dataset, candidate_ids=invalid_ids))
    invalid_context = dataset.context.copy()
    invalid_context[0, 1, 0] += 0.1
    with pytest.raises(ValueError, match="byte-identical public context"):
        validate_dataset(replace(dataset, context=invalid_context))
    invalid_candidate = dataset.candidates.copy()
    invalid_candidate[0, 1, 0, 0] += 0.1
    with pytest.raises(ValueError, match="identical candidate actions"):
        validate_dataset(replace(dataset, candidates=invalid_candidate))
    prediction = PredictionBundle(
        method_id="bad",
        dataset_id=dataset.dataset_id,
        split=dataset.split,
        intervention="original",
        scores=np.zeros_like(dataset.true_utility),
        candidate_ids=dataset.candidate_ids,
        twin_ids=dataset.twin_ids,
        dataset_digest=dataset_digest(dataset),
        training_data_digests=(dataset_digest(dataset),),
    )
    with pytest.raises(ValueError, match="appears in training"):
        validate_prediction(prediction, dataset)


def test_all_interventions_preserve_their_exact_contract() -> None:
    dataset = small_dataset()
    cases = make_cases(dataset, seed=91)
    assert tuple(cases) == CASES
    assert not np.array_equal(
        cases["binding_breaking"].history_effects,
        dataset.history_effects,
    )
    assert np.array_equal(
        cases["twin_history_swap"].history_effects,
        dataset.history_effects[:, ::-1],
    )
    assert not np.array_equal(
        cases["candidate_slot_permutation"].candidate_ids,
        dataset.candidate_ids,
    )


def test_adapter_never_receives_truth_and_suite_round_trips(tmp_path) -> None:
    dataset = small_dataset()
    suite = tmp_path / "suite"
    build_intervention_suite(dataset, suite, seed=5, source={"kind": "test"})
    manifest, cases = load_suite(suite)
    assert manifest["integrity"]["passed"] is True
    assert set(cases) == set(CASES)
    calls = []

    def verifier(inputs):
        calls.append(inputs.true_utility.shape)
        assert inputs.true_utility.size == 0
        assert inputs.candidate_effects is None
        return np.zeros(inputs.candidates.shape[:3], dtype=np.float64)

    run_adapter(CallableAdapter("blind", verifier), suite, tmp_path / "blind")
    assert calls == [(0, 2, 0)] * len(CASES)


def test_tie_aware_metrics_and_candidate_id_alignment() -> None:
    dataset = small_dataset()
    scores = np.zeros_like(dataset.true_utility)
    tied = PredictionBundle(
        method_id="tied",
        dataset_id=dataset.dataset_id,
        split=dataset.split,
        intervention="original",
        scores=scores,
        candidate_ids=dataset.candidate_ids.copy(),
        twin_ids=dataset.twin_ids.copy(),
        dataset_digest=dataset_digest(dataset),
    )
    metric = score_prediction(dataset, tied)
    assert metric["shortcut_diagnostics"]["candidate_constant_branch_fraction"] == 1.0
    expected_top1 = np.mean(
        np.sum(
            np.abs(dataset.true_utility - dataset.true_utility.max(axis=2, keepdims=True))
            <= 1.0e-12,
            axis=2,
        )
        / dataset.candidates_per_branch
    )
    assert metric["top1_accuracy"] == pytest.approx(expected_top1)

    reversed_prediction = replace(
        tied,
        scores=tied.scores[:, :, ::-1],
        candidate_ids=tied.candidate_ids[:, :, ::-1],
    )
    np.testing.assert_array_equal(align_scores(dataset, tied), align_scores(dataset, reversed_prediction))

    effect_prediction = replace(
        tied,
        scores=dataset.true_utility[:, :, ::-1].copy(),
        candidate_ids=dataset.candidate_ids[:, :, ::-1].copy(),
        predicted_effects=dataset.candidate_effects[:, :, ::-1].copy(),
    )
    effect_metric = score_prediction(dataset, effect_prediction)
    assert effect_metric["effect_prediction"]["mean_absolute_effect_error"] == 0.0


def test_independent_evaluator_reports_seed_split_bootstrap_and_controls(tmp_path) -> None:
    dataset = make_example_dataset(
        seeds=(41, 42, 43, 44, 45),
        twins_per_seed=8,
        history_length=9,
        candidate_count=8,
    )
    suite = tmp_path / "suite"
    build_intervention_suite(dataset, suite, seed=7, source={"kind": "test"})
    run_adapter(CallableAdapter("linear", _linear_binding), suite, tmp_path / "linear")
    run_adapter(CallableAdapter("mean", _result_mean), suite, tmp_path / "mean")
    result = evaluate_suite(
        suite,
        {"linear": tmp_path / "linear", "mean": tmp_path / "mean"},
        candidate_method="linear",
        baseline_methods=("mean",),
        gates={"minimum_twins_per_seed": 8, "bootstrap_replicates": 100},
    )
    assert result["admission_evaluable"] is True
    assert len(result["scores"]["linear"]["original"]["per_seed"]) == 5
    assert result["comparison"]["bootstrap_baseline_minus_candidate"]["replicates"] == 100
    assert result["intervention_effects"]["binding_regret_increase"] > 0.05
    assert result["checks"]["candidate_slot_invariance"] is True
    assert result["checks"]["anti_saturation_tpa"] is True
    assert result["checks"]["anti_saturation_regret"] is True


def _renamed_split(dataset, split: str, suffix: str):
    candidate_ids = np.char.add(dataset.candidate_ids, suffix)
    return replace(
        dataset,
        split=split,
        twin_ids=np.char.add(dataset.twin_ids, suffix),
        episode_ids=np.char.add(dataset.episode_ids, suffix),
        split_group_ids=np.char.add(dataset.split_group_ids, suffix),
        candidate_ids=candidate_ids,
    )


def test_group_disjoint_multi_split_manifest_and_leakage_rejection(tmp_path) -> None:
    first = _renamed_split(small_dataset(), "id_test", "-id")
    second = _renamed_split(small_dataset(), "held_mechanism", "-held")
    suites = {}
    for dataset in (first, second):
        suite = tmp_path / dataset.split
        build_intervention_suite(dataset, suite, seed=19, source={"kind": "test"})
        suites[dataset.split] = suite
    manifest = build_benchmark_manifest(
        tmp_path / "benchmark.json",
        benchmark_id="test-benchmark-v1",
        suites=suites,
        admission_splits=("id_test", "held_mechanism"),
        source={"kind": "test"},
    )
    assert manifest["split_leakage_audit"]["passed"] is True
    leaked = replace(second, split_group_ids=first.split_group_ids.copy())
    assert split_leakage_audit({"id_test": first, "held_mechanism": leaked})["passed"] is False


def test_multi_split_plan_runs_external_outputs_and_requires_every_split(tmp_path) -> None:
    datasets = {
        "id_test": _renamed_split(
            make_example_dataset(
                seeds=(51, 52, 53, 54, 55),
                twins_per_seed=6,
                history_length=9,
                candidate_count=8,
            ),
            "id_test",
            "-id",
        ),
        "held_mechanism": _renamed_split(
            make_example_dataset(
                seeds=(61, 62, 63, 64, 65),
                twins_per_seed=6,
                history_length=9,
                candidate_count=8,
            ),
            "held_mechanism",
            "-held",
        ),
    }
    suites = {}
    split_plans = {}
    for offset, (split, dataset) in enumerate(datasets.items()):
        suite = tmp_path / "suites" / split
        build_intervention_suite(dataset, suite, seed=101 + offset, source={"kind": "test"})
        linear = tmp_path / "predictions" / split / "linear"
        mean = tmp_path / "predictions" / split / "mean"
        run_adapter(CallableAdapter("linear", _linear_binding), suite, linear)
        run_adapter(CallableAdapter("mean", _result_mean), suite, mean)
        suites[split] = suite
        split_plans[split] = {
            "method_dirs": {
                "linear": str(linear),
                "mean": str(mean),
            }
        }

    benchmark_path = tmp_path / "benchmark.json"
    build_benchmark_manifest(
        benchmark_path,
        benchmark_id="end-to-end-test-v1",
        suites=suites,
        admission_splits=tuple(datasets),
        source={"kind": "test"},
    )
    plan = {
        "schema_version": EVALUATION_PLAN_SCHEMA,
        "benchmark_manifest": str(benchmark_path),
        "candidate_method": "linear",
        "baseline_methods": ["mean"],
        "gates": {
            "minimum_twins_per_seed": 6,
            "bootstrap_replicates": 80,
        },
        "bootstrap_seed": 3901,
        "splits": split_plans,
    }
    plan_path = tmp_path / "evaluation_plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    result = evaluate_benchmark_plan(plan_path)
    assert result["decision"] == "BENCHMARK_METHOD_GO"
    assert result["split_pass"] == {"id_test": True, "held_mechanism": True}
    assert set(result["split_results"]) == set(datasets)
    assert result["split_leakage_audit"]["passed"] is True

    incomplete = dict(plan)
    incomplete["splits"] = {"id_test": split_plans["id_test"]}
    plan_path.write_text(json.dumps(incomplete), encoding="utf-8")
    with pytest.raises(ValueError, match="cover every benchmark split"):
        evaluate_benchmark_plan(plan_path)
