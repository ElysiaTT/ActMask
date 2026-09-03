"""Dependency-light Counterfactual Action-Effect Binding Audit."""

from .adapters import CallableAdapter, PredictionAdapter, run_adapter
from .benchmark import build_benchmark_manifest, evaluate_benchmark_plan, split_leakage_audit
from .evaluator import DEFAULT_GATES, evaluate_suite
from .interventions import build_intervention_suite
from .schema import (
    DATASET_SCHEMA,
    PREDICTION_SCHEMA,
    SUITE_SCHEMA,
    BindingDataset,
    PredictionBundle,
    load_dataset,
    load_prediction,
    save_dataset,
    save_prediction,
    validate_dataset,
    validate_prediction,
)

__all__ = [
    "BindingDataset",
    "CallableAdapter",
    "DATASET_SCHEMA",
    "DEFAULT_GATES",
    "PREDICTION_SCHEMA",
    "PredictionAdapter",
    "PredictionBundle",
    "SUITE_SCHEMA",
    "build_benchmark_manifest",
    "build_intervention_suite",
    "evaluate_suite",
    "evaluate_benchmark_plan",
    "load_dataset",
    "load_prediction",
    "run_adapter",
    "save_dataset",
    "save_prediction",
    "split_leakage_audit",
    "validate_dataset",
    "validate_prediction",
]
