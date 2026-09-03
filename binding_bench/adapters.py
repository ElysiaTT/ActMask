"""Adapter boundary between arbitrary verifiers and the independent evaluator."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, runtime_checkable

import numpy as np

from .interventions import CASES, load_suite
from .schema import (
    BindingDataset,
    PredictionBundle,
    dataset_digest,
    load_prediction,
    save_prediction,
)


@runtime_checkable
class PredictionAdapter(Protocol):
    """A verifier that consumes public history/candidates and emits larger-is-better scores."""

    method_id: str
    training_data_digests: tuple[str, ...]

    def predict(self, inputs: BindingDataset) -> Mapping[str, np.ndarray] | np.ndarray:
        """Return scores or a mapping with scores and optional effect uncertainty."""


@dataclass
class CallableAdapter:
    method_id: str
    function: Callable[[BindingDataset], Mapping[str, np.ndarray] | np.ndarray]
    training_data_digests: tuple[str, ...] = ()

    def predict(self, inputs: BindingDataset) -> Mapping[str, np.ndarray] | np.ndarray:
        return self.function(inputs)


def _as_output(value: Mapping[str, np.ndarray] | np.ndarray) -> dict[str, np.ndarray]:
    if isinstance(value, np.ndarray):
        return {"scores": value}
    if "scores" not in value:
        raise ValueError("adapter output must contain scores")
    allowed = {"scores", "predicted_effects", "uncertainty", "candidate_ids"}
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"unknown adapter outputs: {sorted(unknown)}")
    return {name: np.asarray(item) for name, item in value.items()}


def run_adapter(
    adapter: PredictionAdapter,
    suite_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Run a frozen method on every case without exposing evaluator truth to it."""

    suite_manifest, cases = load_suite(suite_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, str] = {}
    for case in CASES:
        dataset = cases[case]
        public = dataset.public_view()
        output = _as_output(adapter.predict(public))
        prediction = PredictionBundle(
            method_id=str(adapter.method_id),
            dataset_id=dataset.dataset_id,
            split=dataset.split,
            intervention=case,
            scores=output["scores"],
            candidate_ids=output.get("candidate_ids", dataset.candidate_ids.copy()),
            twin_ids=dataset.twin_ids.copy(),
            dataset_digest=dataset_digest(dataset),
            training_data_digests=tuple(str(value) for value in adapter.training_data_digests),
            predicted_effects=output.get("predicted_effects"),
            uncertainty=output.get("uncertainty"),
        )
        relative = f"{case}.npz"
        save_prediction(output_dir / relative, prediction, dataset)
        files[case] = relative
    manifest = {
        "schema_version": "actmask-binding-method-output-v1",
        "method_id": str(adapter.method_id),
        "suite_dataset_id": suite_manifest["dataset_id"],
        "suite_split": suite_manifest["split"],
        "training_data_digests": list(adapter.training_data_digests),
        "predictions": files,
    }
    (output_dir / "predictions.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def load_method_predictions(
    output_dir: str | Path,
    cases: Mapping[str, BindingDataset],
) -> tuple[dict[str, Any], dict[str, PredictionBundle]]:
    output_dir = Path(output_dir)
    manifest = json.loads((output_dir / "predictions.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "actmask-binding-method-output-v1":
        raise ValueError("unsupported method output manifest")
    if set(manifest.get("predictions", {})) != set(CASES):
        raise ValueError("method output must cover every intervention case")
    predictions = {
        case: load_prediction(output_dir / manifest["predictions"][case], cases[case])
        for case in CASES
    }
    method_ids = {prediction.method_id for prediction in predictions.values()}
    if method_ids != {manifest["method_id"]}:
        raise ValueError("method IDs are inconsistent across prediction files")
    for case, prediction in predictions.items():
        if prediction.intervention != case:
            raise ValueError(f"prediction intervention is mislabeled for {case}")
    return manifest, predictions


__all__ = [
    "CallableAdapter",
    "PredictionAdapter",
    "load_method_predictions",
    "run_adapter",
]
