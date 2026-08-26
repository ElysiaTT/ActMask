"""GPU-qualified shadow execution of the frozen A4 temporal diagnostic.

The official final-v2 A4 wrapper remains CPU-only.  This companion uses only
the previously qualified utility-logit/ranking forward path on CUDA and emits
a separate artifact, so it can be compared against the official CPU result
without overwriting it.  Dataset construction, views, fair scores, statistics,
checkpoints, seeds, and ordering are delegated to the established diagnostic.
"""

from __future__ import annotations

import argparse
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from actmask.experiments import milestone2e_temporal_diagnostics as temporal
from actmask.experiments.milestone2e import DEFAULT_CONFIG
from actmask.experiments.milestone2e_final_cache import code_provenance_for_files
from actmask.experiments.milestone2e_final_v2 import FINAL_ROOT, LEGACY_ROOT, _timestamp, _write_json_new
from actmask.experiments.milestone2e_gpu_equivalence import _to_device, _verify_cross_backend_inputs
from actmask.experiments.milestone2e_full import FactorialPrediction


GPU_TEMPORAL_SCHEMA_VERSION = "milestone2e-final-v2-temporal-gpu-qualification-v1"
DEFAULT_OUTPUT_PATH = FINAL_ROOT / "temporal_gpu_qualification" / "all_validation_promising.json"
UTILITY_QUALIFICATION_PATH = FINAL_ROOT / "gpu_equivalence" / "ranking_test_c20_first_batch.json"

# The CUDA environment has NumPy 1.x, where the identically defined API is
# named ``trapz``.  Keep the compatibility local to this shadow execution;
# the CPU project environment already supplies ``trapezoid``.
NUMPY_TRAPEZOID_COMPATIBILITY = not hasattr(np, "trapezoid")
if NUMPY_TRAPEZOID_COMPATIBILITY:
    np.trapezoid = np.trapz


def _utility_qualification() -> dict[str, Any]:
    if not UTILITY_QUALIFICATION_PATH.exists():
        raise FileNotFoundError(f"missing GPU utility qualification: {UTILITY_QUALIFICATION_PATH}")
    import json

    value = json.loads(UTILITY_QUALIFICATION_PATH.read_text())
    if value.get("artifact_schema") != "milestone2e-final-v2-cpu-cuda-equivalence-v1":
        raise ValueError("GPU utility qualification has an incompatible schema")
    rows = value.get("per_seed")
    if not isinstance(rows, list) or len(rows) != 5:
        raise ValueError("GPU utility qualification does not cover all five frozen seeds")
    for row in rows:
        utility = row.get("outputs", {}).get("utility_logits", {})
        if utility.get("allclose_atol_1e-5_rtol_1e-4") is not True or row.get("rankings_identical_for_complete_groups") is not True:
            raise ValueError("GPU utility/logit ranking equivalence did not pass for every frozen seed")
    return value


def _gpu_predict_factory(device: torch.device):
    @torch.no_grad()
    def predict(model: Any, dataset: Dataset[Any], *, batch_size: int, view: str) -> FactorialPrediction:
        probabilities: list[np.ndarray] = []
        logits: list[np.ndarray] = []
        success: list[np.ndarray] = []
        utility: list[np.ndarray] = []
        metadata: list[dict[str, Any]] = []
        model.eval()
        for batch in temporal._loader(dataset, batch_size):
            gpu_batch = dict(batch)
            gpu_batch["observable"] = _to_device(batch["observable"], device)
            if "hidden_state" in batch:
                gpu_batch["hidden_state"] = _to_device(batch["hidden_state"], device)
            output = temporal._forward_diagnostic(model, gpu_batch, view=view)
            raw = output["utility_logits"].detach().cpu().numpy()
            logits.append(raw)
            probabilities.append(1.0 / (1.0 + np.exp(-raw)))
            success.append(batch["targets"]["success"].detach().cpu().numpy())
            utility.append(batch["targets"]["candidate_utility"].detach().cpu().numpy())
            metadata.extend(temporal._metadata_rows(batch["metadata"]))
        if not probabilities:
            raise ValueError("GPU temporal diagnostic received an empty dataset")
        return FactorialPrediction(
            utility_probability=np.concatenate(probabilities),
            utility_logits=np.concatenate(logits),
            success=np.concatenate(success),
            candidate_utility=np.concatenate(utility),
            metadata=metadata,
            mask_probability=np.empty(0, dtype=np.float64),
            mask_target=np.empty(0, dtype=np.float64),
        )

    return predict


def _source_files() -> list[Path]:
    return [
        Path(__file__),
        Path(temporal.__file__),
        Path(__file__).with_name("milestone2e_gpu_equivalence.py"),
        Path(__file__).with_name("milestone2e_final_v2.py"),
        Path(__file__).parent.parent / "models" / "milestone2e.py",
        Path(__file__).parent.parent / "data" / "milestone2b_dataset.py",
    ]


def run(
    config_path: Path = DEFAULT_CONFIG,
    *,
    legacy_root: Path = LEGACY_ROOT,
    final_root: Path = FINAL_ROOT,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Run a separate CUDA shadow report for all validation-promising A4 arms."""

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for the qualified temporal shadow run")
    final_root = Path(final_root)
    output_path = final_root / "temporal_gpu_qualification" / "all_validation_promising.json" if output_path is None else Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite GPU temporal qualification: {output_path}")
    qualification = _utility_qualification()
    frozen = _verify_cross_backend_inputs(Path(config_path), legacy_root=Path(legacy_root), final_root=final_root)
    device = torch.device("cuda:0")

    original_load_checkpoint = temporal._load_checkpoint
    original_predict = temporal.predict_temporal_diagnostic

    def load_checkpoint_gpu(**kwargs: Any):
        model, specification = original_load_checkpoint(**kwargs)
        return model.to(device).eval(), specification

    try:
        temporal._load_checkpoint = load_checkpoint_gpu
        temporal.predict_temporal_diagnostic = _gpu_predict_factory(device)
        with tempfile.TemporaryDirectory(prefix="actmask_m2e_temporal_gpu_") as directory:
            scratch_path = Path(directory) / "established_temporal_gpu_shadow.json"
            established = temporal.run(
                config_path=Path(config_path),
                output_path=scratch_path,
                output_root=Path(legacy_root),
                all_validation_promising=True,
            )
    finally:
        temporal._load_checkpoint = original_load_checkpoint
        temporal.predict_temporal_diagnostic = original_predict
        torch.cuda.empty_cache()

    if established.get("status") != "completed_frozen_c20":
        raise ValueError(f"GPU temporal shadow did not complete: {established.get('status')!r}")
    established["cpu_only"] = False
    established["gpu_used"] = True
    established["scope"] = (
        "GPU-qualified shadow of the CPU-only frozen C20 diagnostic. It is a separate comparison artifact and "
        "does not overwrite the official CPU final-v2 A4 report."
    )
    report = {
        "artifact_schema": GPU_TEMPORAL_SCHEMA_VERSION,
        "stage": "A4_all_validation_promising_temporal_gpu_qualification",
        "completed_at_utc": _timestamp(),
        "gpu_used": True,
        "device": torch.cuda.get_device_name(device),
        "numpy_trapezoid_compatibility": {
            "applied": NUMPY_TRAPEZOID_COMPATIBILITY,
            "implementation": "numpy.trapz" if NUMPY_TRAPEZOID_COMPATIBILITY else "numpy.trapezoid",
            "reason": "CUDA environment compatibility; both APIs calculate the same trapezoidal integral.",
        },
        "validation_selected_arm": frozen["validation_selected_arm"],
        "utility_only_equivalence_qualification": qualification,
        "source_code_provenance": code_provenance_for_files(_source_files(), repository_root=Path.cwd()),
        "established_temporal_gpu_shadow": established,
        "comparison_policy": "Compare this artifact to the independently retained CPU A4 envelope before treating it as corroborating evidence.",
    }
    _write_json_new(output_path, report)
    return report


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-root", type=Path, default=LEGACY_ROOT)
    parser.add_argument("--output-root", type=Path, default=FINAL_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    arguments = parser.parse_args(argv)
    report = run(legacy_root=arguments.legacy_root, final_root=arguments.output_root, output_path=arguments.output)
    print("complete")
    return report


if __name__ == "__main__":
    main()
