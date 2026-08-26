"""CPU--CUDA numerical equivalence check for the frozen Milestone-2E inputs.

This is an operational qualification experiment, not a replacement for the
CPU-only final-v2 evaluation.  It uses the validation-selected five checkpoints
and the first immutable C20 test batch, compares the CPU and CUDA forward
passes, and records score/ranking agreement in a new final-v2 artifact.  It
never alters the existing final reports and deliberately refuses to run without
CUDA.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from actmask.experiments.milestone2e import DEFAULT_CONFIG, _load_config
from actmask.experiments.milestone2e_final_v2 import (
    FAIR_RANKING_SELECTION_FILE,
    FINAL_ROOT,
    FIXED_CANDIDATE_COUNTS,
    LEGACY_ROOT,
    _load_final_fair_ranking_selection,
    _load_primary_models,
    _selected_primary,
    _timestamp,
    _write_json_new,
    build_final_v2_datasets,
)
from actmask.experiments.milestone2e_final_cache import code_provenance_for_files
from actmask.experiments.milestone2e_full import _forward, _json_digest


EQUIVALENCE_SCHEMA_VERSION = "milestone2e-final-v2-cpu-cuda-equivalence-v1"
DEFAULT_OUTPUT_PATH = FINAL_ROOT / "gpu_equivalence" / "ranking_test_c20_first_batch.json"
OUTPUT_NAMES = ("utility_logits", "mask_logits")


def _to_device(value: Any, device: torch.device) -> Any:
    """Recursively move tensors while preserving observable mapping structure."""

    if isinstance(value, torch.Tensor):
        return value.to(device=device, non_blocking=False)
    if isinstance(value, Mapping):
        return {key: _to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_to_device(item, device) for item in value]
    return value


def _tensor_digest(value: torch.Tensor) -> str:
    raw = value.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _observable_manifest(observable: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, value in observable.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"observable field {name!r} is not a tensor")
        result[str(name)] = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": _tensor_digest(value),
        }
    return result


def _complete_group_rankings(logits: np.ndarray, candidate_set_id: torch.Tensor) -> dict[str, list[int]]:
    ids = candidate_set_id.detach().cpu().numpy().reshape(-1)
    result: dict[str, list[int]] = {}
    for group_id in np.unique(ids):
        indices = np.flatnonzero(ids == group_id)
        # A partial group would not have a meaningful ranking comparison.
        if indices.size != 20:
            continue
        ordered = indices[np.argsort(-np.asarray(logits)[indices], kind="stable")]
        result[str(int(group_id))] = ordered.astype(int).tolist()
    return result


def _comparison(cpu: torch.Tensor, cuda: torch.Tensor) -> dict[str, Any]:
    cpu_value = cpu.detach().cpu()
    cuda_value = cuda.detach().cpu()
    difference = (cpu_value - cuda_value).abs()
    relative = difference / cpu_value.abs().clamp_min(1.0e-12)
    return {
        "shape": list(cpu_value.shape),
        "max_abs_difference": float(difference.max().item()),
        "mean_abs_difference": float(difference.mean().item()),
        "max_relative_difference": float(relative.max().item()),
        "allclose_atol_1e-5_rtol_1e-4": bool(torch.allclose(cpu_value, cuda_value, atol=1.0e-5, rtol=1.0e-4)),
    }


def _verify_cross_backend_inputs(
    config_path: Path, *, legacy_root: Path, final_root: Path
) -> dict[str, Any]:
    """Validate frozen inputs while explicitly permitting only backend drift.

    ``verify_inputs`` correctly requires the CPU runtime used by final Stage-A
    artifacts.  A CUDA qualification must differ only in that runtime, so this
    narrow verifier preserves all source/config/checkpoint invariants and
    records the permitted backend difference.
    """

    config = _load_config(config_path)
    if str(config["experiment"]["device"]).lower() != "cpu":
        raise ValueError("the frozen source protocol must declare CPU")
    if tuple(int(value) for value in config["ranking"]["candidate_counts"]) != FIXED_CANDIDATE_COUNTS:
        raise ValueError("candidate-count protocol differs from frozen final-v2 inputs")
    selection = _selected_primary(config, legacy_root=legacy_root)
    fair_selection = _load_final_fair_ranking_selection(final_root / FAIR_RANKING_SELECTION_FILE, config=config)
    stored = fair_selection.get("source_code_provenance")
    if not isinstance(stored, Mapping):
        raise ValueError("fair comparator selection lacks source provenance")
    from actmask.experiments.milestone2e_final_v2 import _source_files

    current = code_provenance_for_files(_source_files(), repository_root=Path.cwd())
    for key in ("git_revision", "source_files_sha256", "source_tree_sha256"):
        if stored.get(key) != current.get(key):
            raise ValueError(f"cross-backend qualification found changed frozen source provenance: {key}")
    return {
        "config": config,
        "validation_selected_arm": selection,
        "frozen_cpu_runtime_provenance": dict(stored),
        "current_cuda_runtime_provenance": dict(current),
    }


def run(
    config_path: Path = DEFAULT_CONFIG,
    *,
    legacy_root: Path = LEGACY_ROOT,
    final_root: Path = FINAL_ROOT,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Compare CPU and CUDA forward passes for all frozen primary checkpoints."""

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; equivalence qualification cannot run")
    output_path = final_root / "gpu_equivalence" / "ranking_test_c20_first_batch.json" if output_path is None else Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite GPU-equivalence artifact: {output_path}")

    verification = _verify_cross_backend_inputs(
        Path(config_path), legacy_root=Path(legacy_root), final_root=Path(final_root)
    )
    config = verification["config"]
    dataset = build_final_v2_datasets(config)["ranking_test/20"]
    batch_size = int(config["training"]["evaluation_batch_size"])
    batch = next(iter(DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)))
    observable = batch["observable"]
    if not isinstance(observable, Mapping):
        raise TypeError("ranking C20 batch lacks an observable mapping")
    candidate_set_id = batch["metadata"]["candidate_set_id"]
    if not isinstance(candidate_set_id, torch.Tensor):
        raise TypeError("ranking C20 batch lacks tensor candidate-set IDs")

    selection = verification["validation_selected_arm"]
    cpu_models, specifications = _load_primary_models(config, legacy_root=legacy_root, selection=selection)
    device = torch.device("cuda:0")
    cuda_observable = _to_device(observable, device)
    per_seed: list[dict[str, Any]] = []
    all_passed = True

    for seed in sorted(cpu_models):
        cpu_model = cpu_models[seed].cpu().eval()
        cuda_model = copy.deepcopy(cpu_model).to(device).eval()
        with torch.no_grad():
            cpu_output = _forward(cpu_model, {"observable": observable})
            torch.cuda.synchronize(device)
            cuda_output = _forward(cuda_model, {"observable": cuda_observable})
            torch.cuda.synchronize(device)
        output_comparison = {
            name: _comparison(cpu_output[name], cuda_output[name]) for name in OUTPUT_NAMES
        }
        cpu_ranks = _complete_group_rankings(cpu_output["utility_logits"].detach().cpu().numpy(), candidate_set_id)
        cuda_ranks = _complete_group_rankings(cuda_output["utility_logits"].detach().cpu().numpy(), candidate_set_id)
        rank_agreement = cpu_ranks == cuda_ranks
        passed = bool(rank_agreement and all(value["allclose_atol_1e-5_rtol_1e-4"] for value in output_comparison.values()))
        all_passed = all_passed and passed
        per_seed.append(
            {
                "seed": int(seed),
                "checkpoint": {
                    "path": specifications[seed].path,
                    "sha256": specifications[seed].sha256,
                    "best_epoch": specifications[seed].best_epoch,
                },
                "outputs": output_comparison,
                "complete_c20_groups_compared": len(cpu_ranks),
                "rankings_identical_for_complete_groups": bool(rank_agreement),
                "passed": passed,
            }
        )
        del cuda_model, cuda_output
        torch.cuda.empty_cache()

    report = {
        "artifact_schema": EQUIVALENCE_SCHEMA_VERSION,
        "stage": "gpu_qualification_cpu_cuda_numerical_equivalence",
        "completed_at_utc": _timestamp(),
        "purpose": "Qualification only; does not replace the CPU-only Stage-A artifact.",
        "input_protocol": {
            "config_digest": _json_digest(config),
            "validation_selected_arm": selection,
            "dataset": "ranking_test/20",
            "batch_size": batch_size,
            "observable_manifest": _observable_manifest(observable),
            "candidate_order": "DataLoader(shuffle=False), first frozen C20 batch",
        },
        "runtime": {
            "frozen_cpu_provenance": verification["frozen_cpu_runtime_provenance"],
            "cuda_provenance": verification["current_cuda_runtime_provenance"],
            "torch": torch.__version__,
            "cuda_build": torch.version.cuda,
            "device": torch.cuda.get_device_name(device),
            "device_capability": list(torch.cuda.get_device_capability(device)),
        },
        "criteria": {
            "utility_and_mask": "torch.allclose(atol=1e-5, rtol=1e-4)",
            "ranking": "identical stable descending utility-logit order for every complete C20 group in batch",
        },
        "per_seed": per_seed,
        "overall_passed": bool(all_passed),
        "gpu_used": True,
        "existing_cpu_stage_a_process_modified": False,
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
    print("passed" if report["overall_passed"] else "failed")
    return report


if __name__ == "__main__":
    main()
