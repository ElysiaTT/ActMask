"""Reproducible B0 CUDA qualification for the Milestone 3P GPU branch.

This checks a fixed observable-only mini-batch against CPU evaluation.  It is
not a simulator benchmark and does not load, train, or alter a learned model.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from actmask.data.milestone3p_counterfactual import CounterfactualTemporalDataset
from actmask.models.milestone2b_baselines import (
    CurrentPositionProximity,
    EstimatedTimeAlignedTrajectoryProximity,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "actmask" / "milestone3p_gpu_benchmark"
OUTPUT_PATH = OUTPUT_ROOT / "gpu_qualification_runtime.json"
SCHEMA_VERSION = "milestone3p-gpu-qualification-v1"


def _timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _nvidia_smi() -> dict[str, str]:
    fields = "index,name,driver_version,memory.total,memory.used,memory.free,utilization.gpu"
    completed = subprocess.run(
        ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader"],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    return {
        "exit_code": str(completed.returncode),
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def _batch() -> dict[str, torch.Tensor]:
    dataset = CounterfactualTemporalDataset(
        split="test", regime="matched_counterfactual", split_counts={"train": 4, "val": 2, "test": 2}, include_hidden_state=False
    )
    samples = [dataset[index]["observable"] for index in range(10)]
    return {name: torch.stack([sample[name] for sample in samples], dim=0) for name in samples[0]}


def _scores(model: torch.nn.Module, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    with torch.inference_mode():
        return model(batch)["utility_score"].detach()


def _ranking_top_indices(score: torch.Tensor) -> list[int]:
    values = score.detach().float().cpu().numpy()
    return [int(np.argmax(values[offset : offset + 5]) + offset) for offset in range(0, len(values), 5)]


def _model_report(name: str, model: torch.nn.Module, batch: dict[str, torch.Tensor]) -> dict[str, Any]:
    cpu_model = model.cpu().eval()
    cpu_batch = {key: value.cpu() for key, value in batch.items()}
    cpu = _scores(cpu_model, cpu_batch).float().cpu()

    gpu_model = model.cuda().eval()
    gpu_batch = {key: value.cuda(non_blocking=False) for key, value in batch.items()}
    torch.cuda.synchronize()
    gpu_first = _scores(gpu_model, gpu_batch).float().cpu()
    torch.cuda.synchronize()
    gpu_second = _scores(gpu_model, gpu_batch).float().cpu()

    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
        mixed = gpu_model(gpu_batch)["utility_score"].detach().float().cpu()
    torch.cuda.synchronize()
    return {
        "method": name,
        "batch_rows": int(len(cpu)),
        "cpu_cuda_fp32": {
            "max_abs_error": float((cpu - gpu_first).abs().max().item()),
            "allclose_atol_1e-5_rtol_1e-4": bool(torch.allclose(cpu, gpu_first, atol=1.0e-5, rtol=1.0e-4)),
            "ranking_top_indices_equal": _ranking_top_indices(cpu) == _ranking_top_indices(gpu_first),
        },
        "cuda_repeatability": {
            "max_abs_error": float((gpu_first - gpu_second).abs().max().item()),
            "bitwise_equal": bool(torch.equal(gpu_first, gpu_second)),
        },
        "mixed_precision_fp16": {
            "finite": bool(torch.isfinite(mixed).all()),
            "max_abs_error_vs_cuda_fp32": float((gpu_first - mixed).abs().max().item()),
            "ranking_top_indices_equal": _ranking_top_indices(gpu_first) == _ranking_top_indices(mixed),
        },
    }


def run(*, output_path: Path = OUTPUT_PATH) -> dict[str, Any]:
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite GPU qualification: {output_path}")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("CUDA is not visible; B0 GPU qualification cannot run")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(20300417)
    torch.cuda.manual_seed_all(20300417)
    device = torch.cuda.get_device_properties(0)
    free, total = torch.cuda.mem_get_info(0)
    batch = _batch()
    reports = [
        _model_report("CurrentPositionProximity", CurrentPositionProximity(), batch),
        _model_report("EstimatedTimeAlignedTrajectoryProximity", EstimatedTimeAlignedTrajectoryProximity(), batch),
    ]
    passed = all(
        item["cpu_cuda_fp32"]["allclose_atol_1e-5_rtol_1e-4"]
        and item["cpu_cuda_fp32"]["ranking_top_indices_equal"]
        and item["cuda_repeatability"]["bitwise_equal"]
        and item["mixed_precision_fp16"]["finite"]
        and item["mixed_precision_fp16"]["ranking_top_indices_equal"]
        for item in reports
    )
    result = {
        "artifact_schema": SCHEMA_VERSION,
        "stage": "B0_gpu_qualification",
        "recorded_at": _timestamp(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_visible": True,
        "gpu": {
            "index": 0,
            "name": device.name,
            "compute_capability": f"{device.major}.{device.minor}",
            "memory_total_bytes": int(total),
            "memory_free_bytes_before_qualification": int(free),
        },
        "nvidia_smi": _nvidia_smi(),
        "deterministic_configuration": {
            "torch_use_deterministic_algorithms": True,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "limitation": "Determinism is checked for these small observable baselines only; simulator kernels and future learned CUDA operators require their own qualification.",
        },
        "frozen_minibatch": {
            "source": "CounterfactualTemporalDataset matched_counterfactual test; fixed first ten observable candidate rows",
            "hidden_state_excluded": True,
        },
        "methods": reports,
        "passed": passed,
        "decision": "B0_PASSED_PROCEED_TO_APPROVED_B2_ONLY_IF_LOCAL_RUNTIME_AND_DATA_ARE_AVAILABLE" if passed else "B0_FAILED_DO_NOT_START_GPU_BENCHMARK",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> int:
    report = run()
    print(json.dumps({"passed": report["passed"], "decision": report["decision"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
