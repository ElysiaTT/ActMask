"""Read-only fail-closed GPU host preflight for the binding benchmark."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _normalize_machine(value: str) -> str:
    lowered = value.lower()
    if lowered in {"amd64", "x64", "x86-64"}:
        return "x86_64"
    if lowered in {"arm64", "aarch64"}:
        return "aarch64"
    return lowered


def _run(command: list[str], *, timeout: int = 30) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"returncode": None, "stdout": "", "stderr": str(error)}
    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def _torch_probe() -> dict[str, Any]:
    source = (
        "import json,torch; "
        "print(json.dumps({'version':torch.__version__,'cuda_available':torch.cuda.is_available(),"
        "'cuda_version':torch.version.cuda,'device_count':torch.cuda.device_count()}))"
    )
    execution = _run([sys.executable, "-c", source])
    output: dict[str, Any] = {"execution": execution}
    if execution["returncode"] == 0:
        try:
            output.update(json.loads(execution["stdout"].splitlines()[-1]))
        except (json.JSONDecodeError, IndexError):
            output["parse_error"] = "torch probe did not emit valid JSON"
    return output


def _maniskill_probe() -> dict[str, Any]:
    source = (
        "import json,mani_skill; "
        "print(json.dumps({'version':getattr(mani_skill,'__version__','unknown')}))"
    )
    execution = _run([sys.executable, "-c", source])
    output: dict[str, Any] = {"execution": execution}
    if execution["returncode"] == 0:
        try:
            output.update(json.loads(execution["stdout"].splitlines()[-1]))
        except (json.JSONDecodeError, IndexError):
            output["parse_error"] = "ManiSkill probe did not emit valid JSON"
    return output


def _nvidia_probe() -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return {"available": False, "gpus": [], "execution": None}
    execution = _run(
        [
            executable,
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    gpus = []
    if execution["returncode"] == 0:
        for line in execution["stdout"].splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) == 3:
                try:
                    memory_mib = int(fields[1])
                except ValueError:
                    continue
                gpus.append(
                    {
                        "name": fields[0],
                        "memory_mib": memory_mib,
                        "memory_gib": memory_mib / 1024.0,
                        "driver_version": fields[2],
                    }
                )
    return {"available": bool(gpus), "gpus": gpus, "execution": execution}


def check_readiness(config: dict[str, Any], workspace: str | Path) -> dict[str, Any]:
    workspace = Path(workspace).resolve()
    if not workspace.is_dir():
        raise ValueError(f"workspace does not exist: {workspace}")
    disk = shutil.disk_usage(workspace)
    torch_probe = _torch_probe()
    maniskill_probe = _maniskill_probe()
    nvidia = _nvidia_probe()
    observed_packages = {
        name: _distribution_version(name) for name in config["required_packages"]
    }
    package_checks = {
        name: observed_packages[name] is not None
        and (required is None or observed_packages[name] == required)
        for name, required in config["required_packages"].items()
    }
    os_name = platform.system()
    machine = _normalize_machine(platform.machine())
    python_major_minor = f"{sys.version_info.major}.{sys.version_info.minor}"
    maximum_vram = max(
        (float(gpu["memory_gib"]) for gpu in nvidia["gpus"]),
        default=0.0,
    )
    remote_host = config.get("remote_host", {})
    remote_identity_fields = (
        "hostname",
        "provider",
        "region",
        "instance_type",
        "expected_gpu_model",
    )
    checks = {
        "platform": os_name == config["required_platform"],
        "architecture": machine == config["required_architecture"],
        "python": python_major_minor == config["required_python_major_minor"],
        "packages": all(package_checks.values()),
        "maniskill_import": maniskill_probe.get("execution", {}).get("returncode") == 0,
        "torch_import": torch_probe.get("execution", {}).get("returncode") == 0,
        "cuda_available": (
            bool(torch_probe.get("cuda_available"))
            if config["required_cuda_available"]
            else True
        ),
        "nvidia_smi": bool(nvidia["available"]) if config["required_nvidia_smi"] else True,
        "vram": maximum_vram >= float(config["minimum_vram_gib"]),
        "disk": disk.free / 2**30 >= float(config["minimum_free_disk_gib"]),
        "dataset_license_reviewed": bool(config["dataset_license_reviewed"]),
        "gpu_hours_bounded": float(config["maximum_gpu_hours"]) > 0,
        "cost_amount_set": config["approved_cost_amount"] is not None,
        "gpu_hourly_price_set": (
            isinstance(config.get("gpu_hourly_price"), (int, float))
            and not isinstance(config.get("gpu_hourly_price"), bool)
            and float(config["gpu_hourly_price"]) >= 0.0
        ),
        "cost_authorized": bool(config["cost_authorized"]),
        "remote_host_identity_set": all(
            isinstance(remote_host.get(field), str) and bool(remote_host[field].strip())
            for field in remote_identity_fields
        ),
        "remote_host_approved": bool(config["remote_host_approved"]),
    }
    return {
        "schema_version": "actmask-binding-gpu-readiness-v1",
        "utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "observed": {
            "workspace": str(workspace),
            "platform": os_name,
            "platform_release": platform.platform(),
            "architecture": machine,
            "python": sys.version,
            "python_major_minor": python_major_minor,
            "packages": observed_packages,
            "free_disk_gib": disk.free / 2**30,
            "nvidia": nvidia,
            "maximum_vram_gib": maximum_vram,
            "torch": torch_probe,
            "maniskill": maniskill_probe,
        },
        "checks": checks,
        "failures": [name for name, passed in checks.items() if not passed],
        "decision": "GPU_START_GO" if all(checks.values()) else "GPU_START_NO_GO",
        "scope_note": "This read-only preflight does not install packages, download datasets, start training, or authorize spend.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("schema_version") != "actmask-binding-gpu-readiness-config-v1":
        raise SystemExit("unsupported readiness config")
    result = check_readiness(config, args.workspace)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"decision": result["decision"], "failures": result["failures"]}, indent=2))
    raise SystemExit(0 if result["decision"] == "GPU_START_GO" else 2)


if __name__ == "__main__":
    main()
