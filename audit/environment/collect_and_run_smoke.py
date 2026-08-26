#!/usr/bin/env python3
"""Collect exact environment evidence and run the bounded state/GPU smoke."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT = Path("/data/project/tzh/papers/ActMask")
EVIDENCE = PROJECT / "audit" / "environment"
SMOKE_OUTPUT = EVIDENCE / "smoke"
STATUS = EVIDENCE / "environment_status.json"
IMPORT_PROBE = EVIDENCE / "import_probe.json"
COMMAND_HISTORY = EVIDENCE / "command_history.json"
REQUESTED = {
    "python": "3.11.15",
    "torch": "2.6.0+cu124",
    "torch_cuda": "12.4",
    "mani_skill": "3.0.1",
    "sapien": "3.0.3",
    "opencv_python_auxiliary": "4.10.0.84",
}


def _run(command: list[str]) -> dict:
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    return {
        "command": command,
        "exit_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path: Path) -> dict:
    return {
        "path": str(path),
        "exists": path.exists(),
        "bytes": path.stat().st_size if path.exists() else None,
        "sha256": _sha256(path) if path.exists() else None,
    }


def _shell(command: list[str], *, home: Path | None = None) -> str:
    prefix = f"cd {shlex.quote(str(PROJECT))} && "
    if home is not None:
        prefix += f"HOME={shlex.quote(str(home))} PYTHONPATH={shlex.quote(str(PROJECT))} "
    return prefix + shlex.join(command)


def main() -> None:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    SMOKE_OUTPUT.mkdir(parents=True, exist_ok=True)
    result: dict = {
        "schema_version": 1,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "requested_versions": REQUESTED,
        "environment_root": str(Path(sys.executable).resolve().parent.parent),
        "python": {
            "executable": sys.executable,
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
        },
        "process_environment": {
            "HOME": os.environ.get("HOME"),
            "PYTHONPATH": os.environ.get("PYTHONPATH"),
            "cwd": os.getcwd(),
        },
        "imports": {},
        "observed_versions": {},
        "command_results": {},
        "errors": [],
    }
    result["command_results"]["pip_check"] = _run([sys.executable, "-m", "pip", "check"])
    result["command_results"]["nvidia_smi"] = _run([
        "nvidia-smi", "--query-gpu=index,name,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    ])
    lock_commands = {
        "pip_freeze_all": [sys.executable, "-m", "pip", "freeze", "--all"],
        "conda_explicit_lock": [
            "/root/miniconda3/bin/conda", "list", "--explicit", "-p",
            "/data/envs/actmask-audit",
        ],
    }
    lock_paths = {
        "pip_freeze_all": EVIDENCE / "requirements.lock.txt",
        "conda_explicit_lock": EVIDENCE / "conda-explicit.lock.txt",
    }
    for name, command in lock_commands.items():
        detail = _run(command)
        result["command_results"][name] = detail
        lock_paths[name].write_text(detail["stdout"])
        if detail["exit_code"] != 0:
            result["errors"].append(f"{name} exited {detail['exit_code']}: {detail['stderr']}")

    import_command = [sys.executable, str(EVIDENCE / "import_probe_worker.py")]
    import_run = _run(import_command)
    result["command_results"]["native_import_probe"] = import_run
    (EVIDENCE / "logs" / "native_import_probe.log").write_text(
        import_run["stdout"] + import_run["stderr"]
    )
    if import_run["exit_code"] == 0 and IMPORT_PROBE.exists():
        probe = json.loads(IMPORT_PROBE.read_text())
        result["imports"] = probe.get("imports", {})
        result["observed_versions"] = probe.get("observed_versions", {})
        result["torch_cuda"] = probe.get("torch_cuda", {})
        result["errors"].extend(probe.get("errors", []))
    else:
        result["errors"].append(
            f"native import probe exited {import_run['exit_code']}; see logs/native_import_probe.log"
        )

    physx = Path(os.environ.get("HOME", "")) / ".sapien" / "physx" / "105.1-physx-5.3.1.patch0" / "libPhysXGpu_64.so"
    result["physx_gpu_asset"] = {
        "path": str(physx),
        "exists": physx.exists(),
        "resolved_path": str(physx.resolve()) if physx.exists() else None,
        "bytes": physx.stat().st_size if physx.exists() else None,
        "sha256": _sha256(physx) if physx.exists() else None,
    }

    smoke_command = [sys.executable, str(EVIDENCE / "state_smoke_worker.py")]
    smoke_run = _run(smoke_command)
    result["command_results"]["state_gpu_smoke_subprocess"] = smoke_run
    smoke_log = EVIDENCE / "logs" / "state_gpu_smoke_subprocess.log"
    smoke_log.write_text(smoke_run["stdout"] + smoke_run["stderr"])
    smoke_json = SMOKE_OUTPUT / "maniskill_state_gpu_smoke.json"
    if smoke_run["exit_code"] == 0 and smoke_json.exists():
        result["state_gpu_smoke"] = {
            "ok": True,
            "subprocess_exit_code": 0,
            "result": json.loads(smoke_json.read_text()),
        }
    else:
        result["state_gpu_smoke"] = {
            "ok": False,
            "subprocess_exit_code": smoke_run["exit_code"],
            "log": str(smoke_log),
        }
        result["errors"].append(
            f"state/GPU smoke subprocess exited {smoke_run['exit_code']}; see {smoke_log}"
        )

    observed = result["observed_versions"]
    torch_cuda = result.get("torch_cuda", {})
    observed["python"] = platform.python_version()
    observed["torch_cuda"] = torch_cuda.get("compiled_cuda")
    version_checks = {
        "python_exact": platform.python_version() == REQUESTED["python"],
        "torch_exact": observed.get("torch") == REQUESTED["torch"],
        "torch_cuda_exact": torch_cuda.get("compiled_cuda") == REQUESTED["torch_cuda"],
        "mani_skill_exact": observed.get("mani_skill") == REQUESTED["mani_skill"],
        "sapien_exact": observed.get("sapien") == REQUESTED["sapien"],
    }
    auxiliary_version_checks = {
        "opencv_python_recorded_version": (
            observed.get("opencv_python") == REQUESTED["opencv_python_auxiliary"]
        ),
    }
    result["version_checks"] = version_checks
    result["auxiliary_version_checks"] = auxiliary_version_checks
    result["pip_check_ok"] = result["command_results"]["pip_check"]["exit_code"] == 0
    smoke_result = result.get("state_gpu_smoke", {}).get("result", {})
    state_smoke = smoke_result.get("state_observation_smoke", {})
    vector_smoke = smoke_result.get("gpu_vector_smoke", {})
    result["gpu_import_smoke_ok"] = bool(
        result.get("state_gpu_smoke", {}).get("ok")
        and torch_cuda.get("available")
        and all(item.get("ok") for item in result["imports"].values())
        and state_smoke.get("success_field_present") is True
        and state_smoke.get("deterministic_reset") is True
        and vector_smoke.get("num_envs") == 8
        and vector_smoke.get("control_steps") == 40
        and result["physx_gpu_asset"]["exists"] is True
        and Path(str(result["process_environment"]["HOME"])).resolve() == (EVIDENCE / "home").resolve()
    )
    result["core_versions_exact"] = all(version_checks.values())
    result["core_environment_smoke_passed"] = bool(
        result["core_versions_exact"]
        and all(auxiliary_version_checks.values())
        and result["pip_check_ok"]
        and result["gpu_import_smoke_ok"]
    )
    # There is no historical full dependency lock, source commit, simulator
    # snapshot, or per-step trace. Exact requested core versions plus a passing
    # smoke therefore do not establish bit-exact legacy reproduction.
    result["legacy_exact_reproduction"] = False
    result["smoke_import"] = bool(all(item.get("ok") for item in result["imports"].values()))
    result["smoke_gpu"] = bool(result["gpu_import_smoke_ok"])
    result["pip_check"] = bool(result["pip_check_ok"])
    prior_3p = PROJECT / "outputs/actmask/milestone3p_gpu_benchmark/gpu_qualification_runtime.json"
    prior_3s = PROJECT / "outputs/actmask/milestone3s_paper_package/artifact_manifest.json"
    prior_3p_json = json.loads(prior_3p.read_text())
    prior_3s_json = json.loads(prior_3s.read_text())
    result["prior_environment_references"] = {
        "milestone3p_gpu_qualification_runtime": {
            **_file_record(prior_3p),
            "gpu": prior_3p_json.get("gpu", {}).get("name"),
            "driver": prior_3p_json.get("nvidia_smi", {}).get("stdout", "").split(",")[2].strip(),
            "platform": prior_3p_json.get("platform"),
            "torch": prior_3p_json.get("torch_version"),
        },
        "milestone3s_artifact_manifest": {
            **_file_record(prior_3s),
            "gpu": prior_3s_json.get("runtime", {}).get("gpu_type"),
            "driver": None,
            "platform": prior_3s_json.get("runtime", {}).get("platform"),
            "python": prior_3s_json.get("runtime", {}).get("python_version"),
            "environment_path": prior_3s_json.get("runtime", {}).get("conda_environment_path"),
            "package_symlink_repairs": len(prior_3s_json.get("runtime", {}).get("repaired_symlink_targets", [])),
        },
        "interpretation": (
            "These records describe different stages: milestone3p names an RTX 4090 D, while the later "
            "milestone3s packaging manifest names an A16. They are retained together rather than silently reconciled."
        ),
    }
    current_gpu_fields = [
        item.strip() for item in result["command_results"]["nvidia_smi"]["stdout"].strip().split(",")
    ]
    current_gpu = current_gpu_fields[1] if len(current_gpu_fields) >= 4 else None
    current_driver = current_gpu_fields[2] if len(current_gpu_fields) >= 4 else None
    result["current_hardware"] = {
        "gpu": current_gpu,
        "driver": current_driver,
        "platform": platform.platform(),
        "libc": platform.libc_ver(),
    }
    result["differences"] = [
        {
            "kind": "historical_hardware_records_differ_by_stage",
            "milestone3p": "NVIDIA GeForce RTX 4090 D",
            "milestone3s": "NVIDIA A16",
            "current": current_gpu,
        },
        {
            "kind": "driver_mismatch_vs_milestone3p",
            "milestone3p": "580.76.05",
            "current": current_driver,
        },
        {
            "kind": "libc_mismatch_vs_historical_platform_records",
            "historical": "glibc2.39",
            "current": "".join(platform.libc_ver()),
        },
        {
            "kind": "environment_construction_differs",
            "historical": "milestone3s records repaired cross-environment package symlinks",
            "current": "isolated conda prefix plus pip packages; only the pre-existing PhysX GPU asset is symlinked under audit HOME",
        },
    ] + [
        {"kind": "requested_core_version_mismatch", "check": name}
        for name, passed in version_checks.items() if not passed
    ] + [
        {"kind": "recorded_auxiliary_version_mismatch", "check": name}
        for name, passed in auxiliary_version_checks.items() if not passed
    ]
    result["legacy_provenance_gaps"] = [
        "No historical full dependency lock is present.",
        "No Git/source commit identifies the historical generator code.",
        "No simulator snapshot or per-step execution trace is present.",
    ]
    recorded_commands = []
    if COMMAND_HISTORY.exists():
        recorded_commands = json.loads(COMMAND_HISTORY.read_text()).get("commands", [])
    dynamic_commands = [
        {
            "name": name,
            "command": detail["command"],
            "shell_command": _shell(
                detail["command"],
                home=(EVIDENCE / "home") if name in {
                    "native_import_probe", "state_gpu_smoke_subprocess"
                } else None,
            ),
            "cwd": str(PROJECT),
            "exit_code": detail["exit_code"],
        }
        for name, detail in result["command_results"].items()
    ]
    result["commands"] = recorded_commands + dynamic_commands + [{
        "name": "collect_environment_evidence",
        "command": [sys.executable, str(EVIDENCE / "collect_and_run_smoke.py")],
        "shell_command": _shell(
            [sys.executable, str(EVIDENCE / "collect_and_run_smoke.py")],
            home=EVIDENCE / "home",
        ),
        "cwd": str(PROJECT),
        "exit_code": 0 if result["core_environment_smoke_passed"] else 1,
    }]
    result["locks"] = {
        name: _file_record(path) for name, path in lock_paths.items()
    }
    result["artifacts"] = {
        "local_sapien_wheel": _file_record(
            PROJECT / "outputs/actmask/milestone3p_gpu_benchmark/wheelhouse/sapien-3.0.3-cp311-cp311-manylinux_2_28_x86_64.whl"
        ),
        "local_maniskill_wheel": _file_record(
            PROJECT / "outputs/actmask/milestone3p_gpu_benchmark/wheelhouse/mani_skill-3.0.1-py3-none-any.whl"
        ),
        "physx_gpu_library": _file_record(
            PROJECT / ".maniskill_data/sapien_physx/105.1-physx-5.3.1.patch0/libPhysXGpu_64.so"
        ),
        "physx_archive": _file_record(
            PROJECT / ".maniskill_data/sapien_physx/105.1-physx-5.3.1.patch0/linux-so.zip"
        ),
        "audit_home_physx_symlink": {
            **_file_record(physx),
            "is_symlink": physx.is_symlink(),
            "resolved_path": str(physx.resolve()) if physx.exists() else None,
        },
    }
    log_paths = {
        "torch_install": EVIDENCE / "logs" / "pip_torch_2.6.0_cu124.log",
        "simulator_install": EVIDENCE / "logs" / "pip_maniskill_sapien.log",
        "native_import_probe": EVIDENCE / "logs" / "native_import_probe.log",
        "state_gpu_smoke_subprocess": EVIDENCE / "logs" / "state_gpu_smoke_subprocess.log",
        "state_gpu_smoke_json": SMOKE_OUTPUT / "maniskill_state_gpu_smoke.json",
        "base_install_report": EVIDENCE / "pip_base_install_report.json",
        "torch_install_report": EVIDENCE / "pip_torch_2.6.0_cu124_report.json",
        "simulator_install_report": EVIDENCE / "pip_maniskill_sapien_report.json",
    }
    result["logs"] = {name: _file_record(path) for name, path in log_paths.items()}
    result["blockers"] = list(result["errors"]) if result["errors"] else []
    result["status"] = "passed" if result["core_environment_smoke_passed"] else "environment_blocked"
    if result["status"] == "environment_blocked" and not result["blockers"]:
        result["blockers"].append(
            "Core environment qualification failed: inspect version_checks, pip_check, and state_gpu_smoke."
        )
    result["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    STATUS.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": result["status"],
        "observed_versions": observed,
        "version_checks": version_checks,
        "pip_check_ok": result["pip_check_ok"],
        "gpu_import_smoke_ok": result["gpu_import_smoke_ok"],
        "errors": result["errors"],
    }, indent=2, sort_keys=True))
    if not result["core_environment_smoke_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
