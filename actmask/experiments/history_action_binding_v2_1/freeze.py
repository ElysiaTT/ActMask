"""Freeze scientific sources, derive the official seed, and verify boundaries."""

from __future__ import annotations

import json
import os
from pathlib import Path

from actmask.experiments.history_action_binding_v2_1.common import (
    ENV_ROOT,
    OUTPUT_ROOT,
    PROJECT_ROOT,
    PROTECTED_GLOBS,
    PROTECTED_PATHS,
    canonical_json,
    protocol_sha256,
    scientific_core_manifest,
    sha256_bytes,
    sha256_file,
    write_json,
)


def _record(path: Path) -> dict:
    if path.is_symlink():
        target = os.readlink(path)
        return {
            "kind": "symlink",
            "bytes": len(target.encode("utf-8")),
            "sha256": sha256_bytes(target.encode("utf-8")),
            "target": target,
        }
    return {"kind": "file", "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _project_protected_files() -> list[Path]:
    paths: set[Path] = set()
    for relative in PROTECTED_PATHS:
        path = PROJECT_ROOT / relative
        if path.is_file() or path.is_symlink():
            paths.add(path)
        elif path.is_dir():
            paths.update(item for item in path.rglob("*") if item.is_file() or item.is_symlink())
    for pattern in PROTECTED_GLOBS:
        for path in PROJECT_ROOT.glob(pattern):
            if path.is_file() or path.is_symlink():
                paths.add(path)
            elif path.is_dir():
                paths.update(item for item in path.rglob("*") if item.is_file() or item.is_symlink())
    return sorted(paths)


def protected_project_manifest() -> dict:
    files = {str(path.relative_to(PROJECT_ROOT)): _record(path) for path in _project_protected_files()}
    return {
        "root": str(PROJECT_ROOT),
        "files": files,
        "file_count": len(files),
        "total_bytes": sum(item["bytes"] for item in files.values()),
        "combined_sha256": sha256_bytes(canonical_json(files).encode("utf-8")),
    }


def _environment_files() -> list[Path]:
    if not ENV_ROOT.is_dir():
        raise FileNotFoundError(ENV_ROOT)
    files = []
    for path in ENV_ROOT.rglob("*"):
        if not (path.is_file() or path.is_symlink()):
            continue
        relative = path.relative_to(ENV_ROOT)
        if "__pycache__" in relative.parts or path.suffix in (".pyc", ".pyo"):
            continue
        files.append(path)
    return sorted(files)


def protected_environment_manifest() -> dict:
    files = {str(path.relative_to(ENV_ROOT)): _record(path) for path in _environment_files()}
    return {
        "root": str(ENV_ROOT),
        "excluded_ephemeral": ["**/__pycache__/**", "*.pyc", "*.pyo"],
        "files": files,
        "file_count": len(files),
        "total_bytes": sum(item["bytes"] for item in files.values()),
        "combined_sha256": sha256_bytes(canonical_json(files).encode("utf-8")),
    }


def verify_protected_boundaries() -> dict:
    project_path = OUTPUT_ROOT / "protected_project_before.json"
    environment_path = OUTPUT_ROOT / "protected_environment_before.json"
    if not project_path.exists() or not environment_path.exists():
        raise FileNotFoundError("protected before-manifests are missing")
    expected_project = json.loads(project_path.read_text())
    expected_environment = json.loads(environment_path.read_text())
    current_project = protected_project_manifest()
    current_environment = protected_environment_manifest()
    project_match = current_project == expected_project
    environment_match = current_environment == expected_environment
    return {
        "passed": bool(project_match and environment_match),
        "project_match": project_match,
        "environment_match": environment_match,
        "project_expected_combined_sha256": expected_project["combined_sha256"],
        "project_current_combined_sha256": current_project["combined_sha256"],
        "environment_expected_combined_sha256": expected_environment["combined_sha256"],
        "environment_current_combined_sha256": current_environment["combined_sha256"],
    }


def create_freeze() -> dict:
    freeze_path = OUTPUT_ROOT / "freeze_record.json"
    if freeze_path.exists():
        raise FileExistsError("freeze_record.json already exists; v2.1 cannot be refrozen")
    if (OUTPUT_ROOT / "official_symbolic_preflight").exists():
        raise RuntimeError("official output exists before freeze")
    prereg = PROJECT_ROOT / "docs/history_action_binding_v2_1_preregister.md"
    diagnosis = OUTPUT_ROOT / "development" / "v2_failure_diagnosis.json"
    development = OUTPUT_ROOT / "development" / "development_validation.json"
    for required in (prereg, diagnosis, development):
        if not required.exists():
            raise FileNotFoundError(required)
    development_record = json.loads(development.read_text())
    if not development_record["tasks"]["task_a_discrete_higher_order"]["all_development_seeds_passed"]:
        raise RuntimeError("Task A repair did not pass every declared development seed")

    project_manifest = protected_project_manifest()
    environment_manifest = protected_environment_manifest()
    write_json(OUTPUT_ROOT / "protected_project_before.json", project_manifest)
    write_json(OUTPUT_ROOT / "protected_environment_before.json", environment_manifest)
    core = scientific_core_manifest()
    basis = {
        "protocol_sha256": protocol_sha256(),
        "scientific_core_combined_sha256": core["combined_sha256"],
        "preregister_sha256": sha256_file(prereg),
        "v2_failure_diagnosis_sha256": sha256_file(diagnosis),
        "development_validation_sha256": sha256_file(development),
        "protected_project_combined_sha256": project_manifest["combined_sha256"],
        "protected_environment_combined_sha256": environment_manifest["combined_sha256"],
    }
    basis_sha256 = sha256_bytes(canonical_json(basis).encode("utf-8"))
    official_seed = int(basis_sha256[:16], 16) % 2_000_000_000 + 1
    record = {
        "status": "FROZEN_BEFORE_OFFICIAL_PREFLIGHT",
        "official_preflight_started": False,
        "official_preflight_results_seen": False,
        "scientific_core_manifest": core,
        "preregister_sha256": basis["preregister_sha256"],
        "protocol_sha256": basis["protocol_sha256"],
        "seed_derivation_basis": basis,
        "seed_derivation_basis_sha256": basis_sha256,
        "official_seed_derivation": "int(first_16_hex(seed_derivation_basis_sha256), 16) % 2000000000 + 1",
        "official_seed": official_seed,
        "development_seed_overlap": official_seed in set(json.loads(development.read_text()).get("development_seeds", [])),
        "protected_project_manifest": "protected_project_before.json",
        "protected_environment_manifest": "protected_environment_before.json",
    }
    write_json(freeze_path, record)
    (OUTPUT_ROOT / "freeze_record.sha256").write_text(
        sha256_file(freeze_path) + "  freeze_record.json\n"
    )
    return record


def main() -> None:
    print(json.dumps(create_freeze(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
