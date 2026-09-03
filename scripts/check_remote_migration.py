"""Fail-closed, non-mutating audit of files intended for remote transport."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


MANIFEST_SCHEMA = "actmask-binding-remote-manifest-v1"
TEXT_SUFFIXES = {".json", ".md", ".py", ".sh", ".txt", ".toml", ".yaml", ".yml"}
PRIVATE_KEY_MARKERS = (
    b"BEGIN OPENSSH " + b"PRIVATE KEY",
    b"BEGIN RSA " + b"PRIVATE KEY",
    b"BEGIN EC " + b"PRIVATE KEY",
    b"AWS_SECRET_" + b"ACCESS_KEY=",
)
SENSITIVE_NAMES = {
    ".env",
    "id_rsa",
    "id_ed25519",
    "credentials",
    "credentials.json",
}
ACTIVE_ABSOLUTE_PATH_MARKERS = (
    b"C:" + b"\\Users\\",
    b"D:" + b"\\study\\",
    b"/data/" + b"project/",
    b"/home/" + b"tzh/",
)


def _run(command: list[str], *, root: Path, timeout: int = 180) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"returncode": None, "stdout": "", "stderr": str(error)}
    return {
        "returncode": completed.returncode,
        "stdout": (completed.stdout or "").strip(),
        "stderr": (completed.stderr or "").strip(),
    }


def _git(root: Path, *arguments: str, timeout: int = 60) -> dict[str, Any]:
    return _run(["git", *arguments], root=root, timeout=timeout)


def _manifest_paths(manifest: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    for key in (
        "active_code",
        "active_scripts",
        "active_configs",
        "active_docs",
        "historical_provenance",
        "curated_evidence",
    ):
        paths.extend(str(value) for value in manifest.get(key, []))
    return sorted(set(paths))


def _candidate_paths(root: Path) -> list[str]:
    result = _git(
        root,
        "ls-files",
        "--cached",
        "--others",
        "--exclude-standard",
        "-z",
    )
    if result["returncode"] != 0:
        raise RuntimeError(result["stderr"] or "git ls-files failed")
    return sorted(value for value in result["stdout"].split("\0") if value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_ignored(root: Path, relative: str) -> bool:
    return _git(root, "check-ignore", "--no-index", "-q", "--", relative)["returncode"] == 0


def _scan_sensitive(root: Path, paths: Iterable[str]) -> dict[str, list[str]]:
    suspicious_names: list[str] = []
    marker_hits: list[str] = []
    for relative in paths:
        path = root / relative
        if not path.is_file():
            continue
        lowered = path.name.lower()
        if (
            lowered in SENSITIVE_NAMES
            or lowered.endswith((".pem", ".key"))
            or lowered.startswith(("id_rsa", "id_ed25519"))
        ):
            suspicious_names.append(relative)
        if path.suffix.lower() not in TEXT_SUFFIXES or path.stat().st_size > 10 * 2**20:
            continue
        payload = path.read_bytes()
        if any(marker in payload for marker in PRIVATE_KEY_MARKERS):
            marker_hits.append(relative)
    return {
        "suspicious_filenames": suspicious_names,
        "private_key_marker_hits": marker_hits,
    }


def _active_path_leaks(root: Path, paths: Iterable[str]) -> dict[str, list[str]]:
    leaks: dict[str, list[str]] = {}
    for relative in paths:
        path = root / relative
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        payload = path.read_bytes()
        names = [
            marker.decode("ascii", errors="replace")
            for marker in ACTIVE_ABSOLUTE_PATH_MARKERS
            if marker in payload
        ]
        if names:
            leaks[relative] = names
    return leaks


def _validate_json_files(root: Path, paths: Iterable[str]) -> dict[str, str]:
    failures: dict[str, str] = {}
    for relative in paths:
        path = root / relative
        if path.suffix.lower() != ".json" or not path.is_file():
            continue
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            failures[relative] = str(error)
    return failures


def audit(root: Path, manifest_path: Path, *, run_tests: bool) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise ValueError("unsupported remote manifest schema")
    required = _manifest_paths(manifest)
    portability_paths: list[str] = []
    for key in ("active_code", "active_scripts", "active_configs", "active_docs"):
        portability_paths.extend(str(value) for value in manifest.get(key, []))
    missing = [relative for relative in required if not (root / relative).is_file()]
    candidates = _candidate_paths(root)
    candidate_set = set(candidates)
    ignored = {
        relative: _is_ignored(root, relative)
        for relative in manifest["generated_paths_must_be_ignored"]
    }
    maximum_bytes = int(float(manifest["maximum_unignored_file_mib"]) * 2**20)
    large_files = {
        relative: (root / relative).stat().st_size
        for relative in candidates
        if (root / relative).is_file() and (root / relative).stat().st_size > maximum_bytes
    }
    sensitive = _scan_sensitive(root, candidates)
    active_portability = _active_path_leaks(root, portability_paths)
    json_failures = _validate_json_files(root, required)
    shell_files = [relative for relative in required if relative.endswith(".sh")]
    shell_line_endings = {
        relative: b"\r\n" not in (root / relative).read_bytes()
        for relative in shell_files
        if (root / relative).is_file()
    }
    # Windows often resolves bash.exe to the WSL launcher, which cannot provide
    # a meaningful native-Linux syntax check from this working tree.
    bash = shutil.which("bash") if platform.system() == "Linux" else None
    bash_syntax = (
        _run([bash, "-n", *shell_files], root=root)
        if bash is not None and shell_files
        else {"returncode": None, "stdout": "", "stderr": "bash unavailable; syntax check deferred"}
    )
    tests = (
        _run([sys.executable, *manifest["focused_test"]], root=root)
        if run_tests
        else {"returncode": None, "stdout": "", "stderr": "focused tests skipped"}
    )
    diff_check = _git(root, "diff", "--check")
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    origin = _git(root, "remote", "get-url", "origin")
    head = _git(root, "rev-parse", "HEAD")
    branch = _git(root, "branch", "--show-current")
    untracked_manifest_paths = [relative for relative in required if relative not in candidate_set]
    tracked = set(
        value
        for value in _git(root, "ls-files", "-z")["stdout"].split("\0")
        if value
    )
    manifest_files_untracked = [relative for relative in required if relative not in tracked]
    file_sha256 = {
        relative: _sha256(root / relative)
        for relative in required
        if (root / relative).is_file()
    }
    planned_presence = {
        relative: (root / relative).is_file()
        for relative in manifest.get("planned_not_implemented", [])
    }

    layout_checks = {
        "manifest_files_exist": not missing,
        "generated_paths_ignored": all(ignored.values()),
        "no_large_unignored_files": not large_files,
        "no_sensitive_files_or_key_markers": not any(sensitive.values()),
        "active_files_have_no_machine_absolute_paths": not active_portability,
        "required_json_is_valid": not json_failures,
        "remote_shell_files_use_lf": all(shell_line_endings.values()),
        "remote_shell_syntax": bash_syntax["returncode"] in (0, None),
        "tracked_diff_has_no_whitespace_errors": diff_check["returncode"] == 0,
        "focused_tests": tests["returncode"] in (0, None),
    }
    transfer_checks = {
        "origin_configured": origin["returncode"] == 0 and bool(origin["stdout"]),
        "head_exists": head["returncode"] == 0 and bool(head["stdout"]),
        "manifest_files_tracked": not manifest_files_untracked,
        "worktree_clean": status["returncode"] == 0 and not status["stdout"],
    }
    layout_ready = all(layout_checks.values())
    transfer_ready = layout_ready and all(transfer_checks.values())
    return {
        "schema_version": "actmask-binding-remote-migration-audit-v1",
        "utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "manifest": str(manifest_path),
        "layout": {
            "decision": "REMOTE_LAYOUT_GO" if layout_ready else "REMOTE_LAYOUT_NO_GO",
            "checks": layout_checks,
            "missing_files": missing,
            "generated_ignore_checks": ignored,
            "large_unignored_files": large_files,
            "sensitive_scan": sensitive,
            "active_absolute_path_leaks": active_portability,
            "json_failures": json_failures,
            "shell_lf_checks": shell_line_endings,
            "bash_syntax": bash_syntax,
            "diff_check": diff_check,
            "focused_tests": tests,
        },
        "git_transfer": {
            "decision": "GIT_TRANSFER_GO" if transfer_ready else "GIT_TRANSFER_NO_GO",
            "checks": transfer_checks,
            "branch": branch["stdout"],
            "head": head["stdout"],
            "origin": origin["stdout"],
            "status": status["stdout"].splitlines(),
            "manifest_files_untracked": manifest_files_untracked,
            "required_files_absent_from_git_candidates": untracked_manifest_paths,
        },
        "file_sha256": file_sha256,
        "planned_gpu_module_presence": planned_presence,
        "overall_decision": "REMOTE_TRANSFER_GO" if transfer_ready else "REMOTE_TRANSFER_NO_GO",
        "scope_note": (
            "The audit never stages, commits, pushes, installs dependencies, copies credentials, "
            "downloads datasets, or starts GPU work."
        ),
    }


def render_markdown(result: dict[str, Any]) -> str:
    layout = result["layout"]
    transfer = result["git_transfer"]
    failing_layout = [name for name, passed in layout["checks"].items() if not passed]
    failing_transfer = [name for name, passed in transfer["checks"].items() if not passed]
    tests = layout["focused_tests"]
    test_text = tests["stdout"].replace("\n", " ") if tests["stdout"] else tests["stderr"]
    return f"""# Remote migration audit

Generated: {result['utc']}
Layout: **{layout['decision']}**
Git transport: **{transfer['decision']}**
Overall: **{result['overall_decision']}**

## Evidence

- Required-file hashes: `{len(result['file_sha256'])}` recorded.
- Layout failures: `{', '.join(failing_layout) or 'none'}`.
- Transfer failures: `{', '.join(failing_transfer) or 'none'}`.
- Focused tests: `{test_text or 'not run'}`.
- Large unignored files over the manifest limit: `{len(layout['large_unignored_files'])}`.
- Sensitive filename/private-key marker hits: `{sum(len(v) for v in layout['sensitive_scan'].values())}`.
- Manifest files not yet tracked by Git: `{len(transfer['manifest_files_untracked'])}`.

## Interpretation

`REMOTE_LAYOUT_GO` means the selected source, evidence, ignore rules, and remote
entrypoints are coherent. `GIT_TRANSFER_GO` additionally requires every
manifest file to be committed and the worktree to be clean. This report does
not authorize GPU execution; the separate GPU readiness gate remains binding.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("configs/binding_remote_manifest.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument("--run-tests", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    manifest_path = args.manifest
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path
    result = audit(root, manifest_path.resolve(), run_tests=args.run_tests)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.markdown_output is not None:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(render_markdown(result), encoding="utf-8")
    print(
        json.dumps(
            {
                "layout": result["layout"]["decision"],
                "git_transfer": result["git_transfer"]["decision"],
                "overall": result["overall_decision"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
