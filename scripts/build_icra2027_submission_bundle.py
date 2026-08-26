#!/usr/bin/env python3
"""Build a deterministic anonymous ICRA 2027 paper and artifact bundle."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
DEFAULT_OUTPUT = DIST / "icra2027_submission_bundle"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_claim_sources() -> dict:
    path = ROOT / "scripts/build_icra2027_claim_package.py"
    spec = importlib.util.spec_from_file_location("icra_claim_builder_for_bundle", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.SOURCES


def copy(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def add_tree(source: Path, destination: Path) -> None:
    for path in sorted(source.rglob("*")):
        if path.is_file():
            copy(path, destination / path.relative_to(source))


def write_zip(directory: Path, archive: Path) -> None:
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as handle:
        for path in sorted(directory.rglob("*")):
            if not path.is_file():
                continue
            relative = Path(directory.name) / path.relative_to(directory)
            info = zipfile.ZipInfo(str(relative), date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (0o644 & 0xFFFF) << 16
            handle.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def build(output: Path, force: bool) -> dict:
    output = output.resolve()
    expected_parent = DIST.resolve()
    if output.parent != expected_parent or not output.name.startswith("icra2027_submission_bundle"):
        raise ValueError(f"refusing unsafe output path: {output}")
    if output.exists():
        if not force:
            raise FileExistsError(f"output exists; pass --force to replace: {output}")
        shutil.rmtree(output)
    archive = output.with_suffix(".zip")
    if archive.exists():
        if not force:
            raise FileExistsError(f"archive exists; pass --force to replace: {archive}")
        archive.unlink()
    output.mkdir(parents=True)

    paper_files = {
        "main.pdf": "paper.pdf",
        "main.tex": "main.tex",
        "references.bib": "references.bib",
        "ieeeconf.cls": "ieeeconf.cls",
        "IEEEtran.bst": "IEEEtran.bst",
        "figures/state_audit_construction.png": "figures/state_audit_construction.png",
        "figures/state_audit_construction.manifest.json": "figures/state_audit_construction.manifest.json",
        "figures/visual_matched_histories.png": "figures/visual_matched_histories.png",
        "figures/visual_matched_histories.manifest.json": "figures/visual_matched_histories.manifest.json",
    }
    for source, destination in paper_files.items():
        copy(ROOT / "paper_icra2027" / source, output / "paper" / destination)
    add_tree(ROOT / "paper_icra2027/generated", output / "paper/generated")

    artifact = output / "artifact"
    copy(ROOT / "submission/icra2027_artifact_README.md", artifact / "README.md")
    for name in (
        "build_icra2027_claim_package.py",
        "run_icra2027_wav_external_audit.py",
        "verify_icra2027_wav_external_audit.py",
    ):
        copy(ROOT / "scripts" / name, artifact / "scripts" / name)
    for name in (
        "test_icra2027_claim_package.py",
        "test_icra2027_wav_external_audit.py",
    ):
        copy(ROOT / "tests" / name, artifact / "tests" / name)
    for name in (
        "icra2027_wav_external_audit_prereg.md",
        "icra2027_wav_external_audit_results.md",
    ):
        copy(ROOT / "docs" / name, artifact / "docs" / name)
    for frozen in load_claim_sources().values():
        relative = Path(frozen["path"])
        copy(ROOT / relative, artifact / relative)
    copy(
        ROOT / "outputs/actmask/milestone3r_nl_v2_rankfix/raw_scores.npz",
        artifact / "outputs/actmask/milestone3r_nl_v2_rankfix/raw_scores.npz",
    )
    copy(
        ROOT / "outputs/actmask/icra2027_wav_external_audit/verification.json",
        artifact / "outputs/actmask/icra2027_wav_external_audit/verification.json",
    )
    add_tree(
        ROOT / "outputs/actmask/icra2027_submission",
        artifact / "outputs/actmask/icra2027_submission",
    )
    copy(ROOT / "submission/verify_bundle.py", output / "verify_bundle.py")

    readme = (
        "# ICRA 2027 anonymous submission bundle\n\n"
        "Submit `paper/paper.pdf` to PaperPlaza. The `paper/` directory also "
        "contains its minimal compilable source. The `artifact/` directory is a "
        "compact anonymous verification artifact; see its README.\n\n"
        "Run `python verify_bundle.py` before upload. The bundle is locally "
        "prepared only; this script does not upload or contact any submission system.\n"
    )
    (output / "README.md").write_text(readme, encoding="utf-8")

    files = [
        {
            "path": str(path.relative_to(output)),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "bundle_manifest.json"
    ]
    manifest = {
        "schema": "actmask-icra2027-anonymous-submission-bundle-v1",
        "paper": "paper/paper.pdf",
        "artifact": "artifact/README.md",
        "files": files,
    }
    (output / "bundle_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_zip(output, archive)
    return {
        "output": str(output),
        "archive": str(archive),
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": sha256(archive),
        "files": len(files) + 1,
        "paper_sha256": sha256(output / "paper/paper.pdf"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    report = build(args.output, args.force)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
