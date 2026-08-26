#!/usr/bin/env python3
"""Verify hashes and anonymity-sensitive text in the submission bundle."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "bundle_manifest.json"
TEXT_SUFFIXES = {".bib", ".json", ".md", ".py", ".tex", ".txt"}
BANNED_PATTERNS = {
    "absolute_data_path": re.compile(r"(?<![A-Za-z0-9_.-])/data/", re.IGNORECASE),
    "absolute_home_path": re.compile(r"(?<![A-Za-z0-9_.-])/home/", re.IGNORECASE),
    "personal_site": re.compile(r"elysiatt", re.IGNORECASE),
    "local_file_uri": re.compile(r"file://", re.IGNORECASE),
    "local_account": re.compile(r"\btzh\b", re.IGNORECASE),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    expected = {row["path"]: row for row in manifest["files"]}
    observed = {
        str(path.relative_to(ROOT)): path
        for path in ROOT.rglob("*")
        if path.is_file() and path != MANIFEST
    }
    missing = sorted(set(expected) - set(observed))
    extra = sorted(set(observed) - set(expected))
    mismatched = sorted(
        name for name in set(expected) & set(observed)
        if observed[name].stat().st_size != expected[name]["bytes"]
        or sha256(observed[name]) != expected[name]["sha256"]
    )
    leaks = []
    email = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.IGNORECASE)
    for name, path in observed.items():
        if path.resolve() == Path(__file__).resolve() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for label, pattern in BANNED_PATTERNS.items():
            if pattern.search(text):
                leaks.append({"path": name, "token": label})
        for match in email.findall(text):
            leaks.append({"path": name, "token": match})
    result = {
        "passed": not missing and not extra and not mismatched and not leaks,
        "files": len(observed),
        "missing": missing,
        "extra": extra,
        "mismatched": mismatched,
        "anonymity_leaks": leaks,
    }
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
