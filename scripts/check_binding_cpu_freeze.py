"""Verify the reviewed CPU transport source snapshot (LF-normalized SHA-256)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def check(root: Path):
    manifest = json.loads((root / "configs/binding_cpu_freeze.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "actmask-cpu-freeze-v1" or not manifest.get("files"):
        raise ValueError("invalid freeze manifest")
    changed = []
    for relative, expected in manifest["files"].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            changed.append(relative)
            continue
        actual = hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        if actual != expected:
            changed.append(relative)
    if changed:
        raise ValueError(f"CPU freeze drift; review and explicitly re-freeze: {changed}")
    return {"passed": True, "files": len(manifest["files"]), "scope": manifest["scope"]}


if __name__ == "__main__":
    print(json.dumps(check(Path(__file__).resolve().parents[1]), indent=2))
