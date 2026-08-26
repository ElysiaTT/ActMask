import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_state_audit_figure_is_source_checked_and_deterministic(tmp_path):
    script = ROOT / "scripts/build_icra2027_state_audit_figure.py"
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    for output in (first, second):
        subprocess.run(
            [sys.executable, str(script), "--output", str(output)],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    manifest = json.loads(first.with_suffix(".manifest.json").read_text())
    assert manifest["pass"] is True
    assert len(manifest["sources"]) == 3
    assert manifest["figure"]["width"] == 2240
    assert manifest["figure"]["height"] == 928
    assert _sha256(first) == _sha256(second)
