import hashlib
import json
import subprocess
import sys
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_visual_history_figure_is_audited_and_deterministic(tmp_path):
    output = tmp_path / "visual_matched_histories.png"
    command = [
        sys.executable,
        str(ROOT / "scripts/build_icra2027_visual_history_figure.py"),
        "--output",
        str(output),
    ]
    subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    manifest = json.loads(output.with_suffix(".manifest.json").read_text())
    first_hash = sha256(output)

    assert manifest["pass"] is True
    assert len(manifest["sources"]) == 9
    assert len(manifest["audits"]) == 3
    assert all(row["exact_final_two_rgb"] for row in manifest["audits"])
    assert all(row["exact_final_two_depth"] for row in manifest["audits"])
    assert all(row["outcomes"] == [True, False] for row in manifest["audits"])
    with Image.open(output) as image:
        assert image.size == (1832, 554)

    subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    assert sha256(output) == first_hash
