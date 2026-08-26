import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_icra2027_manuscript_passes_submission_audit(tmp_path):
    command = [
        sys.executable,
        str(ROOT / "scripts/verify_icra2027_manuscript.py"),
        "--pdf",
        str(ROOT / "paper_icra2027/main.pdf"),
        "--report-dir",
        str(tmp_path),
    ]
    subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    report = json.loads((tmp_path / "manuscript_audit.json").read_text())

    assert report["passed"] is True
    assert report["pdf"]["pages"] <= 8
    assert report["pdf"]["page_size"].startswith("612 x 792 pts")
    assert all(row["status"] == "PASS" for row in report["checks"])
    assert (tmp_path / "manuscript_audit.md").is_file()
