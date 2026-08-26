import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_world_cluster_bootstrap_is_deterministic_and_world_keyed(tmp_path):
    script = ROOT / "scripts/build_icra2027_state_world_bootstrap.py"
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    for output in (first, second):
        subprocess.run(
            [sys.executable, str(script), "--output", str(output)],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

    report = json.loads(first.read_text(encoding="utf-8"))
    assert report["pass"] is True
    assert report["test_units"] == {
        "base_worlds": 300,
        "candidate_pairs": 6000,
        "candidate_pairs_per_world": 20,
        "family_mechanism_cells": 5,
    }
    assert report["pair_order"]["gru"] == 1.0
    assert report["pair_order"]["analytic"] == 0.6
    assert report["pair_order"]["delta"] == 0.4
    assert report["pair_order"]["world_cluster_ci95"] == [
        0.37666666666666665,
        0.4216666666666667,
    ]
    assert _sha256(first) == _sha256(second)
