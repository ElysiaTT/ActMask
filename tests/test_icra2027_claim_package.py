import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_icra2027_claim_package_is_audited_and_deterministic(tmp_path):
    script = ROOT / "scripts/build_icra2027_claim_package.py"
    command = [sys.executable, str(script), "--output-dir", str(tmp_path)]
    subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    ledger = json.loads((tmp_path / "claim_ledger.json").read_text())
    visual = json.loads((tmp_path / "tables/visual_admission.json").read_text())
    first_hashes = {
        str(path.relative_to(tmp_path)): sha256(path)
        for path in tmp_path.rglob("*")
        if path.is_file()
    }

    assert manifest["pass"] is True
    assert len(manifest["sources"]) == 16
    assert all(row["status"] == "PASS" for row in manifest["checks"])
    assert len(manifest["checks"]) == 36
    assert len(ledger["claims"]) == 10
    assert ledger["claims"][1]["value"] == {
        "base_worlds": 300,
        "ci95": [0.37666666666666665, 0.4216666666666667],
        "delta": 0.4,
    }
    assert visual["confidence_intervals_reported"] is False
    assert [row["centroid_velocity"] for row in visual["rows"]] == [1.0, 1.0, 1.0]
    assert (tmp_path / "tables/state_baselines.tex").is_file()
    assert (tmp_path / "tables/state_family_mechanism.tex").is_file()
    assert (tmp_path / "tables/state_ranking.tex").is_file()
    wav = json.loads((tmp_path / "tables/wav_external_audit.json").read_text())
    assert wav["metric"] == "official dynamic accuracy"
    assert len(wav["rows"]) == 5
    assert min(
        row["paired_mean"] - row["strongest_single_mean"] for row in wav["rows"]
    ) > 0.459
    assert (tmp_path / "tables/wav_external_audit.tex").is_file()
    admission = json.loads((tmp_path / "tables/admission_decision.json").read_text())
    assert admission["rows"][0]["decision"] == "Admit controlled comparison"
    assert admission["rows"][1]["decision"] == "Admit transition evidence"
    assert admission["rows"][2]["decision"] == "Reject method claim"
    assert admission["rows"][3]["decision"] == "Reject: no re-execution"
    assert (tmp_path / "tables/admission_decision.tex").is_file()

    subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    second_hashes = {
        str(path.relative_to(tmp_path)): sha256(path)
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    assert second_hashes == first_hashes
