"""Freeze provenance and deterministically reproduce a small 3Q subset."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from actmask.data.milestone3q_signed import generate
from actmask.experiments.milestone3q_signed_baselines import run as run_baselines


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(output_root: str | Path) -> dict:
    project = Path(__file__).resolve().parents[2]
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    frozen_root = project / "outputs" / "actmask" / "milestone3q_signed_dynamics" / "full_four_task"
    frozen_files = sorted(
        path for path in frozen_root.iterdir()
        if path.is_file() and path.suffix in {".json", ".jsonl", ".npz"}
    )
    frozen_sources = [
        project / "actmask/data/milestone3q_signed.py",
        project / "actmask/data/maniskill_3q_tasks.py",
        project / "actmask/experiments/milestone3q_signed_baselines.py",
        project / "tests/test_milestone3q_signed_dynamics.py",
        project / "docs/milestone3q_plan.md",
        project / "docs/milestone3q_results.md",
    ]
    provenance = {
        str(path.relative_to(project)): {"sha256": _sha256(path), "bytes": path.stat().st_size}
        for path in [*frozen_files, *frozen_sources]
    }
    frozen_summary = json.loads((frozen_root / "signed_generation_summary.json").read_text(encoding="utf-8"))
    if frozen_summary["tasks"]["fixed_phase_rotating_capture_window"]["action_horizon"] != 13:
        raise AssertionError("frozen 3Q rotating horizon is not 13")

    subset = output / "phase0_subset"
    subset_summary = generate(subset, worlds_per_regime=4)
    baseline = run_baselines(subset, (17, 29, 43), "phase0_reproduction")
    result = baseline["mean_pair_order_accuracy"]
    correct = result["id_correct_history"]
    reversed_history = result["id_reversed_history"]
    expected_chance = ("static_mlp", "unordered_history_mlp")
    if any(correct[name] != 0.5 for name in expected_chance):
        raise AssertionError(f"3Q subset static/unordered reproduction differs: {correct}")
    if correct["signed_finite_difference"] != 1.0 or correct["ordered_temporal_mlp"] != 1.0:
        raise AssertionError(f"3Q subset signed reproduction differs: {correct}")
    if any(reversed_history[name] != 0.0 for name in ("signed_finite_difference", "ordered_temporal_mlp")):
        raise AssertionError(f"3Q subset reversal reproduction differs: {reversed_history}")
    report = dict(
        schema_version=1,
        frozen_3q_root=str(frozen_root),
        provenance=provenance,
        rotating_action_horizon=13,
        subset_generation=subset_summary,
        subset_baseline=baseline,
        status="pass",
    )
    (output / "frozen_3q_reproduction.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    project = Path(__file__).resolve().parents[2]
    print(json.dumps(run(project / "outputs" / "actmask" / "milestone3r_robust_signed_dynamics"), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
