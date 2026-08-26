"""Versioned A4 wrapper for all-validation-promising temporal diagnostics.

The established temporal implementation already defines the frozen thirteen
views and validation-promising eligibility rule.  This wrapper runs it in an
ephemeral location, then stores its complete report inside an immutable
final-v2 envelope with corrected-NDCG provenance.  Consequently no unversioned
or legacy-NDCG temporal JSON becomes a persistent Stage-A artifact.
"""

from __future__ import annotations

import argparse
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from actmask.experiments.milestone2c import PROJECT_ROOT
from actmask.experiments.milestone2e import DEFAULT_CONFIG, NDCG_DEFINITION, RANKING_METRIC_SCHEMA, _load_config
from actmask.experiments.milestone2e_final_cache import code_provenance_for_files
from actmask.experiments.milestone2e_final_v2 import (
    FINAL_ROOT,
    LEGACY_ROOT,
    _timestamp,
    _write_json_new,
    verify_inputs as verify_ranking_inputs,
)
from actmask.experiments.milestone2e_temporal_diagnostics import (
    TEMPORAL_SHORTCUT_VIEWS,
    run as run_established_temporal,
)
from actmask.experiments.milestone2e_full import _json_digest


TEMPORAL_SCHEMA_VERSION = "milestone2e-final-v2-temporal-diagnostics-v1"
DEFAULT_OUTPUT_PATH = FINAL_ROOT / "temporal" / "all_validation_promising.json"


def _source_files() -> list[Path]:
    return [
        Path(__file__),
        Path(__file__).with_name("milestone2e_temporal_diagnostics.py"),
        Path(__file__).with_name("milestone2e_final_v2.py"),
        Path(__file__).parent.parent / "data" / "milestone2b_dataset.py",
        Path(__file__).parent.parent / "models" / "milestone2e.py",
    ]


def _validate_established_report(report: Mapping[str, Any]) -> None:
    if report.get("status") != "completed_frozen_c20":
        raise ValueError(f"all-validation-promising temporal diagnostic did not complete: {report.get('status')!r}")
    if report.get("all_validation_promising") is not True:
        raise ValueError("temporal diagnostic was not run in all-validation-promising mode")
    phase4 = report.get("phase4_temporal_shortcut")
    if not isinstance(phase4, Mapping) or tuple(phase4.get("view_order", ())) != TEMPORAL_SHORTCUT_VIEWS:
        raise ValueError("temporal diagnostic does not contain all frozen temporal/shortcut views")
    variants = phase4.get("per_variant")
    if not isinstance(variants, Mapping) or not variants:
        raise ValueError("temporal diagnostic lacks validation-defined per-variant reports")


def verify_inputs(
    config_path: Path = DEFAULT_CONFIG,
    *,
    legacy_root: Path = LEGACY_ROOT,
    final_root: Path = FINAL_ROOT,
) -> dict[str, Any]:
    """Read-only provenance check before expensive A4 execution."""

    ranking = verify_ranking_inputs(config_path, legacy_root=legacy_root, final_root=final_root)
    config = _load_config(Path(config_path))
    if str(config["experiment"]["device"]).lower() != "cpu":
        raise ValueError("frozen temporal diagnostic must remain CPU-only")
    return {
        "status": "verified",
        "cpu_only": True,
        "gpu_used": False,
        "config_digest": _json_digest(config),
        "ranking_input_verification": ranking,
        "temporal_views": list(TEMPORAL_SHORTCUT_VIEWS),
    }


def run(
    config_path: Path = DEFAULT_CONFIG,
    *,
    legacy_root: Path = LEGACY_ROOT,
    final_root: Path = FINAL_ROOT,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Run A4 once and emit an immutable corrected-schema final-v2 envelope."""

    final_root = Path(final_root)
    output_path = final_root / "temporal" / "all_validation_promising.json" if output_path is None else Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite final-v2 temporal artifact: {output_path}")
    verification = verify_inputs(config_path, legacy_root=legacy_root, final_root=final_root)
    config = _load_config(Path(config_path))
    with tempfile.TemporaryDirectory(prefix="actmask_m2e_temporal_") as directory:
        scratch_path = Path(directory) / "established_temporal.json"
        established = run_established_temporal(
            config_path=Path(config_path),
            output_path=scratch_path,
            output_root=Path(legacy_root),
            all_validation_promising=True,
        )
        _validate_established_report(established)
        if not scratch_path.exists():
            raise RuntimeError("established temporal diagnostic returned without its ephemeral artifact")
    report = {
        "artifact_schema": TEMPORAL_SCHEMA_VERSION,
        "stage": "A4_all_validation_promising_temporal_shortcut_diagnostics",
        "completed_at_utc": _timestamp(),
        "cpu_only": True,
        "gpu_used": False,
        "config_digest": _json_digest(config),
        "metric_definition_version": RANKING_METRIC_SCHEMA,
        "ndcg_definition": NDCG_DEFINITION,
        "validation_selected_arm": verification["ranking_input_verification"]["validation_selected_arm"],
        "source_code_provenance": code_provenance_for_files(_source_files(), repository_root=Path(PROJECT_ROOT)),
        "established_temporal_report": established,
        "fair_baseline_scope_note": (
            "Temporal acceptance is evaluated from learned-only causal and shortcut controls. "
            "The established report's analytic score is retained for context only; the Stage-A fair comparator "
            "for ranking advantage is the separately frozen ranking-validation selection in A2/A5."
        ),
    }
    _write_json_new(output_path, report)
    return {"status": "complete", "output_path": str(output_path), "cpu_only": True, "gpu_used": False}


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-inputs", action="store_true")
    parser.add_argument("--legacy-root", type=Path, default=LEGACY_ROOT)
    parser.add_argument("--output-root", type=Path, default=FINAL_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    arguments = parser.parse_args(argv)
    if arguments.verify_inputs:
        report = verify_inputs(legacy_root=arguments.legacy_root, final_root=arguments.output_root)
        print(report["status"])
        return report
    report = run(
        legacy_root=arguments.legacy_root,
        final_root=arguments.output_root,
        output_path=arguments.output,
    )
    print(report["status"])
    return report


if __name__ == "__main__":
    main()
