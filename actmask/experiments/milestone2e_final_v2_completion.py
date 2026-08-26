"""CPU-only Phase A6 completion extensions for corrected Milestone 2E-Final.

This is a final-v2 consumer, not a rerun of the legacy completion module.  It
requires the corrected A2/A3/A5 artifacts, reads the frozen legacy checkpoints
only, and writes a single immutable final-v2 artifact.  Ranking-level point
permutation evidence is referenced from A3, where it was regenerated with the
corrected utility-gain NDCG; the legacy extension's old-NDCG ranking files are
never read.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from actmask.experiments.milestone2c import PROJECT_ROOT, _loader
from actmask.experiments.milestone2e import DEFAULT_CONFIG, RANKING_METRIC_SCHEMA, _load_config
from actmask.experiments.milestone2e_completion_extensions import (
    STATIC_VELOCITY_THRESHOLD,
    _latency_evaluation,
    _mask_evaluation,
)
from actmask.experiments.milestone2e_final_cache import code_provenance_for_files
from actmask.experiments.milestone2e_final_v2 import (
    ARTIFACT_SCHEMA_VERSION,
    FINAL_ROOT,
    LEGACY_ROOT,
    _load_primary_models,
    _read_mapping,
    _sha256_file,
    _timestamp,
    _write_json_new,
    build_final_v2_datasets,
    verify_inputs as verify_ranking_inputs,
)
from actmask.experiments.milestone2e_full import _json_digest


COMPLETION_SCHEMA_VERSION = "milestone2e-final-v2-completion-extensions-v1"
DEFAULT_OUTPUT_PATH = FINAL_ROOT / "completion" / "completion_extensions.json"
POINT_PERMUTATION_VIEW = "group_shared_all_frame_point_permutation"


def _source_files() -> list[Path]:
    return [
        Path(__file__),
        Path(__file__).with_name("milestone2e_final_v2.py"),
        Path(__file__).with_name("milestone2e_completion_extensions.py"),
        Path(__file__).parent.parent / "data" / "milestone2e_final_v2_corruptions.py",
        Path(__file__).parent.parent / "models" / "milestone2e.py",
    ]


def _summary(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, Any]:
    raw_values = [row[field] for row in rows]
    if any(value is None for value in raw_values):
        return {"mean": None, "std": None, "values": list(raw_values)}
    values = np.asarray([float(value) for value in raw_values], dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1 if len(values) > 1 else 0)),
        "values": values.tolist(),
    }


def _correspondence_one_seed(model: Any, dataset: Dataset[Any], *, batch_size: int) -> dict[str, Any]:
    """Measure observable source association without giving a fair model hidden state."""

    counts: defaultdict[str, int] = defaultdict(int)
    errors: list[float] = []
    with torch.no_grad():
        for batch in _loader(dataset, batch_size):
            observable = batch["observable"]
            identity = batch["hidden_state"]["evaluation_identity_history"].long()
            result = model.associate(observable)
            history = observable["points_history"]
            visibility = observable["visibility_history"].bool()
            exact_velocity = batch["hidden_state"]["exact_point_velocity"]
            batch_now, frames, points, _ = history.shape
            anchors_visible = visibility[:, -1]
            dynamic = torch.linalg.vector_norm(exact_velocity, dim=-1) > STATIC_VELOCITY_THRESHOLD
            static = ~dynamic
            for frame in range(frames - 1):
                source_identity = identity[:, frame]
                same = identity[:, -1, :, None] == source_identity[:, None, :]
                expected = same.long().argmax(dim=-1)
                matchable = anchors_visible & same.any(dim=-1) & visibility[:, frame].gather(1, expected)
                accepted = result.source_index[:, frame] >= 0
                correct = accepted & matchable & (result.source_index[:, frame] == expected)
                counts["matchable"] += int(matchable.sum())
                counts["correct"] += int(correct.sum())
                counts["static_matchable"] += int((matchable & static).sum())
                counts["static_correct"] += int((correct & static).sum())
                counts["dynamic_matchable"] += int((matchable & dynamic).sum())
                counts["dynamic_correct"] += int((correct & dynamic).sum())
                selected = history[:, frame].gather(
                    1,
                    result.source_index[:, frame].clamp_min(0)[..., None].expand(batch_now, points, 3),
                )
                truth = history[:, frame].gather(1, expected[..., None].expand(batch_now, points, 3))
                valid_error = accepted & matchable
                if valid_error.any():
                    errors.extend(
                        torch.linalg.vector_norm(selected - truth, dim=-1)[valid_error].detach().cpu().tolist()
                    )
    error_array = np.asarray(errors, dtype=np.float64)
    return {
        "overall_correspondence_accuracy": counts["correct"] / max(counts["matchable"], 1),
        "static_point_correspondence_accuracy": counts["static_correct"] / max(counts["static_matchable"], 1),
        "dynamic_point_correspondence_accuracy": counts["dynamic_correct"] / max(counts["dynamic_matchable"], 1),
        "matchable_points": counts["matchable"],
        "static_matchable_points": counts["static_matchable"],
        "dynamic_matchable_points": counts["dynamic_matchable"],
        "correspondence_error_median": float(np.quantile(error_array, 0.50)) if error_array.size else None,
        "correspondence_error_p95": float(np.quantile(error_array, 0.95)) if error_array.size else None,
        "correspondence_error_count": int(error_array.size),
        "static_velocity_threshold": STATIC_VELOCITY_THRESHOLD,
    }


def _correspondence_metrics(
    *, config: Mapping[str, Any], models: Mapping[int, Any], datasets: Mapping[str, Dataset[Any]]
) -> dict[str, Any]:
    rows = [
        {"seed": int(seed), **_correspondence_one_seed(
            models[seed], datasets["point_test"], batch_size=int(config["training"]["evaluation_batch_size"])
        )}
        for seed in sorted(models)
    ]
    metric_names = (
        "overall_correspondence_accuracy",
        "static_point_correspondence_accuracy",
        "dynamic_point_correspondence_accuracy",
        "correspondence_error_median",
        "correspondence_error_p95",
    )
    return {
        "dataset": "frozen ID point-test worlds; fair association receives observable fields only",
        "per_seed": rows,
        "five_seed_summary": {name: _summary(rows, name) for name in metric_names},
    }


def _point_permutation_audit(invariance: Mapping[str, Any]) -> dict[str, Any]:
    """Extract only the regenerated corrected-NDCG A3 point-order evidence."""

    counts = invariance.get("candidate_counts")
    if not isinstance(counts, Mapping):
        raise ValueError("A3 invariance artifact lacks candidate-count reports")
    report: dict[str, Any] = {
        "source": "A3 corrected final-v2 invariance artifact",
        "view": POINT_PERMUTATION_VIEW,
        "metric_definition_version": RANKING_METRIC_SCHEMA,
        "candidate_counts": {},
    }
    for count in (5, 10, 20, 50):
        views = counts.get(f"C{count}", {}).get("views")
        if not isinstance(views, Mapping):
            raise ValueError(f"A3 invariance artifact lacks C{count} views")
        original, permuted = views.get("original"), views.get(POINT_PERMUTATION_VIEW)
        if not isinstance(original, Mapping) or not isinstance(permuted, Mapping):
            raise ValueError(f"A3 invariance artifact lacks C{count} point-permutation evidence")
        paired = permuted.get("paired_view_minus_original_group_mean")
        if not isinstance(paired, Mapping):
            raise ValueError(f"A3 C{count} point-permutation evidence is not paired")
        report["candidate_counts"][f"C{count}"] = {
            "original": dict(original),
            "point_permuted": dict(permuted),
            "paired_point_permuted_minus_original_group_mean": dict(paired),
        }
    return report


def verify_inputs(
    config_path: Path = DEFAULT_CONFIG,
    *,
    legacy_root: Path = LEGACY_ROOT,
    final_root: Path = FINAL_ROOT,
) -> dict[str, Any]:
    """Fail closed unless corrected A2/A3/A5 evidence and frozen CPU inputs exist."""

    final_root = Path(final_root)
    ranking = verify_ranking_inputs(config_path, legacy_root=legacy_root, final_root=final_root)
    if torch.cuda.is_available() or torch.cuda.device_count() != 0:
        raise RuntimeError("final-v2 completion must run in the frozen CPU-only Python environment")
    config = _load_config(Path(config_path))
    schema = _read_mapping(final_root / "evaluation_schema.json")
    if schema.get("metric_definition_version") != RANKING_METRIC_SCHEMA:
        raise ValueError("final-v2 evaluation schema is not corrected utility-gain NDCG")
    invariance = _read_mapping(final_root / "invariance" / "primary_ranking_invariance.json")
    id_ood = _read_mapping(final_root / "id_ood" / "primary_id_ood_ranking.json")
    if invariance.get("artifact_schema") != ARTIFACT_SCHEMA_VERSION or invariance.get("stage") != "A3_final_ranking_invariance":
        raise ValueError("A3 corrected invariance artifact is absent or incompatible")
    if id_ood.get("artifact_schema") != ARTIFACT_SCHEMA_VERSION or id_ood.get("stage") != "A5_complete_id_ood_ranking":
        raise ValueError("A5 corrected ID/OOD artifact is absent or incompatible")
    _point_permutation_audit(invariance)
    return {
        "status": "verified",
        "cpu_only": True,
        "gpu_used": False,
        "config_digest": _json_digest(config),
        "ranking_input_verification": ranking,
        "a3_invariance": str(final_root / "invariance" / "primary_ranking_invariance.json"),
        "a5_id_ood": str(final_root / "id_ood" / "primary_id_ood_ranking.json"),
    }


def run(
    config_path: Path = DEFAULT_CONFIG,
    *,
    legacy_root: Path = LEGACY_ROOT,
    final_root: Path = FINAL_ROOT,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Run A6 secondary evaluations and write one immutable final-v2 report."""

    final_root = Path(final_root)
    output_path = final_root / "completion" / "completion_extensions.json" if output_path is None else Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite final-v2 completion artifact: {output_path}")
    verification = verify_inputs(config_path, legacy_root=legacy_root, final_root=final_root)
    config = _load_config(Path(config_path))
    torch.set_num_threads(int(config["training"]["num_threads"]))
    selection = dict(verification["ranking_input_verification"]["validation_selected_arm"])
    models, specifications = _load_primary_models(config, legacy_root=Path(legacy_root), selection=selection)
    datasets = build_final_v2_datasets(config)
    invariance = _read_mapping(final_root / "invariance" / "primary_ranking_invariance.json")
    report = {
        "artifact_schema": COMPLETION_SCHEMA_VERSION,
        "stage": "A6_phase_6_7_completion_extensions",
        "completed_at_utc": _timestamp(),
        "cpu_only": True,
        "gpu_used": False,
        "config_digest": _json_digest(config),
        "metric_definition_version": RANKING_METRIC_SCHEMA,
        "validation_selected_arm": selection,
        "checkpoint_provenance": [asdict(specifications[seed]) for seed in sorted(specifications)],
        "source_code_provenance": code_provenance_for_files(_source_files(), repository_root=Path(PROJECT_ROOT)),
        "input_artifacts": {
            "a3_invariance": str(final_root / "invariance" / "primary_ranking_invariance.json"),
            "a3_invariance_sha256": _sha256_file(final_root / "invariance" / "primary_ranking_invariance.json"),
            "a5_id_ood": str(final_root / "id_ood" / "primary_id_ood_ranking.json"),
            "a5_id_ood_sha256": _sha256_file(final_root / "id_ood" / "primary_id_ood_ranking.json"),
        },
        "ranking_point_permutation": _point_permutation_audit(invariance),
        "mask_secondary_metrics": _mask_evaluation(config=config, models=models, datasets=datasets),
        "correspondence_metrics": _correspondence_metrics(config=config, models=models, datasets=datasets),
        "shared_scene_latency": _latency_evaluation(config=config, models=models, datasets=datasets),
        "protocol_notes": [
            "Ranking point-permutation values are A3 regenerated final-v2 values with standard utility-gain NDCG; no legacy incremental ranking file was read.",
            "Mask IoU uses the frozen 0.50 threshold; mask AP is threshold-independent.",
            "Latency measures a complete shared candidate world on CPU and reports process RSS/HWM snapshots from the helper.",
            "Correspondence uses evaluator-only hidden identity for labels/accuracy only; the selected fair model receives observable fields only.",
        ],
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
