"""Isolated, non-decision CPU diagnostic for Milestone 2E.

The full 2E protocol is intentionally expensive: it evaluates five seeds,
four candidate counts, and broad OOD/invariance diagnostics.  This module is
for the execution contract's one permitted reduced diagnostic when the full
matrix exceeds its CPU budget.  It never trains a model, changes a split,
changes labels, or writes into the full-protocol artifacts.  Instead, it loads
only provenance-compatible checkpoints and evaluates the first fixed C20
candidate worlds of the already-frozen test dataset.

Its report is explicitly non-decisional.  It can help decide whether obtaining
a revised CPU budget is scientifically justified, but it cannot select a model
or replace the full five-seed C5/C10/C20/C50 study.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset, Subset

from actmask.experiments.milestone2e import OUTPUT_DIR, _load_config
from actmask.experiments.milestone2e_full import (
    FULL_OUTPUT_DIR,
    PRIMARY_HEAD,
    DiagnosticIdentityDataset,
    _aggregate_seed_rows,
    _evaluate_ranking,
    _fair_prediction,
    _json_digest,
    _model,
    build_datasets,
)
from actmask.experiments.milestone2e import validate_checkpoint_metadata


DEFAULT_OUTPUT_DIR = OUTPUT_DIR / "reduced_diagnostic"
DEFAULT_SEEDS = (1601, 2603, 3607)
PRIMARY_VARIANTS = (
    ("GroundTruthIdentityCorrespondence", "OriginalVectorBackbone"),
    ("GroundTruthIdentityCorrespondence", "ScalarInvariantFeatureBackbone"),
    ("GroundTruthIdentityCorrespondence", "ActionFrameRelativeBackbone"),
    ("CoordinateInvariantMutualNN", "OriginalVectorBackbone"),
    ("CoordinateInvariantMutualNN", "ScalarInvariantFeatureBackbone"),
    ("CoordinateInvariantMutualNN", "ActionFrameRelativeBackbone"),
    ("SoftCoordinateInvariantCorrespondence", "ActionFrameRelativeBackbone"),
    ("NoExplicitCorrespondence", "ActionFrameRelativeBackbone"),
)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _load_checkpoint(
    *, config: Mapping[str, Any], correspondence: str, backbone: str, seed: int
) -> tuple[Any, dict[str, Any]]:
    """Load one exact full-protocol checkpoint without creating an artifact."""

    token = f"{correspondence}__{backbone}__{PRIMARY_HEAD}__seed{seed}"
    checkpoint_path = FULL_OUTPUT_DIR / "checkpoints" / f"{token}.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"required full-protocol checkpoint is absent: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"checkpoint is not a mapping: {checkpoint_path}")
    validate_checkpoint_metadata(
        checkpoint,
        {
            "architecture": "FactorialTemporalActMask",
            "correspondence": correspondence,
            "backbone": backbone,
            "ranking_head": PRIMARY_HEAD,
            "seed": int(seed),
            "config_digest": _json_digest(config),
        },
    )
    calibration = checkpoint.get("calibration")
    if not isinstance(calibration, Mapping) or not {"coefficient", "intercept"}.issubset(calibration):
        raise ValueError(f"checkpoint lacks validation-only Platt calibration: {checkpoint_path}")
    model = _model(config, correspondence, backbone)
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval(), {
        "checkpoint": str(checkpoint_path),
        "calibration": {
            "coefficient": float(calibration["coefficient"]),
            "intercept": float(calibration["intercept"]),
        },
        "best_epoch": int(checkpoint.get("best_epoch", 0)),
    }


def _fixed_group_prefix(dataset: Dataset[Any], *, groups: int, candidate_count: int) -> Subset[Any]:
    """Take complete, contiguous C20 worlds and prove their indexing contract."""

    if groups < 1:
        raise ValueError("groups must be positive")
    if not isinstance(dataset, DiagnosticIdentityDataset):
        raise TypeError("reduced diagnostic requires the identity-preserving ranking wrapper")
    if len(dataset.group_records) < groups:
        raise ValueError(f"requested {groups} worlds but frozen test has {len(dataset.group_records)}")
    expected = groups * candidate_count
    subset = Subset(dataset, range(expected))
    observed_group_ids: list[int] = []
    for group_index in range(groups):
        start = group_index * candidate_count
        members = [subset[index] for index in range(start, start + candidate_count)]
        group_ids = {int(sample["metadata"]["candidate_set_id"]) for sample in members}
        candidate_ids = sorted(int(sample["metadata"]["candidate_id"]) for sample in members)
        if len(group_ids) != 1 or candidate_ids != list(range(candidate_count)):
            raise AssertionError("reduced diagnostic would split or corrupt a frozen candidate world")
        observed_group_ids.append(next(iter(group_ids)))
    if observed_group_ids != [int(record.group_id) for record in dataset.group_records[:groups]]:
        raise AssertionError("reduced diagnostic prefix differs from frozen ranking-world order")
    return subset


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Milestone 2E reduced diagnostic", "",
        "**Status:** non-decision CPU budget diagnostic.", "",
        "This evaluates fixed complete C20 candidate worlds from the frozen test",
        "dataset using existing provenance-compatible full-protocol checkpoints.",
        "It does not select a model and cannot replace the required final study.", "",
        f"- Candidate worlds: {report['groups']} of {report['frozen_test_groups']}.",
        f"- Seeds: {', '.join(str(seed) for seed in report['seeds'])}.",
        "- Score: frozen 0.5 learned / 0.5 MultiHypothesis hybrid with the checkpoint's validation-only calibration.",
        "",
        "| Correspondence | Backbone | Top-1 mean | Normalized regret mean |", "|---|---|---:|---:|",
    ]
    for row in report["rows"]:
        summary = row["summary"]
        top1 = summary["top1_success"]["mean"]
        regret = summary["normalized_regret"]["mean"]
        lines.append(f"| {row['correspondence']} | {row['backbone']} | {top1:.4f} | {regret:.4f} |")
    return "\n".join(lines) + "\n"


def run(*, groups: int = 10, seeds: Sequence[int] = DEFAULT_SEEDS, output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    """Run the one reduced diagnostic without touching full-protocol outputs."""

    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic output: {output_dir}")
    config = _load_config()
    if str(config["experiment"]["device"]).lower() != "cpu":
        raise ValueError("the Milestone 2E reduced diagnostic is CPU-only")
    requested_seeds = tuple(int(seed) for seed in seeds)
    if requested_seeds != DEFAULT_SEEDS:
        raise ValueError(f"reduced diagnostic requires the predeclared seeds {DEFAULT_SEEDS}")
    datasets = build_datasets(config)
    full_dataset = datasets["ranking_test/20"]
    reduced_dataset = _fixed_group_prefix(full_dataset, groups=groups, candidate_count=20)
    fair = _fair_prediction(reduced_dataset, batch_size=int(config["training"]["evaluation_batch_size"]))
    rows: list[dict[str, Any]] = []
    for correspondence, backbone in PRIMARY_VARIANTS:
        per_seed: list[dict[str, Any]] = []
        for seed in requested_seeds:
            model, provenance = _load_checkpoint(
                config=config, correspondence=correspondence, backbone=backbone, seed=seed
            )
            metrics, _ = _evaluate_ranking(
                model,
                reduced_dataset,
                fair,
                provenance["calibration"],
                int(config["training"]["evaluation_batch_size"]),
            )
            per_seed.append({"seed": seed, "metrics": metrics, **provenance})
        rows.append(
            {
                "correspondence": correspondence,
                "backbone": backbone,
                "per_seed": per_seed,
                "summary": _aggregate_seed_rows(per_seed),
            }
        )
    report: dict[str, Any] = {
        "status": "non-decision reduced diagnostic",
        "cpu_only": True,
        "config_digest": _json_digest(config),
        "candidate_count": 20,
        "groups": int(groups),
        "frozen_test_groups": len(full_dataset.group_records),
        "seeds": list(requested_seeds),
        "variants": [list(variant) for variant in PRIMARY_VARIANTS],
        "scope": "fixed complete prefix of frozen C20 worlds; no training, selection, or protocol change",
        "rows": rows,
    }
    _write_json(output_dir / "summary.json", report)
    (output_dir / "summary.md").write_text(_markdown(report))
    return report


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", type=int, default=10, help="complete frozen C20 worlds to evaluate")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    arguments = parser.parse_args(argv)
    report = run(groups=arguments.groups, output_dir=arguments.output_dir)
    print(f"completed reduced diagnostic: {len(report['rows'])} variants × {len(report['seeds'])} seeds")
    return report


if __name__ == "__main__":
    main()
