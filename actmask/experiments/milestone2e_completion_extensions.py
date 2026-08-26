"""CPU-only completion extensions for the frozen Milestone 2E protocol.

The original full runner intentionally keeps its long factorial execution
simple and resumable.  This module is a *post-completion*, read-only consumer
of that run.  It fills four reporting gaps without retraining, changing a
split, or mutating any earlier result:

* ranking-level point-order robustness under a group-shared point permutation;
* shared-world C5/C10/C20/C50 latency, throughput, memory snapshots, and
  5 Hz / 10 Hz verdicts;
* C20 point-mask AP and IoU at a fixed, predeclared 0.50 threshold; and
* static-point correspondence accuracy plus median correspondence error.

It deliberately writes one new artifact only:
``outputs/actmask/milestone2e/completion_extensions.json``.  The module
refuses to run until a non-smoke ``summary.json`` exists, selected checkpoints
match the frozen configuration exactly, and CUDA is unavailable.  This makes a
stale, partial, GPU-backed, or cross-arm result a hard failure rather than an
apparently valid extension report.
"""

from __future__ import annotations

import argparse
import json
import resource
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, Subset

from actmask.data.milestone2b_dataset import _clone_tensor_tree, validate_observable_mapping
from actmask.data.milestone2c_dataset import CorrespondenceCorruptionDataset, RigidTransformDataset
from actmask.eval.pr_metrics import precision_recall_curve
from actmask.eval.statistics import bootstrap_ci
from actmask.experiments.milestone2c import _loader
from actmask.experiments.milestone2e import (
    DEFAULT_CONFIG,
    OUTPUT_DIR,
    RANKING_METRICS,
    RankingBundle,
    _load_config,
    _write_json,
    paired_ranking_report,
    ranking_metrics_tie_aware,
    validate_checkpoint_metadata,
)
from actmask.experiments.milestone2e_full import (
    FULL_OUTPUT_DIR,
    PRIMARY_HEAD,
    DiagnosticIdentityDataset,
    FactorialPrediction,
    _fair_prediction,
    _forward,
    _hybrid_scores,
    _json_digest,
    _model,
    build_datasets,
    predict_factorial,
)
from actmask.models.milestone2e import CORRESPONDENCE_VARIANTS


DEFAULT_OUTPUT_PATH = OUTPUT_DIR / "completion_extensions.json"
PRIMARY_SELECTION_FILE = "validation_primary_selection.json"
SUMMARY_FILE = "summary.json"
POINT_PERMUTATION_SEED = 20261331
STATIC_VELOCITY_THRESHOLD = 0.02
MASK_IOU_THRESHOLD = 0.50


@dataclass(frozen=True)
class CheckpointSpec:
    """Provenance for a checkpoint used by this extension."""

    correspondence: str
    backbone: str
    ranking_head: str
    seed: int
    path: str
    best_epoch: int
    calibration: dict[str, float]


def _read_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def _assert_cpu_contract(config: Mapping[str, Any]) -> None:
    """Fail closed if this process could silently move the protocol to CUDA."""

    if str(config["experiment"]["device"]).lower() != "cpu":
        raise ValueError("Milestone 2E completion extensions are CPU-only")
    if torch.cuda.is_available() or torch.cuda.device_count() != 0:
        raise RuntimeError(
            "CUDA is visible to this process; refusing to run a CPU-only Milestone 2E extension"
        )


def _selection_from_summary(
    *, output_root: Path, config: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, str]]:
    """Validate the final full-run marker and its validation-only selection."""

    summary_path = output_root / SUMMARY_FILE
    summary = _read_mapping(summary_path)
    if summary.get("milestone") != "2E" or summary.get("cpu_only") is not True:
        raise ValueError(f"{summary_path} is not a CPU-only Milestone 2E final report")
    if summary.get("smoke") is not False:
        raise ValueError(f"{summary_path} is a smoke or incomplete report, not the frozen protocol")

    selection_path = output_root / PRIMARY_SELECTION_FILE
    selection_raw = _read_mapping(selection_path)
    keys = ("correspondence", "backbone")
    if any(not isinstance(selection_raw.get(key), str) for key in keys):
        raise ValueError(f"invalid validation-primary selection artifact: {selection_path}")
    selection = {key: str(selection_raw[key]) for key in keys}
    if selection["correspondence"] == "GroundTruthIdentityCorrespondence":
        raise ValueError("the diagnostic-only identity arm cannot be the final fair verifier")
    summary_selection = summary.get("validation_selected_primary")
    if not isinstance(summary_selection, Mapping) or any(
        summary_selection.get(key) != selection[key] for key in keys
    ):
        raise ValueError("summary and validation-primary selection artifacts disagree")

    required_counts = [5, 10, 20, 50]
    counts = [int(value) for value in config["ranking"]["candidate_counts"]]
    if counts != required_counts:
        raise ValueError("completion extension requires the frozen C5/C10/C20/C50 protocol")
    seeds = [int(value) for value in config["experiment"]["training_seeds"]]
    if len(seeds) != 5 or len(set(seeds)) != 5:
        raise ValueError("completion extension requires five distinct deterministic primary seeds")
    return summary, selection


def _checkpoint_path(
    output_root: Path, *, correspondence: str, backbone: str, ranking_head: str, seed: int
) -> Path:
    token = f"{correspondence}__{backbone}__{ranking_head}__seed{seed}"
    return output_root / "checkpoints" / f"{token}.pt"


def _load_checkpoint(
    *,
    config: Mapping[str, Any],
    output_root: Path,
    correspondence: str,
    backbone: str,
    ranking_head: str,
    seed: int,
) -> tuple[Any, CheckpointSpec]:
    """Load only an exactly provenance-compatible CPU checkpoint."""

    path = _checkpoint_path(
        output_root,
        correspondence=correspondence,
        backbone=backbone,
        ranking_head=ranking_head,
        seed=seed,
    )
    if not path.exists():
        raise FileNotFoundError(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"checkpoint is not a mapping: {path}")
    validate_checkpoint_metadata(
        checkpoint,
        {
            "architecture": "FactorialTemporalActMask",
            "correspondence": correspondence,
            "backbone": backbone,
            "ranking_head": ranking_head,
            "seed": int(seed),
            "config_digest": _json_digest(config),
        },
    )
    calibration = checkpoint.get("calibration")
    if not isinstance(calibration, Mapping) or not {"coefficient", "intercept"}.issubset(calibration):
        raise ValueError(f"checkpoint lacks validation-only Platt calibration: {path}")
    model = _model(config, correspondence, backbone).cpu()
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    if any(parameter.device.type != "cpu" for parameter in model.parameters()):
        raise RuntimeError(f"checkpoint did not load on CPU only: {path}")
    spec = CheckpointSpec(
        correspondence=correspondence,
        backbone=backbone,
        ranking_head=ranking_head,
        seed=int(seed),
        path=str(path),
        best_epoch=int(checkpoint.get("best_epoch", 0)),
        calibration={
            "coefficient": float(calibration["coefficient"]),
            "intercept": float(calibration["intercept"]),
        },
    )
    return model, spec


def _load_primary_models(
    *, config: Mapping[str, Any], output_root: Path, selection: Mapping[str, str]
) -> tuple[dict[int, Any], dict[int, CheckpointSpec]]:
    models: dict[int, Any] = {}
    specs: dict[int, CheckpointSpec] = {}
    for value in config["experiment"]["training_seeds"]:
        seed = int(value)
        model, spec = _load_checkpoint(
            config=config,
            output_root=output_root,
            correspondence=str(selection["correspondence"]),
            backbone=str(selection["backbone"]),
            ranking_head=PRIMARY_HEAD,
            seed=seed,
        )
        models[seed], specs[seed] = model, spec
    return models, specs


def _group_rng(group_id: int) -> np.random.Generator:
    words = [
        POINT_PERMUTATION_SEED,
        int(group_id) & 0xFFFFFFFF,
        (int(group_id) >> 32) & 0xFFFFFFFF,
        0x504E54,
    ]
    return np.random.default_rng(np.random.SeedSequence(words))


def _nonidentity_point_permutation(points: int, *, group_id: int) -> torch.Tensor:
    if points < 2:
        raise ValueError("point permutation needs at least two points")
    values = _group_rng(group_id).permutation(points)
    if np.array_equal(values, np.arange(points)):
        values[[0, 1]] = values[[1, 0]]
    return torch.as_tensor(values, dtype=torch.long)


def _reorder_point_fields(sample: dict[str, Any], order: torch.Tensor) -> None:
    """Apply one order to every point-indexed field, preserving semantics.

    The same order is used in every observation frame and every candidate in a
    world.  That makes this an order-invariance test, not a corruption of
    temporal correspondence.  Labels are reordered with the final-frame point
    axis, while candidate scores, utility, and metadata remain untouched.
    """

    if order.ndim != 1 or order.dtype != torch.long:
        raise TypeError("point order must be a one-dimensional long tensor")
    observable = sample["observable"]
    points = int(observable["points_history"].shape[1])
    if order.numel() != points or set(order.tolist()) != set(range(points)):
        raise ValueError("point order is not a complete permutation")
    observable["points_history"] = observable["points_history"][:, order].clone()
    observable["visibility_history"] = observable["visibility_history"][:, order].clone()
    observable["estimated_velocity"] = observable["estimated_velocity"][order].clone()
    observable["velocity_confidence"] = observable["velocity_confidence"][order].clone()

    targets = sample["targets"]
    for name in (
        "ground_truth_future_mask",
        "hard_positive_mask",
        "hard_negative_mask",
        "target_object_mask",
    ):
        if name in targets:
            targets[name] = targets[name][order].clone()
    if "contact_matrix" in targets:
        targets["contact_matrix"] = targets["contact_matrix"][:, order].clone()

    hidden = sample.get("hidden_state")
    if isinstance(hidden, Mapping):
        # ``hidden_state`` is evaluator-only provenance.  Reordering it is
        # necessary to keep the oracle arm's labels aligned, never to expose it
        # to a fair model.
        for name in (
            "evaluation_identity_history",
            "exact_future_point_trajectories",
            "exact_point_velocity",
            "exact_point_acceleration",
            "exact_contact_matrix",
            "exact_contact_state",
        ):
            value = hidden.get(name)
            if not isinstance(value, torch.Tensor):
                continue
            if name in {"evaluation_identity_history", "exact_future_point_trajectories", "exact_contact_matrix"}:
                hidden[name] = value[:, order].clone()
            else:
                hidden[name] = value[order].clone()
    validate_observable_mapping(observable)


class PointPermutationDataset(Dataset[dict[str, Any]]):
    """Deterministic ranking-level point-order view with group consistency."""

    view_name = "group_shared_all_frame_point_permutation"

    def __init__(self, dataset: Dataset[dict[str, Any]]) -> None:
        records = tuple(getattr(dataset, "group_records", ()))
        if not records or len(dataset) % len(records):
            raise TypeError("point permutation requires complete grouped candidate worlds")
        self.dataset = dataset
        self.group_records = records
        self.candidate_count = len(dataset) // len(records)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = _clone_tensor_tree(self.dataset[index])
        group_id = int(sample["metadata"]["candidate_set_id"])
        points = int(sample["observable"]["points_history"].shape[1])
        _reorder_point_fields(sample, _nonidentity_point_permutation(points, group_id=group_id))
        sample["metadata"].update(
            {
                "ranking_view": self.view_name,
                "point_permutation_seed": POINT_PERMUTATION_SEED,
            }
        )
        return sample


def _load_original_ranking_metrics(
    *,
    output_root: Path,
    config: Mapping[str, Any],
    selection: Mapping[str, str],
    candidate_count: int,
    seed: int,
) -> dict[str, Any]:
    """Use the frozen factorial result as the unmodified paired baseline."""

    token = (
        f"{selection['correspondence']}__{selection['backbone']}__"
        f"C{candidate_count}__seed{seed}.json"
    )
    path = output_root / "factorial_incremental" / token
    saved = _read_mapping(path)
    expected = {
        "config_digest": _json_digest(config),
        "correspondence": str(selection["correspondence"]),
        "backbone": str(selection["backbone"]),
        "candidate_count": int(candidate_count),
        "seed": int(seed),
    }
    mismatch = {key: (saved.get(key), value) for key, value in expected.items() if saved.get(key) != value}
    if mismatch:
        raise ValueError(f"incompatible frozen factorial row {path}: {mismatch}")
    metrics = saved.get("metrics")
    if not isinstance(metrics, Mapping) or not isinstance(metrics.get("per_group"), list):
        raise ValueError(f"frozen factorial row lacks ranked per-group metrics: {path}")
    if len(metrics["per_group"]) != int(config["ranking"]["test_groups"]):
        raise ValueError(f"frozen factorial row has an incomplete set of candidate worlds: {path}")
    return dict(metrics)


def _evaluate_hybrid(
    model: Any,
    *,
    spec: CheckpointSpec,
    dataset: Dataset[Any],
    fair: FactorialPrediction,
    batch_size: int,
) -> dict[str, Any]:
    learned = predict_factorial(model, dataset, batch_size=batch_size)
    scores, _ = _hybrid_scores(learned, fair, spec.calibration)
    return ranking_metrics_tie_aware(
        RankingBundle(
            scores=np.asarray(scores, dtype=np.float64),
            success=learned.success,
            utility=learned.candidate_utility,
            metadata=learned.metadata,
        ),
        score_name="frozen_hybrid_alpha_0.5",
    )


def _metric_summary(rows: Sequence[Mapping[str, Any]], *, metric_source: str) -> dict[str, Any]:
    """Mean/std over five independent trained seeds for ranking metrics."""

    result: dict[str, Any] = {"seeds": [int(row["seed"]) for row in rows]}
    for metric in RANKING_METRICS:
        values = np.asarray([float(row[metric_source][metric]) for row in rows], dtype=np.float64)
        result[metric] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1 if values.size > 1 else 0)),
            "values": values.tolist(),
        }
    return result


def _point_permutation_ranking(
    *,
    config: Mapping[str, Any],
    output_root: Path,
    selection: Mapping[str, str],
    models: Mapping[int, Any],
    specs: Mapping[int, CheckpointSpec],
    datasets: Mapping[str, Dataset[Any]],
) -> dict[str, Any]:
    """Evaluate the missing ranking-level point-order view at C5/C10/C20/C50."""

    batch_size = int(config["training"]["evaluation_batch_size"])
    resamples = int(config["statistics"]["bootstrap_resamples"])
    permutations = int(config["statistics"]["permutation_samples"])
    report: dict[str, Any] = {
        "view": PointPermutationDataset.view_name,
        "point_permutation_seed": POINT_PERMUTATION_SEED,
        "definition": (
            "One deterministic non-identity point permutation per candidate world, applied to every "
            "history frame and all point-indexed targets/provenance for every candidate in that world."
        ),
        "score": "frozen_hybrid_alpha_0.5",
        "candidate_counts": {},
    }
    for count in (5, 10, 20, 50):
        original_dataset = datasets[f"ranking_test/{count}"]
        permuted_dataset = PointPermutationDataset(original_dataset)
        fair = _fair_prediction(permuted_dataset, batch_size=batch_size)
        rows: list[dict[str, Any]] = []
        for seed in sorted(models):
            original = _load_original_ranking_metrics(
                output_root=output_root,
                config=config,
                selection=selection,
                candidate_count=count,
                seed=seed,
            )
            changed = _evaluate_hybrid(
                models[seed],
                spec=specs[seed],
                dataset=permuted_dataset,
                fair=fair,
                batch_size=batch_size,
            )
            rows.append(
                {
                    "seed": int(seed),
                    "original": original,
                    "point_permuted": changed,
                    "paired_point_permuted_minus_original": paired_ranking_report(
                        changed,
                        original,
                        seed=20261390 + count * 100 + int(seed),
                        resamples=resamples,
                        permutations=permutations,
                    ),
                }
            )
        report["candidate_counts"][f"C{count}"] = {
            "per_seed": rows,
            "original_summary": _metric_summary(rows, metric_source="original"),
            "point_permuted_summary": _metric_summary(rows, metric_source="point_permuted"),
        }
    return report


def _mask_scalar_metrics(probability: np.ndarray, target: np.ndarray) -> dict[str, float]:
    score = np.asarray(probability, dtype=np.float64).reshape(-1)
    label = np.asarray(target, dtype=np.float64).reshape(-1) >= 0.5
    if score.shape != label.shape or not score.size:
        raise ValueError("mask scores and labels must be non-empty and aligned")
    if not np.isfinite(score).all() or np.any((score < 0.0) | (score > 1.0)):
        raise ValueError("mask probabilities must be finite values in [0, 1]")
    prediction = score >= MASK_IOU_THRESHOLD
    true_positive = int(np.logical_and(prediction, label).sum())
    false_positive = int(np.logical_and(prediction, ~label).sum())
    false_negative = int(np.logical_and(~prediction, label).sum())
    denominator = true_positive + false_positive + false_negative
    curve = precision_recall_curve(score, label.astype(np.float64))
    return {
        "mask_ap": float(curve.average_precision),
        "mask_pr_auc": float(curve.pr_auc),
        "mask_iou": float(true_positive / denominator) if denominator else 1.0,
        "mask_precision": float(true_positive / (true_positive + false_positive))
        if true_positive + false_positive
        else 0.0,
        "mask_recall": float(true_positive / (true_positive + false_negative))
        if true_positive + false_negative
        else 0.0,
    }


def _mask_metrics(
    prediction: FactorialPrediction, *, seed: int, bootstrap_resamples: int
) -> dict[str, Any]:
    """Report pooled and candidate-world macro mask AP/IoU without test tuning."""

    probability = np.asarray(prediction.mask_probability, dtype=np.float64)
    target = np.asarray(prediction.mask_target, dtype=np.float64)
    if probability.shape != target.shape or probability.ndim != 2:
        raise ValueError("factorial mask prediction must be [candidate, point] and target-aligned")
    if len(prediction.metadata) != probability.shape[0]:
        raise ValueError("mask metadata and candidate predictions disagree")
    groups: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(prediction.metadata):
        groups[int(row["candidate_set_id"])].append(index)
    per_group: list[dict[str, Any]] = []
    for group_id, indices in sorted(groups.items()):
        metric = _mask_scalar_metrics(probability[indices], target[indices])
        per_group.append({"candidate_set_id": int(group_id), **metric})
    if len(per_group) < 2:
        raise ValueError("mask metrics need at least two complete candidate worlds")
    macro: dict[str, Any] = {}
    for offset, name in enumerate(("mask_ap", "mask_pr_auc", "mask_iou", "mask_precision", "mask_recall")):
        values = [float(row[name]) for row in per_group]
        macro[name] = bootstrap_ci(
            values,
            seed=20261420 + int(seed) * 100 + offset,
            resamples=bootstrap_resamples,
        )
    return {
        "iou_threshold": MASK_IOU_THRESHOLD,
        "threshold_policy": "Fixed predeclared 0.50; no validation or test threshold selection was performed.",
        "micro_pooled": _mask_scalar_metrics(probability, target),
        "macro_candidate_set": macro,
        "per_candidate_set": per_group,
    }


def _mask_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"seeds": [int(row["seed"]) for row in rows]}
    for source, names in (
        ("micro_pooled", ("mask_ap", "mask_pr_auc", "mask_iou", "mask_precision", "mask_recall")),
        ("macro_candidate_set", ("mask_ap", "mask_pr_auc", "mask_iou", "mask_precision", "mask_recall")),
    ):
        result[source] = {}
        for name in names:
            if source == "micro_pooled":
                values = np.asarray([float(row["metrics"][source][name]) for row in rows], dtype=np.float64)
            else:
                values = np.asarray(
                    [float(row["metrics"][source][name]["mean"]) for row in rows], dtype=np.float64
                )
            result[source][name] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1 if values.size > 1 else 0)),
                "values": values.tolist(),
            }
    return result


def _mask_evaluation(
    *, config: Mapping[str, Any], models: Mapping[int, Any], datasets: Mapping[str, Dataset[Any]]
) -> dict[str, Any]:
    """Evaluate mask secondary metrics on the frozen C20 ID test worlds."""

    dataset = datasets["ranking_test/20"]
    rows: list[dict[str, Any]] = []
    for seed in sorted(models):
        prediction = predict_factorial(
            models[seed], dataset, batch_size=int(config["training"]["evaluation_batch_size"])
        )
        rows.append(
            {
                "seed": int(seed),
                "metrics": _mask_metrics(
                    prediction,
                    seed=int(seed),
                    bootstrap_resamples=int(config["statistics"]["bootstrap_resamples"]),
                ),
            }
        )
    return {
        "dataset": "ranking_test/C20 ID frozen candidate worlds",
        "per_seed": rows,
        "summary": _mask_summary(rows),
    }


def _correspondence_static_and_median(
    model: Any, dataset: Dataset[Any], *, batch_size: int
) -> dict[str, Any]:
    """Add the two missing source-ID metrics to the existing matching report."""

    if model.correspondence == "NoExplicitCorrespondence":
        return {
            "association_type": "set_aggregation_no_explicit_source_identity",
            "identity_claiming": False,
            "static_point_correspondence_accuracy": None,
            "correspondence_error_median": None,
            "reason": "NoExplicitCorrespondence intentionally makes no point-source identity claim.",
        }
    counts: defaultdict[str, int] = defaultdict(int)
    errors: list[float] = []
    with torch.no_grad():
        for batch in _loader(dataset, batch_size):
            observable = batch["observable"]
            identity = batch["hidden_state"]["evaluation_identity_history"].long()
            result = model.associate(
                observable,
                identity_history=identity
                if model.correspondence == "GroundTruthIdentityCorrespondence"
                else None,
            )
            history = observable["points_history"]
            visible = observable["visibility_history"].bool()
            exact_velocity = batch["hidden_state"]["exact_point_velocity"]
            batch_now, frames, points, _ = history.shape
            anchors_visible = visible[:, -1]
            dynamic = torch.linalg.vector_norm(exact_velocity, dim=-1) > STATIC_VELOCITY_THRESHOLD
            static = ~dynamic
            for frame in range(frames - 1):
                source_id = identity[:, frame]
                same = identity[:, -1, :, None] == source_id[:, None, :]
                expected = same.long().argmax(dim=-1)
                matchable = anchors_visible & same.any(dim=-1) & visible[:, frame].gather(1, expected)
                accepted = result.source_index[:, frame] >= 0
                correct = accepted & matchable & (result.source_index[:, frame] == expected)
                counts["matchable"] += int(matchable.sum())
                counts["correct"] += int(correct.sum())
                counts["dynamic_matchable"] += int((matchable & dynamic).sum())
                counts["dynamic_correct"] += int((correct & dynamic).sum())
                counts["static_matchable"] += int((matchable & static).sum())
                counts["static_correct"] += int((correct & static).sum())
                selected = history[:, frame].gather(
                    1,
                    result.source_index[:, frame]
                    .clamp_min(0)[..., None]
                    .expand(batch_now, points, 3),
                )
                truth = history[:, frame].gather(1, expected[..., None].expand(batch_now, points, 3))
                valid_error = accepted & matchable
                if valid_error.any():
                    errors.extend(
                        torch.linalg.vector_norm(selected - truth, dim=-1)[valid_error]
                        .detach()
                        .cpu()
                        .tolist()
                    )
    return {
        "association_type": str(model.correspondence),
        "identity_claiming": True,
        "static_velocity_threshold": STATIC_VELOCITY_THRESHOLD,
        "static_point_correspondence_accuracy": counts["static_correct"]
        / max(counts["static_matchable"], 1),
        "static_matchable_points": counts["static_matchable"],
        "static_correct_points": counts["static_correct"],
        "correspondence_error_median": float(np.quantile(errors, 0.50)) if errors else None,
        "correspondence_error_count": len(errors),
        # These values mirror fields already present in matching_quality.json
        # and make the extension internally auditable without replacing it.
        "overall_correspondence_accuracy": counts["correct"] / max(counts["matchable"], 1),
        "dynamic_point_correspondence_accuracy": counts["dynamic_correct"]
        / max(counts["dynamic_matchable"], 1),
    }


def _correspondence_extension(
    *, config: Mapping[str, Any], output_root: Path, datasets: Mapping[str, Dataset[Any]]
) -> dict[str, Any]:
    """Measure missing matching fields for every existing full-run view."""

    seeds = [int(value) for value in config["experiment"]["training_seeds"]]
    probe_seed = seeds[0]
    point_test = datasets["point_test"]
    views: dict[str, Dataset[Any]] = {
        "clean": point_test,
        "noise": DiagnosticIdentityDataset(
            CorrespondenceCorruptionDataset(point_test, mode="nearest_neighbor_noise", seed=20261041)
        ),
        "per_frame_permutation": DiagnosticIdentityDataset(
            CorrespondenceCorruptionDataset(point_test, mode="per_frame_permutation", seed=20261042)
        ),
        "resampling": DiagnosticIdentityDataset(
            CorrespondenceCorruptionDataset(point_test, mode="per_frame_resampling", seed=20261043)
        ),
        "occlusion": DiagnosticIdentityDataset(
            CorrespondenceCorruptionDataset(point_test, mode="full_part_occlusion", seed=20261044)
        ),
        "rotation_resampling": DiagnosticIdentityDataset(
            RigidTransformDataset(
                CorrespondenceCorruptionDataset(point_test, mode="per_frame_resampling", seed=20261043),
                mode="random_so3",
            )
        ),
        "axis_occlusion": DiagnosticIdentityDataset(
            RigidTransformDataset(
                CorrespondenceCorruptionDataset(point_test, mode="full_part_occlusion", seed=20261044),
                mode="axis_permutation",
            )
        ),
    }
    report: dict[str, Any] = {
        "probe_seed": probe_seed,
        "static_velocity_threshold": STATIC_VELOCITY_THRESHOLD,
        "views": {},
        "checkpoint_provenance": {},
    }
    for correspondence in CORRESPONDENCE_VARIANTS:
        model, spec = _load_checkpoint(
            config=config,
            output_root=output_root,
            correspondence=correspondence,
            backbone="ScalarInvariantFeatureBackbone",
            ranking_head=PRIMARY_HEAD,
            seed=probe_seed,
        )
        report["checkpoint_provenance"][correspondence] = asdict(spec)
        report["views"][correspondence] = {
            name: _correspondence_static_and_median(
                model, view, batch_size=int(config["training"]["evaluation_batch_size"])
            )
            for name, view in views.items()
        }
    return report


def _memory_snapshot() -> dict[str, float | None]:
    """Best-effort process RSS/HWM snapshot; not a GPU allocator measurement."""

    current_kib: float | None = None
    peak_kib: float | None = None
    status = Path("/proc/self/status")
    if status.exists():
        for line in status.read_text().splitlines():
            if line.startswith("VmRSS:"):
                current_kib = float(line.split()[1])
            elif line.startswith("VmHWM:"):
                peak_kib = float(line.split()[1])
    usage_kib = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Linux reports ru_maxrss in KiB.  The project runs on Linux; retain the
    # /proc readings separately so interpretation remains explicit.
    return {
        "rss_mib": current_kib / 1024.0 if current_kib is not None else None,
        "peak_rss_mib": peak_kib / 1024.0 if peak_kib is not None else usage_kib / 1024.0,
        "ru_maxrss_mib": usage_kib / 1024.0,
    }


def _shared_world_batch(dataset: Dataset[Any], *, candidate_count: int) -> Mapping[str, Any]:
    """Return exactly one complete candidate world, never a mixed batch."""

    batch = next(iter(_loader(Subset(dataset, range(candidate_count)), candidate_count)))
    success = batch["targets"]["success"]
    groups = batch["metadata"]["candidate_set_id"]
    ids = batch["metadata"]["candidate_id"]
    group_values = groups.detach().cpu().tolist() if isinstance(groups, torch.Tensor) else list(groups)
    id_values = ids.detach().cpu().tolist() if isinstance(ids, torch.Tensor) else list(ids)
    if int(success.shape[0]) != candidate_count or len(set(int(value) for value in group_values)) != 1:
        raise AssertionError("latency batch is not one complete shared candidate world")
    if set(int(value) for value in id_values) != set(range(candidate_count)):
        raise AssertionError("latency batch does not contain the complete candidate set")
    return batch


def _latency_one(
    model: Any, dataset: Dataset[Any], *, candidate_count: int, config: Mapping[str, Any]
) -> dict[str, Any]:
    """CPU wall-clock latency for one shared candidate world."""

    batch = _shared_world_batch(dataset, candidate_count=candidate_count)
    warmup = int(config["runtime"]["warmup"])
    repeats = int(config["runtime"]["repeats"])
    with torch.no_grad():
        for _ in range(warmup):
            _forward(model, batch)
        before = _memory_snapshot()
        samples_ms: list[float] = []
        for _ in range(repeats):
            started = time.perf_counter()
            _forward(model, batch)
            samples_ms.append((time.perf_counter() - started) * 1000.0)
        after = _memory_snapshot()
    samples = np.asarray(samples_ms, dtype=np.float64)
    p50 = float(np.quantile(samples, 0.50))
    p95 = float(np.quantile(samples, 0.95))
    mean = float(samples.mean())
    return {
        "candidate_count": int(candidate_count),
        "shared_candidate_set_id": int(
            batch["metadata"]["candidate_set_id"][0].detach().cpu()
            if isinstance(batch["metadata"]["candidate_set_id"], torch.Tensor)
            else batch["metadata"]["candidate_set_id"][0]
        ),
        "warmup": warmup,
        "repeats": repeats,
        "latency_p50_ms": p50,
        "latency_p95_ms": p95,
        "latency_mean_ms": mean,
        "throughput_candidates_per_second": float(candidate_count / (mean / 1000.0)),
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "supports_5hz_p95": bool(p95 <= 200.0),
        "supports_10hz_p95": bool(p95 <= 100.0),
        "memory_process_snapshot_mib": {
            "after_warmup": before,
            "after_repeats": after,
            "interpretation": (
                "CPU process RSS/HWM, including loaded model and dataset cache; it is a practical "
                "process-level snapshot rather than per-inference tensor allocator memory."
            ),
        },
    }


def _latency_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"seeds": [int(row["seed"]) for row in rows]}
    for name in (
        "latency_p50_ms",
        "latency_p95_ms",
        "latency_mean_ms",
        "throughput_candidates_per_second",
    ):
        values = np.asarray([float(row["metrics"][name]) for row in rows], dtype=np.float64)
        result[name] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1 if values.size > 1 else 0)),
            "values": values.tolist(),
        }
    result["all_seeds_support_5hz_p95"] = bool(
        all(bool(row["metrics"]["supports_5hz_p95"]) for row in rows)
    )
    result["all_seeds_support_10hz_p95"] = bool(
        all(bool(row["metrics"]["supports_10hz_p95"]) for row in rows)
    )
    return result


def _latency_evaluation(
    *, config: Mapping[str, Any], models: Mapping[int, Any], datasets: Mapping[str, Dataset[Any]]
) -> dict[str, Any]:
    """Measure primary-verifier shared-scene inference for every frozen C-size."""

    report: dict[str, Any] = {
        "measurement": "single complete shared candidate world; CPU perf_counter around model forward",
        "counts": {},
    }
    for count in (5, 10, 20, 50):
        rows = [
            {
                "seed": int(seed),
                "metrics": _latency_one(
                    models[seed],
                    datasets[f"ranking_test/{count}"],
                    candidate_count=count,
                    config=config,
                ),
            }
            for seed in sorted(models)
        ]
        report["counts"][f"C{count}"] = {"per_seed": rows, "summary": _latency_summary(rows)}
    return report


def verify_inputs(
    config_path: Path = DEFAULT_CONFIG, *, output_root: Path = FULL_OUTPUT_DIR
) -> dict[str, Any]:
    """Read-only provenance check used before launching the expensive extension."""

    config = _load_config(Path(config_path))
    _assert_cpu_contract(config)
    torch.set_num_threads(int(config["training"]["num_threads"]))
    summary, selection = _selection_from_summary(output_root=Path(output_root), config=config)
    _, specs = _load_primary_models(config=config, output_root=Path(output_root), selection=selection)
    return {
        "status": "verified",
        "cpu_only": True,
        "config_digest": _json_digest(config),
        "summary": str(Path(output_root) / SUMMARY_FILE),
        "selected_primary": selection,
        "primary_checkpoints": [asdict(specs[seed]) for seed in sorted(specs)],
        "summary_decision": summary.get("decision"),
    }


def run(
    config_path: Path = DEFAULT_CONFIG,
    *,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    output_root: Path = FULL_OUTPUT_DIR,
) -> dict[str, Any]:
    """Run the post-completion CPU-only extensions without changing old artifacts."""

    output_path, output_root = Path(output_path), Path(output_root)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite completion extension artifact: {output_path}")
    if output_path.resolve() in {
        (output_root / SUMMARY_FILE).resolve(),
        (output_root / PRIMARY_SELECTION_FILE).resolve(),
    }:
        raise ValueError("completion extension output path must be distinct from frozen artifacts")
    config = _load_config(Path(config_path))
    _assert_cpu_contract(config)
    torch.set_num_threads(int(config["training"]["num_threads"]))
    summary, selection = _selection_from_summary(output_root=output_root, config=config)
    models, specs = _load_primary_models(config=config, output_root=output_root, selection=selection)
    datasets = build_datasets(config)

    point_permutation = _point_permutation_ranking(
        config=config,
        output_root=output_root,
        selection=selection,
        models=models,
        specs=specs,
        datasets=datasets,
    )
    mask = _mask_evaluation(config=config, models=models, datasets=datasets)
    correspondence = _correspondence_extension(
        config=config, output_root=output_root, datasets=datasets
    )
    latency = _latency_evaluation(config=config, models=models, datasets=datasets)
    report = {
        "milestone": "2E",
        "artifact": "bounded_completion_extensions",
        "status": "complete",
        "cpu_only": True,
        "config_digest": _json_digest(config),
        "source_full_summary": str(output_root / SUMMARY_FILE),
        "validation_selected_primary": selection,
        "primary_checkpoint_provenance": [asdict(specs[seed]) for seed in sorted(specs)],
        "frozen_protocol_assertions": {
            "final_non_smoke_summary_required": True,
            "validation_only_primary_selection_verified": True,
            "checkpoint_architecture_and_config_digest_verified": True,
            "cuda_unavailable_at_runtime": True,
            "no_training_or_checkpoint_write": True,
            "only_new_output": str(output_path),
        },
        "ranking_point_permutation": point_permutation,
        "mask_secondary_metrics": mask,
        "correspondence_static_and_median": correspondence,
        "shared_scene_latency": latency,
        "limitations": [
            "Point permutation preserves physical point identity across time by using one group-shared permutation; independent per-frame permutations remain separately reported in matching_quality.json.",
            "Mask IoU uses a fixed 0.50 threshold because no validation-selected mask threshold exists in the frozen 2E checkpoints; mask AP is threshold-independent.",
            "Mask secondary metrics are evaluated on the frozen ID C20 primary difficulty only; this extension does not add a separate mask OOD sweep or C5/C10/C50 mask sweep.",
            "Memory is a CPU process RSS/HWM snapshot and includes process-level cache/model state, not an isolated allocator measurement.",
            "The extension reports new diagnostics only and never revises the frozen selection, factorial result, prior milestone artifacts, or final-decision branch.",
        ],
        "source_summary_decision": summary.get("decision"),
    }
    _write_json(output_path, report)
    return report


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify-inputs",
        action="store_true",
        help="perform only final-artifact/checkpoint provenance checks; do not evaluate or write output",
    )
    arguments = parser.parse_args(argv)
    if arguments.verify_inputs:
        report = verify_inputs()
        print(report["status"])
        return report
    report = run()
    print(report["status"])
    return report


if __name__ == "__main__":
    main()
