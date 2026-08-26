"""Full CPU-only protocol for the Milestone 2E bottleneck study.

This runner is intentionally separate from the lightweight scoring audit in
``milestone2e.py``.  It never opens, mutates, or selects from a 2A/2B/2C
artifact.  Identity provenance is passed only to the explicitly diagnostic
ground-truth correspondence arm; every fair arm sees the strict observable
mapping alone.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import random
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import Dataset, Subset

from actmask.data.milestone2b_dataset import (
    MILESTONE2B_OOD_AXES,
    Milestone2BDataset,
    _clone_tensor_tree,
)
from actmask.data.milestone2c_dataset import (
    CorrespondenceCorruptionDataset,
    HardCandidateActionDataset,
    RigidTransformDataset,
    _orthogonal_transform,
)
from actmask.eval.calibration import apply_platt, fit_platt_scaling, reliability_metrics
from actmask.eval.statistics import bootstrap_ci
from actmask.experiments.milestone2c import (
    FAIR_BASELINES_2C,
    PROJECT_ROOT,
    _loader,
    _loss,
    _metadata_rows,
    _model_kwargs,
    _positive_weight,
    _seed_everything,
    predict_baseline,
)
from actmask.experiments.milestone2e import (
    DEFAULT_CONFIG,
    OUTPUT_DIR,
    RANKING_METRICS,
    RankingBundle,
    _load_config,
    _write_json,
    _write_text,
    audit_scoring_and_indexing,
    paired_ranking_report,
    ranking_metrics_tie_aware,
    validate_checkpoint_metadata,
)
from actmask.models.correspondence import _euclidean_cost, _legacy_axis_biased_cost
from actmask.models.milestone2e import (
    BACKBONE_VARIANTS,
    CORRESPONDENCE_VARIANTS,
    FactorialTemporalActMask,
)
from actmask.training.milestone2b_losses import Milestone2BLossCoefficients


FULL_OUTPUT_DIR = OUTPUT_DIR
PRIMARY_HEAD = "success_bce_plus_pairwise"
FAIR_BASELINE = "MultiHypothesisTrajectoryProximity"


@dataclass
class FactorialPrediction:
    """Strict-observable learned scores plus labels needed only by evaluation."""

    utility_probability: np.ndarray
    utility_logits: np.ndarray
    success: np.ndarray
    candidate_utility: np.ndarray
    metadata: list[dict[str, Any]]
    mask_probability: np.ndarray
    mask_target: np.ndarray


class DiagnosticIdentityDataset(Dataset[dict[str, Any]]):
    """Attach synthetic identity only for the excluded diagnostic arm.

    Point indices in ordinary 2B samples are their synthetic identities.  A
    correspondence corruption view already supplies its per-frame identity;
    this wrapper preserves that evaluator-only provenance.  The field remains
    under ``hidden_state`` and cannot enter the observable contract.
    """

    def __init__(self, dataset: Dataset[dict[str, Any]]) -> None:
        self.dataset = dataset
        self.group_records = getattr(dataset, "group_records", ())

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = _clone_tensor_tree(self.dataset[index])
        hidden = sample.setdefault("hidden_state", {})
        if "evaluation_identity_history" not in hidden:
            frames, points, _ = sample["observable"]["points_history"].shape
            hidden["evaluation_identity_history"] = torch.arange(
                points, dtype=torch.long
            ).repeat(frames, 1)
        return sample


def _json_digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def _dataset_kwargs(config: Mapping[str, Any]) -> dict[str, Any]:
    data = config["dataset"]
    return {
        "num_points": int(data["num_points"]),
        "observation_frames": int(data["observation_frames"]),
        "future_steps": int(data["future_steps"]),
        "observation_noise_std": float(data["observation_noise_std"]),
        "point_dropout_rate": float(data["point_dropout_rate"]),
        "occlusion_rate": float(data["occlusion_rate"]),
    }


def _identity_set(dataset: Dataset[Any]) -> set[int]:
    return {int(record.group_id) for record in getattr(dataset, "group_records", ())}


def build_datasets(config: Mapping[str, Any]) -> dict[str, Dataset[Any]]:
    """Build group-disjoint training, validation, ID and OOD ranking worlds."""

    data = config["dataset"]
    common = _dataset_kwargs(config)
    train_counts = {
        "train": int(data["train_groups"]),
        "val": int(data["val_groups"]),
        "test": 1,
    }
    point_eval_counts = {"train": 1, "val": 1, "test": int(data["id_test_groups"])}
    rank_val_counts = {"train": 1, "val": 1, "test": int(config["ranking"]["validation_groups"])}
    rank_test_counts = {"train": 1, "val": 1, "test": int(config["ranking"]["test_groups"])}
    train_seed, eval_seed = int(data["training_master_seed"]), int(data["evaluation_master_seed"])
    datasets: dict[str, Dataset[Any]] = {
        "train": DiagnosticIdentityDataset(Milestone2BDataset(
            split="train", master_seed=train_seed, split_counts=train_counts,
            ood_groups_per_axis=1, **common,
        )),
        "val": DiagnosticIdentityDataset(Milestone2BDataset(
            split="val", master_seed=train_seed, split_counts=train_counts,
            ood_groups_per_axis=1, **common,
        )),
        "point_test": DiagnosticIdentityDataset(Milestone2BDataset(
            split="test", master_seed=eval_seed, split_counts=point_eval_counts,
            ood_groups_per_axis=int(data["ood_groups_per_axis"]), **common,
        )),
    }
    for count in config["ranking"]["candidate_counts"]:
        number = int(count)
        datasets[f"ranking_val/{number}"] = DiagnosticIdentityDataset(HardCandidateActionDataset(
            candidate_count=number, split="test", master_seed=eval_seed + 1000 + number,
            split_counts=rank_val_counts, ood_groups_per_axis=1, **common,
        ))
        datasets[f"ranking_test/{number}"] = DiagnosticIdentityDataset(HardCandidateActionDataset(
            candidate_count=number, split="test", master_seed=eval_seed + 2000 + number,
            split_counts=rank_test_counts, ood_groups_per_axis=1, **common,
        ))
    # C20 is the frozen primary difficulty for OOD confirmation.  It uses
    # fresh candidate worlds and each domain's standard held-out construction.
    for axis in MILESTONE2B_OOD_AXES:
        datasets[f"ranking_ood/{axis}"] = DiagnosticIdentityDataset(HardCandidateActionDataset(
            candidate_count=20, split="test", domain=axis, master_seed=eval_seed + 3000,
            split_counts=rank_test_counts, ood_groups_per_axis=int(data["ood_groups_per_axis"]), **common,
        ))
    names = sorted(datasets)
    for offset, left_name in enumerate(names):
        left = _identity_set(datasets[left_name])
        for right_name in names[offset + 1 :]:
            overlap = left.intersection(_identity_set(datasets[right_name]))
            if overlap:
                raise ValueError(f"group leakage between {left_name} and {right_name}: {next(iter(overlap))}")
    return datasets


def _coefficients(head: str) -> Milestone2BLossCoefficients:
    if head == "success_bce":
        values = {"mask_bce": 1.0, "contact_time_bce": 0.20, "success_bce": 0.45}
    elif head == "pairwise":
        values = {"mask_bce": 1.0, "contact_time_bce": 0.20, "pairwise_ranking": 0.30}
    elif head == "success_bce_plus_pairwise":
        values = {"mask_bce": 1.0, "contact_time_bce": 0.20, "success_bce": 0.45, "pairwise_ranking": 0.30}
    else:
        raise ValueError(f"unknown ranking-head diagnostic {head!r}")
    fields = Milestone2BLossCoefficients.__dataclass_fields__
    return Milestone2BLossCoefficients(**{name: float(values.get(name, 0.0)) for name in fields})


def _model(config: Mapping[str, Any], correspondence: str, backbone: str) -> FactorialTemporalActMask:
    kwargs = _model_kwargs(config)
    return FactorialTemporalActMask(
        correspondence=correspondence,
        backbone=backbone,
        maximum_normalized_distance=float(config["correspondence"]["maximum_normalized_distance"]),
        soft_temperature=float(config["correspondence"]["soft_temperature"]),
        soft_minimum_peak_weight=float(config["correspondence"]["soft_minimum_peak_weight"]),
        **kwargs,
    ).cpu()


def _forward(model: FactorialTemporalActMask, batch: Mapping[str, Any]) -> dict[str, Tensor]:
    if model.correspondence == "GroundTruthIdentityCorrespondence":
        return model(
            batch["observable"],
            identity_history=batch["hidden_state"]["evaluation_identity_history"],
        )
    return model(batch["observable"])


@torch.no_grad()
def predict_factorial(
    model: FactorialTemporalActMask, dataset: Dataset[Any], *, batch_size: int
) -> FactorialPrediction:
    model.eval()
    utility_probability: list[np.ndarray] = []
    utility_logits: list[np.ndarray] = []
    success: list[np.ndarray] = []
    candidate_utility: list[np.ndarray] = []
    mask_probability: list[np.ndarray] = []
    mask_target: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    for batch in _loader(dataset, batch_size):
        output = _forward(model, batch)
        raw_utility = output["utility_logits"].detach().cpu().numpy()
        utility_logits.append(raw_utility)
        utility_probability.append(1.0 / (1.0 + np.exp(-raw_utility)))
        success.append(batch["targets"]["success"].detach().cpu().numpy())
        candidate_utility.append(batch["targets"]["candidate_utility"].detach().cpu().numpy())
        mask_probability.append(torch.sigmoid(output["mask_logits"]).detach().cpu().numpy())
        mask_target.append(batch["targets"]["ground_truth_future_mask"].detach().cpu().numpy())
        metadata.extend(_metadata_rows(batch["metadata"]))
    return FactorialPrediction(
        utility_probability=np.concatenate(utility_probability),
        utility_logits=np.concatenate(utility_logits),
        success=np.concatenate(success),
        candidate_utility=np.concatenate(candidate_utility),
        metadata=metadata,
        mask_probability=np.concatenate(mask_probability),
        mask_target=np.concatenate(mask_target),
    )


def _ranking_bundle(prediction: FactorialPrediction, scores: np.ndarray) -> RankingBundle:
    return RankingBundle(
        scores=np.asarray(scores, dtype=np.float64), success=prediction.success,
        utility=prediction.candidate_utility, metadata=prediction.metadata,
    )


def _normalize_by_group(scores: np.ndarray, metadata: Sequence[Mapping[str, Any]]) -> np.ndarray:
    result = np.zeros_like(np.asarray(scores, dtype=np.float64))
    grouped: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(metadata):
        grouped[int(row["candidate_set_id"])].append(index)
    for indices in grouped.values():
        value = np.asarray(scores)[indices]
        span = float(value.max() - value.min())
        result[indices] = 0.0 if span <= 1.0e-12 else (value - value.min()) / span
    return result


def _same_candidate_index(first: Sequence[Mapping[str, Any]], second: Sequence[Mapping[str, Any]]) -> None:
    left = [(int(row["candidate_set_id"]), int(row["candidate_id"])) for row in first]
    right = [(int(row["candidate_set_id"]), int(row["candidate_id"])) for row in second]
    if left != right:
        raise AssertionError("learned and fair predictions disagree on candidate ordering")


def _fair_prediction(dataset: Dataset[Any], *, batch_size: int) -> FactorialPrediction:
    # The frozen fair comparator has no learned parameters and is not tuned in
    # this study.  It receives the public observable mapping only.
    baseline = FAIR_BASELINES_2C[FAIR_BASELINE]().eval()
    prediction = predict_baseline(baseline, dataset, batch_size=batch_size)
    utility = np.asarray(prediction.utility, dtype=np.float64)
    bounded = np.clip(utility, 1.0e-5, 1.0 - 1.0e-5)
    return FactorialPrediction(
        utility_probability=utility,
        utility_logits=np.log(bounded / (1.0 - bounded)),
        success=np.asarray(prediction.success, dtype=np.float64),
        candidate_utility=np.asarray(prediction.candidate_utility, dtype=np.float64),
        metadata=prediction.metadata,
        mask_probability=np.asarray(prediction.probabilities, dtype=np.float64),
        mask_target=np.asarray(prediction.targets, dtype=np.float64),
    )


def _concatenate_factorial_predictions(parts: Sequence[FactorialPrediction]) -> FactorialPrediction:
    if not parts:
        raise ValueError("cannot concatenate zero predictions")
    return FactorialPrediction(
        utility_probability=np.concatenate([part.utility_probability for part in parts]),
        utility_logits=np.concatenate([part.utility_logits for part in parts]),
        success=np.concatenate([part.success for part in parts]),
        candidate_utility=np.concatenate([part.candidate_utility for part in parts]),
        metadata=[row for part in parts for row in part.metadata],
        mask_probability=np.concatenate([part.mask_probability for part in parts]),
        mask_target=np.concatenate([part.mask_target for part in parts]),
    )


def _hybrid_scores(
    learned: FactorialPrediction, fair: FactorialPrediction, calibration: Mapping[str, float]
) -> tuple[np.ndarray, np.ndarray]:
    _same_candidate_index(learned.metadata, fair.metadata)
    calibrated = apply_platt(learned.utility_probability, dict(calibration))
    score = 0.5 * _normalize_by_group(calibrated, learned.metadata) + 0.5 * _normalize_by_group(
        fair.utility_probability, fair.metadata
    )
    return score, calibrated


def _selection_metric(metrics: Mapping[str, Any]) -> tuple[float, float]:
    # Lower regret wins; top-1 is an entirely validation-only deterministic
    # secondary criterion.  No transformed or test view enters this choice.
    return (-float(metrics["normalized_regret"]), float(metrics["top1_success"]))


def train_factorial(
    *, config: Mapping[str, Any], datasets: Mapping[str, Dataset[Any]], correspondence: str,
    backbone: str, ranking_head: str, seed: int,
) -> tuple[FactorialTemporalActMask, dict[str, Any]]:
    """Train one fresh model and select its epoch only on validation C20."""

    _seed_everything(int(seed), int(config["training"]["num_threads"]))
    model = _model(config, correspondence, backbone)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    coefficients = _coefficients(ranking_head)
    positive_weight = _positive_weight(datasets["train"], float(config["training"]["max_pos_weight"]))
    fair_validation = _fair_prediction(
        datasets["ranking_val/20"], batch_size=int(config["training"]["evaluation_batch_size"])
    )
    best_state: dict[str, Tensor] | None = None
    best_calibration: dict[str, float] | None = None
    best_key = (-math.inf, -math.inf)
    best_epoch = 0
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        model.train()
        losses: list[float] = []
        for batch in _loader(datasets["train"], int(config["training"]["batch_size"])):
            optimizer.zero_grad(set_to_none=True)
            output = _forward(model, batch)
            loss, _ = _loss(coefficients, output, batch, positive_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        validation = predict_factorial(
            model, datasets["ranking_val/20"], batch_size=int(config["training"]["evaluation_batch_size"])
        )
        calibration = fit_platt_scaling(validation.utility_probability, validation.success)
        score, _ = _hybrid_scores(validation, fair_validation, calibration)
        metrics = ranking_metrics_tie_aware(_ranking_bundle(validation, score), score_name="frozen_hybrid_alpha_0.5")
        key = _selection_metric(metrics)
        rows.append({
            "epoch": epoch, "train_loss": float(np.mean(losses)),
            "validation_top1": float(metrics["top1_success"]),
            "validation_normalized_regret": float(metrics["normalized_regret"]),
            "platt_coefficient": float(calibration["coefficient"]),
            "platt_intercept": float(calibration["intercept"]),
        })
        if key > best_key:
            best_key, best_epoch = key, epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            best_calibration = dict(calibration)
    if best_state is None or best_calibration is None:
        raise RuntimeError("no validation checkpoint was selected")
    model.load_state_dict(best_state)
    digest = _json_digest(config)
    token = f"{correspondence}__{backbone}__{ranking_head}__seed{seed}"
    checkpoint = FULL_OUTPUT_DIR / "checkpoints" / f"{token}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    saved = {
        "architecture": "FactorialTemporalActMask", "correspondence": correspondence,
        "backbone": backbone, "ranking_head": ranking_head, "seed": int(seed),
        "config_digest": digest, "state_dict": best_state, "best_epoch": best_epoch,
        "validation_selection": {"metric": "C20 frozen hybrid normalized regret, then Top-1", "value": best_key},
        "calibration": best_calibration, "training_seconds": time.perf_counter() - started,
    }
    torch.save(saved, checkpoint)
    # Load-time provenance validation makes stale 2C or cross-arm checkpoints
    # a hard failure, rather than silently sharing a result.
    validate_checkpoint_metadata(saved, {
        "architecture": "FactorialTemporalActMask", "correspondence": correspondence,
        "backbone": backbone, "config_digest": digest,
    })
    _write_csv(FULL_OUTPUT_DIR / "training" / f"{token}.csv", rows)
    return model.eval(), {
        "token": token, "checkpoint": str(checkpoint), "correspondence": correspondence,
        "backbone": backbone, "ranking_head": ranking_head, "seed": int(seed),
        "best_epoch": best_epoch, "calibration": best_calibration,
        "validation_normalized_regret": -best_key[0], "validation_top1": best_key[1],
        "training_seconds": saved["training_seconds"],
    }


def load_or_train_factorial(
    *, config: Mapping[str, Any], datasets: Mapping[str, Dataset[Any]], correspondence: str,
    backbone: str, ranking_head: str, seed: int,
) -> tuple[FactorialTemporalActMask, dict[str, Any]]:
    """Resume only an exactly compatible 2E checkpoint, otherwise train it.

    A long CPU evaluation may be interrupted after all model selection has
    completed.  Reusing a checkpoint is safe only when its architecture,
    correspondence, backbone, ranking-head identity, seed and full config
    digest agree.  This does not permit reusing a stale 2C artifact.
    """

    token = f"{correspondence}__{backbone}__{ranking_head}__seed{seed}"
    checkpoint_path = FULL_OUTPUT_DIR / "checkpoints" / f"{token}.pt"
    digest = _json_digest(config)
    if not checkpoint_path.exists():
        return train_factorial(
            config=config, datasets=datasets, correspondence=correspondence,
            backbone=backbone, ranking_head=ranking_head, seed=seed,
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"checkpoint {checkpoint_path} is not a mapping")
    validate_checkpoint_metadata(checkpoint, {
        "architecture": "FactorialTemporalActMask", "correspondence": correspondence,
        "backbone": backbone, "ranking_head": ranking_head, "seed": int(seed),
        "config_digest": digest,
    })
    model = _model(config, correspondence, backbone)
    model.load_state_dict(checkpoint["state_dict"])
    calibration = checkpoint.get("calibration")
    if not isinstance(calibration, Mapping) or not {"coefficient", "intercept"}.issubset(calibration):
        raise ValueError(f"checkpoint {checkpoint_path} lacks validation-only calibration")
    selection = checkpoint.get("validation_selection", {})
    value = selection.get("value", (-math.inf, -math.inf)) if isinstance(selection, Mapping) else (-math.inf, -math.inf)
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f"checkpoint {checkpoint_path} has invalid validation selection")
    return model.eval(), {
        "token": token, "checkpoint": str(checkpoint_path), "correspondence": correspondence,
        "backbone": backbone, "ranking_head": ranking_head, "seed": int(seed),
        "best_epoch": int(checkpoint.get("best_epoch", 0)),
        "calibration": {"coefficient": float(calibration["coefficient"]), "intercept": float(calibration["intercept"])},
        "validation_normalized_regret": float(-float(value[0])),
        "validation_top1": float(value[1]),
        "training_seconds": float(checkpoint.get("training_seconds", 0.0)),
        "resumed_checkpoint": True,
    }


def _evaluate_ranking(
    model: FactorialTemporalActMask, dataset: Dataset[Any], fair: FactorialPrediction,
    calibration: Mapping[str, float], batch_size: int,
) -> tuple[dict[str, Any], FactorialPrediction]:
    learned = predict_factorial(model, dataset, batch_size=batch_size)
    score, calibrated = _hybrid_scores(learned, fair, calibration)
    metrics = ranking_metrics_tie_aware(_ranking_bundle(learned, score), score_name="frozen_hybrid_alpha_0.5")
    metrics["utility_calibration"] = reliability_metrics(calibrated, learned.success, bins=15)
    return metrics, learned


def _aggregate_seed_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"seeds": [int(row["seed"]) for row in rows]}
    for metric in RANKING_METRICS:
        values = np.asarray([float(row["metrics"][metric]) for row in rows], dtype=np.float64)
        result[metric] = {"mean": float(values.mean()), "std": float(values.std(ddof=1 if values.size > 1 else 0)), "values": values.tolist()}
    return result


def _correspondence_quality(
    model: FactorialTemporalActMask, dataset: Dataset[Any], *, batch_size: int
) -> dict[str, Any]:
    """Source-ID quality and confidence reliability, before any ranking score."""

    if model.correspondence == "NoExplicitCorrespondence":
        # It is an explicit set-aggregation control, not a method making source
        # identity claims.  Do not turn no-match into a fictitious zero accuracy.
        accepted = rejected = anchors = 0
        with torch.no_grad():
            for batch in _loader(dataset, batch_size):
                result = model.associate(batch["observable"])
                anchors += int(batch["observable"]["visibility_history"][:, -1].sum()) * (result.aligned_history.shape[1] - 1)
                accepted += int((result.source_index[:, :-1] >= 0).sum())
                rejected += int((result.source_index[:, :-1] < 0).sum())
        return {
            "association_type": "set_aggregation_no_explicit_source_identity",
            "mutual_match_rate": None, "rejected_unmatched_rate": float(rejected / max(anchors, 1)),
            "duplicate_match_rate": None, "correspondence_error_mean": None,
            "correspondence_error_p95": None, "correspondence_accuracy": None,
            "dynamic_point_correspondence_accuracy": None, "accepted_matches": accepted,
            "identity_claiming": False,
        }
    counts = defaultdict(int)
    errors: list[float] = []
    confidences: list[float] = []
    confidence_correct: list[float] = []
    by_velocity: list[tuple[float, float]] = []
    by_density: list[tuple[float, float]] = []
    by_ambiguity: list[tuple[float, float]] = []
    with torch.no_grad():
        for batch in _loader(dataset, batch_size):
            observable = batch["observable"]
            result = model.associate(
                observable,
                identity_history=batch["hidden_state"]["evaluation_identity_history"]
                if model.correspondence == "GroundTruthIdentityCorrespondence" else None,
            )
            history, visible = observable["points_history"], observable["visibility_history"].bool()
            identity = batch["hidden_state"]["evaluation_identity_history"].long()
            exact_velocity = batch["hidden_state"]["exact_point_velocity"]
            batch_size_now, frames, points, _ = history.shape
            anchors_visible = visible[:, -1]
            dynamic = torch.linalg.vector_norm(exact_velocity, dim=-1) > 0.02
            anchor = history[:, -1]
            pair_distance = torch.cdist(anchor, anchor).masked_fill(torch.eye(points, dtype=torch.bool)[None], float("inf"))
            density = 1.0 / pair_distance.amin(dim=-1).clamp_min(1.0e-5)
            for frame in range(frames - 1):
                source_id = identity[:, frame]
                same = identity[:, -1, :, None] == source_id[:, None, :]
                expected = same.long().argmax(dim=-1)
                matchable = anchors_visible & same.any(dim=-1) & visible[:, frame].gather(1, expected)
                accepted = result.source_index[:, frame] >= 0
                correct = accepted & matchable & (result.source_index[:, frame] == expected)
                counts["anchors"] += int(anchors_visible.sum())
                counts["mutual"] += int((result.mutual[:, frame] & anchors_visible).sum())
                counts["rejected"] += int(((~accepted) & anchors_visible).sum())
                counts["matchable"] += int(matchable.sum())
                counts["correct"] += int(correct.sum())
                counts["dynamic_matchable"] += int((matchable & dynamic).sum())
                counts["dynamic_correct"] += int((correct & dynamic).sum())
                used = result.source_index[:, frame]
                for row in used:
                    valid = row[row >= 0]
                    counts["accepted"] += int(valid.numel())
                    counts["duplicate_excess"] += int(valid.numel() - torch.unique(valid).numel())
                selected = history[:, frame].gather(1, result.source_index[:, frame].clamp_min(0)[..., None].expand(batch_size_now, points, 3))
                truth = history[:, frame].gather(1, expected[..., None].expand(batch_size_now, points, 3))
                valid_error = accepted & matchable
                if valid_error.any():
                    errors.extend(torch.linalg.vector_norm(selected - truth, dim=-1)[valid_error].tolist())
                confidence = result.confidence
                confidences.extend(confidence[matchable].detach().cpu().tolist())
                confidence_correct.extend(correct[matchable].to(torch.float32).detach().cpu().tolist())
                cost = torch.cdist(anchor, history[:, frame]).masked_fill(~visible[:, frame, None, :], float("inf"))
                nearest = cost.amin(dim=-1)
                ambiguity = (cost.topk(k=min(2, points), largest=False).values[..., 1] - nearest) / nearest.clamp_min(1.0e-5)
                selected_anchor = matchable
                by_velocity.extend(zip(torch.linalg.vector_norm(exact_velocity, dim=-1)[selected_anchor].tolist(), correct[selected_anchor].to(torch.float32).tolist(), strict=True))
                by_density.extend(zip(density[selected_anchor].tolist(), correct[selected_anchor].to(torch.float32).tolist(), strict=True))
                by_ambiguity.extend(zip(ambiguity[selected_anchor].tolist(), correct[selected_anchor].to(torch.float32).tolist(), strict=True))

    def binned(records: Sequence[tuple[float, float]], name: str) -> list[dict[str, float | int]]:
        if not records:
            return []
        value = np.asarray([row[0] for row in records], dtype=np.float64)
        correct = np.asarray([row[1] for row in records], dtype=np.float64)
        edges = np.unique(np.quantile(value, [0.0, 1 / 3, 2 / 3, 1.0]))
        if edges.size < 2:
            return [{"bin": name, "lower": float(value.min()), "upper": float(value.max()), "count": int(value.size), "accuracy": float(correct.mean())}]
        output: list[dict[str, float | int]] = []
        for index in range(edges.size - 1):
            include = (value >= edges[index]) & (value <= edges[index + 1] if index + 1 == edges.size - 1 else value < edges[index + 1])
            output.append({"bin": f"{name}_{index + 1}", "lower": float(edges[index]), "upper": float(edges[index + 1]), "count": int(include.sum()), "accuracy": float(correct[include].mean()) if include.any() else 0.0})
        return output

    confidence_report = reliability_metrics(confidences, confidence_correct, bins=10) if confidences else None
    return {
        "association_type": model.correspondence, "identity_claiming": True,
        "mutual_match_rate": counts["mutual"] / max(counts["anchors"], 1),
        "rejected_unmatched_rate": counts["rejected"] / max(counts["anchors"], 1),
        "duplicate_match_rate": counts["duplicate_excess"] / max(counts["accepted"], 1),
        "correspondence_error_mean": float(np.mean(errors)) if errors else None,
        "correspondence_error_p95": float(np.quantile(errors, 0.95)) if errors else None,
        "correspondence_accuracy": counts["correct"] / max(counts["matchable"], 1),
        "dynamic_point_correspondence_accuracy": counts["dynamic_correct"] / max(counts["dynamic_matchable"], 1),
        "accepted_matches": counts["accepted"], "matchable_points": counts["matchable"],
        "confidence_reliability": confidence_report,
        "dynamic_accuracy_by_velocity": binned(by_velocity, "velocity"),
        "accuracy_by_density": binned(by_density, "density"),
        "accuracy_by_ambiguity": binned(by_ambiguity, "ambiguity"),
    }


def _association_equivariance(
    model: FactorialTemporalActMask, dataset: Dataset[Any], *, mode: str, batch_size: int
) -> dict[str, Any]:
    """Direct correspondence-level equivariance, independent of ranking."""

    transformed = DiagnosticIdentityDataset(RigidTransformDataset(dataset, mode=mode))
    index_equal = visibility_equal = True
    max_distance_error = max_history_error = max_velocity_error = 0.0
    for source, changed in zip(_loader(dataset, batch_size), _loader(transformed, batch_size), strict=True):
        identity = source["hidden_state"]["evaluation_identity_history"] if model.correspondence == "GroundTruthIdentityCorrespondence" else None
        reference = model.associate(source["observable"], identity_history=identity)
        identity_changed = changed["hidden_state"]["evaluation_identity_history"] if model.correspondence == "GroundTruthIdentityCorrespondence" else None
        converted = model.associate(changed["observable"], identity_history=identity_changed)
        index_equal = index_equal and bool(torch.equal(reference.source_index, converted.source_index))
        visibility_equal = visibility_equal and bool(torch.equal(reference.aligned_visibility, converted.aligned_visibility))
        finite = torch.isfinite(reference.normalized_distance)
        if finite.any():
            max_distance_error = max(max_distance_error, float((reference.normalized_distance[finite] - converted.normalized_distance[finite]).abs().max()))
        matrices = torch.stack([
            _orthogonal_transform(20260911, int(group), mode)
            for group in source["metadata"]["candidate_set_id"].tolist()
        ])
        expected_history = torch.einsum("bfpi,bji->bfpj", reference.aligned_history, matrices)
        expected_velocity = torch.einsum("bpi,bji->bpj", reference.velocity, matrices)
        max_history_error = max(max_history_error, float((converted.aligned_history - expected_history).abs().max()))
        max_velocity_error = max(max_velocity_error, float((converted.velocity - expected_velocity).abs().max()))
    return {
        "mode": mode, "source_index_equal": index_equal, "visibility_equal": visibility_equal,
        "max_normalized_distance_error": max_distance_error,
        "max_aligned_history_equivariance_error": max_history_error,
        "max_velocity_equivariance_error": max_velocity_error,
        "passed": bool(index_equal and visibility_equal and max_distance_error <= 3.0e-5 and max_history_error <= 5.0e-5 and max_velocity_error <= 5.0e-5),
    }


def _negative_control() -> dict[str, Any]:
    anchors = torch.tensor([[[0.0, 0.0, 0.0]]])
    source = torch.tensor([[[0.15, 0.0, 0.0], [0.0, 0.20, 0.0]]])
    swap = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    rotation = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    legacy = _legacy_axis_biased_cost(anchors, source).argmin(dim=-1)
    legacy_permuted = _legacy_axis_biased_cost(anchors @ swap.T, source @ swap.T).argmin(dim=-1)
    legacy_rotated = _legacy_axis_biased_cost(anchors @ rotation.T, source @ rotation.T).argmin(dim=-1)
    euclidean = _euclidean_cost(anchors, source).argmin(dim=-1)
    euclidean_permuted = _euclidean_cost(anchors @ swap.T, source @ swap.T).argmin(dim=-1)
    return {
        "legacy_formula": "|dx| + 0.28 * ||d_yz||", "diagnostic_only": True,
        "axis_permutation_detected": bool(not torch.equal(legacy, legacy_permuted)),
        "rotation_detected": bool(not torch.equal(legacy, legacy_rotated)),
        "euclidean_is_invariant": bool(torch.equal(euclidean, euclidean_permuted)),
    }


def _rank_view(dataset: Dataset[Any], view: str) -> Dataset[Any]:
    if view == "original":
        return dataset
    if view in {"random_so3", "axis_permutation", "sign_flip"}:
        return DiagnosticIdentityDataset(RigidTransformDataset(dataset, mode=view))
    if view == "resampled":
        return DiagnosticIdentityDataset(CorrespondenceCorruptionDataset(dataset, mode="per_frame_resampling", seed=20261031))
    if view == "occluded":
        return DiagnosticIdentityDataset(CorrespondenceCorruptionDataset(dataset, mode="full_part_occlusion", seed=20261032))
    raise ValueError(view)


def _plot_matching_quality(report: Mapping[str, Any]) -> str | None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    rows = report.get("CoordinateInvariantMutualNN", {}).get("clean", {}).get("dynamic_accuracy_by_velocity", [])
    if not rows:
        return None
    figure, axis = plt.subplots(figsize=(5.0, 3.1))
    axis.bar([str(row["bin"]) for row in rows], [float(row["accuracy"]) for row in rows])
    axis.set(xlabel="dynamic velocity tercile", ylabel="correspondence accuracy", ylim=(0.0, 1.02))
    figure.tight_layout()
    path = FULL_OUTPUT_DIR / "figures" / "matching_dynamic_accuracy_by_velocity.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return str(path)


def _latency(model: FactorialTemporalActMask, dataset: Dataset[Any], config: Mapping[str, Any]) -> dict[str, float | int]:
    batch = next(iter(_loader(dataset, int(config["runtime"]["candidates"]))))
    warmup, repeats = int(config["runtime"]["warmup"]), int(config["runtime"]["repeats"])
    with torch.no_grad():
        for _ in range(warmup):
            _forward(model, batch)
        samples: list[float] = []
        for _ in range(repeats):
            started = time.perf_counter()
            _forward(model, batch)
            samples.append((time.perf_counter() - started) * 1000.0)
    array = np.asarray(samples)
    candidates = int(batch["targets"]["success"].shape[0])
    return {
        "candidates": candidates, "latency_p50_ms": float(np.quantile(array, 0.50)),
        "latency_p95_ms": float(np.quantile(array, 0.95)),
        "throughput_candidates_per_second": float(candidates / (array.mean() / 1000.0)),
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
    }


def _decision(
    *, selected: Mapping[str, Any], factorial: Mapping[str, Any], invariance: Mapping[str, Any], ood: Mapping[str, Any]
) -> dict[str, Any]:
    gt = factorial.get("GroundTruthIdentityCorrespondence", {})
    estimated = factorial.get(str(selected["correspondence"]), {})
    gt_value = gt.get(str(selected["backbone"]), {}).get("C20", {}).get("top1_success", {}).get("mean")
    estimated_value = estimated.get(str(selected["backbone"]), {}).get("C20", {}).get("top1_success", {}).get("mean")
    gap = float(gt_value - estimated_value) if gt_value is not None and estimated_value is not None else None
    transformations_pass = all(
        abs(float(row["metrics"]["top1_success"]) - float(row["original"]["top1_success"])) <= 0.02
        and abs(float(row["metrics"]["normalized_regret"]) - float(row["original"]["normalized_regret"])) <= 0.02
        for row in invariance.get("selected_seed_rows", [])
    )
    ood_positive = sum(float(row["metrics"]["top1_success"]) >= float(row["id_c20_top1"]) for row in ood.get("axis_seed_rows", []))
    if transformations_pass and ood_positive >= 2 and (gap is None or gap <= 0.05):
        branch = "GO FOR GPU SIMULATOR INTEGRATION"
    elif gap is not None and gap > 0.05:
        branch = "STOP POINT-CORRESPONDENCE LINE"
    elif transformations_pass:
        branch = "PIVOT TO ANALYTIC VERIFIER"
    else:
        branch = "STOP CANDIDATE-VERIFIER LINE"
    return {
        "technical_branch": branch, "estimated_vs_gt_top1_gap_C20": gap,
        "transformation_gate_passed": transformations_pass,
        "ood_positive_axis_seed_count": ood_positive,
        "project_go_no_go": "UNCHANGED: NO-GO FOR GPU SIMULATOR INTEGRATION",
        "reason": "Milestone 2E does not recompute every earlier frozen pivot gate under the unchanged prior protocol.",
    }


def _markdown(report: Mapping[str, Any]) -> str:
    selected = report["validation_selected_primary"]
    decision = report["decision"]
    lines = [
        "# Milestone 2E Results", "",
        f"**Technical branch:** `{decision['technical_branch']}`.", "",
        f"**Project decision:** `{decision['project_go_no_go']}`.", "",
        "## Frozen protocol", "",
        "- CPU only; grouped disjoint train/validation/test candidate worlds; five deterministic seeds.",
        "- C5/C10/C20/C50, validation-only epoch selection and Platt calibration; score is the frozen 0.5 learned / 0.5 MultiHypothesis hybrid.",
        "- GT identity is diagnostic-only and excluded from fair comparisons; transformed tests never select a checkpoint.", "",
        "## Scoring audit", "",
        f"- Passed: `{report['scoring_audit']['passed']}`; candidate labels/IDs are aligned and equal-score ties never use candidate ID.",
        f"- Selected fair arm from validation only: `{selected['correspondence']} + {selected['backbone']}`.", "",
        "## Correspondence formulas", "",
        "- Old diagnostic-only rule: `|dx| + 0.28 * ||d_yz||`.",
        "- New primary rule: with anchor `a_t`, observable motion `v`, and earlier frame time `s`, choose mutual nearest source `j` minimizing `||a_t - v(t-s) - x_{s,j}||_2`, accepting only confidence and normalized-distance passing matches; otherwise source index is `-1`.",
        "- Soft arm uses the same full-3D predicted cost in a confidence-gated soft assignment; no axis is privileged.", "",
        "## Main result", "",
        f"- Estimated-vs-GT diagnostic C20 Top-1 gap: `{decision['estimated_vs_gt_top1_gap_C20']}`.",
        f"- Negative control detects both an axis permutation and rotation: `{report['negative_control']['axis_permutation_detected']}` / `{report['negative_control']['rotation_detected']}`.",
        "",
        "Detailed factorial, matching, ranking-invariance, OOD, latency, per-seed and paired-bootstrap results are stored in the JSON/CSV files beside this summary.",
    ]
    return "\n".join(lines)


def run(config_path: Path = DEFAULT_CONFIG, *, smoke: bool = False) -> dict[str, Any]:
    config = copy.deepcopy(_load_config(config_path))
    if len(config["experiment"]["training_seeds"]) != 5:
        raise ValueError("2E full protocol requires exactly five fixed deterministic seeds")
    if list(config["ranking"]["candidate_counts"]) != [5, 10, 20, 50]:
        raise ValueError("2E full protocol requires C5/C10/C20/C50")
    # A fully resumed run can load every checkpoint and never enter
    # ``train_factorial``, which is otherwise where the configured CPU thread
    # limit is applied.  Set it at the protocol boundary as well so resumed
    # evaluation neither inherits an arbitrary host-wide thread count nor
    # changes its CPU execution contract.
    torch.set_num_threads(int(config["training"]["num_threads"]))
    if smoke:
        # Explicit local development mode only; artifacts remain separated and
        # its result cannot be used as the full-protocol decision.
        config["experiment"]["training_seeds"] = [int(config["experiment"]["training_seeds"][0])]
        config["dataset"]["id_test_groups"] = 3
        config["dataset"]["ood_groups_per_axis"] = 2
        config["ranking"]["validation_groups"] = 3
        config["ranking"]["test_groups"] = 3
        config["training"]["epochs"] = 1
        config["runtime"]["warmup"], config["runtime"]["repeats"] = 1, 2
    FULL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    audit = audit_scoring_and_indexing(config)
    _write_json(FULL_OUTPUT_DIR / "scoring_indexing_audit.json", audit)
    datasets = build_datasets(config)
    seeds = [int(value) for value in config["experiment"]["training_seeds"]]
    models: dict[tuple[str, str, str, int], FactorialTemporalActMask] = {}
    records: list[dict[str, Any]] = []

    # Complete correspondence x representation factorial, retrained from
    # scratch.  This is the main comparison; every fair arm retains all seeds.
    for correspondence in CORRESPONDENCE_VARIANTS:
        for backbone in BACKBONE_VARIANTS:
            for seed in seeds:
                model, record = load_or_train_factorial(
                    config=config, datasets=datasets, correspondence=correspondence,
                    backbone=backbone, ranking_head=PRIMARY_HEAD, seed=seed,
                )
                models[(correspondence, backbone, PRIMARY_HEAD, seed)] = model
                records.append(record)
    _write_json(FULL_OUTPUT_DIR / "training_records_primary.json", records)

    validation_rows = [row for row in records if row["correspondence"] != "GroundTruthIdentityCorrespondence"]
    grouped_validation: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in validation_rows:
        grouped_validation[(str(row["correspondence"]), str(row["backbone"]))].append(row)
    selected_key = min(
        grouped_validation,
        key=lambda key: (
            float(np.mean([row["validation_normalized_regret"] for row in grouped_validation[key]])),
            -float(np.mean([row["validation_top1"] for row in grouped_validation[key]])),
            key,
        ),
    )
    selected = {"correspondence": selected_key[0], "backbone": selected_key[1], "selection": "mean five-seed validation C20 normalized regret, then Top-1"}
    _write_json(FULL_OUTPUT_DIR / "validation_primary_selection.json", selected)

    # Ranking-head diagnostic is constrained to the validation-selected fair
    # architecture.  It cannot alter the correspondence/backbone comparison.
    head_records: list[dict[str, Any]] = []
    for head in ("success_bce", "pairwise"):
        for seed in seeds:
            model, record = load_or_train_factorial(
                config=config, datasets=datasets, correspondence=selected_key[0],
                backbone=selected_key[1], ranking_head=head, seed=seed,
            )
            models[(selected_key[0], selected_key[1], head, seed)] = model
            head_records.append(record)
    _write_json(FULL_OUTPUT_DIR / "training_records_ranking_head.json", head_records)

    batch_size = int(config["training"]["evaluation_batch_size"])
    # The fair baseline is expensive on 500 C50 worlds but independent of all
    # learned arms.  Cache each frozen candidate-count view independently, so
    # an interrupted CPU run resumes without silently recomputing it.
    fair_cache_dir = FULL_OUTPUT_DIR / "fair_cache"
    fair_cache_dir.mkdir(parents=True, exist_ok=True)
    resume_digest = _json_digest(config)
    fair_test: dict[int, FactorialPrediction] = {}
    for count in config["ranking"]["candidate_counts"]:
        number = int(count)
        fair_path = fair_cache_dir / f"{FAIR_BASELINE}__C{number}.pt"
        if fair_path.exists():
            saved_fair = torch.load(fair_path, map_location="cpu", weights_only=False)
            if not isinstance(saved_fair, Mapping) or saved_fair.get("config_digest") != resume_digest:
                raise ValueError(f"incompatible frozen fair cache: {fair_path}")
            cached_prediction = saved_fair.get("prediction")
            if not isinstance(cached_prediction, FactorialPrediction):
                raise ValueError(f"invalid frozen fair cache: {fair_path}")
            fair_test[number] = cached_prediction
        else:
            # C50 can outlive an interactive CPU foreground lease.  Save
            # small deterministic slices so no completed observable-only
            # baseline computation is discarded on a later resume.
            dataset = datasets[f"ranking_test/{number}"]
            chunk_size = 500
            parts: list[FactorialPrediction] = []
            for start in range(0, len(dataset), chunk_size):
                stop = min(len(dataset), start + chunk_size)
                chunk_path = fair_cache_dir / f"{FAIR_BASELINE}__C{number}__{start:06d}_{stop:06d}.pt"
                if chunk_path.exists():
                    saved_chunk = torch.load(chunk_path, map_location="cpu", weights_only=False)
                    if not isinstance(saved_chunk, Mapping) or saved_chunk.get("config_digest") != resume_digest:
                        raise ValueError(f"incompatible frozen fair chunk: {chunk_path}")
                    prediction = saved_chunk.get("prediction")
                    if not isinstance(prediction, FactorialPrediction):
                        raise ValueError(f"invalid frozen fair chunk: {chunk_path}")
                else:
                    prediction = _fair_prediction(Subset(dataset, range(start, stop)), batch_size=batch_size)
                    torch.save({"config_digest": resume_digest, "prediction": prediction}, chunk_path)
                parts.append(prediction)
            fair_test[number] = _concatenate_factorial_predictions(parts)
            torch.save({"config_digest": resume_digest, "prediction": fair_test[number]}, fair_path)
    factorial_rows: list[dict[str, Any]] = []
    factorial_summary: dict[str, Any] = defaultdict(lambda: defaultdict(dict))
    incremental_dir = FULL_OUTPUT_DIR / "factorial_incremental"
    incremental_dir.mkdir(parents=True, exist_ok=True)
    config_digest = _json_digest(config)
    for correspondence in CORRESPONDENCE_VARIANTS:
        for backbone in BACKBONE_VARIANTS:
            for count in config["ranking"]["candidate_counts"]:
                per_seed: list[dict[str, Any]] = []
                for seed in seeds:
                    model = models[(correspondence, backbone, PRIMARY_HEAD, seed)]
                    record = next(item for item in records if item["correspondence"] == correspondence and item["backbone"] == backbone and item["seed"] == seed)
                    saved_path = incremental_dir / f"{correspondence}__{backbone}__C{int(count)}__seed{seed}.json"
                    if saved_path.exists():
                        saved_row = _read_json(saved_path)
                        expected = {
                            "config_digest": config_digest, "correspondence": correspondence,
                            "backbone": backbone, "candidate_count": int(count), "seed": seed,
                        }
                        if any(saved_row.get(key) != value for key, value in expected.items()):
                            raise ValueError(f"incompatible incremental factorial result: {saved_path}")
                        row = {
                            "correspondence": correspondence, "backbone": backbone,
                            "candidate_count": int(count), "seed": seed,
                            "metrics": saved_row["metrics"],
                        }
                    else:
                        metrics, _ = _evaluate_ranking(model, datasets[f"ranking_test/{count}"], fair_test[int(count)], record["calibration"], batch_size)
                        row = {"correspondence": correspondence, "backbone": backbone, "candidate_count": int(count), "seed": seed, "metrics": metrics}
                        _write_json(saved_path, {"config_digest": config_digest, **row})
                    factorial_rows.append(row)
                    per_seed.append(row)
                factorial_summary[correspondence][backbone][f"C{count}"] = _aggregate_seed_rows(per_seed)
    _write_json(FULL_OUTPUT_DIR / "factorial_ranking_per_seed.json", factorial_rows)
    _write_json(FULL_OUTPUT_DIR / "factorial_ranking_summary.json", factorial_summary)

    # Ranking-label/head isolation, evaluated only after its explicit training
    # choice.  It does not replace the predeclared primary head.
    head_report: dict[str, Any] = {}
    for head in ("success_bce", "pairwise", PRIMARY_HEAD):
        arm_rows: list[dict[str, Any]] = []
        for seed in seeds:
            model = models[(selected_key[0], selected_key[1], head, seed)]
            source = records if head == PRIMARY_HEAD else head_records
            record = next(item for item in source if item["ranking_head"] == head and item["seed"] == seed and item["correspondence"] == selected_key[0] and item["backbone"] == selected_key[1])
            metrics, _ = _evaluate_ranking(model, datasets["ranking_test/20"], fair_test[20], record["calibration"], batch_size)
            arm_rows.append({"seed": seed, "metrics": metrics})
        head_report[head] = _aggregate_seed_rows(arm_rows)
    _write_json(FULL_OUTPUT_DIR / "ranking_head_diagnostic.json", head_report)

    # Matching quality is measured before a learned ranking is considered.
    matching_views: dict[str, Dataset[Any]] = {
        "clean": datasets["point_test"],
        "noise": DiagnosticIdentityDataset(CorrespondenceCorruptionDataset(datasets["point_test"], mode="nearest_neighbor_noise", seed=20261041)),
        "per_frame_permutation": DiagnosticIdentityDataset(CorrespondenceCorruptionDataset(datasets["point_test"], mode="per_frame_permutation", seed=20261042)),
        "resampling": DiagnosticIdentityDataset(CorrespondenceCorruptionDataset(datasets["point_test"], mode="per_frame_resampling", seed=20261043)),
        "occlusion": DiagnosticIdentityDataset(CorrespondenceCorruptionDataset(datasets["point_test"], mode="full_part_occlusion", seed=20261044)),
        "rotation_resampling": DiagnosticIdentityDataset(RigidTransformDataset(CorrespondenceCorruptionDataset(datasets["point_test"], mode="per_frame_resampling", seed=20261043), mode="random_so3")),
        "axis_occlusion": DiagnosticIdentityDataset(RigidTransformDataset(CorrespondenceCorruptionDataset(datasets["point_test"], mode="full_part_occlusion", seed=20261044), mode="axis_permutation")),
    }
    matching: dict[str, Any] = {}
    correspondence_tests: dict[str, Any] = {}
    for correspondence in CORRESPONDENCE_VARIANTS:
        probe = models[(correspondence, "ScalarInvariantFeatureBackbone", PRIMARY_HEAD, seeds[0])]
        matching[correspondence] = {name: _correspondence_quality(probe, view, batch_size=batch_size) for name, view in matching_views.items()}
        correspondence_tests[correspondence] = {
            mode: _association_equivariance(probe, matching_views["per_frame_permutation"], mode=mode, batch_size=batch_size)
            for mode in ("random_so3", "axis_permutation", "sign_flip")
        }
    _write_json(FULL_OUTPUT_DIR / "matching_quality.json", matching)
    _write_json(FULL_OUTPUT_DIR / "correspondence_equivariance.json", correspondence_tests)
    matching_plot = _plot_matching_quality(matching)
    negative = _negative_control()
    if not (negative["axis_permutation_detected"] and negative["rotation_detected"] and negative["euclidean_is_invariant"]):
        raise AssertionError("axis-biased negative control did not validate the invariance test")
    _write_json(FULL_OUTPUT_DIR / "negative_control.json", negative)

    # Invariance is separately a ranking test.  First expose vector/scalar/
    # action-frame representation sensitivity at the selected correspondence.
    views = ("original", "random_so3", "axis_permutation", "sign_flip", "resampled", "occluded")
    representation_rows: list[dict[str, Any]] = []
    for backbone in BACKBONE_VARIANTS:
        for seed in seeds:
            model = models[(selected_key[0], backbone, PRIMARY_HEAD, seed)]
            record = next(item for item in records if item["correspondence"] == selected_key[0] and item["backbone"] == backbone and item["seed"] == seed)
            original: dict[str, Any] | None = None
            for view in views:
                data_view = _rank_view(datasets["ranking_test/20"], view)
                fair = _fair_prediction(data_view, batch_size=batch_size)
                metrics, _ = _evaluate_ranking(model, data_view, fair, record["calibration"], batch_size)
                if original is None:
                    original = metrics
                representation_rows.append({"backbone": backbone, "seed": seed, "view": view, "metrics": metrics, "original": original})
    _write_json(FULL_OUTPUT_DIR / "representation_ranking_invariance.json", representation_rows)

    # The validation-selected fair primary receives every frozen candidate-set
    # size and every required transformed observation view, with group-paired
    # CIs against its own original-world result.
    selected_seed_rows: list[dict[str, Any]] = []
    final_invariance: dict[str, Any] = {}
    for count in config["ranking"]["candidate_counts"]:
        count_rows: list[dict[str, Any]] = []
        for seed in seeds:
            model = models[(selected_key[0], selected_key[1], PRIMARY_HEAD, seed)]
            record = next(item for item in records if item["correspondence"] == selected_key[0] and item["backbone"] == selected_key[1] and item["seed"] == seed)
            original: dict[str, Any] | None = None
            for index, view in enumerate(views):
                data_view = _rank_view(datasets[f"ranking_test/{count}"], view)
                fair = _fair_prediction(data_view, batch_size=batch_size)
                metrics, _ = _evaluate_ranking(model, data_view, fair, record["calibration"], batch_size)
                if original is None:
                    original = metrics
                    paired = None
                else:
                    paired = paired_ranking_report(
                        metrics, original, seed=20261100 + 100 * int(count) + 10 * seed + index,
                        resamples=int(config["statistics"]["bootstrap_resamples"]), permutations=int(config["statistics"]["permutation_samples"]),
                    )
                row = {"candidate_count": int(count), "seed": seed, "view": view, "metrics": metrics, "original": original, "paired_view_minus_original": paired}
                count_rows.append(row)
                if int(count) == 20 and view != "original":
                    selected_seed_rows.append(row)
        final_invariance[f"C{count}"] = count_rows
    _write_json(FULL_OUTPUT_DIR / "final_ranking_invariance.json", final_invariance)

    # OOD uses unchanged C20 construction and reports grouped bootstrap CIs per
    # held-out domain.  It is kept distinct from the paired transformed views.
    ood_rows: list[dict[str, Any]] = []
    id_rows = [row for row in factorial_rows if row["correspondence"] == selected_key[0] and row["backbone"] == selected_key[1] and row["candidate_count"] == 20]
    id_by_seed = {int(row["seed"]): row for row in id_rows}
    for axis in MILESTONE2B_OOD_AXES:
        dataset = datasets[f"ranking_ood/{axis}"]
        fair = _fair_prediction(dataset, batch_size=batch_size)
        for seed in seeds:
            model = models[(selected_key[0], selected_key[1], PRIMARY_HEAD, seed)]
            record = next(item for item in records if item["correspondence"] == selected_key[0] and item["backbone"] == selected_key[1] and item["seed"] == seed)
            metrics, _ = _evaluate_ranking(model, dataset, fair, record["calibration"], batch_size)
            top1_ci = bootstrap_ci([float(row["top1_success"]) for row in metrics["per_group"]], seed=20261200 + seed, resamples=int(config["statistics"]["bootstrap_resamples"]))
            ood_rows.append({"axis": axis, "seed": seed, "metrics": metrics, "top1_bootstrap": top1_ci, "id_c20_top1": float(id_by_seed[seed]["metrics"]["top1_success"])})
    ood = {"axis_seed_rows": ood_rows}
    _write_json(FULL_OUTPUT_DIR / "ood_ranking.json", ood)

    latency = {
        backbone: _latency(models[(selected_key[0], backbone, PRIMARY_HEAD, seeds[0])], datasets["ranking_test/20"], config)
        for backbone in BACKBONE_VARIANTS
    }
    _write_json(FULL_OUTPUT_DIR / "latency.json", latency)
    invariance_for_decision = {"selected_seed_rows": selected_seed_rows}
    decision = _decision(selected=selected, factorial=factorial_summary, invariance=invariance_for_decision, ood=ood)
    report = {
        "milestone": "2E", "cpu_only": True, "smoke": bool(smoke), "scoring_audit": audit,
        "validation_selected_primary": selected, "training_records": records,
        "ranking_head_training_records": head_records, "factorial_summary": factorial_summary,
        "ranking_head_diagnostic": head_report, "matching_quality": matching,
        "matching_plot": matching_plot, "correspondence_equivariance": correspondence_tests,
        "negative_control": negative, "final_ranking_invariance_file": "final_ranking_invariance.json",
        "representation_invariance_file": "representation_ranking_invariance.json",
        "ood_file": "ood_ranking.json", "latency": latency, "decision": decision,
        "artifact_integrity": "Only outputs/actmask/milestone2e was written; 2A/2B/2C were not overwritten.",
    }
    _write_json(FULL_OUTPUT_DIR / "summary.json", report)
    _write_text(FULL_OUTPUT_DIR / "summary.md", _markdown(report))
    _write_text(PROJECT_ROOT / "docs" / "milestone2e_results.md", _markdown(report))
    return report


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="explicitly reduced non-decision development run")
    arguments = parser.parse_args(argv)
    report = run(smoke=arguments.smoke)
    print(report["decision"]["technical_branch"])
    return report


if __name__ == "__main__":
    main()
