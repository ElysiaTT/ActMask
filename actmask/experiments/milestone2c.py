"""CPU-only Milestone 2C synthetic-study closure experiment.

The runner deliberately separates validation selection from expanded test
evaluation.  It writes only ``outputs/actmask/milestone2c`` and preserves all
earlier milestones as frozen provenance references.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from actmask.data.milestone2b_dataset import (
    MILESTONE2B_OOD_AXES,
    SAMPLES_PER_GROUP,
    Milestone2BDataset,
    milestone2b_dataset_statistics,
)
from actmask.data.milestone2c_dataset import (
    CORRESPONDENCE_MODES,
    CorrespondenceCorruptionDataset,
    HardCandidateActionDataset,
    candidate_set_statistics,
)
from actmask.eval.calibration import (
    apply_platt,
    apply_temperature,
    fit_platt_scaling,
    fit_temperature_scaling,
    reliability_metrics,
    risk_coverage_curve,
)
from actmask.eval.pr_metrics import precision_recall_curve, select_validation_threshold
from actmask.eval.statistics import bootstrap_ci, paired_summary
from actmask.models.correspondence import (
    IndexCorrespondenceTemporalActMask,
    InvariantFeatureMotionTemporalActMask,
    LegacyAxisBiasedNeighborhoodTemporalActMask,
    LocalNeighborhoodTemporalActMask,
    MotionConsistentMutualTemporalActMask,
    SetHistoryTemporalActMask,
)
from actmask.models.milestone2c_baselines import (
    FAIR_BASELINES_2C,
    GeometryWithLearnedResidual,
)
from actmask.training.milestone2b_losses import (
    Milestone2BLossCoefficients,
    base_supervised_losses,
    counterfactual_assignment_loss,
    irrelevant_background_stability_loss,
    pairwise_candidate_ranking_loss,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "actmask" / "milestone2c_cpu.yaml"
FROZEN_CONFIG = PROJECT_ROOT / "configs" / "actmask" / "milestone2b_frozen_cpu.yaml"
OUTPUT_NAME = "milestone2c"


LOSS_SPECS: tuple[tuple[str, dict[str, float]], ...] = (
    ("mask_only", {"mask_bce": 1.0}),
    ("mask_contact", {"mask_bce": 1.0, "contact_time_bce": 0.20}),
    (
        "mask_utility",
        {"mask_bce": 1.0, "success_bce": 0.45, "pairwise_ranking": 0.30},
    ),
    (
        "mask_contact_utility",
        {
            "mask_bce": 1.0,
            "contact_time_bce": 0.20,
            "success_bce": 0.45,
            "pairwise_ranking": 0.30,
        },
    ),
    (
        "mask_contact_utility_velocity_cf",
        {
            "mask_bce": 1.0,
            "contact_time_bce": 0.20,
            "success_bce": 0.45,
            "pairwise_ranking": 0.30,
            "velocity_counterfactual": 0.25,
        },
    ),
    (
        "mask_contact_utility_action_cf",
        {
            "mask_bce": 1.0,
            "contact_time_bce": 0.20,
            "success_bce": 0.45,
            "pairwise_ranking": 0.30,
            "action_counterfactual": 0.25,
        },
    ),
    (
        "mask_contact_utility_irrelevant",
        {
            "mask_bce": 1.0,
            "contact_time_bce": 0.20,
            "success_bce": 0.45,
            "pairwise_ranking": 0.30,
            "irrelevant_stability": 0.10,
        },
    ),
    (
        "mask_all_counterfactual",
        {
            "mask_bce": 1.0,
            "velocity_counterfactual": 0.25,
            "action_counterfactual": 0.25,
            "irrelevant_stability": 0.10,
        },
    ),
    (
        "full_2b",
        {
            "mask_bce": 1.0,
            "contact_time_bce": 0.20,
            "success_bce": 0.45,
            "pairwise_ranking": 0.30,
            "velocity_counterfactual": 0.25,
            "action_counterfactual": 0.25,
            "irrelevant_stability": 0.10,
        },
    ),
)


@dataclass
class PredictionBundle:
    logits: np.ndarray
    probabilities: np.ndarray
    targets: np.ndarray
    contact_utility: np.ndarray
    utility: np.ndarray
    metadata: list[dict[str, Any]]
    hard_positive: np.ndarray
    hard_negative: np.ndarray
    success: np.ndarray
    candidate_utility: np.ndarray


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Tensor):
        return value.item() if value.numel() == 1 else value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set)):
        return [_jsonable(item) for item in value]
    return value


def _digest(value: Any) -> str:
    encoded = json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n")


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n")


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
            writer.writerow({key: _jsonable(row.get(key, "")) for key in keys})


def _seed_everything(seed: int, threads: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(int(threads))
    torch.use_deterministic_algorithms(True)


def _load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict):
        raise ValueError("Milestone 2C configuration must be a mapping")
    if config["experiment"]["device"] != "cpu":
        raise ValueError("Milestone 2C is strictly CPU-only")
    seeds = list(config["experiment"]["training_seeds"])
    if len(seeds) < 5 or len(set(seeds)) != len(seeds):
        raise ValueError("Milestone 2C requires at least five unique training seeds")
    dataset = config["dataset"]
    if int(dataset["id_test_groups"]) < 500 or int(dataset["ood_groups_per_axis"]) < 250:
        raise ValueError("expanded evaluation must use >=500 ID and >=250 OOD groups")
    if int(config["training"]["batch_size"]) % SAMPLES_PER_GROUP:
        raise ValueError("training batch size must contain complete 11-sample groups")
    return config


def _loader(dataset: Dataset[Any], batch_size: int) -> DataLoader[Any]:
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)


def _metadata_rows(metadata: Mapping[str, Any]) -> list[dict[str, Any]]:
    size = int(metadata["group_id"].shape[0])
    records: list[dict[str, Any]] = []
    for index in range(size):
        record: dict[str, Any] = {}
        for key, value in metadata.items():
            if isinstance(value, Tensor):
                item = value[index]
                record[key] = item.item() if item.numel() == 1 else item.tolist()
            else:
                record[key] = value[index]
        records.append(record)
    return records


def _ordinary_mask(metadata: Mapping[str, Any]) -> Tensor:
    value = metadata.get("is_irrelevant_perturbation")
    if isinstance(value, Tensor):
        return ~value.bool()
    return torch.ones(len(metadata["group_id"]), dtype=torch.bool)


def _model_kwargs(config: Mapping[str, Any], *, frames: int | None = None, horizon: int | None = None) -> dict[str, Any]:
    model = config["model"]
    dataset = config["dataset"]
    return {
        "observation_frames": int(frames if frames is not None else dataset["observation_frames"]),
        "horizon_steps": int(horizon if horizon is not None else dataset["future_steps"]),
        "history_hidden_dim": int(model["history_hidden_dim"]),
        "action_hidden_dim": int(model["action_hidden_dim"]),
        "fusion_hidden_dim": int(model["fusion_hidden_dim"]),
        "contact_hidden_dim": int(model["contact_hidden_dim"]),
        "utility_hidden_dim": int(model["utility_hidden_dim"]),
    }


def build_model(architecture: str, config: Mapping[str, Any], *, utility_head: bool = True) -> nn.Module:
    kwargs = _model_kwargs(config)
    kwargs["utility_head"] = bool(utility_head)
    if architecture == "IndexCorrespondenceTemporalActMask":
        return IndexCorrespondenceTemporalActMask(**kwargs)
    if architecture == "SetHistoryTemporalActMask":
        return SetHistoryTemporalActMask(**kwargs)
    if architecture == "LocalNeighborhoodTemporalActMask":
        return LocalNeighborhoodTemporalActMask(**kwargs)
    if architecture == "MotionConsistentMutualTemporalActMask":
        return MotionConsistentMutualTemporalActMask(**kwargs)
    if architecture == "LegacyAxisBiasedNeighborhoodTemporalActMask":
        return LegacyAxisBiasedNeighborhoodTemporalActMask(**kwargs)
    if architecture == "InvariantFeatureMotionTemporalActMask":
        return InvariantFeatureMotionTemporalActMask(**kwargs)
    raise ValueError(f"unknown architecture {architecture!r}")


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


def _identity_sets(dataset: Dataset[Any]) -> dict[str, set[int]]:
    records = getattr(dataset, "group_records", ())
    return {
        field: {int(getattr(record, field)) for record in records}
        for field in (
            "group_id",
            "base_scene_id",
            "pair_id",
            "trajectory_family_id",
            "geometry_seed",
            "scenario_seed",
        )
    }


def build_datasets(config: Mapping[str, Any]) -> dict[str, Dataset[Any]]:
    """Build compact train/val and seed-namespace-separated large evaluation."""

    data = config["dataset"]
    common = _dataset_kwargs(config)
    train_counts = {
        "train": int(data["train_groups"]),
        "val": int(data["val_groups"]),
        "test": 1,
    }
    eval_counts = {"train": 1, "val": 1, "test": int(data["id_test_groups"])}
    train_master = int(data["training_master_seed"])
    eval_master = int(data["evaluation_master_seed"])
    datasets: dict[str, Dataset[Any]] = {
        "train": Milestone2BDataset(
            split="train", master_seed=train_master, split_counts=train_counts,
            ood_groups_per_axis=1, **common,
        ),
        "val": Milestone2BDataset(
            split="val", master_seed=train_master, split_counts=train_counts,
            ood_groups_per_axis=1, **common,
        ),
        "test": Milestone2BDataset(
            split="test", master_seed=eval_master, split_counts=eval_counts,
            ood_groups_per_axis=int(data["ood_groups_per_axis"]), **common,
        ),
    }
    for axis in MILESTONE2B_OOD_AXES:
        datasets[f"ood/{axis}"] = Milestone2BDataset(
            split="test", domain=axis, master_seed=eval_master,
            split_counts=eval_counts,
            ood_groups_per_axis=int(data["ood_groups_per_axis"]), **common,
        )
    rank_counts_val = {"train": 1, "val": 1, "test": int(config["ranking"]["validation_groups"])}
    rank_counts_test = {"train": 1, "val": 1, "test": int(config["ranking"]["test_groups"])}
    for count in config["ranking"]["candidate_counts"]:
        datasets[f"ranking_val/{count}"] = HardCandidateActionDataset(
            candidate_count=int(count), split="test", master_seed=eval_master + 101 + int(count),
            split_counts=rank_counts_val, ood_groups_per_axis=1, **common,
        )
        datasets[f"ranking_test/{count}"] = HardCandidateActionDataset(
            candidate_count=int(count), split="test", master_seed=eval_master + 201 + int(count),
            split_counts=rank_counts_test, ood_groups_per_axis=1, **common,
        )
    names = sorted(datasets)
    for left_index, left_name in enumerate(names):
        left = _identity_sets(datasets[left_name])
        for right_name in names[left_index + 1 :]:
            right = _identity_sets(datasets[right_name])
            for field in left:
                overlap = left[field].intersection(right[field])
                if overlap:
                    raise ValueError(
                        f"{field} leaks between {left_name} and {right_name}: {next(iter(overlap))}"
                    )
    return datasets


def _positive_weight(dataset: Dataset[Any], maximum: float) -> Tensor:
    positives = 0
    total = 0
    for index in range(len(dataset)):
        sample = dataset[index]
        if sample["metadata"].get("is_irrelevant_perturbation", False):
            continue
        target = sample["targets"]["ground_truth_future_mask"]
        positives += int(target.sum())
        total += int(target.numel())
    return torch.tensor(min(float(maximum), (total - positives) / max(positives, 1)))


def _batch_group_indices(metadata: Mapping[str, Any]) -> list[dict[str, int]]:
    group_ids = metadata["group_id"].detach().cpu().tolist()
    variants = metadata["dynamics_variant"].detach().cpu().tolist()
    types = list(metadata["candidate_type"])
    irrelevant = metadata["is_irrelevant_perturbation"].detach().cpu().tolist()
    groups: dict[int, dict[str, int]] = defaultdict(dict)
    for index, (group, variant, candidate_type, is_irrelevant) in enumerate(
        zip(group_ids, variants, types, irrelevant, strict=True)
    ):
        if bool(is_irrelevant):
            groups[int(group)]["irrelevant"] = index
        else:
            groups[int(group)][f"{int(variant)}:{candidate_type}"] = index
    return [groups[key] for key in sorted(groups)]


def _loss(coefficients: Milestone2BLossCoefficients, output: Mapping[str, Tensor], batch: Mapping[str, Any], pos_weight: Tensor) -> tuple[Tensor, dict[str, float]]:
    targets = batch["targets"]
    metadata = batch["metadata"]
    keep = _ordinary_mask(metadata)
    supervised = base_supervised_losses(
        mask_logits=output["mask_logits"][keep],
        contact_logits=output["contact_logits"][keep],
        utility_logits=output["utility_logits"][keep],
        mask_targets=targets["ground_truth_future_mask"][keep],
        contact_targets=targets["contact_matrix"][keep],
        success_targets=targets["success"][keep],
        positive_weight=pos_weight,
    )
    candidate_set = metadata["candidate_set_id"][keep].detach().cpu().tolist()
    terms: dict[str, Tensor] = {
        **supervised,
        "pairwise_ranking": pairwise_candidate_ranking_loss(
            output["utility_logits"][keep], targets["success"][keep], candidate_set
        ),
    }
    zero = output["mask_logits"].sum() * 0.0
    velocity_terms: list[Tensor] = []
    action_terms: list[Tensor] = []
    irrelevant_terms: list[Tensor] = []
    for indices in _batch_group_indices(metadata):
        required = {"0:successful", "1:successful", "0:wrong_timing", "irrelevant"}
        if not required.issubset(indices):
            continue
        reference = indices["0:successful"]
        velocity = indices["1:successful"]
        action = indices["0:wrong_timing"]
        irrelevant = indices["irrelevant"]
        velocity_terms.append(
            counterfactual_assignment_loss(
                output["mask_logits"][reference], output["mask_logits"][velocity],
                targets["ground_truth_future_mask"][reference],
                targets["ground_truth_future_mask"][velocity],
            )
        )
        action_terms.append(
            counterfactual_assignment_loss(
                output["mask_logits"][reference], output["mask_logits"][action],
                targets["ground_truth_future_mask"][reference],
                targets["ground_truth_future_mask"][action],
            )
        )
        irrelevant_terms.append(
            irrelevant_background_stability_loss(
                output["mask_logits"][reference], output["mask_logits"][irrelevant],
                targets["ground_truth_future_mask"][reference],
                targets["ground_truth_future_mask"][irrelevant],
            )
        )
    terms["velocity_counterfactual"] = torch.stack(velocity_terms).mean() if velocity_terms else zero
    terms["action_counterfactual"] = torch.stack(action_terms).mean() if action_terms else zero
    terms["irrelevant_stability"] = torch.stack(irrelevant_terms).mean() if irrelevant_terms else zero
    weights = coefficients.to_dict()
    total = sum(weights[name] * value for name, value in terms.items())
    return total, {name: float(value.detach()) for name, value in terms.items()}


def _perturb_observable(observable: Mapping[str, Tensor], perturbation: str | None) -> dict[str, Tensor]:
    result = {key: value.clone() for key, value in observable.items()}
    if perturbation is None:
        return result
    if perturbation == "history":
        for key in (
            "points_history", "visibility_history", "timestamps", "estimated_velocity",
            "velocity_confidence", "observation_delay",
        ):
            result[key] = torch.roll(result[key], shifts=5, dims=0)
    elif perturbation == "action":
        for key in ("action_command", "nominal_action_delay"):
            result[key] = torch.roll(result[key], shifts=1, dims=0)
    else:
        raise ValueError(f"unknown observable perturbation {perturbation!r}")
    return result


@torch.no_grad()
def predict_model(
    model: nn.Module,
    dataset: Dataset[Any],
    *,
    batch_size: int,
    feature_mode: str = "full",
    perturbation: str | None = None,
) -> PredictionBundle:
    model.eval()
    logits: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    contact: list[np.ndarray] = []
    utility: list[np.ndarray] = []
    hard_positive: list[np.ndarray] = []
    hard_negative: list[np.ndarray] = []
    success: list[np.ndarray] = []
    candidate_utility: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    for batch in _loader(dataset, batch_size):
        observable = _perturb_observable(batch["observable"], perturbation)
        output = model(observable, feature_mode=feature_mode)
        raw = output["mask_logits"].detach().cpu().numpy()
        probability = 1.0 / (1.0 + np.exp(-raw))
        per_point_contact = torch.sigmoid(output["contact_logits"]).amax(dim=-1)
        topk = min(8, per_point_contact.shape[1])
        logits.append(raw)
        probabilities.append(probability)
        targets.append(batch["targets"]["ground_truth_future_mask"].numpy())
        contact.append(per_point_contact.topk(topk, dim=1).values.mean(dim=1).numpy())
        utility.append(torch.sigmoid(output["utility_logits"]).detach().cpu().numpy())
        hard_positive.append(batch["targets"]["hard_positive_mask"].numpy())
        hard_negative.append(batch["targets"]["hard_negative_mask"].numpy())
        success.append(batch["targets"]["success"].numpy())
        candidate_utility.append(batch["targets"]["candidate_utility"].numpy())
        metadata.extend(_metadata_rows(batch["metadata"]))
    return PredictionBundle(
        logits=np.concatenate(logits), probabilities=np.concatenate(probabilities),
        targets=np.concatenate(targets), contact_utility=np.concatenate(contact),
        utility=np.concatenate(utility), metadata=metadata,
        hard_positive=np.concatenate(hard_positive), hard_negative=np.concatenate(hard_negative),
        success=np.concatenate(success), candidate_utility=np.concatenate(candidate_utility),
    )


@torch.no_grad()
def predict_baseline(model: nn.Module, dataset: Dataset[Any], *, batch_size: int) -> PredictionBundle:
    model.eval()
    logits: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    utility: list[np.ndarray] = []
    hard_positive: list[np.ndarray] = []
    hard_negative: list[np.ndarray] = []
    success: list[np.ndarray] = []
    candidate_utility: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    for batch in _loader(dataset, batch_size):
        output = model(batch["observable"])
        raw = output["mask_logits"].detach().cpu().numpy()
        logits.append(raw)
        probabilities.append(1.0 / (1.0 + np.exp(-raw)))
        targets.append(batch["targets"]["ground_truth_future_mask"].numpy())
        score = output["utility_score"].detach().cpu().numpy()
        utility.append(score)
        hard_positive.append(batch["targets"]["hard_positive_mask"].numpy())
        hard_negative.append(batch["targets"]["hard_negative_mask"].numpy())
        success.append(batch["targets"]["success"].numpy())
        candidate_utility.append(batch["targets"]["candidate_utility"].numpy())
        metadata.extend(_metadata_rows(batch["metadata"]))
    values = np.concatenate(utility)
    return PredictionBundle(
        logits=np.concatenate(logits), probabilities=np.concatenate(probabilities),
        targets=np.concatenate(targets), contact_utility=values.copy(), utility=values,
        metadata=metadata, hard_positive=np.concatenate(hard_positive),
        hard_negative=np.concatenate(hard_negative), success=np.concatenate(success),
        candidate_utility=np.concatenate(candidate_utility),
    )


def _flat_metrics(probability: np.ndarray, target: np.ndarray, threshold: float) -> dict[str, float]:
    score = np.asarray(probability, dtype=np.float64).reshape(-1)
    label = np.asarray(target, dtype=np.float64).reshape(-1) >= 0.5
    prediction = score >= float(threshold)
    tp = int(np.logical_and(prediction, label).sum())
    fp = int(np.logical_and(prediction, ~label).sum())
    fn = int(np.logical_and(~prediction, label).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    curve = precision_recall_curve(score, label.astype(np.float64))
    return {
        "average_precision": float(curve.average_precision), "pr_auc": float(curve.pr_auc),
        "iou": float(iou), "precision": float(precision), "recall": float(recall), "f1": float(f1),
    }


def select_threshold(bundle: PredictionBundle, *, temperature: dict[str, float] | None = None) -> dict[str, Any]:
    keep = np.asarray([not bool(row.get("is_irrelevant_perturbation", False)) for row in bundle.metadata])
    probabilities = (
        apply_temperature(bundle.logits, temperature) if temperature is not None else bundle.probabilities
    )
    selection = select_validation_threshold(probabilities[keep], bundle.targets[keep])
    return {**selection.to_dict(), "temperature": temperature}


def calibrated_probability(bundle: PredictionBundle, temperature: dict[str, float] | None) -> np.ndarray:
    return apply_temperature(bundle.logits, temperature) if temperature else bundle.probabilities


def _coefficients(loss_name: str) -> Milestone2BLossCoefficients:
    for name, values in LOSS_SPECS:
        if name == loss_name:
            return Milestone2BLossCoefficients(**{field: float(values.get(field, 0.0)) for field in Milestone2BLossCoefficients.__dataclass_fields__})
    raise ValueError(f"unknown loss specification {loss_name!r}")


def _bundle_keep(bundle: PredictionBundle) -> np.ndarray:
    return np.asarray(
        [not bool(row.get("is_irrelevant_perturbation", False)) for row in bundle.metadata],
        dtype=np.bool_,
    )


def _validation_ap(bundle: PredictionBundle) -> float:
    keep = _bundle_keep(bundle)
    return float(
        precision_recall_curve(
            bundle.probabilities[keep].reshape(-1), bundle.targets[keep].reshape(-1)
        ).average_precision
    )


def train_model(
    *,
    architecture: str,
    loss_name: str,
    seed: int,
    config: Mapping[str, Any],
    datasets: Mapping[str, Dataset[Any]],
    output_dir: Path,
) -> tuple[nn.Module, dict[str, Any]]:
    """Train one deterministic learned configuration and retain its best val AP."""

    _seed_everything(int(seed), int(config["training"]["num_threads"]))
    model = build_model(architecture, config).cpu()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    coefficients = _coefficients(loss_name)
    positive_weight = _positive_weight(
        datasets["train"], float(config["training"]["max_pos_weight"])
    )
    best_state: dict[str, Tensor] | None = None
    best_ap, best_epoch = -math.inf, 0
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        model.train()
        loss_total, batches = 0.0, 0
        terms_total: dict[str, float] = defaultdict(float)
        for batch in _loader(datasets["train"], int(config["training"]["batch_size"])):
            optimizer.zero_grad(set_to_none=True)
            output = model(batch["observable"], feature_mode="full")
            loss, terms = _loss(coefficients, output, batch, positive_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_total += float(loss.detach())
            batches += 1
            for name, value in terms.items():
                terms_total[name] += value
        validation = predict_model(
            model, datasets["val"],
            batch_size=int(config["training"]["evaluation_batch_size"]),
        )
        val_ap = _validation_ap(validation)
        row = {
            "epoch": epoch,
            "train_loss": loss_total / max(1, batches),
            "validation_ap": val_ap,
            **{f"loss_{name}": value / max(1, batches) for name, value in terms_total.items()},
        }
        rows.append(row)
        if val_ap > best_ap:
            best_ap, best_epoch = val_ap, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("no learned checkpoint was selected")
    model.load_state_dict(best_state)
    token = f"{architecture}_{loss_name}_seed{seed}"
    checkpoint = output_dir / "checkpoints" / f"{token}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "architecture": architecture,
            "loss_name": loss_name,
            "seed": int(seed),
            "model_kwargs": _model_kwargs(config),
            "state_dict": best_state,
            "best_epoch": best_epoch,
            "best_validation_ap": best_ap,
            "loss_coefficients": coefficients.to_dict(),
        },
        checkpoint,
    )
    _write_csv(output_dir / "training" / f"{token}.csv", rows)
    return model, {
        "architecture": architecture,
        "loss_name": loss_name,
        "seed": int(seed),
        "checkpoint": str(checkpoint),
        "best_epoch": best_epoch,
        "best_validation_ap": best_ap,
        "positive_weight": float(positive_weight),
        "loss_coefficients": coefficients.to_dict(),
        "runtime_seconds": time.perf_counter() - started,
    }


def _residual_loss(output: Mapping[str, Tensor], batch: Mapping[str, Any], positive_weight: Tensor) -> Tensor:
    keep = _ordinary_mask(batch["metadata"])
    return F.binary_cross_entropy_with_logits(
        output["mask_logits"][keep],
        batch["targets"]["ground_truth_future_mask"][keep],
        pos_weight=positive_weight,
    )


def train_residual_baseline(
    *, seed: int, config: Mapping[str, Any], datasets: Mapping[str, Dataset[Any]], output_dir: Path
) -> tuple[GeometryWithLearnedResidual, dict[str, Any]]:
    """Fit the one learned fair residual using the same public train/val data."""

    _seed_everything(int(seed) + 7919, int(config["training"]["num_threads"]))
    model = GeometryWithLearnedResidual().cpu()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.004, weight_decay=1.0e-4)
    positive_weight = _positive_weight(datasets["train"], float(config["training"]["max_pos_weight"]))
    best_state: dict[str, Tensor] | None = None
    best_ap, best_epoch = -math.inf, 0
    rows: list[dict[str, Any]] = []
    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        model.train()
        total, count = 0.0, 0
        for batch in _loader(datasets["train"], int(config["training"]["batch_size"])):
            optimizer.zero_grad(set_to_none=True)
            loss = _residual_loss(model(batch["observable"]), batch, positive_weight)
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
            count += 1
        val_ap = _validation_ap(
            predict_baseline(model, datasets["val"], batch_size=int(config["training"]["evaluation_batch_size"]))
        )
        rows.append({"epoch": epoch, "train_loss": total / max(1, count), "validation_ap": val_ap})
        if val_ap > best_ap:
            best_ap, best_epoch = val_ap, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("no residual-baseline checkpoint was selected")
    model.load_state_dict(best_state)
    checkpoint = output_dir / "checkpoints" / f"GeometryWithLearnedResidual_seed{seed}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"seed": seed, "state_dict": best_state, "best_epoch": best_epoch, "best_validation_ap": best_ap}, checkpoint)
    _write_csv(output_dir / "training" / f"GeometryWithLearnedResidual_seed{seed}.csv", rows)
    return model, {"seed": seed, "checkpoint": str(checkpoint), "best_epoch": best_epoch, "best_validation_ap": best_ap}


def _average_bundles(bundles: Sequence[PredictionBundle]) -> PredictionBundle:
    if not bundles:
        raise ValueError("cannot average an empty prediction list")
    first = bundles[0]
    for bundle in bundles[1:]:
        if bundle.targets.shape != first.targets.shape or bundle.metadata != first.metadata:
            raise ValueError("only identically ordered prediction bundles can be averaged")
    return PredictionBundle(
        logits=np.mean([bundle.logits for bundle in bundles], axis=0),
        probabilities=np.mean([bundle.probabilities for bundle in bundles], axis=0),
        targets=first.targets, contact_utility=np.mean([bundle.contact_utility for bundle in bundles], axis=0),
        utility=np.mean([bundle.utility for bundle in bundles], axis=0), metadata=first.metadata,
        hard_positive=first.hard_positive, hard_negative=first.hard_negative,
        success=first.success, candidate_utility=first.candidate_utility,
    )


def _grouped_indices(bundle: PredictionBundle, *, include_irrelevant: bool = False) -> dict[int, list[int]]:
    groups: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(bundle.metadata):
        if include_irrelevant or not bool(row.get("is_irrelevant_perturbation", False)):
            groups[int(row["group_id"])].append(index)
    return dict(sorted(groups.items()))


def group_mask_metrics(
    bundle: PredictionBundle, *, threshold: float, probability: np.ndarray | None = None
) -> dict[str, Any]:
    """Macro group metrics; group is the independent statistical unit."""

    values = bundle.probabilities if probability is None else probability
    rows: list[dict[str, Any]] = []
    for group_id, indices in _grouped_indices(bundle).items():
        metric = _flat_metrics(values[indices], bundle.targets[indices], threshold)
        prediction = values[indices] >= threshold
        hard_positive = bundle.hard_positive[indices].astype(bool)
        hard_negative = bundle.hard_negative[indices].astype(bool)
        metric.update(
            {
                "group_id": group_id,
                "hard_positive_recall": float(prediction[hard_positive].mean()) if hard_positive.any() else 0.0,
                "hard_negative_specificity": float((~prediction[hard_negative]).mean()) if hard_negative.any() else 0.0,
                "hard_positive_points": int(hard_positive.sum()),
                "hard_negative_points": int(hard_negative.sum()),
            }
        )
        rows.append(metric)
    aggregate = {
        key: float(np.mean([row[key] for row in rows]))
        for key in (
            "average_precision", "pr_auc", "iou", "precision", "recall", "f1",
            "hard_positive_recall", "hard_negative_specificity",
        )
    }
    aggregate.update({"groups": len(rows), "per_group": rows})
    return aggregate


def _assignment_match(
    first_probability: np.ndarray,
    second_probability: np.ndarray,
    first_target: np.ndarray,
    second_target: np.ndarray,
) -> float | None:
    if np.array_equal(first_target, second_target):
        return None
    epsilon = 1.0e-7
    first = np.clip(first_probability, epsilon, 1.0 - epsilon)
    second = np.clip(second_probability, epsilon, 1.0 - epsilon)
    correct = float(
        -np.mean(first_target * np.log(first) + (1.0 - first_target) * np.log(1.0 - first))
        -np.mean(second_target * np.log(second) + (1.0 - second_target) * np.log(1.0 - second))
    )
    swapped = float(
        -np.mean(second_target * np.log(first) + (1.0 - second_target) * np.log(1.0 - first))
        -np.mean(first_target * np.log(second) + (1.0 - first_target) * np.log(1.0 - second))
    )
    return 0.5 if math.isclose(correct, swapped, abs_tol=1.0e-10) else float(correct < swapped)


def counterfactual_metrics(bundle: PredictionBundle) -> dict[str, Any]:
    """Counterfactual identification and irrelevant perturbation stability."""

    index: dict[int, dict[str, int]] = defaultdict(dict)
    for sample, row in enumerate(bundle.metadata):
        key = int(row["group_id"])
        if bool(row.get("is_irrelevant_perturbation", False)):
            index[key]["irrelevant"] = sample
        else:
            index[key][f"{int(row['dynamics_variant'])}:{row['candidate_type']}"] = sample
    velocity: list[float] = []
    action: list[float] = []
    stability: list[float] = []
    for values in index.values():
        needed = {"0:successful", "1:successful", "0:wrong_timing", "irrelevant"}
        if not needed.issubset(values):
            continue
        reference = values["0:successful"]
        velocity_index = values["1:successful"]
        action_index = values["0:wrong_timing"]
        irrelevant = values["irrelevant"]
        for recipient, other in ((velocity, velocity_index), (action, action_index)):
            match = _assignment_match(
                bundle.probabilities[reference], bundle.probabilities[other],
                bundle.targets[reference], bundle.targets[other],
            )
            if match is not None:
                recipient.append(match)
        stability.append(
            1.0 - float(np.mean(np.abs(bundle.probabilities[reference] - bundle.probabilities[irrelevant])))
        )
    return {
        "velocity_counterfactual_matching_accuracy": float(np.mean(velocity)) if velocity else 0.0,
        "velocity_pairs": len(velocity),
        "action_counterfactual_matching_accuracy": float(np.mean(action)) if action else 0.0,
        "action_pairs": len(action),
        "irrelevant_perturbation_stability": float(np.mean(stability)) if stability else 0.0,
        "irrelevant_pairs": len(stability),
    }


def _ranking_group_rows(
    bundle: PredictionBundle, scores: np.ndarray, *, score_name: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group_id, indices in _grouped_indices(bundle).items():
        ordered = sorted(indices, key=lambda index: (-float(scores[index]), int(bundle.metadata[index]["candidate_id"])))
        utility = bundle.candidate_utility[indices].astype(np.float64)
        success = bundle.success[indices] >= 0.5
        chosen = ordered[0]
        ranks = {value: rank for rank, value in enumerate(ordered, start=1)}
        best_index = indices[int(np.argmax(utility))]
        best_rank = ranks[best_index]
        pairwise: list[float] = []
        for positive in np.asarray(indices)[success]:
            for negative in np.asarray(indices)[~success]:
                if scores[positive] > scores[negative]:
                    pairwise.append(1.0)
                elif scores[positive] < scores[negative]:
                    pairwise.append(0.0)
                else:
                    pairwise.append(0.5)
        rows.append(
            {
                "group_id": group_id,
                "score_name": score_name,
                "top1_success": float(bundle.success[chosen] >= 0.5),
                "top3_success": float(any(bundle.success[item] >= 0.5 for item in ordered[:3])),
                "ranking_auc": float(np.mean(pairwise)) if pairwise else 0.5,
                "mrr": float(1.0 / best_rank),
                "ndcg": float(1.0 / math.log2(best_rank + 1)),
                "regret": float(np.max(utility) - bundle.candidate_utility[chosen]),
                "selected_utility": float(bundle.candidate_utility[chosen]),
                "oracle_utility": float(np.max(utility)),
            }
        )
    return rows


def ranking_metrics(bundle: PredictionBundle, scores: np.ndarray, *, score_name: str) -> dict[str, Any]:
    rows = _ranking_group_rows(bundle, scores, score_name=score_name)
    result = {
        key: float(np.mean([row[key] for row in rows]))
        for key in ("top1_success", "top3_success", "ranking_auc", "mrr", "ndcg", "regret", "selected_utility", "oracle_utility")
    }
    result.update({"candidate_sets": len(rows), "per_group": rows, "score_name": score_name})
    return result


def _utility_scores(bundle: PredictionBundle) -> dict[str, np.ndarray]:
    topk = np.sort(bundle.probabilities, axis=1)[:, -min(8, bundle.probabilities.shape[1]):].mean(axis=1)
    return {
        "learned_utility": bundle.utility,
        "contact_utility": bundle.contact_utility,
        "mean_mask_probability": bundle.probabilities.mean(axis=1),
        "topk_mask_confidence": topk,
    }


def _normalize_by_group(bundle: PredictionBundle, values: np.ndarray) -> np.ndarray:
    normalized = np.zeros_like(values, dtype=np.float64)
    for _, indices in _grouped_indices(bundle).items():
        part = values[indices].astype(np.float64)
        lo, hi = float(part.min()), float(part.max())
        normalized[indices] = (part - lo) / (hi - lo) if hi > lo + 1.0e-9 else 0.5
    return normalized


def select_utility_rule(
    learned: PredictionBundle, fair: PredictionBundle, validation: Dataset[Any]
) -> dict[str, Any]:
    """Select learned/fair/hybrid candidate score using validation sets only."""

    if len(learned.metadata) != len(fair.metadata) or learned.metadata != fair.metadata:
        raise ValueError("learned and fair ranking bundles must share order")
    candidates: list[tuple[str, np.ndarray]] = []
    candidates.extend(_utility_scores(learned).items())
    candidates.append(("fair_geometric", fair.utility))
    learned_score = _normalize_by_group(learned, learned.utility)
    fair_score = _normalize_by_group(learned, fair.utility)
    for alpha in np.linspace(0.0, 1.0, 11):
        candidates.append((f"hybrid_alpha_{alpha:.1f}", alpha * learned_score + (1.0 - alpha) * fair_score))
    records = [ranking_metrics(learned, score, score_name=name) for name, score in candidates]
    selected = min(
        records,
        key=lambda item: (float(item["regret"]), -float(item["top1_success"]), str(item["score_name"])),
    )
    selected_score = dict(candidates)[str(selected["score_name"])]
    return {"selection_split": "validation", "records": records, "selected": selected, "scores": selected_score}


def frozen_2b_regression() -> dict[str, Any]:
    """Verify the immutable 2B provenance and reproduce the fixed seed AP."""

    frozen = yaml.safe_load(FROZEN_CONFIG.read_text())
    reference = frozen["reference"]
    source_config = PROJECT_ROOT / str(reference["source_config"])
    source_output = PROJECT_ROOT / str(reference["source_output"])
    manifest = json.loads((source_output / "run_manifest.json").read_text())
    resolved = json.loads((source_output / "resolved_config.json").read_text())
    split = json.loads((source_output / "split_manifest.json").read_text())
    config_digest = _digest(resolved)
    split_digest = _digest(split)
    if config_digest != str(reference["resolved_config_sha256"]):
        raise AssertionError("Milestone 2B resolved configuration digest drifted")
    if split_digest != str(reference["split_manifest_sha256"]):
        raise AssertionError("Milestone 2B split manifest digest drifted")
    if manifest.get("device") != "cpu" or not manifest.get("tests", {}).get("all_passed"):
        raise AssertionError("frozen Milestone 2B provenance is not CPU/test clean")

    # Replaying the checkpoint uses the original 2B dataset and predictor; no
    # 2C data enters the calculation.  Import locally to keep this runner's
    # public contract independent from old internal helper names.
    from actmask.experiments import milestone2b as m2b

    # The archived 2B run predates this runner and may otherwise initialize a
    # wide OpenMP pool.  Regression is intentionally one deterministic CPU
    # thread; it measures correctness, not a legacy throughput number.
    torch.set_num_threads(1)
    old_config = m2b._load_config(source_config)
    old_kwargs = m2b._dataset_kwargs(old_config)
    test_dataset = Milestone2BDataset(split="test", domain="id", **old_kwargs)
    seed = int(reference["frozen_seed"])
    checkpoint = source_output / "seeds" / str(seed) / "checkpoints" / "TemporalActMask.pt"
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = m2b._build_model("TemporalActMask", old_config).cpu()
    model.load_state_dict(saved["state_dict"])
    threshold_record = json.loads((source_output / "seeds" / str(seed) / "thresholds.json").read_text())
    threshold = float(threshold_record["thresholds"]["TemporalActMask"])
    bundle = m2b.predict_learned(model, test_dataset, int(old_config["evaluation"]["batch_size"]))
    observed = float(m2b.mask_metrics(bundle, threshold)["average_precision"])
    expected = float(reference["expected_seed_test_ap"])
    tolerance = float(reference["tolerance_ap"])
    if abs(observed - expected) > tolerance:
        raise AssertionError(f"2B fixed seed AP regressed: {observed:.6f} vs {expected:.6f}")
    return {
        "source_config": str(source_config),
        "source_output": str(source_output),
        "config_digest": config_digest,
        "split_digest": split_digest,
        "frozen_seed": seed,
        "expected_ap": expected,
        "observed_ap": observed,
        "tolerance_ap": tolerance,
        "passed": True,
    }


def _architecture_selection(
    *, config: Mapping[str, Any], datasets: Mapping[str, Dataset[Any]], output_dir: Path
) -> tuple[str, dict[str, Any], dict[tuple[str, int], nn.Module]]:
    """Compare index, set-history, and local association on validation only."""

    architectures = ["IndexCorrespondenceTemporalActMask", *list(config["model"]["architecture_candidates"])]
    trained: dict[tuple[str, int], nn.Module] = {}
    rows: list[dict[str, Any]] = []
    corruption = CorrespondenceCorruptionDataset(
        datasets["val"], mode=str(config["correspondence"]["selection_mode"]), seed=20260720
    )
    for architecture in architectures:
        for seed in config["experiment"]["training_seeds"]:
            model, record = train_model(
                architecture=architecture, loss_name="full_2b", seed=int(seed),
                config=config, datasets=datasets, output_dir=output_dir,
            )
            trained[(architecture, int(seed))] = model
            clean = _validation_ap(predict_model(model, datasets["val"], batch_size=int(config["training"]["evaluation_batch_size"])))
            corrupted = _validation_ap(predict_model(model, corruption, batch_size=int(config["training"]["evaluation_batch_size"])))
            rows.append({**record, "clean_validation_ap": clean, "permuted_validation_ap": corrupted, "ap_drop": clean - corrupted})
    grouped: list[dict[str, Any]] = []
    for architecture in architectures:
        selected = [row for row in rows if row["architecture"] == architecture]
        grouped.append(
            {
                "architecture": architecture,
                "seeds": len(selected),
                "clean_validation_ap": float(np.mean([row["clean_validation_ap"] for row in selected])),
                "permuted_validation_ap": float(np.mean([row["permuted_validation_ap"] for row in selected])),
                "ap_drop": float(np.mean([row["ap_drop"] for row in selected])),
            }
        )
    gate = float(config["gates"]["permutation_ap_drop_max"])
    eligible = [row for row in grouped if row["architecture"] != "IndexCorrespondenceTemporalActMask" and row["ap_drop"] <= gate]
    if not eligible:
        raise RuntimeError("no correspondence-robust architecture passes the validation permutation gate")
    chosen = max(eligible, key=lambda row: (row["permuted_validation_ap"], row["clean_validation_ap"], row["architecture"]))
    report = {"selection_split": "validation", "per_seed": rows, "architectures": grouped, "selected": chosen, "permutation_drop_gate": gate}
    _write_json(output_dir / "architecture_selection.json", report)
    _write_csv(output_dir / "tables" / "architecture_selection.csv", grouped)
    return str(chosen["architecture"]), report, trained


def _loss_ablation(
    *,
    architecture: str,
    config: Mapping[str, Any],
    datasets: Mapping[str, Dataset[Any]],
    output_dir: Path,
    already_trained: Mapping[tuple[str, int], nn.Module],
) -> tuple[str, dict[str, Any], dict[tuple[str, int], nn.Module]]:
    """Run all nine objectives, then choose the smallest validation-only one."""

    models: dict[tuple[str, int], nn.Module] = {}
    rows: list[dict[str, Any]] = []
    for loss_name, weights in LOSS_SPECS:
        for seed in config["experiment"]["training_seeds"]:
            if loss_name == "full_2b" and (architecture, int(seed)) in already_trained:
                model = already_trained[(architecture, int(seed))]
                record = {
                    "architecture": architecture, "loss_name": loss_name, "seed": int(seed),
                    "checkpoint": "reused_from_architecture_selection",
                }
            else:
                model, record = train_model(
                    architecture=architecture, loss_name=loss_name, seed=int(seed),
                    config=config, datasets=datasets, output_dir=output_dir,
                )
            models[(loss_name, int(seed))] = model
            prediction = predict_model(model, datasets["val"], batch_size=int(config["training"]["evaluation_batch_size"]))
            mask = group_mask_metrics(prediction, threshold=0.5)
            counterfactual = counterfactual_metrics(prediction)
            rank_data = datasets["ranking_val/10"]
            rank_prediction = predict_model(model, rank_data, batch_size=int(config["training"]["evaluation_batch_size"]))
            ranking = ranking_metrics(rank_prediction, rank_prediction.utility, score_name="learned_utility")
            rows.append(
                {
                    **record, "active_losses": [key for key, value in weights.items() if value > 0.0],
                    "validation_ap": mask["average_precision"], "validation_f1": mask["f1"],
                    "velocity_cf": counterfactual["velocity_counterfactual_matching_accuracy"],
                    "action_cf": counterfactual["action_counterfactual_matching_accuracy"],
                    "irrelevant_stability": counterfactual["irrelevant_perturbation_stability"],
                    "ranking_top1": ranking["top1_success"], "ranking_regret": ranking["regret"],
                    "ranking_auc": ranking["ranking_auc"],
                }
            )
    aggregate: list[dict[str, Any]] = []
    for loss_name, weights in LOSS_SPECS:
        current = [row for row in rows if row["loss_name"] == loss_name]
        aggregate.append(
            {
                "loss_name": loss_name,
                "active_losses": [key for key, value in weights.items() if value > 0.0],
                "active_loss_count": len([value for value in weights.values() if value > 0.0]),
                **{key: float(np.mean([row[key] for row in current])) for key in (
                    "validation_ap", "validation_f1", "velocity_cf", "action_cf", "irrelevant_stability", "ranking_top1", "ranking_regret", "ranking_auc",
                )},
                "seeds": len(current),
            }
        )
    best_ap = max(row["validation_ap"] for row in aggregate)
    constraints = [
        row for row in aggregate
        if row["validation_ap"] >= best_ap - 0.008
        and row["velocity_cf"] >= float(config["gates"]["velocity_counterfactual_accuracy"])
        and row["action_cf"] >= float(config["gates"]["action_counterfactual_accuracy"])
        and row["irrelevant_stability"] >= float(config["gates"]["irrelevant_stability"])
    ]
    if not constraints:
        constraints = aggregate
    selected = min(
        constraints,
        key=lambda row: (row["active_loss_count"], row["ranking_regret"], -row["ranking_top1"], -row["validation_ap"]),
    )
    selected_name = str(selected["loss_name"])
    intended = {
        "contact_time_bce": "contact/F1",
        "success_bce": "utility top-1",
        "pairwise_ranking": "utility ranking AUC",
        "velocity_counterfactual": "velocity counterfactual matching",
        "action_counterfactual": "action counterfactual matching",
        "irrelevant_stability": "irrelevant perturbation stability",
    }
    full = next(row for row in aggregate if row["loss_name"] == "full_2b")
    evidence = []
    for term, metric in intended.items():
        with_term = [row for row in aggregate if term in row["active_losses"]]
        without_term = [row for row in aggregate if term not in row["active_losses"]]
        metric_key = {
            "contact_time_bce": "validation_f1", "success_bce": "ranking_top1", "pairwise_ranking": "ranking_auc",
            "velocity_counterfactual": "velocity_cf", "action_counterfactual": "action_cf", "irrelevant_stability": "irrelevant_stability",
        }[term]
        evidence.append(
            {
                "term": term, "intended_metric": metric,
                "with_term_best": max(row[metric_key] for row in with_term),
                "without_term_best": max(row[metric_key] for row in without_term),
                "retained": term in selected["active_losses"],
                "selected_metric": selected[metric_key],
                "full_2b_metric": full[metric_key],
            }
        )
    report = {"selection_split": "validation", "per_seed": rows, "aggregate": aggregate, "selected": selected, "auxiliary_evidence": evidence}
    _write_json(output_dir / "loss_ablation.json", report)
    _write_csv(output_dir / "tables" / "loss_ablation.csv", aggregate)
    _write_csv(output_dir / "tables" / "loss_auxiliary_evidence.csv", evidence)
    return selected_name, report, models


def _baseline_validation(
    *,
    config: Mapping[str, Any], datasets: Mapping[str, Dataset[Any]], output_dir: Path,
) -> tuple[str, dict[str, Any], dict[str, nn.Module], list[GeometryWithLearnedResidual]]:
    """Fit/evaluate every fair method on validation and freeze one comparator."""

    batch_size = int(config["training"]["evaluation_batch_size"])
    baselines: dict[str, nn.Module] = {}
    residuals: list[GeometryWithLearnedResidual] = []
    for name, constructor in FAIR_BASELINES_2C.items():
        if name == "GeometryWithLearnedResidual":
            continue
        baselines[name] = constructor().cpu()
    for seed in config["experiment"]["training_seeds"]:
        model, _ = train_residual_baseline(seed=int(seed), config=config, datasets=datasets, output_dir=output_dir)
        residuals.append(model)
    rows: list[dict[str, Any]] = []
    for name, model in baselines.items():
        bundle = predict_baseline(model, datasets["val"], batch_size=batch_size)
        rows.append({"method": name, "validation_ap": _validation_ap(bundle), "learned": False})
    residual_bundle = _average_bundles(
        [predict_baseline(model, datasets["val"], batch_size=batch_size) for model in residuals]
    )
    rows.append({"method": "GeometryWithLearnedResidual", "validation_ap": _validation_ap(residual_bundle), "learned": True, "seeds": len(residuals)})
    selected = max(rows, key=lambda row: (row["validation_ap"], row["method"]))
    report = {"selection_split": "validation", "methods": rows, "selected": selected}
    _write_json(output_dir / "fair_baseline_selection.json", report)
    _write_csv(output_dir / "tables" / "fair_baseline_validation.csv", rows)
    return str(selected["method"]), report, baselines, residuals


def _fair_bundle(
    name: str,
    baselines: Mapping[str, nn.Module],
    residuals: Sequence[GeometryWithLearnedResidual],
    dataset: Dataset[Any],
    batch_size: int,
) -> PredictionBundle:
    if name == "GeometryWithLearnedResidual":
        return _average_bundles([predict_baseline(model, dataset, batch_size=batch_size) for model in residuals])
    return predict_baseline(baselines[name], dataset, batch_size=batch_size)


def _ensemble_model_bundle(
    models: Sequence[nn.Module], dataset: Dataset[Any], batch_size: int,
    *, feature_mode: str = "full", perturbation: str | None = None,
) -> PredictionBundle:
    return _average_bundles(
        [predict_model(model, dataset, batch_size=batch_size, feature_mode=feature_mode, perturbation=perturbation) for model in models]
    )


def _paired_metric_report(
    learned: dict[str, Any], fair: dict[str, Any], *, config: Mapping[str, Any], seed: int
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    fair_by_group = {int(row["group_id"]): row for row in fair["per_group"]}
    for metric in ("average_precision", "iou", "f1", "hard_positive_recall", "hard_negative_specificity"):
        first = [row[metric] for row in learned["per_group"]]
        second = [fair_by_group[int(row["group_id"])][metric] for row in learned["per_group"]]
        report[metric] = paired_summary(
            first, second, seed=int(seed) + len(report) * 37,
            resamples=int(config["statistics"]["bootstrap_resamples"]),
            permutations=int(config["statistics"]["permutation_samples"]),
        )
    return report


def _rule_scores_for_test(rule: Mapping[str, Any], learned: PredictionBundle, fair: PredictionBundle) -> np.ndarray:
    name = str(rule["selected"]["score_name"])
    if name == "fair_geometric":
        return fair.utility
    if name.startswith("hybrid_alpha_"):
        alpha = float(name.rsplit("_", 1)[1])
        return alpha * _normalize_by_group(learned, learned.utility) + (1.0 - alpha) * _normalize_by_group(learned, fair.utility)
    return _utility_scores(learned)[name]


def correspondence_evaluation(
    *,
    models: Sequence[nn.Module],
    config: Mapping[str, Any],
    test: Dataset[Any],
    output_dir: Path,
) -> dict[str, Any]:
    """Stress test all required corruption types without using point IDs."""

    batch_size = int(config["training"]["evaluation_batch_size"])
    clean = _ensemble_model_bundle(models, test, batch_size)
    clean_ap = _validation_ap(clean)
    rows: list[dict[str, Any]] = []
    modes = ["clean", *list(config["correspondence"]["modes"])]
    variants: list[tuple[str, int | None]] = [(mode, None) for mode in modes]
    variants.extend(("per_frame_permutation", int(points)) for points in config["correspondence"]["point_counts"])
    for index, (mode, points) in enumerate(variants):
        dataset: Dataset[Any] = test if mode == "clean" and points is None else CorrespondenceCorruptionDataset(
            test, mode=mode, point_count=points, seed=20260730 + index
        )
        bundle = clean if mode == "clean" and points is None else _ensemble_model_bundle(models, dataset, batch_size)
        ap = _validation_ap(bundle)
        rows.append(
            {
                "mode": mode, "point_count": points if points is not None else int(config["dataset"]["num_points"]),
                "average_precision": ap, "ap_drop": clean_ap - ap, "retention": ap / max(clean_ap, 1.0e-8),
                "groups": len(_grouped_indices(bundle)),
            }
        )
    permutation = next(row for row in rows if row["mode"] == "per_frame_permutation" and row["point_count"] == int(config["dataset"]["num_points"]))
    resample = next(row for row in rows if row["mode"] == "per_frame_resampling")
    report = {
        "clean_ap": clean_ap, "rows": rows,
        "permutation_drop": permutation["ap_drop"], "resampling_retention": resample["retention"],
        "permutation_gate": float(config["gates"]["permutation_ap_drop_max"]),
        "resampling_gate": float(config["gates"]["resampling_retention_min"]),
    }
    _write_json(output_dir / "correspondence_evaluation.json", report)
    _write_csv(output_dir / "tables" / "correspondence_corruptions.csv", rows)
    return report


def ranking_evaluation(
    *, models: Sequence[nn.Module], fair_name: str, baselines: Mapping[str, nn.Module], residuals: Sequence[GeometryWithLearnedResidual],
    config: Mapping[str, Any], datasets: Mapping[str, Dataset[Any]], output_dir: Path,
) -> dict[str, Any]:
    """Select utility on validation and measure 5/10/20-candidate decisions."""

    batch_size = int(config["training"]["evaluation_batch_size"])
    validation_learned = _ensemble_model_bundle(models, datasets["ranking_val/20"], batch_size)
    validation_fair = _fair_bundle(fair_name, baselines, residuals, datasets["ranking_val/20"], batch_size)
    selected_rule = select_utility_rule(validation_learned, validation_fair, datasets["ranking_val/20"])
    platt = fit_platt_scaling(selected_rule["scores"], validation_learned.success)
    rows: list[dict[str, Any]] = []
    paired: dict[str, Any] = {}
    calibration: dict[str, Any] = {"platt_parameters": platt, "selection_split": "validation"}
    for count in config["ranking"]["candidate_counts"]:
        dataset = datasets[f"ranking_test/{count}"]
        learned = _ensemble_model_bundle(models, dataset, batch_size)
        fair = _fair_bundle(fair_name, baselines, residuals, dataset, batch_size)
        learned_scores = _rule_scores_for_test(selected_rule, learned, fair)
        learned_metric = ranking_metrics(learned, learned_scores, score_name=str(selected_rule["selected"]["score_name"]))
        fair_metric = ranking_metrics(learned, fair.utility, score_name="fair_geometric")
        rows.extend(
            [
                {"candidate_count": int(count), "method": "TemporalActMask utility", **{key: value for key, value in learned_metric.items() if key != "per_group"}},
                {"candidate_count": int(count), "method": fair_name, **{key: value for key, value in fair_metric.items() if key != "per_group"}},
            ]
        )
        paired[str(count)] = {
            metric: paired_summary(
                [row[metric] for row in learned_metric["per_group"]], [row[metric] for row in fair_metric["per_group"]],
                seed=20260800 + int(count) + offset,
                resamples=int(config["statistics"]["bootstrap_resamples"]), permutations=int(config["statistics"]["permutation_samples"]),
            )
            for offset, metric in enumerate(("top1_success", "top3_success", "ranking_auc", "regret"))
        }
        calibrated = apply_platt(learned_scores, platt)
        calibration[str(count)] = {
            "utility_reliability": reliability_metrics(calibrated, learned.success, bins=int(config["calibration"]["bins"])),
            "risk_coverage": risk_coverage_curve(calibrated, learned.success),
        }
    target = paired["20"]
    top1_delta = float(target["top1_success"]["delta"])
    regret_fair = next(row for row in rows if row["candidate_count"] == 20 and row["method"] == fair_name)["regret"]
    regret_learned = next(row for row in rows if row["candidate_count"] == 20 and row["method"] == "TemporalActMask utility")["regret"]
    regret_reduction = (float(regret_fair) - float(regret_learned)) / max(float(regret_fair), 1.0e-9)
    support = bool(target["top1_success"]["ci_low"] > 0.0 or target["regret"]["ci_high"] < 0.0)
    report = {
        "utility_selection": {key: value for key, value in selected_rule.items() if key != "scores"},
        "rows": rows, "paired": paired, "calibration": calibration,
        "C20_top1_delta": top1_delta, "C20_regret_reduction": regret_reduction, "C20_paired_support": support,
        "top1_gate": float(config["gates"]["utility_top1_delta"]), "regret_gate": float(config["gates"]["utility_regret_reduction"]),
    }
    _write_json(output_dir / "ranking_evaluation.json", report)
    _write_csv(output_dir / "tables" / "ranking_5_10_20.csv", rows)
    return report


def _runtime_observable(*, batch: int, points: int, frames: int) -> dict[str, Tensor]:
    generator = torch.Generator().manual_seed(20260831 + batch + points + frames)
    timestamps = torch.linspace(-0.3, 0.0, frames).repeat(batch, 1)
    history = torch.randn(batch, frames, points, 3, generator=generator) * 0.15
    visibility = torch.rand(batch, frames, points, generator=generator) > 0.08
    history = history * visibility[:, :, :, None]
    return {
        "points_history": history,
        "visibility_history": visibility,
        "timestamps": timestamps,
        "estimated_velocity": torch.randn(batch, points, 3, generator=generator) * 0.25,
        "velocity_confidence": torch.rand(batch, points, generator=generator),
        "action_command": torch.cat((
            torch.full((batch, 6), 0.15), torch.full((batch, 1), 0.8), torch.full((batch, 1), 0.06),
        ), dim=1),
        "nominal_action_delay": torch.full((batch,), 0.08),
        "observation_delay": torch.full((batch,), 0.03),
    }


def _runtime_case(*, config: Mapping[str, Any], points: int, frames: int, horizon: int, candidates: int) -> dict[str, Any]:
    kwargs = _model_kwargs(config, frames=frames, horizon=horizon)
    kwargs["utility_head"] = True
    model = LocalNeighborhoodTemporalActMask(**kwargs).cpu().eval()
    observable = _runtime_observable(batch=candidates, points=points, frames=frames)
    for _ in range(int(config["runtime"]["warmup"])):
        with torch.inference_mode():
            model(observable)
    elapsed: list[float] = []
    for _ in range(int(config["runtime"]["repeats"])):
        started = time.perf_counter()
        with torch.inference_mode():
            model(observable)
        elapsed.append(time.perf_counter() - started)
    parameter_bytes = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
    tensor_bytes = sum(value.numel() * value.element_size() for value in observable.values())
    return {
        "points": points, "history_frames": frames, "horizon_steps": horizon, "candidate_count": candidates,
        "latency_p50_ms": float(np.percentile(elapsed, 50) * 1000.0),
        "latency_p95_ms": float(np.percentile(elapsed, 95) * 1000.0),
        "throughput_candidates_per_second": float(candidates / max(float(np.median(elapsed)), 1.0e-9)),
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "approx_memory_mb": float((parameter_bytes + tensor_bytes) / (1024.0 * 1024.0)),
        "repeats": int(config["runtime"]["repeats"]),
    }


def runtime_benchmark(*, config: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    """Measure one-variable CPU scaling plus the shared 10-candidate setting."""

    runtime = config["runtime"]
    rows: list[dict[str, Any]] = []
    for points in runtime["point_counts"]:
        rows.append({"sweep": "points", **_runtime_case(config=config, points=int(points), frames=4, horizon=8, candidates=10)})
    for frames in runtime["history_lengths"]:
        rows.append({"sweep": "history", **_runtime_case(config=config, points=128, frames=int(frames), horizon=8, candidates=10)})
    for horizon in runtime["horizons"]:
        rows.append({"sweep": "horizon", **_runtime_case(config=config, points=128, frames=4, horizon=int(horizon), candidates=10)})
    for candidates in runtime["candidate_counts"]:
        rows.append({"sweep": "candidates", **_runtime_case(config=config, points=128, frames=4, horizon=8, candidates=int(candidates))})
    shared = _runtime_case(config=config, points=int(config["dataset"]["num_points"]), frames=int(config["dataset"]["observation_frames"]), horizon=int(config["dataset"]["future_steps"]), candidates=10)
    shared["sweep"] = "shared_10_candidate_current_cpu"
    rows.append(shared)
    report = {
        "rows": rows, "shared_10_candidate": shared,
        "five_hz_supported": bool(shared["latency_p95_ms"] <= 1000.0 / float(config["gates"]["shared_ten_candidate_hz"])),
        "threshold_hz": float(config["gates"]["shared_ten_candidate_hz"]),
        "platform": {"processor": platform.processor(), "python": platform.python_version(), "torch": torch.__version__, "threads": torch.get_num_threads()},
    }
    _write_json(output_dir / "runtime.json", report)
    _write_csv(output_dir / "tables" / "runtime_scaling.csv", rows)
    return report


def _save_figures(
    *, output_dir: Path, id_learned: PredictionBundle, id_fair: PredictionBundle, id_probability: np.ndarray,
    correspondence: Mapping[str, Any], ranking: Mapping[str, Any], calibration: Mapping[str, Any], runtime: Mapping[str, Any],
) -> dict[str, Any]:
    """Write compact deterministic data figures; no image models or simulator assets."""

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as error:  # pragma: no cover - environment defensive
        return {"created": [], "error": str(error)}
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    created: list[str] = []
    keep = _bundle_keep(id_learned)
    for name, score, label in (
        ("pr_id.png", id_probability[keep].reshape(-1), "TemporalActMask"),
        ("pr_fair_id.png", id_fair.probabilities[keep].reshape(-1), "fair geometric"),
    ):
        curve = precision_recall_curve(score, id_learned.targets[keep].reshape(-1))
        fig, axis = plt.subplots(figsize=(4.4, 3.2))
        axis.plot(curve.recall, curve.precision, label=f"{label}, AP={curve.average_precision:.3f}")
        axis.set(xlabel="Recall", ylabel="Precision", ylim=(0, 1.02), xlim=(0, 1.02))
        axis.legend(loc="lower left")
        fig.tight_layout(); fig.savefig(figures / name, dpi=150); plt.close(fig); created.append(name)
    corr_rows = list(correspondence["rows"])
    fig, axis = plt.subplots(figsize=(7.2, 3.2))
    names = [f"{row['mode']}\nN={row['point_count']}" for row in corr_rows]
    axis.bar(np.arange(len(corr_rows)), [row["average_precision"] for row in corr_rows])
    axis.set_xticks(np.arange(len(corr_rows)), names, rotation=45, ha="right", fontsize=7)
    axis.set(ylabel="AP", ylim=(0, 1.02)); fig.tight_layout(); fig.savefig(figures / "correspondence_ap.png", dpi=150); plt.close(fig); created.append("correspondence_ap.png")
    rank_rows = [row for row in ranking["rows"] if row["method"] == "TemporalActMask utility"]
    fig, axis = plt.subplots(figsize=(4.4, 3.2))
    axis.plot([row["candidate_count"] for row in rank_rows], [row["top1_success"] for row in rank_rows], marker="o", label="top-1")
    axis.plot([row["candidate_count"] for row in rank_rows], [row["top3_success"] for row in rank_rows], marker="s", label="top-3")
    axis.set(xlabel="Candidates per scene", ylabel="Success", ylim=(0, 1.02)); axis.legend(); fig.tight_layout(); fig.savefig(figures / "ranking_scaling.png", dpi=150); plt.close(fig); created.append("ranking_scaling.png")
    bins = calibration["mask_reliability"]["bins"]
    used = [row for row in bins if row["count"]]
    fig, axis = plt.subplots(figsize=(3.6, 3.4))
    axis.plot([0, 1], [0, 1], "--", color="gray")
    axis.scatter([row["mean_probability"] for row in used], [row["empirical_frequency"] for row in used])
    axis.set(xlabel="Predicted probability", ylabel="Empirical frequency", xlim=(0, 1), ylim=(0, 1)); fig.tight_layout(); fig.savefig(figures / "mask_calibration.png", dpi=150); plt.close(fig); created.append("mask_calibration.png")
    runtime_rows = [row for row in runtime["rows"] if row["sweep"] == "points"]
    fig, axis = plt.subplots(figsize=(4.4, 3.2))
    axis.plot([row["points"] for row in runtime_rows], [row["latency_p95_ms"] for row in runtime_rows], marker="o")
    axis.set(xlabel="Points", ylabel="p95 latency (ms)", xscale="log"); fig.tight_layout(); fig.savefig(figures / "runtime_points.png", dpi=150); plt.close(fig); created.append("runtime_points.png")
    return {"created": created, "selection": "fixed aggregate curves and all corruption rows; no post-hoc cherry-picking"}


def _bundle_with_probability(bundle: PredictionBundle, probability: np.ndarray) -> PredictionBundle:
    return PredictionBundle(
        logits=bundle.logits, probabilities=np.asarray(probability), targets=bundle.targets,
        contact_utility=bundle.contact_utility, utility=bundle.utility, metadata=bundle.metadata,
        hard_positive=bundle.hard_positive, hard_negative=bundle.hard_negative,
        success=bundle.success, candidate_utility=bundle.candidate_utility,
    )


def _gate_report(
    *,
    config: Mapping[str, Any], id_paired: Mapping[str, Any], ood_paired: Mapping[str, Any],
    counterfactual: Mapping[str, Any], evidence: Mapping[str, float], correspondence: Mapping[str, Any],
    ranking: Mapping[str, Any], runtime: Mapping[str, Any],
) -> dict[str, Any]:
    gates = config["gates"]
    id_ci = bool(id_paired["average_precision"]["ci_low"] > 0.0)
    ood_supported = [
        name for name, result in ood_paired.items()
        if float(result["average_precision"]["ci_low"]) > 0.0
    ]
    history_or_action = max(float(evidence["history_ap_drop"]), float(evidence["action_ap_drop"])) >= float(gates["action_or_history_ap_drop"])
    cf = (
        float(counterfactual["velocity_counterfactual_matching_accuracy"]) >= float(gates["velocity_counterfactual_accuracy"])
        and float(counterfactual["action_counterfactual_matching_accuracy"]) >= float(gates["action_counterfactual_accuracy"])
        and float(counterfactual["irrelevant_perturbation_stability"]) >= float(gates["irrelevant_stability"])
    )
    correspondence_pass = (
        float(correspondence["permutation_drop"]) <= float(gates["permutation_ap_drop_max"])
        and float(correspondence["resampling_retention"]) >= float(gates["resampling_retention_min"])
    )
    utility_pass = bool(ranking["C20_paired_support"]) and (
        float(ranking["C20_top1_delta"]) >= float(gates["utility_top1_delta"])
        or float(ranking["C20_regret_reduction"]) >= float(gates["utility_regret_reduction"])
    )
    checks = {
        "id_paired_ap_ci_excludes_zero": id_ci,
        "ood_paired_ap_supported_conditions": len(ood_supported) >= int(gates["paired_ood_supported_conditions"]),
        "history_or_action_evidence": history_or_action,
        "counterfactual_and_stability": cf,
        "no_fixed_point_ids": correspondence_pass,
        "utility_10_20_candidate_value": utility_pass,
        "cpu_shared_10_candidate_5hz": bool(runtime["five_hz_supported"]),
    }
    all_pass = all(checks.values())
    decision = "GO FOR GPU SIMULATOR INTEGRATION" if all_pass else "NO-GO FOR SIMULATOR INTEGRATION"
    return {
        "checks": checks,
        "id_paired_ap": id_paired["average_precision"],
        "ood_supported_axes": ood_supported,
        "ood_supported_count": len(ood_supported),
        "history_action_evidence": evidence,
        "counterfactual": dict(counterfactual),
        "correspondence": {key: correspondence[key] for key in ("permutation_drop", "resampling_retention", "permutation_gate", "resampling_gate")},
        "utility": {key: ranking[key] for key in ("C20_top1_delta", "C20_regret_reduction", "C20_paired_support", "top1_gate", "regret_gate")},
        "runtime": {"shared_10_candidate": runtime["shared_10_candidate"], "five_hz_supported": runtime["five_hz_supported"]},
        "decision": decision,
    }


def _summary_markdown(result: Mapping[str, Any]) -> str:
    gate = result["gates"]
    id_pair = gate["id_paired_ap"]
    lines = [
        "# ActMask Milestone 2C Summary",
        "",
        f"**Decision: {gate['decision']}**",
        "",
        "This CPU-only synthetic closure kept all 2A/2B artifacts immutable and used an evaluation-only seed namespace.",
        "",
        "## Key evidence",
        "",
        f"- 2B frozen regression: AP {result['frozen_2b']['observed_ap']:.4f} (expected {result['frozen_2b']['expected_ap']:.4f}).",
        f"- Selected architecture/loss: `{result['selected_architecture']}` / `{result['selected_loss']}`.",
        f"- ID AP delta vs selected fair baseline: {id_pair['delta']:+.4f}, 95% CI [{id_pair['ci_low']:+.4f}, {id_pair['ci_high']:+.4f}], paired p={id_pair['p_value']:.4g}, n={id_pair['groups']}.",
        f"- OOD axes with positive paired AP CI: {gate['ood_supported_count']}.",
        f"- Per-frame permutation AP drop: {gate['correspondence']['permutation_drop']:.4f}; resampling retention: {gate['correspondence']['resampling_retention']:.3f}.",
        f"- 20-candidate utility: top-1 delta {gate['utility']['C20_top1_delta']:+.3f}; regret reduction {gate['utility']['C20_regret_reduction']:.1%}; paired support={gate['utility']['C20_paired_support']}.",
        f"- Shared 10-candidate CPU p95 latency: {gate['runtime']['shared_10_candidate']['latency_p95_ms']:.2f} ms; 5 Hz supported={gate['runtime']['five_hz_supported']}.",
        "",
        "## Gate status",
        "",
    ]
    lines.extend(f"- {'PASS' if passed else 'FAIL'} — {name}" for name, passed in gate["checks"].items())
    lines += [
        "",
        "All performance claims in this report are synthetic CPU-only results. The decision does not claim RGB-D, simulator trajectory, or scene-flow validation; it only determines whether the project may begin that next, GPU-dependent integration phase.",
    ]
    return "\n".join(lines)


def run_milestone2c(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Execute the full CPU-only 2C closure and emit a reproducible manifest."""

    config = _load_config(config_path)
    _seed_everything(int(config["experiment"]["training_seeds"][0]), int(config["training"]["num_threads"]))
    output_dir = PROJECT_ROOT / str(config["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    _write_json(output_dir / "resolved_config.json", config)
    frozen = frozen_2b_regression()
    datasets = build_datasets(config)
    # Manifest counts are read without eagerly generating every large evaluation sample.
    dataset_statistics = {
        name: {"groups": len(getattr(dataset, "group_records", ())), "expanded_samples": len(dataset), "materialization": "deferred_to_evaluation"}
        for name, dataset in datasets.items() if not name.startswith("ranking_")
    }
    dataset_statistics["ranking"] = {
        name: {"groups": len(getattr(dataset, "group_records", ())), "expanded_samples": len(dataset), "candidate_count": int(name.rsplit("/", 1)[1]), "materialization": "deferred_to_ranking_evaluation"}
        for name, dataset in datasets.items() if name.startswith("ranking_")
    }
    _write_json(output_dir / "dataset_statistics.json", dataset_statistics)

    architecture, architecture_report, architecture_models = _architecture_selection(config=config, datasets=datasets, output_dir=output_dir)
    loss_name, loss_report, loss_models = _loss_ablation(
        architecture=architecture, config=config, datasets=datasets, output_dir=output_dir, already_trained=architecture_models,
    )
    models = [loss_models[(loss_name, int(seed))] for seed in config["experiment"]["training_seeds"]]
    fair_name, fair_report, baselines, residuals = _baseline_validation(config=config, datasets=datasets, output_dir=output_dir)
    batch_size = int(config["training"]["evaluation_batch_size"])

    validation_learned = _ensemble_model_bundle(models, datasets["val"], batch_size)
    validation_fair = _fair_bundle(fair_name, baselines, residuals, datasets["val"], batch_size)
    valid_keep = _bundle_keep(validation_learned)
    temperature = fit_temperature_scaling(
        torch.from_numpy(validation_learned.logits[valid_keep]), torch.from_numpy(validation_learned.targets[valid_keep])
    )
    learned_threshold = select_threshold(validation_learned, temperature=temperature)
    fair_threshold = select_threshold(validation_fair)
    calibration = {
        "temperature": temperature, "mask_threshold": learned_threshold, "fair_threshold": fair_threshold,
        "selection_split": "validation only",
    }

    id_learned = _ensemble_model_bundle(models, datasets["test"], batch_size)
    id_fair = _fair_bundle(fair_name, baselines, residuals, datasets["test"], batch_size)
    id_probability = calibrated_probability(id_learned, temperature)
    id_metrics = group_mask_metrics(id_learned, threshold=float(learned_threshold["threshold"]), probability=id_probability)
    fair_metrics = group_mask_metrics(id_fair, threshold=float(fair_threshold["threshold"]))
    id_paired = _paired_metric_report(id_metrics, fair_metrics, config=config, seed=20260900)
    full_id_bundle = _bundle_with_probability(id_learned, id_probability)
    counterfactual = counterfactual_metrics(full_id_bundle)
    history = _ensemble_model_bundle(models, datasets["test"], batch_size, feature_mode="no_history")
    action = _ensemble_model_bundle(models, datasets["test"], batch_size, feature_mode="no_action")
    shuffled_history = _ensemble_model_bundle(models, datasets["test"], batch_size, perturbation="history")
    shuffled_action = _ensemble_model_bundle(models, datasets["test"], batch_size, perturbation="action")
    full_ap = id_metrics["average_precision"]
    evidence = {
        "full_ap": full_ap,
        "no_history_ap": _validation_ap(history), "no_action_ap": _validation_ap(action),
        "shuffled_history_ap": _validation_ap(shuffled_history), "shuffled_action_ap": _validation_ap(shuffled_action),
        "history_ap_drop": full_ap - _validation_ap(history), "action_ap_drop": full_ap - _validation_ap(action),
    }

    ood_rows: list[dict[str, Any]] = []
    ood_paired: dict[str, Any] = {}
    for offset, axis in enumerate(MILESTONE2B_OOD_AXES):
        dataset = datasets[f"ood/{axis}"]
        learned = _ensemble_model_bundle(models, dataset, batch_size)
        fair = _fair_bundle(fair_name, baselines, residuals, dataset, batch_size)
        probability = calibrated_probability(learned, temperature)
        learned_metric = group_mask_metrics(learned, threshold=float(learned_threshold["threshold"]), probability=probability)
        fair_metric = group_mask_metrics(fair, threshold=float(fair_threshold["threshold"]))
        paired = _paired_metric_report(learned_metric, fair_metric, config=config, seed=20261000 + offset * 101)
        ood_paired[axis] = paired
        ood_rows.append({"dataset": f"ood/{axis}", "groups": learned_metric["groups"], "learned_ap": learned_metric["average_precision"], "fair_ap": fair_metric["average_precision"], "ap_delta": paired["average_precision"]["delta"], "ap_ci_low": paired["average_precision"]["ci_low"], "ap_ci_high": paired["average_precision"]["ci_high"], "ap_p": paired["average_precision"]["p_value"]})
    _write_json(output_dir / "id_ood_statistics.json", {"id": {"learned": id_metrics, "fair": fair_metrics, "paired": id_paired}, "ood": ood_paired})
    _write_csv(output_dir / "tables" / "ood_paired_statistics.csv", ood_rows)

    test_keep = _bundle_keep(id_learned)
    calibration["mask_reliability"] = reliability_metrics(
        id_probability[test_keep].reshape(-1), id_learned.targets[test_keep].reshape(-1),
        bins=int(config["calibration"]["bins"]),
    )
    calibration["mask_risk_coverage"] = risk_coverage_curve(
        id_probability[test_keep].reshape(-1), id_learned.targets[test_keep].reshape(-1)
    )
    _write_json(output_dir / "calibration.json", calibration)

    correspondence = correspondence_evaluation(models=models, config=config, test=datasets["test"], output_dir=output_dir)
    ranking = ranking_evaluation(models=models, fair_name=fair_name, baselines=baselines, residuals=residuals, config=config, datasets=datasets, output_dir=output_dir)
    runtime = runtime_benchmark(config=config, output_dir=output_dir)
    figures = _save_figures(output_dir=output_dir, id_learned=id_learned, id_fair=id_fair, id_probability=id_probability, correspondence=correspondence, ranking=ranking, calibration=calibration, runtime=runtime)
    _write_json(output_dir / "figures" / "selection_manifest.json", figures)
    gate = _gate_report(config=config, id_paired=id_paired, ood_paired=ood_paired, counterfactual=counterfactual, evidence=evidence, correspondence=correspondence, ranking=ranking, runtime=runtime)
    result = {
        "milestone": "2C", "cpu_only": True, "started_at": _utc_now(), "runtime_seconds": time.perf_counter() - started,
        "frozen_2b": frozen, "selected_architecture": architecture, "architecture_selection": architecture_report,
        "selected_loss": loss_name, "loss_ablation": loss_report, "fair_baseline": fair_report,
        "id": {"learned": id_metrics, "fair": fair_metrics, "paired": id_paired}, "ood": ood_paired,
        "counterfactual": counterfactual, "history_action_evidence": evidence, "correspondence": correspondence,
        "ranking": ranking, "calibration": calibration, "runtime": runtime, "figures": figures, "gates": gate,
    }
    _write_json(output_dir / "summary.json", result)
    _write_text(output_dir / "summary.md", _summary_markdown(result))
    _write_json(output_dir / "run_manifest.json", {
        "milestone": "2C", "status": "complete", "decision": gate["decision"], "device": "cpu", "cuda_used": False,
        "config_digest": _digest(config), "output_dir": str(output_dir), "training_seeds": config["experiment"]["training_seeds"],
        "evaluation_master_seed": config["dataset"]["evaluation_master_seed"], "completed_at": _utc_now(), "runtime_seconds": result["runtime_seconds"],
    })
    return result


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    result = run_milestone2c(args.config)
    print(result["gates"]["decision"])
    return result


if __name__ == "__main__":
    main()
