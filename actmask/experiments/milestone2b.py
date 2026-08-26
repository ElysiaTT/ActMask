"""CPU-only end-to-end experiment for Milestone 2B.

The experiment keeps the physical simulator state behind a hard data-contract
boundary.  Learned models and fair baselines are called with the observable
mapping only; the exact hidden trajectory is used solely by the excluded
oracle and by post-hoc visual explanation.
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
import statistics
import sys
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from actmask.data.milestone2b_dataset import (
    MILESTONE2B_OOD_AXES,
    SAMPLES_PER_GROUP,
    Milestone2BDataset,
    assert_milestone2b_no_group_leakage,
    milestone2b_dataset_statistics,
)
from actmask.eval.pr_metrics import precision_recall_curve, select_validation_threshold
from actmask.models.milestone2b_baselines import (
    FAIR_BASELINES,
    ExactHiddenTrajectoryOracle,
)
from actmask.models.temporal_actmask import ActMaskMLP, TemporalActMask
from actmask.training.milestone2b_losses import (
    Milestone2BLossCoefficients,
    base_supervised_losses,
    counterfactual_assignment_loss,
    irrelevant_background_stability_loss,
    pairwise_candidate_ranking_loss,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "actmask" / "milestone2b_cpu.yaml"
REQUIRED_SEED_COUNT = 5
PRIMARY_METHOD = "TemporalActMask"
LEARNED_VARIANTS = (
    "ActMaskMLP",
    "TemporalActMask",
    "TemporalNoCounterfactual",
    "TemporalNoUtilityHead",
)
ABLATION_METHODS = (
    "ActMaskMLP",
    "TemporalActMask",
    "TemporalNoHistory",
    "TemporalShuffledHistory",
    "TemporalNoAction",
    "TemporalShuffledAction",
    "TemporalNoCounterfactual",
    "TemporalNoVisibility",
    "TemporalNoUtilityHead",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Tensor):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"))


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n")


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _jsonable(row.get(key, "")) for key in fields})


def _seed_everything(seed: int, num_threads: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(num_threads)
    torch.use_deterministic_algorithms(True)


def _load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict):
        raise ValueError("Milestone 2B config must be a mapping")
    if config["experiment"]["device"] != "cpu":
        raise ValueError("Milestone 2B is deliberately CPU-only")
    seeds = list(config["experiment"]["seeds"])
    if len(seeds) < REQUIRED_SEED_COUNT or len(set(seeds)) != len(seeds):
        raise ValueError("the canonical experiment requires at least five unique seeds")
    if int(config["training"]["batch_size"]) % SAMPLES_PER_GROUP:
        raise ValueError("training batch size must preserve complete 11-sample groups")
    return config


def _dataset_kwargs(config: Mapping[str, Any]) -> dict[str, Any]:
    data = config["dataset"]
    return {
        "master_seed": int(data["seed"]),
        "split_counts": {key: int(value) for key, value in data["groups"].items()},
        "ood_groups_per_axis": int(data["ood_groups_per_axis"]),
        "num_points": int(data["num_points"]),
        "observation_frames": int(data["observation_frames"]),
        "future_steps": int(data["future_steps"]),
        "observation_noise_std": float(data["observation_noise_std"]),
        "point_dropout_rate": float(data["point_dropout_rate"]),
        "occlusion_rate": float(data["occlusion_rate"]),
        "include_hidden_state": True,
    }


def build_datasets(config: Mapping[str, Any]) -> dict[str, Milestone2BDataset]:
    kwargs = _dataset_kwargs(config)
    result = {
        split: Milestone2BDataset(split=split, domain="id", **kwargs)
        for split in ("train", "val", "test")
    }
    result.update(
        {
            f"ood/{axis}": Milestone2BDataset(split="test", domain=axis, **kwargs)
            for axis in MILESTONE2B_OOD_AXES
        }
    )
    assert_milestone2b_no_group_leakage(result["train"].manifest)
    partitions: dict[str, set[int]] = {}
    for name, dataset in result.items():
        partitions[name] = {int(record.group_id) for record in dataset.group_records}
    names = sorted(partitions)
    for first_index, first in enumerate(names):
        for second in names[first_index + 1 :]:
            overlap = partitions[first].intersection(partitions[second])
            if overlap:
                raise ValueError(f"group leakage between {first} and {second}: {overlap}")
    return result


def _loader(dataset: Dataset[Any], batch_size: int) -> DataLoader[Any]:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )


def _to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, Tensor):
        return value.to(device)
    if isinstance(value, Mapping):
        return {key: _to_device(item, device) for key, item in value.items()}
    return value


def _model_config(config: Mapping[str, Any]) -> dict[str, int]:
    model = config["model"]
    data = config["dataset"]
    return {
        "observation_frames": int(data["observation_frames"]),
        "horizon_steps": int(data["future_steps"]),
        "history_hidden_dim": int(model["history_hidden_dim"]),
        "action_hidden_dim": int(model["action_hidden_dim"]),
        "fusion_hidden_dim": int(model["fusion_hidden_dim"]),
        "contact_hidden_dim": int(model["contact_hidden_dim"]),
        "utility_hidden_dim": int(model["utility_hidden_dim"]),
    }


def _build_model(variant: str, config: Mapping[str, Any]) -> nn.Module:
    if variant == "ActMaskMLP":
        return ActMaskMLP()
    return TemporalActMask(
        **_model_config(config), utility_head=variant != "TemporalNoUtilityHead"
    )


def _positive_weight(dataset: Dataset[Any], maximum: float) -> float:
    positive = 0
    total = 0
    for index in range(len(dataset)):
        sample = dataset[index]
        if sample["metadata"]["is_irrelevant_perturbation"]:
            continue
        target = sample["targets"]["ground_truth_future_mask"]
        positive += int(target.sum())
        total += int(target.numel())
    negative = total - positive
    return min(float(maximum), float(negative / max(positive, 1)))


def _batch_indices(metadata: Mapping[str, Any]) -> list[dict[str, int]]:
    group_ids = metadata["group_id"].detach().cpu().tolist()
    variants = metadata["dynamics_variant"].detach().cpu().tolist()
    types = list(metadata["candidate_type"])
    irrelevant = metadata["is_irrelevant_perturbation"].detach().cpu().tolist()
    groups: dict[int, dict[str, int]] = defaultdict(dict)
    for index, (group_id, variant, candidate_type, is_irrelevant) in enumerate(
        zip(group_ids, variants, types, irrelevant, strict=True)
    ):
        if is_irrelevant:
            groups[int(group_id)]["irrelevant"] = index
        else:
            groups[int(group_id)][f"{int(variant)}:{candidate_type}"] = index
    return [groups[group_id] for group_id in sorted(groups)]


def _loss_coefficients(config: Mapping[str, Any], variant: str) -> Milestone2BLossCoefficients:
    values = dict(config["training"]["loss_coefficients"])
    if variant == "TemporalNoCounterfactual":
        values["velocity_counterfactual"] = 0.0
        values["action_counterfactual"] = 0.0
        values["irrelevant_stability"] = 0.0
    if variant == "TemporalNoUtilityHead":
        values["success_bce"] = 0.0
        values["pairwise_ranking"] = 0.0
    return Milestone2BLossCoefficients(**values)


def _training_loss(
    output: Mapping[str, Tensor],
    batch: Mapping[str, Any],
    coefficients: Milestone2BLossCoefficients,
    positive_weight: Tensor,
) -> tuple[Tensor, dict[str, float]]:
    targets = batch["targets"]
    metadata = batch["metadata"]
    keep = ~metadata["is_irrelevant_perturbation"].bool()
    base = base_supervised_losses(
        mask_logits=output["mask_logits"][keep],
        contact_logits=output["contact_logits"][keep],
        utility_logits=output["utility_logits"][keep],
        mask_targets=targets["ground_truth_future_mask"][keep],
        contact_targets=targets["contact_matrix"][keep],
        success_targets=targets["success"][keep],
        positive_weight=positive_weight,
    )
    candidate_ids = metadata["candidate_set_id"][keep].detach().cpu().tolist()
    ranking = pairwise_candidate_ranking_loss(
        output["utility_logits"][keep], targets["success"][keep], candidate_ids
    )
    groups = _batch_indices(metadata)
    velocity_terms: list[Tensor] = []
    action_terms: list[Tensor] = []
    irrelevant_terms: list[Tensor] = []
    for indices in groups:
        reference = indices["0:successful"]
        velocity = indices["1:successful"]
        action = indices["0:wrong_timing"]
        irrelevant = indices["irrelevant"]
        logits = output["mask_logits"]
        mask = targets["ground_truth_future_mask"]
        velocity_terms.append(
            counterfactual_assignment_loss(
                logits[reference], logits[velocity], mask[reference], mask[velocity]
            )
        )
        action_terms.append(
            counterfactual_assignment_loss(
                logits[reference], logits[action], mask[reference], mask[action]
            )
        )
        irrelevant_terms.append(
            irrelevant_background_stability_loss(
                logits[reference], logits[irrelevant], mask[reference], mask[irrelevant]
            )
        )
    zero = output["mask_logits"].sum() * 0.0
    terms = {
        **base,
        "pairwise_ranking": ranking,
        "velocity_counterfactual": torch.stack(velocity_terms).mean() if velocity_terms else zero,
        "action_counterfactual": torch.stack(action_terms).mean() if action_terms else zero,
        "irrelevant_stability": torch.stack(irrelevant_terms).mean() if irrelevant_terms else zero,
    }
    weights = coefficients.to_dict()
    total = sum(terms[name] * weights[name] for name in terms)
    return total, {name: float(value.detach()) for name, value in terms.items()}


def _ordinary_mask(metadata: Mapping[str, Any]) -> Tensor:
    return ~metadata["is_irrelevant_perturbation"].bool()


@torch.no_grad()
def _validation_ap(model: nn.Module, dataset: Dataset[Any], batch_size: int) -> float:
    model.eval()
    scores: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for batch in _loader(dataset, batch_size):
        observable = _to_device(batch["observable"], torch.device("cpu"))
        output = model(observable)
        keep = _ordinary_mask(batch["metadata"])
        scores.append(torch.sigmoid(output["mask_logits"][keep]).cpu().numpy().reshape(-1))
        labels.append(
            batch["targets"]["ground_truth_future_mask"][keep].numpy().reshape(-1)
        )
    return precision_recall_curve(np.concatenate(scores), np.concatenate(labels)).average_precision


def train_variant(
    *,
    variant: str,
    seed: int,
    config: Mapping[str, Any],
    datasets: Mapping[str, Milestone2BDataset],
    seed_dir: Path,
) -> tuple[nn.Module, dict[str, Any]]:
    training = config["training"]
    _seed_everything(seed + 97 * LEARNED_VARIANTS.index(variant), int(training["num_threads"]))
    model = _build_model(variant, config).cpu()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    coefficients = _loss_coefficients(config, variant)
    positive_weight = _positive_weight(
        datasets["train"], float(training["max_pos_weight"])
    )
    positive_weight_tensor = torch.tensor(positive_weight, dtype=torch.float32)
    best_ap = -math.inf
    best_epoch = 0
    best_state: dict[str, Tensor] | None = None
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch in range(1, int(training["epochs"]) + 1):
        model.train()
        total_loss = 0.0
        batches = 0
        component_sum: dict[str, float] = defaultdict(float)
        for batch in _loader(datasets["train"], int(training["batch_size"])):
            optimizer.zero_grad(set_to_none=True)
            observable = _to_device(batch["observable"], torch.device("cpu"))
            targets = _to_device(batch["targets"], torch.device("cpu"))
            batch_for_loss = {
                "targets": targets,
                "metadata": batch["metadata"],
            }
            output = model(observable)
            loss, terms = _training_loss(
                output, batch_for_loss, coefficients, positive_weight_tensor
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += float(loss.detach())
            batches += 1
            for name, value in terms.items():
                component_sum[name] += value
        val_ap = _validation_ap(
            model, datasets["val"], int(config["evaluation"]["batch_size"])
        )
        record = {
            "epoch": epoch,
            "train_loss": total_loss / max(batches, 1),
            "validation_ap": val_ap,
            **{
                f"loss_{name}": value / max(batches, 1)
                for name, value in component_sum.items()
            },
        }
        history.append(record)
        if val_ap > best_ap:
            best_ap = val_ap
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)
    checkpoint = seed_dir / "checkpoints" / f"{variant}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "variant": variant,
            "seed": seed,
            "state_dict": best_state,
            "model_config": _model_config(config),
            "best_epoch": best_epoch,
            "best_validation_ap": best_ap,
            "loss_coefficients": coefficients.to_dict(),
        },
        checkpoint,
    )
    _write_csv(seed_dir / "training" / f"{variant}.csv", history)
    return model, {
        "variant": variant,
        "seed": seed,
        "checkpoint": str(checkpoint),
        "best_epoch": best_epoch,
        "best_validation_ap": best_ap,
        "positive_weight": positive_weight,
        "loss_coefficients": coefficients.to_dict(),
        "runtime_seconds": time.perf_counter() - started,
    }


def _copy_observable(observable: Mapping[str, Tensor]) -> dict[str, Tensor]:
    return {key: value.clone() for key, value in observable.items()}


def _perturb_observable(
    observable: Mapping[str, Tensor], perturbation: str | None
) -> dict[str, Tensor]:
    result = _copy_observable(observable)
    if perturbation == "history":
        # Shift across the two dynamics worlds inside each complete group,
        # breaking the history/target assignment while leaving actions and
        # labels untouched.  This is a deterministic intervention, not noise.
        shift = 5 if result["points_history"].shape[0] > 5 else 1
        for key in (
            "points_history",
            "visibility_history",
            "timestamps",
            "estimated_velocity",
            "velocity_confidence",
            "observation_delay",
        ):
            result[key] = torch.roll(result[key], shifts=shift, dims=0)
    elif perturbation == "action":
        for key in ("action_command", "nominal_action_delay"):
            result[key] = torch.roll(result[key], shifts=1, dims=0)
    elif perturbation is not None:
        raise ValueError(f"unknown perturbation {perturbation!r}")
    return result


class PredictionBundle:
    def __init__(
        self,
        *,
        probabilities: np.ndarray,
        targets: np.ndarray,
        utility: np.ndarray,
        contact_utility: np.ndarray,
        metadata: list[dict[str, Any]],
        future_hypotheses: np.ndarray | None,
    ) -> None:
        self.probabilities = probabilities
        self.targets = targets
        self.utility = utility
        self.contact_utility = contact_utility
        self.metadata = metadata
        self.future_hypotheses = future_hypotheses


def _metadata_rows(batch_metadata: Mapping[str, Any]) -> list[dict[str, Any]]:
    size = int(batch_metadata["group_id"].shape[0])
    rows: list[dict[str, Any]] = []
    for index in range(size):
        row: dict[str, Any] = {}
        for key, value in batch_metadata.items():
            if isinstance(value, Tensor):
                item = value[index]
                row[key] = item.item() if item.numel() == 1 else item.tolist()
            else:
                row[key] = value[index]
        rows.append(row)
    return rows


@torch.no_grad()
def predict_learned(
    model: nn.Module,
    dataset: Dataset[Any],
    batch_size: int,
    *,
    feature_mode: str = "full",
    perturbation: str | None = None,
) -> PredictionBundle:
    model.eval()
    probabilities: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    utility: list[np.ndarray] = []
    contact: list[np.ndarray] = []
    future: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    for batch in _loader(dataset, batch_size):
        observable = _perturb_observable(batch["observable"], perturbation)
        if isinstance(model, TemporalActMask):
            output = model(observable, feature_mode=feature_mode)
        else:
            output = model(observable)
        mask_probability = torch.sigmoid(output["mask_logits"])
        contact_probability = torch.sigmoid(output["contact_logits"])
        per_point_contact = contact_probability.amax(dim=-1)
        topk = min(8, per_point_contact.shape[1])
        probabilities.append(mask_probability.cpu().numpy())
        targets.append(batch["targets"]["ground_truth_future_mask"].numpy())
        utility.append(torch.sigmoid(output["utility_logits"]).cpu().numpy())
        contact.append(per_point_contact.topk(topk, dim=1).values.mean(dim=1).cpu().numpy())
        future.append(output["future_hypotheses"].cpu().numpy())
        metadata.extend(_metadata_rows(batch["metadata"]))
    return PredictionBundle(
        probabilities=np.concatenate(probabilities),
        targets=np.concatenate(targets),
        utility=np.concatenate(utility),
        contact_utility=np.concatenate(contact),
        metadata=metadata,
        future_hypotheses=np.concatenate(future),
    )


@torch.no_grad()
def predict_baseline(
    model: nn.Module, dataset: Dataset[Any], batch_size: int
) -> PredictionBundle:
    model.eval()
    probabilities: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    utility: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    futures: list[np.ndarray] = []
    has_future = True
    for batch in _loader(dataset, batch_size):
        # This is the central no-leak call: only the strict observable mapping
        # crosses into every fair baseline.
        output = model(batch["observable"])
        probabilities.append(torch.sigmoid(output["mask_logits"]).cpu().numpy())
        targets.append(batch["targets"]["ground_truth_future_mask"].numpy())
        utility.append(output["utility_score"].cpu().numpy())
        metadata.extend(_metadata_rows(batch["metadata"]))
        if "future_hypotheses" in output:
            future = output["future_hypotheses"]
            if future.ndim == 3:
                future = future[:, :, None, :]
            futures.append(future.cpu().numpy())
        else:
            has_future = False
    utility_array = np.concatenate(utility)
    return PredictionBundle(
        probabilities=np.concatenate(probabilities),
        targets=np.concatenate(targets),
        utility=utility_array,
        contact_utility=utility_array.copy(),
        metadata=metadata,
        future_hypotheses=np.concatenate(futures) if has_future and futures else None,
    )


@torch.no_grad()
def predict_oracle(dataset: Dataset[Any], batch_size: int) -> PredictionBundle:
    oracle = ExactHiddenTrajectoryOracle()
    probabilities: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    utility: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    for batch in _loader(dataset, batch_size):
        output = oracle(batch["hidden_state"])
        probabilities.append(torch.sigmoid(output["mask_logits"]).numpy())
        targets.append(batch["targets"]["ground_truth_future_mask"].numpy())
        utility.append(batch["targets"]["candidate_utility"].numpy())
        metadata.extend(_metadata_rows(batch["metadata"]))
    values = np.concatenate(utility)
    return PredictionBundle(
        probabilities=np.concatenate(probabilities),
        targets=np.concatenate(targets),
        utility=values,
        contact_utility=values.copy(),
        metadata=metadata,
        future_hypotheses=None,
    )


def _ordinary_indices(bundle: PredictionBundle) -> np.ndarray:
    return np.asarray(
        [not bool(row["is_irrelevant_perturbation"]) for row in bundle.metadata],
        dtype=np.bool_,
    )


def _binary_metrics(scores: np.ndarray, targets: np.ndarray, threshold: float) -> dict[str, Any]:
    score = np.asarray(scores, dtype=np.float64).reshape(-1)
    target = np.asarray(targets, dtype=np.float64).reshape(-1) >= 0.5
    predicted = score >= float(threshold)
    tp = int(np.logical_and(predicted, target).sum())
    fp = int(np.logical_and(predicted, ~target).sum())
    fn = int(np.logical_and(~predicted, target).sum())
    tn = int(np.logical_and(~predicted, ~target).sum())
    curve = precision_recall_curve(score, target.astype(np.float64))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    return {
        "average_precision": curve.average_precision,
        "pr_auc": curve.pr_auc,
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "points": int(target.size),
    }


def mask_metrics(
    bundle: PredictionBundle,
    threshold: float,
    *,
    sample_indices: np.ndarray | Sequence[int] | None = None,
) -> dict[str, Any]:
    if sample_indices is None:
        sample_indices = _ordinary_indices(bundle)
    return _binary_metrics(
        bundle.probabilities[sample_indices], bundle.targets[sample_indices], threshold
    )


def hard_case_metrics(
    bundle: PredictionBundle, dataset: Dataset[Any], threshold: float
) -> dict[str, float | int]:
    hard_positive_correct = hard_positive_total = 0
    hard_negative_correct = hard_negative_total = 0
    for index, row in enumerate(bundle.metadata):
        if row["is_irrelevant_perturbation"]:
            continue
        sample = dataset[index]
        prediction = bundle.probabilities[index] >= threshold
        hp = sample["targets"]["hard_positive_mask"].numpy().astype(bool)
        hn = sample["targets"]["hard_negative_mask"].numpy().astype(bool)
        hard_positive_correct += int(prediction[hp].sum())
        hard_positive_total += int(hp.sum())
        hard_negative_correct += int((~prediction[hn]).sum())
        hard_negative_total += int(hn.sum())
    return {
        "hard_positive_recall": hard_positive_correct / max(hard_positive_total, 1),
        "hard_negative_specificity": hard_negative_correct / max(hard_negative_total, 1),
        "hard_positive_points": hard_positive_total,
        "hard_negative_points": hard_negative_total,
    }


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

    def bce(probability: np.ndarray, target: np.ndarray) -> float:
        return float(
            -np.mean(target * np.log(probability) + (1.0 - target) * np.log(1.0 - probability))
        )

    correct = bce(first, first_target) + bce(second, second_target)
    swapped = bce(first, second_target) + bce(second, first_target)
    if math.isclose(correct, swapped, rel_tol=0.0, abs_tol=1.0e-10):
        return 0.5
    return float(correct < swapped)


def counterfactual_metrics(bundle: PredictionBundle) -> dict[str, Any]:
    groups: dict[int, dict[str, int]] = defaultdict(dict)
    for index, row in enumerate(bundle.metadata):
        group = int(row["group_id"])
        if row["is_irrelevant_perturbation"]:
            groups[group]["irrelevant"] = index
        else:
            groups[group][f"{int(row['dynamics_variant'])}:{row['candidate_type']}"] = index
    velocity_values: list[float] = []
    action_values: list[float] = []
    stability_values: list[float] = []
    for group in sorted(groups):
        indices = groups[group]
        reference = indices["0:successful"]
        velocity = indices["1:successful"]
        action = indices["0:wrong_timing"]
        irrelevant = indices["irrelevant"]
        velocity_match = _assignment_match(
            bundle.probabilities[reference],
            bundle.probabilities[velocity],
            bundle.targets[reference],
            bundle.targets[velocity],
        )
        action_match = _assignment_match(
            bundle.probabilities[reference],
            bundle.probabilities[action],
            bundle.targets[reference],
            bundle.targets[action],
        )
        if velocity_match is not None:
            velocity_values.append(velocity_match)
        if action_match is not None:
            action_values.append(action_match)
        stability_values.append(
            1.0
            - float(
                np.mean(
                    np.abs(
                        bundle.probabilities[reference]
                        - bundle.probabilities[irrelevant]
                    )
                )
            )
        )
    return {
        "velocity_counterfactual_matching_accuracy": float(np.mean(velocity_values)) if velocity_values else 0.0,
        "velocity_pairs": len(velocity_values),
        "action_counterfactual_matching_accuracy": float(np.mean(action_values)) if action_values else 0.0,
        "action_pairs": len(action_values),
        "irrelevant_perturbation_stability": float(np.mean(stability_values)),
        "irrelevant_pairs": len(stability_values),
    }


def _candidate_sets(bundle: PredictionBundle) -> list[list[int]]:
    sets: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(bundle.metadata):
        if row["is_irrelevant_perturbation"]:
            continue
        sets[int(row["candidate_set_id"])].append(index)
    return [sets[key] for key in sorted(sets)]


def _ranking_score_arrays(
    bundle: PredictionBundle,
    geometric_bundle: PredictionBundle,
    *,
    model_method: str,
) -> dict[str, np.ndarray]:
    probability = bundle.probabilities
    topk = min(8, probability.shape[1])
    topk_score = np.sort(probability, axis=1)[:, -topk:].mean(axis=1)
    utility_name = (
        "utility_fallback_no_head"
        if model_method == "TemporalNoUtilityHead"
        else "learned_utility_head"
    )
    return {
        "mean_mask_probability": probability.mean(axis=1),
        "mask_mass": probability.sum(axis=1),
        "top_k_mask_confidence": topk_score,
        "time_aligned_predicted_contact_utility": bundle.contact_utility,
        utility_name: bundle.utility,
        "fair_geometric_score": geometric_bundle.utility,
        "excluded_oracle_utility": np.asarray(
            [float(row["target_utility"]) for row in bundle.metadata]
        ),
    }


def ranking_metrics(
    bundle: PredictionBundle,
    geometric_bundle: PredictionBundle,
    *,
    model_method: str = PRIMARY_METHOD,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    scores_by_method = _ranking_score_arrays(
        bundle, geometric_bundle, model_method=model_method
    )
    sets = _candidate_sets(bundle)
    for method, scores in scores_by_method.items():
        top1: list[float] = []
        top3: list[float] = []
        pairwise: list[float] = []
        regrets: list[float] = []
        for indices in sets:
            ordered = sorted(
                indices,
                key=lambda index: (-float(scores[index]), int(bundle.metadata[index]["candidate_id"])),
            )
            # Success and utility are filled by attach_target_metadata below.
            success = np.asarray([float(bundle.metadata[index]["target_success"]) for index in indices])
            utility = np.asarray([float(bundle.metadata[index]["target_utility"]) for index in indices])
            top1.append(float(bundle.metadata[ordered[0]]["target_success"] >= 0.5))
            top3.append(float(any(bundle.metadata[index]["target_success"] >= 0.5 for index in ordered[:3])))
            successful = [indices[pos] for pos in np.flatnonzero(success >= 0.5)]
            failed = [indices[pos] for pos in np.flatnonzero(success < 0.5)]
            for positive in successful:
                for negative in failed:
                    if scores[positive] > scores[negative]:
                        pairwise.append(1.0)
                    elif scores[positive] < scores[negative]:
                        pairwise.append(0.0)
                    else:
                        pairwise.append(0.5)
            chosen_utility = float(bundle.metadata[ordered[0]]["target_utility"])
            regrets.append(float(utility.max() - chosen_utility))
        rows.append(
            {
                "scoring_method": method,
                "model_method": model_method,
                "excluded_oracle": method == "excluded_oracle_utility",
                "top1_success_accuracy": float(np.mean(top1)),
                "top3_success_recall": float(np.mean(top3)),
                "pairwise_ranking_accuracy": float(np.mean(pairwise)) if pairwise else 0.0,
                "ranking_auc": float(np.mean(pairwise)) if pairwise else 0.0,
                "regret": float(np.mean(regrets)),
                "candidate_sets": len(sets),
            }
        )
    return rows


def attach_target_metadata(bundle: PredictionBundle, dataset: Dataset[Any]) -> None:
    if len(bundle.metadata) != len(dataset):
        raise ValueError("prediction and dataset lengths differ")
    for index, row in enumerate(bundle.metadata):
        targets = dataset[index]["targets"]
        row["target_success"] = float(targets["success"])
        row["target_utility"] = float(targets["candidate_utility"])


def _condition_indices(bundle: PredictionBundle, condition: str) -> np.ndarray:
    return np.asarray(
        [
            index
            for index, row in enumerate(bundle.metadata)
            if not row["is_irrelevant_perturbation"] and row["condition"] == condition
        ],
        dtype=np.int64,
    )


def _scenario_indices(bundle: PredictionBundle, scenario: str) -> np.ndarray:
    return np.asarray(
        [
            index
            for index, row in enumerate(bundle.metadata)
            if not row["is_irrelevant_perturbation"] and row["scenario"] == scenario
        ],
        dtype=np.int64,
    )


def _method_spec(method: str) -> tuple[str, str, str | None]:
    if method == "TemporalNoHistory":
        return "TemporalActMask", "no_history", None
    if method == "TemporalShuffledHistory":
        return "TemporalActMask", "full", "history"
    if method == "TemporalNoAction":
        return "TemporalActMask", "no_action", None
    if method == "TemporalShuffledAction":
        return "TemporalActMask", "full", "action"
    if method == "TemporalNoVisibility":
        return "TemporalActMask", "no_visibility", None
    return method, "full", None


def evaluate_seed(
    *,
    seed: int,
    config: Mapping[str, Any],
    datasets: Mapping[str, Milestone2BDataset],
    models: Mapping[str, nn.Module],
    seed_dir: Path,
    retain_visualization_bundles: bool = False,
) -> dict[str, Any]:
    batch_size = int(config["evaluation"]["batch_size"])
    fair_models = {name: factory() for name, factory in FAIR_BASELINES.items()}
    val_bundles: dict[str, PredictionBundle] = {}
    thresholds: dict[str, float] = {}
    validation_rows: list[dict[str, Any]] = []
    for method in ABLATION_METHODS:
        model_name, feature_mode, perturbation = _method_spec(method)
        bundle = predict_learned(
            models[model_name],
            datasets["val"],
            batch_size,
            feature_mode=feature_mode,
            perturbation=perturbation,
        )
        val_bundles[method] = bundle
        keep = _ordinary_indices(bundle)
        selection = select_validation_threshold(
            bundle.probabilities[keep], bundle.targets[keep]
        )
        thresholds[method] = selection.threshold
        metrics = mask_metrics(bundle, selection.threshold)
        validation_rows.append(
            {"seed": seed, "method": method, "split": "val", **selection.to_dict(), **metrics}
        )
    for method, model in fair_models.items():
        bundle = predict_baseline(model, datasets["val"], batch_size)
        val_bundles[method] = bundle
        keep = _ordinary_indices(bundle)
        selection = select_validation_threshold(
            bundle.probabilities[keep], bundle.targets[keep]
        )
        thresholds[method] = selection.threshold
        metrics = mask_metrics(bundle, selection.threshold)
        validation_rows.append(
            {"seed": seed, "method": method, "split": "val", **selection.to_dict(), **metrics}
        )
    fair_validation = {
        row["method"]: float(row["average_precision"])
        for row in validation_rows
        if row["method"] in FAIR_BASELINES
    }
    strongest_fair = max(fair_validation, key=fair_validation.get)
    _write_csv(seed_dir / "validation_metrics.csv", validation_rows)
    _write_json(
        seed_dir / "thresholds.json",
        {
            "selection_split": "ID validation only",
            "thresholds": thresholds,
            "strongest_fair_baseline": strongest_fair,
            "fair_validation_ap": fair_validation,
        },
    )

    dataset_names = ["test", *[f"ood/{axis}" for axis in MILESTONE2B_OOD_AXES]]
    all_bundles: dict[str, dict[str, PredictionBundle]] = {}
    mask_rows: list[dict[str, Any]] = []
    scenario_rows: list[dict[str, Any]] = []
    condition_rows: list[dict[str, Any]] = []
    cf_rows: list[dict[str, Any]] = []
    hard_rows: list[dict[str, Any]] = []
    ranking_rows: list[dict[str, Any]] = []
    for dataset_name in dataset_names:
        dataset = datasets[dataset_name]
        bundles: dict[str, PredictionBundle] = {}
        for method in ABLATION_METHODS:
            model_name, feature_mode, perturbation = _method_spec(method)
            bundles[method] = predict_learned(
                models[model_name],
                dataset,
                batch_size,
                feature_mode=feature_mode,
                perturbation=perturbation,
            )
        for method, model in fair_models.items():
            bundles[method] = predict_baseline(model, dataset, batch_size)
        bundles["ExactHiddenTrajectoryOracle"] = predict_oracle(dataset, batch_size)
        for bundle in bundles.values():
            attach_target_metadata(bundle, dataset)
        if retain_visualization_bundles and dataset_name in {
            "test",
            "ood/unseen_noise_level",
            "ood/unseen_occlusion_rate",
        }:
            all_bundles[dataset_name] = bundles
        for method, bundle in bundles.items():
            excluded = method == "ExactHiddenTrajectoryOracle"
            threshold = 0.5 if excluded else thresholds[method]
            metrics = mask_metrics(bundle, threshold)
            mask_rows.append(
                {
                    "seed": seed,
                    "dataset": dataset_name,
                    "distribution": "ID" if dataset_name == "test" else "OOD",
                    "method": method,
                    "excluded_oracle": excluded,
                    "threshold": threshold,
                    **metrics,
                }
            )
            hard_rows.append(
                {
                    "seed": seed,
                    "dataset": dataset_name,
                    "method": method,
                    "excluded_oracle": excluded,
                    **hard_case_metrics(bundle, dataset, threshold),
                }
            )
            cf_rows.append(
                {
                    "seed": seed,
                    "dataset": dataset_name,
                    "method": method,
                    "excluded_oracle": excluded,
                    **counterfactual_metrics(bundle),
                }
            )
            scenarios = sorted(
                {
                    str(row["scenario"])
                    for row in bundle.metadata
                    if not row["is_irrelevant_perturbation"]
                }
            )
            for scenario in scenarios:
                indices = _scenario_indices(bundle, scenario)
                if len(indices):
                    scenario_rows.append(
                        {
                            "seed": seed,
                            "dataset": dataset_name,
                            "method": method,
                            "scenario": scenario,
                            **mask_metrics(bundle, threshold, sample_indices=indices),
                        }
                    )
            conditions = sorted(
                {
                    str(row["condition"])
                    for row in bundle.metadata
                    if not row["is_irrelevant_perturbation"]
                }
            )
            for condition in conditions:
                indices = _condition_indices(bundle, condition)
                if len(indices):
                    condition_rows.append(
                        {
                            "seed": seed,
                            "dataset": dataset_name,
                            "method": method,
                            "condition": condition,
                            **mask_metrics(bundle, threshold, sample_indices=indices),
                        }
                    )
        primary = bundles[PRIMARY_METHOD]
        geometric = bundles[strongest_fair]
        for row in ranking_metrics(
            primary, geometric, model_method=PRIMARY_METHOD
        ):
            ranking_rows.append({"seed": seed, "dataset": dataset_name, **row})
        no_utility = bundles["TemporalNoUtilityHead"]
        for row in ranking_metrics(
            no_utility, geometric, model_method="TemporalNoUtilityHead"
        ):
            if row["scoring_method"] == "utility_fallback_no_head":
                ranking_rows.append({"seed": seed, "dataset": dataset_name, **row})

    for name, rows in (
        ("mask_metrics.csv", mask_rows),
        ("scenario_metrics.csv", scenario_rows),
        ("condition_metrics.csv", condition_rows),
        ("counterfactual_metrics.csv", cf_rows),
        ("hard_case_metrics.csv", hard_rows),
        ("ranking_metrics.csv", ranking_rows),
    ):
        _write_csv(seed_dir / "evaluation" / name, rows)
    return {
        "seed": seed,
        "strongest_fair_baseline": strongest_fair,
        "thresholds": thresholds,
        "validation_rows": validation_rows,
        "mask_rows": mask_rows,
        "scenario_rows": scenario_rows,
        "condition_rows": condition_rows,
        "counterfactual_rows": cf_rows,
        "hard_rows": hard_rows,
        "ranking_rows": ranking_rows,
        "bundles": all_bundles,
    }


def _numeric_summary(values: Sequence[float]) -> dict[str, float]:
    clean = [float(value) for value in values]
    return {
        "mean": float(np.mean(clean)),
        "std": float(np.std(clean, ddof=1)) if len(clean) > 1 else 0.0,
        "min": float(np.min(clean)),
        "max": float(np.max(clean)),
        "ci95_half_width": (
            1.96 * float(np.std(clean, ddof=1)) / math.sqrt(len(clean))
            if len(clean) > 1
            else 0.0
        ),
        "n": len(clean),
    }


def _summarize_rows(
    rows: Sequence[Mapping[str, Any]], group_fields: Sequence[str], metric_fields: Sequence[str]
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(str(row[field]) for field in group_fields)].append(row)
    result: list[dict[str, Any]] = []
    for key in sorted(groups):
        members = groups[key]
        record: dict[str, Any] = dict(zip(group_fields, key, strict=True))
        record["seeds"] = len({int(row["seed"]) for row in members})
        for metric in metric_fields:
            summary = _numeric_summary([float(row[metric]) for row in members])
            for statistic, value in summary.items():
                record[f"{metric}_{statistic}"] = value
        result.append(record)
    return result


def _lookup_summary(
    rows: Sequence[Mapping[str, Any]], **criteria: str
) -> Mapping[str, Any]:
    matches = [
        row
        for row in rows
        if all(str(row.get(key)) == str(value) for key, value in criteria.items())
    ]
    if len(matches) != 1:
        raise KeyError(f"summary lookup {criteria} returned {len(matches)} rows")
    return matches[0]


def summarize_results(seed_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    mask_rows = [row for result in seed_results for row in result["mask_rows"]]
    validation_rows = [row for result in seed_results for row in result["validation_rows"]]
    scenario_rows = [row for result in seed_results for row in result["scenario_rows"]]
    condition_rows = [row for result in seed_results for row in result["condition_rows"]]
    counterfactual_rows = [row for result in seed_results for row in result["counterfactual_rows"]]
    hard_rows = [row for result in seed_results for row in result["hard_rows"]]
    ranking_rows = [row for result in seed_results for row in result["ranking_rows"]]
    mask_summary = _summarize_rows(
        mask_rows,
        ("dataset", "distribution", "method", "excluded_oracle"),
        ("average_precision", "pr_auc", "iou", "precision", "recall", "f1"),
    )
    validation_summary = _summarize_rows(
        validation_rows,
        ("method",),
        ("average_precision", "pr_auc", "iou", "precision", "recall", "f1"),
    )
    scenario_summary = _summarize_rows(
        scenario_rows,
        ("dataset", "method", "scenario"),
        ("average_precision", "pr_auc", "iou", "precision", "recall", "f1"),
    )
    condition_summary = _summarize_rows(
        condition_rows,
        ("dataset", "method", "condition"),
        ("average_precision", "pr_auc", "iou", "precision", "recall", "f1"),
    )
    cf_summary = _summarize_rows(
        counterfactual_rows,
        ("dataset", "method", "excluded_oracle"),
        (
            "velocity_counterfactual_matching_accuracy",
            "action_counterfactual_matching_accuracy",
            "irrelevant_perturbation_stability",
        ),
    )
    hard_summary = _summarize_rows(
        hard_rows,
        ("dataset", "method", "excluded_oracle"),
        ("hard_positive_recall", "hard_negative_specificity"),
    )
    ranking_summary = _summarize_rows(
        ranking_rows,
        ("dataset", "model_method", "scoring_method", "excluded_oracle"),
        (
            "top1_success_accuracy",
            "top3_success_recall",
            "pairwise_ranking_accuracy",
            "ranking_auc",
            "regret",
        ),
    )
    fair_methods = set(FAIR_BASELINES)
    fair_validation = [row for row in validation_summary if row["method"] in fair_methods]
    strongest_fair = max(fair_validation, key=lambda row: row["average_precision_mean"])["method"]
    id_primary = _lookup_summary(mask_summary, dataset="test", method=PRIMARY_METHOD)
    gaps: list[dict[str, Any]] = []
    for row in mask_summary:
        if row["method"] == PRIMARY_METHOD and row["distribution"] == "OOD":
            gaps.append(
                {
                    "dataset": row["dataset"],
                    "id_average_precision": id_primary["average_precision_mean"],
                    "ood_average_precision": row["average_precision_mean"],
                    "id_minus_ood_ap": id_primary["average_precision_mean"] - row["average_precision_mean"],
                }
            )
    return {
        "mask_summary": mask_summary,
        "validation_summary": validation_summary,
        "scenario_summary": scenario_summary,
        "condition_summary": condition_summary,
        "counterfactual_summary": cf_summary,
        "hard_case_summary": hard_summary,
        "ranking_summary": ranking_summary,
        "strongest_fair_baseline": strongest_fair,
        "id_ood_gaps": gaps,
    }


def evaluate_gates(
    summary: Mapping[str, Any], config: Mapping[str, Any], *, tests_passed: bool
) -> dict[str, Any]:
    gates = config["evaluation"]["gates"]
    mask = summary["mask_summary"]
    cf = summary["counterfactual_summary"]
    ranking = summary["ranking_summary"]
    primary_ap = float(
        _lookup_summary(mask, dataset="test", method=PRIMARY_METHOD)["average_precision_mean"]
    )
    method_ap = {
        method: float(_lookup_summary(mask, dataset="test", method=method)["average_precision_mean"])
        for method in ABLATION_METHODS
    }
    primary_cf = _lookup_summary(cf, dataset="test", method=PRIMARY_METHOD)
    top1 = float(
        _lookup_summary(
            ranking,
            dataset="test",
            model_method=PRIMARY_METHOD,
            scoring_method="learned_utility_head",
        )["top1_success_accuracy_mean"]
    )
    previous = float(config["evaluation"]["previous_full_top1"])
    strongest = str(summary["strongest_fair_baseline"])
    desired_conditions = {"noisy", "occluded", "delayed", "nonlinear"}
    ood_comparisons: list[dict[str, Any]] = []
    for temporal in summary["condition_summary"]:
        if temporal["method"] != PRIMARY_METHOD or temporal["condition"] not in desired_conditions:
            continue
        matches = [
            row
            for row in summary["condition_summary"]
            if row["dataset"] == temporal["dataset"]
            and row["condition"] == temporal["condition"]
            and row["method"] == strongest
        ]
        if not matches or not str(temporal["dataset"]).startswith("ood/"):
            continue
        baseline = matches[0]
        delta = float(temporal["average_precision_mean"] - baseline["average_precision_mean"])
        ood_comparisons.append(
            {
                "dataset": temporal["dataset"],
                "condition": temporal["condition"],
                "temporal_ap": temporal["average_precision_mean"],
                "baseline_ap": baseline["average_precision_mean"],
                "delta_ap": delta,
                "passes": delta > 0.0,
            }
        )
    records = {
        "action_removal_ap_drop": {
            "value": primary_ap - method_ap["TemporalNoAction"],
            "threshold": float(gates["action_ap_drop"]),
            "comparison": ">=",
        },
        "action_shuffle_ap_drop": {
            "value": primary_ap - method_ap["TemporalShuffledAction"],
            "threshold": float(gates["action_ap_drop"]),
            "comparison": ">=",
        },
        "history_removal_ap_drop": {
            "value": primary_ap - method_ap["TemporalNoHistory"],
            "threshold": float(gates["history_removal_ap_drop"]),
            "comparison": ">=",
        },
        "history_shuffle_ap_drop": {
            "value": primary_ap - method_ap["TemporalShuffledHistory"],
            "threshold": float(gates["history_shuffle_ap_drop"]),
            "comparison": ">=",
        },
        "velocity_counterfactual_matching": {
            "value": float(primary_cf["velocity_counterfactual_matching_accuracy_mean"]),
            "threshold": float(gates["velocity_counterfactual_accuracy"]),
            "comparison": ">=",
        },
        "action_counterfactual_matching": {
            "value": float(primary_cf["action_counterfactual_matching_accuracy_mean"]),
            "threshold": float(gates["action_counterfactual_accuracy"]),
            "comparison": ">=",
        },
        "irrelevant_perturbation_stability": {
            "value": float(primary_cf["irrelevant_perturbation_stability_mean"]),
            "threshold": float(gates["irrelevant_stability"]),
            "comparison": ">=",
        },
        "top1_improvement_over_previous_full": {
            "value": top1 - previous,
            "threshold": float(gates["top1_improvement"]),
            "comparison": ">=",
            "temporal_top1": top1,
            "previous_full_top1": previous,
        },
        "temporal_beats_fair_baseline_on_required_ood_subset": {
            "value": float(max((row["delta_ap"] for row in ood_comparisons), default=-math.inf)),
            "threshold": 0.0,
            "comparison": ">",
            "strongest_fair_baseline": strongest,
            "subsets": ood_comparisons,
        },
        "oracle_excluded": {
            "value": 1.0,
            "threshold": 1.0,
            "comparison": "==",
        },
        "all_cpu_tests_pass": {
            "value": float(bool(tests_passed)),
            "threshold": 1.0,
            "comparison": "==",
        },
    }
    for record in records.values():
        comparison = record["comparison"]
        if comparison == ">=":
            record["passes"] = bool(record["value"] >= record["threshold"])
        elif comparison == ">":
            record["passes"] = bool(record["value"] > record["threshold"])
        else:
            record["passes"] = bool(record["value"] == record["threshold"])
    return {
        "fixed_before_evaluation": True,
        "records": records,
        "all_pass": all(bool(record["passes"]) for record in records.values()),
        "decision": "GO" if all(bool(record["passes"]) for record in records.values()) else "NO-GO",
    }


def _mean_std(value: Mapping[str, Any], metric: str) -> str:
    return f"{float(value[f'{metric}_mean']):.4f} ± {float(value[f'{metric}_std']):.4f}"


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(str(item) for item in row) + " |" for row in rows)
    return "\n".join(lines)


def render_summary(
    *,
    summary: Mapping[str, Any],
    gate_report: Mapping[str, Any],
    dataset_statistics: Mapping[str, Any],
    training_records: Sequence[Mapping[str, Any]],
    runtime_seconds: float,
    test_report: Mapping[str, Any],
) -> str:
    mask = summary["mask_summary"]
    primary = _lookup_summary(mask, dataset="test", method=PRIMARY_METHOD)
    strongest = str(summary["strongest_fair_baseline"])
    fair = _lookup_summary(mask, dataset="test", method=strongest)
    oracle = _lookup_summary(mask, dataset="test", method="ExactHiddenTrajectoryOracle")
    ablation_rows = []
    for method in ABLATION_METHODS:
        row = _lookup_summary(mask, dataset="test", method=method)
        ablation_rows.append(
            [method, _mean_std(row, "average_precision"), _mean_std(row, "iou"), _mean_std(row, "f1")]
        )
    cf = _lookup_summary(
        summary["counterfactual_summary"], dataset="test", method=PRIMARY_METHOD
    )
    hard = _lookup_summary(summary["hard_case_summary"], dataset="test", method=PRIMARY_METHOD)
    ranking_rows = []
    for row in summary["ranking_summary"]:
        if row["dataset"] != "test":
            continue
        name = row["scoring_method"]
        if row["model_method"] == "TemporalNoUtilityHead":
            name += " [TemporalNoUtilityHead]"
        if row["excluded_oracle"] == "True":
            name += " (excluded)"
        ranking_rows.append(
            [
                name,
                _mean_std(row, "top1_success_accuracy"),
                _mean_std(row, "top3_success_recall"),
                _mean_std(row, "pairwise_ranking_accuracy"),
                _mean_std(row, "regret"),
            ]
        )
    baseline_rows = []
    for method in (*FAIR_BASELINES.keys(), "ExactHiddenTrajectoryOracle"):
        row = _lookup_summary(mask, dataset="test", method=method)
        baseline_rows.append(
            [
                method,
                "excluded oracle" if method == "ExactHiddenTrajectoryOracle" else "fair observable-only",
                _mean_std(row, "average_precision"),
            ]
        )
    gate_rows = []
    for name, record in gate_report["records"].items():
        value = float(record["value"])
        value_text = "-inf" if math.isinf(value) and value < 0 else f"{value:.4f}"
        gate_rows.append(
            [name, value_text, f"{record['comparison']} {float(record['threshold']):.4f}", "PASS" if record["passes"] else "FAIL"]
        )
    ood_rows = []
    for record in gate_report["records"]["temporal_beats_fair_baseline_on_required_ood_subset"].get("subsets", []):
        ood_rows.append(
            [
                record["dataset"],
                record["condition"],
                f"{float(record['temporal_ap']):.4f}",
                f"{float(record['baseline_ap']):.4f}",
                f"{float(record['delta_ap']):+.4f}",
            ]
        )
    train_runtime = sum(float(row["runtime_seconds"]) for row in training_records)
    coefficient_rows = [
        [name, f"{float(value):.2f}"]
        for name, value in training_records[0]["loss_coefficients"].items()
    ] if training_records else []
    test_text = (
        f"PASS — {test_report.get('passed', '?')} passed in {float(test_report.get('runtime_seconds', 0.0)):.2f}s"
        if test_report.get("all_passed")
        else f"NOT PASSING/NOT FINALIZED — {test_report.get('detail', 'test result unavailable')}"
    )
    decision = gate_report["decision"]
    lines = [
        "# ActMask Milestone 2B — Make Dynamics Necessary",
        "",
        f"**Final recommendation: {decision}.** This decision is the conjunction of the fixed acceptance gates below; no gate was changed after evaluation.",
        "",
        "## What changed",
        "",
        "`TemporalActMask` encodes four noisy observation frames, visible-history motion and confidence, constructs future point hypotheses and the nominal gripper trajectory, fuses time-aligned relative position/distance/velocity/contact features, and predicts point masks plus optional per-time contact logits. Its candidate utility head consumes temporal contact summaries, mask summaries, action embedding, visibility and confidence. The original concatenation model remains available as `ActMaskMLP`.",
        "",
        "Training combines mask BCE, contact-time BCE, success BCE, pairwise action ranking, velocity/action counterfactual assignment, and irrelevant-background stability losses. Coefficients are recorded in every checkpoint and the run manifest.",
        "",
        _markdown_table(["Loss", "Coefficient"], coefficient_rows),
        "",
        "## Dataset and information boundary",
        "",
        "Every group contains identical-geometry dynamics counterfactuals, five candidate actions per world, and one irrelevant-observation perturbation. Group assignment happens before expansion. The role balance is 10 hard-positive, 4 easy-positive, 18 hard-negative and 16 easy-negative points; labels are recomputed from exact execution trajectories.",
        "",
        "Observable fields available identically to learned and fair methods: `points_history`, `visibility_history`, `timestamps`, `estimated_velocity`, `velocity_confidence`, `action_command`, `nominal_action_delay`, `observation_delay`.",
        "",
        "Hidden fields used only for label construction and excluded-oracle evaluation: `future_times`, `exact_future_point_trajectories`, `exact_point_velocity`, `exact_point_acceleration`, `exact_gripper_execution_trajectory`, `exact_execution_delay`, `exact_execution_duration`, `exact_contact_matrix`, `exact_contact_state`.",
        "",
        f"Training-set hard-positive fraction: **{float(dataset_statistics['train']['hard_positive_fraction']):.4f}**; hard-negative fraction: **{float(dataset_statistics['train']['hard_negative_fraction']):.4f}** (both fixed minimum 0.40).",
        "",
        "## Five-seed ID results",
        "",
        f"TemporalActMask ID AP is **{_mean_std(primary, 'average_precision')}**. The strongest validation-selected fair baseline is **{strongest}**, with ID AP **{_mean_std(fair, 'average_precision')}**. The hidden oracle reaches **{_mean_std(oracle, 'average_precision')}** and is excluded from every fair selection and GO gate.",
        "",
        _markdown_table(["Method", "AP", "IoU", "F1"], ablation_rows),
        "",
        "## Fair baselines and excluded oracle",
        "",
        _markdown_table(["Method", "Status", "ID AP"], baseline_rows),
        "",
        "## Dynamics and hard cases",
        "",
        f"Velocity counterfactual matching: **{_mean_std(cf, 'velocity_counterfactual_matching_accuracy')}**. Action counterfactual matching: **{_mean_std(cf, 'action_counterfactual_matching_accuracy')}**. Irrelevant stability: **{_mean_std(cf, 'irrelevant_perturbation_stability')}**.",
        "",
        f"Hard-positive recall: **{_mean_std(hard, 'hard_positive_recall')}**; hard-negative specificity: **{_mean_std(hard, 'hard_negative_specificity')}**. Scenario-level results are in `aggregate/scenario_summary.csv`.",
        "",
        "## Candidate-action ranking",
        "",
        _markdown_table(["Scoring method", "Top-1 success", "Top-3 recall", "Pairwise/AUC", "Regret"], ranking_rows),
        "",
        "## Required OOD comparisons",
        "",
        f"The fair comparator is frozen from ID validation: **{strongest}**. Positive deltas favor TemporalActMask.",
        "",
        _markdown_table(["OOD dataset", "Condition", "Temporal AP", "Fair AP", "Delta"], ood_rows) if ood_rows else "No qualifying OOD subset was present (gate fails).",
        "",
        "All nine held-out axes and ID-minus-OOD gaps are recorded in `aggregate/mask_summary.csv` and `aggregate/id_ood_gaps.csv`; clean/noisy/occluded/delayed/nonlinear rows are in `aggregate/condition_summary.csv`. Each held-out range is paired with its relevant stress scenario, so axis-level gaps are compound domain shifts and are not claimed as single-factor causal effects.",
        "",
        "## Fixed gates",
        "",
        _markdown_table(["Gate", "Observed", "Requirement", "Result"], gate_rows),
        "",
        "## Reproducibility and tests",
        "",
        f"Seeds: 1401, 2403, 3407, 4411, 5413. Device: CPU. Total experiment runtime: **{runtime_seconds:.2f}s**; summed learned-training runtime: **{train_runtime:.2f}s**. Test result: **{test_text}**.",
        "",
        "The run manifest stores configuration and split digests, environment, timestamps, checkpoint provenance, fixed thresholds, and completion status. Visual selections are deterministic and logged; a missing qualifying failure is rendered as a truthful placeholder rather than invented.",
        "",
        "## Recommendation",
        "",
        (
            "All fixed gates pass. Milestone 2B is GO for completing the synthetic dynamics study, but this does not by itself authorize simulator integration."
            if decision == "GO"
            else "At least one fixed gate fails. Milestone 2B remains NO-GO, and simulator integration must not begin. The failed rows above are the authoritative reasons."
        ),
    ]
    return "\n".join(lines)


def write_reports(
    *,
    output_dir: Path,
    config: Mapping[str, Any],
    summary: Mapping[str, Any],
    dataset_statistics: Mapping[str, Any],
    training_records: Sequence[Mapping[str, Any]],
    runtime_seconds: float,
    test_report: Mapping[str, Any],
    write_docs: bool,
) -> dict[str, Any]:
    aggregate = output_dir / "aggregate"
    for filename, key in (
        ("mask_summary.csv", "mask_summary"),
        ("validation_summary.csv", "validation_summary"),
        ("scenario_summary.csv", "scenario_summary"),
        ("condition_summary.csv", "condition_summary"),
        ("counterfactual_summary.csv", "counterfactual_summary"),
        ("hard_case_summary.csv", "hard_case_summary"),
        ("ranking_summary.csv", "ranking_summary"),
        ("id_ood_gaps.csv", "id_ood_gaps"),
    ):
        _write_csv(aggregate / filename, summary[key])
    _write_json(aggregate / "summary.json", summary)
    gates = evaluate_gates(summary, config, tests_passed=bool(test_report.get("all_passed")))
    _write_json(output_dir / "acceptance_gates.json", gates)
    _write_json(output_dir / "test_report.json", test_report)
    markdown = render_summary(
        summary=summary,
        gate_report=gates,
        dataset_statistics=dataset_statistics,
        training_records=training_records,
        runtime_seconds=runtime_seconds,
        test_report=test_report,
    )
    _write_text(output_dir / "summary.md", markdown)
    if write_docs:
        _write_text(PROJECT_ROOT / "docs" / "milestone2b_results.md", markdown)
    return gates


def _manifest_records(datasets: Mapping[str, Milestone2BDataset]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, dataset in datasets.items():
        result[name] = [asdict(record) for record in dataset.group_records]
    return result


def _maybe_generate_visualizations(
    *,
    output_dir: Path,
    datasets: Mapping[str, Milestone2BDataset],
    seed_result: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        from actmask.visualization.milestone2b import generate_milestone2b_visualizations
    except (ImportError, ModuleNotFoundError):
        return {"status": "unavailable", "detail": "visualization module not installed"}
    bundles = seed_result.get("bundles", {})
    if "test" not in bundles:
        return {"status": "unavailable", "detail": "retained test predictions missing"}
    strongest = str(seed_result["strongest_fair_baseline"])
    partitions = {
        "id_test": datasets["test"],
        "noisy_ood": datasets["ood/unseen_noise_level"],
        "occluded_ood": datasets["ood/unseen_occlusion_rate"],
    }
    temporal = {
        "id_test": bundles["test"][PRIMARY_METHOD],
        "noisy_ood": bundles["ood/unseen_noise_level"][PRIMARY_METHOD],
        "occluded_ood": bundles["ood/unseen_occlusion_rate"][PRIMARY_METHOD],
    }
    baseline = {
        "id_test": bundles["test"][strongest],
        "noisy_ood": bundles["ood/unseen_noise_level"][strongest],
        "occluded_ood": bundles["ood/unseen_occlusion_rate"][strongest],
    }
    artifacts = generate_milestone2b_visualizations(
        dataset=partitions,
        temporal_predictions=temporal,
        fair_baseline_predictions=baseline,
        threshold=float(seed_result["thresholds"][PRIMARY_METHOD]),
        fair_baseline_threshold=float(seed_result["thresholds"][strongest]),
        output_dir=output_dir / "visualizations",
        baseline_name=strongest,
        manifest_metadata={"seed": seed_result["seed"], "oracle_fields_rendered": False},
    )
    return {
        "status": "complete",
        "manifest": str(artifacts.manifest_path),
        "paths": {name: str(path) for name, path in artifacts.paths.items()},
        "qualifying": {
            selection.category: bool(selection.qualifying)
            for selection in artifacts.selections
        },
    }


def run_milestone2b(
    *,
    config_path: Path = DEFAULT_CONFIG,
    output_dir: Path | None = None,
    smoke: bool = False,
    seed_limit: int | None = None,
    epochs_override: int | None = None,
) -> dict[str, Any]:
    started_wall = _utc_now()
    started = time.perf_counter()
    config = _load_config(config_path)
    if smoke:
        config["experiment"]["seeds"] = [int(config["experiment"]["seeds"][0])]
        config["training"]["epochs"] = 2
        config["dataset"]["groups"] = {"train": 4, "val": 2, "test": 2}
        config["dataset"]["ood_groups_per_axis"] = 1
    if seed_limit is not None:
        if seed_limit <= 0:
            raise ValueError("seed_limit must be positive")
        config["experiment"]["seeds"] = list(config["experiment"]["seeds"])[
            :seed_limit
        ]
    if epochs_override is not None:
        if epochs_override <= 0:
            raise ValueError("epochs_override must be positive")
        config["training"]["epochs"] = int(epochs_override)
    output = output_dir or PROJECT_ROOT / str(config["experiment"]["output_dir"])
    output = output.resolve()
    protected = (PROJECT_ROOT / "outputs" / "actmask" / "milestone2_evaluation").resolve()
    if output == protected or protected in output.parents:
        raise ValueError("Milestone 2B may not overwrite Milestone 2A outputs")
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "resolved_config.json", config)
    manifest: dict[str, Any] = {
        "status": "running",
        "milestone": "2B",
        "started_at": started_wall,
        "config_path": str(config_path.resolve()),
        "config_digest": _digest(config),
        "output_dir": str(output),
        "device": "cpu",
        "deterministic_algorithms": True,
        "seeds": list(config["experiment"]["seeds"]),
        "canonical_five_seed_run": not smoke and len(config["experiment"]["seeds"]) >= 5,
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "cuda_used": False,
            "cpu_count": os.cpu_count(),
        },
    }
    _write_json(output / "run_manifest.json", manifest)
    datasets = build_datasets(config)
    split_records = _manifest_records(datasets)
    manifest["split_manifest_digest"] = _digest(split_records)
    manifest["group_counts"] = {name: len(value) for name, value in split_records.items()}
    _write_json(output / "split_manifest.json", split_records)
    dataset_statistics = {
        name: milestone2b_dataset_statistics(dataset) for name, dataset in datasets.items()
    }
    _write_json(output / "dataset_statistics.json", dataset_statistics)

    training_records: list[dict[str, Any]] = []
    seed_results: list[dict[str, Any]] = []
    visualization_report: dict[str, Any] = {"status": "not_started"}
    seeds = [int(seed) for seed in config["experiment"]["seeds"]]
    for seed_index, seed in enumerate(seeds):
        seed_dir = output / "seeds" / str(seed)
        models: dict[str, nn.Module] = {}
        for variant in LEARNED_VARIANTS:
            model, record = train_variant(
                variant=variant,
                seed=seed,
                config=config,
                datasets=datasets,
                seed_dir=seed_dir,
            )
            models[variant] = model
            training_records.append(record)
        result = evaluate_seed(
            seed=seed,
            config=config,
            datasets=datasets,
            models=models,
            seed_dir=seed_dir,
            retain_visualization_bundles=seed_index == 0,
        )
        if seed_index == 0:
            visualization_report = _maybe_generate_visualizations(
                output_dir=output, datasets=datasets, seed_result=result
            )
        seed_results.append(result)
        manifest["completed_seeds"] = seed_index + 1
        _write_json(output / "run_manifest.json", manifest)
    summary = summarize_results(seed_results)
    runtime = time.perf_counter() - started
    _write_json(output / "training_records.json", training_records)
    _write_json(output / "visualization_report.json", visualization_report)
    _write_json(output / "runtime.json", {"experiment_seconds": runtime})
    initial_test_report = {
        "all_passed": False,
        "detail": "full CPU pytest suite has not yet been finalized",
        "passed": 0,
        "runtime_seconds": 0.0,
    }
    gates = write_reports(
        output_dir=output,
        config=config,
        summary=summary,
        dataset_statistics=dataset_statistics,
        training_records=training_records,
        runtime_seconds=runtime,
        test_report=initial_test_report,
        write_docs=not smoke and len(seeds) >= 5,
    )
    manifest.update(
        {
            "status": "experiment_complete_tests_pending",
            "completed_at": _utc_now(),
            "runtime_seconds": runtime,
            "visualizations": visualization_report,
            "provisional_decision": gates["decision"],
            "loss_coefficients": dict(config["training"]["loss_coefficients"]),
            "thresholds_by_seed": {
                str(result["seed"]): result["thresholds"] for result in seed_results
            },
        }
    )
    _write_json(output / "run_manifest.json", manifest)
    return {"output_dir": str(output), "summary": summary, "gates": gates, "manifest": manifest}


def finalize_test_report(
    *,
    output_dir: Path,
    all_passed: bool,
    passed: int,
    runtime_seconds: float,
    detail: str,
) -> dict[str, Any]:
    output = output_dir.resolve()
    config = json.loads((output / "resolved_config.json").read_text())
    summary = json.loads((output / "aggregate" / "summary.json").read_text())
    dataset_statistics = json.loads((output / "dataset_statistics.json").read_text())
    training_records = json.loads((output / "training_records.json").read_text())
    runtime = float(json.loads((output / "runtime.json").read_text())["experiment_seconds"])
    test_report = {
        "all_passed": bool(all_passed),
        "passed": int(passed),
        "runtime_seconds": float(runtime_seconds),
        "detail": str(detail),
        "command": "python -m pytest -q",
        "finalized_at": _utc_now(),
    }
    gates = write_reports(
        output_dir=output,
        config=config,
        summary=summary,
        dataset_statistics=dataset_statistics,
        training_records=training_records,
        runtime_seconds=runtime,
        test_report=test_report,
        write_docs=True,
    )
    manifest_path = output / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update(
        {
            "status": "complete" if all_passed else "complete_with_test_failures",
            "tests": test_report,
            "decision": gates["decision"],
            "finalized_at": _utc_now(),
        }
    )
    _write_json(manifest_path, manifest)
    return {"gates": gates, "manifest": manifest}


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seed-limit", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--finalize-tests", action="store_true")
    parser.add_argument("--tests-passed", action="store_true")
    parser.add_argument("--passed-count", type=int, default=0)
    parser.add_argument("--test-runtime", type=float, default=0.0)
    parser.add_argument("--test-detail", default="")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = _parse_args(argv)
    if args.finalize_tests:
        if args.output_dir is None:
            raise ValueError("--finalize-tests requires --output-dir")
        result = finalize_test_report(
            output_dir=args.output_dir,
            all_passed=args.tests_passed,
            passed=args.passed_count,
            runtime_seconds=args.test_runtime,
            detail=args.test_detail,
        )
    else:
        result = run_milestone2b(
            config_path=args.config,
            output_dir=args.output_dir,
            smoke=args.smoke,
            seed_limit=args.seed_limit,
            epochs_override=args.epochs,
        )
    concise = {
        "output_dir": result.get("output_dir", str(args.output_dir) if args.output_dir else None),
        "decision": result["gates"]["decision"],
        "all_gates_pass": result["gates"]["all_pass"],
        "status": result["manifest"]["status"],
    }
    print(json.dumps(concise, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
