"""Run the deterministic CPU-only ActMask Milestone 2 evaluation.

Canonical invocation from the repository root::

    python -m actmask.experiments.milestone2 \
        --config configs/actmask/milestone2_cpu.yaml

The module deliberately owns orchestration and artifact provenance while the
data and evaluation modules own synthetic-scene semantics and geometric
predictors.  No test-set threshold fitting is performed: every decision
threshold is selected on the ID validation split and then frozen for ID test,
scenario, counterfactual, and OOD evaluation.
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
import re
import shlex
import sys
import time
from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, MutableMapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from actmask.eval.ranking import RANKING_SCORE_TIE_TOLERANCE, RANKING_TIE_POLICY
from actmask.utils import load_config, resolve_config_path, resolve_project_path


CANONICAL_COMMAND = (
    "python -m actmask.experiments.milestone2 "
    "--config configs/actmask/milestone2_cpu.yaml"
)
TRAINED_VARIANTS = ("Full", "NoVelocity", "NoAction", "PositionOnly")
SHUFFLED_VARIANTS = ("ShuffledVelocity", "ShuffledAction", "TimeShuffled")
REQUIRED_OOD_AXES = (
    "unseen_velocity_magnitude",
    "unseen_acceleration",
    "unseen_action_duration",
    "unseen_gripper_radius",
    "unseen_trajectory_curvature",
    "increased_point_noise",
    "increased_distractor_count",
)
ORACLE_DEFAULT_NAME = "GeometricOracle"
SCHEMA_VERSION = 2


def _ranking_tie_metadata() -> dict[str, Any]:
    """Return the public semantics for candidate-score ties."""

    return {
        "tie_policy": RANKING_TIE_POLICY,
        "score_tie_tolerance": RANKING_SCORE_TIE_TOLERANCE,
        "tie_handling": (
            "scores within the absolute tolerance are a tied block; top-k and "
            "regret are uniform expectations within that block, pairwise/AUC "
            "ties score 0.5, and candidate_id is display provenance only"
        ),
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _jsonable(value: Any) -> Any:
    """Convert tensors, arrays, paths, and NumPy scalars to JSON values."""

    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Tensor):
        if value.numel() == 1:
            return _jsonable(value.detach().cpu().item())
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(value), handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(value)
        if value and not value.endswith("\n"):
            handle.write("\n")
    temporary.replace(path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            key = str(key)
            if key not in seen:
                seen.add(key)
                keys.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: _jsonable(row.get(key, ""))
                    if not isinstance(row.get(key, ""), (dict, list, tuple))
                    else _canonical_json(row.get(key, ""))
                    for key in keys
                }
            )
    temporary.replace(path)


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "unnamed"


def _seed_everything(seed: int, num_threads: int = 1) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if num_threads > 0:
        torch.set_num_threads(num_threads)


def _validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    resolved = deepcopy(dict(config))
    experiment = dict(resolved.get("experiment", {}))
    device = str(experiment.get("device", resolved.get("device", "cpu"))).lower()
    if device != "cpu":
        raise ValueError("Milestone 2A is CPU-only; experiment.device must be 'cpu'")
    seeds = experiment.get("seeds")
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("experiment.seeds must be a non-empty list")
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
        raise TypeError("every experiment seed must be an integer")
    if len(set(seeds)) != len(seeds):
        raise ValueError("experiment.seeds must not contain duplicates")

    training = dict(resolved.get("training", {}))
    variants = tuple(str(item) for item in training.get("variants", TRAINED_VARIANTS))
    if variants != TRAINED_VARIANTS:
        raise ValueError(
            "training.variants must be exactly Full, NoVelocity, NoAction, PositionOnly"
        )
    if int(training.get("epochs", 0)) <= 0:
        raise ValueError("training.epochs must be positive")
    if int(training.get("batch_size", 0)) <= 0:
        raise ValueError("training.batch_size must be positive")
    if int(training.get("num_workers", 0)) != 0:
        raise ValueError("deterministic CPU validation requires training.num_workers: 0")

    evaluation = dict(resolved.get("evaluation", {}))
    shuffled = tuple(
        str(item) for item in evaluation.get("shuffled_variants", SHUFFLED_VARIANTS)
    )
    if shuffled != SHUFFLED_VARIANTS:
        raise ValueError(
            "evaluation.shuffled_variants must be exactly ShuffledVelocity, "
            "ShuffledAction, TimeShuffled"
        )
    required_baselines = {
        "CurrentPositionProximity",
        "FuturePositionProximity",
        "TimeAlignedTrajectoryProximity",
        "GeometricOracle",
    }
    configured_baselines = {
        str(item) for item in evaluation.get("baselines", required_baselines)
    }
    missing_baselines = sorted(required_baselines - configured_baselines)
    if missing_baselines:
        raise ValueError(
            "evaluation.baselines is missing required Milestone-2 methods: "
            f"{missing_baselines}"
        )
    if str(evaluation.get("oracle_name", ORACLE_DEFAULT_NAME)) != ORACLE_DEFAULT_NAME:
        raise ValueError(
            "evaluation.oracle_name must be 'GeometricOracle' so the excluded "
            "upper bound is explicit"
        )

    ranking = dict(resolved.get("ranking", {}))
    if ranking:
        expected_candidate_types = (
            "successful",
            "failed",
            "wrong_timing",
            "wrong_direction",
            "near_miss",
        )
        if tuple(str(item) for item in ranking.get("candidate_types", ())) != expected_candidate_types:
            raise ValueError(
                "ranking.candidate_types must be exactly successful, failed, "
                "wrong_timing, wrong_direction, near_miss"
            )
        if int(ranking.get("candidates_per_scene", 0)) != len(expected_candidate_types):
            raise ValueError("ranking.candidates_per_scene must equal 5")

    axes = tuple(str(item) for item in dict(resolved.get("dataset", {})).get("ood_axes", ()))
    if set(axes) != set(REQUIRED_OOD_AXES) or len(axes) != len(REQUIRED_OOD_AXES):
        raise ValueError(
            "dataset.ood_axes must contain each of the seven required held-out axes exactly once"
        )
    return resolved


def _environment_manifest(config: Mapping[str, Any], config_path: Path | None) -> dict[str, Any]:
    experiment = dict(config.get("experiment", {}))
    return {
        "schema_version": SCHEMA_VERSION,
        "canonical_command": CANONICAL_COMMAND,
        "invoked_command": " ".join(shlex.quote(part) for part in [sys.executable, *sys.argv]),
        "working_directory": str(Path.cwd().resolve()),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "matplotlib_version": matplotlib.__version__,
        "config_path": str(config_path) if config_path is not None else None,
        "config_digest_sha256": _digest(config),
        "requested_device": str(experiment.get("device", "cpu")),
        "effective_device": "cpu",
        "cpu_assertion": {
            "passed": str(experiment.get("device", "cpu")).lower() == "cpu",
            "cuda_used": False,
            "cuda_available": bool(torch.cuda.is_available()),
        },
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "pid": os.getpid(),
    }


def _as_float_tensor(value: Any) -> Tensor:
    return torch.as_tensor(value, dtype=torch.float32)


class FeatureView(Dataset[Mapping[str, Any]]):
    """Apply a learned feature ablation or a deterministic evaluation shuffle."""

    def __init__(self, dataset: Dataset[Any], mode: str, seed: int = 0) -> None:
        if mode not in (*TRAINED_VARIANTS, *SHUFFLED_VARIANTS):
            raise ValueError(f"Unknown feature mode {mode!r}")
        self.dataset = dataset
        self.mode = mode
        self.seed = int(seed)
        self._permutation: np.ndarray | None = None
        if mode in SHUFFLED_VARIANTS:
            if len(dataset) < 2:
                raise ValueError(f"{mode} requires at least two samples")
            rng = np.random.default_rng(self.seed % (2**32))
            permutation = rng.permutation(len(dataset))
            # A cyclic shift deterministically removes all remaining fixed
            # points without changing the sampled source multiset.
            fixed = np.flatnonzero(permutation == np.arange(len(dataset)))
            if len(fixed):
                permutation = np.roll(permutation, 1)
            self._permutation = permutation

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Mapping[str, Any]:
        item = dict(self.dataset[index])
        if self.mode in {"NoVelocity", "PositionOnly"}:
            item["velocities"] = torch.zeros_like(_as_float_tensor(item["velocities"]))
        if self.mode in {"NoAction", "PositionOnly"}:
            item["action"] = torch.zeros_like(_as_float_tensor(item["action"]))
        if self._permutation is not None:
            source = self.dataset[int(self._permutation[index])]
            if self.mode == "ShuffledVelocity":
                item["velocities"] = _as_float_tensor(source["velocities"]).clone()
            elif self.mode == "ShuffledAction":
                item["action"] = _as_float_tensor(source["action"]).clone()
            else:
                # TimeShuffled changes exactly the action duration.  Spatial
                # action geometry and gripper radius stay with the recipient,
                # making this a specific timing intervention rather than a
                # second action shuffle.
                action = _as_float_tensor(item["action"]).clone()
                action[6] = _as_float_tensor(source["action"])[6]
                item["action"] = action
        return item


class MaterializedDataset(Dataset[Mapping[str, Any]]):
    """Cache deterministic procedural samples once for fair, fast reuse."""

    def __init__(self, dataset: Dataset[Any]) -> None:
        self.samples = [dataset[index] for index in range(len(dataset))]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Mapping[str, Any]:
        return self.samples[index]


class _ModelFieldsView(Dataset[Mapping[str, Any]]):
    """Avoid collating large exact trajectories for learned/fair predictors."""

    def __init__(self, dataset: Dataset[Any]) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Mapping[str, Any]:
        sample = self.dataset[index]
        return {key: sample[key] for key in ("points", "velocities", "action", "mask")}


def _loader(
    dataset: Dataset[Any], *, batch_size: int, shuffle: bool, seed: int
) -> DataLoader[Any]:
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(
        _ModelFieldsView(dataset),
        batch_size=int(batch_size),
        shuffle=shuffle,
        num_workers=0,
        generator=generator,
        drop_last=False,
    )


def _model_inputs(batch: Mapping[str, Any]) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    points = _as_float_tensor(batch["points"])
    velocities = _as_float_tensor(batch["velocities"])
    action = _as_float_tensor(batch["action"])
    target = _as_float_tensor(batch["mask"])
    for name, tensor in (
        ("points", points),
        ("velocities", velocities),
        ("action", action),
        ("mask", target),
    ):
        if tensor.device.type != "cpu":
            raise RuntimeError(f"CPU assertion failed: {name} was on {tensor.device}")
    return points, velocities, action, target


def _build_model(model_config: Mapping[str, Any]) -> nn.Module:
    """Build the Milestone-2 model without changing milestone-1 imports."""

    from actmask.models.actmask_v2 import ActMaskV2

    model = ActMaskV2(**dict(model_config)).cpu()
    if any(parameter.device.type != "cpu" for parameter in model.parameters()):
        raise RuntimeError("CPU assertion failed while constructing ActMaskV2")
    return model


def _build_variant_model(variant: str, model_config: Mapping[str, Any]) -> nn.Module:
    from actmask.models.ablations import AblatedActMask, FeatureMode

    modes = {
        "Full": FeatureMode.FULL,
        "NoVelocity": FeatureMode.NO_VELOCITY,
        "NoAction": FeatureMode.NO_ACTION,
        "PositionOnly": FeatureMode.POSITION_ONLY,
    }
    if variant not in modes:
        raise ValueError(f"Unknown trainable variant {variant!r}")
    return AblatedActMask(_build_model(model_config), modes[variant]).cpu()


def _binary_counts(scores: np.ndarray, targets: np.ndarray, threshold: float) -> dict[str, Any]:
    predicted = np.asarray(scores, dtype=np.float64) >= float(threshold)
    actual = np.asarray(targets, dtype=np.float64) >= 0.5
    tp = int(np.logical_and(predicted, actual).sum())
    fp = int(np.logical_and(predicted, ~actual).sum())
    fn = int(np.logical_and(~predicted, actual).sum())
    tn = int(np.logical_and(~predicted, ~actual).sum())

    def ratio(numerator: float, denominator: float) -> float:
        return float(numerator / denominator) if denominator else 0.0

    return {
        "precision": ratio(tp, tp + fp),
        "recall": ratio(tp, tp + fn),
        "f1": ratio(2 * tp, 2 * tp + fp + fn),
        "iou": ratio(tp, tp + fp + fn),
        "accuracy": ratio(tp + tn, tp + fp + fn + tn),
        "positive_prevalence": ratio(tp + fn, tp + fp + fn + tn),
        "predicted_positive_rate": ratio(tp + fp, tp + fp + fn + tn),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "threshold": float(threshold),
        "num_points": int(tp + fp + fn + tn),
    }


def precision_recall_curve(
    scores: np.ndarray, targets: np.ndarray
) -> dict[str, np.ndarray | float]:
    """Adapt the shared Milestone-2 PR API to the report writer's mapping."""

    from actmask.eval.pr_metrics import precision_recall_curve as shared_pr_curve

    curve = shared_pr_curve(scores, targets)
    return {
        "precision": curve.precision,
        "recall": curve.recall,
        "thresholds": curve.thresholds,
        "average_precision": curve.average_precision,
        "pr_auc": curve.pr_auc,
    }


def select_validation_threshold(scores: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    """Select validation F1 threshold through the shared evaluation API."""

    from actmask.eval.pr_metrics import select_validation_threshold as shared_select

    selection = shared_select(scores, targets)
    curve = precision_recall_curve(scores, targets)
    result = dict(selection.to_dict())
    result.update(
        {
            "precision": selection.validation_precision,
            "recall": selection.validation_recall,
            "f1": selection.validation_f1,
            "iou": selection.validation_iou,
            "selection_split": "id_validation",
            "selection_metric": "f1",
            "average_precision": float(curve["average_precision"]),
            "pr_auc": float(curve["pr_auc"]),
        }
    )
    return result


def _metrics_with_ap(
    scores: np.ndarray, targets: np.ndarray, threshold: float
) -> dict[str, Any]:
    metrics = _binary_counts(scores, targets, threshold)
    metrics["average_precision"] = float(
        precision_recall_curve(scores, targets)["average_precision"]
    )
    return metrics


def _optimizer(model: nn.Module, config: Mapping[str, Any]) -> torch.optim.Optimizer:
    name = str(config.get("optimizer", "adamw")).lower()
    kwargs = {
        "lr": float(config.get("learning_rate", 3.0e-3)),
        "weight_decay": float(config.get("weight_decay", 0.0)),
    }
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), **kwargs)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), **kwargs)
    raise ValueError(f"Unsupported optimizer {name!r}")


def _positive_weight(dataset: Dataset[Any], maximum: float | None) -> float:
    positives = 0.0
    total = 0
    for index in range(len(dataset)):
        mask = _as_float_tensor(dataset[index]["mask"])
        positives += float(mask.sum().item())
        total += int(mask.numel())
    negatives = float(total) - positives
    value = negatives / positives if positives else 1.0
    if maximum is not None:
        value = min(value, float(maximum))
    return max(value, 1.0e-8)


@dataclass
class PredictionBundle:
    scores: np.ndarray
    targets: np.ndarray
    metadata: list[dict[str, Any]]

    def __post_init__(self) -> None:
        if self.scores.shape != self.targets.shape:
            raise ValueError("prediction and target shapes differ")
        if self.scores.shape[0] != len(self.metadata):
            raise ValueError("metadata length differs from prediction sample count")


@dataclass
class DatasetBundle:
    train: Dataset[Any]
    validation: Dataset[Any]
    test: Dataset[Any]
    irrelevant: Dataset[Any]
    ranking: Dataset[Any]
    ood: dict[str, Dataset[Any]]
    manifest: dict[str, Any]


def _dataset_constructor_kwargs(config: Mapping[str, Any]) -> dict[str, Any]:
    data = dict(config.get("dataset", {}))
    groups = dict(data.get("groups", {}))
    counts = tuple(int(groups[name]) for name in ("train", "val", "test"))
    if any(count <= 0 for count in counts):
        raise ValueError("dataset.groups train/val/test counts must all be positive")
    total = sum(counts)
    fractions = tuple(count / total for count in counts)
    return {
        "num_points": int(data.get("num_points", 96)),
        "trajectory_steps": int(data.get("trajectory_steps", 32)),
        "master_seed": int(data.get("seed", 20260713)),
        "total_id_groups": total,
        "split_fractions": fractions,
        "split_counts": {
            "train": counts[0],
            "val": counts[1],
            "test": counts[2],
        },
        "ood_groups_per_domain": int(data.get("ood_groups_per_axis", 10)),
        "point_noise_std": data.get("position_noise_std"),
        "velocity_noise_std": float(data.get("velocity_noise_std", 0.006)),
        "temporal_velocity_delay": float(data.get("temporal_velocity_delay", 0.0)),
        "point_dropout_rate": float(
            data.get("point_dropout_rate", data.get("point_dropout", 0.0))
        ),
    }


OOD_DOMAIN_BY_AXIS = {
    "unseen_velocity_magnitude": "ood_velocity",
    "unseen_acceleration": "ood_acceleration",
    "unseen_action_duration": "ood_duration",
    "unseen_gripper_radius": "ood_radius",
    "unseen_trajectory_curvature": "ood_curvature",
    "increased_point_noise": "ood_point_noise",
    "increased_distractor_count": "ood_distractor_count",
}


def _identifier_values(dataset: Dataset[Any], key: str) -> set[str]:
    values: set[str] = set()
    for index in range(len(dataset)):
        metadata = _metadata_for_sample(dataset[index], index)
        if key in metadata:
            values.add(_canonical_json(metadata[key]))
    return values


def _verify_no_group_leakage(splits: Mapping[str, Dataset[Any]]) -> dict[str, Any]:
    identifiers = (
        "group_id",
        "base_scene_id",
        "pair_id",
        "geometry_seed",
        "scenario_seed",
        "trajectory_family_id",
    )
    intersections: dict[str, dict[str, list[str]]] = {}
    names = list(splits)
    for identifier in identifiers:
        per_split = {name: _identifier_values(splits[name], identifier) for name in names}
        identifier_intersections: dict[str, list[str]] = {}
        for left_index, left in enumerate(names):
            for right in names[left_index + 1 :]:
                overlap = sorted(per_split[left] & per_split[right])
                identifier_intersections[f"{left}__{right}"] = overlap
                if overlap:
                    raise RuntimeError(
                        f"Grouped split leakage: {identifier} overlaps between {left} and {right}"
                    )
        intersections[identifier] = identifier_intersections
    return {"passed": True, "checked_identifiers": list(identifiers), "intersections": intersections}


def _split_statistics(dataset: Dataset[Any]) -> dict[str, Any]:
    scenarios: MutableMapping[str, int] = defaultdict(int)
    domains: MutableMapping[str, int] = defaultdict(int)
    groups: MutableMapping[str, list[Any]] = defaultdict(list)
    positive = 0.0
    points = 0
    for index in range(len(dataset)):
        sample = dataset[index]
        metadata = _metadata_for_sample(sample, index)
        scenario = str(metadata.get("scenario", "unknown"))
        domain = str(metadata.get("domain", metadata.get("ood_axis", "id")))
        group = str(metadata.get("group_id", metadata.get("pair_id", index)))
        scenarios[scenario] += 1
        domains[domain] += 1
        groups[group].append(metadata.get("sample_id", index))
        mask = _as_float_tensor(sample["mask"])
        positive += float(mask.sum().item())
        points += int(mask.numel())
    return {
        "num_samples": len(dataset),
        "num_groups": len(groups),
        "num_points": points,
        "positive_points": int(positive),
        "positive_prevalence": float(positive / points) if points else 0.0,
        "scenarios": dict(sorted(scenarios.items())),
        "domains": dict(sorted(domains.items())),
        "groups": [
            {"group_id": group, "sample_ids": _jsonable(sample_ids)}
            for group, sample_ids in sorted(groups.items())
        ],
    }


def build_datasets(config: Mapping[str, Any]) -> DatasetBundle:
    """Build one shared deterministic ID split and all seven OOD sets."""

    from actmask.data import (
        ActionRankingDataset,
        Milestone2DynamicDataset,
        assert_no_group_leakage,
        build_group_manifest,
        make_irrelevant_background_counterfactual,
        split_statistics,
    )

    kwargs = _dataset_constructor_kwargs(config)
    configured_groups = dict(dict(config.get("dataset", {})).get("groups", {}))
    official_group_manifest = build_group_manifest(
        master_seed=int(kwargs["master_seed"]),
        total_id_groups=int(kwargs["total_id_groups"]),
        split_fractions=kwargs["split_fractions"],
        split_counts=kwargs["split_counts"],
        ood_groups_per_domain=int(kwargs["ood_groups_per_domain"]),
    )
    assert_no_group_leakage(official_group_manifest)
    train = MaterializedDataset(
        Milestone2DynamicDataset(split="train", domain="id", **kwargs)
    )
    validation = MaterializedDataset(
        Milestone2DynamicDataset(split="val", domain="id", **kwargs)
    )
    test = MaterializedDataset(
        Milestone2DynamicDataset(split="test", domain="id", **kwargs)
    )
    irrelevant_samples: list[Mapping[str, Any]] = []
    for index in range(0, len(test), 2):
        original = {
            key: value.clone() if isinstance(value, Tensor) else value
            for key, value in test[index].items()
        }
        changed = make_irrelevant_background_counterfactual(
            original,
            seed=int(kwargs["master_seed"]) + index * 7919,
            max_points=max(1, int(kwargs["num_points"]) // 8),
        )
        changed_mask = torch.as_tensor(changed["changed_input_mask"], dtype=torch.bool)
        original["changed_input_mask"] = changed_mask.clone()
        original["invariant_mask"] = ~changed_mask
        source_sample_id = torch.as_tensor(original["sample_id"], dtype=torch.int64).clone()
        original["source_sample_id"] = source_sample_id.clone()
        changed["source_sample_id"] = source_sample_id.clone()
        # The unperturbed sample stays an ordinary record; only its matched
        # partner carries the irrelevant-background intervention type.  Their
        # unique sample IDs prevent ordinary scenario pairing while the shared
        # source_sample_id gives the counterfactual evaluator an explicit,
        # auditable join key.
        for variant_id, sample in enumerate((original, changed)):
            sample["variant_id"] = torch.tensor(variant_id, dtype=torch.int64)
            sample["sample_id"] = torch.as_tensor(original["sample_id"]) * 10 + variant_id
            if variant_id == 0:
                sample["counterfactual_type"] = "none"
                sample["scenario"] = "irrelevant_background_reference"
            else:
                sample["counterfactual_type"] = "irrelevant_background"
                sample["scenario"] = "irrelevant_background_change"
            irrelevant_samples.append(sample)
    irrelevant = MaterializedDataset(irrelevant_samples)
    # Candidate-action ranking is evaluation-only.  It expands each held-out
    # base scene's successful reference trajectory into the fixed five-action
    # set while preserving the ID split and never contributes to training or
    # validation threshold selection.
    ranking = MaterializedDataset(ActionRankingDataset(test))
    axes = [str(item) for item in dict(config.get("dataset", {}))["ood_axes"]]
    ood = {
        axis: MaterializedDataset(
            Milestone2DynamicDataset(
                split="test", domain=OOD_DOMAIN_BY_AXIS[axis], **kwargs
            )
        )
        for axis in axes
    }

    id_splits = {"train": train, "validation": validation, "test": test}
    leakage = _verify_no_group_leakage(id_splits)
    expected_groups = configured_groups
    statistics = {name: _split_statistics(dataset) for name, dataset in id_splits.items()}
    for configured_name, manifest_name in (
        ("train", "train"),
        ("val", "validation"),
        ("test", "test"),
    ):
        actual = int(statistics[manifest_name]["num_groups"])
        expected = int(expected_groups[configured_name])
        if actual != expected:
            raise RuntimeError(
                f"data API allocated {actual} {manifest_name} groups; config requires {expected}"
            )
    ood_statistics = {axis: _split_statistics(dataset) for axis, dataset in ood.items()}
    required_ood_groups = int(dict(config.get("dataset", {})).get("ood_groups_per_axis", 10))
    for axis, stats in ood_statistics.items():
        if int(stats["num_groups"]) != required_ood_groups:
            raise RuntimeError(
                f"OOD axis {axis} has {stats['num_groups']} groups, expected {required_ood_groups}"
            )
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generator": "Milestone2DynamicDataset",
        "constructor": kwargs,
        "official_group_manifest": official_group_manifest,
        "official_split_statistics": split_statistics(official_group_manifest),
        "id": statistics,
        "ood": ood_statistics,
        "group_leakage_check": leakage,
        "irrelevant_counterfactual": _split_statistics(irrelevant),
        "action_ranking": _split_statistics(ranking),
        "ood_axis_to_domain": dict(OOD_DOMAIN_BY_AXIS),
    }
    manifest["split_digest_sha256"] = _digest(manifest)
    return DatasetBundle(train, validation, test, irrelevant, ranking, ood, manifest)


def _metadata_for_sample(sample: Mapping[str, Any], index: int) -> dict[str, Any]:
    nested = sample.get("metadata", {})
    metadata = dict(nested) if isinstance(nested, Mapping) else {}
    keys = (
        "sample_id",
        "group_id",
        "base_scene_id",
        "pair_id",
        "geometry_seed",
        "variant_id",
        "scenario",
        "scenario_seed",
        "trajectory_family",
        "trajectory_family_id",
        "counterfactual_type",
        "success",
        "split",
        "domain",
        "ood_axis",
        "changed_input_mask",
        "invariant_mask",
        "source_sample_id",
        "candidate_set_id",
        "ranking_scene_id",
        "candidate_id",
        "candidate_type",
        "candidate_success",
        "candidate_utility",
        "oracle_candidate_id",
        "oracle_utility",
        "temporal_velocity_delay",
    )
    for key in keys:
        if key in sample and key not in metadata:
            metadata[key] = sample[key]
    metadata = _jsonable(metadata)
    assert isinstance(metadata, dict)
    metadata.setdefault("sample_id", index)
    metadata.setdefault("group_id", metadata.get("pair_id", index))
    metadata.setdefault("variant_id", 0)
    metadata.setdefault("scenario", "unknown")
    metadata.setdefault("success", 0.0)
    return metadata


def _dataset_metadata(dataset: Dataset[Any]) -> list[dict[str, Any]]:
    # Unwrap FeatureView so shuffled source features cannot alter target-side
    # identifiers or scenario metadata.
    base = dataset.dataset if isinstance(dataset, FeatureView) else dataset
    return [_metadata_for_sample(base[index], index) for index in range(len(base))]


def _predict_learned(
    model: nn.Module, dataset: Dataset[Any], *, batch_size: int
) -> PredictionBundle:
    model = model.cpu().eval()
    scores: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.no_grad():
        for batch in _loader(dataset, batch_size=batch_size, shuffle=False, seed=0):
            points, velocities, action, target = _model_inputs(batch)
            logits = model(points, velocities, action)
            if logits.device.type != "cpu":
                raise RuntimeError("CPU assertion failed: model returned a non-CPU tensor")
            scores.append(torch.sigmoid(logits).cpu().numpy())
            targets.append(target.cpu().numpy())
    return PredictionBundle(
        scores=np.concatenate(scores, axis=0),
        targets=np.concatenate(targets, axis=0),
        metadata=_dataset_metadata(dataset),
    )


def _train_epoch(
    model: nn.Module,
    dataset: Dataset[Any],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    *,
    batch_size: int,
    order_seed: int,
) -> float:
    model.train()
    weighted_loss = 0.0
    total = 0
    for batch in _loader(dataset, batch_size=batch_size, shuffle=True, seed=order_seed):
        points, velocities, action, target = _model_inputs(batch)
        optimizer.zero_grad(set_to_none=True)
        logits = model(points, velocities, action)
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()
        weighted_loss += float(loss.detach().item()) * int(target.numel())
        total += int(target.numel())
    if not total:
        raise ValueError("training dataset is empty")
    return weighted_loss / total


def _load_checkpoint(path: Path) -> Mapping[str, Any]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"checkpoint at {path} is not a mapping")
    return checkpoint


def train_variant(
    *,
    variant: str,
    seed: int,
    train_dataset: Dataset[Any],
    val_dataset: Dataset[Any],
    model_config: Mapping[str, Any],
    training_config: Mapping[str, Any],
    run_dir: Path,
    experiment_digest: str,
    split_digest: str,
    resume: bool,
) -> tuple[nn.Module, dict[str, Any], dict[str, Any]]:
    """Train one ablation from scratch, or validate and resume its checkpoint."""

    if variant not in TRAINED_VARIANTS:
        raise ValueError(f"{variant!r} is not a trainable feature variant")
    variant_dir = run_dir / _safe_name(variant)
    checkpoint_path = variant_dir / "checkpoint.pt"
    metrics_path = variant_dir / "training_metrics.json"
    threshold_path = variant_dir / "validation_threshold.json"
    if resume and checkpoint_path.is_file() and metrics_path.is_file() and threshold_path.is_file():
        checkpoint = _load_checkpoint(checkpoint_path)
        expected = {
            "variant": variant,
            "seed": int(seed),
            "experiment_digest": experiment_digest,
            "split_digest": split_digest,
        }
        if all(checkpoint.get(key) == value for key, value in expected.items()):
            model = _build_variant_model(variant, dict(checkpoint["model_config"]))
            model.load_state_dict(checkpoint["model_state_dict"])
            with metrics_path.open("r", encoding="utf-8") as handle:
                history_report = json.load(handle)
            with threshold_path.open("r", encoding="utf-8") as handle:
                threshold_report = json.load(handle)
            return model, history_report, threshold_report

    _seed_everything(int(seed), int(training_config.get("num_threads", 1)))
    # The shared AblatedActMask zeroes complete learned embeddings.  All four
    # variants therefore consume exactly the same samples/order without biased
    # encoders turning a zero-valued raw input into a non-zero feature.
    train_view = train_dataset
    val_view = val_dataset
    model = _build_variant_model(variant, model_config)
    optimizer = _optimizer(model, training_config)
    pos_weight = (
        _positive_weight(train_dataset, training_config.get("max_pos_weight"))
        if bool(training_config.get("compute_pos_weight", True))
        else 1.0
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
    epochs = int(training_config["epochs"])
    batch_size = int(training_config["batch_size"])
    started = time.perf_counter()
    history: list[dict[str, Any]] = []
    best_ap = -1.0
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, Tensor] | None = None

    for epoch in range(1, epochs + 1):
        train_loss = _train_epoch(
            model,
            train_view,
            optimizer,
            criterion,
            batch_size=batch_size,
            order_seed=seed + epoch * 1009,
        )
        validation = _predict_learned(model, val_view, batch_size=batch_size)
        with torch.no_grad():
            logits = torch.logit(
                torch.as_tensor(validation.scores, dtype=torch.float32).clamp(1e-7, 1 - 1e-7)
            )
            val_loss = float(
                criterion(logits, torch.as_tensor(validation.targets, dtype=torch.float32)).item()
            )
        threshold_report = select_validation_threshold(validation.scores, validation.targets)
        val_ap = float(threshold_report["average_precision"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": val_loss,
                "validation_average_precision": val_ap,
                "validation_best_f1": threshold_report["f1"],
                "validation_best_threshold": threshold_report["threshold"],
            }
        )
        if (val_ap, -val_loss) > (best_ap, -best_loss):
            best_ap = val_ap
            best_loss = val_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    assert best_state is not None
    model.load_state_dict(best_state)
    final_validation = _predict_learned(model, val_view, batch_size=batch_size)
    threshold_report = select_validation_threshold(
        final_validation.scores, final_validation.targets
    )
    threshold_report.update(
        {
            "variant": variant,
            "seed": int(seed),
            "checkpoint_epoch": best_epoch,
            "frozen_for": ["id_test", "scenario", "counterfactual", "all_ood_axes"],
        }
    )
    duration = time.perf_counter() - started
    history_report = {
        "schema_version": SCHEMA_VERSION,
        "variant": variant,
        "seed": int(seed),
        "trained_from_scratch": True,
        "device": "cpu",
        "epochs": epochs,
        "best_epoch": best_epoch,
        "checkpoint_selection": "highest ID-validation average precision; loss tie-break",
        "best_validation_average_precision": best_ap,
        "positive_weight": pos_weight,
        "runtime_seconds": duration,
        "experiment_digest": experiment_digest,
        "split_digest": split_digest,
        "history": history,
    }
    variant_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": SCHEMA_VERSION,
            "variant": variant,
            "seed": int(seed),
            "model_config": dict(model_config),
            "model_state_dict": best_state,
            "best_epoch": best_epoch,
            "validation_threshold": float(threshold_report["threshold"]),
            "validation_average_precision": float(threshold_report["average_precision"]),
            "experiment_digest": experiment_digest,
            "split_digest": split_digest,
            "device": "cpu",
        },
        checkpoint_path,
    )
    _write_json(metrics_path, history_report)
    _write_json(threshold_path, threshold_report)
    return model, history_report, threshold_report


def _construct_with_supported_kwargs(factory: Any, **kwargs: Any) -> Any:
    import inspect

    signature = inspect.signature(factory)
    supported = {key: value for key, value in kwargs.items() if key in signature.parameters}
    return factory(**supported)


def build_baselines(config: Mapping[str, Any]) -> dict[str, nn.Module]:
    """Instantiate named fair geometric baselines from public observations.

    ``GeometricOracle`` is intentionally absent: it is evaluated by
    :func:`_predict_oracle` because it consumes exact simulated trajectories
    and is an upper bound rather than a deployable/fair learned comparison.
    Legacy aliases remain accepted so earlier local M2 artifacts can still be
    inspected without changing milestone-1 code.
    """

    from actmask.models.baselines import (
        ActionProximityMask,
        AnalyticClosestApproachMask,
        CurrentPositionProximity,
        FutureEndpointProximityMask,
        FuturePositionProximity,
        MinFutureSpatialDistanceMask,
        MotionMagnitudeMask,
        NoMask,
        TimeAlignedTrajectoryProximity,
    )

    try:
        from actmask.models.baselines import TimeAlignedFutureProximityMask
    except ImportError:
        class TimeAlignedFutureProximityMask(nn.Module):  # type: ignore[no-redef]
            """Compatibility implementation of the shared fair baseline API."""

            def __init__(self, trajectory_steps: int = 32, temperature: float = 0.05) -> None:
                super().__init__()
                self.trajectory_steps = int(trajectory_steps)
                self.temperature = float(temperature)

            def forward(self, points: Tensor, velocities: Tensor, action: Tensor) -> Tensor:
                fractions = torch.linspace(
                    0.0,
                    1.0,
                    self.trajectory_steps,
                    dtype=points.dtype,
                    device=points.device,
                )
                times = fractions[None, :, None, None] * action[:, None, None, 6:7]
                future_points = points[:, None, :, :] + times * velocities[:, None, :, :]
                gripper = action[:, None, None, :3] + fractions[None, :, None, None] * (
                    action[:, None, None, 3:6] - action[:, None, None, :3]
                )
                minimum = torch.linalg.vector_norm(future_points - gripper, dim=-1).amin(dim=1)
                return (action[:, 7:8] - minimum) / self.temperature

    steps = int(dict(config.get("dataset", {})).get("trajectory_steps", 32))
    available: dict[str, Any] = {
        "NoMask": NoMask,
        "MotionMagnitudeMask": MotionMagnitudeMask,
        # Required Milestone-2 names and information budgets:
        # current p/action; p/v/action endpoint; p/v/action synchronously.
        "CurrentPositionProximity": CurrentPositionProximity,
        "FuturePositionProximity": FuturePositionProximity,
        "TimeAlignedTrajectoryProximity": TimeAlignedTrajectoryProximity,
        # Backward-compatible names retained for prior configs/tests.
        "ActionProximityMask": ActionProximityMask,
        "FutureEndpointProximityMask": FutureEndpointProximityMask,
        "MinFutureSpatialDistanceMask": MinFutureSpatialDistanceMask,
        "TimeAlignedFutureProximityMask": TimeAlignedFutureProximityMask,
        "AnalyticClosestApproachMask": AnalyticClosestApproachMask,
    }
    requested = [
        str(name)
        for name in dict(config.get("evaluation", {})).get("baselines", available)
        if str(name) != str(dict(config.get("evaluation", {})).get("oracle_name", ORACLE_DEFAULT_NAME))
    ]
    missing = [name for name in requested if name not in available]
    if missing:
        raise ValueError(f"Unknown fair baselines in config: {missing}")
    return {
        name: _construct_with_supported_kwargs(available[name], trajectory_steps=steps)
        for name in requested
    }


def _predict_oracle(dataset: Dataset[Any]) -> PredictionBundle:
    """Score exact simulated trajectories; this is an excluded upper bound."""

    score_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    for index in range(len(dataset)):
        sample = dataset[index]
        distance = _as_float_tensor(sample["min_contact_distance"]).cpu().numpy()
        action = _as_float_tensor(sample["action"]).cpu().numpy()
        radius = max(float(action[7]), 0.0)
        temperature = max(radius * 0.05, 1.0e-4)
        logit = (radius - distance) / temperature
        score_rows.append(1.0 / (1.0 + np.exp(-np.clip(logit, -60.0, 60.0))))
        target_rows.append(_as_float_tensor(sample["mask"]).cpu().numpy())
    return PredictionBundle(
        scores=np.stack(score_rows),
        targets=np.stack(target_rows),
        metadata=_dataset_metadata(dataset),
    )


def _pr_auc(curve: Mapping[str, Any]) -> float:
    if "pr_auc" in curve:
        return float(curve["pr_auc"])
    precision = np.asarray(curve["precision"], dtype=np.float64)
    recall = np.asarray(curve["recall"], dtype=np.float64)
    if not len(precision):
        return 0.0
    return float(np.trapezoid(np.r_[1.0, precision], np.r_[0.0, recall]))


def _pair_groups(bundle: PredictionBundle, indices: Iterable[int]) -> list[list[int]]:
    groups: MutableMapping[str, list[int]] = defaultdict(list)
    for index in indices:
        metadata = bundle.metadata[index]
        key = _canonical_json(metadata.get("group_id", metadata.get("pair_id", index)))
        groups[key].append(index)
    pairs: list[list[int]] = []
    for members in groups.values():
        ordered = sorted(
            members,
            key=lambda item: int(bundle.metadata[item].get("variant_id", item)),
        )
        if len(ordered) >= 2:
            pairs.append(ordered[:2])
    return pairs


def _paired_metrics(
    bundle: PredictionBundle, indices: Sequence[int], threshold: float
) -> dict[str, Any]:
    consistency: list[float] = []
    ranking: list[float] = []
    signed_accuracy: list[float] = []
    probability_response: list[float] = []
    predicted_change_rates: list[float] = []
    target_change_rates: list[float] = []
    stability: list[float] = []
    for first, second in _pair_groups(bundle, indices):
        first_prediction = bundle.scores[first] >= threshold
        second_prediction = bundle.scores[second] >= threshold
        first_target = bundle.targets[first] >= 0.5
        second_target = bundle.targets[second] >= 0.5
        predicted_change = np.logical_xor(first_prediction, second_prediction)
        target_change = np.logical_xor(first_target, second_target)
        union = int(np.logical_or(predicted_change, target_change).sum())
        intersection = int(np.logical_and(predicted_change, target_change).sum())
        consistency.append(float(intersection / union) if union else 1.0)
        predicted_change_rates.append(float(predicted_change.mean()))
        target_change_rates.append(float(target_change.mean()))

        label_delta = bundle.targets[second] - bundle.targets[first]
        probability_delta = bundle.scores[second] - bundle.scores[first]
        changed = np.abs(label_delta) > 0.5
        if changed.any():
            signed_accuracy.append(float(np.mean(label_delta[changed] * probability_delta[changed] > 0)))
            probability_response.append(float(np.mean(np.abs(probability_delta[changed]))))
        unchanged = ~changed
        if unchanged.any():
            stability.append(float(1.0 - np.mean(np.abs(probability_delta[unchanged]))))

        first_success = bool(bundle.metadata[first].get("success", 0.0))
        second_success = bool(bundle.metadata[second].get("success", 0.0))
        if first_success != second_success:
            first_score = float(bundle.scores[first].mean())
            second_score = float(bundle.scores[second].mean())
            delta = first_score - second_score
            if abs(delta) <= 1e-12:
                ranking.append(0.5)
            else:
                ranking.append(float((delta > 0) == first_success))

    def mean(values: Sequence[float]) -> float:
        return float(np.mean(values)) if values else 0.0

    return {
        "paired_consistency": mean(consistency),
        "paired_comparisons": len(consistency),
        "success_ranking_accuracy": mean(ranking),
        "ranking_comparisons": len(ranking),
        "counterfactual_signed_change_accuracy": mean(signed_accuracy),
        "counterfactual_probability_response": mean(probability_response),
        "predicted_change_prevalence": mean(predicted_change_rates),
        "target_change_prevalence": mean(target_change_rates),
        "irrelevant_prediction_stability": mean(stability),
    }


def _prediction_records_from_bundle(
    bundle: PredictionBundle,
    *,
    distribution: str,
    ood_axis: str | None,
    dataset: Dataset[Any] | None = None,
    retain_inputs: bool = False,
) -> list[Any]:
    """Convert one prediction bundle into the shared evaluation record API.

    Keeping this conversion in the orchestrator makes the scalar provenance
    explicit and lets scenario, counterfactual, and ranking evaluators share
    exactly the same predictions.  Counterfactual masks/source identities are
    deliberately preserved rather than recomputed from loader order.
    """

    from actmask.eval.prediction import PredictionRecord

    records: list[PredictionRecord] = []
    if retain_inputs and dataset is None:
        raise ValueError("retain_inputs=True requires the source dataset")
    for index, metadata in enumerate(bundle.metadata):
        source = dataset[index] if retain_inputs and dataset is not None else None
        records.append(
            PredictionRecord(
                probabilities=torch.as_tensor(bundle.scores[index], dtype=torch.float32),
                targets=torch.as_tensor(bundle.targets[index]),
                sample_id=metadata.get("sample_id", index),
                group_id=metadata.get("group_id"),
                base_scene_id=metadata.get("base_scene_id"),
                geometry_seed=metadata.get("geometry_seed"),
                scenario_seed=metadata.get("scenario_seed"),
                pair_id=metadata.get("pair_id", metadata.get("group_id")),
                variant_id=metadata.get("variant_id", 0),
                scenario=str(metadata.get("scenario", "unknown")),
                success=float(metadata.get("success", 0.0)),
                counterfactual_type=str(metadata.get("counterfactual_type", "none")),
                distribution=distribution,
                ood_axis=ood_axis or "none",
                changed_input_mask=metadata.get("changed_input_mask"),
                invariant_mask=metadata.get("invariant_mask"),
                ranking_scene_id=metadata.get("ranking_scene_id"),
                candidate_id=metadata.get("candidate_id"),
                candidate_type=str(metadata.get("candidate_type", "")),
                candidate_success=metadata.get("candidate_success"),
                candidate_utility=metadata.get("candidate_utility"),
                oracle_candidate_id=metadata.get("oracle_candidate_id"),
                oracle_utility=metadata.get("oracle_utility"),
                metadata=dict(metadata),
                points=(
                    torch.as_tensor(source["points"], dtype=torch.float32)
                    if source is not None
                    else None
                ),
                velocities=(
                    torch.as_tensor(source["velocities"], dtype=torch.float32)
                    if source is not None
                    else None
                ),
                action=(
                    torch.as_tensor(source["action"], dtype=torch.float32)
                    if source is not None
                    else None
                ),
            )
        )
    return records


def _scenario_rows(
    bundle: PredictionBundle,
    *,
    threshold: float,
    method_id: str,
    method_kind: str,
    training_variant: str | None,
    eval_perturbation: str | None,
    seed: int,
    split: str,
    distribution: str,
    ood_axis: str | None,
) -> list[dict[str, Any]]:
    from actmask.eval.aggregation import aggregate_scenario_metrics
    records = _prediction_records_from_bundle(
        bundle, distribution=distribution, ood_axis=ood_axis
    )
    shared_rows = aggregate_scenario_metrics(
        records,
        threshold,
        method_id=method_id,
        method_kind=method_kind,
        seed=seed,
        split=split,
        distribution=distribution,
        ood_axis=ood_axis or "none",
        threshold_source="id_validation",
        training_variant=training_variant,
        eval_perturbation=eval_perturbation or "none",
    )
    by_scenario: MutableMapping[str, list[int]] = defaultdict(list)
    for index, metadata in enumerate(bundle.metadata):
        by_scenario[str(metadata.get("scenario", "unknown"))].append(index)
    by_scenario = {"__all__": list(range(len(bundle.metadata))), **dict(by_scenario)}
    rows: list[dict[str, Any]] = []
    for shared_row in shared_rows:
        scenario = str(shared_row["scenario"])
        indices = by_scenario[scenario]
        scores = bundle.scores[indices]
        additional = _paired_metrics(bundle, indices, threshold)
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                **shared_row,
                "sample_support": len(indices),
                "num_points": int(scores.size),
                "predicted_positive_rate": float((scores >= threshold).mean()),
                "counterfactual_signed_change_accuracy": additional[
                    "counterfactual_signed_change_accuracy"
                ],
                "counterfactual_probability_response": additional[
                    "counterfactual_probability_response"
                ],
                "predicted_change_prevalence": additional[
                    "predicted_change_prevalence"
                ],
                "target_change_prevalence": additional["target_change_prevalence"],
                "irrelevant_prediction_stability": additional[
                    "irrelevant_prediction_stability"
                ],
            }
        )
    return rows


def _counterfactual_rows(
    bundle: PredictionBundle,
    *,
    threshold: float,
    method_id: str,
    method_kind: str,
    training_variant: str | None,
    eval_perturbation: str | None,
    seed: int,
    split: str,
    distribution: str,
    ood_axis: str | None,
) -> list[dict[str, Any]]:
    from actmask.eval.counterfactual import aggregate_counterfactual_metrics

    records = _prediction_records_from_bundle(
        bundle, distribution=distribution, ood_axis=ood_axis
    )
    rows = aggregate_counterfactual_metrics(
        records,
        threshold=float(threshold),
        method_id=method_id,
        method_kind=method_kind,
        seed=int(seed),
        split=split,
        distribution=distribution,
        ood_axis=ood_axis or "none",
        threshold_source="id_validation",
        training_variant=training_variant,
        eval_perturbation=eval_perturbation or "none",
    )
    return [{"schema_version": SCHEMA_VERSION, **row} for row in rows]


def _save_pr_curve(
    root: Path,
    *,
    seed: int,
    evaluation_set: str,
    method_id: str,
    bundle: PredictionBundle,
) -> dict[str, Any]:
    curve = precision_recall_curve(bundle.scores, bundle.targets)
    directory = root / "pr_curves" / f"seed_{seed}" / _safe_name(evaluation_set)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{_safe_name(method_id)}.npz"
    np.savez_compressed(
        path,
        precision=np.asarray(curve["precision"]),
        recall=np.asarray(curve["recall"]),
        thresholds=np.asarray(curve["thresholds"]),
        average_precision=np.asarray(float(curve["average_precision"])),
        scores=bundle.scores.astype(np.float32),
        targets=bundle.targets.astype(np.uint8),
    )
    return {
        "method_id": method_id,
        "path": str(path),
        "average_precision": float(curve["average_precision"]),
        "precision": np.asarray(curve["precision"]),
        "recall": np.asarray(curve["recall"]),
    }


def _plot_pr_curves(
    root: Path, *, seed: int, evaluation_set: str, curves: Sequence[Mapping[str, Any]]
) -> Path:
    directory = root / "pr_curves" / f"seed_{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{_safe_name(evaluation_set)}.png"
    figure, axis = plt.subplots(figsize=(7.0, 5.2))
    for curve in curves:
        axis.step(
            curve["recall"],
            curve["precision"],
            where="post",
            linewidth=1.25,
            label=f"{curve['method_id']} (AP={curve['average_precision']:.3f})",
        )
    axis.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.02))
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7, loc="lower left")
    axis.set_title(f"Milestone 2A PR curves: {evaluation_set}, seed {seed}")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def _load_trained_variant(
    *,
    variant: str,
    seed: int,
    run_dir: Path,
    experiment_digest: str,
    split_digest: str,
) -> tuple[nn.Module, dict[str, Any], dict[str, Any]]:
    variant_dir = run_dir / _safe_name(variant)
    checkpoint_path = variant_dir / "checkpoint.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Missing {variant} checkpoint for seed {seed}: {checkpoint_path}"
        )
    checkpoint = _load_checkpoint(checkpoint_path)
    expected = {
        "variant": variant,
        "seed": int(seed),
        "experiment_digest": experiment_digest,
        "split_digest": split_digest,
    }
    mismatched = {
        key: (checkpoint.get(key), expected_value)
        for key, expected_value in expected.items()
        if checkpoint.get(key) != expected_value
    }
    if mismatched:
        raise RuntimeError(f"Checkpoint provenance mismatch at {checkpoint_path}: {mismatched}")
    model = _build_variant_model(variant, dict(checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    with (variant_dir / "training_metrics.json").open("r", encoding="utf-8") as handle:
        training_report = json.load(handle)
    with (variant_dir / "validation_threshold.json").open("r", encoding="utf-8") as handle:
        threshold_report = json.load(handle)
    return model, training_report, threshold_report


def _shuffled_view(dataset: Dataset[Any], mode: str, seed: int) -> Dataset[Any]:
    """Use the evaluation API's cross-group derangement when available."""

    try:
        from actmask.models.ablations import (
            shuffled_action_dataset,
            shuffled_time_dataset,
            shuffled_velocity_dataset,
        )

        if mode == "ShuffledVelocity":
            return shuffled_velocity_dataset(dataset, seed=seed)
        if mode == "ShuffledAction":
            return shuffled_action_dataset(dataset, seed=seed)
        if mode == "TimeShuffled":
            return shuffled_time_dataset(dataset, seed=seed)
    except (ImportError, TypeError):
        # FeatureView remains deterministic and is kept as a compatibility
        # fallback for partial/staged installations.
        pass
    return FeatureView(dataset, mode, seed=seed)


def _method_bundle(
    *,
    method_id: str,
    dataset: Dataset[Any],
    models: Mapping[str, nn.Module],
    baselines: Mapping[str, nn.Module],
    seed: int,
    batch_size: int,
    shuffle_seed: int,
    oracle_name: str,
) -> PredictionBundle:
    if method_id in TRAINED_VARIANTS:
        return _predict_learned(models[method_id], dataset, batch_size=batch_size)
    if method_id in SHUFFLED_VARIANTS:
        return _predict_learned(
            models["Full"],
            _shuffled_view(dataset, method_id, shuffle_seed),
            batch_size=batch_size,
        )
    if method_id in baselines:
        return _predict_learned(baselines[method_id], dataset, batch_size=batch_size)
    if method_id == oracle_name:
        return _predict_oracle(dataset)
    raise KeyError(method_id)


def _method_description(method_id: str, oracle_name: str) -> tuple[str, str | None, str | None]:
    if method_id in TRAINED_VARIANTS:
        return "learned", method_id, None
    if method_id in SHUFFLED_VARIANTS:
        return "learned_counterfactual", "Full", method_id
    if method_id == oracle_name:
        return "oracle_upper_bound_excluded", None, None
    return "geometric_baseline", None, None


def evaluate_seed(
    *,
    seed: int,
    datasets: DatasetBundle,
    models: Mapping[str, nn.Module],
    learned_thresholds: Mapping[str, Mapping[str, Any]],
    config: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    evaluation_config = dict(config.get("evaluation", {}))
    batch_size = int(evaluation_config.get("batch_size", 16))
    oracle_name = str(evaluation_config.get("oracle_name", ORACLE_DEFAULT_NAME))
    baselines = build_baselines(config)
    methods = [*TRAINED_VARIANTS, *SHUFFLED_VARIANTS, *baselines.keys(), oracle_name]
    thresholds: dict[str, dict[str, Any]] = {
        variant: dict(learned_thresholds[variant]) for variant in TRAINED_VARIANTS
    }
    validation_bundles: dict[str, PredictionBundle] = {}

    for name, baseline in baselines.items():
        bundle = _predict_learned(baseline, datasets.validation, batch_size=batch_size)
        validation_bundles[name] = bundle
        selection = select_validation_threshold(bundle.scores, bundle.targets)
        selection.update(
            {
                "method_id": name,
                "seed": int(seed),
                "frozen_for": ["id_test", "scenario", "counterfactual", "all_ood_axes"],
            }
        )
        thresholds[name] = selection
    oracle_validation = _predict_oracle(datasets.validation)
    validation_bundles[oracle_name] = oracle_validation
    oracle_selection = select_validation_threshold(
        oracle_validation.scores, oracle_validation.targets
    )
    oracle_selection.update(
        {
            "method_id": oracle_name,
            "seed": int(seed),
            "frozen_for": ["id_test", "scenario", "counterfactual", "all_ood_axes"],
            "fair_comparison": False,
        }
    )
    thresholds[oracle_name] = oracle_selection
    for shuffled_variant in SHUFFLED_VARIANTS:
        thresholds[shuffled_variant] = {
            **dict(thresholds["Full"]),
            "method_id": shuffled_variant,
            "inherited_from": "Full",
            "selection_note": (
                "Full model native ID-validation threshold; evaluation-time "
                "feature/time shuffle was not fitted"
            ),
        }

    geometric_candidates = [
        name
        for name in baselines
        if name not in {"NoMask", "MotionMagnitudeMask"}
    ]
    if not geometric_candidates:
        raise RuntimeError("At least one fair action-conditioned geometric baseline is required")
    selected_fair = max(
        geometric_candidates,
        key=lambda name: (float(thresholds[name]["average_precision"]), name),
    )
    fair_selection = {
        "seed": int(seed),
        "selection_split": "id_validation",
        "selection_metric": "average_precision",
        "eligible_methods": geometric_candidates,
        "validation_average_precision": {
            name: float(thresholds[name]["average_precision"])
            for name in geometric_candidates
        },
        "selected_method": selected_fair,
        "oracle_excluded": oracle_name,
        "test_metrics_were_not_used": True,
    }

    validation_curves: list[dict[str, Any]] = []
    # Learned validation curves are recomputed from the selected checkpoints;
    # shuffled variants intentionally do not participate in threshold fitting.
    for variant in TRAINED_VARIANTS:
        bundle = _predict_learned(models[variant], datasets.validation, batch_size=batch_size)
        validation_curves.append(
            _save_pr_curve(
                output_dir,
                seed=seed,
                evaluation_set="id_validation",
                method_id=variant,
                bundle=bundle,
            )
        )
    for name, bundle in validation_bundles.items():
        validation_curves.append(
            _save_pr_curve(
                output_dir,
                seed=seed,
                evaluation_set="id_validation",
                method_id=name,
                bundle=bundle,
            )
        )
    if bool(dict(config.get("reporting", {})).get("save_pr_plots", True)):
        _plot_pr_curves(
            output_dir, seed=seed, evaluation_set="id_validation", curves=validation_curves
        )

    evaluation_sets: list[tuple[str, Dataset[Any], str, str, str | None]] = [
        ("id_test", datasets.test, "id_test", "ID", "none"),
        (
            "id_irrelevant_counterfactual",
            datasets.irrelevant,
            "counterfactual_test",
            "ID",
            "none",
        ),
    ] + [
        (f"ood_{axis}", dataset, "ood_test", "OOD", axis)
        for axis, dataset in datasets.ood.items()
    ]
    scenario_rows: list[dict[str, Any]] = []
    counterfactual_rows: list[dict[str, Any]] = []
    ranking_rows: list[dict[str, Any]] = []
    global_rows: list[dict[str, Any]] = []
    method_evaluation_rows: MutableMapping[str, list[dict[str, Any]]] = defaultdict(list)
    pr_artifacts: list[str] = []
    failure_records: MutableMapping[str, list[Any]] = defaultdict(list)
    seed_dir = output_dir / "runs" / f"seed_{seed}"

    for set_index, (set_name, dataset, split, distribution, ood_axis) in enumerate(evaluation_sets):
        curves: list[dict[str, Any]] = []
        for method_index, method_id in enumerate(methods):
            shuffle_seed = seed * 1_000_003 + set_index * 10_007 + method_index * 101
            bundle = _method_bundle(
                method_id=method_id,
                dataset=dataset,
                models=models,
                baselines=baselines,
                seed=seed,
                batch_size=batch_size,
                shuffle_seed=shuffle_seed,
                oracle_name=oracle_name,
            )
            threshold = float(thresholds[method_id]["threshold"])
            method_kind, training_variant, eval_perturbation = _method_description(
                method_id, oracle_name
            )
            rows = _scenario_rows(
                bundle,
                threshold=threshold,
                method_id=method_id,
                method_kind=method_kind,
                training_variant=training_variant,
                eval_perturbation=eval_perturbation,
                seed=seed,
                split=split,
                distribution=distribution,
                ood_axis=ood_axis,
            )
            scenario_rows.extend(rows)
            global_row = next(row for row in rows if row["scenario"] == "__all__")
            global_rows.append(global_row)
            method_evaluation_rows[method_id].append(global_row)
            counterfactual_rows.extend(
                _counterfactual_rows(
                    bundle,
                    threshold=threshold,
                    method_id=method_id,
                    method_kind=method_kind,
                    training_variant=training_variant,
                    eval_perturbation=eval_perturbation,
                    seed=seed,
                    split=split,
                    distribution=distribution,
                    ood_axis=ood_axis,
                )
            )
            if method_id in {"Full", selected_fair}:
                failure_records[method_id].extend(
                    _prediction_records_from_bundle(
                        bundle,
                        distribution=distribution,
                        ood_axis=ood_axis,
                        dataset=dataset,
                        retain_inputs=True,
                    )
                )
            curve = _save_pr_curve(
                output_dir,
                seed=seed,
                evaluation_set=set_name,
                method_id=method_id,
                bundle=bundle,
            )
            curves.append(curve)
            pr_artifacts.append(str(curve["path"]))
        if bool(dict(config.get("reporting", {})).get("save_pr_plots", True)):
            pr_artifacts.append(
                str(
                    _plot_pr_curves(
                        output_dir, seed=seed, evaluation_set=set_name, curves=curves
                    )
                )
            )

    # Candidate actions are held-out, multi-action views of the ID test scenes.
    # They are never seen during model training, checkpoint selection, or
    # threshold selection.  Ranking scores are model probabilities; labels are
    # used only after scoring to compute the requested action-selection metrics.
    from actmask.eval.ranking import action_ranking_records

    for method_index, method_id in enumerate(methods):
        shuffle_seed = seed * 1_000_003 + 91_337 + method_index * 101
        bundle = _method_bundle(
            method_id=method_id,
            dataset=datasets.ranking,
            models=models,
            baselines=baselines,
            seed=seed,
            batch_size=batch_size,
            shuffle_seed=shuffle_seed,
            oracle_name=oracle_name,
        )
        threshold = float(thresholds[method_id]["threshold"])
        method_kind, training_variant, eval_perturbation = _method_description(
            method_id, oracle_name
        )
        records = _prediction_records_from_bundle(
            bundle,
            distribution="ID",
            ood_axis="none",
            dataset=datasets.ranking,
            retain_inputs=True,
        )
        if method_id == oracle_name:
            # The excluded oracle ranks actions by exact target-contact utility,
            # which defines the regret reference.  It never participates in a
            # fair learned/geometric comparison.
            for record in records:
                record.metadata["ranking_score"] = float(
                    record.candidate_utility
                    if record.candidate_utility is not None
                    else record.success
                )
        ranking_rows.extend(
            action_ranking_records(
                records,
                method_id=method_id,
                method_kind=method_kind,
                seed=seed,
                split="id_candidate_ranking",
                distribution="ID",
                ood_axis="none",
                training_variant=training_variant,
                eval_perturbation=eval_perturbation or "none",
                candidate_set_key="candidate_set_id",
                candidate_id_key="candidate_id",
                require_success=True,
            )
        )
        if method_id in {"Full", selected_fair}:
            failure_records[method_id].extend(records)

    failure_artifacts: dict[str, Any] | None = None
    if bool(
        dict(config.get("reporting", {})).get("save_failure_visualizations", False)
    ):
        from actmask.visualization.milestone2_failures import (
            generate_failure_visualizations,
        )

        artifacts = generate_failure_visualizations(
            failure_records["Full"],
            failure_records[selected_fair],
            seed_dir / "failure_cases",
            actmask_threshold=float(thresholds["Full"]["threshold"]),
            geometric_threshold=float(thresholds[selected_fair]["threshold"]),
            candidate_set_key="candidate_set_id",
            candidate_id_key="candidate_id",
            actmask_name="Full ActMask",
            geometric_name=selected_fair,
            manifest_metadata={
                "seed": int(seed),
                "geometric_baseline": selected_fair,
                "actmask_threshold": float(thresholds["Full"]["threshold"]),
                "geometric_threshold": float(thresholds[selected_fair]["threshold"]),
                "selection_split": "id_validation",
            },
        )
        failure_artifacts = {
            "manifest": str(artifacts.manifest_path),
            "paths": {key: str(path) for key, path in artifacts.paths.items()},
            "qualifying": {
                selection.category: bool(selection.qualifying)
                for selection in artifacts.selections
            },
        }

    _write_json(
        seed_dir / "action_ranking_metrics.json",
        {
            "schema_version": SCHEMA_VERSION,
            "seed": int(seed),
            "rows": ranking_rows,
            "score_reduction": "mean predicted point-mask probability unless explicitly overridden",
            "oracle_note": "GeometricOracle is an excluded exact-trajectory upper bound",
            **_ranking_tie_metadata(),
        },
    )
    for variant in TRAINED_VARIANTS:
        _write_json(
            seed_dir / _safe_name(variant) / "evaluation_metrics.json",
            {
                "schema_version": SCHEMA_VERSION,
                "seed": int(seed),
                "method_id": variant,
                "threshold": thresholds[variant],
                "global_rows": method_evaluation_rows[variant],
            },
        )
    report = {
        "schema_version": SCHEMA_VERSION,
        "seed": int(seed),
        "device": "cpu",
        "thresholds": thresholds,
        "strongest_fair_baseline_selection": fair_selection,
        "global_metrics": global_rows,
        "scenario_row_count": len(scenario_rows),
        "counterfactual_row_count": len(counterfactual_rows),
        "action_ranking_row_count": len(ranking_rows),
        "failure_artifacts": failure_artifacts,
        "pr_artifacts": pr_artifacts,
    }
    _write_json(seed_dir / "run_metrics.json", report)
    return {
        "report": report,
        "scenario_rows": scenario_rows,
        "counterfactual_rows": counterfactual_rows,
        "ranking_rows": ranking_rows,
        "thresholds": list(thresholds.values()),
        "fair_selection": fair_selection,
        "failure_artifacts": failure_artifacts,
    }


SUMMARY_METRICS = (
    "iou",
    "precision",
    "recall",
    "f1",
    "average_precision",
    "pr_auc",
    "positive_prevalence",
    "paired_consistency",
    "success_ranking_accuracy",
    "counterfactual_signed_change_accuracy",
    "counterfactual_probability_response",
    "irrelevant_prediction_stability",
)

COUNTERFACTUAL_SUMMARY_METRICS = (
    "change_iou",
    "change_ap",
    "change_pr_auc",
    "changed_region_probability_delta",
    "invariant_binary_agreement",
    "invariant_probability_mae",
    "invariant_probability_stability",
    "matching_accuracy",
    "matching_margin",
    "signed_direction_accuracy",
    "signed_response_accuracy",
)


def seed_summary_records(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate each method/scenario across seeds with full uncertainty fields."""

    grouped: MutableMapping[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    keys = (
        "method_id",
        "method_kind",
        "training_variant",
        "eval_perturbation",
        "split",
        "distribution",
        "ood_axis",
        "scenario",
    )
    for row in rows:
        grouped[tuple(row.get(key) for key in keys)].append(row)
    summaries: list[dict[str, Any]] = []
    for group_key, group_rows in sorted(grouped.items(), key=lambda item: str(item[0])):
        base = dict(zip(keys, group_key, strict=True))
        seeds = sorted({int(row["seed"]) for row in group_rows})
        for metric in SUMMARY_METRICS:
            values = np.asarray(
                [float(row[metric]) for row in group_rows if row.get(metric) is not None],
                dtype=np.float64,
            )
            if values.size != len(group_rows):
                continue
            summaries.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    **base,
                    "metric": metric,
                    "seed_count": len(seeds),
                    "seeds": seeds,
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                }
            )
    return summaries


def counterfactual_seed_summary(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Report counterfactual diagnostics across seeds without pooling seeds.

    The raw counterfactual file retains every run/scenario/type record.  This
    companion summary uses one matching record per seed for each exact
    method/split/scenario/counterfactual-type cell, so its uncertainty fields
    remain comparable to the mask and ranking multi-seed summaries.
    """

    keys = (
        "method_id",
        "method_kind",
        "training_variant",
        "eval_perturbation",
        "split",
        "distribution",
        "ood_axis",
        "counterfactual_type",
        "scenario",
    )
    grouped: MutableMapping[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("row_type") != "run":
            continue
        grouped[tuple(row.get(key) for key in keys)].append(row)
    summaries: list[dict[str, Any]] = []
    for group_key, group_rows in sorted(grouped.items(), key=lambda item: str(item[0])):
        base = dict(zip(keys, group_key, strict=True))
        seeds = sorted({int(row["seed"]) for row in group_rows if row.get("seed") is not None})
        if len(seeds) != len(group_rows):
            raise ValueError("duplicate or missing seed in counterfactual rows")
        for metric in COUNTERFACTUAL_SUMMARY_METRICS:
            values = np.asarray(
                [float(row[metric]) for row in group_rows if row.get(metric) is not None],
                dtype=np.float64,
            )
            if values.size != len(group_rows) or not np.isfinite(values).all():
                continue
            summaries.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "row_type": "counterfactual_summary",
                    **base,
                    "metric": metric,
                    "seed_count": len(seeds),
                    "seeds": seeds,
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                }
            )
    return summaries


RANKING_SUMMARY_METRICS = (
    "top1_successful_action_accuracy",
    "top3_success_recall",
    "top3_success_hit",
    "pairwise_ranking_accuracy",
    "ranking_auc",
    "global_binary_ranking_auc",
    "oracle_regret",
)


def action_ranking_seed_summary(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Aggregate only one ranking-aggregate row per method/seed.

    Per-scene ranking rows are intentionally excluded: treating every scene as
    an independent seed observation would understate seed variation and violate
    the five-seed reporting contract.
    """

    aggregate_rows = [
        row for row in rows if row.get("row_type") == "action_ranking_aggregate"
    ]
    keys = (
        "method_id",
        "method_kind",
        "training_variant",
        "eval_perturbation",
        "split",
        "distribution",
        "ood_axis",
    )
    grouped: MutableMapping[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in aggregate_rows:
        grouped[tuple(row.get(key) for key in keys)].append(row)
    output: list[dict[str, Any]] = []
    for group_key, group_rows in sorted(grouped.items(), key=lambda item: str(item[0])):
        seeds = sorted({int(row["seed"]) for row in group_rows if row.get("seed") is not None})
        if len(seeds) != len(group_rows):
            raise ValueError("duplicate or missing seed in action ranking aggregate rows")
        base = dict(zip(keys, group_key, strict=True))
        for metric in RANKING_SUMMARY_METRICS:
            values = [row.get(metric) for row in group_rows]
            if any(value is None for value in values):
                continue
            numeric = np.asarray(values, dtype=np.float64)
            if not np.isfinite(numeric).all():
                continue
            output.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    **base,
                    "metric": metric,
                    "seed_count": len(seeds),
                    "seeds": seeds,
                    "mean": float(np.mean(numeric)),
                    "std": float(np.std(numeric, ddof=1)) if len(numeric) > 1 else 0.0,
                    "min": float(np.min(numeric)),
                    "max": float(np.max(numeric)),
                }
            )
    return output


def id_ood_generalization_gap_rows(
    scenario_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Report paired ID-minus-OOD point-mask generalization gaps per seed.

    A gap is positive when a method performs better on ID.  Rows are emitted
    only for the global scenario and never use test metrics for selection.
    """

    id_rows: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    ood_rows: list[Mapping[str, Any]] = []
    key_fields = ("method_id", "seed", "method_kind", "training_variant", "eval_perturbation")
    for row in scenario_rows:
        if row.get("scenario") != "__all__":
            continue
        if row.get("split") == "id_test":
            id_rows[tuple(row.get(key) for key in key_fields)] = row
        elif row.get("split") == "ood_test":
            ood_rows.append(row)
    gaps: list[dict[str, Any]] = []
    for row in sorted(ood_rows, key=lambda item: str(tuple(item.get(key) for key in key_fields))):
        key = tuple(row.get(field) for field in key_fields)
        id_row = id_rows.get(key)
        if id_row is None:
            continue
        gap: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "row_type": "id_ood_gap",
            "method_id": row.get("method_id"),
            "method_kind": row.get("method_kind"),
            "training_variant": row.get("training_variant"),
            "eval_perturbation": row.get("eval_perturbation"),
            "seed": row.get("seed"),
            "ood_axis": row.get("ood_axis"),
            "id_split": "id_test",
            "ood_split": "ood_test",
        }
        for metric in ("iou", "precision", "recall", "f1", "average_precision", "pr_auc"):
            if id_row.get(metric) is not None and row.get(metric) is not None:
                gap[f"id_{metric}"] = float(id_row[metric])
                gap[f"ood_{metric}"] = float(row[metric])
                gap[f"id_minus_ood_{metric}"] = float(id_row[metric]) - float(row[metric])
        gaps.append(gap)
    return gaps


def id_ood_generalization_gap_summary(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Aggregate every ID-minus-OOD metric across deterministic seeds.

    These rows deliberately retain the OOD axis.  Aggregating an axis first
    and then aggregating across axes would hide the requested held-out-axis
    generalization gaps, so each method/axis/metric receives its own full
    mean, standard deviation, minimum, and maximum record.
    """

    metric_prefix = "id_minus_ood_"
    key_fields = (
        "method_id",
        "method_kind",
        "training_variant",
        "eval_perturbation",
        "ood_axis",
    )
    grouped: MutableMapping[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(field) for field in key_fields)].append(row)
    summaries: list[dict[str, Any]] = []
    for group_key, group_rows in sorted(grouped.items(), key=lambda item: str(item[0])):
        base = dict(zip(key_fields, group_key, strict=True))
        seeds = sorted({int(row["seed"]) for row in group_rows if row.get("seed") is not None})
        if len(seeds) != len(group_rows):
            raise ValueError("duplicate or missing seed in ID/OOD gap rows")
        metrics = sorted(
            {
                str(key)
                for row in group_rows
                for key in row
                if str(key).startswith(metric_prefix) and row.get(key) is not None
            }
        )
        for metric in metrics:
            values = np.asarray(
                [float(row[metric]) for row in group_rows if row.get(metric) is not None],
                dtype=np.float64,
            )
            # A seed-dependent missing metric is not a valid five-seed
            # generalization estimate.  Omit it instead of silently changing
            # the denominator.
            if values.size != len(group_rows):
                continue
            if not np.isfinite(values).all():
                continue
            summaries.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "row_type": "id_ood_gap_summary",
                    **base,
                    "metric": metric,
                    "seed_count": len(seeds),
                    "seeds": seeds,
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                }
            )
    return summaries


def _summary_lookup(
    summaries: Sequence[Mapping[str, Any]],
    *,
    method: str,
    metric: str,
    split: str = "id_test",
    scenario: str = "__all__",
    ood_axis: str | None = "none",
) -> Mapping[str, Any] | None:
    for row in summaries:
        if (
            row.get("method_id") == method
            and row.get("metric") == metric
            and row.get("split") == split
            and row.get("scenario") == scenario
            and row.get("ood_axis") == ood_axis
        ):
            return row
    return None


def _ranking_summary_lookup(
    summaries: Sequence[Mapping[str, Any]], *, method: str, metric: str
) -> Mapping[str, Any] | None:
    for row in summaries:
        if row.get("method_id") == method and row.get("metric") == metric:
            return row
    return None


def _feature_ablation_rows(
    summaries: Sequence[Mapping[str, Any]], scenario_rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    methods = (*TRAINED_VARIANTS, *SHUFFLED_VARIANTS)
    output: list[dict[str, Any]] = []
    full_by_seed = {
        int(row["seed"]): row
        for row in scenario_rows
        if row.get("method_id") == "Full"
        and row.get("split") == "id_test"
        and row.get("scenario") == "__all__"
    }
    for method in methods:
        record: dict[str, Any] = {
            "method_id": method,
            "training_variant": "Full" if method in SHUFFLED_VARIANTS else method,
            "evaluation_perturbation": method if method in SHUFFLED_VARIANTS else "none",
            "threshold_policy": (
                "inherit Full ID-validation threshold"
                if method in SHUFFLED_VARIANTS
                else "native ID-validation threshold"
            ),
        }
        for metric in ("average_precision", "pr_auc", "iou", "f1", "paired_consistency"):
            summary = _summary_lookup(summaries, method=method, metric=metric)
            if summary is None:
                continue
            for statistic in ("mean", "std", "min", "max"):
                record[f"{metric}_{statistic}"] = float(summary[statistic])
        deltas: list[float] = []
        if method != "Full":
            for row in scenario_rows:
                if (
                    row.get("method_id") == method
                    and row.get("split") == "id_test"
                    and row.get("scenario") == "__all__"
                    and int(row["seed"]) in full_by_seed
                ):
                    deltas.append(
                        float(full_by_seed[int(row["seed"])]["average_precision"])
                        - float(row["average_precision"])
                    )
        record["full_minus_variant_ap_mean"] = float(np.mean(deltas)) if deltas else 0.0
        output.append(record)
    return output


def _feature_markdown(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# Feature ablation results",
        "",
        "All thresholds were selected on ID validation. Shuffled evaluations use the Full model's native validation threshold.",
        "",
        "| Variant | AP mean ± std | IoU mean ± std | Pair consistency | Full − variant AP |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {method_id} | {ap:.4f} ± {ap_std:.4f} | {iou:.4f} ± {iou_std:.4f} | "
            "{pair:.4f} | {delta:.4f} |".format(
                method_id=row["method_id"],
                ap=float(row.get("average_precision_mean", 0.0)),
                ap_std=float(row.get("average_precision_std", 0.0)),
                iou=float(row.get("iou_mean", 0.0)),
                iou_std=float(row.get("iou_std", 0.0)),
                pair=float(row.get("paired_consistency_mean", 0.0)),
                delta=float(row.get("full_minus_variant_ap_mean", 0.0)),
            )
        )
    return "\n".join(lines) + "\n"


def _most_common(values: Sequence[str]) -> str:
    counts: MutableMapping[str, int] = defaultdict(int)
    for value in values:
        counts[value] += 1
    return max(sorted(counts), key=lambda value: counts[value])


def _render_summary(
    *,
    config: Mapping[str, Any],
    summaries: Sequence[Mapping[str, Any]],
    ranking_summaries: Sequence[Mapping[str, Any]],
    feature_rows: Sequence[Mapping[str, Any]],
    fair_selections: Sequence[Mapping[str, Any]],
    runtime_seconds: float,
    test_count: int,
) -> str:
    selected = _most_common([str(item["selected_method"]) for item in fair_selections])
    full_ap = _summary_lookup(summaries, method="Full", metric="average_precision")
    full_iou = _summary_lookup(summaries, method="Full", metric="iou")
    fair_ap = _summary_lookup(summaries, method=selected, metric="average_precision")
    fair_iou = _summary_lookup(summaries, method=selected, metric="iou")
    oracle_name = str(dict(config.get("evaluation", {})).get("oracle_name", ORACLE_DEFAULT_NAME))
    oracle_ap = _summary_lookup(summaries, method=oracle_name, metric="average_precision")
    if any(value is None for value in (full_ap, full_iou, fair_ap, fair_iou, oracle_ap)):
        raise RuntimeError("Cannot render summary because required aggregate rows are missing")
    assert full_ap is not None and full_iou is not None
    assert fair_ap is not None and fair_iou is not None and oracle_ap is not None

    feature_by_name = {str(row["method_id"]): row for row in feature_rows}
    velocity_deltas = [
        float(feature_by_name[name]["full_minus_variant_ap_mean"])
        for name in ("NoVelocity", "ShuffledVelocity")
    ]
    action_deltas = [
        float(feature_by_name[name]["full_minus_variant_ap_mean"])
        for name in ("NoAction", "ShuffledAction")
    ]
    meaningful_cutoff = 0.02
    velocity_evidence = min(velocity_deltas) >= meaningful_cutoff
    action_evidence = min(action_deltas) >= meaningful_cutoff
    beats_fair = float(full_ap["mean"]) > float(fair_ap["mean"])

    full_rank_top1 = _ranking_summary_lookup(
        ranking_summaries, method="Full", metric="top1_successful_action_accuracy"
    )
    fair_rank_top1 = _ranking_summary_lookup(
        ranking_summaries, method=selected, metric="top1_successful_action_accuracy"
    )
    full_rank_top3 = _ranking_summary_lookup(
        ranking_summaries, method="Full", metric="top3_success_recall"
    )
    ranking_lines: list[str]
    if full_rank_top1 is not None and fair_rank_top1 is not None and full_rank_top3 is not None:
        ranking_delta = float(full_rank_top1["mean"]) - float(fair_rank_top1["mean"])
        ranking_lines = [
            "## Candidate-action ranking",
            "",
            "Candidate sets contain successful, failed, wrong-timing, wrong-direction, and near-miss actions. Non-oracle methods score candidates by mean predicted point-mask probability; GeometricOracle uses exact candidate utility only as an excluded upper bound. Labels are used only after ranking.",
            "",
            f"Scores within an absolute `{RANKING_SCORE_TIE_TOLERANCE:g}` tie band use uniform expected top-k/regret values; pairwise and AUC ties score 0.5, while candidate IDs are display provenance only.",
            "",
            f"Full top-1 successful-action accuracy is `{float(full_rank_top1['mean']):.4f} ± {float(full_rank_top1['std']):.4f}`; top-3 success recall is `{float(full_rank_top3['mean']):.4f} ± {float(full_rank_top3['std']):.4f}`. Relative to {selected}, Full's top-1 delta is `{ranking_delta:.4f}`.",
            "",
        ]
    else:
        ranking_lines = [
            "## Candidate-action ranking",
            "",
            "Ranking aggregate rows were unavailable for this partial-stage report.",
            "",
        ]

    ood_rows: list[tuple[str, float]] = []
    for axis in REQUIRED_OOD_AXES:
        row = _summary_lookup(
            summaries,
            method="Full",
            metric="average_precision",
            split="ood_test",
            ood_axis=axis,
        )
        if row is not None:
            ood_rows.append((axis, float(row["mean"])))
    weakest_axis, weakest_ap = min(ood_rows, key=lambda item: item[1]) if ood_rows else ("missing", 0.0)
    recommendation = (
        "GO to a scoped Milestone 2B simulator integration"
        if beats_fair and velocity_evidence and action_evidence and len(ood_rows) == 7
        else "NO-GO for simulator integration until the failed validation gates are addressed"
    )
    seeds = list(dict(config.get("experiment", {}))["seeds"])
    lines = [
        "# ActMask Milestone 2A summary",
        "",
        f"This report is computed from {len(seeds)} deterministic CPU seeds and {test_count} ID test samples. No best-seed filtering was used.",
        "",
        "## Main comparison",
        "",
        f"The strongest fair geometric baseline was selected on ID-validation AP before test: **{selected}**. The exact-trajectory oracle is an upper bound and is excluded from this fair comparison.",
        "",
        "| Method | ID test AP mean ± std (min–max) | ID test IoU mean ± std |",
        "| --- | ---: | ---: |",
        f"| Full ActMask | {float(full_ap['mean']):.4f} ± {float(full_ap['std']):.4f} ({float(full_ap['min']):.4f}–{float(full_ap['max']):.4f}) | {float(full_iou['mean']):.4f} ± {float(full_iou['std']):.4f} |",
        f"| {selected} | {float(fair_ap['mean']):.4f} ± {float(fair_ap['std']):.4f} ({float(fair_ap['min']):.4f}–{float(fair_ap['max']):.4f}) | {float(fair_iou['mean']):.4f} ± {float(fair_iou['std']):.4f} |",
        f"| {oracle_name} (excluded oracle) | {float(oracle_ap['mean']):.4f} ± {float(oracle_ap['std']):.4f} | — |",
        "",
        f"Full {'outperforms' if beats_fair else 'does not outperform'} the selected fair baseline in mean ID-test AP.",
        "",
        "## Feature-use diagnostics",
        "",
        f"Velocity evidence gate (both NoVelocity and ShuffledVelocity lose at least {meaningful_cutoff:.2f} AP): **{'pass' if velocity_evidence else 'fail'}**; AP losses {velocity_deltas[0]:.4f} and {velocity_deltas[1]:.4f}.",
        f"Action evidence gate (both NoAction and ShuffledAction lose at least {meaningful_cutoff:.2f} AP): **{'pass' if action_evidence else 'fail'}**; AP losses {action_deltas[0]:.4f} and {action_deltas[1]:.4f}.",
        "",
        "## OOD diagnostics",
        "",
        f"All {len(ood_rows)} required OOD axes were evaluated with frozen ID-validation thresholds. The weakest Full-model mean AP is `{weakest_ap:.4f}` on `{weakest_axis}`.",
        "",
        *ranking_lines,
        "## Recommendation",
        "",
        f"**{recommendation}.** This automated gate is based on the fair-baseline, feature-dependence, and OOD-completeness checks above; scenario rows and failure visualizations should still be inspected before integration.",
        "",
        "## Reproducibility",
        "",
        f"- Command: `{CANONICAL_COMMAND}`",
        f"- Runtime: {runtime_seconds:.3f} seconds",
        f"- Device assertion: CPU-only; CUDA used = false",
        f"- Seeds: `{seeds}`",
        "- Threshold source: ID validation only; frozen for ID test, scenarios, counterfactuals, and OOD",
    ]
    return "\n".join(lines) + "\n"


def _write_summary_from_evaluation_rows(
    *,
    config: Mapping[str, Any],
    output_dir: Path,
    scenario_rows: Sequence[Mapping[str, Any]],
    ranking_rows: Sequence[Mapping[str, Any]],
    fair_selections: Sequence[Mapping[str, Any]],
    runtime_seconds: float,
    test_count: int,
) -> None:
    """Render the human summary from persisted evaluation semantics only."""

    summaries = seed_summary_records(scenario_rows)
    ranking_summaries = action_ranking_seed_summary(ranking_rows)
    feature_rows = _feature_ablation_rows(summaries, scenario_rows)
    _write_text(
        output_dir / "summary.md",
        _render_summary(
            config=config,
            summaries=summaries,
            ranking_summaries=ranking_summaries,
            feature_rows=feature_rows,
            fair_selections=fair_selections,
            runtime_seconds=runtime_seconds,
            test_count=test_count,
        ),
    )


def write_reports(
    *,
    config: Mapping[str, Any],
    output_dir: Path,
    scenario_rows: Sequence[Mapping[str, Any]],
    counterfactual_rows: Sequence[Mapping[str, Any]],
    ranking_rows: Sequence[Mapping[str, Any]],
    thresholds: Sequence[Mapping[str, Any]],
    fair_selections: Sequence[Mapping[str, Any]],
    runtime_seconds: float,
    test_count: int,
) -> dict[str, Any]:
    summaries = seed_summary_records(scenario_rows)
    counterfactual_summaries = counterfactual_seed_summary(counterfactual_rows)
    ranking_summaries = action_ranking_seed_summary(ranking_rows)
    gap_rows = id_ood_generalization_gap_rows(scenario_rows)
    gap_summaries = id_ood_generalization_gap_summary(gap_rows)
    feature_rows = _feature_ablation_rows(summaries, scenario_rows)
    scenario_report = {
        "schema_version": SCHEMA_VERSION,
        "threshold_policy": "ID-validation selected and frozen",
        "rows": list(scenario_rows),
    }
    counterfactual_report = {
        "schema_version": SCHEMA_VERSION,
        "definitions": {
            "paired_consistency": "IoU between predicted and ground-truth XOR change masks",
            "counterfactual_signed_change_accuracy": "fraction of GT-changing points whose probability changes in the correct direction",
            "irrelevant_prediction_stability": "one minus mean absolute probability change on GT-invariant points",
            "matching_accuracy": "per-pair correct-versus-swapped prediction/ground-truth assignment accuracy; exact ties score 0.5",
            "matching_margin": "mean Bernoulli-log-likelihood of correct assignments minus swapped assignments",
        },
        "rows": list(counterfactual_rows),
    }
    counterfactual_multi_seed_report = {
        "schema_version": SCHEMA_VERSION,
        "seed_count": len(dict(config.get("experiment", {}))["seeds"]),
        "statistics": ["mean", "std", "min", "max"],
        "rows": counterfactual_summaries,
    }
    ranking_report = {
        "schema_version": SCHEMA_VERSION,
        "score_reduction": "mean predicted point-mask probability unless an explicit score is recorded",
        "candidate_types": [
            "successful",
            "failed",
            "wrong_timing",
            "wrong_direction",
            "near_miss",
        ],
        "metrics": [
            "top1_successful_action_accuracy",
            "top3_success_recall",
            "pairwise_ranking_accuracy",
            "ranking_auc",
            "oracle_regret",
        ],
        "oracle_note": "GeometricOracle is an exact-trajectory upper bound and excluded from fair learned/geometric comparisons",
        **_ranking_tie_metadata(),
        "rows": list(ranking_rows),
    }
    ranking_multi_seed_report = {
        "schema_version": SCHEMA_VERSION,
        "seed_count": len(dict(config.get("experiment", {}))["seeds"]),
        "statistics": ["mean", "std", "min", "max"],
        "rows": ranking_summaries,
    }
    gap_report = {
        "schema_version": SCHEMA_VERSION,
        "definition": "ID metric minus same-seed metric on the named OOD axis; positive means ID performance is higher",
        "rows": gap_rows,
    }
    gap_summary_report = {
        "schema_version": SCHEMA_VERSION,
        "seed_count": len(dict(config.get("experiment", {}))["seeds"]),
        "statistics": ["mean", "std", "min", "max"],
        "rows": gap_summaries,
    }
    multi_seed_report = {
        "schema_version": SCHEMA_VERSION,
        "seed_count": len(dict(config.get("experiment", {}))["seeds"]),
        "statistics": ["mean", "std", "min", "max"],
        "rows": summaries,
    }
    _write_json(output_dir / "scenario_metrics.json", scenario_report)
    _write_csv(output_dir / "scenario_metrics.csv", scenario_rows)
    _write_json(output_dir / "counterfactual_metrics.json", counterfactual_report)
    _write_csv(output_dir / "counterfactual_metrics.csv", counterfactual_rows)
    _write_json(
        output_dir / "counterfactual_multi_seed_summary.json",
        counterfactual_multi_seed_report,
    )
    _write_csv(
        output_dir / "counterfactual_multi_seed_summary.csv", counterfactual_summaries
    )
    _write_json(output_dir / "action_ranking_metrics.json", ranking_report)
    _write_csv(output_dir / "action_ranking_metrics.csv", ranking_rows)
    _write_json(
        output_dir / "action_ranking_multi_seed_summary.json", ranking_multi_seed_report
    )
    _write_csv(
        output_dir / "action_ranking_multi_seed_summary.csv", ranking_summaries
    )
    _write_json(output_dir / "id_ood_generalization_gap.json", gap_report)
    _write_csv(output_dir / "id_ood_generalization_gap.csv", gap_rows)
    _write_json(
        output_dir / "id_ood_generalization_gap_summary.json", gap_summary_report
    )
    _write_csv(
        output_dir / "id_ood_generalization_gap_summary.csv", gap_summaries
    )
    _write_json(output_dir / "validation_thresholds.json", {"rows": list(thresholds)})
    _write_json(
        output_dir / "strongest_fair_baseline_selection.json",
        {"rows": list(fair_selections), "oracle_excluded": True},
    )
    _write_json(output_dir / "multi_seed_summary.json", multi_seed_report)
    _write_csv(output_dir / "multi_seed_summary.csv", summaries)
    _write_csv(output_dir / "feature_ablation.csv", feature_rows)
    _write_text(output_dir / "feature_ablation.md", _feature_markdown(feature_rows))
    _write_summary_from_evaluation_rows(
        config=config,
        output_dir=output_dir,
        scenario_rows=scenario_rows,
        ranking_rows=ranking_rows,
        fair_selections=fair_selections,
        runtime_seconds=runtime_seconds,
        test_count=test_count,
    )
    return {
        "summary_rows": len(summaries),
        "scenario_rows": len(scenario_rows),
        "counterfactual_rows": len(counterfactual_rows),
        "counterfactual_summary_rows": len(counterfactual_summaries),
        "action_ranking_rows": len(ranking_rows),
        "action_ranking_summary_rows": len(ranking_summaries),
        "id_ood_gap_rows": len(gap_rows),
        "id_ood_gap_summary_rows": len(gap_summaries),
        "feature_ablation_rows": len(feature_rows),
        "summary_path": str(output_dir / "summary.md"),
    }


def run_milestone2(
    config: Mapping[str, Any],
    *,
    config_path: Path | None = None,
    stages: Sequence[str] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Execute configured stages and return the completion manifest.

    Tests may pass fewer seeds in an in-memory tiny config.  The checked-in
    production YAML is separately asserted by tests to contain exactly five.
    """

    config = _validate_config(config)
    experiment = dict(config.get("experiment", {}))
    configured_stages = tuple(
        str(stage) for stage in experiment.get("stages", ("prepare", "train", "evaluate", "report"))
    )
    selected_stages = tuple(stages) if stages is not None else configured_stages
    allowed_stages = ("prepare", "train", "evaluate", "report")
    unknown = set(selected_stages) - set(allowed_stages)
    if unknown:
        raise ValueError(f"Unknown stages: {sorted(unknown)}")
    output_dir = resolve_project_path(
        experiment.get("output_dir", "outputs/actmask/milestone2_evaluation")
    )
    milestone1_dir = resolve_project_path("outputs/actmask")
    if output_dir == milestone1_dir:
        raise ValueError("Milestone 2 output_dir must be nested below outputs/actmask, not replace it")
    output_dir.mkdir(parents=True, exist_ok=True)
    started_wall = _utc_now()
    started = time.perf_counter()
    seeds = [int(seed) for seed in experiment["seeds"]]
    training_config = dict(config.get("training", {}))
    model_config = dict(config.get("model", {}))
    _seed_everything(seeds[0], int(training_config.get("num_threads", 1)))
    environment = _environment_manifest(config, config_path)
    experiment_digest = str(environment["config_digest_sha256"])
    run_manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "started_at": started_wall,
        "stages_requested": list(selected_stages),
        "seeds": seeds,
        "seed_count": len(seeds),
        "trained_variants": list(TRAINED_VARIANTS),
        "evaluation_perturbations": list(SHUFFLED_VARIANTS),
        "environment": environment,
        "resolved_config": config,
        "output_dir": str(output_dir),
    }
    _write_json(output_dir / "run_manifest.json", run_manifest)
    _write_json(output_dir / "resolved_config.json", config)

    try:
        datasets = build_datasets(config)
        _write_json(output_dir / "split_manifest.json", datasets.manifest)
        split_digest = str(datasets.manifest["split_digest_sha256"])
        models_by_seed: dict[int, dict[str, nn.Module]] = {}
        thresholds_by_seed: dict[int, dict[str, dict[str, Any]]] = {}
        training_runtime = 0.0

        if "train" in selected_stages:
            for seed in seeds:
                seed_started = time.perf_counter()
                run_dir = output_dir / "runs" / f"seed_{seed}"
                models: dict[str, nn.Module] = {}
                thresholds: dict[str, dict[str, Any]] = {}
                training_reports: dict[str, Any] = {}
                for variant in TRAINED_VARIANTS:
                    model, training_report, threshold_report = train_variant(
                        variant=variant,
                        seed=seed,
                        train_dataset=datasets.train,
                        val_dataset=datasets.validation,
                        model_config=model_config,
                        training_config=training_config,
                        run_dir=run_dir,
                        experiment_digest=experiment_digest,
                        split_digest=split_digest,
                        resume=bool(experiment.get("resume", True)) and not force,
                    )
                    models[variant] = model
                    thresholds[variant] = threshold_report
                    training_reports[variant] = training_report
                seed_manifest = {
                    "schema_version": SCHEMA_VERSION,
                    "seed": seed,
                    "device": "cpu",
                    "variants": list(TRAINED_VARIANTS),
                    "identical_data_digest": split_digest,
                    "identical_training_settings_digest": _digest(training_config),
                    "identical_model_architecture_digest": _digest(model_config),
                    "from_scratch_per_variant": True,
                    "feature_ablation_is_only_controlled_difference": True,
                    "runtime_seconds": time.perf_counter() - seed_started,
                    "training_reports": {
                        variant: str(run_dir / variant / "training_metrics.json")
                        for variant in TRAINED_VARIANTS
                    },
                }
                _write_json(run_dir / "run_manifest.json", seed_manifest)
                models_by_seed[seed] = models
                thresholds_by_seed[seed] = thresholds
                training_runtime += float(seed_manifest["runtime_seconds"])

        if "evaluate" in selected_stages:
            scenario_rows: list[dict[str, Any]] = []
            counterfactual_rows: list[dict[str, Any]] = []
            ranking_rows: list[dict[str, Any]] = []
            threshold_rows: list[dict[str, Any]] = []
            fair_selections: list[dict[str, Any]] = []
            for seed in seeds:
                if seed not in models_by_seed:
                    models_by_seed[seed] = {}
                    thresholds_by_seed[seed] = {}
                    for variant in TRAINED_VARIANTS:
                        model, _, threshold_report = _load_trained_variant(
                            variant=variant,
                            seed=seed,
                            run_dir=output_dir / "runs" / f"seed_{seed}",
                            experiment_digest=experiment_digest,
                            split_digest=split_digest,
                        )
                        models_by_seed[seed][variant] = model
                        thresholds_by_seed[seed][variant] = threshold_report
                seed_result = evaluate_seed(
                    seed=seed,
                    datasets=datasets,
                    models=models_by_seed[seed],
                    learned_thresholds=thresholds_by_seed[seed],
                    config=config,
                    output_dir=output_dir,
                )
                scenario_rows.extend(seed_result["scenario_rows"])
                counterfactual_rows.extend(seed_result["counterfactual_rows"])
                ranking_rows.extend(seed_result["ranking_rows"])
                threshold_rows.extend(seed_result["thresholds"])
                fair_selections.append(seed_result["fair_selection"])
            # Raw computed records are persisted immediately, so report-only
            # regeneration never has to infer or fabricate an evaluation.
            _write_json(
                output_dir / "scenario_metrics.json",
                {"schema_version": SCHEMA_VERSION, "rows": scenario_rows},
            )
            _write_json(
                output_dir / "counterfactual_metrics.json",
                {"schema_version": SCHEMA_VERSION, "rows": counterfactual_rows},
            )
            _write_json(
                output_dir / "action_ranking_metrics.json",
                {"schema_version": SCHEMA_VERSION, "rows": ranking_rows},
            )
            _write_json(output_dir / "validation_thresholds.json", {"rows": threshold_rows})
            _write_json(
                output_dir / "strongest_fair_baseline_selection.json",
                {"rows": fair_selections, "oracle_excluded": True},
            )
        elif "report" in selected_stages:
            def load_rows(path: Path) -> list[dict[str, Any]]:
                if not path.is_file():
                    raise FileNotFoundError(
                        f"Report stage requires existing evaluation artifact {path}"
                    )
                with path.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                return list(payload["rows"])

            scenario_rows = load_rows(output_dir / "scenario_metrics.json")
            counterfactual_rows = load_rows(output_dir / "counterfactual_metrics.json")
            ranking_rows = load_rows(output_dir / "action_ranking_metrics.json")
            threshold_rows = load_rows(output_dir / "validation_thresholds.json")
            fair_selections = load_rows(
                output_dir / "strongest_fair_baseline_selection.json"
            )
        else:
            scenario_rows = []
            counterfactual_rows = []
            ranking_rows = []
            threshold_rows = []
            fair_selections = []

        report_result: dict[str, Any] | None = None
        if "report" in selected_stages:
            report_result = write_reports(
                config=config,
                output_dir=output_dir,
                scenario_rows=scenario_rows,
                counterfactual_rows=counterfactual_rows,
                ranking_rows=ranking_rows,
                thresholds=threshold_rows,
                fair_selections=fair_selections,
                runtime_seconds=time.perf_counter() - started,
                test_count=len(datasets.test),
            )

        # Reporting itself can take several seconds.  Freeze the completion
        # runtime only after all report artifacts exist, then use that *same*
        # value in the two user-facing authoritative locations.  This avoids
        # a summary/manifest discrepancy and applies equally to report-only
        # regeneration.
        completed_runtime = time.perf_counter() - started
        if "report" in selected_stages:
            _write_summary_from_evaluation_rows(
                config=config,
                output_dir=output_dir,
                scenario_rows=scenario_rows,
                ranking_rows=ranking_rows,
                fair_selections=fair_selections,
                runtime_seconds=completed_runtime,
                test_count=len(datasets.test),
            )

        completed_all = set(("train", "evaluate", "report")).issubset(selected_stages)
        run_manifest.update(
            {
                "status": "complete" if completed_all else "stage_complete",
                "completed_at": _utc_now(),
                "runtime_seconds": completed_runtime,
                "training_runtime_seconds": training_runtime,
                "split_digest_sha256": split_digest,
                "report": report_result,
                "completion_assertions": {
                    "cpu_only": True,
                    "seed_count": len(seeds),
                    "all_trained_variants": list(TRAINED_VARIANTS),
                    "all_shuffled_variants": list(SHUFFLED_VARIANTS),
                    "ood_axes": list(REQUIRED_OOD_AXES),
                    "group_leakage_check_passed": True,
                    "thresholds_selected_on_id_validation_only": True,
                    "oracle_excluded_from_fair_comparison": True,
                    "candidate_action_ranking_evaluated": "evaluate" in selected_stages,
                    "id_ood_generalization_gaps_reported": "report" in selected_stages,
                    "failure_visualizations_requested": bool(
                        dict(config.get("reporting", {})).get(
                            "save_failure_visualizations", False
                        )
                    ),
                    "milestone1_outputs_deleted": False,
                },
            }
        )
        _write_json(output_dir / "run_manifest.json", run_manifest)
        return run_manifest
    except Exception as error:
        run_manifest.update(
            {
                "status": "failed",
                "failed_at": _utc_now(),
                "runtime_seconds": time.perf_counter() - started,
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )
        _write_json(output_dir / "run_manifest.json", run_manifest)
        raise


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CPU-only ActMask Milestone 2A")
    parser.add_argument("--config", required=True, help="YAML experiment configuration")
    parser.add_argument(
        "--stage",
        action="append",
        choices=("prepare", "train", "evaluate", "report"),
        help="Run only selected stage(s); repeat the flag to compose stages",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Retrain and overwrite matching Milestone-2 run artifacts",
    )
    parser.add_argument(
        "--output-dir", help="Override experiment.output_dir (Milestone-2 artifacts only)"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = _parse_args(argv)
    config_path = resolve_config_path(args.config)
    config = load_config(config_path)
    if args.output_dir is not None:
        config.setdefault("experiment", {})["output_dir"] = args.output_dir
    result = run_milestone2(
        config,
        config_path=config_path,
        stages=args.stage,
        force=bool(args.force),
    )
    print(
        f"Milestone 2A {result['status']} in {result['runtime_seconds']:.3f}s; "
        f"artifacts: {result['output_dir']}"
    )
    return result


if __name__ == "__main__":
    main()
