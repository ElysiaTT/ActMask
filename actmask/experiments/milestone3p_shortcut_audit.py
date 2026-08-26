"""CPU-only Milestone 3P static-shortcut audit and counterfactual prototype.

The implementation is intentionally small and self-contained: it audits the
existing Milestone 2E candidate data, derives strict-observable static features,
builds a deterministic static-balanced and matched-dynamic benchmark, evaluates
compact baselines, and writes only versioned Milestone 3P artifacts.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from actmask.data.milestone2b_dataset import _clone_tensor_tree, estimate_velocity_from_observation_history, validate_observable_mapping
from actmask.data.milestone2c_dataset import HardCandidateActionDataset
from actmask.data.milestone3p_counterfactual import (
    CANDIDATE_COUNT,
    COUNTERFACTUAL_FAMILIES,
    STATIC_MATCH_TOLERANCE,
    CounterfactualTemporalDataset,
    matched_pair_samples,
)
from actmask.eval.calibration import reliability_metrics
from actmask.eval.statistics import paired_bootstrap_ci
from actmask.experiments.milestone2e import RankingBundle, ranking_metrics_tie_aware
from actmask.models.milestone2b_baselines import CurrentPositionProximity, EstimatedTimeAlignedTrajectoryProximity


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "actmask" / "milestone3p_cpu_shortcut_audit"
M2E_FINAL_ROOT = PROJECT_ROOT / "outputs" / "actmask" / "milestone2e_final_v2"
SCHEMA_VERSION = "milestone3p-cpu-shortcut-audit-v1"
SEEDS = (1301, 2303, 3307)
CONFIRMATION_SEEDS = (1301, 2303, 3307, 4309, 5311)
SPLIT_COUNTS = {"train": 48, "val": 16, "test": 24}
STATIC_FEATURE_NAMES = (
    "start_distance_proxy",
    "endpoint_distance_proxy",
    "path_distance_proxy",
    "path_clearance_proxy",
    "path_length",
    "action_duration",
    "nominal_action_delay",
    "observation_delay",
    "object_centroid_relative_path",
    "scene_centroid_x",
    "scene_centroid_y",
    "scene_centroid_z",
    "scene_spread",
    "visible_fraction",
)
DYNAMIC_FEATURE_NAMES = (
    "predicted_contact_distance_q10",
    "predicted_contact_distance_mean12",
    "predicted_contact_distance_min",
    "estimated_speed_mean12",
    "estimated_acceleration_mean12",
    "time_to_path_midpoint",
)


@dataclass
class CandidateRows:
    samples: list[dict[str, Any]]
    static: np.ndarray
    dynamic: np.ndarray
    scene: np.ndarray
    action: np.ndarray
    template: np.ndarray
    success: np.ndarray
    utility: np.ndarray
    metadata: list[dict[str, Any]]


def _timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().item() if value.numel() == 1 else value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _write_json_new(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite Milestone 3P artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n")


def _write_csv_new(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite Milestone 3P table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({str(key) for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _jsonable(row.get(key)) for key in columns})


def _visible_current(sample: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    observable = sample["observable"]
    validate_observable_mapping(observable)
    history = np.asarray(observable["points_history"].detach().cpu(), dtype=np.float64)
    visibility = np.asarray(observable["visibility_history"].detach().cpu(), dtype=bool)
    current = history[-1]
    mask = visibility[-1]
    if not np.any(mask):
        raise ValueError("static feature extraction needs at least one current visible point")
    return current, mask, np.asarray(observable["action_command"].detach().cpu(), dtype=np.float64)


def _distance_to_segment(points: np.ndarray, action: np.ndarray) -> np.ndarray:
    start, end = action[:3], action[3:6]
    segment = end - start
    denominator = max(float(np.dot(segment, segment)), 1.0e-12)
    fraction = np.clip(np.sum((points - start) * segment[None], axis=1) / denominator, 0.0, 1.0)
    closest = start[None] + fraction[:, None] * segment[None]
    return np.linalg.norm(points - closest, axis=1)


def extract_static_features(sample: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Derive static-only values from the strict observable mapping alone.

    The "target" quantities are geometry-only nearest-cluster proxies.  No
    target mask, future state, candidate success, utility or hidden mapping is
    read by this function.
    """

    current, mask, action = _visible_current(sample)
    points = current[mask]
    start_distance = np.linalg.norm(points - action[:3], axis=1)
    end_distance = np.linalg.norm(points - action[3:6], axis=1)
    path_distance = _distance_to_segment(points, action)
    closest = points[np.argsort(path_distance, kind="stable")[: min(12, len(points))]]
    centroid = closest.mean(axis=0)
    path_center = 0.5 * (action[:3] + action[3:6])
    observable = sample["observable"]
    values = np.asarray(
        [
            np.quantile(start_distance, 0.20),
            np.quantile(end_distance, 0.20),
            np.quantile(path_distance, 0.20),
            np.quantile(path_distance, 0.60),
            np.linalg.norm(action[3:6] - action[:3]),
            action[6],
            float(observable["nominal_action_delay"]),
            float(observable["observation_delay"]),
            np.linalg.norm(centroid - path_center),
            points[:, 0].mean(),
            points[:, 1].mean(),
            points[:, 2].mean(),
            float(np.sqrt(np.mean(np.var(points, axis=0)))),
            float(mask.mean()),
        ],
        dtype=np.float64,
    )
    scene = np.asarray([points[:, 0].mean(), points[:, 1].mean(), points[:, 2].mean(), float(np.sqrt(np.mean(np.var(points, axis=0)))), float(mask.mean())], dtype=np.float64)
    action_features = np.asarray([*action[:6], action[6], action[7]], dtype=np.float64)
    metadata = sample["metadata"]
    template = int(metadata.get("candidate_template_identity", metadata.get("candidate_slot", 0)))
    return values, scene, action_features, template


def extract_temporal_features(sample: Mapping[str, Any]) -> np.ndarray:
    """Time-align observable motion with the candidate trajectory midpoint."""

    observable = sample["observable"]
    validate_observable_mapping(observable)
    history = np.asarray(observable["points_history"].detach().cpu(), dtype=np.float64)
    visibility = np.asarray(observable["visibility_history"].detach().cpu(), dtype=bool)
    timestamps = np.asarray(observable["timestamps"].detach().cpu(), dtype=np.float64)
    current, current_mask, action = _visible_current(sample)
    velocity = np.asarray(observable["estimated_velocity"].detach().cpu(), dtype=np.float64)
    dt = max(float(timestamps[-1] - timestamps[-2]), 1.0e-4)
    raw_acceleration = (history[-1] - 2.0 * history[-2] + history[-3]) / max(dt**2, 1.0e-6)
    raw_acceleration[~np.logical_and(visibility[-1], np.logical_and(visibility[-2], visibility[-3]))] = 0.0
    delay = float(observable["nominal_action_delay"])
    elapsed = delay + float(action[6]) * 0.5
    predicted = current + elapsed * velocity + 0.5 * elapsed**2 * raw_acceleration
    path_center = 0.5 * (action[:3] + action[3:6])
    distance = np.linalg.norm(predicted[current_mask] - path_center[None], axis=1)
    order = np.argsort(distance, kind="stable")[: min(12, len(distance))]
    selected = distance[order]
    speeds = np.linalg.norm(velocity[current_mask], axis=1)[order]
    accelerations = np.linalg.norm(raw_acceleration[current_mask], axis=1)[order]
    return np.asarray(
        [
            np.quantile(selected, 0.10),
            selected.mean(),
            selected.min(),
            speeds.mean(),
            accelerations.mean(),
            elapsed,
        ],
        dtype=np.float64,
    )


def _materialize(dataset: Dataset[dict[str, Any]]) -> CandidateRows:
    samples = [dataset[index] for index in range(len(dataset))]
    static, scene, action, template = [], [], [], []
    dynamic, success, utility, metadata = [], [], [], []
    for sample in samples:
        feature, scene_feature, action_feature, template_id = extract_static_features(sample)
        static.append(feature)
        scene.append(scene_feature)
        action.append(action_feature)
        template.append(template_id)
        dynamic.append(extract_temporal_features(sample))
        success.append(float(sample["targets"]["success"]))
        utility.append(float(sample["targets"]["candidate_utility"]))
        metadata.append(dict(sample["metadata"]))
    return CandidateRows(
        samples=samples,
        static=np.asarray(static, dtype=np.float64),
        dynamic=np.asarray(dynamic, dtype=np.float64),
        scene=np.asarray(scene, dtype=np.float64),
        action=np.asarray(action, dtype=np.float64),
        template=np.asarray(template, dtype=np.int64),
        success=np.asarray(success, dtype=np.float64),
        utility=np.asarray(utility, dtype=np.float64),
        metadata=metadata,
    )


def _template_one_hot(values: np.ndarray) -> np.ndarray:
    ids = np.asarray(values, dtype=np.int64)
    width = max(int(ids.max()) + 1 if ids.size else 1, CANDIDATE_COUNT)
    output = np.zeros((len(ids), width), dtype=np.float64)
    output[np.arange(len(ids)), np.clip(ids, 0, width - 1)] = 1.0
    return output


def _metric_report(rows: CandidateRows, score: np.ndarray, *, score_name: str, probability: np.ndarray | None = None) -> dict[str, Any]:
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    if score.shape != rows.success.shape or not np.isfinite(score).all():
        raise ValueError("score must be finite and aligned with candidate rows")
    report = ranking_metrics_tie_aware(
        RankingBundle(scores=score, success=rows.success, utility=rows.utility, metadata=rows.metadata),
        score_name=score_name,
    )
    if probability is None:
        center, scale = float(np.median(score)), max(float(np.std(score)), 1.0e-6)
        probability = 1.0 / (1.0 + np.exp(-(score - center) / scale))
    report["calibration"] = reliability_metrics(np.clip(np.asarray(probability, dtype=np.float64), 1.0e-5, 1.0 - 1.0e-5), rows.success, bins=10)
    report["evaluation_artifact_schema"] = SCHEMA_VERSION
    report["metric_engine_schema"] = report["ranking_metric_schema"]
    return report


def _summary(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not reports:
        raise ValueError("cannot summarize zero reports")
    result: dict[str, Any] = {"seeds": [int(report.get("seed", 0)) for report in reports]}
    for metric in ("top1_success", "pairwise_ranking_accuracy", "ranking_ap", "ranking_roc_auc", "normalized_regret", "ndcg"):
        values = np.asarray([float(report["metrics"][metric]) for report in reports], dtype=np.float64)
        result[metric] = {"mean": float(values.mean()), "std": float(values.std(ddof=1 if len(values) > 1 else 0)), "values": values.tolist()}
    for metric in ("ece", "brier", "average_precision", "pr_auc"):
        values = np.asarray([float(report["metrics"]["calibration"][metric]) for report in reports], dtype=np.float64)
        result.setdefault("calibration", {})[metric] = {"mean": float(values.mean()), "std": float(values.std(ddof=1 if len(values) > 1 else 0)), "values": values.tolist()}
    return result


class _TinyNetwork(nn.Module):
    def __init__(self, features: int, *, hidden: int = 20, linear: bool = False) -> None:
        super().__init__()
        self.network = nn.Linear(features, 1) if linear else nn.Sequential(nn.Linear(features, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value).reshape(-1)


@dataclass
class _FittedNetwork:
    model: _TinyNetwork
    mean: np.ndarray
    scale: np.ndarray

    def predict(self, features: np.ndarray) -> np.ndarray:
        value = (np.asarray(features, dtype=np.float32) - self.mean) / self.scale
        with torch.no_grad():
            logits = self.model(torch.from_numpy(value)).detach().cpu().numpy()
        return 1.0 / (1.0 + np.exp(-logits))


def _fit_network(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    *,
    seed: int,
    linear: bool = False,
    pair_indices: Sequence[tuple[int, int]] = (),
    mismatch_negative: bool = False,
) -> _FittedNetwork:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    mean = np.asarray(train_x, dtype=np.float64).mean(axis=0).astype(np.float32)
    scale = np.asarray(train_x, dtype=np.float64).std(axis=0).clip(1.0e-5).astype(np.float32)
    train = torch.from_numpy(((np.asarray(train_x, dtype=np.float32) - mean) / scale).astype(np.float32))
    target = torch.from_numpy(np.asarray(train_y, dtype=np.float32))
    validation = torch.from_numpy(((np.asarray(val_x, dtype=np.float32) - mean) / scale).astype(np.float32))
    validation_target = torch.from_numpy(np.asarray(val_y, dtype=np.float32))
    model = _TinyNetwork(train.shape[1], linear=linear).cpu()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.025 if linear else 0.012, weight_decay=2.0e-4)
    positive_count = float(target.sum().item())
    negative_count = float(target.numel()) - positive_count
    positive_weight = torch.tensor(max(negative_count / max(positive_count, 1.0), 1.0), dtype=torch.float32)
    best_state: dict[str, torch.Tensor] | None = None
    best_loss = math.inf
    for epoch in range(140):
        optimizer.zero_grad(set_to_none=True)
        logits = model(train)
        loss = F.binary_cross_entropy_with_logits(logits, target, pos_weight=positive_weight)
        if pair_indices:
            first = torch.as_tensor([pair[0] for pair in pair_indices], dtype=torch.long)
            second = torch.as_tensor([pair[1] for pair in pair_indices], dtype=torch.long)
            difference = logits[first] - logits[second]
            sign = torch.where(target[first] > target[second], 1.0, -1.0)
            loss = loss + 0.25 * F.softplus(-sign * difference).mean()
        if mismatch_negative:
            loss = loss + 0.05 * logits.square().mean()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            current = float(F.binary_cross_entropy_with_logits(model(validation), validation_target).item())
        if current < best_loss:
            best_loss = current
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("tiny network did not fit")
    model.load_state_dict(best_state)
    model.eval()
    return _FittedNetwork(model=model, mean=mean, scale=scale)


@dataclass
class _TreeNode:
    probability: float
    feature: int | None = None
    threshold: float | None = None
    left: "_TreeNode | None" = None
    right: "_TreeNode | None" = None


class _ShallowTree:
    def __init__(self, depth: int = 3) -> None:
        self.depth = int(depth)
        self.root: _TreeNode | None = None

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "_ShallowTree":
        x, y = np.asarray(features, dtype=np.float64), np.asarray(labels, dtype=np.float64)

        def build(indices: np.ndarray, depth: int) -> _TreeNode:
            probability = float(y[indices].mean()) if len(indices) else 0.0
            if depth >= self.depth or len(indices) < 10 or probability in {0.0, 1.0}:
                return _TreeNode(probability=probability)
            baseline = probability * (1.0 - probability)
            best: tuple[float, int, float, np.ndarray, np.ndarray] | None = None
            for feature in range(x.shape[1]):
                values = x[indices, feature]
                for threshold in np.unique(np.quantile(values, (0.2, 0.4, 0.6, 0.8))):
                    left, right = indices[values <= threshold], indices[values > threshold]
                    if len(left) < 4 or len(right) < 4:
                        continue
                    impurity = (len(left) * y[left].mean() * (1.0 - y[left].mean()) + len(right) * y[right].mean() * (1.0 - y[right].mean())) / len(indices)
                    gain = baseline - float(impurity)
                    if best is None or gain > best[0]:
                        best = (gain, feature, float(threshold), left, right)
            if best is None or best[0] <= 1.0e-8:
                return _TreeNode(probability=probability)
            _, feature, threshold, left, right = best
            return _TreeNode(probability=probability, feature=feature, threshold=threshold, left=build(left, depth + 1), right=build(right, depth + 1))

        self.root = build(np.arange(len(y)), 0)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.root is None:
            raise RuntimeError("tree is not fitted")
        result = []
        for row in np.asarray(features, dtype=np.float64):
            node = self.root
            while node.feature is not None:
                node = node.left if row[node.feature] <= float(node.threshold) else node.right
                if node is None:
                    raise AssertionError("tree has an incomplete branch")
            result.append(node.probability)
        return np.asarray(result, dtype=np.float64)


def _baseline_observable_scores(rows: CandidateRows, model: nn.Module) -> np.ndarray:
    """Run a strict-observable analytic baseline without labels or hidden state."""

    model = model.cpu().eval()
    values: list[float] = []
    with torch.no_grad():
        for sample in rows.samples:
            observable = {key: value.unsqueeze(0) for key, value in sample["observable"].items()}
            output = model(observable)
            score = output.get("utility_score")
            if score is None:
                raise ValueError("observable baseline did not emit utility_score")
            values.append(float(score.detach().cpu().reshape(-1)[0]))
    return np.asarray(values, dtype=np.float64)


def _static_raw_scores(rows: CandidateRows) -> dict[str, np.ndarray]:
    return {
        "EndpointDistance": -rows.static[:, STATIC_FEATURE_NAMES.index("endpoint_distance_proxy")],
        "StaticAnalyticRaw": -(
            0.62 * rows.static[:, STATIC_FEATURE_NAMES.index("path_distance_proxy")]
            + 0.28 * rows.static[:, STATIC_FEATURE_NAMES.index("endpoint_distance_proxy")]
            + 0.10 * rows.static[:, STATIC_FEATURE_NAMES.index("path_length")]
        ),
    }


def _prior_audit() -> dict[str, Any]:
    paths = [
        PROJECT_ROOT / "actmask" / "data" / "milestone2b_dataset.py",
        PROJECT_ROOT / "actmask" / "data" / "milestone2c_dataset.py",
        PROJECT_ROOT / "actmask" / "experiments" / "milestone2e_final_v2_temporal_gpu.py",
        M2E_FINAL_ROOT / "temporal_gpu_qualification" / "all_validation_promising.json",
    ]
    digest = {str(path.relative_to(PROJECT_ROOT) if path.is_relative_to(PROJECT_ROOT) else path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths if path.exists() and path.is_file()}
    temporal = json.loads((M2E_FINAL_ROOT / "temporal_gpu_qualification" / "all_validation_promising.json").read_text())
    report = temporal["established_temporal_gpu_shadow"]
    selected = report["phase4_temporal_shortcut"]["per_variant"]["SoftCoordinateInvariantCorrespondence__ScalarInvariantFeatureBackbone"]
    correct = selected["views"]["correct_history"]["summary"]["learned_only"]["top1_success"]["mean"]
    static = selected["views"]["static_geometry_only"]["summary"]["learned_only"]["top1_success"]["mean"]
    source = (PROJECT_ROOT / "actmask" / "data" / "milestone2b_dataset.py").read_text()
    return {
        "artifact_schema": SCHEMA_VERSION,
        "stage": "A0_prior_evidence_and_implementation_audit",
        "prior_result": {"correct_history_learned_top1": correct, "static_geometry_only_learned_top1": static, "static_minus_full": float(static - correct)},
        "static_only_definition": "The 2E temporal diagnostic sets feature_mode=no_history; geometry and current action remain available while history-derived inputs are removed.",
        "correct_history_definition": "The validation-selected verifier receives the strict observable history mapping, never hidden_state.",
        "candidate_success_construction": "Milestone2BDataset builds future point trajectories and execution trajectories, computes contact, then derives target-contact utility and binary success. Those future tensors are targets/hidden_state, not fair-method inputs.",
        "candidate_generation": "HardCandidateActionDataset materializes visible action path endpoints, duration, radius and template-derived candidate variants; candidate IDs are independently permuted but path/duration/template geometry can still correlate with success.",
        "grouped_splits": "Milestone2B assigns base-scene group records before candidate and dynamics expansion; grouped manifests reject identity overlap across train/val/test/OOD partitions.",
        "static_clue_audit": {
            "endpoint_distance": True,
            "current_path_distance": True,
            "trajectory_length": True,
            "action_timing": True,
            "candidate_template": True,
            "source_contains_future_label_as_observable": False,
            "future_field_is_in_observable_whitelist": "exact_future_point_trajectories" in source[source.find("OBSERVABLE_FIELDS"):source.find("HIDDEN_STATE_FIELDS")],
        },
        "input_separation": "validate_observable_mapping rejects every hidden field; this audit's static extractor calls it and reads only current visible points, action command and observable timing.",
        "source_sha256": digest,
    }


def _report_seed(rows: CandidateRows, score: np.ndarray, probability: np.ndarray, *, method: str, seed: int) -> dict[str, Any]:
    return {"method": method, "seed": int(seed), "metrics": _metric_report(rows, score, score_name=method, probability=probability)}


def _deterministic_reports(rows: CandidateRows, score: np.ndarray, *, method: str, seeds: Sequence[int]) -> dict[str, Any]:
    center, spread = float(np.median(score)), max(float(np.std(score)), 1.0e-6)
    probability = 1.0 / (1.0 + np.exp(-(np.asarray(score) - center) / spread))
    reports = [_report_seed(rows, score, probability, method=method, seed=seed) for seed in seeds]
    return {"method": method, "learned": False, "seed_results": reports, "summary": _summary(reports), "deterministic_repeats": True}


def _neural_reports(
    train: CandidateRows,
    validation: CandidateRows,
    test: CandidateRows,
    *,
    method: str,
    train_features: np.ndarray,
    validation_features: np.ndarray,
    test_features: np.ndarray,
    seeds: Sequence[int],
    linear: bool = False,
    pair_indices: Sequence[tuple[int, int]] = (),
    mismatch_negative: bool = False,
) -> dict[str, Any]:
    reports: list[dict[str, Any]] = []
    for seed in seeds:
        fitted = _fit_network(
            train_features,
            train.success,
            validation_features,
            validation.success,
            seed=int(seed),
            linear=linear,
            pair_indices=pair_indices,
            mismatch_negative=mismatch_negative,
        )
        probability = fitted.predict(test_features)
        reports.append(_report_seed(test, probability, probability, method=method, seed=int(seed)))
    return {"method": method, "learned": True, "seed_results": reports, "summary": _summary(reports)}


def _tree_reports(train: CandidateRows, test: CandidateRows, *, method: str, features_train: np.ndarray, features_test: np.ndarray, seeds: Sequence[int]) -> dict[str, Any]:
    tree = _ShallowTree(depth=3).fit(features_train, train.success)
    probability = tree.predict(features_test)
    reports = [_report_seed(test, probability, probability, method=method, seed=int(seed)) for seed in seeds]
    return {"method": method, "learned": False, "seed_results": reports, "summary": _summary(reports), "deterministic_repeats": True}


def _static_method_reports(train: CandidateRows, validation: CandidateRows, test: CandidateRows, *, seeds: Sequence[int]) -> dict[str, Any]:
    """Evaluate every prescribed static method without temporal feature access."""

    reports: dict[str, Any] = {}
    reports["CurrentPositionProximity"] = _deterministic_reports(
        test,
        _baseline_observable_scores(test, CurrentPositionProximity()),
        method="CurrentPositionProximity",
        seeds=seeds,
    )
    for name, score in _static_raw_scores(test).items():
        reports[name.removesuffix("Raw")] = _deterministic_reports(test, score, method=name.removesuffix("Raw"), seeds=seeds)
    reports["StaticLogisticRegression"] = _neural_reports(
        train, validation, test,
        method="StaticLogisticRegression", train_features=train.static, validation_features=validation.static,
        test_features=test.static, seeds=seeds, linear=True,
    )
    reports["StaticShallowTree"] = _tree_reports(
        train, test, method="StaticShallowTree", features_train=train.static, features_test=test.static, seeds=seeds,
    )
    reports["StaticTwoLayerMLP"] = _neural_reports(
        train, validation, test,
        method="StaticTwoLayerMLP", train_features=train.static, validation_features=validation.static,
        test_features=test.static, seeds=seeds,
    )
    reports["CandidateTemplateOnly"] = _neural_reports(
        train, validation, test,
        method="CandidateTemplateOnly", train_features=_template_one_hot(train.template), validation_features=_template_one_hot(validation.template),
        test_features=_template_one_hot(test.template), seeds=seeds, linear=True,
    )
    reports["SceneOnly"] = _neural_reports(
        train, validation, test,
        method="SceneOnly", train_features=train.scene, validation_features=validation.scene,
        test_features=test.scene, seeds=seeds, linear=True,
    )
    reports["ActionOnly"] = _neural_reports(
        train, validation, test,
        method="ActionOnly", train_features=train.action, validation_features=validation.action,
        test_features=test.action, seeds=seeds, linear=True,
    )
    return reports


def _time_aligned_scores(rows: CandidateRows) -> np.ndarray:
    return _baseline_observable_scores(rows, EstimatedTimeAlignedTrajectoryProximity())


def _temporal_method_reports(train: CandidateRows, validation: CandidateRows, test: CandidateRows, *, seeds: Sequence[int]) -> dict[str, Any]:
    reports: dict[str, Any] = {}
    reports["TimeAlignedRelativeFeatures"] = _deterministic_reports(
        test, _time_aligned_scores(test), method="TimeAlignedRelativeFeatures", seeds=seeds
    )
    reports["TemporalTwoLayerMLP"] = _neural_reports(
        train, validation, test,
        method="TemporalTwoLayerMLP", train_features=np.concatenate((train.static, train.dynamic), axis=1),
        validation_features=np.concatenate((validation.static, validation.dynamic), axis=1),
        test_features=np.concatenate((test.static, test.dynamic), axis=1), seeds=seeds,
    )
    return reports


def _load_previous_dynamic_scores(rows: CandidateRows, *, feature_mode: str = "full") -> dict[str, Any]:
    """Evaluate the frozen 2E dynamic verifier without any retraining.

    Failure to load is treated as a reproducible error by the caller rather
    than silently substituting another method.
    """

    from actmask.experiments.milestone2e import _load_config
    from actmask.experiments.milestone2e_final_v2 import LEGACY_ROOT, _load_primary_models, _selected_primary
    from actmask.experiments.milestone2e_full import _forward

    config = _load_config()
    selection = _selected_primary(config, legacy_root=LEGACY_ROOT)
    models, _ = _load_primary_models(config, legacy_root=LEGACY_ROOT, selection=selection)
    seed_results: list[dict[str, Any]] = []
    for seed, model in sorted(models.items()):
        values: list[float] = []
        model = model.cpu().eval()
        with torch.no_grad():
            for sample in rows.samples:
                observable = {key: value.unsqueeze(0) for key, value in sample["observable"].items()}
                if feature_mode == "full":
                    output = _forward(model, {"observable": observable})
                else:
                    canonical = model.canonicalize(observable)
                    output = model.core(canonical, feature_mode=feature_mode)
                values.append(float(output["utility_logits"].detach().cpu().reshape(-1)[0]))
        score = np.asarray(values, dtype=np.float64)
        center, spread = float(np.median(score)), max(float(np.std(score)), 1.0e-6)
        probability = 1.0 / (1.0 + np.exp(-(score - center) / spread))
        seed_results.append(_report_seed(rows, score, probability, method=f"PreviousDynamicVerifier_{feature_mode}", seed=int(seed)))
    return {
        "method": "PreviousDynamicVerifier" if feature_mode == "full" else f"PreviousDynamicVerifier_{feature_mode}",
        "learned": True,
        "frozen_checkpoint_reuse": True,
        "seed_results": seed_results,
        "summary": _summary(seed_results),
    }


def _corrupt_sample(sample: Mapping[str, Any], mode: str, *, donor: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Observable-only temporal corruptions; labels and actions stay fixed."""

    if mode not in {"correct", "removed", "reversed", "frame_permutation", "mismatched", "zeroed_motion", "shuffled_motion"}:
        raise ValueError(f"unknown temporal corruption {mode!r}")
    result = _clone_tensor_tree(sample)
    if mode == "correct":
        return result
    observable = result["observable"]
    history = observable["points_history"]
    visibility = observable["visibility_history"]
    if mode == "mismatched":
        if donor is None:
            raise ValueError("mismatched corruption needs a donor sample")
        for key in ("points_history", "visibility_history", "timestamps", "estimated_velocity", "velocity_confidence"):
            observable[key] = donor["observable"][key].clone()
    elif mode == "removed":
        history[:] = history[-1:].expand_as(history)
        observable["estimated_velocity"] = torch.zeros_like(observable["estimated_velocity"])
        observable["velocity_confidence"] = torch.zeros_like(observable["velocity_confidence"])
    elif mode == "reversed":
        observable["points_history"] = torch.flip(history, dims=(0,))
        observable["visibility_history"] = torch.flip(visibility, dims=(0,))
    elif mode == "frame_permutation":
        # Permute each candidate set independently but deterministically.  A
        # set-shared order avoids injecting an action-specific cue, while the
        # group-specific seed prevents a single global transformation from
        # accidentally preserving a model's temporal convention.
        group_id = int(result["metadata"].get("group_id", 0))
        order_value = np.random.default_rng(group_id ^ 0x3F4A2C19).permutation(history.shape[0])
        if np.array_equal(order_value, np.arange(history.shape[0])):
            order_value = np.roll(order_value, 1)
        order = torch.as_tensor(order_value, dtype=torch.long)
        observable["points_history"] = history[order]
        observable["visibility_history"] = visibility[order]
    elif mode == "zeroed_motion":
        observable["estimated_velocity"] = torch.zeros_like(observable["estimated_velocity"])
        observable["velocity_confidence"] = torch.zeros_like(observable["velocity_confidence"])
    elif mode == "shuffled_motion":
        order = torch.arange(observable["estimated_velocity"].shape[0] - 1, -1, -1)
        observable["estimated_velocity"] = observable["estimated_velocity"][order]
        observable["velocity_confidence"] = observable["velocity_confidence"][order]
    if mode in {"reversed", "frame_permutation"}:
        array = observable["points_history"].detach().cpu().numpy()
        mask = observable["visibility_history"].detach().cpu().numpy()
        stamps = observable["timestamps"].detach().cpu().numpy()
        velocity, confidence = estimate_velocity_from_observation_history(array, mask, stamps)
        observable["estimated_velocity"] = torch.from_numpy(velocity)
        observable["velocity_confidence"] = torch.from_numpy(confidence)
    validate_observable_mapping(observable)
    return result


def _corrupt_rows(rows: CandidateRows, mode: str) -> CandidateRows:
    samples: list[dict[str, Any]] = []
    for index, sample in enumerate(rows.samples):
        donor = rows.samples[(index + CANDIDATE_COUNT) % len(rows.samples)] if mode == "mismatched" else None
        samples.append(_corrupt_sample(sample, mode, donor=donor))
    static, scene, action, template, dynamic = [], [], [], [], []
    for sample in samples:
        value, scene_value, action_value, template_value = extract_static_features(sample)
        static.append(value); scene.append(scene_value); action.append(action_value); template.append(template_value); dynamic.append(extract_temporal_features(sample))
    return CandidateRows(samples=samples, static=np.asarray(static), dynamic=np.asarray(dynamic), scene=np.asarray(scene), action=np.asarray(action), template=np.asarray(template), success=rows.success.copy(), utility=rows.utility.copy(), metadata=[dict(value) for value in rows.metadata])


def _group_metric_values(report: Mapping[str, Any], metric: str) -> dict[int, float]:
    return {int(row["group_id"]): float(row[metric]) for row in report["per_group"]}


def _paired_dynamic_advantage(static_report: Mapping[str, Any], dynamic_report: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for offset, metric in enumerate(("top1_success", "normalized_regret")):
        static = _group_metric_values(static_report, metric)
        dynamic = _group_metric_values(dynamic_report, metric)
        ids = sorted(set(static).intersection(dynamic))
        if metric == "top1_success":
            result[metric] = paired_bootstrap_ci([dynamic[group] for group in ids], [static[group] for group in ids], seed=20300501 + offset)
        else:
            # Lower regret is better; reverse the subtraction so positive is a dynamic advantage.
            result[metric] = paired_bootstrap_ci([static[group] for group in ids], [dynamic[group] for group in ids], seed=20300501 + offset)
    return result


def _counterfactual_pair_accuracy(rows: CandidateRows, score: np.ndarray) -> dict[str, Any]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows.metadata):
        grouped[str(row.get("static_match_key", ""))].append(index)
    values: list[float] = []
    label_flips = 0
    for indices in grouped.values():
        if len(indices) != 2:
            continue
        first, second = indices
        if rows.success[first] == rows.success[second]:
            continue
        label_flips += 1
        better = first if rows.success[first] > rows.success[second] else second
        worse = second if better == first else first
        values.append(1.0 if score[better] > score[worse] else 0.0 if score[better] < score[worse] else 0.5)
    return {"pair_accuracy": float(np.mean(values)) if values else 0.5, "label_flip_pairs": label_flips, "eligible_pairs": len(values), "chance_reference": 0.5}


def _feature_identity_audit(dataset: CounterfactualTemporalDataset) -> dict[str, Any]:
    pairs = matched_pair_samples(dataset)
    maximum = {"current_points": 0.0, "action": 0.0, "duration": 0.0, "static_features": 0.0}
    flips = 0
    static_rows: list[np.ndarray] = []
    labels: list[float] = []
    for first, second in pairs:
        left, _, _, _ = extract_static_features(first)
        right, _, _, _ = extract_static_features(second)
        maximum["current_points"] = max(maximum["current_points"], float(torch.max(torch.abs(first["observable"]["points_history"][-1] - second["observable"]["points_history"][-1])).item()))
        maximum["action"] = max(maximum["action"], float(torch.max(torch.abs(first["observable"]["action_command"] - second["observable"]["action_command"])).item()))
        maximum["duration"] = max(maximum["duration"], abs(float(first["observable"]["action_command"][6]) - float(second["observable"]["action_command"][6])))
        maximum["static_features"] = max(maximum["static_features"], float(np.max(np.abs(left - right))))
        if float(first["targets"]["success"]) != float(second["targets"]["success"]):
            flips += 1
        static_rows.extend((left, right)); labels.extend((float(first["targets"]["success"]), float(second["targets"]["success"])))
    # Matching makes a static vector ambiguous: for every exact vector with a
    # label flip, the Bayes-optimal static prediction is chance.
    by_feature: dict[bytes, list[float]] = defaultdict(list)
    for feature, label in zip(static_rows, labels, strict=True):
        by_feature[np.asarray(feature, dtype=np.float64).tobytes()].append(label)
    ambiguous = [value for value in by_feature.values() if len(value) == 2 and min(value) != max(value)]
    bayes = [max(np.mean(value), 1.0 - np.mean(value)) for value in ambiguous]
    return {
        "pairs": len(pairs),
        "label_flip_pairs": flips,
        "maximum_difference": maximum,
        "tolerance": STATIC_MATCH_TOLERANCE,
        "static_leakage_bayes_accuracy_on_label_flip_pairs": float(np.mean(bayes)) if bayes else 1.0,
        "passed": bool(flips >= 3 and all(value <= STATIC_MATCH_TOLERANCE for value in maximum.values()) and bool(bayes) and float(np.mean(bayes)) <= 0.500001),
    }


def _subset_rows(rows: CandidateRows, predicate: Callable[[Mapping[str, Any]], bool]) -> CandidateRows:
    indices = [index for index, row in enumerate(rows.metadata) if predicate(row)]
    if not indices:
        raise ValueError("row subset is empty")
    return CandidateRows(
        samples=[rows.samples[index] for index in indices],
        static=rows.static[indices], dynamic=rows.dynamic[indices], scene=rows.scene[indices], action=rows.action[indices],
        template=rows.template[indices], success=rows.success[indices], utility=rows.utility[indices], metadata=[rows.metadata[index] for index in indices],
    )


def _mutual_information(feature: np.ndarray, label: np.ndarray, *, bins: int = 5) -> float:
    value = np.asarray(feature, dtype=np.float64)
    target = np.asarray(label, dtype=np.int64)
    edges = np.unique(np.quantile(value, np.linspace(0.0, 1.0, bins + 1)))
    if len(edges) < 3:
        return 0.0
    ids = np.clip(np.digitize(value, edges[1:-1], right=False), 0, len(edges) - 2)
    joint = np.zeros((len(edges) - 1, 2), dtype=np.float64)
    for left, right in zip(ids, target, strict=True):
        joint[int(left), int(right)] += 1.0
    joint /= max(joint.sum(), 1.0)
    px, py = joint.sum(axis=1, keepdims=True), joint.sum(axis=0, keepdims=True)
    valid = joint > 0.0
    return float(np.sum(joint[valid] * np.log2(joint[valid] / (px @ py)[valid])))


def _correlation_analysis(train: CandidateRows, test: CandidateRows) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    analysis: dict[str, Any] = {
        "feature_bins": {},
        "mutual_information_bits": {},
        "individual_feature_test_auc": {},
        "pearson_correlation_with_success": {},
    }
    for index, name in enumerate(STATIC_FEATURE_NAMES):
        value = train.static[:, index]
        edges = np.unique(np.quantile(value, np.linspace(0.0, 1.0, 6)))
        if len(edges) < 3:
            bins = [{"bin": 0, "count": int(len(value)), "success_rate": float(train.success.mean()), "lower": float(value.min()), "upper": float(value.max())}]
        else:
            ids = np.digitize(value, edges[1:-1], right=False)
            bins = []
            for bin_index in range(len(edges) - 1):
                selected = ids == bin_index
                bins.append({"bin": bin_index, "count": int(selected.sum()), "success_rate": float(train.success[selected].mean()) if selected.any() else 0.0, "lower": float(edges[bin_index]), "upper": float(edges[bin_index + 1])})
        analysis["feature_bins"][name] = bins
        analysis["mutual_information_bits"][name] = _mutual_information(value, train.success)
        analysis["pearson_correlation_with_success"][name] = (
            float(np.corrcoef(value, train.success)[0, 1]) if float(np.std(value)) > 1.0e-12 else 0.0
        )
        fitted = _fit_network(train.static[:, [index]], train.success, train.static[:, [index]], train.success, seed=111 + index, linear=True)
        probability = fitted.predict(test.static[:, [index]])
        analysis["individual_feature_test_auc"][name] = float(_metric_report(test, probability, score_name=f"individual_{name}", probability=probability)["ranking_roc_auc"])
        for row in bins:
            rows.append({"feature": name, **row})
    template_values: dict[str, list[float]] = defaultdict(list)
    for template, label in zip(train.template, train.success, strict=True):
        template_values[str(int(template))].append(float(label))
    analysis["success_rate_by_candidate_template"] = {name: {"count": len(values), "success_rate": float(np.mean(values))} for name, values in sorted(template_values.items())}
    combined = _fit_network(train.static, train.success, train.static, train.success, seed=2718, linear=True)
    combined_probability = combined.predict(test.static)
    analysis["all_static_features_combination_test"] = _metric_report(
        test, combined_probability, score_name="all_static_features_combination", probability=combined_probability
    )
    return analysis, rows


def _candidate_table(rows: CandidateRows, *, split: str, regime: str) -> list[dict[str, Any]]:
    table: list[dict[str, Any]] = []
    for index, metadata in enumerate(rows.metadata):
        row = {
            "split": split,
            "regime": regime,
            "group_id": int(metadata["group_id"]),
            "candidate_set_id": int(metadata["candidate_set_id"]),
            "candidate_slot": int(metadata.get("candidate_slot", -1)),
            "success": float(rows.success[index]),
            "utility": float(rows.utility[index]),
            "template_identity_explicit_diagnostic": int(rows.template[index]),
        }
        row.update({name: float(rows.static[index, feature]) for feature, name in enumerate(STATIC_FEATURE_NAMES)})
        table.append(row)
    return table


def _pair_indices(rows: CandidateRows) -> list[tuple[int, int]]:
    by_key: dict[str, list[int]] = defaultdict(list)
    for index, metadata in enumerate(rows.metadata):
        by_key[str(metadata.get("static_match_key", ""))].append(index)
    return [tuple(indices) for indices in by_key.values() if len(indices) == 2 and rows.success[indices[0]] != rows.success[indices[1]]]


def _anti_shortcut_ablation(
    matched_train: CandidateRows,
    matched_val: CandidateRows,
    matched_test: CandidateRows,
    balanced_train: CandidateRows,
    balanced_val: CandidateRows,
    *,
    seeds: Sequence[int],
) -> dict[str, Any]:
    """Small interventions, all on the same lightweight temporal MLP."""

    feature = lambda value: np.concatenate((value.static, value.dynamic), axis=1)
    interventions = {
        "ordinary_random_sampling": {"train": matched_train, "val": matched_val, "pair": (), "mismatch": False},
        "counterfactual_pair_sampling": {"train": matched_train, "val": matched_val, "pair": _pair_indices(matched_train), "mismatch": False},
        "static_feature_balanced_batching": {"train": balanced_train, "val": balanced_val, "pair": (), "mismatch": False},
        "pairwise_counterfactual_loss": {"train": matched_train, "val": matched_val, "pair": _pair_indices(matched_train), "mismatch": True},
        "history_mismatch_negatives": {"train": matched_train, "val": matched_val, "pair": (), "mismatch": True},
    }
    result: dict[str, Any] = {}
    for name, setup in interventions.items():
        seed_rows: list[dict[str, Any]] = []
        pair_accuracy: list[float] = []
        for seed in seeds:
            fitted = _fit_network(feature(setup["train"]), setup["train"].success, feature(setup["val"]), setup["val"].success, seed=int(seed), pair_indices=setup["pair"], mismatch_negative=bool(setup["mismatch"]))
            probability = fitted.predict(feature(matched_test))
            metric = _metric_report(matched_test, probability, score_name=name, probability=probability)
            seed_rows.append({"seed": int(seed), "metrics": metric})
            pair_accuracy.append(_counterfactual_pair_accuracy(matched_test, probability)["pair_accuracy"])
        result[name] = {
            "seed_results": seed_rows,
            "summary": _summary(seed_rows),
            "counterfactual_pair_accuracy": {"mean": float(np.mean(pair_accuracy)), "std": float(np.std(pair_accuracy, ddof=1 if len(pair_accuracy) > 1 else 0)), "values": pair_accuracy},
        }
    return result


def _permutation_feature_importance(
    train: CandidateRows,
    validation: CandidateRows,
    test: CandidateRows,
    *,
    seeds: Sequence[int],
) -> dict[str, Any]:
    """Grouped-ranking effect of permuting one static cue at a time.

    The static MLP is fitted only on the train/validation partitions.  Test
    values are then permuted globally with a fixed seed; the permutation is a
    diagnostic, never a model-selection input.
    """

    values: dict[str, list[dict[str, float]]] = {name: [] for name in STATIC_FEATURE_NAMES}
    baseline_records: list[dict[str, Any]] = []
    for seed in seeds:
        fitted = _fit_network(train.static, train.success, validation.static, validation.success, seed=int(seed))
        baseline_probability = fitted.predict(test.static)
        baseline = _metric_report(test, baseline_probability, score_name="StaticTwoLayerMLP_permutation_base", probability=baseline_probability)
        baseline_records.append({"seed": int(seed), "metrics": baseline})
        for feature_index, name in enumerate(STATIC_FEATURE_NAMES):
            permuted = test.static.copy()
            order = np.random.default_rng(int(seed) * 1009 + feature_index * 37).permutation(len(permuted))
            permuted[:, feature_index] = permuted[order, feature_index]
            probability = fitted.predict(permuted)
            current = _metric_report(test, probability, score_name=f"StaticTwoLayerMLP_permute_{name}", probability=probability)
            values[name].append(
                {
                    "seed": float(seed),
                    "top1_decrease": float(baseline["top1_success"] - current["top1_success"]),
                    "pairwise_decrease": float(baseline["pairwise_ranking_accuracy"] - current["pairwise_ranking_accuracy"]),
                    "ranking_ap_decrease": float(baseline["ranking_ap"] - current["ranking_ap"]),
                    "normalized_regret_increase": float(current["normalized_regret"] - baseline["normalized_regret"]),
                }
            )
    features: dict[str, Any] = {}
    for name, records in values.items():
        features[name] = {
            "seed_records": records,
            "mean_top1_decrease": float(np.mean([row["top1_decrease"] for row in records])),
            "mean_pairwise_decrease": float(np.mean([row["pairwise_decrease"] for row in records])),
            "mean_ranking_ap_decrease": float(np.mean([row["ranking_ap_decrease"] for row in records])),
            "mean_normalized_regret_increase": float(np.mean([row["normalized_regret_increase"] for row in records])),
        }
    return {
        "artifact_schema": SCHEMA_VERSION,
        "method": "StaticTwoLayerMLP",
        "regime": "original_distribution",
        "test_permutation_protocol": "one static feature is independently permuted across held-out candidate rows; models are not refit and no result is used for selection",
        "baseline": _summary(baseline_records),
        "features": features,
    }


def _paired_temporal_causal_drop(correct: Mapping[str, Any], corrupted: Mapping[str, Any]) -> dict[str, Any]:
    """Group-paired causal drop: correct minus corrupted Top-1 and vice versa regret."""

    correct_top1 = _group_metric_values(correct, "top1_success")
    corrupted_top1 = _group_metric_values(corrupted, "top1_success")
    correct_regret = _group_metric_values(correct, "normalized_regret")
    corrupted_regret = _group_metric_values(corrupted, "normalized_regret")
    ids = sorted(set(correct_top1).intersection(corrupted_top1))
    return {
        "top1_drop": paired_bootstrap_ci(
            [correct_top1[group] for group in ids], [corrupted_top1[group] for group in ids], seed=20300579
        ),
        "normalized_regret_increase": paired_bootstrap_ci(
            [corrupted_regret[group] for group in ids], [correct_regret[group] for group in ids], seed=20300580
        ),
    }


def _frozen_dynamic_verifier_corruption_report(rows: CandidateRows) -> dict[str, Any]:
    """Apply every specified history corruption to the frozen 2E verifier."""

    modes = ("correct", "removed", "reversed", "frame_permutation", "mismatched", "zeroed_motion", "shuffled_motion")
    reports: dict[str, Any] = {}
    for mode in modes:
        reports[mode] = _load_previous_dynamic_scores(_corrupt_rows(rows, mode), feature_mode="full")
    correct_by_seed = {int(item["seed"]): item["metrics"] for item in reports["correct"]["seed_results"]}
    drops: dict[str, Any] = {}
    for mode in modes[1:]:
        corrupted_by_seed = {int(item["seed"]): item["metrics"] for item in reports[mode]["seed_results"]}
        per_seed = {
            str(seed): _paired_temporal_causal_drop(correct_by_seed[seed], corrupted_by_seed[seed])
            for seed in sorted(set(correct_by_seed).intersection(corrupted_by_seed))
        }
        drops[mode] = {
            "per_seed_group_paired_bootstrap": per_seed,
            "mean_top1_drop_over_seeds": float(
                np.mean([float(value["top1_drop"]["delta"]) for value in per_seed.values()])
            ),
            "mean_normalized_regret_increase_over_seeds": float(
                np.mean([float(value["normalized_regret_increase"]["delta"]) for value in per_seed.values()])
            ),
        }
    return {
        "artifact_schema": SCHEMA_VERSION,
        "method": "PreviousDynamicVerifier",
        "regime": "matched_counterfactual",
        "views": reports,
        "temporal_causal_drop": drops,
        "note": "Each frame-permutation view uses a deterministic, group-specific frame permutation; labels and actions are unchanged in every corruption.",
    }


def _history_mismatch_training_arrays(rows: CandidateRows) -> tuple[np.ndarray, np.ndarray]:
    """Create zero-labelled static/history mismatches for the small ablation."""

    features = np.concatenate((rows.static, rows.dynamic), axis=1)
    donor = np.roll(rows.dynamic, CANDIDATE_COUNT, axis=0)
    mismatched = np.concatenate((rows.static, donor), axis=1)
    return np.concatenate((features, mismatched), axis=0), np.concatenate((rows.success, np.zeros(len(rows.success), dtype=np.float64)), axis=0)


def _anti_shortcut_training_extension(
    matched_train: CandidateRows,
    matched_val: CandidateRows,
    matched_test: CandidateRows,
    balanced_train: CandidateRows,
    balanced_val: CandidateRows,
    original_test: CandidateRows,
    *,
    seeds: Sequence[int],
) -> dict[str, Any]:
    """Evaluate anti-shortcut interventions on matched, original and corrupt views."""

    feature = lambda rows: np.concatenate((rows.static, rows.dynamic), axis=1)
    ordinary_train_x, ordinary_train_y = feature(matched_train), matched_train.success
    mismatch_train_x, mismatch_train_y = _history_mismatch_training_arrays(matched_train)
    setups = {
        "ordinary_random_sampling": (ordinary_train_x, ordinary_train_y, matched_val, (), False),
        "counterfactual_pair_sampling": (ordinary_train_x, ordinary_train_y, matched_val, _pair_indices(matched_train), False),
        "static_feature_balanced_batching": (feature(balanced_train), balanced_train.success, balanced_val, (), False),
        "pairwise_counterfactual_loss": (ordinary_train_x, ordinary_train_y, matched_val, _pair_indices(matched_train), False),
        "history_mismatch_negatives": (mismatch_train_x, mismatch_train_y, matched_val, (), False),
    }
    original_removed = _corrupt_rows(original_test, "removed")
    matched_removed = _corrupt_rows(matched_test, "removed")
    result: dict[str, Any] = {}
    for name, (train_x, train_y, validation_rows, pairs, mismatch_flag) in setups.items():
        validation_x = feature(validation_rows)
        validation_y = validation_rows.success
        records: list[dict[str, Any]] = []
        for seed in seeds:
            fitted = _fit_network(
                train_x, train_y, validation_x, validation_y, seed=int(seed), pair_indices=pairs, mismatch_negative=mismatch_flag
            )
            matched_probability = fitted.predict(feature(matched_test))
            original_probability = fitted.predict(feature(original_test))
            original_removed_probability = fitted.predict(feature(original_removed))
            matched_removed_probability = fitted.predict(feature(matched_removed))
            records.append(
                {
                    "seed": int(seed),
                    "matched_metrics": _metric_report(matched_test, matched_probability, score_name=f"{name}_matched", probability=matched_probability),
                    "original_metrics": _metric_report(original_test, original_probability, score_name=f"{name}_original", probability=original_probability),
                    "original_removed_metrics": _metric_report(original_removed, original_removed_probability, score_name=f"{name}_original_removed", probability=original_removed_probability),
                    "matched_removed_metrics": _metric_report(matched_removed, matched_removed_probability, score_name=f"{name}_matched_removed", probability=matched_removed_probability),
                    "counterfactual_pair_accuracy": _counterfactual_pair_accuracy(matched_test, matched_probability),
                }
            )
        matched_summary = _summary([{"seed": row["seed"], "metrics": row["matched_metrics"]} for row in records])
        original_summary = _summary([{"seed": row["seed"], "metrics": row["original_metrics"]} for row in records])
        original_removed_summary = _summary([{"seed": row["seed"], "metrics": row["original_removed_metrics"]} for row in records])
        matched_removed_summary = _summary([{"seed": row["seed"], "metrics": row["matched_removed_metrics"]} for row in records])
        result[name] = {
            "seed_records": records,
            "matched_summary": matched_summary,
            "original_summary": original_summary,
            "original_removed_history_summary": original_removed_summary,
            "matched_removed_history_summary": matched_removed_summary,
            "mean_counterfactual_pair_accuracy": float(np.mean([row["counterfactual_pair_accuracy"]["pair_accuracy"] for row in records])),
            "original_history_removed_top1_drop": float(
                original_summary["top1_success"]["mean"] - original_removed_summary["top1_success"]["mean"]
            ),
            "matched_history_removed_top1_drop": float(
                matched_summary["top1_success"]["mean"] - matched_removed_summary["top1_success"]["mean"]
            ),
        }
    ordinary = result["ordinary_random_sampling"]
    for name, payload in result.items():
        payload["versus_ordinary_random_sampling"] = {
            "matched_top1_delta": float(payload["matched_summary"]["top1_success"]["mean"] - ordinary["matched_summary"]["top1_success"]["mean"]),
            "original_top1_delta": float(payload["original_summary"]["top1_success"]["mean"] - ordinary["original_summary"]["top1_success"]["mean"]),
            "matched_pair_accuracy_delta": float(payload["mean_counterfactual_pair_accuracy"] - ordinary["mean_counterfactual_pair_accuracy"]),
            "original_history_removed_drop_delta": float(payload["original_history_removed_top1_drop"] - ordinary["original_history_removed_top1_drop"]),
        }
    return {
        "artifact_schema": SCHEMA_VERSION,
        "protocol": "all interventions use the same compact static-plus-temporal MLP and are evaluated on held-out original and matched worlds; no result selects a method",
        "interventions": result,
    }


def _temporal_corruption_report(rows: CandidateRows) -> dict[str, Any]:
    modes = ("correct", "reversed", "frame_permutation", "mismatched", "zeroed_motion", "shuffled_motion", "removed")
    reports: dict[str, Any] = {}
    correct_score: np.ndarray | None = None
    for mode in modes:
        current = _corrupt_rows(rows, mode)
        score = _time_aligned_scores(current)
        report = _metric_report(current, score, score_name=f"TimeAlignedRelativeFeatures_{mode}")
        reports[mode] = {"metrics": report, "counterfactual_pair_accuracy": _counterfactual_pair_accuracy(current, score)}
        if mode == "correct":
            correct_score = score
            reports[mode]["score"] = score
    if correct_score is None:
        raise AssertionError("correct temporal corruption view is missing")
    correct = reports["correct"]["metrics"]
    drops: dict[str, Any] = {}
    for mode in modes[1:]:
        current = reports[mode]["metrics"]
        drops[mode] = {
            "top1_drop": float(correct["top1_success"] - current["top1_success"]),
            "normalized_regret_increase": float(current["normalized_regret"] - correct["normalized_regret"]),
        }
    reports["temporal_causal_drop"] = drops
    reports["correct"].pop("score", None)
    return reports


def _plot_figures(output_root: Path, *, original: Mapping[str, Any], balanced: Mapping[str, Any], matched: Mapping[str, Any], correlation: Mapping[str, Any]) -> None:
    figure_root = output_root / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    # This function is called only on a brand-new output directory.
    names = ["CurrentPositionProximity", "StaticTwoLayerMLP", "TimeAlignedRelativeFeatures", "TemporalTwoLayerMLP"]
    regimes = [("original", original), ("static_balanced", balanced), ("matched_counterfactual", matched)]
    values = [[float(payload["methods"][name]["summary"]["top1_success"]["mean"]) for name in names] for _, payload in regimes]
    fig, axis = plt.subplots(figsize=(8, 4.5))
    positions = np.arange(len(regimes))
    width = 0.19
    for index, name in enumerate(names):
        axis.bar(positions + (index - 1.5) * width, [value[index] for value in values], width=width, label=name)
    axis.set_xticks(positions, [name for name, _ in regimes])
    axis.set_ylim(0.0, 1.05)
    axis.set_ylabel("Top-1 successful-action accuracy")
    axis.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figure_root / "regime_top1.png", dpi=160)
    plt.close(fig)

    items = sorted(correlation["mutual_information_bits"].items(), key=lambda item: item[1], reverse=True)
    fig, axis = plt.subplots(figsize=(9, 4.5))
    axis.bar(range(len(items)), [value for _, value in items])
    axis.set_xticks(range(len(items)), [name.replace("_", "\n") for name, _ in items], rotation=45, ha="right", fontsize=7)
    axis.set_ylabel("Mutual information with success (bits)")
    fig.tight_layout()
    fig.savefig(figure_root / "static_feature_mutual_information.png", dpi=160)
    plt.close(fig)


def _run_full_tests(output_root: Path) -> dict[str, Any]:
    test_root = output_root / "tests"
    test_root.mkdir(parents=True, exist_ok=True)
    log = test_root / "full_pytest.log"
    if log.exists():
        raise FileExistsError(log)
    started = time.perf_counter()
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"], cwd=PROJECT_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
        env={**dict(__import__("os").environ), "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
    )
    log.write_text(completed.stdout)
    return {"command": f"{sys.executable} -m pytest -q", "exit_code": int(completed.returncode), "elapsed_seconds": time.perf_counter() - started, "log": str(log), "passed": bool(completed.returncode == 0), "summary": completed.stdout.splitlines()[-1] if completed.stdout.splitlines() else ""}


def _environment_record() -> dict[str, Any]:
    """Record the deliberately CPU-only Stage-A execution environment."""

    return {
        "artifact_schema": SCHEMA_VERSION,
        "recorded_at": _timestamp(),
        "stage": "A_cpu_static_shortcut_audit",
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "torch_build_cuda": torch.version.cuda,
        "requested_device": "cpu",
        "torch_threads": int(torch.get_num_threads()),
        "cuda_or_simulator_used": False,
        "prohibited_stage_a_resources": ["CUDA execution", "simulator", "RGB-D assets", "pretrained weights", "external data or downloads"],
        "project_root": str(PROJECT_ROOT),
    }


def _assert_rows_cpu_only(rows: CandidateRows) -> None:
    """Fail fast if a Stage-A input was accidentally materialized on a GPU."""

    for sample in rows.samples:
        for value in sample["observable"].values():
            if isinstance(value, torch.Tensor) and value.device.type != "cpu":
                raise AssertionError("Milestone 3P Stage A requires CPU observable tensors")


def _build_rows() -> dict[str, dict[str, CandidateRows]]:
    """Build original and counterfactual regimes with group-disjoint splits."""

    original: dict[str, CandidateRows] = {}
    balanced: dict[str, CandidateRows] = {}
    matched: dict[str, CandidateRows] = {}
    negative: dict[str, CandidateRows] = {}
    for split in ("train", "val", "test"):
        original[split] = _materialize(
            HardCandidateActionDataset(
                candidate_count=CANDIDATE_COUNT,
                split=split,
                split_counts=SPLIT_COUNTS,
                ood_groups_per_axis=1,
                include_hidden_state=False,
            )
        )
        for regime, destination in (
            ("static_balanced", balanced),
            ("matched_counterfactual", matched),
            ("negative_shortcut_control", negative),
        ):
            destination[split] = _materialize(
                CounterfactualTemporalDataset(
                    split=split,
                    regime=regime,
                    split_counts=SPLIT_COUNTS,
                    include_hidden_state=False,
                )
            )
    for collection in (original, balanced, matched, negative):
        for rows in collection.values():
            _assert_rows_cpu_only(rows)
    return {"original_distribution": original, "static_balanced": balanced, "matched_counterfactual": matched, "negative_shortcut_control": negative}


def _regime_result(
    *,
    name: str,
    rows: Mapping[str, CandidateRows],
    include_previous_verifier: bool,
    seeds: Sequence[int],
) -> dict[str, Any]:
    """Evaluate the fixed static and temporal method catalogue for one regime."""

    train, validation, test = rows["train"], rows["val"], rows["test"]
    methods = _static_method_reports(train, validation, test, seeds=seeds)
    methods.update(_temporal_method_reports(train, validation, test, seeds=seeds))
    if include_previous_verifier:
        methods["PreviousDynamicVerifier"] = _load_previous_dynamic_scores(test)
    return {
        "artifact_schema": SCHEMA_VERSION,
        "regime": name,
        "split_protocol": {
            "unit_of_split": "base scene / counterfactual pair group",
            "candidate_rows": {split: int(len(value.success)) for split, value in rows.items()},
            "candidate_sets": {split: int(len({int(item["candidate_set_id"]) for item in value.metadata})) for split, value in rows.items()},
            "no_candidate_or_pair_crosses_splits": True,
        },
        "static_methods": [
            "CurrentPositionProximity", "EndpointDistance", "StaticAnalytic", "StaticLogisticRegression",
            "StaticShallowTree", "StaticTwoLayerMLP", "CandidateTemplateOnly", "SceneOnly", "ActionOnly",
        ],
        "dynamic_methods": ["TimeAlignedRelativeFeatures", "TemporalTwoLayerMLP"] + (["PreviousDynamicVerifier"] if include_previous_verifier else []),
        "methods": methods,
    }


def _method_top1(regime: Mapping[str, Any], method: str) -> float:
    return float(regime["methods"][method]["summary"]["top1_success"]["mean"])


def _best_static_method(regime: Mapping[str, Any]) -> tuple[str, float]:
    names = regime["static_methods"]
    return max(((str(name), _method_top1(regime, str(name))) for name in names), key=lambda item: item[1])


def _score_for_observable_baseline(rows: CandidateRows, name: str) -> np.ndarray:
    if name == "CurrentPositionProximity":
        return _baseline_observable_scores(rows, CurrentPositionProximity())
    if name == "TimeAlignedRelativeFeatures":
        return _time_aligned_scores(rows)
    raise ValueError(f"unknown observable baseline {name!r}")


def _family_dynamic_advantages(rows: CandidateRows) -> dict[str, Any]:
    """Paired static-vs-temporal evidence across every held-out task family."""

    result: dict[str, Any] = {}
    for family in COUNTERFACTUAL_FAMILIES:
        subset = _subset_rows(rows, lambda metadata, family=family: metadata.get("counterfactual_family") == family)
        static_score = _score_for_observable_baseline(subset, "CurrentPositionProximity")
        dynamic_score = _score_for_observable_baseline(subset, "TimeAlignedRelativeFeatures")
        static = _metric_report(subset, static_score, score_name="CurrentPositionProximity")
        dynamic = _metric_report(subset, dynamic_score, score_name="TimeAlignedRelativeFeatures")
        result[family] = {
            "candidate_sets": int(len({int(item["candidate_set_id"]) for item in subset.metadata})),
            "static": static,
            "dynamic": dynamic,
            "dynamic_advantage_paired_bootstrap": _paired_dynamic_advantage(static, dynamic),
            "static_counterfactual_pair_accuracy": _counterfactual_pair_accuracy(subset, static_score),
            "dynamic_counterfactual_pair_accuracy": _counterfactual_pair_accuracy(subset, dynamic_score),
        }
    return result


def _static_shortcut_metrics(original: Mapping[str, Any], matched: Mapping[str, Any]) -> dict[str, Any]:
    original_name, original_value = _best_static_method(original)
    matched_name, matched_value = _best_static_method(matched)
    static_primary = original["methods"]["CurrentPositionProximity"]["seed_results"][0]["metrics"]
    dynamic_primary = original["methods"]["PreviousDynamicVerifier"]["seed_results"][0]["metrics"]
    return {
        "selection_rule": "The all-static maximum is reported descriptively; the preregistered static comparator for paired counterfactual inference is CurrentPositionProximity.",
        "original_best_static": {"method": original_name, "top1_success": original_value},
        "matched_best_static": {"method": matched_name, "top1_success": matched_value},
        "matched_minus_original_best_static_top1": float(matched_value - original_value),
        "original_static_vs_frozen_dynamic": {
            "static_method": "CurrentPositionProximity",
            "dynamic_method": "PreviousDynamicVerifier",
            "top1_static_minus_dynamic": float(static_primary["top1_success"] - dynamic_primary["top1_success"]),
            "normalized_regret_static_minus_dynamic": float(static_primary["normalized_regret"] - dynamic_primary["normalized_regret"]),
        },
    }


def _confirmation_run(rows: Mapping[str, CandidateRows]) -> dict[str, Any]:
    """Five-seed confirmation for the two learned compact methods only."""

    train, validation, test = rows["train"], rows["val"], rows["test"]
    return {
        "seeds": list(CONFIRMATION_SEEDS),
        "static_two_layer_mlp": _neural_reports(
            train, validation, test, method="StaticTwoLayerMLP_confirmation",
            train_features=train.static, validation_features=validation.static, test_features=test.static,
            seeds=CONFIRMATION_SEEDS,
        ),
        "temporal_two_layer_mlp": _neural_reports(
            train, validation, test, method="TemporalTwoLayerMLP_confirmation",
            train_features=np.concatenate((train.static, train.dynamic), axis=1),
            validation_features=np.concatenate((validation.static, validation.dynamic), axis=1),
            test_features=np.concatenate((test.static, test.dynamic), axis=1),
            seeds=CONFIRMATION_SEEDS,
        ),
    }


def _decision_matrix(
    *,
    original: Mapping[str, Any],
    balanced: Mapping[str, Any],
    matched: Mapping[str, Any],
    matching: Mapping[str, Any],
    corruption: Mapping[str, Any],
    negative: Mapping[str, Any],
    family_advantages: Mapping[str, Any],
    tests: Mapping[str, Any],
    prior: Mapping[str, Any],
) -> dict[str, Any]:
    original_static_name, original_static = _best_static_method(original)
    matched_static_name, matched_static = _best_static_method(matched)
    frozen_dynamic = _method_top1(original, "PreviousDynamicVerifier")
    paired = matched["counterfactual_dynamic_advantage"]["top1_success"]
    drops = corruption["temporal_causal_drop"]
    strong_corruptions = [
        name for name, item in drops.items()
        if float(item["top1_drop"]) >= 0.10 and float(item["normalized_regret_increase"]) >= 0.10
    ]
    family_successes = [
        family for family, item in family_advantages.items()
        if float(item["dynamic_advantage_paired_bootstrap"]["top1_success"]["delta"]) >= 0.10
        and float(item["dynamic_advantage_paired_bootstrap"]["top1_success"]["ci_low"]) > 0.0
    ]
    negative_top1 = _method_top1(negative, "CurrentPositionProximity")
    positive_top1_delta = float(paired["delta"])
    gates = {
        "original_distribution_static_is_strong": {
            "passed": bool(original_static >= 0.90 or original_static >= frozen_dynamic - 0.05),
            "observed": {"best_static_method": original_static_name, "best_static_top1": original_static, "frozen_dynamic_top1": frozen_dynamic},
        },
        "matched_static_is_chance_or_loses_at_least_0_20": {
            "passed": bool(matched_static <= 0.60 or original_static - matched_static >= 0.20),
            "observed": {"best_static_method": matched_static_name, "best_static_top1": matched_static, "drop_from_original": original_static - matched_static},
        },
        "dynamic_advantage_at_least_0_10_with_paired_ci": {
            "passed": bool(float(paired["delta"]) >= 0.10 and float(paired["ci_low"]) > 0.0),
            "observed": paired,
        },
        "correct_history_beats_at_least_two_temporal_corruptions": {
            "passed": bool(len(strong_corruptions) >= 2),
            "observed": {"corruptions_with_top1_drop_and_regret_increase_at_least_0_10": strong_corruptions, "all_drops": drops},
        },
        "counterfactual_static_matching_and_no_static_leakage": {"passed": bool(matching["passed"]), "observed": matching},
        "negative_static_shortcut_control": {
            "passed": bool(negative_top1 >= 0.90),
            "observed": {"CurrentPositionProximity_top1": negative_top1},
        },
        "positive_matched_dynamic_control": {
            "passed": bool(positive_top1_delta >= 0.10),
            "observed": matched["counterfactual_dynamic_advantage"],
        },
        "at_least_three_dynamic_task_families": {
            "passed": bool(len(family_successes) >= 3),
            "observed": {"successful_families": family_successes, "all_families": family_advantages},
        },
        "strict_observable_input_separation": {
            "passed": bool(not prior["static_clue_audit"]["source_contains_future_label_as_observable"] and not prior["static_clue_audit"]["future_field_is_in_observable_whitelist"]),
            "observed": prior["static_clue_audit"],
        },
        "complete_test_suite": {"passed": bool(tests["passed"]), "observed": tests},
    }
    all_passed = all(bool(value["passed"]) for value in gates.values())
    return {
        "gate_count": len(gates),
        "gates": gates,
        "all_go_conditions_passed": all_passed,
        "decision": "GO_STAGE_B_LOCAL_GPU_RESOURCE_AUDIT" if all_passed else "D_BENCHMARK_DESIGN_OR_DYNAMIC_METHOD_INSUFFICIENT",
        "non_go_interpretation": None if all_passed else "Do not begin GPU simulator integration; inspect failed gates and choose a benchmark or method revision.",
        "static_balanced_regime_recorded": balanced["regime"],
    }


def run_cpu_shortcut_audit(*, output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    """Run the complete CPU-only Stage-A Milestone 3P audit once."""

    output_root = Path(output_root)
    if output_root.exists():
        raise FileExistsError(f"Milestone 3P output root already exists: {output_root}")
    torch.set_num_threads(1)
    output_root.mkdir(parents=True, exist_ok=False)
    _write_json_new(output_root / "environment.json", _environment_record())
    prior = _prior_audit()
    _write_json_new(output_root / "prior_audit.json", prior)

    all_rows = _build_rows()
    original = _regime_result(name="original_distribution", rows=all_rows["original_distribution"], include_previous_verifier=True, seeds=SEEDS)
    balanced = _regime_result(name="static_balanced", rows=all_rows["static_balanced"], include_previous_verifier=True, seeds=SEEDS)
    matched = _regime_result(name="matched_counterfactual", rows=all_rows["matched_counterfactual"], include_previous_verifier=True, seeds=SEEDS)
    negative = _regime_result(name="negative_shortcut_control", rows=all_rows["negative_shortcut_control"], include_previous_verifier=False, seeds=SEEDS)

    matched_test_dataset = CounterfactualTemporalDataset(
        split="test", regime="matched_counterfactual", split_counts=SPLIT_COUNTS, include_hidden_state=False
    )
    matching = _feature_identity_audit(matched_test_dataset)
    matched_static_score = _score_for_observable_baseline(all_rows["matched_counterfactual"]["test"], "CurrentPositionProximity")
    matched_dynamic_score = _score_for_observable_baseline(all_rows["matched_counterfactual"]["test"], "TimeAlignedRelativeFeatures")
    matched_static_metric = _metric_report(all_rows["matched_counterfactual"]["test"], matched_static_score, score_name="CurrentPositionProximity")
    matched_dynamic_metric = _metric_report(all_rows["matched_counterfactual"]["test"], matched_dynamic_score, score_name="TimeAlignedRelativeFeatures")
    matched["matching_audit"] = matching
    matched["counterfactual_dynamic_advantage"] = _paired_dynamic_advantage(matched_static_metric, matched_dynamic_metric)
    matched["counterfactual_pair_accuracy"] = {
        "CurrentPositionProximity": _counterfactual_pair_accuracy(all_rows["matched_counterfactual"]["test"], matched_static_score),
        "TimeAlignedRelativeFeatures": _counterfactual_pair_accuracy(all_rows["matched_counterfactual"]["test"], matched_dynamic_score),
    }
    matched["dynamic_advantage_definition"] = "paired group bootstrap: temporal TimeAlignedRelativeFeatures minus static CurrentPositionProximity; regret is static minus temporal so positive is temporal improvement"
    matched["five_seed_confirmation"] = _confirmation_run(all_rows["matched_counterfactual"])

    correlation, correlation_rows = _correlation_analysis(all_rows["original_distribution"]["train"], all_rows["original_distribution"]["test"])
    correlation.update({"artifact_schema": SCHEMA_VERSION, "regime": "original_distribution", "static_feature_names": list(STATIC_FEATURE_NAMES)})
    feature_audit = {
        "artifact_schema": SCHEMA_VERSION,
        "static_feature_definition": {
            "strict_observable_only": True,
            "current_geometry_only": True,
            "feature_names": list(STATIC_FEATURE_NAMES),
            "candidate_template_is_explicit_diagnostic_only": True,
            "not_read": ["hidden_state", "future trajectories", "success", "candidate utility", "target masks"],
        },
        "candidate_tables": {
            "original_distribution_test": _candidate_table(all_rows["original_distribution"]["test"], split="test", regime="original_distribution"),
            "static_balanced_test": _candidate_table(all_rows["static_balanced"]["test"], split="test", regime="static_balanced"),
            "matched_counterfactual_test": _candidate_table(all_rows["matched_counterfactual"]["test"], split="test", regime="matched_counterfactual"),
        },
        "matched_feature_identity": matching,
    }
    corruption = _temporal_corruption_report(all_rows["matched_counterfactual"]["test"])
    family_advantages = _family_dynamic_advantages(all_rows["matched_counterfactual"]["test"])
    anti_shortcut = _anti_shortcut_ablation(
        all_rows["matched_counterfactual"]["train"], all_rows["matched_counterfactual"]["val"],
        all_rows["matched_counterfactual"]["test"], all_rows["static_balanced"]["train"],
        all_rows["static_balanced"]["val"], seeds=SEEDS,
    )

    _write_json_new(output_root / "static_feature_audit.json", feature_audit)
    _write_json_new(output_root / "static_correlation_analysis.json", correlation)
    _write_csv_new(output_root / "tables" / "static_feature_bins.csv", correlation_rows)
    _write_csv_new(
        output_root / "tables" / "candidate_static_features.csv",
        [row for table in feature_audit["candidate_tables"].values() for row in table],
    )
    _write_json_new(output_root / "original_distribution_results.json", original)
    _write_json_new(output_root / "static_balanced_results.json", balanced)
    _write_json_new(output_root / "matched_counterfactual_results.json", matched)
    _write_json_new(output_root / "temporal_corruption_results.json", corruption)
    _write_json_new(output_root / "anti_shortcut_training_results.json", anti_shortcut)
    _plot_figures(output_root, original=original, balanced=balanced, matched=matched, correlation=correlation)
    tests = _run_full_tests(output_root)
    decision = _decision_matrix(
        original=original, balanced=balanced, matched=matched, matching=matching, corruption=corruption,
        negative=negative, family_advantages=family_advantages, tests=tests, prior=prior,
    )
    summary = {
        "artifact_schema": SCHEMA_VERSION,
        "completed_at": _timestamp(),
        "stage": "A_cpu_static_shortcut_audit",
        "prior_evidence": prior["prior_result"],
        "static_shortcut_gap": _static_shortcut_metrics(original, matched),
        "matched_counterfactual_dynamic_advantage": matched["counterfactual_dynamic_advantage"],
        "temporal_corruption_summary": corruption["temporal_causal_drop"],
        "negative_control_top1": _method_top1(negative, "CurrentPositionProximity"),
        "family_dynamic_advantages": family_advantages,
        "decision": decision,
        "stage_b_authorized": bool(decision["all_go_conditions_passed"]),
    }
    _write_json_new(output_root / "cpu_summary.json", summary)
    return summary


def _run_completion_test_suite(output_root: Path) -> dict[str, Any]:
    """Run the complete regression suite without overwriting the initial log."""

    test_root = output_root / "tests"
    test_root.mkdir(parents=True, exist_ok=True)
    log = test_root / "full_pytest_completion_extension.log"
    if log.exists():
        raise FileExistsError(log)
    started = time.perf_counter()
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"], cwd=PROJECT_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
        env={**dict(__import__("os").environ), "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
    )
    log.write_text(completed.stdout)
    return {
        "command": f"{sys.executable} -m pytest -q",
        "exit_code": int(completed.returncode),
        "elapsed_seconds": time.perf_counter() - started,
        "log": str(log),
        "passed": bool(completed.returncode == 0),
        "summary": completed.stdout.splitlines()[-1] if completed.stdout.splitlines() else "",
    }


def run_cpu_completion_extension(*, output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    """Add missing Stage-A evidence without modifying the frozen first run."""

    output_root = Path(output_root)
    required = output_root / "cpu_summary.json"
    if not required.exists():
        raise FileNotFoundError(f"initial 3P CPU audit is required first: {required}")
    new_paths = (
        output_root / "permutation_feature_importance.json",
        output_root / "frozen_dynamic_verifier_corruptions.json",
        output_root / "anti_shortcut_training_extension.json",
        output_root / "cpu_completion_extension.json",
    )
    existing = [path for path in new_paths if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite completion evidence: {existing}")
    torch.set_num_threads(1)
    started = time.perf_counter()
    all_rows = _build_rows()
    permutation = _permutation_feature_importance(
        all_rows["original_distribution"]["train"], all_rows["original_distribution"]["val"],
        all_rows["original_distribution"]["test"], seeds=SEEDS,
    )
    frozen_corruptions = _frozen_dynamic_verifier_corruption_report(all_rows["matched_counterfactual"]["test"])
    anti_shortcut = _anti_shortcut_training_extension(
        all_rows["matched_counterfactual"]["train"], all_rows["matched_counterfactual"]["val"],
        all_rows["matched_counterfactual"]["test"], all_rows["static_balanced"]["train"],
        all_rows["static_balanced"]["val"], all_rows["original_distribution"]["test"], seeds=SEEDS,
    )
    _write_json_new(output_root / "permutation_feature_importance.json", permutation)
    _write_json_new(output_root / "frozen_dynamic_verifier_corruptions.json", frozen_corruptions)
    _write_json_new(output_root / "anti_shortcut_training_extension.json", anti_shortcut)
    tests = _run_completion_test_suite(output_root)
    ranked_features = sorted(
        (
            {"feature": name, **payload}
            for name, payload in permutation["features"].items()
        ),
        key=lambda row: float(row["mean_top1_decrease"]), reverse=True,
    )
    summary = {
        "artifact_schema": SCHEMA_VERSION,
        "stage": "A_cpu_completion_extension",
        "completed_at": _timestamp(),
        "elapsed_seconds": time.perf_counter() - started,
        "new_evidence": {
            "permutation_feature_importance": "permutation_feature_importance.json",
            "frozen_dynamic_verifier_corruptions": "frozen_dynamic_verifier_corruptions.json",
            "anti_shortcut_extension": "anti_shortcut_training_extension.json",
        },
        "top_permutation_features": ranked_features[:5],
        "frozen_dynamic_temporal_causal_drops": frozen_corruptions["temporal_causal_drop"],
        "anti_shortcut_interventions": {
            name: {
                "matched_top1": payload["matched_summary"]["top1_success"]["mean"],
                "original_top1": payload["original_summary"]["top1_success"]["mean"],
                "original_history_removed_top1_drop": payload["original_history_removed_top1_drop"],
                "matched_pair_accuracy": payload["mean_counterfactual_pair_accuracy"],
            }
            for name, payload in anti_shortcut["interventions"].items()
        },
        "complete_test_suite": tests,
    }
    _write_json_new(output_root / "cpu_completion_extension.json", summary)
    return summary


def run_cpu_final_verification(*, output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    """Write immutable evidence that the final source and test set pass."""

    output_root = Path(output_root)
    if not (output_root / "cpu_completion_extension.json").exists():
        raise FileNotFoundError("the CPU completion extension must be present before final verification")
    path = output_root / "cpu_final_verification.json"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite final verification: {path}")
    test_root = output_root / "tests"
    log = test_root / "full_pytest_final.log"
    if log.exists():
        raise FileExistsError(f"refusing to overwrite final test log: {log}")
    test_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"], cwd=PROJECT_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
        env={**dict(__import__("os").environ), "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
    )
    log.write_text(completed.stdout)
    report = {
        "artifact_schema": SCHEMA_VERSION,
        "stage": "A_cpu_final_verification",
        "verified_at": _timestamp(),
        "command": f"{sys.executable} -m pytest -q",
        "exit_code": int(completed.returncode),
        "elapsed_seconds": time.perf_counter() - started,
        "passed": bool(completed.returncode == 0),
        "summary": completed.stdout.splitlines()[-1] if completed.stdout.splitlines() else "",
        "log": str(log),
    }
    _write_json_new(path, report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the CPU-only Milestone 3P static-shortcut audit.")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--run", action="store_true", help="perform the immutable full Stage-A audit")
    action.add_argument("--complete-extension", action="store_true", help="add missing immutable Stage-A evidence to a completed audit")
    action.add_argument("--verify-completion", action="store_true", help="write final full-suite evidence for the completed CPU audit")
    arguments = parser.parse_args(argv)
    if arguments.run:
        summary = run_cpu_shortcut_audit()
        message = {"decision": summary["decision"]["decision"], "stage_b_authorized": summary["stage_b_authorized"]}
    elif arguments.complete_extension:
        summary = run_cpu_completion_extension()
        message = {"stage": summary["stage"], "tests_passed": summary["complete_test_suite"]["passed"]}
    else:
        summary = run_cpu_final_verification()
        message = {"stage": summary["stage"], "tests_passed": summary["passed"], "summary": summary["summary"]}
    print(json.dumps(_jsonable(message), sort_keys=True))
    return 0


__all__ = [
    "CandidateRows",
    "DYNAMIC_FEATURE_NAMES",
    "OUTPUT_ROOT",
    "SCHEMA_VERSION",
    "STATIC_FEATURE_NAMES",
    "_materialize",
    "_metric_report",
    "_prior_audit",
    "extract_static_features",
    "extract_temporal_features",
    "run_cpu_completion_extension",
    "run_cpu_final_verification",
    "run_cpu_shortcut_audit",
]


if __name__ == "__main__":
    raise SystemExit(main())
