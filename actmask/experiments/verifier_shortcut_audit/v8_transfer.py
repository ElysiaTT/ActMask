"""V8 frozen family-to-family verifier transfer evaluation.

This module does not train, tune, or recalibrate a model.  It loads each
preregistered fair verifier checkpoint together with the checkpoint-recorded
train-only normalizer and evaluates it on every frozen N1--N6 test partition.
The labels remain the historical construction labels; they are explicitly not
physical-validity ground truth.
"""
from __future__ import annotations

import csv
import gc
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from .common import OUT, SEEDS, read_json, sha256_file, write_json
from .metrics import binary_metrics, sigmoid
from .modeling import VERIFIER_CLASSES
from .train_core import DEVICE


TRAINING_SOURCES = (
    "within_n1",
    "within_n2",
    "within_n3",
    "within_n4",
    "within_n5",
    "within_n6",
    "group_e_to_r",
    "group_r_to_e",
)
TARGET_IDS = ("N1", "N2", "N3", "N4", "N5", "N6")
SOURCE_CONTENTS = {
    "within_n1": ["N1"],
    "within_n2": ["N2"],
    "within_n3": ["N3"],
    "within_n4": ["N4"],
    "within_n5": ["N5"],
    "within_n6": ["N6"],
    "group_e_to_r": ["N2", "N3", "N5", "N6"],
    "group_r_to_e": ["N1", "N4"],
}
INPUT_NAMES = (
    "history_state",
    "candidate_action",
    "language_embedding",
    "visual_history",
)
METRIC_KEYS = (
    "balanced_accuracy",
    "AUROC",
    "AUPRC",
    "ECE",
    "Brier",
    "false_accept_rate",
    "false_reject_rate",
)
LABEL_SEMANTICS = (
    "historical constructed positive-versus-generator-negative label; "
    "not physical-validity ground truth"
)


class FrozenAuditInputs:
    """Relevant immutable audit arrays and frozen family test indices."""

    def __init__(self) -> None:
        with np.load(OUT / "unified_audit_arrays.npz", allow_pickle=False) as stored:
            self.arrays = {
                name: stored[name].astype(np.float32, copy=True)
                for name in INPUT_NAMES
            }
            self.labels = stored["label"].astype(np.int8, copy=True)
            self.family_index = stored["family_index"].astype(np.int8, copy=True)
        with np.load(
            OUT / "audit_environment_indices.npz", allow_pickle=False
        ) as stored:
            self.target_indices = {
                target: stored[
                    f"spec__within_n{number}__test"
                ].astype(np.int64, copy=True)
                for number, target in enumerate(TARGET_IDS, start=1)
            }
        statistics = read_json(OUT / "unified_audit_statistics.json")
        self.target_names = {
            target: statistics["families"][number - 1]
            for number, target in enumerate(TARGET_IDS, start=1)
        }
        self._validate()

    def _validate(self) -> None:
        if len(self.labels) != len(self.family_index):
            raise RuntimeError("unified label/family arrays have inconsistent lengths")
        seen: set[int] = set()
        for number, target in enumerate(TARGET_IDS, start=1):
            indices = self.target_indices[target]
            if len(indices) == 0:
                raise RuntimeError(f"{target} frozen test partition is empty")
            if len(np.unique(indices)) != len(indices):
                raise RuntimeError(f"{target} frozen test partition contains duplicates")
            overlap = seen.intersection(int(value) for value in indices)
            if overlap:
                raise RuntimeError(
                    f"frozen family test partitions overlap at {min(overlap)}"
                )
            seen.update(int(value) for value in indices)
            if not np.all(self.family_index[indices] == number - 1):
                raise RuntimeError(
                    f"{target} test partition contains another construction family"
                )
            unique_labels, counts = np.unique(
                self.labels[indices], return_counts=True
            )
            label_counts = dict(
                zip(unique_labels.tolist(), counts.tolist(), strict=True)
            )
            if set(label_counts) != {0, 1}:
                raise RuntimeError(
                    f"{target} test partition lacks one constructed label class"
                )
            if label_counts[0] != label_counts[1]:
                raise RuntimeError(
                    f"{target} frozen test partition is unexpectedly unbalanced"
                )


class FrozenNormalizerInputs:
    """GPU tensors normalized only with a frozen source-training normalizer."""

    def __init__(
        self,
        inputs: FrozenAuditInputs,
        source: str,
        normalizer_path: Path,
        expected_sha256: str,
    ) -> None:
        self.source = source
        if normalizer_path.resolve() != (
            OUT / "verifier_normalizers" / f"{source}.npz"
        ).resolve():
            raise RuntimeError(
                f"{source}: checkpoint points to an unexpected normalizer "
                f"{normalizer_path}"
            )
        actual_sha256 = sha256_file(normalizer_path)
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                f"{source}: normalizer hash mismatch "
                f"{actual_sha256} != {expected_sha256}"
            )
        with np.load(normalizer_path, allow_pickle=False) as stored:
            expected_keys = {
                f"{name}_{statistic}"
                for name in INPUT_NAMES
                for statistic in ("mean", "std")
            }
            if set(stored.files) != expected_keys:
                raise RuntimeError(
                    f"{source}: frozen normalizer keys changed: {stored.files}"
                )
            statistics = {
                name: (
                    stored[f"{name}_mean"].astype(np.float32, copy=True),
                    stored[f"{name}_std"].astype(np.float32, copy=True),
                )
                for name in INPUT_NAMES
            }
        self.values: dict[str, torch.Tensor] = {}
        for name, raw in inputs.arrays.items():
            mean, std = statistics[name]
            if not np.isfinite(mean).all() or not np.isfinite(std).all():
                raise RuntimeError(f"{source}/{name}: non-finite normalizer")
            if np.any(std <= 0):
                raise RuntimeError(f"{source}/{name}: non-positive normalizer scale")
            normalized = (raw - mean) / std
            if not np.isfinite(normalized).all():
                raise RuntimeError(f"{source}/{name}: non-finite normalized input")
            self.values[name] = torch.from_numpy(normalized).to(DEVICE)
        self.path = normalizer_path.resolve()
        self.sha256 = actual_sha256

    def batch(
        self, indices: torch.Tensor, required: tuple[str, ...]
    ) -> dict[str, torch.Tensor]:
        return {name: self.values[name][indices] for name in required}

    def close(self) -> None:
        self.values.clear()
        gc.collect()
        torch.cuda.empty_cache()


def _checkpoint_path(architecture: str, source: str, seed: int) -> Path:
    return (
        OUT
        / "verifier_checkpoints"
        / architecture
        / source
        / f"seed_{seed}.pt"
    )


def _record_path(architecture: str, source: str, seed: int) -> Path:
    return (
        OUT
        / "verifier_run_records"
        / architecture
        / source
        / f"seed_{seed}.json"
    )


def _prediction_path(
    architecture: str, source: str, seed: int, target: str
) -> Path:
    return (
        OUT
        / "family_transfer_raw_predictions"
        / architecture
        / source
        / f"seed_{seed}"
        / f"target_{target.lower()}.npz"
    )


def _load_checkpoint(
    architecture: str, source: str, seed: int
) -> tuple[dict[str, Any], Path, str]:
    path = _checkpoint_path(architecture, source, seed)
    record_path = _record_path(architecture, source, seed)
    if not path.is_file() or not record_path.is_file():
        raise RuntimeError(
            f"missing frozen V7 checkpoint or record: {architecture}/{source}/{seed}"
        )
    checkpoint_sha256 = sha256_file(path)
    record = read_json(record_path)
    if not record.get("complete"):
        raise RuntimeError(f"incomplete V7 record: {record_path}")
    if record.get("artifact_hashes", {}).get("checkpoint") != checkpoint_sha256:
        raise RuntimeError(f"checkpoint hash does not match V7 record: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    expected_identity = {
        "category": "verifier",
        "model_name": architecture,
        "spec": source,
        "seed": seed,
    }
    for key, expected in expected_identity.items():
        if checkpoint.get(key) != expected:
            raise RuntimeError(
                f"{path}: checkpoint {key}={checkpoint.get(key)!r}, "
                f"expected {expected!r}"
            )
    if checkpoint.get("schema") != "vsa-trained-model-checkpoint-v1":
        raise RuntimeError(f"{path}: unexpected checkpoint schema")
    if (
        record.get("normalizer_sha256") != checkpoint.get("normalizer_sha256")
        or Path(checkpoint["normalizer_path"]).resolve()
        != Path(record["normalizer_path"]).resolve()
    ):
        raise RuntimeError(f"{path}: V7 record/checkpoint normalizer disagreement")
    return checkpoint, path.resolve(), checkpoint_sha256


def _predict_logits(
    model: torch.nn.Module,
    indices: np.ndarray,
    data: FrozenNormalizerInputs,
    required: tuple[str, ...],
    batch_size: int = 2048,
) -> np.ndarray:
    model.eval()
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            index = torch.as_tensor(
                indices[start : start + batch_size],
                device=DEVICE,
                dtype=torch.long,
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logit, _ = model(data.batch(index, required))
            chunks.append(logit.float().cpu().numpy())
    return np.concatenate(chunks).astype(np.float32)


def _atomic_savez(path: Path, **payload: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    temporary.replace(path)


def _scalar_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for key in METRIC_KEYS:
        value = metrics[key]
        if value is None or not np.isfinite(float(value)):
            raise RuntimeError(f"undefined transfer metric {key}")
        result[key] = float(value)
    return result


def _evaluate_one_model(
    *,
    inputs: FrozenAuditInputs,
    data: FrozenNormalizerInputs,
    architecture: str,
    source: str,
    seed: int,
    checkpoint: dict[str, Any],
    checkpoint_path: Path,
    checkpoint_sha256: str,
) -> list[dict[str, Any]]:
    model = VERIFIER_CLASSES[architecture]().to(DEVICE)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    required = tuple(model.required_inputs)
    temperature = float(checkpoint["temperature"])
    threshold = float(checkpoint["threshold"])
    if not np.isfinite(temperature) or temperature <= 0:
        raise RuntimeError(f"{checkpoint_path}: invalid frozen temperature")
    if not np.isfinite(threshold):
        raise RuntimeError(f"{checkpoint_path}: invalid frozen threshold")

    lengths = [len(inputs.target_indices[target]) for target in TARGET_IDS]
    concatenated_indices = np.concatenate(
        [inputs.target_indices[target] for target in TARGET_IDS]
    )
    concatenated_logits = _predict_logits(
        model, concatenated_indices, data, required
    )
    rows: list[dict[str, Any]] = []
    start = 0
    for target, length in zip(TARGET_IDS, lengths, strict=True):
        end = start + length
        indices = concatenated_indices[start:end]
        logits = concatenated_logits[start:end]
        labels = inputs.labels[indices]
        raw_probability = sigmoid(logits).astype(np.float32)
        calibrated_probability = sigmoid(logits / temperature).astype(np.float32)
        prediction_path = _prediction_path(
            architecture, source, seed, target
        )
        _atomic_savez(
            prediction_path,
            index=indices.astype(np.int32),
            label=labels.astype(np.int8),
            constructed_label=labels.astype(np.int8),
            logit=logits.astype(np.float32),
            raw_probability=raw_probability,
            calibrated_probability=calibrated_probability,
            # Keep calibration scalars at checkpoint precision.  A float32
            # threshold can cross tied/near-tied scores and make a raw artifact
            # fail to reproduce the metrics reported in the matrix.
            temperature=np.asarray([temperature], dtype=np.float64),
            threshold=np.asarray([threshold], dtype=np.float64),
            seed=np.asarray([seed], dtype=np.int32),
            architecture=np.asarray([architecture]),
            source_training_spec=np.asarray([source]),
            target_family_id=np.asarray([target]),
            target_family_name=np.asarray([inputs.target_names[target]]),
            physical_ground_truth=np.asarray([False], dtype=np.bool_),
            future_information_used=np.asarray([False], dtype=np.bool_),
        )
        prediction_sha256 = sha256_file(prediction_path)
        metrics = binary_metrics(
            calibrated_probability, labels, threshold
        )
        unique_labels, counts = np.unique(labels, return_counts=True)
        label_counts = dict(
            zip(unique_labels.tolist(), counts.tolist(), strict=True)
        )
        rows.append(
            {
                "architecture": architecture,
                "source_training_spec": source,
                "source_training_families": SOURCE_CONTENTS[source],
                "target_family_id": target,
                "target_family_name": inputs.target_names[target],
                "seed": seed,
                "samples": int(length),
                "constructed_label_counts": {
                    "negative": int(label_counts[0]),
                    "positive": int(label_counts[1]),
                },
                "temperature": temperature,
                "threshold": threshold,
                "metrics": metrics,
                "raw_prediction_path": str(prediction_path.resolve()),
                "raw_prediction_sha256": prediction_sha256,
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha256": checkpoint_sha256,
                "normalizer_path": str(data.path),
                "normalizer_sha256": data.sha256,
                "normalization": "source training partition only; frozen from V7",
                "calibration_and_threshold": (
                    "source validation partition only; frozen from V7"
                ),
                "label_semantics": LABEL_SEMANTICS,
                "physical_ground_truth": False,
                "future_information_used": False,
            }
        )
        start = end
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return rows


def _aggregate(
    rows: Iterable[dict[str, Any]],
    group_keys: tuple[str, ...],
    scope: str,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in group_keys)].append(row)
    aggregate_rows = []
    for key, members in sorted(grouped.items()):
        item = dict(zip(group_keys, key, strict=True))
        metrics: dict[str, dict[str, Any]] = {}
        for metric in METRIC_KEYS:
            values = np.asarray(
                [member["metrics"][metric] for member in members],
                dtype=np.float64,
            )
            metrics[metric] = {
                "mean": float(values.mean()),
                "standard_deviation": float(values.std()),
                "minimum": float(values.min()),
                "maximum": float(values.max()),
                "values": [float(value) for value in values],
            }
        item.update(
            {
                "aggregation_scope": scope,
                "evaluations": len(members),
                "seeds": sorted({int(member["seed"]) for member in members}),
                "samples_per_evaluation": sorted(
                    {int(member["samples"]) for member in members}
                ),
                "metrics": metrics,
                "label_semantics": LABEL_SEMANTICS,
                "physical_ground_truth": False,
            }
        )
        aggregate_rows.append(item)
    return aggregate_rows


def _balanced_accuracy_matrices(
    seed_aggregate: list[dict[str, Any]],
    all_verifier_aggregate: list[dict[str, Any]],
) -> dict[str, Any]:
    per_architecture: dict[str, dict[str, dict[str, float]]] = {}
    for row in seed_aggregate:
        architecture = row["architecture"]
        source = row["source_training_spec"]
        target = row["target_family_id"]
        per_architecture.setdefault(architecture, {}).setdefault(source, {})[
            target
        ] = row["metrics"]["balanced_accuracy"]["mean"]
    all_verifiers: dict[str, dict[str, float]] = {}
    for row in all_verifier_aggregate:
        all_verifiers.setdefault(row["source_training_spec"], {})[
            row["target_family_id"]
        ] = row["metrics"]["balanced_accuracy"]["mean"]
    return {
        "row_order_source_training_specs": list(TRAINING_SOURCES),
        "column_order_target_families": list(TARGET_IDS),
        "per_architecture_seed_mean": per_architecture,
        "all_architectures_and_seeds_mean": all_verifiers,
    }


def _csv_metric_fields(
    metrics: dict[str, Any], aggregate: bool
) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for metric in METRIC_KEYS:
        if aggregate:
            fields[metric] = metrics[metric]["mean"]
            fields[f"{metric}_standard_deviation"] = metrics[metric][
                "standard_deviation"
            ]
        else:
            fields[metric] = metrics[metric]
            fields[f"{metric}_standard_deviation"] = ""
    return fields


def _write_csv(
    path: Path,
    per_seed_rows: list[dict[str, Any]],
    seed_aggregate: list[dict[str, Any]],
    all_verifier_aggregate: list[dict[str, Any]],
) -> None:
    rows: list[dict[str, Any]] = []
    for row in per_seed_rows:
        rows.append(
            {
                "aggregation_scope": "single_seed",
                "architecture": row["architecture"],
                "source_training_spec": row["source_training_spec"],
                "source_training_families": "+".join(
                    row["source_training_families"]
                ),
                "target_family_id": row["target_family_id"],
                "target_family_name": row["target_family_name"],
                "seed": row["seed"],
                "evaluations": 1,
                "samples": row["samples"],
                "constructed_negative_samples": row[
                    "constructed_label_counts"
                ]["negative"],
                "constructed_positive_samples": row[
                    "constructed_label_counts"
                ]["positive"],
                **_csv_metric_fields(row["metrics"], aggregate=False),
                "temperature": row["temperature"],
                "threshold": row["threshold"],
                "raw_prediction_path": row["raw_prediction_path"],
                "raw_prediction_sha256": row["raw_prediction_sha256"],
                "checkpoint_sha256": row["checkpoint_sha256"],
                "normalizer_sha256": row["normalizer_sha256"],
                "label_semantics": LABEL_SEMANTICS,
                "physical_ground_truth": False,
            }
        )
    source_lookup = {
        (row["architecture"], row["source_training_spec"], row["target_family_id"]): row
        for row in per_seed_rows
    }
    for row in seed_aggregate:
        example = source_lookup[
            (
                row["architecture"],
                row["source_training_spec"],
                row["target_family_id"],
            )
        ]
        rows.append(
            {
                "aggregation_scope": "mean_over_seeds",
                "architecture": row["architecture"],
                "source_training_spec": row["source_training_spec"],
                "source_training_families": "+".join(
                    example["source_training_families"]
                ),
                "target_family_id": row["target_family_id"],
                "target_family_name": example["target_family_name"],
                "seed": "aggregate",
                "evaluations": row["evaluations"],
                "samples": example["samples"],
                "constructed_negative_samples": example[
                    "constructed_label_counts"
                ]["negative"],
                "constructed_positive_samples": example[
                    "constructed_label_counts"
                ]["positive"],
                **_csv_metric_fields(row["metrics"], aggregate=True),
                "temperature": "",
                "threshold": "",
                "raw_prediction_path": "",
                "raw_prediction_sha256": "",
                "checkpoint_sha256": "",
                "normalizer_sha256": example["normalizer_sha256"],
                "label_semantics": LABEL_SEMANTICS,
                "physical_ground_truth": False,
            }
        )
    family_name_lookup = {
        row["target_family_id"]: row["target_family_name"]
        for row in per_seed_rows
    }
    for row in all_verifier_aggregate:
        rows.append(
            {
                "aggregation_scope": "mean_over_all_architectures_and_seeds",
                "architecture": "ALL_FAIR_VERIFIERS",
                "source_training_spec": row["source_training_spec"],
                "source_training_families": "+".join(
                    SOURCE_CONTENTS[row["source_training_spec"]]
                ),
                "target_family_id": row["target_family_id"],
                "target_family_name": family_name_lookup[
                    row["target_family_id"]
                ],
                "seed": "aggregate",
                "evaluations": row["evaluations"],
                "samples": row["samples_per_evaluation"][0],
                "constructed_negative_samples": (
                    row["samples_per_evaluation"][0] // 2
                ),
                "constructed_positive_samples": (
                    row["samples_per_evaluation"][0] // 2
                ),
                **_csv_metric_fields(row["metrics"], aggregate=True),
                "temperature": "",
                "threshold": "",
                "raw_prediction_path": "",
                "raw_prediction_sha256": "",
                "checkpoint_sha256": "",
                "normalizer_sha256": "",
                "label_semantics": LABEL_SEMANTICS,
                "physical_ground_truth": False,
            }
        )
    fieldnames = [
        "aggregation_scope",
        "architecture",
        "source_training_spec",
        "source_training_families",
        "target_family_id",
        "target_family_name",
        "seed",
        "evaluations",
        "samples",
        "constructed_negative_samples",
        "constructed_positive_samples",
    ]
    for metric in METRIC_KEYS:
        fieldnames.extend((metric, f"{metric}_standard_deviation"))
    fieldnames.extend(
        (
            "temperature",
            "threshold",
            "raw_prediction_path",
            "raw_prediction_sha256",
            "checkpoint_sha256",
            "normalizer_sha256",
            "label_semantics",
            "physical_ground_truth",
        )
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _validate_raw_artifacts(rows: list[dict[str, Any]]) -> None:
    expected = (
        len(TRAINING_SOURCES)
        * len(VERIFIER_CLASSES)
        * len(SEEDS)
        * len(TARGET_IDS)
    )
    if len(rows) != expected:
        raise RuntimeError(f"expected {expected} transfer rows, found {len(rows)}")
    identities = {
        (
            row["architecture"],
            row["source_training_spec"],
            row["seed"],
            row["target_family_id"],
        )
        for row in rows
    }
    if len(identities) != expected:
        raise RuntimeError("duplicate family-transfer evaluation identities")
    for row in rows:
        path = Path(row["raw_prediction_path"])
        if sha256_file(path) != row["raw_prediction_sha256"]:
            raise RuntimeError(f"raw transfer prediction hash mismatch: {path}")
        with np.load(path, allow_pickle=False) as stored:
            count = int(row["samples"])
            for key in (
                "index",
                "label",
                "constructed_label",
                "logit",
                "raw_probability",
                "calibrated_probability",
            ):
                if stored[key].shape != (count,):
                    raise RuntimeError(f"{path}#{key}: unexpected shape")
            if not np.array_equal(stored["label"], stored["constructed_label"]):
                raise RuntimeError(f"{path}: constructed label copy changed")
            if bool(stored["physical_ground_truth"][0]):
                raise RuntimeError(f"{path}: invalid physical-ground-truth claim")


def run() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("V8 family transfer requires the confirmed CUDA device")
    inputs = FrozenAuditInputs()
    per_seed_rows: list[dict[str, Any]] = []
    checkpoint_manifest: list[dict[str, Any]] = []
    for source_number, source in enumerate(TRAINING_SOURCES, start=1):
        first_checkpoint, _, _ = _load_checkpoint(
            next(iter(VERIFIER_CLASSES)), source, SEEDS[0]
        )
        normalizer_path = Path(first_checkpoint["normalizer_path"])
        normalizer_sha256 = str(first_checkpoint["normalizer_sha256"])
        data = FrozenNormalizerInputs(
            inputs, source, normalizer_path, normalizer_sha256
        )
        for architecture in VERIFIER_CLASSES:
            for seed in SEEDS:
                checkpoint, checkpoint_path, checkpoint_sha256 = _load_checkpoint(
                    architecture, source, seed
                )
                if (
                    Path(checkpoint["normalizer_path"]).resolve() != data.path
                    or checkpoint["normalizer_sha256"] != data.sha256
                ):
                    raise RuntimeError(
                        f"{architecture}/{source}/{seed}: normalizer differs "
                        "within a frozen training specification"
                    )
                checkpoint_manifest.append(
                    {
                        "architecture": architecture,
                        "source_training_spec": source,
                        "seed": seed,
                        "path": str(checkpoint_path),
                        "sha256": checkpoint_sha256,
                        "normalizer_path": str(data.path),
                        "normalizer_sha256": data.sha256,
                    }
                )
                per_seed_rows.extend(
                    _evaluate_one_model(
                        inputs=inputs,
                        data=data,
                        architecture=architecture,
                        source=source,
                        seed=seed,
                        checkpoint=checkpoint,
                        checkpoint_path=checkpoint_path,
                        checkpoint_sha256=checkpoint_sha256,
                    )
                )
        data.close()
        print(
            f"V8 family transfer source={source_number}/{len(TRAINING_SOURCES)} "
            f"name={source} rows={len(per_seed_rows)}",
            flush=True,
        )
    _validate_raw_artifacts(per_seed_rows)
    seed_aggregate = _aggregate(
        per_seed_rows,
        ("architecture", "source_training_spec", "target_family_id"),
        "mean_over_seeds",
    )
    all_verifier_aggregate = _aggregate(
        per_seed_rows,
        ("source_training_spec", "target_family_id"),
        "mean_over_all_architectures_and_seeds",
    )
    matrices = _balanced_accuracy_matrices(
        seed_aggregate, all_verifier_aggregate
    )
    output = {
        "schema": "vsa-generator-transfer-matrix-v1",
        "study_id": "VERIFIER-SHORTCUT-AUDIT-V1",
        "complete": True,
        "evaluation_protocol": (
            "load each frozen fair-verifier checkpoint and its corresponding "
            "source-train-only normalizer; apply the frozen source-validation "
            "temperature and threshold; test on every frozen N1--N6 test set"
        ),
        "training_sources": {
            source: SOURCE_CONTENTS[source] for source in TRAINING_SOURCES
        },
        "target_families": inputs.target_names,
        "architectures": list(VERIFIER_CLASSES),
        "seeds": list(SEEDS),
        "standard_metrics": list(METRIC_KEYS),
        "per_seed_evaluations": len(per_seed_rows),
        "seed_aggregate_evaluations": len(seed_aggregate),
        "all_verifier_aggregate_evaluations": len(all_verifier_aggregate),
        "raw_prediction_directory": str(
            (OUT / "family_transfer_raw_predictions").resolve()
        ),
        "checkpoint_manifest": checkpoint_manifest,
        "per_seed": per_seed_rows,
        "aggregate_over_seeds": seed_aggregate,
        "aggregate_over_architectures_and_seeds": all_verifier_aggregate,
        "balanced_accuracy_matrices": matrices,
        "historical_action_tensors_modified": False,
        "historical_labels_modified": False,
        "future_information_used": False,
        "label_semantics": LABEL_SEMANTICS,
        "physical_ground_truth": False,
    }
    json_path = OUT / "generator_transfer_matrix.json"
    csv_path = OUT / "generator_transfer_matrix.csv"
    write_json(json_path, output)
    _write_csv(
        csv_path, per_seed_rows, seed_aggregate, all_verifier_aggregate
    )
    return {
        "complete": True,
        "checkpoint_count": len(checkpoint_manifest),
        "per_seed_evaluations": len(per_seed_rows),
        "seed_aggregates": len(seed_aggregate),
        "all_verifier_aggregates": len(all_verifier_aggregate),
        "raw_prediction_files": len(per_seed_rows),
        "generator_transfer_matrix_json": str(json_path.resolve()),
        "generator_transfer_matrix_json_sha256": sha256_file(json_path),
        "generator_transfer_matrix_csv": str(csv_path.resolve()),
        "generator_transfer_matrix_csv_sha256": sha256_file(csv_path),
        "physical_ground_truth": False,
    }


def main() -> None:
    print(json.dumps(run(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
