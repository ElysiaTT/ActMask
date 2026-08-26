"""Shared deterministic GPU training machinery for V6 and V7."""
from __future__ import annotations

import gc
import hashlib
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .common import OUT, read_json, sha256_file, write_json
from .metrics import (
    binary_metrics,
    fit_temperature,
    select_balanced_accuracy_threshold,
    sigmoid,
)
from .modeling import (
    GroupAdditiveControl,
    TemporalActionOnlyControl,
    VectorLinearControl,
    VectorMLPControl,
    model_description,
)


DEVICE = torch.device("cuda")
CONTROL_NAMES = (
    "logistic_regression_combined_nuisance",
    "shallow_mlp_combined_nuisance",
    "temporal_action_only_TCN",
    "current_state_action_summary_MLP",
    "provenance_only_classifier",
    "combined_non_relational_nuisance_classifier",
)


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("high")


def _normalization(value: np.ndarray, train_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    selected = value[train_indices]
    if value.ndim == 3:
        selected = selected.reshape(-1, value.shape[-1])
    mean = selected.mean(0).astype(np.float32)
    std = selected.std(0).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std


def _normalized_tensor(
    value: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> torch.Tensor:
    normalized = (value.astype(np.float32) - mean) / std
    if not np.isfinite(normalized).all():
        raise RuntimeError("non-finite normalized input")
    return torch.from_numpy(normalized).to(DEVICE)


class AuditStore:
    """CPU source arrays shared by sequential training specifications."""

    def __init__(self):
        with np.load(OUT / "unified_audit_arrays.npz") as stored:
            self.arrays = {name: stored[name].copy() for name in stored.files}
        with np.load(OUT / "nuisance_features.npz") as stored:
            self.nuisance = {name: stored[name].copy() for name in stored.files}
        with np.load(OUT / "audit_environment_indices.npz") as stored:
            self.indices = {name: stored[name].copy() for name in stored.files}
        with np.load(OUT / "audit_variant_arrays.npz") as stored:
            self.variants = {name: stored[name].copy() for name in stored.files}
        self.environment_manifest = read_json(OUT / "audit_environment_manifest.json")
        self.schema = read_json(OUT / "nuisance_feature_schema.json")
        self.labels = self.arrays["label"].astype(np.float32)

    def spec_indices(self, spec: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return tuple(
            self.indices[f"spec__{spec}__{partition}"].astype(np.int64)
            for partition in ("train", "validation", "test")
        )  # type: ignore[return-value]


class FairSpecData:
    def __init__(self, store: AuditStore, spec: str):
        self.store = store
        self.spec = spec
        train, _, _ = store.spec_indices(spec)
        source = store.arrays
        self.statistics: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for name in (
            "history_state",
            "candidate_action",
            "language_embedding",
            "visual_history",
        ):
            self.statistics[name] = _normalization(source[name], train)
        normalizer_path = OUT / "verifier_normalizers" / f"{spec}.npz"
        normalizer_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            normalizer_path,
            **{
                f"{name}_{statistic}": value
                for name, pair in self.statistics.items()
                for statistic, value in zip(("mean", "std"), pair, strict=True)
            },
        )
        self.normalizer_path = normalizer_path
        self.values = {
            name: _normalized_tensor(source[name], *self.statistics[name])
            for name in self.statistics
        }
        self.labels = torch.from_numpy(store.labels).to(DEVICE)

    def batch(
        self,
        indices: torch.Tensor,
        required: tuple[str, ...],
        overrides: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        values = self.values if overrides is None else {**self.values, **overrides}
        return {name: values[name][indices] for name in required}

    def normalized_override(self, source_name: str, value: np.ndarray) -> torch.Tensor:
        return _normalized_tensor(value, *self.statistics[source_name])

    def close(self) -> None:
        self.values.clear()
        del self.labels
        gc.collect()
        torch.cuda.empty_cache()


class ControlSpecData:
    def __init__(self, store: AuditStore, spec: str):
        self.store = store
        self.spec = spec
        train, _, _ = store.spec_indices(spec)
        language_names = store.schema["groups"]["language"]
        language_positions = [
            index
            for index, name in enumerate(language_names)
            if (
                name.startswith("task_string_identity[")
                or name.startswith("language_embedding[")
                or name in (
                    "instruction_character_length",
                    "instruction_token_length",
                )
            )
        ]
        self.language_positions = np.asarray(language_positions, dtype=np.int64)
        raw = {
            "action_summary": store.nuisance["action"].astype(np.float32),
            "context": store.nuisance["context"].astype(np.float32),
            "provenance": store.nuisance["provenance"].astype(np.float32),
            "language_core": store.nuisance["language"][
                :, self.language_positions
            ].astype(np.float32),
            "current_state": store.arrays["current_state"].astype(np.float32),
            "candidate_action": store.arrays["candidate_action"].astype(np.float32),
        }
        self.statistics = {
            name: _normalization(value, train) for name, value in raw.items()
        }
        normalizer_path = OUT / "shortcut_control_normalizers" / f"{spec}.npz"
        normalizer_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            normalizer_path,
            language_positions=self.language_positions,
            **{
                f"{name}_{statistic}": value
                for name, pair in self.statistics.items()
                for statistic, value in zip(("mean", "std"), pair, strict=True)
            },
        )
        self.normalizer_path = normalizer_path
        self.values = {
            name: _normalized_tensor(value, *self.statistics[name])
            for name, value in raw.items()
        }
        self.values["combined"] = torch.cat(
            [
                self.values["action_summary"],
                self.values["context"],
                self.values["provenance"],
                self.values["language_core"],
            ],
            dim=-1,
        )
        self.values["current_action"] = torch.cat(
            (self.values["current_state"], self.values["action_summary"]), dim=-1
        )
        self.labels = torch.from_numpy(store.labels).to(DEVICE)

    @property
    def dimensions(self) -> dict[str, int]:
        return {name: int(value.shape[-1]) for name, value in self.values.items()}

    def create_model(self, control: str) -> nn.Module:
        if control == "logistic_regression_combined_nuisance":
            return VectorLinearControl(self.dimensions["combined"])
        if control == "shallow_mlp_combined_nuisance":
            return VectorMLPControl(self.dimensions["combined"])
        if control == "temporal_action_only_TCN":
            return TemporalActionOnlyControl()
        if control == "current_state_action_summary_MLP":
            return VectorMLPControl(self.dimensions["current_action"])
        if control == "provenance_only_classifier":
            return VectorLinearControl(self.dimensions["provenance"])
        if control == "combined_non_relational_nuisance_classifier":
            return GroupAdditiveControl(
                {
                    name: self.dimensions[name]
                    for name in (
                        "action_summary",
                        "context",
                        "provenance",
                        "language_core",
                    )
                }
            )
        raise KeyError(control)

    def batch(self, indices: torch.Tensor, control: str) -> dict[str, torch.Tensor]:
        return self.batch_with_values(indices, control, self.values)

    def batch_with_values(
        self,
        indices: torch.Tensor,
        control: str,
        values: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        if control in (
            "logistic_regression_combined_nuisance",
            "shallow_mlp_combined_nuisance",
        ):
            combined = values.get("combined")
            if combined is None:
                combined = torch.cat(
                    [
                        values["action_summary"],
                        values["context"],
                        values["provenance"],
                        values["language_core"],
                    ],
                    dim=-1,
                )
            return {"vector": combined[indices]}
        if control == "temporal_action_only_TCN":
            return {"candidate_action": values["candidate_action"][indices]}
        if control == "current_state_action_summary_MLP":
            current_action = values.get("current_action")
            if current_action is None:
                current_action = torch.cat(
                    (values["current_state"], values["action_summary"]), dim=-1
                )
            return {"vector": current_action[indices]}
        if control == "provenance_only_classifier":
            return {"vector": values["provenance"][indices]}
        if control == "combined_non_relational_nuisance_classifier":
            return {
                name: values[name][indices]
                for name in (
                    "action_summary",
                    "context",
                    "provenance",
                    "language_core",
                )
            }
        raise KeyError(control)

    def normalized_group(self, name: str, raw: np.ndarray) -> torch.Tensor:
        return _normalized_tensor(raw, *self.statistics[name])

    def close(self) -> None:
        self.values.clear()
        del self.labels
        gc.collect()
        torch.cuda.empty_cache()


@dataclass
class TrainingResult:
    metadata: dict[str, Any]
    validation_logits: np.ndarray
    validation_representation: np.ndarray
    test_logits: np.ndarray
    test_representation: np.ndarray


def _predict(
    model: nn.Module,
    indices: np.ndarray,
    batch_function: Callable[[torch.Tensor], dict[str, torch.Tensor]],
    batch_size: int = 2048,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits = []
    representations = []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            index = torch.as_tensor(
                indices[start : start + batch_size], device=DEVICE, dtype=torch.long
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logit, representation = model(batch_function(index))
            logits.append(logit.float().cpu().numpy())
            representations.append(representation.float().cpu().numpy())
    return (
        np.concatenate(logits).astype(np.float32),
        np.concatenate(representations).astype(np.float32),
    )


def train_one(
    *,
    model: nn.Module,
    model_name: str,
    category: str,
    spec: str,
    seed: int,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    test_indices: np.ndarray,
    labels_tensor: torch.Tensor,
    labels_numpy: np.ndarray,
    batch_function: Callable[[torch.Tensor], dict[str, torch.Tensor]],
    normalizer_path: Path,
    output_paths: dict[str, Path],
    force: bool = False,
) -> TrainingResult:
    record_path = output_paths["record"]
    if not force and record_path.exists():
        record = read_json(record_path)
        required = {
            name: path
            for name, path in output_paths.items()
            if name in ("checkpoint", "prediction", "representation", "curve", "runtime")
        }
        if record.get("complete") and all(
            path.is_file()
            and record["artifact_hashes"].get(name) == sha256_file(path)
            for name, path in required.items()
        ):
            with np.load(output_paths["prediction"]) as stored:
                validation_logits = stored["validation_logit"].copy()
                test_logits = stored["test_logit"].copy()
            with np.load(output_paths["representation"]) as stored:
                validation_representation = stored[
                    "validation_representation"
                ].astype(np.float32)
                test_representation = stored["test_representation"].astype(np.float32)
            return TrainingResult(
                metadata=record,
                validation_logits=validation_logits,
                validation_representation=validation_representation,
                test_logits=test_logits,
                test_representation=test_representation,
            )
    for path in output_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    seed_everything(seed)
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.002, weight_decay=0.0003
    )
    batch_size = 512
    maximum_epochs = 40
    patience_limit = 6
    curve = []
    best_state = None
    best_epoch = -1
    best_validation_ba = -1.0
    patience = 0
    wall_start = time.monotonic()
    gpu_start = torch.cuda.Event(enable_timing=True)
    gpu_end = torch.cuda.Event(enable_timing=True)
    gpu_start.record()
    train_labels = labels_numpy[train_indices].astype(np.float32)
    validation_labels = labels_numpy[validation_indices].astype(np.int8)
    for epoch in range(maximum_epochs):
        model.train()
        generator = np.random.default_rng(seed * 100003 + epoch)
        order = generator.permutation(train_indices)
        losses = []
        for start in range(0, len(order), batch_size):
            batch_indices = torch.as_tensor(
                order[start : start + batch_size],
                device=DEVICE,
                dtype=torch.long,
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logit, _ = model(batch_function(batch_indices))
                loss = F.binary_cross_entropy_with_logits(
                    logit, labels_tensor[batch_indices]
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        validation_logits, _ = _predict(
            model, validation_indices, batch_function
        )
        validation_probability = sigmoid(validation_logits)
        epoch_threshold, validation_ba = select_balanced_accuracy_threshold(
            validation_probability, validation_labels
        )
        curve.append(
            {
                "epoch": epoch,
                "train_BCE": float(np.mean(losses)),
                "validation_balanced_accuracy": validation_ba,
                "validation_threshold": epoch_threshold,
            }
        )
        if validation_ba > best_validation_ba + 1e-12:
            best_validation_ba = validation_ba
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            patience = 0
        else:
            patience += 1
            if patience >= patience_limit:
                break
    if best_state is None:
        raise RuntimeError(f"{category}/{model_name}/{spec}/{seed}: no best state")
    model.load_state_dict(best_state)
    validation_logits, validation_representation = _predict(
        model, validation_indices, batch_function
    )
    test_logits, test_representation = _predict(
        model, test_indices, batch_function
    )
    temperature = fit_temperature(validation_logits, validation_labels)
    validation_calibrated = sigmoid(validation_logits / temperature).astype(np.float32)
    threshold, calibrated_validation_ba = select_balanced_accuracy_threshold(
        validation_calibrated, validation_labels
    )
    test_calibrated = sigmoid(test_logits / temperature).astype(np.float32)
    gpu_end.record()
    torch.cuda.synchronize()
    gpu_seconds = float(gpu_start.elapsed_time(gpu_end) / 1000.0)
    wall_seconds = float(time.monotonic() - wall_start)
    validation_metrics = binary_metrics(
        validation_calibrated, validation_labels, threshold
    )
    test_metrics = binary_metrics(
        test_calibrated, labels_numpy[test_indices].astype(np.int8), threshold
    )
    checkpoint = {
        "schema": "vsa-trained-model-checkpoint-v1",
        "category": category,
        "model_name": model_name,
        "spec": spec,
        "seed": seed,
        "state_dict": best_state,
        "best_epoch": best_epoch,
        "temperature": temperature,
        "threshold": threshold,
        "normalizer_path": str(normalizer_path.resolve()),
        "normalizer_sha256": sha256_file(normalizer_path),
        "model_description": model_description(model),
    }
    torch.save(checkpoint, output_paths["checkpoint"])
    np.savez_compressed(
        output_paths["prediction"],
        validation_index=validation_indices.astype(np.int32),
        validation_label=validation_labels.astype(np.int8),
        validation_logit=validation_logits.astype(np.float32),
        validation_raw_probability=sigmoid(validation_logits).astype(np.float32),
        validation_calibrated_probability=validation_calibrated,
        test_index=test_indices.astype(np.int32),
        test_label=labels_numpy[test_indices].astype(np.int8),
        test_logit=test_logits.astype(np.float32),
        test_raw_probability=sigmoid(test_logits).astype(np.float32),
        test_calibrated_probability=test_calibrated,
        temperature=np.asarray([temperature], dtype=np.float32),
        threshold=np.asarray([threshold], dtype=np.float32),
    )
    np.savez_compressed(
        output_paths["representation"],
        validation_index=validation_indices.astype(np.int32),
        validation_representation=validation_representation.astype(np.float16),
        test_index=test_indices.astype(np.int32),
        test_representation=test_representation.astype(np.float16),
    )
    write_json(
        output_paths["curve"],
        {
            "schema": "vsa-training-curve-v1",
            "category": category,
            "model": model_name,
            "spec": spec,
            "seed": seed,
            "selection_metric": "validation balanced accuracy",
            "curve": curve,
            "best_epoch": best_epoch,
            "best_validation_balanced_accuracy_uncalibrated": best_validation_ba,
            "calibrated_validation_balanced_accuracy": calibrated_validation_ba,
            "early_stopped": len(curve) < maximum_epochs,
        },
    )
    write_json(
        output_paths["runtime"],
        {
            "schema": "vsa-training-runtime-v1",
            "category": category,
            "model": model_name,
            "spec": spec,
            "seed": seed,
            "device": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "CUDA": torch.version.cuda,
            "epochs_run": len(curve),
            "gpu_seconds": gpu_seconds,
            "wall_seconds": wall_seconds,
            "maximum_GPU_hours": 20,
        },
    )
    artifact_hashes = {
        name: sha256_file(path)
        for name, path in output_paths.items()
        if name in ("checkpoint", "prediction", "representation", "curve", "runtime")
    }
    metadata = {
        "schema": "vsa-training-run-record-v1",
        "complete": True,
        "category": category,
        "model": model_name,
        "spec": spec,
        "seed": seed,
        "model_description": model_description(model),
        "train_samples": int(len(train_indices)),
        "validation_samples": int(len(validation_indices)),
        "test_samples": int(len(test_indices)),
        "train_positive_rate": float(train_labels.mean()),
        "best_epoch": best_epoch,
        "epochs_run": len(curve),
        "temperature": temperature,
        "threshold": threshold,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "GPU_seconds": gpu_seconds,
        "wall_seconds": wall_seconds,
        "artifact_paths": {
            name: str(path.resolve())
            for name, path in output_paths.items()
            if name != "record"
        },
        "artifact_hashes": artifact_hashes,
        "normalizer_path": str(normalizer_path.resolve()),
        "normalizer_sha256": sha256_file(normalizer_path),
        "future_information_used": False,
        "physical_ground_truth": False,
    }
    write_json(record_path, metadata)
    del model, optimizer, best_state
    gc.collect()
    torch.cuda.empty_cache()
    return TrainingResult(
        metadata=metadata,
        validation_logits=validation_logits,
        validation_representation=validation_representation,
        test_logits=test_logits,
        test_representation=test_representation,
    )


def output_paths(
    category: str, model: str, spec: str, seed: int
) -> dict[str, Path]:
    stem = Path(model) / spec / f"seed_{seed}"
    if category == "verifier":
        return {
            "checkpoint": OUT / "verifier_checkpoints" / stem.with_suffix(".pt"),
            "prediction": OUT
            / "raw_verifier_predictions"
            / stem.with_suffix(".npz"),
            "representation": OUT
            / "verifier_representations"
            / stem.with_suffix(".npz"),
            "curve": OUT
            / "verifier_training_curves"
            / stem.with_suffix(".json"),
            "runtime": OUT / "verifier_runtime" / stem.with_suffix(".json"),
            "record": OUT / "verifier_run_records" / stem.with_suffix(".json"),
        }
    if category == "shortcut_control":
        return {
            "checkpoint": OUT
            / "shortcut_control_checkpoints"
            / stem.with_suffix(".pt"),
            "prediction": OUT
            / "shortcut_control_raw_predictions"
            / stem.with_suffix(".npz"),
            "representation": OUT
            / "shortcut_control_representations"
            / stem.with_suffix(".npz"),
            "curve": OUT
            / "shortcut_control_training_curves"
            / stem.with_suffix(".json"),
            "runtime": OUT
            / "shortcut_control_runtime"
            / stem.with_suffix(".json"),
            "record": OUT
            / "shortcut_control_run_records"
            / stem.with_suffix(".json"),
        }
    raise KeyError(category)
