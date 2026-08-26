"""Deterministic GPU training primitives for the IID-learnability diagnosis."""
from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from actmask.experiments.verifier_shortcut_audit.metrics import (
    binary_metrics,
    fit_temperature,
    select_balanced_accuracy_threshold,
    sigmoid,
)
from actmask.experiments.verifier_shortcut_audit.modeling import (
    VERIFIER_CLASSES,
    model_description,
)

from .common import (
    ARCHITECTURES,
    OUT,
    SOURCE,
    STUDY_ID,
    read_json,
    read_jsonl,
    sha256_file,
    sha256_json,
    verify_bound_source_snapshot,
    verify_payload_sha256,
    write_hashed_json,
    write_json,
)


DEVICE = torch.device("cuda")


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("high")


def _normalization(
    value: np.ndarray, train_indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    selected = value[train_indices]
    if value.ndim == 3:
        selected = selected.reshape(-1, value.shape[-1])
    mean = selected.mean(0).astype(np.float32)
    std = selected.std(0).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std


def _to_normalized_tensor(
    value: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> torch.Tensor:
    normalized = (value.astype(np.float32) - mean) / std
    if not np.isfinite(normalized).all():
        raise RuntimeError("non-finite normalized value")
    return torch.from_numpy(normalized).to(DEVICE)


class StudyStore:
    """Read-only frozen VSA arrays; no method writes to the source study."""

    def __init__(self) -> None:
        self.source_preservation = verify_bound_source_snapshot()
        if not self.source_preservation["pass"]:
            raise RuntimeError(
                f"source study changed: {self.source_preservation}"
            )
        with np.load(SOURCE / "unified_audit_arrays.npz") as stored:
            self.arrays = {name: stored[name].copy() for name in stored.files}
        with np.load(SOURCE / "nuisance_features.npz") as stored:
            self.nuisance = {name: stored[name].copy() for name in stored.files}
        with np.load(SOURCE / "audit_environment_indices.npz") as stored:
            self.indices = {name: stored[name].copy() for name in stored.files}
        for collection in (self.arrays, self.nuisance, self.indices):
            for value in collection.values():
                value.setflags(write=False)
        self.array_sha256 = {
            name: array_sha256(value) for name, value in self.arrays.items()
        }
        self.nuisance_sha256 = {
            name: array_sha256(value) for name, value in self.nuisance.items()
        }
        self.indices_sha256 = {
            name: array_sha256(value) for name, value in self.indices.items()
        }
        self.rows = tuple(
            MappingProxyType(row)
            for row in read_jsonl(SOURCE / "unified_audit_manifest.jsonl")
        )
        self.labels = self.arrays["label"].astype(np.float32)
        self.labels.setflags(write=False)

    def spec_indices(
        self, spec: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return tuple(
            self.indices[f"spec__{spec}__{partition}"].astype(np.int64)
            for partition in ("train", "validation", "test")
        )  # type: ignore[return-value]


class SpecData:
    """Train-only-normalized fair inputs resident on the selected GPU."""

    INPUTS = (
        "history_state",
        "candidate_action",
        "language_embedding",
        "visual_history",
    )

    def __init__(self, store: StudyStore, spec: str) -> None:
        self.store = store
        self.spec = spec
        train, _, _ = store.spec_indices(spec)
        self.statistics = {
            name: _normalization(store.arrays[name], train)
            for name in self.INPUTS
        }
        normalizer_path = OUT / "normalizers" / f"{spec}.npz"
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
            name: _to_normalized_tensor(
                store.arrays[name], *self.statistics[name]
            )
            for name in self.INPUTS
        }

    def batch(
        self,
        indices: torch.Tensor,
        required: tuple[str, ...],
        overrides: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        source = self.values if overrides is None else {**self.values, **overrides}
        return {name: source[name][indices] for name in required}

    def constant_overrides(self, required: tuple[str, ...]) -> dict[str, torch.Tensor]:
        return {name: torch.zeros_like(self.values[name]) for name in required}

    def input_identity(
        self,
        required: tuple[str, ...],
        *,
        variant: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "schema": "vild-input-identity-v1",
            "spec": self.spec,
            "required_inputs": list(required),
            "normalized_inputs": {
                name: {
                    "source_array_sha256": self.store.array_sha256[name],
                    "train_mean_sha256": array_sha256(
                        self.statistics[name][0]
                    ),
                    "train_std_sha256": array_sha256(
                        self.statistics[name][1]
                    ),
                }
                for name in required
            },
            "variant": (
                {"kind": "frozen_fair_inputs"}
                if variant is None
                else variant
            ),
        }

    def close(self) -> None:
        self.values.clear()
        gc.collect()
        torch.cuda.empty_cache()


class VectorDiagnosticMLP(nn.Module):
    """Frozen diagnostic control; it is not an additional verifier."""

    required_inputs = ("vector",)

    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(dimension, 256),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(256, 128),
            nn.GELU(),
        )
        self.head = nn.Linear(128, 1)

    def forward(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        representation = self.encoder(batch["vector"])
        return self.head(representation).squeeze(-1), representation


@dataclass(frozen=True)
class TrainProtocol:
    name: str
    learning_rate: float
    weight_decay: float
    batch_size: int
    maximum_epochs: int
    early_stop_patience: int
    selection: str
    precision: str
    gradient_clip: float = 2.0


def protocol_for(name: str) -> TrainProtocol:
    protocols = {
        "positive_sanity": TrainProtocol(
            name, 0.002, 0.0, 128, 100, 15, "validation_bce", "float32"
        ),
        "shuffled_sanity": TrainProtocol(
            name, 0.002, 0.0, 128, 40, 10, "validation_bce", "float32"
        ),
        "constant_sanity": TrainProtocol(
            name, 0.002, 0.0, 128, 10, 10, "validation_bce", "float32"
        ),
        "alternate": TrainProtocol(
            name, 0.001, 0.0, 128, 200, 30, "validation_bce", "float32"
        ),
        "final_original": TrainProtocol(
            name, 0.002, 0.0003, 512, 40, 6, "validation_ba", "bfloat16"
        ),
        "vector_control": TrainProtocol(
            name, 0.001, 0.0003, 128, 100, 15, "validation_bce", "float32"
        ),
    }
    if name not in protocols:
        raise KeyError(name)
    return protocols[name]


def validate_protocol_against_config(protocol: TrainProtocol) -> None:
    config = read_json(OUT / "preregistered_config.json")
    expected = config.get("machine_training_protocols", {}).get(
        protocol.name
    )
    actual = asdict(protocol)
    if expected is None:
        raise RuntimeError(
            f"missing frozen machine protocol for {protocol.name}"
        )
    sequence_name = (
        "final_original_batch_sequence"
        if protocol.name == "final_original"
        else "diagnostic_batch_sequence"
    )
    allowed_batches = config["OOM_fallback"][sequence_name]
    comparable = dict(actual)
    comparable["batch_size"] = expected["batch_size"]
    if comparable != expected or actual["batch_size"] not in allowed_batches:
        raise RuntimeError(
            "training protocol does not match frozen machine protocol: "
            f"name={protocol.name}, expected={expected}, actual={actual}, "
            f"allowed_batches={allowed_batches}"
        )


def output_paths(
    group: str, model: str, spec: str, seed: int
) -> dict[str, Path]:
    stem = Path(group) / model / spec / f"seed_{seed}"
    return {
        "checkpoint": OUT / "checkpoints" / stem.with_suffix(".pt"),
        "prediction": OUT / "raw_predictions" / stem.with_suffix(".npz"),
        "curve": OUT / "training_curves" / stem.with_suffix(".json"),
        "runtime": OUT / "runtime" / stem.with_suffix(".json"),
        "record": OUT / "run_records" / stem.with_suffix(".json"),
    }


def _autocast(precision: str):
    return torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
        enabled=precision == "bfloat16",
    )


def predict(
    model: nn.Module,
    indices: np.ndarray,
    batch_function: Callable[[torch.Tensor], dict[str, torch.Tensor]],
    precision: str,
    batch_size: int = 2048,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits = []
    representations = []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            index = torch.as_tensor(
                indices[start : start + batch_size],
                device=DEVICE,
                dtype=torch.long,
            )
            with _autocast(precision):
                logit, representation = model(batch_function(index))
            logits.append(logit.float().cpu().numpy())
            representations.append(representation.float().cpu().numpy())
    return (
        np.concatenate(logits).astype(np.float32),
        np.concatenate(representations).astype(np.float32),
    )


def _bce(logits: np.ndarray, labels: np.ndarray) -> float:
    value = np.asarray(logits, dtype=np.float64)
    target = np.asarray(labels, dtype=np.float64)
    return float(
        np.mean(
            np.maximum(value, 0.0)
            - value * target
            + np.log1p(np.exp(-np.abs(value)))
        )
    )


def evaluate_logits(
    logits: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    probability = sigmoid(logits)
    metrics = binary_metrics(probability, labels.astype(np.int8), threshold)
    return {
        **metrics,
        "BCE": _bce(logits, labels),
        "logit_mean": float(np.mean(logits)),
        "logit_standard_deviation": float(np.std(logits)),
        "logit_minimum": float(np.min(logits)),
        "logit_maximum": float(np.max(logits)),
    }


def _artifact_hashes(paths: dict[str, Path]) -> dict[str, str]:
    return {
        name: sha256_file(path)
        for name, path in paths.items()
        if name != "record" and path.is_file()
    }


RESOURCE_LEDGER = OUT / "resource_ledger.jsonl"


def _resource_ledger_rows() -> list[dict[str, Any]]:
    return read_jsonl(RESOURCE_LEDGER) if RESOURCE_LEDGER.exists() else []


def _verify_resource_ledger() -> dict[str, Any]:
    rows = _resource_ledger_rows()
    previous = None
    failures = []
    starts: dict[str, dict[str, Any]] = {}
    ends: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        payload = dict(row)
        recorded = payload.pop("record_sha256", None)
        expected = sha256_json(payload)
        if row.get("sequence") != index:
            failures.append({"row": index, "reason": "sequence"})
        if row.get("previous_record_sha256") != previous:
            failures.append({"row": index, "reason": "previous_hash"})
        if recorded != expected:
            failures.append({"row": index, "reason": "record_hash"})
        attempt_id = row.get("attempt_id")
        if row.get("event") == "START":
            if attempt_id in starts:
                failures.append({"row": index, "reason": "duplicate_start"})
            starts[attempt_id] = row
        elif row.get("event") == "END":
            if attempt_id not in starts or attempt_id in ends:
                failures.append({"row": index, "reason": "unpaired_end"})
            ends[attempt_id] = row
        else:
            failures.append({"row": index, "reason": "unknown_event"})
        previous = recorded
    return {
        "pass": not failures,
        "records": len(rows),
        "failures": failures,
        "starts": starts,
        "ends": ends,
        "open_attempt_ids": sorted(set(starts) - set(ends)),
        "head_record_sha256": previous,
    }


def _append_resource_event(event: dict[str, Any]) -> None:
    ledger = _verify_resource_ledger()
    if not ledger["pass"]:
        raise RuntimeError(
            f"resource ledger is invalid: {ledger['failures']}"
        )
    row = {
        "sequence": ledger["records"],
        "study_id": STUDY_ID,
        "previous_record_sha256": ledger["head_record_sha256"],
        **event,
    }
    row["record_sha256"] = sha256_json(row)
    RESOURCE_LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with RESOURCE_LEDGER.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def start_resource_attempt(
    *,
    logical_identity_sha256: str,
    group: str,
    model: str,
    spec: str,
    seed: int,
    attempt: int,
    batch_size: int,
) -> dict[str, Any]:
    started_unix = time.time()
    attempt_id = sha256_json(
        {
            "logical_identity_sha256": logical_identity_sha256,
            "group": group,
            "model": model,
            "spec": spec,
            "seed": seed,
            "attempt": attempt,
            "batch_size": batch_size,
            "started_unix_ns": time.time_ns(),
            "process_id": os.getpid(),
        }
    )
    gpu_start = torch.cuda.Event(enable_timing=True)
    gpu_start.record()
    _append_resource_event(
        {
            "event": "START",
            "attempt_id": attempt_id,
            "logical_identity_sha256": logical_identity_sha256,
            "group": group,
            "model": model,
            "spec": spec,
            "seed": seed,
            "attempt": attempt,
            "batch_size": batch_size,
            "started_unix": started_unix,
            "process_id": os.getpid(),
        }
    )
    return {
        "attempt_id": attempt_id,
        "started_unix": started_unix,
        "gpu_start": gpu_start,
    }


def finish_resource_attempt(
    token: dict[str, Any],
    *,
    status: str,
    error: str | None = None,
) -> dict[str, Any]:
    wall_seconds = float(max(0.0, time.time() - token["started_unix"]))
    method = "CUDA_event"
    try:
        gpu_end = torch.cuda.Event(enable_timing=True)
        gpu_end.record()
        torch.cuda.synchronize()
        gpu_seconds = float(
            token["gpu_start"].elapsed_time(gpu_end) / 1000.0
        )
    except Exception as timing_error:  # pragma: no cover - defensive OOM path
        gpu_seconds = wall_seconds
        method = "wall_time_conservative_fallback"
        if error is None:
            error = f"timing_error: {timing_error}"
    row = {
        "event": "END",
        "attempt_id": token["attempt_id"],
        "status": status,
        "gpu_seconds": gpu_seconds,
        "wall_seconds": wall_seconds,
        "accounting_method": method,
        "error": error,
        "ended_unix": time.time(),
    }
    _append_resource_event(row)
    return row


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def build_run_identity(
    *,
    model_name: str,
    group: str,
    spec: str,
    seed: int,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    test_indices: np.ndarray | None,
    labels_numpy: np.ndarray,
    protocol: TrainProtocol,
    normalizer_path: Path | None,
    input_identity: dict[str, Any],
) -> dict[str, Any]:
    identity = {
        "schema": "vild-training-run-identity-v1",
        "study_id": STUDY_ID,
        "config_sha256": sha256_file(OUT / "preregistered_config.json"),
        "model": model_name,
        "group": group,
        "spec": spec,
        "seed": seed,
        "protocol": asdict(protocol),
        "train_indices_sha256": array_sha256(
            np.asarray(train_indices, dtype=np.int64)
        ),
        "validation_indices_sha256": array_sha256(
            np.asarray(validation_indices, dtype=np.int64)
        ),
        "test_indices_sha256": (
            None
            if test_indices is None
            else array_sha256(np.asarray(test_indices, dtype=np.int64))
        ),
        "labels_sha256": array_sha256(
            np.asarray(labels_numpy, dtype=np.float32)
        ),
        "normalizer_sha256": (
            None
            if normalizer_path is None
            else sha256_file(normalizer_path)
        ),
        "input_identity": input_identity,
        "input_identity_sha256": sha256_json(input_identity),
    }
    identity["identity_sha256"] = sha256_json(identity)
    return identity


def _cached(
    paths: dict[str, Path],
    identity_sha256: str,
    identity_field: str = "run_identity_sha256",
) -> dict[str, Any] | None:
    if not paths["record"].is_file():
        return None
    record = read_json(paths["record"])
    if not record.get("complete"):
        return None
    if not verify_payload_sha256(record):
        return None
    if record.get(identity_field) != identity_sha256:
        return None
    expected = record.get("artifact_hashes", {})
    required = {"checkpoint", "prediction", "curve", "runtime"}
    if set(expected) != required:
        return None
    if all(
        paths[name].is_file() and sha256_file(paths[name]) == digest
        for name, digest in expected.items()
    ):
        return record
    return None


def resource_status() -> dict[str, Any]:
    ledger = _verify_resource_ledger()
    gpu_seconds = sum(
        float(row.get("gpu_seconds", 0.0))
        for row in ledger["ends"].values()
    )
    output_bytes = sum(
        path.stat().st_size for path in OUT.rglob("*") if path.is_file()
    )
    learned_schemas = {
        "vild-training-run-record-v1",
        "vild-memorization-run-record-v1",
        "vild-gradient-run-record-v1",
    }
    record_attempts: dict[str, str] = {}
    coverage_failures = []
    records_root = OUT / "run_records"
    record_paths = (
        list(records_root.rglob("*.json")) if records_root.exists() else []
    )
    for path in record_paths:
        try:
            record = read_json(path)
        except Exception as error:
            coverage_failures.append(
                {
                    "path": str(path.relative_to(OUT)),
                    "reason": f"unreadable: {error}",
                }
            )
            continue
        if record.get("schema") not in learned_schemas:
            coverage_failures.append(
                {
                    "path": str(path.relative_to(OUT)),
                    "reason": "unknown_run_record_schema",
                    "schema": record.get("schema"),
                }
            )
            continue
        relative = str(path.relative_to(OUT))
        if not record.get("complete") or not verify_payload_sha256(record):
            coverage_failures.append(
                {"path": relative, "reason": "incomplete_or_invalid_self_hash"}
            )
            continue
        attempt_id = record.get("resource_attempt_id")
        if not isinstance(attempt_id, str):
            coverage_failures.append(
                {"path": relative, "reason": "missing_resource_attempt_id"}
            )
            continue
        if attempt_id in record_attempts:
            coverage_failures.append(
                {
                    "path": relative,
                    "reason": "duplicate_resource_attempt_id",
                    "other": record_attempts[attempt_id],
                }
            )
        record_attempts[attempt_id] = relative
        end = ledger["ends"].get(attempt_id)
        if attempt_id not in ledger["starts"] or end is None:
            coverage_failures.append(
                {"path": relative, "reason": "missing_ledger_START_or_END"}
            )
        elif end.get("status") != "SUCCESS":
            coverage_failures.append(
                {
                    "path": relative,
                    "reason": "ledger_END_not_SUCCESS",
                    "status": end.get("status"),
                }
            )
    unmatched_successful_attempts = [
        attempt_id
        for attempt_id, end in ledger["ends"].items()
        if end.get("status") == "SUCCESS"
        and attempt_id not in record_attempts
    ]
    return {
        "GPU_seconds": gpu_seconds,
        "GPU_hours": gpu_seconds / 3600.0,
        "output_bytes": output_bytes,
        "output_GiB": output_bytes / 1024**3,
        "within_GPU_limit": gpu_seconds / 3600.0 < 5.0,
        "within_artifact_limit": output_bytes < 10 * 1024**3,
        "resource_ledger_pass": ledger["pass"],
        "resource_ledger_records": ledger["records"],
        "completed_GPU_attempts": len(ledger["ends"]),
        "open_GPU_attempts": ledger["open_attempt_ids"],
        "resource_record_coverage_pass": not coverage_failures,
        "resource_record_coverage_failures": coverage_failures,
        "resource_linked_run_records": len(record_attempts),
        "superseded_or_orphan_successful_attempts": (
            unmatched_successful_attempts
        ),
    }


def enforce_resources() -> None:
    status = resource_status()
    if (
        not status["resource_ledger_pass"]
        or not status["resource_record_coverage_pass"]
        or status["open_GPU_attempts"]
        or not status["within_GPU_limit"]
        or not status["within_artifact_limit"]
    ):
        raise RuntimeError(f"frozen resource hard stop reached: {status}")


def train_one(
    *,
    model_factory: Callable[[], nn.Module],
    model_name: str,
    group: str,
    spec: str,
    seed: int,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    test_indices: np.ndarray | None,
    labels_numpy: np.ndarray,
    batch_function: Callable[[torch.Tensor], dict[str, torch.Tensor]],
    protocol: TrainProtocol,
    input_identity: dict[str, Any],
    normalizer_path: Path | None = None,
    logical_identity: dict[str, Any] | None = None,
    resources_prechecked: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for learned diagnostics")
    validate_protocol_against_config(protocol)
    paths = output_paths(group, model_name, spec, seed)
    run_identity = build_run_identity(
        model_name=model_name,
        group=group,
        spec=spec,
        seed=seed,
        train_indices=train_indices,
        validation_indices=validation_indices,
        test_indices=test_indices,
        labels_numpy=labels_numpy,
        protocol=protocol,
        normalizer_path=normalizer_path,
        input_identity=input_identity,
    )
    if not force:
        cached = _cached(paths, run_identity["identity_sha256"])
        if cached is not None:
            return cached
    if not resources_prechecked:
        enforce_resources()
    if logical_identity is None:
        logical_identity = run_identity
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    seed_everything(seed)
    model = model_factory().to(DEVICE)
    labels_tensor = torch.from_numpy(labels_numpy.astype(np.float32)).to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=protocol.learning_rate,
        weight_decay=protocol.weight_decay,
    )
    validation_labels = labels_numpy[validation_indices].astype(np.int8)
    curve = []
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    best_value: float | None = None
    patience = 0
    wall_start = time.monotonic()
    gpu_start = torch.cuda.Event(enable_timing=True)
    gpu_end = torch.cuda.Event(enable_timing=True)
    gpu_start.record()
    for epoch in range(protocol.maximum_epochs):
        model.train()
        generator = np.random.default_rng(seed * 100003 + epoch)
        order = generator.permutation(train_indices)
        losses = []
        for start in range(0, len(order), protocol.batch_size):
            index = torch.as_tensor(
                order[start : start + protocol.batch_size],
                device=DEVICE,
                dtype=torch.long,
            )
            optimizer.zero_grad(set_to_none=True)
            with _autocast(protocol.precision):
                logit, _ = model(batch_function(index))
                loss = F.binary_cross_entropy_with_logits(
                    logit, labels_tensor[index]
                )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite training loss in "
                    f"{group}/{model_name}/{spec}/seed_{seed}"
                )
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), protocol.gradient_clip
            )
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        validation_logits, _ = predict(
            model,
            validation_indices,
            batch_function,
            protocol.precision,
            batch_size=protocol.batch_size,
        )
        if not np.isfinite(validation_logits).all():
            raise FloatingPointError(
                f"non-finite validation logits in "
                f"{group}/{model_name}/{spec}/seed_{seed}"
            )
        validation_probability = sigmoid(validation_logits)
        validation_threshold, validation_ba = select_balanced_accuracy_threshold(
            validation_probability, validation_labels
        )
        validation_bce = _bce(validation_logits, validation_labels)
        selection_value = (
            validation_ba
            if protocol.selection == "validation_ba"
            else -validation_bce
        )
        curve.append(
            {
                "epoch": epoch,
                "train_BCE": float(np.mean(losses)),
                "validation_BCE": validation_bce,
                "validation_balanced_accuracy_optimal_threshold": validation_ba,
                "validation_threshold": validation_threshold,
                "last_batch_gradient_norm_before_clip": float(
                    gradient_norm.detach().cpu()
                ),
            }
        )
        if best_value is None or selection_value > best_value + 1e-12:
            best_value = selection_value
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            patience = 0
        else:
            patience += 1
            if patience >= protocol.early_stop_patience:
                break
    if best_state is None:
        raise RuntimeError("no selected model state")
    model.load_state_dict(best_state)
    train_logits, _ = predict(
        model,
        train_indices,
        batch_function,
        protocol.precision,
        batch_size=protocol.batch_size,
    )
    validation_logits, _ = predict(
        model,
        validation_indices,
        batch_function,
        protocol.precision,
        batch_size=protocol.batch_size,
    )
    if not np.isfinite(train_logits).all() or not np.isfinite(
        validation_logits
    ).all():
        raise FloatingPointError(
            f"non-finite selected logits in "
            f"{group}/{model_name}/{spec}/seed_{seed}"
        )
    temperature = fit_temperature(validation_logits, validation_labels)
    validation_calibrated = sigmoid(
        validation_logits / temperature
    ).astype(np.float32)
    threshold, validation_ba = select_balanced_accuracy_threshold(
        validation_calibrated, validation_labels
    )
    test_logits = np.empty(0, dtype=np.float32)
    if test_indices is not None:
        test_logits, _ = predict(
            model,
            test_indices,
            batch_function,
            protocol.precision,
            batch_size=protocol.batch_size,
        )
        if not np.isfinite(test_logits).all():
            raise FloatingPointError(
                f"non-finite test logits in "
                f"{group}/{model_name}/{spec}/seed_{seed}"
            )
    gpu_end.record()
    torch.cuda.synchronize()
    gpu_seconds = float(gpu_start.elapsed_time(gpu_end) / 1000.0)
    wall_seconds = float(time.monotonic() - wall_start)

    train_metrics_fixed = evaluate_logits(
        train_logits, labels_numpy[train_indices], 0.5
    )
    validation_metrics_fixed = evaluate_logits(
        validation_logits, validation_labels, 0.5
    )
    validation_metrics_selected = binary_metrics(
        validation_calibrated, validation_labels, threshold
    )
    test_metrics_selected = None
    test_metrics_fixed = None
    oracle_test = None
    if test_indices is not None:
        test_labels = labels_numpy[test_indices].astype(np.int8)
        test_calibrated = sigmoid(test_logits / temperature).astype(np.float32)
        test_metrics_selected = binary_metrics(
            test_calibrated, test_labels, threshold
        )
        test_metrics_fixed = evaluate_logits(test_logits, test_labels, 0.5)
        oracle_threshold, oracle_ba = select_balanced_accuracy_threshold(
            test_calibrated, test_labels
        )
        oracle_test = {
            "diagnostic_only": True,
            "threshold": oracle_threshold,
            "balanced_accuracy": oracle_ba,
            "gain_over_validation_selected_threshold": (
                oracle_ba - test_metrics_selected["balanced_accuracy"]
            ),
        }
    checkpoint = {
        "schema": "vild-model-checkpoint-v1",
        "study_id": STUDY_ID,
        "group": group,
        "model_name": model_name,
        "spec": spec,
        "seed": seed,
        "state_dict": best_state,
        "best_epoch": best_epoch,
        "temperature": temperature,
        "threshold": threshold,
        "protocol": asdict(protocol),
        "model_description": model_description(model),
        "normalizer_path": (
            str(normalizer_path.resolve()) if normalizer_path is not None else None
        ),
        "normalizer_sha256": (
            sha256_file(normalizer_path) if normalizer_path is not None else None
        ),
        "run_identity": run_identity,
        "run_identity_sha256": run_identity["identity_sha256"],
        "logical_identity": logical_identity,
        "logical_identity_sha256": logical_identity["identity_sha256"],
    }
    torch.save(checkpoint, paths["checkpoint"])
    prediction_payload: dict[str, np.ndarray] = {
        "train_index": train_indices.astype(np.int32),
        "train_label": labels_numpy[train_indices].astype(np.int8),
        "train_logit": train_logits.astype(np.float32),
        "validation_index": validation_indices.astype(np.int32),
        "validation_label": validation_labels,
        "validation_logit": validation_logits.astype(np.float32),
        "temperature": np.asarray([temperature], dtype=np.float64),
        "threshold": np.asarray([threshold], dtype=np.float64),
    }
    if test_indices is not None:
        prediction_payload.update(
            {
                "test_index": test_indices.astype(np.int32),
                "test_label": labels_numpy[test_indices].astype(np.int8),
                "test_logit": test_logits.astype(np.float32),
            }
        )
    np.savez_compressed(paths["prediction"], **prediction_payload)
    write_json(
        paths["curve"],
        {
            "schema": "vild-training-curve-v1",
            "study_id": STUDY_ID,
            "group": group,
            "model": model_name,
            "spec": spec,
            "seed": seed,
            "protocol": asdict(protocol),
            "curve": curve,
            "best_epoch": best_epoch,
            "selection": protocol.selection,
            "run_identity_sha256": run_identity["identity_sha256"],
            "logical_identity_sha256": logical_identity["identity_sha256"],
        },
    )
    write_json(
        paths["runtime"],
        {
            "schema": "vild-training-runtime-v1",
            "study_id": STUDY_ID,
            "group": group,
            "model": model_name,
            "spec": spec,
            "seed": seed,
            "device": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "CUDA": torch.version.cuda,
            "gpu_seconds": gpu_seconds,
            "wall_seconds": wall_seconds,
            "batch_size": protocol.batch_size,
            "OOM_fallback_used": False,
            "run_identity_sha256": run_identity["identity_sha256"],
            "logical_identity_sha256": logical_identity["identity_sha256"],
        },
    )
    artifact_hashes = _artifact_hashes(paths)
    record = {
        "schema": "vild-training-run-record-v1",
        "study_id": STUDY_ID,
        "complete": True,
        "group": group,
        "model": model_name,
        "spec": spec,
        "seed": seed,
        "protocol": asdict(protocol),
        "model_description": model_description(model),
        "train_samples": int(len(train_indices)),
        "validation_samples": int(len(validation_indices)),
        "test_samples": int(0 if test_indices is None else len(test_indices)),
        "train_positive_rate": float(labels_numpy[train_indices].mean()),
        "validation_positive_rate": float(
            labels_numpy[validation_indices].mean()
        ),
        "test_positive_rate": (
            None if test_indices is None else float(labels_numpy[test_indices].mean())
        ),
        "epochs_run": len(curve),
        "best_epoch": best_epoch,
        "temperature": temperature,
        "threshold": threshold,
        "train_metrics_fixed_threshold": train_metrics_fixed,
        "validation_metrics_fixed_threshold": validation_metrics_fixed,
        "validation_metrics_selected_threshold": validation_metrics_selected,
        "test_metrics_fixed_threshold": test_metrics_fixed,
        "test_metrics_selected_threshold": test_metrics_selected,
        "oracle_test_threshold_diagnostic": oracle_test,
        "GPU_seconds": gpu_seconds,
        "wall_seconds": wall_seconds,
        "artifact_paths": {
            name: str(path.resolve())
            for name, path in paths.items()
            if name != "record"
        },
        "artifact_hashes": artifact_hashes,
        "heldout_test_used_for_selection": False,
        "physical_ground_truth": False,
        "run_identity": run_identity,
        "run_identity_sha256": run_identity["identity_sha256"],
        "logical_identity": logical_identity,
        "logical_identity_sha256": logical_identity["identity_sha256"],
    }
    record = write_hashed_json(paths["record"], record)
    del model, optimizer, labels_tensor, best_state
    gc.collect()
    torch.cuda.empty_cache()
    return record


def gradient_diagnostic(
    *,
    architecture: str,
    spec: str,
    seed: int,
    indices: np.ndarray,
    labels_numpy: np.ndarray,
    batch_function: Callable[[torch.Tensor], dict[str, torch.Tensor]],
    learning_rate: float = 0.001,
    microbatch_size: int | None = None,
    resources_prechecked: bool = False,
) -> dict[str, Any]:
    if architecture not in VERIFIER_CLASSES:
        raise KeyError(architecture)
    if not resources_prechecked:
        enforce_resources()
    config = read_json(OUT / "preregistered_config.json")
    frozen_learning_rate = float(
        config["gradient_flow"]["optimizer_step_learning_rate"]
    )
    if learning_rate != frozen_learning_rate:
        raise RuntimeError(
            f"gradient learning rate {learning_rate} != frozen "
            f"{frozen_learning_rate}"
        )
    seed_everything(seed)
    model = VERIFIER_CLASSES[architecture]().to(DEVICE)
    model.train()
    if microbatch_size is None:
        microbatch_size = len(indices)
    label_all = torch.from_numpy(
        labels_numpy[indices].astype(np.float32)
    ).to(DEVICE)
    initial = {
        name: value.detach().clone() for name, value in model.named_parameters()
    }
    gpu_start = torch.cuda.Event(enable_timing=True)
    gpu_end = torch.cuda.Event(enable_timing=True)
    gpu_start.record()
    model.zero_grad(set_to_none=True)
    logits = []
    loss_value = 0.0
    for start in range(0, len(indices), microbatch_size):
        selected = indices[start : start + microbatch_size]
        index = torch.as_tensor(selected, device=DEVICE, dtype=torch.long)
        label = label_all[start : start + len(selected)]
        logit, _ = model(batch_function(index))
        loss = F.binary_cross_entropy_with_logits(logit, label)
        weight = len(selected) / len(indices)
        (loss * weight).backward()
        loss_value += float(loss.detach().cpu()) * weight
        logits.append(logit.detach())
    logit = torch.cat(logits)
    gradient_squared = 0.0
    gradient_tensors = 0
    trainable_tensors = 0
    nonfinite = []
    per_parameter = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        trainable_tensors += 1
        if parameter.grad is None:
            per_parameter.append({"name": name, "gradient_norm": 0.0})
            continue
        gradient = parameter.grad.detach().float()
        if not torch.isfinite(gradient).all():
            nonfinite.append(name)
        norm = float(torch.linalg.vector_norm(gradient).cpu())
        if norm > 0:
            gradient_tensors += 1
        gradient_squared += norm * norm
        per_parameter.append({"name": name, "gradient_norm": norm})
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=0.0
    )
    optimizer.step()
    update_squared = 0.0
    parameter_squared = 0.0
    for name, parameter in model.named_parameters():
        delta = parameter.detach() - initial[name]
        update_squared += float(torch.sum(delta.float() ** 2))
        parameter_squared += float(torch.sum(initial[name].float() ** 2))
    model.eval()
    with torch.no_grad():
        post_logits = []
        for start in range(0, len(indices), microbatch_size):
            selected = indices[start : start + microbatch_size]
            index = torch.as_tensor(
                selected, device=DEVICE, dtype=torch.long
            )
            post_logit, _ = model(batch_function(index))
            post_logits.append(post_logit)
        post_logit = torch.cat(post_logits)
        post_loss = F.binary_cross_entropy_with_logits(
            post_logit, label_all
        )
    gpu_end.record()
    torch.cuda.synchronize()
    gpu_seconds = float(gpu_start.elapsed_time(gpu_end) / 1000.0)
    result = {
        "architecture": architecture,
        "spec": spec,
        "seed": seed,
        "samples": int(len(indices)),
        "effective_microbatch_size": int(microbatch_size),
        "initial_loss": loss_value,
        "post_update_loss": float(post_loss.detach().cpu()),
        "initial_logit_mean": float(logit.detach().float().mean().cpu()),
        "initial_logit_standard_deviation": float(
            logit.detach().float().std(unbiased=False).cpu()
        ),
        "finite_logits": bool(torch.isfinite(logit).all()),
        "finite_loss": math.isfinite(loss_value),
        "nonfinite_gradient_parameters": nonfinite,
        "total_gradient_norm": float(math.sqrt(gradient_squared)),
        "trainable_parameter_tensors": trainable_tensors,
        "nonzero_gradient_parameter_tensors": gradient_tensors,
        "parameter_tensor_gradient_coverage": float(
            gradient_tensors / max(trainable_tensors, 1)
        ),
        "parameter_update_norm": float(math.sqrt(update_squared)),
        "relative_parameter_update_norm": float(
            math.sqrt(update_squared / max(parameter_squared, 1e-30))
        ),
        "per_parameter": per_parameter,
        "GPU_seconds": gpu_seconds,
    }
    del model, optimizer, initial, index, label_all
    gc.collect()
    torch.cuda.empty_cache()
    return result
