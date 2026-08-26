"""Phases B--E: sanity, memorization, gradient, optimization, and input controls."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch.nn import functional as F

from actmask.experiments.verifier_shortcut_audit.modeling import (
    VERIFIER_CLASSES,
    model_description,
)

from .common import (
    ARCHITECTURES,
    EASY_FAMILIES,
    OUT,
    ROOT,
    SEEDS,
    SOURCE,
    SPECS,
    STUDY_ID,
    append_log,
    read_json,
    read_jsonl,
    sha256_file,
    sha256_json,
    verify_bound_source_snapshot,
    verify_payload_sha256,
    write_hashed_json,
    write_json,
)
from .training import (
    DEVICE,
    SpecData,
    StudyStore,
    TrainProtocol,
    VectorDiagnosticMLP,
    _cached,
    array_sha256,
    build_run_identity,
    enforce_resources,
    evaluate_logits,
    finish_resource_attempt,
    gradient_diagnostic,
    output_paths,
    predict,
    protocol_for,
    resource_status,
    seed_everything,
    start_resource_attempt,
    train_one,
)


def _verify_frozen_implementation() -> dict[str, Any]:
    config = read_json(OUT / "preregistered_config.json")
    if config.get("schema") != "vild-preregistered-config-v1.1":
        raise RuntimeError("amended v1.1 preregistration is required")
    expected_hash = (
        OUT / "preregistered_config.sha256"
    ).read_text(encoding="utf-8").strip()
    if sha256_file(OUT / "preregistered_config.json") != expected_hash:
        raise RuntimeError("preregistration hash mismatch")
    failures = []
    for row in config["implementation_freeze"]:
        path = ROOT / row["path"]
        if (
            not path.is_file()
            or path.stat().st_size != row["bytes"]
            or sha256_file(path) != row["sha256"]
        ):
            failures.append(row["path"])
    if failures:
        raise RuntimeError(f"frozen training implementation changed: {failures}")
    evidence_failures = []
    for row in config["pretraining_evidence_freeze"]:
        path = OUT / row["path"]
        if (
            not path.is_file()
            or path.stat().st_size != row["bytes"]
            or sha256_file(path) != row["sha256"]
        ):
            evidence_failures.append(row["path"])
    if evidence_failures:
        raise RuntimeError(
            f"frozen pre-training evidence changed: {evidence_failures}"
        )
    source_check = verify_bound_source_snapshot(config)
    if not source_check["pass"]:
        raise RuntimeError(f"source study changed: {source_check}")
    return {
        "config_sha256": expected_hash,
        "implementation_files": len(config["implementation_freeze"]),
        "failures": failures,
        "pretraining_evidence_files": len(
            config["pretraining_evidence_freeze"]
        ),
        "pretraining_evidence_failures": evidence_failures,
        "source_preservation": source_check,
    }


def _hash_rank(store: StudyStore, indices: np.ndarray) -> np.ndarray:
    return np.asarray(
        sorted(
            indices.tolist(),
            key=lambda index: hashlib.sha256(
                store.rows[int(index)]["sample_id"].encode("utf-8")
            ).hexdigest(),
        ),
        dtype=np.int64,
    )


def _balanced_first(
    store: StudyStore,
    indices: np.ndarray,
    per_label: int,
) -> np.ndarray:
    selected = []
    for label in (0, 1):
        candidates = indices[store.labels[indices] == label]
        ranked = np.asarray(
            sorted(
                candidates.tolist(),
                key=lambda index: store.rows[int(index)]["sample_id"],
            ),
            dtype=np.int64,
        )
        if len(ranked) < per_label:
            raise RuntimeError(
                f"need {per_label} label={label}, found {len(ranked)}"
            )
        selected.extend(ranked[:per_label].tolist())
    return np.asarray(sorted(selected), dtype=np.int64)


def _update_artifact_hashes(paths: dict[str, Path], record: dict[str, Any]) -> None:
    record["artifact_hashes"] = {
        name: sha256_file(path)
        for name, path in paths.items()
        if name != "record" and path.is_file()
    }
    write_hashed_json(paths["record"], record)


def _require_cache_resource_integrity() -> None:
    status = resource_status()
    if (
        not status["resource_ledger_pass"]
        or status["open_GPU_attempts"]
        or not status["resource_record_coverage_pass"]
    ):
        raise RuntimeError(
            "cached learned artifact has an incomplete or invalid resource "
            f"transaction: {status}"
        )


def _replace_nonfinite_numbers(
    value: Any, path: str = "$"
) -> tuple[Any, list[str]]:
    if isinstance(value, float):
        if math.isfinite(value):
            return value, []
        return None, [path]
    if isinstance(value, dict):
        output = {}
        paths = []
        for key, item in value.items():
            safe, found = _replace_nonfinite_numbers(
                item, f"{path}.{key}"
            )
            output[key] = safe
            paths.extend(found)
        return output, paths
    if isinstance(value, list):
        output = []
        paths = []
        for index, item in enumerate(value):
            safe, found = _replace_nonfinite_numbers(
                item, f"{path}[{index}]"
            )
            output.append(safe)
            paths.extend(found)
        return output, paths
    return value, []


def _train_with_fallback(
    *,
    model_factory: Callable[[], torch.nn.Module],
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
    normalizer_path: Path | None,
) -> dict[str, Any]:
    config = read_json(OUT / "preregistered_config.json")
    sequence = (
        config["OOM_fallback"]["final_original_batch_sequence"]
        if protocol.name == "final_original"
        else config["OOM_fallback"]["diagnostic_batch_sequence"]
    )
    sequence = [size for size in sequence if size <= protocol.batch_size]
    if protocol.batch_size not in sequence:
        sequence.insert(0, protocol.batch_size)
    logical_identity = build_run_identity(
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
    paths = output_paths(group, model_name, spec, seed)
    cached = _cached(
        paths,
        logical_identity["identity_sha256"],
        "logical_identity_sha256",
    )
    if (
        cached is not None
        and cached.get("protocol", {}).get("batch_size") in sequence
    ):
        _require_cache_resource_integrity()
        return cached
    errors = []
    for attempt, batch_size in enumerate(sequence):
        selected = dataclasses.replace(protocol, batch_size=batch_size)
        enforce_resources()
        token = start_resource_attempt(
            logical_identity_sha256=logical_identity["identity_sha256"],
            group=group,
            model=model_name,
            spec=spec,
            seed=seed,
            attempt=attempt,
            batch_size=batch_size,
        )
        try:
            record = train_one(
                model_factory=model_factory,
                model_name=model_name,
                group=group,
                spec=spec,
                seed=seed,
                train_indices=train_indices,
                validation_indices=validation_indices,
                test_indices=test_indices,
                labels_numpy=labels_numpy,
                batch_function=batch_function,
                protocol=selected,
                input_identity=input_identity,
                normalizer_path=normalizer_path,
                logical_identity=logical_identity,
                resources_prechecked=True,
            )
            paths = output_paths(group, model_name, spec, seed)
            runtime = read_json(paths["runtime"])
            runtime["resource_attempt_id"] = token["attempt_id"]
            runtime["OOM_fallback_used"] = bool(attempt)
            runtime["OOM_attempts"] = errors
            runtime["final_batch_size"] = batch_size
            write_json(paths["runtime"], runtime)
            record["resource_attempt_id"] = token["attempt_id"]
            record["OOM_fallback_used"] = bool(attempt)
            record["OOM_attempts"] = errors
            record["requested_protocol"] = dataclasses.asdict(protocol)
            record["protocol"] = dataclasses.asdict(selected)
            _update_artifact_hashes(paths, record)
            finalized_record = read_json(paths["record"])
        except torch.cuda.OutOfMemoryError as error:
            ledger_end = finish_resource_attempt(
                token,
                status="CUDA_OOM",
                error=str(error),
            )
            errors.append(
                {
                    "attempt": attempt,
                    "batch_size": batch_size,
                    "error": str(error),
                    "resource_attempt_id": token["attempt_id"],
                    "accounted_GPU_seconds": ledger_end["gpu_seconds"],
                }
            )
            torch.cuda.empty_cache()
            continue
        except BaseException as error:
            finish_resource_attempt(
                token,
                status="ERROR",
                error=f"{type(error).__name__}: {error}",
            )
            raise
        finish_resource_attempt(token, status="SUCCESS")
        return finalized_record
    raise RuntimeError(
        f"OOM fallback exhausted for {group}/{model_name}/{spec}/{seed}: {errors}"
    )


def _memorize_attempt(
    *,
    store: StudyStore,
    data: SpecData,
    architecture: str,
    spec: str,
    indices: np.ndarray,
    mode: str,
    batch_size: int,
    logical_identity: dict[str, Any],
) -> dict[str, Any]:
    if mode not in {"tiny", "full"}:
        raise KeyError(mode)
    group = f"{mode}_memorization"
    paths = output_paths(group, architecture, spec, 17)
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    seed_everything(17)
    model = VERIFIER_CLASSES[architecture]().to(DEVICE)
    required = tuple(model.required_inputs)
    labels = torch.from_numpy(store.labels.astype(np.float32)).to(DEVICE)
    config = read_json(OUT / "preregistered_config.json")
    if mode == "tiny":
        frozen = config["tiny_memorization"]
        learning_rate = float(frozen["learning_rate"])
        maximum_units = int(frozen["maximum_updates"])
        evaluation_interval = int(frozen["evaluation_interval_updates"])
        criteria = {
            "balanced_accuracy": float(
                frozen["pass"]["training_balanced_accuracy_min"]
            ),
            "AUROC": float(frozen["pass"]["training_AUROC_min"]),
            "BCE": float(frozen["pass"]["training_BCE_max"]),
        }
    else:
        frozen = config["full_training_memorization"]
        learning_rate = float(frozen["learning_rate"])
        maximum_units = int(frozen["maximum_epochs"])
        evaluation_interval = 1
        criteria = {
            "balanced_accuracy": float(
                frozen["pass"]["training_balanced_accuracy_min"]
            ),
            "AUROC": float(frozen["pass"]["training_AUROC_min"]),
            "BCE": float(frozen["pass"]["training_BCE_max"]),
        }
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=float(frozen["weight_decay"]),
    )
    curve = []
    consecutive = 0
    success_stop_reached = False
    best_bce = float("inf")
    best_state = None
    best_unit = -1
    wall_start = time.monotonic()
    gpu_start = torch.cuda.Event(enable_timing=True)
    gpu_end = torch.cuda.Event(enable_timing=True)
    gpu_start.record()
    if mode == "tiny":
        for update in range(1, maximum_units + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            for start in range(0, len(indices), batch_size):
                index = torch.as_tensor(
                    indices[start : start + batch_size],
                    device=DEVICE,
                    dtype=torch.long,
                )
                logit, _ = model(data.batch(index, required))
                loss = F.binary_cross_entropy_with_logits(
                    logit, labels[index]
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"non-finite tiny memorization loss in "
                        f"{architecture}/{spec}"
                    )
                weighted_loss = loss * (len(index) / len(indices))
                weighted_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(frozen["gradient_clip"])
            )
            optimizer.step()
            if update % evaluation_interval:
                continue
            eval_logits, _ = predict(
                model,
                indices,
                lambda x: data.batch(x, required),
                "float32",
                batch_size=batch_size,
            )
            metrics = evaluate_logits(
                eval_logits, store.labels[indices], 0.5
            )
            row = {"update": update, "train_metrics": metrics}
            curve.append(row)
            if metrics["BCE"] < best_bce:
                best_bce = metrics["BCE"]
                best_unit = update
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
            passed = (
                metrics["balanced_accuracy"]
                >= criteria["balanced_accuracy"]
                and metrics["AUROC"] >= criteria["AUROC"]
                and metrics["BCE"] <= criteria["BCE"]
            )
            consecutive = consecutive + 1 if passed else 0
            if consecutive >= 3:
                success_stop_reached = True
                best_bce = metrics["BCE"]
                best_unit = update
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
                break
    else:
        for epoch in range(maximum_units):
            model.train()
            generator = np.random.default_rng(17 * 100003 + epoch)
            order = generator.permutation(indices)
            train_losses = []
            for start in range(0, len(order), batch_size):
                index = torch.as_tensor(
                    order[start : start + batch_size],
                    device=DEVICE,
                    dtype=torch.long,
                )
                optimizer.zero_grad(set_to_none=True)
                logit, _ = model(data.batch(index, required))
                loss = F.binary_cross_entropy_with_logits(
                    logit, labels[index]
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"non-finite full memorization loss in "
                        f"{architecture}/{spec}"
                    )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(frozen["gradient_clip"])
                )
                optimizer.step()
                train_losses.append(float(loss.detach().cpu()))
            eval_logits, _ = predict(
                model,
                indices,
                lambda x: data.batch(x, required),
                "float32",
                batch_size=batch_size,
            )
            metrics = evaluate_logits(
                eval_logits, store.labels[indices], 0.5
            )
            row = {
                "epoch": epoch,
                "batch_train_BCE_mean": float(np.mean(train_losses)),
                "train_metrics": metrics,
            }
            curve.append(row)
            if metrics["BCE"] < best_bce:
                best_bce = metrics["BCE"]
                best_unit = epoch
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
            passed = (
                metrics["balanced_accuracy"]
                >= criteria["balanced_accuracy"]
                and metrics["AUROC"] >= criteria["AUROC"]
                and metrics["BCE"] <= criteria["BCE"]
            )
            consecutive = consecutive + 1 if passed else 0
            if consecutive >= 3:
                success_stop_reached = True
                best_bce = metrics["BCE"]
                best_unit = epoch
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
                break
    if best_state is None:
        raise RuntimeError(f"no memorization state for {architecture}/{spec}")
    model.load_state_dict(best_state)
    final_logits, _ = predict(
        model,
        indices,
        lambda x: data.batch(x, required),
        "float32",
        batch_size=batch_size,
    )
    final_metrics = evaluate_logits(
        final_logits, store.labels[indices], 0.5
    )
    passed = (
        final_metrics["balanced_accuracy"] >= criteria["balanced_accuracy"]
        and final_metrics["AUROC"] >= criteria["AUROC"]
        and final_metrics["BCE"] <= criteria["BCE"]
    )
    gpu_end.record()
    torch.cuda.synchronize()
    gpu_seconds = float(gpu_start.elapsed_time(gpu_end) / 1000.0)
    wall_seconds = float(time.monotonic() - wall_start)
    torch.save(
        {
            "schema": "vild-memorization-checkpoint-v1",
            "mode": mode,
            "architecture": architecture,
            "spec": spec,
            "seed": 17,
            "state_dict": best_state,
            "best_unit": best_unit,
            "model_description": model_description(model),
            "logical_identity": logical_identity,
            "logical_identity_sha256": logical_identity[
                "identity_sha256"
            ],
        },
        paths["checkpoint"],
    )
    np.savez_compressed(
        paths["prediction"],
        index=indices.astype(np.int32),
        label=store.labels[indices].astype(np.int8),
        logit=final_logits.astype(np.float32),
        threshold=np.asarray([0.5], dtype=np.float64),
    )
    write_json(
        paths["curve"],
        {
            "schema": "vild-memorization-curve-v1",
            "mode": mode,
            "architecture": architecture,
            "spec": spec,
            "seed": 17,
            "curve": curve,
            "best_unit": best_unit,
            "success_stop_reached": success_stop_reached,
            "logical_identity_sha256": logical_identity[
                "identity_sha256"
            ],
        },
    )
    write_json(
        paths["runtime"],
        {
            "schema": "vild-training-runtime-v1",
            "group": group,
            "model": architecture,
            "spec": spec,
            "seed": 17,
            "device": torch.cuda.get_device_name(0),
            "gpu_seconds": gpu_seconds,
            "wall_seconds": wall_seconds,
            "batch_size": batch_size,
            "OOM_fallback_used": False,
            "logical_identity_sha256": logical_identity[
                "identity_sha256"
            ],
        },
    )
    record = {
        "schema": "vild-memorization-run-record-v1",
        "complete": True,
        "mode": mode,
        "group": group,
        "architecture": architecture,
        "model": architecture,
        "spec": spec,
        "family": spec.replace("within_", "").upper(),
        "seed": 17,
        "samples": int(len(indices)),
        "criteria": criteria,
        "units_run": (
            int(curve[-1]["update"]) if mode == "tiny" else len(curve)
        ),
        "evaluations_run": len(curve),
        "updates_run": (
            int(curve[-1]["update"]) if mode == "tiny" else None
        ),
        "epochs_run": len(curve) if mode == "full" else None,
        "best_unit": best_unit,
        "success_stop_reached": success_stop_reached,
        "consecutive_pass_evaluations_at_stop": consecutive,
        "final_metrics": final_metrics,
        "pass": passed,
        "GPU_seconds": gpu_seconds,
        "wall_seconds": wall_seconds,
        "effective_microbatch_size": batch_size,
        "logical_identity": logical_identity,
        "logical_identity_sha256": logical_identity[
            "identity_sha256"
        ],
        "artifact_paths": {
            name: str(path.resolve())
            for name, path in paths.items()
            if name != "record"
        },
    }
    _update_artifact_hashes(paths, record)
    del model, optimizer, labels, best_state
    torch.cuda.empty_cache()
    return read_json(paths["record"])


def _memorize_one(
    *,
    store: StudyStore,
    data: SpecData,
    architecture: str,
    spec: str,
    indices: np.ndarray,
    mode: str,
) -> dict[str, Any]:
    if mode not in {"tiny", "full"}:
        raise KeyError(mode)
    config = read_json(OUT / "preregistered_config.json")
    section_name = (
        "tiny_memorization"
        if mode == "tiny"
        else "full_training_memorization"
    )
    frozen = config[section_name]
    requested_batch_size = int(frozen["batch_size"])
    required = tuple(VERIFIER_CLASSES[architecture]().required_inputs)
    input_identity = data.input_identity(required)
    logical_identity = {
        "schema": "vild-memorization-run-identity-v1",
        "study_id": STUDY_ID,
        "config_sha256": sha256_file(OUT / "preregistered_config.json"),
        "mode": mode,
        "group": f"{mode}_memorization",
        "architecture": architecture,
        "spec": spec,
        "seed": 17,
        "indices_sha256": array_sha256(
            np.asarray(indices, dtype=np.int64)
        ),
        "selected_labels_sha256": array_sha256(
            store.labels[indices].astype(np.float32)
        ),
        "normalizer_sha256": sha256_file(data.normalizer_path),
        "input_identity": input_identity,
        "input_identity_sha256": sha256_json(input_identity),
        "frozen_protocol": frozen,
        "requested_batch_size": requested_batch_size,
    }
    logical_identity["identity_sha256"] = sha256_json(logical_identity)
    group = f"{mode}_memorization"
    paths = output_paths(group, architecture, spec, 17)
    sequence = [
        size
        for size in config["OOM_fallback"]["diagnostic_batch_sequence"]
        if size <= requested_batch_size
    ]
    if requested_batch_size not in sequence:
        sequence.insert(0, requested_batch_size)
    cached = _cached(
        paths,
        logical_identity["identity_sha256"],
        "logical_identity_sha256",
    )
    if (
        cached is not None
        and cached.get("effective_microbatch_size") in sequence
    ):
        _require_cache_resource_integrity()
        return cached

    errors = []
    for attempt, batch_size in enumerate(sequence):
        enforce_resources()
        token = start_resource_attempt(
            logical_identity_sha256=logical_identity["identity_sha256"],
            group=group,
            model=architecture,
            spec=spec,
            seed=17,
            attempt=attempt,
            batch_size=batch_size,
        )
        try:
            record = _memorize_attempt(
                store=store,
                data=data,
                architecture=architecture,
                spec=spec,
                indices=indices,
                mode=mode,
                batch_size=batch_size,
                logical_identity=logical_identity,
            )
            runtime = read_json(paths["runtime"])
            runtime["resource_attempt_id"] = token["attempt_id"]
            runtime["OOM_fallback_used"] = bool(attempt)
            runtime["OOM_attempts"] = errors
            runtime["requested_batch_size"] = requested_batch_size
            runtime["final_batch_size"] = batch_size
            write_json(paths["runtime"], runtime)
            record["resource_attempt_id"] = token["attempt_id"]
            record["OOM_fallback_used"] = bool(attempt)
            record["OOM_attempts"] = errors
            record["requested_batch_size"] = requested_batch_size
            record["effective_microbatch_size"] = batch_size
            _update_artifact_hashes(paths, record)
            finalized_record = read_json(paths["record"])
        except torch.cuda.OutOfMemoryError as error:
            ledger_end = finish_resource_attempt(
                token,
                status="CUDA_OOM",
                error=str(error),
            )
            errors.append(
                {
                    "attempt": attempt,
                    "batch_size": batch_size,
                    "error": str(error),
                    "resource_attempt_id": token["attempt_id"],
                    "accounted_GPU_seconds": ledger_end["gpu_seconds"],
                }
            )
            torch.cuda.empty_cache()
            continue
        except BaseException as error:
            finish_resource_attempt(
                token,
                status="ERROR",
                error=f"{type(error).__name__}: {error}",
            )
            raise
        finish_resource_attempt(token, status="SUCCESS")
        return finalized_record
    raise RuntimeError(
        f"memorization OOM fallback exhausted for "
        f"{architecture}/{spec}/{mode}: {errors}"
    )


def _run_gradient_diagnostics(
    store: StudyStore,
) -> dict[str, Any]:
    rows = []
    for family, spec in zip(EASY_FAMILIES, SPECS, strict=True):
        data = SpecData(store, spec)
        train, _, _ = store.spec_indices(spec)
        selected = _balanced_first(store, train, 32)
        for architecture in ARCHITECTURES:
            required = tuple(VERIFIER_CLASSES[architecture]().required_inputs)
            runtime_path = (
                OUT
                / "runtime"
                / "gradient"
                / architecture
                / spec
                / "seed_17.json"
            )
            record_path = (
                OUT
                / "run_records"
                / "gradient"
                / architecture
                / spec
                / "seed_17.json"
            )
            config = read_json(OUT / "preregistered_config.json")
            input_identity = data.input_identity(required)
            logical_identity = {
                "schema": "vild-gradient-run-identity-v1",
                "study_id": STUDY_ID,
                "config_sha256": sha256_file(
                    OUT / "preregistered_config.json"
                ),
                "architecture": architecture,
                "family": family,
                "spec": spec,
                "seed": 17,
                "indices_sha256": array_sha256(
                    np.asarray(selected, dtype=np.int64)
                ),
                "selected_labels_sha256": array_sha256(
                    store.labels[selected].astype(np.float32)
                ),
                "normalizer_sha256": sha256_file(data.normalizer_path),
                "input_identity": input_identity,
                "input_identity_sha256": sha256_json(input_identity),
                "frozen_protocol": config["gradient_flow"],
                "requested_batch_size": int(len(selected)),
            }
            logical_identity["identity_sha256"] = sha256_json(
                logical_identity
            )
            cached = None
            if record_path.is_file():
                candidate = read_json(record_path)
                expected_hashes = candidate.get("artifact_hashes", {})
                if (
                    candidate.get("complete")
                    and verify_payload_sha256(candidate)
                    and candidate.get("logical_identity_sha256")
                    == logical_identity["identity_sha256"]
                    and set(expected_hashes) == {"runtime"}
                    and runtime_path.is_file()
                    and sha256_file(runtime_path)
                    == expected_hashes["runtime"]
                ):
                    cached = candidate["result"]
            if cached is not None:
                _require_cache_resource_integrity()
                rows.append(cached)
                continue
            sequence = [
                size
                for size in config["OOM_fallback"][
                    "diagnostic_batch_sequence"
                ]
                if size <= len(selected)
            ]
            if len(selected) not in sequence:
                sequence.insert(0, len(selected))
            errors = []
            result = None
            for attempt, batch_size in enumerate(sequence):
                enforce_resources()
                token = start_resource_attempt(
                    logical_identity_sha256=logical_identity[
                        "identity_sha256"
                    ],
                    group="gradient",
                    model=architecture,
                    spec=spec,
                    seed=17,
                    attempt=attempt,
                    batch_size=batch_size,
                )
                try:
                    result = gradient_diagnostic(
                        architecture=architecture,
                        spec=spec,
                        seed=17,
                        indices=selected,
                        labels_numpy=store.labels,
                        batch_function=(
                            lambda index, r=required: data.batch(index, r)
                        ),
                        learning_rate=float(
                            config["gradient_flow"][
                                "optimizer_step_learning_rate"
                            ]
                        ),
                        microbatch_size=batch_size,
                        resources_prechecked=True,
                    )
                    result["family"] = family
                    result["pass"] = (
                        result["finite_logits"]
                        and result["finite_loss"]
                        and not result["nonfinite_gradient_parameters"]
                        and result["total_gradient_norm"] >= 1e-8
                        and result["parameter_tensor_gradient_coverage"] >= 0.80
                        and result["parameter_update_norm"] >= 1e-8
                    )
                    result["resource_attempt_id"] = token["attempt_id"]
                    result["OOM_fallback_used"] = bool(attempt)
                    result["OOM_attempts"] = errors
                    result, nonfinite_paths = _replace_nonfinite_numbers(
                        result
                    )
                    result["nonfinite_serialized_paths"] = nonfinite_paths
                    if nonfinite_paths:
                        result["pass"] = False
                    write_json(
                        runtime_path,
                        {
                            "schema": "vild-gradient-runtime-v1",
                            "gpu_seconds": result["GPU_seconds"],
                            "resource_attempt_id": token["attempt_id"],
                            "architecture": architecture,
                            "spec": spec,
                            "seed": 17,
                            "batch_size": batch_size,
                            "OOM_fallback_used": bool(attempt),
                            "OOM_attempts": errors,
                            "logical_identity_sha256": logical_identity[
                                "identity_sha256"
                            ],
                        },
                    )
                    write_hashed_json(
                        record_path,
                        {
                            "schema": "vild-gradient-run-record-v1",
                            "complete": True,
                            "logical_identity": logical_identity,
                            "logical_identity_sha256": logical_identity[
                                "identity_sha256"
                            ],
                            "resource_attempt_id": token["attempt_id"],
                            "result": result,
                            "artifact_paths": {
                                "runtime": str(runtime_path.resolve())
                            },
                            "artifact_hashes": {
                                "runtime": sha256_file(runtime_path)
                            },
                        },
                    )
                except torch.cuda.OutOfMemoryError as error:
                    ledger_end = finish_resource_attempt(
                        token,
                        status="CUDA_OOM",
                        error=str(error),
                    )
                    errors.append(
                        {
                            "attempt": attempt,
                            "batch_size": batch_size,
                            "error": str(error),
                            "resource_attempt_id": token["attempt_id"],
                            "accounted_GPU_seconds": ledger_end[
                                "gpu_seconds"
                            ],
                        }
                    )
                    torch.cuda.empty_cache()
                    continue
                except BaseException as error:
                    finish_resource_attempt(
                        token,
                        status="ERROR",
                        error=f"{type(error).__name__}: {error}",
                    )
                    raise
                finish_resource_attempt(token, status="SUCCESS")
                break
            if result is None:
                raise RuntimeError(
                    f"gradient OOM fallback exhausted for "
                    f"{architecture}/{spec}: {errors}"
                )
            rows.append(result)
        data.close()
    report = {
        "schema": "vild-gradient-flow-report-v1",
        "study_id": STUDY_ID,
        "rows": rows,
        "runs": len(rows),
        "passes": sum(row["pass"] for row in rows),
        "pass": all(row["pass"] for row in rows),
        "criteria": read_json(OUT / "preregistered_config.json")[
            "gradient_flow"
        ]["pass"],
    }
    write_json(OUT / "gradient_flow_report.json", report)
    return report


def _run_memorization(store: StudyStore) -> dict[str, Any]:
    tiny_rows = []
    full_rows = []
    for family, spec in zip(EASY_FAMILIES, SPECS, strict=True):
        data = SpecData(store, spec)
        train, _, _ = store.spec_indices(spec)
        tiny = _balanced_first(store, train, 32)
        for architecture in ARCHITECTURES:
            tiny_rows.append(
                _memorize_one(
                    store=store,
                    data=data,
                    architecture=architecture,
                    spec=spec,
                    indices=tiny,
                    mode="tiny",
                )
            )
            full_rows.append(
                _memorize_one(
                    store=store,
                    data=data,
                    architecture=architecture,
                    spec=spec,
                    indices=train,
                    mode="full",
                )
            )
        data.close()
    architecture_rows = []
    for architecture in ARCHITECTURES:
        tiny_selected = [
            row for row in tiny_rows if row["architecture"] == architecture
        ]
        full_selected = [
            row for row in full_rows if row["architecture"] == architecture
        ]
        architecture_rows.append(
            {
                "architecture": architecture,
                "tiny_families_passed": sum(row["pass"] for row in tiny_selected),
                "full_families_passed": sum(row["pass"] for row in full_selected),
                "all_tiny_pass": all(row["pass"] for row in tiny_selected),
                "all_full_pass": all(row["pass"] for row in full_selected),
            }
        )
    report = {
        "schema": "vild-memorization-report-v1",
        "study_id": STUDY_ID,
        "tiny_rows": tiny_rows,
        "full_rows": full_rows,
        "tiny_runs": len(tiny_rows),
        "full_runs": len(full_rows),
        "tiny_passes": sum(row["pass"] for row in tiny_rows),
        "full_passes": sum(row["pass"] for row in full_rows),
        "architecture_summary": architecture_rows,
        "all_tiny_pass": all(row["pass"] for row in tiny_rows),
        "all_full_pass": all(row["pass"] for row in full_rows),
    }
    write_json(OUT / "memorization_report.json", report)
    return report


def _run_sanity(store: StudyStore) -> dict[str, Any]:
    spec = "within_n2"
    data = SpecData(store, spec)
    train, validation, _ = store.spec_indices(spec)
    train_ranked = _hash_rank(store, train)[:256]
    validation_ranked = _hash_rank(store, validation)[:128]
    synthetic_labels = store.labels.copy()
    synthetic_labels[train_ranked] = np.arange(len(train_ranked)) % 2
    synthetic_labels[validation_ranked] = np.arange(len(validation_ranked)) % 2
    synthetic_action = data.values["candidate_action"].clone()
    for selected in (train_ranked, validation_ranked):
        index = torch.as_tensor(selected, device=DEVICE, dtype=torch.long)
        target = torch.from_numpy(
            synthetic_labels[selected].astype(np.float32)
        ).to(DEVICE)
        synthetic_action[index, :, 0] = target[:, None] * 8.0 - 4.0
    synthetic_action_sha256 = array_sha256(
        synthetic_action.detach().cpu().numpy()
    )
    synthetic_labels_sha256 = array_sha256(synthetic_labels)

    positive_rows = []
    shuffled_rows = []
    constant_rows = []
    shuffled_labels = store.labels.copy()
    shuffled_train = np.random.default_rng(918273).permutation(
        shuffled_labels[train]
    )
    shuffled_validation = np.random.default_rng(918274).permutation(
        shuffled_labels[validation]
    )
    shuffled_labels[train] = shuffled_train
    shuffled_labels[validation] = shuffled_validation
    for architecture in ARCHITECTURES:
        required = tuple(VERIFIER_CLASSES[architecture]().required_inputs)
        positive_override = {"candidate_action": synthetic_action}
        positive = _train_with_fallback(
            model_factory=VERIFIER_CLASSES[architecture],
            model_name=architecture,
            group="positive_sanity",
            spec=spec,
            seed=17,
            train_indices=train_ranked,
            validation_indices=validation_ranked,
            test_indices=None,
            labels_numpy=synthetic_labels,
            batch_function=lambda index, r=required, o=positive_override: data.batch(
                index, r, o
            ),
            protocol=protocol_for("positive_sanity"),
            input_identity=data.input_identity(
                required,
                variant={
                    "kind": "positive_observable_signal",
                    "candidate_action_override_sha256": (
                        synthetic_action_sha256
                    ),
                    "synthetic_labels_sha256": synthetic_labels_sha256,
                },
            ),
            normalizer_path=data.normalizer_path,
        )
        positive_metrics = positive["validation_metrics_fixed_threshold"]
        positive = dict(positive)
        positive["source_record_payload_sha256"] = positive.pop(
            "record_payload_sha256"
        )
        positive["sanity_pass"] = (
            positive_metrics["balanced_accuracy"] >= 0.98
            and positive_metrics["AUROC"] >= 0.995
        )
        positive_rows.append(positive)

        shuffled = _train_with_fallback(
            model_factory=VERIFIER_CLASSES[architecture],
            model_name=architecture,
            group="shuffled_sanity",
            spec=spec,
            seed=918273,
            train_indices=train,
            validation_indices=validation,
            test_indices=None,
            labels_numpy=shuffled_labels,
            batch_function=lambda index, r=required: data.batch(index, r),
            protocol=protocol_for("shuffled_sanity"),
            input_identity=data.input_identity(
                required,
                variant={"kind": "shuffled_label_negative"},
            ),
            normalizer_path=data.normalizer_path,
        )
        shuffled_metrics = shuffled["validation_metrics_fixed_threshold"]
        shuffled = dict(shuffled)
        shuffled["source_record_payload_sha256"] = shuffled.pop(
            "record_payload_sha256"
        )
        shuffled["sanity_pass"] = (
            shuffled_metrics["balanced_accuracy"] <= 0.60
            and 0.35 <= shuffled_metrics["AUROC"] <= 0.65
        )
        shuffled_rows.append(shuffled)

        constant_override = data.constant_overrides(required)
        constant = _train_with_fallback(
            model_factory=VERIFIER_CLASSES[architecture],
            model_name=architecture,
            group="constant_sanity",
            spec=spec,
            seed=17,
            train_indices=train,
            validation_indices=validation,
            test_indices=None,
            labels_numpy=store.labels,
            batch_function=lambda index, r=required, o=constant_override: data.batch(
                index, r, o
            ),
            protocol=protocol_for("constant_sanity"),
            input_identity=data.input_identity(
                required,
                variant={
                    "kind": "constant_input_negative",
                    "override": "all_required_tensors_exact_zero",
                },
            ),
            normalizer_path=data.normalizer_path,
        )
        constant_metrics = constant["validation_metrics_fixed_threshold"]
        constant = dict(constant)
        constant["source_record_payload_sha256"] = constant.pop(
            "record_payload_sha256"
        )
        constant["sanity_pass"] = (
            abs(constant_metrics["balanced_accuracy"] - 0.5) <= 1e-12
            and constant_metrics["logit_standard_deviation"] <= 1e-6
        )
        constant_rows.append(constant)
        del constant_override
        torch.cuda.empty_cache()
    data.close()
    report = {
        "schema": "vild-sanity-control-report-v1",
        "study_id": STUDY_ID,
        "positive_observable_signal": {
            "rows": positive_rows,
            "passes": sum(row["sanity_pass"] for row in positive_rows),
            "pass": all(row["sanity_pass"] for row in positive_rows),
            "test_partition_used": False,
        },
        "shuffled_label_negative": {
            "rows": shuffled_rows,
            "passes": sum(row["sanity_pass"] for row in shuffled_rows),
            "pass": all(row["sanity_pass"] for row in shuffled_rows),
            "test_partition_used": False,
        },
        "constant_input_negative": {
            "rows": constant_rows,
            "passes": sum(row["sanity_pass"] for row in constant_rows),
            "pass": all(row["sanity_pass"] for row in constant_rows),
            "test_partition_used": False,
        },
    }
    report["pass"] = all(
        report[name]["pass"]
        for name in (
            "positive_observable_signal",
            "shuffled_label_negative",
            "constant_input_negative",
        )
    )
    write_json(OUT / "sanity_control_report.json", report)
    return report


def _run_alternate(store: StudyStore) -> dict[str, Any]:
    records = []
    protocol = protocol_for("alternate")
    for family, spec in zip(EASY_FAMILIES, SPECS, strict=True):
        data = SpecData(store, spec)
        train, validation, test = store.spec_indices(spec)
        for architecture in ARCHITECTURES:
            required = tuple(VERIFIER_CLASSES[architecture]().required_inputs)
            for seed in SEEDS:
                record = _train_with_fallback(
                    model_factory=VERIFIER_CLASSES[architecture],
                    model_name=architecture,
                    group="alternate_schedule",
                    spec=spec,
                    seed=seed,
                    train_indices=train,
                    validation_indices=validation,
                    test_indices=test,
                    labels_numpy=store.labels,
                    batch_function=lambda index, r=required: data.batch(index, r),
                    protocol=protocol,
                    input_identity=data.input_identity(required),
                    normalizer_path=data.normalizer_path,
                )
                records.append(record)
                print(
                    f"ALT family={family} arch={architecture} seed={seed} "
                    f"epochs={record['epochs_run']} "
                    f"test_BA={record['test_metrics_selected_threshold']['balanced_accuracy']:.4f}",
                    flush=True,
                )
        data.close()
    cells = []
    qualifying_architectures = set()
    for architecture in ARCHITECTURES:
        for family, spec in zip(EASY_FAMILIES, SPECS, strict=True):
            selected = [
                row
                for row in records
                if row["model"] == architecture and row["spec"] == spec
            ]
            passing_seeds = [
                row["seed"]
                for row in selected
                if row["test_metrics_selected_threshold"]["balanced_accuracy"]
                >= 0.65
            ]
            cell = {
                "architecture": architecture,
                "family": family,
                "spec": spec,
                "seed_balanced_accuracy": {
                    str(row["seed"]): row["test_metrics_selected_threshold"][
                        "balanced_accuracy"
                    ]
                    for row in selected
                },
                "passing_seeds": passing_seeds,
                "qualifies": len(passing_seeds) >= 2,
            }
            if cell["qualifies"]:
                qualifying_architectures.add(architecture)
            cells.append(cell)
    result = {
        "runs": records,
        "run_count": len(records),
        "cells": cells,
        "qualifying_architectures": sorted(qualifying_architectures),
        "qualifying_architecture_count": len(qualifying_architectures),
        "study_pass": len(qualifying_architectures) >= 2,
        "test_used_for_selection": False,
    }
    return result


def _vector_normalized(
    value: np.ndarray, train: np.ndarray
) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    mean = value[train].mean(0).astype(np.float32)
    std = value[train].std(0).astype(np.float32)
    std[std < 1e-6] = 1.0
    normalized = (value.astype(np.float32) - mean) / std
    if not np.isfinite(normalized).all():
        raise RuntimeError("non-finite diagnostic vector")
    return torch.from_numpy(normalized).to(DEVICE), mean, std


def _run_representation_controls(store: StudyStore) -> dict[str, Any]:
    engineered = np.concatenate(
        (
            store.arrays["current_state"].astype(np.float32),
            store.nuisance["action"].astype(np.float32),
        ),
        axis=1,
    )
    flattened = np.concatenate(
        (
            store.arrays["current_state"].astype(np.float32),
            store.arrays["candidate_action"].reshape(len(store.labels), -1),
        ),
        axis=1,
    )
    records = []
    for family, spec in zip(EASY_FAMILIES, SPECS, strict=True):
        train, validation, test = store.spec_indices(spec)
        for control, value in (
            ("engineered_current_action_summary", engineered),
            ("flattened_raw_current_action", flattened),
        ):
            tensor, mean, std = _vector_normalized(value, train)
            normalizer = (
                OUT / "normalizers" / f"{control}__{spec}.npz"
            )
            np.savez_compressed(normalizer, mean=mean, std=std)
            dimension = int(value.shape[1])
            for seed in SEEDS:
                record = _train_with_fallback(
                    model_factory=lambda d=dimension: VectorDiagnosticMLP(d),
                    model_name=control,
                    group="representation_control",
                    spec=spec,
                    seed=seed,
                    train_indices=train,
                    validation_indices=validation,
                    test_indices=test,
                    labels_numpy=store.labels,
                    batch_function=lambda index, t=tensor: {
                        "vector": t[index]
                    },
                    protocol=protocol_for("vector_control"),
                    input_identity={
                        "schema": "vild-input-identity-v1",
                        "kind": control,
                        "spec": spec,
                        "source_vector_sha256": array_sha256(value),
                        "train_mean_sha256": array_sha256(mean),
                        "train_std_sha256": array_sha256(std),
                    },
                    normalizer_path=normalizer,
                )
                record = dict(record)
                record["source_record_payload_sha256"] = record.pop(
                    "record_payload_sha256"
                )
                record["control"] = control
                record["family"] = family
                records.append(record)
                print(
                    f"REP family={family} control={control} seed={seed} "
                    f"test_BA={record['test_metrics_selected_threshold']['balanced_accuracy']:.4f}",
                    flush=True,
                )
            del tensor
            torch.cuda.empty_cache()
    cells = []
    for control in (
        "engineered_current_action_summary",
        "flattened_raw_current_action",
    ):
        for family, spec in zip(EASY_FAMILIES, SPECS, strict=True):
            selected = [
                row
                for row in records
                if row["control"] == control and row["spec"] == spec
            ]
            passing = [
                row["seed"]
                for row in selected
                if row["test_metrics_selected_threshold"]["balanced_accuracy"]
                >= 0.65
            ]
            cells.append(
                {
                    "control": control,
                    "family": family,
                    "spec": spec,
                    "seed_balanced_accuracy": {
                        str(row["seed"]): row[
                            "test_metrics_selected_threshold"
                        ]["balanced_accuracy"]
                        for row in selected
                    },
                    "passing_seeds": passing,
                    "qualifies": len(passing) >= 2,
                }
            )
    engineered_cells = [
        row
        for row in cells
        if row["control"] == "engineered_current_action_summary"
    ]
    flattened_cells = [
        row
        for row in cells
        if row["control"] == "flattened_raw_current_action"
    ]
    return {
        "runs": records,
        "run_count": len(records),
        "cells": cells,
        "engineered_qualifying_families": [
            row["family"] for row in engineered_cells if row["qualifies"]
        ],
        "engineered_control_support": sum(
            row["qualifies"] for row in engineered_cells
        )
        >= 2,
        "flattened_qualifying_families": [
            row["family"] for row in flattened_cells if row["qualifies"]
        ],
        "test_used_for_selection": False,
        "controls_are_additional_verifier_architectures": False,
    }


def run() -> dict[str, Any]:
    freeze = _verify_frozen_implementation()
    alignment = read_json(OUT / "alignment_audit.json")
    split = read_json(OUT / "normalization_and_split_audit.json")
    if not alignment["pass"] or not split["pass"]:
        raise RuntimeError("phase A must pass before learned diagnostics")
    store = StudyStore()
    memorization = _run_memorization(store)
    sanity = _run_sanity(store)
    gradient = _run_gradient_diagnostics(store)
    alternate = _run_alternate(store)
    representation_controls = _run_representation_controls(store)
    historical = read_json(OUT / "historical_baseline_evidence.json")

    optimization = {
        "schema": "vild-optimization-diagnostics-v1",
        "study_id": STUDY_ID,
        "historical_runs": historical["historical_runs"],
        "historical_architecture_summary": historical["architecture_summary"],
        "historical_temperature_upper_bound": {
            "value": historical["temperature_search_upper_bound"],
            "runs_at_bound": historical["runs_at_temperature_upper_bound"],
            "runs": len(historical["historical_runs"]),
        },
        "bfloat16_quantization_by_family": [
            {
                "family": row["family"],
                **row["bfloat16_quantization"],
            }
            for row in historical["family_evidence"]
        ],
        "alternate_schedule": alternate,
        "gradient_flow_pass": gradient["pass"],
        "sanity_controls_pass": sanity["pass"],
        "test_used_for_selection": False,
    }
    write_json(OUT / "optimization_diagnostics.json", optimization)

    representation = {
        "schema": "vild-representation-mismatch-report-v1",
        "study_id": STUDY_ID,
        "input_information_equivalence": historical[
            "input_information_equivalence"
        ],
        "family_pair_amplification": [
            {
                "family": row["family"],
                "normalized_pair_RMSE": row["normalized_pair_RMSE"],
                "top_train_selected_univariate_action_features": row[
                    "top_train_selected_univariate_action_features"
                ],
                "historical_controls": row["historical_controls"],
            }
            for row in historical["family_evidence"]
        ],
        "exact_input_collision_audit": alignment[
            "required_input_collision_audit"
        ],
        "minimum_exact_input_ceiling": min(
            row["empirical_exact_input_accuracy_and_BA_ceiling"]
            for row in alignment["required_input_collision_audit"]
        ),
        "diagnostic_controls": representation_controls,
        "engineered_control_support": representation_controls[
            "engineered_control_support"
        ],
        "causal_root_cause_decision_deferred_until_final_confirmation": True,
        "additional_verifier_architectures": 0,
    }
    write_json(OUT / "representation_mismatch_report.json", representation)
    events = {row["event"] for row in read_jsonl(OUT / "run_log.jsonl")}
    if "B_E_LEARNED_DIAGNOSTICS_COMPLETE" not in events:
        append_log(
            "B_E_LEARNED_DIAGNOSTICS_COMPLETE",
            gradient_runs=gradient["runs"],
            gradient_pass=gradient["pass"],
            tiny_memorization_runs=memorization["tiny_runs"],
            tiny_memorization_passes=memorization["tiny_passes"],
            full_memorization_runs=memorization["full_runs"],
            full_memorization_passes=memorization["full_passes"],
            sanity_runs=15,
            sanity_pass=sanity["pass"],
            alternate_runs=alternate["run_count"],
            alternate_study_pass=alternate["study_pass"],
            representation_control_runs=representation_controls["run_count"],
            engineered_control_support=representation_controls[
                "engineered_control_support"
            ],
            GPU_hours=resource_status()["GPU_hours"],
            memorization_sha256=sha256_file(OUT / "memorization_report.json"),
            sanity_sha256=sha256_file(OUT / "sanity_control_report.json"),
            gradient_sha256=sha256_file(OUT / "gradient_flow_report.json"),
            optimization_sha256=sha256_file(
                OUT / "optimization_diagnostics.json"
            ),
            representation_sha256=sha256_file(
                OUT / "representation_mismatch_report.json"
            ),
        )
    return {
        "implementation_freeze": freeze,
        "gradient_pass": gradient["pass"],
        "tiny_memorization": f"{memorization['tiny_passes']}/{memorization['tiny_runs']}",
        "full_memorization": f"{memorization['full_passes']}/{memorization['full_runs']}",
        "sanity_pass": sanity["pass"],
        "alternate_study_pass": alternate["study_pass"],
        "alternate_qualifying_architectures": alternate[
            "qualifying_architectures"
        ],
        "engineered_control_support": representation_controls[
            "engineered_control_support"
        ],
        "resource_status": resource_status(),
    }


def main() -> None:
    print(json.dumps(run(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
