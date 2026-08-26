"""One-time repair: preserve validation threshold/temperature as float64 in raw NPZ."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .common import OUT, append_log, read_json, sha256_file, write_json
from .train_core import output_paths


def _array_hash(value: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(value).tobytes()).hexdigest()


def _rewrite(
    path: Path, threshold: float, temperature: float
) -> dict[str, Any]:
    old_hash = sha256_file(path)
    with np.load(path) as stored:
        payload = {name: stored[name].copy() for name in stored.files}
    probability_name = (
        "test_calibrated_probability"
        if "test_calibrated_probability" in payload
        else "calibrated_probability"
    )
    logit_name = "test_logit" if "test_logit" in payload else "logit"
    probability_hash = _array_hash(payload[probability_name])
    logit_hash = _array_hash(payload[logit_name])
    old_threshold = float(payload["threshold"][0])
    old_temperature = float(payload["temperature"][0])
    payload["threshold"] = np.asarray([threshold], dtype=np.float64)
    payload["temperature"] = np.asarray([temperature], dtype=np.float64)
    np.savez_compressed(path, **payload)
    with np.load(path) as stored:
        if _array_hash(stored[probability_name]) != probability_hash:
            raise RuntimeError(f"probability changed during precision repair: {path}")
        if _array_hash(stored[logit_name]) != logit_hash:
            raise RuntimeError(f"logit changed during precision repair: {path}")
        if stored["threshold"].dtype != np.float64:
            raise RuntimeError(f"threshold precision repair failed: {path}")
        if stored["temperature"].dtype != np.float64:
            raise RuntimeError(f"temperature precision repair failed: {path}")
    return {
        "path": str(path.resolve()),
        "old_sha256": old_hash,
        "new_sha256": sha256_file(path),
        "old_threshold_float32": old_threshold,
        "new_threshold_float64": threshold,
        "absolute_threshold_delta": abs(threshold - old_threshold),
        "old_temperature_float32": old_temperature,
        "new_temperature_float64": temperature,
        "absolute_temperature_delta": abs(temperature - old_temperature),
        "logit_sha256_unchanged": logit_hash,
        "calibrated_probability_sha256_unchanged": probability_hash,
    }


def _update_run_records(
    category: str,
    rows: list[dict[str, Any]],
    new_hashes: dict[str, str],
) -> int:
    seen: set[tuple[str, str, int]] = set()
    updated = 0
    for row in rows:
        model_key = "architecture" if category == "verifier" else "control"
        key = (row[model_key], row["training_spec"], int(row["seed"]))
        if key in seen:
            continue
        seen.add(key)
        paths = output_paths(category, key[0], key[1], key[2])
        record_path = paths["record"]
        if not record_path.is_file():
            continue
        record = read_json(record_path)
        prediction_path = str(paths["prediction"].resolve())
        if prediction_path in new_hashes:
            record["artifact_hashes"]["prediction"] = new_hashes[prediction_path]
            write_json(record_path, record)
            updated += 1
    return updated


def repair() -> dict[str, Any]:
    verifier_path = OUT / "verifier_per_environment.json"
    control_path = OUT / "shortcut_control_per_environment.json"
    verifier = read_json(verifier_path)
    controls = read_json(control_path)
    records = []
    cache: dict[str, dict[str, Any]] = {}
    checkpoint_values: dict[str, tuple[float, float, str]] = {}

    def process(row: dict[str, Any], category: str, model_key: str) -> None:
        raw_path = Path(row["raw_prediction_path"])
        key = str(raw_path.resolve())
        checkpoint_path = output_paths(
            category,
            row[model_key],
            row["training_spec"],
            int(row["seed"]),
        )["checkpoint"]
        checkpoint_key = str(checkpoint_path.resolve())
        if checkpoint_key not in checkpoint_values:
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
            checkpoint_values[checkpoint_key] = (
                float(checkpoint["threshold"]),
                float(checkpoint["temperature"]),
                sha256_file(checkpoint_path),
            )
            del checkpoint
        threshold, temperature, checkpoint_sha256 = checkpoint_values[
            checkpoint_key
        ]
        if key not in cache:
            cache[key] = _rewrite(
                raw_path,
                threshold,
                temperature,
            )
            records.append(
                {
                    **cache[key],
                    "category": category,
                    "model": row[model_key],
                    "training_spec": row["training_spec"],
                    "seed": int(row["seed"]),
                    "checkpoint_path": str(checkpoint_path.resolve()),
                    "checkpoint_sha256": checkpoint_sha256,
                }
            )
        row["raw_prediction_sha256"] = cache[key]["new_sha256"]

    for row in verifier["rows"]:
        process(row, "verifier", "architecture")
    for row in controls["rows"]:
        process(row, "shortcut_control", "control")
    write_json(verifier_path, verifier)
    write_json(control_path, controls)
    new_hashes = {
        record["path"]: record["new_sha256"] for record in records
    }
    per_seed_path = OUT / "verifier_per_seed.json"
    per_seed = read_json(per_seed_path)
    per_seed_updates = 0
    for row in per_seed["training_runs"]:
        prediction_path = row["artifact_paths"]["prediction"]
        if prediction_path in new_hashes:
            row["artifact_hashes"]["prediction"] = new_hashes[prediction_path]
            per_seed_updates += 1
    write_json(per_seed_path, per_seed)
    verifier_record_updates = _update_run_records(
        "verifier", verifier["rows"], new_hashes
    )
    control_record_updates = _update_run_records(
        "shortcut_control", controls["rows"], new_hashes
    )
    audit = {
        "schema": "vsa-prediction-serialization-precision-repair-v1",
        "confirmed_bug": (
            "validation-selected float64 threshold and temperature were serialized "
            "to raw NPZ as float32, changing thresholded decisions for near-boundary "
            "samples during independent recomputation"
        ),
        "repair_scope": (
            "serialization precision only; checkpoints, logits, calibrated "
            "probabilities, labels, samples, splits, and model weights unchanged"
        ),
        "files_repaired": len(records),
        "verifier_files": sum(
            record["category"] == "verifier" for record in records
        ),
        "shortcut_control_files": sum(
            record["category"] == "shortcut_control" for record in records
        ),
        "verifier_per_seed_records_updated": per_seed_updates,
        "verifier_run_records_updated": verifier_record_updates,
        "shortcut_control_run_records_updated": control_record_updates,
        "maximum_absolute_threshold_delta": max(
            record["absolute_threshold_delta"] for record in records
        ),
        "maximum_absolute_temperature_delta": max(
            record["absolute_temperature_delta"] for record in records
        ),
        "all_logits_unchanged": True,
        "all_calibrated_probabilities_unchanged": True,
        "model_retraining_performed": False,
        "records": records,
    }
    write_json(OUT / "prediction_serialization_precision_repair.json", audit)
    append_log(
        "CONFIRMED_PREDICTION_SERIALIZATION_PRECISION_REPAIR",
        files_repaired=len(records),
        verifier_files=audit["verifier_files"],
        shortcut_control_files=audit["shortcut_control_files"],
        model_retraining_performed=False,
        logits_unchanged=True,
        calibrated_probabilities_unchanged=True,
        repair_audit_sha256=sha256_file(
            OUT / "prediction_serialization_precision_repair.json"
        ),
    )
    return {
        "files_repaired": len(records),
        "verifier_files": audit["verifier_files"],
        "shortcut_control_files": audit["shortcut_control_files"],
        "maximum_absolute_threshold_delta": audit[
            "maximum_absolute_threshold_delta"
        ],
        "model_retraining_performed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    print(json.dumps(repair(), indent=2))


if __name__ == "__main__":
    main()
