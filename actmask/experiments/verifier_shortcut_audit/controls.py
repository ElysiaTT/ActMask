"""V6 train and evaluate the six preregistered shortcut controls."""
from __future__ import annotations

import argparse
import gc
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .common import OUT, append_log, read_json, sha256_file, write_json
from .metrics import binary_metrics, sigmoid
from .train_core import (
    CONTROL_NAMES,
    DEVICE,
    AuditStore,
    ControlSpecData,
    _predict,
    output_paths,
    seed_everything,
    train_one,
)


SEED = 17


def _config(store: AuditStore) -> dict[str, Any]:
    return {
        "schema": "vsa-shortcut-control-config-v1",
        "study_id": "VERIFIER-SHORTCUT-AUDIT-V1",
        "controls": {
            "logistic_regression_combined_nuisance": {
                "architecture": "single affine logit",
                "inputs": "action + context + provenance + core language nuisance",
            },
            "shallow_mlp_combined_nuisance": {
                "architecture": "two-layer 128/64 GELU MLP",
                "inputs": "action + context + provenance + core language nuisance",
            },
            "temporal_action_only_TCN": {
                "architecture": "two-layer temporal convolution",
                "inputs": "full candidate action chunk only",
            },
            "current_state_action_summary_MLP": {
                "architecture": "two-layer 128/64 GELU MLP",
                "inputs": "current state + all frozen action-summary features",
            },
            "provenance_only_classifier": {
                "architecture": "single affine logit",
                "inputs": "frozen provenance nuisance features only",
            },
            "combined_non_relational_nuisance_classifier": {
                "architecture": "independent group MLP logits summed without cross-group interaction",
                "inputs": "action, context, provenance, core language groups",
            },
        },
        "core_language_excludes": [
            "precomputed shuffled-language condition",
            "precomputed empty-language condition",
            "precomputed task-ID-only condition",
        ],
        "reason_for_exclusion": (
            "counterfactual evaluation alternatives are not simultaneously observable "
            "training inputs"
        ),
        "seed": SEED,
        "training_specifications": sorted(
            store.environment_manifest["training_specs"]
        ),
        "training_protocol": {
            "optimizer": "AdamW",
            "learning_rate": 0.002,
            "weight_decay": 0.0003,
            "batch_size": 512,
            "epochs_max": 40,
            "early_stop_patience": 6,
            "gradient_clip": 2.0,
            "selection": "validation balanced accuracy",
            "calibration": "validation-only scalar temperature",
            "threshold": "validation-only balanced-accuracy optimum",
            "normalization": "train partition only per training specification",
        },
        "audit_only": True,
        "physical_ground_truth": False,
        "environment_manifest_sha256": sha256_file(
            OUT / "audit_environment_manifest.json"
        ),
        "nuisance_schema_sha256": sha256_file(OUT / "nuisance_feature_schema.json"),
    }


def _raw_prediction_path(control: str, name: str) -> Path:
    return (
        OUT
        / "shortcut_control_raw_predictions"
        / control
        / "pooled_n1_n6"
        / "seed_17_evaluations"
        / f"{name}.npz"
    )


def _save_evaluation(
    *,
    path: Path,
    indices: np.ndarray,
    labels: np.ndarray,
    logits: np.ndarray,
    temperature: float,
    threshold: float,
    environment: str,
    condition: str,
    weights: np.ndarray | None = None,
) -> dict[str, Any]:
    calibrated = sigmoid(logits / temperature).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "index": indices.astype(np.int32),
        "label": labels[indices].astype(np.int8),
        "logit": logits.astype(np.float32),
        "raw_probability": sigmoid(logits).astype(np.float32),
        "calibrated_probability": calibrated,
        "temperature": np.asarray([temperature], dtype=np.float32),
        "threshold": np.asarray([threshold], dtype=np.float32),
    }
    if weights is not None:
        payload["weight"] = weights[indices].astype(np.float32)
    np.savez_compressed(path, **payload)
    metrics = binary_metrics(
        calibrated,
        labels[indices].astype(np.int8),
        threshold,
        None if weights is None else weights[indices],
    )
    return {
        "environment": environment,
        "condition": condition,
        "samples": int(len(indices)),
        "metrics": metrics,
        "raw_prediction_path": str(path.resolve()),
        "raw_prediction_sha256": sha256_file(path),
    }


def _language_values(
    store: AuditStore, data: ControlSpecData, condition: str
) -> dict[str, torch.Tensor]:
    if condition == "correct":
        return data.values
    raw = store.nuisance["language"][:, data.language_positions].astype(np.float32)
    names = [
        store.schema["groups"]["language"][int(index)]
        for index in data.language_positions
    ]
    task_positions = [
        index for index, name in enumerate(names) if name.startswith("task_string_identity[")
    ]
    embedding_positions = [
        index for index, name in enumerate(names) if name.startswith("language_embedding[")
    ]
    length_positions = [
        index
        for index, name in enumerate(names)
        if name in ("instruction_character_length", "instruction_token_length")
    ]
    modified = raw.copy()
    if condition == "shuffled":
        modified[:, embedding_positions] = store.variants["language_shuffled"]
    elif condition == "empty":
        modified[:] = 0.0
    elif condition == "task_id_only":
        modified[:] = 0.0
        modified[:, task_positions] = raw[:, task_positions]
        modified[:, embedding_positions] = store.variants["language_task_id_only"]
        modified[:, length_positions] = 0.0
    else:
        raise KeyError(condition)
    values = {key: value for key, value in data.values.items() if key != "combined"}
    values["language_core"] = data.normalized_group("language_core", modified)
    return values


def _visual_values(
    store: AuditStore, data: ControlSpecData, condition: str
) -> dict[str, torch.Tensor]:
    if condition == "full":
        return data.values
    raw = store.nuisance["context"].astype(np.float32).copy()
    names = store.schema["groups"]["context"]
    visual_prefixes = (
        "current_frame_frozen_visual[",
        "history_visual_difference[",
        "current_rgb_mean[",
        "current_rgb_std[",
        "current_rgb_histogram[",
        "static_border_mean[",
        "static_border_std[",
        "center_workspace_mean[",
        "center_workspace_std[",
        "history_frame_difference_",
    )
    visual_positions = [
        index for index, name in enumerate(names) if name.startswith(visual_prefixes)
    ]
    keep_prefixes = {
        "current_only": (
            "current_frame_frozen_visual[",
            "current_rgb_mean[",
            "current_rgb_std[",
            "current_rgb_histogram[",
            "static_border_mean[",
            "static_border_std[",
            "center_workspace_mean[",
            "center_workspace_std[",
        ),
        "difference_only": (
            "history_visual_difference[",
            "history_frame_difference_",
        ),
        "static_border_only": (
            "static_border_mean[",
            "static_border_std[",
        ),
        "center_workspace_only": (
            "center_workspace_mean[",
            "center_workspace_std[",
        ),
        "global_statistics_only": (
            "current_rgb_mean[",
            "current_rgb_std[",
            "current_rgb_histogram[",
        ),
    }[condition]
    keep = {
        index
        for index, name in enumerate(names)
        if name.startswith(keep_prefixes)
    }
    zero = [index for index in visual_positions if index not in keep]
    raw[:, zero] = 0.0
    values = {key: value for key, value in data.values.items() if key != "combined"}
    values["context"] = data.normalized_group("context", raw)
    return values


def _evaluate_pooled(
    store: AuditStore, records: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    labels = store.labels
    environments = store.environment_manifest["environments"]
    evaluation_rows: list[dict[str, Any]] = []
    with np.load(OUT / "audit_environment_weights.npz") as stored_weights:
        weight_arrays = {name: stored_weights[name].copy() for name in stored_weights.files}
    for control in CONTROL_NAMES:
        data = ControlSpecData(store, "pooled_n1_n6")
        seed_everything(SEED)
        model = data.create_model(control).to(DEVICE)
        checkpoint_path = output_paths(
            "shortcut_control", control, "pooled_n1_n6", SEED
        )["checkpoint"]
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
        temperature = float(checkpoint["temperature"])
        threshold = float(checkpoint["threshold"])

        def evaluate(
            environment: str,
            condition: str,
            key: str,
            values: dict[str, torch.Tensor] | None = None,
            weights: np.ndarray | None = None,
        ) -> None:
            indices = store.indices[key].astype(np.int64)
            active_values = data.values if values is None else values
            logits, _ = _predict(
                model,
                indices,
                lambda index: data.batch_with_values(index, control, active_values),
            )
            row = _save_evaluation(
                path=_raw_prediction_path(
                    control,
                    f"{environment.lower()}__{condition}",
                ),
                indices=indices,
                labels=labels,
                logits=logits,
                temperature=temperature,
                threshold=threshold,
                environment=environment,
                condition=condition,
                weights=weights,
            )
            row.update(
                {
                    "control": control,
                    "training_spec": "pooled_n1_n6",
                    "seed": SEED,
                }
            )
            evaluation_rows.append(row)

        evaluate("E5", "all", environments["E5"]["all_index_key"])
        for family, key in environments["E5"]["family_index_keys"].items():
            evaluate("E5", family.split("_", 1)[0], key)
        evaluate("E6", "all", environments["E6"]["all_index_key"])
        for family, key in environments["E6"]["family_index_keys"].items():
            evaluate("E6", family.split("_", 1)[0], key)
        evaluate("E7", "source_heldout", environments["E7"]["partition_keys"]["test"])
        for condition, key in environments["E9"]["index_keys"].items():
            evaluate("E9", condition, key)
        for condition in ("correct", "shuffled", "empty", "task_id_only"):
            values = _language_values(store, data, condition)
            evaluate("E10", condition, environments["E10"]["index_key"], values)
            if values is not data.values:
                del values
                torch.cuda.empty_cache()
        for condition in (
            "full",
            "current_only",
            "difference_only",
            "static_border_only",
            "center_workspace_only",
            "global_statistics_only",
        ):
            values = _visual_values(store, data, condition)
            evaluate("E11", condition, environments["E11"]["index_key"], values)
            if values is not data.values:
                del values
                torch.cuda.empty_cache()
        for condition, weights in weight_arrays.items():
            evaluate(
                "E12",
                condition,
                environments["E12"]["index_key"],
                weights=weights,
            )
        del model, checkpoint
        data.close()
    return evaluation_rows


def _summaries(
    training_records: list[dict[str, Any]],
    evaluation_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    all_rows = []
    for record in training_records:
        all_rows.append(
            {
                "control": record["model"],
                "training_spec": record["spec"],
                "seed": record["seed"],
                "environment": read_json(
                    OUT / "audit_environment_manifest.json"
                )["training_specs"][record["spec"]]["environment"],
                "condition": "default_test",
                "samples": record["test_samples"],
                "metrics": record["test_metrics"],
                "raw_prediction_path": record["artifact_paths"]["prediction"],
                "raw_prediction_sha256": record["artifact_hashes"]["prediction"],
            }
        )
    all_rows.extend(evaluation_rows)
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in all_rows:
        grouped[(row["control"], row["environment"])].append(
            row["metrics"]["balanced_accuracy"]
        )
    summary_rows = [
        {
            "control": control,
            "environment": environment,
            "evaluations": len(values),
            "balanced_accuracy_mean": float(np.mean(values)),
            "balanced_accuracy_minimum": float(np.min(values)),
            "balanced_accuracy_maximum": float(np.max(values)),
        }
        for (control, environment), values in sorted(grouped.items())
    ]
    total_gpu_seconds = float(sum(row["GPU_seconds"] for row in training_records))
    summary = {
        "schema": "vsa-shortcut-control-summary-v1",
        "complete": len(training_records) == len(CONTROL_NAMES) * 19,
        "controls": len(CONTROL_NAMES),
        "training_specifications": 19,
        "training_runs": len(training_records),
        "seeds": [SEED],
        "total_GPU_seconds": total_gpu_seconds,
        "total_GPU_hours": total_gpu_seconds / 3600.0,
        "summary_rows": summary_rows,
        "audit_only": True,
        "physical_ground_truth": False,
    }
    per_environment = {
        "schema": "vsa-shortcut-control-per-environment-v1",
        "rows": all_rows,
        "raw_scores_saved_for_all_environments": all(
            any(row["environment"] == f"E{number}" for row in all_rows)
            for number in range(1, 13)
        ),
    }
    return summary, per_environment


def run(
    selected_specs: set[str] | None = None,
    selected_models: set[str] | None = None,
    force: bool = False,
    finalize: bool = True,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("V6 requires the confirmed CUDA device")
    store = AuditStore()
    config = _config(store)
    write_json(OUT / "shortcut_control_config.json", config)
    all_specs = sorted(store.environment_manifest["training_specs"])
    specs = [
        spec for spec in all_specs if selected_specs is None or spec in selected_specs
    ]
    models = [
        model
        for model in CONTROL_NAMES
        if selected_models is None or model in selected_models
    ]
    started = time.monotonic()
    completed = 0
    for spec_number, spec in enumerate(specs, start=1):
        data = ControlSpecData(store, spec)
        train_indices, validation_indices, test_indices = store.spec_indices(spec)
        for model_number, control in enumerate(models, start=1):
            seed_everything(SEED)
            model = data.create_model(control)
            result = train_one(
                model=model,
                model_name=control,
                category="shortcut_control",
                spec=spec,
                seed=SEED,
                train_indices=train_indices,
                validation_indices=validation_indices,
                test_indices=test_indices,
                labels_tensor=data.labels,
                labels_numpy=store.labels,
                batch_function=lambda index, c=control: data.batch(index, c),
                normalizer_path=data.normalizer_path,
                output_paths=output_paths(
                    "shortcut_control", control, spec, SEED
                ),
                force=force,
            )
            completed += 1
            print(
                "V6 "
                f"spec={spec_number}/{len(specs)} model={model_number}/{len(models)} "
                f"name={control} epochs={result.metadata['epochs_run']} "
                f"test_BA={result.metadata['test_metrics']['balanced_accuracy']:.4f} "
                f"gpu_s={result.metadata['GPU_seconds']:.2f}",
                flush=True,
            )
        data.close()
    if not finalize or selected_specs is not None or selected_models is not None:
        return {
            "partial": True,
            "runs_completed_or_verified": completed,
            "wall_seconds": time.monotonic() - started,
        }
    training_records = []
    for spec in all_specs:
        for control in CONTROL_NAMES:
            path = output_paths(
                "shortcut_control", control, spec, SEED
            )["record"]
            if not path.is_file():
                raise RuntimeError(f"missing completed V6 run {path}")
            training_records.append(read_json(path))
    evaluation_rows = _evaluate_pooled(store, training_records)
    summary, per_environment = _summaries(training_records, evaluation_rows)
    write_json(OUT / "shortcut_control_summary.json", summary)
    write_json(OUT / "shortcut_control_per_environment.json", per_environment)
    append_log(
        "V6_SHORTCUT_CONTROLS_COMPLETE",
        controls=len(CONTROL_NAMES),
        training_runs=len(training_records),
        evaluation_rows=len(per_environment["rows"]),
        raw_scores_saved_for_all_environments=per_environment[
            "raw_scores_saved_for_all_environments"
        ],
        GPU_hours=summary["total_GPU_hours"],
        summary_sha256=sha256_file(OUT / "shortcut_control_summary.json"),
    )
    return {
        "partial": False,
        "training_runs": len(training_records),
        "evaluation_rows": len(per_environment["rows"]),
        "GPU_hours": summary["total_GPU_hours"],
        "raw_scores_saved_for_all_environments": per_environment[
            "raw_scores_saved_for_all_environments"
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", action="append")
    parser.add_argument("--model", action="append", choices=CONTROL_NAMES)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-finalize", action="store_true")
    arguments = parser.parse_args()
    print(
        json.dumps(
            run(
                selected_specs=set(arguments.spec) if arguments.spec else None,
                selected_models=set(arguments.model) if arguments.model else None,
                force=arguments.force,
                finalize=not arguments.no_finalize,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
