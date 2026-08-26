"""V7 train and evaluate the five preregistered fair verifiers."""
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

from .common import OUT, SEEDS, append_log, read_json, sha256_file, write_json
from .metrics import binary_metrics, sigmoid
from .modeling import VERIFIER_CLASSES, model_description
from .train_core import (
    DEVICE,
    AuditStore,
    FairSpecData,
    _predict,
    output_paths,
    seed_everything,
    train_one,
)


ARCHITECTURES = tuple(VERIFIER_CLASSES)


def _config(store: AuditStore) -> dict[str, Any]:
    descriptions = {}
    for seed_offset, (name, model_class) in enumerate(VERIFIER_CLASSES.items()):
        seed_everything(1700 + seed_offset)
        model = model_class()
        descriptions[name] = {
            **model_description(model),
            "required_inputs": list(model.required_inputs),
            "forbidden_inputs": [
                "family identity",
                "source episode",
                "source path",
                "progress scalar",
                "matching distance",
                "mining rank",
                "reason code",
                "evidence type",
                "future state",
                "future RGB",
            ],
        }
    return {
        "schema": "vsa-verifier-config-v1",
        "study_id": "VERIFIER-SHORTCUT-AUDIT-V1",
        "learned_architectures": descriptions,
        "learned_architecture_count": len(descriptions),
        "maximum_learned_architectures": 5,
        "seeds": list(SEEDS),
        "training_specifications": sorted(
            store.environment_manifest["training_specs"]
        ),
        "identical_training_protocol": {
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
            "mixed_precision": "bfloat16 CUDA autocast",
        },
        "candidate_groups_identical": True,
        "historical_action_tensors_modified": False,
        "historical_labels_modified": False,
        "future_information_used": False,
        "physical_ground_truth": False,
        "environment_manifest_sha256": sha256_file(
            OUT / "audit_environment_manifest.json"
        ),
        "unified_arrays_sha256": sha256_file(OUT / "unified_audit_arrays.npz"),
    }


def _raw_prediction_path(architecture: str, seed: int, name: str) -> Path:
    return (
        OUT
        / "raw_verifier_predictions"
        / architecture
        / "pooled_n1_n6"
        / f"seed_{seed}_evaluations"
        / f"{name}.npz"
    )


def _representation_path(architecture: str, seed: int, name: str) -> Path:
    return (
        OUT
        / "verifier_representations"
        / architecture
        / "pooled_n1_n6"
        / f"seed_{seed}_evaluations"
        / f"{name}.npz"
    )


def _save_evaluation(
    *,
    prediction_path: Path,
    representation_path: Path,
    indices: np.ndarray,
    labels: np.ndarray,
    logits: np.ndarray,
    representation: np.ndarray,
    temperature: float,
    threshold: float,
    environment: str,
    condition: str,
    weights: np.ndarray | None = None,
) -> dict[str, Any]:
    calibrated = sigmoid(logits / temperature).astype(np.float32)
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    representation_path.parent.mkdir(parents=True, exist_ok=True)
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
    np.savez_compressed(prediction_path, **payload)
    np.savez_compressed(
        representation_path,
        index=indices.astype(np.int32),
        representation=representation.astype(np.float16),
    )
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
        "raw_prediction_path": str(prediction_path.resolve()),
        "raw_prediction_sha256": sha256_file(prediction_path),
        "representation_path": str(representation_path.resolve()),
        "representation_sha256": sha256_file(representation_path),
    }


def _evaluate_pooled(store: AuditStore) -> list[dict[str, Any]]:
    labels = store.labels
    environments = store.environment_manifest["environments"]
    evaluation_rows: list[dict[str, Any]] = []
    with np.load(OUT / "audit_environment_weights.npz") as stored_weights:
        weight_arrays = {name: stored_weights[name].copy() for name in stored_weights.files}
    for architecture in ARCHITECTURES:
        for seed in SEEDS:
            data = FairSpecData(store, "pooled_n1_n6")
            seed_everything(seed)
            model = VERIFIER_CLASSES[architecture]().to(DEVICE)
            checkpoint_path = output_paths(
                "verifier", architecture, "pooled_n1_n6", seed
            )["checkpoint"]
            checkpoint = torch.load(
                checkpoint_path, map_location=DEVICE, weights_only=False
            )
            model.load_state_dict(checkpoint["state_dict"])
            temperature = float(checkpoint["temperature"])
            threshold = float(checkpoint["threshold"])
            required = tuple(model.required_inputs)

            def evaluate(
                environment: str,
                condition: str,
                key: str,
                overrides: dict[str, torch.Tensor] | None = None,
                weights: np.ndarray | None = None,
            ) -> None:
                indices = store.indices[key].astype(np.int64)
                logits, representation = _predict(
                    model,
                    indices,
                    lambda index: data.batch(index, required, overrides),
                )
                stem = f"{environment.lower()}__{condition}"
                row = _save_evaluation(
                    prediction_path=_raw_prediction_path(
                        architecture, seed, stem
                    ),
                    representation_path=_representation_path(
                        architecture, seed, stem
                    ),
                    indices=indices,
                    labels=labels,
                    logits=logits,
                    representation=representation,
                    temperature=temperature,
                    threshold=threshold,
                    environment=environment,
                    condition=condition,
                    weights=weights,
                )
                row.update(
                    {
                        "architecture": architecture,
                        "training_spec": "pooled_n1_n6",
                        "seed": seed,
                    }
                )
                evaluation_rows.append(row)

            evaluate("E5", "all", environments["E5"]["all_index_key"])
            for family, key in environments["E5"]["family_index_keys"].items():
                evaluate("E5", family.split("_", 1)[0], key)
            evaluate("E6", "all", environments["E6"]["all_index_key"])
            for family, key in environments["E6"]["family_index_keys"].items():
                evaluate("E6", family.split("_", 1)[0], key)
            evaluate(
                "E7",
                "source_heldout",
                environments["E7"]["partition_keys"]["test"],
            )
            for condition, key in environments["E9"]["index_keys"].items():
                evaluate("E9", condition, key)
            language_variants = {
                "correct": "language_correct",
                "shuffled": "language_shuffled",
                "empty": "language_empty",
                "task_id_only": "language_task_id_only",
            }
            for condition, variant in language_variants.items():
                overrides = {
                    "language_embedding": data.normalized_override(
                        "language_embedding", store.variants[variant]
                    )
                }
                evaluate(
                    "E10",
                    condition,
                    environments["E10"]["index_key"],
                    overrides,
                )
                del overrides
                torch.cuda.empty_cache()
            visual_variants = {
                "full": "visual_full",
                "current_only": "visual_current_only",
                "difference_only": "visual_history_difference_only",
                "static_border_only": "visual_static_border_only",
                "center_workspace_only": "visual_center_workspace_only",
                "global_statistics_only": "visual_global_statistics_only",
            }
            for condition, variant in visual_variants.items():
                overrides = {
                    "visual_history": data.normalized_override(
                        "visual_history", store.variants[variant]
                    )
                }
                evaluate(
                    "E11",
                    condition,
                    environments["E11"]["index_key"],
                    overrides,
                )
                del overrides
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
    store: AuditStore,
    training_records: list[dict[str, Any]],
    evaluation_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    all_rows = []
    for record in training_records:
        all_rows.append(
            {
                "architecture": record["model"],
                "training_spec": record["spec"],
                "seed": record["seed"],
                "environment": store.environment_manifest["training_specs"][
                    record["spec"]
                ]["environment"],
                "condition": "default_test",
                "samples": record["test_samples"],
                "metrics": record["test_metrics"],
                "raw_prediction_path": record["artifact_paths"]["prediction"],
                "raw_prediction_sha256": record["artifact_hashes"]["prediction"],
                "representation_path": record["artifact_paths"]["representation"],
                "representation_sha256": record["artifact_hashes"]["representation"],
            }
        )
    all_rows.extend(evaluation_rows)
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    seed_grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in all_rows:
        grouped[(row["architecture"], row["environment"])].append(
            row["metrics"]["balanced_accuracy"]
        )
        seed_grouped[(row["architecture"], row["seed"])].append(
            row["metrics"]["balanced_accuracy"]
        )
    environment_summary = [
        {
            "architecture": architecture,
            "environment": environment,
            "evaluations": len(values),
            "balanced_accuracy_mean": float(np.mean(values)),
            "balanced_accuracy_standard_deviation": float(np.std(values)),
            "balanced_accuracy_minimum": float(np.min(values)),
            "balanced_accuracy_maximum": float(np.max(values)),
        }
        for (architecture, environment), values in sorted(grouped.items())
    ]
    seed_summary = [
        {
            "architecture": architecture,
            "seed": seed,
            "evaluations": len(values),
            "balanced_accuracy_mean": float(np.mean(values)),
            "balanced_accuracy_standard_deviation": float(np.std(values)),
        }
        for (architecture, seed), values in sorted(seed_grouped.items())
    ]
    total_gpu_seconds = float(sum(row["GPU_seconds"] for row in training_records))
    summary = {
        "schema": "vsa-verifier-summary-v1",
        "complete": len(training_records) == len(ARCHITECTURES) * 19 * len(SEEDS),
        "learned_architectures": len(ARCHITECTURES),
        "maximum_learned_architectures": 5,
        "training_specifications": 19,
        "seeds": list(SEEDS),
        "training_runs": len(training_records),
        "evaluation_rows": len(all_rows),
        "total_GPU_seconds": total_gpu_seconds,
        "total_GPU_hours": total_gpu_seconds / 3600.0,
        "environment_summary": environment_summary,
        "historical_action_tensors_modified": False,
        "historical_labels_modified": False,
        "future_information_used": False,
        "physical_ground_truth": False,
    }
    per_seed = {
        "schema": "vsa-verifier-per-seed-v1",
        "summary": seed_summary,
        "training_runs": training_records,
    }
    per_environment = {
        "schema": "vsa-verifier-per-environment-v1",
        "summary": environment_summary,
        "rows": all_rows,
        "raw_scores_saved_for_all_environments": all(
            any(row["environment"] == f"E{number}" for row in all_rows)
            for number in range(1, 13)
        ),
    }
    return summary, per_seed, per_environment


def run(
    selected_specs: set[str] | None = None,
    selected_architectures: set[str] | None = None,
    selected_seeds: set[int] | None = None,
    force: bool = False,
    finalize: bool = True,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("V7 requires the confirmed CUDA device")
    store = AuditStore()
    config = _config(store)
    if config["learned_architecture_count"] > 5:
        raise RuntimeError("preregistered five-architecture maximum exceeded")
    write_json(OUT / "verifier_config.json", config)
    all_specs = sorted(store.environment_manifest["training_specs"])
    specs = [
        spec for spec in all_specs if selected_specs is None or spec in selected_specs
    ]
    architectures = [
        architecture
        for architecture in ARCHITECTURES
        if selected_architectures is None or architecture in selected_architectures
    ]
    seeds = [seed for seed in SEEDS if selected_seeds is None or seed in selected_seeds]
    started = time.monotonic()
    completed = 0
    for spec_number, spec in enumerate(specs, start=1):
        data = FairSpecData(store, spec)
        train_indices, validation_indices, test_indices = store.spec_indices(spec)
        for architecture_number, architecture in enumerate(architectures, start=1):
            for seed_number, seed in enumerate(seeds, start=1):
                seed_everything(seed)
                model = VERIFIER_CLASSES[architecture]()
                required = tuple(model.required_inputs)
                result = train_one(
                    model=model,
                    model_name=architecture,
                    category="verifier",
                    spec=spec,
                    seed=seed,
                    train_indices=train_indices,
                    validation_indices=validation_indices,
                    test_indices=test_indices,
                    labels_tensor=data.labels,
                    labels_numpy=store.labels,
                    batch_function=lambda index, r=required: data.batch(index, r),
                    normalizer_path=data.normalizer_path,
                    output_paths=output_paths(
                        "verifier", architecture, spec, seed
                    ),
                    force=force,
                )
                completed += 1
                print(
                    "V7 "
                    f"spec={spec_number}/{len(specs)} "
                    f"arch={architecture_number}/{len(architectures)} "
                    f"seed={seed_number}/{len(seeds)} name={architecture} "
                    f"epochs={result.metadata['epochs_run']} "
                    f"test_BA={result.metadata['test_metrics']['balanced_accuracy']:.4f} "
                    f"gpu_s={result.metadata['GPU_seconds']:.2f}",
                    flush=True,
                )
        data.close()
    if (
        not finalize
        or selected_specs is not None
        or selected_architectures is not None
        or selected_seeds is not None
    ):
        return {
            "partial": True,
            "runs_completed_or_verified": completed,
            "wall_seconds": time.monotonic() - started,
        }
    training_records = []
    for spec in all_specs:
        for architecture in ARCHITECTURES:
            for seed in SEEDS:
                path = output_paths("verifier", architecture, spec, seed)["record"]
                if not path.is_file():
                    raise RuntimeError(f"missing completed V7 run {path}")
                training_records.append(read_json(path))
    evaluation_rows = _evaluate_pooled(store)
    summary, per_seed, per_environment = _summaries(
        store, training_records, evaluation_rows
    )
    write_json(OUT / "verifier_summary.json", summary)
    write_json(OUT / "verifier_per_seed.json", per_seed)
    write_json(OUT / "verifier_per_environment.json", per_environment)
    append_log(
        "V7_FAIR_VERIFIER_ZOO_COMPLETE",
        architectures=len(ARCHITECTURES),
        training_runs=len(training_records),
        evaluation_rows=len(per_environment["rows"]),
        raw_scores_saved_for_all_environments=per_environment[
            "raw_scores_saved_for_all_environments"
        ],
        GPU_hours=summary["total_GPU_hours"],
        within_resource_limit=summary["total_GPU_hours"] <= 20.0,
        summary_sha256=sha256_file(OUT / "verifier_summary.json"),
    )
    return {
        "partial": False,
        "architectures": len(ARCHITECTURES),
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
    parser.add_argument("--architecture", action="append", choices=ARCHITECTURES)
    parser.add_argument("--seed", action="append", type=int, choices=SEEDS)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-finalize", action="store_true")
    arguments = parser.parse_args()
    print(
        json.dumps(
            run(
                selected_specs=set(arguments.spec) if arguments.spec else None,
                selected_architectures=(
                    set(arguments.architecture) if arguments.architecture else None
                ),
                selected_seeds=set(arguments.seed) if arguments.seed else None,
                force=arguments.force,
                finalize=not arguments.no_finalize,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
