"""Phase A: immutable row alignment, normalization, split, and batch audits."""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from typing import Any

import numpy as np

from actmask.experiments.verifier_shortcut_audit.nuisance import action_features
from actmask.experiments.verifier_shortcut_audit.modeling import VERIFIER_CLASSES

from .common import (
    EASY_FAMILIES,
    OUT,
    SEEDS,
    SOURCE,
    SPECS,
    STUDY_ID,
    append_log,
    read_json,
    read_jsonl,
    sha256_file,
    verify_bound_source_snapshot,
    write_json,
)


def _array_sha(value: np.ndarray, dtype: np.dtype[Any] | type[np.generic]) -> str:
    return hashlib.sha256(np.asarray(value, dtype=dtype).tobytes()).hexdigest()


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


def _mismatch_summary(values: list[Any]) -> dict[str, Any]:
    return {"count": len(values), "first_25": values[:25]}


def run() -> dict[str, Any]:
    config = read_json(OUT / "preregistered_config.json")
    expected_config_hash = (
        OUT / "preregistered_config.sha256"
    ).read_text(encoding="utf-8").strip()
    if sha256_file(OUT / "preregistered_config.json") != expected_config_hash:
        raise RuntimeError("frozen preregistration changed")
    source_preservation = verify_bound_source_snapshot(config)
    if not source_preservation["pass"]:
        raise RuntimeError(f"source study changed: {source_preservation}")

    rows = read_jsonl(SOURCE / "unified_audit_manifest.jsonl")
    with np.load(SOURCE / "unified_audit_arrays.npz") as stored:
        arrays = {name: stored[name].copy() for name in stored.files}
    with np.load(SOURCE / "nuisance_features.npz") as stored:
        nuisance = {name: stored[name].copy() for name in stored.files}
    with np.load(SOURCE / "audit_environment_indices.npz") as stored:
        indices = {name: stored[name].copy() for name in stored.files}
    schema = read_json(SOURCE / "nuisance_feature_schema.json")
    environment = read_json(SOURCE / "audit_environment_manifest.json")

    family_names = sorted({row["family"] for row in rows})
    family_lookup = {name: index for index, name in enumerate(family_names)}
    row_index_mismatch = []
    label_mismatch = []
    family_mismatch = []
    action_hash_mismatch = []
    context_anchor_mismatch = []
    source_anchor_mismatch = []
    sample_ids = []
    for index, row in enumerate(rows):
        sample_ids.append(row["sample_id"])
        if row["unified_array_index"] != index:
            row_index_mismatch.append(
                [index, row["unified_array_index"], row["sample_id"]]
            )
        if row["construction_label"] != int(arrays["label"][index]):
            label_mismatch.append(row["sample_id"])
        if family_lookup[row["family"]] != int(arrays["family_index"][index]):
            family_mismatch.append(row["sample_id"])
        actual_action_hash = hashlib.sha256(
            np.asarray(arrays["candidate_action"][index], dtype=np.float32).tobytes()
        ).hexdigest()
        if actual_action_hash != row["action_sha256"]:
            action_hash_mismatch.append(row["sample_id"])
        expected_context = f"a{int(arrays['context_anchor_index'][index]):06d}"
        if row["anchor_id"] != expected_context:
            context_anchor_mismatch.append(
                [row["sample_id"], row["anchor_id"], expected_context]
            )
        expected_source = f"a{int(arrays['source_anchor_index'][index]):06d}"
        if row["source_anchor_id"] != expected_source:
            source_anchor_mismatch.append(
                [row["sample_id"], row["source_anchor_id"], expected_source]
            )

    duplicate_sample_ids = [
        sample_id for sample_id, count in Counter(sample_ids).items() if count != 1
    ]
    nuisance_label_mismatch = np.flatnonzero(
        nuisance["label"].astype(np.int8) != arrays["label"].astype(np.int8)
    ).tolist()
    finite_arrays = {
        name: bool(np.isfinite(value).all())
        for name, value in arrays.items()
        if np.issubdtype(value.dtype, np.number)
    }
    finite_nuisance = {
        name: bool(np.isfinite(value).all())
        for name, value in nuisance.items()
        if np.issubdtype(value.dtype, np.number)
    }

    pair_groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        family = row["family"].split("_", 1)[0]
        if family in EASY_FAMILIES:
            pair_groups[(family, row["anchor_id"])].append(index)
    pair_failures = []
    context_tensor_failures = []
    for (family, anchor), pair in sorted(pair_groups.items()):
        if len(pair) != 2 or set(arrays["label"][pair].tolist()) != {0, 1}:
            pair_failures.append([family, anchor, pair])
            continue
        for name in (
            "history_state",
            "current_state",
            "visual_history",
            "language_embedding",
            "context_anchor_index",
            "context_episode_index",
            "task_index",
        ):
            if not np.array_equal(arrays[name][pair[0]], arrays[name][pair[1]]):
                context_tensor_failures.append([family, anchor, name])

    current_difference = np.abs(
        arrays["current_state"] - arrays["history_state"][:, -1]
    )
    current_state_relation = {
        "exact_equal": bool(np.array_equal(
            arrays["current_state"], arrays["history_state"][:, -1]
        )),
        "maximum_absolute_difference": float(current_difference.max()),
        "mean_absolute_difference": float(current_difference.mean()),
        "rows_over_1e-6": int(np.any(current_difference > 1e-6, axis=1).sum()),
    }

    required_input_collision_rows = []
    for architecture, model_class in VERIFIER_CLASSES.items():
        model = model_class()
        required = tuple(model.required_inputs)
        del model
        for family, spec in zip(EASY_FAMILIES, SPECS, strict=True):
            for partition in ("train", "validation", "test"):
                selected = indices[f"spec__{spec}__{partition}"].astype(np.int64)
                grouped: dict[str, Counter[int]] = defaultdict(Counter)
                for index in selected:
                    digest = hashlib.sha256()
                    for name in required:
                        value = np.asarray(arrays[name][int(index)])
                        digest.update(name.encode("utf-8"))
                        digest.update(str(value.dtype).encode("ascii"))
                        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
                        digest.update(value.tobytes())
                    grouped[digest.hexdigest()][int(arrays["label"][index])] += 1
                conflicts = [
                    {"input_sha256": digest, "labels": dict(sorted(counts.items()))}
                    for digest, counts in grouped.items()
                    if len(counts) > 1
                ]
                correct = sum(max(counts.values()) for counts in grouped.values())
                required_input_collision_rows.append(
                    {
                        "architecture": architecture,
                        "family": family,
                        "spec": spec,
                        "partition": partition,
                        "required_inputs": list(required),
                        "samples": int(len(selected)),
                        "unique_input_hashes": len(grouped),
                        "conflicting_label_input_groups": len(conflicts),
                        "first_25_conflicts": conflicts[:25],
                        "empirical_exact_input_accuracy_and_BA_ceiling": float(
                            correct / len(selected)
                        ),
                    }
                )

    recomputed_action = np.empty_like(nuisance["action"])
    recomputed_names: list[str] | None = None
    for index, action in enumerate(arrays["candidate_action"]):
        value, names = action_features(action)
        recomputed_action[index] = value
        if recomputed_names is None:
            recomputed_names = names
        elif names != recomputed_names:
            raise RuntimeError("action feature names changed across rows")
    action_error = np.abs(recomputed_action - nuisance["action"])
    action_feature_recomputation = {
        "features": int(recomputed_action.shape[1]),
        "schema_names_exact": recomputed_names == schema["groups"]["action"],
        "maximum_absolute_error": float(action_error.max()),
        "mean_absolute_error": float(action_error.mean()),
        "values_over_1e-6": int(np.sum(action_error > 1e-6)),
        "pass": (
            recomputed_names == schema["groups"]["action"]
            and float(action_error.max()) <= 1e-6
        ),
    }

    alignment_failures = (
        row_index_mismatch
        + label_mismatch
        + family_mismatch
        + action_hash_mismatch
        + context_anchor_mismatch
        + source_anchor_mismatch
        + duplicate_sample_ids
        + nuisance_label_mismatch
        + pair_failures
        + context_tensor_failures
    )
    alignment_pass = (
        not alignment_failures
        and all(finite_arrays.values())
        and all(finite_nuisance.values())
        and action_feature_recomputation["pass"]
        and source_preservation["pass"]
        and current_state_relation["exact_equal"]
        and current_state_relation["rows_over_1e-6"] == 0
    )
    alignment_report = {
        "schema": "vild-alignment-audit-v1",
        "study_id": STUDY_ID,
        "pass": alignment_pass,
        "samples": len(rows),
        "unique_sample_ids": len(set(sample_ids)),
        "families": family_names,
        "array_shapes": {
            name: list(value.shape) for name, value in arrays.items()
        },
        "source_preservation": source_preservation,
        "checks": {
            "row_index_mismatch": _mismatch_summary(row_index_mismatch),
            "label_mismatch": _mismatch_summary(label_mismatch),
            "family_mismatch": _mismatch_summary(family_mismatch),
            "action_hash_mismatch": _mismatch_summary(action_hash_mismatch),
            "context_anchor_mismatch": _mismatch_summary(context_anchor_mismatch),
            "source_anchor_mismatch": _mismatch_summary(source_anchor_mismatch),
            "duplicate_sample_ids": _mismatch_summary(duplicate_sample_ids),
            "nuisance_label_mismatch": _mismatch_summary(nuisance_label_mismatch),
            "easy_family_pair_failures": _mismatch_summary(pair_failures),
            "paired_context_tensor_failures": _mismatch_summary(
                context_tensor_failures
            ),
            "finite_unified_arrays": finite_arrays,
            "finite_nuisance_arrays": finite_nuisance,
        },
        "easy_family_context_label_pairs": len(pair_groups),
        "current_state_vs_last_history_state": current_state_relation,
        "required_input_collision_audit": required_input_collision_rows,
        "action_feature_recomputation": action_feature_recomputation,
        "construction_labels_only": True,
        "physical_ground_truth": False,
    }

    split_rows = []
    split_failures = []
    normalizer_rows = []
    normalizer_failures = []
    batch_rows = []
    batch_failures = []
    for family, spec in zip(EASY_FAMILIES, SPECS, strict=True):
        partition = {
            name: indices[f"spec__{spec}__{name}"].astype(np.int64)
            for name in ("train", "validation", "test")
        }
        expected_spec = environment["training_specs"][spec]["partitions"]
        family_full_name = next(
            name for name in family_names if name.startswith(f"{family}_")
        )
        split_row: dict[str, Any] = {
            "family": family,
            "spec": spec,
            "partitions": {},
            "overlaps": {},
        }
        for name, value in partition.items():
            labels = arrays["label"][value]
            families = {rows[int(index)]["family"] for index in value}
            index_hash = _array_sha(value, np.int32)
            expected_hash = expected_spec[name]["index_sha256"]
            entry = {
                "samples": int(len(value)),
                "unique_indices": int(len(np.unique(value))),
                "positive_rate": float(labels.mean()),
                "labels": {
                    "0": int(np.sum(labels == 0)),
                    "1": int(np.sum(labels == 1)),
                },
                "families": sorted(families),
                "index_sha256": index_hash,
                "expected_index_sha256": expected_hash,
                "hash_match": index_hash == expected_hash,
                "indices_in_range": bool(
                    np.all((value >= 0) & (value < len(rows)))
                ),
            }
            split_row["partitions"][name] = entry
            if (
                len(value) != len(np.unique(value))
                or entry["positive_rate"] != 0.5
                or families != {family_full_name}
                or index_hash != expected_hash
                or not entry["indices_in_range"]
            ):
                split_failures.append([spec, name, entry])
        for left, right in (
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test"),
        ):
            key = f"{left}_{right}"
            left_index, right_index = partition[left], partition[right]
            overlap = {
                "sample_indices": int(
                    len(set(left_index.tolist()) & set(right_index.tolist()))
                ),
                "context_episodes": int(
                    len(
                        set(arrays["context_episode_index"][left_index].tolist())
                        & set(arrays["context_episode_index"][right_index].tolist())
                    )
                ),
                "source_episodes": int(
                    len(
                        set(arrays["source_episode_index"][left_index].tolist())
                        & set(arrays["source_episode_index"][right_index].tolist())
                    )
                ),
            }
            split_row["overlaps"][key] = overlap
            if any(overlap.values()):
                split_failures.append([spec, key, overlap])
        split_rows.append(split_row)

        normalizer_path = SOURCE / "verifier_normalizers" / f"{spec}.npz"
        with np.load(normalizer_path) as stored:
            frozen = {name: stored[name].copy() for name in stored.files}
        normalizer_entry = {
            "spec": spec,
            "path": str(normalizer_path.resolve()),
            "sha256": sha256_file(normalizer_path),
            "inputs": {},
        }
        for name in (
            "history_state",
            "candidate_action",
            "language_embedding",
            "visual_history",
        ):
            mean, std = _normalization(arrays[name], partition["train"])
            mean_error = float(
                np.max(np.abs(mean - frozen[f"{name}_mean"]))
            )
            std_error = float(np.max(np.abs(std - frozen[f"{name}_std"])))
            input_entry = {
                "mean_max_absolute_error": mean_error,
                "std_max_absolute_error": std_error,
                "finite": bool(np.isfinite(mean).all() and np.isfinite(std).all()),
                "std_min": float(std.min()),
                "pass": (
                    mean_error <= config[
                        "normalization_split_batch_audit"
                    ]["comparison_to_frozen_normalizers_absolute_tolerance"]
                    and std_error <= config[
                        "normalization_split_batch_audit"
                    ]["comparison_to_frozen_normalizers_absolute_tolerance"]
                    and np.isfinite(mean).all()
                    and np.isfinite(std).all()
                    and float(std.min()) > 0
                ),
            }
            normalizer_entry["inputs"][name] = input_entry
            if not input_entry["pass"]:
                normalizer_failures.append([spec, name, input_entry])
        normalizer_rows.append(normalizer_entry)

        for seed in SEEDS:
            generator = np.random.default_rng(seed * 100003)
            order = generator.permutation(partition["train"])
            batch_entry = {
                "spec": spec,
                "seed": seed,
                "epoch": 0,
                "samples": int(len(order)),
                "unique_samples": int(len(np.unique(order))),
                "same_set_as_train": set(order.tolist())
                == set(partition["train"].tolist()),
                "order_sha256": _array_sha(order, np.int32),
                "no_validation_or_test_indices": not bool(
                    set(order.tolist())
                    & (
                        set(partition["validation"].tolist())
                        | set(partition["test"].tolist())
                    )
                ),
            }
            batch_entry["pass"] = (
                batch_entry["unique_samples"] == batch_entry["samples"]
                and batch_entry["same_set_as_train"]
                and batch_entry["no_validation_or_test_indices"]
            )
            batch_rows.append(batch_entry)
            if not batch_entry["pass"]:
                batch_failures.append(batch_entry)

    normalization_split_pass = (
        not split_failures and not normalizer_failures and not batch_failures
    )
    normalization_report = {
        "schema": "vild-normalization-split-audit-v1",
        "study_id": STUDY_ID,
        "pass": normalization_split_pass,
        "normalization_uses_train_partition_only": True,
        "heldout_test_used_for_statistics_or_selection": False,
        "split_rows": split_rows,
        "normalizer_rows": normalizer_rows,
        "batch_rows": batch_rows,
        "failures": {
            "split": _mismatch_summary(split_failures),
            "normalizer": _mismatch_summary(normalizer_failures),
            "batch": _mismatch_summary(batch_failures),
        },
    }
    write_json(OUT / "alignment_audit.json", alignment_report)
    write_json(
        OUT / "normalization_and_split_audit.json", normalization_report
    )

    events = {
        row["event"]
        for row in read_jsonl(OUT / "run_log.jsonl")
    }
    event = (
        "A_DATA_PIPELINE_AUDIT_REVERIFIED_AFTER_AMENDMENT"
        if config.get("schema") == "vild-preregistered-config-v1.1"
        else "A_DATA_PIPELINE_AUDIT_COMPLETE"
    )
    if event not in events:
        append_log(
            event,
            alignment_pass=alignment_pass,
            normalization_split_pass=normalization_split_pass,
            rows=len(rows),
            action_hashes_recomputed=len(rows),
            action_summary_values_recomputed=int(recomputed_action.size),
            split_failures=len(split_failures),
            normalizer_failures=len(normalizer_failures),
            batch_failures=len(batch_failures),
            alignment_sha256=sha256_file(OUT / "alignment_audit.json"),
            normalization_split_sha256=sha256_file(
                OUT / "normalization_and_split_audit.json"
            ),
            new_training_started=False,
        )
    return {
        "alignment_pass": alignment_pass,
        "normalization_split_pass": normalization_split_pass,
        "rows": len(rows),
        "action_feature_max_error": action_feature_recomputation[
            "maximum_absolute_error"
        ],
        "current_state_exactly_last_history_state": current_state_relation[
            "exact_equal"
        ],
        "split_failures": len(split_failures),
        "normalizer_failures": len(normalizer_failures),
        "batch_failures": len(batch_failures),
    }


def main() -> None:
    print(json.dumps(run(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
