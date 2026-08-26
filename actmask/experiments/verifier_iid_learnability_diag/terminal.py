"""Frozen terminal diagnosis for VERIFIER-IID-LEARNABILITY-DIAG-V1.

The module is deliberately downstream-only.  It consumes completed diagnostic
reports and the 60-run final IID confirmation, applies the amended v1.1
decision order, writes the six terminal artifacts, appends one hash-chained
terminal event, and can independently re-verify the complete package.

It does not train, repair, alter the source VSA study, or create a new verifier
architecture.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .common import (
    ARCHITECTURES,
    EASY_FAMILIES,
    OUT,
    PYTHON,
    ROOT,
    SEEDS,
    SOURCE,
    SPECS,
    STUDY_ID,
    append_log,
    canonical_json,
    read_json,
    read_jsonl,
    sha256_file,
    sha256_json,
    verify_snapshot,
    verify_payload_sha256,
    write_json,
)


TERMINAL_EVENT = "TERMINAL_ROOT_CAUSE_DECISION_COMPLETE"
CONFIG_SCHEMA = "vild-preregistered-config-v1.1"
FINAL_RESULT_SCHEMA = "vild-iid-learnability-results-v1"
FINAL_GROUP = "final_confirmation"
FINAL_RUNS = 60
FINAL_CELLS = 20
FINAL_BA_GATE = 0.65

REQUIRED_OUTPUTS = (
    "preregistered_config.json",
    "alignment_audit.json",
    "normalization_and_split_audit.json",
    "sanity_control_report.json",
    "memorization_report.json",
    "gradient_flow_report.json",
    "optimization_diagnostics.json",
    "representation_mismatch_report.json",
    "per_architecture_diagnosis.json",
    "iid_learnability_results.json",
    "root_cause_decision.json",
    "final_report.md",
    "executive_summary_zh.md",
    "next_steps.md",
    "reproducibility_manifest.json",
    "run_log.jsonl",
)

AMENDMENT_FILES = (
    "preregistered_config.json",
    "preregistered_config.sha256",
    "preregistered_config_initial.json",
    "preregistered_config_initial.sha256",
    "preregistration_amendment.json",
    "source_preservation_snapshot.json",
)

DIAGNOSTIC_INPUTS = (
    "alignment_audit.json",
    "normalization_and_split_audit.json",
    "historical_baseline_evidence.json",
    "sanity_control_report.json",
    "memorization_report.json",
    "gradient_flow_report.json",
    "optimization_diagnostics.json",
    "representation_mismatch_report.json",
    "iid_learnability_results.json",
)

TERMINAL_GENERATED = (
    "per_architecture_diagnosis.json",
    "root_cause_decision.json",
    "final_report.md",
    "executive_summary_zh.md",
    "next_steps.md",
    "reproducibility_manifest.json",
)

EXPECTED_DECISION_ORDER = (
    "VILD_IMPLEMENTATION_FAULT_REPAIRED_AND_IID_RESTORED",
    "VILD_ALIGNMENT_OR_IMPLEMENTATION_FAULT_CONFIRMED",
    "VILD_IID_LEARNABILITY_RESTORED_WITHOUT_REPAIR",
    "VILD_CALIBRATION_THRESHOLD_FAILURE",
    "VILD_OPTIMIZATION_TRAINING_FAILURE",
    "VILD_REPRESENTATION_INDUCTIVE_BIAS_MISMATCH",
    "VILD_CAPACITY_OR_ARCHITECTURE_FAILURE",
    "VILD_INSUFFICIENT_OR_HETEROGENEOUS_EVIDENCE",
)

CELL_ORDER = (
    "exact_input_nonidentifiability",
    "gradient_or_pipeline_failure",
    "calibration_failure",
    "optimization_failure",
    "representation_mismatch",
    "capacity_failure",
    "IID_learned",
    "insufficient",
)

ALLOWED_TRAINING_GROUPS = {
    "gradient",
    "tiny_memorization",
    "full_memorization",
    "positive_sanity",
    "shuffled_sanity",
    "constant_sanity",
    "alternate_schedule",
    "representation_control",
    "final_confirmation",
}

ALLOWED_MODELS = set(ARCHITECTURES) | {
    "engineered_current_action_summary",
    "flattened_raw_current_action",
}

FORBIDDEN_PATH_TOKENS = (
    "shortcutresistant",
    "mitigation",
    "sam3",
    "robot_mask",
    "world_model",
    "world-model",
    "physical_validity",
    "physical-validity",
    "foundation_model",
)

MANIFEST_EXCLUSIONS = {
    "reproducibility_manifest.json",
    "run_log.jsonl",
}


def _text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value.strip() + "\n", encoding="utf-8")
    temporary.replace(path)


def _entry(path: Path, base: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(base)),
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def _require_files(names: tuple[str, ...]) -> None:
    missing = [
        name
        for name in names
        if not (OUT / name).is_file() or (OUT / name).stat().st_size == 0
    ]
    if missing:
        raise RuntimeError("missing or empty prerequisites: " + ", ".join(missing))


def _read_inputs() -> dict[str, Any]:
    _require_files(AMENDMENT_FILES + DIAGNOSTIC_INPUTS + ("run_log.jsonl",))
    return {
        "config": read_json(OUT / "preregistered_config.json"),
        "alignment": read_json(OUT / "alignment_audit.json"),
        "normalization": read_json(
            OUT / "normalization_and_split_audit.json"
        ),
        "historical": read_json(OUT / "historical_baseline_evidence.json"),
        "sanity": read_json(OUT / "sanity_control_report.json"),
        "memorization": read_json(OUT / "memorization_report.json"),
        "gradient": read_json(OUT / "gradient_flow_report.json"),
        "optimization": read_json(OUT / "optimization_diagnostics.json"),
        "representation": read_json(
            OUT / "representation_mismatch_report.json"
        ),
        "iid": read_json(OUT / "iid_learnability_results.json"),
    }


def _audit_run_log(require_terminal: bool) -> dict[str, Any]:
    path = OUT / "run_log.jsonl"
    rows = read_jsonl(path) if path.is_file() else []
    failures: list[dict[str, Any]] = []
    previous = None
    for index, row in enumerate(rows):
        payload = dict(row)
        recorded_hash = payload.pop("record_sha256", None)
        recomputed_hash = hashlib.sha256(
            canonical_json(payload).encode("utf-8")
        ).hexdigest()
        if row.get("sequence") != index:
            failures.append({"row": index, "reason": "sequence"})
        if row.get("study_id") != STUDY_ID:
            failures.append({"row": index, "reason": "study_id"})
        if row.get("previous_record_sha256") != previous:
            failures.append({"row": index, "reason": "previous_record_sha256"})
        if recorded_hash != recomputed_hash:
            failures.append({"row": index, "reason": "record_sha256"})
        previous = recorded_hash

    events = [row.get("event") for row in rows]
    required_order = [
        "PREREGISTRATION_FROZEN",
        "PREREGISTRATION_AMENDED_BEFORE_TRAINING",
        "A_DATA_PIPELINE_AUDIT_REVERIFIED_AFTER_AMENDMENT",
        "B_E_LEARNED_DIAGNOSTICS_COMPLETE",
        "FINAL_IID_CONFIRMATION_COMPLETE",
    ]
    if require_terminal:
        required_order.append(TERMINAL_EVENT)
    positions: list[int] = []
    for event in required_order:
        matches = [index for index, value in enumerate(events) if value == event]
        if len(matches) != 1:
            failures.append(
                {
                    "reason": "required_event_count",
                    "event": event,
                    "count": len(matches),
                }
            )
        elif matches:
            positions.append(matches[0])
    if len(positions) == len(required_order) and positions != sorted(positions):
        failures.append(
            {
                "reason": "required_event_order",
                "events": required_order,
                "positions": positions,
            }
        )
    terminal_rows = [row for row in rows if row.get("event") == TERMINAL_EVENT]
    if terminal_rows and rows[-1].get("event") != TERMINAL_EVENT:
        failures.append({"reason": "terminal_event_not_last"})
    return {
        "pass": not failures,
        "records": len(rows),
        "events": events,
        "final_record_sha256": previous,
        "terminal_event_count": len(terminal_rows),
        "terminal_event": terminal_rows[0] if len(terminal_rows) == 1 else None,
        "failures": failures,
    }


def _verify_amendment_and_snapshots(
    config: dict[str, Any],
) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    config_path = OUT / "preregistered_config.json"
    config_sha256 = sha256_file(config_path)
    frozen_hash = (OUT / "preregistered_config.sha256").read_text(
        encoding="utf-8"
    ).strip()
    initial_path = OUT / "preregistered_config_initial.json"
    initial_sha256 = sha256_file(initial_path)
    frozen_initial_hash = (
        OUT / "preregistered_config_initial.sha256"
    ).read_text(encoding="utf-8").strip()
    amendment = read_json(OUT / "preregistration_amendment.json")
    snapshot = read_json(OUT / "source_preservation_snapshot.json")

    if config.get("schema") != CONFIG_SCHEMA:
        failures.append(
            {
                "field": "config.schema",
                "expected": CONFIG_SCHEMA,
                "actual": config.get("schema"),
            }
        )
    if frozen_hash != config_sha256:
        failures.append({"field": "preregistered_config.sha256"})
    if frozen_initial_hash != initial_sha256:
        failures.append({"field": "preregistered_config_initial.sha256"})
    if config.get("amendment", {}).get("parent_sha256") != initial_sha256:
        failures.append({"field": "config.amendment.parent_sha256"})
    if amendment.get("parent_config_sha256") != initial_sha256:
        failures.append({"field": "amendment.parent_config_sha256"})
    if amendment.get("amended_config_sha256") != config_sha256:
        failures.append({"field": "amendment.amended_config_sha256"})
    if amendment.get("timing") != "before any new training":
        failures.append({"field": "amendment.timing"})
    if amendment.get("training_artifacts_present") is not False:
        failures.append({"field": "amendment.training_artifacts_present"})
    if tuple(config.get("root_cause_decision_order", ())) != EXPECTED_DECISION_ORDER:
        failures.append(
            {
                "field": "root_cause_decision_order",
                "actual": config.get("root_cause_decision_order"),
            }
        )
    if tuple(
        config.get("per_architecture_diagnosis", {}).get("cell_order", ())
    ) != CELL_ORDER:
        failures.append(
            {
                "field": "per_architecture_diagnosis.cell_order",
                "actual": config.get("per_architecture_diagnosis", {}).get(
                    "cell_order"
                ),
            }
        )
    if tuple(config.get("required_outputs", ())) != REQUIRED_OUTPUTS:
        failures.append(
            {
                "field": "required_outputs",
                "expected": list(REQUIRED_OUTPUTS),
                "actual": config.get("required_outputs"),
            }
        )

    implementation_failures = []
    for row in config.get("implementation_freeze", []):
        path = ROOT / row["path"]
        if (
            not path.is_file()
            or path.stat().st_size != row["bytes"]
            or sha256_file(path) != row["sha256"]
        ):
            implementation_failures.append(row["path"])
    if implementation_failures:
        failures.append(
            {
                "field": "implementation_freeze",
                "failures": implementation_failures,
            }
        )
    pretraining_evidence_failures = []
    for row in config.get("pretraining_evidence_freeze", []):
        path = OUT / row["path"]
        if (
            not path.is_file()
            or path.stat().st_size != row["bytes"]
            or sha256_file(path) != row["sha256"]
        ):
            pretraining_evidence_failures.append(row["path"])
    if pretraining_evidence_failures:
        failures.append(
            {
                "field": "pretraining_evidence_freeze",
                "failures": pretraining_evidence_failures,
            }
        )
    if amendment.get("implementation_freeze") != config.get(
        "implementation_freeze"
    ):
        failures.append({"field": "amendment.implementation_freeze"})
    if amendment.get("pretraining_evidence_freeze") != config.get(
        "pretraining_evidence_freeze"
    ):
        failures.append({"field": "amendment.pretraining_evidence_freeze"})

    source_check = verify_snapshot(SOURCE, snapshot["files"])
    if not source_check["pass"]:
        failures.append(
            {"field": "source_preservation_snapshot", "detail": source_check}
        )
    if snapshot.get("tree_sha256") != source_check["snapshot_sha256"]:
        failures.append({"field": "source_snapshot.tree_sha256"})
    if config.get("source_study", {}).get("snapshot_sha256") != source_check[
        "snapshot_sha256"
    ]:
        failures.append({"field": "config.source_study.snapshot_sha256"})

    return {
        "pass": not failures,
        "config_sha256": config_sha256,
        "initial_config_sha256": initial_sha256,
        "amendment_sha256": sha256_file(
            OUT / "preregistration_amendment.json"
        ),
        "implementation_files": len(config.get("implementation_freeze", [])),
        "implementation_failures": implementation_failures,
        "pretraining_evidence_files": len(
            config.get("pretraining_evidence_freeze", [])
        ),
        "pretraining_evidence_failures": pretraining_evidence_failures,
        "source_snapshot": source_check,
        "failures": failures,
    }


def _identity_set(
    rows: list[dict[str, Any]],
    *,
    include_seed: bool = True,
) -> set[tuple[Any, ...]]:
    result = set()
    for row in rows:
        architecture = row.get("architecture", row.get("model"))
        family = row.get(
            "family",
            str(row.get("spec", "")).replace("within_", "").upper(),
        )
        if include_seed:
            result.add((architecture, family, int(row["seed"])))
        else:
            result.add((architecture, family))
    return result


def _diagnostic_completeness(inputs: dict[str, Any]) -> dict[str, Any]:
    alignment = inputs["alignment"]
    normalization = inputs["normalization"]
    sanity = inputs["sanity"]
    memorization = inputs["memorization"]
    gradient = inputs["gradient"]
    optimization = inputs["optimization"]
    representation = inputs["representation"]
    historical = inputs["historical"]
    expected_cells = {
        (architecture, family)
        for architecture in ARCHITECTURES
        for family in EASY_FAMILIES
    }
    expected_final = {
        (architecture, family, seed)
        for architecture in ARCHITECTURES
        for family in EASY_FAMILIES
        for seed in SEEDS
    }

    collision_rows = alignment.get("required_input_collision_audit", [])
    collision_identity = {
        (row["architecture"], row["family"], row["partition"])
        for row in collision_rows
    }
    expected_collision = {
        (architecture, family, partition)
        for architecture in ARCHITECTURES
        for family in EASY_FAMILIES
        for partition in ("train", "validation", "test")
    }
    sanity_counts = {
        name: len(sanity.get(name, {}).get("rows", []))
        for name in (
            "positive_observable_signal",
            "shuffled_label_negative",
            "constant_input_negative",
        )
    }
    tiny_rows = memorization.get("tiny_rows", [])
    full_rows = memorization.get("full_rows", [])
    gradient_rows = gradient.get("rows", [])
    alternate = optimization.get("alternate_schedule", {})
    controls = representation.get("diagnostic_controls", {})
    checks = {
        "alignment_report_complete": (
            alignment.get("samples") == 12676
            and alignment.get("unique_sample_ids") == 12676
            and alignment.get("action_feature_recomputation", {}).get("pass")
            is True
            and alignment.get("current_state_vs_last_history_state", {}).get(
                "exact_equal"
            )
            is True
            and collision_identity == expected_collision
        ),
        "normalization_split_report_complete": (
            len(normalization.get("split_rows", [])) == 4
            and len(normalization.get("batch_rows", [])) == 12
            and normalization.get("normalization_uses_train_partition_only")
            is True
            and normalization.get(
                "heldout_test_used_for_statistics_or_selection"
            )
            is False
        ),
        "historical_evidence_complete": (
            historical.get("complete") is True
            and len(historical.get("historical_runs", [])) == 60
        ),
        "sanity_controls_complete": (
            sanity_counts
            == {
                "positive_observable_signal": 5,
                "shuffled_label_negative": 5,
                "constant_input_negative": 5,
            }
        ),
        "memorization_complete": (
            len(tiny_rows) == 20
            and len(full_rows) == 20
            and _identity_set(tiny_rows, include_seed=False) == expected_cells
            and _identity_set(full_rows, include_seed=False) == expected_cells
        ),
        "gradient_complete": (
            len(gradient_rows) == 20
            and _identity_set(gradient_rows, include_seed=False)
            == expected_cells
        ),
        "alternate_schedule_complete": (
            alternate.get("run_count") == 60
            and len(alternate.get("runs", [])) == 60
            and _identity_set(alternate.get("runs", [])) == expected_final
            and len(alternate.get("cells", [])) == 20
            and alternate.get("test_used_for_selection") is False
        ),
        "representation_controls_complete": (
            controls.get("run_count") == 24
            and len(controls.get("runs", [])) == 24
            and len(controls.get("cells", [])) == 8
            and controls.get("test_used_for_selection") is False
            and controls.get("controls_are_additional_verifier_architectures")
            is False
            and representation.get("additional_verifier_architectures") == 0
        ),
    }
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "report_hashes": {
            name: sha256_file(OUT / name) for name in DIAGNOSTIC_INPUTS[:-1]
        },
        "diagnostic_outcomes": {
            "alignment_pass": bool(alignment.get("pass")),
            "normalization_split_pass": bool(normalization.get("pass")),
            "sanity_pass": bool(sanity.get("pass")),
            "positive_sanity_pass": bool(
                sanity.get("positive_observable_signal", {}).get("pass")
            ),
            "negative_sanity_pass": bool(
                sanity.get("shuffled_label_negative", {}).get("pass")
                and sanity.get("constant_input_negative", {}).get("pass")
            ),
            "gradient_pass": bool(gradient.get("pass")),
            "all_tiny_memorization_pass": bool(
                memorization.get("all_tiny_pass")
            ),
            "all_full_memorization_pass": bool(
                memorization.get("all_full_pass")
            ),
            "alternate_study_pass": bool(alternate.get("study_pass")),
            "engineered_control_support": bool(
                representation.get("engineered_control_support")
            ),
        },
    }


def _validate_final_confirmation(
    iid: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    expected = {
        (architecture, spec, seed)
        for architecture in ARCHITECTURES
        for spec in SPECS
        for seed in SEEDS
    }
    records = iid.get("training_runs", [])
    observed = {
        (row.get("model"), row.get("spec"), int(row.get("seed", -1)))
        for row in records
    }
    if iid.get("schema") != FINAL_RESULT_SCHEMA:
        failures.append({"field": "schema", "actual": iid.get("schema")})
    if iid.get("study_id") != STUDY_ID:
        failures.append({"field": "study_id"})
    expected_protocol = config["machine_training_protocols"][
        "final_original"
    ]
    if iid.get("protocol") != expected_protocol:
        failures.append(
            {
                "field": "protocol",
                "expected": expected_protocol,
                "actual": iid.get("protocol"),
            }
        )
    expected_scope = {
        "architectures": list(ARCHITECTURES),
        "families": list(EASY_FAMILIES),
        "seeds": list(SEEDS),
        "expected_runs": FINAL_RUNS,
        "completed_runs": FINAL_RUNS,
    }
    if iid.get("scope") != expected_scope:
        failures.append(
            {
                "field": "scope",
                "expected": expected_scope,
                "actual": iid.get("scope"),
            }
        )
    if iid.get("prerequisite_diagnostics", {}).get("pass") is not True:
        failures.append({"field": "prerequisite_diagnostics.pass"})
    for field in (
        "implementation_freeze_before",
        "implementation_freeze_after",
    ):
        freeze = iid.get(field, {})
        if (
            freeze.get("failures")
            or freeze.get("pretraining_evidence_failures")
            or freeze.get("config_sha256")
            != sha256_file(OUT / "preregistered_config.json")
        ):
            failures.append({"field": field})
    for field in (
        "source_preservation_before",
        "source_preservation_after",
    ):
        if iid.get(field, {}).get("pass") is not True:
            failures.append({"field": field})
    if iid.get("run_count") != FINAL_RUNS or len(records) != FINAL_RUNS:
        failures.append(
            {
                "field": "run_count",
                "declared": iid.get("run_count"),
                "records": len(records),
            }
        )
    if observed != expected:
        failures.append(
            {
                "field": "final_run_cartesian_product",
                "missing": sorted(expected - observed),
                "extra": sorted(observed - expected),
            }
        )

    allowed_batches = config["OOM_fallback"]["final_original_batch_sequence"]
    artifact_failures = []
    record_failures = []
    for architecture, spec, seed in sorted(expected):
        path = (
            OUT
            / "run_records"
            / FINAL_GROUP
            / architecture
            / spec
            / f"seed_{seed}.json"
        )
        if not path.is_file():
            record_failures.append({"path": str(path), "reason": "missing"})
            continue
        disk = read_json(path)
        matching = [
            row
            for row in records
            if row.get("model") == architecture
            and row.get("spec") == spec
            and int(row.get("seed", -1)) == seed
        ]
        if len(matching) != 1 or matching[0] != disk:
            record_failures.append(
                {
                    "path": str(path),
                    "reason": "result_record_mismatch_or_duplicate",
                }
            )
            continue
        if not verify_payload_sha256(disk):
            record_failures.append(
                {
                    "path": str(path),
                    "field": "record_payload_sha256",
                    "expected": "valid self-hash",
                    "actual": disk.get("record_payload_sha256"),
                }
            )
        identity = {
            "complete": True,
            "group": FINAL_GROUP,
            "model": architecture,
            "spec": spec,
            "seed": seed,
            "heldout_test_used_for_selection": False,
            "physical_ground_truth": False,
        }
        for field, value in identity.items():
            if disk.get(field) != value:
                record_failures.append(
                    {
                        "path": str(path),
                        "field": field,
                        "expected": value,
                        "actual": disk.get(field),
                    }
                )
        if disk.get("requested_protocol") != expected_protocol:
            record_failures.append(
                {
                    "path": str(path),
                    "field": "requested_protocol",
                    "expected": expected_protocol,
                    "actual": disk.get("requested_protocol"),
                }
            )
        logical_identity = disk.get("logical_identity")
        if not isinstance(logical_identity, dict):
            record_failures.append(
                {
                    "path": str(path),
                    "field": "logical_identity",
                    "expected": "dictionary",
                    "actual": type(logical_identity).__name__,
                }
            )
        else:
            logical_payload = dict(logical_identity)
            recorded_logical_hash = logical_payload.pop(
                "identity_sha256", None
            )
            recomputed_logical_hash = sha256_json(logical_payload)
            if (
                recorded_logical_hash != recomputed_logical_hash
                or disk.get("logical_identity_sha256")
                != recomputed_logical_hash
            ):
                record_failures.append(
                    {
                        "path": str(path),
                        "field": "logical_identity_sha256",
                        "expected": recomputed_logical_hash,
                        "actual": disk.get("logical_identity_sha256"),
                    }
                )
            logical_expected = {
                "config_sha256": sha256_file(
                    OUT / "preregistered_config.json"
                ),
                "model": architecture,
                "group": FINAL_GROUP,
                "spec": spec,
                "seed": seed,
                "protocol": expected_protocol,
            }
            for field, value in logical_expected.items():
                if logical_identity.get(field) != value:
                    record_failures.append(
                        {
                            "path": str(path),
                            "field": f"logical_identity.{field}",
                            "expected": value,
                            "actual": logical_identity.get(field),
                        }
                    )
        protocol = disk.get("protocol", {})
        for field, value in expected_protocol.items():
            actual = protocol.get(field)
            valid = (
                actual in allowed_batches
                if field == "batch_size"
                else actual == value
            )
            if not valid:
                record_failures.append(
                    {
                        "path": str(path),
                        "field": f"protocol.{field}",
                        "expected": (
                            allowed_batches if field == "batch_size" else value
                        ),
                        "actual": actual,
                    }
                )
        required_artifacts = {"checkpoint", "prediction", "curve", "runtime"}
        raw_artifact_hashes = disk.get("artifact_hashes")
        raw_artifact_paths = disk.get("artifact_paths")
        artifact_hashes = (
            raw_artifact_hashes
            if isinstance(raw_artifact_hashes, dict)
            else {}
        )
        artifact_paths = (
            raw_artifact_paths if isinstance(raw_artifact_paths, dict) else {}
        )
        if set(artifact_hashes) != required_artifacts:
            artifact_failures.append(
                {
                    "record": str(path),
                    "field": "artifact_hashes",
                    "expected": sorted(required_artifacts),
                    "actual": sorted(artifact_hashes),
                }
            )
        if set(artifact_paths) != required_artifacts:
            artifact_failures.append(
                {
                    "record": str(path),
                    "field": "artifact_paths",
                    "expected": sorted(required_artifacts),
                    "actual": sorted(artifact_paths),
                }
            )
        stem = Path(FINAL_GROUP) / architecture / spec / f"seed_{seed}"
        expected_artifact_paths = {
            "checkpoint": OUT
            / "checkpoints"
            / stem.with_suffix(".pt"),
            "prediction": OUT
            / "raw_predictions"
            / stem.with_suffix(".npz"),
            "curve": OUT
            / "training_curves"
            / stem.with_suffix(".json"),
            "runtime": OUT / "runtime" / stem.with_suffix(".json"),
        }
        for name in required_artifacts:
            digest = artifact_hashes.get(name)
            artifact_path = Path(artifact_paths.get(name, ""))
            expected_artifact_path = expected_artifact_paths[name].resolve()
            if (
                not isinstance(digest, str)
                or artifact_path != expected_artifact_path
                or not artifact_path.is_file()
                or sha256_file(artifact_path) != digest
            ):
                artifact_failures.append(
                    {
                        "record": str(path),
                        "artifact": name,
                        "path": str(artifact_path),
                        "expected_path": str(expected_artifact_path),
                    }
                )
        runtime_path = expected_artifact_paths["runtime"]
        if runtime_path.is_file():
            runtime = read_json(runtime_path)
            if (
                not isinstance(disk.get("resource_attempt_id"), str)
                or runtime.get("resource_attempt_id")
                != disk.get("resource_attempt_id")
            ):
                artifact_failures.append(
                    {
                        "record": str(path),
                        "artifact": "runtime",
                        "field": "resource_attempt_id",
                        "record_value": disk.get("resource_attempt_id"),
                        "runtime_value": runtime.get("resource_attempt_id"),
                    }
                )

    independent = iid.get("independent_metric_recomputation", {})
    independent_rows = independent.get("rows", [])
    independent_identities = {
        (row.get("architecture"), row.get("family"), row.get("seed"))
        for row in independent_rows
    }
    expected_independent_identities = {
        (architecture, family, seed)
        for architecture in ARCHITECTURES
        for family in EASY_FAMILIES
        for seed in SEEDS
    }
    if not (
        independent.get("pass") is True
        and independent.get("raw_prediction_files") == FINAL_RUNS
        and len(independent_rows) == FINAL_RUNS
        and independent_identities == expected_independent_identities
        and not independent.get("metric_mismatches")
        and not independent.get("calibration_scalar_mismatches")
        and not independent.get("artifact_failures")
    ):
        failures.append({"field": "independent_metric_recomputation"})

    cells = iid.get("cells", [])
    if len(cells) != FINAL_CELLS:
        failures.append({"field": "cells", "actual": len(cells)})
    recomputed_qualifying = []
    recomputed_calibration = []
    cell_failures = []
    expected_cell_ids = {
        (architecture, family)
        for architecture in ARCHITECTURES
        for family in EASY_FAMILIES
    }
    observed_cell_ids = {
        (cell.get("architecture"), cell.get("family")) for cell in cells
    }
    if observed_cell_ids != expected_cell_ids:
        cell_failures.append(
            {
                "field": "cell_cartesian_product",
                "missing": sorted(expected_cell_ids - observed_cell_ids),
                "extra": sorted(observed_cell_ids - expected_cell_ids),
            }
        )
    by_architecture: dict[str, dict[str, bool]] = defaultdict(dict)
    calibration_by_architecture: dict[str, dict[str, bool]] = defaultdict(dict)
    for cell in cells:
        seed_rows = sorted(cell.get("seed_rows", []), key=lambda row: row["seed"])
        if [row["seed"] for row in seed_rows] != list(SEEDS):
            cell_failures.append(
                {
                    "architecture": cell.get("architecture"),
                    "family": cell.get("family"),
                    "field": "seeds",
                }
            )
            continue
        passing = [
            row["seed"]
            for row in seed_rows
            if float(row["balanced_accuracy"]) >= FINAL_BA_GATE
        ]
        calibration_seeds = [
            row["seed"]
            for row in seed_rows
            if row.get("AUROC") is not None
            and float(row["AUROC"]) >= 0.70
            and float(row["balanced_accuracy"]) < FINAL_BA_GATE
            and float(row["oracle_threshold_diagnostic_gain"]) >= 0.10
        ]
        qualifies = len(passing) >= 2
        calibration_qualifies = len(calibration_seeds) >= 2
        if (
            cell.get("passing_seeds") != passing
            or cell.get("qualifies") is not qualifies
            or cell.get("calibration_failure_evidence_seeds")
            != calibration_seeds
            or cell.get("calibration_failure_cell_qualifies")
            is not calibration_qualifies
        ):
            cell_failures.append(
                {
                    "architecture": cell.get("architecture"),
                    "family": cell.get("family"),
                    "field": "cell_gate_recomputation",
                }
            )
        by_architecture[cell["architecture"]][cell["family"]] = qualifies
        calibration_by_architecture[cell["architecture"]][
            cell["family"]
        ] = calibration_qualifies

    for architecture in ARCHITECTURES:
        if any(by_architecture[architecture].values()):
            recomputed_qualifying.append(architecture)
        if any(calibration_by_architecture[architecture].values()):
            recomputed_calibration.append(architecture)
    per_architecture = iid.get("per_architecture", [])
    if (
        len(per_architecture) != len(ARCHITECTURES)
        or {row.get("architecture") for row in per_architecture}
        != set(ARCHITECTURES)
    ):
        failures.append({"field": "per_architecture"})
    else:
        for row in per_architecture:
            architecture = row["architecture"]
            qualifying_families = [
                family
                for family in EASY_FAMILIES
                if by_architecture[architecture][family]
            ]
            calibration_families = [
                family
                for family in EASY_FAMILIES
                if calibration_by_architecture[architecture][family]
            ]
            if (
                row.get("qualifying_families") != qualifying_families
                or row.get("qualifying_family_count")
                != len(qualifying_families)
                or row.get("qualifies") is not bool(qualifying_families)
                or row.get("calibration_failure_families")
                != calibration_families
                or row.get("calibration_failure_qualifies")
                is not bool(calibration_families)
            ):
                failures.append(
                    {
                        "field": "per_architecture_aggregation",
                        "architecture": architecture,
                    }
                )
    study_pass = len(recomputed_qualifying) >= 2
    calibration_pass = not study_pass and len(recomputed_calibration) >= 2
    if (
        iid.get("qualifying_architectures") != recomputed_qualifying
        or iid.get("qualifying_architecture_count")
        != len(recomputed_qualifying)
        or iid.get("study_pass") is not study_pass
        or iid.get("IID_gate", {}).get("study_pass") is not study_pass
    ):
        failures.append({"field": "IID_gate_recomputation"})
    calibration = iid.get("calibration_failure_diagnostic", {})
    if (
        calibration.get("qualifying_architectures")
        != recomputed_calibration
        or calibration.get("gate_pass") is not calibration_pass
        or calibration.get("oracle_test_threshold_used_for_selection")
        is not False
    ):
        failures.append({"field": "calibration_gate_recomputation"})
    if (
        iid.get("valid_for_terminal_decision") is not True
        or iid.get("test_used_for_selection") is not False
        or iid.get("additional_verifier_architectures") != 0
        or iid.get("physical_ground_truth") is not False
    ):
        failures.append({"field": "terminal_validity_flags"})

    failures.extend(cell_failures)
    failures.extend(record_failures)
    failures.extend(artifact_failures)
    return {
        "pass": not failures,
        "expected_runs": FINAL_RUNS,
        "observed_runs": len(records),
        "expected_cells": FINAL_CELLS,
        "observed_cells": len(cells),
        "qualifying_architectures": recomputed_qualifying,
        "study_pass": study_pass,
        "calibration_qualifying_architectures": recomputed_calibration,
        "calibration_gate_pass": calibration_pass,
        "metric_scalar_comparisons": independent.get(
            "metric_scalar_comparisons"
        ),
        "artifact_failures": artifact_failures,
        "failures": failures,
    }


def _optional_evidence(pattern: str) -> list[Path]:
    return sorted(
        path
        for path in OUT.rglob(pattern)
        if path.is_file()
        and path.name
        not in {
            "preregistered_config.json",
            "preregistered_config_initial.json",
            "root_cause_decision.json",
        }
    )


def _repair_evidence(config: dict[str, Any]) -> dict[str, Any]:
    paths = _optional_evidence("*repair*.json")
    events = [
        row
        for row in read_jsonl(OUT / "run_log.jsonl")
        if "REPAIR" in str(row.get("event", "")).upper()
    ]
    if not paths and not events:
        return {
            "applied": False,
            "valid": True,
            "records": [],
            "events": [],
        }
    records = [{"path": str(path), "payload": read_json(path)} for path in paths]
    applied_records = [
        row
        for row in records
        if row["payload"].get(
            "repair_applied", row["payload"].get("applied", False)
        )
        is True
    ]
    valid_records = []
    for row in applied_records:
        payload = row["payload"]
        before_after = bool(
            payload.get("before_after_evidence")
            or (
                payload.get("before") is not None
                and payload.get("after") is not None
            )
        )
        valid_records.append(
            bool(
                payload.get(
                    "confirmed_bug", payload.get("bug_confirmed", False)
                )
                and before_after
                and payload.get("source_study_modified", False) is False
                and payload.get(
                    "post_repair_runs_use_new_versioned_paths",
                    payload.get("versioned_outputs", False),
                )
            )
        )
    maximum = int(config["repair_policy"]["maximum_isolated_repairs"])
    valid = (
        len(applied_records) <= maximum
        and len(applied_records) == len(valid_records)
        and all(valid_records)
        and (not events or bool(applied_records))
    )
    return {
        "applied": bool(applied_records),
        "valid": valid,
        "maximum_allowed": maximum,
        "applied_records": len(applied_records),
        "records": records,
        "events": events,
    }


def _confirmed_fault_evidence() -> dict[str, Any]:
    paths = _optional_evidence("*fault*.json")
    records = [{"path": str(path), "payload": read_json(path)} for path in paths]
    valid = [
        row
        for row in records
        if row["payload"].get(
            "causally_reproduced",
            row["payload"].get("reproduced_causal_defect", False),
        )
        is True
        and row["payload"].get(
            "confirmed_fault",
            row["payload"].get("confirmed_bug", False),
        )
        is True
    ]
    return {
        "confirmed": bool(valid),
        "valid_records": valid,
        "records": records,
    }


def _map_by_architecture(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("architecture", row.get("model"))): row for row in rows
    }


def _compute_per_architecture(
    inputs: dict[str, Any],
    amendment: dict[str, Any],
    final_validation: dict[str, Any],
) -> dict[str, Any]:
    alignment = inputs["alignment"]
    sanity = inputs["sanity"]
    memorization = inputs["memorization"]
    gradient = inputs["gradient"]
    optimization = inputs["optimization"]
    representation = inputs["representation"]
    iid = inputs["iid"]

    collision: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in alignment["required_input_collision_audit"]:
        collision[(row["architecture"], row["family"])].append(
            float(row["empirical_exact_input_accuracy_and_BA_ceiling"])
        )
    gradient_map = {
        (row["architecture"], row["family"]): bool(row["pass"])
        for row in gradient["rows"]
    }
    tiny_map = {
        (row["architecture"], row["family"]): bool(row["pass"])
        for row in memorization["tiny_rows"]
    }
    full_map = {
        (row["architecture"], row["family"]): bool(row["pass"])
        for row in memorization["full_rows"]
    }
    sanity_maps = {
        name: _map_by_architecture(sanity[name]["rows"])
        for name in (
            "positive_observable_signal",
            "shuffled_label_negative",
            "constant_input_negative",
        )
    }
    alternate_map = {
        (row["architecture"], row["family"]): bool(row["qualifies"])
        for row in optimization["alternate_schedule"]["cells"]
    }
    control_map = {
        (row["control"], row["family"]): bool(row["qualifies"])
        for row in representation["diagnostic_controls"]["cells"]
    }
    final_map = {
        (row["architecture"], row["family"]): row for row in iid["cells"]
    }
    final_study_pass = bool(iid["study_pass"])
    engineered_global = bool(representation["engineered_control_support"])
    alignment_pipeline_pass = bool(
        alignment["pass"] and inputs["normalization"]["pass"]
    )

    cells = []
    architecture_rows = []
    for architecture in ARCHITECTURES:
        positive_pass = bool(
            sanity_maps["positive_observable_signal"][architecture][
                "sanity_pass"
            ]
        )
        shuffled_pass = bool(
            sanity_maps["shuffled_label_negative"][architecture]["sanity_pass"]
        )
        constant_pass = bool(
            sanity_maps["constant_input_negative"][architecture]["sanity_pass"]
        )
        pipeline_pass = positive_pass and shuffled_pass and constant_pass
        architecture_cells = []
        for family in EASY_FAMILIES:
            key = (architecture, family)
            ceiling = min(collision[key])
            gradient_pass = gradient_map[key]
            tiny_pass = tiny_map[key]
            full_pass = full_map[key]
            alternate_pass = alternate_map[key]
            engineered_pass = control_map[
                ("engineered_current_action_summary", family)
            ]
            flattened_pass = control_map[
                ("flattened_raw_current_action", family)
            ]
            final_cell = final_map[key]
            final_pass = bool(final_cell["qualifies"])
            calibration_pass = bool(
                final_cell["calibration_failure_cell_qualifies"]
            )

            conditions = {
                "exact_input_nonidentifiability": ceiling < FINAL_BA_GATE,
                "gradient_or_pipeline_failure": (
                    not gradient_pass or not pipeline_pass
                ),
                "calibration_failure": (
                    not final_study_pass and calibration_pass
                ),
                "optimization_failure": (
                    not final_pass and alternate_pass
                ),
                "representation_mismatch": (
                    not final_pass
                    and not alternate_pass
                    and alignment_pipeline_pass
                    and gradient_pass
                    and positive_pass
                    and tiny_pass
                    and full_pass
                    and engineered_global
                    and engineered_pass
                    and ceiling >= FINAL_BA_GATE
                ),
                "capacity_failure": (
                    not final_pass
                    and not alternate_pass
                    and (not tiny_pass or not full_pass)
                    and ceiling >= 0.95
                ),
                "IID_learned": final_pass,
                "insufficient": True,
            }
            diagnosis = next(name for name in CELL_ORDER if conditions[name])
            row = {
                "architecture": architecture,
                "family": family,
                "spec": f"within_{family.lower()}",
                "diagnosis": diagnosis,
                "first_true_rule": diagnosis,
                "rule_order": list(CELL_ORDER),
                "evidence": {
                    "minimum_exact_input_BA_ceiling": ceiling,
                    "gradient_pass": gradient_pass,
                    "positive_sanity_pass": positive_pass,
                    "shuffled_sanity_pass": shuffled_pass,
                    "constant_sanity_pass": constant_pass,
                    "pipeline_controls_pass": pipeline_pass,
                    "alignment_and_normalization_pass": (
                        alignment_pipeline_pass
                    ),
                    "tiny_memorization_pass": tiny_pass,
                    "full_memorization_pass": full_pass,
                    "alternate_schedule_cell_qualifies": alternate_pass,
                    "engineered_control_cell_qualifies": engineered_pass,
                    "flattened_control_cell_qualifies": flattened_pass,
                    "final_confirmation_cell_qualifies": final_pass,
                    "final_passing_seeds": final_cell["passing_seeds"],
                    "calibration_failure_cell_qualifies": calibration_pass,
                    "calibration_evidence_seeds": final_cell[
                        "calibration_failure_evidence_seeds"
                    ],
                },
            }
            cells.append(row)
            architecture_cells.append(row)
        counts = Counter(row["diagnosis"] for row in architecture_cells)
        supported = [
            name for name in CELL_ORDER if counts.get(name, 0) >= 2
        ]
        architecture_diagnosis = (
            supported[0]
            if len(supported) == 1
            else "heterogeneous_or_insufficient"
        )
        architecture_rows.append(
            {
                "architecture": architecture,
                "family_cells": architecture_cells,
                "diagnosis_counts": {
                    name: counts.get(name, 0) for name in CELL_ORDER
                },
                "labels_supported_by_at_least_two_families": supported,
                "architecture_diagnosis": architecture_diagnosis,
                "aggregation_rule": (
                    "one label only when it occurs in at least two families "
                    "and no second label independently reaches two families"
                ),
            }
        )
    return {
        "schema": "vild-per-architecture-diagnosis-v1",
        "study_id": STUDY_ID,
        "cell_order": list(CELL_ORDER),
        "architecture_aggregation": inputs["config"][
            "per_architecture_diagnosis"
        ]["architecture_aggregation"],
        "cells": cells,
        "cell_count": len(cells),
        "architectures": architecture_rows,
        "architecture_count": len(architecture_rows),
        "final_confirmation_study_pass": final_study_pass,
        "final_confirmation_validation": final_validation,
        "input_hashes": {
            name: sha256_file(OUT / name) for name in DIAGNOSTIC_INPUTS
        },
        "construction_labels_only": True,
        "physical_ground_truth": False,
    }


def _capacity_support(
    inputs: dict[str, Any],
) -> dict[str, Any]:
    collision: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in inputs["alignment"]["required_input_collision_audit"]:
        collision[(row["architecture"], row["family"])].append(
            float(row["empirical_exact_input_accuracy_and_BA_ceiling"])
        )
    tiny = {
        (row["architecture"], row["family"]): bool(row["pass"])
        for row in inputs["memorization"]["tiny_rows"]
    }
    full = {
        (row["architecture"], row["family"]): bool(row["pass"])
        for row in inputs["memorization"]["full_rows"]
    }
    rows = []
    qualifying = []
    for architecture in ARCHITECTURES:
        families = []
        for family in EASY_FAMILIES:
            key = (architecture, family)
            if (
                (not tiny[key] or not full[key])
                and min(collision[key]) >= 0.95
            ):
                families.append(family)
        if len(families) >= 2:
            qualifying.append(architecture)
        rows.append(
            {
                "architecture": architecture,
                "memorization_failure_families_with_ceiling_ge_0_95": families,
                "qualifies": len(families) >= 2,
            }
        )
    return {
        "per_architecture": rows,
        "qualifying_architectures": qualifying,
        "qualifying_architecture_count": len(qualifying),
        "gate_pass": len(qualifying) >= 2,
    }


def _memorization_support(inputs: dict[str, Any]) -> dict[str, Any]:
    rows = inputs["memorization"]["architecture_summary"]
    qualifying = [
        row["architecture"]
        for row in rows
        if row["all_tiny_pass"] and row["all_full_pass"]
    ]
    return {
        "qualifying_architectures": qualifying,
        "qualifying_architecture_count": len(qualifying),
        "at_least_four_architectures": len(qualifying) >= 4,
    }


def _compute_root_decision(
    inputs: dict[str, Any],
    amendment: dict[str, Any],
    diagnostics: dict[str, Any],
    final_validation: dict[str, Any],
    per_architecture: dict[str, Any],
    repair: dict[str, Any],
    fault: dict[str, Any],
) -> dict[str, Any]:
    iid = inputs["iid"]
    alignment_pass = bool(
        inputs["alignment"]["pass"] and inputs["normalization"]["pass"]
    )
    sanity_pass = bool(inputs["sanity"]["pass"])
    positive_sanity_pass = bool(
        inputs["sanity"]["positive_observable_signal"]["pass"]
    )
    gradient_pass = bool(inputs["gradient"]["pass"])
    alternate_pass = bool(
        inputs["optimization"]["alternate_schedule"]["study_pass"]
    )
    engineered_support = bool(
        inputs["representation"]["engineered_control_support"]
    )
    exact_ceiling_minimum = min(
        float(row["empirical_exact_input_accuracy_and_BA_ceiling"])
        for row in inputs["alignment"]["required_input_collision_audit"]
    )
    memorization = _memorization_support(inputs)
    capacity = _capacity_support(inputs)
    final_pass = bool(iid["study_pass"])
    calibration_pass = bool(
        iid["calibration_failure_diagnostic"]["gate_pass"]
    )
    pipeline_control_failure = not sanity_pass
    reproduced_defect = bool(fault["confirmed"])

    conditions = {
        "VILD_IMPLEMENTATION_FAULT_REPAIRED_AND_IID_RESTORED": (
            repair["applied"] and repair["valid"] and final_pass
        ),
        "VILD_ALIGNMENT_OR_IMPLEMENTATION_FAULT_CONFIRMED": (
            not final_pass
            and not repair["applied"]
            and (
                not alignment_pass
                or (pipeline_control_failure and reproduced_defect)
            )
        ),
        "VILD_IID_LEARNABILITY_RESTORED_WITHOUT_REPAIR": (
            not repair["applied"] and final_pass
        ),
        "VILD_CALIBRATION_THRESHOLD_FAILURE": (
            not final_pass and calibration_pass
        ),
        "VILD_OPTIMIZATION_TRAINING_FAILURE": (
            not final_pass
            and alignment_pass
            and sanity_pass
            and alternate_pass
        ),
        "VILD_REPRESENTATION_INDUCTIVE_BIAS_MISMATCH": (
            not final_pass
            and not alternate_pass
            and alignment_pass
            and gradient_pass
            and positive_sanity_pass
            and memorization["at_least_four_architectures"]
            and engineered_support
            and exact_ceiling_minimum >= FINAL_BA_GATE
        ),
        "VILD_CAPACITY_OR_ARCHITECTURE_FAILURE": (
            not final_pass
            and not alternate_pass
            and alignment_pass
            and gradient_pass
            and capacity["gate_pass"]
        ),
        "VILD_INSUFFICIENT_OR_HETEROGENEOUS_EVIDENCE": True,
    }
    order = tuple(inputs["config"]["root_cause_decision_order"])
    if order != EXPECTED_DECISION_ORDER:
        raise RuntimeError(f"unexpected frozen decision order: {order}")
    decision = next(name for name in order if conditions[name])
    evidence = {
        "amended_preregistration_and_snapshots_pass": amendment["pass"],
        "diagnostic_completeness_pass": diagnostics["pass"],
        "final_confirmation_integrity_pass": final_validation["pass"],
        "repair": repair,
        "confirmed_fault_evidence": fault,
        "alignment_and_normalization_pass": alignment_pass,
        "sanity_controls_pass": sanity_pass,
        "positive_sanity_pass": positive_sanity_pass,
        "gradient_flow_pass": gradient_pass,
        "alternate_schedule_study_pass": alternate_pass,
        "engineered_control_support": engineered_support,
        "minimum_exact_input_BA_ceiling": exact_ceiling_minimum,
        "memorization_support": memorization,
        "capacity_support": capacity,
        "final_confirmation_study_pass": final_pass,
        "final_qualifying_architectures": iid[
            "qualifying_architectures"
        ],
        "calibration_failure_gate_pass": calibration_pass,
        "calibration_qualifying_architectures": iid[
            "calibration_failure_diagnostic"
        ]["qualifying_architectures"],
    }
    rule_rows = [
        {
            "order": index,
            "decision": name,
            "proven": bool(conditions[name]),
            "selected": name == decision,
            "frozen_rule": inputs["config"]["decision_rules"].get(
                {
                    "VILD_IMPLEMENTATION_FAULT_REPAIRED_AND_IID_RESTORED": (
                        "implementation_fault_repaired_and_restored"
                    ),
                    "VILD_ALIGNMENT_OR_IMPLEMENTATION_FAULT_CONFIRMED": (
                        "unrepaired_alignment_or_implementation_fault"
                    ),
                    "VILD_IID_LEARNABILITY_RESTORED_WITHOUT_REPAIR": (
                        "iid_restored_without_repair"
                    ),
                    "VILD_CALIBRATION_THRESHOLD_FAILURE": (
                        "calibration_threshold_failure"
                    ),
                    "VILD_OPTIMIZATION_TRAINING_FAILURE": (
                        "optimization_training_failure"
                    ),
                    "VILD_REPRESENTATION_INDUCTIVE_BIAS_MISMATCH": (
                        "representation_inductive_bias_mismatch"
                    ),
                    "VILD_CAPACITY_OR_ARCHITECTURE_FAILURE": (
                        "capacity_or_architecture_failure"
                    ),
                    "VILD_INSUFFICIENT_OR_HETEROGENEOUS_EVIDENCE": (
                        "insufficient_or_heterogeneous"
                    ),
                }[name]
            ),
        }
        for index, name in enumerate(order, start=1)
    ]
    return {
        "schema": "vild-root-cause-decision-v1",
        "study_id": STUDY_ID,
        "decision": decision,
        "terminal": True,
        "decision_order": list(order),
        "rule_evaluations": rule_rows,
        "evidence": evidence,
        "per_architecture_diagnosis": [
            {
                "architecture": row["architecture"],
                "diagnosis": row["architecture_diagnosis"],
                "diagnosis_counts": row["diagnosis_counts"],
            }
            for row in per_architecture["architectures"]
        ],
        "input_hashes": {
            "iid_learnability_results.json": sha256_file(
                OUT / "iid_learnability_results.json"
            ),
            "per_architecture_diagnosis.json": sha256_file(
                OUT / "per_architecture_diagnosis.json"
            ),
            **{
                name: sha256_file(OUT / name)
                for name in DIAGNOSTIC_INPUTS[:-1]
            },
        },
        "label_semantics": inputs["config"]["claim_boundary"]["labels"],
        "physical_ground_truth": False,
        "robot_safety_claim": False,
        "source_study_modified": False,
        "additional_verifier_architectures": 0,
    }


DECISION_EXPLANATIONS = {
    "VILD_IMPLEMENTATION_FAULT_REPAIRED_AND_IID_RESTORED": (
        "A causally isolated implementation defect was documented, the single "
        "allowed repair used versioned new outputs, and the frozen final IID "
        "confirmation gate was restored."
    ),
    "VILD_ALIGNMENT_OR_IMPLEMENTATION_FAULT_CONFIRMED": (
        "The final IID gate did not pass and the ordered evidence confirms an "
        "alignment invariant failure or a reproduced causal pipeline defect."
    ),
    "VILD_IID_LEARNABILITY_RESTORED_WITHOUT_REPAIR": (
        "No repair was applied and the unchanged five-architecture final IID "
        "confirmation now satisfies the frozen two-architecture gate."
    ),
    "VILD_CALIBRATION_THRESHOLD_FAILURE": (
        "The final IID gate did not pass, while at least two architectures meet "
        "the same-cell, two-seed ranking and oracle-gap calibration diagnostic."
    ),
    "VILD_OPTIMIZATION_TRAINING_FAILURE": (
        "Alignment and sanity controls pass, the original final schedule does "
        "not, and the unchanged-architecture alternate schedule satisfies the "
        "frozen two-seed study gate."
    ),
    "VILD_REPRESENTATION_INDUCTIVE_BIAS_MISMATCH": (
        "Final and alternate fair verifiers do not pass despite alignment, "
        "gradients, positive controls, memorization, exact-input ceilings, and "
        "engineered deterministic input controls supporting learnability."
    ),
    "VILD_CAPACITY_OR_ARCHITECTURE_FAILURE": (
        "Final and alternate schedules do not pass, and at least two "
        "architectures fail tiny or full memorization in at least two families "
        "despite exact-input ceilings of at least 0.95."
    ),
    "VILD_INSUFFICIENT_OR_HETEROGENEOUS_EVIDENCE": (
        "No earlier frozen rule is proven across the required architecture and "
        "family support; the evidence is insufficient or heterogeneous."
    ),
}


DECISION_EXPLANATIONS_ZH = {
    "VILD_IMPLEMENTATION_FAULT_REPAIRED_AND_IID_RESTORED": (
        "已因果隔离并记录一个实现缺陷；唯一允许的修复只写入版本化新产物，且冻结的 "
        "IID 最终确认门恢复通过。"
    ),
    "VILD_ALIGNMENT_OR_IMPLEMENTATION_FAULT_CONFIRMED": (
        "最终 IID 门未通过，并确认了对齐不变量失败或可复现的因果流水线缺陷。"
    ),
    "VILD_IID_LEARNABILITY_RESTORED_WITHOUT_REPAIR": (
        "没有应用修复，保持五个架构不变的最终 IID 确认已满足冻结门槛。"
    ),
    "VILD_CALIBRATION_THRESHOLD_FAILURE": (
        "最终 IID 门未通过，但至少两个架构在同一 family cell、至少两个种子上满足 "
        "冻结的排序与 oracle-gap 校准诊断。"
    ),
    "VILD_OPTIMIZATION_TRAINING_FAILURE": (
        "对齐与 sanity controls 通过；原最终训练日程失败，而架构不变的替代日程通过 "
        "冻结的两种子研究门。"
    ),
    "VILD_REPRESENTATION_INDUCTIVE_BIAS_MISMATCH": (
        "最终和替代 fair verifier 均失败，但对齐、梯度、正控制、记忆、精确输入上限 "
        "和确定性的 engineered controls 共同支持输入可学，指向表示/归纳偏置不匹配。"
    ),
    "VILD_CAPACITY_OR_ARCHITECTURE_FAILURE": (
        "最终和替代日程均失败，且至少两个架构在至少两个 family 上不能记忆训练集， "
        "尽管精确输入 BA 上限不低于 0.95。"
    ),
    "VILD_INSUFFICIENT_OR_HETEROGENEOUS_EVIDENCE": (
        "没有任何更早的冻结规则获得足够的跨架构、跨 family 证据；证据不足或异质。"
    ),
}


def _resource_status() -> dict[str, Any]:
    # The append-only ledger, unlike successful runtime files, also accounts
    # for OOM and failed attempts.  Reuse the training-side verifier so the
    # terminal decision applies the identical frozen accounting semantics.
    from .training import resource_status

    return resource_status()


def _forbidden_artifact_audit(inputs: dict[str, Any]) -> dict[str, Any]:
    forbidden_paths = []
    for path in OUT.rglob("*"):
        if not path.is_file():
            continue
        relative = str(path.relative_to(OUT)).lower()
        if any(token in relative for token in FORBIDDEN_PATH_TOKENS):
            forbidden_paths.append(str(path.relative_to(OUT)))
    record_issues = []
    run_record_root = OUT / "run_records"
    if run_record_root.is_dir():
        for path in sorted(run_record_root.rglob("*.json")):
            row = read_json(path)
            # Gradient diagnostics use their own frozen run-record schema.
            # Their group is encoded by the schema/path and the architecture
            # lives in logical_identity, unlike ordinary training records.
            if row.get("schema") == "vild-gradient-run-record-v1":
                group = "gradient"
                model = row.get("logical_identity", {}).get("architecture")
            else:
                group = row.get("group")
                model = row.get("model")
            if group not in ALLOWED_TRAINING_GROUPS:
                record_issues.append(
                    {
                        "path": str(path.relative_to(OUT)),
                        "field": "group",
                        "actual": group,
                    }
                )
            if model not in ALLOWED_MODELS:
                record_issues.append(
                    {
                        "path": str(path.relative_to(OUT)),
                        "field": "model",
                        "actual": model,
                    }
                )
            if row.get("heldout_test_used_for_selection") is True:
                record_issues.append(
                    {
                        "path": str(path.relative_to(OUT)),
                        "field": "heldout_test_used_for_selection",
                    }
                )
            if row.get("physical_ground_truth") is True:
                record_issues.append(
                    {
                        "path": str(path.relative_to(OUT)),
                        "field": "physical_ground_truth",
                    }
                )
    flags = {
        "additional_verifier_architectures": inputs["iid"].get(
            "additional_verifier_architectures"
        ),
        "test_used_for_selection": inputs["iid"].get(
            "test_used_for_selection"
        ),
        "physical_ground_truth": inputs["iid"].get("physical_ground_truth"),
    }
    passed = (
        not forbidden_paths
        and not record_issues
        and flags["additional_verifier_architectures"] == 0
        and flags["test_used_for_selection"] is False
        and flags["physical_ground_truth"] is False
    )
    return {
        "pass": passed,
        "forbidden_path_tokens": list(FORBIDDEN_PATH_TOKENS),
        "forbidden_paths": forbidden_paths,
        "run_record_issues": record_issues,
        "result_flags": flags,
        "unrelated_process_termination_performed": False,
    }


def _final_table(iid: dict[str, Any]) -> str:
    lines = []
    for row in iid["per_architecture"]:
        families = ", ".join(row["qualifying_families"]) or "none"
        summary = row["balanced_accuracy_across_12_runs"]
        lines.append(
            "| `{}` | {} | {:.3f} | {:.3f}–{:.3f} |".format(
                row["architecture"],
                families,
                summary["mean"],
                summary["minimum"],
                summary["maximum"],
            )
        )
    return "\n".join(lines)


def _diagnosis_table(per_architecture: dict[str, Any]) -> str:
    return "\n".join(
        "| `{}` | `{}` | {} |".format(
            row["architecture"],
            row["architecture_diagnosis"],
            ", ".join(row["labels_supported_by_at_least_two_families"])
            or "none",
        )
        for row in per_architecture["architectures"]
    )


def _rule_table(root: dict[str, Any]) -> str:
    return "\n".join(
        "| {} | `{}` | {} | {} |".format(
            row["order"],
            row["decision"],
            "yes" if row["proven"] else "no",
            "selected" if row["selected"] else "",
        )
        for row in root["rule_evaluations"]
    )


def _final_report(
    inputs: dict[str, Any],
    per_architecture: dict[str, Any],
    root: dict[str, Any],
    amendment: dict[str, Any],
    diagnostics: dict[str, Any],
    final_validation: dict[str, Any],
    forbidden: dict[str, Any],
) -> str:
    iid = inputs["iid"]
    outcome = diagnostics["diagnostic_outcomes"]
    resources = _resource_status()
    return f"""
# VERIFIER-IID-LEARNABILITY-DIAG-V1 final report

## Terminal decision

**`{root['decision']}`**

{DECISION_EXPLANATIONS[root['decision']]}

The decision follows the amended v1.1 order exactly. No later rule can
override an earlier proven rule, and no single architecture is forced into a
global root cause.

## Scope and evidence boundary

This study diagnoses why five frozen fair verifier architectures did or did
not learn the historical `N2/N3/N5/N6` within-family IID construction labels.
Those labels are **not physical failure truth**, task-success truth, or robot
safety evidence. No new physical-validity label, robot mask, SAM3 output,
world model, mitigation model, or additional verifier architecture was
created.

The source `VERIFIER-SHORTCUT-AUDIT-V1` tree remains byte-identical to its
preservation snapshot: {amendment['source_snapshot']['expected_files']} files,
{inputs['config']['source_study']['snapshot_bytes'] / 1024**3:.3f} GiB.
The pre-training amendment, its parent config, all frozen implementation
hashes, and the amended config hash pass.

## Completed diagnostics

- Alignment audit: **{'PASS' if inputs['alignment']['pass'] else 'FAIL'}**,
  over 12,676 rows with exact action-summary recomputation and complete
  architecture/input collision ceilings.
- Normalization/split/batch audit:
  **{'PASS' if inputs['normalization']['pass'] else 'FAIL'}**; statistics are
  train-only and held-out test data were not used for statistics or selection.
- Positive, shuffled-label, and constant-input controls: overall
  **{'PASS' if outcome['sanity_pass'] else 'FAIL'}**.
- Tiny/full memorization:
  {inputs['memorization']['tiny_passes']}/20 and
  {inputs['memorization']['full_passes']}/20 cells pass.
- Gradient flow: {inputs['gradient']['passes']}/20 cells pass.
- Alternate unchanged-architecture schedule:
  **{'PASS' if outcome['alternate_study_pass'] else 'FAIL'}**.
- Engineered deterministic input-control support:
  **{'PASS' if outcome['engineered_control_support'] else 'FAIL'}**.
- Diagnostic completeness checks: **{'PASS' if diagnostics['pass'] else 'FAIL'}**.

## Frozen final confirmation

Exactly 5 architectures x 4 easy families x 3 seeds = 60 final runs were
completed. The final result contains 20 architecture-family cells.
Independent recomputation reread every raw prediction, refit validation-only
temperature, reselected the validation-only BA threshold, and recomputed all
stored metrics. It checked
{final_validation['metric_scalar_comparisons']} scalar metric values with no
accepted mismatch. Held-out tests and oracle test thresholds were never used
for selection.

| Architecture | Qualifying families | Mean BA over 12 runs | BA range |
|---|---|---:|---:|
{_final_table(iid)}

Final IID study gate:
**{'PASS' if iid['study_pass'] else 'FAIL'}**;
{iid['qualifying_architecture_count']} qualifying architectures
(`{', '.join(iid['qualifying_architectures']) or 'none'}`), with 2 required.

Calibration-failure diagnostic:
**{'PASS' if iid['calibration_failure_diagnostic']['gate_pass'] else 'FAIL'}**;
oracle thresholds are diagnostic only.

## Per-architecture diagnosis

Cells use the frozen order: exact-input non-identifiability, gradient/pipeline,
calibration, optimization, representation, capacity, IID learned, then
insufficient. An architecture receives a label only when exactly one cell
label occurs in at least two families; otherwise it remains heterogeneous.

| Architecture | Aggregated diagnosis | Labels supported by >=2 families |
|---|---|---|
{_diagnosis_table(per_architecture)}

The full 20-cell evidence table is machine-readable in
`per_architecture_diagnosis.json`.

## Ordered root-cause gate

| Order | Candidate decision | Proven | Outcome |
|---:|---|---|---|
{_rule_table(root)}

The selected result is `{root['decision']}`. Its interpretation is limited to
the frozen architectures, inputs, training protocols, families, and
construction labels.

## Integrity and resources

- Amended protocol/source/implementation integrity:
  **{'PASS' if amendment['pass'] else 'FAIL'}**.
- 60-run final confirmation integrity:
  **{'PASS' if final_validation['pass'] else 'FAIL'}**.
- Forbidden-artifact and forbidden-input audit:
  **{'PASS' if forbidden['pass'] else 'FAIL'}**.
- Recorded GPU time: {resources['GPU_hours']:.4f} hours / 5.0-hour cap.
- Recursive new artifacts before terminal-log append:
  {resources['output_GiB']:.3f} GiB / 10.0-GiB cap.
- Resource ledger hash chain, closed-attempt state, and current-record coverage:
  **{'PASS' if resources['resource_ledger_pass'] and not resources['open_GPU_attempts'] and resources['resource_record_coverage_pass'] else 'FAIL'}**.
- Source-study modification: none.
- Unrelated GPU process termination: none.

The reproducibility manifest recursively records immutable outputs, complete
study code, frozen implementation rows, input hashes, source snapshot
verification, resources, and the pre-terminal run-log snapshot. The manifest
excludes itself and the append-only run log to avoid recursive hashes; the
final log event binds the manifest, root decision, per-architecture diagnosis,
and 60-run result hashes.

## Claim boundary

Passing an IID construction-label gate does not establish physical action
validity or safety. Failing it does not establish that a robot action is
invalid. Engineered-control success identifies learnability of deterministic
functions of frozen inputs, not causal physical verification. Test-oracle
thresholds are diagnostics only. Any follow-up must use a separately
preregistered branch and preserve this study unchanged.
"""


def _executive_zh(
    inputs: dict[str, Any],
    root: dict[str, Any],
    final_validation: dict[str, Any],
) -> str:
    iid = inputs["iid"]
    return f"""
# 执行摘要

最终决定：**`{root['decision']}`**。

{DECISION_EXPLANATIONS_ZH[root['decision']]}

本研究严格使用 amended v1.1 冻结顺序：完成 5 个原始 verifier 架构、4 个
easy families、3 个冻结种子的 60 次最终确认，以及 20 个 architecture-family
cells。独立复算读取全部 raw predictions，重新拟合 validation-only temperature、
重新选择 validation-only BA threshold，并核对
{final_validation['metric_scalar_comparisons']} 个指标标量。最终 IID gate
{'通过' if iid['study_pass'] else '未通过'}，合格架构为
{iid['qualifying_architecture_count']}/2：
`{', '.join(iid['qualifying_architectures']) or 'none'}`。

所有证据都只针对历史 benchmark-construction labels；它们不是物理失败真值、
任务成功真值或机器人安全证据。没有使用 test/oracle threshold 做模型选择，
没有增加 verifier 架构，没有运行 mitigation、SAM3 或 world model，也没有生成
机器人 mask 或物理有效性标签。源 VSA 研究保持字节级不变。

完整 cell 诊断见 `per_architecture_diagnosis.json`，冻结根因门见
`root_cause_decision.json`。下一步必须依据 `next_steps.md` 另开预注册分支；
不得回写、调参或重解释本终局产物。
"""


def _next_steps(root: dict[str, Any]) -> str:
    decision = root["decision"]
    actions = {
        "VILD_IMPLEMENTATION_FAULT_REPAIRED_AND_IID_RESTORED": [
            "Preserve the original and versioned repaired outputs plus before/after evidence.",
            "Independently reproduce the repaired IID result without changing this branch.",
            "Only then preregister a fresh shortcut audit; do not retroactively replace the VSA result.",
        ],
        "VILD_ALIGNMENT_OR_IMPLEMENTATION_FAULT_CONFIRMED": [
            "Freeze a minimal causal reproducer and the exact failed invariant.",
            "Implement at most one isolated repair in a new versioned branch with before/after evidence.",
            "Repeat all sanity, memorization, gradient, and 60-run confirmation gates before interpretation.",
        ],
        "VILD_IID_LEARNABILITY_RESTORED_WITHOUT_REPAIR": [
            "Independently repeat the unchanged final protocol to measure reproducibility.",
            "Audit why the historical run differed without selecting explanations on test outcomes.",
            "Preregister a new shortcut audit before making any learned-shortcut or mitigation claim.",
        ],
        "VILD_CALIBRATION_THRESHOLD_FAILURE": [
            "Preregister calibration alternatives using validation data only.",
            "Keep test-oracle thresholds diagnostic and prohibit their use for selection.",
            "Repeat the full three-seed IID gate before any broader verifier claim.",
        ],
        "VILD_OPTIMIZATION_TRAINING_FAILURE": [
            "Preregister an optimizer/schedule comparison while keeping architectures and inputs fixed.",
            "Select exclusively on frozen validation criteria and retain negative sanity controls.",
            "Confirm the schedule result across all families and seeds before a new shortcut audit.",
        ],
        "VILD_REPRESENTATION_INDUCTIVE_BIAS_MISMATCH": [
            "Preregister representation/inductive-bias candidates as a new architecture study.",
            "Require information-equivalent controls, memorization, gradients, and three-seed IID confirmation.",
            "Do not call deterministic summary-control performance physical action verification.",
        ],
        "VILD_CAPACITY_OR_ARCHITECTURE_FAILURE": [
            "Preregister capacity and architecture diagnostics with parameter-count and memorization controls.",
            "Keep the five frozen architectures and this terminal result unchanged as baselines.",
            "Authorize no shortcut mitigation until a fresh IID prerequisite is established.",
        ],
        "VILD_INSUFFICIENT_OR_HETEROGENEOUS_EVIDENCE": [
            "Do not force a single global root cause from conflicting architecture/family cells.",
            "Preregister targeted replications for the unresolved cells and explicit stopping rules.",
            "Collect additional behavioral evidence separately if the target claim is physical validity.",
        ],
    }[decision]
    numbered = "\n".join(
        f"{index}. {action}" for index, action in enumerate(actions, start=1)
    )
    return f"""
# Next steps

Terminal decision: **`{decision}`**.

{numbered}

For every branch:

- preserve `verifier_iid_learnability_diag_v1` and the source VSA study
  byte-identically;
- do not use held-out test or oracle thresholds for selection;
- do not run `ShortcutResistantVerifierTraining` from this diagnosis;
- do not create physical-validity labels or claim robot safety;
- freeze seeds, metrics, resources, repairs, and decision order before new
  training.
"""


def _package_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for name in ("numpy", "torch", "scipy", "pandas", "pyarrow"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def _environment() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "python_expected": str(PYTHON),
        "python_observed": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": _package_versions(),
    }
    try:
        import torch

        payload["torch"] = {
            "version": torch.__version__,
            "CUDA_build": torch.version.cuda,
            "CUDA_available": bool(torch.cuda.is_available()),
            "devices": [
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "total_memory_bytes": int(
                        torch.cuda.get_device_properties(index).total_memory
                    ),
                }
                for index in range(torch.cuda.device_count())
            ],
        }
    except Exception as error:  # pragma: no cover - environment fallback
        payload["torch_probe_error"] = f"{type(error).__name__}: {error}"
    return payload


def _build_manifest(
    inputs: dict[str, Any],
    amendment: dict[str, Any],
    diagnostics: dict[str, Any],
    final_validation: dict[str, Any],
    forbidden: dict[str, Any],
    root: dict[str, Any],
) -> dict[str, Any]:
    output_paths = [
        path
        for path in sorted(OUT.rglob("*"))
        if path.is_file()
        and str(path.relative_to(OUT)) not in MANIFEST_EXCLUSIONS
    ]
    code_root = (
        ROOT / "actmask" / "experiments" / "verifier_iid_learnability_diag"
    )
    study_code = sorted(code_root.rglob("*.py"))
    frozen_external = [
        ROOT / row["path"]
        for row in inputs["config"]["implementation_freeze"]
        if not (ROOT / row["path"]).is_relative_to(code_root)
    ]
    code_paths = sorted(set(study_code + frozen_external))
    input_names = AMENDMENT_FILES + DIAGNOSTIC_INPUTS + (
        "per_architecture_diagnosis.json",
        "root_cause_decision.json",
        "final_report.md",
        "executive_summary_zh.md",
        "next_steps.md",
    )
    log_path = OUT / "run_log.jsonl"
    resources = _resource_status()
    required_nonrecursive = [
        name
        for name in REQUIRED_OUTPUTS
        if name not in {"reproducibility_manifest.json", "run_log.jsonl"}
    ]
    return {
        "schema": "vild-reproducibility-manifest-v1",
        "study_id": STUDY_ID,
        "terminal_decision": root["decision"],
        "inventory_policy": {
            "recursive_output_inventory": True,
            "output_root": str(OUT.resolve()),
            "exclusions": {
                "reproducibility_manifest.json": (
                    "self-hash recursion has no stable representation"
                ),
                "run_log.jsonl": (
                    "append-only log receives the terminal event after the "
                    "manifest is written"
                ),
            },
            "all_other_regular_output_files_hashed": True,
        },
        "output_files": [_entry(path, OUT) for path in output_paths],
        "output_file_count": len(output_paths),
        "output_bytes_excluding_manifest_and_run_log": sum(
            path.stat().st_size for path in output_paths
        ),
        "implementation_files": [_entry(path, ROOT) for path in code_paths],
        "implementation_file_count": len(code_paths),
        "frozen_training_implementation": inputs["config"][
            "implementation_freeze"
        ],
        "input_hashes": [
            _entry(OUT / name, OUT) for name in input_names
        ],
        "source_preservation": {
            "source_root": str(SOURCE.resolve()),
            "snapshot_path": "source_preservation_snapshot.json",
            "snapshot_sha256": sha256_file(
                OUT / "source_preservation_snapshot.json"
            ),
            "tree_snapshot_sha256": amendment["source_snapshot"][
                "snapshot_sha256"
            ],
            "verified": amendment["source_snapshot"],
        },
        "append_only_run_log_before_terminal_event": {
            "path": "run_log.jsonl",
            "bytes": log_path.stat().st_size,
            "sha256": sha256_file(log_path),
            "records": len(read_jsonl(log_path)),
            "policy": (
                "terminal event must be the only appended content and must bind "
                "the resulting manifest SHA256"
            ),
        },
        "required_outputs": {
            "names": list(REQUIRED_OUTPUTS),
            "nonrecursive_rows": [
                _entry(OUT / name, OUT) for name in required_nonrecursive
            ],
            "manifest_validated_by_schema_and_postwrite_verification": True,
            "run_log_validated_by_hash_chain_and_terminal_event": True,
        },
        "final_confirmation": {
            "runs": final_validation["observed_runs"],
            "cells": final_validation["observed_cells"],
            "independent_metric_recomputation_pass": final_validation["pass"],
            "qualifying_architectures": final_validation[
                "qualifying_architectures"
            ],
            "study_pass": final_validation["study_pass"],
        },
        "integrity": {
            "amended_preregistration_implementation_source": amendment,
            "diagnostic_completeness": diagnostics,
            "forbidden_artifact_audit": forbidden,
        },
        "resources_before_manifest_and_terminal_event": resources,
        "resource_limits": inputs["config"]["resource_limits"],
        "environment": _environment(),
        "claim_boundary": inputs["config"]["claim_boundary"],
    }


def _verify_manifest() -> dict[str, Any]:
    manifest = read_json(OUT / "reproducibility_manifest.json")
    failures = []
    for row in manifest.get("output_files", []):
        path = OUT / row["path"]
        if (
            not path.is_file()
            or path.stat().st_size != row["bytes"]
            or sha256_file(path) != row["sha256"]
        ):
            failures.append({"scope": "output", "path": row["path"]})
    for row in manifest.get("implementation_files", []):
        path = ROOT / row["path"]
        if (
            not path.is_file()
            or path.stat().st_size != row["bytes"]
            or sha256_file(path) != row["sha256"]
        ):
            failures.append({"scope": "implementation", "path": row["path"]})
    for row in manifest.get("input_hashes", []):
        path = OUT / row["path"]
        if (
            not path.is_file()
            or path.stat().st_size != row["bytes"]
            or sha256_file(path) != row["sha256"]
        ):
            failures.append({"scope": "input", "path": row["path"]})
    current = {
        str(path.relative_to(OUT))
        for path in OUT.rglob("*")
        if path.is_file()
        and str(path.relative_to(OUT)) not in MANIFEST_EXCLUSIONS
    }
    recorded = {row["path"] for row in manifest.get("output_files", [])}
    untracked = sorted(current - recorded)
    stale = sorted(recorded - current)
    if untracked:
        failures.append({"scope": "inventory", "untracked": untracked})
    if stale:
        failures.append({"scope": "inventory", "stale": stale})
    return {
        "pass": not failures,
        "schema": manifest.get("schema"),
        "output_files": len(recorded),
        "implementation_files": len(
            manifest.get("implementation_files", [])
        ),
        "input_hashes": len(manifest.get("input_hashes", [])),
        "untracked": untracked,
        "stale": stale,
        "failures": failures,
        "sha256": sha256_file(OUT / "reproducibility_manifest.json"),
    }


def _verify_terminal_log_binding(
    log_audit: dict[str, Any],
) -> dict[str, Any]:
    manifest = read_json(OUT / "reproducibility_manifest.json")
    terminal = log_audit.get("terminal_event")
    failures = []
    if terminal is None:
        failures.append("terminal_event")
    else:
        expected = {
            "decision": read_json(OUT / "root_cause_decision.json")[
                "decision"
            ],
            "root_cause_sha256": sha256_file(
                OUT / "root_cause_decision.json"
            ),
            "per_architecture_sha256": sha256_file(
                OUT / "per_architecture_diagnosis.json"
            ),
            "iid_results_sha256": sha256_file(
                OUT / "iid_learnability_results.json"
            ),
            "reproducibility_manifest_sha256": sha256_file(
                OUT / "reproducibility_manifest.json"
            ),
            "final_runs": FINAL_RUNS,
            "required_outputs": len(REQUIRED_OUTPUTS),
        }
        for field, value in expected.items():
            if terminal.get(field) != value:
                failures.append(field)
    lines = (OUT / "run_log.jsonl").read_bytes().splitlines(keepends=True)
    preterminal = b"".join(lines[:-1]) if lines else b""
    snapshot = manifest["append_only_run_log_before_terminal_event"]
    if (
        len(preterminal) != snapshot["bytes"]
        or hashlib.sha256(preterminal).hexdigest() != snapshot["sha256"]
    ):
        failures.append("preterminal_snapshot")
    return {"pass": not failures, "failures": failures}


def verify() -> dict[str, Any]:
    missing_or_empty = [
        name
        for name in REQUIRED_OUTPUTS
        if not (OUT / name).is_file() or (OUT / name).stat().st_size == 0
    ]
    result: dict[str, Any] = {
        "schema": "vild-terminal-verification-v1",
        "study_id": STUDY_ID,
        "required_outputs": len(REQUIRED_OUTPUTS),
        "missing_or_empty": missing_or_empty,
        "checks": {},
    }
    if missing_or_empty:
        result["pass"] = False
        return result

    inputs = _read_inputs()
    amendment = _verify_amendment_and_snapshots(inputs["config"])
    diagnostics = _diagnostic_completeness(inputs)
    final_validation = _validate_final_confirmation(
        inputs["iid"], inputs["config"]
    )
    repair = _repair_evidence(inputs["config"])
    fault = _confirmed_fault_evidence()
    expected_per_architecture = _compute_per_architecture(
        inputs, amendment, final_validation
    )
    actual_per_architecture = read_json(
        OUT / "per_architecture_diagnosis.json"
    )
    per_architecture_match = actual_per_architecture == expected_per_architecture
    expected_root = _compute_root_decision(
        inputs,
        amendment,
        diagnostics,
        final_validation,
        expected_per_architecture,
        repair,
        fault,
    )
    actual_root = read_json(OUT / "root_cause_decision.json")
    root_match = actual_root == expected_root
    forbidden = _forbidden_artifact_audit(inputs)
    resources = _resource_status()
    manifest = _verify_manifest()
    log_audit = _audit_run_log(require_terminal=True)
    log_binding = _verify_terminal_log_binding(log_audit)
    result["checks"] = {
        "amended_preregistration_implementation_and_source_snapshot": amendment[
            "pass"
        ],
        "diagnostic_reports_complete": diagnostics["pass"],
        "exactly_60_final_runs_and_20_cells_recomputed": final_validation[
            "pass"
        ],
        "per_architecture_diagnosis_recomputed": per_architecture_match,
        "root_cause_decision_recomputed_in_frozen_order": root_match,
        "all_16_required_outputs_present": not missing_or_empty,
        "recursive_manifest_hashes_and_inventory": manifest["pass"],
        "run_log_hash_chain_and_required_event_order": log_audit["pass"],
        "terminal_event_binds_manifest_and_decisions": log_binding["pass"],
        "GPU_hours_within_5": resources["within_GPU_limit"],
        "new_artifacts_within_10_GiB": resources["within_artifact_limit"],
        "resource_ledger_hash_chain_valid": resources[
            "resource_ledger_pass"
        ],
        "resource_record_coverage_valid": resources[
            "resource_record_coverage_pass"
        ],
        "no_open_GPU_attempts": not resources["open_GPU_attempts"],
        "forbidden_artifacts_and_inputs_absent": forbidden["pass"],
        "repair_policy_valid": repair["valid"],
    }
    result["details"] = {
        "decision": actual_root.get("decision"),
        "amendment": amendment,
        "diagnostics": diagnostics,
        "final_confirmation": final_validation,
        "manifest": manifest,
        "run_log": log_audit,
        "terminal_log_binding": log_binding,
        "resources": resources,
        "forbidden": forbidden,
        "repair": repair,
        "per_architecture_match": per_architecture_match,
        "root_match": root_match,
    }
    result["pass"] = all(result["checks"].values())
    return result


def _write_terminal() -> dict[str, Any]:
    inputs = _read_inputs()
    prelog = _audit_run_log(require_terminal=False)
    if prelog["terminal_event_count"]:
        result = verify()
        if not result["pass"]:
            raise RuntimeError(
                "terminal event already exists but verification fails"
            )
        return result
    if not prelog["pass"]:
        raise RuntimeError(f"run-log chain/order failed: {prelog['failures']}")

    amendment = _verify_amendment_and_snapshots(inputs["config"])
    diagnostics = _diagnostic_completeness(inputs)
    final_validation = _validate_final_confirmation(
        inputs["iid"], inputs["config"]
    )
    repair = _repair_evidence(inputs["config"])
    fault = _confirmed_fault_evidence()
    forbidden = _forbidden_artifact_audit(inputs)
    resources = _resource_status()
    preconditions = {
        "amendment": amendment["pass"],
        "diagnostic_completeness": diagnostics["pass"],
        "final_confirmation": final_validation["pass"],
        "repair_policy": repair["valid"],
        "forbidden_artifacts": forbidden["pass"],
        "resource_ledger": resources["resource_ledger_pass"],
        "resource_record_coverage": resources[
            "resource_record_coverage_pass"
        ],
        "no_open_GPU_attempts": not resources["open_GPU_attempts"],
        "GPU_hours_strictly_below_5": resources["within_GPU_limit"],
        "artifacts_strictly_below_10_GiB": resources[
            "within_artifact_limit"
        ],
    }
    if not all(preconditions.values()):
        raise RuntimeError(f"terminal preconditions failed: {preconditions}")

    per_architecture = _compute_per_architecture(
        inputs, amendment, final_validation
    )
    write_json(OUT / "per_architecture_diagnosis.json", per_architecture)
    root = _compute_root_decision(
        inputs,
        amendment,
        diagnostics,
        final_validation,
        per_architecture,
        repair,
        fault,
    )
    write_json(OUT / "root_cause_decision.json", root)
    _text(
        OUT / "final_report.md",
        _final_report(
            inputs,
            per_architecture,
            root,
            amendment,
            diagnostics,
            final_validation,
            forbidden,
        ),
    )
    _text(
        OUT / "executive_summary_zh.md",
        _executive_zh(inputs, root, final_validation),
    )
    _text(OUT / "next_steps.md", _next_steps(root))
    manifest = _build_manifest(
        inputs,
        amendment,
        diagnostics,
        final_validation,
        forbidden,
        root,
    )
    write_json(OUT / "reproducibility_manifest.json", manifest)

    manifest_check = _verify_manifest()
    if not manifest_check["pass"]:
        raise RuntimeError(
            f"manifest failed before terminal append: {manifest_check}"
        )
    missing = [
        name
        for name in REQUIRED_OUTPUTS
        if name != "run_log.jsonl"
        and (
            not (OUT / name).is_file() or (OUT / name).stat().st_size == 0
        )
    ]
    if missing:
        raise RuntimeError(f"terminal required outputs missing: {missing}")
    resources = _resource_status()
    if (
        not resources["resource_ledger_pass"]
        or not resources["resource_record_coverage_pass"]
        or resources["open_GPU_attempts"]
        or not resources["within_GPU_limit"]
        or not resources["within_artifact_limit"]
    ):
        raise RuntimeError(f"terminal resource limit failed: {resources}")

    append_log(
        TERMINAL_EVENT,
        decision=root["decision"],
        root_cause_sha256=sha256_file(OUT / "root_cause_decision.json"),
        per_architecture_sha256=sha256_file(
            OUT / "per_architecture_diagnosis.json"
        ),
        iid_results_sha256=sha256_file(
            OUT / "iid_learnability_results.json"
        ),
        reproducibility_manifest_sha256=sha256_file(
            OUT / "reproducibility_manifest.json"
        ),
        final_runs=FINAL_RUNS,
        required_outputs=len(REQUIRED_OUTPUTS),
        final_confirmation_study_pass=inputs["iid"]["study_pass"],
        independent_metric_recomputation_pass=inputs["iid"][
            "independent_metric_recomputation"
        ]["pass"],
        source_study_unchanged=amendment["source_snapshot"]["pass"],
        additional_verifier_architectures=0,
        forbidden_artifacts_present=False,
        resource_ledger_pass=resources["resource_ledger_pass"],
        resource_record_coverage_pass=resources[
            "resource_record_coverage_pass"
        ],
        open_GPU_attempts=resources["open_GPU_attempts"],
        GPU_hours=resources["GPU_hours"],
        output_GiB_before_terminal_event=resources["output_GiB"],
    )
    result = verify()
    if not result["pass"]:
        raise RuntimeError("post-terminal verification failed")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="perform read-only package verification",
    )
    arguments = parser.parse_args()
    result = verify() if arguments.verify_only else _write_terminal()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
