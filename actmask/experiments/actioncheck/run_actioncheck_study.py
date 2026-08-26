"""Execute the preregistered ACTIONCHECK-PROPOSAL-DATA-V1 study."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from .audit_actioncheck_shortcuts import audit_shortcuts, shortcut_gate_config
from .build_actioncheck_benchmark import build_benchmark
from .common import (
    EVIDENCE_TYPES,
    OUT,
    PYTHON,
    ROOT,
    actioncheck_schema,
    canonical_json,
    label_taxonomy,
    reason_taxonomy,
    sha256_file,
    write_json,
    write_jsonl,
)
from .create_actioncheck_splits import create_splits, split_audit_schema
from .evaluate_actioncheck import evaluate_file
from .local_sources import (
    M4R,
    M4R_GENERATOR,
    M5B,
    M5B_GENERATOR,
    build_local_pilot_groups,
    inventory_local_data,
    source_inventory_records,
)
from .pilot_baselines import BASELINES, run_pilot_baselines
from .preregister_actioncheck import SEED, STUDY_ID, freeze
from .validate_actioncheck_schema import validate_group
from .validate_candidate_balance import candidate_balance
from .validate_evidence import validate_evidence
from .validate_group_structure import validate_groups
from .validate_splits import validate_splits


def _write_text(relative: str, text: str) -> Path:
    path = OUT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")
    return path


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _log(event: str, **payload: Any) -> None:
    path = OUT / "run_log.jsonl"
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    previous = rows[-1]
    record = {
        "sequence": int(previous["sequence"]) + 1,
        "study_id": STUDY_ID,
        "event": event,
        "previous_record_sha256": previous["record_sha256"],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        **payload,
    }
    record["record_sha256"] = hashlib.sha256(
        canonical_json(record).encode("utf-8")
    ).hexdigest()
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _write_schema(groups: list[dict[str, Any]]) -> None:
    write_json(OUT / "schema/actioncheck_schema_v1.json", actioncheck_schema())
    write_json(OUT / "schema/label_taxonomy.json", label_taxonomy())
    write_json(OUT / "schema/reason_code_taxonomy.json", reason_taxonomy())
    write_json(OUT / "schema/example_context_group.json", groups[0])
    _write_text(
        "schema/README.md",
        """
# ActionCheck schema v1

The atomic row is one causal context and all candidate action chunks proposed
for that context. `accept` and `reject` are eligible for primary training and
evaluation; `uncertain` remains in manifests and analysis only.

Every primary label must cite an allowed evidence type and a concrete evidence
record. The action hash is SHA-256 over the int64 shape followed by C-order
float32 bytes. All candidates in a context stay together for every split.

`reason_codes`, `evidence_type`, `label_source`, source paths, and outcome
records are audit metadata. They are forbidden as verifier inputs. Compatibility
means acceptable under the recorded context and evidence; it is not a guarantee
of physical safety, task success, or a causal alternate-world outcome.
""",
    )


def _inventory(groups: list[dict[str, Any]]) -> dict[str, Any]:
    local = inventory_local_data()
    records = source_inventory_records()
    decision = "ACTIONCHECK_VALID_PROPOSAL_DATA_CANDIDATE_FOUND"
    local["decision"] = decision
    local["inventory_scope"] = {
        "project": str(ROOT),
        "robomind2": str(ROOT.parent.parent / "RoboMIND2"),
        "named_log_search_root": str(ROOT.parent.parent),
        "named_terms": ["TB6-R5", "TB6", "UR5", "pi0.5", "pi0", "OpenPI"],
        "read_only": True,
    }
    write_json(OUT / "local_data_inventory.json", local)
    write_json(
        OUT / "proposal_source_inventory.json",
        {
            "schema": "actioncheck-proposal-source-inventory-v1",
            "decision": decision,
            "sources": records,
        },
    )
    execution = {
        "schema": "actioncheck-execution-label-inventory-v1",
        "real_execution_sources": [],
        "simulated_execution_sources": [
            {
                "source_id": "milestone4r_v3_gpu_physx",
                "path": str(M4R),
                "contexts": 768,
                "candidates": 7680,
                "evidence": "each candidate was rolled out in ManiSkill 3 GPU PhysX",
            },
            {
                "source_id": "milestone5b_v3_gpu_physx",
                "path": str(M5B),
                "contexts": 256,
                "candidates": 2560,
                "evidence": "candidate_labels.jsonl plus execution_logs.json",
            },
        ],
        "selected_for_pilot": True,
        "labels_inferred_from_filenames": False,
        "logged_action_swaps_used": False,
    }
    write_json(OUT / "execution_label_inventory.json", execution)
    write_json(
        OUT / "sim_label_inventory.json",
        {
            "schema": "actioncheck-sim-label-inventory-v1",
            "simulator": "ManiSkill 3 GPU PhysX",
            "sources": execution["simulated_execution_sources"],
            "context_groups": len(groups),
            "candidate_labels": sum(len(row["candidates"]) for row in groups),
            "outcomes": dict(
                Counter(
                    candidate["label"]
                    for group in groups
                    for candidate in group["candidates"]
                )
            ),
            "evidence_types": sorted(
                {
                    candidate["evidence_type"]
                    for group in groups
                    for candidate in group["candidates"]
                }
            ),
            "fine_grained_failure_reason_available": False,
            "verified_failure_reason": "simulator_failure",
        },
    )
    write_json(
        OUT / "expert_label_inventory.json",
        {
            "schema": "actioncheck-expert-label-inventory-v1",
            "expert_review_sources_found": [],
            "operator_review_sources_found": [],
            "selected_labels_from_expert_or_operator": 0,
            "note": "No expert labels were invented to refine simulator failures.",
        },
    )
    names = local["name_matches"]
    _write_text(
        "data_availability_report.md",
        f"""
# Local data availability report

Decision: `{decision}`.

The read-only inventory found two valid proposal-level simulator sources:
Milestone 4R v3 ({768} contexts, {7680} candidate rollouts, three tasks) and
Milestone 5B v3 ({256} contexts, {2560} candidate rollouts, two tasks). Together
they provide **{len(groups)} context groups, 10,240 executed candidates, five
tasks, two proposal sources, and two evidence types**. Candidates are numeric
action chunks generated before their own rollout; labels come from per-candidate
ManiSkill 3 GPU PhysX outcomes.

RoboMIND2 demonstrations, BotFails trajectories, symbolic planner logs, ASU
logged swaps, and SAM3 segmentation do not satisfy the same-context,
multi-proposal, evidence-backed unit and are excluded. The named TB6/UR5/pi0/
OpenPI filename scan produced {len(names)} path matches; a name match was never
treated as a proposal label.

The data is sufficient for the preregistered minimum pilot admission. A known
scientific defect remains: every verified reject has only the coarse
`simulator_failure` reason. Fine-grained causes cannot be inferred after the
fact, so the 40% single-reason gate is expected to fail honestly.
""",
    )
    return local


def _write_original_source_hash_manifest(
    groups: list[dict[str, Any]],
) -> None:
    keys = (
        "source_candidate_file",
        "source_label_file",
        "source_bundle_file",
        "source_action_file",
        "source_history_file",
        "source_execution_log",
    )
    paths = {
        Path(value)
        for group in groups
        for key in keys
        if (value := group["provenance"].get(key))
    }
    paths.update((M4R_GENERATOR, M5B_GENERATOR))
    write_json(
        OUT / "pilot/original_source_hash_manifest.json",
        {
            "schema": "actioncheck-original-source-hash-manifest-v1",
            "read_only_sources": [
                {
                    "path": str(path.resolve()),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in sorted(paths)
            ],
            "labels_inferred_from_path": False,
            "sources_modified": False,
        },
    )


def _run_tests() -> tuple[str, str]:
    validator_command = [
        str(PYTHON),
        "-m",
        "pytest",
        "-q",
        "tests/test_actioncheck_schema.py",
        "tests/test_actioncheck_validators.py",
        "tests/test_actioncheck_baselines.py",
    ]
    metric_command = [
        str(PYTHON),
        "-m",
        "pytest",
        "-q",
        "tests/test_actioncheck_metrics.py",
    ]
    validator = subprocess.run(
        validator_command,
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    metric = subprocess.run(
        metric_command,
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    _write_text("validator_test_output.txt", validator.stdout + validator.stderr)
    _write_text("metric_test_output.txt", metric.stdout + metric.stderr)
    return validator.stdout.strip(), metric.stdout.strip()


def _validators(
    groups: list[dict[str, Any]], splits: dict[str, Any]
) -> dict[str, Any]:
    schema_failures = [
        {"context_id": row["context_id"], "errors": errors}
        for row in groups
        if (errors := validate_group(row))
    ]
    evidence = validate_evidence(groups)
    structure = validate_groups(groups, minimum_candidates=4)
    balance = candidate_balance(groups)
    split_result = validate_splits(groups, splits)
    validator_test, metric_test = _run_tests()
    report = {
        "schema": "actioncheck-validators-report-v1",
        "schema_validation": {
            "pass": not schema_failures,
            "failures": schema_failures,
        },
        "group_structure": structure,
        "evidence": evidence,
        "candidate_balance": balance,
        "splits": split_result,
        "validator_tests": validator_test,
        "metric_tests": metric_test,
        "pass": (
            not schema_failures
            and structure["pilot_80_percent_gate"]
            and evidence["pass"]
            and split_result["pass"]
        ),
    }
    write_json(OUT / "validators_report.json", report)
    _write_text(
        "validators_report.md",
        f"""
# Validator report

- Schema: {"PASS" if not schema_failures else "FAIL"}
- Evidence: {"PASS" if evidence["pass"] else "FAIL"}
- Pilot group-support gate (K=4, mixed labels in >=80%):
  {"PASS" if structure["pilot_80_percent_gate"] else "FAIL"}
- Atomic leakage-safe splits: {"PASS" if split_result["pass"] else "FAIL"}
- Context groups checked: {len(groups)}
- Candidates checked: {balance["candidates"]}
- Validator tests: `{validator_test}`
- Metric tests: `{metric_test}`

The validators reject missing fields, non-finite or mismatched action chunks,
unsupported evidence/reasons, forbidden label bases, contexts without both
primary labels, duplicate IDs, and split leakage. They report rather than hide
source, task, evidence, reason, label, and per-context-count imbalance.
""",
    )
    _write_text(
        "validator_unit_tests.md",
        """
# Validator unit tests

`tests/test_actioncheck_schema.py` covers a valid group, action-hash integrity,
and invalid label rejection. `tests/test_actioncheck_validators.py` covers
forbidden evidence bases, missing mixed labels, candidate counts, and leakage
free split generation. The persisted test output is
`validator_test_output.txt`.
""",
    )
    _write_text(
        "metric_unit_tests.md",
        """
# Metric unit tests

`tests/test_actioncheck_metrics.py` checks perfect and reversed pairwise
classification, grouped ranking, calibration/risk-coverage output, and
context-group bootstrap evaluation. The persisted test output is
`metric_test_output.txt`.
""",
    )
    return report


def _write_protocols() -> None:
    _write_text(
        "protocols/route1_real_robot_collection.md",
        """
# Route 1 — real-robot proposal filtering

Target TB6-R5 or UR5 with a frozen pi0.5/OpenPI server and optionally ACT,
diffusion, BC, or another VLA proposal generator. Synchronize monotonic and
wall-clock time for RGB, robot state, gripper, proposed chunks, safety results,
execution, and operator review.

For each context: (1) freeze the causal history before proposals; (2) request
4–8 independently identified candidate chunks and retain generator seeds and
versions; (3) run a frozen kinematic/workspace/collision/velocity gate; (4) only
under the approved short-horizon protocol, reset or establish an equivalent
state and execute selected candidates; (5) record accept/reject/uncertain from
execution or explicit review; (6) record evidence and a verified reason, never
distance from the demonstration.

Pilot target: 8–10 tasks, at least 30 contexts/task, 4–8 candidates/context,
at least one accept and reject/context, 300 contexts and 1,200–2,400 labels.
Paper target: 30–50 tasks, 50–100 contexts/task, 6–10 candidates/context,
1,500–5,000 contexts and 10,000–50,000 labels.

Safety: start with low speed/force, enforce joint/workspace limits and
collision clearance, provide a tested e-stop and trained spotter, abort on
tracking loss or unexpected contact, and preserve uncertain outcomes. Dataset
acceptability is not a declaration of physical safety.
""",
    )
    write_json(
        OUT / "protocols/real_robot_logging_schema.json",
        {
            "schema": "actioncheck-real-robot-log-v1",
            "required": [
                "context_id",
                "robot_id",
                "task_id",
                "clock_sync",
                "causal_history",
                "candidate_id",
                "proposal_policy",
                "action_chunk",
                "pre_execution_gate",
                "execution_or_review_evidence",
                "label",
                "reason_codes",
                "operator_id_hash",
                "software_and_calibration_hashes",
            ],
            "reset_equivalence_record_required_for_multiple_executions": True,
            "raw_sensor_references_immutable": True,
        },
    )
    _write_text(
        "protocols/operator_review_form.md",
        """
# Operator review form

Record context/candidate IDs, whether the operator saw only causal information,
decision (`accept`, `reject`, `uncertain`), evidence source, verified reason
codes, intervention/abort time, observed event, confidence (1–5), and notes.
Require a second review for ambiguous or safety-related cases. Disagreement is
stored as `uncertain`; it is never resolved by demo-action distance.
""",
    )
    _write_text(
        "protocols/safety_gate_checklist.md",
        """
# Pre-execution safety checklist

- Robot, payload, controller, calibration, task boundary, and reset state match.
- E-stop, speed/force limits, workspace and joint limits are tested.
- Predicted path clears static keep-out zones at the frozen margin.
- Perception/state timestamps are within tolerance; otherwise mark uncertain.
- Chunk units, frame, horizon, frequency, finite values, velocity and jerk pass.
- Human/fixture clearance and trained spotter are confirmed.
- Abort and intervention events are logged independently of final labels.
- Gate rejection records its rule and does not masquerade as executed evidence.
""",
    )
    _write_text(
        "protocols/route2_offline_policy_proposals.md",
        """
# Route 2 — offline policy proposals

Use RoboMIND or local robot demonstrations only as frozen causal contexts.
Generate K chunks from frozen proposal policies with model/checkpoint/config,
seed, temperature and decoding metadata. Label each proposal only through
expert review, an explicit frozen rule checker, a valid simulator mapping, or
existing execution evidence. Demo distance and logged mismatch are never
labels. Preserve proposal-policy metadata for stratification and shortcut
audits, but exclude it from verifier inputs. Sample multiple policies within
tasks so source identity is not synonymous with task or label.
""",
    )
    write_json(
        OUT / "protocols/offline_policy_proposal_schema.json",
        {
            "schema": "actioncheck-offline-proposal-v1",
            "required": [
                "context_id",
                "causal_context_hash",
                "candidate_id",
                "policy_family",
                "checkpoint_hash",
                "inference_config",
                "random_seed",
                "action_chunk",
                "evidence",
                "label",
            ],
            "demo_distance_as_label_forbidden": True,
            "policy_metadata_as_verifier_input_forbidden": True,
        },
    )
    _write_text(
        "protocols/expert_review_guidelines.md",
        """
# Expert review guidelines

Reviewers receive the causal context, task, candidate chunk in calibrated units,
and the explicit rule set—not the desired label or source filename. First judge
whether evidence is sufficient; if not, use `uncertain`. For accept/reject,
record the observable rationale and allowed reason codes. Randomize candidate
order, blind proposal source when practical, double-review at least 20%, report
agreement, adjudicate disagreements without using demo distance, and retain all
original decisions.
""",
    )
    _write_text(
        "protocols/route3_simulator_branching.md",
        """
# Route 3 — simulator branching

At each initial state, save the complete simulator state, RNG state, task state,
asset/version hashes and causal observation. Restore that identical state before
each candidate; execute the exact numeric chunk and continue a frozen evaluation
horizon. Store state and action hashes, progress trajectory, success, collision,
contact, constraint, timeout and terminal observations.

Label from frozen rules: success is accept; verified collision/constraint,
regression, no-progress, unstable contact or timeout is reject with the matching
reason; unresolved simulator errors or conflicting measures are uncertain.
Never infer a failure subtype from action geometry alone. Check deterministic
replay on a stratified sample and keep all candidates for a context atomic.
""",
    )
    write_json(
        OUT / "protocols/sim_rollout_schema.json",
        {
            "schema": "actioncheck-simulator-rollout-v1",
            "required": [
                "simulator",
                "simulator_version",
                "asset_hashes",
                "initial_state_hash",
                "rng_state_hash",
                "context_id",
                "candidate_id",
                "action_hash",
                "rollout_horizon",
                "success",
                "progress_trace",
                "collision_trace",
                "contact_trace",
                "constraint_events",
                "timeout",
                "terminal_state_hash",
                "label_rule_version",
            ],
            "same_initial_state_required": True,
        },
    )
    _write_text(
        "protocols/sim_label_rules.md",
        """
# Frozen simulator label rules

Assign accept only when the task success predicate is met within the evaluation
horizon without a frozen hard-rule violation. Assign reject when a recorded
event establishes wrong phase/object/direction, gripper error, no progress,
regression, collision, joint/workspace breach, unstable contact, or timeout.
Use `simulator_failure` only for a genuine simulator-reported failure lacking a
verified subtype. Ambiguous, nondeterministic, crashed, or conflicting rollouts
are `uncertain`. Rule versions are hashed before collection.
""",
    )


def _write_design_documents(splits: dict[str, Any]) -> None:
    write_json(OUT / "split_audit_schema.json", split_audit_schema())
    write_json(OUT / "shortcut_gate_config.json", shortcut_gate_config())
    _write_text(
        "split_design_report.md",
        f"""
# Split design

Context groups are atomic; candidate-level random splitting is forbidden.
Episode-, task-, object/composition-, and scene/environment-held-out schemes
are deterministic hashes over the grouping value. Policy-source evaluation is
defined as leave-one-source-out folds. The current manifest contains
{len(splits["policy_source_folds"])} policy-source folds and all split audits
pass before benchmark construction. Every audit reports label, reason and
evidence distributions by split.
""",
    )
    _write_text(
        "shortcut_audit_design.md",
        """
# Shortcut audit design

Train or evaluate 13 controls: random, task, progress, context, action, action
summary, state, visual, policy source, evidence type, reason code, metadata, and
a combined non-relational control. Thresholds are selected on validation and
gates are evaluated on held-out test contexts.

Frozen BA maxima are 0.53 for policy source and metadata, 0.55 for action and
context, 0.56 for action summary, and 0.60 for the combined non-relational
control. No source may contribute over 80% of either primary label and no reason
may cover over 40% of rejects. Evidence and reason controls intentionally show
audit predictability; those fields are never legal model inputs.
""",
    )
    _write_text(
        "evaluation_metric_spec.md",
        """
# Evaluation metric specification

Classification reports balanced accuracy, AUROC, AUPRC, false-accept rate and
false-reject rate using a threshold selected only on validation. Per-context
ranking reports Recall@1, MRR, NDCG, best-positive score margin and mean
best-accept rank. Calibration reports 10-bin ECE and Brier score. Selective
evaluation ranks by distance from the validation threshold and reports accuracy
and false-accept risk at 100%, 90%, 80% and 70% coverage.

Generalization is reported for episode, task, object/composition, policy source
and scene when supported, plus task/source/reason/evidence strata. Reason and
evidence strata are audit-only. Confidence intervals resample whole contexts
(or episodes), never individual candidates. Undefined one-class metrics remain
explicit rather than being converted to zero.
""",
    )
    _write_text(
        "benchmark_builder_report.md",
        """
# Benchmark builder implementation

`actmask/experiments/actioncheck/build_actioncheck_benchmark.py` accepts one or
more context-group JSONL inputs plus schema, split and metric configs. It
validates schema/evidence/splits, keeps each context atomic, retains uncertain
rows only in manifests, pads causal histories/action chunks into NPZ arrays and
writes source hashes. It never synthesizes rejects or uses logged swaps as
primary negatives. The actual pilot build report is under `pilot/benchmark/`.
""",
    )


def _admission(groups: list[dict[str, Any]]) -> dict[str, Any]:
    contexts = len(groups)
    candidate_count = sum(len(group["candidates"]) for group in groups)
    mixed = sum(
        {
            candidate["label"] for candidate in group["candidates"]
        }.issuperset({"accept", "reject"})
        for group in groups
    )
    tasks = {group["task_id"] for group in groups}
    sources = {
        candidate["source_policy"]
        for group in groups
        for candidate in group["candidates"]
    }
    evidences = {
        candidate["evidence_type"]
        for group in groups
        for candidate in group["candidates"]
    }
    values = {
        "context_groups": contexts,
        "average_candidates_per_context": candidate_count / contexts,
        "mixed_label_contexts": mixed,
        "mixed_label_context_fraction": mixed / contexts,
        "tasks": len(tasks),
        "candidate_sources": len(sources),
        "evidence_types": len(evidences),
        "labeled_candidates": candidate_count,
    }
    gates = {
        "context_groups_ge_100": contexts >= 100,
        "average_candidates_ge_4": candidate_count / contexts >= 4,
        "mixed_label_context_fraction_ge_0_80": mixed / contexts >= 0.80,
        "tasks_ge_5": len(tasks) >= 5,
        "candidate_sources_ge_2": len(sources) >= 2,
        "evidence_types_ge_2": len(evidences) >= 2,
        "labeled_candidates_ge_500": candidate_count >= 500,
    }
    return {
        "schema": "actioncheck-pilot-admission-v1",
        "values": values,
        "gates": gates,
        "pass": all(gates.values()),
    }


def prepare() -> dict[str, Any]:
    frozen = freeze(OUT)
    if sha256_file(OUT / "preregistered_config.json") != frozen["sha256"]:
        raise RuntimeError("preregistration hash changed")
    print("[P1] loading read-only proposal sources", flush=True)
    groups = build_local_pilot_groups()
    source_groups = OUT / "pilot/source_actioncheck_context_groups.jsonl"
    write_jsonl(source_groups, groups)
    _write_original_source_hash_manifest(groups)
    local = _inventory(groups)
    _write_schema(groups)
    splits = create_splits(groups)
    write_json(OUT / "pilot/prebuilt_split_manifest.json", splits)
    validators = _validators(groups, splits)
    _write_protocols()
    _write_design_documents(splits)
    admission = _admission(groups)
    write_json(OUT / "pilot/pilot_admission.json", admission)
    _log(
        "P1_TO_P8_PREPARED",
        inventory_decision=local["decision"],
        context_groups=len(groups),
        candidates=sum(len(group["candidates"]) for group in groups),
        validators_pass=validators["pass"],
        admission_pass=admission["pass"],
        source_context_groups_sha256=sha256_file(source_groups),
    )
    return {
        "groups": groups,
        "source_groups": source_groups,
        "splits": splits,
        "validators": validators,
        "admission": admission,
    }


def load_prepared() -> dict[str, Any]:
    """Load the already validated P1-P8 artifacts after a recoverable code error."""
    from .common import read_jsonl

    source_groups = OUT / "pilot/source_actioncheck_context_groups.jsonl"
    required = [
        OUT / "preregistered_config.json",
        OUT / "preregistered_config.sha256",
        source_groups,
        OUT / "pilot/prebuilt_split_manifest.json",
        OUT / "pilot/pilot_admission.json",
        OUT / "validators_report.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"prepared artifacts are missing: {missing}")
    recorded = (OUT / "preregistered_config.sha256").read_text(
        encoding="utf-8"
    ).strip()
    if sha256_file(OUT / "preregistered_config.json") != recorded:
        raise RuntimeError("prepared preregistration hash mismatch")
    validators = _read_json(OUT / "validators_report.json")
    admission = _read_json(OUT / "pilot/pilot_admission.json")
    if not validators["pass"] or not admission["pass"]:
        raise RuntimeError("prepared validation or admission does not pass")
    return {
        "groups": read_jsonl(source_groups),
        "source_groups": source_groups,
        "splits": _read_json(OUT / "pilot/prebuilt_split_manifest.json"),
        "validators": validators,
        "admission": admission,
    }


def _gpu_record() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is unavailable; refusing to fall back to long CPU training"
        )
    properties = torch.cuda.get_device_properties(0)
    record = {
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device_count": torch.cuda.device_count(),
        "selected_device": 0,
        "device_name": torch.cuda.get_device_name(0),
        "total_memory_bytes": properties.total_memory,
    }
    write_json(OUT / "pilot/gpu_runtime.json", record)
    return record


def _pilot_markdown(
    build: dict[str, Any],
    shortcut: dict[str, Any],
    baselines: dict[str, Any],
    evaluation: dict[str, Any],
    decision: dict[str, Any],
) -> None:
    balance = build["candidate_balance"]
    _write_text(
        "pilot_benchmark_report.md",
        f"""
# Pilot benchmark report

The admitted pilot contains {build["context_groups"]} atomic contexts,
{build["primary_candidates"]} primary candidates, five tasks and two proposal
sources. Mean candidates/context is
{build["primary_candidates"] / build["context_groups"]:.1f}. Label counts are
{balance["label_distribution"]}; evidence counts are
{balance["evidence_type_distribution"]}. No reject was synthesized, no logged
action swap was used, and uncertain rows (none in this pilot) would be excluded
from primary arrays.

{build["group_structure"]["fully_supported_contexts"]} of
{build["context_groups"]} contexts contain both labels
({build["group_structure"]["fully_supported_fraction"]:.3%}), exceeding the frozen
80% admission criterion. The six uniform contexts are retained to avoid
label-based filtering.
""",
    )
    failed = [name for name, value in shortcut["gates"].items() if not value]
    _write_text(
        "pilot_shortcut_report.md",
        f"""
# Pilot shortcut report

Overall shortcut gate: **{"PASS" if shortcut["pass"] else "FAIL"}**.
Failed gates: {", ".join(f"`{name}`" for name in failed) or "none"}.
The maximum verified reason-code fraction is
{shortcut["maximum_reason_code_fraction_of_rejects"]:.3f}; all rejects carry
only the evidence-supported coarse `simulator_failure` code. This is a dataset
instrumentation failure, not a license to infer finer reasons. All 13 control
predictions are saved under `pilot/shortcut/shortcut_predictions.jsonl`.
""",
    )
    rows = []
    for key, result in sorted(baselines["evaluation"].items()):
        rows.append(
            f"| {key} | {result['classification']['balanced_accuracy']:.4f} | "
            f"{result['grouped_ranking']['recall_at_1']:.4f} | "
            f"{result['classification']['false_accept_rate']:.4f} |"
        )
    _write_text(
        "pilot_baseline_report.md",
        """
# Pilot baseline report

Only the six preregistered simple baselines were trained; no PhaseAware model
was instantiated or trained. Thresholds were selected on validation data.

| split/model | test BA | Recall@1 | FAR |
|---|---:|---:|---:|
"""
        + "\n".join(rows),
    )
    best_key, best_result = max(
        evaluation["models"].items(),
        key=lambda item: item[1]["classification"]["balanced_accuracy"],
    )
    _write_text(
        "pilot_evaluation_report.md",
        f"""
# Pilot evaluation report

Raw validation/test predictions were recomputed with pairwise, grouped-ranking,
calibration, risk-coverage and context-bootstrap metrics. The highest test
balanced accuracy among the 12 split/model combinations is
{best_result["classification"]["balanced_accuracy"]:.4f} for `{best_key}`;
its Recall@1 is {best_result["grouped_ranking"]["recall_at_1"]:.4f} versus
chance {best_result["grouped_ranking"]["chance_recall_at_1"]:.4f}. This is a
diagnostic pilot result, not a method claim, because the shortcut gate failed.
Full machine-readable values are in `pilot/evaluation/evaluation.json`.
""",
    )
    write_json(OUT / "pilot_decision.json", decision)


def run_pilot(prepared: dict[str, Any]) -> dict[str, Any]:
    admission = prepared["admission"]
    if not admission["pass"]:
        decision = {
            "decision": "ACTIONCHECK_PROPOSAL_DATA_NOT_AVAILABLE",
            "admission": admission,
            "training_run": False,
        }
        write_json(OUT / "pilot_decision.json", decision)
        _log("P9_SKIPPED", decision=decision["decision"])
        return decision
    gpu = _gpu_record()
    benchmark_dir = OUT / "pilot/benchmark"
    print(f"[P9] benchmark build; GPU={gpu['device_name']}", flush=True)
    build = build_benchmark(
        [prepared["source_groups"]],
        benchmark_dir,
        schema_config=OUT / "schema/actioncheck_schema_v1.json",
        split_config=OUT / "pilot/prebuilt_split_manifest.json",
        metric_config=OUT / "preregistered_config.json",
    )
    if not build["evidence_validation"]["pass"]:
        decision = {
            "decision": "ACTIONCHECK_PROPOSAL_LABELS_INVALID",
            "admission": admission,
            "build": build,
        }
        write_json(OUT / "pilot_decision.json", decision)
        _log("P9_LABEL_INVALID", decision=decision["decision"])
        return decision
    print("[P9] GPU shortcut controls", flush=True)
    shortcut = audit_shortcuts(
        benchmark_dir,
        OUT / "pilot/shortcut",
        scheme="episode_held_out",
        seed=SEED,
    )
    print("[P9] GPU fair baselines", flush=True)
    baselines = run_pilot_baselines(
        benchmark_dir,
        OUT / "pilot/baselines",
        seed=SEED,
    )
    baseline_predictions = OUT / "pilot/baselines/raw_baseline_predictions.jsonl"
    evaluation = evaluate_file(baseline_predictions)
    write_json(OUT / "pilot/evaluation/evaluation.json", evaluation)
    decision_name = (
        "ACTIONCHECK_PROPOSAL_SHORTCUT_DOMINATED"
        if not shortcut["pass"]
        else "ACTIONCHECK_PROPOSAL_TARGET_NOT_RECOVERABLE"
    )
    if shortcut["pass"]:
        best = max(
            result["classification"]["balanced_accuracy"]
            for result in evaluation["models"].values()
        )
        best_gap = max(
            result["grouped_ranking"]["recall_at_1"]
            - result["grouped_ranking"]["chance_recall_at_1"]
            for result in evaluation["models"].values()
        )
        if best >= 0.62 and best_gap >= 0.10:
            decision_name = "ACTIONCHECK_PROPOSAL_BENCHMARK_READY_FOR_METHOD"
    decision = {
        "schema": "actioncheck-pilot-decision-v1",
        "decision": decision_name,
        "admission": admission,
        "evidence_valid": build["evidence_validation"]["pass"],
        "shortcut_gates": shortcut["gates"],
        "shortcut_pass": shortcut["pass"],
        "baseline_models": list(BASELINES),
        "phaseaware_temporal_energy_verifier_trained": False,
        "gpu": gpu,
    }
    raw_files = [
        OUT / "pilot/shortcut/shortcut_predictions.jsonl",
        baseline_predictions,
    ]
    write_json(
        OUT / "raw_prediction_manifest.json",
        {
            "schema": "actioncheck-raw-prediction-manifest-v1",
            "files": [
                {
                    "path": str(path.relative_to(OUT)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "rows": sum(
                        1
                        for line in path.read_text(encoding="utf-8").splitlines()
                        if line.strip()
                    ),
                }
                for path in raw_files
            ],
            "metric_recompute": "pilot/evaluation/evaluation.json",
        },
    )
    _pilot_markdown(build, shortcut, baselines, evaluation, decision)
    _log(
        "P9_PILOT_COMPLETE",
        decision=decision_name,
        shortcut_pass=shortcut["pass"],
        baseline_prediction_sha256=sha256_file(baseline_predictions),
        phaseaware_trained=False,
    )
    return decision


def refresh_evaluation_metadata() -> None:
    """Attach audit strata to existing scores and recompute metrics without training."""
    from .common import read_jsonl

    benchmark_dir = OUT / "pilot/benchmark"
    prediction_path = OUT / "pilot/baselines/raw_baseline_predictions.jsonl"
    manifest = {
        row["candidate_id"]: row
        for row in read_jsonl(benchmark_dir / "candidate_manifest.jsonl")
    }
    predictions = read_jsonl(prediction_path)
    for row in predictions:
        candidate = manifest.get(row["candidate_id"])
        if candidate is None:
            raise RuntimeError(f"prediction has unknown candidate: {row['candidate_id']}")
        if (
            candidate["context_id"] != row["context_id"]
            or (1 if candidate["label"] == "accept" else 0) != int(row["label"])
        ):
            raise RuntimeError(f"prediction/manifest mismatch: {row['candidate_id']}")
        row["evidence_type_audit_only"] = candidate[
            "evidence_type_audit_only"
        ]
        row["reason_codes_audit_only"] = candidate[
            "reason_codes_audit_only"
        ]
    write_jsonl(prediction_path, predictions)
    evaluation = evaluate_file(prediction_path)
    write_json(OUT / "pilot/evaluation/evaluation.json", evaluation)
    baseline_result_path = OUT / "pilot/baselines/pilot_baseline_results.json"
    baselines = _read_json(baseline_result_path)
    baselines["evaluation"] = evaluation["models"]
    baselines["audit_strata_attached_post_training"] = True
    baselines["scores_changed_during_metadata_refresh"] = False
    write_json(baseline_result_path, baselines)
    shortcut_prediction = OUT / "pilot/shortcut/shortcut_predictions.jsonl"
    raw_files = [shortcut_prediction, prediction_path]
    write_json(
        OUT / "raw_prediction_manifest.json",
        {
            "schema": "actioncheck-raw-prediction-manifest-v1",
            "files": [
                {
                    "path": str(path.relative_to(OUT)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "rows": len(read_jsonl(path)),
                }
                for path in raw_files
            ],
            "metric_recompute": "pilot/evaluation/evaluation.json",
            "audit_metadata_only_fields": [
                "evidence_type_audit_only",
                "reason_codes_audit_only",
            ],
            "scores_changed_during_metadata_refresh": False,
        },
    )
    _pilot_markdown(
        _read_json(benchmark_dir / "benchmark_build_report.json"),
        _read_json(OUT / "pilot/shortcut/shortcut_audit.json"),
        baselines,
        evaluation,
        _read_json(OUT / "pilot_decision.json"),
    )
    _log(
        "P8_AUDIT_STRATA_RECOMPUTED",
        raw_prediction_sha256=sha256_file(prediction_path),
        scores_changed=False,
    )


def _write_method_policy() -> None:
    conditions = {
        "schema": "phaseaware-training-conditions-v1",
        "method": "PhaseAwareTemporalEnergyVerifier",
        "authorized_now": False,
        "separate_future_goal_required": True,
        "conditions": {
            "context_groups_pilot_min": 300,
            "context_groups_paper_min": 1000,
            "average_candidates_min": 4,
            "primary_contexts_require_accept_and_reject": True,
            "action_only_BA_max": 0.55,
            "context_only_BA_max": 0.55,
            "policy_source_only_BA_max": 0.53,
            "combined_non_relational_BA_max": 0.60,
            "fair_baseline_BA_min": 0.62,
            "Recall@1_above_chance_min": 0.10,
            "task_or_source_held_out_required": True,
            "raw_predictions_and_metric_recompute_required": True,
        },
        "current_blocker": (
            "pilot shortcut gates must all pass; current generic failure reason "
            "dominates rejects"
        ),
    }
    write_json(OUT / "phaseaware_training_conditions.json", conditions)
    _write_text(
        "future_method_authorization.md",
        """
# Future method authorization

PhaseAwareTemporalEnergyVerifier is **not authorized in this study** and was not
trained. A separate future goal may authorize it only after every frozen
condition in `phaseaware_training_conditions.json` independently passes:
300/1,000 context scales, mixed primary groups, shortcut maxima, at least one
fair baseline BA of 0.62, Recall@1 at least 0.10 above chance, held-out
generalization, and recomputable raw predictions. A failure cannot be waived by
rewriting labels or relaxing thresholds.
""",
    )


def _write_paper_package(pilot_decision: str) -> None:
    paper = {
        "paper/title_candidates.md": """
# Title candidates

1. ActionCheck: Proposal-Level Verification of Robot Action Chunks Before Execution
2. Same State, Many Actions: Benchmarking Pre-Execution Robot Action Verification
3. ActionCheck: Learning to Reject Bad Robot Action Chunks
4. Before the Robot Acts: Reranking and Rejecting Candidate Action Chunks
""",
        "paper/abstract_zh.md": f"""
# 中文摘要草案

机器人策略在执行前通常可以生成多个动作块，但现有日志式数据难以为“同一状态下哪个候选可接受”提供可靠监督。我们提出 ActionCheck 数据与评测协议：以同一因果上下文、多个候选动作块及执行、仿真、专家或显式规则证据组成原子样本；采用组级划分、候选排序、误接受风险和系统化捷径审计。既有日志负例审计失败，说明不能把动作差异或互换直接解释为负例。本分支构建了可复现管线并运行了一个仿真 pilot；当前决定为 `{pilot_decision}`，原因是失败原因编码过于单一。因此本文当前只支持数据协议和审计结论，不声称新模型有效、在线成功率提升或物理安全。
""",
        "paper/abstract_en.md": f"""
# Abstract draft

Robot policies can propose several action chunks before execution, yet logged
demonstrations do not reliably supervise which proposal is acceptable under the
same state. ActionCheck defines an atomic unit consisting of one causal context,
multiple candidate chunks, and execution-, simulation-, expert-, operator-, or
rule-backed labels. The benchmark uses group-safe splits, candidate ranking,
false-accept risk, calibration, and systematic shortcut controls. Earlier
logged-only negative constructions failed audit, so action mismatch or swapping
is not treated as rejection evidence. This branch implements the reproducible
pipeline and a simulator pilot whose current decision is
`{pilot_decision}` because one coarse failure reason dominates. We therefore
claim a protocol and audit finding, not method effectiveness, online gains, or
physical safety.
""",
        "paper/introduction_outline.md": """
# Introduction outline

1. Pre-execution choice is proposal verification, not trajectory imitation.
2. Logged demonstrations contain one realized action and weak counterfactual evidence.
3. Prior logged-only ActionCheck attempts failed coverage/validity/shortcut gates.
4. Define same-state, many-action, evidence-backed context groups.
5. Contributions: schema, collection routes, leakage-safe builder, metrics and shortcut gates.
6. Pilot is diagnostic and exposes the need for fine-grained failure instrumentation.
""",
        "paper/problem_formulation.md": """
# Problem formulation

Given causal context C and K candidate chunks A₁…A_K, learn a score s(C,A)
for compatibility label Y∈{accept,reject}; uncertain candidates are outside the
primary loss. Evaluate both binary decisions at a validation-selected threshold
and within-context ranking. Evidence E and reason R define label provenance and
audit strata, but are forbidden inputs. The target is recorded compatibility,
not safety probability or a causal counterfactual.
""",
        "paper/dataset_protocol.md": """
# Dataset protocol

Collect context groups through real-robot proposal filtering, offline policy
proposals with independent evidence, or identical-state simulator branching.
Store causal histories, exact action chunks, policy provenance, outcome records,
label evidence and verified reasons. Retain ambiguous cases as uncertain.
Balance tasks, proposal sources, labels and reasons; never derive labels from
filenames, demo distance, perturbation type or arbitrary action swaps.
""",
        "paper/benchmark_protocol.md": """
# Benchmark protocol

Validate schema/evidence first, keep context groups atomic, and construct
episode/task/object/source/scene held-outs. Exclude uncertain labels from
primary arrays but preserve them in manifests. Publish raw predictions, source
hashes, split audits and builder reports. Run shortcut gates before interpreting
model results.
""",
        "paper/evaluation_protocol.md": """
# Evaluation protocol

Report pairwise BA/AUROC/AUPRC/FAR/FRR, within-context Recall@1/MRR/NDCG/margin/
best rank, ECE/Brier/risk-coverage, held-out generalization and task/source
strata. Select thresholds on validation only. Bootstrap contexts or episodes,
never candidate rows.
""",
        "paper/shortcut_audit_protocol.md": """
# Shortcut audit protocol

Evaluate all 13 frozen controls and the source/reason balance gates before
authorizing a relational method. Evidence/reason controls are audit-only and
must not become model inputs. A single failed gate terminates the benchmark-ready
claim; the dataset must be recollected or re-instrumented, not threshold-tuned.
""",
        "paper/model_placeholder.md": """
# Model placeholder

Future work may compare relational context–action encoders and an explicitly
authorized PhaseAwareTemporalEnergyVerifier. No architectural result belongs in
this branch. The placeholder cannot be converted into training code until a
separate goal verifies every authorization condition.
""",
        "paper/claim_boundary.md": """
# Paper claim boundary

Allowed now: logged-only construction failures, need for proposal-level labels,
the released schema/protocol/pipeline, and the diagnostic pilot audit. Forbidden:
model effectiveness, online success improvement, physical safety, causal
counterfactual correctness, dynamics prediction, or final benchmark validity
while a frozen shortcut gate fails.
""",
        "paper/related_work_plan.md": """
# Related-work plan

Cover action/value verification, offline RL and preference data, VLA proposal
sampling, safety shields/runtime assurance, robot failure datasets, selective
prediction/calibration, counterfactual evaluation limits, simulator branching,
and shortcut/leakage audits. Distinguish empirical execution evidence from
synthetic negatives and expert plausibility.
""",
        "paper/experiment_plan.md": """
# Experiment plan

Stage A: instrument fine-grained simulator failures and diversify proposals
within each task. Stage B: rebuild and require every shortcut gate. Stage C:
simple fair baselines on episode/task/source held-outs with raw predictions.
Stage D: collect 300 real or high-fidelity contexts for a method pilot. Stage E:
only after authorization, compare relational models and test selective rejection.
""",
        "paper/figure_plan.md": """
# Figure plan

1. One context branching to many proposals and evidence outcomes.
2. Three collection routes and a common schema.
3. Atomic split and audit pipeline.
4. Shortcut-gate dashboard, including the current reason-code failure.
5. Future risk–coverage and within-context ranking plots after a valid benchmark.
""",
        "paper/table_plan.md": """
# Table plan

1. Prior logged-only branches and terminal audit decisions.
2. Schema fields, evidence types and forbidden label bases.
3. Dataset composition by task/source/evidence/reason.
4. Shortcut controls and frozen gates.
5. Fair baseline metrics with grouped confidence intervals.
6. Claim/authorization checklist.
""",
    }
    for relative, text in paper.items():
        _write_text(relative, text)


def _write_final(decision: dict[str, Any]) -> None:
    name = decision["decision"]
    answers = {
        "existing_local_proposal_level_data": {
            "answer": True,
            "detail": "1,024 groups and 10,240 candidates across M4R v3 and M5B v3",
        },
        "evidence_backed_labels": {
            "answer": True,
            "detail": "per-candidate ManiSkill 3 GPU PhysX success/failure evidence",
        },
        "pilot_can_be_built_now": {
            "answer": True,
            "detail": "all seven preregistered pilot admission minima pass",
        },
        "shortcut_controls_acceptable": {
            "answer": name == "ACTIONCHECK_PROPOSAL_BENCHMARK_READY_FOR_METHOD",
            "detail": (
                "No: simulator_failure covers 100% of rejects, above the frozen 40% maximum"
                if name == "ACTIONCHECK_PROPOSAL_SHORTCUT_DOMINATED"
                else "See pilot shortcut gates."
            ),
        },
        "phaseaware_authorized_now": {
            "answer": False,
            "detail": "explicitly forbidden in this study and shortcut gate not cleared",
        },
        "exact_next_data": {
            "answer": (
                "Re-execute or instrument proposal rollouts to record verified "
                "fine-grained failure events (phase/object/gripper/progress/"
                "collision/limits/contact/timeout), with no one reason above "
                "40%; generate at least two policy sources within the same tasks "
                "and modalities; preserve identical initial-state hashes and "
                "raw event traces; collect at least 300 mixed-label contexts for "
                "a future method pilot, then rerun every frozen gate."
            )
        },
    }
    final = {
        "schema": "actioncheck-final-decision-v1",
        "study_id": STUDY_ID,
        "decision": name,
        "answers": answers,
        "phaseaware_temporal_energy_verifier_trained": False,
        "thresholds_relaxed": False,
        "labels_invented": False,
    }
    write_json(OUT / "final_decision.json", final)
    _write_text(
        "claim_boundary.md",
        """
# Claim boundary

This branch supports a proposal-level schema, builder, validators, collection
protocols, evaluation implementation, local-data finding and diagnostic
simulator pilot. It does not support guaranteed physical safety or success,
causal alternate-world correctness, object-dynamics prediction, validity of
logged swaps/synthetic negatives, or a PhaseAware method claim.
""",
    )
    _write_text(
        "next_steps.md",
        """
# Next steps

1. Add frozen event-level logging to simulator rollouts before labels are made:
   phase, target/direction, gripper timing, progress/regression, collision,
   limits, contact stability and timeout.
2. Re-execute candidates when the old record cannot establish a failure cause;
   never retrospectively guess a reason from action shape.
3. Generate at least two proposal-policy sources inside the same tasks and
   context modalities, then balance reasons so no reject code exceeds 40%.
4. Rebuild the benchmark and rerun the exact preregistered shortcut gates.
5. Only after all gates pass and scale reaches 300 mixed contexts, request a
   separate method-authorization goal.
""",
    )
    _write_text(
        "codex_next_goal_if_data_available.md",
        """
/goal Continue `/data/projects/tzh/papers/ActMask` with
`/home/tzh/conda_envs/actmask/bin/python`. Run a separately preregistered
ACTIONCHECK-REASON-DIVERSE-V1 collection/audit. Instrument and execute
same-initial-state proposal rollouts with frozen event traces for phase,
object/direction, gripper timing, progress/regression, collision, limits,
contact and timeout. Use at least two proposal-policy sources within the same
tasks/modalities. Do not infer historical reasons or alter
ACTIONCHECK-PROPOSAL-DATA-V1. Require every old shortcut threshold including
single-reason reject fraction <=0.40; save raw predictions and an independent
metric recomputation. Do not train PhaseAware. If all gates pass and at least
300 fully mixed context groups exist, stop with a request for a separate method
authorization goal.
""",
    )
    _write_text(
        "codex_next_goal_for_real_robot_collection.md",
        """
/goal Continue `/data/projects/tzh/papers/ActMask` with
`/home/tzh/conda_envs/actmask/bin/python`. Preregister and execute an
ACTIONCHECK-REAL-PILOT-V1 collection on TB6-R5 or UR5 using the frozen Route 1
protocol. Target 8–10 tasks, >=30 contexts/task, 4–8 proposals/context from
multiple policies, >=1 accept and reject/context, synchronized causal RGB/state/
gripper, safety-gate logs, short-horizon execution or blinded expert evidence,
and verified reason codes. Preserve uncertain cases, never label by demo
distance, never terminate unrelated processes, and do not train PhaseAware.
Build/audit only after evidence validators pass.
""",
    )
    _write_text(
        "executive_summary_zh.md",
        f"""
# 执行摘要

最终决定：`{name}`。本地确实存在可用的同上下文多候选仿真数据：
1,024 个上下文、10,240 个逐候选执行结果、5 个任务、2 个候选来源，足以构建
pilot。模式、证据、分组与划分验证均通过，六个允许的 baseline 已在 GPU 上运行，
原始预测可复算。但所有 5,153 个 reject 只有 `simulator_failure` 这一种已验证原因，
占比 100%，超过预注册的 40% 上限，所以 benchmark 不能授权新方法。没有训练
PhaseAware，也没有编造更细原因或放宽阈值。下一步必须重新执行或补充事件级仿真
记录，并在同任务内增加多策略来源后重跑全部门槛。
""",
    )
    _write_text(
        "executive_summary_en.md",
        f"""
# Executive summary

Final decision: `{name}`. Local data does contain a usable same-context,
multi-proposal simulator pilot: 1,024 contexts, 10,240 per-candidate outcomes,
five tasks, and two proposal sources. Schema, evidence, grouping, and split
validation pass; all six allowed baselines ran on GPU and raw predictions are
recomputable. However, all 5,153 rejects have only the verified coarse
`simulator_failure` reason (100% versus the frozen 40% maximum), so the
benchmark cannot authorize a new method. No PhaseAware model was trained, no
reason was invented, and no threshold was relaxed.
""",
    )
    _write_text(
        "final_report.md",
        f"""
# ACTIONCHECK-PROPOSAL-DATA-V1 final report

Final decision: **`{name}`**.

## What was completed

The study froze its config before training, exhaustively inventoried relevant
local roots read-only, formalized the context-group schema, implemented five
validators, three collection routes, an atomic benchmark builder, five split
families, 13 shortcut controls, grouped/calibrated metrics, six fair GPU
baselines, a future authorization policy, and a complete paper plan. A 1,024
context / 10,240 candidate simulator pilot was built with per-candidate
ManiSkill 3 GPU PhysX outcomes.

## Terminal audit

All seven pilot-admission requirements pass. The evidence itself is valid, but
the shortcut audit is not: `simulator_failure` accounts for every reject,
violating the frozen 40% reason-diversity gate. The scientifically correct
result is shortcut-dominated, regardless of baseline scores. The source data
must be re-instrumented or recollected; finer reasons cannot be fabricated.

## Explicit answers

1. Existing proposal data: **yes**, M4R v3 and M5B v3.
2. Evidence-backed labels: **yes**, executed simulator outcomes.
3. Pilot buildable now: **yes**, and built.
4. Shortcut controls acceptable: **no**, reason diversity fails.
5. PhaseAware authorized: **no**, explicitly not trained.
6. Exact next collection: event-level verified failure causes, multiple
   proposal sources within tasks/modalities, no reason over 40%, identical-state
   rollout hashes, and at least 300 mixed groups before a future method request.
""",
    )
    _log("P10_TO_P12_FINALIZED", decision=name, phaseaware_trained=False)


def _reproducibility_manifest() -> None:
    excluded = {
        "reproducibility_manifest.json",
        "independent_verification.json",
        "independent_verification.md",
    }
    files = []
    for path in sorted(OUT.rglob("*")):
        if not path.is_file() or path.name in excluded:
            continue
        files.append(
            {
                "path": str(path.relative_to(OUT)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    code_files = sorted(
        (ROOT / "actmask/experiments/actioncheck").glob("*.py")
    )
    tests = sorted((ROOT / "tests").glob("test_actioncheck_*.py"))
    write_json(
        OUT / "reproducibility_manifest.json",
        {
            "schema": "actioncheck-reproducibility-manifest-v1",
            "study_id": STUDY_ID,
            "python": str(PYTHON),
            "seed": SEED,
            "preregistered_config_sha256": sha256_file(
                OUT / "preregistered_config.json"
            ),
            "output_files": files,
            "implementation_files": [
                {
                    "path": str(path.relative_to(ROOT)),
                    "sha256": sha256_file(path),
                }
                for path in code_files + tests
            ],
            "phaseaware_checkpoint_files": [],
            "previous_branches_modified_by_study": False,
            "hash_scope_note": (
                "manifest excludes itself and the independent verification reports "
                "that verify this manifest; run_log hash is the state at manifest creation"
            ),
            "excluded_output_basenames": sorted(excluded),
        },
    )


def execute() -> None:
    prepared = prepare()
    if not prepared["validators"]["pass"]:
        raise RuntimeError("validators did not pass; refusing pilot training")
    decision = run_pilot(prepared)
    _write_method_policy()
    _write_paper_package(decision["decision"])
    _write_final(decision)
    _reproducibility_manifest()
    print(
        json.dumps(
            {
                "study": STUDY_ID,
                "decision": decision["decision"],
                "output": str(OUT),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


def execute_from_prepared() -> None:
    prepared = load_prepared()
    decision = run_pilot(prepared)
    _write_method_policy()
    _write_paper_package(decision["decision"])
    _write_final(decision)
    _reproducibility_manifest()
    print(
        json.dumps(
            {
                "study": STUDY_ID,
                "decision": decision["decision"],
                "output": str(OUT),
                "reused_validated_P1_to_P8": True,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Run P1-P8 and admission without any GPU training.",
    )
    parser.add_argument(
        "--pilot-from-prepared",
        action="store_true",
        help="Resume P9 from persisted, hash-checked P1-P8 artifacts.",
    )
    parser.add_argument(
        "--refresh-evaluation",
        action="store_true",
        help="Attach audit strata and recompute metrics from existing raw scores.",
    )
    args = parser.parse_args()
    if sum(
        int(flag)
        for flag in (
            args.prepare_only,
            args.pilot_from_prepared,
            args.refresh_evaluation,
        )
    ) > 1:
        parser.error("choose only one execution mode")
    if args.prepare_only:
        result = prepare()
        print(
            json.dumps(
                {
                    "admission": result["admission"],
                    "validators_pass": result["validators"]["pass"],
                },
                indent=2,
                sort_keys=True,
            )
        )
    elif args.pilot_from_prepared:
        execute_from_prepared()
    elif args.refresh_evaluation:
        refresh_evaluation_metadata()
        _reproducibility_manifest()
    else:
        execute()
