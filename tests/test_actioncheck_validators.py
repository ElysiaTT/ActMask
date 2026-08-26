"""Evidence, grouping, splitting and builder contracts."""
from __future__ import annotations

import json
from copy import deepcopy

from actmask.experiments.actioncheck.build_actioncheck_benchmark import (
    build_benchmark,
)
from actmask.experiments.actioncheck.create_actioncheck_splits import create_splits
from actmask.experiments.actioncheck.validate_candidate_balance import (
    candidate_balance,
)
from actmask.experiments.actioncheck.validate_evidence import validate_evidence
from actmask.experiments.actioncheck.validate_group_structure import validate_groups
from actmask.experiments.actioncheck.validate_splits import validate_splits
from actmask.experiments.actioncheck.common import write_jsonl
from tests.test_actioncheck_schema import example_group


def groups(count: int = 12) -> list[dict]:
    result = []
    for index in range(count):
        group = deepcopy(example_group())
        group["context_id"] = f"context-{index:03d}"
        group["episode_id"] = f"episode-{index:03d}"
        group["task_id"] = f"task-{index % 5}"
        group["split_group"]["object_composition_id"] = f"object-{index % 4}"
        group["split_group"]["scene_id"] = f"scene-{index:03d}"
        for candidate in group["candidates"]:
            candidate["candidate_id"] = (
                f"{group['context_id']}:{candidate['candidate_id']}"
            )
            candidate["source_policy"] = f"policy-{index % 2}"
        result.append(group)
    return result


def test_group_evidence_balance_and_split_validators_pass() -> None:
    rows = groups()
    assert validate_evidence(rows)["pass"]
    assert validate_groups(rows, minimum_candidates=2)["all_contexts_pass"]
    balance = candidate_balance(rows)
    assert balance["label_distribution"] == {"accept": 12, "reject": 12}
    split = create_splits(rows)
    audit = validate_splits(rows, split)
    assert audit["pass"]
    assert audit["no_context_leakage"] and audit["no_episode_leakage"]
    assert audit["task_held_out_complete"]
    assert audit["policy_source_held_out_complete"]


def test_forbidden_reject_basis_and_missing_sim_evidence_fail() -> None:
    row = example_group()
    reject = row["candidates"][1]
    reject["sampling_metadata"]["label_basis"] = "action_distance"
    reject["sim_result"] = None
    report = validate_evidence([row])
    assert not report["pass"]
    errors = report["failures"][0]["errors"]
    assert "forbidden_label_basis:action_distance" in errors
    assert "sim_evidence_without_boolean_result" in errors


def test_builder_keeps_context_atomic_and_excludes_uncertain(tmp_path) -> None:
    rows = groups(15)
    uncertain = deepcopy(rows[0]["candidates"][0])
    uncertain["candidate_id"] = "uncertain-1"
    uncertain["label"] = "uncertain"
    uncertain["evidence_type"] = "ambiguous"
    uncertain["reason_codes"] = ["ambiguous"]
    uncertain["sim_result"] = None
    rows[0]["candidates"].append(uncertain)
    source = tmp_path / "groups.jsonl"
    write_jsonl(source, rows)
    report = build_benchmark([source], tmp_path / "benchmark")
    assert report["context_groups"] == 15
    assert report["primary_candidates"] == 30
    assert report["uncertain_candidates"] == 1
    assert report["split_audit"]["pass"]
    split = json.loads(
        (tmp_path / "benchmark" / "split_manifest.json").read_text()
    )
    assert split["candidate_level_random_splitting"] is False
