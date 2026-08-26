"""RM-OD contracts for source freeze, K-way groups and later score audits."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "outputs" / "actmask" / "rm_od_future_object_dynamics"


def load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text())


def test_preregistered_config_and_source_identity_are_frozen() -> None:
    config = load("preregistered_config.json")
    expected = (ROOT / "preregistered_config.sha256").read_text().strip()
    assert hashlib.sha256((ROOT / "preregistered_config.json").read_bytes()).hexdigest() == expected
    assert config["candidate_retrieval"]["k"] == 8
    assert config["source"]["selected_episodes"] == 100
    inventory = load("source_inventory.json")
    selected = load("selected_trajectory_manifest.json")
    assert not inventory["expanded"] and inventory["same_task_local_total"] == 151
    assert selected["episode_count"] == 100 and selected["task_count"] == 10
    hashes = load("source_hash_manifest.json")
    assert len(hashes["files"]) == 200 and all(len(row["sha256"]) == 64 for row in hashes["files"])


def test_kway_groups_have_matched_same_task_negatives_and_balanced_positive_indices() -> None:
    audit = load("candidate_group_audit.json")
    matching = load("hard_negative_matching_audit.json")
    task = load("task_manifest.json")
    assert task["k"] == 8 and task["groups"] == audit["groups"]
    assert audit["all_groups_exact_k"] and audit["one_positive_per_group"]
    assert audit["positive_index_balanced"] and audit["candidate_tasks_match_anchor"]
    # Repair 1 deliberately makes every K-way candidate set share its source
    # trajectory, eliminating the initially observed source-path shortcut.
    assert matching["same_task"]
    assert matching["negative_source_variant"] == "same_episode_different_anchor"
    assert matching["source_path_balanced_within_group"]
    assert audit["candidate_source_path_constant_within_group"]


def test_temporal_alignment_consequence_audit_and_split_isolation() -> None:
    alignment = load("processed_subset/alignment_audit.json")
    future = load("future_consequence_audit.json")
    leakage = load("leakage_audit.json")
    splits = load("split_audit.json")
    assert alignment["all_pass"] and alignment["timestamps_strict"] and alignment["frames_strict"]
    assert future["future_features_target_or_audit_only"] and future["fair_query_inputs_contain_no_future_features"]
    assert leakage["pass"] and not leakage["source_path_or_folder_used_as_fair_input"]
    assert splits["all_candidate_future_sources_remain_in_own_partition"]
    assert all(not value["candidate_source_leakage"] for value in splits["splits"].values())
