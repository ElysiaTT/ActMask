from __future__ import annotations

import copy

import numpy as np
import torch

from actmask.data.milestone3p_counterfactual import (
    STATIC_MATCH_TOLERANCE,
    CounterfactualTemporalDataset,
    assert_no_counterfactual_split_leakage,
    build_counterfactual_manifest,
    matched_pair_samples,
)
from actmask.experiments.milestone3p_shortcut_audit import extract_static_features


_COUNTS = {"train": 4, "val": 2, "test": 2}


def test_milestone3p_counterfactual_manifest_is_split_disjoint() -> None:
    manifest = build_counterfactual_manifest(split_counts=_COUNTS)
    assert_no_counterfactual_split_leakage(manifest)
    identities = [record.group_id for records in manifest.values() for record in records]
    assert len(identities) == len(set(identities))


def test_milestone3p_matched_pairs_have_exact_static_inputs_and_label_flips() -> None:
    dataset = CounterfactualTemporalDataset(
        split="test", regime="matched_counterfactual", split_counts=_COUNTS, include_hidden_state=False
    )
    label_flips = 0
    for first, second in matched_pair_samples(dataset):
        first_static, _, _, _ = extract_static_features(first)
        second_static, _, _, _ = extract_static_features(second)
        assert torch.equal(first["observable"]["points_history"][-1], second["observable"]["points_history"][-1])
        assert torch.equal(first["observable"]["action_command"], second["observable"]["action_command"])
        assert np.max(np.abs(first_static - second_static)) <= STATIC_MATCH_TOLERANCE
        assert "hidden_state" not in first
        if float(first["targets"]["success"]) != float(second["targets"]["success"]):
            label_flips += 1
    assert label_flips >= 3


def test_milestone3p_static_extractor_ignores_prior_history() -> None:
    dataset = CounterfactualTemporalDataset(
        split="test", regime="matched_counterfactual", split_counts=_COUNTS, include_hidden_state=False
    )
    original = dataset[0]
    altered = copy.deepcopy(original)
    altered["observable"]["points_history"][:-1] += 7.0
    original_static, _, _, _ = extract_static_features(original)
    altered_static, _, _, _ = extract_static_features(altered)
    assert np.array_equal(original_static, altered_static)


def test_milestone3p_generation_is_deterministic_and_candidate_aligned() -> None:
    first = CounterfactualTemporalDataset(
        split="test", regime="matched_counterfactual", split_counts=_COUNTS, include_hidden_state=False
    )
    second = CounterfactualTemporalDataset(
        split="test", regime="matched_counterfactual", split_counts=_COUNTS, include_hidden_state=False
    )
    for index in range(len(first)):
        left, right = first[index], second[index]
        assert left["metadata"] == right["metadata"]
        assert torch.equal(left["observable"]["points_history"], right["observable"]["points_history"])
        assert torch.equal(left["observable"]["action_command"], right["observable"]["action_command"])
    for left, right in matched_pair_samples(first):
        assert left["metadata"]["candidate_slot"] == right["metadata"]["candidate_slot"]
        assert left["metadata"]["static_match_key"] == right["metadata"]["static_match_key"]
