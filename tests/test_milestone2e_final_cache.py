"""Focused contract tests for the versioned Milestone 2E-Final baseline cache."""

from __future__ import annotations

import torch
import pytest

from actmask.experiments.milestone2e_final_cache import (
    CORRECTED_NDCG_METRIC_SCHEMA_VERSION,
    LEGACY_NDCG_METRIC_SCHEMA_VERSION,
    FairBaselineCache,
    FairBaselineCacheKey,
    FairBaselineWorld,
    StaleFairBaselineCacheError,
)


_CODE_PROVENANCE = {
    "code_version": "cache-test-v1",
    "source_tree_sha256": "a" * 64,
}


def _key(
    *,
    config: dict[str, object] | None = None,
    candidate_count: int = 5,
    split_id: str = "id",
    observation_view_id: str = "original",
    corruption_id: str = "none",
    metric_schema_version: str = CORRECTED_NDCG_METRIC_SCHEMA_VERSION,
    reverse_candidate_order: bool = False,
    baseline_parameters: dict[str, object] | None = None,
) -> FairBaselineCacheKey:
    worlds = tuple(
        FairBaselineWorld(
            world_id=f"world-{index}",
            candidate_set_id=f"candidate-set-{index}",
            candidate_ids=tuple(
                f"candidate-{index}-{candidate}"
                for candidate in (
                    reversed(range(candidate_count)) if reverse_candidate_order else range(candidate_count)
                )
            ),
        )
        for index in range(2)
    )
    return FairBaselineCacheKey.from_context(
        baseline_name="MultiHypothesisTrajectoryProximity",
        baseline_parameters=baseline_parameters,
        config=config or {"dataset": {"observation_noise_std": 0.006}, "ranking": {"candidate_counts": [5, 10, 20, 50]}},
        code_provenance=_CODE_PROVENANCE,
        worlds=worlds,
        split_id=split_id,
        observation_view_id=observation_view_id,
        corruption_id=corruption_id,
        candidate_count=candidate_count,
        metric_schema_version=metric_schema_version,
    )


def test_fair_baseline_cache_hit_after_initial_miss_and_self_describes_entry(tmp_path) -> None:
    cache = FairBaselineCache(tmp_path / "fair_baseline_cache")
    key = _key()
    calls: list[str] = []

    first = cache.get_or_compute(key, lambda: calls.append("computed") or {"utility": [0.2, 0.8]})
    second = cache.get_or_compute(key, lambda: pytest.fail("a valid cache hit must not recompute"))

    assert calls == ["computed"]
    assert not first.cache_hit
    assert second.cache_hit
    assert second.payload == {"utility": [0.2, 0.8]}
    assert first.path == second.path
    envelope = torch.load(first.path, map_location="cpu", weights_only=False)
    assert envelope["schema_version"] == key.schema_version
    assert envelope["config_hash"] == key.config_hash
    assert envelope["code_provenance"] == _CODE_PROVENANCE
    assert envelope["evaluation_schema_version"] == key.evaluation_schema_version
    assert envelope["metric_schema_version"] == CORRECTED_NDCG_METRIC_SCHEMA_VERSION
    assert envelope["split_id"] == "id"
    assert envelope["world_ids"] == ["world-0", "world-1"]
    assert envelope["candidate_set_ids"] == ["candidate-set-0", "candidate-set-1"]
    assert envelope["candidate_count"] == 5
    assert envelope["observation_view_id"] == "original"
    assert envelope["corruption_id"] == "none"
    assert envelope["baseline_name"] == "MultiHypothesisTrajectoryProximity"
    assert envelope["baseline_parameters"] == {}
    assert envelope["created_at_utc"].endswith("Z")


def test_relevant_configuration_change_is_a_miss_not_a_stale_hit(tmp_path) -> None:
    cache = FairBaselineCache(tmp_path / "fair_baseline_cache")
    original = _key(config={"dataset": {"observation_noise_std": 0.006}})
    changed = _key(config={"dataset": {"observation_noise_std": 0.020}})

    first = cache.get_or_compute(original, lambda: "original-config")
    assert cache.get(changed) is None
    second = cache.get_or_compute(changed, lambda: "changed-config")

    assert first.path != second.path
    assert first.payload == "original-config"
    assert second.payload == "changed-config"
    assert not second.cache_hit
    assert cache.get(original).payload == "original-config"


def test_candidate_count_keys_keep_c5_c10_c20_and_c50_separate(tmp_path) -> None:
    cache = FairBaselineCache(tmp_path / "fair_baseline_cache")
    results = {
        count: cache.get_or_compute(_key(candidate_count=count), lambda count=count: f"C{count}")
        for count in (5, 10, 20, 50)
    }

    assert len({result.path for result in results.values()}) == 4
    assert {count: result.payload for count, result in results.items()} == {
        5: "C5",
        10: "C10",
        20: "C20",
        50: "C50",
    }
    assert all(not result.cache_hit for result in results.values())


def test_id_and_ood_views_do_not_collide_even_with_matching_world_membership(tmp_path) -> None:
    cache = FairBaselineCache(tmp_path / "fair_baseline_cache")
    id_key = _key(split_id="id", corruption_id="none")
    ood_key = _key(split_id="ood/unseen_velocity", corruption_id="velocity_magnitude_shift")

    id_result = cache.get_or_compute(id_key, lambda: "id")
    assert cache.get(ood_key) is None
    ood_result = cache.get_or_compute(ood_key, lambda: "ood")

    assert id_result.path != ood_result.path
    assert id_result.payload == "id"
    assert ood_result.payload == "ood"


def test_observation_view_candidate_membership_and_baseline_parameters_are_all_keyed(tmp_path) -> None:
    cache = FairBaselineCache(tmp_path / "fair_baseline_cache")
    original = _key(baseline_parameters={"trajectory_hypotheses": 8})
    rotated = _key(
        observation_view_id="global_so3",
        baseline_parameters={"trajectory_hypotheses": 8},
    )
    reordered = _key(
        reverse_candidate_order=True,
        baseline_parameters={"trajectory_hypotheses": 8},
    )
    changed_baseline = _key(baseline_parameters={"trajectory_hypotheses": 16})

    results = [
        cache.get_or_compute(key, lambda index=index: f"payload-{index}")
        for index, key in enumerate((original, rotated, reordered, changed_baseline))
    ]

    assert len({result.path for result in results}) == 4
    assert [result.payload for result in results] == ["payload-0", "payload-1", "payload-2", "payload-3"]


def test_legacy_and_corrected_ndcg_metric_schemas_cannot_share_a_cache_entry(tmp_path) -> None:
    cache = FairBaselineCache(tmp_path / "fair_baseline_cache")
    corrected = _key(metric_schema_version=CORRECTED_NDCG_METRIC_SCHEMA_VERSION)
    legacy = _key(metric_schema_version=LEGACY_NDCG_METRIC_SCHEMA_VERSION)

    corrected_result = cache.get_or_compute(corrected, lambda: "corrected-ndcg")
    assert cache.get(legacy) is None
    legacy_result = cache.get_or_compute(legacy, lambda: "legacy-ndcg")

    assert corrected_result.path != legacy_result.path
    assert corrected_result.payload == "corrected-ndcg"
    assert legacy_result.payload == "legacy-ndcg"


def test_hand_edited_or_stale_metadata_is_rejected_instead_of_used(tmp_path) -> None:
    cache = FairBaselineCache(tmp_path / "fair_baseline_cache")
    key = _key()
    result = cache.get_or_compute(key, lambda: {"utility": [0.2, 0.8]})
    envelope = torch.load(result.path, map_location="cpu", weights_only=False)
    envelope["config_hash"] = "stale-config-digest"
    torch.save(envelope, result.path)

    with pytest.raises(StaleFairBaselineCacheError, match="direct provenance mismatch"):
        cache.get(key)
