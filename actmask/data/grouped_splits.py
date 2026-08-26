"""Deterministic group-level split manifests for the hardened benchmark.

The manifest is built before counterfactual variants are expanded.  A record
therefore represents one indivisible base scene / pair and can occur in only
one ID split.  OOD records use separate integer namespaces so they cannot
silently alias an in-distribution scene.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


ID_SPLITS = ("train", "val", "test")
OOD_DOMAINS = (
    "ood_velocity",
    "ood_acceleration",
    "ood_duration",
    "ood_radius",
    "ood_curvature",
    "ood_point_noise",
    "ood_distractor_count",
)
DEFAULT_SCENARIOS = (
    "velocity_counterfactual",
    "action_counterfactual",
    "timing_counterfactual",
    "constant_acceleration",
    "circular_motion",
    "delayed_onset",
    "multiple_moving_objects",
    "fast_moving_distractor",
    "static_near_noncontact",
    "far_future_entry",
    "near_miss_action",
    "radius_counterfactual",
)


@dataclass(frozen=True)
class GroupRecord:
    """Identity and routing metadata for one complete counterfactual group."""

    group_id: int
    base_scene_id: int
    pair_id: int
    trajectory_family_id: int
    geometry_seed: int
    scenario_seed: int
    scenario: str
    split: str
    domain: str


GroupManifest = dict[str, tuple[GroupRecord, ...]]


def _positive_integer(name: str, value: int, *, allow_zero: bool = False) -> int:
    if not isinstance(value, (int, np.integer)) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    lower = 0 if allow_zero else 1
    if value < lower:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {qualifier}")
    return value


def _allocate_counts(
    total: int,
    fractions: Sequence[float],
    split_counts: Mapping[str, int] | None,
) -> tuple[int, int, int]:
    if split_counts is not None:
        unknown = set(split_counts) - set(ID_SPLITS)
        missing = set(ID_SPLITS) - set(split_counts)
        if unknown or missing:
            raise ValueError(
                "split_counts must contain exactly train, val, and test; "
                f"missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        counts = tuple(
            _positive_integer(
                f"split_counts[{name!r}]", split_counts[name], allow_zero=True
            )
            for name in ID_SPLITS
        )
        if sum(counts) != total:
            raise ValueError(
                f"split_counts sum to {sum(counts)}, expected total_id_groups={total}"
            )
        return counts

    if len(fractions) != 3:
        raise ValueError("split_fractions must contain train, val, and test fractions")
    values = np.asarray(fractions, dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("split_fractions must be finite and non-negative")
    fraction_sum = float(values.sum())
    if fraction_sum <= 0.0:
        raise ValueError("at least one split fraction must be positive")
    quotas = values / fraction_sum * total
    floors = np.floor(quotas).astype(np.int64)
    remainder = total - int(floors.sum())
    # Stable largest-remainder allocation: equal remainders prefer train, then
    # validation, then test.  Counts always sum exactly to total.
    order = np.argsort(-(quotas - floors), kind="stable")
    floors[order[:remainder]] += 1
    return tuple(int(value) for value in floors)  # type: ignore[return-value]


def _seed_namespace(master_seed: int) -> int:
    if not isinstance(master_seed, (int, np.integer)) or isinstance(master_seed, bool):
        raise TypeError("master_seed must be an integer")
    # Thirty-one seed bits + three domain bits + 27 local-index bits fit in a
    # positive signed int64.  Negative seeds map deterministically by masking.
    return int(master_seed) & 0x7FFFFFFF


def _group_id(seed_namespace: int, domain_index: int, local_index: int) -> int:
    if local_index >= 2**27:
        raise ValueError("a domain cannot contain 2**27 or more groups")
    return (seed_namespace << 31) | (domain_index << 27) | local_index


def _record(
    *,
    seed_namespace: int,
    domain_index: int,
    local_index: int,
    scenario: str,
    split: str,
    domain: str,
) -> GroupRecord:
    group_id = _group_id(seed_namespace, domain_index, local_index)
    # The doubled identifiers remain below 2**63 and are unique throughout a
    # manifest.  They are also safe scalar torch.int64 values.
    return GroupRecord(
        group_id=group_id,
        base_scene_id=group_id,
        pair_id=group_id,
        # A trajectory family is the complete future-dynamics realization for
        # one base scene, before its counterfactual action/input variants are
        # expanded.  It intentionally shares the group identity so it is
        # indivisible across train/validation/test/OOD partitions.
        trajectory_family_id=group_id,
        geometry_seed=group_id * 2,
        scenario_seed=group_id * 2 + 1,
        scenario=scenario,
        split=split,
        domain=domain,
    )


def _stratified_order(
    total: int, scenarios: tuple[str, ...], master_seed: int
) -> list[int]:
    """Interleave independently shuffled scenario buckets deterministically."""

    buckets: list[list[int]] = []
    seed_word = master_seed & 0xFFFFFFFF
    for scenario_index in range(len(scenarios)):
        bucket = list(range(scenario_index, total, len(scenarios)))
        rng = np.random.default_rng(
            np.random.SeedSequence([seed_word, 0x53504C49, scenario_index])
        )
        rng.shuffle(bucket)
        buckets.append(bucket)

    order: list[int] = []
    max_size = max((len(bucket) for bucket in buckets), default=0)
    for offset in range(max_size):
        for bucket in buckets:
            if offset < len(bucket):
                order.append(bucket[offset])
    return order


def build_group_manifest(
    *,
    master_seed: int = 2026,
    total_id_groups: int = 96,
    split_fractions: Sequence[float] = (0.6, 0.2, 0.2),
    split_counts: Mapping[str, int] | None = None,
    ood_groups_per_domain: int = 12,
    scenarios: Sequence[str] = DEFAULT_SCENARIOS,
) -> GroupManifest:
    """Build a deterministic, leakage-free ID/OOD group manifest.

    ``split_counts`` is the authoritative option when exact counts are needed;
    otherwise normalized ``split_fractions`` use largest-remainder allocation.
    Manifest keys are ``train``, ``val``, ``test`` and the seven names in
    :data:`OOD_DOMAINS`.  Each OOD key contains test-only groups.
    """

    total_id_groups = _positive_integer("total_id_groups", total_id_groups)
    ood_groups_per_domain = _positive_integer(
        "ood_groups_per_domain", ood_groups_per_domain, allow_zero=True
    )
    scenario_tuple = tuple(str(value) for value in scenarios)
    if not scenario_tuple or any(not value for value in scenario_tuple):
        raise ValueError("scenarios must contain at least one non-empty name")
    if len(set(scenario_tuple)) != len(scenario_tuple):
        raise ValueError("scenario names must be unique")

    seed_namespace = _seed_namespace(master_seed)
    train_count, val_count, test_count = _allocate_counts(
        total_id_groups, split_fractions, split_counts
    )
    ordered_indices = _stratified_order(total_id_groups, scenario_tuple, master_seed)
    boundaries = (train_count, train_count + val_count, total_id_groups)
    manifest: GroupManifest = {}
    start = 0
    for split, stop in zip(ID_SPLITS, boundaries, strict=True):
        manifest[split] = tuple(
            _record(
                seed_namespace=seed_namespace,
                domain_index=0,
                local_index=local_index,
                scenario=scenario_tuple[local_index % len(scenario_tuple)],
                split=split,
                domain="id",
            )
            for local_index in ordered_indices[start:stop]
        )
        start = stop

    for domain_index, domain in enumerate(OOD_DOMAINS, start=1):
        manifest[domain] = tuple(
            _record(
                seed_namespace=seed_namespace,
                domain_index=domain_index,
                local_index=local_index,
                scenario=scenario_tuple[local_index % len(scenario_tuple)],
                split="test",
                domain=domain,
            )
            for local_index in range(ood_groups_per_domain)
        )

    assert_no_group_leakage(manifest)
    return manifest


def assert_no_group_leakage(manifest: Mapping[str, Sequence[GroupRecord]]) -> None:
    """Raise ``ValueError`` when any scene identity occurs in two partitions."""

    identity_fields = (
        "group_id",
        "base_scene_id",
        "pair_id",
        "trajectory_family_id",
        "geometry_seed",
        "scenario_seed",
    )
    owners: dict[str, dict[int, str]] = {field: {} for field in identity_fields}
    for partition, records in manifest.items():
        for record in records:
            expected_partition = record.split if record.domain == "id" else record.domain
            if partition != expected_partition:
                raise ValueError(
                    f"record {record.group_id} is stored under {partition!r}, "
                    f"expected {expected_partition!r}"
                )
            for field in identity_fields:
                value = int(getattr(record, field))
                previous = owners[field].get(value)
                if previous is not None and previous != partition:
                    raise ValueError(
                        f"{field}={value} leaks across {previous!r} and {partition!r}"
                    )
                owners[field][value] = partition


def split_statistics(
    manifest: Mapping[str, Sequence[GroupRecord]],
) -> dict[str, dict[str, object]]:
    """Return JSON-ready group and scenario counts for each partition."""

    statistics: dict[str, dict[str, object]] = {}
    for partition, records in manifest.items():
        scenario_counts: dict[str, int] = {}
        for record in records:
            scenario_counts[record.scenario] = scenario_counts.get(record.scenario, 0) + 1
        statistics[partition] = {
            "groups": len(records),
            "samples": len(records) * 2,
            "scenarios": dict(sorted(scenario_counts.items())),
        }
    return statistics


__all__ = [
    "DEFAULT_SCENARIOS",
    "ID_SPLITS",
    "OOD_DOMAINS",
    "GroupManifest",
    "GroupRecord",
    "assert_no_group_leakage",
    "build_group_manifest",
    "split_statistics",
]
