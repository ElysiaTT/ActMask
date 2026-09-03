"""Deterministic interventions and semantic integrity checks."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from .schema import (
    SUITE_SCHEMA,
    BindingDataset,
    dataset_digest,
    load_dataset,
    save_dataset,
    validate_dataset,
)


CASES = (
    "original",
    "pair_preserving",
    "binding_breaking",
    "twin_history_swap",
    "candidate_slot_permutation",
)


def _history_copy(dataset: BindingDataset) -> dict[str, np.ndarray]:
    return {
        "history_actions": dataset.history_actions.copy(),
        "history_effects": dataset.history_effects.copy(),
        "history_times": dataset.history_times.copy(),
        "history_mask": dataset.history_mask.copy(),
    }


def pair_preserving(dataset: BindingDataset, seed: int) -> BindingDataset:
    values = _history_copy(dataset)
    rng = np.random.default_rng(seed)
    for twin in range(dataset.twins):
        for branch in range(2):
            valid = np.flatnonzero(dataset.history_mask[twin, branch])
            permutation = rng.permutation(valid)
            for key in ("history_actions", "history_effects", "history_times"):
                values[key][twin, branch, valid] = getattr(dataset, key)[
                    twin, branch, permutation
                ]
    return replace(dataset, **values)


def binding_breaking(dataset: BindingDataset, seed: int) -> BindingDataset:
    """Preserve action/effect marginals but destroy within-history correspondence."""

    values = _history_copy(dataset)
    rng = np.random.default_rng(seed)
    for twin in range(dataset.twins):
        for branch in range(2):
            valid = np.flatnonzero(dataset.history_mask[twin, branch])
            # A nonzero cyclic shift is always a derangement for length > 1.
            shift = int(rng.integers(1, len(valid)))
            values["history_effects"][twin, branch, valid] = dataset.history_effects[
                twin, branch, np.roll(valid, shift)
            ]
    return replace(dataset, **values)


def twin_history_swap(dataset: BindingDataset) -> BindingDataset:
    values = _history_copy(dataset)
    for key in values:
        values[key] = values[key][:, ::-1].copy()
    return replace(dataset, **values)


def candidate_slot_permutation(dataset: BindingDataset, seed: int) -> BindingDataset:
    rng = np.random.default_rng(seed)
    candidates = dataset.candidates.copy()
    candidate_ids = dataset.candidate_ids.copy()
    true_utility = dataset.true_utility.copy()
    candidate_effects = (
        None if dataset.candidate_effects is None else dataset.candidate_effects.copy()
    )
    for twin in range(dataset.twins):
        for branch in range(2):
            permutation = rng.permutation(dataset.candidates_per_branch)
            candidates[twin, branch] = dataset.candidates[twin, branch, permutation]
            candidate_ids[twin, branch] = dataset.candidate_ids[twin, branch, permutation]
            true_utility[twin, branch] = dataset.true_utility[twin, branch, permutation]
            if candidate_effects is not None:
                candidate_effects[twin, branch] = dataset.candidate_effects[
                    twin, branch, permutation
                ]
    return replace(
        dataset,
        candidates=candidates,
        candidate_ids=candidate_ids,
        true_utility=true_utility,
        candidate_effects=candidate_effects,
    )


def make_cases(dataset: BindingDataset, *, seed: int) -> dict[str, BindingDataset]:
    validate_dataset(dataset)
    return {
        "original": dataset,
        "pair_preserving": pair_preserving(dataset, seed + 1),
        "binding_breaking": binding_breaking(dataset, seed + 2),
        "twin_history_swap": twin_history_swap(dataset),
        "candidate_slot_permutation": candidate_slot_permutation(dataset, seed + 3),
    }


def _rows_by_id(dataset: BindingDataset, value: np.ndarray) -> dict[tuple[str, int, str], np.ndarray]:
    output: dict[tuple[str, int, str], np.ndarray] = {}
    for twin in range(dataset.twins):
        for branch in range(2):
            for slot, candidate_id in enumerate(dataset.candidate_ids[twin, branch]):
                output[(str(dataset.twin_ids[twin]), branch, str(candidate_id))] = np.asarray(
                    value[twin, branch, slot]
                )
    return output


def validate_interventions(cases: dict[str, BindingDataset]) -> dict[str, Any]:
    if set(cases) != set(CASES):
        raise ValueError(f"suite cases must be exactly {CASES}")
    for case in CASES:
        validate_dataset(cases[case])
    original = cases["original"]
    for case, current in cases.items():
        if current.dataset_id != original.dataset_id or current.split != original.split:
            raise ValueError(f"{case} changes dataset identity")
        if not np.array_equal(current.twin_ids, original.twin_ids):
            raise ValueError(f"{case} changes twin IDs")

    pair = cases["pair_preserving"]
    pair_tokens_equal = True
    for twin in range(original.twins):
        for branch in range(2):
            valid = original.history_mask[twin, branch]
            original_tokens = np.concatenate(
                (
                    original.history_actions[twin, branch, valid],
                    original.history_effects[twin, branch, valid],
                    original.history_times[twin, branch, valid, None],
                ),
                axis=1,
            )
            pair_tokens = np.concatenate(
                (
                    pair.history_actions[twin, branch, valid],
                    pair.history_effects[twin, branch, valid],
                    pair.history_times[twin, branch, valid, None],
                ),
                axis=1,
            )
            original_order = np.lexsort(original_tokens.T[::-1])
            pair_order = np.lexsort(pair_tokens.T[::-1])
            pair_tokens_equal &= np.array_equal(
                original_tokens[original_order], pair_tokens[pair_order]
            )

    broken = cases["binding_breaking"]
    binding_marginals_equal = (
        np.array_equal(broken.history_actions, original.history_actions)
        and np.array_equal(broken.history_times, original.history_times)
    )
    binding_changed = False
    for twin in range(original.twins):
        for branch in range(2):
            valid = original.history_mask[twin, branch]
            before = original.history_effects[twin, branch, valid]
            after = broken.history_effects[twin, branch, valid]
            before_order = np.lexsort(before.T[::-1])
            after_order = np.lexsort(after.T[::-1])
            binding_marginals_equal &= np.array_equal(before[before_order], after[after_order])
            binding_changed |= not np.array_equal(before, after)

    swapped = cases["twin_history_swap"]
    twin_swap_exact = all(
        np.array_equal(getattr(swapped, key), getattr(original, key)[:, ::-1])
        for key in ("history_actions", "history_effects", "history_times", "history_mask")
    )
    slot = cases["candidate_slot_permutation"]
    original_utility = _rows_by_id(original, original.true_utility)
    slot_utility = _rows_by_id(slot, slot.true_utility)
    slot_truth_exact = original_utility.keys() == slot_utility.keys() and all(
        np.array_equal(original_utility[key], slot_utility[key]) for key in original_utility
    )
    slot_changed = not np.array_equal(slot.candidate_ids, original.candidate_ids)

    checks = {
        "pair_preserving_token_multisets_equal": bool(pair_tokens_equal),
        "binding_breaking_marginals_equal": bool(binding_marginals_equal),
        "binding_breaking_correspondence_changed": bool(binding_changed),
        "twin_history_swap_exact": bool(twin_swap_exact),
        "candidate_slot_truth_by_id_equal": bool(slot_truth_exact),
        "candidate_slot_order_changed": bool(slot_changed),
        "truth_unchanged_for_history_interventions": bool(
            all(
                np.array_equal(cases[case].true_utility, original.true_utility)
                for case in ("pair_preserving", "binding_breaking", "twin_history_swap")
            )
        ),
    }
    if not all(checks.values()):
        failures = [name for name, passed in checks.items() if not passed]
        raise ValueError(f"intervention integrity failed: {failures}")
    return {"passed": True, "checks": checks}


def build_intervention_suite(
    dataset: BindingDataset,
    output_dir: str | Path,
    *,
    seed: int,
    source: dict[str, Any],
) -> dict[str, Any]:
    """Write an immutable evaluator suite and a provenance manifest."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = make_cases(dataset, seed=seed)
    intervention_audit = validate_interventions(cases)
    manifest_cases: dict[str, Any] = {}
    for name, current in cases.items():
        relative = f"cases/{name}.npz"
        audit = save_dataset(output_dir / relative, current)
        manifest_cases[name] = {
            "path": relative,
            "dataset_digest": audit["dataset_digest"],
        }
    manifest = {
        "schema_version": SUITE_SCHEMA,
        "dataset_id": dataset.dataset_id,
        "split": dataset.split,
        "intervention_seed": int(seed),
        "source": dict(source),
        "cases": manifest_cases,
        "integrity": intervention_audit,
    }
    (output_dir / "suite.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def load_suite(path: str | Path) -> tuple[dict[str, Any], dict[str, BindingDataset]]:
    path = Path(path)
    manifest_path = path / "suite.json" if path.is_dir() else path
    root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SUITE_SCHEMA:
        raise ValueError("unsupported or missing suite schema")
    if set(manifest.get("cases", {})) != set(CASES):
        raise ValueError("suite manifest has incomplete cases")
    cases = {
        name: load_dataset(root / manifest["cases"][name]["path"]) for name in CASES
    }
    for name, dataset in cases.items():
        if dataset_digest(dataset) != manifest["cases"][name]["dataset_digest"]:
            raise ValueError(f"suite digest mismatch for {name}")
    validate_interventions(cases)
    return manifest, cases


__all__ = [
    "CASES",
    "binding_breaking",
    "build_intervention_suite",
    "candidate_slot_permutation",
    "load_suite",
    "make_cases",
    "pair_preserving",
    "twin_history_swap",
    "validate_interventions",
]
