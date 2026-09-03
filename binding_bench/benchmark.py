"""Multi-split manifests, leakage audit, and all-split admission decision."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .evaluator import evaluate_suite
from .interventions import load_suite
from .schema import BindingDataset, dataset_digest


BENCHMARK_SCHEMA = "actmask-binding-benchmark-v1"
EVALUATION_PLAN_SCHEMA = "actmask-binding-evaluation-plan-v1"


def _overlap(values: Mapping[str, set[str]]) -> dict[str, list[str]]:
    names = tuple(values)
    output: dict[str, list[str]] = {}
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            shared = sorted(values[left].intersection(values[right]))
            if shared:
                output[f"{left}::{right}"] = shared[:20]
    return output


def split_leakage_audit(datasets: Mapping[str, BindingDataset]) -> dict[str, Any]:
    """Fail if any source episode, twin, or declared group crosses a split."""

    if len(datasets) < 2:
        raise ValueError("split leakage audit requires at least two datasets")
    twin_sets = {name: set(map(str, value.twin_ids.tolist())) for name, value in datasets.items()}
    episode_sets = {
        name: set(map(str, value.episode_ids.reshape(-1).tolist()))
        for name, value in datasets.items()
    }
    group_sets = {
        name: set(map(str, value.split_group_ids.tolist()))
        for name, value in datasets.items()
    }
    overlap = {
        "twin_ids": _overlap(twin_sets),
        "episode_ids": _overlap(episode_sets),
        "split_group_ids": _overlap(group_sets),
    }
    checks = {f"disjoint_{name}": not values for name, values in overlap.items()}
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "overlap_examples": overlap,
    }


def build_benchmark_manifest(
    path: str | Path,
    *,
    benchmark_id: str,
    suites: Mapping[str, str | Path],
    admission_splits: Sequence[str],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    """Freeze split suite paths and prove group-disjoint evaluation partitions."""

    path = Path(path)
    root = path.parent
    loaded: dict[str, BindingDataset] = {}
    entries: dict[str, Any] = {}
    dataset_ids: set[str] = set()
    for split, suite_path in suites.items():
        absolute = Path(suite_path).resolve()
        manifest, cases = load_suite(absolute)
        original = cases["original"]
        if original.split != split:
            raise ValueError(f"suite key {split!r} does not match dataset split {original.split!r}")
        loaded[split] = original
        dataset_ids.add(original.dataset_id)
        try:
            relative = absolute.relative_to(root.resolve()).as_posix()
        except ValueError:
            relative = str(absolute)
        entries[split] = {
            "suite_path": relative,
            "dataset_digest": dataset_digest(original),
            "twins": original.twins,
            "seeds": sorted(set(map(int, original.seed_ids.tolist()))),
            "split_groups": len(set(original.split_group_ids.tolist())),
            "suite_manifest_schema": manifest["schema_version"],
        }
    if len(dataset_ids) != 1:
        raise ValueError("all splits must share one versioned dataset_id")
    admission_splits = tuple(admission_splits)
    if not admission_splits or any(split not in loaded for split in admission_splits):
        raise ValueError("admission_splits must be a nonempty subset of suites")
    leakage = split_leakage_audit(loaded)
    if not leakage["passed"]:
        raise ValueError(f"cross-split leakage detected: {leakage['overlap_examples']}")
    manifest = {
        "schema_version": BENCHMARK_SCHEMA,
        "benchmark_id": benchmark_id,
        "dataset_id": next(iter(dataset_ids)),
        "source": dict(source),
        "admission_splits": list(admission_splits),
        "splits": entries,
        "split_leakage_audit": leakage,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def load_benchmark_manifest(path: str | Path) -> tuple[dict[str, Any], dict[str, Path]]:
    path = Path(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != BENCHMARK_SCHEMA:
        raise ValueError("unsupported benchmark manifest schema")
    suite_paths: dict[str, Path] = {}
    datasets: dict[str, BindingDataset] = {}
    for split, entry in manifest.get("splits", {}).items():
        suite_path = Path(entry["suite_path"])
        if not suite_path.is_absolute():
            suite_path = path.parent / suite_path
        _, cases = load_suite(suite_path)
        original = cases["original"]
        if original.split != split or dataset_digest(original) != entry["dataset_digest"]:
            raise ValueError(f"benchmark manifest mismatch for split {split}")
        suite_paths[split] = suite_path
        datasets[split] = original
    leakage = split_leakage_audit(datasets)
    if not leakage["passed"] or leakage != manifest["split_leakage_audit"]:
        raise ValueError("benchmark split leakage audit is stale or failed")
    return manifest, suite_paths


def evaluate_benchmark_plan(path: str | Path) -> dict[str, Any]:
    """Execute a JSON plan that maps every split to frozen external outputs."""

    path = Path(path)
    plan = json.loads(path.read_text(encoding="utf-8"))
    if plan.get("schema_version") != EVALUATION_PLAN_SCHEMA:
        raise ValueError("unsupported evaluation plan schema")
    benchmark_path = Path(plan["benchmark_manifest"])
    if not benchmark_path.is_absolute():
        benchmark_path = path.parent / benchmark_path
    benchmark, suite_paths = load_benchmark_manifest(benchmark_path)
    if set(plan.get("splits", {})) != set(suite_paths):
        raise ValueError("evaluation plan must cover every benchmark split")
    split_results: dict[str, Any] = {}
    for split, split_plan in plan["splits"].items():
        method_dirs = {}
        for method_id, method_path in split_plan["method_dirs"].items():
            method_path = Path(method_path)
            if not method_path.is_absolute():
                method_path = path.parent / method_path
            method_dirs[method_id] = method_path
        split_results[split] = evaluate_suite(
            suite_paths[split],
            method_dirs,
            candidate_method=plan["candidate_method"],
            baseline_methods=tuple(plan["baseline_methods"]),
            gates=plan.get("gates"),
            bootstrap_seed=int(plan.get("bootstrap_seed", 91073)) + sorted(suite_paths).index(split),
        )
    admission_splits = tuple(benchmark["admission_splits"])
    split_pass = {split: split_results[split]["decision"] == "METHOD_GO" for split in admission_splits}
    return {
        "schema_version": "actmask-binding-benchmark-evaluation-v1",
        "benchmark_id": benchmark["benchmark_id"],
        "candidate_method": plan["candidate_method"],
        "baseline_methods": plan["baseline_methods"],
        "split_leakage_audit": benchmark["split_leakage_audit"],
        "admission_splits": list(admission_splits),
        "split_pass": split_pass,
        "split_results": split_results,
        "decision": "BENCHMARK_METHOD_GO" if all(split_pass.values()) else "BENCHMARK_METHOD_NO_GO",
        "scope_note": "This decision compares a frozen method on a frozen benchmark; GPU use remains a separate readiness decision.",
    }


__all__ = [
    "BENCHMARK_SCHEMA",
    "EVALUATION_PLAN_SCHEMA",
    "build_benchmark_manifest",
    "evaluate_benchmark_plan",
    "load_benchmark_manifest",
    "split_leakage_audit",
]
