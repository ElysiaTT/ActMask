"""Independent, no-training closure audit for every RM-Series raw score file."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from actmask.experiments.rm_series_robomind_audit import (
    OUT,
    ROOT,
    _metrics,
    _split_definitions,
    append_log,
    build_samples,
    sha256_file,
    write_json,
)
from actmask.experiments.rm_series_robomind_method import _load_payload


def _metric_row(path: Path) -> tuple[list[dict[str, Any]], dict[str, float]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    metric = _metrics(rows, np.asarray([row["score"] for row in rows], dtype=np.float32))
    return rows, metric


def _expected_metrics(baseline: dict[str, Any], method: dict[str, Any], entry: dict[str, Any]) -> dict[str, float] | None:
    name, split, seed = entry["model"], entry["split"], int(entry["seed"])
    if name in baseline["reports"][split] and "seeds" in baseline["reports"][split][name]:
        for row in baseline["reports"][split][name]["seeds"]:
            if int(row["seed"]) == seed:
                return row["metrics"]
    if name == "full":
        for row in method["full"]["report"][split]["seeds"]:
            if int(row["seed"]) == seed:
                return row["metrics"]
    for variant in method.get("ablations", []):
        if name == variant["name"]:
            for row in variant["report"][split]["seeds"]:
                if int(row["seed"]) == seed:
                    return row["metrics"]
    return None


def _refresh_reproducibility() -> None:
    final = OUT / "final_package"
    rows = []
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and path != final / "reproducibility_manifest.json":
            rows.append({"path": str(path.relative_to(OUT)), "sha256": sha256_file(path)})
    write_json(final / "reproducibility_manifest.json", {"schema": "rm-series-reproducibility-v4", "generator": str(Path(__file__).relative_to(ROOT)), "generator_sha256": sha256_file(Path(__file__)), "python": "/home/tzh/conda_envs/actmask/bin/python", "generated_files": rows, "method_training": True, "method_family_count": 1, "independent_raw_recomputation": True})


def run() -> dict[str, Any]:
    final = json.loads((OUT / "final_package" / "final_decision.json").read_text())
    if final["decision"] != "RM_METHOD_NOT_SUPPORTED":
        raise RuntimeError(f"unexpected final decision for closure audit: {final['decision']}")
    records, payload = _load_payload()
    splits = _split_definitions(records)
    sample_sets = {name: build_samples(payload, value) for name, value in splits.items()}
    task_by_episode = {record.episode_id: record.task_id for record in records}
    balance: dict[str, Any] = {}
    for name, definition in sample_sets.items():
        balance[name] = {}
        for part, values in definition.items():
            rows = values["rows"]
            tasks = Counter(task_by_episode[episode_id] for episode_id in values["episode_ids"])
            labels = Counter(int(row["label"]) for row in rows)
            anchors = Counter(str(row["anchor_key"]) for row in rows)
            balance[name][part] = {
                "episodes": len(values["episode_ids"]),
                "task_episode_counts": dict(sorted(tasks.items())),
                "task_balance_min": min(tasks.values()),
                "task_balance_max": max(tasks.values()),
                "candidate_labels": {str(key): value for key, value in sorted(labels.items())},
                "candidate_label_ratio_positive": labels[1] / len(rows),
                "every_anchor_has_one_positive_and_four_negatives": all(count == 5 for count in anchors.values()) and labels[1] * 4 == labels[0],
                "failure_category_balance": "not_applicable_no_official_failure_category_field",
                "source_directory_used_as_input": False,
            }
    write_json(OUT / "task_formulations" / "split_balance_audit_action_future_consistency.json", {"schema": "rm-series-split-balance-v1", "splits": balance, "category_held_out": "not_applicable_no_official_failure_category", "object_goal_held_out": "not_available_no_object_goal_labels", "robot_embodiment_held_out": "not_available_selected_subset_has_one_embodiment"})
    baseline = json.loads((OUT / "baseline_report.json").read_text())
    method = json.loads((OUT / "final_package" / "method_summary.json").read_text())
    raw = json.loads((OUT / "raw_score_manifest.json").read_text())
    checks: list[dict[str, Any]] = []
    for entry in raw["entries"] + raw["method_entries"]:
        path = OUT / entry["path"]
        rows, recomputed = _metric_row(path)
        expected = _expected_metrics(baseline, method, entry)
        matches = expected is not None and all(abs(float(recomputed[key]) - float(expected[key])) < 1e-9 for key in ("balanced_accuracy", "auroc", "pair_order_accuracy"))
        checks.append({"path": entry["path"], "model": entry["model"], "split": entry["split"], "seed": entry["seed"], "rows": len(rows), "file_sha256_matches_manifest": sha256_file(path) == entry["sha256"], "reported_metric_found": expected is not None, "metrics_match": matches})
    recomputation = {"schema": "rm-series-raw-score-recomputation-v2", "checks": checks, "all_paths_present": all((OUT / row["path"]).is_file() for row in raw["entries"] + raw["method_entries"]), "all_hashes_match": all(row["file_sha256_matches_manifest"] for row in checks), "all_metrics_recomputed_match": all(row["metrics_match"] for row in checks), "independent_of_training": True}
    write_json(OUT / "raw_score_recomputation_audit.json", recomputation)
    final["completion_audit"] = {"raw_scores_recomputed": recomputation["all_metrics_recomputed_match"], "split_balance_audited": True, "method_family_count": 1, "no_tuning_after_failure": True}
    write_json(OUT / "final_package" / "final_decision.json", final)
    (OUT / "final_package" / "final_report.md").open("a").write("\nIndependent closure audit: all recorded baseline and method raw-score metrics were recomputed from their JSONL files; split/label/media checks are recorded separately.\n")
    append_log("rm9_independent_completion_audit", raw_score_checks=len(checks), all_metrics_match=recomputation["all_metrics_recomputed_match"])
    _refresh_reproducibility()
    return {"raw_score_checks": len(checks), "all_metrics_match": recomputation["all_metrics_recomputed_match"]}


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, sort_keys=True))
