"""Finalize RM-OD only after the frozen benchmark gates have been audited."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from actmask.experiments.rm_od_freeze_and_inventory import OUT, append_log, write_json
from actmask.experiments.rm_od_metrics import group_metrics


DECISION = "RM_OD_ACTION_PATH_SHORTCUT"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(name: str) -> dict[str, Any]:
    return json.loads((OUT / name).read_text())


def _close(actual: float | int, expected: float | int) -> bool:
    return abs(float(actual) - float(expected)) <= 1e-12


def _recompute(entries: list[dict[str, Any]]) -> dict[str, Any]:
    checked: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for entry in entries:
        path = OUT / entry["path"]
        item: dict[str, Any] = {"path": entry["path"], "present": path.is_file()}
        if not path.is_file():
            failures.append(item | {"reason": "missing"})
            checked.append(item)
            continue
        item["sha256_matches"] = _sha(path) == entry["sha256"]
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        actual = group_metrics(rows)
        item["metric_matches"] = all(_close(actual[key], entry["metrics"][key]) for key in actual)
        item["groups"] = len(rows)
        if not item["sha256_matches"] or not item["metric_matches"]:
            failures.append(item)
        checked.append(item)
    return {
        "schema": "rm-od-raw-score-recompute-v1",
        "entries_checked": len(checked),
        "pass": not failures,
        "failures": failures,
        "entries": checked,
    }


def _docs(baseline: dict[str, Any], feasibility: dict[str, Any], recompute: dict[str, Any]) -> None:
    docs = OUT / "docs"
    docs.mkdir(exist_ok=True)
    summary = baseline["summary"]
    episode = baseline["reports"]["episode_held_out"]
    task = baseline["reports"]["task_held_out"]
    plan = """# RM-OD plan (frozen and completed)

RM-OD evaluated K=8 action-conditioned future visual retrieval on 100 locally
available RoboMIND 2.0 Franka trajectories from 10 official tasks. The frozen
configuration, SHA-256 and source file hashes are in the output root.

The task was audited before any relational method was allowed. Inputs for fair
models were pre-action front-RGB/state histories, an action chunk, and each
candidate action/future RGB item. Paths, episode IDs, positive indices,
segmentation/object IDs and future actions were excluded from fair inputs.

Outcome: the benchmark was rejected at the baseline gate. No
ObjectRelationalFutureRetriever was trained.
"""
    results = f"""# RM-OD results

Final decision: `{DECISION}`.

The source-path repair succeeded: its direct proxy Recall@1 was 0.125 (the
K=8 random value). It exposed a second structural shortcut in this source
granularity: episode-progress Recall@1 was {episode['episode_progress']['mean']['recall_at_1']:.3f}; action-only was {episode['action_gru']['mean']['recall_at_1']:.3f}; state+action GRU was {episode['state_action_gru']['mean']['recall_at_1']:.3f}; and current-RGB-only was {episode['current_rgb_only']['mean']['recall_at_1']:.3f} on the episode-held-out split.

The strongest fair visual baseline reached {summary['best_fair_visual_recall_at_1']:.3f}, above the frozen saturation ceiling of 0.750, while the privileged future diagnostic was {summary['privileged_recall_at_1']:.3f}. This is not usable relational-method headroom.

Task-held-out values retained the same pattern: action-only {task['action_gru']['mean']['recall_at_1']:.3f}, state+action {task['state_action_gru']['mean']['recall_at_1']:.3f}, current RGB {task['current_rgb_only']['mean']['recall_at_1']:.3f}, strongest fair visual {summary['task_held_out_recall_at_1']:.3f}.

Raw-score re-computation: {recompute['entries_checked']} artifacts checked; pass={recompute['pass']}.
"""
    handoff = f"""# RM-OD handoff

RM-OD is complete and rejected as `{DECISION}`. Do not train or report an
ObjectRelationalFutureRetriever on this frozen task.

Why a second repair was not made: each selected source trajectory has exactly
eight frozen anchors and K=8. The first repair had to make all candidates
share the positive source trajectory to make the source-path proxy random.
That exhausts the eight distinct anchors, making time/progress a deterministic
identifier (Recall@1=1.0). Returning to cross-trajectory candidates restores
the prior source-path proxy Recall@1=1.0. A second change would therefore
violate either the frozen source-path gate or the frozen matching/positive
definition, rather than repair an implementation defect.

{json.dumps(feasibility, ensure_ascii=False, indent=2)}

A future data collection/benchmark needs repeated near-identical starting
states and action chunks across independent trajectories, plus independently
varying object outcomes, before this claim can be tested.
"""
    (docs / "rm_od_plan.md").write_text(plan)
    (docs / "rm_od_results.md").write_text(results)
    (docs / "rm_od_handoff.md").write_text(handoff)


def run() -> dict[str, Any]:
    baseline = _load("baseline_report.json")
    repair = _load("repair1_source_path_precheck.json")
    candidate = _load("candidate_group_audit.json")
    groups = [json.loads(line) for line in (OUT / "task_groups.jsonl").read_text().splitlines() if line]
    config_hash = (OUT / "preregistered_config.sha256").read_text().strip()
    if baseline["summary"]["decision"] != DECISION:
        raise RuntimeError("finalizer is only valid for the observed RM-OD shortcut decision")
    if not repair["pass"] or not candidate["candidate_source_path_constant_within_group"]:
        raise RuntimeError("the documented source-path repair is missing")
    if len(groups) != 1600 or any(len(row["candidates"]) != 8 for row in groups):
        raise RuntimeError("unexpected frozen K=8 group structure")

    episode = baseline["reports"]["episode_held_out"]
    feasibility = {
        "schema": "rm-od-repair-feasibility-audit-v1",
        "additional_hard_negative_repair_performed": False,
        "reason": "no legal second repair exists under the frozen K=8, eight-anchor, same-trajectory-positive and source-path gates",
        "frozen_anchors_per_source_trajectory": 8,
        "retrieval_k": 8,
        "groups": len(groups),
        "source_path_repair1_recall_at_1": repair["max_test_recall_at_1"],
        "initial_cross_source_path_recall_at_1": 1.0,
        "same_source_progress_recall_at_1": episode["episode_progress"]["mean"]["recall_at_1"],
        "same_source_action_only_recall_at_1": episode["action_gru"]["mean"]["recall_at_1"],
        "same_source_current_rgb_only_recall_at_1": episode["current_rgb_only"]["mean"]["recall_at_1"],
        "logical_constraint": "K candidates from one source must occupy the K frozen distinct anchors; matching their progress/action/state by using other sources reintroduces the observed path identifier.",
        "conclusion": "a second rewrite would be an invalid task redefinition, not an allowed hard-negative repair",
    }
    write_json(OUT / "repair_feasibility_audit.json", feasibility)

    prior_manifest = _load("raw_score_manifest.json")
    precheck = _load("initial_shortcut_precheck.json")
    entries = precheck["entries"] + prior_manifest["entries"]
    recompute = _recompute(entries)
    write_json(OUT / "metric_recomputation_audit.json", recompute)
    if not recompute["pass"]:
        raise RuntimeError("raw-score recomputation failed")
    raw_manifest = {
        "schema": "rm-od-raw-score-manifest-v2",
        "entries": entries,
        "method_entries": [],
        "all_entries_present": all((OUT / row["path"]).is_file() for row in entries),
        "metric_recomputation": "metric_recomputation_audit.json",
    }
    write_json(OUT / "raw_score_manifest.json", raw_manifest)

    summary = baseline["summary"]
    baseline_summary = {
        "schema": "rm-od-baseline-summary-v1",
        "decision": DECISION,
        "random_recall_at_1": 0.125,
        "episode_held_out": {name: value["mean"] for name, value in baseline["reports"]["episode_held_out"].items()},
        "task_held_out": {name: value["mean"] for name, value in baseline["reports"]["task_held_out"].items()},
        "frozen_gates": summary["gates"],
        "strongest_fair_visual": {"name": summary["best_fair_visual_standard"], "recall_at_1": summary["best_fair_visual_recall_at_1"]},
    }
    write_json(OUT / "baseline_summary.json", baseline_summary)
    final = {
        "schema": "rm-od-final-decision-v1",
        "decision": DECISION,
        "method_ran": False,
        "method_reason": "The frozen benchmark did not pass its shortcut and saturation gates.",
        "preregistered_config_sha256": config_hash,
        "source": "RoboMIND 2.0 only; 100 selected local Franka trajectories / 10 tasks",
        "baseline_decision": baseline["summary"]["decision"],
        "failed_gates": [name for name, passed in summary["gates"].items() if not passed],
        "repair_history": {"initial_cross_source_path_shortcut": 1.0, "repair1_source_path_proxy": repair["max_test_recall_at_1"], "repair_feasibility": "repair_feasibility_audit.json"},
        "claim_status": "No method-performance claim. The result is an audit finding that this frozen source/task construction admits action/path/progress shortcuts.",
    }
    write_json(OUT / "final_decision.json", final)

    claim = """# Claim boundary

This artifact does **not** support a claim that ActMask predicts failure,
success, physical counterfactuals, or a general real-robot object-dynamics
method. It supports only the negative audit conclusion: this frozen RoboMIND
K=8 retrieval construction permits source/progress/action shortcuts and is
not a valid benchmark for such a method claim.
"""
    positioning = """# Paper positioning

RM-OD is not method evidence. It may be cited internally as a reproducible
benchmark-design audit: source-balanced same-trajectory negatives remove a
path proxy but make temporal progress, state/action, and current-frame cues
decisive. Do not present it as an ActMask improvement or a real-robot
object-dynamics result.
"""
    (OUT / "claim_boundary.md").write_text(claim)
    (OUT / "paper_positioning.md").write_text(positioning)
    _docs(baseline, feasibility, recompute)

    report = f"""# RM-OD final report

## Decision

`{DECISION}` — benchmark rejected before method training.

The initial cross-trajectory design let source provenance identify the answer
(Recall@1 1.000). Repair 1 balanced provenance within each group and lowered
that proxy to 0.125. The repaired K=8 construction then made progress a
perfect shortcut (Recall@1 1.000), with episode-held-out action-only 0.902,
state+action 0.900, and current-RGB-only 0.667. The strongest fair visual
standard was 0.887, above the 0.750 saturation ceiling. Therefore the frozen
headroom gate failed and ObjectRelationalFutureRetriever was not run.

All {recompute['entries_checked']} raw score artifacts were hash-checked and
metric-recomputed successfully. See `repair_feasibility_audit.json` for why a
second repair would violate the frozen task rather than repair it.
"""
    (OUT / "final_report.md").write_text(report)

    tracked = [
        "preregistered_config.json", "source_inventory.json", "selected_trajectory_manifest.json", "source_hash_manifest.json",
        "task_manifest.json", "hard_negative_matching_audit.json", "candidate_group_audit.json", "future_consequence_audit.json",
        "robot_motion_matching_audit.json", "leakage_audit.json", "split_audit.json", "initial_shortcut_precheck.json",
        "repair1_source_path_precheck.json", "repair_feasibility_audit.json", "baseline_report.json", "baseline_summary.json",
        "raw_score_manifest.json", "metric_recomputation_audit.json", "final_decision.json", "final_report.md",
        "claim_boundary.md", "paper_positioning.md", "docs/rm_od_plan.md", "docs/rm_od_results.md", "docs/rm_od_handoff.md",
    ]
    reproducibility = {
        "schema": "rm-od-reproducibility-manifest-v1",
        "python": "/home/tzh/conda_envs/actmask/bin/python",
        "config_sha256": config_hash,
        "source_hash_manifest_sha256": _sha(OUT / "source_hash_manifest.json"),
        "raw_score_entries": len(entries),
        "metric_recomputation_pass": recompute["pass"],
        "method_trained": False,
        "tracked_artifact_sha256": {name: _sha(OUT / name) for name in tracked},
        "commands": [
            "python -m actmask.experiments.rm_od_freeze_and_inventory",
            "python -m actmask.experiments.rm_od_build_task",
            "python -m actmask.experiments.rm_od_shortcut_precheck",
            "python -m actmask.experiments.rm_od_repair_source_proxy",
            "python -m actmask.experiments.rm_od_baselines",
            "python -m actmask.experiments.rm_od_finalize",
        ],
    }
    write_json(OUT / "reproducibility_manifest.json", reproducibility)
    append_log("phase7_final_decision", decision=DECISION, method_ran=False, raw_score_entries=len(entries), recomputation_pass=recompute["pass"])
    return final


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, sort_keys=True))
