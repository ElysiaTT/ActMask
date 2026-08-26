"""Versioned correction of the 3R-NL-v2 candidate-ranking unit.

The frozen pair-order and OOD report remains immutable.  This module retrains
only the already-selected GRU plus static/action controls, saves every ID score,
and ranks candidates within (family, mechanism, world, branch/history).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from actmask.experiments.milestone3r_metrics import pair_order
from actmask.experiments.milestone3r_nl_v2_probe import FACTORIES, _batch, _fit, _load, _split


ROOT = Path(__file__).resolve().parents[2]
ID_ROOT = ROOT / "outputs/actmask/milestone3r_nl_v2/full/id"
PARENT_REPORT = ROOT / "outputs/actmask/milestone3r_nl_v2/full_evaluation/full_evaluation.json"
DEFAULT_OUT = ROOT / "outputs/actmask/milestone3r_nl_v2_rankfix"

SOURCE_HASHES = {
    "outputs/actmask/milestone3r_nl_v2/full/id/model_inputs.npz": "30ccf4025c911d4c0bb9ab43e9dc7a3fd5de599d30a280886c214ac73a7da6f2",
    "outputs/actmask/milestone3r_nl_v2/full/id/labels.npz": "74997e73d1944013f8a71aa7953a687e27250b2d0235d4e29356f8a267a33d3d",
    "outputs/actmask/milestone3r_nl_v2/full/id/metadata.jsonl": "ff02dece7a19876fe3053450fffa6ce1937e92d855b34611ef82403b3d5bdbd3",
    "outputs/actmask/milestone3r_nl_v2/full_evaluation/full_evaluation.json": "abc371e50fda1e4969f37683201ef0aa9a01e6e5d00ac828521adbceaf7a96e4",
}

DIAGNOSTIC_SEEDS = (17, 29, 43)
CONFIRMATION_SEEDS = (17, 29, 43, 59, 71)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sources() -> list[dict[str, Any]]:
    inventory = []
    for relative, expected in SOURCE_HASHES.items():
        path = ROOT / relative
        observed = _sha256(path)
        if observed != expected:
            raise RuntimeError(f"rank-fix source hash mismatch: {relative}")
        inventory.append(
            {"path": relative, "bytes": path.stat().st_size, "sha256": observed}
        )
    return inventory


def rank_by_history(
    labels: np.ndarray,
    scores: np.ndarray,
    rows: list[dict[str, Any]],
    indices: np.ndarray,
    count: int,
) -> dict[str, float | int]:
    """Rank a candidate prefix separately for each observable history."""
    groups: dict[tuple[str, str, int, int], list[int]] = defaultdict(list)
    for index in indices:
        row = rows[int(index)]
        if row["candidate_id"] < count:
            groups[(row["family"], row["mechanism"], row["world_id"], row["branch"])].append(int(index))

    top1, top3, ndcg, regret = [], [], [], []
    feasible_top1, feasible_top3, feasible_ndcg = [], [], []
    mixed_contexts = 0
    malformed = 0
    for group_indices in groups.values():
        if len(group_indices) != count:
            malformed += 1
            continue
        group = np.asarray(group_indices)
        order = group[np.argsort(-scores[group], kind="stable")]
        maximum = scores[order[0]]
        ties = order[np.isclose(scores[order], maximum)]
        top_value = float(labels[ties].mean())
        top3_value = float(labels[order[:3]].any())
        relevance = labels[order]
        ideal = np.sort(labels[group])[::-1]
        weights = 1.0 / np.log2(np.arange(2, len(order) + 2))
        ideal_dcg = float((ideal * weights).sum())
        ndcg_value = float((relevance * weights).sum() / ideal_dcg) if ideal_dcg > 0 else 0.0
        regret_value = float(labels[group].max() - top_value)
        top1.append(top_value)
        top3.append(top3_value)
        ndcg.append(ndcg_value)
        regret.append(regret_value)
        if labels[group].any():
            feasible_top1.append(top_value)
            feasible_top3.append(top3_value)
            feasible_ndcg.append(ndcg_value)
        if 0 < labels[group].sum() < len(group):
            mixed_contexts += 1

    if malformed:
        raise RuntimeError(f"malformed candidate groups for C{count}: {malformed}")
    return {
        "contexts": len(top1),
        "feasible_contexts": len(feasible_top1),
        "feasible_context_fraction": float(len(feasible_top1) / len(top1)),
        "mixed_contexts": mixed_contexts,
        "mixed_context_fraction": float(mixed_contexts / len(top1)),
        "top1_success": float(np.mean(top1)),
        "top1_success_when_feasible": float(np.mean(feasible_top1)),
        "top3_success_recall": float(np.mean(top3)),
        "top3_success_recall_when_feasible": float(np.mean(feasible_top3)),
        "corrected_utility_gain_ndcg": float(np.mean(ndcg)),
        "ndcg_when_feasible": float(np.mean(feasible_ndcg)),
        "normalized_regret": float(np.mean(regret)),
    }


def _pair_metric(labels: np.ndarray, scores: np.ndarray, rows: list[dict[str, Any]], indices: np.ndarray) -> float:
    pair_ids = np.asarray([rows[int(index)]["pair_id"] for index in indices])
    return pair_order(labels[indices], scores[indices], pair_ids)


def _append_ledger(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def run(output: Path) -> dict[str, Any]:
    sources = _sources()
    output.mkdir(parents=True, exist_ok=True)
    ledger = output / "resource_ledger.jsonl"
    started = time.time()
    attempt_id = hashlib.sha256(f"{started}:{os.getpid()}".encode()).hexdigest()
    _append_ledger(
        ledger,
        {
            "event": "BEGIN",
            "attempt_id": attempt_id,
            "pid": os.getpid(),
            "resource": "GPU",
            "timestamp_unix": started,
            "protocol": "M3R-NL-V2-RANKFIX-V1",
        },
    )
    status = "FAILED"
    try:
        parent = json.loads(PARENT_REPORT.read_text(encoding="utf-8"))
        values, rows, labels = _load(ID_ROOT)
        device = torch.device("cuda:0")
        batch = _batch(values, device)
        train = _split(rows, "train")
        test = _split(rows, "test")
        frame_dim = values["history"].shape[2] + 3
        action_dim = values["candidate_actions"].shape[1] * 3 + 3
        horizon = values["history"].shape[1]

        raw_scores: dict[str, np.ndarray] = {}
        confirmation = []
        for seed in CONFIRMATION_SEEDS:
            model = _fit(
                lambda: FACTORIES["GRU"](frame_dim, action_dim, horizon),
                batch,
                train,
                labels,
                seed,
                120,
            )
            with torch.no_grad():
                score = model(batch).cpu().numpy().astype(np.float32)
            raw_scores[f"gru_seed_{seed}"] = score
            confirmation.append(score)
        raw_scores["gru_mean"] = np.mean(confirmation, axis=0).astype(np.float32)

        for method in ("StaticMLP", "ActionMLP"):
            prefix = {"StaticMLP": "static", "ActionMLP": "action"}[method]
            method_scores = []
            for seed in DIAGNOSTIC_SEEDS:
                model = _fit(
                    lambda method=method: FACTORIES[method](frame_dim, action_dim, horizon),
                    batch,
                    train,
                    labels,
                    seed,
                    100,
                )
                with torch.no_grad():
                    score = model(batch).cpu().numpy().astype(np.float32)
                key = f"{prefix}_seed_{seed}"
                raw_scores[key] = score
                method_scores.append(score)
            raw_scores[f"{prefix}_mean"] = np.mean(method_scores, axis=0).astype(np.float32)

        rankings = {}
        score_lookup = {
            "dynamic": raw_scores["gru_mean"],
            "StaticMLP": raw_scores["static_mean"],
            "ActionMLP": raw_scores["action_mean"],
        }
        for count in (5, 10, 20):
            rankings[f"C{count}"] = {
                method: rank_by_history(labels, score, rows, test, count)
                for method, score in score_lookup.items()
            }

        raw_path = output / "raw_scores.npz"
        np.savez_compressed(raw_path, **raw_scores)
        pair_score = _pair_metric(labels, raw_scores["gru_mean"], rows, test)
        pair_static = _pair_metric(labels, raw_scores["static_mean"], rows, test)
        pair_action = _pair_metric(labels, raw_scores["action_mean"], rows, test)
        c5_is_nondegenerate = all(
            rankings["C5"][method]["mixed_context_fraction"] == 1.0
            for method in rankings["C5"]
        )
        corrected_gate = all(
            rankings[name]["dynamic"]["top1_success"]
            - max(
                rankings[name]["StaticMLP"]["top1_success"],
                rankings[name]["ActionMLP"]["top1_success"],
            )
            >= 0.1
            for name in ("C5", "C10", "C20")
        )
        report = {
            "schema": "milestone3r-nl-v2-ranking-correction-v1",
            "sources": sources,
            "correction": {
                "invalid_original_group_key": ["family", "mechanism", "world_id"],
                "correct_group_key": ["family", "mechanism", "world_id", "branch"],
                "reason": "candidate actions must be ranked within one observable history; pooling counterfactual branches changes the decision unit",
                "metric_fix": "normalized regret is max achievable success minus expected tie-broken Top-1 success",
                "c5_status": "valid: every C5 history context contains both successful and unsuccessful candidates",
                "submission_ranking_scope": ["C5", "C10", "C20"],
            },
            "seeds": {"confirmation": list(CONFIRMATION_SEEDS), "controls": list(DIAGNOSTIC_SEEDS)},
            "training_steps": {"confirmation": 120, "controls": 100},
            "test_examples": int(len(test)),
            "test_pair_groups": int(len({rows[int(index)]["pair_id"] for index in test})),
            "test_history_contexts": int(
                len({(rows[int(index)]["family"], rows[int(index)]["mechanism"], rows[int(index)]["world_id"], rows[int(index)]["branch"]) for index in test})
            ),
            "pair_order_recheck": {
                "GRU": pair_score,
                "StaticMLP": pair_static,
                "ActionMLP": pair_action,
                "matches_parent_gru": pair_score == parent["interventions"]["correct"],
            },
            "rankings": rankings,
            "gates": {
                "c5_history_nondegenerate": c5_is_nondegenerate,
                "c5_c10_c20_dynamic_advantage": corrected_gate,
                "pair_order_unchanged": pair_score == parent["interventions"]["correct"],
            },
            "raw_scores": {
                "path": "raw_scores.npz",
                "sha256": _sha256(raw_path),
                "arrays": {key: list(value.shape) for key, value in raw_scores.items()},
            },
            "gpu_peak_allocated_mb": float(torch.cuda.max_memory_allocated() / 1024**2),
            "pass": c5_is_nondegenerate and corrected_gate and pair_score == parent["interventions"]["correct"],
        }
        report_path = output / "rankfix_report.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        status = "COMPLETED"
        return report
    finally:
        ended = time.time()
        _append_ledger(
            ledger,
            {
                "event": "END",
                "attempt_id": attempt_id,
                "resource": "GPU",
                "status": status,
                "timestamp_unix": ended,
                "wall_seconds": ended - started,
                "gpu_seconds": ended - started,
            },
        )


def _recomputed_rankings(output: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    _sources()
    report = json.loads((output / "rankfix_report.json").read_text(encoding="utf-8"))
    raw_path = output / report["raw_scores"]["path"]
    if _sha256(raw_path) != report["raw_scores"]["sha256"]:
        raise RuntimeError("rank-fix raw-score hash mismatch")
    _, rows, labels = _load(ID_ROOT)
    test = _split(rows, "test")
    with np.load(raw_path) as raw:
        score_lookup = {
            "dynamic": raw["gru_mean"],
            "StaticMLP": raw["static_mean"],
            "ActionMLP": raw["action_mean"],
        }
        recomputed = {
            f"C{count}": {
                method: rank_by_history(labels, score, rows, test, count)
                for method, score in score_lookup.items()
            }
            for count in (5, 10, 20)
        }
    return report, recomputed


def rebuild_report(output: Path) -> dict[str, Any]:
    report, recomputed = _recomputed_rankings(output)
    report["rankings"] = recomputed
    report["correction"]["c5_status"] = (
        "valid: every C5 history context contains both successful and unsuccessful candidates"
    )
    report["correction"]["submission_ranking_scope"] = ["C5", "C10", "C20"]
    c5_is_nondegenerate = all(
        recomputed["C5"][method]["mixed_context_fraction"] == 1.0
        for method in recomputed["C5"]
    )
    corrected_gate = all(
        recomputed[name]["dynamic"]["top1_success"]
        - max(
            recomputed[name]["StaticMLP"]["top1_success"],
            recomputed[name]["ActionMLP"]["top1_success"],
        )
        >= 0.1
        for name in ("C5", "C10", "C20")
    )
    report["gates"].pop("c10_c20_dynamic_advantage", None)
    report["gates"]["c5_history_nondegenerate"] = c5_is_nondegenerate
    report["gates"]["c5_c10_c20_dynamic_advantage"] = corrected_gate
    report["pass"] = bool(
        c5_is_nondegenerate
        and corrected_gate
        and report["gates"]["pair_order_unchanged"]
    )
    (output / "rankfix_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def verify(output: Path) -> dict[str, Any]:
    report, recomputed = _recomputed_rankings(output)
    if recomputed != report["rankings"]:
        raise RuntimeError("independent rank-fix recomputation mismatch")
    return {
        "pass": True,
        "raw_score_hash": report["raw_scores"]["sha256"],
        "rankings": recomputed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--rebuild-report", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if args.verify_only and args.rebuild_report:
        parser.error("choose only one of --verify-only and --rebuild-report")
    if args.verify_only:
        result = verify(output)
    elif args.rebuild_report:
        result = rebuild_report(output)
    else:
        result = run(output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
