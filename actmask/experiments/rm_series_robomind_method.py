"""The single preregistered RM-Series method trial.

This is intentionally callable only after the baseline package reports
``RM_BENCHMARK_HEADROOM_FOUND_METHOD_NOT_RUN``.  It trains one visual
action-conditioned relation verifier under the fixed three-seed protocol; it
does not perform architecture search or change the benchmark formulation.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from actmask.data.robomind_adapter import RoboMINDRecord
from actmask.experiments.rm_series_robomind_audit import (
    ACTION_CHUNK,
    FAIR_FIELDS,
    HISTORY,
    OUT,
    ROOT,
    SEEDS,
    _metrics,
    _split_definitions,
    append_log,
    build_samples,
    sha256_file,
    write_json,
    write_jsonl,
)


METHOD = "VisualRelDynVerifier"
METHOD_ROOT = OUT / "method" / METHOD


class VisualRelDynVerifier(nn.Module):
    """Frozen-image-stat token relation verifier with action cross attention."""

    def __init__(self, visual_dim: int = 22, state_dim: int = 32, action_dim: int = 32, hidden: int = 48) -> None:
        super().__init__()
        self.visual_encoder = nn.Sequential(nn.Linear(visual_dim, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.state_encoder = nn.Sequential(nn.Linear(state_dim, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.action_encoder = nn.Sequential(nn.Linear(action_dim, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.visual_memory = nn.GRU(hidden, hidden, batch_first=True)
        self.action_memory = nn.GRU(hidden, hidden, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(hidden, num_heads=4, batch_first=True)
        self.relation = nn.Sequential(nn.Linear(hidden * 4, hidden), nn.GELU(), nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU())
        self.head = nn.Linear(hidden, 1)

    def forward(self, visual: torch.Tensor, state: torch.Tensor, action: torch.Tensor, *, ablation: str = "full") -> torch.Tensor:
        if ablation == "no_visual_relation":
            visual = torch.zeros_like(visual)
        if ablation == "no_action":
            action = torch.zeros_like(action)
        if ablation == "no_history":
            visual = visual[:, -1:].expand_as(visual)
            state = state[:, -1:].expand_as(state)
        history_tokens = self.visual_encoder(visual) + self.state_encoder(state)
        action_tokens = self.action_encoder(action)
        visual_memory, _ = self.visual_memory(history_tokens)
        action_memory, _ = self.action_memory(action_tokens)
        query = action_memory[:, -1:]
        cross, _ = self.cross_attention(query, visual_memory, visual_memory, need_weights=False)
        visual_summary = visual_memory[:, -1]
        action_summary = action_memory[:, -1]
        cross_summary = cross[:, 0]
        relation = torch.cat([visual_summary, action_summary, cross_summary, visual_summary * action_summary], dim=1)
        return self.head(self.relation(relation))


def _load_payload() -> tuple[list[RoboMINDRecord], dict[str, dict[str, Any]]]:
    root = OUT / "processed_subset"
    manifest = json.loads((root / "data_manifest.json").read_text())
    records: list[RoboMINDRecord] = []
    payload: dict[str, dict[str, Any]] = {}
    for row in manifest["episodes"]:
        record = RoboMINDRecord(**{key: row[key] for key in RoboMINDRecord.__dataclass_fields__})
        numeric = np.load(root / row["robot_state_path"])
        visual = np.load(root / row["visual_feature_path"])
        records.append(record)
        payload[record.episode_id] = {
            "record": record,
            "state": numeric["state"].astype(np.float32),
            "action": numeric["action"].astype(np.float32),
            "timestamp": numeric["timestamp"],
            "frame_index": numeric["frame_index"],
            "anchors": visual["anchor"].astype(int).tolist(),
            "visual": visual["pre_anchor_feature"].astype(np.float32),
        }
    return records, payload


def _tensors(rows: list[dict[str, Any]], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    visual = np.stack([row["history_visual"] for row in rows]).astype(np.float32)
    state = np.stack([row["history_state"] for row in rows]).astype(np.float32)
    action = np.stack([row["candidate_action"] for row in rows]).astype(np.float32)
    label = np.asarray([row["label"] for row in rows], dtype=np.float32)[:, None]
    return tuple(torch.as_tensor(value, device=device) for value in (visual, state, action, label))  # type: ignore[return-value]


def _standardize(train: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], test: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    normalized: list[tuple[torch.Tensor, torch.Tensor]] = []
    for index in range(3):
        mean = train[index].mean((0, 1), keepdim=True)
        std = train[index].std((0, 1), keepdim=True).clamp_min(1e-5)
        normalized.append(((train[index] - mean) / std, (test[index] - mean) / std))
    train_out = (normalized[0][0], normalized[1][0], normalized[2][0], train[3])
    test_out = (normalized[0][1], normalized[1][1], normalized[2][1], test[3])
    return train_out, test_out


def _train_predict(train_rows: list[dict[str, Any]], test_rows: list[dict[str, Any]], seed: int, output: Path, *, ablation: str = "full") -> tuple[np.ndarray, dict[str, Any]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train, test = _standardize(_tensors(train_rows, device), _tensors(test_rows, device))
    model = VisualRelDynVerifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
    positive_weight = torch.tensor([(len(train_rows) - sum(row["label"] for row in train_rows)) / max(sum(row["label"] for row in train_rows), 1)], device=device)
    log: list[dict[str, float | int]] = []
    model.train()
    for epoch in range(120):
        optimizer.zero_grad(set_to_none=True)
        logits = model(train[0], train[1], train[2], ablation=ablation)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, train[3], pos_weight=positive_weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if epoch in {0, 39, 79, 119}:
            log.append({"epoch": epoch + 1, "loss": float(loss.detach().cpu())})
    output.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "seed": seed, "ablation": ablation}, output / "checkpoint.pt")
    write_jsonl(output / "train_log.jsonl", log)  # type: ignore[arg-type]
    with torch.no_grad():
        model.eval()
        scores = torch.sigmoid(model(test[0], test[1], test[2], ablation=ablation)).squeeze(1).cpu().numpy()
    return scores, {"device": str(device), "epochs": 120, "final_loss": log[-1]["loss"], "ablation": ablation}


def _run_variant(sample_sets: dict[str, dict[str, dict[str, Any]]], name: str, *, ablation: str = "full") -> dict[str, Any]:
    report: dict[str, Any] = {}
    raw_entries: list[dict[str, Any]] = []
    for split_name, splits in sample_sets.items():
        train_rows, test_rows = splits["train"]["rows"], splits["test"]["rows"]
        seed_rows: list[dict[str, Any]] = []
        for seed in SEEDS:
            destination = METHOD_ROOT / name / split_name / f"seed_{seed}"
            scores, train_info = _train_predict(train_rows, test_rows, seed, destination, ablation=ablation)
            metrics = _metrics(test_rows, scores)
            raw = [{"sample_id": row["sample_id"], "anchor_key": row["anchor_key"], "label": row["label"], "score": float(score), "split": split_name, "model": name, "seed": seed} for row, score in zip(test_rows, scores)]
            write_jsonl(destination / "raw_scores.jsonl", raw)
            write_json(destination / "metrics.json", metrics)
            write_json(destination / "model_config.json", {"family": METHOD, "variant": name, "ablation": ablation, "visual_encoder": "frozen deterministic 22-D RGB statistic token", "temporal_visual_memory": "GRU", "state_encoder": 32, "candidate_action_encoder": 32, "action_conditioned_cross_attention": True, "relation_module": "multiplicative visual-state/action relation", "goal_language_used": False})
            write_json(destination / "preprocessing_config.json", {"fair_inputs": list(FAIR_FIELDS), "forbidden": ["episode id", "task id", "folder/source path", "future observations", "labels"], "history_steps": HISTORY, "candidate_action_steps": ACTION_CHUNK, "normalization": "train split mean/std"})
            write_json(destination / "training_config.json", {**train_info, "seed": seed, "optimizer": "AdamW", "learning_rate": 0.003, "weight_decay": 0.0001, "positive_class_weight": 4.0})
            (destination / "seed.txt").write_text(f"{seed}\n")
            (destination / "command.txt").write_text("/home/tzh/conda_envs/actmask/bin/python -m actmask.experiments.rm_series_robomind_method\n")
            (destination / "data_sha256.txt").write_text(sha256_file(OUT / "processed_subset" / "data_manifest.json") + "\n")
            seed_rows.append({"seed": seed, "metrics": metrics, "artifact_dir": str(destination.relative_to(OUT))})
            raw_entries.append({"model": name, "ablation": ablation, "split": split_name, "seed": seed, "path": str((destination / "raw_scores.jsonl").relative_to(OUT)), "sha256": sha256_file(destination / "raw_scores.jsonl")})
        report[split_name] = {
            "seeds": seed_rows,
            "mean": {metric: float(np.mean([item["metrics"][metric] for item in seed_rows])) for metric in ("balanced_accuracy", "auroc", "pair_order_accuracy")},
            "std": {metric: float(np.std([item["metrics"][metric] for item in seed_rows])) for metric in ("balanced_accuracy", "auroc", "pair_order_accuracy")},
        }
    return {"name": name, "ablation": ablation, "report": report, "raw_entries": raw_entries}


def _artifact_complete(entries: list[dict[str, Any]]) -> bool:
    required = ("checkpoint.pt", "model_config.json", "preprocessing_config.json", "training_config.json", "seed.txt", "data_sha256.txt", "command.txt", "train_log.jsonl", "raw_scores.jsonl", "metrics.json")
    for row in entries:
        run = OUT / Path(row["path"]).parent
        if not all((run / name).is_file() for name in required):
            return False
    return True


def _write_final(method_summary: dict[str, Any], decision: str) -> None:
    final = OUT / "final_package"
    baseline = json.loads((OUT / "baseline_report.json").read_text())["summary"]
    final_decision = {
        "schema": "rm-series-final-decision-v1",
        "decision": decision,
        "selected_source": "RoboMIND2.0",
        "method_authorized_before_run": True,
        "method_authorized": decision == "RM_REAL_METHOD_SUPPORTED",
        "method_ran": True,
        "method_family": METHOD,
        "outcome_or_failure_claim_authorized": False,
        "physical_counterfactual_claim_authorized": False,
        "claim_boundary": "Logged real-robot action-conditioned future consistency only. No outcome/failure or physical-counterfactual claim is supported.",
        "baseline_summary": baseline,
        "method_summary": method_summary,
    }
    write_json(final / "final_decision.json", final_decision)
    write_json(final / "method_summary.json", method_summary)
    write_json(final / "baseline_summary.json", baseline)
    write_json(final / "raw_score_manifest.json", json.loads((OUT / "raw_score_manifest.json").read_text()))
    (final / "final_report.md").write_text(
        f"# RM-Series final report\n\nDecision: `{decision}`.\n\n"
        "RoboMIND2.0 local source: 100 real Franka trajectories across 10 repeated tasks. The source has no official outcome/failure semantics, so the target is only logged action-conditioned future consistency. "
        f"The one preregistered `{METHOD}` family was trained for seeds 17, 29 and 43 on fixed episode-held-out and task-held-out splits. "
        f"Episode-held-out method balanced accuracy: {method_summary['full']['report']['episode_held_out']['mean']['balanced_accuracy']:.3f}; "
        f"best standard: {baseline['best_standard_balanced_accuracy']:.3f}. No physical or outcome claim is made.\n"
    )
    (final / "next_steps.md").write_text(
        "# Next steps\n\nThe final decision is controlling. Do not tune this method family further. If outcome/failure verification is needed, obtain official RoboMIND labels, failure categories and onset annotations before creating a new benchmark.\n"
    )
    (OUT / "docs" / "rm_series_results.md").write_text(
        f"# RM-Series results\n\nBaseline headroom authorized one fixed `{METHOD}` trial. Final decision: `{decision}`. See `final_package/method_summary.json` and `final_package/final_report.md`.\n"
    )
    (OUT / "docs" / "rm_series_handoff.md").write_text(
        f"# RM-Series handoff\n\nThe only method trial has completed. Controlling decision: `{decision}`. No further tuning is authorized by this series.\n"
    )
    generated = []
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and path != final / "reproducibility_manifest.json":
            generated.append({"path": str(path.relative_to(OUT)), "sha256": sha256_file(path)})
    write_json(final / "reproducibility_manifest.json", {"schema": "rm-series-reproducibility-v2", "generator": str(Path(__file__).relative_to(ROOT)), "generator_sha256": sha256_file(Path(__file__)), "python": "/home/tzh/conda_envs/actmask/bin/python", "source_hash_manifest": "../source_hash_manifest.json", "generated_files": generated, "method_training": True, "method_family_count": 1})


def run() -> dict[str, Any]:
    predecision = json.loads((OUT / "final_package" / "final_decision.json").read_text())["decision"]
    if predecision != "RM_BENCHMARK_HEADROOM_FOUND_METHOD_NOT_RUN":
        raise RuntimeError(f"method trial forbidden unless baseline headroom is found; current decision={predecision}")
    records, payload = _load_payload()
    sample_sets = {name: build_samples(payload, split) for name, split in _split_definitions(records).items()}
    preregistration = {
        "schema": "rm-series-method-preregistration-v1",
        "family": METHOD,
        "seeds": list(SEEDS),
        "architecture": "frozen RGB-stat visual tokens + state tokens + action tokens + temporal GRUs + action-conditioned cross attention + relation head",
        "fair_inputs": list(FAIR_FIELDS),
        "goal_language_used": False,
        "one_repair_allowed_only_for_implementation_bug": True,
        "no_hyperparameter_search": True,
    }
    write_json(METHOD_ROOT / "preregistration.json", preregistration)
    append_log("rm7_method_trial_started", family=METHOD, seeds=list(SEEDS))
    full = _run_variant(sample_sets, "full")
    baseline = json.loads((OUT / "baseline_report.json").read_text())["summary"]
    full_episode = full["report"]["episode_held_out"]["mean"]["balanced_accuracy"]
    full_task = full["report"]["task_held_out"]["mean"]["balanced_accuracy"]
    headroom = baseline["oracle_balanced_accuracy"] - baseline["best_standard_balanced_accuracy"]
    initial_gates = {
        "episode_improvement_ge_005": full_episode >= baseline["best_standard_balanced_accuracy"] + 0.05,
        "hardest_valid_split_improves": full_task > baseline["task_held_out_best_balanced_accuracy"],
        "headroom_closed_ge_030": (full_episode - baseline["best_standard_balanced_accuracy"]) >= 0.30 * headroom,
        "stable_across_seeds": full["report"]["episode_held_out"]["std"]["balanced_accuracy"] <= 0.05,
        "artifacts_complete": _artifact_complete(full["raw_entries"]),
    }
    ablations: list[dict[str, Any]] = []
    if all(initial_gates.values()):
        for name, mode in (("no_visual_relation", "no_visual_relation"), ("no_action", "no_action"), ("no_history", "no_history")):
            ablations.append(_run_variant(sample_sets, name, ablation=mode))
        drops = {item["name"]: full_episode - item["report"]["episode_held_out"]["mean"]["balanced_accuracy"] for item in ablations}
        ablation_gate = sum(value >= 0.03 for value in drops.values()) >= 2
    else:
        drops = {}
        ablation_gate = False
    gates = {**initial_gates, "two_ablations_degrade_ge_003": ablation_gate if ablations else "not_run_method_initial_gate_failed"}
    method_summary = {"schema": "rm-series-method-summary-v1", "family": METHOD, "full": full, "ablations": ablations, "ablation_drops": drops, "baseline": baseline, "headroom": headroom, "gates": gates, "outcome_label_used": False, "physical_counterfactual_claim": False}
    combined_raw = json.loads((OUT / "raw_score_manifest.json").read_text())
    combined_raw["method_entries"] = full["raw_entries"] + [entry for item in ablations for entry in item["raw_entries"]]
    combined_raw["all_method_artifacts_complete"] = _artifact_complete(combined_raw["method_entries"])
    write_json(OUT / "raw_score_manifest.json", combined_raw)
    if all(initial_gates.values()) and ablation_gate:
        decision = "RM_REAL_METHOD_SUPPORTED"
    else:
        decision = "RM_METHOD_NOT_SUPPORTED"
    append_log("rm7_method_trial_complete", family=METHOD, decision=decision, episode_score=full_episode, task_score=full_task)
    _write_final(method_summary, decision)
    return {"decision": decision, "episode_score": full_episode, "task_score": full_task, "gates": gates}


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, sort_keys=True))
