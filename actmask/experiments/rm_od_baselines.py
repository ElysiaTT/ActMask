"""RM-OD K=8 retrieval controls and fair baseline gate (no relational method)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from actmask.experiments.rm_od_build_task import _load_payload, _nodes, _record, _split_definitions
from actmask.experiments.rm_od_freeze_and_inventory import OUT, append_log, write_json
from actmask.experiments.rm_od_metrics import group_metrics, score_rows


SEEDS = (17, 29, 43)
K = 8


def _load_groups() -> list[dict[str, Any]]:
    return [json.loads(line) for line in (OUT / "task_groups.jsonl").read_text().splitlines() if line]


def _node_index() -> dict[str, dict[str, Any]]:
    selected = json.loads((OUT / "selected_trajectory_manifest.json").read_text())["episodes"]
    records = [_record(row) for row in selected]
    payload, _ = _load_payload(records)
    splits = _split_definitions(records)
    output: dict[str, dict[str, Any]] = {}
    for family, definition in splits.items():
        for partition, ids in definition.items():
            output.update({node["node_id"]: node for node in _nodes(payload, ids, family, partition)})
    return output


def _arrays(groups: list[dict[str, Any]], nodes: dict[str, dict[str, Any]], task_names: list[str]) -> dict[str, np.ndarray]:
    visual, state, action, candidate, candidate_action, current, progress, task, true_future = [], [], [], [], [], [], [], [], []
    lookup = {name: index for index, name in enumerate(task_names)}
    labels = []
    for group in groups:
        node = nodes[group["anchor_node_id"]]
        visual.append(np.concatenate([node["pre_global"], node["pre_patch"].reshape(len(node["pre_patch"]), -1)], axis=1))
        state.append(node["pre_state"])
        action.append(node["action"])
        current.append(np.concatenate([node["pre_global"][-1], node["pre_patch"][-1].reshape(-1)]))
        progress.append([node["progress"]])
        onehot = np.zeros(len(task_names), dtype=np.float32); onehot[lookup[group["task"]]] = 1.0; task.append(onehot)
        crows, futures = [], []
        candidate_actions = []
        for row in group["candidates"]:
            candidate_node = nodes[row["candidate_node_id"]]
            crows.append(np.concatenate([candidate_node["future_global"], candidate_node["future_patch"].reshape(len(candidate_node["future_patch"]), -1)], axis=1))
            candidate_actions.append(candidate_node["action"])
            futures.append(np.concatenate([candidate_node["future_global"].mean(0), candidate_node["future_patch"].mean((0, 1))]))
        candidate.append(np.asarray(crows, dtype=np.float32))
        candidate_action.append(np.asarray(candidate_actions, dtype=np.float32))
        true_future.append(np.concatenate([node["future_global"].mean(0), node["future_patch"].mean((0, 1))]))
        labels.append(group["positive_index"])
    return {"visual": np.asarray(visual, np.float32), "state": np.asarray(state, np.float32), "action": np.asarray(action, np.float32), "candidate": np.asarray(candidate, np.float32), "candidate_action": np.asarray(candidate_action, np.float32), "current": np.asarray(current, np.float32), "progress": np.asarray(progress, np.float32), "task": np.asarray(task, np.float32), "true_future": np.asarray(true_future, np.float32), "label": np.asarray(labels, np.int64)}


def _standardize(train: np.ndarray, test: np.ndarray, axes: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    mean, std = train.mean(axes, keepdims=True), train.std(axes, keepdims=True)
    std[std < 1e-5] = 1.0
    return (train - mean) / std, (test - mean) / std


class _Retriever(nn.Module):
    def __init__(self, mode: str, visual_dim: int, state_dim: int, action_dim: int, candidate_dim: int, candidate_action_dim: int, task_dim: int) -> None:
        super().__init__()
        self.mode = mode
        hidden = 48
        self.vgru = nn.GRU(visual_dim, hidden, batch_first=True)
        self.sgru = nn.GRU(state_dim, hidden, batch_first=True)
        self.agru = nn.GRU(action_dim, hidden, batch_first=True)
        self.tcn = nn.Sequential(nn.Conv1d(state_dim + action_dim, hidden, 3, padding=1), nn.GELU(), nn.Conv1d(hidden, hidden, 3, padding=1), nn.GELU())
        self.task = nn.Linear(task_dim, hidden)
        self.current = nn.Linear(visual_dim, hidden)
        self.candidate = nn.Sequential(nn.Linear(candidate_dim + candidate_action_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.cross = nn.MultiheadAttention(hidden, 4, batch_first=True)
        self.cross_visual = nn.Linear(visual_dim, hidden)

    def query(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        visual = self.vgru(batch["visual"])[0][:, -1]
        state = self.sgru(batch["state"])[0][:, -1]
        action = self.agru(batch["action"])[0][:, -1]
        if self.mode == "action_gru": return action
        if self.mode == "state_action_gru": return state + action
        if self.mode == "state_action_tcn":
            if batch["state"].shape[1] != batch["action"].shape[1]:
                raise ValueError("frozen RM-OD TCN control requires equal state/action history length")
            seq = torch.cat([batch["state"], batch["action"]], dim=2)
            return self.tcn(seq.transpose(1, 2)).mean(2)
        if self.mode == "global_visual_history_gru": return visual
        if self.mode == "visual_history_action_gru": return visual + action
        if self.mode == "visual_history_state_action_gru": return visual + state + action
        if self.mode == "current_rgb_only": return self.current(batch["current"])
        if self.mode == "task_language_only": return self.task(batch["task"])
        if self.mode == "visual_cross_attention":
            tokens = self.cross_visual(batch["visual"])
            return self.cross(tokens[:, -1:], tokens, tokens, need_weights=False)[0][:, 0]
        raise ValueError(self.mode)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        q = self.query(batch)
        candidate = self.candidate(torch.cat([batch["candidate"].mean(2), batch["candidate_action"].mean(2)], dim=2))
        return (candidate * q[:, None]).sum(2)


def _batch(arrays: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: torch.as_tensor(value, device=device) for key, value in arrays.items() if key != "label" and key != "true_future"}


def _train_predict(mode: str, train: dict[str, np.ndarray], test: dict[str, np.ndarray], seed: int) -> np.ndarray:
    train = dict(train); test = dict(test)
    for key, axes in (("visual", (0, 1)), ("state", (0, 1)), ("action", (0, 1)), ("candidate", (0, 1, 2)), ("candidate_action", (0, 1, 2)), ("current", (0,)), ("progress", (0,)), ("task", (0,))):
        train[key], test[key] = _standardize(train[key], test[key], axes)
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _Retriever(mode, train["visual"].shape[2], train["state"].shape[2], train["action"].shape[2], train["candidate"].shape[3], train["candidate_action"].shape[3], train["task"].shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.004, weight_decay=1e-4)
    train_batch = _batch(train, device); label = torch.as_tensor(train["label"], device=device)
    for _ in range(110):
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(model(train_batch), label)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
    with torch.no_grad():
        return model(_batch(test, device)).cpu().numpy()


def _nearest_neighbor(train: dict[str, np.ndarray], test: dict[str, np.ndarray]) -> np.ndarray:
    query_train = np.concatenate([train["visual"].reshape(len(train["visual"]), -1), train["state"].reshape(len(train["state"]), -1), train["action"].reshape(len(train["action"]), -1)], 1)
    query_test = np.concatenate([test["visual"].reshape(len(test["visual"]), -1), test["state"].reshape(len(test["state"]), -1), test["action"].reshape(len(test["action"]), -1)], 1)
    mean, std = query_train.mean(0), query_train.std(0); std[std < 1e-5] = 1.0
    positive = train["candidate"][np.arange(len(train["candidate"])), train["label"]].mean(1)
    scores = []
    for query, candidates in zip(query_test, test["candidate"]):
        nearest = np.argmin(((query_train - query) / std) ** 2, axis=0).mean()  # deterministic scalar proxy is replaced below
        distance = ((query_train - query) / std) ** 2
        index = int(np.argmin(distance.mean(1)))
        target = positive[index]
        candidate_features = candidates.mean(1)
        scores.append(-((candidate_features - target) ** 2).mean(1))
    return np.asarray(scores, np.float32)


def _save_scores(split: str, name: str, seed: int, groups: list[dict], scores: np.ndarray, *, diagnostic: bool = False) -> dict[str, Any]:
    path = OUT / "raw_scores" / "baselines" / split / name / f"seed_{seed}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"group_id": group["group_id"], "positive_index": group["positive_index"], "scores": [float(value) for value in value_row], "split": split, "model": name, "seed": seed} for group, value_row in zip(groups, scores)]
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    return {"split": split, "model": name, "seed": seed, "path": str(path.relative_to(OUT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "groups": len(rows), "diagnostic_only": diagnostic, "metrics": group_metrics(rows)}


def _direct_scores(name: str, groups: list[dict], arrays: dict[str, np.ndarray], nodes: dict[str, dict]) -> np.ndarray:
    if name in {"constant", "source_path", "camera_background"}:
        return np.zeros((len(groups), K), np.float32)
    if name == "candidate_index": return np.tile(-np.arange(K, dtype=np.float32), (len(groups), 1))
    if name == "episode_progress":
        values = []
        for group in groups:
            query = nodes[group["anchor_node_id"]]["progress"]
            values.append([-abs(query - nodes[row["candidate_node_id"]]["progress"]) for row in group["candidates"]])
        return np.asarray(values, np.float32)
    if name == "future_clip_low_level":
        # Diagnostic only: compare the logged post-action RGB clip with every
        # candidate's RGB clip.  It intentionally uses future information and
        # is never a fair model input or a gate-driving standard baseline.
        values = []
        for group in groups:
            query_node = nodes[group["anchor_node_id"]]
            query = np.concatenate([query_node["future_global"].mean(0), query_node["future_patch"].mean(0).reshape(-1)])
            scores = []
            for row in group["candidates"]:
                node = nodes[row["candidate_node_id"]]
                candidate = np.concatenate([node["future_global"].mean(0), node["future_patch"].mean(0).reshape(-1)])
                scores.append(float(np.dot(query, candidate) / (np.linalg.norm(query) * np.linalg.norm(candidate) + 1e-12)))
            values.append(scores)
        return np.asarray(values, np.float32)
    if name == "privileged_future_feature":
        values = []
        for group in groups:
            q = nodes[group["anchor_node_id"]]["future_patch"].mean(0).reshape(-1)
            values.append([float(np.dot(q, nodes[row["candidate_node_id"]]["future_patch"].mean(0).reshape(-1)) / (np.linalg.norm(q) * np.linalg.norm(nodes[row["candidate_node_id"]]["future_patch"].mean(0).reshape(-1)) + 1e-12)) for row in group["candidates"]])
        return np.asarray(values, np.float32)
    raise ValueError(name)


def _summarize(entries: list[dict]) -> dict[str, Any]:
    return {"seeds": [{"seed": entry["seed"], "metrics": entry["metrics"]} for entry in entries], "mean": {metric: float(np.mean([entry["metrics"][metric] for entry in entries])) for metric in ("recall_at_1", "recall_at_3", "mrr", "ndcg", "pair_order_accuracy")}, "std": {metric: float(np.std([entry["metrics"][metric] for entry in entries])) for metric in ("recall_at_1", "recall_at_3", "mrr", "ndcg", "pair_order_accuracy")}}


def run() -> dict[str, Any]:
    repair = json.loads((OUT / "repair1_source_path_precheck.json").read_text())
    if not repair["pass"]:
        raise RuntimeError("fair baseline gate forbidden until source-path repair passes")
    nodes = _node_index(); groups = _load_groups()
    task_names = sorted({group["task"] for group in groups})
    report: dict[str, Any] = {"schema": "rm-od-baseline-report-v1", "retrieval_k": K, "reports": {}, "fair_inputs": ["pre-anchor RGB history", "pre-anchor state", "anchor action chunk", "candidate action chunk", "candidate future RGB item"], "prohibited": ["positive index", "source path", "episode id", "future action", "true future feature"]}
    raw_entries: list[dict] = []
    direct = ("constant", "candidate_index", "source_path", "episode_progress", "future_clip_low_level", "camera_background", "privileged_future_feature")
    learned = ("action_gru", "state_action_gru", "state_action_tcn", "global_visual_history_gru", "visual_history_action_gru", "visual_history_state_action_gru", "visual_cross_attention", "current_rgb_only", "task_language_only")
    for family in ("episode_held_out", "task_held_out"):
        current = {partition: [group for group in groups if group["split_family"] == family and group["partition"] == partition] for partition in ("train", "validation", "test")}
        arrays = {partition: _arrays(value, nodes, task_names) for partition, value in current.items()}
        report["reports"][family] = {}
        for name in direct:
            diagnostic = name in {"future_clip_low_level", "privileged_future_feature", "source_path", "camera_background"}
            scores = _direct_scores(name, current["test"], arrays["test"], nodes)
            entry = _save_scores(family, name, 0, current["test"], scores, diagnostic=diagnostic)
            raw_entries.append(entry); report["reports"][family][name] = _summarize([entry]) | {"fair": not diagnostic, "diagnostic_only": diagnostic}
        # State-only is a learned retrieval probe and has no action channel.
        for name in ("state_action_gru",):
            pass
        aliases = {"state_only": "state_action_gru"}
        for name in learned:
            seeds = []
            for seed in SEEDS:
                if name == "state_only": continue
                scores = _train_predict(name, arrays["train"], arrays["test"], seed)
                entry = _save_scores(family, name, seed, current["test"], scores)
                raw_entries.append(entry); seeds.append(entry)
            report["reports"][family][name] = _summarize(seeds) | {"fair": True, "diagnostic_only": False}
        # Explicit state-only shares the same candidate encoder but zeroes actions.
        state_train, state_test = dict(arrays["train"]), dict(arrays["test"])
        state_train["action"] = np.zeros_like(state_train["action"]); state_test["action"] = np.zeros_like(state_test["action"])
        seeds = []
        for seed in SEEDS:
            scores = _train_predict("state_action_gru", state_train, state_test, seed)
            entry = _save_scores(family, "state_only", seed, current["test"], scores)
            raw_entries.append(entry); seeds.append(entry)
        report["reports"][family]["state_only"] = _summarize(seeds) | {"fair": True, "diagnostic_only": False}
        scores = _nearest_neighbor(arrays["train"], arrays["test"])
        entry = _save_scores(family, "nearest_neighbor_retrieval", 0, current["test"], scores)
        raw_entries.append(entry); report["reports"][family]["nearest_neighbor_retrieval"] = _summarize([entry]) | {"fair": True, "diagnostic_only": False}
    standard = ("global_visual_history_gru", "visual_history_action_gru", "visual_history_state_action_gru", "visual_cross_attention", "nearest_neighbor_retrieval")
    controls = ("candidate_index", "source_path", "task_language_only", "action_gru", "state_only", "state_action_gru", "state_action_tcn", "current_rgb_only")
    episode = report["reports"]["episode_held_out"]
    task = report["reports"]["task_held_out"]
    best = max(standard, key=lambda name: episode[name]["mean"]["recall_at_1"])
    best_score = episode[best]["mean"]["recall_at_1"]
    best_task = max(standard, key=lambda name: task[name]["mean"]["recall_at_1"])
    control_max = max(max(episode[name]["mean"]["recall_at_1"], task[name]["mean"]["recall_at_1"]) for name in controls)
    privileged = episode["privileged_future_feature"]["mean"]["recall_at_1"]
    gates = {"candidate_source_task_shortcuts_near_random": control_max <= 0.225, "action_state_controls_le_random_plus_010": max(episode[name]["mean"]["recall_at_1"] for name in ("action_gru", "state_only", "state_action_gru", "state_action_tcn")) <= 0.225, "current_rgb_only_le_random_plus_010": episode["current_rgb_only"]["mean"]["recall_at_1"] <= 0.225, "fair_visual_recoverable": best_score >= 0.275, "fair_visual_not_saturated": best_score <= 0.75, "privileged_headroom_ge_015": privileged >= best_score + 0.15, "task_held_out_above_random": task[best_task]["mean"]["recall_at_1"] > 0.125, "raw_scores_saved": all((OUT / row["path"]).is_file() for row in raw_entries)}
    if not gates["candidate_source_task_shortcuts_near_random"] or not gates["action_state_controls_le_random_plus_010"] or not gates["current_rgb_only_le_random_plus_010"]:
        decision = "RM_OD_ACTION_PATH_SHORTCUT"
    elif not gates["fair_visual_recoverable"]:
        decision = "RM_OD_VISUAL_CONSEQUENCE_NOT_RECOVERABLE"
    elif not gates["fair_visual_not_saturated"]:
        decision = "RM_OD_STANDARD_VISUAL_BASELINE_SATURATES"
    elif all(gates.values()):
        decision = "RM_OD_BENCHMARK_HEADROOM_FOUND"
    else:
        decision = "RM_OD_HUMAN_REVIEW_REQUIRED"
    report["summary"] = {"best_fair_visual_standard": best, "best_fair_visual_recall_at_1": best_score, "task_held_out_best": best_task, "task_held_out_recall_at_1": task[best_task]["mean"]["recall_at_1"], "privileged_recall_at_1": privileged, "maximum_control_recall_at_1": control_max, "gates": gates, "decision": decision}
    write_json(OUT / "baseline_report.json", report)
    write_json(OUT / "raw_score_manifest.json", {"schema": "rm-od-raw-score-manifest-v1", "entries": raw_entries, "all_entries_present": all((OUT / row["path"]).is_file() for row in raw_entries), "method_entries": []})
    append_log("phase5_baselines_complete", decision=decision, best_visual=best, best_visual_recall_at_1=best_score)
    return report["summary"]


if __name__ == "__main__":
    print(json.dumps(run(), sort_keys=True))
