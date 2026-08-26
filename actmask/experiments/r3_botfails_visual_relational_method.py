"""One gated R3 method family: VisualRelDynVerifier with fixed ablations."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import imageio.v3 as iio
import numpy as np
import torch
from torch import nn
from torchvision.models import resnet18

from actmask.data.r2_real_outcome_adapter import r2_append_log
from actmask.data.real_robot_dataset_adapter import sha256_file, write_json, write_jsonl
from actmask.experiments.r3_botfails_visual_baselines import SEEDS, _metrics, _threshold
from actmask.experiments.r3_botfails_visual_pipeline import BACKBONE, DEVICE, FEATURE_DIR, HISTORY, OFFSETS, OUT, R3TaskError, SOURCE_ROOT, TASK_DIR, VIEW, _preprocess, build_task


METHOD_DIR = OUT / "method_results" / "visual_rel_dyn_verifier"


def _patch_backbone() -> nn.Module:
    state = torch.load(BACKBONE, map_location="cpu", weights_only=True)
    model = resnet18(weights=None)
    model.load_state_dict(state)
    trunk = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2, model.layer3, model.layer4)
    trunk.eval().to(DEVICE)
    for parameter in trunk.parameters():
        parameter.requires_grad_(False)
    return trunk


def extract_patch_tokens() -> dict[str, Any]:
    """Mode C: frozen spatial tokens, invoked only after the visual gate passes."""

    path = METHOD_DIR / "patch_tokens.npz"
    if path.is_file():
        values = np.load(path)
        return {"path": path, "samples": int(len(values["tokens"])), "sha256": sha256_file(path), "status": "reused"}
    task = build_task()
    windows = task["windows"]
    by_episode: dict[str, list[tuple[int, Mapping[str, Any]]]] = defaultdict(list)
    for index, row in enumerate(windows):
        by_episode[str(row["episode_id"])].append((index, row))
    token = np.zeros((len(windows), len(OFFSETS), 49, 512), np.float16)
    trunk = _patch_backbone()
    for episode_id, entries in sorted(by_episode.items()):
        required = {frame for _, row in entries for frame in row["rgb_indices"]}
        frame_map: dict[int, np.ndarray] = {}
        video = SOURCE_ROOT / VIEW / f"{episode_id}.mp4"
        for index, frame in enumerate(iio.imiter(video, plugin="FFMPEG")):
            if index in required:
                frame_map[index] = np.asarray(frame, dtype=np.uint8)
            if len(frame_map) == len(required):
                break
        if set(frame_map) != required:
            raise R3TaskError(f"patch extraction missing frames for {episode_id}")
        indices = sorted(required)
        with torch.no_grad():
            batch = _preprocess([frame_map[index] for index in indices]).to(DEVICE)
            values = trunk(batch).detach().cpu().numpy().transpose(0, 2, 3, 1).reshape(len(indices), 49, 512).astype(np.float16)
        lookup = {index: values[position] for position, index in enumerate(indices)}
        for row_index, row in entries:
            token[row_index] = np.stack([lookup[index] for index in row["rgb_indices"]])
    METHOD_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, tokens=token)
    manifest = {"schema": "r3-patch-token-manifest-v1", "samples": len(windows), "shape": list(token.shape), "backbone": "frozen ResNet-18 layer4, 7x7 spatial tokens", "causal": "only the same anchor-[31,23,15,7] history frames used by fair Mode B", "sha256": sha256_file(path)}
    write_json(METHOD_DIR / "patch_token_manifest.json", manifest)
    r2_append_log(OUT / "run_log.jsonl", event="r3_patch_tokens_extracted_after_visual_gate", payload=manifest)
    return {"path": path, "samples": len(windows), "sha256": manifest["sha256"], "status": "created"}


class VisualRelDynVerifier(nn.Module):
    """Frozen visual tokens with temporal association and state/action relation."""

    def __init__(self, ablation: str) -> None:
        super().__init__()
        self.ablation = ablation
        self.reduce = nn.Linear(512, 64)
        self.state = nn.GRU(6, 48, batch_first=True)
        self.action = nn.GRU(6, 32, batch_first=True)
        # visual mean, associated prior-frame memory, state, action, state*action projection
        self.sa = nn.Linear(48 + 32, 48)
        self.head = nn.Sequential(nn.LayerNorm(64 + 64 + 48 + 32 + 48), nn.Linear(256, 96), nn.GELU(), nn.Dropout(0.15), nn.Linear(96, 1))

    def forward(self, tokens: torch.Tensor, history: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        value = self.reduce(tokens.float())  # B,T,P,64
        if self.ablation == "no_history":
            memory = value[:, -1]
            associated = torch.zeros_like(memory[:, 0])
            visual = memory.mean(dim=1)
        else:
            current = value[:, -1]  # B,49,64
            memory = value[:, :-1].reshape(value.shape[0], -1, value.shape[-1])
            if self.ablation == "no_association":
                associated = memory.mean(dim=1)
            else:
                weights = torch.softmax(current @ memory.transpose(1, 2) / math.sqrt(current.shape[-1]), dim=-1)
                associated = (weights @ memory).mean(dim=1)
            visual = value.mean(dim=(1, 2))
        state = self.state(history)[1][-1]
        control = self.action(action)[1][-1]
        if self.ablation == "no_robot_state":
            state = torch.zeros_like(state)
        if self.ablation == "no_action":
            control = torch.zeros_like(control)
        interaction = self.sa(torch.cat((state, control), dim=1))
        if self.ablation == "no_visual_relation":
            associated = torch.zeros_like(associated)
        return self.head(torch.cat((visual, associated, state, control, interaction), dim=1)).squeeze(-1)


def _standardize(train: np.ndarray, value: np.ndarray) -> np.ndarray:
    mean, std = train.mean(axis=0), train.std(axis=0)
    return ((value - mean) / np.where(std < 1e-5, 1.0, std)).astype(np.float32)


def _train_variant(tokens: np.ndarray, history: np.ndarray, action: np.ndarray, label: np.ndarray, split: np.ndarray, ablation: str, seed: int) -> tuple[np.ndarray, np.ndarray, float, dict[str, torch.Tensor]]:
    train, validation, test = (split == name for name in ("train", "validation", "test"))
    history = _standardize(history[train].reshape(train.sum(), -1), history.reshape(len(history), -1)).reshape(history.shape)
    action = _standardize(action[train].reshape(train.sum(), -1), action.reshape(len(action), -1)).reshape(action.shape)
    torch.manual_seed(seed); np.random.seed(seed)
    model = VisualRelDynVerifier(ablation).to(DEVICE)
    tok, hist, act, target = (torch.as_tensor(value, device=DEVICE) for value in (tokens, history, action, label.astype(np.float32)))
    positive_weight = torch.tensor([(label[train] == 0).sum() / max(1, (label[train] == 1).sum())], dtype=torch.float32, device=DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.2e-3, weight_decay=5e-4)
    best: tuple[float, dict[str, torch.Tensor]] | None = None
    for _ in range(160):
        model.train(); optimizer.zero_grad()
        loss = nn.functional.binary_cross_entropy_with_logits(model(tok[train], hist[train], act[train]), target[train], pos_weight=positive_weight)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        model.eval()
        with torch.no_grad(): score = torch.sigmoid(model(tok[validation], hist[validation], act[validation])).detach().cpu().numpy()
        metric = _metrics(label[validation], score, _threshold(label[validation], score))["auroc"]
        if best is None or metric > best[0]:
            best = (metric, {name: value.detach().cpu().clone() for name, value in model.state_dict().items()})
    assert best is not None
    model.load_state_dict(best[1]); model.eval()
    with torch.no_grad():
        val = torch.sigmoid(model(tok[validation], hist[validation], act[validation])).detach().cpu().numpy()
        score = torch.sigmoid(model(tok[test], hist[test], act[test])).detach().cpu().numpy()
    return val, score, _threshold(label[validation], val), best[1]


def _run_split(tokens: np.ndarray, history: np.ndarray, action: np.ndarray, label: np.ndarray, records: list[Mapping[str, Any]], *, split_key: str) -> dict[str, Any]:
    split = np.asarray([str(row[split_key]) for row in records])
    test_rows = [row for row, owner in zip(records, split) if owner == "test"]
    test_label = label[split == "test"]
    result: dict[str, Any] = {}
    raw_root = METHOD_DIR / "raw_scores" / split_key
    raw_root.mkdir(parents=True, exist_ok=True)
    for ablation in ("full", "no_visual_relation", "no_action", "no_history", "no_association", "no_robot_state"):
        values = [_train_variant(tokens, history, action, label, split, ablation, seed) for seed in SEEDS]
        val = np.stack([item[0] for item in values]); score = np.stack([item[1] for item in values]); threshold = float(np.mean([item[2] for item in values]))
        mean = score.mean(axis=0)
        metric = _metrics(test_label, mean, threshold)
        metric["validation_metric"] = _metrics(label[split == "validation"], val.mean(axis=0), threshold)
        metric["per_seed_balanced_accuracy"] = [_metrics(test_label, row, threshold)["balanced_accuracy"] for row in score]
        metric["seed_range"] = float(max(metric["per_seed_balanced_accuracy"]) - min(metric["per_seed_balanced_accuracy"]))
        result[ablation] = metric
        if ablation == "full":
            for seed, item in zip(SEEDS, values):
                torch.save({"state_dict": item[3], "seed": seed, "ablation": ablation, "split": split_key}, METHOD_DIR / f"checkpoint_full_{split_key}_{seed}.pt")
        write_jsonl(raw_root / f"{ablation}.jsonl", [{"variant": ablation, "window_id": row["window_id"], "episode_id": row["episode_id"], "task_name": row["task_name"], "horizon_steps": row["horizon_steps"], "label": int(target), "score_mean": float(value), "seed_scores": [float(one[index]) for one in score]} for index, (row, target, value) in enumerate(zip(test_rows, test_label, mean))])
    return result


def run() -> dict[str, Any]:
    baseline = json.loads((OUT / "baseline_results" / "visual_baseline_report.json").read_text())
    gate = baseline["recoverability_gate"]
    if not gate["gate_pass"]:
        raise R3TaskError("VisualRelDynVerifier is not authorized before R3_VISUAL_HEADROOM_FOUND")
    selected_horizon = str(gate["selected_horizon"])
    selected_model = str(gate["selected_model"])
    token_info = extract_patch_tokens()
    task = build_task()
    features = np.load(METHOD_DIR / "patch_tokens.npz")
    mask = np.asarray([str(row["horizon_steps"]) == selected_horizon for row in task["windows"]])
    records = [row for row, use in zip(task["windows"], mask) if use]
    tokens, history, action, label = (value[mask] for value in (features["tokens"], task["history"], task["action"], task["label"]))
    task_results = _run_split(tokens, history, action, label, records, split_key="task_split")
    episode_results = _run_split(tokens, history, action, label, records, split_key="episode_split")
    baseline_task = float(baseline["reports_by_split"]["task_heldout"][selected_horizon]["metrics"][selected_model]["balanced_accuracy"])
    baseline_episode = float(baseline["reports_by_split"]["episode_heldout"][selected_horizon]["metrics"][selected_model]["balanced_accuracy"])
    task_gain = float(task_results["full"]["balanced_accuracy"] - baseline_task)
    episode_gain = float(episode_results["full"]["balanced_accuracy"] - baseline_episode)
    task_ablation_drops = [task_results["full"]["balanced_accuracy"] - task_results[name]["balanced_accuracy"] for name in task_results if name != "full"]
    accepted = (task_gain >= 0.05 or task_results["full"]["auroc"] - float(baseline["reports_by_split"]["task_heldout"][selected_horizon]["metrics"][selected_model]["auroc"]) >= 0.05) and episode_gain > 0 and sum(drop >= 0.03 for drop in task_ablation_drops) >= 2 and task_results["full"]["seed_range"] <= 0.10 and episode_results["full"]["seed_range"] <= 0.10
    report = {"schema": "r3-visual-rel-dyn-verifier-v1", "method_family_trials": 1, "selected_horizon": int(selected_horizon), "baseline_model": selected_model, "seeds": list(SEEDS), "patch_tokens": token_info, "task_heldout": task_results, "episode_heldout": episode_results, "improvement": {"task": task_gain, "episode": episode_gain}, "acceptance": {"pass": accepted, "decision": "R3_REAL_VISUAL_METHOD_SUPPORTED" if accepted else "R3_RELATIONAL_METHOD_NOT_SUPPORTED", "task_baseline": baseline_task, "episode_baseline": baseline_episode, "required_task_gain": 0.05, "ablation_drops_task": task_ablation_drops}, "fair_inputs": ["pre-anchor patch tokens", "pre-anchor robot state", "candidate action"], "forbidden": ["future RGB", "labels", "failure category", "episode/source identity"]}
    write_json(METHOD_DIR / "method_report.json", report)
    r2_append_log(OUT / "run_log.jsonl", event="r3_visual_relational_method_complete", payload={"accepted": accepted, "task_gain": task_gain, "episode_gain": episode_gain})
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        parser.error("choose --run")
    report = run()
    print(json.dumps(report["acceptance"], sort_keys=True))


if __name__ == "__main__":
    main()
