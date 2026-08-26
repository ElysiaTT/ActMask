"""S4/S5: fixed loss-regime benchmark for RM-SPD.

The script deliberately keeps SAM3-derived arrays outside model inputs.  They
are read only to form frozen training weights and evaluation partitions.  All
reported targets are the real future frozen visual features from S3.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from actmask.experiments.rm_spd_common import OUT, append_log, iter_jsonl, require_preregistered_config, sha256_file, write_json, write_jsonl


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REGIMES = ("full_frame", "hard_sam3", "soft_sam3", "tri_state_sam3")
FIXED = ("copy_current", "mean_future_residual", "task_language_only", "episode_progress_diagnostic", "action_magnitude_only", "action_chunk_only", "state_only", "state_action", "camera_background_statistics")
LEARNED = ("current_visual_only", "global_visual_history_gru", "visual_action_gru", "visual_state_action_gru", "patch_action_mlp", "patch_state_action_mlp", "temporal_patch_transformer", "two_head_patch_gru")
CONTACT_HEAVY = {"robogene_twoArm_franka_align_assemble_toy", "robogene_twoArm_franka_assemble_pvc_clamps", "robogene_twoArm_franka_clean_hang_lab_cloth", "robogene_twoArm_franka_clean_plate"}


@dataclass
class Data:
    ids: list[str]
    task: list[str]
    episode_split: list[str]
    task_split: list[str]
    history: torch.Tensor
    target: torch.Tensor
    residual: torch.Tensor
    probability: torch.Tensor
    region: torch.Tensor
    hard: torch.Tensor
    state: torch.Tensor
    action: torch.Tensor
    progress: torch.Tensor


def _seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def _load() -> Data:
    rows = list(iter_jsonl(OUT / "future_feature_target_manifest.jsonl"))
    if len(rows) != 100:
        raise RuntimeError("S4/S5 requires exactly 100 frozen S3 targets")
    arrays: dict[str, list[np.ndarray]] = {k: [] for k in ("history", "target", "residual", "probability", "region", "hard", "state", "action", "progress")}
    ids: list[str] = []; task: list[str] = []; episode_split: list[str] = []; task_split: list[str] = []
    for row in sorted(rows, key=lambda x: str(x["sample_id"])):
        z = np.load(OUT / row["target_path"], allow_pickle=False)
        arrays["history"].append(z["history_features"].astype(np.float32)); arrays["target"].append(z["future_target_features"].astype(np.float32)); arrays["residual"].append(z["future_feature_residual"].astype(np.float32))
        arrays["probability"].append(z["future_robotness_probability"].astype(np.float32)); arrays["region"].append(z["future_region_state"].astype(np.int64)); arrays["hard"].append(z["future_hard_sam3_union"].astype(np.int64))
        arrays["state"].append(z["state_history"].astype(np.float32)); arrays["action"].append(z["action_chunk"].astype(np.float32))
        # This is explicitly a diagnostic, never a fair primary input.
        arrays["progress"].append(np.asarray([max(row["history_frames"]), np.mean(list(row["future_frames"].values()))], dtype=np.float32))
        ids.append(str(row["sample_id"])); task.append(str(row["task_id"])); episode_split.append(str(row["episode_split"])); task_split.append(str(row["task_split"]))
    return Data(ids, task, episode_split, task_split, *[torch.from_numpy(np.stack(arrays[k])) for k in ("history", "target", "residual", "probability", "region", "hard", "state", "action", "progress")])


def _weights(data: Data, regime: str) -> torch.Tensor:
    if regime == "full_frame": return torch.ones_like(data.probability)
    if regime == "hard_sam3": return (data.hard == 0).float()
    if regime == "soft_sam3": return (1.0 - data.probability).clamp(0, 1).pow(2.0)
    if regime == "tri_state_sam3": return torch.where(data.region == 0, 1.0, torch.where(data.region == 1, 0.2, 0.0)).float()
    raise ValueError(regime)


def _zscore(x: torch.Tensor, train: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = x[train].mean(0, keepdim=True); std = x[train].std(0, keepdim=True).clamp_min(1e-5)
    return (x - mean) / std, mean, std


def _controls(data: Data, train: torch.Tensor) -> dict[str, torch.Tensor]:
    action = data.action.reshape(len(data.ids), -1); state = data.state.reshape(len(data.ids), -1)
    visual = data.history.mean(2).reshape(len(data.ids), -1)
    camera = torch.cat((data.history.mean((1, 2)), data.history.std((1, 2))), 1)
    action_mag = torch.linalg.vector_norm(data.action, dim=-1).mean(1, keepdim=True)
    progress = data.progress
    out = {"action": action, "state": state, "visual": visual, "camera": camera, "action_mag": action_mag, "progress": progress}
    return {k: _zscore(v, train)[0] for k, v in out.items()}


class GlobalMLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__(); self.net = nn.Sequential(nn.Linear(dim, 512), nn.GELU(), nn.LayerNorm(512), nn.Linear(512, 3 * 512))
    def forward(self, x: torch.Tensor, _: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1, 3, 1, 512).expand(-1, -1, 49, -1)


class GlobalGRU(nn.Module):
    def __init__(self, control_dim: int):
        super().__init__(); self.gru = nn.GRU(512, 256, batch_first=True); self.head = nn.Sequential(nn.Linear(256 + control_dim, 512), nn.GELU(), nn.Linear(512, 3 * 512))
    def forward(self, visual: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
        h, _ = self.gru(visual.mean(2)); return self.head(torch.cat((h[:, -1], control), 1)).view(-1, 3, 1, 512).expand(-1, -1, 49, -1)


class PatchMLP(nn.Module):
    def __init__(self, control_dim: int):
        super().__init__(); self.net = nn.Sequential(nn.Linear(1024 + control_dim, 768), nn.GELU(), nn.Linear(768, 3 * 512))
    def forward(self, visual: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
        b = len(visual); x = torch.cat((visual.reshape(b, 49, 1024), control[:, None].expand(-1, 49, -1)), -1)
        return self.net(x).view(b, 49, 3, 512).permute(0, 2, 1, 3)


class PatchTransformer(nn.Module):
    def __init__(self, control_dim: int):
        super().__init__(); self.inp = nn.Linear(512, 192); layer = nn.TransformerEncoderLayer(192, 4, 384, batch_first=True, activation="gelu"); self.enc = nn.TransformerEncoder(layer, 2); self.control = nn.Linear(control_dim, 192); self.head = nn.Sequential(nn.Linear(384, 512), nn.GELU(), nn.Linear(512, 3 * 512))
    def forward(self, visual: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
        b = len(visual); tokens = self.enc(self.inp(visual).reshape(b, 98, 192)); last = tokens[:, 49:98]
        c = self.control(control)[:, None].expand(-1, 49, -1); return self.head(torch.cat((last, c), -1)).view(b, 49, 3, 512).permute(0, 2, 1, 3)


class TwoHeadPatchGRU(nn.Module):
    def __init__(self, control_dim: int):
        super().__init__(); self.gru = nn.GRU(512, 320, batch_first=True); self.control = nn.Linear(control_dim, 128); self.scene = nn.Linear(448, 3 * 512); self.motion = nn.Linear(448, 3 * 512)
    def forward(self, visual: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
        b = len(visual); seq = visual.permute(0, 2, 1, 3).reshape(b * 49, 2, 512); h, _ = self.gru(seq); c = self.control(control)[:, None].expand(-1, 49, -1).reshape(b * 49, 128); z = torch.cat((h[:, -1], c), -1)
        return ((self.scene(z) + self.motion(z)) / 2).view(b, 49, 3, 512).permute(0, 2, 1, 3)


def _task_onehot(data: Data, train: torch.Tensor) -> torch.Tensor:
    known = sorted({data.task[i] for i in train.nonzero().flatten().tolist()}); pos = {x: i for i, x in enumerate(known)}; x = torch.zeros((len(data.ids), len(known)), dtype=torch.float32)
    for i, name in enumerate(data.task):
        if name in pos: x[i, pos[name]] = 1
    return x


def _fixed_prediction(name: str, data: Data, train: torch.Tensor, control: dict[str, torch.Tensor]) -> torch.Tensor:
    base = torch.zeros_like(data.residual)
    if name == "copy_current": return base
    if name == "mean_future_residual": return data.residual[train].mean(0, keepdim=True).expand_as(base)
    if name == "task_language_only":
        global_mean = data.residual[train].mean(0); result = []
        for task in data.task:
            idx = train & torch.tensor([x == task for x in data.task]); result.append(data.residual[idx].mean(0) if idx.any() else global_mean)
        return torch.stack(result)
    # Fixed shortcut controls use ridge regression; progress remains a diagnostic.
    source = {"episode_progress_diagnostic": control["progress"], "action_magnitude_only": control["action_mag"], "action_chunk_only": control["action"], "state_only": control["state"], "state_action": torch.cat((control["state"], control["action"]), 1), "camera_background_statistics": control["camera"]}[name]
    # The multi-output solve is intentionally on GPU: it has 75k regression
    # targets and is a poor fit for the small CPU allocation of this instance.
    x = torch.cat((torch.ones((len(source), 1)), source), 1).to(DEVICE); y = data.residual.reshape(len(source), -1).to(DEVICE); train_gpu = train.to(DEVICE); xt = x[train_gpu]; beta = torch.linalg.solve(xt.T @ xt + 1e-2 * torch.eye(xt.shape[1], device=DEVICE), xt.T @ y[train_gpu]); return (x @ beta).view_as(base.to(DEVICE)).cpu()


def _model(name: str, control: dict[str, torch.Tensor]) -> tuple[nn.Module, torch.Tensor, bool]:
    if name == "current_visual_only": return PatchMLP(0), torch.empty((len(control["visual"]), 0)), True
    if name == "global_visual_history_gru": return GlobalGRU(0), torch.empty((len(control["visual"]), 0)), False
    if name == "visual_action_gru": return GlobalGRU(control["action"].shape[1]), control["action"], False
    if name == "visual_state_action_gru":
        c = torch.cat((control["state"], control["action"]), 1); return GlobalGRU(c.shape[1]), c, False
    if name == "patch_action_mlp": return PatchMLP(control["action"].shape[1]), control["action"], True
    if name == "patch_state_action_mlp":
        c = torch.cat((control["state"], control["action"]), 1); return PatchMLP(c.shape[1]), c, True
    if name == "temporal_patch_transformer":
        c = torch.cat((control["state"], control["action"]), 1); return PatchTransformer(c.shape[1]), c, True
    if name == "two_head_patch_gru":
        c = torch.cat((control["state"], control["action"]), 1); return TwoHeadPatchGRU(c.shape[1]), c, True
    raise ValueError(name)


def _fit(name: str, data: Data, train: torch.Tensor, regime: str, control: dict[str, torch.Tensor]) -> torch.Tensor:
    _seed(17); model, c, patch = _model(name, control); model.to(DEVICE); opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4); weight = _weights(data, regime).to(DEVICE); target = data.residual.to(DEVICE); visual = data.history.to(DEVICE); c = c.to(DEVICE); indices = train.nonzero().flatten().to(DEVICE)
    for _ in range(45):
        model.train(); pred = model(visual[indices], c[indices]); w = weight[indices, :, :, None]; loss = ((pred - target[indices]).square() * w).sum() / (w.sum() * 512).clamp_min(1.0); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    model.eval()
    with torch.inference_mode(): return model(visual, c).detach().cpu()


def _metric_one(pred_res: np.ndarray, target_res: np.ndarray, target: np.ndarray, region: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    pred = target - target_res + pred_res
    valid = weights > 0
    scene = region == 0; uncertain = region == 1; core = region == 2
    def avg(value: np.ndarray, mask: np.ndarray) -> float:
        return float(value[mask].mean()) if mask.any() else float("nan")
    pn = np.linalg.norm(pred, axis=-1).clip(1e-7); tn = np.linalg.norm(target, axis=-1).clip(1e-7); cosine = 1.0 - (pred * target).sum(-1) / (pn * tn)
    l2 = np.linalg.norm(pred - target, axis=-1); mag = np.abs(np.linalg.norm(pred_res, axis=-1) - np.linalg.norm(target_res, axis=-1))
    low = np.linalg.norm(target_res, axis=-1) <= np.percentile(np.linalg.norm(target_res, axis=-1), 25)
    retrieval: list[float] = []
    for h in range(len(pred)):
        m = scene[h]
        if m.any():
            q = pred[h, m]; cand = target[h]; nnidx = (q @ cand.T / (np.linalg.norm(q, axis=1, keepdims=True).clip(1e-7) * np.linalg.norm(cand, axis=1)[None].clip(1e-7))).argmax(1); retrieval.extend((nnidx == np.where(m)[0]).astype(float).tolist())
    return {"scene_cosine_error": avg(cosine, scene), "scene_l2_error": avg(l2, scene), "patch_retrieval_accuracy": float(np.mean(retrieval)) if retrieval else float("nan"), "change_magnitude_error": avg(mag, scene), "static_border_error_proxy": avg(l2, scene & low), "robot_core_l2_diagnostic": avg(l2, core), "uncertain_region_coverage": float(uncertain.mean()), "weighted_l2": avg(l2, valid), "full_frame_l2": float(l2.mean())}


def _evaluate(raw_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    metrics: list[dict[str, Any]] = []
    for row in raw_rows:
        z = np.load(OUT / row["prediction_path"], allow_pickle=False); pred, target_res, target, region, weight = z["predicted_residual"].astype(np.float32), z["target_residual"].astype(np.float32), z["target_feature"].astype(np.float32), z["region_state"], z["loss_weight"].astype(np.float32)
        m = _metric_one(pred, target_res, target, region, weight)
        metrics.append({**{k: row[k] for k in ("model", "regime", "sample_id", "task_id", "episode_split", "task_split")}, **m})
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in metrics: grouped.setdefault((row["model"], row["regime"]), []).append(row)
    summary: dict[str, Any] = {}
    numeric = ("scene_cosine_error", "scene_l2_error", "patch_retrieval_accuracy", "change_magnitude_error", "static_border_error_proxy", "robot_core_l2_diagnostic", "uncertain_region_coverage", "weighted_l2", "full_frame_l2")
    for (model, regime), rows in grouped.items():
        def mean(where: list[dict[str, Any]], key: str) -> float:
            x = [r[key] for r in where if not math.isnan(r[key])]; return float(np.mean(x)) if x else float("nan")
        entry: dict[str, Any] = {"test": {key: mean(rows, key) for key in numeric}, "task_heldout": {key: mean([r for r in rows if r["task_split"] == "test"], key) for key in numeric}, "contact_heavy": {key: mean([r for r in rows if r["task_id"] in CONTACT_HEAVY], key) for key in numeric}, "per_horizon_note": "raw predictions retain all three horizons; aggregate sample metrics are recomputable from prediction files"}
        summary[f"{model}::{regime}"] = entry
    return metrics, summary


def _run() -> dict[str, Any]:
    config = require_preregistered_config(); started = time.monotonic(); data = _load(); train = torch.tensor([x == "train" for x in data.episode_split]); test = torch.tensor([x == "test" for x in data.episode_split]); control = _controls(data, train); raw_dir = OUT / "raw_predictions"; raw_dir.mkdir(exist_ok=True); raw_rows: list[dict[str, Any]] = []
    # The test split is never used for fit or selection.  We write one raw file
    # per prediction to make the later metric recomputation independent.
    for regime in REGIMES:
        w = _weights(data, regime)
        predictions: dict[str, torch.Tensor] = {name: _fixed_prediction(name, data, train, control) for name in FIXED}
        for name in LEARNED:
            predictions[name] = _fit(name, data, train, regime, control)
            print(f"RM_SPD_BASELINE={name} regime={regime}", flush=True)
        for name, residual in predictions.items():
            for i in test.nonzero().flatten().tolist():
                rel = f"raw_predictions/{name}__{regime}__{data.ids[i]}.npz"; path = OUT / rel
                np.savez_compressed(path, predicted_residual=residual[i].numpy().astype(np.float16), target_residual=data.residual[i].numpy().astype(np.float16), target_feature=data.target[i].numpy().astype(np.float16), region_state=data.region[i].numpy().astype(np.uint8), loss_weight=w[i].numpy().astype(np.float16), history_feature=data.history[i].numpy().astype(np.float16))
                raw_rows.append({"schema": "rm-spd-raw-prediction-v1", "model": name, "regime": regime, "sample_id": data.ids[i], "task_id": data.task[i], "episode_split": data.episode_split[i], "task_split": data.task_split[i], "prediction_path": rel, "prediction_semantics": "predicted future-feature residual from fair historical features/state/action only", "sam3_role": "stored only as region/weight for scoring; absent from model inputs", "sha256": sha256_file(path)})
    write_jsonl(OUT / "raw_prediction_manifest.json", raw_rows)
    metrics, summary = _evaluate(raw_rows); write_jsonl(OUT / "baseline_per_sample_metrics.jsonl", metrics); write_json(OUT / "baseline_summary.json", {"schema": "rm-spd-baseline-summary-v1", "models": list(FIXED + LEARNED), "regimes": list(REGIMES), "test_samples": int(test.sum()), "summary": summary, "metric_definitions": {"scene": "region_state==scene_trusted", "retrieval": "within-horizon patch-index retrieval using predicted/actual future feature cosine", "static_border_error_proxy": "low-real-change scene-trusted patch feature error; not segmentation accuracy"}})
    # Independent recomputation is a second read of the raw files, not cached
    # in-memory predictions.  Its digest is included in the final audit.
    recomputed, recomputed_summary = _evaluate(list(iter_jsonl(OUT / "raw_prediction_manifest.json"))); digest = hashlib.sha256(json.dumps(recomputed_summary, sort_keys=True, allow_nan=True).encode()).hexdigest(); primary_json = json.dumps(summary, sort_keys=True, allow_nan=True); recomputed_json = json.dumps(recomputed_summary, sort_keys=True, allow_nan=True); write_json(OUT / "baseline_metrics_recomputed.json", {"schema": "rm-spd-independent-metric-recompute-v1", "records": len(recomputed), "summary_sha256": digest, "matches_primary_summary": recomputed_json == primary_json})
    append_log("S4_S5_BASELINES_COMPLETE", models=len(FIXED + LEARNED), regimes=len(REGIMES), raw_predictions=len(raw_rows), runtime_seconds=round(time.monotonic() - started, 4))
    return {"models": len(FIXED + LEARNED), "raw_predictions": len(raw_rows), "runtime_seconds": round(time.monotonic() - started, 4), "device": str(DEVICE), "config_sha256": sha256_file(OUT / "preregistered_config.json")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--audit-only", action="store_true"); args = parser.parse_args()
    if args.audit_only:
        rows = list(iter_jsonl(OUT / "raw_prediction_manifest.json")); _, summary = _evaluate(rows); print(json.dumps({"records": len(rows), "summary_sha256": hashlib.sha256(json.dumps(summary, sort_keys=True, allow_nan=True).encode()).hexdigest()}, sort_keys=True))
    else: print(json.dumps(_run(), sort_keys=True))
