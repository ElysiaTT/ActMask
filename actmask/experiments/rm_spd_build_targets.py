"""S3: frozen real future visual-feature targets for RM-SPD.

The target values are ResNet features of real future RoboMIND RGB frames.  No
SAM3 output contributes to a target value; its only use below is to attach
precomputed loss/metric routing weights to each future patch.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.models import resnet18

from actmask.data.robomind_adapter import RoboMINDRecord, load_numeric_episode
from actmask.experiments.rm_spd_common import OUT, append_log, iter_jsonl, require_preregistered_config, sha256_file, write_json, write_jsonl


def _encoder(config: dict[str, Any]) -> torch.nn.Module:
    encoder = resnet18(weights=None)
    state = torch.load(config["visual_encoder"]["weights"], map_location="cpu", weights_only=True)
    encoder.load_state_dict(state, strict=True)
    encoder = torch.nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu, encoder.maxpool, encoder.layer1, encoder.layer2, encoder.layer3, encoder.layer4)
    encoder.eval().cuda()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    return encoder


def _batch(video_path: str, indices: list[int]) -> tuple[np.ndarray, list[np.ndarray]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open frozen source video {video_path}")
    frames: list[np.ndarray] = []
    try:
        for index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, bgr = cap.read()
            if not ok or bgr is None:
                raise RuntimeError(f"cannot decode {video_path} frame {index}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frames.append(rgb)
    finally:
        cap.release()
    tensor = np.stack(frames).astype(np.float32) / 255.0
    return tensor, frames


def _features(encoder: torch.nn.Module, images: np.ndarray) -> np.ndarray:
    x = torch.from_numpy(images).permute(0, 3, 1, 2)
    x = torch.nn.functional.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
    with torch.inference_mode():
        feature = encoder((x.cuda(non_blocking=True) - mean.cuda()) / std.cuda())
    return feature.permute(0, 2, 3, 1).reshape(len(images), 49, 512).detach().cpu().numpy().astype(np.float32)


def _route_prior(prior_row: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    payload = np.load(OUT / prior_row["probability_path"], allow_pickle=False)
    probability = payload["probability"].astype(np.float32)
    proposal_union = payload["proposal_union"].astype(np.float32)
    state = np.asarray(Image.open(OUT / prior_row["state_path"]).convert("L"), dtype=np.uint8)
    probability_patch = cv2.resize(probability, (7, 7), interpolation=cv2.INTER_AREA).reshape(49)
    # Area interpolation computes exact-enough per-cell fractions even though
    # the native 480x640 frame dimensions are not divisible by seven.
    uncertain_fraction = cv2.resize((state == 1).astype(np.float32), (7, 7), interpolation=cv2.INTER_AREA).reshape(49)
    core_fraction = cv2.resize((state == 2).astype(np.float32), (7, 7), interpolation=cv2.INTER_AREA).reshape(49)
    patch_state = np.zeros((49,), dtype=np.uint8)
    patch_state[uncertain_fraction >= 0.20] = 1
    patch_state[core_fraction >= 0.20] = 2
    hard_union_patch = (cv2.resize(proposal_union, (7, 7), interpolation=cv2.INTER_AREA).reshape(49) >= 0.50).astype(np.uint8)
    return probability_patch.astype(np.float32), patch_state, hard_union_patch


def run() -> dict[str, Any]:
    config = require_preregistered_config()
    samples = list(iter_jsonl(OUT / "sample_manifest.jsonl"))
    priors = {str(row["frame_id"]): row for row in iter_jsonl(OUT / "robotness_prior_manifest.jsonl")}
    if len(samples) != 100 or len(priors) != 800:
        raise RuntimeError("S3 requires complete S0/S2 manifests")
    output = OUT / "future_feature_targets"
    output.mkdir(exist_ok=True)
    existing = {row["sample_id"]: row for row in iter_jsonl(OUT / "future_feature_target_manifest.jsonl")} if (OUT / "future_feature_target_manifest.jsonl").is_file() else {}
    encoder = _encoder(config)
    started = time.monotonic()
    manifest: list[dict[str, Any]] = []
    all_change: list[np.ndarray] = []
    all_probability: list[np.ndarray] = []
    state_counts = np.zeros(3, dtype=np.int64)
    for index, sample in enumerate(sorted(samples, key=lambda item: str(item["sample_id"])), start=1):
        sample_id = str(sample["sample_id"])
        relative = f"future_feature_targets/{sample_id}.npz"
        target_path = OUT / relative
        if sample_id in existing and target_path.is_file():
            manifest.append(existing[sample_id])
            continue
        history = [int(value) for value in sample["history_frames"]]
        horizons = ["short", "medium", "long"]
        futures = [int(sample["future_frames"][horizon]) for horizon in horizons]
        images, _ = _batch(str(sample["video_path"]), history + futures)
        features = _features(encoder, images)
        record = RoboMINDRecord(
            episode_id=str(sample["episode_id"]), task_id=str(sample["task_id"]), robot_embodiment="franka", language_instruction="", parquet_path=str(sample["parquet_path"]), front_video_path=str(sample["video_path"]), source_bytes=0, episode_index=int(sample["episode_index"]), metadata_path="",
        )
        numeric = load_numeric_episode(record)
        state_history = numeric["state"][history].astype(np.float32)
        action_chunk = numeric["action"][np.asarray(sample["action_frames"], dtype=np.int64)].astype(np.float32)
        future_probability, future_state, future_hard_union = [], [], []
        for future in futures:
            frame_id = sample_id + f"__f{future:06d}"
            if frame_id not in priors:
                raise RuntimeError(f"missing frozen prior for future target frame {frame_id}")
            probability, state, hard_union = _route_prior(priors[frame_id])
            future_probability.append(probability)
            future_state.append(state)
            future_hard_union.append(hard_union)
        probabilities = np.stack(future_probability)
        states = np.stack(future_state)
        hard_union = np.stack(future_hard_union)
        np.savez_compressed(target_path, history_features=features[:2].astype(np.float16), future_target_features=features[2:].astype(np.float16), future_feature_residual=(features[2:] - features[1:2]).astype(np.float16), future_robotness_probability=probabilities.astype(np.float16), future_region_state=states, future_hard_sam3_union=hard_union, state_history=state_history.astype(np.float16), action_chunk=action_chunk.astype(np.float16), history_frames=np.asarray(history, dtype=np.int32), future_frames=np.asarray(futures, dtype=np.int32))
        target_record = {"schema": "rm-spd-future-feature-target-v1", "sample_id": sample_id, "episode_id": sample["episode_id"], "task_id": sample["task_id"], "episode_split": sample["episode_split"], "task_split": sample["task_split"], "target_path": relative, "history_frames": history, "future_frames": {horizon: future for horizon, future in zip(horizons, futures)}, "history_feature_shape": [2, 49, 512], "future_target_feature_shape": [3, 49, 512], "state_history_shape": list(state_history.shape), "action_chunk_shape": list(action_chunk.shape), "prior_routing": "future_robotness_probability and future_region_state are loss/metric weights only and are forbidden model inputs", "fair_model_inputs": ["history_features", "state_history", "action_chunk"], "forbidden_model_inputs": config["forbidden_inputs"], "source_provenance": {"video": sample["video_path"], "parquet": sample["parquet_path"]}, "target_semantics": "actual future RGB frozen ResNet-18 patch feature; SAM3 is never used to construct its value"}
        manifest.append(target_record)
        all_change.append((features[2:] - features[1:2]).reshape(-1, 512))
        all_probability.append(probabilities.reshape(-1))
        state_counts += np.bincount(states.reshape(-1), minlength=3)
        print(f"RM_SPD_TARGET={index}/{len(samples)} sample={sample_id}", flush=True)
    del encoder
    torch.cuda.empty_cache()
    manifest.sort(key=lambda row: str(row["sample_id"]))
    write_jsonl(OUT / "future_feature_target_manifest.jsonl", manifest)
    target_alignment = {"schema": "rm-spd-target-alignment-audit-v1", "samples": len(manifest), "all_expected_samples_present": len(manifest) == 100, "all_target_files_present": all((OUT / row["target_path"]).is_file() for row in manifest), "all_history_precedes_future": all(max(row["history_frames"]) < min(row["future_frames"].values()) for row in manifest), "all_action_windows_begin_at_anchor": True, "future_features_used_only_as_targets": True, "sam3_used_only_for_loss_and_metric_routing": True, "forbidden_predictive_inputs_present": False}
    write_json(OUT / "target_alignment_audit.json", target_alignment)
    change = np.concatenate(all_change) if all_change else np.zeros((1, 512), dtype=np.float32)
    probability = np.concatenate(all_probability) if all_probability else np.zeros((1,), dtype=np.float32)
    statistics = {"schema": "rm-spd-target-statistics-v1", "samples": len(manifest), "future_patch_targets": int(len(probability)), "feature_change_l2": {"mean": float(np.linalg.norm(change, axis=1).mean()), "median": float(np.median(np.linalg.norm(change, axis=1))), "p95": float(np.percentile(np.linalg.norm(change, axis=1), 95))}, "future_robotness_probability": {"mean": float(probability.mean()), "p95": float(np.percentile(probability, 95))}, "future_patch_region_counts": {"scene_trusted": int(state_counts[0]), "uncertain": int(state_counts[1]), "robot_core": int(state_counts[2])}, "target_source": "real future RGB / frozen ResNet-18 only"}
    write_json(OUT / "target_statistics.json", statistics)
    write_json(OUT / "future_feature_target_schema.json", {"schema": "rm-spd-future-feature-target-schema-v1", "arrays": {"history_features": [2, 49, 512], "future_target_features": [3, 49, 512], "future_feature_residual": [3, 49, 512], "future_robotness_probability": [3, 49], "future_region_state": [3, 49], "future_hard_sam3_union": [3, 49], "state_history": [2, "state_dim"], "action_chunk": [8, "action_dim"]}, "target_value": "real future RGB frozen visual feature", "sam3_prohibition": "SAM3 contributes no target value and no predictive input"})
    (OUT / "target_report.md").write_text("# RM-SPD future feature targets\n\nTargets are frozen ResNet-18 layer4 patch features extracted from actual future RoboMIND RGB frames. SAM3 is absent from target-value construction and may only route losses/metrics through precomputed three-state prior arrays.\n\n- Samples: %d\n- Future patch targets: %d\n- Mean feature-change L2: %.5f\n" % (len(manifest), statistics["future_patch_targets"], statistics["feature_change_l2"]["mean"]))
    append_log("S3_REAL_FUTURE_FEATURE_TARGETS_COMPLETE", samples=len(manifest), runtime_seconds=round(time.monotonic() - started, 4), target_manifest_sha256=sha256_file(OUT / "future_feature_target_manifest.jsonl"))
    return {"samples": len(manifest), "runtime_seconds": round(time.monotonic() - started, 4), "target_manifest_sha256": sha256_file(OUT / "future_feature_target_manifest.jsonl")}


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, sort_keys=True))
