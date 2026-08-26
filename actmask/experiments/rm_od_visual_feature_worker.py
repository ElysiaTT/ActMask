"""Bounded, read-only RGB patch-stat extractor for RM-OD.

The worker is intentionally short-lived because the container OpenCV build
keeps native video-decoder memory.  It produces deterministic *frozen* global
and 4x4 patch RGB statistics, not labels or trainable visual features.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def _features(video_path: str, steps: list[int]) -> tuple[np.ndarray, np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {video_path}")
    globals_: list[np.ndarray] = []
    patches: list[np.ndarray] = []
    try:
        for step in steps:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(step))
            ok, frame = cap.read()
            if not ok or frame is None:
                raise ValueError(f"cannot decode step {step}: {video_path}")
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            small = cv2.resize(rgb, (8, 8), interpolation=cv2.INTER_AREA)
            token = small.reshape(4, 2, 4, 2, 3).mean((1, 3)).reshape(16, 3)
            globals_.append(np.concatenate([rgb.mean((0, 1)), rgb.std((0, 1))]).astype(np.float32))
            patches.append(token.astype(np.float32))
    finally:
        cap.release()
    return np.stack(globals_), np.stack(patches)


def run(manifest: Path, output: Path) -> dict[str, int]:
    output.mkdir(parents=True, exist_ok=True)
    rows = json.loads(manifest.read_text())
    for row in rows:
        anchors, history, future = int(row["anchors"]), int(row["history"]), int(row["future"])
        pre_global, pre_patch = _features(str(row["video_path"]), [int(value) for value in row["pre_steps"]])
        future_global, future_patch = _features(str(row["video_path"]), [int(value) for value in row["future_steps"]])
        if pre_global.shape != (anchors * history, 6) or pre_patch.shape != (anchors * history, 16, 3):
            raise ValueError(f"unexpected pre-feature shape for {row['episode_id']}: {pre_global.shape}, {pre_patch.shape}")
        if future_global.shape != (anchors * future, 6) or future_patch.shape != (anchors * future, 16, 3):
            raise ValueError(f"unexpected future-feature shape for {row['episode_id']}: {future_global.shape}, {future_patch.shape}")
        np.savez_compressed(
            output / f"{row['episode_id'].replace(':', '_')}.npz",
            pre_global=pre_global.reshape(anchors, history, 6),
            pre_patch=pre_patch.reshape(anchors, history, 16, 3),
            future_global=future_global.reshape(anchors, future, 6),
            future_patch=future_patch.reshape(anchors, future, 16, 3),
        )
    return {"episodes": len(rows)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.manifest, args.output), sort_keys=True))
