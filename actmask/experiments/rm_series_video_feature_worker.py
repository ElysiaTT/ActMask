"""Short-lived, bounded OpenCV worker for RM-Series visual statistics.

Some container OpenCV builds retain native decoder buffers across many video
opens.  The parent audit therefore launches this dependency-light worker for
small batches.  It reads source MP4s only and writes compact deterministic
pre-anchor features; it never sees labels, split assignments, or models.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def frame_stats(video_path: str, steps: list[int]) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"unreadable front video: {video_path}")
    values: list[np.ndarray] = []
    try:
        for step in steps:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(step))
            ok, frame = cap.read()
            if not ok or frame is None:
                raise ValueError(f"cannot decode frame {step} from {video_path}")
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            small = cv2.resize(rgb, (4, 4), interpolation=cv2.INTER_AREA)
            values.append(np.concatenate([rgb.mean((0, 1)), rgb.std((0, 1)), small.mean(axis=2).reshape(-1)]).astype(np.float32))
    finally:
        cap.release()
    return np.stack(values)


def run(manifest_path: Path, output_root: Path) -> dict[str, int]:
    output_root.mkdir(parents=True, exist_ok=True)
    entries = json.loads(manifest_path.read_text())
    for row in entries:
        value = frame_stats(str(row["video_path"]), [int(item) for item in row["history_frame_steps"]])
        shape = (int(row["anchors"]), int(row["history"]), 22)
        expected_flat = (shape[0] * shape[1], shape[2])
        if value.shape != expected_flat:
            raise ValueError(f"unexpected extracted visual shape {value.shape}, expected {expected_flat}")
        np.savez_compressed(output_root / f"{row['episode_id'].replace(':', '_')}.npz", value=value.reshape(shape))
    return {"episodes": len(entries)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.manifest, args.output), sort_keys=True))
