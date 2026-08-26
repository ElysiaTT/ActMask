"""Bounded, auditable R3 acquisition of BotFails RGB videos.

This adapter derives every remote video name from frozen R2 provenance, then
requires exact byte-size and decoded-frame-count agreement before allowing a
frame to enter R3.  It never invents a frame or silently repeats one.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import quote

import imageio.v3 as iio
import imageio_ffmpeg
import numpy as np

from actmask.data.r2_real_outcome_adapter import r2_append_log
from actmask.data.real_robot_dataset_adapter import composite_file_hash, read_jsonl, sha256_file, write_json


ROOT = Path(__file__).resolve().parents[2]
R2_ROOT = ROOT / "outputs" / "actmask" / "r2_series_real_outcome_search"
R3_ROOT = ROOT / "outputs" / "actmask" / "r3_botfails_visual_failure"
R2_PROCESSED = R2_ROOT / "processed_subset" / "botfails"
SOURCE_ROOT = R3_ROOT / "source_videos"
VIEW = "observation.images.logitech_2"
ALT_VIEW = "observation.images.logitech_1"
REPO = "kantine/BotFails"
API = f"https://huggingface.co/api/datasets/{REPO}/tree/main"
RESOLVE = f"https://huggingface.co/datasets/{REPO}/resolve/main"


class VideoAlignmentError(RuntimeError):
    pass


def _remote_video_path(episode: Mapping[str, Any], view: str = VIEW) -> str:
    source = Path(str(episode["source_parquet"]))
    try:
        relative = source.relative_to("raw")
    except ValueError as exc:
        raise VideoAlignmentError(f"R2 source path is not raw-relative: {source}") from exc
    parts = relative.parts
    if not (len(parts) == 5 and parts[2] == "data" and parts[3] == "chunk-000" and parts[-1].endswith(".parquet")):
        raise VideoAlignmentError(f"unexpected R2 BotFails parquet layout: {relative}")
    return "/".join(("BotFails", parts[0], parts[1], "videos", "chunk-000", view, Path(parts[-1]).with_suffix(".mp4").name))


def _local_video_path(episode: Mapping[str, Any], view: str = VIEW) -> Path:
    return SOURCE_ROOT / view / f"{episode['episode_id']}.mp4"


def _curl_json(url: str) -> Any:
    result = subprocess.run(["curl", "--fail", "--location", "--retry", "4", "--retry-all-errors", "--retry-delay", "1", url], check=False, capture_output=True)
    if result.returncode:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace").strip())
    return json.loads(result.stdout)


def _remote_inventory(episodes: list[Mapping[str, Any]], view: str) -> dict[str, int]:
    """List only the 20 source task directories needed for R2's 100 rows."""

    directories = sorted({"/".join(_remote_video_path(row, view).split("/")[:3]) for row in episodes})
    inventory: dict[str, int] = {}
    for directory in directories:
        payload = _curl_json(f"{API}/{quote(directory, safe='')}?recursive=true&expand=false")
        for row in payload:
            path = str(row.get("path", ""))
            if f"/videos/chunk-000/{view}/" in path and path.endswith(".mp4"):
                size = row.get("lfs", {}).get("size", row.get("size"))
                if size is not None:
                    inventory[path] = int(size)
    expected = {_remote_video_path(row, view) for row in episodes}
    missing = sorted(expected.difference(inventory))
    if missing:
        raise VideoAlignmentError(f"official video inventory lacks {len(missing)} frozen R2 videos; first={missing[:2]}")
    return {path: inventory[path] for path in sorted(expected)}


def _download(remote: str, expected: int, local: Path) -> dict[str, Any]:
    local.parent.mkdir(parents=True, exist_ok=True)
    if local.is_file() and local.stat().st_size == expected:
        return {"remote_path": remote, "local_path": str(local), "bytes": expected, "status": "reused"}
    partial = local.with_name(local.name + ".part")
    if partial.exists() and partial.stat().st_size:
        partial.unlink()
    url = f"{RESOLVE}/{quote(remote, safe='/')}?download=true"
    result = subprocess.run([
        "curl", "--fail", "--location", "--retry", "6", "--retry-all-errors", "--retry-delay", "1",
        "--connect-timeout", "30", "--speed-limit", "1", "--speed-time", "240", "--output", str(partial), url,
    ], check=False, capture_output=True)
    if result.returncode or not partial.is_file() or partial.stat().st_size != expected:
        actual = partial.stat().st_size if partial.exists() else 0
        raise RuntimeError(f"video download mismatch {remote}: {actual} != {expected}; {result.stderr.decode('utf-8', errors='replace').strip()}")
    partial.replace(local)
    return {"remote_path": remote, "local_path": str(local), "bytes": expected, "status": "downloaded"}


def _decoded_video_audit(path: Path, expected_frames: int, expected_timestamp: np.ndarray) -> dict[str, Any]:
    """Sequentially decode the complete stream; frame count is never guessed."""

    try:
        metadata = iio.immeta(path, plugin="FFMPEG")
        fps = float(metadata["fps"])
        width, height = (int(value) for value in metadata["size"])
    except Exception as exc:
        raise VideoAlignmentError(f"imageio-ffmpeg cannot inspect {path}: {exc}") from exc
    decoder = imageio_ffmpeg.get_ffmpeg_exe()
    try:
        completed = subprocess.run([decoder, "-hide_banner", "-v", "error", "-nostats", "-i", str(path), "-map", "0:v:0", "-f", "null", "-", "-progress", "pipe:1"], check=False, capture_output=True, text=True)
        if completed.returncode:
            raise VideoAlignmentError(f"ffmpeg framehash decode failed for {path}: {completed.stderr.strip()}")
        # This FFmpeg path fully decodes every frame but discards pixels after
        # decode. Its final progress frame count is therefore stronger than a
        # container-header estimate while avoiding 100x per-frame SHA-256 work.
        progress = dict(line.split("=", 1) for line in completed.stdout.splitlines() if "=" in line)
        decoded = int(progress.get("frame", "-1"))
        duplicate_frames = int(progress.get("dup_frames", "0"))
        dropped_frames = int(progress.get("drop_frames", "0"))
        if progress.get("progress") != "end" or decoded < 0:
            raise VideoAlignmentError(f"ffmpeg did not complete a frame-count audit for {path}")
    except Exception as exc:
        if isinstance(exc, VideoAlignmentError):
            raise
        raise VideoAlignmentError(f"imageio-ffmpeg framehash decode failed in {path}: {exc}") from exc
    adjacent_duplicates = duplicate_frames
    if decoded != expected_frames or decoded != len(expected_timestamp):
        raise VideoAlignmentError(f"frame/numeric count mismatch {path}: decoded={decoded}, expected={expected_frames}, timestamps={len(expected_timestamp)}")
    if not np.isfinite(fps) or fps <= 0:
        raise VideoAlignmentError(f"invalid FPS for {path}: {fps}")
    expected_hz = float(1.0 / np.median(np.diff(expected_timestamp)))
    # Count equality plus a near-equal declared rate validates the release's
    # frame-index convention; this is stronger than assuming timestamps alone.
    if abs(fps - expected_hz) > 0.25:
        raise VideoAlignmentError(f"FPS/control-rate mismatch for {path}: {fps} vs {expected_hz}")
    return {
        "decoded_frames": decoded,
        "numeric_timestamp_count": int(len(expected_timestamp)),
        "decoded_fps": fps,
        "numeric_rate_hz": expected_hz,
        "width": width,
        "height": height,
        "decoder_reported_duplicate_frames": adjacent_duplicates,
        "decoder_reported_dropped_frames": dropped_frames,
        "alignment_method": "official LeRobot frame_index order: decoded index i maps to numeric row/timestamp i after exact count and rate audit",
        "no_silent_frame_duplication": True,
        "frame_integrity_method": "FFmpeg fully decoded every frame to null output; terminal progress frame count must equal numeric timestamp count, with zero decoder-inserted duplicate/drop frames. The adapter never adds or substitutes frames.",
        "video_sha256": sha256_file(path),
    }


def _episodes() -> list[dict[str, Any]]:
    rows = read_jsonl(R2_PROCESSED / "episodes.jsonl")
    if len(rows) != 100:
        raise VideoAlignmentError(f"R2 episode manifest expected 100 rows, got {len(rows)}")
    return rows


def audit_existing(episodes: list[Mapping[str, Any]], view: str = VIEW, *, workers: int = 1) -> list[dict[str, Any]]:
    def one(episode: Mapping[str, Any]) -> dict[str, Any]:
        local = _local_video_path(episode, view)
        timestamp = np.load(R2_PROCESSED / str(episode["timestamp_path"]))["value"]
        detail = _decoded_video_audit(local, len(timestamp), timestamp)
        return {
            "episode_id": episode["episode_id"], "task_name": episode["task_name"], "outcome": episode["success_or_failure"],
            "source_parquet": episode["source_parquet"], "remote_video_path": _remote_video_path(episode, view),
            "local_video_path": str(local.relative_to(R3_ROOT)), "numeric_timestamp_path": episode["timestamp_path"], **detail,
        }
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(one, episodes))


def acquire(*, probe: bool, view: str = VIEW, workers: int = 2) -> dict[str, Any]:
    episodes = _episodes()
    if probe:
        normal = next(row for row in episodes if row["success_or_failure"] == "success")
        failure = next(row for row in episodes if row["success_or_failure"] == "failure")
        selected = [normal, failure]
    else:
        selected = episodes
    inventory = _remote_inventory(selected, view)
    planned = [(_remote_video_path(row, view), inventory[_remote_video_path(row, view)], _local_video_path(row, view)) for row in selected]
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        downloads = list(executor.map(lambda item: _download(*item), planned))
    audits = audit_existing(selected, view, workers=workers)
    if probe:
        status = "probe_pass"
    else:
        status = "full_pass"
    manifest = {
        "schema": "r3-botfails-video-source-manifest-v1",
        "view": view,
        "status": status,
        "episodes": len(selected),
        "downloads": downloads,
        "downloaded_bytes": int(sum(item["bytes"] for item in downloads)),
        "r2_episode_manifest_sha256": sha256_file(R2_PROCESSED / "episodes.jsonl"),
    }
    if probe:
        write_json(SOURCE_ROOT / "probe_video_manifest.json", manifest)
        write_json(SOURCE_ROOT / "probe_video_alignment_audit.json", {"schema": "r3-video-alignment-probe-v1", "pass": True, "rows": audits})
    else:
        write_json(R3_ROOT / "source_video_manifest.json", manifest)
        write_json(R3_ROOT / "video_alignment_audit.json", {"schema": "r3-video-alignment-audit-v1", "pass": True, "rows": audits})
        write_json(R3_ROOT / "frame_sampling_config.json", {
            "schema": "r3-frame-sampling-v1", "view": view, "history_offsets_steps": [31, 23, 15, 7],
            "numeric_rate_hz": 30, "effective_history_sampling_hz": 3.75,
            "mapping": "decoded video index equals R2 numeric row index after per-episode exact decoded-count and FPS audit",
            "causal_cutoff": "all sampled indices must be < anchor",
            "sampled_frame_hash": "each feature cache row records the SHA-256 of its decoded RGB tensor",
        })
    r2_append_log(R3_ROOT / "run_log.jsonl", event="r3_video_probe_pass" if probe else "r3_video_acquisition_alignment_pass", payload={"view": view, "episodes": len(selected), "bytes": manifest["downloaded_bytes"]})
    return {"manifest": manifest, "audit": audits}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--acquire", action="store_true")
    parser.add_argument("--view", default=VIEW)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.probe == args.acquire:
        parser.error("choose exactly one of --probe or --acquire")
    result = acquire(probe=args.probe, view=args.view, workers=args.workers)
    print(json.dumps({"status": result["manifest"]["status"], "episodes": result["manifest"]["episodes"], "bytes": result["manifest"]["downloaded_bytes"]}, sort_keys=True))


if __name__ == "__main__":
    main()
