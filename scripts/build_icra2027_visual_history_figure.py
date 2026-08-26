#!/usr/bin/env python3
"""Render an audited RGB history montage from frozen visual-pilot bundles."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
VISUAL_ROOT = ROOT / "outputs/actmask/milestone4r_v3_all_family_candidate_diversity/full_probe"
DEFAULT_OUTPUT = ROOT / "paper_icra2027/figures/visual_matched_histories.png"
WORLD_ID = 21
CANDIDATE_SLOT = 0

TASKS = (
    ("Moving cube", "MovingCubeIntercept"),
    ("Moving window", "SignedMovingWindowPlacement"),
    ("Rotating window", "FixedPhaseRotatingCaptureWindow"),
)

SOURCE_HASHES = {
    "MovingCubeIntercept_bundles.npz": "b3524da17477a2c716f3d947dbde69415edca4f64843c78147aabbcccbe476dd",
    "MovingCubeIntercept_candidates.jsonl": "911bddb0c5ccad60e70a1bc831b3a68df5fa8719d4ae421195f45b1c6cb1334e",
    "MovingCubeIntercept_labels.npz": "13f3799b04b80d20272649a0d5ab0f0ad0a271469c18b700172ea9f5f9ad12c1",
    "SignedMovingWindowPlacement_bundles.npz": "9b13ecbc26c5484493c045d310c8a4f46b4c07a4e9367a10e8c3622f34cb03d9",
    "SignedMovingWindowPlacement_candidates.jsonl": "faf7efdadba69c905e59f991c672f7ef5c59c1934ac7c05ac6605ae6bd037f78",
    "SignedMovingWindowPlacement_labels.npz": "31d5a7e86c367343d32b64cf2f38963f8adedfea1d9ca7b4ae7cc7fb7793edcd",
    "FixedPhaseRotatingCaptureWindow_bundles.npz": "128168803466ab705f44798f4b70acd14aa4f6a9a76ac3c0ea5dc4329e46343c",
    "FixedPhaseRotatingCaptureWindow_candidates.jsonl": "ee0d00fe35572bc8d5a961233f196c7dbbf57300dc55c7f172e6df74f8f6d05a",
    "FixedPhaseRotatingCaptureWindow_labels.npz": "1e45fd25ee14f3ddd30f8fcf2b6248dec53e689ffade95326c2d803014ebb566",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = Path("/usr/share/fonts/truetype/dejavu") / name
    if path.is_file():
        return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _load_and_check(task: str) -> tuple[np.ndarray, dict[str, Any]]:
    bundle_path = VISUAL_ROOT / f"{task}_bundles.npz"
    row_path = VISUAL_ROOT / f"{task}_candidates.jsonl"
    label_path = VISUAL_ROOT / f"{task}_labels.npz"
    for path in (bundle_path, row_path, label_path):
        observed = _sha256(path)
        expected = SOURCE_HASHES[path.name]
        if observed != expected:
            raise RuntimeError(f"frozen visual source hash mismatch: {path.name}")

    with np.load(bundle_path) as bundle:
        rgb = bundle["rgb"]
        depth = bundle["depth_mm"]
        if rgb.shape != (256, 6, 128, 128, 3) or depth.shape != (256, 6, 128, 128):
            raise RuntimeError(f"unexpected bundle shape for {task}")
        first_ref, second_ref = 2 * WORLD_ID, 2 * WORLD_ID + 1
        if not np.array_equal(rgb[first_ref, 4:], rgb[second_ref, 4:]):
            raise RuntimeError(f"final RGB suffix is not exactly matched for {task}")
        if not np.array_equal(depth[first_ref, 4:], depth[second_ref, 4:]):
            raise RuntimeError(f"final depth suffix is not exactly matched for {task}")
        if np.array_equal(rgb[first_ref, :4], rgb[second_ref, :4]):
            raise RuntimeError(f"early RGB histories do not differ for {task}")
        histories = rgb[[first_ref, second_ref]].copy()

    rows = _read_rows(row_path)
    with np.load(label_path) as labels_file:
        labels = labels_file["success"]
    selected = [
        (row, bool(labels[index]))
        for index, row in enumerate(rows)
        if row["world_id"] == WORLD_ID and row["candidate_slot"] == CANDIDATE_SLOT
    ]
    if len(selected) != 2:
        raise RuntimeError(f"expected one matched candidate pair for {task}")
    selected.sort(key=lambda item: item[0]["history_ref"])
    expected_refs = [2 * WORLD_ID, 2 * WORLD_ID + 1]
    if [item[0]["history_ref"] for item in selected] != expected_refs:
        raise RuntimeError(f"unexpected history references for {task}")
    if any(item[0]["condition"] != "complete" or item[0]["split"] != "test" for item in selected):
        raise RuntimeError(f"figure example is not a complete-history held-test pair for {task}")
    if selected[0][1] is not True or selected[1][1] is not False:
        raise RuntimeError(f"candidate outcomes do not form the frozen positive/negative pair for {task}")
    if selected[0][0]["candidate_actions"] != selected[1][0]["candidate_actions"]:
        raise RuntimeError(f"candidate action differs within pair for {task}")

    audit = {
        "task": task,
        "world_id": WORLD_ID,
        "candidate_slot": CANDIDATE_SLOT,
        "history_refs": expected_refs,
        "outcomes": [True, False],
        "condition": "complete",
        "split": "test",
        "exact_final_two_rgb": True,
        "exact_final_two_depth": True,
        "same_candidate_action": True,
        "early_rgb_differs": True,
    }
    return histories, audit


def build(output: Path) -> dict[str, Any]:
    thumb = 128
    border = 3
    frame_gap = 6
    branch_gap = 28
    left = 190
    top = 54
    row_gap = 24
    legend = 44
    row_height = thumb + row_gap
    branch_width = 6 * thumb + 5 * frame_gap
    width = left + 2 * branch_width + branch_gap + 18
    height = top + len(TASKS) * row_height + legend
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = _font(24, bold=True)
    label_font = _font(22, bold=True)
    small_font = _font(19)
    early_color = (214, 107, 35)
    suffix_color = (38, 105, 166)

    draw.text((left + branch_width // 2, 8), "History A: candidate outcome 1", font=title_font, fill=(20, 20, 20), anchor="ma")
    second_center = left + branch_width + branch_gap + branch_width // 2
    draw.text((second_center, 8), "History B: same candidate, outcome 0", font=title_font, fill=(20, 20, 20), anchor="ma")
    audits = []

    for row_index, (display, task) in enumerate(TASKS):
        histories, audit = _load_and_check(task)
        audits.append(audit)
        y = top + row_index * row_height
        draw.text((12, y + thumb // 2), display, font=label_font, fill=(20, 20, 20), anchor="lm")
        for branch in range(2):
            x_start = left + branch * (branch_width + branch_gap)
            for frame in range(6):
                x = x_start + frame * (thumb + frame_gap)
                image = Image.fromarray(histories[branch, frame], mode="RGB")
                canvas.paste(image, (x, y))
                color = early_color if frame < 4 else suffix_color
                draw.rectangle((x, y, x + thumb - 1, y + thumb - 1), outline=color, width=border)
                draw.text((x + 6, y + 5), f"t{frame + 1}", font=small_font, fill="white", stroke_width=2, stroke_fill="black")

    legend_y = top + len(TASKS) * row_height + 2
    draw.rectangle((left, legend_y + 8, left + 24, legend_y + 30), fill=early_color)
    draw.text((left + 34, legend_y + 19), "early history differs", font=small_font, fill=(20, 20, 20), anchor="lm")
    legend_x = left + 280
    draw.rectangle((legend_x, legend_y + 8, legend_x + 24, legend_y + 30), fill=suffix_color)
    draw.text((legend_x + 34, legend_y + 19), "final two RGB-D frames match exactly", font=small_font, fill=(20, 20, 20), anchor="lm")

    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG", compress_level=9, optimize=False)
    manifest = {
        "schema": "actmask-icra2027-visual-history-figure-v1",
        "source_policy": "Nine frozen artifacts; exact byte hashes required.",
        "sources": [
            {"path": str((VISUAL_ROOT / name).relative_to(ROOT)), "sha256": digest}
            for name, digest in sorted(SOURCE_HASHES.items())
        ],
        "figure": {
            "filename": output.name,
            "width": width,
            "height": height,
            "sha256": _sha256(output),
        },
        "audits": audits,
        "pass": True,
    }
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = build(args.output.resolve())
    print(json.dumps({"pass": manifest["pass"], "figure": manifest["figure"]}, indent=2))


if __name__ == "__main__":
    main()
