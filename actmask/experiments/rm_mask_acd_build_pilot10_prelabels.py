#!/usr/bin/env python3
"""Build a ten-frame, explicitly non-reference robot-mask prelabel pilot."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "actmask" / "rm_mask_acd_verified"
PACKAGE = OUT / "manual_annotation_package"
PILOT = OUT / "pilot10_codex_prelabels"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    rows = [json.loads(line) for line in (PACKAGE / "annotation_manifest.jsonl").read_text().splitlines() if line]
    chosen, seen = [], set()
    for row in rows:
        if row["task_id"] not in seen:
            seen.add(row["task_id"]); chosen.append(row)
    if len(chosen) != 10:
        raise RuntimeError("expected one first sampled frame for each of ten tasks")
    status = json.loads((PACKAGE / "sam3_proposal_status.json").read_text())
    proposals = {row["id"]: row for row in status["records"]}
    masks, overlays = PILOT / "robot_exclusion_prelabels", PILOT / "overlays"
    masks.mkdir(parents=True, exist_ok=True); overlays.mkdir(exist_ok=True)
    manifest = []
    thumbs = []
    for row in chosen:
        proposal = proposals[row["id"]]
        raw = Image.open(PACKAGE / row["frame_path"]).convert("RGB")
        mask = Image.open(PACKAGE / proposal["proposal_path"]).convert("L")
        mask_path = masks / f"{row['id']}.png"; mask.save(mask_path)
        rgb = np.asarray(raw).copy(); active = np.asarray(mask) > 0
        rgb[active] = (0.55 * np.array([255, 70, 70]) + 0.45 * rgb[active]).astype(np.uint8)
        overlay_path = overlays / f"{row['id']}.jpg"; Image.fromarray(rgb).save(overlay_path, quality=94)
        thumb = Image.fromarray(rgb).resize((320, 240)); thumbs.append((thumb, row))
        manifest.append({"id": row["id"], "task_id": row["task_id"], "episode_id": row["episode_id"], "frame_index": row["frame_index"], "image": f"../../manual_annotation_package/{row['frame_path']}", "robot_exclusion_prelabel": f"robot_exclusion_prelabels/{row['id']}.png", "overlay": f"overlays/{row['id']}.jpg", "mask_area_fraction": proposal["robot_proposal_area_fraction"], "provenance": "SAM3 prompt-union prelabel; review aid only; not human label or independent M4 reference"})
    sheet = Image.new("RGB", (5 * 320, 2 * 270), "white"); draw = ImageDraw.Draw(sheet)
    for index, (thumb, row) in enumerate(thumbs):
        x, y = (index % 5) * 320, (index // 5) * 270
        sheet.paste(thumb, (x, y)); draw.text((x + 4, y + 243), row["task_id"].replace("robogene_twoArm_franka_", "")[:42], fill="black")
    sheet_path = PILOT / "pilot10_contact_sheet.jpg"; sheet.save(sheet_path, quality=94)
    payload = {"schema": "rm-mask-acd-pilot10-prelabels-v1", "status": "review_pending", "count": 10, "selection": "first stratified frame from each frozen task", "records": manifest, "contact_sheet": {"path": sheet_path.name, "sha256": sha256(sheet_path)}, "hard_limitations": ["Masks originate from the same SAM3 proposal process used only for annotation assistance.", "They are not independent human annotations and cannot satisfy M3/M4.", "The pilot labels only robot exclusion; manipulated objects and grippers must be separately reviewed in the real annotation package."]}
    (PILOT / "pilot_manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    (PILOT / "README.md").write_text("# Ten-frame robot exclusion prelabel pilot\n\nPink pixels are SAM3 proposal-only robot exclusion masks. This is a visual-quality pilot, not a ground-truth label set. Inspect `pilot10_contact_sheet.jpg` before deciding whether to fund complete human annotation.\n")
    print(json.dumps({"count": len(manifest), "contact_sheet": str(sheet_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
