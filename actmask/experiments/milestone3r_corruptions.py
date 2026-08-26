"""Materialize audited 3R observable corruptions without touching 3Q data."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from actmask.data.maniskill_pilot import MODEL_INPUT_KEYS, load_model_inputs
from actmask.data.milestone3q_signed import TASK_IDS
from actmask.data.milestone3r_observations import PREREGISTERED_SPECS, apply_corruption


def run(source_root: str | Path, output_root: str | Path, *, specs=PREREGISTERED_SPECS) -> dict:
    source, output = Path(source_root), Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"source": str(source), "specifications": [], "tasks": {}}
    for spec in specs:
        variant = output / spec.corruption_id
        variant.mkdir(exist_ok=True)
        manifest["specifications"].append(dict(corruption_id=spec.corruption_id, family=spec.family, severity=spec.severity, seed=spec.seed))
        manifest["tasks"][spec.corruption_id] = {}
        for task in TASK_IDS:
            values = load_model_inputs(source / f"{task}_model_inputs.npz")
            corrupted, audit = apply_corruption(values, spec)
            if set(corrupted) != set(MODEL_INPUT_KEYS):
                raise AssertionError("corruption changed fair input schema")
            source_labels = np.load(source / f"{task}_labels.npz")["success"]
            source_actions = values["candidate_actions"]
            if not np.array_equal(source_actions, corrupted["candidate_actions"]):
                raise AssertionError("corruption changed candidate actions")
            np.savez_compressed(variant / f"{task}_model_inputs.npz", **corrupted)
            np.savez_compressed(variant / f"{task}_labels.npz", **dict(np.load(source / f"{task}_labels.npz")))
            (variant / f"{task}_metadata.jsonl").write_bytes((source / f"{task}_metadata.jsonl").read_bytes())
            copied_labels = np.load(variant / f"{task}_labels.npz")["success"]
            if not np.array_equal(source_labels, copied_labels):
                raise AssertionError("corruption changed labels")
            manifest["tasks"][spec.corruption_id][task] = dict(audit, labels_unchanged=True, candidate_actions_unchanged=True)
    (output / "corruption_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest
