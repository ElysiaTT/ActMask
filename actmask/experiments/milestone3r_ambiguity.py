"""Materialize no-identity correspondence variants for three frozen 3Q tasks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from actmask.data.maniskill_pilot import load_model_inputs
from actmask.data.milestone3r_ambiguity import AMBIGUITY_TASKS, build_ambiguity


def run(source_root: str | Path, output_root: str | Path) -> dict:
    source, output = Path(source_root), Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"tasks": {}, "fair_variants": ["nearest_neighbor", "soft_correspondence", "no_identity_set"], "oracle_identity_diagnostic_only": True}
    for task_index, task in enumerate(AMBIGUITY_TASKS):
        values = load_model_inputs(source / f"{task}_model_inputs.npz")
        fair, diagnostic = build_ambiguity(values["history"], values["visibility"], seed=401 + task_index)
        if any("id" in key.lower() for key in fair):
            raise AssertionError("simulator identity leaked into fair ambiguity inputs")
        np.savez_compressed(output / f"{task}_ambiguity_inputs.npz", **fair, timestamps=values["timestamps"], candidate_actions=values["candidate_actions"], tcp_state=values["tcp_state"])
        np.savez_compressed(output / f"{task}_labels.npz", **dict(np.load(source / f"{task}_labels.npz")))
        if not np.array_equal(np.load(output / f"{task}_labels.npz")["success"], np.load(source / f"{task}_labels.npz")["success"]):
            raise AssertionError("ambiguity layer changed labels")
        if not np.array_equal(values["candidate_actions"], np.load(output / f"{task}_ambiguity_inputs.npz")["candidate_actions"]):
            raise AssertionError("ambiguity layer changed actions")
        manifest["tasks"][task] = dict(diagnostic, labels_unchanged=True, candidate_actions_unchanged=True)
    (output / "ambiguity_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest
