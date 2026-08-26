"""Audit hidden-state isolation and counterfactual matching for the pilot."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from actmask.data.maniskill_pilot import MODEL_INPUT_KEYS, load_model_inputs


def _read_metadata(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _audit_pair_groups(
    rows: list[dict],
    history: np.ndarray,
    actions: np.ndarray,
    group_key: str,
    expected_regime: str,
) -> dict:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        if row["regime"] == expected_regime and row[group_key] is not None:
            groups[row[group_key]].append(index)
    max_current_state_error = 0.0
    min_history_difference = float("inf")
    max_action_error = 0.0
    max_static_error = 0.0
    flipped = 0
    for indices in groups.values():
        if len(indices) != 2:
            raise AssertionError(f"{group_key} does not contain exactly two members: {indices}")
        left, right = indices
        max_current_state_error = max(
            max_current_state_error,
            float(np.abs(history[left, -1] - history[right, -1]).max()),
        )
        min_history_difference = min(
            min_history_difference,
            float(np.abs(history[left, :-1] - history[right, :-1]).max()),
        )
        max_action_error = max(max_action_error, float(np.abs(actions[left] - actions[right]).max()))
        max_static_error = max(
            max_static_error,
            float(np.abs(np.asarray(rows[left]["static_features"]) - np.asarray(rows[right]["static_features"])).max()),
        )
        flipped += int(bool(rows[left]["label_success"]) != bool(rows[right]["label_success"]))
    if flipped != len(groups):
        raise AssertionError(f"Only {flipped}/{len(groups)} {group_key} groups flip labels")
    tolerance = 1e-6
    if max(max_current_state_error, max_action_error, max_static_error) > tolerance:
        raise AssertionError(
            f"{group_key} exceeded matching tolerance {tolerance}: "
            f"current={max_current_state_error}, action={max_action_error}, static={max_static_error}"
        )
    if min_history_difference <= tolerance:
        raise AssertionError(f"{group_key} has no observable preceding-motion difference")
    return dict(
        groups=len(groups),
        flipped_groups=flipped,
        max_current_state_error=max_current_state_error,
        min_preceding_history_difference=min_history_difference,
        max_action_error=max_action_error,
        max_static_feature_error=max_static_error,
        tolerance=tolerance,
    )


def run_audit(output_dir: str | Path) -> dict:
    output_dir = Path(output_dir)
    result = dict(model_input_keys=list(MODEL_INPUT_KEYS), tasks={})
    for metadata_path in sorted(output_dir.glob("*_metadata.jsonl")):
        task_name = metadata_path.name.removesuffix("_metadata.jsonl")
        inputs = load_model_inputs(output_dir / f"{task_name}_model_inputs.npz")
        rows = _read_metadata(metadata_path)
        if inputs["history"].shape[0] != len(rows):
            raise AssertionError("model input / metadata row count mismatch")
        labels = np.load(output_dir / f"{task_name}_labels.npz")["success"]
        if not np.array_equal(labels.astype(bool), np.asarray([row["label_success"] for row in rows])):
            raise AssertionError("label file / metadata label mismatch")
        result["tasks"][task_name] = dict(
            examples=len(rows),
            matched_counterfactual=_audit_pair_groups(
                rows,
                inputs["history"],
                inputs["candidate_actions"],
                "pair_group",
                "matched_counterfactual",
            ),
            static_balanced=_audit_pair_groups(
                rows,
                inputs["history"],
                inputs["candidate_actions"],
                "static_match_group",
                "static_balanced",
            ),
        )
    result["status"] = "pass"
    return result


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    output_dir = root / "outputs" / "actmask" / "milestone3p_gpu_benchmark" / "maniskill_pilot"
    result = run_audit(output_dir)
    (output_dir / "counterfactual_audit.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
