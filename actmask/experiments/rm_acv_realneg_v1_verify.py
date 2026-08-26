"""Independent artifact verifier for the terminal RM-ACV-REALNEG-V1 branch."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from actmask.experiments.rm_acv_pilot166_common import (
    binary_metrics,
    read_jsonl,
    sha256_file,
    sha256_json,
)
from actmask.experiments.rm_acv_pilot166_finalize import verify as verify_pilot
from actmask.experiments.rm_acv_realneg_v1_prepare import FAMILIES, OUT, PILOT, ROOT


N7, N8 = FAMILIES
EXPECTED_DECISION = "RM_ACV_REALNEG_LOW_COVERAGE"
REQUIRED_ALWAYS = (
    "preregistered_config.json",
    "preregistered_config.sha256",
    "imported_source_manifest.jsonl",
    "imported_anchor_manifest.jsonl",
    "imported_anchor_arrays.npz",
    "causal_visual_features.npz",
    "training_normalization.npz",
    "pair_candidate_manifest.jsonl",
    "n7_pair_manifest.jsonl",
    "n8_pair_manifest.jsonl",
    "real_negative_manifest.jsonl",
    "real_negative_statistics.json",
    "real_negative_failure_log.jsonl",
    "action_reuse_audit.json",
    "context_reuse_audit.json",
    "pair_grouping_audit.json",
    "episode_leakage_audit.json",
    "duplicate_audit.json",
    "balance_audit.json",
    "n7_shortcut_report.json",
    "n8_shortcut_report.json",
    "shortcut_predictions.jsonl",
    "behavioral_validity_report.json",
    "pair_ambiguity_manifest.jsonl",
    "action_mode_audit.json",
    "threshold_sensitivity_report.json",
    "real_family_decision.json",
    "phase_r6_decision.json",
    "final_decision.json",
    "final_report.md",
    "pair_mining_report.md",
    "real_negative_report.md",
    "shortcut_audit_report.md",
    "behavioral_validity_report.md",
    "balance_and_leakage_report.md",
    "claim_boundary.md",
    "next_steps.md",
    "resource_usage.json",
    "run_log.jsonl",
    "reproducibility_manifest.json",
)
PAPER_FILES = (
    "title_candidates.md",
    "abstract_zh.md",
    "abstract_en.md",
    "introduction_outline.md",
    "problem_formulation.md",
    "real_action_negative_design.md",
    "shortcut_audit.md",
    "benchmark.md",
    "method.md",
    "experiments.md",
    "results_tables.md",
    "figure_plan.md",
    "claim_boundary.md",
    "scale_up_plan.md",
)
UNAUTHORIZED_OUTPUTS = (
    "combined_benchmark_report.md",
    "baseline_report.md",
    "episode_heldout_report.md",
    "task_cv_report.md",
    "candidate_ranking_report.md",
    "calibration_report.md",
    "error_analysis_report.md",
    "raw_prediction_manifest.json",
    "final_method_decision.json",
    "method_report.md",
    "method_summary.json",
    "method_per_seed.json",
    "ablation_summary.json",
    "selective_rejection_report.json",
    "raw_method_prediction_manifest.json",
    "checkpoint_manifest.json",
    "method_reproducibility_manifest.json",
)


def _json(relative: str) -> dict[str, Any]:
    return json.loads((OUT / relative).read_text(encoding="utf-8"))


def _metrics_match(expected: dict[str, Any], actual: dict[str, Any]) -> bool:
    if expected.keys() != actual.keys():
        return False
    for key, value in expected.items():
        other = actual[key]
        if isinstance(value, (int, float)) and isinstance(other, (int, float)):
            if not np.isclose(float(value), float(other), rtol=0.0, atol=1e-12):
                return False
        elif value != other:
            return False
    return True


def _verify_log() -> bool:
    previous = ""
    sequence = 0
    for line in (OUT / "run_log.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        actual = row.pop("event_sha256")
        if row["previous_event_sha256"] != previous or row["seq"] != sequence + 1:
            return False
        if sha256_json(row) != actual:
            return False
        previous = actual
        sequence += 1
    return sequence >= 14


def _verify_imports(*, rehash_source: bool) -> tuple[bool, dict[str, Any]]:
    pilot = verify_pilot(rehash_source=rehash_source)
    audit = _json("import_verification_audit.json")
    rows_ok = True
    for row in audit["imports"]:
        pilot_path = ROOT / row["pilot_path"]
        imported_path = OUT / row["imported_path"]
        rows_ok &= (
            pilot_path.is_file()
            and imported_path.is_file()
            and sha256_file(pilot_path) == row["sha256"]
            and sha256_file(imported_path) == row["sha256"]
            and row["match"]
        )
    return bool(pilot["passed"] and rows_ok and audit["pass"]), pilot


def _verify_pairs_and_samples() -> tuple[bool, dict[str, Any]]:
    anchors = read_jsonl(OUT / "imported_anchor_manifest.jsonl")
    anchors.sort(key=lambda row: int(row["array_index"]))
    with np.load(OUT / "imported_anchor_arrays.npz") as stored:
        arrays = {name: stored[name].copy() for name in stored.files}
    with np.load(OUT / "causal_visual_features.npz") as stored:
        visual = stored["visual"].copy()
    samples = read_jsonl(OUT / "real_negative_manifest.jsonl")
    pairs_by_family = {
        N7: read_jsonl(OUT / "n7_pair_manifest.jsonl"),
        N8: read_jsonl(OUT / "n8_pair_manifest.jsonl"),
    }
    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        by_pair[sample["pair_id"]].append(sample)
    valid = len(anchors) == 992 and len(samples) == 1476
    valid &= arrays["positive_action"].shape == (992, 16, 32)
    valid &= arrays["history_state"].shape == (992, 8, 32)
    valid &= visual.shape[0] == 992
    expected_roles = {
        "Ci_Ai": (0, 0, 1),
        "Cj_Aj": (1, 1, 1),
        "Ci_Aj": (0, 1, 0),
        "Cj_Ai": (1, 0, 0),
    }
    action_counts: dict[str, dict[str, Counter[int]]] = defaultdict(
        lambda: defaultdict(Counter)
    )
    context_counts: dict[str, dict[str, Counter[int]]] = defaultdict(
        lambda: defaultdict(Counter)
    )
    duplicate_splits: dict[tuple[str, str], set[str]] = defaultdict(set)
    all_pair_ids: set[str] = set()
    n7_anchor_counts: Counter[int] = Counter()
    n8_episode_pairs: set[tuple[str, str]] = set()
    for family, pairs in pairs_by_family.items():
        for pair in pairs:
            all_pair_ids.add(pair["pair_id"])
            pair_copy = dict(pair)
            pair_hash = pair_copy.pop("pair_sha256")
            valid &= pair_hash == sha256_json(pair_copy)
            valid &= not pair["action_values_modified"]
            valid &= pair["construction_failure_reason"] is None
            rows = by_pair[pair["pair_id"]]
            valid &= len(rows) == 4 and {row["role"] for row in rows} == set(expected_roles)
            row_by_role = {row["role"]: row for row in rows}
            for role, (context_slot, action_slot, label) in expected_roles.items():
                row = row_by_role[role]
                row_copy = dict(row)
                sample_hash = row_copy.pop("sample_sha256")
                valid &= sample_hash == sha256_json(row_copy)
                context_index = pair["context_indices"][context_slot]
                action_index = pair["context_indices"][action_slot]
                expected_action_hash = hashlib.sha256(
                    arrays["positive_action"][action_index].tobytes()
                ).hexdigest()
                expected_context_hash = hashlib.sha256(
                    arrays["history_state"][context_index].tobytes()
                    + visual[context_index].tobytes()
                    + arrays["instruction"][context_index].tobytes()
                ).hexdigest()
                valid &= (
                    row["family"] == family
                    and row["label"] == label
                    and row["context_index"] == context_index
                    and row["action_index"] == action_index
                    and row["action_hash"] == expected_action_hash
                    and row["context_hash"] == expected_context_hash
                    and row["episode_split"] == pair["episode_split"]
                )
                action_counts[family][row["action_hash"]][label] += 1
                context_counts[family][row["context_id"]][label] += 1
                duplicate_splits[(row["context_hash"], row["action_hash"])].add(
                    row["episode_split"]
                )
            positive_hashes = sorted(
                row["sample_sha256"] for row in rows if row["label"] == 1
            )
            negative_hashes = sorted(
                row["sample_sha256"] for row in rows if row["label"] == 0
            )
            valid &= sorted(pair["positive_sample_hashes"]) == positive_hashes
            valid &= sorted(pair["negative_sample_hashes"]) == negative_hashes
            valid &= all(
                row["positive"] == row["negative"] == 1
                for row in pair["action_reuse_counts"] + pair["context_reuse_counts"]
            )
            if family == N7:
                n7_anchor_counts.update(pair["context_indices"])
            else:
                episode_pair = tuple(sorted(pair["episode_ids_provenance_only"]))
                valid &= episode_pair not in n8_episode_pairs
                n8_episode_pairs.add(episode_pair)
    valid &= set(by_pair) == all_pair_ids
    valid &= len(pairs_by_family[N7]) == 289 and len(pairs_by_family[N8]) == 80
    valid &= max(n7_anchor_counts.values(), default=0) == 1
    valid &= all(
        counts[0] == counts[1]
        for family in action_counts.values()
        for counts in family.values()
    )
    valid &= all(
        counts[0] == counts[1]
        for family in context_counts.values()
        for counts in family.values()
    )
    valid &= not any(len(splits) > 1 for splits in duplicate_splits.values())
    return bool(valid), {
        "anchors": len(anchors),
        "samples": len(samples),
        "N7_pairs": len(pairs_by_family[N7]),
        "N8_pairs": len(pairs_by_family[N8]),
        "N7_unique_anchors": len(n7_anchor_counts),
        "N8_unique_episode_pairs": len(n8_episode_pairs),
    }


def _verify_predictions() -> tuple[bool, dict[str, Any]]:
    samples = {
        row["sample_id"]: row for row in read_jsonl(OUT / "real_negative_manifest.jsonl")
    }
    predictions = read_jsonl(OUT / "shortcut_predictions.jsonl")
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    unique = set()
    valid = True
    for row in predictions:
        key = (row["family"], row["control"], row["split"], row["sample_id"])
        valid &= key not in unique
        unique.add(key)
        sample = samples.get(row["sample_id"])
        valid &= bool(
            sample
            and sample["family"] == row["family"]
            and sample["label"] == row["label"]
            and sample["episode_split"] == row["split"]
        )
        grouped[(row["family"], row["control"], row["split"])].append(row)
    recomputed = 0
    for family, report_file in (
        (N7, "n7_shortcut_report.json"),
        (N8, "n8_shortcut_report.json"),
    ):
        report = _json(report_file)
        for control_name, control in report["controls"].items():
            if "test" not in control:
                continue
            threshold = (
                0.5
                if control_name.startswith("exact_repeated_")
                else float(control["training"]["threshold_selected_on_validation"])
            )
            splits = ("test",) if control_name.startswith("exact_repeated_") else (
                "validation",
                "test",
            )
            for split in splits:
                rows = grouped[(family, control_name, split)]
                labels = np.asarray([row["label"] for row in rows], dtype=np.int64)
                scores = np.asarray([row["score"] for row in rows], dtype=np.float64)
                actual = binary_metrics(labels, scores, threshold)
                valid &= bool(rows) and _metrics_match(control[split], actual)
                recomputed += 1
    expected_controls = sum(
        len(_json(name)["controls"])
        for name in ("n7_shortcut_report.json", "n8_shortcut_report.json")
    )
    return bool(valid), {
        "raw_predictions": len(predictions),
        "unique_predictions": len(unique),
        "metric_blocks_recomputed": recomputed,
        "reported_controls": expected_controls,
    }


def _verify_behavior_and_decision() -> tuple[bool, dict[str, Any]]:
    rows = read_jsonl(OUT / "pair_ambiguity_manifest.jsonl")
    report = _json("behavioral_validity_report.json")
    per_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    valid = len(rows) == 369 and len({row["pair_id"] for row in rows}) == len(rows)
    required = {
        "episode_ids_provenance_only",
        "training_neighborhood_episode_support",
        "neighborhood_anchor_jaccard_overlap",
        "action_mode_overlap",
        "reciprocal_consistency",
        "distinct_local_action_mode_by_factor",
        "alignment_support",
        "padding_support",
    }
    for row in rows:
        valid &= required.issubset(row)
        valid &= not row["alignment_uncertain"]
        valid &= not row["padding_or_terminal_truncation"]
        valid &= all(row["training_neighborhood_episode_support"])
        per_family[row["family"]].append(row)
    for family, family_rows in per_family.items():
        expected = report["families"][family]
        distinct = float(
            np.mean(
                [
                    row["distinct_local_action_mode_by_factor"]["1.0"]
                    for row in family_rows
                ]
            )
        )
        ambiguous = float(np.mean([row["ambiguous"] for row in family_rows]))
        valid &= np.isclose(distinct, expected["distinct_mode_fraction"], atol=1e-12)
        valid &= np.isclose(ambiguous, expected["ambiguous_fraction"], atol=1e-12)
        valid &= expected["pass"] is False
    phase = _json("phase_r6_decision.json")
    final = _json("final_decision.json")
    family_decision = _json("real_family_decision.json")
    valid &= phase == final
    valid &= phase["decision"] == EXPECTED_DECISION and phase["terminal"]
    valid &= not phase["combined_benchmark_construction_authorized"]
    valid &= not phase["learned_fair_baselines_trained"]
    valid &= not phase["proposed_method_authorized"]
    valid &= not phase["valid_new_families"]
    valid &= all(
        row["decision"] == "REALNEG_FAMILY_LOW_COVERAGE"
        for row in family_decision["families"].values()
    )
    return bool(valid), {
        "ambiguity_rows": len(rows),
        "decision": phase["decision"],
        "family_decisions": phase["new_family_decisions"],
    }


def _verify_reproducibility() -> tuple[bool, dict[str, Any]]:
    manifest = _json("reproducibility_manifest.json")
    tracked = {row["path"]: row for row in manifest["generated_files"]}
    actual_files = {
        str(path.relative_to(OUT))
        for path in OUT.rglob("*")
        if path.is_file() and path.name != "reproducibility_manifest.json"
    }
    valid = set(tracked) == actual_files
    for relative, row in tracked.items():
        path = OUT / relative
        valid &= path.is_file() and path.stat().st_size == row["bytes"]
        valid &= path.is_file() and sha256_file(path) == row["sha256"]
    for row in manifest["generators"]:
        path = ROOT / row["path"]
        valid &= path.is_file() and sha256_file(path) == row["sha256"]
    return bool(valid), {
        "tracked_files": len(tracked),
        "actual_files_excluding_manifest": len(actual_files),
        "generators": len(manifest["generators"]),
    }


def verify(*, rehash_source: bool = False) -> dict[str, Any]:
    missing = [
        relative for relative in REQUIRED_ALWAYS if not (OUT / relative).is_file()
    ]
    missing.extend(
        f"paper/{relative}"
        for relative in PAPER_FILES
        if not (OUT / "paper" / relative).is_file()
    )
    if missing:
        return {
            "passed": False,
            "decision": None,
            "missing": sorted(missing),
            "failures": ["missing_required_outputs"],
            "checks": {},
        }
    import_pass, pilot = _verify_imports(rehash_source=rehash_source)
    pair_pass, pair_details = _verify_pairs_and_samples()
    prediction_pass, prediction_details = _verify_predictions()
    behavior_pass, behavior_details = _verify_behavior_and_decision()
    reproducibility_pass, reproducibility_details = _verify_reproducibility()
    config_hash_pass = (
        (OUT / "preregistered_config.sha256").read_text(encoding="utf-8").strip()
        == sha256_file(OUT / "preregistered_config.json")
    )
    resource = _json("resource_usage.json")
    resource_pass = (
        resource["pre_admission_GPU_hours_upper_bound"]
        <= resource["pre_admission_GPU_hours_budget"]
        and resource["new_artifact_bytes"] <= 20 * 1024**3
        and resource["new_negative_family_designs"] <= 2
        and resource["whole_family_redesigns"] == {"N7": 1, "N8": 1}
        and resource["per_example_manual_corrections"] == 0
        and resource["unrelated_processes_terminated"] == 0
    )
    historical_hash = _json("phase_r6_decision.json")["historical_survivors"][
        "historical_manifest_sha256"
    ]
    historical_pass = historical_hash == sha256_file(
        PILOT / "negative_candidate_manifest.jsonl"
    )
    unauthorized_absent = not any((OUT / name).exists() for name in UNAUTHORIZED_OUTPUTS)
    checks = {
        "required_outputs": not missing,
        "preregistered_config_hash": config_hash_pass,
        "frozen_pilot_and_import_hashes": import_pass,
        "reciprocal_pair_and_sample_recomputation": pair_pass,
        "raw_shortcut_prediction_recomputation": prediction_pass,
        "behavior_and_terminal_decision_recomputation": behavior_pass,
        "hash_chained_run_log": _verify_log(),
        "generated_artifact_and_generator_hashes": reproducibility_pass,
        "historical_N1_N4_branch_unchanged": historical_pass,
        "resource_budget": resource_pass,
        "unauthorized_outputs_absent": unauthorized_absent,
    }
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "passed": not failures,
        "decision": _json("phase_r6_decision.json")["decision"],
        "missing": sorted(missing),
        "failures": failures,
        "checks": checks,
        "pair_details": pair_details,
        "prediction_details": prediction_details,
        "behavior_details": behavior_details,
        "reproducibility_details": reproducibility_details,
        "source_verification": {
            "rehash_performed": rehash_source,
            "pilot_passed": pilot["passed"],
            "pilot_failures": pilot["failures"],
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rehash-source", action="store_true")
    args = parser.parse_args()
    result = verify(rehash_source=args.rehash_source)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    raise SystemExit(0 if result["passed"] else 1)
