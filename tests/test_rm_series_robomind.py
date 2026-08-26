"""RM-Series contracts: real source metadata, fair inputs and final package."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "outputs" / "actmask" / "rm_series_robomind"


def load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text())


def test_robomind_metadata_and_access_status_are_real_and_bounded() -> None:
    probe = load("access_probe/robomind_file_inventory_probe.json")
    access = load("access_probe/robomind_access_status.json")
    subset = load("source_subset/subset_manifest.json")
    assert probe["readable_parquet_files"] >= 100
    assert probe["readable_hdf5_files"] > 0 and probe["metadata_info_files"] > 0
    assert access["status"] == "LOCAL_READ_ACCESS_AVAILABLE"
    assert not access["gated_access_bypassed"]
    assert subset["episodes"] == 100 and subset["tasks"] == 10
    assert subset["visual_budget_under_10_gb"] and subset["new_remote_download_bytes"] == 0


def test_no_folder_label_inference_and_only_consistency_task_is_authorized() -> None:
    failure_probe = load("access_probe/robomind_failure_metadata_probe.json")
    labels = load("processed_subset/label_semantics_audit.json")
    formulation = load("task_formulations/task_formulation_action_future_consistency.json")
    assert not failure_probe["directory_names_used_as_labels"]
    assert not labels["outcome_or_failure_supervision_authorized"]
    assert labels["formulation_c_authorized"]
    assert formulation["formulation"] == "C" and not formulation["outcome_label_used"]
    assert "physical-counterfactual" not in formulation["claim_boundary"].lower()


def test_adapter_alignment_modality_and_source_hashes() -> None:
    manifest = load("processed_subset/data_manifest.json")
    alignment = load("processed_subset/alignment_audit.json")
    modality = load("processed_subset/modality_audit.json")
    hashes = load("source_hash_manifest.json")
    assert manifest["episode_count"] == 100 and manifest["task_count"] == 10
    assert manifest["fair_input_audit"]["pass"]
    assert all(row["pass"] for row in alignment["episodes"])
    assert modality["front_rgb_videos"] == 100 and modality["state_action_aligned"] == 100
    assert len(hashes["files"]) == 200
    assert all(len(row["sha256"]) == 64 for row in hashes["files"])


def test_splits_negative_sampling_and_no_leakage_are_audited() -> None:
    splits = load("task_formulations/split_audit_action_future_consistency.json")
    negatives = load("task_formulations/negative_sampling_audit_action_future_consistency.json")
    leakage = load("task_formulations/leakage_audit_action_future_consistency.json")
    assert all(value["pass"] for value in splits.values())
    assert splits["task_held_out"]["task_disjoint"]
    assert negatives["status"] == "pass" and not negatives["fallback_pools_used"]
    assert negatives["counts"]["true_continuation"] > 0
    assert negatives["counts"]["same_task_other_attempt"] > 0
    assert leakage["pass"] and not leakage["future_observations_used_in_fair_inputs"]
    assert not leakage["source_or_folder_inputs_used"]


def test_raw_scores_recompute_one_recorded_baseline_metric() -> None:
    report = load("baseline_report.json")["reports"]
    raw = load("raw_score_manifest.json")
    entry = next(item for item in raw["entries"] if item["model"] == "state_action_tcn" and item["split"] == "episode_held_out" and item["seed"] == 17)
    rows = [json.loads(line) for line in (ROOT / entry["path"]).read_text().splitlines() if line]
    label = [int(row["label"]) for row in rows]
    pred = [float(row["score"]) >= 0.5 for row in rows]
    tpr = sum(value and predicted for value, predicted in zip(label, pred)) / sum(label)
    tnr = sum((not value) and (not predicted) for value, predicted in zip(label, pred)) / sum(not value for value in label)
    expected = report["episode_held_out"]["state_action_tcn"]["seeds"][0]["metrics"]["balanced_accuracy"]
    assert abs((tpr + tnr) / 2 - expected) < 1e-9
    assert raw["all_entries_present"] and all((ROOT / item["path"]).is_file() for item in raw["entries"])


def test_final_decision_artifacts_and_hash_chained_log() -> None:
    final = load("final_package/final_decision.json")
    allowed = {
        "RM_ACCESS_REQUIRED_ROBOMIND_GATED", "RM_REJECTED_LABEL_SEMANTICS_UNAVAILABLE", "RM_ADAPTER_OR_ALIGNMENT_FAILED",
        "RM_RECOVERABILITY_INSUFFICIENT", "RM_BASELINES_SATURATE", "RM_SHORTCUT_DOMINATES",
        "RM_BENCHMARK_HEADROOM_FOUND_METHOD_NOT_RUN", "RM_METHOD_NOT_SUPPORTED", "RM_REAL_METHOD_SUPPORTED",
        "RM_METHOD_NOT_ROBUST", "RM_HUMAN_REVIEW_REQUIRED",
    }
    assert final["decision"] in allowed
    if final["method_ran"]:
        assert final["decision"] in {"RM_METHOD_NOT_SUPPORTED", "RM_REAL_METHOD_SUPPORTED", "RM_METHOD_NOT_ROBUST"}
        assert (ROOT / "final_package" / "method_summary.json").is_file()
    else:
        assert final["decision"] == "RM_BENCHMARK_HEADROOM_FOUND_METHOD_NOT_RUN"
    for name in ["final_report.md", "dataset_access_report.md", "adapter_report.md", "task_formulation_report.md", "baseline_summary.json", "raw_score_manifest.json", "reproducibility_manifest.json", "claim_boundary.md", "paper_positioning.md", "next_steps.md"]:
        assert (ROOT / "final_package" / name).is_file()
    reproducibility = load("final_package/reproducibility_manifest.json")
    for item in reproducibility["generated_files"]:
        path = ROOT / item["path"]
        assert path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
    previous = ""
    for line in (ROOT / "run_log.jsonl").read_text().splitlines():
        row = json.loads(line)
        assert row["prev_event_sha256"] == previous
        asserted = dict(row)
        actual = asserted.pop("event_sha256")
        recomputed = hashlib.sha256(json.dumps(asserted, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        assert actual == recomputed
        previous = actual


def test_preregistered_method_artifacts_are_complete_without_tuning() -> None:
    summary = load("final_package/method_summary.json")
    raw = load("raw_score_manifest.json")
    assert summary["family"] == "VisualRelDynVerifier"
    assert summary["gates"]["artifacts_complete"]
    assert not summary["gates"]["episode_improvement_ge_005"]
    assert summary["gates"]["two_ablations_degrade_ge_003"] == "not_run_method_initial_gate_failed"
    assert raw["all_method_artifacts_complete"] and len(raw["method_entries"]) == 6
    required = {"checkpoint.pt", "model_config.json", "preprocessing_config.json", "training_config.json", "seed.txt", "data_sha256.txt", "command.txt", "train_log.jsonl", "raw_scores.jsonl", "metrics.json"}
    for item in raw["method_entries"]:
        run = ROOT / Path(item["path"]).parent
        assert all((run / name).is_file() for name in required)


def test_independent_completion_audit_covers_all_scores_and_split_balance() -> None:
    recomputation = load("raw_score_recomputation_audit.json")
    balance = load("task_formulations/split_balance_audit_action_future_consistency.json")
    media = load("task_formulations/media_duplicate_audit_action_future_consistency.json")
    reports = load("baseline_report.json")["reports"]
    assert recomputation["all_paths_present"] and recomputation["all_hashes_match"] and recomputation["all_metrics_recomputed_match"]
    assert len(recomputation["checks"]) == 100
    for split in balance["splits"].values():
        for part in split.values():
            assert part["every_anchor_has_one_positive_and_four_negatives"]
            assert part["failure_category_balance"] == "not_applicable_no_official_failure_category_field"
    assert all(item["pass"] for item in media.values())
    assert all(value["source_path_dataset_partition_shortcut"]["mean"]["balanced_accuracy"] <= 0.65 for value in reports.values())
    assert all(value["future_frame_plus_action_leakage_diagnostic"]["diagnostic_only"] for value in reports.values())
