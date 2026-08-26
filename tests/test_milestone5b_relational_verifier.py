"""Non-heavy audit for the independently versioned Milestone 5B bundle."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/"outputs"/"actmask"/"milestone5b_relational_verifier"; SOURCE=ROOT/"outputs"/"actmask"/"milestone5b0_relational_probe"; STATE=SOURCE/"state_probe"
CONFIG="02714137ef31f447ab39d865d04290b347dd557cbdaf0b74822f752297b78b0d"; DATA="ba9ae19a2854ee796296c715c0f7b2326c59c8e2511f2312a370ed1c4ebad36a"
REQ=("checkpoint.pt","model_config.json","preprocessing_config.json","training_config.json","seed.txt","data_sha256.txt","command.txt","train_log.jsonl","raw_scores.jsonl","metrics.json")
def test_frozen_source_and_config_hashes():
 assert hashlib.sha256((OUT/"preregistered_config.json").read_bytes()).hexdigest()==CONFIG
 h=hashlib.sha256()
 for f in ("histories.npz","candidate_actions.npy","candidate_labels.jsonl"): h.update((STATE/f).read_bytes())
 assert h.hexdigest()==DATA
def test_relational_artifacts_and_raw_schema():
 for seed in (17,29,43):
  base=OUT/"learned"/"RelDynVerifier"/f"seed_{seed}"; assert all((base/x).is_file() for x in REQ)
  rows=[json.loads(x) for x in (base/"raw_scores.jsonl").read_text().splitlines()]; assert len(rows)==2560 and all({"history_id","pair_id","candidate_slot","score","success","split"}<=set(x) for x in rows)
 assert json.loads((OUT/"input_field_audit.json").read_text())["pass"]
def test_ablations_ci_and_decision_schema():
 report=json.loads((OUT/"ablation_report.json").read_text()); assert set(report["ablations"])=={"no_pairwise_relations","no_action_conditioning","no_temporal_memory","no_permutation_augmentation"}
 assert json.loads((OUT/"ci_consistency_audit.json").read_text())["all_pass"]
 assert json.loads((OUT/"raw_score_manifest.json").read_text())["all_artifacts_saved"]
 assert json.loads((OUT/"final_decision.json").read_text())["decision"] in {"A. METHOD_PASSED_AUTHORIZE_SCALEUP","B. METHOD_NO_BETTER_THAN_STANDARD_TEMPORAL","C. METHOD_SATURATES_ORACLE_SUSPECT_LEAKAGE","D. ARTIFACT_OR_REPRO_INVALID","E. TASK_TOO_SMALL_OR_UNSTABLE","F. HUMAN_REVIEW_REQUIRED"}
