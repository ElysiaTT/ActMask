"""Non-heavy verification for the observable-cue data repair."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"outputs"/"actmask"/"milestone5b_data_observable_cues";S=OUT/"state_probe";HASH="296a68818f531e8ff9e51e7f36b1080dd3e281c10029fb60d50fd7f1931a80f1"
def read(p):return json.loads((OUT/p).read_text())
def test_config_manifest_and_moment_cue_matching():
 assert hashlib.sha256((OUT/"preregistered_config.json").read_bytes()).hexdigest()==HASH
 assert read("moment_cue_matching_audit.json")["all_pass"] and json.loads((S/"data_manifest.json").read_text())["candidate_executions"]==2560
def test_cues_identifiability_and_no_shortcuts():
 a=read("fair_identifiability_audit.json");assert a["pass"] and a["exact_conflicting_fair_classes"]==0 and a["fair_bayes_majority_proxy"]==1
 assert a["cue_only_bayes_proxy"]<=.6 and a["current_only_bayes_proxy"]<=.6 and a["unordered_cue_state_bayes_proxy"]<=.65
 assert read("candidate_diversity_audit.json")["pass"] and read("field_separation_audit.json")["pass"]
def test_artifacts_metrics_and_final_schema():
 assert read("raw_score_manifest.json")["all_raw_scores_saved"] and read("ci_consistency_audit.json")["all_pass"]
 final=read("final_decision.json");assert final["decision"] in {"A. DATA_REPAIR_PASSED_AUTHORIZE_5B_V2","B. STILL_FAIR_OBSERVATION_UNIDENTIFIABLE","C. OBSERVABLE_CUE_TRIVIALIZES_TASK","D. GLOBAL_MOTION_SHORTCUT_RETURNS","E. CANDIDATE_RANKING_INVALID","F. TASK_TOO_HARD_OR_NO_HEADROOM","G. ARTIFACT_LOGGING_INVALID","H. HUMAN_DECISION_REQUIRED"}
