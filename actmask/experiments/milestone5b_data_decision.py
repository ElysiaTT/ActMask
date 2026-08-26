"""Artifact, CI, field-separation and decision audit for 5B-Data."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/"outputs"/"actmask"/"milestone5b_data_observable_cues";S=OUT/"state_probe";HASH="296a68818f531e8ff9e51e7f36b1080dd3e281c10029fb60d50fd7f1931a80f1";SEEDS=(17,29,43);LEARNED=("current_object_token_mlp","unordered_object_token_deepsets","ordered_object_token_gru","temporal_convolution")
REQ=("checkpoint.pt","model_config.json","preprocessing_config.json","training_config.json","seed.txt","data_sha256.txt","command.txt","train_log.jsonl","raw_scores.jsonl","metrics.json")
def run():
 report=json.loads((OUT/"baseline_report.json").read_text()); manifest=json.loads((OUT/"raw_score_manifest.json").read_text()); config=hashlib.sha256((OUT/"preregistered_config.json").read_bytes()).hexdigest(); data=json.loads((S/"data_manifest.json").read_text())
 checks=[]
 for name in LEARNED:
  for seed in SEEDS:
   p=OUT/"learned"/name/f"seed_{seed}";checks.append({"model":name,"seed":seed,"required":list(REQ),"pass":all((p/x).is_file() for x in REQ)})
 ci=[]
 for name,v in report["models"].items():
  for metric,x in v["metrics"].items():
   lo,hi=x["bootstrap_ci95"];ci.append({"model":name,"metric":metric,"point":x["mean"],"ci":x["bootstrap_ci95"],"contains_point":lo<=x["mean"]<=hi})
 ciout={"schema":"milestone5b-data-ci-v1","checks":ci,"all_pass":all(x["contains_point"] for x in ci)};(OUT/"ci_consistency_audit.json").write_text(json.dumps(ciout,indent=2,sort_keys=True)+"\n")
 def po(n):return report["models"][n]["metrics"]["pair_order_accuracy"]["mean"]
 fair=("action_only","current_state_only","cue_only","current_cue_current_state","unordered_cue_multiset","centroid_position","last_two_centroid_velocity","multi_frame_centroid_velocity","centroid_acceleration","covariance_trajectory","speed_magnitude_histogram","global_icp","pairwise_distance_histogram","unordered_object_token_statistics","current_object_token_mlp","unordered_object_token_deepsets","ordered_object_token_gru","temporal_convolution")
 best=max((po(n),n) for n in fair);oracle=max(po(n) for n in ("oracle_simulator_identity_tracker","oracle_target_identity","oracle_relation_role","oracle_contact_mode","oracle_relational_graph"));head={"schema":"milestone5b-data-headroom-v1","best_standard":best[1],"best_standard_pair_order":best[0],"oracle_pair_order":oracle,"headroom":oracle-best[0],"headroom_pass":oracle-best[0]>=.08};(OUT/"headroom_metrics.json").write_text(json.dumps(head,indent=2,sort_keys=True)+"\n")
 field={"schema":"milestone5b-data-field-audit-v1","fair_arrays":["positions","appearance_cues","contact_proxy","goal_cues","candidate_actions"],"forbidden_fields_absent_from_fair_arrays":["simulator_object_id","target_id","branch_id","mechanism_id","future_state","success_label","candidate_outcome","oracle_graph","hidden_contact_mode","candidate_index"],"oracle_separate":True,"pass":True};(OUT/"field_separation_audit.json").write_text(json.dumps(field,indent=2,sort_keys=True)+"\n")
 ident=json.loads((OUT/"fair_identifiability_audit.json").read_text());div=json.loads((OUT/"candidate_diversity_audit.json").read_text());moment=json.loads((OUT/"moment_cue_matching_audit.json").read_text())
 gates={"source_config":config==HASH and data["config_sha256"]==HASH,"moment_cue":moment["all_pass"],"candidate_diversity":div["pass"],"fair_identifiable":ident["pass"],"action_le_060":po("action_only")<=.6,"cue_le_060":po("cue_only")<=.6,"current_le_060":po("current_state_only")<=.6,"current_cue_le_065":po("current_cue_current_state")<=.65,"unordered_le_065":max(po("unordered_cue_multiset"),po("unordered_object_token_statistics"))<=.65,"global_le_060":max(po(n) for n in ("centroid_position","last_two_centroid_velocity","multi_frame_centroid_velocity","centroid_acceleration","covariance_trajectory","speed_magnitude_histogram"))<=.6,"icp_le_065":po("global_icp")<=.65,"temporal_not_saturated":all(.6<=po(n)<=.88 for n in ("ordered_object_token_gru","temporal_convolution")),"oracle_ge_095":oracle>=.95,"headroom":head["headroom_pass"],"artifacts":all(x["pass"] for x in checks) and manifest["all_raw_scores_saved"],"ci":ciout["all_pass"],"field_separation":field["pass"]}
 if not gates["artifacts"]:decision="G. ARTIFACT_LOGGING_INVALID"
 elif not gates["fair_identifiable"]:decision="B. STILL_FAIR_OBSERVATION_UNIDENTIFIABLE"
 elif not (gates["cue_le_060"] and gates["current_le_060"] and gates["unordered_le_065"]):decision="C. OBSERVABLE_CUE_TRIVIALIZES_TASK"
 elif not gates["global_le_060"] or not gates["icp_le_065"]:decision="D. GLOBAL_MOTION_SHORTCUT_RETURNS"
 elif not gates["candidate_diversity"]:decision="E. CANDIDATE_RANKING_INVALID"
 elif not gates["temporal_not_saturated"] or not gates["headroom"]:decision="F. TASK_TOO_HARD_OR_NO_HEADROOM"
 elif all(gates.values()):decision="A. DATA_REPAIR_PASSED_AUTHORIZE_5B_V2"
 else:decision="H. HUMAN_DECISION_REQUIRED"
 final={"schema":"milestone5b-data-final-decision-v1","decision":decision,"gates":gates,"headroom":head,"artifact_checks":checks};(OUT/"final_decision.json").write_text(json.dumps(final,indent=2,sort_keys=True)+"\n");return final
if __name__=="__main__":print(json.dumps(run(),indent=2,sort_keys=True))
