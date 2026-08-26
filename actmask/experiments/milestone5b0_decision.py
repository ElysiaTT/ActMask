"""Artifact/CI audit and frozen acceptance decision for Milestone 5B-0."""
from __future__ import annotations
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/"outputs"/"actmask"/"milestone5b0_relational_probe"

def metric(report,name): return report["models"][name]["metrics"]["pair_order_accuracy"]["mean"]

def run(output:Path=OUT):
 report=json.loads((output/"state_probe_baseline_report.json").read_text()); manifest=json.loads((output/"raw_score_manifest.json").read_text())
 learned=("current_object_token_mlp","unordered_object_token_deepsets","ordered_object_token_gru","temporal_convolution")
 required=("checkpoint.pt","model_config.json","preprocessing_config.json","seed.txt","data_sha256.txt","raw_scores.jsonl","metrics.json","command.txt")
 artifact_checks=[]
 for name in learned:
  for seed in (17,29,43):
   base=output/"learned"/name/f"seed_{seed}"; artifact_checks.append({"model":name,"seed":seed,"all_required":all((base/x).is_file() for x in required),"files":list(required)})
 ci=[]
 for name,value in report["models"].items():
  for key,summary in value["metrics"].items():
   if "bootstrap_ci95" in summary:
    lo,hi=summary["bootstrap_ci95"]; ci.append({"model":name,"metric":key,"point":summary["mean"],"ci":summary["bootstrap_ci95"],"contains_point":lo<=summary["mean"]<=hi})
 ci_result={"schema":"milestone5b0-ci-consistency-v1","ci_checks":ci,"all_pass":all(x["contains_point"] for x in ci),"raw_score_manifest_entries":len(manifest["entries"]),"raw_scores_saved":manifest["all_raw_scores_saved"]}
 (output/"ci_consistency_audit.json").write_text(json.dumps(ci_result,indent=2,sort_keys=True)+"\n")
 fair=("action_only","current_state_only","centroid_position","last_two_centroid_velocity","multi_frame_centroid_velocity","centroid_acceleration","covariance_trajectory","speed_magnitude_histogram","global_icp","pairwise_distance_histogram","unordered_object_token_statistics","current_object_token_mlp","unordered_object_token_deepsets","ordered_object_token_gru","temporal_convolution")
 best=max((metric(report,n),n) for n in fair); oracle=max(metric(report,n) for n in ("oracle_object_identity_tracker","oracle_relational_graph")); headroom=oracle-best[0]
 head={"schema":"milestone5b0-headroom-v1","oracle_pair_order":oracle,"best_fair_standard_pair_order":best[0],"best_fair_standard":best[1],"headroom":headroom,"required_min":.08,"pass":headroom>=.08}
 (output/"headroom_metrics.json").write_text(json.dumps(head,indent=2,sort_keys=True)+"\n")
 integrity=json.loads((output/"integrity_audit.json").read_text()); diversity=json.loads((output/"candidate_diversity_audit.json").read_text()); moments=json.loads((output/"moment_matching_audit.json").read_text())
 gates={"integrity":integrity["all_pass"],"moments":moments["all_pass"],"candidate_diversity":diversity["pass"],"artifacts":all(x["all_required"] for x in artifact_checks),"ci":ci_result["all_pass"],"oracle_ge_090":oracle>=.9,"action_current_le_060":max(metric(report,"action_only"),metric(report,"current_state_only"))<=.6,"centroid_le_060":max(metric(report,n) for n in ("centroid_position","last_two_centroid_velocity","multi_frame_centroid_velocity","centroid_acceleration"))<=.6,"global_le_065":max(metric(report,n) for n in ("covariance_trajectory","speed_magnitude_histogram","global_icp"))<=.65,"unordered_le_065":metric(report,"unordered_object_token_deepsets")<=.65,"ordered_gru_range":.6<=metric(report,"ordered_object_token_gru")<=.88,"headroom":head["pass"]}
 decision="A. RELATIONAL_PROBE_PASSED_AUTHORIZE_5B" if all(gates.values()) else "G. HUMAN_DECISION_REQUIRED"
 final={"schema":"milestone5b0-final-decision-v1","decision":decision,"gates":gates,"artifact_checks":artifact_checks,"headroom":head,"note":"Authorization is limited to the future Action-Conditioned Relational Dynamics Verifier milestone; no final verifier was trained here."}
 (output/"final_decision.json").write_text(json.dumps(final,indent=2,sort_keys=True)+"\n"); return final

if __name__=="__main__": print(json.dumps(run(),indent=2,sort_keys=True))
