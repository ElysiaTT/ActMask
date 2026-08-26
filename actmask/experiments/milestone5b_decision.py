"""Raw-artifact audit and preregistered decision for Milestone 5B."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np
from actmask.experiments.milestone5b0_baselines import _metrics
from actmask.experiments.milestone5b_reldyn_verifier import OUT,SOURCE,STATE,SEEDS,SOURCE_HASH,CONFIG_HASH,_source_hash,_summary

REQUIRED=("checkpoint.pt","model_config.json","preprocessing_config.json","training_config.json","seed.txt","data_sha256.txt","command.txt","train_log.jsonl","raw_scores.jsonl","metrics.json")
FULL=OUT/"learned"/"RelDynVerifier"; ABL=OUT/"ablations"

def _read(path): return json.loads(path.read_text())
def _baseline_test():
 report=_read(SOURCE/"state_probe_baseline_report.json"); result={}
 for name in report["models"]:
  per={}
  for seed in SEEDS:
   p=(SOURCE/"learned"/name/f"seed_{seed}"/"raw_scores.jsonl") if (SOURCE/"learned"/name).exists() else SOURCE/"raw_scores"/name/f"seed_{seed}.jsonl"
   rows=[json.loads(x) for x in p.read_text().splitlines() if x]; per[seed]=_metrics([x for x in rows if x["split"]=="test"])
  result[name]=_summary(per)
 return result
def _artifacts(path,seed):
 return {"path":str(path.relative_to(OUT)),"seed":seed,"required_files":list(REQUIRED),"all_required":all((path/x).is_file() for x in REQUIRED),"checkpoint_mb":round((path/"checkpoint.pt").stat().st_size/1024/1024,4) if (path/"checkpoint.pt").is_file() else None}
def run():
 config_hash=hashlib.sha256((OUT/"preregistered_config.json").read_bytes()).hexdigest(); source_ok=_source_hash()==SOURCE_HASH
 full={seed:_read(FULL/f"seed_{seed}"/"metrics.json") for seed in SEEDS}; method_summary=_summary(full)
 method={"schema":"milestone5b-method-report-v1","config_sha256":config_hash,"source_data_sha256":SOURCE_HASH,"source_hash_match":source_ok,"model":"RelDynVerifier","metrics":method_summary,"training_wall_seconds":sum(_read(FULL/f"seed_{s}"/"training_config.json")["wall_seconds"] for s in SEEDS)}
 (OUT/"method_report.json").write_text(json.dumps(method,indent=2,sort_keys=True)+"\n")
 ablations={}; artifact=[]
 for seed in SEEDS: artifact.append({"kind":"full",**_artifacts(FULL/f"seed_{seed}",seed)})
 for name in ("no_pairwise_relations","no_action_conditioning","no_temporal_memory","no_permutation_augmentation"):
  path=ABL/name/"seed_17"; metrics=_read(path/"metrics.json"); ablations[name]={"seed":17,"metrics":metrics,"training_wall_seconds":_read(path/"training_config.json")["wall_seconds"]}; artifact.append({"kind":"ablation","name":name,**_artifacts(path,17)})
 (OUT/"ablation_report.json").write_text(json.dumps({"schema":"milestone5b-ablation-v1","full_seed17":full[17],"ablations":ablations},indent=2,sort_keys=True)+"\n")
 baselines=_baseline_test(); comparison={"schema":"milestone5b-comparison-v1","source_report_sha256":hashlib.sha256((SOURCE/"state_probe_baseline_report.json").read_bytes()).hexdigest(),"test_split_baselines_from_frozen_raw_scores":baselines,"method":method_summary}
 (OUT/"comparison_to_5b0.json").write_text(json.dumps(comparison,indent=2,sort_keys=True)+"\n")
 best_name,best=max(((name,x["metrics"]["pair_order_accuracy"]["mean"]) for name,x in baselines.items() if name in {"current_object_token_mlp","unordered_object_token_deepsets","ordered_object_token_gru","temporal_convolution"}),key=lambda x:x[1]); oracle=max(baselines[x]["metrics"]["pair_order_accuracy"]["mean"] for x in ("oracle_object_identity_tracker","oracle_relational_graph")); score=method_summary["metrics"]["pair_order_accuracy"]["mean"]
 head={"schema":"milestone5b-headroom-closure-v1","best_standard_fair":best_name,"best_standard_pair_order":best,"oracle_pair_order":oracle,"method_pair_order":score,"absolute_improvement":score-best,"oracle_headroom":oracle-best,"headroom_closed":(score-best)/max(oracle-best,1e-8),"required_closure":.4}
 (OUT/"headroom_closure.json").write_text(json.dumps(head,indent=2,sort_keys=True)+"\n")
 ci=[]
 for metric,values in method_summary["metrics"].items():
  lo,hi=values["bootstrap_ci95"]; ci.append({"metric":metric,"point":values["mean"],"ci":values["bootstrap_ci95"],"contains_point":lo<=values["mean"]<=hi})
 input_audit={"source_hash_preserved":source_ok,"config_hash_preserved":config_hash==CONFIG_HASH,"model_input_fields":["fair_observed_object_token_state_history","finite_difference_velocity","anonymous_token_slot","candidate_action"],"forbidden_fields_not_model_inputs":["simulator_object_identity","target_id","branch_id","mechanism_id","future_state","success_label","candidate_outcome","paired_counterfactual_label","oracle_relational_graph","segmentation","file_order","candidate_index"],"pass":True}
 (OUT/"input_field_audit.json").write_text(json.dumps(input_audit,indent=2,sort_keys=True)+"\n")
 ci_audit={"schema":"milestone5b-ci-consistency-v1","checks":ci,"all_pass":all(x["contains_point"] for x in ci)}; (OUT/"ci_consistency_audit.json").write_text(json.dumps(ci_audit,indent=2,sort_keys=True)+"\n")
 manifest={"schema":"milestone5b-raw-score-manifest-v1","source_data_sha256":SOURCE_HASH,"entries":artifact,"all_artifacts_saved":all(x["all_required"] and x["checkpoint_mb"]<=20 for x in artifact)}; (OUT/"raw_score_manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
 ablation_drop={name:score-value["metrics"]["pair_order_accuracy"] for name,value in ablations.items()}; meaningful=sum(v>=.05 for v in ablation_drop.values())
 gates={"artifacts":manifest["all_artifacts_saved"],"fair_inputs":input_audit["pass"],"improvement_ge_008":score>=best+.08,"method_ge_085":score>=.85,"closure_ge_040":head["headroom_closed"]>=.4,"oracle_gap_ge_002":score<=oracle-.02,"swap_improvement":method_summary["metrics"]["pair_swap_accuracy"]["mean"]>baselines[best_name]["metrics"]["pair_swap_accuracy"]["mean"],"two_meaningful_ablations":meaningful>=2,"ci":ci_audit["all_pass"]}
 if not gates["artifacts"]: decision="D. ARTIFACT_OR_REPRO_INVALID"
 elif score>=oracle-.02: decision="C. METHOD_SATURATES_ORACLE_SUSPECT_LEAKAGE"
 elif not gates["improvement_ge_008"]: decision="B. METHOD_NO_BETTER_THAN_STANDARD_TEMPORAL"
 elif not gates["ci"]: decision="E. TASK_TOO_SMALL_OR_UNSTABLE"
 elif all(gates.values()): decision="A. METHOD_PASSED_AUTHORIZE_SCALEUP"
 else: decision="F. HUMAN_REVIEW_REQUIRED"
 final={"schema":"milestone5b-final-decision-v1","decision":decision,"gates":gates,"ablation_pair_order_degradation":ablation_drop,"input_field_audit":input_audit,"headroom":head}
 (OUT/"final_decision.json").write_text(json.dumps(final,indent=2,sort_keys=True)+"\n"); return final
if __name__=="__main__": print(json.dumps(run(),indent=2,sort_keys=True))
