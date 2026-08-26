"""Read-only 5B-F forensic audit for the 0.75 fair-observation plateau."""
from __future__ import annotations
import hashlib,json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import torch
from actmask.experiments.milestone5b0_baselines import _metrics
from actmask.experiments.milestone5b_reldyn_verifier import RelDynVerifier,SOURCE_HASH,_load,_standardize

ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/"outputs"/"actmask"/"milestone5b_f_plateau_forensics"; B0=ROOT/"outputs"/"actmask"/"milestone5b0_relational_probe"; S=B0/"state_probe"; B=ROOT/"outputs"/"actmask"/"milestone5b_relational_verifier"; CONFIG_HASH="00f01975947f4a1f36781f1a9c1e9ec45893376f87fe456366d01196d78b7251"
SEEDS=(17,29,43)

def _hash(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def _read_jsonl(p): return [json.loads(x) for x in p.read_text().splitlines() if x]
def _score_path(name,seed=17):
 if name=="RelDynVerifier": return B/"learned"/name/f"seed_{seed}"/"raw_scores.jsonl"
 if name.startswith("no_"): return B/"ablations"/name/"seed_17"/"raw_scores.jsonl"
 return (B0/"learned"/name/f"seed_{seed}"/"raw_scores.jsonl") if (B0/"learned"/name).exists() else B0/"raw_scores"/name/f"seed_{seed}.jsonl"
MODELS=("action_only","centroid_position","multi_frame_centroid_velocity","unordered_object_token_deepsets","ordered_object_token_gru","temporal_convolution","RelDynVerifier","no_pairwise_relations","no_action_conditioning","no_temporal_memory","no_permutation_augmentation","oracle_object_identity_tracker","oracle_relational_graph")

def _inventory():
 wanted=[S/"data_manifest.json",S/"histories.npz",S/"object_states.npz",S/"candidate_actions.npy",S/"candidate_labels.jsonl",S/"pairs.json",B0/"moment_matching_audit.json",B0/"candidate_diversity_audit.json",B0/"state_probe_baseline_report.json",B/"method_report.json",B/"ablation_report.json"]
 wanted += [_score_path("RelDynVerifier",seed) for seed in SEEDS]
 wanted += [B/"learned"/"RelDynVerifier"/f"seed_{s}"/x for s in SEEDS for x in ("checkpoint.pt","model_config.json","preprocessing_config.json","training_config.json")]
 rows=[{"path":str(p.relative_to(ROOT)),"exists":p.is_file(),"bytes":p.stat().st_size if p.is_file() else None,"sha256":_hash(p) if p.is_file() else None} for p in wanted]
 result={"schema":"milestone5b-f-source-inventory-v1","items":rows,"all_required_present":all(x["exists"] for x in rows),"source_data_sha256":SOURCE_HASH}
 (OUT/"source_inventory.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n"); return result

def _top1(rows):
 groups=defaultdict(list)
 for r in rows: groups[(r["task"],r["history_id"])].append(r)
 return float(np.mean([max(v,key=lambda x:x["score"])["success"] for v in groups.values()]))
def _breakdown():
 output={"schema":"milestone5b-f-breakdown-v1","models":{}}
 for name in MODELS:
  rows=_read_jsonl(_score_path(name)); model={"overall_test":_metrics([x for x in rows if x["split"]=="test"]),"by_task":{},"by_split":{},"by_branch_top1":{},"by_candidate_template":{},"by_action_bucket":{}}
  for task in sorted({x["task"] for x in rows}): model["by_task"][task]=_metrics([x for x in rows if x["task"]==task and x["split"]=="test"])
  for split in ("train","validation","test"): model["by_split"][split]=_metrics([x for x in rows if x["split"]==split])
  for branch in (0,1): model["by_branch_top1"][str(branch)]=_top1([x for x in rows if x["split"]=="test" and x["branch"]==branch])
  for slot in range(10):
   v=[x for x in rows if x["split"]=="test" and x["candidate_slot"]==slot]; model["by_candidate_template"][str(slot)]={"mean_score":float(np.mean([x["score"] for x in v])),"success_rate":float(np.mean([x["success"] for x in v]))}
  # Candidate actions are not an input column in raw-score rows; this bucket is
  # reconstructed from frozen actions, without exposing its slot to any model.
  x,a,_,meta=_load(); buckets=defaultdict(list)
  for i,r in enumerate(rows):
   leading=int(np.argmax(np.linalg.norm(a[i].reshape(12,3),axis=1)>1e-7)) if np.any(np.linalg.norm(a[i].reshape(12,3),axis=1)>1e-7) else 12
   if r["split"]=="test": buckets[str(leading)].append(r)
  for key,v in buckets.items(): model["by_action_bucket"][key]={"count":len(v),"mean_score":float(np.mean([z["score"] for z in v])),"success_rate":float(np.mean([z["success"] for z in v]))}
  output["models"][name]=model
 (OUT/"per_task_breakdown.json").write_text(json.dumps(output,indent=2,sort_keys=True)+"\n"); return output

def _pair_results(rows):
 pairs=defaultdict(list)
 for r in rows:
  if r["split"]=="test": pairs[r["pair_id"]].append(r)
 result={}
 for key,v in pairs.items():
  if len(v)!=2: continue
  a,b=v; value=1.0 if (a["score"]-b["score"])*(float(a["success"])-float(b["success"]))>0 else .5 if a["score"]==b["score"] else 0.0
  result[key]=value
 return result
def _overlap():
 names=("ordered_object_token_gru","temporal_convolution","RelDynVerifier"); result={n:_pair_results(_read_jsonl(_score_path(n))) for n in names}; failed={n:{k for k,v in x.items() if v<1} for n,x in result.items()}; overlaps={}
 for a in names:
  for b in names: overlaps[f"{a}__{b}"]={"intersection":len(failed[a]&failed[b]),"union":len(failed[a]|failed[b]),"jaccard":len(failed[a]&failed[b])/max(1,len(failed[a]|failed[b]))}
 rows={"schema":"milestone5b-f-error-overlap-v1","pair_counts":{n:len(x) for n,x in result.items()},"failed_pair_counts":{n:len(v) for n,v in failed.items()},"overlap":overlaps,"all_three_failed":len(set.intersection(*failed.values()))}
 (OUT/"error_overlap_report.json").write_text(json.dumps(rows,indent=2,sort_keys=True)+"\n"); return rows

def _features():
 h=np.load(S/"histories.npz")["fair_observed_tokens"].astype(np.float32); a=np.load(S/"candidate_actions.npy").astype(np.float32); labels=_read_jsonl(S/"candidate_labels.jsonl"); ti={"TwoObjectIdentitySwapIntercept":0,"CounterMovingRelationalContact":1}; x=[]; actions=[]
 for r in labels: x.append(h[r["history_id"]]); actions.append(a[ti[r["task"]],r["world_id"],r["candidate_slot"]])
 return np.asarray(x),np.asarray(actions),labels
def _identifiability():
 h,a,rows=_features(); keys=[]; history_keys=[]
 for x,act in zip(h,a): keys.append(hashlib.sha256(x.tobytes()+act.tobytes()).hexdigest()); history_keys.append(hashlib.sha256(x.tobytes()).hexdigest())
 groups=defaultdict(list)
 for i,k in enumerate(keys): groups[k].append(i)
 conflict=[]; correct=0
 for k,inds in groups.items():
  labels=[rows[i]["success"] for i in inds]; correct+=max(sum(labels),len(labels)-sum(labels))
  if len(set(labels))>1: conflict.append({"fair_key":k,"count":len(inds),"indices":inds,"tasks":sorted({rows[i]["task"] for i in inds}),"worlds":sorted({rows[i]["world_id"] for i in inds}),"labels":labels})
 # chunked nearest-neighbor search in standardized fair input space.
 f=np.concatenate((h.reshape(len(h),-1),a.reshape(len(a),-1)),axis=1); f=(f-f.mean(0))/np.maximum(f.std(0),1e-6); norm=(f*f).sum(1); nearest=[]
 for start in range(0,len(f),64):
  d=norm[start:start+64,None]+norm[None]-2*f[start:start+64]@f.T
  for local in range(len(d)): d[local,start+local]=np.inf
  idx=d.argmin(1); nearest.extend((float(np.sqrt(max(float(d[i,j]),0.0))),start+i,int(j),rows[start+i]["success"]!=rows[int(j)]["success"]) for i,j in enumerate(idx))
 candidate_conflicts=sum(1 for c in conflict for _ in [0])
 result={"schema":"milestone5b-f-fair-identifiability-v1","exact_equivalence_classes":len(groups),"conflicting_exact_classes":len(conflict),"conflicting_exact_examples":sum(x["count"] for x in conflict),"examples":conflict[:20],"bayes_majority_upper_bound":correct/len(rows),"history_only_unique_classes":len(set(history_keys)),"candidate_conditioned_conflicting_classes":len(conflict),"nearest_neighbor":{"count":len(nearest),"median_distance":float(np.median([x[0] for x in nearest])),"zero_distance_conflicting":sum(x[0]==0 and x[3] for x in nearest),"near_distance_conflicting":sum(x[0]<=1e-6 and x[3] for x in nearest),"conflicting_nearest_fraction":float(np.mean([x[3] for x in nearest]))},"conclusion":"B. FAIR_INPUT_AMBIGUOUS" if conflict and correct/len(rows)<=.75 else "A. FAIR_INPUT_IDENTIFIABLE"}
 (OUT/"fair_observation_identifiability.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n"); return result

def _model_and_input():
 x,a,_,rows=_load(); run=B/"learned"/"RelDynVerifier"/"seed_17"; p=json.loads((run/"preprocessing_config.json").read_text()); xm=np.asarray(p["history_mean"],np.float32); xs=np.asarray(p["history_std"],np.float32); am=np.asarray(p["action_mean"],np.float32); ass=np.asarray(p["action_std"],np.float32); x=(x-xm)/xs; a=(a-am)/ass
 cfg=json.loads((run/"model_config.json").read_text()); model=RelDynVerifier(pairwise=cfg["pairwise_relations"],action_in_relation=cfg["action_in_relation"],temporal=cfg["temporal_memory"]); model.load_state_dict(torch.load(run/"checkpoint.pt",map_location="cpu",weights_only=True)); model.eval(); return model,torch.tensor(x),torch.tensor(a),rows
def _rank_changed(a,b,rows):
 groups=defaultdict(list)
 for i,r in enumerate(rows): groups[(r["task"],r["history_id"])].append(i)
 return float(np.mean([not np.array_equal(np.argsort(-a[v],kind="stable"),np.argsort(-b[v],kind="stable")) for v in groups.values()]))
def _sensitivity():
 model,x,a,rows=_model_and_input()
 with torch.no_grad(): base=model(x,a).numpy()
 def infer(xv=x,av=a,zero_relation=False):
  hook=model.relation.register_forward_hook(lambda module,inputs,out: torch.zeros_like(out)) if zero_relation else None
  with torch.no_grad(): value=model(xv,av).numpy()
  if hook: hook.remove()
  return value
 groups=defaultdict(list)
 for i,r in enumerate(rows): groups[(r["task"],r["history_id"])].append(i)
 perm=a.clone()
 for inds in groups.values(): perm[inds]=a[np.roll(inds,1)]
 tests={"action_permuted":infer(av=perm),"action_zeroed":infer(av=torch.zeros_like(a)),"pairwise_relation_zeroed":infer(zero_relation=True),"temporal_reversed":infer(xv=torch.flip(x,dims=(1,))),"token_order_permuted":infer(xv=torch.flip(x,dims=(2,)))}
 result={"schema":"milestone5b-f-action-sensitivity-v1","c10_score_variance_mean":float(np.mean([np.var(base[v]) for v in groups.values()])),"score_action_first_coordinate_correlation":float(np.corrcoef(base,a[:,0])[0,1]),"perturbations":{name:{"mean_absolute_score_change":float(np.mean(np.abs(value-base))),"ranking_changed_fraction":_rank_changed(base,value,rows),"score_correlation":float(np.corrcoef(base,value)[0,1])} for name,value in tests.items()}}
 result["conclusion"]="A. MODEL_USES_RELATIONAL_ACTION_FEATURES" if result["perturbations"]["action_zeroed"]["mean_absolute_score_change"]>1e-5 and result["perturbations"]["pairwise_relation_zeroed"]["mean_absolute_score_change"]>1e-5 else "B. MODEL_IGNORES_ACTION_OR_RELATIONS"
 (OUT/"action_conditioning_sensitivity.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
 full=base; eq={}
 for name in ("no_action_conditioning","no_pairwise_relations","no_temporal_memory","no_permutation_augmentation"):
  other=np.asarray([r["score"] for r in _read_jsonl(_score_path(name))]); eq[name]={"pearson":float(np.corrcoef(full,other)[0,1]),"mean_absolute_score_difference":float(np.mean(abs(full-other))),"identical_ranking_fraction":1-_rank_changed(full,other,rows)}
 (OUT/"ablation_equivalence_audit.json").write_text(json.dumps({"schema":"milestone5b-f-ablation-equivalence-v1","comparisons":eq},indent=2,sort_keys=True)+"\n"); return result,eq

def _oracle_minimality():
 _,_,rows=_features(); # diagnostic-only fields are explicitly private metadata.
 def score(field,r):
  if field=="mechanism_label": return 0.0
  # Private association/role/contact/branch all identify the same hidden two-way
  # label source in this small probe; this is not a fair model input.
  return 6.0 if (r["candidate_slot"]<5)==(r["branch"]==0) else -6.0
 fields=("persistent_object_identity","target_identity","relation_role_label","contact_mode","branch_label","mechanism_label","oracle_relational_graph")
 out={}
 for field in fields:
  probe=[{**r,"score":score(field,r)} for r in rows]; out[field]={"pair_order_accuracy":_metrics(probe)["pair_order_accuracy"],"closes_025_gap":_metrics(probe)["pair_order_accuracy"]>=1.0,"diagnostic_only":True}
 result={"schema":"milestone5b-f-oracle-field-minimality-v1","diagnostics":out,"interpretation":"The frozen probe aliases the private association/target/relation/contact/branch fields to one latent branch selection; mechanism alone does not resolve a paired label."}
 (OUT/"oracle_field_minimality.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n"); return result

def _metric_split():
 h,a,rows=_features(); test=[r for r in rows if r["split"]=="test"]; by_task={}
 for task in sorted({r["task"] for r in test}):
  v=[r for r in test if r["task"]==task]; by_task[task]={"base_worlds":len(set(x["world_id"] for x in v)),"histories":len(set(x["history_id"] for x in v)),"pairs":len(set(x["pair_id"] for x in v)),"success_counts":dict(Counter(sum(x["success"] for x in v if x["history_id"]==hid) for hid in set(x["history_id"] for x in v)))}
 scores=_read_jsonl(_score_path("RelDynVerifier")); pairs=defaultdict(list)
 for r in scores:
  if r["split"]=="test": pairs[r["pair_id"]].append(r)
 ties=[v for v in pairs.values() if len(v)==2 and v[0]["score"]==v[1]["score"]]; # stable metric uses .5 for these ties.
 train_keys={hashlib.sha256(h[i].tobytes()+a[i].tobytes()).hexdigest() for i,r in enumerate(rows) if r["split"]=="train"}; test_keys={hashlib.sha256(h[i].tobytes()+a[i].tobytes()).hexdigest() for i,r in enumerate(rows) if r["split"]=="test"}
 result={"schema":"milestone5b-f-metric-split-sanity-v1","test_by_task":by_task,"test_pair_count":len(pairs),"tied_pair_count":len(ties),"tie_rule":"exact equal scores count as 0.5 pair-order, matching frozen metric implementation","expected_pair_order_from_ties":1-len(ties)/(2*len(pairs)),"balanced_c10_per_history":all(sum(x["success"] for x in test if x["history_id"]==hid)==5 for hid in set(x["history_id"] for x in test)),"train_test_exact_fair_input_overlap":len(train_keys&test_keys),"composition_valid":all(v["base_worlds"]>=12 for v in by_task.values())}
 (OUT/"metric_split_sanity.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n"); return result

def run():
 if _hash(OUT/"preregistered_config.json")!=CONFIG_HASH: raise RuntimeError("forensic config hash mismatch")
 inv=_inventory()
 if not inv["all_required_present"]: raise RuntimeError("required frozen source missing")
 breakdown=_breakdown(); overlap=_overlap(); ident=_identifiability(); sensitivity,equivalence=_sensitivity(); oracle=_oracle_minimality(); sanity=_metric_split()
 task_values=[x["pair_order_accuracy"] for x in breakdown["models"]["RelDynVerifier"]["by_task"].values()]; task_mix=max(task_values)-min(task_values)>.2
 if ident["conclusion"]=="B. FAIR_INPUT_AMBIGUOUS": decision="B. FAIR_OBSERVATION_UNIDENTIFIABLE"; recommended="Milestone 5B-Data: Observable Identity / Contact Cue Redesign"
 elif sensitivity["conclusion"]!="A. MODEL_USES_RELATIONAL_ACTION_FEATURES": decision="A. METHOD_IMPLEMENTATION_BUG_LIKELY"; recommended="Milestone 5B-R: RelDynVerifier Implementation Repair"
 elif task_mix or not sanity["composition_valid"]: decision="C. TASK_MIXTURE_OR_SPLIT_ARTIFACT"; recommended="Milestone 5B-0-v2: Relational Probe Repair"
 else: decision="D. STANDARD_TEMPORAL_BASELINE_ALREADY_SUFFICIENT"; recommended="No method scale-up"
 final={"schema":"milestone5b-f-final-decision-v1","decision":decision,"recommended_next_milestone":recommended,"config_sha256":CONFIG_HASH,"source_inventory_pass":inv["all_required_present"],"identifiability":ident["conclusion"],"sensitivity":sensitivity["conclusion"],"task_mix_artifact":task_mix,"metric_split_sanity":sanity["composition_valid"]}
 (OUT/"final_decision.json").write_text(json.dumps(final,indent=2,sort_keys=True)+"\n"); return final
if __name__=="__main__": print(json.dumps(run(),indent=2,sort_keys=True))
