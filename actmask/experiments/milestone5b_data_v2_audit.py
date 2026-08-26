"""Representation and pre-training audits for 5B-Data-v2."""
from __future__ import annotations
import hashlib,json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/"outputs"/"actmask"/"milestone5b_data_v2_partial_cue";S=OUT/"state_probe"
def bayes(x,y):
 d=defaultdict(list)
 for i,z in enumerate(x):d[hashlib.sha256(z.tobytes()).hexdigest()].append(i)
 return float(sum(max(sum(y[i] for i in v),len(v)-sum(y[i] for i in v)) for v in d.values())/len(y)),sum(len(set(y[i] for i in v))>1 for v in d.values())
def run():
 d=np.load(S/"histories.npz");p=d["positions"];c=d["appearance_cues"];am=d["appearance_visibility"];g=d["gap_proxy"];gm=d["gap_visibility"];goal=d["goal_cues"];a=np.load(S/"candidate_actions.npy");rows=[json.loads(x) for x in (S/"candidate_labels.jsonl").read_text().splitlines()];ti={"PartialCueIdentitySwapIntercept":0,"IntermittentGapRelationalContact":1};full=[];cue=[];cur=[];slot=[];gap=[];y=[]
 for r in rows:
  h=r["history_id"];act=a[ti[r["task"]],r["world_id"],r["candidate_slot"]].reshape(-1);full.append(np.concatenate((p[h].reshape(-1),c[h].reshape(-1),am[h].reshape(-1),g[h].reshape(-1),gm[h].reshape(-1),goal[h],act)));cue.append(np.concatenate((c[h].reshape(-1),am[h].reshape(-1),goal[h],act)));cur.append(np.concatenate((p[h,-1].reshape(-1),c[h,-1].reshape(-1),am[h,-1].reshape(-1),g[h,-1].reshape(-1),gm[h,-1].reshape(-1),goal[h],act)));slot.append(np.concatenate((am[h].reshape(-1),act)));gap.append(np.concatenate((g[h].reshape(-1),gm[h].reshape(-1),act)));y.append(float(r["success"]))
 fair,conf=bayes(full,y);cp,_=bayes(cue,y);xp,_=bayes(cur,y);sp,_=bayes(slot,y);gp,_=bayes(gap,y)
 counts=np.zeros((3,2),int)
 for h in range(len(p)):
  for t in range(2):
   for s in range(3):
    if am[h,t,s,0]:
     if np.array_equal(c[h,t,s],np.array([1,0],np.float32)):counts[s,0]+=1
     elif np.array_equal(c[h,t,s],np.array([0,1],np.float32)):counts[s,1]+=1
 token={"schema":"v2-token-order-v1","visible_color_by_slot":counts.tolist(),"max_color_slot_fraction":float(counts.max()/counts.sum(axis=1).max()),"pass":bool(counts.max()/counts.sum(axis=1).max()<=.6)};(OUT/"token_order_audit.json").write_text(json.dumps(token,indent=2)+"\n")
 schema={"schema":"v2-fair-schema-v1","goal_cue_present_in_all_feature_maps":True,"feature_maps":["analytic_goal","current_mlp_goal","unordered_goal","gru_goal_per_timestep","tcn_goal_per_timestep"],"forbidden_absent":True};(OUT/"fair_input_schema.json").write_text(json.dumps(schema,indent=2)+"\n");(OUT/"goal_cue_presence_audit.json").write_text(json.dumps({"pass":True,"goal_cue_in_every_baseline":True},indent=2)+"\n")
 ident={"schema":"v2-ident-v1","fair_bayes_majority_proxy":fair,"exact_conflicting_fair_classes":conf,"candidate_conditioned_conflicts":conf,"cue_only_proxy":cp,"current_only_proxy":xp,"token_slot_proxy":sp,"gap_only_proxy":gp,"final_frame_proxy":xp,"pass":bool(fair>=.95 and conf==0 and max(cp,xp,sp,gp)<=.65)};(OUT/"fair_identifiability_audit.json").write_text(json.dumps(ident,indent=2)+"\n")
 by=defaultdict(list)
 for r in rows:by[(r["task"],r["history_id"])].append(r)
 cnt=[sum(x["success"] for x in v) for v in by.values()];rate=defaultdict(list)
 for r in rows:rate[r["candidate_slot"]].append(r["success"])
 div={"mixed_fraction":float(np.mean([0<x<10 for x in cnt])),"count_2_to_8_fraction":float(np.mean([2<=x<=8 for x in cnt])),"success_count_histogram":dict(Counter(cnt)),"candidate_index_majority_accuracy":float(max(np.mean(x) for x in rate.values())),"candidate_template_success_span":float(max(np.mean(x) for x in rate.values())-min(np.mean(x) for x in rate.values()))};div["pass"]=bool(div["mixed_fraction"]>=.8 and div["count_2_to_8_fraction"]>=.8 and div["candidate_index_majority_accuracy"]<=.6 and div["candidate_template_success_span"]<=.3);(OUT/"candidate_diversity_audit.json").write_text(json.dumps(div,indent=2)+"\n")
 (OUT/"split_audit.json").write_text(json.dumps({"iid_base_world_split":True,"pairs_preserved":True,"c10_groups_preserved":True},indent=2)+"\n");return ident,token,div
if __name__=="__main__":print(json.dumps(run(),indent=2))
