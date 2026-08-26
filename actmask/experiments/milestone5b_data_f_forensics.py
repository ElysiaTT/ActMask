"""Read-only saturation forensic for the 5B-Data observable-cue probe."""
from __future__ import annotations
import hashlib,json,sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from torch import nn
from actmask.experiments.milestone5b_data_baselines import GRU,TCN,metrics

ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/"outputs"/"actmask"/"milestone5b_data_f_saturation_forensics"; SRC=ROOT/"outputs"/"actmask"/"milestone5b_data_observable_cues"; S=SRC/"state_probe"; HASH="02ecc9f500b64a64e2dbd32bfbaa0e33b265113108f6f5211108e867ce758e69";SEED=17
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows():return [json.loads(x) for x in (S/"candidate_labels.jsonl").read_text().splitlines() if x]
def raw_data():
 d=np.load(S/"histories.npz");p=d["positions"].astype(np.float32);c=d["appearance_cues"].astype(np.float32);k=d["contact_proxy"].astype(np.float32);g=d["goal_cues"].astype(np.float32);aall=np.load(S/"candidate_actions.npy").astype(np.float32);rs=rows();ti={"ObservableIdentitySwapIntercept":0,"ObservableRelationalContact":1};x=[];a=[]
 for r in rs:x.append(np.concatenate((p[r["history_id"]],c[r["history_id"]],k[r["history_id"]]),-1));a.append(aall[ti[r["task"]],r["world_id"],r["candidate_slot"]].reshape(-1))
 return np.asarray(x),np.asarray(a,np.float32),rs,p,c,k,g
def inventory():
 files=[S/x for x in ("data_manifest.json","worlds.json","histories.npz","object_states.npz","candidate_actions.npy","candidate_labels.jsonl","pairs.json")]+[SRC/x for x in ("baseline_report.json","raw_score_manifest.json","final_decision.json")]
 files += [SRC/"learned"/name/f"seed_{s}"/x for name in ("ordered_object_token_gru","temporal_convolution") for s in (17,29,43) for x in ("checkpoint.pt","model_config.json","preprocessing_config.json","raw_scores.jsonl")]
 result={"schema":"milestone5b-data-f-inventory-v1","items":[{"path":str(x.relative_to(ROOT)),"exists":x.is_file(),"sha256":sha(x) if x.is_file() else None,"bytes":x.stat().st_size if x.is_file() else None} for x in files]};result["all_required_present"]=all(x["exists"] for x in result["items"]);(OUT/"source_inventory.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n");return result
def score_path(name,seed=17):return SRC/"learned"/name/f"seed_{seed}"/"raw_scores.jsonl"
def raw(name):return [json.loads(x) for x in score_path(name).read_text().splitlines() if x]
def top1(rs):
 d=defaultdict(list)
 for r in rs:d[(r["task"],r["history_id"])].append(r)
 return float(np.mean([max(v,key=lambda z:z["score"])["success"] for v in d.values()]))
def breakdown():
 x,a,rs,p,c,k,g=raw_data();out={"schema":"milestone5b-data-f-saturation-breakdown-v1","models":{}}
 gap=np.max(k,axis=(1,2,3));q=np.quantile(gap,[.33,.67]); cue_colors={"target":"red_[1,0]","distractor":"blue_[0,1]"}
 for name in ("ordered_object_token_gru","temporal_convolution"):
  r=raw(name);m={"overall_test":metrics([z for z in r if z["split"]=="test"]),"by_task":{},"by_branch_top1":{},"by_candidate_template":{},"by_gap_bucket":{},"cue_color":cue_colors,"target_cue":"red_[1,0]","crossing_pattern":{"identity":"terminal_merge_marker_association","contact":"terminal_merge_gap_association"}}
  for task in sorted({z["task"] for z in r}):m["by_task"][task]=metrics([z for z in r if z["split"]=="test" and z["task"]==task])
  for b in (0,1):m["by_branch_top1"][str(b)]=top1([z for z in r if z["split"]=="test" and z["branch"]==b])
  for slot in range(10):
   v=[z for z in r if z["split"]=="test" and z["candidate_slot"]==slot];m["by_candidate_template"][str(slot)]={"mean_score":float(np.mean([z["score"] for z in v])),"success_rate":float(np.mean([z["success"] for z in v]))}
  for label,fn in (("low",lambda v:v<=q[0]),("mid",lambda v:(v>q[0])&(v<=q[1])),("high",lambda v:v>q[1])):
   v=[z for z in r if z["split"]=="test" and fn(gap[z["history_id"]])];m["by_gap_bucket"][label]={"top1":top1(v) if v else None,"count":len(v)}
  out["models"][name]=m
 (OUT/"saturation_breakdown.json").write_text(json.dumps(out,indent=2,sort_keys=True)+"\n");return out
def load_model(name):
 run=SRC/"learned"/name/"seed_17";pre=json.loads((run/"preprocessing_config.json").read_text());x,a,rs,*_=raw_data();x=x.reshape(len(x),6,-1);xm=np.asarray(pre["feature_mean"],np.float32);xs=np.asarray(pre["feature_std"],np.float32);am=np.asarray(pre["action_mean"],np.float32);ast=np.asarray(pre["action_std"],np.float32);x=(x-xm)/xs;a=(a-am)/ast;model=GRU(x.shape[-1],a.shape[-1]) if name=="ordered_object_token_gru" else TCN(x.shape[-1],a.shape[-1]);model.load_state_dict(torch.load(run/"checkpoint.pt",map_location="cpu",weights_only=True));model.eval();return model,x,a,rs,pre
def assess_scores(scores,rs):
 rawrows=[{**r,"score":float(scores[i])} for i,r in enumerate(rs)];return metrics([r for r in rawrows if r["split"]=="test"])["pair_order_accuracy"]
def cue_ablations():
 allout={"schema":"milestone5b-data-f-cue-ablation-v1","models":{}}
 for name in ("ordered_object_token_gru","temporal_convolution"):
  model,x,a,rs,pre=load_model(name)
  def infer(v):
   with torch.no_grad():return model(torch.tensor(v),torch.tensor(a)).numpy()
  base=infer(x);variants={}
  def add(key,v,available=True):
   score=infer(v) if available else base;variants[key]={"pair_order":assess_scores(score,rs),"mean_abs_score_change":float(np.mean(abs(score-base))),"input_available_to_saved_model":available}
  def tokens(v):return v.reshape(len(v),6,2,6)
  def flat(v):return v.reshape(len(v),6,12)
  z=tokens(x.copy());z[:,:,:,3:5]=0;add("remove_appearance",flat(z))
  z=tokens(x.copy());z[:,:,:,3:5]=z[:,:,:,3:5][:,:,:,::-1];add("swap_red_blue",flat(z))
  z=tokens(x.copy());z[:,1::2,:,3:5]=z[:,1::2,::-1,3:5];add("randomize_cue_to_token_association",flat(z))
  add("remove_target_goal",x,available=False) # target goal was absent from saved baseline feature map.
  z=tokens(x.copy());z[:,:,:,5]=0;add("remove_gap",flat(z))
  z=tokens(x.copy());z[:,:,:,5]=np.round(z[:,:,:,5]*4)/4;add("quantize_gap",flat(z))
  rng=np.random.default_rng(99);z=tokens(x.copy());z[:,:,:,5]+=rng.normal(0,.02,z[:,:,:,5].shape);add("noise_gap",flat(z))
  z=tokens(x.copy());z[:,-1,:,3:5]=0;add("hide_cue_final",flat(z))
  z=tokens(x.copy());z[:,2:4,:,3:5]=0;add("hide_cue_middle",flat(z))
  z=tokens(x.copy());z[:,1:,:,3:5]=0;add("cue_only_first_frame",flat(z))
  add("reverse_time",x[:,::-1].copy())
  z=tokens(x.copy())[:,:,::-1];add("consistent_token_swap",flat(z))
  z=tokens(x.copy());z[:,1::2]=z[:,1::2,::-1];add("inconsistent_token_swap",flat(z))
  allout["models"][name]={"baseline_pair_order":assess_scores(base,rs),"target_goal_in_checkpoint_input":False,"perturbations":variants}
 # A target cue absent from the learned feature map plus token/cue behavior is
 # an implementation artifact, not evidence that a fair target instruction was used.
 (OUT/"cue_ablation_diagnostics.json").write_text(json.dumps(allout,indent=2,sort_keys=True)+"\n");return allout
def split_stress():
 x,a,rs,p,c,k,g=raw_data();records=[]
 definitions={
  "held_out_cue_role":{"train":"branch==0","test":"branch==1","pair_order_valid":False,"reason":"counterfactual pair members are separated"},
  "held_out_trajectory_pattern":{"train":"identity task","test":"contact task","pair_order_valid":True,"reason":"equivalent to leave-one-task for this two-pattern dataset"},
  "held_out_relation_role":{"train":"branch==0","test":"branch==1","pair_order_valid":False,"reason":"counterfactual pair members are separated"},
  "held_out_gap_range":{"train":"world_id<32","test":"world_id>=32","pair_order_valid":True,"reason":"base-world split preserves pairs"},
  "held_out_action_direction":{"train":"slots 0..4","test":"slots 5..9","pair_order_valid":False,"reason":"C10 ranking groups lose candidate families"},
  "leave_one_task_mechanism_out":{"train":"identity task","test":"contact task","pair_order_valid":True,"reason":"base task split preserves complete C10 groups"}}
 for name,d in definitions.items():
  test=[]
  for r in rs:
   if name in {"held_out_trajectory_pattern","leave_one_task_mechanism_out"}:keep=r["task"]=="ObservableRelationalContact"
   elif name=="held_out_gap_range":keep=r["world_id"]>=32
   elif name in {"held_out_cue_role","held_out_relation_role"}:keep=r["branch"]==1
   else:keep=r["candidate_slot"]>=5
   if keep:test.append(r)
  d={**d,"test_rows":len(test),"test_base_worlds":len({r["world_id"] for r in test}),"requires_retraining_for_fair_ood_claim":True,"existing_iid_checkpoint_is_fair_ood_test":False};records.append({"split":name,**d})
 result={"schema":"milestone5b-data-f-split-stress-v1","diagnostic_splits":records,"tiny_diagnostic_retrains":_tiny_split_retrains(x,a,rs),"conclusion":"Existing checkpoints were trained on the IID default split, so they cannot substantiate held-out-composition claims. Tiny diagnostic GRUs are logged separately and are not a final method result."}
 (OUT/"split_stress_diagnostics.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n");return result
def _tiny_split_retrains(x,a,rs):
 """Two small allowed diagnostic GRUs on frozen data only (no new method)."""
 results={}
 definitions={"held_out_gap_range":(np.array([r["world_id"]<32 for r in rs]),np.array([r["world_id"]>=32 for r in rs])),"leave_one_task":(np.array([r["task"]=="ObservableIdentitySwapIntercept" for r in rs]),np.array([r["task"]=="ObservableRelationalContact" for r in rs]))}
 y=np.asarray([float(r["success"]) for r in rs],np.float32);base=x.reshape(len(x),6,-1)
 for name,(train,test) in definitions.items():
  mean=base[train].mean((0,1),keepdims=True);std=base[train].std((0,1),keepdims=True);std[std<1e-6]=1;am=a[train].mean(0,keepdims=True);ast=a[train].std(0,keepdims=True);ast[ast<1e-6]=1;vx=(base-mean)/std;va=(a-am)/ast
  torch.manual_seed(17);model=GRU(vx.shape[-1],va.shape[-1]);opt=torch.optim.AdamW(model.parameters(),lr=.005,weight_decay=1e-4);tx=torch.tensor(vx);ta=torch.tensor(va);ty=torch.tensor(y);ti=torch.tensor(np.where(train)[0]);logs=[]
  for epoch in range(10):
   loss=nn.functional.binary_cross_entropy_with_logits(model(tx,ta)[ti],ty[ti]);opt.zero_grad();loss.backward();opt.step();logs.append({"epoch":epoch+1,"loss":float(loss.detach())})
  with torch.no_grad():score=model(tx,ta).numpy()
  run=OUT/"diagnostic_retrains"/"ordered_gru"/name/"seed_17";run.mkdir(parents=True,exist_ok=True);torch.save(model.state_dict(),run/"checkpoint.pt");(run/"model_config.json").write_text(json.dumps({"architecture":"GRU","hidden":48,"diagnostic_only":True},indent=2)+"\n");(run/"preprocessing_config.json").write_text(json.dumps({"feature_mean":mean.tolist(),"feature_std":std.tolist(),"action_mean":am.tolist(),"action_std":ast.tolist()},sort_keys=True)+"\n");(run/"training_config.json").write_text(json.dumps({"seed":17,"epochs":10,"split":name,"diagnostic_only":True},indent=2)+"\n");(run/"seed.txt").write_text("17\n");(run/"data_sha256.txt").write_text(sha(S/"histories.npz")+"\n");(run/"command.txt").write_text(" ".join(sys.argv)+"\n");(run/"train_log.jsonl").write_text("".join(json.dumps(x)+"\n" for x in logs));rawrows=[{**r,"score":float(score[i]),"diagnostic_split":name,"model":"tiny_ordered_gru","seed":17} for i,r in enumerate(rs)];(run/"raw_scores.jsonl").write_text("".join(json.dumps(r,sort_keys=True)+"\n" for r in rawrows));met=metrics([r for i,r in enumerate(rawrows) if test[i]]);(run/"metrics.json").write_text(json.dumps(met,indent=2,sort_keys=True)+"\n");results[name]={"train_rows":int(train.sum()),"test_rows":int(test.sum()),"pair_order":met["pair_order_accuracy"],"artifact_path":str(run.relative_to(OUT))}
 return results
def recommendation(cue):
 result={"schema":"milestone5b-data-f-redesign-v1","minimal_repair":"Represent detections in anonymous, independently permuted order; include the target appearance goal as an explicit fair input; hide appearance during a crossing interval; reveal a continuous local gap only before contact, not as a terminal scalar mode; require a held-out cue-role/contact-combination split.","axes":["partial observable identity through crossing occlusion","intermittent geometric gap proxy","three-object distractor","compositional cue-role/contact split","two-stage target-track plus contact timing"],"must_audit":["goal cue present in all baseline feature maps","token-order permutation invariance","cue removal and goal removal sensitivity","held-out base-world composition without pair breakage","no direct final-frame proxy"],"basis":"Saved checkpoints omit target goal and retain stable red-first token slot, so 5B-Data saturation cannot support an otherwise fair benchmark claim."};(OUT/"redesign_recommendation.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n");return result
def run():
 if sha(OUT/"preregistered_config.json")!=HASH:raise RuntimeError("config hash mismatch")
 inv=inventory()
 if not inv["all_required_present"]:raise RuntimeError("missing frozen source")
 b=breakdown();cue=cue_ablations();splits=split_stress();rec=recommendation(cue)
 artifact=all(not item["target_goal_in_checkpoint_input"] for item in cue["models"].values())
 final={"schema":"milestone5b-data-f-final-decision-v1","decision":"D. METRIC_OR_IMPLEMENTATION_ARTIFACT" if artifact else "E. HUMAN_REVIEW_REQUIRED","config_sha256":HASH,"source_inventory_pass":inv["all_required_present"],"target_goal_absent_from_baseline_inputs":artifact,"recommended_next_milestone":"Milestone 5B-Data-v2: Partial Cue / No Direct Proxy Redesign","reason":"The saved standard baselines omit the configured target goal and use canonical red-first detection ordering; their saturation is not a valid fair baseline comparison."};(OUT/"final_decision.json").write_text(json.dumps(final,indent=2,sort_keys=True)+"\n");return final
if __name__=="__main__":print(json.dumps(run(),indent=2,sort_keys=True))
