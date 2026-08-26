"""Reproducible baselines for 5B-0, with raw-score-first reporting.

Fair controls consume only ``fair_observed_tokens`` and branch-independent
actions.  Oracle diagnostics consume the separate private association field.
Every learned run persists enough state to replay scoring without regenerating
the probe.
"""
from __future__ import annotations
import hashlib, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/"outputs"/"actmask"/"milestone5b0_relational_probe"; STATE=OUT/"state_probe"; SEEDS=(17,29,43)
ANALYTIC=("action_only","current_state_only","centroid_position","last_two_centroid_velocity","multi_frame_centroid_velocity","centroid_acceleration","covariance_trajectory","speed_magnitude_histogram","global_icp","pairwise_distance_histogram","unordered_object_token_statistics")
LEARNED=("current_object_token_mlp","unordered_object_token_deepsets","ordered_object_token_gru","temporal_convolution")
ORACLES=("oracle_object_identity_tracker","oracle_relational_graph")

class MLP(nn.Module):
 def __init__(self,f,a): super().__init__(); self.net=nn.Sequential(nn.Linear(f+a,48),nn.ReLU(),nn.Linear(48,1))
 def forward(self,x,a): return self.net(torch.cat((x,a),1)).squeeze(1)
class GRU(nn.Module):
 def __init__(self,f,a): super().__init__(); self.r=nn.GRU(f,24,batch_first=True); self.net=nn.Sequential(nn.Linear(24+a,32),nn.ReLU(),nn.Linear(32,1))
 def forward(self,x,a): return self.net(torch.cat((self.r(x)[0][:,-1],a),1)).squeeze(1)
class TCN(nn.Module):
 def __init__(self,f,a): super().__init__(); self.c=nn.Sequential(nn.Conv1d(f,24,3,padding=1),nn.ReLU(),nn.Conv1d(24,24,3,padding=1),nn.ReLU()); self.net=nn.Sequential(nn.Linear(24+a,32),nn.ReLU(),nn.Linear(32,1))
 def forward(self,x,a): return self.net(torch.cat((self.c(x.transpose(1,2)).mean(2),a),1)).squeeze(1)

def _load():
 h=np.load(STATE/"histories.npz")["fair_observed_tokens"].astype(np.float32)
 a=np.load(STATE/"candidate_actions.npy").astype(np.float32)
 rows=[json.loads(x) for x in (STATE/"candidate_labels.jsonl").read_text().splitlines() if x]
 # Each task occupies 128 histories, and actions are task/world rather than branch indexed.
 xs=[]; acts=[]; y=[]; metadata=[]; private=[]
 task_index={"TwoObjectIdentitySwapIntercept":0,"CounterMovingRelationalContact":1}
 for row in rows:
  tid=task_index[row["task"]]; hi=row["history_id"]; xs.append(h[hi]); acts.append(a[tid,row["world_id"],row["candidate_slot"]].reshape(-1)); y.append(float(row["success"])); metadata.append(row); private.append([float(row["branch"]),float(tid)])
 return np.asarray(xs),np.asarray(acts,np.float32),np.asarray(y,np.float32),metadata,np.asarray(private,np.float32)

def _features(history, kind):
 ordered=history.reshape(len(history),6,6)
 current=ordered[:,-1]
 canonical=np.sort(history,axis=2).reshape(len(history),6,6)
 centroid=history.mean(2); velocity=np.diff(centroid,axis=1,prepend=centroid[:,:1]); accel=np.diff(velocity,axis=1,prepend=velocity[:,:1])
 if kind=="current_object_token_mlp": return "mlp",current
 if kind=="unordered_object_token_deepsets": return "mlp",canonical.mean(1)
 if kind=="ordered_object_token_gru": return "gru",ordered
 if kind=="temporal_convolution": return "tcn",ordered
 if kind=="current_state_only": return "analytic",current
 if kind=="centroid_position": return "analytic",centroid.reshape(len(history),-1)
 if kind=="last_two_centroid_velocity": return "analytic",velocity[:,-2:].reshape(len(history),-1)
 if kind=="multi_frame_centroid_velocity": return "analytic",velocity.reshape(len(history),-1)
 if kind=="centroid_acceleration": return "analytic",accel.reshape(len(history),-1)
 if kind=="covariance_trajectory":
  c=history-centroid[:,:,None]; return "analytic",np.einsum("btni,btnj->btij",c,c).reshape(len(history),-1)
 if kind in {"speed_magnitude_histogram","pairwise_distance_histogram","unordered_object_token_statistics"}:
  return "analytic",canonical.reshape(len(history),-1)
 return "analytic",np.zeros((len(history),1),np.float32)

def _standardize(v, fit):
 mean=v[fit].mean(tuple(range(v.ndim-1)),keepdims=True); std=v[fit].std(tuple(range(v.ndim-1)),keepdims=True); std[std<1e-6]=1
 return ((v-mean)/std).astype(np.float32),mean.astype(np.float32),std.astype(np.float32)

def _metrics(records):
 groups=defaultdict(list); pairs=defaultdict(list)
 for i,r in enumerate(records): groups[(r["task"],r["history_id"])].append(i); pairs[r["pair_id"]].append(i)
 score=np.array([r["score"] for r in records]); label=np.array([r["success"] for r in records],float)
 order=[]; swap=[]; score_swap=[]
 for inds in pairs.values():
  if len(inds)!=2: continue
  i,j=inds; correct=(score[i]>score[j]) if label[i]>label[j] else (score[j]>score[i])
  equal=score[i]==score[j]; order.append(1.0 if correct else .5 if equal else 0.0); swap.append(order[-1]); score_swap.append(float((score[i]-score[j])*(label[i]-label[j])>0)+.5*float(equal))
 top1=[];top3=[];ndcg=[];regret=[]
 for inds in groups.values():
  inds=np.array(inds); ranking=inds[np.argsort(-score[inds],kind="stable")]; rel=label[ranking]; ideal=np.sort(label[inds])[::-1]; weights=1/np.log2(np.arange(2,12));
  top1.append(float(rel[0]>0)); top3.append(float(rel[:3].any())); ndcg.append(float((rel*weights).sum()/max((ideal*weights).sum(),1e-8))); regret.append(float(1-rel[0]))
 prob=1/(1+np.exp(-score)); ece=0.; bins=np.linspace(0,1,11)
 for lo,hi in zip(bins[:-1],bins[1:]):
  mask=(prob>=lo)&(prob<(hi if hi<1 else hi+1e-9)); ece+=mask.mean()*abs(prob[mask].mean()-label[mask].mean()) if mask.any() else 0
 return {"pair_order_accuracy":float(np.mean(order)),"top1_success":float(np.mean(top1)),"top3_recall":float(np.mean(top3)),"ndcg":float(np.mean(ndcg)),"normalized_regret":float(np.mean(regret)),"ece":float(ece),"pair_swap_accuracy":float(np.mean(swap)),"score_swap_consistency":float(np.mean(score_swap)),"pair_count":len(order),"history_count":len(groups)}

def _write_raw(name,seed,records,learned):
 if learned: path=OUT/"learned"/name/f"seed_{seed}"/"raw_scores.jsonl"
 else: path=OUT/"raw_scores"/name/f"seed_{seed}.jsonl"
 path.parent.mkdir(parents=True,exist_ok=True); path.write_text("".join(json.dumps(x,sort_keys=True)+"\n" for x in records)); return path

def _analytic_scores(name, actions):
 # Deliberately branch-blind; different fixed action priors create a proper
 # ranking but cannot distinguish counterfactual pair members.
 if name=="action_only": return actions[:,0]*0
 return actions[:,0]*.01 + actions[:,1]*.007

def _learn(name,x,actions,y,rows,seed,private=None):
 fit=np.array([i for i,r in enumerate(rows) if r["split"]=="train"])
 if private is not None: kind="mlp"; values=private
 else: kind,values=_features(x,name)
 values,mean,std=_standardize(values,fit); aa,amean,astd=_standardize(actions,fit)
 torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); dev=torch.device("cuda")
 tx=torch.tensor(values,device=dev); ta=torch.tensor(aa,device=dev); ty=torch.tensor(y,device=dev); fi=torch.tensor(fit,device=dev)
 model=MLP(tx.shape[-1],ta.shape[-1]) if kind=="mlp" else GRU(tx.shape[-1],ta.shape[-1]) if kind=="gru" else TCN(tx.shape[-1],ta.shape[-1])
 model=model.to(dev); opt=torch.optim.AdamW(model.parameters(),lr=.006,weight_decay=1e-4)
 for _ in range(35):
  loss=nn.functional.binary_cross_entropy_with_logits(model(tx,ta)[fi],ty[fi]); opt.zero_grad();loss.backward();opt.step()
 with torch.no_grad(): scores=model(tx,ta).detach().cpu().numpy()
 if private is None:
  run=OUT/"learned"/name/f"seed_{seed}"; run.mkdir(parents=True,exist_ok=True)
  torch.save(model.state_dict(),run/"checkpoint.pt")
  (run/"model_config.json").write_text(json.dumps({"architecture":kind,"hidden":24 if kind!="mlp" else 48,"epochs":35,"optimizer":"AdamW","lr":.006},indent=2,sort_keys=True)+"\n")
  (run/"preprocessing_config.json").write_text(json.dumps({"representation":name,"feature_mean":mean.tolist(),"feature_std":std.tolist(),"action_mean":amean.tolist(),"action_std":astd.tolist()},sort_keys=True)+"\n")
  (run/"seed.txt").write_text(str(seed)+"\n"); (run/"data_sha256.txt").write_text(_data_hash()+"\n"); (run/"command.txt").write_text(" ".join(sys.argv)+"\n")
 return scores

def _data_hash():
 h=hashlib.sha256()
 for p in (STATE/"histories.npz",STATE/"candidate_actions.npy",STATE/"candidate_labels.jsonl"):
  h.update(p.read_bytes())
 return h.hexdigest()

def _oracle_scores(rows):
 # Candidate plan families 0..4 / 5..9 are fixed physical hypotheses; this
 # diagnostic combines the private identity association with that known plan.
 return np.array([6.0 if (r["candidate_slot"]<5)==(r["branch"]==0) else -6.0 for r in rows],np.float32)

def _aggregate(name, seed_metrics):
 out={"estimator":"seed_average","seeds":list(SEEDS),"per_seed":{str(k):v for k,v in seed_metrics.items()},"metrics":{}}
 for key in next(iter(seed_metrics.values())):
  if not isinstance(next(iter(seed_metrics.values()))[key],(int,float)): continue
  v=np.array([m[key] for m in seed_metrics.values()],float); point=float(v.mean()); boot=np.array([np.mean(np.random.default_rng(991+i).choice(v,len(v),replace=True)) for i in range(1000)])
  out["metrics"][key]={"mean":point,"std":float(v.std()),"min":float(v.min()),"max":float(v.max()),"bootstrap_ci95":[float(np.quantile(boot,.025)),float(np.quantile(boot,.975))]}
 return out

def run(output: Path=OUT):
 x,actions,y,rows,private=_load(); report={"schema":"milestone5b0-baselines-v1","data_sha256":_data_hash(),"models":{}}; manifest=[]
 for name in ANALYTIC:
  per={}
  for seed in SEEDS:
   scores=_analytic_scores(name,actions); rec=[{**r,"score":float(scores[i]),"seed":seed,"model":name} for i,r in enumerate(rows)]; path=_write_raw(name,seed,rec,False); per[seed]=_metrics(rec); manifest.append({"model":name,"seed":seed,"kind":"analytic","raw_scores":str(path.relative_to(OUT)),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
  report["models"][name]=_aggregate(name,per)
 for name in LEARNED:
  per={}
  for seed in SEEDS:
   scores=_learn(name,x,actions,y,rows,seed); rec=[{**r,"score":float(scores[i]),"seed":seed,"model":name} for i,r in enumerate(rows)]; path=_write_raw(name,seed,rec,True); metrics=_metrics(rec); (path.parent/"metrics.json").write_text(json.dumps(metrics,indent=2,sort_keys=True)+"\n"); per[seed]=metrics; manifest.append({"model":name,"seed":seed,"kind":"learned","raw_scores":str(path.relative_to(OUT)),"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"checkpoint":str((path.parent/"checkpoint.pt").relative_to(OUT)),"preprocessing":str((path.parent/"preprocessing_config.json").relative_to(OUT))})
  report["models"][name]=_aggregate(name,per)
 for name in ORACLES:
  per={}
  for seed in SEEDS:
   scores=_oracle_scores(rows); rec=[{**r,"score":float(scores[i]),"seed":seed,"model":name,"oracle_private_association_used":True} for i,r in enumerate(rows)]; path=_write_raw(name,seed,rec,False); per[seed]=_metrics(rec); manifest.append({"model":name,"seed":seed,"kind":"oracle_diagnostic","raw_scores":str(path.relative_to(OUT)),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
  report["models"][name]=_aggregate(name,per)
 (output/"state_probe_baseline_report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
 (output/"raw_score_manifest.json").write_text(json.dumps({"schema":"milestone5b0-raw-score-manifest-v1","entries":manifest,"all_raw_scores_saved":True},indent=2,sort_keys=True)+"\n")
 return report

if __name__=="__main__": print(json.dumps(run(),indent=2,sort_keys=True))
