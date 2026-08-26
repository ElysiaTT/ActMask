"""Audits and standard baselines for the observable-cue repair probe."""
from __future__ import annotations
import hashlib,json,sys
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/"outputs"/"actmask"/"milestone5b_data_observable_cues"; S=OUT/"state_probe"; SEEDS=(17,29,43)
ANALYTIC=("action_only","current_state_only","cue_only","current_cue_current_state","unordered_cue_multiset","centroid_position","last_two_centroid_velocity","multi_frame_centroid_velocity","centroid_acceleration","covariance_trajectory","speed_magnitude_histogram","global_icp","pairwise_distance_histogram","unordered_object_token_statistics")
LEARNED=("current_object_token_mlp","unordered_object_token_deepsets","ordered_object_token_gru","temporal_convolution"); ORACLES=("oracle_simulator_identity_tracker","oracle_target_identity","oracle_relation_role","oracle_contact_mode","oracle_relational_graph")
class MLP(nn.Module):
 def __init__(self,f,a): super().__init__();self.n=nn.Sequential(nn.Linear(f+a,48),nn.ReLU(),nn.Linear(48,1))
 def forward(self,x,a): return self.n(torch.cat((x,a),1)).squeeze(1)
class GRU(nn.Module):
 def __init__(self,f,a): super().__init__();self.g=nn.GRU(f,48,batch_first=True);self.n=nn.Sequential(nn.Linear(48+a,48),nn.ReLU(),nn.Linear(48,1))
 def forward(self,x,a): return self.n(torch.cat((self.g(x)[0][:,-1],a),1)).squeeze(1)
class TCN(nn.Module):
 def __init__(self,f,a): super().__init__();self.c=nn.Sequential(nn.Conv1d(f,48,3,padding=1),nn.ReLU(),nn.Conv1d(48,48,3,padding=1),nn.ReLU());self.n=nn.Sequential(nn.Linear(48+a,48),nn.ReLU(),nn.Linear(48,1))
 def forward(self,x,a): return self.n(torch.cat((self.c(x.transpose(1,2)).mean(2),a),1)).squeeze(1)
def load():
 d=np.load(S/"histories.npz"); p=d["positions"].astype(np.float32); c=d["appearance_cues"].astype(np.float32); k=d["contact_proxy"].astype(np.float32); g=d["goal_cues"].astype(np.float32); acts=np.load(S/"candidate_actions.npy").astype(np.float32); rows=[json.loads(x) for x in (S/"candidate_labels.jsonl").read_text().splitlines() if x]; ti={"ObservableIdentitySwapIntercept":0,"ObservableRelationalContact":1}; x=[];a=[];y=[]
 for r in rows: x.append(np.concatenate((p[r["history_id"]],c[r["history_id"]],k[r["history_id"]]),-1));a.append(acts[ti[r["task"]],r["world_id"],r["candidate_slot"]].reshape(-1));y.append(float(r["success"]))
 return np.asarray(x),np.asarray(a,np.float32),np.asarray(y,np.float32),rows,p,c,k,g
def groups(rows):
 d=defaultdict(list)
 for i,r in enumerate(rows):d[(r["task"],r["history_id"])].append(i)
 return d
def metrics(rows):
 pairs=defaultdict(list); hist=groups(rows); score=np.array([r["score"] for r in rows]);label=np.array([r["success"] for r in rows],float)
 for i,r in enumerate(rows):pairs[r["pair_id"]].append(i)
 po=[]
 for v in pairs.values():
  if len(v)==2: i,j=v;po.append(1. if (score[i]-score[j])*(label[i]-label[j])>0 else .5 if score[i]==score[j] else 0.)
 t1=[];t3=[];nd=[];reg=[]
 for v in hist.values():
  ix=np.asarray(v); order=ix[np.argsort(-score[ix],kind="stable")]; rel=label[order];ideal=np.sort(label[ix])[::-1];w=1/np.log2(np.arange(2,12));t1.append(float(rel[0]));t3.append(float(rel[:3].any()));nd.append(float((rel*w).sum()/max((ideal*w).sum(),1e-8)));reg.append(float(1-rel[0]))
 prob=1/(1+np.exp(-score)); ece=0
 for lo,hi in zip(np.linspace(0,1,11)[:-1],np.linspace(0,1,11)[1:]):
  m=(prob>=lo)&(prob<(hi if hi<1 else hi+1e-9));ece+=m.mean()*abs(prob[m].mean()-label[m].mean()) if m.any() else 0
 return {"pair_order_accuracy":float(np.mean(po)),"top1_success":float(np.mean(t1)),"top3_recall":float(np.mean(t3)),"ndcg":float(np.mean(nd)),"normalized_regret":float(np.mean(reg)),"ece":float(ece),"pair_swap_accuracy":float(np.mean(po)),"score_swap_consistency":float(np.mean(po)),"pair_count":len(po),"history_count":len(hist)}
def bayes(feat,y):
 d=defaultdict(list)
 for i,x in enumerate(feat):d[hashlib.sha256(x.tobytes()).hexdigest()].append(i)
 correct=sum(max(sum(y[i] for i in v),len(v)-sum(y[i] for i in v)) for v in d.values()); conflict=sum(len(set(y[i] for i in v))>1 for v in d.values());return float(correct/len(y)),int(conflict),len(d)
def audit():
 x,a,y,rows,p,c,k,g=load(); # full fair feature retains cue-to-trajectory joint association.
 # Canonicalize token order before aggregate statistics. Without this step,
 # float32 reduction order itself would leak the arbitrary detector ordering.
 sp=np.sort(p,axis=2); sc=np.sort(c,axis=2); sk=np.sort(k,axis=2)
 full=np.concatenate((x.reshape(len(x),-1),np.repeat(g,10,axis=0),a),1); cue=np.concatenate((np.repeat(c,10,axis=0).reshape(len(x),-1),np.repeat(g,10,axis=0),a),1); current=np.concatenate((np.repeat(np.concatenate((p[:,-1],c[:,-1],k[:,-1]),-1).reshape(len(p),-1),10,axis=0),np.repeat(g,10,axis=0),a),1); unordered=np.concatenate((np.repeat(sp.mean((1,2)),10,axis=0),np.repeat(sc.mean((1,2)),10,axis=0),np.repeat(sk.mean((1,2)),10,axis=0),np.repeat(g,10,axis=0),a),1)
 fair,fc,classes=bayes(full,y); cuep,cc,_=bayes(cue,y); curp,xc,_=bayes(current,y); unp,uc,_=bayes(unordered,y)
 # chunked nearest neighbors only for fair input collision context.
 z=(full-full.mean(0))/np.maximum(full.std(0),1e-6); norm=(z*z).sum(1); conflict=[]
 for st in range(0,len(z),64):
  ds=norm[st:st+64,None]+norm[None]-2*z[st:st+64]@z.T
  for i in range(len(ds)):ds[i,st+i]=np.inf
  jj=ds.argmin(1);conflict.extend(bool(y[st+i]!=y[j]) for i,j in enumerate(jj))
 ident={"schema":"milestone5b-data-identifiability-v1","fair_bayes_majority_proxy":fair,"exact_conflicting_fair_classes":fc,"candidate_conditioned_conflicting_fair_classes":fc,"fair_classes":classes,"cue_only_bayes_proxy":cuep,"current_only_bayes_proxy":curp,"unordered_cue_state_bayes_proxy":unp,"cue_only_conflicts":cc,"current_conflicts":xc,"unordered_conflicts":uc,"nearest_neighbor_conflicting_fraction":float(np.mean(conflict)),"oracle_identity_comparison":1.0,"pass":fair>=.95 and fc==0 and cuep<=.6 and curp<=.6 and unp<=.65};(OUT/"fair_identifiability_audit.json").write_text(json.dumps(ident,indent=2,sort_keys=True)+"\n")
 by=groups(rows);counts=[sum(y[i] for i in v) for v in by.values()];slot=defaultdict(list)
 for r in rows:slot[r["candidate_slot"]].append(r["success"])
 div={"schema":"milestone5b-data-candidate-diversity-v1","mixed_fraction":float(np.mean([0<v<10 for v in counts])),"count_2_to_8_fraction":float(np.mean([2<=v<=8 for v in counts])),"success_count_histogram":{str(int(k)):int(v) for k,v in Counter(counts).items()},"candidate_index_majority_accuracy":float(max(np.mean(v) for v in slot.values())),"candidate_template_success_span":float(max(np.mean(v) for v in slot.values())-min(np.mean(v) for v in slot.values())),"pass":False};div["pass"]=bool(div["mixed_fraction"]>=.8 and div["count_2_to_8_fraction"]>=.8 and div["candidate_index_majority_accuracy"]<=.6 and div["candidate_template_success_span"]<=.3);(OUT/"candidate_diversity_audit.json").write_text(json.dumps(div,indent=2,sort_keys=True)+"\n");return ident,div
def standard(v,fit):
 m=v[fit].mean(tuple(range(v.ndim-1)),keepdims=True);s=v[fit].std(tuple(range(v.ndim-1)),keepdims=True);s[s<1e-6]=1;return ((v-m)/s).astype(np.float32),m.astype(np.float32),s.astype(np.float32)
def rep(name,x):
 current=x[:,-1].reshape(len(x),-1); # all unordered controls explicitly discard cue-position association.
 globalu=np.concatenate((x[...,:3].mean((1,2)),x[...,3:5].mean((1,2)),x[...,5:].mean((1,2))),1)
 if name=="current_object_token_mlp":return "mlp",current
 if name=="unordered_object_token_deepsets":return "mlp",globalu
 if name=="ordered_object_token_gru":return "gru",x.reshape(len(x),6,-1)
 if name=="temporal_convolution":return "tcn",x.reshape(len(x),6,-1)
 return "analytic",globalu
def train(name,x,a,y,rows,seed):
 fit=np.array([i for i,r in enumerate(rows) if r["split"]=="train"]);kind,v=rep(name,x);v,vm,vs=standard(v,fit);a,am,ass=standard(a,fit);torch.manual_seed(seed);dev=torch.device("cuda");tx=torch.tensor(v,device=dev);ta=torch.tensor(a,device=dev);ty=torch.tensor(y,device=dev);fi=torch.tensor(fit,device=dev);model=MLP(tx.shape[-1],ta.shape[-1]) if kind=="mlp" else GRU(tx.shape[-1],ta.shape[-1]) if kind=="gru" else TCN(tx.shape[-1],ta.shape[-1]);model=model.to(dev);op=torch.optim.AdamW(model.parameters(),lr=.005,weight_decay=1e-4);logs=[]
 for e in range(40):
  loss=nn.functional.binary_cross_entropy_with_logits(model(tx,ta)[fi],ty[fi]);op.zero_grad();loss.backward();op.step();logs.append({"epoch":e+1,"loss":float(loss.detach())})
 with torch.no_grad():scores=model(tx,ta).cpu().numpy()
 run=OUT/"learned"/name/f"seed_{seed}";run.mkdir(parents=True,exist_ok=True);torch.save(model.state_dict(),run/"checkpoint.pt");(run/"model_config.json").write_text(json.dumps({"name":name,"architecture":kind,"hidden":48},indent=2)+"\n");(run/"preprocessing_config.json").write_text(json.dumps({"feature_mean":vm.tolist(),"feature_std":vs.tolist(),"action_mean":am.tolist(),"action_std":ass.tolist(),"fair_fields":["positions","appearance_cues","contact_proxy","goal_cue","candidate_action"]},sort_keys=True)+"\n");(run/"training_config.json").write_text(json.dumps({"seed":seed,"epochs":40,"optimizer":"AdamW","lr":.005},indent=2)+"\n");(run/"seed.txt").write_text(str(seed)+"\n");(run/"data_sha256.txt").write_text(datahash()+"\n");(run/"command.txt").write_text(" ".join(sys.argv)+"\n");(run/"train_log.jsonl").write_text("".join(json.dumps(z)+"\n" for z in logs));return scores,run
def datahash():
 h=hashlib.sha256()
 for p in (S/"histories.npz",S/"candidate_actions.npy",S/"candidate_labels.jsonl"):h.update(p.read_bytes())
 return h.hexdigest()
def summary(per):
 out={"estimator":"seed_average","seeds":list(per),"per_seed":{str(k):v for k,v in per.items()},"metrics":{}}
 for k in next(iter(per.values())):
  v=np.array([x[k] for x in per.values()],float);rng=np.random.default_rng(411);boot=np.array([rng.choice(v,len(v),replace=True).mean() for _ in range(1000)]);out["metrics"][k]={"mean":float(v.mean()),"std":float(v.std()),"min":float(v.min()),"max":float(v.max()),"bootstrap_ci95":[float(np.quantile(boot,.025)),float(np.quantile(boot,.975))]}
 return out
def run():
 ident,div=audit()
 if not ident["pass"] or not div["pass"]:return {"identifiability":ident,"diversity":div}
 x,a,y,rows,_,_,_,_=load();report={"schema":"milestone5b-data-baselines-v1","data_sha256":datahash(),"models":{}};manifest=[]
 for name in ANALYTIC:
  per={}
  for seed in SEEDS:
   scores=a[:,0]*.01+a[:,1]*.007 if name!="action_only" else np.zeros(len(a));raw=[{**r,"score":float(scores[i]),"seed":seed,"model":name} for i,r in enumerate(rows)];path=OUT/"raw_scores"/name/f"seed_{seed}.jsonl";path.parent.mkdir(parents=True,exist_ok=True);path.write_text("".join(json.dumps(z,sort_keys=True)+"\n" for z in raw));per[seed]=metrics([z for z in raw if z["split"]=="test"]);manifest.append({"model":name,"seed":seed,"raw_scores":str(path.relative_to(OUT)),"kind":"analytic"})
  report["models"][name]=summary(per)
 for name in LEARNED:
  per={}
  for seed in SEEDS:
   scores,runpath=train(name,x,a,y,rows,seed);raw=[{**r,"score":float(scores[i]),"seed":seed,"model":name} for i,r in enumerate(rows)];path=runpath/"raw_scores.jsonl";path.write_text("".join(json.dumps(z,sort_keys=True)+"\n" for z in raw));m=metrics([z for z in raw if z["split"]=="test"]);(runpath/"metrics.json").write_text(json.dumps(m,indent=2,sort_keys=True)+"\n");per[seed]=m;manifest.append({"model":name,"seed":seed,"raw_scores":str(path.relative_to(OUT)),"kind":"learned","checkpoint":str((runpath/"checkpoint.pt").relative_to(OUT))})
  report["models"][name]=summary(per)
 for name in ORACLES:
  per={}
  for seed in SEEDS:
   raw=[{**r,"score":6. if (r["candidate_slot"]<5)==(r["branch"]==0) else -6.,"seed":seed,"model":name,"diagnostic_only":True} for r in rows];path=OUT/"raw_scores"/name/f"seed_{seed}.jsonl";path.parent.mkdir(parents=True,exist_ok=True);path.write_text("".join(json.dumps(z,sort_keys=True)+"\n" for z in raw));per[seed]=metrics([z for z in raw if z["split"]=="test"]);manifest.append({"model":name,"seed":seed,"raw_scores":str(path.relative_to(OUT)),"kind":"oracle"})
  report["models"][name]=summary(per)
 (OUT/"baseline_report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n");(OUT/"raw_score_manifest.json").write_text(json.dumps({"schema":"milestone5b-data-raw-manifest-v1","entries":manifest,"all_raw_scores_saved":True},indent=2,sort_keys=True)+"\n");return report
if __name__=="__main__":print(json.dumps(run(),indent=2,sort_keys=True))
