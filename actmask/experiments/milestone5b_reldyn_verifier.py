"""Fair, action-conditioned relational dynamics verifier for frozen 5B-0.

The model never reads task, world, branch, candidate-slot, labels, or oracle
fields as inputs.  Those fields are used only to select the frozen split and
to compute supervised losses/metrics after scores have been produced.
"""
from __future__ import annotations
import hashlib, json, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from torch import nn

from actmask.experiments.milestone5b0_baselines import _metrics

ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/"outputs"/"actmask"/"milestone5b_relational_verifier"; SOURCE=ROOT/"outputs"/"actmask"/"milestone5b0_relational_probe"; STATE=SOURCE/"state_probe"; SEEDS=(17,29,43)
SOURCE_HASH="ba9ae19a2854ee796296c715c0f7b2326c59c8e2511f2312a370ed1c4ebad36a"; CONFIG_HASH="02714137ef31f447ab39d865d04290b347dd557cbdaf0b74822f752297b78b0d"

def _source_hash():
 h=hashlib.sha256()
 for name in ("histories.npz","candidate_actions.npy","candidate_labels.jsonl"): h.update((STATE/name).read_bytes())
 return h.hexdigest()

def _load():
 if _source_hash()!=SOURCE_HASH: raise RuntimeError("frozen 5B-0 source hash mismatch")
 history=np.load(STATE/"histories.npz")["fair_observed_tokens"].astype(np.float32)
 actions=np.load(STATE/"candidate_actions.npy").astype(np.float32)
 rows=[json.loads(line) for line in (STATE/"candidate_labels.jsonl").read_text().splitlines() if line]
 task_index={"TwoObjectIdentitySwapIntercept":0,"CounterMovingRelationalContact":1}
 x=[]; a=[]; y=[]
 for r in rows:
  x.append(history[r["history_id"]]); a.append(actions[task_index[r["task"]],r["world_id"],r["candidate_slot"]].reshape(-1)); y.append(float(r["success"]))
 return np.asarray(x),np.asarray(a,np.float32),np.asarray(y,np.float32),rows

def _standardize(v, fit):
 mean=v[fit].mean(tuple(range(v.ndim-1)),keepdims=True); std=v[fit].std(tuple(range(v.ndim-1)),keepdims=True); std[std<1e-6]=1
 return ((v-mean)/std).astype(np.float32),mean.astype(np.float32),std.astype(np.float32)

class RelDynVerifier(nn.Module):
 def __init__(self, *, pairwise=True, action_in_relation=True, temporal=True):
  super().__init__(); self.pairwise=pairwise; self.action_in_relation=action_in_relation; self.temporal=temporal
  self.object=nn.Sequential(nn.Linear(6,32),nn.ReLU(),nn.Linear(32,32),nn.ReLU())
  self.action=nn.Sequential(nn.Linear(36,32),nn.ReLU(),nn.Linear(32,32),nn.ReLU())
  if pairwise: self.relation=nn.Sequential(nn.Linear(103 if action_in_relation else 71,48),nn.ReLU(),nn.Linear(48,48),nn.ReLU())
  else: self.relation=nn.Sequential(nn.Linear(32,48),nn.ReLU(),nn.Linear(48,48),nn.ReLU())
  self.memory=nn.GRU(48,48,batch_first=True) if temporal else None
  self.head=nn.Sequential(nn.Linear(80,48),nn.ReLU(),nn.Linear(48,1))
 def forward(self, position, action):
  # Fair velocity is finite differenced from the input history only.
  velocity=torch.diff(position,dim=1,prepend=position[:,:1]); tokens=torch.cat((position,velocity),-1); z=self.object(tokens); ae=self.action(action)
  if self.pairwise:
   # Directed 0->1 and 1->0 pairs, pooled symmetrically after temporal encoding.
   values=[]
   for i,j in ((0,1),(1,0)):
    relp=position[:,:,i]-position[:,:,j]; relv=velocity[:,:,i]-velocity[:,:,j]; dist=torch.linalg.vector_norm(relp,dim=-1,keepdim=True)
    q=torch.cat((z[:,:,i],z[:,:,j],relp,relv,dist),-1)
    if self.action_in_relation: q=torch.cat((q,ae[:,None].expand(-1,q.shape[1],-1)),-1)
    values.append(self.relation(q))
   q=torch.stack(values,1) # B, directed pair, T, feature
   if self.memory is not None:
    b,p,t,f=q.shape; q=self.memory(q.reshape(b*p,t,f))[0][:,-1].reshape(b,p,f)
   else: q=q[:,:,-1]
   relational=q.mean(1)
  else:
   q=self.relation(z.mean(2))
   relational=self.memory(q)[0][:,-1] if self.memory is not None else q[:,-1]
  return self.head(torch.cat((relational,ae),-1)).squeeze(-1)

def _groups(rows, split):
 result=defaultdict(list)
 for i,r in enumerate(rows):
  if r["split"]==split: result[(r["task"],r["history_id"])].append(i)
 return list(result.values())

def _train_variant(name, seed, *, pairwise=True, action_in_relation=True, temporal=True, augment=True, epochs=50):
 x,a,y,rows=_load(); fit=np.array([i for i,r in enumerate(rows) if r["split"]=="train"]); groups=_groups(rows,"train")
 x,xmean,xstd=_standardize(x,fit); a,amean,astd=_standardize(a,fit)
 torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); rng=torch.Generator(device="cuda"); rng.manual_seed(seed+991)
 dev=torch.device("cuda"); tx=torch.tensor(x,device=dev); ta=torch.tensor(a,device=dev); ty=torch.tensor(y,device=dev); fi=torch.tensor(fit,device=dev)
 model=RelDynVerifier(pairwise=pairwise,action_in_relation=action_in_relation,temporal=temporal).to(dev); opt=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=1e-4); logs=[]; start=time.time()
 for epoch in range(epochs):
  inp=tx
  if augment:
   swap=(torch.rand((len(tx),),generator=rng,device=dev)<.5); inp=tx.clone(); inp[swap]=torch.flip(inp[swap],dims=(2,))
  logits=model(inp,ta); bce=nn.functional.binary_cross_entropy_with_logits(logits[fi],ty[fi]); listwise=[]
  for inds in groups:
   ii=torch.tensor(inds,device=dev); positive=ty[ii]>.5; listwise.append(torch.logsumexp(logits[ii],0)-torch.logsumexp(logits[ii][positive],0))
  rank=torch.stack(listwise).mean()
  # Input-order invariance is enforced on an explicit full-token swap.
  swapped=tx[:,:,torch.tensor([1,0],device=dev)]; consistency=nn.functional.mse_loss(logits[fi],model(swapped,ta)[fi]) if augment else logits.new_zeros(())
  loss=bce+.4*rank+.1*consistency; opt.zero_grad(); loss.backward(); opt.step()
  logs.append({"epoch":epoch+1,"loss":float(loss.detach()),"bce":float(bce.detach()),"ranking":float(rank.detach()),"permutation_consistency":float(consistency.detach())})
 with torch.no_grad(): scores=model(tx,ta).detach().cpu().numpy()
 run=OUT/("learned" if name=="RelDynVerifier" else "ablations")/name/f"seed_{seed}"; run.mkdir(parents=True,exist_ok=True)
 model_config={"name":name,"pairwise_relations":pairwise,"action_in_relation":action_in_relation,"temporal_memory":temporal,"permutation_augmentation":augment,"object_hidden":32,"action_hidden":32,"relation_hidden":48,"temporal_hidden":48}
 preprocess={"history_mean":xmean.tolist(),"history_std":xstd.tolist(),"action_mean":amean.tolist(),"action_std":astd.tolist(),"fair_input_fields":["object_positions","finite_difference_velocity","anonymous_token_slots","candidate_action"],"forbidden_input_fields_verified":["simulator_object_identity","target_id","branch_id","mechanism_id","future_state","candidate_index","oracle_fields"]}
 training={"seed":seed,"epochs":epochs,"optimizer":"AdamW","learning_rate":.003,"weight_decay":.0001,"loss_weights":{"bce":1.0,"listwise":.4,"permutation_consistency":.1},"wall_seconds":time.time()-start,"device":"cuda"}
 torch.save(model.state_dict(),run/"checkpoint.pt"); (run/"model_config.json").write_text(json.dumps(model_config,indent=2,sort_keys=True)+"\n"); (run/"preprocessing_config.json").write_text(json.dumps(preprocess,indent=2,sort_keys=True)+"\n"); (run/"training_config.json").write_text(json.dumps(training,indent=2,sort_keys=True)+"\n"); (run/"seed.txt").write_text(f"{seed}\n"); (run/"data_sha256.txt").write_text(SOURCE_HASH+"\n"); (run/"command.txt").write_text(" ".join(sys.argv)+"\n"); (run/"train_log.jsonl").write_text("".join(json.dumps(x,sort_keys=True)+"\n" for x in logs))
 raw=[{**r,"score":float(scores[i]),"seed":seed,"model":name} for i,r in enumerate(rows)]; (run/"raw_scores.jsonl").write_text("".join(json.dumps(r,sort_keys=True)+"\n" for r in raw))
 test=[r for r in raw if r["split"]=="test"]; metrics=_metrics(test); (run/"metrics.json").write_text(json.dumps(metrics,indent=2,sort_keys=True)+"\n")
 return run,metrics,training

def _summary(per_seed):
 result={"estimator":"seed_average","seeds":list(per_seed),"per_seed":{str(k):v for k,v in per_seed.items()},"metrics":{}}
 for key in next(iter(per_seed.values())):
  if not isinstance(next(iter(per_seed.values()))[key],(float,int)): continue
  values=np.asarray([x[key] for x in per_seed.values()],float); rng=np.random.default_rng(590); boot=np.asarray([rng.choice(values,len(values),replace=True).mean() for _ in range(1000)])
  result["metrics"][key]={"mean":float(values.mean()),"std":float(values.std()),"min":float(values.min()),"max":float(values.max()),"bootstrap_ci95":[float(np.quantile(boot,.025)),float(np.quantile(boot,.975))]}
 return result

def _frozen_baselines():
 names=json.loads((SOURCE/"state_probe_baseline_report.json").read_text())["models"].keys(); result={}
 for name in names:
  per={}
  for seed in SEEDS:
   raw_path=(SOURCE/"learned"/name/f"seed_{seed}"/"raw_scores.jsonl") if (SOURCE/"learned"/name).exists() else SOURCE/"raw_scores"/name/f"seed_{seed}.jsonl"
   rows=[json.loads(x) for x in raw_path.read_text().splitlines() if x]; per[seed]=_metrics([r for r in rows if r["split"]=="test"])
  result[name]=_summary(per)
 return result

def run():
 if hashlib.sha256((OUT/"preregistered_config.json").read_bytes()).hexdigest()!=CONFIG_HASH: raise RuntimeError("5B config hash mismatch")
 full={}; times=[]; manifest=[]
 for seed in SEEDS:
  path,metrics,training=_train_variant("RelDynVerifier",seed); full[seed]=metrics; times.append(training["wall_seconds"]); manifest.append({"kind":"full","seed":seed,"path":str(path.relative_to(OUT)),"required_files":sorted(x.name for x in path.iterdir())})
 method={"schema":"milestone5b-method-report-v1","config_sha256":CONFIG_HASH,"source_data_sha256":SOURCE_HASH,"model":"RelDynVerifier","metrics":_summary(full),"training_wall_seconds":sum(times)}
 (OUT/"method_report.json").write_text(json.dumps(method,indent=2,sort_keys=True)+"\n")
 ablation_specs={"no_pairwise_relations":dict(pairwise=False),"no_action_conditioning":dict(action_in_relation=False),"no_temporal_memory":dict(temporal=False),"no_permutation_augmentation":dict(augment=False)}; ablations={}
 for name,kwargs in ablation_specs.items():
  path,metrics,training=_train_variant(name,17,**kwargs); ablations[name]={"seed":17,"metrics":metrics,"training_wall_seconds":training["wall_seconds"]}; manifest.append({"kind":"ablation","seed":17,"name":name,"path":str(path.relative_to(OUT)),"required_files":sorted(x.name for x in path.iterdir())})
 (OUT/"ablation_report.json").write_text(json.dumps({"schema":"milestone5b-ablation-v1","full_seed17":full[17],"ablations":ablations},indent=2,sort_keys=True)+"\n")
 baseline=_frozen_baselines(); (OUT/"comparison_to_5b0.json").write_text(json.dumps({"schema":"milestone5b-comparison-v1","source_report_sha256":hashlib.sha256((SOURCE/"state_probe_baseline_report.json").read_bytes()).hexdigest(),"test_split_baselines_from_frozen_raw_scores":baseline,"method":method["metrics"]},indent=2,sort_keys=True)+"\n")
 (OUT/"raw_score_manifest.json").write_text(json.dumps({"schema":"milestone5b-raw-manifest-v1","source_data_sha256":SOURCE_HASH,"entries":manifest,"all_artifacts_saved":True},indent=2,sort_keys=True)+"\n")
 return method,baseline,ablations

if __name__=="__main__": print(json.dumps(run()[0],indent=2,sort_keys=True))
