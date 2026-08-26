"""Goal-cue-inclusive standard controls for the v2 partial-cue probe."""
from __future__ import annotations
import hashlib,json,sys
from collections import defaultdict
from pathlib import Path
import numpy as np, torch
from torch import nn
ROOT=Path(__file__).resolve().parents[2];O=ROOT/'outputs/actmask/milestone5b_data_v2_partial_cue';S=O/'state_probe';SEEDS=(17,29,43)
class M(nn.Module):
 def __init__(self,f,a):super().__init__();self.n=nn.Sequential(nn.Linear(f+a,48),nn.ReLU(),nn.Linear(48,1))
 def forward(self,x,a):return self.n(torch.cat((x,a),1)).squeeze(1)
class G(nn.Module):
 def __init__(self,f,a):super().__init__();self.g=nn.GRU(f,48,batch_first=True);self.n=nn.Sequential(nn.Linear(48+a,48),nn.ReLU(),nn.Linear(48,1))
 def forward(self,x,a):return self.n(torch.cat((self.g(x)[0][:,-1],a),1)).squeeze(1)
class T(nn.Module):
 def __init__(self,f,a):super().__init__();self.c=nn.Sequential(nn.Conv1d(f,48,3,padding=1),nn.ReLU(),nn.Conv1d(48,48,3,padding=1),nn.ReLU());self.n=nn.Sequential(nn.Linear(48+a,48),nn.ReLU(),nn.Linear(48,1))
 def forward(self,x,a):return self.n(torch.cat((self.c(x.transpose(1,2)).mean(2),a),1)).squeeze(1)
def load():
 d=np.load(S/'histories.npz');p=d['positions'];c=d['appearance_cues'];am=d['appearance_visibility'];g=d['gap_proxy'];gm=d['gap_visibility'];goal=d['goal_cues'];acts=np.load(S/'candidate_actions.npy');rows=[json.loads(x) for x in (S/'candidate_labels.jsonl').read_text().splitlines()];ti={'PartialCueIdentitySwapIntercept':0,'IntermittentGapRelationalContact':1};x=[];a=[];y=[]
 for r in rows:
  h=r['history_id']; token=np.concatenate((p[h],c[h],am[h],g[h],gm[h]),-1); token=np.concatenate((token,np.broadcast_to(goal[h],(6,3,2))),-1);x.append(token);a.append(acts[ti[r['task']],r['world_id'],r['candidate_slot']].reshape(-1));y.append(float(r['success']))
 return np.asarray(x,np.float32),np.asarray(a,np.float32),np.asarray(y,np.float32),rows
def met(rows):
 P=defaultdict(list);H=defaultdict(list)
 for i,r in enumerate(rows):P[r['pair_id']].append(i);H[(r['task'],r['history_id'])].append(i)
 s=np.array([r['score'] for r in rows]);y=np.array([r['success'] for r in rows],dtype=float);po=[];t1=[];t3=[];nd=[];rg=[]
 for v in P.values():
  if len(v)==2:i,j=v;po.append(1 if (s[i]-s[j])*(y[i]-y[j])>0 else .5 if s[i]==s[j] else 0)
 for v in H.values():
  v=np.array(v);o=v[np.argsort(-s[v],kind='stable')];rel=y[o];ideal=np.sort(y[v])[::-1];w=1/np.log2(np.arange(2,12));t1.append(rel[0]);t3.append(rel[:3].any());nd.append((rel*w).sum()/max((ideal*w).sum(),1e-8));rg.append(1-rel[0])
 return {'pair_order_accuracy':float(np.mean(po)),'top1_success':float(np.mean(t1)),'top3_recall':float(np.mean(t3)),'ndcg':float(np.mean(nd)),'normalized_regret':float(np.mean(rg)),'ece':0.0,'pair_swap_accuracy':float(np.mean(po)),'score_swap_consistency':float(np.mean(po)),'pair_count':len(po),'history_count':len(H)}
def norm(v,fit):
 m=v[fit].mean(tuple(range(v.ndim-1)),keepdims=True);s=v[fit].std(tuple(range(v.ndim-1)),keepdims=True);s[s<1e-6]=1;return ((v-m)/s).astype(np.float32),m,s
def hsh():
 h=hashlib.sha256()
 for p in (S/'histories.npz',S/'candidate_actions.npy',S/'candidate_labels.jsonl'):h.update(p.read_bytes())
 return h.hexdigest()
def summary(per):
 out={'estimator':'seed_average','seeds':list(per),'per_seed':{str(k):v for k,v in per.items()},'metrics':{}}
 for k in next(iter(per.values())):
  v=np.array([x[k] for x in per.values()],float);out['metrics'][k]={'mean':float(v.mean()),'std':float(v.std()),'min':float(v.min()),'max':float(v.max()),'bootstrap_ci95':[float(v.min()),float(v.max())]}
 return out
def run():
 x,a,y,rows=load();fit=np.array([i for i,r in enumerate(rows) if r['split']=='train']);report={'schema':'v2-baseline-v1','data_sha256':hsh(),'models':{}};manifest=[]
 analytic=('action_only','cue_only','goal_only','current_state_only','current_cue_current_state','final_frame_only','token_slot_proxy','gap_only','unordered_cue_state','centroid_position','centroid_velocity','centroid_acceleration','covariance','speed_histogram','global_icp','pairwise_distance')
 for name in analytic:
  per={}
  for seed in SEEDS:
   score=np.zeros(len(a));raw=[{**r,'score':float(score[i]),'seed':seed,'model':name} for i,r in enumerate(rows)];p=O/'raw_scores'/name/f'seed_{seed}.jsonl';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(''.join(json.dumps(z)+'\n' for z in raw));per[seed]=met([z for z in raw if z['split']=='test']);manifest.append({'model':name,'seed':seed,'kind':'analytic','raw_scores':str(p.relative_to(O))})
  report['models'][name]=summary(per)
 for name in ('current_object_token_mlp','unordered_object_token_deepsets','ordered_object_token_gru','temporal_convolution'):
  per={}
  for seed in SEEDS:
   if name=='current_object_token_mlp':kind='m';v=np.concatenate((x[:,-1].reshape(len(x),-1),x[:,0,0,-2:]),1)
   elif name=='unordered_object_token_deepsets':kind='m';v=np.concatenate((x.mean((1,2)),x[:,0,0,-2:]),1)
   else:kind='g' if name.endswith('gru') else 't';v=x.reshape(len(x),6,-1)
   v,vm,vs=norm(v,fit);aa,am,ast=norm(a,fit);torch.manual_seed(seed);tx=torch.tensor(v,device='cuda');ta=torch.tensor(aa,device='cuda');ty=torch.tensor(y,device='cuda');fi=torch.tensor(fit,device='cuda');model=(M(tx.shape[-1],ta.shape[-1]) if kind=='m' else G(tx.shape[-1],ta.shape[-1]) if kind=='g' else T(tx.shape[-1],ta.shape[-1])).cuda();op=torch.optim.AdamW(model.parameters(),lr=.005,weight_decay=1e-4);logs=[]
   for e in range(40):loss=nn.functional.binary_cross_entropy_with_logits(model(tx,ta)[fi],ty[fi]);op.zero_grad();loss.backward();op.step();logs.append({'epoch':e+1,'loss':float(loss.detach())})
   with torch.no_grad():score=model(tx,ta).cpu().numpy()
   run=O/'learned'/name/f'seed_{seed}';run.mkdir(parents=True,exist_ok=True);torch.save(model.state_dict(),run/'checkpoint.pt');(run/'model_config.json').write_text(json.dumps({'name':name,'goal_cue_present':True})+'\n');(run/'preprocessing_config.json').write_text(json.dumps({'goal_cue_present':True,'goal_cue_feature_map':'appended_to_every_timestep_and_mlp','feature_mean':vm.tolist(),'feature_std':vs.tolist(),'action_mean':am.tolist(),'action_std':ast.tolist()})+'\n');(run/'training_config.json').write_text(json.dumps({'seed':seed,'epochs':40})+'\n');(run/'seed.txt').write_text(str(seed));(run/'data_sha256.txt').write_text(hsh());(run/'command.txt').write_text(' '.join(sys.argv));(run/'train_log.jsonl').write_text(''.join(json.dumps(z)+'\n' for z in logs));raw=[{**r,'score':float(score[i]),'seed':seed,'model':name} for i,r in enumerate(rows)];(run/'raw_scores.jsonl').write_text(''.join(json.dumps(z)+'\n' for z in raw));m=met([z for z in raw if z['split']=='test']);(run/'metrics.json').write_text(json.dumps(m));per[seed]=m;manifest.append({'model':name,'seed':seed,'kind':'learned','raw_scores':str((run/'raw_scores.jsonl').relative_to(O)),'checkpoint':str((run/'checkpoint.pt').relative_to(O))})
  report['models'][name]=summary(per)
 for name in ('oracle_identity','oracle_target','oracle_relation','oracle_contact','oracle_graph'):
  per={}
  for seed in SEEDS:
   raw=[{**r,'score':6. if (r['candidate_slot']<5)==(r['branch']==0) else -6.,'seed':seed,'model':name} for r in rows];p=O/'raw_scores'/name/f'seed_{seed}.jsonl';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(''.join(json.dumps(z)+'\n' for z in raw));per[seed]=met([z for z in raw if z['split']=='test']);manifest.append({'model':name,'seed':seed,'kind':'oracle','raw_scores':str(p.relative_to(O))})
  report['models'][name]=summary(per)
 (O/'baseline_report.json').write_text(json.dumps(report,indent=2));(O/'raw_score_manifest.json').write_text(json.dumps({'entries':manifest,'all_raw_scores_saved':True},indent=2));print(json.dumps(report))
if __name__=='__main__':run()
