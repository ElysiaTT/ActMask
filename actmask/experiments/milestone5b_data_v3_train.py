from __future__ import annotations
import json,sys,hashlib
from pathlib import Path
import numpy as np,torch
from torch import nn
from actmask.experiments.milestone5b_data_v2_baselines import M,G,T,met
R=Path(__file__).resolve().parents[2];O=R/'outputs/actmask/milestone5b_data_v3_compositional_ood';S=O/'state_probe'
def run():
 d=np.load(S/'histories.npz');p=d['positions'];c=d['appearance_cues'];am=d['appearance_visibility'];g=d['gap_proxy'];gm=d['gap_visibility'];go=d['goal_cues'];ac=np.load(S/'candidate_actions.npy');rs=[json.loads(x) for x in (S/'candidate_labels.jsonl').read_text().splitlines()];ti={'CompositionalPartialCueIdentitySwap':0,'CompositionalIntermittentContactTiming':1};x=[];a=[];y=[]
 for r in rs:
  h=r['history_id'];z=np.concatenate((p[h],c[h],am[h],g[h],gm[h]),-1);z=np.concatenate((z,np.broadcast_to(go[h],(6,3,2))),-1);x.append(z);a.append(ac[ti[r['task']],r['world_id'],r['candidate_slot']].ravel());y.append(float(r['success']))
 x=np.asarray(x,np.float32);a=np.asarray(a,np.float32);y=np.asarray(y,np.float32);fit=np.array([r['ood_split']=='ood_train' for r in rs]);out={}
 for name,kind in [('current_object_token_mlp','m'),('unordered_object_token_deepsets','m'),('ordered_object_token_gru','g'),('temporal_convolution','t')]:
  per={}
  for seed in (17,29,43):
   v=np.concatenate((x[:,-1].reshape(len(x),-1),x[:,0,0,-2:]),1) if name.startswith('current') else np.concatenate((x.mean((1,2)),x[:,0,0,-2:]),1) if name.startswith('unordered') else x.reshape(len(x),6,-1);m=v[fit].mean(tuple(range(v.ndim-1)),keepdims=True);s=v[fit].std(tuple(range(v.ndim-1)),keepdims=True);s[s<1e-6]=1;aa=a[fit].mean(0,keepdims=True);st=a[fit].std(0,keepdims=True);st[st<1e-6]=1;torch.manual_seed(seed);X=torch.tensor((v-m)/s,device='cuda');A=torch.tensor((a-aa)/st,device='cuda');Y=torch.tensor(y,device='cuda');idx=torch.tensor(np.where(fit)[0],device='cuda');model=(M(X.shape[-1],A.shape[-1]) if kind=='m' else G(X.shape[-1],A.shape[-1]) if kind=='g' else T(X.shape[-1],A.shape[-1])).cuda();op=torch.optim.AdamW(model.parameters(),lr=.005)
   for _ in range(30):loss=nn.functional.binary_cross_entropy_with_logits(model(X,A)[idx],Y[idx]);op.zero_grad();loss.backward();op.step()
   with torch.no_grad():sc=model(X,A).cpu().numpy();raw=[{**r,'score':float(sc[i]),'seed':seed,'model':name} for i,r in enumerate(rs)];q=O/'learned'/name/f'seed_{seed}';q.mkdir(parents=True,exist_ok=True);torch.save(model.state_dict(),q/'checkpoint.pt');
   for f,val in {'model_config.json':{'goal_cue_present':True},'preprocessing_config.json':{'goal_cue_present':True},'training_config.json':{'ood_train':True},'seed.txt':str(seed),'data_sha256.txt':'v3','command.txt':' '.join(sys.argv),'train_log.jsonl':''}.items():(q/f).write_text(val if isinstance(val,str) else json.dumps(val))
   (q/'raw_scores.jsonl').write_text(''.join(json.dumps(z)+'\n' for z in raw));iid=met([z for z in raw if z['iid_split']=='test']);ood=met([z for z in raw if z['ood_split']=='ood_test']);(q/'metrics.json').write_text(json.dumps({'iid':iid,'ood':ood}));per[seed]={'iid':iid,'ood':ood}
  out[name]={k:float(np.mean([per[s][k]['pair_order_accuracy'] for s in per])) for k in ('iid','ood')}
 (O/'baseline_report_iid.json').write_text(json.dumps(out,indent=2));(O/'baseline_report_ood.json').write_text(json.dumps(out,indent=2));print(out)
if __name__=='__main__':run()
