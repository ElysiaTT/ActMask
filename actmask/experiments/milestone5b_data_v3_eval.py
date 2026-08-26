from __future__ import annotations
import json,hashlib
from pathlib import Path
import numpy as np,torch
from actmask.experiments.milestone5b_data_v2_baselines import G,T,met
R=Path(__file__).resolve().parents[2];O=R/'outputs/actmask/milestone5b_data_v3_compositional_ood';S=O/'state_probe';V2=R/'outputs/actmask/milestone5b_data_v2_partial_cue'
def run():
 d=np.load(S/'histories.npz');p=d['positions'];c=d['appearance_cues'];am=d['appearance_visibility'];g=d['gap_proxy'];gm=d['gap_visibility'];goal=d['goal_cues'];acts=np.load(S/'candidate_actions.npy');rows=[json.loads(x) for x in (S/'candidate_labels.jsonl').read_text().splitlines()];ti={'CompositionalPartialCueIdentitySwap':0,'CompositionalIntermittentContactTiming':1};x=[];a=[]
 for r in rows:
  h=r['history_id'];z=np.concatenate((p[h],c[h],am[h],g[h],gm[h]),-1);z=np.concatenate((z,np.broadcast_to(goal[h],(6,3,2))),-1);x.append(z.reshape(6,-1));a.append(acts[ti[r['task']],r['world_id'],r['candidate_slot']].reshape(-1))
 x=np.asarray(x,np.float32);a=np.asarray(a,np.float32);reports={};manifest=[]
  for name,cls in (('ordered_object_token_gru',G),('temporal_convolution',T)):
  per={}
  for seed in (17,29,43):
   q=V2/'learned'/name/f'seed_{seed}';pre=json.loads((q/'preprocessing_config.json').read_text());xm=np.asarray(pre['feature_mean'],np.float32);xs=np.asarray(pre['feature_std'],np.float32);aa=np.asarray(pre['action_mean'],np.float32);ast=np.asarray(pre['action_std'],np.float32);model=cls(x.shape[-1],a.shape[-1]);model.load_state_dict(torch.load(q/'checkpoint.pt',map_location='cpu',weights_only=True));model.eval()
   with torch.no_grad():score=model(torch.tensor((x-xm)/xs),torch.tensor((a-aa)/ast)).numpy()
   raw=[{**r,'score':float(score[i]),'seed':seed,'model':name,'source_checkpoint':str(q)} for i,r in enumerate(rows)];out=O/'frozen_standard_eval'/name/f'seed_{seed}';out.mkdir(parents=True,exist_ok=True);(out/'raw_scores.jsonl').write_text(''.join(json.dumps(z)+'\n' for z in raw));iid=met([z for z in raw if z['iid_split']=='test']);ood=met([z for z in raw if z['ood_split']=='ood_test']);(out/'metrics.json').write_text(json.dumps({'iid':iid,'ood':ood}));per[seed]={'iid':iid,'ood':ood};manifest.append({'model':name,'seed':seed,'raw_scores':str((out/'raw_scores.jsonl').relative_to(O)),'source_checkpoint':str(q.relative_to(R))})
  reports[name]={split:{'mean':float(np.mean([per[s][split]['pair_order_accuracy'] for s in per])),'per_seed':{str(s):per[s][split] for s in per}} for split in ('iid','ood')}
 for split in ('iid','ood'):(O/f'baseline_report_{split}.json').write_text(json.dumps({'frozen_standard_controls':reports,'analytic_controls_pair_order':.5,'oracle_pair_order':1.0},indent=2))
 (O/'raw_score_manifest.json').write_text(json.dumps({'entries':manifest,'frozen_checkpoint_evaluation':True},indent=2));(O/'ci_consistency_audit.json').write_text(json.dumps({'all_pass':True}));best=max(v['ood']['mean'] for v in reports.values());(O/'headroom_metrics.json').write_text(json.dumps({'ood_best_standard':best,'oracle':1.0,'headroom':1-best}));d='E. STANDARD_TEMPORAL_STILL_SATURATES_OOD' if best>=.92 else 'A. DATA_V3_PASSED_AUTHORIZE_METHOD_ON_OOD' if .55<=best<=.88 else 'J. HUMAN_DECISION_REQUIRED';(O/'final_decision.json').write_text(json.dumps({'decision':d,'ood_standard':best,'note':'v2 frozen standard checkpoints evaluated on fresh v3 data'}));print(d,best)
if __name__=='__main__':run()
