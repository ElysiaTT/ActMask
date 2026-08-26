"""V3 compositional-OOD wrapper: fresh GPU labels plus partial-cue 3-token observations."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np
from actmask.data.milestone5b_data_v2_probe import TASKS,history,plans
from actmask.data.milestone4r_v3_preprobe import _run_family
from actmask.data.milestone4r_v2_preprobe import run as contact_run
ROOT=Path(__file__).resolve().parents[2];O=ROOT/'outputs/actmask/milestone5b_data_v3_compositional_ood';S=O/'state_probe'
def iid(w):return 'train' if w<39 else 'validation' if w<52 else 'test'
def ood(w):return 'ood_test' if w%4==3 else 'ood_train'
def run():
 S.mkdir(parents=True,exist_ok=True);log=S/'execution_logs';log.mkdir(exist_ok=True);cfg=hashlib.sha256((O/'preregistered_config.json').read_bytes()).hexdigest();P=[];C=[];AM=[];G=[];GM=[];Goal=[];A=[];rows=[];worlds=[];pairs=[];raw={}
 for ti,(name,task,cands,fn,seed) in enumerate(TASKS):
  newname=('CompositionalPartialCueIdentitySwap','CompositionalIntermittentContactTiming')[ti];r=_run_family(log,family=newname,task_id=task,candidates=cands,planner=fn,worlds=64,seed=seed) if ti==0 else contact_run(log/newname,worlds=64,seed=seed);raw[newname]=r;act=plans(task,cands,fn,seed);A.append(act);lookup={(x['world_id'],x['branch'],x['candidate_slot']):bool(x['success']) for x in r['records']}
  for w in range(64):
   factor={'target_color':'red','crossing_direction':'left_to_right' if w%2==0 else 'right_to_left','blocker_side':'left' if (w//2)%2==0 else 'right','contact_timing':'early' if (w//4)%2==0 else 'late','distractor_motion':'arc_a' if (w//8)%2==0 else 'arc_b','candidate_direction':'early_vs_late'}; hs=[]
   for b in (0,1):
    (p,c,am,g,gm),_=history(ti,w,b);h=len(P);hs.append(h);P.append(p);C.append(c);AM.append(am);G.append(g);GM.append(gm);Goal.append([1.,0.])
    for slot in range(10):rows.append({'task':newname,'world_id':w,'branch':b,'iid_split':iid(w),'ood_split':ood(w),'history_id':h,'candidate_slot':slot,'pair_id':f'{newname}:{w}:{slot}','success':lookup[(w,b,slot)],'factor_combination':factor,'label_source':'ManiSkill 3 GPU PhysX execution'})
   worlds.append({'task':newname,'world_id':w,'iid_split':iid(w),'ood_split':ood(w),'factor_combination':factor});pairs.append({'task':newname,'world_id':w,'history_ids':hs,'complete_c10':True})
 np.savez_compressed(S/'histories.npz',positions=np.asarray(P),appearance_cues=np.asarray(C),appearance_visibility=np.asarray(AM),gap_proxy=np.asarray(G),gap_visibility=np.asarray(GM),goal_cues=np.asarray(Goal),timestamps=np.arange(6,dtype=np.float32));np.save(S/'candidate_actions.npy',np.asarray(A));(S/'candidate_labels.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows));(S/'worlds.json').write_text(json.dumps(worlds));(S/'pairs.json').write_text(json.dumps(pairs));(S/'execution_logs.json').write_text(json.dumps(raw));manifest={'config_sha256':cfg,'worlds':128,'histories':256,'candidate_rows':2560,'candidate_executions':2560,'rendering':False,'simulator':'ManiSkill 3 GPU PhysX'};(S/'data_manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest))
if __name__=='__main__':run()
