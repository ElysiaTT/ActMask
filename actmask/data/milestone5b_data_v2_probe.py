"""Anonymous-order partial-cue GPU-PhysX state probe for 5B-Data-v2."""
from __future__ import annotations
import hashlib, itertools, json
from pathlib import Path
import gymnasium as gym
import numpy as np
import torch
from actmask.data import maniskill_state_tasks as _registered  # noqa: F401
from actmask.data.milestone4r_v2_preprobe import C10 as CONTACT, _planned_actions as contact_plan, run as contact_run
from actmask.data.milestone4r_v3_preprobe import CUBE_C10, _cube_actions, _run_family

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"outputs"/"actmask"/"milestone5b_data_v2_partial_cue"
S=OUT/"state_probe"
TASKS=(("PartialCueIdentitySwapIntercept","ActMaskMovingCubeIntercept-v1",CUBE_C10,_cube_actions,41000),("IntermittentGapRelationalContact","ActMaskMovingContainerPlacement-v1",CONTACT,contact_plan,42000))
PERMS=tuple(itertools.permutations(range(3)))
RED=np.array([1,0],np.float32); BLUE=np.array([0,1],np.float32); GRAY=np.array([.5,.5],np.float32)

def split(w): return "train" if w<39 else "validation" if w<52 else "test"
def clone(x): return {k:v.clone() for k,v in x.items()}
def plans(task,candidates,fn,seed):
 e=gym.make(task,obs_mode="state",num_envs=64,sim_backend="gpu",render_backend="none")
 try:
  e.reset(seed=list(range(seed,seed+64)));
  base=e.unwrapped; zero=torch.zeros((64,3),device=base.device)
  for _ in range(5): e.step(zero)
  state=clone(base.get_state_dict())
  return np.stack([fn(base,state,c,12) for c in candidates],1).astype(np.float32)
 finally: e.close()
def paths(kind,w):
 u=np.linspace(-1.,0.,6,dtype=np.float32); centre=np.array([(w%8-3.5)*.012,(w//8-3.5)*.009,.08],np.float32); scale=.065+.002*(w%5)
 if kind==0: delta=np.stack((scale*u,.035*np.sin(np.pi*(u+1)),np.zeros_like(u)),1)
 else: delta=np.stack((.045*np.sin(np.pi*(u+1)),scale*u,.012*np.sin(np.pi*(u+1))),1)
 a,b=centre+delta,centre-delta
 d=centre+np.stack((.028*np.cos(np.pi*(u+1)),.028*np.sin(np.pi*(u+1)),np.zeros_like(u)),1)
 return a.astype(np.float32),b.astype(np.float32),d.astype(np.float32)
def history(kind,w,branch):
 a,b,d=paths(kind,w); red,blue=(a,b) if branch==0 else (b,a)
 pos=np.stack((red,blue,d),1); cue=np.stack((np.tile(RED,(6,1)),np.tile(BLUE,(6,1)),np.tile(GRAY,(6,1))),1)
 appmask=np.zeros((6,3,1),np.float32); appmask[:2]=1; cue[2:]=0
 gap=np.zeros((6,3,1),np.float32); gapmask=np.zeros((6,3,1),np.float32)
 if kind==1:
  shared=np.linalg.norm(a[:2]-a[-1],axis=1);gap[:2,:,0]=shared[:,None]; gapmask[:2]=1
 # Terminal merge removes a current-frame answer; all early permutations are
 # anonymous and balanced through world/branch/time offsets.
 pos[-1,0]=pos[-1,1]=(a[-1]+b[-1])/2
 latent=np.stack((np.tile(RED,(6,1)),np.tile(BLUE,(6,1)),np.tile(GRAY,(6,1))),1)
 out=[]; latent_order=[]
 for t in range(6):
  perm=PERMS[(w+t+kind)%6]; out.append((pos[t,list(perm)],cue[t,list(perm)],appmask[t,list(perm)],gap[t,list(perm)],gapmask[t,list(perm)]));latent_order.append(latent[t,list(perm)])
 return tuple(np.asarray([x[i] for x in out],np.float32) for i in range(5)),np.asarray(latent_order,np.float32)
def moments(pos):
 order=np.empty_like(pos)
 for t,x in enumerate(pos): order[t]=x[np.lexsort((x[:,2],x[:,1],x[:,0]))]
 c=order.mean(1);v=np.diff(c,axis=0,prepend=c[:1]);acc=np.diff(v,axis=0,prepend=v[:1]);z=order-c[:,None]
 return {"centroid":c,"centroid_velocity":v,"centroid_acceleration":acc,"covariance":np.einsum("tni,tnj->tij",z,z)/3,"pairwise_distance_multiset":np.sort(np.stack((np.linalg.norm(order[:,0]-order[:,1],axis=1),np.linalg.norm(order[:,0]-order[:,2],axis=1),np.linalg.norm(order[:,1]-order[:,2],axis=1)),1),1),"speed_magnitude_multiset":np.sort(np.linalg.norm(np.diff(order,axis=0,prepend=order[:1]),axis=2),1),"unordered_state_multiset":order}
def run():
 S.mkdir(parents=True,exist_ok=True); log=S/"raw_candidate_execution_logs";log.mkdir(exist_ok=True);cfg=hashlib.sha256((OUT/"preregistered_config.json").read_bytes()).hexdigest(); allp=[];allc=[];allam=[];allg=[];allgm=[];allgoal=[];alllatent=[];allact=[];labels=[];worlds=[];pairs=[];audits=[];raw={}
 for ti,(name,task,candidates,fn,seed) in enumerate(TASKS):
  payload=_run_family(log,family=name,task_id=task,candidates=candidates,planner=fn,worlds=64,seed=seed) if ti==0 else contact_run(log/name,worlds=64,seed=seed)
  raw[name]=payload; action=plans(task,candidates,fn,seed);allact.append(action);lookup={(r["world_id"],r["branch"],r["candidate_slot"]):bool(r["success"]) for r in payload["records"]}
  for w in range(64):
   (p0,c0,am0,g0,gm0),l0=history(ti,w,0);(p1,c1,am1,g1,gm1),l1=history(ti,w,1);m0,m1=moments(p0),moments(p1)
   err={k:float(np.max(abs(m0[k]-m1[k]))) for k in m0}
   err["candidate_actions"]=0.;err["cue_multiset"]=float(np.max(np.abs(np.sort(c0,axis=1)-np.sort(c1,axis=1))));err["goal_cue"]=0.
   err["visibility_schedule"]=float(np.max(np.abs(np.sort(am0,axis=1)-np.sort(am1,axis=1)))+np.max(np.abs(np.sort(gm0,axis=1)-np.sort(gm1,axis=1))))
   audits.append({"task":name,"world_id":w,"max_abs_error":err,"association_differs":not np.array_equal(p0[:2],p1[:2]),"pass":all(v==0 for v in err.values())});hids=[];worlds.append({"task":name,"world_id":w,"split":split(w),"goal_cue":RED.tolist(),"rendering":False,"simulator":"ManiSkill 3 GPU PhysX"})
   for b,values in enumerate(((p0,c0,am0,g0,gm0,l0),(p1,c1,am1,g1,gm1,l1))):
    p,c,am,gap,gm,latent=values;hid=len(allp);hids.append(hid);allp.append(p);allc.append(c);allam.append(am);allg.append(gap);allgm.append(gm);allgoal.append(RED);alllatent.append(latent)
    for slot in range(10):labels.append({"task":name,"world_id":w,"branch":b,"split":split(w),"history_id":hid,"candidate_slot":slot,"pair_id":f"{name}:{w}:{slot}","success":lookup[(w,b,slot)],"label_source":"ManiSkill 3 GPU PhysX execution"})
   pairs.append({"task":name,"world_id":w,"history_ids":hids,"candidate_actions_identical":True,"matching_pass":audits[-1]["pass"]})
 np.savez_compressed(S/"histories.npz",positions=np.asarray(allp),appearance_cues=np.asarray(allc),appearance_visibility=np.asarray(allam),gap_proxy=np.asarray(allg),gap_visibility=np.asarray(allgm),goal_cues=np.asarray(allgoal),timestamps=np.arange(6,dtype=np.float32));np.savez_compressed(S/"object_states.npz",current_positions=np.asarray(allp)[:,-1],current_cues=np.asarray(allc)[:,-1],current_gap=np.asarray(allg)[:,-1]);np.save(S/"candidate_actions.npy",np.asarray(allact));(S/"candidate_labels.jsonl").write_text("".join(json.dumps(x,sort_keys=True)+"\n" for x in labels));(S/"worlds.json").write_text(json.dumps(worlds,indent=2,sort_keys=True)+"\n");(S/"pairs.json").write_text(json.dumps(pairs,indent=2,sort_keys=True)+"\n");(S/"raw_candidate_execution_logs.json").write_text(json.dumps(raw,indent=2,sort_keys=True)+"\n")
 match={"schema":"milestone5b-data-v2-moment-cue-order-v1","pairs":audits,"all_pass":all(x["pass"] for x in audits)};(OUT/"moment_cue_order_matching_audit.json").write_text(json.dumps(match,indent=2,sort_keys=True)+"\n");manifest={"schema":"milestone5b-data-v2-state-probe-v1","config_sha256":cfg,"worlds":128,"histories":256,"candidate_rows":2560,"candidate_executions":2560,"rendering":False,"simulator":"ManiSkill 3 GPU PhysX"};(S/"data_manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n");return manifest
if __name__=="__main__":print(json.dumps(run(),indent=2,sort_keys=True))
