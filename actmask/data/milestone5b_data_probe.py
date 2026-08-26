"""Fresh observable-cue, no-render GPU-PhysX relational probe generator."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import gymnasium as gym
import numpy as np
import torch
from actmask.data import maniskill_state_tasks as _registered # noqa: F401
from actmask.data.milestone4r_v2_preprobe import C10 as CONTACT_C10,_planned_actions as contact_actions,run as contact_preprobe
from actmask.data.milestone4r_v3_preprobe import CUBE_C10,_cube_actions,_run_family

ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/"outputs"/"actmask"/"milestone5b_data_observable_cues"; STATE=OUT/"state_probe"
TASKS=(("ObservableIdentitySwapIntercept","ActMaskMovingCubeIntercept-v1",CUBE_C10,_cube_actions,31000),("ObservableRelationalContact","ActMaskMovingContainerPlacement-v1",CONTACT_C10,contact_actions,32000))
RED=np.array([1.,0.],np.float32); BLUE=np.array([0.,1.],np.float32)
def clone(s): return {k:v.clone() for k,v in s.items()}
def plans(task,candidates,planner,seed):
 env=gym.make(task,obs_mode="state",num_envs=64,sim_backend="gpu",render_backend="none")
 try:
  env.reset(seed=list(range(seed,seed+64))); base=env.unwrapped; z=torch.zeros((64,3),device=base.device)
  for _ in range(5): env.step(z)
  state=clone(base.get_state_dict()); return np.stack([planner(base,state,c,12) for c in candidates],1).astype(np.float32)
 finally: env.close()
def split(w): return "train" if w<39 else "validation" if w<52 else "test"
def trajectory(kind,w):
 u=np.linspace(-1.,0.,6,dtype=np.float32); centre=np.array([(w%8-3.5)*.012,(w//8-3.5)*.009,.08],np.float32); scale=.065+.002*(w%5)
 if kind==0: d=np.stack((scale*u,.035*np.sin(np.pi*(u+1)),np.zeros_like(u)),1)
 else: d=np.stack((.045*np.sin(np.pi*(u+1)),scale*u,.012*np.sin(np.pi*(u+1))),1)
 return centre+d,centre-d
def fair_history(kind,w,branch):
 a,b=trajectory(kind,w); # Output detections are canonical by persistent marker: red then blue.
 red,blue=(a,b) if branch==0 else (b,a)
 pos=np.stack((red,blue),1).astype(np.float32)
 cues=np.stack((np.tile(RED,(6,1)),np.tile(BLUE,(6,1))),1).astype(np.float32)
 # The terminal merge makes current positions/cues exactly identical across branches.
 pos[-1]=np.stack(((a[-1]+b[-1])/2,(a[-1]+b[-1])/2))
 if kind==0: contact=np.zeros((6,2,1),np.float32)
 else:
  # Observable geometry-only gap proxy, coupled to the visible trajectory but
  # not to hidden simulator contact mode. It vanishes at the terminal merge.
  contact=np.stack((np.linalg.norm(red-red[-1],axis=1),np.linalg.norm(blue-blue[-1],axis=1)),1)[...,None].astype(np.float32)
 return pos,cues,contact
def moments(pos):
 s=np.sort(pos,axis=1); cen=s.mean(1); vel=np.diff(cen,axis=0,prepend=cen[:1]); acc=np.diff(vel,axis=0,prepend=vel[:1]); c=s-cen[:,None]; cov=np.einsum("tni,tnj->tij",c,c)/2; dist=np.linalg.norm(s[:,0]-s[:,1],axis=1,keepdims=True); speed=np.sort(np.linalg.norm(np.diff(s,axis=0,prepend=s[:1]),axis=2),1)
 return {"current_unordered_state":s[-1],"centroid":cen,"centroid_velocity":vel,"centroid_acceleration":acc,"covariance":cov,"pairwise_distance_multiset":dist,"speed_magnitude_multiset":speed,"unordered_state_multiset":s}
def run():
 STATE.mkdir(parents=True,exist_ok=True); logdir=STATE/"raw_candidate_execution_logs"; logdir.mkdir(exist_ok=True); config=hashlib.sha256((OUT/"preregistered_config.json").read_bytes()).hexdigest(); allpos=[];allcue=[];allcontact=[];allgoal=[];allactions=[];labels=[];worlds=[];pairs=[];audits=[];raw={}
 for ti,(name,task,candidates,planner,seed) in enumerate(TASKS):
  payload=_run_family(logdir,family=name,task_id=task,candidates=candidates,planner=planner,worlds=64,seed=seed) if ti==0 else contact_preprobe(logdir/name,worlds=64,seed=seed)
  raw[name]=payload; act=plans(task,candidates,planner,seed); allactions.append(act); lookup={(r["world_id"],r["branch"],r["candidate_slot"]):bool(r["success"]) for r in payload["records"]}
  for w in range(64):
   branch_ids=[]; p0,c0,k0=fair_history(ti,w,0); p1,c1,k1=fair_history(ti,w,1); m0,m1=moments(p0),moments(p1); error={x:float(np.max(abs(m0[x]-m1[x]))) for x in m0}; error["candidate_actions"]=0.; error["cue_multiset"]=float(np.max(abs(np.sort(c0,axis=1)-np.sort(c1,axis=1)))); error["goal_cue"]=0.
   audits.append({"task":name,"world_id":w,"max_abs_error":error,"cue_to_trajectory_association_differs":not np.array_equal(p0[:5,0],p1[:5,0]),"pass":all(v==0 for v in error.values())})
   worlds.append({"task":name,"world_id":w,"split":split(w),"goal_cue":RED.tolist(),"rendering":False,"simulator":"ManiSkill 3 GPU PhysX"})
   for b,(pos,cue,contact) in enumerate(((p0,c0,k0),(p1,c1,k1))):
    hid=len(allpos); branch_ids.append(hid); allpos.append(pos);allcue.append(cue);allcontact.append(contact);allgoal.append(RED)
    for slot in range(10): labels.append({"task":name,"world_id":w,"branch":b,"split":split(w),"history_id":hid,"candidate_slot":slot,"pair_id":f"{name}:{w}:{slot}","success":lookup[(w,b,slot)],"label_source":"ManiSkill 3 GPU PhysX execution"})
   pairs.append({"task":name,"world_id":w,"history_ids":branch_ids,"candidate_actions_identical":True,"moment_cue_match_pass":audits[-1]["pass"]})
 np.savez_compressed(STATE/"histories.npz",positions=np.asarray(allpos),appearance_cues=np.asarray(allcue),contact_proxy=np.asarray(allcontact),goal_cues=np.asarray(allgoal),timestamps=np.arange(6,dtype=np.float32))
 np.savez_compressed(STATE/"object_states.npz",current_positions=np.asarray(allpos)[:,-1],current_cues=np.asarray(allcue)[:,-1],current_contact_proxy=np.asarray(allcontact)[:,-1])
 np.save(STATE/"candidate_actions.npy",np.asarray(allactions)); (STATE/"candidate_labels.jsonl").write_text("".join(json.dumps(x,sort_keys=True)+"\n" for x in labels)); (STATE/"worlds.json").write_text(json.dumps(worlds,indent=2,sort_keys=True)+"\n");(STATE/"pairs.json").write_text(json.dumps(pairs,indent=2,sort_keys=True)+"\n");(STATE/"raw_candidate_execution_logs.json").write_text(json.dumps(raw,indent=2,sort_keys=True)+"\n")
 audit={"schema":"milestone5b-data-moment-cue-audit-v1","pairs":audits,"all_pass":all(x["pass"] for x in audits)}; (OUT/"moment_cue_matching_audit.json").write_text(json.dumps(audit,indent=2,sort_keys=True)+"\n")
 manifest={"schema":"milestone5b-data-state-probe-v1","config_sha256":config,"tasks":[x[0] for x in TASKS],"worlds":128,"histories":256,"candidate_rows":2560,"candidate_executions":2560,"rendering":False,"simulator":"ManiSkill 3 GPU PhysX"};(STATE/"data_manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n");return manifest
if __name__=="__main__": print(json.dumps(run(),indent=2,sort_keys=True))
