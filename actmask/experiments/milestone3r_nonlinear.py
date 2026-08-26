"""Strict matched nonlinear signed-pair generator for the 3R task."""
from __future__ import annotations
import json
from pathlib import Path
import gymnasium as gym
import numpy as np
import torch
from actmask.data import maniskill_3r_tasks  # register
from actmask.data.milestone3q_signed import _clone, _loop_history, _numpy

NAMES=("acceleration","jerk","piecewise","delayed","damping","collision_change","curved","regime_switch")

def _plan(base, mechanism:int, horizon:int=20):
    p=base.proxy.pose.p.clone(); base_pos=base.payload_base.clone(); result=[]
    for q in range(horizon):
        x=torch.full((base.num_envs,),float(q+1),device=base.device)
        vals=torch.stack((.0025*x*x,.00030*x**3,.018*torch.clamp(x-3,min=0),.026*torch.clamp(x-5,min=0),.18*(1-torch.exp(-.27*x)),torch.where(x<7,.018*x,.126+.008*(x-7)),.20*torch.sin(.07*x),torch.where(x<6,.008*x,.048+.032*(x-6))),1)[:,mechanism]
        goal=base_pos.clone(); goal[:,1]+=vals
        if mechanism==6: goal[:,0]+=.08*(1-torch.cos(.07*x))
        action=torch.clamp((goal-p)/base.proxy_step_size,-1,1); p+=action*base.proxy_step_size; result.append(_numpy(action))
    return np.stack(result,axis=1)

def _execute(env, base, action, sign):
    base.set_hidden_future_mode(sign); complete=torch.zeros(base.num_envs,dtype=torch.bool,device=base.device)
    for t in range(action.shape[1]):
        *_,info=env.step(torch.from_numpy(action[:,t]).to(base.device)); complete|=info['success'].bool()
    return complete.cpu().numpy()

def run(output_root:str|Path, worlds:int=16, seed:int=9201):
    out=Path(output_root);out.mkdir(parents=True,exist_ok=True); rows=[];histories=[];actions=[];labels=[]
    for mid,name in enumerate(NAMES):
        env=gym.make('ActMaskNonlinearSignedIntercept-v1',obs_mode='state',num_envs=worlds,sim_backend='gpu',render_backend='none')
        try:
            seeds=list(range(seed+mid*100,seed+mid*100+worlds));env.reset(seed=seeds);base=env.unwrapped;base.set_nonlinear_mechanism(mid);decision=_clone(base.get_state_dict());plan=_plan(base,mid)
            pair=[]
            for member,sign in enumerate((1,-1)):
                base.set_state_dict(decision);base.set_nonlinear_mechanism(mid);base.scene._gpu_apply_all();base.scene._gpu_fetch_all()
                history=_loop_history(base,decision,sign,"loop"); base.set_state_dict(decision);base.set_nonlinear_mechanism(mid);base.scene._gpu_apply_all();base.scene._gpu_fetch_all()
                success=_execute(env,base,plan,sign); histories.append(history);actions.append(plan);labels.append(success);pair.append(success)
                for local in range(worlds): rows.append(dict(mechanism=name,world_id=local,seed=seeds[local],signed_group=f"{name}:{local}",member=member,success=bool(success[local])))
            if not np.all(pair[0] & ~pair[1]): raise AssertionError(f"{name}: GPU execution did not flip every pair")
        finally:env.close()
    history=np.concatenate(histories);action=np.concatenate(actions);label=np.concatenate(labels)
    groups={}
    for i,r in enumerate(rows):groups.setdefault(r['signed_group'],[]).append(i)
    for members in groups.values():
        a,b=members
        if np.abs(history[a,-1]-history[b,-1]).max()>1e-6 or np.abs(action[a]-action[b]).max()>1e-6 or label[a]==label[b]:raise AssertionError('strict pair audit failed')
    np.savez_compressed(out/'nonlinear_model_inputs.npz',history=history,timestamps=np.tile(np.linspace(-.15,0,4,dtype=np.float32),(len(history),1)),visibility=np.ones((len(history),4,1),np.float32),observation_confidence=np.ones((len(history),4,1),np.float32),candidate_actions=action,nominal_action_timing=np.tile(np.arange(20,dtype=np.float32)/20,(len(history),1)),tcp_state=np.zeros((len(history),3),np.float32))
    np.savez_compressed(out/'nonlinear_labels.npz',success=label)
    (out/'nonlinear_metadata.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows),encoding='utf-8')
    summary=dict(mechanisms=list(NAMES),worlds_per_mechanism=worlds,examples=int(len(label)),groups=len(groups),success_rate=float(label.mean()),strict_audit='pass')
    (out/'nonlinear_summary.json').write_text(json.dumps(summary,indent=2)+'\n');return summary
