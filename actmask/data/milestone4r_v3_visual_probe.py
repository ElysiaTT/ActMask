"""Clean all-family 4R-v3 visual generation after all preprobes passed."""
from __future__ import annotations
import json
from pathlib import Path
import gymnasium as gym
import numpy as np
import torch
from mani_skill.utils import sapien_utils
from actmask.data import maniskill_state_tasks as _registered  # noqa: F401
from actmask.data.maniskill_pilot import CandidateSpec,candidate_specs
from actmask.data.milestone4r_robust_pilot import CAMERAS,CONDITIONS,TASKS,TASK_KEY,_appearance_transform,_camera_local_distractor_offset,_camera_schedule,_condition_schedule,_history_with_oracle_visibility,_mask,_reordered_pair
from actmask.data.milestone4v_visual_pilot import _candidate_rollout
from actmask.data.milestone4r_v3_preprobe import CUBE_C10,PHASE_C10,_cube_actions,_phase_actions
from actmask.data.milestone4r_v2_preprobe import C10 as PLACEMENT_C10,_planned_actions as _placement_actions

def _planned_rollout(env,decision,candidate,planner,branch,horizon):
    base=env.unwrapped; actions=planner(base,decision,candidate,horizon); base.set_state_dict(decision);base.scene._gpu_apply_all();base.scene._gpu_fetch_all();base._elapsed_steps.zero_();base.set_hidden_future_mode(float(branch));info=None
    for step in range(horizon): _,_,_,_,info=env.step(torch.as_tensor(actions[:,step],device=base.device))
    assert info is not None
    return actions[0],bool(info["success"][0].detach().cpu()),decision["tcp_proxy"][0,:3].detach().cpu().numpy().astype(np.float32)

def generate(output_dir,*,worlds_per_task=128,horizon=12,seed=57000):
    if worlds_per_task!=128: raise ValueError("v3 full protocol requires 128 worlds per family")
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True);summary={"schema":"milestone4r-v3-full-visual-probe-v1","worlds_per_task":128,"candidate_executions":0,"tasks":{}}
    for offset,(task,task_id) in enumerate(TASKS.items()):
      br=[];bd=[];rm=[];dm=[];vis=[];tsall=[];rows=[];labels=[];cal=[];assoc=[]; specs=list(candidate_specs(TASK_KEY[task]));cams=_camera_schedule(128);conds=_condition_schedule(128)
      for world in range(128):
       split,camera=cams[world];condition=conds[world];pose=sapien_utils.look_at(eye=CAMERAS[camera],target=[0,0,.04]);env=gym.make(task_id,obs_mode="sensor_data",num_envs=1,sim_backend="gpu",robust_distractor=True,robust_distractor_camera_offset=_camera_local_distractor_offset(camera),sensor_configs={"pose":pose,"width":128,"height":128})
       try:
        rgb,depth,target,distractor,decision=_history_with_oracle_visibility(env,seed+offset*10000+world,6);rgb,depth=_appearance_transform(rgb,task_offset=offset,world=world),depth.copy();pv=np.ones(depth.shape,dtype=np.uint8)
        if condition=="target_occlusion":
         pv[2:4]=np.where(target[2:4],0,pv[2:4]);rgb=np.where(pv[...,None]>0,rgb,0).astype(np.uint8);depth=np.where(pv>0,depth,0).astype(np.uint16);frames=[]
         for f in (2,3):
          d=int((distractor[f]&(pv[f]>0)).sum()); before=int(target[f].sum());
          if not d: raise RuntimeError("distractor absent")
          frames.append({"frame":f,"target_visible_before":before,"target_visible_after":int((target[f]&(pv[f]>0)).sum()),"distractor_visible_after":d,"natural_target_occlusion":before==0})
         assoc.append({"world_id":world,"camera":camera,"frames":frames})
        rmask,dmask,tstamp=_mask(world,condition)
        for branch,(x,y,z) in enumerate(((rgb,depth,pv),_reordered_pair(rgb,depth,pv))):
         ref=len(br);br.append(x.copy());bd.append(y.copy());vis.append(z.copy());rm.append(rmask);dm.append(dmask);tsall.append(tstamp)
         for slot in range(10):
          if task=="MovingCubeIntercept": actions,success,tcp=_planned_rollout(env,decision,CUBE_C10[slot],_cube_actions,branch,horizon)
          elif task=="FixedPhaseRotatingCaptureWindow": actions,success,tcp=_planned_rollout(env,decision,PHASE_C10[slot],_phase_actions,branch,horizon)
          elif task=="SignedMovingWindowPlacement": actions,success,tcp=_planned_rollout(env,decision,PLACEMENT_C10[slot],_placement_actions,branch,horizon)
          else: raise AssertionError(task)
          rows.append({"history_ref":ref,"world_id":world,"candidate_slot":slot,"split":split,"pair_group":f"{task}:{world}:{slot}","condition":condition,"candidate_actions":actions.tolist(),"tcp_state":tcp.tolist()});labels.append(success)
        p=env.unwrapped.get_sensor_params()["actmask_fixed_camera"];cal.append({"world_id":world,"camera":camera,"intrinsic":p["intrinsic_cv"][0].detach().cpu().numpy().tolist(),"extrinsic":p["extrinsic_cv"][0].detach().cpu().numpy().tolist()})
       finally: env.close()
      order=np.random.default_rng(9301+offset).permutation(len(rows));rows=[rows[int(i)] for i in order];labels=[labels[int(i)] for i in order]
      np.savez_compressed(out/f"{task}_bundles.npz",rgb=np.stack(br),depth_mm=np.stack(bd),rgb_mask=np.stack(rm),depth_mask=np.stack(dm),visibility=np.stack(vis),confidence=np.stack(vis),timestamps=np.stack(tsall));(out/f"{task}_candidates.jsonl").write_text("".join(json.dumps(r,sort_keys=True)+"\n" for r in rows));np.savez_compressed(out/f"{task}_labels.npz",success=np.asarray(labels,dtype=np.bool_));(out/f"{task}_camera_calibration.json").write_text(json.dumps(cal,indent=2,sort_keys=True)+"\n");(out/f"{task}_oracle_association_diagnostic.json").write_text(json.dumps({"schema":"milestone4r-v3-oracle-association-diagnostic-v1","generator_private":True,"target_occlusions":assoc},indent=2,sort_keys=True)+"\n");summary["candidate_executions"]+=len(rows);summary["tasks"][task]={"bundles":len(br),"candidates":len(rows),"conditions":sorted(set(r["condition"] for r in rows))}
    (out/"probe_summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n");return summary
if __name__=="__main__":
 project=Path(__file__).resolve().parents[2];print(json.dumps(generate(project/"outputs"/"actmask"/"milestone4r_v3_all_family_candidate_diversity"/"full_probe"),indent=2))
