"""Re-render only 4V held-out test histories under preregistered cameras."""
from __future__ import annotations
import json
from pathlib import Path
import gymnasium as gym
import numpy as np
from mani_skill.utils import sapien_utils
from actmask.data.milestone4v_visual_pilot import TASKS, _history

CAMERAS={"held_viewpoint":([-0.55,-0.55,.5],[0,0,.04]),"moderate_perturbation":([.65,-.43,.56],[.02,0,.04])}

def run(root):
 root=Path(root); result={"schema":"milestone4v-camera-ood-v1","cameras":{}}
 for name,(eye,target) in CAMERAS.items():
  report={}
  for offset,(task,task_id) in enumerate(TASKS.items()):
   pose=sapien_utils.look_at(eye=eye,target=target)
   env=gym.make(task_id,obs_mode="sensor_data",num_envs=1,sim_backend="gpu",sensor_configs={"pose":pose,"width":128,"height":128})
   rgb=[]; depth=[]; refs=[]
   try:
    for world in range(128):
     if world % 5 != 4: continue
     for branch in (0,1):
      a,b,_=_history(env,4400+offset*10000+world,branch,6); rgb.append(a); depth.append(b); refs.append(world*2+branch)
   finally: env.close()
   np.savez_compressed(root/f"{task}_{name}_test_histories.npz",source_history_ref=np.asarray(refs),rgb=np.stack(rgb),depth_mm=np.stack(depth))
   report[task]={"histories":len(refs),"source_history_refs":refs}
  result["cameras"][name]=report
 (root/"camera_ood_manifest.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n");return result
if __name__=="__main__":
 p=Path(__file__).resolve().parents[2]/"outputs"/"actmask"/"milestone4v_visual_pilot"/"visual_pilot"; print(json.dumps(run(p),indent=2,sort_keys=True))
