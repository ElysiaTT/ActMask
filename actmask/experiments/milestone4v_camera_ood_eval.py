"""Clean-train / preregistered-camera-OOD-test evaluation."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from actmask.data.milestone4v_visual_pilot import TASKS
from actmask.experiments.milestone4v_visual_baselines import _features,_rows,_train,_pair_order

def _pool(rgb,depth):
 x=np.concatenate((rgb.astype(np.float32)/255.,depth.astype(np.float32)[...,None]/1000.),-1)
 return x.reshape(len(x),6,16,8,16,8,4).mean((3,5)).reshape(len(x),6,-1)
def run(root):
 root=Path(root); out={"schema":"milestone4v-camera-ood-eval-v1","tasks":{}}
 for task in TASKS:
  rows=_rows(root/f"{task}_candidates.jsonl"); refs=np.asarray([r["history_ref"] for r in rows]); acts=np.asarray([r["candidate_actions"] for r in rows],np.float32).reshape(len(rows),-1); y=np.load(root/f"{task}_labels.npz")["success"].astype(np.float32)
  train=np.asarray([i for i,r in enumerate(rows) if r["split"]=="train"]); test=np.asarray([i for i,r in enumerate(rows) if r["split"]=="test"]); report={}
  clean=_features(root,task)
  for camera in ("held_viewpoint","moderate_perturbation"):
   a=np.load(root/f"{task}_{camera}_test_histories.npz"); ood=_pool(a["rgb"],a["depth_mm"]); remap={int(k):v for k,v in zip(a["source_history_ref"],ood)}; histories=clean.copy()
   for ref,value in remap.items(): histories[ref]=value
   vals=[]
   for seed in (17,29,43):
    score=_train("ordered_gru",histories[refs],acts,y,train,test,seed); vals.append(_pair_order(y[test],score,[rows[i] for i in test])[0])
   report[camera]=float(np.mean(vals))
  out["tasks"][task]=report
 (root/"camera_ood_report.json").write_text(json.dumps(out,indent=2,sort_keys=True)+"\n");return out
if __name__=="__main__":
 p=Path(__file__).resolve().parents[2]/"outputs"/"actmask"/"milestone4v_visual_pilot"/"visual_pilot";print(json.dumps(run(p),indent=2,sort_keys=True))
