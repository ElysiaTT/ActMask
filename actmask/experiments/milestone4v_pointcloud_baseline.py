"""Fair depth-backprojected point-cloud temporal GRU control."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from actmask.data.milestone4v_visual_pilot import TASKS
from actmask.experiments.milestone4v_visual_baselines import _rows,_train,_pair_order

def _pc(root,task):
 d=np.load(root/f"{task}_histories.npz")["depth_mm"].astype(np.float32)/1000.; n,t,h,w=d.shape
 u,v=np.meshgrid((np.arange(w)-w/2)/(w/2),(np.arange(h)-h/2)/(h/2)); xyz=np.stack((u*d,v*d,d),-1)
 # fixed 8x8 voxel-like pooled geometry, no segmentation/object identity
 return xyz.reshape(n,t,16,8,16,8,3).mean((3,5)).reshape(n,t,-1).astype(np.float32)
def run(root):
 root=Path(root);out={"schema":"milestone4v-pointcloud-baseline-v1","tasks":{}}
 for task in TASKS:
  f=_pc(root,task); rows=_rows(root/f"{task}_candidates.jsonl"); refs=np.array([r["history_ref"] for r in rows]); acts=np.array([r["candidate_actions"] for r in rows],np.float32).reshape(len(rows),-1);y=np.load(root/f"{task}_labels.npz")["success"].astype(np.float32)
  tr=np.array([i for i,r in enumerate(rows) if r["split"]=="train"]);te=np.array([i for i,r in enumerate(rows) if r["split"]=="test"]);vals=[]
  for seed in (17,29,43):
   score=_train("ordered_gru",f[refs],acts,y,tr,te,seed);vals.append(_pair_order(y[te],score,[rows[i] for i in te])[0])
  out["tasks"][task]={"ordered_pointcloud_gru_pair_order":float(np.mean(vals)),"uses_segmentation":False}
 (root/"pointcloud_baseline_report.json").write_text(json.dumps(out,indent=2,sort_keys=True)+"\n");return out
if __name__=="__main__":
 p=Path(__file__).resolve().parents[2]/"outputs"/"actmask"/"milestone4v_visual_pilot"/"visual_pilot";print(json.dumps(run(p),indent=2,sort_keys=True))
