"""Pre-registered deterministic RGB-D corruption matrix for ordered visual GRU."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from actmask.data.milestone4v_visual_pilot import TASKS
from actmask.experiments.milestone4v_visual_baselines import _features, _rows, _train, _pair_order

LEVELS={"depth_gaussian_noise":(.002,.008,.02),"depth_quantization":(.002,.01,.03),"point_dropout":(.05,.2,.4),"burst_frame_dropout":(1,2,3),"image_occlusion":(.08,.2,.35),"appearance_variation":(.05,.15,.3)}

def run(root):
 root=Path(root); out={"schema":"milestone4v-corruption-v1","tasks":{}}
 for task in TASKS:
  rows=_rows(root/f"{task}_candidates.jsonl"); refs=np.asarray([r["history_ref"] for r in rows]); acts=np.asarray([r["candidate_actions"] for r in rows],np.float32).reshape(len(rows),-1); y=np.load(root/f"{task}_labels.npz")["success"].astype(np.float32)
  train=np.asarray([i for i,r in enumerate(rows) if r["split"]=="train"]); test=np.asarray([i for i,r in enumerate(rows) if r["split"]=="test"]); tr=[rows[i] for i in test]; report={}
  for name, levels in LEVELS.items():
   report[name]={}
   for label, level in zip(("mild","medium","severe"),levels):
    frames=_features(root,task,name,level)[refs]
    values=[]
    for seed in (17,29,43):
     score=_train("ordered_gru",frames,acts,y,train,test,seed); values.append(_pair_order(y[test],score,tr)[0])
    report[name][label]=float(np.mean(values))
  out["tasks"][task]=report
 (root/"visual_corruption_report.json").write_text(json.dumps(out,indent=2,sort_keys=True)+"\n"); return out
if __name__=="__main__":
 p=Path(__file__).resolve().parents[2]/"outputs"/"actmask"/"milestone4v_visual_pilot"/"visual_pilot"; print(json.dumps(run(p),indent=2,sort_keys=True))
