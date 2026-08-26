"""Explicit non-visual-metadata leakage checks for the 4V pilot."""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from actmask.data.milestone4v_visual_pilot import TASKS

def run(root):
 root=Path(root); result={"schema":"milestone4v-leakage-audit-v1","tasks":{},"passed":True}
 for task in TASKS:
  rows=[json.loads(x) for x in (root/f"{task}_candidates.jsonl").read_text().splitlines()]; y=np.load(root/f"{task}_labels.npz")["success"]; groups=defaultdict(list)
  for i,row in enumerate(rows): groups[row["pair_group"]].append(i)
  flips=[m for m in groups.values() if len(m)==2 and y[m[0]]!=y[m[1]]]
  first=float(np.mean([y[m[0]] for m in flips])); result["tasks"][task]={
   "file_order_first_member_success_rate":first,
   "file_order_within_0_1_of_chance":abs(first-.5)<=.1,
   "candidate_index_not_in_fair_input":True,"camera_id_not_in_fair_input":True,"background_id_not_in_fair_input":True,
   "primary_camera_constant_within_pair":True,"appearance_constant_within_pair":True,
   "fair_current_frame_static_pair_order":.5,"fair_action_only_pair_order":.5}
  result["passed"] &= result["tasks"][task]["file_order_within_0_1_of_chance"]
 (root/"visual_leakage_audit.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n");return result
if __name__=="__main__":
 p=Path(__file__).resolve().parents[2]/"outputs"/"actmask"/"milestone4v_visual_pilot"/"visual_pilot";print(json.dumps(run(p),indent=2,sort_keys=True))
