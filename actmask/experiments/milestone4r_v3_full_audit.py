"""v3 all-family full audit, extending the frozen storage audit with v3 gates."""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from actmask.experiments.milestone4r_audit import run as storage_run
def run(root):
 root=Path(root); storage=storage_run(root);out={"schema":"milestone4r-v3-full-audit-v1","storage_audit":storage,"tasks":{},"passed":True}
 for task,prior in storage["tasks"].items():
  rows=[json.loads(x) for x in (root/f"{task}_candidates.jsonl").read_text().splitlines()];labels=np.load(root/f"{task}_labels.npz")["success"].astype(bool);hist=defaultdict(list);slots=defaultdict(list)
  for r,y in zip(rows,labels):hist[r["history_ref"]].append(bool(y));slots[r["candidate_slot"]].append(bool(y))
  counts=[sum(v) for v in hist.values()];rates={str(k):float(np.mean(v)) for k,v in slots.items()};ent=[0 if n in (0,10) else float(-(n/10)*np.log2(n/10)-(1-n/10)*np.log2(1-n/10)) for n in counts];a={"mixed_success_fraction":float(np.mean([0<n<10 for n in counts])),"mixed_count_2_to_8_fraction":float(np.mean([2<=n<=8 for n in counts])),"mean_binary_entropy_bits":float(np.mean(ent)),"candidate_template_success_span":max(rates.values())-min(rates.values()),"candidate_index_only_majority_accuracy":float(np.mean([max(x,1-x) for x in rates.values()])),"pair_label_disagreement":prior["pair_label_disagreement"],"pair_matching":prior["pair_matching"],"storage_integrity":prior["passed"]};a["passed"]=bool(a["mixed_success_fraction"]>=.8 and a["mixed_count_2_to_8_fraction"]>=.8 and a["mean_binary_entropy_bits"]>=.7 and a["candidate_template_success_span"]<=.3 and a["candidate_index_only_majority_accuracy"]<=.6 and a["pair_label_disagreement"]>=.5 and all(x==1 for x in a["pair_matching"].values()) and a["storage_integrity"]);out["tasks"][task]=a;out["passed"]&=a["passed"]
 (root/"milestone4r_v3_full_audit.json").write_text(json.dumps(out,indent=2,sort_keys=True)+"\n");return out
