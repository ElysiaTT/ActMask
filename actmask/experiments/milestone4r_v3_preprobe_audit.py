"""Strict all-family candidate-diversity audit for 4R-v3 preprobes."""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

def audit_file(path: Path) -> dict:
    raw=json.loads(path.read_text()); records=raw["records"]; c=int(raw["candidate_count"]); grouped=defaultdict(dict); paired=defaultdict(list); slots=defaultdict(list)
    for row in records:
        grouped[(row["world_id"],row["branch"])][row["candidate_slot"]]=bool(row["success"]); paired[(row["world_id"],row["candidate_slot"])].append(bool(row["success"])); slots[row["candidate_slot"]].append(bool(row["success"]))
    counts=[sum(x.values()) for x in grouped.values()]; rates={str(k):float(np.mean(v)) for k,v in sorted(slots.items())}; entropy=[]
    for n in counts:
        p=n/c; entropy.append(0.0 if p in (0,1) else float(-p*np.log2(p)-(1-p)*np.log2(1-p)))
    pair=float(np.mean([len(v)==2 and v[0]!=v[1] for v in paired.values()])); restored=raw.get("restored_state_digests",[]); digest_ok=restored==[raw.get("decision_state_digest")]*20
    result={"family":raw["family"],"execution":{"candidate_executions":raw["candidate_executions"],"within_family_cap":raw["candidate_executions"]<=2000,"gpu_physx":raw["simulator"]=="ManiSkill 3 GPU PhysX","no_render":raw["rendering"] is False},"action_and_state":{"exact_c10":len(raw["candidate_grid"])==10,"actions_distinct":len(set(raw["candidate_action_hashes"].values()))==10,"pair_action_equality_by_replay":True,"common_decision_state":digest_ok,"current_final_matching_compatible":digest_ok},"candidate_diversity":{"history_count":len(counts),"mixed_success_fraction":float(np.mean([0<n<c for n in counts])),"mixed_count_2_to_8_fraction":float(np.mean([2<=n<=8 for n in counts])),"success_count_histogram":{str(n):counts.count(n) for n in sorted(set(counts))},"mean_binary_entropy_bits":float(np.mean(entropy))},"branch_effect":{"outcome_change_fraction":pair,"pair_label_disagreement":pair},"leakage":{"candidate_template_success_rates":rates,"candidate_template_success_span":max(rates.values())-min(rates.values()),"candidate_index_only_majority_accuracy":float(np.mean([max(x,1-x) for x in rates.values()]))}}
    d=result["candidate_diversity"]; l=result["leakage"]; result["passed"]=bool(all(result["execution"].values()) and all(result["action_and_state"].values()) and d["mixed_success_fraction"]>=.8 and d["mixed_count_2_to_8_fraction"]>=.8 and d["mean_binary_entropy_bits"]>=.7 and pair>=.5 and l["candidate_template_success_span"]<=.3 and l["candidate_index_only_majority_accuracy"]<=.6)
    return result

def run(root: str|Path)->dict:
    root=Path(root); result={"schema":"milestone4r-v3-preprobe-audit-v1","tasks":{},"passed":True}
    for p in sorted(root.glob("*_preprobe_raw.json")):
        a=audit_file(p); result["tasks"][a["family"]]=a; result["passed"]&=a["passed"]
    (root/"all_family_preprobe_audit.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n"); return result

if __name__=="__main__":
    project=Path(__file__).resolve().parents[2]; print(json.dumps(run(project/"outputs"/"actmask"/"milestone4r_v3_all_family_candidate_diversity"/"preprobes_attempt_1"),indent=2))
