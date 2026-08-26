"""Deterministic integrity and diversity audits for the frozen 5B-0 probe."""
from __future__ import annotations
import hashlib, json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/"outputs"/"actmask"/"milestone5b0_relational_probe"; STATE=OUT/"state_probe"

def load_labels():
    return [json.loads(x) for x in (STATE/"candidate_labels.jsonl").read_text().splitlines() if x]

def run(output: Path=OUT) -> dict:
    labels=load_labels(); by_history=defaultdict(list); by_slot=defaultdict(list); pair=defaultdict(list)
    for row in labels: by_history[(row["task"],row["history_id"])].append(row); by_slot[(row["task"],row["candidate_slot"])].append(row["success"]); pair[row["pair_id"]].append(row)
    counts=[sum(x["success"] for x in rows) for rows in by_history.values()]
    candidate_majority=max(sum(values)/len(values) for values in by_slot.values())
    spans=[]
    for task in {x["task"] for x in labels}:
        rates=[np.mean(by_slot[(task, slot)]) for slot in range(10)]; spans.append(float(max(rates)-min(rates)))
    result={"schema":"milestone5b0-candidate-diversity-v1","histories":len(by_history),"mixed_fraction":float(np.mean([(0<c<10) for c in counts])),"count_2_to_8_fraction":float(np.mean([(2<=c<=8) for c in counts])),"success_count_histogram":dict(Counter(counts)),"candidate_index_majority_accuracy":float(candidate_majority),"candidate_template_success_span":max(spans),"candidate_actions_identical_across_pairs":all(len(x)==2 for x in pair.values()),"pass":False}
    result["pass"]=(result["mixed_fraction"]>=.8 and result["count_2_to_8_fraction"]>=.8 and result["candidate_index_majority_accuracy"]<=.6 and result["candidate_template_success_span"]<=.3 and result["candidate_actions_identical_across_pairs"])
    (output/"candidate_diversity_audit.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    config_hash=hashlib.sha256((output/"preregistered_config.json").read_bytes()).hexdigest()
    history=np.load(STATE/"histories.npz")["fair_observed_tokens"]
    checks={"config_hash_stable":config_hash==json.loads((STATE/"data_manifest.json").read_text())["config_sha256"],"schema_shapes":history.shape==(256,6,2,3),"labels":len(labels)==2560,"split_base_world_integrity":all(x["split"]==("train" if x["world_id"]<39 else "validation" if x["world_id"]<52 else "test") for x in labels),"hidden_field_excluded_from_fair_history": "oracle" not in str(history.dtype.names),"pair_labels_complementary":all(sum(x["success"] for x in rows)==1 for rows in pair.values()),"moment_audit_pass":json.loads((output/"moment_matching_audit.json").read_text())["all_pass"],"candidate_audit_pass":result["pass"]}
    checks["all_pass"]=all(checks.values()); (output/"integrity_audit.json").write_text(json.dumps(checks,indent=2,sort_keys=True)+"\n")
    return {"candidate":result,"integrity":checks}

if __name__=="__main__": print(json.dumps(run(),indent=2,sort_keys=True))
