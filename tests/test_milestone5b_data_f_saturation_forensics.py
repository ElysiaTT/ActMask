"""Read-only verification for the observable-cue saturation forensic."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"outputs"/"actmask"/"milestone5b_data_f_saturation_forensics";HASH="02ecc9f500b64a64e2dbd32bfbaa0e33b265113108f6f5211108e867ce758e69"
def read(name):return json.loads((OUT/name).read_text())
def test_config_and_source_inventory():
 assert hashlib.sha256((OUT/"preregistered_config.json").read_bytes()).hexdigest()==HASH
 assert read("source_inventory.json")["all_required_present"]
def test_saturation_breakdown_and_checkpoint_perturbations():
 b=read("saturation_breakdown.json")["models"]
 assert all(v["overall_test"]["pair_order_accuracy"]==1 for v in b.values())
 a=read("cue_ablation_diagnostics.json")["models"]
 for v in a.values():
  assert v["target_goal_in_checkpoint_input"] is False
  assert v["perturbations"]["remove_appearance"]["pair_order"]==1
  assert v["perturbations"]["consistent_token_swap"]["pair_order"]==0
def test_split_stress_artifacts_and_final_decision():
 d=read("split_stress_diagnostics.json")["tiny_diagnostic_retrains"]
 assert d["held_out_gap_range"]["pair_order"]==1 and d["leave_one_task"]["pair_order"]==0
 for value in d.values():assert (OUT/value["artifact_path"]/"checkpoint.pt").is_file() and (OUT/value["artifact_path"]/"raw_scores.jsonl").is_file()
 assert read("final_decision.json")["decision"]=="D. METRIC_OR_IMPLEMENTATION_ARTIFACT"
