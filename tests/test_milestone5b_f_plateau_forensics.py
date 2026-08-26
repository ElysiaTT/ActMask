"""Read-only forensic completion checks for Milestone 5B-F."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/"outputs"/"actmask"/"milestone5b_f_plateau_forensics"
HASH="00f01975947f4a1f36781f1a9c1e9ec45893376f87fe456366d01196d78b7251"
def read(name): return json.loads((OUT/name).read_text())
def test_frozen_config_and_source_inventory():
 assert hashlib.sha256((OUT/"preregistered_config.json").read_bytes()).hexdigest()==HASH
 assert read("source_inventory.json")["all_required_present"]
def test_breakdown_and_error_overlap_are_exact():
 b=read("per_task_breakdown.json")["models"]
 assert all(b["RelDynVerifier"]["by_task"][task]["pair_order_accuracy"]==.75 for task in b["RelDynVerifier"]["by_task"])
 overlap=read("error_overlap_report.json"); assert overlap["all_three_failed"]==120 and all(v["jaccard"]==1 for v in overlap["overlap"].values())
def test_fair_collision_and_candidate_conditioned_bound():
 audit=read("fair_observation_identifiability.json")
 assert audit["conclusion"]=="B. FAIR_INPUT_AMBIGUOUS" and audit["conflicting_exact_classes"]==640
 assert audit["candidate_conditioned_conflicting_classes"]==640 and audit["bayes_majority_upper_bound"]==.75
def test_saved_checkpoint_sensitivity_and_oracle_separation():
 sensitivity=read("action_conditioning_sensitivity.json"); assert sensitivity["conclusion"]=="A. MODEL_USES_RELATIONAL_ACTION_FEATURES"
 assert sensitivity["perturbations"]["action_zeroed"]["mean_absolute_score_change"]>0
 oracle=read("oracle_field_minimality.json")["diagnostics"]; assert oracle["persistent_object_identity"]["pair_order_accuracy"]==1 and oracle["mechanism_label"]["pair_order_accuracy"]==.5
def test_metric_ties_and_final_decision():
 sanity=read("metric_split_sanity.json"); assert sanity["test_pair_count"]==240 and sanity["tied_pair_count"]==120 and sanity["expected_pair_order_from_ties"]==.75
 assert read("final_decision.json")["decision"]=="B. FAIR_OBSERVATION_UNIDENTIFIABLE"
