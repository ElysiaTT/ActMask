"""Non-heavy completion audit for the immutable 5B-0 output bundle."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/"outputs"/"actmask"/"milestone5b0_relational_probe"; STATE=OUT/"state_probe"
HASH="390a71f61e8f6d9c874f29638cf3756e41275e69f9360727bdc2348e25e6fc57"
def test_frozen_config_and_gpu_physx_schema():
 assert hashlib.sha256((OUT/"preregistered_config.json").read_bytes()).hexdigest()==HASH
 manifest=json.loads((STATE/"data_manifest.json").read_text())
 assert manifest["config_sha256"]==HASH and manifest["candidate_executions"]==2560 and manifest["rendering"] is False
 assert manifest["simulator"]=="ManiSkill 3 GPU PhysX"
def test_moments_candidates_and_splits():
 assert json.loads((OUT/"moment_matching_audit.json").read_text())["all_pass"]
 diversity=json.loads((OUT/"candidate_diversity_audit.json").read_text()); assert diversity["pass"] and diversity["success_count_histogram"]=={"5":256}
 assert json.loads((OUT/"integrity_audit.json").read_text())["all_pass"]
def test_raw_score_artifacts_and_estimator_consistency():
 manifest=json.loads((OUT/"raw_score_manifest.json").read_text()); assert manifest["all_raw_scores_saved"] and len(manifest["entries"])==51
 for model in ("current_object_token_mlp","unordered_object_token_deepsets","ordered_object_token_gru","temporal_convolution"):
  for seed in (17,29,43):
   base=OUT/"learned"/model/f"seed_{seed}"
   assert all((base/name).is_file() for name in ("checkpoint.pt","model_config.json","preprocessing_config.json","seed.txt","data_sha256.txt","raw_scores.jsonl","metrics.json","command.txt"))
   rows=[json.loads(x) for x in (base/"raw_scores.jsonl").read_text().splitlines()]
   assert len(rows)==2560 and all({"history_id","candidate_slot","pair_id","success","score","split","seed","model"}<=set(x) for x in rows)
   # Metrics are reproducible from the stored raw scores rather than a report-only value.
   from actmask.experiments.milestone5b0_baselines import _metrics
   assert _metrics(rows)==json.loads((base/"metrics.json").read_text())
 assert json.loads((OUT/"ci_consistency_audit.json").read_text())["all_pass"]
def test_acceptance_decision_and_headroom():
 final=json.loads((OUT/"final_decision.json").read_text()); assert final["decision"]=="A. RELATIONAL_PROBE_PASSED_AUTHORIZE_5B" and all(final["gates"].values())
 assert json.loads((OUT/"headroom_metrics.json").read_text())["headroom"]>=.08
