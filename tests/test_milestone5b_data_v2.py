import json
from pathlib import Path
R=Path(__file__).resolve().parents[1];O=R/'outputs/actmask/milestone5b_data_v2_partial_cue'
def test_v2_audits_and_decision():
 assert json.loads((O/'fair_identifiability_audit.json').read_text())['pass']
 assert json.loads((O/'token_order_audit.json').read_text())['pass']
 assert json.loads((O/'candidate_diversity_audit.json').read_text())['pass']
 assert json.loads((O/'final_decision.json').read_text())['decision']=='F. STANDARD_TEMPORAL_STILL_SATURATES'
