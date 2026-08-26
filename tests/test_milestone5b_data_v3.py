import json
from pathlib import Path
R=Path(__file__).resolve().parents[1];O=R/'outputs/actmask/milestone5b_data_v3_compositional_ood'
def test_v3_integrity_and_decision():
 assert json.loads((O/'fair_identifiability_audit.json').read_text())['pass']
 assert json.loads((O/'candidate_diversity_audit.json').read_text())['pass']
 assert json.loads((O/'split_audit.json').read_text())['pass']
 assert json.loads((O/'final_decision.json').read_text())['decision']=='E. STANDARD_TEMPORAL_STILL_SATURATES_OOD'
