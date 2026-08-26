from __future__ import annotations
import hashlib,json
from pathlib import Path
R=Path(__file__).resolve().parents[2];O=R/'outputs/actmask/milestone5b_data_v2_partial_cue';S=O/'state_probe';H='6b80b1fd501174f585a1246427ac03d236446472f240a66f70e19c51f66b8b49'
def run():
 r=json.loads((O/'baseline_report.json').read_text());ident=json.loads((O/'fair_identifiability_audit.json').read_text());tok=json.loads((O/'token_order_audit.json').read_text());div=json.loads((O/'candidate_diversity_audit.json').read_text());mom=json.loads((O/'moment_cue_order_matching_audit.json').read_text());man=json.loads((O/'raw_score_manifest.json').read_text())
 def p(n):return r['models'][n]['metrics']['pair_order_accuracy']['mean']
 ci=[]
 for n,v in r['models'].items():
  for k,x in v['metrics'].items():
   lo,hi=x['bootstrap_ci95'];ci.append({'model':n,'metric':k,'contains_point':lo<=x['mean']<=hi})
 (O/'ci_consistency_audit.json').write_text(json.dumps({'all_pass':all(x['contains_point'] for x in ci),'checks':ci},indent=2))
 best=max((p(n),n) for n in ('current_object_token_mlp','unordered_object_token_deepsets','ordered_object_token_gru','temporal_convolution'));oracle=max(p(n) for n in ('oracle_identity','oracle_target','oracle_relation','oracle_contact','oracle_graph'));head={'best_standard':best[1],'best_standard_pair_order':best[0],'oracle_pair_order':oracle,'headroom':oracle-best[0],'pass':oracle-best[0]>=.08};(O/'headroom_metrics.json').write_text(json.dumps(head,indent=2))
 req=('checkpoint.pt','model_config.json','preprocessing_config.json','training_config.json','seed.txt','data_sha256.txt','command.txt','train_log.jsonl','raw_scores.jsonl','metrics.json');art=all(all((O/'learned'/n/f'seed_{s}'/x).is_file() for x in req) for n in ('current_object_token_mlp','unordered_object_token_deepsets','ordered_object_token_gru','temporal_convolution') for s in (17,29,43))
 gates={'config':hashlib.sha256((O/'preregistered_config.json').read_bytes()).hexdigest()==H,'goal':json.loads((O/'goal_cue_presence_audit.json').read_text())['pass'],'token':tok['pass'],'moment':mom['all_pass'],'ident':ident['pass'],'diversity':div['pass'],'artifacts':art and man['all_raw_scores_saved'],'ci':all(x['contains_point'] for x in ci),'temporal_range':all(.6<=p(n)<=.88 for n in ('ordered_object_token_gru','temporal_convolution')),'headroom':head['pass']}
 d='F. STANDARD_TEMPORAL_STILL_SATURATES' if not gates['temporal_range'] or not gates['headroom'] else 'A. DATA_V2_PASSED_AUTHORIZE_5B_V2' if all(gates.values()) else 'J. HUMAN_DECISION_REQUIRED';(O/'final_decision.json').write_text(json.dumps({'decision':d,'gates':gates,'headroom':head},indent=2));return d
if __name__=='__main__':print(run())
