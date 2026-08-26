import json
from actmask.experiments.milestone3r_completion_audit import run


def test_audit_reports_unmet_nonlinear_gate(tmp_path):
    root=tmp_path/'root';evaluation=root/'evaluation';evaluation.mkdir(parents=True);(root/'corruption_full').mkdir()
    (root/'frozen_3q_reproduction.json').write_text(json.dumps({'reproduction_passed':True}))
    (root/'corruption_full/corruption_manifest.json').write_text(json.dumps({'tasks':{str(i):{'task':{'labels_unchanged':True,'candidate_actions_unchanged':True}} for i in range(52)}}))
    (evaluation/'selected_vs_analytic.json').write_text(json.dumps({'selected_analytic':'LastTwo','selected_learned':{'seeds':[1,2,3,4,5]},'robust_test_mean':{},'grouped_bootstrap':{'ci95':[.1,.2]}}))
    (evaluation/'ranking_c5_c10_c20.json').write_text(json.dumps({'aggregate':[{'candidate_count':5},{'candidate_count':10},{'candidate_count':20}],'efficiency':{'c20_inference_latency_ms_p95':1}}))
    (evaluation/'causal_controls_five_seed.json').write_text(json.dumps({'diagnostics':{'reversal_causal_drop':1}}))
    (evaluation/'nonlinear_held_mechanism_ood.json').write_text(json.dumps({'records':[{'delta':0},{'delta':0}]}))
    (evaluation/'training_conditions_validation.json').write_text(json.dumps({'seeds':[1,2,3],'conditions':list(range(6))}))
    assert 'two_nonlinear_ood_positive_axes' in run(root)['blocking_requirements']
