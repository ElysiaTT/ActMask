"""Evidence-based completion audit for the 3R specification."""
from __future__ import annotations

import json
from pathlib import Path


def _load(path): return json.loads(Path(path).read_text())


def run(root='outputs/actmask/milestone3r_robust_signed_dynamics'):
    root=Path(root);evaluation=root/'evaluation';freeze=_load(root/'frozen_3q_reproduction.json');corrupt=_load(root/'corruption_full/corruption_manifest.json');compare=_load(evaluation/'selected_vs_analytic.json');ranking=_load(evaluation/'ranking_c5_c10_c20.json');causal=_load(evaluation/'causal_controls_five_seed.json');nonlinear=_load(evaluation/'nonlinear_held_mechanism_ood.json');conditions=_load(evaluation/'training_conditions_validation.json')
    task_audits=[audit for variant in corrupt['tasks'].values() for audit in variant.values()]
    robust=compare['robust_test_mean'];efficiency=ranking['efficiency'];checks={
        'freeze_reproduced':freeze.get('status')=='pass' or freeze.get('reproduction_passed',False),
        'all_corruption_audits':len(task_audits)==52 and all(x['labels_unchanged'] and x['candidate_actions_unchanged'] for x in task_audits),
        'analytic_oracle_excluded':compare['selected_analytic']!='OracleCleanState',
        'three_seed_training_conditions':len(conditions['seeds'])==3 and len(conditions['conditions'])>=6,
        'five_seed_final_confirmation':len(compare['selected_learned']['seeds'])==5,
        'c5_c10_c20':{str(count):any(x['candidate_count']==count for x in ranking['aggregate']) for count in (5,10,20)},
        'robust_advantage_ci_excludes_zero':compare['grouped_bootstrap']['ci95'][0]>0,
        'reversal_drop_at_least_015':causal['diagnostics']['reversal_causal_drop']>=.15,
        'c20_at_least_5hz':efficiency['c20_inference_latency_ms_p95']<=200,
        'two_nonlinear_ood_positive_axes':sum(r['delta']>0 for r in nonlinear['records'])>=2,
        'full_regression_passed':False,
    }
    report=dict(stage_decision='E_human_decision_required',checks=checks,robust_pair_order=robust,nonlinear_ood=nonlinear['records'],blocking_requirements=[name for name,value in checks.items() if value is False or (isinstance(value,dict) and not all(value.values()))],notes=['The full suite excluding the unrelated rendering test passed 151 tests; the remaining renderer test was not run to completion under the CPU-use constraint.','The two required held nonlinear OOD mechanisms tie the fair analytic baseline, so the visual-authorization gate cannot be met without a separately authorized new preregistration.'])
    (evaluation/'completion_audit.json').write_text(json.dumps(report,indent=2));return report
