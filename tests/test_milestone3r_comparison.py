import json
from actmask.experiments.milestone3r_comparison import compare


def test_comparison_selects_analytic_on_validation_only(tmp_path):
    variants=['gaussian_severe','heteroscedastic_medium','timestamp_jitter','random_dropout','last_frame_dropout','burst_occlusion','observation_latency','partial_coordinates']
    analytic=[]; learned=[]
    for variant in variants:
        analytic += [dict(split='val',variant=variant,task='a',method='A',pair_order_accuracy=.9),dict(split='val',variant=variant,task='a',method='B',pair_order_accuracy=.1),dict(split='val',variant=variant,task='a',method='Oracle',fair=False,pair_order_accuracy=1.0),dict(split='test',variant=variant,task='a',method='A',pair_order_accuracy=.2),dict(split='test',variant=variant,task='a',method='B',pair_order_accuracy=1.0)]
        learned.append(dict(split='test',variant=variant,task='a',seed=17,pair_order_accuracy=.5))
    ap=tmp_path/'a.json';lp=tmp_path/'l.json';ap.write_text(json.dumps({'records':analytic}));lp.write_text(json.dumps({'records':learned}))
    result=compare(ap,lp)
    assert result['selected_analytic']=='A'
    assert result['robust_test_mean']['delta']==.3
