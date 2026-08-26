"""Evaluate all fair analytic estimators across clean and 3R corruptions."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from actmask.data.maniskill_pilot import load_model_inputs
from actmask.data.milestone3q_signed import TASK_IDS
from actmask.experiments.milestone3r_analytic_gpu import ESTIMATORS_GPU, _last_two
from actmask.experiments.milestone3r_metrics import pair_order

def _rows(path):return [json.loads(x) for x in path.read_text().splitlines()]
def _split(rows,split):
 remainder={'val':3,'test':4}[split]
 return np.array([i for i,r in enumerate(rows) if r['regime']=='signed_order' and r['world_id']%5==remainder])
def run(clean_root,corruption_root,output):
    clean_root,corruption_root,output=map(Path,(clean_root,corruption_root,output))
    manifest=json.loads((corruption_root/'corruption_manifest.json').read_text())
    variants=['clean']+[s['corruption_id'] for s in manifest['specifications']]
    records=[]
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    for variant in variants:
        root=clean_root if variant=='clean' else corruption_root/variant
        for task in TASK_IDS:
            value=load_model_inputs(root/f'{task}_model_inputs.npz')
            labels=np.load(root/f'{task}_labels.npz')['success']
            rows=_rows(root/f'{task}_metadata.jsonl')
            pairs=np.asarray([r['signed_group'] for r in rows])
            h=torch.from_numpy(value['history']).float().to(device)
            ts=torch.from_numpy(value['timestamps']).float().to(device)
            vis=torch.from_numpy(value['visibility']).float().to(device)
            for name,fn in ESTIMATORS_GPU.items():
                for split in ('val','test'):
                    ix=_split(rows,split)
                    index=torch.from_numpy(ix).to(device)
                    score=fn(h[index],ts[index],vis[index])[:,1].cpu().numpy()
                    records.append(dict(variant=variant,task=task,method=name,fair=True,split=split,pair_order_accuracy=pair_order(labels[ix],score,pairs[ix])))
            # This diagnostic sees the uncorrupted clean state and never enters
            # fair estimator selection or learned-versus-analytic comparison.
            oracle=load_model_inputs(clean_root/f'{task}_model_inputs.npz')
            oh=torch.from_numpy(oracle['history']).float().to(device)
            ot=torch.from_numpy(oracle['timestamps']).float().to(device)
            ov=torch.from_numpy(oracle['visibility']).float().to(device)
            oracle_score=_last_two(oh,ot,ov)[:,1].cpu().numpy()
            for split in ('val','test'):
                ix=_split(rows,split)
                records.append(dict(variant=variant,task=task,method='OracleCleanState',fair=False,split=split,pair_order_accuracy=pair_order(labels[ix],oracle_score[ix],pairs[ix])))
    report=dict(records=records,aggregate={})
    methods=tuple(ESTIMATORS_GPU)+('OracleCleanState',)
    for variant in variants:
        report['aggregate'][variant]={name:{split:float(np.nanmean([r['pair_order_accuracy'] for r in records if r['variant']==variant and r['method']==name and r['split']==split])) for split in ('val','test')} for name in methods}
    output.mkdir(parents=True,exist_ok=True)
    (output/'analytic_corruption_matrix.json').write_text(json.dumps(report,indent=2))
    return report
