"""Held-mechanism nonlinear OOD test for the selected lightweight GRU."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn

from actmask.experiments.milestone3r_analytic_gpu import _last_two
from actmask.experiments.milestone3r_metrics import pair_order, paired_bootstrap
from actmask.models.milestone3r_temporal import GRUTemporal


HELD_OUT = ('jerk', 'regime_switch')
def _batch(x, device): return {k:torch.from_numpy(x[k]).float().to(device) for k in ('history','timestamps','visibility','observation_confidence','candidate_actions','tcp_state')}


def run(root, output_root, seeds=(17,29,43,59,71), steps=120):
    root,output=map(Path,(root,output_root));x=np.load(root/'nonlinear_model_inputs.npz');labels=np.load(root/'nonlinear_labels.npz')['success'].astype(np.float32);rows=[json.loads(line) for line in (root/'nonlinear_metadata.jsonl').read_text().splitlines()];device=torch.device('cuda:0');batch=_batch(x,device);d=x['history'].shape[2]+3;a=x['candidate_actions'].shape[1]*3+3;records=[];deltas=[];groups=[]
    for held in HELD_OUT:
        train=np.asarray([i for i,r in enumerate(rows) if r['mechanism']!=held and r['world_id']<12]);test=np.asarray([i for i,r in enumerate(rows) if r['mechanism']==held and r['world_id']>=12]);scores=[]
        for seed in seeds:
            torch.manual_seed(seed);model=GRUTemporal(d,a).to(device);opt=torch.optim.AdamW(model.parameters(),lr=.02);ix=torch.from_numpy(train).to(device);view={k:v[ix] for k,v in batch.items()};target=torch.from_numpy(labels[train]).to(device)
            for _ in range(steps):
                loss=nn.functional.binary_cross_entropy_with_logits(model(view),target);opt.zero_grad();loss.backward();opt.step()
            with torch.no_grad():scores.append(model(batch).cpu().numpy())
        learned=np.mean(scores,0);analytic=_last_two(batch['history'],batch['timestamps'],batch['visibility'])[:,1].cpu().numpy();pairs=np.asarray([r['signed_group'] for r in rows]);lp=pair_order(labels[test],learned[test],pairs[test]);ap=pair_order(labels[test],analytic[test],pairs[test]);records.append(dict(held_out_mechanism=held,learned_pair_order_accuracy=lp,analytic_pair_order_accuracy=ap,delta=lp-ap,test_pairs=int(len(test)//2),train_examples=int(len(train))))
        for group in np.unique(pairs[test]):
            ix=test[pairs[test]==group]
            if len(ix)==2: deltas.append((1. if learned[ix[labels[ix].astype(bool)][0]]>learned[ix[~labels[ix].astype(bool)][0]] else 0.)-(1. if analytic[ix[labels[ix].astype(bool)][0]]>analytic[ix[~labels[ix].astype(bool)][0]] else 0.));groups.append(f'{held}:{group}')
    report=dict(protocol='mechanism held out from training; test worlds 12--15 only',held_out_mechanisms=list(HELD_OUT),records=records,paired_bootstrap=paired_bootstrap(np.asarray(deltas),np.asarray(groups)),seeds=list(seeds),steps=steps)
    output.mkdir(parents=True,exist_ok=True);(output/'nonlinear_held_mechanism_ood.json').write_text(json.dumps(report,indent=2));return report
