"""Three-seed grouped nonlinear diagnostic for 3R analytic and learned methods."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from actmask.experiments.milestone3r_analytic import ESTIMATORS
from actmask.experiments.milestone3r_metrics import pair_order
from actmask.models.milestone3r_temporal import OrderedTemporalMLP,GRUTemporal,TemporalConv1D,MaskedHistoryMLP

def run(root):
 root=Path(root);x=np.load(root/'nonlinear_model_inputs.npz');y=np.load(root/'nonlinear_labels.npz')['success'].astype(np.float32);rows=[json.loads(l) for l in open(root/'nonlinear_metadata.jsonl')];train=np.array([i for i,r in enumerate(rows) if r['world_id']<12]);test=np.array([i for i,r in enumerate(rows) if r['world_id']>=12]);h=x['history'];t=x['timestamps'];v=x['visibility'];pairs=np.array([r['signed_group'] for r in rows]);out=[]
 for name,fn in ESTIMATORS.items():
  vel=fn(h,t,v); score=vel[:,1];out.append(dict(seed=None,method=name,pair_order_accuracy=pair_order(y[test],score[test],pairs[test])))
 device=torch.device('cuda:0');batch={k:torch.from_numpy(x[k]).to(device) for k in ('history','timestamps','visibility','observation_confidence','candidate_actions','tcp_state')};idim=h.shape[2]+3;adim=x['candidate_actions'].shape[1]*3+3
 for seed in (17,29,43):
  for name,model in [('OrderedTemporalMLP',OrderedTemporalMLP(4*idim,adim)),('GRU',GRUTemporal(idim,adim)),('TemporalConv1D',TemporalConv1D(idim,adim)),('MaskedHistoryMLP',MaskedHistoryMLP(4*idim,adim))]:
   torch.manual_seed(seed);model.to(device);opt=torch.optim.AdamW(model.parameters(),lr=.02);target=torch.from_numpy(y).to(device);ix=torch.from_numpy(train).to(device)
   for _ in range(180):
    loss=nn.functional.binary_cross_entropy_with_logits(model({k:z[ix] for k,z in batch.items()}),target[ix]);opt.zero_grad();loss.backward();opt.step()
   with torch.no_grad():s=model(batch).cpu().numpy();out.append(dict(seed=seed,method=name,pair_order_accuracy=pair_order(y[test],s[test],pairs[test])))
 report=dict(records=out,train_worlds=12,test_worlds=4);(root/'nonlinear_three_seed_diagnostic.json').write_text(json.dumps(report,indent=2));return report
