"""Pre-registered small-data validity evaluation for 3R-NL v1."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import torch
from torch import nn

from actmask.data.maniskill_pilot import load_model_inputs
from actmask.experiments.milestone3r_analytic import ESTIMATORS
from actmask.experiments.milestone3r_metrics import pair_order
from actmask.models.milestone3r_temporal import ActionMLP, GRUTemporal, MaskedHistoryMLP, OrderedTemporalMLP, StaticMLP, TemporalConv1D, UnorderedHistoryMLP


def _rows(path): return [json.loads(line) for line in Path(path).read_text().splitlines()]
def _ix(rows, split): return np.asarray([i for i,r in enumerate(rows) if r['split']==split])
def _batch(values, device): return {k:torch.from_numpy(values[k]).float().to(device) for k in ('history','timestamps','visibility','observation_confidence','candidate_actions','tcp_state')}
def _view(batch, ix):
    index=torch.from_numpy(ix).to(next(iter(batch.values())).device);return {k:v[index] for k,v in batch.items()}
def _score_velocity(velocity, actions):
    # State observations end with the observable target/goal position; action
    # is a 3-D TCP displacement. No hidden task buffer is used.
    return (velocity[:, -3:]*actions.sum(1)).sum(1)


def run(root, output_root, seeds=(17,29,43), steps=180):
    root,output=map(Path,(root,output_root));values=load_model_inputs(root/'model_inputs.npz');labels=np.load(root/'labels.npz')['success'].astype(np.float32);rows=_rows(root/'metadata.jsonl');train,val,test=_ix(rows,'train'),_ix(rows,'val'),_ix(rows,'test');pairs=np.asarray([r['signed_group'] for r in rows]);device=torch.device('cuda:0');records=[]
    for name,fn in ESTIMATORS.items():
        velocity=fn(values['history'],values['timestamps'],values['visibility']);score=_score_velocity(velocity,values['candidate_actions'])
        for split,ix in (('validation',val),('test',test)):
            records.append(dict(method=name,kind='analytic',seed=None,split=split,pair_order_accuracy=pair_order(labels[ix],score[ix],pairs[ix])))
    batch=_batch(values,device);horizon=values['history'].shape[1];frame_dim=values['history'].shape[2]+3;action_dim=values['candidate_actions'].shape[1]*3+3
    factories={
        'GRU':lambda:GRUTemporal(frame_dim,action_dim),
        'TemporalConv1D':lambda:TemporalConv1D(frame_dim,action_dim),
        'OrderedTemporalMLP':lambda:OrderedTemporalMLP(horizon*frame_dim,action_dim),
        'MaskedHistoryMLP':lambda:MaskedHistoryMLP(horizon*frame_dim,action_dim),
        'UnorderedHistoryMLP':lambda:UnorderedHistoryMLP(horizon*frame_dim,action_dim),
        'StaticMLP':lambda:StaticMLP(values['history'].shape[2],action_dim),
        'ActionMLP':lambda:ActionMLP(action_dim),
    }
    target=torch.from_numpy(labels[train]).to(device);training=_view(batch,train)
    for name,factory in factories.items():
        for seed in seeds:
            torch.manual_seed(seed);model=factory().to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=.015)
            for _ in range(steps):
                loss=nn.functional.binary_cross_entropy_with_logits(model(training),target);optimizer.zero_grad();loss.backward();optimizer.step()
            with torch.no_grad():score=model(batch).cpu().numpy()
            for split,ix in (('validation',val),('test',test)):
                records.append(dict(method=name,kind='learned',seed=seed,split=split,pair_order_accuracy=pair_order(labels[ix],score[ix],pairs[ix])))
    summary=[]
    for method in sorted({x['method'] for x in records}):
        for split in ('validation','test'):
            values_for_method=[x['pair_order_accuracy'] for x in records if x['method']==method and x['split']==split]
            summary.append(dict(method=method,split=split,pair_order_accuracy=float(np.mean(values_for_method)),observations=len(values_for_method)))
    validation={x['method']:x['pair_order_accuracy'] for x in summary if x['split']=='validation'}
    validity=dict(last_two_near_chance=validation['LastTwoFrameFiniteDifference']<=.55,static_near_chance=validation['StaticMLP']<=.55,unordered_near_chance=validation['UnorderedHistoryMLP']<=.55,long_history_above_point70=max(validation[x] for x in ('GRU','TemporalConv1D','OrderedTemporalMLP','MaskedHistoryMLP'))>.70)
    report=dict(protocol='fixed three-seed pre-registered probe; validation gate read before any full data generation',records=records,summary=summary,validity=validity,probe_valid=all(validity.values()),seeds=list(seeds),steps=steps,split_counts=dict(train=int(len(train)),validation=int(len(val)),test=int(len(test))))
    output.mkdir(parents=True,exist_ok=True);(output/'probe_evaluation.json').write_text(json.dumps(report,indent=2));return report
