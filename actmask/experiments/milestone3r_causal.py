"""Causal-history and negative-control evaluation for the selected temporal model."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn

from actmask.data.maniskill_pilot import load_model_inputs
from actmask.data.milestone3q_signed import TASK_IDS
from actmask.experiments.milestone3r_metrics import pair_order, ranking
from actmask.models.milestone3r_temporal import ActionMLP, GRUTemporal, StaticMLP, UnorderedHistoryMLP


def _rows(path): return [json.loads(x) for x in Path(path).read_text().splitlines()]
def _ix(rows, split): return np.asarray([i for i,r in enumerate(rows) if r['regime']=='signed_order' and r['world_id']%5=={'train':0,'test':4}[split]])
def _batch(value, device): return {k:torch.from_numpy(value[k]).float().to(device) for k in ('history','timestamps','visibility','observation_confidence','candidate_actions','tcp_state')}
def _metric(rows, indices, labels, score):
    groups=np.asarray([f"{rows[i]['task']}:{rows[i]['world_id']}" for i in indices]);pairs=np.asarray([rows[i]['signed_group'] for i in indices]);r=ranking(labels[indices],score[indices],groups);r['pair_order_accuracy']=pair_order(labels[indices],score[indices],pairs);return r


def _alter(batch, name):
    out={k:v.clone() for k,v in batch.items()};h=out['history'];n,t,_=h.shape
    if name=='correct_ordered': return out
    if name=='exact_reversed': out['history']=torch.flip(h,(1,));out['visibility']=torch.flip(out['visibility'],(1,));out['observation_confidence']=torch.flip(out['observation_confidence'],(1,));return out
    if name=='independent_permutation':
        generator=torch.Generator(device=h.device);generator.manual_seed(731);keys=torch.rand((n,t),generator=generator,device=h.device);perm=keys.argsort(1);out['history']=h.gather(1,perm[...,None].expand_as(h));return out
    if name=='unordered_frame_multiset': out['history']=torch.sort(h,dim=1).values;return out
    if name=='duplicated_final_frame': out['history']=h[:,-1:].expand_as(h);return out
    if name=='last_two_frames_only': out['history']=torch.cat((h[:,-2:-1].expand(-1,t-1,-1),h[:,-1:]),1);return out
    if name=='zeroed_motion': out['history']=h[:,-1:].expand_as(h);return out
    if name=='velocity_sign_removed':
        speed=torch.linalg.vector_norm(h[:,1:]-h[:,:-1],dim=2,keepdim=True);out['history']=h[:,-1:].expand_as(h).clone();out['history'][:,1:,:1]=speed;return out
    if name=='speed_magnitude_only':
        mag=torch.linalg.vector_norm(h-h[:,-1:],dim=2,keepdim=True);projected=torch.zeros_like(h);projected[...,:1]=mag;out['history']=h[:,-1:]+projected;return out
    if name=='shuffled_timestamps': out['timestamps']=torch.flip(out['timestamps'],(1,));return out
    if name=='mismatched_history':
        generator=torch.Generator(device=h.device);generator.manual_seed(193);perm=torch.randperm(n,generator=generator,device=h.device);out['history']=h[perm];out['visibility']=out['visibility'][perm];out['observation_confidence']=out['observation_confidence'][perm];return out
    raise ValueError(name)


def _fit(factory, batch, train, labels, steps, seed):
    device=batch['history'].device;torch.manual_seed(seed);model=factory().to(device);opt=torch.optim.AdamW(model.parameters(),lr=.015);target=torch.from_numpy(labels[train]).to(device);ix=torch.from_numpy(train).to(device);view={k:v[ix] for k,v in batch.items()}
    for _ in range(steps):
        loss=nn.functional.binary_cross_entropy_with_logits(model(view),target);opt.zero_grad();loss.backward();opt.step()
    return model.eval()


def run(clean_root, output_root, seeds=(17,29,43,59,71), steps=60):
    clean_root,output=map(Path,(clean_root,output_root));device=torch.device('cuda:0');records=[];causal_names=('correct_ordered','exact_reversed','independent_permutation','unordered_frame_multiset','duplicated_final_frame','last_two_frames_only','zeroed_motion','velocity_sign_removed','speed_magnitude_only','shuffled_timestamps','mismatched_history')
    for task in TASK_IDS:
        rows=_rows(clean_root/f'{task}_metadata.jsonl');train,test=_ix(rows,'train'),_ix(rows,'test');labels=np.load(clean_root/f'{task}_labels.npz')['success'].astype(np.float32);value=load_model_inputs(clean_root/f'{task}_model_inputs.npz');batch=_batch(value,device);d=value['history'].shape[2]+3;a=value['candidate_actions'].shape[1]*3+3
        temporal_scores={name:[] for name in causal_names}
        for seed in seeds:
            model=_fit(lambda:GRUTemporal(d,a),batch,train,labels,steps,seed)
            with torch.no_grad():
                for name in causal_names: temporal_scores[name].append(model(_alter(batch,name)).cpu().numpy())
        for name,scores in temporal_scores.items(): records.append(dict(task=task,method='SelectedGRUFiveSeed',condition=name,**_metric(rows,test,labels,np.mean(scores,0))))
        controls={'StaticMLP':lambda:StaticMLP(value['history'].shape[2],a),'ActionMLP':lambda:ActionMLP(a),'UnorderedHistoryMLP':lambda:UnorderedHistoryMLP(4*d,a)}
        for name,factory in controls.items():
            scores=[]
            for seed in seeds:
                model=_fit(factory,batch,train,labels,steps,seed)
                with torch.no_grad():scores.append(model(batch).cpu().numpy())
            records.append(dict(task=task,method=name,condition='clean_control',**_metric(rows,test,labels,np.mean(scores,0))))
    aggregate=[]
    for method in sorted({r['method'] for r in records}):
        for condition in sorted({r['condition'] for r in records if r['method']==method}):
            items=[r for r in records if r['method']==method and r['condition']==condition];aggregate.append(dict(method=method,condition=condition,**{k:float(np.mean([r[k] for r in items])) for k in ('top1_success','pair_order_accuracy','normalized_regret','ranking_ap')}))
    lookup={(r['method'],r['condition']):r for r in aggregate};correct=lookup[('SelectedGRUFiveSeed','correct_ordered')]['top1_success'];reversed_top=lookup[('SelectedGRUFiveSeed','exact_reversed')]['top1_success'];report=dict(records=records,aggregate=aggregate,diagnostics=dict(reversal_causal_drop=correct-reversed_top,correct_top1=correct,reversed_top1=reversed_top),seeds=list(seeds),steps=steps)
    output.mkdir(parents=True,exist_ok=True);(output/'causal_controls_five_seed.json').write_text(json.dumps(report,indent=2));return report
