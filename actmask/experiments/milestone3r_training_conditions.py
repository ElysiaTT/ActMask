"""Validation-only 3R training-condition audit for the fixed GRU architecture."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from torch import nn

from actmask.data.maniskill_pilot import load_model_inputs
from actmask.data.milestone3q_signed import TASK_IDS
from actmask.experiments.milestone3r_metrics import pair_order
from actmask.models.milestone3r_temporal import GRUTemporal


def _rows(path): return [json.loads(x) for x in Path(path).read_text().splitlines()]
def _idx(rows, split): return np.asarray([i for i,r in enumerate(rows) if r['regime']=='signed_order' and r['world_id']%5=={'train':0,'val':3}[split]])
def _batch(value, device): return {k:torch.from_numpy(value[k]).float().to(device) for k in ('history','timestamps','visibility','observation_confidence','candidate_actions','tcp_state')}
def _view(batch, ix):
    index=torch.from_numpy(ix).to(next(iter(batch.values())).device);return {k:v[index] for k,v in batch.items()}


def pairwise_group_loss(scores, labels, groups):
    """Logistic positive-over-negative loss, only for complete signed groups."""
    terms=[]
    for group in torch.unique(groups):
        ix=torch.nonzero(groups==group).flatten();y=labels[ix]
        if len(ix)==2 and y.sum()==1:
            terms.append(torch.nn.functional.softplus(-(scores[ix][y.bool()][0]-scores[ix][~y.bool()][0])))
    return torch.stack(terms).mean() if terms else scores.new_zeros(())


def run(clean_root, corruption_root, output_root, seeds=(17,29,43), steps=60):
    clean_root,corruption_root,output=map(Path,(clean_root,corruption_root,output_root));device=torch.device('cuda:0');records=[];conditions=('clean_only','mixed_corruption','curriculum_severity','frame_dropout_augmentation','timestamp_jitter_augmentation','pairwise_ranking_loss');eval_variants=('clean','gaussian_severe','random_dropout','timestamp_jitter','last_frame_dropout','observation_latency','partial_coordinates')
    for task in TASK_IDS:
        rows=_rows(clean_root/f'{task}_metadata.jsonl');train,val=_idx(rows,'train'),_idx(rows,'val');labels=np.load(clean_root/f'{task}_labels.npz')['success'].astype(np.float32);pairs=np.asarray([r['signed_group'] for r in rows]);values={'clean':load_model_inputs(clean_root/f'{task}_model_inputs.npz')}
        for variant in (set(eval_variants)|{'gaussian_mild','gaussian_medium'})-{'clean'}: values[variant]=load_model_inputs(corruption_root/variant/f'{task}_model_inputs.npz')
        batches={name:_batch(value,device) for name,value in values.items()};d=values['clean']['history'].shape[2]+3;a=values['clean']['candidate_actions'].shape[1]*3+3;target=torch.from_numpy(labels[train]).to(device);group_values=np.asarray([pairs[i] for i in train]);by_group=defaultdict(list)
        for local,group in enumerate(group_values): by_group[group].append(local)
        pos_ix=[];neg_ix=[]
        for members in by_group.values():
            if len(members)==2:
                positive=[i for i in members if labels[train[i]]==1]
                negative=[i for i in members if labels[train[i]]==0]
                if len(positive)==len(negative)==1: pos_ix.append(positive[0]);neg_ix.append(negative[0])
        pair_pos=torch.tensor(pos_ix,device=device);pair_neg=torch.tensor(neg_ix,device=device)
        source_map={'clean_only':('clean',),'mixed_corruption':('clean','gaussian_medium','random_dropout','timestamp_jitter'),'frame_dropout_augmentation':('clean','random_dropout'),'timestamp_jitter_augmentation':('clean','timestamp_jitter'),'pairwise_ranking_loss':('clean',)}
        for condition in conditions:
            for seed in seeds:
                torch.manual_seed(seed);model=GRUTemporal(d,a).to(device);opt=torch.optim.AdamW(model.parameters(),lr=.015)
                if condition!='curriculum_severity':
                    names=source_map[condition];view={k:torch.cat([_view(batches[name],train)[k] for name in names]) for k in batches['clean']};y=torch.cat([target for _ in names])
                for step in range(steps):
                    if condition=='curriculum_severity':
                        name=('clean','gaussian_mild','gaussian_medium')[min(2,3*step//steps)];view=_view(batches[name],train);y=target
                    scores=model(view);loss=nn.functional.binary_cross_entropy_with_logits(scores,y)
                    if condition=='pairwise_ranking_loss': loss=loss+0.5*torch.nn.functional.softplus(-(scores[pair_pos]-scores[pair_neg])).mean()
                    opt.zero_grad();loss.backward();opt.step()
                with torch.no_grad():
                    for variant in eval_variants:
                        score=model(batches[variant]).cpu().numpy();records.append(dict(task=task,condition=condition,seed=seed,variant=variant,split='val',pair_order_accuracy=pair_order(labels[val],score[val],pairs[val])))
    summary=[]
    for condition in conditions:
        items=[r for r in records if r['condition']==condition];summary.append(dict(condition=condition,validation_mean=float(np.mean([r['pair_order_accuracy'] for r in items])),observations=len(items)))
    model=GRUTemporal(values['clean']['history'].shape[2]+3,values['clean']['candidate_actions'].shape[1]*3+3);report=dict(protocol='three-seed validation-only; final selected recipe unchanged; curriculum is sequential clean/mild/medium phases',conditions=list(conditions),records=records,summary=summary,gru_parameter_count=sum(p.numel() for p in model.parameters()),seeds=list(seeds),steps=steps)
    output.mkdir(parents=True,exist_ok=True);(output/'training_conditions_validation.json').write_text(json.dumps(report,indent=2));return report
