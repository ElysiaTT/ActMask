"""Five-seed fair correspondence comparison; oracle identity is diagnostic only."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn

from actmask.data.milestone3r_ambiguity import AMBIGUITY_TASKS
from actmask.experiments.milestone3r_metrics import pair_order, ranking
from actmask.models.milestone3r_temporal import GRUTemporal


def _rows(path): return [json.loads(x) for x in Path(path).read_text().splitlines()]
def _split(rows, kind): return np.asarray([i for i,r in enumerate(rows) if r['regime']=='signed_order' and r['world_id']%5=={'train':0,'test':4}[kind]])
def _batch(history, visibility, common, device): return dict(history=torch.from_numpy(history).float().to(device),visibility=torch.from_numpy(visibility).float().to(device),observation_confidence=torch.from_numpy(visibility).float().to(device),timestamps=torch.from_numpy(common['timestamps']).float().to(device),candidate_actions=torch.from_numpy(common['candidate_actions']).float().to(device),tcp_state=torch.from_numpy(common['tcp_state']).float().to(device))


def run(clean_root, ambiguity_root, output_root, seeds=(17,29,43,59,71), steps=60):
    clean_root,ambiguity_root,output=map(Path,(clean_root,ambiguity_root,output_root));device=torch.device('cuda:0');records=[]
    for task in AMBIGUITY_TASKS:
        rows=_rows(clean_root/f'{task}_metadata.jsonl');train,test=_split(rows,'train'),_split(rows,'test');labels=np.load(clean_root/f'{task}_labels.npz')['success'].astype(np.float32);common=np.load(ambiguity_root/f'{task}_ambiguity_inputs.npz');objects=common['object_history'];objvis=common['object_visibility'];
        variants={
            'oracle_identity_diagnostic':(objects[:,:,0],objvis[:,:,0]),
            'nearest_neighbor_fair':(common['nearest_neighbor_history'],objvis.max(2)),
            'soft_correspondence_fair':(common['soft_correspondence_history'],objvis.max(2)),
            'no_identity_set_fair':(objects.reshape(len(objects),objects.shape[1],-1),objvis.reshape(len(objects),objvis.shape[1],-1)),
        }
        for variant,(history,visibility) in variants.items():
            batch=_batch(history,visibility,common,device);d=history.shape[2]+2*visibility.shape[2]+1;a=common['candidate_actions'].shape[1]*3+3;train_batch={k:v[torch.from_numpy(train).to(device)] for k,v in batch.items()};target=torch.from_numpy(labels[train]).to(device);scores=[]
            for seed in seeds:
                torch.manual_seed(seed);model=GRUTemporal(d,a).to(device);opt=torch.optim.AdamW(model.parameters(),lr=.015)
                for _ in range(steps):
                    loss=nn.functional.binary_cross_entropy_with_logits(model(train_batch),target);opt.zero_grad();loss.backward();opt.step()
                with torch.no_grad():scores.append(model(batch).cpu().numpy())
            score=np.mean(scores,0);pairs=np.asarray([r['signed_group'] for r in rows]);groups=np.asarray([f"{task}:{rows[i]['world_id']}" for i in test]);metrics=ranking(labels[test],score[test],groups);metrics['pair_order_accuracy']=pair_order(labels[test],score[test],pairs[test]);records.append(dict(task=task,variant=variant,diagnostic_only=variant.startswith('oracle'),**metrics))
    oracle=np.mean([r['top1_success'] for r in records if r['variant']=='oracle_identity_diagnostic']);estimated=np.mean([r['top1_success'] for r in records if r['variant']=='nearest_neighbor_fair']);report=dict(records=records,aggregate={variant:float(np.mean([r['top1_success'] for r in records if r['variant']==variant])) for variant in {r['variant'] for r in records}},correspondence_robustness_gap=float(oracle-estimated),definition='oracle Top1 minus nearest-neighbor estimated Top1; oracle is excluded from fair comparisons',seeds=list(seeds),steps=steps)
    output.mkdir(parents=True,exist_ok=True);(output/'correspondence_five_seed.json').write_text(json.dumps(report,indent=2));return report
