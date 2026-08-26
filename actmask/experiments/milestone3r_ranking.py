"""Frozen-candidate C5/C10/C20 ranking evaluation for the selected 3R GRU."""
from __future__ import annotations

import json, time
from pathlib import Path
import numpy as np
import torch
from torch import nn

from actmask.data.maniskill_pilot import load_model_inputs
from actmask.data.milestone3q_signed import TASK_IDS
from actmask.experiments.milestone3r_analytic_gpu import _last_two
from actmask.experiments.milestone3r_metrics import calibration_error, pair_order, paired_bootstrap, ranking, roc_auc
from actmask.models.milestone3r_temporal import GRUTemporal


def _rows(path): return [json.loads(x) for x in Path(path).read_text().splitlines()]
def _indices(rows, split): return np.asarray([i for i,r in enumerate(rows) if r['regime']=='signed_order' and r['world_id']%5=={'train':0,'val':3,'test':4}[split]])
def _batch(value, device): return {k:torch.from_numpy(value[k]).float().to(device) for k in ('history','timestamps','visibility','observation_confidence','candidate_actions','tcp_state')}
def _groups(rows, indices, count): return np.asarray([f"{r['task']}:{r['world_id']}" for i,r in enumerate(rows) if i in set(indices) and r['candidate_id']<count])


def _metrics(rows, indices, labels, scores, count):
    take=np.asarray([i for i in indices if rows[i]['candidate_id']<count]); groups=np.asarray([f"{rows[i]['task']}:{rows[i]['world_id']}" for i in take]); y=labels[take];s=scores[take]
    result=ranking(y,s,groups);result.update(pairwise_ranking_accuracy=pair_order(y,s,np.asarray([rows[i]['signed_group'] for i in take])), roc_auc=roc_auc(y,s), calibration_error=calibration_error(y,s), candidate_count=count, worlds=int(len(np.unique(groups))))
    return result, take, groups


def run(clean_root, corruption_root, output_root, seeds=(17,29,43,59,71), steps=60):
    started_total=time.perf_counter();clean_root,corruption_root,output=map(Path,(clean_root,corruption_root,output_root));manifest=json.loads((corruption_root/'corruption_manifest.json').read_text());variants=['clean']+[x['corruption_id'] for x in manifest['specifications']];device=torch.device('cuda:0');records=[];bootstrap_values=[];training_seconds=[];peak_memory=0;checkpoint_bytes=[];latency_ms=[]
    for task in TASK_IDS:
        rows=_rows(clean_root/f'{task}_metadata.jsonl');train=_indices(rows,'train');test=_indices(rows,'test');labels=np.load(clean_root/f'{task}_labels.npz')['success'].astype(np.float32);clean=load_model_inputs(clean_root/f'{task}_model_inputs.npz');all_values={variant:(clean if variant=='clean' else load_model_inputs(corruption_root/variant/f'{task}_model_inputs.npz')) for variant in variants};batches={variant:_batch(value,device) for variant,value in all_values.items()};d=clean['history'].shape[2]+3;a=clean['candidate_actions'].shape[1]*3+3;ensemble={variant:[] for variant in variants}
        train_batch={k:value[torch.from_numpy(train).to(device)] for k,value in batches['clean'].items()};target=torch.from_numpy(labels[train]).to(device)
        for seed in seeds:
            torch.manual_seed(seed);model=GRUTemporal(d,a).to(device);opt=torch.optim.AdamW(model.parameters(),lr=.015);torch.cuda.reset_peak_memory_stats(device);started=time.perf_counter()
            for _ in range(steps):
                loss=nn.functional.binary_cross_entropy_with_logits(model(train_batch),target);opt.zero_grad();loss.backward();opt.step()
            torch.cuda.synchronize(device);training_seconds.append(time.perf_counter()-started);peak_memory=max(peak_memory,int(torch.cuda.max_memory_allocated(device)));checkpoint_bytes.append(sum(x.numel()*x.element_size() for x in model.state_dict().values()));model.eval()
            with torch.no_grad():
                for variant,batch in batches.items(): ensemble[variant].append(model(batch).cpu().numpy())
            if seed == seeds[-1]:
                first_world=rows[test[0]]['world_id'];rank_ix=np.asarray([i for i in test if rows[i]['world_id']==first_world and rows[i]['candidate_id']<20]);rank_batch={k:v[torch.from_numpy(rank_ix).to(device)] for k,v in batches['clean'].items()}
                for _ in range(20): model(rank_batch)
                torch.cuda.synchronize(device)
                for _ in range(100):
                    begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);begin.record();model(rank_batch);end.record();end.synchronize();latency_ms.append(begin.elapsed_time(end))
        for variant in variants:
            learned=np.mean(ensemble[variant],axis=0);base=_last_two(batches[variant]['history'],batches[variant]['timestamps'],batches[variant]['visibility'])[:,1].cpu().numpy()
            for count in (5,10,20):
                for method,score in (('SelectedGRUFiveSeed',learned),('LastTwoFrameFiniteDifference',base)):
                    metric,take,groups=_metrics(rows,test,labels,score,count);records.append(dict(task=task,variant=variant,method=method,**metric))
                l,_take,groups=_metrics(rows,test,labels,learned,count);b,_take,_=_metrics(rows,test,labels,base,count)
                # one paired top-1 difference per frozen scene/world
                for group in np.unique(groups):
                    ix=np.flatnonzero(groups==group); ys=labels[take][ix];ls=learned[take][ix];bs=base[take][ix];bootstrap_values.append(dict(task=task,variant=variant,count=count,group=group,delta=float(ys[np.argmax(ls)]-ys[np.argmax(bs)])))
    aggregate=[]
    for variant in variants:
        for count in (5,10,20):
            for method in ('SelectedGRUFiveSeed','LastTwoFrameFiniteDifference'):
                items=[r for r in records if r['variant']==variant and r['candidate_count']==count and r['method']==method]
                aggregate.append(dict(variant=variant,candidate_count=count,method=method,**{k:float(np.mean([r[k] for r in items])) for k in ('top1_success','top3_recall','ranking_ap','corrected_utility_gain_ndcg','normalized_regret','failure_when_success_exists','pairwise_ranking_accuracy','roc_auc','calibration_error')}))
    boot=[]
    for count in (5,10,20):
        rows=[r for r in bootstrap_values if r['count']==count];boot.append(dict(candidate_count=count,**paired_bootstrap(np.asarray([r['delta'] for r in rows]),np.asarray([r['group'] for r in rows]))))
    report=dict(protocol='frozen signed-order test worlds; fixed candidate-id prefixes C5/C10/C20; selected GRU five-seed score ensemble',records=records,aggregate=aggregate,top1_paired_bootstrap=boot,efficiency=dict(training_seconds_per_seed=training_seconds,median_training_seconds=float(np.median(training_seconds)),checkpoint_bytes=int(np.median(checkpoint_bytes)),peak_gpu_memory_bytes=peak_memory,c20_inference_latency_ms_median=float(np.median(latency_ms)),c20_inference_latency_ms_p95=float(np.quantile(latency_ms,.95)),c20_candidates_per_second=float(20_000_000/sum(latency_ms)),total_wall_seconds=time.perf_counter()-started_total),seeds=list(seeds),steps=steps)
    output.mkdir(parents=True,exist_ok=True);(output/'ranking_c5_c10_c20.json').write_text(json.dumps(report,indent=2));return report
