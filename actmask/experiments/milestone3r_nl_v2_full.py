"""Frozen full-run assembly and five-seed evaluation for 3R-NL-v2."""
from __future__ import annotations
import hashlib, json, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch

from actmask.data.maniskill_pilot import load_model_inputs
from actmask.experiments.milestone3r_analytic import ESTIMATORS
from actmask.experiments.milestone3r_metrics import pair_order, paired_bootstrap
from actmask.experiments.milestone3r_nl_v2_probe import FACTORIES, _batch, _corruption, _fit, _lines, _load, _split, _swap_metrics, _velocity_score, diversity

CONFIG=Path('outputs/actmask/milestone3r_nl_v2/full_run_config.json')
CONFIG_HASH='12c692d032ba37bdb14761a1c63ff8ceccff72af5f037d936ebaed8748016a47'

def _hash(): return hashlib.sha256(CONFIG.read_bytes()).hexdigest()

def assemble(sources, destination):
    """Merge independently executed cells and materialize aligned C5/C10."""
    if _hash()!=CONFIG_HASH: raise RuntimeError('full v2 config hash changed')
    sources=[Path(x) for x in sources]; destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    values=[load_model_inputs(x/'model_inputs.npz') for x in sources]; labels=[np.load(x/'labels.npz') for x in sources]
    merged={key:np.concatenate([v[key] for v in values]) for key in values[0]}
    merged_labels={key:np.concatenate([v[key] for v in labels]) for key in labels[0]}
    rows=sum((_lines(x/'metadata.jsonl') for x in sources),[]); outcomes=sum((_lines(x/'candidate_outcomes.jsonl') for x in sources),[])
    np.savez_compressed(destination/'model_inputs.npz',**merged);np.savez_compressed(destination/'labels.npz',**merged_labels)
    (destination/'metadata.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows));(destination/'candidate_outcomes.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in outcomes))
    report=dict(config_sha256=CONFIG_HASH,sources=[str(x) for x in sources],examples=len(rows),pairs=len(rows)//2,candidate_outcome_records=len(outcomes),candidate_execution_added=0)
    (destination/'assembly_report.json').write_text(json.dumps(report,indent=2))
    for count in (5,10):
        keep=np.asarray([r['candidate_id']<count for r in rows]);prefix=destination.parent/f'{destination.name}_c{count}';prefix.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(prefix/'model_inputs.npz',**{k:v[keep] for k,v in merged.items()});np.savez_compressed(prefix/'labels.npz',**{k:v[keep] for k,v in merged_labels.items()})
        selected=[r for r,flag in zip(rows,keep) if flag];(prefix/'metadata.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in selected))
        (prefix/'derivation_report.json').write_text(json.dumps(dict(source=str(destination),candidate_prefix=count,examples=len(selected),simulator_executions_added=0),indent=2))
    return report

def _pair_metric(labels,scores,rows,indices): return pair_order(labels[indices],scores[indices],np.asarray([rows[i]['pair_id'] for i in indices]))

def _pair_correct(labels,scores,rows,indices):
    groups=defaultdict(list)
    for i in indices:groups[rows[i]['pair_id']].append(i)
    out={}
    for key,ix in groups.items():
        if len(ix)==2:
            p,n=(ix[0],ix[1]) if labels[ix[0]] else (ix[1],ix[0])
            out[key]=float(scores[p]>scores[n])+.5*float(np.isclose(scores[p],scores[n]))
    return out

def _rank(labels,scores,rows,indices,count):
    groups=defaultdict(list)
    for i in indices:
        if rows[i]['candidate_id']<count:groups[(rows[i]['family'],rows[i]['mechanism'],rows[i]['world_id'],rows[i]['branch'])].append(i)
    top=[];top3=[];ndcg=[];regret=[]
    for ix in groups.values():
        ix=np.asarray(ix);order=ix[np.argsort(-scores[ix],kind='stable')];mx=scores[order[0]];ties=order[np.isclose(scores[order],mx)]
        top_value=float(labels[ties].mean());top.append(top_value);top3.append(float(labels[order[:3]].any()));y=labels[order];dcg=float((y/np.log2(np.arange(2,len(y)+2))).sum());ideal=np.sort(labels[ix])[::-1];idcg=float((ideal/np.log2(np.arange(2,len(ideal)+2))).sum());ndcg.append(dcg/max(idcg,1e-9));regret.append(float(labels[ix].max()-top_value))
    return dict(contexts=len(top),top1_success=float(np.mean(top)),top3_success_recall=float(np.mean(top3)),corrected_utility_gain_ndcg=float(np.mean(ndcg)),normalized_regret=float(np.mean(regret)))

def _per_cell(labels,scores,analytic,rows,indices):
    output=[]
    for family,mech in sorted({(rows[i]['family'],rows[i]['mechanism']) for i in indices}):
        ix=np.asarray([i for i in indices if rows[i]['family']==family and rows[i]['mechanism']==mech]);best=max(analytic,key=lambda n:_pair_metric(labels,analytic[n],rows,ix));output.append(dict(family=family,mechanism=mech,learned=_pair_metric(labels,scores,rows,ix),best_analytic=best,analytic=_pair_metric(labels,analytic[best],rows,ix)))
    return output

def evaluate(root, output, diagnostic_seeds=(17,29,43), final_seeds=(17,29,43,59,71), diagnostic_steps=100, final_steps=120):
    if _hash()!=CONFIG_HASH:raise RuntimeError('full v2 config hash changed')
    root,output=Path(root),Path(output);value,rows,labels=_load(root/'id');device=torch.device('cuda:0');batch=_batch(value,device);train,val,test=(_split(rows,x) for x in ('train','validation','test'));frame=value['history'].shape[2]+3;action=value['candidate_actions'].shape[1]*3+3;horizon=value['history'].shape[1]
    analytic={};records=[]
    for name,fn in ESTIMATORS.items():
        score=_velocity_score(fn(value['history'],value['timestamps'],value['visibility']),value['candidate_actions']);analytic[name]=score
        for split,ix in (('validation',val),('test',test)):records.append(dict(method=name,kind='analytic',seed=None,split=split,pair_order_accuracy=_pair_metric(labels,score,rows,ix)))
    diagnostic={}
    for name,build in FACTORIES.items():
        for seed in diagnostic_seeds:
            model=_fit(lambda:build(frame,action,horizon),batch,train,labels,seed,diagnostic_steps);diagnostic[(name,seed)]=model
            with torch.no_grad(): score=model(batch).cpu().numpy()
            for split,ix in (('validation',val),('test',test)):records.append(dict(method=name,kind='learned_diagnostic',seed=seed,split=split,pair_order_accuracy=_pair_metric(labels,score,rows,ix)))
    vals={name:float(np.mean([r['pair_order_accuracy'] for r in records if r['kind']=='learned_diagnostic' and r['method']==name and r['split']=='validation'])) for name in FACTORIES};selected=max(('GRU','TemporalConv1D','OrderedTemporalMLP'),key=vals.get)
    final=[];final_models=[]
    for seed in final_seeds:
        m=_fit(lambda:FACTORIES[selected](frame,action,horizon),batch,train,labels,seed,final_steps);final_models.append(m)
        with torch.no_grad():score=m(batch).cpu().numpy()
        final.append(score)
        for split,ix in (('validation',val),('test',test)):records.append(dict(method=selected+'FiveSeed',kind='learned_confirmation',seed=seed,split=split,pair_order_accuracy=_pair_metric(labels,score,rows,ix)))
    mean=np.mean(final,0);best=max(analytic,key=lambda n:_pair_metric(labels,analytic[n],rows,val));lp=_pair_correct(labels,mean,rows,test);ap=_pair_correct(labels,analytic[best],rows,test);keys=np.asarray(sorted(set(lp)&set(ap)));delta=np.asarray([lp[x]-ap[x] for x in keys]);bootstrap=paired_bootstrap(delta,keys,seed=7301)
    controls={}
    for name in ('StaticMLP','ActionMLP'):
        with torch.no_grad():controls[name]=np.mean([diagnostic[(name,s)](batch).cpu().numpy() for s in diagnostic_seeds],0)
    rankings={f'C{count}':dict(dynamic=_rank(labels,mean,rows,test,count),**{name:_rank(labels,score,rows,test,count) for name,score in controls.items()}) for count in (5,10,20)}
    swap=_swap_metrics(final_models[0],batch,labels,rows,test);interventions={'correct':_pair_metric(labels,mean,rows,test)}
    for name in ('last_two_only','early_independent_permutation','generic_full_reverse'):
        with torch.no_grad():score=np.mean([m(_corruption(batch,name)).cpu().numpy() for m in final_models],0)
        interventions[name]=_pair_metric(labels,score,rows,test)
    # independent OOD against ID-trained selected model; held mechanism uses only damping/drive ID fitting
    ood=[]
    for axis in ('physical_parameter_ood','temporal_delay_ood'):
        v,r,y=_load(root/axis); b=_batch(v,device); ix=np.arange(len(y))
        with torch.no_grad():
            score=np.mean([m(b).cpu().numpy() for m in final_models],0)
        an={n:_velocity_score(fn(v['history'],v['timestamps'],v['visibility']),v['candidate_actions']) for n,fn in ESTIMATORS.items()};best_axis=max(an,key=lambda n:_pair_metric(y,an[n],r,ix));ood.append(dict(axis=axis,learned=_pair_metric(y,score,r,ix),best_analytic=best_axis,analytic=_pair_metric(y,an[best_axis],r,ix)))
    mech_train=np.asarray([i for i in train if rows[i]['mechanism']=='history_identifiable_damping_drive'])
    held_models=[_fit(lambda:FACTORIES[selected](frame,action,horizon),batch,mech_train,labels,s,final_steps) for s in final_seeds]
    v,r,y=_load(root/'held_mechanism_ood'); b=_batch(v,device); ix=np.arange(len(y))
    with torch.no_grad():
        score=np.mean([m(b).cpu().numpy() for m in held_models],0)
    an={n:_velocity_score(fn(v['history'],v['timestamps'],v['visibility']),v['candidate_actions']) for n,fn in ESTIMATORS.items()};best_axis=max(an,key=lambda n:_pair_metric(y,an[n],r,ix));ood.append(dict(axis='held_mechanism_ood',learned=_pair_metric(y,score,r,ix),best_analytic=best_axis,analytic=_pair_metric(y,an[best_axis],r,ix)))
    # per-world C20 latency after warmup on selected model
    lat=[]
    for start in range(0,min(len(test),2000),40):
        ix=test[start:start+40];torch.cuda.synchronize();t=time.perf_counter();
        with torch.no_grad():final_models[0](_batch({k:value[k][ix] for k in value if k!='nominal_action_timing'},device))
        torch.cuda.synchronize();lat.append((time.perf_counter()-t)*1000)
    data_audit=diversity(root/'id')['summary'];gate=dict(learned_analytic_advantage=bootstrap['delta']>=.05 and bootstrap['ci95'][0]>0,ood_axes=sum(x['learned']>x['analytic'] for x in ood)>=2,family_support=sum(x['learned']>x['analytic'] for x in _per_cell(labels,mean,analytic,rows,test))>=3,history_advantage=interventions['correct']-interventions['last_two_only']>=.1,swap_accuracy=swap['counterfactual_swap_accuracy']>=.8,ranking_dynamic_advantage=all(rankings[k]['dynamic']['top1_success']-max(rankings[k]['StaticMLP']['top1_success'],rankings[k]['ActionMLP']['top1_success'])>=.1 for k in rankings),c20_latency=float(np.percentile(lat,95))<=200,nondegenerate=data_audit['nondegenerate_world_fraction']>=.8)
    report=dict(full_config_sha256=CONFIG_HASH,records=records,selected_learned=selected,validation=vals,selected_analytic=best,bootstrap=bootstrap,per_task_mechanism=_per_cell(labels,mean,analytic,rows,test),ood=ood,swap=swap,interventions=interventions,history_advantage=interventions['correct']-interventions['last_two_only'],rankings=rankings,c20_latency_ms=dict(p95=float(np.percentile(lat,95)),mean=float(np.mean(lat)),samples=len(lat)),gpu_peak_allocated_mb=float(torch.cuda.max_memory_allocated()/1024**2),candidate_diversity=data_audit,gate=gate,full_gate_passed=all(gate.values()),seeds=dict(diagnostic=list(diagnostic_seeds),confirmation=list(final_seeds)))
    output.mkdir(parents=True,exist_ok=True);(output/'full_evaluation.json').write_text(json.dumps(report,indent=2));return report

def refresh_data_audit(root, output):
    """Refresh only post-hoc diversity bookkeeping after an audit-code fix.

    This never retrains a model or changes any simulator/score result.
    """
    root,output=Path(root),Path(output);path=output/'full_evaluation.json';report=json.loads(path.read_text())
    audit=diversity(root/'id')['summary'];report['candidate_diversity']=audit;report['gate']['nondegenerate']=audit['nondegenerate_world_fraction']>=.8;report['full_gate_passed']=all(report['gate'].values());report['data_audit_refresh']='C20 audit accepts >=10 candidates; all model and simulator evidence is unchanged.'
    path.write_text(json.dumps(report,indent=2));return report
