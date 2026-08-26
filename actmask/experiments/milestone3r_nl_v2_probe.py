"""Diversity, swap, OOD and fair-model validity evaluation for 3R-NL-v2."""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from torch import nn

from actmask.data.maniskill_pilot import load_model_inputs
from actmask.experiments.milestone3r_analytic import ESTIMATORS
from actmask.experiments.milestone3r_metrics import pair_order
from actmask.experiments.milestone3r_nl_v2 import CONFIG_HASH, swap_history
from actmask.models.milestone3r_temporal import ActionMLP, GRUTemporal, OrderedTemporalMLP, StaticMLP, TemporalConv1D, UnorderedHistoryMLP

INPUTS=("history","timestamps","visibility","observation_confidence","candidate_actions","tcp_state")
FACTORIES={
 "GRU":lambda f,a,h:GRUTemporal(f,a), "TemporalConv1D":lambda f,a,h:TemporalConv1D(f,a),
 "OrderedTemporalMLP":lambda f,a,h:OrderedTemporalMLP(f*h,a), "UnorderedHistoryMLP":lambda f,a,h:UnorderedHistoryMLP(f*h,a),
 "StaticMLP":lambda f,a,h:StaticMLP(f-3,a), "ActionMLP":lambda f,a,h:ActionMLP(a),
}

def _lines(path): return [json.loads(x) for x in Path(path).read_text().splitlines()]
def _batch(value,dev): return {k:torch.from_numpy(value[k]).float().to(dev) for k in INPUTS}
def _take(batch,ix):
 index=torch.from_numpy(np.asarray(ix)).to(next(iter(batch.values())).device);return {k:v[index] for k,v in batch.items()}
def _split(rows,name): return np.asarray([i for i,r in enumerate(rows) if r["split"]==name])
def _velocity_score(v,a): return (v[:,-3:]*a.sum(1)).sum(1)

def diversity(root):
    outcomes=_lines(Path(root)/"candidate_outcomes.jsonl"); worlds=defaultdict(list)
    for r in outcomes: worlds[(r["family"],r["mechanism"],r["world_id"],r["condition"])].append(r)
    details=[]
    for key,items in worlds.items():
        items=sorted(items,key=lambda x:x["candidate_id"]); b0=np.asarray([x["branch0_success"] for x in items],float);b1=np.asarray([x["branch1_success"] for x in items],float)
        entropy=lambda p: float(-(p*np.log2(max(p,1e-12))+(1-p)*np.log2(max(1-p,1e-12)))) if 0<p<1 else 0.
        details.append(dict(family=key[0],mechanism=key[1],world_id=key[2],condition=key[3],candidates=len(items),branch0_success=int(b0.sum()),branch1_success=int(b1.sum()),branch0_fraction=float(b0.mean()),branch1_fraction=float(b1.mean()),label_entropy=float((entropy(b0.mean())+entropy(b1.mean()))/2),branch_outcome_change_rate=float((b0!=b1).mean()),pair_flip_rate=float(np.mean([x["pair_flip"] for x in items])),nondegenerate=bool(len(items)>=10 and all(.2<=x.mean()<=.8 for x in (b0,b1)) and b0.sum() not in (0,len(b0)) and b1.sum() not in (0,len(b1)))))
    candidate_rates={str(c):float(np.mean([.5*(x["branch0_success"]+x["branch1_success"]) for x in outcomes if x["candidate_id"]==c])) for c in range(max(x["candidate_id"] for x in outcomes)+1)}
    rate_mean=float(np.mean(list(candidate_rates.values())))
    report=dict(worlds=details,summary=dict(worlds=len(details),nondegenerate_world_fraction=float(np.mean([x["nondegenerate"] for x in details])),mean_branch_outcome_change_rate=float(np.mean([x["branch_outcome_change_rate"] for x in details])),mean_label_entropy=float(np.mean([x["label_entropy"] for x in details])),candidate_template_success_rates=candidate_rates,candidate_template_rate_span=float(max(candidate_rates.values())-min(candidate_rates.values())),max_candidate_index_leakage=float(max(abs(x-rate_mean) for x in candidate_rates.values())),degenerate_world_rate=float(np.mean([not x["nondegenerate"] for x in details]))))
    return report

def _fair_pair_metric(labels,scores,rows,indices):
    pair=np.asarray([rows[i]["pair_id"] for i in indices]);return pair_order(labels[indices],scores[indices],pair)

def _ranking_metrics(labels,scores,rows,indices,diversity_report):
    valid={(x["family"],x["mechanism"],x["world_id"],x["condition"]):x["nondegenerate"] for x in diversity_report["worlds"]}
    groups=defaultdict(list)
    for i in indices:
        r=rows[i];groups[(r["family"],r["mechanism"],r["world_id"],r["condition"])].append(i)
    top=[];top3=[];ndcg=[]
    for key,ix in groups.items():
        if not valid.get(key,False):continue
        ix=np.asarray(ix);order=np.argsort(-scores[ix],kind="stable");ordered=ix[order];best=scores[ordered[0]];ties=ordered[np.isclose(scores[ordered],best)]
        top.append(float(labels[ties].mean())) # expected tie-broken Top-1
        top3.append(float(labels[ordered[:3]].any()))
        relevance=labels[ordered];dcg=float((relevance/np.log2(np.arange(2,len(ordered)+2))).sum());ideal=np.sort(labels[ix])[::-1];idcg=float((ideal/np.log2(np.arange(2,len(ordered)+2))).sum());ndcg.append(dcg/max(idcg,1e-9))
    return dict(nondegenerate_worlds=len(top),top1_success=float(np.mean(top)),top3_success_recall=float(np.mean(top3)),corrected_utility_gain_ndcg=float(np.mean(ndcg)),normalized_regret=float(1-np.mean(top)))

def _swap_metrics(model,batch,labels,rows,indices):
    with torch.no_grad(): original=model(batch).cpu().numpy(); transformed={k:v.clone() for k,v in batch.items()};transformed["history"]=torch.from_numpy(swap_history(batch["history"].cpu().numpy())).to(batch["history"].device);changed=model(transformed).cpu().numpy()
    pairs=defaultdict(list)
    for i in indices:pairs[rows[i]["pair_id"]].append(i)
    correct=[]; consistency=[]
    for ix in pairs.values():
        if len(ix)!=2:continue
        p,n=(ix[0],ix[1]) if labels[ix[0]]==1 else (ix[1],ix[0])
        # T maps source index to its exact partner; use both rank exchange and
        # close score equality so static ties cannot satisfy consistency.
        correct.append(float(changed[p] < changed[n]))
        consistency.append(float(abs(changed[p]-original[n])<1e-5 and abs(changed[n]-original[p])<1e-5 and changed[p]<changed[n]))
    return dict(counterfactual_swap_accuracy=float(np.mean(correct)),score_swap_consistency=float(np.mean(consistency)))

def _corruption(batch,name):
    b={k:v.clone() for k,v in batch.items()};h=b["history"]
    if name=="last_two_only":b["history"][:,:-2]=b["history"][:,-1:].expand(-1,h.shape[1]-2,-1)
    elif name=="generic_full_reverse":
        for k in ("history","timestamps","visibility","observation_confidence"):b[k]=torch.flip(b[k],(1,))
    elif name=="early_independent_permutation":
        b["history"][:,:4]=torch.flip(b["history"][:,:4],(1,))
    else:raise ValueError(name)
    return b

def _fit(factory,batch,train,labels,seed,steps):
    torch.manual_seed(seed);m=factory().to(batch["history"].device);o=torch.optim.AdamW(m.parameters(),lr=.015);target=torch.from_numpy(labels[train]).float().to(batch["history"].device);x=_take(batch,train)
    for _ in range(steps):
        loss=nn.functional.binary_cross_entropy_with_logits(m(x),target);o.zero_grad();loss.backward();o.step()
    return m.eval()

def _load(root):
    root=Path(root);v=load_model_inputs(root/"model_inputs.npz");r=_lines(root/"metadata.jsonl");y=np.load(root/"labels.npz")["success"].astype(np.float32);return v,r,y

def ood_overlap_audit(root):
    root=Path(root)
    def values(name): return _lines(root/name/"metadata.jsonl")
    idrows=values("id");physical=values("physical_parameter_ood");temporal=values("temporal_delay_ood");held=values("held_mechanism_ood")
    def span(rows,key):
        v=[r["hidden_parameters"][key] for r in rows];return [float(min(v)),float(max(v))]
    def overlap(left,right): return max(left[0],right[0])<=min(left[1],right[1])
    result=dict(
      physical_parameter=dict(id_damping=span([r for r in idrows if r["mechanism"]=="history_identifiable_damping_drive"],"damping"),ood_damping=span(physical,"damping"),id_frequency=span([r for r in idrows if r["mechanism"]=="history_identifiable_damping_drive"],"drive_frequency"),ood_frequency=span(physical,"drive_frequency")),
      temporal_delay=dict(id_delay=span([r for r in idrows if r["mechanism"]=="hysteretic_mode_memory"],"execution_delay"),ood_delay=span(temporal,"execution_delay")),
      held_mechanism=dict(train_mechanisms=sorted(set(r["mechanism"] for r in idrows if r["mechanism"]=="history_identifiable_damping_drive")),test_mechanisms=sorted(set(r["mechanism"] for r in held)))
    )
    result["physical_parameter"]["damping_overlap"]=overlap(result["physical_parameter"]["id_damping"],result["physical_parameter"]["ood_damping"])
    result["physical_parameter"]["frequency_overlap"]=overlap(result["physical_parameter"]["id_frequency"],result["physical_parameter"]["ood_frequency"])
    result["temporal_delay"]["overlap"]=overlap(result["temporal_delay"]["id_delay"],result["temporal_delay"]["ood_delay"])
    result["held_mechanism"]["overlap"]=bool(set(result["held_mechanism"]["train_mechanisms"]) & set(result["held_mechanism"]["test_mechanisms"]))
    return result

def run(root, output, seeds=(17,29,43), steps=150):
    root,output=Path(root),Path(output);idroot=root/"id";value,rows,labels=_load(idroot)
    if set(value)!={"history","timestamps","visibility","observation_confidence","candidate_actions","nominal_action_timing","tcp_state"}:raise AssertionError("input whitelist violation")
    audit=diversity(idroot);(output.mkdir(parents=True,exist_ok=True));(output/"candidate_diversity_audit.json").write_text(json.dumps(audit,indent=2))
    device=torch.device("cuda:0");batch=_batch(value,device);train,val,test=(_split(rows,x) for x in ("train","validation","test"));f=value["history"].shape[2]+3;a=value["candidate_actions"].shape[1]*3+3;h=value["history"].shape[1]
    records=[]; analytic={}
    for name,fn in ESTIMATORS.items():
        score=_velocity_score(fn(value["history"],value["timestamps"],value["visibility"]),value["candidate_actions"]);analytic[name]=score
        for split,ix in (("validation",val),("test",test)):records.append(dict(method=name,kind="analytic",split=split,seed=None,pair_order_accuracy=_fair_pair_metric(labels,score,rows,ix)))
    all_models={}
    for name,build in FACTORIES.items():
        for seed in seeds:
            m=_fit(lambda:build(f,a,h),batch,train,labels,seed,steps);all_models[(name,seed)]=m
            with torch.no_grad():score=m(batch).cpu().numpy()
            for split,ix in (("validation",val),("test",test)):records.append(dict(method=name,kind="learned",split=split,seed=seed,pair_order_accuracy=_fair_pair_metric(labels,score,rows,ix)))
    valmean={name:float(np.mean([x["pair_order_accuracy"] for x in records if x["kind"]=="learned" and x["method"]==name and x["split"]=="validation"])) for name in FACTORIES};selected=max(("GRU","TemporalConv1D","OrderedTemporalMLP"),key=valmean.get)
    # Mean fixed-seed prediction and causal swaps on all ID test pairs.
    models=[all_models[(selected,s)] for s in seeds];
    with torch.no_grad():mean_score=np.mean([m(batch).cpu().numpy() for m in models],0)
    per_task_mechanism=[]
    for family,mechanism in sorted({(r["family"],r["mechanism"]) for r in rows}):
        ix=np.asarray([i for i in test if rows[i]["family"]==family and rows[i]["mechanism"]==mechanism])
        if len(ix):
            best=max(analytic,key=lambda name:_fair_pair_metric(labels,analytic[name],rows,ix))
            per_task_mechanism.append(dict(family=family,mechanism=mechanism,learned_pair_order_accuracy=_fair_pair_metric(labels,mean_score,rows,ix),best_analytic=best,best_analytic_pair_order_accuracy=_fair_pair_metric(labels,analytic[best],rows,ix)))
    swap=_swap_metrics(models[0],batch,labels,rows,test)
    swap_controls={name:_swap_metrics(all_models[(name,seeds[0])],batch,labels,rows,test) for name in ("UnorderedHistoryMLP","StaticMLP","ActionMLP")}
    interventions={"correct":_fair_pair_metric(labels,mean_score,rows,test)}
    for name in ("last_two_only","generic_full_reverse","early_independent_permutation"):
        with torch.no_grad():score=np.mean([m(_corruption(batch,name)).cpu().numpy() for m in models],0)
        interventions[name]=_fair_pair_metric(labels,score,rows,test)
    ranking={"dynamic":_ranking_metrics(labels,mean_score,rows,test,audit)}
    for name in ("StaticMLP","ActionMLP"):
        with torch.no_grad(): score=np.mean([all_models[(name,s)](batch).cpu().numpy() for s in seeds],0)
        ranking[name]=_ranking_metrics(labels,score,rows,test,audit)
    # Evaluate independently generated OOD sets using ID-trained models.
    ood=[]
    for axis in ("physical_parameter_ood","temporal_delay_ood"):
        value2,rows2,y2=_load(root/axis);b2=_batch(value2,device);ix=np.arange(len(y2))
        with torch.no_grad():score=np.mean([m(b2).cpu().numpy() for m in models],0)
        ood.append(dict(axis=axis,pair_order_accuracy=_fair_pair_metric(y2,score,rows2,ix)))
    # Mechanism OOD fits damping/drive ID rows only and evaluates fresh held mechanism.
    mech_train=np.asarray([i for i,r in enumerate(rows) if r["mechanism"]=="history_identifiable_damping_drive" and r["split"]=="train"])
    mech_models=[_fit(lambda:FACTORIES[selected](f,a,h),batch,mech_train,labels,s,steps) for s in seeds]
    value2,rows2,y2=_load(root/"held_mechanism_ood");b2=_batch(value2,device)
    with torch.no_grad():score=np.mean([m(b2).cpu().numpy() for m in mech_models],0)
    ood.append(dict(axis="held_mechanism_ood",pair_order_accuracy=_fair_pair_metric(y2,score,rows2,np.arange(len(y2)))))
    ranges=ood_overlap_audit(root);(output/"ood_overlap_audit.json").write_text(json.dumps(ranges,indent=2))
    report=dict(config_sha256=CONFIG_HASH,records=records,validation_selected_learned=selected,validation_mean=valmean,selected_analytic=max(analytic,key=lambda n:_fair_pair_metric(labels,analytic[n],rows,val)),candidate_diversity=audit["summary"],swap=swap,swap_controls=swap_controls,interventions=interventions,earlier_history_advantage=interventions["correct"]-interventions["last_two_only"],ranking=ranking,ood=ood,per_task_mechanism=per_task_mechanism,ood_overlap_audit=ranges,seeds=list(seeds),steps=steps,gpu_peak_allocated_mb=float(torch.cuda.max_memory_allocated()/1024**2))
    (output/"probe_evaluation.json").write_text(json.dumps(report,indent=2));return report
