"""Lightweight fair RGB-D visual controls for the frozen 4V pilot protocol."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from actmask.data.milestone4v_visual_pilot import TASKS


def _rows(path: Path): return [json.loads(line) for line in path.read_text().splitlines()]


def _features(root: Path, task: str, corruption: str | None = None, level: float = 0.0):
    h = np.load(root / f"{task}_histories.npz")
    rgb = h["rgb"].astype(np.float32) / 255.0
    depth = h["depth_mm"].astype(np.float32) / 1000.0
    if corruption:
        # histories 2*w and 2*w+1 are counterfactual partners: shared world
        # seed ensures a corruption never leaks the branch label.
        for index in range(len(rgb)):
            rng = np.random.default_rng(104729 + index // 2)
            if corruption == "depth_gaussian_noise": depth[index] += rng.normal(0, level, depth[index].shape)
            elif corruption == "depth_quantization": depth[index] = np.round(depth[index] / level) * level
            elif corruption == "point_dropout": depth[index][rng.random(depth[index].shape) < level] = 0
            elif corruption == "burst_frame_dropout": depth[index, 1:1 + int(level)] = 0; rgb[index, 1:1 + int(level)] = 0
            elif corruption == "image_occlusion":
                side = int(128 * level); y = 64 - side // 2; rgb[index, :, y:y+side, y:y+side] = 0; depth[index, :, y:y+side, y:y+side] = 0
            elif corruption == "appearance_variation": rgb[index] = np.clip(rgb[index] * (1 - level) + level * 0.5, 0, 1)
    # fixed 8x8 average pooling: no segmentation, IDs, pretrained encoder, or metadata
    x = np.concatenate((rgb, depth[..., None]), axis=-1).reshape(len(rgb), 6, 16, 8, 16, 8, 4).mean((3, 5))
    return x.reshape(len(rgb), 6, -1)


class _Static(nn.Module):
    def __init__(self, n): super().__init__(); self.net = nn.Sequential(nn.Linear(n, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, x): return self.net(x).squeeze(-1)


class _GRU(nn.Module):
    def __init__(self, n, action):
        super().__init__(); self.gru = nn.GRU(n, 64, batch_first=True); self.head = nn.Sequential(nn.Linear(64 + action, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, x, action): return self.head(torch.cat((self.gru(x)[0][:, -1], action), 1)).squeeze(-1)


def _auc(y, score):
    p = y.astype(bool); npos=int(p.sum()); nneg=len(y)-npos
    if not npos or not nneg: return float("nan")
    order=np.argsort(score, kind="mergesort"); rank=np.empty(len(y)); rank[order]=np.arange(1,len(y)+1)
    return float((rank[p].sum()-npos*(npos+1)/2)/(npos*nneg))


def _pair_order(y, score, rows):
    groups=defaultdict(list)
    for i,row in enumerate(rows): groups[row["pair_group"]].append(i)
    values=[]
    for m in groups.values():
        if len(m)==2 and y[m[0]] != y[m[1]]:
            good,bad=(m[0],m[1]) if y[m[0]] else (m[1],m[0]); values.append(float(score[good]>score[bad]) + .5*float(score[good]==score[bad]))
    return float(np.mean(values)), len(values)


def _topk(y, score, rows, k):
    groups=defaultdict(list)
    for i,row in enumerate(rows): groups[row["world_id"]].append(i)
    return float(np.mean([y[np.asarray(m)[np.argsort(-score[m])[:k]]].any() for m in groups.values()]))


def _train(kind, frames, actions, labels, train, test, seed):
    torch.manual_seed(seed); device=torch.device("cuda")
    # Fit normalization on training worlds only. This is essential because RGB
    # is in [0,1] while depth has a different physical scale.
    fm=frames[train].mean((0, 1), keepdims=True); fs=frames[train].std((0, 1), keepdims=True); fs[fs < 1e-5] = 1
    am=actions[train].mean(0, keepdims=True); ass=actions[train].std(0, keepdims=True); ass[ass < 1e-5] = 1
    act=torch.tensor((actions-am)/ass,device=device); x=torch.tensor((frames-fm)/fs,device=device); y=torch.tensor(labels,dtype=torch.float32,device=device)
    if kind=="static": model=_Static(x.shape[2]+act.shape[1]).to(device); fn=lambda z: model(torch.cat((z[:,-1],act),1))
    elif kind=="unordered": model=_Static(x.shape[2]+act.shape[1]).to(device); fn=lambda z: model(torch.cat((z.mean(1),act),1))
    elif kind=="action_only": model=_Static(act.shape[1]).to(device); fn=lambda z: model(act)
    else: model=_GRU(x.shape[2],act.shape[1]).to(device); fn=lambda z: model(z,act)
    opt=torch.optim.AdamW(model.parameters(),lr=.01,weight_decay=1e-4)
    ti=torch.tensor(train,device=device)
    for _ in range(120):
        loss=nn.functional.binary_cross_entropy_with_logits(fn(x)[ti],y[ti]); opt.zero_grad();loss.backward();opt.step()
    with torch.no_grad(): return torch.sigmoid(fn(x)[torch.tensor(test,device=device)]).cpu().numpy()


def run(root: str | Path):
    root=Path(root); report={"schema":"milestone4v-visual-baselines-v1","seeds":[17,29,43],"tasks":{}}
    for task in TASKS:
        histories=_features(root,task); rows=_rows(root/f"{task}_candidates.jsonl"); y=np.load(root/f"{task}_labels.npz")["success"].astype(np.float32)
        refs=np.asarray([r["history_ref"] for r in rows]); frames=histories[refs]; actions=np.asarray([r["candidate_actions"] for r in rows],dtype=np.float32).reshape(len(rows),-1)
        train=np.asarray([i for i,r in enumerate(rows) if r["split"]=="train"]); test=np.asarray([i for i,r in enumerate(rows) if r["split"]=="test"])
        task_report={}
        for kind in ("action_only","static","unordered","ordered_gru"):
            records=[]
            for seed in report["seeds"]:
                score=_train(kind,frames,actions,y,train,test,seed); ty=y[test]; tr=[rows[i] for i in test]; pair,n=_pair_order(ty,score,tr)
                records.append({"pair_order":pair,"pair_count":n,"top1_success":_topk(ty,score,tr,1),"top3_recall":_topk(ty,score,tr,3),"roc_auc":_auc(ty,score)})
            task_report[kind]={key:float(np.mean([r[key] for r in records])) for key in records[0]}
        report["tasks"][task]=task_report
    (root/"visual_baseline_report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    return report


if __name__=="__main__":
    p=Path(__file__).resolve().parents[2]/"outputs/actmask/milestone4v_visual_pilot/visual_pilot"; print(json.dumps(run(p),indent=2,sort_keys=True))
