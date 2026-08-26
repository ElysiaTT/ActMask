"""Small observable-only temporal classifiers used by Milestone 3R."""
from __future__ import annotations
import torch
from torch import nn

def _flat(batch):
    h=batch['history']; extra=torch.cat((batch['timestamps'][...,None],batch['visibility'],batch['observation_confidence']),-1)
    return torch.cat((h,extra),-1), torch.cat((batch['candidate_actions'].flatten(1),batch['tcp_state']),1)
class OrderedTemporalMLP(nn.Module):
 def __init__(self,dim,action_dim,hidden=48):
  super().__init__();self.net=nn.Sequential(nn.Linear(dim+action_dim,hidden),nn.ReLU(),nn.Linear(hidden,1))
 def forward(self,b):h,a=_flat(b);return self.net(torch.cat((h.flatten(1),a),1)).squeeze(-1)
class GRUTemporal(nn.Module):
 def __init__(self,input_dim,action_dim,hidden=32):
  super().__init__();self.gru=nn.GRU(input_dim,hidden,batch_first=True);self.out=nn.Sequential(nn.Linear(hidden+action_dim,32),nn.ReLU(),nn.Linear(32,1))
 def forward(self,b):h,a=_flat(b);_,z=self.gru(h);return self.out(torch.cat((z[-1],a),1)).squeeze(-1)
class TemporalConv1D(nn.Module):
 def __init__(self,input_dim,action_dim,hidden=32):
  super().__init__();self.conv=nn.Sequential(nn.Conv1d(input_dim,hidden,3,padding=1),nn.ReLU(),nn.Conv1d(hidden,hidden,3,padding=1),nn.ReLU());self.out=nn.Linear(hidden+action_dim,1)
 def forward(self,b):h,a=_flat(b);z=self.conv(h.transpose(1,2)).mean(-1);return self.out(torch.cat((z,a),1)).squeeze(-1)
class MaskedHistoryMLP(OrderedTemporalMLP):
 def forward(self,b):
  copied={k:v for k,v in b.items()};copied['history']=b['history']*b['visibility'];return super().forward(copied)
class StaticMLP(nn.Module):
 def __init__(self,dim,action_dim):super().__init__();self.net=nn.Sequential(nn.Linear(dim+action_dim,32),nn.ReLU(),nn.Linear(32,1))
 def forward(self,b):_,a=_flat(b);return self.net(torch.cat((b['history'][:,-1],a),1)).squeeze(-1)
class ActionMLP(nn.Module):
 def __init__(self,action_dim):super().__init__();self.net=nn.Sequential(nn.Linear(action_dim,24),nn.ReLU(),nn.Linear(24,1))
 def forward(self,b):_,a=_flat(b);return self.net(a).squeeze(-1)
class UnorderedHistoryMLP(OrderedTemporalMLP):
 def forward(self,b):
  copied={k:v for k,v in b.items()};copied['history']=torch.sort(b['history'],dim=1).values;return super().forward(copied)
