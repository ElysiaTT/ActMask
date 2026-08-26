import torch
from actmask.models.milestone3r_temporal import OrderedTemporalMLP,GRUTemporal,TemporalConv1D,MaskedHistoryMLP,StaticMLP,ActionMLP,UnorderedHistoryMLP
def test_all_3r_lightweight_models_accept_only_observable_batch():
 b=dict(history=torch.zeros(3,4,6),timestamps=torch.zeros(3,4),visibility=torch.ones(3,4,1),observation_confidence=torch.ones(3,4,1),candidate_actions=torch.zeros(3,5,3),tcp_state=torch.zeros(3,3)); ad=18;id=9
 models=[OrderedTemporalMLP(4*id,ad),GRUTemporal(id,ad),TemporalConv1D(id,ad),MaskedHistoryMLP(4*id,ad),StaticMLP(6,ad),ActionMLP(ad),UnorderedHistoryMLP(4*id,ad)]
 assert all(m(b).shape==(3,) for m in models)
