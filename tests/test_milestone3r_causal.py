import torch
from actmask.experiments.milestone3r_causal import _alter


def test_reverse_and_final_frame_causal_constructions():
    batch={'history':torch.tensor([[[1.],[2.],[3.],[4.]]]),'timestamps':torch.arange(4).view(1,4).float(),'visibility':torch.ones(1,4,1),'observation_confidence':torch.ones(1,4,1),'candidate_actions':torch.zeros(1,2,3),'tcp_state':torch.zeros(1,3)}
    assert _alter(batch,'exact_reversed')['history'][0,:,0].tolist()==[4,3,2,1]
    assert _alter(batch,'duplicated_final_frame')['history'][0,:,0].tolist()==[4,4,4,4]
