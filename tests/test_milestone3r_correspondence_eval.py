import numpy as np
from actmask.experiments.milestone3r_correspondence_eval import _batch


def test_set_variant_keeps_no_identity_key_and_two_objects():
    common={'timestamps':np.zeros((2,4),np.float32),'candidate_actions':np.zeros((2,3,3),np.float32),'tcp_state':np.zeros((2,3),np.float32)}
    result=_batch(np.zeros((2,4,6),np.float32),np.ones((2,4,2),np.float32),common,'cpu')
    assert set(result)=={'history','visibility','observation_confidence','timestamps','candidate_actions','tcp_state'} and result['history'].shape[-1]==6
