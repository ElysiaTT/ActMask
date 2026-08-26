import numpy as np
from actmask.experiments.milestone3r_ranking import _metrics


def test_candidate_prefix_alignment_and_metrics():
    rows=[dict(task='t',world_id=4,candidate_id=i,signed_group=f'g{i//2}') for i in range(6)]
    labels=np.array([1,0,1,0,1,0],np.float32);scores=np.array([3,0,2,0,1,0.],np.float32)
    result,take,groups=_metrics(rows,np.arange(6),labels,scores,5)
    assert take.tolist()==[0,1,2,3,4] and len(groups)==5 and result['worlds']==1
    assert result['top1_success']==1 and result['top3_recall']==1
