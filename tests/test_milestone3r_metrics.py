import numpy as np
from actmask.experiments.milestone3r_metrics import ranking,pair_order,paired_bootstrap
def test_ranking_and_pair_metrics():
 y=np.array([1,0,0,1]);s=np.array([2.,0.,1.,3.]);g=np.array([0,0,1,1]);r=ranking(y,s,g);assert r['top1_success']==1 and r['top3_recall']==1;assert pair_order(y,s,g)==1;assert paired_bootstrap(np.ones(4),g)['ci95']==[1.,1.]
