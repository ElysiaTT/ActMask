import numpy as np
from actmask.experiments.milestone3r_analytic import ESTIMATORS
def test_irregular_time_estimators_recover_linear_velocity():
 t=np.array([[0.,.1,.4,.9]],np.float32);v=np.array([.3,-.2],np.float32);h=(t[:,:,None]*v)[...];mask=np.ones((1,4,1),np.float32)
 for name in ("LastTwoFrameFiniteDifference","MultiFrameLinearVelocity","RobustLinearVelocity","ConstantAccelerationFit","KalmanFilterConstantVelocity"):
  got=ESTIMATORS[name](h,t,mask)[0];assert np.allclose(got,v,atol=.08),name
