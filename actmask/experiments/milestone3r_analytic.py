"""Observable-only lightweight signed-motion estimators for 3R."""
from __future__ import annotations
import numpy as np

def _valid(mask): return mask.squeeze(-1).astype(bool) if mask.ndim==3 else mask.astype(bool)
def last_two_frame(history,timestamps,visibility):
    m=_valid(visibility); out=np.zeros((len(history),history.shape[2]),np.float32)
    for i in range(len(history)):
        ids=np.flatnonzero(m[i]);
        if len(ids)>=2: out[i]=(history[i,ids[-1]]-history[i,ids[-2]])/max(timestamps[i,ids[-1]]-timestamps[i,ids[-2]],1e-4)
    return out
def multi_frame_linear(history,timestamps,visibility,robust=False):
    m=_valid(visibility); out=np.zeros((len(history),history.shape[2]),np.float32)
    for i in range(len(history)):
        ids=np.flatnonzero(m[i]);
        if len(ids)<2: continue
        x=timestamps[i,ids]; y=history[i,ids]; keep=np.ones(len(ids),bool)
        for _ in range(2 if robust else 1):
            a=np.stack((x[keep],np.ones(keep.sum())),1); coef=np.linalg.lstsq(a,y[keep],rcond=None)[0]; residual=np.linalg.norm(a@coef-y[keep],axis=1)
            if robust and len(residual)>2: keep=np.abs(residual-np.median(residual))<=np.quantile(np.abs(residual-np.median(residual)),.8)
        out[i]=coef[0]
    return out
def constant_acceleration(history,timestamps,visibility):
    m=_valid(visibility); out=np.zeros((len(history),history.shape[2]),np.float32)
    for i in range(len(history)):
        ids=np.flatnonzero(m[i]);
        if len(ids)<3: out[i]=last_two_frame(history[i:i+1],timestamps[i:i+1],visibility[i:i+1])[0];continue
        x=timestamps[i,ids]; x=x-x[-1]; a=np.stack((x*x,x,np.ones(len(x))),1); out[i]=np.linalg.lstsq(a,history[i,ids],rcond=None)[0][1]
    return out
def alpha_beta(history,timestamps,visibility,alpha=.72,beta=.18):
    m=_valid(visibility); out=np.zeros((len(history),history.shape[2]),np.float32)
    for i in range(len(history)):
        pos=history[i,0].copy(); vel=np.zeros_like(pos)
        for t in range(1,history.shape[1]):
            dt=max(timestamps[i,t]-timestamps[i,t-1],1e-4); pred=pos+vel*dt
            if m[i,t]: r=history[i,t]-pred;pos=pred+alpha*r;vel=vel+(beta/dt)*r
            else: pos=pred
        out[i]=vel
    return out
def kalman(history,timestamps,visibility,acceleration=False,q=.01,r=.02):
    # independent dimensions, standard linear Gaussian update
    m=_valid(visibility); n,d=history.shape[0],history.shape[2]; out=np.zeros((n,d),np.float32)
    order=3 if acceleration else 2
    for i in range(n):
        state=np.zeros((d,order));state[:,0]=history[i,0];P=np.broadcast_to(np.eye(order),(d,order,order)).copy()
        for t in range(1,history.shape[1]):
            dt=max(timestamps[i,t]-timestamps[i,t-1],1e-4);F=np.array([[1,dt,.5*dt*dt],[0,1,dt],[0,0,1]]) if acceleration else np.array([[1,dt],[0,1]])
            state=state@F.T;P=F@P@F.T+q*np.eye(order)
            if m[i,t]:
                H=np.zeros(order);H[0]=1;S=P[:,0,0]+r;K=P[:,:,0]/S[:,None];res=history[i,t]-state[:,0];state+=K*res[:,None];P=(np.eye(order)-K[:,:,None]*H)@P
        out[i]=state[:,1]
    return out
def exponential_smoothing(history,timestamps,visibility,gamma=.55):
    raw=last_two_frame(history,timestamps,visibility);out=np.zeros_like(raw);out[0]=raw[0]
    # per-history recurrent smoothing, not across examples
    m=_valid(visibility)
    for i in range(len(history)):
        v=np.zeros(history.shape[2]);prev=None
        for t in range(1,history.shape[1]):
            if m[i,t] and m[i,t-1]:
                z=(history[i,t]-history[i,t-1])/max(timestamps[i,t]-timestamps[i,t-1],1e-4);v=z if prev is None else gamma*z+(1-gamma)*v;prev=t
        out[i]=v
    return out
ESTIMATORS={"LastTwoFrameFiniteDifference":last_two_frame,"MultiFrameLinearVelocity":multi_frame_linear,"RobustLinearVelocity":lambda h,t,v:multi_frame_linear(h,t,v,True),"ConstantAccelerationFit":constant_acceleration,"AlphaBetaFilter":alpha_beta,"KalmanFilterConstantVelocity":kalman,"KalmanFilterConstantAcceleration":lambda h,t,v:kalman(h,t,v,True),"ExponentialSmoothingVelocity":exponential_smoothing}
