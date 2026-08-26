"""Batched CUDA implementations of the fixed observable 3R estimators."""
from __future__ import annotations

import torch


def _mask(visibility):
    return visibility.squeeze(-1).bool() if visibility.ndim == 3 else visibility.bool()


def _last_two(h, t, v):
    m = _mask(v); n, length, dim = h.shape; ids = torch.arange(length, device=h.device).expand(n, length)
    last = torch.where(m, ids, torch.full_like(ids, -1)).max(1).values
    prev = torch.where(m & (ids < last[:, None]), ids, torch.full_like(ids, -1)).max(1).values
    good = prev >= 0; li = last.clamp_min(0); pi = prev.clamp_min(0)
    gather = lambda x, idx: x.gather(1, idx[:, None, None].expand(-1, 1, dim)).squeeze(1)
    dt = (t.gather(1, li[:, None]).squeeze(1) - t.gather(1, pi[:, None]).squeeze(1)).clamp_min(1e-4)
    out = (gather(h, li) - gather(h, pi)) / dt[:, None]
    return out * good[:, None]


def _linear(h, t, v, robust=False):
    m = _mask(v).float()
    def fit(w):
        sx=(w*t).sum(1); sy=(w[...,None]*h).sum(1); sxx=(w*t*t).sum(1); sxy=(w[...,None]*t[...,None]*h).sum(1); count=w.sum(1)
        denom=(count*sxx-sx*sx).clamp_min(1e-8)
        slope=(count[:,None]*sxy-sx[:,None]*sy)/denom[:,None]
        intercept=(sy-slope*sx[:,None])/count.clamp_min(1)[:,None]
        return slope,intercept,count
    slope,intercept,count=fit(m)
    if robust:
        residual=torch.linalg.vector_norm(h-(slope[:,None,:]*t[:,:,None]+intercept[:,None,:]),dim=2)
        center=residual.masked_fill(m==0,float('nan')).nanmedian(1).values
        deviation=(residual-center[:,None]).abs().masked_fill(m==0,float('nan'))
        cutoff=torch.nanquantile(deviation,.8,dim=1)
        slope,_,count=fit(m*(deviation<=cutoff[:,None]))
    return slope*(count>=2)[:,None]


def _constant_acceleration(h,t,v):
    m=_mask(v).float();n,length,dim=h.shape;ids=torch.arange(length,device=h.device).expand(n,length);last=torch.where(m.bool(),ids,torch.full_like(ids,-1)).max(1).values.clamp_min(0);origin=t.gather(1,last[:,None]);x=t-origin
    a=torch.stack((x*x,x,torch.ones_like(x)),2);normal=torch.einsum('nl,nli,nlj->nij',m,a,a)+torch.eye(3,device=h.device)[None]*1e-6;rhs=torch.einsum('nl,nli,nld->nid',m,a,h)
    coef=torch.linalg.solve(normal,rhs)[:,1];return torch.where((m.sum(1)>=3)[:,None],coef,_last_two(h,t,v))


def _alpha_beta(h,t,v,alpha=.72,beta=.18):
    m=_mask(v);pos=h[:,0].clone();vel=torch.zeros_like(pos)
    for step in range(1,h.shape[1]):
        dt=(t[:,step]-t[:,step-1]).clamp_min(1e-4);pred=pos+vel*dt[:,None];res=h[:,step]-pred;pos=torch.where(m[:,step,None],pred+alpha*res,pred);vel=torch.where(m[:,step,None],vel+(beta/dt)[:,None]*res,vel)
    return vel


def _kalman(h,t,v,acceleration=False,q=.01,r=.02):
    m=_mask(v);n,_,dim=h.shape;order=3 if acceleration else 2;state=torch.zeros((n,dim,order),device=h.device);state[:,:,0]=h[:,0];p=torch.eye(order,device=h.device).expand(n,dim,order,order).clone();eye=torch.eye(order,device=h.device);H=torch.zeros(order,device=h.device);H[0]=1
    for step in range(1,h.shape[1]):
        dt=(t[:,step]-t[:,step-1]).clamp_min(1e-4);f=torch.eye(order,device=h.device).expand(n,order,order).clone();f[:,0,1]=dt
        if acceleration: f[:,0,2]=.5*dt*dt;f[:,1,2]=dt
        state=torch.matmul(state,f.transpose(-1,-2));p=torch.matmul(torch.matmul(f[:,None],p),f[:,None].transpose(-1,-2))+q*eye[None,None]
        s=p[...,0,0]+r;k=p[..., :,0]/s[...,None];res=h[:,step]-state[:,:,0];updated_state=state+k*res[...,None];updated_p=torch.matmul(eye[None,None]-k[...,None]*H[None,None,None],p)
        observed=m[:,step].float()
        state=state+observed[:,None,None]*(updated_state-state)
        p=p+observed[:,None,None,None]*(updated_p-p)
    return state[:,:,1]


def _exponential(h,t,v,gamma=.55):
    m=_mask(v);velocity=torch.zeros_like(h[:,0]);seen=torch.zeros(len(h),device=h.device,dtype=torch.bool)
    for step in range(1,h.shape[1]):
        valid=m[:,step]&m[:,step-1];z=(h[:,step]-h[:,step-1])/(t[:,step]-t[:,step-1]).clamp_min(1e-4)[:,None];candidate=torch.where(seen[:,None],gamma*z+(1-gamma)*velocity,z);velocity=torch.where(valid[:,None],candidate,velocity);seen|=valid
    return velocity


ESTIMATORS_GPU={
    'LastTwoFrameFiniteDifference':_last_two,
    'MultiFrameLinearVelocity':_linear,
    'RobustLinearVelocity':lambda h,t,v:_linear(h,t,v,True),
    'ConstantAccelerationFit':_constant_acceleration,
    'AlphaBetaFilter':_alpha_beta,
    'KalmanFilterConstantVelocity':_kalman,
    'KalmanFilterConstantAcceleration':lambda h,t,v:_kalman(h,t,v,True),
    'ExponentialSmoothingVelocity':_exponential,
}
