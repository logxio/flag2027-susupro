"""Cross-modal dual-tower metric and pair-discrimination architectures (MSM, PAEFF, ConvLoRA, VER).

Blockwise input BatchNorm1d standardizes multi-encoder embeddings within each branch.
Training uses same-gender hard-negative and cross-recording PK contrastive sampling.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
torch.set_num_threads(1)
def maxsep(n):
 # Closed-form maximum class-separation simplex target matrix P, O(n^2) allocation.
 p=np.zeros((n-1,n),np.float32)
 for r in range(n-1):
  k=n-1-r;s=np.sqrt(k*n/((k+1)*(n-1)));p[r,r]=s;p[r,r+1:]=-s/k
 return p
def opl(f,y):
 f=F.normalize(f,dim=1);same=y[:,None]==y[None,:];eye=torch.eye(len(y),dtype=torch.bool,device=f.device);pos=same&~eye;neg=~same;s=f@f.T
 return 1-s[pos].sum()/(pos.sum()+1e-6)+.7*s[neg].abs().sum()/(neg.sum()+1e-6)
def info(u,w,y,valid=None):
 s=u@w.T/.07;pos=y[:,None]==y[None,:]
 if valid is not None:s=s.masked_fill(~valid,-1e4);pos=pos&valid
 def ll(a,p):return -(torch.logsumexp(a.masked_fill(~p,-1e4),1)-torch.logsumexp(a,1)).mean()
 return .5*(ll(s,pos)+ll(s.T,pos.T))
class RiemannianGrad(torch.autograd.Function):
 @staticmethod
 def forward(ctx,x):ctx.save_for_backward(x);return x
 @staticmethod
 def backward(ctx,g):
  x,=ctx.saved_tensors;return g*(1-.05*x.pow(2).sum(-1,keepdim=True)).pow(2)/4
def to_poincare(u):
 sc=.05**.5;n=u.norm(dim=-1,keepdim=True).clamp_min(1e-5);x=(sc*n).clamp(-15,15).tanh()*u/(sc*n);xn=x.norm(dim=-1,keepdim=True).clamp_min(1e-5);x=torch.where(xn>(1-1e-3)/sc,x/xn*((1-1e-3)/sc),x);return RiemannianGrad.apply(x)
def poincare_dist(x,y):
 c=.05;xy=(x*y).sum(-1);x2=(x*x).sum(-1);y2=(y*y).sum(-1)
 # Equivalent acosh metric avoids an N x N x D broadcast allocation.
 den=((1-c*x2)*(1-c*y2)).clamp_min(1e-6)
 return torch.acosh((1+2*c*((x-y)**2).sum(-1)/den).clamp_min(1+1e-6))/(c**.5)
class Input(nn.Module):
 def __init__(self,dims):
  super().__init__();self.dims=list(map(int,dims));self.bn=nn.ModuleList([nn.BatchNorm1d(d) for d in self.dims])
 def forward(self,x):return torch.cat([F.normalize(b(a),dim=1) for a,b in zip(x.split(self.dims,1),self.bn)],1)
class Branch(nn.Module):
 def __init__(self,dims,arch,D):
  super().__init__();self.pre=Input(dims);d=sum(dims)
  if arch=='msm':self.net=nn.Sequential(nn.Linear(d,D),nn.BatchNorm1d(D),nn.ReLU(),nn.Dropout(.91))
  elif arch=='paeff':self.net=nn.Sequential(nn.Linear(d,256),nn.BatchNorm1d(256),nn.ReLU(),nn.Dropout(.4),nn.Linear(256,D))
  else:self.net=nn.Sequential(nn.Dropout(.5),nn.Linear(d,512),nn.LayerNorm(512),nn.GELU(),nn.Dropout(.75),nn.Linear(512,D))
 def forward(self,x):return F.normalize(self.net(self.pre(x)),dim=1)
class ConvBranch(nn.Module):
 """1D convolutional bottleneck + frozen linear projection with rank-4 LoRA residual adapter."""
 def __init__(self,dims,D=128):
  super().__init__();d=sum(dims);self.pre=Input(dims);self.drop=nn.Dropout(.5)
  self.conv1=nn.Conv1d(1,8,3,stride=3);self.bn=nn.BatchNorm1d(8);self.conv2=nn.Conv1d(8,1,3,stride=3)
  n=(d//3)//3;self.base=nn.Linear(n,256);self.A=nn.Linear(n,4,bias=False);self.B=nn.Linear(4,256,bias=False)
  for p in self.base.parameters():p.requires_grad=False
  nn.init.zeros_(self.B.weight)
  self.tail=nn.Sequential(nn.LayerNorm(256),nn.GELU(),nn.Dropout(.75),nn.Linear(256,D))
 def forward(self,x):
  x=self.drop(self.pre(x));x=self.conv2(self.bn(F.relu(self.conv1(x[:,None]))))[:,0];x=self.base(x)+self.B(self.A(x))*.25;return F.normalize(self.tail(x),dim=1)
class Model(nn.Module):
 def __init__(self,fdims,vdims,nclasses,arch):
  super().__init__();self.arch=arch;D=nclasses-1 if self.arch=='msm' else 256 if self.arch=='paeff' else 128
  self.face=ConvBranch(fdims,D) if self.arch=='convlora' else Branch(fdims,self.arch,D)
  self.voice=ConvBranch(vdims,D) if self.arch=='convlora' else Branch(vdims,self.arch,D)
  if arch=='msm':self.register_buffer('P',torch.from_numpy(maxsep(nclasses)));self.w1=nn.Parameter(torch.rand(1));self.w2=nn.Parameter(torch.rand(1))
  if arch=='paeff':self.conv=nn.Conv1d(1,1,5,padding=2);self.mix=nn.Linear(D,D);self.cls=nn.Linear(D,nclasses)
  self.head=nn.Sequential(nn.Linear(4*D,128),nn.ReLU(),nn.Dropout(.3),nn.Linear(128,1)) if arch=='ver' else None
 def encode(self,f,v):return self.face(f),self.voice(v)
 def score(self,u,w):
  if self.arch=='ver':return self.head(torch.cat([u,w,u*w,(u-w).abs()],1))[:,0]
  if self.arch=='paeff':return -((F.gelu(u)-F.gelu(w))**2).sum(1)
  return (u*w).sum(1)
 def loss(self,f,v,y,g,valid):
  u,w=self.encode(f,v)
  if self.arch=='msm':
   m=self.w1*u+self.w2*w;return F.cross_entropy(m@self.P,y)+opl(m,y)
  if self.arch=='paeff':
   att=torch.sigmoid(self.conv((F.gelu(u)*F.gelu(w))[:,None])[:,0]);m=torch.tanh(self.mix(u*att+(1-att)*w));uh,wh=to_poincare(u),to_poincare(w)
   # Multi-positive CLIP avoids treating the other K utterances as false negatives.
   return .35*F.cross_entropy(self.cls(m),y)+.35*opl(m,y)+.3*info(uh,wh,y,valid)+.02*poincare_dist(uh,wh).mean()
  loss=info(u,w,y,valid)+.05*opl(torch.cat([u,w]),torch.cat([y,y]))
  if self.arch=='ver':
   same=(y[:,None]==y[None,:])&valid;pos=same.nonzero();ns=((y[:,None]!=y[None,:])&(g[:,None]==g[None,:])).nonzero();na=(y[:,None]!=y[None,:]).nonzero();k=len(pos)
   pick=lambda a,n:a[torch.randperm(len(a),device=a.device)[:n]]
   neg=torch.cat([pick(ns,3*k//4),pick(na,k-min(3*k//4,len(ns)))]);idx=torch.cat([pos,neg]);lab=torch.cat([torch.ones(len(pos),device=u.device),torch.zeros(len(neg),device=u.device)])
   loss+=F.binary_cross_entropy_with_logits(self.score(u[idx[:,0]],w[idx[:,1]]),lab)
  return loss
def module_export(m):
 """Small inference graph, omitting the 5096-class training-only simplex."""
 if isinstance(m,Input):return dict(type='input',dims=m.dims,layers=[module_export(x) for x in m.bn])
 if isinstance(m,Branch):return dict(type='branch',pre=module_export(m.pre),net=module_export(m.net))
 if isinstance(m,ConvBranch):return dict(type='convbranch',pre=module_export(m.pre),conv1=module_export(m.conv1),bn=module_export(m.bn),conv2=module_export(m.conv2),base=module_export(m.base),A=module_export(m.A),B=module_export(m.B),tail=module_export(m.tail))
 if isinstance(m,nn.Sequential):return dict(type='seq',layers=[module_export(x) for x in m])
 z=dict(type=type(m).__name__)
 if isinstance(m,(nn.Linear,nn.BatchNorm1d,nn.LayerNorm,nn.Conv1d)):
  z.update({k:v.detach().cpu().numpy().astype(np.float32) for k,v in m.state_dict().items()})
 if isinstance(m,(nn.BatchNorm1d,nn.LayerNorm)):z['eps']=m.eps
 if isinstance(m,nn.Conv1d):z.update(stride=m.stride[0],padding=m.padding[0])
 return z
from model_inference import unit,forward,score_graph
