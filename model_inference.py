"""Deterministic NumPy inference for exported cross-modal dual-tower graphs."""
import numpy as np

def unit(x):return x/np.maximum(np.sqrt(np.einsum('nd,nd->n',x,x,optimize=False))[:,None],1e-12)
def forward(m,x):
 from scipy.special import erf
 typ=m['type']
 if typ=='input':return np.column_stack([unit(forward(a,b)) for a,b in zip(m['layers'],np.split(x,np.cumsum(m['dims'])[:-1],axis=1))])
 if typ=='branch':return unit(forward(m['net'],forward(m['pre'],x)))
 if typ=='convbranch':
  x=forward(m['pre'],x);x=forward(m['conv1'],x[:,None]);x=forward(m['bn'],np.maximum(x,0));x=forward(m['conv2'],x)[:,0];x=forward(m['base'],x)+forward(m['B'],forward(m['A'],x))*.25;return unit(forward(m['tail'],x))
 if typ=='seq':
  for a in m['layers']:x=forward(a,x)
  return x
 if typ=='Linear':return np.einsum('nd,kd->nk',x,m['weight'],optimize=False)+m.get('bias',0)
 if typ=='BatchNorm1d':
  sh=(1,-1,1) if x.ndim==3 else (1,-1);return (x-m['running_mean'].reshape(sh))/np.sqrt(m['running_var'].reshape(sh)+m['eps'])*m['weight'].reshape(sh)+m['bias'].reshape(sh)
 if typ=='LayerNorm':
  mu=np.mean(x,axis=-1,keepdims=True);v=np.mean((x-mu)**2,axis=-1,keepdims=True);return (x-mu)/np.sqrt(v+m['eps'])*m['weight']+m['bias']
 if typ=='Conv1d':
  if m['padding']:x=np.pad(x,((0,0),(0,0),(m['padding'],m['padding'])))
  k=m['weight'].shape[-1];out=(x.shape[-1]-k)//m['stride']+1
  patches=np.stack([x[:,:,i*m['stride']:i*m['stride']+k] for i in range(out)],axis=2)
  return np.einsum('ncsk,ock->nos',patches,m['weight'],optimize=False)+m['bias'][None,:,None]
 if typ=='ReLU':return np.maximum(x,0)
 if typ=='GELU':return .5*x*(1+erf(x/np.sqrt(2)))
 if typ=='Dropout':return x
 raise ValueError(typ)
def score_graph(m,f,v):
 u=forward(m['face'],np.asarray(f,np.float64));w=forward(m['voice'],np.asarray(v,np.float64))
 if m['arch']=='ver':return -forward(m['head'],np.column_stack([u,w,u*w,np.abs(u-w)]))[:,0]
 if m['arch']=='paeff':
  from scipy.special import erf
  u=.5*u*(1+erf(u/np.sqrt(2)));w=.5*w*(1+erf(w/np.sqrt(2)));return np.einsum('nd,nd->n',u-w,u-w,optimize=False)
 return 1-np.einsum('nd,nd->n',u,w,optimize=False)
