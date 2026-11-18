"""Train the conditional bilinear fusion head (VER_FUSE_X) via 5-fold speaker-disjoint CV
and Leave-Language-Out (English -> Urdu/Hindi) validation.
"""
from __future__ import annotations
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[k]='1'
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
from scipy.linalg import solve
from scipy.special import expit
import core

def load(path):
    with np.load(path, allow_pickle=False) as z:
        return dict(z)

def select(z, mask):
    return {k: (v if k=='names' else v[mask]) for k,v in z.items()}

def validate(z):
    required={'X','C','names','y','face_person','voice_person','language','split'}
    if required-set(z):
        raise ValueError('Missing fields: '+str(sorted(required-set(z))))
    n=len(z['y'])
    if z['X'].ndim!=2 or z['C'].ndim!=2 or z['C'].shape[1]<8:
        raise ValueError('Expected X matrix and at least eight conditioning columns')
    if len(z['names'])!=z['X'].shape[1] or len(set(z['names']))!=len(z['names']):
        raise ValueError('Feature names must be unique and match X columns')
    if any(len(v)!=n for k,v in z.items() if k!='names'):
        raise ValueError('Row dimensions disagree')
    if not set(z['split']).issubset({'train','development'}):
        raise ValueError('Only explicitly designated training/development rows are accepted')
    if set(z['y'])!={0,1} or not np.isfinite(z['C']).all():
        raise ValueError('Both binary classes and finite conditioning data are required')
    if np.isinf(z['X']).any():
        raise ValueError('X permits finite values and NaN missing entries only')
    if any(str(x).strip()=='' for k in ('face_person','voice_person') for x in z[k]):
        raise ValueError('Each endpoint requires a speaker identity')

def fit(z, penalty):
    x=z['X']; ok=np.isfinite(x); counts=ok.sum(0)
    mean=np.where(ok,x,0).sum(0)/np.maximum(counts,1)
    sd=np.sqrt(np.where(ok,(x-mean)**2,0).sum(0)/np.maximum(counts,1))
    head=dict(names=z['names'],xmean=mean,xsd=np.maximum(sd,1e-6),
              cmean=z['C'].mean(0),csd=np.maximum(z['C'].std(0),1e-6),
              availability=np.any(~ok,axis=0))
    a=np.column_stack([core.fusion_design(head,z),np.ones(len(x))])
    target=1.-2.*z['y']
    gram=a.T@a/len(a)
    reg=np.full(a.shape[1],penalty);reg[-1]=1e-10
    coef=solve(gram+np.diag(reg),a.T@target/len(a),assume_a='pos')
    head.update(w=coef[:-1],bias=coef[-1])
    return head

def eer(y,q):
    # Lower distance means a positive face/voice match; include tied scores together.
    order=np.argsort(q,kind='stable');y=np.asarray(y)[order];q=np.asarray(q)[order]
    end=np.r_[np.flatnonzero(q[1:]!=q[:-1]),len(q)-1]
    pos=np.r_[0,np.cumsum(y)[end]];neg=np.r_[0,np.cumsum(1-y)[end]]
    fnr=1-pos/max(1,int(y.sum()));fpr=neg/max(1,int((1-y).sum()))
    diff=fpr-fnr;i=int(np.flatnonzero(diff>=0)[0])
    if i==0:return float(100*fpr[i])
    t=-diff[i-1]/(diff[i]-diff[i-1])
    return float(100*(fpr[i-1]+t*(fpr[i]-fpr[i-1])))

def speaker_folds(z,seed):
    people=sorted(set(z['face_person'])|set(z['voice_person']))
    if len(people)<10:raise ValueError('At least ten identities are needed for five folds')
    order=np.random.default_rng(seed).permutation(people)
    return {str(p):int(i%5) for i,p in enumerate(order)}

def masks(z,folds,k):
    f=np.array([folds[str(p)] for p in z['face_person']]);v=np.array([folds[str(p)] for p in z['voice_person']])
    return (f!=k)&(v!=k),(f==k)&(v==k)

def smoke(seed):
    rng=np.random.default_rng(seed);people=[f'example_{i:03d}' for i in range(70)]
    stub=dict(face_person=np.array(people),voice_person=np.array(people))
    folds=speaker_folds(stub,seed);rows=[]
    for k in range(5):
        group=[p for p in people if folds[p]==k]
        for j,p in enumerate(group):
            for repeat in range(2):
                rows.extend([(p,p,1),(p,group[(j+1)%len(group)],0)])
    y=np.array([r[2] for r in rows]);x=rng.normal(size=(len(y),8));x[:,0]+=1-2*y
    x[::17,3]=np.nan
    return dict(X=x,C=rng.normal(size=(len(y),16)),names=np.array([f'example_stream_{i}' for i in range(8)]),
                y=y,face_person=np.array([r[0] for r in rows]),voice_person=np.array([r[1] for r in rows]),
                language=np.array(['English']*len(y)),split=np.array(['train']*len(y)))

def main():
    route=argparse.ArgumentParser(add_help=False)
    route.add_argument('--stage',choices=('fusion','network'),default='fusion')
    stage,rest=route.parse_known_args()
    if stage.stage=='network':
        from train_network import main as train_network
        sys.argv=[sys.argv[0],*rest]
        return train_network()
    sys.argv=[sys.argv[0],*rest]
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-data',type=Path);p.add_argument('--validation-data',type=Path)
    p.add_argument('--output-dir',type=Path,default=Path('output/training'))
    p.add_argument('--seed',type=int,default=20261118)
    p.add_argument('--ridge',type=float,nargs='+',default=[.1,1.,10.])
    p.add_argument('--smoke-test',action='store_true');a=p.parse_args()
    if not a.smoke_test and a.train_data is None:p.error('--train-data is required')
    if any(v<=0 for v in a.ridge):p.error('ridge penalties must be positive')
    z=smoke(a.seed) if a.smoke_test else load(a.train_data);validate(z)
    folds=speaker_folds(z,a.seed);records=[]
    external=load(a.validation_data) if a.validation_data else None
    if external is not None:
        validate(external)
        if set(external['language'])-{'Urdu','Hindi'}:raise ValueError('External validation expects Urdu/Hindi only')
        overlap=(set(z['face_person'])|set(z['voice_person']))&(set(external['face_person'])|set(external['voice_person']))
        if overlap:raise ValueError('Training and external validation identities overlap')
        if not np.array_equal(z['names'],external['names']):raise ValueError('External stream order mismatch')
    for penalty in a.ridge:
        values=[];details=[]
        for k in range(5):
            tr,va=masks(z,folds,k)
            if set(z['y'][tr])!={0,1} or set(z['y'][va])!={0,1}:
                raise ValueError(f'Fold {k} needs positive and negative trials wholly within its identity group')
            train=select(z,tr);valid=select(z,va);h=fit(train,penalty)
            training_people=set(train['face_person'])|set(train['voice_person'])
            held_people=set(valid['face_person'])|set(valid['voice_person'])
            if training_people&held_people:raise ValueError('Speaker partition mismatch')
            value=eer(valid['y'],core.independent_predict(h,valid));values.append(value)
            details.append(dict(fold=k,train_rows=int(tr.sum()),valid_rows=int(va.sum()),
                                train_people=sorted(training_people),held_people=sorted(held_people),eer_percent=value))
        llo={}
        if external is not None:
            h=fit(z,penalty)
            for lang in sorted(set(external['language'])):
                v=select(external,external['language']==lang)
                if set(v['y'])!={0,1}:raise ValueError('Each language needs both trial classes')
                llo[lang]=eer(v['y'],core.independent_predict(h,v))
        objective=float(np.mean(values+list(llo.values())))
        records.append(dict(ridge=penalty,selection_eer_percent=objective,folds=details,language_eer_percent=llo))
    best=min(records,key=lambda r:r['selection_eer_percent']);head=fit(z,best['ridge'])
    a.output_dir.mkdir(parents=True,exist_ok=True);target=a.output_dir/'reference-head.npz'
    np.savez_compressed(target,**head)
    restored=load(target);pred=expit(core.independent_predict(restored,z))
    one=np.array([expit(core.independent_predict(restored,select(z,slice(i,i+1))))[0] for i in range(len(pred))])
    if not np.array_equal(pred,one):raise ValueError('Exported singleton predictions differ')
    report=dict(scope='5-fold speaker-disjoint and leave-language-out conditional bilinear fusion fit',
                smoke_test=a.smoke_test,fixture_data=a.smoke_test,seed=a.seed,selected_ridge=best['ridge'],
                rows=len(z['y']),speaker_to_fold=folds,selection_results=records,
                singleton_max_diff=float(np.max(abs(pred-one))),
                head_sha256=hashlib.sha256(target.read_bytes()).hexdigest())
    (a.output_dir/'training_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('selection_results','speaker_to_fold')}))
if __name__=='__main__':main()
