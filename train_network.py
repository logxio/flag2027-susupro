"""Utterance-level training of the cross-modal dual-tower metric and pair discrimination architectures.

Banks contain per-utterance embeddings and speaker/recording metadata.
A training bank provides F, V, face_person, voice_person, face_recording,
voice_recording, face_gender, voice_gender, face_dims, voice_dims and split fields.
"""
from __future__ import annotations
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
import argparse
import json
from pathlib import Path
import numpy as np

def main():
    import torch
    from torch import nn
    from torch.nn import functional as F
    import models
    from model_inference import score_graph
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bank',type=Path);p.add_argument('--output-dir',type=Path,default=Path('output/network'))
    p.add_argument('--arch',choices=('cone','msm','paeff','convlora','ver'),default='ver')
    p.add_argument('--steps',type=int,default=1200);p.add_argument('--seed',type=int,default=20261118)
    p.add_argument('--held-fold',type=int,choices=range(5));p.add_argument('--exclude-language')
    p.add_argument('--smoke-test',action='store_true');a=p.parse_args()
    if not a.smoke_test and not a.bank:p.error('--bank is required')
    if a.steps<1:p.error('steps must be positive')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.manual_seed(a.seed)
    rng=np.random.default_rng(a.seed)
    if a.smoke_test:
        people=np.repeat([f'example_{i:03d}' for i in range(20)],3)
        recordings=np.tile(['a','b','c'],20);vectors=rng.normal(size=(20,48)).repeat(3,axis=0).astype(np.float32)
        bank=dict(F=vectors+rng.normal(size=vectors.shape).astype(np.float32)*.1,
                  V=vectors+rng.normal(size=vectors.shape).astype(np.float32)*.1,
                  face_person=people,voice_person=people,face_recording=recordings,voice_recording=recordings,
                  face_gender=np.repeat(np.arange(20)%2,3),voice_gender=np.repeat(np.arange(20)%2,3),
                  face_dims=np.array([48]),voice_dims=np.array([48]),split=np.array(['train']))
    else:
        with np.load(a.bank,allow_pickle=False) as z:bank=dict(z)
    if not set(bank['split']).issubset({'train','development'}):raise ValueError('Explicit development/training bank required')
    for side,name in [('face','F'),('voice','V')]:
        x=bank[name]
        if x.ndim!=2 or x.shape[1]!=sum(bank[side+'_dims']) or not np.isfinite(x).all():raise ValueError('Invalid '+side+' embeddings')
        if any(len(bank[side+'_'+k])!=len(x) for k in ('person','recording','gender')):raise ValueError('Metadata row mismatch')
    people=sorted(set(bank['face_person'])&set(bank['voice_person']))
    folds={str(x):i%5 for i,x in enumerate(rng.permutation(people))}
    held={s for s,k in folds.items() if a.held_fold==k}
    pools={side:{} for side in ('face','voice')}
    for side in pools:
        allowed=~np.isin(bank[side+'_person'],list(held))
        if a.exclude_language:
            if side+'_language' not in bank:raise ValueError('Language metadata required')
            allowed&=bank[side+'_language']!=a.exclude_language
        for person in people:
            ids=np.flatnonzero(allowed&(bank[side+'_person']==person))
            if len(ids):pools[side][person]=ids
    people=sorted(set(pools['face'])&set(pools['voice']));lookup={s:i for i,s in enumerate(people)}
    if len(people)<2:raise ValueError('At least two fitting speakers required')
    usable=[]
    for person in people:
        f=pools['face'][person];v=pools['voice'][person]
        if not all(any(bank['voice_recording'][v]!=bank['face_recording'][i]) for i in f):
            raise ValueError('Each face needs a different-recording positive voice for '+person)
        usable.append(person)
    if a.arch=='cone':
        class ConeTower(nn.Module):
            def __init__(self,dims):
                super().__init__();self.dims=list(map(int,dims));self.bns=nn.ModuleList([nn.BatchNorm1d(d) for d in self.dims])
                self.net=nn.Sequential(nn.Dropout(.75),nn.Linear(sum(dims),384),nn.BatchNorm1d(384),nn.GELU(),nn.Dropout(.3),nn.Linear(384,192),nn.BatchNorm1d(192,affine=False))
            def forward(self,x):
                return F.normalize(self.net(torch.cat([F.normalize(b(v),dim=1) for b,v in zip(self.bns,x.split(self.dims,1))],1)),dim=1)
        class ConePair(nn.Module):
            def __init__(self):
                super().__init__();self.face=ConeTower(bank['face_dims']);self.voice=ConeTower(bank['voice_dims'])
            def encode(self,f,v):return self.face(f),self.voice(v)
            def loss(self,f,v,y,g,valid):
                u,w=self.encode(f,v);return models.info(u,w,y,valid)+.05*models.opl(torch.cat([u,w]),torch.cat([y,y]))
        net=ConePair()
    else:net=models.Model(bank['face_dims'],bank['voice_dims'],len(people),a.arch)
    opt=torch.optim.AdamW(net.parameters(),lr=.0001 if a.arch=='msm' else .001,weight_decay=.01)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,a.steps)
    for step in range(a.steps):
        candidates=usable
        if step%4!=3:
            gender=step%2
            candidates=[s for s in usable if int(np.median(bank['face_gender'][pools['face'][s]]))==gender]
            if len(candidates)<2:candidates=usable
        selected=rng.choice(candidates,min(16,len(candidates)),replace=False)
        fi=[];vi=[];labels=[];genders=[]
        for person in selected:
            for _ in range(2):
                f=int(rng.choice(pools['face'][person]));vv=pools['voice'][person]
                vv=vv[bank['voice_recording'][vv]!=bank['face_recording'][f]]
                v=int(rng.choice(vv));fi.append(f);vi.append(v);labels.append(lookup[person]);genders.append(bank['face_gender'][f])
        valid=bank['face_recording'][fi][:,None]!=bank['voice_recording'][vi][None,:]
        tx=lambda x:torch.from_numpy(np.asarray(x))
        net.train();loss=net.loss(tx(bank['F'][fi].astype(np.float32)),tx(bank['V'][vi].astype(np.float32)),tx(np.array(labels,np.int64)),tx(np.array(genders,np.int64)),tx(valid))
        if not torch.isfinite(loss):raise ValueError('Nonfinite optimization objective')
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5.);opt.step();sched.step()
    net.eval();a.output_dir.mkdir(parents=True,exist_ok=True)
    torch.save(dict(state=net.state_dict(),arch=a.arch,face_dims=bank['face_dims'].tolist(),voice_dims=bank['voice_dims'].tolist(),classes=len(people)),a.output_dir/'network.pt')
    f=bank['F'][fi[:4]].astype(np.float32);v=bank['V'][vi[:4]].astype(np.float32)
    with torch.inference_mode():
        u,w=net.encode(torch.from_numpy(f),torch.from_numpy(v))
        if a.arch=='cone':expected=(1-(u*w).sum(1)).numpy()
        else:expected=(-net.score(u,w)+(1 if a.arch in ('msm','convlora') else 0)).numpy()
    exported=False;delta=None
    if a.arch!='cone':
        graph=dict(arch=a.arch,face=models.module_export(net.face),voice=models.module_export(net.voice),head=models.module_export(net.head) if net.head is not None else None)
        actual=score_graph(graph,f,v);delta=float(abs(expected-actual).max())
        if delta>.005:raise ValueError('Torch and exported inference mismatch')
        # Trusted local development export. Never deserialize files from untrusted sources.
        import pickle
        with (a.output_dir/'network.pkl').open('wb') as stream:pickle.dump(graph,stream,protocol=5)
        exported=True
    report=dict(scope='Cross-modal dual-tower metric network training',arch=a.arch,steps=a.steps,seed=a.seed,
                smoke_test=a.smoke_test,fixture_data=a.smoke_test,train_people=people,held_people=sorted(held),
                speaker_to_fold=folds,excluded_language=a.exclude_language,loss=float(loss.detach()),
                numpy_export=exported,numpy_torch_max_diff=delta)
    (a.output_dir/'network_report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
