"""Deterministic feature extraction and fixed-slice embedding averaging for frozen pretrained encoders.

Exports include deterministic upstream preprocessing in evaluation mode with gradients disabled.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

def unit(x):
    x=np.asarray(x,dtype=np.float64).reshape(-1)
    if not np.isfinite(x).all() or not len(x):raise ValueError('Encoder output must be a finite vector')
    return x/max(float(np.linalg.norm(x)),1e-12)

def fixed_slices(waveform,sample_rate=16000,seconds=6.,count=3):
    x=np.asarray(waveform,dtype=np.float32)
    if x.ndim!=1 or not len(x) or not np.isfinite(x).all():raise ValueError('Expected nonempty finite mono waveform')
    n=round(sample_rate*seconds)
    if n<=0 or count<=0:raise ValueError('Positive slice duration and count are required')
    if len(x)<n:x=np.tile(x,int(np.ceil(n/len(x))))
    starts=np.linspace(0,len(x)-n,count).round().astype(int)
    return np.stack([x[s:s+n] for s in starts])

def spectral_correlation(face,voice,face_templates,voice_templates,weights):
    """Pearson response correlation against fixed, aligned training templates."""
    f=np.asarray(face_templates)@unit(face);v=np.asarray(voice_templates)@unit(voice)
    w=np.asarray(weights,dtype=float)
    if w.shape!=f.shape or f.shape!=v.shape or (w<0).any() or w.sum()<=0:
        raise ValueError('Aligned training templates and nonnegative weights are required')
    w=w/w.sum();f=f-np.dot(w,f);v=v-np.dot(w,v)
    return float(np.dot(w,f*v)/max(np.sqrt(np.dot(w,f*f)*np.dot(w,v*v)),1e-12))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path);p.add_argument('--encoder',help='Registry key')
    p.add_argument('--input',type=Path,help='NPY: RGB uint8 HWC image or mono float32 waveform at registered sample rate')
    p.add_argument('--output',type=Path);p.add_argument('--smoke-test',action='store_true')
    a=p.parse_args()
    if a.smoke_test:
        wave=np.arange(48000,dtype=np.float32)/48000
        x=fixed_slices(wave);assert x.shape==(3,96000) and np.array_equal(x,fixed_slices(wave))
        f=np.array([1.,0.]);t=np.eye(2);assert spectral_correlation(f,f,t,t,np.ones(2))==1.
        print(json.dumps(dict(scope='Deterministic fixed-slice averaging and training-template spectral correlation verified',slices=list(x.shape),deterministic=True)))
        return
    if any(v is None for v in (a.manifest,a.encoder,a.input,a.output)):p.error('--manifest --encoder --input --output are required')
    import torch
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    registry=json.loads(a.manifest.read_text());spec=registry[a.encoder]
    weight=(a.manifest.parent/spec['export_path']).resolve()
    if hashlib.sha256(weight.read_bytes()).hexdigest()!=spec['sha256']:raise ValueError('Encoder checksum mismatch')
    model=torch.jit.load(str(weight),map_location='cpu').eval()
    for parameter in model.parameters():parameter.requires_grad_(False)
    x=np.load(a.input,allow_pickle=False)
    if spec['modality']=='voice':
        slices=fixed_slices(x,spec.get('sample_rate',16000),spec.get('seconds',6.),spec.get('slices',3))
    elif spec['modality']=='face':
        if x.ndim!=3 or x.shape[2]!=3 or x.dtype!=np.uint8:raise ValueError('Expected RGB uint8 image')
        slices=x[None]
    else:raise ValueError('Unknown modality')
    with torch.inference_mode():
        values=[unit(model(torch.from_numpy(s[None].copy())).cpu().numpy()) for s in slices]
    result=unit(np.mean(values,axis=0))
    a.output.parent.mkdir(parents=True,exist_ok=True);np.save(a.output,result,allow_pickle=False)
    a.output.with_suffix('.json').write_text(json.dumps(dict(encoder=a.encoder,export_sha256=spec['sha256'],
        input_sha256=hashlib.sha256(a.input.read_bytes()).hexdigest(),frozen=True,dimension=len(result),
        sample_rate=spec.get('sample_rate'),slice_count=len(slices)),indent=2)+'\n')
if __name__=='__main__':main()
