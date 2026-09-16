"""Seven reproducible figures from recorded metrics only; never loads a model."""
import argparse
import csv
import gzip
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIELDS=['step','tokens','tokens_per_parameter','loss','gradient_norm','matrix_lr','lr_multiplier','muon_momentum','muon_weight_decay','seconds','tokens_per_second','wall_seconds','peak_allocated_bytes','peak_reserved_bytes','general_tokens','general_passes','technical_tokens','technical_passes']

def export(run,destination):
    destination.mkdir(parents=True,exist_ok=True)
    losses=json.loads((run/'losses.json').read_text());evaluations=json.loads((run/'evaluations.json').read_text())
    with gzip.open(destination/'updates.csv.gz','wt',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader()
        for x in losses:
            row={k:x[k] for k in FIELDS if k in x}
            for domain in ('general','technical'):
                row[domain+'_tokens']=x['data_progress'][domain]['tokens'];row[domain+'_passes']=x['data_progress'][domain]['passes']
            w.writerow(row)
    with (destination/'validation.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['step','tokens','wall_seconds','aggregate_bpb','general_bpb','technical_bpb','evaluation_seconds']);w.writeheader()
        for x in evaluations:
            v=x['evaluation'];w.writerow(dict(step=x['step'],tokens=x['tokens'],wall_seconds=x['wall_seconds'],aggregate_bpb=v['aggregate_bpb'],general_bpb=v['domains']['general']['bpb'],technical_bpb=v['domains']['technical']['bpb'],evaluation_seconds=v['seconds']))
    (destination/'chart-metadata.json').write_text(json.dumps(dict(config=json.loads((run/'config.json').read_text()),lower_bpb_is_better=True,loss_smoothing_updates=128,throughput_smoothing_updates=128,raw_metrics_not_smoothed=True,scope='Single fixed-seed closed-pool reference; no statistical superiority claim'),indent=2)+'\n')

def frontier(values):return np.minimum.accumulate(np.asarray(values,dtype=float))

def render(data,output):
    output.mkdir(parents=True,exist_ok=True);u=pd.read_csv(data/'updates.csv.gz');v=pd.read_csv(data/'validation.csv')
    if len(u)==0 or len(v)==0:raise ValueError('Missing measured curve')
    assert u.step.is_monotonic_increasing and v.step.is_monotonic_increasing
    blue='#0072B2';orange='#D55E00';green='#009E73'
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.titlesize':13,'axes.labelsize':11,'figure.dpi':130,'savefig.dpi':300,'svg.hashsalt':'foundation-long-v1','axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.2})
    def save(fig,name):
        fig.text(.01,.008,'Dense D12 / width768 | seed301 | fixed accepted data + tokenizer | base-model reference',fontsize=8,color='#555555')
        fig.tight_layout(rect=(0,.045,1,1));fig.savefig(output/(name+'.png'),bbox_inches='tight');fig.savefig(output/(name+'.svg'),bbox_inches='tight',metadata={'Date':None});plt.close(fig)
    mt=u.tokens/1e6;vt=v.tokens/1e6
    fig,ax=plt.subplots(figsize=(9,4.6));ax.plot(mt,u.loss,color=blue,alpha=.15,lw=.4,label='Every update');ax.plot(mt,u.loss.rolling(128,min_periods=1).mean(),color=blue,lw=1.8,label='128-update mean');ax.set(xlabel='Training tokens (millions)',ylabel='Training cross-entropy (nats/token)',title='Training loss versus tokens');ax.legend();save(fig,'01-training-loss')
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for ax,mask,title in [(axes[0],np.ones(len(v),dtype=bool),'Full learning curve'),(axes[1],v.tokens>=8388608,'Pilot budget onward')]:
        ax.plot(vt[mask],v.aggregate_bpb[mask],'-o',color=blue,ms=3);ax.set(xlabel='Training tokens (millions)',ylabel='Validation BPB (lower is better)',title=title)
    save(fig,'02-validation-tokens')
    fig,ax=plt.subplots(figsize=(9,4.6));ax.plot(vt,v.general_bpb,'-o',color=blue,ms=3,label='General');ax.plot(vt,v.technical_bpb,'-s',color=orange,ms=3,label='Technical');ax.set(xlabel='Training tokens (millions)',ylabel='Validation BPB (lower is better)',title='Per-domain validation: do not hide regressions');ax.legend();save(fig,'03-domain-validation')
    fig,ax=plt.subplots(figsize=(9,4.6));ax.plot(v.wall_seconds/3600,v.aggregate_bpb,'-o',color=blue,ms=3);ax.set(xlabel='Run wall time (hours, includes diagnostics)',ylabel='Validation BPB (lower is better)',title='Validation BPB versus wall-clock time');save(fig,'04-validation-wall-time')
    fig,axes=plt.subplots(1,2,figsize=(12,4.5));axes[0].plot(mt,u.tokens_per_second/1000,color=blue,alpha=.12,lw=.4,label='Every update');axes[0].plot(mt,u.tokens_per_second.rolling(128,min_periods=1).mean()/1000,color=blue,lw=1.7,label='128-update mean');axes[0].set(xlabel='Training tokens (millions)',ylabel='Throughput (thousand tokens/s)',title='Measured update throughput');axes[0].legend(fontsize=9)
    axes[1].plot(mt,u.peak_allocated_bytes/2**30,color=blue,label='Allocated');axes[1].plot(mt,u.peak_reserved_bytes/2**30,color=orange,label='Reserved');axes[1].set(xlabel='Training tokens (millions)',ylabel='CUDA allocator peak (GiB)',title='Running peak VRAM (not whole-board usage)');axes[1].set_ylim(bottom=0);axes[1].legend();save(fig,'05-throughput-vram')
    fig,ax=plt.subplots(figsize=(9,4.6));ax.plot(mt,u.matrix_lr,color=blue,lw=2);ax.set(xlabel='Training tokens (millions)',ylabel='Matrix learning rate',title='Recorded learning-rate schedule');save(fig,'06-learning-rate')
    fig,ax=plt.subplots(figsize=(9,4.6));ax.scatter(vt,v.aggregate_bpb,color=blue,s=26,label='Every scheduled validation');ax.step(vt,frontier(v.aggregate_bpb),where='post',color=green,lw=2,label='Best-so-far frontier');ax.set(xlabel='Training tokens (millions)',ylabel='Validation BPB (lower is better)',title='Autoresearch-style progress: every evaluation, no hidden failures');ax.legend();save(fig,'07-validation-frontier')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path);p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.run:export(a.run,a.data)
    render(a.data,a.output)
