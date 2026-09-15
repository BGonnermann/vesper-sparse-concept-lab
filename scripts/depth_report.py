"""Source-backed equal-token and measured-time endpoint comparisons."""
import csv
import statistics
from pathlib import Path
import sys

import autoresearch as r
import depth_campaign as c
import experiment_reports as reports


def recommendation(rows):
    by={(x['seed'],x['depth'],x['budget']):x for x in rows}
    seeds=sorted(s for s in {x['seed'] for x in rows} if all((s,d,b) in by for d in (6,12) for b in (1,2,4)))
    contrasts=[]
    for low in (1,2):
        pairs=[]
        for seed in seeds:
            a,b=by[seed,6,low*2],by[seed,12,low]
            pairs.append(dict(seed=seed,d6_budget=low*2,d12_budget=low,
                d6_minus_d12_bpb=a['bpb']-b['bpb'],d6_over_d12_time=a['update_seconds']/b['update_seconds']))
        contrasts.append(dict(d6_budget=low*2,d12_budget=low,pairs=pairs,
            mean_delta=statistics.mean(x['d6_minus_d12_bpb'] for x in pairs) if pairs else None))
    result='inconclusive'
    if len(seeds)>=3 and all(.85<=p['d6_over_d12_time']<=1.15 for x in contrasts for p in x['pairs']):
        if all(x['mean_delta']<=-.001 and all(p['d6_minus_d12_bpb']<0 for p in x['pairs']) for x in contrasts):result='D6-width768'
        if all(x['mean_delta']>=.001 and all(p['d6_minus_d12_bpb']>0 for p in x['pairs']) for x in contrasts):result='D12-width768'
    return dict(recommendation=result,complete_seeds=seeds,approximately_time_matched_contrasts=contrasts,
        scope='Frozen validation rule for the overlapping measured time range; no extrapolation to D6@8x or held-out selection')


def collect():
    rows=[]
    for item in c.records():
        if item['status']!='completed':continue
        folder=c.HERE/item['trial'];training=c.read(folder/'training.json');fixed=c.read(folder/'fixed-training.json');model=c.read(folder/'model.json')
        rows.append(dict(trial=item['trial'],seed=item['seed'],depth=item['depth'],budget=item['budget'],
            bpb=item['metrics']['val_bpb'],tokens=training['training_tokens'],updates=training['optimizer_updates'],
            update_seconds=fixed['all_update_seconds'],trial_seconds=item['wall_seconds'],
            timed_tokens_per_second=training['timed_training_tokens']/training['timed_training_seconds'],
            all_update_tokens_per_second=training['training_tokens']/fixed['all_update_seconds'],
            parameters=model['total_parameters'],active_parameters=model['active_parameters'],
            peak_allocated_mib=item['memory']['peak_allocated_bytes']/2**20,
            peak_reserved_mib=item['memory']['peak_reserved_bytes']/2**20,
            checkpoint_sha256=item['checkpoint_sha256'],result_sha256=r.digest(folder/'result.json')))
    assert len({(x['seed'],x['depth'],x['budget']) for x in rows})==len(rows),'Duplicate successful cell'
    return rows


def write(final=False):
    rows=collect();decision=recommendation(rows)
    root=r.ROOT/'reports/experiments'/reports.identity(c.HERE,r.ROOT)[0];root.mkdir(parents=True,exist_ok=True)
    r.write_json(c.HERE/'analysis.json',dict(status='completed' if final else 'running',rows=rows,**decision))
    if rows:
        with (root/'measurements.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(12,4.8),sharey=True)
        for depth,color,marker,style in ((6,'#1768ac','o','-'),(12,'#b65b19','s','--')):
            subset=[x for x in rows if x['depth']==depth]
            for ax,key in zip(axes,('tokens','update_seconds')):
                scale=1e6 if key=='tokens' else 60
                for seed in sorted({x['seed'] for x in subset}):
                    points=sorted((x for x in subset if x['seed']==seed),key=lambda x:x['budget'])
                    ax.plot([x[key]/scale for x in points],[x['bpb'] for x in points],color=color,marker=marker,ls=style,alpha=.25,lw=.8,ms=3)
                points=[]
                for budget in (1,2,4):
                    group=[x for x in subset if x['budget']==budget and x['seed'] in decision['complete_seeds']]
                    if group:points.append((statistics.mean(x[key] for x in group)/scale,statistics.mean(x['bpb'] for x in group),budget))
                if points:
                    ax.plot([x[0] for x in points],[x[1] for x in points],color=color,marker=marker,ls=style,lw=2,label=f'D{depth}, complete-seed mean')
                    if key=='update_seconds':
                        for x,y,b in points:ax.annotate(f'{b}x',(x,y),xytext=(5,6),textcoords='offset points',fontsize=8,color=color)
                ax.grid(axis='y',color='#dddddd',linewidth=.6)
                ax.spines[['top','right']].set_visible(False)
        axes[0].set_xlabel('Training tokens (millions)');axes[1].set_xlabel('Measured training-update time (minutes)')
        axes[0].set_ylabel('Validation bits per byte, lower is better')
        axes[0].set_title('Equal-token budget endpoints');axes[1].set_title('Quality against measured training time')
        if decision['complete_seeds']:axes[0].legend(frameon=False,fontsize=8)
        fig.suptitle('TinyStories: dense width768, independently annealed runs',fontsize=12)
        fig.text(.5,.01,'Thin lines: individual seeds. Lines connect measured endpoints only; no interpolation or convergence claim.',ha='center',fontsize=8)
        fig.tight_layout(rect=(0,.04,1,.95));fig.savefig(root/'learning-curves.png',dpi=180);plt.close(fig)
    lines=['# Longer-training depth campaign'+(' final report' if final else ' progress'),'',
        f"Recommendation under the frozen validation rule: **{decision['recommendation']}**. Complete paired seeds: {decision['complete_seeds']}.",'',
        'TinyStories only. Equal-token comparisons are not equal-compute comparisons. Test results never drive selection.', '',
        '[Frozen plan](PLAN.md) · [Exact measurements](measurements.csv) · [Reproducibility receipts](report.json)','']
    if rows:lines+=['![Learning curves](learning-curves.png)','']
    lines+=['## Every completed run','',
        '| Seed | Depth | Budget | Validation BPB | Actual tokens | Update s | Trial s | Timed tok/s | Parameters | Allocated / reserved MiB |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for x in sorted(rows,key=lambda x:(x['seed'],x['budget'],x['depth'])):
        lines.append(f"| {x['seed']} | {x['depth']} | {x['budget']}x | {x['bpb']:.6f} | {x['tokens']:,} | {x['update_seconds']:.1f} | {x['trial_seconds']:.1f} | {x['timed_tokens_per_second']:.0f} | {x['parameters']:,} | {x['peak_allocated_mib']:.1f} / {x['peak_reserved_mib']:.1f} |")
    by={(x['seed'],x['depth'],x['budget']):x for x in rows}
    lines+=['','## Equal-token paired differences','', '| Seed | Budget | D12 minus D6 BPB | D12 / D6 update time |','|---:|---:|---:|---:|']
    deltas={}
    for seed in sorted({x['seed'] for x in rows}):
        for budget in (1,2,4):
            if all((seed,d,budget) in by for d in (6,12)):
                a,b=by[seed,12,budget],by[seed,6,budget];delta=a['bpb']-b['bpb'];deltas.setdefault(budget,[]).append(delta)
                lines.append(f"| {seed} | {budget}x | {delta:+.6f} | {a['update_seconds']/b['update_seconds']:.3f} |")
    for budget,values in deltas.items():
        sd=statistics.stdev(values) if len(values)>1 else None
        lines+=['',f'{budget}x: mean paired delta {statistics.mean(values):+.6f}; '+(f'sample SD {sd:.6f}' if sd is not None else 'SD unavailable')+f'; {sum(x<0 for x in values)}/{len(values)} favor D12.']
    lines+=['','## Approximately time-matched comparisons','', '| Seed | D6 budget | D12 budget | D6 minus D12 BPB | D6 / D12 update time |','|---:|---:|---:|---:|---:|']
    for contrast in decision['approximately_time_matched_contrasts']:
        for p in contrast['pairs']:lines.append(f"| {p['seed']} | {p['d6_budget']}x | {p['d12_budget']}x | {p['d6_minus_d12_bpb']:+.6f} | {p['d6_over_d12_time']:.3f} |")
    lines+=['','These are actual unequal-time endpoints. The frozen approximate-match tolerance is0.85–1.15 per pair; no exact equal-time quality is inferred. Both contrast means must improve by at least0.001 BPB, with every seed agreeing, for a directional recommendation.','',
        '## Protocol and interpretation','',
        'Each point starts from initialization. LR and Muon weight decay follow progress i/N; momentum warms over300 absolute updates. This is a family of independently annealed budget endpoints. The final applied LR is2/N of base. Matrix LR0.04, BF16, context512, batch16,384 tokens, microbatch2 and checkpointing off are fixed.','',
        'Update time includes all updates; timed throughput excludes the first11 updates. Trial wall time also includes setup, validation and artifact verification, excluding publication. Allocator peaks exclude driver and desktop allocations; whole-board samples remain in each trial receipt. All parameters are active; no memory table or auxiliary architecture is used.','']
    for depth in (6,12):
        changes=[by[s,depth,4]['bpb']-by[s,depth,2]['bpb'] for s in decision['complete_seeds']]
        if changes:lines.append(f"D{depth} 4x-minus-2x mean: {statistics.mean(changes):+.6f}; {sum(x<0 for x in changes)}/{len(changes)} improve. "+('Learning curves remain unfinished; convergence is not established.' if statistics.mean(changes)<0 else 'This budget comparison does not establish convergence.'))
    stream=c.read(c.HERE/'stream-result.json')
    lines+=['','## Training stream','',f"Tape: {stream['microbatches']:,} microbatches, {stream['training_tokens']:,} tokens, epochs {stream['epochs']}. Exact duplicate microbatches: {stream['duplicate_microbatches']}. {stream['previous_tape_matching_microbatches']:,} microbatches also occur in the previous campaign tape. All32,768 rows independently reconstructed. Each run consumes distinct indices; separate models/budgets reuse data by design. Near-duplicates and repeated phrases are not excluded.",'',
        '## Failures and retained evidence','']
    failures=[x for x in c.records() if x['status']!='completed']
    lines += [f"- {x.get('trial')}: {x['status']}, {x.get('error','')}" for x in failures] or ['No training failures recorded.']
    lines+=['','## Final test stage','', 'Pending. Exact candidate/reference checkpoint identities will be frozen before test evaluation; no subsequent training or reselection.','',
        '## Reproducibility','',f"Frozen matrix SHA256: {r.digest(c.HERE/'matrix-plan.json')}. Every trial retains its checkpoint, source snapshot, data/config hashes, consumed-order hash chain, optimizer coverage, schedule and immutable-evaluation receipt. See the progress index for individual trial reports. Existing artifacts and production baseline remain unchanged."]
    (root/'CAMPAIGN.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (root/'PLAN.md').write_bytes((c.HERE/'PLAN.md').read_bytes())
    reports.emit(c.HERE);reports.index()
    return decision


if __name__=='__main__':write('--final' in sys.argv)
