"""Verify four completed captured trials and write the preliminary comparison."""
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
import autoresearch as r
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
rows=[]
records={}
for seed in (42,43):
    for variant in ('dense','moe'):
        out=HERE/f'{variant}-{seed}'
        rec=read(out/'result.json')
        assert rec['status']=='completed'
        r.validate_execution(out,rec['snapshot_files'])
        r.validate_run_artifacts(out,rec)
        records[variant,seed]=rec
        t=rec['artifacts']['training']; model=rec['artifacts']['model']; routing=rec['artifacts']['routing']
        assert t['optimizer_updates']==512 and t['training_tokens']==8388608
        row=dict(seed=seed,model=variant,bpb=rec['metrics']['val_bpb'],
            total_parameters=model['total_parameters'],active_parameters=model['active_parameters'],
            width=model['width'],training_tokens=t['training_tokens'],timed_tokens=t['timed_training_tokens'],
            timed_seconds=t['timed_training_seconds'],timed_tokens_per_second=t['timed_training_tokens']/t['timed_training_seconds'],
            all_update_seconds=read(out/'fixed-training.json')['all_update_seconds'],wall_seconds=rec['wall_seconds'],
            peak_allocated_mib=rec['memory']['peak_allocated_bytes']/2**20,
            peak_reserved_mib=rec['memory']['peak_reserved_bytes']/2**20,
            train_expert_fractions=routing['train']['fractions'],eval_expert_fractions=routing['eval']['fractions'],
            dropped_tokens=routing['train']['dropped_tokens'],result=str(out/'result.json'))
        rows.append(row)
    a,b=records['dense',seed],records['moe',seed]
    for key in ('protocol','seed','data_seal','upstream'):
        assert a[key]==b[key],key
    for key in r.PROJECT_FILES.values(): assert a[key]==b[key],key
    for name in ('schedule.json','batches.json'):
        x=read(HERE/f'dense-{seed}'/name); y=read(HERE/f'moe-{seed}'/name)
        if name=='schedule.json': assert x['updates']==y['updates'] and x['definition']==y['definition']
        else: assert x==y
    assert a['artifacts']['model']['activation_checkpointing'] is False
    assert b['artifacts']['model']['activation_checkpointing'] is False
reference=records['dense',42]
for rec in records.values():
    for key in ('data_seal','upstream',*r.PROJECT_FILES.values()): assert rec[key]==reference[key]
    assert {k:v for k,v in rec['protocol'].items() if k!='seed'}=={k:v for k,v in reference['protocol'].items() if k!='seed'}
differences=[dict(seed=s,moe_minus_dense_bpb=records['moe',s]['metrics']['val_bpb']-records['dense',s]['metrics']['val_bpb']) for s in (42,43)]
prior=read(HERE/'prior-artifacts.json')
for name,h in prior.items():
    assert r.digest(ROOT/name)==h, 'Prior artifact changed: '+name
r.verify_seal(ROOT/'.autoresearch/cache',reference['data_seal'])
report=dict(status='completed',preliminary=True,rows=rows,paired_differences=differences,
    mean_paired_bpb_difference=sum(d['moe_minus_dense_bpb'] for d in differences)/2,
    identical_paired_batches=True,identical_schedule=True,matched_data_tokenizer_source_hashes=True,
    preserved_prior_files=len(prior),training_runs=4,additional_training_runs=0,
    schedule=reference['protocol']['schedule'],batch_tape_sha256=reference['protocol']['batch_tape_sha256'])
r.write_json(HERE/'comparison.json',report)
lines=['# Equal-token dense vs packed MoE: preliminary', '',
    'Four captured-source runs; seeds 42 and 43; depth 6 / width 384; activation checkpointing off.',
    'Every run completed 512 updates x 16,384 = **8,388,608 training tokens**, including all 11 timing-warmup updates.', '',
    '| Seed | Model | Validation BPB | Timed tokens/s | Wall s | Allocated / reserved MiB |',
    '|---|---|---:|---:|---:|---:|']
for row in rows:
    lines.append(f"| {row['seed']} | {row['model']} | {row['bpb']:.6f} | {row['timed_tokens_per_second']:,.1f} | {row['wall_seconds']:.2f} | {row['peak_allocated_mib']:.2f} / {row['peak_reserved_mib']:.2f} |")
lines += ['', '## Paired BPB differences (MoE minus dense; negative favors MoE)']
for d in differences: lines.append(f"- Seed {d['seed']}: {d['moe_minus_dense_bpb']:+.6f}")
lines += [f"- Mean: {report['mean_paired_bpb_difference']:+.6f}", '', '## Protocol and evidence',
    '- Same sealed TinyStories data/tokenizer, BF16 precision, context 512, microbatch 2, accumulation 16, full causal attention, optimizer hyperparameters and 65,536-token BPB evaluation.',
    '- LR uses zero-based step i / 512, not a timer: no LR warmup; multiplier 1 through i=256, then 2*(1-i/512). Last multiplier is 1/256. The endpoint at i=512 is not an extra update.',
    '- Muon momentum remains .85 + .1*min(i/300,1); Muon weight decay is .2*(1-i/512). Each schedule.json records all 512 entries and initial optimizer groups.',
    '- The first 11 updates are measurement warmup, not additional training or LR warmup. Timed throughput uses exactly 8,208,384 tokens over the remaining 501 uninstrumented updates.',
    '- A shared tape preserves the existing tokenizer and best-fit packing. A separate seeded CPU generator permutes its 8,192 microbatches; paired runs consumed identical indices and batch hash chains. Seeds 42/43 use different orders.',
    '- Tape packing/hashing is preparation, outside reported timed throughput and run wall time. Run wall time includes startup, initialization hashing, tape loading, warmup, training, checkpoint save, evaluation and result checks. Timed updates include host-to-device transfers, forward/backward and optimizer work; no profiler.',
    '- Full initialization hashes match the preflight for every run. Shared dense/MoE weights match within a seed; embeddings, routers and additional expert input weights change across seeds. Zero-initialized projections intentionally remain zero.',
    '- Project/upstream modules execute from verified snapshots under isolated Python. Protocol, schedule, dataset/tokenizer hashes and actual batch receipts match across paired runs.',
    f'- Preserved all {len(prior)} previously recorded artifact files byte-for-byte.', '', '## Parameters and utilization',
    '- Dense: 26,345,772 total and active parameters. MoE: 47,588,652 total; 26,354,988 structural active parameters. No memory table. Active counts do not include dispatch overhead or all-expert optimizer work.']
for row in rows:
    if row['model']=='moe':
        lines.append(f"- MoE seed {row['seed']}, train expert shares by layer (E0/E1/E2/E3), zero dropped tokens:")
        for i,shares in enumerate(row['train_expert_fractions']):
            lines.append(f"  Layer {i}: "+' / '.join(f'{x:.2%}' for x in shares))
lines += ['', '## Interpretation',
    'Equal tokens are not equal total parameters or compute. MoE retains packing, per-layer host boundary synchronization, smaller expert GEMMs and optimizer work across all experts. This experiment does not separately profile those costs.',
    'Two paired seeds provide preliminary evidence only, not a robust estimate of variance, generalization, or an efficiency gain. BPB excludes auxiliary routing loss. No further tuning, architecture features, training runs or cloud jobs were started.']
(HERE/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(report,indent=2),flush=True)
