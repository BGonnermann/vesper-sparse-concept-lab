"""Final acceptance/report for the frozen four-run memory experiment; no training."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import autoresearch as r
import experiment_reports as reports
HERE=ROOT/'runs/autoresearch/ngram-20260915'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))


def main():
    rows=[];records={};schedules={};batch_receipts={}
    for label in ('D','DG','M','MG'):
        out=HERE/label;record=read(out/'result.json');assert record['status']=='completed'
        r.validate_execution(out,record['snapshot_files']);r.validate_run_artifacts(out,record)
        records[label]=record;schedules[label]=read(out/'schedule.json');batch_receipts[label]=read(out/'batches.json')
        model=record['artifacts']['model'];training=record['artifacts']['training']
        assert training['optimizer_updates']==512 and training['training_tokens']==8388608
        rows.append({'label':label,'seed':42,'validation_bpb':record['metrics']['val_bpb'],
            'total_parameters':model['total_parameters'],'active_parameters':model['active_parameters'],
            'memory_table_bytes':model['memory_table_bytes'],'tokens':training['training_tokens'],
            'timed_tokens':training['timed_training_tokens'],'timed_seconds':training['timed_training_seconds'],
            'timed_tokens_per_second':training['timed_training_tokens']/training['timed_training_seconds'],
            'wall_seconds':record['wall_seconds'],
            'peak_allocated_mib':record['memory']['peak_allocated_bytes']/2**20,
            'peak_reserved_mib':record['memory']['peak_reserved_bytes']/2**20,
            'expert_utilization':record['artifacts']['routing']['train']['fractions'],
            'dropped_tokens':record['artifacts']['routing']['train']['dropped_tokens'],
            'memory_diagnostics':read(out/'ngram-diagnostics.json')})
        diagnostic=rows[-1]['memory_diagnostics']
        if diagnostic['enabled']:
            samples=diagnostic['samples']
            rows[-1]['memory_summary']={'sampled_gate_mean':sum(s['gate_mean'] for s in samples)/len(samples),
                'sampled_gate_min':min(s['gate_min'] for s in samples),
                'sampled_gate_max':max(s['gate_max'] for s in samples),
                'final_update_gradient_norms':diagnostic['final_update_gradient_norms'],
                'final_update_parameter_delta_norms':diagnostic['final_update_parameter_delta_norms']}
    for label,record in records.items():
        for key in ('protocol','data_seal','upstream','seed',*r.PROJECT_FILES.values()):assert record[key]==records['D'][key]
        assert schedules[label]['updates']==schedules['D']['updates']
        assert batch_receipts[label]==batch_receipts['D']
    bpb={row['label']:row['validation_bpb'] for row in rows}
    contrasts={'DG_minus_D':bpb['DG']-bpb['D'],'MG_minus_M':bpb['MG']-bpb['M']}
    contrasts['interaction']=contrasts['MG_minus_M']-contrasts['DG_minus_D']
    prior=read(HERE/'prior-artifacts.json')
    for name,h in prior.items():assert r.digest(ROOT/name)==h,name
    r.verify_seal(ROOT/'.autoresearch/cache',records['D']['data_seal'])
    report={'status':'completed','preliminary':True,'seeds':[42], 'rows':rows,
        'primary_contrasts':contrasts,'protocol':records['D']['protocol'],
        'plan_revision':read(HERE/'plan-revision.json'),'preflight':read(HERE/'preflight.json'),
        'matched_settings_and_batches':True,'prior_artifacts_unchanged':len(prior),
        'ngram_tape_diagnostics_sha256':r.digest(HERE/'ngram-tape-diagnostics.json'),
        'limitations':['One seed and fixed condition order; preliminary only.',
            'Equal tokens are unequal parameters/compute; added-capacity benefit not isolated.',
            'Bitwise training reproducibility is not established; use the fresh controls, not historical BPB as a substitute.',
            'Memory diagnostics are final-update/sampled final-model measurements, not all-update distributions.',
            'No full-model KV-cache/resumable-training claim; only bounded lookup and CPU state fixtures.'],
        'training_runs':4,'extra_seeds':0,'cloud_jobs':0}
    r.write_json(HERE/'comparison.json',report)
    lines=['# Causal n-gram memory: four-condition preliminary result','',
        'Frozen plan: `'+report['plan_revision']['git_revision']+'`; SHA256 `'+report['plan_revision']['plan_sha256']+'`.',
        'D/DG/M/MG all completed seed42, 512 updates, 8,388,608 training tokens, checkpointing off.',
        'Same sealed TinyStories/tokenizer, context512, microbatch2, accumulation16 and 65,536 evaluation tokens.',
        '', '| Condition | BPB | Timed tokens/s | Wall seconds | Allocated / reserved MiB | Total / active parameters |',
        '|---|---:|---:|---:|---:|---:|']
    for row in rows:lines.append(f"| {row['label']} | {row['validation_bpb']:.6f} | {row['timed_tokens_per_second']:,.1f} | {row['wall_seconds']:.2f} | {row['peak_allocated_mib']:.2f} / {row['peak_reserved_mib']:.2f} | {row['total_parameters']:,} / {row['active_parameters']:,} |")
    lines+=['','## Primary contrasts (negative favors memory)',f"- DG minus D: {contrasts['DG_minus_D']:+.6f} BPB.",
        f"- MG minus M: {contrasts['MG_minus_M']:+.6f} BPB.",f"- Interaction: {contrasts['interaction']:+.6f}; no significance claim.",
        '', '## Controls and timing',
        '- Step schedule unchanged: no LR warmup; multiplier1 through zero-based256, then2*(1-i/512); final1/256. Muon momentum/decay and existing optimizer groups unchanged; added memory uses frozen AdamW group.',
        '- All11 timing-warmup updates are included in the512 budget. Timed throughput uses8,208,384 tokens over501 updates. CPU tape preparation, collision analysis, final gradient/delta inspection and32 post-training diagnostic inference batches are outside timed training.',
        '- Wall time includes launch, initialization, training, diagnostics, checkpoint save, evaluation and result verification. No profiler or extra optimizer updates.',
        '- Captured actual executed hashes, paired initializations, complete consumed batch receipts and schedules verified. The later publishing commit is not the training revision.',
        f'- All{len(prior)} previously recorded artifact files remain byte-identical. No deletion.',
        '', '## Memory diagnostics']
    tape=read(HERE/'ngram-tape-diagnostics.json')
    for n,value in tape.items():
        lines.append(f"- Order{n}: valid rate {value['valid_rate']:.4%}; {value['distinct_suffixes']:,} distinct suffixes; {value['accessed_rows']}/8192 occupied rows; {value['colliding_extra_suffixes']:,} extra distinct suffixes sharing buckets.")
    lines+=['- Tables total4MiB; added1,098,113 parameters. Exact per-bucket occupancy/access counts remain in the hashed local diagnostics artifact.',
        '- Per-condition final gradient/update norms and32 sampled gate/rate diagnostics are included in comparison.json and each compact run report. These are not validation fitting or full-training gate distributions.',
        '- VRAM peaks cover the complete process, including diagnostics; diagnostic work is excluded from timed throughput.',
        '- Historical seed42 packed MoE BPB was0.638883 versus the fresh M control reported above. Initialization/data receipts match, but implementation snapshots differ; this is not a controlled test of the cause of that difference.',
        '', '## Limitations']+[ '- '+s for s in report['limitations'] ]
    lines+=['','## Sampled final-model memory observations']
    for row in rows:
        if 'memory_summary' in row:
            s=row['memory_summary']
            lines.append(f"- {row['label']}: gate mean {s['sampled_gate_mean']:.6f}, range {s['sampled_gate_min']:.6f} to {s['sampled_gate_max']:.6f}.")
            for table in ('E2','E3'):
                lines.append(f"- {row['label']} {table}: final gradient L2 {s['final_update_gradient_norms'][table]:.6g}; final update L2 {s['final_update_parameter_delta_norms'][table]:.6g}.")
    lines+=['','Stopped after the four authorized runs. No tuning, NCP, deletion or cloud jobs.']
    (HERE/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    for label in records:reports.emit(HERE/label)
    reports.emit(HERE);reports.index()
    print(json.dumps({'contrasts':contrasts,'rows':[{k:v for k,v in row.items() if k not in ('memory_diagnostics','expert_utilization')} for row in rows]},indent=2),flush=True)


if __name__=='__main__':main()
