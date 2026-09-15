"""Compact campaign progress and final receipts; never runs training."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import statistics
import sys

import autoresearch as r
import experiment_reports as reports
from overnight_campaign import HERE, ROOT, read, rank


def main(final=False):
    rows, records, failed, in_progress = [], [], [], []
    for path in sorted(HERE.glob('trial-*/result.json')):
        record = read(path); directory = path.parent
        selection_path = directory / 'selection.json'
        if selection_path.exists():
            decision = read(selection_path)
            r.write_json(directory / 'analysis.json', {'selection': {k:v for k,v in decision.items() if k != 'evidence'},
                'selection_with_evidence_sha256': r.digest(selection_path),
                'selection_with_evidence_local_path': str(selection_path.relative_to(ROOT)),
                'comparison_class': 'equal_token_512_updates',
                'initialization_pairing': read(directory / 'initialization-pairing.json') if (directory / 'initialization-pairing.json').exists() else None})
        if record['status'] != 'completed':
            if record['status'] in ('running', 'prepared'):
                in_progress.append({'run': directory.name, 'status': record['status']})
                continue
            failed.append({'run': directory.name, 'status': record['status'], 'error': record.get('error')})
            if record['status'] != 'running':
                reports.emit(directory)
            continue
        record['_path'] = str(directory); records.append(record)
        training = record['artifacts']['training']; model = record['artifacts']['model']; memory = record['memory']
        assert training['training_tokens'] == 8388608 and training['optimizer_updates'] == 512
        row = dict(run=directory.name, label=record['label'], phase=record['phase'], seed=record['seed'],
            bpb=record['metrics']['val_bpb'], total_parameters=model['total_parameters'],
            active_parameters=model['active_parameters'], width=model['width'],
            memory_table_bytes=model['memory_table_bytes'], tokens=training['training_tokens'],
            timed_tokens_per_second=training['timed_training_tokens']/training['timed_training_seconds'],
            all_update_seconds=read(directory / 'fixed-training.json')['all_update_seconds'],
            wall_seconds=record['wall_seconds'], allocated_mib=memory['peak_allocated_bytes']/2**20,
            reserved_mib=memory['peak_reserved_bytes']/2**20,
            expert_utilization=record['artifacts']['routing']['train']['fractions'])
        rows.append(row); reports.emit(directory)
    choices = read(HERE / 'search-space.json')
    ranking = rank(records, choices) if records else []
    freeze = read(HERE / 'confirmation-selection.json') if (HERE / 'confirmation-selection.json').exists() else None
    paired = []
    if freeze:
        winner = freeze['winner']
        for seed in (42, 43, 44):
            candidate = [x['bpb'] for x in rows if x['label'] == winner['label'] and x['seed'] == seed]
            control = [x['bpb'] for x in rows if x['label'] == winner['control'] and x['seed'] == seed]
            if candidate and control:
                paired.append(dict(seed=seed, candidate_mean_bpb=statistics.mean(candidate),
                    control_mean_bpb=statistics.mean(control), candidate_repeats=len(candidate),
                    control_repeats=len(control), difference=statistics.mean(candidate)-statistics.mean(control)))
    preservation = None
    if final:
        prior = read(HERE / 'prior-artifacts.json')
        changed = [name for name,h in prior.items() if not (ROOT/name).is_file() or r.digest(ROOT/name) != h]
        preservation = dict(checked_files=len(prior), changed_or_missing=changed)
        r.write_json(HERE / 'preservation.json', preservation)
        assert not changed, 'Prior artifacts changed'
    status = 'failed' if any(x['status'] not in ('running','prepared') for x in failed) else ('completed' if final else 'running')
    result = dict(kind='adaptive_equal_token_campaign', status=status,
        settings=read(HERE/'contract.json'), runs=len(rows), seeds=sorted({x['seed'] for x in rows}),
        paired_differences=paired, limitations=[
            'All trial scores are equal-token, not equal-time or equal-compute comparisons.',
            'Repeated seed42 runs measure execution variability, not independent seeds.',
            'Adaptive selection reuses validation; selection bias and no untouched test.',
            'MoE expert-count changes do not fully pair router/extra-expert initialization.',
            'Historical M divergence remains unexplained; no deterministic-training claim.',
            'NCP deferred; no combined architecture candidates executed.'])
    r.write_json(HERE/'result.json',result)
    r.write_json(HERE/'comparison.json',dict(ranking=ranking, frozen_confirmation=freeze,
        paired_differences=paired, failures=failed, in_progress=in_progress, preservation=preservation,
        gpu_seconds=sum(x['all_update_seconds'] for x in rows),
        child_wall_seconds=sum(x['wall_seconds'] for x in rows),
        disk_free_bytes=shutil.disk_usage(ROOT).free))
    # Bound each compact input below the reporting workflow's size limit.
    for offset in range(0,len(rows),30):
        r.write_json(HERE/f'rows-{offset//30:03d}-result.json',dict(rows=rows[offset:offset+30]))
    lines=['# Overnight adaptive campaign', '', f'Status: {status}. Completed full trials: {len(rows)}. No candidate-count cap.', '',
        'All rows use 512 updates / 8,388,608 tokens, checkpointing off, context512, microbatch2, fixed validation and the same step schedule. '
        'These are equal-token comparisons; time is a separate cost axis, not an equal-time leaderboard.', '',
        '| Run / label | Phase / seed | BPB | Timed tok/s | GPU update s | Child wall s | Alloc / reserved MiB | Total / active params |',
        '|---|---|---:|---:|---:|---:|---|---|']
    for row in rows:
        lines.append(f"| {row['run']} | {row['phase']} / {row['seed']} | {row['bpb']:.6f} | {row['timed_tokens_per_second']:.1f} | {row['all_update_seconds']:.1f} | {row['wall_seconds']:.1f} | {row['allocated_mib']:.1f} / {row['reserved_mib']:.1f} | {row['total_parameters']:,} / {row['active_parameters']:,} |")
    lines += ['', 'Timed throughput excludes the first11 updates; all-update GPU time includes them. '
        'Child wall includes startup, diagnostics, checkpoint saving and evaluation. PyTorch allocator memory excludes other processes/driver allocations. '
        'Exact widths, memory-table bytes, expert utilization and per-trial selection reasons are in compact JSON and child receipts.', '',
        '## Confirmation', '', json.dumps(paired,indent=2) if paired else 'Confirmation pending or no frozen comparison.', '',
        'Averaging repeats within a seed does not create new independent seeds. Three paired seeds, when complete, '
        'remain preliminary; repeated validation-guided selection is not an untouched test.', '',
        '## Boundaries and provenance', '',
        'No NCP, architecture combinations, dependency changes, cloud jobs, paid services or deletion. '
        'Each training child executes captured source/configuration; publishing commits are not retrospective execution provenance. '
        'Failed/invalid trials are retained and excluded from quality rankings. Hypotheses were saved before process launch.', '',
        '## Failures', '', json.dumps(failed,indent=2) if failed else 'None.', '',
        '## Preservation', '', json.dumps(preservation,indent=2) if preservation else 'Full preservation verification reserved for final report.']
    body='\n'.join(lines)+'\n'
    (HERE/'campaign-report.md').write_text(body,encoding='utf-8')
    reports.emit(HERE); reports.index()
    report_id,_=reports.identity(HERE,ROOT)
    (ROOT/'reports/experiments'/report_id/'campaign-report.md').write_text(body,encoding='utf-8')
    print(json.dumps(dict(status=status,completed=len(rows),report_id=report_id,paired_differences=paired),indent=2))


if __name__=='__main__':
    main('--final' in sys.argv)
