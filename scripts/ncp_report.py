"""Artifact-only campaign report; no training or publication side effects."""
import json
import math
from pathlib import Path
import statistics

import autoresearch as r
import experiment_reports as reports
from ncp_campaign import HERE, read, remaining


def rows():
    output=[]
    for path in sorted(HERE.glob('trial-*/result.json')):
        record=read(path); row=dict(trial=path.parent.name,label=record['label'],seed=record['seed'],
            phase=record['phase'],status=record['status'],hypothesis=record['selection']['hypothesis'],
            error=record.get('error'),configuration=record.get('candidate'),wall_seconds=record.get('wall_seconds'),
            source_hashes={k:v for k,v in record.items() if k.endswith('_sha256')},
            report_id=reports.identity(path.parent,r.ROOT)[0])
        if record['status']=='completed':
            fixed=read(path.parent/'fixed-training.json'); training=read(path.parent/'training.json')
            model=read(path.parent/'model.json'); memory=read(path.parent/'memory.json')
            gpu=read(path.parent/'gpu-samples.json')['samples']
            board=[]
            for sample in gpu:
                try: board.append(float(sample.split(',')[1]))
                except (IndexError,ValueError): pass
            row.update(bpb=record['metrics']['val_bpb'],total_parameters=model['total_parameters'],
                active_parameters=model['active_parameters'],ncp_parameters=model.get('ncp_parameters',0),
                width=model['width'],memory_table_bytes=model['memory_table_bytes'],
                codebook_buffer_bytes=model.get('ncp_codebook_buffer_bytes',0),
                codebook_effective_bytes=model.get('ncp_effective_codebook_bytes',0),
                all_update_seconds=fixed['all_update_seconds'],
                timed_tokens_per_second=training['timed_training_tokens']/training['timed_training_seconds'],
                allocated_mib=memory['peak_allocated_bytes']/2**20,reserved_mib=memory['peak_reserved_bytes']/2**20,
                board_sampled_peak_mib=max(board) if board else None,ncp_health=record.get('ncp_health'))
            if row['ncp_health']:
                settings=record['candidate']['ncp']
                injected=(settings['feedback_scale']*row['ncp_health']['mean_feedback_rms']
                    if settings['mode']=='feedback' else 0.)
                row['injected_feedback_rms']=injected
                row['injected_to_hidden_rms_ratio']=injected/row['ncp_health']['mean_hidden_rms']
                samples=read(path.parent/'ncp-diagnostics.json')['samples']
                counts=[[sum(sample['prediction_counts'][i][j] for sample in samples)
                    for j in range(len(samples[0]['prediction_counts'][i]))]
                    for i in range(len(samples[0]['prediction_counts']))]
                perplexities=[]
                for values in counts:
                    probabilities=[value/sum(values) for value in values if value]
                    perplexities.append(math.exp(-sum(p*math.log(p) for p in probabilities)))
                row['prediction_argmax_utilization']=dict(used_entries=[sum(v>0 for v in values) for values in counts],
                    perplexity=perplexities,counts=counts,
                    interpretation='Post-training diagnostic of most likely code; does not describe all soft or signed mixing weights and does not change the predeclared target-code collapse gate')
        else:
            model_path=path.parent/'model.json'
            if model_path.exists():
                model=read(model_path)
                row.update(total_parameters=model['total_parameters'],active_parameters=model['active_parameters'])
            else:
                receipt=HERE/f'failed-counts-{"-".join(path.parent.name.split("-")[:2])}-result.json'
                if receipt.exists():
                    reconstructed=read(receipt)
                    assert reconstructed['trial']==path.parent.name
                    row.update(total_parameters=reconstructed['model']['total_parameters'],
                        active_parameters=reconstructed['model']['active_parameters'],parameter_counts_reconstructed=True)
        output.append(row)
    return output


def pairs(data,a,b,seeds):
    result=[]
    for seed in seeds:
        left=[x for x in data if x['label']==a and x['seed']==seed and x['status']=='completed']
        right=[x for x in data if x['label']==b and x['seed']==seed and x['status']=='completed']
        if not left or not right: continue
        # Independent unit is seed; repetitions never count as additional seeds.
        result.append(dict(seed=seed,candidate=a,control=b,candidate_bpb=statistics.mean(x['bpb'] for x in left),
            control_bpb=statistics.mean(x['bpb'] for x in right),
            delta_bpb=statistics.mean(x['bpb'] for x in left)-statistics.mean(x['bpb'] for x in right),
            candidate_repeats=len(left),control_repeats=len(right),
            update_time_ratio=statistics.mean(x['all_update_seconds'] for x in left)/statistics.mean(x['all_update_seconds'] for x in right)))
    return result


def write(final=False):
    data=rows(); complete=[x for x in data if x['status']=='completed']
    result=dict(kind='ncp_campaign',status='completed' if final else 'running',rows=data,
        depth_confirmation=pairs(data,'D12','D6',[43,44]),remaining_seconds=remaining())
    controls=[x for x in complete if x['label']=='D6' and x['seed']==42]
    eligible=[x for x in complete if x['seed']==42 and x['configuration'].get('ncp',{}).get('mode')=='feedback'
        and not x['ncp_health']['collapsed']
        and (x['configuration']['ncp']['prediction_weight']>0 or x['configuration']['ncp']['ce_weight']>0)]
    if controls and eligible:
        best=min(eligible,key=lambda x:x['bpb'])
        delta=best['bpb']-controls[0]['bpb']
        result['exploratory_best']=dict(label=best['label'],bpb=best['bpb'],delta_bpb=delta,
            screening_threshold_met=delta<=-.001,seed=42,trial=best['trial'],
            added_parameters=best['total_parameters']-controls[0]['total_parameters'],
            parameter_ratio=best['total_parameters']/controls[0]['total_parameters'],
            update_time_ratio=best['all_update_seconds']/controls[0]['all_update_seconds'])
    depth6=[x for x in complete if x['label']=='D6']
    depth12=[x for x in complete if x['label']=='D12']
    if depth6 and depth12:
        result['depth_capacity_tradeoff']=dict(depth6_width=depth6[0]['width'],depth12_width=depth12[0]['width'],
            depth6_parameters=depth6[0]['total_parameters'],depth12_parameters=depth12[0]['total_parameters'],
            parameter_ratio=depth12[0]['total_parameters']/depth6[0]['total_parameters'])
    result['ncp_ablations'] = []
    for feedback, auxiliary in [('NCP','AUX'),('N-RMS','N-RMS-AUX')]:
        result['ncp_ablations'].extend(pairs(data,feedback,auxiliary,[42]))
    freeze=HERE/'confirmation-selection.json'
    if freeze.exists():
        chosen=read(freeze)
        result['frozen_selection']=chosen
        result['ncp_confirmation']=pairs(data,chosen['label'],'D6',chosen['seeds'])
    r.write_json(HERE/'result.json',result)
    text=['# NCP campaign '+('final report' if final else 'progress'),'',
        f'{len(complete)} completed of {len(data)} attempted full trials. Budget: 11:41:32 to 19:41:32 UTC, 2026-09-15.',
        '512 updates and 8,388,608 tokens per full trial. Seed42 screens are exploratory. Lower BPB is better.','',
        (f'Strongest eligible selection-seed NCP: {result["exploratory_best"]["label"]}, '
         f'{result["exploratory_best"]["bpb"]:.6f} BPB, delta {result["exploratory_best"]["delta_bpb"]:+.6f} versus D6. '
         f'Adds {result["exploratory_best"]["added_parameters"]:,} parameters; measured update-time ratio '
         f'{result["exploratory_best"]["update_time_ratio"]:.2f}. '
         'This is a search result; independent confirmation is reported separately.'
         if 'exploratory_best' in result else 'No completed eligible NCP screening result yet.'),'',
        '## Implementation','',
        'Dense encoder pools complete multi-token chunks; causal chunk Transformers predict segmented discrete-codebook weights. '
        'Only predicted concepts feed the token decoder, delayed by k-1 positions. Detached future chunks supervise NCP MSE; '
        'VQ MSE fits a transformed frozen random codebook basis. Token BPB excludes both auxiliary losses.',
        'This is a simplified ConceptLM-inspired prototype, not a paper reproduction. Initial softmax feedback differs from the '
        'official GPT2/Pythia raw-logit multiplication; raw-logit variants are separately labeled. Native SDPA, initialization, positional features, codebook transforms '
        'and the small TinyStories fixed-token experiment also differ. Official revision: a0ab281286f5c0337c35de3181cc992c562eacaa.',
        'See [campaign plan](../../../docs/ncp-campaign.md) and the published source receipt for exact references.','',
        '## Every attempted trial','',
        '| Trial | Seed | BPB | Update s | Trial s | Timed tok/s | Total / active params | Alloc / reserved MiB | Target-code collapse |',
        '|---|---:|---:|---:|---:|---:|---|---|---|']
    for row in data:
        name=f'[{row["trial"]}](../{row["report_id"]}/README.md)'
        if row['status']!='completed':
            wall=f'{row["wall_seconds"]:.1f}' if row.get('wall_seconds') is not None else 'unavailable'
            memory_path=HERE/row['trial']/'memory.json'
            memory=read(memory_path) if memory_path.exists() else None
            peak=f'{memory["peak_allocated_bytes"]/2**20:.1f} / {memory["peak_reserved_bytes"]/2**20:.1f}' if memory else 'unavailable'
            counts=(f'{row["total_parameters"]:,} / {row["active_parameters"]:,}'+(' *' if row.get('parameter_counts_reconstructed') else '')
                if 'total_parameters' in row else 'unavailable')
            text.append(f'| {name} | {row["seed"]} | {row["status"]} | unavailable | {wall} | unavailable | {counts} | {peak} | unavailable |')
        else:
            h=row.get('ncp_health')
            text.append(f'| {name} | {row["seed"]} | {row["bpb"]:.6f} | {row["all_update_seconds"]:.1f} | {row["wall_seconds"]:.1f} | {row["timed_tokens_per_second"]:.0f} | {row["total_parameters"]:,} / {row["active_parameters"]:,} | {row["allocated_mib"]:.1f} / {row["reserved_mib"]:.1f} | {h["collapsed"] if h else "n/a"} |')
    text.append('\n* Early-failure parameter counts were reconstructed exactly on a meta device from captured source and logged model configuration. No missing performance measurement was reconstructed.')
    for heading,items in [('Depth6 versus depth12, reference LR .04',result['depth_confirmation']),('Frozen NCP confirmation',result.get('ncp_confirmation',[]))]:
        if heading.startswith('Depth6') and 'depth_capacity_tradeoff' in result:
            tradeoff=result['depth_capacity_tradeoff']
            text.extend(['',f'Depth6 uses width {tradeoff["depth6_width"]} and {tradeoff["depth6_parameters"]:,} parameters; '
                f'depth12 uses width {tradeoff["depth12_width"]} and {tradeoff["depth12_parameters"]:,} parameters '
                f'({tradeoff["parameter_ratio"]:.2f} times as many). This comparison changes both depth and width.'])
        text.extend(['','## '+heading,'','| Seed | Candidate | Control | Candidate BPB | Control BPB | Delta BPB | Update-time ratio |','|---:|---|---|---:|---:|---:|---:|'])
        for x in items: text.append(f'| {x["seed"]} | {x["candidate"]} | {x["control"]} | {x["candidate_bpb"]:.6f} | {x["control_bpb"]:.6f} | {x["delta_bpb"]:+.6f} | {x["update_time_ratio"]:.2f} |')
        if not items: text.append('No completed pair yet.')
        else:
            text.append(f'\nMean paired delta: {statistics.mean(x["delta_bpb"] for x in items):+.6f} BPB. '
                f'Mean update-time ratio: {statistics.mean(x["update_time_ratio"] for x in items):.2f}.')
    text.extend(['','## Feedback versus auxiliary-only ablations','',
        'Negative delta favors predicted-concept feedback. These selection-seed comparisons are exploratory.','',
        '| Feedback configuration | Auxiliary-only configuration | Seed | Feedback BPB | Auxiliary BPB | Delta |',
        '|---|---|---:|---:|---:|---:|'])
    for x in result['ncp_ablations']:
        text.append(f'| {x["candidate"]} | {x["control"]} | {x["seed"]} | {x["candidate_bpb"]:.6f} | {x["control_bpb"]:.6f} | {x["delta_bpb"]:+.6f} |')
    text.extend(['','## Attempts, decisions and failures',''])
    for row in data:
        text.append(f'- {row["trial"]}: {row["hypothesis"]}'+(f' Failure: {row["error"]}' if row.get('error') else ''))
    text.extend(['','## Correctness and diagnosis','',
        'The first two unit-weight attempts hit an inherited finite total-loss100 guard. A source-verified CPU checkpoint probe '
        'found dense hidden RMS12.54 too, so those stops do not establish NCP-specific divergence. The corrected guard checks '
        'token CE separately and still rejects nonfinite total loss. Initial failed attempts remain preserved, and their '
        'historical hypotheses using the word divergence are superseded by this diagnosis. Completed BPB runs never hit that gate.',
        'Trial25 copied a newer controller file while its long-running parent retained an earlier imported controller. '
        'Both versions and a correction receipt are retained. Their trial, preflight, candidate and health function bodies '
        'are identical; the difference is a GPU lock wrapper. The loaded-controller reference is reconstructed from the '
        'same-process import history, not direct process-memory inspection. Captured training-child sources and data are '
        'independently verified. The next controller archives immutable startup source bytes to prevent recurrence.',
        'CPU/CUDA tests cover prefix causality, future-label isolation, VQ/encoder gradients, optimizer coverage, '
        'save/load, codebook learning and evaluation immutability. Every completed training child executes captured sources; '
        'the final evidence audit also verifies saved checkpoints and committed source-archive bytes.',
        '', '## Measurement limits','',
        'Equal-token quality comparisons; measured runtime is a separate cost axis. No equal-time quality claim. '
        'Depth changes width too. Timed throughput excludes the first11 updates; all-update time includes them. '
        'Trial wall time includes preparation, child execution and verification, excluding reporting/publication. '
        'Allocator peaks exclude driver/desktop use. Whole-board sampled VRAM, dictionary bytes, exact configurations, '
        'source/data/checkpoint hashes, auxiliary losses and utilization are in the JSON receipts. '
        'Active counts describe structural training participation, not amortized per-token compute. NCP runs at chunk rate; '
        'the capacity-control MLP runs at token rate, and auxiliary-only concepts do not feed token logits. '
        'Diagnostic feedback_rms is the unscaled prediction; injected_feedback_rms applies the configured gain and is zero for auxiliary-only runs. '
        'For raw-logit mixing, reported entropy describes softmax classification probabilities, not the signed reconstruction weights. '
        'Codebook assignments do not prove semantic concepts. Repeated validation selection is exploratory, not held-out generalization.',
        'All artifacts are retained locally. No cloud, dependency upgrades, paid services or deletion.'])
    (HERE/'summary.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    for path in HERE.glob('trial-*/result.json'): reports.emit(path.parent)
    reports.emit(HERE); reports.index()
    destination=r.ROOT/'reports/experiments'/reports.identity(HERE,r.ROOT)[0]
    (destination/'CAMPAIGN.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    return result


if __name__=='__main__':
    import sys
    write('--final' in sys.argv)
