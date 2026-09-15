"""Receipt-derived synthesis and operational evidence for the campaign report."""
import statistics

import ncp_campaign as c


def opening(result):
    pairs=result.get('frozen_comparisons')
    if not pairs or not all(x['completed_pairs']==4 for x in pairs['summaries'].values()):
        return []
    dense=pairs['summaries']['D6'];capacity=pairs['summaries']['FROZEN-CAP']
    return ['', '## Main findings', '',
        f'Frozen NCP failed the predeclared primary same-sign rule. Across four fresh paired seeds, '
        f'its mean difference from dense is {dense["mean_delta_bpb"]:+.6f} BPB, with '
        f'{dense["negative_signs"]}/4 seeds favoring NCP. It did not demonstrate an advantage over the '
        f'exactly parameter-matched residual MLP: mean NCP-minus-MLP {capacity["mean_delta_bpb"]:+.6f} BPB.',
        'Dense depth gains replicated on seeds43/44, including the separate fixed-width comparisons. '
        'NCP initialization diagnostics and ablations below are explanatory follow-ups, not candidate reselection.',
        'See [compact findings and measured costs](FINDINGS.md). All quality comparisons use the same token budget and validation corpus.']


def append(text,result):
    frozen=result.get('frozen_selection')
    if frozen:
        rows=[x for x in result['rows'] if x['status']=='completed' and x['label']==frozen['label'] and x['seed'] in frozen['seeds']]
        text.extend(['', '## Frozen codebook and gradient diagnostics', '',
            '| Seed | Target entries used per segment | Target perplexity | Argmax entries used | Concept accuracy | Injected / hidden RMS | Weighted auxiliary loss |',
            '|---:|---|---|---|---:|---:|---:|'])
        for row in rows:
            health=row['ncp_health']
            perplexity='/'.join(f'{v:.2f}' for v in health['target_perplexity'])
            text.append(f'| {row["seed"]} | {health["used_entries"]} | {perplexity} | '
                f'{row["prediction_argmax_utilization"]["used_entries"]} | {health["mean_next_concept_accuracy"]:.3f} | '
                f'{row["injected_to_hidden_rms_ratio"]:.3f} | {health["training_loss_means"]["weighted_total"]:.4f} |')
        all_gradient=all(all(value>0 for value in x['ncp_health']['final_update_gradient_norms'].values()) for x in rows)
        all_update=all(all(value>0 for value in x['ncp_health']['final_update_parameter_delta_norms'].values()) for x in rows)
        text.append('\nUtilization and concept accuracy use32 post-training training-tape microbatches. '
            'They are not held-out quality or semantic-concept evidence. The collapse gate uses target entries/perplexity; '
            'argmax usage does not describe every soft mixing weight. Auxiliary loss is never included in token BPB. '
            f'Every recorded final-update NCP parameter gradient is nonzero: {all_gradient}; '
            f'every recorded NCP parameter update is nonzero: {all_update}.')
    grid=result.get('depth_width_grid')
    if grid and grid['status']=='completed':
        text.extend(['', '## Depth and width synthesis', '',
            '| Change | Mean paired BPB delta | Mean update-time ratio | Parameter ratio |',
            '|---|---:|---:|---:|'])
        for axis in sorted({x['axis'] for x in grid['comparisons']}):
            rows=[x for x in grid['comparisons'] if x['axis']==axis]
            text.append(f'| {axis.replace("_"," ")} | {statistics.mean(x["delta_bpb"] for x in rows):+.6f} | '
                f'{statistics.mean(x["update_time_ratio"] for x in rows):.2f} | '
                f'{rows[0]["candidate_parameters"]/rows[0]["control_parameters"]:.2f} |')
        text.append('\nBoth seeds favor greater depth at each fixed width and greater width at each fixed depth. '
            'Similar measured runtime across widths applies to this native Windows microbatch2 setup; it does not imply equal FLOPs '
            'or prove a general hardware-efficiency advantage. No profiler-based cause is claimed.')
    text.extend(['', '## Next best experiment', '',
        'Compare dense D6-width768 with D12-width768 over longer fixed-token budgets on new paired seeds, '
        'keeping the optimizer policy fixed and evaluating once on an untouched test split. Record the quality-versus-time curve '
        'as a separate analysis. This tests whether the replicated depth benefit persists as training matures and whether its '
        'roughly doubled update time is worthwhile. Keep the present NCP configuration frozen for any later comparison against '
        'the residual-MLP control; these mixed results do not justify combining it with MoE or n-gram memory. '
        'No next campaign is launched automatically.'])
    audit_path=c.HERE/'audit-result.json'
    if audit_path.exists():
        audit=c.read(audit_path)
        text.extend(['', '## Saved-evidence audit and storage', '',
            f'Audit measured {audit["measured_at_utc"]}: {len(audit["verified_trials"])} completed trials checked; '
            f'{len(audit["preserved_noncompleted"])} noncompleted attempts retained. '
            f'Committed source archives byte-verified: {audit["all_sources_committed_and_byte_verified"]}.',
            f'Campaign logical files: {audit["measured_campaign_logical_bytes"]/2**30:.2f} GiB; '
            f'new trial checkpoints: {audit["measured_checkpoint_bytes"]/2**30:.2f} GiB; '
            f'free disk at audit: {audit["free_disk_bytes"]/2**30:.2f} GiB. '
            'Initial free disk was237GiB; the initial forecast was15–40GiB of new storage. '
            'Existing roughly20GiB of checkpoints remain preserved. No checkpoint copies were made solely for inference probes.'])
    for filename,key in [('batch-evidence-result.json','verified_trials'),('basis-evidence-result.json','checks')]:
        path=c.HERE/filename
        if path.exists():
            receipt=c.read(path)
            text.append(f'\n{filename}: {receipt["status"]}; {len(receipt[key])} checks. '+receipt['interpretation'])
    gate=c.HERE/'active-preflight.json'
    if gate.exists():
        receipt=c.read(gate)
        text.append('\nLatest shared gate: '+receipt['status']+'. CPU/CUDA logs, exact test/source hashes and full-context fit receipts are published in report.json.')
    text.extend(['', 'A reporting-only preflight failure from a legacy fixture missing its seed was preserved and repaired before '
        'subsequent GPU training. An isolated preview text-encoding problem was caught before integration; original and repaired '
        'preview evidence remain saved. Neither produced a training score. Temporary publication deferrals retained local evidence '
        'and were followed by verified pushes.'])
