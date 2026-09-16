"""Verify paired receipts and publish compact foundation evidence without corpora/weights."""
from datetime import datetime
import json
import math
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
from foundation_data import ROOT,digest,write

HOME=ROOT/'runs/foundation_campaign'
DEST=ROOT/'reports/foundation-v1'
SEEDS=(201,202,203)
ARMS=('control','mixture','tokenizer')
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))

def verify_pairs():
    rows={}
    for seed in SEEDS:
        for arm in ARMS:
            name=f'pilot-{arm}-s{seed}';folder=HOME/name;r=read(folder/'result.json')
            assert r['status']=='completed' and r['seed']==seed and r['updates']==512
            assert r['training_tokens']==8388608 and sum(r['stream']['source_tokens'].values())==8388608
            assert r['parameters']['total_parameters']==r['parameters']['active_parameters']==135267480
            assert r['parameters']['memory_table_bytes']==r['parameters']['ncp_parameters']==r['parameters']['expert_parameters']==0
            assert r['checkpointing'] is False
            assert digest(folder/'checkpoint.pt')==r['checkpoint_sha256']
            losses=read(folder/'losses.json');assert len(losses)==512 and [x['step'] for x in losses]==list(range(1,513))
            assert all(math.isfinite(x['mean_loss']) and math.isfinite(x['min_loss']) and math.isfinite(x['max_loss']) for x in losses)
            replay=read(HOME/(name+'-replay.json'));assert replay['status']=='verified'
            assert replay['run_result_sha256']==digest(folder/'result.json')
            assert replay['checkpoint_sha256']==r['checkpoint_sha256']
            r['replay']=replay;r['run']=name;r['result_sha256']=digest(folder/'result.json');rows[seed,arm]=r
        control=rows[seed,'control']
        for arm in ARMS[1:]:
            other=rows[seed,arm]
            for key in ('candidate','parameters','initialization_sha256','dataset_fingerprint','source_hashes'):
                assert control[key]==other[key],('Unfair pair',seed,arm,key)
            assert control['validation']['input_fingerprint']==other['validation']['input_fingerprint']
            assert read(HOME/control['run']/'schedule.json')==read(HOME/other['run']/'schedule.json')
            for domain in ('general','technical'):
                assert control['validation']['domains'][domain]['bytes']==other['validation']['domains'][domain]['bytes']
        assert rows[seed,'mixture']['tokenizer']==control['tokenizer']
        assert rows[seed,'mixture']['stream']['stream_sha256']==control['stream']['stream_sha256']
        assert rows[seed,'tokenizer']['general_weight']==control['general_weight']==.8
    assert len({rows[s,'control']['initialization_sha256'] for s in SEEDS})==3
    return rows

def main():
    rows=verify_pairs();DEST.mkdir(parents=True,exist_ok=True)
    for source,name in [(ROOT/'data/foundation-v1/manifest.json','data-manifest.json'),
                        (HOME/'tokenizer/audit.json','tokenizer-audit.json'),
                        (HOME/'source-token-counts.json','source-token-counts.json'),
                        (HOME/'vocabulary-cost.json','vocabulary-cost.json'),
                        (HOME/'data-independent-audit.json','data-audit.json'),
                        (HOME/'baseline-reproduction/verification-repair.json','baseline-reproduction.json'),
                        (HOME/'pilot-plan.json','pilot-plan.json'),
                        (HOME/'schedule.json','schedule.json')]:
        shutil.copy2(source,DEST/name)
    pairs={}
    for arm in ARMS[1:]:
        deltas=[rows[s,arm]['validation']['aggregate_bpb']-rows[s,'control']['validation']['aggregate_bpb'] for s in SEEDS]
        domains={d:statistics.mean(rows[s,arm]['validation']['domains'][d]['bpb']-rows[s,'control']['validation']['domains'][d]['bpb'] for s in SEEDS) for d in ('general','technical')}
        ratios=[rows[s,arm]['training']['timed_tokens_per_second']/rows[s,'control']['training']['timed_tokens_per_second'] for s in SEEDS]
        passed=all(x<0 for x in deltas) and statistics.mean(deltas)<=-.01 and max(domains.values())<=.02 and min(ratios)>=.8
        pairs[arm]=dict(seeds=list(SEEDS),paired_bpb_deltas=deltas,mean_delta=statistics.mean(deltas),sd_delta=statistics.stdev(deltas),
            domain_mean_deltas=domains,throughput_ratios=ratios,screen_passed=passed,
            promotion='Not promoted: pilot alone is insufficient; untouched confirmation and review remain required')
    compact=[]
    for (seed,arm),r in rows.items():
        compact.append({k:r[k] for k in ('run','seed','training_tokens','general_weight','candidate','parameters','dataset_fingerprint','tokenizer','git_revision','source_hashes','initialization_sha256','checkpoint_sha256','result_sha256','stream','training','validation','wall_seconds','tinystories_legacy_bpb','replay','hardware','runtime')})
    write(DEST/'results.json',dict(status='verified',rows=compact,pairs=pairs,verified_at=datetime.now().astimezone().isoformat(),reporter_sha256=digest(__file__)))
    lines=['# Dense foundation campaign: verified pilot results','',
        '**Keep the frozen dense architecture and original mainline tokenizer.** The new data/evaluation foundation is operational. The comparisons below are small-corpus pilots, not evidence of general reasoning or coding capability. No architecture, mixture or tokenizer is automatically promoted.','',
        '## Verified facts','',
        '- Frozen dense D12/width768, LR.04, BF16, context512, microbatch2, batch16,384, checkpointing off. Total/active parameters135,267,480; memory table0.',
        '- Pinned WikiText-2 raw and CPython documentation:37,263,470downloaded bytes;1,049accepted documents; corpus fingerprint `debe7de4f51d9487e316aedf1a7dc828350d7a77f20d99a913a0143e456b121f`.',
        '- Current tokenizer SHA256 `4d1991faca1391dbc13ba13ef7ed19a3ae77dde50d4a6927fd068090b80cda5c`; candidate `736bb1a35ce3b2a3143095bfc1390858d67dd7f606fda25137b83095bace0102`. Both8192tokens; candidate retraining was byte-identical.',
        '- Every paired run passed parameter, initialization, token-budget, schedule, data, evaluator-input and checkpoint checks; every validation checkpoint was independently replayed. Exact code/runtime/hardware/source hashes and per-domain losses are in [results.json](results.json).','',
        '## Baseline reproduction','']
    b=read(DEST/'baseline-reproduction.json');t=b['training']
    lines += [f"Seed101,8,388,608tokens: legacy TinyStories BPB **{b['metrics']['val_bpb']:.6f}**, reference.589157, absolute difference{b['absolute_reference_bpb_difference']:.6f}. Timed throughput{t['timed_training_tokens']/t['timed_training_seconds']:.0f}tok/s; wall{b['wall_seconds']:.1f}s; peak allocated/reserved{b['memory']['peak_allocated_bytes']/2**20:.1f}/{b['memory']['peak_reserved_bytes']/2**20:.1f}MiB. The wrapper's missing-metadata failure is retained separately from the successful artifact verification.",'',
        '## Controlled results','',
        'Each row:512updates,8,388,608tokens, independent initialization. Same-token comparisons are not same-byte exposure or equal compute. Primary metric is aggregate exact-byte BPB; lower is better. Allocator memory excludes desktop/driver use.','',
        '| Arm | Seed | Aggregate BPB | General BPB | Technical BPB | Timed tok/s | Update s | Wall s | Alloc/reserved MiB |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in compact:
        v=r['validation'];t=r['training'];arm=r['run'].split('-')[1]
        lines.append(f"| {arm} | {r['seed']} | {v['aggregate_bpb']:.6f} | {v['domains']['general']['bpb']:.6f} | {v['domains']['technical']['bpb']:.6f} | {t['timed_tokens_per_second']:.0f} | {t['all_update_seconds']:.1f} | {r['wall_seconds']:.1f} | {t['peak_allocated_bytes']/2**20:.1f}/{t['peak_reserved_bytes']/2**20:.1f} |")
    lines+=['','Control=current tokenizer80/20general/technical source token mass. Mixture=current50/50. Tokenizer=candidate80/20. Actual sampled masses and repeated corpus passes are in each stream receipt.','']
    for arm,p in pairs.items():
        lines += [f"**{arm} vs control:** mean paired BPB delta{p['mean_delta']:+.6f}, paired SD{p['sd_delta']:.6f}; deltas{[round(x,6) for x in p['paired_bpb_deltas']]}. Domain mean deltas{p['domain_mean_deltas']}. Predeclared screen passed: **{p['screen_passed']}**. No promotion.",'']
    lines += ['## Inferences and limitations','',
        '- Data ingestion trains cleanly. There is no fair TinyStories-only versus new-corpus training comparison, so these results do **not** establish that more data helped.',
        '- Tokenizer compression, raw-byte exposure, context in bytes and repeated passes differ. Only the recorded paired results establish improvement or regression on this held-out pilot; no wider capability claim follows.',
        '- The legacy per-token decoded Unicode byte table overcounts split UTF-8 pieces. New evaluation uses exact document bytes. Historical BPB is preserved, not pooled with the new series.',
        '- RST/Wikipedia markup, small topic coverage, coarse language filtering and lexical rather than semantic dedup remain limitations. Test corpus model scores remain unopened.','',
        '## Failures','',
        '- Baseline wrapper omitted condition metadata after successful training; original failed result retained, independent correction receipt verifies the artifacts.',
        '- First new-path smoke hit noncontiguous target tensors in the pinned model before completing an update; contiguous copies fixed it; a new smoke completed.',
        '- Initial Git commit failed because author identity was unset. Command-local campaign assistant identity was used; no global Git configuration changed.','',
        '## Decisions made','',
        '- Dense mainline; experimental NCP/MoE/n-grams preserved and excluded.',
        '- Preserve original tokenizer and baseline; use paired pilot evidence to decide the next confirmation, not automatic promotion.',
        '- No cloud, paid services, dependency upgrades, destructive cleanup or push. Unrelated user files remain untouched.','',
        '## Unresolved / not completed','',
        '- No broad science, mathematics or standalone code corpus; no general-capability benchmark claims.',
        '- No depth/width search or checkpointing re-sweep on the new data.',
        '- No optimizer/RNG-resumable training checkpoints (retained weights are evaluation checkpoints).',
        '- No untouched final-test promotion stage, external benchmark contamination certification, or distributed/off-machine reproduction.','',
        '## Next three highest-value actions','',
        '1. Review per-domain paired results and fixed samples; freeze one follow-up hypothesis before opening any test scores.',
        '2. Add a small explicitly licensed science/math/code source with the same provenance, dedup and held-out controls; improve markup handling independently.',
        '3. Run fresh-seed confirmation at a larger unique-data/token budget, then one frozen test stage before considering tokenizer or mixture promotion.','',
        '## Reproduce / continue','',
        'See [campaign protocol](../../docs/foundation-campaign.md) and [tokenizer audit](../../docs/tokenizer-foundation-v1.md). Example validation replay (native PowerShell):','',
        '```powershell',
        "$py = '.autoresearch/upstream/.venv/Scripts/python.exe'",
        '& $py scripts/evaluate_foundation.py runs/foundation_campaign/pilot-control-s201 --output runs/foundation_campaign/manual-replay.json',
        '& $py scripts/report_foundation.py','```','',
        'Local artifacts, logs, weights, candidate tokenizer and evaluation text remain under `runs/foundation_campaign/`; corpus under `data/foundation-v1/`. Compact manifests and receipts only are published here.']
    (DEST/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(pairs,indent=2))

if __name__=='__main__':main()
