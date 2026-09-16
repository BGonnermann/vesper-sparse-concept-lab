"""Verify and report the predeclared six-run training-pool replacement comparison."""
from datetime import datetime
import json
from pathlib import Path
import shutil
import statistics
from foundation_data import ROOT,digest,write
from foundation_baseline import HOME

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))

def main():
    rows=[];pairs=[]
    for seed in (211,212,213):
        pair=[]
        for arm in ('old','expanded'):
            name=f'data-{arm}-s{seed}';folder=HOME/name;r=read(folder/'result.json');assert r['status']=='completed'
            assert r['updates']==512 and r['training_tokens']==8388608
            assert r['parameters']['total_parameters']==r['parameters']['active_parameters']==135267480
            assert r['checkpointing'] is False and r['parameters']['memory_table_bytes']==0
            assert digest(folder/'checkpoint.pt')==r['checkpoint_sha256']
            assert len(read(folder/'losses.json'))==512
            replay=read(HOME/(name+'-replay.json'));assert replay['status']=='verified' and replay['run_result_sha256']==digest(folder/'result.json')
            r['run']=name;r['replay']=replay;r['result_sha256']=digest(folder/'result.json');pair.append(r);rows.append(r)
        a,b=pair
        for key in ('candidate','parameters','initialization_sha256','tokenizer','source_hashes','general_weight'):
            assert a[key]==b[key],key
        assert a['stream']['source_tokens']==b['stream']['source_tokens']
        assert a['stream']['stream_sha256']['technical']==b['stream']['stream_sha256']['technical']
        assert a['stream']['stream_sha256']['general']!=b['stream']['stream_sha256']['general']
        assert a['validation']['input_fingerprint']==b['validation']['input_fingerprint']
        assert read(HOME/a['run']/'schedule.json')==read(HOME/b['run']/'schedule.json')
        pairs.append(dict(seed=seed,aggregate_delta=b['validation']['aggregate_bpb']-a['validation']['aggregate_bpb'],
            domain_deltas={d:b['validation']['domains'][d]['bpb']-a['validation']['domains'][d]['bpb'] for d in ('general','technical')},
            throughput_ratio=b['training']['timed_tokens_per_second']/a['training']['timed_tokens_per_second']))
    assert len({x['initialization_sha256'] for x in rows})==3
    delta=statistics.mean(p['aggregate_delta'] for p in pairs);sd=statistics.stdev(p['aggregate_delta'] for p in pairs)
    domains={d:statistics.mean(p['domain_deltas'][d] for p in pairs) for d in ('general','technical')}
    passed=all(p['aggregate_delta']<0 for p in pairs) and delta<=-.01 and max(domains.values())<=.02 and min(p['throughput_ratio'] for p in pairs)>=.8
    dest=ROOT/'reports/foundation-v1';write(dest/'expanded-results.json',dict(status='verified',pairs=pairs,rows=rows,mean_delta=delta,sd_delta=sd,domain_mean_deltas=domains,screen_passed=passed,reporter_sha256=digest(__file__),verified_at=datetime.now().astimezone().isoformat()))
    for source,name in [(ROOT/'data/foundation-expanded-v1/manifest.json','expanded-data-manifest.json'),(HOME/'expanded-tokenizer-audit.json','expanded-tokenizer-audit.json'),(HOME/'expanded-plan.json','expanded-plan.json')]:shutil.copy2(source,dest/name)
    lines=['# Bounded unique-general-data comparison','',
        f'Expanded-minus-original aggregate BPB: **{delta:+.6f}** (paired SD{sd:.6f}). Predeclared validation screen passed: **{passed}**. No promotion from validation alone; see the final report for the frozen test gate.','',
        'Three fresh paired seeds211/212/213;8,388,608tokens per run; candidate tokenizer fixed; D12/width768,135,267,480total/active parameters, memory table0, checkpointing off. Same optimizer, schedule, actual domain token mass, technical stream and exact held-out text.','',
        '| Training pool | Seed | Aggregate BPB | General BPB | Technical BPB | Timed tok/s | Update s | Wall s | Alloc/reserved MiB | General stream passes |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        v=r['validation'];t=r['training']
        lines.append(f"| {r['run'].split('-')[1]} | {r['seed']} | {v['aggregate_bpb']:.6f} | {v['domains']['general']['bpb']:.6f} | {v['domains']['technical']['bpb']:.6f} | {t['timed_tokens_per_second']:.0f} | {t['all_update_seconds']:.1f} | {r['wall_seconds']:.1f} | {t['peak_allocated_bytes']/2**20:.1f}/{t['peak_reserved_bytes']/2**20:.1f} | {r['stream']['stream_passes']['general']:.3f} |")
    lines+=['',f'Paired aggregate deltas: {[round(p["aggregate_delta"],6) for p in pairs]}. Domain mean deltas: {domains}.','',
        '## Interpretation','',
        'The original general pool contains609documents/10,890,601normalized bytes; the expanded pool contains1,999documents/36,367,702bytes. They share only33exact normalized documents. **This is pool replacement, not a nested data-size-only ablation:** article/topic coverage and repeated exposure change together. Do not attribute the result solely to having more bytes.',
        '',
        'Both pools use the same source repository revision, lexical/quality filters and source license. The expanded pool contains9,993,911candidate-tokenizer text tokens (before BOS), versus the smaller original pool recorded in its source audit. Technical training remains264documents/5,230,783bytes. Validation/test files are byte-identical across corpora; test scores were not used in this comparison; the final test stage is reported separately.',
        '',
        'Source payload downloaded for this extension:314,076,578bytes; total campaign source payload351,340,048bytes. Expanded fingerprint: `f783433992d354c5887f5130f3f9f7892f171916b4e9ce5294fda2ed87a5b6eb`. Candidate tokenizer SHA256 remains `736bb1a35ce3b2a3143095bfc1390858d67dd7f606fda25137b83095bace0102`. Exact code/runtime/checkpoint identities and immutable replay receipts are in [expanded-results.json](expanded-results.json).','',
        'This result is conditional on the short fixed-token budget and selected held-out target. It neither proves that larger datasets always help nor that diversity is intrinsically harmful. This validation-stage report makes no promotion; see the final combined validation/test gate.','',
        '## Reproduce','',
        '```powershell',"$py = '.autoresearch/upstream/.venv/Scripts/python.exe'",'& $py scripts/foundation_expand.py',
        '& $py scripts/expanded_data_campaign.py',
        '& $py scripts/evaluate_foundation.py runs/foundation_campaign/data-expanded-s211 --data data/foundation-expanded-v1 --output runs/foundation_campaign/manual-expanded-replay.json',
        '& $py scripts/report_expanded_data.py','```','',
        'The controller only admits jobs within the original campaign deadline. For later studies, freeze a new protocol/deadline and new output names rather than modifying earlier receipts.']
    (dest/'EXPANDED-DATA.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print(delta,sd,domains,passed)
if __name__=='__main__':main()
