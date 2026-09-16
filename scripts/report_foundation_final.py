"""Final validation+test gates; no test-driven reselection or training."""
from datetime import datetime
import json
import math
from pathlib import Path
import shutil
import statistics
import subprocess
from foundation_data import ROOT,digest,write
from foundation_baseline import HOME
from finalize_foundation_test import NAMES

DEST=ROOT/'reports/foundation-v1'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))

def gate(rows,reference,candidate,seeds,split):
    def metric(name):return rows[name]['validation'] if split=='validation' else rows[name]['test']['evaluation']
    deltas=[metric(candidate.format(s=s))['aggregate_bpb']-metric(reference.format(s=s))['aggregate_bpb'] for s in seeds]
    domains={d:statistics.mean(metric(candidate.format(s=s))['domains'][d]['bpb']-metric(reference.format(s=s))['domains'][d]['bpb'] for s in seeds) for d in ('general','technical')}
    ratios=[rows[candidate.format(s=s)]['training']['timed_tokens_per_second']/rows[reference.format(s=s)]['training']['timed_tokens_per_second'] for s in seeds]
    mean=statistics.mean(deltas);sd=statistics.stdev(deltas)
    passed=all(x<0 for x in deltas) and mean<=-.01 and -mean>2*sd and max(domains.values())<=.02 and min(ratios)>=.8
    return dict(seeds=list(seeds),deltas=deltas,mean_delta=mean,paired_sd=sd,domain_mean_deltas=domains,throughput_ratios=ratios,passed=passed)

def main():
    freeze=read(HOME/'final-test-freeze.json');assert freeze['status']=='frozen'
    assert freeze['plan_sha256']==digest(ROOT/'docs/foundation-final-test.md')
    assert freeze['evaluator_sha256']==digest(ROOT/'scripts/evaluate_foundation_test.py')
    rows={}
    for name in NAMES:
        r=read(HOME/name/'result.json');test=read(HOME/(name+'-test.json'));assert r['status']=='completed' and test['status']=='verified'
        assert test['freeze_sha256']==digest(HOME/'final-test-freeze.json') and test['split']=='test'
        assert test['run_result_sha256']==digest(HOME/name/'result.json')==freeze['runs'][name]['result_sha256']
        assert test['checkpoint_sha256']==r['checkpoint_sha256']==digest(HOME/name/'checkpoint.pt')
        assert test['weights_before']==test['weights_after'] and test['evaluation']['input_fingerprint']==freeze['input_fingerprint']
        assert r['training_tokens']==8388608 and r['parameters']['total_parameters']==r['parameters']['active_parameters']==135267480
        assert r['training']['peak_allocated_bytes']<16*2**30
        assert math.isfinite(test['evaluation']['aggregate_bpb'])
        r['test']=test;rows[name]=r
    definitions=dict(mixture=('pilot-control-s{s}','pilot-mixture-s{s}',(201,202,203)),
        tokenizer=('pilot-control-s{s}','pilot-tokenizer-s{s}',(201,202,203)),
        expanded_data=('data-old-s{s}','data-expanded-s{s}',(211,212,213)))
    comparisons={name:{split:gate(rows,*args,split) for split in ('validation','test')} for name,args in definitions.items()}
    token_pass=all(x['passed'] for x in comparisons['tokenizer'].values())
    data_pass=token_pass and all(x['passed'] for x in comparisons['expanded_data'].values())
    # Validation failure cannot be rescued by test performance.
    promotion=dict(tokenizer_for_general_technical_profile=token_pass,expanded_pool_for_general_technical_profile=data_pass,mixture_50_50=False,
        original_tinystories_baseline_preserved=True,production_weights_replaced=False)
    summary=dict(status='verified',comparisons=comparisons,promotion=promotion,freeze_sha256=digest(HOME/'final-test-freeze.json'),
        test_results={name:r['test'] for name,r in rows.items()},reporter_sha256=digest(__file__))
    write(DEST/'final-test-results.json',summary);shutil.copy2(HOME/'final-test-freeze.json',DEST/'final-test-freeze.json')
    if token_pass:
        dataset='foundation-expanded-v1' if data_pass else 'foundation-v1';manifest=read(ROOT/'data'/dataset/'manifest.json')
        profile=dict(schema=1,profile_id='foundation-general-technical-v2',scope='Small base-model local research profile; no reasoning/coding capability claim',
            architecture_config='experiments/mainline/dense-v1.json',total_parameters=135267480,active_parameters=135267480,memory_table_bytes=0,
            tokenizer=dict(path='runs/foundation_campaign/tokenizer/candidate.json',sha256=digest(HOME/'tokenizer/candidate.json'),vocabulary=8192),
            data=dict(path='data/'+dataset,fingerprint=manifest['fingerprint'],general_weight=.8,technical_weight=.2),
            training=dict(sequence_length=512,microbatch_size=2,tokens_per_update=16384,optimizer_updates=512,activation_checkpointing=False,matrix_lr=.04,amp='bfloat16'),
            evidence=dict(final_test_freeze_sha256=digest(HOME/'final-test-freeze.json'),
                final_test_receipts_sha256={name:digest(HOME/(name+'-test.json')) for name in NAMES}),
            regression_profile='experiments/mainline/tinystories-regression-v1.json')
        path=ROOT/'experiments/mainline/foundation-general-v2.json'
        if path.exists():assert read(path)==profile,'Do not silently mutate a promoted profile'
        else:write(path,profile)
    test_lines=['# Final held-out test: all15pilot models','',
        'One frozen test stage; same32selected documents, exact UTF-8 byte accounting, context512. No training or reselection followed test opening. All rows have135,267,480total/active parameters,8,388,608training tokens and checkpointing off.','',
        '| Run | Test aggregate BPB | General BPB | Technical BPB | Eval tok/s | Eval allocated MiB |',
        '|---|---:|---:|---:|---:|---:|']
    for name,r in rows.items():
        v=r['test']['evaluation'];test_lines.append(f"| {name} | {v['aggregate_bpb']:.6f} | {v['domains']['general']['bpb']:.6f} | {v['domains']['technical']['bpb']:.6f} | {v['tokens_per_second']:.0f} | {v['peak_allocated_bytes']/2**20:.1f} |")
    test_lines+=['','Exact per-document scores, nats, bytes, token counts, memory, checkpoint/source identities and paired gate calculations: [final-test-results.json](final-test-results.json). Evaluation-only memory excludes optimizer state and differs from training peak.']
    (DEST/'TEST.md').write_text('\n'.join(test_lines)+'\n',encoding='utf-8')
    # Preserve the original report verbatim before replacing the landing report.
    if not (DEST/'PILOT.md').exists():shutil.copy2(DEST/'REPORT.md',DEST/'PILOT.md')
    conclusion=('Use the pilot BPE candidate and expanded general-data pool for the next local general/technical phase.' if data_pass else
                'Use the pilot BPE candidate with the original pilot corpus for the next local general/technical phase.' if token_pass else
                'Keep the original dense/tokenizer baseline; no candidate passed both frozen gates.')
    lines=['# Dense foundation campaign: final report','',f'**{conclusion}** Dense architecture and the original TinyStories regression baseline are preserved. This is a foundation-profile decision, not a useful-assistant or production-weight claim.','',
        '## Verified facts','',
        '- Native Windows execution on the observed RTX5070Ti. Frozen D12/width768,135,267,480total and structurally active parameters, memory table0, BF16 AMP, context512, microbatch2,16,384tokens/update, checkpointing off.',
        '- Fifteen full pilot runs, six fresh seeds (201–203 and211–213),8,388,608tokens/run;125,829,120pilot training tokens. A separate seed101baseline reproduction and two successful two-update smokes are recorded separately.',
        '- Original corpus:1,049documents. Expanded corpus:2,439documents, with unchanged held-out files and unchanged264-document technical training pool. Total retained source-download payload:351,340,048bytes; no cloud, paid services or dependency upgrades.',
        '- V1 dataset fingerprint: `debe7de4f51d9487e316aedf1a7dc828350d7a77f20d99a913a0143e456b121f`; expanded: `f783433992d354c5887f5130f3f9f7892f171916b4e9ce5294fda2ed87a5b6eb`.',
        '- Both tokenizers have8192entries and50,331,648vocabulary-dependent parameters including the existing six value-embedding tables. Original SHA256: `4d1991faca1391dbc13ba13ef7ed19a3ae77dde50d4a6927fd068090b80cda5c`; candidate: `736bb1a35ce3b2a3143095bfc1390858d67dd7f606fda25137b83095bace0102`.',
        '- Every pilot checkpoint reproduced its validation score within1e-6 before one frozen test stage. All15test outcomes and unchanged-weight checks are reported. Exact run/source/runtime identities are in the linked JSON receipts.','',
        '## Experimental results','',
        'Negative deltas favor the candidate. Each comparison changes one factor; do not treat equal tokens as equal raw-byte exposure.','',
        '| Change vs its matched control | Validation BPB delta (paired SD) | Test BPB delta (paired SD) | Both conservative gates |',
        '|---|---:|---:|---|']
    for name,c in comparisons.items():
        a,b=c['validation'],c['test'];lines.append(f"| {name} | {a['mean_delta']:+.6f} ({a['paired_sd']:.6f}) | {b['mean_delta']:+.6f} ({b['paired_sd']:.6f}) | {all(x['passed'] for x in c.values())} |")
    lines+=['','Per-domain paired mean BPB deltas (positive means regression):','',
        '| Change | Validation general | Validation technical | Test general | Test technical |',
        '|---|---:|---:|---:|---:|']
    for name,c in comparisons.items():
        a,b=c['validation']['domain_mean_deltas'],c['test']['domain_mean_deltas']
        lines.append(f"| {name} | {a['general']:+.6f} | {a['technical']:+.6f} | {b['general']:+.6f} | {b['technical']:+.6f} |")
    lines+=['','The50/50mixture remains rejected: its validation general-domain regression failed the predeclared guard. The additional test gate also requires improvement >2paired SD, all3paired aggregate signs favorable, mean gain >=.01BPB, domain mean regressions <=.02BPB, and acceptable costs.','',
        'All per-domain and individual results: [initial9-run pilot](PILOT.md), [fresh-seed data-only comparison](EXPANDED-DATA.md), [all15test models](TEST.md). Aggregate results must not hide the domain tables.','']
    throughput=[r['training']['timed_tokens_per_second'] for r in rows.values()];wall=[r['wall_seconds'] for r in rows.values()]
    lines+=[f"Measured training throughput: {min(throughput):,.0f}–{max(throughput):,.0f}tok/s; full per-run wall time: {min(wall):.1f}–{max(wall):.1f}s. Peak training allocator memory: {max(r['training']['peak_allocated_bytes'] for r in rows.values())/2**20:.1f}MiB allocated / {max(r['training']['peak_reserved_bytes'] for r in rows.values())/2**20:.1f}MiB reserved. Whole-board samples and full cost breakdowns are retained separately.",'',
        'Baseline reproduction: seed101,8,388,608tokens, legacy TinyStories BPB.589531 versus recorded.589157;25,798timed tok/s,334.9s wall,2,277.2/2,408.0MiB allocator peak. Its wrapper metadata failure and separate successful verification are both retained.','',
        '## Inferences','',
        '- The tokenizer improvement is measured model quality, not compression alone, and is conditional on these general/technical corpora and short-context protocol.',
        '- The expanded pool comparison changes article/topic coverage and repeated exposure together: only33general documents overlap exactly. It is **not** a nested data-size-only ablation. Report the measured pool-replacement result, not a universal claim that more data helps.',
        '- Exact-byte evaluation fixes a Unicode accounting problem in the legacy evaluator. Historical packed TinyStories BPB is preserved and must not be pooled with the new document-aligned series.','',
        '## Generated-sample inspection','',
        'Fixed prompts/seed20260915/top-k40/temperature.8/64tokens expose major limitations. The candidate on the original corpus emits constant returns for `count_words` and inconsistent arithmetic. The expanded-pool seed211model emits `x + y` repetitions for word counting and does not explain why the sky is blue. These are base-LM continuations, not an instruction-following test, but they provide no support for useful coding, mathematics, reasoning or factual-reliability claims. All fixed samples are retained; none were used to select a lucky seed.','',
        '## Decisions made','',
        '- Dense remains mainline. Frozen NCP is a negative result under its tested conditions; MoE/n-grams remain experimental and untouched.',
        f"- Tokenizer eligible for the versioned general/technical profile: **{token_pass}**. Expanded pool eligible with that tokenizer: **{data_pass}**. Preserve regression-v1 and every earlier artifact.",
        '- No production checkpoint replacement, weight upload, automatic push or history rewrite. No further training after the final-test freeze.','',
        '## Failures','',
        '- Baseline training succeeded but its wrapper omitted condition metadata; original failure receipt retained, independent verification repaired only the metadata interpretation.',
        '- First new-path smoke hit noncontiguous targets before completing an update. Contiguous copies fixed it; new smoke attempts passed.',
        '- Initial commit failed because Git author identity was unset; command-local campaign assistant identity was used without global configuration changes.','',
        '## Unresolved questions / work not completed','',
        '- No frontier/general-reasoning claim; no broad science, math or standalone code corpus; no production-quality assistant.',
        '- No new depth/width search, checkpointing re-sweep, optimizer/RNG-resumable checkpoints or8GB-device deployment measurement.',
        '- Lexical dedup is not semantic or external-benchmark decontamination. The now-opened test split is a regression set for future work, not an untouched selection set.',
        '- This is a bounded pilot campaign within the8-hour ceiling, not an8-hour-duration training run. Actual clock, test counts, health, disk, file and commit inventory are in [closeout](CLOSEOUT.md).','',
        '## Next three highest-value actions','',
        '1. Use the gated foundation profile in a new predeclared study with longer learning curves and fresh held-out data; do not retune against this now-opened test.',
        '2. Add bounded, explicitly licensed science/math/code sources; independently ablate markup cleanup and data quality rather than architecture mechanisms.',
        '3. Add small task-level evaluations and broaden tokenizer/domain audits before making any useful-capability claim.','',
        '## Reproduce / continue','',
        'See [main protocol](../../docs/foundation-campaign.md), [data extension](../../docs/foundation-data-extension.md), [test freeze](../../docs/foundation-final-test.md), and [tokenizer audit](../../docs/tokenizer-foundation-v1.md). Native Windows commands and artifact locations are in [closeout](CLOSEOUT.md). All weights/corpora/tokenizer artifacts remain outside Git.']
    (DEST/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(comparisons=comparisons,promotion=promotion),indent=2))

if __name__=='__main__':main()
