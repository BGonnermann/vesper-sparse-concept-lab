"""Independent post-run accounting and provenance checks; CPU only, no model forward."""
import math
import random
from pathlib import Path
from collections import Counter
from long_baseline_runtime import read,load_state,source_identity
from foundation_data import ROOT,digest,verify,write

HOME=ROOT/'runs/foundation_long_20260916'

def main():
    run=HOME/'reference';cfg=read(run/'config.json');result=read(run/'result.json');assert result['status']=='completed'
    assert cfg==read(ROOT/'experiments/mainline/foundation-long-v1.json')
    assert digest(run/'config.json')==result['config_sha256'];assert source_identity()==result['source_hashes']
    p=read(ROOT/cfg['profile']);m=verify(ROOT/p['data']['path']);assert m['fingerprint']==p['data']['fingerprint'];assert digest(ROOT/p['tokenizer']['path'])==p['tokenizer']['sha256']
    assert digest(run/'optimizer.json')==digest(ROOT/'runs/foundation_campaign/data-expanded-s211/optimizer.json'),'Optimizer group/initial LR coverage differs'
    pre=read(HOME/'preflight-artifacts.json');lengths=pre['stream']['unique_stream_tokens'];rng=random.Random(cfg['seed']);counts=Counter()
    losses=read(run/'losses.json');assert len(losses)==cfg['updates'];previous_wall=0.
    for step,row in enumerate(losses,1):
        assert row['step']==step and row['tokens']==step*16384 and row['wall_seconds']>=previous_wall
        previous_wall=row['wall_seconds']
        for key in ('loss','gradient_norm','seconds','tokens_per_second','wall_seconds','matrix_lr'):assert math.isfinite(row[key]) and row[key]>=0
        assert row['seconds']>0 and abs(row['tokens_per_second']-16384/row['seconds'])<1e-8
        assert row['matrix_lr']==.04*min(1.,2*(1-(step-1)/cfg['updates']))
        for _ in range(16):counts['general' if rng.random()<p['data']['general_weight'] else 'technical']+=1024
        for domain in lengths:
            progress=row['data_progress'][domain];assert progress['tokens']==counts[domain] and progress['stream_tokens']==lengths[domain]
            assert abs(progress['passes']-counts[domain]/lengths[domain])<1e-12
    state=load_state(result['latest']['path']);assert state['step']==cfg['updates'];assert state['source_hashes']==source_identity()
    assert state['sampler']['rng']==rng.getstate() and state['sampler']['offsets']==dict(counts) and state['sampler']['count']==cfg['updates']*16
    assert state['sampler']['identity']==pre['stream']['stream_sha256']
    assert state['scheduler']==dict(updates=cfg['updates'],policy='autoresearch_train.fixed_schedule',next_step=cfg['updates'])
    del state
    values=read(run/'evaluations.json');assert [v['step'] for v in values]==cfg['evaluation_steps']
    stage=HOME/'final-evaluation';proof=read(stage/'verification.json');assert proof['status']=='verified' and proof['weights_before']==proof['weights_after']
    assert proof['checkpoint_sha256']==result['latest']['sha256']==digest(result['latest']['path'])
    all_values=[x['evaluation'] for x in values]+[read(stage/'validation.json'),read(stage/'test.json')]
    for v in all_values:
        total_nats=0.;total_bytes=0
        for d in v['domains'].values():
            assert d['bytes']>0 and d['tokens']>0 and math.isfinite(d['nats'])
            assert abs(d['bpb']-d['nats']/(math.log(2)*d['bytes']))<1e-12
            assert sum(x['bytes'] for x in d['documents'])==d['bytes'] and sum(x['tokens'] for x in d['documents'])==d['tokens']
            total_nats+=d['nats'];total_bytes+=d['bytes']
        assert abs(v['aggregate_bpb']-total_nats/(math.log(2)*total_bytes))<1e-12
    sample_values=[read(run/f'samples-{step:06d}.json') for step in cfg['sample_steps']]
    prompts=[x['prompt'] for x in sample_values[0]];assert len(prompts)==6
    for samples in sample_values:
        assert [x['prompt'] for x in samples]==prompts
        assert all(x['seed']==20260915 and x['temperature']==.8 and x['top_k']==40 and x['text'].startswith(x['prompt']) for x in samples)
    final_samples=read(stage/'samples.json');same=final_samples==sample_values[-1]
    receipt=dict(status='verified',updates=cfg['updates'],training_tokens=result['tokens'],independently_reconstructed_source_tokens=dict(counts),sampler_rng_and_offsets_verified=True,
        optimizer_group_receipt_identical_to_accepted_pilot=True,evaluation_aggregates_recomputed=len(all_values),fixed_sample_prompts_and_decoder_verified=True,
        final_generation_byte_identical_in_fresh_process=same,profile_sha256=digest(ROOT/cfg['profile']),config_sha256=digest(run/'config.json'),final_checkpoint_sha256=result['latest']['sha256'],source_hashes=source_identity())
    write(HOME/'independent-audit.json',receipt);write(ROOT/'reports/foundation-long-v1/independent-audit.json',receipt);print(receipt,flush=True)

if __name__=='__main__':main()
