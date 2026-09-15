"""Source-discrepancy, objective and capacity ablations after integration gates."""
import copy
import sys

import autoresearch as r
import ncp_campaign as c
import ncp_search as search
from ncp_interactions import fingerprint


def options():
    anchor=c.candidate('N-RMS')
    variants={}
    for label,key,value,hypothesis in [
        ('R-raw','mixing','raw_logits','Test the pinned official implementation raw-logit codebook reconstruction against paper-inspired softmax; all other N-RMS settings fixed'),
        ('R-no_prediction','prediction_weight',0.,'Remove next-concept MSE while retaining VQ fitting and predicted latent feedback; test whether next-concept supervision contributes beyond the added latent path'),
        ('R-gain4','feedback_scale',4.,'Test stronger normalized feedback, gain4, because anchor predicted RMS is only6.2% of hidden RMS; stronger feedback remains an unproven hypothesis'),
        ('R-gain8','feedback_scale',8.,'Test the upper bounded feedback gain8; compare with gain1 anchor and gain4 to assess amplitude sensitivity')]:
        candidate=copy.deepcopy(anchor); candidate['ncp'][key]=value
        variants[label]=dict(candidate=r.validate_candidate(candidate),hypothesis=hypothesis,control='N-RMS')
    reference=next(p.parent for p in sorted(c.HERE.glob('trial-*/result.json'))
        if (record:=c.read(p))['label']=='N-RMS' and record['seed']==42 and record['status']=='completed')
    model=c.read(reference/'model.json')
    added=model['ncp_parameters']; divisor=2*model['width']
    assert added%divisor==0,'Exact two-matrix capacity control is not representable'
    capacity=dict(depth=anchor['depth'],matrix_lr=anchor['matrix_lr'],feedforward='dense',
        capacity=dict(kind='residual_mlp_v1',hidden=added//divisor,after_layer=anchor['ncp']['after_layer'],lr=anchor['ncp']['lr']))
    variants['CAP-L2-K64']=dict(candidate=r.validate_candidate(capacity),control='D6',
        hypothesis=f'Parameter-matched token-level residual MLP adds exactly {added} parameters at the same insertion and AdamW LR as N-RMS; test added-capacity effects without concept prediction; compute is not matched',
        expected_total_parameters=model['total_parameters'],reference=str(reference))
    return variants


def prepare():
    choices=options()
    for label,option in choices.items(): r.write_json(c.HERE/'candidates'/f'{label}.json',option['candidate'])
    r.write_json(c.HERE/'mechanism-search-space.json',choices)
    # Gain8 checks the largest amplitude; gain4 has identical memory/parameter shapes.
    r.write_json(c.HERE/'fit-candidates.json',{label:choices[label]['candidate']
        for label in ('R-raw','R-no_prediction','R-gain8','CAP-L2-K64')})
    return choices


def main(publish=False):
    if c.remaining()<=7200:
        c.log('Confirmation/report reserve reached; skip mechanism preflight and screening')
        return
    choices=prepare()
    c.preflight()
    fit=c.read(c.HERE/'fit-result.json')['results']
    assert fit['CAP-L2-K64']['model']['total_parameters']==choices['CAP-L2-K64']['expected_total_parameters']
    for label,option in choices.items():
        if c.remaining()<=7200: return
        if any(x['seed']==42 and x.get('candidate') and fingerprint(x['candidate'])==fingerprint(option['candidate'])
                for x in search.records()): continue
        result=c.trial(label,42,option['hypothesis'],'mechanism_ablation',option['control'])
        if result is None: return
        if result['status']=='completed' and 'expected_total_parameters' in option:
            assert result['artifacts']['model']['total_parameters']==option['expected_total_parameters']
        if result['status']!='completed':
            log=(sorted(c.HERE.glob('trial-*/result.json'))[-1].parent/'run.log').read_text(errors='replace')
            if 'Invalid training loss' not in log and 'out of memory' not in log.lower():
                raise RuntimeError('Unclassified mechanism failure requires diagnosis')
        if publish:
            try: search.publish()
            except Exception as exc: c.log('Publication deferred; local mechanism result retained: '+repr(exc))


if __name__=='__main__':
    if '--prepare-only' in sys.argv: prepare()
    else: main('--publish' in sys.argv)
