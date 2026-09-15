"""Evidence-bounded two-factor NCP interactions; selection seed only."""
import copy
import hashlib
import itertools
import json
import sys

import autoresearch as r
import ncp_campaign as c
import ncp_search as search

ALLOWED=set(search.AXES)|{'mixing'}


def fingerprint(candidate):
    normalized=copy.deepcopy(candidate)
    if 'ncp' in normalized:
        normalized['ncp'].setdefault('mixing','softmax')
        normalized['ncp'].setdefault('pool_normalization','none')
    return hashlib.sha256(json.dumps(normalized,sort_keys=True).encode()).hexdigest()


def proposed(history,anchor):
    baseline=next(x for x in history if x['label']=='N-RMS' and x['seed']==42 and x['status']=='completed')
    base_settings=dict(anchor['ncp'],mixing=anchor['ncp'].get('mixing','softmax'))
    best={}
    for row in history:
        if row['seed']!=42 or row['status']!='completed' or not row.get('candidate',{}).get('ncp'):
            continue
        candidate=row['candidate']; settings=dict(candidate['ncp'],mixing=candidate['ncp'].get('mixing','softmax'))
        if (candidate['depth']!=anchor['depth'] or candidate['matrix_lr']!=anchor['matrix_lr']
                or settings['mode']!='feedback' or settings['prediction_weight']<=0
                or (row.get('ncp_health') or {}).get('collapsed',False)):
            continue
        differences=[key for key in set(settings)|set(base_settings) if settings.get(key)!=base_settings.get(key)]
        if len(differences)!=1 or differences[0] not in ALLOWED:
            continue
        axis=differences[0]
        if row['metrics']['val_bpb']>baseline['metrics']['val_bpb']-.001:
            continue
        if axis not in best or row['metrics']['val_bpb']<best[axis]['metrics']['val_bpb']:
            best[axis]=row
    attempted={fingerprint(x['candidate']) for x in history if x['seed']==42 and x.get('candidate')}
    options=[]
    for left,right in itertools.combinations(sorted(best),2):
        constituents=[best[left],best[right]]
        candidate=copy.deepcopy(anchor)
        for axis,row in zip((left,right),constituents): candidate['ncp'][axis]=row['candidate']['ncp'][axis]
        identity=fingerprint(candidate)
        if identity in attempted: continue
        control=min(constituents,key=lambda x:x['metrics']['val_bpb'])
        effects=sum(x['metrics']['val_bpb']-baseline['metrics']['val_bpb'] for x in constituents)
        hypothesis=(f'Test interaction of {left}={candidate["ncp"][left]} and {right}={candidate["ncp"][right]}; '
            f'individual BPB {constituents[0]["metrics"]["val_bpb"]:.6f} and {constituents[1]["metrics"]["val_bpb"]:.6f} '
            f'versus normalized anchor {baseline["metrics"]["val_bpb"]:.6f}; compare against stronger individual {control["label"]}')
        options.append(dict(label='I2-'+identity[:10],candidate=r.validate_candidate(candidate),
            control=control['label'],hypothesis=hypothesis,axes=[left,right],
            constituents=[x['label'] for x in constituents],ranking_effect_sum=effects))
    return sorted(options,key=lambda x:(x['ranking_effect_sum'],x['label']))


def main(publish=False):
    while c.remaining()>7200:
        options=proposed(search.records(),c.candidate('N-RMS'))
        r.write_json(c.HERE/'interaction-options.json',dict(options=options,
            policy='Two distinct one-factor axes, each at least .001 BPB better than fixed N-RMS anchor; seed42 only; no repeated configuration'))
        if not options:
            c.log('Evidence-qualified two-factor interactions exhausted; next stage requires a new documented hypothesis')
            return
        selected=options[0]
        r.write_json(c.HERE/'candidates'/f'{selected["label"]}.json',selected['candidate'])
        result=c.trial(selected['label'],42,selected['hypothesis'],'interaction',selected['control'])
        if result is None: return
        if result['status']!='completed':
            log=(sorted(c.HERE.glob('trial-*/result.json'))[-1].parent/'run.log').read_text(errors='replace')
            if 'Invalid training loss' not in log and 'out of memory' not in log.lower():
                raise RuntimeError('Unclassified interaction failure needs diagnosis')
        if publish:
            try: search.publish()
            except Exception as exc: c.log('Publication deferred; local interaction retained: '+repr(exc))


def self_test():
    anchor=c.candidate('N-RMS')
    def row(label,bpb,seed=42,**change):
        candidate=copy.deepcopy(anchor);candidate['ncp'].update(change)
        return dict(label=label,seed=seed,status='completed',candidate=candidate,
            metrics={'val_bpb':bpb},ncp_health={'collapsed':False})
    baseline=row('N-RMS',.636)
    first=row('gain',.633,feedback_scale=.1)
    second=row('insert',.634,after_layer=1)
    history=[baseline,first,second,row('unseen',.60,seed=43,entries=16),row('off',.60,prediction_weight=0.)]
    options=proposed(history,anchor)
    assert len(options)==1 and set(options[0]['constituents'])=={'gain','insert'}
    done=dict(label=options[0]['label'],seed=42,status='failed',candidate=options[0]['candidate'],metrics=None)
    assert not proposed(history+[done],anchor)
    assert fingerprint(anchor)==fingerprint(dict(anchor,ncp=dict(anchor['ncp'],mixing='softmax')))
    print('PASS: beneficial-axis evidence, independent-seed exclusion, failed-configuration deduplication, default normalization')


if __name__=='__main__':
    if '--self-test' in sys.argv: self_test()
    else: main('--publish' in sys.argv)
