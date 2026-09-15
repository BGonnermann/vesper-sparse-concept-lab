"""One bounded third-factor stage around a qualifying measured interaction."""
import copy
import sys

import autoresearch as r
import ncp_campaign as c
import ncp_search as search
import ncp_interactions as interaction


def settings(candidate):
    return dict(candidate['ncp'],mixing=candidate['ncp'].get('mixing','softmax'),
        pool_normalization=candidate['ncp'].get('pool_normalization','none'))


def prepare(history):
    anchor=next(x for x in history if x['label']=='N-RMS' and x['seed']==42 and x['status']=='completed')
    base=settings(anchor['candidate']); singles={}; pairs=[]
    healthy=[x for x in history if x['seed']==42 and x['status']=='completed'
        and x.get('candidate',{}).get('ncp',{}).get('mode')=='feedback'
        and not x['ncp_health']['collapsed']
        and (x['candidate']['ncp']['prediction_weight']>0 or x['candidate']['ncp']['ce_weight']>0)]
    for row in healthy:
        value=settings(row['candidate']); changes=[k for k in base if value[k]!=base[k]]
        if len(changes)==1 and changes[0] in interaction.ALLOWED and row['metrics']['val_bpb']<=anchor['metrics']['val_bpb']-.001:
            axis=changes[0]
            if axis not in singles or row['metrics']['val_bpb']<singles[axis]['metrics']['val_bpb']: singles[axis]=row
        if row.get('phase')=='interaction' and len(changes)==2:
            control=next(x for x in healthy if x['label']==row['selection']['control'])
            if row['metrics']['val_bpb']<=control['metrics']['val_bpb']-.001: pairs.append(row)
    if not pairs: return dict(anchor=None,options=[],reason='No interaction improves its stronger constituent by .001 BPB')
    winner=min(pairs,key=lambda x:x['metrics']['val_bpb'])
    changed={k for k in base if settings(winner['candidate'])[k]!=base[k]}
    attempted={interaction.fingerprint(x['candidate']) for x in history if x['seed']==42 and x.get('candidate')}
    options=[]
    for axis,row in sorted(singles.items()):
        if axis in changed: continue
        candidate=copy.deepcopy(winner['candidate']);candidate['ncp'][axis]=row['candidate']['ncp'][axis]
        identity=interaction.fingerprint(candidate)
        if identity in attempted: continue
        options.append(dict(label='I3-'+identity[:10],candidate=r.validate_candidate(candidate),control=winner['label'],
            hypothesis=f'Frozen third-factor stage: add {axis}={candidate["ncp"][axis]} from healthy individual {row["label"]} '
                f'({row["metrics"]["val_bpb"]:.6f} BPB) to qualifying pair {winner["label"]} ({winner["metrics"]["val_bpb"]:.6f}); '
                'test incremental benefit, not assumed additivity; enforce utilization gate',axis=axis))
    return dict(anchor=winner['label'],anchor_bpb=winner['metrics']['val_bpb'],options=options,
        policy='Freeze strongest healthy seed42 pair improving stronger constituent by .001; add one distinct individually beneficial axis; no fourth-factor recursion')


def main(publish=False):
    path=c.HERE/'third-factor-selection.json'
    if path.exists(): plan=c.read(path)
    else:
        history=search.records()
        assert not interaction.proposed(history,c.candidate('N-RMS')),'Declared pair screen is unfinished'
        plan=prepare(history);plan['selected_at']=c.datetime.now(c.timezone.utc).isoformat()
        r.write_json(path,plan)
    for option in plan['options']:
        if c.remaining()<=7200: return
        if any(x['label']==option['label'] and x['seed']==42 for x in search.records()): continue
        r.write_json(c.HERE/'candidates'/f'{option["label"]}.json',option['candidate'])
        result=c.trial(option['label'],42,option['hypothesis'],'third_factor',option['control'])
        if result is None: return
        if result['status']!='completed': raise RuntimeError('Preserved third-factor failure needs diagnosis')
        if publish:
            try: search.publish()
            except Exception as exc: c.log('Publication deferred; third-factor result retained: '+repr(exc))
    c.log('Frozen third-factor space exhausted; no automatic fourth-factor search')


if __name__=='__main__': main('--publish' in sys.argv)
