"""Bind confirmation comparisons to frozen configurations and individual trials."""
import statistics

import autoresearch as r
import ncp_campaign as c
import ncp_frozen_contract as contract


def collect(frozen):
    expected={'D6':c.candidate('D6'),**frozen['variants']}
    found={}
    for path in sorted(c.HERE.glob('trial-*/result.json')):
        record=c.read(path); key=(record['label'],record['seed'])
        if key[0] not in expected or key[1] not in frozen['seeds']: continue
        if record['status']!='completed': continue
        contract.validate(path.parent,frozen,key[0],key[1])
        assert key not in found,('Ambiguous repeated confirmation condition',key)
        assert record['candidate']==expected[key[0]],('Frozen configuration mismatch',path)
        assert r.digest(path.parent/'candidate.json')==record['snapshot_files']['candidate.json']
        if key[0]!='D6':
            assert r.digest(path.parent/'candidate.json')==frozen['variant_file_hashes'][key[0]]
        found[key]=(path,record)
    output=[]
    for seed in frozen['seeds']:
        candidate_key=(frozen['label'],seed)
        if candidate_key not in found: continue
        left_path,left=found[candidate_key]
        for control in ('D6','FROZEN-AUX','FROZEN-CAP','FROZEN-NOPRED'):
            if (control,seed) not in found: continue
            right_path,right=found[(control,seed)]
            assert left['protocol']==right['protocol'],('Paired protocol mismatch',seed,control)
            assert left['data_seal']==right['data_seal'],('Paired data identity mismatch',seed,control)
            assert c.read(left_path.parent/'batches.json')==c.read(right_path.parent/'batches.json')
            left_schedule=c.read(left_path.parent/'schedule.json')
            right_schedule=c.read(right_path.parent/'schedule.json')
            for field in ('definition','updates'):
                assert left_schedule[field]==right_schedule[field]
            # The added module has its own final AdamW group; compare shared
            # backbone groups rather than requiring dense to have that group.
            shared=left_schedule['initial_optimizer_groups'][:-1]
            assert shared==right_schedule['initial_optimizer_groups'][:len(shared)]
            if control!='D6':
                assert left_schedule['initial_optimizer_groups']==right_schedule['initial_optimizer_groups']
            left_initial=c.read(left_path.parent/'initialization.json')['parameters']
            right_initial=c.read(right_path.parent/'initialization.json')['parameters']
            backbone=lambda values:{k:v for k,v in values.items() if not k.startswith(('ncp.','capacity.'))}
            assert backbone(left_initial)==backbone(right_initial),('Paired backbone initialization mismatch',seed,control)
            if control in ('FROZEN-AUX','FROZEN-NOPRED'):
                assert left_initial==right_initial,('Paired NCP initialization mismatch',seed,control)
            a=c.read(left_path.parent/'fixed-training.json')['all_update_seconds']
            b=c.read(right_path.parent/'fixed-training.json')['all_update_seconds']
            output.append(dict(seed=seed,role='primary' if seed in frozen['primary_seeds'] else 'sensitivity',
                candidate=frozen['label'],control=control,candidate_trial=left_path.parent.name,
                control_trial=right_path.parent.name,candidate_result_sha256=r.digest(left_path),
                control_result_sha256=r.digest(right_path),candidate_bpb=left['metrics']['val_bpb'],
                control_bpb=right['metrics']['val_bpb'],delta_bpb=left['metrics']['val_bpb']-right['metrics']['val_bpb'],
                update_time_ratio=a/b,noncollapsed=not left['ncp_health']['collapsed'],
                configuration_and_protocol_verified=True))
    summaries={}
    for control in ('D6','FROZEN-AUX','FROZEN-CAP','FROZEN-NOPRED'):
        rows=[x for x in output if x['control']==control]
        primary=[x for x in rows if x['role']=='primary']
        summaries[control]=dict(completed_pairs=len(rows),expected_pairs=len(frozen['seeds']),
            mean_delta_bpb=statistics.mean(x['delta_bpb'] for x in rows) if rows else None,
            negative_signs=sum(x['delta_bpb']<0 for x in rows),
            primary_complete=len(primary)==len(frozen['primary_seeds']),
            primary_same_sign_improvement=(all(x['noncollapsed'] and x['delta_bpb']<0 for x in primary)
                if len(primary)==len(frozen['primary_seeds']) else None))
    return dict(pairs=output,summaries=summaries,
        interpretation='Four predeclared paired seeds;45/46 primary,43/44 sensitivity; same validation split; no held-out claim')


if __name__=='__main__':
    result=collect(c.read(c.HERE/'confirmation-selection.json'))
    r.write_json(c.HERE/'frozen-pairs-result.json',dict(kind='frozen_paired_evidence',status='completed',**result))
