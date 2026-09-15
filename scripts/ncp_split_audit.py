"""Read-only exact-document overlap audit of sealed train/validation row ranges."""
import ast
import hashlib
from pathlib import Path

import pyarrow.parquet as pq

import autoresearch as r
import ncp_campaign as c


def main():
    prepare=r.RUNTIME/'prepare.py'
    tree=ast.parse(prepare.read_text(encoding='utf-8'))
    assignment=next(node for node in tree.body if isinstance(node,ast.Assign)
        and any(isinstance(target,ast.Name) and target.id=='DATASET_CONFIGS' for target in node.targets))
    splits=ast.literal_eval(assignment.value)['tinystories']['splits']
    assert splits=={'test':(0,10000),'val':(10000,20000),'train':(20000,None)}
    seal=c.read(r.ROOT/'.autoresearch/data-seal.json')
    relative='datasets/tinystories/data/tinystories_gpt4_clean.parquet'
    path=r.ROOT/'.autoresearch/cache'/relative
    assert r.digest(path)==seal[relative]
    validation={'exact':{},'whitespace_normalized':{}}
    hits={'exact':[],'whitespace_normalized':[]}
    counts=dict(validation_rows=0,training_rows=0,excluded_test_rows=0,empty_validation=0,empty_training=0)
    index=0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=4096,columns=['text']):
        for value in batch.column('text').to_pylist():
            row=index;index+=1
            if row<10000:
                counts['excluded_test_rows']+=1
                continue
            assert isinstance(value,str)
            phase='validation' if row<20000 else 'training'
            counts[phase+'_rows']+=1
            if not value.strip(): counts['empty_'+phase]+=1
            variants={'exact':value,'whitespace_normalized':' '.join(value.split())}
            for mode,text in variants.items():
                key=hashlib.sha256(text.encode('utf-8')).hexdigest()
                if phase=='validation':
                    validation[mode].setdefault(key,[]).append(row)
                elif key in validation[mode]:
                    hits[mode].append(dict(training_row=row,validation_rows=validation[mode][key],text_sha256=key))
    assert counts['validation_rows']==10000 and counts['training_rows']>0
    assert r.digest(path)==seal[relative]
    checks={mode:dict(overlapping_training_rows=len(values),
        overlapping_validation_rows=len({row for item in values for row in item['validation_rows']}),
        unique_validation_documents=len(validation[mode]),matches=values) for mode,values in hits.items()}
    r.write_json(c.HERE/'split-audit-result.json',dict(kind='sealed_document_split_audit',status='completed',
        measured_at=c.datetime.now(c.timezone.utc).isoformat(),data_sha256=seal[relative],
        prepare_sha256=r.digest(prepare),script_sha256=r.digest(Path(__file__)),splits=splits,counts=counts,checks=checks,
        interpretation='Scans sealed train/validation row ranges for exact and whitespace-normalized document duplicates. No text is retained or printed; test rows are excluded from comparison. Does not test near-duplicates, shared phrases, dataset-generation leakage or held-out generalization. No splits, data or model weights changed.'))
    print(counts)
    print({mode:{key:value for key,value in item.items() if key!='matches'} for mode,item in checks.items()})


if __name__=='__main__':
    main()
