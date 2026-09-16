"""Separate bounded corpus revision: unique general text; same technical pool and held-out bytes."""
import argparse
import heapq
import json
from pathlib import Path
import shutil
import time
import pyarrow.parquet as pq
import foundation_data as base

ROOT=base.ROOT

def read_rows(path):return [json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines()]

def build(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if (output/'manifest.json').exists():
        m=base.verify(output)
        if m['ingestor_sha256']!=base.digest(__file__):raise ValueError('Expanded ingestor changed; choose new output directory')
        return m
    parent=ROOT/'data/foundation-v1';original=base.verify(parent);raw=output/'raw';receipts=[];started=time.perf_counter()
    if shutil.disk_usage(output).free<31*2**30:raise RuntimeError('Disk reserve')
    for shard in range(2):
        name=f'train-{shard:05d}-of-00002.parquet'
        receipts.append(base.download(f'https://huggingface.co/datasets/Salesforce/wikitext/resolve/{base.WIKI_REV}/wikitext-103-raw-v1/{name}',raw/name,limit=200*2**20))
    def texts():
        for shard in range(2):
            for batch in pq.ParquetFile(raw/f'train-{shard:05d}-of-00002.parquet').iter_batches(batch_size=4096,columns=['text']):
                yield from batch.column(0).to_pylist()
    heap=[];count=0
    for row in base.wiki_articles(texts(),'train'):
        count+=1;row['id']=row['id'].replace('wikitext:','wikitext103:',1)
        key=int(base.sha(row['id'].encode()),16)
        # Length check before storing bounded candidates, repeated in the standard cleaner.
        if not 200<=len(row['text'])<=300000:continue
        if len(heap)<2000:heapq.heappush(heap,(-key,row['id'],row))
        elif key < -heap[0][0]:heapq.heapreplace(heap,(-key,row['id'],row))
    chosen=[];size=0
    for _,_,row in sorted(heap,key=lambda x:-x[0]):
        n=len(row['text'].encode())
        if size+n>64*2**20:continue
        chosen.append(row);size+=n
    technical=[x for x in read_rows(parent/'train.jsonl') if x['domain']=='technical']
    heldout=read_rows(parent/'test.jsonl')+read_rows(parent/'validation.jsonl')
    clean,stats=base.clean(heldout+technical+chosen)
    for split in ('validation','test'):
        old=sorted((x for x in heldout if x['split']==split),key=lambda x:x['id'])
        new=sorted((x for x in clean if x['split']==split),key=lambda x:x['id'])
        if old!=new:raise ValueError('Expanded preprocessing changed held-out selection')
        shutil.copy2(parent/(split+'.jsonl'),output/(split+'.jsonl'))
    old=sorted(technical,key=lambda x:x['id']);new=sorted((x for x in clean if x['split']=='train' and x['domain']=='technical'),key=lambda x:x['id'])
    if old!=new:raise ValueError('Technical training pool changed')
    train=sorted((x for x in clean if x['split']=='train'),key=lambda x:x['id'])
    (output/'train.jsonl').write_text(''.join(json.dumps(x,sort_keys=True,ensure_ascii=False)+'\n' for x in train),encoding='utf-8',newline='\n')
    outputs={split+'.jsonl':base.digest(output/(split+'.jsonl')) for split in ('train','validation','test')}
    sources=json.loads(json.dumps(original['sources']));sources[0]['subset']='wikitext-103-raw-v1 train: lowest-hash <=2000 articles, <=64MiB; original v1 wikitext-2-raw-v1 held-out files unchanged'
    sources[0]['risks']+='; additional topics and unique bytes do not establish improved learning'
    m=dict(schema=1,pipeline_sha256=base.digest(base.__file__),ingestor_sha256=base.digest(__file__),policy=base.POLICY,
        parent_manifest_sha256=base.digest(parent/'manifest.json'),parent_fingerprint=original['fingerprint'],outputs=outputs,
        fingerprint=base.sha(json.dumps(outputs,sort_keys=True).encode()),sources=sources,statistics=stats,
        raw_general_articles=count,selected_general_articles=len(chosen),selected_general_bytes=size,
        accepted_documents=len(clean),downloads=receipts,downloaded_bytes=sum(x['bytes'] for x in receipts),
        inherited_download_bytes=original['downloaded_bytes'],inherited_sources=original['downloads'],
        selected_id_fingerprint=base.sha('\n'.join(x['id'] for x in chosen).encode()),
        heldout_unchanged=True,technical_training_unchanged=True,wall_seconds=time.perf_counter()-started,
        limitations=original['limitations']+['Only general-domain training pool changed; tokenizer is not retrained on expanded corpus'])
    base.write(output/'manifest.json',m);return m

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'data/foundation-expanded-v1');a=p.parse_args();print(json.dumps(build(a.output),indent=2))
