"""Audit the sealed tokenizer and a same-vocabulary bounded training-only BPE candidate."""
import argparse
import base64
from collections import defaultdict
import json
import os
from pathlib import Path
import sys
import time
import autoresearch as r
from foundation_data import ROOT, digest, sha, verify, write

PROBES={'code':'def longest_identifier_with_many_parts_123(x):\n\treturn x ** 2  # comment\n',
        'math':'∀ ε > 0: ∑ᵢ xᵢ² ≤ ||x||²; integral_0^1 x^2 dx = 1/3; 1234567890',
        'unicode':'café cafe\u0301 中文 Ελληνικά 😀 👩\u200d💻',
        'whitespace':'\t  indented\n\n\r\n trailing  \n',
        'long_identifier':'some_extremely_long_identifier_'+('abcXYZ123_'*100)}

def current():
    os.environ.update(r.environment());sys.path.insert(0,str(r.RUNTIME))
    import prepare
    r.verify_runtime();r.verify_seal(r.STATE/'cache',json.loads((r.STATE/'data-seal.json').read_text()))
    tok=prepare.Tokenizer.from_directory(dataset='tinystories')
    path=next((r.STATE/'cache').rglob('tokenizer.pkl'))
    return tok.enc,dict(kind='sealed-tinystories',sha256=digest(path),path=str(path)),prepare

def load_candidate(path):
    import tiktoken
    d=json.loads(Path(path).read_text());return tiktoken.Encoding(name='foundation-bpe-v1',pat_str=d['pattern'],
        mergeable_ranks={base64.b64decode(k):v for k,v in d['ranks']},special_tokens=d['special_tokens'])

def load(name):
    enc,receipt,prepare=current()
    if name=='current':return enc,receipt,prepare
    path=Path(name);return load_candidate(path),dict(kind='foundation-bpe-v1',sha256=digest(path),path=str(path)),prepare

def audit(data, output):
    manifest=verify(data);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    original,identity,prepare=current()
    train=[json.loads(x) for x in (Path(data)/'train.jsonl').read_text(encoding='utf-8').splitlines()]
    sample=[];used=defaultdict(int)
    for row in sorted(train,key=lambda x:sha(x['id'].encode())):
        if used[row['domain']]>=4_000_000:continue
        text=row['text'][:4_000_000-used[row['domain']]];used[row['domain']]+=len(text);sample.append(text)
    target=output/'candidate.json'
    if target.exists():raise ValueError('Preserve existing candidate; choose a new audit directory')
    import rustbpe
    started=time.perf_counter();trainer=rustbpe.Tokenizer()
    trainer.train_from_iterator(iter(sample),original.n_vocab-len(prepare.SPECIAL_TOKENS),pattern=prepare.SPLIT_PATTERN)
    ranks={bytes(k):v for k,v in trainer.get_mergeable_ranks()}
    write(target,dict(pattern=trainer.get_pattern(),ranks=[[base64.b64encode(k).decode(),v] for k,v in sorted(ranks.items(),key=lambda x:x[1])],
                      special_tokens={s:len(ranks)+i for i,s in enumerate(prepare.SPECIAL_TOKENS)}))
    candidate_seconds=time.perf_counter()-started
    validation=[json.loads(x) for x in (Path(data)/'validation.jsonl').read_text(encoding='utf-8').splitlines()]
    results=[]
    for name,enc,ident in [('current',original,identity),('candidate',load_candidate(target),dict(sha256=digest(target)))]:
        domains={};counts=defaultdict(lambda:dict(documents=0,tokens=0,bytes=0))
        for row in train+validation:
            ids=enc.encode_ordinary(row['text']);key=row['source']+':'+row['split']
            counts[key]['documents']+=1;counts[key]['tokens']+=len(ids);counts[key]['bytes']+=len(row['text'].encode())
        for domain in sorted({x['domain'] for x in validation}):
            rows=[x for x in validation if x['domain']==domain];t0=time.perf_counter();ids=[enc.encode_ordinary(x['text']) for x in rows];elapsed=time.perf_counter()-t0
            tokens=sum(map(len,ids));nbytes=sum(len(x['text'].encode()) for x in rows);vocab=set(i for seq in ids for i in seq)
            if not all(enc.decode(seq)==row['text'] for seq,row in zip(ids,rows)):raise ValueError('Tokenizer round trip failed')
            if not all(sum(len(enc.decode_single_token_bytes(i)) for i in seq)==len(row['text'].encode()) for seq,row in zip(ids,rows)):raise ValueError('Raw-byte accounting mismatch')
            domains[domain]=dict(documents=len(rows),tokens=tokens,bytes=nbytes,bytes_per_token=nbytes/tokens,
                tokens_per_document=[len(x) for x in ids],vocabulary_used=len(vocab),vocabulary_fraction=len(vocab)/enc.n_vocab,
                unknown_tokens=0,roundtrip=True,encode_seconds=elapsed,encode_tokens_per_second=tokens/elapsed)
        probes={}
        for key,text in PROBES.items():
            ids=enc.encode_ordinary(text);probes[key]=dict(tokens=len(ids),bytes=len(text.encode()),roundtrip=enc.decode(ids)==text,
                raw_token_bytes=sum(len(enc.decode_single_token_bytes(i)) for i in ids),
                legacy_decoded_token_bytes=sum(len(enc.decode([i]).encode()) for i in ids))
        results.append(dict(name=name,identity=ident,vocabulary=enc.n_vocab,embedding_parameters=enc.n_vocab*768,
            embedding_plus_unembedding_parameters=2*enc.n_vocab*768,domains=domains,probes=probes,source_counts=dict(counts)))
    result=dict(dataset_fingerprint=manifest['fingerprint'],script_sha256=digest(__file__),candidate_training_seconds=candidate_seconds,
        candidate_training_characters=dict(used),candidate_training_text_sha256=sha('\n'.join(sample).encode()),
        results=results,recommendation='Keep current mainline tokenizer pending fair fresh-seed model comparison. Candidate compression alone is not model quality.',
        byte_accounting='Use exact UTF-8 document bytes; legacy single-token decoded Unicode lengths can overcount. Do not pool new and legacy BPB.')
    write(output/'audit.json',result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=ROOT/'data/foundation-v1');p.add_argument('--output',type=Path,default=ROOT/'runs/foundation_campaign/tokenizer');a=p.parse_args()
    print(json.dumps(audit(a.data,a.output),indent=2))
