"""Independent retained-corpus invariants; no training or dataset mutation."""
from collections import Counter,defaultdict
from itertools import combinations
import json
import re
from pathlib import Path
import time
from foundation_data import ROOT,sha,digest,write,verify
from foundation_tokenizer import current,load_candidate


def audit():
    start=time.perf_counter();root=ROOT/'data/foundation-v1';manifest=verify(root)
    assert manifest['fingerprint']==sha(json.dumps(manifest['outputs'],sort_keys=True).encode())
    docs=[];seen=set();paragraph_owner={};counts=Counter()
    for split in ('train','validation','test'):
        for line in (root/(split+'.jsonl')).read_text(encoding='utf-8').splitlines():
            row=json.loads(line);assert row['split']==split and row['id'] not in seen;seen.add(row['id'])
            assert sha(row['text'].encode())==row['content_sha256']
            # Deliberately use literal word tuples, not the pipeline's hashed inverted index.
            words=re.findall(r'\w+',row['text'].casefold());sig={tuple(words[i:i+5]) for i in range(len(words)-4)}
            docs.append((row,sig));counts[row['source']+':'+split+':documents']+=1;counts[row['source']+':'+split+':bytes']+=len(row['text'].encode())
            for paragraph in row['text'].split('\n\n'):
                if len(paragraph.strip())<200:continue
                key=re.sub(r'\s+',' ',paragraph).strip()
                if key in paragraph_owner:assert paragraph_owner[key]==row['id'],'Shared long paragraph in different retained documents'
                paragraph_owner[key]=row['id']
    assert len({x['content_sha256'] for x,sig in docs})==len(docs)
    assert all(manifest['statistics'][key]==value for key,value in counts.items())
    pair_count=0;cross_split_count=0
    for (a,x),(b,y) in combinations(docs,2):
        pair_count+=1;cross_split_count+=a['split']!=b['split']
        # Length-ratio upper bound is exact; skip pairs which cannot attain .8 Jaccard.
        if not x or not y or min(len(x),len(y))/max(len(x),len(y))<.8:continue
        overlap=len(x&y);similarity=overlap/(len(x)+len(y)-overlap)
        assert similarity<.8,(a['id'],b['id'],similarity)
    enc,identity,_=current();candidate=load_candidate(ROOT/'runs/foundation_campaign/tokenizer/candidate.json')
    coverage={}
    for name,tok in [('current',enc),('candidate',candidate)]:
        pieces={tok.decode_single_token_bytes(i) for i in range(tok.n_vocab)}
        count=sum(bytes([b]) in pieces for b in range(256));assert count==256;coverage[name]=count
    result=dict(status='verified',documents=len(docs),all_document_pairs=pair_count,cross_split_pairs=cross_split_count,
        exact_duplicate_pairs=0,near_duplicate_pairs_at_point8=0,shared_long_paragraphs_between_documents=0,
        tokenizer_byte_fallback_coverage=coverage,dataset_fingerprint=manifest['fingerprint'],manifest_sha256=digest(root/'manifest.json'),
        script_sha256=digest(__file__),wall_seconds=time.perf_counter()-start,
        limitations='Lexical thresholds only; does not establish absence of semantic overlap or benchmark contamination')
    write(ROOT/'runs/foundation_campaign/independent-pairwise-audit.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':audit()
