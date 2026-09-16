"""Bounded, pinned pilot ingestion. No external dataset execution or dependency changes."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import shutil
import unicodedata
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
WIKI_REV = 'b08601e04326c79dfdd32d625aee71d232d685c3'
PY_REV = 'de54cf5be371a6f5e2e9f208c38def5f81d3ef02'
POLICY = dict(version=1, normalization='NFC; CRLF/CR to LF; preserve indentation',
              min_chars=200, max_chars=300000, near_jaccard=.8,
              long_shared_paragraph_chars=200, split_priority=['test','validation','train'])

def sha(data): return hashlib.sha256(data).hexdigest()
def digest(path): return sha(Path(path).read_bytes())
def write(path, data):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    tmp.replace(path)

def download(url, path, limit=50*2**20):
    path=Path(path); receipt=path.with_suffix(path.suffix+'.receipt.json')
    if path.exists():
        if not receipt.exists(): raise ValueError(f'Unsealed download: {path}; preserve and choose a new output directory')
        r=json.loads(receipt.read_text())
        if r['url']!=url or r['sha256']!=digest(path): raise ValueError(f'Corrupted download cache: {path}')
        return r
    path.parent.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(path.parent).free < 30*2**30+limit: raise RuntimeError('30 GiB disk reserve would be violated')
    # The fixed inventory is < 1 GiB even at these per-file ceilings; no unbounded URL input.
    tmp=path.with_suffix(path.suffix+'.partial'); size=0
    with urllib.request.urlopen(url,timeout=60) as response, tmp.open('wb') as f:
        while True:
            chunk=response.read(1024*1024)
            if not chunk: break
            size+=len(chunk)
            if size>limit: raise RuntimeError(f'Download exceeded byte ceiling: {url}')
            f.write(chunk)
    tmp.replace(path)
    r=dict(url=url,bytes=size,sha256=digest(path)); write(receipt,r); return r

def normalize(text):
    if not isinstance(text,str): raise ValueError('Document text must be a string')
    return unicodedata.normalize('NFC',text.replace('\r\n','\n').replace('\r','\n')).strip()

def shingles(text):
    words=re.findall(r'\w+',text.casefold())
    return {sha(' '.join(words[i:i+5]).encode())[:16] for i in range(max(0,len(words)-4))}

def paragraphs(text):
    return {sha(re.sub(r'\s+',' ',p).strip().encode()) for p in text.split('\n\n') if len(p.strip())>=200}

def clean(records):
    """Held-out first, globally deduped BEFORE any sampling. Deterministic winner policy."""
    seen_ids=set(); counts=Counter(); normalized=[]
    for row in records:
        if not isinstance(row,dict) or not all(k in row for k in ('id','text','source','domain','split')):
            raise ValueError('Malformed record: id/text/source/domain/split required')
        if row['split'] not in POLICY['split_priority']: raise ValueError('Unknown split')
        if row['id'] in seen_ids: raise ValueError('Duplicate document id')
        seen_ids.add(row['id']); text=normalize(row['text'])
        if not 200<=len(text)<=300000 or '\x00' in text or '\ufffd' in text:
            counts[row['source']+':quality_rejected']+=1;continue
        # Sources are curated English editions; script check is a coarse safeguard, not language identification.
        letters=[c for c in text if c.isalpha()]
        if not letters or sum('LATIN' in unicodedata.name(c,'') for c in letters)/len(letters)<.8:
            counts[row['source']+':script_rejected']+=1;continue
        normalized.append(dict(row,text=text,content_sha256=sha(text.encode())))
    exact=set(); index=defaultdict(set); para_index=set(); signatures=[]; accepted=[]
    for row in sorted(normalized,key=lambda x:(POLICY['split_priority'].index(x['split']),x['id'])):
        key=row['source']; h=row['content_sha256']; sig=shingles(row['text']); paras=paragraphs(row['text'])
        if h in exact: counts[key+':exact_rejected']+=1;continue
        if paras & para_index: counts[key+':shared_paragraph_rejected']+=1;continue
        candidates=set()
        for s in sig: candidates.update(index[s])
        if any(len(sig&signatures[i])/max(1,len(sig|signatures[i]))>=.8 for i in candidates):
            counts[key+':near_rejected']+=1;continue
        i=len(accepted); exact.add(h);para_index.update(paras);signatures.append(sig)
        for s in sig:index[s].add(i)
        accepted.append(row);counts[key+':'+row['split']+':documents']+=1
        counts[key+':'+row['split']+':bytes']+=len(row['text'].encode())
    return accepted,dict(sorted(counts.items()))

def wiki_articles(texts, split):
    current=[];number=0
    for text in texts:
        if re.match(r'^\s*= [^=].*[^=] =\s*$',text) and current:
            yield dict(id=f'wikitext:{split}:{number}',text=''.join(current),source='wikitext',domain='general',split=split)
            current=[];number+=1
        current.append(text)
    if current:yield dict(id=f'wikitext:{split}:{number}',text=''.join(current),source='wikitext',domain='general',split=split)

def verify(directory):
    directory=Path(directory); manifest=json.loads((directory/'manifest.json').read_text())
    if manifest['pipeline_sha256']!=digest(__file__): raise ValueError('Pipeline revision changed; use a new output directory')
    for name,h in manifest['outputs'].items():
        if digest(directory/name)!=h: raise ValueError(f'Corrupted cached output: {name}')
    return manifest

def build(directory):
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
    if (directory/'manifest.json').exists():return verify(directory)
    import pyarrow.parquet as pq
    raw=directory/'raw'; receipts=[]; records=[]
    for split in ('test','validation','train'):
        name=f'{split}-00000-of-00001.parquet'
        receipts.append(download(f'https://huggingface.co/datasets/Salesforce/wikitext/resolve/{WIKI_REV}/wikitext-2-raw-v1/{name}',raw/name))
        records.extend(wiki_articles(pq.read_table(raw/name)['text'].to_pylist(),split))
    receipts.append(download(f'https://huggingface.co/datasets/Salesforce/wikitext/raw/{WIKI_REV}/README.md',raw/'wikitext-README.md'))
    receipts.append(download(f'https://codeload.github.com/python/cpython/zip/{PY_REV}',raw/'cpython.zip'))
    with zipfile.ZipFile(raw/'cpython.zip') as z:
        for name in sorted(z.namelist()):
            relative='/'.join(name.split('/')[1:])
            if not (relative.startswith(('Doc/tutorial/','Doc/library/')) and relative.endswith('.rst')):continue
            text=z.read(name).decode('utf-8')
            # Whole files stay in one split; tutorial and library are technical, not a coding benchmark.
            bucket=int(sha(relative.encode())[:8],16)%10
            split='test' if bucket==0 else 'validation' if bucket==1 else 'train'
            records.append(dict(id='cpython:'+relative,text=text,source='cpython',domain='technical',split=split))
        for relative in ('LICENSE','Doc/license.rst'):
            name=next(n for n in z.namelist() if n.endswith('/'+relative))
            (raw/relative.replace('/','-')).write_bytes(z.read(name))
    accepted,stats=clean(records)
    outputs={}
    for split in ('train','validation','test'):
        path=directory/(split+'.jsonl')
        data=''.join(json.dumps(x,sort_keys=True,ensure_ascii=False)+'\n' for x in accepted if x['split']==split)
        path.write_text(data,encoding='utf-8',newline='\n');outputs[path.name]=digest(path)
    manifest=dict(schema=1,policy=POLICY,pipeline_sha256=digest(__file__),outputs=outputs,
        fingerprint=sha(json.dumps(outputs,sort_keys=True).encode()),statistics=stats,
        raw_documents=len(records),accepted_documents=len(accepted),downloads=receipts,
        downloaded_bytes=sum(r['bytes'] for r in receipts),sources=[
          dict(id='wikitext',revision=WIKI_REV,subset='wikitext-2-raw-v1, whole reconstructed articles, official splits',
               url='https://huggingface.co/datasets/Salesforce/wikitext',license='CC-BY-SA-3.0 / GFDL (dataset card)',
               sampling_weight=.8,risks='Small curated Wikipedia sample; markup, old facts, attribution/share-alike obligations; not broad web coverage'),
          dict(id='cpython',revision=PY_REV,subset='Doc/tutorial/*.rst and Doc/library/*.rst, whole files',
               url='https://github.com/python/cpython',license='PSF-2.0; documentation examples additionally 0BSD; see retained LICENSE and Doc/license.rst',
               sampling_weight=.2,risks='Documentation markup; boilerplate; narrow Python-only technical domain; not a clean code benchmark')],
        limitations=['No science/math source selected yet','Script filter is not a language classifier',
                      'Near dedup uses exact 5-word shingle Jaccard >= .8 and shared long paragraphs, not semantic equivalence',
                      'Evaluation scores must not be used to alter these splits','No external benchmark contamination guarantee'])
    write(directory/'manifest.json',manifest);return manifest

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['build','verify']);p.add_argument('--output',type=Path,default=ROOT/'data/foundation-v1')
    a=p.parse_args();print(json.dumps(build(a.output) if a.command=='build' else verify(a.output),indent=2))
