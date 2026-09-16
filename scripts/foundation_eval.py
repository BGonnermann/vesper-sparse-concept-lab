"""Document-aligned byte evaluation; independent from legacy packed TinyStories BPB."""
import hashlib
import math
import statistics
import time
import torch

PROMPTS=['The purpose of a public library is', 'To explain why the sky appears blue,',
         'def count_words(text):\n    ', 'If x + 3 = 7, then', 'A careful experiment should']

def selected_inputs(records, documents=16, characters=8192):
    splits={r.get('split') for r in records}
    if len(splits)!=1 or not splits <= {'validation','test'}:raise ValueError('Evaluation requires one isolated held-out split')
    result=[]
    for domain in sorted({r['domain'] for r in records}):
        rows=sorted((r for r in records if r['domain']==domain),key=lambda r:hashlib.sha256(r['id'].encode()).hexdigest())[:documents]
        for row in rows:
            text=row['text'][:characters]
            result.append(dict(id=row['id'],domain=domain,text=text,sha256=hashlib.sha256(text.encode()).hexdigest()))
    if not result:raise ValueError('No evaluation documents')
    return result

def token_windows(enc,text,bos,context=512):
    ids=enc.encode_ordinary(text)
    if not ids or enc.decode(ids)!=text:raise ValueError('Empty or non-round-tripping evaluation document')
    if sum(len(enc.decode_single_token_bytes(i)) for i in ids)!=len(text.encode()):raise ValueError('Incorrect token-byte accounting')
    previous=[bos]+ids
    for start in range(0,len(ids),context):
        yield previous[start:start+min(context,len(ids)-start)],ids[start:start+context]

@torch.no_grad()
def evaluate(model,enc,inputs,bos,device='cuda'):
    model.eval();out={};start=time.perf_counter();total_tokens=0
    if device=='cuda':torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    for row in inputs:
        nats=0.;tokens=0
        for x,y in token_windows(enc,row['text'],bos):
            x=torch.tensor([x],dtype=torch.long,device=device);y=torch.tensor([y],dtype=torch.long,device=device)
            with torch.autocast(device_type=device,dtype=torch.bfloat16,enabled=device=='cuda'):
                loss=model(x,y,reduction='none')
            if not torch.isfinite(loss).all():raise ValueError('Non-finite evaluation loss')
            nats+=float(loss.double().sum());tokens+=y.numel()
        domain=out.setdefault(row['domain'],dict(nats=0.,bytes=0,tokens=0,documents=[]))
        nbytes=len(row['text'].encode());domain['nats']+=nats;domain['bytes']+=nbytes;domain['tokens']+=tokens
        domain['documents'].append(dict(id=row['id'],sha256=row['sha256'],bytes=nbytes,tokens=tokens,bpb=nats/(math.log(2)*nbytes)))
        total_tokens+=tokens
    for d in out.values():
        d['bpb']=d['nats']/(math.log(2)*d['bytes']);values=[x['bpb'] for x in d['documents']]
        d['document_bpb_min']=min(values);d['document_bpb_max']=max(values);d['document_bpb_sd']=statistics.pstdev(values)
    if device=='cuda':torch.cuda.synchronize()
    elapsed=time.perf_counter()-start
    return dict(domains=out,aggregate_bpb=sum(x['nats'] for x in out.values())/(math.log(2)*sum(x['bytes'] for x in out.values())),
        macro_domain_bpb=statistics.mean(x['bpb'] for x in out.values()),tokens=total_tokens,seconds=elapsed,tokens_per_second=total_tokens/elapsed,
        peak_allocated_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else None,
        peak_reserved_bytes=torch.cuda.max_memory_reserved() if device=='cuda' else None,
        accounting='All selected UTF-8 bytes exactly once; BOS predicts first token; disjoint target windows of 512 tokens, no cross-document context; no BOS loss',
        input_fingerprint=hashlib.sha256(''.join(x['sha256'] for x in inputs).encode()).hexdigest())

@torch.no_grad()
def generate(model,enc,bos,device='cuda',seed=20260915,new_tokens=64):
    model.eval();generator=torch.Generator(device=device).manual_seed(seed);result=[]
    for prompt in PROMPTS:
        ids=[bos]+enc.encode_ordinary(prompt)
        for _ in range(new_tokens):
            x=torch.tensor([ids[-512:]],device=device)
            with torch.autocast(device_type=device,dtype=torch.bfloat16,enabled=device=='cuda'):logits=model(x)[0,-1].float()
            # Keep special tokens out of human-readable diagnostic samples.
            logits[bos:]=-float('inf');values,indices=torch.topk(logits,40)
            chosen=torch.multinomial(torch.softmax(values/.8,dim=-1),1,generator=generator)
            ids.append(int(indices[chosen]))
        result.append(dict(prompt=prompt,text=enc.decode(ids[1:]),seed=seed,temperature=.8,top_k=40))
    return result
