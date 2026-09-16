"""Replay a completed pilot checkpoint on its exact frozen validation inputs.

Does not train, score the test split, modify prior receipts, or select a checkpoint.
"""
import argparse
import hashlib
import json
import itertools
from pathlib import Path
import subprocess
import sys
import time
import torch

import autoresearch as r
from foundation_data import ROOT,digest,sha,verify,write
from foundation_tokenizer import load
from foundation_eval import evaluate


def replay(run,output,data):
    run=Path(run);output=Path(output)
    if output.exists():raise ValueError('Preserve earlier evaluation receipts; choose a new output')
    result=json.loads((run/'result.json').read_text());assert result['status']=='completed'
    assert digest(run/'checkpoint.pt')==result['checkpoint_sha256']
    manifest=verify(data);assert manifest['fingerprint']==result['dataset_fingerprint']
    for name,h in result['source_hashes'].items():
        assert digest(run/'source'/name)==h,'Captured source corruption'
        assert digest(ROOT/'scripts'/name)==h,'Live source differs; use the recorded revision to replay'
    tokenizer=result['tokenizer'];name='current' if tokenizer['kind']=='sealed-tinystories' else tokenizer['path']
    enc,identity,prepare=load(name);assert identity['sha256']==tokenizer['sha256']
    import train
    from autoresearch_model import model_class,with_model_width
    train.MAX_SEQ_LEN=512;train.WINDOW_PATTERN='L';torch.set_num_threads(2);torch.set_float32_matmul_precision('high')
    runtime=train.detect_runtime();train._configure_step_kernels(runtime)
    config=with_model_width(train.build_model_config(12,enc.n_vocab,runtime,result['checkpointing']),result['candidate'])
    with torch.device('meta'):model=model_class(train,result['candidate'])(config)
    model.to_empty(device=runtime.device);model.init_weights(embed_dtype=runtime.amp_dtype)
    state=torch.load(run/'checkpoint.pt',map_location='cpu',weights_only=True)
    assert all(bool(torch.isfinite(t).all()) for t in state.values()),'Non-finite checkpoint tensor'
    model.load_state_dict(state,strict=True);del state
    assert model.parameter_report()==result['parameters']
    def weights_hash():
        h=hashlib.sha256()
        for name,p in model.named_parameters():
            h.update(name.encode());h.update(p.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        return h.hexdigest()
    inputs=json.loads((run/'evaluation-inputs.json').read_text(encoding='utf-8'))
    source={x['id']:x for x in (json.loads(line) for line in (Path(data)/'validation.jsonl').read_text(encoding='utf-8').splitlines())}
    for row in inputs:
        assert row['text']==source[row['id']]['text'][:8192]
        assert sha(row['text'].encode())==row['sha256']
        assert row['domain']==source[row['id']]['domain']
    before=weights_hash();evaluation=evaluate(model,enc,inputs,enc.encode_single_token(prepare.BOS_TOKEN));after=weights_hash()
    assert before==after,'Evaluation modified weights'
    assert evaluation['input_fingerprint']==result['validation']['input_fingerprint']
    delta=abs(evaluation['aggregate_bpb']-result['validation']['aggregate_bpb'])
    assert delta<1e-6,('Validation replay mismatch',delta)
    stories=[]
    for i,text in enumerate(itertools.islice(prepare._iter_tinystories_texts('val','tinystories'),16)):
        text=text[:8192]
        stories.append(dict(id=f'tinystories:val:first16:{i}',domain='tinystories',text=text,sha256=sha(text.encode())))
    regression=evaluate(model,enc,stories,enc.encode_single_token(prepare.BOS_TOKEN))
    assert weights_hash()==before
    receipt=dict(status='verified',tinystories_document_bpb=regression,
        tinystories_data_seal=json.loads((r.STATE/'data-seal.json').read_text()),
        tinystories_split_rows=prepare.DATASET_CONFIGS['tinystories']['splits']['val'],
        run=str(run),run_result_sha256=digest(run/'result.json'),checkpoint_sha256=result['checkpoint_sha256'],
        dataset_fingerprint=manifest['fingerprint'],tokenizer=identity,seed=result['seed'],parameters=result['parameters'],
        training_code_revision=result['git_revision'],evaluation_code_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        evaluator_sha256=digest(__file__),weights_before=before,weights_after=after,evaluation=evaluation,absolute_bpb_difference=delta)
    write(output,receipt);print(json.dumps(receipt,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--data',type=Path,default=ROOT/'data/foundation-v1');a=p.parse_args()
    from ncp_campaign import gpu_lock
    with gpu_lock():replay(a.run,a.output,a.data)
