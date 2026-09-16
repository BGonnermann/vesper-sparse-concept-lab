"""One frozen final-test checkpoint evaluation; explicit test-file provenance, no training."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import torch
import autoresearch as r
from autoresearch_model import model_class,with_model_width
from foundation_data import ROOT,digest,write,verify
from foundation_tokenizer import load
from foundation_eval import selected_inputs,evaluate
from ncp_campaign import gpu_lock


def require_frozen(identity,record,checkpoint_hash):
    if record['status']!='completed' or checkpoint_hash!=identity['checkpoint_sha256'] or record['checkpoint_sha256']!=checkpoint_hash:
        raise ValueError('Unfrozen or incomplete checkpoint')
    if record['seed']!=identity['seed'] or record['dataset_fingerprint']!=identity['dataset_fingerprint']:
        raise ValueError('Checkpoint seed/data identity mismatch')

def main(run,output,freeze_path):
    run=Path(run);output=Path(output)
    if output.exists():raise ValueError('Final test output already exists; do not overwrite')
    freeze=json.loads(Path(freeze_path).read_text());assert freeze['status']=='frozen'
    assert digest(__file__)==freeze['evaluator_sha256']
    identity=freeze['runs'][run.name];record=json.loads((run/'result.json').read_text())
    assert digest(run/'result.json')==identity['result_sha256']
    require_frozen(identity,record,digest(run/'checkpoint.pt'))
    for name,h in record['source_hashes'].items():
        assert digest(run/'source'/name)==h and digest(ROOT/'scripts'/name)==h
    data=Path(freeze['test_data_directory']);manifest=verify(data)
    assert manifest['outputs']['test.jsonl']==freeze['test_file_sha256']==digest(data/'test.jsonl')
    rows=[json.loads(line) for line in (data/'test.jsonl').read_text(encoding='utf-8').splitlines()]
    assert all(x['split']=='test' for x in rows)
    inputs=selected_inputs(rows)
    assert hashlib.sha256(''.join(x['sha256'] for x in inputs).encode()).hexdigest()==freeze['input_fingerprint']
    tokenizer=record['tokenizer'];name='current' if tokenizer['kind']=='sealed-tinystories' else tokenizer['path']
    enc,tokenizer_identity,prepare=load(name);assert tokenizer_identity['sha256']==tokenizer['sha256']
    import train
    torch.set_num_threads(2);torch.set_float32_matmul_precision('high');train.MAX_SEQ_LEN=512;train.WINDOW_PATTERN='L'
    runtime=train.detect_runtime();train._configure_step_kernels(runtime)
    config=with_model_width(train.build_model_config(12,enc.n_vocab,runtime,False),record['candidate'])
    with torch.device('meta'):model=model_class(train,record['candidate'])(config)
    model.to_empty(device=runtime.device);model.init_weights(embed_dtype=runtime.amp_dtype)
    state=torch.load(run/'checkpoint.pt',map_location='cpu',weights_only=True)
    assert all(bool(torch.isfinite(t).all()) for t in state.values())
    model.load_state_dict(state,strict=True);del state
    assert model.parameter_report()==record['parameters']
    def weights_hash():
        h=hashlib.sha256()
        for name,p in model.named_parameters():h.update(name.encode());h.update(p.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        return h.hexdigest()
    before=weights_hash();result=evaluate(model,enc,inputs,enc.encode_single_token(prepare.BOS_TOKEN));after=weights_hash();assert before==after
    assert result['input_fingerprint']==freeze['input_fingerprint']
    assert digest(run/'checkpoint.pt')==identity['checkpoint_sha256']
    receipt=dict(status='verified',split='test',run=run.name,freeze_sha256=digest(freeze_path),checkpoint_sha256=identity['checkpoint_sha256'],
        run_result_sha256=identity['result_sha256'],training_dataset_fingerprint=record['dataset_fingerprint'],
        test_dataset_fingerprint=manifest['fingerprint'],test_file_sha256=freeze['test_file_sha256'],tokenizer=tokenizer_identity,
        seed=record['seed'],parameters=record['parameters'],training_code_revision=record['git_revision'],
        evaluation_code_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),evaluator_sha256=digest(__file__),
        weights_before=before,weights_after=after,evaluation=result)
    write(output,receipt);print(json.dumps(receipt,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--freeze',type=Path,required=True);a=p.parse_args()
    with gpu_lock():main(a.run,a.output,a.freeze)
