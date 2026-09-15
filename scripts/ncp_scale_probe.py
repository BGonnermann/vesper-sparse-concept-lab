"""Read-only CPU checkpoint probe of block0 latent scale, using captured modules."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    run=args.run.resolve();record=json.loads((run/'result.json').read_text())
    assert record['status']=='completed' and record['candidate']['feedforward']=='dense'
    manifest=json.loads((run/'snapshot.json').read_text())
    for name,value in manifest.items(): assert digest(run/name)==value
    checkpoint=run/'checkpoint_pre_eval.pt'
    assert digest(checkpoint)==record['checkpoint_sha256']
    sys.path[:0]=[str(run/'source/project'),str(run/'source/upstream')]
    import torch
    import train
    from autoresearch_model import model_class
    from autoresearch_train import batch_order
    torch.set_num_threads(2)
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    protocol=record['protocol'];train.MAX_SEQ_LEN=protocol['sequence_length'];train.WINDOW_PATTERN='L'
    runtime=SimpleNamespace(attention_backend='sdpa',amp_dtype=torch.bfloat16)
    config=train.build_model_config(record['candidate']['depth'],state['transformer.wte.weight'].shape[0],runtime,False)
    with torch.device('meta'): model=model_class(train,record['candidate'])(config)
    model.to_empty(device='cpu');model.init_weights(embed_dtype=torch.bfloat16)
    model.load_state_dict(state,strict=True);model.eval()
    tape_path=Path(protocol['batch_tape']);assert digest(tape_path)==protocol['batch_tape_sha256']
    tape=torch.load(tape_path,map_location='cpu',weights_only=True)
    values=[]
    with torch.no_grad(),torch.autocast('cpu',dtype=torch.bfloat16):
        for index in batch_order(len(tape),record['seed'])[:32]:
            ids=tape[index,0];length=ids.shape[1]
            x=train.norm(model.transformer.wte(ids));x=model.resid_lambdas[0]*x+model.x0_lambdas[0]*x
            ve=model.value_embeds['0'](ids) if '0' in model.value_embeds else None
            hidden=model.transformer.h[0](x,ve,(model.cos[:,:length],model.sin[:,:length]),model.window_sizes[0])
            pooled=hidden.reshape(hidden.shape[0],-1,4,hidden.shape[-1]).mean(2)
            values.append(dict(hidden_rms=float(hidden.float().square().mean().sqrt()),
                raw_chunk4_rms=float(pooled.float().square().mean().sqrt())))
    result=dict(kind='read_only_cpu_checkpoint_scale',status='completed',label=record['label'],seed=record['seed'],
        source_hashes=manifest,checkpoint_sha256=record['checkpoint_sha256'],script_sha256=digest(Path(__file__)),
        data_hashes={'batch_tape':protocol['batch_tape_sha256']},
        mean_hidden_rms=sum(v['hidden_rms'] for v in values)/len(values),
        mean_raw_chunk4_rms=sum(v['raw_chunk4_rms'] for v in values)/len(values),samples=values,
        interpretation='Same first32 seeded training microbatches; CPU BF16 SDPA checkpoint probe, not a GPU quality score or timed training experiment')
    args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('samples','source_hashes')},indent=2))


if __name__=='__main__': main()
