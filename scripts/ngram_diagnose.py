"""Inference-only n-gram checkpoint diagnostics. Never constructs an optimizer."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
STUDY=ROOT/'runs/autoresearch/ngram-20260915'
EARLIER=ROOT/'runs/autoresearch/equal-token-20260915/moe-42'
HERE=ROOT/'runs/autoresearch/ngram-diagnostics-20260915'


def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def arm(label):
    sys.path.insert(0,str(ROOT/'scripts'))
    import autoresearch as r
    import experiment_reports as reports
    choices={'DG':(STUDY/'DG',STUDY/'DG'),'MG':(STUDY/'MG',STUDY/'MG'),
        'M-old':(EARLIER,EARLIER),'M-new':(STUDY/'M',STUDY/'M'),
        'M-new-old-source':(EARLIER,STUDY/'M'),'M-old-new-source':(STUDY/'M',EARLIER)}
    source,checkpoint=choices[label.split('-retry')[0]]
    original=read(source/'result.json');checkpoint_record=read(checkpoint/'result.json')
    assert original['status']==checkpoint_record['status']=='completed'
    assert original['protocol']==checkpoint_record['protocol']
    r.verify_seal(ROOT/'.autoresearch/cache',original['data_seal'])
    out=HERE/label;out.mkdir(exist_ok=False)
    manifest=read(source/'snapshot.json')
    for name,h in manifest.items():
        assert sha(source/name)==h
        target=out/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source/name,target)
    shutil.copy2(Path(__file__),out/'evaluate.py')
    metadata={'label':label,'source_run':str(source),'checkpoint_run':str(checkpoint),
        'checkpoint':str(checkpoint/'checkpoint_pre_eval.pt'),'checkpoint_sha256':sha(checkpoint/'checkpoint_pre_eval.pt'),
        'snapshot_manifest':manifest,'initialization':read(source/'initialization.json'),
        'recorded_bpb':checkpoint_record['metrics']['val_bpb'],'bpb_absolute_tolerance':0.000002,
        'diagnostic_code_sha256':sha(out/'evaluate.py'),'data_seal':original['data_seal']}
    write(out/'diagnostic-config.json',metadata)
    record={'kind':'inference_diagnostic','status':'prepared','candidate':original['candidate'],
        'protocol':original['protocol'],'seed':42,'data_seal':original['data_seal'],
        'diagnostic_configuration':metadata,'metrics':None}
    write(out/'result.json',record)
    process=None;start=None
    try:
        with (out/'run.log').open('x',encoding='utf-8') as log:
            print('LOG READY; evaluation not started: '+str(out/'run.log'),flush=True)
            deadline=time.monotonic()+180
            while not (out/'run.start').exists():
                if time.monotonic()>deadline:raise TimeoutError('Announcement gate expired')
                time.sleep(.1)
            start=time.monotonic()
            process=subprocess.Popen([str(r.runtime_python()),'-I','-B','-u',str(out/'evaluate.py'),'worker'],
                cwd=out,env=r.environment(),stdout=log,stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
            code=process.wait(timeout=180)
        if code:raise RuntimeError(f'Evaluation failed: exit {code}')
        result=read(out/'analysis.json');assert result['reproduction_passed']
        record.update(status='completed',metrics={'val_bpb':result['evaluations']['final_enabled']['bpb']})
    except BaseException as exc:
        if process is not None and process.poll() is None:r.stop_process(process)
        record.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        record['wall_seconds']=time.monotonic()-start if start else 0
        write(out/'result.json',record);reports.emit(out);reports.index()
    print(json.dumps({'label':label,'evaluations':{k:v['bpb'] for k,v in result['evaluations'].items()},
                      'wall_seconds':record['wall_seconds']}),flush=True)


def worker():
    import torch
    out=Path(__file__).resolve().parent
    meta=read(out/'diagnostic-config.json')
    assert sys.flags.isolated and sha(Path(__file__))==meta['diagnostic_code_sha256']
    for name,h in meta['snapshot_manifest'].items():assert sha(out/name)==h
    assert sha(Path(meta['checkpoint']))==meta['checkpoint_sha256']
    sys.path[:0]=[str(out/'source/project'),str(out/'source/upstream')]
    import prepare
    protocol=read(out/'protocol.json');candidate=read(out/'candidate.json')
    prepare.MAX_SEQ_LEN=protocol['sequence_length'];prepare.EVAL_TOKENS=protocol['eval_tokens']
    import train
    # Match the captured adapter's full-attention override before model construction.
    train.WINDOW_PATTERN='L'
    from autoresearch_model import model_class
    runtime=train.detect_runtime();train._configure_step_kernels(runtime)
    torch.set_float32_matmul_precision('high');torch.set_num_threads(2)
    torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    tokenizer=prepare.Tokenizer.from_directory(dataset='tinystories')
    config=train.build_model_config(candidate['depth'],tokenizer.get_vocab_size(),runtime,False)
    with torch.device('meta'):model=model_class(train,candidate)(config)
    model.to_empty(device=runtime.device);model.init_weights(embed_dtype=runtime.amp_dtype);model.eval()
    def parameters():
        return {n:hashlib.sha256(p.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest() for n,p in model.named_parameters()}
    assert parameters()==meta['initialization']['parameters']
    original_loader=prepare.make_dataloader
    tape=[];loader=original_loader(tokenizer,2,512,'val',device='cpu',dataset=tokenizer.dataset)
    for _ in range(64):
        x,y,_=next(loader);tape.append(torch.stack([x,y]).clone())
    tape=torch.stack(tape);validation_hash=hashlib.sha256(tape.numpy().tobytes()).hexdigest()
    def fixed_loader(*args,**kwargs):
        assert args[1:4]==(2,512,'val')
        buffer=torch.empty((2,2,512),dtype=torch.int64,device=runtime.device)
        for batch in tape:
            buffer.copy_(batch);yield buffer[0],buffer[1],1
    prepare.make_dataloader=fixed_loader
    memory=getattr(model,'ngram_memory',None)
    def distribution(values):
        x=torch.cat(values).float()
        return {'count':x.numel(),'mean':x.mean().item(),'rms':x.square().mean().sqrt().item(),
            'min':x.min().item(),'max':x.max().item(),
            'quantiles_0_1_25_50_75_99_100':torch.quantile(x,torch.tensor([0.,.01,.25,.5,.75,.99,1.])).tolist()}
    def evaluate(observe=False,disabled=False):
        before=parameters();hooks=[];layers={str(i):[] for i in range(config.n_layer)};stats={}
        def layer_hook(i):
            def hook(module,inputs,output):
                hidden=output[0] if isinstance(output,tuple) else output
                layers[str(i)].append(hidden.detach().float().square().mean(-1).sqrt().flatten().cpu())
            return hook
        if observe:
            for i,block in enumerate(model.transformer.h):hooks.append(block.register_forward_hook(layer_hook(i)))
        if memory is not None and observe:
            def memory_hook(module,inputs,output):
                hidden,ids=inputs;gate,residual=module.components(hidden,ids)
                valid=module.keys(ids)[1].any(-1)
                h=hidden.float().square().mean(-1).sqrt()
                r=residual.float().square().mean(-1).sqrt()
                effective=(output.float()-hidden.float()).square().mean(-1).sqrt()
                values={'gate':gate.flatten(),'gate_valid_suffix':gate.squeeze(-1)[valid],
                    'hidden_rms':h.flatten(),'residual_rms':r.flatten(),'effective_delta_rms':effective.flatten(),
                    'residual_to_hidden_ratio':(r/h.clamp_min(1e-12)).flatten(),
                    'effective_to_hidden_ratio':(effective/h.clamp_min(1e-12)).flatten(),
                    'valid_suffix':valid.float().flatten()}
                for k,v in values.items():stats.setdefault(k,[]).append(v.detach().float().cpu())
            hooks.append(memory.register_forward_hook(memory_hook))
        if disabled:
            assert memory is not None
            hooks.append(memory.register_forward_hook(lambda module,inputs,output:inputs[0]))
        try:
            with torch.inference_mode(),torch.autocast(runtime.device_type,dtype=runtime.amp_dtype):
                bpb=prepare.evaluate_bpb(model,tokenizer,2,device=runtime.device,dataset=tokenizer.dataset,eval_tokens=65536)
        finally:
            for hook in hooks:hook.remove()
        assert parameters()==before,'Inference modified parameters'
        result={'bpb':bpb,'validation_tokens':65536,'validation_tape_sha256':validation_hash,
                'parameter_immutability_verified':True,'residual_disabled':disabled}
        if observe:
            result['backbone_layer_output_rms']={k:distribution(v) for k,v in layers.items()}
            result['memory_after_block_index']=1 if memory is not None else None
            result['memory']={k:distribution(v) for k,v in stats.items()}
            if stats:
                result['memory']['global_residual_to_hidden_rms_ratio']=result['memory']['residual_rms']['rms']/result['memory']['hidden_rms']['rms']
                result['memory']['global_effective_to_hidden_rms_ratio']=result['memory']['effective_delta_rms']['rms']/result['memory']['hidden_rms']['rms']
        return result
    evaluations={}
    if memory is not None:evaluations['initial_enabled']=evaluate(observe=True)
    state=torch.load(meta['checkpoint'],map_location=runtime.device,weights_only=True)
    model.load_state_dict(state,strict=True);del state
    evaluations['final_enabled']=evaluate(observe=memory is not None)
    delta=abs(evaluations['final_enabled']['bpb']-meta['recorded_bpb'])
    passed=delta<=meta['bpb_absolute_tolerance']
    evaluations['final_enabled_repeat']=evaluate()
    if not passed:
        write(out/'analysis.json',{'reproduction_passed':False,'evaluations':evaluations,'error':'Recorded enabled BPB not reproduced','absolute_error':delta})
        raise RuntimeError(f'Enabled BPB reproduction failed: {delta} > {meta["bpb_absolute_tolerance"]}')
    if memory is not None:evaluations['final_residual_disabled']=evaluate(disabled=True)
    imported={}
    for module in tuple(sys.modules.values()):
        origin=getattr(module,'__file__',None)
        if origin and Path(origin).resolve().is_relative_to(out/'source'):
            p=Path(origin).resolve();imported[p.relative_to(out).as_posix()]=sha(p)
    for name,h in imported.items():assert meta['snapshot_manifest'][name]==h
    for name,h in meta['snapshot_manifest'].items():assert sha(out/name)==h
    assert sha(Path(meta['checkpoint']))==meta['checkpoint_sha256']
    write(out/'analysis.json',{'reproduction_passed':True,'evaluations':evaluations,
        'recorded_bpb':meta['recorded_bpb'],'enabled_absolute_error':delta,'absolute_tolerance':meta['bpb_absolute_tolerance'],
        'executed_source_hashes':imported,'diagnostic_code_sha256':meta['diagnostic_code_sha256'],
        'checkpoint_sha256':meta['checkpoint_sha256'],'torch':torch.__version__,'cuda':torch.version.cuda,
        'gpu':torch.cuda.get_device_name(0),'deterministic_algorithms_enabled':torch.are_deterministic_algorithms_enabled(),
        'matmul_precision':torch.get_float32_matmul_precision(),'cudnn_deterministic':torch.backends.cudnn.deterministic,
        'cudnn_benchmark':torch.backends.cudnn.benchmark,'training_updates':0,
        'ablation_interpretation':'Inference residual removal from a jointly trained model; not a separately trained memory-free control.'})
    print(json.dumps({k:v['bpb'] for k,v in evaluations.items()}),flush=True)


def collisions():
    import numpy as np
    import torch
    protocol=read(STUDY/'protocol.json');path=Path(protocol['batch_tape'])
    assert sha(path)==protocol['batch_tape_sha256']
    tape=torch.load(path,map_location='cpu',weights_only=True)
    boundary=read(STUDY/'boundary-and-batches.json');bos=boundary['bos_token_id'];vocab=boundary['vocab_size']
    x=tape[:,0].numpy().reshape(-1,512);result={}
    for n in (2,3):
        shape=(x.shape[0],513-n);valid=np.ones(shape,dtype=bool);encoded=np.zeros(shape,dtype=np.int64)
        for j in range(n):
            part=x[:,j:j+shape[1]];valid &= part!=bos;encoded=encoded*vocab+part
        unique,f=np.unique(encoded[valid],return_counts=True);h=np.full(unique.shape,n,dtype=np.int64)
        for j in range(n):h=(h*1000003+(unique//vocab**(n-1-j))%vocab+1)%2147483647
        bucket=h%8192;F=np.bincount(bucket,weights=f,minlength=8192);counts=np.bincount(bucket,minlength=8192)
        dominant=np.zeros(8192,dtype=np.int64);np.maximum.at(dominant,bucket,f)
        N=int(f.sum());collision=counts[bucket]>1
        result[str(n)]={'valid_occurrences':N,'distinct_ngrams':int(unique.size),'occupied_buckets':int((counts>0).sum()),
            'buckets_with_multiple_ngrams':int((counts>1).sum()),'extra_distinct_colliders':int(unique.size)-(counts>0).sum().item(),
            'frequency_weighted_any_collision_exposure':float(f[collision].sum()/N),
            'non_dominant_occurrence_fraction':float(1-dominant.sum()/N),
            'frequency_weighted_other_ngram_bucket_mass':float(np.sum(f*(1-f/F[bucket]))/N),
            'global_pair_same_bucket_different_ngram_probability':float((np.square(F).sum()-np.square(f.astype(float)).sum())/N**2),
            'distinct_per_bucket_quantiles':np.quantile(counts,[0,.25,.5,.75,1]).tolist()}
    write(HERE/'collisions.json',{'training_tape_sha256':protocol['batch_tape_sha256'],'orders':result,
        'definitions':{'any_collision':'P(a training occurrence maps to a bucket containing another distinct observed suffix)',
            'non_dominant':'1 - sum_b max_i f_i / total occurrences; counts suffixes other than one most frequent suffix in each bucket',
            'other_ngram_mass':'sum_i f_i*(1-f_i/F_bucket(i))/N; frequency-weighted competing suffix mass conditional on the accessed bucket',
            'global_pair':'Probability two independent training occurrences share a bucket but have distinct suffix identities'},
        'limitation':'Address sharing is measured; harmful gradient interference or insufficient capacity is not established.'})
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    if sys.argv[1]=='worker':worker()
    elif sys.argv[1]=='collisions':collisions()
    else:arm(sys.argv[1])
