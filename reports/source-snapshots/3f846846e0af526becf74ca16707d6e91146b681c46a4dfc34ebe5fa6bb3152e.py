"""Prepare one unchanged packed batch corpus; prove seed/order/init contracts."""
import hashlib
import json
from pathlib import Path
import sys
import torch
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / '.autoresearch/upstream')]
import autoresearch as r
import autoresearch_train as adapter
import autoresearch_model as models
import prepare
import train

def read(p): return json.loads(p.read_text())
def hashes(model):
    return {n:hashlib.sha256(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
            for n,p in model.named_parameters()}

r.verify_runtime()
r.verify_seal(ROOT / '.autoresearch/cache',read(ROOT / '.autoresearch/data-seal.json'))
prepare.MAX_SEQ_LEN = 512
tokenizer = prepare.Tokenizer.from_directory(dataset='tinystories')
loader = prepare.make_dataloader(tokenizer,2,512,'train',device='cpu',dataset=tokenizer.dataset)
tape = torch.empty((8192,2,2,512),dtype=torch.int64)
for i in range(8192):
    x,y,_ = next(loader)
    tape[i,0].copy_(x); tape[i,1].copy_(y)
    if i % 1024 == 0: print('Packed batches',i,flush=True)
torch.save(tape,HERE / 'batches.pt')
tape_hash = r.digest(HERE / 'batches.pt')
hashes_by_batch = [hashlib.sha256(b.numpy().tobytes()).digest() for b in tape]
schedule = adapter.fixed_schedule()
r.write_json(HERE / 'expected-schedule.json',schedule)
initial = {}
for seed in (42,43):
    order = adapter.batch_order(8192,seed)
    h = hashlib.sha256()
    for i in order: h.update(hashes_by_batch[i])
    r.write_json(HERE / f'order-{seed}.json',dict(indices=order,batch_hash_chain=h.hexdigest()))
    protocol = read(ROOT / 'experiments/autoresearch/protocol-no-checkpoint.json')
    protocol.update(protocol_id='vesper-tinystories-equal-token-v1',stopping_rule='optimizer_updates',
        optimizer_updates=512,seed=seed,batch_tape=str(HERE / 'batches.pt'),batch_tape_sha256=tape_hash,
        schedule=dict(clock='optimizer_step',progress='zero_based_step / 512',lr_warmup_updates=0,
            decay_start_step=256,final_lr_fraction=0.0,measurement_warmup_updates=11,muon_momentum_warmup_updates=300))
    r.validate_protocol(protocol)
    r.write_json(HERE / f'protocol-{seed}.json',protocol)
    for variant in ('dense','moe'):
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        candidate = read(ROOT / f'experiments/autoresearch/{variant}-depth6.json')
        runtime = train.detect_runtime()
        config = train.build_model_config(6,tokenizer.get_vocab_size(),runtime,False)
        cls = models.model_class(train,candidate)
        with torch.device('meta'): model = cls(config)
        model.to_empty(device='cuda'); model.init_weights(embed_dtype=torch.bfloat16)
        initial[variant,seed] = hashes(model)
        r.write_json(HERE / f'initialization-{variant}-{seed}.json',initial[variant,seed])
        del model
    for name,h in initial['dense',seed].items():
        mapped = name.replace('.mlp.c_', '.mlp.experts.0.c_')
        assert h == initial['moe',seed][mapped], name
for variant in ('dense','moe'):
    a,b = initial[variant,42],initial[variant,43]
    assert a['transformer.wte.weight'] != b['transformer.wte.weight']
    if variant == 'moe':
        for i in range(6):
            for suffix in ('router.weight','experts.1.c_fc.weight'):
                key = f'transformer.h.{i}.mlp.{suffix}'
                assert a[key] != b[key],key
assert read(HERE/'order-42.json')['indices'] != read(HERE/'order-43.json')['indices']
print('PASS: exact tape, different seeded orders, paired shared initialization, seed-controlled routers/experts',flush=True)
