"""Recompute consumed-data evidence from actual sealed tape bytes on CPU."""
import hashlib
from pathlib import Path

import torch

import autoresearch as r
import ncp_campaign as c


def main():
    protocol = c.read(r.ROOT / 'runs/autoresearch/equal-token-20260915/protocol-42.json')
    path = Path(protocol['batch_tape'])
    assert r.digest(path) == protocol['batch_tape_sha256']
    tape = torch.load(path, map_location='cpu', weights_only=True)
    assert tape.dtype == torch.int64 and tuple(tape.shape) == (8192, 2, 2, 512)
    assert bool(torch.equal(tape[:, 0, :, 1:], tape[:, 1, :, :-1])), 'Labels are not shifted causal token targets'
    assert int(tape.min()) >= 0 and int(tape.max()) < 8192
    hashes = [hashlib.sha256(batch.numpy().tobytes()).digest() for batch in tape]
    expected = {}
    for seed in (42, 43, 44, 45, 46):
        indices = torch.randperm(len(tape), generator=torch.Generator().manual_seed(seed)).tolist()
        chain = hashlib.sha256()
        for index in indices:
            chain.update(hashes[index])
        expected[seed] = dict(indices=indices, chain=chain.hexdigest())
    checked = []
    for file in sorted(c.HERE.glob('trial-*/result.json')):
        record = c.read(file)
        if record['status'] != 'completed':
            continue
        batches = c.read(file.parent / 'batches.json')
        seed = record['protocol'].get('batch_order_seed', record['seed'])
        assert record['protocol']['batch_tape_sha256'] == protocol['batch_tape_sha256']
        assert batches['batch_tape_sha256'] == protocol['batch_tape_sha256']
        assert batches['consumed_indices'] == expected[seed]['indices']
        assert batches['consumed_batch_hash_chain'] == expected[seed]['chain'], file
        assert batches['training_tokens'] == 8388608 and batches['microbatches'] == 8192
        checked.append(dict(trial=file.parent.name, initialization_seed=record['seed'], batch_order_seed=seed,
                            batches_sha256=r.digest(file.parent / 'batches.json'), verified=True))
    result = dict(kind='sealed_tape_byte_audit', status='completed', measured_at=c.datetime.now(c.timezone.utc).isoformat(),
        tape_sha256=r.digest(path), shape=list(tape.shape), next_token_shift_verified=True,
        vocabulary_range_verified=True, order_chains={str(seed):value['chain'] for seed,value in expected.items()},
        verified_trials=checked, script_sha256=r.digest(Path(__file__)),
        interpretation='CPU recomputation from the actual sealed tape, not GPU training or a new quality measurement')
    r.write_json(c.HERE / 'batch-evidence-result.json', result)
    print(f'PASS: actual tape shift/range/bytes and {len(checked)} completed training order chains')


if __name__=='__main__':
    torch.set_num_threads(2)
    main()
