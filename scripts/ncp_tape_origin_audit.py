"""Reconstruct every tape microbatch from captured train-loader source, without saving a copy."""
import hashlib
import os
from pathlib import Path
import sys
import time

import autoresearch as r
import ncp_campaign as c


def main():
    started=time.monotonic()
    r.verify_runtime()
    seal=c.read(r.ROOT/'.autoresearch/data-seal.json')
    r.verify_seal(r.ROOT/'.autoresearch/cache',seal)
    trial=c.HERE/'trial-0001-D6-s42'
    record=c.read(trial/'result.json')
    source=trial/'source/upstream/prepare.py'
    expected=record['snapshot_files']['source/upstream/prepare.py']
    assert r.digest(source)==expected
    os.environ.update(r.environment())
    sys.path.insert(0,str(source.parent))
    import prepare
    import torch
    assert Path(prepare.__file__).resolve()==source.resolve()
    torch.set_num_threads(2)
    protocol=record['protocol']
    tape_path=Path(protocol['batch_tape'])
    assert r.digest(tape_path)==protocol['batch_tape_sha256']
    tape=torch.load(tape_path,map_location='cpu',weights_only=True)
    assert tuple(tape.shape)==(8192,2,2,512)
    prepare.MAX_SEQ_LEN=512
    tokenizer=prepare.Tokenizer.from_directory(dataset='tinystories')
    loader=prepare.make_dataloader(tokenizer,2,512,'train',device='cpu',dataset='tinystories')
    chain=hashlib.sha256()
    receipt=dict(kind='captured_train_tape_origin_audit',status='running',verified_microbatches=0,
        source_sha256=expected,tape_sha256=protocol['batch_tape_sha256'],data_seal=seal,
        script_sha256=r.digest(Path(__file__)),training_updates=0)
    output=c.HERE/'tape-origin-audit-result.json'
    r.write_json(output,receipt)
    try:
        for index in range(len(tape)):
            assert c.remaining()>1500,'Final publication reserve reached'
            x,y,epoch=next(loader)
            assert torch.equal(x,tape[index,0]) and torch.equal(y,tape[index,1]),('Tape differs from train loader',index)
            for tensor in (x,y): chain.update(tensor.contiguous().numpy().tobytes())
            receipt['verified_microbatches']=index+1
            if (index+1)%1024==0:
                r.write_json(output,receipt)
                c.log(f'Train-tape origin verified {index+1}/8192 microbatches')
        r.verify_seal(r.ROOT/'.autoresearch/cache',seal)
        assert r.digest(source)==expected and r.digest(tape_path)==protocol['batch_tape_sha256']
        receipt.update(status='completed',training_tokens=8388608,epoch=epoch,
            reconstructed_tensor_byte_stream_sha256=chain.hexdigest(),wall_seconds=time.monotonic()-started,
            measured_at=c.datetime.now(c.timezone.utc).isoformat(),
            interpretation='Every saved tape input/target tensor equals a fresh CPU execution of the captured train split loader and sealed tokenizer/data. No tape copy saved, no data/model changes, no optimizer updates. Complements tape hash and consumed-order checks.')
    except BaseException as exc:
        receipt.update(status='failed',error=repr(exc),wall_seconds=time.monotonic()-started)
        raise
    finally:
        r.write_json(output,receipt)
    c.log('PASS: all8192 tape microbatches reconstructed from captured train loader')


if __name__=='__main__':
    main()
