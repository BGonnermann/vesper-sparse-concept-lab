"""CPU-only reconstruction of every consumed pilot tape; never trains a model."""
import json
from pathlib import Path
import time
from foundation_data import ROOT,digest,write,verify
from foundation_tokenizer import current,load_candidate
from foundation_train import make_tape,records
from foundation_baseline import HOME,log
from finalize_foundation_test import NAMES


def main():
    output=HOME/'tape-reconstruction.json'
    if output.exists():raise ValueError('Preserve earlier tape audit')
    start=time.perf_counter();enc,identity,prepare=current();candidate=load_candidate(HOME/'tokenizer/candidate.json')
    encodings={identity['sha256']:enc,digest(HOME/'tokenizer/candidate.json'):candidate}
    pools={}
    for name in ('foundation-v1','foundation-expanded-v1'):
        directory=ROOT/'data'/name;manifest=verify(directory)
        pools[manifest['fingerprint']]=(manifest,records(directory/'train.jsonl'))
    checks=[]
    for name in NAMES:
        r=json.loads((HOME/name/'result.json').read_text());assert r['status']=='completed'
        assert r['source_hashes']['foundation_train.py']==digest(ROOT/'scripts/foundation_train.py')
        manifest,rows=pools[r['dataset_fingerprint']]
        assert json.loads((HOME/name/'data-manifest.json').read_text())==manifest
        tok=encodings[r['tokenizer']['sha256']]
        bos=tok.encode_single_token(prepare.BOS_TOKEN)
        tape,stream=make_tape(rows,tok,bos,r['seed'],r['general_weight'],r['updates'])
        assert stream==r['stream'],name
        assert tape.shape==(8192,2,513) and sum(stream['source_tokens'].values())==8388608
        del tape
        checks.append(dict(run=name,tape_sha256=stream['tape_sha256'],training_tokens=8388608,status='identical'))
        log('CPU training-tape reconstruction identical: '+name)
    write(output,dict(status='verified',runs=checks,wall_seconds=time.perf_counter()-start,script_sha256=digest(__file__),optimizer_updates=0))

if __name__=='__main__':main()
