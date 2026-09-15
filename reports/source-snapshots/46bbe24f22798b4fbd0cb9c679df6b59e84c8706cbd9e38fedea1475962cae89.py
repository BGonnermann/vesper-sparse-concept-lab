"""CPU-only process exclusion check; keeps its tiny lock file as a receipt."""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

import autoresearch as r


def load(path):
    spec=importlib.util.spec_from_file_location('checked_campaign_controller',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--controller',type=Path,required=True)
    parser.add_argument('--lock',type=Path)
    parser.add_argument('--try-only',action='store_true')
    args=parser.parse_args();module=load(args.controller.resolve())
    if args.try_only:
        try:
            with module.gpu_lock(args.lock): return 0
        except RuntimeError: return 7
    stamp=str(time.time_ns());path=module.HERE/f'lock-check-{stamp}.lock'
    for source,prefix in ((args.controller,'lock-controller'),(Path(__file__),'lock-check-source')):
        archived=module.HERE/f'{prefix}-{r.digest(source)}.py'
        if archived.exists(): assert archived.read_bytes()==source.read_bytes()
        else: archived.write_bytes(source.read_bytes())
    command=[str(r.runtime_python()),'-B',str(Path(__file__).resolve()),'--controller',str(args.controller.resolve()),'--lock',str(path),'--try-only']
    with module.gpu_lock(path):
        blocked=subprocess.run(command,timeout=15,capture_output=True,text=True)
        assert blocked.returncode==7,blocked.stderr
    released=subprocess.run(command,timeout=15,capture_output=True,text=True)
    assert released.returncode==0,released.stderr
    try:
        with module.gpu_lock(path): raise ValueError('exercise exception release')
    except ValueError: pass
    after_error=subprocess.run(command,timeout=15,capture_output=True,text=True)
    assert after_error.returncode==0,after_error.stderr
    assert module.CONTROLLER_SOURCE==args.controller.read_bytes()
    fixture=module.HERE/f'source-freeze-fixture-{stamp}.py'
    original=args.controller.read_bytes();fixture.write_bytes(original)
    frozen=load(fixture)
    fixture.write_bytes(original+b'\n# Simulated edit after module import.\n')
    archived=module.HERE/f'source-freeze-archive-{stamp}.py'
    archived_hash=frozen.capture_controller(archived)
    assert archived.read_bytes()==original and archived.read_bytes()!=fixture.read_bytes()
    assert archived_hash==r.digest(args.controller)
    result=dict(kind='native_gpu_lock_check',status='completed',cross_process_exclusion=True,
        normal_release=True,exception_release=True,startup_source_bytes_captured=True,
        source_archive_after_file_edit_verified=True,archived_controller_sha256=archived_hash,
        controller_sha256=r.digest(args.controller),script_sha256=r.digest(Path(__file__)),
        lock_file=str(path),gpu_work=False)
    r.write_json(module.HERE/f'lock-check-{stamp}-result.json',result)
    print(json.dumps(result,indent=2))
    return 0


if __name__=='__main__': sys.exit(main())
