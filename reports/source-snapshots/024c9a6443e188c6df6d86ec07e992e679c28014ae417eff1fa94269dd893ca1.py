"""Bounded preflight and four captured equal-token trials; no live-source training."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
import autoresearch as r

def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def announce(logpath):
    log = logpath.open('x', encoding='utf-8')
    print('LOG READY; child not started: ' + str(logpath), flush=True)
    deadline = time.monotonic() + 180
    while not logpath.with_suffix('.start').exists():
        if time.monotonic() > deadline:
            raise TimeoutError('Announcement expired')
        time.sleep(.1)
    return log

def preflight():
    logpath = HERE / 'preflight.log'
    with announce(logpath) as log:
        for device in ('cpu', 'cuda'):
            env = dict(r.environment(), MOE_TEST_DEVICE=device)
            result = subprocess.run([str(r.runtime_python()), '-B', '-m', 'unittest', 'discover', '-s', 'tests', '-v'],
                cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=150)
            log.write(f'\n{device} correctness exit: {result.returncode}\n'); log.flush()
            if result.returncode:
                raise RuntimeError(device + ' correctness failed')
        result = subprocess.run([str(r.runtime_python()), '-B', '-u', str(HERE / 'prepare_batches.py')],
            cwd=ROOT, env=r.environment(), stdout=log, stderr=subprocess.STDOUT, timeout=240)
        if result.returncode:
            raise RuntimeError('Batch/seed preflight failed')
    r.write_json(HERE / 'preflight.json', dict(passed=True, log_sha256=r.digest(logpath),
        source_hashes={n:r.digest(ROOT / 'scripts' / n) for n in r.PROJECT_FILES}))
    print('CPU/CUDA and batch/seed preflight PASSED', flush=True)

def trial(variant, seed):
    assert (variant, seed) in [('dense',42),('moe',42),('moe',43),('dense',43)]
    gate = read(HERE / 'preflight.json')
    assert gate['passed'] and gate['log_sha256'] == r.digest(HERE / 'preflight.log')
    assert gate['source_hashes'] == {n:r.digest(ROOT / 'scripts' / n) for n in r.PROJECT_FILES}
    setup = r.verify_runtime()
    seal = read(ROOT / '.autoresearch/data-seal.json')
    r.verify_seal(ROOT / '.autoresearch/cache', seal)
    candidate = r.validate_candidate(read(ROOT / f'experiments/autoresearch/{variant}-depth6.json'))
    protocol = r.validate_protocol(read(HERE / f'protocol-{seed}.json'))
    out = HERE / f'{variant}-{seed}'
    out.mkdir(exist_ok=False)
    record = dict(kind='fixed_updates', condition=variant, record_version=4, candidate=candidate,
        protocol=protocol, upstream=setup, data_seal=seal, seed=seed, git_commit=None, git_dirty=None,
        preflight_sha256=r.digest(HERE / 'preflight.json'), orchestrator_sha256=r.digest(Path(__file__)),
        git_metadata_note='Not queried; actual executed source hashes recorded.')
    record.update({key:r.digest(ROOT / 'scripts' / name) for name,key in r.PROJECT_FILES.items()})
    snapshot = r.capture_run_snapshot(out, record)
    record.update(snapshot_files=snapshot, status='prepared', command=r.snapshot_command(r.runtime_python(),out))
    r.write_json(out / 'result.json',record)
    start = None
    process = None
    try:
        with announce(out / 'run.log') as log:
            start = time.monotonic()
            record['status'] = 'running'
            r.write_json(out / 'result.json',record)
            process = subprocess.Popen(record['command'],cwd=out,env=r.environment(),stdout=log,
                stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
            record['returncode'] = process.wait(timeout=900)
        if record['returncode']:
            raise RuntimeError(f'Training exit {record["returncode"]}')
        record['metrics'] = r.parse_summary((out / 'run.log').read_text(encoding='utf-8'))
        record['artifacts'] = r.validate_run_artifacts(out,record)
        record['execution'] = r.validate_execution(out,snapshot)
        training = record['artifacts']['training']
        batches = read(out / 'batches.json')
        expected = read(HERE / f'order-{seed}.json')
        assert record['metrics']['num_steps'] == training['optimizer_updates'] == 512
        assert training['training_tokens'] == batches['training_tokens'] == 8388608
        assert batches['consumed_indices'] == expected['indices']
        assert batches['consumed_batch_hash_chain'] == expected['batch_hash_chain']
        assert training['timed_training_tokens'] == 8208384
        assert read(out / 'schedule.json')['updates'] == read(HERE / 'expected-schedule.json')
        assert read(out / 'initialization.json')['parameters'] == read(HERE / f'initialization-{variant}-{seed}.json')
        record['memory'] = read(out / 'memory.json')
        record['status'] = 'completed'
    except BaseException as exc:
        if process is not None and process.poll() is None:
            r.stop_process(process)
        record.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        record['wall_seconds'] = time.monotonic()-start if start else 0
        r.write_json(out / 'result.json',record)
    print(json.dumps({k:record[k] for k in ('status','metrics','wall_seconds','memory')}),flush=True)

if __name__ == '__main__':
    if sys.argv[1] == 'preflight': preflight()
    else: trial(sys.argv[1],int(sys.argv[2]))
