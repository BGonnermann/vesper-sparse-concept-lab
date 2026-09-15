"""Sequential, bounded inference probes of the four frozen confirmation checkpoints."""
from pathlib import Path
import subprocess
import sys
import time

import autoresearch as r
import ncp_campaign as c
from ncp_frozen_pairs import collect
import ncp_search as search


def main(publish=False):
    frozen = c.read(c.HERE / 'confirmation-selection.json')
    pairs = collect(frozen)
    assert all(x['completed_pairs'] == 4 for x in pairs['summaries'].values())
    script = c.HERE / 'feedback-probe.py'
    source = (r.ROOT / 'scripts/ncp_feedback_probe.py').read_bytes()
    if script.exists():
        assert script.read_bytes() == source
    else:
        script.write_bytes(source)
    driver = c.HERE / 'feedback-interventions-driver.py'
    if driver.exists():
        assert driver.read_bytes() == Path(__file__).read_bytes()
    else:
        driver.write_bytes(Path(__file__).read_bytes())
    plan_path = c.HERE / 'feedback-interventions-plan.json'
    plan = dict(kind='feedback_interventions_plan', seeds=frozen['seeds'],
                conditions=['original', 'feedback_zero', 'codebook_rows_rotated'],
                child_timeout_seconds=180, training_updates=0,
                script_sha256=r.digest(script), driver_sha256=r.digest(driver),
                checkpoint_trials={str(x['seed']): x['candidate_trial'] for x in pairs['pairs'] if x['control']=='D6'},
                hypothesis='Does the trained token predictor rely on predicted feedback and its learned code identity? No retraining or selection.',
                data_seal=c.read(r.ROOT / '.autoresearch/data-seal.json'))
    if plan_path.exists():
        assert c.read(plan_path) == plan
    else:
        r.write_json(plan_path, plan)
    outputs = []
    for seed in plan['seeds']:
        trial = c.HERE / plan['checkpoint_trials'][str(seed)]
        record = c.read(trial / 'result.json')
        assert record['data_seal'] == plan['data_seal'], 'Probe data differs from checkpoint training/evaluation data'
        output = c.HERE / f'feedback-intervention-s{seed}-result.json'
        if output.exists():
            item = c.read(output)
            assert item['status']=='completed' and item['script_sha256']==plan['script_sha256']
            assert item['checkpoint_sha256']==c.read(c.HERE/plan['checkpoint_trials'][str(seed)]/'result.json')['checkpoint_sha256']
            outputs.append(item)
            continue
        assert c.remaining() > 1380, 'Preserve final report reserve'
        log = c.HERE / f'feedback-intervention-s{seed}-{int(time.time())}.log'
        c.log(f'START inference intervention seed{seed}; log={log}')
        with c.gpu_lock():
            r.verify_seal(r.ROOT / '.autoresearch/cache', plan['data_seal'])
            failure = None
            try:
                with log.open('x', encoding='utf-8') as handle:
                    child = subprocess.run([str(r.runtime_python()), '-B', '-I', str(script), '--run',
                        str(trial), '--output', str(output)], cwd=r.ROOT, env=r.environment(),
                        stdout=handle, stderr=subprocess.STDOUT, timeout=180)
                assert child.returncode == 0, f'Child exit {child.returncode}'
            except BaseException as exc:
                failure = repr(exc)
            try:
                r.verify_seal(r.ROOT / '.autoresearch/cache', plan['data_seal'])
            except BaseException as exc:
                failure = str(failure) + '; post-probe data seal: ' + repr(exc)
            if failure:
                partial = c.read(output) if output.exists() else {}
                partial.update(kind='frozen_checkpoint_inference_intervention', status='failed', trial=trial.name,
                    seed=seed, checkpoint_sha256=record['checkpoint_sha256'], script_sha256=plan['script_sha256'],
                    error=failure, log_sha256=r.digest(log) if log.exists() else None)
                r.write_json(output, partial)
                c.log(f'FAILED inference intervention seed{seed}: {failure}; log={log}')
                raise RuntimeError(f'Preserved probe failure: {log}')
        item = c.read(output)
        outputs.append(item)
        c.log(f'END inference intervention seed{seed}: '+str({x['condition']:x['bpb'] for x in item['outcomes']}))
    r.write_json(c.HERE / 'feedback-interventions-result.json', dict(kind='feedback_interventions',
                 status='completed', plan_sha256=r.digest(plan_path), probes=outputs))
    from ncp_report import write
    write()
    if publish:
        search.publish()


if __name__ == '__main__':
    main('--publish' in sys.argv)
