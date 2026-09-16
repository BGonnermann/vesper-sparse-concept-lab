"""CPU-only publication checks; does not train or open the test set."""
import ast
from datetime import datetime
import json
from pathlib import Path
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
from foundation_data import ROOT,digest,write
from report_long_baseline import HOME,DEST,read

def main():
    result=read(HOME/'reference/result.json');assert result['status']=='completed'
    audit=read(HOME/'independent-audit.json');assert audit['status']=='verified'
    generation=read(HOME/'manual-generation-01.json');assert generation['status']=='verified' and generation['weights_before']==generation['weights_after']
    assert generation['checkpoint_sha256']==result['latest']['sha256']
    assert generation['samples']==read(HOME/'reference/samples-028672.json')
    shutil.copy2(HOME/'manual-generation-01.json',DEST/'inference-verification.json')
    syntax=[]
    for step in (512,14336,28672):
        sample=next(x for x in read(HOME/f'reference/samples-{step:06d}.json') if x['prompt'].startswith('def count_words'))
        try:ast.parse(sample['text']);item=dict(step=step,valid_python_syntax=True)
        except SyntaxError as exc:item=dict(step=step,valid_python_syntax=False,error=exc.msg,line=exc.lineno,column=exc.offset)
        syntax.append(item)
    write(DEST/'sample-syntax-audit.json',dict(executed_generated_code=False,limitations='Length-limited samples; syntax checks are not functional benchmarks',results=syntax))
    chart_hashes={}
    for path in sorted((DEST/'charts').iterdir()):
        assert digest(path)==digest(HOME/'charts-rebuilt'/path.name),f'Chart rerender mismatch: {path.name}'
        chart_hashes[path.name]=digest(path)
        if path.suffix=='.svg':ET.parse(path)
        else:
            from PIL import Image
            with Image.open(path) as image:assert min(image.size)>1000 and abs(image.info['dpi'][0]-300)<.1
    assert len(chart_hashes)==14
    write(DEST/'chart-reproduction.json',dict(status='verified',source='Committed chart-data only, fresh process',byte_identical_files=chart_hashes))
    test_logs={}
    for name,count in (('tests-final-cpu.log',17),('tests-closeout-cuda.log',8)):
        path=HOME/name;text=path.read_text();assert re.search(rf'Ran {count} tests\b',text) and '\nOK' in text
        shutil.copy2(path,DEST/(name+'.txt'));test_logs[name]=dict(tests=count,sha256=digest(path))
    copies=('preflight-artifacts.json','runtime-identity.json','schedule.json','closeout-command-repair.json')
    for name in copies:shutil.copy2(HOME/name,DEST/name)
    shutil.copy2(HOME/'preflight-controller.log',DEST/'initial-resume-failure.log.txt')
    ancillary=[]
    for path in sorted(HOME.glob('*/result.json')):
        if path.parent.name=='reference':continue
        value=read(path);ancillary.append(dict(name=path.parent.name,status=value['status'],tokens=value['tokens'],wall_seconds=value['wall_seconds'],result_path=str(path),result_sha256=digest(path)))
    write(DEST/'ancillary-work.json',dict(full_model_runs=ancillary,total_training_tokens=sum(x['tokens'] for x in ancillary),note='Full-model preflight/calibration only; excludes tiny correctness fixtures and inference'))
    for path in sorted((HOME/'reference/source').glob('*.py')):
        assert digest(path)==audit['source_hashes'][path.name]
    snapshot=HOME/'reporting-source';snapshot.mkdir(exist_ok=True)
    for name in ('evaluate_long_baseline.py','chart_long_baseline.py','report_long_baseline.py','audit_long_baseline.py','supervise_long_resume.py','finish_long_baseline.py','closeout_long_baseline.py'):shutil.copy2(ROOT/'scripts'/name,snapshot/name)
    query="Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(w)?\\.exe$' -and $_.CommandLine -match '(?<![_a-z])(long_baseline(?:_campaign)?|finish_long_baseline|evaluate_long_baseline|supervise_long_resume)\\.py' } | Select-Object ProcessId,CommandLine | ConvertTo-Json -Depth 3"
    proc=subprocess.run(['powershell.exe','-NoProfile','-Command',query],capture_output=True,text=True,check=True)
    processes=json.loads(proc.stdout) if proc.stdout.strip() else [];assert not processes,'Owned campaign work still running'
    now=datetime.now().astimezone();schedule=read(HOME/'schedule.json');start=datetime.fromisoformat(schedule['start']);attempt=read(HOME/'reference/attempt-01.json')
    free=shutil.disk_usage(ROOT).free;assert free>=30*2**30
    closeout=dict(status='verification_complete_pending_publication',as_of=now.isoformat(),campaign_start=schedule['start'],training_finished_at=attempt['finished_at'],training_cutoff=schedule['training_cutoff'],eight_hour_ceiling=schedule['stop'],elapsed_campaign_hours=(now-start).total_seconds()/3600,
        post_training_verification_minutes=(now-datetime.fromisoformat(attempt['finished_at'])).total_seconds()/60,owned_training_or_evaluation_processes=processes,free_disk_bytes=free,focused_tests=test_logs,
        source_snapshot_path=str(HOME/'reference/source'),reporting_snapshot_path=str(snapshot),checkpoint_sha256=result['latest']['sha256'],new_source_download_bytes=0,dependency_upgrades=False,cloud_training=False,
        interpretation='Completed fixed budget; substantial overfitting, not a stronger mainline model',stop_reason='Fixed preregistered budget and final verification complete; no extra training to consume the ceiling')
    assert now<datetime.fromisoformat(schedule['stop']);write(DEST/'closeout.json',closeout);write(HOME/'closeout.json',closeout)
    lines=['# Campaign closeout','',f"As of **{now.isoformat()}**, verification is complete; publication follows. Elapsed campaign time: **{closeout['elapsed_campaign_hours']:.3f} hours**, not eight elapsed hours.",'',
        f"Started {schedule['start']}. Training finished **{attempt['finished_at']}**, before the {schedule['training_cutoff']} cutoff. The eight-hour ceiling was {schedule['stop']}; 90 minutes were reserved after the cutoff. The calibrated fixed budget finished earlier, leaving additional reporting headroom. No budget was changed for quality and no extra training was performed to consume the remaining time.",'',
        '## Verification','',
        '- **17 focused CPU tests and 8 focused CUDA tests passed.** Earlier test logs remain locally. The full 260-test legacy suite was not rerun; shared training/evaluation sources were unchanged.',
        '- Exact restored model/optimizer/RNG/sampler state verified before continued full-model preflight updates. Native BF16 future trajectories are not bitwise deterministic; the preserved initial strict failure is not hidden.',
        '- All 28,672 update records and independent source counts/RNG/offsets checked; optimizer group/initial-LR coverage receipt matches the accepted pilot byte-for-byte.',
        '- All 34 validation/replay/test aggregates recomputed from nats and actual UTF-8 bytes. Final fresh-process validation difference was zero. Test evaluated once on the fixed final endpoint; no test-based checkpoint selection.',
        '- Final generation reproduced byte-identically in fresh processes, including the documented inference command. Weight/checkpoint hashes remained unchanged.',
        '- All seven charts inspected; all 14 PNG/SVG files reproduced byte-identically from exported data in a fresh process. JSON, image dimensions/DPI, SVG parsing and Markdown links checked.',
        f"- No owned training/evaluation processes remain. Free disk at verification: **{free/2**30:.2f} GiB**. No new data downloads, cloud training, dependency upgrades or experimental mechanisms.",'',
        '## Evidence and retained artifacts','',
        '- [Independent accounting audit](independent-audit.json), [chart reproduction](chart-reproduction.json), [inference verification](inference-verification.json), [runtime identity](runtime-identity.json).',
        '- [CPU tests](tests-final-cpu.log.txt), [CUDA tests](tests-closeout-cuda.log.txt), [full-model preflight/calibration accounting](ancillary-work.json).',
        '- [Initial strict resume failure](initial-resume-failure.log.txt) preserved. A closeout orchestration command also initially supplied an unsupported `env` argument; it failed before tests ran. [Repair record](closeout-command-repair.json); no model/evaluator change was needed.',
        '- Large checkpoints, original/pruned checkpoint hash receipts, raw updates, failed attempts, logs, raw corpora and caches stay outside Git. [Checkpoint manifest](checkpoint-manifest.json) gives local paths and hashes.',
        '- Captured training source bytes are under `runs/foundation_long_20260916/reference/source`; final reporting sources under `runs/foundation_long_20260916/reporting-source`. Preserve these archives: Git newline conversion can change raw hashes, including an existing mixed-newline model source. An isolated replication checkout must restore the recorded bytes and runtime/data artifacts; do not weaken provenance checks or overwrite active user files.',
        '- Unrelated untracked monitor/depth-audit files were not edited, staged or deleted. No original pilot/TinyStories/model artifacts were replaced.',
        '- Commits are published by normal pushes on `research/foundation-long-20260916`; see Git history and the publication inventory. No force push.', '',
        '## Continuation boundary','',
        '**This campaign is closed to further training.** The supervised recovery wrapper rejects a completed run or an opened test marker. Do not restart its queue or extend its budget. Future work needs a separate frozen configuration/run directory and fresh held-out evidence. The old regression test is not an untouched benchmark.',
        '', 'Use the [report](REPORT.md) for inference/chart commands and the exact final, early-frontier and accepted-pilot comparison bars. Merely beating this degraded endpoint is not sufficient evidence of progress.']
    (DEST/'CLOSEOUT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    for path in DEST.rglob('*.json'):read(path)
    for path in list(DEST.glob('*.md'))+[ROOT/'docs/foundation-long-v1.md',ROOT/'README.md']:
        for target in re.findall(r'\]\(([^)]+)\)',path.read_text(encoding='utf-8')):
            if target.startswith(('http:','https:','#')):continue
            assert (path.parent/target.split('#')[0]).exists(),f'Broken link {path}: {target}'
    print(json.dumps(closeout,indent=2),flush=True)

if __name__=='__main__':main()
