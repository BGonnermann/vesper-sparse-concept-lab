"""Compact provenance, health, testing, clock and continuation inventory. No training."""
from datetime import datetime
import importlib.metadata
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import autoresearch as r
from foundation_data import ROOT,digest,write
from foundation_baseline import HOME,log
from finalize_foundation_test import NAMES

DEST=ROOT/'reports/foundation-v1'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))

def main():
    assert read(HOME/'tape-reconstruction.json')['status']=='verified'
    assert read(DEST/'final-test-results.json')['status']=='verified'
    tests={}
    for device in ('cpu','cuda'):
        path=HOME/f'tests-release-{device}.log';text=path.read_text(errors='replace')
        match=re.search(r'Ran (\d+) tests in ([\d.]+)s',text);assert match and '\nOK\n' in text
        tests[device]=dict(tests=int(match[1]),seconds=float(match[2]),log_sha256=digest(path))
    runtime_dir=HOME/'runtime-inputs';runtime_dir.mkdir(exist_ok=True)
    for name in ('uv.lock','pyproject.toml'):
        target=runtime_dir/name
        if not target.exists():shutil.copy2(r.RUNTIME/name,target)
        assert digest(target)==digest(r.RUNTIME/name)
    environment=dict(upstream=r.verify_runtime(),python=sys.version,
        installed_packages={d.metadata['Name']:d.version for d in sorted(importlib.metadata.distributions(),key=lambda d:d.metadata['Name'].lower())},
        retained_runtime_inputs={name:dict(path=str(runtime_dir/name),sha256=digest(runtime_dir/name)) for name in ('uv.lock','pyproject.toml')},
        note='Existing environment retained; no dependency installation or upgrade during this campaign. Original resolved lock remains available locally.')
    write(DEST/'runtime-identity.json',environment)
    temperatures=[];board=[]
    for name in NAMES:
        path=HOME/(name+'-gpu.json') if name.startswith('pilot-') else HOME/(name+'.log.health.json')
        value=read(path);samples=[x['sample'] for x in value] if isinstance(value,list) else value['samples']
        for sample in samples:
            fields=sample.split(',');temperatures.append(float(fields[0]));board.append(float(fields[1]))
    freeze=read(HOME/'final-test-freeze.json');freeze_time=datetime.fromisoformat(freeze['frozen_at'])
    assert all(datetime.fromtimestamp((HOME/name/'result.json').stat().st_mtime).astimezone()<freeze_time for name in NAMES),'A training receipt changed after final-test opening'
    for name in NAMES:
        assert digest(HOME/name/'result.json')==freeze['runs'][name]['result_sha256']
    schedule=read(HOME/'schedule.json');now=datetime.now().astimezone();start=datetime.fromisoformat(schedule['start'])
    original=read(HOME/'baseline-reproduction/result.json')
    facts=dict(status='verified',schedule=schedule,closeout_at=now.isoformat(),elapsed_seconds=(now-start).total_seconds(),
        training_frozen_at=freeze['frozen_at'],verification_minutes_since_training_freeze=(now-freeze_time).total_seconds()/60,
        tests=tests,max_training_gpu_temperature_c=max(temperatures),max_sampled_whole_board_memory_mib=max(board),
        free_disk_bytes=shutil.disk_usage(ROOT).free,source_download_payload_bytes=351340048,
        source_card_and_api_metadata_additional=True,no_cloud=True,no_dependency_upgrades=True,no_push=True,
        baseline_code_revision=original['git_commit'],baseline_data_seal=original['data_seal'],
        training_code_revisions=sorted({read(HOME/name/'result.json')['git_revision'] for name in NAMES}),
        shared_training_source_sha256=digest(ROOT/'scripts/foundation_train.py'),
        initial_untracked_preserved=['scripts/depth_split_audit.py'],additional_unrelated_untracked_preserved=['docs/project-monitor.md','scripts/monitor_checks.py','scripts/project_monitor.py'])
    write(DEST/'closeout.json',facts)
    for name in ('independent-pairwise-audit.json','expanded-pairwise-audit.json','tape-reconstruction.json','tinystories-source-manifest.json','data-rebuild-result.json'):
        shutil.copy2(HOME/name,DEST/name)
    if (HOME/'expanded-rebuild-result.json').exists():shutil.copy2(HOME/'expanded-rebuild-result.json',DEST/'expanded-rebuild-result.json')
    commits=subprocess.check_output(['git','log','--reverse','--format=%h %s','e1c55b3..HEAD'],text=True).strip()
    (DEST/'IMPLEMENTATION-COMMITS.txt').write_text(commits+'\n',encoding='utf-8')
    files=set(subprocess.check_output(['git','diff','--name-only','e1c55b3'],text=True).splitlines())
    untracked=subprocess.check_output(['git','ls-files','--others','--exclude-standard'],text=True).splitlines()
    excluded=set(facts['initial_untracked_preserved']+facts['additional_unrelated_untracked_preserved'])
    files.update(x for x in untracked if x not in excluded and (x.startswith(('reports/foundation-v1/','experiments/mainline/')) or 'foundation' in x))
    files.update(['reports/foundation-v1/CHANGED-FILES.txt','reports/foundation-v1/CLOSEOUT.md'])
    (DEST/'CHANGED-FILES.txt').write_text('\n'.join(sorted(files-excluded))+'\n',encoding='utf-8')
    lines=['# Verification, provenance and continuation','',
        f"Campaign start: **{schedule['start']}**. Original experiment cutoff: **{schedule['experiment_cutoff']}**; eight-hour ceiling: **{schedule['planned_stop']}**. Closeout audit: **{now.isoformat()}** ({(now-start).total_seconds()/3600:.2f}hours elapsed). All experiment training stopped before the final-test freeze at **{freeze['frozen_at']}**.",
        '',
        'The predeclared experiments and final test stage completed early. Do not describe this as eight hours of training or eight elapsed hours. No extra training was launched merely to consume the remaining budget after opening test results.',
        '',
        f"Final tests: **{tests['cpu']['tests']} CPU and {tests['cuda']['tests']} CUDA tests passed**. All15validation replays passed; all15final-test evaluations verified unchanged weights. Every8,388,608-token training tape was reconstructed on CPU with identical source counts and tensor hash. These audits perform no optimizer updates.",
        '',
        f"Maximum sampled training temperature: **{max(temperatures):.0f}°C**, below the85°C sustained stop threshold. Maximum sampled whole-board memory: **{max(board):.0f}MiB** (includes desktop/driver allocations; sampling is not an exact peak). Free disk at closeout: **{facts['free_disk_bytes']/2**30:.2f}GiB**, above the30GiB floor. No CUDA-device failures or non-finite pilot losses were observed. The first smoke's tensor-layout error was a software validation failure, not a GPU health failure.",
        '',
        '## Data and license checks','',
        '- Original output reconstruction was byte-identical. Expanded output reconstruction, when present, is recorded separately (wall-time metadata is not expected to match). Independent literal-shingle pair audits checked549,676original-corpus pairs and2,973,141expanded-corpus pairs, with zero retained exact or >=.8Jaccard near duplicates and zero shared long paragraphs across retained documents. These are lexical, not semantic/benchmark-wide guarantees.',
        '- New training sources: pinned WikiText raw (CC-BY-SA3.0/GFDL) and CPython documentation (PSF terms, examples additionally0BSD). Source URLs/revisions/byte counts and hashes are in the manifests; licenses/cards remain in local raw directories.',
        '- Existing TinyStories cache provenance was resolved to `karpathy/tinystories-gpt4-clean` revision `0397e27157956705a0260709da3095bb9c43d6a7`: the pinned LFS hash exactly matches the sealed parquet. License: CDLA-Sharing1.0. The original launcher used a main URL; this revision match was established afterward, not invented retroactively. No parquet redownload occurred. Full-corpus token count is unknown and recorded as null.',
        '',
        '## Code and artifact identities','',
        f"Baseline reproduction recorded revision `{facts['baseline_code_revision']}`. Pilot training revisions: `{', '.join(facts['training_code_revisions'])}`; differences include reporting/orchestration commits. Shared executed trainer SHA256: `{facts['shared_training_source_sha256']}`. Each run retains its own complete source hash inventory and copies.",
        '',
        '- [Changed files](CHANGED-FILES.txt) and [implementation commits through this audit](IMPLEMENTATION-COMMITS.txt). The final publication commit is necessarily identified by Git history/final response rather than embedding its own hash into itself.',
        '- All checkpoints, tokenizer files, raw/processed datasets, logs and evaluation text are local under `runs/foundation_campaign/`, `data/foundation-v1/`, and `data/foundation-expanded-v1/`; none are tracked as model/corpus artifacts.',
        '- [Runtime identity](runtime-identity.json) records installed versions, pinned upstream and retained resolved-lock location/hash. Existing dependencies were not upgraded. Exact runtime inputs are retained under `runs/foundation_campaign/runtime-inputs/`.',
        '- Original and newly appearing unrelated user files were left untouched and uncommitted. No push: the working tree began dirty. No history rewrite or artifact deletion.',
        '',
        '## Exact native PowerShell commands','',
        'Existing sealed runtime/cache required; do not run setup over it or replace its dependencies. A fresh-machine reproduction must use the recorded upstream revision, resolved lock and pinned data identities rather than resolve latest dependencies.',
        '',
        '```powershell',"$py = '.autoresearch/upstream/.venv/Scripts/python.exe'",
        '& $py scripts/foundation_data.py verify',
        '& $py scripts/foundation_expand.py',
        '& $py -m unittest discover -s tests -v',
        "$env:MOE_TEST_DEVICE = 'cuda'; & $py -m unittest discover -s tests -v",
        '& $py scripts/evaluate_foundation.py runs/foundation_campaign/data-expanded-s211 --data data/foundation-expanded-v1 --output runs/foundation_campaign/manual-replay-01.json',
        '& $py scripts/audit_foundation.py --data data/foundation-expanded-v1 --output runs/foundation_campaign/manual-pair-audit-01.json',
        'git log --oneline e1c55b3..HEAD','```','',
        'Future study only: the monitored profile launcher refuses to place new training into the closed campaign. It is unit-tested; its underlying trainer completed the15real GPU pilots. No new training was executed to test this wrapper after final-test opening.',
        '',
        '```powershell',
        '& $py scripts/run_foundation_profile.py --profile experiments/mainline/foundation-general-v2.json --seed 301 --output runs/foundation-next/trial-s301 --timeout 1200',
        '```','',
        'Freeze a new study and fresh held-out data before making new improvement claims. This campaign\'s opened test set is now a regression set. For tokenizer reconstruction use `foundation_tokenizer.py` with a new output directory and verify the recorded candidate SHA256. The old frozen queue commands are historical controllers, not a request to restart training after test opening.',
        '',
        'Live campaign log: `runs/foundation_campaign/progress.log`. Retained failure and health logs remain alongside every attempt. Unknown measurements have not been filled with estimates.']
    (DEST/'CLOSEOUT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    log('Closeout inventory written; all core evidence verified. No further training scheduled.')

if __name__=='__main__':main()
