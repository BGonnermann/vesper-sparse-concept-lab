"""Deterministic, offline research receipts and explicitly requested Git publication."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 1
TERMINAL = {'completed', 'failed', 'timeout', 'invalid', 'interrupted'}
DETAILS = {'candidate', 'protocol', 'metrics', 'seed', 'seeds', 'condition', 'kind', 'status',
           'wall_seconds', 'error', 'returncode', 'interpretation', 'limitations', 'limits',
           'paired_differences', 'mean_paired_bpb_difference', 'rows', 'training', 'profile',
           'probes', 'runs', 'settings', 'config', 'precision', 'gpu', 'torch',
           'correctness_passed', 'performance_gate_passed', 'aggregate_throughput_gain',
           'gate_passed', 'off_throughput_gain', 'comfortable_memory', 'pairs', 'variants',
           'cpu_tests_passed', 'cuda_tests_passed', 'matched_fields', 'matched_comparison_fields',
           'data_hashes', 'source_hashes', 'executed_original_sources', 'checkpoint_hashes',
           'batch_sha256', 'bank_sha256', 'protocol_id', 'activation_checkpointing', 'schedule'}


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n'


def save(path, text):
    """Replace atomically only when content changed; retries create no duplicates."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding='utf-8') == text:
        return
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(text, encoding='utf-8')
    temporary.replace(path)


def clean(text, root):
    text = text.replace(str(root), '<repo>').replace(root.as_posix(), '<repo>')
    text = re.sub(r'(?i)(authorization\s*[:=]\s*|(?:api[_-]?key|access[_-]?token|password|secret)\s*[:=]\s*)[^\s,;]+', r'\1[REDACTED]', text)
    text = re.sub(r'\b(?:gh[pousr]_[A-Za-z0-9_]{15,}|github_pat_[A-Za-z0-9_]+|sk-[A-Za-z0-9_-]{16,})\b', '[REDACTED]', text)
    return text


def compact(value, root):
    if isinstance(value, dict):
        return {k: compact(v, root) for k,v in value.items()
                if k not in {'consumed_indices', 'indices', 'command', 'parameters', 'snapshot_files', 'execution', 'data_seal'}}
    if isinstance(value, list):
        if len(value) > 64:
            return {'omitted_items':len(value), 'note':'Full values remain in the hashed local artifact.'}
        return [compact(v, root) for v in value]
    if isinstance(value, str): return clean(value, root)
    return value


def load(path, diagnostics):
    try:
        return json.loads(path.read_text(encoding='utf-8-sig'), parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)))
    except (OSError, ValueError) as exc:
        diagnostics.append(f'{path.name}: unreadable JSON ({type(exc).__name__})')
        return None


def identity(source, root):
    relative = source.resolve().relative_to((root/'runs/autoresearch').resolve()).as_posix()
    safe = re.sub(r'[^A-Za-z0-9_.-]+', '--', relative)
    return safe + '-' + hashlib.sha256(relative.encode()).hexdigest()[:8], relative


def emit(source, root=ROOT):
    source, root = Path(source).resolve(), Path(root).resolve()
    experiment_id, relative = identity(source, root)
    diagnostics, evidence, details, excerpts, archived = [], {}, {}, {}, {}
    files = sorted(p for p in source.iterdir() if p.is_file() and p.suffix in {'.json','.md','.log','.py'})
    for path in files:
        evidence[path.name] = {'sha256':sha(path), 'bytes':path.stat().st_size,
                               'local_path':'runs/autoresearch/'+relative+'/'+path.name}
        if path.suffix == '.json' and path.stat().st_size <= 131072:
            if path.name.startswith(('prior-', 'initialization-', 'order-')): continue
            value = load(path, diagnostics)
            if isinstance(value, dict):
                if path.name in {'model.json','memory.json','training.json','routing.json','environment.json','candidate.json','protocol.json','fixed-training.json','comparison.json','analysis.json','dense.json','moe.json','ngram-diagnostics.json','ngram-tape-diagnostics.json','plan-revision.json','boundary-and-batches.json'} or path.name.startswith('protocol-') or path.name.endswith('-result.json') or (path.name=='result.json' and 'kind' not in value):
                    details[path.name] = compact(value, root)
                else:
                    details[path.name] = compact({k:v for k,v in value.items() if k in DETAILS}, root)
        elif path.suffix == '.md' and path.stat().st_size <= 32768:
            excerpts[path.name] = clean(path.read_text(encoding='utf-8-sig'), root)
        elif path.suffix == '.log':
            with path.open('rb') as handle:
                handle.seek(max(0,path.stat().st_size-6144))
                tail = handle.read().decode('utf-8',errors='replace')
            selected = [line[:500] for line in tail.splitlines() if re.search(r'(?i)error|exception|traceback|failed|failure|timeout|denied|^OK$|Ran \d+ tests|exit[:=]',line)]
            if selected: diagnostics.append(path.name+':\n'+'\n'.join(selected[-8:]))
    record = load(source/'result.json',diagnostics) if (source/'result.json').exists() else {}
    record = record if isinstance(record,dict) else {}
    status = record.get('status')
    comparison = load(source/'comparison.json',diagnostics) if (source/'comparison.json').exists() else {}
    if status is None and isinstance(comparison,dict): status = comparison.get('status')
    if status is None and (source/'completion.json').exists():
        completion = load(source/'completion.json',diagnostics)
        if isinstance(completion,dict): status = completion.get('status')
    if status is None:
        status = 'stopped' if (source/'stopped.md').exists() else 'stage_evidence_only'
    execution = load(source/'execution.json',diagnostics) if (source/'execution.json').exists() else None
    recorded_hashes = {k:v for k,v in record.items() if k.endswith('_sha256')}
    snapshot = load(source/'snapshot.json',diagnostics) if (source/'snapshot.json').exists() else None
    # Archive only small Python source, never weights, dependencies, datasets or traces.
    candidates = list(source.glob('source/**/*.py')) + list(source.glob('*.py'))
    for path in sorted(candidates):
        if path.is_symlink() or not path.resolve().is_relative_to(source) or path.stat().st_size > 131072: continue
        digest = sha(path)
        name = path.relative_to(source).as_posix()
        destination = root/'reports/source-snapshots'/f'{digest}.py'
        destination.parent.mkdir(parents=True,exist_ok=True)
        if not destination.exists(): destination.write_bytes(path.read_bytes())
        if sha(destination) != digest: raise ValueError('Archived source hash mismatch')
        archived[name] = {'sha256':digest,'published_path':destination.relative_to(root).as_posix()}
        if isinstance(snapshot,dict) and name in snapshot and snapshot[name]!=digest:
            diagnostics.append('Captured source differs from recorded snapshot: '+name)
    protocol = record.get('protocol') or details.get('protocol.json')
    budget = 'fixed_updates' if isinstance(protocol,dict) and protocol.get('stopping_rule')=='optimizer_updates' else ('wall_time_budget' if isinstance(protocol,dict) and 'training_seconds' in protocol else 'stage_or_unknown')
    compatibility = {'kind':record.get('kind'),'protocol':protocol,'data_seal':record.get('data_seal'),
                     'source_hashes':recorded_hashes,'upstream':record.get('upstream')}
    fingerprint = hashlib.sha256(canonical(compatibility).encode()).hexdigest() if protocol else None
    def seeds_in(value):
        if isinstance(value,dict):
            found = [value['seed']] if type(value.get('seed')) is int else []
            for child in value.values(): found.extend(seeds_in(child))
            return found
        if isinstance(value,list): return [seed for child in value for seed in seeds_in(child)]
        return []
    seeds = sorted(set(seeds_in(record) + seeds_in(comparison))) or None
    report = {'schema_version':SCHEMA,'experiment_id':experiment_id,'artifact_directory':'runs/autoresearch/'+relative,
        'outcome':status,'budget_family':budget,'compatibility_fingerprint':fingerprint,
        'configuration':compact(record.get('candidate'),root),'protocol':compact(protocol,root),
        'seeds':seeds,
        'metrics':compact(record.get('metrics'),root),'details':details,
        'provenance':{'recorded_git_commit':record.get('git_commit'),'recorded_git_dirty':record.get('git_dirty'),
            'recorded_code_hashes':recorded_hashes,'recorded_data_hashes':record.get('data_seal'),
            'recorded_upstream':record.get('upstream'),'execution_receipt':execution,'snapshot_manifest':snapshot,
            'archived_source':archived,
            'qualification':'Execution receipt is historical evidence, not re-execution. Missing fields are unknown. Publishing commit is NOT the training commit.'},
        'diagnostics':[clean(d,root) for d in diagnostics], 'evidence':evidence,
        'limitations':['Different budgets, schedules, checkpointing and source versions are separate comparisons.',
            'No missing measurements or provenance were filled from the current working tree.',
            'Two paired seeds are preliminary; opposite signs do not establish equivalence.'],
        'reporter_sha256':sha(Path(__file__))}
    destination = root/'reports/experiments'/experiment_id
    save(destination/'report.json',canonical(report))
    lines = [f'# {relative}', '', f'Outcome: **{status}**. Budget family: **{budget}**.',
        '', 'Full compact configuration, metrics, seeds, hashes and diagnostics: [report.json](report.json).',
        '', 'Provenance: historical recorded commits/hashes only. This publication does not retroactively identify training source.',
        'Unknown fields remain null. Do not pool different budgets/schedules. Two seeds are not proof of equivalence.']
    if record.get('metrics'): lines += ['', '```json',canonical(record['metrics']).strip(),'```']
    if diagnostics: lines += ['', '## Diagnostics', '```text','\n\n'.join(clean(d,root) for d in diagnostics),'```']
    for name, excerpt in excerpts.items(): lines += ['', f'## Retained narrative: {name}', '', excerpt]
    save(destination/'README.md','\n'.join(lines).rstrip()+'\n')
    return report


def index(root=ROOT):
    root=Path(root)
    rows=[]
    for path in sorted((root/'reports/experiments').glob('*/report.json')):
        value=json.loads(path.read_text())
        rows.append({k:value[k] for k in ('experiment_id','artifact_directory','outcome','budget_family','compatibility_fingerprint','metrics')})
    save(root/'reports/index.json',canonical({'schema_version':SCHEMA,'experiments':rows}))
    lines=['# Research progress','', 'Generated offline from retained evidence. No cross-budget ranking or equivalence claim.',
           'Publication revision is distinct from executed training provenance. [Reporting and retention](../docs/research-publication.md).',
           '', '| Experiment | Outcome | Budget family | BPB |','|---|---|---|---:|']
    for row in rows:
        score=(row['metrics'] or {}).get('val_bpb','unknown')
        lines.append(f"| [{row['artifact_directory'].split('autoresearch/')[1]}](experiments/{row['experiment_id']}/README.md) | {row['outcome']} | {row['budget_family']} | {score} |")
    save(root/'reports/README.md','\n'.join(lines)+'\n')
    return rows


def backfill(root=ROOT):
    base=Path(root)/'runs/autoresearch'
    sources=set(p for p in base.iterdir() if p.is_dir()) if base.exists() else set()
    for path in base.rglob('*'):
        if path.is_file() and path.name in {'result.json','comparison.json','stopped.md','completion.json','analysis.json','profiling-report.md','protocol.json'} and 'source' not in path.relative_to(base).parts:
            sources.add(path.parent)
    for source in sorted(sources): emit(source,root)
    return index(root)


def publish(root=ROOT, execute=subprocess.run):
    """Push an already-reviewed commit; retry never creates a commit or rewrites reports."""
    def git(*args):
        return execute(['git',*args],cwd=root,check=True,capture_output=True,text=True).stdout.strip()
    branch=git('branch','--show-current')
    if not branch: raise ValueError('Detached HEAD: choose the publication branch explicitly.')
    head=git('rev-parse','HEAD')
    git('push','origin',f'HEAD:refs/heads/{branch}')
    remote=git('ls-remote','--heads','origin',f'refs/heads/{branch}').split()
    if not remote or remote[0]!=head: raise RuntimeError('Remote verification did not match local HEAD.')
    return {'branch':branch,'publication_commit':head,'verified_remote':True,
            'note':'Publication commit is not retrospective training provenance.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['one','backfill','publish'])
    parser.add_argument('source',nargs='?',type=Path)
    args=parser.parse_args()
    if args.action=='one':
        if args.source is None: parser.error('one requires an artifact directory')
        emit(args.source); index()
    elif args.action=='backfill': print(f'Reported {len(backfill())} experiment/stage directories.')
    else:
        try: print(canonical(publish()))
        except (OSError,ValueError,subprocess.SubprocessError,RuntimeError) as exc:
            print(f'Publication failed; local reports/results retained. Retry publish: {exc}',file=sys.stderr)
            return 1
    return 0


if __name__=='__main__': raise SystemExit(main())
