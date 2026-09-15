"""Recover structural counts from a failed trial's captured code and logged config."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); run=args.run.resolve()
    assert sys.flags.isolated,'Use -I for captured-code reconstruction'
    record=json.loads((run/'result.json').read_text())
    assert record['status']=='failed'
    manifest=json.loads((run/'snapshot.json').read_text())
    for name,value in manifest.items(): assert digest(run/name)==value
    line=next(x for x in (run/'run.log').read_text().splitlines() if x.startswith('Model config: '))
    config=ast.literal_eval(line.removeprefix('Model config: ').replace('torch.bfloat16',"'bfloat16'"))
    sys.path[:0]=[str(run/'source/project'),str(run/'source/upstream')]
    import torch
    import train
    import autoresearch_model
    config['compute_dtype']=torch.bfloat16
    with torch.device('meta'):
        model=autoresearch_model.model_class(train,record['candidate'])(train.GPTConfig(**config))
    report=model.parameter_report()
    assert report['total_parameters']==sum(p.numel() for p in model.parameters())
    for module in (train,autoresearch_model):
        assert Path(module.__file__).resolve().is_relative_to(run/'source')
    result=dict(kind='failed_trial_structural_reconstruction',status='completed',trial=run.name,
        model=report,source_hashes=manifest,logged_configuration=line,
        log_sha256=digest(run/'run.log'),script_sha256=digest(__file__),
        interpretation='Exact structural parameter counts reconstructed on meta device from captured source and logged model configuration; no training, allocation peak, throughput or BPB was reconstructed')
    args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(trial=run.name,total=report['total_parameters'],ncp=report['ncp_parameters'])))


if __name__=='__main__': main()
