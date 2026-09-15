from pathlib import Path
import json,time,subprocess,sys
root=Path.cwd();here=root/'runs/autoresearch/ncp-20260915'
end=time.monotonic()+900
while True:
    record=json.loads((here/'trial-0006-N-RMS-s42/result.json').read_text())
    if record['status'] in ('completed','failed'): break
    if time.monotonic()>end: raise TimeoutError('Waiting for current GPU trial')
    time.sleep(5)
plan=[('N-RMS-AUX',42,'screen','D6','Normalized-state auxiliary-only ablation isolates concept supervision from predicted feedback'),('D6',43,'depth-confirmation','D6','Fresh seed43 control for mandatory independent depth comparison at reference LR0.04'),('D12',43,'depth-confirmation','D6','Independent seed43 depth12 versus depth6; fixed tokens and LR0.04, report extra width/parameters/time'),('D6',44,'depth-confirmation','D6','Fresh seed44 control for mandatory independent depth comparison at reference LR0.04'),('D12',44,'depth-confirmation','D6','Independent seed44 depth12 versus depth6; fixed tokens and LR0.04, report extra width/parameters/time')]
for label,seed,phase,control,hypothesis in plan:
    result=subprocess.run([sys.executable,'-B','-u','scripts/ncp_campaign.py','trial','--label',label,'--seed',str(seed),'--phase',phase,'--control',control,'--hypothesis',hypothesis],cwd=root,timeout=1000)
    if result.returncode:
        files=sorted(here.glob('trial-*/result.json'));last=json.loads(files[-1].read_text())
        log=(files[-1].parent/'run.log').read_text(errors='replace')
        if 'Invalid training loss' not in log: raise RuntimeError('Unclassified failure requires diagnosis: '+str(files[-1]))
