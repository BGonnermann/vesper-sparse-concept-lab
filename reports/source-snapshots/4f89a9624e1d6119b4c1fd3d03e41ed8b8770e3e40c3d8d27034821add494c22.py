from pathlib import Path
import json,time,subprocess,sys
root=Path.cwd();here=root/'runs/autoresearch/ncp-20260915'
end=time.monotonic()+900
while True:
    record=json.loads((here/'trial-0011-D12-s44/result.json').read_text())
    if record['status'] in ('completed','failed'): break
    if time.monotonic()>end: raise TimeoutError('Waiting for final depth GPU trial')
    time.sleep(5)
if record['status']!='completed': raise RuntimeError('Depth confirmation requires diagnosis')
result=subprocess.run([sys.executable,'-B','-u','scripts/ncp_campaign.py','preflight'],cwd=root,timeout=500)
if result.returncode: raise RuntimeError('Shared gate failed; no training launch')
for label in ('NCP','AUX'):
    result=subprocess.run([sys.executable,'-B','-u','scripts/ncp_campaign.py','trial','--label',label,'--seed','42','--phase','guard-correction','--hypothesis','Retry original unit-weight '+label+' after distinguishing finite auxiliary MSE from token CE in the stopping guard; same training objective and update budget'],cwd=root,timeout=1000)
    if result.returncode:
        path=sorted(here.glob('trial-*/result.json'))[-1]
        log=(path.parent/'run.log').read_text(errors='replace')
        if 'Invalid training loss' not in log: raise RuntimeError('Unclassified failure requires diagnosis')
