from pathlib import Path
import json,time,subprocess,sys
root=Path.cwd(); here=root/'runs/autoresearch/ncp-20260915'
end=time.monotonic()+900
while True:
    record=json.loads((here/'trial-0001-D6-s42/result.json').read_text())
    if record['status'] in ('completed','failed'): break
    if time.monotonic()>end: raise TimeoutError('Waiting for initial control')
    time.sleep(5)
if record['status']!='completed': raise RuntimeError(record)
for label,hypothesis in [('NCP','Source-inspired discrete chunk prediction plus causal predicted feedback may improve BPB at equal tokens'),('AUX','Remove predicted feedback at identical NCP parameter count to isolate auxiliary supervision')]:
    result=subprocess.run([sys.executable,'-B','-u','scripts/ncp_campaign.py','trial','--label',label,'--seed','42','--hypothesis',hypothesis],cwd=root,timeout=1000)
    if result.returncode: raise RuntimeError('Preserved failed '+label+' trial; diagnose before more trials')
