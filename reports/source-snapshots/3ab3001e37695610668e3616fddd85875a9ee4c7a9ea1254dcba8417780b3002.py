from pathlib import Path
import json,time,subprocess,sys
root=Path.cwd();here=root/'runs/autoresearch/ncp-20260915'
end=time.monotonic()+900
while True:
    record=json.loads((here/'trial-0005-N-prediction_weight0.03-s42/result.json').read_text())
    if record['status'] in ('completed','failed'): break
    if time.monotonic()>end: raise TimeoutError('Waiting for current GPU trial')
    time.sleep(5)
result=subprocess.run([sys.executable,'-B','-u','scripts/ncp_campaign.py','preflight'],cwd=root,timeout=500)
if result.returncode: raise RuntimeError('Shared gate failed; training not launched')
result=subprocess.run([sys.executable,'-B','-u','scripts/ncp_campaign.py','trial','--label','N-RMS','--seed','42','--hypothesis','After raw-latent scale growth, normalize pooled concept states at original alpha=beta=1 to test stability and token quality'],cwd=root,timeout=1000)
