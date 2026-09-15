"""Recover early controller bytes only when they exactly match recorded hashes."""
import hashlib
import subprocess

import autoresearch as r
import ncp_campaign as c


def main():
    revisions=subprocess.check_output(['git','log','--format=%H','--all','--','scripts/ncp_campaign.py'],cwd=r.ROOT,text=True).split()
    blobs={}
    for revision in revisions:
        content=subprocess.check_output(['git','show',revision+':scripts/ncp_campaign.py'],cwd=r.ROOT)
        blobs[hashlib.sha256(content).hexdigest()]=(revision,content)
    recovered=[]
    for name in ('trial-0001-D6-s42','trial-0002-NCP-s42','trial-0003-AUX-s42'):
        directory=c.HERE/name; record=c.read(directory/'result.json')
        expected=record['orchestrator_sha256']; revision,content=blobs[expected]
        path=directory/'orchestrator-recovered-from-git.py'
        if path.exists(): assert path.read_bytes()==content
        else: path.write_bytes(content)
        recovered.append(dict(trial=name,sha256=r.digest(path),git_revision=revision,
            git_path='scripts/ncp_campaign.py',recovered_file=path.name))
    r.write_json(c.HERE/'controller-recovery-result.json',dict(kind='controller_source_recovery',status='completed',
        recovered_at=c.datetime.now(c.timezone.utc).isoformat(),recovered=recovered,
        scope='Git blob bytes exactly match each original recorded controller SHA256; recovered after execution, not an original trial-time archive or process-memory inspection'))
    c.log('Recovered controller source for trials1-3 from Git; exact original recorded SHA256 matches; provenance receipt retained')


if __name__=='__main__': main()
