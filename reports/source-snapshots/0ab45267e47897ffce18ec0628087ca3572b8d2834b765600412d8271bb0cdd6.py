"""Publish this phase's locally generated reports through the authorized Git path."""
from pathlib import Path
import sys,time

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'scripts'))
import autoresearch as r
import ncp_campaign as c
import ncp_search as search

def events():
    return {line.split(' Publication deferred;')[0] for line in (c.HERE/'campaign.log').read_text().splitlines()
        if 'Publication deferred; local mechanism result retained:' in line}

started=time.monotonic();seen=events();last_published=time.time();complete_since=None;publications=0
labels={'R-raw','R-no_prediction','R-gain4','R-gain8','CAP-L2-K64'}
while time.monotonic()-started<1800 and c.remaining()>1200:
    current=events();new=current-seen
    finished=[p for p in c.HERE.glob('trial-*/result.json')
        if (record:=c.read(p))['label'] in labels and record['seed']==42 and record['status'] not in ('running','prepared')]
    all_done=len(finished)==len(labels)
    if all_done and complete_since is None: complete_since=time.monotonic()
    final_due=all_done and time.monotonic()-complete_since>=10 and last_published<max(p.stat().st_mtime for p in finished)
    if new or final_due:
        search.publish();last_published=time.time();publications+=1;seen=current
        r.write_json(c.HERE/'publication-watch-result.json',dict(kind='phase_publication_watch',status='running',publications=publications,gpu_work=False))
    if all_done and last_published>=max(p.stat().st_mtime for p in finished): break
    time.sleep(5)
r.write_json(c.HERE/'publication-watch-result.json',dict(kind='phase_publication_watch',status='completed' if all_done else 'timed_out',publications=publications,gpu_work=False))
c.log('Phase publication watcher finished; publications='+str(publications))
