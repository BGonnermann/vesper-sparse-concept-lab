"""Distinguish inspected Windows-checkout hashes from pinned upstream Git blobs."""
import hashlib
from pathlib import Path
import subprocess

import autoresearch as r
import ncp_campaign as c


def main():
    receipt=c.read(c.HERE/'sources-result.json')
    repository=c.HERE/'official'
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repository,text=True).strip()
    assert revision==receipt['revision']
    checks=[]
    for name,expected in receipt['source_hashes'].items():
        checkout=(repository/name).read_bytes()
        assert hashlib.sha256(checkout).hexdigest()==expected,'Inspected file changed'
        blob=subprocess.check_output(['git','show',revision+':'+name],cwd=repository)
        assert checkout.replace(b'\r\n',b'\n')==blob,'Difference extends beyond Windows newlines'
        checks.append(dict(file=name,inspected_checkout_sha256=expected,
            upstream_git_blob_sha256=hashlib.sha256(blob).hexdigest(),
            bytes_identical=checkout==blob,lf_normalized_bytes_identical=True,
            checkout_crlf_count=checkout.count(b'\r\n'),upstream_crlf_count=blob.count(b'\r\n')))
    r.write_json(c.HERE/'official-source-byte-audit-result.json',dict(kind='official_source_byte_audit',
        status='completed',revision=revision,checks=checks,script_sha256=r.digest(Path(__file__)),
        measured_at=c.datetime.now(c.timezone.utc).isoformat(),
        interpretation='Original sources-result hashes identify the inspected Windows checkout with CRLF newlines. Raw pinned Git-blob hashes are recorded separately here. Bytes match exactly after LF normalization; no other differences. Official model code was not imported into or copied into training. Own executed-source archives retain exact byte hashes.'))
    print('PASS: four inspected files match the pinned Git blobs after newline normalization; both hash forms recorded')


if __name__=='__main__':
    main()
