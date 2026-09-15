"""Apply reviewed module-initializer override only after the depth grid finishes."""
from pathlib import Path

import autoresearch as r
import ncp_campaign as c
import ncp_depth_grid
import ncp_module_seeds


def main():
    with c.gpu_lock():
        assert ncp_depth_grid.summary()['status']=='completed'
        assert ncp_module_seeds.entry()['status']=='qualified'
        preview = c.HERE/'module-extension-preview'
        receipt = c.read(preview/'receipt-aligned.json')
        assert receipt['status']=='completed'
        test = r.ROOT/'tests/test_ncp_module_seed.py'
        assert not test.exists()
        for name,hashes in receipt['files'].items():
            assert r.digest(r.ROOT/'scripts'/name)==hashes['original_sha256'],name
            assert r.digest(preview/'scripts'/name)==hashes['modified_sha256'],name
        assert r.digest(preview/'tests/test_ncp_module_seed.py')==receipt['tests']['test_sha256']
        frozen = c.read(c.HERE/'confirmation-selection.json')
        for name in ('autoresearch_model.py','autoresearch_ncp.py','autoresearch_bootstrap.py','autoresearch_memory.py'):
            assert c.sources()[name]==frozen['confirmation_source_hashes'][name]
        for name in receipt['files']:
            (r.ROOT/'scripts'/name).write_bytes((preview/'scripts'/name).read_bytes())
        test.write_bytes((preview/'tests/test_ncp_module_seed.py').read_bytes())
        archive = c.HERE/'module-integration-driver.py'
        assert not archive.exists()
        archive.write_bytes(Path(__file__).read_bytes())
        r.write_json(c.HERE/'module-integration-result.json',dict(kind='module_seed_integration',status='completed',
            applied_at=c.datetime.now(c.timezone.utc).isoformat(),changes=receipt['files'],
            test_sha256=r.digest(test),preview_receipt_sha256=r.digest(preview/'receipt-aligned.json'),
            driver_sha256=r.digest(archive),
            interpretation='Applied after depth-grid driver exit; only explicit module initialization changes, before optimizer construction. Fresh full gates required.'))
        c.log('Integrated reviewed module-initializer override; fresh full gates required')


if __name__=='__main__':
    main()
