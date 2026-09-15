"""Apply the reviewed isolated preview after frozen confirmation has completed."""
from pathlib import Path

import autoresearch as r
import ncp_campaign as c
from ncp_frozen_pairs import collect


def main():
    with c.gpu_lock():
        frozen = c.read(c.HERE / 'confirmation-selection.json')
        evidence = collect(frozen)
        assert all(x['completed_pairs']==4 for x in evidence['summaries'].values())
        preview = c.HERE / 'order-extension-preview'
        receipt = c.read(preview / 'receipt-v2.json')
        assert receipt['status']=='completed' and receipt['revision']==2
        test = r.ROOT / 'tests/test_ncp_order.py'
        assert not test.exists(), 'New test destination already exists'
        for name, hashes in receipt['changes'].items():
            assert r.digest(r.ROOT / 'scripts' / name)==hashes['original_sha256'], name
            assert r.digest(preview / 'scripts' / name)==hashes['modified_sha256'], name
        assert r.digest(preview / 'tests/test_ncp_order.py')==receipt['checks']['test_sha256']
        for name in ('autoresearch_model.py','autoresearch_ncp.py','autoresearch_bootstrap.py','autoresearch_memory.py'):
            assert r.digest(r.ROOT / 'scripts' / name)==frozen['confirmation_source_hashes'][name]
        for name in receipt['changes']:
            (r.ROOT / 'scripts' / name).write_bytes((preview / 'scripts' / name).read_bytes())
        test.write_bytes((preview / 'tests/test_ncp_order.py').read_bytes())
        archive = c.HERE / 'order-integration-driver.py'
        assert not archive.exists()
        archive.write_bytes(Path(__file__).read_bytes())
        r.write_json(c.HERE / 'order-integration-result.json', dict(kind='order_seed_integration',status='completed',
            applied_at=c.datetime.now(c.timezone.utc).isoformat(), changes=receipt['changes'],
            test_sha256=r.digest(test), preview_receipt_sha256=r.digest(preview / 'receipt-v2.json'),
            driver_sha256=r.digest(archive), confirmation_selection_sha256=r.digest(c.HERE / 'confirmation-selection.json'),
            interpretation='Applied after all four frozen confirmation seeds and controls completed; no model/mechanism source changes. Fresh full gates required before training.'))
        c.log('Integrated reviewed order-seed preview after frozen confirmation; fresh full gates required')


if __name__=='__main__':
    main()
