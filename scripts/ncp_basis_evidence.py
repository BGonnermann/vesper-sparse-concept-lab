"""Check frozen codebook-buffer identity across initializer-matched checkpoints."""
import hashlib
from pathlib import Path

import torch

import autoresearch as r
import ncp_campaign as c


def main():
    frozen = c.read(c.HERE / 'confirmation-selection.json')
    records = [(p.parent, c.read(p)) for p in c.HERE.glob('trial-*/result.json')]
    checks = []
    anchors = {}
    for seed in frozen['seeds'] + [42]:
        matches = [(path, row) for path, row in records if row['status']=='completed'
                   and row['seed']==seed and row['label']==frozen['label']]
        assert len(matches)==1
        path, record = matches[0]
        checkpoint = path / 'checkpoint_pre_eval.pt'
        assert r.digest(checkpoint)==record['checkpoint_sha256']
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        anchors[seed] = (path.name, state['ncp.codebook.basis'].clone())
        del state
    for path, record in records:
        if record['status']!='completed' or record['label'] not in ('FROZEN-AUX','FROZEN-NOPRED','CROSS-NCP','MODULE-NCP','FROZEN-NOFUTURE'):
            continue
        seed = record['protocol'].get('ncp_initialization_seed',record['seed'])
        checkpoint = path / 'checkpoint_pre_eval.pt'
        assert r.digest(checkpoint)==record['checkpoint_sha256']
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        basis = state['ncp.codebook.basis']
        assert torch.equal(basis, anchors[seed][1]), ('Frozen basis differs for the same initializer', path)
        checks.append(dict(trial=path.name, initialization_seed=record['seed'],ncp_initialization_seed=seed,
            batch_order_seed=record['protocol'].get('batch_order_seed',record['seed']), anchor_trial=anchors[seed][0],
            checkpoint_sha256=record['checkpoint_sha256'], basis_shape=list(basis.shape),
            basis_sha256=hashlib.sha256(basis.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest(), verified=True))
        del state
    assert len(checks)>=8
    r.write_json(c.HERE / 'basis-evidence-result.json', dict(kind='frozen_basis_identity_audit',status='completed',
        measured_at=c.datetime.now(c.timezone.utc).isoformat(), checks=checks, script_sha256=r.digest(Path(__file__)),
        interpretation='Actual saved frozen basis buffers match initializer-matched NCP anchors; trainable codebook transforms are expected to differ after different training. Complements named-parameter initialization receipts.'))
    print(f'PASS: {len(checks)} frozen-basis checkpoint comparisons')


if __name__=='__main__':
    torch.set_num_threads(2)
    main()
