"""Prospective checks for separate initializer and tape-order identities."""
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'tests')]
from autoresearch import digest, validate_protocol
from autoresearch_train import batch_order, fixed_training
from test_fixed_updates import Toy


class RandomToy(Toy):
    def init_weights(self, **kwargs):
        torch.nn.init.normal_(self.weight, std=.1)

    def forward(self, x, y):
        return (self.weight - x.float().mean() / 512).square().mean()


class OrderIdentityTests(unittest.TestCase):
    def test_protocol_accepts_only_bounded_explicit_order_seed(self):
        protocol = json.loads((ROOT / 'runs/autoresearch/equal-token-20260915/protocol-42.json').read_text())
        self.assertEqual(validate_protocol(dict(protocol)), protocol)
        for seed in (42, 45):
            self.assertEqual(validate_protocol(dict(protocol, batch_order_seed=seed))['batch_order_seed'], seed)
        for seed in (True, 0, 41, 43, 46, '45'):
            with self.assertRaises(ValueError):
                validate_protocol(dict(protocol, batch_order_seed=seed))

    def test_actual_loop_separates_initialization_and_order(self):
        receipts = {}
        cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tape = torch.arange(512, dtype=torch.int64)[:, None, None, None].expand(512, 2, 1, 2).clone()
            tape_path = root / 'tape.pt'
            torch.save(tape, tape_path)
            for seed, order_seed in ((42, None), (42, 42), (42, 45), (45, 42), (45, 45)):
                path = root / f'{seed}-{order_seed}'
                path.mkdir()
                protocol = dict(seed=seed, tokens_per_update=2, sequence_length=2, optimizer_updates=512,
                                batch_tape=str(tape_path), batch_tape_sha256=digest(tape_path), schedule={})
                if order_seed is not None:
                    protocol['batch_order_seed'] = order_seed
                train = SimpleNamespace(GPT=RandomToy, UNEMBEDDING_LR=.004, EMBEDDING_LR=.6, SCALAR_LR=.5,
                                        ADAM_BETAS=(.8, .95), MATRIX_LR=.04, WEIGHT_DECAY=.2)
                runtime = SimpleNamespace(device='cpu', device_type='cpu', amp_dtype=torch.bfloat16)
                try:
                    os.chdir(path)
                    with patch('torch.cuda.synchronize'), patch.object(torch.Tensor, 'pin_memory', lambda self: self), contextlib.redirect_stdout(io.StringIO()):
                        result = fixed_training(train, protocol, runtime, None, None, 1, False)
                    receipts[(seed, order_seed)] = dict(initial=json.loads((path / 'initialization.json').read_text()),
                        batches=json.loads((path / 'batches.json').read_text()), weights=result['model'].weight.detach().clone())
                finally:
                    os.chdir(cwd)
        default, explicit = receipts[(42, None)], receipts[(42, 42)]
        self.assertEqual(default['initial'], explicit['initial'])
        self.assertEqual(default['batches']['consumed_indices'], explicit['batches']['consumed_indices'])
        torch.testing.assert_close(default['weights'], explicit['weights'], rtol=0, atol=0)
        for seed in (42, 45):
            left, right = receipts[(seed, 42)], receipts[(seed, 45)]
            self.assertEqual(left['initial'], right['initial'])
            self.assertNotEqual(left['batches']['consumed_indices'], right['batches']['consumed_indices'])
            self.assertFalse(torch.equal(left['weights'], right['weights']), 'Changing order must reach the actual optimizer path')
        for order_seed in (42, 45):
            left, right = receipts[(42, order_seed)], receipts[(45, order_seed)]
            self.assertNotEqual(left['initial']['parameters'], right['initial']['parameters'])
            self.assertEqual(left['batches']['consumed_indices'], right['batches']['consumed_indices'])
            for item in (left, right):
                self.assertEqual(item['batches']['consumed_indices'], batch_order(512, order_seed))
                self.assertEqual(sorted(item['batches']['consumed_indices']), list(range(512)))
                self.assertEqual(item['batches']['batch_order_seed'], order_seed)


if __name__ == '__main__':
    unittest.main(verbosity=2)
