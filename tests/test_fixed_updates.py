"""Fixed budget, schedule, and independent seeded ordering contracts."""
import sys
from pathlib import Path
import unittest
import contextlib
import io
import os
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from autoresearch_train import fixed_schedule, batch_order, fixed_training
from autoresearch import digest


class Toy(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(2))

    def init_weights(self, **kwargs):
        torch.nn.init.ones_(self.weight)

    def forward(self, x, y):
        return self.weight.square().mean()

    def setup_optimizer(self, **kwargs):
        self.optimizer = torch.optim.SGD([{'params':list(self.parameters()), 'kind':'adamw', 'initial_lr':.001}],lr=.001)
        self.calls = 0
        step = self.optimizer.step
        def counted():
            self.calls += 1
            return step()
        self.optimizer.step = counted
        return self.optimizer

    def num_scaling_params(self): return {'total':2}
    def estimate_flops(self): return 12


class FixedTests(unittest.TestCase):
    def test_loss_failure_preserves_partial_state_and_accounting(self):
        import json
        class Unstable(Toy):
            def forward(self,x,y): return self.weight.square().mean()*200
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory);torch.save(torch.zeros(512,2,1,2,dtype=torch.int64),path/'tape.pt')
            protocol=dict(seed=43,tokens_per_update=2,sequence_length=2,optimizer_updates=512,
                batch_tape=str(path/'tape.pt'),batch_tape_sha256=digest(path/'tape.pt'),schedule={})
            train=SimpleNamespace(GPT=Unstable,UNEMBEDDING_LR=.004,EMBEDDING_LR=.6,SCALAR_LR=.5,
                ADAM_BETAS=(.8,.95),MATRIX_LR=.04,WEIGHT_DECAY=.2)
            runtime=SimpleNamespace(device='cpu',device_type='cpu',amp_dtype=torch.bfloat16)
            cwd=Path.cwd()
            try:
                os.chdir(path)
                with patch('torch.cuda.synchronize'),patch.object(torch.Tensor,'pin_memory',lambda self:self),contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaisesRegex(RuntimeError,'Invalid training loss'):
                        fixed_training(train,protocol,runtime,None,None,1,False)
                self.assertTrue((path/'training-failure.json').exists(),'Missing partial failure receipt')
                receipt=json.loads((path/'training-failure.json').read_text())
                self.assertEqual(receipt['optimizer_updates'],1)
                self.assertEqual(receipt['training_tokens'],2)
                self.assertTrue((path/'checkpoint_failure.pt').exists())
            finally: os.chdir(cwd)

    def test_actual_loop_stops_at_512_including_warmup(self):
        import json
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            torch.save(torch.zeros(512,2,1,2,dtype=torch.int64),path/'tape.pt')
            protocol = dict(seed=43,tokens_per_update=2,sequence_length=2,optimizer_updates=512,
                            batch_tape=str(path/'tape.pt'),batch_tape_sha256=digest(path/'tape.pt'),schedule={})
            train = SimpleNamespace(GPT=Toy,UNEMBEDDING_LR=.004,EMBEDDING_LR=.6,SCALAR_LR=.5,
                                    ADAM_BETAS=(.8,.95),MATRIX_LR=.04,WEIGHT_DECAY=.2)
            runtime = SimpleNamespace(device='cpu',device_type='cpu',amp_dtype=torch.bfloat16)
            cwd = Path.cwd()
            try:
                os.chdir(path)
                with patch('torch.cuda.synchronize'), patch.object(torch.Tensor,'pin_memory',lambda self:self), contextlib.redirect_stdout(io.StringIO()):
                    result = fixed_training(train,protocol,runtime,None,None,1,False)
                self.assertEqual(result['step'],512)
                self.assertEqual(result['model'].calls,512)
                batches = json.loads((path/'batches.json').read_text())
                self.assertEqual(batches['microbatches'],512)
                self.assertEqual(batches['consumed_indices'],batch_order(512,43))
                self.assertGreater(result['total_training_time'],0)
            finally:
                os.chdir(cwd)

    def test_schedule_exact_budget_and_endpoints(self):
        rows = fixed_schedule()
        self.assertEqual(len(rows), 512)
        self.assertEqual([r['step'] for r in rows], list(range(512)))
        self.assertEqual(len(rows) * 16384, 8388608)
        self.assertTrue(all(r['lr_multiplier'] == 1 for r in rows[:257]))
        self.assertEqual(rows[-1]['lr_multiplier'], 1 / 256)
        self.assertEqual(rows[300]['muon_momentum'], .95)
        self.assertEqual(rows[-1]['muon_weight_decay'], .2 / 512)

    def test_seed_controls_complete_order_independent_of_model_rng(self):
        a = batch_order(8192, 42)
        torch.manual_seed(991)
        torch.rand(1234)
        self.assertEqual(a, batch_order(8192, 42))
        self.assertNotEqual(a, batch_order(8192, 43))
        self.assertEqual(sorted(a), list(range(8192)))


if __name__ == '__main__':
    unittest.main()
