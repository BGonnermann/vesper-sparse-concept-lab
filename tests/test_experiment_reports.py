"""Offline report determinism, incomplete provenance and publication failure tests."""
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

SPEC=importlib.util.spec_from_file_location('reports',Path(__file__).resolve().parents[1]/'scripts/experiment_reports.py')
reports=importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(reports)


class ReportTests(unittest.TestCase):
    def test_ncp_health_and_prospective_hypothesis_are_published(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); source=self.fixture(root)
            (source/'ncp-health.json').write_text(json.dumps(dict(collapsed=True,target_perplexity=[1.],used_entries=[1])))
            (source/'selection.json').write_text(json.dumps(dict(hypothesis='Test delayed predicted feedback',phase='screen',control='D6')))
            result=reports.emit(source,root)
            self.assertTrue(result['details']['ncp-health.json']['collapsed'])
            self.assertEqual(result['details']['selection.json']['hypothesis'],'Test delayed predicted feedback')

    def test_runner_calls_reporting_after_success_and_failure(self):
        import sys
        spec=importlib.util.spec_from_file_location('report_runner',Path(__file__).resolve().parents[1]/'scripts/autoresearch.py')
        runner=importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
        with tempfile.TemporaryDirectory() as d, patch.object(runner,'emit_compact_report') as emit:
            root=Path(d)
            for code in (0,1):
                runner.run_trial([sys.executable,'-c',f'print("val_bpb: 1\\npeak_vram_mb: 1\\nnum_steps: 1");raise SystemExit({code})'],root,root/str(code),10,{})
            self.assertEqual(emit.call_count,2)

    def test_reporting_failure_does_not_rewrite_result(self):
        spec=importlib.util.spec_from_file_location('report_runner',Path(__file__).resolve().parents[1]/'scripts/autoresearch.py')
        runner=importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); source=self.fixture(root)
            before=(source/'result.json').read_bytes()
            with patch.object(runner,'ROOT',root), patch.object(runner,'checked',side_effect=OSError('offline')):
                runner.emit_compact_report(source)
            self.assertEqual(before,(source/'result.json').read_bytes())
            self.assertTrue((source/'reporting-error.json').exists())

    def fixture(self,root,status='completed',protocol=None):
        source=root/'runs/autoresearch/example'; source.mkdir(parents=True)
        (source/'result.json').write_text(json.dumps(dict(status=status,seed=43,
            protocol=protocol or {'training_seconds':300},candidate={'depth':6},
            git_commit='old-training-commit',git_dirty=True,metrics={'val_bpb':.63} if status=='completed' else None,
            runner_sha256='historical-hash',data_seal={'dataset':'old-data-hash'})))
        return source

    def test_retry_is_byte_identical_and_keeps_historical_provenance(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); source=self.fixture(root)
            first=reports.emit(source,root); reports.index(root)
            before={str(p.relative_to(root)):p.read_bytes() for p in (root/'reports').rglob('*') if p.is_file()}
            reports.backfill(root)
            after={str(p.relative_to(root)):p.read_bytes() for p in (root/'reports').rglob('*') if p.is_file()}
            self.assertEqual(before,after)
            self.assertEqual(first['provenance']['recorded_git_commit'],'old-training-commit')
            self.assertIsNone(first['provenance']['execution_receipt'])

    def test_failure_diagnostics_and_missing_provenance_are_not_success(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); source=self.fixture(root,'failed')
            (source/'run.log').write_text('RuntimeError: failed password=do-not-publish\n')
            result=reports.emit(source,root)
            self.assertEqual(result['outcome'],'failed'); self.assertIsNone(result['metrics'])
            self.assertIn('[REDACTED]',str(result['diagnostics']))
            self.assertNotIn('do-not-publish',str(result))

    def test_budget_and_schedule_fingerprints_are_distinct(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); source=self.fixture(root)
            a=reports.emit(source,root)
            value=json.loads((source/'result.json').read_text())
            value['protocol']={'stopping_rule':'optimizer_updates','optimizer_updates':512,'schedule':{'clock':'step'}}
            (source/'result.json').write_text(json.dumps(value))
            b=reports.emit(source,root)
            self.assertNotEqual(a['compatibility_fingerprint'],b['compatibility_fingerprint'])
            self.assertNotEqual(a['budget_family'],b['budget_family'])

    def test_malformed_record_is_diagnostic_not_completed(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); source=self.fixture(root)
            (source/'result.json').write_text('{broken')
            result=reports.emit(source,root)
            self.assertNotEqual(result['outcome'],'completed')
            self.assertTrue(result['diagnostics'])

    def test_upload_failure_preserves_reports_and_retry_verifies_remote(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); source=self.fixture(root); reports.emit(source,root)
            before=(source/'result.json').read_bytes()
            def fail(command,**kwargs):
                if command[1]=='push': raise subprocess.CalledProcessError(1,command)
                return subprocess.CompletedProcess(command,0,stdout='main' if command[1]=='branch' else 'abc')
            with self.assertRaises(subprocess.CalledProcessError): reports.publish(root,execute=fail)
            self.assertEqual(before,(source/'result.json').read_bytes())
            def succeed(command,**kwargs):
                return subprocess.CompletedProcess(command,0,stdout={'branch':'main','rev-parse':'abc','push':'','ls-remote':'abc refs/heads/main'}[command[1]])
            self.assertTrue(reports.publish(root,execute=succeed)['verified_remote'])


if __name__=='__main__': unittest.main()
