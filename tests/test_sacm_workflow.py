"""Staged execution contracts; all launch/probe calls are mocked."""
import copy
import io
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_sacm_workflow as workflow
import run_sacm_experiment as runner
import analyze_sacm_experiments as analysis
import test_sacm_analysis as analysis_fixtures


class WorkflowTests(unittest.TestCase):
    def manifest(self, seed=2023, condition='pbm', status='completed'):
        return {'stage': 'full', 'condition': condition, 'seed': seed, 'status': status,
                'source_sha256': {'src/a.py': 'source'}, 'data_sha256': {'data': 'hash'},
                'git_head': 'commit', 'versions': {'python': '3.10'}, 'platform': 'test',
                'environment': {'PYTHONHASHSEED': str(seed), 'HF_HUB_OFFLINE': '1'},
                'baseline_provenance': {'cache_environment': {'HF_HOME': 'cache'}, 'news_batch_size': 64}}

    def completed(self, cohort, condition='pbm', seed=2023, name=None, status='completed'):
        directory = cohort / 'runs' / (name or f'{condition}-{seed}')
        directory.mkdir(parents=True)
        (directory / 'manifest.json').write_text(json.dumps(self.manifest(seed, condition, status)))
        return directory

    def freeze(self, cohort):
        cohort.mkdir(parents=True, exist_ok=True)
        value = {'signature': runner.cohort_signature(self.manifest())}
        (cohort / 'cohort.json').write_text(json.dumps(value))
        return value

    def test_default_does_not_write_probe_or_launch(self):
        with tempfile.TemporaryDirectory() as temp:
            cohort = Path(temp) / 'absent'
            with patch.object(workflow, 'invoke') as invoke, patch.object(workflow, 'prepare') as prepare, \
                 patch.object(workflow, 'launch_cell') as launch, patch('sys.stdout', new=io.StringIO()):
                self.assertEqual(workflow.main(['first', '--cohort', str(cohort)]), 0)
            invoke.assert_not_called()
            prepare.assert_not_called()
            launch.assert_not_called()
            self.assertFalse(cohort.exists())

    def test_failed_attempt_retained_and_duplicate_completed_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            cohort = Path(temp)
            accepted = self.completed(cohort)
            failed = self.completed(cohort, name='failed-attempt', status='failed')
            self.assertEqual(workflow.completed_runs(cohort), {('pbm', 2023): accepted})
            self.assertTrue(failed.exists())
            self.completed(cohort, name='duplicate')
            with self.assertRaisesRegex(ValueError, 'Duplicate completed'):
                workflow.completed_runs(cohort)

    def test_first_reuses_only_after_audit_and_never_enters_core(self):
        with tempfile.TemporaryDirectory() as temp:
            cohort = Path(temp)
            self.freeze(cohort)
            accepted = self.completed(cohort)
            self.completed(cohort, name='failure', status='failed')
            with patch.object(workflow, 'prepare', return_value=self.manifest()), \
                 patch.object(workflow.shutil, 'disk_usage', return_value=SimpleNamespace(free=100 * 1024**3)), \
                 patch.object(workflow, 'audit_cell', return_value={'passed': True}) as audit, \
                 patch.object(workflow, 'launch_cell') as launch, \
                 patch.object(workflow, 'calibrate_first', return_value={'seconds': 1}), \
                 patch.object(workflow, 'matrix_acceptance') as matrix, patch('sys.stdout', new=io.StringIO()):
                self.assertEqual(workflow.main(['first', '--cohort', str(cohort), '--execute']), 0)
            audit.assert_called_once()
            self.assertEqual(audit.call_args.args[0], accepted)
            launch.assert_not_called()
            matrix.assert_not_called()

    def test_first_launches_exactly_one_cell(self):
        with tempfile.TemporaryDirectory() as temp:
            cohort = Path(temp) / 'new'
            with patch.object(workflow, 'invoke') as invoke, \
                 patch.object(workflow, 'prepare', return_value=self.manifest()), \
                 patch.object(workflow.shutil, 'disk_usage', return_value=SimpleNamespace(free=100 * 1024**3)), \
                 patch.object(workflow, 'launch_cell', return_value=cohort / 'runs/pbm') as launch, \
                 patch.object(workflow, 'calibrate_first', return_value={'seconds': 1}), \
                 patch.object(workflow, 'matrix_acceptance') as matrix, patch('sys.stdout', new=io.StringIO()):
                self.assertEqual(workflow.main(['first', '--cohort', str(cohort), '--execute']), 0)
            self.assertEqual(launch.call_count, 1)
            self.assertEqual(launch.call_args.args[2:4], ('pbm', 2023))
            self.assertEqual(invoke.call_count, 1)  # engineering contract gate only
            self.assertTrue((cohort / 'cohort.json').exists())
            matrix.assert_not_called()

    def test_missing_first_cell_refuses_continuation(self):
        with tempfile.TemporaryDirectory() as temp:
            cohort = Path(temp)
            self.freeze(cohort)
            with patch.object(workflow, 'prepare', return_value=self.manifest()), \
                 patch.object(workflow.shutil, 'disk_usage', return_value=SimpleNamespace(free=100 * 1024**3)), \
                 patch.object(workflow, 'launch_cell') as launch:
                with self.assertRaisesRegex(ValueError, 'first run is required'):
                    workflow.main(['core', '--cohort', str(cohort), '--execute'])
            launch.assert_not_called()

    def test_signature_excludes_seed_but_includes_source_data_cache_and_environment(self):
        original = self.manifest()
        self.assertEqual(runner.cohort_signature(original), runner.cohort_signature(self.manifest(seed=2027)))
        mutations = [lambda m: m['source_sha256'].update({'src/a.py': 'changed'}),
                     lambda m: m['data_sha256'].update(data='changed'),
                     lambda m: m['baseline_provenance']['cache_environment'].update(HF_HOME='changed'),
                     lambda m: m['environment'].update(HF_HUB_OFFLINE='0')]
        frozen = {'signature': runner.cohort_signature(original)}
        for mutation in mutations:
            changed = copy.deepcopy(original)
            mutation(changed)
            with self.assertRaisesRegex(ValueError, 'signature mismatch'):
                workflow.require_signature(changed, frozen)

    def test_changed_source_stops_before_training(self):
        with tempfile.TemporaryDirectory() as temp:
            cohort = Path(temp)
            self.freeze(cohort)
            self.completed(cohort)
            changed = self.manifest()
            changed['source_sha256']['src/a.py'] = 'changed'
            with patch.object(workflow, 'prepare', return_value=changed), \
                 patch.object(workflow.shutil, 'disk_usage', return_value=SimpleNamespace(free=100 * 1024**3)), \
                 patch.object(workflow, 'launch_cell') as launch:
                with self.assertRaisesRegex(ValueError, 'signature mismatch'):
                    workflow.main(['core', '--cohort', str(cohort), '--execute'])
            launch.assert_not_called()

    def test_selected_cells_acceptance_cannot_claim_full_matrix(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = analysis_fixtures.SACMAnalysisTests().make_run(root, 'pbm', stage='full')
            local = analysis.analyze(root, 'full', 'core', run_paths=[path], required_cells={('pbm', 2023)})
            self.assertTrue(local['complete'], local['issues'])
            self.assertEqual(local['acceptance_scope'], 'selected-cells-only')
            full = analysis.analyze(root, 'full', 'core', run_paths=[path])
            self.assertFalse(full['complete'])
            self.assertEqual(full['acceptance_scope'], 'full-matrix')
            self.assertEqual(full['expected_runs'], 15)

    def test_existing_core_is_revalidated_and_not_retrained(self):
        with tempfile.TemporaryDirectory() as temp:
            cohort = Path(temp)
            self.freeze(cohort)
            for condition, seed in workflow.cells('core'):
                self.completed(cohort, condition, seed)
            with patch.object(workflow, 'prepare', return_value=self.manifest()), \
                 patch.object(workflow.shutil, 'disk_usage', return_value=SimpleNamespace(free=100 * 1024**3)), \
                 patch.object(workflow, 'audit_cell', return_value={}) as audit, \
                 patch.object(workflow, 'launch_cell') as launch, \
                 patch.object(workflow.analysis, 'analyze', return_value={'complete': True}) as paired, \
                 patch.object(workflow, 'matrix_acceptance') as matrix, patch('sys.stdout', new=io.StringIO()):
                self.assertEqual(workflow.main(['core', '--cohort', str(cohort), '--execute']), 0)
            launch.assert_not_called()
            self.assertEqual(audit.call_count, 16)  # first prerequisite plus all 15 cells
            self.assertEqual(paired.call_count, 5)
            matrix.assert_called_once()

    def test_bad_previously_completed_cell_is_not_silently_replaced(self):
        with tempfile.TemporaryDirectory() as temp:
            cohort = Path(temp)
            self.freeze(cohort)
            self.completed(cohort)
            with patch.object(workflow, 'prepare', return_value=self.manifest()), \
                 patch.object(workflow.shutil, 'disk_usage', return_value=SimpleNamespace(free=100 * 1024**3)), \
                 patch.object(workflow, 'audit_cell', side_effect=ValueError('rejected')) as audit, \
                 patch.object(workflow, 'launch_cell') as launch:
                with self.assertRaisesRegex(ValueError, 'rejected'):
                    workflow.main(['first', '--cohort', str(cohort), '--execute'])
            audit.assert_called_once()
            launch.assert_not_called()

    def test_between_cell_runner_drift_failure_stops_queue(self):
        with tempfile.TemporaryDirectory() as temp:
            cohort = Path(temp)
            self.freeze(cohort)
            self.completed(cohort)
            with patch.object(workflow, 'prepare', return_value=self.manifest()), \
                 patch.object(workflow.shutil, 'disk_usage', return_value=SimpleNamespace(free=100 * 1024**3)), \
                 patch.object(workflow, 'audit_cell', return_value={}), \
                 patch.object(workflow, 'launch_cell', side_effect=RuntimeError('Frozen cohort signature mismatch')) as launch, \
                 patch.object(workflow, 'matrix_acceptance') as matrix, patch('sys.stdout', new=io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, 'signature mismatch'):
                    workflow.main(['core', '--cohort', str(cohort), '--execute'])
            self.assertEqual(launch.call_count, 1)
            self.assertEqual(launch.call_args.args[2:4], ('zero', 2023))
            matrix.assert_not_called()

    def test_failed_zero_control_blocks_heterogeneous_launch(self):
        with tempfile.TemporaryDirectory() as temp:
            cohort = Path(temp)
            self.freeze(cohort)
            self.completed(cohort)
            self.completed(cohort, 'zero')
            with patch.object(workflow, 'prepare', return_value=self.manifest()), \
                 patch.object(workflow.shutil, 'disk_usage', return_value=SimpleNamespace(free=100 * 1024**3)), \
                 patch.object(workflow, 'audit_cell', return_value={}), \
                 patch.object(workflow.analysis, 'analyze', return_value={'complete': False, 'issues': ['mismatch']}), \
                 patch.object(workflow, 'launch_cell') as launch, patch('sys.stdout', new=io.StringIO()):
                with self.assertRaisesRegex(ValueError, 'Zero-alpha acceptance failed'):
                    workflow.main(['core', '--cohort', str(cohort), '--execute'])
            launch.assert_not_called()


if __name__ == '__main__':
    unittest.main()
