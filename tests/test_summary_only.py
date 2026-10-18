"""Summary-only is passive and never reported as a candidate audit."""
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'scripts'))
from candidate_archive import ARCHIVE_NAME
from click_models import ClickConfig, ClickPolicy
from click_observations import ClickObservations
import analyze_sacm_experiments as analysis
import run_sacm_experiment as runner
import test_sacm_analysis as fixtures


class SummaryOnlyTests(unittest.TestCase):
    def cohort(self, root):
        folders = {c: fixtures.SACMAnalysisTests().make_run(root, c) for c in ('pbm', 'zero', 'heterogeneous')}
        before = analysis.analyze(root, 'pilot')
        for folder in folders.values():
            (folder/'candidates.csv').unlink()  # Temporary fixture only.
            manifest = json.loads((folder/'manifest.json').read_text())
            manifest['candidate_evidence'] = {'mode': 'summary-only', 'file': None, 'bytes': 0}
            (folder/'manifest.json').write_text(json.dumps(manifest))
            rows = [dict(json.loads(line), candidate_export=False) for line in (folder/'summaries.jsonl').read_text().splitlines()]
            (folder/'summaries.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
        return folders, before

    def test_default_never_opens_archive_and_keeps_all_statistics_and_rng(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            policy = ClickPolicy(ClickConfig(click_model='sacm_static', eta=1), 2023)
            _, _, diag = policy.select([4.,3.,2.,1.], [1,1,0,0], ['N-human','N-llm','M-human','M-llm'], 'U', ['event','U'])
            random_state, numpy_state = random.getstate(), np.random.get_state()
            snapshots = []
            for export in (False, True):
                folder = root/str(export)
                with ClickObservations(folder, 'sacm_static', export_candidates=export) as sink:
                    sink.record_initial(['N-human','M-human'], 'U', 'I', 1, 1)
                    first = sink.finish_epoch()
                    sink.record(diag, [1,1,3], round_number=2, training_epoch=1, event_key=['event','U'])
                    second = sink.finish_epoch()
                self.assertEqual((folder/ARCHIVE_NAME).exists(), export)
                self.assertFalse((folder/'candidates.csv').exists())
                self.assertTrue(all(r['candidate_export'] is export for r in (first, second)))
                snapshots.append([{k:v for k,v in r.items() if k!='candidate_export'} for r in (first,second)])
            self.assertEqual(snapshots[0], snapshots[1])
            self.assertEqual(random.getstate(), random_state)
            np.testing.assert_array_equal(np.random.get_state()[1], numpy_state[1])
            self.assertEqual(np.random.get_state()[2:], numpy_state[2:])
            with patch('click_observations.CandidateArchive') as archive:
                with ClickObservations(root/'default', 'rank') as sink:
                    sink.record_initial(['N-human'], 'U', 'I', 1, 1)
                    sink.finish_epoch()
                archive.assert_not_called()
            self.assertEqual({p.name for p in (root/'default').iterdir()}, {'summaries.jsonl'})

    def test_summary_only_keeps_main_results_but_not_candidate_audit(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, before = self.cohort(root)
            result = analysis.analyze(root, 'pilot')
            self.assertTrue(result['complete'], result['issues'])
            self.assertFalse(result['candidate_audit_complete'])
            self.assertEqual(result['validation_level'], 'summary-only')
            self.assertTrue(any('aggregate-only' in w for w in result['warnings']))
            for name in ('round_summary','relative_delta','primary_endpoint_by_seed','primary_endpoint_seed_summary','utility_raw','final_round_seed_summary'):
                self.assertEqual(result[name], before[name], name)
            strict = analysis.analyze(root, 'pilot', require_candidates=True)
            self.assertFalse(strict['complete'])
            self.assertTrue(any('candidate details required' in s for s in strict['issues']))

    def test_unexplained_missing_file_still_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for c in ('pbm','zero','heterogeneous'):
                folder = fixtures.SACMAnalysisTests().make_run(root, c)
            (folder/'candidates.csv').unlink()
            result = analysis.analyze(root, 'pilot')
            self.assertFalse(result['complete'])
            self.assertTrue(any('missing candidate diagnostics' in s for s in result['issues']))

    def test_false_declaration_and_inconsistent_summary_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folders, _ = self.cohort(root)
            path = folders['heterogeneous']/'summaries.jsonl'
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            rows[0]['candidate_export'] = True
            path.write_text('\n'.join(json.dumps(r) for r in rows))
            self.assertFalse(analysis.analyze(root, 'pilot')['complete'])
            rows[0]['candidate_export'] = False
            rows[-1]['click_ratio'] = .999
            path.write_text('\n'.join(json.dumps(r) for r in rows))
            result = analysis.analyze(root, 'pilot')
            self.assertFalse(result['complete'])
            self.assertTrue(any('differs from summary counts' in s for s in result['issues']))

    def test_runner_requires_explicit_archive_choice(self):
        args = SimpleNamespace(python='python', stage='pilot', seed=2023, condition='pbm')
        self.assertNotIn('--candidate_export', runner.build_command(args, Path('output')))
        args.candidate_export = 'archive'
        command = runner.build_command(args, Path('output'))
        self.assertEqual(command[command.index('--candidate_export')+1], 'archive')


if __name__ == '__main__':
    unittest.main()
