import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from round_observations import RoundObservations

spec = importlib.util.spec_from_file_location('round_validator', ROOT / 'scripts/validate_round_artifacts.py')
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)


class ArtifactValidationTests(unittest.TestCase):
    def fixture(self, root):
        run = root / 'run'
        run.mkdir()
        behavior = root / 'behavior.tsv'
        behavior.write_text('1\tu\tt\tA\tA-1 B-0\n', encoding='utf-8')
        human, llm = root / 'human.tsv', root / 'llm.tsv'
        human.write_text('A\thuman\n', encoding='utf-8')
        llm.write_text('A\tgenerated\n', encoding='utf-8')
        args = SimpleNamespace(test_behaviors_file=str(behavior), seed=2023, epochs=2, loop_epochs=2,
                               human_news_file=str(human), llm_news_file=str(llm))
        vectors = {'A': torch.tensor([1., 0.]), 'B': torch.tensor([0., 1.])}
        writer = RoundObservations(run / 'round_artifacts', args, vectors, vectors)
        diagnostics = {'total_impressions': 3, 'skipped_homogeneous_labels': 1,
                       'skipped_first_pair_equal': 1, 'evaluated_impressions': 1,
                       'positive_pair_ties': 1, 'positive_pair_human_wins': 1,
                       'positive_pair_llm_wins': 0, 'first_score_pairs_human_wins': 1,
                       'first_score_pairs_llm_wins': 0, 'first_score_pairs_ties': 1}
        encoder = torch.nn.Linear(2, 2)
        writer.save_round(1, encoder, 0., diagnostics)
        writer.save_round(2, encoder, .5, diagnostics)
        records = [dict(round=1, training_epoch=1, clicks=2, click_llm=0),
                   dict(round=1, training_epoch=2, clicks=2, click_llm=0),
                   dict(round=2, training_epoch=1, clicks=2, click_llm=0),
                   dict(round=2, training_epoch=2, clicks=2, click_llm=1)]
        (run / 'summaries.jsonl').write_text('\n'.join(json.dumps(r) for r in records), encoding='utf-8')
        return run

    def mutate_snapshot(self, run, mutation):
        path = run / 'round_artifacts/round-002.pt'
        value = torch.load(path, map_location='cpu', weights_only=True)
        mutation(value)
        torch.save(value, path)

    def test_valid_last_epoch_and_cpu_rng_preservation(self):
        with tempfile.TemporaryDirectory() as temp:
            run = self.fixture(Path(temp))
            before = torch.get_rng_state().clone()
            report = validator.audit_round_artifacts(run, 2)
            self.assertTrue(report['passed'], report)
            self.assertEqual(report['artifacts']['rounds_checked'], 2)
            self.assertGreater(report['artifacts']['bytes'], 0)
            self.assertTrue(torch.equal(before, torch.get_rng_state()))
            self.assertFalse(validator.audit_round_artifacts(run)['passed'])

    def test_snapshot_contract_failures(self):
        mutations = {
            'round': lambda p: p.update(round=1),
            'args': lambda p: p['args'].update(seed=999),
            'provenance': lambda p: p['provenance'].update(torch_version='wrong'),
            'ratio': lambda p: p.update(native_feedback_ratio=.25),
            'nan': lambda p: p['user_encoder_state_dict']['weight'].fill_(float('nan')),
            'empty': lambda p: p.update(user_encoder_state_dict={}),
            'diagnostics': lambda p: p.update(native_eval_diagnostics=None),
            'counts': lambda p: p['native_eval_diagnostics'].update(evaluated_impressions=2),
        }
        for name, mutation in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temp:
                run = self.fixture(Path(temp))
                self.mutate_snapshot(run, mutation)
                report = validator.audit_round_artifacts(run, 2)
                self.assertFalse(report['passed'], report)
                self.assertTrue(report['issues'])

    def test_manifest_vectors_behavior_and_missing_round_failures(self):
        for kind in ('schema', 'vectors', 'behavior', 'missing', 'unexpected', 'corrupt'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temp:
                run = self.fixture(Path(temp))
                folder = run / 'round_artifacts'
                if kind == 'schema':
                    path = folder / 'manifest.json'
                    value = json.loads(path.read_text())
                    value['schema'] = 'future-schema'
                    path.write_text(json.dumps(value))
                elif kind == 'vectors':
                    with (folder / 'text_vectors.pt').open('ab') as stream:
                        stream.write(b'changed')
                elif kind == 'behavior':
                    (Path(temp) / 'behavior.tsv').write_text('changed')
                elif kind == 'missing':
                    (folder / 'round-002.pt').unlink()
                elif kind == 'unexpected':
                    (folder / 'round-003.pt').write_bytes((folder / 'round-002.pt').read_bytes())
                else:
                    (folder / 'round-002.pt').write_bytes(b'not a snapshot')
                self.assertFalse(validator.audit_round_artifacts(run, 2)['passed'])

    def test_missing_final_epoch_is_not_accepted_as_last_epoch(self):
        with tempfile.TemporaryDirectory() as temp:
            run = self.fixture(Path(temp))
            path = run / 'summaries.jsonl'
            records = [json.loads(line) for line in path.read_text().splitlines()]
            records = [r for r in records if not (r['round'] == 1 and r['training_epoch'] == 2)]
            path.write_text('\n'.join(json.dumps(r) for r in records))
            report = validator.audit_round_artifacts(run, 2)
            self.assertFalse(report['passed'])
            self.assertTrue(any('epoch set' in issue for issue in report['issues']))

    def test_outer_launcher_provenance_links_and_command(self):
        for kind in ('valid', 'valid_backslash', 'valid_mixed', 'conflicting_alias', 'source', 'data', 'text_data', 'command', 'seed'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temp:
                run = self.fixture(Path(temp))
                saved = json.loads((run / 'round_artifacts/manifest.json').read_text())
                outer = {'seed': 2023,
                         'source_sha256': {'src/' + k: v for k, v in saved['provenance']['code_sha256'].items()},
                         'data_sha256': {saved['test_behaviors_file']: saved['test_behaviors_sha256']},
                         'command': ['python', 'run.py', '--seed', '2023', '--epochs', '2', '--loop_epochs', '2']}
                outer['data_sha256'].update({entry['path']: entry['sha256']
                                            for entry in saved['provenance']['text_inputs'].values()})
                if kind == 'source':
                    outer['source_sha256']['src/train_loop.py'] = 'wrong'
                elif kind == 'data':
                    outer['data_sha256'][saved['test_behaviors_file']] = 'wrong'
                elif kind == 'text_data':
                    outer['data_sha256'][saved['args']['human_news_file']] = 'wrong'
                elif kind == 'command':
                    outer['command'][5] = '5'
                elif kind == 'seed':
                    outer['seed'] = 999
                elif kind == 'valid_backslash':
                    outer['source_sha256'] = {k.replace('/', '\\'): v for k, v in outer['source_sha256'].items()}
                elif kind == 'valid_mixed':
                    value = outer['source_sha256'].pop('src/train_loop.py')
                    outer['source_sha256']['src\\train_loop.py'] = value
                elif kind == 'conflicting_alias':
                    outer['source_sha256']['src\\train_loop.py'] = 'wrong'
                (run / 'manifest.json').write_text(json.dumps(outer))
                report = validator.audit_round_artifacts(run, 2)
                self.assertEqual(report['passed'], kind in ('valid', 'valid_backslash', 'valid_mixed'), report)

    def test_cli_exit_codes_report_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = self.fixture(root)
            output = root / 'reports/pass.json'
            args = ['validate_round_artifacts.py', '--run', str(run), '--output', str(output), '--expected-rounds', '2']
            with patch.object(sys, 'argv', args):
                self.assertEqual(validator.main(), 0)
            self.assertTrue(json.loads(output.read_text())['passed'])
            with patch.object(sys, 'argv', args), self.assertRaises(FileExistsError):
                validator.main()
            args[args.index(str(output))] = str(root / 'failure.json')
            self.mutate_snapshot(run, lambda p: p.update(native_feedback_ratio=.2))
            with patch.object(sys, 'argv', args):
                self.assertEqual(validator.main(), 2)


if __name__ == '__main__':
    unittest.main()
