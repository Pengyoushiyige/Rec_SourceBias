import importlib.util
import json
from unittest.mock import patch
import random
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from round_observations import RoundObservations, sha256
spec = importlib.util.spec_from_file_location('offline_rounds', ROOT / 'scripts/evaluate_saved_rounds.py')
offline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(offline)


class RoundTests(unittest.TestCase):
    def test_save_once_nooverwrite_and_rng_unchanged(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            behavior = root / 'behaviors.tsv'
            behavior.write_text('1\tu\tt\tA\tA-1 B-0\n')
            args = SimpleNamespace(test_behaviors_file=str(behavior), seed=2023)
            vectors = {'A': torch.tensor([1., 2.]), 'PADDED_NEWS': torch.zeros(2)}
            model = torch.nn.Linear(2, 2)
            model.train()
            before = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
            observations = RoundObservations(root / 'artifacts', args, vectors, vectors)
            digest = sha256(root / 'artifacts/text_vectors.pt')
            observations.save_round(1, model, .4)
            observations.save_round(2, model, .5)
            after = (random.getstate(), np.random.get_state(), torch.get_rng_state())
            self.assertEqual(before[0], after[0])
            self.assertEqual(before[1][0], after[1][0])
            np.testing.assert_array_equal(before[1][1], after[1][1])
            self.assertEqual(before[1][2:], after[1][2:])
            self.assertTrue(torch.equal(before[2], after[2]))
            self.assertTrue(model.training)
            self.assertEqual(digest, sha256(root / 'artifacts/text_vectors.pt'))
            loaded = torch.load(root / 'artifacts/round-001.pt', weights_only=True)
            copy = torch.nn.Linear(2, 2)
            copy.load_state_dict(loaded['user_encoder_state_dict'])
            for key, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, copy.state_dict()[key]))
            with self.assertRaises(FileExistsError):
                observations.save_round(1, model, .3)
            with self.assertRaises(FileExistsError):
                RoundObservations(root / 'artifacts', args, vectors, vectors)

    def test_real_bert4rec_offline_cli_without_plm(self):
        import utils
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            behavior = root / 'behaviors.tsv'
            behavior.write_text('1\tu\tt\tA\tA-1 B-0\n')
            args = SimpleNamespace(test_behaviors_file=str(behavior), seed=2023,
                model_type='BERT4Rec', num_words_abstract=2, news_attributes=['abstract'],
                word_embedding_dim=768, user_log_length=2, batch_size=1)
            encoder = utils.get_model(args.model_type)(args, None).user_encoder.eval()
            human = {'A': torch.linspace(0, 1, 768), 'B': torch.linspace(1, 0, 768),
                     'PADDED_NEWS': torch.zeros(768)}
            llm = {'A': human['A'] * 2, 'B': human['B'] * 2, 'PADDED_NEWS': torch.zeros(768)}
            observations = RoundObservations(root / 'artifacts', args, human, llm)
            observations.save_round(1, encoder, 0., {'evaluated_impressions': 1})
            command = ['evaluate_saved_rounds.py', '--artifacts', str(root / 'artifacts'),
                       '--output', str(root / 'output'), '--ratios', 'native,0.5,0,1']
            with patch.object(sys, 'argv', command):
                offline.main()
            results = json.loads((root / 'output/analysis.json').read_text())
            self.assertEqual([r['history_ratio'] for r in results], [0., .5, 0., 1.])
            self.assertTrue((root / 'output/metrics.csv').exists())
            self.assertEqual(results[0]['history_assignment_sha256'], results[2]['history_assignment_sha256'])
            with patch.object(sys, 'argv', command), self.assertRaises(FileExistsError):
                offline.main()

    def test_common_assignments_stateless_nested(self):
        first = offline.history_sources(2023, 'A B C D', 3, .5)
        random.seed(888)
        np.random.seed(999)
        second = offline.history_sources(2023, 'A B C D', 3, .5)
        self.assertEqual(first, second)
        self.assertEqual([False] * 3, offline.history_sources(2023, 'A B C D', 3, 0))
        self.assertEqual([True] * 3, offline.history_sources(2023, 'A B C D', 3, 1))

    def test_evaluation_cpu_duplicates_and_skip_counts(self):
        class Encoder(torch.nn.Module):
            def forward(self, logs, masks):
                return (logs * masks.unsqueeze(-1)).sum(1)
        args = SimpleNamespace(batch_size=2, user_log_length=2)
        human = {'A': torch.tensor([1., 0.]), 'B': torch.tensor([0., 1.]), 'PADDED_NEWS': torch.zeros(2)}
        llm = {'A': torch.tensor([2., 0.]), 'B': torch.tensor([0., 2.]), 'PADDED_NEWS': torch.zeros(2)}
        rows = [(0, 'i', 'A', ['A', 'A', 'B'], np.array([1, 1, 0])),
                (1, 'j', 'A', ['B'], np.array([0])),
                (2, 'k', 'A', ['A'], np.array([1])),
                (3, 'l', ' ', ['A', 'B'], np.array([1, 0]))]
        result = offline.evaluate(Encoder(), args, {'human': human, 'llm': llm}, rows, .5, 2023, 'cpu')
        self.assertEqual(result['counts'], {'total': 4, 'all_zero': 1, 'all_one': 1, 'first_score_equal': 1, 'evaluated': 1})
        self.assertEqual(set(result['metrics']), {'Human', 'LLM', 'Human Target', 'LLM Target'})
        self.assertEqual(len(result['metrics']['Human']), 8)


if __name__ == '__main__':
    unittest.main()
