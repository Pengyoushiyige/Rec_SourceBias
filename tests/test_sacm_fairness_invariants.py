"""Independent fairness invariants; no changes to simulator semantics."""
import itertools
import random
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from click_models import ClickConfig, ClickPolicy


class FairnessInvariants(unittest.TestCase):
    def policy(self, mode='sacm_static', seed=2023, eta=1, **kwargs):
        return ClickPolicy(ClickConfig(click_model=mode, eta=eta, **kwargs), seed)

    def test_ai_low_cue_beats_human_high_cue_at_equal_base_weight(self):
        policy = self.policy(eta=0, sacm_alpha_mode='global', sacm_alpha_value=.5)
        policy.cues.update({'X-human': .9, 'X-llm': .1})
        _, p, _ = policy.select([1, 1], [1, 1], ['X-human', 'X-llm'], 'U', 'equal-base')
        np.testing.assert_allclose(p, [11/30, 19/30], atol=1e-15)
        self.assertGreater(p[1], p[0])

    def test_ai_can_win_despite_worse_rank_when_discount_dominates(self):
        policy = self.policy(sacm_alpha_mode='global', sacm_alpha_value=.9)
        policy.cues.update({'X-human': .9, 'X-llm': .1})
        _, p, d = policy.select([2, 1], [1, 1], ['X-human', 'X-llm'], 'U', 'inverse-cue')
        self.assertEqual(d['ranks'], [1, 2])
        np.testing.assert_allclose(p, np.array([.19, .455]) / .645, atol=1e-15)
        self.assertGreater(p[1], p[0])

    def test_all_input_permutations_share_paired_ranks_probabilities_and_click(self):
        ids = ['X-human', 'X-llm', 'Y-human', 'Y-llm']
        policies = [self.policy('pbm_probability'),
                    self.policy(sacm_alpha_mode='global', sacm_alpha_value=0),
                    self.policy(sacm_cue_mode='disabled')]
        selected, expected, diagnostic = policies[0].select([1]*4, [1]*4, ids, 'U', 'tie')
        for perm in itertools.permutations(range(4)):
            shuffled = [ids[i] for i in perm]
            for policy in policies:
                index, p, d = policy.select([1]*4, [1]*4, shuffled, 'U', 'tie')
                self.assertEqual(shuffled[index], ids[selected])
                np.testing.assert_array_equal(p, expected[list(perm)])
                self.assertEqual(d['ranks'], [diagnostic['ranks'][i] for i in perm])

    def test_tie_priority_varies_across_seeds_without_fixed_source_winner(self):
        winners = []
        for seed in range(200):
            _, _, d = self.policy('pbm_probability', seed=seed).select(
                [1, 1], [1, 1], ['X-human', 'X-llm'], 'U', 'fixed-event')
            winners.append(d['ranks'][0] == 1)
        self.assertGreater(sum(winners), 60)
        self.assertLess(sum(winners), 140)
        self.assertIn(True, winners)
        self.assertIn(False, winners)

    def test_namespaces_and_global_random_states_are_independent(self):
        import torch
        random.seed(19)
        np.random.seed(19)
        torch.manual_seed(19)
        py_state, np_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
        fresh, exercised = self.policy(), self.policy()
        ids = ['X-human', 'X-llm']
        expected = fresh.select([1, 1], [1, 1], ids, 'U', 'event')
        for key in range(20):
            exercised.alpha('other-' + str(key))
            exercised.cue(str(key) + '-human')
            exercised._rng('tie', key).random(20)
            exercised._rng('click', key).random(20)
        actual = exercised.select([1, 1], [1, 1], ids, 'U', 'event')
        self.assertEqual(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])
        self.assertEqual(actual[2], expected[2])
        self.assertEqual(random.getstate(), py_state)
        after = np.random.get_state()
        self.assertEqual(np_state[0], after[0])
        np.testing.assert_array_equal(np_state[1], after[1])
        self.assertEqual(np_state[2:], after[2:])
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))

    def test_duplicate_occurrences_preserve_probability_mass_and_selected_identity(self):
        ids = ['X-human', 'X-human', 'X-llm', 'Y-llm']
        policy = self.policy('pbm_probability')
        selected, p, _ = policy.select([1]*4, [1]*4, ids, 'U', 'duplicates')
        expected = {name: sum(p[i] for i, item in enumerate(ids) if item == name) for name in set(ids)}
        for perm in itertools.permutations(range(4)):
            shuffled = [ids[i] for i in perm]
            index, q, d = policy.select([1]*4, [1]*4, shuffled, 'U', 'duplicates')
            self.assertEqual(shuffled[index], ids[selected])
            self.assertEqual(sorted(d['ranks']), [1, 2, 3, 4])
            self.assertEqual(len(q), 4)
            mass = {name: sum(q[i] for i, item in enumerate(shuffled) if item == name) for name in set(ids)}
            self.assertEqual(mass, expected)
            self.assertAlmostEqual(float(q.sum()), 1)


if __name__ == '__main__':
    unittest.main()
