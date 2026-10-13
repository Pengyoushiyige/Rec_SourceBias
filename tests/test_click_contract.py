"""Independent CPU-only checks of the preregistered conditional-click contract."""
import sys
import unittest
import random
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from click_models import ClickConfig, ClickPolicy


class ClickContractTests(unittest.TestCase):
    ids = ["N1-human", "N1-llm", "N2-human", "N2-llm"]

    def policy(self, mode="sacm_static", **kwargs):
        return ClickPolicy(ClickConfig(click_model=mode, eta=1, **kwargs), seed=2023)

    def test_probability_by_manual_rank_not_raw_score(self):
        _, p, d = self.policy("pbm_probability").select(
            [-1, -4, -2, -3], [1, 1, 0, 0], self.ids, "U1", "event-math")
        np.testing.assert_allclose(p, [.8, .2, 0, 0], atol=1e-15)
        np.testing.assert_array_equal(d["ranks"], [1, 4, 2, 3])

    def test_sacm_formula_independently_computed(self):
        policy = self.policy(sacm_alpha_mode="global", sacm_alpha_value=.7)
        _, p, d = policy.select([4, 3, 2, 1], [1, 1, 0, 0], self.ids, "U1", "event")
        weights = np.array([1, .5, 0, 0]) * (1 - .7 * np.asarray(d["cues"]))
        np.testing.assert_allclose(p, weights / weights.sum(), atol=1e-15)
        self.assertEqual(d["sources"], ["human", "llm", "human", "llm"])

    def test_degeneracy_is_exact_for_each_event(self):
        pbm = self.policy("pbm_probability")
        for sacm in [self.policy(sacm_alpha_mode="global", sacm_alpha_value=0),
                     self.policy(sacm_cue_mode="disabled")]:
            for event in range(40):
                a, p, _ = pbm.select([4, 3, 2, 1], [1, 1, 0, 0], self.ids, "U1", event)
                b, q, _ = sacm.select([4, 3, 2, 1], [1, 1, 0, 0], self.ids, "U1", event)
                self.assertEqual(a, b)
                np.testing.assert_array_equal(p, q)

    def test_static_state_is_order_independent(self):
        a, b = self.policy(), self.policy()
        expected_alpha = {u: a.alpha(u) for u in ["U1", "U2", "U3"]}
        expected_cue = {i: a.cue(i) for i in self.ids}
        for u in reversed(list(expected_alpha)):
            self.assertEqual(expected_alpha[u], b.alpha(u))
        for i in reversed(self.ids):
            self.assertEqual(expected_cue[i], b.cue(i))
        for event in range(3):
            a.select([4, 3, 2, 1], [1, 1, 0, 0], self.ids, "U1", event)
        self.assertEqual(a.alpha("U1"), expected_alpha["U1"])
        self.assertTrue(all(0 <= x <= 1 for x in expected_cue.values()))

    def test_no_global_rng_consumption(self):
        random.seed(19)
        np.random.seed(19)
        py_state, np_state = random.getstate(), np.random.get_state()
        self.policy().select([4, 3, 2, 1], [1, 1, 0, 0], self.ids, "U1", "event")
        self.assertEqual(random.getstate(), py_state)
        after = np.random.get_state()
        self.assertEqual(np_state[0], after[0])
        np.testing.assert_array_equal(np_state[1], after[1])
        self.assertEqual(np_state[2:], after[2:])

    def test_ties_canonical_and_permutation_equivariant(self):
        policy = self.policy("pbm_probability")
        a, p, _ = policy.select([1]*4, [1]*4, self.ids, "U1", "tie")
        perm = [3, 1, 0, 2]
        shuffled = [self.ids[i] for i in perm]
        b, q, _ = policy.select([1]*4, [1]*4, shuffled, "U1", "tie")
        self.assertEqual(self.ids[a], shuffled[b])
        np.testing.assert_array_equal(q, p[perm])

    def test_cross_source_ties_have_no_fixed_human_priority(self):
        policy = self.policy('pbm_probability')
        first_human = 0
        for event in range(200):
            _, _, diagnostic = policy.select([1, 1], [1, 1], self.ids[:2], 'U1', event)
            first_human += diagnostic['ranks'][0] == 1
        self.assertGreater(first_human, 60)
        self.assertLess(first_human, 140)
        pbm = self.policy('pbm_probability')
        zero = self.policy(sacm_alpha_mode='global', sacm_alpha_value=0)
        for event in range(20):
            a, p, _ = pbm.select([1, 1], [1, 1], self.ids[:2], 'U1', event)
            b, q, _ = zero.select([1, 1], [1, 1], self.ids[:2], 'U1', event)
            self.assertEqual(a, b)
            np.testing.assert_array_equal(p, q)

    def test_tie_resolution_does_not_consume_global_random_state(self):
        random.seed(71)
        np.random.seed(71)
        py_state, np_state = random.getstate(), np.random.get_state()
        self.policy('pbm_probability').select([1, 1], [1, 1], self.ids[:2], 'U1', 'tie-rng')
        self.assertEqual(random.getstate(), py_state)
        after = np.random.get_state()
        np.testing.assert_array_equal(np_state[1], after[1])
        self.assertEqual(np_state[2:], after[2:])

    def test_continuous_alpha_and_cue_not_binary_source_penalty(self):
        ids = self.ids[:2]
        previous = None
        for alpha in (0, .2, .4, .6):
            policy = self.policy(sacm_alpha_mode='global', sacm_alpha_value=alpha)
            policy.cues.update({ids[0]: .9, ids[1]: .1})
            _, probabilities, _ = policy.select([2, 1], [1, 1], ids, 'U1', 'continuous')
            expected = np.array([1, .5]) * (1 - alpha * np.array([.9, .1]))
            np.testing.assert_allclose(probabilities, expected / expected.sum(), atol=1e-12)
            if previous is not None:
                self.assertLess(probabilities[0], previous[0])
            previous = probabilities
        # Holding rank/alpha/cue fixed, changing source names cannot change weights.
        renamed = ['N1-llm', 'N1-human']
        other = self.policy(sacm_alpha_mode='global', sacm_alpha_value=.6)
        other.cues.update({renamed[0]: .9, renamed[1]: .1})
        _, probabilities, _ = other.select([2, 1], [1, 1], renamed, 'U1', 'continuous')
        np.testing.assert_allclose(probabilities, previous, atol=1e-12)
        for cue in (.1, .3, .5):
            equal = self.policy(sacm_alpha_mode='global', sacm_alpha_value=.6)
            equal.cues.update({ids[0]: cue, ids[1]: cue})
            _, probabilities, _ = equal.select([2, 1], [1, 1], ids, 'U1', 'same-cue')
            np.testing.assert_allclose(probabilities, [2/3, 1/3], atol=1e-12)

    def test_errors_identify_impression(self):
        for scores, labels, ids in [([float("nan"), 2], [1, 1], self.ids[:2]),
                                     ([1, 2], [0, 0], self.ids[:2]),
                                     ([1, 2], [1, 1], ["bad", "N1-llm"])]:
            with self.subTest(scores=scores, ids=ids):
                with self.assertRaisesRegex(ValueError, "broken-event"):
                    self.policy().select(scores, labels, ids, "U1", "broken-event")

    def test_zero_mass_fails_with_context(self):
        policy = self.policy(sacm_alpha_mode="global", sacm_alpha_value=1,
                             sacm_cue_mode="binary")
        with self.assertRaisesRegex(ValueError, "zero-event"):
            policy.select([1], [1], ["N1-llm"], "U1", "zero-event")

    def test_higher_cue_lowers_probability_at_equal_base_weight(self):
        policy = ClickPolicy(ClickConfig(click_model="sacm_static", eta=0,
                            sacm_alpha_mode="global", sacm_alpha_value=.5,
                            sacm_cue_mode="binary"), seed=2023)
        _, p, _ = policy.select([1, 1], [1, 1], self.ids[:2], "U1", "cue")
        np.testing.assert_allclose(p, [2/3, 1/3])

    def test_legacy_selector_matches_original_golden_rng(self):
        from Dataset import get_click_index
        scores, labels = np.array([1, 4, 2, 3]), np.array([1, 1, 0, 1])
        for eta in [-1, 1, 2]:
            policy = ClickPolicy(ClickConfig(click_model="rank", eta=eta), seed=2023)
            np.random.seed(41)
            expected = [get_click_index(scores, labels, eta) for _ in range(30)]
            np.random.seed(41)
            actual = [policy.select(scores, labels, self.ids, "U1", i)[0] for i in range(30)]
            self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
