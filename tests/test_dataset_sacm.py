"""Two-round micro integration, without pretrained weights or GPU execution."""
import contextlib
import csv
import io
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from Dataset import BaseDataset
from click_models import ClickConfig, ClickPolicy
from click_observations import ClickObservations
from metrics import fast_recall_score


class DatasetSACMTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.tsv = self.root / "behaviors.tsv"
        self.tsv.write_text("I1\tU1\t0\tN1\tN1-1 N2-0\n"
                            "I2\tU2\t0\tN2\tN2-1 N1-0\n", encoding="utf-8")
        self.human = {"N1": torch.tensor([1., 0.]), "N2": torch.tensor([0., 1.])}
        self.llm = {"N1": torch.tensor([.5, .5]), "N2": torch.tensor([.3, .7])}
        self.users = {"N1": torch.tensor([1., 0.]), "N2": torch.tensor([0., 1.])}
        self.args = SimpleNamespace(eta=1, seed=2023, negative_sampling_ratio=2, user_log_length=2)

    def tearDown(self):
        self.temp.cleanup()

    def dataset(self, epoch, policy, sink=None):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return BaseDataset(self.args, str(self.tsv), self.human, self.llm,
                               self.users, epoch, click_policy=policy, observation_sink=sink)

    def test_two_round_data_reaches_optimizer_and_streaming_counts(self):
        policy = ClickPolicy(ClickConfig(click_model="sacm_static", eta=1), 2023)
        with ClickObservations(self.root / "obs", "sacm_static") as sink:
            initial = self.dataset(0, policy, sink)
            first = sink.finish_epoch()
            self.assertEqual(first["simulated_impressions"], 0)
            self.assertEqual(first["click_ratio"], 0)
            self.assertIsNone(first["exposure_ratio_at3"])
            for row in initial.behaviors_parsed.itertuples():
                self.assertTrue(all(n.endswith("-human") for n in row.candidate_news.split()))
            simulated = self.dataset(1, policy, sink)
            second = sink.finish_epoch()
            self.assertEqual(second["round"], 2)
            self.assertEqual(second["simulated_impressions"], 2)
            self.assertEqual(second["training_candidates"], 6)
            self.assertEqual(second["exposure_count_at3"], 6)
            self.assertEqual(second["exposure_count_at5"], 8)
            batch = next(iter(simulated.get_dataloader(2, False, 0)))
            self.assertEqual(batch["targets"].tolist(), [0, 0])
            self.assertEqual(tuple(batch["log_mask"].shape), (2, 2))
            vectors = {**{i+"-human": v for i, v in self.human.items()},
                       **{i+"-llm": v for i, v in self.llm.items()}}
            candidates = torch.stack([torch.stack([vectors[i] for i in row])
                                      for row in batch["candidate_news_id"]])
            learnable_user = torch.nn.Parameter(torch.tensor([[.1, .2], [.2, .1]]))
            previous = learnable_user.detach().clone()
            optimizer = torch.optim.SGD([learnable_user], lr=.1)
            loss = torch.nn.functional.cross_entropy(
                torch.bmm(candidates, learnable_user.unsqueeze(-1)).squeeze(-1), batch["targets"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            self.assertTrue(torch.isfinite(loss))
            self.assertFalse(torch.equal(previous, learnable_user))
        with (self.root / "obs" / "candidates.csv").open(encoding="utf-8", newline="") as handle:
            all_rows = list(csv.DictReader(handle))
        initial_rows = [r for r in all_rows if r["score"] == ""]
        rows = [r for r in all_rows if r["score"] != ""]
        self.assertEqual(len(all_rows), 14)
        self.assertEqual(sum(int(r["training_count"]) for r in initial_rows), first["training_candidates"])
        self.assertEqual(sum(int(r["clicked"]) for r in initial_rows), first["clicks"])
        self.assertEqual(len(rows), 8)
        self.assertEqual(sum(int(row["clicked"]) for row in rows), 2)
        self.assertEqual(sum(int(row["in_top3"]) for row in rows), 6)
        self.assertAlmostEqual(sum(int(row["clicked"]) * (row["source"] == "llm")
                                   for row in rows) / 2, second["click_ratio"])
        self.assertEqual(sum(int(row["training_count"]) for row in rows), second["training_candidates"])
        self.assertEqual(sum(int(row["training_count"]) for row in rows if row["source"] == "llm"),
                         second["training_llm"])
        snapshots = [json.loads(line) for line in (self.root/"obs"/"summaries.jsonl").read_text().splitlines()]
        self.assertEqual(len(snapshots), 2)

    def test_paired_zero_alpha_identical_dataset_and_rng_state(self):
        outputs = []
        for config in [ClickConfig(click_model="pbm_probability", eta=1),
                       ClickConfig(click_model="sacm_static", eta=1,
                                   sacm_alpha_mode="global", sacm_alpha_value=0)]:
            random.seed(2023)
            np.random.seed(2023)
            ds = self.dataset(1, ClickPolicy(config, 2023))
            outputs.append((ds.behaviors_parsed.to_dict("records"), random.getstate(), np.random.get_state()))
        self.assertEqual(outputs[0][0], outputs[1][0])
        self.assertEqual(outputs[0][1], outputs[1][1])
        np.testing.assert_array_equal(outputs[0][2][1], outputs[1][2][1])
        self.assertEqual(outputs[0][2][2:], outputs[1][2][2:])

    def test_recall_denominator_is_all_positives(self):
        self.assertEqual(fast_recall_score([1, 0, 1, 0], [4, 3, 2, 1], k=[1, 3, 5]), [.5, 1., 1.])
        self.assertEqual(fast_recall_score([0, 0], [2, 1], k=[3, 5]), [0., 0.])

    def test_telemetry_counts_repeated_negative_samples(self):
        policy = ClickPolicy(ClickConfig(click_model="pbm_probability", eta=1), 2023)
        _, _, diag = policy.select([4, 3, 2, 1], [1, 0, 0, 0],
                                   ["N1-human", "N1-llm", "N2-human", "N2-llm"],
                                   "U1", "duplicate-negatives")
        with ClickObservations(self.root / "duplicates", "pbm_probability") as sink:
            sink.record(diag, [1, 1, 3], round_number=2, training_epoch=1,
                        event_key="duplicate-negatives")
            summary = sink.finish_epoch()
        self.assertEqual(summary["training_candidates"], 4)
        self.assertEqual(summary["training_llm"], 3)
        self.assertEqual(summary["training_ratio"], .75)
        self.assertEqual(summary["click_ratio"], 0)
        with (self.root/"duplicates"/"candidates.csv").open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([int(row["training_count"]) for row in rows], [1, 2, 0, 1])

    def test_actual_trainer_run_bridge_two_rounds(self):
        # Exercise production orchestration; replace expensive model/encoder/train kernels only.
        from train_loop import Trainer
        trainer = Trainer.__new__(Trainer)
        trainer.args = SimpleNamespace(**vars(self.args), loop_epochs=2, epochs=2,
                    human_news_file="human", llm_news_file="llm", debias=False,
                    behaviors_file1=str(self.tsv), behaviors_file2=str(self.tsv),
                    batch_size=2, num_workers=0, click_diagnostics_dir=str(self.root/"bridge"))
        trainer.device = "cpu"
        trainer.last_ratio = 0
        trainer.model = torch.nn.Linear(2, 2)
        trainer.click_policy = ClickPolicy(ClickConfig(click_model="sacm_static", eta=1), 2023)
        trainer.get_news2vector = lambda path: dict(self.human if path == "human" else self.llm,
                                                  PADDED_NEWS=torch.zeros(2))
        trainer.get_user2vector = Mock(return_value=self.users)
        trainer.get_new_model = lambda: torch.nn.Linear(2, 2)
        seen = []
        def consume(dataloader):
            batch = next(iter(dataloader))
            seen.append(batch["candidate_news_id"])
            self.assertEqual(batch["targets"].tolist(), [0, 0])
            return torch.tensor(.1)
        trainer.train_epoch = consume
        trainer.evaluates = Mock()
        shared = trainer.click_policy
        with patch("train_loop.seed_torch") as seed_spy, \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            trainer.run()
        self.assertEqual(seed_spy.call_count, 2)
        self.assertIs(trainer.click_policy, shared)
        self.assertEqual(set(shared.alphas), {"U1", "U2"})
        self.assertEqual(len(seen), 4)
        self.assertEqual(trainer.evaluates.call_count, 2)
        self.assertEqual(trainer.get_user2vector.call_args_list[1].kwargs["ratio"], 0)
        records = [json.loads(line) for line in (self.root/"bridge"/"summaries.jsonl").read_text().splitlines()]
        self.assertEqual([(r["round"], r["training_epoch"]) for r in records], [(1,1),(1,2),(2,1),(2,2)])
        self.assertEqual([r["simulated_impressions"] for r in records], [0,0,2,2])
        self.assertEqual(trainer.last_ratio, records[-1]["click_ratio"])


if __name__ == "__main__":
    unittest.main()
