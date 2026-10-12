"""Exercise actual trainer method bodies with controlled CPU fixtures, no BERT load."""
import ast
import json
import logging
import math
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
METHODS = (("train_loop.py", "Trainer", "train_epoch"),
           ("train_loop.py", "Trainer", "train_debias_double_align_epoch"),
           ("train_human.py", "Trainer_main", "train_epoch"))


def method_from_source(filename, classname, methodname, extra=None):
    tree = ast.parse((ROOT / "src" / filename).read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == classname)
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == methodname)
    module = ast.Module(body=[ast.ClassDef(name="Harness", bases=[], keywords=[], body=[method], decorator_list=[])], type_ignores=[])
    namespace = {"logging": logging, "tqdm": lambda items: items, "json": json}
    namespace.update(extra or {})
    exec(compile(ast.fix_missing_locations(module), str(ROOT / "src" / filename), "exec"), namespace)
    return getattr(namespace["Harness"], methodname)


class EmptyLoaderTests(unittest.TestCase):
    def test_empty_epoch_is_explicit_failure(self):
        for filename, classname, methodname in METHODS:
            with self.subTest(filename=filename, method=methodname):
                method = method_from_source(filename, classname, methodname)
                with self.assertRaisesRegex(ValueError, "empty dataloader"):
                    method(SimpleNamespace(), [])


class FeedbackBridgeTests(unittest.TestCase):
    def test_three_round_two_epoch_feedback_uses_last_epoch(self):
        ratios = ((0.0, 0.0), (0.2, 0.8), (0.3, 0.7))
        generated = []
        class Dataset:
            def __init__(self, args, *vectors, epoch, training_epoch, **kwargs):
                self.new_ratio = ratios[epoch][training_epoch]
                generated.append((epoch + 1, training_epoch + 1, self.new_ratio))
            def get_dataloader(self, **kwargs):
                return ["controlled batch"]
        class Model:
            def train(self): return self
            def eval(self): return self
            def to(self, device): return self
        received_history, evaluations, model_resets = [], [], []
        def reset():
            model_resets.append(True)
            return Model()
        args = SimpleNamespace(human_news_file="hgc", llm_news_file="aigc", debias=False,
                               loop_epochs=3, epochs=2, seed=2023, behaviors_file1="split1",
                               behaviors_file2="split2", batch_size=128, num_workers=0)
        trainer = SimpleNamespace(args=args, device="cpu", model=Model(), last_ratio=0.0,
                                  click_policy=object(), get_news2vector=lambda path: {"PADDED_NEWS": "padding"},
                                  get_user2vector=lambda *a, ratio: received_history.append(ratio) or {},
                                  get_new_model=reset, train_epoch=lambda loader: 0.0,
                                  evaluates=lambda round_, ratio: evaluations.append((round_, ratio)))
        method = method_from_source("train_loop.py", "Trainer", "_run",
                                    {"BaseDataset": Dataset, "seed_torch": lambda seed: None})
        method(trainer)
        self.assertEqual(received_history, [0.0, 0.0, 0.8])
        self.assertNotEqual(received_history[2], (ratios[1][0] + ratios[1][1]) / 2)
        self.assertEqual(evaluations, [(1, 0.0), (2, 0.8), (3, 0.7)])
        self.assertEqual(len(model_resets), 3)
        self.assertEqual(len(generated), 6)


class StatsControlFlowTests(unittest.TestCase):
    """Fake numeric tensors test real loop/denominators, not Torch mathematics."""
    def test_means_first_batch_logs_and_step_counts(self):
        class Scalar(float):
            @property
            def data(self): return self
            def cpu(self): return self
            def detach(self): return self
            def float(self): return self
            def backward(self): backward_calls.append(True)
            def __add__(self, value): return Scalar(float(self) + value)
            def __radd__(self, value): return Scalar(value + float(self))
        class Tensor:
            shape = (1, 1, 2)
            def to(self, *a): return self
            def view(self, *a): return self
            def reshape(self, *a): return self
            def unsqueeze(self, *a): return self
            def squeeze(self, *a, **kw): return self
            def expand(self, *a): return self
            def __getitem__(self, key): return self
            def __setitem__(self, key, value): pass
            def __add__(self, value): return self
            def __mul__(self, value): return self
            def __rmul__(self, value): return 0.0 if value == 0 else self
            def __neg__(self): return self
        tensor = Tensor()
        fake_torch = SimpleNamespace(**{name: (lambda *a, **kw: tensor)
                                       for name in ("stack", "bmm", "bernoulli", "full", "log", "sum", "mean")})
        fake_f = SimpleNamespace(l1_loss=lambda *a: tensor, softmax=lambda *a, **kw: tensor)
        for filename, classname, methodname in METHODS:
            for count in (1, 2):
                with self.subTest(filename=filename, method=methodname, batches=count):
                    backward_calls, steps = [], []
                    args = SimpleNamespace(negative_sampling_ratio=1, word_embedding_dim=2, user_log_length=1,
                                           log_steps=1, batch_size=128, disturb_ratio=0, news_loss=0, user_loss=0)
                    batches = [{"candidate_news_id": [["x"]], "clicked_news_id": [["x"]],
                                "targets": SimpleNamespace(to=lambda *a, value=value: value), "log_mask": tensor}
                               for value in (2.0, 4.0)]
                    model = SimpleNamespace(user_encoder=lambda *a: tensor, criterion=lambda scores, target: Scalar(target))
                    trainer = SimpleNamespace(args=args, device="cpu", model=model, text2vector={"x": tensor},
                                              rewrite_text2vector={"x": tensor},
                                              optimizer=SimpleNamespace(zero_grad=lambda: None, step=lambda: steps.append(True)))
                    method = method_from_source(filename, classname, methodname,
                        {"torch": fake_torch, "F": fake_f, "utils": SimpleNamespace(acc=lambda *a: 0.5)})
                    with self.assertLogs(level="INFO") as logs:
                        result = float(method(trainer, batches[:count]))
                    self.assertEqual(result, 2.0 if count == 1 else 3.0)
                    self.assertEqual(len(steps), count)
                    self.assertEqual(len(backward_calls), count)
                    self.assertIn("train_loss: 2.00000, acc: 0.50000", logs.output[0])
                    if count == 2:
                        self.assertIn("train_loss: 3.00000, acc: 0.50000", logs.output[1])


class CPULossTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Delayed import lets static/feedback tests run on memory-constrained hosts.
        import torch
        cls.torch = torch

    def fixture(self):
        torch = self.torch
        parameter = torch.nn.Parameter(torch.tensor([1.0, 0.0]))
        observed_losses, steps = [], []
        def criterion(scores, targets):
            result = torch.nn.functional.cross_entropy(scores, targets)
            observed_losses.append(result.item())
            return result
        def zero_grad():
            parameter.grad = None
        def step():
            self.assertIsNotNone(parameter.grad)
            steps.append(parameter.grad.clone())
        args = SimpleNamespace(negative_sampling_ratio=1, word_embedding_dim=2, user_log_length=1,
                               log_steps=1, batch_size=128, disturb_ratio=0.0, news_loss=0.0, user_loss=0.0)
        vectors = {"p": torch.tensor([1.0, 0.0]), "n": torch.tensor([0.0, 1.0]), "h": torch.tensor([0.5, 0.5])}
        model = SimpleNamespace(user_encoder=lambda logs, mask: parameter.expand(logs.shape[0], -1), criterion=criterion)
        trainer = SimpleNamespace(args=args, device="cpu", model=model, text2vector=vectors,
                                  rewrite_text2vector=vectors, optimizer=SimpleNamespace(zero_grad=zero_grad, step=step))
        batches = [{"candidate_news_id": [["p"], ["n"]], "clicked_news_id": [["h"]],
                    "targets": torch.tensor([target]), "log_mask": torch.ones(1, 1)} for target in (0, 1)]
        return trainer, batches, observed_losses, steps

    def test_one_and_two_batches_mean_and_optimization(self):
        for filename, classname, methodname in METHODS:
            for batch_count in (1, 2):
                with self.subTest(filename=filename, method=methodname, batches=batch_count):
                    trainer, batches, losses, steps = self.fixture()
                    method = method_from_source(filename, classname, methodname,
                        {"torch": self.torch, "F": self.torch.nn.functional,
                         "utils": SimpleNamespace(acc=lambda target, score: float((score.argmax(-1) == target).float().mean()))})
                    with self.assertLogs(level="INFO") as captured:
                        result = float(method(trainer, batches[:batch_count]))
                    self.assertTrue(math.isfinite(result))
                    self.assertAlmostEqual(result, sum(losses) / batch_count, places=6)
                    self.assertEqual(len(steps), batch_count)
                    self.assertFalse(any("nan" in entry.lower() or "inf" in entry.lower().split("train_loss:")[-1]
                                         for entry in captured.output))
                    self.assertTrue(all("train_loss:" in entry for entry in captured.output))


if __name__ == "__main__":
    unittest.main()
