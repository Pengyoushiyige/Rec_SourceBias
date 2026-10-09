"""Synthetic cohorts verify analysis contracts without executing an experiment."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import analyze_sacm_experiments as analysis
import run_sacm_experiment as runner


class SACMAnalysisTests(unittest.TestCase):
    def make_run(self, root, condition, seed=2023, stage="pilot"):
        folder = root / f"{condition}-{seed}"
        folder.mkdir()
        manifest = dict(condition=condition, seed=seed, stage=stage, status="completed",
                        source_sha256={"src/a.py": "source"}, data_sha256={"data": "hash"},
                        git_head="commit", versions={"python": "3.10"}, platform="test",
                        environment={"PYTHONHASHSEED": str(seed), "HF_HUB_OFFLINE": "1"})
        (folder/"manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        rounds = 3 if stage == "pilot" else 20
        epochs = 1 if stage == "pilot" else 5
        rows, metrics = [], []
        for r in range(1, rounds+1):
            for ep in range(1, epochs+1):
                n = 0 if r == 1 else 1
                rows.append(dict(round=r, training_epoch=ep, click_llm=1 if condition == "heterogeneous" else 2,
                     clicks=4, training_llm=3, training_candidates=8, training_positive_llm=2,
                     impressions=4, exposure_llm_at3=2*n, exposure_count_at3=6*n,
                     exposure_llm_at5=4*n, exposure_count_at5=8*n))
            for kind in ("Human", "LLM", "Human Target", "LLM Target"):
                value = 60 if kind in ("Human", "Human Target") else 40
                metrics.append(dict(loop_epoch=r, test_ratio=.5, test_type=kind,
                                    **{name: value for name in ("MAP3","MAP5","nDCG3","nDCG5","Recall3","Recall5")}))
        for filename, values in [("summaries.jsonl", rows), ("metrics.jsonl", metrics)]:
            (folder/filename).write_text("\n".join(json.dumps(row) for row in values), encoding="utf-8")
        (folder/"candidates.csv").write_text("round,candidate_id\n", encoding="utf-8")
        return folder

    def test_complete_pilot_paired_results_and_relative_delta(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for condition in ("pbm", "zero", "heterogeneous"):
                self.make_run(root, condition)
            result = analysis.analyze(root, "pilot")
        self.assertTrue(result["complete"], result["issues"])
        self.assertTrue(all(row["independent_seeds"] == 1 for row in result["final_round_seed_summary"]))
        self.assertTrue(all(row["sample_sd"] is None for row in result["final_round_seed_summary"]))
        main = [r for r in result["final_round_seed_summary"]
                if r["condition"] == "heterogeneous" and r["metric"] == "click_ratio"]
        self.assertEqual(main[0]["mean_paired_delta"], -.25)
        self.assertTrue(result["relative_delta"])
        self.assertTrue(all(row["relative_delta"] == 40 for row in result["relative_delta"]))

    def test_missing_metric_type_excludes_run(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for condition in ("pbm", "zero", "heterogeneous"):
                folder = self.make_run(root, condition)
            path = folder / "metrics.jsonl"
            path.write_text("\n".join(path.read_text().splitlines()[:-1]), encoding="utf-8")
            result = analysis.analyze(root, "pilot")
        self.assertFalse(result["complete"])
        self.assertTrue(any("four dev metric types" in message for message in result["issues"]))
        self.assertFalse(any(row["condition"] == "heterogeneous" for row in result["final_round_seed_summary"]))

    def test_command_preserves_r07_resource_protocol(self):
        for stage, rounds, epochs in [("pilot", "3", "1"), ("full", "20", "5")]:
            command = runner.build_command(SimpleNamespace(python="python", stage=stage,
                                           seed=2023, condition="zero"), Path("output"))
            options = {command[i]: command[i+1] for i in range(2, len(command), 2)}
            self.assertEqual(options["--batch_size"], "128")
            self.assertEqual(options["--news_batch_size"], "64")
            self.assertEqual(options["--loop_epochs"], rounds)
            self.assertEqual(options["--epochs"], epochs)
            self.assertEqual(options["--click_model"], "sacm_static")
            self.assertEqual(options["--sacm_alpha_value"], "0")
            self.assertIn("dev", options["--test_behaviors_file"])

    def test_log_capture_keeps_four_types_at_each_round(self):
        from run_source_bias_capture import parse_log
        lines = []
        for round_id in (1, 2, 3):
            lines.append(f"[INFO 2026-10-09 12:00:00] Epoch: {round_id}")
            for kind in ("Human", "LLM", "Human Target", "LLM Target"):
                lines += ["Test Ratio: 0.5", f"Test Type: {kind}"]
                for name in ("MAP1", "MAP3", "MAP5", "Recall3", "Recall5", "nDCG1", "nDCG3", "nDCG5"):
                    lines.append(f"{name}: {round_id * 10}")
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "run.log"
            path.write_text("\n".join(lines), encoding="utf-8")
            metrics, _ = parse_log(path)
        self.assertEqual(len(metrics), 12)
        self.assertEqual({(r["loop_epoch"], r["test_type"]) for r in metrics},
                         {(round_id, kind) for round_id in (1,2,3)
                          for kind in ("Human", "LLM", "Human Target", "LLM Target")})
        self.assertTrue(all(r["MAP3"] == r["loop_epoch"]*10 for r in metrics))


if __name__ == "__main__":
    unittest.main()
