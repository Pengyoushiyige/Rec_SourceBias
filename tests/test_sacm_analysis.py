"""Synthetic cohorts verify analysis contracts without executing an experiment."""
import json
import csv
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
        rows, metrics, candidates = [], [], []
        for r in range(1, rounds+1):
            for ep in range(1, epochs+1):
                epoch_candidates = []
                for event in range(4):
                    initial = r == 1
                    clicked = 0 if initial else (1 if event % 2 == 0 else 0)
                    if condition == "heterogeneous" and not initial:
                        clicked = 1 if event == 0 else 0
                    for i in range(2 if initial else 6):
                        source = "human" if initial or i % 2 == 0 else "llm"
                        epoch_candidates.append(dict(round=r, training_epoch=ep,
                            event_key=json.dumps(["initial" if initial else "simulated", str(event), "user"]),
                            user_id="user", candidate_id=f"item{i if initial else i//2}-{source}", source=source,
                            clicked=int(i == clicked), training_count=int(i == clicked) + int(i == 1-clicked),
                            score="" if initial else .5, label="" if initial else 1,
                            rank="" if initial else i+1, cue="" if initial else .5,
                            alpha="" if initial else (0.2 if condition == "heterogeneous" else 0),
                            probability="" if initial else 1/6,
                            in_top3="" if initial else int(i < 3), in_top5="" if initial else int(i < 5)))
                candidates.extend(epoch_candidates)
                counts = dict(round=r, training_epoch=ep, clicks=4, impressions=4,
                    click_llm=sum(x['clicked'] for x in epoch_candidates if x['source']=='llm'),
                    training_candidates=sum(x['training_count'] for x in epoch_candidates),
                    training_llm=sum(x['training_count'] for x in epoch_candidates if x['source']=='llm'))
                counts['training_positive_llm'] = counts['click_llm']
                for k in (3,5):
                    counts[f'exposure_count_at{k}'] = sum(int(x[f'in_top{k}'] or 0) for x in epoch_candidates)
                    counts[f'exposure_llm_at{k}'] = sum(int(x[f'in_top{k}'] or 0) for x in epoch_candidates if x['source']=='llm')
                rows.append(counts)
            for kind in ("Human", "LLM", "Human Target", "LLM Target"):
                value = 60 if kind in ("Human", "Human Target") else 40
                metrics.append(dict(loop_epoch=r, test_ratio=rows[-1]['click_llm']/rows[-1]['clicks'], test_type=kind,
                                    **{name: value for name in ("MAP3","MAP5","nDCG3","nDCG5","Recall3","Recall5")}))
        for filename, values in [("summaries.jsonl", rows), ("metrics.jsonl", metrics)]:
            (folder/filename).write_text("\n".join(json.dumps(row) for row in values), encoding="utf-8")
        with (folder/"candidates.csv").open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(candidates[0]))
            writer.writeheader()
            writer.writerows(candidates)
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
        self.assertEqual(result['primary_endpoint_by_seed'][0]['first_round'], 2)
        self.assertIsNone(result['round_summary'][0]['last_epoch_exposure_ratio_at5'])
        self.assertEqual(result['primary_endpoint_seed_summary']['independent_seeds'], 1)

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

    def cohort(self, root):
        return {condition: self.make_run(root, condition) for condition in ('pbm', 'zero', 'heterogeneous')}

    def edit_candidates(self, folder, edit):
        path = folder / 'candidates.csv'
        with path.open(newline='', encoding='utf-8') as handle:
            reader = csv.DictReader(handle)
            fields, rows = reader.fieldnames, list(reader)
        edit(rows)
        with path.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def test_repeated_candidate_positions_preserve_baseline_multiplicity(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folders = self.cohort(root)
            def repeat_pair(rows):
                for row in rows:
                    if int(row['round']) > 1:
                        row['candidate_id'] = row['candidate_id'].replace('item2-', 'item1-')
            for folder in folders.values():
                self.edit_candidates(folder, repeat_pair)
            result = analysis.analyze(root, 'pilot')
        self.assertTrue(result['complete'], result['issues'])
        self.assertEqual(result['primary_endpoint_seed_summary']['independent_seeds'], 1)

    def test_asymmetric_repeated_candidates_are_still_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folders = self.cohort(root)
            def unmatched_repeat(rows):
                for row in rows:
                    if int(row['round']) > 1 and row['source'] == 'human':
                        row['candidate_id'] = row['candidate_id'].replace('item2-', 'item1-')
            self.edit_candidates(folders['heterogeneous'], unmatched_repeat)
            result = analysis.analyze(root, 'pilot')
        self.assertFalse(result['complete'])
        self.assertTrue(any('pairs or relevance labels misaligned' in issue for issue in result['issues']))

    def test_header_only_is_incomplete_and_suppresses_aggregate(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folders = self.cohort(root)
            self.edit_candidates(folders['heterogeneous'], lambda rows: rows.clear())
            result = analysis.analyze(root, 'pilot', 'core')
        self.assertFalse(result['complete'])
        self.assertEqual(result['final_round_seed_summary'], [])
        self.assertIsNone(result['primary_endpoint_seed_summary'])

    def test_repeated_negative_sampling_is_counted_by_multiplicity(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = self.make_run(Path(temp), 'heterogeneous')
            path = folder / 'summaries.jsonl'
            records = [json.loads(line) for line in path.read_text().splitlines()]
            def repeated(rows):
                rows[8]['training_count'] = '3'
            self.edit_candidates(folder, repeated)
            records[1]['training_candidates'] += 2
            self.assertEqual(analysis.audit_candidates(folder / 'candidates.csv', records), [])
            records[1]['training_candidates'] -= 1
            self.assertTrue(any('training_candidates' in s for s in analysis.audit_candidates(folder / 'candidates.csv', records)))

    def test_zero_probability_tolerance_and_utility_normalization(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folders = self.cohort(root)
            def perturb(rows):
                rows[8]['probability'] = str(float(rows[8]['probability']) + 5e-13)
                rows[9]['probability'] = str(float(rows[9]['probability']) - 5e-13)
            self.edit_candidates(folders['zero'], perturb)
            path = folders['zero'] / 'metrics.jsonl'
            records = [json.loads(line) for line in path.read_text().splitlines()]
            records[0]['MAP3'] += 5e-5  # Percentage units -> 5e-7 normalized.
            path.write_text('\n'.join(json.dumps(r) for r in records))
            self.assertTrue(analysis.analyze(root, 'pilot')['complete'])
            records[0]['MAP3'] += .001
            path.write_text('\n'.join(json.dumps(r) for r in records))
            result = analysis.analyze(root, 'pilot')
            self.assertFalse(result['complete'])
            self.assertTrue(any('normalized units' in s for s in result['issues']))

    def test_summary_distinguishes_event_weight_and_last_epoch_feedback(self):
        records = []
        for epoch, n, click, exposure in ((1, 10, 2, 2), (2, 2, 2, 2)):
            records.append(dict(round=2, training_epoch=epoch, click_llm=click, clicks=n,
                training_llm=click, training_candidates=n, training_positive_llm=click, impressions=n,
                exposure_llm_at3=exposure, exposure_count_at3=n,
                exposure_llm_at5=exposure, exposure_count_at5=n))
        result = analysis.summarize(records)[0]
        self.assertAlmostEqual(result['click_ratio'], 1/3)
        self.assertEqual(result['feedback_ratio'], 1)
        self.assertEqual(result['last_epoch_exposure_ratio_at5'], 1)

    def test_full_phases_have_frozen_run_requirements(self):
        with tempfile.TemporaryDirectory() as temp:
            for phase, count in [('core', 15), ('extended', 25), ('all', 35)]:
                result = analysis.analyze(Path(temp), 'full', phase)
                self.assertEqual(result['expected_runs'], count)
                self.assertFalse(result['complete'])
                self.assertIsNone(result['primary_endpoint_seed_summary'])

    def test_dev_ratio_must_match_last_epoch_feedback(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folders = self.cohort(root)
            path = folders['heterogeneous'] / 'metrics.jsonl'
            records = [json.loads(line) for line in path.read_text().splitlines()]
            for row in records:
                if row['loop_epoch'] == 2:
                    row['test_ratio'] = .99
            path.write_text('\n'.join(json.dumps(r) for r in records))
            result = analysis.analyze(root, 'pilot')
        self.assertFalse(result['complete'])
        self.assertTrue(any('last epoch feedback_ratio' in s for s in result['issues']))

    def test_primary_averages_last_epoch_exposure_across_noninitial_rounds(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folders = self.cohort(root)
            folder = folders['heterogeneous']
            def rotate_ranks(rows):
                for row in rows:
                    if row['round'] == '3':
                        rank = int(row['rank']) % 6 + 1
                        row.update(rank=str(rank), in_top3=str(int(rank <= 3)), in_top5=str(int(rank <= 5)))
            self.edit_candidates(folder, rotate_ranks)
            path = folder / 'summaries.jsonl'
            records = [json.loads(line) for line in path.read_text().splitlines()]
            records[-1].update(exposure_llm_at3=8, exposure_llm_at5=12)
            path.write_text('\n'.join(json.dumps(r) for r in records))
            result = analysis.analyze(root, 'pilot')
        self.assertTrue(result['complete'], result['issues'])
        self.assertAlmostEqual(result['primary_endpoint_by_seed'][0]['paired_delta'], .1)
        self.assertAlmostEqual(result['primary_endpoint_seed_summary']['mean_paired_delta'], .1)

    def test_full_core_complete_five_seed_summary(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for seed in range(2023, 2028):
                for condition in ('pbm', 'zero', 'heterogeneous'):
                    self.make_run(root, condition, seed=seed, stage='full')
            result = analysis.analyze(root, 'full', 'core')
        self.assertTrue(result['complete'], result['issues'])
        self.assertEqual(len(result['primary_endpoint_by_seed']), 5)
        self.assertEqual(result['primary_endpoint_seed_summary']['independent_seeds'], 5)
        self.assertEqual(result['primary_endpoint_seed_summary']['sample_sd'], 0)

    def test_clicked_negative_or_zero_probability_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = self.make_run(Path(temp), 'heterogeneous')
            records = [json.loads(line) for line in (folder/'summaries.jsonl').read_text().splitlines()]
            def corrupt(rows):
                rows[9]['label'] = '0'  # Clicked LLM in first simulated impression.
            self.edit_candidates(folder, corrupt)
            issues = analysis.audit_candidates(folder/'candidates.csv', records)
        self.assertTrue(any('relevance label' in s for s in issues))

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
