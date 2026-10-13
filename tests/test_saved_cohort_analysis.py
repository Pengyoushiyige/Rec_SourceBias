"""Synthetic 15-cell acceptance of controlled histories, with no model training."""
import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import analyze_saved_cohort as controlled
import analyze_sacm_experiments as native_analysis
import report_sacm_full as full_report
import test_sacm_analysis as native_fixture


def dump(path, data):
    path.write_text(json.dumps(data), encoding='utf-8')


class ControlledCohortTests(unittest.TestCase):
    def fixture(self, root):
        behavior = root / 'behavior.tsv'
        behavior.write_text('1\tu\tt\tA\tA-1 B-0\n', encoding='utf-8')
        output = root / 'reevaluations'
        paths = []
        for seed in range(2023, 2028):
            for condition in controlled.CORE_CONDITIONS:
                path = root / ('%s-%s' % (condition, seed))
                artifacts = path / 'round_artifacts'
                artifacts.mkdir(parents=True)
                paths.append(path)
                run = dict(condition=condition, seed=seed, stage='full', status='completed',
                           source_sha256={'code': 'fixed'}, data_sha256={'input': 'fixed'}, git_head='fixed',
                           versions={'python': 'fixed'}, platform='test', baseline_provenance={'mode': 'fixed'},
                           environment={'PYTHONHASHSEED': str(seed), 'offline': '1'})
                dump(path / 'manifest.json', run)
                (artifacts / 'text_vectors.pt').write_bytes(b'same vectors')
                saved = {'vectors_file': 'text_vectors.pt', 'vectors_sha256': controlled.sha256(artifacts / 'text_vectors.pt'),
                         'test_behaviors_file': str(behavior), 'test_behaviors_sha256': controlled.sha256(behavior)}
                dump(artifacts / 'manifest.json', saved)
                folder = output / path.name
                folder.mkdir(parents=True)
                manifest = {'input_manifest': saved, 'device': 'cpu', 'namespace': 'sacm-controlled-evaluation-v1',
                            'artifacts': str(artifacts), 'evaluator_sha256': controlled.sha256(ROOT / 'scripts/evaluate_saved_rounds.py')}
                dump(folder / 'manifest.json', manifest)
                results, summaries, csv_rows = [], [], []
                for r in range(1, 21):
                    snapshot = artifacts / ('round-%03d.pt' % r)
                    snapshot.write_bytes(('checkpoint%s' % r).encode())
                    summaries.append({'round': r, 'training_epoch': 5, 'clicks': 10, 'click_llm': 5})
                    for label in controlled.RATIO_LABELS:
                        value = .52 if condition == 'heterogeneous' else .5
                        metrics = {g: {m: value for m in controlled.METRICS} for g in controlled.GROUPS}
                        record = {'round': r, 'ratio_label': label, 'history_ratio': .5 if label == 'native' else float(label),
                                  'native_feedback_ratio': .5, 'assignment_seed': seed,
                                  'checkpoint_sha256': controlled.sha256(snapshot),
                                  'counts': {'total': 4, 'all_zero': 1, 'all_one': 1, 'first_score_equal': 0, 'evaluated': 2},
                                  'history_assignment_sha256': hashlib.sha256(('%s-%s' % (seed, label)).encode()).hexdigest(),
                                  'metrics': metrics, 'exposure': {'3': .5, '5': .5}}
                        results.append(record)
                        csv_rows.extend({'round': r, 'ratio_label': label, 'metric_group': g, 'metric': m, 'value': v}
                                        for g, scores in metrics.items() for m, v in scores.items())
                (path / 'summaries.jsonl').write_text('\n'.join(json.dumps(r) for r in summaries), encoding='utf-8')
                dump(folder / 'analysis.json', results)
                with (folder / 'metrics.csv').open('w', newline='', encoding='utf-8') as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
                    writer.writeheader(); writer.writerows(csv_rows)
        return paths, output

    def test_full_population_and_seed_level_summary(self):
        with tempfile.TemporaryDirectory() as temp:
            paths, output = self.fixture(Path(temp))
            result = controlled.analyze_saved_cohort(paths, output)
            self.assertTrue(result['complete'], result['issues'])
            self.assertEqual(result['accepted_runs'], 15)
            for row in result['seed_summary']:
                self.assertEqual(row['independent_seeds'], 5)
                self.assertAlmostEqual(row['mean_paired_delta'], .02 if row['condition'] == 'heterogeneous' and row['group'] != 'dev candidates' else 0)
                self.assertEqual(row['sample_sd'], 0)
            destination = Path(temp) / 'report'
            controlled.write_report(destination, result)
            self.assertTrue((destination / 'report.zh-CN.md').is_file())
            with self.assertRaises(FileExistsError):
                controlled.write_report(destination, result)

    def test_missing_or_duplicate_cells_no_aggregate(self):
        with tempfile.TemporaryDirectory() as temp:
            paths, output = self.fixture(Path(temp))
            for selected in (paths[:-1], paths + paths[:1]):
                result = controlled.analyze_saved_cohort(selected, output)
                self.assertFalse(result['complete'])
                self.assertEqual(result['seed_summary'], [])

    def test_corruption_and_pairing_failure_scenarios(self):
        for mode in ('assignment', 'round', 'nan', 'checkpoint', 'csv', 'source', 'vectors', 'evaluated'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temp:
                paths, output = self.fixture(Path(temp))
                path = paths[-1]
                file = output / path.name / 'analysis.json'
                data = json.loads(file.read_text())
                if mode == 'assignment':
                    data[1]['history_assignment_sha256'] = '0' * 64
                elif mode == 'round':
                    data.pop()
                elif mode == 'nan':
                    data[0]['metrics']['Human']['nDCG5'] = float('nan')
                elif mode == 'checkpoint':
                    data[0]['checkpoint_sha256'] = '0' * 64
                elif mode == 'evaluated':
                    data[0]['counts']['evaluated'] = 0
                elif mode == 'csv':
                    (output / path.name / 'metrics.csv').write_text('round,ratio_label,metric_group,metric,value\n')
                elif mode == 'source':
                    manifest = json.loads((path / 'manifest.json').read_text())
                    manifest['baseline_provenance'] = {'mode': 'changed'}
                    dump(path / 'manifest.json', manifest)
                elif mode == 'vectors':
                    (path / 'round_artifacts/text_vectors.pt').write_bytes(b'changed')
                dump(file, data)
                result = controlled.analyze_saved_cohort(paths, output)
                self.assertFalse(result['complete'], mode)
                self.assertEqual(result['seed_summary'], [])

    def test_chinese_report_requires_both_gates_and_exports_seed_plot(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaises(ValueError):
                full_report.write_full_report(root / 'rejected', {'complete': False}, {'complete': True}, root, root)
            self.assertFalse((root / 'rejected').exists())
            builder = native_fixture.SACMAnalysisTests()
            for seed in range(2023, 2028):
                for condition in controlled.CORE_CONDITIONS:
                    builder.make_run(root, condition, seed, stage='full')
            native = native_analysis.analyze(root, 'full', 'core')
            self.assertTrue(native['complete'], native['issues'])
            destination = root / 'accepted'
            full_report.write_full_report(destination, native,
                                          {'complete': True, 'phase': 'core', 'expected_runs': 15, 'accepted_runs': 15}, root, root)
            report = (destination / 'report.zh-CN.md').read_text(encoding='utf-8')
            self.assertIn('2027', report)
            self.assertIn('末轮指标', report)
            self.assertTrue((destination / 'exposure5_paired_trajectories.svg').is_file())
            self.assertTrue((destination / 'exposure5_paired_trajectories.png').is_file())


if __name__ == '__main__':
    unittest.main()
