"""Independent denominator and read-only CLI checks for tie auditing."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'audit_click_ties.py'
spec = importlib.util.spec_from_file_location('audit_click_ties', SCRIPT)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def row(candidate, score, rank, event='scored', round_number='2'):
    return dict(round=round_number, training_epoch='1', event_key=json.dumps([event, 'U']),
                candidate_id=candidate, source=candidate.rsplit('-', 1)[1],
                score=str(score), rank=str(rank), label='1')


class TieAuditTests(unittest.TestCase):
    def test_paired_and_different_item_cross_source_ties_have_distinct_denominators(self):
        same = audit.count_event([row('X-human', 1, 1), row('X-llm', 1, 2)])
        different = audit.count_event([row('X-human', 3, 1), row('X-llm', 2, 2),
                                       row('Y-human', 2, 3), row('Y-llm', 1, 4)])
        self.assertEqual(same['exact_tied_pairs'], 1)
        self.assertEqual(different['paired_positions'], 2)
        self.assertEqual(different['exact_tied_pairs'], 0)
        self.assertEqual(different['paired_tie_impressions'], 0)
        self.assertEqual(different['cross_source_tie_impressions'], 1)
        counts = {key: same[key] + different[key] for key in same}
        rates = audit.with_ratios(counts)
        self.assertEqual(rates['paired_exact_tie_rate'], 1/3)
        self.assertEqual(rates['paired_tie_impression_rate'], .5)
        self.assertEqual(rates['cross_source_tie_impression_rate'], 1)

    def test_unscored_initialization_is_not_in_simulated_denominator(self):
        counts = audit.count_event([row('X-human', '', '', 'initial', '1')])
        self.assertEqual(counts['initial_unscored_impressions'], 1)
        self.assertEqual(counts['scored_impressions'], 0)
        self.assertEqual(counts['paired_positions'], 0)
        self.assertIsNone(audit.with_ratios(counts)['paired_exact_tie_rate'])

    def test_duplicate_positions_are_not_deduplicated(self):
        counts = audit.count_event([row('X-human', 1, 1), row('X-human', 1, 3),
                                   row('X-llm', 1, 2), row('X-llm', 1, 4)])
        self.assertEqual(counts['paired_positions'], 2)
        self.assertEqual(counts['exact_tied_pairs'], 2)
        self.assertEqual(counts['tied_pair_human_first'], 2)

    def test_invalid_pairs_scores_and_partial_scoring_fail(self):
        cases = [[row('X-human', 1, 1)],
                 [row('X-human', 'nan', 1), row('X-llm', 1, 2)],
                 [row('X-human', 'inf', 1), row('X-llm', 1, 2)],
                 [row('X-human', 1, 1), row('X-llm', '', 2)],
                 [row('X-human', 1, 1), row('X-human', 2, 2),
                  row('X-llm', 1, 3), row('X-llm', 1, 4)]]
        for rows in cases:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                audit.count_event(rows)

    def test_cli_preserves_input_hash_and_refuses_output_overwrite(self):
        with tempfile.TemporaryDirectory(prefix='sacm-tie-test-') as folder:
            source, output = Path(folder) / 'candidates.csv', Path(folder) / 'audit.json'
            rows = [row('X-human', '', '', 'initial', '1'),
                    row('X-human', 1, 1), row('X-llm', 1, 2)]
            with source.open('w', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            command = [sys.executable, '-B', str(SCRIPT), str(source), '--output', str(output)]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(output.read_text(encoding='utf-8'))
            self.assertEqual(payload['runs'][0]['sha256'], digest)
            self.assertEqual(payload['totals']['scored_impressions'], 1)
            self.assertEqual(payload['totals']['initial_unscored_impressions'], 1)
            self.assertEqual(payload['totals']['paired_exact_tie_rate'], 1)
            original_output = output.read_bytes()
            again = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(again.returncode, 0)
            self.assertIn('Output already exists', again.stderr)
            self.assertEqual(output.read_bytes(), original_output)
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), digest)

    def test_invalid_initialization_event_and_round_are_rejected(self):
        cases = [[row('X-human', '', '', 'scored', '2')],
                 [row('X-human', '', '', 'initial', '2')],
                 [row('X-human', 1, 1, 'initial', '1')],
                 [row('X-human', '', '', 'scored', '1')],
                 [row('X-human', 1, 1, 'scored', '0'), row('X-llm', 1, 2, 'scored', '0')]]
        with tempfile.TemporaryDirectory(prefix='sacm-tie-invalid-') as folder:
            path = Path(folder) / 'candidates.csv'
            for rows in cases:
                with path.open('w', newline='', encoding='utf-8') as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
                with self.subTest(rows=rows), self.assertRaises(ValueError):
                    audit.audit_file(path)


if __name__ == '__main__':
    unittest.main()
