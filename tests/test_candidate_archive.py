"""Lossless evidence, legacy acceptance, bounded queries and passive export."""
import copy
import csv
import gzip
import io
import itertools
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from candidate_archive import ARCHIVE_NAME, FIELDS, SCHEMA, CandidateArchive, find_candidates, iter_candidate_rows
from click_models import ClickConfig, ClickPolicy
from click_observations import ClickObservations
import analyze_sacm_experiments as analysis
import audit_click_ties as ties
import test_sacm_analysis as fixtures


def compact_legacy_rows(directory, rows):
    archive = CandidateArchive(directory, 'sacm_static')
    try:
        for _, group in itertools.groupby(rows, lambda r: (r['round'], r['training_epoch'], r['event_key'])):
            event = list(group)
            first = event[0]
            diag = None
            if first['score'] != '':
                diag = {'alpha': float(first['alpha']), 'cues': [float(r['cue']) for r in event]}
                for key, field, cast in (('scores', 'score', float), ('labels', 'label', int),
                                         ('ranks', 'rank', int), ('probabilities', 'probability', float)):
                    diag[key] = [cast(r[field]) for r in event]
            archive.record(round_number=int(first['round']), training_epoch=int(first['training_epoch']),
                           event_key=first['event_key'], user_id=first['user_id'],
                           candidate_ids=[r['candidate_id'] for r in event],
                           selected=next(i for i, r in enumerate(event) if int(r['clicked'])),
                           training_counts=[int(r['training_count']) for r in event], diagnostics=diag)
    finally:
        archive.close()


class CandidateArchiveTests(unittest.TestCase):
    def test_lossless_fields_duplicate_positions_precision_and_rng(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            policy = ClickPolicy(ClickConfig(click_model='sacm_static', eta=1), 2023)
            key = (1, 0, r'E:\data\behaviors.tsv', 7, 'I1', '用户1')
            _, _, diag = policy.select([np.nextafter(1., 2.), np.nextafter(1., 2.), 1., 1.],
                                       [1, 1, 1, 1], ['N-human', 'N-human', 'N-llm', 'N-llm'], '用户1', key)
            random_state, numpy_state = random.getstate(), np.random.get_state()
            with ClickObservations(directory, 'sacm_static') as sink:
                sink.record_initial(['N-human', 'N-human', 'M-human'], '用户1', 'I1', 1, 1)
                sink.finish_epoch()
                sink.record(diag, [1, 1, 3], round_number=2, training_epoch=1, event_key=key)
                summary = sink.finish_epoch()
            self.assertEqual(random.getstate(), random_state)
            after = np.random.get_state()
            self.assertEqual(after[0], numpy_state[0])
            np.testing.assert_array_equal(after[1], numpy_state[1])
            self.assertEqual(after[2:], numpy_state[2:])
            self.assertFalse((directory / 'candidates.csv').exists())
            rows = list(iter_candidate_rows(directory))
            self.assertEqual(len(rows), 7)
            self.assertEqual(rows[0]['event_key'], json.dumps(['initial', 'I1', '用户1']))
            self.assertTrue(all(rows[0][f] == '' for f in ('score', 'rank', 'cue', 'alpha', 'probability', 'label')))
            for i, row in enumerate(rows[3:]):
                expected = dict(round='2', training_epoch='1', event_key=json.dumps(key, ensure_ascii=False, separators=(',', ':')),
                    user_id='用户1', candidate_id=diag['candidate_ids'][i], source=diag['sources'][i],
                    score=str(float(diag['scores'][i])), label=str(int(diag['labels'][i])),
                    rank=str(int(diag['ranks'][i])), cue=str(float(diag['cues'][i])), alpha=str(float(diag['alpha'])),
                    probability=str(float(diag['probabilities'][i])), clicked=str(int(i == diag['selected_index'])),
                    training_count=str(int(i == diag['selected_index']) + [1, 1, 3].count(i)),
                    in_top3=str(int(diag['ranks'][i] <= 3)), in_top5=str(int(diag['ranks'][i] <= 5)))
                self.assertEqual(row, expected)
            self.assertEqual(sum(int(r['training_count']) for r in rows[3:]), summary['training_candidates'])
            records = [json.loads(line) for line in (directory/'summaries.jsonl').read_text().splitlines()]
            self.assertEqual(analysis.audit_candidates(find_candidates(directory), records), [])

    def test_static_catalogs_and_compression_remove_repeated_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            policy = ClickPolicy(ClickConfig(click_model='pbm_probability', eta=1), 2023)
            ids = [f'N{i//2}-' + ('human' if i % 2 == 0 else 'llm') for i in range(10)]
            with ClickObservations(directory, 'pbm_probability') as sink:
                for i in range(400):
                    key = (1, 0, r'E:\Lab\Rec_SourceBias\dataset\Amazon_Beauty\train\behaviors2.tsv', i, 'I', 'U')
                    _, _, diag = policy.select(list(range(10, 0, -1)), [1]*10, ids, 'U', key)
                    sink.record(diag, [0, 0, 1, 3], round_number=2, training_epoch=1, event_key=key)
                sink.finish_epoch()
            with gzip.open(directory/ARCHIVE_NAME, 'rt', encoding='utf-8') as handle:
                records = [json.loads(line) for line in handle]
            self.assertEqual(records[0]['schema'], SCHEMA)
            self.assertEqual(sum(r.get('type') == 'user' for r in records), 1)
            self.assertEqual(sum(r.get('type') == 'candidate' for r in records), 10)
            self.assertEqual(sum(r.get('type') == 'event' for r in records), 400)
            event = next(r for r in records if r.get('type') == 'event')
            self.assertFalse({'alpha', 'cue', 'source', 'in_top3', 'in_top5'} & event.keys())
            baseline = io.StringIO(newline='')
            writer = csv.DictWriter(baseline, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(iter_candidate_rows(directory))
            self.assertLess((directory/ARCHIVE_NAME).stat().st_size, len(baseline.getvalue().encode('utf-8')) * .25)

    def test_mixed_and_fully_compact_cohorts_match_legacy_acceptance_and_ties(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture = fixtures.SACMAnalysisTests()
            folders = {c: fixture.make_run(root, c) for c in ('pbm', 'zero', 'heterogeneous')}
            before = analysis.analyze(root, 'pilot')
            self.assertTrue(before['complete'], before['issues'])
            tie_before = ties.audit_file(folders['zero']/'candidates.csv')['totals']
            for condition in ('zero', 'heterogeneous', 'pbm'):
                folder = folders[condition]
                rows = list(iter_candidate_rows(folder))
                compact_legacy_rows(folder, rows)
                normalized = [{k: str(float(v)) if k in ('score', 'cue', 'alpha', 'probability') and v != '' else v
                               for k, v in row.items()} for row in rows]
                self.assertEqual(list(iter_candidate_rows(folder)), normalized)
                (folder/'candidates.csv').unlink()  # Temporary test fixture only.
                self.assertEqual(analysis.analyze(root, 'pilot'), before)
            self.assertEqual(ties.audit_file(folders['zero']/ARCHIVE_NAME)['totals'], tie_before)
            result = subprocess.run([sys.executable, '-B', str(ROOT/'scripts/audit_click_ties.py'), str(root)],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(len(json.loads(result.stdout)['runs']), 3)

    def test_corruption_truncation_and_schema_are_not_silently_accepted(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            with ClickObservations(directory, 'pbm_probability') as sink:
                sink.record_initial(['N-human'], 'U', 'I', 1, 1)
                summary = sink.finish_epoch()
            path = directory/ARCHIVE_NAME
            original_bytes = path.read_bytes()
            with gzip.open(path, 'rt', encoding='utf-8') as handle:
                original = [json.loads(line) for line in handle]
            mutations = [lambda r: r[0].update(schema='unknown'),
                         lambda r: r[-1].update(user=-1),
                         lambda r: r[-1].update(candidates=[999]),
                         lambda r: r[-1].update(training=[[0, 1], [0, 2]]),
                         lambda r: r[-1].update(scores=[1.]),
                         lambda r: r[-1].update(type='unknown')]
            for mutate in mutations:
                records = copy.deepcopy(original)
                mutate(records)
                with gzip.open(path, 'wt', encoding='utf-8') as handle:
                    handle.writelines(json.dumps(r)+'\n' for r in records)
                with self.subTest(mutation=mutate):
                    self.assertTrue(analysis.audit_candidates(path, [summary]))
            path.write_bytes(original_bytes[:-7])
            self.assertTrue(analysis.audit_candidates(path, [summary]))
            with gzip.open(path, 'wt', encoding='utf-8') as handle:
                handle.write(json.dumps(original[0])+'\n')
            self.assertTrue(analysis.audit_candidates(path, [summary]))

    def test_existing_evidence_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            for name in ('candidates.csv', 'summaries.jsonl', ARCHIVE_NAME):
                path = directory/name
                path.write_bytes(b'original evidence')
                with self.subTest(name=name), self.assertRaises(FileExistsError):
                    ClickObservations(directory, 'rank')
                self.assertEqual(path.read_bytes(), b'original evidence')
                path.unlink()

    def test_query_defaults_to_lightweight_summary_and_limits_details(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folder = fixtures.SACMAnalysisTests().make_run(root, 'pbm')
            command = [sys.executable, '-B', str(ROOT/'scripts/query_experiment.py'), str(folder)]
            # An unreadable archive proves the default query never opens evidence.
            (folder/ARCHIVE_NAME).write_bytes(b'broken')
            result = subprocess.run(command+['--round', '2', '--limit', '1'], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(result.stdout)
            self.assertEqual(payload['view'], 'summaries')
            self.assertEqual(payload['rows'][0]['round'], 2)
            self.assertFalse(payload['truncated'])
            details = subprocess.run(command+['--view', 'candidates'], capture_output=True, text=True)
            self.assertNotEqual(details.returncode, 0)
            (folder/ARCHIVE_NAME).unlink()
            compact_legacy_rows(folder, list(iter_candidate_rows(folder)))
            details = subprocess.run(command+['--view', 'candidates', '--round', '2', '--user-id', 'user',
                                             '--candidate-id', 'item0-human', '--limit', '2'],
                                     capture_output=True, text=True)
            self.assertEqual(details.returncode, 0, details.stderr)
            payload = json.loads(details.stdout)
            self.assertTrue(payload['truncated'])
            self.assertEqual(len(payload['rows']), 2)
            self.assertTrue(all(r['candidate_id'] == 'item0-human' and r['round'] == '2' for r in payload['rows']))
            invalid = subprocess.run(command+['--user-id', 'user'], capture_output=True, text=True)
            self.assertNotEqual(invalid.returncode, 0)


if __name__ == '__main__':
    unittest.main()
