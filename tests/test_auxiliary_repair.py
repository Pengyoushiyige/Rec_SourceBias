"""Recorded validation repairs may reuse training; unrecorded changes stay blocked."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_sacm_experiment as runner
import run_sacm_workflow as workflow

OLD = hashlib.sha256(b'old validation').hexdigest()
NEW = hashlib.sha256(b'fixed validation').hexdigest()
OTHER = hashlib.sha256(b'unknown modification').hexdigest()


class RepairTests(unittest.TestCase):
    def manifest(self):
        return {'source_sha256': {'scripts\\validate_round_artifacts.py': OLD, 'src\\train_loop.py': OLD},
                'data_sha256': {'data': OLD}, 'git_head': 'fixed', 'versions': {'torch': 'fixed'},
                'platform': 'test', 'baseline_provenance': {'mode': 'fixed'},
                'environment': {'PYTHONHASHSEED': '2023', 'offline': '1'}}

    def repairs(self):
        return [{'path': 'scripts/validate_round_artifacts.py', 'old_sha256': OLD, 'new_sha256': NEW}]

    def test_original_and_repaired_runs_pair_without_hiding_raw_hash(self):
        old = self.manifest()
        new = copy.deepcopy(old)
        new['source_sha256']['scripts\\validate_round_artifacts.py'] = NEW
        new['auxiliary_source_repairs'] = self.repairs()
        self.assertEqual(runner.cohort_signature(old), runner.cohort_signature(new))
        self.assertEqual(new['source_sha256']['scripts\\validate_round_artifacts.py'], NEW)
        workflow.require_signature(old, {'signature': runner.cohort_signature(old), 'auxiliary_source_repairs': self.repairs()})

    def test_unknown_auxiliary_or_training_change_still_rejected(self):
        old = self.manifest()
        frozen = {'signature': runner.cohort_signature(old), 'auxiliary_source_repairs': self.repairs()}
        changed = copy.deepcopy(old)
        changed['source_sha256']['scripts\\validate_round_artifacts.py'] = OTHER
        with self.assertRaises(ValueError):
            workflow.require_signature(changed, frozen)
        changed = copy.deepcopy(old)
        changed['source_sha256']['src\\train_loop.py'] = NEW
        with self.assertRaises(ValueError):
            workflow.require_signature(changed, frozen)
        changed['auxiliary_source_repairs'] = [{'path': 'src/train_loop.py', 'old_sha256': OLD, 'new_sha256': NEW}]
        with self.assertRaises(ValueError):
            runner.source_signature(changed)

    def test_repair_sidecar_preserves_original_and_detects_tampering(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            original = root / 'cohort.json'
            frozen = {'signature': runner.cohort_signature(self.manifest())}
            original.write_text(json.dumps(frozen), encoding='utf-8')
            digest = runner.sha256(original)
            repaired = dict(frozen, original_cohort_file=str(original), original_cohort_sha256=digest,
                            auxiliary_source_repairs=self.repairs())
            sidecar = root / 'cohort-repair.json'
            sidecar.write_text(json.dumps(repaired), encoding='utf-8')
            self.assertEqual(workflow.frozen_path(root), sidecar)
            self.assertEqual(runner.load_frozen_signature(sidecar), repaired)
            self.assertEqual(runner.sha256(original), digest)
            original.write_text('{}', encoding='utf-8')
            with self.assertRaises(ValueError):
                runner.load_frozen_signature(sidecar)


if __name__ == '__main__':
    unittest.main()
