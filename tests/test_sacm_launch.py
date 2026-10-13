"""Launch contract tests: no model loading or GPU execution."""
import contextlib
import argparse
import ast
import io
import json
import logging
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_sacm_experiment as runner


class SACMLaunchTests(unittest.TestCase):
    def parse_baseline_command(self, command):
        # Execute the real parser function without importing utils/model modules.
        source = ROOT / 'src/parameters.py'
        tree = ast.parse(source.read_text(encoding='utf-8'))
        function = next(node for node in tree.body
                        if isinstance(node, ast.FunctionDef) and node.name == 'parse_args')
        namespace = {'argparse': argparse, 'logging': logging}
        module = ast.Module(body=[function], type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), str(source), 'exec'), namespace)
        with mock.patch.object(sys, 'argv', [command[1], *command[2:]]), \
             mock.patch.object(sys, 'path', [str(ROOT / 'src'), *sys.path]):
            return namespace['parse_args']()

    def test_generated_commands_are_accepted_by_actual_baseline_parser(self):
        for stage in ('pilot', 'full'):
            for condition in runner.CONDITIONS:
                with self.subTest(stage=stage, condition=condition):
                    args = SimpleNamespace(python='python', stage=stage, condition=condition, seed=2023)
                    command = runner.build_command(args, Path('fixture'))
                    parsed = self.parse_baseline_command(command)
                    data = ROOT / 'dataset/Amazon_Beauty/train'
                    self.assertEqual(parsed.behaviors_file, str(data / 'behaviors.tsv'))
                    self.assertEqual(parsed.behaviors_file1, str(data / 'behaviors1.tsv'))
                    self.assertEqual(parsed.behaviors_file2, str(data / 'behaviors2.tsv'))
                    self.assertEqual(parsed.batch_size, 128)
                    self.assertEqual(parsed.news_batch_size, 64)
                    self.assertEqual(parsed.eta, 1)
                    self.assertEqual(parsed.click_model, 'pbm_probability' if condition == 'pbm' else 'sacm_static')

    def test_original_misspelled_options_reproduce_exit_two(self):
        args = SimpleNamespace(python='python', stage='pilot', condition='pbm', seed=2023)
        command = runner.build_command(args, Path('fixture'))
        invalid = [option.replace('--behaviors_file1', '--behaviors1_file')
                   .replace('--behaviors_file2', '--behaviors2_file') for option in command]
        with contextlib.redirect_stderr(io.StringIO()) as errors, self.assertRaises(SystemExit) as exit_error:
            self.parse_baseline_command(invalid)
        self.assertEqual(exit_error.exception.code, 2)
        self.assertIn('unrecognized arguments: --behaviors1_file', errors.getvalue())

    def test_commit_gate_uses_commit_not_free_ram_or_vram(self):
        snapshot = {'available': True, 'commit_available_bytes': 2 * 1024 ** 3,
                    'physical_available_bytes': 8 * 1024 ** 3}
        with self.assertRaisesRegex(RuntimeError, 'system commit headroom'):
            runner.require_commit_headroom(snapshot)
        runner.require_commit_headroom({'available': True,
                                       'commit_available_bytes': runner.MINIMUM_COMMIT_HEADROOM})
        runner.require_commit_headroom({'available': False})

    def test_low_commit_records_failure_without_probe_or_training(self):
        snapshot = {'available': True, 'commit_available_bytes': 1024 ** 3}
        for execute in (False, True):
            with self.subTest(execute=execute), tempfile.TemporaryDirectory() as temp:
                args = ['--stage', 'pilot', '--condition', 'pbm', '--output-root', temp]
                if execute:
                    args += ['--execute']
                with mock.patch.object(runner, 'windows_memory_snapshot', return_value=snapshot), \
                     mock.patch.object(runner.subprocess, 'run') as process:
                    with self.assertRaisesRegex(RuntimeError, 'No training was started'):
                        runner.main(args)
                process.assert_not_called()
                manifests = list(Path(temp).glob('*/manifest.json'))
                self.assertEqual(len(manifests), 1)
                manifest = json.loads(manifests[0].read_text(encoding='utf-8'))
                self.assertEqual(manifest['status'], 'failed')
                self.assertEqual(manifest['memory_before_probe'], snapshot)
                self.assertNotIn('command', manifest)
                self.assertTrue(manifest['finished_at'])

    @unittest.skipUnless(sys.platform == 'win32', 'Windows commit counters')
    def test_windows_memory_snapshot_reads_real_counters(self):
        snapshot = runner.windows_memory_snapshot()
        self.assertTrue(snapshot['available'], snapshot)
        self.assertGreater(snapshot['commit_limit_bytes'], 0)
        self.assertEqual(snapshot['commit_available_bytes'],
                         snapshot['commit_limit_bytes'] - snapshot['commit_used_bytes'])

    def test_frozen_command_and_provenance(self):
        args = SimpleNamespace(python="python", stage="full", condition="heterogeneous", seed=2023)
        command = runner.build_command(args, Path("fixture"))
        options = dict(zip(command[2::2], command[3::2]))
        self.assertEqual(options["--epochs"], "5")
        self.assertEqual(options["--loop_epochs"], "20")
        self.assertEqual(options["--batch_size"], "128")
        self.assertEqual(options["--news_batch_size"], "64")
        self.assertEqual(options["--num_workers"], "0")
        self.assertEqual(options["--click_model"], "sacm_static")
        provenance = runner.baseline_provenance({"HF_HOME": "offline-cache", "SECRET": "hidden"})
        self.assertEqual(provenance["text_encoding_mode"], "train")
        self.assertEqual(provenance["news_batch_size"], 64)
        self.assertEqual(provenance["cache_environment"], {"HF_HOME": "offline-cache"})
        self.assertIn("last-epoch", provenance["history_feedback"])

    def test_invalid_phase_has_no_output_side_effect(self):
        for stage, phase, condition in (("pilot", "all", "global"), ("full", "core", "low"), ("full", "extended", "pbm")):
            with self.subTest(stage=stage, phase=phase, condition=condition), tempfile.TemporaryDirectory() as temp:
                destination = Path(temp) / "must-not-exist"
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exc:
                    runner.main(["--stage", stage, "--phase", phase, "--condition", condition, "--output-root", str(destination)])
                self.assertEqual(exc.exception.code, 2)
                self.assertFalse(destination.exists())

    def matrix(self, stage, phase):
        shell = shutil.which("pwsh") or shutil.which("powershell")
        if not shell:
            self.skipTest("PowerShell unavailable")
        script = str(ROOT / "scripts/run_sacm_matrix.ps1").replace("'", "''")
        command = "function CaptureLaunch { ConvertTo-Json -InputObject @($args) -Compress; $global:LASTEXITCODE = 0 }; "
        command += f"& '{script}' -Stage {stage} -Phase {phase} -Python CaptureLaunch -OutputRoot fixture-cohort"
        result = subprocess.run([shell, "-NoProfile", "-Command", command], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        return [json.loads(line) for line in result.stdout.splitlines() if line.strip()]

    def test_phase_counts_and_pilot_invariance(self):
        for stage, phase, expected, conditions in (
            ("pilot", "extended", 3, runner.CORE_CONDITIONS),
            ("full", "core", 15, runner.CORE_CONDITIONS),
            ("full", "extended", 20, runner.EXTENDED_CONDITIONS),
            ("full", "all", 35, runner.CONDITIONS),
        ):
            with self.subTest(stage=stage, phase=phase):
                invocations = self.matrix(stage, phase)
                self.assertEqual(len(invocations), expected)
                pairs = set()
                for invocation in invocations:
                    options = dict(zip(invocation[1::2], invocation[2::2]))
                    pairs.add((options["--condition"], options["--seed"]))
                    self.assertIn(options["--condition"], conditions)
                    self.assertEqual(options["--phase"], "core" if stage == "pilot" else phase)
                    self.assertEqual(options["--output-root"], "fixture-cohort")
                    self.assertNotIn("--execute", invocation)
                self.assertEqual(len(pairs), expected)


if __name__ == "__main__":
    unittest.main()
