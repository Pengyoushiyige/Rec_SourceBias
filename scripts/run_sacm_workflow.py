"""Staged full-cohort handoff. Default is read-only planning; --execute opts in."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

import analyze_sacm_experiments as analysis
import run_sacm_experiment as runner

ROOT = runner.ROOT
SEEDS = tuple(range(2023, 2028))


def stamp() -> str:
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:6]


def read_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_new(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)


def cells(phase: str) -> list[tuple[str, int]]:
    conditions = runner.CORE_CONDITIONS if phase == 'core' else runner.CONDITIONS
    return [(condition, seed) for seed in SEEDS for condition in conditions]


def completed_runs(cohort: Path) -> dict[tuple[str, int], Path]:
    """Failed attempts remain on disk, but cannot substitute for accepted cells."""
    found = {}
    for path in sorted((cohort / 'runs').glob('*/manifest.json')):
        manifest = read_json(path)
        if manifest.get('status') != 'completed':
            continue
        key = (manifest.get('condition'), manifest.get('seed'))
        if manifest.get('stage') != 'full' or key not in cells('all'):
            raise ValueError('Unexpected completed run: %s' % path)
        if key in found:
            raise ValueError('Duplicate completed cell %s; do not silently select by outcome' % (key,))
        found[key] = path.parent
    return found


def require_signature(manifest: dict, frozen: dict) -> None:
    evidence = dict(manifest)
    evidence['auxiliary_source_repairs'] = frozen.get('auxiliary_source_repairs', [])
    if runner.cohort_signature(evidence) != frozen['signature']:
        raise ValueError('Frozen source/data/environment signature mismatch')


def frozen_path(cohort: Path) -> Path:
    repair = cohort / 'cohort-repair.json'
    return repair if repair.is_file() else cohort / 'cohort.json'


def audit_cell(path: Path, frozen: dict) -> dict:
    from validate_round_artifacts import audit_round_artifacts
    manifest = read_json(path / 'manifest.json')
    require_signature(manifest, frozen)
    if manifest.get('return_code') != 0:
        raise ValueError('Completed run must have return_code=0: %s' % path)
    key = (manifest['condition'], manifest['seed'])
    args = argparse.Namespace(python=manifest['command'][0], stage='full', condition=key[0], seed=key[1])
    if manifest['command'] != runner.build_command(args, path):
        raise ValueError('Frozen training command mismatch: %s' % path)
    local = analysis.analyze(path.parent, 'full', 'all', run_paths=[path], required_cells={key})
    if not local['complete']:
        raise ValueError('Training telemetry rejected: ' + '; '.join(local['issues']))
    artifacts = audit_round_artifacts(path, expected_rounds=20)
    if not artifacts['passed']:
        raise ValueError('Saved rounds rejected: ' + '; '.join(artifacts['issues']))
    return {'run': str(path.resolve()), 'cell': list(key), 'training': local, 'artifacts': artifacts}


def invoke(command: list[str], log_path: Path | None = None) -> None:
    if log_path is None:
        result = subprocess.run(command, cwd=ROOT, check=False)
    else:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open('x', encoding='utf-8') as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError('Command failed (exit %s); %s' % (result.returncode, log_path or command[1]))


def prepare(cohort: Path, python: str, frozen_path: Path | None = None) -> dict:
    destination = cohort / 'metadata' / 'preflight' / stamp()
    command = [python, '-B', str(ROOT / 'scripts/run_sacm_experiment.py'), '--stage', 'full',
               '--phase', 'core', '--condition', 'pbm', '--seed', '2023',
               '--python', python, '--output-root', str(destination)]
    if frozen_path:
        command += ['--expected-signature', str(frozen_path)]
    invoke(command)
    paths = list(destination.glob('*/manifest.json'))
    if len(paths) != 1 or read_json(paths[0]).get('status') != 'prepared':
        raise ValueError('Preflight did not produce exactly one prepared run')
    return read_json(paths[0])


def launch_cell(cohort: Path, python: str, condition: str, seed: int, frozen: dict) -> Path:
    invoke([python, '-B', str(ROOT / 'scripts/run_sacm_experiment.py'), '--stage', 'full',
            '--phase', 'core' if condition in runner.CORE_CONDITIONS else 'extended',
            '--condition', condition, '--seed', str(seed), '--python', python,
            '--output-root', str(cohort / 'runs'), '--expected-signature', str(frozen_path(cohort)), '--execute'])
    path = completed_runs(cohort).get((condition, seed))
    if path is None:
        raise ValueError('Successful runner did not leave a completed cell')
    report = audit_cell(path, frozen)
    write_new(cohort / 'metadata' / 'acceptance' / (path.name + '-' + stamp() + '.json'), report)
    return path


def matrix_acceptance(cohort: Path, python: str, phase: str, frozen: dict) -> Path:
    completed = completed_runs(cohort)
    needed = cells(phase)
    missing = [key for key in needed if key not in completed]
    if missing:
        raise ValueError('Missing completed cells: %s' % missing)
    for key in needed:
        audit_cell(completed[key], frozen)
    output = cohort / 'analysis' / (phase + '-' + stamp())
    selection = cohort / 'metadata' / ('selected-' + phase + '-' + stamp() + '.json')
    write_new(selection, [str(completed[key]) for key in needed])
    invoke([python, '-B', str(ROOT / 'scripts/analyze_sacm_experiments.py'), str(cohort / 'runs'),
            '--stage', 'full', '--phase', phase, '--selected-runs', str(selection), '--output', str(output)],
           cohort / 'metadata' / ('analysis-' + stamp() + '.log'))
    result = read_json(output / 'analysis.json')
    if not result['complete'] or result['acceptance_scope'] != 'full-matrix':
        raise ValueError('Full matrix acceptance failed')
    write_new(cohort / 'metadata' / ('accepted-' + phase + '-' + stamp() + '.json'),
              {'phase': phase, 'analysis': str(output), 'selected_runs': str(selection)})
    print('Full matrix accepted: %s' % output)
    return output


def calibrate_first(cohort: Path, path: Path) -> dict:
    manifest = read_json(path / 'manifest.json')
    seconds = (datetime.fromisoformat(manifest['finished_at']) -
               datetime.fromisoformat(manifest['started_at'])).total_seconds()
    size = sum(p.stat().st_size for p in path.rglob('*') if p.is_file())
    return {'run': str(path), 'measured_seconds_including_setup': seconds, 'measured_bytes': size,
            'remaining_core_runs': 14, 'rough_remaining_core_hours': seconds * 14 / 3600,
            'rough_remaining_core_bytes': size * 14,
            'limits': 'Single PBM timing/size projection; SACM, resource pressure and analysis may differ.'}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('step', choices=['first', 'core', 'extended', 'reevaluate', 'status'])
    parser.add_argument('--cohort', type=Path, required=True)
    parser.add_argument('--python', default=str(ROOT / '.venv/Scripts/python.exe'))
    parser.add_argument('--phase', choices=['core', 'all'], default='core', help='Reevaluation population')
    parser.add_argument('--execute', action='store_true')
    options = parser.parse_args(argv)
    cohort = options.cohort.resolve()
    found = completed_runs(cohort)
    if not options.execute or options.step == 'status':
        print(json.dumps({'cohort': str(cohort), 'step': options.step, 'training_started': False,
                          'completed_cells_not_yet_revalidated': [list(key) for key in found],
                          'missing_core': [list(k) for k in cells('core') if k not in found],
                          'missing_all': [list(k) for k in cells('all') if k not in found],
                          'next_action': 'Run first with --execute in a visible terminal; then core, extended and reevaluate.'}, indent=2))
        return 0
    frozen_record_path = frozen_path(cohort)
    # Long workloads are intentionally never started by an import or a default invocation.
    if options.step in ('first', 'core', 'extended'):
        ancestor = cohort
        while not ancestor.exists():
            ancestor = ancestor.parent
        minimum = (40 if options.step == 'extended' else 30) * 1024 ** 3
        if shutil.disk_usage(ancestor).free < minimum:
            raise RuntimeError('Insufficient disk space: reserve %s GiB for this stage' % (minimum // 1024 ** 3))
    if options.step == 'first' and not frozen_record_path.exists():
        if found:
            raise ValueError('Cannot freeze a cohort containing completed runs')
        gate = cohort / 'metadata' / 'gates' / stamp()
        invoke([options.python, '-B', str(ROOT / 'scripts/verify_sacm_contract.py'), '--output', str(gate)])
        prepared = prepare(cohort, options.python)
        write_new(frozen_record_path, {'schema': 'sacm-staged-full-v1', 'created_at': datetime.now(timezone.utc).isoformat(),
                               'signature': runner.cohort_signature(prepared), 'gate': str(gate),
                               'seeds': list(SEEDS), 'rounds': 20, 'epochs': 5,
                               'core_conditions': list(runner.CORE_CONDITIONS),
                               'extended_conditions': list(runner.EXTENDED_CONDITIONS)})
    if not frozen_record_path.exists():
        raise ValueError('Run first to create the frozen cohort before this step')
    frozen = runner.load_frozen_signature(frozen_record_path)
    require_signature(prepare(cohort, options.python, frozen_record_path), frozen)
    if options.step == 'first':
        path = found.get(('pbm', 2023))
        if path:
            audit_cell(path, frozen)
            print('Reusing accepted PBM seed2023; no training repeated.')
        else:
            path = launch_cell(cohort, options.python, 'pbm', 2023, frozen)
        calibration = calibrate_first(cohort, path)
        write_new(cohort / 'metadata' / ('first-calibration-' + stamp() + '.json'), calibration)
        print(json.dumps(calibration, indent=2))
        print('First run accepted. Stop here; next visible-terminal step is core.')
        return 0
    first = found.get(('pbm', 2023))
    if first is None:
        raise ValueError('PBM seed2023 first run is required before continuing')
    audit_cell(first, frozen)
    if options.step == 'extended':
        matrix_acceptance(cohort, options.python, 'core', frozen)
    if options.step in ('core', 'extended'):
        queue = cells('core') if options.step == 'core' else [k for k in cells('all') if k[0] in runner.EXTENDED_CONDITIONS]
        for condition, seed in queue:
            key = (condition, seed)
            if key in found:
                audit_cell(found[key], frozen)
                print('Reusing validated cell: %s / %s' % key)
            else:
                found[key] = launch_cell(cohort, options.python, condition, seed, frozen)
            # Immediately stop on a failed zero-alpha control, before running its SACM pair.
            if condition == 'zero':
                paired = {('pbm', seed), ('zero', seed)}
                checked = analysis.analyze(cohort / 'runs', 'full', 'core',
                                          run_paths=[found[k] for k in paired], required_cells=paired)
                if not checked['complete']:
                    raise ValueError('Zero-alpha acceptance failed: ' + '; '.join(checked['issues']))
        matrix_acceptance(cohort, options.python, 'core' if options.step == 'core' else 'all', frozen)
        return 0
    if options.step == 'reevaluate':
        native_output = matrix_acceptance(cohort, options.python, options.phase, frozen)
        from analyze_saved_cohort import analyze_saved_cohort, write_report
        paths = [found[key] for key in cells(options.phase)]
        for path in paths:
            output = cohort / 'reevaluations' / path.name
            if not output.exists():
                invoke([options.python, '-B', str(ROOT / 'scripts/evaluate_saved_rounds.py'),
                        '--artifacts', str(path / 'round_artifacts'), '--output', str(output),
                        '--ratios', 'native,0.5,0,1', '--device', 'cpu'])
        result = analyze_saved_cohort(paths, cohort / 'reevaluations', options.phase)
        output = cohort / 'analysis' / ('controlled-' + options.phase + '-' + stamp())
        write_report(output, result)
        if not result['complete']:
            raise ValueError('Controlled reevaluation failed: ' + '; '.join(result['issues']))
        from report_sacm_full import write_full_report
        report_output = cohort / 'analysis' / ('report-' + options.phase + '-' + stamp())
        write_full_report(report_output, read_json(native_output / 'analysis.json'), result,
                          native_output, output)
        print('Controlled evaluation accepted: %s' % output)
        print('Chinese report: %s' % (report_output / 'report.zh-CN.md'))
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print('Stopped: %s' % error, file=sys.stderr)
        raise SystemExit(2)
