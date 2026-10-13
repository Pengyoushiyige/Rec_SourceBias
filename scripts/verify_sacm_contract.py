"""CPU contract gate; no full training/download/install (tiny optimizer tests allowed)."""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def source_hashes() -> dict[str, str]:
    files = [p for folder in ('src', 'scripts', 'tests') for p in (ROOT / folder).rglob('*.py')]
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}


def git_output(*args: str) -> str:
    result = subprocess.run(['git', '-C', str(ROOT), *args], capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else 'unavailable: ' + result.stderr.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New gate report directory')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    before = source_hashes()
    syntax_errors = []
    for relative in before:
        try:
            ast.parse((ROOT / relative).read_text(encoding='utf-8-sig'), filename=relative)
        except (SyntaxError, UnicodeError) as error:
            syntax_errors.append(f'{relative}: {error}')
    command = [sys.executable, '-B', '-m', 'unittest', 'discover', '-s', 'tests', '-v']
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    # CPU/offline only; never repair resources or install dependencies here.
    result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True,
                            encoding='utf-8', errors='replace', check=False)
    (args.output / 'tests.log').write_text(result.stdout + result.stderr, encoding='utf-8')
    after = source_hashes()
    passed = result.returncode == 0 and not syntax_errors and before == after
    payload = dict(schema_version=1, gate='sacm-static-local-cpu-contract-v1',
                   checked_at=datetime.now(timezone.utc).isoformat(),
                   passed=passed, python=sys.executable, python_version=sys.version,
                   git_head=git_output('rev-parse', 'HEAD'), git_status=git_output('status', '--short'),
                   command=command, tests_return_code=result.returncode,
                   test_environment={k: env[k] for k in ('PYTHONDONTWRITEBYTECODE',
                                     'CUDA_VISIBLE_DEVICES', 'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE')},
                   syntax_errors=syntax_errors, source_unchanged_during_gate=before == after,
                   source_sha256=before,
                   limits='CPU/synthetic engineering gate, not full GPU/ML efficacy or security/type audit')
    (args.output / 'gate.json').write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(dict(passed=passed, report=str(args.output.resolve()), tests_return_code=result.returncode)))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
