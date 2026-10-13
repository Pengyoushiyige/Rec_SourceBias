"""Frozen r07 SACM runner. Preparation is default; GPU work requires --execute."""
import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE_CONDITIONS = ('pbm', 'zero', 'heterogeneous')
EXTENDED_CONDITIONS = ('global', 'low', 'high', 'weak_cue')
CONDITIONS = CORE_CONDITIONS + EXTENDED_CONDITIONS
EXPECTED = {
    'news_parsed_bert-base-uncased.tsv': '92d0c4984a80fc4f6ed105a05e73a8e6c63e8f827a256d17095c92939766a5e8',
    'news_gpt-3.5-turbo-0613_parsed_bert-base-uncased.tsv': 'b56d2885cbdbcbb1ac5ce13b9793ed057516a1b06991b1776fd25ddea5c40218',
}

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def git(*args):
    result = subprocess.run(['git', '-C', str(ROOT), *args], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr.strip())
    return result.stdout.strip()

def build_command(args: argparse.Namespace, out: Path) -> list[str]:
    data = ROOT / 'dataset' / 'Amazon_Beauty'
    command = [args.python, str(ROOT / 'src/run.py'), '--mode', 'loop', '--dataset', 'Amazon_Beauty',
               '--model_type', 'BERT4Rec', '--llm_model', 'gpt-3.5-turbo-0613',
               '--batch_size', '128', '--news_batch_size', '64', '--gpu', '0', '--num_workers', '0',
               '--eta', '1', '--seed', str(args.seed), '--epochs', '1' if args.stage == 'pilot' else '5',
               '--loop_epochs', '3' if args.stage == 'pilot' else '20',
               '--human_news_file', str(data / 'text' / next(iter(EXPECTED))),
               '--llm_news_file', str(data / 'text' / list(EXPECTED)[1]),
               '--test_behaviors_file', str(data / 'dev/behaviors.tsv'),
               '--click_model', 'pbm_probability' if args.condition == 'pbm' else 'sacm_static',
               '--click_diagnostics_dir', str(out),
               '--round_artifacts_dir', str(out / 'round_artifacts')]
    for filename, option in (('behaviors', '--behaviors_file'),
                             ('behaviors1', '--behaviors_file1'),
                             ('behaviors2', '--behaviors_file2')):
        command.extend([option, str(data / 'train' / (filename + '.tsv'))])
    command.extend(condition_options(args.condition))
    return command

def condition_options(condition: str) -> list[str]:
    options = []
    if condition in ('zero', 'global'):
        options += ['--sacm_alpha_mode', 'global', '--sacm_alpha_value', str(0 if condition == 'zero' else 2 / 7)]
    else:
        a, b = {'low': (1, 9), 'high': (5, 2)}.get(condition, (2, 5))
        options += ['--sacm_alpha_mode', 'user_beta', '--sacm_alpha_a', str(a), '--sacm_alpha_b', str(b)]
    h, l = ((2, 2), (2, 2)) if condition == 'weak_cue' else ((2, 5), (5, 2))
    options += ['--sacm_cue_mode', 'beta', '--sacm_hgc_cue_a', str(h[0]), '--sacm_hgc_cue_b', str(h[1]),
                '--sacm_aigc_cue_a', str(l[0]), '--sacm_aigc_cue_b', str(l[1])]
    return options

def baseline_provenance(env: dict) -> dict:
    return {
        'text_encoding_mode': 'train',
        'text_encoder_parameters': 'frozen; train-mode dropout retained',
        'news_batch_size': 64,
        'news_vector_cache': 'process-local CPU vectors keyed by absolute text-file path; reused across rounds',
        'pretrained_cache': 'Hugging Face local offline model/config/tokenizer cache',
        'cache_environment': {k: env[k] for k in ('HF_HOME', 'HF_HUB_CACHE', 'HUGGINGFACE_HUB_CACHE', 'TRANSFORMERS_CACHE') if k in env},
        'model_update': 'reset recommendation model and optimizer each round',
        'history_feedback': 'global last-epoch click ratio replaces original history sources',
        'probability_ties': 'event-keyed uniform permutation within exact cross-source ties; legacy rank unchanged',
        'round_observations': 'shared frozen text vectors once; user encoder checkpoint and native evaluation counters every round',
    }


AUXILIARY_REPAIR_FILES = frozenset({
    'scripts/validate_round_artifacts.py', 'scripts/run_sacm_experiment.py',
    'scripts/run_sacm_workflow.py', 'scripts/analyze_sacm_experiments.py',
})


def source_signature(manifest: dict) -> dict:
    """Retain raw actual hashes, allow only explicitly recorded auxiliary repairs."""
    source = dict(manifest.get('source_sha256') or {})
    aliases = {path.replace('\\', '/'): path for path in source}
    for repair in manifest.get('auxiliary_source_repairs', []):
        name = repair['path']
        if name not in AUXILIARY_REPAIR_FILES or name not in aliases:
            raise ValueError('Unapproved auxiliary repair path: ' + name)
        before, after = repair['old_sha256'], repair['new_sha256']
        if any(not isinstance(digest, str) or len(digest) != 64 or
               any(c not in '0123456789abcdef' for c in digest) for digest in (before, after)):
            raise ValueError('Invalid auxiliary repair hash')
        key = aliases[name]
        if source[key] not in (before, after):
            raise ValueError('Auxiliary source changed beyond recorded repair: ' + name)
        source[key] = before
    return source


def load_frozen_signature(path: Path) -> dict:
    frozen = json.loads(path.read_text(encoding='utf-8'))
    if frozen.get('auxiliary_source_repairs'):
        original = Path(frozen['original_cohort_file'])
        if sha256(original) != frozen['original_cohort_sha256']:
            raise ValueError('Original cohort record changed after repair')
        if json.loads(original.read_text(encoding='utf-8'))['signature'] != frozen['signature']:
            raise ValueError('Repair must preserve original frozen signature')
    return frozen


def cohort_signature(manifest: dict) -> dict:
    """Seed-independent identity used to freeze a paired cohort before training."""
    signature = {key: manifest.get(key) for key in
                 ('source_sha256', 'data_sha256', 'git_head', 'versions', 'platform', 'baseline_provenance')}
    signature['environment'] = {key: value for key, value in manifest.get('environment', {}).items()
                                if key != 'PYTHONHASHSEED'}
    signature['source_sha256'] = source_signature(manifest)
    if not all(signature.values()):
        raise ValueError('Missing frozen cohort signature fields')
    return signature

MINIMUM_COMMIT_HEADROOM = 4 * 1024 ** 3


def windows_memory_snapshot() -> dict:
    """Read system commit without importing torch or changing the page file."""
    if sys.platform != 'win32':
        return {'available': False, 'reason': 'Windows commit counters are not available on this platform'}
    import ctypes
    from ctypes import wintypes

    class PerformanceInformation(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in (
                'CommitTotal', 'CommitLimit', 'CommitPeak', 'PhysicalTotal',
                'PhysicalAvailable', 'SystemCache', 'KernelTotal', 'KernelPaged',
                'KernelNonpaged', 'PageSize')
        ] + [(name, wintypes.DWORD) for name in ('HandleCount', 'ProcessCount', 'ThreadCount')]

    try:
        info = PerformanceInformation()
        info.cb = ctypes.sizeof(info)
        api = ctypes.WinDLL('psapi', use_last_error=True)
        api.GetPerformanceInfo.argtypes = [ctypes.POINTER(PerformanceInformation), wintypes.DWORD]
        api.GetPerformanceInfo.restype = wintypes.BOOL
        if not api.GetPerformanceInfo(ctypes.byref(info), info.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return {
            'available': True,
            'observed_at': datetime.now(timezone.utc).isoformat(),
            'commit_used_bytes': info.CommitTotal * info.PageSize,
            'commit_limit_bytes': info.CommitLimit * info.PageSize,
            'commit_available_bytes': (info.CommitLimit - info.CommitTotal) * info.PageSize,
            'physical_available_bytes': info.PhysicalAvailable * info.PageSize,
            'process_count': info.ProcessCount,
        }
    except OSError as error:
        return {'available': False, 'error': str(error)}


def require_commit_headroom(snapshot: dict) -> None:
    """A conservative launch gate, not a guarantee of training memory needs."""
    available = snapshot.get('commit_available_bytes')
    if snapshot.get('available') and available is not None and available < MINIMUM_COMMIT_HEADROOM:
        raise RuntimeError(
            f'Insufficient Windows system commit headroom: {available / 1024 ** 3:.2f} GiB available; '
            'at least 4 GiB is required before the environment probe. '
            'Close unused applications and check Task Manager > Performance > Memory > Committed, '
            'then start a new pilot cohort. Free GPU VRAM is a separate resource. '
            'No training was started; batch_size=128 and news_batch_size=64 remain unchanged. '
            'This is a conservative launch threshold, not a guarantee that training will fit.'
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('pilot', 'full'), default='pilot')
    parser.add_argument('--phase', choices=('core', 'extended', 'all'), default='all')
    parser.add_argument('--condition', choices=CONDITIONS, default='heterogeneous')
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--output-root', default=None)
    parser.add_argument('--python', default=str(ROOT / '.venv/Scripts/python.exe'))
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--expected-signature', type=Path,
                        help='Frozen cohort JSON; reject source/data/environment drift before training')
    args = parser.parse_args(argv)
    if args.stage == 'pilot':
        args.phase = 'core'
    allowed = CORE_CONDITIONS if args.phase == 'core' else EXTENDED_CONDITIONS if args.phase == 'extended' else CONDITIONS
    if args.condition not in allowed:
        parser.error(f'condition {args.condition} is outside {args.stage}/{args.phase}')
    output_root = args.output_root or str(ROOT / ('results/sacm_runs' if args.execute else 'results/sacm_preflight'))
    out = Path(output_root).resolve() / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') +
         f'-{args.stage}-{args.condition}-{args.seed}-' + uuid.uuid4().hex[:8])
    out.mkdir(parents=True, exist_ok=False)
    manifest = {'schema_version': 2, 'stage': args.stage, 'phase': args.phase, 'condition': args.condition, 'seed': args.seed,
                'status': 'preflight', 'started_at': datetime.now(timezone.utc).isoformat(),
                'output_dir': str(out), 'execute': args.execute}
    def save():
        (out / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    save()
    try:
        manifest['memory_before_probe'] = windows_memory_snapshot()
        manifest['minimum_commit_headroom_bytes'] = MINIMUM_COMMIT_HEADROOM
        require_commit_headroom(manifest['memory_before_probe'])
        data = ROOT / 'dataset/Amazon_Beauty'
        paths = list((data / 'text').glob('*.tsv'))
        paths += list((data / 'train').glob('*.tsv')) + list((data / 'dev').glob('*.tsv'))
        manifest['data_sha256'] = {str(p.relative_to(ROOT)): sha256(p) for p in paths}
        for name, expected in EXPECTED.items():
            actual = sha256(data / 'text' / name)
            if actual != expected:
                raise RuntimeError(f'r07 data hash mismatch: {name}: {actual} != {expected}')
        for p in ('train/behaviors.tsv', 'train/behaviors1.tsv', 'train/behaviors2.tsv', 'dev/behaviors.tsv'):
            if not (data / p).is_file():
                raise FileNotFoundError(data / p)
        manifest['git_head'] = git('rev-parse', 'HEAD')
        manifest['git_status'] = git('status', '--porcelain')
        manifest['source_sha256'] = {str(p.relative_to(ROOT)): sha256(p) for folder in ('src', 'scripts')
                                    for p in (ROOT / folder).rglob('*.py')}
        env = os.environ.copy()
        env.update(PYTHONUNBUFFERED='1', PYTHONHASHSEED=str(args.seed), CUDA_VISIBLE_DEVICES='0',
                   PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:128', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        keys = ('PYTHONHASHSEED', 'CUDA_VISIBLE_DEVICES', 'PYTORCH_CUDA_ALLOC_CONF', 'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE')
        manifest['environment'] = {k: env[k] for k in keys}
        manifest['baseline_provenance'] = baseline_provenance(env)
        probe_code = ('import json,sys,torch,transformers; from transformers import AutoConfig,AutoTokenizer; '
                      'AutoConfig.from_pretrained("bert-base-uncased",local_files_only=True); '
                      'AutoTokenizer.from_pretrained("bert-base-uncased",local_files_only=True); '
                      'print(json.dumps(dict(python=sys.version,torch=torch.__version__,transformers=transformers.__version__,cuda=torch.version.cuda,cuda_available=torch.cuda.is_available(),offline_config_tokenizer=True)))')
        probe = subprocess.run([args.python, '-c', probe_code],
                               capture_output=True, text=True, env=env, cwd=ROOT)
        manifest['memory_after_probe'] = windows_memory_snapshot()
        manifest['environment_probe_return_code'] = probe.returncode
        if probe.returncode:
            manifest['environment_probe_stderr'] = probe.stderr
            memory_error = any(marker in probe.stderr.lower() for marker in (
                'memory allocation', 'winerror 1455', 'pagefile', 'paging file'))
            detail = ('Windows memory allocation failed during environment setup; '
                      'release system commit headroom before retrying. No training was started. '
                      if memory_error else '')
            raise RuntimeError('Environment probe failed: ' + detail + probe.stderr)
        manifest['versions'] = json.loads(probe.stdout.strip())
        if args.execute and not manifest['versions']['cuda_available']:
            raise RuntimeError('CUDA unavailable; execution cannot fall back to CPU')
        manifest['platform'] = platform.platform()
        if args.expected_signature:
            frozen = load_frozen_signature(args.expected_signature)
            manifest['auxiliary_source_repairs'] = frozen.get('auxiliary_source_repairs', [])
            manifest['expected_signature_file'] = str(args.expected_signature.resolve())
            expected = frozen['signature']
            if cohort_signature(manifest) != expected:
                raise RuntimeError('Frozen cohort signature mismatch; no training was started')
        command = build_command(args, out)
        manifest['command'] = command
        manifest['status'] = 'running' if args.execute else 'prepared'
        save()
        print(json.dumps({'output_dir': str(out), 'command': command, 'status': manifest['status']}, indent=2))
        if not args.execute:
            return 0
        with (out / 'run.log').open('x', encoding='utf-8') as log:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
            for line in process.stdout:
                print(line, end='')
                log.write(line)
                log.flush()
            code = process.wait()
        manifest['return_code'] = code
        from run_source_bias_capture import parse_log, read_log_text, strip_log_prefix
        metrics, events = parse_log(out / 'run.log')
        recalls, context = {}, {}
        for raw in read_log_text(out / 'run.log').splitlines():
            line = strip_log_prefix(raw).strip()
            if line.startswith('Epoch:'):
                context['loop_epoch'] = int(line.split(':', 1)[1])
            elif line.startswith('Test Ratio:'):
                context['test_ratio'] = float(line.split(':', 1)[1])
            elif line.startswith('Test Type:'):
                context['test_type'] = line.split(':', 1)[1].strip()
            else:
                match = re.match(r'^Recall([35]):\s*([-+0-9.eE]+)', line)
                if match:
                    key = tuple(context.get(k) for k in ('loop_epoch', 'test_ratio', 'test_type'))
                    recalls.setdefault(key, {})['Recall' + match[1]] = float(match[2])
        for row in metrics:
            key = tuple(row.get(k) for k in ('loop_epoch', 'test_ratio', 'test_type'))
            row.update(recalls.get(key, {}))
        for name, rows in (('metrics.jsonl', metrics), ('events.jsonl', events)):
            if rows:
                with (out / name).open('x', encoding='utf-8') as handle:
                    for row in rows:
                        handle.write(json.dumps(row) + '\n')
        manifest['metrics_rows'] = len(metrics)
        manifest['status'] = 'completed' if code == 0 else 'failed'
        return code
    except BaseException as exc:
        manifest['status'] = 'failed'
        manifest['error'] = repr(exc)
        raise
    finally:
        manifest['finished_at'] = datetime.now(timezone.utc).isoformat()
        save()

if __name__ == '__main__':
    raise SystemExit(main())
