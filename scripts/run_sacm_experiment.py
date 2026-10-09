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
CONDITIONS = ('pbm', 'zero', 'heterogeneous', 'global', 'low', 'high', 'weak_cue')
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
               '--click_diagnostics_dir', str(out)]
    for name in ('behaviors', 'behaviors1', 'behaviors2'):
        command.extend(['--' + name + '_file', str(data / 'train' / (name + '.tsv'))])
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

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('pilot', 'full'), default='pilot')
    parser.add_argument('--condition', choices=CONDITIONS, default='heterogeneous')
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--output-root', default=None)
    parser.add_argument('--python', default=str(ROOT / '.venv/Scripts/python.exe'))
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    output_root = args.output_root or str(ROOT / ('results/sacm_runs' if args.execute else 'results/sacm_preflight'))
    out = Path(output_root).resolve() / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') +
         f'-{args.stage}-{args.condition}-{args.seed}-' + uuid.uuid4().hex[:8])
    out.mkdir(parents=True, exist_ok=False)
    manifest = {'schema_version': 1, 'stage': args.stage, 'condition': args.condition, 'seed': args.seed,
                'status': 'preflight', 'started_at': datetime.now(timezone.utc).isoformat(),
                'output_dir': str(out), 'execute': args.execute}
    def save():
        (out / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    save()
    try:
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
        probe_code = ('import json,sys,torch,transformers; from transformers import AutoConfig,AutoTokenizer; '
                      'AutoConfig.from_pretrained("bert-base-uncased",local_files_only=True); '
                      'AutoTokenizer.from_pretrained("bert-base-uncased",local_files_only=True); '
                      'print(json.dumps(dict(python=sys.version,torch=torch.__version__,transformers=transformers.__version__,cuda=torch.version.cuda,cuda_available=torch.cuda.is_available(),offline_config_tokenizer=True)))')
        probe = subprocess.run([args.python, '-c', probe_code],
                               capture_output=True, text=True, env=env, cwd=ROOT)
        if probe.returncode:
            raise RuntimeError('Environment probe failed: ' + probe.stderr)
        manifest['versions'] = json.loads(probe.stdout.strip())
        if args.execute and not manifest['versions']['cuda_available']:
            raise RuntimeError('CUDA unavailable; execution cannot fall back to CPU')
        manifest['platform'] = platform.platform()
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
