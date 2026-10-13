"""Read-only CPU validation of passive per-round model observations."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

SCHEMA = 'sacm-round-observations-v1'
ROOT = Path(__file__).resolve().parents[1]


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_tensors(mapping, label, issues):
    if not isinstance(mapping, dict) or not mapping:
        issues.append(f'{label}: missing or empty tensor dictionary')
        return
    for key, tensor in mapping.items():
        if not isinstance(tensor, torch.Tensor) or tensor.numel() == 0:
            issues.append(f'{label}/{key}: missing or empty tensor')
        elif not torch.isfinite(tensor).all().item():
            issues.append(f'{label}/{key}: non-finite tensor')


def _diagnostics(value, label, issues):
    names = ('total_impressions', 'skipped_homogeneous_labels', 'skipped_first_pair_equal',
             'evaluated_impressions', 'positive_pair_ties', 'positive_pair_human_wins',
             'positive_pair_llm_wins', 'first_score_pairs_human_wins',
             'first_score_pairs_llm_wins', 'first_score_pairs_ties')
    if not isinstance(value, dict) or any(type(value.get(k)) is not int or value[k] < 0 for k in names):
        issues.append(f'{label}: missing or invalid native_eval_diagnostics')
        return
    scored = value['total_impressions'] - value['skipped_homogeneous_labels']
    if value['total_impressions'] != sum(value[k] for k in names[1:4]):
        issues.append(f'{label}: evaluation skip/evaluated counts do not partition total')
    if sum(value[k] for k in names[7:]) != scored:
        issues.append(f'{label}: first-score pair counts do not partition scored impressions')
    if value['first_score_pairs_ties'] != value['skipped_first_pair_equal']:
        issues.append(f'{label}: first-score ties differ from skipped impressions')
    if sum(value[k] for k in names[4:7]) < scored:
        issues.append(f'{label}: positive pair count below scored impressions')


def _outer_provenance(run, manifest, issues):
    """Link passive snapshots to the launcher evidence when available."""
    path = run / 'manifest.json'
    if not path.exists():
        return  # Standalone synthetic/passive artifacts remain supported.
    outer = json.loads(path.read_text(encoding='utf-8'))
    provenance = manifest['provenance']
    source = {}
    for path_key, digest in outer.get('source_sha256', {}).items():
        normalized = path_key.replace('\\', '/')
        if normalized in source and source[normalized] != digest:
            issues.append(f'conflicting source hashes for normalized path {normalized}')
        source[normalized] = digest
    code = provenance.get('code_sha256', {})
    for name in ('train_loop.py', 'round_observations.py', 'Dataset.py', 'click_models.py'):
        if not code.get(name) or code[name] != source.get('src/' + name):
            issues.append(f'outer manifest source hash differs for src/{name}')
    data = {(ROOT / key).resolve(): value for key, value in outer.get('data_sha256', {}).items()}
    behavior = Path(manifest['test_behaviors_file']).resolve()
    if data.get(behavior) != manifest.get('test_behaviors_sha256'):
        issues.append('outer manifest behavior hash differs')
    for name, entry in provenance.get('text_inputs', {}).items():
        text_path = Path(entry['path']).resolve()
        if data.get(text_path) != entry.get('sha256') or _sha256(text_path) != entry.get('sha256'):
            issues.append(f'outer manifest or actual text hash differs for {name}')
        if Path(manifest['args'].get(name, '')).resolve() != text_path:
            issues.append(f'saved args text path differs for {name}')
    args = manifest['args']
    if outer.get('seed') != args.get('seed'):
        issues.append('outer manifest seed differs from saved args')
    command = outer.get('command')
    if not isinstance(command, list) or len(command) < 2 or (len(command) - 2) % 2:
        issues.append('outer manifest lacks supported option/value training command')
        return
    for option, raw in zip(command[2::2], command[3::2]):
        name = option.removeprefix('--')
        if not option.startswith('--') or name not in args:
            issues.append(f'outer command option absent in saved args: {option}')
            continue
        actual = args[name]
        try:
            expected = type(actual)(raw) if type(actual) in (int, float) else raw
            matches = actual == expected
            if name.endswith(('_file', '_dir')) and isinstance(actual, str):
                matches = Path(actual).resolve() == Path(raw).resolve()
            if not matches:
                issues.append(f'outer command differs from saved args: {option}')
        except (TypeError, ValueError):
            issues.append(f'invalid outer command value: {option}')


def audit_round_artifacts(run_dir, expected_rounds=20):
    """Return passed/issues/artifacts without modifying or trusting saved state."""
    run = Path(run_dir).resolve()
    directory = run / 'round_artifacts'
    issues = []
    result = {'passed': False, 'issues': issues, 'run_dir': str(run),
              'artifacts': {'expected_rounds': expected_rounds, 'rounds_found': 0,
                            'rounds_checked': 0, 'bytes': 0}}
    try:
        if type(expected_rounds) is not int or expected_rounds < 1:
            raise ValueError('expected_rounds must be a positive integer')
        files = list(directory.glob('round-*.pt'))
        result['artifacts']['rounds_found'] = len(files)
        result['artifacts']['bytes'] = sum(p.stat().st_size for p in directory.rglob('*') if p.is_file())
        expected = {f'round-{n:03d}.pt' for n in range(1, expected_rounds + 1)}
        found = {p.name for p in files}
        if found != expected:
            issues.append(f'round file set mismatch: missing={sorted(expected-found)}, unexpected={sorted(found-expected)}')
        manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
        if manifest.get('schema') != SCHEMA:
            issues.append('manifest schema mismatch')
        for field in ('args', 'provenance'):
            if not isinstance(manifest.get(field), dict) or not manifest[field]:
                issues.append(f'manifest missing {field}')
        vector_name = manifest.get('vectors_file')
        if vector_name != 'text_vectors.pt':
            raise ValueError('manifest vectors_file must be text_vectors.pt')
        vectors_path = directory / vector_name
        if _sha256(vectors_path) != manifest.get('vectors_sha256'):
            issues.append('text vectors sha256 mismatch')
        vectors = torch.load(vectors_path, map_location='cpu', weights_only=True)
        for source in ('human', 'llm'):
            _finite_tensors(vectors.get(source), f'vectors/{source}', issues)
        behavior = Path(manifest['test_behaviors_file'])
        if not behavior.is_absolute():
            issues.append('behavior path must be absolute')
        if _sha256(behavior) != manifest.get('test_behaviors_sha256'):
            issues.append('behavior sha256 mismatch')
        if Path(manifest['args']['test_behaviors_file']).resolve() != behavior.resolve():
            issues.append('behavior path differs from saved args')
        _outer_provenance(run, manifest, issues)
        records = [json.loads(line) for line in (run / 'summaries.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
        last = {}
        seen = set()
        for record in records:
            r, e = record['round'], record['training_epoch']
            if type(r) is not int or type(e) is not int or e < 1:
                raise ValueError('invalid summary round/epoch')
            if (r, e) in seen:
                issues.append(f'duplicate summary round/epoch: {(r, e)}')
            seen.add((r, e))
            if r not in last or e > last[r]['training_epoch']:
                last[r] = record
        if set(last) != set(range(1, expected_rounds + 1)):
            issues.append('summary round set mismatch')
        expected_epochs = manifest['args'].get('epochs')
        if expected_epochs is not None:
            if type(expected_epochs) is not int or expected_epochs < 1:
                issues.append('saved epochs must be a positive integer')
            elif seen != {(r, e) for r in range(1, expected_rounds + 1)
                          for e in range(1, expected_epochs + 1)}:
                issues.append('summary round/epoch set differs from saved epochs')
        if manifest['args'].get('loop_epochs', expected_rounds) != expected_rounds:
            issues.append('saved loop_epochs differs from expected_rounds')
        for path in sorted(files):
            try:
                snapshot = torch.load(path, map_location='cpu', weights_only=True)
                number = int(path.stem.split('-')[1])
                if type(snapshot.get('round')) is not int or snapshot['round'] != number:
                    issues.append(f'{path.name}: snapshot round mismatch')
                for field in ('schema', 'args', 'provenance', 'vectors_sha256', 'test_behaviors_file', 'test_behaviors_sha256'):
                    if field not in snapshot or snapshot[field] != manifest.get(field):
                        issues.append(f'{path.name}: {field} differs from manifest')
                _finite_tensors(snapshot.get('user_encoder_state_dict'), path.name, issues)
                _diagnostics(snapshot.get('native_eval_diagnostics'), path.name, issues)
                ratio = snapshot.get('native_feedback_ratio')
                if isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or not math.isfinite(ratio) or not 0 <= ratio <= 1:
                    issues.append(f'{path.name}: invalid feedback ratio')
                elif number in last:
                    row = last[number]
                    clicks, llm = row['clicks'], row['click_llm']
                    if type(clicks) is not int or type(llm) is not int or clicks <= 0 or not 0 <= llm <= clicks:
                        issues.append(f'{path.name}: invalid final epoch click counts')
                    elif abs(ratio - llm / clicks) > 1e-12:
                        issues.append(f'{path.name}: feedback ratio differs from final epoch')
                result['artifacts']['rounds_checked'] += 1
            except Exception as error:
                issues.append(f'{path.name}: {type(error).__name__}: {error}')
    except Exception as error:
        issues.append(f'{type(error).__name__}: {error}')
    result['passed'] = not issues
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--output', required=True, help='New JSON report path; never overwritten')
    parser.add_argument('--expected-rounds', type=int, default=20)
    args = parser.parse_args()
    report = audit_round_artifacts(args.run, args.expected_rounds)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({'passed': report['passed'], 'issues': report['issues'], 'report': str(output)}))
    return 0 if report['passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
