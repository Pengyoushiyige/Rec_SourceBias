"""Audit SACM run completeness and paired seed differences without pooling rounds."""
import argparse
import csv
import json
import math
import statistics
import hashlib
from collections import Counter
from itertools import zip_longest
from pathlib import Path
from run_sacm_experiment import CONDITIONS

RATIOS = {
    'click_ratio': ('click_llm', 'clicks'),
    'training_ratio': ('training_llm', 'training_candidates'),
    'training_positive_ratio': ('training_positive_llm', 'impressions'),
    'exposure_ratio_at3': ('exposure_llm_at3', 'exposure_count_at3'),
    'exposure_ratio_at5': ('exposure_llm_at5', 'exposure_count_at5'),
}

def summarize(records: list[dict]) -> list[dict]:
    rows = []
    for round_id in sorted({r['round'] for r in records}):
        group = [r for r in records if r['round'] == round_id]
        row = {'round': round_id}
        for metric, (num, den) in RATIOS.items():
            valid = [r for r in group if r.get(num) is not None and r.get(den) is not None]
            numerator = sum(r[num] for r in valid)
            denominator = sum(r[den] for r in valid)
            row[metric] = numerator / denominator if denominator else None
        last = max(group, key=lambda r: r['training_epoch'])
        row['last_training_epoch'] = last['training_epoch']
        row['feedback_ratio'] = last['click_llm'] / last['clicks'] if last['clicks'] else None
        for k in (3, 5):
            den = last[f'exposure_count_at{k}']
            row[f'last_epoch_exposure_ratio_at{k}'] = last[f'exposure_llm_at{k}'] / den if den else None
        rows.append(row)
    return rows

def relative_delta(human: float, llm: float) -> float | None:
    denominator = (human + llm) / 2
    return 100 * (human - llm) / denominator if denominator else None

def audit_candidates(path: Path, records: list[dict]) -> list[str]:
    """Recount CSV events in streaming order; keep at most one candidate event."""
    fields = ('round', 'training_epoch', 'event_key', 'user_id', 'candidate_id', 'source',
              'score', 'label', 'rank', 'cue', 'alpha', 'probability', 'clicked',
              'training_count', 'in_top3', 'in_top5')
    counts, issues, seen = {}, [], set()
    current, event = None, []
    def flush():
        if not event:
            return
        event_hash = hashlib.blake2b(repr(current).encode('utf-8'), digest_size=16).digest()
        if event_hash in seen:
            raise ValueError('duplicate/non-contiguous candidate event')
        seen.add(event_hash)
        key = current[:2]
        c = counts.setdefault(key, {name: 0 for name in
            ('impressions', 'clicks', 'click_llm', 'training_candidates', 'training_llm',
             'training_positive_llm', 'simulated_impressions', 'exposure_llm_at3',
             'exposure_count_at3', 'exposure_llm_at5', 'exposure_count_at5')})
        if sum(r['clicked'] for r in event) != 1:
            raise ValueError('event requires exactly one click')
        initial = json.loads(current[2])[0] == 'initial'
        if (key[0] == 1) != initial:
            raise ValueError('initial candidates required only in round one')
        if initial and any(r['source'] != 'human' for r in event):
            raise ValueError('initial round must preserve HGC-only training')
        if len({r['user_id'] for r in event}) != 1:
            raise ValueError('candidate event has inconsistent user IDs')
        c['impressions'] += 1
        c['simulated_impressions'] += int(not initial)
        if not initial:
            if sorted(r['rank'] for r in event) != list(range(1, len(event) + 1)):
                raise ValueError('candidate ranks must be a permutation')
            ranked = sorted(event, key=lambda r: r['rank'])
            if any(x['score'] < y['score'] for x, y in zip(ranked, ranked[1:])):
                raise ValueError('candidate score order differs from ranking')
            # Baseline impressions can repeat item IDs. Audit candidate positions,
            # preserving both multiplicity and labels rather than deduplicating IDs.
            paired = {source: Counter((r['candidate_id'].rsplit('-', 1)[0], r['label'])
                                      for r in event if r['source'] == source)
                      for source in ('human', 'llm')}
            if not paired['human'] or paired['human'] != paired['llm']:
                raise ValueError('candidate HGC/AIGC pairs or relevance labels misaligned')
            if abs(sum(r['probability'] for r in event) - 1) > 1e-12:
                raise ValueError('candidate probabilities do not sum to one')
        for r in event:
            c['clicks'] += r['clicked']
            c['click_llm'] += r['clicked'] * (r['source'] == 'llm')
            c['training_positive_llm'] += r['clicked'] * (r['source'] == 'llm')
            c['training_candidates'] += r['training_count']
            c['training_llm'] += r['training_count'] * (r['source'] == 'llm')
            for k in (3, 5):
                flag = 0 if initial else r[f'in_top{k}']
                c[f'exposure_count_at{k}'] += flag
                c[f'exposure_llm_at{k}'] += flag * (r['source'] == 'llm')
    try:
        with path.open(newline='', encoding='utf-8') as handle:
            reader = csv.DictReader(handle)
            if not set(fields).issubset(reader.fieldnames or []):
                return ['missing candidate diagnostics columns']
            for raw in reader:
                key = (int(raw['round']), int(raw['training_epoch']), raw['event_key'])
                if current != key:
                    flush()
                    current, event = key, []
                r = dict(raw)
                for name in ('clicked', 'training_count'):
                    r[name] = int(raw[name])
                if r['clicked'] not in (0, 1) or r['training_count'] < r['clicked']:
                    raise ValueError('invalid click/training multiplicity')
                if r['source'] not in ('human', 'llm') or not r['candidate_id'].endswith('-' + r['source']):
                    raise ValueError('candidate source suffix mismatch')
                initial = json.loads(key[2])[0] == 'initial'
                if not initial:
                    for name in ('score', 'cue', 'alpha', 'probability'):
                        r[name] = float(raw[name])
                        if not math.isfinite(r[name]):
                            raise ValueError('non-finite candidate value')
                    if any(not 0 <= r[name] <= 1 for name in ('probability', 'cue', 'alpha')):
                        raise ValueError('candidate probability/cue/alpha must lie in [0, 1]')
                    for name in ('rank', 'in_top3', 'in_top5'):
                        r[name] = int(raw[name])
                    r['label'] = int(raw['label'])
                    if r['label'] not in (0, 1) or r['clicked'] > r['label'] or (r['label'] == 0 and r['probability'] != 0):
                        raise ValueError('click/probability must respect relevance label')
                    if r['clicked'] and r['probability'] <= 0:
                        raise ValueError('clicked candidate must have positive probability')
                    for k in (3, 5):
                        if r[f'in_top{k}'] != int(r['rank'] <= k):
                            raise ValueError('top-k flag differs from rank')
                event.append(r)
            flush()
        expected = {(r['round'], r['training_epoch']): r for r in records}
        if set(counts) != set(expected):
            issues.append('candidate round/epoch events incomplete or unexpected (including header-only CSV)')
        for key, c in counts.items():
            summary = expected.get(key, {})
            for name, value in c.items():
                if name == 'simulated_impressions' and name not in summary:
                    continue
                if summary.get(name) != value:
                    issues.append(f'candidate summary count mismatch {key}: {name} actual={value} summary={summary.get(name)}')
        for r in records:
            if r.get('click_ratio') is not None and r.get('clicks') and abs(r['click_ratio'] - r['click_llm']/r['clicks']) > 1e-12:
                issues.append('recorded feedback click_ratio differs from counts')
    except (ValueError, KeyError, TypeError, OSError, json.JSONDecodeError) as error:
        issues.append(f'invalid candidate diagnostics: {error}')
    return issues


def compare_zero_candidates(left: Path, right: Path) -> list[str]:
    """Compare matched candidate rows with numeric tolerance, using bounded memory."""
    with left.open(newline='', encoding='utf-8') as a, right.open(newline='', encoding='utf-8') as b:
        for index, (x, y) in enumerate(zip_longest(csv.DictReader(a), csv.DictReader(b)), 1):
            if x is None or y is None:
                return ['PBM/zero candidate event counts differ']
            for name in ('round', 'training_epoch', 'event_key', 'user_id', 'candidate_id',
                         'source', 'clicked', 'training_count', 'rank', 'in_top3', 'in_top5'):
                if x[name] != y[name]:
                    return [f'PBM/zero matched candidate mismatch row {index}: {name}']
            if x['probability'] or y['probability']:
                if not x['probability'] or not y['probability'] or abs(float(x['probability']) - float(y['probability'])) > 1e-12:
                    return [f'PBM/zero probability differs beyond 1e-12 row {index}']
    return []


def analyze(root: Path, stage: str, phase: str = 'all') -> dict:
    if stage not in ('pilot', 'full') or phase not in ('core', 'extended', 'all'):
        raise ValueError('unsupported stage/phase')
    conditions = ('pbm', 'zero', 'heterogeneous') if stage == 'pilot' or phase == 'core' else (
        ('pbm', 'global', 'low', 'high', 'weak_cue') if phase == 'extended' else CONDITIONS)
    seeds = (2023,) if stage == 'pilot' else tuple(range(2023, 2028))
    expected_rounds = 3 if stage == 'pilot' else 20
    runs, issues, utility, seen, cohort, candidates = {}, [], {}, set(), None, {}
    for path in sorted(root.glob('*/manifest.json')):
        try:
            m = json.loads(path.read_text(encoding='utf-8'))
            if not all(name in m for name in ('stage', 'condition', 'seed', 'status')):
                raise ValueError('missing required manifest fields')
        except (OSError, ValueError, TypeError) as error:
            issues.append(f'{path.parent.name}: invalid manifest: {error}')
            continue
        if m['stage'] != stage or m['condition'] not in conditions or m['seed'] not in seeds:
            continue
        key = (m['condition'], m['seed'])
        if m['status'] != 'completed':
            issues.append(f'{path.parent.name}: status={m["status"]}')
            continue
        if key in seen:
            raise ValueError(f'Duplicate completed run for {key}; select a single cohort directory')
        seen.add(key)
        before = len(issues)
        signature = {k: m.get(k) for k in ('source_sha256', 'data_sha256', 'git_head', 'versions', 'platform')}
        signature['environment'] = {k: v for k, v in m.get('environment', {}).items() if k != 'PYTHONHASHSEED'}
        if not all(signature.get(k) for k in ('source_sha256', 'data_sha256', 'git_head', 'versions', 'environment')):
            issues.append(f'{key}: missing reproducibility signature')
        if cohort is None:
            cohort = signature
        elif signature != cohort:
            issues.append(f'{key}: differs from cohort data/source/environment/version signature')
        telemetry = path.parent / 'summaries.jsonl'
        if not telemetry.is_file():
            issues.append(f'{key}: missing summaries.jsonl')
            continue
        try:
            records = [json.loads(line) for line in telemetry.read_text(encoding='utf-8').splitlines() if line.strip()]
            for record in records:
                if not all(isinstance(record.get(name), int) and not isinstance(record.get(name), bool)
                           for name in ('round', 'training_epoch')):
                    raise ValueError('round/training_epoch must be integer')
                for name in {field for pair in RATIOS.values() for field in pair}:
                    if not isinstance(record.get(name), int) or isinstance(record.get(name), bool) or record[name] < 0:
                        raise ValueError(f'{name} must be a nonnegative integer count')
        except (OSError, ValueError, TypeError, AttributeError) as error:
            issues.append(f'{key}: invalid summary telemetry: {error}')
            continue
        expected_epochs = 1 if stage == 'pilot' else 5
        event_keys = [(r['round'], r['training_epoch']) for r in records]
        if len(event_keys) != len(set(event_keys)):
            issues.append(f'{key}: duplicate round/training_epoch telemetry')
        expected_keys = {(round_id, epoch) for round_id in range(1, expected_rounds + 1) for epoch in range(1, expected_epochs + 1)}
        if set(event_keys) != expected_keys:
            issues.append(f'{key}: incomplete or unexpected round/training_epoch tuples')
        if any(not math.isfinite(value) for r in records for value in r.values() if isinstance(value, (int, float))):
            issues.append(f'{key}: non-finite telemetry')
        for r in records:
            for metric, (num, den) in RATIOS.items():
                if num not in r or den not in r:
                    issues.append(f'{key}: missing counts for {metric}')
                elif r[num] is not None and r[den] is not None and not (0 <= r[num] <= r[den]):
                    issues.append(f'{key}: invalid counts for {metric}')
        rounds = summarize(records)
        if {r['round'] for r in rounds} != set(range(1, expected_rounds + 1)):
            issues.append(f'{key}: missing or unexpected rounds')
        if not (path.parent / 'candidates.csv').is_file() or (path.parent / 'candidates.csv').stat().st_size == 0:
            issues.append(f'{key}: missing candidate diagnostics')
        else:
            issues.extend(f'{key}: {problem}' for problem in audit_candidates(path.parent / 'candidates.csv', records))
        metrics_path = path.parent / 'metrics.jsonl'
        if metrics_path.is_file():
            try:
                metric_rows = [json.loads(line) for line in metrics_path.read_text(encoding='utf-8').splitlines() if line.strip()]
                if any(not all(name in row for name in ('loop_epoch', 'test_type', 'test_ratio')) for row in metric_rows):
                    raise ValueError('missing metric event fields')
                if any(not isinstance(row['test_ratio'], (int, float)) or not math.isfinite(row['test_ratio']) for row in metric_rows):
                    raise ValueError('invalid metric test_ratio')
            except (OSError, ValueError, TypeError) as error:
                issues.append(f'{key}: invalid utility telemetry: {error}')
                continue
            metric_events = [(r['loop_epoch'], r['test_type']) for r in metric_rows]
            expected_events = {(round_id, kind) for round_id in range(1, expected_rounds + 1)
                               for kind in ('Human', 'LLM', 'Human Target', 'LLM Target')}
            if len(metric_events) != len(set(metric_events)) or set(metric_events) != expected_events:
                issues.append(f'{key}: requires exactly four dev metric types per round')
            for r in metric_rows:
                for name in ('test_ratio', 'MAP3', 'MAP5', 'nDCG3', 'nDCG5', 'Recall3', 'Recall5'):
                    if name not in r or not isinstance(r[name], (int, float)) or not math.isfinite(r[name]):
                        issues.append(f'{key}: missing or non-finite {name}')
            for round_id in range(1, expected_rounds + 1):
                if len({r['test_ratio'] for r in metric_rows if r['loop_epoch'] == round_id}) != 1:
                    issues.append(f'{key}: inconsistent evaluation ratio within round {round_id}')
                feedback = next((r['feedback_ratio'] for r in rounds if r['round'] == round_id), None)
                if feedback is None or any(abs(r['test_ratio'] - feedback) > 1e-12
                                          for r in metric_rows if r['loop_epoch'] == round_id):
                    issues.append(f'{key}: evaluation ratio differs from last epoch feedback_ratio in round {round_id}')
        else:
            issues.append(f'{key}: missing dev utility metrics')
        if len(issues) == before:
            runs[key] = rounds
            utility[key] = metric_rows
            candidates[key] = path.parent / 'candidates.csv'
    for seed in seeds:
        if ('pbm', seed) in runs and ('zero', seed) in runs:
            issues.extend(compare_zero_candidates(candidates[('pbm', seed)], candidates[('zero', seed)]))
            if runs[('pbm', seed)] != runs[('zero', seed)]:
                issues.append(f'seed={seed}: PBM/zero click/feedback/exposure counts differ')
            left = {(r['loop_epoch'], r['test_type']): r for r in utility[('pbm', seed)]}
            for r in utility[('zero', seed)]:
                base = left[(r['loop_epoch'], r['test_type'])]
                for name in ('MAP3', 'MAP5', 'nDCG3', 'nDCG5', 'Recall3', 'Recall5'):
                    if abs(r[name] / 100 - base[name] / 100) > 1e-6:
                        issues.append(f'seed={seed}: PBM/zero utility {name} differs beyond 1e-6 in normalized units')
                if r['test_ratio'] != base['test_ratio']:
                    issues.append(f'seed={seed}: PBM/zero evaluation feedback ratio differs')
    for condition in conditions:
        for seed in seeds:
            if (condition, seed) not in runs:
                issues.append(f'Missing completed run: {condition}, seed={seed}')
    paired = []
    for condition in conditions:
        if condition == 'pbm':
            continue
        for seed in seeds:
            baseline = {r['round']: r for r in runs.get(('pbm', seed), [])}
            for row in runs.get((condition, seed), []):
                base = baseline.get(row['round'])
                if base is None:
                    continue
                for metric in (*RATIOS, 'feedback_ratio', 'last_epoch_exposure_ratio_at3', 'last_epoch_exposure_ratio_at5'):
                    if row[metric] is not None and base[metric] is not None:
                        paired.append({'condition': condition, 'seed': seed, 'round': row['round'],
                                       'metric': metric, 'paired_delta': row[metric] - base[metric]})
    utility_pairs = []
    relative_deltas = []
    for key, metrics in utility.items():
        baseline = {(r['loop_epoch'], r['test_type']): r for r in utility.get(('pbm', key[1]), [])}
        by_protocol = {(r['loop_epoch'], r['test_ratio'], r['test_type']): r for r in metrics}
        for row in metrics:
            event = (row['loop_epoch'], row['test_ratio'], row['test_type'])
            base = baseline.get((row['loop_epoch'], row['test_type']))
            for metric in ('MAP3', 'MAP5', 'nDCG3', 'nDCG5', 'Recall3', 'Recall5'):
                if metric not in row:
                    issues.append(f'{key}: missing {metric} at {event}')
                    continue
                if base is not None and metric in base and key[0] != 'pbm':
                    utility_pairs.append({'condition': key[0], 'seed': key[1], 'round': row['loop_epoch'],
                                          'test_ratio': row['test_ratio'], 'baseline_test_ratio': base['test_ratio'], 'test_type': row['test_type'],
                                          'metric': metric, 'paired_delta': row[metric] - base[metric]})
                reference = by_protocol.get((row['loop_epoch'], row['test_ratio'], 'LLM Target'))
                if row['test_type'] == 'Human Target' and reference:
                    relative_deltas.append({'condition': key[0], 'seed': key[1], 'round': row['loop_epoch'],
                                           'test_ratio': row['test_ratio'], 'test_type': row['test_type'],
                                           'metric': metric, 'relative_delta': relative_delta(row[metric], reference[metric])})
    final = []
    for condition in conditions:
        for metric in (*RATIOS, 'feedback_ratio', 'last_epoch_exposure_ratio_at3', 'last_epoch_exposure_ratio_at5'):
            group = [r for r in paired if r['condition'] == condition and r['metric'] == metric and r['round'] == expected_rounds]
            if group:
                values = [r['paired_delta'] for r in group]
                final.append({'condition': condition, 'metric': metric, 'independent_seeds': len(values),
                              'mean_paired_delta': statistics.mean(values),
                              'sample_sd': statistics.stdev(values) if len(values) > 1 else None})
    utility_final = []
    groups = {(r['condition'], r['metric'], r['test_type']) for r in utility_pairs}
    for condition, metric, kind in sorted(groups):
        values = [r['paired_delta'] for r in utility_pairs if r['condition'] == condition and r['metric'] == metric
                  and r['test_type'] == kind and r['round'] == expected_rounds]
        if values:
            utility_final.append({'condition': condition, 'metric': metric, 'test_type': kind,
                                  'independent_seeds': len(values), 'mean_paired_delta': statistics.mean(values),
                                  'sample_sd': statistics.stdev(values) if len(values) > 1 else None})
    primary_values = []
    for seed in seeds:
        groups = {condition: {r['round']: r for r in runs.get((condition, seed), [])}
                  for condition in ('pbm', 'heterogeneous')}
        if all(set(group) == set(range(1, expected_rounds+1)) for group in groups.values()):
            means = {}
            for condition, group in groups.items():
                values = [group[r]['last_epoch_exposure_ratio_at5'] for r in range(2, expected_rounds+1)]
                if all(value is not None for value in values):
                    means[condition] = statistics.mean(values)
            if len(means) == 2:
                primary_values.append({'seed': seed, 'first_round': 2, 'last_round': expected_rounds,
                    'pbm_mean_exposure_at5': means['pbm'], 'heterogeneous_mean_exposure_at5': means['heterogeneous'],
                    'paired_delta': means['heterogeneous'] - means['pbm']})
            else:
                issues.append(f'seed={seed}: missing primary exposure outside initial round')
    primary_summary = None
    if not issues and phase != 'extended' and len(primary_values) == len(seeds):
        values = [r['paired_delta'] for r in primary_values]
        primary_summary = {'independent_seeds': len(values), 'mean_paired_delta': statistics.mean(values),
                           'sample_sd': statistics.stdev(values) if len(values) > 1 else None}
    round_rows = [{'condition': key[0], 'seed': key[1], **row} for key, group in runs.items() for row in group]
    if issues:
        final, utility_final = [], []
    return {'stage': stage, 'phase': phase, 'expected_runs': len(conditions) * len(seeds),
            'complete': not issues, 'issues': issues, 'round_summary': round_rows,
            'primary_endpoint_by_seed': primary_values, 'primary_endpoint_seed_summary': primary_summary,
            'final_round_seed_summary': final,
            'utility_final_round_seed_summary': utility_final, 'utility_paired_deltas': utility_pairs,
            'relative_delta': relative_deltas,
            'paired_round_trajectories': paired,
            'limitations': ['Ratios describe conditional simulator impressions; exposure is within its candidate set.',
                            'Round trajectories are repeated observations, not independent samples.',
                            'Relative Delta = 100*(Human Target - LLM Target)/their mean, same round and ratio.',
                            'Incomplete or mismatched runs are excluded from aggregates.']}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    p.add_argument('--stage', choices=('pilot', 'full'), default='pilot')
    p.add_argument('--phase', choices=('core', 'extended', 'all'), default='all')
    p.add_argument('--output', type=Path, required=True, help='New analysis directory; never overwrite raw outputs')
    args = p.parse_args()
    result = analyze(args.root, args.stage, args.phase)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'analysis.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    rows = result['paired_round_trajectories']
    for name in ('relative_delta', 'utility_paired_deltas', 'utility_final_round_seed_summary', 'round_summary', 'primary_endpoint_by_seed'):
        table = result[name]
        if table:
            with (args.output / (name + '.csv')).open('x', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=list(table[0]))
                writer.writeheader()
                writer.writerows(table)
    if rows:
        with (args.output / 'paired_deltas.csv').open('x', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    # A dependency-free scientific trajectory figure; mean is across seeds at each round.
    plotted = [r for r in rows if r['metric'] == 'click_ratio']
    if plotted:
        colors = ('#1f77b4', '#d62728', '#2ca02c', '#9467bd', '#ff7f0e', '#17becf')
        svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="900" height="500" viewBox="0 0 900 500">',
               '<rect width="900" height="500" fill="white"/>',
               '<text x="60" y="28" font-family="sans-serif" font-size="18">AIGC conditional click share: paired difference from PBM</text>',
               '<path d="M60 55V420H720" fill="none" stroke="black"/>',
               '<text x="320" y="470" font-family="sans-serif">Feedback round (one based)</text>']
        ymax = max(abs(r['paired_delta']) for r in plotted) or 0.01
        last = max(r['round'] for r in plotted) or 1
        for index, condition in enumerate(sorted({r['condition'] for r in plotted})):
            points = []
            for round_id in sorted({r['round'] for r in plotted}):
                values = [r['paired_delta'] for r in plotted if r['condition'] == condition and r['round'] == round_id]
                if values:
                    points.append(f'{60 + round_id / last * 660:.2f},{237.5 - statistics.mean(values) / ymax * 170:.2f}')
            color = colors[index % len(colors)]
            svg += [f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2"/>',
                    f'<text x="735" y="{80 + index * 28}" font-family="sans-serif" fill="{color}">{condition}</text>']
        svg += [f'<text x="7" y="70" font-family="sans-serif">{ymax:.3f}</text>',
                '<text x="20" y="242" font-family="sans-serif">0</text>',
                f'<text x="3" y="418" font-family="sans-serif">-{ymax:.3f}</text>', '</svg>']
        (args.output / 'click_share_paired_trajectory.svg').write_text('\n'.join(svg), encoding='utf-8')
    bars = [r for r in result['utility_final_round_seed_summary'] if r['test_type'] == 'Human Target']
    if bars:
        extent = max(abs(r['mean_paired_delta']) for r in bars) or 0.01
        height = 75 + 28 * len(bars)
        svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="900" height="{height}">',
               '<rect width="100%" height="100%" fill="white"/>',
               '<text x="20" y="25" font-family="sans-serif" font-size="17">Final-round Human Target utility: paired mean across seeds</text>']
        for index, row in enumerate(bars):
            y = 50 + 28 * index
            value = row['mean_paired_delta']
            width = abs(value) / extent * 200
            x = 560 if value >= 0 else 560 - width
            svg += [f'<text x="20" y="{y + 14}" font-family="sans-serif">{row["condition"]} {row["metric"]} (n={row["independent_seeds"]})</text>',
                    f'<rect x="{x:.2f}" y="{y}" width="{width:.2f}" height="18" fill="{"#1f77b4" if value >= 0 else "#d62728"}"/>',
                    f'<text x="780" y="{y + 14}" font-family="sans-serif">{value:.5f}</text>']
        svg += [f'<path d="M560 45V{height - 10}" stroke="black"/>', '</svg>']
        (args.output / 'utility_final_paired.svg').write_text('\n'.join(svg), encoding='utf-8')
    print(json.dumps(result, indent=2))
    return 0 if result['complete'] else 2

if __name__ == '__main__':
    raise SystemExit(main())
