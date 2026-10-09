"""Audit SACM run completeness and paired seed differences without pooling rounds."""
import argparse
import csv
import json
import math
import statistics
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
        rows.append(row)
    return rows

def relative_delta(human: float, llm: float) -> float | None:
    denominator = (human + llm) / 2
    return 100 * (human - llm) / denominator if denominator else None

def analyze(root: Path, stage: str) -> dict:
    conditions = ('pbm', 'zero', 'heterogeneous') if stage == 'pilot' else CONDITIONS
    seeds = (2023,) if stage == 'pilot' else tuple(range(2023, 2028))
    expected_rounds = 3 if stage == 'pilot' else 20
    runs, issues, utility, seen, cohort = {}, [], {}, set(), None
    for path in sorted(root.glob('*/manifest.json')):
        m = json.loads(path.read_text(encoding='utf-8'))
        if m['stage'] != stage:
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
        records = [json.loads(line) for line in telemetry.read_text(encoding='utf-8').splitlines() if line.strip()]
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
        metrics_path = path.parent / 'metrics.jsonl'
        if metrics_path.is_file():
            metric_rows = [json.loads(line) for line in metrics_path.read_text(encoding='utf-8').splitlines() if line.strip()]
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
        else:
            issues.append(f'{key}: missing dev utility metrics')
        if len(issues) == before:
            runs[key] = rounds
            utility[key] = metric_rows
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
                for metric in RATIOS:
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
        for metric in RATIOS:
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
    return {'stage': stage, 'complete': not issues, 'issues': issues, 'final_round_seed_summary': final,
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
    p.add_argument('--output', type=Path, required=True, help='New analysis directory; never overwrite raw outputs')
    args = p.parse_args()
    result = analyze(args.root, args.stage)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'analysis.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    rows = result['paired_round_trajectories']
    for name in ('relative_delta', 'utility_paired_deltas', 'utility_final_round_seed_summary'):
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
