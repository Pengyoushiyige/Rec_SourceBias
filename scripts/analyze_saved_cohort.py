"""Accept common-history reevaluations before seed-level descriptive comparisons."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import statistics

from run_sacm_experiment import CONDITIONS, CORE_CONDITIONS, ROOT, cohort_signature, sha256

RATIO_LABELS = ('native', '0.5', '0', '1')
GROUPS = ('Human', 'LLM', 'Human Target', 'LLM Target')
METRICS = ('nDCG1', 'nDCG3', 'nDCG5', 'MAP1', 'MAP3', 'MAP5', 'Recall3', 'Recall5')


def load(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def analyze_saved_cohort(run_paths: list[Path], reevaluations: Path, phase: str = 'core') -> dict:
    if phase not in ('core', 'all'):
        raise ValueError('Expected core or all population')
    conditions = CORE_CONDITIONS if phase == 'core' else CONDITIONS
    expected = {(condition, seed) for condition in conditions for seed in range(2023, 2028)}
    issues, seen, rows = [], set(), {}
    signature, vector_hashes, assignments = None, {}, {}
    for run_path in run_paths:
        try:
            run = load(run_path / 'manifest.json')
            key = (run['condition'], run['seed'])
            if key not in expected or key in seen or run['stage'] != 'full' or run['status'] != 'completed':
                raise ValueError('Unexpected/duplicate/incomplete run cell')
            seen.add(key)
            current = cohort_signature(run)
            if signature is None:
                signature = current
            elif current != signature:
                raise ValueError('Cohort source/data/environment signature differs')
            artifacts = run_path / 'round_artifacts'
            saved = load(artifacts / 'manifest.json')
            output = reevaluations / run_path.name
            manifest = load(output / 'manifest.json')
            if manifest['input_manifest'] != saved or manifest['device'] != 'cpu':
                raise ValueError('Reevaluation input/device mismatch')
            if manifest['namespace'] != 'sacm-controlled-evaluation-v1':
                raise ValueError('History assignment contract differs')
            if manifest['evaluator_sha256'] != sha256(ROOT / 'scripts/evaluate_saved_rounds.py'):
                raise ValueError('Evaluator source changed; existing reevaluation cannot be reused')
            if Path(manifest['artifacts']).resolve() != artifacts.resolve():
                raise ValueError('Reevaluation points to different saved rounds')
            if sha256(artifacts / saved['vectors_file']) != saved['vectors_sha256']:
                raise ValueError('Vector hash mismatch')
            if sha256(Path(saved['test_behaviors_file'])) != saved['test_behaviors_sha256']:
                raise ValueError('Behavior hash mismatch')
            # Same seed must share the same actual text representation, not only text inputs.
            previous = vector_hashes.setdefault(key[1], saved['vectors_sha256'])
            if previous != saved['vectors_sha256']:
                raise ValueError('Paired seed uses different fixed text vectors')
            records = load(output / 'analysis.json')
            expected_events = {(r, label) for r in range(1, 21) for label in RATIO_LABELS}
            if not isinstance(records, list) or len(records) != len(expected_events):
                raise ValueError('Expected exactly 20 rounds x 4 reevaluation ratios')
            actual = {(r['round'], r['ratio_label']) for r in records}
            if actual != expected_events:
                raise ValueError('Missing/duplicate/unexpected reevaluation event')
            summaries = [json.loads(line) for line in (run_path / 'summaries.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
            feedback = {r['round']: r['click_llm'] / r['clicks'] for r in summaries if r['training_epoch'] == 5}
            table = {}
            for record in records:
                round_id, label = record['round'], record['ratio_label']
                if record['assignment_seed'] != key[1]:
                    raise ValueError('Wrong assignment seed')
                if record['checkpoint_sha256'] != sha256(artifacts / ('round-%03d.pt' % round_id)):
                    raise ValueError('Reevaluation model checkpoint changed')
                ratio = feedback[round_id] if label == 'native' else float(label)
                if not finite(record['history_ratio']) or abs(record['history_ratio'] - ratio) > 1e-12:
                    raise ValueError('Unexpected history ratio')
                if not finite(record['native_feedback_ratio']) or abs(record['native_feedback_ratio'] - feedback[round_id]) > 1e-12:
                    raise ValueError('Feedback ratio differs from training')
                counts = record['counts']
                if any(not isinstance(counts[name], int) or isinstance(counts[name], bool) or counts[name] < 0
                       for name in ('total', 'all_zero', 'all_one', 'first_score_equal', 'evaluated')):
                    raise ValueError('Invalid evaluation counts')
                if counts['evaluated'] == 0 or counts['total'] != sum(counts[n] for n in ('all_zero', 'all_one', 'first_score_equal', 'evaluated')):
                    raise ValueError('Evaluation skip/count partition inconsistent or empty')
                digest = record['history_assignment_sha256']
                if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
                    raise ValueError('Invalid history assignment signature')
                if label != 'native':
                    previous = assignments.setdefault((key[1], label), digest)
                    if previous != digest:
                        raise ValueError('Common concrete histories differ across rounds/conditions')
                if set(record['metrics']) != set(GROUPS):
                    raise ValueError('Incomplete metric groups')
                for group in GROUPS:
                    metrics = record['metrics'][group]
                    if set(metrics) != set(METRICS) or any(not finite(v) or not 0 <= v <= 1 for v in metrics.values()):
                        raise ValueError('Invalid metrics or unit; expected 0..1')
                    for metric, value in metrics.items():
                        table[(round_id, label, group, metric)] = value
                for k in ('3', '5'):
                    value = record['exposure'][k]
                    if not finite(value) or not 0 <= value <= 1:
                        raise ValueError('Invalid dev exposure diagnostic')
                    table[(round_id, label, 'dev candidates', 'exposure@' + k)] = value
            with (output / 'metrics.csv').open(encoding='utf-8', newline='') as stream:
                csv_rows = list(csv.DictReader(stream))
            csv_table = {(int(r['round']), r['ratio_label'], r['metric_group'], r['metric']): float(r['value']) for r in csv_rows}
            metric_table = {event: value for event, value in table.items() if event[2] != 'dev candidates'}
            if len(csv_rows) != len(metric_table) or csv_table != metric_table:
                raise ValueError('metrics.csv differs from actual reevaluation records')
            rows[key] = table
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
            issues.append('%s: %s' % (run_path.name, error))
    missing = expected - set(rows)
    if missing:
        issues.append('Missing accepted reevaluations: %s' % sorted(missing))
    paired, relative, history_effects, seed_means, summary = [], [], [], [], []
    if not issues:
        for (condition, seed), table in sorted(rows.items()):
            base = rows[('pbm', seed)]
            for (round_id, label, group, metric), value in table.items():
                if label != 'native' and condition != 'pbm':
                    paired.append({'condition': condition, 'seed': seed, 'round': round_id, 'history_ratio': float(label),
                                   'group': group, 'metric': metric, 'value': value,
                                   'pbm_value': base[(round_id, label, group, metric)],
                                   'paired_delta': value - base[(round_id, label, group, metric)]})
                if label in ('0', '1'):
                    history_effects.append({'condition': condition, 'seed': seed, 'round': round_id, 'history_ratio': float(label),
                                            'group': group, 'metric': metric,
                                            'delta_from_half_history': value - table[(round_id, '0.5', group, metric)]})
                if group == 'Human Target':
                    llm = table[(round_id, label, 'LLM Target', metric)]
                    denominator = (value + llm) / 2
                    relative.append({'condition': condition, 'seed': seed, 'round': round_id, 'ratio_label': label,
                                     'metric': metric, 'human_target': value, 'llm_target': llm,
                                     'relative_delta_percent': 100 * (value - llm) / denominator if denominator else None})
        groups = {(r['condition'], r['group'], r['metric']) for r in paired}
        for condition, group, metric in sorted(groups):
            for seed in range(2023, 2028):
                values = [r['paired_delta'] for r in paired if r['condition'] == condition and r['seed'] == seed
                          and r['group'] == group and r['metric'] == metric and r['history_ratio'] == .5 and r['round'] >= 2]
                seed_means.append({'condition': condition, 'seed': seed, 'group': group, 'metric': metric,
                                   'first_round': 2, 'last_round': 20, 'mean_paired_delta': statistics.mean(values)})
            values = [r['mean_paired_delta'] for r in seed_means if r['condition'] == condition and r['group'] == group and r['metric'] == metric]
            summary.append({'condition': condition, 'group': group, 'metric': metric, 'independent_seeds': 5,
                            'mean_paired_delta': statistics.mean(values), 'sample_sd': statistics.stdev(values)})
    return {'complete': not issues, 'phase': phase, 'issues': issues, 'expected_runs': len(expected),
            'accepted_runs': len(rows), 'paired_trajectories': paired, 'relative_delta': relative,
            'history_effects': history_effects, 'seed_means': seed_means, 'seed_summary': summary,
            'limits': 'Auxiliary dev evaluation under controlled histories; not the primary training-candidate endpoint. '
                      'Native uses fresh common draws; not exact GPU-log reproduction. No significance/fairness claim.'}


def write_report(output: Path, result: dict) -> None:
    output.mkdir(parents=True, exist_ok=False)
    (output / 'analysis.json').write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    for name in ('paired_trajectories', 'relative_delta', 'history_effects', 'seed_means', 'seed_summary'):
        rows = result[name]
        if rows:
            with (output / (name + '.csv')).open('x', encoding='utf-8', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    lines = ['# 固定历史比例复评报告', '',
             '验收：%s；完整运行数：%s/%s。' % ('通过' if result['complete'] else '失败', result['accepted_runs'], result['expected_runs']), '',
             '不同轮模型在公共50%历史下比较；同一模型在0/50/100%历史下比较。统计单位为seed，轮次不作独立重复。', '',
             '辅助dev曝光不能替代正式训练候选曝光。native复评使用新公共抽样；不要求复现GPU日志。', '']
    if result['issues']:
        lines += ['## 未解决问题', ''] + ['- ' + issue for issue in result['issues']]
    else:
        lines += ['## 第2–20轮公共50%历史：相对PBM的seed内平均差', '',
                  '| 条件 | 指标组 | 指标 | 配对均值 | seed间SD |', '|---|---|---|---:|---:|']
        lines += ['| {condition} | {group} | {metric} | {mean_paired_delta:.6f} | {sample_sd:.6f} |'.format(**row)
                  for row in result['seed_summary']]
        lines += ['', '来源升高、降低、不稳定及效用损失均完整报告；本报告不设置成功阈值或显著性结论。']
    (output / 'report.zh-CN.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
