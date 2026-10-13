"""Generate an evidence-linked Chinese report only after both acceptance gates."""
from pathlib import Path
import statistics


def write_full_report(output: Path, native: dict, controlled: dict, native_output: Path, controlled_output: Path) -> None:
    if not native['complete'] or native.get('acceptance_scope') != 'full-matrix' or not controlled['complete']:
        raise ValueError('Complete native and controlled acceptance required for a final report')
    if (native.get('stage') != 'full' or native.get('phase') != controlled.get('phase')
            or native.get('expected_runs') != controlled.get('expected_runs')
            or controlled.get('accepted_runs') != controlled.get('expected_runs')):
        raise ValueError('Native/controlled full populations must match')
    primary = native['primary_endpoint_by_seed']
    if len(primary) != 5 or {r['seed'] for r in primary} != set(range(2023, 2028)) or native['primary_endpoint_seed_summary']['independent_seeds'] != 5:
        raise ValueError('Five independent paired seeds required')
    output.mkdir(parents=True, exist_ok=False)
    lines = ['# SACM 20轮正式实验报告', '',
             '## 数据与验收', '',
             ('核心三条件' if native['phase'] == 'core' else '核心三条件及四个预定附加条件') +
             '、seeds2023–2027、20轮×5epoch；Baseline比例反馈与每轮重置机制保留。',
             '原生完整性与公共历史复评均通过；统计单位为seed，不把轮次/epoch/用户当独立重复。',
             '原生证据：[%s](%s)。' % ('analysis.json', (native_output / 'analysis.json').resolve().as_posix()),
             '受控证据：[%s](%s)。' % ('analysis.json', (controlled_output / 'analysis.json').resolve().as_posix()), '',
             '## 主要终点：第2–20轮末epoch平均AIGC exposure@5', '',
             '| seed | PBM平均 | 异质SACM平均 | SACM−PBM（百分点） |', '|---|---:|---:|---:|']
    lines += ['| {seed} | {pbm_mean_exposure_at5:.6f} | {heterogeneous_mean_exposure_at5:.6f} | {delta:.4f} |'.format(delta=r['paired_delta'] * 100, **r) for r in primary]
    summary = native['primary_endpoint_seed_summary']
    signs = [r['paired_delta'] for r in primary]
    lines += ['', '配对均值 %.4f 个百分点，seed间样本SD %.4f；降低/升高/零差异seed数为 %s/%s/%s。' %
              (summary['mean_paired_delta'] * 100, summary['sample_sd'] * 100,
               sum(v < 0 for v in signs), sum(v > 0 for v in signs), sum(v == 0 for v in signs)),
              '首轮曝光缺失已排除，没有填0；不设显著性、成功缓解或公平性阈值。', '',
              '![五seed配对曝光轨迹](exposure5_paired_trajectories.png)', '',
              '## 推荐效用与Relative Delta', '',
              '原生效用是各条件自身历史构成下的整体表现。以下为末轮指标，原日志单位为百分数：', '',
              '| 条件 | 指标类型 | seed | nDCG@5 | MAP@5 | Recall@5 |', '|---|---|---:|---:|---:|---:|']
    lines += ['| {condition} | {test_type} | {seed} | {nDCG5:.5f} | {MAP5:.5f} | {Recall5:.5f} |'.format(**r)
              for r in native['utility_raw'] if r['loop_epoch'] == 20 and r['test_type'] in ('Human Target', 'LLM Target')]
    lines += ['', 'Relative Delta各轮结果与原始分子/分母对应：原生relative_delta.csv及utility_raw.csv。'
              '来源方向反转需同时解释两种来源的原始效用；AIGC曝光降低不自动等于偏差缓解。', '',
              '## 固定历史比例辅助分析', '',
              '见受控report.zh-CN.md、seed_means.csv、paired_trajectories.csv和history_effects.csv。'
              '公共50%历史固定具体来源赋值；0/100%端点比较同一模型对评价历史构成的反应。'
              'native复评使用新公共抽样与CPU后端，不要求精确复现原GPU日志。'
              '受控dev曝光不是主要训练候选曝光，固定比例也不使训练数据相同。', '',
              '## 后续研究决策', '',
              '| 观察结果 | 后续问题 |', '|---|---|',
              '| 来源变化稳定且效用代价小 | 优先标定真实用户敏感度与线索，再检验跨模型/数据迁移 |',
              '| 曝光下降伴随效用损失 | 报告来源与效用权衡曲线，不只追求低曝光 |',
              '| 点击改变但长期曝光差异消失 | 对照训练正例、反馈与保存模型，定位抵消环节 |',
              '| 差异依赖评价历史 | 加强固定历史评估，收紧模型来源偏好的表述 |',
              '| 方向反转或seed不稳定 | 保留全部结果；独立预定追加seed预算再估计 |',
              '| 弱线索下作用减弱 | 研究带噪声线索、披露方式与真实感知 |',
              '| 零敏感度失败 | 暂停科研解释，调查工程原因 |', '',
              '四个附加条件按预定矩阵完成，不按核心结果筛选。逐用户历史、无点击、动态敏感度、连续在线训练留给独立后续协议。']
    (output / 'report.zh-CN.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    # Standard scientific plotting; retain each seed trace and the seed mean.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    traces = [r for r in native['paired_round_trajectories']
              if r['condition'] == 'heterogeneous' and r['metric'] == 'last_epoch_exposure_ratio_at5' and r['round'] >= 2]
    fig, axis = plt.subplots(figsize=(8, 4.5))
    for seed in range(2023, 2028):
        rows = sorted((r for r in traces if r['seed'] == seed), key=lambda r: r['round'])
        axis.plot([r['round'] for r in rows], [100 * r['paired_delta'] for r in rows], alpha=.55, linewidth=1, label=str(seed))
    rounds = list(range(2, 21))
    axis.plot(rounds, [100 * statistics.mean(r['paired_delta'] for r in traces if r['round'] == n) for n in rounds],
              color='black', linewidth=2.2, label='Seed mean')
    axis.axhline(0, color='gray', linestyle='--', linewidth=.8)
    axis.set(xlabel='Feedback round', ylabel='SACM minus PBM exposure@5 (percentage points)', xticks=[2, 5, 10, 15, 20])
    axis.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    for suffix in ('png', 'svg'):
        fig.savefig(output / ('exposure5_paired_trajectories.' + suffix), dpi=180)
    plt.close(fig)
