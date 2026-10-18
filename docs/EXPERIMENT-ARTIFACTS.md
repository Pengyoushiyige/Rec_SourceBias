# 实验产物导出合同

从本次实现起，新运行不再导出 `candidates.csv`。旧实验不迁移、不删除；科学指标、候选顺序、随机流、训练与评价逻辑不变。

## 日常入口

| 产物 | 内容 | 默认用途 |
|---|---|---|
| `manifest.json` | 命令、输入/源码 hash、环境、状态及候选证据格式/大小 | 定位实验和复现条件 |
| `summaries.jsonl` | 每轮/epoch 的点击、训练来源、Top-3/5 曝光计数与比例、alpha/cue 累计量 | 日常机制分析；保留原始分子分母 |
| `metrics.jsonl` | 每轮各 dev 评价类型的 MAP、nDCG、Recall 和实际 test ratio | 评价与 Relative Delta 计算 |
| `events.jsonl`、`run.log` | loss、反馈事件和原始运行日志 | 排错与过程追溯 |
| `candidate_events.jsonl.gz` | 压缩、规范化的完整候选事件证据 | 有明确问题时查询/独立审计 |
| `round_artifacts/`（启用时） | 单次文本向量缓存、各轮用户编码器与评价计数 | 固定模型重评价；不是完整可恢复训练 checkpoint |

小型分析结果表仍可用 CSV；取消的是默认逐候选展开的巨型 CSV。汇总不是事件明细的等价副本，不能据此声称做过逐用户或逐候选审计。

## 压缩事件格式

标准库 gzip + JSONL，无新增依赖。首行为 `schema=click-candidate-events-v1` 与 click model；后续为三种记录：

- `user`：用户 ID、实际 alpha。以 `(user_id, alpha)` 建立插入顺序编号；初始化的空值单独处理。
- `candidate`：候选 ID、实际 cue。以 `(candidate_id, cue)` 建立插入顺序编号。
- `event`：round、training_epoch、原始序列化 event_key、用户编号、候选编号序列、selected 位置、稀疏 training 位置/重数；有分数的事件另含 scores/labels/ranks/probabilities 数组。

同一用户/候选的固定属性只存一次。候选编号序列允许重复，不能按 ID 去重；训练负采样重复通过重数保留。来源从 ID 后缀推导，Top-K 标记从实际 rank 推导。概率保留为当时的观测证据，不以事后重算替代。浮点使用 Python JSON 的完整往返精度，不降为 float32、不舍入。

初始化事件没有 score/label/rank/cue/alpha/probability，读取时保持空值；不能补零。event_key 的导出文本保持原样，模拟器内部身份及其参与点击/tie 随机抽样的方式不变。

读取器保留用户/候选字典及一个事件的展开内容，不把全部事件载入内存。未知 schema、非法索引、数组错位和截断 gzip 会失败。写入拒绝混用既有 CSV/汇总/压缩证据；不覆盖旧运行。

## 给人和 AI 的受限查询

以下命令中 `RUN_DIRECTORY` 替换为实际独立运行目录：

```powershell
# 默认只读轻量汇总，不打开候选归档；返回最多 20 行。
.\.venv\Scripts\python.exe -B scripts/query_experiment.py RUN_DIRECTORY --round 12

# dev 指标。
.\.venv\Scripts\python.exe -B scripts/query_experiment.py RUN_DIRECTORY --view metrics --round 12

# 确有事件问题时才读取明细；可组合 round/epoch/user/item 过滤。
.\.venv\Scripts\python.exe -B scripts/query_experiment.py RUN_DIRECTORY --view candidates --round 12 --epoch 5 --user-id USER_ID --limit 10
```

返回 JSON 包含实际 source、rows、limit、truncated。最多允许 1000 行；截断结果不是总体统计。gzip 明细按顺序扫描，晚期/无匹配查询可能读完整归档；这不是带索引数据库。查询不导出第二份全量文件。

`analyze_sacm_experiments.py` 的完整验收仍独立重数候选事件；`audit_click_ties.py` 仍做精确同分审计。这些主动审计可能扫描全量，日常汇总查询不会。

两个消费者自动优先选择新压缩证据，也支持历史 `candidates.csv`。两者并存时只选新证据；新证据损坏不能静默回退旧 CSV，也不能重复统计。新源码 hash 必须进入新 cohort 的冻结签名，不能把改动伪装为旧 cohort 的同版本续跑。
