# SACM-static r07 执行说明

在可见 PowerShell 终端进入 `E:\Lab\Rec_SourceBias`，使用现有 `.venv`。默认只预检，显式 `-Execute` / `--execute` 才开始训练。

```powershell
Set-Location E:\Lab\Rec_SourceBias
.\.venv\Scripts\python.exe scripts/run_sacm_experiment.py --stage pilot --condition heterogeneous
powershell -File scripts/run_sacm_matrix.ps1 -Stage pilot
powershell -File scripts/run_sacm_matrix.ps1 -Stage pilot -Execute -OutputRoot results/sacm_pilot
powershell -File scripts/run_sacm_matrix.ps1 -Stage full -Execute -OutputRoot results/sacm_full
.\.venv\Scripts\python.exe scripts/analyze_sacm_experiments.py results/sacm_pilot --stage pilot --output results/sacm_analysis_pilot
```

默认预检输出为 `results/sacm_preflight`，默认执行输出为 `results/sacm_runs`。独立 cohort 使用不同输出根目录，分析目录必须不存在。wrapper 与矩阵默认选择 `.venv/Scripts/python.exe`；可通过 `--python` / `-Python` 指定其他已有环境。预检验证 r07 文本 SHA、行为文件、软件版本和离线配置/tokenizer 缓存；不加载模型权重或训练。CUDA 不可用时执行停止。

每次调用生成唯一运行目录，`manifest.json` 记录命令、数据/源码 SHA、Git 状态、环境及 Python/Torch/Transformers/CUDA 版本。`run.log` 保存原始日志，`metrics.jsonl` 保存 dev 指标，`candidates.csv` / `summaries.jsonl` 保存模拟明细。失败保留记录，矩阵遇错停止。分析拒绝不完整或输入签名不一致的运行进入汇总。

Pilot 为 PBM、alpha=0 SACM、异质 SACM，seed 2023，3 rounds × 1 epoch；full 为七条件 × seeds 2023–2027，20 rounds × 5 epochs。统一 BERT4Rec、Amazon Beauty ChatGPT、batch 128、news batch 64、GPU 0、dev 评估、eta 1。round 和 training_epoch 均从 1 开始。HF 离线模式阻止下载，allocator 固定 `max_split_size_mb:128`。

历史 r07 单 seed full 约 16 分钟，仅作为初始预算；诊断开销和 GPU 状态会改变时长。35 次 full 按历史估计约 9–10 小时，应先检查 pilot。首轮 HGC 初始化，后续轮验证 SACM 路径。

模拟器选择“发生一次点击时点击哪个正例”，不估计无条件 CTR。点击比例、训练候选比例、训练正例比例、候选集合内 Top-K 曝光分母不同，现有 `new_ratio` / `llm_ratio` 保留原语义。每轮要求 Human、LLM、Human Target、LLM Target 四类 dev 指标完整且当前轮 test ratio 相同。

Relative Δ 按同 condition、seed、round、test ratio 的 Human Target 和 LLM Target 计算：`100 × (H-L) / ((H+L)/2)`；分母零标记 undefined。每轮轨迹属于重复观测，最终轮按独立 seed 汇总配对均值与样本标准差。效用比较按相同 seed、round、test type 配对，同时保存 SACM 与 PBM 的实际 test ratio：来源混合比例随反馈轨迹变化，这是动态历史条件下的整体效用比较。分析输出 CSV 和独立 SVG 图；不完整矩阵返回非零状态。

Beta 分布是预注册仿真假设。工程验证不代表研究效果，正式长 GPU 实验按用户可见终端工作流启动。
