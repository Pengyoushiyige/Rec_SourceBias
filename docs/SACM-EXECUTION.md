# SACM-static r07 执行说明

在可见 PowerShell 终端进入 `E:\Lab\Rec_SourceBias`，使用现有 `.venv`。默认仅预检，显式 `-Execute` / `--execute` 才开始训练。本文的 GPU 命令由用户启动。

## 只读检查与预检

```powershell
Set-Location E:\Lab\Rec_SourceBias
.\.venv\Scripts\python.exe -B scripts/run_sacm_experiment.py --help
.\.venv\Scripts\python.exe -B scripts/analyze_sacm_experiments.py --help
```

以上 `--help` 不创建运行目录。以下预检会写入独立 manifest 和准备记录，但不加载模型权重、不训练、不修改数据及配置。验证 r07 文本 SHA、行为文件、软件版本和离线配置/tokenizer 缓存。

```powershell
.\.venv\Scripts\python.exe -B scripts/run_sacm_experiment.py --stage pilot --phase core --condition heterogeneous
powershell -File scripts/run_sacm_matrix.ps1 -Stage pilot
```

默认预检输出为 `results/sacm_preflight`，执行输出为 `results/sacm_runs`。wrapper 与矩阵默认选择 `.venv/Scripts/python.exe`，可用 `--python` / `-Python` 指定已有环境。CUDA 不可用时执行停止；离线缓存或 Torch 环境异常也阻止执行，不自动下载或切换 CPU。

## 分阶段执行与分析

```powershell
# Pilot 仅用于工程验收：三条件 × seed2023，3轮 × 1 epoch。
powershell -File scripts/run_sacm_matrix.ps1 -Stage pilot -Execute -OutputRoot results/sacm_pilot_20261013
.\.venv\Scripts\python.exe -B scripts/analyze_sacm_experiments.py results/sacm_pilot_20261013 --stage pilot --phase core --output results/sacm_analysis_pilot_20261013

# 通过 pilot 后：核心三条件 × 五 seed，共15次运行。
powershell -File scripts/run_sacm_matrix.ps1 -Stage full -Phase core -Execute -OutputRoot results/sacm_full_20261013
.\.venv\Scripts\python.exe -B scripts/analyze_sacm_experiments.py results/sacm_full_20261013 --stage full --phase core --output results/sacm_analysis_core_20261013

# 固定同一 cohort 输出根目录补齐四个预定条件，共20次运行。
powershell -File scripts/run_sacm_matrix.ps1 -Stage full -Phase extended -Execute -OutputRoot results/sacm_full_20261013
.\.venv\Scripts\python.exe -B scripts/analyze_sacm_experiments.py results/sacm_full_20261013 --stage full --phase all --output results/sacm_analysis_all_20261013
```

`-Phase` 默认 `all`，兼容完整七条件35次运行；pilot 无论指定哪个 phase 始终只有 pbm、zero、heterogeneous 三条件、一个 seed，并记录有效 phase 为 core。full core 是 pbm/zero/heterogeneous；extended 是 global/low/high/weak_cue。full 为 seeds2023–2027，20轮 × 5 epoch。

同一 cohort 的 core/extended 必须采用同样数据、源码、环境与协议；先预检和验收再冻结执行代码。独立重跑使用新 OutputRoot。每次调用生成唯一运行目录，既有运行不覆盖；失败不会自动跳过重跑。出现同 condition/seed 重复时，分析不得自动挑一条，应整理为单独 cohort。分析输出目录必须不存在。矩阵遇失败停止，保留已完成结果及失败记录。

历史 r07 单 seed full 约16分钟，仅为粗略预算，不能保证当前时长。core 15次约4小时，extended 20次约5–6小时，全部35次约9–10小时；诊断开销、机器内存、GPU状态均可能改变时长。运行终端持续显示日志；每个独立目录的 `run.log` 是当前检查点，`manifest.json` 保存 prepared/running/completed/failed 状态。

## 保留机制与证据记录

统一 BERT4Rec、Amazon Beauty ChatGPT、batch128、news_batch64、GPU0、workers0、dev评估、eta1。round 与 training_epoch 从1开始。保留成对HGC/AIGC候选、原相关性、完整候选排序、条件单点击、总体比例历史替换、每轮模型及优化器重置、既有编码与负采样。首轮HGC初始化、后续训练历史使用HGC版本的不对称保留。

文本编码参数冻结但仍使用 train-mode，保留 dropout。每次进程按文本绝对路径缓存 CPU news vectors，并跨轮复用；另有 Hugging Face 本地离线模型/config/tokenizer缓存。`manifest.json` 的 `baseline_provenance` 记录这些类别、news_batch64和实际设置的缓存环境路径；命令及 data/source SHA记录实际输入文本、行为文件和代码版本。预检只验证 config/tokenizer可读，模型权重能否加载仍须 pilot 实测。HF离线模式及 allocator `max_split_size_mb:128` 固定。

loss 日志按实际完成 batch 数计算均值，首 batch不除零，空loader明确失败；criterion/backward/optimizer.step及优化机制不变。这是 batch均值而非按样本数加权均值。

每次运行保存 command/data/source SHA/Git状态/环境/Python-Torch-Transformers-CUDA版本；`run.log` 保存原始日志，`metrics.jsonl` 保存dev指标，`candidates.csv` 与 `summaries.jsonl` 保存候选事件及模拟统计。分析核对实际事件与汇总计数、轮次、epoch、条件及seed完整性，而非仅检查文件非空。

## 终点和验收

主要终点：每个条件/seed取第2–20轮各轮最后 epoch 的 exposure@5，计算19轮平均，再求 heterogeneous − pbm 的seed内配对差。首轮曝光为空并排除，不填0。`primary_endpoint_by_seed.csv` 报告全部五个seed，`analysis.json` 的 `primary_endpoint_seed_summary` 报告配对均值和样本标准差；轮次和用户记录不作独立重复。不预设显著性或成功阈值。

`round_summary.csv` 分开记录 `feedback_ratio`（末 epoch）与整轮所有epoch的点击/训练比例，并保留 `last_epoch_exposure_ratio_at5`。反馈至下一轮的比例必须来自末epoch，不能拿整轮均值替代。次要结果包括末轮曝光、exposure@3、训练来源、Relative Delta及nDCG/MAP/Recall。

pilot验收：完整三条件三轮，来源盲与zero机制的匹配事件点击和反馈比例一致，概率比较容差1e-12、效用容差1e-6。超出容差则调查原因，不用于挑参数。工程测试含空loader、单/多batch平均，以及三轮多epoch末epoch反馈桥接；fake tensor控制流证据不替代真实Torch CPU数值验收，正式pilot必须实际运行。

每轮Human、LLM、Human Target、LLM Target四类dev指标必须完整，同轮test ratio一致。Relative Delta按同condition/seed/round/test ratio计算 `100*(H-L)/((H+L)/2)`，零分母为undefined。效用按相同seed/round/test type配对，记录两条件实际test ratio；这是各自动态历史条件下的整体表现。

Beta分布是预注册仿真假设。条件单点击不估计总体CTR；曝光降低不自动等于公平性改善；仿真不能替代真实用户标定。报告降低、升高、无稳定差异及效用损失，不改Top-K点击、无点击、逐用户历史或连续在线训练。

## Windows 提交额度不足时

启动器在导入Torch前记录memory_before_probe，提交余量低于4GiB时早停并保留failed manifest。4GiB是保守启动门槛，不保证实际训练一定可运行；物理空闲/显存空闲不能替代commit余量。请保存工作并关闭闲置应用，在任务管理器性能/内存的已提交数值中确认足够余量，然后运行新的pilot cohort。不要因预检2MiB分配失败而改bs128/news_bs64；此时尚未开始训练。probe失败同时保存返回码、stderr与memory_after_probe。当前程序不会调整页面文件、结束用户进程或自动重试GPU。
