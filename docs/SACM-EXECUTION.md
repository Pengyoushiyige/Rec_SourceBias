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

## 当前正式实验：先首格，验收后复用

三轮pilot已通过，不重跑。当前推荐使用`run_sacm_workflow.py`；它默认只读显示计划，`--execute`由用户在可见终端指定。

```powershell
Set-Location E:\Lab\Rec_SourceBias
# 使用尚未使用的新cohort名；首次完整运行只训练PBM seed2023，然后验收并停止。
.\.venv\Scripts\python.exe -B scripts/run_sacm_workflow.py first --cohort results/sacm_full_staged_20261013 --execute

# 首格通过后：重新验收已完成格并复用，补齐其余14格；zero失败立即停止。
.\.venv\Scripts\python.exe -B scripts/run_sacm_workflow.py core --cohort results/sacm_full_staged_20261013 --execute

# 无论核心结果方向如何，补齐预定四条件×五seed；复用核心对照。
.\.venv\Scripts\python.exe -B scripts/run_sacm_workflow.py extended --cohort results/sacm_full_staged_20261013 --execute

# 已保存模型的CPU公共历史复评，native/0.5/0/1；不重新训练。
.\.venv\Scripts\python.exe -B scripts/run_sacm_workflow.py reevaluate --phase all --cohort results/sacm_full_staged_20261013 --execute
```

也可在core完成后先使用`reevaluate --phase core`，随后all复评会核对并复用已有core离线输出。已有离线输出损坏时拒绝，不覆盖或悄悄重新选择。

`cohort.json`冻结源码、输入、Git HEAD、软件版本、系统、缓存来源和非seed环境。每格在实际启动器`Popen`之前再次核对签名。冻结后不改代码、提交或搬迁数据；不相容变更使用新cohort。默认`status`只显示完成/缺格清单，目录存在本身不算验收通过。

首格检查20份state、实际共享向量、有限张量、native同分计数、100个round/epoch组合、候选计数、80条dev指标与反馈连接。`metadata/first-calibration-*.json`记录实际耗时/bytes及剩余14格粗略预算；core预留30GiB、extended阶段至少40GiB剩余，all初始建议70GiB。模型状态没有optimizer/RNG，复用是完成格或离线分析复用，不是中断续训。

原始记录在`runs/`；预检、CPU gate和单格验收在`metadata/`；失败记录原地保留但不会替代完成格，重复completed格拒绝自动选择。`analysis/`有原生JSON/CSV（含utility_raw原始指标）、配对图和完整性结果；`reevaluations/`有各运行20轮×4历史比例的CPU结果。两种验收均通过才写`analysis/report-*/report.zh-CN.md`及五seed exposure@5轨迹PNG/SVG；尚未跑完不生成正式结论。

主要终点仍为训练候选第2–20轮末epoch曝光；公共历史dev复评是辅助分析，seed才是独立重复。来源下降、升高、不稳定与效用损失均保留。不会按核心结果挑选附加条件。

## 原有矩阵入口（历史兼容）

### 已完成首格的验收修复与续接（2026-10-13）

`results/sacm_full_staged_20261013-153814`的PBM seed2023训练已完整结束。停止原因是验收器查找`src/文件名`，而实际Windows manifest记录`src\文件名`。现在统一路径键并保留真正hash冲突拒绝。修复后不重复该训练。

原始`cohort.json`保持不变；`cohort-repair.json`记录四个辅助验收/协调脚本的旧、新精确hash，且校验原始冻结文件的hash。新manifest仍记录实际当前source hash和修复声明；只有这四项明确修复参与兼容匹配。模型、点击、训练源码、输入、环境、版本或任何未记录变化仍拒绝。后续workflow自动选择该修复记录。此机制不是允许任意更改代码。

使用相同cohort运行`core --execute`会先重新验收并复用已有PBM，再补齐核心剩余14格；`extended`再完成预定20格；`reevaluate --phase all`使用保存状态进行CPU复评并出完整报告。均由用户在可见终端启动。中断训练的运行不会从模型状态续训；这里复用的是已完成的运行。

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

正式耗时以首个20轮运行实测校准，不沿用历史r07的16分钟估计。运行终端持续显示日志；每个独立目录的`run.log`是当前检查点，`manifest.json`保存prepared/running/completed/failed状态。原有matrix会从头执行，不用于续接已经验收的首格；当前分阶段执行使用上面的workflow。

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

## 定向审查后的观测增强（2026-10-13）

概率PBM/SACM的跨来源精确同分，现在使用事件键下的均匀随机排列，消除固定human字典序优先；旧rank模式和非同分顺序保持。旧pilot共28,920个评分事件没有跨来源精确同分，逐事件回放点击与概率全部未变，所以不重跑旧pilot。

runner现在默认传入`--round_artifacts_dir`，每个新运行输出目录内生成：

- `round_artifacts/text_vectors.pt`：当次训练已使用的HGC/AIGC固定CPU向量，只保存一次，不重新编码。
- `round_artifacts/round-001.pt`…：逐轮用户编码器状态、native feedback ratio、原生评估跳过/同分计数、参数及hash。
- `round_artifacts/manifest.json`：源代码、测试行为与固定向量签名。目录和文件禁止覆盖。

不重复保存BERT权重，不改变训练损失、原生评价抽样、模型模式或反馈规则。状态保存已验证不消耗Python/NumPy/Torch CPU随机流。GPU保存只是张量转CPU/序列化，未进行新GPU工程运行。

在可见终端，对某个已完成运行保存的状态做公共历史评估：

```powershell
.\.venv\Scripts\python.exe -B scripts/evaluate_saved_rounds.py --artifacts "完整运行目录\round_artifacts" --output "新的离线评估目录" --ratios native,0.5 --device cpu
```

也可追加`--ratios native,0.5,0,1`做历史构成端点检查。默认仅ratio0.5，默认CPU；不训练、不下载预训练模型。相同seed、历史字符串、位置和ratio使用完全相同的具体来源抽样，键不包含condition、输出目录或轮次；确认两条件behavior hash/seed/固定向量相容及history_assignment_sha256一致后配对。

`metrics.csv`使用0..1指标单位，与原日志metric×100区别；`analysis.json`含分组指标、来源分数比较、曝光和skipcounts。这里的exposure是dev候选评估代理，不替代主实验train候选的exposure。`native`使用原反馈比例但新的固定公共抽样，不能声称逐位复现原生日志；相同评估器内比较native与0.5，才可帮助解释历史混合影响。

旧pilot无逐轮模型，不能补算其反事实。新正式cohort必须从首个运行开始使用这套观测，不与旧源码签名混合。

存储初始预算：根据实际pilot CSV与12.1MiB/轮用户编码器、约69.7MiB/运行固定向量张量估算，core15约20GiB、all35约46GiB；分别预留30GiB/70GiB覆盖序列化、日志及离线结果。这是估计，不是上限。当前E盘空闲检查约134GiB。不要为了节约空间删除原始数据或失败记录。
