# SACM 可持续 ML 维护合同

## 机制公平，而非强制结果平衡

概率 PBM 与 SACM 共用 score 排序、位置权重、候选与事件身份及点击抽样；SACM 新增的点击折损只有 `1-alpha_u*S_i`。cue 是知觉线索，不是 ground-truth source；来源条件的重叠 Beta 先验保持原协议。人工高 cue 同样受更强折损。相同分数经随机同分排列后仍有不同 rank，并不保证相同基础权重。

当前概率排序合同为 `cross-source-exact-uniform-v1`：规范成员顺序后，混合来源精确同分组采用独立 `tie` namespace 的事件固定均匀排列。PBM/SACM 共享；不根据来源固定优先。不改变同来源同分或旧 rank，不将近似分数判为同分。该版本标签是本文合同名；运行 provenance 仍以实际源码 hash 为准，不能声称 CLI 已实现版本切换。

不强行加入 Registry/Hydra。现有 ClickPolicy 是小型策略模块，不是新的可注册推荐模型。职责保留为身份/静态参数、排序、基础权重、cue 折损、条件抽样、独立观测。只有未来确有多种新增模型组件才考虑工厂。

## 每次修改的本地门禁

```powershell
Set-Location E:\Lab\Rec_SourceBias
.\.venv\Scripts\python.exe -B scripts/verify_sacm_contract.py --output results/sacm_gate_NEW_UNIQUE_DIRECTORY
.\.venv\Scripts\python.exe -B scripts/audit_click_ties.py results/sacm_pilot_20261013-120212 --output results/tie_audit_NEW_UNIQUE_FILE.json
```

使用新的实际目录/文件名；已有输出拒绝覆盖。门禁 CPU/offline，保存 tests.log、gate.json、源码 hash、Git dirty 状态、Python、命令与退出码；验证期间源码不得变化。它包含小型合成优化器 smoke，但不启动真实完整训练、下载模型或安装依赖。不宣称类型/安全全面审计、GPU 确定性或研究有效性。

独立测试必须保留：旧 selector golden；PBM/零alpha/cueoff精确配对；全候选排列；同分两来源都能先排；独立随机流与静态状态；人工cue=.9/AIcue=.1、alpha=.5、等基础权重下11/30对19/30；重复位置概率质量；异常不静默回退；两轮 Dataset/Trainer 接入。

## 日志审计与分母

audit_click_ties.py 流式读取既有 candidates.csv，不修改实验结果：

- 配对单位为 `(base item,label)` 下的两来源出现次数；不去重，不使用笛卡尔积。
- 精确配对同分数 / 全部配对位置数，另报包含配对同分 impression 数 / 全部有分数的模拟 impression 数。
- 跨来源任意 item 同分的 impression 数另列，不能混同同 item 配对。
- 重复副本按每来源 CSV 顺序配对；若同源同item出现不同分数则身份有歧义，拒绝并要求 occurrence ID。先排比例不是所有重复副本两两比较。
- 初始化 round1 无分数，不计入模拟分母；round2+空/部分空、非有限、来源错配、未配对多重性、非法rank或降序错误均拒绝。
- 无 tie 时先排比例为 null，不写成50%或0%；CSV hash记录证据。审计不是完整实验矩阵验收，后者仍用 analyze_sacm_experiments.py。

## CPU 与 GPU 各自门禁

CPU 负责算法/数据契约；这些排序与解析本身不是 CUDA 运算。GPU 验证真实 BERT4Rec 初始化、训练、反馈和导出。既有三条件 GPU pilot 已完成；当前同分改动在实际记录中未触发，旧回放一致，不因该问题重复 GPU pilot。后续新完整运行必须验收新 round_artifacts，首格完整后再续配对格。若需独立 GPU smoke，至少两轮才覆盖 SACM，且使用新输出目录；按既有用户可见终端约定执行，不修改 batch128/news_batch64。

## 变更与交接

研究审查先于编码：说明改的是实现错误、观测，还是模型假设。编码与验证不同 Agent 负责；交接列输入版本、命令、配置、输出、验收、问题、下一动作。新 paired cohort 冻结源码/数据/环境；缺格与失败保留，不按结果选择条件。

下一协议版本候选，不在本轮实施：绝对路径参与 event_key 导致搬迁改变抽样；需逻辑 dataset/split/hash 身份版本。候选实例 occurrence_id 应与 item/cue 身份分开。全部同源同分交换对称、平均 examination、近似同分容差、逐用户真实历史追加、动态alpha均是明确新合同，不在当前维修中暗改。不要把模型状态导出称为可恢复训练 checkpoint（未含optimizer/RNG）。
