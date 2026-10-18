# GitHub 同步规则

Git 保存代码、配置及轻量研究证据；大体积实验载荷留在本地或独立产物存储。忽略上传不等于可以删除，远端代码仓库不承担这些文件的备份。

| 产物 | 默认同步 |
|---|---|
| 代码、文档、运行配置 | 是 |
| manifest、summaries.jsonl、metrics.jsonl、轻量事件日志 | 是 |
| 小型指标 CSV、报告、SVG/PNG 图 | 是 |
| results 下的 .pt/.pth/.ckpt/.safetensors/.onnx | 否 |
| results 下的 .npy/.npz/.h5/.hdf5/.parquet/.arrow | 否 |
| candidates.csv、candidate_events*.jsonl.gz | 否 |

规则只限制 results 下的指定文件，不全局忽略所有 CSV、JSON、SVG 或模型格式。指标趋势可由轻量文件分析；完整候选审计和固定模型重评价仍需要本地候选归档、向量及模型快照。共享这些证据时需另行提供产物位置和 hash，不宣称 clone 后即可完成所有复算。

## 已跟踪文件与历史提交

`.gitignore` 仅阻止未跟踪文件被默认加入，不能让已经跟踪的文件自动消失，也不能从既有提交中移除文件。不要用 `git add -f` 绕过规则。

GitHub 拒绝普通 Git 中大于100 MiB的文件，超过50 MiB会警告。即使最新提交已删除大文件，只要待推送祖先提交包含它，仍然失败。[GitHub 官方说明](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)

2026-10-18 核对：远端 codex/sacm-static=5679d26d954624f13cad23120edf42aa4e24019e。本地未发布提交为 e05057fa8089779c3007eeccf6f4f227b2705eab 和 83addc6ab142c4c4e997c3372303beb81f5af31c；第一个含4张约892–1032 MiB的CSV。第二个删CSV不够，需在备份后仅清理这段未发布历史，同时从提交快照中排除已跟踪重载荷。实际执行前重新核对分支与远端，保留工作目录的模型/数据和两个原提交的本地备份引用。历史重写待用户确认，不运行 reset --hard/clean/filter 全仓库。

这两次本地提交若被清理并仍基于原远端 tip，新分支依然是远端的后代，可正常 fast-forward 推送，无需强推。不要 `push --all` 或 `push --mirror`，否则可能连含大文件的本地备份引用一起上传。

## 清理执行结果（后续状态）

用户已确认并执行仅两个未发布提交的历史清理。原提交保留在本地 refs/codex/backups/before-artifact-cleanup-20261018，不作为推送目标。102个本地载荷文件保留，代码与轻量证据的Git树条目逐项核对不变；待推送新增对象中没有超过50MiB的文件。忽略规则及本文档已加入清理后的提交。该状态取代上文清理前的待确认状态；没有执行远程推送。今后可仅对codex/sacm-static正常推送，不使用force/mirror。
