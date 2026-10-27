# 测试与对照实验操作手册

本机可先运行单元测试。真实 SWE-bench 任务需要可访问模型 API、Hugging Face 数据集和 Docker 镜像。mini-SWE-agent 默认 SWE-bench 镜像针对 x86 Linux；在 Apple Silicon 上建议使用 x86_64 Linux 主机运行基线、增强版和官方评分。[mini 文档](https://mini-swe-agent.com/latest/usage/swebench/)说明了这一架构限制。

以下命令从项目根目录执行；本机主项目目录是 `/Users/minmin/Desktop/coding agent/diagnostic-swe-agent`。若复制到 Linux，请改用 Linux 上的项目路径。示例采用 Lite 的 `dev` split 和同一个主模型。请先按模型提供商说明设置 API 密钥，勿将密钥写入仓库。

## 1. 建立环境并运行本地测试

```bash
cd '/Users/minmin/Desktop/coding agent/diagnostic-swe-agent'
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e .
.venv/bin/python -m unittest discover -s tests -v
```

需要运行官方评分的主机上，另装 SWE-bench harness 到同一个虚拟环境：

```bash
git clone https://github.com/SWE-bench/SWE-bench.git ../SWE-bench
uv pip install --python .venv/bin/python -e ../SWE-bench
docker info
```

`python3` 在当前 macOS 是 3.9，低于项目要求的 3.11；使用上面的 `uv` 命令创建 3.12 环境。`docker info` 必须成功后再运行真实任务。记录 `../SWE-bench` 的 Git commit，方便复现实验。

## 2. 冻结同一组任务

先用 1 个实例检查完整流程；扩大实验时另建清单与输出目录：

```bash
export MODEL='anthropic/claude-sonnet-4-5-20250929'
export DIAG_MODEL="$MODEL"
.venv/bin/diagnostic-swe prepare --dataset lite --split dev --limit 1 \
  --output experiments/smoke.json
```

`smoke.json` 保存所选任务 ID、问题描述与起始版本。不要分别为基线和增强版重新抽样。生成 mini 基线命令所需的精确 ID 过滤表达式：

```bash
FILTER=$(.venv/bin/python -c 'import json,re; d=json.load(open("experiments/smoke.json")); print("^(?:"+"|".join(re.escape(x["instance_id"]) for x in d["tasks"])+")$")')
```

## 3. 原版 mini-SWE-agent 基线

```bash
.venv/bin/mini-extra swebench \
  --subset lite --split dev --filter "$FILTER" --workers 1 \
  --model "$MODEL" --output experiments/baseline \
  -c swebench.yaml -c agent.step_limit=0 -c agent.cost_limit=0
```

预期产物是 `experiments/baseline/preds.json`。最后两项配置仅用于使原版与当前增强版采用相同的终止条件；原版默认配置中的值不同。核对 `preds.json` 的任务 ID 与 `smoke.json` 完全一致，再进行评分。

## 4. 运行包含诊断 Agent 的版本

```bash
.venv/bin/diagnostic-swe run \
  --manifest experiments/smoke.json --output experiments/enhanced \
  --variant full --model "$MODEL" --diagnosis-model "$DIAG_MODEL"
cat experiments/enhanced/task_status.jsonl
```

预期产物是 `experiments/enhanced/predictions.jsonl`，每题的轨迹、事件和诊断报告位于 `experiments/enhanced/tasks/`。若 `task_status.jsonl` 出现 `error` 或 `has_patch: false`，先查看对应记录和事件；输出目录不能重复使用，修复后应换新目录重新运行。

`task_status.jsonl` 的 `base_commit_check` 同时记录数据集清单的 `base_commit` 和容器实际 `HEAD`。两者不一致时，增强版继续使用与原版 mini 相同的容器起点，并在任务工作区快照中保存实际 `HEAD`；比较或评分前应检查这项差异。在线诊断由失败监测规则触发；另外，未配置聚焦测试且提交前检查不确定时，非空补丁会触发一次模型驱动的行为覆盖审查。`full` 本身不保证每题都有恢复动作。

## 5. 用官方 SWE-bench harness 分别评分

```bash
.venv/bin/diagnostic-swe evaluate \
  --manifest experiments/smoke.json --predictions experiments/baseline/preds.json \
  --run-id baseline-smoke-001 --output experiments/official --workers 1
.venv/bin/diagnostic-swe evaluate \
  --manifest experiments/smoke.json --predictions experiments/enhanced/predictions.jsonl \
  --run-id enhanced-smoke-001 --output experiments/official --workers 1
```

报告分别保存在 `experiments/official/logs/evaluation/baseline-smoke-001/results.json` 与 `.../enhanced-smoke-001/results.json`；逐题失败分析另写入对应 `run-id` 目录的 `diagnostic-analysis.json`。对已有逐题 `report.json`、`test_output.txt` 可运行 `.venv/bin/diagnostic-swe audit --artifacts <评测日志目录> --output <新JSON文件>`。此分析不会改变官方评分，也不向同轮 Agent 提供隐藏测试。每次更改补丁或评测镜像后必须使用新的 `run-id`，因为官方 harness 会复用已有评分缓存。[官方评估说明](https://github.com/SWE-bench/SWE-bench/blob/main/docs/guides/evaluation.md)

三题历史排查材料和一次性的 pydicom 镜像修复脚本保存在 `research/case_studies/three_tasks_2026-10-26/`，不属于通用评测流程。若以后出现环境疑点，应先在原始代码和相同镜像中复测，再选择该项目兼容的依赖版本；基线和增强版必须使用相同镜像。

## 6. 配对比较

```bash
.venv/bin/diagnostic-swe compare \
  --manifest experiments/smoke.json \
  --baseline-report experiments/official/logs/evaluation/baseline-smoke-001/results.json \
  --enhanced-report experiments/official/logs/evaluation/enhanced-smoke-001/results.json \
  --enhanced-run-dir experiments/enhanced \
  --output experiments/comparison-smoke.json
```

`comparison-smoke.json` 含相同任务集合上的两组解决数、解决率、增强后解决的 `improved_ids`、退步的 `regressed_ids`，以及每题的官方状态。`error`、`empty_patch` 应保留为独立状态，不能算作已解决。`enhanced_interventions` 是过程统计，不替代官方解决率。1 个任务只检验流程是否跑通；正式比较应增加任务数量并固定同一清单、主模型、提示词和官方评估版本。

消融实验可对同一清单分别运行 `--variant no-diagnosis` 和 `--variant no-recovery`，使用新的输出目录与 `run-id`。二者仍保留提交前验证，所以 `no-diagnosis` 不是原版 mini 基线。
