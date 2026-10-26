# Diagnostic SWE Agent

这个项目在 mini-SWE-agent 2.4.6 上增加可观测接入层，随后逐步加入独立诊断与恢复。默认不配置介入钩子时，`DiagnosticAgent` 沿用原版 `DefaultAgent` 的生成命令、执行命令和结束任务流程。

`run_task(...)` 接收已构造的模型和环境，创建独立任务目录，输出：

- `trajectory.json`：mini-SWE-agent 自带的完整轨迹；
- `events.jsonl`：按顺序记录模型响应、原始命令结果和异常，供后续监测与诊断使用。

事件日志保留环境返回的原始输出；mini-SWE-agent 给模型看的观察仍按原配置格式化。任务 ID 必须是单个安全路径名，已有任务目录不会被覆盖。

每条命令事件还含 `observation`：返回状态、错误签名、测试失败信号，以及命令前后 Git 快照是否变化。Git 取证只读；在非 Git 目录或取证失败时，代码变化记为 `unknown`，不会误记为 `unchanged`。快照比较覆盖已跟踪文件的 diff 和未跟踪文件的路径；尚不识别内容变化但路径不变的未跟踪文件。

使用 `LocalEnvironment` 时，请把 `output_root` 放在被观察的任务仓库之外，避免轨迹文件本身进入 Git 快照。SWE-bench 的 Docker 环境与宿主机输出目录天然分离。

当前也已定义 `StepEvent`、`DiagnosisRequest`、`DiagnosisReport` 等协议。事件日志可通过 `read_events()` 校验并读取；诊断请求、报告可通过 `write_record()` / `read_record()` 独立存盘。

`DiagnosticAgent` 已接入每步回调、步间 `pause()` / `resume()`、`inject_feedback()` 和提交前判断。`run_task()` 可选传入 `on_step(events, agent)`、`on_submission(patch, agent)`；提交钩子返回 `SubmissionDecision(accept=True)` 或带反馈的 `SubmissionDecision(accept=False, feedback="...")`。退回的反馈会成为该提交命令的工具结果，便于模型继续修复。默认不传钩子时，仍按 mini-SWE-agent 的原有流程执行。暂停只在下一次模型查询前生效，当前正在执行的命令会先结束。

钩子自身抛出异常时会记录 `adapter_exception` 事件，不会中断主 Agent；提交钩子异常时默认接受提交。增强版 runner 已将 `validation.py` 接入提交钩子。

`FailureMonitor.observe(new_events)` 可在 `on_step` 中消费新事件，返回 `TriggerDecision` 或 `None`。规则使用可核查的信号：独立一行的 `[[REQUEST_DIAGNOSIS]]`（或模型响应 `extra.request_diagnosis=true`）、连续工具异常、候选代码修改后的首次测试失败、改动代码后同类测试再次失败、未改代码的重复命令失败，以及三次相同成功命令仍无新输出。重复类规则要求证据来自不同模型轮次，避免把一次回复中的并行命令误当成反复尝试。测试命令被 shell 管道或尾部 `echo $?` 掩盖退出码时，观测层会保留原始 shell 状态，同时记录输出中的失败信号。一次触发后需要新代码变化、反馈注入或显式 `resume_after_intervention()` 才会再次触发。监测器只做判定，增强版 runner 根据决定调用诊断 Agent。原版提示词尚未要求主 Agent 输出主动求助标记。

`WorkspaceManager` 在 Agent 开始修改前调用 `snapshot()`，要求任务仓库是干净的 Git 工作区，且宿主机产物目录位于仓库外。`export_candidate_patch()` 将当前已跟踪文件相对初始 commit 的修改保存为补丁，并列出未跟踪路径；补丁文件不包含未跟踪文件内容。`restore()` 会运行 Git 重置与清理命令，因此只有构造时明确设置 `isolated_workspace=True` 才能调用，应用场景是一次性 SWE-bench 任务容器。忽略文件不在快照和恢复范围内。`build_request()` 使用当前介入轮次的原始事件、候选补丁和既往诊断报告构造独立诊断请求。

`DiagnosticWorkflow(model, environment).diagnose(request)` 是独立诊断入口。LangGraph 先用监测信号缩小失败类别候选，再让诊断模型分类，并按类别运行只读命令（如 `git diff --check`、`git ls-files`），最后生成 `DiagnosisReport`。当事件包含失败的 pytest 测试类时，诊断会先读取 pytest 版本及该测试类的源码片段；若 pytest 8 以上版本遇到依靠 nose 风格 `setup()` 初始化缺失属性的测试，会明确归为镜像兼容性问题，建议修复镜像，不让主 Agent 修改任务代码来掩盖它。分类覆盖需求理解、定位、实现、验证、环境和流程控制；允许返回 `unknown`。`LiteLLMDiagnosticModel(model_name)` 每次只接收诊断专用的 system/user 消息，不传主 Agent 历史消息或执行工具。调用时应传入**原始任务环境**，不要传 `RecordingEnvironment`，避免诊断命令混入主 Agent 事件流。

模型只能引用请求事件、任务、diff 和只读取证的有效编号；恢复建议必须至少引用本次触发事件。无效 JSON、越界分类或缺失证据会降级为无恢复动作。诊断模块只产出建议，由 `recovery.py` 决定是否执行。使用方法：

```python
from diagnostic_swe.diagnosis import DiagnosticWorkflow, LiteLLMDiagnosticModel

workflow = DiagnosticWorkflow(LiteLLMDiagnosticModel("provider/model"), environment)
report = workflow.diagnose(request)  # request 为 build_request() 的返回值
```

`RecoveryController(workspace).apply_recovery(report=..., request=..., agent=...)` 应在同步 `on_step` 回调内调用，保证主 Agent 的当前命令已经结束。`feedback` 保留工作区和主 Agent 上下文，只排入简短诊断建议；`restore` 先保存当前已跟踪文件补丁，再将隔离工作区恢复到初始 Git 状态并让原 Agent 继续；`restart` 还需传入 `restart_factory`，该工厂只构造不运行新的 `DiagnosticAgent`。恢复控制器会预先给新 Agent 排入反馈，并让旧 Agent 在下一个模型查询前以 `DiagnosticRestart` 状态退出；随后由 runner 调用新 Agent 的 `run()`。`DiagnosticAgent` 的反馈队列会在新运行的首轮查询前注入消息。清理会删除未跟踪文件，因此只要检测到未跟踪路径，`restore` 和 `restart` 都会拒绝执行，现场保持原样。

`verify_recovery(request, subsequent_events)` 只有在介入后**同一失败命令**重跑成功时才报告 `improved`；相同失败再现报告 `unchanged`，未重跑或出现不同失败报告 `inconclusive`。`verify_submission(environment, snapshot=..., test_command=...)` 核对任务仓库、非空候选补丁、未跟踪文件、Git diff 格式及调用方明确配置的聚焦测试，并在测试后复查工作区。mini 默认提示词生成的临时 `patch.txt` 仅在其内容与实际提交完全相同时例外；其他未跟踪文件仍拒绝。测试命令必须来自可信运行配置，不能取自诊断建议；未配置测试时结果为 `inconclusive`。这些检查不替代后续 SWE-bench 官方评估。

提交前检查为 `inconclusive` 且未配置聚焦测试时，runner 还会检查一种可核查的覆盖缺口：任务涉及警告或输出，新增补丁只过滤 Python warnings，却没有处理 stdout/stderr。命中后只做一次提交前独立诊断，依据任务文本、补丁和提交检查事件，建议主 Agent 用可控对象复现两条输出通道，必要时捕获并记录日志；`full` 版本会退回首次提交并给出该建议。已通过聚焦测试或无此缺口的任务不会因此触发诊断。这项静态检查只是提示未验证的行为，不声称候选补丁已在官方目标测试中失败。

`run_diagnostic_task(...)` 把以上模块接入单任务运行。调用方提供 `model_factory`，使每次重启都创建新的主 Agent 模型实例；任务环境保持同一个隔离仓库。触发诊断后，它会保存请求、报告和候选补丁，执行报告允许的恢复动作，并在后续事件中寻找同一失败命令的复测结果。相同失败和相同补丁只诊断一次。提交前验证明确失败时，反馈会退回主 Agent；验证结果不确定时通常允许提交，输出过滤覆盖缺口例外。每次尝试分别保存 `attempt-001/` 等轨迹与事件，任务目录还包含 `diagnoses/`、提交检查、最终补丁和 `manifest.json`。`run_task(...)` 仍是无需诊断的基础入口。

```python
from diagnostic_swe import run_diagnostic_task

result = run_diagnostic_task(
    task_id="repo__issue", task=problem_statement,
    model_factory=create_main_model, environment=task_environment,
    diagnosis_model=diagnosis_model,
    system_template=system_template, instance_template=instance_template,
    output_root=output_root, test_command="pytest tests/test_target.py -q",
    isolated_workspace=True,
)
print(result.final_patch_path)
```

实际 SWE-bench 任务中的聚焦测试命令需由可信任务配置提供；不应从诊断模型建议中直接执行。`output_root` 必须位于任务仓库之外。

## SWE-bench 实验入口

`diagnostic-swe` 提供五个命令。先冻结同一数据集、split 和任务 ID，再分别运行基线与增强版，最后比较官方评分。下面以 Lite 的 `dev` split 前 10 个按 ID 排序的任务为例：

```bash
diagnostic-swe prepare --dataset lite --split dev --limit 10 --output experiments/lite10.json
diagnostic-swe run --manifest experiments/lite10.json --output experiments/enhanced \
  --model 'provider/main-model' --diagnosis-model 'provider/diagnosis-model'
diagnostic-swe evaluate --manifest experiments/lite10.json \
  --predictions experiments/enhanced/predictions.jsonl --run-id enhanced-lite10 \
  --output experiments/official
diagnostic-swe audit --artifacts experiments/official/logs/evaluation/enhanced-lite10 \
  --output experiments/enhanced-lite10-analysis.json
diagnostic-swe compare --manifest experiments/lite10.json \
  --baseline-report /path/to/baseline/results.json \
  --enhanced-report experiments/official/logs/evaluation/enhanced-lite10/results.json \
  --enhanced-run-dir experiments/enhanced --output experiments/comparison.json
```

`prepare` 也支持重复使用 `--instance-id` 精确指定任务，或传入本地 JSON/JSONL 数据集。可选的 `--test-commands` 是可信 JSON 字典，键为任务 ID，值为聚焦测试命令；未提供时提交检查只给出不确定结论，最终成功率仍以官方评估为准。任务清单保存问题描述、起始 commit 和镜像信息，后续不会重新选择任务。增强版使用 mini-SWE-agent 2.4.6 自带的 SWE-bench Docker 环境和默认 `swebench.yaml`；可用重复的 `--config` 传入与基线相同的配置。请保持两组实验的数据集、split、任务 ID、主模型与提示词一致。基线由你单独运行；比较命令接受包含这些任务的官方 `results.json`，基线报告可以额外包含其他任务。

`run --variant full` 运行完整诊断与恢复。`no-diagnosis` 关闭监测后的诊断调用，`no-recovery` 保留诊断报告但不执行恢复；两种消融均保留增强版的提交前验证，不能把 `no-diagnosis` 当作原版 mini-SWE-agent 基线。每种 variant 使用新的输出目录和官方 `run-id`。`run` 按任务逐个写入 `predictions.jsonl` 和 `task_status.jsonl`；采用 Agent 实际提交的补丁供官方评估，完整工作区 diff 另存为诊断产物。任务异常或没有正式提交时写入空补丁，官方报告会明确标记空补丁。对照结果记录配对解决率、改善与退步的任务 ID、逐题状态，以及诊断/恢复/重启次数。

官方评估需另行安装 [SWE-bench harness](https://github.com/SWE-bench/SWE-bench/blob/main/docs/guides/evaluation.md) 并具备可用 Docker。`evaluate` 直接运行 `swebench.harness.run_evaluation`，不会自行判分。评测完成后会自动分析逐题的 `report.json` 与 `test_output.txt`，在同一 `run-id` 目录写入 `diagnostic-analysis.json`；`audit` 可对已有产物单独运行。分析会区分目标测试失败与保留测试失败，提示可能的 pytest 兼容问题，但不会改动官方评分，也不会把隐藏测试反馈送回同一轮 Agent。真实模型、Docker 镜像和官方 harness 尚未在本地端到端运行；当前验证覆盖了命令拼装、产物格式和对照逻辑。

从环境搭建到基线、增强版、官方评分和配对比较的完整命令见 [TESTING.md](TESTING.md)。

## 开发验证

建议 Python 3.11 或更新版本：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python -m unittest discover -s tests -v
```

逐模块进度见 [TODO.md](TODO.md)，整体顺序见 [IMPLEMENTATION_ORDER.md](IMPLEMENTATION_ORDER.md)。真实 SWE-bench 基线由用户独立运行；这个接入层之后将用于增强版的可比对实验。
