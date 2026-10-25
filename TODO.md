# 项目 TODO（按实现依赖排序）

- [x] 建立可运行骨架：用原版 mini-SWE-agent 跑单任务，保存轨迹，并记录模型与环境边界事件。
- [x] `schemas.py`：定义版本化 `StepEvent`、`TriggerDecision`、`DiagnosisRequest`、`DiagnosisReport`，校验事件顺序与诊断证据引用。
- [x] `storage.py`：写入和读取经过校验的 JSONL 事件；保存、读取诊断请求与报告，防止覆盖已有记录。
- [x] `observation.py`：对命令输出归一化，提取错误信号并比较命令前后的 Git 工作区状态；仍保留完整原始输出。
- [x] `mini_adapter.py`：把每步观察、步间暂停、反馈注入和提交拦截接入 mini-SWE-agent 执行循环。
- [x] `monitor.py`：根据事件窗口和 Agent 显式求助生成可解释、可追溯的介入决定；暂不自动运行诊断。
- [x] `workspace.py`、`evidence.py`：保存干净 Git 起点、导出候选补丁、恢复隔离工作区，并构造独立诊断请求。
- [x] `diagnosis/`：用 LangGraph 实现规则候选分类、独立模型判断、类别专属只读取证与证据校验后的诊断报告。
- [x] `recovery.py`、`validation.py`：执行反馈、隔离工作区原地恢复或新上下文重启；对同一失败命令复测，并在提交前检查完整补丁和聚焦测试。
- [x] 扩展 `runner.py`：串联监测、诊断、恢复、复测和提交验证；重启使用独立运行轨迹并导出最终候选补丁。
- [x] `benchmark/`、`cli.py`：固定任务集、运行增强版与两个消融配置，调用官方评估并生成逐题对照结果。

基线由用户运行。本项目不训练 FailFast，也不设计预算管理。
