"""Apply an evidence-linked recommendation at a main-agent step boundary."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol
from uuid import uuid4

from .schemas import DiagnosisReport, DiagnosisRequest, RecoveryMode
from .workspace import CandidatePatch, WorkspaceManager


class RecoveryError(RuntimeError):
    """A recommendation cannot be applied without risking the task workspace."""


class FeedbackAgent(Protocol):
    def inject_feedback(self, feedback: str) -> None: ...


class RestartableAgent(FeedbackAgent, Protocol):
    def stop_for_restart(self) -> None: ...


@dataclass(frozen=True)
class AppliedRecovery:
    mode: RecoveryMode
    agent: FeedbackAgent
    feedback: str
    saved_candidate: CandidatePatch | None = None


def _feedback(report: DiagnosisReport, *, workspace_reset: bool) -> str:
    prefix = "[独立诊断]"
    reset = "工作区已恢复到任务初始 Git 状态；此前文件修改不再存在。\n" if workspace_reset else ""
    return (
        f"{prefix}\n"
        f"{reset}"
        f"问题类别：{report.category.value}\n"
        f"诊断假设：{report.hypothesis.strip()}\n"
        f"建议下一步：{report.recommended_action.strip()}\n"
        "请先核查该假设，再继续修改；诊断本身不是测试通过的证明。"
    )


def _check_report(report: DiagnosisReport, request: DiagnosisRequest) -> None:
    if report.request_id != request.request_id:
        raise RecoveryError("diagnosis report does not belong to this request")
    if report.recovery_mode is RecoveryMode.NONE:
        return
    cited = {
        int(item.reference)
        for item in report.evidence
        if item.source == "event" and item.reference.isdecimal()
    }
    if not cited.intersection(request.trigger.evidence_event_sequences):
        raise RecoveryError("recovery recommendation must cite a triggering event")
    if not report.recommended_action.strip():
        raise RecoveryError("recovery recommendation is empty")


def _ensure_active(agent: FeedbackAgent) -> None:
    messages = getattr(agent, "messages", None)
    if messages and isinstance(messages[-1], dict) and messages[-1].get("role") == "exit":
        raise RecoveryError("cannot resume an agent that already exited")


class RecoveryController:
    """Call synchronously from on_step after the current command has finished.

    This method does not run the coding agent. The future runner starts or resumes
    the returned agent after this boundary. restart_factory must construct a fresh
    agent without running it or modifying the task checkout.
    """

    def __init__(self, workspace: WorkspaceManager):
        self.workspace = workspace

    def apply_recovery(
        self,
        *,
        report: DiagnosisReport,
        request: DiagnosisRequest,
        agent: RestartableAgent,
        restart_factory: Callable[[], FeedbackAgent] | None = None,
    ) -> AppliedRecovery:
        _check_report(report, request)
        mode = report.recovery_mode
        if mode is RecoveryMode.NONE:
            return AppliedRecovery(mode, agent, "")

        if mode is RecoveryMode.FEEDBACK:
            _ensure_active(agent)
            feedback = _feedback(report, workspace_reset=False)
            agent.inject_feedback(feedback)
            return AppliedRecovery(mode, agent, feedback)

        if not self.workspace.isolated_workspace:
            raise RecoveryError("restore and restart require an isolated task workspace")
        if mode is RecoveryMode.RESTART and restart_factory is None:
            raise RecoveryError("restart requires a fresh-agent factory")
        if mode is RecoveryMode.RESTART and not callable(getattr(agent, "stop_for_restart", None)):
            raise RecoveryError("current agent cannot stop for restart")
        if mode is RecoveryMode.RESTORE:
            _ensure_active(agent)

        # Export before any destructive Git command. A patch does not preserve
        # untracked contents, so reject this path while such files exist.
        candidate = self.workspace.export_candidate_patch(f"before-recovery-{uuid4().hex}")
        if candidate.untracked_paths:
            raise RecoveryError(
                "untracked files would be lost; preserve them before restore: "
                + ", ".join(candidate.untracked_paths)
            )

        target_agent = agent
        feedback = _feedback(report, workspace_reset=True)
        if mode is RecoveryMode.RESTART:
            assert restart_factory is not None
            try:
                target_agent = restart_factory()
            except Exception as error:
                raise RecoveryError(f"fresh agent could not be constructed: {error}") from error
            if target_agent is agent:
                raise RecoveryError("restart factory returned the current agent")
            _ensure_active(target_agent)
            # Detect a factory/agent that cannot receive feedback before reset.
            try:
                target_agent.inject_feedback(feedback)
            except Exception as error:
                raise RecoveryError(f"fresh agent could not receive feedback: {error}") from error

        self.workspace.restore()
        if mode is RecoveryMode.RESTORE:
            target_agent.inject_feedback(feedback)
        else:
            agent.stop_for_restart()
        return AppliedRecovery(mode, target_agent, feedback, candidate)
