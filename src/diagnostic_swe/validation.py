"""Evidence-based checks after recovery and before accepting a submission."""

import re
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict

from .observation import ActionObservation, Outcome
from .schemas import DiagnosisRequest, EventKind, StepEvent
from .workspace import WorkspaceSnapshot


class CheckStatus(StrEnum):
    IMPROVED = "improved"
    UNCHANGED = "unchanged"
    INCONCLUSIVE = "inconclusive"


class RecoveryCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: CheckStatus
    reason: str
    before_event_sequence: int | None = None
    after_event_sequence: int | None = None


class SubmissionStatus(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    INCONCLUSIVE = "inconclusive"


class SubmissionCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: SubmissionStatus
    reason: str
    diff_check_output: str = ""
    test_output: str = ""


def _observation(event: StepEvent) -> ActionObservation | None:
    if event.kind not in {EventKind.ACTION_RESULT, EventKind.ACTION_EXCEPTION}:
        return None
    raw = event.data.get("observation")
    if not isinstance(raw, dict):
        return None
    try:
        return ActionObservation.model_validate(raw)
    except ValueError:
        return None


def verify_recovery(
    request: DiagnosisRequest,
    subsequent_events: list[StepEvent],
    *,
    new_attempt: bool = False,
) -> RecoveryCheck:
    """Compare the triggered failing command with a later re-run of that command.

    Set new_attempt when events come from a restarted agent's fresh journal.
    Merely injecting feedback, changing code, or resetting Git is inconclusive.
    Different commands and different failure signatures do not prove improvement.
    """

    sequences = [event.sequence for event in subsequent_events]
    if sequences != sorted(set(sequences)):
        raise ValueError("subsequent events must be ordered without duplicates")
    triggered = set(request.trigger.evidence_event_sequences)
    before = next(
        ((event, observation) for event in reversed(request.recent_events)
         if event.sequence in triggered
         if (observation := _observation(event)) is not None
         if observation.command.strip()),
        None,
    )
    if before is None:
        return RecoveryCheck(status=CheckStatus.INCONCLUSIVE,
                             reason="触发证据中没有可重跑的命令。")
    before_event, before_observation = before
    if before_observation.outcome not in {Outcome.COMMAND_FAILURE, Outcome.TOOL_ERROR}:
        return RecoveryCheck(status=CheckStatus.INCONCLUSIVE,
                             reason="触发命令不是失败结果，无法据此判断恢复。",
                             before_event_sequence=before_event.sequence)

    last_sequence = request.recent_events[-1].sequence
    matching = [
        (event, observation)
        for event in subsequent_events
        if new_attempt or event.sequence > last_sequence
        if (observation := _observation(event)) is not None
        if observation.command.strip() == before_observation.command.strip()
    ]
    if not matching:
        return RecoveryCheck(status=CheckStatus.INCONCLUSIVE,
                             reason="介入后尚未重新执行触发失败的命令。",
                             before_event_sequence=before_event.sequence)

    after_event, after_observation = matching[-1]
    if after_observation.outcome is Outcome.SUCCESS:
        status, reason = CheckStatus.IMPROVED, "同一命令在介入后执行成功。"
    elif (after_observation.outcome is before_observation.outcome
          and after_observation.error_signature == before_observation.error_signature):
        status, reason = CheckStatus.UNCHANGED, "同一命令仍出现相同失败信号。"
    else:
        status, reason = CheckStatus.INCONCLUSIVE, "同一命令的结果已变化，但尚不能确认问题已解决。"
    return RecoveryCheck(status=status, reason=reason,
                         before_event_sequence=before_event.sequence,
                         after_event_sequence=after_event.sequence)


_COMMIT = re.compile(r"[0-9a-f]{40,64}\Z")


def verify_submission(
    environment: Any,
    *,
    snapshot: WorkspaceSnapshot,
    test_command: str | None,
    cwd: str = "",
    submitted_patch: str | None = None,
) -> SubmissionCheck:
    """Check the whole candidate diff and a caller-configured focused test.

    test_command must come from trusted run configuration, never from the model's
    recommendation. Without a test command, a clean diff alone is inconclusive.
    Official SWE-bench evaluation remains a separate later step.
    """

    if not _COMMIT.fullmatch(snapshot.base_commit):
        raise ValueError("invalid snapshot commit")

    def execute(command: str) -> tuple[int | None, str, str]:
        try:
            result = environment.execute({"command": command}, cwd=cwd)
        except Exception as error:
            return None, "", f"{type(error).__name__}: {error}"
        return result.get("returncode"), str(result.get("output", "")), str(result.get("exception_info", ""))

    def untracked_problem(status_output: str) -> str | None:
        paths = [entry[3:] for entry in status_output.split("\0") if entry.startswith("?? ")]
        if not paths:
            return None
        # mini's SWE-bench prompt writes this exact temporary submission file.
        # It is safe to exclude only when its bytes match what the agent submitted.
        if paths == ["patch.txt"] and submitted_patch is not None:
            code, content, error = execute("cat patch.txt")
            if code == 0 and not error and content == submitted_patch:
                return None
            return "提交文件 patch.txt 与实际提交内容不一致。"
        return "存在未跟踪文件，其内容不会进入当前候选补丁。"

    root_code, root_output, root_error = execute("git rev-parse --show-toplevel")
    if root_error or root_code != 0 or root_output.strip() != snapshot.repo_root:
        return SubmissionCheck(status=SubmissionStatus.INCONCLUSIVE,
                               reason="无法确认当前环境对应任务初始仓库。")

    status_code, status_output, status_error = execute("git status --porcelain=v1 -z --untracked-files=all")
    if status_error or status_code != 0:
        return SubmissionCheck(status=SubmissionStatus.INCONCLUSIVE,
                               reason="无法检查未跟踪文件。")
    if problem := untracked_problem(status_output):
        return SubmissionCheck(status=SubmissionStatus.FAIL,
                               reason=problem)

    changed_code, changed_output, changed_error = execute(
        f"git diff --name-only {snapshot.base_commit}"
    )
    if changed_error or changed_code != 0:
        return SubmissionCheck(status=SubmissionStatus.INCONCLUSIVE,
                               reason="无法确认候选补丁是否包含文件变更。")
    if not changed_output.strip():
        return SubmissionCheck(status=SubmissionStatus.FAIL,
                               reason="候选补丁为空。")

    diff_code, diff_output, diff_error = execute(f"git diff --check {snapshot.base_commit}")
    if diff_error or diff_code is None:
        return SubmissionCheck(status=SubmissionStatus.INCONCLUSIVE,
                               reason=f"无法检查候选补丁：{diff_error or '工具未返回状态码'}",
                               diff_check_output=diff_output)
    if diff_code != 0:
        return SubmissionCheck(status=SubmissionStatus.FAIL,
                               reason="候选补丁存在 Git diff 格式错误。",
                               diff_check_output=diff_output)
    if not test_command or not test_command.strip():
        return SubmissionCheck(status=SubmissionStatus.INCONCLUSIVE,
                               reason="未配置聚焦测试命令，无法确认提交有效。",
                               diff_check_output=diff_output)

    test_code, test_output, test_error = execute(test_command)
    if test_error or test_code is None:
        return SubmissionCheck(status=SubmissionStatus.INCONCLUSIVE,
                               reason=f"聚焦测试未能正常执行：{test_error or '工具未返回状态码'}",
                               diff_check_output=diff_output, test_output=test_output)
    if test_code != 0:
        return SubmissionCheck(status=SubmissionStatus.FAIL,
                               reason="聚焦测试仍然失败。",
                               diff_check_output=diff_output, test_output=test_output)

    post_status_code, post_status, post_error = execute("git status --porcelain=v1 -z --untracked-files=all")
    if post_error or post_status_code != 0:
        return SubmissionCheck(status=SubmissionStatus.INCONCLUSIVE,
                               reason="测试通过，但无法检查测试后的工作区。",
                               diff_check_output=diff_output, test_output=test_output)
    if problem := untracked_problem(post_status):
        return SubmissionCheck(status=SubmissionStatus.FAIL,
                               reason=f"测试产生或保留了未跟踪文件：{problem}",
                               diff_check_output=diff_output, test_output=test_output)
    final_code, final_output, final_error = execute(f"git diff --check {snapshot.base_commit}")
    if final_error or final_code is None:
        return SubmissionCheck(status=SubmissionStatus.INCONCLUSIVE,
                               reason="测试通过，但无法检查测试后的补丁。",
                               diff_check_output=final_output, test_output=test_output)
    if final_code != 0:
        return SubmissionCheck(status=SubmissionStatus.FAIL,
                               reason="测试后的候选补丁存在 Git diff 格式错误。",
                               diff_check_output=final_output, test_output=test_output)
    final_changed_code, final_changed, final_changed_error = execute(
        f"git diff --name-only {snapshot.base_commit}"
    )
    if final_changed_error or final_changed_code != 0 or not final_changed.strip():
        return SubmissionCheck(status=SubmissionStatus.INCONCLUSIVE,
                               reason="测试通过，但测试后无法确认非空候选补丁。",
                               diff_check_output=final_output, test_output=test_output)
    return SubmissionCheck(status=SubmissionStatus.PASS,
                           reason="候选补丁格式检查与聚焦测试均通过。",
                           diff_check_output=final_output, test_output=test_output)
