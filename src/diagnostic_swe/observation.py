"""Normalize shell outcomes and compare read-only Git workspace snapshots."""

import hashlib
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from minisweagent.exceptions import Submitted
from pydantic import BaseModel, ConfigDict


_ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_TEST_FAILURE = re.compile(r"(?im)^(?:FAILED\s+\S+|--- FAIL:|FAILURES\b|\d+ failed\b|E\s+(?:AssertionError|assert\b))")
_TEST_COMMAND = re.compile(r"\b(?:pytest|unittest|tox|nox|ctest)\b|\b(?:go|cargo)\s+test\b")
_REPORTED_TEST_EXIT = re.compile(
    r"(?im)^(?:pytest|test|collect)\s+exit(?:\s+status|\s+code)?\s*:\s*[1-9]\d*\b"
)
_ERROR_LINE = re.compile(r"(?i)^(?:FAILED\s+\S+|--- FAIL:|E\s+\S+|.*(?:Error|Exception):\s*\S+)")


class Outcome(StrEnum):
    SUCCESS = "success"
    COMMAND_FAILURE = "command_failure"
    TOOL_ERROR = "tool_error"
    SUBMITTED = "submitted"


class ChangeState(StrEnum):
    CHANGED = "changed"
    UNCHANGED = "unchanged"
    UNKNOWN = "unknown"


class CodeChange(BaseModel):
    model_config = ConfigDict(extra="forbid")

    state: ChangeState
    dirty_files_after: list[str] = []
    before_fingerprint: str | None = None
    after_fingerprint: str | None = None
    diff_after: str = ""


class ActionObservation(BaseModel):
    """Signals only; the original result remains in the enclosing StepEvent."""

    model_config = ConfigDict(extra="forbid")

    command: str
    outcome: Outcome
    returncode: int | None
    exception_info: str
    output_sha256: str
    error_signature: str | None
    test_failed: bool
    code_change: CodeChange


@dataclass(frozen=True)
class GitSnapshot:
    status: str
    diff: str
    dirty_files: tuple[str, ...]
    fingerprint: str


def _status_paths(status: str) -> tuple[str, ...]:
    """Read paths from `git status --porcelain=v1 -z`, including rename pairs."""

    fields = status.split("\0")
    paths: set[str] = set()
    index = 0
    while index < len(fields):
        entry = fields[index]
        index += 1
        if not entry:
            continue
        if len(entry) < 4 or entry[2] != " ":
            continue
        paths.add(entry[3:])
        if "R" in entry[:2] or "C" in entry[:2]:
            if index < len(fields) and fields[index]:
                paths.add(fields[index])
            index += 1
    return tuple(sorted(paths))


class GitChangeProbe:
    """Query the same environment as the agent without changing its files."""

    def __init__(self, environment: Any):
        self.environment = environment

    def capture(self, cwd: str = "") -> GitSnapshot | None:
        try:
            status_result = self.environment.execute(
                {"command": "git status --porcelain=v1 -z --untracked-files=normal"}, cwd=cwd
            )
            if status_result.get("returncode") != 0 or status_result.get("exception_info"):
                return None
            diff_result = self.environment.execute({"command": "git diff --no-ext-diff HEAD"}, cwd=cwd)
            if diff_result.get("returncode") != 0 or diff_result.get("exception_info"):
                return None
        except Exception:
            # Observation cannot turn a working agent command into a failed one.
            return None

        status = str(status_result.get("output", ""))
        diff = str(diff_result.get("output", ""))
        fingerprint = hashlib.sha256((status + "\0" + diff).encode("utf-8")).hexdigest()
        return GitSnapshot(status, diff, _status_paths(status), fingerprint)


def _code_change(before: GitSnapshot | None, after: GitSnapshot | None) -> CodeChange:
    if before is None or after is None:
        state = ChangeState.UNKNOWN
    elif before.fingerprint == after.fingerprint:
        state = ChangeState.UNCHANGED
    else:
        state = ChangeState.CHANGED
    return CodeChange(
        state=state,
        dirty_files_after=list(after.dirty_files) if after else [],
        before_fingerprint=before.fingerprint if before else None,
        after_fingerprint=after.fingerprint if after else None,
        diff_after=after.diff if after and state is ChangeState.CHANGED else "",
    )


def _error_signature(output: str) -> str | None:
    lines = [" ".join(line.split()) for line in _ANSI_ESCAPE.sub("", output).splitlines()]
    lines = [line for line in lines if line]
    if not lines:
        return None
    selected = next((line for line in reversed(lines) if _ERROR_LINE.search(line)), lines[-1])
    return selected[:500]


def record_step(
    action: dict,
    result: dict,
    before: GitSnapshot | None,
    after: GitSnapshot | None,
) -> ActionObservation:
    """Derive stable signals without replacing or truncating the raw result."""

    output = str(result.get("output", ""))
    exception_info = str(result.get("exception_info", ""))
    returncode = result.get("returncode")
    if exception_info or returncode == -1:
        outcome = Outcome.TOOL_ERROR
    elif returncode == 0:
        outcome = Outcome.SUCCESS
    else:
        outcome = Outcome.COMMAND_FAILURE
    command = str(action.get("command", ""))
    clean_output = _ANSI_ESCAPE.sub("", output)
    failure_marker = bool(_TEST_FAILURE.search(clean_output))
    reported_exit = bool(_REPORTED_TEST_EXIT.search(clean_output))
    # Shell pipelines and a trailing `echo $?` can hide the test process's
    # nonzero status. Keep the actual shell outcome, but record the test signal.
    test_failed = (
        (outcome is Outcome.COMMAND_FAILURE and failure_marker)
        or (outcome is Outcome.SUCCESS and bool(_TEST_COMMAND.search(command))
            and (failure_marker or reported_exit))
    )
    return ActionObservation(
        command=command,
        outcome=outcome,
        returncode=returncode,
        exception_info=exception_info,
        output_sha256=hashlib.sha256(output.encode("utf-8")).hexdigest(),
        error_signature=_error_signature(exception_info or output)
        if outcome is not Outcome.SUCCESS or test_failed else None,
        test_failed=test_failed,
        code_change=_code_change(before, after),
    )


def record_exception(action: dict, error: Exception, before: GitSnapshot | None) -> ActionObservation:
    """An exception carries no regular shell result or reliable post-action snapshot."""

    outcome = Outcome.SUBMITTED if isinstance(error, Submitted) else Outcome.TOOL_ERROR
    return ActionObservation(
        command=str(action.get("command", "")),
        outcome=outcome,
        returncode=None,
        exception_info=f"{type(error).__name__}: {error}",
        output_sha256=hashlib.sha256(b"").hexdigest(),
        error_signature=f"{type(error).__name__}: {error}"[:500] if outcome is Outcome.TOOL_ERROR else None,
        test_failed=False,
        code_change=_code_change(before, None),
    )
