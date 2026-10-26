"""Constrained, read-only evidence probes selected by observed failures."""

import re
import shlex
from dataclasses import dataclass
from typing import Any

from ..schemas import DiagnosisRequest, EventKind, FailureCategory


_FAILED_CLASS = re.compile(
    r"(?m)^FAILED\s+([A-Za-z0-9_./-]+\.py)::([A-Za-z_][A-Za-z0-9_]*)::"
)


@dataclass(frozen=True)
class ProbeResult:
    name: str
    command: str
    output: str
    returncode: int | None
    exception_info: str

    @property
    def evidence_id(self) -> str:
        return f"tool:{self.name}"


_COMMANDS: dict[FailureCategory, tuple[tuple[str, str], ...]] = {
    FailureCategory.ENVIRONMENT: (("pwd", "pwd"), ("status", "git status --short")),
    FailureCategory.LOCALIZATION: (("files", "git ls-files"),),
    FailureCategory.IMPLEMENTATION: (("diff-check", "git diff --check"),
                                     ("status", "git status --short")),
    FailureCategory.VERIFICATION: (("diff-check", "git diff --check"),),
}


class ReadOnlyProbes:
    """Never execute a command chosen by the model or task text."""

    def __init__(self, environment: Any, *, cwd: str = ""):
        self.environment = environment
        self.cwd = cwd

    def _execute(self, name: str, command: str) -> ProbeResult:
        try:
            raw = self.environment.execute({"command": command}, cwd=self.cwd)
            return ProbeResult(name, command, str(raw.get("output", ""))[:12000],
                               raw.get("returncode"), str(raw.get("exception_info", "")))
        except Exception as error:
            return ProbeResult(name, command, "", None,
                               f"{type(error).__name__}: {error}")

    def inspect_failed_tests(self, request: DiagnosisRequest) -> list[ProbeResult]:
        """Check pytest's version and at most two validated failing test classes."""

        failures: list[tuple[str, str]] = []
        for event in request.recent_events:
            if event.kind not in {EventKind.ACTION_RESULT, EventKind.ACTION_EXCEPTION}:
                continue
            result = event.data.get("result")
            if not isinstance(result, dict):
                continue
            output = result.get("output")
            if not isinstance(output, str):
                continue
            for path, class_name in _FAILED_CLASS.findall(output):
                if (path.startswith("/") or ".." in path.split("/")
                        or (path, class_name) in failures):
                    continue
                failures.append((path, class_name))
                if len(failures) == 2:
                    break
            if len(failures) == 2:
                break
        if not failures:
            return []

        results = [self._execute("pytest-version", "python -m pytest --version")]
        for index, (path, class_name) in enumerate(failures, 1):
            # The path and class are restricted by _FAILED_CLASS; quote them as
            # shell arguments as well. The command reads only this test file.
            command = ("git grep -n -A 24 -B 3 -F "
                       + shlex.quote(f"class {class_name}") + " -- " + shlex.quote(path))
            results.append(self._execute(f"failed-test-source-{index}", command))
        return results

    def inspect(self, category: FailureCategory) -> list[ProbeResult]:
        return [self._execute(name, command) for name, command in _COMMANDS.get(category, ())]
