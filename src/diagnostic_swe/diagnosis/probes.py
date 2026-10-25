"""Fixed, read-only evidence probes selected by the failure category."""

from dataclasses import dataclass
from typing import Any

from ..schemas import FailureCategory


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

    def inspect(self, category: FailureCategory) -> list[ProbeResult]:
        results = []
        for name, command in _COMMANDS.get(category, ()):
            try:
                raw = self.environment.execute({"command": command}, cwd=self.cwd)
                results.append(ProbeResult(name, command, str(raw.get("output", "")),
                                           raw.get("returncode"), str(raw.get("exception_info", ""))))
            except Exception as error:
                results.append(ProbeResult(name, command, "", None,
                                           f"{type(error).__name__}: {error}"))
        return results
