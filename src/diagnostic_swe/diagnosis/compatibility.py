"""Recognize a specific pytest runner mismatch from independent evidence."""

import re
from dataclasses import dataclass

from ..schemas import DiagnosisRequest, EventKind
from .probes import ProbeResult


_PYTEST_MAJOR = re.compile(r"\bpytest\s+(\d+)(?:\.\d+)*\b", re.IGNORECASE)
_MISSING_ATTRIBUTE = re.compile(
    r"AttributeError:\s*'([A-Za-z_][A-Za-z0-9_]*)' object has no attribute '([A-Za-z_][A-Za-z0-9_]*)'"
)


@dataclass(frozen=True)
class LegacyPytestSetup:
    class_name: str
    attribute_name: str
    event_sequence: int
    source_evidence_id: str


def _initializes_in_legacy_setup(source: str, class_name: str, attribute_name: str) -> bool:
    # git grep prefixes matches with "path:line:" and context with "path-line-".
    # Strip those prefixes so indentation still identifies the setup body.
    lines = [re.sub(r"^.*?\.py[:\-]\d+[:\-]", "", line)
             for line in source.splitlines() if line != "--"]
    start = next((index for index, line in enumerate(lines)
                  if re.match(rf"^\s*class {re.escape(class_name)}\s*[:(]", line)), None)
    if start is None:
        return False
    class_indent = len(lines[start]) - len(lines[start].lstrip())
    body: list[str] = []
    for line in lines[start + 1:]:
        indent = len(line) - len(line.lstrip())
        if line.strip() and indent <= class_indent:
            break
        body.append(line)
    if any(re.match(r"^\s*def setup_method\s*\(", line) for line in body):
        return False
    assignment = re.compile(rf"\bself\.{re.escape(attribute_name)}\s*=")
    for index, line in enumerate(body):
        if not re.match(r"^\s*def setup\s*\(", line):
            continue
        method_indent = len(line) - len(line.lstrip())
        for statement in body[index + 1:]:
            indent = len(statement) - len(statement.lstrip())
            if statement.strip() and indent <= method_indent:
                break
            if assignment.search(statement):
                return True
    return False


def detect_legacy_pytest_setup(
    request: DiagnosisRequest, probes: list[ProbeResult],
) -> LegacyPytestSetup | None:
    """Require pytest 8+, a missing fixture field, and a legacy setup method."""

    version = next((probe for probe in probes if probe.name == "pytest-version"), None)
    if version is None or version.returncode != 0:
        return None
    match = _PYTEST_MAJOR.search(version.output)
    if match is None or int(match.group(1)) < 8:
        return None

    for event in request.recent_events:
        if event.kind not in {EventKind.ACTION_RESULT, EventKind.ACTION_EXCEPTION}:
            continue
        result = event.data.get("result")
        if not isinstance(result, dict) or not isinstance(result.get("output"), str):
            continue
        for failed in _MISSING_ATTRIBUTE.finditer(result["output"]):
            class_name, attribute_name = failed.groups()
            for probe in probes:
                if (not probe.name.startswith("failed-test-source-")
                        or probe.returncode != 0
                        or not _initializes_in_legacy_setup(
                            probe.output, class_name, attribute_name)):
                    continue
                return LegacyPytestSetup(class_name, attribute_name,
                                         event.sequence, probe.evidence_id)
    return None
