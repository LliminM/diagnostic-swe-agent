"""Find a narrow, evidence-based gap before accepting an unverified patch."""

import re


_OUTPUT_TASK = re.compile(
    r"\b(?:(?:future|deprecation)?warnings?|stdout|stderr|output|console)\b",
    re.IGNORECASE,
)
_WARNING_FILTER = re.compile(r"\bwarnings\.(?:catch_warnings|simplefilter|filterwarnings)\s*\(")
_STREAM_CAPTURE = re.compile(
    r"\b(?:redirect_stdout|redirect_stderr|capture_output|sys\.stdout|sys\.stderr)\b"
)


def has_unverified_output_suppression(problem_statement: str, patch: str) -> bool:
    """A warning-only edit does not establish where stdout/stderr goes.

    This is a request for a focused behavior check, not proof that the patch is
    wrong. Only added patch lines count; old imports or removed code do not.
    """

    if not _OUTPUT_TASK.search(problem_statement):
        return False
    added = "\n".join(line[1:] for line in patch.splitlines()
                      if line.startswith("+") and not line.startswith("+++"))
    return bool(_WARNING_FILTER.search(added) and not _STREAM_CAPTURE.search(added))
