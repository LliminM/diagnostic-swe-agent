"""Narrow the failure taxonomy using observable signals before model review."""

from pydantic import BaseModel, ConfigDict, Field

from ..observation import ActionObservation, Outcome
from ..schemas import DiagnosisRequest, EventKind, FailureCategory, TriggerReason


class Classification(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: FailureCategory
    rationale: str = Field(min_length=1)


def candidate_categories(request: DiagnosisRequest) -> tuple[FailureCategory, ...]:
    """This is a shortlist, not a diagnosis; UNKNOWN is always available."""

    reason = request.trigger.reason
    candidates: set[FailureCategory] = set()
    if reason is TriggerReason.TOOL_ERROR:
        candidates.update({FailureCategory.ENVIRONMENT, FailureCategory.CONTROL_FLOW})
    elif reason in {TriggerReason.REPEATED_TEST_FAILURE, TriggerReason.POST_EDIT_TEST_FAILURE}:
        candidates.update({FailureCategory.IMPLEMENTATION, FailureCategory.VERIFICATION,
                           FailureCategory.LOCALIZATION, FailureCategory.ENVIRONMENT})
    elif reason is TriggerReason.REPEATED_COMMAND_FAILURE:
        candidates.update({FailureCategory.ENVIRONMENT, FailureCategory.IMPLEMENTATION,
                           FailureCategory.LOCALIZATION})
    elif reason is TriggerReason.NO_PROGRESS:
        candidates.update({FailureCategory.CONTROL_FLOW, FailureCategory.UNDERSTANDING,
                           FailureCategory.LOCALIZATION})
    elif reason is TriggerReason.SUBMISSION_OUTPUT_GAP:
        candidates.update({FailureCategory.VERIFICATION, FailureCategory.IMPLEMENTATION})
    else:
        candidates.update(category for category in FailureCategory if category is not FailureCategory.UNKNOWN)

    for event in request.recent_events:
        if event.kind not in {EventKind.ACTION_RESULT, EventKind.ACTION_EXCEPTION}:
            continue
        raw = event.data.get("observation")
        if not isinstance(raw, dict):
            continue
        try:
            observation = ActionObservation.model_validate(raw)
        except ValueError:
            continue
        if observation.outcome is Outcome.TOOL_ERROR:
            candidates.add(FailureCategory.ENVIRONMENT)
        if observation.test_failed:
            candidates.update({FailureCategory.IMPLEMENTATION, FailureCategory.VERIFICATION})

    candidates.add(FailureCategory.UNKNOWN)
    return tuple(category for category in FailureCategory if category in candidates)
