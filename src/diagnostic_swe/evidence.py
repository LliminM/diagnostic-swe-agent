"""Build the diagnosis agent's explicit, independent input context."""

from collections.abc import Sequence

from .schemas import DiagnosisReport, DiagnosisRequest, EventKind, StepEvent, TriggerDecision
from .workspace import CandidatePatch


_EPISODE_BOUNDARIES = {EventKind.FEEDBACK_INJECTED, EventKind.SUBMISSION_REJECTED}


def build_request(
    *,
    request_id: str,
    task_id: str,
    problem_statement: str,
    trigger: TriggerDecision,
    events: Sequence[StepEvent],
    candidate: CandidatePatch,
    prior_reports: Sequence[DiagnosisReport] = (),
) -> DiagnosisRequest:
    """Package the current episode, candidate diff and previous diagnoses.

    The caller supplies the full event stream recorded so far. We include events
    since the latest intervention, never silently omit an event cited by the
    trigger, and preserve raw outputs rather than relying on model summaries.
    """

    if not events:
        raise ValueError("cannot build diagnosis request without events")
    sequences = [event.sequence for event in events]
    if sequences != sorted(set(sequences)):
        raise ValueError("events must be ordered without duplicates")

    last_boundary = max(
        (index for index, event in enumerate(events) if event.kind in _EPISODE_BOUNDARIES),
        default=0,
    )
    episode = list(events[last_boundary:])
    available = {event.sequence for event in episode}
    if not set(trigger.evidence_event_sequences) <= available:
        raise ValueError("trigger evidence is missing from the current episode")

    return DiagnosisRequest(
        request_id=request_id,
        task_id=task_id,
        problem_statement=problem_statement,
        trigger=trigger,
        recent_events=episode,
        workspace_diff=candidate.patch,
        untracked_paths=list(candidate.untracked_paths),
        prior_reports=list(prior_reports),
    )
