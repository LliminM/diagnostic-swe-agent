"""Versioned data exchanged by the runner, monitor and diagnosis workflow."""

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, model_validator


class EventKind(StrEnum):
    RUN_STARTED = "run_started"
    MODEL_RESPONSE = "model_response"
    MODEL_EXCEPTION = "model_exception"
    ACTION_RESULT = "action_result"
    ACTION_EXCEPTION = "action_exception"
    RUN_FINISHED = "run_finished"
    RUN_EXCEPTION = "run_exception"
    PAUSED = "paused"
    RESUMED = "resumed"
    FEEDBACK_INJECTED = "feedback_injected"
    SUBMISSION_ACCEPTED = "submission_accepted"
    SUBMISSION_REJECTED = "submission_rejected"
    RESTART_REQUESTED = "restart_requested"
    DIAGNOSIS_STARTED = "diagnosis_started"
    DIAGNOSIS_REPORTED = "diagnosis_reported"
    DIAGNOSIS_SKIPPED = "diagnosis_skipped"
    DIAGNOSIS_FAILED = "diagnosis_failed"
    RECOVERY_APPLIED = "recovery_applied"
    RECOVERY_FAILED = "recovery_failed"
    RECOVERY_CHECKED = "recovery_checked"
    SUBMISSION_CHECKED = "submission_checked"
    ADAPTER_EXCEPTION = "adapter_exception"


class StepEvent(BaseModel):
    """One ordered, raw boundary event from a single task run."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    sequence: int = Field(ge=0)
    turn: int = Field(ge=0)
    timestamp: datetime
    kind: EventKind
    data: dict[str, Any]

    @model_validator(mode="after")
    def timestamp_is_timezone_aware(self) -> "StepEvent":
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ValueError("event timestamp must include a timezone")
        return self


class TriggerReason(StrEnum):
    REPEATED_COMMAND_FAILURE = "repeated_command_failure"
    REPEATED_TEST_FAILURE = "repeated_test_failure"
    POST_EDIT_TEST_FAILURE = "post_edit_test_failure"
    NO_PROGRESS = "no_progress"
    TOOL_ERROR = "tool_error"
    AGENT_REQUEST = "agent_request"
    SUBMISSION_UNVERIFIED = "submission_unverified"


class TriggerDecision(BaseModel):
    """A monitor's reason for starting an independent diagnosis."""

    model_config = ConfigDict(extra="forbid")

    reason: TriggerReason
    evidence_event_sequences: list[NonNegativeInt] = Field(min_length=1)
    explanation: str = Field(min_length=1)


class FailureCategory(StrEnum):
    UNDERSTANDING = "understanding"
    LOCALIZATION = "localization"
    IMPLEMENTATION = "implementation"
    VERIFICATION = "verification"
    ENVIRONMENT = "environment"
    CONTROL_FLOW = "control_flow"
    UNKNOWN = "unknown"


class RecoveryMode(StrEnum):
    NONE = "none"
    FEEDBACK = "feedback"
    RESTORE = "restore"
    RESTART = "restart"


class EvidenceRef(BaseModel):
    """A concrete observation supporting one diagnostic claim."""

    model_config = ConfigDict(extra="forbid")

    source: Literal["event", "diff", "task", "tool"]
    reference: str = Field(min_length=1)
    observation: str = Field(min_length=1)


class DiagnosisReport(BaseModel):
    """Structured diagnosis; later recovery code decides whether to apply it."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    request_id: str = Field(min_length=1)
    category: FailureCategory
    hypothesis: str = Field(min_length=1)
    evidence: list[EvidenceRef]
    recommended_action: str = ""
    recovery_mode: RecoveryMode = RecoveryMode.NONE

    @model_validator(mode="after")
    def actionable_report_needs_support(self) -> "DiagnosisReport":
        if self.recovery_mode is not RecoveryMode.NONE and (not self.evidence or not self.recommended_action):
            raise ValueError("a recovery recommendation requires evidence and an action")
        return self


class DiagnosisRequest(BaseModel):
    """An explicit evidence snapshot, separate from the main agent's mutable context."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    request_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    problem_statement: str = Field(min_length=1)
    trigger: TriggerDecision
    recent_events: list[StepEvent] = Field(min_length=1)
    workspace_diff: str = ""
    untracked_paths: list[str] = Field(default_factory=list)
    prior_reports: list[DiagnosisReport] = Field(default_factory=list)

    @model_validator(mode="after")
    def trigger_has_supplied_evidence(self) -> "DiagnosisRequest":
        sequences = [event.sequence for event in self.recent_events]
        if sequences != sorted(set(sequences)):
            raise ValueError("recent_events must be ordered without duplicates")
        available = set(sequences)
        if not set(self.trigger.evidence_event_sequences) <= available:
            raise ValueError("trigger evidence must be present in recent_events")
        return self
