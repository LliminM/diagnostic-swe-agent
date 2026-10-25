"""Intervention hooks around mini-SWE-agent without changing its default behavior."""

import logging
from dataclasses import dataclass
from threading import Event, Lock
from typing import Callable

from minisweagent.agents.default import DefaultAgent
from minisweagent.exceptions import InterruptAgentFlow, Submitted

from .schemas import EventKind, StepEvent
from .storage import EventJournal


@dataclass(frozen=True)
class SubmissionDecision:
    accept: bool
    feedback: str = ""

    def __post_init__(self) -> None:
        if not self.accept and not self.feedback.strip():
            raise ValueError("a rejected submission needs feedback for the agent")


StepHook = Callable[[tuple[StepEvent, ...], "DiagnosticAgent"], None]
SubmissionHook = Callable[[str, "DiagnosticAgent"], SubmissionDecision]


class DiagnosticAgent(DefaultAgent):
    """Expose step boundaries, queued feedback and a submission decision hook.

    pause() takes effect before the next model query. A synchronous on_step hook
    also holds the agent at that boundary until it returns.
    """

    def __init__(
        self,
        model,
        env,
        *,
        journal: EventJournal,
        on_step: StepHook | None = None,
        on_submission: SubmissionHook | None = None,
        **kwargs,
    ):
        super().__init__(model, env, **kwargs)
        self.journal = journal
        self.on_step = on_step
        self.on_submission = on_submission
        self._seen_sequence = journal.next_sequence
        self._run_gate = Event()
        self._run_gate.set()
        self._feedback_lock = Lock()
        self._pending_feedback: list[str] = []
        self._restart_requested = False
        self._hook_logger = logging.getLogger(__name__)

    @property
    def is_paused(self) -> bool:
        return not self._run_gate.is_set()

    def pause(self) -> None:
        """Stop before the next query; an action already running can finish."""

        if not self.is_paused:
            self._run_gate.clear()
            self.journal.append(EventKind.PAUSED, {})

    def resume(self) -> None:
        if self.is_paused:
            self._run_gate.set()
            self.journal.append(EventKind.RESUMED, {})

    def stop_for_restart(self) -> None:
        """Finish this run before another agent starts from a fresh context."""

        if not self._restart_requested:
            self._restart_requested = True
            self.journal.append(EventKind.RESTART_REQUESTED, {})

    def inject_feedback(self, feedback: str) -> None:
        """Queue a user message for the next step boundary."""

        if not feedback.strip():
            raise ValueError("feedback must not be empty")
        if self.messages and self.messages[-1].get("role") == "exit":
            raise RuntimeError("cannot inject feedback after the agent exits")
        with self._feedback_lock:
            self._pending_feedback.append(feedback)

    def _flush_feedback(self) -> None:
        with self._feedback_lock:
            pending = self._pending_feedback
            self._pending_feedback = []
        for feedback in pending:
            self.add_messages(self.model.format_message(role="user", content=feedback))
            self.journal.append(EventKind.FEEDBACK_INJECTED, {"content": feedback})

    def _record_hook_error(self, hook: str, error: Exception) -> None:
        self._hook_logger.exception("%s hook failed", hook)
        self.journal.append(
            EventKind.ADAPTER_EXCEPTION,
            {"hook": hook, "type": type(error).__name__, "message": str(error)},
        )

    def _notify_step(self) -> None:
        events = self.journal.events_since(self._seen_sequence)
        self._seen_sequence = self.journal.next_sequence
        if self.on_step is not None and events:
            try:
                self.on_step(events, self)
            except Exception as error:
                self._record_hook_error("on_step", error)

    def step(self) -> list[dict]:
        self._run_gate.wait()
        if self._restart_requested:
            raise InterruptAgentFlow(
                {"role": "exit", "content": "DiagnosticRestart",
                 "extra": {"exit_status": "DiagnosticRestart", "submission": ""}}
            )
        self._flush_feedback()
        try:
            return super().step()
        finally:
            self._notify_step()

    def execute_actions(self, message: dict) -> list[dict]:
        if self.on_submission is None:
            return super().execute_actions(message)

        actions = message.get("extra", {}).get("actions", [])
        outputs: list[dict] = []
        for index, action in enumerate(actions):
            try:
                outputs.append(self.env.execute(action))
            except Submitted as submitted:
                submission = submitted.messages[0].get("extra", {}).get("submission", "")
                try:
                    decision = self.on_submission(submission, self)
                    if not isinstance(decision, SubmissionDecision):
                        raise TypeError("on_submission must return SubmissionDecision")
                except Exception as error:
                    self._record_hook_error("on_submission", error)
                    decision = SubmissionDecision(accept=True)

                if decision.accept:
                    self.journal.append(EventKind.SUBMISSION_ACCEPTED, {})
                    raise

                self.journal.append(EventKind.SUBMISSION_REJECTED, {"feedback": decision.feedback})
                outputs.append(
                    {"output": decision.feedback, "returncode": 1, "exception_info": "",
                     "extra": {"diagnostic_submission_rejected": True}}
                )
                for _ in actions[index + 1:]:
                    outputs.append(
                        {"output": "Skipped after rejected submission", "returncode": -1,
                         "exception_info": "", "extra": {"diagnostic_action_skipped": True}}
                    )
                break

        return self.add_messages(*self.model.format_observation_messages(message, outputs, self.get_template_vars()))
