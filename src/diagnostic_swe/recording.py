"""Pass-through wrappers at mini-SWE-agent's model and shell boundaries."""

from typing import Any

from .observation import GitChangeProbe, record_exception, record_step
from .schemas import EventKind
from .storage import EventJournal


class RecordingModel:
    """Delegate the mini model protocol, preserving its original responses."""

    def __init__(self, inner: Any, journal: EventJournal):
        self.inner = inner
        self.journal = journal
        self.config = inner.config

    def query(self, messages: list[dict], **kwargs: Any) -> dict:
        self.journal.turn += 1
        try:
            response = self.inner.query(messages, **kwargs)
        except Exception as error:
            self.journal.append(
                EventKind.MODEL_EXCEPTION,
                {
                    "type": type(error).__name__,
                    "message": str(error),
                    "messages": getattr(error, "messages", None),
                },
            )
            raise
        self.journal.append(EventKind.MODEL_RESPONSE, {"response": response})
        return response

    def format_message(self, **kwargs: Any) -> dict:
        return self.inner.format_message(**kwargs)

    def format_observation_messages(
        self, message: dict, outputs: list[dict], template_vars: dict | None = None
    ) -> list[dict]:
        return self.inner.format_observation_messages(message, outputs, template_vars)

    def get_template_vars(self, **kwargs: Any) -> dict:
        return self.inner.get_template_vars(**kwargs)

    def serialize(self) -> dict:
        return self.inner.serialize()


class RecordingEnvironment:
    """Record raw command results before the model formats or truncates them."""

    def __init__(self, inner: Any, journal: EventJournal):
        self.inner = inner
        self.journal = journal
        self.config = inner.config
        self._probe = GitChangeProbe(inner)
        self._snapshots = {}

    def execute(self, action: dict, cwd: str = "", **kwargs: Any) -> dict:
        before = self._snapshots.get(cwd)
        if before is None:
            before = self._probe.capture(cwd)
        try:
            result = self.inner.execute(action, cwd=cwd, **kwargs)
        except Exception as error:
            self.journal.append(
                EventKind.ACTION_EXCEPTION,
                {
                    "action": action,
                    "type": type(error).__name__,
                    "message": str(error),
                    "messages": getattr(error, "messages", None),
                    "observation": record_exception(action, error, before).model_dump(mode="json"),
                },
            )
            raise
        after = self._probe.capture(cwd)
        self._snapshots[cwd] = after
        observation = record_step(action, result, before, after)
        self.journal.append(
            EventKind.ACTION_RESULT,
            {"action": action, "result": result, "observation": observation.model_dump(mode="json")},
        )
        return result

    def get_template_vars(self, **kwargs: Any) -> dict:
        return self.inner.get_template_vars(**kwargs)

    def serialize(self) -> dict:
        return self.inner.serialize()
