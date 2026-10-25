"""Append-only event storage and typed diagnosis artifact serialization."""

from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import TypeVar

from pydantic import BaseModel

from .schemas import EventKind, StepEvent


class EventJournal:
    """Write one validated event per line; never mix separate run attempts."""

    def __init__(self, path: Path):
        self.path = path
        self._sequence = 0
        self._events: list[StepEvent] = []
        self._lock = Lock()
        self.turn = 0
        self.path.open("x", encoding="utf-8").close()

    def append(self, kind: EventKind, data: dict) -> StepEvent:
        with self._lock:
            event = StepEvent(
                sequence=self._sequence,
                turn=self.turn,
                timestamp=datetime.now(timezone.utc),
                kind=kind,
                data=data,
            )
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(event.model_dump_json() + "\n")
            self._events.append(event)
            self._sequence += 1
        return event

    @property
    def next_sequence(self) -> int:
        with self._lock:
            return self._sequence

    def events_since(self, sequence: int) -> tuple[StepEvent, ...]:
        with self._lock:
            if sequence < 0 or sequence > self._sequence:
                raise ValueError("sequence is outside this journal")
            return tuple(self._events[sequence:])


def read_events(path: Path) -> list[StepEvent]:
    """Load a complete event stream, rejecting malformed or out-of-order logs."""

    events = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError(f"blank event line at {line_number}")
            try:
                event = StepEvent.model_validate_json(line)
            except Exception as error:
                raise ValueError(f"invalid event at line {line_number}") from error
            if event.sequence != len(events):
                raise ValueError(f"event sequence gap at line {line_number}")
            events.append(event)
    return events


RecordT = TypeVar("RecordT", bound=BaseModel)


def write_record(path: Path, record: BaseModel) -> None:
    """Save a diagnosis request or report without overwriting prior evidence."""

    payload = record.model_dump_json(indent=2) + "\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(payload)


def read_record(path: Path, record_type: type[RecordT]) -> RecordT:
    return record_type.model_validate_json(path.read_text(encoding="utf-8"))
