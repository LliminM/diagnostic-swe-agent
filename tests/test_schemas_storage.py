import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError

from diagnostic_swe.schemas import (
    DiagnosisReport,
    DiagnosisRequest,
    EvidenceRef,
    EventKind,
    FailureCategory,
    RecoveryMode,
    StepEvent,
    TriggerDecision,
    TriggerReason,
)
from diagnostic_swe.storage import EventJournal, read_events, read_record, write_record


class SchemaStorageTests(unittest.TestCase):
    def test_event_log_is_lossless_and_validated_when_read(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "events.jsonl"
            journal = EventJournal(path)
            journal.turn = 2
            raw = "错误：测试失败\n" * 3000
            journal.append(
                EventKind.ACTION_RESULT,
                {"action": {"command": "pytest"}, "result": {"output": raw, "returncode": 1}},
            )
            events = read_events(path)
            self.assertEqual(events[0].data["result"]["output"], raw)
            self.assertEqual(events[0].turn, 2)
            self.assertEqual(events[0].schema_version, 1)

            with self.assertRaises(FileExistsError):
                EventJournal(path)

            with path.open("a", encoding="utf-8") as stream:
                stream.write(events[0].model_copy(update={"sequence": 3}).model_dump_json() + "\n")
            with self.assertRaisesRegex(ValueError, "sequence gap"):
                read_events(path)

    def test_request_and_report_round_trip_as_separate_artifacts(self):
        with tempfile.TemporaryDirectory() as temp:
            event = StepEvent(
                sequence=4, turn=2, timestamp=datetime.now(timezone.utc),
                kind=EventKind.ACTION_RESULT, data={"result": {"returncode": 1}},
            )
            request = DiagnosisRequest(
                request_id="diagnosis-1", task_id="repo__issue-1", problem_statement="Fix a failing parser",
                trigger=TriggerDecision(
                    reason=TriggerReason.REPEATED_TEST_FAILURE,
                    evidence_event_sequences=[4], explanation="The same test failed again",
                ),
                recent_events=[event], workspace_diff="diff --git a/parser.py b/parser.py",
            )
            report = DiagnosisReport(
                request_id="diagnosis-1", category=FailureCategory.IMPLEMENTATION,
                hypothesis="The parser misses empty input",
                evidence=[EvidenceRef(source="event", reference="4", observation="The failing test still returns 1")],
                recommended_action="Handle empty input and rerun the focused test",
                recovery_mode=RecoveryMode.FEEDBACK,
            )
            request_path = Path(temp) / "request.json"
            report_path = Path(temp) / "report.json"
            write_record(request_path, request)
            write_record(report_path, report)
            self.assertEqual(read_record(request_path, DiagnosisRequest), request)
            self.assertEqual(read_record(report_path, DiagnosisReport), report)
            with self.assertRaises(FileExistsError):
                write_record(report_path, report)

    def test_request_rejects_missing_trigger_evidence(self):
        event = StepEvent(
            sequence=1, turn=1, timestamp=datetime.now(timezone.utc),
            kind=EventKind.ACTION_RESULT, data={},
        )
        with self.assertRaisesRegex(ValidationError, "trigger evidence"):
            DiagnosisRequest(
                request_id="diagnosis-1", task_id="task-1", problem_statement="Fix bug",
                trigger=TriggerDecision(
                    reason=TriggerReason.TOOL_ERROR,
                    evidence_event_sequences=[2], explanation="Tool failed",
                ),
                recent_events=[event],
            )

    def test_actionable_report_requires_supporting_evidence(self):
        with self.assertRaisesRegex(ValidationError, "requires evidence"):
            DiagnosisReport(
                request_id="diagnosis-1", category=FailureCategory.UNKNOWN,
                hypothesis="Maybe an environment issue", evidence=[],
                recommended_action="Restart the task", recovery_mode=RecoveryMode.RESTART,
            )


if __name__ == "__main__":
    unittest.main()
