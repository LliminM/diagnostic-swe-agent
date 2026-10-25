import json
import unittest
from datetime import datetime, timezone

from diagnostic_swe.diagnosis import DiagnosticWorkflow, LiteLLMDiagnosticModel
from diagnostic_swe.schemas import (
    DiagnosisRequest, EventKind, FailureCategory, RecoveryMode, StepEvent,
    TriggerDecision, TriggerReason,
)


class FakeModel:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def complete_json(self, system_prompt, user_prompt):
        self.calls.append((system_prompt, json.loads(user_prompt)))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class FakeEnvironment:
    def __init__(self):
        self.commands = []

    def execute(self, action, cwd=""):
        self.commands.append((action["command"], cwd))
        return {"returncode": 0, "output": " M parser.py\n", "exception_info": ""}


def request(reason=TriggerReason.REPEATED_TEST_FAILURE):
    events = [
        StepEvent(sequence=1, turn=1, timestamp=datetime.now(timezone.utc),
                  kind=EventKind.ACTION_RESULT,
                  data={"action": {"command": "pytest tests/test_parser.py"},
                        "result": {"output": "FAILED tests/test_parser.py::test_empty"}}),
        StepEvent(sequence=2, turn=2, timestamp=datetime.now(timezone.utc),
                  kind=EventKind.ACTION_RESULT,
                  data={"action": {"command": "pytest tests/test_parser.py"},
                        "result": {"output": "FAILED tests/test_parser.py::test_empty"}}),
    ]
    return DiagnosisRequest(
        request_id="diag-1", task_id="repo__issue", problem_statement="Handle empty input",
        trigger=TriggerDecision(reason=reason, evidence_event_sequences=[1, 2],
                                explanation="Same test fails after edit"),
        recent_events=events, workspace_diff="diff --git a/parser.py b/parser.py\n+return None",
    )


class DiagnosisWorkflowTests(unittest.TestCase):
    def test_classification_routes_to_read_only_probes_and_grounded_report(self):
        model = FakeModel(
            {"category": "implementation", "rationale": "The edited parser still fails"},
            {"hypothesis": "Empty input branch still returns the wrong value",
             "evidence_ids": ["event:2", "diff", "tool:diff-check"],
             "recommended_action": "Inspect the empty-input branch and rerun the failing test.",
             "recovery_mode": "feedback"},
        )
        environment = FakeEnvironment()
        report = DiagnosticWorkflow(model, environment, cwd="/testbed").diagnose(request())

        self.assertEqual(report.category, FailureCategory.IMPLEMENTATION)
        self.assertEqual(report.recovery_mode, RecoveryMode.FEEDBACK)
        self.assertEqual({item.source for item in report.evidence}, {"event", "diff", "tool"})
        self.assertEqual(environment.commands,
                         [("git diff --check", "/testbed"), ("git status --short", "/testbed")])
        self.assertEqual(len(model.calls), 2)
        self.assertIn("event:2", model.calls[1][1]["evidence"])
        self.assertNotIn("main_agent_messages", model.calls[1][1])

    def test_unsupported_citations_cannot_authorize_recovery(self):
        model = FakeModel(
            {"category": "implementation", "rationale": "test failure"},
            {"hypothesis": "Maybe parser logic", "evidence_ids": ["task", "event:999"],
             "recommended_action": "Restart now", "recovery_mode": "restart"},
        )
        report = DiagnosticWorkflow(model, FakeEnvironment()).diagnose(request())
        self.assertEqual(report.recovery_mode, RecoveryMode.NONE)
        self.assertEqual(report.recommended_action, "")
        self.assertEqual([item.source for item in report.evidence], ["task"])

    def test_category_outside_shortlist_fails_closed(self):
        model = FakeModel({"category": "control_flow", "rationale": "guess"})
        environment = FakeEnvironment()
        report = DiagnosticWorkflow(model, environment).diagnose(request())
        self.assertEqual(report.category, FailureCategory.UNKNOWN)
        self.assertEqual(report.recovery_mode, RecoveryMode.NONE)
        self.assertEqual(environment.commands, [])
        self.assertEqual(len(model.calls), 1)

    def test_model_error_fails_closed_after_classification(self):
        model = FakeModel({"category": "localization", "rationale": "files may be wrong"},
                          ValueError("bad JSON"))
        environment = FakeEnvironment()
        report = DiagnosticWorkflow(model, environment).diagnose(request())
        self.assertEqual(report.category, FailureCategory.UNKNOWN)
        self.assertEqual(report.recovery_mode, RecoveryMode.NONE)
        self.assertEqual(environment.commands, [("git ls-files", "")])

    def test_litellm_model_rejects_shared_messages(self):
        with self.assertRaisesRegex(ValueError, "controlled"):
            LiteLLMDiagnosticModel("provider/model", messages=[{"role": "user", "content": "x"}])


if __name__ == "__main__":
    unittest.main()
