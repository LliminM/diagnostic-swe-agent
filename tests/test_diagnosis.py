import json
import subprocess
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from diagnostic_swe.diagnosis import DiagnosticWorkflow, LiteLLMDiagnosticModel
from diagnostic_swe.diagnosis.classification import candidate_categories
from diagnostic_swe.diagnosis.compatibility import detect_legacy_pytest_setup
from diagnostic_swe.diagnosis.output_gap import has_unverified_output_suppression
from diagnostic_swe.diagnosis.probes import ProbeResult, ReadOnlyProbes
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


class LegacyPytestEnvironment(FakeEnvironment):
    def execute(self, action, cwd=""):
        command = action["command"]
        self.commands.append((command, cwd))
        if command == "python -m pytest --version":
            return {"returncode": 0, "output": "pytest 8.3.5\n", "exception_info": ""}
        if command.startswith("git grep"):
            return {"returncode": 0, "output": (
                "pydicom/tests/test_valuerep.py:209:class TestBadValueRead:\n"
                "pydicom/tests/test_valuerep.py-213-    def setup(self):\n"
                "pydicom/tests/test_valuerep.py-217-        self.tag = TagLike()\n"
            ), "exception_info": ""}
        return {"returncode": 0, "output": " M pydicom/valuerep.py\n", "exception_info": ""}


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


def legacy_pytest_request():
    failed = StepEvent(
        sequence=2, turn=2, timestamp=datetime.now(timezone.utc),
        kind=EventKind.ACTION_RESULT, data={
            "observation": {"command": "python -m pytest pydicom/tests/test_valuerep.py -q"},
            "result": {"output": (
                "E AttributeError: 'TestBadValueRead' object has no attribute 'tag'\n"
                "FAILED pydicom/tests/test_valuerep.py::TestBadValueRead::test_read_bad_value_in_VR_default\n"
            )},
        },
    )
    return DiagnosisRequest(
        request_id="diag-legacy", task_id="pydicom__pydicom-1139",
        problem_statement="Make PersonName iterable",
        trigger=TriggerDecision(reason=TriggerReason.POST_EDIT_TEST_FAILURE,
                                evidence_event_sequences=[2], explanation="Test failed after edit"),
        recent_events=[failed], workspace_diff="diff --git a/pydicom/valuerep.py b/pydicom/valuerep.py",
    )


class DiagnosisWorkflowTests(unittest.TestCase):
    def test_output_suppression_review_needs_task_and_added_warning_only_patch(self):
        task = "Deprecation warnings from numpy. There should be no future warnings."
        warning_only = ("diff --git a/raw_building.py b/raw_building.py\n"
                        "+with warnings.catch_warnings():\n"
                        "+    warnings.simplefilter(\"ignore\")\n"
                        "+    getattr(module, name)\n")
        self.assertTrue(has_unverified_output_suppression(task, warning_only))
        self.assertFalse(has_unverified_output_suppression(
            "Make PersonName iterable", warning_only))
        self.assertFalse(has_unverified_output_suppression(
            task, warning_only + "+with redirect_stdout(buffer), redirect_stderr(buffer):\n"))
        self.assertFalse(has_unverified_output_suppression(
            task, "-with warnings.catch_warnings():\n+return value\n"))

    def test_pytest_8_legacy_setup_is_diagnosed_as_environment_with_source_evidence(self):
        model = FakeModel()
        environment = LegacyPytestEnvironment()
        report = DiagnosticWorkflow(model, environment).diagnose(legacy_pytest_request())
        self.assertEqual(report.category, FailureCategory.ENVIRONMENT)
        self.assertEqual(report.recovery_mode, RecoveryMode.FEEDBACK)
        self.assertIn("pytest", report.recommended_action)
        self.assertEqual({item.source for item in report.evidence}, {"event", "tool", "diff"})
        self.assertEqual(model.calls, [])
        self.assertTrue(any(command.startswith("git grep") for command, _ in environment.commands))

    def test_legacy_setup_detection_requires_pytest_8_and_legacy_method(self):
        request = legacy_pytest_request()
        version = ProbeResult("pytest-version", "python -m pytest --version", "pytest 7.4.4", 0, "")
        source = ProbeResult("failed-test-source-1", "git grep ...",
                             "class TestBadValueRead:\n    def setup(self):\n"
                             "        self.tag = TagLike()\n", 0, "")
        self.assertIsNone(detect_legacy_pytest_setup(request, [version, source]))
        self.assertIsNone(detect_legacy_pytest_setup(request, [
            ProbeResult("pytest-version", "python -m pytest --version", "pytest 8.3.5", 0, ""),
            ProbeResult("failed-test-source-1", "git grep ...",
                        "class TestBadValueRead:\n    def setup_method(self):\n"
                        "        self.tag = TagLike()\n", 0, ""),
        ]))
        self.assertIsNone(detect_legacy_pytest_setup(request, [
            ProbeResult("pytest-version", "python -m pytest --version", "pytest 8.3.5", 0, ""),
            ProbeResult("failed-test-source-1", "git grep ...",
                        "class TestBadValueRead:\n    def setup(self):\n"
                        "        self.other = TagLike()\n", 0, ""),
        ]))
        self.assertIsNone(detect_legacy_pytest_setup(request, [
            ProbeResult("pytest-version", "python -m pytest --version", "pytest 8.3.5", 0, ""),
            ProbeResult("failed-test-source-1", "git grep ...",
                        "class TestBadValueRead:\n    def setup(self):\n"
                        "        self.other = TagLike()\n"
                        "    def test_case(self):\n        self.tag = TagLike()\n", 0, ""),
        ]))

    def test_failed_test_probe_rejects_parent_paths(self):
        candidate = legacy_pytest_request()
        event = candidate.recent_events[0].model_copy(update={"data": {
            "result": {"output": "FAILED ../../escape.py::TestBadValueRead::test_case\n"},
        }})
        unsafe = candidate.model_copy(update={"recent_events": [event]})
        environment = LegacyPytestEnvironment()
        self.assertEqual(ReadOnlyProbes(environment).inspect_failed_tests(unsafe), [])
        self.assertEqual(environment.commands, [])

    def test_failed_test_source_probe_works_with_real_git_grep_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "pydicom/tests/test_valuerep.py"
            source.parent.mkdir(parents=True)
            source.write_text("class TestBadValueRead:\n"
                              "    def setup(self):\n"
                              "        self.tag = object()\n"
                              "    def test_case(self):\n"
                              "        assert self.tag\n", encoding="utf-8")
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "add", "."], check=True)

            class GitEnvironment:
                def execute(self, action, cwd=""):
                    if action["command"] == "python -m pytest --version":
                        return {"returncode": 0, "output": "pytest 8.3.5", "exception_info": ""}
                    done = subprocess.run(action["command"], shell=True, cwd=root,
                                          capture_output=True, text=True)
                    return {"returncode": done.returncode, "output": done.stdout,
                            "exception_info": done.stderr}

            candidate = legacy_pytest_request()
            probes = ReadOnlyProbes(GitEnvironment()).inspect_failed_tests(candidate)
            self.assertIn("def setup(self)", probes[1].output)
            self.assertIsNotNone(detect_legacy_pytest_setup(candidate, probes))

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
        self.assertEqual(model.calls[0][1]["workspace_diff"], request().workspace_diff)

    def test_post_edit_test_failure_can_be_classified_as_environment(self):
        candidates = candidate_categories(request(TriggerReason.POST_EDIT_TEST_FAILURE))
        self.assertIn(FailureCategory.ENVIRONMENT, candidates)
        self.assertIn(FailureCategory.IMPLEMENTATION, candidates)

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
