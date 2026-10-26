import subprocess
import tempfile
import unittest
from pathlib import Path

from minisweagent.environments.local import LocalEnvironment
from minisweagent.exceptions import Submitted

from diagnostic_swe.observation import ChangeState, Outcome, record_step
from diagnostic_swe.recording import RecordingEnvironment
from diagnostic_swe.schemas import EventKind
from diagnostic_swe.storage import EventJournal, read_events


class ObservationTests(unittest.TestCase):
    def test_normalization_uses_raw_output_without_false_failure_on_success(self):
        failure = record_step(
            {"command": "pytest -q"},
            {"output": "\x1b[31mFAILED tests/test_parser.py::test_empty - AssertionError\x1b[0m\n", "returncode": 1,
             "exception_info": ""},
            None, None,
        )
        self.assertEqual(failure.outcome, Outcome.COMMAND_FAILURE)
        self.assertTrue(failure.test_failed)
        self.assertEqual(failure.error_signature, "FAILED tests/test_parser.py::test_empty - AssertionError")
        self.assertEqual(failure.code_change.state, ChangeState.UNKNOWN)

        success = record_step(
            {"command": "cat log.txt"},
            {"output": "FAILED was written in an old log", "returncode": 0, "exception_info": ""},
            None, None,
        )
        self.assertEqual(success.outcome, Outcome.SUCCESS)
        self.assertFalse(success.test_failed)
        self.assertIsNone(success.error_signature)

    def test_masked_pytest_failure_is_recorded_without_rewriting_shell_outcome(self):
        timeout = record_step(
            {"command": "timeout 30s python -m pytest -q tests/test_grid.py; echo \"pytest exit status: $?\""},
            {"output": "pytest exit status: 124\n", "returncode": 0, "exception_info": ""},
            None, None,
        )
        self.assertEqual(timeout.outcome, Outcome.SUCCESS)
        self.assertTrue(timeout.test_failed)
        self.assertEqual(timeout.error_signature, "pytest exit status: 124")

        piped = record_step(
            {"command": "pytest -q tests/test_grid.py | tail -20"},
            {"output": "FAILED tests/test_grid.py::test_case\n1 failed in 0.2s\n",
             "returncode": 0, "exception_info": ""},
            None, None,
        )
        self.assertTrue(piped.test_failed)

        old_log = record_step(
            {"command": "cat old_pytest.log"},
            {"output": "FAILED tests/test_grid.py::test_case\n", "returncode": 0,
             "exception_info": ""},
            None, None,
        )
        self.assertFalse(old_log.test_failed)

    def test_git_probe_detects_edits_and_keeps_model_visible_result_unchanged(self):
        with tempfile.TemporaryDirectory() as temp:
            repo = Path(temp) / "repo"
            repo.mkdir()
            subprocess.run(["git", "init", "-q", str(repo)], check=True)
            (repo / "code.py").write_text("before\n")
            subprocess.run(["git", "-C", str(repo), "add", "code.py"], check=True)
            subprocess.run(
                ["git", "-C", str(repo), "-c", "user.name=Test", "-c", "user.email=test@example.com",
                 "commit", "-qm", "initial"], check=True,
            )
            journal = EventJournal(Path(temp) / "events.jsonl")
            environment = RecordingEnvironment(LocalEnvironment(cwd=str(repo)), journal)

            result = environment.execute({"command": "printf 'after\\n' > code.py"})
            self.assertEqual(result, {"output": "", "returncode": 0, "exception_info": ""})
            second = environment.execute({"command": "cat code.py"})
            self.assertEqual(second["output"], "after\n")

            events = read_events(journal.path)
            self.assertEqual(len(events), 2)
            self.assertEqual(events[0].kind, EventKind.ACTION_RESULT)
            first_change = events[0].data["observation"]["code_change"]
            self.assertEqual(first_change["state"], "changed")
            self.assertEqual(first_change["dirty_files_after"], ["code.py"])
            self.assertIn("+after", first_change["diff_after"])
            self.assertEqual(events[1].data["observation"]["code_change"]["state"], "unchanged")

            with self.assertRaises(Submitted):
                environment.execute({"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"})
            submit_event = read_events(journal.path)[-1]
            self.assertEqual(submit_event.data["observation"]["outcome"], "submitted")
            self.assertEqual(submit_event.data["observation"]["code_change"]["state"], "unknown")


if __name__ == "__main__":
    unittest.main()
