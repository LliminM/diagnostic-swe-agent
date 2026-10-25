import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from minisweagent.exceptions import Submitted

from diagnostic_swe import run_task


class FakeModel:
    def __init__(self):
        self.config = SimpleNamespace(model_name="fake")
        self.calls = 0

    def query(self, messages, **kwargs):
        self.calls += 1
        if self.calls == 1:
            return self.format_message(
                role="assistant", content="run check", extra={"actions": [{"command": "check"}]}
            )
        return self.format_message(role="assistant", content="submit", extra={"actions": [{"command": "submit"}]})

    def format_message(self, **kwargs):
        return kwargs

    def format_observation_messages(self, message, outputs, template_vars=None):
        return [self.format_message(role="user", content=str(item["output"])) for item in outputs]

    def get_template_vars(self, **kwargs):
        return kwargs

    def serialize(self):
        return {}


class FakeEnvironment:
    def __init__(self):
        self.config = SimpleNamespace(cwd="/fake")

    def execute(self, action, cwd="", **kwargs):
        if action["command"] == "submit":
            raise Submitted(
                {"role": "exit", "content": "patch", "extra": {"exit_status": "Submitted", "submission": "patch"}}
            )
        return {"output": "failed test detail", "returncode": 1, "exception_info": ""}

    def get_template_vars(self, **kwargs):
        return kwargs

    def serialize(self):
        return {}


class RunnerTests(unittest.TestCase):
    def test_records_raw_failure_and_submission_without_changing_agent_flow(self):
        with tempfile.TemporaryDirectory() as temp:
            result = run_task(
                task_id="repo__issue-1",
                task="fix this",
                model=FakeModel(),
                environment=FakeEnvironment(),
                system_template="system",
                instance_template="{{task}}",
                output_root=Path(temp),
            )
            self.assertEqual(result.exit_status, "Submitted")
            self.assertEqual(result.submission, "patch")

            events = [json.loads(line) for line in result.events_path.read_text().splitlines()]
            self.assertEqual([event["sequence"] for event in events], list(range(len(events))))
            self.assertEqual(
                [event["kind"] for event in events],
                ["run_started", "model_response", "action_result", "model_response", "action_exception", "run_finished"],
            )
            self.assertEqual(events[2]["data"]["result"]["returncode"], 1)
            self.assertEqual(events[2]["data"]["result"]["output"], "failed test detail")
            self.assertEqual(events[2]["data"]["observation"]["outcome"], "command_failure")
            self.assertEqual(events[2]["data"]["observation"]["code_change"]["state"], "unknown")
            self.assertEqual(events[4]["data"]["type"], "Submitted")
            self.assertEqual(events[4]["data"]["observation"]["outcome"], "submitted")
            self.assertEqual([event["turn"] for event in events[1:5]], [1, 1, 2, 2])

            trajectory = json.loads(result.trajectory_path.read_text())
            self.assertEqual(trajectory["info"]["exit_status"], "Submitted")
            self.assertEqual(trajectory["messages"][-1]["extra"]["submission"], "patch")

    def test_rejects_unsafe_or_existing_run_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            arguments = dict(
                task="task", model=FakeModel(), environment=FakeEnvironment(),
                system_template="system", instance_template="{{task}}", output_root=Path(temp),
            )
            with self.assertRaises(ValueError):
                run_task(task_id="../outside", **arguments)
            run_task(task_id="safe", **arguments)
            with self.assertRaises(FileExistsError):
                run_task(task_id="safe", **arguments)


if __name__ == "__main__":
    unittest.main()
