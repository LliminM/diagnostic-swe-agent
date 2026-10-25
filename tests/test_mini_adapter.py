import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

from minisweagent.exceptions import Submitted

from diagnostic_swe import DiagnosticAgent, SubmissionDecision, run_task
from diagnostic_swe.recording import RecordingEnvironment, RecordingModel
from diagnostic_swe.schemas import EventKind
from diagnostic_swe.storage import EventJournal, read_events


class ScriptedModel:
    def __init__(self, commands):
        self.config = SimpleNamespace(model_name="scripted")
        self.commands = iter(commands)
        self.seen_messages = []
        self.queried = threading.Event()

    def query(self, messages, **kwargs):
        self.seen_messages.append(list(messages))
        self.queried.set()
        command = next(self.commands)
        return self.format_message(
            role="assistant", content=command, extra={"actions": [{"command": command}]}
        )

    def format_message(self, **kwargs):
        return kwargs

    def format_observation_messages(self, message, outputs, template_vars=None):
        return [self.format_message(role="user", content=result["output"]) for result in outputs]

    def get_template_vars(self, **kwargs):
        return kwargs

    def serialize(self):
        return {}


class ScriptedEnvironment:
    def __init__(self):
        self.config = SimpleNamespace(cwd="/fake")

    def execute(self, action, cwd="", **kwargs):
        command = action["command"]
        if command.startswith("git "):
            return {"output": "not a git repo", "returncode": 128, "exception_info": ""}
        if command == "submit":
            raise Submitted(
                {"role": "exit", "content": "patch", "extra": {"exit_status": "Submitted", "submission": "patch"}}
            )
        return {"output": "checked", "returncode": 0, "exception_info": ""}

    def get_template_vars(self, **kwargs):
        return kwargs

    def serialize(self):
        return {}


class MiniAdapterTests(unittest.TestCase):
    def test_step_hook_observes_events_and_feedback_reaches_next_query(self):
        model = ScriptedModel(["check", "submit"])
        observed = []

        def on_step(events, agent):
            observed.append(tuple(event.kind for event in events))
            if any(event.kind is EventKind.ACTION_RESULT for event in events):
                agent.inject_feedback("Inspect the parser edge case")

        with tempfile.TemporaryDirectory() as temp:
            result = run_task(
                task_id="case-1", task="fix parser", model=model, environment=ScriptedEnvironment(),
                system_template="system", instance_template="{{task}}", output_root=Path(temp),
                on_step=on_step,
            )
            self.assertEqual(result.exit_status, "Submitted")
            self.assertTrue(any(EventKind.ACTION_RESULT in batch for batch in observed))
            self.assertTrue(
                any(message.get("content") == "Inspect the parser edge case" for message in model.seen_messages[1])
            )
            self.assertIn(EventKind.FEEDBACK_INJECTED, [event.kind for event in read_events(result.events_path)])

    def test_rejected_submission_becomes_tool_feedback_then_can_be_accepted(self):
        model = ScriptedModel(["submit", "submit"])
        decisions = []

        def on_submission(patch, agent):
            decisions.append(patch)
            if len(decisions) == 1:
                return SubmissionDecision(accept=False, feedback="The patch failed the focused test")
            return SubmissionDecision(accept=True)

        with tempfile.TemporaryDirectory() as temp:
            result = run_task(
                task_id="case-2", task="fix parser", model=model, environment=ScriptedEnvironment(),
                system_template="system", instance_template="{{task}}", output_root=Path(temp),
                on_submission=on_submission,
            )
            self.assertEqual(result.exit_status, "Submitted")
            self.assertEqual(decisions, ["patch", "patch"])
            self.assertTrue(
                any(message.get("content") == "The patch failed the focused test" for message in model.seen_messages[1])
            )
            kinds = [event.kind for event in read_events(result.events_path)]
            self.assertIn(EventKind.SUBMISSION_REJECTED, kinds)
            self.assertIn(EventKind.SUBMISSION_ACCEPTED, kinds)

    def test_pause_blocks_next_query_until_resumed(self):
        with tempfile.TemporaryDirectory() as temp:
            journal = EventJournal(Path(temp) / "events.jsonl")
            model = ScriptedModel(["submit"])
            agent = DiagnosticAgent(
                RecordingModel(model, journal), RecordingEnvironment(ScriptedEnvironment(), journal),
                journal=journal, system_template="system", instance_template="{{task}}",
                step_limit=0, cost_limit=0,
            )
            result = []
            agent.pause()
            worker = threading.Thread(target=lambda: result.append(agent.run("fix parser")), daemon=True)
            worker.start()
            self.assertFalse(model.queried.wait(timeout=0.1))
            agent.inject_feedback("Diagnostic context")
            agent.resume()
            worker.join(timeout=2)
            self.assertFalse(worker.is_alive())
            self.assertEqual(result[0]["exit_status"], "Submitted")
            self.assertTrue(any(m.get("content") == "Diagnostic context" for m in model.seen_messages[0]))
            kinds = [event.kind for event in read_events(journal.path)]
            self.assertLess(kinds.index(EventKind.PAUSED), kinds.index(EventKind.RESUMED))

    def test_submission_hook_failure_is_recorded_and_does_not_end_as_exception(self):
        def broken_hook(patch, agent):
            raise RuntimeError("verification service unavailable")

        with tempfile.TemporaryDirectory() as temp:
            with self.assertLogs("diagnostic_swe.mini_adapter", level="ERROR"):
                result = run_task(
                    task_id="case-3", task="fix parser", model=ScriptedModel(["submit"]),
                    environment=ScriptedEnvironment(), system_template="system",
                    instance_template="{{task}}", output_root=Path(temp), on_submission=broken_hook,
                )
            self.assertEqual(result.exit_status, "Submitted")
            kinds = [event.kind for event in read_events(result.events_path)]
            self.assertIn(EventKind.ADAPTER_EXCEPTION, kinds)
            self.assertIn(EventKind.SUBMISSION_ACCEPTED, kinds)

    def test_restart_request_exits_before_another_model_query(self):
        model = ScriptedModel(["check", "submit"])

        def on_step(events, agent):
            if any(event.kind is EventKind.ACTION_RESULT for event in events):
                agent.stop_for_restart()

        with tempfile.TemporaryDirectory() as temp:
            result = run_task(
                task_id="case-restart", task="fix parser", model=model,
                environment=ScriptedEnvironment(), system_template="system",
                instance_template="{{task}}", output_root=Path(temp), on_step=on_step,
            )
            self.assertEqual(result.exit_status, "DiagnosticRestart")
            self.assertEqual(len(model.seen_messages), 1)
            kinds = [event.kind for event in read_events(result.events_path)]
            self.assertIn(EventKind.RESTART_REQUESTED, kinds)


if __name__ == "__main__":
    unittest.main()
