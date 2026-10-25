import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from minisweagent.environments.local import LocalEnvironment

from diagnostic_swe.runner import run_diagnostic_task
from diagnostic_swe.schemas import EventKind
from diagnostic_swe.storage import read_events
from diagnostic_swe.validation import SubmissionCheck, SubmissionStatus


SUBMIT = "printf 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\\nready\\n'"


class ScriptedModel:
    def __init__(self, steps):
        self.steps = iter(steps)
        self.config = SimpleNamespace(model_name="scripted")
        self.seen_messages = []

    def query(self, messages, **kwargs):
        self.seen_messages.append(list(messages))
        content, command = next(self.steps)
        return self.format_message(
            role="assistant", content=content,
            extra={"actions": [{"command": command}]},
        )

    def format_message(self, **kwargs):
        return kwargs

    def format_observation_messages(self, message, outputs, template_vars=None):
        return [self.format_message(role="user", content=item["output"]) for item in outputs]

    def get_template_vars(self, **kwargs):
        return kwargs

    def serialize(self):
        return {}


class ScriptedDiagnosisModel:
    def __init__(self, mode="feedback", category="control_flow"):
        self.mode = mode
        self.category = category
        self.calls = []

    def complete_json(self, system_prompt, user_prompt):
        self.calls.append(json.loads(user_prompt))
        if "candidate_categories" in self.calls[-1]:
            return {"category": self.category, "rationale": "Observed failed action"}
        return {
            "hypothesis": "The attempted approach is stuck",
            "evidence_ids": ["event:1"],
            "recommended_action": "Inspect the failure and rerun the check.",
            "recovery_mode": self.mode,
        }


def make_repo(root):
    repo = root / "repo"
    repo.mkdir()
    subprocess.run(["git", "-C", str(repo), "init", "-q"], check=True)
    (repo / "module.py").write_text("before\n")
    subprocess.run(["git", "-C", str(repo), "add", "module.py"], check=True)
    subprocess.run([
        "git", "-C", str(repo), "-c", "user.name=Test", "-c",
        "user.email=test@example.com", "commit", "-qm", "initial",
    ], check=True)
    return repo


def run(root, repo, factory, diagnosis, *, isolated=True):
    return run_diagnostic_task(
        task_id="repo__issue", task="Fix module", model_factory=factory,
        environment=LocalEnvironment(cwd=str(repo)), diagnosis_model=diagnosis,
        system_template="system", instance_template="{{task}}", output_root=root / "runs",
        test_command="true", isolated_workspace=isolated,
    )


class DiagnosticRunnerTests(unittest.TestCase):
    def test_feedback_diagnosis_and_submission_are_persisted(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            repo = make_repo(root)
            model = ScriptedModel([
                ("[[REQUEST_DIAGNOSIS]]", "false"),
                ("apply fix", "printf 'fixed\\n' > module.py"),
                ("submit", SUBMIT),
            ])
            diagnosis = ScriptedDiagnosisModel()
            result = run(root, repo, lambda: model, diagnosis)

            self.assertEqual(result.exit_status, "Submitted")
            self.assertEqual(len(result.attempts), 1)
            self.assertEqual(len(result.diagnosis_report_paths), 1)
            self.assertIn("+fixed", result.final_patch_path.read_text())
            self.assertTrue(any("[独立诊断]" in message.get("content", "")
                                for message in model.seen_messages[1]))
            self.assertEqual(SubmissionCheck.model_validate_json(
                result.submission_check_paths[0].read_text()).status, SubmissionStatus.PASS)
            self.assertEqual(len(result.recovery_check_paths), 1)
            self.assertTrue((result.run_dir / "manifest.json").exists())
            kinds = [event.kind for event in read_events(result.attempts[0].events_path)]
            self.assertIn(EventKind.DIAGNOSIS_REPORTED, kinds)
            self.assertIn(EventKind.RECOVERY_APPLIED, kinds)
            self.assertIn(EventKind.SUBMISSION_CHECKED, kinds)
            self.assertEqual(len(diagnosis.calls), 2)

    def test_restart_ends_old_attempt_and_uses_fresh_model_context(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            repo = make_repo(root)
            first = ScriptedModel([
                ("[[REQUEST_DIAGNOSIS]]", "printf 'broken\\n' > module.py"),
            ])
            second = ScriptedModel([
                ("fix", "printf 'fixed\\n' > module.py"),
                ("submit", SUBMIT),
            ])
            models = iter([first, second])
            result = run(root, repo, lambda: next(models),
                         ScriptedDiagnosisModel(mode="restart", category="implementation"))

            self.assertEqual([item.exit_status for item in result.attempts],
                             ["DiagnosticRestart", "Submitted"])
            self.assertEqual(len(first.seen_messages), 1)
            self.assertTrue(any("[独立诊断]" in message.get("content", "")
                                for message in second.seen_messages[0]))
            self.assertIn("+fixed", result.final_patch_path.read_text())
            self.assertNotIn("+broken", result.final_patch_path.read_text())
            self.assertEqual(len(result.recovery_check_paths), 1)
            self.assertIn(EventKind.RESTART_REQUESTED,
                          [event.kind for event in read_events(result.attempts[0].events_path)])

    def test_failed_submission_is_returned_to_agent_then_rechecked(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            repo = make_repo(root)
            model = ScriptedModel([
                ("bad edit", "printf 'bad   \\n' > module.py"),
                ("submit", SUBMIT),
                ("correct edit", "printf 'fixed\\n' > module.py"),
                ("submit", SUBMIT),
            ])
            result = run(root, repo, lambda: model, ScriptedDiagnosisModel())
            statuses = [SubmissionCheck.model_validate_json(path.read_text()).status
                        for path in result.submission_check_paths]
            self.assertEqual(statuses, [SubmissionStatus.FAIL, SubmissionStatus.PASS])
            self.assertEqual(result.exit_status, "Submitted")
            self.assertTrue(any("提交前验证失败" in message.get("content", "")
                                for message in model.seen_messages[2]))

    def test_existing_output_directory_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            repo = make_repo(root)
            model = ScriptedModel([("edit", "printf 'fixed\\n' > module.py"),
                                   ("submit", SUBMIT)])
            run(root, repo, lambda: model, ScriptedDiagnosisModel())
            with self.assertRaises(FileExistsError):
                run(root, repo, lambda: ScriptedModel([]), ScriptedDiagnosisModel())

    def test_untracked_file_blocks_restart_and_falls_back_to_feedback(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            repo = make_repo(root)
            model = ScriptedModel([
                ("[[REQUEST_DIAGNOSIS]]", "printf 'new\\n' > new.py"),
                ("keep new file", "git add new.py"),
                ("submit", SUBMIT),
            ])
            result = run(root, repo, lambda: model,
                         ScriptedDiagnosisModel(mode="restart", category="implementation"))
            self.assertEqual(len(result.attempts), 1)
            self.assertEqual(result.exit_status, "Submitted")
            self.assertIn("new.py", result.final_patch_path.read_text())
            kinds = [event.kind for event in read_events(result.attempts[0].events_path)]
            self.assertIn(EventKind.RECOVERY_FAILED, kinds)
            self.assertIn(EventKind.RECOVERY_APPLIED, kinds)
            self.assertNotIn(EventKind.RESTART_REQUESTED, kinds)

    def test_same_incident_and_patch_are_not_diagnosed_twice(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            repo = make_repo(root)
            model = ScriptedModel([
                ("[[REQUEST_DIAGNOSIS]]", "false"),
                ("[[REQUEST_DIAGNOSIS]]", "false"),
                ("fix", "printf 'fixed\\n' > module.py"),
                ("submit", SUBMIT),
            ])
            diagnosis = ScriptedDiagnosisModel()
            result = run(root, repo, lambda: model, diagnosis)
            self.assertEqual(len(result.diagnosis_report_paths), 1)
            self.assertEqual(len(diagnosis.calls), 2)
            self.assertIn(EventKind.DIAGNOSIS_SKIPPED,
                          [event.kind for event in read_events(result.attempts[0].events_path)])


if __name__ == "__main__":
    unittest.main()
