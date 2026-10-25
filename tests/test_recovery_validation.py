import subprocess
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from minisweagent.environments.local import LocalEnvironment

from diagnostic_swe.recovery import RecoveryController, RecoveryError
from diagnostic_swe.schemas import (
    DiagnosisReport, DiagnosisRequest, EventKind, EvidenceRef, FailureCategory,
    RecoveryMode, StepEvent, TriggerDecision, TriggerReason,
)
from diagnostic_swe.validation import (
    CheckStatus, SubmissionStatus, verify_recovery, verify_submission,
)
from diagnostic_swe.workspace import WorkspaceManager


class FakeAgent:
    def __init__(self):
        self.feedback = []
        self.restart_requested = False

    def inject_feedback(self, message):
        self.feedback.append(message)

    def stop_for_restart(self):
        self.restart_requested = True


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], check=True,
                          capture_output=True, text=True).stdout


def repo_and_workspace(root):
    repo = root / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    (repo / "module.py").write_text("before\n")
    git(repo, "add", "module.py")
    git(repo, "-c", "user.name=Test", "-c", "user.email=test@example.com",
        "commit", "-qm", "initial")
    workspace = WorkspaceManager(LocalEnvironment(cwd=str(repo)),
                                 root / "artifacts", isolated_workspace=True)
    snapshot = workspace.snapshot()
    return repo, workspace, snapshot


def action(sequence, *, outcome, signature, command="pytest test_module.py"):
    return StepEvent(
        sequence=sequence, turn=sequence + 1, timestamp=datetime.now(timezone.utc),
        kind=EventKind.ACTION_RESULT,
        data={"observation": {
            "command": command, "outcome": outcome, "returncode": 0 if outcome == "success" else 1,
            "exception_info": "", "output_sha256": "hash", "error_signature": signature,
            "test_failed": outcome != "success",
            "code_change": {"state": "unchanged"},
        }},
    )


def request():
    return DiagnosisRequest(
        request_id="diag-1", task_id="repo__issue", problem_statement="Fix parser",
        trigger=TriggerDecision(reason=TriggerReason.REPEATED_TEST_FAILURE,
                                evidence_event_sequences=[1], explanation="test failed"),
        recent_events=[action(1, outcome="command_failure", signature="AssertionError")],
    )


def report(mode):
    return DiagnosisReport(
        request_id="diag-1", category=FailureCategory.IMPLEMENTATION,
        hypothesis="Wrong empty-input branch",
        evidence=[EvidenceRef(source="event", reference="1", observation="AssertionError")],
        recommended_action="Review the branch and rerun the failing test.", recovery_mode=mode,
    )


class RecoveryTests(unittest.TestCase):
    def test_feedback_keeps_workspace_and_queues_short_message(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            repo, workspace, _ = repo_and_workspace(root)
            (repo / "module.py").write_text("after\n")
            agent = FakeAgent()
            result = RecoveryController(workspace).apply_recovery(
                report=report(RecoveryMode.FEEDBACK), request=request(), agent=agent)
            self.assertIs(result.agent, agent)
            self.assertIsNone(result.saved_candidate)
            self.assertEqual((repo / "module.py").read_text(), "after\n")
            self.assertIn("Wrong empty-input branch", agent.feedback[0])

    def test_restore_preserves_candidate_then_resets_isolated_checkout(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            repo, workspace, snapshot = repo_and_workspace(root)
            (repo / "module.py").write_text("after\n")
            agent = FakeAgent()
            result = RecoveryController(workspace).apply_recovery(
                report=report(RecoveryMode.RESTORE), request=request(), agent=agent)
            self.assertEqual((repo / "module.py").read_text(), "before\n")
            self.assertEqual(git(repo, "rev-parse", "HEAD").strip(), snapshot.base_commit)
            self.assertIn("+after", result.saved_candidate.patch)
            self.assertTrue(result.saved_candidate.patch_path.exists())
            self.assertIn("工作区已恢复", agent.feedback[0])

    def test_restart_returns_fresh_agent_with_feedback(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            repo, workspace, _ = repo_and_workspace(root)
            (repo / "module.py").write_text("after\n")
            old_agent, new_agent = FakeAgent(), FakeAgent()
            result = RecoveryController(workspace).apply_recovery(
                report=report(RecoveryMode.RESTART), request=request(), agent=old_agent,
                restart_factory=lambda: new_agent)
            self.assertIs(result.agent, new_agent)
            self.assertEqual(old_agent.feedback, [])
            self.assertTrue(old_agent.restart_requested)
            self.assertEqual(len(new_agent.feedback), 1)
            self.assertEqual((repo / "module.py").read_text(), "before\n")

    def test_untracked_file_blocks_destructive_recovery(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            repo, workspace, _ = repo_and_workspace(root)
            (repo / "new.py").write_text("important contents\n")
            with self.assertRaisesRegex(RecoveryError, "untracked files"):
                RecoveryController(workspace).apply_recovery(
                    report=report(RecoveryMode.RESTART), request=request(), agent=FakeAgent(),
                    restart_factory=FakeAgent)
            self.assertEqual((repo / "new.py").read_text(), "important contents\n")

    def test_report_must_match_request_and_cite_trigger(self):
        with tempfile.TemporaryDirectory() as temp:
            _, workspace, _ = repo_and_workspace(Path(temp))
            bad = report(RecoveryMode.FEEDBACK).model_copy(
                update={"evidence": [EvidenceRef(source="task", reference="problem_statement",
                                                observation="Fix parser")]})
            with self.assertRaisesRegex(RecoveryError, "triggering event"):
                RecoveryController(workspace).apply_recovery(
                    report=bad, request=request(), agent=FakeAgent())

    def test_failed_restart_factory_leaves_checkout_unchanged(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            repo, workspace, _ = repo_and_workspace(root)
            (repo / "module.py").write_text("after\n")

            def broken_factory():
                raise RuntimeError("agent unavailable")

            with self.assertRaisesRegex(RecoveryError, "agent unavailable"):
                RecoveryController(workspace).apply_recovery(
                    report=report(RecoveryMode.RESTART), request=request(), agent=FakeAgent(),
                    restart_factory=broken_factory)
            self.assertEqual((repo / "module.py").read_text(), "after\n")


class ValidationTests(unittest.TestCase):
    def test_recovery_requires_same_command_retest(self):
        baseline = request()
        unrelated = action(2, outcome="success", signature=None, command="echo done")
        self.assertEqual(verify_recovery(baseline, [unrelated]).status, CheckStatus.INCONCLUSIVE)
        passed = action(3, outcome="success", signature=None)
        self.assertEqual(verify_recovery(baseline, [unrelated, passed]).status, CheckStatus.IMPROVED)
        repeated = action(4, outcome="command_failure", signature="AssertionError")
        self.assertEqual(verify_recovery(baseline, [passed, repeated]).status, CheckStatus.UNCHANGED)
        changed = action(5, outcome="command_failure", signature="ImportError")
        self.assertEqual(verify_recovery(baseline, [changed]).status, CheckStatus.INCONCLUSIVE)
        restarted = action(0, outcome="success", signature=None)
        self.assertEqual(verify_recovery(baseline, [restarted], new_attempt=True).status,
                         CheckStatus.IMPROVED)
        with self.assertRaisesRegex(ValueError, "ordered"):
            verify_recovery(baseline, [passed, passed])

    def test_submission_checks_full_diff_and_configured_test(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            repo, _, snapshot = repo_and_workspace(root)
            (repo / "module.py").write_text("after\n")
            env = LocalEnvironment(cwd=str(repo))
            passed = verify_submission(env, snapshot=snapshot,
                                       test_command="true")
            self.assertEqual(passed.status, SubmissionStatus.PASS)
            failed = verify_submission(env, snapshot=snapshot,
                                       test_command="false")
            self.assertEqual(failed.status, SubmissionStatus.FAIL)
            missing = verify_submission(env, snapshot=snapshot, test_command=None)
            self.assertEqual(missing.status, SubmissionStatus.INCONCLUSIVE)

    def test_submission_rejects_diff_format_errors_before_tests(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            repo, _, snapshot = repo_and_workspace(root)
            (repo / "module.py").write_text("after   \n")
            result = verify_submission(LocalEnvironment(cwd=str(repo)), snapshot=snapshot,
                                       test_command="true")
            self.assertEqual(result.status, SubmissionStatus.FAIL)
            self.assertIn("trailing whitespace", result.diff_check_output)

    def test_submission_rejects_empty_patch_and_untracked_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            repo, _, snapshot = repo_and_workspace(root)
            env = LocalEnvironment(cwd=str(repo))
            self.assertEqual(verify_submission(env, snapshot=snapshot, test_command="true").status,
                             SubmissionStatus.FAIL)
            (repo / "new.py").write_text("new content\n")
            result = verify_submission(env, snapshot=snapshot, test_command="true")
            self.assertEqual(result.status, SubmissionStatus.FAIL)
            self.assertIn("未跟踪", result.reason)

    def test_submission_accepts_only_matching_mini_patch_file(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            repo, _, snapshot = repo_and_workspace(root)
            (repo / "module.py").write_text("after\n")
            patch_text = git(repo, "diff", "--", "module.py")
            (repo / "patch.txt").write_text(patch_text)
            env = LocalEnvironment(cwd=str(repo))
            accepted = verify_submission(env, snapshot=snapshot, test_command="true",
                                         submitted_patch=patch_text)
            self.assertEqual(accepted.status, SubmissionStatus.PASS)
            mismatch = verify_submission(env, snapshot=snapshot, test_command="true",
                                         submitted_patch="different")
            self.assertEqual(mismatch.status, SubmissionStatus.FAIL)
            (repo / "other.txt").write_text("not in patch\n")
            extra_file = verify_submission(env, snapshot=snapshot, test_command="true",
                                           submitted_patch=patch_text)
            self.assertEqual(extra_file.status, SubmissionStatus.FAIL)

    def test_submission_rechecks_workspace_after_test(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            repo, _, snapshot = repo_and_workspace(root)
            (repo / "module.py").write_text("after\n")
            result = verify_submission(LocalEnvironment(cwd=str(repo)), snapshot=snapshot,
                                       test_command="printf 'generated\\n' > generated.py")
            self.assertEqual(result.status, SubmissionStatus.FAIL)
            self.assertIn("测试产生", result.reason)


if __name__ == "__main__":
    unittest.main()
