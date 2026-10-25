import subprocess
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from minisweagent.environments.local import LocalEnvironment

from diagnostic_swe.evidence import build_request
from diagnostic_swe.schemas import (
    DiagnosisReport,
    EventKind,
    FailureCategory,
    StepEvent,
    TriggerDecision,
    TriggerReason,
)
from diagnostic_swe.workspace import CandidatePatch, WorkspaceError, WorkspaceManager


def git(repo: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *arguments], check=True, capture_output=True, text=True,
    )
    return result.stdout


def initialized_repo(parent: Path) -> Path:
    repo = parent / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    (repo / "code.py").write_text("before\n")
    git(repo, "add", "code.py")
    git(repo, "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-qm", "initial")
    return repo


def step_event(sequence: int, kind: EventKind) -> StepEvent:
    return StepEvent(
        sequence=sequence, turn=sequence + 1, timestamp=datetime.now(timezone.utc),
        kind=kind, data={"sequence": sequence},
    )


class WorkspaceEvidenceTests(unittest.TestCase):
    def test_snapshot_export_and_restore_even_after_agent_commit(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            repo = initialized_repo(root)
            manager = WorkspaceManager(
                LocalEnvironment(cwd=str(repo)), root / "artifacts", isolated_workspace=True,
            )
            snapshot = manager.snapshot()
            self.assertEqual(manager.load_snapshot(), snapshot)

            (repo / "code.py").write_text("after\n")
            git(repo, "add", "code.py")
            git(repo, "-c", "user.name=Test", "-c", "user.email=test@example.com",
                "commit", "-qm", "agent edit")
            (repo / "scratch.py").write_text("temporary\n")

            candidate = manager.export_candidate_patch("attempt-1")
            self.assertIn("+after", candidate.patch)
            self.assertEqual(candidate.untracked_paths, ("scratch.py",))
            self.assertEqual(candidate.patch_path.read_text(), candidate.patch)
            self.assertEqual(candidate.base_commit, snapshot.base_commit)

            manager.restore()
            self.assertEqual((repo / "code.py").read_text(), "before\n")
            self.assertFalse((repo / "scratch.py").exists())
            self.assertEqual(git(repo, "rev-parse", "HEAD").strip(), snapshot.base_commit)
            self.assertEqual(git(repo, "status", "--porcelain"), "")
            self.assertTrue(candidate.patch_path.exists())

    def test_dirty_start_and_nonisolated_restore_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            repo = initialized_repo(root)
            manager = WorkspaceManager(LocalEnvironment(cwd=str(repo)), root / "artifacts")
            (repo / "code.py").write_text("already edited\n")
            with self.assertRaisesRegex(WorkspaceError, "must be clean"):
                manager.snapshot()
            self.assertFalse((root / "artifacts" / "initial_snapshot.json").exists())

            (repo / "code.py").write_text("before\n")
            manager.snapshot()
            (repo / "code.py").write_text("new edit\n")
            with self.assertRaisesRegex(PermissionError, "isolated_workspace"):
                manager.restore()
            self.assertEqual((repo / "code.py").read_text(), "new edit\n")
            with self.assertRaises(ValueError):
                manager.export_candidate_patch("../outside")

    def test_artifacts_inside_task_repo_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            repo = initialized_repo(Path(temp))
            manager = WorkspaceManager(LocalEnvironment(cwd=str(repo)), repo / "runs")
            with self.assertRaisesRegex(WorkspaceError, "outside the task repository"):
                manager.snapshot()

    def test_request_contains_current_episode_diff_and_prior_report(self):
        events = [
            step_event(0, EventKind.RUN_STARTED),
            step_event(1, EventKind.ACTION_RESULT),
            step_event(2, EventKind.FEEDBACK_INJECTED),
            step_event(3, EventKind.MODEL_RESPONSE),
            step_event(4, EventKind.ACTION_RESULT),
            step_event(5, EventKind.ACTION_RESULT),
        ]
        trigger = TriggerDecision(
            reason=TriggerReason.REPEATED_TEST_FAILURE,
            evidence_event_sequences=[4, 5], explanation="Same test still fails",
        )
        prior = DiagnosisReport(
            request_id="old", category=FailureCategory.UNKNOWN,
            hypothesis="Earlier hypothesis was inconclusive", evidence=[],
        )
        candidate = CandidatePatch("a" * 40, Path("/tmp/candidate.patch"), "diff --git a/code.py b/code.py", ("new.py",))
        request = build_request(
            request_id="diag-2", task_id="repo__issue", problem_statement="Fix parser",
            trigger=trigger, events=events, candidate=candidate, prior_reports=[prior],
        )
        self.assertEqual([event.sequence for event in request.recent_events], [2, 3, 4, 5])
        self.assertEqual(request.workspace_diff, candidate.patch)
        self.assertEqual(request.untracked_paths, ["new.py"])
        self.assertEqual(request.prior_reports, [prior])

        stale = TriggerDecision(
            reason=TriggerReason.REPEATED_COMMAND_FAILURE,
            evidence_event_sequences=[1], explanation="Old failure",
        )
        with self.assertRaisesRegex(ValueError, "current episode"):
            build_request(
                request_id="diag-3", task_id="repo__issue", problem_statement="Fix parser",
                trigger=stale, events=events, candidate=candidate,
            )


if __name__ == "__main__":
    unittest.main()
