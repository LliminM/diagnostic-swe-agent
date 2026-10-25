"""Snapshot and restore a clean, isolated Git task checkout."""

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .storage import read_record, write_record


_SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_COMMIT = re.compile(r"[0-9a-f]{40,64}\Z")


class WorkspaceError(RuntimeError):
    """The checkout cannot safely be snapshotted or restored."""


class WorkspaceSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    repo_root: str
    base_commit: str = Field(pattern=r"^[0-9a-f]{40,64}$")
    captured_at: datetime


@dataclass(frozen=True)
class CandidatePatch:
    base_commit: str
    patch_path: Path
    patch: str
    untracked_paths: tuple[str, ...]


class WorkspaceManager:
    """Operate through the same mini-SWE-agent environment as the coding agent.

    Restore is allowed only for an explicitly isolated, disposable checkout.
    The initial snapshot requires a clean Git worktree. Ignored files are outside
    this snapshot's scope.
    """

    def __init__(
        self,
        environment: Any,
        artifact_dir: Path,
        *,
        cwd: str = "",
        isolated_workspace: bool = False,
    ):
        self.environment = environment
        self.artifact_dir = Path(artifact_dir)
        self.cwd = cwd
        self.isolated_workspace = isolated_workspace

    def _run(self, command: str) -> str:
        result = self.environment.execute({"command": command}, cwd=self.cwd)
        if result.get("returncode") != 0 or result.get("exception_info"):
            raise WorkspaceError(f"workspace command failed: {command}: {result.get('output', '')}")
        return str(result.get("output", ""))

    def _repo_root(self) -> str:
        root = self._run("git rev-parse --show-toplevel").strip()
        if not root.startswith("/"):
            raise WorkspaceError("Git repository root must be absolute")
        return root

    def _check_artifact_location(self, repo_root: str) -> None:
        if self.artifact_dir.resolve().is_relative_to(Path(repo_root).resolve()):
            raise WorkspaceError("artifact_dir must be outside the task repository")

    def snapshot(self) -> WorkspaceSnapshot:
        """Persist the clean starting commit before the agent edits any files."""

        root = self._repo_root()
        self._check_artifact_location(root)
        status = self._run("git status --porcelain=v1 -z --untracked-files=all")
        if status:
            raise WorkspaceError("initial Git worktree must be clean")
        commit = self._run("git rev-parse HEAD").strip()
        if not _COMMIT.fullmatch(commit):
            raise WorkspaceError("Git returned an invalid commit ID")
        snapshot = WorkspaceSnapshot(
            repo_root=root, base_commit=commit, captured_at=datetime.now(timezone.utc)
        )
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        write_record(self.artifact_dir / "initial_snapshot.json", snapshot)
        return snapshot

    def load_snapshot(self) -> WorkspaceSnapshot:
        return read_record(self.artifact_dir / "initial_snapshot.json", WorkspaceSnapshot)

    def export_candidate_patch(self, name: str = "candidate") -> CandidatePatch:
        """Save tracked-file changes as a patch and report non-ignored untracked paths."""

        if not _SAFE_NAME.fullmatch(name) or name in {".", ".."}:
            raise ValueError("patch name must be one safe path component")
        snapshot = self.load_snapshot()
        if self._repo_root() != snapshot.repo_root:
            raise WorkspaceError("current repository differs from the saved snapshot")
        patch = self._run(f"git diff --no-ext-diff --binary {snapshot.base_commit}")
        status = self._run("git status --porcelain=v1 -z --untracked-files=all")
        untracked = tuple(sorted(
            entry[3:] for entry in status.split("\0") if entry.startswith("?? ")
        ))
        patch_path = self.artifact_dir / f"{name}.patch"
        with patch_path.open("x", encoding="utf-8") as stream:
            stream.write(patch)
        return CandidatePatch(snapshot.base_commit, patch_path, patch, untracked)

    def restore(self, snapshot: WorkspaceSnapshot | None = None) -> None:
        """Reset tracked files and remove non-ignored untracked files in the isolated checkout."""

        if not self.isolated_workspace:
            raise PermissionError("restore requires isolated_workspace=True")
        saved_snapshot = self.load_snapshot()
        snapshot = snapshot or saved_snapshot
        if snapshot.repo_root != saved_snapshot.repo_root or snapshot.base_commit != saved_snapshot.base_commit:
            raise WorkspaceError("restore target differs from the saved initial snapshot")
        if not _COMMIT.fullmatch(snapshot.base_commit):
            raise WorkspaceError("snapshot contains an invalid commit ID")
        if self._repo_root() != snapshot.repo_root:
            raise WorkspaceError("current repository differs from the saved snapshot")
        if self._run(f"git cat-file -t {snapshot.base_commit}").strip() != "commit":
            raise WorkspaceError("saved base commit is unavailable")

        self._run(f"git reset --hard {snapshot.base_commit}")
        self._run("git clean -fd")
        if self._run("git rev-parse HEAD").strip() != snapshot.base_commit:
            raise WorkspaceError("restore did not return to the saved commit")
        if self._run("git status --porcelain=v1 -z --untracked-files=all"):
            raise WorkspaceError("worktree is still dirty after restore")
