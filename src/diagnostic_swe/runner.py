"""Run one mini-SWE-agent task with observable I/O and optional hooks."""

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .diagnosis import DiagnosticWorkflow
from .evidence import build_request
from .mini_adapter import DiagnosticAgent, StepHook, SubmissionDecision, SubmissionHook
from .monitor import FailureMonitor
from .recording import RecordingEnvironment, RecordingModel
from .recovery import RecoveryController, RecoveryError
from .schemas import DiagnosisReport, DiagnosisRequest, EventKind, RecoveryMode, StepEvent, TriggerDecision
from .storage import EventJournal, write_record
from .validation import CheckStatus, SubmissionStatus, verify_recovery, verify_submission
from .workspace import WorkspaceManager


_TASK_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")


@dataclass(frozen=True)
class RunResult:
    task_id: str
    exit_status: str
    submission: str
    run_dir: Path
    trajectory_path: Path
    events_path: Path


def run_task(
    *,
    task_id: str,
    task: str,
    model: Any,
    environment: Any,
    system_template: str,
    instance_template: str,
    output_root: Path,
    on_step: StepHook | None = None,
    on_submission: SubmissionHook | None = None,
) -> RunResult:
    """Create a fresh run; save mini's trajectory and raw boundary events.

    The model, environment and prompts are supplied by the caller. This makes the
    same entry point usable for a fake local task and a SWE-bench instance.
    """
    if not _TASK_ID.fullmatch(task_id) or task_id in {".", ".."}:
        raise ValueError("task_id must be one safe path component")

    run_dir = Path(output_root) / task_id
    run_dir.mkdir(parents=True, exist_ok=False)
    trajectory_path = run_dir / "trajectory.json"
    events_path = run_dir / "events.jsonl"
    journal = EventJournal(events_path)
    journal.append(EventKind.RUN_STARTED, {"task_id": task_id})

    agent = DiagnosticAgent(
        RecordingModel(model, journal),
        RecordingEnvironment(environment, journal),
        journal=journal,
        on_step=on_step,
        on_submission=on_submission,
        system_template=system_template,
        instance_template=instance_template,
        output_path=trajectory_path,
        # Disable upstream stopping defaults. This project has no budget manager.
        step_limit=0,
        cost_limit=0,
        wall_time_limit_seconds=0,
    )
    try:
        info = agent.run(task)
    except Exception as error:
        journal.append(EventKind.RUN_EXCEPTION, {"type": type(error).__name__, "message": str(error)})
        raise

    exit_status = str(info.get("exit_status", ""))
    submission = str(info.get("submission", ""))
    journal.append(EventKind.RUN_FINISHED, {"exit_status": exit_status})
    return RunResult(task_id, exit_status, submission, run_dir, trajectory_path, events_path)


@dataclass(frozen=True)
class AttemptResult:
    number: int
    exit_status: str
    trajectory_path: Path
    events_path: Path


@dataclass(frozen=True)
class DiagnosticRunResult:
    task_id: str
    exit_status: str
    submission: str
    run_dir: Path
    attempts: tuple[AttemptResult, ...]
    final_patch_path: Path
    diagnosis_request_paths: tuple[Path, ...]
    diagnosis_report_paths: tuple[Path, ...]
    recovery_check_paths: tuple[Path, ...]
    submission_check_paths: tuple[Path, ...]


@dataclass
class _Attempt:
    number: int
    agent: DiagnosticAgent
    journal: EventJournal
    monitor: FailureMonitor
    trajectory_path: Path

    @property
    def events_path(self) -> Path:
        return self.journal.path


@dataclass
class _PendingCheck:
    request: DiagnosisRequest
    source_attempt: int
    path: Path


class _DiagnosticTaskRunner:
    """Own one isolated task checkout and all of its agent attempts."""

    def __init__(
        self,
        *,
        task_id: str,
        task: str,
        model_factory: Callable[[], Any],
        environment: Any,
        diagnosis_model: Any,
        system_template: str,
        instance_template: str,
        output_root: Path,
        test_command: str | None,
        isolated_workspace: bool,
        diagnosis_enabled: bool,
        recovery_enabled: bool,
    ):
        if not _TASK_ID.fullmatch(task_id) or task_id in {".", ".."}:
            raise ValueError("task_id must be one safe path component")
        self.task_id = task_id
        self.task = task
        self.model_factory = model_factory
        self.environment = environment
        self.system_template = system_template
        self.instance_template = instance_template
        self.test_command = test_command
        self.run_dir = Path(output_root) / task_id
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.diagnoses_dir = self.run_dir / "diagnoses"
        self.diagnoses_dir.mkdir()
        self.workspace = WorkspaceManager(
            environment, self.run_dir / "workspace", isolated_workspace=isolated_workspace
        )
        self.snapshot = self.workspace.snapshot()
        self.controller = RecoveryController(self.workspace)
        self.diagnoser = DiagnosticWorkflow(diagnosis_model, environment) if diagnosis_enabled else None
        self.recovery_enabled = recovery_enabled
        self.reports: list[DiagnosisReport] = []
        self.request_paths: list[Path] = []
        self.report_paths: list[Path] = []
        self.recovery_check_paths: list[Path] = []
        self.submission_check_paths: list[Path] = []
        self.pending_checks: list[_PendingCheck] = []
        self.seen_incidents: set[str] = set()
        self.diagnosis_number = 0
        self.submission_number = 0
        self.next_attempt: _Attempt | None = None
        self.fatal_error: Exception | None = None

    def _create_attempt(self, number: int) -> _Attempt:
        model = self.model_factory()
        attempt_dir = self.run_dir / f"attempt-{number:03d}"
        attempt_dir.mkdir(exist_ok=False)
        journal = EventJournal(attempt_dir / "events.jsonl")
        journal.append(EventKind.RUN_STARTED, {"task_id": self.task_id, "attempt": number})
        monitor = FailureMonitor()
        trajectory_path = attempt_dir / "trajectory.json"
        attempt: _Attempt

        def on_step(events: tuple[StepEvent, ...], agent: DiagnosticAgent) -> None:
            self._on_step(attempt, events, agent)

        def on_submission(patch: str, agent: DiagnosticAgent):
            return self._on_submission(attempt, patch)

        agent = DiagnosticAgent(
            RecordingModel(model, journal), RecordingEnvironment(self.environment, journal),
            journal=journal, on_step=on_step, on_submission=on_submission,
            system_template=self.system_template, instance_template=self.instance_template,
            output_path=trajectory_path,
            step_limit=0, cost_limit=0, wall_time_limit_seconds=0,
        )
        attempt = _Attempt(number, agent, journal, monitor, trajectory_path)
        return attempt

    @staticmethod
    def _incident_key(
        trigger: TriggerDecision, events: tuple[StepEvent, ...], patch: str,
    ) -> str:
        evidence = []
        wanted = set(trigger.evidence_event_sequences)
        for event in events:
            if event.sequence not in wanted:
                continue
            observation = event.data.get("observation")
            if isinstance(observation, dict):
                evidence.append({"command": observation.get("command"),
                                 "outcome": observation.get("outcome"),
                                 "error_signature": observation.get("error_signature")})
            else:
                evidence.append({"kind": event.kind.value,
                                 "content": event.data.get("response", {}).get("content", "")})
        payload = {"reason": trigger.reason.value, "evidence": evidence, "patch": patch}
        return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    def _check_recoveries(self, attempt: _Attempt) -> None:
        for pending in list(self.pending_checks):
            check = verify_recovery(
                pending.request, list(attempt.journal.events_since(0)),
                new_attempt=attempt.number != pending.source_attempt,
            )
            if check.status is CheckStatus.INCONCLUSIVE:
                continue
            write_record(pending.path, check)
            self.recovery_check_paths.append(pending.path)
            self.pending_checks.remove(pending)
            attempt.journal.append(EventKind.RECOVERY_CHECKED, {
                "request_id": pending.request.request_id, "status": check.status.value,
                "path": str(pending.path),
            })

    def _on_step(
        self, attempt: _Attempt, events: tuple[StepEvent, ...], agent: DiagnosticAgent,
    ) -> None:
        self._check_recoveries(attempt)
        if self.diagnoser is None:
            return
        trigger = attempt.monitor.observe(events)
        if trigger is None:
            return
        self.diagnosis_number += 1
        label = f"diag-{self.diagnosis_number:03d}"
        try:
            candidate = self.workspace.export_candidate_patch(f"{label}-candidate")
            all_events = attempt.journal.events_since(0)
            incident = self._incident_key(trigger, all_events, candidate.patch)
            if incident in self.seen_incidents:
                attempt.journal.append(EventKind.DIAGNOSIS_SKIPPED, {
                    "reason": "same failure and candidate patch already diagnosed", "label": label,
                })
                attempt.monitor.resume_after_intervention()
                return
            self.seen_incidents.add(incident)
            request = build_request(
                request_id=label, task_id=self.task_id, problem_statement=self.task,
                trigger=trigger, events=all_events, candidate=candidate,
                prior_reports=self.reports,
            )
            request_path = self.diagnoses_dir / f"{label}-request.json"
            write_record(request_path, request)
            self.request_paths.append(request_path)
            attempt.journal.append(EventKind.DIAGNOSIS_STARTED, {
                "request_id": label, "path": str(request_path),
            })
            report = self.diagnoser.diagnose(request)
            report_path = self.diagnoses_dir / f"{label}-report.json"
            write_record(report_path, report)
            self.reports.append(report)
            self.report_paths.append(report_path)
            attempt.journal.append(EventKind.DIAGNOSIS_REPORTED, {
                "request_id": label, "category": report.category.value,
                "mode": report.recovery_mode.value, "path": str(report_path),
            })
        except Exception as error:
            attempt.journal.append(EventKind.DIAGNOSIS_FAILED, {
                "request_id": label, "type": type(error).__name__, "message": str(error),
            })
            return

        if not self.recovery_enabled:
            attempt.monitor.resume_after_intervention()
            return

        created_attempt: _Attempt | None = None

        def restart_factory() -> DiagnosticAgent:
            nonlocal created_attempt
            created_attempt = self._create_attempt(attempt.number + 1)
            return created_attempt.agent

        try:
            applied = self.controller.apply_recovery(
                report=report, request=request, agent=agent, restart_factory=restart_factory,
            )
        except RecoveryError as error:
            attempt.journal.append(EventKind.RECOVERY_FAILED, {
                "request_id": label, "message": str(error),
            })
            if report.recovery_mode not in {RecoveryMode.RESTORE, RecoveryMode.RESTART}:
                return
            # Preserve the checkout and still give the coding agent the diagnosis.
            fallback = report.model_copy(update={"recovery_mode": RecoveryMode.FEEDBACK})
            try:
                applied = self.controller.apply_recovery(
                    report=fallback, request=request, agent=agent,
                )
            except RecoveryError:
                return
        except Exception as error:
            attempt.journal.append(EventKind.RECOVERY_FAILED, {
                "request_id": label, "type": type(error).__name__, "message": str(error),
            })
            self.fatal_error = error
            agent.stop_for_restart()
            return

        if applied.mode is RecoveryMode.RESTART:
            self.next_attempt = created_attempt
        attempt.journal.append(EventKind.RECOVERY_APPLIED, {
            "request_id": label, "mode": applied.mode.value,
            "saved_patch": str(applied.saved_candidate.patch_path) if applied.saved_candidate else None,
        })
        if applied.mode is not RecoveryMode.NONE:
            self.pending_checks.append(_PendingCheck(
                request, attempt.number, self.diagnoses_dir / f"{label}-recovery-check.json"
            ))

    def _on_submission(self, attempt: _Attempt, patch: str) -> SubmissionDecision:
        self.submission_number += 1
        label = f"submission-{self.submission_number:03d}"
        check = verify_submission(
            self.environment, snapshot=self.snapshot, test_command=self.test_command,
            submitted_patch=patch,
        )
        path = self.run_dir / f"{label}-check.json"
        write_record(path, check)
        self.submission_check_paths.append(path)
        attempt.journal.append(EventKind.SUBMISSION_CHECKED, {
            "status": check.status.value, "path": str(path),
        })
        if check.status is SubmissionStatus.FAIL:
            return SubmissionDecision(accept=False, feedback=f"提交前验证失败：{check.reason}\n{check.test_output or check.diff_check_output}")
        return SubmissionDecision(accept=True)

    def _finalize_recovery_checks(self, attempt: _Attempt) -> None:
        for pending in self.pending_checks:
            check = verify_recovery(
                pending.request, list(attempt.journal.events_since(0)),
                new_attempt=attempt.number != pending.source_attempt,
            )
            write_record(pending.path, check)
            self.recovery_check_paths.append(pending.path)
        self.pending_checks.clear()

    def run(self) -> DiagnosticRunResult:
        attempt = self._create_attempt(1)
        attempts: list[AttemptResult] = []
        while True:
            try:
                info = attempt.agent.run(self.task)
            except Exception as error:
                attempt.journal.append(EventKind.RUN_EXCEPTION, {
                    "type": type(error).__name__, "message": str(error),
                })
                raise
            status = str(info.get("exit_status", ""))
            attempt.journal.append(EventKind.RUN_FINISHED, {"exit_status": status})
            attempts.append(AttemptResult(
                attempt.number, status, attempt.trajectory_path, attempt.events_path,
            ))
            if self.fatal_error is not None:
                raise RuntimeError("recovery failed after the task workspace may have changed") from self.fatal_error
            if status == "DiagnosticRestart":
                if self.next_attempt is None:
                    raise RuntimeError("agent requested restart without a prepared next attempt")
                attempt, self.next_attempt = self.next_attempt, None
                continue
            break

        self._finalize_recovery_checks(attempt)
        final_candidate = self.workspace.export_candidate_patch("final")
        result = DiagnosticRunResult(
            task_id=self.task_id, exit_status=status,
            submission=str(info.get("submission", "")), run_dir=self.run_dir,
            attempts=tuple(attempts), final_patch_path=final_candidate.patch_path,
            diagnosis_request_paths=tuple(self.request_paths),
            diagnosis_report_paths=tuple(self.report_paths),
            recovery_check_paths=tuple(self.recovery_check_paths),
            submission_check_paths=tuple(self.submission_check_paths),
        )
        manifest = {
            "task_id": result.task_id, "exit_status": result.exit_status,
            "submission": result.submission, "base_commit": self.snapshot.base_commit,
            "attempts": [
                {"number": item.number, "exit_status": item.exit_status,
                 "trajectory_path": str(item.trajectory_path), "events_path": str(item.events_path)}
                for item in result.attempts
            ],
            "final_patch_path": str(result.final_patch_path),
            "diagnosis_request_paths": [str(path) for path in result.diagnosis_request_paths],
            "diagnosis_report_paths": [str(path) for path in result.diagnosis_report_paths],
            "recovery_check_paths": [str(path) for path in result.recovery_check_paths],
            "submission_check_paths": [str(path) for path in result.submission_check_paths],
        }
        (self.run_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return result


def run_diagnostic_task(
    *,
    task_id: str,
    task: str,
    model_factory: Callable[[], Any],
    environment: Any,
    diagnosis_model: Any,
    system_template: str,
    instance_template: str,
    output_root: Path,
    test_command: str | None = None,
    isolated_workspace: bool = False,
    diagnosis_enabled: bool = True,
    recovery_enabled: bool = True,
) -> DiagnosticRunResult:
    """Run one task with monitoring, independent diagnosis and optional recovery.

    Each restart obtains a new model and a new event journal. The environment is
    shared because it represents this task's checkout. Only an explicitly isolated
    checkout may be reset. Official benchmark scoring is not performed here.
    """

    return _DiagnosticTaskRunner(
        task_id=task_id, task=task, model_factory=model_factory,
        environment=environment, diagnosis_model=diagnosis_model,
        system_template=system_template, instance_template=instance_template,
        output_root=output_root, test_command=test_command,
        isolated_workspace=isolated_workspace,
        diagnosis_enabled=diagnosis_enabled, recovery_enabled=recovery_enabled,
    ).run()
