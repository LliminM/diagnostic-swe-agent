"""Deterministic, evidence-linked rules for requesting a diagnosis."""

import re
from collections import deque
from collections.abc import Iterable

from .observation import ActionObservation, ChangeState, Outcome
from .schemas import EventKind, StepEvent, TriggerDecision, TriggerReason


_HELP_MARKER = re.compile(r"(?m)^\s*\[\[REQUEST_DIAGNOSIS\]\]\s*$")
_PRIORITY = {
    TriggerReason.AGENT_REQUEST: 0,
    TriggerReason.TOOL_ERROR: 1,
    TriggerReason.REPEATED_TEST_FAILURE: 2,
    TriggerReason.POST_EDIT_TEST_FAILURE: 3,
    TriggerReason.REPEATED_COMMAND_FAILURE: 4,
    TriggerReason.NO_PROGRESS: 5,
}


class FailureMonitor:
    """Consume new StepEvents in sequence and emit at most one decision per episode.

    A code edit or explicit feedback starts a fresh episode. The monitor does not
    pause the agent or run a diagnosis; the runner will act on its decision later.
    """

    def __init__(self, history_size: int = 16):
        if history_size < 3:
            raise ValueError("history_size must be at least 3")
        self._actions: deque[tuple[StepEvent, ActionObservation]] = deque(maxlen=history_size)
        self._last_sequence = -1
        self._armed = True
        self._edited_in_episode = False

    def resume_after_intervention(self) -> None:
        """Forget old failures after a diagnosis has been handled."""

        self._actions.clear()
        self._armed = True
        self._edited_in_episode = False

    def observe(self, events: Iterable[StepEvent]) -> TriggerDecision | None:
        candidates: list[TriggerDecision] = []
        for event in events:
            if event.sequence <= self._last_sequence:
                raise ValueError("events must be supplied once, in increasing sequence order")
            self._last_sequence = event.sequence

            if event.kind in {EventKind.FEEDBACK_INJECTED, EventKind.SUBMISSION_REJECTED}:
                self.resume_after_intervention()
                candidates.clear()
                continue
            if event.kind in {EventKind.RUN_FINISHED, EventKind.SUBMISSION_ACCEPTED}:
                candidates.clear()
                continue

            if event.kind is EventKind.MODEL_RESPONSE and self._wants_diagnosis(event):
                candidates.append(
                    TriggerDecision(
                        reason=TriggerReason.AGENT_REQUEST,
                        evidence_event_sequences=[event.sequence],
                        explanation="主 Agent 使用显式标记请求诊断。",
                    )
                )

            if event.kind not in {EventKind.ACTION_RESULT, EventKind.ACTION_EXCEPTION}:
                continue
            raw_observation = event.data.get("observation")
            if raw_observation is None:
                continue
            observation = ActionObservation.model_validate(raw_observation)

            if observation.outcome is Outcome.SUBMITTED:
                candidates.clear()
                continue
            if observation.code_change.state is ChangeState.CHANGED:
                if not self._armed:
                    self.resume_after_intervention()
                candidates = [item for item in candidates if item.reason is TriggerReason.AGENT_REQUEST]
                self._edited_in_episode = bool(observation.code_change.dirty_files_after)
            self._actions.append((event, observation))
            candidates.extend(self._action_candidates())

        if not self._armed or not candidates:
            return None
        decision = min(candidates, key=lambda item: _PRIORITY[item.reason])
        self._armed = False
        return decision

    @staticmethod
    def _wants_diagnosis(event: StepEvent) -> bool:
        response = event.data.get("response", {})
        if not isinstance(response, dict):
            return False
        extra = response.get("extra", {})
        if isinstance(extra, dict) and extra.get("request_diagnosis") is True:
            return True
        content = response.get("content", "")
        return isinstance(content, str) and bool(_HELP_MARKER.search(content))

    def _action_candidates(self) -> list[TriggerDecision]:
        actions = list(self._actions)
        current_event, current = actions[-1]
        decisions = []

        if len(actions) >= 2:
            previous_event, previous = actions[-2]
            if (current.outcome is Outcome.TOOL_ERROR and previous.outcome is Outcome.TOOL_ERROR
                    and current_event.turn > previous_event.turn):
                decisions.append(
                    TriggerDecision(
                        reason=TriggerReason.TOOL_ERROR,
                        evidence_event_sequences=[previous_event.sequence, current_event.sequence],
                        explanation="连续两次工具执行异常。",
                    )
                )

        if current.test_failed and current.error_signature:
            for index in range(len(actions) - 2, -1, -1):
                previous_event, previous = actions[index]
                if (not previous.test_failed or previous.error_signature != current.error_signature
                        or current_event.turn <= previous_event.turn):
                    continue
                edits = [event.sequence for event, observation in actions[index + 1:-1]
                         if observation.code_change.state is ChangeState.CHANGED]
                if edits:
                    decisions.append(
                        TriggerDecision(
                            reason=TriggerReason.REPEATED_TEST_FAILURE,
                            evidence_event_sequences=[previous_event.sequence, *edits, current_event.sequence],
                            explanation="修改代码后，同一个测试失败信号再次出现。",
                        )
                    )
                break

        if current.test_failed and self._edited_in_episode:
            decisions.append(
                TriggerDecision(
                    reason=TriggerReason.POST_EDIT_TEST_FAILURE,
                    evidence_event_sequences=[current_event.sequence],
                    explanation="已有候选代码修改，首次测试失败即需要独立诊断。",
                )
            )

        if current.outcome is Outcome.COMMAND_FAILURE:
            for index in range(len(actions) - 2, -1, -1):
                previous_event, previous = actions[index]
                if previous.outcome is not Outcome.COMMAND_FAILURE:
                    continue
                if current_event.turn <= previous_event.turn:
                    continue
                if previous.command.strip() != current.command.strip():
                    continue
                if (previous.error_signature or previous.output_sha256) != (
                    current.error_signature or current.output_sha256
                ):
                    continue
                if all(observation.code_change.state is ChangeState.UNCHANGED
                       for _, observation in actions[index + 1:]):
                    decisions.append(
                        TriggerDecision(
                            reason=TriggerReason.REPEATED_COMMAND_FAILURE,
                            evidence_event_sequences=[previous_event.sequence, current_event.sequence],
                            explanation="未修改代码却再次执行同一命令，并得到相同失败。",
                        )
                    )
                break

        if len(actions) >= 3:
            tail = actions[-3:]
            same_result = len({(obs.command.strip(), obs.output_sha256) for _, obs in tail}) == 1
            distinct_turns = len({event.turn for event, _ in tail}) == 3
            if same_result and distinct_turns and all(
                obs.outcome is Outcome.SUCCESS and obs.code_change.state is ChangeState.UNCHANGED
                for _, obs in tail
            ):
                decisions.append(
                    TriggerDecision(
                        reason=TriggerReason.NO_PROGRESS,
                        evidence_event_sequences=[event.sequence for event, _ in tail],
                        explanation="连续三次重复相同的成功命令，输出与代码状态均未变化。",
                    )
                )

        return decisions
