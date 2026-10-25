import unittest
from datetime import datetime, timezone

from diagnostic_swe.monitor import FailureMonitor
from diagnostic_swe.observation import ActionObservation, ChangeState, CodeChange, Outcome
from diagnostic_swe.schemas import EventKind, StepEvent, TriggerReason


def event(sequence, kind, data):
    return StepEvent(
        sequence=sequence, turn=sequence + 1, timestamp=datetime.now(timezone.utc), kind=kind, data=data,
    )


def action(
    sequence, command="pytest -q", outcome=Outcome.COMMAND_FAILURE,
    signature="FAILED tests/test_parser.py::test_empty", output_hash="same-output",
    test_failed=False, change=ChangeState.UNCHANGED,
):
    observation = ActionObservation(
        command=command, outcome=outcome, returncode=0 if outcome is Outcome.SUCCESS else 1,
        exception_info="", output_sha256=output_hash,
        error_signature=signature if outcome is not Outcome.SUCCESS else None,
        test_failed=test_failed, code_change=CodeChange(state=change),
    )
    return event(sequence, EventKind.ACTION_RESULT, {"observation": observation.model_dump(mode="json")})


class FailureMonitorTests(unittest.TestCase):
    def test_explicit_request_only_uses_structured_flag_or_exact_marker(self):
        monitor = FailureMonitor()
        self.assertIsNone(monitor.observe([event(0, EventKind.MODEL_RESPONSE, {
            "response": {"content": "Maybe ask for help later", "extra": {}}})]))
        decision = monitor.observe([event(1, EventKind.MODEL_RESPONSE, {
            "response": {"content": "Reasoning\n[[REQUEST_DIAGNOSIS]]\n", "extra": {}}})])
        self.assertEqual(decision.reason, TriggerReason.AGENT_REQUEST)
        self.assertEqual(decision.evidence_event_sequences, [1])

    def test_repeated_command_failure_requires_no_intervening_edit(self):
        monitor = FailureMonitor()
        self.assertIsNone(monitor.observe([action(0)]))
        decision = monitor.observe([action(1)])
        self.assertEqual(decision.reason, TriggerReason.REPEATED_COMMAND_FAILURE)
        self.assertEqual(decision.evidence_event_sequences, [0, 1])
        self.assertIsNone(monitor.observe([action(2)]))

        monitor.observe([event(3, EventKind.FEEDBACK_INJECTED, {"content": "Try another path"})])
        self.assertIsNone(monitor.observe([action(4)]))
        self.assertEqual(monitor.observe([action(5)]).evidence_event_sequences, [4, 5])

        changed = FailureMonitor()
        self.assertIsNone(changed.observe([action(0), action(1, command="edit", outcome=Outcome.SUCCESS,
                                                    change=ChangeState.CHANGED, output_hash="edit")]))
        self.assertIsNone(changed.observe([action(2)]))

    def test_same_test_failure_after_code_edit_is_distinct_trigger(self):
        monitor = FailureMonitor()
        self.assertIsNone(monitor.observe([action(0, test_failed=True)]))
        self.assertIsNone(monitor.observe([action(1, command="edit parser", outcome=Outcome.SUCCESS,
                                                  change=ChangeState.CHANGED, output_hash="edit")]))
        decision = monitor.observe([action(2, test_failed=True)])
        self.assertEqual(decision.reason, TriggerReason.REPEATED_TEST_FAILURE)
        self.assertEqual(decision.evidence_event_sequences, [0, 1, 2])

    def test_no_progress_needs_three_identical_successes(self):
        monitor = FailureMonitor()
        first = action(0, command="cat parser.py", outcome=Outcome.SUCCESS, output_hash="same")
        second = action(1, command="cat parser.py", outcome=Outcome.SUCCESS, output_hash="same")
        third = action(2, command="cat parser.py", outcome=Outcome.SUCCESS, output_hash="same")
        self.assertIsNone(monitor.observe([first, second]))
        decision = monitor.observe([third])
        self.assertEqual(decision.reason, TriggerReason.NO_PROGRESS)
        self.assertEqual(decision.evidence_event_sequences, [0, 1, 2])

        different = FailureMonitor()
        self.assertIsNone(different.observe([first, second, action(
            2, command="cat parser.py", outcome=Outcome.SUCCESS, output_hash="new-output")]))

    def test_tool_errors_trigger_but_submission_does_not(self):
        monitor = FailureMonitor()
        submitted = action(0, command="submit", outcome=Outcome.SUBMITTED)
        self.assertIsNone(monitor.observe([submitted]))
        self.assertIsNone(monitor.observe([action(1, outcome=Outcome.TOOL_ERROR)]))
        decision = monitor.observe([action(2, outcome=Outcome.TOOL_ERROR)])
        self.assertEqual(decision.reason, TriggerReason.TOOL_ERROR)
        self.assertEqual(decision.evidence_event_sequences, [1, 2])

    def test_batch_with_new_edit_clears_stale_failure_candidate(self):
        monitor = FailureMonitor()
        batch = [
            action(0), action(1),
            action(2, command="edit parser", outcome=Outcome.SUCCESS,
                   change=ChangeState.CHANGED, output_hash="edit"),
        ]
        self.assertIsNone(monitor.observe(batch))

    def test_rejects_replayed_event_sequence(self):
        monitor = FailureMonitor()
        item = action(0)
        monitor.observe([item])
        with self.assertRaisesRegex(ValueError, "increasing sequence"):
            monitor.observe([item])

    def test_parallel_actions_in_one_model_turn_do_not_count_as_retries(self):
        monitor = FailureMonitor()
        first = action(0).model_copy(update={"turn": 1})
        second = action(1).model_copy(update={"turn": 1})
        self.assertIsNone(monitor.observe([first, second]))

        repeated_reads = FailureMonitor()
        reads = [action(index, command="cat parser.py", outcome=Outcome.SUCCESS,
                        output_hash="same").model_copy(update={"turn": 1}) for index in range(3)]
        self.assertIsNone(repeated_reads.observe(reads))


if __name__ == "__main__":
    unittest.main()
