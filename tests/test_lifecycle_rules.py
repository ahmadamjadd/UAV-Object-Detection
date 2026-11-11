"""State contracts do not depend on SQLite, API, worker, or training imports."""

import unittest

from pydantic import ValidationError

from foxtrot.config.lifecycle import (
    ACTIVE_STATES, EXECUTION_TRANSITIONS, TERMINAL_STATES, TRACKING_TRANSITIONS,
    ErrorDetails, ExecutionState as E, TrackingState as T,
    validate_execution_transition, validate_tracking_transition,
)


class LifecycleRuleTests(unittest.TestCase):
    def test_execution_table_accepts_exactly_the_declared_edges(self):
        for current in E:
            for target in E:
                with self.subTest(current=current, target=target):
                    if current == target or target in EXECUTION_TRANSITIONS[current]:
                        validate_execution_transition(current, target)
                    else:
                        with self.assertRaises(ValueError):
                            validate_execution_transition(current, target)

    def test_tracking_table_accepts_exactly_the_declared_edges(self):
        for current in T:
            for target in T:
                with self.subTest(current=current, target=target):
                    if current == target or target in TRACKING_TRANSITIONS[current]:
                        validate_tracking_transition(current, target)
                    else:
                        with self.assertRaises(ValueError):
                            validate_tracking_transition(current, target)

    def test_terminal_execution_cannot_restart(self):
        for state in TERMINAL_STATES:
            self.assertFalse(EXECUTION_TRANSITIONS[state])
        self.assertEqual(ACTIVE_STATES, {E.PREPARING, E.TRAINING, E.EVALUATING})

    def test_wrong_enum_and_unvalidated_strings_are_rejected(self):
        for current, target in (("queued", E.PREPARING), (E.TRAINING, "failed"), (E.TRAINING, T.FAILED)):
            with self.assertRaises(TypeError):
                validate_execution_transition(current, target)
        with self.assertRaises(TypeError):
            validate_tracking_transition(T.SYNCING, E.FAILED)

    def test_transition_tables_are_immutable(self):
        with self.assertRaises(TypeError):
            EXECUTION_TRANSITIONS[E.FINISHED] = frozenset({E.QUEUED})
        with self.assertRaises(AttributeError):
            TRACKING_TRANSITIONS[T.SYNCED].add(T.SYNCING)

    def test_error_requires_a_nonblank_reason_and_is_frozen(self):
        reason = ErrorDetails(code="fixture_error", message="Fixture stopped")
        self.assertEqual(ErrorDetails.model_validate_json(reason.model_dump_json()), reason)
        for values in ({"code": "bad code", "message": "x"}, {"code": "failed", "message": " "},
                       {"code": "failed", "message": "x", "secret": "not-accepted"}):
            with self.assertRaises(ValidationError):
                ErrorDetails(**values)
        with self.assertRaises(ValidationError):
            reason.message = "changed"
