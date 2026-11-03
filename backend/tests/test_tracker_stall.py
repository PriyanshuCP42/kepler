"""A simulation BRAIN leaves pending with frozen progress is cancelled, not waited on forever."""

from alpha_harness.engine.tracker import STALL_AFTER_SECONDS, stalled


def test_progress_that_stands_still_long_enough_is_a_stall():
    seen = (0.9, 1000.0)
    assert not stalled(seen, 0.9, 1000.0 + STALL_AFTER_SECONDS - 1)
    assert stalled(seen, 0.9, 1000.0 + STALL_AFTER_SECONDS)


def test_any_movement_or_no_history_is_not_a_stall():
    assert not stalled((0.15, 0.0), 0.2, STALL_AFTER_SECONDS * 3)
    assert not stalled(None, 0.9, STALL_AFTER_SECONDS * 3)


def test_a_batch_that_never_reports_progress_still_stalls():
    assert stalled((None, 0.0), None, STALL_AFTER_SECONDS)
