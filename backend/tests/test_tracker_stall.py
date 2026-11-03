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


async def test_a_batch_stuck_before_a_restart_whose_cancel_hangs_is_let_go():
    from datetime import timedelta
    from types import SimpleNamespace

    from alpha_harness.brain.errors import BrainTransportError
    from alpha_harness.db.models import SimStatus, utcnow
    from alpha_harness.engine.tracker import SimulationTracker

    tracker = SimulationTracker.__new__(SimulationTracker)
    tracker._progress_since = {}
    ended = []

    async def cancel(_record_id):
        raise BrainTransportError("DELETE timed out", maybe_delivered=True)

    async def apply_terminal(record, outcome):
        ended.append((record.id, outcome.status, outcome.message))
        return True

    tracker.cancel = cancel
    tracker._apply_terminal = apply_terminal
    stuck = SimpleNamespace(
        id=7,
        platform_id="p7",
        progress=0.9,
        submitted_at=utcnow().replace(tzinfo=None) - timedelta(seconds=STALL_AFTER_SECONDS + 60),
    )

    assert await tracker._stall(stuck, 0.9)
    assert ended and ended[0][1] == SimStatus.ERROR and "45 minutes" in ended[0][2]

    fresh = SimpleNamespace(id=8, platform_id="p8", progress=0.9, submitted_at=utcnow())
    assert not await tracker._stall(fresh, 0.9)
