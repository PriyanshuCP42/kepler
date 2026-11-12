"""The Daily Autopilot's rules: when it starts tasks, asks for more, and raises alerts."""

import asyncio

from alpha_harness.agents.autopilot import Day, decide
from alpha_harness.alerts import Alerts

HOUR = 3600.0


def _day(**over):
    """Mid-day, signed in, on pace, cores free, two tasks waiting to start."""
    base = {
        "signed_in": True,
        "used": 2500,
        "target": 4500,
        "seconds_left": 12 * HOUR,
        "submitted_today": 1,
        "free_slots": 4,
        "queued_tasks": 0,
        "idle_tasks": 2,
        "planned_today": False,
    }
    return Day(**(base | over))


def test_signed_out_it_only_raises_the_alert():
    d = decide(_day(signed_in=False), enabled=True)
    assert [a[0] for a in d.alerts] == ["signed-out"]
    assert d.start_tasks == 0 and not d.plan_tasks


def test_with_cores_free_it_starts_waiting_tasks_until_the_target_is_spent():
    assert decide(_day(), enabled=True).start_tasks == 2
    assert decide(_day(idle_tasks=5), enabled=True).start_tasks == 2  # a few at a time
    assert decide(_day(used=4500), enabled=True).start_tasks == 0
    # Busy cores, or tasks already waiting for them: nothing more is started.
    assert decide(_day(free_slots=0), enabled=True).start_tasks == 0
    assert decide(_day(queued_tasks=1), enabled=True).start_tasks == 0


def test_behind_pace_with_nothing_waiting_it_asks_the_strategist_once_a_day():
    behind = _day(idle_tasks=0, used=500, seconds_left=8 * HOUR)
    assert decide(behind, enabled=True).plan_tasks
    assert not decide(
        _day(idle_tasks=0, used=500, seconds_left=8 * HOUR, planned_today=True), enabled=True
    ).plan_tasks
    # On pace: no need.
    assert not decide(_day(idle_tasks=0, used=3200, seconds_left=8 * HOUR), enabled=True).plan_tasks


def test_off_it_starts_nothing_but_still_warns():
    late = _day(seconds_left=2 * HOUR, used=600, submitted_today=0)
    d = decide(late, enabled=False)
    assert d.start_tasks == 0 and not d.plan_tasks
    assert sorted(a[0] for a in d.alerts) == ["simulations", "submission"]


def test_alerts_wait_until_late_in_the_day():
    early = _day(seconds_left=10 * HOUR, used=100, submitted_today=0)
    assert decide(early, enabled=True).alerts == []
    # 1,000 simulations and a submission: nothing to warn about however late.
    assert decide(_day(seconds_left=HOUR, used=1200), enabled=True).alerts == []


def test_an_alert_of_one_kind_is_raised_at_most_once_an_hour():
    alerts = Alerts()
    alerts.enabled = False  # no desktop notification from a test

    async def run():
        first = await alerts.raise_("submission", "t", "x")
        again = await alerts.raise_("submission", "t", "x")
        other = await alerts.raise_("simulations", "t", "x")
        return first, again, other

    assert asyncio.run(run()) == (True, False, True)
    assert [k for k, _, _ in alerts.raised] == ["submission", "simulations"]
