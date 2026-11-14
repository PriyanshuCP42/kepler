"""The Daily Autopilot: keeps the day's research running without anyone at the keyboard.

Every :data:`TICK_SECONDS` it reads the day and acts:

- **Spend the quota.** While today's simulations are under the target and cores are free,
  it starts tasks that were added but never run, oldest first. With none waiting and the day
  behind pace, it asks the template engine to write some (the Strategist if the engine finds
  nothing), at most once every :data:`PLAN_EVERY_SECONDS`.
- **Rework the near misses.** Every :data:`REWORK_EVERY_SECONDS` it hands the strongest recent
  Alphas BRAIN refused by a little to the Repair Engineer and starts the task (see
  :mod:`.rework`): on this account that, not a broad search, is what reached BRAIN's bar.
- **Check the picks.** Every :data:`CHECK_EVERY_SECONDS` it asks BRAIN's submission check of
  the strongest Today's Picks that have not had one, a few at a time, inside BRAIN's hourly
  limit on correlation checks. A pick it has asked about is not asked again for a day.
- **Guard the day's rules.** A signed-out session, fewer than :data:`DAILY_SIMULATIONS`
  simulations late in the day, and no submission late in the day each raise an alert.

It never submits: the picks it leaves on Today's Picks are yours to submit. Off until
switched on in Settings, since starting tasks spends quota.

:func:`decide` holds every rule and is pure, so the rules are tested without BRAIN.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

import structlog
from sqlalchemy import select

from ..alerts import Alerts
from ..brain.filters import platform_midnight
from ..db.models import Study, StudyStatus, utcnow
from . import challenges

if TYPE_CHECKING:
    from .bus import AgentBus

log = structlog.get_logger(__name__)

TICK_SECONDS = 300.0
CHECK_EVERY_SECONDS = 1200.0
#: Picks sent to BRAIN's check each round: BRAIN limits correlation checks per hour.
CHECKS_PER_ROUND = 5
#: How long a pick asked about is left alone before it may be asked again.
RECHECK_SECONDS = 24 * 3600.0
#: Tasks started in one round, so a burst of starts does not crowd the cores at once.
STARTS_PER_TICK = 2
#: Simulations a BRAIN day needs: the USA/GLB challenge's daily rule.
DAILY_SIMULATIONS = 1000
#: How late in the day the simulation and submission alerts are raised.
SIMULATION_ALERT_SECONDS = 3 * 3600.0
SUBMISSION_ALERT_SECONDS = 4 * 3600.0
#: How far under the day's pace counts as behind.
PACE_SLACK = 200
#: Only tasks added this recently are started; older idle ones were left on purpose.
IDLE_TASK_DAYS = 14
DAY_SECONDS = 24 * 3600.0
#: Regions the template engine plans for, and how many tasks each gets per plan: USA and GLB
#: for the challenge, ASI for the third Osmosis region.
ENGINE_REGIONS = ("USA", "GLB", "ASI", "IND", "EUR", "KOR", "GBR", "HKG", "AMR")
ENGINE_TASKS_PER_REGION = 2
#: Fewest seconds between two plans: a day behind pace gets several, never a flood.
PLAN_EVERY_SECONDS = 3 * 3600.0
#: How often near misses are handed to the Repair Engineer, and at most how many reworks a
#: BRAIN day starts.
REWORK_EVERY_SECONDS = 1800.0
REWORKS_PER_DAY = 12
REWORK_CORES = 2


@dataclass(slots=True)
class Day:
    """What the Autopilot reads before acting."""

    signed_in: bool
    used: int
    target: int
    seconds_left: float
    submitted_today: int
    free_slots: int
    queued_tasks: int
    idle_tasks: int
    #: A plan was written within :data:`PLAN_EVERY_SECONDS`.
    planned_recently: bool


@dataclass(slots=True)
class Decision:
    start_tasks: int = 0
    plan_tasks: bool = False
    alerts: list[tuple[str, str, str]] = field(default_factory=list)
    note: str = ""


def decide(day: Day, *, enabled: bool) -> Decision:
    """What to do this round. Alerts are raised whether or not the Autopilot is on."""
    d = Decision()
    if not day.signed_in:
        d.alerts.append(
            (
                "signed-out",
                "Kepler is signed out of BRAIN",
                "Nothing simulates until you sign in again.",
            )
        )
        d.note = "Signed out: waiting for you to sign in."
        return d

    elapsed = max(0.0, min(1.0, 1 - day.seconds_left / DAY_SECONDS))
    expected = day.target * elapsed
    behind = day.used < expected - PACE_SLACK
    if day.seconds_left < SIMULATION_ALERT_SECONDS and day.used < DAILY_SIMULATIONS:
        hours = day.seconds_left / 3600
        d.alerts.append(
            (
                "simulations",
                f"Only {day.used:,} simulations today",
                f"The challenge needs {DAILY_SIMULATIONS:,} a day; {hours:.1f} hours left.",
            )
        )
    if day.seconds_left < SUBMISSION_ALERT_SECONDS and day.submitted_today == 0:
        hours = day.seconds_left / 3600
        d.alerts.append(
            (
                "submission",
                "Nothing submitted today",
                f"{hours:.1f} hours left in the BRAIN day. Today's Picks lists what is ready.",
            )
        )

    if not enabled:
        d.note = "Off: alerts only."
        return d
    if day.used >= day.target:
        d.note = f"Today's target of {day.target:,} simulations is spent."
        return d
    if day.queued_tasks or day.free_slots <= 0:
        d.note = "Cores are busy; tasks are waiting their turn."
        return d
    if day.idle_tasks:
        d.start_tasks = min(STARTS_PER_TICK, day.idle_tasks)
        d.note = f"Starting {d.start_tasks} waiting task(s) to keep the quota moving."
        return d
    if behind and not day.planned_recently:
        d.plan_tasks = True
        d.note = "Behind pace with nothing waiting: asking the Strategist for tasks."
        return d
    d.note = "Nothing waiting to start." if behind else "On pace."
    return d


class DailyAutopilot:
    def __init__(self, state: Any, bus: AgentBus) -> None:
        self.state = state
        self.bus = bus
        self.enabled = False
        self.daily_target = 4500
        self.alerts = Alerts()
        self._task: asyncio.Task[None] | None = None
        self._checks: set[asyncio.Task[Any]] = set()
        self._asked: dict[str, float] = {}
        self._last_check = 0.0
        self._planned_at = -PLAN_EVERY_SECONDS
        self._last_rework = 0.0
        self._reworks: tuple[str, int] = ("", 0)
        #: The last round, for the screen.
        self.last: dict[str, Any] = {}

    async def start(self) -> None:
        self._task = asyncio.create_task(self._run(), name="daily-autopilot")

    async def stop(self) -> None:
        for task in [self._task, *self._checks]:
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

    async def _run(self) -> None:
        while True:
            try:
                await self.tick()
            except Exception:
                log.warning("autopilot.tick_failed", exc_info=True)
            await asyncio.sleep(TICK_SECONDS)

    async def read_day(self) -> Day:
        state = self.state
        today = platform_midnight()
        seconds_left = (today + timedelta(days=1) - datetime.now(today.tzinfo)).total_seconds()
        signed_in = bool(state.auth.session.authenticated)
        used = await state.tracker.used_today() if signed_in else 0
        engine = await state.engine.status() if signed_in else {}
        since = utcnow() - timedelta(days=IDLE_TASK_DAYS)
        async with state.db.session() as session:
            rows = (
                await session.scalars(
                    select(Study).where(
                        Study.status.in_((StudyStatus.IDLE, StudyStatus.QUEUED)),
                        Study.created_at >= since,
                    )
                )
            ).all()
        members = await state.alphas.submitted_members()
        day = today.date()
        submitted = sum(
            1 for m in members if challenges.platform_day(m.get("date_submitted")) == day
        )
        return Day(
            signed_in=signed_in,
            used=int(used or 0),
            target=self.daily_target,
            seconds_left=seconds_left,
            submitted_today=submitted,
            free_slots=int(engine.get("slotsFree") or 0),
            queued_tasks=sum(1 for r in rows if r.status == StudyStatus.QUEUED),
            idle_tasks=sum(1 for r in rows if r.status == StudyStatus.IDLE),
            planned_recently=time.monotonic() - self._planned_at < PLAN_EVERY_SECONDS,
        )

    async def tick(self) -> Decision:
        """One round: read the day, act on it, report."""
        day = await self.read_day()
        decision = decide(day, enabled=self.enabled)
        for kind, title, text in decision.alerts:
            await self.alerts.raise_(kind, title, text)
        started: list[int] = []
        if decision.start_tasks:
            started = await self._start(decision.start_tasks)
        if decision.plan_tasks:
            await self._plan()
        checked: list[str] = []
        reworked: list[str] = []
        if self.enabled and day.signed_in:
            checked = await self._check_picks()
            reworked = await self._rework()
        self.last = {
            "at": utcnow().isoformat(),
            "enabled": self.enabled,
            "used": day.used,
            "target": day.target,
            "secondsLeft": int(day.seconds_left),
            "submittedToday": day.submitted_today,
            "note": decision.note,
            "started": started,
            "checked": checked,
            "reworked": reworked,
            "alerts": [title for _, title, _ in decision.alerts],
        }
        self.bus.say("conductor", f"Autopilot: {decision.note}")
        return decision

    async def _start(self, count: int) -> list[int]:
        from ..api.lab_tasks import requeue

        since = utcnow() - timedelta(days=IDLE_TASK_DAYS)
        async with self.state.db.session() as session:
            ids = list(
                await session.scalars(
                    select(Study.id)
                    .where(Study.status == StudyStatus.IDLE, Study.created_at >= since)
                    .order_by(Study.created_at)
                    .limit(count)
                )
            )
        if ids:
            await requeue(self.state, ids)
            self.bus.count("conductor", "tasks started by the Autopilot", len(ids))
            log.info("autopilot.started", tasks=ids)
        return ids

    async def _plan(self) -> None:
        """New tasks from the template engine, for each region it knows; the Strategist's
        searches only if the engine finds nothing to try."""
        from ..labs import engine

        self._planned_at = time.monotonic()
        added = 0
        for region in ENGINE_REGIONS:
            try:
                added += len(await engine.plan(self.state, region, ENGINE_TASKS_PER_REGION))
            except Exception:
                log.warning("autopilot.engine_plan_failed", region=region, exc_info=True)
        if added:
            self.bus.count("strategist", "engine tasks written", added)
            log.info("autopilot.planned", engine=added)
            return
        try:
            planned = await self.state.strategist.add_tasks(count=3, simulations=500, cores=2)
        except Exception:
            log.warning("autopilot.plan_failed", exc_info=True)
            return
        log.info("autopilot.planned", strategist=len(planned.get("added", [])))

    async def _check_picks(self) -> list[str]:
        now = time.monotonic()
        if now - self._last_check < CHECK_EVERY_SECONDS:
            return []
        self._last_check = now
        from ..api.alphas import check_alpha
        from ..api.autopilot import todays_picks

        picks = await todays_picks(self.state)
        wanted = [
            p.alpha_id
            for p in picks.check
            if "BRAIN_CHECK_NOT_RUN" in p.reasons
            and now - self._asked.get(p.alpha_id, -RECHECK_SECONDS) >= RECHECK_SECONDS
        ][:CHECKS_PER_ROUND]
        for alpha_id in wanted:
            self._asked[alpha_id] = now
            task = asyncio.create_task(
                _quietly(check_alpha(alpha_id, self.state)), name=f"autopilot-check-{alpha_id}"
            )
            self._checks.add(task)
            task.add_done_callback(self._checks.discard)
        if wanted:
            self.bus.count("selector", "picks sent to BRAIN's check", len(wanted))
        ready = len(picks.ready)
        self.bus.set_stat("selector", "ready to submit", ready)
        self.bus.say("selector", f"Today's Picks: {ready} ready, {len(picks.check)} to check.")
        return wanted

    async def _rework(self) -> list[str]:
        """Hand the strongest recent near misses to the Repair Engineer and start the task."""
        now = time.monotonic()
        if now - self._last_rework < REWORK_EVERY_SECONDS:
            return []
        self._last_rework = now
        day = platform_midnight().date().isoformat()
        if self._reworks[0] != day:
            self._reworks = (day, 0)
        if self._reworks[1] >= REWORKS_PER_DAY:
            return []
        from ..api.lab_tasks import requeue
        from . import rework

        found = await rework.candidates(self.state)
        if not found:
            return []
        ids = [c.alpha_id for c in found]
        added = await self.state.repair.add_task(len(ids), rework.PER_ALPHA, REWORK_CORES, ids)
        if added is None:
            return []
        await requeue(self.state, [added.id])
        self._reworks = (day, self._reworks[1] + 1)
        self.bus.count("repair", "near misses reworked by the Autopilot", len(ids))
        log.info("autopilot.reworked", task=added.id, alphas=ids)
        return ids


async def _quietly(awaitable: Any) -> None:
    """A background check: its failure is logged, never raised into the loop."""
    try:
        await awaitable
    except Exception:
        log.warning("autopilot.check_failed", exc_info=True)


__all__ = ["DailyAutopilot", "Day", "Decision", "decide"]
