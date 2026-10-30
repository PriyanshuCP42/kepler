"""The Conductor: keeps the building running.

Its first duty is the watchdog. A task that failed because of the machinery (the database
was busy, a connection dropped, BRAIN answered 5xx) did nothing wrong, and leaving it
Failed until someone notices wastes a day's quota. So every :data:`TICK_SECONDS` the
Conductor looks for tasks that failed that way and queues them again, at most
:data:`MAX_RESTARTS` times each, waiting longer after every try. A task that failed for a
reason of its own (a bad template, nothing left to search) is left alone.

It also reports the day's quota and whether BRAIN is reachable, so a signed-out session
shows as a blocked BRAIN Gateway instead of tasks that quietly stop.
"""

from __future__ import annotations

import asyncio
import contextlib
import re
import time
from datetime import timedelta
from typing import TYPE_CHECKING, Any

import structlog
from sqlalchemy import select

from ..db.models import Study, StudyStatus, utcnow

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from .bus import AgentBus

log = structlog.get_logger(__name__)

TICK_SECONDS = 60.0
MAX_RESTARTS = 3
#: Wait before restart n (1-based) is this many seconds times n.
BACKOFF_SECONDS = 120.0
#: Only failures this recent are restarted; older ones were seen and left by the user.
RECENT = timedelta(hours=24)
#: What a failure caused by the machinery, not the task, says.
INFRASTRUCTURE = re.compile(
    r"database is locked|QueuePool limit|connection timed out|timed out|"
    r"connection (?:reset|refused|aborted)|Server disconnected|BrainTransport|"
    r"ReadTimeout|ConnectTimeout|RemoteProtocolError|\b50[0-4]\b|Service Unavailable",
    re.IGNORECASE,
)


def is_infrastructure(message: str | None) -> bool:
    return bool(message and INFRASTRUCTURE.search(message))


class Conductor:
    def __init__(
        self,
        state: Any,
        bus: AgentBus,
        requeue: Callable[[Any, list[int]], Awaitable[None]],
    ) -> None:
        self.state = state
        self.bus = bus
        self._requeue = requeue
        self._restarts: dict[int, int] = {}
        self._next_try: dict[int, float] = {}
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        self._task = asyncio.create_task(self._run(), name="conductor")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

    async def _run(self) -> None:
        while True:
            try:
                await self.tick()
            except Exception:
                log.warning("conductor.tick_failed", exc_info=True)
            await asyncio.sleep(TICK_SECONDS)

    async def tick(self) -> list[int]:
        """One look around the building. Returns the tasks restarted."""
        await self._report_brain()
        restarted = await self._restart_failed()
        if restarted:
            self.bus.handoff(
                "conductor", "writers", "restart", len(restarted), "restarted after a fault"
            )
            self.bus.count("conductor", "tasks restarted", len(restarted))
            self.bus.say(
                "conductor",
                f"Restarted {len(restarted)} task(s) that failed on an infrastructure fault.",
            )
        return restarted

    async def _restart_failed(self) -> list[int]:
        since = utcnow() - RECENT
        async with self.state.db.session() as session:
            rows = (
                await session.scalars(
                    select(Study).where(
                        Study.status == StudyStatus.FAILED, Study.updated_at >= since
                    )
                )
            ).all()
        now = time.monotonic()
        due: list[int] = []
        for row in rows:
            if not is_infrastructure(row.message):
                continue
            tries = self._restarts.get(row.id, 0)
            if tries >= MAX_RESTARTS:
                self.bus.block("conductor", f"Task {row.id} keeps failing: {row.message[:80]}")
                continue
            if now < self._next_try.get(row.id, 0.0):
                continue
            self._restarts[row.id] = tries + 1
            self._next_try[row.id] = now + BACKOFF_SECONDS * (tries + 1)
            due.append(row.id)
            log.info("conductor.restart", task=row.id, attempt=tries + 1, reason=row.message[:120])
        if due:
            await self._requeue(self.state, due)
        return due

    async def _report_brain(self) -> None:
        session = self.state.auth.session
        if not session.authenticated:
            self.bus.block("brain", "Signed out of BRAIN. Sign in to keep simulating.")
        else:
            left = session.expires_in_seconds
            self.bus.block("brain", None)
            if left is not None:
                self.bus.set_stat("brain", "session minutes left", int(left // 60))
        used = await self.state.tracker.used_today()
        self.bus.set_stat("brain", "simulations today", used)
        ceiling = self.state.settings.simulation_operating_ceiling
        self.bus.set_stat("conductor", "quota used", f"{used:,} of {ceiling:,}")
        if not self.bus.has_detail("conductor"):
            self.bus.say("conductor", "Watching tasks and the day's quota.")
