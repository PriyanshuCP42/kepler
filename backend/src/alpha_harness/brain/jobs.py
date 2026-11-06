"""BRAIN's slow answers, asked for once however many want them, and kept once they come.

A Production Correlation can take BRAIN ten minutes or more, longer than one request can
wait. Polled inside the request, the wait was thrown away when the request gave up: asking
again polled from the start, and every page or script asking at once polled the same job
side by side, spending the tight hourly window BRAIN meters correlations under.

Here each job is polled by one background task, keyed by what it fetches. A caller waits on
it for at most a request's worth of time; one that gives up leaves the task running, and the
task keeps BRAIN's answer when it comes, so the next read is immediate. A caller arriving
while the job runs joins it rather than starting another.
"""

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

import structlog

from .client import POLL_TIMEOUT
from .errors import BrainPollTimeout

log = structlog.get_logger(__name__)

#: How long a job is polled in the background before it is given up.
JOB_TIMEOUT = 1800.0

#: Fetches the answer, polling for at most the seconds it is given.
Fetch = Callable[[float], Awaitable[dict[str, Any]]]
#: Keeps the answer, once it has come.
Keep = Callable[[dict[str, Any]], Awaitable[None]]


class BrainJobs:
    """One background poll per job, shared by every caller asking for it."""

    def __init__(self, ceiling: float = JOB_TIMEOUT) -> None:
        self.ceiling = ceiling
        self._running: dict[str, asyncio.Task[dict[str, Any]]] = {}

    def running(self, key: str) -> bool:
        task = self._running.get(key)
        return task is not None and not task.done()

    async def run(
        self, key: str, fetch: Fetch, keep: Keep, wait: float = POLL_TIMEOUT
    ) -> dict[str, Any]:
        """BRAIN's answer for ``key``, kept by ``keep``. Waits at most ``wait`` seconds; past
        that it raises :class:`BrainPollTimeout` and the job carries on in the background."""
        task = self._running.get(key)
        if task is None or task.done():
            task = asyncio.create_task(self._drive(key, fetch, keep), name=f"brain-job:{key}")
            task.add_done_callback(self._settled)
            self._running[key] = task
        try:
            # Shielded: a caller giving up must not cancel the poll the next caller joins.
            return await asyncio.wait_for(asyncio.shield(task), wait)
        except TimeoutError:
            minutes = round(self.ceiling / 60)
            raise BrainPollTimeout(
                f"{key} still pending after {wait:.0f}s. Kepler keeps polling it for up to "
                f"{minutes} minutes and keeps the answer: ask again later."
            ) from None

    async def _drive(self, key: str, fetch: Fetch, keep: Keep) -> dict[str, Any]:
        try:
            body = await fetch(self.ceiling)
            await keep(body)
            return body
        finally:
            if self._running.get(key) is asyncio.current_task():
                del self._running[key]

    @staticmethod
    def _settled(task: asyncio.Task[dict[str, Any]]) -> None:
        # Read here, so a job whose callers all gave up does not log "never retrieved".
        if task.cancelled():
            return
        if (exc := task.exception()) is not None:
            log.warning("brain.job.failed", job=task.get_name(), error=str(exc))
