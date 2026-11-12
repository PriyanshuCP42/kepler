"""Desktop alerts: a notification when something needs you, at most once an hour per kind.

On macOS a notification goes through ``osascript``; the title and text are passed as
arguments, never spliced into the script, so nothing in them can change what runs.
Elsewhere an alert is only logged.
"""

from __future__ import annotations

import asyncio
import sys
import time

import structlog

log = structlog.get_logger(__name__)

#: The least time between two alerts of the same kind.
REPEAT_SECONDS = 60 * 60
_SCRIPT = (
    "on run argv",
    "display notification (item 2 of argv) with title (item 1 of argv)",
    "end run",
)


class Alerts:
    def __init__(self, *, repeat_seconds: float = REPEAT_SECONDS) -> None:
        self.enabled = True
        self.repeat_seconds = repeat_seconds
        self._last: dict[str, float] = {}
        #: Every alert raised, newest last, for the screen and tests.
        self.raised: list[tuple[str, str, str]] = []

    def due(self, kind: str, now: float | None = None) -> bool:
        """Whether an alert of ``kind`` may be raised again yet."""
        last = self._last.get(kind)
        moment = time.monotonic() if now is None else now
        return last is None or moment - last >= self.repeat_seconds

    async def raise_(self, kind: str, title: str, text: str) -> bool:
        """Raise an alert unless one of its kind was raised within the hour. ``True`` if raised."""
        if not self.due(kind):
            return False
        self._last[kind] = time.monotonic()
        self.raised.append((kind, title, text))
        self.raised[:] = self.raised[-50:]
        log.warning("alert", kind=kind, title=title, text=text)
        if self.enabled and sys.platform == "darwin":
            await _notify(title, text)
        return True


async def _notify(title: str, text: str) -> None:
    args = [arg for line in _SCRIPT for arg in ("-e", line)]
    try:
        process = await asyncio.create_subprocess_exec(
            "osascript",
            *args,
            title,
            text,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await asyncio.wait_for(process.wait(), timeout=10)
    except OSError, TimeoutError:
        log.warning("alert.notify_failed", exc_info=True)


__all__ = ["REPEAT_SECONDS", "Alerts"]
