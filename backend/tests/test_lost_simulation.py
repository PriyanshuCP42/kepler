"""A simulation BRAIN answers only with server errors is given up on while BRAIN is up."""

import asyncio
from types import SimpleNamespace

from alpha_harness.brain.errors import BrainError
from alpha_harness.engine import tracker


class _Endpoints:
    def __init__(self, *, auth_ok: bool, others_ok: bool) -> None:
        self.auth_ok, self.others_ok, self.auth_calls = auth_ok, others_ok, 0

    async def read_simulation(self, platform_id: str):
        if platform_id == "lost" or not self.others_ok:
            return SimpleNamespace(status=500, body={"detail": "boom"}, retry_after=None)
        return SimpleNamespace(status=200, body={"progress": 0.5}, retry_after=1.0)

    async def get_auth(self):
        self.auth_calls += 1
        if not self.auth_ok:
            raise BrainError("429 on /authentication")
        return object()


def _run(endpoints: _Endpoints, monkeypatch, minutes: int = 40) -> str:
    """Read ``lost`` and one healthy simulation every 16 s; the lost one's last outcome."""
    tr = tracker.SimulationTracker.__new__(tracker.SimulationTracker)
    tr.endpoints, tr._next_poll, tr._failing_since, tr._progress_since = endpoints, {}, {}, {}
    tr._answered_at, tr._probed = None, None
    clock = [1000.0]
    monkeypatch.setattr(tracker.time, "monotonic", lambda: clock[0])

    async def loop() -> str:
        kind = "retry"
        for _ in range(minutes * 60 // 16):
            await tr.read("healthy")
            read = await tr.read("lost")
            if read is not None:
                kind = read[0].kind
                if kind == "gone":
                    return kind
            clock[0] += 16
        return kind

    return asyncio.run(loop())


def test_a_lost_simulation_ends_even_when_the_auth_probe_is_throttled(monkeypatch):
    # Other reads succeed, so BRAIN is up: the throttled probe must not keep it waiting.
    endpoints = _Endpoints(auth_ok=False, others_ok=True)
    assert _run(endpoints, monkeypatch) == "gone"
    assert endpoints.auth_calls == 0


def test_an_outage_keeps_every_simulation_waiting_and_probes_sparingly(monkeypatch):
    endpoints = _Endpoints(auth_ok=False, others_ok=False)
    assert _run(endpoints, monkeypatch, minutes=40) == "retry"
    # Ten minutes past the 30-minute mark, at most one probe a minute.
    assert endpoints.auth_calls <= 11


def test_with_nothing_else_read_the_probe_decides(monkeypatch):
    endpoints = _Endpoints(auth_ok=True, others_ok=False)
    assert _run(endpoints, monkeypatch) == "gone"
