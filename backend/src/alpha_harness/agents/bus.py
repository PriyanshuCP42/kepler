"""The agent roster and the work they hand each other, as the Research HQ screen draws it.

Every stage of the pipeline is an *agent* with a desk on one of three floors. When a stage
passes work on (candidates to the validator, a batch to BRAIN, a scored alpha to the families
desk) it calls :meth:`AgentBus.handoff`. Handoffs of the same kind between the same two agents
that arrive within :data:`COALESCE_SECONDS` travel as one parcel with a count, so a harvest of
400 trials is one courier carrying "x400", not 400 couriers.

Agents marked ``planned`` are on the roster but not built yet. Work can still be handed to
them (a near miss goes to the repair desk), which shows the backlog waiting for them.

Nothing here decides anything. It reports; the stages it reports on run without it.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

import structlog

if TYPE_CHECKING:
    from ..realtime import Hub

log = structlog.get_logger(__name__)

TOPIC_AGENTS = "agents"
TOPIC_HANDOFF = "agent-handoff"

#: Handoffs between the same two agents, of the same kind, closer together than this travel
#: as one parcel.
COALESCE_SECONDS = 1.5
#: Least time between two roster broadcasts.
SNAPSHOT_SECONDS = 1.0
#: How long an agent counts as working after its last handoff.
WORKING_SECONDS = 8.0
#: Handoffs kept for the screen's feed and for a client that connects late.
HISTORY = 60

Status = Literal["live", "planned"]
State = Literal["idle", "working", "blocked"]


@dataclass(slots=True)
class Agent:
    id: str
    name: str
    role: str
    floor: int
    slot: int
    status: Status
    detail: str = ""
    blocked: str | None = None
    stats: dict[str, int | float | str] = field(default_factory=dict)
    last_event: float | None = None
    last_event_at: str | None = None

    def state(self, now: float) -> State:
        if self.blocked:
            return "blocked"
        if self.last_event is not None and now - self.last_event < WORKING_SECONDS:
            return "working"
        return "idle"

    def to_dict(self, now: float) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "role": self.role,
            "floor": self.floor,
            "slot": self.slot,
            "status": self.status,
            "state": self.state(now),
            "detail": self.blocked or self.detail,
            "stats": dict(self.stats),
            "lastEventAt": self.last_event_at,
        }


#: The building, top floor first. Floor 3 plans, floor 2 produces, floor 1 judges.
ROSTER: tuple[Agent, ...] = (
    Agent(
        "conductor",
        "Conductor",
        "Plans the day, paces the quota, restarts stalled tasks",
        3,
        0,
        "live",
    ),
    Agent(
        "strategist", "Strategist", "Chooses which niches get today's simulations", 3, 1, "planned"
    ),
    Agent(
        "hypothesis", "Hypothesis Writer", "Turns an economic idea into a template", 3, 2, "planned"
    ),
    Agent("writers", "Lab Writers", "The five labs, writing candidate alphas", 2, 0, "live"),
    Agent("prescreen", "Pre-screen", "Ranks candidates by predicted pass rate", 2, 1, "planned"),
    Agent(
        "validator",
        "Validator",
        "Refuses anything BRAIN would reject, before it costs quota",
        2,
        2,
        "live",
    ),
    Agent(
        "brain",
        "BRAIN Gateway",
        "Sends simulations to WorldQuant BRAIN and brings results back",
        2,
        3,
        "live",
    ),
    Agent(
        "critic",
        "Critic",
        "Judges each finished alpha: checks, decay, cost, simplicity",
        1,
        0,
        "live",
    ),
    Agent(
        "families",
        "Family Clerk",
        "Groups alphas by PnL correlation, checks them against your pool",
        1,
        1,
        "live",
    ),
    Agent(
        "repair", "Repair Engineer", "Fixes near misses by their failing metric", 1, 2, "planned"
    ),
    Agent(
        "selector", "Selector", "Keeps the shortlist of distinct, submittable alphas", 1, 3, "live"
    ),
    Agent(
        "learning", "Learning Analyst", "Turns outcomes into tomorrow's priorities", 1, 4, "planned"
    ),
)


@dataclass(slots=True)
class _Pending:
    source: str
    target: str
    kind: str
    count: int
    label: str
    first: float


class AgentBus:
    """The roster's live state, broadcast to the Research HQ screen."""

    def __init__(self, hub: Hub | None = None) -> None:
        self._hub = hub
        self._agents = {a.id: dataclasses.replace(a, stats={}) for a in ROSTER}
        self._history: deque[dict[str, Any]] = deque(maxlen=HISTORY)
        self._pending: dict[tuple[str, str, str], _Pending] = {}
        self._flusher: asyncio.Task[None] | None = None
        self._last_snapshot = 0.0
        self._snapshot_due = False

    # -- reporting -----------------------------------------------------------

    def handoff(self, source: str, target: str, kind: str, count: int = 1, label: str = "") -> None:
        """Record work passing from one agent to the next. Never raises, never waits."""
        if count <= 0 or source not in self._agents or target not in self._agents:
            return
        now = time.monotonic()
        key = (source, target, kind)
        pending = self._pending.get(key)
        if pending is None:
            self._pending[key] = _Pending(source, target, kind, count, label, now)
        else:
            pending.count += count
            pending.label = label or pending.label
        for agent_id in (source, target):
            self._touch(agent_id, now)
        self._wake()

    def count(self, agent_id: str, stat: str, by: int = 1) -> None:
        """Add to one of an agent's counters."""
        agent = self._agents.get(agent_id)
        if agent is None:
            return
        current = agent.stats.get(stat, 0)
        agent.stats[stat] = (current if isinstance(current, int | float) else 0) + by
        self._dirty()

    def set_stat(self, agent_id: str, stat: str, value: int | float | str) -> None:
        agent = self._agents.get(agent_id)
        if agent is not None:
            agent.stats[stat] = value
            self._dirty()

    def say(self, agent_id: str, detail: str) -> None:
        """What the agent is doing right now, in a sentence."""
        agent = self._agents.get(agent_id)
        if agent is not None:
            agent.detail = detail
            self._touch(agent_id, time.monotonic())
            self._wake()

    def block(self, agent_id: str, reason: str | None) -> None:
        """Mark an agent stuck (``reason``) or clear it (``None``)."""
        agent = self._agents.get(agent_id)
        if agent is not None and agent.blocked != reason:
            agent.blocked = reason
            self._dirty()

    # -- reading -------------------------------------------------------------

    def has_detail(self, agent_id: str) -> bool:
        agent = self._agents.get(agent_id)
        return bool(agent and agent.detail)

    def snapshot(self) -> dict[str, Any]:
        now = time.monotonic()
        return {
            "agents": [a.to_dict(now) for a in self._agents.values()],
            "handoffs": list(self._history),
        }

    # -- delivery ------------------------------------------------------------

    def _touch(self, agent_id: str, now: float) -> None:
        agent = self._agents[agent_id]
        agent.last_event = now
        agent.last_event_at = datetime.now(UTC).isoformat()
        self._snapshot_due = True

    def _dirty(self) -> None:
        self._snapshot_due = True
        self._wake()

    def _wake(self) -> None:
        if self._flusher is not None and not self._flusher.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return  # called outside the event loop, e.g. a script: nothing to broadcast to
        self._flusher = loop.create_task(self._flush_loop(), name="agent-bus")

    async def _flush_loop(self) -> None:
        # Runs while there is something to send, then ends; the next report starts it again.
        # It also outlives the last handoff by one WORKING_SECONDS, so the screen hears the
        # agents go back to idle.
        quiet_since: float | None = None
        while True:
            await asyncio.sleep(0.25)
            now = time.monotonic()
            due = [k for k, p in self._pending.items() if now - p.first >= COALESCE_SECONDS]
            for key in due:
                await self._deliver(self._pending.pop(key))
            if self._snapshot_due and now - self._last_snapshot >= SNAPSHOT_SECONDS:
                await self._broadcast(TOPIC_AGENTS, self.snapshot())
                self._snapshot_due = False
                self._last_snapshot = now
            busy = self._pending or self._snapshot_due
            if busy:
                quiet_since = None
                continue
            quiet_since = quiet_since or now
            if now - quiet_since > WORKING_SECONDS + SNAPSHOT_SECONDS:
                await self._broadcast(TOPIC_AGENTS, self.snapshot())
                return

    async def _deliver(self, pending: _Pending) -> None:
        event = {
            "id": uuid.uuid4().hex[:12],
            "from": pending.source,
            "to": pending.target,
            "kind": pending.kind,
            "count": pending.count,
            "label": pending.label,
            "at": datetime.now(UTC).isoformat(),
        }
        self._history.append(event)
        self._snapshot_due = True
        await self._broadcast(TOPIC_HANDOFF, event, replay=False)

    async def _broadcast(self, topic: str, payload: Any, *, replay: bool = True) -> None:
        if self._hub is None:
            return
        try:
            await self._hub.broadcast(topic, payload, replay=replay)
        except Exception:
            log.warning("agents.broadcast_failed", topic=topic, exc_info=True)

    async def stop(self) -> None:
        if self._flusher is not None:
            self._flusher.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._flusher
