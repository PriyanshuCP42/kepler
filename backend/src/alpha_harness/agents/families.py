"""The Family Clerk: which passing alphas are new ideas, and which repeat your pool.

BRAIN's checks pass hundreds of alphas that are variants of a few ideas. Submitting several
of one idea adds count but raises the pool's average correlation, which lowers Combined
Alpha Performance. So every alpha the Critic passes is, in order of score:

1. given its daily PnL (one read from BRAIN, not a simulation) if the vault lacks it;
2. compared with every submitted alpha over the window BRAIN correlates on; at
   :data:`POOL_LIMIT` or more it becomes a near miss for the repair desk;
3. placed in a *family*: the first family leader it moves with at :data:`FAMILY_LIMIT` or
   more, or a family of its own, which makes it a leader and a shortlist candidate.

Only leaders are shortlisted, so the Selector always chooses among distinct ideas.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from typing import TYPE_CHECKING, Any

import structlog
from sqlalchemy import select

from ..db.models import CriticVerdict, utcnow
from ..labs.ga import Series, correlation, series_of
from ..vault.metrics import window_start

if TYPE_CHECKING:
    from collections.abc import Callable
    from datetime import date

    from ..db.sqlite import Database
    from ..vault.backfill import Backfill
    from ..vault.store import AlphaVault
    from .bus import AgentBus

log = structlog.get_logger(__name__)

#: Correlation with a submitted alpha at which a candidate repeats your pool.
POOL_LIMIT = 0.6
#: Correlation at which two candidates are one idea.
FAMILY_LIMIT = 0.6
#: Candidates placed per round. Each may cost three reads from BRAIN for its PnL.
ROUND = 20
#: Longest one alpha's PnL read may take before it is left for a later round.
FETCH_SECONDS = 45.0
#: Pause between rounds when there is work, and when there is none.
BUSY_SECONDS = 20.0
IDLE_SECONDS = 120.0
#: How long the submitted pool's series are reused before being read again.
POOL_TTL = 1800.0


def _windowed(days: dict[date, float]) -> Series | None:
    if not days:
        return None
    start = window_start(max(days))
    return series_of({d: v for d, v in days.items() if d >= start})


class FamilyClerk:
    def __init__(
        self,
        db: Database,
        alphas: AlphaVault,
        backfill: Backfill,
        bus: AgentBus,
        *,
        can_fetch: Callable[[], bool] = lambda: True,
    ) -> None:
        self.db = db
        #: Whether BRAIN will answer a PnL read now. Signed out, the clerk places only alphas
        #: whose PnL is already stored instead of waiting on reads that cannot succeed.
        self.can_fetch = can_fetch
        self.alphas = alphas
        self.backfill = backfill
        self.bus = bus
        self._task: asyncio.Task[None] | None = None
        self._pool: dict[str, Series] = {}
        self._pool_read = float("-inf")
        self._leaders: dict[str, Series] = {}
        self._leaders_loaded = False
        self._wake = asyncio.Event()

    async def start(self) -> None:
        self._task = asyncio.create_task(self._run(), name="family-clerk")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

    def nudge(self) -> None:
        """Start the next round now rather than after the idle pause."""
        self._wake.set()

    async def _run(self) -> None:
        while True:
            placed = 0
            try:
                placed = await self.round()
            # One bad round must not end the clerk's shift.
            except Exception:
                log.warning("families.round_failed", exc_info=True)
                self.bus.block("families", "Last round failed; retrying shortly.")
            self._wake.clear()
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._wake.wait(), BUSY_SECONDS if placed else IDLE_SECONDS)

    async def round(self) -> int:
        """Place up to :data:`ROUND` unplaced passing alphas. Returns how many were placed."""
        async with self.db.session() as session:
            rows = (
                await session.scalars(
                    select(CriticVerdict)
                    .where(
                        CriticVerdict.verdict == "pass", CriticVerdict.family_checked_at.is_(None)
                    )
                    .order_by(CriticVerdict.score.desc())
                    .limit(ROUND * 10)
                )
            ).all()
        self.bus.block("families", None)
        if not rows:
            self.bus.say("families", "Waiting for alphas the Critic passes.")
            return 0
        stored = await self.alphas.daily_pnl([r.alpha_id for r in rows])
        fetching = self.can_fetch()
        chosen = (rows if fetching else [r for r in rows if r.alpha_id in stored])[:ROUND]
        if not chosen:
            self.bus.say(
                "families",
                f"Signed out of BRAIN: {len(rows)}+ passing alphas wait for their daily PnL.",
            )
            return 0
        self.bus.say("families", f"Placing {len(chosen)} passing alphas into families.")
        days, unread = await self._series_for([r.alpha_id for r in chosen], stored)
        pool = await self._pool_series()
        await self._load_leaders()

        # An alpha whose PnL read failed is left for a later round, not placed without it.
        results = [
            self._place(row.alpha_id, days.get(row.alpha_id), pool)
            for row in chosen
            if row.alpha_id not in unread
        ]
        await self._save(results)

        repeats = sum(1 for r in results if r["verdict"] == "near_miss")
        leaders = sum(1 for r in results if r["family_id"] == r["alpha_id"])
        self.bus.count("families", "placed", len(results))
        if repeats:
            self.bus.count("families", "repeat your pool", repeats)
            self.bus.handoff("families", "repair", "pool_repeat", repeats, "too close to your pool")
        if leaders:
            self.bus.count("families", "new families", leaders)
            self.bus.handoff("families", "selector", "new_family", leaders, "new distinct idea")
        self.bus.set_stat("selector", "shortlist", len(self._leaders))
        return len(results)

    def _place(
        self, alpha_id: str, days: dict[date, float] | None, pool: dict[str, Series]
    ) -> dict[str, Any]:
        series = _windowed(days or {})
        outcome: dict[str, Any] = {
            "alpha_id": alpha_id,
            "verdict": "pass",
            "family_id": None,
            "pool": None,
            "note": None,
        }
        if series is None:
            outcome["note"] = "Not enough daily PnL to compare with your pool."
            return outcome
        worst: tuple[str, float] | None = None
        for other_id, other in pool.items():
            rho = correlation(series, other)
            if rho is not None and (worst is None or rho > worst[1]):
                worst = (other_id, rho)
        outcome["pool"] = None if worst is None else round(worst[1], 3)
        if worst is not None and worst[1] >= POOL_LIMIT:
            outcome["verdict"] = "near_miss"
            outcome["note"] = f"Moves with submitted {worst[0]} at {worst[1]:.2f}."
            return outcome
        for leader_id, leader in self._leaders.items():
            rho = correlation(series, leader)
            if rho is not None and abs(rho) >= FAMILY_LIMIT:
                outcome["family_id"] = leader_id
                return outcome
        self._leaders[alpha_id] = series
        outcome["family_id"] = alpha_id
        return outcome

    async def _save(self, results: list[dict[str, Any]]) -> None:
        now = utcnow()
        async with self.db.session() as session:
            for result in results:
                row = await session.get(CriticVerdict, result["alpha_id"])
                if row is None:
                    continue
                row.family_checked_at = now
                row.family_id = result["family_id"]
                row.pool_correlation = result["pool"]
                gates = dict(row.gates or {})
                if result["pool"] is None and result["verdict"] == "pass":
                    gates["pool"] = {"ok": None, "value": None, "note": result["note"]}
                else:
                    ok = result["verdict"] == "pass"
                    note = (
                        result["note"]
                        or f"Highest correlation with your pool is {result['pool']:.2f}."
                    )
                    gates["pool"] = {"ok": ok, "value": result["pool"], "note": note}
                row.gates = gates
                if result["verdict"] == "near_miss":
                    row.verdict = "near_miss"
                    row.reasons = [*(row.reasons or []), result["note"]]

    async def _series_for(
        self, ids: list[str], stored: dict[str, dict[date, float]]
    ) -> tuple[dict[str, dict[date, float]], set[str]]:
        """Each alpha's daily PnL, reading it from BRAIN where the vault lacks it.

        Returns the series and the ids whose read failed or timed out.
        """
        days = {a: stored[a] for a in ids if a in stored}
        unread: set[str] = set()
        for alpha_id in [a for a in ids if a not in days]:
            try:
                await asyncio.wait_for(self.backfill.fetch_returns(alpha_id), FETCH_SECONDS)
                self.bus.count("families", "PnL read from BRAIN")
            except Exception:
                unread.add(alpha_id)
                log.info("families.returns_failed", alpha_id=alpha_id, exc_info=True)
        fetched = [a for a in ids if a not in days and a not in unread]
        if fetched:
            days.update(await self.alphas.daily_pnl(fetched))
        return days, unread

    async def _pool_series(self) -> dict[str, Series]:
        if time.monotonic() - self._pool_read < POOL_TTL:
            return self._pool
        members = await self.alphas.submitted_members()
        ids = [str(m["alpha_id"]) for m in members]
        days = await self.alphas.daily_pnl(ids)
        self._pool = {a: s for a in ids if (s := _windowed(days.get(a, {}))) is not None}
        self._pool_read = time.monotonic()
        self.bus.set_stat("families", "pool alphas compared", len(self._pool))
        return self._pool

    async def _load_leaders(self) -> None:
        if self._leaders_loaded:
            return
        async with self.db.session() as session:
            ids = list(
                (
                    await session.scalars(
                        select(CriticVerdict.alpha_id).where(
                            CriticVerdict.verdict == "pass",
                            CriticVerdict.family_id == CriticVerdict.alpha_id,
                        )
                    )
                ).all()
            )
        days = await self.alphas.daily_pnl(ids)
        self._leaders = {a: s for a in ids if (s := _windowed(days.get(a, {}))) is not None}
        self._leaders_loaded = True
