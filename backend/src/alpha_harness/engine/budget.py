"""The quota gate: every simulation is authorized here before it is sent.

Authorization is a reservation, made atomically against the day's committed total, so two
tasks asking at once cannot both take the last units. It fails closed: an unreadable,
non-positive or over-budget request is refused, never waved through.

Committed units on a platform day are those ``RESERVED``, ``CONFIRMED`` or ``UNCERTAIN``.
An uncertain send (a timeout whose outcome is unknown) keeps counting until a reconcile
settles it; it is not released and it is not blindly resent.

``observed_used`` lets the caller supply what the platform or the tracker says is already
spent (``QuotaSnapshot``, ``Tracker.used_today``); the gate charges the larger of that and its
own ledger, so spend made outside the ledger is never forgotten.

Agents get a :class:`QuotaGate` only through a narrow facade (``authorize`` / ``settle``); the
ceiling, reserve and owner budgets come from settings and cannot be changed through it.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from typing import TYPE_CHECKING

import structlog
from sqlalchemy import func, select

from ..brain.errors import BrainDailyLimitReached, BrainError
from ..brain.filters import PLATFORM_TZ
from ..brain.schemas import SimulationRequest, SimulationType
from ..db.models import QuotaReservation, utcnow
from .lifecycle import RA_CHILDREN

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from ..brain.client import BrainResponse
    from ..db.sqlite import AsyncSession, Database

log = structlog.get_logger(__name__)

COUNTED = ("RESERVED", "CONFIRMED", "UNCERTAIN")


class Denied(RuntimeError):
    """The gate refused a request. ``reason`` is machine-readable."""

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(f"{reason}: {detail}")
        self.reason = reason


class Outcome(StrEnum):
    CONFIRMED = "CONFIRMED"
    RELEASED = "RELEASED"  # nothing reached the platform
    UNCERTAIN = "UNCERTAIN"  # sent, result unknown


@dataclass(frozen=True, slots=True)
class Grant:
    token: str
    units: int
    day: str
    #: ``False`` when this token had already been authorized and nothing new was reserved.
    fresh: bool


def platform_day(now: datetime | None = None) -> str:
    return (now or utcnow()).astimezone(PLATFORM_TZ).date().isoformat()


class QuotaGate:
    def __init__(
        self,
        db: Database,
        *,
        hard_cap: int,
        operating_ceiling: int,
        protected_reserve: int = 0,
        owner_budgets: dict[str, int] | None = None,
        stale_seconds: float = 900.0,
    ) -> None:
        if not 0 < operating_ceiling <= hard_cap:
            raise ValueError("operating_ceiling must be in (0, hard_cap]")
        if not 0 <= protected_reserve < operating_ceiling:
            raise ValueError("protected_reserve must be in [0, operating_ceiling)")
        self.db = db
        self.hard_cap = hard_cap
        self.operating_ceiling = operating_ceiling
        self.protected_reserve = protected_reserve
        self.owner_budgets = dict(owner_budgets or {})
        self.stale_seconds = stale_seconds
        self._lock = asyncio.Lock()

    async def committed(self, day: str | None = None, owner: str | None = None) -> int:
        async with self.db.session() as session:
            stmt = select(
                func.coalesce(
                    func.sum(func.coalesce(QuotaReservation.actual_units, QuotaReservation.units)),
                    0,
                )
            ).where(
                QuotaReservation.day == (day or platform_day()),
                QuotaReservation.state.in_(COUNTED),
            )
            if owner is not None:
                stmt = stmt.where(QuotaReservation.owner == owner)
            return int(await session.scalar(stmt) or 0)

    async def available(self, *, observed_used: int = 0, use_reserve: bool = False) -> int:
        """Units that may still be reserved now."""
        spent = max(await self.committed(), observed_used)
        limit = self.operating_ceiling - (0 if use_reserve else self.protected_reserve)
        return max(0, min(limit, self.hard_cap) - spent)

    async def authorize(
        self,
        *,
        token: str,
        owner: str,
        units: int,
        observed_used: int = 0,
        use_reserve: bool = False,
    ) -> Grant:
        """Reserve ``units`` for ``owner`` or raise :class:`Denied`. Idempotent per ``token``."""
        if not token or not owner:
            raise Denied("invalid", "token and owner are required")
        if units <= 0:
            raise Denied("invalid", f"units must be a positive integer, got {units!r}")
        if observed_used < 0:
            raise Denied("invalid", "observed_used cannot be negative")

        async with self._lock:
            day = platform_day()
            async with self.db.session() as session:
                existing = await session.scalar(
                    select(QuotaReservation).where(QuotaReservation.token == token)
                )
                if existing is not None:
                    if existing.state == "RELEASED":
                        raise Denied("released", f"token {token} was released; use a new one")
                    return Grant(token, existing.units, existing.day, fresh=False)

                spent = max(await self._committed_in(session, day), observed_used)
                limit = self.operating_ceiling - (0 if use_reserve else self.protected_reserve)
                limit = min(limit, self.hard_cap)
                if spent + units > limit:
                    raise Denied(
                        "over_budget",
                        f"{units} requested, {max(0, limit - spent)} available "
                        f"(spent {spent}, limit {limit})",
                    )
                budget = self.owner_budgets.get(owner)
                if budget is not None:
                    used = await self._committed_in(session, day, owner)
                    if used + units > budget:
                        raise Denied(
                            "owner_budget",
                            f"{owner} has {max(0, budget - used)} left of {budget}",
                        )
                session.add(
                    QuotaReservation(
                        token=token, owner=owner, units=units, state="RESERVED", day=day
                    )
                )
        log.info("quota.reserved", token=token, owner=owner, units=units)
        return Grant(token, units, day, fresh=True)

    @staticmethod
    async def _committed_in(session: AsyncSession, day: str, owner: str | None = None) -> int:
        stmt = select(
            func.coalesce(
                func.sum(func.coalesce(QuotaReservation.actual_units, QuotaReservation.units)), 0
            )
        ).where(QuotaReservation.day == day, QuotaReservation.state.in_(COUNTED))
        if owner is not None:
            stmt = stmt.where(QuotaReservation.owner == owner)
        return int(await session.scalar(stmt) or 0)

    async def settle(
        self, token: str, outcome: Outcome, *, actual_units: int | None = None, note: str = ""
    ) -> None:
        """Record what became of a reservation. Only an unsettled or uncertain one can move."""
        if actual_units is not None and actual_units < 0:
            raise Denied("invalid", "actual_units cannot be negative")
        async with self._lock, self.db.session() as session:
            row = await session.scalar(
                select(QuotaReservation).where(QuotaReservation.token == token)
            )
            if row is None:
                raise Denied("unknown", f"no reservation {token}")
            if row.state in ("CONFIRMED", "RELEASED"):
                if row.state == outcome.value:
                    return  # repeated settle is harmless
                raise Denied("settled", f"{token} is already {row.state}")
            row.state = outcome.value
            row.actual_units = actual_units if outcome is Outcome.CONFIRMED else None
            if outcome is Outcome.CONFIRMED and actual_units is None:
                row.actual_units = row.units
            row.note = note or row.note
            row.settled_at = utcnow()
        log.info("quota.settled", token=token, outcome=outcome.value, actual=actual_units)

    def units_for(self, payload: SimulationRequest | list[SimulationRequest]) -> int:
        """Quota a send will cost: one per simulation, a region-agnostic one per region it is
        translated into (the most it can be charged)."""
        requests = payload if isinstance(payload, list) else [payload]
        return sum(RA_CHILDREN if r.type == SimulationType.REGION_AGNOSTIC else 1 for r in requests)

    async def guard(
        self,
        payload: SimulationRequest | list[SimulationRequest],
        send: Callable[[SimulationRequest | list[SimulationRequest]], Awaitable[BrainResponse]],
        *,
        owner: str = "engine",
        observed_used: Callable[[], Awaitable[int]] | None = None,
    ) -> BrainResponse:
        """Send through the gate: reserve, send, then settle by what happened.

        A refusal surfaces as :class:`BrainDailyLimitReached`, which every caller already
        treats as "stop until the quota resets" and requeues the work for. A definite failure
        releases the units; one whose delivery is unknown (``maybe_delivered``, or an
        unexpected error) stays ``UNCERTAIN`` and counted. The send is never retried here.
        """
        units = self.units_for(payload)
        seen = await observed_used() if observed_used is not None else 0
        token = uuid.uuid4().hex
        try:
            await self.authorize(token=token, owner=owner, units=units, observed_used=seen)
        except Denied as exc:
            log.warning("quota.denied", owner=owner, units=units, reason=exc.reason)
            raise BrainDailyLimitReached(f"Local quota gate refused the send: {exc}") from exc
        try:
            response = await send(payload)
        except BrainError as exc:
            if exc.maybe_delivered:
                await self.settle(token, Outcome.UNCERTAIN, note=exc.message[:200])
            else:
                await self.settle(token, Outcome.RELEASED, note=exc.message[:200])
            raise
        except BaseException:
            await self.settle(token, Outcome.UNCERTAIN, note="unexpected error during send")
            raise
        await self.settle(token, Outcome.CONFIRMED, actual_units=units)
        return response

    async def recover(self) -> int:
        """After a restart, a reservation still open past the stale window may have been sent.
        It becomes ``UNCERTAIN`` (still counted); reconcile settles it. Returns how many."""
        cutoff = utcnow() - timedelta(seconds=self.stale_seconds)
        async with self._lock, self.db.session() as session:
            rows = (
                await session.scalars(
                    select(QuotaReservation).where(
                        QuotaReservation.state == "RESERVED", QuotaReservation.created_at < cutoff
                    )
                )
            ).all()
            for row in rows:
                row.state = "UNCERTAIN"
                row.note = "open at restart; outcome unknown"
        if rows:
            log.warning("quota.recovered_uncertain", count=len(rows))
        return len(rows)

    async def status(self, *, observed_used: int = 0) -> dict[str, int | str]:
        day = platform_day()
        by_state: dict[str, int] = {}
        async with self.db.session() as session:
            rows = await session.execute(
                select(
                    QuotaReservation.state,
                    func.coalesce(
                        func.sum(
                            func.coalesce(QuotaReservation.actual_units, QuotaReservation.units)
                        ),
                        0,
                    ),
                )
                .where(QuotaReservation.day == day)
                .group_by(QuotaReservation.state)
            )
            for state, total in rows.all():
                by_state[str(state)] = int(total)
        committed = sum(int(by_state.get(s, 0)) for s in COUNTED)
        return {
            "day": day,
            "hard_cap": self.hard_cap,
            "operating_ceiling": self.operating_ceiling,
            "protected_reserve": self.protected_reserve,
            "reserved": int(by_state.get("RESERVED", 0)),
            "confirmed": int(by_state.get("CONFIRMED", 0)),
            "uncertain": int(by_state.get("UNCERTAIN", 0)),
            "observed_used": observed_used,
            "available": await self.available(observed_used=observed_used),
            "committed": committed,
        }


__all__ = ["COUNTED", "Denied", "Grant", "Outcome", "QuotaGate", "platform_day"]
