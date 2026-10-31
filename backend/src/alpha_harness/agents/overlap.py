"""How much a new expression repeats ideas you already hold, before its PnL exists.

The Family Clerk measures real correlation, but only once BRAIN has the daily PnL, long after
the lab has scored the simulation. So the lab scores it against a stand-in: how many of its
data fields it shares with an alpha in the same region that is

- submitted, so already in your pool;
- the leader of a shortlist family, so the idea is already covered; or
- found by the Family Clerk to move with your pool, so its fields are known to crowd it.

The last set is how the real correlations correct the stand-in: every pool repeat the clerk
finds makes its fields count against the next candidate that reads them.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

import structlog
from sqlalchemy import or_, select

from ..db.models import CriticVerdict
from ..labs.fastexpr import ParseError, data_fields, parse

if TYPE_CHECKING:
    from ..db.sqlite import Database
    from ..vault.store import AlphaVault

log = structlog.get_logger(__name__)

#: Seconds between reloads of the pool.
TTL = 1800.0
#: Correlation with the pool at which the clerk turns a pass into a near miss.
POOL_LIMIT = 0.6


def fields_of(expression: str | None) -> frozenset[str]:
    if not expression:
        return frozenset()
    try:
        return frozenset(data_fields(parse(expression)))
    except ParseError:
        return frozenset()


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class Overlap:
    def __init__(self, db: Database, alphas: AlphaVault) -> None:
        self.db = db
        self.alphas = alphas
        #: Field sets held, by region.
        self.pool: dict[str, list[frozenset[str]]] = {}
        self._read = 0.0
        self._lock = asyncio.Lock()

    async def refresh(self) -> None:
        """Reload the pool when it is older than :data:`TTL`. Cheap otherwise."""
        if time.monotonic() - self._read < TTL:
            return
        async with self._lock:
            if time.monotonic() - self._read < TTL:
                return
            try:
                self.pool = await self._load()
            except Exception:
                # A failed read keeps the last pool: scoring carries on without the penalty
                # going stale for more than one TTL.
                log.warning("overlap.load_failed", exc_info=True)
            self._read = time.monotonic()

    def worst(self, expression: str | None, region: str | None) -> float:
        """The largest share of fields it has in common with one held alpha in its region."""
        own = fields_of(expression)
        return max((jaccard(own, held) for held in self.pool.get(str(region), [])), default=0.0)

    async def _load(self) -> dict[str, list[frozenset[str]]]:
        held = {str(m["alpha_id"]) for m in await self.alphas.submitted_members()}
        async with self.db.session() as session:
            held.update(
                (
                    await session.scalars(
                        select(CriticVerdict.alpha_id).where(
                            or_(
                                (CriticVerdict.verdict == "pass")
                                & (CriticVerdict.family_id == CriticVerdict.alpha_id),
                                CriticVerdict.pool_correlation >= POOL_LIMIT,
                            )
                        )
                    )
                ).all()
            )
        ids = sorted(held)
        pool: dict[str, set[frozenset[str]]] = {}
        for start in range(0, len(ids), 500):
            chunk = ids[start : start + 500]
            placeholders = ", ".join("?" for _ in chunk)
            rows = await self.alphas.catalog.query(
                f"SELECT region, expression FROM alpha WHERE alpha_id IN ({placeholders})",  # noqa: S608
                chunk,
            )
            for row in rows:
                if fields := fields_of(row["expression"]):
                    pool.setdefault(str(row["region"]), set()).add(fields)
        log.info("overlap.loaded", alphas=len(ids), regions=len(pool))
        return {region: list(sets) for region, sets in pool.items()}
