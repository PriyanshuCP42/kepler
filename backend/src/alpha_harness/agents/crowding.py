"""Which datasets make alphas that everyone else already has, learned from Production Correlation.

BRAIN refuses an alpha whose Production Correlation (against every user's alphas) reaches 0.7,
and Kepler holds Ready alphas under :data:`PROD_TARGET`, the same 0.7: the ceiling you set.
Production Correlation is known only once BRAIN is asked, so the
datasets each measured alpha reads stand in for it before then: a dataset whose alphas came
back high is **crowded**, one whose alphas came back low is **clean**.

Price and volume data, which nearly every expression reads, says nothing about any one alpha,
so it is left out of the reckoning.

Measured late, crowding costs a day: a search spends its simulations on one idea and learns
next morning that everyone has it. So the :class:`ProdWatch` asks BRAIN about each alpha the
Critic passes as it lands, a few at a time, and a search whose first :data:`SURVIVORS` measured
passes all reach BRAIN's limit is stopped (:func:`search_crowded`).
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING, Any

import structlog
from sqlalchemy import func, select

from ..db.models import BrainCache, CriticVerdict, utcnow
from ..labs.fastexpr import ParseError, data_fields, parse
from ..vault.yields import is_submitted

if TYPE_CHECKING:
    from collections.abc import Iterable

    from ..db.sqlite import Database

log = structlog.get_logger(__name__)

#: The Production Correlation Kepler holds Ready alphas under: BRAIN's own limit, as you set it.
PROD_TARGET = 0.7
#: Mean Production Correlation of a dataset's measured alphas at which it counts as crowded.
CROWDED = 0.65
#: And at or under which it counts as clean.
CLEAN = 0.55
#: Measured alphas a dataset needs before its mean is trusted.
MIN_MEASURED = 5
#: The category of the price and volume data nearly every expression reads: background.
BACKGROUND_CATEGORY = "pv"
#: Seconds the reckoning is kept.
TTL = 1800.0

#: BRAIN's own Production Correlation limit.
BRAIN_LIMIT = 0.7
#: Measured passes a search needs, all at BRAIN's limit, before it is stopped as crowded.
SURVIVORS = 3
#: Seconds between the watch's rounds.
WATCH_SECONDS = 600.0
#: Alphas the watch sends BRAIN per round: its correlation checks are limited per hour, and a
#: long job would hold the one job slot the Desk's own checks need.
WATCH_BATCH = 10
#: How far back the watch looks for passes the Critic landed.
WATCH_DAYS = 3
#: Where a search's crowding outcome is kept on its study, so it is decided once.
KEY = "crowding"

_PROD_PREFIX = "correlation:prod:"


@dataclass(frozen=True, slots=True)
class DatasetCrowding:
    dataset: str
    measured: int
    mean: float

    @property
    def crowded(self) -> bool:
        return self.measured >= MIN_MEASURED and self.mean >= CROWDED

    @property
    def clean(self) -> bool:
        return self.measured >= MIN_MEASURED and self.mean <= CLEAN


def reckon(
    measured: dict[str, float],
    datasets_of: dict[str, set[str]],
    background: set[str] | frozenset[str] = frozenset(),
) -> dict[str, DatasetCrowding]:
    """Each dataset's measured alphas and their mean Production Correlation, ``background``
    left out. Pure."""
    values: dict[str, list[float]] = {}
    for alpha_id, value in measured.items():
        for dataset in datasets_of.get(alpha_id, ()):
            values.setdefault(dataset, []).append(value)
    return {
        d: DatasetCrowding(d, len(v), sum(v) / len(v))
        for d, v in values.items()
        if d not in background
    }


def expected(datasets: Iterable[str], table: dict[str, DatasetCrowding]) -> float | None:
    """The highest trusted mean among ``datasets``: what an alpha on them can expect, or
    ``None`` when none of them has been measured enough."""
    means = [c.mean for d in datasets if (c := table.get(d)) and c.measured >= MIN_MEASURED]
    return max(means) if means else None


class Crowding:
    """The reckoning over every Production Correlation BRAIN has given, kept for :data:`TTL`."""

    def __init__(self, state: Any) -> None:
        self.state = state
        self._table: dict[str, DatasetCrowding] = {}
        self._fields: dict[str, str] = {}
        self._read = float("-inf")

    async def table(self) -> dict[str, DatasetCrowding]:
        if time.monotonic() - self._read < TTL:
            return self._table
        async with self.state.db.session() as session:
            rows = await session.execute(
                select(BrainCache.key, func.json_extract(BrainCache.body, "$.max")).where(
                    BrainCache.key.like(f"{_PROD_PREFIX}%")
                )
            )
            measured = {
                key.removeprefix(_PROD_PREFIX): float(value)
                for key, value in rows.tuples()
                if isinstance(value, int | float)
            }
        fields = await self.state.catalog.query(
            "SELECT field_id, any_value(dataset_id) AS dataset_id, "
            "any_value(lower(category_id)) AS category FROM data_field "
            "WHERE dataset_id IS NOT NULL GROUP BY field_id"
        )
        self._fields = {str(r["field_id"]): str(r["dataset_id"]) for r in fields}
        background = {
            str(r["dataset_id"]) for r in fields if r.get("category") == BACKGROUND_CATEGORY
        }
        datasets_of: dict[str, set[str]] = {}
        if measured:
            ids = list(measured)
            placeholders = ", ".join("?" for _ in ids)
            for row in await self.state.alphas.catalog.query(
                f"SELECT alpha_id, expression FROM alpha WHERE alpha_id IN ({placeholders})",  # noqa: S608
                ids,
            ):
                datasets_of[str(row["alpha_id"])] = self.datasets(row.get("expression"))
        self._table = reckon(measured, datasets_of, background)
        self._read = time.monotonic()
        return self._table

    def datasets(self, expression: str | None) -> set[str]:
        """The datasets an expression reads, by the fields it names."""
        if not expression:
            return set()
        try:
            names = data_fields(parse(expression))
        except ParseError:
            return set()
        return {d for n in names if (d := self._fields.get(n))}

    async def measured(self, alpha_ids: list[str]) -> dict[str, float]:
        """The Production Correlation BRAIN has given for each of ``alpha_ids`` that has one."""
        return await measured_in(self.state.db, alpha_ids)

    async def too_crowded(self, expression: str | None, known: float | None) -> bool:
        """Whether an alpha's own Production Correlation, or failing that its datasets', is
        over the target: whatever is built on it inherits the crowd."""
        if known is not None:
            return known > PROD_TARGET
        guess = await self.expected_for(expression)
        return guess is not None and guess >= CROWDED

    async def expected_for(self, expression: str | None) -> float | None:
        table = await self.table()  # Reads the field map ``datasets`` needs first.
        return expected(self.datasets(expression), table)


async def measured_in(db: Database, alpha_ids: list[str]) -> dict[str, float]:
    """The Production Correlation BRAIN has given for each of ``alpha_ids`` that has one."""
    if not alpha_ids:
        return {}
    async with db.session() as session:
        rows = await session.execute(
            select(BrainCache.key, func.json_extract(BrainCache.body, "$.max")).where(
                BrainCache.key.in_([f"{_PROD_PREFIX}{a}" for a in alpha_ids])
            )
        )
        return {
            key.removeprefix(_PROD_PREFIX): float(value)
            for key, value in rows.tuples()
            if isinstance(value, int | float)
        }


def search_crowded(measured: list[float]) -> bool | None:
    """Whether a search's measured passes say its idea is crowded: ``True`` once
    :data:`SURVIVORS` are measured and every one reaches :data:`BRAIN_LIMIT`, ``False`` as soon
    as one is under it, ``None`` while it is too early to say. Pure."""
    if any(v < BRAIN_LIMIT for v in measured):
        return False
    return True if len(measured) >= SURVIVORS else None


def crowded_message(measured: list[float]) -> str:
    values = ", ".join(f"{v:.2f}" for v in sorted(measured))
    return (
        f"Stopped as crowded: its first {len(measured)} alphas the Critic passed all reached "
        f"BRAIN's Production Correlation limit of {BRAIN_LIMIT} ({values}). Others already "
        "hold this idea; unsent simulations were dropped."
    )


class ProdWatch:
    """Asks BRAIN for the Production Correlation of each alpha the Critic passes as it lands,
    :data:`WATCH_BATCH` at a time, so a crowded idea shows within the hour. No simulation
    quota; a round waits while another job holds BRAIN's rate limit."""

    def __init__(self, state: Any) -> None:
        self.state = state
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        self._task = asyncio.create_task(self._run(), name="prod-watch")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

    async def _run(self) -> None:
        while True:
            await asyncio.sleep(WATCH_SECONDS)
            # One bad round must not end the watch.
            try:
                await self.round()
            except Exception:
                log.warning("prod_watch.round_failed", exc_info=True)

    async def round(self) -> list[str]:
        """Start one batch; returns the alphas sent, none while signed out or busy."""
        if not self.state.auth.session.authenticated or self.state.backfill.busy:
            return []
        wanted = await self.unmeasured()
        if not wanted:
            return []
        # Imported here: the API layer builds on this module.
        from ..api.lab_tasks import start_prod_correlation

        try:
            await start_prod_correlation(self.state, wanted)
        except RuntimeError:
            return []  # another job took the slot meanwhile
        log.info("prod_watch.sent", alphas=len(wanted))
        return wanted

    async def unmeasured(self) -> list[str]:
        """The best recent passes BRAIN has not measured, at most :data:`WATCH_BATCH`."""
        since = utcnow() - timedelta(days=WATCH_DAYS)
        async with self.state.db.session() as session:
            ids = list(
                await session.scalars(
                    select(CriticVerdict.alpha_id)
                    .where(CriticVerdict.verdict == "pass", CriticVerdict.landed_at >= since)
                    .order_by(CriticVerdict.score.desc())
                    .limit(WATCH_BATCH * 10)
                )
            )
        known = await measured_in(self.state.db, ids)
        stored = await self.state.alphas.by_ids([a for a in ids if a not in known])
        return [a for a in ids if a in stored and not is_submitted(stored[a])][:WATCH_BATCH]
