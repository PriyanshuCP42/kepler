"""Which datasets make alphas that everyone else already has, learned from Production Correlation.

BRAIN refuses an alpha whose Production Correlation (against every user's alphas) reaches 0.7,
and Kepler aims lower, :data:`PROD_TARGET`: an alpha that close to others' adds little to
your combined performance. Production Correlation is known only once BRAIN is asked, so the
datasets each measured alpha reads stand in for it before then: a dataset whose alphas came
back high is **crowded**, one whose alphas came back low is **clean**.

Price and volume data, which nearly every expression reads, says nothing about any one alpha,
so it is left out of the reckoning.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from sqlalchemy import func, select

from ..db.models import BrainCache
from ..labs.fastexpr import ParseError, data_fields, parse

if TYPE_CHECKING:
    from collections.abc import Iterable

#: The Production Correlation Kepler holds alphas to, under BRAIN's own 0.7.
PROD_TARGET = 0.6
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
        if not alpha_ids:
            return {}
        async with self.state.db.session() as session:
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
